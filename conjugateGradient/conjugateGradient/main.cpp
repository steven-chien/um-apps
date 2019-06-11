/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample implements a conjugate gradient solver on GPU
 * using CUBLAS and CUSPARSE
 *
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Using updated (v2) interfaces to cublas */
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper function CUDA error checking and initialization

double gpu_start;
double gpu_stop;
double cpu_start;
double cpu_stop;
double application_start;
double application_stop;
double compute_migrate_start;
double compute_migrate_stop;
double malloc_start;
double malloc_stop;
double free_start;
double free_stop;
double cuda_malloc_start;
double cuda_malloc_stop;
double cuda_free_start;
double cuda_free_stop;
double init_data_start;
double init_data_stop;
double h2d_memcpy_start;
double h2d_memcpy_stop;
double d2h_memcpy_start;
double d2h_memcpy_stop;
double h2d_prefetch_start;
double h2d_prefetch_stop;
double d2h_prefetch_start;
double d2h_prefetch_stop;
double advise_start;
double advise_stop;
double cublas_init_start;
double cublas_init_stop;
double cublas_destroy_start;
double cublas_destroy_stop;

const char *sSDKname     = "conjugateGradient";

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *I, int *J, float *val, int N, int nz)
{
    I[0] = 0, J[0] = 0, J[1] = 1;
    val[0] = (float)rand()/RAND_MAX + 10.0f;
    val[1] = (float)rand()/RAND_MAX;
    int start;

    for (int i = 1; i < N; i++)
    {
        if (i > 1)
        {
            I[i] = I[i-1]+3;
        }
        else
        {
            I[1] = 2;
        }

        start = (i-1)*3 + 2;
        J[start] = i - 1;
        J[start+1] = i;

        if (i < N-1)
        {
            J[start+2] = i + 1;
        }

        val[start] = val[start-1];
        val[start+1] = (float)rand()/RAND_MAX + 10.0f;

        if (i < N-1)
        {
            val[start+2] = (float)rand()/RAND_MAX;
        }
    }

    I[N] = nz;
}

double mysecond(){
    struct timeval tp;
    struct timezone tzp;
    int i;

    i = gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

int main(int argc, char **argv)
{
    /////////////////////////////// START TIMER ////////////////////////////////////
    application_start = mysecond();

    int M = 0, N = 0, nz = 0, *I = NULL, *J = NULL;
    float *val = NULL;
    const float tol = 1e-5f;
    //const int max_iter = 1000;
    int max_iter = 50;
    float *x;
    float *rhs;
    float a, b, na, r0, r1;
    int *d_col, *d_row;
    float *d_val, *d_x, dot;
    float *d_r, *d_p, *d_Ax;
    int k;
    float alpha, beta, alpham1;

    N = 36000000;
    nz = (N-2)*3 + 4;

    if (checkCmdLineFlag(argc, (const char **)argv, "size")) {
        N = getCmdLineArgumentInt(argc, (const char **)argv, "size");
        nz = (N-2)*3 + 4;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "iterations")) {
        max_iter = getCmdLineArgumentInt(argc, (const char **)argv, "iterations");
    }

    // This will pick the best possible CUDA capable device
    cudaDeviceProp deviceProp;
    int devID = findCudaDevice(argc, (const char **)argv);

    if (devID < 0)
    {
        printf("exiting...\n");
        exit(EXIT_SUCCESS);
    }

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    // Statistics about the GPU device
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
           deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    /* Generate a random tridiagonal symmetric matrix in CSR format */
    //M = N = 1048576;
    malloc_start = mysecond();
    I = (int *)malloc(sizeof(int)*(N+1));
    J = (int *)malloc(sizeof(int)*nz);
    val = (float *)malloc(sizeof(float)*nz);
    x = (float *)malloc(sizeof(float)*N);
    rhs = (float *)malloc(sizeof(float)*N);
    malloc_stop = mysecond();

    init_data_start = malloc_stop;
    genTridiag(I, J, val, N, nz);

    for (int i = 0; i < N; i++)
    {
        rhs[i] = 1.0;
        x[i] = 0.0;
    }
    init_data_stop = mysecond();

    cublas_init_start = init_data_stop;
    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);

    checkCudaErrors(cublasStatus);

    /* Get handle to the CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);

    checkCudaErrors(cusparseStatus);

    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);

    checkCudaErrors(cusparseStatus);

    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
    cublas_init_stop = mysecond();

printf("%f MiB\n", (sizeof(int)*(N+1)+sizeof(int)*nz+sizeof(float)*nz+sizeof(float)*N+sizeof(float)*N+N*sizeof(float)+N*sizeof(float)+N*sizeof(float))/1048576.0);
    cuda_malloc_start = cublas_init_stop;
    checkCudaErrors(cudaMalloc((void **)&d_col, nz*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_row, (N+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_val, nz*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_x, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_r, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_p, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Ax, N*sizeof(float)));
    cuda_malloc_stop = mysecond();

    h2d_memcpy_start = cuda_malloc_stop;
    cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val, nz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, rhs, N*sizeof(float), cudaMemcpyHostToDevice);
    h2d_memcpy_stop = mysecond();

    compute_migrate_start = h2d_memcpy_stop;

    alpha = 1.0;
    alpham1 = -1.0;
    beta = 0.0;
    r0 = 0.;

    gpu_start = mysecond();

    cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);

    cublasSaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);
    cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);

    k = 1;

    //while (r1 > tol*tol && k <= max_iter)
    while (k <= max_iter)
    {
        if (k > 1)
        {
            b = r1 / r0;
            cublasStatus = cublasSscal(cublasHandle, N, &b, d_p, 1);
            cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
        }
        else
        {
            cublasStatus = cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);
        }

        cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ax);
        cublasStatus = cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);
        a = r1 / dot;

        cublasStatus = cublasSaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);
        na = -a;
        cublasStatus = cublasSaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);

        r0 = r1;
        cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
        cudaDeviceSynchronize();
        printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }

    gpu_stop = mysecond();

    d2h_memcpy_start = gpu_stop;
    cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
    d2h_memcpy_stop = mysecond();

    cpu_start = d2h_memcpy_stop;
    printf("Final residual: %e\n",sqrt(r1));

    fprintf(stdout,"&&&& conjugateGradientUM %s\n", (sqrt(r1) < tol) ? "PASSED" : "FAILED");

    float rsum, diff, err = 0.0;

    for (int i = 0; i < N; i++)
    {
        rsum = 0.0;

        for (int j = I[i]; j < I[i+1]; j++)
        {
            rsum += val[j]*x[J[j]];
        }

        diff = fabs(rsum - rhs[i]);

        if (diff > err)
        {
            err = diff;
        }
    }
    cpu_stop = mysecond();

    compute_migrate_stop = cpu_stop;

    cublas_destroy_start = cpu_stop;
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
    cublas_destroy_stop = mysecond();

    free_start = cublas_destroy_stop;
    free(I);
    free(J);
    free(val);
    free(x);
    free(rhs);
    free_stop = mysecond();

    cuda_free_start = free_stop;
    cudaFree(d_col);
    cudaFree(d_row);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ax);
    cuda_free_stop = mysecond();

    application_stop = cuda_free_stop;

    printf("\nGPU Time: %f\n", gpu_stop - gpu_start);
    printf("CPU Time: %f\n", cpu_stop - cpu_start);
    printf("malloc timer: %f\n", malloc_stop - malloc_start);
    printf("free timer: %f\n", free_stop - free_start);
    printf("cuda malloc timer: %f\n", cuda_malloc_stop - cuda_malloc_start);
    printf("cuda free timer: %f\n", cuda_free_stop - cuda_free_start);
    printf("Init data timer: %f\n", init_data_stop - init_data_start);
    printf("\nH2D timer: %f\n", h2d_memcpy_stop - h2d_memcpy_start);
    printf("D2H timer: %f\n", d2h_memcpy_stop - d2h_memcpy_start);
    printf("\ncublas init timer: %f\n", cublas_init_stop - cublas_init_start);
    printf("cublas destroy timer: %f\n", cublas_destroy_stop - cublas_destroy_start);
    printf("misc timer: %f\n", malloc_start - application_start);
    printf("\ncompute migrate timer: %f\n", compute_migrate_stop - compute_migrate_start);
    printf("application timer: %f\n\n", application_stop - application_start);

    printf("Test Summary:  Error amount = %f\n", err);
    //exit((k <= max_iter) ? 0 : 1);
    return 0;
}
