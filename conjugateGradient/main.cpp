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

/* Using updated (v2) interfaces to cublas and cusparse */
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper function CUDA error checking and initialization

static double gpu_start;
static double gpu_stop;
static double cpu_start;
static double cpu_stop;
static double application_start;
static double application_stop;
static double compute_migrate_start;
static double compute_migrate_stop;
static double malloc_start;
static double malloc_stop;
static double free_start;
static double free_stop;
static double cuda_malloc_start;
static double cuda_malloc_stop;
static double cuda_free_start;
static double cuda_free_stop;
static double init_data_start;
static double init_data_stop;
#ifndef CUDA_UM
static double h2d_memcpy_start;
static double h2d_memcpy_stop;
#endif
static double d2h_memcpy_start;
static double d2h_memcpy_stop;
#ifdef CUDA_UM_PREFETCH
static double h2d_prefetch_start;
static double h2d_prefetch_stop;
static double d2h_prefetch_start;
static double d2h_prefetch_stop;
#endif
#ifdef CUDA_UM_ADVISE
static double advise_start;
static double advise_stop;
static double advise_read_start;
static double advise_read_stop;
#endif
static double cublas_init_start;
static double cublas_init_stop;
static double cublas_destroy_start;
static double cublas_destroy_stop;

const char *sSDKname     = "conjugateGradientUM";

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
#ifdef CUDA_UM
    printf("Using CUDA Unified Memory: yes\n");
#else
    printf("Using CUDA Unified Memory: no\n");
#endif

#ifdef CUDA_UM_ADVISE
    printf("Using CUDA Unified Memory Advise: yes\n");
#else
    printf("Using CUDA Unified Memory Advise: no\n");
#endif

#ifdef CUDA_UM_PREFETCH
    printf("Using CUDA Unified Memory Prefetch: yes\n");
#else
    printf("Using CUDA Unified Memory Prefetch: no\n");
#endif


    /////////////////////////////// START TIMER ////////////////////////////////////
    application_start = mysecond();

    int N = 0, nz = 0, *I = NULL, *J = NULL;
    int *h_I = NULL, *h_J = NULL;
    float *h_x = NULL, *h_r = NULL, *h_val = NULL, *h_rhs = NULL;
    float *val = NULL;
    const float tol = 1e-5f;
    int max_iter = 50;
    float dot;
    float *x;
    float *rhs;
    float a, b, na, r0, r1;
    float *r, *p, *Ax;
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

    printf("Starting [%s]...\n", sSDKname);

    // This will pick the best possible CUDA capable device
    cudaDeviceProp deviceProp;
    int devID = findCudaDevice(argc, (const char **)argv);
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    if (!deviceProp.managedMemory) { 
        // This samples requires being run on a device that supports Unified Memory
        fprintf(stderr, "Unified Memory not supported on this device\n");
        exit(EXIT_WAIVED);
    }

    // Statistics about the GPU device
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
           deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    double start_time = mysecond();
    /* Generate a random tridiagonal symmetric matrix in CSR format */
    //N = 1048576;

#ifndef CUDA_UM
    malloc_start = mysecond();
    h_I = (int *)malloc(sizeof(int)*(N+1));
    h_J = (int *)malloc(sizeof(int)*nz);
    h_val = (float *)malloc(sizeof(float)*nz);
    h_x = (float *)malloc(sizeof(float)*N);
    h_rhs = (float *)malloc(sizeof(float)*N);
    malloc_stop = mysecond();

    cuda_malloc_start = cublas_init_stop;
    checkCudaErrors(cudaMalloc((void **)&I, (N+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&J, nz*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&val, nz*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&x, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&r, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&p, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&Ax, N*sizeof(float)));
    cuda_malloc_stop = mysecond();
#else
    cuda_malloc_start = mysecond();
    checkCudaErrors(cudaMallocManaged((void **)&I, sizeof(int)*(N+1)));
    checkCudaErrors(cudaMallocManaged((void **)&J, sizeof(int)*nz));
    checkCudaErrors(cudaMallocManaged((void **)&val, sizeof(float)*nz));
    checkCudaErrors(cudaMallocManaged((void **)&x, sizeof(float)*N));
    checkCudaErrors(cudaMallocManaged((void **)&rhs, sizeof(float)*N));
    // temp memory for CG
    checkCudaErrors(cudaMallocManaged((void **)&r, N*sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&p, N*sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&Ax, N*sizeof(float)));
    cuda_malloc_stop = mysecond();
#endif

#ifdef CUDA_UM_ADVISE
    advise_start = cuda_malloc_stop;
    checkCudaErrors(cudaMemAdvise(I, sizeof(int)*(N+1), cudaMemAdviseSetPreferredLocation, devID));
    checkCudaErrors(cudaMemAdvise(I, sizeof(int)*(N+1), cudaMemAdviseSetAccessedBy, cudaCpuDeviceId));

    checkCudaErrors(cudaMemAdvise(J, sizeof(int)*nz, cudaMemAdviseSetPreferredLocation, devID));
    checkCudaErrors(cudaMemAdvise(J, sizeof(int)*nz, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId));

    checkCudaErrors(cudaMemAdvise(val, sizeof(float)*nz, cudaMemAdviseSetPreferredLocation, devID));
    checkCudaErrors(cudaMemAdvise(val, sizeof(float)*nz, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId));

    checkCudaErrors(cudaMemAdvise(x, sizeof(float)*N, cudaMemAdviseSetPreferredLocation, devID));
    checkCudaErrors(cudaMemAdvise(x, sizeof(float)*N, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId));

    checkCudaErrors(cudaMemAdvise(r, sizeof(float)*N, cudaMemAdviseSetPreferredLocation, devID));
    checkCudaErrors(cudaMemAdvise(p, sizeof(float)*N, cudaMemAdviseSetPreferredLocation, devID));
    checkCudaErrors(cudaMemAdvise(Ax, sizeof(float)*N, cudaMemAdviseSetPreferredLocation, devID));
    advise_stop = mysecond();
#endif

    init_data_start = mysecond();
#ifndef CUDA_UM
    genTridiag(h_I, h_J, h_val, N, nz);
#else
    genTridiag(I, J, val, N, nz);
#endif

    for (int i = 0; i < N; i++)
    {
#ifndef CUDA_UM
        h_rhs[i] = 1.0;
        h_x[i] = 0.0;
#else
        rhs[i] = 1.0;
        x[i] = 0.0;
        r[i] = rhs[i];
#endif
    }
    init_data_stop = mysecond();

#ifdef CUDA_UM_ADVISE
    advise_read_start = init_data_stop;
    checkCudaErrors(cudaMemAdvise(I, sizeof(int)*(N+1), cudaMemAdviseSetReadMostly, devID));
    checkCudaErrors(cudaMemAdvise(J, sizeof(int)*nz, cudaMemAdviseSetReadMostly, devID));
    checkCudaErrors(cudaMemAdvise(val, sizeof(float)*nz, cudaMemAdviseSetReadMostly, devID));
    advise_read_stop = mysecond();
#endif


#ifdef CUDA_UM_PREFETCH
    h2d_prefetch_start = mysecond();
    cudaMemPrefetchAsync(r, sizeof(float)*N, devID);
    cudaMemPrefetchAsync(I, sizeof(int)*(N+1), devID);
    cudaMemPrefetchAsync(J, sizeof(int)*nz, devID);
    cudaMemPrefetchAsync(val, sizeof(float)*nz, devID);
    h2d_prefetch_stop = mysecond();
#endif

#ifndef CUDA_UM
    h2d_memcpy_start = mysecond();
    cudaMemcpy(J, h_J, nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(I, h_I, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(val, h_val, nz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x, h_x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(r, h_rhs, N*sizeof(float), cudaMemcpyHostToDevice);
    h2d_memcpy_stop = mysecond();
#endif

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

    cudaDeviceSynchronize();
    cublas_init_stop = mysecond();

    compute_migrate_start = mysecond();

    alpha = 1.0;
    alpham1 = -1.0;
    beta = 0.0;
    r0 = 0.;

    gpu_start = mysecond();

    cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, val, I, J, x, &beta, Ax);

    cublasSaxpy(cublasHandle, N, &alpham1, Ax, 1, r, 1);
    cublasStatus = cublasSdot(cublasHandle, N, r, 1, r, 1, &r1);

    k = 1;

    //while (r1 > tol*tol && k <= max_iter)
    while (k <= max_iter)
    {
        if (k > 1)
        {
            b = r1 / r0;
            cublasStatus = cublasSscal(cublasHandle, N, &b, p, 1);
            cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, r, 1, p, 1);
        }
        else
        {
            cublasStatus = cublasScopy(cublasHandle, N, r, 1, p, 1);
        }

        cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, val, I, J, p, &beta, Ax);
        cublasStatus = cublasSdot(cublasHandle, N, p, 1, Ax, 1, &dot);
        a = r1 / dot;

        cublasStatus = cublasSaxpy(cublasHandle, N, &a, p, 1, x, 1);
        na = -a;
        cublasStatus = cublasSaxpy(cublasHandle, N, &na, Ax, 1, r, 1);

        r0 = r1;
        cublasStatus = cublasSdot(cublasHandle, N, r, 1, r, 1, &r1);
        cudaDeviceSynchronize();
        printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }

    gpu_stop = mysecond();

#ifndef CUDA_UM
    d2h_memcpy_start = gpu_stop;
    cudaMemcpy(h_x, x, N*sizeof(float), cudaMemcpyDeviceToHost);
    d2h_memcpy_stop = mysecond();
#endif

#ifdef CUDA_UM_PREFETCH
    d2h_prefetch_start = gpu_stop;
    cudaMemPrefetchAsync(I, sizeof(int)*(N+1), cudaCpuDeviceId);
    cudaMemPrefetchAsync(J, sizeof(int)*nz, cudaCpuDeviceId);
    cudaMemPrefetchAsync(x, sizeof(float)*N, cudaCpuDeviceId);
    cudaMemPrefetchAsync(val, sizeof(float)*nz, cudaCpuDeviceId);
    d2h_prefetch_stop = mysecond();
#endif

    cpu_start = mysecond();

    printf("Final residual: %e\n",sqrt(r1));

    fprintf(stdout,"&&&& conjugateGradientUM %s\n", (sqrt(r1) < tol) ? "PASSED" : "FAILED");

    float rsum, diff, err = 0.0;
#ifndef CUDA_UM
    for (int i = 0; i < N; i++)
    {
        rsum = 0.0;

        for (int j = h_I[i]; j < h_I[i+1]; j++)
        {
            rsum += h_val[j]*h_x[h_J[j]];
        }

        diff = fabs(rsum - h_rhs[i]);

        if (diff > err)
        {
            err = diff;
        }
    }
#else
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
#endif

    cpu_stop = mysecond();

    compute_migrate_stop = cpu_stop;

    cublas_destroy_start = cpu_stop;
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
    cublas_destroy_stop = mysecond();

    cuda_free_start = cublas_destroy_stop;
    cudaFree(I);
    cudaFree(J);
    cudaFree(val);
    cudaFree(x);
    cudaFree(rhs);
    cudaFree(r);
    cudaFree(p);
    cudaFree(Ax);
    cuda_free_stop = mysecond();

    application_stop = cuda_free_stop;

    printf("\nGPU Time: %f\n", gpu_stop - gpu_start);
    printf("CPU Time: %f\n", cpu_stop - cpu_start);
    printf("cuda malloc timer: %f\n", cuda_malloc_stop - cuda_malloc_start);
    printf("cuda free timer: %f\n", cuda_free_stop - cuda_free_start);
    printf("Init data timer: %f\n", init_data_stop - init_data_start);
#ifndef CUDA_UM
    printf("\nH2D timer: %f\n", h2d_memcpy_stop - h2d_memcpy_start);
    printf("D2H timer: %f\n", d2h_memcpy_stop - d2h_memcpy_start);
#endif
#ifdef CUDA_UM_ADVISE
    printf("\nadvise timer: %f\n", (advise_stop - advise_start) + (advise_read_stop - advise_read_start));
#endif
#ifdef CUDA_UM_PREFETCH
    printf("\nH2D async prefetch timer: %f\n", h2d_prefetch_stop - h2d_prefetch_start);
    printf("D2H async prefetch timer: %f\n", d2h_prefetch_stop - d2h_prefetch_start);
#endif
    printf("\ncublas init timer: %f\n", cublas_init_stop - cublas_init_start);
    printf("cublas destroy timer: %f\n", cublas_destroy_stop - cublas_destroy_start);
//    printf("misc timer: %f\n", cuda_malloc_start - application_start);
    printf("\ncompute migrate timer: %f\n", compute_migrate_stop - compute_migrate_start);
    printf("application timer: %f\n\n", application_stop - application_start);

    //printf("Test Summary:  Error amount = %f, result = %s\n", err, (k <= max_iter) ? "SUCCESS" : "FAILURE");
    printf("Test Summary:  Error amount = %f\n", err);
    //exit((k <= max_iter) ? EXIT_SUCCESS : EXIT_FAILURE);
    return 0;
}
