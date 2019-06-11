/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* This example demonstrates how to use the CUBLAS library
 * by scaling an array of floating-point values on the device
 * and comparing the result to the same operation performed
 * on the host.
 */

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

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
double advise_read_start;
double advise_read_stop;
double cublas_init_start;
double cublas_init_stop;
double cublas_destroy_start;
double cublas_destroy_stop;
double misc_start;
double misc_stop;
double misc_timer;

/* Matrix size */
//#define N (275)
unsigned long int N = 275;
int nIter = 1;
bool validate = false;

#include <sys/time.h>
double mysecond(){
    struct timeval tp;
    struct timezone tzp;
    int i;

    i = gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

/* Host implementation of a simple version of sgemm */
static void simple_sgemm(int n, float alpha, const float *A, const float *B,
                         float beta, float *C) {
  int i;
  int j;
  int k;

  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      float prod = 0;

      for (k = 0; k < n; ++k) {
        prod += A[k * n + i] * B[j * n + k];
      }

      C[j * n + i] = alpha * prod + beta * C[j * n + i];
    }
  }
}

/* Main */
int main(int argc, char **argv) {
  application_start = mysecond();

  cublasStatus_t status;
  float *h_A;
  float *h_B;
  float *h_C;
  float *h_C_ref;
  float *d_A = 0;
  float *d_B = 0;
  float *d_C = 0;
  float alpha = 1.0f;
  float beta = 0.0f;
  unsigned long int n2 = N * N;
  int i;
  float error_norm;
  float ref_norm;
  float diff;
  cublasHandle_t handle;

  int dev = findCudaDevice(argc, (const char **)argv);

  if (dev == -1) {
    return EXIT_FAILURE;
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "size")) {
    N = getCmdLineArgumentInt(argc, (const char **)argv, "size");
    n2 = N * N;
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "iterations")) {
    nIter = getCmdLineArgumentInt(argc, (const char **)argv, "iterations");
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "validate")) {
    validate = true;
  }

  printf("size: %ldx%ld\n", N, N);

  /* Initialize CUBLAS */
  printf("simpleCUBLAS test running..\n");

  cublas_init_start = mysecond();
  status = cublasCreate(&handle);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! CUBLAS initialization error\n");
    return EXIT_FAILURE;
  }
  cublas_init_stop = mysecond();

  /* Allocate host memory for the matrices */
  /* Allocate host memory for reading back the result from device memory */
  malloc_start = cublas_init_stop;
  h_C_ref = reinterpret_cast<float *>(malloc(n2 * sizeof(h_C[0])));

  if (h_C_ref == 0) {
    fprintf(stderr, "!!!! host memory allocation error (C)\n");
    return EXIT_FAILURE;
  }
  malloc_stop = mysecond();

  /* Allocate device memory for the matrices */
  cuda_malloc_start = malloc_stop;
  if (cudaMallocManaged(reinterpret_cast<void **>(&d_A), n2 * sizeof(d_A[0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
    return EXIT_FAILURE;
  }

  if (cudaMallocManaged(reinterpret_cast<void **>(&d_B), n2 * sizeof(d_B[0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
    return EXIT_FAILURE;
  }

  if (cudaMallocManaged(reinterpret_cast<void **>(&d_C), n2 * sizeof(d_C[0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
    return EXIT_FAILURE;
  }

  h_A = d_A;
  h_B = d_B;
  h_C = d_C;
  cuda_malloc_stop = mysecond();

  advise_start = cuda_malloc_stop;
  cudaMemAdvise(d_A, n2 * sizeof(d_A[0]), cudaMemAdviseSetPreferredLocation, dev);
  cudaMemAdvise(d_A, n2 * sizeof(d_A[0]), cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
  cudaMemAdvise(d_B, n2 * sizeof(d_B[0]), cudaMemAdviseSetPreferredLocation, dev);
  cudaMemAdvise(d_B, n2 * sizeof(d_B[0]), cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
  cudaMemAdvise(d_C, n2 * sizeof(d_C[0]), cudaMemAdviseSetPreferredLocation, dev);
  cudaMemAdvise(d_C, n2 * sizeof(d_C[0]), cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
  advise_stop = mysecond();

  /* Fill the matrices with test data */
  init_data_start = advise_stop;
  for (i = 0; i < n2; i++) {
    h_A[i] = rand() / static_cast<float>(RAND_MAX);
    h_B[i] = rand() / static_cast<float>(RAND_MAX);
    h_C[i] = rand() / static_cast<float>(RAND_MAX);
  }
  memcpy(h_C_ref, h_C, n2 * sizeof(h_C[0]));
  init_data_stop = mysecond();

  advise_read_start = init_data_stop;
  cudaMemAdvise(d_A, n2 * sizeof(d_A[0]), cudaMemAdviseSetReadMostly, dev);
  cudaMemAdvise(d_B, n2 * sizeof(d_B[0]), cudaMemAdviseSetReadMostly, dev);
  advise_read_stop = mysecond();

//  /* Initialize the device matrices with the host matrices */
//  status = cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1);
//
//  if (status != CUBLAS_STATUS_SUCCESS) {
//    fprintf(stderr, "!!!! device access error (write A)\n");
//    return EXIT_FAILURE;
//  }
//
//  status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);
//
//  if (status != CUBLAS_STATUS_SUCCESS) {
//    fprintf(stderr, "!!!! device access error (write B)\n");
//    return EXIT_FAILURE;
//  }
//
//  status = cublasSetVector(n2, sizeof(h_C[0]), h_C, 1, d_C, 1);
//
//  if (status != CUBLAS_STATUS_SUCCESS) {
//    fprintf(stderr, "!!!! device access error (write C)\n");
//    return EXIT_FAILURE;
//  }

  if (validate) {
    /* Performs operation using plain C code */
    cpu_start = mysecond();
    simple_sgemm(N, alpha, h_A, h_B, beta, h_C_ref);
    cpu_stop = mysecond();
    //h_C_ref = h_C;
  }

  compute_migrate_start = mysecond();
  gpu_start = compute_migrate_start;
  for (int i = 0; i < nIter; i++) {
    /* Performs operation using cublas */
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A,
                         N, d_B, N, &beta, d_C, N);
  
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "!!!! kernel execution error.\n");
      return EXIT_FAILURE;
    }
  }
  cudaDeviceSynchronize();
  gpu_stop = mysecond();

//  /* Read the result back */
//  status = cublasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1);
//
//  if (status != CUBLAS_STATUS_SUCCESS) {
//    fprintf(stderr, "!!!! device access error (read C)\n");
//    return EXIT_FAILURE;
//  }

  if (validate) {
    misc_start = mysecond();
    /* Check result against reference */
    error_norm = 0;
    ref_norm = 0;
  
    for (i = 0; i < n2; ++i) {
      diff = h_C_ref[i] - h_C[i];
      error_norm += diff * diff;
      ref_norm += h_C_ref[i] * h_C_ref[i];
    }
  
    error_norm = static_cast<float>(sqrt(static_cast<double>(error_norm)));
    ref_norm = static_cast<float>(sqrt(static_cast<double>(ref_norm)));
  
    if (fabs(ref_norm) < 1e-7) {
      fprintf(stderr, "!!!! reference norm is 0\n");
      return EXIT_FAILURE;
    }

    if (error_norm / ref_norm < 1e-6f) {
      printf("simpleCUBLAS test passed.\n");
      //exit(EXIT_SUCCESS);
    } else {
      printf("simpleCUBLAS test failed.\n");
      //exit(EXIT_FAILURE);
    }
    misc_stop = mysecond();
  }
  else {
    d2h_memcpy_start = mysecond();
    memcpy(h_C_ref, d_C, n2 * sizeof(float));
    d2h_memcpy_stop = mysecond();
  }

  compute_migrate_stop = mysecond();

  /* Memory clean up */
//  free(h_A);
//  free(h_B);
//  free(h_C);
  free_start = compute_migrate_stop;
  free(h_C_ref);
  free_stop = mysecond();

  cuda_free_start = free_stop;
  if (cudaFree(d_A) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (A)\n");
    return EXIT_FAILURE;
  }

  if (cudaFree(d_B) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (B)\n");
    return EXIT_FAILURE;
  }

  if (cudaFree(d_C) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (C)\n");
    return EXIT_FAILURE;
  }
  cuda_free_stop = mysecond();

  /* Shutdown */
  cublas_destroy_start = mysecond();
  status = cublasDestroy(handle);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! shutdown error (A)\n");
    return EXIT_FAILURE;
  }
  cublas_destroy_stop = mysecond();

  application_stop = cublas_destroy_stop;

  printf("\nGPU Time: %f\n", gpu_stop - gpu_start);
    printf("CPU Time: %f\n", (cpu_stop - cpu_start) + (misc_stop - misc_start));
    printf("malloc timer: %f\n", malloc_stop - malloc_start);
    printf("free timer: %f\n", free_stop - free_start);
    printf("cuda malloc timer: %f\n", cuda_malloc_stop - cuda_malloc_start);
    printf("cuda free timer: %f\n", cuda_free_stop - cuda_free_start);
    printf("Init data timer: %f\n", init_data_stop - init_data_start);
    printf("misc timer: %f\n", (cublas_init_start - application_start) + (misc_stop - misc_start));
    printf("cublas init timer: %f\n", cublas_init_stop - cublas_init_start);
    printf("cublas destroy timer: %f\n", cublas_destroy_stop - cublas_destroy_start);
    printf("\nAdvise timer: %f\n", (advise_stop - advise_start) + (advise_read_stop - advise_read_start));
    printf("\nH2D async prefetch timer: %f\n", h2d_prefetch_stop - h2d_prefetch_start);
    printf("D2H async prefetch timer: %f\n", d2h_prefetch_stop - d2h_prefetch_start);
    printf("\nH2D timer: %f\n", h2d_memcpy_stop - h2d_memcpy_start);
    printf("D2H timer: %f\n", d2h_memcpy_stop - d2h_memcpy_start);
    printf("\ncompute migrate timer: %f\n", compute_migrate_stop - compute_migrate_start);
    printf("applicaiton timer: %f\n\n", application_stop - application_start);
}
