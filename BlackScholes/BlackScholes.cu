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
 * This sample evaluates fair call and put prices for a
 * given set of European options by Black-Scholes formula.
 * See supplied whitepaper for more explanations.
 */


#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>        // helper functions CUDA error checking and initialization

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
// because we trigger migration via std::memcpy
static double d2h_memcpy_start;
static double d2h_memcpy_stop;
#ifdef CUDA_UM_ADVISE
static double advise_start;
static double advise_stop;
#endif
#ifdef CUDA_UM_PREFETCH
static double h2d_prefetch_start;
static double h2d_prefetch_stop;
static double d2h_prefetch_start;
static double d2h_prefetch_stop;
#endif

double mysecond(){
    struct timeval tp;
    struct timezone tzp;
    int i;

    i = gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(
    float *h_CallResult,
    float *h_PutResult,
    float *h_StockPrice,
    float *h_OptionStrike,
    float *h_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
);

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
#include "BlackScholes_kernel.cuh"

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
unsigned long int OPT_N = 128000000;
//const int OPT_N = 180000000;
int  NUM_ITERATIONS = 512;
//const int  NUM_ITERATIONS = 1;

bool validate = false;

unsigned long int          OPT_SZ = OPT_N * sizeof(float);
const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
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

    //////////////////////////////// START APPLICATION TIMER /////////////////////////////////////
    application_start = mysecond();

    if (checkCmdLineFlag(argc, (const char **)argv, "size")) {
        OPT_N = getCmdLineArgumentInt(argc, (const char **)argv, "size");
        OPT_SZ = OPT_N * sizeof(float);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "iterations")) {
        NUM_ITERATIONS = getCmdLineArgumentInt(argc, (const char **)argv, "iterations");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "validate")) {
        validate = true;
    }

    // Start logs
    printf("[%s] - Starting...\n", argv[0]);

    //'h_' prefix - CPU (host) memory space
    float
    //Results calculated by CPU for reference
    *h_CallResultCPU,
    *h_PutResultCPU,
    //CPU copy of GPU results
    *h_CallResultGPU,
    *h_PutResultGPU,
    //CPU instance of input data
    *h_StockPrice,
    *h_OptionStrike,
    *h_OptionYears;

    //'d_' prefix - GPU (device) memory space
    float
    //Results calculated by GPU
    *d_CallResult,
    *d_PutResult,
    //GPU instance of input data
    *d_StockPrice,
    *d_OptionStrike,
    *d_OptionYears;

    double
    delta, ref, sum_delta, sum_ref, max_delta, L1norm, gpuTime, gpuTotalTime, cpuTime;

    StopWatchInterface *cTimer = NULL;
    int i;

    int devID = findCudaDevice(argc, (const char **)argv);
    printf("GPU ID: %d, CPU ID: %d\n", devID, cudaCpuDeviceId);

    sdkCreateTimer(&cTimer);

    printf("Initializing data...\n");
    printf("...allocating CPU memory for options.\n");
    malloc_start = mysecond();
    h_CallResultCPU = (float *)malloc(OPT_SZ);
    h_PutResultCPU  = (float *)malloc(OPT_SZ);
#ifndef CUDA_UM
    h_CallResultCPU = (float *)malloc(OPT_SZ);
    h_PutResultCPU  = (float *)malloc(OPT_SZ);
    h_CallResultGPU = (float *)malloc(OPT_SZ);
    h_PutResultGPU  = (float *)malloc(OPT_SZ);
    h_StockPrice    = (float *)malloc(OPT_SZ);
    h_OptionStrike  = (float *)malloc(OPT_SZ);
    h_OptionYears   = (float *)malloc(OPT_SZ);
#endif
    malloc_stop = mysecond();

    printf("...allocating GPU memory for options.\n");
#ifndef CUDA_UM
    cuda_malloc_start = mysecond();
    checkCudaErrors(cudaMalloc((void **)&d_CallResult,   OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_PutResult,    OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_StockPrice,   OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_OptionStrike, OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_OptionYears,  OPT_SZ));
    cuda_malloc_stop = mysecond();
#else
    cuda_malloc_start = mysecond();
    checkCudaErrors(cudaMallocManaged((void **)&d_CallResult,   OPT_SZ));
    checkCudaErrors(cudaMallocManaged((void **)&d_PutResult,    OPT_SZ));
    checkCudaErrors(cudaMallocManaged((void **)&d_StockPrice,   OPT_SZ));
    checkCudaErrors(cudaMallocManaged((void **)&d_OptionStrike, OPT_SZ));
    checkCudaErrors(cudaMallocManaged((void **)&d_OptionYears,  OPT_SZ));
    cuda_malloc_stop = mysecond();

    h_CallResultGPU = d_CallResult;
    h_PutResultGPU  = d_PutResult;
    h_StockPrice    = d_StockPrice;
    h_OptionStrike  = d_OptionStrike;
    h_OptionYears   = d_OptionYears;
#endif

    printf("...generating input data in CPU mem.\n");
    srand(5347);

    //Generate options set
    init_data_start = mysecond();
    for (i = 0; i < OPT_N; i++)
    {
        h_CallResultCPU[i] = 0.0f;
        h_PutResultCPU[i]  = -1.0f;
        h_StockPrice[i]    = RandFloat(5.0f, 30.0f);
        h_OptionStrike[i]  = RandFloat(1.0f, 100.0f);
        h_OptionYears[i]   = RandFloat(0.25f, 10.0f);
    }
    init_data_stop = mysecond();

#ifdef CUDA_UM_ADVISE
    advise_start = mysecond();
    cudaMemAdvise(h_StockPrice, OPT_SZ, cudaMemAdviseSetReadMostly, devID);
    cudaMemAdvise(h_OptionStrike, OPT_SZ, cudaMemAdviseSetReadMostly, devID);
    cudaMemAdvise(h_OptionYears, OPT_SZ, cudaMemAdviseSetReadMostly, devID);
    advise_stop = mysecond();
#endif

#ifndef CUDA_UM
    printf("...copying input data to GPU mem.\n");
    //Copy options data to GPU memory for further processing
    h2d_memcpy_start = mysecond();
    checkCudaErrors(cudaMemcpy(d_StockPrice,  h_StockPrice,   OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionStrike, h_OptionStrike,  OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionYears,  h_OptionYears,   OPT_SZ, cudaMemcpyHostToDevice));
    h2d_memcpy_stop = mysecond();
#endif
    printf("Data init done.\n\n");

#ifdef CUDA_UM_PREFETCH
    printf("...prefetching input data to GPU mem.\n");
    h2d_prefetch_start = mysecond();
    checkCudaErrors(cudaMemPrefetchAsync(h_StockPrice, OPT_SZ, devID));
    checkCudaErrors(cudaMemPrefetchAsync(h_OptionStrike, OPT_SZ, devID));
    checkCudaErrors(cudaMemPrefetchAsync(h_OptionYears, OPT_SZ, devID));

    checkCudaErrors(cudaMemPrefetchAsync(h_CallResultGPU, OPT_SZ, devID));
    checkCudaErrors(cudaMemPrefetchAsync(h_PutResultGPU, OPT_SZ, devID));
    h2d_prefetch_stop = mysecond();
#endif

    printf("Executing Black-Scholes GPU kernel (%i iterations)...\n", NUM_ITERATIONS);
    //checkCudaErrors(cudaDeviceSynchronize());
    /////////////////////// START TIMER //////////////////////////////////
    compute_migrate_start = mysecond();

    for (i = 0; i < NUM_ITERATIONS; i++)
    {
        BlackScholesGPU<<<DIV_UP((OPT_N/2), 128), 128/*480, 128*/>>>(
            (float2 *)d_CallResult,
            (float2 *)d_PutResult,
            (float2 *)d_StockPrice,
            (float2 *)d_OptionStrike,
            (float2 *)d_OptionYears,
            RISKFREE,
            VOLATILITY,
            OPT_N
        );
        getLastCudaError("BlackScholesGPU() execution failed\n");
    }

    checkCudaErrors(cudaDeviceSynchronize());

    gpuTotalTime = (mysecond() - compute_migrate_start) * 1000.0;
    gpuTime = gpuTotalTime / NUM_ITERATIONS;

#ifdef CUDA_UM_PREFETCH
    d2h_prefetch_start = mysecond();
    checkCudaErrors(cudaMemPrefetchAsync(h_CallResultGPU, OPT_SZ, cudaCpuDeviceId));
    checkCudaErrors(cudaMemPrefetchAsync(h_PutResultGPU, OPT_SZ, cudaCpuDeviceId));
    d2h_prefetch_stop = mysecond();
#endif

    //Both call and put is calculated
    printf("Options count             : %ld    \n", 2 * OPT_N);
    printf("BlackScholesGPU() time    : %f msec\n", gpuTime);
    printf("Effective memory bandwidth: %f GB/s\n", ((double)(5 * OPT_N * sizeof(float)) * 1E-9) / (gpuTime * 1E-3));
    printf("Gigaoptions per second    : %f     \n\n", ((double)(2 * OPT_N) * 1E-9) / (gpuTime * 1E-3));

    printf("BlackScholes, Throughput = %.4f GOptions/s, Time = %.5f s, Size = %lu options, NumDevsUsed = %u, Workgroup = %u\n",
           (((double)(2.0 * OPT_N) * 1.0E-9) / (gpuTime * 1.0E-3)), gpuTime*1e-3, (2 * OPT_N), 1, 128);

#ifndef CUDA_UM
    printf("\nReading back GPU results...\n");
    //Read back GPU results to compare them to CPU results
    d2h_memcpy_start = mysecond();
    checkCudaErrors(cudaMemcpy(h_CallResultGPU, d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_PutResultGPU,  d_PutResult,  OPT_SZ, cudaMemcpyDeviceToHost));
    d2h_memcpy_stop = mysecond();
#endif

    if (validate) {
        printf("Checking the results...\n");
        printf("...running CPU calculations.\n\n");
        sdkResetTimer(&cTimer);
        sdkStartTimer(&cTimer);
        //Calculate options values on CPU
        BlackScholesCPU(
            h_CallResultCPU,
            h_PutResultCPU,
            h_StockPrice,
            h_OptionStrike,
            h_OptionYears,
            RISKFREE,
            VOLATILITY,
            OPT_N
        );
        sdkStopTimer(&cTimer);
        cpuTime = sdkGetTimerValue(&cTimer);

        printf("Comparing the results...\n");
        //Calculate max absolute difference and L1 distance
        //between CPU and GPU results
        sum_delta = 0;
        sum_ref   = 0;
        max_delta = 0;

        for (i = 0; i < OPT_N; i++)
        {
            ref   = h_CallResultCPU[i];
            delta = fabs(h_CallResultCPU[i] - h_CallResultGPU[i]);

            if (delta > max_delta)
            {
                max_delta = delta;
            }

            sum_delta += delta;
            sum_ref   += fabs(ref);
        }

        L1norm = sum_delta / sum_ref;
        printf("L1 norm: %E\n", L1norm);
        printf("Max absolute error: %E\n\n", max_delta);

        if (L1norm > 1e-6)
        {
            printf("Test failed!\n");
            exit(EXIT_FAILURE);
        }
    }
    else {
        /* fetch back gpu results to trigger migration */
#ifdef CUDA_UM
        d2h_memcpy_start = mysecond();
        memcpy(h_CallResultCPU, h_CallResultGPU, OPT_SZ);
        memcpy(h_PutResultCPU, h_PutResultGPU, OPT_SZ);
        d2h_memcpy_stop = mysecond();
#endif
    }
    //////////////////////////////// END TIMER /////////////////////////////////////
    compute_migrate_stop = mysecond();

    printf("Shutting down...\n");
    printf("...releasing GPU memory.\n");
    cuda_free_start = compute_migrate_stop;
    checkCudaErrors(cudaFree(d_OptionYears));
    checkCudaErrors(cudaFree(d_OptionStrike));
    checkCudaErrors(cudaFree(d_StockPrice));
    checkCudaErrors(cudaFree(d_PutResult));
    checkCudaErrors(cudaFree(d_CallResult));
    cuda_free_stop = mysecond();

    printf("...releasing CPU memory.\n");
    free_start = mysecond();
#ifndef CUDA_UM
    free(h_OptionYears);
    free(h_OptionStrike);
    free(h_StockPrice);
    free(h_PutResultGPU);
    free(h_CallResultGPU);
#endif
    free(h_PutResultCPU);
    free(h_CallResultCPU);
    free_stop = mysecond();

    printf("Shutdown done.\n");
    //////////////////////////////// END APPLICATION TIMER /////////////////////////////////////
    application_stop = mysecond();
    sdkDeleteTimer(&cTimer);

    printf("\n[BlackScholes] - Test Summary\n");
    printf("\nGPU Time: %f\n", gpuTotalTime/1000.0);
    printf("CPU Time: %f\n", cpuTime/1000.0);
    printf("malloc timer: %f\n", malloc_stop - malloc_start);
    printf("free timer: %f\n", free_stop - free_start);
    printf("cuda malloc timer: %f\n", cuda_malloc_stop - cuda_malloc_start);
    printf("cuda free timer: %f\n", cuda_free_stop - cuda_free_start);
    printf("Init data timer: %f\n", init_data_stop - init_data_start);
#ifndef CUDA_UM
    printf("\nH2D timer: %f\n", h2d_memcpy_stop - h2d_memcpy_start);
#endif
    printf("D2H timer: %f\n", d2h_memcpy_stop - d2h_memcpy_start);
#ifdef CUDA_UM_PREFETCH
    printf("\nH2D async prefetch timer: %f\n", h2d_prefetch_stop - h2d_prefetch_start);
    printf("D2H async prefetch timer: %f\n", d2h_prefetch_stop - d2h_prefetch_start);
#endif 
#ifdef CUDA_UM_ADVISE
    printf("advise timer: %f\n", advise_stop - advise_start);
#endif
    printf("\nKernel timer: %f\n", compute_migrate_stop - compute_migrate_start);
    printf("misc timer: %f\n", malloc_start - application_start);
    printf("application timer: %f\n", application_stop - application_start);

    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n\n");
    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}
