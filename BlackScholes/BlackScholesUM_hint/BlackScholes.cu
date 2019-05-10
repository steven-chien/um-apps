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
int OPT_N = 128000000;
//const int OPT_N = 180000000;
int  NUM_ITERATIONS = 512;
//const int  NUM_ITERATIONS = 1;

bool validate = false;

long unsigned int          OPT_SZ = OPT_N * sizeof(float);
const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
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
    delta, ref, sum_delta, sum_ref, max_delta, L1norm, gpuTime, gpuTotalTime, cpuTime, globalTime;

    StopWatchInterface *hTimer = NULL;
    StopWatchInterface *cTimer = NULL;
    StopWatchInterface *gTimer = NULL;
    StopWatchInterface *compute_migrate_timer = NULL;
    int i;

    int devID = findCudaDevice(argc, (const char **)argv);
    printf("GPU ID: %d, CPU ID: %d\n", devID, cudaCpuDeviceId);

    sdkCreateTimer(&hTimer);
    sdkCreateTimer(&cTimer);
    sdkCreateTimer(&gTimer);
    sdkCreateTimer(&compute_migrate_timer);

    sdkResetTimer(&gTimer);
    sdkStartTimer(&gTimer);

    printf("Initializing data...\n");
    printf("...allocating CPU memory for options.\n");
    h_CallResultCPU = (float *)malloc(OPT_SZ);
    h_PutResultCPU  = (float *)malloc(OPT_SZ);
//    h_CallResultGPU = (float *)malloc(OPT_SZ);
//    h_PutResultGPU  = (float *)malloc(OPT_SZ);
//    h_StockPrice    = (float *)malloc(OPT_SZ);
//    h_OptionStrike  = (float *)malloc(OPT_SZ);
//    h_OptionYears   = (float *)malloc(OPT_SZ);

    printf("...allocating GPU memory for options.\n");
    checkCudaErrors(cudaMallocManaged((void **)&d_CallResult,   OPT_SZ));
    checkCudaErrors(cudaMallocManaged((void **)&d_PutResult,    OPT_SZ));
    checkCudaErrors(cudaMallocManaged((void **)&d_StockPrice,   OPT_SZ));
    checkCudaErrors(cudaMallocManaged((void **)&d_OptionStrike, OPT_SZ));
    checkCudaErrors(cudaMallocManaged((void **)&d_OptionYears,  OPT_SZ));

    h_CallResultGPU = d_CallResult;
    h_PutResultGPU  = d_PutResult;
    h_StockPrice    = d_StockPrice;
    h_OptionStrike  = d_OptionStrike;
    h_OptionYears   = d_OptionYears;

    printf("...generating input data in CPU mem.\n");
    srand(5347);

    //Generate options set
    for (i = 0; i < OPT_N; i++)
    {
        h_CallResultCPU[i] = 0.0f;
        h_PutResultCPU[i]  = -1.0f;
        h_StockPrice[i]    = RandFloat(5.0f, 30.0f);
        h_OptionStrike[i]  = RandFloat(1.0f, 100.0f);
        h_OptionYears[i]   = RandFloat(0.25f, 10.0f);
    }

    sdkResetTimer(&compute_migrate_timer);
    sdkStartTimer(&compute_migrate_timer);

    cudaMemAdvise(h_StockPrice, OPT_SZ, cudaMemAdviseSetReadMostly, devID);
    cudaMemAdvise(h_OptionStrike, OPT_SZ, cudaMemAdviseSetReadMostly, devID);
    cudaMemAdvise(h_OptionYears, OPT_SZ, cudaMemAdviseSetReadMostly, devID);

//    cudaMemPrefetchAsync(h_StockPrice, OPT_SZ, devID, NULL);
//    cudaMemPrefetchAsync(h_OptionStrike, OPT_SZ, devID, NULL);
//    cudaMemPrefetchAsync(h_OptionYears, OPT_SZ, devID, NULL);

//    cudaMemPrefetchAsync(h_CallResultGPU, OPT_SZ, 0, NULL);
//    cudaMemPrefetchAsync(h_PutResultGPU, OPT_SZ, 0, NULL);

    printf("...copying input data to GPU mem.\n");
    //Copy options data to GPU memory for further processing
//    checkCudaErrors(cudaMemcpy(d_StockPrice,  h_StockPrice,   OPT_SZ, cudaMemcpyHostToDevice));
//    checkCudaErrors(cudaMemcpy(d_OptionStrike, h_OptionStrike,  OPT_SZ, cudaMemcpyHostToDevice));
//    checkCudaErrors(cudaMemcpy(d_OptionYears,  h_OptionYears,   OPT_SZ, cudaMemcpyHostToDevice));
    printf("Data init done.\n\n");


    printf("Executing Black-Scholes GPU kernel (%i iterations)...\n", NUM_ITERATIONS);
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

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

//    cudaMemPrefetchAsync(h_CallResultGPU, OPT_SZ, cudaCpuDeviceId, NULL);
    //cudaMemPrefetchAsync(h_PutResultGPU, OPT_SZ, cudaCpuDeviceId, NULL);

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    gpuTotalTime = sdkGetTimerValue(&hTimer);
    gpuTime = gpuTotalTime / NUM_ITERATIONS;

    //Both call and put is calculated
    printf("Options count             : %i     \n", 2 * OPT_N);
    printf("BlackScholesGPU() time    : %f msec\n", gpuTime);
    printf("Effective memory bandwidth: %f GB/s\n", ((double)(5 * OPT_N * sizeof(float)) * 1E-9) / (gpuTime * 1E-3));
    printf("Gigaoptions per second    : %f     \n\n", ((double)(2 * OPT_N) * 1E-9) / (gpuTime * 1E-3));

    printf("BlackScholes, Throughput = %.4f GOptions/s, Time = %.5f s, Size = %u options, NumDevsUsed = %u, Workgroup = %u\n",
           (((double)(2.0 * OPT_N) * 1.0E-9) / (gpuTime * 1.0E-3)), gpuTime*1e-3, (2 * OPT_N), 1, 128);

    printf("\nReading back GPU results...\n");
    //Read back GPU results to compare them to CPU results
//    checkCudaErrors(cudaMemcpy(h_CallResultGPU, d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost));
//    checkCudaErrors(cudaMemcpy(h_PutResultGPU,  d_PutResult,  OPT_SZ, cudaMemcpyDeviceToHost));


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
        for (i = 0; i < OPT_N; i++) {
            h_CallResultCPU[i] = h_CallResultGPU[i];
            h_PutResultCPU[i] = h_PutResultGPU[i];
        }
    }
    sdkStopTimer(&compute_migrate_timer);
    double compute_migrate_time = sdkGetTimerValue(&compute_migrate_timer);

    printf("Shutting down...\n");
    printf("...releasing GPU memory.\n");
    checkCudaErrors(cudaFree(d_OptionYears));
    checkCudaErrors(cudaFree(d_OptionStrike));
    checkCudaErrors(cudaFree(d_StockPrice));
    checkCudaErrors(cudaFree(d_PutResult));
    checkCudaErrors(cudaFree(d_CallResult));

    printf("...releasing CPU memory.\n");
//    free(h_OptionYears);
//    free(h_OptionStrike);
//    free(h_StockPrice);
//    free(h_PutResultGPU);
//    free(h_CallResultGPU);
    free(h_PutResultCPU);
    free(h_CallResultCPU);

    sdkStopTimer(&gTimer);
    globalTime = sdkGetTimerValue(&gTimer);

    sdkDeleteTimer(&hTimer);
    sdkDeleteTimer(&gTimer);
    sdkDeleteTimer(&cTimer);
    sdkDeleteTimer(&compute_migrate_timer);

    printf("Shutdown done.\n");

    printf("\n[BlackScholes] - Test Summary\n");
    printf("\nGPU Time: %f, CPU Time: %f, Global Time: %f, compute migrate: %f\n", gpuTotalTime, cpuTime, globalTime, compute_migrate_time);

    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n\n");
    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}
