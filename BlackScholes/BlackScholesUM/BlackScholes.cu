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

double application_start;
double application_stop;
double compute_migrate_start;
double compute_migrate_stop;
double init_data_start;
double init_data_stop;
double malloc_start;
double malloc_stop;
double free_start;
double free_stop;
double cuda_malloc_start;
double cuda_malloc_stop;
double cuda_free_start;
double cuda_free_stop;
double d2h_memcpy_start;
double d2h_memcpy_stop;

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
    malloc_stop = mysecond();

    printf("...allocating GPU memory for options.\n");
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

    printf("Data init done.\n\n");

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

    //Both call and put is calculated
    printf("Options count             : %ld    \n", 2 * OPT_N);
    printf("BlackScholesGPU() time    : %f msec\n", gpuTime);
    printf("Effective memory bandwidth: %f GB/s\n", ((double)(5 * OPT_N * sizeof(float)) * 1E-9) / (gpuTime * 1E-3));
    printf("Gigaoptions per second    : %f     \n\n", ((double)(2 * OPT_N) * 1E-9) / (gpuTime * 1E-3));

    printf("BlackScholes, Throughput = %.4f GOptions/s, Time = %.5f s, Size = %lu options, NumDevsUsed = %u, Workgroup = %u\n",
           (((double)(2.0 * OPT_N) * 1.0E-9) / (gpuTime * 1.0E-3)), gpuTime*1e-3, (2 * OPT_N), 1, 128);

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
        d2h_memcpy_start = mysecond();
        memcpy(h_CallResultCPU, h_CallResultGPU, OPT_SZ);
        memcpy(h_PutResultCPU, h_PutResultGPU, OPT_SZ);
        d2h_memcpy_stop = mysecond();
    }
    compute_migrate_stop = mysecond();

    printf("Shutting down...\n");
    printf("...releasing GPU memory.\n");
    cuda_free_start = mysecond();
    checkCudaErrors(cudaFree(d_OptionYears));
    checkCudaErrors(cudaFree(d_OptionStrike));
    checkCudaErrors(cudaFree(d_StockPrice));
    checkCudaErrors(cudaFree(d_PutResult));
    checkCudaErrors(cudaFree(d_CallResult));
    cuda_free_stop = mysecond();

    printf("...releasing CPU memory.\n");
    free_start = mysecond();
//    free(h_OptionYears);
//    free(h_OptionStrike);
//    free(h_StockPrice);
//    free(h_PutResultGPU);
//    free(h_CallResultGPU);
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
    printf("misc timer: %f\n", malloc_start - application_start);
    printf("\nD2H timer: %f\n", d2h_memcpy_stop - d2h_memcpy_start);
    printf("\ncompute migrate timer: %f\n", compute_migrate_stop - compute_migrate_start);
    printf("application timer: %f\n", application_stop - application_start);

    //printf("Test passed\n");
    exit(EXIT_SUCCESS);
}
