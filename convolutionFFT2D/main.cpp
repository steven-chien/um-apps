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
 * This sample demonstrates how 2D convolutions
 * with very large kernel sizes
 * can be efficiently implemented
 * using FFT transformations.
 */


#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>

// Helper functions for CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include "convolutionFFT2D_common.h"

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
static double cufft_init_start;
static double cufft_init_stop;
static double cufft_destroy_start;
static double cufft_destroy_stop;

//#define DATA_H 8000
//#define DATA_W 8000
long unsigned int DATA_H = 8000;
long unsigned int DATA_W = 8000;
bool validate = false;
//int nIter = 5;
int fft0Iter = 0;
int fft1Iter = 0;
int fft2Iter = 0;

int devID;

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
int snapTransformSize(int dataSize)
{
    int hiBit;
    unsigned int lowPOT, hiPOT;

    dataSize = iAlignUp(dataSize, 16);

    for (hiBit = 31; hiBit >= 0; hiBit--)
        if (dataSize & (1U << hiBit))
        {
            break;
        }

    lowPOT = 1U << hiBit;

    if (lowPOT == (unsigned int)dataSize)
    {
        return dataSize;
    }

    hiPOT = 1U << (hiBit + 1);

    if (hiPOT <= 1024)
    {
        return hiPOT;
    }
    else
    {
        return iAlignUp(dataSize, 512);
    }
}

float getRand(void)
{
    return (float)(rand() % 16);
}

bool test0(void)
{
    /////////////////////// START TIMER ///////////////////////////
    application_start = mysecond();

    float
    *h_Data,
    *h_Kernel,
    *h_ResultCPU,
    *h_ResultGPU;

    float
    *d_Data,
    *d_PaddedData,
    *d_Kernel,
    *d_PaddedKernel;

    fComplex
    *d_DataSpectrum,
    *d_KernelSpectrum;

    cufftHandle
    fftPlanFwd,
    fftPlanInv;

    bool bRetVal;
    StopWatchInterface *hTimer = NULL;
    sdkCreateTimer(&hTimer);

    printf("Testing built-in R2C / C2R FFT-based convolution\n");
    const int kernelH = 7;
    const int kernelW = 6;
    const int kernelY = 3;
    const int kernelX = 4;
    const int   dataH = DATA_H;
    const int   dataW = DATA_W;
//    const long int   dataH = 11000;
//    const long int   dataW = 10000;
    const int    fftH = snapTransformSize(dataH + kernelH - 1);
    const int    fftW = snapTransformSize(dataW + kernelW - 1);

    printf("...allocating memory\n");
#ifdef CUDA_UM
    malloc_start = mysecond();
    h_ResultCPU = (float *)malloc(dataH   * dataW * sizeof(float));
    malloc_stop = mysecond();

    cuda_malloc_start = malloc_stop;
    checkCudaErrors(cudaMallocManaged((void **)&d_Data,   dataH   * dataW   * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&d_Kernel, kernelH * kernelW * sizeof(float)));
    h_Data = d_Data; h_Kernel = d_Kernel;

    checkCudaErrors(cudaMallocManaged((void **)&d_PaddedData,   fftH * fftW * sizeof(float)));
    h_ResultGPU = d_PaddedData;

    checkCudaErrors(cudaMallocManaged((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));

    checkCudaErrors(cudaMallocManaged((void **)&d_DataSpectrum,   fftH * (fftW / 2 + 1) * sizeof(fComplex)));

    checkCudaErrors(cudaMallocManaged((void **)&d_KernelSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    cuda_malloc_stop = mysecond();
#else
    malloc_start = mysecond();
    h_Data      = (float *)malloc(dataH   * dataW * sizeof(float));
    h_Kernel    = (float *)malloc(kernelH * kernelW * sizeof(float));
    h_ResultCPU = (float *)malloc(dataH   * dataW * sizeof(float));
    h_ResultGPU = (float *)malloc(fftH    * fftW * sizeof(float));
    malloc_stop = mysecond();

    printf("%f MiB\n", (dataH   * dataW   * sizeof(float)+kernelH * kernelW * sizeof(float)+fftH * fftW * sizeof(float)+fftH * fftW * sizeof(float)+fftH * (fftW / 2 + 1) * sizeof(fComplex)+fftH * (fftW / 2 + 1) * sizeof(fComplex))/1048576.0);
    cuda_malloc_start = malloc_stop;
    checkCudaErrors(cudaMalloc((void **)&d_Data,   dataH   * dataW   * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Kernel, kernelH * kernelW * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&d_PaddedData,   fftH * fftW * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&d_DataSpectrum,   fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    checkCudaErrors(cudaMalloc((void **)&d_KernelSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    cuda_malloc_stop = mysecond();
#endif

#ifdef CUDA_UM_ADVISE
    advise_start = mysecond();
    cudaMemAdvise(d_PaddedData, sizeof(float) * fftH * fftW, cudaMemAdviseSetPreferredLocation, devID);
    cudaMemAdvise(d_PaddedData, sizeof(float) * fftH * fftW, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
    cudaMemAdvise(d_PaddedKernel, sizeof(float) * fftH * fftW, cudaMemAdviseSetPreferredLocation, devID);
    cudaMemAdvise(d_DataSpectrum, sizeof(fComplex) * fftH * (fftW / 2 + 1), cudaMemAdviseSetPreferredLocation, devID);
    cudaMemAdvise(d_KernelSpectrum, sizeof(fComplex) * fftH * (fftW / 2 + 1), cudaMemAdviseSetPreferredLocation, devID);
    advise_stop =  mysecond();
#endif

    printf("...generating random input data\n");
    srand(2010);

    init_data_start = mysecond();
#ifdef CUDA_UM
    memset(d_KernelSpectrum, 0, fftH * (fftW / 2 + 1) * sizeof(fComplex));
    memset(d_PaddedKernel, 0, fftH * fftW * sizeof(float));
    memset(d_PaddedData,   0, fftH * fftW * sizeof(float));
#else
    checkCudaErrors(cudaMemset(d_KernelSpectrum, 0, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    checkCudaErrors(cudaMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(float)));
    checkCudaErrors(cudaMemset(d_PaddedData,   0, fftH * fftW * sizeof(float)));
#endif

    for (int i = 0; i < dataH * dataW; i++)
    {
        h_Data[i] = getRand();
    }

    for (int i = 0; i < kernelH * kernelW; i++)
    {
        h_Kernel[i] = getRand();
    }
    init_data_stop = mysecond();

#ifdef CUDA_UM_ADVISE
    advise_read_start = mysecond();
    cudaMemAdvise(d_Data, dataH   * dataW   * sizeof(float), cudaMemAdviseSetReadMostly, devID);
    cudaMemAdvise(d_Kernel, kernelH   * kernelW   * sizeof(float), cudaMemAdviseSetReadMostly, devID);
    cudaMemAdvise(d_Kernel, sizeof(float) * kernelH * kernelW, cudaMemAdviseSetReadMostly, devID);
    cudaMemAdvise(d_Data, sizeof(float) * dataH * dataW, cudaMemAdviseSetReadMostly, devID);
    advise_read_stop = mysecond();
#endif

#ifdef CUDA_UM_PREFETCH
    h2d_prefetch_start = mysecond();
    checkCudaErrors(cudaMemPrefetchAsync(d_Data, dataH * dataW * sizeof(float), devID));
    checkCudaErrors(cudaMemPrefetchAsync(d_Kernel, kernelH * kernelW * sizeof(float), devID));
    h2d_prefetch_stop = mysecond();
#endif

    printf("...creating R2C & C2R FFT plans for %i x %i\n", fftH, fftW);
    cufft_init_start = mysecond();
    checkCudaErrors(cufftPlan2d(&fftPlanFwd, fftH, fftW, CUFFT_R2C));
    checkCudaErrors(cufftPlan2d(&fftPlanInv, fftH, fftW, CUFFT_C2R));
    cufft_init_stop = mysecond();

#ifndef CUDA_UM
    printf("...uploading to GPU and padding convolution kernel and input data\n");
    h2d_memcpy_start = mysecond();
    checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel, kernelH * kernelW * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Data,   h_Data,   dataH   * dataW *   sizeof(float), cudaMemcpyHostToDevice));
    h2d_memcpy_stop = mysecond();
#endif

    compute_migrate_start = mysecond();
    gpu_start = compute_migrate_start;

    padKernel(
        d_PaddedKernel,
        d_Kernel,
        fftH,
        fftW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
    );

    padDataClampToBorder(
        d_PaddedData,
        d_Data,
        fftH,
        fftW,
        dataH,
        dataW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
    );

    //Not including kernel transformation into time measurement,
    //since convolution kernel is not changed very frequently
    printf("...transforming convolution kernel\n");
    checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_PaddedKernel, (cufftComplex *)d_KernelSpectrum));

    printf("...running GPU FFT convolution: ");
    checkCudaErrors(cudaDeviceSynchronize());
    double conv_start = mysecond();
    checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_PaddedData, (cufftComplex *)d_DataSpectrum));
    modulateAndNormalize(d_DataSpectrum, d_KernelSpectrum, fftH, fftW, 1);
    checkCudaErrors(cufftExecC2R(fftPlanInv, (cufftComplex *)d_DataSpectrum, (cufftReal *)d_PaddedData));

    checkCudaErrors(cudaDeviceSynchronize());
    gpu_stop = mysecond();
    double gpuTime = (gpu_stop - conv_start) * 1000.0;
    printf("%f MPix/s (%f ms)\n", (double)dataH * (double)dataW * 1e-6 / (gpuTime * 0.001), gpuTime);

#ifndef CUDA_UM
    printf("...reading back GPU convolution results\n");
    d2h_memcpy_start = mysecond();
    checkCudaErrors(cudaMemcpy(h_ResultGPU, d_PaddedData, fftH * fftW * sizeof(float), cudaMemcpyDeviceToHost));
    d2h_memcpy_stop = mysecond();
#endif

#ifdef CUDA_UM_PREFETCH
    d2h_prefetch_start = mysecond();
    checkCudaErrors(cudaMemPrefetchAsync(h_ResultGPU, fftH * fftW * sizeof(float), cudaCpuDeviceId));
    d2h_prefetch_stop = mysecond();
#endif

    if (validate) {
	cpu_start = mysecond();
#ifdef CUDA_UM_PREFETCH
        checkCudaErrors(cudaMemPrefetchAsync(h_Data, dataH * dataW * sizeof(float), cudaCpuDeviceId));
        checkCudaErrors(cudaMemPrefetchAsync(h_Kernel, kernelH * kernelW * sizeof(float), cudaCpuDeviceId));
#endif
        printf("...running reference CPU convolution\n");
        convolutionClampToBorderCPU(
            h_ResultCPU,
            h_Data,
            h_Kernel,
            dataH,
            dataW,
            kernelH,
            kernelW,
            kernelY,
            kernelX
        );

        printf("...comparing the results: ");
        double sum_delta2 = 0;
        double sum_ref2   = 0;
        double max_delta_ref = 0;

        for (int y = 0; y < dataH; y++)
            for (int x = 0; x < dataW; x++)
            {
                double  rCPU = (double)h_ResultCPU[y * dataW + x];
                double  rGPU = (double)h_ResultGPU[y * fftW  + x];
                double delta = (rCPU - rGPU) * (rCPU - rGPU);
                double   ref = rCPU * rCPU + rCPU * rCPU;

                if ((delta / ref) > max_delta_ref)
                {
                    max_delta_ref = delta / ref;
                }

                sum_delta2 += delta;
                sum_ref2   += ref;
            }

        double L2norm = sqrt(sum_delta2 / sum_ref2);
        printf("rel L2 = %E (max delta = %E)\n", L2norm, sqrt(max_delta_ref));
        bRetVal = (L2norm < 1e-6) ? true : false;
        printf(bRetVal ? "L2norm Error OK\n" : "L2norm Error too high!\n");
	cpu_stop = mysecond();
    }
    else {
#ifdef CUDA_UM
	d2h_memcpy_start = mysecond();
        memcpy(h_ResultCPU, h_ResultGPU, dataH   * dataW * sizeof(float));
	d2h_memcpy_stop = mysecond();
#endif
    }

    compute_migrate_stop = mysecond();

    printf("...shutting down\n");

    cufft_destroy_start = compute_migrate_stop;
    checkCudaErrors(cufftDestroy(fftPlanInv));
    checkCudaErrors(cufftDestroy(fftPlanFwd));
    cufft_destroy_stop = mysecond();

    cuda_free_start = cufft_destroy_stop;
    checkCudaErrors(cudaFree(d_DataSpectrum));
    checkCudaErrors(cudaFree(d_KernelSpectrum));
    checkCudaErrors(cudaFree(d_PaddedData));
    checkCudaErrors(cudaFree(d_PaddedKernel));
    checkCudaErrors(cudaFree(d_Data));
    checkCudaErrors(cudaFree(d_Kernel));
    cuda_free_stop = mysecond();

    free_start = cuda_free_stop;
    free(h_ResultCPU);
#ifndef CUDA_UM
    free(h_ResultGPU);
    free(h_Data);
    free(h_Kernel);
#endif
    free_stop = mysecond();

    application_stop = free_stop;

    printf("\nGPU Time: %f\n", gpu_stop - gpu_start);
    printf("CPU Time: %f\n", cpu_stop - cpu_start);
    printf("malloc timer: %f\n", malloc_stop - malloc_start);
    printf("free timer: %f\n", free_stop - free_start);
    printf("cuda malloc timer: %f\n", cuda_malloc_stop - cuda_malloc_start);
    printf("cuda free timer: %f\n", cuda_free_stop - cuda_free_start);
    printf("Init data timer: %f\n", init_data_stop - init_data_start);
    //printf("misc timer: %f\n", malloc_start - application_start);
#ifdef CUDA_UM_ADVISE
    printf("\nadvise timer: %f\n", (advise_stop - advise_start) + (advise_read_stop - advise_read_start));
#endif
#ifdef CUDA_UM_PREFETCH
    printf("\nH2D async prefetch timer: %f\n", h2d_prefetch_stop - h2d_prefetch_start);
    printf("D2H async prefetch timer: %f\n", d2h_prefetch_stop - d2h_prefetch_start);
#endif
#ifndef CUDA_UM
    printf("\nH2D timer: %f\n", h2d_memcpy_stop - h2d_memcpy_start);
#endif
    printf("D2H timer: %f\n", d2h_memcpy_stop - d2h_memcpy_start);
    printf("\ncufft init timer: %f\n", cufft_init_stop - cufft_init_start);
    printf("cufft destroy timer: %f\n", cufft_destroy_stop - cufft_destroy_start);
    printf("\ncompute migrate timer: %f\n", compute_migrate_stop - compute_migrate_start);
    printf("application timer: %f\n\n", application_stop - application_start);

    sdkDeleteTimer(&hTimer);

    return bRetVal;
}

bool  test1(void)
{
    /////////////////////// START TIMER ///////////////////////////
    application_start = mysecond();

    float
    *h_Data,
    *h_Kernel,
    *h_ResultCPU,
    *h_ResultGPU;

    float
    *d_Data,
    *d_Kernel,
    *d_PaddedData,
    *d_PaddedKernel;

    fComplex
    *d_DataSpectrum0,
    *d_KernelSpectrum0,
    *d_DataSpectrum,
    *d_KernelSpectrum;

    cufftHandle fftPlan;

    bool bRetVal;
    StopWatchInterface *hTimer = NULL;
    sdkCreateTimer(&hTimer);

    printf("Testing custom R2C / C2R FFT-based convolution\n");
    const uint fftPadding = 16;
    const int kernelH = 7;
    const int kernelW = 6;
    const int kernelY = 3;
    const int kernelX = 4;
    const int   dataH = DATA_H;
    const int   dataW = DATA_W;
    const int    fftH = snapTransformSize(dataH + kernelH - 1);
    const int    fftW = snapTransformSize(dataW + kernelW - 1);

    printf("...allocating memory\n");
#ifdef CUDA_UM
    //h_Data      = (float *)malloc(dataH   * dataW * sizeof(float));
    //h_Kernel    = (float *)malloc(kernelH * kernelW * sizeof(float));
    malloc_start = mysecond();
    h_ResultCPU = (float *)malloc(dataH   * dataW * sizeof(float));
    malloc_stop = mysecond();
    //h_ResultGPU = (float *)malloc(fftH    * fftW * sizeof(float));

    cuda_malloc_start = malloc_stop;
    checkCudaErrors(cudaMallocManaged((void **)&d_Data,   dataH   * dataW   * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&d_Kernel, kernelH * kernelW * sizeof(float)));
    h_Data = d_Data; h_Kernel = d_Kernel;

    checkCudaErrors(cudaMallocManaged((void **)&d_PaddedData,   fftH * fftW * sizeof(float)));
    h_ResultGPU = d_PaddedData;

    checkCudaErrors(cudaMallocManaged((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));

    checkCudaErrors(cudaMallocManaged((void **)&d_DataSpectrum0,   fftH * (fftW / 2) * sizeof(fComplex)));

    checkCudaErrors(cudaMallocManaged((void **)&d_KernelSpectrum0, fftH * (fftW / 2) * sizeof(fComplex)));

    checkCudaErrors(cudaMallocManaged((void **)&d_DataSpectrum,    fftH * (fftW / 2 + fftPadding) * sizeof(fComplex)));

    checkCudaErrors(cudaMallocManaged((void **)&d_KernelSpectrum,  fftH * (fftW / 2 + fftPadding) * sizeof(fComplex)));
    cuda_malloc_stop = mysecond();
#else
    malloc_start = mysecond();
    h_Data      = (float *)malloc(dataH   * dataW * sizeof(float));
    h_Kernel    = (float *)malloc(kernelH * kernelW * sizeof(float));
    h_ResultCPU = (float *)malloc(dataH   * dataW * sizeof(float));
    h_ResultGPU = (float *)malloc(fftH    * fftW * sizeof(float));
    malloc_stop = mysecond();

    printf("%f MiB\n", (dataH   * dataW   * sizeof(float)+kernelH * kernelW * sizeof(float)+fftH * fftW * sizeof(float)+fftH * fftW * sizeof(float)+fftH * (fftW / 2) * sizeof(fComplex)+fftH * (fftW / 2) * sizeof(fComplex)+fftH * (fftW / 2 + fftPadding) * sizeof(fComplex)+fftH * (fftW / 2 + fftPadding) * sizeof(fComplex))/1048576.0);
    cuda_malloc_start = mysecond();
    checkCudaErrors(cudaMalloc((void **)&d_Data,   dataH   * dataW   * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Kernel, kernelH * kernelW * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&d_PaddedData,   fftH * fftW * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&d_DataSpectrum0,   fftH * (fftW / 2) * sizeof(fComplex)));
    checkCudaErrors(cudaMalloc((void **)&d_KernelSpectrum0, fftH * (fftW / 2) * sizeof(fComplex)));
    checkCudaErrors(cudaMalloc((void **)&d_DataSpectrum,    fftH * (fftW / 2 + fftPadding) * sizeof(fComplex)));
    checkCudaErrors(cudaMalloc((void **)&d_KernelSpectrum,  fftH * (fftW / 2 + fftPadding) * sizeof(fComplex)));
    cuda_malloc_stop = mysecond();
#endif

#ifdef CUDA_UM_ADVISE
    advise_start = mysecond();;
    cudaMemAdvise(d_PaddedData, sizeof(float) * fftH * fftW, cudaMemAdviseSetPreferredLocation, devID);
    cudaMemAdvise(d_PaddedData, sizeof(float) * fftH * fftW, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
    cudaMemAdvise(d_PaddedKernel, fftH * fftW * sizeof(float), cudaMemAdviseSetPreferredLocation, devID);
    cudaMemAdvise(d_DataSpectrum0, fftH * (fftW / 2) * sizeof(fComplex), cudaMemAdviseSetPreferredLocation, devID);
    cudaMemAdvise(d_KernelSpectrum0, fftH * (fftW / 2) * sizeof(fComplex), cudaMemAdviseSetPreferredLocation, devID);
    cudaMemAdvise(d_DataSpectrum, fftH * (fftW / 2 + fftPadding) * sizeof(fComplex), cudaMemAdviseSetPreferredLocation, devID);
    cudaMemAdvise(d_KernelSpectrum, fftH * (fftW / 2 + fftPadding) * sizeof(fComplex), cudaMemAdviseSetPreferredLocation, devID);
    advise_stop = mysecond();
#endif

    printf("...generating random input data\n");
    srand(2010);

    init_data_start = mysecond();
#ifdef CUDA_UM
    memset(d_PaddedData,   0, fftH * fftW * sizeof(float));
    memset(d_PaddedKernel, 0, fftH * fftW * sizeof(float));
#else
    checkCudaErrors(cudaMemset(d_PaddedData,   0, fftH * fftW * sizeof(float)));
    checkCudaErrors(cudaMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(float)));
#endif

    for (int i = 0; i < dataH * dataW; i++)
    {
        h_Data[i] = getRand();
    }

    for (int i = 0; i < kernelH * kernelW; i++)
    {
        h_Kernel[i] = getRand();
    }
    init_data_stop = mysecond();

#ifdef CUDA_UM_ADVISE
    advise_read_start = init_data_stop;
    cudaMemAdvise(d_Data, dataH   * dataW   * sizeof(float), cudaMemAdviseSetReadMostly, devID);
    cudaMemAdvise(d_Kernel, kernelH   * kernelW   * sizeof(float), cudaMemAdviseSetReadMostly, devID);
    advise_read_stop = mysecond();
#endif

#ifdef CUDA_UM_PREFETCH
    h2d_prefetch_start = mysecond();
    checkCudaErrors(cudaMemPrefetchAsync(d_Data, dataH * dataW * sizeof(float), devID));
    checkCudaErrors(cudaMemPrefetchAsync(d_Kernel, kernelH * kernelW * sizeof(float), devID));
    h2d_prefetch_stop = mysecond();
#endif

    printf("...creating C2C FFT plan for %i x %i\n", fftH, fftW / 2);
    cufft_init_start = mysecond();
    checkCudaErrors(cufftPlan2d(&fftPlan, fftH, fftW / 2, CUFFT_C2C));
    cufft_init_stop = mysecond();

#ifndef CUDA_UM
    printf("...uploading to GPU and padding convolution kernel and input data\n");
    h2d_memcpy_start = mysecond();
    checkCudaErrors(cudaMemcpy(d_Data,   h_Data,   dataH   * dataW *   sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel, kernelH * kernelW * sizeof(float), cudaMemcpyHostToDevice));
    h2d_memcpy_stop = mysecond();
#endif

    compute_migrate_start = mysecond();
    gpu_start = compute_migrate_start;

    padDataClampToBorder(
        d_PaddedData,
        d_Data,
        fftH,
        fftW,
        dataH,
        dataW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
    );

    padKernel(
        d_PaddedKernel,
        d_Kernel,
        fftH,
        fftW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
    );

    //CUFFT_INVERSE works just as well...
    const int FFT_DIR = CUFFT_FORWARD;

    //Not including kernel transformation into time measurement,
    //since convolution kernel is not changed very frequently
    printf("...transforming convolution kernel\n");
    checkCudaErrors(cufftExecC2C(fftPlan, (cufftComplex *)d_PaddedKernel, (cufftComplex *)d_KernelSpectrum0, FFT_DIR));
    spPostprocess2D(d_KernelSpectrum, d_KernelSpectrum0, fftH, fftW / 2, fftPadding, FFT_DIR);

    printf("...running GPU FFT convolution: ");
    checkCudaErrors(cudaDeviceSynchronize());
    double conv_start = mysecond();

    checkCudaErrors(cufftExecC2C(fftPlan, (cufftComplex *)d_PaddedData, (cufftComplex *)d_DataSpectrum0, FFT_DIR));

    spPostprocess2D(d_DataSpectrum, d_DataSpectrum0, fftH, fftW / 2, fftPadding, FFT_DIR);
    modulateAndNormalize(d_DataSpectrum, d_KernelSpectrum, fftH, fftW, fftPadding);
    spPreprocess2D(d_DataSpectrum0, d_DataSpectrum, fftH, fftW / 2, fftPadding, -FFT_DIR);

    checkCudaErrors(cufftExecC2C(fftPlan, (cufftComplex *)d_DataSpectrum0, (cufftComplex *)d_PaddedData, -FFT_DIR));

    checkCudaErrors(cudaDeviceSynchronize());
    gpu_stop = mysecond();
    double gpuTime = (gpu_stop - conv_start) * 1000.0;
    printf("%f MPix/s (%f ms)\n", (double)dataH * (double)dataW * 1e-6 / (gpuTime * 0.001), gpuTime);

    printf("...reading back GPU FFT results\n");
#ifndef CUDA_UM
    d2h_memcpy_start = mysecond();
    checkCudaErrors(cudaMemcpy(h_ResultGPU, d_PaddedData, fftH * fftW * sizeof(float), cudaMemcpyDeviceToHost));
    d2h_memcpy_stop = mysecond();
#endif
#ifdef CUDA_UM_PREFETCH
    d2h_prefetch_start = mysecond();
    checkCudaErrors(cudaMemPrefetchAsync(h_ResultGPU, fftH * fftW * sizeof(float), cudaCpuDeviceId));
    d2h_prefetch_stop = mysecond();
#endif

    if (validate) {
	cpu_start = mysecond();
#ifdef CUDA_UM_PREFETCH
        checkCudaErrors(cudaMemPrefetchAsync(h_Data, dataH * dataW * sizeof(float), cudaCpuDeviceId));
        checkCudaErrors(cudaMemPrefetchAsync(h_Kernel, kernelH * kernelW * sizeof(float), cudaCpuDeviceId));
#endif

        printf("...running reference CPU convolution\n");
        convolutionClampToBorderCPU(
            h_ResultCPU,
            h_Data,
            h_Kernel,
            dataH,
            dataW,
            kernelH,
            kernelW,
            kernelY,
            kernelX
        );

        printf("...comparing the results: ");
        double sum_delta2 = 0;
        double sum_ref2   = 0;
        double max_delta_ref = 0;

        for (int y = 0; y < dataH; y++)
            for (int x = 0; x < dataW; x++)
            {
                double  rCPU = (double)h_ResultCPU[y * dataW + x];
                double  rGPU = (double)h_ResultGPU[y * fftW  + x];
                double delta = (rCPU - rGPU) * (rCPU - rGPU);
                double   ref = rCPU * rCPU + rCPU * rCPU;

                if ((delta / ref) > max_delta_ref)
                {
                    max_delta_ref = delta / ref;
                }

                sum_delta2 += delta;
                sum_ref2   += ref;
            }

        double L2norm = sqrt(sum_delta2 / sum_ref2);
        printf("rel L2 = %E (max delta = %E)\n", L2norm, sqrt(max_delta_ref));
        bRetVal = (L2norm < 1e-6) ? true : false;
        printf(bRetVal ? "L2norm Error OK\n" : "L2norm Error too high!\n");
	cpu_stop = mysecond();
    }
    else {
#ifdef CUDA_UM
	d2h_memcpy_start = mysecond();
        memcpy(h_ResultCPU, h_ResultGPU, dataH   * dataW * sizeof(float));
	d2h_memcpy_stop = mysecond();
#endif
    }

    compute_migrate_stop = mysecond();

    printf("...shutting down\n");
    cufft_destroy_start = compute_migrate_stop;
    checkCudaErrors(cufftDestroy(fftPlan));
    cufft_destroy_stop = compute_migrate_stop;

    cuda_free_start = cufft_destroy_stop;
    checkCudaErrors(cudaFree(d_KernelSpectrum));
    checkCudaErrors(cudaFree(d_DataSpectrum));
    checkCudaErrors(cudaFree(d_KernelSpectrum0));
    checkCudaErrors(cudaFree(d_DataSpectrum0));
    checkCudaErrors(cudaFree(d_PaddedKernel));
    checkCudaErrors(cudaFree(d_PaddedData));
    checkCudaErrors(cudaFree(d_Kernel));
    checkCudaErrors(cudaFree(d_Data));
    cuda_free_stop = mysecond();

#ifdef CUDA_UM
    free_start = mysecond();
    free(h_ResultCPU);
    free_stop = mysecond();
#else
    free_start = mysecond();
    free(h_ResultGPU);
    free(h_ResultCPU);
    free(h_Kernel);
    free(h_Data);
    free_stop = mysecond();
#endif

    application_stop = mysecond();

    printf("\nGPU Time: %f\n", gpu_stop - gpu_start);
    printf("CPU Time: %f\n", cpu_stop - cpu_start);
    printf("malloc timer: %f\n", malloc_stop - malloc_start);
    printf("free timer: %f\n", free_stop - free_start);
    printf("cuda malloc timer: %f\n", cuda_malloc_stop - cuda_malloc_start);
    printf("cuda free timer: %f\n", cuda_free_stop - cuda_free_start);
    printf("Init data timer: %f\n", init_data_stop - init_data_start);
    //printf("misc timer: %f\n", malloc_start - application_start);
#ifdef CUDA_UM_ADVISE
    printf("\nadvise timer: %f\n", (advise_stop - advise_start) + (advise_read_stop - advise_read_start));
#endif
#ifdef CUDA_UM_PREFETCH
    printf("H2D async prefetch timer: %f\n", h2d_prefetch_stop - h2d_prefetch_start);
    printf("D2H async prefetch timer: %f\n", d2h_prefetch_stop - d2h_prefetch_start);
#endif
#ifndef CUDA_UM
    printf("\nH2D timer: %f\n", h2d_memcpy_stop - h2d_memcpy_start);
#endif
    printf("D2H timer: %f\n", d2h_memcpy_stop - d2h_memcpy_start);
    printf("\ncufft init timer: %f\n", cufft_init_stop - cufft_init_start);
    printf("cufft destroy timer: %f\n", cufft_destroy_stop - cufft_destroy_start);
    printf("\ncompute migrate timer: %f\n", compute_migrate_stop - compute_migrate_start);
    printf("application timer: %f\n\n", application_stop - application_start);

    sdkDeleteTimer(&hTimer);

    return bRetVal;
}

bool test2(void)
{
    /////////////////////// START TIMER ///////////////////////////
    application_start = mysecond();

    float
    *h_Data,
    *h_Kernel,
    *h_ResultCPU,
    *h_ResultGPU;

    float
    *d_Data,
    *d_Kernel,
    *d_PaddedData,
    *d_PaddedKernel;

    fComplex
    *d_DataSpectrum0,
    *d_KernelSpectrum0;

    cufftHandle
    fftPlan;

    bool bRetVal;
    StopWatchInterface *hTimer = NULL;
    sdkCreateTimer(&hTimer);

    printf("Testing updated custom R2C / C2R FFT-based convolution\n");
    const int kernelH = 7;
    const int kernelW = 6;
    const int kernelY = 3;
    const int kernelX = 4;
    const int dataH = DATA_H;
    const int dataW = DATA_W;
    const int fftH = snapTransformSize(dataH + kernelH - 1);
    const int fftW = snapTransformSize(dataW + kernelW - 1);

    printf("...allocating memory\n");
#ifdef CUDA_UM
    //h_Data      = (float *)malloc(dataH   * dataW * sizeof(float));
    //h_Kernel    = (float *)malloc(kernelH * kernelW * sizeof(float));
    malloc_start = mysecond();
    h_ResultCPU = (float *)malloc(dataH   * dataW * sizeof(float));
    malloc_stop = mysecond();
    //h_ResultGPU = (float *)malloc(fftH    * fftW * sizeof(float));

    cuda_malloc_start = malloc_stop;
    checkCudaErrors(cudaMallocManaged((void **)&d_Data,   dataH   * dataW   * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&d_Kernel, kernelH * kernelW * sizeof(float)));
    h_Data = d_Data; h_Kernel = d_Kernel;

    checkCudaErrors(cudaMallocManaged((void **)&d_PaddedData,   fftH * fftW * sizeof(float)));
    h_ResultGPU = d_PaddedData;

    checkCudaErrors(cudaMallocManaged((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));

    checkCudaErrors(cudaMallocManaged((void **)&d_DataSpectrum0,   fftH * (fftW / 2) * sizeof(fComplex)));

    checkCudaErrors(cudaMallocManaged((void **)&d_KernelSpectrum0, fftH * (fftW / 2) * sizeof(fComplex)));
    cuda_malloc_stop = mysecond();
#else
    malloc_start = mysecond();
    h_Data      = (float *)malloc(dataH   * dataW * sizeof(float));
    h_Kernel    = (float *)malloc(kernelH * kernelW * sizeof(float));
    h_ResultCPU = (float *)malloc(dataH   * dataW * sizeof(float));
    h_ResultGPU = (float *)malloc(fftH    * fftW * sizeof(float));
    malloc_stop = mysecond();

    printf("%f MiB\n", (dataH   * dataW   * sizeof(float)+kernelH * kernelW * sizeof(float)+fftH * fftW * sizeof(float)+fftH * fftW * sizeof(float)+fftH * (fftW / 2) * sizeof(fComplex)+fftH * (fftW / 2) * sizeof(fComplex))/1048576.0);
    cuda_malloc_start = mysecond();
    checkCudaErrors(cudaMalloc((void **)&d_Data,   dataH   * dataW   * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Kernel, kernelH * kernelW * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&d_PaddedData,   fftH * fftW * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&d_DataSpectrum0,   fftH * (fftW / 2) * sizeof(fComplex)));
    checkCudaErrors(cudaMalloc((void **)&d_KernelSpectrum0, fftH * (fftW / 2) * sizeof(fComplex)));
    cuda_malloc_stop = mysecond();
#endif

#ifdef CUDA_UM_ADVISE
    advise_start = cuda_malloc_stop;
    cudaMemAdvise(d_PaddedData, fftH * fftW * sizeof(float), cudaMemAdviseSetPreferredLocation, devID);
    cudaMemAdvise(d_PaddedData, fftH * fftW * sizeof(float), cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
    cudaMemAdvise(d_PaddedKernel, fftH * fftW * sizeof(float), cudaMemAdviseSetPreferredLocation, devID);
    cudaMemAdvise(d_DataSpectrum0, fftH * (fftW / 2) * sizeof(fComplex), cudaMemAdviseSetPreferredLocation, devID);
    cudaMemAdvise(d_KernelSpectrum0, fftH * (fftW / 2) * sizeof(fComplex), cudaMemAdviseSetPreferredLocation, devID);
    advise_stop = mysecond();
#endif

    printf("...generating random input data\n");
    srand(2010);

    init_data_start = mysecond();
#ifdef CUDA_UM
    memset(d_PaddedData,   0, fftH * fftW * sizeof(float));
    memset(d_PaddedKernel, 0, fftH * fftW * sizeof(float));
#else
    checkCudaErrors(cudaMemset(d_PaddedData,   0, fftH * fftW * sizeof(float)));
    checkCudaErrors(cudaMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(float)));
#endif

    for (int i = 0; i < dataH * dataW; i++)
    {
        h_Data[i] = getRand();
    }

    for (int i = 0; i < kernelH * kernelW; i++)
    {
        h_Kernel[i] = getRand();
    }
    init_data_stop = mysecond();

#ifdef CUDA_UM_ADVISE
    advise_read_start = init_data_stop;
    cudaMemAdvise(h_Data, dataH * dataW * sizeof(float), cudaMemAdviseSetReadMostly, devID);
    cudaMemAdvise(h_Kernel, kernelH * kernelW * sizeof(float), cudaMemAdviseSetReadMostly, devID);
    advise_read_stop = mysecond();
#endif

#ifdef CUDA_UM_PREFETCH
    h2d_prefetch_start = mysecond();
    checkCudaErrors(cudaMemPrefetchAsync(d_Data, dataH * dataW * sizeof(float), devID));
    checkCudaErrors(cudaMemPrefetchAsync(d_Kernel, kernelH * kernelW * sizeof(float), devID));
    h2d_prefetch_stop = mysecond();
#endif

    printf("...creating C2C FFT plan for %i x %i\n", fftH, fftW / 2);
    cufft_init_start = mysecond();
    checkCudaErrors(cufftPlan2d(&fftPlan, fftH, fftW / 2, CUFFT_C2C));
    cufft_init_stop = mysecond();

    printf("...uploading to GPU and padding convolution kernel and input data\n");
#ifndef CUDA_UM
    checkCudaErrors(cudaMemcpy(d_Data,   h_Data,   dataH   * dataW *   sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel, kernelH * kernelW * sizeof(float), cudaMemcpyHostToDevice));
#endif

    gpu_start = mysecond();
    compute_migrate_start = gpu_start;

    padDataClampToBorder(
        d_PaddedData,
        d_Data,
        fftH,
        fftW,
        dataH,
        dataW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
    );

    padKernel(
        d_PaddedKernel,
        d_Kernel,
        fftH,
        fftW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
    );

    //CUFFT_INVERSE works just as well...
    const int FFT_DIR = CUFFT_FORWARD;

    //Not including kernel transformation into time measurement,
    //since convolution kernel is not changed very frequently
    printf("...transforming convolution kernel\n");
    checkCudaErrors(cufftExecC2C(fftPlan, (cufftComplex *)d_PaddedKernel, (cufftComplex *)d_KernelSpectrum0, FFT_DIR));

    printf("...running GPU FFT convolution: ");
    checkCudaErrors(cudaDeviceSynchronize());
    double conv_start = mysecond();

    checkCudaErrors(cufftExecC2C(fftPlan, (cufftComplex *)d_PaddedData, (cufftComplex *)d_DataSpectrum0, FFT_DIR));
    spProcess2D(d_DataSpectrum0, d_DataSpectrum0, d_KernelSpectrum0, fftH, fftW / 2, FFT_DIR);
    checkCudaErrors(cufftExecC2C(fftPlan, (cufftComplex *)d_DataSpectrum0, (cufftComplex *)d_PaddedData, -FFT_DIR));

    checkCudaErrors(cudaDeviceSynchronize());
    gpu_stop = mysecond();
    double gpuTime = (gpu_stop - conv_start) * 1000.0;
    printf("%f MPix/s (%f ms)\n", (double)dataH * (double)dataW * 1e-6 / (gpuTime * 0.001), gpuTime);

    printf("...reading back GPU FFT results\n");
#ifndef CUDA_UM
    d2h_memcpy_start = mysecond();
    checkCudaErrors(cudaMemcpy(h_ResultGPU, d_PaddedData, fftH * fftW * sizeof(float), cudaMemcpyDeviceToHost));
    d2h_memcpy_stop = mysecond();
#endif

#ifdef CUDA_UM_PREFETCH
    d2h_prefetch_start = mysecond();
    checkCudaErrors(cudaMemPrefetchAsync(h_ResultGPU, fftH * fftW * sizeof(float), cudaCpuDeviceId));
    d2h_prefetch_stop = mysecond();
#endif

    if (validate) {
	cpu_start = mysecond();
#ifdef CUDA_UM_PREFETCH
        checkCudaErrors(cudaMemPrefetchAsync(h_Data, dataH * dataW * sizeof(float), cudaCpuDeviceId));
        checkCudaErrors(cudaMemPrefetchAsync(h_Kernel, kernelH * kernelW * sizeof(float), cudaCpuDeviceId));
#endif

        printf("...running reference CPU convolution\n");
        convolutionClampToBorderCPU(
            h_ResultCPU,
            h_Data,
            h_Kernel,
            dataH,
            dataW,
            kernelH,
            kernelW,
            kernelY,
            kernelX
        );

        printf("...comparing the results: ");
        double sum_delta2 = 0;
        double sum_ref2   = 0;
        double max_delta_ref = 0;

        for (int y = 0; y < dataH; y++)
        {
            for (int x = 0; x < dataW; x++)
            {
                double  rCPU = (double)h_ResultCPU[y * dataW + x];
                double  rGPU = (double)h_ResultGPU[y * fftW  + x];
                double delta = (rCPU - rGPU) * (rCPU - rGPU);
                double   ref = rCPU * rCPU + rCPU * rCPU;

                if ((delta / ref) > max_delta_ref)
                {
                    max_delta_ref = delta / ref;
                }

                sum_delta2 += delta;
                sum_ref2   += ref;
            }
        }

        double L2norm = sqrt(sum_delta2 / sum_ref2);
        printf("rel L2 = %E (max delta = %E)\n", L2norm, sqrt(max_delta_ref));
        bRetVal = (L2norm < 1e-6) ? true : false;
        printf(bRetVal ? "L2norm Error OK\n" : "L2norm Error too high!\n");
	cpu_stop = mysecond();
    }
    else {
#ifdef CUDA_UM
	d2h_memcpy_start = mysecond();
        memcpy(h_ResultCPU, h_ResultGPU, dataH   * dataW * sizeof(float));
	d2h_memcpy_stop = mysecond();
#endif
    }

    compute_migrate_stop = mysecond();

    printf("...shutting down\n");
    cufft_destroy_start = compute_migrate_stop;
    checkCudaErrors(cufftDestroy(fftPlan));
    cufft_destroy_stop = mysecond();

    cuda_free_start = cufft_destroy_stop;
    checkCudaErrors(cudaFree(d_KernelSpectrum0));
    checkCudaErrors(cudaFree(d_DataSpectrum0));
    checkCudaErrors(cudaFree(d_PaddedKernel));
    checkCudaErrors(cudaFree(d_PaddedData));
    checkCudaErrors(cudaFree(d_Kernel));
    checkCudaErrors(cudaFree(d_Data));
    cuda_free_stop = mysecond();

#ifdef CUDA_UM
    free_start = cuda_free_stop;
    free(h_ResultCPU);
    free_stop = mysecond();
#else
    free_start = mysecond();
    free(h_ResultGPU);
    free(h_ResultCPU);
    free(h_Kernel);
    free(h_Data);
    free_stop = mysecond();
#endif

    application_stop = mysecond();

    printf("\nGPU Time: %f\n", gpu_stop - gpu_start);
    printf("CPU Time: %f\n", cpu_stop - cpu_start);
    printf("malloc timer: %f\n", malloc_stop - malloc_start);
    printf("free timer: %f\n", free_stop - free_start);
    printf("cuda malloc timer: %f\n", cuda_malloc_stop - cuda_malloc_start);
    printf("cuda free timer: %f\n", cuda_free_stop - cuda_free_start);
    printf("Init data timer: %f\n", init_data_stop - init_data_start);
    //printf("misc timer: %f\n", malloc_start - application_start);
#ifdef CUDA_UM_ADVISE
    printf("\nadvise timer: %f\n", (advise_stop - advise_start) + (advise_read_stop - advise_read_start));
#endif
#ifdef CUDA_UM_PREFETCH
    printf("H2D async prefetch timer: %f\n", h2d_prefetch_stop - h2d_prefetch_start);
    printf("D2H async prefetch timer: %f\n", d2h_prefetch_stop - d2h_prefetch_start);
#endif
#ifndef CUDA_UM
    printf("\nH2D timer: %f\n", h2d_memcpy_stop - h2d_memcpy_start);
#endif
    printf("D2H timer: %f\n", d2h_memcpy_stop - d2h_memcpy_start);
    printf("\ncufft init timer: %f\n", cufft_init_stop - cufft_init_start);
    printf("cufft destroy timer: %f\n", cufft_destroy_stop - cufft_destroy_start);
    printf("\ncompute migrate timer: %f\n", compute_migrate_stop - compute_migrate_start);
    printf("application timer: %f\n\n", application_stop - application_start);

    sdkDeleteTimer(&hTimer);

    return bRetVal;
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


    if (checkCmdLineFlag(argc, (const char **)argv, "size")) {
        DATA_H = DATA_W = getCmdLineArgumentInt(argc, (const char **)argv, "size");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "fft0Iter")) {
        fft0Iter = getCmdLineArgumentInt(argc, (const char **)argv, "fft0Iter");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "fft1Iter")) {
        fft1Iter = getCmdLineArgumentInt(argc, (const char **)argv, "fft1Iter");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "fft2Iter")) {
        fft2Iter = getCmdLineArgumentInt(argc, (const char **)argv, "fft2Iter");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "validate")) {
        validate = true;
    }

    printf("[%s] - Starting...\n", argv[0]);

    //Use command-line specified CUDA device, otherwise use device with highest Gflops/s
    devID = findCudaDevice(argc, (const char **)argv);

    int nFailures = 0;

    StopWatchInterface *hTimer = NULL;
    sdkCreateTimer(&hTimer);
    double elapsedTime;

    for (int i = 0; i < fft0Iter; i++) {
        sdkStartTimer(&hTimer);
        if (!test0())
        {
            nFailures++;
        }
        sdkStopTimer(&hTimer);
        elapsedTime = sdkGetTimerValue(&hTimer);
        sdkResetTimer(&hTimer);
        printf("0: %d: %f\n", i, elapsedTime);
    }

    for (int i = 0; i < fft1Iter; i++) {
        sdkStartTimer(&hTimer);
        if (!test1())
        {
            nFailures++;
        }
        sdkStopTimer(&hTimer);
        elapsedTime = sdkGetTimerValue(&hTimer);
        sdkResetTimer(&hTimer);
        printf("1: %d: %f\n", i, elapsedTime);
    }

    for (int i = 0; i < fft2Iter; i++) {
        sdkStartTimer(&hTimer);
        if (!test2())
        {
            nFailures++;
        }
        sdkStopTimer(&hTimer);
        elapsedTime = sdkGetTimerValue(&hTimer);
        sdkResetTimer(&hTimer);
        printf("2: %d: %f\n", i, elapsedTime);
    }

    printf("Test Summary: %d errors\n", nFailures);

    if (nFailures > 0)
    {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}