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

//#define DATA_H 8000
//#define DATA_W 8000
long unsigned int DATA_H = 8000;
long unsigned int DATA_W = 8000;
bool validate = false;
int nIter = 5;

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
//    h_Data      = (float *)malloc(dataH   * dataW * sizeof(float));
//    h_Kernel    = (float *)malloc(kernelH * kernelW * sizeof(float));
    h_ResultCPU = (float *)malloc(dataH   * dataW * sizeof(float));
//    h_ResultGPU = (float *)malloc(fftH    * fftW * sizeof(float));

    checkCudaErrors(cudaMallocManaged((void **)&d_Data,   dataH   * dataW   * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&d_Kernel, kernelH * kernelW * sizeof(float)));
    h_Data = d_Data; h_Kernel = d_Kernel;

    checkCudaErrors(cudaMallocManaged((void **)&d_PaddedData,   fftH * fftW * sizeof(float)));
    h_ResultGPU = d_PaddedData;

    checkCudaErrors(cudaMallocManaged((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));

    checkCudaErrors(cudaMallocManaged((void **)&d_DataSpectrum,   fftH * (fftW / 2 + 1) * sizeof(fComplex)));

    checkCudaErrors(cudaMallocManaged((void **)&d_KernelSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));

    memset(d_KernelSpectrum, 0, fftH * (fftW / 2 + 1) * sizeof(fComplex));

    printf("...generating random input data\n");
    srand(2010);

    for (int i = 0; i < dataH * dataW; i++)
    {
        h_Data[i] = getRand();
    }

    for (int i = 0; i < kernelH * kernelW; i++)
    {
        h_Kernel[i] = getRand();
    }

    printf("...creating R2C & C2R FFT plans for %i x %i\n", fftH, fftW);
    checkCudaErrors(cufftPlan2d(&fftPlanFwd, fftH, fftW, CUFFT_R2C));
    checkCudaErrors(cufftPlan2d(&fftPlanInv, fftH, fftW, CUFFT_C2R));

    printf("...uploading to GPU and padding convolution kernel and input data\n");
    //checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel, kernelH * kernelW * sizeof(float), cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemcpy(d_Data,   h_Data,   dataH   * dataW *   sizeof(float), cudaMemcpyHostToDevice));
    memset(d_PaddedKernel, 0, fftH * fftW * sizeof(float));
    memset(d_PaddedData,   0, fftH * fftW * sizeof(float));

    double migrate_compute_start = mysecond();

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
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_PaddedData, (cufftComplex *)d_DataSpectrum));
    modulateAndNormalize(d_DataSpectrum, d_KernelSpectrum, fftH, fftW, 1);
    checkCudaErrors(cufftExecC2R(fftPlanInv, (cufftComplex *)d_DataSpectrum, (cufftReal *)d_PaddedData));

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    double gpuTime = sdkGetTimerValue(&hTimer);
    printf("%f MPix/s (%f ms)\n", (double)dataH * (double)dataW * 1e-6 / (gpuTime * 0.001), gpuTime);

    printf("...reading back GPU convolution results\n");
    //checkCudaErrors(cudaMemcpy(h_ResultGPU, d_PaddedData, fftH * fftW * sizeof(float), cudaMemcpyDeviceToHost));

    if (validate) {
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
    }
    else {
        memcpy(h_ResultCPU, h_ResultGPU, dataH   * dataW * sizeof(float));
    }

    double migrate_compute_time = mysecond() - migrate_compute_start;
    printf("0: migrate compute: %f\n", migrate_compute_time);

    printf("...shutting down\n");
    sdkDeleteTimer(&hTimer);

    checkCudaErrors(cufftDestroy(fftPlanInv));
    checkCudaErrors(cufftDestroy(fftPlanFwd));

    checkCudaErrors(cudaFree(d_DataSpectrum));
    checkCudaErrors(cudaFree(d_KernelSpectrum));
    checkCudaErrors(cudaFree(d_PaddedData));
    checkCudaErrors(cudaFree(d_PaddedKernel));
    checkCudaErrors(cudaFree(d_Data));
    checkCudaErrors(cudaFree(d_Kernel));

    //free(h_ResultGPU);
    free(h_ResultCPU);
    //free(h_Data);
    //free(h_Kernel);

    return bRetVal;
}

bool  test1(void)
{
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
    //h_Data      = (float *)malloc(dataH   * dataW * sizeof(float));
    //h_Kernel    = (float *)malloc(kernelH * kernelW * sizeof(float));
    h_ResultCPU = (float *)malloc(dataH   * dataW * sizeof(float));
    //h_ResultGPU = (float *)malloc(fftH    * fftW * sizeof(float));

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

    printf("...generating random input data\n");
    srand(2010);

    for (int i = 0; i < dataH * dataW; i++)
    {
        h_Data[i] = getRand();
    }

    for (int i = 0; i < kernelH * kernelW; i++)
    {
        h_Kernel[i] = getRand();
    }

    printf("...creating C2C FFT plan for %i x %i\n", fftH, fftW / 2);
    checkCudaErrors(cufftPlan2d(&fftPlan, fftH, fftW / 2, CUFFT_C2C));

    printf("...uploading to GPU and padding convolution kernel and input data\n");
    //checkCudaErrors(cudaMemcpy(d_Data,   h_Data,   dataH   * dataW *   sizeof(float), cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel, kernelH * kernelW * sizeof(float), cudaMemcpyHostToDevice));
    memset(d_PaddedData,   0, fftH * fftW * sizeof(float));
    memset(d_PaddedKernel, 0, fftH * fftW * sizeof(float));

    double migrate_compute_start = mysecond();

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
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    checkCudaErrors(cufftExecC2C(fftPlan, (cufftComplex *)d_PaddedData, (cufftComplex *)d_DataSpectrum0, FFT_DIR));

    spPostprocess2D(d_DataSpectrum, d_DataSpectrum0, fftH, fftW / 2, fftPadding, FFT_DIR);
    modulateAndNormalize(d_DataSpectrum, d_KernelSpectrum, fftH, fftW, fftPadding);
    spPreprocess2D(d_DataSpectrum0, d_DataSpectrum, fftH, fftW / 2, fftPadding, -FFT_DIR);

    checkCudaErrors(cufftExecC2C(fftPlan, (cufftComplex *)d_DataSpectrum0, (cufftComplex *)d_PaddedData, -FFT_DIR));

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    double gpuTime = sdkGetTimerValue(&hTimer);
    printf("%f MPix/s (%f ms)\n", (double)dataH * (double)dataW * 1e-6 / (gpuTime * 0.001), gpuTime);

    printf("...reading back GPU FFT results\n");
    //checkCudaErrors(cudaMemcpy(h_ResultGPU, d_PaddedData, fftH * fftW * sizeof(float), cudaMemcpyDeviceToHost));

    if (validate) {
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
    }
    else {
        memcpy(h_ResultCPU, h_ResultGPU, dataH   * dataW * sizeof(float));
    }

    double migrate_compute_time = mysecond() - migrate_compute_start;
    printf("1: migrate compute: %f\n", migrate_compute_time);

    printf("...shutting down\n");
    sdkDeleteTimer(&hTimer);
    checkCudaErrors(cufftDestroy(fftPlan));

    checkCudaErrors(cudaFree(d_KernelSpectrum));
    checkCudaErrors(cudaFree(d_DataSpectrum));
    checkCudaErrors(cudaFree(d_KernelSpectrum0));
    checkCudaErrors(cudaFree(d_DataSpectrum0));
    checkCudaErrors(cudaFree(d_PaddedKernel));
    checkCudaErrors(cudaFree(d_PaddedData));
    checkCudaErrors(cudaFree(d_Kernel));
    checkCudaErrors(cudaFree(d_Data));

    //free(h_ResultGPU);
    free(h_ResultCPU);
    //free(h_Kernel);
    //free(h_Data);

    return bRetVal;
}

bool test2(void)
{
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
    //h_Data      = (float *)malloc(dataH   * dataW * sizeof(float));
    //h_Kernel    = (float *)malloc(kernelH * kernelW * sizeof(float));
    h_ResultCPU = (float *)malloc(dataH   * dataW * sizeof(float));
    //h_ResultGPU = (float *)malloc(fftH    * fftW * sizeof(float));

    checkCudaErrors(cudaMallocManaged((void **)&d_Data,   dataH   * dataW   * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&d_Kernel, kernelH * kernelW * sizeof(float)));
    h_Data = d_Data; h_Kernel = d_Kernel;

    checkCudaErrors(cudaMallocManaged((void **)&d_PaddedData,   fftH * fftW * sizeof(float)));
    h_ResultGPU = d_PaddedData;

    checkCudaErrors(cudaMallocManaged((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));

    checkCudaErrors(cudaMallocManaged((void **)&d_DataSpectrum0,   fftH * (fftW / 2) * sizeof(fComplex)));

    checkCudaErrors(cudaMallocManaged((void **)&d_KernelSpectrum0, fftH * (fftW / 2) * sizeof(fComplex)));

    printf("...generating random input data\n");
    srand(2010);

    for (int i = 0; i < dataH * dataW; i++)
    {
        h_Data[i] = getRand();
    }

    for (int i = 0; i < kernelH * kernelW; i++)
    {
        h_Kernel[i] = getRand();
    }

    printf("...creating C2C FFT plan for %i x %i\n", fftH, fftW / 2);
    checkCudaErrors(cufftPlan2d(&fftPlan, fftH, fftW / 2, CUFFT_C2C));

    printf("...uploading to GPU and padding convolution kernel and input data\n");
    //checkCudaErrors(cudaMemcpy(d_Data,   h_Data,   dataH   * dataW *   sizeof(float), cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel, kernelH * kernelW * sizeof(float), cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemset(d_PaddedData,   0, fftH * fftW * sizeof(float)));
    //checkCudaErrors(cudaMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(float)));
    memset(d_PaddedData,   0, fftH * fftW * sizeof(float));
    memset(d_PaddedKernel, 0, fftH * fftW * sizeof(float));

    double migrate_compute_start = mysecond();

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
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    checkCudaErrors(cufftExecC2C(fftPlan, (cufftComplex *)d_PaddedData, (cufftComplex *)d_DataSpectrum0, FFT_DIR));
    spProcess2D(d_DataSpectrum0, d_DataSpectrum0, d_KernelSpectrum0, fftH, fftW / 2, FFT_DIR);
    checkCudaErrors(cufftExecC2C(fftPlan, (cufftComplex *)d_DataSpectrum0, (cufftComplex *)d_PaddedData, -FFT_DIR));

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    double gpuTime = sdkGetTimerValue(&hTimer);
    printf("%f MPix/s (%f ms)\n", (double)dataH * (double)dataW * 1e-6 / (gpuTime * 0.001), gpuTime);

    printf("...reading back GPU FFT results\n");
    //checkCudaErrors(cudaMemcpy(h_ResultGPU, d_PaddedData, fftH * fftW * sizeof(float), cudaMemcpyDeviceToHost));

    if (validate) {
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
    }
    else {
        memcpy(h_ResultCPU, h_ResultGPU, dataH   * dataW * sizeof(float));
    }

    double migrate_compute_time = mysecond() - migrate_compute_start;
    printf("2: migrate compute: %f\n", migrate_compute_time);

    printf("...shutting down\n");
    sdkDeleteTimer(&hTimer);
    checkCudaErrors(cufftDestroy(fftPlan));

    checkCudaErrors(cudaFree(d_KernelSpectrum0));
    checkCudaErrors(cudaFree(d_DataSpectrum0));
    checkCudaErrors(cudaFree(d_PaddedKernel));
    checkCudaErrors(cudaFree(d_PaddedData));
    checkCudaErrors(cudaFree(d_Kernel));
    checkCudaErrors(cudaFree(d_Data));

    //free(h_ResultGPU);
    free(h_ResultCPU);
    //free(h_Kernel);
    //free(h_Data);

    return bRetVal;
}



int main(int argc, char **argv)
{
    if (checkCmdLineFlag(argc, (const char **)argv, "size")) {
        DATA_H = DATA_W = getCmdLineArgumentInt(argc, (const char **)argv, "size");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "iterations")) {
        nIter = getCmdLineArgumentInt(argc, (const char **)argv, "iterations");
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

    for (int i = 0; i < nIter; i++) {
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

    for (int i = 0; i < nIter; i++) {
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

    for (int i = 0; i < nIter; i++) {
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
