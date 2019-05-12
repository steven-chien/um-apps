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

#include "FDTD3d.h"

#include <iostream>
#include <iomanip>

#include "FDTD3dReference.h"
#include "FDTD3dGPU.h"

#include <helper_functions.h>

#include <math.h>
#include <assert.h>

#include <cuda_runtime.h>

#ifndef CLAMP
#define CLAMP(a, min, max) ( MIN(max, MAX(a, min)) )
#endif

bool validate = false;

//// Name of the log file
//const char *printfFile = "FDTD3d.txt";

// Forward declarations
bool runTest(int argc, const char **argv);
void showHelp(const int argc, const char **argv);

int main(int argc, char **argv)
{
    bool bTestResult = false;
    // Start the log
    printf("%s Starting...\n\n", argv[0]);

    // Check help flag
    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        printf("Displaying help on console\n");
        showHelp(argc, (const char **)argv);
        bTestResult = true;
    }
    else
    {
        struct timeval start, stop;
        struct timezone tzp;
        int i;
        i = gettimeofday(&start,&tzp);

        // Execute
        bTestResult = runTest(argc, (const char **)argv);
        i = gettimeofday(&stop,&tzp);

        double elapsedTime = ( (double) stop.tv_sec + (double) stop.tv_usec * 1.e-6 ) - ( (double) start.tv_sec + (double) start.tv_usec * 1.e-6 );
        printf("Time: %f\n", elapsedTime);
    }

    // Finish
    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

void showHelp(const int argc, const char **argv)
{
    if (argc > 0)
        std::cout << std::endl << argv[0] << std::endl;

    std::cout << std::endl << "Syntax:" << std::endl;
    std::cout << std::left;
    std::cout << "    " << std::setw(20) << "--device=<device>" << "Specify device to use for execution" << std::endl;
    std::cout << "    " << std::setw(20) << "--dimx=<N>" << "Specify number of elements in x direction (excluding halo)" << std::endl;
    std::cout << "    " << std::setw(20) << "--dimy=<N>" << "Specify number of elements in y direction (excluding halo)" << std::endl;
    std::cout << "    " << std::setw(20) << "--dimz=<N>" << "Specify number of elements in z direction (excluding halo)" << std::endl;
    std::cout << "    " << std::setw(20) << "--radius=<N>" << "Specify radius of stencil" << std::endl;
    std::cout << "    " << std::setw(20) << "--timesteps=<N>" << "Specify number of timesteps" << std::endl;
    std::cout << "    " << std::setw(20) << "--block-size=<N>" << "Specify number of threads per block" << std::endl;
    std::cout << std::endl;
    std::cout << "    " << std::setw(20) << "--noprompt" << "Skip prompt before exit" << std::endl;
    std::cout << std::endl;
}

bool runTest(int argc, const char **argv)
{
    float *host_output;
    float *device_output;
    float *input;
    float *coeff;

    long unsigned int defaultDim;
    long unsigned int dimx;
    long unsigned int dimy;
    long unsigned int dimz;
    long unsigned int outerDimx;
    long unsigned int outerDimy;
    long unsigned int outerDimz;
    long unsigned int radius;
    long unsigned int timesteps;
    //size_t volumeSize;
    long unsigned int volumeSize;
    memsize_t memsize;
    int devID = 0;

    const float lowerBound = 0.0f;
    const float upperBound = 1.0f;

    // Determine default dimensions
    printf("Set-up, based upon target device GMEM size...\n");
    // Get the memory size of the target device
    printf(" getTargetDeviceGlobalMemSize\n");
    getTargetDeviceGlobalMemSize(&memsize, argc, argv);

    // We can never use all the memory so to keep things simple we aim to
    // use around half the total memory
    memsize /= 2;

    // Most of our memory use is taken up by the input and output buffers -
    // two buffers of equal size - and for simplicity the volume is a cube:
    //   dim = floor( (N/2)^(1/3) )
    defaultDim = (int)floor(pow((memsize / (2.0 * sizeof(float))), 1.0/3.0));

    // By default, make the volume edge size an integer multiple of 128B to
    // improve performance by coalescing memory accesses, in a real
    // application it would make sense to pad the lines accordingly
    int roundTarget = 128 / sizeof(float);
    defaultDim = defaultDim / roundTarget * roundTarget;
    defaultDim -= k_radius_default * 2;

    // Check dimension is valid
    if (defaultDim < k_dim_min)
    {
        printf("insufficient device memory (maximum volume on device is %d, must be between %d and %d).\n", defaultDim, k_dim_min, k_dim_max);
        exit(EXIT_FAILURE);
    }
    else if (defaultDim > k_dim_max)
    {
        defaultDim = k_dim_max;
    }

    // For QA testing, override default volume size
    if (checkCmdLineFlag(argc, argv, "qatest"))
    {
        defaultDim = MIN(defaultDim, k_dim_qa);
    }

    //set default dim
    dimx = defaultDim;
    dimy = defaultDim;
    dimz = defaultDim;
    radius    = k_radius_default;
    timesteps = k_timesteps_default;

    // Parse command line arguments
    if (checkCmdLineFlag(argc, argv, "dimx"))
    {
        dimx = CLAMP(getCmdLineArgumentInt(argc, argv, "dimx"), k_dim_min, k_dim_max);
    }

    if (checkCmdLineFlag(argc, argv, "dimy"))
    {
        dimy = CLAMP(getCmdLineArgumentInt(argc, argv, "dimy"), k_dim_min, k_dim_max);
    }

    if (checkCmdLineFlag(argc, argv, "dimz"))
    {
        dimz = CLAMP(getCmdLineArgumentInt(argc, argv, "dimz"), k_dim_min, k_dim_max);
    }

    if (checkCmdLineFlag(argc, argv, "radius"))
    {
        radius = CLAMP(getCmdLineArgumentInt(argc, argv, "radius"), k_radius_min, k_radius_max);
    }

    if (checkCmdLineFlag(argc, argv, "timesteps"))
    {
        timesteps = CLAMP(getCmdLineArgumentInt(argc, argv, "timesteps"), k_timesteps_min, k_timesteps_max);
    }

    if (checkCmdLineFlag(argc, argv, "validate"))
    {
        validate = true;
    }

    // Determine volume size
    outerDimx = dimx + 2 * radius;
    outerDimy = dimy + 2 * radius;
    outerDimz = dimz + 2 * radius;
    volumeSize = outerDimx * outerDimy * outerDimz;
    //const int padding = (128 / sizeof(float)) - radius;
    const long unsigned int padding = (128 / sizeof(float)) - radius;
    //const size_t paddedVolumeSize = volumeSize + padding;
    const long unsigned int paddedVolumeSize = volumeSize + padding;

    cudaStream_t s1;
    cudaStreamCreate(&s1);

printf("padded volume size: %ld\n", paddedVolumeSize);
    // Allocate memory
    host_output = (float *)calloc(volumeSize, sizeof(float));
    //input       = (float *)malloc(volumeSize * sizeof(float));
    cudaMallocManaged((void**)&device_output, paddedVolumeSize * sizeof(float));
    cudaMallocManaged((void**)&input, sizeof(float)*paddedVolumeSize);
    //coeff       = (float *)malloc((radius + 1) * sizeof(float));
    cudaMallocManaged((void**)&coeff, sizeof(float)*(radius+1));

    cudaMemAdvise(coeff, sizeof(float)*(radius+1), cudaMemAdviseSetPreferredLocation, devID);
    cudaMemAdvise(coeff, sizeof(float)*(radius+1), cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);

    cudaMemAdvise(device_output, sizeof(float)*paddedVolumeSize, cudaMemAdviseSetPreferredLocation, devID);
    cudaMemAdvise(device_output, sizeof(float)*paddedVolumeSize, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);

    // Create coefficients
    for (int i = 0 ; i <= radius ; i++)
    {
        coeff[i] = 0.1f;
    }
    cudaMemAdvise(coeff, sizeof(float)*(radius+1), cudaMemAdviseSetReadMostly, devID);

    // Generate data
    printf(" generateRandomData\n\n");
    generateRandomData(input+padding, outerDimx, outerDimy, outerDimz, lowerBound, upperBound);
    memcpy(device_output+padding, input+padding, sizeof(float) * volumeSize);
    printf("FDTD on %d x %d x %d volume with symmetric filter radius %d for %d timesteps...\n\n", dimx, dimy, dimz, radius, timesteps);

    cudaMemPrefetchAsync(coeff, sizeof(float)*(radius+1), devID, s1);
    cudaMemPrefetchAsync(input, sizeof(float)*paddedVolumeSize, devID, s1);
    cudaMemPrefetchAsync(device_output, sizeof(float)*paddedVolumeSize, devID, s1);

    double compute_migrate_start = 0.0;

    if (validate) {
        // Allocate memory
        //device_output = (float *)calloc(volumeSize, sizeof(float));
        double cpu_start = mysecond();
        // Execute on the host
        printf("fdtdReference...\n");
        fdtdReference(host_output, input+padding, coeff, dimx, dimy, dimz, radius, timesteps);
        printf("fdtdReference complete\n");
        double elapsedtime = mysecond() - cpu_start;
        printf("cpu time: %f\n", elapsedtime);
    }

    // Execute on the device
    double gpu_start = mysecond();
    printf("fdtdGPU...\n");
    fdtdGPU(&device_output, input, coeff, dimx, dimy, dimz, radius, timesteps, argc, argv, &compute_migrate_start);
    printf("fdtdGPU complete\n");
    double gpuElapsedTime = mysecond() - gpu_start;
    printf("gpu time: %f\n", gpuElapsedTime);

    cudaMemPrefetchAsync(device_output, sizeof(float)*volumeSize, cudaCpuDeviceId);

    if (validate) {
        // Compare the results
        float tolerance = 0.0001f;
        printf("\nCompareData (tolerance %f)...\n", tolerance);
        return compareData(device_output, host_output, dimx, dimy, dimz, radius, tolerance);
    }
    else {
        memcpy(host_output, device_output, sizeof(float) * volumeSize);
    }

    double compute_migrate_time = mysecond() - compute_migrate_start;
    printf("compute migrate time: %f\n", compute_migrate_time);

    return 0;
}
