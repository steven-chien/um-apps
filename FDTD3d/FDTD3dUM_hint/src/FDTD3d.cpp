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
double misc_start;
double misc_stop;
double misc_timer;

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
    /////////////////////// START TIMER ///////////////////////////
    application_start = mysecond();

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
    int timesteps;
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
        printf("insufficient device memory (maximum volume on device is %ld, must be between %d and %d).\n", defaultDim, k_dim_min, k_dim_max);
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


    // Allocate memory
    malloc_start = mysecond();
    host_output = (float *)calloc(volumeSize, sizeof(float));
    malloc_stop = mysecond();
    //input       = (float *)malloc(volumeSize * sizeof(float));
    cuda_malloc_start = malloc_stop;
    cudaMallocManaged((void**)&device_output, paddedVolumeSize * sizeof(float));
    cudaMallocManaged((void**)&input, sizeof(float)*paddedVolumeSize);
    //coeff       = (float *)malloc((radius + 1) * sizeof(float));
    cudaMallocManaged((void**)&coeff, sizeof(float)*(radius+1));
    cuda_malloc_stop = mysecond();

    advise_start = cuda_malloc_stop;
    cudaMemAdvise(coeff, sizeof(float)*(radius+1), cudaMemAdviseSetPreferredLocation, devID);
    cudaMemAdvise(coeff, sizeof(float)*(radius+1), cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);

    cudaMemAdvise(device_output, sizeof(float)*paddedVolumeSize, cudaMemAdviseSetPreferredLocation, devID);
    cudaMemAdvise(device_output, sizeof(float)*paddedVolumeSize, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
    advise_stop = mysecond();

    // Create coefficients
    init_data_start = advise_stop;
    for (int i = 0 ; i <= radius ; i++)
    {
        coeff[i] = 0.1f;
    }

    // Generate data
    printf(" generateRandomData\n\n");
    generateRandomData(input+padding, outerDimx, outerDimy, outerDimz, lowerBound, upperBound);
    memcpy(device_output+padding, input+padding, sizeof(float) * volumeSize);
    printf("FDTD on %ld x %ld x %ld volume with symmetric filter radius %ld for %d timesteps...\n\n", dimx, dimy, dimz, radius, timesteps);
    init_data_stop = mysecond();

    advise_read_start = init_data_stop;
    cudaMemAdvise(coeff, sizeof(float)*(radius+1), cudaMemAdviseSetReadMostly, devID);
    advise_read_stop = mysecond();

    if (validate) {
        // Allocate memory
        //device_output = (float *)calloc(volumeSize, sizeof(float));
        // Execute on the host
	cpu_start = mysecond();
        printf("fdtdReference...\n");
        fdtdReference(host_output, input+padding, coeff, dimx, dimy, dimz, radius, timesteps);
        printf("fdtdReference complete\n");
	cpu_stop = mysecond();
    }

    // Execute on the device
    printf("fdtdGPU...\n");
    fdtdGPU(&device_output, input, coeff, dimx, dimy, dimz, radius, timesteps, argc, argv);
    printf("fdtdGPU complete\n");

    if (validate) {
        // Compare the results
        float tolerance = 0.0001f;
        printf("\nCompareData (tolerance %f)...\n", tolerance);
        return compareData(device_output, host_output, dimx, dimy, dimz, radius, tolerance);
    }
    else {
	d2h_memcpy_start = mysecond();
        memcpy(host_output, device_output, sizeof(float) * volumeSize);
	d2h_memcpy_stop = mysecond();
    }

    compute_migrate_stop = mysecond();
    application_stop = compute_migrate_stop;

    printf("\nGPU Time: %f\n", gpu_stop - gpu_start);
    printf("CPU Time: %f\n", cpu_stop - cpu_start);
    printf("malloc timer: %f\n", malloc_stop - malloc_start);
    printf("free timer: %f\n", free_stop - free_start);
    printf("cuda malloc timer: %f\n", cuda_malloc_stop - cuda_malloc_start);
    printf("cuda free timer: %f\n", cuda_free_stop - cuda_free_start);
    printf("Init data timer: %f\n", init_data_stop - init_data_start);
    printf("misc timer: %f\n", malloc_start - application_start + misc_timer);
    printf("\nAdivse timer: %f\n", (advise_stop - advise_start) + (advise_read_stop - advise_read_start));
    //printf("\nH2D timer: %f\n", h2d_memcpy_stop - h2d_memcpy_start);
    printf("\nD2H timer: %f\n", d2h_memcpy_stop - d2h_memcpy_start);
    printf("\ncompute migrate timer: %f\n", compute_migrate_stop - compute_migrate_start);
    printf("applicaiton timer: %f\n\n", application_stop - application_start);

    return 0;
}
