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

#ifndef _FDTD3D_H_
#define _FDTD3D_H_

// The values are set to give reasonable runtimes, they can
// be changed but note that running very large dimensions can
// take a very long time and you should avoid running on your
// primary display in this case.
//#define k_dim_min           96
//#define k_dim_max           376
//#define k_dim_qa            248

#define k_dim_min           96
//#define k_dim_max           1024
#define k_dim_max           2048
#define k_dim_qa            248

// Note that the radius is defined here as exactly 4 since the
// kernel code uses a constant. If you want a different radius
// you must change the kernel accordingly.
#define k_radius_min        4
#define k_radius_max        4
#define k_radius_default    4

// The values are set to give reasonable runtimes, they can
// be changed but note that running a very large number of
// timesteps can take a very long time and you should avoid
// running on your primary display in this case.
#define k_timesteps_min     1
//#define k_timesteps_max     10
#define k_timesteps_max     20
#define k_timesteps_default 5

#include <sys/time.h>
inline double mysecond(){
    struct timeval tp;
    struct timezone tzp;
    int i;

    i = gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

extern double gpu_start;
extern double gpu_stop;
extern double cpu_start;
extern double cpu_stop;
extern double application_start;
extern double application_stop;
extern double compute_migrate_start;
extern double compute_migrate_stop;
extern double malloc_start;
extern double malloc_stop;
extern double free_start;
extern double free_stop;
extern double cuda_malloc_start;
extern double cuda_malloc_stop;
extern double cuda_free_start;
extern double cuda_free_stop;
extern double init_data_start;
extern double init_data_stop;
extern double h2d_memcpy_start;
extern double h2d_memcpy_stop;
extern double d2h_memcpy_start;
extern double d2h_memcpy_stop;
extern double h2d_prefetch_start;
extern double h2d_prefetch_stop;
extern double d2h_prefetch_start;
extern double d2h_prefetch_stop;
extern double advise_start;
extern double advise_stop;
extern double advise_read_start;
extern double advise_read_stop;
extern double misc_start;
extern double misc_stop;
extern double misc_timer;

#endif
