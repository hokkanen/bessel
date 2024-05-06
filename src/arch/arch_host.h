#ifndef BESSEL_ARCH_HOST_H
#define BESSEL_ARCH_HOST_H

/* Include required headers */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#include <curand_kernel.h>
#endif

/* All macros and functions can be compiled with a C compiler */

/* Host backend initialization */
inline static void arch_init(int node_rank)
{
  // Nothing needs to be done here
}

/* Host backend finalization */
inline static void arch_finalize(int rank)
{
#ifdef _OPENMP
  printf("Rank %d, OpenMP (offload) finalized.\n", rank);

#else
  printf("Rank %d, Host finalized.\n", rank);
#endif
}

/* Host function for memory allocation */
inline static void *arch_allocate(size_t bytes)
{
  return malloc(bytes);
}

/* Host function for memory deallocation */
inline static void arch_free(void *ptr)
{
  free(ptr);
}

/* Host-to-host memory copy */
inline static void arch_memcpy_d2d(void *dst, void *src, size_t bytes)
{
#ifdef _OPENMP
  const int dev = omp_get_default_device();
  omp_target_memcpy(dst, src, bytes, 0, 0, dev, dev);
#else
  memcpy(dst, src, bytes);
#endif
}

/* Atomic add function for host use */
#pragma omp declare target
inline static void arch_atomic_add(float *array_loc, float value)
{
#pragma omp atomic update
  *array_loc += value;
}
#pragma omp end declare target

/* A function for getting a random float from the standard distribution */
#pragma omp declare target
inline static float arch_random_float(unsigned long long seed, unsigned long long seq, unsigned int idx, float mean, float stdev)
{
  /* Re-seed the first case */
  if (idx == 0)
  {
    /* Overflow is defined behavior with unsigned, and therefore ok here */
    srand((unsigned int)seed + (unsigned int)seq);
  }
  float z0 = 0;
#if _OPENMP /* Curand works with OpenMP when compiling with nvc */
  curandStatePhilox4_32_10_t state;
  /* curand_init() reproduces the same random number with the same seed and seq */
  curand_init(seed, seq, 0, &state);
  /* curand_normal() gives a random float from a normal distribution with mean = 0 and stdev = 1 */
  z0 = stdev * curand_normal(&state) + mean;
#else
  /* Use Box Muller algorithm to get a float from a normal distribution */
  const float two_pi = 2.0f * M_PI;
  const float u1 = (float)rand() / (float)RAND_MAX;
  const float u2 = (float)rand() / (float)RAND_MAX;
  const float factor = stdev * sqrtf(-2.0f * logf(u1));
  const float trig_arg = two_pi * u2;
  /* Box Muller algorithm produces two random normally distributed floats, z0 and z1 */
  z0 = factor * cosf(trig_arg) + mean; /* Need only one */
  // float z1 = factor * sinf (trig_arg) + mean;
#endif
  return z0;
}
#pragma omp end declare target

/* Parallel for driver macro for the host loops */
#define arch_parallel_for(loop_size, inc, loop_body) \
  {                                                  \
#pragma omp target teams distribute parallel for     \
    for (inc = 0; inc < loop_size; inc++)            \
    {                                                \
      loop_body;                                     \
    }                                                \
  }

/* Parallel for driver macro for the host loops */
#define arch_parallel_reduce(loop_size, inc, sum, num_sum, loop_body)            \
  {                                                                              \
#pragma omp target teams distribute parallel for reduction(+ : sum[0 : num_sum]) \
    for (inc = 0; inc < loop_size; inc++)                                        \
    {                                                                            \
      loop_body;                                                                 \
    }                                                                            \
  }
#endif // !BESSEL_ARCH_HOST_H
