#ifndef BESSEL_ARCH_HOST_H
#define BESSEL_ARCH_HOST_H

/* Include required headers */
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string.h>

#ifdef _OPENACC
#include <openacc.h>
#include <curand_kernel.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#include <curand_kernel.h>
#endif

/* Define architecture-specific macros */
#define ARCH_LOOP_LAMBDA [=]

/* Namespace for architecture-specific functions */
namespace arch
{
  /* Host backend initialization */
  inline static void init(int node_rank)
  {
    // Nothing needs to be done here
  }

  /* Host backend finalization */
  inline static void finalize(int rank)
  {
#ifdef _OPENACC
    printf("Rank %d, OpenACC (offload) finalized.\n", rank);
#elif defined(_OPENMP)
    printf("Rank %d, OpenMP (offload) finalized.\n", rank);
#else
    printf("Rank %d, Host finalized.\n", rank);
#endif
  }

  /* Host function for memory allocation */
  inline static void *allocate(size_t bytes)
  {
    return malloc(bytes);
  }

  /* Host function for memory deallocation */
  inline static void free(void *ptr)
  {
    ::free(ptr);
  }

  /* Host-to-host memory copy */
  inline static void memcpy_d2d(void *dst, void *src, size_t bytes)
  {
#ifdef _OPENACC
    int dev = acc_get_device_num(acc_device_nvidia);
    acc_memcpy_device(dst, src, bytes);
#elif defined(_OPENMP)
    const int dev = omp_get_default_device();
    omp_target_memcpy(dst, src, bytes, 0, 0, dev, dev);
#else
    memcpy(dst, src, bytes);
#endif
  }

  /* Atomic add function for host use */
#pragma acc routine
#pragma omp declare target
  template <typename T>
  inline static void atomic_add(T *array_loc, T value)
  {
#pragma acc atomic update
#pragma omp atomic update
    *array_loc += value;
  }
#pragma omp end declare target

  /* A function to make sure the seed is of right type */
  template <typename T>
  static unsigned long long random_state_seed(T &seed)
  {
    return (unsigned long long)seed;
  }

  /* A function for initializing a random number generator state */
#pragma acc routine
#pragma omp declare target
  template <typename T>
  inline static auto random_state_init(T &seed, unsigned long long pos)
  {
    /* Curand works with OpenACC and OpenMP when compiling with nvc++ */
#if defined(_OPENACC) || defined(_OPENMP)
    curandStatePhilox4_32_10_t state;
    /* curand_init() reproduces the same random number with the same seed and pos */
    curand_init(seed, pos, 0, &state);
    return state;
#else
    /* Re-seed the first case (overflow is defined behavior with unsigned, and ok here) */
    srand((unsigned)seed + (unsigned)pos);
    return 0;
#endif
  }
#pragma omp end declare target

  /* A function for freeing a random number generator state (not needed by host) */
#pragma acc routine
#pragma omp declare target
  template <typename T, typename T2>
  inline static void random_state_free(T &seed, T2 &generator)
  {
    (void)seed;
    (void)generator;
  }
#pragma omp end declare target

/* A function for getting a random float from the standard distribution */
#pragma acc routine
#pragma omp declare target
  template <typename T, typename T2>
  inline static T random_float(T2 &state, T mean, T stdev)
  {
    float z0 = 0;
    /* Curand works with OpenACC and OpenMP when compiling with nvc++ */
#if defined(_OPENACC) || defined(_OPENMP)
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
    return (T)z0;
  }
#pragma omp end declare target

  /* Parallel for driver function for the host loops */
  template <typename Lambda>
  inline static void parallel_for(unsigned loop_size, Lambda loop_body)
  {
    /* OpenACC requires specifying all levels (gang, worker and vector) here to prevent inner loop parallelization */
#pragma acc parallel loop independent gang worker vector
#pragma omp target teams distribute parallel for
    /* Execute the standard for loop */
    for (unsigned i = 0; i < loop_size; i++)
    {
      loop_body(i);
    }
  }

  /* Parallel reduce driver function for the host reductions */
  template <unsigned NReductions, typename Lambda, typename T>
  inline static void parallel_reduce(const unsigned loop_size, T (&sum)[NReductions], Lambda loop_body)
  {
    /* Introduce aux pointer to avoid nvc++ (24.3-0) omp offload compile crash */
    T *aux_sum = &(sum[0]);
    /* OpenACC requires specifying all levels (gang, worker and vector) here to prevent inner loop parallelization */
#pragma acc parallel loop independent gang worker vector reduction(+ : aux_sum[0 : NReductions])
#pragma omp target teams distribute parallel for reduction(+ : aux_sum[0 : NReductions])
    /* Execute the reduction loop (OpenACC requires specifying all parallelization levels here to prevent inner loop parallelization)*/
    for (unsigned i = 0; i < loop_size; i++)
    {
      loop_body(i, aux_sum);
    }
  }
}
#endif // !BESSEL_ARCH_HOST_H
