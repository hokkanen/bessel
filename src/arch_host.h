#ifndef BESSEL_ARCH_HOST_H
#define BESSEL_ARCH_HOST_H

/* Include required headers */
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string.h>

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
    printf("Rank %d, Host finalized.\n", rank);
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
#ifdef _OPENMP
    const int dev = omp_get_default_device();
    omp_target_memcpy(dst, src, bytes, 0, 0, dev, dev);
#else
    memcpy(dst, src, bytes);
#endif
  }

  /* Atomic add function for host use */
  template <typename T>
  inline static void atomic_add(T *array_loc, T value)
  {
#pragma omp atomic update
    *array_loc += value;
  }

  /* A function to make sure the seed is of right type */
  template <typename T>
  static unsigned long long random_state_seed(T& seed)
  {
    return (unsigned long long)seed;
  }

  /* A function for initializing a random number generator state */
#pragma omp declare target
  template <typename T>
  inline static auto random_state_init(T& seed, unsigned int iter, unsigned long long pos)
  {
#if _OPENMP /* Curand works with OpenMP when compiling with nvc++ */
    /* curand_init() reproduces the same random number with the same seed and pos */
    curandStatePhilox4_32_10_t state;
    curand_init(seed, pos, 0, &state);
    return state;
#else
    /* Re-seed the first case (overflow is defined behavior with unsigned, and ok here) */
    srand((unsigned int)seed + (unsigned int)pos);
    return 0;
#endif
  }

  /* A function for freeing a random number generator state (not needed by host) */
#pragma omp declare target
  template <typename T, typename T2>
  inline static void random_state_free(T& seed, T2& generator)
  {
    (void)seed;
    (void)generator;
  }

/* A function for getting a random float from the standard distribution */
#pragma omp declare target
  template <typename T, typename T2>
  inline static T random_float(T2& state, T mean, T stdev)
  {
    T z0 = 0;
#if _OPENMP /* Curand works with OpenMP when compiling with nvc++ */
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
  /* Parallel for driver function for the host loops */
  template <typename Lambda>
  inline static void parallel_for(unsigned int loop_size, Lambda loop_body)
  {
#pragma omp target teams distribute parallel for
    for (unsigned int i = 0; i < loop_size; i++)
    {
      loop_body(i);
    }
  }

  /* Parallel reduce driver function for the host reductions */
  template <unsigned int NReductions, typename Lambda, typename T>
  inline static void parallel_reduce(const unsigned int loop_size, T (&sum)[NReductions], Lambda loop_body)
  {
#pragma omp target teams distribute parallel for reduction(+ : sum[0 : NReductions])
    for (unsigned int i = 0; i < loop_size; i++)
    {
      loop_body(i, sum);
    }
  }
}
#endif // !BESSEL_ARCH_HOST_H
