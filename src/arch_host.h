#ifndef BESSEL_ARCH_HOST_H
#define BESSEL_ARCH_HOST_H

/* Include required headers */
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string.h>

/* Define architecture-specific macros */
#define ARCH_LOOP_LAMBDA [=]

/* Namespace for architecture-specific functions */
namespace arch
{
  /* Host backend initialization */
  inline static void init(int node_rank) {
    // Nothing needs to be done here
  }

  /* Host backend finalization */
  inline static void finalize(int rank) {
    printf("Rank %d, Host finalized.\n", rank);
  }

  /* Host function for memory allocation */
  inline static void* allocate(size_t bytes) {
    return malloc(bytes);
  }

  /* Host function for memory deallocation */
  inline static void free(void* ptr) {
    ::free(ptr);
  }
  
  /* Host-to-host memory copy */
  inline static void memcpy_d2d(void* dst, void* src, size_t bytes){
    memcpy(dst, src, bytes);
  }

  /* Atomic add function for host use */
  template <typename T>
  inline static void atomic_add(T *array_loc, T value){
    *array_loc += value;
  }

  /* A function for getting a random float from the standard distribution */
  template <typename T>
  inline static T random_float(unsigned long long seed, unsigned long long seq, unsigned int idx, T mean, T stdev){
    /* Re-seed the first case */
    if(idx == 0){
      /* Overflow is defined behavior with unsigned, and therefore ok here */
      srand((unsigned int)seed + (unsigned int)seq);
    }
    /* Use Box Muller algorithm to get a float from a normal distribution */
    const float two_pi = 2.0f * M_PI;
    const float u1 = (float) rand() / RAND_MAX;
    const float u2 = (float) rand() / RAND_MAX;
    const float factor = stdev * sqrtf (-2.0f * logf (u1));
    const float trig_arg = two_pi * u2;  
    /* Box Muller algorithm produces two random normally distributed floats, z0 and z1 */
    const float z0 = factor * cosf (trig_arg) + mean; /* Need only one */
    // float z1 = factor * sinf (trig_arg) + mean; 
    return z0;
  }

  /* Parallel for driver function for the host loops */
  template <typename Lambda>
  inline static void parallel_for(unsigned int loop_size, Lambda loop_body) {
    for(unsigned int i = 0; i < loop_size; i++){
      loop_body(i);
    }
  }

  /* Parallel reduce driver function for the host reductions */
  template <unsigned int NReductions, typename Lambda, typename T>
  inline static void parallel_reduce(const unsigned int loop_size, T (&sum)[NReductions], Lambda loop_body) {
    for(unsigned int i = 0; i < loop_size; i++){
      loop_body(i, sum);
    }
  }
}
#endif // !BESSEL_ARCH_HOST_H
