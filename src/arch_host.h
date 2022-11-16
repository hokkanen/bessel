#ifndef BESSEL_ARCH_HOST_H
#define BESSEL_ARCH_HOST_H

/* Include required headers */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* All macros and functions can be compiled with a C compiler */

/* Host backend initialization */
inline static void arch_init(int node_rank) {
  // Nothing needs to be done here
}

/* Host backend finalization */
inline static void arch_finalize(int rank) {
  printf("Rank %d, Host finalized.\n", rank);
}

/* Host function for memory allocation */
inline static void* arch_allocate(size_t bytes) {
  return malloc(bytes);
}

/* Host function for memory deallocation */
inline static void arch_free(void* ptr) {
  free(ptr);
}

/* Host-to-host memory copy */
inline static void arch_memcpy_d2d(void* dst, void* src, size_t bytes){
  memcpy(dst, src, bytes);
}

/* Atomic add function for host use */
inline static void arch_atomic_add(float *array_loc, float value){
  *array_loc += value;
}

/* A function for getting a random float from the standard distribution */
inline static float arch_random_float(unsigned long long seed, unsigned long long seq, unsigned int idx, float mean, float stdev){
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

/* Parallel for driver macro for the host loops */
#define arch_parallel_for(loop_size, inc, loop_body)                \
{                                                                   \
  for(inc = 0; inc < loop_size; inc++){                             \
    loop_body;                                                      \
  }                                                                 \
}

/* Parallel for driver macro for the host loops */
#define arch_parallel_reduce(loop_size, inc, sum, loop_body)        \
{                                                                   \
  for(inc = 0; inc < loop_size; inc++){                             \
    loop_body;                                                      \
  }                                                                 \
}
#endif // !BESSEL_ARCH_HOST_H
