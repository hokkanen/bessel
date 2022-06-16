#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// All macros and functions can be compiled with a C compiler

inline static void devices_init(int node_rank) {
  // Nothing needs to be done here
}

inline static void devices_finalize(int rank) {
  printf("Rank %d, Host finalized.\n", rank);
}

inline static void* devices_allocate(size_t bytes) {
  return malloc(bytes);
}

inline static void devices_free(void* ptr) {
  free(ptr);
}

inline static void devices_memcpy_d2d(void* dst, void* src, size_t bytes){
  memcpy(dst, src, bytes);
}

inline static void devices_atomic_add(double *array_loc, double value){
  *array_loc += value;
}

inline static double devices_random_double(unsigned long long seed, unsigned long long idx, double mean, double stdev){
  
  // Use Box Muller algorithm to get a double from a normal distribution
  const double two_pi = 2.0 * M_PI;
	double u1 = (double) rand() / RAND_MAX;
	double u2 = (double) rand() / RAND_MAX;
	double factor = stdev * sqrt ( -2 * log (u1) );
	double trig_arg = two_pi * u2;
	
  // Box Muller algorithm produces two random normally distributed doubles, z0 and z1
  double z0 = factor * cos (trig_arg) + mean; // Need only one
	// double z1 = factor * sin (trig_arg) + mean; 
  return z0;
}

#define devices_parallel_for(loop_size, inc, loop_body)             \
{                                                                   \
  for(inc = 0; inc < loop_size; inc++){                             \
    loop_body;                                                      \
  }                                                                 \
}
