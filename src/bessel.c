/*
 * HAVE_DEF is set during compile time
 * and determines which accelerator backend is used
 * by including the respective header file
 */

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Namespaces "comms" and "devices" declared here
#include "comms.h"

#define N_BESSEL 12
#define ITERATIONS 10000
#define POPULATION 10000
#define SAMPLE 50

int main(int argc, char *argv []){

  // Set timer
  clock_t begin = clock();

  // Initialize processes and devices
  comms_init_procs(&argc, &argv);
  unsigned int my_rank = comms_get_rank();

  // Memory allocation
  float* b_error_mean = (float*)devices_allocate(N_BESSEL * sizeof(float));

  // Get the time value for the seed (only the value from the root rank is used)
  int timeval = (int)time(0);

  // Broadcast root rank time value to all ranks
  comms_bcast(&timeval, 1, 0);

  // Get the random master seed value (rank + root rank time value gives a unique seed)
  srand(my_rank + (unsigned int)timeval);
  unsigned long long seed = (unsigned long long)rand();

  // Initialize the mean error array
  int j;
  devices_parallel_for(N_BESSEL, j, 
  {
    b_error_mean[j] = 0.0f;
  });

  // Run the loop over iterations
  int iter;
  devices_parallel_for(ITERATIONS, iter, 
  {
    float p_mean = 0.0f;
    float s_mean = 0.0f;
    
    for(int i = 0; i < POPULATION; ++i){
      unsigned long long seq = ((unsigned long long)iter * (unsigned long long)POPULATION) + (unsigned long long)i;
      float rnd_val = devices_random_float(seed, seq, i, 100.0f, 15.0f);
      p_mean += rnd_val;
      if(i < SAMPLE) s_mean += rnd_val;
      if(iter == 0 && i < 3) printf("Rank %u, rnd_val[%d]: %.5f \n", my_rank, i, rnd_val);
    }
    
    p_mean /= POPULATION;
    s_mean /= SAMPLE;
    
    float b_stdev[N_BESSEL];
    float b_sum = 0.0f;
    float p_stdev = 0.0f;
    
    for(int i = 0; i < POPULATION; ++i){
      unsigned long long seq = ((unsigned long long)iter * (unsigned long long)POPULATION) + (unsigned long long)i;
      float rnd_val = devices_random_float(seed, seq, i, 100.0f, 15.0f);
      float p_diff = rnd_val - p_mean;
      p_stdev += p_diff * p_diff;
      if(i < SAMPLE){
        float b_diff = rnd_val - s_mean;
        b_sum += b_diff * b_diff;   
      }
      //if(iter == 0 && i < 3) printf("Rank %u, rnd_val[%d]: %.5f? \n", my_rank, i, rnd_val);
    }
    p_stdev /= POPULATION;
    p_stdev = sqrtf(p_stdev);
    //printf("p_stdev: %f\n",p_stdev);
    
    for(int j = 0; j < N_BESSEL; ++j){
      float sub = j * (1.2f / N_BESSEL);
      b_stdev[j] = b_sum / (SAMPLE - sub);
      b_stdev[j] = sqrtf(b_stdev[j]);
      float diff = p_stdev - b_stdev[j];
      //printf("b_stdev[%d]: %f, error[iter: %d][sub: %f]: %f\n", j, b_stdev[j], iter, sub, sqrt(diff * diff));  
      // Sum the errors of each iteration
      devices_atomic_add(&b_error_mean[j], sqrtf(diff * diff));
    }     
  });

  // Each process sends its rank to reduction, root process collects the result
  comms_reduce_procs(b_error_mean, N_BESSEL);

  // Divide the error sum to find the averaged error for each tested Bessel value
  if(my_rank == 0){
    for(int j = 0; j < N_BESSEL; ++j){
      b_error_mean[j] /= (comms_get_procs() * ITERATIONS);
      float sub = j * (1.2f / N_BESSEL);
      printf("Mean error for Bessel = %.2f is %.10f\n", sub, b_error_mean[j]);
    }
  }
  
  // Memory deallocations
  devices_free((void*)b_error_mean);

  // Finalize processes and devices
  comms_finalize_procs();

  // Print timing
  if(my_rank == 0){
    clock_t diff = clock() - begin;
    int msec = diff  * 1000.0f / CLOCKS_PER_SEC;
    printf("%d[ms]\n", msec);
  }

  return 0;
}