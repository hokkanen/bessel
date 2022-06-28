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

// Set problem dimensions
#define N_ITER 10000
#define N_POPU 10000
#define N_SAMPLE 50

int main(int argc, char *argv []){

  // Initialize processes and devices
  comms_init_procs(&argc, &argv);
  const unsigned int my_rank = comms_get_rank();

  // Set spacing and range for beta
  const unsigned int n_beta = 40;
  const float range_beta = 4.0f;

  // Set timer
  clock_t begin = clock();

  // Memory allocation
  float* mse_stdev = (float*)devices_allocate(n_beta * sizeof(float));
  float* mse_var = (float*)devices_allocate(n_beta * sizeof(float));

  // Get the time value for the seed (only the value from the root rank is used)
  int timeval = (int)time(0);

  // Broadcast root rank time value to all ranks
  comms_bcast(&timeval, 1, 0);

  // Get the random master seed value (rank + root rank time value gives a unique seed)
  srand(my_rank + (unsigned int)timeval);
  unsigned long long seed = (unsigned long long)rand();

  // Initialize the mean error array
  int j;
  devices_parallel_for(n_beta, j, 
  {
    mse_stdev[j] = 0.0f;
    mse_var[j] = 0.0f;
  });

  // Run the loop over iterations
  int iter;
  devices_parallel_for(N_ITER, iter, 
  {
    float p_mean = 0.0f;
    float s_mean = 0.0f;
    
    for(int i = 0; i < N_POPU; ++i){
      unsigned long long seq = ((unsigned long long)iter * (unsigned long long)N_POPU) + (unsigned long long)i;
      float rnd_val = devices_random_float(seed, seq, i, 100.0f, 15.0f);
      p_mean += rnd_val;
      if(i < N_SAMPLE) s_mean += rnd_val;
      if(iter == 0 && i < 3) printf("Rank %u, rnd_val[%d]: %.5f \n", my_rank, i, rnd_val);
    }
    
    p_mean /= N_POPU;
    s_mean /= N_SAMPLE;
    
    float b_var[n_beta];
    float b_sum = 0.0f;
    float p_var = 0.0f;
    
    for(int i = 0; i < N_POPU; ++i){
      unsigned long long seq = ((unsigned long long)iter * (unsigned long long)N_POPU) + (unsigned long long)i;
      float rnd_val = devices_random_float(seed, seq, i, 100.0f, 15.0f);
      float p_diff = rnd_val - p_mean;
      p_var += p_diff * p_diff;
      if(i < N_SAMPLE){
        float b_diff = rnd_val - s_mean;
        b_sum += b_diff * b_diff;   
      }
      //if(iter == 0 && i < 3) printf("Rank %u, rnd_val[%d]: %.5f? \n", my_rank, i, rnd_val);
    }
    p_var /= N_POPU;
    //printf("p_var: %f\n",p_var);
    
    for(int j = 0; j < n_beta; ++j){
      float sub = j * (range_beta / n_beta) - range_beta / 2.0f;
      b_var[j] = b_sum / (N_SAMPLE - sub);
      float diff_stdev = sqrtf(p_var) - sqrtf(b_var[j]);
      float diff_var = p_var - b_var[j];
      //printf("b_var[%d]: %f, error[iter: %d][sub: %f]: %f\n", j, b_var[j], iter, sub, sqrt(diff_var * diff_var));  
      
      // Sum the errors of each iteration
      devices_atomic_add(&mse_stdev[j], diff_stdev * diff_stdev);
      devices_atomic_add(&mse_var[j], diff_var * diff_var);
    }     
  });

  // Each process sends its values to reduction, root process collects the results
  comms_reduce_procs(mse_stdev, n_beta);
  comms_reduce_procs(mse_var, n_beta);

  // Divide the error sums to find the averaged errors for each tested beta value
  if(my_rank == 0){
    for(int j = 0; j < n_beta; ++j){
      mse_stdev[j] /= (comms_get_procs() * N_ITER);
      mse_var[j] /= (comms_get_procs() * N_ITER);
      float rmse_stdev = sqrtf(mse_stdev[j]);
      float rmse_var = sqrtf(mse_var[j]);
      float sub = j * (range_beta / n_beta) - range_beta / 2.0f;
      printf("Beta = %.2f: RMSE for stdev = %.5f and var = %.5f\n", sub, rmse_stdev, rmse_var);
    }
  }
  
  // Memory deallocations
  devices_free((void*)mse_stdev);
  devices_free((void*)mse_var);

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
