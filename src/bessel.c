#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Namespaces "comms" and "arch" declared here */
#include "comms.h"
#include "./arch/arch_api.h"

/* Set problem dimensions */
#define N_ITER 10000
#define N_POPU 10000
#define N_SAMPLE 50

int main(int argc, char *argv []){

  /* Initialize processes and devices */
  comms_init_procs(&argc, &argv);
  const unsigned int my_rank = comms_get_rank();

  /* Some device backends require an initialization */
  arch_init(comms_get_node_rank());

  /* Set spacing and range for beta */
  const unsigned int n_beta = 40;
  const float range_beta = 4.0f;

  /* Set timer */
  clock_t begin = clock();

  /* Array for the mean squared errors (the sum of errors across iterations) */
  float mse[2 * n_beta];

  /* Initialize mse array to zero */
  memset(mse,'0', 2 * n_beta * sizeof(float));

  /* Get the time value for the seed (only the value from the root rank is used) */
  int timeval = (int)time(0);

  /* Broadcast root rank time value to all ranks */
  comms_bcast(&timeval, 1, 0);

  /* Get the random master seed value (rank + root rank time value gives a unique seed) */
  srand(my_rank + timeval);
  unsigned long long seed = (unsigned long long)rand();

  /* Run the loop over iterations */
  unsigned int iter;
  arch_parallel_reduce(N_ITER, iter, mse, 2 * n_beta,
  {
    /* Calculate the mean and the squared sum of the population and the sample */
    float p_mean = 0.0f;
    float p_squared = 0.0f;
    float s_mean = 0.0f;
    float s_squared = 0.0f;
    
    for(unsigned int i = 0; i < N_POPU; ++i){
      unsigned long long seq = ((unsigned long long)iter * (unsigned long long)N_POPU) + (unsigned long long)i;
      float rnd_val = arch_random_float(seed, seq, i, 100.0f, 15.0f);
      p_mean += rnd_val;
      p_squared += rnd_val * rnd_val;
      if (i < N_SAMPLE)
      {
        s_mean += rnd_val;
        s_squared += rnd_val * rnd_val;
      }
      if(iter == 0 && i < 3) 
        printf("Rank %u, iter: %u ,rnd_val[%u]: %.5f \n", my_rank, iter, i, rnd_val);
    }
    p_mean /= N_POPU;
    s_mean /= N_SAMPLE;

    /* Calculate variance for the population and sum for the sample */
    float p_var = (p_squared - N_POPU * p_mean * p_mean) / N_POPU;
    float s_sum = s_squared - N_SAMPLE * s_mean * s_mean;
    // printf("p_var: %f\n",p_var);
    
    /* Calculate the mean squared error in the sample standard deviation and variance for different beta */
    float s_var[n_beta];
    float *mse_stdev = &mse[0];
    float *mse_var = &mse[n_beta];
    for(unsigned int j = 0; j < n_beta; ++j){
      float sub = j * (range_beta / n_beta) - range_beta / 2.0f;
      s_var[j] = s_sum / (N_SAMPLE - sub);
      float diff_stdev = sqrtf(p_var) - sqrtf(s_var[j]);
      float diff_var = p_var - s_var[j];
      //printf("s_var[%u]: %f, error[iter: %u][sub: %f]: %f\n", j, s_var[j], iter, sub, sqrt(diff_var * diff_var));  
      
      /* Sum the errors of each iteration */
      mse_stdev[j] += diff_stdev * diff_stdev;
      mse_var[j] += diff_var * diff_var;
    }     
  });

  /* Each process sends its values to reduction, root process collects the results */
  comms_reduce_procs(mse, 2 * n_beta);

  /* Divide the error sums to find the averaged errors for each tested beta value */
  if(my_rank == 0){
    float *mse_stdev = &mse[0];
    float *mse_var = &mse[n_beta];
    for(unsigned int j = 0; j < n_beta; ++j){
      mse_stdev[j] /= (comms_get_procs() * N_ITER);
      mse_var[j] /= (comms_get_procs() * N_ITER);
      float rmse_stdev = sqrtf(mse_stdev[j]);
      float rmse_var = sqrtf(mse_var[j]);
      float sub = j * (range_beta / n_beta) - range_beta / 2.0f;
      printf("Beta = %.2f: RMSE for stdev = %.5f and var = %.5f\n", sub, rmse_stdev, rmse_var);
    }
  }
  /* Some device backends require a finalization */
  arch_finalize(comms_get_rank());

  /* Finalize processes and devices */
  comms_finalize_procs();

  /* Print timing */
  if(my_rank == 0){
    clock_t diff = clock() - begin;
    unsigned int msec = diff  * 1000.0f / CLOCKS_PER_SEC;
    printf("%u[ms]\n", msec);
  }

  return 0;
}
