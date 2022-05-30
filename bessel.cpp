/*
 * HAVE_DEF is set during compile time
 * and determines which accelerator backend is used
 * by including the respective header file
 */

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdio.h>

#if defined(HAVE_CUDA)
  #include "devices_cuda.h"
#elif defined(HAVE_HIP)
  #include "devices_hip.h"
#elif defined(HAVE_KOKKOS)
  #include "devices_kokkos.h"
#else
  #include "devices_host.h"
#endif

#include "comms.h"

#define N_BESSEL 12
#define ITERATIONS 1000
#define POPULATION 1000
#define SAMPLE 100

int main(int argc, char *argv []){

  // Set timer
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  // Initialize processes and devices
  comms::init_procs(&argc, &argv);
  int my_rank = comms::get_rank();

  // Memory allocation
  double* b_error = (double*)devices::allocate(ITERATIONS * N_BESSEL * sizeof(double));
  double* b_error_mean = (double*)devices::allocate(N_BESSEL * sizeof(double));

  // Use a non-deterministic random number generator for the master seed value
  std::random_device rd;

  // Use 64 bit Mersenne Twister 19937 generator
  std::mt19937_64 mt(rd());

  // Get a random unsigned long long from a uniform int distribution
  std::uniform_int_distribution<unsigned long long> dist(0, 1e10);

  // Get the non-deterministic random master seed value
  unsigned long long seed = dist(mt);

  // Initialize the mean error array
  devices::parallel_for(N_BESSEL, 
    DEVICE_LAMBDA(const int j) {
      b_error_mean[j] = 0.0;
    }
  );

  // Run the loop over iterations
  devices::parallel_for(ITERATIONS, 
    DEVICE_LAMBDA(const int iter) {

      double p_mean = 0.0;
      double s_mean = 0.0;
      
      for(int i = 0; i < POPULATION; ++i){
        unsigned long long seq = ((unsigned long long)iter * (unsigned long long)POPULATION) + (unsigned long long)i;
        double rnd_val = devices::random_double(seed, seq, 100.0, 15.0);
        p_mean += rnd_val;
        if(i < SAMPLE) s_mean += rnd_val;
        if(iter == 0 && i < 3) printf("Rank %d, rnd_val[%d]: %.5f \n", my_rank, i, rnd_val);
      }
      
      p_mean /= POPULATION;
      s_mean /= SAMPLE;
      
      double b_stdev[N_BESSEL];
      double b_sum = 0.0;
      double p_stdev = 0.0;
      
      for(int i = 0; i < POPULATION; ++i){
        unsigned long long seq = ((unsigned long long)iter * (unsigned long long)POPULATION) + (unsigned long long)i;
        double rnd_val = devices::random_double(seed, seq, 100.0, 15.0);
        double p_diff = rnd_val - p_mean;
        p_stdev += p_diff * p_diff;
        if(i < SAMPLE){
          double b_diff = rnd_val - s_mean;
          b_sum += b_diff * b_diff;   
        }
        //if(iter == 0 && i < 3) printf("Rank %d, rnd_val[%d]: %.5f? \n", my_rank, i, rnd_val);
      }
      p_stdev /= POPULATION;
      p_stdev = sqrt(p_stdev);
      //printf("p_stdev: %f\n",p_stdev);
      
      for(int j = 0; j < N_BESSEL; ++j){
        double sub = j * (1.2 / N_BESSEL);
        b_stdev[j] = b_sum / (SAMPLE - sub);
        b_stdev[j] = sqrt(b_stdev[j]);
        double diff = p_stdev - b_stdev[j];
        b_error[N_BESSEL * iter + j] = sqrt(diff * diff);
        //printf("b_stdev[%d]: %f, b_error[iter: %d][sub: %f]: %f\n", j, b_stdev[j], iter, sub, b_error[N_BESSEL * iter + j]);  
      }     
    }
  );

  // Sum the errors of each iteration
  devices::parallel_for(ITERATIONS, 
    DEVICE_LAMBDA(const int iter) {
      for(int j = 0; j < N_BESSEL; ++j){     
        devices::atomic_add(&b_error_mean[j], b_error[N_BESSEL * iter + j]);
      }
    }
  );

  // Each process sends its rank to reduction, root process collects the result
  comms::reduce_procs(b_error_mean, N_BESSEL);

  // Divide the error sum to find the averaged error for each tested Bessel value
  if(my_rank == 0){
    for(int j = 0; j < N_BESSEL; ++j){
      b_error_mean[j] /= (comms::get_procs() * ITERATIONS);
      double sub = j * (1.2 / N_BESSEL);
      printf("Mean error for Bessel = %.2f is %.10f\n", sub, b_error_mean[j]);
    }
  }
  
  // Memory deallocations
  devices::free((void*)b_error);
  devices::free((void*)b_error_mean);

  // Finalize processes and devices
  comms::finalize_procs();

  // Print timing
  if(my_rank == 0){
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
  }

  return 0;
}