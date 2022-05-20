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

#define N_BESSEL 12
#define ITERATIONS 10000
#define POPULATION 1000
#define SAMPLE 10

using namespace std;

int main(){

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  // Some device backends require an initialization
  devices::init();

  // Memory allocation
  double* b_error = (double*)devices::allocate(ITERATIONS * N_BESSEL * sizeof(double));
  double* b_mean_error = (double*)devices::allocate(N_BESSEL * sizeof(double));
  double* rnd_val = (double*)devices::allocate(ITERATIONS * POPULATION * sizeof(double));

  devices::parallel_for(N_BESSEL, 
    DEVICE_LAMBDA(const int j) {
      b_mean_error[j] = 0.0;
    }
  );

  devices::random_array(100.0, 15.0, rnd_val, ITERATIONS * POPULATION);

  devices::parallel_for(ITERATIONS, 
    DEVICE_LAMBDA(const int iter) {

      double p_mean = 0.0;
      double s_mean = 0.0;
      
      for(int i = 0; i < POPULATION; ++i){
        p_mean += rnd_val[POPULATION * iter + i];
        if(i < SAMPLE) s_mean += rnd_val[POPULATION * iter + i];
        if(iter == 0 && i < 3) printf("rnd_val[%d][%d]: %.5f \n", iter, i, rnd_val[POPULATION * iter + i]);
      }
      
      p_mean /= POPULATION;
      s_mean /= SAMPLE;
      
      double b_stdev[N_BESSEL];
      double b_sum = 0.0;
      double p_stdev = 0.0;
      
      for(int i = 0; i < POPULATION; ++i){
        double p_diff = rnd_val[POPULATION * iter + i] - p_mean;
        p_stdev += p_diff * p_diff;
        if(i < SAMPLE){
          double b_diff = rnd_val[POPULATION * iter + i] - s_mean;
          b_sum += b_diff * b_diff;   
        }
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
  devices::parallel_for(ITERATIONS, 
    DEVICE_LAMBDA(const int iter) {
      for(int j = 0; j < N_BESSEL; ++j){     
        devices::atomic_add(&b_mean_error[j], b_error[N_BESSEL * iter + j]);
      }
    }
  );
  for(int j = 0; j < N_BESSEL; ++j){
    b_mean_error[j] /= ITERATIONS;
    double sub = j * (1.2 / N_BESSEL);
    printf("Mean error for Bessel = %.2f is %.10f\n", sub, b_mean_error[j]);
  }
  
  // Memory deallocations
  devices::free((void*)b_error);
  devices::free((void*)b_mean_error);
  devices::free((void*)rnd_val);

  // Some devices backends require a finalization
  devices::finalize();

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Duration = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

  return 0;
}