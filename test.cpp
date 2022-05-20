/******************************************************************************

                              Online C++ Compiler.
               Code, Compile, Run and Debug C++ program online.
Write your code in this editor and press "Run" button to compile and execute it.

*******************************************************************************/

#define N_BESSEL 20
#define ITERATIONS 1000
#define POPULATION 100000
#define SAMPLE 2

#include <cmath>
#include <iostream>
#include <random>
#include <stdio.h>

using namespace std;

int main()
{
double b_error[ITERATIONS][N_BESSEL + 1]; 
double b_mean_error[N_BESSEL + 1];
   double rnd_val[POPULATION];

    std::random_device rd;
    std::mt19937 mt(rd());
   //std::uniform_real_distribution<double> dist(0,1);
   std::normal_distribution<double> dist(100,15);
   
   for(int j = 0; j < N_BESSEL + 1; ++j){
       b_mean_error[j] = 0.0;
   }

#pragma omp parallel for
 for(int iter = 0; iter < ITERATIONS; ++iter){
   for(int i = 0; i < POPULATION; ++i){
     rnd_val[i] = dist(mt);
     if(iter == 1 && i < 3) printf("rnd_val[%d]: %.5f \n", i, rnd_val[i]);
   }
     
     double p_mean = 0.0;
     double s_mean = 0.0;
     
     for(int i = 0; i < POPULATION; ++i){
       p_mean += rnd_val[i];
       if(i < SAMPLE) s_mean += rnd_val[i];
     }
     
     p_mean /= POPULATION;
     s_mean /= SAMPLE;
     

     
     double b_stdev[N_BESSEL + 1];
          double b_sum = 0.0;
     double p_stdev = 0.0;

     
     for(int i = 0; i < POPULATION; ++i){
         
       double p_diff = rnd_val[i] - p_mean;
       p_stdev += p_diff * p_diff;
       if(i < SAMPLE){
         double b_diff = rnd_val[i] - s_mean;
         b_sum += b_diff * b_diff;   
       }
     }
          p_stdev /= POPULATION;
     p_stdev = sqrt(p_stdev);
     //printf("p_stdev: %f\n",p_stdev);
     
     for(int j = 0; j < N_BESSEL + 1; ++j){
         double sub = 2.0 - j * (2.0 / N_BESSEL);
       b_stdev[j] = b_sum / (SAMPLE - sub);
       b_stdev[j] = sqrt(b_stdev[j]);
       double diff = p_stdev - b_stdev[j];
       b_error[iter][j] = sqrt(diff * diff);
       //printf("b_stdev[%d]: %f, b_error[iter: %d][sub: %f]: %f\n", j, b_stdev[j], iter, sub, b_error[iter][j]);
       
     }
      
          
          
 }
 for(int iter = 0; iter < ITERATIONS; ++iter){
     for(int j = 0; j < N_BESSEL + 1; ++j){
         
         b_mean_error[j] += b_error[iter][j];
     }
 }
 for(int j = 0; j < N_BESSEL + 1; ++j){
     b_mean_error[j] /= ITERATIONS;
     double sub = 2.0 - j * (2.0 / N_BESSEL);
     printf("Mean error for Bessel = %.2f is %.10f\n", sub, b_mean_error[j]);
 }
     
    return 0;
}