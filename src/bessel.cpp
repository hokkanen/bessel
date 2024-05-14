#include <chrono>
#include <cmath>
#include <iostream>
#include <climits>
#include <random>
#include <stdio.h>

/* Namespaces "comms" and "arch" declared here */
#include "./comms.h"
#include "./arch/arch_api.h"

/* Use matplotplusplus optionally for plotting */
#ifdef HAVE_MATPLOT
#include <matplot/matplot.h>
#endif

/* Set problem dimensions */
#define N_ITER 10000
#define N_POPU 10000
#define N_SAMPLE 50

int main(int argc, char *argv[])
{
  /* Initialize processes */
  comms::init_procs(&argc, &argv);
  /* Initialize devices */
  arch::init(comms::get_node_rank());
  const unsigned my_rank = comms::get_global_rank();
  {
    /* Set spacing and range for beta */
    constexpr unsigned n_beta = 40;
    constexpr float range_beta = 4.0f;

    /* Set timer */
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    /* Array for the mean squared errors (the sum of errors across iterations) */
    float mse[2 * n_beta] = {0};

    /* Use a non-deterministic random number generator for the master seed value */
    std::random_device rd;

    /* Use 64 bit Mersenne Twister 19937 generator */
    std::mt19937_64 mt(rd());

    /* Get a random unsigned long long from a uniform integer distribution (srand requires 32b uint) */
    std::uniform_int_distribution<unsigned long long> dist(0, UINT_MAX);

    /* Get the non-deterministic random master seed value */
    unsigned long long master_seed = dist(mt);

    /* Make sure the seed type is compatible with the chosen backend */
    auto seed_state = arch::random_state_seed(master_seed);

    /* Run the loop over iterations */
    arch::parallel_reduce(
        N_ITER, mse,
        ARCH_LOOP_LAMBDA(const unsigned iter, auto &lmse) {
          unsigned long long pos = (unsigned long long)iter * (unsigned long long)N_POPU;
          /* Calculate the mean and the squared sum of the population and the sample */
          float p_mean = 0.0f;
          float p_squared = 0.0f;
          float s_mean = 0.0f;
          float s_squared = 0.0f;
          {
            auto state = arch::random_state_init(seed_state, pos);
            for (unsigned i = 0; i < N_POPU; ++i)
            {
              float rnd_val = arch::random_float(state, 100.0f, 15.0f);
              p_mean += rnd_val;
              p_squared += rnd_val * rnd_val;
              if (i < N_SAMPLE)
              {
                s_mean += rnd_val;
                s_squared += rnd_val * rnd_val;
              }
              // if (iter < 1 && i < 3)
                // printf("Rank %u, iter: %u ,rnd_val[%u]: %.5f \n", my_rank, iter, i, rnd_val);
            }
            arch::random_state_free(seed_state, state);
          }
          p_mean /= N_POPU;
          s_mean /= N_SAMPLE;

          /* Calculate variance for the population and sum for the sample */
          float p_var = (p_squared - N_POPU * p_mean * p_mean) / N_POPU;
          float s_sum = s_squared - N_SAMPLE * s_mean * s_mean;
          // printf("p_var: %f\n",p_var);

          /* Calculate the mean squared error in the sample standard deviation and variance for different beta */
          float s_var[n_beta];
          for (unsigned j = 0; j < n_beta; ++j)
          {
            float sub = j * (range_beta / n_beta) - range_beta / 2.0f;
            s_var[j] = s_sum / (N_SAMPLE - sub);
            float diff_stdev = sqrtf(p_var) - sqrtf(s_var[j]);
            float diff_var = p_var - s_var[j];
            // printf("s_var[%u]: %f, error[iter: %u][sub: %f]: %f\n", j, s_var[j], iter, sub, sqrt(diff_var * diff_var));

            /* Sum the errors of each iteration */
            lmse[j] += (diff_stdev * diff_stdev);
            lmse[n_beta + j] += (diff_var * diff_var);
          }
        });

    /* Each process sends its values to reduction, root process collects the results */
    comms::reduce_procs(mse, 2 * n_beta);

#ifdef HAVE_MATPLOT
    // Define vectors for matplot
    std::vector<float> x;
    std::vector<float> y1;
    std::vector<float> y2;
#endif

    /* Divide the error sums to find the averaged errors for each tested beta value */
    if (my_rank == 0)
    {
      float *mse_stdev = &mse[0];
      float *mse_var = &mse[n_beta];
      for (unsigned j = 0; j < n_beta; ++j)
      {
        mse_stdev[j] /= (comms::get_global_procs() * N_ITER);
        mse_var[j] /= (comms::get_global_procs() * N_ITER);
        float rmse_stdev = sqrtf(mse_stdev[j]);
        float rmse_var = sqrtf(mse_var[j]);
        float sub = j * (range_beta / n_beta) - range_beta / 2.0f;
        printf("Beta = %.2f: RMSE for stdev = %.5f and var = %.5f\n", sub, rmse_stdev, rmse_var);
#ifdef HAVE_MATPLOT
        /* Add data for matplot */
        x.push_back(sub);
        y1.push_back(rmse_stdev);
        y2.push_back(rmse_var);
#endif
      }
    }

    /* Print timing */
    if (my_rank == 0)
    {
      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
      std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    }
  }
  /* Finalize devices */
  arch::finalize(my_rank);
  /* Finalize processes */
  comms::finalize_procs();

#ifdef HAVE_MATPLOT
  /* Plot the standard deviation error (1st line) */
  auto p1 = matplot::plot(x, y1);
  p1->display_name("stdev");
  matplot::hold(matplot::on);

  /* Plot the variance error (2nd line) */
  auto p2 = matplot::plot(x, y2);
  p2->use_y2(true).display_name("variance");
  matplot::hold(matplot::off);

  /* Create legend */
  auto l = matplot::legend();
  l->location(matplot::legend::general_alignment::topright);

  /* Set labels and style */
  matplot::title("Root mean squared error (RMSE) for Bessel's correction");
  matplot::xlabel("Beta");
  matplot::ylabel("RMSE");
  matplot::grid(matplot::on);

  /* Show plot */
  matplot::show();
#endif

  return 0;
}
