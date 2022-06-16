#include <cstdio>
#include <cstdlib>
#include <random>
#include <string.h>

#define DEVICE_LAMBDA [=]

namespace devices
{
  inline static void init(int node_rank) {
    // Nothing needs to be done here
  }

  inline static void finalize(int rank) {
    printf("Rank %d, Host finalized.\n", rank);
  }

  inline static void* allocate(size_t bytes) {
    return malloc(bytes);
  }

  inline static void free(void* ptr) {
    ::free(ptr);
  }
  
  inline static void memcpy_d2d(void* dst, void* src, size_t bytes){
    memcpy(dst, src, bytes);
  }

  template <typename Lambda>
  inline static void parallel_for(int loop_size, Lambda loop_body) {
    for(int i = 0; i < loop_size; i++){
      loop_body(i);
    }
  }

  template <typename Lambda, typename T>
  inline static void parallel_reduce(const int loop_size, Lambda loop_body, T *sum) {
    for(int i = 0; i < loop_size; i++){
      loop_body(i, *sum);
    }
  }

  template <typename T>
  inline static void atomic_add(T *array_loc, T value){
    *array_loc += value;
  }

  template <typename T>
  inline static T random_float(unsigned long long seed, unsigned long long idx, T mean, T stdev){
    
    // Seed with std::seed_seq to reproduce the same random number with the same seed and idx (same as curand)
    std::seed_seq seedseq{seed, idx};

    // Use 64 bit Mersenne Twister 19937 generator
    std::mt19937 mt(seedseq);

    //std::uniform_real_distribution<float> dist(0,1);
    std::normal_distribution<float> dist(mean, stdev);

    return dist(mt);
  }
}
