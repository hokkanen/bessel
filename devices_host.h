#include <cstdlib>
#include <random>
#include <string.h>

#define DEVICE_LAMBDA [=]

namespace devices
{
  inline void init(int node_rank) {
    // Nothing needs to be done here
    (void) node_rank;
  }

  inline void finalize() {
    printf("Host finalized.\n");
  }

  inline void* allocate(size_t bytes) {
    return malloc(bytes);
  }

  inline void free(void* ptr) {
    ::free(ptr);
  }
  
  inline void memcpy_d2d(void* dst, void* src, size_t bytes){
    memcpy(dst, src, bytes);
  }

  template <typename Lambda>
  inline void parallel_for(int loop_size, Lambda loop_body) {
    #pragma parallel for  
    for(int i = 0; i < loop_size; i++){
      loop_body(i);
    }
  }

  template <typename Lambda, typename T>
  inline void parallel_reduce(const int loop_size, Lambda loop_body, T *sum) {
    for(int i = 0; i < loop_size; i++){
      loop_body(i, *sum);
    }
  }

  template <typename T>
  inline static void atomic_add(T *array_loc, T value){
    *array_loc += value;
  }

  template <typename T>
  inline T random_double(unsigned long long seed, unsigned long long idx, T mean, T stdev){
    
    std::random_device rd;
    std::mt19937 mt(rd());
    //std::uniform_real_distribution<double> dist(0,1);
    std::normal_distribution<double> dist(mean, stdev);

    return dist(mt);
  }
}
