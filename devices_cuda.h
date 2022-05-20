#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>

#define DEVICE_LAMBDA [=] __host__ __device__

namespace devices
{
  __forceinline__ void init() {
    cudaSetDevice(0);
  }

  __forceinline__ void finalize() {
    printf("CUDA finalized.\n");
  }

  __forceinline__ void* allocate(size_t bytes) {
    void* ptr;
    cudaMallocManaged(&ptr, bytes);
    return ptr;
  }

  __forceinline__ void free(void* ptr) {
    cudaFree(ptr);
  }

  __forceinline__ void memcpy_d2d(void* dst, void* src, size_t bytes){
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
  }

  template <typename LambdaBody> 
  __global__ static void cudaKernel(LambdaBody lambda, const int loop_size)
  {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < loop_size)
    {
      lambda(i);
    }
  }

  template <typename T>
  __forceinline__ void parallel_for(int loop_size, T loop_body) {
    const int blocksize = 64;
    const int gridsize = (loop_size - 1 + blocksize) / blocksize;
    cudaKernel<<<gridsize, blocksize>>>(loop_body, loop_size);
    cudaStreamSynchronize(0);
  }

  template <typename Lambda, typename T>
  __forceinline__ void parallel_reduce(const int loop_size, Lambda loop_body, T *sum) {
    const int blocksize = 64;
    const int gridsize = (loop_size - 1 + blocksize) / blocksize;

    T* buf;
    cudaMalloc(&buf, sizeof(T));
    cudaMemcpy(buf, sum, sizeof(T), cudaMemcpyHostToDevice);

    auto lambda_outer = 
      DEVICE_LAMBDA(const int i)
      {
        T lsum = 0;
        loop_body(i, lsum);
        atomicAdd(buf, lsum);
      };

    cudaKernel<<<gridsize, blocksize>>>(lambda_outer, loop_size);
    cudaStreamSynchronize(0);

    cudaMemcpy(sum, buf, sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(buf);
  }

  template <typename T>
  __host__ __device__ __forceinline__ static void atomic_add(T *array_loc, T value){
      //Define this function depending on whether it runs on GPU or CPU
#ifdef __CUDA_ARCH__
      atomicAdd(array_loc, value);
#else
      *array_loc += value;
#endif
  }

  template <typename T>
  __host__ __device__ T random_double(unsigned long long seed, int idx, T mean, T stdev){    
    
    T var = 0;
#ifdef __CUDA_ARCH__
    curandState state;
    curand_init(seed, idx, 0, &state);
    var = stdev * curand_normal_double(&state);
#endif
    return mean + var;
  }
}
