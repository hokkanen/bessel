#ifndef BESSEL_DEVICES_CUDA_H
#define BESSEL_DEVICES_CUDA_H

#include <cstdio>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <curand_kernel.h>

#define CUDA_ERR(err) (cuda_error(err, __FILE__, __LINE__))
inline static void cuda_error(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("\n\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(1);
  }
}

#define DEVICE_LAMBDA [=] __host__ __device__

namespace devices
{
  __forceinline__ static void init(int node_rank) {
    int num_devices = 0;
    CUDA_ERR(cudaGetDeviceCount(&num_devices));
    CUDA_ERR(cudaSetDevice(node_rank % num_devices));
  }

  __forceinline__ static void finalize(int rank) {
    printf("Rank %d, CUDA finalized.\n", rank);
  }

  __forceinline__ static void* allocate(size_t bytes) {
    void* ptr;
    CUDA_ERR(cudaMallocManaged(&ptr, bytes));
    return ptr;
  }

  __forceinline__ static void free(void* ptr) {
    CUDA_ERR(cudaFree(ptr));
  }

  __forceinline__ static void memcpy_d2d(void* dst, void* src, size_t bytes){
    CUDA_ERR(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
  }

  template <typename T>
  __host__ __device__ __forceinline__ static void atomic_add(T *array_loc, T value){
    // Define this function depending on whether it runs on GPU or CPU
    #ifdef __CUDA_ARCH__
      atomicAdd(array_loc, value);
    #else
      *array_loc += value;
    #endif
  }

  template <typename T>
  __host__ __device__ __forceinline__ static T random_float(unsigned long long seed, unsigned long long seq, int idx, T mean, T stdev){    
    T var = 0;
    #ifdef __CUDA_ARCH__
      curandStatePhilox4_32_10_t state;
  
      // curand_init() reproduces the same random number with the same seed and seq
      curand_init(seed, seq, 0, &state);
  
      // curand_normal() gives a random float from a normal distribution with mean = 0 and stdev = 1
      var = stdev * curand_normal(&state) + mean;
    #endif
    return var;
  }

  template <typename Lambda> 
  __global__ static void for_kernel(Lambda lambda, const int loop_size)
  {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < loop_size)
    {
      lambda(i);
    }
  }

  /* A general device kernel for reductions */
  template <uint NReductions, typename Lambda, typename T>
  __global__ static void reduction_kernel(Lambda loop_body, const T *init_val, T *rslt, const uint n_total)
  {
    /* Specialize BlockReduce for a 1D block of ARCH_BLOCKSIZE_R threads of type `T` */
    typedef cub::BlockReduce<T, 64> BlockReduce;
  
    /* Static shared memory declaration */
    __shared__ typename BlockReduce::TempStorage temp_storage[NReductions];
  
    /* Get the global 1D thread index*/
    const uint idx_glob = blockIdx.x * blockDim.x + threadIdx.x;
  
    /* Check the loop limits*/
    if (idx_glob < n_total) {
  
      /* Static thread data declaration */
      T thread_data[NReductions];
    
      /* Initialize thread data values */
      for(uint i = 0; i < NReductions; i++)
        thread_data[i] = init_val[i];
    
      /* Evaluate the loop body */
      loop_body(idx_glob, thread_data);
    
      /* Perform reductions */
      for(uint i = 0; i < NReductions; i++){
        /* Compute the block-wide sum for thread 0 which stores it */
        T aggregate = BlockReduce(temp_storage[i]).Sum(thread_data[i]);
        /* The first thread of each block stores the block-wide aggregate atomically */
        if(threadIdx.x == 0) 
          atomicAdd(&rslt[i], aggregate);
      }
    }
  }

  template <typename Lambda>
  __forceinline__ static void parallel_for(int loop_size, Lambda loop_body) {
    const int blocksize = 64;
    const int gridsize = (loop_size - 1 + blocksize) / blocksize;
    for_kernel<<<gridsize, blocksize>>>(loop_body, loop_size);
    CUDA_ERR(cudaStreamSynchronize(0));
  }

  /* Parallel reduce driver function for the CUDA reductions */
  template <uint NReductions, typename Lambda, typename T>
  __forceinline__ static void parallel_reduce(const uint loop_size, T (&sum)[NReductions], Lambda loop_body) {
  
    /* Set the kernel dimensions */
    const uint blocksize = 64;
    const uint gridsize = (loop_size - 1 + blocksize) / blocksize;
  
    /* Create a device buffer for the reduction results */
    T* d_buf;
    CUDA_ERR(cudaMalloc(&d_buf, NReductions*sizeof(T)));
    CUDA_ERR(cudaMemcpy(d_buf, sum, NReductions*sizeof(T), cudaMemcpyHostToDevice));
    
    /* Create a device buffer to transfer the initial values to device */
    T* d_const_buf;
    CUDA_ERR(cudaMalloc(&d_const_buf, NReductions*sizeof(T)));
    CUDA_ERR(cudaMemcpy(d_const_buf, d_buf, NReductions*sizeof(T), cudaMemcpyDeviceToDevice));
  
    /* Call the kernel (the number of reductions known at compile time) */
    reduction_kernel<NReductions><<<gridsize, blocksize>>>(loop_body, d_const_buf, d_buf, loop_size);
    /* Synchronize after kernel call */
    CUDA_ERR(cudaStreamSynchronize(0));
    
    /* Copy the results back to host and free the allocated memory back to pool*/
    CUDA_ERR(cudaMemcpy(sum, d_buf, NReductions*sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_ERR(cudaFree(d_buf));
    CUDA_ERR(cudaFree(d_const_buf));
  }
}

#endif // !BESSEL_DEVICES_CUDA_H
