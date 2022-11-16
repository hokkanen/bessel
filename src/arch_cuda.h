#ifndef BESSEL_ARCH_CUDA_H
#define BESSEL_ARCH_CUDA_H

/* Include required headers */
#include <cstdio>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <curand_kernel.h>

/* Some of the macros and functions that can be compiled with a C compiler */
extern "C"{  
    
  /* Set CUDA blocksize */
  #define ARCH_BLOCKSIZE 256

  /* Define the CUDA error checking macro */
  #define CUDA_ERR(err) (cuda_error(err, __FILE__, __LINE__))
  inline static void cuda_error(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
      printf("\n\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
      exit(1);
    }
  }
  
  /* Device backend initialization */
  __forceinline__ static void arch_init(int node_rank) {
    int num_devices = 0;
    CUDA_ERR(cudaGetDeviceCount(&num_devices));
    CUDA_ERR(cudaSetDevice(node_rank % num_devices));
  }

  /* Device backend finalization */
  __forceinline__ static void arch_finalize(int rank) {
    printf("Rank %d, CUDA finalized.\n", rank);
  }

  /* Device function for memory allocation */
  __forceinline__ static void* arch_allocate(size_t bytes) {
    void* ptr;
    CUDA_ERR(cudaMallocManaged(&ptr, bytes));
    return ptr;
  }

  /* Device function for memory deallocation */
  __forceinline__ static void arch_free(void* ptr) {
    CUDA_ERR(cudaFree(ptr));
  }

  /* Device-to-device memory copy */
  __forceinline__ static void arch_memcpy_d2d(void* dst, void* src, size_t bytes){
    CUDA_ERR(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
  }

  /* The macros and functions that require C++ compiler */
  extern "C++"{

    /* Atomic add function for both host and device use */
    template <typename T>
    __host__ __device__ __forceinline__ static void arch_atomic_add(T *array_loc, T value){
      /* Define this function depending on whether it runs on GPU or CPU */
      #ifdef __CUDA_ARCH__
        atomicAdd(array_loc, value);
      #else
        *array_loc += value;
      #endif
    }
  
    /* A function for getting a random float from the standard distribution */
    template <typename T>
    __host__ __device__ __forceinline__ static T arch_random_float(unsigned long long seed, unsigned long long seq, unsigned int idx, T mean, T stdev){    
      T var = 0;
      #ifdef __CUDA_ARCH__
        curandStatePhilox4_32_10_t state;
    
        /* curand_init() reproduces the same random number with the same seed and seq */
        curand_init(seed, seq, 0, &state);
    
        /* curand_normal() gives a random float from a normal distribution with mean = 0 and stdev = 1 */
        var = stdev * curand_normal(&state) + mean;
      #endif
      return var;
    }

    /* A general device kernel for simple for-loops (internal use only) */
    template <typename Lambda> 
    __global__ static void _arch_for_kernel(Lambda lambda, const unsigned int loop_size)
    {
      const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
      if(i < loop_size)
      {
        lambda(i);
      }
    }

    /* A general device kernel for reductions (internal use only) */
    template <unsigned int NReductions, typename Lambda, typename T>
    __global__ static void _arch_reduction_kernel(Lambda loop_body, const T *init_val, T *rslt, const unsigned int n_total)
    {
      /* Specialize BlockReduce for a 1D block of ARCH_BLOCKSIZE_R threads of type `T` */
      typedef cub::BlockReduce<T, ARCH_BLOCKSIZE> BlockReduce;
    
      /* Static shared memory declaration */
      __shared__ typename BlockReduce::TempStorage temp_storage[NReductions];
    
      /* Static thread data declaration */
      T thread_data[NReductions] = {0};
    
      /* Get the global 1D thread index */
      const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
      /* Check the loop limits */
      if (idx < n_total){
        /* Initialize thread data values */
        for(unsigned int i = 0; i < NReductions; i++)
          thread_data[i] = init_val[i];
      
        /* Evaluate the loop body */
        loop_body(idx, thread_data);
      }
  
      /* Perform reductions */
      for(unsigned int i = 0; i < NReductions; i++){
        /* Compute the block-wide sum for thread 0 which stores it */
        T aggregate = BlockReduce(temp_storage[i]).Sum(thread_data[i]);
        /* The first thread of each block stores the block-wide aggregate atomically */
        if(threadIdx.x == 0) 
          atomicAdd(&rslt[i], aggregate);
      }
    }

    /* Parallel reduce driver function for the CUDA reductions (internal use only) */
    template <unsigned int NReductions, typename Lambda, typename T>
    __forceinline__ static void _arch_parallel_reduce_driver(const unsigned int loop_size, T (&sum)[NReductions], Lambda loop_body) {
    
      /* Set the kernel dimensions */
      const unsigned int blocksize = ARCH_BLOCKSIZE;
      const unsigned int gridsize = (loop_size - 1 + blocksize) / blocksize;
    
      /* Create a device buffer for the reduction results */
      T* d_buf;
      CUDA_ERR(cudaMalloc(&d_buf, NReductions*sizeof(T)));
      CUDA_ERR(cudaMemcpy(d_buf, sum, NReductions*sizeof(T), cudaMemcpyHostToDevice));
      
      /* Create a device buffer to transfer the initial values to device */
      T* d_const_buf;
      CUDA_ERR(cudaMalloc(&d_const_buf, NReductions*sizeof(T)));
      CUDA_ERR(cudaMemcpy(d_const_buf, d_buf, NReductions*sizeof(T), cudaMemcpyDeviceToDevice));
    
      /* Call the kernel (the number of reductions known at compile time) */
      _arch_reduction_kernel<NReductions><<<gridsize, blocksize>>>(loop_body, d_const_buf, d_buf, loop_size);
      /* Synchronize after kernel call */
      CUDA_ERR(cudaStreamSynchronize(0));
      
      /* Copy the results back to host and free the allocated memory back to pool*/
      CUDA_ERR(cudaMemcpy(sum, d_buf, NReductions*sizeof(T), cudaMemcpyDeviceToHost));
      CUDA_ERR(cudaFree(d_buf));
      CUDA_ERR(cudaFree(d_const_buf));
    }

    /* Parallel for driver macro for the CUDA loops */
    #define arch_parallel_for(loop_size, inc, loop_body)                       \
    {                                                                          \
      const unsigned int blocksize = ARCH_BLOCKSIZE;                           \
      const unsigned int gridsize = (loop_size - 1 + blocksize) / blocksize;   \
      auto lambda_body = [=] __host__ __device__ (unsigned int inc) loop_body; \
      _arch_for_kernel<<<gridsize, blocksize>>>(lambda_body, loop_size);       \
      CUDA_ERR(cudaStreamSynchronize(0));                                      \
      (void)inc;                                                               \
    }

    /* Parallel reduce wrapper macro for the CUDA reductions */
    #define arch_parallel_reduce(loop_size, inc, sum, loop_body)               \
    {                                                                          \
      auto lambda_body = [=] __host__ __device__ (unsigned int inc, float *sum)\
        loop_body;                                                             \
      _arch_parallel_reduce_driver(loop_size, sum, lambda_body);               \
      (void)inc;                                                               \
    }
  }
}

#endif // !BESSEL_ARCH_CUDA_H
