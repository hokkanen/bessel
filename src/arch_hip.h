#ifndef BESSEL_ARCH_HIP_H
#define BESSEL_ARCH_HIP_H

/* Include required headers */
#include <cstdio>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <hiprand_kernel.h>

/* Allow using Umpire memory manager for memory pools */
#if defined(HAVE_UMPIRE)
  #include "umpire/interface/c_fortran/umpire.h"
#endif

/* All macros and functions require a C++ compiler (HIP API does not support C) */

/* Set HIP blocksize */
#define ARCH_BLOCKSIZE 256

/* Define the HIP error checking macro */
#define HIP_ERR(err) (hip_error(err, __FILE__, __LINE__))
inline static void hip_error(hipError_t err, const char *file, int line) {
  if (err != hipSuccess) {
    printf("\n\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
    exit(1);
  }
}

/* Device backend initialization */
__forceinline__ static void devices_init(int node_rank) {
  int num_devices = 0;
  HIP_ERR(hipGetDeviceCount(&num_devices));
  HIP_ERR(hipSetDevice(node_rank % num_devices));
  #if defined(HAVE_UMPIRE)
    umpire_resourcemanager rm;
    umpire_resourcemanager_get_instance(&rm);
    umpire_allocator allocator;
    umpire_resourcemanager_get_allocator_by_name(&rm, "UM", &allocator);
    umpire_allocator pool;
    umpire_resourcemanager_make_allocator_quick_pool(&rm, "pool", allocator, 1024, 1024, &pool);
  #endif
}

/* Device backend finalization */
__forceinline__ static void devices_finalize(int rank) {
  printf("Rank %d, HIP finalized.\n", rank);
}

/* Device function for memory allocation */
__forceinline__ static void* devices_allocate(size_t bytes) {
  void* ptr;
  #if defined(HAVE_UMPIRE)
    umpire_resourcemanager rm;
    umpire_resourcemanager_get_instance(&rm);
    umpire_allocator pool;
    umpire_resourcemanager_get_allocator_by_name(&rm, "pool", &pool);
    ptr = umpire_allocator_allocate(&pool, bytes);
  #else
    HIP_ERR(hipMallocManaged(&ptr, bytes));
  #endif
  return ptr;
}

/* Device function for memory deallocation */
__forceinline__ static void devices_free(void* ptr) {
  #if defined(HAVE_UMPIRE)
    umpire_resourcemanager rm;
    umpire_resourcemanager_get_instance(&rm);
    umpire_allocator pool;
    umpire_resourcemanager_get_allocator_by_name(&rm, "pool", &pool);
    umpire_allocator_deallocate(&pool, ptr);
  #else
    HIP_ERR(hipFree(ptr));
  #endif
}

/* Device-to-device memory copy */
__forceinline__ static void devices_memcpy_d2d(void* dst, void* src, size_t bytes){
  HIP_ERR(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice));
}

/* Atomic add function for both host and device use */
template <typename T>
__host__ __device__ __forceinline__ static void devices_atomic_add(T *array_loc, T value){
    /* Define this function depending on whether it runs on GPU or CPU */
  #if __HIP_DEVICE_COMPILE__
    atomicAdd(array_loc, value);
  #else
    *array_loc += value;
  #endif
}

/* A function for getting a random float from the standard distribution */
template <typename T>
__host__ __device__ __forceinline__ static T devices_random_float(unsigned long long seed, unsigned long long seq, unsigned int idx, T mean, T stdev){    
T var = 0;
  #if __HIP_DEVICE_COMPILE__
    hiprandStatePhilox4_32_10_t state;

    /* hiprand_init() reproduces the same random number with the same seed and seq */
    hiprand_init(seed, seq, 0, &state);

    /* hiprand_normal() gives a random float from a normal distribution with mean = 0 and stdev = 1 */
    var = stdev * hiprand_normal(&state) + mean;
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
  typedef hipcub::BlockReduce<T, ARCH_BLOCKSIZE> BlockReduce;

  /* Static shared memory declaration */
  __shared__ typename BlockReduce::TempStorage temp_storage[NReductions];

  /* Get the global 1D thread index*/
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  /* Check the loop limits*/
  if (idx < n_total) {

    /* Static thread data declaration */
    T thread_data[NReductions];
  
    /* Initialize thread data values */
    for(unsigned int i = 0; i < NReductions; i++)
      thread_data[i] = init_val[i];
  
    /* Evaluate the loop body */
    loop_body(idx, thread_data);
  
    /* Perform reductions */
    for(unsigned int i = 0; i < NReductions; i++){
      /* Compute the block-wide sum for thread 0 which stores it */
      T aggregate = BlockReduce(temp_storage[i]).Sum(thread_data[i]);
      /* The first thread of each block stores the block-wide aggregate atomically */
      if(threadIdx.x == 0) 
        atomicAdd(&rslt[i], aggregate);
    }
  }
}

/* Parallel reduce driver function for the HIP reductions (internal use only) */
template <unsigned int NReductions, typename Lambda, typename T>
__forceinline__ static void _arch_parallel_reduce_driver(const unsigned int loop_size, T (&sum)[NReductions], Lambda loop_body) {

  /* Set the kernel dimensions */
  const unsigned int blocksize = ARCH_BLOCKSIZE;
  const unsigned int gridsize = (loop_size - 1 + blocksize) / blocksize;

  /* Create a device buffer for the reduction results */
  T* d_buf;
  HIP_ERR(hipMalloc(&d_buf, NReductions*sizeof(T)));
  HIP_ERR(hipMemcpy(d_buf, sum, NReductions*sizeof(T), hipMemcpyHostToDevice));
  
  /* Create a device buffer to transfer the initial values to device */
  T* d_const_buf;
  HIP_ERR(hipMalloc(&d_const_buf, NReductions*sizeof(T)));
  HIP_ERR(hipMemcpy(d_const_buf, d_buf, NReductions*sizeof(T), hipMemcpyDeviceToDevice));

  /* Call the kernel (the number of reductions known at compile time) */
  _arch_reduction_kernel<NReductions><<<gridsize, blocksize>>>(loop_body, d_const_buf, d_buf, loop_size);
  /* Synchronize after kernel call */
  HIP_ERR(hipStreamSynchronize(0));
  
  /* Copy the results back to host and free the allocated memory back to pool*/
  HIP_ERR(hipMemcpy(sum, d_buf, NReductions*sizeof(T), hipMemcpyDeviceToHost));
  HIP_ERR(hipFree(d_buf));
  HIP_ERR(hipFree(d_const_buf));
}

/* Parallel for driver macro for the HIP loops */
#define devices_parallel_for(loop_size, inc, loop_body)                    \
{                                                                          \
  const unsigned int blocksize = ARCH_BLOCKSIZE;                           \
  const unsigned int gridsize = (loop_size - 1 + blocksize) / blocksize;   \
  auto lambda_body = [=] __host__ __device__ (unsigned int inc) loop_body; \
  _arch_for_kernel<<<gridsize, blocksize>>>(lambda_body, loop_size);       \
  HIP_ERR(hipStreamSynchronize(0));                                        \
  (void)inc;                                                               \
}

/* Parallel reduce wrapper macro for the HIP reductions */
#define devices_parallel_reduce(loop_size, inc, sum, loop_body)            \
{                                                                          \
  auto lambda_body = [=] __host__ __device__ (unsigned int inc, float *sum)\
    loop_body;                                                             \
  _arch_parallel_reduce_driver(loop_size, sum, lambda_body);               \
  (void)inc;                                                               \
}

#endif // !BESSEL_ARCH_HIP_H
