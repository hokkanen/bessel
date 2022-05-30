#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>

#define CUDA_ERR(err) (cuda_error(err, __FILE__, __LINE__))
static inline void cuda_error(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("\n\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(1);
	}
}

#define DEVICE_LAMBDA [=] __host__ __device__

namespace devices
{
  __forceinline__ void init(int node_rank) {
    int num_devices = 0;
    CUDA_ERR(cudaGetDeviceCount(&num_devices));
    CUDA_ERR(cudaSetDevice(node_rank % num_devices));
  }

  __forceinline__ void finalize() {
    printf("CUDA finalized.\n");
  }

  __forceinline__ void* allocate(size_t bytes) {
    void* ptr;
    CUDA_ERR(cudaMallocManaged(&ptr, bytes));
    return ptr;
  }

  __forceinline__ void free(void* ptr) {
    CUDA_ERR(cudaFree(ptr));
  }

  __forceinline__ void memcpy_d2d(void* dst, void* src, size_t bytes){
    CUDA_ERR(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
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
    CUDA_ERR(cudaStreamSynchronize(0));
  }

  template <typename Lambda, typename T>
  __forceinline__ void parallel_reduce(const int loop_size, Lambda loop_body, T *sum) {
    const int blocksize = 64;
    const int gridsize = (loop_size - 1 + blocksize) / blocksize;

    T* buf;
    CUDA_ERR(cudaMalloc(&buf, sizeof(T)));
    CUDA_ERR(cudaMemcpy(buf, sum, sizeof(T), cudaMemcpyHostToDevice));

    auto lambda_outer = 
      DEVICE_LAMBDA(const int i)
      {
        T lsum = 0;
        loop_body(i, lsum);
        atomicAdd(buf, lsum);
      };

    cudaKernel<<<gridsize, blocksize>>>(lambda_outer, loop_size);
    CUDA_ERR(cudaStreamSynchronize(0));

    CUDA_ERR(cudaMemcpy(sum, buf, sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_ERR(cudaFree(buf));
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
  __host__ __device__ T random_double(unsigned long long seed, unsigned long long idx, T mean, T stdev){    
    
    T var = 0;
#ifdef __CUDA_ARCH__
    curandState state;

    // curand_init() reproduces the same random number with the same seed and idx
    curand_init(seed, idx, 0, &state);

    // curand_normal_double() gives a random double from a normal distribution with mean = 0 and stdev = 1
    var = stdev * curand_normal_double(&state);
#endif
    return mean + var;
  }
}
