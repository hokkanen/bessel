#ifndef BESSEL_ARCH_SYCL_H
#define BESSEL_ARCH_SYCL_H

/* Include required headers */
#include <cstdio>
#include <sycl/sycl.hpp>
#include "oneapi/mkl/rng/device.hpp"

/* Define architecture-specific macros */
#define ARCH_LOOP_LAMBDA [=]

/* Set SYCL workgroup size */
#define ARCH_BLOCKSIZE 256

/* Namespace for architecture-specific functions */
namespace arch
{
    /* Device backend initialization */
    inline static sycl::queue init(int node_rank)
    {
        sycl::queue q;
        return q;
    }

    /* Device backend finalization */
    template <typename Q>
    inline static void finalize(Q &q, int rank)
    {
        printf("Rank %d, SYCL finalized.\n", rank);
    }

    /* Device function for memory allocation */
    template <typename Q>
    inline static void *allocate(Q &q, size_t bytes)
    {
        return sycl::malloc_shared<char>(bytes, q);
    }

    /* Device function for memory deallocation */
    template <typename Q>
    inline static void free(Q &q, void *ptr)
    {
        sycl::free(ptr, q);
    }

    /* Device-to-device memory copy */
    template <typename Q>
    inline static void memcpy_d2d(Q &q, void *dst, void *src, size_t bytes)
    {
        q.memcpy(dst, src, bytes).wait();
    }

    /* Atomic add function for both host and device use */
    template <typename T>
    inline static void atomic_add(T *array_loc, T value)
    {
        // Create an atomic reference to the memory location
        sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_ref(*array_loc);
        // Perform the atomic addition
        atomic_ref.fetch_add(value);
    }

    /* A function to make sure the seed is of right type */
    template <typename Q, typename T>
    inline static auto random_state_seed(Q &q, T &seed)
    {
        return (unsigned long long)seed;
    }

    /* A function for initializing a random number generator state */
    template <typename T>
    inline static auto random_state_init(T &seed, unsigned long long pos)
    {
        // Create an engine object
        oneapi::mkl::rng::device::philox4x32x10<> engine(seed, pos);
        // Create a distribution object
        oneapi::mkl::rng::device::gaussian<float> distr;
        // Return a pair of engine and distribution objects
        return std::make_pair(distr, engine);
    }

    /* A function for freeing a random number generator state */
    template <typename T, typename T2>
    inline static void random_state_free(T &seed, T2 &generator)
    {
        (void)seed;
        (void)generator;
    }

    /* A function for getting a random float from the standard distribution */
    template <typename T, typename T2>
    inline static T random_float(T2 &state, T mean, T stdev)
    {
        /* generate() gives a random float from a normal distribution with mean = 0 and stdev = 1 */
        float z0 = stdev * oneapi::mkl::rng::device::generate(state.first, state.second) + mean;
        return (T)z0;
    }

    /* Parallel for driver function for the SYCL loops */
    template <typename Q, typename Lambda>
    inline static void parallel_for(Q &q, unsigned loop_size, const Lambda loop_body)
    {
        // The actual kernel workgroup size should be a multiple of the block size
        unsigned kernel_size = ((loop_size + ARCH_BLOCKSIZE - 1) / ARCH_BLOCKSIZE) * ARCH_BLOCKSIZE;
        // Create a wrapper that extracts the thread index and checks for loop bounds
        auto lambda_wrapper = [=](const sycl::nd_item<1> nd_item)
        {
            unsigned index = nd_item.get_global_id(0);
            if (index < loop_size)
                loop_body(index);
        };
        // Evaluate the parallel for loop
        q.parallel_for(sycl::nd_range<1>{sycl::range<1>(kernel_size), sycl::range<1>(ARCH_BLOCKSIZE)},
                       lambda_wrapper)
            .wait();
    }

    // The reduction type for SYCL (using 'auto' for this in bessel.cpp fails with CUDA/KOKKOS backends)
    template <unsigned N>
    using Reducer = typename sycl::detail::reduction_impl<float, sycl::plus<void>, 1, N, true, float *>::reducer_type &;

    /* Parallel reduce driver function for the SYCL reductions */
    template <unsigned NReductions, typename Q, typename Lambda, typename T>
    inline static void parallel_reduce(Q &q, const unsigned loop_size, T (&sum)[NReductions], const Lambda loop_body)
    {
        // Allocate memory for the reduction data
        T *sum_buf = sycl::malloc_device<T>(NReductions, q);
        // Copy data from sum to sum_buf
        q.memcpy(sum_buf, sum, NReductions * sizeof(T)).wait();
        // The actual kernel workgroup size should be a multiple of the block size
        const unsigned kernel_size = ((loop_size + ARCH_BLOCKSIZE - 1) / ARCH_BLOCKSIZE) * ARCH_BLOCKSIZE;
        // Create a wrapper that extracts the thread index and checks for loop bounds
        auto lambda_wrapper = [=](const sycl::nd_item<1> nd_item, auto &lsum)
        {
            unsigned index = nd_item.get_global_id(0);
            if (index < loop_size)
                loop_body(index, lsum);
        };
        // Evaluate the parallel reduction loop
        q.parallel_for(sycl::nd_range<1>{sycl::range<1>(kernel_size), sycl::range<1>(ARCH_BLOCKSIZE)},
                       sycl::reduction(sycl::span<T, NReductions>(sum_buf, NReductions), T(0), std::plus<>()),
                       lambda_wrapper)
            .wait();
        // Copy data back from sum_buf to sum
        q.memcpy(sum, sum_buf, NReductions * sizeof(T)).wait();
        // Free the reduction data allocation
        sycl::free(sum_buf, q);
    }
}

#endif // !BESSEL_ARCH_SYCL_H
