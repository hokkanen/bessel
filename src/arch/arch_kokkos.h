#ifndef BESSEL_ARCH_KOKKOS_H
#define BESSEL_ARCH_KOKKOS_H

/* Include required headers */
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

/* Define architecture-specific macros */
#define ARCH_LOOP_LAMBDA KOKKOS_LAMBDA

/* Namespace for architecture-specific functions */
namespace arch
{
    /* Device backend initialization */
    inline static void init(int node_rank)
    {
        Kokkos::initialize(Kokkos::InitializationSettings()
                               .set_device_id(node_rank));
    }

    /* Device backend finalization */
    inline static void finalize(int rank)
    {
        Kokkos::finalize();
        printf("Rank %d, Kokkos finalized.\n", rank);
    }

    /* Device function for memory allocation */
    inline static void *allocate(size_t bytes)
    {
        return Kokkos::kokkos_malloc<Kokkos::SharedSpace>(bytes);
    }

    /* Device function for memory deallocation */
    inline static void free(void *ptr)
    {
        Kokkos::kokkos_free<Kokkos::SharedSpace>(ptr);
    }

    /* Device-to-device memory copy */
    inline static void memcpy_d2d(void *dst, void *src, size_t bytes)
    {
        Kokkos::View<char *> dst_view((char *)dst, bytes);
        Kokkos::View<char *> src_view((char *)src, bytes);
        Kokkos::deep_copy(dst_view, src_view);
    }

    /* Atomic add function for both host and device use */
    template <typename T>
    KOKKOS_INLINE_FUNCTION static void atomic_add(T *array_loc, T value)
    {
        Kokkos::atomic_add(array_loc, value);
    }

    /* A function to make sure the seed is of right type (returns rng pool for Kokkos) */
    template <typename T>
    inline static auto random_state_seed(T &seed)
    {
        Kokkos::Random_XorShift64_Pool<> rng_pool(seed);
        return rng_pool;
    }

    /* A function for initializing a random number generator state */
    template <typename T>
    KOKKOS_INLINE_FUNCTION static auto random_state_init(T &seed, unsigned long long pos)
    {
        return seed.get_state();
    }

    /* A function for freeing a random number generator state */
    template <typename T, typename T2>
    KOKKOS_INLINE_FUNCTION static void random_state_free(T &seed, T2 &generator)
    {
        seed.free_state(generator);
    }

    /* A function for getting a random float from the standard distribution */
    template <typename T, typename T2>
    KOKKOS_INLINE_FUNCTION static T random_float(T2 &generator, T mean, T stdev)
    {
        /* Use Box Muller algorithm to get a float from a normal distribution */
        const float two_pi = 2.0f * M_PI;
        const float u1 = generator.frand(0., 1.);
        const float u2 = generator.frand(0., 1.);
        const float factor = stdev * sqrtf(-2.0f * logf(u1));
        const float trig_arg = two_pi * u2;

        /* Box Muller algorithm produces two random normally distributed floats, z0 and z1 */
        float z0 = factor * cosf(trig_arg) + mean; /* Need only one */
        // float z1 = factor * sinf (trig_arg) + mean;
        return (T)z0;
    }

    /* Parallel for driver function for the Kokkos loops */
    template <typename Lambda>
    inline static void parallel_for(unsigned loop_size, Lambda loop_body)
    {
        Kokkos::parallel_for(loop_size, loop_body);
        Kokkos::fence();
    }

    /* Aux struct to perform reductions into an array with Kokkos */
    template <size_t N>
    struct AuxReducer
    {
        float values[N];
        KOKKOS_INLINE_FUNCTION void operator+=(AuxReducer const &other)
        {
            for (unsigned i = 0; i < N; ++i)
            {
                values[i] += other.values[i];
            }
        }
    };

    /* Parallel reduce driver function for the Kokkos reductions */
    template <unsigned NReductions, typename Lambda, typename T>
    inline static void parallel_reduce(const unsigned loop_size, T (&sum)[NReductions], Lambda loop_body)
    {
        // Copy initial values to the AuxReducer object
        AuxReducer<NReductions> aux_sum;
        for (int i = 0; i < NReductions; ++i)
            aux_sum.values[i] = sum[i];

        // Create a wrapper lambda that takes in AuxReducer
        auto lambda_wrapper = KOKKOS_LAMBDA(const unsigned iter, AuxReducer<NReductions> &laux_sum)
        {
            loop_body(iter, &(laux_sum.values[0]));
        };

        // Run the Kokkos reduction
        Kokkos::parallel_reduce(loop_size, lambda_wrapper, aux_sum);
        Kokkos::fence();

        // Copy values from the AuxReducer object back to 'sum'
        for (int i = 0; i < NReductions; ++i)
            sum[i] = aux_sum.values[i];
    }
}

#endif // !BESSEL_ARCH_KOKKOS_H
