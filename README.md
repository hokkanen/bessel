# Monte Carlo multi-architecture simulation

This example uses the Monte Carlo method to simulate the values of Bessel's correction that minimize the root mean squared error in the calculation of the sample standard deviation and variance for the chosen sample and population sizes. The sample variance is typically calculated as $$s^2 = \frac{1}{N - \beta}\sum_{i=1}^{N}(x_i - \bar{x})^2$$ where $N$ is the number of samples, $\bar{x}$ is the sample mean, and $\beta = 1$ is called Bessel's correction to make the estimator unbiased. However, this is not the only useful value; according to [wikipedia](https://en.wikipedia.org/wiki/Variance#Population_variance_and_sample_variance), common values are $\beta = 0$ for the biased sample variance, $\beta = 1$ for unbiased sample variance, $\beta = -1$ for minimization of root mean squared error for the sample variance of a normal distribution, and $\beta = 1.5$ for approximate minimization of bias in unbiased estimation of standard deviation for the normal distribution.

The simulation calculates the root mean squared error for different values of $\beta$. The implementation evaluates the following sum in a single loop by $$\sum_{i=1}^{N}(x_i - \bar{x})^2 = \sum_{i=1}^{N}x_i^2 - N \bar{x}^2.$$ The sample standard deviation is then simply calculated by $$s = \sqrt{s^2} = \sqrt{\frac{1}{N - \beta}\sum_{i=1}^{N}x_i^2 - N \bar{x}^2}$$ after which the root mean squared errors are obtained by comparing these results to the exact population variance and standard deviation.


The implementation uses a special construct for the parallel loops in [bessel.cpp](src/bessel.cpp). In the `c` branch (C example), this is based on a preprocessor macro, whereas the `cpp` branch (C++ example) is based on a lambda function, an approach similar to some accelerator frameworks such as SYCL, Kokkos, RAJA, and others. Either option allows conditional compilation of the loops for multiple architectures while keeping the source code clean and readable. An example of the usage of curand and hiprand and Kokkos random number generation libraries inside a GPU kernel are given in [arch_cuda.h](src/arch/arch_cuda.h), [arch_hip.h](src/arch/arch_hip.h) and [arch_kokkos.h](src/arch/arch_kokkos.h). Furthermore, in [arch_host.h](src/arch/arch_host.h), sequential host execution is implemented together with optional OpenMP offloading that is combined with curand random number generator and can be compiled with NVIDIA `nvc++` compiler.

The code can be conditionally compiled for either CUDA, HIP, Kokkos or HOST execution with or without MPI. The HOST implementation can also be further offloaded to GPUs by OpenMP. The correct definitions for each accelerator backend option are selected in [arch_api.h](src/arch/arch_api.h) by choosing the respective header file. Some compilation combinations are shown below, but also other combination are possible.

```
// Compile to run sequentially on CPU
make

// Compile to run parallel on CPUs with OpenMPI
make MPI=OMPI (or MPI=CRAY)

// Compile to run parallel on CPU with KOKKOS (OpenMP)
make KOKKOS=OPENMP

// Compile to run parallel on GPU with OpenMP offloading (NVIDIA GPUs)
make OMP=CUDA

// Compile to run parallel on GPU with CUDA
make CUDA=1

// Compile to run parallel on GPU with HIP (NVIDIA GPUs)
make HIP=CUDA

// Compile to run parallel on GPU with HIP (AMD GPUs)
make HIP=ROCM

// Compile to run parallel on many GPUs with HIP and OpenMPI (NVIDIA GPUs)
make HIP=CUDA MPI=OMPI

// Compile to run parallel on many GPUs with KOKKOS and Cray MPI
make KOKKOS=ROCM MPI=CRAY
```

Moreover, the `cpp` branch supports [Matplotplusplus](https://alandefreitas.github.io/matplotplusplus/) library for presenting the results in graphical form, and the `c` branch has an example implementation for [Umpire](https://umpire.readthedocs.io/en/develop/) memory manager for CUDA and HIP backends. The compilaton for these can be enabled by `MATPLOT=1` and `UMPIRE=1`. For example,
```
// Compile to run parallel on many GPUs with KOKKOS, Cray MPI, and Matplotplusplus (AMD GPUs)
make KOKKOS=ROCM MPI=CRAY MATPLOT=1
```

## Umpire install for `c` branch

Umpire can be installed with CUDA (HIP) support by
```
git clone --recursive https://github.com/LLNL/Umpire.git
cd Umpire && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/path/umpire -DUMPIRE_ENABLE_C=On -DENABLE_CUDA=On
# cmake .. -DCMAKE_INSTALL_PREFIX=/path/umpire -DUMPIRE_ENABLE_C=On -DENABLE_HIP=On -DCMAKE_CXX_COMPILER=hipcc
make install
```

## Running
The executable can be run using 4 MPI processes with slurm by: 
```
srun ./bessel 4
```