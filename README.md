# Monte Carlo multi-architecture example

This example uses the Monte Carlo method to simulate the value of Bessel's correction that minimizes the root mean squared error in the calculation of the sample standard deviation and variance for the chosen sample and population sizes. The sample standard deviation is typically calculated as $$s = \sqrt{\frac{1}{N - \beta}\sum_{i=1}^{N}(x_i - \bar{x})^2}$$ where $$\beta = 1.$$ The simulation calculates the root mean squared error for different values of $\beta$.

The implementation uses a special construct for the parallel loops in [bessel.cpp](src/bessel.cpp). In the C example, this is based on a preprocessor macro, whereas the C++ example is based on a lambda function, an approach similar to some accelerator frameworks such as SYCL, Kokkos, RAJA, etc. Either option allows conditional compilation of the loops for multiple architectures while keeping the source code clean and readable. An example of the usage of curand and hiprand random number generation libraries inside a GPU kernel are given in [devices_cuda.h](src/devices_cuda.h) and [devices_hip.h](src/devices_hip.h).

The code can be conditionally compiled for either CUDA, HIP, or HOST execution with or without MPI or Matplot (for plotting the results). The correct definitions for each accelerator backend option are selected in [comms.h](src/comms.h) by choosing the respective header file. The compilation instructions are shown below:

```
// Compile to run sequentially on CPU
make

// Compile to run parallel on CPU with KOKKOS (OpenMP)
make KOKKOS=OPENMP

// Compile to run parallel on CPUs with MPI
make MPI=1

// Compile to run parallel on GPU with CUDA
make CUDA=1

// Compile to run parallel on GPU with HIP (NVIDIA GPUs)
make HIP=CUDA

// Compile to run parallel on GPU with HIP (AMD GPUs)
make HIP=ROCM

// Compile to run parallel on many GPUs with KOKKOS and MPI (NVIDIA GPUs)
make KOKKOS=CUDA MPI=1

// Compile to run parallel on many GPUs with KOKKOS, MPI, and Matplotplusplus (AMD GPUs)
make KOKKOS=ROCM MPI=1 MATPLOT=1
```
## Additional notes

The executable can be run on Lumi with 4 MPI processes with srun: 
```
srun --account=Project_462000007 -N1 -n4 --partition=eap --cpus-per-task=1 --gpus-per-task=1 --time=00:15:00 ./bessel 4
```