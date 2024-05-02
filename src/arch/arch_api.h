#ifndef ARCH_API_H
#define ARCH_API_H

/* Each HAVE_DEF is set during compile time
 * and determine which dependencies are used
 * by including the respective header files
 */

#if defined(HAVE_CUDA)
#include "arch_cuda.h"
#elif defined(HAVE_HIP)
#include "arch_hip.h"
#elif defined(HAVE_KOKKOS)
#include "arch_kokkos.h"
#else
#include "arch_host.h"
#endif

#endif // !ARCH_API_H
