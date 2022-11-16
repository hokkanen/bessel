#ifndef BESSEL_COMMS_H
#define BESSEL_COMMS_H

/* Each HAVE_DEF is set during compile time
 * and determine which dependencies are used
 * by including the respective header files
 */

#if defined(HAVE_MPI)
  #include "mpi.h"
#endif

#if defined(HAVE_CUDA)
  #include "arch_cuda.h"
#elif defined(HAVE_HIP)
  #include "arch_hip.h"
#else
  #include "arch_host.h"
#endif

namespace comms{
  int get_procs();
  int get_rank();
  int get_node_procs();
  int get_node_rank();

  void barrier_procs();
  void reduce_procs(float *sbuf, int count);
  
  void init_procs(int *argc, char **argv[]);
  void finalize_procs();
  
}

#endif // !BESSEL_COMMS_H
