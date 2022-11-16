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

int comms_get_procs();
int comms_get_rank();
int comms_get_node_procs();
int comms_get_node_rank();

void comms_barrier_procs();
void comms_bcast(unsigned int *buf, unsigned int count, unsigned int root);
void comms_reduce_procs(float *sbuf, unsigned int count);

void comms_init_procs(int *argc, char **argv[]);
void comms_finalize_procs();

#endif // !BESSEL_COMMS_H
