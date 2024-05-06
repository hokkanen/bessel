#ifndef BESSEL_COMMS_H
#define BESSEL_COMMS_H

/* HAVE_MPI is set during compile
 * time and determines whether the
 * program is compiled with MPI or not
 */

#if defined(HAVE_MPI)
#include "mpi.h"
#endif

namespace comms
{
  int get_global_procs();
  int get_global_rank();
  int get_node_procs();
  int get_node_rank();

  void barrier_procs();
  void reduce_procs(float *sbuf, int count);

  void init_procs(int *argc, char **argv[]);
  void finalize_procs();

}

#endif // !BESSEL_COMMS_H
