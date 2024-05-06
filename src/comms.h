#ifndef BESSEL_COMMS_H
#define BESSEL_COMMS_H

/* HAVE_MPI is set during compile
 * time and determines whether the
 * program is compiled with MPI or not
 */

#if defined(HAVE_MPI)
  #include "mpi.h"
#endif

int comms_get_procs();
int comms_get_rank();
int comms_get_node_procs();
int comms_get_node_rank();

void comms_barrier_procs();
void comms_bcast(int *buf, int count, int root);
void comms_reduce_procs(float *sbuf, int count);

void comms_init_procs(int *argc, char **argv[]);
void comms_finalize_procs();

#endif // !BESSEL_COMMS_H
