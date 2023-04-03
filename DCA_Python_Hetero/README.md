# Multi-GPU multiprocessing dynamic contingency analysis. 

PrefY11 is global. fY11, postfY11, and following time domain integration to solve DAEs are local on each process.

Approach 1: ./launch_mpi.sh runs up to 1024 contingency cases in total on a GPU cluster using MPI. Real-time reponses for the systems smaller than Polish3120.

Approach 2: ./launch_multiproc.sh runs up to 1024 contingency cases on a GPU cluster using multiprocessing package. Real-time reponses for the systems smaller than Polish3120. This one has a CPU-based DCA for benchmarking purposes.