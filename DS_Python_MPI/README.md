Go to the Palmetto Cluster and request computing resources. I suggest 16 cores and 32 GB memory for a general testing task. For the largest case 8100.txt, more memory should be requested.

Simply download the input folder and the run.py to your current directory, and run with the following command.

module purge

module load gcc/4.8.1 openmpi/1.10.3 python/3.4

mpirun -np 4 --mca mpi_cuda_support 0 python run.py 3000.txt
