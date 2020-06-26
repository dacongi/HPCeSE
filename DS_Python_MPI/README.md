module purge
module load gcc/4.8.1 openmpi/1.10.3 python/3.4
mpirun -np 4 --mca mpi_cuda_support 0 python run.py 3000.txt
