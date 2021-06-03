qsub -I -l select=1:ncpus=16:mpiprocs=16:mem=62gb:phase=19a:interconnect=hdr,walltime=12:00:00

module purge
module load anaconda3/5.1.0-gcc/8.3.1
module load openmpi/3.1.5-gcc/8.3.1-cuda11_0-ucx

mpirun -np 16 python3 petsc.py 3g9b.txt

This folder includes numerical integration Adams-Bashforth and Modified Euler method implementation.
