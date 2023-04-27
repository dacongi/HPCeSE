# Heterogeneous version of DS, single source code using MPI+CPU or MPI+GPU

qsub -I -X -l select=1:ncpus=16:mem=64gb:ngpus=1:gpu_model=v100:phase=28:interconnect=hdr,walltime=10:20:00

nvidia-cuda-mps-control -d

export CUPY_ACCELERATORS=cub

export OMP_NUM_THREADS=1
## Normal run
mpirun -n 1 python -W ignore great.py gpu 18432

mpirun -n 16 tau_exec python -W ignore great.py 2304

mpirun -n 3 --mca ucx tau_exec python -W ignore great.py 2304
## Woodbury run
mpirun -n 1 python -W ignore s.py 18432

mpirun -n 1 python -W ignore s.py gpu 18432