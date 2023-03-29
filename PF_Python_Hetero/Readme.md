# Power system load flow study, Newton Raphson iterative method, Jacobian matrix

#qsub -I -X -l select=1:ncpus=16:mem=64gb:ngpus=1:gpu_model=a100:phase=28:interconnect=hdr,walltime=10:20:00

#The code is developed with CPU+MPI and GPU in Python

#mpirun -np 1 --mca opal_cuda_support 1 python -W ignore Powerflow_DS_p.py gpu 19968

#mpirun -np 16 --mca opal_cuda_support 0 python -W ignore Powerflow_DS_p.py 19968