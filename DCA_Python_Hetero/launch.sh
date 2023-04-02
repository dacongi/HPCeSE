#!/bin/sh
export CUPY_ACCELERATORS=cub
export OMP_NUM_THREADS=1
export LD_PRELOAD=""
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
module load anaconda3/2022.05-gcc
source activate paper5
#nvidia-cuda-mps-control -d
#mpirun -np 8 python -W ignore we1.py 391.txt
#echo quit | nvidia-cuda-mps-control
#conda deactivate
#nvidia-smi -c EXCLUSIVE_PROCESS

for ((i=0; i<2; i++)); do
    mkdir /tmp/mps_$i
    mkdir /tmp/mps_log_$i
    #export CUDA_VISIBLE_DEVICES=$i
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$i
    export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$i
    
    nvidia-cuda-mps-control -d
    mpirun -np 8 python -W ignore we1.py $i 30001.txt &
done
#conda deactivate

#for ((i=0; i< 2; i++)); do
   #export CUDA_VISIBLE_DEVICES=$i
   #export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$i
   #echo quit | nvidia-cuda-mps-control
   #rm -fr /tmp/mps_$i
   #rm -fr /tmp/mps_log_$i
#done
