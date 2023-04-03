#!/bin/sh
export CUPY_ACCELERATORS=cub
export OMP_NUM_THREADS=1
export LD_PRELOAD=""
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
module load anaconda3/2022.05-gcc
source activate paper5

echo -n 'Test system: ' 
read TEST_SYS
echo -n "Total number of cases (max:1024): "
read TCASE
gpus_count=$(nvidia-smi -L | wc -l)
echo Total number of GPU: $gpus_count
echo -n "Number of GPUs used: " 
read NGPU
echo -n "Number of processes each GPU launched: " 
read NPROCS
echo Running on $NGPU GPUs

for ((i=0; i<gpus_count; i++)); do

    mkdir /tmp/mps_$i
    mkdir /tmp/mps_log_$i
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$i
    export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$i
    
    nvidia-cuda-mps-control -d
done

for ((i=0; i<gpus_count; i++)); do

    export CUDA_VISIBLE_DEVICES=$i
    mpirun -np $NPROCS python -W ignore DCA_mpi.py $TCASE $NGPU $i $TEST_SYS &
done
wait

echo quit | nvidia-cuda-mps-control

for ((i=0; i<gpus_count; i++)); do
    #export CUDA_VISIBLE_DEVICES=$i
    #export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$i
    rm -fr /tmp/mps_$i
    rm -fr /tmp/mps_log_$i
done
