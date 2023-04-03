#!/bin/sh
#export CUPY_ACCELERATORS=cub
#export CUPY_CUDA_COMPILE_WITH_DEBUG=1
#export OMP_NUM_THREADS=1
#export LD_PRELOAD=""
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
cd ~/HPCeSE/DCA_Python_Hetero
module load anaconda3/2022.05-gcc/9.5.0
source activate paper5
#nvprof  -o output_%p.nvprof #
#nvprof --profile-child-processes -o output.%p mpirun -np 8 python we.py
#nvprof --profile-child-processes -o output.%p python launch_GPU.py 1451.txt 32 8

echo -n 'Test system: ' 
read TEST_SYS
echo -n "Total number of cases (max: 1024): "
read TCASE
echo -n "Type cpu or gpu: "
read VAR

if [[ $VAR = 'gpu' ]]
then
  echo Using $VAR
  sleep 1
  #echo ------------------------------------------------------
  gpus_count=$(nvidia-smi -L | wc -l)
  echo Total number of GPU: $gpus_count
  echo -n "Number of GPUs used: " 
  read NGPU
  NCASE=$((TCASE / NGPU))
  echo Number of cases each GPU solved: $NCASE
  echo -n "Number of processes each GPU launched: " 
  read NPROCS
  #echo ------------------------------------------------------
  echo Running on $NGPU GPUs

  for ((i=0; i<NGPU; i++)); do

     mkdir /tmp/mps_$i
     mkdir /tmp/mps_log_$i

     export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$i
     export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$i

     nvidia-cuda-mps-control -d
  done
  
  for ((i=0; i<NGPU; i++)); do
  
     export CUDA_VISIBLE_DEVICES=$i
     python -W ignore launch_GPU.py $i $TEST_SYS $NCASE $NPROCS &
  done
  wait
  
  echo quit | nvidia-cuda-mps-control
  
  for ((i=0; i<NGPU; i++)); do
  
     rm -fr /tmp/mps_$i
     rm -fr /tmp/mps_log_$i
  done
  
else
  echo Using CPU
  sleep 1
  echo -n "Number of processes launched: " 
  read NPROCS
  #echo ------------------------------------------------------
  echo Total number of cases: $TCASE
  echo Number of processes launched: $NPROCS
  echo 'Test system:' $TEST_SYS
  #echo ------------------------------------------------------
  echo Running on $NPROCS CPUs
  python -W ignore launch_CPU.py $TEST_SYS $TCASE $NPROCS

fi