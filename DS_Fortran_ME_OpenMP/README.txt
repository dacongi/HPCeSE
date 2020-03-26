qsub -I -l select=1:ncpus=16
export OMP_NUM_THREADS=16
./ds 3g9b.txt a.out b.out


This folder includes the numerical integration modified eular method implementation
