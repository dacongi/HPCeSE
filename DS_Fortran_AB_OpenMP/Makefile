#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#! Makefile: Dynamic simulation makefile 
#! Author: Shuangshuang Jin
#! Last updated: 9-30-2019
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

FC = gfortran 
CC = mpicc

FFLAGS = -w -g -O2 -ffree-form -fopenmp
CFLAGS = -w -g -O2 -ffree-form -fopenmp

LFLAGS = -w -g -O2 -ffree-form -fopenmp
OBJS = main.o inputmodule.o readinputsize.o alloc.o yswitch.o red_ybus.o y_sparse.o mac_em.o i_simu_innerloop.o

all: allo ds
allo: inputmodule.o
	$(FC) $(FFLAGS) -c inputmodule.f

ds: $(OBJS)
	$(FC) $(LFLAGS) -o ds $(OBJS) -lm -lblas -llapack 

clean: 
	rm -rf *.o *.mod *.out *.chk
