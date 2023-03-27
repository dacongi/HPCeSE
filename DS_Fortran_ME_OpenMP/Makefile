FC = pgfortran 
CC = 

FFLAGS = -w -g -O2 -Mfree -mp
CFLAGS = -w -g -O2 -Mfree -mp

LFLAGS = -w -g -O2 -Mfree -mp
OBJS = main.o inputmodule.o readinputsize.o alloc.o yswitch.o red_ybus.o y_sparse.o mac_em.o i_simu_innerloop.o

all: allo ds
allo: inputmodule.o
	$(FC) $(FFLAGS) -c inputmodule.f

ds: $(OBJS)
	$(FC) $(LFLAGS) -o ds $(OBJS) -lm -lblas -llapack 

clean: 
	rm -rf *.o *.mod *.out *.chk
