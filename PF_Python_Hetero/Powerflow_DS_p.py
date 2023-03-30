#mpirun -np 1 --mca opal_cuda_support 1 python -W ignore Powerflow_DS_p.py gpu 19968
#qsub -I -X -l select=1:ncpus=16:mem=64gb:ngpus=1:gpu_model=a100:phase=28:interconnect=hdr,walltime=10:20:00
import time
import sys
import math as mt
from mpi4py import MPI
from func import m_ang_ab,m_spd_ab,solver,sp_mat,methods,parser,array_partition,stack
import numpy as np
import cupy as cp

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

jay=1j
basmva=100
sys_freq=60
itera = 0
flag = 1
tol = 1e-9
iter_max = 30

complex128,float64,count_nonzero,where,ones,zeros,arange,exp,a_append,eye,logical_and,array,amax = methods(sys.argv[-1])
csr_matrix,csc_matrix = sp_mat(sys.argv[-1])
vstack,hstack=stack(sys.argv[-1])

t0 = time.time()
if rank == 0:
    n=0
    m=0
    x=[]
    with open("input/"+sys.argv[-1]+'.txt') as f:
        for line in f:
            line = line.split()
            if len(line)==1 and m==0:
                nbus=n
                a=x
                m=m+1
                x=[]
                continue
            elif len(line)==1 and m==1:
                nbrch=n-nbus
                b=x
                m=m+1
                x=[]
                continue
            elif len(line)==1 and m==2:
                ngen=n-nbrch-nbus
                c=x
                m=m+1
                x=[]
                continue
            elif len(line)==1 and m==3:
                nsw=n-nbus-nbrch-ngen
                d=x
                continue
            x.append(line)
            n=n+1

    ipt_bus=array(a,dtype=float64)
    ipt_brch=array(b,dtype=float64)
    nPV=count_nonzero(ipt_bus[:,9]==2)
    t00=time.time()
    print(t00-t0)
    print("===== Data read in successfully")
    print("===== nbus:", nbus, " nbrch:", nbrch)

    # assign bus data
    bus_int=ipt_bus[:,0]
    V=ipt_bus[:,1]
    b_ang=ipt_bus[:,2]
    b_pg=ipt_bus[:,3]
    b_qg=ipt_bus[:,4]
    Pl=ipt_bus[:,5]
    Ql=ipt_bus[:,6]
    Gb=ipt_bus[:,7]
    Bb=ipt_bus[:,8]
    b_type=ipt_bus[:,9]
    qg_max=ipt_bus[:,10]
    qg_min=ipt_bus[:,11]

    # assign branch data
    from_bus=ipt_brch[:,0].astype(int)
    to_bus=ipt_brch[:,1].astype(int)
    r=ipt_brch[:,2]
    rx=ipt_brch[:,3]
    chrg=jay*(0.5*ipt_brch[:,4])
    liney_ratio=ipt_brch[:,5]
    phase_shift=ipt_brch[:,6]

    ref=where(b_type==1)[0]
    PQV=where(b_type>=2)[0]
    PQ=where(b_type==3)[0]
    PV=where(b_type==2)[0]

    noref_id=ones(nbus)
    nogen_id=ones(nbus)
    a=arange(nbrch)
    tap=ones((nbrch),dtype=complex128)
    c_from=csr_matrix((nbus,nbrch),dtype=complex128)
    c_line=csr_matrix((nbus,nbrch),dtype=complex128)
    c_to=csr_matrix((nbus,nbrch),dtype=complex128)
    chrgfull=csr_matrix((nbrch,nbrch),dtype=complex128)
    yyfull=csr_matrix((nbrch,nbrch),dtype=complex128)
    tap[liney_ratio>0]=exp((-jay*phase_shift[liney_ratio>0])*mt.pi/180)/liney_ratio[liney_ratio>0]

    # prefY11=reduce_y(flag=0)
    z=r+jay*rx
    yy=1/z

    from_int=bus_int[from_bus-1].astype(int)
    to_int=bus_int[to_bus-1].astype(int)
    from_int=from_int-1
    to_int=to_int-1

    # line impedance
    # determine connection matrices including tap chargers and phase shifters
    # sparse matrix formulation
    c_from[from_int,a]=tap[a]
    c_to[to_int,a]=1
    c_line[from_int,a]=c_from[from_int,a]-c_to[from_int,a]
    c_line[to_int,a]=c_from[to_int,a]-c_to[to_int,a]

    chrgfull.setdiag(chrg)
    yyfull.setdiag(yy)

    Y_dummy=chrgfull.dot(c_from.T)
    Y=c_from.dot(Y_dummy)
    Y_dummy=chrgfull.dot(c_to.T)
    Y=c_to.dot(Y_dummy)+Y
    Y_dummy=yyfull.dot(c_line.T)
    Y=c_line.dot(Y_dummy)+Y

    Y.setdiag(Y.diagonal()+Gb+jay*Bb)

    noref_id[ref]=0
    nogen_id[PV]=0
    len_of_noref=len(PQV)
    len_of_load=len(PQ)
    theta=b_ang*mt.pi/180

    bus_volt=V*exp(jay*theta)
    ang_red=csr_matrix((ones(len_of_noref),(arange(len_of_noref),PQV)),shape=(len_of_noref,nbus))
    volt_red=csr_matrix((ones(len_of_load),(arange(len_of_load),PQ)),shape=(len_of_load,nbus))
    S_diag=csr_matrix((nbus,nbus),dtype=complex128)
    V_CPLX_diag=csr_matrix((nbus,nbus),dtype=complex128)
    Vm_diag=csr_matrix((nbus,nbus),dtype=complex128)

    S = bus_volt*((Y.dot(bus_volt)).conj())
    P = S.real
    Q = S.imag
    dP = b_pg - Pl - P
    dQ = b_qg - Ql - Q
    dP = dP*noref_id
    dQ = dQ*noref_id
    dQ = dQ*nogen_id
    mis = max(max(abs(dP)),max(abs(dQ)))

    if mis > tol:
        conv_flag = 1
    else:
        conv_flag = 0

    array_sizes_volt,absolute_ps_volt = array_partition(volt_red.toarray(),size)
    array_sizes_ang,absolute_ps_ang = array_partition(ang_red.toarray(),size)
else:
    conv_flag=None
    absolute_ps_ang=None
    absolute_ps_volt=None
    ang_red=None
    volt_red=None

absolute_ps_ang=comm.bcast(absolute_ps_ang)
absolute_ps_volt=comm.bcast(absolute_ps_volt)
conv_flag=comm.bcast(conv_flag)
ang_red=comm.bcast(ang_red)
volt_red=comm.bcast(volt_red)

while (conv_flag == 1 and itera < iter_max):
    t5=time.time()
    itera = itera + 1
    if rank == 0:
        bus_volt=V*exp(jay*theta)
        S=bus_volt*Y.conj().dot(bus_volt.conj())
        S_diag.setdiag(S)
        V_CPLX_diag.setdiag(bus_volt)
        Vm_diag.setdiag(abs(V))
        SS=V_CPLX_diag.dot(Y.conj()).dot(V_CPLX_diag.conj())

    else:
        SS=None
        S_diag=None
        Vm_diag=None
        
    S_diag=comm.bcast(S_diag)
    SS=comm.bcast(SS)
    Vm_diag=comm.bcast(Vm_diag)
    if sys.argv[-2] == 'gpu':
        with cp.cuda.Device(rank):
            S1=(S_diag+SS).dot(csc_matrix(solver(Vm_diag,volt_red[absolute_ps_volt[rank]:absolute_ps_volt[rank+1]].T,sys.argv[-2]).get()))
    else:
        S1=(S_diag+SS).dot(csc_matrix(solver(Vm_diag,volt_red[absolute_ps_volt[rank]:absolute_ps_volt[rank+1]].T,sys.argv[-2])))

    S2=(S_diag-SS).dot(ang_red[absolute_ps_ang[rank]:absolute_ps_ang[rank+1]].T)
    S1=S1.reshape(S1.shape[0],absolute_ps_volt[rank+1]-absolute_ps_volt[rank])

    J11=-ang_red.dot(S2.imag)
    J12=ang_red.dot(S1.real)
    J21=volt_red.dot(S2.real)
    J22=volt_red.dot(S1.imag)

    if rank > 0:
        comm.send(J11,dest=0,tag=rank)
        comm.send(J12,dest=0,tag=rank)
        comm.send(J21,dest=0,tag=rank)
        comm.send(J22,dest=0,tag=rank)

    if rank == 0:
        t333=time.time()
        for i in range(1,size):
            J11=hstack([J11, comm.recv(source=i,tag=i)])
            J12=hstack([J12, comm.recv(source=i,tag=i)])
            J21=hstack([J21, comm.recv(source=i,tag=i)])
            J22=hstack([J22, comm.recv(source=i,tag=i)])

        J=vstack([hstack([J11,J12]),hstack([J21,J22])])

        dP_red = ang_red.dot(dP)
        dQ_red = volt_red.dot(dQ)
        b = a_append(dP_red,dQ_red)

        sol = solver(J,b,sys.argv[-1])#splu(J).solve(b)
        dang = (ang_red.T).dot(sol[:len(PQV)])
        dV = (volt_red.T).dot(sol[len(PQV):len(PQV)+len(PQ)])

        V = V + dV
        theta = theta + dang
        bus_volt = V*exp(jay*theta)
        
        S = bus_volt*((Y.dot(bus_volt)).conj())
        P = S.real
        Q = S.imag
        dP = b_pg - Pl - P
        dQ = b_qg - Ql - Q
        dP = dP*noref_id
        dQ = dQ*noref_id
        dQ = dQ*nogen_id
        
        mis = max(amax(abs(dP)),amax(abs(dQ)))

        if mis > tol:
            conv_flag = 1
        else:
            conv_flag = 0
            
        b_qg[PV] = Q[PV] + Ql[PV]
        
        i=where(b_qg>qg_max)
        if len(i[0])>0:
            print('There are Qg > Qgmax at iter',itera)
        j=where(b_qg<qg_min)
        if len(j[0])>0:
            print('There are Qg < Qgmin at iter',itera)
        t433=time.time()

    conv_flag=comm.bcast(conv_flag)

    t6=time.time()
    #print(t6-t5)
if rank == 0:
    b_pg[ref]=P[ref] + Pl[ref]
    b_pg[PV]=P[PV] + Pl[PV]
    Pl[PQ]=b_pg[PQ] - P[PQ]
    b_qg[ref]=Q[ref] + Ql[ref]
    b_qg[PV]=Q[PV]+Ql[PV]
    Ql[PQ]=b_qg[PQ]-Q[PQ]
    b_ang=theta*180/mt.pi
    
    t1 = time.time()
    print('Program finish in:', round(t1-t0,2), 'seconds')
#'''