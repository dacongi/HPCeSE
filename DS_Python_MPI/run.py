#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#! run.py: Power System Dynamic Simulation Application
#! Author: Cong Wang 
#! Reference: Shuangshuang Jin's Fortran Code
#! Sample execution: 
#!     qsub -I -l select=1:ncpus=16:mpiprocs=16:mem=32gb,walltime=08:00:00
#!     module purge
#!     module load gcc/4.8.1 openmpi/1.10.3 python/3.4
#!     mpirun -np --mca mpi_cuda_support 0 python run.py 3000.txt
#! Last updated: 6-29-2020
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import sys
from mpi4py import MPI
import numpy as np
import math as mt
from scipy.linalg.lapack import zgesv
from scipy.linalg.blas import zgemm


def y_sparse(flag):

    tap=np.ones((array_sizes_brch[rank]),dtype=np.complex128)
    c_from=np.zeros((nbus,array_sizes_brch[rank]),dtype=np.complex128)
    c_to=np.zeros((nbus,array_sizes_brch[rank]),dtype=np.complex128)
    c_line=np.zeros((nbus,array_sizes_brch[rank]),dtype=np.complex128)
    chrgfull=np.zeros((array_sizes_brch[rank],array_sizes_brch[rank]),dtype=np.complex128)
    yyfull=np.zeros((array_sizes_brch[rank],array_sizes_brch[rank]),dtype=np.complex128)
    Y=np.zeros((nbus,nbus),dtype=np.complex128)
    Y_dummy=np.zeros((array_sizes_brch[rank],nbus),dtype=np.complex128)
    
    if flag == 0:
        busy=recv_ipt_bus
        liney=recv_ipt_brch

    if flag == 1:
        busy=fbus
        liney=fline
  
    if flag == 2:
        busy=posfbus
        liney=posfline
        
    
    Gb=busy[:,7]
    Bb=busy[:,8]
    
    r=liney[:,2]
    rx=liney[:,3]
    chrg=jay*(0.5*liney[:,4]).reshape(array_sizes_brch[rank],1)
    z=r+jay*rx
    yy=(1/z).reshape(array_sizes_brch[rank],1)

    from_bus=liney[:,0].astype(int)
    from_int=bus_int[from_bus-1].astype(int)-1
    to_bus=liney[:,1].astype(int)
    to_int=bus_int[to_bus-1].astype(int)-1
    
    phase_shift=liney[:,6]
    liney_ratio=liney[:,5]
    
    ratio_0 = np.where(liney_ratio > 0)
    tap[ratio_0] = np.exp((-jay*phase_shift[ratio_0])*mt.pi/180)/liney_ratio[ratio_0]
        
    for i in range(array_sizes_brch[rank]):
        c_from[from_int[i],i]=tap[i]
        c_to[to_int[i],i]=complex(1,0)
        c_line[from_int[i],i]=c_from[from_int[i],i]-c_to[from_int[i],i]
        c_line[to_int[i],i]=c_from[to_int[i],i]-c_to[to_int[i],i]
    
    np.fill_diagonal(chrgfull,chrg[:,0])
    np.fill_diagonal(yyfull,yy[:,0])

    Y_dummy=zgemm(alpha,chrgfull,c_from,beta,Y_dummy,0,1,1)
    Y=zgemm(alpha,c_from,Y_dummy,beta,Y,0,0,1)
    Y_dummy=zgemm(alpha,chrgfull,c_to,beta,Y_dummy,0,1,1)
    Y=zgemm(alpha,c_to,Y_dummy,alpha,Y,0,0,1)
    Y_dummy=zgemm(alpha,yyfull,c_line,beta,Y_dummy,0,1,1)
    Y=zgemm(alpha,c_line,Y_dummy,alpha,Y,0,0,1)
    
    o=Gb+jay*Bb
    
    if rank == 0:
        np.fill_diagonal(Y[:np.cumsum(array_sizes_bus)[rank],:np.cumsum(array_sizes_bus)[rank]],Y.diagonal()[:np.cumsum(array_sizes_bus)[rank]]+o)
    else:
        np.fill_diagonal(Y[np.cumsum(array_sizes_bus)[rank-1]:np.cumsum(array_sizes_bus)[rank],np.cumsum(array_sizes_bus)[rank-1]:np.cumsum(array_sizes_bus)[rank]],Y.diagonal()[np.cumsum(array_sizes_bus)[rank-1]:np.cumsum(array_sizes_bus)[rank]]+o)
    
    return Y

def reduce_y(comm,flag):
    
    Y_a=np.zeros((array_sizes_gen[rank],ngen),dtype=np.complex128)
    Y_b=np.zeros((array_sizes_gen[rank],nbus),dtype=np.complex128)
    yl=np.zeros(array_sizes_bus[rank],dtype=np.complex128)
    xd=np.zeros(array_sizes_gen[rank])
    y=np.zeros(array_sizes_gen[rank],dtype=np.complex128)
    perm=np.zeros((ngen,array_sizes_gen[rank]),dtype=np.complex128)
    diagy=np.zeros((ngen,array_sizes_gen[rank]),dtype=np.complex128)
    Ymod_all=np.zeros((ngen,ngen),dtype=np.complex128)
    Ymod=np.zeros((ngen,ngen),dtype=np.complex128,order='F')
    permmod=np.zeros((array_sizes_gen[rank],ngen),dtype=np.complex128,order='F')
    Y_b_full=np.zeros((ngen,nbus),dtype=np.complex128)
    Y_d_full=np.zeros((nbus,nbus),dtype=np.complex128)
    temp=np.zeros((ngen,array_sizes_gen[rank]),dtype=np.complex128,order='F')
    
    x=y_sparse(flag)
    
    if flag == 0:
        busy=recv_ipt_bus
        liney=recv_ipt_brch
        Y_d=x
        V=busy[:,1]
        
    elif flag == 1:
        busy=fbus
        liney=fline
        Y_d=x
        V=fbus[:,1]

    elif flag == 2:
        busy=posfbus
        liney=posfline
        Y_d=x
        V=posfbus[:,1]
    
    Pl=busy[:,5]
    Ql=busy[:,6]
    b_type=busy[:,9]
    b_pg=busy[:,3]
    b_qg=busy[:,4]
    
    b_type_3=np.where(b_type == 3)
    Pl[b_type_3]=Pl[b_type_3]-b_pg[b_type_3]
    Ql[b_type_3]=Ql[b_type_3]-b_qg[b_type_3]
    
    yl=(Pl-jay*Ql)/(V*V)

    if rank == 0:
        np.fill_diagonal(Y_d[:np.cumsum(array_sizes_bus)[rank],:np.cumsum(array_sizes_bus)[rank]],Y_d.diagonal()[:np.cumsum(array_sizes_bus)[rank]]+yl)
    else:
        np.fill_diagonal(Y_d[np.cumsum(array_sizes_bus)[rank-1]:np.cumsum(array_sizes_bus)[rank],np.cumsum(array_sizes_bus)[rank-1]:np.cumsum(array_sizes_bus)[rank]],Y_d.diagonal()[np.cumsum(array_sizes_bus)[rank-1]:np.cumsum(array_sizes_bus)[rank]]+yl)
        
    ra=recv_ipt_gen[:,4]*basmva/recv_ipt_gen[:,2]
    g_dstr=recv_ipt_gen[:,7]
    g_dtr=recv_ipt_gen[:,6]
    g_m=recv_ipt_gen[:,2]
    
    g_dstr_0=np.where(g_dstr == 0)
    xd[g_dstr_0]=g_dtr[g_dstr_0]*basmva/g_m[g_dstr_0]
    
    y=1/(ra+jay*xd)

    recv_g_bus=recv_ipt_gen[:,1].astype(int)
    
    if rank == 0:
        np.fill_diagonal(perm, 1)
        np.fill_diagonal(Y_a, y)
        np.fill_diagonal(diagy, y)
        # Consider one bus with multi machine
        if ngen != nPV:
            for i in range(ngen):
                for j in range(array_sizes_gen[rank]):
                    if i != j and g_bus[i]==g_bus[j]:
                        perm[i,j]=1
            permPV=perm
        else:
            permPV=perm
    else:
        np.fill_diagonal(perm[np.cumsum(array_sizes_gen)[rank-1]:np.cumsum(array_sizes_gen)[rank],:array_sizes_gen[rank]], 1)
        np.fill_diagonal(Y_a[:array_sizes_gen[rank],np.cumsum(array_sizes_gen)[rank-1]:np.cumsum(array_sizes_gen)[rank]], y)
        np.fill_diagonal(diagy[np.cumsum(array_sizes_gen)[rank-1]:np.cumsum(array_sizes_gen)[rank],:array_sizes_gen[rank]], y)
        
        if ngen != nPV:
            for i in range(ngen):
                for j in range(array_sizes_gen[rank]):
                    if i != j+np.cumsum(array_sizes_gen)[rank-1] and g_bus[i]==g_bus[j+np.cumsum(array_sizes_gen)[rank-1]]:
                        perm[i,j]=1
            permPV=perm
        else:
            permPV=perm
    
    zgemm(alpha,diagy,permPV,beta,Ymod,0,1,1)  
    Ymod_all=comm.allreduce(Ymod)
    zgemm(alpha,permPV,Ymod_all,beta,permmod,1,0,1)

    recv_Ymod = np.zeros((array_sizes_gen[rank],ngen),dtype=np.complex128)
    comm.Scatterv([Ymod_all,split_sizes_input1_gen,displacements_input1_gen,MPI.DOUBLE_COMPLEX],recv_Ymod,root=0)
    
    for i in range(ngen):
        Y_b[:,g_bus[i]-1]=-recv_Ymod[:,i]
    
    Y_c=Y_b.T
    
    comm.Allgatherv([Y_b,MPI.DOUBLE_COMPLEX],[Y_b_full,split_sizes_output1_gen,displacements_output1_gen,MPI.DOUBLE_COMPLEX])

    for i in range(array_sizes_gen[rank]):
        for k in range(ngen):
            Y_d[recv_g_bus[i]-1,g_bus[k]-1]=Y_d[recv_g_bus[i]-1,g_bus[k]-1]+permmod[i,k]

    Y_d_full=comm.allreduce(Y_d)

    if flag == 0:
        prefrecV1=-zgesv(Y_d_full,Y_c,1,1)[2]
        #prefrecV1=-np.linalg.solve(Y_d_full,Y_c)
        zgemm(alpha,Y_b_full,prefrecV1,beta,temp,0,0,1)
        prefY11=Y_a+temp.T
        #print(prefY11)
        return prefY11

    if flag == 1:
        frecV1=-zgesv(Y_d_full,Y_c,1,1)[2]
        #frecV1=-np.linalg.solve(Y_d_full,Y_c)
        zgemm(alpha,Y_b_full,frecV1,beta,temp,0,0,1)
        fY11=Y_a+temp.T
        #print(fY11)
        return fY11
    
    if flag == 2:
        posfrecV1=-zgesv(Y_d_full,Y_c,1,1)[2]
        #posfrecV1=-np.linalg.solve(Y_d_full,Y_c)
        zgemm(alpha,Y_b_full,posfrecV1,beta,temp,0,0,1)
        posfY11=Y_a+temp.T
        #print(posfY11)
        return posfY11

if __name__ == '__main__':
    
    #the main program start
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    #create global variables
    m=0
    nbus=0
    nbrch=0
    ngen=0
    nSW=0
    nPV=0
    nPQ=0
    label9='9999999.0000'
    jay=complex(0.0,1.0)
    alpha=complex(1.0,0.0)
    beta=complex(0.0,0.0)
    basmva=100
    sys_freq=60
    basrad=2*mt.pi*sys_freq
    sim_k=0
    n=0
    flagF=0

    if rank == 0:
    #1. read in data
        x=[]
        with open("input/"+sys.argv[-1]) as f:
            for line in f:
                line = line.split()
                if len(line)==1 and m==0:
                    nbus=n
                    ipt_bus=np.concatenate(x,axis=0).reshape(nbus,10).astype(float)
                    m=m+1
                    x=[]
                    continue
                elif len(line)==1 and m==1:
                    nbrch=n-nbus
                    ipt_brch=np.concatenate(x,axis=0).reshape(nbrch,7).astype(float)
                    m=m+1
                    x=[]
                    continue
                elif len(line)==1 and m==2:
                    ngen=n-nbrch-nbus
                    ipt_gen=np.concatenate(x,axis=0).reshape(ngen,19).astype(float)
                    m=m+1
                    x=[]
                    continue
                elif len(line)==1 and m==3:
                    nsw=n-nbus-nbrch-ngen
                    ipt_switch=np.concatenate(x,axis=0).reshape(nsw,7).astype(float)
                    continue
                x.append(line)
                n=n+1

        for a in ipt_bus[:,9]:
            if a == 1:
                nSW=nSW+1
            elif a == 2:
                nPV=nPV+1
            elif a == 3:
                nPQ=nPQ+1
        print("Data read in successfully")

    #2. partition the data being used in the program
        #build a function to partition the nbus related data
        split_array_bus = np.array_split(ipt_bus,size,axis=0)

        #create a list to store each splitted array's size
        array_sizes_bus = []
        for i in range(0,len(split_array_bus),1):
                array_sizes_bus = np.append(array_sizes_bus, len(split_array_bus[i])).astype(int)

        #for splitting ipt_bus matrix
        split_sizes_input_bus = array_sizes_bus*np.shape(split_array_bus[0])[1]
        displacements_input_bus = np.insert(np.cumsum(split_sizes_input_bus),0,0)[0:-1]

        #for splitting nbus*nbus matrix
        split_sizes_input1_bus = array_sizes_bus*nbus
        displacements_input1_bus = np.insert(np.cumsum(split_sizes_input1_bus),0,0)[0:-1]

        #for splitting a single column with nbus of the input data matrix
        displacements_input2_bus = np.insert(np.cumsum(array_sizes_bus),0,0)[0:-1]




        #build a function to partition the nbrch related data
        split_array_brch = np.array_split(ipt_brch,size,axis=0)

        #create a list to store each splitted array's size
        array_sizes_brch = []
        for i in range(0,len(split_array_brch),1):
                array_sizes_brch = np.append(array_sizes_brch, len(split_array_brch[i])).astype(int)

        #for splitting ipt_brch matrix
        split_sizes_input_brch = array_sizes_brch*np.shape(split_array_brch[0])[1]
        displacements_input_brch = np.insert(np.cumsum(split_sizes_input_brch),0,0)[0:-1]

        #for splitting nbrch*nbrch matrix
        split_sizes_input1_brch = array_sizes_brch*nbrch
        displacements_input1_brch = np.insert(np.cumsum(split_sizes_input1_brch),0,0)[0:-1]

        #for splitting a single column with nbrch of the input data matrix
        displacements_input2_brch = np.insert(np.cumsum(array_sizes_brch),0,0)[0:-1]




        #build a function to partition the ngen related data
        split_array_gen = np.array_split(ipt_gen,size,axis=0)

        #create a list to store each splitted array's size
        array_sizes_gen = []
        for i in range(0,len(split_array_gen),1):
                array_sizes_gen = np.append(array_sizes_gen, len(split_array_gen[i])).astype(int)

        #for splitting length ipt_gen matrix
        split_sizes_input_gen = array_sizes_gen*np.shape(split_array_gen[0])[1]
        displacements_input_gen = np.insert(np.cumsum(split_sizes_input_gen),0,0)[0:-1]

        #for splitting ngen*ngen matrix
        split_sizes_input1_gen = array_sizes_gen*ngen
        displacements_input1_gen = np.insert(np.cumsum(split_sizes_input1_gen),0,0)[0:-1]

        #for splitting a single column with ngen of the input data matrix
        displacements_input2_gen = np.insert(np.cumsum(array_sizes_gen),0,0)[0:-1]

        #for gathering to a ngen*nbus matrix
        split_sizes_output1_gen = array_sizes_gen*nbus
        displacements_output1_gen = np.insert(np.cumsum(split_sizes_output1_gen),0,0)[0:-1]

        bus_int=ipt_bus[:,0]
        bus_v=ipt_bus[:,1]
        bus_a=ipt_bus[:,2]
        bus_pg=ipt_bus[:,3]
        bus_qg=ipt_bus[:,4]

        g_bus=ipt_gen[:,1].astype(int)


    else:
        ipt_bus = None
        ipt_brch = None
        ipt_gen = None
        ipt_switch = None
        bus_int = None
        bus_v = None
        bus_a = None
        bus_pg = None
        bus_qg = None
        g_bus = None
        nPV = None
        nSW = None
        nPQ = None
        nsw = None
        recvdata = None

        split_array_bus = None
        array_sizes_bus = None
        split_sizes_input_bus = None
        displacements_input_bus = None
        split_sizes_input1_bus = None
        displacements_input1_bus = None

        split_array_brch = None
        split_sizes_input_brch = None
        displacements_input_brch = None
        array_sizes_brch = None

        split_array_gen = None
        split_sizes_input_gen = None
        displacements_input_gen = None
        array_sizes_gen = None
        displacements_input1_gen = None
        split_sizes_input1_gen = None
        split_sizes_output1_gen = None
        displacements_output1_gen = None
        displacements_input2_gen = None


    #Broadcast splitted array to other processes
    split_array_bus = comm.bcast(split_array_bus,root=0) 
    array_sizes_bus = comm.bcast(array_sizes_bus,root=0)
    split_sizes_input_bus = comm.bcast(split_sizes_input_bus,root=0)
    split_sizes_input1_bus = comm.bcast(split_sizes_input1_bus,root=0)
    displacements_input_bus = comm.bcast(displacements_input_bus,root=0)
    displacements_input1_bus = comm.bcast(displacements_input1_bus,root=0)

    split_array_brch = comm.bcast(split_array_brch,root=0)
    split_sizes_input_brch = comm.bcast(split_sizes_input_brch,root=0)
    displacements_input_brch = comm.bcast(displacements_input_brch,root=0)
    array_sizes_brch = comm.bcast(array_sizes_brch,root=0)

    split_array_gen = comm.bcast(split_array_gen,root=0)
    split_sizes_input_gen = comm.bcast(split_sizes_input_gen,root=0)
    displacements_input_gen = comm.bcast(displacements_input_gen,root=0)
    array_sizes_gen = comm.bcast(array_sizes_gen,root=0)
    split_sizes_input1_gen = comm.bcast(split_sizes_input1_gen,root=0)
    displacements_input1_gen = comm.bcast(displacements_input1_gen,root=0)
    split_sizes_output1_gen = comm.bcast(split_sizes_output1_gen,root=0)
    displacements_output1_gen = comm.bcast(displacements_output1_gen,root=0)
    displacements_input2_gen = comm.bcast(displacements_input2_gen,root=0)

    nbus = comm.bcast(nbus,root=0)
    nbrch = comm.bcast(nbrch,root=0)
    ngen = comm.bcast(ngen,root=0)
    nsw = comm.bcast(nsw,root=0)
    bus_int = comm.bcast(bus_int,root=0)
    bus_v = comm.bcast(bus_v,root=0)
    bus_a = comm.bcast(bus_a,root=0)
    bus_pg = comm.bcast(bus_pg,root=0)
    bus_qg = comm.bcast(bus_qg,root=0)
    g_bus = comm.bcast(g_bus,root=0)
    nPV = comm.bcast(nPV,root=0)
    nSW = comm.bcast(nSW,root=0)
    nPQ = comm.bcast(nPQ,root=0)
    ipt_switch = comm.bcast(ipt_switch,root=0)

    recv_ipt_bus = np.zeros(np.shape(split_array_bus[rank]))
    recv_ipt_brch = np.zeros(np.shape(split_array_brch[rank]))
    recv_ipt_gen = np.zeros(np.shape(split_array_gen[rank]))

    comm.Scatterv([ipt_bus,split_sizes_input_bus,displacements_input_bus,MPI.DOUBLE],recv_ipt_bus,root=0)
    comm.Scatterv([ipt_brch,split_sizes_input_brch,displacements_input_brch,MPI.DOUBLE],recv_ipt_brch,root=0)
    comm.Scatterv([ipt_gen,split_sizes_input_gen,displacements_input_gen,MPI.DOUBLE],recv_ipt_gen,root=0)

    comm.Barrier()

    start_Ybus=MPI.Wtime()
    prefY11 = reduce_y(comm,0)

    fbus=np.copy(recv_ipt_bus)
    fline=np.copy(recv_ipt_brch)
    f_type=ipt_switch[1,5]

    if f_type < 4:
        f_nearbus=ipt_switch[1,1].astype(np.int64)
        bus_idx=bus_int[f_nearbus-1]
        bus_idx=int(bus_idx)

        if f_type == 0:
            #Three phase fault zero impedance to ground
            bf=10000000.0
        if rank == 0 and (bus_idx-1) in range(np.cumsum(array_sizes_bus)[rank]):
                fbus[bus_idx-1,8]=bf

        if rank > 0 and (bus_idx-1) in range(np.cumsum(array_sizes_bus)[rank-1],np.cumsum(array_sizes_bus)[rank]):
                fbus[bus_idx-1-np.cumsum(array_sizes_bus)[rank-1],8]=bf

    fY11 = reduce_y(comm,1)

    if f_type < 4:
        f_farbus=ipt_switch[1,2]
        posfbus=recv_ipt_bus
        posfline=recv_ipt_brch

        line_from=posfline[:,0]
        line_to=posfline[:,1]
        line_x=posfline[:,3]

        i=np.where(np.logical_and(line_from==f_nearbus, line_to==f_farbus))
        line_x[i]=10000000.0
        j=np.where(np.logical_and(line_from==f_farbus, line_to==f_nearbus))
        line_x[j]=10000000.0

    posfY11 = reduce_y(comm,2)

    end_Ybus = MPI.Wtime()
    comm.Barrier()

    if rank == 0:
        print("Finish building Ybus and start simulation")
        
    # Start of simulation
    start_Simulation = MPI.Wtime()
    t_step=np.zeros(3,dtype=np.int64)
    t_width=np.empty(20)

    s1=ipt_switch[:,0]
    s7=ipt_switch[:,6]

    t_step=np.around((s1[1:]-s1[:-1])/s7[:-1])
    t_width=(s1[1:]-s1[:-1])/t_step[:]
    sim_k=sum(t_step).astype(int)

    sim_k=sim_k+1

    # Initialize simulation variables
    mac_ang,mac_spd,dmac_ang,dmac_spd,pelect=np.zeros((5,array_sizes_gen[rank],sim_k),dtype=np.float64)
    curdg,curqg,curd,curq,ed,eq,vex=np.zeros((7,array_sizes_gen[rank]),dtype=np.float64)
    eprime=np.zeros((array_sizes_gen[rank],sim_k),dtype=np.complex128)
    eprime_all=np.zeros((ngen),dtype=np.complex128)

    theta=bus_a*mt.pi/180
    bus_volt=bus_v*np.exp(jay*theta)

    mva=basmva/recv_ipt_gen[:,2]
    tst1=bus_int[recv_ipt_gen[:,1].astype(np.int64)-1].astype(np.int64)

    gen_dtr=recv_ipt_gen[:,6]
    gen_H=recv_ipt_gen[:,15]
    gen_do=recv_ipt_gen[:,16]

    eterm=bus_v[tst1-1].astype(np.float64) # terminal bus voltage
    pelect[:,0]=bus_pg[tst1-1]     # BUS_pg
    qelect=bus_qg[tst1-1]     # BUS_qg

    curr=np.sqrt(pelect[:,0]*pelect[:,0]+qelect*qelect)/(eterm*mva)
    phi=np.arctan2(qelect,pelect[:,0])
    v=eterm*np.exp(jay*theta[tst1-1])
    curr=curr*np.exp(jay*(theta[tst1-1]-phi))
    eprime[:,0]=v+jay*gen_dtr*curr
    mac_ang[:,0]=np.arctan2(eprime[:,0].imag,eprime[:,0].real)
    mac_spd[:,0]=1.0
    rot=jay*np.exp(-jay*mac_ang[:,0])
    eprime[:,0]=eprime[:,0]*rot
    edprime=np.copy(eprime[:,0].real)
    eqprime=np.copy(eprime[:,0].imag)
    curr=curr*rot    
    curdg=curr.real
    curqg=curr.imag
    curd=curr.real/mva
    curq=curr.imag/mva
    v=v*rot
    ed=v.real
    eq=v.imag   
    vex=eqprime
    pmech=np.copy(pelect[:,0]*mva)

    S_Steps=1
    steps3=np.sum(t_step)
    steps2=np.sum(t_step[:2])
    steps1=t_step[0]

    h_sol1=t_width[0]
    h_sol2=h_sol1
    flagF1=0

    for I_Steps in range(1,sim_k+2):
        if I_Steps<steps1:
            S_Steps=I_Steps
            flagF1=0
        elif I_Steps==steps1:
            S_Steps=I_Steps
            flagF1=0
        elif I_Steps==steps1+1:
            S_Steps=I_Steps
            flagF1=1
        elif steps1+1<I_Steps<steps2+1:
            S_Steps=I_Steps-1
            flagF1=1
        elif I_Steps==steps2+1:
            S_Steps=I_Steps-1
            flagF1=1
        elif I_Steps==steps2+2:
            S_Steps=I_Steps-1
            flagF1=2
        elif I_Steps>steps2+2:
            S_Steps=I_Steps-2
            flagF1=2

        #compute internal voltage eprime based on q-axis voltage eqprime and generator angle genAngle
        eprime[:,S_Steps-1] = np.sin(mac_ang[:,S_Steps-1])*edprime+np.cos(mac_ang[:,S_Steps-1])*eqprime-jay*(np.cos(mac_ang[:,S_Steps-1])*edprime-np.sin(mac_ang[:,S_Steps-1])*eqprime)
     
        recvdata = np.copy(eprime[:,S_Steps-1])
        comm.Allgatherv([recvdata,MPI.DOUBLE_COMPLEX],[eprime_all,array_sizes_gen,displacements_input2_gen,MPI.DOUBLE_COMPLEX])

        #compute current current based on reduced Y-Bus prefy/fy/posfy and eprime
        if flagF1 == 0:
            cur = prefY11.dot(eprime_all)
        if flagF1 == 1:
            cur = fY11.dot(eprime_all)
        if flagF1 == 2:
            cur = posfY11.dot(eprime_all)
        
        #compute generators electric real power output pElect;
        curd = np.sin(mac_ang[:,S_Steps-1])*cur.real-np.cos(mac_ang[:,S_Steps-1])*cur.imag
        curq = np.cos(mac_ang[:,S_Steps-1])*cur.real+np.sin(mac_ang[:,S_Steps-1])*cur.imag

        curdg = curd*mva
        curqg = curq*mva
        ed = edprime+gen_dtr*curqg
        eq = eqprime-gen_dtr*curdg
        eterm = np.sqrt(ed*ed+eq*eq)
        pelect[:,S_Steps-1] = eq*curq+ed*curd
        qelect = eq*curd-ed*curq


        dmac_ang[:,S_Steps-1] = basrad*(mac_spd[:,S_Steps-1]-1.0)
        dmac_spd[:,S_Steps-1] = (pmech-pelect[:,S_Steps-1]-gen_do*(mac_spd[:,S_Steps-1]-1.0))/(2.0*gen_H)

        if S_Steps == 1:

            k1_mac_ang = h_sol1*dmac_ang[:,S_Steps-1]
            k1_mac_spd = h_sol1*dmac_spd[:,S_Steps-1]

            k2_mac_ang = h_sol1*(basrad*((mac_spd[:,S_Steps-1]+k1_mac_ang)-1.0))
            k2_mac_spd = h_sol1*(pmech-pelect[:,S_Steps-1]-gen_do*(mac_spd[:,S_Steps-1]+k1_mac_spd-1.0))/(2.0*gen_H)

            mac_ang[:,S_Steps] = mac_ang[:,S_Steps-1]+h_sol1*(k1_mac_ang+k2_mac_ang)/2.0
            mac_spd[:,S_Steps] = mac_spd[:,S_Steps-1]+h_sol1*(k1_mac_spd+k2_mac_spd)/2.0
        else:
            mac_ang[:,S_Steps] = mac_ang[:,S_Steps-1]+h_sol1*(3*basrad*(mac_spd[:,S_Steps-1]-1.0)-dmac_ang[:,S_Steps-2])/2.0
            mac_spd[:,S_Steps] = mac_spd[:,S_Steps-1]+h_sol1*(3*(pmech-pelect[:,S_Steps-1]-gen_do*(mac_spd[:,S_Steps-1]-1.0))/(2.0*gen_H)-dmac_spd[:,S_Steps-2])/2.0 
    
    end_Simulation = MPI.Wtime()
    comm.Barrier()
    #print(mac_ang)
    if rank == 0:
        print("Ybus time: ", end_Ybus - start_Ybus)
        print('Simulation time: ', end_Simulation - start_Simulation)
        print("Overall time: ", end_Ybus - start_Ybus + end_Simulation - start_Simulation)

        #for gathering to a ngen*sim_k matrix
        split_sizes_output2_gen = array_sizes_gen*sim_k
        displacements_output2_gen = np.insert(np.cumsum(split_sizes_output2_gen),0,0)[0:-1]
        mac_ang_full,mac_spd_full=np.zeros((2,ngen,sim_k),dtype=np.float64)
    else:
        split_sizes_output2_gen = None
        displacements_output2_gen = None
        mac_ang_full = None
        mac_spd_full = None

    comm.Gatherv(mac_ang,[mac_ang_full,split_sizes_output2_gen,displacements_output2_gen,MPI.DOUBLE],root=0)
    comm.Gatherv(mac_spd,[mac_spd_full,split_sizes_output2_gen,displacements_output2_gen,MPI.DOUBLE],root=0)
    comm.Barrier()
    
    #if rank == 0:
        #print(mac_ang_full)
