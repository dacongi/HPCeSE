#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#! euler.py: Power System Dynamic Simulation Application(Modified Euler Version)
#! Author: Liwei Wang 
#! Reference: Shuangshuang Jin's Fortran Code and Cong Wang's Python Code
#! Sample execution: 
#!     qsub -I -l select=1:ncpus=16:mpiprocs=16:mem=62gb:phase=19a:interconnect=hdr,walltime=12:00:00 
#!     module purge
#!     module load anaconda3/5.1.0-gcc/8.3.1
#!     module load openmpi/3.1.5-gcc/8.3.1-cuda11_0-ucx
#!     mpirun -np 16 python3 euler.py 3g9b.txt
#! Last updated: 05-01-2021
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import sys
import time
import math as mt
import numpy as np

import petsc4py
petsc4py.init(sys.argv)
from mpi4py import MPI
from petsc4py import PETSc

reduced_time=0
ybus_time=0

#The function for combining the vectors to the result matrix
def merge(mat,vec,i):

    start,end=vec.getOwnershipRange()
    array=vec.getArray()
    idx=[i for i in range(start,end)]
    mat.setValues(idx,i,array)

    mat.assemblyBegin()
    mat.assemblyEnd()

    return mat

#The function for solving the linear system
def axb(A,B,X):

    #Initial the X matrix
    X.zeroEntries()
    B.convert('dense')
    X.convert('dense')
     
    #Setup the precondition
    pcr=PETSc.PC().create(comm=PETSc.COMM_WORLD)
    pcr.setType('lu')
    pcr.setFactorSolverType('superlu_dist')
    pcr.setFactorShift(shift_type=1, amount=1e-10)
    pcr.setFactorOrdering(ord_type='amd')
    pcr.setFromOptions()
    pcr.setOperators(A)
    pcr.setUp()
    F=pcr.getFactorMatrix()

    #Solve the AX=B
    F.matSolve(B,X)
    X.assemblyBegin()
    X.assemblyEnd()

    X.convert('aij')
    pcr.destroy()

    return X
 
#Computing the reduced Y bus   
def reduce_y(flag):
    
    comm = PETSc.COMM_WORLD
    size=comm.Get_size()
    rank=comm.Get_rank()

    start_Sparse = MPI.Wtime()    
    Y=y_sparse(flag)
    end_Sparse = MPI.Wtime()

    start_reduced=MPI.Wtime()      
    ones=PETSc.Vec().createMPI(ngen,comm=comm)
    ones.set(1)

    tmp=PETSc.Vec().createMPI(ngen,comm=comm)

    Y_a=PETSc.Mat().createAIJ([ngen,ngen],comm=comm)
    Y_a.setUp()
    Y_a.zeroEntries()

    frevec=PETSc.Mat().createAIJ([ngen,ngen],comm=comm)
    frevec.setUp()

    Y_b=PETSc.Mat().createAIJ([ngen,nbus],comm=comm)
    Y_b.setUp()
    Y_b.zeroEntries()
    
    Y_c=PETSc.Mat().createAIJ([nbus,ngen],comm=comm)
    Y_c.setUp()

    Y_d=PETSc.Mat().createAIJ([nbus,nbus],comm=comm)
    Y_d.setUp()

    Y_mod=PETSc.Mat().createAIJ([ngen,ngen],comm=comm)
    Y_mod.setUp()

    perm=PETSc.Mat().createAIJ([ngen,ngen],comm=comm)
    perm.setUp()
    perm.zeroEntries()

    permmod=PETSc.Mat().createAIJ([ngen,ngen],comm=comm)
    permmod.setUp()

    permPV=PETSc.Mat().createAIJ([ngen,ngen],comm=comm)
    permPV.setUp()

    dummyeye=PETSc.Mat().createAIJ([nbus,nbus],comm=comm)
    dummyeye.setUp()
    dummyeye.zeroEntries()
    
    diagy=PETSc.Mat().createAIJ([ngen,ngen],comm=comm)
    diagy.setUp()


    if flag==0:
        busy=bus
        liney=line
        Y_d=Y
        V=bus[:,1]

    if flag==1:
        busy=fbus
        liney=fline
        Y_d=Y
        V=fbus[:,1]

    if flag==2:
        busy=posfbus
        liney=posfline
        Y_d=Y
        V=posfbus[:,1]

    rows=int((ngen+size-1)/size)
    a=rank*rows
    b=min((rank+1)*rows,ngen)

    Pl=busy[:,5]
    Ql=busy[:,6]
    b_type=busy[:,9]
    b_pg=busy[:,3]
    b_qg=busy[:,4]
    
    b_type_3=np.where(b_type == 3)
    Pl[b_type_3]=Pl[b_type_3]-b_pg[b_type_3]
    Ql[b_type_3]=Ql[b_type_3]-b_qg[b_type_3]

    yl=PETSc.Vec().createMPI(nbus,comm=comm)
    yl.setValues(bus_index,(Pl-jay*Ql)/(V*V))

    Y_d.setOption(option=18,flag=False)
    Y_d.setDiagonal(yl,addv=2)

    Y_d.assemblyBegin()
    Y_d.assemblyEnd()

    y=PETSc.Vec().createMPI(ngen,comm=comm)
 
    xd=np.zeros(ngen)
    ra=gen[:,4]*basmva/gen[:,2]
    xd_idx=np.where(gen[:,7]==0)
    xd[xd_idx]=gen[xd_idx,6]*basmva/gen[xd_idx,2]

    y.setValues(gen_index,1/(ra+jay*xd))
    y.assemblyBegin()
    y.assemblyEnd()

    perm.setDiagonal(ones)
    Y_a.setDiagonal(y)
    diagy.setDiagonal(y)


    rstart,rend=perm.getOwnershipRange()
    if ngen!=nPV:
        for i in range(rstart,rend):
            for j in range(a,b):
                if i!=j and gen[i,1]==gen[j,1]:
                    perm.setValue(i,1)

            permPV=perm
    else:
        permPV=perm
     

    permPV.assemblyBegin()
    permPV.assemblyEnd()
 
    permPV.transpose()

    Y_mod=diagy.matMult(permPV)

    rstart, rend=Y_mod.getOwnershipRange()    

    for i in range(rstart,rend):
        Y_b.setValue(i, int(gen[i,1]-1), Y_mod.getRow(i)[1])
    
    Y_b.assemblyBegin()
    Y_b.assemblyEnd()

    Y_b.scale(-1)
    Y_c=Y_b.copy()
  
    Y_c.transpose()

    Y_c.assemblyBegin()
    Y_c.assemblyEnd()

    permmod=permPV.matMult(Y_mod)

    permmod.assemblyBegin()
    permmod.assemblyEnd()

    rstart,rend=permmod.getOwnershipRange()
    for i in range(rstart,rend):
        Y_d.setValues(i,[gen[k,1]-1 for k in range(ngen)],permmod.getValues(i,[k for k in range(ngen)]),addv=2)

    Y_d.assemblyBegin()
    Y_d.assemblyEnd()

    X = PETSc.Mat().create(comm=comm)
    X.setSizes([nbus, ngen])
    X.setType('aij')
    X.setUp()
    X.assemblyBegin()
    X.assemblyEnd()

    start_axb=MPI.Wtime()
    Y_c=axb(Y_d,Y_c,X)
    
    end_axb=MPI.Wtime()
   
    Y_c.assemblyBegin()
    Y_c.assemblyEnd()
    
    Y_c.scale(-1)
    Y_b.scale(alpha)
    frevec=Y_b.matMult(Y_c)

    frevec.axpy(beta,frevec)
    frevec.transpose()
    Y_a.axpy(1, frevec)
    Y_a.copy(frevec)
    
    frevec.assemblyBegin()
    frevec.assemblyEnd()

    ones.destroy()
    Y_a.destroy()
    Y_b.destroy()
    Y_c.destroy()
    Y_d.destroy()
    tmp.destroy()
    Y_mod.destroy()
    perm.destroy()
    permmod.destroy()
    permPV.destroy()
    dummyeye.destroy()
    diagy.destroy()

    end_reduced=MPI.Wtime()   
    if (rank==0):
        global reduced_time
        reduced_time=reduced_time+end_reduced-start_reduced
    return frevec


#Computing the sparse Y matrix
def y_sparse(flag):
    
    comm = PETSc.COMM_WORLD
    rank=comm.Get_rank()
    size=comm.Get_size()

    
    if flag==0:
        busy=bus
        liney=line
    
    if flag==1:
        busy=fbus
        liney=fline

    if flag==2:
        busy=posfbus
        liney=posfline

    
    Gb=PETSc.Vec().createMPI(nbus,comm=comm)
    Bb=PETSc.Vec().createMPI(nbus,comm=comm)
    r=PETSc.Vec().createMPI(nbrch,comm=comm)
    rx=PETSc.Vec().createMPI(nbrch,comm=comm)
    chrg=PETSc.Vec().createMPI(nbrch,comm=comm)
    yy=PETSc.Vec().createMPI(nbrch,comm=comm)
    from_bus=PETSc.Vec().createMPI(nbrch,comm=comm)
    to_bus=PETSc.Vec().createMPI(nbrch,comm=comm)
    
    c_from=PETSc.Mat().createAIJ([nbus,nbrch],comm=comm)
    c_from.setUp()
    c_to=c_from.duplicate()
    c_line=c_from.duplicate()
    tran=c_from.duplicate()
    chrgfull=PETSc.Mat().createAIJ([nbrch,nbrch],comm=comm)
    chrgfull.setUp()
    yyfull=PETSc.Mat().createAIJ([nbrch,nbrch],comm=comm)
    yyfull.setUp()
    Y=PETSc.Mat().createAIJ([nbus,nbus],comm=comm)
    Y.setUp()
    Y_dummy=PETSc.Mat().createAIJ([nbrch,nbrch],comm=comm)
    Y_dummy.setUp()

    tap=np.ones(nbrch,dtype=np.complex128)

    Gb.setValues(bus_index,busy[:,7])
    Bb.setValues(bus_index,busy[:,8])
    r.setValues(line_index,liney[:,2])
    rx.setValues(line_index,liney[:,3])

    chrg.setValues(line_index, [0.5*i*jay for i in line[:,4]])

    Gb.assemblyBegin()
    Bb.assemblyBegin()
    r.assemblyBegin()
    rx.assemblyBegin()
    chrg.assemblyBegin()
   
    Gb.assemblyEnd()
    Bb.assemblyEnd()
    r.assemblyEnd()
    rx.assemblyEnd()
    chrg.assemblyEnd()
    
    r.axpy(jay,rx)
    r.reciprocal()
    r.copy(yy)

    from_bus=bus[[(i-1).astype(int) for i in line[:,0]],0].astype(int)-1
    to_bus=bus[[(i-1).astype(int) for i in line[:,1]],0].astype(int)-1

    ratio_0=np.where(line[:,5]>0)
    tap[ratio_0]=np.exp((-jay*line[ratio_0,6])*mt.pi/180)/line[ratio_0,5]

    rows=int((nbrch+size-1)/size)
    c=rank*rows
    d=min((rank+1)*rows,nbrch)

    for i in range(c,d):
        c_from.setValue(from_bus[i],i,tap[i])
        c_to.setValue(to_bus[i],i,complex(1,0))   

    c_from.assemblyBegin()
    c_to.assemblyBegin()
    c_from.assemblyEnd()
    c_to.assemblyEnd()    
   
    c_start, c_end=c_line.getOwnershipRange()

    is_row=PETSc.IS().create()
    is_row.createGeneral(bus_index,comm=comm)
    is_col=PETSc.IS().create()
    is_col.createGeneral(line_index,comm=comm)

    local_from=c_from.createSubMatrices(is_row,is_col)
    local_to=c_to.createSubMatrices(is_row,is_col)

    for i in range(c,d):
        c_line.setValue(from_bus[i],i,local_from[0].getValue(from_bus[i],i)-local_to[0].getValue(from_bus[i],i))         
        c_line.setValue(to_bus[i],i,local_from[0].getValue(to_bus[i],i)-local_to[0].getValue(to_bus[i],i))  
    
    chrgfull.setDiagonal(chrg)
    yyfull.setDiagonal(yy)

    c_line.assemblyBegin()
    Y_dummy.assemblyBegin()
    c_line.assemblyEnd()
    Y_dummy.assemblyEnd()

    Y_time_start=MPI.Wtime()

    Y=PETSc.Mat().createAIJ([nbus,nbus],comm=comm)
    Y.setUp()

    c_from.transpose()
    Y_dummy.axpy(alpha,chrgfull)
    Y_dummy=Y_dummy.matMult(c_from)
    Y_dummy.axpy(beta,Y_dummy)

    c_from.transpose()
    tmp_mat=c_from.copy()
    tmp_mat.scale(alpha)
    Y=tmp_mat.matMult(Y_dummy)
    Y.axpy(beta,Y)

    c_to.transpose()
    tmp_mat=chrgfull.copy()
    tmp_mat.scale(alpha)
    Y_dummy=tmp_mat.matMult(c_to)
    Y_dummy.axpy(beta,Y_dummy)

    c_to.transpose()
    tmp_mat=c_to.copy()
    tmp_mat.scale(alpha)
    Y=tmp_mat.matMult(Y_dummy)
    Y.axpy(beta,Y)

    c_line.transpose()
    tmp_mat=yyfull.copy()
    tmp_mat.scale(alpha)
    Y_dummy=tmp_mat.matMult(c_line)
    Y_dummy.axpy(beta,Y_dummy)

    c_line.transpose()
    tmp_mat=c_line.copy()
    tmp_mat.scale(alpha)
    Y=tmp_mat.matMult(Y_dummy)
    Y.axpy(beta,Y)
  
    Y.assemblyBegin()
    Y.assemblyEnd()

    Gb.axpy(jay,Bb)
    Gb.assemblyBegin()
    Gb.assemblyEnd()

    Y.setOption(option=18,flag=False)
   
    Y.setDiagonal(Gb,addv=2)

    Y.assemblyBegin()
    Y.assemblyEnd()

    Y_time_end=MPI.Wtime()
 
    if(rank==0):
        global ybus_time
        ybus_time=ybus_time+Y_time_end-Y_time_start

    Gb.destroy()
    Bb.destroy()
    r.destroy()
    rx.destroy()
    chrg.destroy()
    yy.destroy()
    c_from.destroy()
    c_to.destroy()
    c_line.destroy()
    chrgfull.destroy()
    yyfull.destroy()
    Y_dummy.destroy()
    
    return Y


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    start=MPI.Wtime()

# Read Input Data
 
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
    simu_k=0
    flagF=0
    index=0
    nbus=0
    nbrch=0
    ngen=0
    nsw=0
    
    #Data sparsing
    if(rank==0):

        print(sys.argv[-1],size)

        item=['bus','brch','gen','sw']
        data=dict()

        for key in item:
            data[key] = []    

        with open("input/"+sys.argv[-1]) as f:
            lines = f.readlines()
        
        for line in lines:
            num = line.strip('\n').split()
            if len(line.split())==1:
                index = index + 1
                continue
            
            data[item[index]].append(num)
            

        bus=np.array(data['bus'],dtype=float)
        line=np.array(data['brch'],dtype=float)
        gen=np.array(data['gen'],dtype=float)
        sw=np.array(data['sw'],dtype=float)     

        nbus=bus.shape[0]
        nbrch=line.shape[0]
        ngen=gen.shape[0]
        nsw=sw.shape[0]

        type_list=bus[:,9].tolist()

        for tp in type_list:
            if tp == 1:
                nSW=nSW+1
            elif tp == 2:
                nPV=nPV+1
            else:
                nPQ=nPQ+1

    else:

        bus=None
        gen=None
        line=None
        sw=None
        nSW=None
        nPV=None
        nPQ=None

    nbus = comm.bcast(nbus,root=0)
    nbrch = comm.bcast(nbrch,root=0)
    ngen = comm.bcast(ngen,root=0)
    nsw = comm.bcast(nsw,root=0)

    nSW = comm.bcast(nSW,root=0)
    nPV = comm.bcast(nPV,root=0)
    nPQ = comm.bcast(nPQ,root=0)

    bus = comm.bcast(bus,root=0) 
    gen = comm.bcast(gen,root=0) 
    line = comm.bcast(line,root=0) 
    sw = comm.bcast(sw,root=0) 

    bus_index=[i for i in range(nbus)]
    line_index=[i for i in range(nbrch)]
    gen_index=[i for i in range(ngen)]

    bus_indx=comm.bcast(bus_index,root=0)
    line_indx=comm.bcast(line_index,root=0)
    gen_indx=comm.bcast(gen_index,root=0)
    
    start_Ybus=MPI.Wtime()

    #Computing pre-fault Y bus
    start_prefY11 = MPI.Wtime()
    prefY11=reduce_y(0)
    end_prefY11 = MPI.Wtime()
    
    fbus=np.copy(bus)
    fline=line
    f_type=sw[1,5]
    
    if f_type<4:
        f_nearbus=sw[1,1].astype(np.int64)
        bus_idx=int(bus[int(f_nearbus-1),0])
        if f_type==0:
            bf=1e7
        fbus[bus_idx-1,8]=bf

    #Computing on-fault Y bus
    start_fY11 = MPI.Wtime()
    fY11=reduce_y(1)
    end_fY11 = MPI.Wtime()

    if f_type<4:
        f_farbus=sw[1,2]
        posfbus=bus
        posfline=line
        
        line_from=posfline[:,0]
        line_to=posfline[:,1]
        line_x=posfline[:,3]

        i=np.where(np.logical_and(line_from==f_nearbus, line_to==f_farbus))
        line_x[i]=1e7
        j=np.where(np.logical_and(line_from==f_farbus, line_to==f_nearbus))
        line_x[j]=1e7

    #Computing pos-fault Y bus
    start_posfY11 = MPI.Wtime()
    posfY11=reduce_y(2)
    end_posfY11 = MPI.Wtime()
    
    end_Ybus = MPI.Wtime() 
    
    # Start of Simulation
    start_Simulation = MPI.Wtime()
    simu_k=0
    psw1=sw[:,0]
    psw7=sw[:,6]

    pbus_idx=bus[:,0]
    pbus_a=bus[:,2]
    pbus_v=bus[:,1]
    pbus_pg=bus[:,3]
    pbus_qg=bus[:,4]
    pgen_mva=gen[:,2]
    pgen_d0=gen[:,16]
    pgen_h=gen[:,15]
    pgen_dtr=gen[:,6]

    mva=PETSc.Vec().createMPI(ngen,comm=comm)
    d0=PETSc.Vec().createMPI(ngen,comm=comm)
    h=PETSc.Vec().createMPI(ngen,comm=comm)

    t_step=(psw1[1:]-psw1[:-1])/psw7[:-1]
    t_width=(psw1[1:]-psw1[:-1])/t_step[:]
    simu_k=sum(t_step).astype(int)

    simu_k=simu_k+1

    mac_ang = PETSc.Mat().create(comm=comm)
    mac_ang.setSizes([ngen, simu_k])
    mac_ang.setType('mpiaij')
    mac_ang.setUp()

    mac_spd = PETSc.Mat().create(comm=comm)
    mac_spd.setSizes([ngen, simu_k])
    mac_spd.setType('mpiaij')
    mac_spd.setUp()

    mac_ang_s0=PETSc.Vec().createMPI(ngen,comm=comm)
    mac_ang_s1=PETSc.Vec().createMPI(ngen,comm=comm)
  
    mac_spd_s0=PETSc.Vec().createMPI(ngen,comm=comm)
    mac_spd_s1=PETSc.Vec().createMPI(ngen,comm=comm)

    dmac_ang_s0=PETSc.Vec().createMPI(ngen,comm=comm)
    dmac_ang_s1=PETSc.Vec().createMPI(ngen,comm=comm)

    dmac_spd_s0=PETSc.Vec().createMPI(ngen,comm=comm)
    dmac_spd_s1=PETSc.Vec().createMPI(ngen,comm=comm)

    k1_mac_ang=PETSc.Vec().createMPI(ngen,comm=comm)
    k1_mac_spd=PETSc.Vec().createMPI(ngen,comm=comm)

    k2_mac_ang=PETSc.Vec().createMPI(ngen,comm=comm)
    k2_mac_spd=PETSc.Vec().createMPI(ngen,comm=comm)

    pelect=PETSc.Vec().createMPI(ngen,comm=comm)

    eprime_s0=PETSc.Vec().createMPI(ngen,comm=comm)
    eprime_s1=PETSc.Vec().createMPI(ngen,comm=comm)

    edprime=PETSc.Vec().createMPI(ngen,comm=comm)
    eqprime=PETSc.Vec().createMPI(ngen,comm=comm)
    
    pmech=PETSc.Vec().createMPI(ngen,comm=comm)

    curr=PETSc.Vec().createMPI(ngen,comm=comm)

    vecTemp1=PETSc.Vec().createMPI(ngen,comm=comm)
    vecTemp2=PETSc.Vec().createMPI(ngen,comm=comm)
    theta=bus[:,2]*mt.pi/180

    flagF=0
        
    pgen_mva=basmva/pgen_mva
    tst1=bus[gen[:,1].astype(np.int64)-1,0].astype(np.int64)
    eterm=pbus_v[tst1-1].astype(np.float64)
    val=pbus_pg[tst1-1]     
    qelect=pbus_qg[tst1-1] 
    
    pelect.setValues(gen_index, val)
    currval=np.sqrt(val*val+qelect*qelect)/(eterm*pgen_mva)
    phi=np.arctan2(qelect,val)
    v=eterm*np.exp(jay*theta[tst1-1])
    currval=currval*np.exp(jay*(theta[tst1-1]-phi))
    tmp=v+jay*pgen_dtr*currval
    eprime_s0.setValues(gen_index,tmp)
    val=np.arctan2(tmp.imag,tmp.real)
    mac_ang_tmp=val
    mac_ang_s0.setValues(gen_index,val)
    mac_spd_s0.setValues(gen_index,[1 for i in range(ngen)])
    eqprime.setValues(gen_index,abs(tmp))
 
    val=[pbus_pg[tst1[i]-1]*pgen_mva[i] for i in range(ngen)]
    pmech.setValues(gen_index, val)
    mva.setValues(gen_index,pgen_mva)
    d0.setValues(gen_index,pgen_d0)
    h.setValues(gen_index,pgen_h)
 
    pelect.assemblyBegin()
    eprime_s0.assemblyBegin()
    mac_ang_s0.assemblyBegin()
    mac_spd_s0.assemblyBegin()
    edprime.assemblyBegin()
    eqprime.assemblyBegin()
    pmech.assemblyBegin()
    mva.assemblyBegin()
    d0.assemblyBegin()
    h.assemblyBegin()
    
    pelect.assemblyEnd()
    eprime_s0.assemblyEnd()
    mac_ang_s0.assemblyEnd()
    mac_spd_s0.assemblyEnd()
    eqprime.assemblyEnd()
    pmech.assemblyEnd()
    mva.assemblyEnd()
    d0.assemblyEnd()
    h.assemblyEnd()


    mac_ang=merge(mac_ang,mac_ang_s0,0)
    mac_spd=merge(mac_spd,mac_spd_s0,0)

    S_Steps=1
    steps3=np.sum(t_step)
    steps2=np.sum(t_step[:2])
    steps1=t_step[0]

    h_sol1=t_width[0]
    h_sol2=h_sol1
    flagF1=0
    flagF1=0

    for I_Steps in range(0,simu_k+1):
        if I_Steps<steps1:
            S_Steps=I_Steps
            flagF1=0
            flagF2=0
        elif I_Steps==steps1:
            S_Steps=I_Steps
            flagF1=0
            flagF2=1
        elif I_Steps==steps1+1:
            S_Steps=I_Steps
            flagF1=1
            flagF2=1
        elif steps1+1<I_Steps<steps2+1:
            S_Steps=I_Steps-1
            flagF1=1
            flagF2=1
        elif I_Steps==steps2+1:
            S_Steps=I_Steps-1
            flagF1=1
            flagF2=2
        elif I_Steps==steps2+2:
            S_Steps=I_Steps-1
            flagF1=2
            flagF2=2
        elif I_Steps>steps2+2:
            S_Steps=I_Steps-2
            flagF1=2
            flagF2=2


        if I_Steps!=0 and I_Steps<simu_k+1:
            mac_ang_s1.copy(mac_ang_s0)
            mac_spd_s1.copy(mac_spd_s0)
            eprime_s1.copy(eprime_s0)
       

        mac_ang_s0.copy(vecTemp1)
        vecTemp1.scale(jay)
        vecTemp1.exp()
        eprime_s0.pointwiseMult(eqprime,vecTemp1)   

     
        if flagF1 == 0:
            prefY11.mult(eprime_s0,curr)
        elif flagF1 == 1:
            fY11.mult(eprime_s0,curr)
        elif flagF1 == 2:
            posfY11.mult(eprime_s0,curr)


        # pelect
        curr.conjugate()
        pelect.pointwiseMult(eprime_s0,curr)
        pelect.copy(vecTemp1)
        vecTemp1.conjugate()
        pelect.axpy(1,vecTemp1)
        pelect.scale(0.5)
        
        
        # dmac_ang
        mac_spd_s0.copy(vecTemp1)
        vecTemp1.shift(-1.0)
        vecTemp1.scale(basrad)
        vecTemp1.copy(dmac_ang_s0)
        #mac_spd_s0.view()
        #dmac_ang_s1.view()
    
        # dmac_spd
        vecTemp1.pointwiseMult(pelect,mva)
        pmech.copy(dmac_spd_s0)
        dmac_spd_s0.axpy(-1,vecTemp1)
        mac_spd_s0.copy(vecTemp1)
        vecTemp1.shift(-1)
        vecTemp2.pointwiseMult(d0,vecTemp1)
        dmac_spd_s1.axpy(-1,vecTemp2)

        mac_ang_s0.copy(mac_ang_s1)
        mac_ang_s1.axpy(h_sol1,dmac_ang_s0)
        mac_spd_s0.copy(mac_spd_s1)
        mac_spd_s1.axpy(h_sol1,dmac_spd_s0)

        mac_ang_s1.copy(vecTemp1)
        vecTemp1.scale(jay)
        vecTemp1.exp()
        eprime_s1.pointwiseMult(eqprime,vecTemp1)

        if flagF2==0:
            prefY11.mult(eprime_s1,curr)
        elif flagF2==1:
            fY11.mult(eprime_s1,curr)
        elif flagF2==0:
            posfY11.mult(eprime_s1,curr)


        # pelect
        curr.conjugate()
        pelect.pointwiseMult(eprime_s1,curr)
        pelect.copy(vecTemp1)
        vecTemp1.conjugate()
        pelect.axpy(1,vecTemp1)
        pelect.scale(0.5)

        # dmac_ang
        mac_spd_s1.copy(vecTemp1)
        vecTemp1.shift(-1.0)
        vecTemp1.scale(basrad)
        vecTemp1.copy(dmac_ang_s1)
    
        # dmac_spd
        vecTemp1.pointwiseMult(pelect,mva)
        pmech.copy(dmac_spd_s1)
        dmac_spd_s1.axpy(-1,vecTemp1)
        mac_spd_s1.copy(vecTemp1)
        vecTemp1.shift(-1)
        vecTemp2.pointwiseMult(d0,vecTemp1)
        dmac_spd_s1.axpy(-1,vecTemp2)

        dmac_ang_s0.copy(vecTemp1)
        vecTemp1.axpy(1.0,dmac_ang_s1)
        vecTemp1.scale(0.5)
        mac_ang_s0.copy(mac_ang_s1)
        mac_ang_s1.axpy(h_sol2,vecTemp1)
        dmac_spd_s0.copy(vecTemp1)
        vecTemp1.axpy(1.0,dmac_spd_s1)
        vecTemp1.scale(0.5)
        mac_spd_s0.copy(mac_spd_s1)
        mac_spd_s1.axpy(h_sol2,vecTemp1)

        #mac_ang_s1.view()
        #mac_spd_s1.view()
        
        mac_ang=merge(mac_ang,mac_ang_s1,S_Steps)
        mac_spd=merge(mac_spd,mac_spd_s1,S_Steps)
        
        
    end_Simulation = MPI.Wtime()

    mac_ang.assemblyBegin() 
    mac_ang.assemblyEnd()
    mac_spd.assemblyBegin()
    mac_spd.assemblyEnd()
  
    end=MPI.Wtime()

    mac_ang=merge(mac_ang,mac_ang_s1,S_Steps)
    mac_spd=merge(mac_spd,mac_spd_s1,S_Steps)

    if(rank==0):
        print('Ybus:',ybus_time)
        print('Reduced ybus:',reduced_time)
        print('Simulation:',end_Simulation - start_Simulation)
        print('Total:',end-start)

