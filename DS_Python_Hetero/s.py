#qsub -I -X -l select=1:ncpus=16:mem=64gb:ngpus=1:gpu_model=a100:phase=28:interconnect=hdr,walltime=10:20:00
#nvidia-cuda-mps-control -d
#export CUPY_ACCELERATORS=cub
#export OMP_NUM_THREADS=1
#mpirun -n 1 python -W ignore great.py 18432
#mpirun -n 16 tau_exec python -W ignore great.py 2304
#mpirun -n 3 --mca ucx tau_exec python -W ignore great.py 2304
#paraprof
#find -type f -name '*profile*' -delete
#find -type f -name '*0.2*' -delete
#qsub -I -X -l select=1:ncpus=16:mem=64gb:ngpus=1:gpu_model=a100:phase=27:interconnect=hdr,walltime=10:20:00
import sys
import time
from mpi4py import MPI
import math as mt
import numpy as np
import cupy as cp
from func import m_ang_ab, m_spd_ab, solver, sp_mat
from func import methods, parser, array_parti, inv, find
#count=cp.cuda.runtime.getDeviceCount()
if __name__ == '__main__':
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    csr_matrix,csc_matrix=sp_mat(sys.argv[-2])
    complex128,float64,count_nonzero,where,ones,zeros,arange,exp,a_append,eye,logical_and,array,amax=methods(sys.argv[-2])
    
    start=time.time()
    jay=1j
    basmva=100
    sys_freq=60
    
    if rank == 0:
        ipt_bus,ipt_brch,ipt_gen,ipt_switch,nbus,nbrch,ngen,nsw=parser(sys.argv[-1],array,float64)

        a=arange(nbrch)
        tap=ones((nbrch),dtype=complex128)
        xd=zeros(ngen)
        sd=ones(ngen)

        c_from=csr_matrix((nbus,nbrch),dtype=complex128)
        c_line=csr_matrix((nbus,nbrch),dtype=complex128)
        c_to=csr_matrix((nbus,nbrch),dtype=complex128)
        chrgfull=csr_matrix((nbrch,nbrch),dtype=complex128)
        yyfull=csr_matrix((nbrch,nbrch),dtype=complex128)
        perm=csr_matrix((ngen,ngen),dtype=complex128)
        Y_b=csr_matrix((ngen,nbus),dtype=complex128)
        Y_a=csr_matrix((ngen,ngen),dtype=complex128)

        print("===== Data read in successfully")
        print("===== nbus:", nbus, " nbrch:", nbrch, " ngen:", ngen)

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
        nPV=count_nonzero(ipt_bus[:,9]==2)

        # assign branch data
        from_bus=ipt_brch[:,0].astype(int)
        to_bus=ipt_brch[:,1].astype(int)
        r=ipt_brch[:,2]
        rx=ipt_brch[:,3]
        chrg=jay*(0.5*ipt_brch[:,4])
        liney_ratio=ipt_brch[:,5]
        phase_shift=ipt_brch[:,6]

        # assign generator data
        g_bus=ipt_gen[:,1].astype(int)
        g_m=ipt_gen[:,2]
        g_r=ipt_gen[:,4]
        g_dtr=ipt_gen[:,6]
        g_dstr=ipt_gen[:,7]
        g_H=ipt_gen[:,15]
        g_do=ipt_gen[:,16]

        # assign switch data
        f_type=ipt_switch[1:nsw-2]
        s1=ipt_switch[:,0]
        s7=ipt_switch[:,6]

        t00=time.time()
        # prefY11=reduce_y(flag=0)
        z=r+jay*rx
        yy=1/z

        from_int=bus_int[from_bus-1].astype(int)
        to_int=bus_int[to_bus-1].astype(int)
        tap[liney_ratio>0]=exp((-jay*phase_shift[liney_ratio>0])*mt.pi/180)/liney_ratio[liney_ratio>0]
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

        Y_dummy=chrgfull.dot(c_from.transpose())
        Y=c_from.dot(Y_dummy)
        Y_dummy=chrgfull.dot(c_to.transpose())
        Y=c_to.dot(Y_dummy)+Y
        Y_2=Y.copy()
        Y_dummy=yyfull.dot(c_line.transpose())
        Y=(c_line.dot(Y_dummy)+Y)
        Y_1=Y.copy()

        Pl[b_type==3]=Pl[b_type==3]-b_pg[b_type==3]
        Ql[b_type==3]=Ql[b_type==3]-b_qg[b_type==3]
        yl=(Pl-jay*Ql)/(V*V)

        ra=g_r*basmva/g_m
        xd[g_dstr==0]=g_dtr[g_dstr==0]*basmva/g_m[g_dstr==0]
        y=1/(ra+jay*xd)

        perm.setdiag(sd)
        Y_a.setdiag(y)
        Y_a = Y_a.tocsc()

        if ngen != nPV:
            e=where(~eye(perm.get_shape()[0],dtype=bool))
            h=where(g_bus[e[0]]==g_bus[e[1]])

            perm[e[0][h],e[1][h]]=1
            permPV=perm
        else:
            permPV=perm

        Ymod=Y_a.dot(permPV.transpose())
        permmod=permPV.transpose().dot(Ymod)

        Y_b = Y_b.tocsc()
        Y_b[:,g_bus-1]=-Ymod.transpose()[:]

        Y.setdiag(Y.diagonal()+yl+Gb+jay*Bb)

        if len(f_type) == 1:
            case_id = 0
        f_type=f_type[case_id]
        f_nearbus=f_type[1].astype(int)
        bus_idx=bus_int[f_nearbus-1]
        bus_idx=bus_idx.astype(int)
        Bb[bus_idx-1]=10000000.0

        # fY11=reduce_y(flag=1)
        Y_1.setdiag(Y_1.diagonal()+yl+Gb+jay*Bb)

        f_farbus=f_type[2].astype(int)
        Bb[bus_idx-1]=0.0

        i=where(logical_and(from_bus==f_nearbus,to_bus==f_farbus))
        j=where(logical_and(from_bus==f_farbus,to_bus==f_nearbus))

        rx[i]=10000000.0
        rx[j]=10000000.0

        # posfY11=reduce_y(flag=2)
        z=r+jay*rx
        yy=1/z

        yyfull.setdiag(yy)
        Y_dummy=yyfull.dot(c_line.transpose())
        Y_2=(c_line.dot(Y_dummy)+Y_2)

        Y_2.setdiag(Y_2.diagonal()+yl+Gb+jay*Bb)

        array_sizes_gen,absolute_ps=array_parti(ipt_gen,size)

    else:
        array_sizes_gen = None
        absolute_ps = None
        ngen = None
        nbus = None
        Y_b = None
        Y_a = None
        Y = None
        Y_1 = None
        Y_2 = None
        bus_int = None
        b_ang = None
        V = None
        b_pg = None
        b_qg = None
        g_bus = None
        g_m = None
        g_dtr = None
        g_H = None
        g_do = None
        s1 = None
        s7 = None
        permmod = None
    
    Y=comm.bcast(Y,root=0)
    Y_1=comm.bcast(Y_1,root=0)
    Y_2=comm.bcast(Y_2,root=0)
    permmod=comm.bcast(permmod,root=0)
    Y_b = comm.bcast(Y_b,root=0)
    Y_a = comm.bcast(Y_a,root=0)
    array_sizes_gen = comm.bcast(array_sizes_gen,root=0)
    absolute_ps = comm.bcast(absolute_ps,root=0)
    nbus = comm.bcast(nbus,root=0)
    ngen = comm.bcast(ngen,root=0)
    bus_int = comm.bcast(bus_int,root=0)
    b_ang = comm.bcast(b_ang,root=0)
    V = comm.bcast(V,root=0)
    b_pg = comm.bcast(b_pg,root=0)
    b_qg = comm.bcast(b_qg,root=0)
    g_m = comm.bcast(g_m,root=0)
    g_bus = comm.bcast(g_bus,root=0)
    g_dtr = comm.bcast(g_dtr,root=0)
    g_H = comm.bcast(g_H,root=0)
    g_do = comm.bcast(g_do,root=0)
    s1 = comm.bcast(s1,root=0)
    s7 = comm.bcast(s7,root=0)
    
    t33=MPI.Wtime()
    if size >= 3: 
        if rank < 3:
            Ys = []
            Ys = [Y,Y_1,Y_2]
            Ys[rank][g_bus[:,None]-1,g_bus-1]=Ys[rank][g_bus[:,None]-1,g_bus-1]+permmod[:]
            Ys[rank]=Ys[rank].tocsc()

        else:
            Ys = [None, None, None]
        Y = comm.bcast(Ys[0],root=0)
        Y_1 = comm.bcast(Ys[1],root=1)
        Y_2 = comm.bcast(Ys[2],root=2)

    else:
        if rank == 0:
            Y[g_bus[:,None]-1,g_bus-1]=Y[g_bus[:,None]-1,g_bus-1]+permmod[:]
            Y_1[g_bus[:,None]-1,g_bus-1]=Y_1[g_bus[:,None]-1,g_bus-1]+permmod[:]
            Y_2[g_bus[:,None]-1,g_bus-1]=Y_2[g_bus[:,None]-1,g_bus-1]+permmod[:]

        Y=comm.bcast(Y,root=0)
        Y_1=comm.bcast(Y_1,root=0)
        Y_2=comm.bcast(Y_2,root=0)
    t44=MPI.Wtime()

    prefrecV1=csc_matrix(-solver(Y,Y_b[absolute_ps[rank]:absolute_ps[rank+1]].T,sys.argv[-2]))
    
    # woodbury for on-fault
    B=Y_1-Y
    X=find(B,sys.argv[-2])
    U=csc_matrix((nbus,1),dtype=complex128)
    U[X[0][0]]=1.0
    C=csc_matrix(X[2][0],dtype=complex128)
    A=U.T
    
    zz=csc_matrix(solver(Y,U,sys.argv[-2]).reshape(nbus,1))

    M=csc_matrix(inv(C,sys.argv[-2]))+A.dot(zz)
    M=csc_matrix(inv(M,sys.argv[-2]))
    
    frecV1=prefrecV1-(zz.dot(M).dot(A).dot(prefrecV1))
    
    # woodbury for post-fault
    B=Y_2-Y_1
    X=find(B,sys.argv[-2])
    U=csc_matrix((nbus,2),dtype=complex128)
    U[X[0][0],0]=1.0
    U[X[0][2],1]=1.0

    C=csc_matrix((2,2),dtype=complex128)
    C[0,0]=B[min(X[0]),min(X[1])]
    C[0,1]=B[min(X[0]),max(X[1])]
    C[1,0]=B[max(X[0]),min(X[1])]
    C[1,1]=B[max(X[0]),max(X[1])]

    A=csc_matrix((2,nbus),dtype=np.complex128)
    A[0,X[1][2]]=1.0
    A[1,X[1][3]]=1.0
    
    zz=csc_matrix(solver(Y_1,U,sys.argv[-2]))

    M=csc_matrix(inv(C,sys.argv[-2]))+A.dot(zz)
    M=csc_matrix(inv(M,sys.argv[-2]))
    
    posfrecV1=frecV1-(zz.dot(M).dot(A).dot(frecV1))
    
    prefY11=Y_a[absolute_ps[rank]:absolute_ps[rank+1]]+Y_b.dot(prefrecV1).T
    fY11=Y_a[absolute_ps[rank]:absolute_ps[rank+1]]+Y_b.dot(frecV1).T
    posfY11=Y_a[absolute_ps[rank]:absolute_ps[rank+1]]+Y_b.dot(posfrecV1).T

    if sys.argv[-2] == 'gpu':
        prefY11=prefY11.get()
        fY11=fY11.get()
        posfY11=posfY11.get()
        bus_int=cp.asnumpy(bus_int)
        b_ang=cp.asnumpy(b_ang)
        V=cp.asnumpy(V)
        b_pg=cp.asnumpy(b_pg)
        b_qg=cp.asnumpy(b_qg)
        g_m=cp.asnumpy(g_m)
        g_bus=cp.asnumpy(g_bus)
        g_dtr=cp.asnumpy(g_dtr)
        g_H=cp.asnumpy(g_H)
        g_do=cp.asnumpy(g_do)
        s1=cp.asnumpy(s1)
        s7=cp.asnumpy(s7)

    # Start of simulation
    basrad=2*np.pi*sys_freq
    t_step=np.around((s1[1:]-s1[:-1])/s7[:-1])
    t_width=(s1[1:]-s1[:-1])/t_step[:]
    sim_k=int(t_step.sum())
    sim_k=sim_k+1

    mac_ang,mac_spd,dmac_ang,dmac_spd=np.empty((4,sim_k,array_sizes_gen[rank]),dtype=np.float64)

    eprime=np.empty((array_sizes_gen[rank]),dtype=np.complex128)
    pelect=np.empty((array_sizes_gen[rank]),dtype=np.float64)

    itemsize = MPI.DOUBLE_COMPLEX.Get_size() 
    if rank == 0:
        nbytes=ngen*itemsize
    else:
        nbytes = 0
    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)
    buf, itemsize = win.Shared_query(0)
    eprime_all=np.ndarray((ngen),dtype=np.complex128,buffer=buf)

    theta=np.radians(b_ang)
    bus_volt=V*np.exp(jay*theta)
    mva=basmva/g_m[absolute_ps[rank]:absolute_ps[rank+1]]
    tst1=bus_int[g_bus[absolute_ps[rank]:absolute_ps[rank+1]]-1].astype(int)
    eterm=V[tst1-1]            # terminal bus voltage
    pelect=b_pg[tst1-1]        # BUS_pg
    qelect=b_qg[tst1-1]        # BUS_qg

    # compute the initial values for generator dynamics
    curr=np.hypot(pelect,qelect)/eterm*mva
    phi=np.arctan2(qelect,pelect)

    v=eterm*np.exp(jay*theta[tst1-1])
    curr=curr*np.exp(jay*(theta[tst1-1]-phi))
    eprime=v+jay*g_dtr[absolute_ps[rank]:absolute_ps[rank+1]]*curr

    mac_ang[0]=np.arctan2(eprime.imag,eprime.real)
    mac_spd[0]=1.0
    rot=jay*np.exp(-jay*mac_ang[0])
    eprime=eprime*rot
    edprime=np.copy(eprime.real)
    eqprime=np.copy(eprime.imag)
    pmech=np.copy(pelect*mva)

    steps3=int(t_step.sum())
    steps2=int(t_step[:2].sum())
    steps1=int(t_step[0])
    h_sol1=t_width[0]
    h_sol2=h_sol1

    t1 = time.time()

    for I_Steps in range(1,sim_k+2):
        #determine fault conditions 
        if I_Steps<=steps1:
            S_Steps=I_Steps
            flagF1=0
        elif I_Steps==steps1+1:
            S_Steps=I_Steps
            flagF1=1
        elif steps1+1<I_Steps<=steps2+1:
            S_Steps=I_Steps-1
            flagF1=1
        elif I_Steps==steps2+2:
            S_Steps=I_Steps-1
            flagF1=2
        elif I_Steps>steps2+2:
            S_Steps=I_Steps-2
            flagF1=2

        # compute internal voltage eprime based on q-axis voltage eqprime and generator angle genAngle
        eprime = np.sin(mac_ang[S_Steps-1])*edprime+np.cos(mac_ang[S_Steps-1])*eqprime-jay*(np.cos(mac_ang[S_Steps-1])*edprime-np.sin(mac_ang[S_Steps-1])*eqprime)

        eprime_all[absolute_ps[rank]:absolute_ps[rank+1]] = eprime
        comm.Barrier()
        # compute current based on reduced Y-Bus prefy/fy/posfy and eprime
        if flagF1 == 0:
            cur = prefY11.dot(eprime_all)
        if flagF1 == 1:
            cur = fY11.dot(eprime_all)
        if flagF1 == 2:
            cur = posfY11.dot(eprime_all)

        # compute generators electric real power output pElect;
        curd = np.sin(mac_ang[S_Steps-1])*cur.real-np.cos(mac_ang[S_Steps-1])*cur.imag
        curq = np.cos(mac_ang[S_Steps-1])*cur.real+np.sin(mac_ang[S_Steps-1])*cur.imag

        curdg = curd*mva
        curqg = curq*mva
        ed = edprime+g_dtr[absolute_ps[rank]:absolute_ps[rank+1]]*curqg
        eq = eqprime-g_dtr[absolute_ps[rank]:absolute_ps[rank+1]]*curdg

        eterm = np.hypot(ed,eq)
        pelect = eq*curq+ed*curd
        qelect = eq*curd-ed*curq

        # f(y)
        dmac_ang[S_Steps-1] = basrad*(mac_spd[S_Steps-1]-1.0)
        dmac_spd[S_Steps-1] = (pmech-mva*pelect-g_do[absolute_ps[rank]:absolute_ps[rank+1]]*(mac_spd[S_Steps-1]-1.0))/(2.0*g_H[absolute_ps[rank]:absolute_ps[rank+1]])

        # 3 steps Adam-Bashforth integration steps
        mac_ang = m_ang_ab(dmac_ang, mac_ang, mac_spd[S_Steps-1], S_Steps, h_sol1, basrad)
        mac_spd = m_spd_ab(dmac_spd, mac_spd, pmech, mva, pelect, g_do[absolute_ps[rank]:absolute_ps[rank+1]], g_H[absolute_ps[rank]:absolute_ps[rank+1]], S_Steps, h_sol1)

    t2 = time.time()
    end=time.time()

    if rank == 0:
        print('Simulation kernel:', round(t2-t1,2), 'seconds')
        print('Program finish in:', round(end-start,2), 'seconds')
        #print(mac_ang)