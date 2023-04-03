import cupy as cp
from mpi4py import MPI
import time
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import splu
import math 
from numba import cuda, types, float64, complex128
import sys

@cuda.jit
def fast_sim(prefY11, fY11, posfY11, eprime, C, mac_ang, mac_spd, edprime, eqprime, g_dtr, eterm, pelect, qelect, dmac_ang, dmac_spd, pmech, g_do, g_H, steps1, steps2, mva, basmva, h_sol1, jay, basrad, curq, curd, curqg, curdg, ed, eq, k1_mac_ang, k1_mac_spd, k2_mac_ang, k2_mac_spd, k3_mac_ang, k3_mac_spd, k4_mac_ang, k4_mac_spd, sim_k, ngen):
    
    tid = cuda.grid(1)
    #print(tid)
    if tid < ngen:
        # Each thread computes one element in the result matrix.
        for I_Steps in range(1,sim_k+2):
            # determine fault conditions 
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

            # compute eprime
            eprime[tid] = math.sin(mac_ang[S_Steps-1,tid])*edprime[tid] + math.cos(mac_ang[S_Steps-1,tid])*eqprime[tid] - jay*(math.cos(mac_ang[S_Steps-1,tid])*edprime[tid] - math.sin(mac_ang[S_Steps-1,tid])*eqprime[tid])
            cuda.syncthreads()

            # The dot product is chunked into dot products of TPB-long vectors.
            if flagF1 == 0:
                A = prefY11
            if flagF1 == 1:
                A = fY11
            if flagF1 == 2:
                A = posfY11
            
            # compute current current based on reduced Y-Bus prefy/fy/posfy and eprime
            C[tid] = 0    
            for i in range(ngen):
                C[tid] += A[tid,i]*eprime[i]
                cuda.syncthreads()
            cuda.syncthreads()

            # compute generators electric real power output pElect
            curd[tid] = math.sin(mac_ang[S_Steps-1,tid])*C[tid].real - math.cos(mac_ang[S_Steps-1,tid])*C[tid].imag
            curq[tid] = math.cos(mac_ang[S_Steps-1,tid])*C[tid].real + math.sin(mac_ang[S_Steps-1,tid])*C[tid].imag
            
            curdg[tid] = curd[tid]*mva[tid]
            curqg[tid] = curq[tid]*mva[tid]

            ed[tid] = edprime[tid] + g_dtr[tid]*curqg[tid]
            eq[tid] = eqprime[tid] - g_dtr[tid]*curdg[tid]

            eterm[tid] = math.hypot(ed[tid],eq[tid])
            pelect[tid] = eq[tid]*curq[tid] + ed[tid]*curd[tid]
            qelect[tid] = eq[tid]*curd[tid] - ed[tid]*curq[tid]

            # f(y)
            dmac_ang[S_Steps-1,tid] = basrad*(mac_spd[S_Steps-1,tid]-1.0)
            dmac_spd[S_Steps-1,tid] = (pmech[tid]-mva[tid]*pelect[tid]-g_do[tid]*(mac_spd[S_Steps-1,tid]-1.0))/(2.0*g_H[tid])
            
            # 3 steps Adam-Bashforth integration steps
            if S_Steps-1 < 2:
                k1_mac_ang[tid] = h_sol1*dmac_ang[S_Steps-1,tid]
                k1_mac_spd[tid] = h_sol1*dmac_spd[S_Steps-1,tid]

                k2_mac_ang[tid] = h_sol1*(basrad*(mac_spd[S_Steps-1,tid]+k1_mac_ang[tid]/2.0-1.0))
                k2_mac_spd[tid] = h_sol1*(pmech[tid]-mva[tid]*pelect[tid]-g_do[tid]*(mac_spd[S_Steps-1,tid]+k1_mac_spd[tid]/2.0-1.0))/(2.0*g_H[tid])

                k3_mac_ang[tid] = h_sol1*(basrad*(mac_spd[S_Steps-1,tid]+k2_mac_ang[tid]/2.0-1.0))
                k3_mac_spd[tid] = h_sol1*(pmech[tid]-mva[tid]*pelect[tid]-g_do[tid]*(mac_spd[S_Steps-1,tid]+k2_mac_spd[tid]/2.0-1.0))/(2.0*g_H[tid])

                k4_mac_ang[tid] = h_sol1*(basrad*(mac_spd[S_Steps-1,tid]+k3_mac_ang[tid]-1.0))
                k4_mac_spd[tid] = h_sol1*(pmech[tid]-mva[tid]*pelect[tid]-g_do[tid]*(mac_spd[S_Steps-1,tid]+k3_mac_spd[tid]-1.0))/(2.0*g_H[tid])

                mac_ang[S_Steps,tid] = mac_ang[S_Steps-1,tid]+(k1_mac_ang[tid]+2*k2_mac_ang[tid]+2*k3_mac_ang[tid]+k4_mac_ang[tid])/6.0
                mac_spd[S_Steps,tid] = mac_spd[S_Steps-1,tid]+(k1_mac_spd[tid]+2*k2_mac_spd[tid]+2*k3_mac_spd[tid]+k4_mac_spd[tid])/6.0

            else:
                mac_ang[S_Steps,tid] = mac_ang[S_Steps-1,tid]+h_sol1*(23*dmac_ang[S_Steps-1,tid]-16*dmac_ang[S_Steps-2,tid]+5*dmac_ang[S_Steps-3,tid])/12.0
                mac_spd[S_Steps,tid] = mac_spd[S_Steps-1,tid]+h_sol1*(23*dmac_spd[S_Steps-1,tid]-16*dmac_spd[S_Steps-2,tid]+5*dmac_spd[S_Steps-3,tid])/12.0

# the main program start        
jay=1j
basmva=100
sys_freq=60
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

t0 = time.time()
if rank == 0:
    cp.cuda.runtime.setDevice(cp.cuda.runtime.getDevice())
    n=0
    m=0
    x=[]
    with open("input/"+sys.argv[-1]) as f:
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
    
    ipt_bus=cp.array(a,dtype=cp.float64)
    ipt_brch=cp.array(b,dtype=cp.float64)
    ipt_gen=cp.array(c,dtype=cp.float64)
    ipt_switch=cp.array(d,dtype=cp.float64)
    nPV=cp.count_nonzero(ipt_bus[:,9]==2)
    #print("Data read in successfully!")
    #print("nbus:", nbus, " nbrch:", nbrch, " ngen:", ngen)
    
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
    f_type=ipt_switch[1:int(sys.argv[-4])][int(sys.argv[-2])::int(sys.argv[-3])]
    #print(cp.cuda.runtime.getDeviceCount())
    s1=ipt_switch[:,0]
    s7=ipt_switch[:,6]
    iterations = int(f_type.shape[0]/size)

    a=cp.arange(nbrch)
    tap=cp.ones((nbrch),dtype=cp.complex128)
    c_from=csr_matrix((nbus,nbrch),dtype=cp.complex128)
    c_line=csr_matrix((nbus,nbrch),dtype=cp.complex128)
    c_to=csr_matrix((nbus,nbrch),dtype=cp.complex128)
    chrgfull=csr_matrix((nbrch,nbrch),dtype=cp.complex128)
    yyfull=csr_matrix((nbrch,nbrch),dtype=cp.complex128)
    xd=cp.zeros(ngen)
    sd=cp.ones(ngen)
    perm=csr_matrix((ngen,ngen),dtype=cp.complex128)
    Y_b=csr_matrix((ngen,nbus),dtype=cp.complex128)
    Y_a=csr_matrix((ngen,ngen),dtype=cp.complex128)
    
    # prefY11=reduce_y(flag=0)
    z=r+jay*rx
    yy=1/z
    from_int=bus_int[from_bus-1].astype(int)
    to_int=bus_int[to_bus-1].astype(int)
    tap[liney_ratio>0]=cp.exp((-jay*phase_shift[liney_ratio>0])*cp.pi/180)/liney_ratio[liney_ratio>0]
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

    if ngen != nPV:
        e=cp.where(~cp.eye(perm.get_shape()[0],dtype=bool))
        h=cp.where(g_bus[e[0]]==g_bus[e[1]])
        perm[e[0][h],e[1][h]]=1
        permPV=perm
    else:
        permPV=perm

    Ymod=Y_a.dot(permPV.transpose())
    permmod=permPV.transpose().dot(Ymod)
    Y_b[:,g_bus-1]=-Ymod.transpose()[:]
    Y.setdiag(Y.diagonal()+yl+Gb+jay*Bb)
    Y[g_bus[:,None]-1,g_bus-1]=Y[g_bus[:,None]-1,g_bus-1]+permmod[:]
    Y_c=Y_b.transpose().toarray()
    prefrecV1=csr_matrix(-splu(Y).solve(Y_c))
    temp=Y_b.dot(prefrecV1)
    prefY11=(Y_a+temp.transpose()).toarray()

else:
    cp.cuda.runtime.setDevice(cp.cuda.runtime.getDevice())
    #print(cp.cuda.runtime.getDevice())
    prefY11 = None
    Y_1 = None
    Y_2 = None
    nPV = None
    nbus = None
    nbrch = None
    ngen = None
    nsw = None
    bus_int = None
    V = None
    b_ang = None
    b_pg = None
    b_qg = None
    Pl = None
    Ql = None
    Gb = None
    Bb = None
    b_type = None
    from_bus = None
    to_bus = None
    chrg = None
    liney_ratio = None
    phase_shift = None
    g_bus = None
    g_m = None
    g_r = None
    g_dtr = None
    g_dstr = None
    g_H = None
    g_do = None
    f_type = None
    s1 = None
    s7 = None
    permmod = None
    Y_c = None
    Y_b = None
    Y_a = None
    r = None
    rx = None
    yyfull = None
    c_line = None
    yl = None
    iterations = None
    
prefY11 = comm.bcast(prefY11,root=0)
Y_1 = comm.bcast(Y_1,root=0)
Y_2 = comm.bcast(Y_2,root=0)
nPV = comm.bcast(nPV,root=0)
nbus = comm.bcast(nbus,root=0)
nbrch = comm.bcast(nbrch,root=0)
ngen = comm.bcast(ngen,root=0)
nsw = comm.bcast(nsw,root=0)
bus_int = comm.bcast(bus_int,root=0)
V = comm.bcast(V,root=0)
b_ang = comm.bcast(b_ang,root=0)
b_pg = comm.bcast(b_pg,root=0)
b_qg = comm.bcast(b_qg,root=0)
Pl = comm.bcast(Pl,root=0)
Ql = comm.bcast(Ql,root=0)
Gb = comm.bcast(Gb,root=0)
Bb = comm.bcast(Bb,root=0)
b_type = comm.bcast(b_type,root=0)
from_bus = comm.bcast(from_bus,root=0)
to_bus = comm.bcast(to_bus,root=0)
chrg = comm.bcast(chrg,root=0)
liney_ratio = comm.bcast(liney_ratio,root=0)
phase_shift = comm.bcast(phase_shift,root=0)
g_bus = comm.bcast(g_bus,root=0)
g_m = comm.bcast(g_m,root=0)
g_r = comm.bcast(g_r,root=0)
g_dtr = comm.bcast(g_dtr,root=0)
g_dstr = comm.bcast(g_dstr,root=0)
g_H = comm.bcast(g_H,root=0)
g_do = comm.bcast(g_do,root=0)
f_type = comm.bcast(f_type,root=0)
s1 = comm.bcast(s1,root=0)
s7 = comm.bcast(s7,root=0)
permmod = comm.bcast(permmod,root=0)
Y_c = comm.bcast(Y_c,root=0)
Y_b = comm.bcast(Y_b,root=0)
Y_a = comm.bcast(Y_a,root=0)
r = comm.bcast(r,root=0)
rx = comm.bcast(rx,root=0)
yyfull = comm.bcast(yyfull,root=0)
c_line = comm.bcast(c_line,root=0)
yl = comm.bcast(yl,root=0)
iterations = comm.bcast(iterations,root=0)
    
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
#print(f_type,sys.argv[-2])
for ii in range (iterations):
    cp.cuda.runtime.setDevice(cp.cuda.runtime.getDevice())
    case_id=rank+ii*size
    #print(case_id,rank)
    # fY11=reduce_y(flag=1)
    ff_type=f_type[case_id]
    f_nearbus=ff_type[1].astype(int)
    bus_idx=bus_int[f_nearbus-1]
    bus_idx=bus_idx.astype(int)
    Bb[bus_idx-1]=10000000.0
    Y_11=Y_1.copy()
    Y_11.setdiag(Y_11.diagonal()+yl+Gb+jay*Bb)
    Y_11[g_bus[:,None]-1,g_bus-1]=Y_11[g_bus[:,None]-1,g_bus-1]+permmod[:]
    frecV1=csr_matrix(-splu(Y_11).solve(Y_c))
    temp=Y_b.dot(frecV1)
    fY11=(Y_a+temp.transpose()).toarray()

    # posfY11=reduce_y(flag=2)
    f_farbus=ff_type[2].astype(int)
    Bb[bus_idx-1]=0.0
    rxx=rx.copy()
    i=cp.where(cp.logical_and(from_bus==f_nearbus, to_bus==f_farbus))
    rxx[i]=10000000.0
    j=cp.where(cp.logical_and(from_bus==f_farbus, to_bus==f_nearbus))
    rxx[j]=10000000.0
    z=r+jay*rxx
    yy=1/z
    yyfull.setdiag(yy)
    Y_dummy=yyfull.dot(c_line.transpose())
    Y_22=(c_line.dot(Y_dummy)+Y_2)
    Y_22.setdiag(Y_22.diagonal()+yl+Gb+jay*Bb)
    Y_22[g_bus[:,None]-1,g_bus-1]=Y_22[g_bus[:,None]-1,g_bus-1]+permmod[:]
    posfrecV1=csr_matrix(-splu(Y_22).solve(Y_c))
    temp=Y_b.dot(posfrecV1)
    posfY11=(Y_a+temp.transpose()).toarray()

    # start of simulation
    basrad=2*cp.pi*sys_freq
    t_step=cp.around((s1[1:]-s1[:-1])/s7[:-1])
    t_width=(s1[1:]-s1[:-1])/t_step[:]
    sim_k=int(t_step.sum())
    sim_k=sim_k+1

    dmac_ang=cp.zeros((sim_k,ngen),dtype=cp.float64)
    dmac_spd=cp.zeros((sim_k,ngen),dtype=cp.float64)
    pelect=cp.zeros((ngen),dtype=cp.float64)
    mac_ang=cp.zeros((sim_k,ngen),dtype=cp.float64)
    mac_spd=cp.zeros((sim_k,ngen),dtype=cp.float64)
    eprime=cp.zeros((ngen),dtype=cp.complex128)

    curq,curd,curqg,curdg,ed,eq,k1_mac_ang,k1_mac_spd,k2_mac_ang,k2_mac_spd,k3_mac_ang,k3_mac_spd,k4_mac_ang,k4_mac_spd=cp.zeros((14,ngen),dtype=cp.float64)
    C = cp.zeros(ngen).astype(cp.complex128)

    theta=cp.radians(b_ang)
    bus_volt=V*cp.exp(jay*theta)
    mva=basmva/g_m
    tst1=bus_int[g_bus-1].astype(int)
    eterm=V[tst1-1] # terminal bus voltage
    pelect=b_pg[tst1-1]     # BUS_pg
    qelect=b_qg[tst1-1]     # BUS_qg

    # compute the initial values for generator dynamics
    curr=cp.hypot(pelect,qelect)/(eterm*mva)
    phi=cp.arctan2(qelect,pelect)
    v=eterm*cp.exp(jay*theta[tst1-1])
    curr=curr*cp.exp(jay*(theta[tst1-1]-phi))
    eprime=v+jay*g_dtr*curr

    mac_ang[0]=cp.arctan2(eprime.imag,eprime.real)
    mac_spd[0]=1.0
    rot=jay*cp.exp(-jay*mac_ang[0])
    eprime=eprime*rot
    edprime=cp.copy(eprime.real)
    eqprime=cp.copy(eprime.imag)
    pmech=cp.copy(pelect*mva)
    #print(mac_ang.nbytes)
    steps3=int(t_step.sum())
    steps2=int(t_step[:2].sum())
    steps1=int(t_step[0])
    h_sol1=float(t_width[0])
    h_sol2=h_sol1

    # set the number of threads in a block
    threadsperblock = ngen
    blockspergrid = math.ceil(ngen/threadsperblock)
    #print(case_id)
    fast_sim[blockspergrid, threadsperblock](prefY11, fY11, posfY11, eprime, C, mac_ang, mac_spd, edprime, eqprime, g_dtr, eterm, pelect, qelect, dmac_ang, dmac_spd, pmech, g_do, g_H, steps1, steps2, mva, basmva, h_sol1, jay, basrad, curq, curd, curqg, curdg, ed, eq, k1_mac_ang, k1_mac_spd, k2_mac_ang, k2_mac_spd, k3_mac_ang, k3_mac_spd, k4_mac_ang, k4_mac_spd, sim_k, ngen)
    #print(mac_ang[:,0],case_id, mac_ang.device)

end.record()
end.synchronize()
#print(cp.cuda.get_elapsed_time(start, end)/1000, rank)
    
comm.Barrier()
t1 = time.time()
if rank == 0:
    print("GPU", int(sys.argv[-2]), "takes", format(t1-t0,'.2f'), "to finish")