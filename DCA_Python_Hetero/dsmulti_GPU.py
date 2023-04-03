import cupy as cp
import time
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import splu
from numba import cuda, types, float64, complex128, vectorize
import math 

jay=1j
basmva=100
sys_freq=60

@cuda.jit
def fast_sim(prefY11, fY11, posfY11, eprime, C, mac_ang, mac_spd, edprime, eqprime, g_dtr, eterm, pelect, qelect, dmac_ang, dmac_spd, pmech, g_do, g_H, steps1, steps2, mva, basmva, h_sol1, jay, basrad, curq, curd, curqg, curdg, ed, eq, k1_mac_ang, k1_mac_spd, k2_mac_ang, k2_mac_spd, k3_mac_ang, k3_mac_spd, k4_mac_ang, k4_mac_spd, sim_k):
    
    row = cuda.grid(1)
    tid = cuda.threadIdx.x
    #print(ty)
    #h_sol1 = 0.005
    # Define an array in the shared memory for fast matrix multiplication for 2D
    # The size and type of the arrays must be known at compile time
    #sA = cuda.shared.array(shape=(TPB, TPB), dtype=complex128)
    #sB = cuda.shared.array(shape=(TPB), dtype=complex128)

    # Each thread computes one element in the result matrix.
    for I_Steps in range(1,sim_k+2):
        #cuda.syncthreads()
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
            
        # compute eprime
        eprime[S_Steps-1,tid] = math.sin(mac_ang[S_Steps-1,tid])*edprime[tid] + math.cos(mac_ang[S_Steps-1,tid])*eqprime[tid] - jay*(math.cos(mac_ang[S_Steps-1,tid])*edprime[tid] - math.sin(mac_ang[S_Steps-1,tid])*eqprime[tid])

        # The dot product is chunked into dot products of TPB-long vectors.
        if flagF1 == 0:
            A = prefY11
        if flagF1 == 1:
            A = fY11
        if flagF1 == 2:
            A = posfY11

        B = eprime[S_Steps-1]
        #C = cur
        if (tid < A.shape[0]):
            sum = 0

            for i in range(A.shape[1]):
                sum += A[tid, i] * B[i]

            C[tid] = sum

        #compute generators electric real power output pElect
        curd[tid] = math.sin(mac_ang[S_Steps-1,tid])*C[tid].real - math.cos(mac_ang[S_Steps-1,tid])*C[tid].imag
        curq[tid] = math.cos(mac_ang[S_Steps-1,tid])*C[tid].real + math.sin(mac_ang[S_Steps-1,tid])*C[tid].imag

        curdg[tid] = curd[tid]*mva[tid]
        curqg[tid] = curq[tid]*mva[tid]

        ed[tid] = edprime[tid] + g_dtr[tid]*curqg[tid]
        eq[tid] = eqprime[tid] - g_dtr[tid]*curdg[tid]

        eterm[tid] = math.hypot(ed[tid],eq[tid])
        pelect[S_Steps-1,tid] = eq[tid]*curq[tid] + ed[tid]*curd[tid]
        qelect[tid] = eq[tid]*curd[tid] - ed[tid]*curq[tid]

        #f(y)
        dmac_ang[S_Steps-1,tid] = basrad*(mac_spd[S_Steps-1,tid]-1.0)
        dmac_spd[S_Steps-1,tid] = (pmech[tid]-mva[tid]*pelect[S_Steps-1,tid]-g_do[tid]*(mac_spd[S_Steps-1,tid]-1.0))/(2.0*g_H[tid])

        if S_Steps-1 < 2:
            k1_mac_ang[tid] = h_sol1*dmac_ang[S_Steps-1,tid]
            k1_mac_spd[tid] = h_sol1*dmac_spd[S_Steps-1,tid]

            k2_mac_ang[tid] = h_sol1*(basrad*(mac_spd[S_Steps-1,tid]+k1_mac_ang[tid]/2.0-1.0))
            k2_mac_spd[tid] = h_sol1*(pmech[tid]-mva[tid]*pelect[S_Steps-1,tid]-g_do[tid]*(mac_spd[S_Steps-1,tid]+k1_mac_spd[tid]/2.0-1.0))/(2.0*g_H[tid])

            k3_mac_ang[tid] = h_sol1*(basrad*(mac_spd[S_Steps-1,tid]+k2_mac_ang[tid]/2.0-1.0))
            k3_mac_spd[tid] = h_sol1*(pmech[tid]-mva[tid]*pelect[S_Steps-1,tid]-g_do[tid]*(mac_spd[S_Steps-1,tid]+k2_mac_spd[tid]/2.0-1.0))/(2.0*g_H[tid])

            k4_mac_ang[tid] = h_sol1*(basrad*(mac_spd[S_Steps-1,tid]+k3_mac_ang[tid]-1.0))
            k4_mac_spd[tid] = h_sol1*(pmech[tid]-mva[tid]*pelect[S_Steps-1,tid]-g_do[tid]*(mac_spd[S_Steps-1,tid]+k3_mac_spd[tid]-1.0))/(2.0*g_H[tid])

            mac_ang[S_Steps,tid] = mac_ang[S_Steps-1,tid]+(k1_mac_ang[tid]+2*k2_mac_ang[tid]+2*k3_mac_ang[tid]+k4_mac_ang[tid])/6.0
            mac_spd[S_Steps,tid] = mac_spd[S_Steps-1,tid]+(k1_mac_spd[tid]+2*k2_mac_spd[tid]+2*k3_mac_spd[tid]+k4_mac_spd[tid])/6.0
        else:
            mac_ang[S_Steps,tid] = mac_ang[S_Steps-1,tid]+h_sol1*(23*dmac_ang[S_Steps-1,tid]-16*dmac_ang[S_Steps-2,tid]+5*dmac_ang[S_Steps-3,tid])/12.0
            mac_spd[S_Steps,tid] = mac_spd[S_Steps-1,tid]+h_sol1*(23*dmac_spd[S_Steps-1,tid]-16*dmac_spd[S_Steps-2,tid]+5*dmac_spd[S_Steps-3,tid])/12.0


def init(inputfile):
    
    n=0
    m=0
    x=[]
    with open("input/"+inputfile) as f:
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
    f_type=ipt_switch[1:nsw-2]
    s1=ipt_switch[:,0]
    s7=ipt_switch[:,6]

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

    # Line impedance
    # Determine connection matrices including tap chargers and phase shifters
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

    return prefY11,Y_1,Y_2,nPV,nbus,nbrch,ngen,nsw,\
           bus_int,V,b_ang,b_pg,b_qg,Pl,Ql,Gb,Bb,b_type,\
           from_bus,to_bus,r,rx,chrg,liney_ratio,phase_shift,\
           g_bus,g_m,g_r,g_dtr,g_dstr,g_H,g_do,f_type,s1,s7,\
           permmod,Y_c,Y_b,Y_a,r,rx,yyfull,c_line,yl


def reduced_Y(Y_1,Y_2,f_type,bus_int,Bb,Gb,yl,g_bus,permmod,Y_c,Y_b,Y_a,r,rx,yyfull,c_line,from_bus,to_bus,case_id):
      
    if len(f_type) == 1:
        case_id = 0
    f_type=f_type[case_id]
    f_nearbus=f_type[1].astype(int)
    bus_idx=bus_int[f_nearbus-1]
    bus_idx=bus_idx.astype(int)
    Bb[bus_idx-1]=10000000.0
    # fY11=reduce_y(flag=1)
    Y_1.setdiag(Y_1.diagonal()+yl+Gb+jay*Bb)
    Y_1[g_bus[:,None]-1,g_bus-1]=Y_1[g_bus[:,None]-1,g_bus-1]+permmod[:]
    
    frecV1=csr_matrix(-splu(Y_1).solve(Y_c))
    temp=Y_b.dot(frecV1)
    fY11=(Y_a+temp.transpose()).toarray()
    
    f_farbus=f_type[2].astype(int)
    Bb[bus_idx-1]=0.0
    i=cp.where(cp.logical_and(from_bus==f_nearbus, to_bus==f_farbus))
    rx[i]=10000000.0
    j=cp.where(cp.logical_and(from_bus==f_farbus, to_bus==f_nearbus))
    rx[j]=10000000.0
    
    z=r+jay*rx
    yy=1/z
    
    yyfull.setdiag(yy)
    Y_dummy=yyfull.dot(c_line.transpose())
    Y_2=(c_line.dot(Y_dummy)+Y_2)

    Y_2.setdiag(Y_2.diagonal()+yl+Gb+jay*Bb)
    Y_2[g_bus[:,None]-1,g_bus-1]=Y_2[g_bus[:,None]-1,g_bus-1]+permmod[:]
    
    posfrecV1=csr_matrix(-splu(Y_2).solve(Y_c))
    temp=Y_b.dot(posfrecV1)
    posfY11=(Y_a+temp.transpose()).toarray()
    #print(case_id)

    return fY11,posfY11
    

def red_Y_sim(prefY11,fY11,posfY11,bus_int,b_ang,V,b_pg,b_qg,g_m,g_bus,g_dtr,g_H,g_do,ngen,s1,s7):
    
    basrad=2*cp.pi*sys_freq
    # Start of simulation
    t_step=cp.around((s1[1:]-s1[:-1])/s7[:-1])
    t_width=(s1[1:]-s1[:-1])/t_step[:]
    sim_k=int(t_step.sum())
    sim_k=sim_k+1

    dmac_ang=cp.zeros((sim_k,ngen),dtype=cp.float64)
    dmac_spd=cp.zeros((sim_k,ngen),dtype=cp.float64)
    pelect=cp.zeros((sim_k,ngen),dtype=cp.float64)
    mac_ang=cp.zeros((sim_k,ngen),dtype=cp.float64)
    mac_spd=cp.zeros((sim_k,ngen),dtype=cp.float64)
    eprime=cp.zeros((sim_k,ngen),dtype=cp.complex128)
    curq,curd,curqg,curdg,ed,eq,k1_mac_ang,k1_mac_spd,k2_mac_ang,k2_mac_spd,k3_mac_ang,k3_mac_spd,k4_mac_ang,k4_mac_spd=cp.zeros((14,ngen),dtype=cp.float64)
    C = cp.empty(ngen).astype(cp.complex128)
    theta=cp.radians(b_ang)
    bus_volt=V*cp.exp(jay*theta)
    mva=basmva/g_m
    tst1=bus_int[g_bus-1].astype(int)
    eterm=V[tst1-1] # terminal bus voltage
    pelect[0]=b_pg[tst1-1]     # BUS_pg
    qelect=b_qg[tst1-1]     # BUS_qg
    #compute the initial values for generator dynamics
    curr=cp.hypot(pelect[0],qelect)/(eterm*mva)
    phi=cp.arctan2(qelect,pelect[0])
    v=eterm*cp.exp(jay*theta[tst1-1])
    curr=curr*cp.exp(jay*(theta[tst1-1]-phi))
    eprime[0]=v+jay*g_dtr*curr
    mac_ang[0]=cp.arctan2(eprime[0].imag,eprime[0].real)
    mac_spd[0]=1.0
    rot=jay*cp.exp(-jay*mac_ang[0])
    eprime[0]=eprime[0]*rot
    edprime=cp.copy(eprime[0].real)
    eqprime=cp.copy(eprime[0].imag)
    pmech=cp.copy(pelect[0]*mva)

    steps3=int(t_step.sum())
    steps2=int(t_step[:2].sum())
    steps1=int(t_step[0])
    h_sol1=float(t_width[0])
    h_sol2=h_sol1

    fast_sim[1, ngen](prefY11, fY11, posfY11, eprime, C, mac_ang, mac_spd, edprime, eqprime, g_dtr, eterm, pelect, qelect, dmac_ang, dmac_spd, pmech, g_do, g_H, steps1, steps2, mva, basmva, h_sol1, jay, basrad, curq, curd, curqg, curdg, ed, eq, k1_mac_ang, k1_mac_spd, k2_mac_ang, k2_mac_spd, k3_mac_ang, k3_mac_spd, k4_mac_ang, k4_mac_spd, sim_k)
    
    #print("Simulation finished")
    return mac_ang, mac_spd

# define a python dynamic simulation class
class DynSim_GPU:

    def __init__(self, filename, n_directions, n_faults, t_sim):
        
        self.filename = filename
        self.n_directions = n_directions
        self.n_faults = n_faults
        self.t_sim = t_sim
        
    # data read in    
    def init(self):
        
        self.prefY11,self.Y_1,self.Y_2,self.nPV,self.nbus,\
        self.nbrch,self.ngen,self.nsw,self.bus_int,self.V,\
        self.b_ang,self.b_pg,self.b_qg,self.Pl,self.Ql,\
        self.Gb,self.Bb,self.b_type,self.from_bus,self.to_bus,\
        self.r,self.rx,self.chrg,self.liney_ratio,self.phase_shift,\
        self.g_bus,self.g_m,self.g_r,self.g_dtr,self.g_dstr,\
        self.g_H,self.g_do,self.f_type,self.s1,self.s7,\
        self.permmod,self.Y_c,self.Y_b,self.Y_a,self.r,self.rx,\
        self.yyfull,self.c_line,self.yl = init(self.filename)
        
        return self.prefY11
    
    # build Y matrix
    def reduced_Y(self, case_id):
    
        self.fY11,\
        self.posfY11 = reduced_Y(self.Y_1,self.Y_2,self.f_type,self.bus_int,\
                                 self.Bb,self.Gb,self.yl,self.g_bus,self.permmod,\
                                 self.Y_c,self.Y_b,self.Y_a,self.r,self.rx,\
                                 self.yyfull,self.c_line,self.from_bus,self.to_bus,case_id)
        
        return
        #return self.fY11, self.posfY11
    
    def reduced_Y_solver(self):
        
        self.mac_ang,\
        self.mac_spd = red_Y_sim(self.prefY11,self.fY11,self.posfY11,\
                                    self.bus_int,self.b_ang,self.V,self.b_pg,\
                                    self.b_qg,self.g_m,self.g_bus,self.g_dtr,\
                                    self.g_H,self.g_do,self.ngen,self.s1,self.s7)
        
        return self.mac_ang, self.mac_spd