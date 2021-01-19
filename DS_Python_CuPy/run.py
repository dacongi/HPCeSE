import cupy as cp #print('cuPy Version:', cp.__version__)
#import cudf; #print('cuDF Version:', cudf.__version__)
import time
#mempool = cp.get_default_memory_pool()
#with cp.cuda.Device(0):
    #mempool.set_limit(size=1024**3)  # 1 GiB
#print(mempool.get_limit())
# create global variables
n=0
m=0
jay=cp.array([1j])
basmva=cp.array([100])
sys_freq=cp.array([60])
basrad=2*cp.pi*sys_freq
# read in data

x=[]
with open("../DSPython/MPI/input/36001.txt") as f:
    for line in f:
        line = line.split()
        if len(line)==1 and m==0:
            #print(x)
            nbus=n
            ipt_bus=x
            m=m+1
            x=[]
            continue
        elif len(line)==1 and m==1:
            #print(x)
            nbrch=n-nbus
            ipt_brch=x
            m=m+1
            x=[]
            continue
        elif len(line)==1 and m==2:
            #print(x)
            ngen=n-nbrch-nbus
            ipt_gen=x
            m=m+1
            x=[]
            continue
        elif len(line)==1 and m==3:
            #print(x)
            nsw=n-nbus-nbrch-ngen
            ipt_switch=x
            continue
        x.append(line)
        n=n+1
        
ipt_bus=cp.array(ipt_bus,dtype=cp.float64)
ipt_gen=cp.array(ipt_gen,dtype=cp.float64)
ipt_brch= cp.array(ipt_brch,dtype=cp.float64)
ipt_switch=cp.array(ipt_switch,dtype=cp.float64)
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
from_bus=ipt_brch[:,0].astype(cp.int)
to_bus=ipt_brch[:,1].astype(cp.int)
r=ipt_brch[:,2]
rx=ipt_brch[:,3]
chrg=jay*(0.5*ipt_brch[:,4])
liney_ratio=ipt_brch[:,5]
phase_shift=ipt_brch[:,6]
# assign generator data
g_bus=ipt_gen[:,1].astype(cp.int)
g_m=ipt_gen[:,2]
g_r=ipt_gen[:,4]
g_dtr=ipt_gen[:,6]
g_dstr=ipt_gen[:,7]
g_H=ipt_gen[:,15]
g_do=ipt_gen[:,16]
# assign switch data
f_type=ipt_switch[1,5]
s1=ipt_switch[:,0]
s7=ipt_switch[:,6]
#print(mempool.total_bytes())
#print(mempool.get_limit())
# initialize arrays for Y and reduced Y
start = time.time()

a=cp.arange(nbrch)
tap=cp.ones((nbrch),dtype=cp.complex128)
c_from=cp.zeros((nbus,nbrch),dtype=cp.complex128)
c_line=cp.zeros((nbus,nbrch),dtype=cp.complex128)
c_to=cp.zeros((nbus,nbrch),dtype=cp.complex128)
chrgfull=cp.zeros((nbrch,nbrch),dtype=cp.complex128)
yyfull=cp.zeros((nbrch,nbrch),dtype=cp.complex128)
xd=cp.zeros(ngen)
perm=cp.zeros((ngen,ngen),dtype=cp.complex128)
Y_b=cp.zeros((ngen,nbus),dtype=cp.complex128)
Y_a=cp.zeros((ngen,ngen),dtype=cp.complex128)
# prefY11=reduce_y(flag=0)
z=r+jay*rx
yy=1/z
#print(yy)
from_int=bus_int[from_bus-1].astype(cp.int)
to_int=bus_int[to_bus-1].astype(cp.int)
#print(to_int)
tap[liney_ratio>0]=cp.exp((-jay*phase_shift[liney_ratio>0])*cp.pi/180)/liney_ratio[liney_ratio>0]
from_int=from_int-1
to_int=to_int-1
#print(from_int)
# Line impedance
# Determine connection matrices including tap chargers and phase shifters
# sparse matrix formulation
c_from[from_int,a]=tap[a]
c_to[to_int,a]=1
c_line[from_int,a]=c_from[from_int,a]-c_to[from_int,a]
c_line[to_int,a]=c_from[to_int,a]-c_to[to_int,a]
# Form Y matrix from primative line ys and connection matrices
cp.fill_diagonal(chrgfull, chrg)
cp.fill_diagonal(yyfull, yy)
#print(c_from)
Y_dummy=cp.matmul(chrgfull,c_from.T)
Y=cp.matmul(c_from,Y_dummy)
Y_dummy=cp.matmul(chrgfull,c_to.T)
Y=cp.matmul(c_to,Y_dummy)+Y
Y_2=cp.copy(Y)
Y_dummy=cp.matmul(yyfull,c_line.T)
Y=cp.matmul(c_line,Y_dummy)+Y
#print(Y)
Pl[b_type==3]=Pl[b_type==3]-b_pg[b_type==3]
Ql[b_type==3]=Ql[b_type==3]-b_qg[b_type==3]
yl=(Pl-jay*Ql)/(V*V)

ra=g_r*basmva/g_m
xd[g_dstr==0]=g_dtr[g_dstr==0]*basmva/g_m[g_dstr==0]
y=1/(ra+jay*xd)
cp.fill_diagonal(perm, 1)
cp.fill_diagonal(Y_a, y)

if ngen != nPV:
    e=cp.where(~cp.eye(perm.shape[0],dtype=bool))
    h=cp.where(g_bus[e[0]]==g_bus[e[1]])
    perm[e[0][h],e[1][h]]=1
    permPV=perm
else:
    permPV=perm

Ymod=cp.matmul(Y_a,permPV.T)
permmod=cp.matmul(permPV.T,Ymod)
# print(Ymod)    
Y_b[:,g_bus-1]=-Ymod.T[:]
Y_1=cp.copy(Y)
cp.fill_diagonal(Y,Y.diagonal()+yl+Gb+jay*Bb)
Y[g_bus[:,None]-1,g_bus-1]=Y[g_bus[:,None]-1,g_bus-1]+permmod[:]

Y_c=Y_b.T
prefrecV1=-cp.linalg.solve(Y,Y_c)
temp=cp.matmul(Y_b,prefrecV1)
prefY11=Y_a+temp.T
#print(prefY11)
# if f_type < 4:
f_nearbus=ipt_switch[1,1].astype(cp.int)
bus_idx=bus_int[f_nearbus-1]
bus_idx=bus_idx.astype(cp.int)
Bb[bus_idx-1]=10000000.0
# fY11=reduce_y(flag=1)
cp.fill_diagonal(Y_1,Y_1.diagonal()+yl+Gb+jay*Bb)
Y_1[g_bus[:,None]-1,g_bus-1]=Y_1[g_bus[:,None]-1,g_bus-1]+permmod[:]
frecV1=-cp.linalg.solve(Y_1,Y_c)
temp=cp.matmul(Y_b,frecV1)
fY11=Y_a+temp.T
#print(fY11)
# if f_type < 4:
f_farbus=ipt_switch[1,2].astype(cp.int)
Bb[bus_idx-1]=0.0
i=cp.where(cp.logical_and(from_bus==f_nearbus, to_bus==f_farbus))
rx[i]=10000000.0
j=cp.where(cp.logical_and(from_bus==f_farbus, to_bus==f_nearbus))
rx[j]=10000000.0
# fY11=reduce_y(flag=2)
z=r+jay*rx
yy=1/z
cp.fill_diagonal(yyfull, yy)
Y_dummy=cp.matmul(yyfull,c_line.T)
Y=cp.matmul(c_line,Y_dummy)+Y_2

cp.fill_diagonal(Y,Y.diagonal()+yl+Gb+jay*Bb)
Y[g_bus[:,None]-1,g_bus-1]=Y[g_bus[:,None]-1,g_bus-1]+permmod[:]
posfrecV1=-cp.linalg.solve(Y,Y_c)
temp=cp.matmul(Y_b,posfrecV1)
posfY11=Y_a+temp.T
#print(posfY11)
#print("Finish reduced Y, start simulation!")
# Start of simulation
t_step=cp.around((s1[1:]-s1[:-1])/s7[:-1])
t_width=(s1[1:]-s1[:-1])/t_step[:]
sim_k=int(t_step.sum())
sim_k=sim_k+1
#print(sim_k)
mac_ang,mac_spd,dmac_ang,dmac_spd,pelect=cp.zeros((5,sim_k,ngen),dtype=cp.float64)
eprime=cp.zeros((sim_k,ngen),dtype=cp.complex128)

theta=cp.radians(b_ang)
bus_volt=V*cp.exp(jay*theta)
mva=basmva/g_m
tst1=bus_int[g_bus-1].astype(cp.int)
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
h_sol1=t_width[0]
h_sol2=h_sol1
#print(mempool.total_bytes())

# numerical integration
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

        #compute internal voltage eprime based on q-axis voltage eqprime and generator angle genAngle
        eprime[S_Steps-1] = cp.sin(mac_ang[S_Steps-1])*edprime+cp.cos(mac_ang[S_Steps-1])*eqprime-jay*(cp.cos(mac_ang[S_Steps-1])*edprime-cp.sin(mac_ang[S_Steps-1])*eqprime)
        
        #compute current current based on reduced Y-Bus prefy/fy/posfy and eprime
        if flagF1 == 0:
            #cur = np.matmul(prefY11,eprime_all)
            cur = cp.matmul(prefY11,eprime[S_Steps-1])
        if flagF1 == 1:
            #cur = np.matmul(fY11,eprime_all)
            cur = cp.matmul(fY11,eprime[S_Steps-1])
        if flagF1 == 2:
            #cur = np.matmul(posfY11,eprime_all)
            cur = cp.matmul(posfY11,eprime[S_Steps-1])

        #compute generators electric real power output pElect;
        curd = cp.sin(mac_ang[S_Steps-1])*cur.real-cp.cos(mac_ang[S_Steps-1])*cur.imag
        curq = cp.cos(mac_ang[S_Steps-1])*cur.real+cp.sin(mac_ang[S_Steps-1])*cur.imag

        curdg = curd*mva
        curqg = curq*mva
        ed = edprime+g_dtr*curqg
        eq = eqprime-g_dtr*curdg
        eterm = cp.hypot(ed,eq)
        pelect[S_Steps-1] = eq*curq+ed*curd
        qelect = eq*curd-ed*curq
        
        #f(y)
        dmac_ang[S_Steps-1] = basrad*(mac_spd[S_Steps-1]-1.0)
        dmac_spd[S_Steps-1] = (pmech-pelect[S_Steps-1]-g_do*(mac_spd[S_Steps-1]-1.0))/(2.0*g_H)

        #3 steps Adam-Bashforth integration steps
        if S_Steps-1 < 2:
            k1_mac_ang = h_sol1*dmac_ang[S_Steps-1]
            k1_mac_spd = h_sol1*dmac_spd[S_Steps-1]

            k2_mac_ang = h_sol1*(basrad*(mac_spd[S_Steps-1]+k1_mac_ang/2.0-1.0))
            k2_mac_spd = h_sol1*(pmech-pelect[S_Steps-1]-g_do*(mac_spd[S_Steps-1]+k1_mac_spd/2.0-1.0))/(2.0*g_H)
           
            k3_mac_ang = h_sol1*(basrad*(mac_spd[S_Steps-1]+k2_mac_ang/2.0-1.0))
            k3_mac_spd = h_sol1*(pmech-pelect[S_Steps-1]-g_do*(mac_spd[S_Steps-1]+k2_mac_spd/2.0-1.0))/(2.0*g_H)
            
            k4_mac_ang = h_sol1*(basrad*(mac_spd[S_Steps-1]+k3_mac_ang-1.0))
            k4_mac_spd = h_sol1*(pmech-pelect[S_Steps-1]-g_do*(mac_spd[S_Steps-1]+k3_mac_spd-1.0))/(2.0*g_H)
            
            mac_ang[S_Steps] = mac_ang[S_Steps-1]+(k1_mac_ang+2*k2_mac_ang+2*k3_mac_ang+k4_mac_ang)/6.0
            mac_spd[S_Steps] = mac_spd[S_Steps-1]+(k1_mac_spd+2*k2_mac_spd+2*k3_mac_spd+k4_mac_spd)/6.0
        else:
            mac_ang[S_Steps] = mac_ang[S_Steps-1]+h_sol1*(23*dmac_ang[S_Steps-1]-16*dmac_ang[S_Steps-2]+5*dmac_ang[S_Steps-3])/12.0
            mac_spd[S_Steps] = mac_spd[S_Steps-1]+h_sol1*(23*dmac_spd[S_Steps-1]-16*dmac_spd[S_Steps-2]+5*dmac_spd[S_Steps-3])/12.0

#print(mac_ang)
#print(mempool.total_bytes())
print("Completed!!!")
print("Performance:", time.time()-start)