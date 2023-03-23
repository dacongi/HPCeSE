#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#! run.py: Power System Dynamic Simulation Application
#! Author: Cong Wang 
#! Sample execution: 
#!     qsub -I -l select=1:ncpus=16:mpiprocs=16:mem=62gb,walltime=08:00:00
#!     module purge
#!     conda activate paper5
#!     python run_cpu.py 8100
#! Last updated: 6-29-2021
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import numpy as np
import time
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import sys

jay=1j
basmva=100
sys_freq=60
t0 = time.time()
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

# move data to numpy
ipt_bus=np.array(a,dtype=np.float64)
ipt_brch=np.array(b,dtype=np.float64)
ipt_gen=np.array(c,dtype=np.float64)
ipt_switch=np.array(d,dtype=np.float64)
nPV=np.count_nonzero(ipt_bus[:,9]==2)
t00=time.time()

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

a=np.arange(nbrch)
tap=np.ones((nbrch),dtype=np.complex128)
c_from=lil_matrix((nbus,nbrch),dtype=np.complex128)
c_line=lil_matrix((nbus,nbrch),dtype=np.complex128)
c_to=lil_matrix((nbus,nbrch),dtype=np.complex128)
chrgfull=lil_matrix((nbrch,nbrch),dtype=np.complex128)
yyfull=lil_matrix((nbrch,nbrch),dtype=np.complex128)
xd=np.zeros(ngen)
perm=lil_matrix((ngen,ngen),dtype=np.complex128)
Y_b=lil_matrix((ngen,nbus),dtype=np.complex128)
Y_a=lil_matrix((ngen,ngen),dtype=np.complex128)

# prefY11=reduce_y(flag=0)
z=r+jay*rx
yy=1/z

from_int=bus_int[from_bus-1].astype(int)
to_int=bus_int[to_bus-1].astype(int)
tap[liney_ratio>0]=np.exp((-jay*phase_shift[liney_ratio>0])*np.pi/180)/liney_ratio[liney_ratio>0]
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

perm.setdiag(1)
Y_a.setdiag(y)

if ngen != nPV:
    e=np.where(~np.eye(perm.get_shape()[0],dtype=bool))
    h=np.where(g_bus[e[0]]==g_bus[e[1]])
    perm[e[0][h],e[1][h]]=1
    permPV=perm
else:
    permPV=perm

Ymod=Y_a.dot(permPV.transpose())
permmod=permPV.transpose().dot(Ymod)

Y_b[:,g_bus-1]=-Ymod.transpose()[:]

Y.setdiag(Y.diagonal()+yl+Gb+jay*Bb)
Y[g_bus[:,None]-1,g_bus-1]=Y[g_bus[:,None]-1,g_bus-1]+permmod[:]

Y=Y.tocsc()
Y_c=(Y_b.transpose()).tocsc()

prefrecV1=-spsolve(Y,Y_c,permc_spec='NATURAL')
#prefrecV1=solve(Y,Y_c)
temp=Y_b.dot(prefrecV1)
prefY11=Y_a+temp.transpose()
#print(prefrecV1.toarray())
t00000=time.time()
#print(t00000-t00)

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

Y_1=Y_1.tocsc()
frecV1=-spsolve(Y_1,Y_c)
temp=Y_b.dot(frecV1)
fY11=Y_a+temp.transpose()
#print(type(fY11))
f_farbus=f_type[2].astype(int)
Bb[bus_idx-1]=0.0
i=np.where(np.logical_and(from_bus==f_nearbus, to_bus==f_farbus))
rx[i]=10000000.0
j=np.where(np.logical_and(from_bus==f_farbus, to_bus==f_nearbus))
rx[j]=10000000.0

# posfY11=reduce_y(flag=2)
z=r+jay*rx
yy=1/z

yyfull.setdiag(yy)
Y_dummy=yyfull.dot(c_line.transpose())
Y_2=(c_line.dot(Y_dummy)+Y_2)

Y_2.setdiag(Y_2.diagonal()+yl+Gb+jay*Bb)
Y_2[g_bus[:,None]-1,g_bus-1]=Y_2[g_bus[:,None]-1,g_bus-1]+permmod[:]

Y_2=Y_2.tocsc()
posfrecV1=-spsolve(Y_2,Y_c)
temp=Y_b.dot(posfrecV1)
posfY11=Y_a+temp.transpose()

t000=time.time()

# Start of simulation
basrad=2*np.pi*sys_freq
t_step=np.around((s1[1:]-s1[:-1])/s7[:-1])
t_width=(s1[1:]-s1[:-1])/t_step[:]
sim_k=int(t_step.sum())
sim_k=sim_k+1

mac_ang,mac_spd,dmac_ang,dmac_spd=np.zeros((4,sim_k,ngen),dtype=np.float64)
eprime=np.zeros((ngen),dtype=np.complex128)
pelect=np.zeros((ngen),dtype=np.float64)

theta=np.radians(b_ang)
bus_volt=V*np.exp(jay*theta)
mva=basmva/g_m
tst1=bus_int[g_bus-1].astype(int)
eterm=V[tst1-1]            # terminal bus voltage
pelect=b_pg[tst1-1]        # BUS_pg
qelect=b_qg[tst1-1]        # BUS_qg

# compute the initial values for generator dynamics
curr=np.hypot(pelect,qelect)/eterm*mva
phi=np.arctan2(qelect,pelect)
v=eterm*np.exp(jay*theta[tst1-1])
curr=curr*np.exp(jay*(theta[tst1-1]-phi))
eprime=v+jay*g_dtr*curr

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

    # compute current current based on reduced Y-Bus prefy/fy/posfy and eprime
    if flagF1 == 0:
        cur = prefY11.dot(eprime)
    if flagF1 == 1:
        cur = fY11.dot(eprime)
    if flagF1 == 2:
        cur = posfY11.dot(eprime)

    # compute generators electric real power output pElect;
    curd = np.sin(mac_ang[S_Steps-1])*cur.real-np.cos(mac_ang[S_Steps-1])*cur.imag
    curq = np.cos(mac_ang[S_Steps-1])*cur.real+np.sin(mac_ang[S_Steps-1])*cur.imag
    
    curdg = curd*mva
    curqg = curq*mva
    ed = edprime+g_dtr*curqg
    eq = eqprime-g_dtr*curdg
    
    eterm = np.hypot(ed,eq)
    pelect = eq*curq+ed*curd
    qelect = eq*curd-ed*curq
    
    # f(y)
    dmac_ang[S_Steps-1] = basrad*(mac_spd[S_Steps-1]-1.0)
    dmac_spd[S_Steps-1] = (pmech-pelect*mva-g_do*(mac_spd[S_Steps-1]-1.0))/(2.0*g_H)

    # 3 steps Adam-Bashforth integration steps
    if S_Steps-1 < 2:
        k1_mac_ang = h_sol1*dmac_ang[S_Steps-1]
        k1_mac_spd = h_sol1*dmac_spd[S_Steps-1]

        k2_mac_ang = h_sol1*(basrad*(mac_spd[S_Steps-1]+k1_mac_ang/2.0-1.0))
        k2_mac_spd = h_sol1*(pmech-pelect*mva-g_do*(mac_spd[S_Steps-1]+k1_mac_spd/2.0-1.0))/(2.0*g_H)

        k3_mac_ang = h_sol1*(basrad*(mac_spd[S_Steps-1]+k2_mac_ang/2.0-1.0))
        k3_mac_spd = h_sol1*(pmech-pelect*mva-g_do*(mac_spd[S_Steps-1]+k2_mac_spd/2.0-1.0))/(2.0*g_H)

        k4_mac_ang = h_sol1*(basrad*(mac_spd[S_Steps-1]+k3_mac_ang-1.0))
        k4_mac_spd = h_sol1*(pmech-pelect*mva-g_do*(mac_spd[S_Steps-1]+k3_mac_spd-1.0))/(2.0*g_H)

        mac_ang[S_Steps] = mac_ang[S_Steps-1]+(k1_mac_ang+2*k2_mac_ang+2*k3_mac_ang+k4_mac_ang)/6.0
        mac_spd[S_Steps] = mac_spd[S_Steps-1]+(k1_mac_spd+2*k2_mac_spd+2*k3_mac_spd+k4_mac_spd)/6.0
    else:
        mac_ang[S_Steps] = mac_ang[S_Steps-1]+h_sol1*(23*dmac_ang[S_Steps-1]-16*dmac_ang[S_Steps-2]+5*dmac_ang[S_Steps-3])/12.0
        mac_spd[S_Steps] = mac_spd[S_Steps-1]+h_sol1*(23*dmac_spd[S_Steps-1]-16*dmac_spd[S_Steps-2]+5*dmac_spd[S_Steps-3])/12.0
            
t2 = time.time()
print("Data reading in", t00-t0)
#print("Ys in", t000-t00)
#print("Before kernel:", t1-t0)
print("Kernel:", t2-t1)
print("Total:", t2-t0)
#print(t2-t1)