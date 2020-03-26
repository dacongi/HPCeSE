!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! inputmodule.f: Define modules and global variables 
! Author: Shuangshuang Jin
! Last updated: 9-30-2019
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        MODULE DEFDP
        INTEGER,PARAMETER::SP=SELECTED_REAL_KIND(6)
        INTEGER,PARAMETER::DP=SELECTED_REAL_KIND(14)
        END MODULE DEFDP

        MODULE CONSTANTS
        USE DEFDP
        INTEGER,PARAMETER::swing_bus=1
        INTEGER,PARAMETER::generator_bus=2
        INTEGER,PARAMETER::load_bus=3
        REAL(KIND=DP),PARAMETER::pi=3.1415926535897932384626
        REAL(KIND=DP),PARAMETER::eps=2.22E-10
        COMPLEX(KIND=DP),PARAMETER::jay=(0,1)
        REAL(KIND=DP),PARAMETER::basmva=100,sys_freq=60
        REAL(KIND=DP),PARAMETER::basrad=2*pi*sys_freq
        INTEGER,PARAMETER::buscol=15
        INTEGER,PARAMETER::linecol=10
        INTEGER,PARAMETER::gencol=19
        END MODULE CONSTANTS

        MODULE INPUTSIZE
        INTEGER::nbus,nbrch,ngen,nSW,nPV,nPQ,nswtch
        END MODULE INPUTSIZE

        MODULE INPUTMATRIX
        USE DEFDP
        REAL(KIND=DP),ALLOCATABLE,DIMENSION(:,:)::bus
        REAL(KIND=DP),ALLOCATABLE,DIMENSION(:,:)::line
        REAL(KIND=DP),ALLOCATABLE,DIMENSION(:,:)::mac_con
        REAL(KIND=DP),ALLOCATABLE,DIMENSION(:,:)::sw_con
        END MODULE INPUTMATRIX

        MODULE BsLNSwch 
        USE DEFDP
        REAL(KIND=DP),ALLOCATABLE,DIMENSION(:,:)::fbus
        REAL(KIND=DP),ALLOCATABLE,DIMENSION(:,:)::fline
        REAL(KIND=DP),ALLOCATABLE,DIMENSION(:,:)::posfbus
        REAL(KIND=DP),ALLOCATABLE,DIMENSION(:,:)::posfline
        END MODULE BsLNSwch

        MODULE BUSINDEX
        INTEGER,ALLOCATABLE,DIMENSION(:)::bus_int
        END MODULE BUSINDEX

        MODULE REDUCEYBUS 
        USE DEFDP
        COMPLEX(KIND=DP),ALLOCATABLE,DIMENSION(:,:)::prefY,fY,posfY
        COMPLEX(KIND=DP),ALLOCATABLE,DIMENSION(:,:)::prefY11,fY11,&
                                                     posfY11
        END MODULE REDUCEYBUS

        MODULE SIMULATION
        USE DEFDP
        !!! Length of simu_t
        INTEGER::simu_k 
        !!! Simulation time sequences
        REAL(KIND=DP),ALLOCATABLE,DIMENSION(:)::simu_t

        REAL(KIND=DP),ALLOCATABLE,DIMENSION(:,:)::pelect,pmech
        REAL(KIND=DP),ALLOCATABLE,DIMENSION(:,:)::psi_re,psi_im
        REAL(KIND=DP),ALLOCATABLE,DIMENSION(:,:)::mac_ang,mac_spd
        REAL(KIND=DP),ALLOCATABLE,DIMENSION(:,:)::dmac_ang,dmac_spd
        REAL(KIND=DP),ALLOCATABLE,DIMENSION(:,:)::edprime,eqprime
         
        REAL(KIND=DP),ALLOCATABLE,DIMENSION(:)::eterm,qelect
        REAL(KIND=DP),ALLOCATABLE,DIMENSION(:)::curdg,curqg
        REAL(KIND=DP),ALLOCATABLE,DIMENSION(:)::curd,curq
        REAL(KIND=DP),ALLOCATABLE,DIMENSION(:,:)::cur_re,cur_im
        REAL(KIND=DP),ALLOCATABLE,DIMENSION(:)::ed,eq,vex
        REAL(KIND=DP),ALLOCATABLE,DIMENSION(:)::theta
        COMPLEX(KIND=DP),ALLOCATABLE,DIMENSION(:)::bus_volt
        END MODULE SIMULATION

