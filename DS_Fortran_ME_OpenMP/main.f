!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! main.f: Dynamic simulation driver program
! Author: Shuangshuang Jin
! Sample execution: 
!     qsub -I -l select=1:ncpus=16
!     make
!     export OMP_NUM_THREADS=16
!     ./ds 3g9b.txt angle.out power.out
! Last updated: 9-30-2019
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        PROGRAM main

        USE DEFDP
        !USE CLOCK
        USE CONSTANTS
        USE INPUTSIZE
        USE INPUTMATRIX
        USE SIMULATION 
        !USE BsLNSwch
        USE OMP_LIB
        USE BUSINDEX
        USE REDUCEYBUS

        IMPLICIT NONE
      
        INTEGER::i,j,k,jj,ii
        INTEGER::tst1,flagF,flag,S_Steps,I_Steps
        INTEGER::steps1,steps2,steps3

        INTEGER::flagF1,flagF2,flag1,flag2
        REAL(KIND=DP)::h_sol1,h_sol2

        INTEGER,PARAMETER::LineFlag=999

        CHARACTER*200::filename,fileAngle,filePower
        INTEGER::ios,error

        INTEGER,DIMENSION(20)::t_step
        REAL(KIND=DP),DIMENSION(20)::t_width

        COMPLEX(KIND=DP)::curr,phi,v,eprime,ei,rot
        REAL(KIND=DP)::t_all0,t_all1

        CALL getarg(1,filename)
        CALL getarg(2,fileAngle)
        CALL getarg(3,filePower)
        IF (iargc() .NE. 3) THEN
           PRINT *, './ds <inputfile> <outputAngle> <outputPower>'
           CALL EXIT(0)
        ENDIF
        !PRINT *,'Running ',filename

        CALL READINPUTSIZE(filename)
      
        PRINT *,'Bus number is: ',nbus
        PRINT *,'Branch number is: ',nbrch
        PRINT *,'Generator number is: ',ngen
        PRINT *,'Switch number is: ',nswtch
        PRINT *,'Swing bus number is: ',nSW
        PRINT *,'PV bus number is: ',nPV  
        PRINT *,'PQ bus number is: ',nPQ

        CALL allocNetwork 

        OPEN(1,FILE=filename)
        !!! Read in bus(nbus,15)
        flagF=nbus
        i=1
        DO WHILE (flagF > 0)
           READ(1,FMT=101,IOSTAT=ios) bus(i,:)
           i=i+1
           flagF=flagF-1
        END DO
   101  FORMAT(15F12.5)
        READ(1,*)
      
        !!! Read in line(nbrch,10)
        flagF=nbrch
        i=1
        DO WHILE (flagF > 0)
           READ(1,FMT=102,IOSTAT=ios) line(i,1:10)
           i=i+1
           flagF=flagF-1
        END DO
   102  FORMAT(10F12.5)
        READ(1,*)
         
        !!! Read in machine(nSW+nPV,10)
        flagF=ngen
        i=1
        DO WHILE (flagF > 0)
           READ(1,FMT=103,IOSTAT=ios) mac_con(i,1:19)
           i=i+1
           flagF=flagF-1
        END DO
   103  FORMAT(19F12.5)
        READ(1,*)

        !!! Read in switch(nSW+nPV,10)
        flagF=nswtch
        i=1
        DO WHILE (flagF > 0)
           READ(1,FMT=104,IOSTAT=ios) sw_con(i,1:7)
           i=i+1
           flagF=flagF-1
        END DO
   104  FORMAT(7F12.5)
        READ(1,*)
        CLOSE(1)
        PRINT *,'Finished data reading'

        t_all0=OMP_GET_WTIME()
        CALL y_switch
        PRINT *,'Finished y_switch'

        PRINT *,'prefY11:' 
        DO i=1,ngen
           DO j=1,ngen
              PRINT *,'(',i,',',j,'):',prefY11(i,j)
           END DO
        END DO       
 
        PRINT *,'fY11:' 
        DO i=1,ngen
           DO j=1,ngen
              PRINT *,'(',i,',',j,'):',fY11(i,j)
           END DO
        END DO       
 
        PRINT *,'posfY11:' 
        DO i=1,ngen
           DO j=1,ngen
              PRINT *,'(',i,',',j,'):',posfY11(i,j)
           END DO
        END DO       
 
        !!! Start of simulation
        simu_k=0
!$OMP DO PRIVATE(i)
        DO i=1,nswtch-1,1
           t_step(i)=AINT((sw_con(i+1,1)-sw_con(i,1))/sw_con(i,7))
           t_width(i)=(sw_con(i+1,1)-sw_con(i,1))/t_step(i)
           simu_k=simu_k+t_step(i)
        END DO       
!$OMP END DO
        simu_k=simu_k+1
        PRINT *,'simu_k= ',simu_k
        ALLOCATE(simu_t(simu_k),STAT=error)

        simu_t(1)=sw_con(1,1)
        k=2
!$OMP DO PRIVATE(i)
        DO i=1,nswtch-1,1
           DO j=1,t_step(i),1
              simu_t(k)=simu_t(k-1)+t_width(i)
              k=k+1
           END DO
        END DO
!$OMP END DO

        CALL allocSimu

!$OMP DO PRIVATE(i)
        DO i=1,nbus,1
           theta(i)=bus(i,3)*pi/180
           bus_volt(i)=bus(i,2)*exp(jay*theta(i))
        END DO
!$OMP END DO

        flagF=0
!$OMP DO PRIVATE(i)
        DO i=1,ngen,1
           mac_con(i,3)=basmva/mac_con(i,3)
           tst1=bus_int(INT(mac_con(i,2)))
           eterm(i)=bus(tst1,2) !terminal bus voltage
           pelect(i,1)=bus(tst1,4) !BUS_pg
           qelect(i)=bus(tst1,5) !BUS_qg
           curr=Dsqrt(pelect(i,1)*pelect(i,1)+qelect(i)*qelect(i))&
                /eterm(i)*mac_con(i,3) 
           phi=Datan2(qelect(i),pelect(i,1))
           v = eterm(i)*exp(jay*theta(tst1))
           curr=curr*exp(jay*(theta(tst1)-phi))
           eprime=v+jay*mac_con(i,7)*curr
           mac_ang(i,1)=Datan2(AIMAG(eprime),real(eprime))
           mac_spd(i,1)=1.0
           rot=jay*exp(-jay*mac_ang(i,1))
           psi_re(i,1)=real(eprime)
           psi_im(i,1)=AIMAG(eprime)
           eprime=eprime*rot
           edprime(i,1)=real(eprime)
           eqprime(i,1)=AIMAG(eprime)
           curr=curr*rot
           curdg(i)=real(curr)
           curqg(i)=AIMAG(curr)
           curd(i)=real(curr)/mac_con(i,3)
           curq(i)=AIMAG(curr)/mac_con(i,3)
           v=v*rot
           ed(i)=real(v)
           eq(i)=AIMAG(v)
           vex(i)=eqprime(i,1)
           pmech(i,1)=pelect(i,1)*mac_con(i,3)
        END DO
!$OMP END DO

        S_Steps=1
        steps3=t_step(1)+t_step(2)+t_step(3)
        steps2=t_step(1)+t_step(2)
        steps1=t_step(1)

        h_sol1=t_width(1)
        h_sol2=h_sol1
        flagF1=0
        flagF2=0
        S_Steps=1

!$OMP PARALLEL PRIVATE(I_Steps)
        DO I_Steps=1,(simu_k+1),1
           !!! Calculate current injections and bus voltages and angles
           IF (I_Steps<steps1) THEN
              S_Steps=I_Steps
              flagF1=0
              flagF2=0
           ELSEIF (I_Steps==steps1) THEN
              S_Steps=I_Steps
              flagF1=0
              flagF2=1
           ELSEIF (I_Steps==steps1+1) THEN
              S_Steps=I_Steps
              flagF1=1
              flagF2=1
           ELSEIF ((I_Steps>steps1+1).AND.(I_Steps<steps2+1)) THEN
              S_Steps=I_Steps-1
              flagF1=1
              flagF2=1
           ELSEIF (I_Steps==steps2+1) THEN
              S_Steps=I_Steps-1
              flagF1=1
              flagF2=2
           ELSEIF (I_Steps==steps2+2) THEN
              S_Steps=I_Steps-1
              flagF1=2 
              flagF2=2 
           ELSEIF (I_Steps>steps2+2) THEN
              S_Steps=I_Steps-2
              flagF1=2
              flagF2=2
           ENDIF

!$OMP DO PRIVATE(k)
           DO k=1,ngen,1
              pmech(k,S_Steps+1)=pmech(k,S_Steps)
              CALL mac_em1(k,S_Steps)
              !IF (k==1) THEN
              !   PRINT *,I_Steps,S_Steps,&
              !           mac_ang(k,S_Steps),mac_spd(k,S_Steps)
              !ENDIF
           END DO
!$OMP END DO

!$OMP DO PRIVATE(k)
           DO k=1,ngen,1
              CALL i_simu_innerloop(k,S_Steps,flagF1)
              CALL mac_em2(k,S_Steps)
              mac_ang(k,S_Steps+1)=mac_ang(k,S_Steps)+&
                                   h_sol1*dmac_ang(k,S_Steps)
              mac_spd(k,S_Steps+1)=mac_spd(k,S_Steps)+&
                                   h_sol1*dmac_spd(k,S_Steps)
              !IF (k==1) THEN
              !   PRINT *,I_Steps,S_Steps,&
              !           mac_ang(k,S_Steps+1),mac_spd(k,S_Steps+1)
              !ENDIF
              edprime(k,S_Steps+1)=edprime(k,S_Steps)   
              eqprime(k,S_Steps+1)=eqprime(k,S_Steps) 
              CALL mac_em1(k,S_Steps+1)  
           END DO
!$OMP END DO

!$OMP DO PRIVATE(k)
           DO k=1,ngen,1
              CALL i_simu_innerloop(k,S_Steps+1,flagF2)
              CALL mac_em2(k,S_Steps+1)
              mac_ang(k,S_Steps+1)=mac_ang(k,S_Steps)+h_sol2*&
                                   (dmac_ang(k,S_Steps)+&
                                   dmac_ang(k,S_Steps+1))/2.0
              mac_spd(k,S_Steps+1)=mac_spd(k,S_Steps)+h_sol2*&
                                   (dmac_spd(k,S_Steps)+&
                                   dmac_spd(k,S_Steps+1))/2.0
              IF (k==1) THEN
                 PRINT *,I_Steps,S_Steps,&
                         mac_ang(k,S_Steps+1),mac_spd(k,S_Steps+1)
              ENDIF
           END DO
!$OMP END DO     

!$OMP BARRIER
        END DO
!$OMP END PARALLEL

        PRINT *,'End of simulation'

        t_all1=OMP_GET_WTIME()
        PRINT *,'Overall time=',t_all1-t_all0
      
        DO i=1,simu_k,1
           WRITE(*,FMT='(1F15.6,1F15.6)') mac_ang(1,i),mac_spd(1,i)
           !WRITE(*,FMT='(1F15.6,1F15.6)') pmech(1,i),pelect(1,i)
        END DO
        
        OPEN(UNIT=2,FILE=fileAngle)
        OPEN(UNIT=3,FILE=filePower)
        DO i=1,simu_k,1
           WRITE(UNIT=2,FMT='(1F15.6,1F15.6)',IOSTAT=ios) &
                mac_ang(1,i),mac_spd(1,i)
           WRITE(UNIT=3,FMT='(1F15.6,1F15.6)',IOSTAT=ios) &
                pmech(1,i),pelect(1,i)
        END DO
        CLOSE(2)
        CLOSE(3)

        END PROGRAM main
