!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! y_sparse.f: Build admittance matrix Y (not sparse) from the line data 
! Author: Shuangshuang Jin
! Last updated: 9-30-2019
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        SUBROUTINE y_sparse(flag)

        USE DEFDP
        USE CONSTANTS
        USE INPUTSIZE
        USE INPUTMATRIX
        USE BUSINDEX
        USE BsLNSwch
        USE OMP_LIB
        USE REDUCEYBUS 

        IMPLICIT NONE

        INTEGER flag
        REAL(KIND=DP),DIMENSION(:,:),pointer::busy
        REAL(KIND=DP),DIMENSION(:,:),pointer::liney

        INTEGER,DIMENSION(:),pointer::from_bus,from_int,to_bus,&
                                      to_int,tap_index
        COMPLEX(KIND=DP),DIMENSION(:,:),pointer::Y
        COMPLEX(KIND=DP),DIMENSION(:,:),pointer::Y_dummy

        REAL(KIND=DP),DIMENSION(:),pointer::r,rx,phase_shift
        COMPLEX(KIND=DP),DIMENSION(:),pointer::z,tap
        COMPLEX(KIND=DP),DIMENSION(:,:),pointer::chrg,yy

        COMPLEX(KIND=DP),DIMENSION(:,:),pointer::chrgfull,yyfull
        COMPLEX(KIND=DP),DIMENSION(:,:),pointer::c_from,c_to,c_line

        COMPLEX(KIND=DP)::alpha,beta
        REAL(KIND=DP),DIMENSION(:),pointer::Gb,Bb

        INTEGER::i,j,k
        INTEGER::busmax,ibus,bus_int_size

        ALLOCATE(busy(nbus,buscol))     
        ALLOCATE(liney(nbrch,linecol))
        ALLOCATE(from_bus(nbrch),from_int(nbrch),to_bus(nbrch),&
                 to_int(nbrch),tap_index(nbrch))
        ALLOCATE(Y(nbus,nbus))
        ALLOCATE(Y_dummy(nbrch,nbus))
        ALLOCATE(r(nbrch),rx(nbrch),phase_shift(nbrch))
        ALLOCATE(z(nbrch),tap(nbrch))
        ALLOCATE(chrg(nbrch,1),yy(nbrch,1))
        ALLOCATE(Gb(nbus),Bb(nbus))

        ALLOCATE(chrgfull(nbrch,nbrch))
        ALLOCATE(yyfull(nbrch,nbrch))
        ALLOCATE(c_from(nbus,nbrch))
        ALLOCATE(c_to(nbus,nbrch))
        ALLOCATE(c_line(nbus,nbrch))

        !!! Pre fault 
        IF (flag==0) THEN
           busy=bus
           liney=line
        END IF

        !!! On fault
        IF (flag==1) THEN 
           busy=fbus
           liney=fline
        END IF

        !!! Post fault
        IF (flag==2) THEN
           busy=posfbus
           liney=posfline
        END IF

!$OMP PARALLEL DO PRIVATE(i)
        DO i = 1,nbrch,1
           r(i)=0
           rx(i)=0
           chrg(i,1)=(0,0)
           z(i)=(0,0)
           yy(i,1)=(0,0)
           from_bus(i)=0
           to_bus(i)=0
           from_int(i)=0
           to_int(i)=0
           tap_index(i)=0
           tap(i)=(1,0)
           phase_shift(i)=0
           DO j=1,nbrch,1
              chrgfull(i,j)=(0,0)
              yyfull(i,j)=(0,0)
           END DO
        END DO
!$OMP END PARALLEL DO

!$OMP PARALLEL DO PRIVATE(i)
        DO i=1,nbus,1
           Gb(i)=busy(i,8)
           Bb(i)=busy(i,9)
           !!! Bus conductance
           !!! Bus susceptance
           DO j=1,nbrch,1
              c_from(i,j)=(0,0)
              c_to(i,j)=(0,0)
              c_line(i,j)=(0,0)
           END DO
        END DO
!$OMP END PARALLEL DO

        !!! Set up internal bus numbers for second indexing of buses
        !busmax = MAXVAL(busy(:,1))
        
        bus_int_size=SIZE(bus_int)
!$OMP PARALLEL DO PRIVATE(i)
        DO i=1,bus_int_size,1
           bus_int(i)=0
        END DO
!$OMP END PARALLEL DO

!$OMP PARALLEL DO PRIVATE(i)
        DO i=1,nbus,1
           bus_int(INT(busy(i,1)))=i
        END DO
!$OMP END PARALLEL DO

        !DO i=1,bus_int_size,1
        !   PRINT *,'i,bus_int=',i,bus_int(i)
        !END DO
 
        !!! Process line data and build admittance matrix Y
!$OMP PARALLEL DO PRIVATE(i)
        DO i=1,nbrch,1
           r(i)=liney(i,3)
           rx(i)=liney(i,4)
           chrg(i,1)=jay*(0.5*liney(i,5))
           z(i)=r(i)+jay*rx(i)
           yy(i,1)=1/z(i)
           from_bus(i)=INT(liney(i,1))
           from_int(i)=bus_int(from_bus(i))
           to_bus(i)=INT(liney(i,2));
           to_int(i)=bus_int(to_bus(i))
           phase_shift(i)=liney(i,7)
           IF (liney(i,6)>0) THEN
              tap_index(i)=i
              tap(i)=1/liney(i,6)
              tap(i)=tap(i)*EXP(-jay*phase_shift(i)*pi/180)
           END IF
        END DO
!$OMP END PARALLEL DO

        !!! Line impedance
        !!! Determine connection matrices including tap chargers and
        !!! phase shifters
        !!! sparse matrix formulation
!$OMP PARALLEL DO PRIVATE(i)
        DO i=1,nbrch,1
           c_from(from_int(i),i)=tap(i)
           c_to(to_int(i),i)=(1,0)
           c_line(from_int(i),i)=c_from(from_int(i),i)-&
                                 c_to(from_int(i),i)
           c_line(to_int(i),i)=c_from(to_int(i),i)-c_to(to_int(i),i)
        END DO
!$OMP END PARALLEL DO

        !!! Form Y matrix from primative line ys and connection matrices
!$OMP PARALLEL DO PRIVATE(i)
        DO i=1,nbrch,1
           chrgfull(i,i)=chrg(i,1)
           yyfull(i,i)=yy(i,1)
        END DO
!$OMP END PARALLEL DO

        !PRINT *,'chrgfull size are: ',size(chrgfull,1),size(chrgfull,2)
        !PRINT *,'yyfull size are: ',size(yyfull,1),size(yyfull,2)

        alpha=DCMPLX(1.0,0.0)
        beta=DCMPLX(0.0,0.0)
        CALL ZGEMM('n','t',nbrch,nbus,nbrch,alpha,chrgfull,nbrch,&
                    c_from,nbus,beta,Y_dummy,nbrch)
        CALL ZGEMM('n','n',nbus,nbus,nbrch,alpha,c_from,nbus,Y_dummy,&
                   nbrch,beta,Y,nbus)
        CALL ZGEMM('n','t',nbrch,nbus,nbrch,alpha,chrgfull,nbrch,&
                   c_to,nbus,beta,Y_dummy,nbrch)
        CALL ZGEMM('n','n',nbus,nbus,nbrch,alpha,c_to,nbus,Y_dummy,&
                   nbrch,alpha,Y,nbus)
       
        !!! Y=MATMUL(MATMUL(C_from,chrgfull),TRANSPOSE(C_from))+&
        !!!   MATMUL(MATMUL(C_to,chrgfull),TRANSPOSE(C_to)) 
        CALL ZGEMM('n','t',nbrch,nbus,nbrch,alpha,yyfull,nbrch,c_line,&
                   nbus,beta,Y_dummy,nbrch)
        CALL ZGEMM('n','n',nbus,nbus,nbrch,alpha,c_line,nbus,Y_dummy,&
                   nbrch,alpha,Y,nbus)

        !!! Y=Y+MATMUL(MATMUL(C_line,yyfull),TRANSPOSE(C_line))
!$OMP PARALLEL DO PRIVATE(i)
        DO i=1,nbus,1
           Y(i,i)=Y(i,i)+Gb(i)+jay*Bb(i) 
        END DO
!$OMP END PARALLEL DO

        !DO i=1,nbus,1
        !   DO j=1,nbus,1
        !      print *,'Y(',i,',',j,')= ',Y(i,j)
        !   END DO
        !END DO

        IF (flag==0) THEN
           prefY=Y
        END IF

        IF (flag==1) THEN
           fY=Y
        END IF

        IF (flag==2) THEN
           posfY=Y
        END IF

        DEALLOCATE(chrgfull)
        DEALLOCATE(yyfull)
        DEALLOCATE(c_from)
        DEALLOCATE(c_to)
        DEALLOCATE(c_line)

        DEALLOCATE(busy)
        DEALLOCATE(liney)
        DEALLOCATE(from_bus,from_int,to_bus,to_int,tap_index)
        DEALLOCATE(Y)
        DEALLOCATE(Y_dummy)
        DEALLOCATE(r,rx,phase_shift)
        DEALLOCATE(z,tap)
        DEALLOCATE(chrg,yy)
        DEALLOCATE(Gb,Bb)

        RETURN
        END SUBROUTINE y_sparse
