!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! red_ybus.f: Construct reduced Ybus at different fault stages 
! Author: Shuangshuang Jin
! Last updated: 9-30-2019
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        SUBROUTINE red_ybus(flag)

        USE DEFDP
        USE CONSTANTS
        USE INPUTSIZE
        USE INPUTMATRIX
        USE BsLNSwch
        USE OMP_LIB
        USE ReduceYbus

        IMPLICIT NONE
        
        INTEGER flag
        CHARACTER*200::filename

        REAL(KIND=DP),DIMENSION(:,:),pointer::busy
        REAL(KIND=DP),DIMENSION(:),pointer::Pl,Ql,V
        REAL(KIND=DP),DIMENSION(:,:),pointer::liney
        REAL(KIND=DP),DIMENSION(:,:),pointer::P
        REAL(KIND=DP),DIMENSION(:),pointer::ra,xd
        REAL(KIND=DP),DIMENSION(:,:),pointer::dummyeye

        INTEGER,DIMENSION(:),pointer::IPIV2

        COMPLEX(KIND=DP),DIMENSION(:,:),pointer::permmod
        COMPLEX(KIND=DP),DIMENSION(:,:),pointer::permPV
        COMPLEX(KIND=DP),DIMENSION(:,:),pointer::Ymod

        COMPLEX(KIND=DP),DIMENSION(:,:),pointer::perm,diagy
        COMPLEX(KIND=DP),DIMENSION(:),pointer::y

        COMPLEX(KIND=DP),DIMENSION(:,:),pointer::Y_c
        COMPLEX(KIND=DP),DIMENSION(:,:),pointer::Y_d
        COMPLEX(KIND=DP),DIMENSION(:,:),pointer::Y_b
        COMPLEX(KIND=DP),DIMENSION(:,:),pointer::Y_a
        COMPLEX(KIND=DP),DIMENSION(:),pointer::yl

        COMPLEX(KIND=DP),DIMENSION(:,:),pointer::temp
        COMPLEX(KIND=DP),ALLOCATABLE,DIMENSION(:,:)::prefrecV1
        COMPLEX(KIND=DP),ALLOCATABLE,DIMENSION(:,:)::frecV1
        COMPLEX(KIND=DP),ALLOCATABLE,DIMENSION(:,:)::posfrecV1

        INTEGER::i,j,k,m,tst,INFO
        
        ALLOCATE(busy(nbus,buscol))
        ALLOCATE(Pl(nbus),Ql(nbus),V(nbus))
        ALLOCATE(liney(nbrch,linecol))
        ALLOCATE(P(nbus,nbus))
        ALLOCATE(ra(ngen),xd(ngen))
        ALLOCATE(dummyeye(nbus,nbus))
        ALLOCATE(IPIV2(nbus))
        ALLOCATE(permmod(ngen,ngen))
        ALLOCATE(permPV(ngen,ngen))
        ALLOCATE(Ymod(ngen,ngen))
        ALLOCATE(perm(ngen,ngen),diagy(ngen,ngen))
        ALLOCATE(y(ngen))
        ALLOCATE(Y_c(nbus,ngen))
        ALLOCATE(Y_d(nbus,nbus))
        ALLOCATE(Y_b(ngen,nbus))
        ALLOCATE(Y_a(ngen,ngen))
        ALLOCATE(yl(nbus))
        ALLOCATE(temp(ngen,ngen))
        ALLOCATE(prefrecV1(nbus,ngen))
        ALLOCATE(frecV1(nbus,ngen))
        ALLOCATE(posfrecV1(nbus,ngen))

        CALL y_sparse(flag)
        PRINT *,'Finished y_sparse'

        IF (flag==0) THEN
           busy=bus
           liney=line
           Y_d=prefY
           V=bus(:,2)
        END IF

        IF (flag==1) THEN
           busy=fbus
           liney=fline
           Y_d=fY
           V=fbus(:,2)
        END IF

        IF (flag==2) THEN
           busy=posfbus
           liney=posfline
           Y_d=posfY
           V=posfbus(:,2)
        END IF

        Pl=busy(:,6)
        Ql=busy(:,7)

!$OMP PARALLEL DO PRIVATE(i)
        DO i=1,nbus,1
           IF (INT(busy(i,10)) .EQ. 3) THEN
              Pl(i)=Pl(i)-busy(i,4) 
              Ql(i)=Ql(i)-busy(i,5)
           END IF
           yl(i)=(Pl(i)-jay*Ql(i))/(V(i)*V(i))
           Y_d(i,i)=Y_d(i,i)+yl(i)
        END DO
!$OMP END PARALLEL DO

        !PRINT *,'Y_d:'
        !DO i=1,nbus
        !   DO j=1,nbus
        !      IF (Y_d(i,j) .NE. 0) THEN
        !         print *,'(',i,',',j,'):',Y_d(i,j)
        !      END IF
        !   END DO
        !END DO  
 
!$OMP PARALLEL DO PRIVATE(i)
        DO i=1,ngen,1
           DO j=1,nbus,1
              Y_b(i,j)=0
           END DO
        END DO
!$OMP END PARALLEL DO

!$OMP PARALLEL DO PRIVATE(i)
        DO i=1,ngen,1
           ra(i)=mac_con(i,5)*basmva/mac_con(i,3)
           IF (mac_con(i,8)==0) THEN
              xd(i)=mac_con(i,7)*basmva/mac_con(i,3)
           END IF
           y(i)=1/(ra(i)+jay*xd(i))
           !print *,y(i)
        END DO 
!$OMP END PARALLEL DO

!$OMP PARALLEL DO PRIVATE(i)
        DO i=1,ngen,1
           DO j=1,ngen,1
              IF (i .NE. j) THEN
                 perm(i,j)=0
                 Y_a(i,j)=0
              ELSE
                 perm(i,j)=1
                 Y_a(i,j)=y(i)
              END IF
           END DO
        END DO
!$OMP END PARALLEL DO

        !!! Consider one bus with multi machine
        IF (ngen .NE. nPV) THEN
           k=1
           DO i=1,ngen,1
              DO j=1,ngen,1
                 IF ((i .NE. j).AND. (mac_con(i,2)==mac_con(j,2))) THEN
                    perm(i,j)=1
                 END IF
              END DO
              IF (perm(i,i) .NE. 0) THEN
                 permPV(k,:)=perm(i,:)
                 k=k+1
              END IF
           END DO
        ELSE
           permPV=perm
        END IF

!$OMP PARALLEL DO PRIVATE(i)
        DO i=1,ngen,1
           DO j=1,ngen,1
              IF (i==j) THEN
                 diagy(i,j)=y(i)
              ELSE
                 diagy(i,j)=0
              END IF
           END DO
        END DO
!$OMP END PARALLEL DO

        !!!Ymod=MATMUL(diagy,TRANSPOSE(permPV))
!$OMP PARALLEL DO PRIVATE(i)
        DO i=1,ngen-1
           DO j=2,ngen
              permPV(i,j)=permPV(j,i)
           END DO
        END DO
!$OMP END PARALLEL DO

!$OMP PARALLEL DO PRIVATE(i)
        DO i=1,ngen
           DO j=1,ngen
              Ymod(i,j)=0
              DO k=1,ngen
                 Ymod(i,j)=Ymod(i,j)+diagy(i,k)*permPV(k,j)
              END DO
           END DO
        END DO
!$OMP END PARALLEL DO

        !!!CALL MATRIXPRODUCT(diagy,ngen,ngen,permPV,ngen,ngen,&
        !!!                   Ymod,ngen,ngen)
        !!!Ymod=MATMUL(diagy,permPV)
!$OMP PARALLEL DO PRIVATE(j)
        DO j=1,ngen,1
           Y_b(:,INT(mac_con(j,2)))=-Ymod(:,j)
        END DO
!$OMP END PARALLEL DO

        !!!permmod=permPV*Ymod
!$OMP PARALLEL DO PRIVATE(i)
        DO i=1,ngen
           DO j=1,ngen
              permmod(i,j)=0
              DO k=1,ngen
                 permmod(i,j)=permmod(i,j)+permPV(i,k)*Ymod(k,j)
              END DO
           END DO
        END DO
!$OMP END PARALLEL DO

        j=1
!$OMP PARALLEL DO PRIVATE(i)
        DO i=1,ngen,1
           DO k=1,ngen,1
              Y_d(INT(mac_con(i,2)),INT(mac_con(k,2)))=&
              Y_d(INT(mac_con(i,2)),INT(mac_con(k,2)))+permmod(i,k)
           END DO
        END DO
!$OMP END PARALLEL DO

        !!!Y_c=TRANSPOSE(Y_b)
!$OMP PARALLEL DO PRIVATE(i)
        DO i=1,ngen
           DO j=1,nbus
              Y_c(j,i)=Y_b(i,j)
           END DO
        END DO
!$OMP END PARALLEL DO
        
!$OMP PARALLEL DO PRIVATE(i)
        DO i=1,nbus
           DO j=1,nbus
              IF (i==j) THEN
                 dummyeye(i,j)=1
              ELSE
                 dummyeye(i,j)=0
              END IF
           END DO
        END DO
!$OMP END PARALLEL DO
        
        !DO i=1,nbus
        !   DO j=1,nbus
        !      print *,Y_d(i,j)
        !   END DO
        !END DO  
 
        IF (flag==0) THEN
           !CALL ZGESV(nbus,ngen,Y_d,nbus,IPIV2,dummyeye,nbus,INFO)
           CALL ZGESV(nbus,ngen,Y_d,nbus,IPIV2,Y_c,nbus,INFO)
           prefrecV1=-Y_c
           !!!prefy11=Y_a+MATMUL(Y_b,prefrecV1)
!$OMP PARALLEL DO PRIVATE(i)
           DO i=1,ngen
              DO j=1,ngen
                 temp(i,j)=0
                 DO k=1,nbus
                    temp(i,j)=temp(i,j)+Y_b(i,k)*prefrecV1(k,j)
                 END DO
              END DO
           END DO
!$OMP END PARALLEL DO
           prefY11=Y_a+temp
        END IF

        IF (flag==1) THEN
           !CALL ZGESV(nbus,ngen,Y_d,nbus,IPIV2,dummyeye,nbus,INFO)
           CALL ZGESV(nbus,ngen,Y_d,nbus,IPIV2,Y_c,nbus,INFO)
           frecV1=-Y_c
           !!!fY11=Y_a+MATMUL(Y_b,frecV1)
!$OMP PARALLEL DO PRIVATE(i)
           DO i=1,ngen
              DO j=1,ngen
                 temp(i,j)=0
                 DO k=1,nbus
                    temp(i,j)=temp(i,j)+Y_b(i,k)*frecV1(k,j)
                 END DO
              END DO
           END DO
!$OMP END PARALLEL DO
           fY11=Y_a+temp
        END IF
       
        IF (flag==2) THEN
           !CALL ZGESV(nbus,ngen,Y_d,nbus,IPIV2,dummyeye,nbus,INFO)
           CALL ZGESV(nbus,ngen,Y_d,nbus,IPIV2,Y_c,nbus,INFO)
           posfrecV1=-Y_c
           !!!posfY11=Y_a+MATMUL(Y_b,posfrecV1)
!$OMP PARALLEL DO PRIVATE(i)
           DO i=1,ngen
              DO j=1,ngen
                 temp(i,j)=0
                 DO k=1,nbus
                    temp(i,j)=temp(i,j)+Y_b(i,k)*posfrecV1(k,j)
                 END DO
              END DO
           END DO
!$OMP END PARALLEL DO
           posfY11=Y_a+temp
        END IF
        
        DEALLOCATE(busy)
        DEALLOCATE(Pl,Ql,V)
        DEALLOCATE(liney)
        DEALLOCATE(P)
        DEALLOCATE(ra,xd)
        DEALLOCATE(dummyeye)
        DEALLOCATE(IPIV2)
        DEALLOCATE(permmod)
        DEALLOCATE(permPV)
        DEALLOCATE(Ymod)
        DEALLOCATE(perm,diagy)
        DEALLOCATE(y)
        DEALLOCATE(Y_c)
        DEALLOCATE(Y_d)
        DEALLOCATE(Y_b)
        DEALLOCATE(Y_a)
        DEALLOCATE(yl)
        
        PRINT *,'End of red_ybus'
       
        RETURN 
        END SUBROUTINE red_ybus
