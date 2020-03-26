!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! y_switch.f: Define pre-fault, on-fault, pos-fault stages and the 
!             corresponding network topology changes 
! Author: Shuangshuang Jin
! Last updated: 9-30-2019
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        SUBROUTINE y_switch
        
        USE DEFDP
        USE CONSTANTS
        USE INPUTSIZE
        USE INPUTMATRIX
        USE BsLNSwch
        USE BUSINDEX

        IMPLICIT NONE
        
        INTEGER::flag,bus_idx
        REAL(KIND=DP)::f_type,f_nearbus,bf,f_farbus
        INTEGER::i

        !!! Pre fault
        flag=0
        CALL red_ybus(flag)

        fbus=bus
        fline=line
        f_type=sw_con(2,6)
        IF (f_type<4) THEN
           f_nearbus=sw_con(2,2)
           bus_idx=bus_int(INT(f_nearbus))
           IF (f_type==0) THEN
              !!! Three phase fault zero imedance to ground
              bf=1.0/1e-7
           END IF
           fbus(bus_idx,9)=bf
        END IF
        !DO i=1,10,1
        !   PRINT *,'fbus(:,',i,')=',fbus(:,i)
        !END DO
        !DO i=1,nbus,1
        !   PRINT *,'fbus(',i,',:)=',fbus(i,:)
        !END DO

        !!! Fault on
        flag=1
        CALL red_ybus(flag)

        IF (f_type<4) THEN
           f_farbus=sw_con(2,3)
           posfbus=bus
           posfline=line
           DO i=1,nbrch,1
              IF ((posfline(i,1)==f_nearbus) .AND. &
                 (posfline(i,2)==f_farbus)) THEN
                 posfline(i,4)=1.0e7
                 EXIT
              END IF
              IF ((posfline(i,1)==f_farbus) .AND. &
                 (posfline(i,2)==f_nearbus)) THEN
                 posfline(i,4)=1.0e7
                 EXIT
              END IF
           END DO
        END IF
        !PRINT *,'ftype= ',f_type
        !PRINT *,'fnearbus= ',f_nearbus
        !PRINT *,'ffarbus= ',f_farbus
        !DO i=1,10,1
        !   PRINT *,'posfline(:,',i,')=',posfline(:,i)
        !END DO
        !DO i=1,nbrch,1
        !   PRINT *,'posfline(',i,',:)=',posfline(i,:)
        !END DO

        !!! Post Fault        
        flag=2
        CALL red_ybus(flag)
        
        RETURN
        END SUBROUTINE y_switch
