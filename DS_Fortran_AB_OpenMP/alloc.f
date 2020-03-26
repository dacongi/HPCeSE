!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! alloc.f: Allocate memory for global dynamic arrays 
! Author: Shuangshuang Jin
! Last updated: 9-30-2019
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        SUBROUTINE allocNetwork 
        
        USE INPUTSIZE
        USE CONSTANTS
        USE REDUCEYBUS
        USE INPUTMATRIX
        USE BsLNSwch
        USE BUSINDEX
        USE SIMULATION 

        IMPLICIT NONE
        INTEGER error

        !!! Allocate network topology related variables
        ALLOCATE(bus(nbus,buscol),line(nbrch,linecol),&
                 mac_con(ngen,gencol),sw_con(nswtch,7),STAT=error)
        IF (error /= 0) THEN
           PRINT *,'Cannot allocate arrays'
        END IF

        ALLOCATE(fbus(nbus,buscol),fline(nbrch,linecol),&
                 posfbus(nbus,buscol),posfline(nbrch,linecol),STAT=error)
        IF (error /= 0) THEN
           PRINT *,'Cannot allocate arrays'
        END IF

        ALLOCATE(bus_int(nbus),STAT=error)
        IF (error /= 0) THEN
           PRINT *,'Cannot allocate arrays'
        END IF

        ALLOCATE(prefY(nbus,nbus),fY(nbus,nbus),posfY(nbus,nbus),&
                 STAT=error)
        IF (error /= 0) THEN
           PRINT *,'Cannot allocate arrays'
        END IF

        ALLOCATE(prefY11(ngen,ngen),fY11(ngen,ngen),posfY11(ngen,ngen),&
                 STAT=error)
        IF (error /= 0) THEN
           PRINT *,'Cannot allocate arrays'
        END IF

        RETURN
        END SUBROUTINE allocNetwork

        SUBROUTINE allocSimu
        
        USE INPUTSIZE,ONLY:ngen,nbus
        USE SIMULATION 

        IMPLICIT NONE
        INTEGER error

        !!! Allocate simulation related variables
        ALLOCATE(mac_ang(ngen,simu_k),mac_spd(ngen,simu_k),&
                 dmac_ang(ngen,simu_k),dmac_spd(ngen,simu_k),&
                 edprime(ngen,simu_k),eqprime(ngen,simu_k),&
                 cur_re(ngen,simu_k),cur_im(ngen,simu_k),&
                 psi_re(ngen,simu_k),psi_im(ngen,simu_k),&
                 pelect(ngen,simu_k),pmech(ngen,simu_k),STAT=error)
        IF (error /= 0) THEN
           PRINT *,'Cannot allocate arrays'
        END IF

        ALLOCATE(eterm(ngen),qelect(ngen),curdg(ngen),curqg(ngen),&
                 curd(ngen),curq(ngen),ed(ngen),eq(ngen),vex(ngen),&
                 theta(nbus),bus_volt(nbus),STAT=error)
        IF (error /= 0) THEN
           PRINT *,'Cannot allocate arrays'
        END IF

        RETURN
        END SUBROUTINE allocSimu


