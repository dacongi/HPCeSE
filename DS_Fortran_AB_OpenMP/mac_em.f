!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! mac_em.f: Generator electromechanical model 
!           Input: i - generator number (not needed now)
!                    - 0 for vectorized computation
!                  simu_k - integer time
! Author: Shuangshuang Jin
! Last updated: 9-30-2019
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

       SUBROUTINE mac_em1(i,sim_k)
        !!! Generator electromechanical model
        !!! mac_em1: network interface computation
       
        USE DEFDP
        USE CONSTANTS,ONLY:basmva,basrad,jay
        USE INPUTSIZE,ONLY:nbus,ngen
        USE SIMULATION
        USE OMP_LIB

        IMPLICIT NONE
        INTEGER::i,sim_k

        psi_re(i,sim_k)=Dsin(mac_ang(i,sim_k))*edprime(i,sim_k)+&
                        Dcos(mac_ang(i,sim_k))*eqprime(i,sim_k)
        psi_im(i,sim_k)=-Dcos(mac_ang(i,sim_k))*edprime(i,sim_k)+&
                        Dsin(mac_ang(i,sim_k))*eqprime(i,sim_k)

        RETURN
        END SUBROUTINE mac_em1

        SUBROUTINE mac_em2(i,sim_k)
        !!! Generator dynamics computation and state matrix building 
       
        USE DEFDP
        USE CONSTANTS,ONLY:basmva,basrad,jay
        USE INPUTSIZE,ONLY:nbus,ngen
        USE SIMULATION 
        USE OMP_LIB
        USE INPUTMATRIX,ONLY:mac_con

        IMPLICIT NONE
        INTEGER::i,sim_k

        curd(i)=Dsin(mac_ang(i,sim_k))*cur_re(i,sim_k)-&
                Dcos(mac_ang(i,sim_k))*cur_im(i,sim_k) ! d-axis current
        curq(i)=Dcos(mac_ang(i,sim_k))*cur_re(i,sim_k)+&
                Dsin(mac_ang(i,sim_k))*cur_im(i,sim_k) ! q-axis current
        curdg(i)=curd(i)*mac_con(i,3)
        curqg(i)=curq(i)*mac_con(i,3)
        ed(i)=edprime(i,sim_k)+mac_con(i,7)*curqg(i)
        eq(i)=eqprime(i,sim_k)-mac_con(i,7)*curdg(i)
        eterm(i)=Dsqrt(ed(i)*ed(i)+eq(i)*eq(i))
        pelect(i,sim_k)=eq(i)*curq(i)+ed(i)*curd(i)
        qelect(i)=eq(i)*curd(i)-ed(i)*curq(i)
        dmac_ang(i,sim_k)=basrad*(mac_spd(i,sim_k)-1.0)
        dmac_spd(i,sim_k)=(pmech(i,sim_k)-pelect(i,sim_k)*&
                          mac_con(i,3)-mac_con(i,17)*(mac_spd(i,sim_k)&
                          -1.0))/(2*mac_con(i,16))
       
        RETURN
        END SUBROUTINE mac_em2 
