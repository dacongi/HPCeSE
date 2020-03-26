!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! i_simu_innerloop.f: Define simulation inner loop 
! Author: Shuangshuang Jin
! Last updated: 9-30-2019
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        SUBROUTINE i_simu_innerloop(ii,S_Steps,flag)
        
        USE DEFDP
        USE CONSTANTS,ONLY:jay
        USE INPUTSIZE,ONLY:nbus,ngen
        USE INPUTMATRIX,ONLY:bus
        USE SIMULATION
        USE REDUCEYBUS

        IMPLICIT NONE
        INTEGER jj,ii,S_Steps,flag
        COMPLEX(KIND=DP)::cur

        cur = 0
        
        IF (flag==0) THEN
           DO jj=1,ngen,1
              cur=cur+prefY11(jj,ii)*(psi_re(jj,S_Steps)+&
                  jay*psi_im(jj,S_Steps))
           END DO
        ELSE IF (flag==1) THEN
           DO jj=1,ngen,1
              cur=cur+fY11(jj,ii)*(psi_re(jj,S_Steps)+&
                  jay*psi_im(jj,S_Steps))
           END DO
        ELSE IF (flag==2) THEN
           DO jj=1,ngen,1
              cur=cur+posfY11(jj,ii)*(psi_re(jj,S_Steps)+&
                  jay*psi_im(jj,S_Steps))
           END DO
        END IF

        cur_re(ii,S_Steps)=REAL(cur)
        cur_im(ii,S_Steps)=AIMAG(cur)

        RETURN
        END SUBROUTINE i_simu_innerloop

