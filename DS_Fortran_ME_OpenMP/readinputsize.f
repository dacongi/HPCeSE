!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! readinputsize.f: Read input data size 
! Author: Shuangshuang Jin
! Last updated: 9-30-2019
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      SUBROUTINE readinputsize(filename)
      
      USE DEFDP
      USE CONSTANTS
      USE INPUTSIZE

      IMPLICIT NONE

      CHARACTER*200 filename
      INTEGER::i,j,k
      INTEGER::tst,flagF,ios
      REAL(KIND=DP)::tstrl1,tstrl2,label9

      !!! Initialize counters
      nbus=0
      nbrch=0
      ngen=0
      nSW=0
      nPV=0
      nPQ=0
      nswtch=0
      label9=9999999.0

      !!! Initialize module INPUTSIZE
      OPEN(1,FILE=filename,IOSTAT=ios)
      IF (ios<0) PRINT *,'Cannot open file: ',filename

      flagF=0
      DO WHILE (flagF .EQ. 0)
         READ(1,FMT=101,IOSTAT=ios) tstrl1,tstrl2
         flagF=INT(tstrl1)
         tst=INT(tstrl2)
         IF (flagF .NE. label9) THEN
            nbus=nbus+1
            IF (tst .EQ. swing_bus) nSW=nSW+1
            IF (tst .EQ. generator_bus) nPV=nPV+1
            IF (tst .EQ. load_bus) nPQ=nPQ+1
            flagF=0
         ELSE
            flagF=1
         END IF
      END DO
  101 FORMAT(F12.5,96X,F12.5)

      flagF=0
      DO WHILE (flagF .EQ. 0)
         READ(1,FMT=102,IOSTAT=ios) tstrl1
         flagF=INT(tstrl1)
         IF (ios<0) EXIT
         IF (flagF .NE. label9) THEN
            nbrch=nbrch+1
            flagF=0
         ELSE 
            flagF=1
         END IF
      END DO
  102 FORMAT(F12.5)
       
      flagF=0
      DO WHILE (flagF .EQ. 0)
         READ(1,FMT=102,IOSTAT=ios) tstrl1
         flagF=INT(tstrl1)
         IF (ios<0) EXIT
         IF (flagF .NE. label9) THEN
            ngen=ngen+1
            flagF=0
         ELSE 
            flagF=1
         END IF
      END DO

      flagF=0
      DO WHILE (flagF .EQ. 0)
         READ(1,FMT=103,IOSTAT=ios) tstrl1
         flagF=INT(tstrl1)
         IF (ios<0) EXIT
         IF (flagF .NE. label9) THEN
            nswtch=nswtch+1
            flagF=0
         ELSE 
            flagF=1
         END IF
      END DO
  103 FORMAT(F12.5)

      CLOSE(1)
      RETURN
      END
