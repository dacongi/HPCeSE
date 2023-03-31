#!/bin/sh

############################################################
#                                                          #
#    Cong's shell script for submitting a parallel MPI     #
#    job to the PBS queue using the qsub command.          #
#                                                          #
############################################################

#     Remarks: A line beginning with # is a comment.
#	       A line beginning with #PBS is a PBS directive.
#              PBS directives must come first; any directives
#                 after the first executable statement are ignored.
   
##########################
#                        #
#   The PBS directives   #
#                        #
##########################

#          Set the name of the job (up to 15 characters, 
#          no blank spaces, start with alphanumeric character)

#PBS -N JobName

#          Specify the number of nodes, CPU/GPU requested and the
#          number of processors per node. 

#PBS -l select=3:ncpus=8:mpiprocs=8:mem=8gb:interconnect=hdr

#          The directive below directs that the standard output and
#          error streams are to be merged, intermixed, as standard
#          output. 

#PBS -j oe

#          Specify the maximum cpu time. 
#          Format:   hhhh:mm:ss   hours:minutes:seconds
#          Be sure to specify a reasonable value here.
#          If the job does not finish by the time reached,
#          the job is terminated.

#PBS -l walltime=00:30:00,place=scatter

##########################################
#                                        #
#   Output some useful job information.  #
#                                        #
##########################################

NCPU=`wc -l < $PBS_NODEFILE`
#NCPU=8
echo ------------------------------------------------------
echo ' This job is allocated on '${NCPU}' cpu(s)'
echo 'Job is running on node(s): '
#cat $PBS_NODEFILE
echo ------------------------------------------------------
echo PBS: qsub is running on $PBS_O_HOST
echo PBS: originating queue is $PBS_O_QUEUE
echo PBS: executing queue is $PBS_QUEUE
echo PBS: working directory is $PBS_O_WORKDIR
echo PBS: execution mode is $PBS_ENVIRONMENT
echo PBS: job identifier is $PBS_JOBID
echo PBS: job name is $PBS_JOBNAME
echo PBS: node file is $PBS_NODEFILE
echo PBS: current home directory is $PBS_O_HOME
echo PBS: PATH = $PBS_O_PATH
echo ------------------------------------------------------

##############################################################
#                                                            #
#   The prologue script automatically makes a directory      #
#   on the local disks for you.  The name of this directory  #
#   depends on the job id, but you need only refer to it     #
#   using ${WORKDIR}.                                        #
#                                                            #
##############################################################

SERVER=$PBS_O_HOST
WORKDIR=/home/cong2/
NPROCESS=24
LAUNCH="mpirun -np $NPROCESS --mca opal_cuda_support 0 python -W ignore"

echo server is $SERVER
echo workdir is $WORKDIR
echo ------------------------------------------------------
echo 'Job is running on node(s): '
#cat $PBS_NODEFILE
echo ------------------------------------------------------
echo 'Job is using '${NPROCESS}' processes'
echo ------------------------------------------------------
echo ' '
echo 'Results:'

############################################################
#                                                          #
#    Execute the run.  Do not run in the background.       #
#                                                          #
############################################################

module load anaconda3/2022.05-gcc/9.5.0
source activate paper5
#module purge
module load openmpi/4.1.4-gcc/9.5.0-ucx

 cd ${WORKDIR}
 echo PBS: working directory is $WORKDIR
 ${LAUNCH} try.py


###############################################################
#                                                             #
#   The epilogue script automatically deletes the directory   #
#   created on the local disk (including all files contained  #
#   therein.                                                  #
#                                                             #
###############################################################

exit