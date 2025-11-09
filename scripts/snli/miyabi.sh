#!/bin/sh
#------ qsub option --------#
#PBS -q regular-c
#PBS -l select=10:mpiprocs=1
#PBS -r y
#PBS -o /home/$USER/ccg2lambda/log/snli.out
#PBS -e /home/$USER/ccg2lambda/log/snli.err
#PBS -N ccg2lambda

#------- Program execution -------#
module load singularity/4.2.1
mpiexec.hydra singularity exec docker://turx/ccg2lambda:latest bash /home/$USER/ccg2lambda/ccg_snli.sh
