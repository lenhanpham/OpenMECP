#!/bin/bash 
#PBS -P f81       
#PBS -q normal 
#PBS -l walltime=48:00:00 
#PBS -l mem=40GB 
#PBS -l ncpus=10      
#PBS -l jobfs=100GB 
#PBS -l wd
#PBS -l storage=scratch/f81+gdata/f81   
 
###  ''Join'' the standard output and standard error into
###  a single file 
#PBS -j oe 
 
### Make the job ''not rerunable'' so that if stopped/killed
### it will not be restarted
#PBS -r n

currdir=$(pwd)

 
INPUT=${PBS_JOBNAME%.*}.inp 
OUTPUT=${PBS_JOBNAME%.*}.log
 
module load gaussian/g16c01 
module load orca
module load nbo/7.0 
export PATH=$PATH:/home/595/np9048/sources/xtb/xtb-6.5.0/bin:/home/595/np9048/sources/OpenMECP/target/release   
module load python3/3.8.5 
 
PROGRAM='g16' 
 
# Setting up the number of cpus to be used by program (equivalent of %NProc=)
export GAUSS_PDEF=$PBS_NCPUS 
NCPUS_MAX=`lscpu | awk '/Core/{print $4} /Socket/{print $2}' | paste -d"*" - - |bc`
if [[ PBS_NCPUS -gt NCPUS_MAX ]]
then
   echo "The number of requested cpus is $PBS_NCPUS. This is more than $NCPUS_MAX"
   echo "what $PROGRAM can handle in this queue. The job is terminated for this reason."
   exit 1
fi
 
# Setting up the amount of memory to be used by program (equivalent of %Mem=)
# amount of the memory overhead set to be 2048 (in MB). You can change
# it revising the line below
GAUSS_MEM_OVERHEAD=7500  
export GAUSS_MDEF="$((PBS_VMEM/1048576-GAUSS_MEM_OVERHEAD))MB"
 
#  Setting up maxdisk limit via GAUSS_RDEF (equivalent in Maxdisk=)
if [ -n "$PBS_NCI_JOBFS_LOCAL" ]
then
   MAXDISK=${PBS_NCI_JOBFS_LOCAL%%[^0-9]*} # in bytes
elif [ -n "$PBS_NCI_JOBFS_GLOBAL" ]
then
   MAXDISK=${PBS_NCI_JOBFS_GLOBAL%%[^0-9]*} # in bytes 
fi
MAXDISK=$((MAXDISK/1048576)) # in MB 
export GAUSS_RDEF="Maxdisk=${MAXDISK}MB"
 
printf '=%.0s' {1..86}; printf '\n'
echo " The Gaussian environment is defined as:"
echo " GAUSS_PDEF=$GAUSS_PDEF"
echo " GAUSS_MDEF=$GAUSS_MDEF"
echo " GAUSS_RDEF=$GAUSS_RDEF"
printf '=%.0s' {1..86}; printf '\n'
 
#$PROGRAM $INPUT >& $OUTPUT 

omecp $INPUT > $OUTPUT 

#rm *.chk  
