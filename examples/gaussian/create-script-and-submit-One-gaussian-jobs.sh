#!/bin/bash
# submit_job.sh

PROJECT=f81

for i in *.inp; do
    if [ ! -f "${i%.*}.log" ]; then
        # Extract nprocs from the input file (handles whitespace and comments)
        NCPUS=$(grep -E '^[[:space:]]*nprocs[[:space:]]*=' "$i" | \
                sed -E 's/^[[:space:]]*nprocs[[:space:]]*=[[:space:]]*([0-9]+).*/\1/')
        
        # Fallback if nprocs not found
        if [ -z "$NCPUS" ] || ! [[ "$NCPUS" =~ ^[0-9]+$ ]]; then
            echo "Warning: Could not read valid nprocs from $i. Using default 8."
            NCPUS=8
        fi

        MEM=$((NCPUS * 4))
        JOBFS=$((NCPUS * 400 / 48))   # Proportional: max 48 CPUs get 400GB

        echo "Submitting $i with ${NCPUS} CPUs, ${MEM}GB RAM, ${JOBFS}GB jobfs"

        cat > "${i%.inp}.cmd" << EOF
#!/bin/bash
#PBS -P ${PROJECT}
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l mem=${MEM}GB
#PBS -l ncpus=${NCPUS}
#PBS -l jobfs=${JOBFS}GB
#PBS -l wd
#PBS -l storage=scratch/f81+gdata/f81
#PBS -j oe
#PBS -r n

INPUT=\${PBS_JOBNAME%.*}.inp
OUTPUT=\${PBS_JOBNAME%.*}.log

module load gaussian/g16c01
module load nbo/7.0
export PATH=\$PATH:/home/595/np9048/sources/xtb/xtb-6.5.0/bin

PROGRAM='g16'

export GAUSS_PDEF=\$PBS_NCPUS

NCPUS_MAX=\$(lscpu | awk '/Core/{print \$4} /Socket/{print \$2}' | paste -d"*" - - | bc)

if [[ \$PBS_NCPUS -gt \$NCPUS_MAX ]]; then
   echo "The number of requested cpus is \$PBS_NCPUS. This is more than \$NCPUS_MAX"
   echo "what \$PROGRAM can handle in this queue. The job is terminated for this reason."
   exit 1
fi

GAUSS_MEM_OVERHEAD=7500
export GAUSS_MDEF="\$((PBS_VMEM/1048576-GAUSS_MEM_OVERHEAD))MB"

if [ -n "\$PBS_NCI_JOBFS_LOCAL" ]; then
   MAXDISK=\${PBS_NCI_JOBFS_LOCAL%%[^0-9]*}
elif [ -n "\$PBS_NCI_JOBFS_GLOBAL" ]; then
   MAXDISK=\${PBS_NCI_JOBFS_GLOBAL%%[^0-9]*}
fi
MAXDISK=\$((MAXDISK/1048576))
export GAUSS_RDEF="Maxdisk=\${MAXDISK}MB"

printf '=%.0s' {1..86}; printf '\n'
echo " The Gaussian environment is defined as:"
echo " GAUSS_PDEF=\$GAUSS_PDEF"
echo " GAUSS_MDEF=\$GAUSS_MDEF"
echo " GAUSS_RDEF=\$GAUSS_RDEF"
printf '=%.0s' {1..86}; printf '\n'

#\$PROGRAM \$INPUT >& \$OUTPUT
omecp \$INPUT > \$OUTPUT
EOF

        qsub "${i%.*}.cmd"
        echo "$i was submitted with ${NCPUS} CPUs, ${MEM}GB memory, and ${JOBFS}GB jobfs"
        rm "${i%.*}.cmd"

    else
        echo "${i%.*}.log exists → skipping $i"
    fi
done