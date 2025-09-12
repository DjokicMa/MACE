#!/bin/bash --login
# This script auto-generates and submits a SLURM batch file for CRYSTAL23 jobs, with queue management support

echo '#!/bin/bash --login' > $1.sh
echo '#SBATCH -J '$1 >> $1.sh
out=$1
file=-%J.o
outfile=$out$file
echo '#SBATCH -o '$outfile >> $1.sh
echo '#SBATCH --cpus-per-task=1' >> $1.sh
echo '#SBATCH --ntasks=32' >> $1.sh
echo '#SBATCH -A mendoza_q # or general' >> $1.sh 
#echo '#SBATCH --exclude=agg-[011-012],amr-[163,178-179]' >> $1.sh
echo '#SBATCH -N 1' >> $1.sh
time=7
wall=-00:00:00
timewall=$time$wall
echo '#SBATCH -t '$timewall >> $1.sh
echo '#SBATCH --mem-per-cpu=5G' >> $1.sh
echo 'export JOB='$1 >> $1.sh
echo 'export DIR=$SLURM_SUBMIT_DIR
export scratch=$SCRATCH/crys23

echo "submit directory: "
echo $SLURM_SUBMIT_DIR

module purge
module load CRYSTAL/23-intel-2023a
module load Python/3.11.3-GCCcore-12.3.0
module load Python-bundle-PyPI/2023.06-GCCcore-12.3.0

mkdir  -p $scratch/$JOB
cp $DIR/$JOB.d12  $scratch/$JOB/INPUT
cd $scratch/$JOB

I_MPI_HYDRA_BOOTSTRAP="ssh" mpirun -n $SLURM_NTASKS $EBROOTCRYSTAL/bin/Pcrystal 2>&1 >& $DIR/${JOB}.out
#srun Pcrystal 2>&1 >& $DIR/${JOB}.out
cp fort.9 ${DIR}/${JOB}.f9 


# ADDED: Auto-submit new jobs when this one completes
# Check multiple possible locations for queue managers
cd $DIR

# First check if MACE_HOME is set and use it
if [ ! -z "$MACE_HOME" ]; then
    if [ -f "$MACE_HOME/mace/queue/manager.py" ]; then
        QUEUE_MANAGER="$MACE_HOME/mace/queue/manager.py"
    elif [ -f "$MACE_HOME/enhanced_queue_manager.py" ]; then
        QUEUE_MANAGER="$MACE_HOME/enhanced_queue_manager.py"
    fi
else
    # MACE_HOME not set, try to find in PATH or relative locations
    # Try using which to find mace_cli (which we know works)
    MACE_CLI=$(which mace_cli 2>/dev/null)
    if [ ! -z "$MACE_CLI" ]; then
        # Found mace_cli, derive MACE_HOME from it
        MACE_HOME=$(dirname "$MACE_CLI")
        if [ -f "$MACE_HOME/mace/queue/manager.py" ]; then
            QUEUE_MANAGER="$MACE_HOME/mace/queue/manager.py"
        fi
    fi
fi

# If still not found, check standard relative locations
if [ -z "$QUEUE_MANAGER" ]; then
    if [ -f $DIR/mace/queue/manager.py ]; then
        QUEUE_MANAGER="$DIR/mace/queue/manager.py"
    elif [ -f $DIR/../../../../mace/queue/manager.py ]; then
        QUEUE_MANAGER="$DIR/../../../../mace/queue/manager.py"
    elif [ -f $DIR/../../../../../mace/queue/manager.py ]; then
        QUEUE_MANAGER="$DIR/../../../../../mace/queue/manager.py"
    elif [ -f $DIR/enhanced_queue_manager.py ]; then
        QUEUE_MANAGER="$DIR/enhanced_queue_manager.py"
    elif [ -f $DIR/../../../../enhanced_queue_manager.py ]; then
        QUEUE_MANAGER="$DIR/../../../../enhanced_queue_manager.py"
    elif [ -f $DIR/crystal_queue_manager.py ]; then
        QUEUE_MANAGER="$DIR/crystal_queue_manager.py"
    elif [ -f $DIR/../../../../crystal_queue_manager.py ]; then
        QUEUE_MANAGER="$DIR/../../../../crystal_queue_manager.py"
    fi
fi

if [ ! -z "$QUEUE_MANAGER" ]; then
    echo "Found queue manager at: $QUEUE_MANAGER"
    python "$QUEUE_MANAGER" --max-jobs 950 --reserve 50 --max-submit 10 --callback-mode completion --max-recovery-attempts 3
else
    echo "Warning: Queue manager not found. Checked:"
    echo "  - \$MACE_HOME/mace/queue/manager.py"
    echo "  - Various relative paths from $DIR"
    echo "  Workflow progression may not continue automatically"
fi' >> $1.sh
sbatch $1.sh
