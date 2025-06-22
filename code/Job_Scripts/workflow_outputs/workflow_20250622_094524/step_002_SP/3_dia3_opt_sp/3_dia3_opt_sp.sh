#!/bin/bash --login
#SBATCH -J 3_dia3_opt_sp
#SBATCH -o 3_dia3_opt_sp-%J.o
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=32
#SBATCH -A mendoza_q
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --mem-per-cpu=4G
export JOB=3_dia3_opt_sp
export DIR=$SLURM_SUBMIT_DIR
export scratch=$SCRATCH/workflow_20250622_094524/step_002_SP

echo "submit directory: "
echo $SLURM_SUBMIT_DIR

module purge
module load CRYSTAL/23-intel-2023a
module load Python/3.11.3-GCCcore-12.3.0
module load Python-bundle-PyPI/2023.06-GCCcore-12.3.0

mkdir  -p $scratch/$JOB
cp $DIR/$JOB.d12  $scratch/$JOB/INPUT
cd $scratch/$JOB

mpirun -n $SLURM_NTASKS /opt/software-current/2023.06/x86_64/intel/skylake_avx512/software/CRYSTAL/23-intel-2023a/bin/Pcrystal 2>&1 >& $DIR/${JOB}.out
#srun Pcrystal 2>&1 >& $DIR/${JOB}.out
cp fort.9 ${DIR}/${JOB}.f9 


# ADDED: Auto-submit new jobs when this one completes
# Enhanced path resolution for workflow directory structure
QUEUE_MANAGER=""
if [ -f $DIR/enhanced_queue_manager.py ]; then
    QUEUE_MANAGER="$DIR/enhanced_queue_manager.py"
elif [ -f $DIR/../../../enhanced_queue_manager.py ]; then
    QUEUE_MANAGER="$DIR/../../../enhanced_queue_manager.py"
elif [ -f $DIR/../../../../enhanced_queue_manager.py ]; then
    QUEUE_MANAGER="$DIR/../../../../enhanced_queue_manager.py"
elif [ -f $DIR/../../../../../enhanced_queue_manager.py ]; then
    QUEUE_MANAGER="$DIR/../../../../../enhanced_queue_manager.py"
fi

if [ ! -z "$QUEUE_MANAGER" ]; then
    cd $(dirname "$QUEUE_MANAGER")
    python enhanced_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion
else
    echo "Warning: enhanced_queue_manager.py not found - workflow progression may not continue automatically"
fi
