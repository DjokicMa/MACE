#!/bin/bash --login
#SBATCH -J $1
#SBATCH -o $1-%J.o
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=32
#SBATCH -A mendoza_q
#SBATCH -N 1
#SBATCH -t 3-00:00:00
#SBATCH --mem-per-cpu=4G

export JOB=$1
export DIR=$SLURM_SUBMIT_DIR
export scratch=$SCRATCH/crys23

echo "submit directory: "
echo $SLURM_SUBMIT_DIR

module purge
module load CRYSTAL/23-intel-2023a
module load Python/3.11.3-GCCcore-12.3.0

mkdir -p $scratch/$JOB
cp $DIR/$JOB.d12 $scratch/$JOB/INPUT
cp $DIR/$JOB.f9 $scratch/$JOB/fort.9
cd $scratch/$JOB

mpirun -n $SLURM_NTASKS /opt/software-current/2023.06/x86_64/intel/skylake_avx512/software/CRYSTAL/23-intel-2023a/bin/Pcrystal 2>&1 >& $DIR/${JOB}.out
cp fort.9 ${DIR}/${JOB}.f9

# ADDED: Auto-submit new jobs when this one completes
# Check multiple possible locations for queue managers (prefer base directory)
if [ -f $DIR/../../../../enhanced_queue_manager.py ]; then
    echo "Using enhanced_queue_manager.py from base directory"
    cd $DIR/../../../../
    python enhanced_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion --max-recovery-attempts 3
elif [ -f $DIR/enhanced_queue_manager.py ]; then
    echo "Using enhanced_queue_manager.py from local directory" 
    cd $DIR
    python enhanced_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion --max-recovery-attempts 3
elif [ -f $DIR/../../../../crystal_queue_manager.py ]; then
    echo "Using crystal_queue_manager.py from base directory"
    cd $DIR/../../../../
    ./crystal_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5
elif [ -f $DIR/crystal_queue_manager.py ]; then
    echo "Using crystal_queue_manager.py from local directory"
    cd $DIR
    ./crystal_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5
else
    echo "Warning: No queue manager found in $DIR or $DIR/../../../../"
fi