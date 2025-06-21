#!/bin/bash --login
#SBATCH -J $1
#SBATCH -o $1-%J.o
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=28
#SBATCH -A mendoza_q
#SBATCH -N 1
#SBATCH -t 1-00:00:00
#SBATCH --mem=80G

export JOB=$1
export DIR=$SLURM_SUBMIT_DIR
export scratch=$SCRATCH/crys23/prop

echo "submit directory: "
echo $SLURM_SUBMIT_DIR

module purge
module load CRYSTAL/23-intel-2023a
module load Python/3.11.3-GCCcore-12.3.0

mkdir -p $scratch/$JOB
cp $DIR/$JOB.d3 $scratch/$JOB/INPUT
cp $DIR/$JOB.f9 $scratch/$JOB/fort.9
cd $scratch/$JOB

mpirun -n $SLURM_NTASKS /opt/software-current/2023.06/x86_64/intel/skylake_avx512/software/CRYSTAL/23-intel-2023a/bin/Pproperties 2>&1 >& $DIR/${JOB}.out
cp fort.9 ${DIR}/${JOB}.f9
cp DOSS.DAT ${DIR}/${JOB}.DOSS.DAT
cp fort.25 ${DIR}/${JOB}.f25

# ADDED: Auto-submit new jobs when this one completes
# Check multiple possible locations for queue managers
if [ -f $DIR/enhanced_queue_manager.py ]; then
    cd $DIR
    python enhanced_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion
elif [ -f $DIR/../../../../enhanced_queue_manager.py ]; then
    cd $DIR/../../../../
    python enhanced_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion
elif [ -f $DIR/crystal_queue_manager.py ]; then
    cd $DIR
    ./crystal_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5
elif [ -f $DIR/../../../../crystal_queue_manager.py ]; then
    cd $DIR/../../../../
    ./crystal_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5
fi
