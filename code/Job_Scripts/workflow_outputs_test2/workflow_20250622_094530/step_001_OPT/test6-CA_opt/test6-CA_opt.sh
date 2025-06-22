#!/bin/bash --login
#SBATCH -J test6-CA_opt
#SBATCH -o test6-CA_opt-%J.o
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=32
#SBATCH -A mendoza_q
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --mem-per-cpu=5G
export JOB=test6-CA_opt
export DIR=$SLURM_SUBMIT_DIR
export scratch=$SCRATCH/workflow_20250622_094530/step_001_OPT/test6-CA_opt

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
# Check multiple possible locations for queue managers
if [ -f $DIR/enhanced_queue_manager.py ]; then
    cd $DIR
    python enhanced_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion --max-recovery-attempts 3
elif [ -f $DIR/../../../../enhanced_queue_manager.py ]; then
    cd $DIR/../../../../
    python enhanced_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion --max-recovery-attempts 3
elif [ -f $DIR/crystal_queue_manager.py ]; then
    cd $DIR
    ./crystal_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5
elif [ -f $DIR/../../../../crystal_queue_manager.py ]; then
    cd $DIR/../../../../
    ./crystal_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5
fi
