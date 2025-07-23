#!/bin/bash --login
#SBATCH -J CoExtPandaCOF102_sym_BULK_OPTGEOM_DZ_optimized_optimized
#SBATCH -o CoExtPandaCOF102_sym_BULK_OPTGEOM_DZ_optimized_optimized-%J.o
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=32
#SBATCH -A mendoza_q # or general
#SBATCH --exclude=agg-[011-012],amr-[163,178-179]
#SBATCH -N 1
#SBATCH -t 7-00:00:00
#SBATCH --mem-per-cpu=5G
export JOB=CoExtPandaCOF102_sym_BULK_OPTGEOM_DZ_optimized_optimized
export DIR=$SLURM_SUBMIT_DIR
export scratch=$SCRATCH/crys23/SP

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
