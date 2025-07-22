#!/bin/bash --login
# Environment-aware version of submitcrystal23.sh

echo '#!/bin/bash --login' > $1.sh
echo '#SBATCH -J '$1 >> $1.sh
out=$1
file=-%J.o
outfile=$out$file
echo '#SBATCH -o '$outfile >> $1.sh
echo '#SBATCH --cpus-per-task=1' >> $1.sh
echo '#SBATCH --ntasks=32' >> $1.sh
echo '#SBATCH -A mendoza_q' >> $1.sh
echo '#SBATCH -N 1' >> $1.sh
time=7
wall=-00:00:00
timewall=$time$wall
echo '#SBATCH -t '$timewall >> $1.sh
echo '#SBATCH --mem-per-cpu=5G' >> $1.sh
echo 'export JOB='$1 >> $1.sh
echo 'export DIR=$SLURM_SUBMIT_DIR
export scratch=$SCRATCH/crys23

# CRITICAL: Set MACE_HOME for the compute node
# Option 1: Hardcode the path (most reliable)
export MACE_HOME=/mnt/iscsi/UsefulScripts/Codebase/reorganization

# Option 2: Pass from current environment (requires --export with sbatch)
# The MACE_HOME should already be set if using --export=ALL

# Option 3: Try to detect from a standard location
# if [ -z "$MACE_HOME" ]; then
#     if [ -d "/shared/mace" ]; then
#         export MACE_HOME=/shared/mace
#     fi
# fi

echo "submit directory: $SLURM_SUBMIT_DIR"
echo "MACE_HOME: $MACE_HOME"

module purge
module load CRYSTAL/23-intel-2023a
module load Python/3.11.3-GCCcore-12.3.0
module load Python-bundle-PyPI/2023.06-GCCcore-12.3.0

mkdir -p $scratch/$JOB
cp $DIR/$JOB.d12 $scratch/$JOB/INPUT
cd $scratch/$JOB

I_MPI_HYDRA_BOOTSTRAP="ssh" mpirun -n $SLURM_NTASKS $EBROOTCRYSTAL/bin/Pcrystal 2>&1 >& $DIR/${JOB}.out
cp fort.9 ${DIR}/${JOB}.f9 

# Enhanced callback using MACE_HOME
if [ -n "$MACE_HOME" ] && [ -f "$MACE_HOME/code/Job_Scripts/enhanced_queue_manager.py" ]; then
    echo "Using enhanced_queue_manager.py from MACE_HOME"
    cd $DIR
    python $MACE_HOME/code/Job_Scripts/enhanced_queue_manager.py \
        --db-path ./materials.db \
        --work-dir . \
        --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion
        
# Fallback to local directory
elif [ -f $DIR/enhanced_queue_manager.py ]; then
    echo "Using enhanced_queue_manager.py from local directory"
    cd $DIR
    python enhanced_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion
    
# Other fallbacks...
else
    echo "Warning: No queue manager found"
    echo "MACE_HOME=$MACE_HOME"
    echo "Checked: $MACE_HOME/code/Job_Scripts/enhanced_queue_manager.py"
    echo "Checked: $DIR/enhanced_queue_manager.py"
fi' >> $1.sh

# Submit with environment export
echo "Submitting with MACE_HOME=$MACE_HOME"
sbatch --export=ALL $1.sh