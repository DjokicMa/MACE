#!/bin/bash --login
# Enhanced version of submitcrystal23.sh that supports both installed and local execution

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


# ENHANCED: Auto-submit new jobs with support for installed crystal-tools
# This version checks for both installed commands and local scripts

# First, check if crystal-tools is installed and available
if command -v crystal-queue >/dev/null 2>&1; then
    echo "Using installed crystal-queue command"
    # The installed command will handle finding the right scripts
    crystal-queue --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion --max-recovery-attempts 3
    
# Otherwise, fall back to checking for local scripts
elif [ -f $DIR/enhanced_queue_manager.py ]; then
    echo "Using enhanced_queue_manager.py from local directory"
    cd $DIR
    python enhanced_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion --max-recovery-attempts 3
    
elif [ -f $DIR/../../../../enhanced_queue_manager.py ]; then
    echo "Using enhanced_queue_manager.py from parent directory"
    cd $DIR/../../../../
    python enhanced_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion --max-recovery-attempts 3
    
elif [ -f $DIR/crystal_queue_manager.py ]; then
    echo "Using crystal_queue_manager.py from local directory"
    cd $DIR
    ./crystal_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5
    
elif [ -f $DIR/../../../../crystal_queue_manager.py ]; then
    echo "Using crystal_queue_manager.py from parent directory"
    cd $DIR/../../../../
    ./crystal_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5
    
else
    echo "Warning: No queue manager found"
    echo "Tried:"
    echo "  - crystal-queue command (not in PATH)"
    echo "  - $DIR/enhanced_queue_manager.py"
    echo "  - $DIR/../../../../enhanced_queue_manager.py"
    echo "  - $DIR/crystal_queue_manager.py"
    echo "  - $DIR/../../../../crystal_queue_manager.py"
    echo ""
    echo "To fix this, either:"
    echo "  1. Install crystal-tools and add to PATH"
    echo "  2. Copy required scripts to working directory"
fi' >> $1.sh
sbatch $1.sh