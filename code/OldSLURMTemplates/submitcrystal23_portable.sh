#!/bin/bash --login
# Portable version that works for any installation location

# Detect MACE_HOME at submission time
if [ -z "$MACE_HOME" ]; then
    # Try to find it relative to this script
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    
    # Check if we're in the repository structure
    if [ -f "$SCRIPT_DIR/enhanced_queue_manager.py" ]; then
        # We're in Job_Scripts directory
        export MACE_HOME="$( cd "$SCRIPT_DIR/../.." && pwd )"
    elif [ -f "$SCRIPT_DIR/../../code/Job_Scripts/enhanced_queue_manager.py" ]; then
        # We're somewhere else in the repo
        export MACE_HOME="$( cd "$SCRIPT_DIR/../.." && pwd )"
    else
        echo "Warning: Could not auto-detect MACE_HOME"
        echo "Please set: export MACE_HOME=/path/to/mace"
        exit 1
    fi
fi

echo "Using MACE_HOME: $MACE_HOME"

# Now create the SLURM script with the detected path
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
echo 'export DIR=$SLURM_SUBMIT_DIR' >> $1.sh
echo 'export scratch=$SCRATCH/crys23' >> $1.sh
echo '' >> $1.sh
echo '# Crystal tools installation path (detected at submission time)' >> $1.sh
echo "export MACE_HOME=$MACE_HOME" >> $1.sh
echo '' >> $1.sh
echo 'echo "submit directory: $SLURM_SUBMIT_DIR"' >> $1.sh
echo 'echo "MACE_HOME: $MACE_HOME"' >> $1.sh
echo '' >> $1.sh
echo 'module purge' >> $1.sh
echo 'module load CRYSTAL/23-intel-2023a' >> $1.sh
echo 'module load Python/3.11.3-GCCcore-12.3.0' >> $1.sh
echo 'module load Python-bundle-PyPI/2023.06-GCCcore-12.3.0' >> $1.sh
echo '' >> $1.sh
echo 'mkdir -p $scratch/$JOB' >> $1.sh
echo 'cp $DIR/$JOB.d12 $scratch/$JOB/INPUT' >> $1.sh
echo 'cd $scratch/$JOB' >> $1.sh
echo '' >> $1.sh
echo 'I_MPI_HYDRA_BOOTSTRAP="ssh" mpirun -n $SLURM_NTASKS $EBROOTCRYSTAL/bin/Pcrystal 2>&1 >& $DIR/${JOB}.out' >> $1.sh
echo 'cp fort.9 ${DIR}/${JOB}.f9' >> $1.sh
echo '' >> $1.sh
echo '# Enhanced callback using detected MACE_HOME' >> $1.sh
echo 'if [ -n "$MACE_HOME" ] && [ -f "$MACE_HOME/code/Job_Scripts/enhanced_queue_manager.py" ]; then' >> $1.sh
echo '    echo "Using enhanced_queue_manager.py from MACE_HOME"' >> $1.sh
echo '    cd $DIR' >> $1.sh
echo '    python $MACE_HOME/code/Job_Scripts/enhanced_queue_manager.py \' >> $1.sh
echo '        --db-path ./materials.db \' >> $1.sh
echo '        --work-dir . \' >> $1.sh
echo '        --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion' >> $1.sh
echo 'elif [ -f $DIR/enhanced_queue_manager.py ]; then' >> $1.sh
echo '    echo "Using enhanced_queue_manager.py from local directory"' >> $1.sh
echo '    cd $DIR' >> $1.sh
echo '    python enhanced_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion' >> $1.sh
echo 'else' >> $1.sh
echo '    echo "Warning: No queue manager found"' >> $1.sh
echo 'fi' >> $1.sh

sbatch $1.sh