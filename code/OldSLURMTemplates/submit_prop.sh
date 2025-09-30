#!/bin/bash
# This script generates and submits SLURM job scripts for running CRYSTAL23 Pproperties calculations.
# Takes a job name as argument, creates a .sh file with SLURM directives, and submits it to the queue.

echo '#!/bin/bash --login' > $1.sh
echo '#SBATCH -J '$1 >> $1.sh
out=$1
file=-%J.o
outfile=$out$file
echo '#SBATCH -o '$outfile >> $1.sh
echo '#SBATCH --cpus-per-task=1' >> $1.sh
echo '#SBATCH --ntasks=28' >> $1.sh
echo '#SBATCH -A mendoza_q' >> $1.sh
#echo '#SBATCH --exclude=agg-[011-012],amr-[163,178-179]' >> $1.sh
echo '#SBATCH -N 1' >> $1.sh
time=2
wall=:00:00
timewall=$time$wall
echo '#SBATCH -t '$timewall >> $1.sh
echo '#SBATCH --mem=80G' >> $1.sh
echo 'export JOB='$1 >> $1.sh
echo 'export DIR=$SLURM_SUBMIT_DIR
export scratch=$SCRATCH/crys23/prop

echo "submit directory: "
echo $SLURM_SUBMIT_DIR
module purge

module load CRYSTAL/23-intel-2023a
module load Python/3.11.3-GCCcore-12.3.0
module load Python-bundle-PyPI/2023.06-GCCcore-12.3.0
mkdir  -p $scratch/$JOB

cp $DIR/$JOB.d3  $scratch/$JOB/INPUT
cp $DIR/$JOB.f9  $scratch/$JOB/fort.9
cd $scratch/$JOB

I_MPI_HYDRA_BOOTSTRAP="ssh" mpirun -n $SLURM_NTASKS $EBROOTCRYSTAL/bin/Pproperties 2>&1 >& $DIR/${JOB}.out
#srun $EBROOTCRYSTAL/bin/Pproperties 2>&1 >& $DIR/${JOB}.out
#srun Pproperties 2>&1 >& $DIR/${JOB}.out

cp fort.9  ${DIR}/${JOB}.f9
cp BAND.DAT  ${DIR}/${JOB}.BAND.DAT
cp fort.25  ${DIR}/${JOB}.f25
cp DOSS.DAT  ${DIR}/${JOB}.DOSS.DAT
cp POTC.DAT  ${DIR}/${JOB}.POTC.DAT
cp SIGMA.DAT ${DIR}/${JOB}.SIGMA.DAT
cp SEEBECK.DAT ${DIR}/${JOB}.SEEBECK.DAT
cp SIGMAS.DAT ${DIR}/${JOB}.SIGMAS.DAT
cp KAPPA.DAT ${DIR}/${JOB}.KAPPA.DAT
cp TDF.DAT ${DIR}/${JOB}.TDF.DAT
cp DENS_CUBE.DAT ${DIR}/${JOB}_DENS.CUBE 2>/dev/null || true
cp POT_CUBE.DAT ${DIR}/${JOB}_POT.CUBE 2>/dev/null || true
cp SPIN_CUBE.DAT ${DIR}/${JOB}_SPIN.CUBE 2>/dev/null || true

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
fi' >> $1.sh
sbatch $1.sh
