#!/bin/bash --login
#SBATCH -J 3,4^2T1-CA_BULK_OPTGEOM_TZ_band
#SBATCH -o 3,4^2T1-CA_BULK_OPTGEOM_TZ_band-%J.o
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=28
#SBATCH -A mendoza_q
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH --mem=80G
export JOB=3,4^2T1-CA_BULK_OPTGEOM_TZ_band
export DIR=$SLURM_SUBMIT_DIR
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
cp *.CUBE ${DIR}/ 2>/dev/null || true
cp *.cube ${DIR}/ 2>/dev/null || true

# ADDED: Auto-submit new jobs when this one completes
# Use centralized MACE installation (scripts are in PATH)
cd $DIR
enhanced_queue_manager.py --max-jobs 250 --reserve 30 --max-submit 5 --callback-mode completion --max-recovery-attempts 3
