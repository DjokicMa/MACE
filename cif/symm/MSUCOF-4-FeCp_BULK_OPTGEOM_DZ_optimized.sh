#!/bin/bash
#SBATCH -J MSUCOF-4-FeCp_BULK_OPTGEOM_DZ_optimized
#SBATCH -o MSUCOF-4-FeCp_BULK_OPTGEOM_DZ_optimized-%J.o
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=32
#SBATCH -A mendoza_q
#SBATCH -N 1
#SBATCH -t 7-00:00:00
#SBATCH --mem-per-cpu=5G
export JOB=MSUCOF-4-FeCp_BULK_OPTGEOM_DZ_optimized
export DIR=$SLURM_SUBMIT_DIR
export scratch=$SCRATCH/crys17

echo "submit directory: "
echo $SLURM_SUBMIT_DIR

module purge
module load CRYSTAL/17-intel-2023a

mkdir  -p $scratch/$JOB
cp $DIR/$JOB.d12  $scratch/$JOB/INPUT
cd $scratch/$JOB

mpirun -n $SLURM_NTASKS Pcrystal 2>&1 >& $DIR/${JOB}.out
cp fort.9 ${DIR}/${JOB}.f9
