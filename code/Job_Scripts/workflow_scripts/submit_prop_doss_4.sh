#!/bin/bash --login
#SBATCH -J crystal_doss
#SBATCH -o crystal_doss-%J.o
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=28
#SBATCH --constraint=intel18
#SBATCH -N 1
#SBATCH -t 1-00:00:00
#SBATCH --mem=80G

export JOB=input
export DIR=$SLURM_SUBMIT_DIR
export scratch=$SCRATCH/crys23/prop

echo "submit directory: "
echo $SLURM_SUBMIT_DIR

module purge
module load CRYSTAL/23-intel-2023a

mkdir -p $scratch/$JOB
cp $DIR/input.d3 $scratch/$JOB/INPUT
cp $DIR/input.f9 $scratch/$JOB/fort.9
cd $scratch/$JOB

mpirun -n $SLURM_NTASKS Pproperties 2>&1 >& $DIR/output.out
cp fort.9 ${DIR}/input.f9
cp DOSS.DAT ${DIR}/input.DOSS.DAT
cp fort.25 ${DIR}/input.f25