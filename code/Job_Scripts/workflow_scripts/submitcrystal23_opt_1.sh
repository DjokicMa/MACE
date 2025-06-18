#!/bin/bash --login
#SBATCH -J crystal_opt
#SBATCH -o crystal_opt-%J.o
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=32
#SBATCH -A mendoza_q
#SBATCH -N 1
#SBATCH -t 7-00:00:00
#SBATCH --mem-per-cpu=5G

export JOB=input
export DIR=$SLURM_SUBMIT_DIR
export scratch=$SCRATCH/crys23

echo "submit directory: "
echo $SLURM_SUBMIT_DIR

module purge
module load CRYSTAL/23-intel-2023a

mkdir -p $scratch/$JOB
cp $DIR/input.d12 $scratch/$JOB/INPUT
cd $scratch/$JOB

mpirun -n $SLURM_NTASKS /opt/software-current/2023.06/x86_64/intel/skylake_avx512/software/CRYSTAL/23-intel-2023a/bin/Pcrystal 2>&1 >& $DIR/output.out
cp fort.9 ${DIR}/input.f9