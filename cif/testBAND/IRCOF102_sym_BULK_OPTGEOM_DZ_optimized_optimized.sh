#!/bin/bash --login
#SBATCH -J IRCOF102_sym_BULK_OPTGEOM_DZ_optimized_optimized
#SBATCH -o IRCOF102_sym_BULK_OPTGEOM_DZ_optimized_optimized-%J.o
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=24
#SBATCH -A general
#SBATCH -N 1
#SBATCH -t 7-00:00:00
#SBATCH --mem-per-cpu=5G
export JOB=IRCOF102_sym_BULK_OPTGEOM_DZ_optimized_optimized
export DIR=$SLURM_SUBMIT_DIR
export scratch=$SCRATCH/crys23/SP

echo "submit directory: "
echo $SLURM_SUBMIT_DIR

module purge
module load CRYSTAL/23-intel-2023a

mkdir  -p $scratch/$JOB
cp $DIR/$JOB.d12  $scratch/$JOB/INPUT
cd $scratch/$JOB

mpirun -n $SLURM_NTASKS /opt/software-current/2023.06/x86_64/intel/skylake_avx512/software/CRYSTAL/23-intel-2023a/bin/Pcrystal 2>&1 >& $DIR/${JOB}.out
#srun Pcrystal 2>&1 >& $DIR/${JOB}.out
cp fort.9 ${DIR}/${JOB}.f9
