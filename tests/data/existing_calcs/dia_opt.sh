#!/bin/bash --login
#SBATCH -J dia_opt
#SBATCH -o dia_opt-%J.o
#SBATCH --ntasks=8
#SBATCH -t 2:00:00
#SBATCH --mem-per-cpu=4G
# edited by hand: fewer cores than MACE writes
export JOB=dia_opt
export DIR=$SLURM_SUBMIT_DIR
mkdir -p $SCRATCH/$JOB && cp $DIR/$JOB.d12 $SCRATCH/$JOB/INPUT && cd $SCRATCH/$JOB
mpirun -n $SLURM_NTASKS Pcrystal >& $DIR/$JOB.out
