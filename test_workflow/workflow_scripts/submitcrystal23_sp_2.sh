#!/bin/bash
#SBATCH --job-name=mat_test_sp
#SBATCH --output=mat_test_sp.o%j
#SBATCH --partition=general-compute-default
#SBATCH --qos=general-compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=4G
#SBATCH --time=3-00:00:00
#SBATCH --account=mendoza_q
#SBATCH --export=NONE

# TEMPLATE FOR SP CALCULATIONS
echo "SP template - will be customized by workflow engine"