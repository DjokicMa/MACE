#!/bin/bash
"""
This script generates and submits SLURM job scripts for running CRYSTAL23 Pproperties calculations.
Takes a job name as argument, creates a .sh file with SLURM directives, and submits it to the queue.
"""

echo '#!/bin/bash --login' > $1.sh
echo '#SBATCH -J '$1 >> $1.sh
out=$1
file=-%J.o
outfile=$out$file
echo '#SBATCH -o '$outfile >> $1.sh
echo '#SBATCH --cpus-per-task=1' >> $1.sh
echo '#SBATCH --ntasks=28' >> $1.sh
echo '#SBATCH --constraint=intel18' >> $1.sh 
#echo '#SBATCH -A general' >> $1.sh
#echo '#SBATCH --exclude=amr-163,amr-178,amr-179,acm-028,acm-034' >> $1.sh
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
mkdir  -p $scratch/$JOB
cp $DIR/$JOB.d3  $scratch/$JOB/INPUT
cp $DIR/$JOB.f9  $scratch/$JOB/fort.9
cd $scratch/$JOB
mpirun -n $SLURM_NTASKS Pproperties 2>&1 >& $DIR/${JOB}.out
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
cp TDF.DAT ${DIR}/${JOB}.TDF.DAT' >> $1.sh
sbatch $1.sh
