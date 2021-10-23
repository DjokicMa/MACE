echo '#!/bin/bash' > $1.sh
echo '#SBATCH -J '$1 >> $1.sh
out=$1
file=-%J.o
outfile=$out$file
echo '#SBATCH -o '$outfile >> $1.sh
echo '#SBATCH --cpus-per-task=1' >> $1.sh
echo '#SBATCH --ntasks=32' >> $1.sh
echo '#SBATCH -p general-long' >> $1.sh
echo '#SBATCH -N 1' >> $1.sh
time=7
wall=-00:00:00
timewall=$time$wall
echo '#SBATCH -t '$timewall >> $1.sh
echo '#SBATCH --mem-per-cpu=5G' >> $1.sh
echo 'export JOB='$1 >> $1.sh
echo 'export DIR=$SLURM_SUBMIT_DIR
export scratch=$SCRATCH/crys17

echo "submit directory: "
echo $SLURM_SUBMIT_DIR

ml -* CRYSTAL/17

mkdir  -p $scratch/$JOB
cp $DIR/$JOB.d12  $scratch/$JOB/INPUT
cd $scratch/$JOB

srun Pcrystal 2>&1 >& $DIR/${JOB}.out
cp fort.9 ${DIR}/${JOB}.f9


mkdir  -p $scratch/$JOB' >> $1.sh

sbatch $1.sh
