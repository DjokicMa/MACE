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
# SCRATCH GUARD. $SCRATCH can be EMPTY inside a job, even under bash --login
# (seen on agx-000). scratch then became /crys23, mkdir was refused, INPUT was
# never written and CRYSTAL stopped with END OF DATA IN INPUT DECK. So the
# directory is checked before anything is staged in it, and when it cannot be
# used the first writable one of these is used instead, keeping the part of
# the path after $SCRATCH:
#   1. /mnt/scratch/$USER resolved - what /etc/profile.d/hpcc.sh sets $SCRATCH
#      to on MSU HPCC, so the same directory a healthy node uses (and the one
#      a recovery looks in for OPTINFO.DAT);
#   2. .mace_scratch under the submit directory (shared, kept after the job);
#   3. $TMPDIR (node-local and removed with the job - last resort).
# With none writable the job stops here rather than run CRYSTAL on an empty
# INPUT. The directory used is written to $DIR/.$JOB.scratch, where the
# recovery reads it back.
MACE_SCRATCH_WANTED=$scratch
if [ -n "$SCRATCH" ]; then MACE_SCRATCH_SUFFIX=${scratch#"$SCRATCH"}; else MACE_SCRATCH_SUFFIX=$scratch; fi
case "$MACE_SCRATCH_SUFFIX" in /*) ;; *) MACE_SCRATCH_SUFFIX=/$MACE_SCRATCH_SUFFIX ;; esac
MACE_USER=${USER:-$(id -un 2>/dev/null)}
MACE_SCRATCH_TRY=()
[ -n "$SCRATCH" ] && MACE_SCRATCH_TRY+=("$scratch")
if [ -n "$MACE_USER" ] && [ -d "/mnt/scratch/$MACE_USER" ]; then
  MACE_SCRATCH_TRY+=("$(readlink -f "/mnt/scratch/$MACE_USER")$MACE_SCRATCH_SUFFIX")
fi
[ -n "$DIR" ] && MACE_SCRATCH_TRY+=("$DIR/.mace_scratch$MACE_SCRATCH_SUFFIX")
[ -n "$TMPDIR" ] && MACE_SCRATCH_TRY+=("$TMPDIR$MACE_SCRATCH_SUFFIX")
[ -z "$SCRATCH" ] && echo "scratch: \$SCRATCH is empty on $(hostname)"
scratch=""
for MACE_D in "${MACE_SCRATCH_TRY[@]}"; do
  if mkdir -p "$MACE_D/$JOB" 2>/dev/null && touch "$MACE_D/$JOB/.mace_write_test" 2>/dev/null; then
    rm -f "$MACE_D/$JOB/.mace_write_test"
    scratch=$MACE_D
    break
  fi
  echo "scratch: cannot write to $MACE_D/$JOB"
done
if [ -z "$scratch" ]; then
  echo "ERROR: no writable scratch directory on $(hostname) (\$SCRATCH=\"$SCRATCH\"); tried: ${MACE_SCRATCH_TRY[*]}" | tee -a "$DIR/$JOB.out"
  echo "  Not running CRYSTAL: it would read an empty INPUT." | tee -a "$DIR/$JOB.out"
  exit 1
fi
if [ "$scratch" != "$MACE_SCRATCH_WANTED" ]; then
  echo "scratch: using $scratch/$JOB instead of $MACE_SCRATCH_WANTED/$JOB"
fi
echo "scratch directory: $scratch/$JOB"
# The last run of this job may have used another directory (a node where the
# fallback was needed, or the other way round): an OPTGEOM RESTART reads
# OPTINFO.DAT and fort.20 from here, so bring them over when only that one has
# them.
MACE_SCRATCH_PREV=$(cat "$DIR/.$JOB.scratch" 2>/dev/null)
if [ -n "$MACE_SCRATCH_PREV" ] && [ "$MACE_SCRATCH_PREV" != "$scratch/$JOB" ] \
   && [ -f "$MACE_SCRATCH_PREV/OPTINFO.DAT" ] && [ ! -f "$scratch/$JOB/OPTINFO.DAT" ]; then
  for MACE_F in OPTINFO.DAT fort.20 fort.9; do
    [ -f "$MACE_SCRATCH_PREV/$MACE_F" ] && cp -p "$MACE_SCRATCH_PREV/$MACE_F" "$scratch/$JOB/$MACE_F"
  done
  echo "scratch: brought OPTINFO.DAT over from the previous run in $MACE_SCRATCH_PREV"
fi
echo "$scratch/$JOB" > "$DIR/.$JOB.scratch" 2>/dev/null
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
# Check multiple possible locations for queue managers
cd $DIR

# First check if MACE_HOME is set and use it
if [ ! -z "$MACE_HOME" ]; then
    if [ -f "$MACE_HOME/mace/queue/manager.py" ]; then
        QUEUE_MANAGER="$MACE_HOME/mace/queue/manager.py"
    elif [ -f "$MACE_HOME/enhanced_queue_manager.py" ]; then
        QUEUE_MANAGER="$MACE_HOME/enhanced_queue_manager.py"
    fi
else
    # MACE_HOME not set, try to find in PATH or relative locations
    # Try using which to find mace_cli (which we know works)
    MACE_CLI=$(which mace_cli 2>/dev/null)
    if [ ! -z "$MACE_CLI" ]; then
        # Found mace_cli, derive MACE_HOME from it
        MACE_HOME=$(dirname "$MACE_CLI")
        if [ -f "$MACE_HOME/mace/queue/manager.py" ]; then
            QUEUE_MANAGER="$MACE_HOME/mace/queue/manager.py"
        fi
    fi
fi

# If still not found, check standard relative locations
if [ -z "$QUEUE_MANAGER" ]; then
    if [ -f $DIR/mace/queue/manager.py ]; then
        QUEUE_MANAGER="$DIR/mace/queue/manager.py"
    elif [ -f $DIR/../../../../mace/queue/manager.py ]; then
        QUEUE_MANAGER="$DIR/../../../../mace/queue/manager.py"
    elif [ -f $DIR/../../../../../mace/queue/manager.py ]; then
        QUEUE_MANAGER="$DIR/../../../../../mace/queue/manager.py"
    elif [ -f $DIR/enhanced_queue_manager.py ]; then
        QUEUE_MANAGER="$DIR/enhanced_queue_manager.py"
    elif [ -f $DIR/../../../../enhanced_queue_manager.py ]; then
        QUEUE_MANAGER="$DIR/../../../../enhanced_queue_manager.py"
    elif [ -f $DIR/crystal_queue_manager.py ]; then
        QUEUE_MANAGER="$DIR/crystal_queue_manager.py"
    elif [ -f $DIR/../../../../crystal_queue_manager.py ]; then
        QUEUE_MANAGER="$DIR/../../../../crystal_queue_manager.py"
    fi
fi

if [ ! -z "$QUEUE_MANAGER" ]; then
    echo "Found queue manager at: $QUEUE_MANAGER"
    # Check for workflow context database
    if [ ! -z "$MACE_CONTEXT_DIR" ] && [ -f "$MACE_CONTEXT_DIR/materials.db" ]; then
        echo "Using workflow context database: $MACE_CONTEXT_DIR/materials.db"
        python "$QUEUE_MANAGER" --max-jobs 950 --reserve 50 --max-submit 10 --callback-mode completion --max-recovery-attempts 3 --db-path "$MACE_CONTEXT_DIR/materials.db"
    else
        python "$QUEUE_MANAGER" --max-jobs 950 --reserve 50 --max-submit 10 --callback-mode completion --max-recovery-attempts 3
    fi
else
    echo "Warning: Queue manager not found. Checked:"
    echo "  - \$MACE_HOME/mace/queue/manager.py"
    echo "  - Various relative paths from $DIR"
    echo "  Workflow progression may not continue automatically"
fi' >> $1.sh
# NOTE: Job submission is now handled by the calling script (properties.py)
# This allows for proper handling of --nosubmit flag
# sbatch $1.sh
