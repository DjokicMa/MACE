#!/bin/bash --login
# This script auto-generates and submits a SLURM batch file for CRYSTAL23 jobs, with queue management support

echo '#!/bin/bash --login' > $1.sh
echo '#SBATCH -J '$1 >> $1.sh
out=$1
file=-%J.o
outfile=$out$file
echo '#SBATCH -o '$outfile >> $1.sh
echo '#SBATCH --cpus-per-task=1' >> $1.sh
echo '#SBATCH --ntasks=32' >> $1.sh
echo '#SBATCH -A mendoza_q # or general' >> $1.sh 
#echo '#SBATCH --exclude=agg-[011-012],amr-[163,178-179]' >> $1.sh
echo '#SBATCH -N 1' >> $1.sh
time=7
wall=-00:00:00
timewall=$time$wall
echo '#SBATCH -t '$timewall >> $1.sh
echo '#SBATCH --mem-per-cpu=5G' >> $1.sh
echo 'export JOB='$1 >> $1.sh
echo 'export DIR=$SLURM_SUBMIT_DIR
export scratch=$SCRATCH/crys23

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
  # $DIR/$JOB.out must hold only this error. Appended to the output of the
  # last run of this job, that run (finished, or failed on something else)
  # would be read as the result of this one. The old output is kept as
  # $JOB.out.prev<N>; where it cannot be moved it is overwritten.
  if [ -f "$DIR/$JOB.out" ]; then
    MACE_N=1
    while [ -e "$DIR/$JOB.out.prev$MACE_N" ]; do MACE_N=$((MACE_N+1)); done
    if mv "$DIR/$JOB.out" "$DIR/$JOB.out.prev$MACE_N" 2>/dev/null; then
      echo "scratch: kept the previous output as $JOB.out.prev$MACE_N"
    else
      echo "scratch: cannot move the previous $JOB.out aside - overwriting it"
    fi
  fi
  echo "ERROR: no writable scratch directory on $(hostname) (\$SCRATCH=\"$SCRATCH\"); tried: ${MACE_SCRATCH_TRY[*]}" | tee "$DIR/$JOB.out"
  echo "  Not running CRYSTAL: it would read an empty INPUT." | tee -a "$DIR/$JOB.out"
  exit 1
fi
if [ "$scratch" != "$MACE_SCRATCH_WANTED" ]; then
  echo "scratch: using $scratch/$JOB instead of $MACE_SCRATCH_WANTED/$JOB"
fi
echo "scratch directory: $scratch/$JOB"
# The last run of this job may have used another directory (a node where the
# fallback was needed, or the other way round): an OPTGEOM RESTART reads
# OPTINFO.DAT and fort.20 from here. When that directory has the newer
# OPTINFO.DAT (or this one has none), OPTINFO.DAT, fort.20 and fort.9 are
# taken from it together, so the history and the density matrix always come
# from the same run - one it lacks is removed here rather than left over from
# an older run.
MACE_SCRATCH_PREV=$(cat "$DIR/.$JOB.scratch" 2>/dev/null)
if [ -n "$MACE_SCRATCH_PREV" ] && [ "$MACE_SCRATCH_PREV" != "$scratch/$JOB" ] \
   && [ -f "$MACE_SCRATCH_PREV/OPTINFO.DAT" ] \
   && [ "$MACE_SCRATCH_PREV/OPTINFO.DAT" -nt "$scratch/$JOB/OPTINFO.DAT" ]; then
  for MACE_F in OPTINFO.DAT fort.20 fort.9; do
    if [ -f "$MACE_SCRATCH_PREV/$MACE_F" ]; then
      cp -p "$MACE_SCRATCH_PREV/$MACE_F" "$scratch/$JOB/$MACE_F"
    else
      rm -f "$scratch/$JOB/$MACE_F"
    fi
  done
  echo "scratch: brought OPTINFO.DAT, fort.20 and fort.9 over from the previous run in $MACE_SCRATCH_PREV"
fi
echo "$scratch/$JOB" > "$DIR/.$JOB.scratch" 2>/dev/null
mkdir  -p $scratch/$JOB
cp $DIR/$JOB.d12  $scratch/$JOB/INPUT
# OPTGEOM RESTART - a walltime-killed optimization continuing from its last
# completed step. The recovery adds RESTART to the OPTGEOM block of the same
# deck, so this job has the same $JOB name and therefore the same scratch
# directory as the killed one. CRYSTAL reads the earlier steps from
# OPTINFO.DAT there, and the SCF guess from fort.20 - the density matrix of
# the last completed SCF. During an optimization CRYSTAL keeps that matrix in
# fort.20 itself, while fort.9 stays EMPTY until the run ends (measured on a
# killed OPT: fort.9 0 bytes, fort.20 rewritten each step), so the fort.20 the
# killed run left behind is the one to use and nothing may overwrite it - which is why
# this runs before the GUESSP staging and tells it to stand aside. Without
# OPTINFO.DAT there is nothing to continue, so RESTART comes back out of the
# scratch copy and the job starts over; $DIR/$JOB.d12 itself is not modified.
RESTART_KEEPS_FORT20=""
if [ -f "$scratch/$JOB/INPUT" ] && sed -n "/^[[:space:]]*OPTGEOM[[:space:]]*$/I,/^[[:space:]]*END/Ip" "$scratch/$JOB/INPUT" | grep -qiE "^[[:space:]]*RESTART[[:space:]]*$"; then
  if [ -f "$scratch/$JOB/OPTINFO.DAT" ]; then
    if [ -s "$scratch/$JOB/fort.20" ]; then
      RESTART_KEEPS_FORT20=1
      echo "RESTART: continuing the optimization from OPTINFO.DAT, SCF guess from its own fort.20"
    elif [ -s "$scratch/$JOB/fort.9" ]; then
      cp "$scratch/$JOB/fort.9" "$scratch/$JOB/fort.20"
      RESTART_KEEPS_FORT20=1
      echo "RESTART: continuing the optimization from OPTINFO.DAT, SCF guess from fort.9"
    else
      echo "RESTART: continuing the optimization from OPTINFO.DAT (no density matrix left in scratch)"
    fi
  else
    sed -i "/^[[:space:]]*OPTGEOM[[:space:]]*$/I,/^[[:space:]]*END/I{/^[[:space:]]*RESTART[[:space:]]*$/Id}" "$scratch/$JOB/INPUT"
    echo "RESTART requested but no OPTINFO.DAT in $scratch/$JOB - dropped it from"
    echo "  this run and starting the optimization over"
  fi
fi
# GUESSP restart. CRYSTAL reads the starting density matrix from fort.20 -
# "copy file fort.9 to fort.20" - and every run saves its own converged matrix
# as $JOB.f9 further down, so the material already has one on disk after any
# earlier run.
#
# Only staged when the deck actually asks for it. A .f9 can be large and
# CRYSTAL ignores fort.20 without GUESSP, so copying it unconditionally would
# be pure I/O; gating on the keyword also keeps the script honest about what
# the deck is doing.
#
# Two sources, in order of specificity:
#   $JOB.f20  a matrix deliberately staged for this job, e.g. a chained step
#             restarting from a DIFFERENT predecessor (OPT -> SP), where the
#             predecessor is named after its own job and not this one.
#             Only a NON-EMPTY one, like the .f9 below: an empty file is no guess.
#   $JOB.f9   this material own matrix from an earlier run of this same job -
#             the walltime-killed restart, where the right guess is the one it
#             already produced. Only a NON-EMPTY one: a run CRYSTAL aborted
#             in the middle of an optimization still copies its fort.9 back,
#             and that is empty (measured on HPCC: a 0-byte $JOB.f9).
# With NEITHER, the GUESSP record has to come back out of the deck. CRYSTAL
# does not quietly fall back to the atomic guess - it stops:
#   ERROR **** GUESSP **** COPY OF WAVEFUNCTION FILE fort.20 CAN NOT BE FOUND
# (measured, not assumed). So a deck carrying GUESSP before it has ever run
# would burn the job it is meant to speed up, and the very first step of any
# chain is exactly that case. Stripping the record from the scratch copy makes
# the deck mean "restart if there is something to restart from"; $DIR/$JOB.d12
# itself is never modified, so the next attempt still asks.
if [ -z "$RESTART_KEEPS_FORT20" ] && grep -qiE "^[[:space:]]*GUESSP" "$scratch/$JOB/INPUT" 2>/dev/null; then
  if [ -s "$DIR/$JOB.f20" ]; then
    cp "$DIR/$JOB.f20" "$scratch/$JOB/fort.20"
    echo "GUESSP: staged $JOB.f20 as fort.20"
  elif [ -s "$DIR/$JOB.f9" ]; then
    cp "$DIR/$JOB.f9" "$scratch/$JOB/fort.20"
    echo "GUESSP: restarting from this job own $JOB.f9 as fort.20"
  else
    sed -i "/^[[:space:]]*[Gg][Uu][Ee][Ss][Ss][Pp][[:space:]]*$/d" "$scratch/$JOB/INPUT"
    echo "GUESSP requested but no $JOB.f20 or $JOB.f9 on disk - dropped it from"
    echo "  this run and cold starting (CRYSTAL aborts on GUESSP with no fort.20)"
  fi
fi
cd $scratch/$JOB

I_MPI_HYDRA_BOOTSTRAP="ssh" mpirun -n $SLURM_NTASKS $EBROOTCRYSTAL/bin/Pcrystal 2>&1 >& $DIR/${JOB}.out
#srun Pcrystal 2>&1 >& $DIR/${JOB}.out
cp fort.9 ${DIR}/${JOB}.f9 


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
# NOTE: Job submission is now handled by the calling script (crystal.py)
# This allows for proper handling of --nosubmit flag
# sbatch $1.sh
