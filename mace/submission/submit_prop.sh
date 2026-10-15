#!/bin/bash
# This script generates and submits SLURM job scripts for running CRYSTAL23 Pproperties calculations.
# Takes a job name as argument, creates a .sh file with SLURM directives, and submits it to the queue.

# The job name becomes the SLURM job name, the file names and the JOB variable
# of the script: one with a space or a shell character cannot be written.
case "$1" in
  ""|*[[:space:]\'\"\`\$\\\(\)\<\>\|\&\;\*\?\[\]\{\}]*)
    echo "ERROR: \"$1\" cannot be a job name (no spaces or any of ' \" \` \$ \\ ( ) < > | & ; * ? [ ] { }); rename the deck" >&2
    exit 1 ;;
esac

echo '#!/bin/bash --login' > $1.sh
echo '#SBATCH -J '$1 >> $1.sh
out=$1
file=-%J.o
outfile=$out$file
echo '#SBATCH -o '$outfile >> $1.sh
echo '#SBATCH --cpus-per-task=1' >> $1.sh
echo '#SBATCH --ntasks=8' >> $1.sh
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
# NO SUBMIT DIRECTORY. Everything below stages from and copies back to the
# submit directory, which SLURM always sets. Run with plain bash, DIR was
# empty: the job read and wrote /, still ended with exit 0 and ran the
# completion callback from the home directory.
if [ -z "$DIR" ]; then
  echo "ERROR: SLURM_SUBMIT_DIR is not set - this script stages from and copies back to the"
  echo "  submit directory. Submit it with sbatch, or run it on this node with: mace run $JOB.d3"
  exit 1
fi

echo "submit directory: "
echo "$SLURM_SUBMIT_DIR"
module purge

module load CRYSTAL/23-intel-2023a
module load Python/3.11.3-GCCcore-12.3.0
module load Python-bundle-PyPI/2023.06-GCCcore-12.3.0
' >> $1.sh
# MACE INSTALL PIN (evaluated when the job script is generated). The job runs
# under bash --login, which re-reads ~/.bashrc; a MACE_HOME/PYTHONPATH exported
# there would decide which MACE runs this job's completion callback (follow-up
# steps, recovery). The job is pinned to the install that generates it: the
# one this generator belongs to (<install>/mace/submission/), or - for a copy
# of it, such as a workflow's workflow_scripts/ template - the MACE_HOME of
# the MACE running the copy (MACE sets it to its own install).
MACE_PIN=""
MACE_GEN_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd)
if [ "${MACE_GEN_DIR##*/}" = "submission" ] && [ -f "$MACE_GEN_DIR/../queue/manager.py" ]; then
  MACE_PIN=$(cd "$MACE_GEN_DIR/../.." && pwd)
elif [ -n "$MACE_HOME" ] && [ -f "$MACE_HOME/mace/queue/manager.py" ]; then
  MACE_PIN=$(cd "$MACE_HOME" && pwd)
fi
case "$MACE_PIN" in *"'"*|*$'\n'*) MACE_PIN="" ;; esac
if [ -n "$MACE_PIN" ]; then
  echo "MACE_SUBMIT_HOME='$MACE_PIN'" >> $1.sh
  echo '# MACE INSTALL. bash --login has re-read ~/.bashrc, whose MACE_HOME and
# PYTHONPATH may name another MACE install; the completion callback below
# (follow-up steps, recovery) runs the install that generated this script.
if [ -f "$MACE_SUBMIT_HOME/mace/queue/manager.py" ]; then
  if [ -n "$MACE_HOME" ] && [ "$MACE_HOME" != "$MACE_SUBMIT_HOME" ]; then
    echo "mace: running the submitting install $MACE_SUBMIT_HOME (the login shell set MACE_HOME=$MACE_HOME)"
  fi
  export MACE_HOME="$MACE_SUBMIT_HOME"
  export PYTHONPATH="$MACE_SUBMIT_HOME${PYTHONPATH:+:$PYTHONPATH}"
else
  echo "mace: the submitting install $MACE_SUBMIT_HOME is not readable on $(hostname); the callback uses MACE_HOME=${MACE_HOME:-(unset)}"
fi' >> $1.sh
fi
echo '
# SCRATCH GUARD. $SCRATCH can be EMPTY inside a job, even under bash --login
# (seen on agx-000). scratch then became /crys23, mkdir was refused, INPUT was
# never written and CRYSTAL stopped with END OF DATA IN INPUT DECK. So the
# directory is checked before anything is staged in it, and when it cannot be
# used the first writable one of these is used instead, keeping the part of
# the path after $SCRATCH:
#   1. /mnt/scratch/$USER resolved - what /etc/profile.d/hpcc.sh sets $SCRATCH
#      to on MSU HPCC, so the same directory a healthy node uses (and the one
#      a recovery looks in for OPTINFO.DAT);
#   2. .mace_scratch under the submit directory (shared; removed after a run
#      that ended normally, see SCRATCH FALLBACK CLEANUP);
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
# ONE SCRATCH DIRECTORY PER DECK. $scratch/$JOB is named after the deck alone,
# so two decks of the same name in different folders shared it: at the same
# time they overwrote each other INPUT and fort.9, one after the other the
# second restarted from the OPTINFO.DAT or fort.20 of the first. The folder
# whose deck uses $scratch/$JOB is recorded in it (.mace_submit_dir); a deck of
# the same name from another folder runs in
# $scratch/.mace_by_dir/<checksum of its folder>/$JOB instead. A deck rerun
# from its own folder keeps its directory (OPTGEOM RESTART and GUESSP read
# OPTINFO.DAT and fort.20 there), and a directory with no record (an earlier
# MACE, or a fort.20 staged by hand) belongs to the first deck that runs in it.
MACE_DIR_REAL=$(cd "$DIR" 2>/dev/null && pwd -P)
[ -n "$MACE_DIR_REAL" ] || MACE_DIR_REAL=$DIR
( set -C; printf "%s\n" "$MACE_DIR_REAL" > "$scratch/$JOB/.mace_submit_dir" ) 2>/dev/null
MACE_OWNER=$(head -n 1 "$scratch/$JOB/.mace_submit_dir" 2>/dev/null)
if [ -n "$MACE_OWNER" ] && [ "$MACE_OWNER" != "$MACE_DIR_REAL" ]; then
  MACE_BY_DIR=/.mace_by_dir/$(printf "%s" "$MACE_DIR_REAL" | cksum | cut -d" " -f1)
  if mkdir -p "$scratch$MACE_BY_DIR/$JOB" 2>/dev/null; then
    echo "scratch: $scratch/$JOB belongs to $JOB in $MACE_OWNER; this deck runs in $scratch$MACE_BY_DIR/$JOB"
    scratch=$scratch$MACE_BY_DIR
    MACE_SCRATCH_WANTED=$MACE_SCRATCH_WANTED$MACE_BY_DIR
    MACE_SCRATCH_SUFFIX=$MACE_SCRATCH_SUFFIX$MACE_BY_DIR
    printf "%s\n" "$MACE_DIR_REAL" > "$scratch/$JOB/.mace_submit_dir" 2>/dev/null
  else
    echo "scratch: cannot create $scratch$MACE_BY_DIR/$JOB - sharing $scratch/$JOB with $JOB in $MACE_OWNER"
    MACE_SHARED=1
  fi
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
# an older run. Only from a directory of the deck in this folder (or one with
# no record): a directory that a same-named deck in another folder has
# claimed since holds the files of that deck.
MACE_SCRATCH_PREV=$(cat "$DIR/.$JOB.scratch" 2>/dev/null)
MACE_PREV_OWNER=$(head -n 1 "$MACE_SCRATCH_PREV/.mace_submit_dir" 2>/dev/null)
if [ -n "$MACE_PREV_OWNER" ] && [ "$MACE_PREV_OWNER" != "$MACE_DIR_REAL" ] \
   && [ "$MACE_SCRATCH_PREV" != "$scratch/$JOB" ]; then
  echo "scratch: not bringing restart files over from $MACE_SCRATCH_PREV - it now belongs to $JOB in $MACE_PREV_OWNER"
  MACE_SCRATCH_PREV=
fi
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
# A CLEAN DIRECTORY FOR EVERY RUN. A properties run needs nothing an earlier
# run left here (its wavefunction is staged from $JOB.f9 every time), but the
# files an earlier run left were used and copied back as this run results: a
# run whose $JOB.f9 was missing read the old fort.9, and a run that failed
# copied the old BAND.DAT back. So the directory is emptied before staging
# (its record of the folder that uses it is kept).
find "$scratch/$JOB" -mindepth 1 -maxdepth 1 ! -name .mace_submit_dir -exec rm -rf {} + 2>/dev/null
mkdir -p "$scratch/$JOB"
# The claim of this folder, written again every run (a purged scratch or a lost
# record), unless the directory is shared with the deck that owns it.
[ -n "$MACE_SHARED" ] || printf "%s\n" "$MACE_DIR_REAL" > "$scratch/$JOB/.mace_submit_dir" 2>/dev/null
# THE WAVEFUNCTION. Pproperties reads fort.9, the $JOB.f9 of the SCF run copied
# next to this deck. Without it (or with an empty one: an aborted run copies
# an empty fort.9 back) it has nothing to compute from: the job stops here
# with the reason in $JOB.out, which holds only this error (an earlier output
# is kept as $JOB.out.prev<N>).
MACE_F9_STATE=""
if [ ! -e "$DIR/$JOB.f9" ]; then MACE_F9_STATE=missing; elif [ ! -s "$DIR/$JOB.f9" ]; then MACE_F9_STATE=empty; fi
if [ -n "$MACE_F9_STATE" ]; then
  if [ -f "$DIR/$JOB.out" ]; then
    MACE_N=1
    while [ -e "$DIR/$JOB.out.prev$MACE_N" ]; do MACE_N=$((MACE_N+1)); done
    mv "$DIR/$JOB.out" "$DIR/$JOB.out.prev$MACE_N" 2>/dev/null
  fi
  echo "ERROR: $JOB.f9 is $MACE_F9_STATE in $DIR - a properties run reads the wavefunction of its SCF run" | tee "$DIR/$JOB.out"
  echo "  from it. Copy that run .f9 next to $JOB.d3 as $JOB.f9 and submit again. Not running Pproperties." | tee -a "$DIR/$JOB.out"
  exit 1
fi
if ! cp "$DIR/$JOB.d3" "$scratch/$JOB/INPUT" || ! cp "$DIR/$JOB.f9" "$scratch/$JOB/fort.9"; then
  echo "ERROR: could not stage $JOB.d3 and $JOB.f9 in $scratch/$JOB. Not running Pproperties." | tee "$DIR/$JOB.out"
  exit 1
fi
cd "$scratch/$JOB" || { echo "ERROR: cannot enter $scratch/$JOB. Not running CRYSTAL."; exit 1; }

I_MPI_HYDRA_BOOTSTRAP="ssh" mpirun -n "$SLURM_NTASKS" "$EBROOTCRYSTAL/bin/Pproperties" 2>&1 >& "$DIR/${JOB}.out"
#srun $EBROOTCRYSTAL/bin/Pproperties 2>&1 >& "$DIR/${JOB}.out"
#srun Pproperties 2>&1 >& "$DIR/${JOB}.out"

# COPY BACK what this run wrote: the directory was emptied before staging, so
# every file here is this run own (fort.9 is the $JOB.f9 it staged, or the
# one Pproperties rewrote).
[ -f fort.9 ] && cp fort.9 "$DIR/$JOB.f9"
[ -f BAND.DAT ] && cp BAND.DAT "$DIR/$JOB.BAND.DAT"
[ -f fort.25 ] && cp fort.25 "$DIR/$JOB.f25"
[ -f DOSS.DAT ] && cp DOSS.DAT "$DIR/$JOB.DOSS.DAT"
[ -f POTC.DAT ] && cp POTC.DAT "$DIR/$JOB.POTC.DAT"
[ -f SIGMA.DAT ] && cp SIGMA.DAT "$DIR/$JOB.SIGMA.DAT"
[ -f SEEBECK.DAT ] && cp SEEBECK.DAT "$DIR/$JOB.SEEBECK.DAT"
[ -f SIGMAS.DAT ] && cp SIGMAS.DAT "$DIR/$JOB.SIGMAS.DAT"
[ -f KAPPA.DAT ] && cp KAPPA.DAT "$DIR/$JOB.KAPPA.DAT"
[ -f TDF.DAT ] && cp TDF.DAT "$DIR/$JOB.TDF.DAT"
[ -f DENS_CUBE.DAT ] && cp DENS_CUBE.DAT "$DIR/${JOB}_DENS.CUBE"
[ -f POT_CUBE.DAT ] && cp POT_CUBE.DAT "$DIR/${JOB}_POT.CUBE"
[ -f SPIN_CUBE.DAT ] && cp SPIN_CUBE.DAT "$DIR/${JOB}_SPIN.CUBE"

# ADDED: Auto-submit new jobs when this one completes
# Check multiple possible locations for queue managers
cd "$DIR" || echo "WARNING: cannot enter $DIR; the steps below use full paths"

# SCRATCH FALLBACK CLEANUP. When the scratch guard had to fall back, the
# output says where CRYSTAL ran. The submit-directory fallback (.mace_scratch)
# is on the shared file system and counts against the quota, so this job
# directory there is removed once the run reached its normal end and its
# files were copied back. A run that did not end normally keeps it, as a
# restart of the optimization (OPTGEOM RESTART) reads OPTINFO.DAT and fort.20
# there, and the queue manager reads fort.87 for the reason the run stopped.
if [ "$scratch" != "$MACE_SCRATCH_WANTED" ]; then
  MACE_NOTE="scratch: CRYSTAL ran in $scratch/$JOB instead of $MACE_SCRATCH_WANTED/$JOB (on $(hostname))"
  if [ "$scratch" = "$DIR/.mace_scratch$MACE_SCRATCH_SUFFIX" ]; then
    if grep -q "EEEEEEEEEE TERMINATION" "$DIR/$JOB.out" 2>/dev/null; then
      rm -rf "$scratch/$JOB"
      MACE_D=$scratch
      while [ "$MACE_D" != "$DIR" ] && rmdir "$MACE_D" 2>/dev/null; do MACE_D=$(dirname "$MACE_D"); done
      MACE_NOTE="$MACE_NOTE; removed it after the run"
    else
      MACE_NOTE="$MACE_NOTE; kept it, as the run did not reach its normal end (remove it by hand once no restart needs it)"
    fi
  fi
  echo "$MACE_NOTE"
  echo "$MACE_NOTE" >> "$DIR/$JOB.out"
fi
# END SCRATCH FALLBACK CLEANUP

# LOCAL RUN. mace run runs this script on the node it is on, without SLURM,
# and records the result itself. No completion callback: it would submit the
# follow-up jobs from that node. Never inside a SLURM job (sbatch exports the
# environment, and a MACE_LOCAL_RUN in it must not cost a job its callback).
if [ -n "$MACE_LOCAL_RUN" ] && [ -z "$SLURM_JOB_ID" ]; then
  echo "mace run: no completion callback (local run on $(hostname))"
  exit 0
fi

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
    if [ -f "$DIR/mace/queue/manager.py" ]; then
        QUEUE_MANAGER="$DIR/mace/queue/manager.py"
    elif [ -f "$DIR/../../../../mace/queue/manager.py" ]; then
        QUEUE_MANAGER="$DIR/../../../../mace/queue/manager.py"
    elif [ -f "$DIR/../../../../../mace/queue/manager.py" ]; then
        QUEUE_MANAGER="$DIR/../../../../../mace/queue/manager.py"
    elif [ -f "$DIR/enhanced_queue_manager.py" ]; then
        QUEUE_MANAGER="$DIR/enhanced_queue_manager.py"
    elif [ -f "$DIR/../../../../enhanced_queue_manager.py" ]; then
        QUEUE_MANAGER="$DIR/../../../../enhanced_queue_manager.py"
    elif [ -f "$DIR/crystal_queue_manager.py" ]; then
        QUEUE_MANAGER="$DIR/crystal_queue_manager.py"
    elif [ -f "$DIR/../../../../crystal_queue_manager.py" ]; then
        QUEUE_MANAGER="$DIR/../../../../crystal_queue_manager.py"
    fi
fi

if [ ! -z "$QUEUE_MANAGER" ]; then
    echo "Found queue manager at: $QUEUE_MANAGER"
    # Check for workflow context database
    if [ ! -z "$MACE_CONTEXT_DIR" ] && [ -f "$MACE_CONTEXT_DIR/materials.db" ]; then
        echo "Using workflow context database: $MACE_CONTEXT_DIR/materials.db"
        python "$QUEUE_MANAGER" --max-jobs 950 --reserve 50 --max-submit 10 --callback-mode completion --max-recovery-attempts 3 --db-path "$MACE_CONTEXT_DIR/materials.db"
    elif [ -n "$MACE_DB_PATH" ] && [ -f "$MACE_DB_PATH" ]; then
        # the database the queue manager recorded this job in (set when the
        # job directory is not where that database is, e.g. --organize)
        echo "Using the database this job was recorded in: $MACE_DB_PATH"
        python "$QUEUE_MANAGER" --max-jobs 950 --reserve 50 --max-submit 10 --callback-mode completion --max-recovery-attempts 3 --db-path "$MACE_DB_PATH"
    else
        python "$QUEUE_MANAGER" --max-jobs 950 --reserve 50 --max-submit 10 --callback-mode completion --max-recovery-attempts 3
    fi
else
    echo "Warning: Queue manager not found. Checked:"
    echo "  - \$MACE_HOME/mace/queue/manager.py"
    echo "  - Various relative paths from $DIR"
    echo "  Workflow progression may not continue automatically"
fi' >> $1.sh
# MACE MARKER (evaluated when the job script is generated): the cksum of the
# job script written above. mace submit writes a script whose marker still
# matches again when resource options are given, and never replaces one that
# was edited (unless --overwrite-sh).
if command -v cksum >/dev/null 2>&1; then
  echo "# MACE-generated; cksum of the lines above: $(cksum < "$1.sh" | cut -d" " -f1-2)" >> "$1.sh"
fi
# NOTE: Job submission is now handled by the calling script (properties.py)
# This allows for proper handling of --nosubmit flag
# sbatch $1.sh
