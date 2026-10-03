"""Continue a walltime-killed geometry optimization instead of starting over.

CRYSTAL23 (manual 7.4.3): adding RESTART to the OPTGEOM block makes the next
run read the previous optimization steps from OPTINFO.DAT, with the SCF guess
taken from fort.20 - the density matrix fort.9 held at the end of the SCF of
the last successful step. The rest of the deck must be the same one the
killed run used, so the only edit here is that one record.

OPTINFO.DAT is written after each optimization cycle, so a job killed while
still inside its FIRST SCF has nothing to restart from; that job is simply
resubmitted as before.

What the output shows (checked against the real OPT corpus):
  * every optimization point is announced as
      "<TYPE> OPTIMIZATION - POINT    N"
    (e.g. "CELL OPTIMIZATION", "COORDINATE AND CELL OPTIMIZATION"), and point
    N+1 is only printed after cycle N finished and OPTINFO.DAT was updated;
  * the first completed cycle also prints
      "GEOMETRY OPTIMIZATION INFORMATION STORED IN OPTINFO.DAT";
  * a run that itself restarted prints
      "RESTARTING FROM A PREVIOUS GEOMETRY OPTIMIZATION RUN"
    and starts numbering at the point it resumed from (a restarted corpus run
    begins at POINT 30 and never prints the OPTINFO line), so its history lives
    in OPTINFO.DAT even if it died before finishing a cycle of its own.
"""

import os
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_POINT_RE = re.compile(r'OPTIMIZATION - POINT\s+(\d+)')
_OPTINFO_STORED = 'GEOMETRY OPTIMIZATION INFORMATION STORED IN OPTINFO.DAT'
_RESTARTED_RUN = 'RESTARTING FROM A PREVIOUS GEOMETRY OPTIMIZATION RUN'

# OPTGEOM's optimization-type records. RESTART goes right after one of these
# when the deck has it, which is where the restarted corpus deck carries it.
_OPT_TYPE_KEYWORDS = {'FULLOPTG', 'CELLONLY', 'ATOMONLY', 'ITATOCEL',
                      'INTREDUN', 'CVOLOPT'}
_OPTGEOM_ENDS = {'END', 'ENDOPT', 'ENDOPTGEOM'}


def optimization_has_progress(out_text: str) -> bool:
    """True when the killed run left optimization history to continue from.

    Either a cycle completed in this run (a second point was reached, or the
    OPTINFO line was printed), or this run was itself a restart and so began
    from an OPTINFO.DAT that is still there.
    """
    if not out_text:
        return False
    if _OPTINFO_STORED in out_text or _RESTARTED_RUN in out_text:
        return True
    points = [int(n) for n in _POINT_RE.findall(out_text)]
    return bool(points) and max(points) >= 2


def _optgeom_span(lines):
    """(start, end) line indices of the first OPTGEOM block, end exclusive of
    the terminator, or None when the deck has no OPTGEOM. Line 0 is the
    deck's title, never a keyword, so it is not considered."""
    start = None
    for i, line in enumerate(lines):
        if i == 0:
            continue
        key = line.strip().upper()
        if start is None:
            if key == 'OPTGEOM':
                start = i
        elif key in _OPTGEOM_ENDS:
            return start, i
    if start is None:
        return None
    return start, len(lines)


def has_optgeom(d12_text: str) -> bool:
    """Whether the deck is a geometry optimization (has an OPTGEOM block)."""
    return _optgeom_span(d12_text.splitlines()) is not None


def has_optgeom_restart(d12_text: str) -> bool:
    """Whether RESTART is already a record of the deck's OPTGEOM block."""
    lines = d12_text.splitlines()
    span = _optgeom_span(lines)
    if span is None:
        return False
    return any(l.strip().upper() == 'RESTART' for l in lines[span[0] + 1:span[1]])


def add_optgeom_restart(d12_text: str) -> Tuple[str, bool]:
    """Insert RESTART into the OPTGEOM block. Returns (text, changed).

    Idempotent: a deck already carrying it (a restart that timed out again) is
    returned untouched. Only the one line is added; every other record and the
    file's line endings stay as they were.
    """
    lines = d12_text.splitlines(keepends=True)
    span = _optgeom_span([l.rstrip('\r\n') for l in lines])
    if span is None or has_optgeom_restart(d12_text):
        return d12_text, False
    start, end = span
    at = start + 1
    if at < end and lines[at].strip().upper() in _OPT_TYPE_KEYWORDS:
        at += 1
    newline = '\r\n' if lines[start].endswith('\r\n') else '\n'
    lines.insert(at, 'RESTART' + newline)
    return ''.join(lines), True


def job_name(job_script_text: str) -> Optional[str]:
    """$JOB as the job script sets it (`export JOB=<name>`), or None."""
    m = re.search(r'^\s*(?:export\s+)?JOB=([^\s;#]+)', job_script_text or '', re.M)
    return m.group(1).strip('\'"') if m else None


#: MSU HPCC sets $SCRATCH to this directory, resolved, in /etc/profile.d/hpcc.sh
#: (`export SCRATCH=$(readlink -f /mnt/scratch/$USER)`). The job script's
#: scratch guard falls back to it when $SCRATCH is empty, and so does this.
HPCC_SCRATCH_ROOT = Path('/mnt/scratch')


def default_scratch() -> Optional[str]:
    """$SCRATCH as MSU HPCC would set it, for a process where it is empty:
    /mnt/scratch/$USER resolved, when that exists here. None otherwise."""
    user = os.environ.get('USER')
    if not user:
        try:
            import getpass
            user = getpass.getuser()
        except Exception:
            return None
    root = HPCC_SCRATCH_ROOT / user
    return str(root.resolve()) if root.is_dir() else None


def scratch_record(submit_dir, job_name: str) -> Path:
    """Where the job script records the scratch directory it ran in."""
    return Path(submit_dir) / f".{job_name}.scratch"


def job_scratch_dir(job_script_text: str, job_name: str,
                    submit_dir=None) -> Optional[Path]:
    """The scratch directory the job script ran CRYSTAL in, or None if it can't
    be resolved here.

    The job script's scratch guard writes the directory it actually used to
    <submit dir>/.<JOB>.scratch, which covers its fallbacks (a directory under
    the submit directory, $TMPDIR) - so that record comes first when the
    submit directory is known. Otherwise it is read from the script's own
    `export scratch=...` line (the template uses $SCRATCH/crys23) and expanded
    with this process's environment. $SCRATCH can be EMPTY on some nodes; it
    is then rebuilt the way the job script's guard does it (/mnt/scratch/$USER
    resolved, when it exists here). Any other unset or empty variable means
    "unknown", never a guessed path.
    """
    if submit_dir is not None:
        try:
            recorded = scratch_record(submit_dir, job_name).read_text().strip()
        except OSError:
            recorded = ''
        if recorded:
            return Path(recorded)
    m = re.search(r'^\s*(?:export\s+)?scratch=(\S+)', job_script_text or '', re.M)
    if not m:
        return None
    raw = m.group(1).strip('\'"')
    values = {}
    for var in re.findall(r'\$\{?(\w+)\}?', raw):
        value = os.environ.get(var) or (default_scratch() if var == 'SCRATCH' else None)
        if not value:
            return None
        values[var] = value
    base = re.sub(r'\$\{?(\w+)\}?', lambda v: values[v.group(1)], raw)
    if '$' in base or not base:
        return None
    return Path(base) / job_name


def optinfo_state(scratch_dir: Optional[Path]) -> str:
    """'present', 'absent' or 'unknown' (the scratch area can't be seen from
    here).

    A missing job directory is 'absent' when the scratch area it would live in
    is visible ($SCRATCH/crys23 or $SCRATCH itself exists): the directory is
    made by every run, so a killed run whose directory is gone - purged, or
    never created - has no OPTINFO.DAT to restart from.
    """
    if scratch_dir is None:
        return 'unknown'
    if scratch_dir.is_dir():
        return 'present' if (scratch_dir / 'OPTINFO.DAT').is_file() else 'absent'
    if scratch_dir.parent.is_dir() or scratch_dir.parent.parent.is_dir():
        return 'absent'
    return 'unknown'


def keep_output(out_path: Path, reason: str) -> Optional[Path]:
    """Copy a failed run's .out to <JOB>.out.<reason><N> (next free N).

    A copy, not a move: the resubmitted job overwrites <JOB>.out anyway, and
    until it starts the failed record still points at that file - moving it
    would leave a failed resubmission with "no output" to analyse on retry.
    """
    out_path = Path(out_path)
    if not out_path.is_file():
        return None
    n = 1
    while True:
        dest = out_path.with_name(f"{out_path.name}.{reason}{n}")
        if not dest.exists():
            break
        n += 1
    shutil.copy2(out_path, dest)
    return dest


def keep_timed_out_output(out_path: Path) -> Optional[Path]:
    """Copy the killed run's .out to <JOB>.out.timeout<N> (next free N)."""
    return keep_output(out_path, 'timeout')


def kept_outputs(out_path: Path) -> list:
    """Earlier runs of this job kept by keep_output, oldest name first."""
    out_path = Path(out_path)
    found = []
    for reason in (TIMEOUT_TAG, ABORT_TAG, LOOP_TAG):
        found.extend(sorted(out_path.parent.glob(f"{out_path.name}.{reason}[0-9]*"),
                            key=lambda p: int(re.sub(r'\D', '', p.name[len(out_path.name):]) or 0)))
    return found


def remove_optgeom_restart(d12_text: str) -> Tuple[str, bool]:
    """Take RESTART out of the OPTGEOM block (only there - FREQCALC has a
    RESTART of its own). Returns (text, changed)."""
    lines = d12_text.splitlines(keepends=True)
    span = _optgeom_span([l.rstrip('\r\n') for l in lines])
    if span is None:
        return d12_text, False
    start, end = span
    kept = [l for k, l in enumerate(lines)
            if not (start < k < end and l.strip().upper() == 'RESTART')]
    if len(kept) == len(lines):
        return d12_text, False
    return ''.join(kept), True


def backup_deck(input_file: Path, text: Optional[str] = None) -> Path:
    """Keep the deck as it is now before its geometry is rewritten:
    <JOB>.d12.orig the first time, then <JOB>.d12.orig2, .orig3 ... An
    existing backup is never overwritten, so <JOB>.d12.orig always holds the
    geometry the job started from. (None of these end in .d12, so nothing
    picks them up as a new job.) `text`: what to keep, when the deck has
    already been changed."""
    input_file = Path(input_file)
    n = 1
    while True:
        dest = input_file.with_name(input_file.name + ('.orig' if n == 1 else f'.orig{n}'))
        try:
            with open(dest, 'x') as f:
                f.write(input_file.read_text() if text is None else text)
            shutil.copystat(input_file, dest)
            return dest
        except FileExistsError:
            n += 1


# ------------------------------------------------ how the run ended

TIMEOUT_TAG = 'timeout'
ABORT_TAG = 'optabort'
LOOP_TAG = 'finalrunloop'

# A run CRYSTAL completes prints this line (702 of the 707 corpus outputs; the
# other five were MPI_Abort'ed on an error, SIGKILLed, or still running).
_NORMAL_END = 'EEEEEEEEEE TERMINATION'
_TRUST_ZERO_RE = re.compile(r'UPDATED TRUST RADIUS\s+0\.0+E\+00')
_TOO_SMALL_TRUST = 'TOO SMALL TRUST RADIUS'
_PXK_TOO_SMALL = 'PXK TOO SMALL'


#: What the job script's scratch guard writes to <JOB>.out when it finds no
#: writable scratch directory and stops before CRYSTAL runs.
SCRATCH_GUARD_ERROR = 'ERROR: no writable scratch directory'


def scratch_guard_stop(out_text: str) -> Optional[str]:
    """The guard's error line when this output is a job the scratch guard
    stopped before CRYSTAL ran, else None."""
    for line in (out_text or '').splitlines():
        if SCRATCH_GUARD_ERROR in line:
            return line.strip()
    return None


def ended_normally(out_text: str) -> bool:
    return _NORMAL_END in (out_text or '')


def scratch_error_text(scratch_dir: Optional[Path]) -> Optional[str]:
    """CRYSTAL's error message from fort.87 in the job's scratch directory,
    when it belongs to the run that just ended.

    A process that stops on an error writes it to fort.87 in its working
    directory; when that process is not the one printing the .out, the .out
    only shows the MPI_Abort (measured on HPCC: "ERROR **** BFGS_ **** PXK
    TOO SMALL" only in fort.87). The scratch directory is reused by the next
    run of the same job, so fort.87 counts only when it is not older than
    INPUT, which the job script copies in at the start of every run.
    """
    if scratch_dir is None:
        return None
    f87, inp = Path(scratch_dir) / 'fort.87', Path(scratch_dir) / 'INPUT'
    try:
        if not f87.is_file() or not inp.is_file():
            return None
        if f87.stat().st_mtime < inp.stat().st_mtime:
            return None
        text = f87.read_text(errors='ignore').strip()
    except OSError:
        return None
    return text or None


def trust_radius_abort(out_text: str, fort87_text: Optional[str] = None) -> Optional[str]:
    """Why this run counts as an optimization that aborted because its step
    size (trust radius) collapsed, or None.

    Measured on HPCC: a RESTART rerun re-evaluates the lowest-energy point,
    finds (almost) no energy change, and the trust radius drops to zero:
        UPDATED TRUST RADIUS     0.000E+00
        INFORMATION **** OPTGEN **** TOO SMALL TRUST RADIUS - ...
        Abort(1) on node 1 ... MPI_Abort(MPI_COMM_WORLD, 1)
    with "ERROR **** BFGS_ **** PXK TOO SMALL" only in scratch fort.87 and
    SLURM reporting COMPLETED. The INFORMATION line alone is harmless - it is
    printed in converged corpus runs too - so it counts only when the run
    then died without CRYSTAL's normal end and without an error of its own
    in the .out.
    """
    if not out_text or ended_normally(out_text):
        return None
    restarted = _RESTARTED_RUN in out_text
    after = ' in a RESTART run' if restarted else ''
    if fort87_text and _PXK_TOO_SMALL in fort87_text.upper():
        line = next((l.strip() for l in fort87_text.splitlines() if _PXK_TOO_SMALL in l.upper()),
                    _PXK_TOO_SMALL)
        return f"optimization step collapsed{after} (fort.87: {line})"
    tail = out_text[max(m.start() for m in _POINT_RE.finditer(out_text)):] \
        if _POINT_RE.search(out_text) else ''
    if 'ERROR ****' in tail or 'Abort(' not in tail:
        return None
    if _TRUST_ZERO_RE.search(tail) or _TOO_SMALL_TRUST in tail:
        return f"optimization step collapsed{after} (trust radius 0, then MPI_Abort)"
    return None


# ------------------------------------------- the FINALRUN re-start loop

_FINALRUN_RESTART = 'FINALRUN OPTION: RE-STARTING OPTIMIZATION'
_DE_RE = re.compile(r'TOTAL ENERGY\([^)]*\)\(AU\)\(\s*\d+\)\s*-?\d+\.\d+E[+-]\d+'
                    r'\s+DE \(AU\)\s+(-?\d+\.\d+E[+-]\d+)')
_OPT_CONVERGED = 'OPT END - CONVERGED'
#: Re-starts in a row whose energy jumps alternate before it counts as a loop.
FINALRUN_LOOP_MIN_RESTARTS = 3
#: Two jumps are "of similar magnitude" within this factor of each other.
_SIMILAR_JUMPS = 2.0


def finalrun_restart_jumps(out_text: str) -> List[float]:
    """The energy change (DE) of the first point after each FINALRUN
    re-start, in order; a re-start with no point after it yet is left out."""
    text = out_text or ''
    starts = [m.end() for m in re.finditer(re.escape(_FINALRUN_RESTART), text)]
    jumps = []
    for k, at in enumerate(starts):
        stop = starts[k + 1] if k + 1 < len(starts) else len(text)
        m = _DE_RE.search(text, at, stop)
        if m:
            jumps.append(float(m.group(1)))
    return jumps


def finalrun_loop(out_text: str) -> Optional[str]:
    """Why this optimization counts as caught in a FINALRUN re-start loop,
    or None.

    Measured on CRYSTAL23 (Si, B3LYP-D3, TOLINTEG 7 7 7 7 14): the
    optimization converges, FINALRUN 4 (the default) re-starts it and
    changes the reference geometry, the energy jumps by about 2.2E-4 Ha and
    the run converges to a second state; the next re-start jumps back by the
    same amount, and so on until MAXCYCLE or the walltime ends the job
    (FINALRUN 3 loops too). The integral screening fixed at each reference
    geometry is what moves: TOLINTEG 8 8 8 9 24 converges in four points.

    It is a loop when the last FINALRUN_LOOP_MIN_RESTARTS or more re-starts
    jumped alternately down and up by similar amounts - whether the run then
    reached MAXCYCLE, was killed, or is still running. A run that converged
    after all is not.
    """
    if not out_text or _OPT_CONVERGED in out_text:
        return None
    jumps = finalrun_restart_jumps(out_text)
    if len(jumps) < FINALRUN_LOOP_MIN_RESTARTS:
        return None
    streak = 1
    for a, b in zip(reversed(jumps[:-1]), reversed(jumps[1:])):
        small, large = sorted((abs(a), abs(b)))
        if a * b < 0 and large <= _SIMILAR_JUMPS * small:
            streak += 1
        else:
            break
    if streak < FINALRUN_LOOP_MIN_RESTARTS:
        return None
    shown = ', '.join(f'{d:+.2E}' for d in jumps[-min(streak, 4):])
    return (f"{len(jumps)} FINALRUN re-starts, the last {streak} jumping alternately "
            f"by {shown}{' ...' if streak > 4 else ''} Ha: the optimization oscillates between two reference geometries")


# ----------------------------------------------- TOLINTEG one level up

#: CRYSTAL's TOLINTEG when the deck sets none (CRYSTAL23 manual).
CRYSTAL_DEFAULT_TOLINTEG = (6, 6, 6, 6, 12)


def _scf_tolinteg_levels() -> List[Tuple[str, Tuple[int, ...]]]:
    """(name, values) of opt2d12's SCF tolerance presets, loosest first."""
    root = Path(__file__).resolve().parents[2] / 'Crystal_d12'
    if root.is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from d12_constants import SCF_TOLERANCE_PRESETS
    return [(p['name'], tuple(int(v) for v in p['TOLINTEG'].split()))
            for _, p in sorted(SCF_TOLERANCE_PRESETS.items())]


def raise_tolinteg(d12_text: str) -> Tuple[str, Optional[str], Optional[str]]:
    """TOLINTEG raised one preset level (Standard 7 7 7 7 14 -> Tight
    8 8 8 9 24 -> Very tight 9 9 9 11 38). Returns (text, old, new); old is
    None when the deck sets no TOLINTEG (CRYSTAL's default 6 6 6 6 12 is
    taken), new is None when there is no higher level.

    The deck's level is the highest preset all its values reach; the next
    preset is laid over it value by value, so no value ever goes down
    (7 7 7 7 30 -> 8 8 8 9 30). Only that record changes; a deck without one
    gets it just before its final END.
    """
    lines = d12_text.splitlines(keepends=True)
    at = next((i for i, l in enumerate(lines) if l.strip().upper() == 'TOLINTEG'), None)
    current = CRYSTAL_DEFAULT_TOLINTEG
    old = None
    if at is not None:
        try:
            values = tuple(int(v) for v in lines[at + 1].split()[:5])
        except (IndexError, ValueError):
            return d12_text, None, None
        if len(values) != 5:
            return d12_text, None, None
        current, old = values, ' '.join(map(str, values))
    levels = [v for _, v in _scf_tolinteg_levels()]
    reached = [k for k, v in enumerate(levels) if all(c >= p for c, p in zip(current, v))]
    nxt = (max(reached) + 1) if reached else 0
    if nxt >= len(levels):
        return d12_text, old, None
    raised = ' '.join(str(max(c, p)) for c, p in zip(current, levels[nxt]))
    if at is not None:
        eol = lines[at + 1][len(lines[at + 1].rstrip('\r\n')):] or '\n'
        lines[at + 1] = raised + eol
        return ''.join(lines), old, raised
    end = next((i for i in range(len(lines) - 1, -1, -1) if lines[i].strip().upper() == 'END'),
               None)
    if end is None:
        return d12_text, old, None
    eol = lines[end][len(lines[end].rstrip('\r\n')):] or '\n'
    lines[end:end] = ['TOLINTEG' + eol, raised + eol]
    return ''.join(lines), old, raised


# ------------------------------------ the job script's RESTART staging

_TEMPLATE = Path(__file__).resolve().parent.parent / 'submission' / 'submitcrystal23.sh'
_STAGING_START = '# OPTGEOM RESTART'
_GUESSP_START = '# GUESSP restart'
_INPUT_COPY_RE = re.compile(r'^cp \$DIR/\$JOB\.d12\s+\$scratch/\$JOB/INPUT[ \t]*$', re.M)
_OLD_GUESSP_IF = 'if grep -qiE "^[[:space:]]*GUESSP" "$scratch/$JOB/INPUT" 2>/dev/null; then'
_NEW_GUESSP_IF = 'if [ -z "$RESTART_KEEPS_FORT20" ] && ' + _OLD_GUESSP_IF[3:]


def restart_staging_block(template_text: Optional[str] = None) -> Optional[str]:
    """The RESTART staging block as the current template writes it into every
    job script (it sits inside the template's single-quoted echo, so the
    generated script carries it byte for byte)."""
    try:
        text = template_text if template_text is not None else _TEMPLATE.read_text()
    except OSError:
        return None
    start = text.find(_STAGING_START)
    end = text.find(_GUESSP_START, start)
    if start < 0 or end < 0:
        return None
    return text[start:end]


_TWOCOMPON_GUESS_START = '#\n# Inside a TWOCOMPON block'
_TWOCOMPON_ELIF = 'el' + _NEW_GUESSP_IF


def twocompon_guess_branch(template_text: Optional[str] = None) -> Optional[str]:
    """The GUESSP-inside-TWOCOMPON branch of the template's GUESSP staging,
    from its comment to the "el" that turns the one-component branch's `if`
    into its `elif` - what a script generated before it (1.1.5) lacks."""
    try:
        text = template_text if template_text is not None else _TEMPLATE.read_text()
    except OSError:
        return None
    start = text.find(_TWOCOMPON_GUESS_START)
    end = text.find(_TWOCOMPON_ELIF, start)
    if start < 0 or end < 0:
        return None
    return text[start:end + 2]


def _with_twocompon_guess(text: str, template_text: Optional[str]) -> Tuple[str, bool]:
    """Insert the TWOCOMPON guess branch before a one-component GUESSP
    staging of the current form that has none. Without it a two-component
    deck's GUESSP staged the job's own (two-component) .f9, and CRYSTAL
    stopped with "io error Read_real_1d"."""
    if _TWOCOMPON_GUESS_START in text:
        return text, False
    code = [l.strip() for l in text.splitlines()]
    if code.count(_NEW_GUESSP_IF) != 1:
        return text, False
    branch = twocompon_guess_branch(template_text)
    if branch is None:
        return text, False
    at = text.index(_NEW_GUESSP_IF)
    return text[:at] + branch + text[at:], True


def refresh_restart_staging(script_text: str, template_text: Optional[str] = None) -> Tuple[str, str]:
    """Give a job script written before the current RESTART staging that
    staging. Returns (text, what happened).

    Such a script (a workflow copies the generator into workflow_scripts/
    when it is planned, so a workflow planned earlier keeps writing them)
    would run a RESTART deck without checking for OPTINFO.DAT,
    and its GUESSP staging could overwrite the killed run's fort.20. The block
    is inserted right after the line that copies the deck to INPUT - the
    point where the current template has it - and an existing GUESSP staging
    of the current form is told to stand aside. Only when that is all the
    script does with fort.20: any other fort.20 handling is left alone and the
    script is not changed. A script that has a RESTART staging gets the
    template's current one in its place. Either way a GUESSP staging without
    the TWOCOMPON branch (a two-component deck needs a one-component matrix,
    not its own .f9) gets it.
    """
    block = restart_staging_block(template_text)
    if _STAGING_START in script_text:
        text = script_text
        start = text.find(_STAGING_START)
        end = text.find(_GUESSP_START, start)
        if block is not None and end > start and text[start:end] != block:
            text = text[:start] + block + text[end:]
        text, branch = _with_twocompon_guess(text, template_text)
        if text == script_text:
            return script_text, 'already has the RESTART staging'
        return text, ('replaced the RESTART staging with the current template\'s'
                      + (' and added its TWOCOMPON guess branch' if branch else ''))
    if block is None:
        return script_text, 'template staging block not found; script unchanged'
    copies = list(_INPUT_COPY_RE.finditer(script_text))
    if len(copies) != 1:
        return script_text, 'no single "cp $DIR/$JOB.d12 $scratch/$JOB/INPUT" line; script unchanged'
    code = [l for l in script_text.splitlines() if l.strip() and not l.lstrip().startswith('#')]
    guessp_ifs = sum(l.strip() == _OLD_GUESSP_IF for l in code)
    fort20 = [l for l in code if 'fort.20' in l]
    if fort20 and guessp_ifs != 1:
        return script_text, 'unrecognised fort.20 handling in the script; script unchanged'
    at = copies[0].end()                  # end of the copy line, before its newline
    text = script_text[:at] + '\n' + block.rstrip('\n') + script_text[at:]
    if guessp_ifs == 1:
        text = text.replace(_OLD_GUESSP_IF, _NEW_GUESSP_IF, 1)
        text, _ = _with_twocompon_guess(text, template_text)
    return text, 'added the RESTART staging of the current template'
