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
from pathlib import Path
from typing import Optional, Tuple

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


def job_scratch_dir(job_script_text: str, job_name: str) -> Optional[Path]:
    """The scratch directory the job script ran CRYSTAL in, or None if it can't
    be resolved here.

    Read from the script's own `export scratch=...` line (the template uses
    $SCRATCH/crys23) rather than assumed, and expanded with this process's
    environment. $SCRATCH is shared between login and compute nodes, but it
    can be EMPTY on some nodes; an unset or empty variable means "unknown",
    never a guessed path.
    """
    m = re.search(r'^\s*(?:export\s+)?scratch=(\S+)', job_script_text or '', re.M)
    if not m:
        return None
    raw = m.group(1).strip('\'"')
    for var in re.findall(r'\$\{?(\w+)\}?', raw):
        if not os.environ.get(var):
            return None
    base = os.path.expandvars(raw)
    if '$' in base or not base:
        return None
    return Path(base) / job_name


def optinfo_state(scratch_dir: Optional[Path]) -> str:
    """'present', 'absent' (the job's scratch dir is there without OPTINFO.DAT)
    or 'unknown' (the scratch dir can't be seen from here)."""
    if scratch_dir is None or not scratch_dir.is_dir():
        return 'unknown'
    return 'present' if (scratch_dir / 'OPTINFO.DAT').is_file() else 'absent'


def keep_timed_out_output(out_path: Path) -> Optional[Path]:
    """Copy the killed run's .out to <JOB>.out.timeout<N> (next free N).

    A copy, not a move: the resubmitted job overwrites <JOB>.out anyway, and
    until it starts the failed record still points at that file - moving it
    would leave a failed resubmission with "no output" to analyse on retry.
    """
    out_path = Path(out_path)
    if not out_path.is_file():
        return None
    n = 1
    while True:
        dest = out_path.with_name(f"{out_path.name}.timeout{n}")
        if not dest.exists():
            break
        n += 1
    shutil.copy2(out_path, dest)
    return dest
