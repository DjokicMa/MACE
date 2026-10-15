"""Carry a job script's SLURM resources over to a freshly generated one.

When error recovery fixes a deck (SCF convergence, SHRINK, ...) it writes a
new ``*_recovery_*`` deck and the queue manager generates that deck's job
script from the template, which knows nothing of what the user set in the
original ``<job>.sh`` (``--ntasks``, ``--mem``, ``-t``, ``-p``, ``--exclude``,
``-A``, ...). carry_resources() puts the original's resource directives into
the new script, keeping the new script's own job name, output file and body.
"""
import re
from typing import Dict, List, Optional, Tuple

from mace.submission.job_script_age import MARKER_PREFIX, posix_cksum

# option spelling -> resource it sets; job name, output and error files are
# the new deck's own and never carried
_SHORT = {"n": "ntasks", "t": "time", "p": "partition", "x": "exclude", "A": "account",
          "N": "nodes", "c": "cpus-per-task", "C": "constraint", "q": "qos", "G": "gpus"}
_CARRIED = {"ntasks", "ntasks-per-node", "time", "partition", "exclude", "account", "nodes",
            "cpus-per-task", "constraint", "qos", "gpus", "gres", "mem", "mem-per-cpu"}
_SBATCH = re.compile(r"^#SBATCH\s+(?:--([A-Za-z][\w-]*)|-([A-Za-z])(?![A-Za-z_-]))")


def _resource(line: str) -> Optional[str]:
    """The resource an ``#SBATCH`` line sets ('mem' for both memory forms,
    which exclude each other), or None for any other line."""
    m = _SBATCH.match(line)
    if not m:
        return None
    name = m.group(1) or _SHORT.get(m.group(2))
    if name not in _CARRIED:
        return None
    return "mem" if name in ("mem", "mem-per-cpu") else name


def _resources(text: str) -> Dict[str, List[str]]:
    found: Dict[str, List[str]] = {}
    for line in text.splitlines():
        key = _resource(line)
        if key:
            found.setdefault(key, []).append(line)
    return found


def parseable(text: str) -> bool:
    """Whether a job script is a SLURM script MACE's generators wrote (a
    ``#SBATCH -J`` line and resource directives), edited or not."""
    return bool(re.search(r"(?m)^#SBATCH\s+(-J|--job-name)\b", text)) and bool(_resources(text))


def _restamp(text: str) -> str:
    """Rewrite the trailing MACE-generated marker for the changed lines above it."""
    lines = text.splitlines(keepends=True)
    if not lines or not lines[-1].startswith(MARKER_PREFIX):
        return text
    body = "".join(lines[:-1]).encode()
    return "".join(lines[:-1]) + f"{MARKER_PREFIX}{posix_cksum(body)} {len(body)}\n"


def carry_resources(original: str, generated: str) -> Tuple[Optional[str], List[str]]:
    """(new script text, resources carried) with ``original``'s resource
    directives in place of ``generated``'s; (None, []) when ``original`` is
    not a script MACE can read."""
    if not parseable(original) or not parseable(generated):
        return None, []
    wanted = _resources(original)
    out: List[str] = []
    placed = set()
    last_sbatch = -1
    for line in generated.splitlines(keepends=True):
        key = _resource(line.rstrip("\n"))
        if key in wanted:
            if key not in placed:
                out.extend(w + "\n" for w in wanted[key])
                placed.add(key)
            last_sbatch = len(out) - 1
            continue
        out.append(line)
        if line.startswith("#SBATCH"):
            last_sbatch = len(out) - 1
    extra = [w + "\n" for key, lines in wanted.items() if key not in placed for w in lines]
    out[last_sbatch + 1:last_sbatch + 1] = extra
    carried = sorted(wanted)
    return _restamp("".join(out)), carried
