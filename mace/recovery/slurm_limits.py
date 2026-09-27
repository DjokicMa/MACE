"""How long a job's own queue actually lets it run, asked of SLURM live.

The timeout recovery doubles the walltime; the result must never exceed what
the queue allows. On MSU HPCC (measured 2026-09-25) that is 7 days on every
general partition and mendoza_q, but 14 days on mendoza_q_long - reachable only
with `-A mendoza_q_long`. The templates carry no `-p` at all (`-A mendoza_q`
alone); a submit plugin routes the job, and `sbatch --test-only` reports the
partition it lands in:
    sbatch: Job N to start at ... using 32 processors on nodes X in partition mendoza_q
while an over-long request is refused:
    allocation failure: Requested time limit is invalid (missing or exceeds some limit)
--test-only validates without submitting anything.

So the limit is: MaxTime of the partition the job is routed to (or of its
`-p` partitions when SLURM does not say), tightened by the user's association
MaxWall for the job's account when one is set. The job is never moved to
another partition or account. Every SLURM call degrades to "unknown" when
SLURM is missing or unresponsive, and the caller falls back to its configured
maximum.
"""

import getpass
import os
import re
import subprocess
from pathlib import Path
from typing import Callable, List, Optional, Tuple

Runner = Callable[..., subprocess.CompletedProcess]

_TIMEOUT = 30


def run_slurm(cmd: List[str], cwd: Optional[str] = None) -> Optional[subprocess.CompletedProcess]:
    """Run a SLURM client command; None when SLURM can't be reached."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=_TIMEOUT, cwd=cwd)
    except (OSError, subprocess.SubprocessError):
        return None


def _directive(script_text: str, short: str, long: str) -> Optional[str]:
    """Value of an #SBATCH option (`-A x`, `-Ax`, `--account=x`, `--account x`);
    a trailing `# comment` (the template has `-A mendoza_q # or general`) is
    not part of it."""
    pattern = (r'^\s*#SBATCH\s+(?:' + re.escape(short) + r'\s*|'
               + re.escape(long) + r'(?:=|\s+))([^\s#]+)')
    m = re.search(pattern, script_text or '', re.M)
    return m.group(1) if m else None


def script_partitions(script_text: str) -> List[str]:
    value = _directive(script_text, '-p', '--partition')
    return [p for p in value.split(',') if p] if value else []


def script_account(script_text: str) -> Optional[str]:
    return _directive(script_text, '-A', '--account')


def parse_walltime(value) -> Optional[int]:
    """SLURM time string to seconds; None for UNLIMITED / unparseable."""
    if not value:
        return None
    value = str(value).strip()
    if value.upper() in ('UNLIMITED', 'INFINITE', 'NONE', 'N/A'):
        return None
    days = 0
    has_days = '-' in value
    if has_days:
        d, _, value = value.partition('-')
        try:
            days = int(d)
        except ValueError:
            return None
    try:
        nums = [int(p) for p in value.split(':')]
    except ValueError:
        return None
    if has_days:
        nums += [0] * (3 - len(nums))
        h, m, s = nums[:3]
    elif len(nums) == 1:
        h, m, s = 0, nums[0], 0
    elif len(nums) == 2:
        h, m, s = 0, nums[0], nums[1]
    else:
        h, m, s = nums[:3]
    return ((days * 24 + h) * 60 + m) * 60 + s


def sbatch_test_only(script_path: Path, walltime: str,
              runner: Runner = None) -> Tuple[Optional[bool], Optional[str], str]:
    """Ask SLURM whether the script would be accepted with this walltime.

    Returns (accepted, partition, message); accepted is None when SLURM could
    not be asked at all. Nothing is submitted.
    """
    runner = runner or run_slurm
    script_path = Path(script_path)
    res = runner(['sbatch', '--test-only', '-t', walltime, str(script_path)],
                 cwd=str(script_path.parent))
    if res is None:
        return None, None, 'sbatch not available'
    text = (res.stdout or '') + (res.stderr or '')
    m = re.search(r'\bin partition (\S+)', text)
    partition = m.group(1) if m else None
    if res.returncode == 0:
        return True, partition, text.strip()
    if 'sbatch: error: Batch job submission failed' in text or 'invalid' in text.lower() \
            or 'allocation failure' in text.lower():
        return False, partition, text.strip()
    # Some other failure (controller down, auth): not an answer about time.
    return None, partition, text.strip()


def partition_max_time(partition: str, runner: Runner = None) -> Optional[int]:
    runner = runner or run_slurm
    # -a: mendoza_q and mendoza_q_long are HIDDEN partitions, and without it
    # scontrol answers "Partition mendoza_q not found" (measured on HPCC).
    res = runner(['scontrol', '-a', 'show', 'partition', partition, '-o'])
    if res is None or res.returncode != 0:
        return None
    m = re.search(r'\bMaxTime=(\S+)', res.stdout or '')
    return parse_walltime(m.group(1)) if m else None


def association_max_wall(account: str, user: Optional[str] = None,
                         runner: Runner = None) -> Optional[int]:
    """Smallest MaxWall set on the user's association with the account."""
    runner = runner or run_slurm
    user = user or os.environ.get('USER') or getpass.getuser()
    res = runner(['sacctmgr', '-nP', 'show', 'assoc', f'user={user}',
                  f'account={account}', 'format=maxwall'])
    if res is None or res.returncode != 0:
        return None
    walls = [parse_walltime(l) for l in (res.stdout or '').splitlines() if l.strip()]
    walls = [w for w in walls if w]
    return min(walls) if walls else None


def queue_walltime_limit(script_path: Path, script_text: str, current_walltime: str,
                         runner: Runner = None) -> Tuple[Optional[int], str]:
    """(limit_seconds, how it was found); limit None when SLURM can't tell.

    `current_walltime` is the time the job already ran with, which SLURM
    accepted, so probing with it reveals the partition without asking for
    anything new.
    """
    accepted, routed, msg = sbatch_test_only(script_path, current_walltime, runner)
    if accepted is None and routed is None:
        return None, f'SLURM not reachable ({msg.splitlines()[0] if msg else "no answer"})'

    partitions = [routed] if routed else script_partitions(script_text)
    if not partitions:
        return None, 'SLURM did not report the partition'
    limits = [partition_max_time(p, runner) for p in partitions]
    limits = [l for l in limits if l]
    if not limits:
        return None, f'no MaxTime for partition {",".join(partitions)}'
    # A job listing several partitions may run in any of them, and SLURM
    # drops those too short for it - the longest is what it can be granted.
    limit = max(limits)
    how = f'partition {",".join(partitions)} MaxTime'
    account = script_account(script_text)
    if account:
        wall = association_max_wall(account, runner=runner)
        if wall and wall < limit:
            limit, how = wall, f'association {account} MaxWall'
    return limit, how


def format_walltime(total_seconds: int) -> str:
    days, rem = divmod(int(total_seconds), 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"


def capped_walltime(script_path: Path, script_text: str, current: int, target: int,
                    fallback_max: Optional[int], runner: Runner = None) -> Tuple[int, str]:
    """The walltime to resubmit with: `target`, cut back to what the job's
    queue allows, never below `current` (which already ran). Returns
    (seconds, one-line reason for the log).

    SLURM is the judge: the target is tried with --test-only first; if it is
    refused, the queue's limit is looked up and that value confirmed the same
    way. Only when SLURM cannot be asked does the configured maximum apply.
    """
    if target <= current:
        return current, 'no increase requested'
    accepted, routed, msg = sbatch_test_only(script_path, format_walltime(target), runner)
    if accepted:
        where = f' in partition {routed}' if routed else ''
        return target, f'SLURM accepts {format_walltime(target)}{where}'
    if accepted is None:
        if fallback_max:
            new = max(min(target, fallback_max), current)
            return new, (f'SLURM not reachable; capped at the configured maximum '
                         f'{format_walltime(fallback_max)}')
        return target, 'SLURM not reachable and no configured maximum'

    limit, how = queue_walltime_limit(script_path, script_text,
                                      format_walltime(current), runner)
    if limit is None:
        # The refusal need not be about time at all (a QOS submit limit, say,
        # refuses every probe), so an unreadable limit means "cannot tell",
        # not "at the limit": use the configured maximum, as when SLURM is
        # unreachable, and let the real submission report any other refusal.
        if fallback_max:
            new = max(min(target, fallback_max), current)
            return new, (f'SLURM refused {format_walltime(target)} and its limit could '
                         f'not be read ({how}); capped at the configured maximum '
                         f'{format_walltime(fallback_max)}')
        return current, (f'SLURM refused {format_walltime(target)} and its limit could '
                         f'not be read ({how}); keeping {format_walltime(current)}')
    new = min(target, limit)
    if new <= current:
        return current, (f'already at the queue limit {format_walltime(limit)} ({how}); '
                         f'walltime not raised')
    ok, _, _ = sbatch_test_only(script_path, format_walltime(new), runner)
    if ok:
        return new, f'capped at the queue limit {format_walltime(limit)} ({how})'
    # The queue's own limit allows `new`, so this refusal is about something
    # other than time; resubmit within the limit and let sbatch report it.
    return new, (f'capped at the queue limit {format_walltime(limit)} ({how}); '
                 f'SLURM refused the probe for another reason')
