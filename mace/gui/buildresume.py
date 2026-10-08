"""The database build continues after a CPU-time-limit kill (plan PLAN_connect
D19, with PLAN_analysis C10, which owns the build itself in dbbuild.py).

When ``clirun`` reports that the build child was killed by a per-process
CPU-time limit (``Result.cpu_limit``), the same task runs the build again
with the same argv **minus** ``--backup`` / ``--keep-backups N`` (the first
attempt took the backup; a second one of a half-updated database would push
the pre-build backup out). ``--skip-unchanged`` makes the next attempt skip
what the killed one already committed. At most MAX_RELAUNCHES relaunches.
Without ``--skip-unchanged`` there is nothing to continue from: the task
fails with a message that says how to continue. A cancel between attempts
stops the relaunch. The CLI itself never relaunches (``mace analyze`` keeps
its behavior); this is the GUI task's rule.

``run_with_resume(attempt, argv, ctx, note)`` is what dbbuild.py calls;
standard library only.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional

MAX_RELAUNCHES = 3
SKIP_FLAG = "--skip-unchanged"
NO_RESUME = ("The database update reached this node's CPU-time limit. Turn on 'Only new and changed files' and "
             "run it again to continue where it stopped.")
RESUMING = ("The database update reached this node's CPU-time limit ({m} min per process) and was restarted to "
            "continue (attempt {n} of {total}). Files already written are kept.")


class BuildStopped(RuntimeError):
    """The build reached the CPU-time limit and cannot continue."""


def resume_argv(argv: List[str]) -> List[str]:
    """The relaunch argv: the same, without --backup and --keep-backups N."""
    out: List[str] = []
    skip = False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "--backup" or a.startswith("--keep-backups="):
            continue
        if a == "--keep-backups":
            skip = True
            continue
        out.append(a)
    return out


def run_with_resume(attempt: Callable[[List[str]], Any], argv: List[str], ctx: Any = None,
                    note: Optional[Callable[[str], None]] = None, limit_minutes: str = "",
                    max_relaunches: int = MAX_RELAUNCHES) -> Any:
    """Run ``attempt(argv)`` (a clirun.Result-like object with ``cpu_limit``)
    and relaunch it per the rule above; returns the last result."""
    total = max_relaunches + 1
    current = list(argv)
    for n in range(1, total + 1):
        res = attempt(current)
        if not getattr(res, "cpu_limit", False):
            return res
        if SKIP_FLAG not in argv:
            raise BuildStopped(NO_RESUME)
        if n == total:
            raise BuildStopped(f"The database update reached this node's CPU-time limit {total} times; run it again "
                               "with 'Only new and changed files' to continue.")
        if ctx is not None and getattr(ctx, "canceled", False):
            return res
        msg = RESUMING.format(m=limit_minutes or "?", n=n + 1, total=total)
        if note is not None:
            note(msg)
        current = resume_argv(argv)
    return res
