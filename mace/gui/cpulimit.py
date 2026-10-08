"""Per-process CPU-time limits (``ulimit -t``, RLIMIT_CPU), plan PLAN_connect D18.

Some clusters' dev nodes limit every process to, say, 120 CPU minutes (soft =
hard): the kernel kills the process that reaches it. MACE GUI's mostly idle
server rarely gets there; its long-lived helpers can (the ML worker, the
pool's workers, a big database build). This module tells how much each one
used, so they are restarted before the limit and the server stops cleanly
when it gets close itself. Detected from getrlimit only, with no host names;
limits a site enforces in other ways (cgroups, a monitoring daemon) are not
seen.

``MACE_GUI_TEST_CPU_LIMIT_S`` (with MACE_GUI_TEST_HOOKS=1) fakes the limit.
Standard library only.
"""
from __future__ import annotations

import os
import time
from typing import Optional

WARN_CHIP = 0.50          # the status-bar chip
WARN_STICKY = 0.80        # the sticky warning
STOP_AT = 0.97            # the server stops itself
RESTART_AT = 0.60         # a helper (ML worker, idle pool) is restarted from here
NEAR_KILL = 0.95          # a kill at or past this share counts as the CPU-time limit


def limit() -> Optional[float]:
    """This process's soft CPU-time limit in seconds, or None (unlimited, not POSIX)."""
    if os.environ.get("MACE_GUI_TEST_HOOKS") == "1" and os.environ.get("MACE_GUI_TEST_CPU_LIMIT_S"):
        try:
            v = float(os.environ["MACE_GUI_TEST_CPU_LIMIT_S"])
            return v if v > 0 else None
        except ValueError:
            pass
    try:
        import resource
    except ImportError:
        return None
    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_CPU)
    except (OSError, ValueError):
        return None
    if soft in (resource.RLIM_INFINITY, -1) or soft <= 0:
        return None
    return float(soft)


_TICK = None


def used(pid: int) -> Optional[float]:
    """utime + stime of one process in seconds (Linux /proc), else None."""
    global _TICK
    try:
        with open(f"/proc/{int(pid)}/stat") as fh:
            data = fh.read()
    except (OSError, ValueError):
        return None
    try:
        fields = data.rsplit(")", 1)[1].split()
        ticks = int(fields[11]) + int(fields[12])
    except (IndexError, ValueError):
        return None
    if _TICK is None:
        try:
            _TICK = os.sysconf("SC_CLK_TCK")
        except (ValueError, OSError, AttributeError):
            _TICK = 100
    return ticks / float(_TICK)


def own_used() -> float:
    return time.process_time()


def check_every(lim: float, base: float = 60.0) -> float:
    """Seconds between two checks of the server's own CPU time: once a minute,
    more often under a small limit, so a check always lands between the stop
    share and the kill (3 % of a 10-minute limit is only 18 s)."""
    return max(2.0, min(base, (1 - STOP_AT) * lim / 3))


def minutes(seconds: Optional[float]) -> str:
    return f"{(seconds or 0) / 60:.0f}"


def killed_by_limit(returncode: Optional[int], last_used: Optional[float], lim: Optional[float],
                    sample_s: float = 0.0) -> bool:
    """A child that died by SIGXCPU, or by SIGKILL with its CPU time (the
    last sample, `sample_s` seconds of CPU old at most) at the limit."""
    if returncode in (-24, 128 + 24):
        return True
    if lim is None or last_used is None or returncode not in (-9, 128 + 9):
        return False
    return last_used >= lim - max((1 - NEAR_KILL) * lim, 2 * sample_s)


def proc_limit(pid: int) -> Optional[float]:
    """Another process's soft CPU-time limit (Linux /proc/<pid>/limits), or None."""
    try:
        with open(f"/proc/{int(pid)}/limits") as fh:
            for line in fh:
                if line.startswith("Max cpu time"):
                    v = line.split()[3]
                    return None if v == "unlimited" else float(v)
    except (OSError, ValueError, IndexError):
        return None
    return None


MESSAGE = "This node's CPU-time limit ({m} min per process) stopped the {what}."
