"""Child processes and private temporary directories for pre-optimization.

Every child (xtb, the ML worker, the ML environment probe) starts in its own
session with an environment built by ``child_env``: the parent's environment
is inherited, then every ``PYTHON*`` variable, ``VIRTUAL_ENV`` and every
inherited ``MACE_*`` variable is dropped (the repository root on PYTHONPATH
would shadow ACEsuit's ``mace`` package and the stdlib ``queue`` module),
and the thread counts and temp directory are forced. Everything else is kept,
including ``LD_LIBRARY_PATH`` and the module system's variables, which HPCC
module interpreters need.

Temp directories live under ``<tmp>/mace-preopt-<uid>/<pid>-<tag>-<random>``
(mode 0700). Their owner removes them in ``finally``; ``sweep`` removes what
a killed process left behind, but never a directory that a live process of
this user still has as its working directory (an orphaned xtb).
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional


class Canceled(Exception):
    """The caller's cancel predicate turned true; children were stopped."""


_DROP_PREFIXES = ("PYTHON", "MACE_")
_DROP_NAMES = ("VIRTUAL_ENV", "__PYVENV_LAUNCHER__")


def child_env(base: Optional[Dict[str, str]] = None, **force: str) -> Dict[str, str]:
    """Inherit, then deny: the parent's environment without PYTHON*,
    VIRTUAL_ENV and MACE_*, then ``force`` on top (values are str()-ed;
    a value of None removes the name)."""
    env = dict(os.environ if base is None else base)
    for k in list(env):
        if k.startswith(_DROP_PREFIXES) or k in _DROP_NAMES:
            del env[k]
    for k, v in force.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = str(v)
    return env


def _uid() -> Optional[int]:
    """This user's uid; None on Windows, which has none (os.getuid is POSIX only)."""
    getuid = getattr(os, "getuid", None)
    return getuid() if getuid is not None else None


def _user_tag() -> str:
    """The temp root's per-user part: the uid, or on Windows the user name."""
    uid = _uid()
    if uid is not None:
        return str(uid)
    try:
        import getpass
        name = getpass.getuser()
    except Exception:                      # no USERNAME and no pwd module
        name = ""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:40] or "user"


def temp_root() -> Path:
    """The per-user root of every pre-optimization temp directory."""
    uid = _uid()
    root = Path(tempfile.gettempdir()) / f"mace-preopt-{_user_tag()}"
    try:
        root.mkdir(mode=0o700, exist_ok=True)
    except FileExistsError:
        pass
    st = root.lstat()
    if not root.is_dir() or root.is_symlink() or (uid is not None and st.st_uid != uid):
        raise RuntimeError(f"{root} is not a private directory of this user")
    if uid is not None and st.st_mode & 0o077:
        os.chmod(root, 0o700)
    return root


def private_dir(tag: str) -> Path:
    """A fresh 0700 directory ``<pid>-<tag>-<random>`` under ``temp_root``."""
    tag = re.sub(r"[^A-Za-z0-9_]+", "_", tag)[:20] or "tmp"
    return Path(tempfile.mkdtemp(prefix=f"{os.getpid()}-{tag}-", dir=str(temp_root())))


def remove(path) -> None:
    if path:
        shutil.rmtree(str(path), ignore_errors=True)


def _alive(pid: int) -> bool:
    if os.name == "nt":
        from mace.gui.registry import pid_alive     # os.kill(pid, 0) would end the process on Windows
        return pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _live_cwds() -> list:
    """Working directories of this user's live processes (Linux /proc)."""
    out = []
    uid = _uid()
    if uid is None:
        return out
    try:
        entries = os.listdir("/proc")
    except OSError:
        return out
    for e in entries:
        if not e.isdigit():
            continue
        try:
            if os.stat(f"/proc/{e}").st_uid != uid:
                continue
            out.append(os.path.realpath(os.readlink(f"/proc/{e}/cwd")))
        except OSError:
            continue
    return out


def sweep() -> int:
    """Remove the temp directories of dead processes; returns how many went.
    A directory whose owner pid is alive, or that a live process of this user
    has as (or under) its working directory, is kept."""
    if os.name == "nt":
        # os.kill(pid, 0) is no liveness check there (signal 0 is CTRL_C_EVENT): nothing is removed
        return 0
    try:
        root = temp_root()
    except (OSError, RuntimeError):
        return 0
    cwds = None
    removed = 0
    for d in root.iterdir():
        m = re.match(r"(\d+)-", d.name)
        if not m or not d.is_dir() or d.is_symlink():
            continue
        if _alive(int(m.group(1))):
            continue
        if cwds is None:
            cwds = _live_cwds()
        real = os.path.realpath(str(d))
        if any(c == real or c.startswith(real + os.sep) for c in cwds):
            continue
        remove(d)
        removed += 1
    return removed


def stop(proc: subprocess.Popen, grace: float = 2.0) -> None:
    """SIGTERM to the child's process group, SIGKILL after ``grace`` s, then reap."""
    if proc is None or proc.poll() is not None:
        return
    killpg = getattr(os, "killpg", None)              # POSIX only, as SIGKILL
    for sig, wait in ((signal.SIGTERM, grace), (getattr(signal, "SIGKILL", signal.SIGTERM), 5.0)):
        try:
            if killpg is None:
                raise ProcessLookupError
            killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            try:
                proc.send_signal(sig)
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


_STACK_RAISED = False


def raise_stack_limit() -> None:
    """Raise this process's soft stack limit to the hard limit, once, so the
    children inherit it (xtb's GFN-FF needs a large stack; preexec_fn is not
    safe in a threaded process)."""
    global _STACK_RAISED
    if _STACK_RAISED:
        return
    _STACK_RAISED = True
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        if soft != hard:
            resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))
    except (ImportError, ValueError, OSError):
        pass


def default_threads() -> int:
    """Threads for an ML worker: SLURM's allocation inside a job, else a
    quarter of the CPUs this process may use, 1 to 4."""
    slurm = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if slurm.isdigit() and int(slurm) > 0:
        return int(slurm)
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    return min(4, max(1, n // 4))


def mem_available_gb() -> Optional[float]:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def wait_child(proc: subprocess.Popen, canceled, timeout: float, poll: float = 0.05) -> int:
    """Wait for ``proc``, checking the pure ``canceled()`` every ``poll`` s.
    A cancel stops the child's group and raises Canceled; the timeout stops
    it and raises TimeoutError. Anything else that ends the wait (Ctrl-C in
    ``mace preopt``: the child runs in its own session and never sees it)
    stops the child too, so it does not outlive its work directory."""
    end = time.monotonic() + timeout
    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                return rc
            if canceled is not None and canceled():
                stop(proc)
                raise Canceled("canceled")
            if time.monotonic() > end:
                stop(proc)
                raise TimeoutError(f"no answer within {timeout:g} s")
            time.sleep(poll)
    except BaseException:
        stop(proc)
        raise
