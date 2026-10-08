"""MACE's own command line as a cancelable subprocess (spec section 8.4.8, plan B2/T1).

Deck previews, the CLI's defaults, ``mace plotting`` and ``mace preflight``
run the real ``mace_cli`` in a child process:

* at nice 10 with one BLAS/OpenMP thread (spec section 9.1), in its own
  session, so a cancel reaches the CLI's own children too;
* inside a task's dispatcher thread that holds one slot of the shared CPU
  budget (compute.GATE), never in a request thread;
* canceled by SIGTERM to the process group, then SIGKILL after 2 s;
* with stdin closed (or a given text), and its output in temporary files,
  so a chatty CLI never fills a pipe.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
MACE_CLI = REPO / "mace_cli"
DRIVER = Path(__file__).with_name("answer_driver.py")
NICENESS = 10
KILL_AFTER_S = 2.0
POLL_S = 0.05
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
# the sources that decide what the CLI asks and writes: a change to any of them is a new CLI
CLI_SOURCES = ("mace_cli", "Crystal_d12", "Crystal_d3", "mace/plotting", "mace/preflight")


class CliCanceled(Exception):
    """The task was canceled; the CLI's process group was stopped."""


class CliTimeout(Exception):
    """The CLI ran past its time limit and was stopped."""


def cli_env() -> Dict[str, str]:
    """The environment every CLI child gets: plain output, no banner, no
    bytecode written into the source tree, one thread per math library, and
    a PYTHONPATH without the MACE checkout or relative entries (childenv.py:
    mace/queue would shadow the standard library's queue). The rest of an
    inherited PYTHONPATH stays, so a cluster's module bundles still import."""
    from .childenv import python_child_env
    env = {k: v for k, v in python_child_env().items() if k not in ("COLORTERM", "TERM")}
    env.update(NO_COLOR="1", MACE_NO_BANNER="1", MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    for k in THREAD_VARS:
        env[k] = "1"
    # test hooks only: a PYTHONPATH in front (a stand-in that hides seekpath, plan T3 step 4)
    if os.environ.get("MACE_GUI_TEST_HOOKS") == "1" and os.environ.get("MACE_GUI_TEST_CLI_PYTHONPATH"):
        env["PYTHONPATH"] = os.environ["MACE_GUI_TEST_CLI_PYTHONPATH"]
    return env


def _niced(argv: List[str]) -> List[str]:
    """argv run at nice 10 (or this server's own niceness, if that is lower
    priority already): through nice(1), not a preexec function, which is not
    safe in a threaded server."""
    nice = shutil.which("nice")
    try:
        step = max(0, NICENESS - os.nice(0))
    except OSError:
        step = 0
    return [nice, "-n", str(step), *argv] if nice and step else list(argv)


def mace_argv(*args: str) -> List[str]:
    return [sys.executable, str(MACE_CLI), *args]


def driver_argv(rules: Path, transcript: Path, *args: str) -> List[str]:
    """``mace_cli args`` with its questions answered by rule (answer_driver.py)."""
    return [sys.executable, str(DRIVER), str(rules), str(transcript), str(MACE_CLI), *args]


def test_delay() -> float:
    """MACE_GUI_TEST_CLI_DELAY (test hooks only): a CLI that is slow to start, for the cancel tests."""
    if os.environ.get("MACE_GUI_TEST_HOOKS") != "1":
        return 0.0
    try:
        return max(0.0, min(60.0, float(os.environ.get("MACE_GUI_TEST_CLI_DELAY", "0"))))
    except ValueError:
        return 0.0


class Result:
    def __init__(self, code: int, out: str, err: str, elapsed: float, pid: int, cpu_limit: bool = False):
        self.returncode, self.stdout, self.stderr, self.elapsed, self.pid = code, out, err, elapsed, pid
        # the child was killed by a per-process CPU-time limit (cpulimit.py): the database
        # build continues in a new attempt, a local run says to submit it as a job instead
        self.cpu_limit = cpu_limit

    @property
    def log(self) -> str:
        return self.stdout + self.stderr


def stop(proc: subprocess.Popen) -> float:
    """SIGTERM to the process group, SIGKILL after 2 s; the seconds it took."""
    t0 = time.monotonic()
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(KILL_AFTER_S)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait(5)
    # anything left in the group (a grandchild that ignored TERM)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    return time.monotonic() - t0


CPU_SAMPLE_S = 0.5


def _reap(proc: subprocess.Popen) -> Optional[float]:
    """Reap the child ourselves when it has ended (os.wait4): its exit code goes
    to proc.returncode, and its final CPU time is returned (None while it runs)."""
    if proc.returncode is not None or not hasattr(os, "wait4"):
        return None
    try:
        pid, status, ru = os.wait4(proc.pid, os.WNOHANG)
    except ChildProcessError:
        return None
    if pid == 0:
        return None
    proc.returncode = os.waitstatus_to_exitcode(status)
    return ru.ru_utime + ru.ru_stime


def stop_gracefully(proc: subprocess.Popen, grace: float, on_kill=None) -> float:
    """ONE SIGTERM to ``proc`` itself (not its group, not twice: a program
    such as ``mace run`` stops its own children on that signal and skips its
    grace on a second one), then up to ``grace`` seconds; only then SIGKILL
    to the whole group, and ``on_kill(seconds)`` is told. The seconds it took."""
    t0 = time.monotonic()
    try:
        proc.send_signal(signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.wait(grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            pass
        if on_kill is not None:
            on_kill(time.monotonic() - t0)
    return time.monotonic() - t0


def run(argv: List[str], cwd: Path, ctx: Any = None, timeout: Optional[float] = 180.0,
        stdin_text: Optional[str] = None, on_line=None, env: Optional[Dict[str, str]] = None, nice: bool = True,
        cancelable: bool = True, stopper=None, on_output=None) -> Result:
    """Run ``argv`` in ``cwd`` and wait; ``ctx`` (a task's context) is
    polled for a cancel every 50 ms and told the child's pid. ``on_line``
    (optional) is called with each complete stdout line as it arrives (the
    DB build's --progress-json lines); the whole stdout is still returned.

    ``env``: the child's environment (default cli_env()); ``nice=False``
    runs it without nice(1); ``timeout=None`` sets no time limit;
    ``cancelable=False`` ignores a cancel once the child runs (a submit
    must not stop halfway); ``stopper(proc)`` replaces the group stop for a
    cancel or the timeout (``mace run`` gets stop_gracefully);
    ``on_output(text)`` is called with the output so far, at most twice a
    second, while the child runs."""
    delay = test_delay()
    if delay:
        argv = [sys.executable, "-c", "import os, sys, time; time.sleep(float(sys.argv[1])); os.execv(sys.argv[2], sys.argv[2:])",
                str(delay), *argv]
    stop_fn = stopper or stop
    t0 = time.monotonic()
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(_niced(argv) if nice else list(argv), cwd=str(cwd),
                                stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                                stdout=out, stderr=err, env=cli_env() if env is None else env,
                                start_new_session=True)
        if ctx is not None and hasattr(ctx, "set_pid"):
            ctx.set_pid(proc.pid)
        if stdin_text is not None:
            try:
                proc.stdin.write(stdin_text.encode())
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        seen, partial = 0, b""

        def feed(final: bool = False) -> None:
            nonlocal seen, partial
            if on_line is None:
                return
            while True:
                chunk = os.pread(out.fileno(), 1 << 16, seen)
                if not chunk:
                    break
                seen += len(chunk)
                partial += chunk
            *lines, partial = partial.split(b"\n")
            if final and partial:
                lines.append(partial)
                partial = b""
            for ln in lines:
                on_line(ln.decode("utf-8", "replace"))

        from . import cpulimit
        lim = cpulimit.proc_limit(proc.pid) or cpulimit.limit()
        last_cpu, next_sample = None, 0.0
        last_out = 0.0
        try:
            while True:
                final = _reap(proc)
                if final is not None:
                    last_cpu = final              # the child's whole CPU time, from wait4
                    break
                if proc.returncode is not None or (not hasattr(os, "wait4") and proc.poll() is not None):
                    break
                time.sleep(POLL_S)
                if time.monotonic() >= next_sample:
                    # how close the child is to its CPU-time limit (its own, read again: a
                    # wrapper may set it after the start); rate-limited
                    next_sample = time.monotonic() + CPU_SAMPLE_S
                    lim = cpulimit.proc_limit(proc.pid) or lim
                    got = cpulimit.used(proc.pid)
                    if got is not None:
                        last_cpu = got
                feed()
                if cancelable and ctx is not None and getattr(ctx, "canceled", False):
                    took = stop_fn(proc)
                    if hasattr(ctx, "note_cancel"):
                        ctx.note_cancel(took)
                    raise CliCanceled("canceled")
                if timeout is not None and time.monotonic() - t0 > timeout:
                    stop_fn(proc)
                    raise CliTimeout(f"the CLI ran past {timeout:.0f} s and was stopped")
                if on_output is not None and time.monotonic() - last_out > 0.5:
                    last_out = time.monotonic()
                    try:
                        # pread: the child shares this file's offset, so never seek it
                        fd = out.fileno()
                        size = min(os.fstat(fd).st_size, 4 << 20)
                        on_output(os.pread(fd, size, 0).decode("utf-8", "replace"))
                    except Exception:
                        pass
        except BaseException:
            if proc.poll() is None:
                stop_fn(proc)
            raise
        finally:
            if ctx is not None and hasattr(ctx, "set_pid"):
                ctx.set_pid(None)
        feed(final=True)
        out.seek(0)
        err.seek(0)
        return Result(proc.returncode, out.read().decode("utf-8", "replace"), err.read().decode("utf-8", "replace"),
                      time.monotonic() - t0, proc.pid,
                      cpu_limit=cpulimit.killed_by_limit(proc.returncode, last_cpu, lim, CPU_SAMPLE_S))


_CLI_HASH: Optional[str] = None


def cli_hash() -> str:
    """A short hash of the CLI's sources (names, sizes, mtimes): cached CLI
    answers are kept only while the CLI that gave them is unchanged."""
    global _CLI_HASH
    if _CLI_HASH is not None:
        return _CLI_HASH
    h = hashlib.sha256()
    for rel in CLI_SOURCES:
        p = REPO / rel
        files = [p] if p.is_file() else sorted(p.rglob("*.py")) if p.is_dir() else []
        for f in files:
            try:
                st = f.stat()
            except OSError:
                continue
            h.update(f"{f.relative_to(REPO)}:{st.st_size}:{st.st_mtime_ns}".encode())
    _CLI_HASH = h.hexdigest()[:16]
    return _CLI_HASH
