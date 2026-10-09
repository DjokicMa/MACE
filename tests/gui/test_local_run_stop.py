"""Run on This Node stops cleanly (PLAN_submit D8): Cancel sends ONE SIGTERM
to `mace run` (which stops its job's session after its own grace period),
and only after KILL_GRACE + 10 s would the GUI kill anything itself. With a
stub Pcrystal that ignores SIGTERM, no process is left after Cancel or
after the server stops, and the stop takes less than KILL_GRACE + 10 s.

The stub binaries are the `mace run` suite's (tests/test_mace_run_cli.py)."""
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_mace_run_cli as MR  # noqa: E402
import submit_util as SU  # noqa: E402

from mace.submission.local_run import KILL_GRACE

pytestmark = pytest.mark.skipif(MR._module_in_login_shell(),
                                reason="a login shell here has `module`: the job script would load the real CRYSTAL")


def _exe(p, text):
    p.write_text(text)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)


def _box(tmp):
    stubs = tmp / "stubs"
    stubs.mkdir()
    _exe(stubs / "mpirun", MR.MPIRUN_STUB)
    for n in ("python", "sbatch"):
        _exe(stubs / n, MR.LOGGER)
    _exe(stubs / "squeue", MR.SQUEUE)
    ebroot = tmp / "ebroot_marker_d8"
    (ebroot / "bin").mkdir(parents=True)
    for n in ("Pcrystal", "Pproperties"):
        _exe(ebroot / "bin" / n, MR.CRYSTAL_STUB)
    for d in ("stubdir", "scratch", "home"):
        (tmp / d).mkdir()
    # mace run's scratch locks: this test's own, never the user's (or another
    # suite's) $XDG_RUNTIME_DIR/mace-run-<uid>, where a run of m.d12 would collide
    (tmp / "run").mkdir(mode=0o700)
    env = {"PATH": f"{stubs}:/usr/bin:/bin", "SCRATCH": str(tmp / "scratch"), "EBROOTCRYSTAL": str(ebroot),
           "STUB_LOG": str(tmp / "stub.log"), "STUB_DIR": str(tmp / "stubdir"), "HOME": str(tmp / "home"),
           "STUB_SLEEP": "60", "STUB_IGNORE_TERM": "1", "USER": SU.USER, "LOGNAME": SU.USER, "LANG": "C.UTF-8",
           "XDG_RUNTIME_DIR": str(tmp / "run")}
    return env, ebroot


def _leftovers(marker):
    r = subprocess.run(["pgrep", "-f", str(marker)], capture_output=True, text=True)
    return [p for p in r.stdout.split() if p.strip()]


def _start(tmp):
    env, ebroot = _box(tmp)
    root = tmp / "proj"
    (root / "decks").mkdir(parents=True)
    (root / "decks" / "m.d12").write_text(MR.DECK)
    proc, base, _ = SU.H.start_server(root, tmp / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1", **env)
    return proc, base, root, ebroot, tmp / "stubdir"


def _launch(base):
    import p4util as U
    st, _, r = U.req(base, "api/submit/preview", {"targets": ["decks/m.d12"], "where": "local",
                                                 "options": {"cores": 1, "timeout": "10m"}, "wait": 60})
    assert st == 200 and r["code"] == 0, r
    st, _, r = U.req(base, "api/submit/run", {"preview_id": r["preview_id"]})
    assert st == 202, r
    return r["task"]["id"]


def test_cancel_no_orphans(tmp_path):
    import p4util as U
    proc, base, root, ebroot, stubdir = _start(tmp_path)
    try:
        tid = _launch(base)
        end = time.time() + 60
        while time.time() < end and not (stubdir / "m.sleeping").exists():
            time.sleep(0.2)
        assert (stubdir / "m.sleeping").exists()
        assert _leftovers(ebroot)
        t0 = time.time()
        U.req(base, f"api/tasks/{tid}/cancel", {})
        while time.time() - t0 < KILL_GRACE + 30:
            st, _, r = U.req(base, "api/tasks")
            t = next(x for x in r["tasks"] if x["id"] == tid)
            if t["state"] in ("canceled", "done", "failed"):
                break
            time.sleep(0.3)
        took = time.time() - t0
        assert t["state"] == "canceled"
        assert took < KILL_GRACE + 10, took
        time.sleep(0.5)
        assert _leftovers(ebroot) == []
        log = (root / "decks" / "m-local.o").read_text()
        assert "STOPPED" in log or "interrupted" in log or "stopped" in log.lower()
    finally:
        proc.terminate()
        proc.wait(timeout=60)
        for pid in _leftovers(ebroot):
            os.kill(int(pid), 9)


def test_shutdown_no_orphans(tmp_path):
    proc, base, root, ebroot, stubdir = _start(tmp_path)
    try:
        _launch(base)
        end = time.time() + 60
        while time.time() < end and not (stubdir / "m.sleeping").exists():
            time.sleep(0.2)
        assert _leftovers(ebroot)
        t0 = time.time()
        proc.terminate()
        proc.wait(timeout=KILL_GRACE + 30)
        assert time.time() - t0 < KILL_GRACE + 15
        time.sleep(0.5)
        assert _leftovers(ebroot) == []
    finally:
        if proc.poll() is None:
            proc.kill()
        for pid in _leftovers(ebroot):
            os.kill(int(pid), 9)
