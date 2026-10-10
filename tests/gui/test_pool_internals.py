"""The process pool's release gates (plan section 5.2).

* pool.py relies on two CPython internals: fail loudly here when a Python
  version moves them (the pool then falls back to the stock spawn context);
* the automatic worker count is at most a quarter of the cores, and within SLURM;
* no worker outlives the server: SIGTERM, Ctrl+C (SIGINT to the process
  group, quietly), a hangup (SIGHUP to the group) and SIGKILL, each with a
  pool job running.
"""
import inspect
import os
import re
import signal
import subprocess
import sys
import time

import pytest

import guiharness as H
import p4util as U
from mace.gui import pool, resources


def test_spawn_internals_are_where_pool_expects_them():
    from multiprocessing import popen_spawn_posix, spawn
    assert list(inspect.signature(popen_spawn_posix.Popen._launch).parameters) == ["self", "process_obj"]
    assert list(inspect.signature(spawn.get_preparation_data).parameters) == ["name"]
    assert pool.internals_ok()
    assert isinstance(pool.make_context(), pool.CleanSpawnContext)


def test_a_moved_internal_falls_back_to_the_stock_spawn_context(monkeypatch, capsys):
    from multiprocessing import spawn
    monkeypatch.setattr(spawn, "get_preparation_data", lambda name, extra: {})
    monkeypatch.setattr(pool, "_WARNED", False)
    ctx = pool.make_context()
    assert not isinstance(ctx, pool.CleanSpawnContext) and ctx.get_start_method() == "spawn"
    assert "standard spawn start method" in capsys.readouterr().err
    pool.make_context()
    assert capsys.readouterr().err == ""                 # once


@pytest.mark.parametrize("ncores,slurm,most", [(24, None, 6), (4, None, 1), (2, None, 1), (64, None, 8),
                                               (24, "2", 2), (24, "16", 6)])
def test_auto_worker_count_is_a_quarter_of_the_cores(monkeypatch, ncores, slurm, most):
    monkeypatch.setattr(resources, "cores", lambda: ncores)
    monkeypatch.delenv("MACE_GUI_MAX_WORKERS", raising=False)
    if slurm is None:
        monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
    else:
        monkeypatch.setenv("SLURM_CPUS_ON_NODE", slurm)
    assert 1 <= resources.effective_workers(None) <= most
    assert resources.effective_workers(None) == min(max(1, ncores // 4), most)


# --- no orphans ------------------------------------------------------------------------

def _session_pids(sid):
    out = []
    for e in os.listdir("/proc"):
        if not e.isdigit():
            continue
        try:
            with open(f"/proc/{e}/stat") as fh:
                fields = fh.read().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if int(fields[3]) == sid and fields[0] != "Z":       # field 6: session id
            out.append(int(e))
    return out


def _foreground_sigint():
    """The server as a terminal starts it: Ctrl+C at its default. A pytest
    started in the background of a script inherits SIGINT ignored, and the
    server would inherit that in turn."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def _nice(pid):
    with open(f"/proc/{pid}/stat") as fh:
        return int(fh.read().rsplit(")", 1)[1].split()[16])


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGKILL],
                         ids=["sigterm", "ctrl-c", "hangup", "sigkill"])
def test_no_worker_outlives_the_server(tmp_path, sig):
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    log = tmp_path / "server.log"          # a file, not a pipe: an orphan holding it never blocks the test
    with open(log, "w") as fh:
        proc = subprocess.Popen(
            [sys.executable, str(H.MACE_CLI), "--no-banner", "gui", "--dir", str(root), "--port", "0",
             "--token", H.TOKEN],
            stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT, text=True,
            env=H.env(xdg, MACE_GUI_TEST_HOOKS="1", MACE_GUI_NO_WARMUP="1", MACE_GUI_MAX_WORKERS="2"),
            start_new_session=True, preexec_fn=_foreground_sigint)
    sid = proc.pid
    try:
        base = None
        end = time.time() + 60
        while time.time() < end and base is None and proc.poll() is None:
            m = re.search(H.URL_RE, log.read_text())
            base = m.group(1) if m else None
            time.sleep(0.1)
        assert base, log.read_text()
        st, _, r = U.post(base, "api/test/task", {"job": "sleep", "seconds": 30, "steps": 60, "wait": 0})
        assert st in (200, 202), r
        workers = []
        end = time.time() + 20
        while time.time() < end and not workers:
            workers = (U.req(base, "api/test/pool")[2].get("pool") or {}).get("pids") or []
            time.sleep(0.2)
        assert workers, "no pool worker started"
        assert all(_nice(w) >= 10 for w in workers)
        if sig in (signal.SIGINT, signal.SIGHUP):
            os.killpg(sid, sig)                    # Ctrl+C and a hangup reach the whole group
        else:
            os.kill(proc.pid, sig)
        proc.wait(timeout=10)
        end = time.time() + 10
        left = _session_pids(sid)
        while left and time.time() < end:
            time.sleep(0.2)
            left = _session_pids(sid)
        out = log.read_text()
        assert not left, (left, out[-2000:])
        if sig != signal.SIGKILL:
            assert "KeyboardInterrupt" not in out and "Traceback" not in out, out[-3000:]
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        for p in _session_pids(sid):
            try:
                os.kill(p, signal.SIGKILL)
            except OSError:
                pass


def test_one_worker_still_writes_a_deck(tmp_path):
    """One worker (MACE_GUI_MAX_WORKERS=1, as on a 4-core salloc): the deck
    task holds the only slot and reads a P1 CIF, whose symmetry (for the
    "this CIF is P1" note) is a gated call; it runs in the task's own slot
    instead of waiting for ever."""
    pytest.importorskip("ase")
    import json
    from ase.io import read, write
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    write(str(root / "structures" / "MgO_p1.cif"), read(str(root / "structures" / "MgO.cif")))   # every atom, P 1
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_MAX_WORKERS="1", MACE_GUI_NO_WARMUP="1")
    try:
        cfg = json.loads((H.REPO_ROOT / "Crystal_d12" / "example_configs" / "high_accuracy_sp.json").read_text())
        st, _, r = U.post(base, "api/deck/preview", {"mode": "cif2d12", "json": cfg, "structure": "structures/MgO_p1.cif"},
                          timeout=90)
        assert st == 200 and "CRYSTAL" in r["deck"], r
        st, _, info = U.req(base, "api/resources")
        assert st == 200
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def test_nothing_is_submitted_once_the_pool_was_killed():
    """A task dispatched while the server stops must not start a worker the
    exiting server would leave half started (EOFError on stderr)."""
    p = pool.Pool(lambda: 1, lambda *a: None)
    p.shutdown(kill=True)
    with pytest.raises(RuntimeError, match="stopping"):
        p.submit("sleep", {}, 0)
    assert p.ex is None and p.started == 0
