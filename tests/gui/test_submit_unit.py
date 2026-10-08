"""Unit tests of the Submit pieces (PLAN_submit D1c, D7, D26, D43, T1):
the recording shim, the argv whitelist, the squeue parser, the database
copy taken while a writer writes, the staging bound."""
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import submit_util as SU
from submit_util import SP_D12, BAND_D3

from mace.gui import jobs, submit as sub
from mace.gui import slurm_preview_shim as shim


def _shim(tmp, fake):
    real = {t: str(fake.bindir / t) for t in shim.TOOLS}
    log = tmp / "shim.log"
    shim.install(tmp / "shim", real, log)
    env = dict(fake.env)
    env["PATH"] = f"{tmp / 'shim'}{os.pathsep}{env['PATH']}"
    return env, log


def _run(env, *argv, cwd=None):
    return subprocess.run(list(argv), env=env, cwd=cwd, capture_output=True, text=True, timeout=60)


def test_shim_passthrough_and_intercept(tmp_path):
    fake = SU.FakeSlurm(tmp_path / "fs", {"nodes": "NodeName=n1 Features=amr"})
    env, log = _shim(tmp_path, fake)
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\n#SBATCH -J x\necho hi\n")
    r = _run(env, "sbatch", str(script))
    assert r.stdout.strip() == "Submitted batch job 900000000"
    assert _run(env, "sbatch", "--parsable", str(script)).stdout.strip() == "900000001"
    for argv in (["scancel", "5"], ["scontrol", "hold", "5"], ["scontrol", "release", "5"], ["scontrol", "update", "x"],
                 ["scontrol", "requeue", "5"], ["sacctmgr", "add", "user", "x"], ["sacctmgr", "-i", "delete", "x"]):
        assert _run(env, *argv).returncode == 0
    assert "amr" in _run(env, "scontrol", "show", "nodes").stdout
    _run(env, "squeue", "-h")
    _run(env, "sinfo")
    _run(env, "sacct", "-j", "1")
    _run(env, "sacctmgr", "-nP", "show", "assoc")
    real = [c["argv"] for c in fake.calls()]
    assert ["scontrol", "show", "nodes"] in real and ["squeue", "-h"] in real and ["sinfo"] in real
    assert not [a for a in real if a[0] in ("sbatch", "scancel") or (a[0] == "scontrol" and a[1] != "show")
                or (a[0] == "sacctmgr" and "show" not in a)]
    rec = [json.loads(l) for l in log.read_text().splitlines()]
    inter = [r["argv"] for r in rec if r.get("intercepted")]
    assert ["scontrol", "release", "5"] in inter and ["scancel", "5"] in inter
    sb = [r for r in rec if r["tool"] == "sbatch" and r.get("intercepted")]
    assert sb[0]["script_text"].startswith("#!/bin/bash") and sb[0]["job_id"] == "900000000"


def test_shim_test_only_passthrough(tmp_path):
    fake = SU.FakeSlurm(tmp_path / "fs")
    env, log = _shim(tmp_path, fake)
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\necho hi\n")
    r = _run(env, "sbatch", "--test-only", str(script))
    assert "to start at" in r.stderr
    assert [c["argv"] for c in fake.calls("sbatch")] == [["sbatch", "--test-only", str(script)]]
    _run(env, "sbatch", str(script))
    assert len(fake.calls("sbatch")) == 1                       # the plain sbatch never reached SLURM
    rec = [json.loads(l) for l in log.read_text().splitlines()]
    assert [r.get("passed") for r in rec if r["tool"] == "sbatch"] == [True, None]


def test_shim_without_slurm_answers_empty(tmp_path):
    shim.install(tmp_path / "shim", {t: None for t in shim.TOOLS}, tmp_path / "log")
    env = {"PATH": f"{tmp_path / 'shim'}:/usr/bin:/bin"}
    assert "MaxArraySize" in _run(env, "scontrol", "show", "config").stdout
    assert _run(env, "squeue", "-h").stdout == ""
    rec = [json.loads(l) for l in (tmp_path / "log").read_text().splitlines()]
    assert all(r.get("no_slurm") for r in rec)


def test_shim_never_on_the_real_run_path(monkeypatch):
    monkeypatch.setattr(sub, "login_env", lambda: {"PATH": "/usr/bin:/bin", "MACE_GUI_TOKEN": "x" * 20})
    env = sub.submit_env()
    assert "shim" not in env["PATH"] and "MACE_GUI_TOKEN" not in env


def test_submit_env_inside_allocation(monkeypatch):
    monkeypatch.delenv("MACE_GUI_TEST_HOOKS", raising=False)
    base = {"PATH": "/bin", "SLURM_JOB_ID": "12", "SLURM_MEM_PER_NODE": "4000", "SLURM_CPUS_ON_NODE": "4",
            "SLURM_CONF": "/etc/slurm.conf", "FAKE_SLURM_LOG": "x", "MACE_GUI_GATEWAY": "g", "OMP_NUM_THREADS": "8"}
    env = sub.submit_env(base)
    assert env == {"PATH": "/bin", "SLURM_CONF": "/etc/slurm.conf", "OMP_NUM_THREADS": "8"}


def test_build_commands_whitelist(tmp_path):
    root = tmp_path
    d = root / "p" / "decks"
    d.mkdir(parents=True)
    for n in ("a", "b"):
        (d / f"{n}.d12").write_text(SP_D12)
    sel = sub.selection(root, [d])
    (c,) = sub.build_commands(root, "slurm", sel, {"array": True, "max_running": 3, "track": True,
                                                   "walltime": "1-00:00:00", "exclude": {"kind": "nodes", "value": "n[1-3],n9"},
                                                   "overwrite_sh": True})
    assert c.argv == ["submit", "--array", "--max-running", "3", "--track", "--walltime", "1-00:00:00",
                      "--exclude-nodes", "n[1-3],n9", "--overwrite-sh", "decks"]
    assert c.display == "cd p && mace submit --array --max-running 3 --track --walltime 1-00:00:00 " \
                        "--exclude-nodes 'n[1-3],n9' --overwrite-sh decks"
    for ok in ("30", "30:00", "2:00:00", "1-00", "1-12:00", "7-00:00:00"):
        (c,) = sub.build_commands(root, "slurm", sel, {"walltime": ok})
        assert c.argv[-3:] == ["--walltime", ok, "decks"]
    # a field typed into twice (found over the HPCC check: sbatch refused it after the decks were recorded)
    for wrong in ("0:10:000:10:00", "1-2-3", ":30", "1:2:3:4", "1--00", "-1:00"):
        with pytest.raises(sub.SubmitError):
            sub.build_commands(root, "slurm", sel, {"walltime": wrong})
    for bad in ({"walltime": "1h; rm"}, {"exclude": {"kind": "nodes", "value": "a b"}},
                {"exclude": {"kind": "type", "value": "$(x)"}}, {"max_running": 2}, {"array": True, "max_running": 0},
                {"progress": "interactive"}, {"progress": "custom"}, {"progress": "nope"}):
        with pytest.raises(sub.SubmitError):
            sub.build_commands(root, "slurm", sel, bad)
    single = sub.selection(root, [d / "a.d12"])
    with pytest.raises(sub.SubmitError) as e:
        sub.build_commands(root, "slurm", single, {"array": True})
    assert "Batch Setup" in str(e.value)
    (c,) = sub.build_commands(root, "local", single, {"cores": 2, "timeout": "10m", "track": True}, dry_run=True)
    assert c.argv == ["run", "-n", "2", "--timeout", "10m", "--track", "--dry-run", "a.d12"]
    assert "--dry-run" not in c.display and "--nosubmit" not in " ".join(c.argv)


def test_selection_refuses_odd_names(tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    (d / "a b.d12").write_text(SP_D12)
    with pytest.raises(sub.SubmitError) as e:
        sub.selection(tmp_path, [d / "a b.d12"])
    assert "rename a b.d12" in str(e.value)


def test_templates_filtered_by_calc_type():
    t = {x["name"]: x for x in sub.templates_for(["SP"])}
    assert "custom" not in t and "interactive" not in t
    assert not t["basic_opt"]["ok"] and "does not contain SP" in t["basic_opt"]["why"]
    assert t["opt_sp"]["ok"]


def test_parse_squeue_folded_and_pipes():
    text = "123|my|job|RUNNING|1:00|7-00:00:00|1|32|n001|/w/a b\n4242_3|m0|PENDING|0:00|1:00:00|1|32|Priority|/w\n" \
           "bad line\n"
    rows = jobs.parse_squeue(text)
    assert rows[0]["name"] == "my|job" and rows[0]["workdir"] == "/w/a b"
    assert rows[1]["id"] == "4242_3" and rows[1]["reason"] == "Priority"
    assert len(rows) == 2


def test_db_backup_copy_while_writing(tmp_path):
    db = tmp_path / "materials.db"
    con = sqlite3.connect(db)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE t (x INTEGER, pad TEXT)")
    con.commit()
    stop = threading.Event()

    def writer():
        c = sqlite3.connect(db, timeout=10)
        i = 0
        while not stop.is_set():
            c.execute("INSERT INTO t VALUES (?, ?)", (i, "x" * 2000))
            c.commit()
            i += 1
            # a writer as fast as a job's completion step, not a tight loop: unthrottled, the
            # database grows by hundreds of MB a second on tmpfs, past dbsafety's one-step
            # size, and the stepped backup restarts until the disk is full
            time.sleep(0.002)
    th = threading.Thread(target=writer)
    th.start()
    try:
        time.sleep(0.3)
        for k in range(3):
            dst = tmp_path / f"copy{k}.db"
            sub._copy_db(db, dst)
            c = sqlite3.connect(dst)
            assert c.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert c.execute("SELECT count(*) FROM t").fetchone()[0] > 0
            c.close()
    finally:
        stop.set()
        th.join()


def test_staging_bound(tmp_path):
    d = tmp_path / "decks"
    d.mkdir()
    (d / "a.d12").write_text(SP_D12)
    (d / "a.f9").write_bytes(b"\0" * 4096)
    sel = sub.selection(tmp_path, [d])
    items, size = sub.stage_plan(sel, tmp_path)
    assert ("stub" in [h for p, h in items if p.name == "a.f9"])
    assert size < 4096                         # the wavefunction is a zero-byte stand-in
    with pytest.raises(sub.SubmitError):
        sub.stage(sel, tmp_path, tmp_path / "m", budget_free=100)
    sub.stage(sel, tmp_path, tmp_path / "m2")
    assert (tmp_path / "m2" / "decks" / "a.f9").stat().st_size == 0


def test_normalize_rules():
    t = "job 900000123 wf_abc123def workflow_20261006_101010 2026-10-06T10:10:10 20261006_101010 /m/x"
    out = sub.normalize(t, Path("/m"), Path("/r"))
    assert out == "job <JOB> <WF> <WF> <TS> <TS> /r/x"


def test_site_check_folder_removal_touches_only_its_own_temp_folder(tmp_path):
    """The Check button removes the folder `mace site check` kept (no SLURM
    on the machine), and nothing else a report might name."""
    from mace.gui.submit_routes import drop_check_folder
    own = tmp_path / "mace-site-check-abc123"
    own.mkdir()
    (own / "sp_check.sh").write_text("#!/bin/bash\n")
    other = tmp_path / "project"
    other.mkdir()
    (other / "keep.d12").write_text("x")
    link = tmp_path / "mace-site-check-link"
    link.symlink_to(other)
    assert drop_check_folder(str(own)) is True and not own.exists()
    for bad in (str(other), str(link), "mace-site-check-rel", "", None, str(tmp_path / "mace-site-check-gone")):
        assert drop_check_folder(bad) is False
    assert (other / "keep.d12").read_text() == "x" and link.is_symlink()


def test_db_copy_and_backup_load_dbsafety_without_the_database_package(tmp_path):
    # The GUI loads dbsafety through dbsafety_loader everywhere else, so the
    # server never imports mace.database (MaterialDatabase, ASE, pyarrow).
    repo = Path(sub.__file__).resolve().parents[2]
    code = f"""
import sqlite3, sys
from pathlib import Path
sys.path.insert(0, {str(repo)!r})
from mace.gui import submit
db = Path({str(tmp_path)!r}) / "materials.db"
c = sqlite3.connect(db); c.execute("CREATE TABLE t (x)"); c.commit(); c.close()
submit._copy_db(db, Path({str(tmp_path)!r}) / "copy" / "materials.db")
made = submit._backup_db(db)
assert made is not None and Path(made).is_file(), made
print("mace.database" in sys.modules, "mace.database.dbsafety" in sys.modules)
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       timeout=120, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["False", "True"]
