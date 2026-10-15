"""Submit from the GUI through the real `mace gui` (PLAN_submit D1, D1b, D2,
D15, D16, D23, D24, D26, D38): the preview runs `mace submit` in a staged copy
with the recording shim (read-only SLURM queries reach the fake SLURM, state
changes are recorded), equals what the real run then submits, and equals
what a shell user's `mace submit` does in a twin folder."""
import json
import os
import shutil
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

import pytest

import submit_util as SU
from submit_util import OPT_D12, SP_D12, BAND_D3, norm


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    s = SU.Served(tmp_path_factory.mktemp("submit"))
    yield s
    s.stop()


def _sbatch_and_control(calls):
    return [c["argv"] for c in calls if c["tool"] in ("sbatch", "scancel") or
            (c["tool"] == "scontrol" and c["argv"][1:2] != ["show"])]


# ---------------------------------------------------------------- D1
@pytest.mark.parametrize("case,options", [
    ("untracked", {}),
    ("track", {"track": True}),
    ("array", {"array": True, "max_running": 2}),
    ("progress", {"progress": "opt_sp"}),
])
def test_preview_equals_real(srv, tmp_path, case, options):
    folder = f"d1_{case}"
    deck = OPT_D12 if case == "progress" else SP_D12
    d = srv.decks(f"{folder}/decks", ["m0", "m1", "m2"], deck)
    srv.clear_calls()
    st, prev = srv.preview([f"{folder}/decks"], options=options)
    assert st == 200 and prev["code"] == 0, prev.get("cli_log")
    # the preview reached SLURM only with read-only queries
    assert not [c for c in srv.calls() if c["tool"] in ("sbatch", "scancel")
                or (c["tool"] == "scontrol" and c["argv"][1:2] != ["show"])]
    assert prev["n_jobs"] == 3
    flags = {"untracked": "", "track": "--track ", "array": "--array --max-running 2 ",
             "progress": "--progress opt_sp "}[case]
    assert prev["command"] == f"cd {folder} && mace submit {flags}decks"
    srv.clear_calls()
    st, res = srv.run(prev["preview_id"])
    assert st == 200 and res["code"] == 0, res
    assert res["script_drift"] == [], res["script_drift"]
    real_calls = _sbatch_and_control(srv.calls())
    # the preview recorded the same state changes the real run made (ids normalized)
    assert sorted(norm(a, srv.root) for a in prev["intercepted"]) == sorted(norm(a, srv.root) for a in real_calls)
    # the preview's job scripts are the real ones
    for rel, text in prev["scripts"].items():
        assert norm(text, srv.root) == norm((srv.root / rel).read_text(), srv.root), rel
    # ... and a shell user's `mace submit` in a twin folder writes the same scripts and calls
    twin = tmp_path / "twin"
    shutil.copytree(srv.root / folder, twin / folder)
    for p in (twin / folder / "decks").glob("*"):
        if p.suffix not in (".d12",):
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    fake2 = SU.FakeSlurm(tmp_path / "fs2")
    args = prev["argv"][0][1:]
    r = SU.twin_submit(tmp_path, fake2, twin / folder, args)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    # (the decks' order is the file system's: compared sorted, as the default-path goldens do)
    assert sorted(norm(a, twin) for a in _sbatch_and_control(fake2.calls())) == \
        sorted(norm(a, srv.root) for a in real_calls)
    for sh in sorted((d).glob("*.sh")):
        assert norm(sh.read_text(), srv.root) == norm((twin / folder / "decks" / sh.name).read_text(), twin), sh.name
    real_new = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
    twin_new = sorted(p.relative_to(twin / folder / "decks").as_posix() for p in (twin / folder / "decks").rglob("*")
                      if p.is_file())
    assert [norm(x) for x in real_new] == [norm(x) for x in twin_new]


def test_untracked_preview_equals_nosubmit(srv, tmp_path):
    srv.decks("d1_nosub/decks", ["a", "b"])
    st, prev = srv.preview(["d1_nosub/decks"], options={"walltime": "0:30:00"})
    assert st == 200 and prev["code"] == 0
    twin = tmp_path / "t"
    shutil.copytree(srv.root / "d1_nosub", twin)
    r = SU.twin_submit(tmp_path, SU.FakeSlurm(tmp_path / "fs3"), twin, ["submit", "--nosubmit", "--walltime", "0:30:00",
                                                                         "decks"])
    assert r.returncode == 0, r.stdout + r.stderr
    for rel, text in prev["scripts"].items():
        name = Path(rel).name
        assert norm(text, srv.root) == norm((twin / "decks" / name).read_text(), twin)
    assert "#SBATCH -t 0:30:00" in prev["scripts"]["d1_nosub/decks/a.sh"]


# ---------------------------------------------------------------- D1b live state through the shim
def test_preview_reads_live_state_exclude(srv):
    srv.fake.update(nodes="NodeName=amr-001 Arch=x86_64 Features=amr\nNodeName=amr-002 Arch=x86_64 Features=amr\n"
                          "NodeName=nvf-001 Arch=x86_64 Features=nvf\n")
    srv.decks("d1b_ex/decks", ["a"])
    st, prev = srv.preview(["d1b_ex/decks"], options={"exclude": {"kind": "type", "value": "amr"}})
    assert st == 200 and prev["code"] == 0, prev["cli_log"]
    assert any(c["argv"][:3] == ["scontrol", "show", "nodes"] for c in srv.calls("scontrol"))
    line = [l for l in prev["scripts"]["d1b_ex/decks/a.sh"].splitlines() if "--exclude" in l]
    st, res = srv.run(prev["preview_id"])
    assert res["code"] == 0 and res["script_drift"] == []
    real = [l for l in (srv.root / "d1b_ex/decks/a.sh").read_text().splitlines() if "--exclude" in l]
    assert line == real


def test_preview_reads_live_state_split(srv):
    srv.fake.update(max_array_tasks=3, max_array_size=3)
    try:
        srv.decks("d1b_split/decks", [f"m{i}" for i in range(5)])
        st, prev = srv.preview(["d1b_split/decks"], options={"array": True})
        assert st == 200 and prev["code"] == 0, prev["cli_log"]
        split = sorted(a["tasks"] for a in prev["arrays"])
        assert len(split) > 1 and sum(split) == 5 and max(split) < 5      # the cluster's limit split the decks
        srv.clear_calls()
        st, res = srv.run(prev["preview_id"])
        holds = [c["argv"] for c in srv.calls("sbatch")]
        assert len(holds) == len(split) and res["script_drift"] == []
        assert len(res["job_ids"]) == 5
    finally:
        srv.fake.update(max_array_tasks=10000, max_array_size=4000001)


def test_preview_reads_live_state_limit(srv):
    from mace.submission import site
    # the user's default association (the job script has no -A line on this
    # test cluster, whose accounting lists nothing) under the built-in account
    acct = site.DEFAULTS["account"]
    srv.fake.update(assoc_mgr=f"ClusterName=c Account={acct} UserName=tester(1000) Partition= ID=1 "
                              "DefAssoc=Yes MaxSubmitJobs=2(2)")
    try:
        srv.decks("d1b_lim/decks", ["a", "b"])
        st, prev = srv.preview(["d1b_lim/decks"], options={"array": True})
        assert st == 200
        twin_text = prev["cli_log"]
        assert prev["code"] != 0 and "submit" in twin_text.lower()
        st, res = srv.run(prev["preview_id"])
        assert st == 409                      # a failed preview submits nothing
    finally:
        srv.fake.update(assoc_mgr="")


# ---------------------------------------------------------------- D2 fresh previews only
def test_run_requires_fresh_preview(srv):
    d = srv.decks("d2/decks", ["a"])
    srv.clear_calls()
    assert srv.run("nope")[0] == 409
    st, prev = srv.preview(["d2/decks/a.d12"])
    pid = prev["preview_id"]
    # edited deck
    (d / "a.d12").write_text(SP_D12.replace("8 8", "6 6"))
    st, res = srv.run(pid)
    assert st == 409 and "changed since the preview" in res["error"]
    assert not srv.calls("sbatch")
    # reused
    st, prev = srv.preview(["d2/decks/a.d12"])
    st, res = srv.run(prev["preview_id"])
    assert st == 200 and res["code"] == 0
    st, res = srv.run(prev["preview_id"])
    assert st == 409 and "already used" in res["error"]
    # a .sh added since
    (d / "b.d12").write_text(SP_D12)
    st, prev = srv.preview(["d2/decks/b.d12"])
    (d / "b.sh").write_text("#!/bin/bash\n")
    assert srv.run(prev["preview_id"])[0] == 409
    (d / "b.sh").unlink()
    # an .out appeared
    st, prev = srv.preview(["d2/decks/b.d12"])
    (d / "b.out").write_text(" EEEEEEEEEE TERMINATION\n")
    st, res = srv.run(prev["preview_id"])
    assert st == 409
    (d / "b.out").unlink()
    # the folder changed (Open Folder)
    st, prev = srv.preview(["d2/decks/b.d12"])
    srv.req("api/test/root_gen", {})
    st, res = srv.run(prev["preview_id"])
    assert st == 409 and "folder changed" in res["error"]


def test_root_gen_on_preview_and_cancel(srv):
    srv.decks("d24/decks", ["a"])
    st, _, info = srv.req("api/info")
    st, _, r = srv.req("api/submit/preview", {"targets": ["d24/decks/a.d12"], "root_gen": int(info["root_gen"]) + 7})
    assert st == 409 and "folder changed" in r["error"]
    st, _, r = srv.req("api/jobs/cancel", {"ids": ["1"], "root_gen": -5})
    assert st == 409


# ---------------------------------------------------------------- D23 overwrite / duplicate
def test_conflicts_finished_wavefunction_queued(srv):
    from conftest import TEST_DATA
    d = srv.decks("d23/decks", ["m"])
    real = TEST_DATA / "OPT" / "1_dia_opt_rev1.out"
    if real.is_file():
        shutil.copy2(real, d / "m.out")
    else:
        (d / "m.out").write_text(SP_D12)
    (d / "m.f9").write_bytes(b"\0" * 1000)
    srv.fake.add_record(id="7001", name="m", state="RUNNING", user=SU.USER, workdir=str(d))
    st, prev = srv.preview(["d23/decks/m.d12"])
    c = prev["conflicts"]
    assert c["finished"][0]["deck"] == "m.d12" and c["finished"][0]["bucket"]
    assert c["wavefunction"][0]["rel"] == "d23/decks/m.f9"
    assert c["queued"][0]["job_id"] == "7001"
    assert prev["needs"]["overwrite"] and prev["needs"]["duplicate"]
    assert "d23/decks/m.out" in prev["job_writes"]
    st, res = srv.run(prev["preview_id"])
    assert st == 409 and res.get("need") == "overwrite"
    st, prev = srv.preview(["d23/decks/m.d12"])
    st, res = srv.run(prev["preview_id"], confirm_overwrite=True)
    assert st == 409 and res.get("need") == "duplicate"
    srv.clear_calls()
    st, prev = srv.preview(["d23/decks/m.d12"])
    st, res = srv.run(prev["preview_id"], confirm_overwrite=True, confirm_duplicate=True)
    assert st == 200 and res["code"] == 0 and srv.calls("sbatch")
    assert (d / "m.f9").exists()                     # copied for history, never moved
    srv.fake.remove_record("7001")


def test_rows_sbatch_refused_are_not_queued(tmp_path):
    """mace submit --track records the decks, then sbatch may refuse the job
    (found on HPCC with a bad --time): "recorded without a job id", status
    pending. Such a row is no queued job; one with a job id still is."""
    from mace.gui import submit as sub
    d = tmp_path / "decks"
    d.mkdir()
    decks = [d / "a.d12", d / "b.d12", d / "c.d12"]
    for x in decks:
        x.write_text(SP_D12)
    db = tmp_path / "materials.db"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE calculations (calc_id TEXT, input_file TEXT, status TEXT, slurm_job_id TEXT)")
        con.executemany("INSERT INTO calculations VALUES (?, ?, ?, ?)", [
            ("a_SP", str(decks[0]), "pending", None), ("b_SP", str(decks[1]), "pending", "8100_1"),
            ("c_SP", str(decks[2]), "submitted", None)])
    rows = sub.queued_rows([db], decks)
    assert sorted(r["calc_id"] for r in rows) == ["b_SP", "c_SP"]


# ---------------------------------------------------------------- D15 tracked rows beside the parents
def test_tracked_rows_land_in_the_folders_db(srv):
    d = srv.decks("d15/batch_SP/decks", ["a", "b", "c"])
    st, prev = srv.preview(["d15/batch_SP/decks"], options={"array": True, "max_running": 1, "track": True})
    assert prev["command"] == "cd d15 && mace submit --array --max-running 1 --track batch_SP/decks"
    st, res = srv.run(prev["preview_id"])
    assert res["code"] == 0
    db = srv.root / "d15" / "materials.db"
    with sqlite3.connect(db) as con:
        rows = con.execute("SELECT input_file, slurm_job_id FROM calculations").fetchall()
    assert sorted(Path(r[0]).name for r in rows) == ["a.d12", "b.d12", "c.d12"]
    assert all("_" in r[1] for r in rows)


# ---------------------------------------------------------------- D16
def test_refused_on_network_bind(tmp_path):
    import sys as _sys
    s = SU.Served.__new__(SU.Served)
    s.tmp = tmp_path
    s.root = tmp_path / "proj"
    s.root.mkdir()
    s.fake = SU.FakeSlurm(tmp_path / "fs")
    s.env = SU.fake_env(s.fake, tmp_path)
    s.env.pop("XDG_CACHE_HOME")
    port = SU.H.free_port()
    env = SU.H.env(tmp_path / "xdg", MACE_GUI_NO_WARMUP="1", **s.env)
    proc = SU.H.ServerProcess([_sys.executable, str(SU.MACE_CLI), "--no-banner", "gui", "--dir", str(s.root), "--port",
                               str(port), "--token", SU.H.TOKEN, "--host", "0.0.0.0", "--insecure-network",
                               "--allow-host", "127.0.0.1"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, env=env, start_new_session=True)
    try:
        base = None
        for _ in range(60):
            line = proc.stdout.readline()
            if "/s/" in line:
                import re
                m = re.search(r"(http://[^/\s]+:\d+/s/[A-Za-z0-9_-]+/)", line)
                if m:
                    base = m.group(1).replace("0.0.0.0", "127.0.0.1")
                    break
        assert base
        s.base, s.proc = base, proc
        s.decks("n/decks", ["a"])
        st, prev = s.preview(["n/decks/a.d12"])
        assert "Submitting is turned off while MACE GUI is open to the network" in prev["warnings"]
        st, res = s.run(prev["preview_id"])
        assert st == 409 and "open to the network" in res["error"]
        assert not s.fake.calls("sbatch")
        st, _, res = s.req("api/jobs/cancel", {"ids": ["12345"]})
        assert st == 409 and "open to the network" in res["error"]
        assert not s.fake.calls("scancel")
    finally:
        proc.terminate()
        proc.wait(timeout=30)


# ---------------------------------------------------------------- D26 / D38
def test_staging_bound_and_mixed_folder(srv):
    from mace.gui import submit as sub
    d = srv.decks("d26/decks", ["a"])
    sel = sub.selection(srv.root, [d])
    with pytest.raises(sub.SubmitError) as e:
        sub.stage(sel, srv.root, srv.tmp / "m", budget_free=10)
    assert "would copy" in str(e.value)
    (d / "b_band.d3").write_text(BAND_D3)
    # Since 1f90ee9a the CLI refuses a .d3 whose .f9 is missing or empty, so
    # the folder carries the SCF wavefunction; the mirror's stand-in for it
    # must not read as empty, or the preview refuses what the real run submits.
    (d / "b_band.f9").write_bytes(b"\0" * 64)
    st, _, r = srv.req("api/submit/preview", {"targets": ["d26/decks"]})
    assert st == 400 and "choose which" in r["error"]
    st, prev = srv.preview(["d26/decks"], mixed="3")
    assert prev["command"] == "cd d26 && echo 3 | mace submit decks" and prev["code"] == 0, prev["cli_log"]
    assert prev["stdin"] == "3\n"
    (d / "b_band.f9").write_bytes(b"")             # an empty .f9 is refused in the preview as in the real run
    st, prev = srv.preview(["d26/decks"], mixed="3")
    assert prev["code"] != 0 and "b_band.f9 is empty" in prev["cli_log"], prev["cli_log"]


@pytest.mark.parametrize("mode", ["wal", "delete"])
def test_growing_files_no_409(srv, mode):
    d = srv.decks(f"d38{mode}/decks", ["g"])
    out = d / "g.out"
    out.write_text("running\n")
    db = d.parent / "materials.db"
    con = sqlite3.connect(db)
    con.execute(f"PRAGMA journal_mode={'WAL' if mode == 'wal' else 'DELETE'}")
    con.execute("CREATE TABLE IF NOT EXISTS calculations (calc_id TEXT, input_file TEXT, status TEXT, slurm_job_id TEXT)")
    con.commit()
    con.close()
    stop = threading.Event()

    def writer():
        i = 0
        while not stop.is_set():
            with open(out, "a") as fh:
                fh.write(f" CYCLE {i}\n")
            if i % 3 == 0:
                c = sqlite3.connect(db, timeout=5)
                c.execute("INSERT INTO calculations VALUES (?, ?, 'completed', NULL)", (f"x{i}", "/elsewhere.d12"))
                c.commit()
                c.close()
            i += 1
            time.sleep(0.2)
    t = threading.Thread(target=writer, daemon=True)
    t.start()
    try:
        st, prev = srv.preview([f"d38{mode}/decks/g.d12"])
        time.sleep(1.0)
        st, res = srv.run(prev["preview_id"], confirm_overwrite=True)
        assert st == 200 and res["code"] == 0, res
    finally:
        stop.set()
        t.join()
    st, prev = srv.preview([f"d38{mode}/decks/g.d12"])
    (d / "g.d12").write_text(SP_D12 + "\n")
    st, res = srv.run(prev["preview_id"], confirm_overwrite=True)
    assert st == 409


# ---------------------------------------------------------------- local runs (D8 preview part)
def test_local_preview_is_a_dry_run(srv):
    d = srv.decks("loc/decks", ["a"])
    before = sorted(p.name for p in d.rglob("*"))
    st, prev = srv.preview(["loc/decks/a.d12"], where="local", options={"cores": 2, "timeout": "10m"})
    assert st == 200, prev
    assert prev["command"] == "cd loc/decks && mace run -n 2 --timeout 10m a.d12"
    assert sorted(p.name for p in d.rglob("*")) == before            # nothing written
    if prev["code"]:
        # the refusal is the CLI's own words (never typed by the GUI)
        r = SU.twin_submit(srv.tmp, srv.fake, d, ["run", "--dry-run", "-n", "2", "--timeout", "10m", "a.d12"])
        assert prev["refused"] and prev["refused"] in (r.stderr + r.stdout)


# ---------------------------------------------------------------- site settings (D34 API part)
def test_site_change_409(srv):
    srv.decks("site/decks", ["a"])
    cfg = srv.tmp / "xdg" / "mace" / "site.toml"
    st, prev = srv.preview(["site/decks/a.d12"])
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text('[slurm]\npartition = "p1"\n')
    try:
        st, res = srv.run(prev["preview_id"])
        assert st == 409 and "Site settings changed since the preview" in res["error"]
        st, prev = srv.preview(["site/decks/a.d12"])
        assert "#SBATCH -p p1" in prev["scripts"]["site/decks/a.sh"]
        st, _, caps = srv.req("api/submit/capabilities")
        assert caps["site"]["partition"]["value"] == "p1"
    finally:
        cfg.unlink()


def test_shutdown_waits_for_a_submission(tmp_path):
    """A submit cannot stop halfway: stopping the server waits for it."""
    s = SU.Served(tmp_path, {"sbatch_delay": 3})
    s.decks("w/decks", ["a"])
    st, prev = s.preview(["w/decks/a.d12"])
    r = s.req("api/submit/run", {"preview_id": prev["preview_id"]})
    assert r[0] in (200, 202)
    time.sleep(0.8)
    t0 = time.time()
    s.proc.terminate()
    s.proc.wait(timeout=60)
    assert time.time() - t0 > 1.0                          # it waited for sbatch
    out = s.proc.stdout.read()
    assert "waiting for a submission to finish" in out
    assert len(s.fake.calls("sbatch")) == 1 and (s.root / "w" / "decks" / "a.sh").is_file()
