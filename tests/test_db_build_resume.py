"""A database build that stops before its end is continued by the next one
(MACE 1.1.6; the GUI's server can be stopped by a dev node's CPU-time limit).

The build already commits each output with its fingerprint, so the next
``--skip-unchanged`` build skips what the stopped one finished: these tests
check that it does, and that it says so ("Resumed: N of M already done") in
its progress JSON, its summary and its terminal output, through the real
``mace_cli`` (killed with SIGKILL, what the kernel sends at the hard CPU-time
limit, after the third output was committed). The GUI side (dbbuild.build)
reads the same records and words a CPU-time-limit stop.
Real corpus outputs (skipped without the corpus)."""
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [p for p in (str(REPO),) if p not in sys.path]

from mace.utils import db_build  # noqa: E402

RUNS = ["OPT/1_dia_opt_rev1", "SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized"]


def _env():
    e = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    e["OMP_NUM_THREADS"] = "1"
    return e


def _argv(*extra):
    return [sys.executable, str(REPO / "mace_cli"), "--no-banner", "analyze", "--extract-properties", "runs",
            "--db-path", "runs/materials.db", "--register", "--progress-json", *extra]


@pytest.fixture()
def runs(tmp_path):
    from conftest import find_data
    srcs = [find_data(r + ".out") for r in RUNS]
    for k in range(5):
        d = tmp_path / "runs" / f"r{k}"
        d.mkdir(parents=True)
        for s in srcs:
            shutil.copy2(s, d / s.name)
            shutil.copy2(s.with_suffix(".d12"), d / s.with_suffix(".d12").name)
    return tmp_path


def _records(stdout):
    return [json.loads(x) for x in stdout.splitlines() if x.startswith("{")]


def _killed_after(cwd, n_extracted, *extra):
    """The build, killed (SIGKILL) as soon as it reports its n-th extracted output."""
    p = subprocess.Popen(_argv(*extra), cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                         env=_env())
    seen = 0
    try:
        for line in p.stdout:
            if line.startswith("{") and json.loads(line).get("status") == "extracted":
                seen += 1
                if seen == n_extracted:
                    p.send_signal(signal.SIGKILL)
                    break
    finally:
        p.wait(timeout=600)
    assert p.returncode == -signal.SIGKILL, "the build ended before it could be stopped"
    return seen


def test_marker_functions(tmp_path):
    db = tmp_path / "materials.db"
    m = db_build.build_marker(db)
    assert m == tmp_path / ".materials.db.building"
    assert db_build.begin_build(db, tmp_path, 7) is None and m.is_file()
    assert json.loads(m.read_text())["outputs"] == 7
    prev = db_build.begin_build(db, tmp_path, 9)                 # a second start: the first never ended
    assert prev["outputs"] == 7
    db_build.end_build(db)
    assert not m.exists()
    db_build.end_build(db)                                       # no marker: nothing to do
    m.write_text("not json")
    assert db_build.begin_build(db, tmp_path, 1) == {}           # an unreadable marker still means "stopped"
    assert db_build.resumed_message(3, 10) == "Resumed: 3 of 10 already done"
    msg = db_build.summary_message({"new": 7, "changed": 0, "resumed": {"already": 3, "total": 10}})
    assert msg.startswith("Resumed: 3 of 10 already done. Database updated: 7 new")


def test_a_killed_build_is_continued_and_says_so(runs):
    cwd = runs
    done_before = _killed_after(cwd, 3, "--skip-unchanged")
    assert (cwd / "runs" / ".materials.db.building").is_file()
    c = sqlite3.connect(str(cwd / "runs" / "materials.db"))
    try:
        assert c.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        committed = c.execute("SELECT count(*) FROM files WHERE file_type = 'output_extracted'").fetchone()[0]
    finally:
        c.close()
    assert committed >= done_before == 3
    # the next build continues: what the killed one committed is unchanged and skipped
    p = subprocess.run(_argv("--skip-unchanged"), cwd=str(cwd), capture_output=True, text=True, env=_env(), timeout=900)
    assert p.returncode == 0, p.stderr[-2000:]
    recs = _records(p.stdout)
    resumed = [r["resumed"] for r in recs if "resumed" in r and not r.get("done")]
    assert resumed == [{"already": committed, "total": 10}], resumed
    assert recs.index(next(r for r in recs if "resumed" in r)) == 0          # before the first file's record
    files = [r for r in recs if "i" in r]
    assert sum(1 for r in files if r["status"] == "skipped" and r["class"] == "unchanged") == committed
    assert sum(1 for r in files if r["status"] == "extracted") == 10 - committed
    done = recs[-1]
    assert done["done"] and done["resumed"] == resumed[0]
    assert done["message"].startswith(f"Resumed: {committed} of 10 already done. Database updated: {10 - committed} new")
    assert f"↻ Resumed: {committed} of 10 already done" in p.stderr
    assert not (cwd / "runs" / ".materials.db.building").exists()
    # a build after a complete one is no resume
    p = subprocess.run(_argv("--skip-unchanged"), cwd=str(cwd), capture_output=True, text=True, env=_env(), timeout=900)
    recs = _records(p.stdout)
    assert p.returncode == 0 and not any("resumed" in r for r in recs) and "resumed" not in recs[-1]
    assert "Resumed" not in p.stderr


def test_without_skip_unchanged_the_next_build_says_it_starts_over(runs):
    _killed_after(runs, 1)
    p = subprocess.run(_argv(), cwd=str(runs), capture_output=True, text=True, env=_env(), timeout=900)
    assert p.returncode == 0, p.stderr[-2000:]
    assert "stopped before its end; this run extracts every output again" in p.stderr
    recs = _records(p.stdout)
    assert not any("resumed" in r for r in recs) and not (runs / "runs" / ".materials.db.building").exists()


def test_dry_run_leaves_no_marker(runs):
    p = subprocess.run([a for a in _argv("--dry-run", "--json") if a != "--progress-json"], cwd=str(runs),
                       capture_output=True, text=True, env=_env(), timeout=900)
    assert p.returncode == 0, p.stderr[-2000:]
    assert not (runs / "runs" / ".materials.db.building").exists()


# ---------------------------------------------------------------- the GUI's build task reads the same records

class _Result:
    def __init__(self, code, cpu_limit):
        self.returncode, self.cpu_limit, self.stderr, self.stdout, self.log = code, cpu_limit, "", "", ""


def _fake_run(lines, code=0, cpu_limit=False):
    def run(argv, cwd, ctx=None, timeout=None, on_line=None):
        for x in lines:
            on_line(json.dumps(x))
        return _Result(code, cpu_limit)
    return run


def test_gui_build_reports_the_resume(monkeypatch, tmp_path):
    from mace.gui import clirun, dbbuild
    lines = [{"resumed": {"already": 2, "total": 3}},
             {"i": 1, "n": 3, "file": "a.out", "status": "skipped", "class": "unchanged"},
             {"i": 2, "n": 3, "file": "b.out", "status": "skipped", "class": "unchanged"},
             {"i": 3, "n": 3, "file": "c.out", "status": "extracted", "class": "new", "material": "c"},
             {"done": True, "new": 1, "changed": 0, "resumed": {"already": 2, "total": 3},
              "message": "Resumed: 2 of 3 already done. Database updated: 1 new, 0 changed, 0 failed outputs; 1 materials (0 new)"}]
    monkeypatch.setattr(clirun, "run", _fake_run(lines))
    seen = []
    opts = {"register": True, "skip_unchanged": True, "backup": False, "keep": 3}
    out = dbbuild.build(tmp_path, ".", "materials.db", opts, None, lambda f, t: seen.append(t))
    assert seen[0] == "Resumed: 2 of 3 already done"
    assert out["resumed"] == {"already": 2, "total": 3} and out["message"].startswith("Resumed: 2 of 3 already done.")


def test_gui_build_words_a_cpu_time_limit_stop(monkeypatch, tmp_path):
    from mace.gui import clirun, dbbuild
    lines = [{"i": 1, "n": 9, "file": "a.out", "status": "extracted", "class": "new"},
             {"i": 2, "n": 9, "file": "b.out", "status": "extracted", "class": "new"}]
    monkeypatch.setattr(clirun, "run", _fake_run(lines, code=-9, cpu_limit=True))
    opts = {"register": True, "skip_unchanged": True, "backup": False, "keep": 3}
    with pytest.raises(RuntimeError, match=r"CPU-time limit after 2 files; what it wrote is kept\. Run it again with "
                                           r"'Only new and changed files' to continue where it stopped"):
        dbbuild.build(tmp_path, ".", "materials.db", opts, None, lambda f, t: None)
