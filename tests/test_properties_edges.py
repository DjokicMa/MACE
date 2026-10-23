"""The Properties tab's edges (1.1.6 review): Extract registers the one run it
names, never the rest of its folder tree (mace analyze --register-output-only,
run as the CLI); a job log named like an output (slurm-123.out, X-123.out) is
not a CRYSTAL run; a run the database records by a path relative to its folder
is found; nothing outside the served folder is used or named; a file that is
not a database is refused (400, not 500); and a database that cannot be written
fails the Relaxation Time dialog's Save to Database, never its fit."""
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from mace.gui import properties_routes as PR

EXAMPLES = REPO_ROOT / "examples" / "gui"
SI = EXAMPLES / "outputs" / "Si_sp.out"
STRAIN = REPO_ROOT / "tests" / "data" / "hpcc_116" / "strain"


def _env():
    e = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    e["OMP_NUM_THREADS"] = "1"
    return e


def _cli(cwd, *args, code=0):
    p = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", *args], capture_output=True,
                       text=True, env=_env(), cwd=str(cwd), timeout=900)
    assert p.returncode == code, p.stdout[-2000:] + p.stderr[-2000:]
    return p


class St:
    """The server state the routes use: the served folder and its path rules."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.test_hooks = False

    def resolve(self, rel, must_exist=True):
        t = (self.root / rel).resolve()
        if t != self.root and self.root not in t.parents:
            raise PermissionError("outside the served directory")
        if must_exist and not t.exists():
            raise FileNotFoundError(rel)
        return t

    def rel(self, p):
        return Path(p).resolve().relative_to(self.root).as_posix()


def _calcs(db):
    conn = sqlite3.connect(db)
    try:
        return sorted(Path(r[0]).name for r in conn.execute("SELECT output_file FROM calculations"))
    finally:
        conn.close()


def test_extract_registers_the_one_output_not_its_folder_tree(tmp_path):
    run = tmp_path / "runs"
    (run / "deeper").mkdir(parents=True)
    a, b, c = "dia_sp_dp_x_p0pct.out", "dia_sp_dp_x_p1pct.out", "dia_sp_dp_x_m1pct.out"     # completed SP runs
    shutil.copy(STRAIN / a, run / a)
    shutil.copy(STRAIN / b, run / b)
    shutil.copy(STRAIN / c, run / "deeper" / c)
    # the GUI's Extract: that run only
    _cli(tmp_path, *PR.extract_argv(f"runs/{a}", "runs/materials.db"))
    assert "--register-output-only" in PR.extract_argv("a.out", "m.db")
    assert _calcs(run / "materials.db") == [a]
    # the CLI's own --output-file --register is unchanged: the folder and the folders below it
    _cli(tmp_path, "analyze", "--extract-properties", "--output-file", f"runs/{a}", "--db-path", "runs/all.db",
         "--register")
    walked = _calcs(run / "all.db")
    assert len(walked) > 1 and set(walked) <= {a, b, c}, walked
    # the flag goes with --output-file and --register
    p = _cli(tmp_path, "analyze", "--extract-properties", "runs", "--db-path", "runs/x.db", "--register-output-only",
             code=2)
    assert "--register-output-only goes with --output-file and --register" in p.stderr


def test_a_job_log_named_like_an_output_is_not_a_run(tmp_path):
    shutil.copy(SI, tmp_path / "X.out")
    (tmp_path / "X-12345.out").write_text("SLURM_JOB_ID=12345\nsrun: job step finished\n")
    (tmp_path / "slurm-12345.out").write_text("SLURM_JOB_ID=12345\nmodule load crystal\n")
    # a CRYSTAL output whose name only looks like a log stays a run
    shutil.copy(SI, tmp_path / "MOF-5.out")
    assert PR.run_of(tmp_path / "slurm-12345.out", tmp_path) is None
    assert PR.run_of(tmp_path / "X-12345.out", tmp_path) == tmp_path / "X.out"
    assert PR.run_of(tmp_path / "MOF-5.out", tmp_path) == tmp_path / "MOF-5.out"
    assert PR.run_view(St(tmp_path), tmp_path / "slurm-12345.out")["state"] == "no-run"


def test_a_run_recorded_relative_to_the_database_folder_is_found(tmp_path):
    # examples/gui/materials.db records outputs/Si_sp.out, relative to its own folder
    root = tmp_path / "served"
    shutil.copytree(EXAMPLES, root / "gui", ignore=shutil.ignore_patterns("*.py", "README.md"))
    for st in (St(root / "gui"), St(root)):              # served at the database's folder, or above it
        view = PR.run_view(st, root / "gui" / "outputs" / "Si_sp.out")
        assert view["state"] == "db" and view["calc"]["calc_id"].startswith("Si_SP_"), view["state"]
    # Extract from above the database's folder stores the run's values with that calculation (not untracked)
    before = len(view["rows"])
    _cli(root, *PR.extract_argv("gui/outputs/Si_sp.out", "gui/materials.db"))
    after = PR.run_view(St(root), root / "gui" / "outputs" / "Si_sp.out")
    assert after["calc"]["calc_id"] == view["calc"]["calc_id"] and len(after["rows"]) > before + 10
    assert "energy_per_atom_ev" in {r["name"] for r in after["rows"]}


def test_nothing_outside_the_served_folder_is_used_or_named(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    shutil.copy(SI, tmp_path / "Top.out")                 # one folder above the served folder
    (root / "Top.BAND.DAT").write_text("# NKPT 2\n")
    assert PR.run_of(root / "Top.BAND.DAT", root) is None
    view = PR.run_view(St(root), root / "Top.BAND.DAT")
    assert view["state"] == "no-run" and str(tmp_path) not in repr(view)
    # a link beside the file that points out of the folder
    os.symlink(tmp_path / "Top.out", root / "Link.out")
    (root / "Link.BAND.DAT").write_text("# NKPT 2\n")
    assert PR.run_of(root / "Link.BAND.DAT", root) is None


def test_a_file_that_is_not_a_database_is_refused(tmp_path):
    (tmp_path / "notes.db").write_text("not sqlite")
    with pytest.raises(ValueError, match="is not a MACE materials database"):
        PR._value(St(tmp_path), {"db": ["notes.db"], "calc": ["x"], "name": ["energy"]})


def test_a_database_that_cannot_be_written_fails_the_save_not_the_fit(tmp_path, monkeypatch):
    from mace.gui import mechanics_routes as MR
    (tmp_path / "strain").mkdir()
    for f in STRAIN.glob("*.out"):
        shutil.copy(f, tmp_path / "strain" / f.name)

    def locked(*a, **k):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(PR, "save_interactive", locked)
    outs = sorted(f"strain/{p.name}" for p in (tmp_path / "strain").glob("*.out"))
    out = MR.fit(St(tmp_path), {"outputs": outs, "carrier": "electron", "m_star": 0.3, "save_db": True})
    assert out["result"]["tau_fs"] > 0
    assert "database is locked" in out["saved_db"]["error"] and "the fit is kept" in out["saved_db"]["error"]
