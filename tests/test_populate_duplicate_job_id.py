"""Registering an output whose SLURM job id another calculation already has
(bug hunt 1.1.5, CLI/parser round).

The same SP run copied into two folders (its .out, .d12 and -<job>.o) made
the second registration fail with "UNIQUE constraint failed:
calculations.slurm_job_id" after the calculation row was created: the row
stayed 'pending' with no output_file, and the build filed the copy's
properties under it (66 such rows in a full-corpus database).

Now an identical copy of a registered run is the same calculation: not
registered again and not extracted (Skipped). A different output with the
same job id is registered without the job id, with a warning. A
registration that fails part-way leaves no row. Real outputs, real CLI."""
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA

STEM = "SP/1LiFSI-1DEC-conf1_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized"
JOB = "65828713"


def _env():
    e = {k: v for k, v in os.environ.items()
         if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH", "MACE_DB_JOURNAL", "MACE_DB_LOCKING")}
    e["OMP_NUM_THREADS"] = "1"
    return e


def _copies(tmp_path, *names):
    if not TEST_DATA.is_dir():
        pytest.skip("test/ data corpus not present (gitignored)")
    root = tmp_path / "runs"
    for n in names:
        d = root / n
        d.mkdir(parents=True)
        for suffix in (".out", ".d12", f"-{JOB}.o"):
            shutil.copyfile(TEST_DATA / (STEM + suffix), d / (Path(STEM).name + suffix))
    return root


def _build(folder, db, *extra):
    p = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "analyze",
                        "--extract-properties", str(folder), "--db-path", str(db), "--register", *extra],
                       capture_output=True, text=True, env=_env(), cwd=str(folder.parent), timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "UNIQUE constraint" not in p.stdout + p.stderr, p.stdout[-3000:]
    return p


def _q(db, sql, *a):
    c = sqlite3.connect(str(db))
    try:
        return c.execute(sql, a).fetchall()
    finally:
        c.close()


def _no_phantoms(db):
    assert _q(db, "SELECT calc_id FROM calculations WHERE status != 'completed' OR output_file IS NULL") == []


def test_an_identical_copy_is_the_same_calculation(tmp_path):
    root = _copies(tmp_path, "A", "B")
    db = tmp_path / "materials.db"
    p = _build(root, db, "--progress-json")
    _no_phantoms(db)
    calcs = _q(db, "SELECT slurm_job_id, output_file FROM calculations")
    assert len(calcs) == 1 and calcs[0][0] == JOB
    recs = {Path(r["file"]).parent.name: r for r in map(json.loads, (ln for ln in p.stdout.splitlines()
                                                                         if ln.startswith('{"i"')))}
    other = "B" if Path(calcs[0][1]).parent.name == "A" else "A"
    assert recs[other]["status"] == "skipped" and recs[other]["class"] == "skipped"


def test_a_second_build_of_the_copy_adds_nothing(tmp_path):
    """The reported repro: build folder A, then folder B into the same database."""
    root = _copies(tmp_path, "A", "B")
    db = tmp_path / "materials.db"
    _build(root / "A", db)
    n_props = _q(db, "SELECT count(*) FROM properties")[0][0]
    _build(root / "B", db)
    _no_phantoms(db)
    assert _q(db, "SELECT count(*) FROM calculations")[0][0] == 1
    assert _q(db, "SELECT count(*) FROM properties")[0][0] == n_props


def test_a_row_the_old_code_left_for_a_copy_is_completed(tmp_path):
    """1.1.5 before this fix left the copy's row 'pending' with no output
    file (and its properties under it): the next build gives it its output."""
    root = _copies(tmp_path, "A", "B")
    db = tmp_path / "materials.db"
    _build(root / "A", db)
    c = sqlite3.connect(str(db))
    with c:
        mid = c.execute("SELECT material_id FROM calculations").fetchone()[0]
        c.execute("INSERT INTO calculations (calc_id, material_id, calc_type, status, created_at, input_file, work_dir) "
                  "VALUES ('phantom', ?, 'SP', 'pending', '2026-10-01', ?, ?)",
                  (mid, str((root / "B" / (Path(STEM).name + ".d12")).resolve()), str((root / "B").resolve())))
    c.close()
    _build(root / "B", db)
    _no_phantoms(db)
    out, job = _q(db, "SELECT output_file, slurm_job_id FROM calculations WHERE calc_id = 'phantom'")[0]
    assert Path(out).parent.name == "B" and job is None


def test_a_different_output_with_the_job_id_is_registered_without_it(tmp_path):
    root = _copies(tmp_path, "A", "B")
    out_b = root / "B" / (Path(STEM).name + ".out")
    with open(out_b, "a") as fh:
        fh.write("\n")
    db = tmp_path / "materials.db"
    _build(root / "A", db)
    # a row the old code left behind for B: pending, no output file
    c = sqlite3.connect(str(db))
    with c:
        mid = c.execute("SELECT material_id FROM calculations").fetchone()[0]
        c.execute("INSERT INTO calculations (calc_id, material_id, calc_type, status, created_at, input_file, work_dir) "
                  "VALUES ('phantom', ?, 'SP', 'pending', '2026-10-01', ?, ?)",
                  (mid, str((root / "B" / (Path(STEM).name + ".d12")).resolve()), str((root / "B").resolve())))
    c.close()
    p = _build(root / "B", db)
    _no_phantoms(db)
    rows = dict(_q(db, "SELECT calc_id, slurm_job_id FROM calculations"))
    assert rows["phantom"] is None and list(rows.values()).count(JOB) == 1 and len(rows) == 2
    assert f"SLURM job {JOB}" in p.stdout


def test_a_failed_registration_leaves_no_row(tmp_path, monkeypatch):
    """create_calculation succeeds, the status update fails: the row goes."""
    if not TEST_DATA.is_dir():
        pytest.skip("test/ data corpus not present (gitignored)")
    from mace.database.materials import MaterialDatabase
    from mace.database import populate_completed_jobs as pcj
    root = _copies(tmp_path, "A")
    db = MaterialDatabase(str(tmp_path / "materials.db"))

    def boom(*a, **k):
        raise sqlite3.IntegrityError("simulated")
    monkeypatch.setattr(db, "update_calculation_status", boom)
    calcs = pcj.scan_for_completed_calculations(root.resolve())
    assert pcj.populate_database(calcs, db) == 0
    assert _q(tmp_path / "materials.db", "SELECT count(*) FROM calculations")[0][0] == 0
