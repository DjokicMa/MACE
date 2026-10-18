"""`mace analyze --extract-properties DIR`, as the docs show it, fills the
database `mace database` reads (1.1.6 review, E-H2).

Without --register the command stored properties with no materials or
calculations rows while reporting counts; `mace database` then listed 0
materials and every export printed a bare "Export failed.". A folder is
registered by default now; --no-register keeps the untracked extraction, and
its empty export says why.
"""
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA

FILES = ["OPT/1_dia_opt_rev1.out", "SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out"]


def _env():
    e = {k: v for k, v in os.environ.items()
         if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH",
                      "MACE_DB_JOURNAL", "MACE_DB_LOCKING")}
    e["OMP_NUM_THREADS"] = "1"
    return e


def _mace(cwd, *args):
    return subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", *args],
                          capture_output=True, text=True, env=_env(), cwd=str(cwd), timeout=900)


@pytest.fixture
def runs(tmp_path):
    if not TEST_DATA.is_dir():
        pytest.skip("test/ data corpus not present (gitignored)")
    d = tmp_path / "runs"
    d.mkdir()
    for r in FILES:
        src = TEST_DATA / r
        shutil.copyfile(src, d / src.name)
        if src.with_suffix(".d12").is_file():
            shutil.copyfile(src.with_suffix(".d12"), d / src.with_suffix(".d12").name)
    return d


def _counts(db):
    c = sqlite3.connect(str(db))
    try:
        return {t: c.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                for t in ("materials", "calculations", "properties")}
    finally:
        c.close()


def test_folder_extraction_registers_by_default(runs, tmp_path):
    db = tmp_path / "materials.db"
    p = _mace(tmp_path, "analyze", "--extract-properties", str(runs), "--db-path", str(db))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    n = _counts(db)
    assert n["materials"] >= 1 and n["calculations"] == 2 and n["properties"] > 0, n
    out = tmp_path / "export.csv"
    e = _mace(tmp_path, "database", "--db-path", str(db), "--action", "export", "--format", "csv",
              "--output", str(out))
    assert e.returncode == 0 and out.is_file(), e.stdout[-3000:] + e.stderr[-3000:]
    assert "1_dia" in out.read_text()


def test_no_register_keeps_untracked_and_export_says_why(runs, tmp_path):
    db = tmp_path / "materials.db"
    p = _mace(tmp_path, "analyze", "--extract-properties", str(runs), "--db-path", str(db),
              "--no-register")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    n = _counts(db)
    assert n["calculations"] == 0 and n["properties"] > 0
    assert "no calculation record" in p.stdout and "--register" in p.stdout
    e = _mace(tmp_path, "database", "--db-path", str(db), "--action", "export", "--format", "csv",
              "--output", str(tmp_path / "x.csv"))
    text = e.stdout + e.stderr
    assert e.returncode != 0
    assert "Export failed: the database has no materials" in " ".join(text.split()), text[-2000:]
