"""One `--register` run records every output; a second adds nothing.

The material a scanned calculation belongs to is checked against the
structure of the material its name gives (distinct_material_id). That
structure was cached per build, also when none was known yet: a material
first recorded from a BAND output (no geometry) kept "unknown" for the
whole run, so an SP of another structure in a folder of SPs joined it, and
the SPs after it in that folder were taken for the same calculation (same
folder and type) and not recorded. A second run, its cache filled from the
recorded SP, recorded them as new materials. Here the BAND output sits at
the top of the folder (scanned first) and three SPs of the 4LG_FSI
TopBottom project (the FSI molecule, the graphene slab, the full slab) in
SP/; through the real CLI.
"""
import os
import shutil
import sqlite3
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, TEST_DATA

PRE = "4LG_FSI_TopBottom_2x2_ABAB_SLAB_OPT"
BAND = [f"BAND/{PRE}_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_band.out",
        f"BAND/{PRE}_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_band.d3"]
SP = [f"SP/{PRE}_{s}.{ext}" for s in ("FSI", "graphene", "symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized")
      for ext in ("out", "d12")]


def _env():
    e = {k: v for k, v in os.environ.items()
         if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH", "MACE_DB_JOURNAL", "MACE_DB_LOCKING")}
    e["OMP_NUM_THREADS"] = "1"
    return e


def _register(runs, db):
    r = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "analyze", "--extract-properties",
                        str(runs), "--register", "--db-path", str(db)],
                       cwd=runs.parent, env=_env(), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    with sqlite3.connect(db) as c:
        return sorted(c.execute("SELECT output_file, material_id FROM calculations").fetchall())


def test_second_register_adds_nothing(tmp_path):
    pytest.importorskip("ase", reason="the build reads the structures with ase")
    if not TEST_DATA.is_dir():
        pytest.skip("test/ data corpus not present (gitignored)")
    runs = tmp_path / "runs"
    (runs / "SP").mkdir(parents=True)
    for rel in BAND:
        shutil.copyfile(TEST_DATA / rel, runs / rel.split("/")[1])
    for rel in SP:
        shutil.copyfile(TEST_DATA / rel, runs / "SP" / rel.split("/")[1])
    db = tmp_path / "materials.db"
    first = _register(runs, db)
    assert len(first) == 4                       # the BAND run and the three SPs
    assert _register(runs, db) == first
    with sqlite3.connect(db) as c:
        untracked = c.execute("SELECT COUNT(*) FROM properties WHERE calc_id IS NULL").fetchone()[0]
    assert untracked == 0
