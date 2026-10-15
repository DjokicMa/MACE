"""Untracked extraction files rows under the material id the registration
gives (bug hunt 1.1.5, CLI/parser round).

Without --register an output's rows went under db_build.material_id_from_filename,
which strips one suffix: "1_dia_opt_rev1" for the OPT and
"1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized" for its SP, where the register path
(materials.create_material_id_from_file) gives "1_dia" for both. One material
was split in two, and a later --register filed it under a third id."""
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA

FILES = ["OPT/1_dia_opt_rev1.out", "SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out",
         "FREQ/2_dia2_opt_rev1_freq_B3LYP-D3-D3_optimized.out"]


def _env():
    e = {k: v for k, v in os.environ.items()
         if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH", "MACE_DB_JOURNAL", "MACE_DB_LOCKING")}
    e["OMP_NUM_THREADS"] = "1"
    return e


def test_untracked_rows_use_the_registration_id(tmp_path):
    if not TEST_DATA.is_dir():
        pytest.skip("test/ data corpus not present (gitignored)")
    d = tmp_path / "runs"
    d.mkdir()
    for r in FILES:
        shutil.copyfile(TEST_DATA / r, d / Path(r).name)
    db = tmp_path / "materials.db"
    p = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "analyze",
                        "--extract-properties", str(d), "--db-path", str(db)],
                       capture_output=True, text=True, env=_env(), cwd=str(tmp_path), timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    c = sqlite3.connect(str(db))
    ids = {r[0] for r in c.execute("SELECT DISTINCT material_id FROM properties")}
    c.close()
    from mace.database.materials import create_material_id_from_file
    assert ids == {create_material_id_from_file(Path(r).name) for r in FILES} == {"1_dia", "2_dia2"}
