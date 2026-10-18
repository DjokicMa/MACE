"""Two different structures whose outputs have the same name in different
folders are two materials (bug hunt 1.1.5).

``A/mat_opt.out`` (diamond) and ``B/mat_opt.out`` (TiO) both map to the id
"mat", so a build filed both runs under one material. Now a new
calculation whose structure clearly differs from the material its id names
(another reduced formula or space group) gets the id plus its folder
("mat_B"), with a warning; the same structure in another folder (a copied
project) stays the same material, and existing rows are never renamed.
Real outputs, real CLI subprocess (the GUI's build runs the same command)."""
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA

DIA = "OPT/1_dia_opt_rev1.out"
TIO = "OPT/TiO_mp-1071163_sg189_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized.out"


def _env():
    e = {k: v for k, v in os.environ.items()
         if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH", "MACE_DB_JOURNAL", "MACE_DB_LOCKING")}
    e["OMP_NUM_THREADS"] = "1"
    return e


def _tree(tmp_path, layout):
    if not TEST_DATA.is_dir():
        pytest.skip("test/ data corpus not present (gitignored)")
    root = tmp_path / "DUP"
    for sub, src in layout.items():
        (root / sub).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TEST_DATA / src, root / sub / "mat_opt.out")
    return root


def _build(folder, db):
    p = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "analyze",
                        "--extract-properties", str(folder), "--db-path", str(db), "--register"],
                       capture_output=True, text=True, env=_env(), cwd=str(folder.parent), timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return p


def _q(db, sql, *a):
    c = sqlite3.connect(str(db))
    try:
        return c.execute(sql, a).fetchall()
    finally:
        c.close()


def _by_folder(db):
    return {Path(o).parent.name: m for m, o in _q(db, "SELECT material_id, output_file FROM calculations")}


def test_different_structures_with_one_name_are_two_materials(tmp_path):
    pytest.importorskip("ase", reason="the build tells structures apart with the structure reader, which needs ase")
    root = _tree(tmp_path, {"A": DIA, "B": TIO, "C": DIA})
    db = tmp_path / "materials.db"
    p = _build(root, db)
    by = _by_folder(db)
    assert by["A"] == by["C"]                       # the same structure in another folder
    assert by["A"] != by["B"]
    assert {by["A"], by["B"]} & {"mat"}             # the first keeps the plain id
    other = next(f for f in ("A", "B") if by[f] != "mat")
    assert by[other] == f"mat_{other}"
    mats = dict(_q(db, "SELECT material_id, formula FROM materials"))
    assert sorted(mats.values()) == ["C", "TiO"]
    assert "a different structure" in p.stdout
    # a second build changes nothing
    _build(root, db)
    assert _by_folder(db) == by
    assert len(_q(db, "SELECT * FROM calculations")) == 3
    orphans = _q(db, "SELECT DISTINCT material_id FROM properties WHERE material_id NOT IN "
                     "(SELECT material_id FROM materials)")
    assert orphans == []


def test_rows_an_earlier_build_merged_stay(tmp_path):
    """A database that filed both runs under "mat" keeps them there."""
    root = _tree(tmp_path, {"A": DIA, "B": TIO})
    db = tmp_path / "materials.db"
    _build(root, db)
    c = sqlite3.connect(str(db))
    with c:          # what 1.1.5 before this fix stored
        c.execute("UPDATE calculations SET material_id = 'mat'")
        c.execute("UPDATE properties SET material_id = 'mat'")
        c.execute("DELETE FROM materials WHERE material_id != 'mat'")
    c.close()
    _build(root, db)
    assert set(_by_folder(db).values()) == {"mat"}
    assert len(_q(db, "SELECT * FROM calculations")) == 2
