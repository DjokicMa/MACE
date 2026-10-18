"""Re-extraction removes stale rows of untracked outputs too.

Audit repro (round 6): `mace analyze --extract-properties d` with 1.1.1 on a
copy of test/DOSS/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss.* stored 58
rows under (material, calc_id NULL), among them band-structure fields a DOSS
run does not have: has_band_structure=1, total_bands=8, band_start=4,
fermi_energy_band=-0.0801 and vbm_energy=6.0 (a band index stored as eV).
Re-extracting with 1.1.5 left all five, because only rows of a tracked
calc_id were cleaned up.

An untracked row cannot name the output that wrote it - every output whose
name maps to the material shares (material, NULL). So the clean-up is scoped
to the run: when the run re-extracted every output of the material it can see
(every .out of the scanned tree, or of a single --output-file's directory,
whose name maps to the material), a row of the material that no output of the
run wrote again is stale and goes. If such an output was left out (a filter),
or the material was given by hand (--material-id), nothing is removed.
"""
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import find_data

REPO = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data" / "charged_runs"
OLD = "2026-08-01T12:00:00.000000"
# The five rows 1.1.1 wrote for the DOSS output (values from the audit's DB).
STALE_111 = [("band_structure", "band_start", 4.0, "4", "dimensionless"),
             ("electronic", "fermi_energy_band", -0.0801, "-0.0801", "Hartree"),
             ("band_structure", "has_band_structure", 1.0, "True", "boolean"),
             ("band_structure", "total_bands", 8.0, "8", "dimensionless"),
             ("electronic", "vbm_energy", 6.0, "6.0", "eV")]


def _mace(args, cwd):
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME")}
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), *args], cwd=cwd, env=env,
                       capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    return r


def _rows(db, material):
    with sqlite3.connect(db) as c:
        return {r[0]: r[1:] for r in c.execute(
            "SELECT property_name, property_value, extracted_at, calc_id FROM properties "
            "WHERE material_id = ?", (material,))}


def _seed(db, material, rows):
    with sqlite3.connect(db) as c:
        for cat, name, val, text, unit in rows:
            c.execute("INSERT INTO properties (material_id, calc_id, property_category, property_name, "
                      "property_value, property_value_text, property_unit, extracted_at, extractor_script) "
                      "VALUES (?, NULL, ?, ?, ?, ?, ?, ?, 'crystal_property_extractor.py')",
                      (material, cat, name, val, text, unit, OLD))


def _extract_twice(tmp_path, outs, material, stale, extra_args=("--no-register",)):
    d = tmp_path / "d"
    d.mkdir()
    for o in outs:
        for f in o.parent.glob(o.stem + ".*"):
            shutil.copy(f, d / f.name)
    db = tmp_path / "c.db"
    _mace(["analyze", "--extract-properties", str(d), "--db-path", str(db), *extra_args], tmp_path)
    _seed(db, material, stale)
    before = _rows(db, material)
    _mace(["analyze", "--extract-properties", str(d), "--db-path", str(db), *extra_args], tmp_path)
    return before, _rows(db, material)


def test_the_audits_doss_rows_go(tmp_path):
    out = find_data("DOSS/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss.out")
    material = "1_dia"            # the registration's id (create_material_id_from_file)
    before, after = _extract_twice(tmp_path, [out], material, STALE_111)
    for _, name, *_ in STALE_111:
        assert name in before
        assert name not in after, (name, after.get(name))
    kept = set(before) - {s[1] for s in STALE_111}
    assert kept and kept <= set(after)
    assert all(v[2] is None for v in after.values())


def test_without_corpus_an_sp_output(tmp_path):
    out = DATA / "li_fsi_ec_neutral.out"
    before, after = _extract_twice(tmp_path, [out], "li_fsi_ec_neutral", STALE_111)
    assert "total_bands" in before and "total_bands" not in after
    assert "vbm_energy" not in after
    assert set(before) - {s[1] for s in STALE_111} <= set(after)


def test_a_sibling_left_out_keeps_the_material_untouched(tmp_path):
    """--filter-type SP skips a sibling DOSS output of the same material:
    its rows cannot be told apart from stale ones, so none are removed."""
    d = tmp_path / "d"
    d.mkdir()
    shutil.copy(DATA / "li_fsi_ec_neutral.out", d / "li_x_sp.out")
    shutil.copy(DATA / "odd_hse06_0d_crash.out", d / "li_x_doss.out")
    db = tmp_path / "c.db"
    _mace(["analyze", "--extract-properties", str(d), "--db-path", str(db), "--no-register"], tmp_path)
    _seed(db, "li_x", STALE_111)
    _mace(["analyze", "--extract-properties", str(d), "--db-path", str(db), "--filter-type", "SP",
           "--no-register"],
          tmp_path)
    assert "total_bands" in _rows(db, "li_x")
