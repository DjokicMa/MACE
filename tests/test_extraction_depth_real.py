"""How deep `mace analyze --extract-properties --register` reads each kind of
CRYSTAL output (PLAN_analysis C11): one real output per type, copied from
the test/ corpus into tmp, through the real CLI (subprocess), then the
database is asked what it holds. Skips without the corpus.

| Type | Expected |
|---|---|
| OPT | final energy, final geometry, space_group_number |
| SP | energy, gap |
| FREQ | vibrational_frequencies; IR intensities (INTENS); Raman (RAMAN); imaginary count |
| BAND | band_structure (k-path, labels), VBM/CBM, gap, direct/indirect |
| DOSS | dos_structure, DOS at E_F |
| TRANSPORT | transport_seebeck_max_uv_per_k + payload |
| CHARGE+POTENTIAL | ECH3/POT3 flags and grid metadata |
"""
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA

FILES = ["OPT/1_dia_opt_rev1.out", "OPT/1_dia_opt_rev1.d12",
         "SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out", "SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.d12",
         "BAND/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band.out", "BAND/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band.d3",
         "BAND/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band.BAND.DAT",
         "DOSS/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss.out", "DOSS/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss.d3",
         "DOSS/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss.DOSS.DAT",
         "TRANSPORT/C1-RCSR-ana_optimized_TRANSPORT.out", "TRANSPORT/C1-RCSR-ana_optimized_TRANSPORT.d3",
         "TRANSPORT/C1-RCSR-ana_optimized_TRANSPORT.SEEBECK.dat"]


def _raman_freq():
    """A FREQ output with IR intensities and a Raman table (found by content, not by name)."""
    for p in sorted((TEST_DATA / "FREQ").glob("*.out")):
        if p.stat().st_size > 3_000_000:
            continue
        t = p.read_text(errors="replace")
        if "AVERAGED ISOTROPIC INTENSITIES" in t and "IR INTENSITIES" in t and "TOTAL CPU TIME" in t:
            return p
    pytest.skip("no FREQ output with Raman intensities in the corpus")


def _charge_potential():
    for p in sorted((TEST_DATA / "ECH3POT3").glob("*.out")):
        return p
    pytest.skip("no CHARGE+POTENTIAL output in the corpus")


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    if not TEST_DATA.is_dir():
        pytest.skip("test/ data corpus not present (gitignored)")
    d = tmp_path_factory.mktemp("depth") / "runs"
    d.mkdir()
    for r in FILES:
        shutil.copyfile(TEST_DATA / r, d / Path(r).name)
    shutil.copyfile(_raman_freq(), d / "ramanrun_freq.out")
    shutil.copyfile(_charge_potential(), d / "chargerun_charge+potential.out")
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    env["OMP_NUM_THREADS"] = "1"
    p = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "analyze", "--extract-properties", "runs",
                        "--db-path", "runs/materials.db", "--register"], capture_output=True, text=True, env=env,
                       cwd=str(d.parent), timeout=900)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    return d / "materials.db"


def props(db, name_part):
    c = sqlite3.connect(str(db))
    try:
        cid = c.execute("SELECT calc_id FROM calculations WHERE output_file LIKE ?", (f"%{name_part}",)).fetchone()
        assert cid, f"no calculation for {name_part}"
        out = {}
        for n, v, t in c.execute("SELECT property_name, property_value, property_value_text FROM properties WHERE calc_id = ?", cid):
            out[n] = v if v is not None else (json.loads(t) if t and t[:1] in "[{" else t)
        return out
    finally:
        c.close()


def test_opt(db):
    p = props(db, "1_dia_opt_rev1.out")
    assert p["total_energy_au"] < 0 and p["atomic_positions"]
    assert int(p["space_group_number"]) == 227


def test_sp(db):
    p = props(db, "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out")
    assert p["total_energy_au"] < 0 and float(p["band_gap"]) > 1


def test_freq_ir_raman_and_imaginary(db):
    p = props(db, "ramanrun_freq.out")
    modes = p["vibrational_frequencies"]
    assert modes and any(m.get("ir_intensity") for m in modes)
    assert p["raman_intensities"] and all({"i_tot", "i_par", "i_perp"} <= set(r) for r in p["raman_intensities"])
    assert int(p["num_raman_modes"]) == len(p["raman_intensities"])
    assert "num_imaginary_frequencies" in p


def test_band(db):
    p = props(db, "_band.out")
    assert p["band_structure"] and p["band_dat_kpath_labels"]
    assert p["band_dat_vbm_ev"] is not None and p["band_dat_cbm_ev"] is not None
    assert float(p["band_fundamental_gap_ev"]) > 1 and "band_gap_is_direct" in p


def test_doss(db):
    p = props(db, "_doss.out")
    assert p["dos_structure"] and "dos_at_fermi_total" in p


def test_transport(db):
    p = props(db, "_TRANSPORT.out")
    assert p["transport_seebeck_max_uv_per_k"] is not None and p["transport"]


def test_charge_potential(db):
    p = props(db, "chargerun_charge+potential.out")
    assert "chargepot_has_ech3" in p and "chargepot_grid_dims" in p and p["charge_potential"]


def test_every_row_has_its_material(db):
    c = sqlite3.connect(str(db))
    try:
        orphans = c.execute("SELECT count(*) FROM properties WHERE material_id NOT IN (SELECT material_id FROM materials)").fetchone()[0]
    finally:
        c.close()
    assert orphans == 0
