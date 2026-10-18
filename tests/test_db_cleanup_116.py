"""The 1.1.6 database clean-up on real corpus outputs: merged keys (one gap
per source, one DOS at E_F, one spin flag, one Fermi energy, the final
geometry once), dead and constant keys gone, renames (cpu_time_total_s, the
Mulliken atoms' mulliken_population), wall_time_s, one calculation_type
vocabulary, one classification scheme, the lowest vibration without the
rigid-body modes, no placeholder lattice or space group, the layer group of
a slab - and the old names still readable for one release."""
import contextlib
import io
import json
import sqlite3
from pathlib import Path

import pytest

from conftest import find_data
from mace.utils import property_keys as PK
from mace.utils.imaginary_modes import is_linear, vibrations
from mace.utils.property_extractor import CrystalPropertyExtractor


def _extract(path):
    x = CrystalPropertyExtractor(db_path=None, enable_tracking=False)
    with contextlib.redirect_stdout(io.StringIO()):
        return x.extract_all_properties(Path(path))


def _no_stale_names(p):
    assert not (set(p) & (set(PK.ALIASES) | PK.REMOVED)), set(p) & (set(PK.ALIASES) | PK.REMOVED)
    assert not [k for k in p if k.startswith('_') and k not in ('_metadata', '_transport_zt')]


def test_unrestricted_molecule_sp():
    out = find_data("SP/1LiFSI-1DEC-conf1_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized.out")
    p = _extract(out)
    _no_stale_names(p)
    assert p["spin_polarized"] is True                     # one flag (was also is_spin_polarized)
    assert p["calculation_type"] == "SP"
    assert "space_group_number" not in p                   # a molecule has none (CRYSTAL prints P 1)
    assert p["cpu_time_total_s"] > 0 and p["wall_time_s"] > 0
    assert p["wall_time_s"] < p["cpu_time_total_s"]        # TOTAL CPU TIME sums the processes
    text = out.read_text()
    assert p["wall_time_s"] == float(text[text.rfind("TELAPSE"):].split()[1])
    atoms = p["mulliken_alpha_plus_beta"]["atoms"]
    assert "mulliken_population" in atoms[0] and "mulliken_charge" not in atoms[0]
    # the processed charges still come out (the processor reads the new field)
    assert p["processed_atomic_charges"]
    assert p["classification_gap_source"] == "scf"
    assert p["electronic_classification"] == PK.classify_gap(p["band_gap"])


def test_slab_sp_has_no_placeholder_c_and_its_layer_group():
    out = find_data("SP/4LG_2x2_AA_opt_HSESOL3C_optimized.out")
    p = _extract(out)
    _no_stale_names(p)
    assert "primitive_c" not in p and "initial_primitive_c" not in p
    assert p["primitive_cell_area"] > 0
    assert p["layer_group_number"] == 1 and p["layer_group_symbol"] == "P 1"


def test_doss_has_one_dos_at_fermi():
    out = find_data("DOSS/1LiFSI-1DEC-conf1_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_CRYSTAL_SP_symm_HSESOL3C_SOLDEF2MSVP_doss.out")
    p = _extract(out)
    _no_stale_names(p)
    assert p["dos_at_fermi_total"] == pytest.approx(p["dos_at_fermi_alpha"] + p["dos_at_fermi_beta"])
    assert PK.KEYS["dos_at_fermi_total"].unit == "states/eV"
    assert p["calculation_type"] == "DOSS"
    assert p["fermi_energy_au"] == -0.29                   # the DOSS print: its .out has no SCF EFERMI


def test_one_fermi_energy_prefers_the_scf():
    # a DOSS output that also holds its SCF: the SCF's EFERMI (full
    # precision), not the DOSS print's 3 decimals (1.1.5 stored both)
    p = _extract(find_data("DOSS/3,4^2T1-CA_rev1_sp_B3LYP-D3-D3_optimized_doss.out"))
    assert p["fermi_energy_au"] == pytest.approx(-0.074381233)


def test_band_run_classified_from_its_band_structure():
    out = find_data("BAND/1LiFSI-1DEC-conf1_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_CRYSTAL_SP_symm_HSESOL3C_SOLDEF2MSVP_band.out")
    p = _extract(out)
    _no_stale_names(p)
    assert p["calculation_type"] == "BAND"
    assert "band_gap" not in p                             # the .out has no SCF
    assert p["classification_gap_source"] == "band_structure"
    assert p["electronic_classification"] == PK.classify_gap(p["band_fundamental_gap_ev"])
    assert "band_start" not in p                           # band 1: not stored
    assert p["fermi_energy_au"] is not None
    assert "band_structure_k_range_max" in p


@pytest.mark.parametrize("pattern", [
    "FREQ/1LiFSI-1DEC-conf1_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_freq_HSESOL3C_optimized.out",
    "FREQ/1_dia_opt_rev1_freq_B3LYP-D3-D3_optimized_supercel222.out"])
def test_lowest_frequency_leaves_out_the_rigid_body_modes(pattern):
    out = find_data(pattern)
    p = _extract(out)
    modes = [f["frequency_cm"] for f in p["vibrational_frequencies"]
             for _ in range(f["mode_end"] - f["mode_start"] + 1)]
    molecule = "DIMENSIONALITY OF THE SYSTEM    0" in out.read_text()
    x = CrystalPropertyExtractor(db_path=None, enable_tracking=False)
    linear = molecule and is_linear([(a['x'], a['y'], a['z']) for a in x._extract_atomic_positions(out.read_text())])
    kept = vibrations(modes, molecule, linear)
    assert len(modes) - len(kept) == (6 if molecule else 3)
    assert p["lowest_frequency_cm"] == min(kept)
    rigid = sorted(modes, key=abs)[:len(modes) - len(kept)]
    assert all(abs(r) <= abs(p["lowest_frequency_cm"]) for r in rigid)
    assert p["highest_frequency_cm"] == max(kept)


def test_opt_stores_the_final_geometry_once():
    out = find_data("OPT/1_dia_opt_BULK_OPTGEOM.out")
    p = _extract(out)
    _no_stale_names(p)
    assert p["calculation_type"] == "OPT"
    assert not [k for k in p if k.startswith("final_") and not k.startswith(("final_gradient", "final_neighbor",
                                                                            "final_min_bond", "final_max_bond",
                                                                            "final_max_coord", "final_total_coord"))]
    assert p["primitive_a"] != p["initial_primitive_a"]    # optimized vs input
    assert "atomic_positions" in p and "initial_atomic_positions" in p


def test_a_2_to_4_ev_gap_is_a_semiconductor():
    p = _extract(find_data("SP/3,4^2T13-CA_rev1_sp_B3LYP-D3-D3_optimized.out"))
    assert 2.0 < p["band_gap"] <= 4.0
    assert p["electronic_classification"] == "semiconductor"   # 1.1.5's .out scheme said insulator above 2 eV


# ---------------------------------------------------------------- old names
@pytest.fixture
def db_with_old_names(tmp_path, monkeypatch):
    """A database whose rows carry 1.1.5 names (not extracted again)."""
    monkeypatch.chdir(tmp_path)
    from mace.database.materials import MaterialDatabase
    db = MaterialDatabase(str(tmp_path / "m.db"))
    db.create_material("m1", "C2", space_group=227)
    cid = db.create_calculation("m1", "TRANSPORT")
    with db._get_connection() as conn:
        for name, value in (("transport_zt_max", 19.5), ("transport_zt_electronic_max", 19.5),
                            ("total_cpu_time", 12.0), ("fermi_energy", -0.2), ("band_gap", 1.1),
                            ("final_primitive_a", 3.57)):
            conn.execute("INSERT INTO properties (material_id, calc_id, property_category, property_name, "
                         "property_value, property_value_text, extracted_at) VALUES (?, ?, 'other', ?, ?, ?, ?)",
                         ("m1", cid, name, value, str(value), "2026-01-01T00:00:00"))
    return db


def test_old_rows_read_under_new_names(db_with_old_names):
    db = db_with_old_names
    names = {p["property_name"]: p["property_value"] for p in db.get_material_properties("m1")}
    assert names == {"transport_zt_electronic_max_avg": 19.5, "cpu_time_total_s": 12.0,
                     "fermi_energy_au": -0.2, "band_gap": 1.1, "primitive_a": 3.57}
    # a query by an old or a new name finds them
    assert [r["property_value"] for r in db.get_properties_by_name("transport_zt_max")] == [19.5]
    assert [r["property_value"] for r in db.get_properties_by_name("cpu_time_total_s")] == [12.0]
    # filters, by either name
    assert [m["material_id"] for m in db.filter_materials_by_properties(["total_cpu_time > 10"])] == ["m1"]
    assert [m["material_id"] for m in db.filter_materials_advanced("cpu_time_total_s > 10")] == ["m1"]


def test_exports_take_old_names_and_write_new_columns(db_with_old_names, tmp_path):
    from mace.database.export.formats import export_materials
    out = export_materials(db_with_old_names, "csv", str(tmp_path / "x.csv"),
                           include_properties=["transport_zt_max", "fermi_energy_au"])
    header = Path(out).read_text().splitlines()[0].split(",")
    assert "transport_zt_electronic_max_avg" in header and "fermi_energy_au" in header
    assert "transport_zt_max" not in header and "total_cpu_time" not in header


def test_gui_inspector_reads_old_names_and_blobs(db_with_old_names, tmp_path):
    from mace.gui import db as gdb
    db = db_with_old_names
    with db._get_connection() as conn:
        cid = conn.execute("SELECT calc_id FROM calculations").fetchone()[0]
        conn.execute("INSERT INTO property_blobs (material_id, calc_id, property_category, property_name, "
                     "property_value_text, property_unit, extracted_at) VALUES ('m1', ?, 'structural', "
                     "'neighbor_analysis', '[1, 2]', 'JSON', '2026-01-01')", (cid,))
    r = gdb.material(Path(db.db_path), "m1")
    props = {p["property_name"]: p for p in r["properties"]}
    assert props["neighbor_analysis"]["property_value_text"] == "[1, 2]"
    assert "transport_zt_electronic_max_avg" in props and "transport_zt_max" not in props
    assert "neighbor_analysis" in gdb.property_names(Path(db.db_path))
    assert "cpu_time_total_s" in gdb.property_names(Path(db.db_path))
