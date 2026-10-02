"""Per-fragment Mulliken charges of a charged run, in the properties and the report.

Computed only for charged runs (Mulliken count different from the sum of
valence charges, a CHARGE NORMALIZATION FACTOR other than 1, or "UNIT CELL NOT
NEUTRAL"), or when asked for; a neutral run's properties are unchanged. The
outputs are synthetic excerpts in CRYSTAL23's layout.
"""
import json

import pytest

from ion_fixtures import Z, cluster, ec, fsi, li_near, mulliken_block, synth_out
from mace.utils.ion_fragments import final_structure, format_fragment_charges, fragment_charges


def li_fsi_ec():
    anion = fsi()
    return cluster((anion, (0.0, 0.0, 0.0)), (ec(), (8.0, 0.0, 0.0))) + [li_near(anion, 2)]


def _out(atoms, pops, count, spins=None, extra=""):
    rows = [(s.upper(), Z[s]) for s, _ in atoms]
    mull = mulliken_block(rows, pops, count, "ALPHA+BETA ELECTRONS" if spins else "")
    if spins:
        mull += mulliken_block(rows, spins, sum(spins), "ALPHA-BETA ELECTRONS")
    return synth_out(atoms, mulliken=mull, extra=extra)


def _neutral_pops(atoms, shifts=None):
    pops = [float(Z[s]) for s, _ in atoms]
    for i, d in (shifts or {}).items():
        pops[i] += d
    return pops


def test_a_neutral_run_gets_nothing():
    atoms = li_fsi_ec()
    assert fragment_charges(_out(atoms, _neutral_pops(atoms), sum(Z[s] for s, _ in atoms))) == {}


def test_a_doped_run_gets_its_count_and_fragment_charges():
    atoms = li_fsi_ec()  # neutral: 138 electrons; run with one removed (DOPING -1)
    li = len(atoms) - 1
    ec_o = [i for i, (s, _) in enumerate(atoms) if s == "O"][-1]
    pops = _neutral_pops(atoms, {li: -0.9, ec_o: -0.1})
    text = _out(atoms, pops, 137.0)
    res = fragment_charges(text)
    assert res["electron_count"]["net_charge"] == pytest.approx(1.0)
    by_ion = {e["ion"] or e["formula"]: e for e in res["fragment_charges"]}
    assert by_ion["Li+"]["mulliken_charge"] == pytest.approx(0.9)
    assert by_ion["FSI-"]["mulliken_charge"] == pytest.approx(0.0)
    # FSI- holding none of its extra electron: flagged (heuristic, > 0.5 e)
    assert "electron delocalised from FSI-" in by_ion["FSI-"]["flag"]
    assert by_ion["Li+"]["flag"] is None


def test_an_anion_at_minus_0_3_is_flagged_and_one_at_minus_0_8_is_not():
    atoms = li_fsi_ec()
    o_fsi = [i for i, (s, _) in enumerate(atoms) if s == "O"][:4]
    li = len(atoms) - 1
    for extra, flagged in ((0.3, True), (0.8, False)):
        shifts = {i: extra / 4 for i in o_fsi}
        shifts[li] = -extra
        res = fragment_charges(_out(atoms, _neutral_pops(atoms, shifts), 138.0), force=True)
        fsi_entry = next(e for e in res["fragment_charges"] if e["ion"] == "FSI-")
        assert fsi_entry["mulliken_charge"] == pytest.approx(-extra)
        assert (fsi_entry["flag"] is not None) == flagged


def test_spin_is_summed_per_fragment():
    atoms = li_fsi_ec()
    spins = [0.0] * len(atoms)
    o_fsi = [i for i, (s, _) in enumerate(atoms) if s == "O"][:4]
    for i in o_fsi:
        spins[i] = 0.25
    res = fragment_charges(_out(atoms, _neutral_pops(atoms), 137.0, spins=spins))
    fsi_entry = next(e for e in res["fragment_charges"] if e["ion"] == "FSI-")
    assert fsi_entry["spin"] == pytest.approx(1.0)


@pytest.mark.parametrize("extra", [" CHARGE NORMALIZATION FACTOR   1.00317460\n",
                                   " WARNING **** INPBAS **** UNIT CELL NOT NEUTRAL\n"])
def test_renormalised_guess_or_charged_cell_triggers_it(extra):
    atoms = li_fsi_ec()
    text = _out(atoms, _neutral_pops(atoms), 138.0, extra=extra)
    assert "fragment_charges" in fragment_charges(text)


def test_report_section_only_with_fragments():
    assert format_fragment_charges({}) == []
    atoms = li_fsi_ec()
    res = fragment_charges(_out(atoms, _neutral_pops(atoms, {len(atoms) - 1: -1.0}), 137.0))
    lines = format_fragment_charges(res)
    assert lines[0].startswith("Fragments (net charge +1, 137 electrons)")
    assert any("Li+" in l for l in lines)


def test_properties_and_database_rows(tmp_path, monkeypatch):
    from mace.database.materials import MaterialDatabase
    from mace.utils.property_extractor import CrystalPropertyExtractor
    monkeypatch.chdir(tmp_path)
    atoms = li_fsi_ec()
    out = tmp_path / "cluster.out"
    out.write_text(_out(atoms, _neutral_pops(atoms, {len(atoms) - 1: -1.0}), 137.0))
    neutral = tmp_path / "neutral.out"
    neutral.write_text(_out(atoms, _neutral_pops(atoms), 138.0))
    db_path = str(tmp_path / "materials.db")
    MaterialDatabase(db_path).create_material("cluster", "x")
    ex = CrystalPropertyExtractor(db_path=db_path)
    props = ex.extract_all_properties(out, material_id="cluster")
    assert props["electron_count"]["net_charge"] == pytest.approx(1.0)
    assert "fragment_charges" not in ex.extract_all_properties(neutral, material_id="cluster")
    ex.save_properties_to_database(props)
    with MaterialDatabase(db_path)._get_connection() as conn:
        row = conn.execute("SELECT property_category, property_value_text FROM properties "
                           "WHERE property_name = 'fragment_charges'").fetchone()
    assert row[0] == "population_analysis"
    assert {e["ion"] for e in json.loads(row[1])} == {"Li+", "FSI-", None}


def test_final_structure_reads_fractional_and_slab_columns():
    text = """
 GEOMETRY FOR WAVE FUNCTION - DIMENSIONALITY OF THE SYSTEM    2
 *******************************************************************************
 LATTICE PARAMETERS (ANGSTROMS AND DEGREES) - BOHR = 0.5291772083 ANGSTROM
 PRIMITIVE CELL
         A              B              C           ALPHA      BETA       GAMMA 
     4.00000000     5.00000000   500.00000000    90.000000  90.000000  90.000000
 *******************************************************************************
 ATOMS IN THE ASYMMETRIC UNIT    2 - ATOMS IN THE UNIT CELL:    2
     ATOM                 X/A                 Y/B             Z(ANGSTROM)
 *******************************************************************************
      1 T   3 LI    5.000000000000E-01  0.000000000000E+00  1.500000000000E+00
      2 T   8 O     0.000000000000E+00  5.000000000000E-01 -1.000000000000E+00
"""
    geom = final_structure(text)
    assert geom["periodic_dirs"] == (0, 1)
    assert geom["xyz"][0] == pytest.approx((2.0, 0.0, 1.5))
    assert geom["xyz"][1] == pytest.approx((0.0, 2.5, -1.0), abs=1e-9)
