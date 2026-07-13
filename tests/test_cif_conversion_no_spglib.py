"""CIF -> D12 conversion must stay structurally correct without spglib.

Real-world failure (phase-3 QA, 2026-07-09, HPCC job 12085937): quick-start
converted 3_dia3.cif on a python without spglib. ASE expands the CIF's 2
asymmetric-unit atoms into the full 40-atom cell, and the no-spglib path
wrote ALL of them while still declaring space group 227 — CRYSTAL re-applied
the operators to the expanded set and died with
"DISTANCE BETWEEN ATOMS 13 54 TOO SMALL: 0.6850 ANGSTROM / ERROR NEIGHB".
The other 7 materials survived only because all their atoms sit on special
positions whose orbits fold exactly.

Fallback contract: per the CIF spec the raw _atom_site_ records ARE the
asymmetric unit, so without spglib the converter must write those (they are
byte-for-byte what the user's known-good manual decks contain).
"""
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))

pytest.importorskip("ase", reason="CIF parsing needs ase (absent in CI)")
import NewCifToD12  # noqa: E402

DATA = Path(__file__).parent / "data"
DIA3_CIF = DATA / "3_dia3_opt_BULK_OPTGEOM_symm.cif"
DIA_CIF = DATA / "1_dia_opt_BULK_OPTGEOM_symm.cif"


def test_parse_cif_keeps_raw_asymmetric_records():
    d = NewCifToD12.parse_cif(str(DIA3_CIF))
    assert len(d["symbols"]) == 40, "ASE-expanded cell expected"
    assert d["cif_atom_symbols"] == ["C", "C"]
    assert len(d["cif_atom_positions"]) == 2
    assert d["spacegroup"] == 227


def test_no_spglib_reduction_uses_cif_records(monkeypatch):
    monkeypatch.setattr(NewCifToD12, "SPGLIB_AVAILABLE", False)
    d = NewCifToD12.parse_cif(str(DIA3_CIF))

    reduced = NewCifToD12.verify_and_reduce_to_asymmetric_unit(d)

    assert len(reduced["symbols"]) == 2, (
        "no-spglib fallback must write the CIF's asymmetric unit, not the "
        "expanded cell (CRYSTAL re-applies the space group -> ERROR NEIGHB)")
    assert reduced["spacegroup"] == 227
    # normalized coordinates must match the user's known-good manual deck
    got = {tuple(round(x, 5) for x in p) for p in reduced["positions"]}
    assert got == {(0.75, 0.75, 0.75), (0.83912, 0.83912, 0.66088)}, got
    assert reduced["atomic_numbers"] == [6, 6]


def test_no_spglib_reduction_1_dia(monkeypatch):
    monkeypatch.setattr(NewCifToD12, "SPGLIB_AVAILABLE", False)
    d = NewCifToD12.parse_cif(str(DIA_CIF))

    reduced = NewCifToD12.verify_and_reduce_to_asymmetric_unit(d)

    assert len(reduced["symbols"]) == 1
    assert reduced["positions"][0] == pytest.approx([0.0, 0.0, 0.0])


def test_spglib_reduction_still_works():
    """Regression guard for the normal spglib path (available locally)."""
    if not NewCifToD12.SPGLIB_AVAILABLE:
        pytest.skip("spglib not installed in this environment")
    d = NewCifToD12.parse_cif(str(DIA3_CIF))

    reduced = NewCifToD12.verify_and_reduce_to_asymmetric_unit(d)

    assert len(reduced["symbols"]) == 2
    assert reduced["spacegroup"] == 227


def test_planner_band_config_with_first_last_band_validates():
    """Real incident (phase-4 QA, 9T2 BAND): the planner-written expert BAND
    config specifies the range as first_band=1/last_band=null ("all bands",
    resolved per material) and has no literal 'bands' key — the validator
    rejected it and CRYSTALOptToD3 exited 0 with no D3 file, so the plan's
    1000-pt seekpath BAND silently never generated."""
    sys.path.insert(0, str(REPO_ROOT / "Crystal_d3"))
    from d3_config import validate_d3_config

    planner_cfg = {"calculation_type": "BAND", "n_points": 1000,
                   "first_band": 1, "last_band": None,
                   "kpath_source": "seekpath_inv", "path": "auto"}
    is_valid, errors = validate_d3_config(planner_cfg)
    assert is_valid, errors

    # a config with NO band range at all must still be rejected
    is_valid, errors = validate_d3_config(
        {"calculation_type": "BAND", "n_points": 1000})
    assert not is_valid


def test_transport_d3_starts_with_newk(tmp_path):
    """Real incident (phase-5 QA, job 12185576): the generated BOLTZTRA deck
    had no NEWK block and CRYSTAL aborted with "NEWK MUST BE CALLED BEFORE
    BOLTZTRA". The transport writer must emit NEWK + shrink + IFE first,
    like the DOSS writer does."""
    sys.path.insert(0, str(REPO_ROOT / "Crystal_d3"))
    import CRYSTALOptToD3 as mod

    gen = mod.D3Generator.__new__(mod.D3Generator)
    gen.input_file = tmp_path / "mat_sp.out"
    gen.input_dir = tmp_path
    gen.base_name = "mat_sp"
    gen.structure_info = {"dimensionality": 3}
    (tmp_path / "mat_sp.d12").write_text("X\nSHRINK\n12 24\nEND\n")

    d3 = gen._write_transport_d3({"temperature_range": (100, 800, 50),
                                  "mu_range": (-2.0, 2.0, 0.01),
                                  "tdf_range": (-5.0, 5.0, 0.01)})

    lines = d3.splitlines()
    assert lines[0] == "NEWK", d3
    assert "BOLTZTRA" in lines
    assert lines.index("NEWK") < lines.index("BOLTZTRA")
    assert "1 0" in lines
    # BOLTZTRA block END + deck terminator END (single END computed fine
    # but aborted with "END OF DATA IN INPUT DECK" on exit)
    assert lines.count("END") == 2
