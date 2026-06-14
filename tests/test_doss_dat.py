"""Regression tests for DOSS.DAT parsing fixes (pre-push review dat-formula#F1/F2/F3).

Real DOSS.DAT files are used (a .DAT is derived data, not a synthetic CRYSTAL .out).
For the headerless edge case (#F1) a real file has its '#' header stripped in tmp --
the legitimate analog to editing a real .d12 for the SPINLOCK round-trip test.
"""
from pathlib import Path

import pytest

from conftest import find_data
from mace.utils.dat_file_processor import DatFileProcessor


def _dia_doss() -> Path:
    return find_data("**/*dia*DOSS.DAT")


def test_doss_accepts_str_path():
    """#F3: str paths must not raise AttributeError (parity with formula extractor)."""
    r = DatFileProcessor().process_doss_dat_file(str(_dia_doss()))  # STR, not Path
    assert r.get("error") is None
    assert r["num_energy_points"] > 0


def test_band_accepts_str_path():
    """#F3: same coercion for the BAND.DAT entry point."""
    bd = find_data("BAND/1LiFSI-1EMS-conf1*band.BAND.DAT")
    r = DatFileProcessor().process_band_dat_file(str(bd))
    assert r.get("error") is None


def test_insulator_dos_at_fermi_is_near_zero():
    """#F2: an insulator must report ~0 DOS at E_F, not the valence-band tail
    just below E=0 (which the old max-over-window returned, ~0.19 for diamond)."""
    da = DatFileProcessor().process_doss_dat_file(_dia_doss())["dos_analysis"]
    assert da["band_gap_ev"] > 4.0          # diamond gap ~5.9 eV
    assert da["metallic"] is False
    assert abs(da["dos_at_fermi"]) < 0.01    # in-gap -> ~0


def test_headerless_doss_still_parses(tmp_path):
    """#F1: a DOSS.DAT lacking the '# NEPTS/NPROJ' header must still parse to
    energy points (the old width=0 path returned ZERO). Strip the header off a
    real file and confirm the modal-column-width fallback recovers the records."""
    src = _dia_doss().read_text().splitlines()
    body = [ln for ln in src if not ln.lstrip().startswith('#')]
    out = tmp_path / "headerless_DOSS.DAT"
    out.write_text("\n".join(body) + "\n")
    r = DatFileProcessor().process_doss_dat_file(out)
    assert r["num_energy_points"] > 0        # was 0 before the width fallback
    assert len(r["total_dos"]) == r["num_energy_points"] * max(r["num_spins"], 1)
