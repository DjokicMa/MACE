"""Regression tests for the BAND.DAT band-gap fix (commit 4a1b6194).

BAND.DAT eigenvalues are Fermi-referenced; the gap must be computed by band
index (referencing-independent), not by splitting at the absolute .out Fermi.
"""
from pathlib import Path

import pytest

from conftest import find_data, REPO_ROOT


def _band_props(extractor, band_out: Path) -> dict:
    content = band_out.read_text(errors="ignore")
    return extractor._extract_band_structure_properties(content, band_out)


def test_insulator_band_gap_is_physical(extractor):
    """Wide-gap electrolyte: was ~1.45 eV (absolute-Fermi bug); must now be the
    real band-path gap (~8.5 eV, matching the band .out's own value)."""
    band_out = find_data("BAND/1LiFSI-1EMS-conf1*band.out", must_contain="FROM BAND")
    p = _band_props(extractor, band_out)
    assert p.get("band_dat_band_gap_ev") is not None
    assert p["band_dat_band_gap_ev"] == pytest.approx(8.47, abs=0.1)
    assert p.get("band_dat_metallic") is False


def test_metal_classified_metallic(extractor):
    """4LG bands overlap along the path -> 0 eV / metallic."""
    band_out = find_data("BAND/4LG_2x2_AA*band.out", must_contain="FROM BAND")
    p = _band_props(extractor, band_out)
    assert p.get("band_dat_band_gap_ev") == pytest.approx(0.0, abs=0.01)
    assert p.get("band_dat_metallic") is True


def test_gap_none_without_occupied_band_count():
    """Without the occupied-band count the gap must be left unset (never
    guessed from an arbitrary Fermi split)."""
    from mace.utils.dat_file_processor import DatFileProcessor
    bd = find_data("BAND/1LiFSI-1EMS-conf1*band.BAND.DAT")
    info = DatFileProcessor().process_band_dat_file(Path(bd))  # no num_occupied_bands
    ep = info["electronic_properties"]
    assert ep["band_gap_ev"] is None
    assert ep.get("note") == "band_gap_requires_occupied_band_count"


def test_persisted_gap_matches_independent_band_index_recompute(extractor):
    """The pipeline's persisted gap equals an independent band-index recompute
    from the raw BAND.DAT, and equals the physically-correct pinned value
    (~8.47 eV for 1EMS-conf1). Pins both the algorithm and the value."""
    from mace.utils.dat_file_processor import DatFileProcessor
    from mace.constants import HARTREE_TO_EV as H
    band_out = find_data("BAND/1LiFSI-1EMS-conf1*band.out", must_contain="FROM BAND")
    p = _band_props(extractor, band_out)
    bd = find_data("BAND/1LiFSI-1EMS-conf1*band.BAND.DAT")
    rows = DatFileProcessor().process_band_dat_file(Path(bd))["eigenvalues"]
    n_occ = 75  # 150 electrons / 2
    vbm = max(r[n_occ - 1] for r in rows if len(r) > n_occ)
    cbm = min(r[n_occ] for r in rows if len(r) > n_occ)
    idx_gap = (cbm - vbm) * H
    assert idx_gap == pytest.approx(8.473, abs=0.01)
    assert p["band_dat_band_gap_ev"] == pytest.approx(idx_gap, abs=0.01)
