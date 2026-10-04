"""The occupied-band count a BAND gap is computed with comes from the real electron count.

The fallback was (header "N. OF ELECTRONS PER CELL") // 2. DOPING does not
change that header (measured on CRYSTAL23: it still shows the count before doping; the real
count is the Mulliken "NO. OF ELECTRONS" line), so a doped parent gave the
wrong HOMO index and a wrong gap. A spin-polarised run prints two Mulliken
blocks, ALPHA+BETA then ALPHA-BETA: the last line is the spin, not the count.
An odd count has no closed-shell HOMO index, so the count is left unset.
"""
from pathlib import Path

import pytest

from ion_fixtures import mulliken_block

BAND_HEADER = """
 *******************************************************************************
 *  BAND STRUCTURE                                                             *
 *******************************************************************************
 N. OF ELECTRONS PER CELL    12 EXCHANGE PSEUDO OVP (F(G))   (T4) 10**   -9
"""

ATOMS = [("LI", 3), ("O", 8), ("H", 1)]


def _num_occ(extractor, tmp_path, monkeypatch, band_text, parent_text=None):
    seen = {}

    def fake(self, path, fermi_energy=None, num_occupied_bands=None):
        seen["n"] = num_occupied_bands
        return {"error": "not needed"}

    from mace.utils import dat_file_processor
    monkeypatch.setattr(dat_file_processor.DatFileProcessor, "process_band_dat_file", fake)
    out = tmp_path / "x_band.out"
    out.write_text(band_text)
    (tmp_path / "x_band.BAND.DAT").write_text("#\n")
    if parent_text is not None:
        (tmp_path / "x.out").write_text(parent_text)
    extractor._extract_band_structure_properties(band_text, out)
    return seen["n"]


def test_a_doped_parent_count_wins_over_the_header(extractor, tmp_path, monkeypatch):
    parent = mulliken_block(ATOMS, [2.0, 9.0, 3.0], 14.0)
    assert _num_occ(extractor, tmp_path, monkeypatch, BAND_HEADER, parent) == 7


def test_the_band_out_own_mulliken_count_is_used(extractor, tmp_path, monkeypatch):
    text = BAND_HEADER + mulliken_block(ATOMS, [2.0, 9.0, 3.0], 14.0)
    assert _num_occ(extractor, tmp_path, monkeypatch, text) == 7


def test_the_alpha_minus_beta_block_is_not_the_count(extractor, tmp_path, monkeypatch):
    parent = (mulliken_block(ATOMS, [2.0, 9.0, 3.0], 14.0, "ALPHA+BETA ELECTRONS")
              + mulliken_block(ATOMS, [0.0, 1.0, 0.0], 1.0, "ALPHA-BETA ELECTRONS"))
    assert _num_occ(extractor, tmp_path, monkeypatch, BAND_HEADER, parent) == 7


def test_an_odd_count_leaves_it_unset(extractor, tmp_path, monkeypatch):
    parent = mulliken_block(ATOMS, [2.0, 9.0, 2.0], 13.0)
    assert _num_occ(extractor, tmp_path, monkeypatch, BAND_HEADER, parent) is None


def test_without_any_mulliken_count_the_header_is_used(extractor, tmp_path, monkeypatch):
    assert _num_occ(extractor, tmp_path, monkeypatch, BAND_HEADER) == 6


def test_an_odd_header_leaves_it_unset(extractor, tmp_path, monkeypatch):
    text = BAND_HEADER.replace("    12 EXCHANGE", "    13 EXCHANGE")
    assert _num_occ(extractor, tmp_path, monkeypatch, text) is None


def test_mulliken_electron_count_reads_alpha_plus_beta_or_unlabeled():
    from mace.utils.property_extractor import mulliken_electron_count
    spin = (mulliken_block(ATOMS, [2.0, 9.0, 3.0], 14.0, "ALPHA+BETA ELECTRONS")
            + mulliken_block(ATOMS, [0.0, 1.0, 0.0], 1.0, "ALPHA-BETA ELECTRONS"))
    assert mulliken_electron_count(spin) == pytest.approx(14.0)
    assert mulliken_electron_count(mulliken_block(ATOMS, [2.0, 9.0, 3.0], 14.0)) == pytest.approx(14.0)
    assert mulliken_electron_count(BAND_HEADER) is None
    # Several SCF prints (an optimisation): the last one counts
    two = mulliken_block(ATOMS, [2.0, 9.0, 3.0], 13.0) + mulliken_block(ATOMS, [2.0, 9.0, 3.0], 14.0)
    assert mulliken_electron_count(two) == pytest.approx(14.0)


@pytest.mark.parametrize("name", ["BAND"])
def test_corpus_spin_output_count_is_not_the_spin(name):
    from conftest import find_data
    from mace.utils.property_extractor import mulliken_electron_count
    path = find_data("BAND/1LiFSI-1Sulfolane-conf3_*_sp_*.out", "ALPHA-BETA ELECTRONS")
    assert mulliken_electron_count(Path(path).read_text(errors="ignore")) == pytest.approx(156.0)
