"""The advanced BAND/DOSS analysis reads only the output's own .DAT files.

CrystalPropertyExtractor._extract_advanced_band_dos_analysis took the FIRST
*.BAND.DAT and *.DOSS.DAT in the output's directory, so in a directory that
holds several calculations an output was classified, and given effective
masses, from another material's band structure. The files are now matched to
the output by stem: ``<stem>.BAND.DAT`` (the run's own), else the BAND/DOSS
run of the same parent (``<base>_band.BAND.DAT`` / ``<base>.BAND.DAT``, where
``<base>`` is the stem without a ``_band``/``_doss`` suffix). Two candidates
of the same rank are ambiguous and skipped, not guessed between.

Driven through the public extract_all_properties on synthetic files.
"""
from pathlib import Path

import pytest

from mace.utils import advanced_electronic_analyzer as aea
from mace.utils.property_extractor import CrystalPropertyExtractor

OUT = """\
 *******************************************************************************
 *                               CRYSTAL23                                     *
 *******************************************************************************
 SYNTHETIC TEST OUTPUT
 == SCF ENDED - CONVERGENCE ON ENERGY      E(AU) -1.0000000000000E+01 CYCLES  10
"""


@pytest.fixture
def seen(monkeypatch):
    calls = []

    def analyze(self, band_file=None, doss_file=None, n_atoms=None):
        calls.append((band_file and Path(band_file).name, doss_file and Path(doss_file).name))
        return {"analysis_method": "stub"}

    monkeypatch.setattr(aea.AdvancedElectronicAnalyzer, "analyze_material", analyze)
    return calls


def _extract(tmp_path, out_name, dats):
    for name in dats:
        (tmp_path / name).write_text("# synthetic\n")
    out = tmp_path / out_name
    out.write_text(OUT)
    return CrystalPropertyExtractor(db_path=None, enable_tracking=False).extract_all_properties(out)


def test_another_materials_dat_files_are_not_used(tmp_path, seen):
    """Before: mat.out analyzed with other_band.BAND.DAT / other_doss.DOSS.DAT."""
    props = _extract(tmp_path, "mat.out", ["other_band.BAND.DAT", "other_doss.DOSS.DAT"])
    assert seen == []
    # nothing analyzed: no band-analysis keys (1.1.6 stores no constant
    # advanced_analysis_available = False)
    assert not [k for k in props if k.startswith(("advanced_analysis", "band_file", "doss_file"))]


def test_the_outputs_own_files_are_used_whatever_sorts_first(tmp_path, seen):
    _extract(tmp_path, "mat_band.out", ["aaa_band.BAND.DAT", "aaa_doss.DOSS.DAT",
                                        "mat_band.BAND.DAT", "mat_doss.DOSS.DAT"])
    assert seen == [("mat_band.BAND.DAT", "mat_doss.DOSS.DAT")]


def test_an_sp_output_finds_its_band_and_doss_children(tmp_path, seen):
    _extract(tmp_path, "mat.out", ["mat_band.BAND.DAT", "mat_doss.DOSS.DAT",
                                   "mat2_band.BAND.DAT"])
    assert seen == [("mat_band.BAND.DAT", "mat_doss.DOSS.DAT")]


def test_the_runs_own_file_wins_over_a_sibling(tmp_path, seen):
    _extract(tmp_path, "mat_band.out", ["mat_band.BAND.DAT", "mat.BAND.DAT"])
    assert seen == [("mat_band.BAND.DAT", None)]


def test_ambiguous_files_are_skipped(tmp_path, seen):
    props = _extract(tmp_path, "mat_doss.out",
                     ["mat_band.BAND.DAT", "mat.BAND.DAT", "mat_doss.DOSS.DAT"])
    assert seen == [(None, "mat_doss.DOSS.DAT")]
    assert "mat_band.BAND.DAT" in props["advanced_analysis_note"]
    assert "mat.BAND.DAT" in props["advanced_analysis_note"]
