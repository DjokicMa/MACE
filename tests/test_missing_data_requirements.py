"""Regression test for the SP missing-data classifier (commit 0d52a696).

electronic_classification is only emitted when a band gap is found, so it is
absent for metals / most bulk SP runs. Keeping it in SP 'required' false-flagged
every metal SP as incomplete. total_energy_au is the only reliable SP required
property; electronic_classification belongs in 'optional'. Self-contained (the
requirements table is a class attribute — no DB needed).
"""
from mace.database.analysis.missing_data import MissingDataAnalyzer

REQ = MissingDataAnalyzer.CALC_TYPE_PROPERTIES


def test_sp_required_is_only_total_energy():
    assert REQ["SP"]["required"] == ["total_energy_au"]


def test_electronic_classification_not_required_for_sp():
    """A metal SP (no band gap -> no electronic_classification) must not be
    flagged incomplete."""
    assert "electronic_classification" not in REQ["SP"]["required"]
    assert "electronic_classification" in REQ["SP"]["optional"]


def test_required_property_names_use_extractor_suffixes():
    """The table must use the names the extractor actually emits (the original
    bug was names like 'total_energy' that the extractor never writes)."""
    # Energy-bearing required entries use the _au suffix the extractor emits.
    assert "total_energy_au" in REQ["OPT"]["required"]
    assert "total_energy_au" in REQ["SP"]["required"]
    # No required list references a bare 'total_energy' (the never-emitted name).
    for calc_type, spec in REQ.items():
        assert "total_energy" not in spec["required"], calc_type
