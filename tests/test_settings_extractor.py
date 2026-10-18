"""Regression tests for input-settings extraction wiring.

manager.py and create_fresh_database.py imported the extractor from a bare
``input_settings_extractor`` module that does not exist in the reorganized
layout, so the import always raised ImportError and the step silently no-opped
(manager) or fell back to basic extraction (create_fresh_database). The real
function lives at ``mace.utils.settings_extractor``.
"""
from pathlib import Path

import pytest

from conftest import find_data, REPO_ROOT


def test_extractor_importable_from_package():
    """The canonical import path must resolve."""
    from mace.utils.settings_extractor import extract_and_store_input_settings
    assert callable(extract_and_store_input_settings)


def test_live_callers_use_package_path_not_bare_module():
    """Guard against re-introducing the broken ``from input_settings_extractor``
    import on the two active code paths."""
    for rel in ("mace/queue/manager.py",
                "mace/database/utils/create_fresh_database.py"):
        src = (REPO_ROOT / rel).read_text()
        assert "from input_settings_extractor import" not in src, (
            f"{rel} still imports the nonexistent bare module")
        assert "from mace.utils.settings_extractor import" in src, (
            f"{rel} should import the extractor from mace.utils.settings_extractor")


def test_extract_and_store_input_settings_roundtrip(tmp_path):
    """End-to-end: a real test/ .d12 extracts and persists into
    calculations.input_settings_json."""
    import json
    from mace.database.materials import MaterialDatabase
    from mace.utils.settings_extractor import extract_and_store_input_settings

    d12 = find_data("**/*.d12")

    db_path = str(tmp_path / "m.db")
    db = MaterialDatabase(db_path=db_path, ase_db_path=str(tmp_path / "s.db"))
    db.create_material(material_id="mat1", formula="X")
    calc_id = db.create_calculation("mat1", "OPT", input_file=str(d12))

    ok = extract_and_store_input_settings(calc_id, Path(d12), db_path)
    assert ok is True

    with db._get_connection() as conn:
        row = conn.execute(
            "SELECT input_settings_json FROM calculations WHERE calc_id = ?",
            (calc_id,)).fetchone()
    assert row is not None and row[0]
    settings = json.loads(row[0])
    # The extractor records the CRYSTAL keyword list it parsed.
    assert "crystal_keywords" in settings


def test_3c_functional_spin_and_keywords_of_a_real_deck():
    """A HSESOL3C deck: the 3c functional (none was recognized), SPIN, and
    the keywords MACE writes (SPIN, the FREQ intensities, SUPERCEL, GUESSP,
    SPINLOCK, ...) among the recorded keywords."""
    from mace.utils.settings_extractor import extract_input_settings
    d12 = find_data("OPT/*HSESOL3C*.d12", must_contain="HSESOL3C")
    s = extract_input_settings(d12)
    assert s["functional_info"]["exchange"] == "HSESOL3C"
    assert s["functional_info"]["spin"] is True
    assert "SPIN" in s["crystal_keywords"]


def test_freq_and_restart_keywords_recorded():
    from mace.utils.settings_extractor import extract_input_settings
    freq = extract_input_settings(find_data("FREQ/1_dia_opt_rev1_freq_B3LYP-D3-D3_optimized_supercel222.d12"))
    assert "SUPERCEL" in freq["crystal_keywords"]
    intens = extract_input_settings(find_data("FREQ/*.d12", must_contain="INTRAMAN"))
    assert {"INTENS", "INTCPHF", "INTRAMAN"} <= set(intens["crystal_keywords"])
    guessp = extract_input_settings(find_data("*/*.d12", must_contain="GUESSP"))
    assert "GUESSP" in guessp["crystal_keywords"]
    lock = extract_input_settings(find_data("*/*.d12", must_contain="SPINLOCK"))
    assert "SPINLOCK" in lock["crystal_keywords"]
