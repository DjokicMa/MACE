"""Regression tests: single-file property extraction must not require/create a DB.

CrystalPropertyExtractor used to open MaterialDatabase in __init__, so merely
constructing it created materials.db — blocking offline single-file extraction
and littering the working dir. The DB is now lazy and tracking is optional.
"""
from pathlib import Path

from mace.utils.property_extractor import CrystalPropertyExtractor

from conftest import find_data


def test_offline_extraction_creates_no_db(monkeypatch, tmp_path):
    """enable_tracking=False: full extraction works and no DB is created."""
    out = find_data("FREQ/EC_MOLECULE_OPT_symm_HSESOL3C*freq_HSESOL3C_optimized_temp.out")
    monkeypatch.chdir(tmp_path)  # any stray materials.db would land here
    ex = CrystalPropertyExtractor(enable_tracking=False)
    props = ex.extract_all_properties(Path(out))
    assert props.get("gibbs_free_energy_au") is not None  # extraction still works
    assert ex.db is None
    assert not (tmp_path / "materials.db").exists()
    assert ex.save_properties_to_database(props) == 0  # no-op, no raise


def test_lazy_db_not_created_on_pure_parse(tmp_path):
    """Default tracking is lazy: pure parsing does not open/create the DB file."""
    out = find_data("FREQ/EC_MOLECULE_OPT_symm_HSESOL3C*freq_HSESOL3C_optimized_temp.out")
    db_path = tmp_path / "lazy.db"
    ex = CrystalPropertyExtractor(db_path=str(db_path))
    ex._extract_frequency_properties(Path(out).read_text(errors="ignore"))
    assert not db_path.exists()


def test_db_created_on_demand(tmp_path):
    """Accessing .db (tracking on) opens the database lazily."""
    db_path = tmp_path / "ondemand.db"
    ex = CrystalPropertyExtractor(db_path=str(db_path))
    assert ex.db is not None
    assert db_path.exists()
