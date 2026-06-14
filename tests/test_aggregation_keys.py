"""Contract test: the property names AggregationEngine groups by must be the
same names CrystalPropertyExtractor actually writes.

CODEBASE_AUDIT.md #11 claimed aggregation.py grouped by conductivity_type /
energy_range / atoms_range using property names "the extractor never writes",
so everything bucketed to Unknown. Verified against real test/ output that this
is FALSE — the extractor emits all of them. This test locks that contract so a
future rename can't silently send everything to Unknown.
"""
from pathlib import Path

import pytest

from conftest import find_data


# property_name -> the group_by mode in AggregationEngine._group_materials that
# reads it (aggregation.py).
GROUPING_KEYS = {
    "conductivity_type": "conductivity_type",
    "total_energy_au": "energy_range",
    "atoms_in_unit_cell": "atoms_range",
    "band_gap": "band_gap_range",
}


@pytest.fixture(scope="module")
def sp_properties(tmp_path_factory):
    """Extracted properties from a real SP/OPT output that has an energy block
    and a band gap (a molecular electrolyte SP with HSESOL3C)."""
    from mace.utils.property_extractor import CrystalPropertyExtractor
    out = find_data("**/*_sp_*.out", must_contain="TOTAL ENERGY")
    if "charge" in out.name or "band" in out.name:
        pytest.skip("matched a charge/band output without the SP energy block")
    # tmp_path_factory (session-scoped, works with this module-scoped fixture) for
    # isolation, instead of a fixed world-writable /tmp file.
    db = tmp_path_factory.mktemp("agg") / "agg.db"
    ex = CrystalPropertyExtractor(db_path=str(db), enable_tracking=False)
    return ex.extract_all_properties(out, material_id="m", calc_id="c")


@pytest.mark.parametrize("key", sorted(GROUPING_KEYS))
def test_aggregation_grouping_key_is_emitted(sp_properties, key):
    assert key in sp_properties, (
        f"aggregation groups by '{key}' ({GROUPING_KEYS[key]}) but the extractor "
        f"no longer emits it — grouping would silently bucket everything to Unknown")
    assert sp_properties[key] is not None
