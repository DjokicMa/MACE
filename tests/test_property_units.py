"""Unit labels of stored properties.

``_get_property_unit`` labels by substrings of the key, so several keys got
a wrong unit: the Hartree gap ``band_gap_advanced_ha`` (gone in 1.1.6) was labeled eV (by
'band_gap'), text and true/false keys such as ``band_gap_type`` or
``conductivity_type`` got eV or S/m, the DOS at E_F from a DOSS.DAT (states
per Hartree per cell) states/eV or dimensionless, and the eV band edges and
kJ/mol energy were dimensionless. Those keys now have their unit by exact
name, checked before the heuristics.
"""
import pytest

from conftest import TEST_DATA

UNITS = {
    # 1.1.6: every unit comes from the key registry (property_keys.KEYS)
    "band_fundamental_gap_ev": "eV",
    # DOS summary: states/eV, both spins (dos_at_fermi_level and
    # doss_dat_at_fermi, per Hartree, are gone)
    "dos_at_fermi_total": "states/eV",
    "dos_at_fermi_alpha": "states/eV",
    # DOSS.DAT energy axis: Hartree relative to E_F
    "doss_dat_energy_min": "Hartree",
    "doss_dat_energy_max": "Hartree",
    "band_dat_vbm_ev": "eV",
    "band_dat_cbm_ev": "eV",
    "electronic_energy_kj_mol": "kJ/mol",
    "band_gap_is_direct": "boolean",
    "band_gap_type": "category",
    "electronic_classification": "category",
    "effective_mass_calculation_method": "category",
    "fermi_energy_au": "Hartree",
    "cpu_time_total_s": "seconds",
    "wall_time_s": "seconds",
}

# labels that are no physical unit
NON_PHYSICAL = {"dimensionless", "category", "boolean", "code", "JSON", "coordinates",
                "symmetry_label", "number", "", None}


@pytest.mark.parametrize("name,unit", sorted(UNITS.items()))
def test_unit_by_exact_name(extractor, name, unit):
    assert extractor._get_property_unit(name) == unit


def _one_output_per_kind():
    if not TEST_DATA.is_dir():
        pytest.skip("test/ data corpus not present (gitignored)")
    picks = []
    for kind in ("OPT", "SP", "FREQ", "BAND", "DOSS", "TRANSPORT", "ECH3POT3"):
        outs = sorted(p for p in (TEST_DATA / kind).glob("*.out") if p.stat().st_size < 3_000_000)
        picks += outs[:3]
    # the outputs with a BAND.DAT and DOSS.DAT next to them (advanced analysis)
    picks += sorted((TEST_DATA / "BAND").glob("1_dia_opt_rev1_sp_*_band.out"))
    picks += sorted((TEST_DATA / "DOSS").glob("1_dia_opt_rev1_sp_*_doss.out"))
    return picks


def test_no_text_or_boolean_key_gets_a_physical_unit(extractor):
    bad = {}
    for out in _one_output_per_kind():
        props = extractor.extract_all_properties(out)
        for name, value in props.items():
            if name.startswith("_"):
                continue
            text = isinstance(value, str) and not value.lstrip().startswith(("[", "{"))
            if isinstance(value, bool) or text:
                try:
                    float(value)
                    continue          # a number printed as text
                except (TypeError, ValueError):
                    pass
                unit = extractor._get_property_unit(name)
                if unit not in NON_PHYSICAL:
                    bad[name] = unit
    assert not bad, bad


def test_re_extraction_refreshes_unit_and_category(extractor):
    """Corrected labels reach a database built by an earlier version: a
    re-extraction's UPDATE of an existing row rewrites its unit and category."""
    import sqlite3
    from conftest import find_data
    out = find_data("SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out")
    props = extractor.extract_all_properties(out)
    assert "band_gap_type" in props
    extractor.save_properties_to_database(props)
    db = extractor.db.db_path
    with sqlite3.connect(db) as c:
        c.execute("UPDATE properties SET property_unit = 'eV', property_category = 'stale' "
                  "WHERE property_name = 'band_gap_type'")
    extractor.save_properties_to_database(extractor.extract_all_properties(out))
    with sqlite3.connect(db) as c:
        rows = c.execute("SELECT property_unit, property_category FROM properties "
                         "WHERE property_name = 'band_gap_type'").fetchall()
    assert rows == [("category", extractor._categorize_property("band_gap_type"))]
