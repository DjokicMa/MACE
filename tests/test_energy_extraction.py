"""Regression tests for energy & FREQ-thermodynamics extraction.

Ground-truth values are read directly from the real CRYSTAL outputs and pinned
here so the gCP / corrected-total / molecular-thermo / enthalpy fixes (commit
fb173e82) cannot silently regress.
"""
import pytest

from conftest import find_data, energy_props
from mace.constants import HARTREE_TO_EV


def test_gcp_extracted_for_3c_method(extractor):
    """3C methods print 'GCP ENERGY (AU)'; it must be captured (was dropped)."""
    out = find_data("BAND/1LiFSI-3EMS-conf1*sp_HSESOL3C_optimized.out")
    p = energy_props(extractor, out)
    assert p.get("gcp_energy_au") == pytest.approx(0.17333783934451, abs=1e-9)
    assert p.get("d3_dispersion_energy_au") == pytest.approx(-0.029832870759267, abs=1e-9)


def test_corrected_total_includes_gcp_single_point(extractor):
    """corrected total = total + D3 + gCP (the ~108 kcal/mol fix)."""
    out = find_data("BAND/1LiFSI-3EMS-conf1*sp_HSESOL3C_optimized.out")
    p = energy_props(extractor, out)
    corrected = p["total_energy_corrected_au"]
    # Matches CRYSTAL's printed "TOTAL ENERGY + DISP + GCP (AU)" line.
    assert corrected == pytest.approx(-3353.7022822551, abs=1e-4)
    # Internally consistent with the stored components.
    assert corrected == pytest.approx(
        p["total_energy_au"] + p["d3_dispersion_energy_au"] + p["gcp_energy_au"], abs=1e-9
    )
    # The historical D3-only field must remain (back-compat) and exclude gCP.
    assert p["total_energy_plus_d3_au"] == pytest.approx(
        p["total_energy_au"] + p["d3_dispersion_energy_au"], abs=1e-9
    )


def test_ev_conversions_use_full_precision_constant(extractor):
    """Item 6a precision fix: the eV fields must equal the AU value times the
    full-precision HARTREE_TO_EV (27.211386245988), not the old truncated 27.2114.
    The tight rel tolerance (1e-12) would fail if any *27.2114 literal returned —
    27.2114 vs full precision diverge at ~1.4e-5."""
    out = find_data("BAND/1LiFSI-3EMS-conf1*sp_HSESOL3C_optimized.out")
    p = energy_props(extractor, out)
    for au_key, ev_key in [
        ("total_energy_au", "total_energy_ev"),
        ("d3_dispersion_energy_au", "d3_dispersion_energy_ev"),
        ("gcp_energy_au", "gcp_energy_ev"),
        ("total_energy_corrected_au", "total_energy_corrected_ev"),
    ]:
        if p.get(au_key) is not None and p.get(ev_key) is not None:
            assert p[ev_key] == pytest.approx(p[au_key] * HARTREE_TO_EV, rel=1e-12), (
                f"{ev_key} not converted with full-precision HARTREE_TO_EV")


def test_corrected_total_uses_final_geometry_in_opt(extractor):
    """In an OPT the printed combined line is the INITIAL geometry; the corrected
    total must come from the FINAL-geometry components instead (4LG was stale by
    ~0.094 AU when the printed line was trusted)."""
    out = find_data("SP/4LG_2x2_AA_opt_HSESOL3C_optimized.out")
    p = energy_props(extractor, out)
    assert p["total_energy_corrected_au"] == pytest.approx(-3644.7719812597, abs=1e-3)
    # Definitely NOT the stale initial-geometry printed value.
    assert abs(p["total_energy_corrected_au"] - (-3644.6775083249)) > 0.05


def test_molecular_freq_thermo_extracted(extractor):
    """Molecular FREQ blocks are headed '...TAKING INTO ACCOUNT MOLECULAR...';
    Gibbs/ET/PV/TS were silently dropped before the regex was broadened."""
    out = find_data("FREQ/1LiFSI-1EC-conf1*freq_HSESOL3C_optimized_temp.out")
    p = energy_props(extractor, out)
    assert p.get("gibbs_free_energy_au") == pytest.approx(-1696.913226294267, abs=1e-6)
    assert p.get("entropy_term_au") == pytest.approx(0.061224798069, abs=1e-6)
    assert p.get("thermal_energy_au") == pytest.approx(0.015661288759, abs=1e-6)
    assert p.get("pv_term_au") == pytest.approx(0.000944185937, abs=1e-6)


def test_enthalpy_equals_gibbs_plus_ts(extractor):
    """H = Gibbs + TS = EL + E0 + ET + PV (exactly, from CRYSTAL's columns)."""
    out = find_data("FREQ/1LiFSI-1EC-conf1*freq_HSESOL3C_optimized_temp.out")
    p = energy_props(extractor, out)
    h = p.get("enthalpy_au")
    assert h is not None
    assert h == pytest.approx(p["gibbs_free_energy_au"] + p["entropy_term_au"], abs=1e-9)
    el, e0 = p["electronic_energy_au"], p["zero_point_energy_au"]
    et, pv = p["thermal_energy_au"], p["pv_term_au"]
    assert h == pytest.approx(el + e0 + et + pv, abs=1e-5)


def test_electronic_energy_extracted(extractor):
    out = find_data("FREQ/1LiFSI-1EC-conf1*freq_HSESOL3C_optimized_temp.out")
    p = energy_props(extractor, out)
    assert p.get("electronic_energy_au") == pytest.approx(-1696.980340890400, abs=1e-6)


def test_crystal_freq_regression_unchanged(extractor):
    """Periodic ('...WITH VIBRATIONAL CONTRIBUTIONS') FREQ extraction must be
    unchanged by the molecular-header broadening."""
    out = find_data("FREQ/1_dia_opt_rev1_freq_*supercel222.out")
    p = energy_props(extractor, out)
    assert p.get("gibbs_free_energy_au") == pytest.approx(-609.632528018002, abs=1e-9)
    assert p.get("zero_point_energy_au") == pytest.approx(0.109109635725, abs=1e-9)
    assert p.get("enthalpy_au") == pytest.approx(
        p["gibbs_free_energy_au"] + p["entropy_term_au"], abs=1e-9
    )
