"""Regression tests for the thermoelectric / electronic TRANSPORT extractor.

CRYSTAL "THERMOELECTRIC AND ELECTRONIC TRANSPORT PROPERTIES CALCULATION"
(BOLTZTRA) runs historically stored only generic metadata -- the real
thermoelectric results (Seebeck coefficient, electrical conductivity, electronic
thermal conductivity, power factor, ZT as functions of chemical potential mu and
temperature T) lived only in the companion BoltzTraP ``.dat`` files and were
dropped on the floor.

``_extract_transport_properties`` now (via the full
``CrystalPropertyExtractor.extract_all_properties`` path) stores:
  - a compact ``transport`` JSON summary (temperatures covered, mu range, peak
    |Seebeck| with its (T, mu) and carrier type, peak power factor, peak
    electronic ZT, units, and the source ``.dat`` files referenced), and
  - flat queryable scalar rows (``transport_seebeck_max_uv_per_k``,
    ``transport_power_factor_max``, ``transport_zt_max``,
    ``transport_n_temperatures``).

These tests exercise the REAL extractor on the REAL ``test/TRANSPORT/*.out``
corpus (with its companion SEEBECK/SIGMA/KAPPA ``.dat`` files) and assert the
new structured data is present, JSON-serializable, native-typed, and physically
plausible. ``enable_tracking=False`` => no DB is touched (never writes
``./materials.db``). They skip cleanly when the gitignored ``test/`` corpus is
absent.
"""
import glob
import json
import math
from pathlib import Path

import pytest

from conftest import TEST_DATA, find_data
from mace.utils.property_extractor import CrystalPropertyExtractor


# A real transport .out whose companion .dat files carry usable data (afi has
# rich, non-degenerate SEEBECK/SIGMA/KAPPA). bcu-f and ana are also usable; ato
# is the degenerate (all-zeroed by BoltzTraP) case exercised separately.
_USABLE_OUT = "TRANSPORT/C1-RCSR-afi_optimized_TRANSPORT.out"
_DEGENERATE_OUT = "TRANSPORT/C1-RCSR-ato_optimized_TRANSPORT.out"

_TRANSPORT_HEADER = "THERMOELECTRIC AND ELECTRONIC TRANSPORT"


@pytest.fixture(scope="module")
def afi_props():
    """Real extractor output for the afi transport calc (usable .dat data),
    via the full ``extract_all_properties`` path. enable_tracking=False so no
    materials.db is created."""
    out = find_data(_USABLE_OUT, must_contain=_TRANSPORT_HEADER)
    extractor = CrystalPropertyExtractor(enable_tracking=False)
    return extractor.extract_all_properties(out)


def _all_transport_outs():
    """All ``test/TRANSPORT/*_TRANSPORT.out`` real BOLTZTRA runs that have a
    companion SEEBECK.dat. Empty (-> skip) when the corpus is absent."""
    if not TEST_DATA.is_dir():
        return []
    outs = []
    for out in sorted((TEST_DATA / "TRANSPORT").glob("*_TRANSPORT.out")):
        seebeck = out.parent / f"{out.stem}.SEEBECK.dat"
        if not seebeck.exists():
            continue
        try:
            if _TRANSPORT_HEADER in out.read_text(errors="ignore"):
                outs.append(out)
        except OSError:
            continue
    return outs


def test_transport_calc_detected(afi_props):
    """The run is recognized as a transport (BOLTZTRA) calculation."""
    assert afi_props.get("calculation_type") == "TRANSPORT"
    assert afi_props.get("has_transport") is True
    assert afi_props.get("transport_data_available") is True


def test_structured_transport_summary_present(afi_props):
    """The actual transport RESULTS (not just a calc-type marker) are stored as
    a compact JSON summary referencing the source .dat files."""
    t = afi_props.get("transport")
    assert isinstance(t, dict), "expected a structured transport payload"
    # Source data files are referenced so the raw tables stay on disk.
    assert t["seebeck_source_file"].endswith("SEEBECK.dat")
    assert any(f.endswith("SEEBECK.dat") for f in t["source_files"])
    assert t["has_usable_data"] is True


def test_temperatures_and_mu_range_real(afi_props):
    """Temperatures and chemical-potential range come from the real grid
    (300/500/700 K; mu spanning roughly -3.4..3.7 eV per the .out header)."""
    t = afi_props["transport"]
    temps = t["temperatures_k"]
    assert temps == sorted(temps)
    assert t["n_temperatures"] == len(temps) >= 1
    # Real CRYSTAL header: T from 300 to 700 K in 200 K steps.
    assert t["temperatures_k"] == [300.0, 500.0, 700.0]
    assert t["temperature_min_k"] == pytest.approx(300.0)
    assert t["temperature_max_k"] == pytest.approx(700.0)
    # mu range is finite and ordered.
    assert math.isfinite(t["mu_min_ev"]) and math.isfinite(t["mu_max_ev"])
    assert t["mu_min_ev"] < t["mu_max_ev"]
    # n_temperatures is also exposed as a flat scalar.
    assert afi_props["transport_n_temperatures"] == t["n_temperatures"]


def test_seebeck_peak_physically_plausible(afi_props):
    """Peak |Seebeck| is finite, positive, and in a physically sane thermo-
    electric range (tens to ~a few thousand uV/K), located at a real (T, mu),
    with a resolved carrier type."""
    t = afi_props["transport"]
    s_uv = t["seebeck_max_abs_uv_per_k"]
    assert math.isfinite(s_uv)
    # A real, non-degenerate thermoelectric Seebeck peak: well above noise,
    # well below the determinant-singularity blow-ups (which exceed 1e6 uV/K).
    assert 10.0 < s_uv < 1.0e5
    loc = t["seebeck_max_at"]
    assert loc is not None
    assert loc["temperature_k"] in t["temperatures_k"]
    assert math.isfinite(loc["mu_ev"])
    # Carrier type resolved (n-type / p-type) from the sign of N(#carriers).
    assert t["carrier_type"] in ("n-type", "p-type")
    # Flat scalar mirrors the summary.
    assert afi_props["transport_seebeck_max_uv_per_k"] == pytest.approx(s_uv)


def test_power_factor_and_zt_finite_and_consistent(afi_props):
    """Peak power factor and electronic ZT are finite, positive, and the
    scalars agree with the JSON summary. PF = S^2 * sigma; ZT_e = PF*T/kappa_e."""
    t = afi_props["transport"]
    pf = t["power_factor_max_w_per_m_k2"]
    zt = t["zt_electronic_max"]
    assert math.isfinite(pf) and pf > 0.0
    assert math.isfinite(zt) and zt > 0.0
    # Electronic ZT (no lattice kappa) is an upper bound; for a real semiconductor
    # it should be a modest O(1)-or-below figure here, not absurd.
    assert zt < 10.0
    assert afi_props["transport_power_factor_max"] == pytest.approx(pf)
    assert afi_props["transport_zt_max"] == pytest.approx(zt)
    # Peak locations are at real grid temperatures.
    for key in ("power_factor_max_at", "zt_electronic_max_at"):
        loc = t[key]
        assert loc is not None
        assert loc["temperature_k"] in t["temperatures_k"]


def test_units_recorded(afi_props):
    """Units are captured (Seebeck in V/K from the .dat header) so the stored
    scalars are self-describing."""
    t = afi_props["transport"]
    units = t["units"]
    assert "V/K" in units["seebeck"]
    assert units["power_factor"] == "W/m/K^2"
    assert "zt" in units


def test_transport_summary_json_serializable_and_native(afi_props):
    """The payload must JSON-serialize (it lands in property_value_text) with no
    non-native (e.g. numpy) types leaking in."""
    t = afi_props["transport"]
    blob = json.dumps(t)  # raises if any value is non-JSON
    again = json.loads(blob)
    assert again["n_temperatures"] == t["n_temperatures"]

    def _check_native(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                _check_native(v)
        elif isinstance(obj, list):
            for v in obj:
                _check_native(v)
        else:
            assert obj is None or isinstance(obj, (bool, int, float, str)), (
                f"non-native type leaked: {type(obj)}"
            )

    _check_native(t)


def test_all_real_transport_files_extract_sane():
    """Across EVERY real transport file the extractor must not crash, must mark
    the run as TRANSPORT, and must produce a JSON-serializable summary. Files
    with usable .dat data yield positive, finite peaks; the degenerate file
    (BoltzTraP zeroed everything) yields has_usable_data=False with no spurious
    scalar peaks."""
    outs = _all_transport_outs()
    if not outs:
        pytest.skip("test/TRANSPORT corpus not present (gitignored)")
    extractor = CrystalPropertyExtractor(enable_tracking=False)
    usable_count = 0
    for out in outs:
        props = extractor.extract_all_properties(out)  # must not raise
        assert props.get("calculation_type") == "TRANSPORT"
        t = props.get("transport")
        assert isinstance(t, dict)
        json.dumps(t)  # serializable
        assert t["n_temperatures"] >= 1
        if t["has_usable_data"]:
            usable_count += 1
            s_uv = t["seebeck_max_abs_uv_per_k"]
            assert 10.0 < s_uv < 1.0e5, f"{out.name}: implausible Seebeck {s_uv}"
            assert t["power_factor_max_w_per_m_k2"] > 0.0
            assert t["zt_electronic_max"] > 0.0
            assert t["carrier_type"] in ("n-type", "p-type")
            # Flat scalars present only when usable.
            assert props["transport_seebeck_max_uv_per_k"] == pytest.approx(s_uv)
            assert props["transport_zt_max"] == pytest.approx(t["zt_electronic_max"])
        else:
            # Degenerate file: no spurious peak scalars emitted.
            assert "transport_seebeck_max_uv_per_k" not in props
            assert "transport_zt_max" not in props
            assert props.get("transport_data_available") is False
    # The corpus must contain at least one genuinely usable transport file.
    assert usable_count >= 1, "expected >=1 usable transport file in test/TRANSPORT"


def test_degenerate_transport_file_graceful():
    """The ato file (BoltzTraP set the Seebeck/conductivity tensors to zero --
    'DETERMINANT OF THE CONDUCTIVITY MATRIX IS VERY SMALL') must not crash and
    must NOT fabricate thermoelectric peaks."""
    try:
        out = find_data(_DEGENERATE_OUT, must_contain=_TRANSPORT_HEADER)
    except Exception:
        pytest.skip("degenerate transport file not present")
    extractor = CrystalPropertyExtractor(enable_tracking=False)
    props = extractor.extract_all_properties(out)
    assert props.get("calculation_type") == "TRANSPORT"
    t = props.get("transport")
    assert isinstance(t, dict)
    # No usable data -> no fabricated peaks, no flat peak scalars.
    if not t["has_usable_data"]:
        assert t["seebeck_max_abs_uv_per_k"] == 0.0
        assert "transport_seebeck_max_uv_per_k" not in props
        assert "transport_zt_max" not in props


def test_non_transport_file_has_no_transport_key(extractor):
    """A non-transport .out (a plain BAND calc) must not gain a transport
    payload (the extractor is calc-type dispatched, additive, and localized)."""
    band_out = find_data(
        "BAND/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band.out",
        must_contain="FROM BAND",
    )
    content = band_out.read_text(errors="ignore")
    props = extractor._extract_transport_properties(content, band_out)
    assert "transport" not in props
    assert "calculation_type" not in props  # not claimed as TRANSPORT
