"""Regression tests for the structured DENSITY-OF-STATES extraction.

`_extract_dos_properties` historically stored only a couple of DOS scalars
(``doss_dat_band_gap_ev``, ``doss_dat_at_fermi``, ...). It now ALSO stores the
actual density of states via the REAL extractor:

  - a compact ``dos_structure`` JSON summary (per-spin DOS @ Fermi, gap-from-DOS
    with band-edge energies, integrated number of states, a DOWNSAMPLED total-DOS
    curve per spin in eV, per-projection integrated weights, source file), and
  - queryable scalar rows (``dos_total_at_fermi``, ``dos_at_fermi_alpha`` /
    ``dos_at_fermi_beta``, ``dos_at_fermi_both_spins_ev``,
    ``dos_band_gap_from_dos_ev``, ``dos_is_metallic``,
    ``dos_integrated_states_to_fermi``, ``dos_num_projections``).

UNITS / FERMI RELATIONSHIPS asserted here (see ``_build_dos_summary``):
  * DOS magnitudes are stored in states/eV/cell (converted from the legacy
    states/HARTREE/cell by dividing by HARTREE_TO_EV), self-consistent with the
    eV energy axis and the ``dos_units`` label;
  * ``dos_at_fermi_per_channel_ev`` (single channel) * HARTREE_TO_EV equals the
    legacy single-channel ``doss_dat_at_fermi`` (states/Ha) EXACTLY — so the
    headline DOS@Fermi has a DEFINED relationship to the legacy scalar (no
    silent 2x), while ``dos_at_fermi_both_spins_ev`` is the alpha+beta sum.

The raw ``DOSS.DAT`` (96 KB - 2 MB) is NOT dumped into the properties table; only
the compact summary (~12 KB) + scalars are persisted, and the raw file is
re-parsed on demand. These tests exercise the real invocation path
(``CrystalPropertyExtractor.extract_all_properties`` on a real ``test/DOSS``
output) and assert the new structured data is present, sane, JSON-serializable
with native types only, consistent with the pre-existing ``doss_dat_*`` scalars,
and handles spin channels. They skip cleanly when the gitignored ``test/`` corpus
is absent. enable_tracking=False -> no DB is touched (never ./materials.db).
"""
import glob
import json
import math
from pathlib import Path

import pytest

from conftest import TEST_DATA, find_data
from mace.utils.property_extractor import CrystalPropertyExtractor, HARTREE_TO_EV


@pytest.fixture(scope="module")
def diamond_dos_props():
    """Real extractor output for the smallest clean DOSS calc (diamond, NPROJ=5,
    unwrapped), via the full ``extract_all_properties`` path."""
    dos_out = find_data(
        "DOSS/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss.out",
    )
    extractor = CrystalPropertyExtractor(enable_tracking=False)
    return extractor.extract_all_properties(dos_out)


def test_structured_dos_present(diamond_dos_props):
    """The actual density of states (not just a gap scalar) is stored."""
    dos = diamond_dos_props.get("dos_structure")
    assert isinstance(dos, dict), "expected a structured dos_structure payload"
    assert dos["num_energy_points"] > 0
    assert dos["num_spins"] >= 1
    assert dos["num_proj"] >= 1
    # Source file is referenced so the raw curve can be re-parsed on demand.
    assert dos["source_file"].endswith("DOSS.DAT")
    # Units / reference are recorded (Fermi-referenced eV).
    assert dos["energy_units"] == "eV"
    assert dos["energy_reference"] == "fermi"


def test_dos_curve_present_monotonic_and_nonnegative(diamond_dos_props):
    """A downsampled total-DOS curve is stored PER SPIN: each is a list of
    [E_eV, DOS] pairs with a monotonically increasing energy grid and
    non-negative DOS (magnitudes), capped to keep the blob small."""
    dos = diamond_dos_props["dos_structure"]
    curves = dos["total_dos_curve_ev"]
    assert isinstance(curves, dict) and curves
    for spin, curve in curves.items():
        assert curve, f"empty DOS curve for {spin}"
        assert len(curve) <= 200, "curve must be downsampled (<=200 pts/spin)"
        es = [pt[0] for pt in curve]
        ds = [pt[1] for pt in curve]
        assert all(math.isfinite(e) for e in es)
        assert all(es[i] <= es[i + 1] for i in range(len(es) - 1)), (
            f"energy grid for {spin} is not monotonic"
        )
        assert all(d >= 0.0 for d in ds), f"negative DOS magnitude for {spin}"


def test_dos_at_fermi_per_spin_finite(diamond_dos_props):
    """DOS @ Fermi is reported PER SPIN CHANNEL (alpha + beta for a spin-
    polarized run), each a finite non-negative number — never silently
    collapsing to a single channel."""
    dos = diamond_dos_props["dos_structure"]
    by_spin = dos["dos_at_fermi_by_spin"]
    assert isinstance(by_spin, dict) and by_spin
    if dos["spin_polarized"]:
        assert set(by_spin) >= {"alpha", "beta"}, "both spin channels required"
    for spin, val in by_spin.items():
        assert val is None or (math.isfinite(val) and val >= 0.0), (
            f"DOS@Fermi for {spin} is not a sane value: {val}"
        )
    # Diamond is an insulator -> ~0 DOS at the Fermi level in both channels.
    for val in by_spin.values():
        assert val is not None and val < 0.01
    # The both-spins aggregate equals the sum of the per-spin finite values; the
    # single-channel headline equals just the first (alpha) channel (NOT 2x).
    finite = [v for v in by_spin.values() if v is not None]
    assert dos["dos_at_fermi_both_spins_ev"] == pytest.approx(sum(finite), abs=1e-9)
    assert dos["dos_at_fermi_per_channel_ev"] == pytest.approx(
        by_spin.get("alpha", finite[0]), abs=1e-9)
    # Back-compat headline tracks the single channel (legacy convention).
    assert dos["total_dos_at_fermi"] == pytest.approx(
        dos["dos_at_fermi_per_channel_ev"], abs=1e-9)


def test_gap_and_band_edges_from_dos(diamond_dos_props):
    """The gap-from-DOS and the band edges (VBM/CBM) straddling the Fermi level
    are stored. Diamond opens a wide gap above E_F."""
    dos = diamond_dos_props["dos_structure"]
    assert dos["metallic"] is False
    assert dos["band_gap_ev"] is not None and dos["band_gap_ev"] > 4.0
    vbm, cbm = dos["vbm_ev"], dos["cbm_ev"]
    assert vbm is not None and cbm is not None
    assert math.isfinite(vbm) and math.isfinite(cbm)
    # The valence-band top is at/below the conduction-band bottom, and the edge
    # span tracks the reported gap width.
    assert cbm > vbm
    assert (cbm - vbm) == pytest.approx(dos["band_gap_ev"], abs=0.05)


def test_integrated_states_and_projection_weights(diamond_dos_props):
    """The integrated occupied-state count (|DOS| up to E_Fermi) and the per-
    projection integrated weights (atom/orbital decomposition magnitudes) are
    stored as native floats."""
    dos = diamond_dos_props["dos_structure"]
    # The meaningless signed-trapezoid ``total_states`` (~0) is GONE; replaced by
    # a meaningful integrated occupied-state count (must be a positive number of
    # states, not the old ~0 cancellation artifact).
    assert "total_states" not in dos, "meaningless total_states must be removed"
    occ = dos["dos_integrated_states_to_fermi"]
    assert isinstance(occ, float) and math.isfinite(occ)
    assert occ > 1.0, "integrated occupied states should be a real (>1) count"

    weights = dos.get("projection_integrated_weights")
    # Diamond's NPROJ=5 header counts the TOTAL column + a duplicate-of-total
    # column; the genuine projections are 3 -> exactly 3 weight entries.
    assert isinstance(weights, dict) and weights
    for name, w in weights.items():
        assert isinstance(name, str)
        assert isinstance(w, float) and math.isfinite(w) and w >= 0.0

    # ISSUE 4: num_proj counts only GENUINE projections == len(weights), and no
    # stored projection column is byte-identical to the TOTAL DOS.
    assert dos["num_proj"] == len(weights), "num_proj must equal weight count"
    assert dos["num_proj"] == 3, "diamond has 3 genuine projections (not 5)"
    assert diamond_dos_props["dos_num_projections"] == len(weights)

    # No projection weight equals the TOTAL-DOS integrated weight: re-parse the
    # raw .DAT and confirm none of the GENUINE projection columns equal total.
    from mace.utils.dat_file_processor import DatFileProcessor
    dat = (
        find_data("DOSS/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss.out").parent
        / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss.DOSS.DAT"
    )
    parsed = DatFileProcessor().process_doss_dat_file(str(dat))
    total_col = parsed["total_dos"]
    kept = set(weights)
    for col_name, col in (parsed.get("projected_dos") or {}).items():
        if col == total_col:
            assert col_name not in kept, (
                f"{col_name} is byte-identical to TOTAL and must be excluded"
            )


def test_dos_gap_consistent_with_scalar(diamond_dos_props):
    """The new structured gap-from-DOS must match the pre-existing validated
    scalar ``doss_dat_band_gap_ev`` (both derive from the same analyzer)."""
    dos = diamond_dos_props["dos_structure"]
    scalar_gap = diamond_dos_props["doss_dat_band_gap_ev"]
    assert scalar_gap is not None
    assert dos["band_gap_ev"] == pytest.approx(scalar_gap, abs=1e-6)
    # The DOS-at-Fermi aggregate is consistent with the analyzer's scalar too
    # (analyzer reports a single channel; the per-spin values share its
    # first-point->=0 convention, so each channel matches it for diamond).
    assert diamond_dos_props["doss_dat_at_fermi"] == pytest.approx(0.0, abs=0.01)


def test_queryable_scalar_rows_added(diamond_dos_props):
    """The gap / DOS@Fermi / integrated-states are also exposed as flat scalar
    props (numerically filterable), without losing the legacy keys."""
    p = diamond_dos_props
    for k in (
        "dos_total_at_fermi",
        "dos_band_gap_from_dos_ev",
        "dos_integrated_states_to_fermi",
        "dos_num_projections",
    ):
        assert k in p, f"missing scalar prop {k}"
    # The meaningless signed total_states scalar is gone.
    assert "dos_total_states" not in p, "meaningless dos_total_states must be gone"
    assert isinstance(p["dos_is_metallic"], bool)
    # Per-spin scalars exposed for a spin-polarized run.
    if p["dos_structure"]["spin_polarized"]:
        assert "dos_at_fermi_alpha" in p and "dos_at_fermi_beta" in p
    # Legacy scalars preserved (layered, not rewritten).
    assert p["doss_dat_band_gap_ev"] is not None
    assert p["doss_dat_num_spins"] >= 1
    assert p["doss_dat_exists"] is True
    assert p["calculation_type"] == "DOSS"
    assert p["has_dos"] is True


def test_dos_units_self_consistent_with_label(diamond_dos_props):
    """ISSUE 1: the stored DOS magnitudes are in states/eV/cell (matching the
    eV energy axis and the ``dos_units`` label), i.e. they were divided by
    HARTREE_TO_EV relative to the raw states/Ha .DAT. Cross-check a KNOWN value:
    the single-channel DOS@Fermi (states/eV) * HARTREE_TO_EV must reproduce the
    raw legacy ``doss_dat_at_fermi`` (states/Ha) under the documented conversion.
    Also assert the stored curve magnitudes equal raw_total/H at the same grid
    point (not the un-converted raw value)."""
    dos = diamond_dos_props["dos_structure"]
    assert dos["dos_units"] == "states/eV/cell"

    legacy_ha = diamond_dos_props["doss_dat_at_fermi"]  # states/Ha, single chan
    per_chan_ev = dos["dos_at_fermi_per_channel_ev"]     # states/eV, single chan
    assert legacy_ha is not None and per_chan_ev is not None
    # The documented, exact relationship: states/eV * H == states/Ha.
    assert per_chan_ev * HARTREE_TO_EV == pytest.approx(legacy_ha, rel=1e-9,
                                                        abs=1e-9)

    # Cross-check the stored curve against the raw .DAT at the SAME strided grid
    # points: every stored DOS magnitude must equal |raw_total|/H (per-eV), NOT
    # the raw per-Ha value. (Reproduce the extractor's ceil-stride downsample so
    # we compare like-for-like rather than fighting the stride.)
    from mace.utils.dat_file_processor import DatFileProcessor
    dat = (
        find_data("DOSS/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss.out").parent
        / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss.DOSS.DAT"
    )
    parsed = DatFileProcessor().process_doss_dat_file(str(dat))
    nepts = parsed["num_energy_points"]
    raw_alpha = [abs(d) for d in parsed["total_dos"][:nepts]]  # states/Ha
    stride = max(1, -(-nepts // 200))
    stored_alpha = [pt[1] for pt in dos["total_dos_curve_ev"]["alpha"]]
    expected = [raw_alpha[i] / HARTREE_TO_EV for i in range(0, nepts, stride)]
    assert len(stored_alpha) == len(expected)
    for got, exp in zip(stored_alpha, expected):
        # EXACT /H relationship at every sampled point.
        assert got == pytest.approx(exp, rel=1e-9, abs=1e-12), (
            "stored DOS is not raw_total/H (states/eV) at a grid point"
        )
    # And the stored magnitudes are unambiguously NOT the raw per-Ha values:
    # the largest stored value is ~1/H of the largest raw value (a >20x drop).
    if max(raw_alpha) > 0:
        assert max(stored_alpha) < max(raw_alpha) / 5.0, (
            "DOS magnitudes appear to still be in states/Ha (not converted)"
        )


def test_dos_at_fermi_relationship_on_spin_polarized_file():
    """ISSUE 2: on a REAL spin-polarized metallic run the headline DOS@Fermi has
    a DEFINED relationship to the legacy single-channel ``doss_dat_at_fermi``
    (no silent factor-of-2). Asserts:
      * per_channel_ev * H == legacy doss_dat_at_fermi  (states/Ha) EXACTLY,
      * both_spins_ev == alpha + beta (the only place the 2x appears, clearly
        named), and == 2 * per_channel for this symmetric run,
      * the back-compat headline ``total_dos_at_fermi`` equals the SINGLE channel
        (does NOT silently double the legacy scalar)."""
    out = find_data("DOSS/1LiFSI-6EC-conf3_*_doss.out")
    extractor = CrystalPropertyExtractor(enable_tracking=False)
    p = extractor.extract_all_properties(out)
    dos = p["dos_structure"]
    assert dos["spin_polarized"] is True

    legacy = p["doss_dat_at_fermi"]  # single channel, states/Ha (e.g. 2559.2)
    per_chan = dos["dos_at_fermi_per_channel_ev"]
    both = dos["dos_at_fermi_both_spins_ev"]
    by_spin = dos["dos_at_fermi_by_spin"]
    assert legacy is not None and per_chan is not None and both is not None

    # The DEFINED relationship: per-channel (eV) * H reproduces the legacy scalar.
    assert per_chan * HARTREE_TO_EV == pytest.approx(legacy, rel=1e-9, abs=1e-9)
    # both-spins is the explicit alpha+beta sum (this is the ONLY 2x, and it is
    # named so it cannot be confused with the per-channel headline).
    assert both == pytest.approx(
        by_spin["alpha"] + by_spin["beta"], abs=1e-9)
    assert both == pytest.approx(2.0 * per_chan, rel=1e-9)
    # Back-compat headline tracks the single channel, NOT the doubled sum: it must
    # NOT equal 2x the legacy scalar (the original silent-2x defect).
    assert dos["total_dos_at_fermi"] == pytest.approx(per_chan, abs=1e-9)
    assert dos["total_dos_at_fermi"] * HARTREE_TO_EV == pytest.approx(
        legacy, rel=1e-9, abs=1e-9)
    assert dos["total_dos_at_fermi"] != pytest.approx(both, rel=1e-6)
    # The flat scalar rows expose BOTH, distinctly.
    assert p["dos_total_at_fermi"] == pytest.approx(per_chan, abs=1e-9)
    assert p["dos_at_fermi_both_spins_ev"] == pytest.approx(both, abs=1e-9)


def test_dos_categorizes_into_density_of_states(diamond_dos_props):
    """The JSON blob key must route to the density_of_states category (a bare
    'density_of_states' key would be miscategorized as 'structural' by the
    earlier 'density' branch; the 'dos_' prefix avoids that)."""
    extractor = CrystalPropertyExtractor(enable_tracking=False)
    assert extractor._categorize_property("dos_structure") == "density_of_states"
    assert extractor._categorize_property("dos_total_at_fermi") == "density_of_states"


def test_dos_structure_is_json_serializable_and_native(diamond_dos_props):
    """The payload must JSON-serialize (it lands in property_value_text) with no
    numpy types leaking in, and stay compact (NOT the full raw curve)."""
    dos = diamond_dos_props["dos_structure"]
    blob = json.dumps(dos)  # raises if any value is non-JSON (e.g. numpy)
    again = json.loads(blob)
    assert again["num_energy_points"] == dos["num_energy_points"]
    # Compact: the raw DOSS.DAT is 96 KB - 2 MB; the summary must be far smaller.
    assert len(blob) < 64 * 1024, "dos_structure blob must stay compact"

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

    _check_native(dos)


def test_missing_doss_dat_falls_back_gracefully(extractor):
    """A DOSS .out with no companion DOSS.DAT must not crash and must not emit a
    dos_structure payload (legacy scalar-only behavior preserved)."""
    dos_out = find_data("DOSS/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss.out")
    content = dos_out.read_text(errors="ignore")
    # Point the extractor at a directory with no companion DOSS.DAT so the
    # .DAT-derived structure is necessarily absent (the .out parse still runs).
    nonexistent = dos_out.parent / "__no_such_dir__" / dos_out.name
    props = extractor._extract_dos_properties(content, nonexistent)
    assert "dos_structure" not in props
    # No crash; calc still marked as a DOSS run via the .out content.
    assert props.get("calculation_type") == "DOSS"
    assert props.get("doss_dat_exists") is not True


# ---------------------------------------------------------------------------
# Spin / index / ragged robustness across the whole DOSS corpus.
# ---------------------------------------------------------------------------


def _all_doss_outs():
    """All `test/DOSS/*_doss.out` that have a DOSS.DAT companion. Empty (-> skip)
    when the gitignored corpus is absent."""
    if not TEST_DATA.is_dir():
        return []
    outs = []
    for out in sorted((TEST_DATA / "DOSS").glob("*_doss.out")):
        companion = out.parent / f"{out.stem}.DOSS.DAT"
        if companion.exists():
            outs.append(out)
    return outs


def test_dos_structure_sane_across_all_doss_files():
    """Across EVERY real DOSS file (incl. line-wrapped NPROJ=26, NEPTS=5002, and
    metallic runs): dos_structure is present, JSON-native, the gap-from-DOS
    matches the legacy scalar, the energy grid is monotonic, DOS is non-negative,
    DOS@Fermi is finite per spin, and per-spin DOS@Fermi indices never desync
    (both blocks share the same grid length)."""
    outs = _all_doss_outs()
    if not outs:
        pytest.skip("test/DOSS corpus not present (gitignored)")
    extractor = CrystalPropertyExtractor(enable_tracking=False)
    checked = 0
    for out in outs:
        props = extractor.extract_all_properties(out)  # must not raise
        dos = props.get("dos_structure")
        if dos is None:
            continue
        checked += 1

        # JSON-serializable, compact, native types.
        blob = json.dumps(dos)
        assert len(blob) < 128 * 1024, f"{out.name}: dos_structure blob too large"

        # Gap-from-DOS consistent with the validated legacy scalar.
        legacy = props.get("doss_dat_band_gap_ev")
        if legacy is not None and dos["band_gap_ev"] is not None:
            assert dos["band_gap_ev"] == pytest.approx(legacy, abs=1e-6), (
                f"{out.name}: dos_structure gap disagrees with doss_dat_band_gap_ev"
            )

        # metallic flag consistent with the legacy scalar.
        if dos["metallic"] is not None and props.get("doss_dat_metallic") is not None:
            assert dos["metallic"] == props["doss_dat_metallic"]

        # Spin channels: both present for a spin-polarized run, no desync.
        by_spin = dos["dos_at_fermi_by_spin"]
        curves = dos["total_dos_curve_ev"]
        if dos["spin_polarized"]:
            assert set(by_spin) >= {"alpha", "beta"}, f"{out.name}: missing a spin"
            assert set(curves) >= {"alpha", "beta"}
        for spin, val in by_spin.items():
            assert val is None or (math.isfinite(val) and val >= 0.0), (
                f"{out.name}: bad DOS@Fermi for {spin}"
            )

        # Per-spin curve sanity (monotonic energy, non-negative DOS, downsampled).
        for spin, curve in curves.items():
            es = [pt[0] for pt in curve]
            ds = [pt[1] for pt in curve]
            assert len(curve) <= 200, f"{out.name}/{spin}: >200 pts (off-by-one)"
            assert all(es[i] <= es[i + 1] for i in range(len(es) - 1)), (
                f"{out.name}/{spin}: non-monotonic energy grid"
            )
            assert all(d >= 0.0 for d in ds), f"{out.name}/{spin}: negative DOS"

        # ISSUE 4: num_proj == number of stored projection weights, and never
        # counts the TOTAL column.
        weights = dos.get("projection_integrated_weights") or {}
        if weights:
            assert dos["num_proj"] == len(weights), (
                f"{out.name}: num_proj {dos['num_proj']} != len(weights) {len(weights)}"
            )

        # ISSUE 2: the single-channel headline never silently doubles the legacy
        # scalar. per_channel * H == legacy doss_dat_at_fermi (when both present).
        legacy_ef = props.get("doss_dat_at_fermi")
        per_chan = dos.get("dos_at_fermi_per_channel_ev")
        if legacy_ef is not None and per_chan is not None:
            assert per_chan * HARTREE_TO_EV == pytest.approx(legacy_ef, rel=1e-6,
                                                             abs=1e-9), (
                f"{out.name}: per-channel DOS@Fermi*H disagrees with legacy scalar"
            )
            both = dos.get("dos_at_fermi_both_spins_ev")
            if both is not None and dos["spin_polarized"]:
                # both-spins is the SUM (>= single channel), not a silent 2x of
                # the headline reported value.
                assert both >= per_chan - 1e-9

        # Native-type guard for the whole payload.
        def _native(o):
            if isinstance(o, dict):
                for v in o.values():
                    _native(v)
            elif isinstance(o, list):
                for v in o:
                    _native(v)
            else:
                assert o is None or isinstance(o, (bool, int, float, str)), (
                    f"{out.name}: non-native {type(o)}"
                )

        _native(dos)

    assert checked > 1, "expected multiple DOSS files with a dos_structure payload"
