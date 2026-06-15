"""Regression tests for the structured band-STRUCTURE extraction.

`_extract_band_structure_properties` historically stored only scalar gap values.
It now also stores the actual band structure via the REAL extractor:
  - a compact ``band_structure`` JSON summary (k-path with real high-symmetry
    labels from the .d3, band energies AT the high-symmetry points, direct vs
    indirect gap with their k-locations, counts, Fermi energy, source file), and
  - queryable scalar rows (``band_direct_gap_ev``, ``band_indirect_gap_ev``,
    ``band_fundamental_gap_ev``, ``band_gap_is_direct``, ...).

These tests exercise the real invocation path
(``CrystalPropertyExtractor.extract_all_properties`` on a real ``test/BAND``
output) and assert the new structured data is present, JSON-serializable, and
consistent with the existing scalar gap. They skip cleanly when the gitignored
``test/`` corpus is absent.
"""
import glob
import json
import math
from pathlib import Path

import pytest

from conftest import TEST_DATA, find_data
from mace.utils.property_extractor import CrystalPropertyExtractor, HARTREE_TO_EV


@pytest.fixture(scope="module")
def diamond_props():
    """Real extractor output for the smallest clean spin-polarized band calc
    (diamond, 36 bands), via the full ``extract_all_properties`` path."""
    band_out = find_data(
        "BAND/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band.out",
        must_contain="FROM BAND",
    )
    # enable_tracking=False -> no DB is touched (never touches ./materials.db).
    extractor = CrystalPropertyExtractor(enable_tracking=False)
    return extractor.extract_all_properties(band_out)


def test_structured_band_structure_present(diamond_props):
    """The actual band STRUCTURE (not just a gap scalar) is stored."""
    bs = diamond_props.get("band_structure")
    assert isinstance(bs, dict), "expected a structured band_structure payload"
    # Counts are sane.
    assert bs["num_kpoints"] > 0
    assert bs["num_bands"] > 0
    assert bs["num_spins"] >= 1
    # Source file is referenced so the raw data can be re-parsed on demand.
    assert bs["source_file"].endswith("BAND.DAT")


def test_kpath_labels_and_nodes_real(diamond_props):
    """High-symmetry labels come from the .d3 SeeKPath path (real G/X/L names),
    aligned with the BAND.DAT panel node indices/distances."""
    bs = diamond_props["band_structure"]
    labels = bs["kpath_labels"]
    nodes = bs["kpath_node_indices"]
    dists = bs["kpath_node_distances"]
    assert labels, "k-path labels must be non-empty"
    # Real labels, not the CRYSTAL placeholder "A".
    assert any(l not in ("A", "") for l in labels)
    assert "GAMMA" in labels  # diamond path is GAMMA-X-U|K-GAMMA-L-W-X
    # One label / distance per node, monotonically increasing distances.
    assert len(labels) == len(nodes) == len(dists)
    assert all(dists[i] <= dists[i + 1] for i in range(len(dists) - 1))
    # Band energies stored AT each high-symmetry node (one row per node).
    hs = bs["high_symmetry_band_energies_ev"]
    assert len(hs) == len(nodes)
    assert all(len(row) == bs["num_bands"] for row in hs if row)


def test_direct_indirect_gap_and_edges(diamond_props):
    """Direct & indirect gaps are finite, non-negative, and physically ordered
    (direct >= indirect); band edges carry k-locations."""
    bs = diamond_props["band_structure"]
    direct = bs["direct_gap_ev"]
    indirect = bs["indirect_gap_ev"]
    for g in (direct, indirect):
        assert isinstance(g, float) and math.isfinite(g) and g >= 0.0
    # The minimum (fundamental) gap cannot exceed the minimum direct gap.
    assert indirect <= direct + 1e-6
    assert bs["fundamental_gap_ev"] == pytest.approx(indirect, abs=1e-6)
    assert isinstance(bs["is_direct_gap"], bool)
    # Diamond is an indirect-gap semiconductor.
    assert bs["is_direct_gap"] is False
    # VBM/CBM carry energy + k-location.
    for edge in ("vbm", "cbm"):
        e = bs[edge]
        assert math.isfinite(e["energy_ev"])
        assert isinstance(e["kpoint_index"], int)
        assert e["band_index"] >= 1


def test_fundamental_gap_consistent_with_scalar(diamond_props):
    """The new structured fundamental gap matches the pre-existing scalar
    ``band_dat_band_gap_ev`` (the validated band-index recompute)."""
    bs = diamond_props["band_structure"]
    scalar_gap = diamond_props["band_dat_band_gap_ev"]
    assert scalar_gap is not None
    assert bs["fundamental_gap_ev"] == pytest.approx(scalar_gap, abs=0.01)


def test_queryable_scalar_rows_added(diamond_props):
    """Direct/indirect/fundamental gaps + gap-directness are also exposed as
    flat scalar props (numerically filterable), without losing the legacy keys."""
    p = diamond_props
    for k in (
        "band_direct_gap_ev",
        "band_indirect_gap_ev",
        "band_fundamental_gap_ev",
        "band_vbm_kpoint_index",
        "band_cbm_kpoint_index",
    ):
        assert k in p, f"missing scalar prop {k}"
    assert isinstance(p["band_gap_is_direct"], bool)
    assert isinstance(p["band_kpath_label_string"], str) and p["band_kpath_label_string"]
    # Legacy scalars preserved (layered, not rewritten).
    assert p["band_dat_band_gap_ev"] is not None
    assert p["band_dat_num_bands"] > 0
    assert p["calculation_type"] == "BAND"


def test_band_structure_is_json_serializable_and_native(diamond_props):
    """The payload must JSON-serialize (it lands in property_value_text) with no
    numpy types leaking in."""
    bs = diamond_props["band_structure"]
    blob = json.dumps(bs)  # raises if any value is non-JSON (e.g. numpy)
    again = json.loads(blob)
    assert again["num_bands"] == bs["num_bands"]

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

    _check_native(bs)


def test_missing_band_dat_falls_back_gracefully(extractor):
    """A band .out with no companion BAND.DAT must not crash and must not emit a
    band_structure payload (scalar-only behavior preserved)."""
    # Pick any band .out, then point the extractor at it from a directory with
    # no BAND.DAT by using a stem that has no companion (handled in-corpus).
    band_out = find_data(
        "BAND/3,4^2T7_CA_BULK_OPTGEOM_TZ_opt_B3LYP-D3-D3_optimized_rev1_sp_B3LYP-D3-D3_optimized_band.out",
        must_contain="FROM BAND",
    )
    companion = band_out.parent / f"{band_out.stem}.BAND.DAT"
    if companion.exists():
        pytest.skip("chosen .out unexpectedly has a BAND.DAT companion")
    content = band_out.read_text(errors="ignore")
    props = extractor._extract_band_structure_properties(content, band_out)
    # No crash; structured payload absent; calc still marked as a BAND run.
    assert "band_structure" not in props
    assert props.get("calculation_type") == "BAND"


# ---------------------------------------------------------------------------
# Regression tests for the two correctness defects an adversarial review found:
#   BUG 1 (index desync): VBM/CBM indices were computed over a FILTERED row
#          list but used to index the UNFILTERED k-geometry, so a single ragged
#          row offset every reported k-point index / distance / label.
#   BUG 2 (spin-block inconsistency): the fundamental gap used only the first
#          spin block, while the legacy scalar uses ALL spins — so a magnetic
#          system whose beta channel has the smaller gap was silently overstated
#          and disagreed with band_dat_band_gap_ev.
# ---------------------------------------------------------------------------


def _all_band_outs():
    """All `test/BAND/*_band.out` that have a BAND.DAT companion and are real
    BAND runs. Empty (-> skip) when the gitignored corpus is absent."""
    if not TEST_DATA.is_dir():
        return []
    outs = []
    for out in sorted((TEST_DATA / "BAND").glob("*_band.out")):
        companion = out.parent / f"{out.stem}.BAND.DAT"
        if not companion.exists():
            continue
        try:
            if "FROM BAND" in out.read_text(errors="ignore"):
                outs.append(out)
        except OSError:
            continue
    return outs


def _find_magnetic_band_out():
    """A genuinely spin-polarized BAND.DAT (alpha block != beta block) with a
    companion .out, or None. Verifies the data is truly magnetic rather than a
    2-block non-magnetic dump (alpha == beta, as for diamond)."""
    try:
        from mace.utils.dat_file_processor import DatFileProcessor
    except ImportError:  # pragma: no cover
        from dat_file_processor import DatFileProcessor
    proc = DatFileProcessor()
    for out in _all_band_outs():
        companion = out.parent / f"{out.stem}.BAND.DAT"
        try:
            di = proc.process_band_dat_file(companion)
        except Exception:
            continue
        ev = di.get("eigenvalues") or []
        nk = di.get("num_k_points") or 0
        ns = di.get("num_spins") or 1
        if ns < 2 or not nk or len(ev) < 2 * nk:
            continue
        alpha, beta = ev[:nk], ev[nk : 2 * nk]
        for ra, rb in zip(alpha, beta):
            if len(ra) != len(rb):
                continue
            if any(abs(x - y) > 1e-6 for x, y in zip(ra, rb)):
                return out
    return None


def test_fundamental_gap_consistent_across_all_band_files():
    """(a) band_fundamental_gap_ev must agree with the validated legacy scalar
    band_dat_band_gap_ev for EVERY real BAND file — not just non-magnetic
    diamond. This is the cross-check that the new analysis considers all spin
    channels (the legacy scalar does) and that the index handling never corrupts
    the gap."""
    outs = _all_band_outs()
    if not outs:
        pytest.skip("test/BAND corpus not present (gitignored)")
    extractor = CrystalPropertyExtractor(enable_tracking=False)
    checked = 0
    for out in outs:
        props = extractor.extract_all_properties(out)
        legacy = props.get("band_dat_band_gap_ev")
        fundamental = props.get("band_fundamental_gap_ev")
        if legacy is None or fundamental is None:
            continue
        checked += 1
        assert fundamental == pytest.approx(legacy, abs=0.01), (
            f"{out.name}: band_fundamental_gap_ev={fundamental} disagrees with "
            f"legacy band_dat_band_gap_ev={legacy}"
        )
    assert checked > 1, "expected multiple BAND files with a computed gap"


def test_fundamental_gap_consistent_on_magnetic_file():
    """(a, magnetic) On a genuinely spin-polarized file (alpha block != beta
    block) the all-spin fundamental gap still matches the all-spin legacy
    scalar. If the corpus has no truly magnetic file, this is skipped and the
    dedicated both-spin-blocks unit test (below) carries the proof instead."""
    out = _find_magnetic_band_out()
    if out is None:
        pytest.skip(
            "no genuinely magnetic (alpha!=beta) BAND.DAT in test/BAND; "
            "see test_both_spin_blocks_considered for the spin-coverage proof"
        )
    extractor = CrystalPropertyExtractor(enable_tracking=False)
    props = extractor.extract_all_properties(out)
    legacy = props.get("band_dat_band_gap_ev")
    fundamental = props.get("band_fundamental_gap_ev")
    assert legacy is not None and fundamental is not None
    assert fundamental == pytest.approx(legacy, abs=0.01), (
        f"magnetic {out.name}: fundamental={fundamental} != legacy={legacy}"
    )


def test_both_spin_blocks_considered():
    """(BUG 2, direct) The fundamental gap must come from whichever spin channel
    carries the smaller gap. Craft a 2-spin payload whose BETA channel has a
    much smaller gap than ALPHA and assert the analysis reports the BETA gap
    (not the alpha-only one) and locates the CBM in the beta block. This is a
    self-contained logic test that always runs (no corpus needed)."""
    extractor = CrystalPropertyExtractor(enable_tracking=False)
    # n=1 occupied band, nb=2, nk=2, ns=2. Eigenvalues are Fermi-referenced Ha.
    #   alpha k0/k1: occ=-0.10, virt=0.30  -> alpha gap = 0.40 Ha
    #   beta  k0   : occ=-0.10, virt=0.05  -> beta CBM 0.05 -> gap 0.15 Ha (min!)
    #   beta  k1   : occ=-0.10, virt=0.20
    eigenvalues = [
        [-0.10, 0.30],
        [-0.10, 0.30],
        [-0.10, 0.05],
        [-0.10, 0.20],
    ]
    dat_info = {
        "eigenvalues": eigenvalues,
        "num_k_points": 2,
        "num_bands": 2,
        "num_spins": 2,
        "k_points": [0.0, 1.0],
    }
    # Avoid file IO for k-path nodes/labels: provide two nodes at the two kpts.
    extractor._parse_band_dat_kpath_nodes = lambda f: ([1, 2], ["A", "B"], 0.0)
    extractor._kpath_labels_from_d3 = lambda o, n: []
    summary, extra = extractor._build_band_structure_summary(
        dat_info, "FERMI ENERGY 0.0", Path("x.out"), Path("x.BAND.DAT"), 1
    )
    beta_gap_ev = 0.15 * HARTREE_TO_EV
    alpha_only_gap_ev = 0.40 * HARTREE_TO_EV
    assert summary["fundamental_gap_ev"] == pytest.approx(beta_gap_ev, abs=1e-6)
    # It must NOT be the alpha-only gap (that was the bug).
    assert abs(summary["fundamental_gap_ev"] - alpha_only_gap_ev) > 0.5
    assert extra["band_fundamental_gap_ev"] == pytest.approx(beta_gap_ev, abs=1e-6)
    # The CBM edge sits at the beta block's k0 (global row 2 -> kidx 0, label A).
    assert summary["cbm"]["kpoint_index"] == 0
    assert summary["cbm"]["kpoint_label"] == "A"
    # k-point index stays a valid index into the per-spin-block geometry.
    assert 0 <= summary["cbm"]["kpoint_index"] < dat_info["num_k_points"]


def test_ragged_file_no_spurious_vbm_or_index_corruption():
    """(BUG 1) On the real ragged 4LG file the analysis must not crash, must
    clamp to a sane (>= 0) gap consistent with the legacy scalar, and must keep
    every reported k-point index inside [0, num_kpoints) (no filtered/unfiltered
    desync). The earlier code reported VBM ~96 eV with a gap of ~-96 eV and an
    index taken from a filtered list."""
    matches = (
        sorted(glob.glob(str(TEST_DATA / "BAND" / "4LG_2x2_AA*_band.out")))
        if TEST_DATA.is_dir()
        else []
    )
    band_out = None
    for m in matches:
        out = Path(m)
        companion = out.parent / f"{out.stem}.BAND.DAT"
        if companion.exists() and "FROM BAND" in out.read_text(errors="ignore"):
            band_out = out
            break
    if band_out is None:
        pytest.skip("ragged 4LG_2x2_AA band file not present in test/BAND")

    extractor = CrystalPropertyExtractor(enable_tracking=False)
    props = extractor.extract_all_properties(band_out)  # must not raise
    bs = props.get("band_structure")
    assert isinstance(bs, dict)
    nk = bs["num_kpoints"]
    assert nk > 0

    legacy = props.get("band_dat_band_gap_ev")
    fundamental = props.get("band_fundamental_gap_ev")
    assert legacy is not None and fundamental is not None
    # No spurious gap: clamped >= 0 and consistent with the legacy scalar
    # (this ragged slab parses as metallic -> ~0 eV, not a negative ~-96 eV).
    assert fundamental >= 0.0
    assert bs["indirect_gap_ev"] >= 0.0
    assert bs["direct_gap_ev"] >= 0.0
    assert fundamental == pytest.approx(legacy, abs=0.01)

    # No index corruption: every reported k-index is a valid index into the
    # per-spin-block k geometry (k_abscissa / node_indices span one block).
    for edge in ("vbm", "cbm"):
        e = bs[edge]
        ki = e["kpoint_index"]
        assert isinstance(ki, int) and 0 <= ki < nk, (
            f"{edge} kpoint_index {ki} out of range [0,{nk})"
        )
        # The k-distance, if present, must come from the in-range k-abscissa.
        if e["kpath_distance"] is not None:
            assert math.isfinite(e["kpath_distance"])
    # The structured scalar VBM/CBM indices match the JSON blob (no desync).
    assert props["band_vbm_kpoint_index"] == bs["vbm"]["kpoint_index"]
    assert props["band_cbm_kpoint_index"] == bs["cbm"]["kpoint_index"]
