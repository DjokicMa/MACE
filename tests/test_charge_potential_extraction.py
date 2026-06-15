"""Regression tests for CHARGE+POTENTIAL (CRYSTAL ECH3/POT3) extraction.

A CRYSTAL ECH3 (3D charge density) / POT3 (3D electrostatic potential) job used
to store NOTHING in the materials DB: ``extract_all_properties`` had no
charge/potential extractor. ``_extract_charge_potential_properties`` now stores
what the .out text HONESTLY exposes, mirroring ``_build_band_structure_summary``:

  - a compact ``charge_potential`` JSON summary (has_charge_density /
    has_electrostatic_potential booleans, grid divisions + number of grid points,
    real-space bounding box, multipole order, spin flag, SCF/Fermi energy, the
    list of computed 3D properties, and the names of the generated ``*.CUBE``
    grid sidecar files), and
  - queryable flat scalar/string rows (``chargepot_has_ech3``,
    ``chargepot_has_pot3``, ``chargepot_grid_points``, ``chargepot_grid_files``,
    ...).

The rich numerical payload (the 3D grids themselves) lives in BINARY/large
``*_DENS.CUBE`` / ``*_POT.CUBE`` / ``*_SPIN.CUBE`` sidecars, which a text
extractor must NOT parse -- this extractor only RECORDS references to them so the
existing mace plotting cube viewer (which globs ``*.CUBE``) can find them.

These tests exercise the REAL invocation path
(``CrystalPropertyExtractor.extract_all_properties`` on a real
``test/ECH3POT3`` output) and skip cleanly when the gitignored ``test/`` corpus
is absent. ``enable_tracking=False`` -> no DB is ever touched.
"""
import glob
import json
from pathlib import Path

import pytest

from conftest import TEST_DATA, find_data
from mace.utils.property_extractor import CrystalPropertyExtractor


@pytest.fixture(scope="module")
def chargepot_out():
    """A real ECH3/POT3 (.out) charge+potential output."""
    return find_data(
        "ECH3POT3/*_charge+potential.out",
        must_contain="ECH3 START",
    )


@pytest.fixture(scope="module")
def chargepot_props(chargepot_out):
    """Real extractor output via the full ``extract_all_properties`` path.

    ``enable_tracking=False`` -> never touches or creates ./materials.db.
    """
    extractor = CrystalPropertyExtractor(enable_tracking=False)
    return extractor.extract_all_properties(chargepot_out)


def test_charge_potential_summary_present(chargepot_props):
    """The compact ``charge_potential`` summary is populated for an ECH3/POT3 calc."""
    cp = chargepot_props.get("charge_potential")
    assert isinstance(cp, dict), "expected a structured charge_potential payload"
    # ECH3 ran (charge density) and POT3 ran (electrostatic potential).
    assert cp["has_charge_density"] is True
    assert cp["has_electrostatic_potential"] is True
    assert "source" in cp and cp["source"]
    # Computed-property provenance list is sane.
    computed = cp["computed_properties"]
    assert isinstance(computed, list)
    assert "charge_density" in computed
    assert "electrostatic_potential" in computed


def test_grid_metadata_from_text(chargepot_props):
    """Grid divisions / number of points are parsed from the .out text and are
    internally consistent (nx*ny*nz == num_grid_points)."""
    cp = chargepot_props["charge_potential"]
    dims = cp["grid_divisions"]
    assert isinstance(dims, list) and len(dims) == 3
    assert all(isinstance(d, int) and d > 0 for d in dims)
    npts = cp["num_grid_points"]
    assert isinstance(npts, int)
    assert npts == dims[0] * dims[1] * dims[2]
    # Real-space bounding box parsed for all three axes, lo < hi each.
    box = cp["coordinate_range"]
    assert set(box.keys()) == {"x", "y", "z"}
    for axis in ("x", "y", "z"):
        lo, hi = box[axis]
        assert hi > lo


def test_grid_file_references(chargepot_props, chargepot_out):
    """The generated ``*.CUBE`` grid sidecars are referenced by name (so the cube
    plotting viewer can find them), and every referenced file actually exists on
    disk in the .out's directory."""
    cp = chargepot_props["charge_potential"]
    grid_files = cp["grid_files"]
    assert isinstance(grid_files, list) and len(grid_files) >= 1
    out_dir = Path(chargepot_out).parent
    for name in grid_files:
        assert name.upper().endswith(".CUBE")
        assert (out_dir / name).exists(), f"referenced grid file missing: {name}"
    # The DENS (charge) and POT (potential) cubes are both present.
    joined = " ".join(grid_files).upper()
    assert "DENS" in joined
    assert "POT" in joined


def test_flat_scalar_rows(chargepot_props):
    """Queryable flat scalar/string rows populate sanely (distinct from the JSON
    blob) and agree with the summary."""
    props = chargepot_props
    assert props["chargepot_has_ech3"] is True
    assert props["chargepot_has_pot3"] is True
    assert isinstance(props["chargepot_grid_points"], int)
    assert props["chargepot_grid_points"] > 0
    # grid_dims string is "nxXnyXnz" and multiplies back to the point count.
    dims_str = props["chargepot_grid_dims"]
    parts = [int(x) for x in dims_str.split("x")]
    assert len(parts) == 3
    assert parts[0] * parts[1] * parts[2] == props["chargepot_grid_points"]
    # Grid files string + count agree.
    files = props["chargepot_grid_files"].split(",")
    assert props["chargepot_num_grid_files"] == len(files)
    # SCF total energy was on the page and parsed as a float.
    assert isinstance(props["chargepot_scf_total_energy_au"], float)


def test_summary_is_json_serializable(chargepot_props):
    """The whole charge_potential payload is JSON-serializable (native types only)."""
    cp = chargepot_props["charge_potential"]
    dumped = json.dumps(cp)  # raises TypeError if a non-native type leaked in
    assert isinstance(dumped, str) and len(dumped) > 0


def test_does_not_skip_or_crash_on_real_corpus(chargepot_props):
    """The extractor neither skips an ECH3/POT3 file nor returns an empty payload
    -- it stores something USEFUL where it previously stored nothing."""
    cp = chargepot_props.get("charge_potential")
    assert cp is not None
    # At minimum: confirmation both 3D properties ran + at least one grid file.
    assert cp["has_charge_density"] and cp["has_electrostatic_potential"]
    assert len(cp.get("grid_files", [])) >= 1


def test_non_charge_calc_is_untouched():
    """A non-ECH3/POT3 output must contribute NO charge_potential key and NO
    ``chargepot_`` scalars -- the extractor self-skips (graceful fallback)."""
    if not TEST_DATA.is_dir():
        pytest.skip("test/ data corpus not present (gitignored)")
    # Pick any .out that is NOT in the ECH3POT3 set and has no ECH3 marker.
    candidates = [
        p for p in sorted(glob.glob(str(TEST_DATA / "*" / "*.out")))
        if "ECH3POT3" not in p
    ]
    chosen = None
    for p in candidates:
        try:
            if "ECH3 START" not in Path(p).read_text(errors="ignore"):
                chosen = p
                break
        except OSError:
            continue
    if chosen is None:
        pytest.skip("no non-charge .out available in test/ corpus")
    extractor = CrystalPropertyExtractor(enable_tracking=False)
    props = extractor.extract_all_properties(Path(chosen))
    assert "charge_potential" not in props
    assert not any(k.startswith("chargepot_") for k in props)


def test_multiple_ech3pot3_files_populate():
    """Across several real ECH3POT3 files, each populates a sane summary with its
    own per-file grid dimensions (grids genuinely vary file-to-file)."""
    if not TEST_DATA.is_dir():
        pytest.skip("test/ data corpus not present (gitignored)")
    files = sorted(glob.glob(str(TEST_DATA / "ECH3POT3" / "*_charge+potential.out")))
    if not files:
        pytest.skip("no ECH3POT3 outputs in test/ corpus")
    extractor = CrystalPropertyExtractor(enable_tracking=False)
    seen_points = []
    for p in files[:5]:
        cp = extractor.extract_all_properties(Path(p)).get("charge_potential")
        assert cp is not None, f"no charge_potential for {Path(p).name}"
        assert cp["has_charge_density"] and cp["has_electrostatic_potential"]
        assert cp["num_grid_points"] > 0
        seen_points.append(cp["num_grid_points"])
    # The corpus has genuinely different grids -> at least two distinct sizes.
    assert len(set(seen_points)) >= 2
