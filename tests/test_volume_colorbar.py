"""The color bar of a density surface colored by its electrostatic potential (Color by Potential, the
Volume view; mace plotting's dual-cube figure).

* The automatic range no longer starts at a false 0: a surface vertex on the grid's first plane came
  back from the inverse voxel transform a round-off outside the grid (-3.6e-15 on h-BN's hexagonal
  cell) and read the interpolator's fill value 0, so an all-positive potential got the range 0 to max.
  The bundled suite's h-BN and diamond CHARGE+POTENTIAL cubes are all positive on their surfaces.
* What the bar says: sign words only where the range holds both signs, else "all positive"/"all
  negative" with the ends named lowest/highest; the GUI (js/maprange.js potentialWords) and the CLI's
  caption (crystal_cubeviz_plotly.potential_scale_words) say the same (node runs the GUI's).
* The bar's ticks: round values at their true places, zero among them only when the range spans it.
"""
import contextlib
import io
import json
import lzma
import shutil
import subprocess

import pytest

from conftest import REPO_ROOT

np = pytest.importorskip("numpy")
JS = REPO_ROOT / "mace" / "gui" / "static" / "js"
SUITE = REPO_ROOT / "examples" / "suite"
RUNS = {"hBN": SUITE / "3-slab-2d" / "hBN" / "CHARGE+POTENTIAL", "diamond": SUITE / "2-dense-3d" / "diamond" / "CHARGE+POTENTIAL"}


@pytest.fixture
def E():
    """The cube engine: it imports plotly, which the CI test jobs do not install."""
    return pytest.importorskip("mace.plotting.engines.crystal_cubeviz_plotly", reason="the cube engine imports plotly")


def _cubes(tmp_path, name):
    """The suite run's DENS and POT cubes, decompressed into tmp_path."""
    out = []
    for kind in ("DENS", "POT"):
        src = RUNS[name] / f"{name}_charge+potential_{kind}.CUBE.xz"
        if not src.is_file():
            pytest.skip("the bundled example suite is not here")
        dst = tmp_path / src.name[:-3]
        dst.write_bytes(lzma.decompress(src.read_bytes()))
        out.append(dst)
    return out


def test_a_vertex_on_the_grids_edge_reads_the_grid_not_zero(E):
    pytest.importorskip("scipy")
    # a hexagonal grid's voxels (h-BN's) and a potential that is positive everywhere
    vv = np.array([[0.0822, -0.0475, 0.0], [0.0, 0.0949, 0.0], [0.0, 0.0, 0.0949]])
    n = np.array([6, 6, 6])
    data = 0.2 + np.random.default_rng(1).random(tuple(n)) * 0.1
    origin = np.array([0.1, -0.2, 0.3])
    # vertices on the first and last planes (index 0 and n-1), as marching cubes gives them, in Angstrom
    idx = np.array([[0, 0, 0], [0, 5, 2], [5, 0, 5], [5, 5, 5], [0, 2.5, 0], [2.2, 0, 4.9]], float)
    bohr = E.BOHR_TO_ANGSTROM
    verts = (origin + idx @ vv) * bohr
    vals = E.interpolate_cube_at_vertices(data, origin, n, vv, verts, bohr)
    assert np.all(vals >= 0.2), vals
    # a point well outside the grid still reads the fill value (nothing is extrapolated)
    far = (origin + np.array([[-1.0, 0, 0]]) @ vv) * bohr
    assert E.interpolate_cube_at_vertices(data, origin, n, vv, far, bohr)[0] == 0.0


@pytest.mark.parametrize("name", ["hBN", "diamond"])
def test_suite_potential_range_is_its_surfaces_own(E, tmp_path, name):
    pytest.importorskip("scipy")
    pytest.importorskip("skimage", reason="the CLI's mesh needs scikit-image")
    from mace.gui import volume
    dens, pot = _cubes(tmp_path, name)
    level = {"hBN": 0.02, "diamond": 0.5}[name]          # the GUI's default levels for these cubes
    r = volume.mep_range(str(pot), shape_path=str(dens), level=level)
    # all positive on these surfaces: the range is the sampled min..max (no false 0 at its low end)
    assert r["method"] == "cli-mesh" and 0 < r["vmin"] < r["vmax"], r
    with contextlib.redirect_stdout(io.StringIO()):
        fig = E.plot_dual_cube_isosurface(volume.read(str(dens)), volume.read(str(pot)), level)
    mesh = next(t for t in fig.data if t.type == "mesh3d")
    assert min(mesh.intensity) > 0 and mesh.cmin == pytest.approx(r["vmin"])
    # the CLI's colorbar names its unit, and its caption says the range is all positive (no sign words)
    assert "Ha e<sup>−1</sup>" in mesh.colorbar.title.text
    cap = next(a.text for a in fig.layout.annotations if "surface" in a.text or "potential" in a.text.lower())
    assert "All positive" in cap and "negative" not in cap.lower(), cap


def test_scale_words(E):
    w = E.potential_scale_words(-0.07, 0.07, "chemistry")
    assert w["spans"] and w["low"] == "red = negative (electron-rich)" and w["high"] == "blue = positive (electron-poor)"
    w = E.potential_scale_words(-0.07, 0.07, "physics")
    assert w["low"] == "blue = negative (electron-rich)" and w["high"] == "red = positive (electron-poor)"
    w = E.potential_scale_words(0.063, 0.734, "chemistry")
    assert not w["spans"] and w["note"] == "all positive" and w["low"] == "red = lowest" and w["high"] == "blue = highest"
    assert E.potential_scale_words(-0.5, -0.1)["note"] == "all negative"
    assert E.potential_scale_words(0.0, 0.3)["note"] == "all positive"


CASES = [(-0.07, 0.07, True), (-0.07, 0.07, False), (0.063, 0.734, True), (0.063, 0.734, False), (-0.5, -0.1, True),
         (0.0, 0.3, True), (-2e-4, 5e-4, True), (3.944, 4.215, True), (0.0, 0.0, True)]


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_gui_bar_says_what_the_cli_says_and_ticks_are_true(E, tmp_path):
    script = tmp_path / "run.mjs"
    script.write_text(f"import * as MR from {json.dumps((JS / 'maprange.js').as_uri())};\n"
                      f"const cases = {json.dumps(CASES)};\n"
                      "console.log(JSON.stringify(cases.map(([lo, hi, chem]) => ({ words: MR.potentialWords(lo, hi, chem),"
                      " ticks: MR.niceTicks(lo, hi, 5), text: MR.niceTicks(lo, hi, 5).map(MR.tickText) }))));\n")
    got = json.loads(subprocess.run(["node", str(script)], capture_output=True, text=True, check=True, timeout=60).stdout)
    for (lo, hi, chem), g in zip(CASES, got):
        assert g["words"] == E.potential_scale_words(lo, hi, "chemistry" if chem else "physics"), (lo, hi, chem)
        t = g["ticks"]
        if hi > lo:
            assert 3 <= len(t) <= 7 and all(lo - 1e-12 <= v <= hi + 1e-12 for v in t), (lo, hi, t)
            assert (0 in t) == (lo <= 0 <= hi), (lo, hi, t)
    # the texts: a real minus sign and superscripts, no 0 in the middle of an all-positive range
    by = {(c[0], c[1]): g for c, g in zip(CASES, got)}
    assert by[(-0.07, 0.07)]["text"] == ["−0.05", "0", "0.05"]
    assert by[(0.063, 0.734)]["text"] == ["0.2", "0.4", "0.6"]
    assert by[(3.944, 4.215)]["text"] == ["4", "4.1", "4.2"]
    assert by[(-2e-4, 5e-4)]["text"] == ["−2×10⁻⁴", "0", "2×10⁻⁴", "4×10⁻⁴"]
