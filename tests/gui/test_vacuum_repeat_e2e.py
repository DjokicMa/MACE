"""Repeats along a non-periodic axis (Settings › Viewport › Vacuum spacing): the
n×n×n presets keep a slab's c at 1; a custom range with more along c repeats
the atoms (and a cube's isosurfaces) at the display vector: Cell (a cube's box
height) or Gap (the atoms' height + the gap). Firefox; the cube is from the
test/ corpus (skipped without it)."""
import shutil
import time

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly", reason="the cube engine imports plotly")
pytest.importorskip("numpy")

CUBE = H.REPO_ROOT / "test" / "ECH3POT3" / "4LG_FSI_TopMiddle_2x2_ABAB_charge+potential_DENS.CUBE"
SLAB = "outputs/graphene_slab_sp.out"
VV = "window.MACE.state.viewers['viewer-volume']"
VM = "window.MACE.state.viewers['viewer-main']"


@pytest.fixture(scope="module")
def drv(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("vacuum")
    root = H.examples_copy(tmp)
    (root / "vol").mkdir()
    if CUBE.is_file():
        shutil.copyfile(CUBE, root / "vol" / "slab_DENS.CUBE")
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1", MACE_GUI_MAX_WORKERS="2")
    try:
        d = H.browser("firefox", width=1280, height=800)
    except BaseException:
        proc.terminate()
        proc.wait(timeout=15)
        raise
    try:
        H.load(d, base)
        d.execute_script("const s = window.__mace.settings; s.set('reduceMotion', 'on'); s.set('showBoundary', false); "
                         "s.set('volume.quality', 'low')")
        yield d
    finally:
        d.quit()
        proc.terminate()
        proc.wait(timeout=15)


def run(drv, cmd):
    drv.execute_script("window.__mace.run(arguments[0])", cmd)
    time.sleep(0.6)


def custom(drv, hi):
    """Display Range › Custom… with these ends (from 0)."""
    run(drv, "view.rangeCustom")
    H.wait(drv, "const d = document.getElementById('range-custom'); return !!(d && d.open)", 5)
    drv.execute_script("""const d = document.getElementById('range-custom');
      arguments[0].forEach((x, k) => { d.querySelector(`[data-hi="${k}"]`).value = String(x); });
      d.querySelector('[data-act="apply"]').click();""", hi)
    H.wait(drv, "return !document.getElementById('range-custom').open", 5)
    time.sleep(1.0)


def label(drv):
    return drv.execute_script("return document.getElementById('tb-range-label').textContent")


def zs(drv, viewer):
    return drv.execute_script(f"return {viewer}.getModel().selectedAtoms({{}}).map((a) => a.z)")


def test_slab_structure_repeats_c_at_the_gap(drv):
    drv.execute_script("const s = window.__mace.settings; s.set('viewport.vacuumMode', 'gap'); s.set('viewport.vacuumGap', 10)")
    kind = drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then((x) => d(x && x.kind))", SLAB)
    assert kind == "structure"
    H.wait(drv, f"const v = {VM}; return !!(v && v.getModel())", 30)
    time.sleep(0.8)
    z1 = zs(drv, VM)
    n0, h = len(z1), max(z1) - min(z1)
    # the default range input keeps c at 1 (and says so)
    run(drv, "view.range3")
    H.wait(drv, f"return window.__mace.viewState().displayedAtoms === {n0 * 9}", 20)
    assert label(drv) == "3×3×1"
    assert "not periodic along c" in drv.execute_script("return document.getElementById('tb-range').dataset.tip")
    # the user asks for 3 along c: three layers, h + 10 Å apart
    custom(drv, [1, 1, 3])
    H.wait(drv, f"return window.__mace.viewState().displayedAtoms === {n0 * 3}", 20)
    assert label(drv) == "1×1×3"
    assert "10 Å gap" in drv.execute_script("return document.getElementById('tb-range').dataset.tip")
    z = sorted(zs(drv, VM))
    for k in range(3):
        layer = z[k * n0:(k + 1) * n0]
        assert abs(min(layer) - (min(z1) + k * (h + 10))) < 1e-3, (k, layer[:3], min(z1), h)
    assert not H.errors(drv), H.errors(drv)
    drv.execute_script("window.__mace.closeAll(); window.__mace.settings.set('viewport.vacuumMode', 'cell')")
    time.sleep(0.3)


def test_slab_cube_2x2x2_at_the_cell_spacing(drv):
    if not CUBE.is_file():
        pytest.skip("test/ corpus not present (gitignored)")
    kind = drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then((x) => d(x && x.kind))", "vol/slab_DENS.CUBE")
    assert kind == "volume"
    H.wait(drv, "const s = window.__mace.volumeView && window.__mace.volumeView.state(); return !!(s && s.surfaces > 0 && s.display)", 90)
    time.sleep(0.6)
    n = drv.execute_script("return window.__mace.volumeView.state().display.atoms")
    z1 = zs(drv, VV)
    # the cube's box height along c (its third voxel vector times its count, bohr to Å)
    head = CUBE.read_text(errors="replace").splitlines()[5].split()
    hc = int(head[0]) * float(head[3]) * 0.529177210903
    run(drv, "view.range2")
    assert label(drv) == "2×2×1"
    custom(drv, [2, 2, 2])
    H.wait(drv, "return window.__mace.volumeView.state().surfaces === 8", 30)
    s = drv.execute_script("return window.__mace.volumeView.state()")
    assert s["display"]["images"] == 8 and not s["display"]["capped"] and s["display"]["atoms"] == 8 * n, s["display"]
    assert label(drv) == "2×2×2" and "Cell vacuum spacing" in drv.execute_script("return document.getElementById('tb-range').dataset.tip")
    # the upper layer is the lower one moved up by the cube's height
    z = sorted(zs(drv, VV))
    lo, hi = z[:4 * n], z[4 * n:]
    assert abs(min(lo) - min(z1)) < 1e-3 and abs(min(hi) - (min(z1) + hc)) < 1e-3 and abs(max(hi) - (max(z1) + hc)) < 1e-3, (min(z1), hc, lo[0], hi[0])
    vz = drv.execute_script("const V = window.__mace.volumeView.vertices(), z = []; for (let i = 2; i < V.length; i += 3) z.push(V[i]); "
                            "return [Math.min(...z), Math.max(...z)]")
    assert vz[1] - vz[0] > hc, (vz, hc)
    assert not H.errors(drv), H.errors(drv)
