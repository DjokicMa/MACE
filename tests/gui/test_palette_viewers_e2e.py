"""Element color palettes (1.1.6) in the views that draw atoms on their own:
FREQ modes (docs/freq.js) and volumes (docs/volume.js). The palette being
edited in View › Element Colors shows in them at once, and closing the
editor puts the default palette back. Real outputs from the test/ corpus
(skipped without it); the repo-data checks are in test_themes_palettes_e2e.
"""
import shutil
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")

FREQ = H.REPO_ROOT / "test" / "FREQ" / "2_dia2_opt_rev1_freq_B3LYP-D3-D3_optimized.out"
CUBE = H.REPO_ROOT / "test" / "ECH3POT3" / ("DEC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_sp_"
                                            "HSESOL3C_optimized_charge+potential_DENS.CUBE")
# the colors every element of a model is drawn in: the map its style carries, per element
ATOM_COLORS = """const v = arguments[0] === 'volume' ? window.__mace.volumeView.viewer() : window.MACE.state.viewers['viewer-freq'];
  const m = v && v.getModel(); if (!m) return null;
  const out = {}; for (const a of m.selectedAtoms({})) { const cs = a.style.sphere && a.style.sphere.colorscheme;
    out[a.elem] = cs && cs.map ? cs.map[a.elem] : null; } return out"""


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    if not (FREQ.is_file() and CUBE.is_file()):
        pytest.skip("test/ corpus not present (gitignored)")
    tmp = tmp_path_factory.mktemp("palviews")
    root = H.examples_copy(tmp)
    (root / "freq").mkdir()
    shutil.copy2(FREQ, root / "freq" / FREQ.name)
    (root / "vol").mkdir()
    shutil.copyfile(CUBE, root / "vol" / "mol_DENS.CUBE")
    a = U.start_app(tmp, root=root, MACE_GUI_MAX_WORKERS="2")
    try:
        yield a
    finally:
        U.stop_app(a)


def draft(drv, colors):
    """Open Element Colors and change these elements (a live draft, not saved)."""
    drv.execute_script("window.__mace.run('view.elementColors')")
    H.wait(drv, "return document.getElementById('palette-editor') && document.getElementById('palette-editor').open", 10)
    for sym, hx in colors.items():
        drv.execute_script("""document.querySelector(`#palette-editor .pe-cell[data-sym="${arguments[0]}"]`).click();
            const c = document.getElementById('pe-color'); c.value = arguments[1]; c.dispatchEvent(new Event('input', {bubbles: true}))""",
                           sym, hx)
    time.sleep(0.3)


@pytest.mark.parametrize("kind", ["freq", "volume"])
def test_the_edited_palette_shows_in_freq_and_volume_views(app, kind):
    drv = app[0]
    U.reset(drv)
    if kind == "freq":
        drv.execute_async_script("const d = arguments[1]; window.__mace.openModes(arguments[0]).then(() => d(1))", f"freq/{FREQ.name}")
        H.wait(drv, "const v = window.__mace.freqView, s = v && v.state(); return s && s.mode", 60)
    else:
        drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then(() => d(1), () => d(0))", "vol/mol_DENS.CUBE")
        H.wait(drv, "const s = window.__mace.volumeView && window.__mace.volumeView.state(); return s && s.surfaces > 0", 90)
    time.sleep(0.4)
    before = drv.execute_script(ATOM_COLORS, kind)
    assert before and before.get("C") == drv.execute_script("return window.MACE.state.colors.schemes.vesta.C"), before
    draft(drv, {"C": "#00aa55", "H": "#aa0055"})        # diamond has only C; the molecule C, H and O
    after = None
    for _ in range(50):
        after = drv.execute_script(ATOM_COLORS, kind)
        if after and after.get("C") == "#00aa55" and after.get("H", "#aa0055") == "#aa0055":
            break
        time.sleep(0.1)
    assert after.get("C") == "#00aa55" and after.get("H", "#aa0055") == "#aa0055", after
    assert kind == "freq" or set(after) >= {"C", "H", "O"}, after
    others = {k: v for k, v in after.items() if k not in ("C", "H")}
    assert others == {k: v for k, v in before.items() if k not in ("C", "H")}, (after, before)
    drv.execute_script("document.getElementById('palette-editor').close()")      # not saved: the default comes back
    for _ in range(50):
        after = drv.execute_script(ATOM_COLORS, kind)
        if after == before:
            break
        time.sleep(0.1)
    assert after == before, (after, before)
    assert not [e for e in H.errors(drv) if "ResizeObserver loop" not in e], H.errors(drv)
