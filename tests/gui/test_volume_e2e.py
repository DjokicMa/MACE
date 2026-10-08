"""The Volume document in headless Firefox (plan E5): real clicks on a fresh
profile, real corpus cubes copied under neutral names (skipped without the
gitignored corpus).

A molecule's density opens as an isosurface whose vertex count follows the
level; a spin density shows both lobes; the hexagonal slab's surface sits on
its atoms (the skewed voxel matrix is right); a density colored by its
potential gets the CLI's range and colors, a custom range and Reverse
Colors change it, and a cube of another cell is refused."""
import shutil
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly", reason="the cube engine imports plotly")
np = pytest.importorskip("numpy")
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.by import By  # noqa: E402

CORPUS = H.REPO_ROOT / "test" / "ECH3POT3"
MOL = "DEC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized_charge+potential"
SLAB = "4LG_FSI_TopMiddle_2x2_ABAB_charge+potential"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    if not (CORPUS / f"{MOL}_DENS.CUBE").is_file():
        pytest.skip("test/ corpus not present (gitignored)")
    tmp = tmp_path_factory.mktemp("volume_e2e")
    root = H.examples_copy(tmp)
    v = root / "vol"
    v.mkdir()
    for t in ("DENS", "POT", "SPIN"):
        shutil.copyfile(CORPUS / f"{MOL}_{t}.CUBE", v / f"mol_{t}.CUBE")
    shutil.copyfile(CORPUS / f"{SLAB}_DENS.CUBE", v / "slab_DENS.CUBE")
    # the same density with one round-off negative, as CRYSTAL writes for many molecules (-1.1e-15)
    lines = (v / "mol_DENS.CUBE").read_text().splitlines(True)
    first = 6 + abs(int(lines[2].split()[0]))
    row = lines[first].split()
    lines[first] = " " + " ".join(["-1.11022E-15"] + row[1:]) + "\n"
    (v / "noisy_DENS.CUBE").write_text("".join(lines))
    a = U.start_app(tmp, root=root, MACE_GUI_MAX_WORKERS="2")
    try:
        yield a
    finally:
        U.stop_app(a)
        shutil.rmtree(v, ignore_errors=True)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    app[0].execute_script("window.__mace.settings.set('volume.quality', 'low')")
    return app


def click(drv, el):
    ActionChains(drv).move_to_element(el).click().perform()


def click_file(drv, path):
    """A real click on the explorer row (found again when the list re-renders under the pointer)."""
    from selenium.common.exceptions import StaleElementReferenceException
    drv.execute_script("window.MACE.state.filter = 'all'; window.MACE.state.text = ''; window.MACE.renderFiles()")
    for attempt in range(6):
        try:
            el = drv.find_element(By.CSS_SELECTOR, f'#file-list .file[data-path="{path}"]')
            drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
            click(drv, el)
            return
        except StaleElementReferenceException:
            time.sleep(0.2)
    raise AssertionError(f"could not click {path}")


def no_errors(drv):
    errs = [e for e in H.errors(drv) if "ResizeObserver loop" not in e]
    assert not errs, errs


def vstate(drv):
    return drv.execute_script("return window.__mace.volumeView && window.__mace.volumeView.state()")


def open_volume(drv, path):
    click_file(drv, path)
    H.wait(drv, "const s = window.__mace.volumeView && window.__mace.volumeView.state(); return s && s.path === arguments[0] && s.surfaces > 0", 90, path)
    time.sleep(0.3)


def set_range(drv, el_id, value):
    el = drv.find_element(By.ID, el_id)
    drv.execute_script("arguments[0].value = arguments[1]; arguments[0].dispatchEvent(new Event('input', {bubbles: true}))", el, str(value))


def test_density_isosurface_and_level(fresh):
    drv = fresh[0]
    open_volume(drv, "vol/mol_DENS.CUBE")
    st = vstate(drv)
    assert st["type"] == "density" and st["surfaces"] == 1 and st["vertices"] > 100
    assert st["stats"]["integral"] > 0
    chip = drv.execute_script("return document.querySelector('#volume-host .vv-chip').textContent")
    assert chip.startswith("Density · ") and "shown (Low)" in chip
    U.shot(drv, "volume_density")
    v0 = st["vertices"]
    lvl = drv.find_element(By.ID, "vo-level")
    lo = float(lvl.get_attribute("min"))
    set_range(drv, "vo-level", lo + 0.2 * (float(lvl.get_attribute("max")) - lo))
    H.wait(drv, "return window.__mace.volumeView.state().vertices !== arguments[0]", 10, v0)
    no_errors(drv)


def test_spin_shows_both_lobes(fresh):
    drv = fresh[0]
    open_volume(drv, "vol/mol_SPIN.CUBE")
    st = vstate(drv)
    assert st["type"] == "spin" and st["plus"] and st["minus"] and st["surfaces"] == 2
    box = drv.find_element(By.ID, "vo-minus")
    click(drv, box)
    H.wait(drv, "return window.__mace.volumeView.state().surfaces === 1", 5)
    click(drv, drv.find_element(By.ID, "vo-minus"))
    H.wait(drv, "return window.__mace.volumeView.state().surfaces === 2", 5)
    no_errors(drv)


def test_slab_surface_sits_on_its_atoms(fresh):
    """The hexagonal slab: with the atoms hidden, the pixels where three carbon atoms project
    are covered by the density surface (a sheared voxel matrix would put it beside them)."""
    drv = fresh[0]
    open_volume(drv, "vol/slab_DENS.CUBE")
    click(drv, drv.find_element(By.ID, "vo-atoms"))
    H.wait(drv, "return !document.querySelector('#vo-atoms').checked", 5)
    time.sleep(0.5)
    hits = drv.execute_async_script("""
        const done = arguments[0], v = window.__mace.volumeView.viewer(), m = window.__mace.volumeView.state();
        const meta = window.__mace.activeDoc().meta, atoms = meta.atoms;
        const carbons = atoms.numbers.map((z, i) => [z, i]).filter(([z]) => z === 6).map(([, i]) => i);
        const pick = [carbons[0], carbons[Math.floor(carbons.length / 2)], carbons[carbons.length - 1]];
        v.render();
        const img = new Image();
        img.onload = () => {
          const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
          const g = c.getContext('2d'); g.drawImage(img, 0, 0);
          const sx = img.width / v.container.offsetWidth, sy = img.height / v.container.offsetHeight;
          const box = v.container.getBoundingClientRect();     // modelToScreen answers in page coordinates
          const bg = g.getImageData(2, 2, 1, 1).data;
          const res = pick.map((i) => {
            const p = v.modelToScreen({x: atoms.cart[i][0], y: atoms.cart[i][1], z: atoms.cart[i][2]});
            const px = g.getImageData(Math.round((p.x - box.left) * sx), Math.round((p.y - box.top) * sy), 1, 1).data;
            return Math.abs(px[0] - bg[0]) + Math.abs(px[1] - bg[1]) + Math.abs(px[2] - bg[2]);
          });
          done(res);
        };
        img.src = v.pngURI();""")
    U.shot(drv, "volume_slab")
    assert all(h > 30 for h in hits), hits
    click(drv, drv.find_element(By.ID, "vo-atoms"))
    no_errors(drv)


def test_mep_colors_range_and_refusal(fresh):
    pytest.importorskip("scipy")
    drv = fresh[0]
    open_volume(drv, "vol/mol_DENS.CUBE")
    click(drv, drv.find_element(By.CSS_SELECTOR, "#volume-host [data-act=mep]"))
    sel = H.wait(drv, "return document.querySelector('#cube-pick')", 10)
    assert sel.get_attribute("value") == "vol/mol_POT.CUBE"          # the same stem's potential, preselected
    click(drv, drv.find_element(By.ID, "cube-ok"))
    H.wait(drv, "const s = window.__mace.volumeView.state(); return s && s.color === 'vol/mol_POT.CUBE' && s.vmin !== null", 90)
    st = vstate(drv)
    assert st["vmin"] == -st["vmax"] and st["vmax"] > 0
    legend = drv.execute_script("return document.querySelector('#volume-host .vv-legend').textContent")
    assert "red = positive (electron-poor)" in legend
    # the CLI's colors: +range red, -range blue
    hexes = drv.execute_script("const v = window.__mace.volumeView, s = v.state(); return [v.gradientHex(s.vmax), v.gradientHex(s.vmin)]")
    r_hi, b_hi = (hexes[0] >> 16) & 255, hexes[0] & 255
    r_lo, b_lo = (hexes[1] >> 16) & 255, hexes[1] & 255
    assert r_hi > b_hi and b_lo > r_lo, hexes
    U.shot(drv, "volume_mep")
    # the GUI's range from 3Dmol's own surface equals the CLI's rule on the CLI's surface (2 %)
    pytest.importorskip("skimage", reason="the CLI side needs scikit-image")
    import contextlib
    import io
    from mace.gui import volume
    from mace.plotting.engines import crystal_cubeviz_plotly as E
    root = fresh[2]
    level = drv.execute_script("return window.__mace.volumeView.state().level")
    with contextlib.redirect_stdout(io.StringIO()):
        fig = E.plot_dual_cube_isosurface(volume.read(str(root / "vol/mol_DENS.CUBE")), volume.read(str(root / "vol/mol_POT.CUBE")), level)
    mesh = next(t for t in fig.data if t.type == "mesh3d")
    assert abs(st["vmax"] - mesh.cmax) <= 0.02 * abs(mesh.cmax), (st["vmax"], mesh.cmax)
    # Reverse Colors swaps them
    click(drv, drv.find_element(By.ID, "vo-reverse"))
    H.wait(drv, "return window.__mace.volumeView.state().reverse === true", 5)
    hx = drv.execute_script("const v = window.__mace.volumeView, s = v.state(); return v.gradientHex(s.vmax)")
    assert (hx & 255) > ((hx >> 16) & 255)
    # a custom range
    click(drv, drv.find_element(By.CSS_SELECTOR, "[data-vo-range=custom]"))
    inp = H.wait(drv, "const e = document.querySelector('#vo-vmax'); return e && !e.disabled && e", 5)
    drv.execute_script("arguments[0].value = '0.05'; arguments[0].dispatchEvent(new Event('change', {bubbles: true}))", inp)
    H.wait(drv, "return window.__mace.volumeView.state().vmax === 0.05", 5)
    # a cube of another cell is refused, with the reason
    click(drv, drv.find_element(By.CSS_SELECTOR, "#volume-host [data-act=mep]"))
    sel = H.wait(drv, "return document.querySelector('#cube-pick')", 10)
    drv.execute_script("arguments[0].value = 'vol/slab_DENS.CUBE'", sel)
    click(drv, drv.find_element(By.ID, "cube-ok"))
    H.wait(drv, "return [...document.querySelectorAll('#toasts .toast2.error .msg')].some((m) => m.textContent.includes('different grids or cells'))", 60)
    drv.execute_script("document.querySelectorAll('#toasts > *').forEach((t) => t.remove())")
    no_errors(drv)


def test_slice_map_and_arrow_keys(fresh):
    pytest.importorskip("plotly")
    drv = fresh[0]
    open_volume(drv, "vol/slab_DENS.CUBE")
    click(drv, drv.find_element(By.ID, "vo-slice"))
    H.wait(drv, "const s = window.__mace.volumeView.state(); return s.slice && s.slice.axis === 'c'", 60)
    H.wait(drv, "const p = document.querySelector('#volume-host .vv-slice-plot'); return p && p.data && p.data.length === 1", 30)
    k0 = drv.execute_script("return window.__mace.volumeView.state().slice.index")
    head = drv.execute_script("return document.querySelector('#volume-host .vv-slice-head').textContent")
    assert head.startswith(f"Plane c = {k0} of 646")
    U.shot(drv, "volume_slice")
    view = drv.find_element(By.ID, "viewer-volume")
    click(drv, view)
    assert drv.execute_script("return document.activeElement.id") == "viewer-volume"
    from selenium.webdriver.common.keys import Keys
    ActionChains(drv).send_keys(Keys.ARROW_UP).perform()
    H.wait(drv, "return window.__mace.volumeView.state().slice.index === arguments[0]", 30, k0 + 1)
    no_errors(drv)


def test_round_off_negatives_leave_a_density_unsigned_and_framed(fresh):
    """A density whose grid holds a -1e-15 is still a density: no negative lobe to show, and the
    first view frames the molecule (an empty lobe once made the fit include the origin, 50 A away,
    and the molecule was drawn as a dot)."""
    drv = fresh[0]
    open_volume(drv, "vol/noisy_DENS.CUBE")
    st = vstate(drv)
    assert st["stats"]["min"] < 0 and st["surfaces"] == 1
    assert not drv.find_elements(By.ID, "vo-minus")
    frac = drv.execute_script("""const v = window.__mace.volumeView.viewer(), c = document.querySelector('#viewer-volume');
      const pts = window.__mace.doc().meta.atoms.cart.map((p) => v.modelToScreen({x: p[0], y: p[1], z: p[2]}));
      const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
      return Math.max((Math.max(...xs) - Math.min(...xs)) / c.clientWidth, (Math.max(...ys) - Math.min(...ys)) / c.clientHeight)""")
    assert frac > 0.3, frac
    head = drv.execute_script("const h = document.querySelector('#volume-host .vv-head'); return [h.scrollWidth, h.clientWidth]")
    assert head[0] <= head[1] + 1, head
    no_errors(drv)
