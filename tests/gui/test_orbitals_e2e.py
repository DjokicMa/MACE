"""The orbital picker (CRYSTAL23 ORBITALS, spec 4.4) in a headless browser:
real clicks and keys on the repo's own ORBITALS fixtures (tests/data/orbitals,
real CRYSTAL23 output), copied into the served demo folder.

- H2O opens at its HOMO as Re ψ with both lobes; the level diagram marks the
  HOMO and LUMO and shows closed-shell occupations as "2 e"; |ψ|² is one
  surface; the arrow keys move along the levels and Enter shows one;
- Si at Γ: the valence triplet (MO 12-14) is flagged as a degenerate set and
  "Sum the set" draws its summed |ψ|² (never a signed sum);
- Si at X and K: the complex K point of a 3D crystal is listed but cannot be
  picked, and says why;
- polyyne: the complex k point of a polymer can be picked, and Im ψ with it;
- Save Cube writes the cube in the job's _orbitals folder and never touches
  the .molden files.

With MACE_GUI_SHOTS=<dir> each case saves a screenshot there.
"""
import hashlib
import shutil
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("numpy")
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.by import By  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "orbitals"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("orbitals_e2e")
    root = H.examples_copy(tmp)
    shutil.copytree(DATA, root / "orb")
    a = U.start_app(tmp, root=root, MACE_GUI_MAX_WORKERS="2")
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    app[0].execute_script("window.__mace.settings.set('volume.quality', 'low')")
    return app


def click(drv, el):
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
    ActionChains(drv).move_to_element(el).click().perform()


def vstate(drv):
    return drv.execute_script("return window.__mace.volumeView && window.__mace.volumeView.state()")


def wait_orbital(drv, path, cond="true", timeout=90):
    """Until the volume view shows an orbital of ``path`` with a surface and ``cond`` (JS on s) holds."""
    try:
        H.wait(drv, "const s = window.__mace.volumeView && window.__mace.volumeView.state();"
                    "return !!(s && s.path === arguments[0] && s.orbital && !s.orbital.busy && s.surfaces > 0 && (" + cond + "))",
               timeout, path)
    except AssertionError:
        seen = drv.execute_script("const m = document.querySelector('#volume-host .vv-msg'), o = document.querySelector('#volume-host .orb-status');"
                                  "return {state: window.__mace.volumeView && window.__mace.volumeView.state(), msg: m && !m.hidden && m.textContent,"
                                  " status: o && o.textContent, toasts: [...document.querySelectorAll('#toasts > *')].map((t) => t.textContent)}")
        raise AssertionError(f"the orbital of {path} never showed ({cond}): {seen}; errors {H.errors(drv)}")
    time.sleep(0.3)
    return vstate(drv)


def open_orbitals(drv, path, cond="true"):
    kind = drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then((x) => d(x && x.kind), (e) => d('error: ' + e.message))", path)
    assert kind == "volume", kind
    return wait_orbital(drv, path, cond)


def no_errors(drv):
    errs = [e for e in H.errors(drv) if "ResizeObserver loop" not in e]
    assert not errs, errs


def snap(drv, name):
    """U.shot with the bottom panel hidden (more room for the view), then put back."""
    import os
    if not os.environ.get("MACE_GUI_SHOTS"):
        return
    shown = drv.execute_script("const b = document.querySelector('#bottom, .bottom-panel'); return !!(b && b.offsetHeight)")
    if shown:
        drv.execute_script("window.__mace.run('view.bottom')")
        time.sleep(0.8)
    drv.execute_script("document.querySelectorAll('#toasts > *').forEach((t) => t.remove())")
    U.shot(drv, name)
    if shown:
        drv.execute_script("window.__mace.run('view.bottom')")
        time.sleep(0.3)


def assert_fits(drv, low=0.6):
    """The box fills the view: the picture's larger screen span is 60-100% of the view (zoomTo alone left a
    2x2x2 Si box at ~15%)."""
    sp = drv.execute_script("return window.__mace.volumeView.drawnSpan()")
    assert sp and low <= max(sp) <= 1.0, sp


def level_row(drv, n):
    return drv.find_element(By.CSS_SELECTOR, f"#volume-host .orb-lv[data-n='{n}']")


def test_h2o_homo_lumo_views_and_keys(fresh):
    drv = fresh[0]
    path = "orb/h2o/h2o_c.molden"
    st = open_orbitals(drv, path)
    o = st["orbital"]
    assert o["mos"] == [5] and o["part"] == "re" and st["signed"] and st["type"] == "orbital"
    assert st["surfaces"] == 2 and st["vertices"] > 100 and st["atoms"] == 3
    assert_fits(drv, low=0.5)        # a molecule: its lobes and atoms, inside a box of atoms +- 5 bohr
    # the default level is the reader's 90 % isovalue
    assert abs(st["level"] - o["isovalue"]) < 1e-12 and o["isovalue"] > 0
    # the diagram: HOMO on MO 5, LUMO on MO 6, closed-shell occupations as "2 e", empty levels as "empty"
    homo, lumo = level_row(drv, 5), level_row(drv, 6)
    assert "HOMO" in homo.text and "LUMO" in lumo.text
    assert "2 e (closed shell)" in homo.text and "empty" in lumo.text
    assert homo.get_attribute("aria-selected") == "true" and "HOMO" in homo.get_attribute("aria-label")
    # the Im view is disabled for a molecule (real orbitals)
    im = drv.find_element(By.CSS_SELECTOR, "#volume-host [data-part='im']")
    assert im.get_attribute("aria-disabled") == "true"
    snap(drv, "orbitals_h2o_homo_re")
    # |psi|^2: one surface, not signed
    click(drv, drv.find_element(By.CSS_SELECTOR, "#volume-host [data-part='abs2']"))
    st = wait_orbital(drv, path, "s.orbital.part === 'abs2'")
    assert not st["signed"] and st["surfaces"] == 1
    click(drv, drv.find_element(By.CSS_SELECTOR, "#volume-host [data-part='re']"))
    wait_orbital(drv, path, "s.orbital.part === 're'")
    # keys: focus the HOMO row, Down moves to MO 4 (lower energy) without drawing it, Enter draws it
    homo = level_row(drv, 5)
    drv.execute_script("arguments[0].focus()", homo)
    ActionChains(drv).send_keys(Keys.ARROW_DOWN).perform()
    time.sleep(0.2)
    assert drv.execute_script("return document.activeElement.dataset.n") == "4"
    assert vstate(drv)["orbital"]["mos"] == [5]
    ActionChains(drv).send_keys(Keys.ENTER).perform()
    st = wait_orbital(drv, path, "s.orbital.mos[0] === 4")
    assert drv.execute_script("return document.activeElement.dataset.n") == "4"
    # Up twice, then Space: the LUMO
    ActionChains(drv).send_keys(Keys.ARROW_UP).send_keys(Keys.ARROW_UP).send_keys(Keys.SPACE).perform()
    st = wait_orbital(drv, path, "s.orbital.mos[0] === 6")
    # the focused row is visibly outlined (:focus-visible after keyboard use)
    outline = drv.execute_script("const e = document.activeElement; return getComputedStyle(e).outlineStyle")
    assert outline not in ("none", ""), outline
    no_errors(drv)


@pytest.mark.guismoke
def test_si_gamma_sum_the_degenerate_triplet(fresh):
    drv = fresh[0]
    path = "orb/si_g/si_g_00001_K000_real.molden"
    st = open_orbitals(drv, path)
    o = st["orbital"]
    # the HOMO is the third member of the valence triplet at Gamma; a 3D cell repeats 2x2x2
    assert o["mos"] == [14] and o["part"] == "re" and o["cells"] == [2, 2, 2] and st["periodic"]
    assert o["gauge_note"] and st["atoms"] > 2
    assert_fits(drv)
    for n in (12, 13, 14):
        assert "degenerate set of 3" in level_row(drv, n).get_attribute("aria-label")
    assert "×3" in level_row(drv, 14).text
    box = drv.find_element(By.ID, "orb-sum")
    assert box.is_enabled()
    click(drv, box)
    st = wait_orbital(drv, path, "s.orbital.mos.length === 3")
    o = st["orbital"]
    assert o["mos"] == [12, 13, 14] and o["part"] == "abs2" and o["degenerate_sum"] and not st["signed"]
    assert st["surfaces"] == 1
    # Re and Im are off while the set is summed
    assert drv.find_element(By.CSS_SELECTOR, "#volume-host [data-part='re']").get_attribute("aria-disabled") == "true"
    # the 3D view drew something
    n = drv.execute_script("""const v = window.__mace.volumeView.viewer(); v.render(); const cv = v.getCanvas();
      const c = document.createElement('canvas'); c.width = cv.width; c.height = cv.height; const g = c.getContext('2d');
      g.drawImage(cv, 0, 0); const d = g.getImageData(0, 0, c.width, c.height).data; let k = 0;
      for (let i = 0; i < d.length; i += 64) if (Math.abs(d[i] - d[0]) + Math.abs(d[i + 1] - d[1]) + Math.abs(d[i + 2] - d[2]) > 30) k++;
      return k;""")
    assert n > 50, n
    snap(drv, "orbitals_si_gamma_triplet_sum")
    no_errors(drv)


def test_si_f_complex_k_is_listed_but_disabled(fresh):
    drv = fresh[0]
    path = "orb/si_f/si_f_00003_K200_real.molden"
    st = open_orbitals(drv, path)
    assert st["orbital"]["k_index"] == 3
    ks = drv.find_elements(By.CSS_SELECTOR, "#volume-host .orb-k")
    assert [k.get_attribute("data-k") for k in ks] == ["3", "4"]
    off = ks[1]
    assert off.get_attribute("aria-disabled") == "true"
    assert "complex k point in a 2D/3D system" in off.get_attribute("title")
    why = drv.find_element(By.ID, off.get_attribute("aria-describedby"))
    assert "orthonormality" in why.get_attribute("textContent")
    assert "C" in off.text and "R" in ks[0].text
    click(drv, off)
    time.sleep(0.5)
    st = vstate(drv)
    assert st["orbital"]["k_index"] == 3
    assert "complex k point" in drv.find_element(By.CSS_SELECTOR, "#volume-host .orb-status").text
    snap(drv, "orbitals_si_f_klist_disabled_complex")
    # the disabled K file itself opens at the usable k point, with a note
    drv.execute_script("window.__mace.closeAll()")
    st = open_orbitals(drv, "orb/si_f/si_f_00004_K110_real.molden")
    assert st["orbital"]["k_index"] == 3
    note = drv.find_element(By.CSS_SELECTOR, "#volume-host .orb-note").text
    assert "k point 4" in note and "cannot be drawn" in note
    no_errors(drv)


def test_polyyne_complex_k_im(fresh):
    drv = fresh[0]
    path = "orb/polyyne/py_f_00001_K000_real.molden"
    st = open_orbitals(drv, path)
    assert st["orbital"]["cells"] == [4, 1, 1]
    assert_fits(drv)
    click(drv, drv.find_element(By.CSS_SELECTOR, "#volume-host .orb-k[data-k='2']"))
    st = wait_orbital(drv, path, "s.orbital.k_index === 2")
    # the view picked before (Re) is kept; Im is offered at a complex k
    assert st["orbital"]["part"] == "re" and st["signed"]
    im = drv.find_element(By.CSS_SELECTOR, "#volume-host [data-part='im']")
    assert im.get_attribute("aria-disabled") is None
    click(drv, im)
    st = wait_orbital(drv, path, "s.orbital.part === 'im'")
    assert st["signed"] and st["surfaces"] == 2 and st["orbital"]["kpoint"]["real"] is False
    snap(drv, "orbitals_polyyne_complex_k_im_4cells")
    no_errors(drv)


def test_open_shell_spin_tabs_and_dark_theme(fresh):
    drv = fresh[0]
    path = "orb/o2/o2_c.molden"
    st = open_orbitals(drv, path)
    tabs = drv.find_elements(By.CSS_SELECTOR, "#volume-host .orb-spins [role='tab']")
    assert [t.get_attribute("data-spin") for t in tabs] == ["ALPHA", "BETA"]
    assert st["orbital"]["spin"] == "ALPHA" and st["orbital"]["mos"] == [9]
    assert "1 e" in level_row(drv, 9).text
    click(drv, tabs[1])
    st = wait_orbital(drv, path, "s.orbital.spin === 'BETA'")
    assert st["orbital"]["mos"] == [7]
    drv.execute_script("window.__mace.settings.set('theme', 'dark')")
    time.sleep(0.6)
    snap(drv, "orbitals_o2_beta_dark")
    drv.execute_script("window.__mace.settings.set('theme', 'light')")
    no_errors(drv)


def test_save_cube_writes_the_orbitals_folder_only(fresh):
    drv, root = fresh[0], fresh[2]
    path = "orb/h2o/h2o_c.molden"
    molden = root / path
    before = hashlib.md5(molden.read_bytes()).hexdigest()
    open_orbitals(drv, path)
    r = drv.execute_async_script("const d = arguments[0]; window.__mace.volumeView.saveCube().then(d, (e) => d({error: e.message}))")
    assert "error" not in r, r
    cube = root / r["path"]
    assert r["path"] == "orb/h2o/h2o_c_orbitals/h2o_c_ab_MO5_re.cube" and cube.is_file()
    from mace.orbitals.cube import cube_meta, read_cube
    meta = cube_meta(read_cube(cube)[5])
    assert meta["mo"] == "5" and meta["part"] == "re"
    # saved again: the same file, not written over
    stamp = cube.stat().st_mtime_ns
    r2 = drv.execute_async_script("const d = arguments[0]; window.__mace.volumeView.saveCube().then(d, (e) => d({error: e.message}))")
    assert r2["cached"] and cube.stat().st_mtime_ns == stamp
    assert hashlib.md5(molden.read_bytes()).hexdigest() == before
    no_errors(drv)


def test_slab_fills_the_view_and_refits_when_the_cells_change(fresh):
    drv = fresh[0]
    path = "orb/slab/slab_f_00003_K200_real.molden"
    st = open_orbitals(drv, path)
    assert st["orbital"]["cells"] == [2, 2, 1] and st["periodic"]
    assert_fits(drv)
    cell = drv.find_element(By.CSS_SELECTOR, "#volume-host .orb-cell[data-axis='0']")
    drv.execute_script("arguments[0].value = '4'; arguments[0].dispatchEvent(new Event('change', {bubbles: true}))", cell)
    wait_orbital(drv, path, "s.orbital.cells[0] === 4")
    assert_fits(drv)
    snap(drv, "orbitals_slab_k200_4x2")
    no_errors(drv)


def test_the_shown_level_is_in_view_on_a_laptop_screen(fresh):
    """At 1280x720 the picker opens with the shown level (the HOMO) in view: its pane scrolls to it (the
    K points, View, Sum, Cells and Quality blocks above the levels filled the pane, so no level showed)."""
    drv = fresh[0]
    U.reset(drv, 1280, 720)
    try:
        path = "orb/si_g/si_g_00001_K000_real.molden"
        open_orbitals(drv, path)
        H.wait(drv, """const p = document.querySelector('#volume-host .vv-orb'), c = p && p.querySelector('.orb-lv[aria-selected="true"]');
          if (!c) return false; const a = c.getBoundingClientRect(), b = p.getBoundingClientRect();
          return a.top >= b.top - 1 && a.bottom <= b.bottom + 1;""", 10)
        no_errors(drv)
    finally:
        U.reset(drv)
