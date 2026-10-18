"""Two editor groups side by side (plan E11) in headless Firefox, real clicks on
a fresh profile, generic CRYSTAL23 fixtures (tests/fixtures/gui_parity) and,
where the corpus is present, a real density/potential pair:

- E11-a: an IR spectrum and an optimization open; View › Editor Groups ›
  Split Editor Right: the spectrum stays left, the structure goes right, both
  draw; a click on a peak opens that mode in the right group;
- E11-b: a tab dragged onto the right half of the editor splits it;
- E11-c: Link Cameras: turning one view turns the other (quaternions equal);
- E11-d: Color Density by Potential and Volume Difference take the other
  group's volume, without asking;
- the layout (open, ratio, which documents) comes back after a reload, and
  Close Group, Focus Other Group and Move Document to Other Group work from
  the keyboard; every document kind opens in the right group."""
import shutil
import time

import pytest

import guiharness as H
import p4util as U
from test_parity_fixtures import unpack

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
pytest.importorskip("plotly")
from selenium.common.exceptions import NoSuchWindowException  # noqa: E402
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.by import By  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402

CORPUS = H.REPO_ROOT / "test" / "ECH3POT3"
PAIR = "1LiFSI-1DEC-conf5_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized_charge+potential"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("split_e2e")
    root = H.examples_copy(tmp)
    unpack(root / "parity")
    shutil.copyfile(root / "structures" / "MgO.cif", root / "structures" / "MgO_copy.cif")    # a twin to compare with
    dens, pot = CORPUS / f"{PAIR}_DENS.CUBE", CORPUS / f"{PAIR}_POT.CUBE"
    if dens.is_file() and pot.is_file():            # a real pair under neutral names (read-only corpus)
        (root / "mep").mkdir()
        shutil.copyfile(dens, root / "mep" / "mol_DENS.CUBE")
        shutil.copyfile(pot, root / "mep" / "mol_POT.CUBE")
    a = U.start_app(tmp, width=1500, height=900, root=root)
    try:
        yield a, root
    finally:
        U.stop_app(a)


def _close_split(drv):
    drv.execute_script("const g = window.__maceGroups; if (g && g.isOpen()) g.closeGroup(true)")
    H.wait(drv, "return !window.__maceGroups.isOpen()", 10)


@pytest.fixture
def fresh(app):
    (a, root) = app
    drv = a[0]
    _close_split(drv)
    U.reset(drv, 1500, 900)
    drv.execute_script("window.__macePane.docs().forEach((d) => window.__macePane.close(d.id, true))")
    drv.execute_script("window.__mace.settings.set('reduceMotion', 'off')")
    return drv, root


def click(drv, el):
    ActionChains(drv).move_to_element(el).click().perform()


def click_file(drv, path, button="left"):
    from selenium.common.exceptions import StaleElementReferenceException
    drv.execute_script("window.MACE.state.filter = 'all'; window.MACE.state.text = ''; window.MACE.renderFiles()")
    for _ in range(6):
        try:
            el = drv.find_element(By.CSS_SELECTOR, f'#file-list .file[data-path="{path}"]')
            drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
            if button == "right":
                ActionChains(drv).move_to_element(el).context_click().perform()
            else:
                click(drv, el)
            return
        except StaleElementReferenceException:
            time.sleep(0.2)
    raise AssertionError(f"could not click {path}")


def menu(drv, top, *path):
    """A real click on a menu bar entry, then on each submenu label, then on the command."""
    el = H.wait(drv, "return [...document.querySelectorAll('.menu-top')].find((x) => x.textContent === arguments[0])", 10, top)
    click(drv, el)
    parent = None
    for label in path[:-1]:
        parent = H.wait(drv, "return [...document.querySelectorAll('.menu .menu-item')].find((b) => b.querySelector('.lbl') && "
                             "b.querySelector('.lbl').textContent === arguments[0])", 5, label)
        click(drv, parent)
    el = H.wait(drv, "return document.querySelector(`.menu .menu-item[data-cmd-id='${arguments[0]}']`)", 5, path[-1])
    if parent is None:
        click(drv, el)
        return
    # the click opened the submenu with its first item focused: real arrow keys to the command, then Enter
    # (moving the pointer across the other items would close the submenu, as a hover does)
    k = drv.execute_script("const m = [...document.querySelectorAll('.menu')].pop(); "
                           "return [...m.querySelectorAll('.menu-item')].findIndex((b) => b.dataset.cmdId === arguments[0])", path[-1])
    assert k >= 0
    # the arrows count from the focused item: wait until the submenu's first item has the focus (it lands
    # late under load, right after a reload, and the keys then picked another command)
    H.wait(drv, "const m = [...document.querySelectorAll('.menu')].pop(); return m && document.activeElement === m.querySelector('.menu-item')", 5)
    ac = ActionChains(drv)
    for _ in range(k):
        ac.send_keys(Keys.ARROW_DOWN)
    ac.send_keys(Keys.ENTER).perform()


def side(drv, js):
    """Run js in the right group's page (`p` is its __macePane, `M` its __mace)."""
    return drv.execute_script("const f = document.querySelector('.group-frame'); if (!f) return null; "
                              "const p = f.contentWindow.__macePane, M = f.contentWindow.__mace, W = f.contentWindow; " + js)


def side_docs(drv):
    return side(drv, "return p ? p.docs() : null") or []


def main_docs(drv):
    return drv.execute_script("return window.__macePane.docs()")


def wait_side(drv, js, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        try:
            r = side(drv, js)
        except Exception:
            r = None
        if r:
            return r
        time.sleep(0.2)
    raise AssertionError(f"right group: timed out waiting for {js}")


def nonblank(drv, rect):
    """The screenshot's pixels inside rect are not one flat color (something was drawn)."""
    from io import BytesIO

    from PIL import Image
    img = Image.open(BytesIO(drv.get_screenshot_as_png())).convert("RGB")
    x, y, w, h = (int(rect[k]) for k in ("x", "y", "width", "height"))
    crop = img.crop((x + w // 4, y + h // 4, x + 3 * w // 4, y + 3 * h // 4))
    return len(set(crop.getdata())) > 8


def no_errors(drv):
    errs = [e for e in H.errors(drv) if "ResizeObserver loop" not in e]
    assert not errs, errs
    errs = [e for e in (side(drv, "return W.__maceErrors || []") or []) if "ResizeObserver loop" not in e]
    assert not errs, errs


def test_ir_spectrum_beside_its_structure_and_a_peak_opens_the_mode_there(fresh):
    drv, root = fresh
    # a real click on the file; under load the click can land while the Explorer draws its list again: once more
    for _ in range(3):
        click_file(drv, "parity/water_freq_IRSPEC.DAT")
        try:
            H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.path === 'parity/water_freq_IRSPEC.DAT'", 20)
            break
        except AssertionError:
            continue
    H.wait(drv, "const s = window.__mace.plotView && window.__mace.plotView.state(); return s && s.drawn && s.peaks > 0", 90)
    click_file(drv, "parity/water_opt.out")
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.path === 'parity/water_opt.out'", 60)
    menu(drv, "View", "Editor Groups", "view.splitRight")
    wait_side(drv, "return p && p.docs().some((d) => d.path === 'parity/water_opt.out')")
    assert [d["path"] for d in main_docs(drv)] == ["parity/water_freq_IRSPEC.DAT"]
    assert [d["path"] for d in side_docs(drv)] == ["parity/water_opt.out"]
    # both draw: the plot on the left, the structure's 3D view on the right
    H.wait(drv, "const el = window.__mace.plotView.el(); return el && el.offsetWidth > 100 && !!el.querySelector('.main-svg')", 30)
    frame = drv.find_element(By.ID, "group-side")
    time.sleep(1.5)
    assert nonblank(drv, frame.rect), "the right group's 3D view drew nothing"
    assert drv.execute_script("return document.getElementById('group-side').classList.contains('focused')")
    # a real click on the strongest peak: its mode opens in the right group
    click(drv, drv.find_element(By.CSS_SELECTOR, "#doc-tabs .doc-tab"))
    H.wait(drv, "const el = window.__mace.plotView.el(), fl = el._fullLayout; return fl && Math.abs(fl.width - el.clientWidth) <= 4", 20)
    time.sleep(0.5)
    want, xy = drv.execute_script("""
        const el = window.__mace.plotView.el(), t = el.data.find((x) => x.meta === 'peaks');
        let k = 0; for (let i = 1; i < t.y.length; i++) if (t.y[i] > t.y[k]) k = i;
        const fl = el._fullLayout, r = el.getBoundingClientRect();
        return [t.customdata[k], [r.left + fl._size.l + fl.xaxis.l2p(t.x[k]), r.top + fl._size.t + fl.yaxis.l2p(t.y[k])]];""")
    body = drv.find_element(By.TAG_NAME, "body")
    ActionChains(drv).move_to_element_with_offset(body, 0, 0).perform()
    ActionChains(drv).move_by_offset(int(xy[0]) - drv.execute_script("return innerWidth / 2 | 0"),
                                     int(xy[1]) - drv.execute_script("return innerHeight / 2 | 0")).perform()
    # plotly sends plotly_click only with a hover point, and its hover runs at most once per 50 ms: a click at
    # the instant the pointer arrives could come before any hover (nothing opened); a person rests first
    time.sleep(0.2)
    ActionChains(drv).click().perform()
    wait_side(drv, "const a = p.active(); return a && a.kind === 'freq' && a.path === 'parity/water_freq.out'")
    wait_side(drv, "const v = M.freqView, s = v && v.state(); return s && s.mode === " + str(int(want)))
    side(drv, "M.freqView.play(false)")
    assert drv.execute_script("return window.__mace.activeDoc().kind") == "plot"      # the spectrum stays left
    no_errors(drv)


def test_a_tab_dragged_onto_the_right_half_splits_the_editor(fresh):
    drv, root = fresh
    click_file(drv, "structures/MgO.cif")
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.path === 'structures/MgO.cif'", 60)
    click_file(drv, "structures/Si.cif")
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.path === 'structures/Si.cif'", 60)
    tab = drv.find_element(By.XPATH, "//div[@id='doc-tabs']//div[contains(@class,'doc-tab')][.//span[@class='t' and text()='Si.cif']]")
    views = drv.find_element(By.ID, "views").rect
    target = (views["x"] + views["width"] * 0.8, views["y"] + views["height"] * 0.5)
    t = tab.rect
    ac = ActionChains(drv)
    ac.move_to_element(tab).click_and_hold()
    steps = 8
    dx, dy = target[0] - (t["x"] + t["width"] / 2), target[1] - (t["y"] + t["height"] / 2)
    for _ in range(steps):
        ac.move_by_offset(int(dx / steps), int(dy / steps))
    ac.pause(0.2).release().perform()
    wait_side(drv, "return p && p.docs().some((d) => d.path === 'structures/Si.cif')")
    assert [d["path"] for d in main_docs(drv)] == ["structures/MgO.cif"]
    assert drv.execute_script("return window.__mace.activeDoc().path") == "structures/MgO.cif"
    no_errors(drv)


def test_link_cameras_turns_both_views(fresh):
    drv, root = fresh
    click_file(drv, "structures/MgO.cif")
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.path === 'structures/MgO.cif'", 60)
    drv.execute_script("window.__maceGroups.openToSide('structures/Si.cif')")
    wait_side(drv, "const a = p.active(); return a && a.path === 'structures/Si.cif' && !!p.viewer()")
    click(drv, drv.find_element(By.CSS_SELECTOR, "#viewer-main"))             # the left group gets the focus
    menu(drv, "View", "Editor Groups", "view.linkCameras")
    H.wait(drv, "return window.__maceGroups.linked()", 5)
    q0 = drv.execute_script("return window.MACE.state.viewers['viewer-main'].getView().slice(4)")
    H.drag(drv, H.viewer_canvas(drv), [(150, 150), (210, 150), (270, 160)], button=2)      # a right-drag orbits
    time.sleep(0.5)
    qa = drv.execute_script("return window.MACE.state.viewers['viewer-main'].getView().slice(4)")
    qb = side(drv, "return p.getQ()")
    import math
    dot = abs(sum(a * b for a, b in zip(q0, qa)))
    assert 2 * math.degrees(math.acos(min(1.0, dot))) > 20, (q0, qa)            # the drag turned the left view
    assert all(abs(a - b) < 1e-6 for a, b in zip(qa, qb)), (qa, qb)
    # each keeps its own zoom: the translations differ where the cells differ
    # unlinked: turning the right view leaves the left one alone
    menu(drv, "View", "Editor Groups", "view.linkCameras")
    H.wait(drv, "return !window.__maceGroups.linked()", 5)
    side(drv, "p.setQ([0, 0, 0.3826834, 0.9238795])")
    assert abs(side(drv, "return p.getQ()")[3] - 0.9238795) < 1e-6
    time.sleep(0.4)
    qa2 = drv.execute_script("return window.MACE.state.viewers['viewer-main'].getView().slice(4)")
    assert all(abs(a - b) < 1e-9 for a, b in zip(qa, qa2))
    no_errors(drv)


def test_volume_difference_takes_the_other_groups_volume(fresh):
    drv, root = fresh
    click_file(drv, "parity/mgo_dens/DENS_CUBE.DAT")
    H.wait(drv, "const s = window.__mace.volumeView && window.__mace.volumeView.state(); return s && s.path === 'parity/mgo_dens/DENS_CUBE.DAT'", 90)
    # Open to the Side from the Explorer's menu (real right-click)
    click_file(drv, "parity/mgo_pato/DENS_CUBE.DAT", button="right")
    el = H.wait(drv, "return document.querySelector(\".menu .menu-item[data-cmd-id='explorer.openToSide']\")", 5)
    click(drv, el)
    wait_side(drv, "const a = p.active(); return a && a.kind === 'volume' && a.path === 'parity/mgo_pato/DENS_CUBE.DAT'", 90)
    click(drv, drv.find_element(By.CSS_SELECTOR, "#doc-tabs .doc-tab"))           # back to the left group
    menu(drv, "Analysis", "analysis.volumeDiff")
    H.wait(drv, "return !!document.querySelector('#vdiff-dialog')", 10)
    rows = drv.execute_script("return [...document.querySelectorAll('#vdiff-dialog .vd-path')].map((s) => s.value)")
    assert rows[:2] == ["parity/mgo_dens/DENS_CUBE.DAT", "parity/mgo_pato/DENS_CUBE.DAT"], rows
    click(drv, drv.find_element(By.CSS_SELECTOR, "#vdiff-dialog [data-close]"))
    no_errors(drv)


def test_color_density_by_potential_takes_the_other_group_without_asking(fresh):
    drv, root = fresh
    if not (root / "mep").is_dir():
        pytest.skip("test/ corpus not present (gitignored): no real density/potential pair")
    click_file(drv, "mep/mol_DENS.CUBE")
    H.wait(drv, "const s = window.__mace.volumeView && window.__mace.volumeView.state(); return s && s.path === 'mep/mol_DENS.CUBE'", 120)
    drv.execute_script("window.__maceGroups.openToSide('mep/mol_POT.CUBE')")
    wait_side(drv, "const a = p.active(); return a && a.kind === 'volume' && a.path === 'mep/mol_POT.CUBE'", 120)
    click(drv, drv.find_element(By.CSS_SELECTOR, "#doc-tabs .doc-tab"))
    menu(drv, "Analysis", "analysis.volumeMep")
    H.wait(drv, "const s = window.__mace.volumeView.state(); return s && s.color === 'mep/mol_POT.CUBE'", 120)
    assert not drv.execute_script("return !!document.querySelector('#cube-dialog')")       # no question asked
    no_errors(drv)


def test_layout_comes_back_and_the_keyboard_moves_and_focuses(fresh):
    drv, root = fresh
    click_file(drv, "structures/NaCl.cif")
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.path === 'structures/NaCl.cif'", 60)
    click_file(drv, "outputs/Si_sp.out")
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.path === 'outputs/Si_sp.out'", 60)
    click(drv, drv.find_element(By.CSS_SELECTOR, "#views"))
    U.keys(drv, Keys.CONTROL, "\\")                                               # Split Editor Right
    wait_side(drv, "return p && p.docs().some((d) => d.path === 'outputs/Si_sp.out')")
    # the right group's page has the focus: Mod+Shift+\ there moves its document back
    H.wait(drv, "return window.__maceGroups.state().focused === 'side'", 10)
    drv.switch_to.frame(drv.find_element(By.CSS_SELECTOR, ".group-frame"))
    try:
        U.keys(drv, Keys.CONTROL, Keys.SHIFT, "\\")
    except NoSuchWindowException:
        pass                                     # the group closed under the key's release (its last document left)
    finally:
        drv.switch_to.default_content()
    H.wait(drv, "return window.__macePane.docs().some((d) => d.path === 'outputs/Si_sp.out')", 30)
    # it was the right group's last document: that group closed, the focus came back with the document
    H.wait(drv, "return !window.__maceGroups.isOpen() && !document.querySelector('.group-frame')", 10)
    assert drv.execute_script("return window.__maceGroups.state().focused") == "main"
    # Mod+Alt+\ moves the focus to the other group, and (pressed there) back
    drv.execute_script("window.__maceGroups.openToSide('structures/LiF.cif')")
    wait_side(drv, "return p.docs().some((d) => d.path === 'structures/LiF.cif')")
    click(drv, drv.find_element(By.CSS_SELECTOR, "#doc-tabs .doc-tab"))
    H.wait(drv, "return window.__maceGroups.state().focused === 'main'", 10)
    U.keys(drv, Keys.CONTROL, Keys.ALT, "\\")
    H.wait(drv, "return window.__maceGroups.state().focused === 'side'", 10)
    drv.switch_to.frame(drv.find_element(By.CSS_SELECTOR, ".group-frame"))
    try:
        U.keys(drv, Keys.CONTROL, Keys.ALT, "\\")
    finally:
        drv.switch_to.default_content()
    H.wait(drv, "return window.__maceGroups.state().focused === 'main'", 10)
    # widen the left group, reload: the same split comes back
    sp = drv.find_element(By.ID, "split-groups")
    ActionChains(drv).click_and_hold(sp).move_by_offset(-120, 0).release().perform()
    ratio = drv.execute_script("return window.__maceGroups.state().ratio")
    assert ratio < 0.48
    H.wait(drv, "return !window.__mace.profile || !window.__mace.profile.pending || !window.__mace.profile.pending()", 10)
    time.sleep(1.5)                                                                   # the profile write
    drv.refresh()
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'", 60)
    wait_side(drv, "return p && p.docs().some((d) => d.path === 'structures/LiF.cif')", 60)
    assert abs(drv.execute_script("return window.__maceGroups.state().ratio") - ratio) < 0.01
    # Close Group from the menu: one group again, and that is remembered too
    menu(drv, "View", "Editor Groups", "view.closeGroup")
    H.wait(drv, "return !window.__maceGroups.isOpen() && !document.querySelector('.group-frame')", 10)
    no_errors(drv)


@pytest.mark.parametrize("path, kind", [
    ("parity/water_freq.out", "freq"), ("parity/mgo_doss.DOSS.DAT", "plot"), ("parity/mgo_dens/DENS_CUBE.DAT", "volume"),
    ("parity/mgo_sp.d12", "structure"), ("parity/mgo_doss.d3", "text"), ("materials.db", "db"),
    ("outputs/Ag2Br3_opt_unfinished.out", "trajectory"), ("parity/mgo_echg.f25", "plot"),
])
def test_every_document_kind_opens_in_the_right_group(fresh, path, kind):
    drv, root = fresh
    if not (root / path).exists():
        pytest.skip(f"{path} not in this copy")
    drv.execute_script("window.__maceGroups.openToSide(arguments[0])", path)
    a, end = None, time.time() + 120
    while time.time() < end:
        a = side(drv, "return p && p.active()")
        if a and a["path"] == path:
            break
        time.sleep(0.3)
    assert a and a["path"] == path and a["kind"] == kind, a
    frame = drv.find_element(By.ID, "group-side")
    end, ok = time.time() + 30, False             # a plot or a volume may still be drawing; an animated
    while not ok and time.time() < end:          # 3D view can read blank between two frames
        ok = nonblank(drv, frame.rect)
        if not ok:
            time.sleep(0.5)
    assert ok, (path, side(drv, "return p.active()"))
    no_errors(drv)


GEOMETRY = """const r = (id) => document.getElementById(id).getBoundingClientRect(), f = document.querySelector('.group-frame').getBoundingClientRect();
    const W = document.querySelector('.group-frame').contentWindow, shown = (e) => !!e && W.getComputedStyle(e).display !== 'none' && e.getBoundingClientRect().width > 0;
    return {frameRight: Math.round(f.right), frameBottom: Math.round(f.bottom), sideRight: Math.round(r('group-side').right),
      inspRight: Math.round(r('inspector').right), bottomBottom: Math.round(r('bottom').bottom), bottomRight: Math.round(r('bottom').right),
      sideLeft: Math.round(r('group-side').left), inert: document.getElementById('inspector').inert,
      paneInsp: shown(W.document.getElementById('inspector')), paneBottom: shown(W.document.getElementById('bottom')),
      paneTab: (W.document.querySelector('.bottom-tab.active') || {}).dataset.bottom || null,
      winInsp: [...document.querySelectorAll('#inspector .insp-section')].map((x) => x.dataset.key)};"""


def test_inspector_and_bottom_panel_follow_the_focused_group(fresh):
    """Plan E11 item 1: the Inspector shows the focused group's document (the
    right group's own, over the Inspector's column) and each group has its
    own bottom panel; a real click in the right group's Inspector changes the
    right group's document."""
    drv, root = fresh
    click_file(drv, "parity/mgo_doss.DOSS.DAT")
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.kind === 'plot'", 60)
    click_file(drv, "parity/water_freq.out", button="right")
    click(drv, H.wait(drv, "return document.querySelector(\".menu .menu-item[data-cmd-id='explorer.openToSide']\")", 5))
    wait_side(drv, "const a = p.active(); return a && a.kind === 'freq'", 90)
    H.wait(drv, "return window.__maceGroups.state().focused === 'side'", 10)
    g = H.wait(drv, GEOMETRY.replace("return {", "const g = {") + " return g.frameRight === g.inspRight && g.paneInsp && g;", 20)
    # the right group has the focus: its Inspector (Mode) over the window's, which is inert; its own bottom panel (Modes)
    assert g["inert"] and g["frameBottom"] == g["bottomBottom"], g
    assert g["paneBottom"] and g["paneTab"] == "modes" and g["bottomRight"] <= g["sideLeft"], g
    # the SCF card (1.1.6) under a FREQ output's Mode (then, read-only, the run's Structure and Cell: the panel context)
    assert side(drv, "return [...W.document.querySelectorAll('#inspector .insp-section')].map((x) => x.dataset.key)")[:2] == ["mode", "scf"]
    # the mode table, on screen under the right group
    assert int(wait_side(drv, "const t = W.document.querySelector('.bottom-pane.modes #freq-table .fm-grid');"
                              " return t && t.getBoundingClientRect().height > 40 && t.getAttribute('aria-rowcount')")) > 3
    # a real click on the left group's tab: the window's Inspector (plot options) is back, the right group keeps its bottom panel
    click(drv, drv.find_element(By.CSS_SELECTOR, "#doc-tabs .doc-tab"))
    g = H.wait(drv, GEOMETRY.replace("return {", "const g = {") + " return g.frameRight === g.sideRight && !g.paneInsp && g;", 20)
    assert not g["inert"] and g["winInsp"][:1] == ["plotopts"] and g["paneBottom"], g
    # a volume on the right: a real click on its own Inspector's "Show + lobe" changes that document only
    drv.execute_script("window.__maceGroups.openToSide('parity/mgo_dens/DENS_CUBE.DAT')")
    wait_side(drv, "const s = M.volumeView && M.volumeView.state(); return s && s.path === 'parity/mgo_dens/DENS_CUBE.DAT' && s.surfaces >= 0")
    H.wait(drv, GEOMETRY.replace("return {", "const g = {") + " return g.frameRight === g.inspRight && g.paneInsp && g;", 20)
    before = side(drv, "return M.volumeView.state().plus")
    frame = drv.find_element(By.CSS_SELECTOR, ".group-frame")
    drv.switch_to.frame(frame)
    try:
        box = H.wait(drv, "return document.getElementById('vo-plus')", 10)
        drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", box)
        ActionChains(drv).move_to_element(box).click().perform()
    finally:
        drv.switch_to.default_content()
    wait_side(drv, f"return M.volumeView.state().plus === {'false' if before else 'true'}")
    assert drv.execute_script("return window.__mace.activeDoc().kind") == "plot"
    no_errors(drv)


def test_ui_scale_is_applied_once_in_the_right_group(fresh):
    drv, root = fresh
    drv.execute_script("window.__mace.settings.set('uiScale', 125)")
    try:
        click_file(drv, "structures/MgO.cif")
        drv.execute_script("window.__maceGroups.openToSide('structures/Si.cif')")
        wait_side(drv, "const a = p.active(); return a && a.path === 'structures/Si.cif'")
        time.sleep(0.5)
        left, right = drv.execute_script(
            "const f = document.querySelector('.group-frame'), W = f.contentWindow;"
            "const k = f.getBoundingClientRect().height / W.innerHeight;"     # the window's zoom, as it scales the frame
            "return [document.querySelector('#doc-tabs .doc-tab').getBoundingClientRect().height,"
            " W.document.querySelector('#doc-tabs .doc-tab').getBoundingClientRect().height * k]")
        assert abs(left - right) < 1.0, (left, right)
    finally:
        drv.execute_script("window.__mace.settings.set('uiScale', 100)")
    no_errors(drv)


def test_compare_the_two_groups_structures(fresh):
    drv, root = fresh
    click_file(drv, "structures/MgO.cif")
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.path === 'structures/MgO.cif'", 60)
    drv.execute_script("window.__maceGroups.openToSide('structures/MgO_copy.cif')")
    wait_side(drv, "const a = p.active(); return a && a.path === 'structures/MgO_copy.cif'")
    click(drv, drv.find_element(By.CSS_SELECTOR, "#doc-tabs .doc-tab"))
    menu(drv, "Analysis", "analysis.compareGroups")
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.kind === 'trajectory' && d.title === 'MgO.cif ↔ MgO_copy.cif'", 60)
    no_errors(drv)
