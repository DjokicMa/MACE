"""The panel context (1.1.6) in headless Firefox and Chromium, real clicks:
every document-scoped side panel follows the document in focus in the center.

- a structure, then a DOS plot: the Atoms table, Coordination and the
  Inspector's Structure and Cell show the plot's own run (MgO from
  mgo_doss.out), read-only and labeled "from mgo_doss.out", never the NaCl
  focused before; Timeline, Measurements and History show a neutral
  placeholder and their tabs are dimmed, not hidden;
- back on the structure: its atoms, its selection, its Inspector scroll and
  its collapsed sections are as they were;
- an unrelated view (the plot gallery): placeholders everywhere, the
  Inspector's too;
- a run that is ambiguous (two SCF outputs beside a .d3): the placeholder
  offers both, by keyboard;
- Tasks, Jobs and Problems are global: the same content whatever is focused;
- a split: each editor group's panels follow that group's document.

Repo data only (examples/gui and the generic CRYSTAL23 parity fixtures)."""
import shutil
import time

import pytest

import guiharness as H
import p4util as U
from test_parity_fixtures import unpack

pytestmark = [pytest.mark.gui, pytest.mark.guismoke]
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.by import By  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402

W, HT = 1500, 900
DOS = "parity/mgo_doss.DOSS.DAT"


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("panelctx")
    root = H.examples_copy(tmp)
    unpack(root / "parity")
    amb = root / "amb"                          # a properties deck beside two SCF runs of its material
    amb.mkdir()
    shutil.copyfile(root / "outputs" / "Si_sp.out", amb / "Si_sp.out")
    shutil.copyfile(root / "outputs" / "Si_sp.out", amb / "Si_opt.out")
    shutil.copyfile(root / "parity" / "mgo_doss.d3", amb / "Si_doss.d3")
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1", MACE_GUI_NO_WARMUP="1")
    yield root, base
    proc.terminate()
    proc.wait(timeout=30)


@pytest.fixture(scope="module", params=["firefox", "chromium"])
def browser(request, served):
    root, base = served
    try:
        drv = H.browser(request.param, width=W, height=HT)
    except Exception as e:                       # noqa: BLE001
        if request.param == "chromium":
            pytest.skip(f"no Chromium: {e}")
        raise
    try:
        H.load(drv, base)
        yield drv, request.param
    finally:
        drv.quit()


@pytest.fixture
def fresh(browser):
    drv, name = browser
    drv.execute_script("const g = window.__maceGroups; if (g && g.isOpen()) g.closeGroup(true)")
    H.wait(drv, "return !window.__maceGroups.isOpen()", 10)
    U.reset(drv, W, HT)
    drv.execute_script("window.__macePane.docs().forEach((d) => window.__macePane.close(d.id, true))")
    try:
        yield drv
    except BaseException:
        H.save_failure(drv, f"panelctx_{name}")
        raise


# ---------------------------------------------------------------- helpers
def click(drv, el):
    drv.execute_script("arguments[0].scrollIntoView({block: 'nearest', inline: 'nearest'})", el)
    ActionChains(drv).move_to_element(el).click().perform()


def open_doc(drv, path):
    drv.execute_async_script("const done = arguments[1]; window.__mace.open(arguments[0]).then(() => done(true), () => done(false))", path)
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && (d.path === arguments[0] || d.srcPath === arguments[0])", 60, path)


def tab(drv, key):
    """A real click on a bottom tab."""
    click(drv, drv.find_element(By.CSS_SELECTOR, f'.bottom-tab[data-bottom="{key}"]'))
    H.wait(drv, "return document.querySelector('.bottom-tab.active').dataset.bottom === arguments[0]", 5, key)


def doc_tab(drv, title):
    el = H.wait(drv, "return [...document.querySelectorAll('#doc-tabs .doc-tab')].find((t) => t.querySelector('.t').textContent === arguments[0])", 10, title)
    click(drv, el)
    H.wait(drv, "return window.__mace.activeDoc().title === arguments[0]", 10, title)


STATE = """const vis = (e) => !!e && getComputedStyle(e).display !== 'none' && e.getClientRects().length > 0;
const pane = document.querySelector('.bottom-pane.active'), ph = pane && pane.querySelector(':scope > .pc-ph');
return {tab: pane && pane.dataset.pane, ph: vis(ph) ? ph.textContent : null, phRole: ph && ph.getAttribute('role'), phLive: ph && ph.getAttribute('aria-live'),
  phTab: ph ? ph.tabIndex : null, text: pane ? pane.innerText : '', label: (document.querySelector('#bottom-tabs .pc-doc') || {}).textContent || '',
  rows: [...document.querySelectorAll('.bottom-pane.atoms tbody tr[data-i]')].map((r) => r.children[1].textContent),
  sel: [...document.querySelectorAll('.bottom-pane.atoms tbody tr[data-i].sel')].map((r) => +r.dataset.i),
  from: (document.querySelector('.bottom-pane.atoms .pc-from') || {}).textContent || '',
  off: [...document.querySelectorAll('.bottom-tab.pc-off')].map((b) => b.dataset.bottom),
  hidden: [...document.querySelectorAll('.bottom-tab')].filter((b) => b.hidden || !vis(b)).map((b) => b.dataset.bottom),
  insp: [...document.querySelectorAll('#inspector .insp-section')].map((s) => s.dataset.key),
  inspText: document.getElementById('inspector').innerText, inspLabel: (document.querySelector('#inspector .pc-doc') || {}).textContent || '',
  inspPh: !!document.querySelector('#inspector .pc-ph') && vis(document.querySelector('#inspector .pc-ph')),
  ctx: window.__mace.panelContext()};"""


def state(drv):
    return drv.execute_script(STATE)


def wait_state(drv, cond, timeout=30):
    end, s = time.time() + timeout, None
    while time.time() < end:
        s = state(drv)
        if cond(s):
            return s
        time.sleep(0.2)
    raise AssertionError(f"panel state: {s}")


# ---------------------------------------------------------------- tests
def test_a_plot_after_a_structure_shows_its_own_runs_atoms_and_placeholders(fresh):
    drv = fresh
    open_doc(drv, "structures/NaCl.cif")
    tab(drv, "atoms")
    s = wait_state(drv, lambda s: len(s["rows"]) == 8)
    assert set(s["rows"]) == {"Na", "Cl"} and s["label"] == "Atoms · NaCl.cif" and s["inspLabel"] == "· NaCl.cif", s
    # its state: two atoms selected, the Symmetry section shut, the Inspector scrolled
    drv.execute_script("window.__mace.editor.select([1, 3], 'replace')")
    time.sleep(0.3)
    sym = drv.find_element(By.CSS_SELECTOR, '#inspector .insp-section[data-key="symmetry"] .insp-head')
    if drv.execute_script("return arguments[0].closest('.insp-section').classList.contains('collapsed')", sym):
        sym.click()                              # (the layout is the server's: an earlier browser shut it)
        H.wait(drv, "return !document.querySelector('#inspector .insp-section[data-key=\"symmetry\"]').classList.contains('collapsed')", 5)
    sym.click()
    H.wait(drv, "return document.querySelector('#inspector .insp-section[data-key=\"symmetry\"]').classList.contains('collapsed')", 5)
    time.sleep(0.6)
    drv.execute_script("const i = document.getElementById('inspector'); i.scrollTop = Math.min(120, i.scrollHeight - i.clientHeight)")
    scrolled = drv.execute_script("return document.getElementById('inspector').scrollTop")
    time.sleep(0.4)

    open_doc(drv, DOS)
    # the bottom panel stays on Atoms (the plot borrows its run's atoms; it used to jump to Tasks)
    H.wait(drv, "return window.__mace.activeDoc() && window.__mace.activeDoc().kind === 'plot'", 30)
    time.sleep(0.3)
    assert drv.execute_script("const t = document.querySelector('.bottom-tab.active'); return t && t.dataset.bottom") == "atoms"
    # Atoms: MgO from the DOS run, read-only and labeled; never NaCl
    tab(drv, "atoms")
    s = wait_state(drv, lambda s: s["rows"] == ["Mg", "O"], 60)
    assert "From mgo_doss.out" in s["from"] and "read-only" in s["from"] and "Projections: Mg (s, p, d, all) on atom 1" in s["from"], s
    assert s["label"] == "Atoms · from mgo_doss.out (a DOS plot mgo_doss.DOSS.DAT)", s
    assert "Na" not in s["rows"] and not s["ph"], s
    assert s["ctx"]["borrowed"] == "parity/mgo_doss.out" and s["ctx"]["related"] == "found", s["ctx"]
    # a borrowed structure refuses edits (no parent edited from a plot)
    assert drv.execute_script("const f = window.__mace.activeDoc() && document.querySelector('.bottom-pane.atoms .pc-from'); return !!f") and \
        "read-only" in s["from"]
    # the Inspector: its plot options, then the run's Structure and Cell, read-only (no Nudge, no Edit Cell…)
    s = wait_state(drv, lambda s: s["insp"] == ["plotopts", "structure", "cell"])
    assert "MgO" in s["inspText"] and "from mgo_doss.out" in s["inspText"] and "Edit Cell" not in s["inspText"], s["inspText"]
    assert "NaCl" not in s["inspText"] and s["inspLabel"] == "· mgo_doss.DOSS.DAT", s
    # Timeline, Measurements, History: a neutral placeholder; their tabs dimmed, every tab still there
    assert {"timeline", "measurements", "history", "modes"} <= set(s["off"]) and "atoms" not in s["off"] and not s["hidden"], s
    for key, head in (("timeline", "No timeline: mgo_doss.DOSS.DAT is a DOS plot."),
                      ("measurements", "No measurements: mgo_doss.DOSS.DAT is a DOS plot."),
                      ("history", "No edit history: mgo_doss.DOSS.DAT is a DOS plot.")):
        tab(drv, key)
        s = state(drv)
        assert s["ph"] and s["ph"].startswith(head), (key, s)
        assert s["phRole"] == "status" and s["phLive"] == "polite" and s["phTab"] == 0, s
        assert "Na1" not in s["text"] and "Ag2Br3" not in s["text"] and "NaCl" not in s["text"], (key, s["text"])
        assert s["label"].endswith("· mgo_doss.DOSS.DAT"), s
    # Coordination: the run's sites
    tab(drv, "coordination")
    s = wait_state(drv, lambda s: "Mg1" in s["text"], 30)
    assert "Na1" not in s["text"], s["text"]

    # back on NaCl: its atoms, selection, collapsed section and Inspector scroll
    doc_tab(drv, "NaCl.cif")
    tab(drv, "atoms")
    s = wait_state(drv, lambda s: len(s["rows"]) == 8)
    assert s["sel"] == [1, 3] and not s["ph"] and s["label"] == "Atoms · NaCl.cif", s
    assert "timeline" in s["off"] and "measurements" not in s["off"], s
    assert drv.execute_script("return document.querySelector('#inspector .insp-section[data-key=\"symmetry\"]').classList.contains('collapsed')")
    H.wait(drv, f"return Math.abs(document.getElementById('inspector').scrollTop - {scrolled}) <= 2", 5)
    U.no_errors(drv)


def test_an_unrelated_view_leaves_nothing_behind(fresh):
    drv = fresh
    open_doc(drv, "structures/NaCl.cif")
    drv.execute_async_script("const d = arguments[0]; window.__mace.openGallery({}).then(() => d(1), () => d(0))")
    H.wait(drv, "return window.__mace.activeDoc().kind === 'gallery'", 30)
    for key, head in (("atoms", "No atoms:"), ("coordination", "No coordination table:"), ("timeline", "No timeline:")):
        tab(drv, key)
        s = wait_state(drv, lambda s: bool(s["ph"]))
        assert s["ph"].startswith(head) and "is the plot gallery." in s["ph"], s
        assert "Na" not in s["rows"] and "Na1" not in s["text"], s
    # the Inspector: its own placeholder, said politely, reachable by keyboard
    assert s["inspPh"] and "No properties:" in s["inspText"] and "NaCl" not in s["inspText"], s
    assert drv.execute_script("const p = document.querySelector('#inspector .pc-ph'); p.focus(); return document.activeElement === p")
    U.no_errors(drv)


def test_an_ambiguous_run_is_picked_from_the_placeholder(fresh):
    drv = fresh
    open_doc(drv, "amb/Si_doss.d3")
    tab(drv, "atoms")
    s = wait_state(drv, lambda s: bool(s["ph"]) and "could come from 2 runs" in s["ph"], 30)
    assert s["ctx"]["related"] == "ambiguous" and sorted(s["ctx"]["candidates"]) == ["amb/Si_opt.out", "amb/Si_sp.out"], s
    # by keyboard: Tab into the placeholder's choices, Enter on one
    btn = drv.find_element(By.XPATH, "//div[contains(@class,'pc-ph')]//button[normalize-space()='Show Si_opt.out']")
    drv.execute_script("arguments[0].focus()", btn)
    ActionChains(drv).send_keys(Keys.ENTER).perform()
    s = wait_state(drv, lambda s: s["rows"] == ["Si", "Si"], 30)
    assert "From Si_opt.out" in s["from"] and "(picked)" in s["from"], s
    U.no_errors(drv)


def test_global_panels_stay_as_they_are(fresh):
    drv = fresh
    open_doc(drv, "structures/NaCl.cif")
    seen = {}
    for key, note in (("tasks", "Tasks · all documents"), ("jobs", "Jobs · this folder"), ("problems", "Problems · all documents")):
        tab(drv, key)
        s = wait_state(drv, lambda s: s["text"].strip(), 30)          # (Jobs asks the server first)
        assert s["label"] == note and not s["ph"] and key not in s["off"], s
        seen[key] = s["text"]
    open_doc(drv, DOS)
    wait_state(drv, lambda s: s["ctx"]["related"] == "found", 60)
    for key in ("tasks", "jobs", "problems"):
        tab(drv, key)
        s = state(drv)
        assert not s["ph"] and key not in s["off"], s
        if key != "tasks":                       # Tasks gains the plot's own read; Jobs and Problems are the folder's
            assert s["text"] == seen[key], (key, s["text"], seen[key])
    U.no_errors(drv)


def side(drv, js):
    return drv.execute_script("const f = document.querySelector('.group-frame'); if (!f) return null; "
                              "const W = f.contentWindow, M = W.__mace, D = W.document; " + js)


def wait_side(drv, js, timeout=90):
    end, r = time.time() + timeout, None
    while time.time() < end:
        try:
            r = side(drv, js)
        except Exception:                        # noqa: BLE001 (the frame is loading)
            r = None
        if r:
            return r
        time.sleep(0.2)
    raise AssertionError(f"right group: timed out waiting for {js}")


def test_each_editor_group_follows_its_own_document(fresh):
    drv = fresh
    open_doc(drv, "structures/NaCl.cif")
    drv.execute_async_script("const done = arguments[1]; window.__maceGroups.openToSide(arguments[0]).then(() => done(true), () => done(false))", DOS)
    c = wait_side(drv, "const c = M && M.panelContext(); return c && c.related === 'found' && c", 90)
    # the right group (focused): the DOS plot's run, MgO; its Inspector names it
    assert c["doc"] == "mgo_doss.DOSS.DAT" and c["borrowed"] == "parity/mgo_doss.out", c
    H.wait(drv, "return window.__maceGroups.state().focused === 'side'", 10)
    lab = wait_side(drv, "const l = D.querySelector('#inspector .pc-doc'); return l && l.textContent.includes('mgo_doss') && l.textContent", 20)
    assert lab == "· mgo_doss.DOSS.DAT"
    assert wait_side(drv, "const r = [...D.querySelectorAll('.bottom-pane.atoms tbody tr[data-i]')].map((r) => r.children[1].textContent);"
                          " return r.length === 2 && r.join(',')", 30) == "Mg,O"
    # the left group keeps NaCl in its own panels
    main = drv.execute_script("return window.__mace.panelContext()")
    assert main["doc"] == "NaCl.cif" and main["own"] and main["panels"]["atoms"]["header"] == "Atoms · NaCl.cif", main
    # focus the left group: the window's Inspector is NaCl's
    doc_tab(drv, "NaCl.cif")
    H.wait(drv, "return window.__maceGroups.state().focused === 'main'", 10)
    s = state(drv)
    assert s["inspLabel"] == "· NaCl.cif" and "MgO" not in s["inspText"], s
    U.no_errors(drv)
