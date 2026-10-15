"""The toolbar's display options work in the mode view (a FREQ document)
(1.1.5 report: "many top-bar options are grayed out: the displayed
supercell, show/hide unit cell, bonds, perspective/orthographic, and so on").

Each option is enabled there, does what it does in the structure view (read
back from what 3Dmol draws), and a supercell moves every periodic image of an
atom as the atom moves (the modes are at Γ). Options that cannot apply say why
(a tooltip reason, not just gray). The comparison view of a run follows Show
Unit Cell and Show Bonds too. Chromium and Firefox; corpus files copied in at
run time (skipped without the test/ corpus).
"""
import shutil
import time

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
np = pytest.importorskip("numpy")

CORPUS = H.REPO_ROOT / "test" / "FREQ"
DIA = "2_dia2_opt_rev1_freq_B3LYP-D3-D3_optimized.out"
TOOLBAR = ["view.cell", "view.bonds", "view.labelsCycle", "view.boundary", "view.projection"]
MENU = ["view.alongA", "view.alongB", "view.alongC", "view.alongAStar", "view.reset", "view.range1", "view.range2", "view.range3",
        "view.rangeCustom", "view.rangeCycle", "view.labels.element", "view.rotateLeft", "view.rotateInPlanePlus"]
FV = "window.MACE.state.viewers['viewer-freq']"


@pytest.fixture(scope="module", params=["chromium", "firefox"])
def app(request, tmp_path_factory):
    if not (CORPUS / DIA).is_file():
        pytest.skip("test/ corpus not present (gitignored)")
    tmp = tmp_path_factory.mktemp("freqbar")
    root = H.examples_copy(tmp)
    (root / "freq").mkdir()
    shutil.copy2(CORPUS / DIA, root / "freq" / DIA)
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1")
    try:
        drv = H.browser(request.param, width=1280, height=800)
    except BaseException:
        proc.terminate()
        proc.wait(timeout=15)
        raise
    try:
        H.load(drv, base)
        drv.execute_script("window.__mace.settings.set('reduceMotion', 'off')")
        yield drv
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


@pytest.fixture
def drv(app, request):
    failed = request.session.testsfailed
    yield app
    if request.session.testsfailed > failed:
        H.save_failure(app, request.node.name)
    app.execute_script("""window.__mace.closeAll(); const s = window.__mace.settings;
      s.set('showCell', true); s.set('labels', 'off'); s.set('projection', 'orthographic');
      const b = document.getElementById('opt-bonds'); if (b && !b.checked) window.__mace.run('view.bonds');""")
    time.sleep(0.3)


def open_modes(drv):
    drv.execute_async_script("const done = arguments[1]; window.__mace.openModes(arguments[0]).then(() => done(1))", f"freq/{DIA}")
    H.wait(drv, "const v = window.__mace.freqView, s = v && v.state(); return s && s.mode && s.display", 60)
    time.sleep(0.4)


def st(drv):
    return drv.execute_script("return window.__mace.freqView.state()")


def run(drv, cmd):
    drv.execute_script("window.__mace.run(arguments[0])", cmd)
    time.sleep(0.4)


def reason(drv, cmd):
    return drv.execute_script("return window.__mace.menuReason(arguments[0])", cmd)


def test_every_display_option_is_enabled_in_the_mode_view(drv):
    open_modes(drv)
    for cmd in TOOLBAR:
        b = drv.execute_script("return document.querySelector(`#toolbar [data-cmd='${arguments[0]}']`)", cmd)
        assert b is not None, cmd
        H.wait(drv, "return document.querySelector(`#toolbar [data-cmd='${arguments[0]}']`).getAttribute('aria-disabled') === 'false'", 5, cmd)
        assert reason(drv, cmd) == "", cmd
    for cmd in MENU:
        assert reason(drv, cmd) == "", (cmd, reason(drv, cmd))
    # the atom style needs the 3D view focused, as in the structure view
    drv.execute_script("document.getElementById('viewer-freq').focus()")
    assert reason(drv, "view.style.spacefill") == ""
    # what cannot apply says why (a tooltip, not just gray)
    for cmd in ["tool.move", "tool.add", "edit.undo", "file.save"]:
        assert reason(drv, cmd), cmd
    assert reason(drv, "tool.move").startswith("The mode view is view-only"), reason(drv, "tool.move")
    assert reason(drv, "view.polyhedra").startswith("Not in the mode view"), reason(drv, "view.polyhedra")
    b = drv.execute_script("return document.querySelector(`#toolbar [data-cmd='tool.move']`)")
    assert b.get_attribute("aria-disabled") == "true" and b.get_attribute("data-reason")
    assert not H.errors(drv), H.errors(drv)


def test_cell_bonds_labels_style_projection(drv):
    open_modes(drv)
    assert st(drv)["display"]["cell"] is True
    shapes0 = drv.execute_script(f"return {FV}.shapes.length")
    run(drv, "view.cell")
    assert st(drv)["display"]["cell"] is False
    assert drv.execute_script(f"return {FV}.shapes.length") < shapes0          # the 12 cell edges are gone
    run(drv, "view.cell")
    assert st(drv)["display"]["cell"] is True
    # bonds: sticks in the style, and kept from the rest geometry in every frame
    assert drv.execute_script(f"const a = {FV}.getModel().atoms[0]; return !!(a.style && a.style.stick)")
    same = drv.execute_script(f"""const f = {FV}.getModel().frames; if (!f || f.length < 2) return null;
      return f.every((fr) => fr.every((a, i) => JSON.stringify(a.bonds) === JSON.stringify(f[0][i].bonds)))""")
    assert same is True
    run(drv, "view.bonds")
    assert st(drv)["display"]["bonds"] is False
    assert drv.execute_script(f"const a = {FV}.getModel().atoms[0]; return !(a.style && a.style.stick)")
    run(drv, "view.bonds")
    # labels at the atoms
    run(drv, "view.labels.element")
    assert st(drv)["display"]["labels"] == "element"
    assert drv.execute_script(f"return {FV}.labels.length") > 0
    run(drv, "view.labels.off")
    assert drv.execute_script(f"return {FV}.labels.length") == 0
    # style
    drv.execute_script("document.getElementById('viewer-freq').focus()")
    run(drv, "view.style.spacefill")
    assert st(drv)["display"]["style"] == "spacefill"
    assert drv.execute_script(f"const a = {FV}.getModel().atoms[0]; return a.style.sphere && a.style.sphere.scale") > 0.9
    drv.execute_script("document.getElementById('viewer-freq').focus()")
    run(drv, "view.style.ballstick")
    # projection
    o0 = drv.execute_script(f"return !!{FV}.camera.ortho")
    run(drv, "view.projection")
    assert drv.execute_script(f"return !!{FV}.camera.ortho") != o0
    run(drv, "view.projection")
    # view along a, and Reset View
    q0 = drv.execute_script(f"return {FV}.getView().slice(4)")
    run(drv, "view.alongA")
    q1 = drv.execute_script(f"return {FV}.getView().slice(4)")
    assert not np.allclose(q0, q1)
    run(drv, "view.reset")
    assert np.allclose(drv.execute_script(f"return {FV}.getView().slice(4)"), q0, atol=1e-6)
    assert not H.errors(drv), H.errors(drv)


def test_supercell_moves_every_image_as_its_atom(drv):
    open_modes(drv)
    n1 = st(drv)["display"]["atoms"]
    n = drv.execute_script("return window.__mace.freqView.doc().structure.numbers.length")
    assert n1 == n
    run(drv, "view.range2")
    s = st(drv)
    assert s["display"]["atoms"] == 8 * n, s["display"]
    assert s["display"]["box"]["hi"] == [2, 2, 2]
    assert drv.execute_script("return document.getElementById('tb-range-label').textContent") == "2×2×2"
    # every image of an atom is displaced exactly as the atom (Γ): frame 6 (the turning point) minus frame 0
    d = drv.execute_script(f"""const f = {FV}.getModel().frames, a = f[6], b = f[0];
      return a.map((x, k) => [x.x - b[k].x, x.y - b[k].y, x.z - b[k].z]);""")
    d = np.array(d)
    assert np.abs(d).max() > 0.05                                        # the mode moves atoms
    # the view lists each atom's 8 images together (boundary atoms off): all 8 move as one
    for i in range(n):
        assert np.allclose(d[8 * i:8 * i + 8], d[8 * i], atol=1e-4), (i, d[8 * i:8 * i + 8])
    # the inspector's 2x2x2 box mirrors the range, and 1x1x1 is back to one cell
    assert drv.execute_script("const b = document.getElementById('fo-super'); return b && b.checked")
    run(drv, "view.range1")
    assert st(drv)["display"]["atoms"] == n
    assert not H.errors(drv), H.errors(drv)


def test_off_gamma_mode_keeps_one_cell_and_says_why(drv):
    open_modes(drv)
    drv.execute_script("window.__mace.freqView.doc().gamma = false; window.__mace.freqView.redraw()")
    time.sleep(0.3)
    for cmd in ["view.range2", "view.rangeCustom", "view.boundary"]:
        assert "not at Γ" in reason(drv, cmd), (cmd, reason(drv, cmd))
    n = drv.execute_script("return window.__mace.freqView.doc().structure.numbers.length")
    assert st(drv)["display"]["atoms"] == n
    assert drv.execute_script("return !document.getElementById('fo-super')")
    drv.execute_script("window.__mace.freqView.doc().gamma = true")
    assert not H.errors(drv), H.errors(drv)


def test_comparison_follows_cell_and_bonds(drv):
    got = drv.execute_async_script(
        "const d = arguments[1]; window.__mace.open(arguments[0]).then((x) => d(x && x.kind))", "outputs/Ag2Br3_opt_unfinished.out")
    assert got == "trajectory"
    H.wait(drv, "const v = window.MACE.state.viewers['viewer-before']; return v && v.getModel() && v.visible()", 30)
    time.sleep(0.4)
    VB = "window.MACE.state.viewers['viewer-before']"
    shapes0 = drv.execute_script(f"return {VB}.shapes.length")
    run(drv, "view.cell")
    assert drv.execute_script(f"return {VB}.shapes.length") < shapes0
    run(drv, "view.cell")
    run(drv, "view.bonds")
    assert drv.execute_script(f"return {VB}.getModelList().every((m) => m.selectedAtoms({{}}).every((a) => !(a.style && a.style.stick)))")
    run(drv, "view.bonds")
    # what the comparison cannot draw says why
    assert reason(drv, "view.range2").startswith("The comparison draws the run's own cell")
    assert not H.errors(drv), H.errors(drv)
