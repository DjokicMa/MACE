"""`mace gui` in a real headless browser (Firefox through selenium).

Skipped when selenium, Firefox or WebGL is missing. Set MACE_GUI_SHOTS to a
directory to also save light and dark screenshots of each tab there.
"""
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent / "gui"))
import guiharness  # noqa: E402

webdriver = pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
if shutil.which("firefox") is None:
    pytest.skip("Firefox is not installed", allow_module_level=True)

from selenium.webdriver.common.by import By  # noqa: E402

EXAMPLES = REPO_ROOT / "examples" / "gui"


@pytest.fixture(scope="module")
def browser(tmp_path_factory):
    from test_gui_examples import TOKEN, _start
    from test_gui_results import write_band
    from test_ipdos_readers import write_doss
    root = tmp_path_factory.mktemp("guibrowser") / "demo"
    shutil.copytree(EXAMPLES, root)
    (root / "results").mkdir()
    write_band(root / "results")
    write_doss(root / "results", material="mgo")
    # the Inspector in Advanced (1.1.6 tiers), as tests/gui/guiharness.py starts it: these tests use the
    # classic editing forms and the lattice rows, which Basic hides
    proc, base, _ = _start(root, env={"MACE_GUI_INSPECTOR_TIER": "advanced"})
    try:  # software GL, never a GPU (spec section 9)
        drv = guiharness.firefox(1500, 950)
    except BaseException:
        proc.terminate()
        raise
    try:
        drv.get(guiharness.login_url(base, TOKEN))
        if not drv.execute_script(
                "return !!document.createElement('canvas').getContext('webgl')"):
            pytest.skip("no WebGL in this headless Firefox")
        _wait(drv, "return document.querySelectorAll('#file-list .file').length > 5")
        # the shell opens on its welcome screen; the files open as tabs
        _wait(drv, "return document.documentElement.dataset.appReady === '1'")
        yield drv, root
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def _wait(drv, script, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        if drv.execute_script(script):
            return
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for: {script}")


def _click(drv, css):
    drv.execute_script("document.querySelector(arguments[0]).click()", css)


def _errors(drv):
    return drv.execute_script("return window.__maceErrors || []")


def _shots(drv, name):
    target = os.environ.get("MACE_GUI_SHOTS")
    if not target:
        return
    Path(target).mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        drv.execute_script("document.documentElement.setAttribute('data-theme', arguments[0]);"
                           "document.dispatchEvent(new CustomEvent('mace-theme'));", theme)
        time.sleep(1.0)
        drv.save_screenshot(str(Path(target) / f"{theme}_{name}.png"))


def test_structure_tab_draws_with_reduced_formula(browser):
    drv, _ = browser
    _click(drv, '.file[data-path="structures/MgO.cif"]')
    # innerText: the card is on screen (in the inspector), not only filled in
    _wait(drv, "return document.querySelector('#s-formula').innerText.includes('MgO')")
    # the card moves from its parking place into the inspector a moment later
    # (innerText of a hidden element is its whole text, so wait for it to be laid out)
    _wait(drv, "const f = document.querySelector('#s-formula'); return f.getClientRects().length > 0 && f.innerText.includes('Z = 4')")
    assert "Z = 4" in drv.find_element(By.ID, "s-formula").text
    # Fm-3m typeset as in the International Tables: a bar over the 3
    assert drv.execute_script(
        "const h=document.querySelector('#s-sym .hm');"
        "return [h.textContent, h.querySelector('.bar').textContent]") == ["Fm3m", "3"]
    _wait(drv, "return !!document.querySelector('#viewer-main canvas')")
    _shots(drv, "1_structure_MgO")
    _click(drv, '.file[data-path="outputs/graphene_slab_sp.out"]')
    _wait(drv, "return document.querySelector('#s-formula').textContent.startsWith('C')")
    # a slab is an editable document since 1.1.6: the Inspector's Cell section (not the read-only lattice
    # card) shows c as not periodic and the area, not a volume
    cell = "#inspector .insp-section[data-key='cell']"
    _wait(drv, f"const e = document.querySelector(\"{cell}\"); return !!e && e.innerText.includes('Area')")
    lattice = drv.find_element(By.CSS_SELECTOR, cell).text
    assert "not periodic" in lattice and "Area" in lattice and "Volume" not in lattice
    _shots(drv, "1b_structure_graphene_slab")
    assert not _errors(drv), _errors(drv)


def test_results_tab_draws_interactive_band_and_dos(browser):
    # the interactive plots come from plotly: an environment without it skips, not times out
    pytest.importorskip("plotly")
    drv, _ = browser
    _click(drv, '.file[data-path="results/mgo_band.BAND.DAT"]')
    # a plot document (Phase 5): the band structure with its DOS partner on one energy axis
    _wait(drv, "return !!document.querySelector('#plot-host .pv-plot .main-svg')")
    text = drv.find_element(By.ID, "plot-host").text.lower()
    assert "closed shell" in text and "mgo_doss.doss.dat" in text
    # the DOS axis fits the states inside the +-6 eV window on screen
    import json
    rng = drv.execute_script(
        "return document.querySelector('#plot-host .pv-plot').layout.xaxis2.range")
    dos = json.loads(drv.execute_async_script(
        "const done=arguments[0]; fetch('api/dos?path=results/mgo_doss.DOSS.DAT',"
        "{headers:{'X-Requested-With':'mace-gui'}}).then(r=>r.text()).then(done)"))
    inside = [abs(p["up"][i]) for p in dos["projections"]
              for i, e in enumerate(dos["energy"]) if -6 <= e - dos["fermi"] <= 6]
    assert rng[0] == 0 and abs(rng[1] - 1.08 * max(inside)) < 1e-6
    # the plot toolbar stays out of the legend's way until the pointer is on the plot
    from selenium.webdriver.common.action_chains import ActionChains
    ActionChains(drv).move_to_element(drv.find_element(By.ID, "statusbar")).perform()
    opacity = "return getComputedStyle(document.querySelector('#plot-host .modebar')).opacity"
    _wait(drv, opacity + " === '0'")
    ActionChains(drv).move_to_element(drv.find_element(By.CSS_SELECTOR, "#plot-host .pv-plot .main-svg")).perform()
    _wait(drv, opacity + " === '1'")
    _shots(drv, "3_results_band_dos")
    assert not _errors(drv), _errors(drv)


def test_optimization_viewer_fits_the_window(browser):
    drv, _ = browser
    _click(drv, '.file[data-path="outputs/Ag2Br3_opt_unfinished.out"]')
    _wait(drv, "return document.querySelector('#tab-opt').classList.contains('active')")
    time.sleep(0.5)
    bottom, height = drv.execute_script(
        "const r=document.querySelector('#tab-opt .viewer-wrap').getBoundingClientRect();"
        "return [r.bottom, window.innerHeight]")
    assert bottom <= height    # the toolbar under the viewer is on screen
    # 3Dmol sizes the hidden viewer to 0x0; opening the same run again draws
    # it, at its old size, before 3Dmol's resize observer fires
    _click(drv, '.file[data-path="structures/MgO.cif"]')
    _wait(drv, "return document.querySelector('#tab-structure').classList.contains('active')")
    time.sleep(0.5)
    _click(drv, '.file[data-path="outputs/Ag2Br3_opt_unfinished.out"]')
    _wait(drv, "return document.querySelector('#tab-opt').classList.contains('active')")
    time.sleep(1.0)
    assert not _errors(drv), _errors(drv)


def test_editor_edits_undoes_and_cleans(browser):
    drv, root = browser
    _click(drv, '.file[data-path="structures/MgO.cif"]')
    _wait(drv, "return document.querySelector('#s-formula').textContent.includes('MgO')")
    # the classic card fills before the document is open in the editor (MgO opened here first, on its own)
    _wait(drv, "return !!(window.__mace.editor && window.__mace.editor.doc())")
    # Phase 3: the structure document is edited in place; the classic panel is
    # in its inspector ("More editing (classic)"), and Edit starts the session
    _click(drv, "#s-edit")
    _wait(drv, "return document.querySelector('#inspector #e-counts') && window.__mace.editor.doc().srv.session")
    _click(drv, "#e-locked")      # P1 editing (Sym is the default for MgO): Li at 8c adds one atom
    _wait(drv, "return window.__mace.editor.doc().symMode === 'p1'")
    # x, y, z sit on one line, whatever the width of the info column
    assert drv.execute_script(
        "return new Set(['#e-x','#e-y','#e-z','#e-dx','#e-dy','#e-dz'].map((s) =>"
        "document.querySelector(s).getBoundingClientRect().top)).size") == 2
    drv.execute_script("""
        document.querySelector('#e-element').value = 'Li';
        ['#e-x', '#e-y', '#e-z'].forEach((id) => document.querySelector(id).value = '0.25');""")
    _click(drv, "#e-add")
    _wait(drv, "return document.querySelector('#e-counts').textContent.startsWith('9 atoms')")
    _click(drv, "#e-undo")
    _wait(drv, "return document.querySelector('#e-counts').textContent.startsWith('8 atoms')")
    # Phase 4: the classic Clean button opens the Clean Geometry panel (a
    # proposal to accept or discard); Accept is one undo step
    _click(drv, "#e-clean")
    _wait(drv, "const f = window.__mace.builderFrame(); return f && f.id === 'clean' && !f.goDisabled", 60)
    # (the Undo above left a redo row: Accept replaces it, so count by position)
    pos = drv.execute_script("return window.__mace.editor.doc().pos")
    drv.execute_script("document.querySelector('dialog.builder-dialog .bld-go').click()")
    _wait(drv, f"const e = window.__mace.editor.doc(); return e.pos === {pos + 1} && e.rows.length === {pos + 2} && e.pending === 0", 30)
    assert drv.execute_script("const e = window.__mace.editor.doc(); return e.rows[e.pos].label") == "Clean Geometry"
    _wait(drv, "return !document.querySelector('#e-undo').disabled")        # the panel redraws on the next frame
    _click(drv, "#e-undo")
    _wait(drv, f"return window.__mace.editor.doc().pos === {pos}")
    # the commit-at-once flow (POST /api/edit/clean, kept for old clients) still works
    drv.execute_script("window.__maceClassicClean()")
    _wait(drv, "return document.querySelector('#e-status').textContent.includes('Clean')", 60)
    _shots(drv, "2_editor_MgO")
    # the Clean result panel (with its "Undo Clean") goes away once another
    # edit lands, so that button never undoes something else
    _wait(drv, "return !document.querySelector('#c-result').classList.contains('hidden')")
    # Before / after reuses the Optimization viewer, which 3Dmol sized to 0x0
    # when its tab was hidden: it must come back without a canvas error
    _click(drv, "#c-compare")
    _wait(drv, "return document.querySelector('#tab-opt').classList.contains('active')")
    time.sleep(1.0)
    assert not _errors(drv), _errors(drv)
    drv.execute_script("const i = window.__mace.docs().findIndex((d) => d.kind === 'structure');"
                       "return window.__mace.activate(i)")
    _wait(drv, "return document.querySelector('#inspector #e-counts')")
    _click(drv, "#e-add")
    _wait(drv, "return document.querySelector('#e-counts').textContent.startsWith('9 atoms')")
    assert "hidden" in drv.find_element(By.ID, "c-result").get_attribute("class")
    _click(drv, "#e-undo")
    _wait(drv, "return document.querySelector('#e-counts').textContent.startsWith('8 atoms')")
    drv.execute_script("document.querySelector('#e-cifname').value = 'MgO_gui.cif'")
    _click(drv, "#e-save")
    # the one save path: Save As, with the card's name filled in
    _wait(drv, "const d = document.getElementById('save-as'); return d && d.open")
    assert drv.execute_script("return document.querySelector('#save-as [data-k=\"name\"]').value") == "MgO_gui.cif"
    drv.execute_script("document.querySelector('#save-as [data-v=\"save\"]').click()")
    # Save As starts in the document's own folder
    _wait(drv, "return !!window.MACE.fileByPath('structures/MgO_gui.cif')")
    assert (root / "structures" / "MgO_gui.cif").is_file()
    assert not _errors(drv), _errors(drv)
