"""Every dialog the Analysis menu opens in 1.1.6 (Phonopy, Mechanical Properties, Relaxation Time, and
the Structure Analysis panel's tools: porosity, PXRD, topology, ...) closes like the other dialogs, in
headless Firefox: its X button and Esc both close it, and a closed dialog is gone from the screen (a
dialog styled display: flex stayed drawn after close(), so it could not be closed at all).
"""
import shutil

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")

FIX = H.REPO_ROOT / "tests" / "fixtures" / "phonopy_crystal"
# (command, its arguments, the dialog's id)
DIALOGS = [("analysis.phonopy", {"path": "si_opt.out"}, "phonopy-dlg"),
           ("analysis.elastic", {"path": "si_opt.out"}, "elastic-dlg"),
           ("analysis.tau", {"path": "si_opt.out"}, "tau-dlg")] + [
    (f"analysis.structure.{tab}", {}, "analysis-panel")
    for tab in ("porosity", "pxrd", "fit", "topology", "polyhedra", "bonds", "morphology", "planes")]


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("andlg")
    root = tmp / "root"
    (root / "structures").mkdir(parents=True)
    shutil.copy(FIX / "Si" / "parent.out", root / "si_opt.out")
    shutil.copy(FIX / "Si" / "parent.d12", root / "si_opt.d12")
    shutil.copy(H.EXAMPLES / "structures" / "MgO.cif", root / "structures" / "MgO.cif")
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_NO_WARMUP="1")
    drv = None
    try:
        drv = H.firefox(1280, 900)
        H.load(drv, base)
        U.open_doc(drv, "structures/MgO.cif")
        yield drv
    finally:
        if drv is not None:
            drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


SHOWN = ("const d = document.getElementById(arguments[0]); if (!d) return false;"
         "const r = d.getBoundingClientRect(); return getComputedStyle(d).display !== 'none' && r.width > 0 && r.height > 0")


def _open(drv, command, args, dlg):
    drv.execute_script("window.__mace.run(arguments[0], arguments[1])", command, args)
    H.wait(drv, f"const d = document.getElementById('{dlg}'); return d && d.open", 30)
    assert drv.execute_script(SHOWN, dlg), f"{dlg} opened but is not drawn"


def _closed(drv, dlg):
    H.wait(drv, f"const d = document.getElementById('{dlg}'); return !d || !d.open", 10)
    assert not drv.execute_script(SHOWN, dlg), f"{dlg} is closed but still drawn"


@pytest.mark.parametrize("command, args, dlg", DIALOGS, ids=[c for c, _, _ in DIALOGS])
def test_analysis_dialog_closes_with_its_x_and_with_esc(app, command, args, dlg):
    drv = app
    _open(drv, command, args, dlg)
    drv.execute_script("document.querySelector(`#${arguments[0]} .dlg-head [data-close]`).click()", dlg)
    _closed(drv, dlg)
    _open(drv, command, args, dlg)
    # Esc in the dialog (its first field, or the dialog itself), as the other panels close
    drv.execute_script(
        "const d = document.getElementById(arguments[0]);"
        "const t = d.querySelector('.dlg-body :is(input, select, textarea, button)') || d;"
        "t.focus(); t.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true, cancelable: true}))", dlg)
    _closed(drv, dlg)
    U.no_errors(drv)
