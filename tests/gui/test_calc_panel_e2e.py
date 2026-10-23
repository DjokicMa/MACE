"""Calculation › Calculators… in headless Firefox: the panel opens from its
command on a structure, lists the backends with the ML potentials labeled by
package (MACE-MP from mace-torch), hides the device row for a
force field, runs a UFF4MOF single point and an EQeq charge job with their
readouts, opens an optimized structure as a new document, and shows the
setup commands of an uninstalled backend. No JS errors."""
import sys
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preopt"))
import preopt_fixtures as F  # noqa: E402


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("calc_e2e")
    root = H.examples_copy(tmp)
    py = F.make_fake_python(tmp / "venv")
    a = U.start_app(tmp, root=root, MACE_GUI_NO_WARMUP="1", MACE_MLIP_PYTHON=str(py), PATH="/usr/bin:/bin",
                    XDG_DATA_HOME=str(tmp / "data"))
    try:
        yield a
    finally:
        U.stop_app(a)


def q(drv, k, prop="value"):
    return drv.execute_script(f"const e = document.querySelector('#calc-panel [data-k=\"{k}\"]'); return e ? e.{prop} : null")


def set_field(drv, k, value):
    drv.execute_script(f"""const e = document.querySelector('#calc-panel [data-k="{k}"]');
        e.value = arguments[0]; e.dispatchEvent(new Event('input', {{bubbles: true}})); e.dispatchEvent(new Event('change', {{bubbles: true}}));""", value)


def open_panel(drv, path):
    U.open_doc(drv, path)
    drv.execute_script("window.__mace.run('calc.calculators')")
    H.wait(drv, "return !!document.querySelector('#calc-panel[open]')", 10)
    H.wait(drv, "const c = document.querySelector('#calc-panel [data-k=\"checking\"]'); return c && c.hidden", 60)


def run_and_wait(drv, timeout=120):
    drv.execute_script("document.querySelector('#calc-panel [data-k=\"run\"]').click()")
    H.wait(drv, "const r = document.querySelector('#calc-panel [data-k=\"summary\"]'); return r && r.textContent.length > 0", timeout)
    return q(drv, "result", "textContent")


def test_panel_uff_eqeq_and_hints(app):
    drv = app[0]
    U.reset(drv)
    open_panel(drv, "structures/MgO.cif")
    labels = drv.execute_script("return [...document.querySelectorAll('#calc-panel [data-k=\"backend\"] option')].map((o) => o.textContent)")
    assert any("UFF" in x for x in labels) and any("mace-torch" in x for x in labels)
    set_field(drv, "backend", "uff")
    assert drv.execute_script("return document.querySelector('#calc-panel [data-k=\"device-row\"]').hidden")
    set_field(drv, "task", "sp")
    text = run_and_wait(drv)
    assert "Single point" in text and "Energy" in text
    set_field(drv, "backend", "eqeq")
    set_field(drv, "task", "charges")
    text = run_and_wait(drv)
    assert "Charges by element" in text and "Mg" in text
    # an uninstalled backend: its setup commands are listed
    assert "conda install" in q(drv, "unavailable", "textContent")
    U.no_errors(drv)


def test_panel_opt_opens_new_document(app):
    drv = app[0]
    U.reset(drv)
    open_panel(drv, "structures/NaCl.cif")
    n0 = drv.execute_script("return window.__mace.docs ? window.__mace.docs().length : document.querySelectorAll('.tab').length")
    set_field(drv, "backend", "uff")
    set_field(drv, "task", "opt")
    set_field(drv, "steps", "20")
    text = run_and_wait(drv)
    assert "Geometry optimization" in text and "CRYSTAL OPT" in text
    end = time.time() + 20
    while time.time() < end:
        n1 = drv.execute_script("return window.__mace.docs ? window.__mace.docs().length : document.querySelectorAll('.tab').length")
        if n1 > n0:
            break
        time.sleep(0.2)
    assert n1 > n0
    U.no_errors(drv)


def test_panel_charge_spin_and_task_head_for_the_models_that_take_them(app):
    drv = app[0]
    U.reset(drv)
    open_panel(drv, "structures/MgO.cif")
    set_field(drv, "backend", "mlip")
    set_field(drv, "method", "orbmol-v2")
    assert not q(drv, "mult-row", "hidden") and not q(drv, "charge-row", "hidden") and not q(drv, "cs-note", "hidden")
    assert q(drv, "charge") == "" and q(drv, "head-row", "hidden")
    set_field(drv, "method", "meta-uma-s-1p1")
    assert not q(drv, "head-row", "hidden")
    heads = drv.execute_script("return [...document.querySelectorAll('#calc-panel [data-k=\"task_head\"] option')].map((o) => o.value)")
    assert heads == ["", "omat", "omol", "oc20", "odac", "omc"]
    set_field(drv, "method", "acesuit-mace-mpa-0-medium")
    assert q(drv, "mult-row", "hidden") and q(drv, "head-row", "hidden") and q(drv, "charge-row", "hidden")
    set_field(drv, "backend", "xtb")
    assert q(drv, "charge") == "0" and q(drv, "mult-row", "hidden")
    U.no_errors(drv)
