"""Settings › ML potentials in headless Firefox: Calculation › ML Potentials…
opens it, Set Up ML Environment… checks the plan, installs through the real
installer (a fake base Python whose venv's pip only logs: no network) with
its log streaming into the dialog, the pane then shows the recorded
environment and CHGNet ready, a weights download waits for its license to
be accepted, and Use an Existing Python… refuses an interpreter that does
not run. No JS errors."""
import sys
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preopt"))
import preopt_fixtures as F  # noqa: E402


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mlip_e2e")
    root = H.examples_copy(tmp)
    base, fenv = F.make_fake_base_python(tmp / "fakepy", tmp / "pip.log")
    a = U.start_app(tmp, root=root, MACE_GUI_NO_WARMUP="1", PATH="/usr/bin:/bin", XDG_DATA_HOME=str(tmp / "data"), **fenv)
    try:
        yield a, str(base), tmp
    finally:
        U.stop_app(a)


def js_set(drv, sel, value=None, checked=None):
    drv.execute_script("""const e = document.querySelector(arguments[0]);
        if (arguments[1] !== null) e.value = arguments[1];
        if (arguments[2] !== null) e.checked = arguments[2];
        e.dispatchEvent(new Event('input', {bubbles: true})); e.dispatchEvent(new Event('change', {bubbles: true}));""",
                       sel, value, checked)


def click(drv, sel):
    drv.execute_script("document.querySelector(arguments[0]).click()", sel)


def text(drv, sel):
    return drv.execute_script("const e = document.querySelector(arguments[0]); return e ? e.textContent : ''", sel)


def test_install_from_settings_and_the_license_gates(app):
    (drv, *_), base, tmp = app
    U.reset(drv)
    drv.execute_script("window.__mace.run('calc.mlip')")
    H.wait(drv, "const e = document.querySelector('#settings #mlip-env'); return !!e && e.textContent.includes('none yet')", 60)
    click(drv, "#mlip-setup")
    H.wait(drv, "return !!document.querySelector('#mlip-setup-dlg[open]')", 10)
    js_set(drv, "#mlip-python", base)
    js_set(drv, "#mlip-setup-dlg [data-pkg='mace-torch']", checked=False)
    js_set(drv, "#mlip-setup-dlg [data-pkg='chgnet']", checked=True)
    H.wait(drv, "return document.querySelector('#mlip-plan').textContent.startsWith('Into')", 30)
    assert drv.execute_script("return document.querySelector('#mlip-fetch').disabled")     # no mace-torch: no weights
    H.wait(drv, "return !document.querySelector('#mlip-start').disabled", 10)
    click(drv, "#mlip-start")
    H.wait(drv, "const r = document.querySelector('#mlip-result'); return r && !r.hidden && r.textContent.length > 0", 120)
    assert "The ML environment is ready" in text(drv, "#mlip-result"), text(drv, "#mlip-result")
    log = text(drv, "#mlip-log")
    assert "== Upgrading pip" in log and "chgnet==0.4.2" in log
    click(drv, "#mlip-setup-dlg [data-close]")
    H.wait(drv, "const e = document.querySelector('#settings #mlip-env'); return !!e && e.textContent.includes('/bin/python')", 60)
    assert "ready" in text(drv, "#mlip-models tr[data-model='chgnet-0.3.0'] .mlip-status")
    # the default model's weights: Download waits for the license to be accepted (nothing is downloaded here)
    H.wait(drv, "return !!document.querySelector(\"[data-fetch='acesuit-mace-mpa-0-medium']\")", 30)
    click(drv, "[data-fetch='acesuit-mace-mpa-0-medium']")
    H.wait(drv, "return !!document.querySelector('#mlip-fetch-dlg[open]')", 10)
    assert "MIT" in text(drv, "#mlip-fetch-dlg .kv")
    assert drv.execute_script("return document.querySelector('#mlip-fetch-go').disabled")
    js_set(drv, "#mlip-fetch-accept", checked=True)
    assert not drv.execute_script("return document.querySelector('#mlip-fetch-go').disabled")
    click(drv, "#mlip-fetch-dlg [data-close]")
    # Set Up again with mace-torch and its weights: Start waits for the weights' license too
    click(drv, "#mlip-setup")
    H.wait(drv, "return !!document.querySelector('#mlip-setup-dlg[open]')", 10)
    js_set(drv, "#mlip-python", base)
    H.wait(drv, "return document.querySelector('#mlip-plan').textContent.startsWith('Repairs')", 30)
    assert drv.execute_script("return document.querySelector('#mlip-fetch').checked && document.querySelector('#mlip-start').disabled")
    js_set(drv, "#mlip-accept", checked=True)
    assert not drv.execute_script("return document.querySelector('#mlip-start').disabled")
    click(drv, "#mlip-setup-dlg [data-close]")
    # Use an Existing Python…: one that does not run is not recorded
    click(drv, "#mlip-use")
    H.wait(drv, "return !!document.querySelector('#mlip-use-dlg[open]')", 10)
    js_set(drv, "#mlip-use-python", str(tmp / "nope" / "python"))
    click(drv, "#mlip-use-go")
    H.wait(drv, "const r = document.querySelector('#mlip-use-result'); return r && !r.hidden", 60)
    assert "Not recorded" in text(drv, "#mlip-use-result")
    click(drv, "#mlip-use-dlg [data-close]")
    U.no_errors(drv)


def test_clean_and_calculators_link_to_the_settings(app):
    (drv, *_), base, tmp = app
    U.reset(drv)
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('calc.calculators')")
    H.wait(drv, "return !!document.querySelector('#calc-panel [data-k=\"mlip-setup\"]')", 20)
    click(drv, "#calc-panel [data-k='mlip-setup']")
    H.wait(drv, "return !!document.querySelector('#settings[open] #mlip-host')", 20)
    U.no_errors(drv)


def test_academic_model_needs_its_own_acknowledgment(app):
    (drv, *_), base, tmp = app
    U.reset(drv)
    drv.execute_script("window.__mace.run('calc.mlip')")
    H.wait(drv, "return !!document.querySelector(\"#settings tr[data-model='acesuit-mace-off24-medium']\")", 60)
    assert "Academic, non-commercial use only" in text(drv, "#settings tr[data-model='acesuit-mace-off24-medium']")
    click(drv, "[data-fetch='acesuit-mace-off24-medium']")
    H.wait(drv, "return !!document.querySelector('#mlip-fetch-dlg[open]')", 10)
    assert "ASL" in text(drv, "#mlip-fetch-academic-note") and "may not be used commercially" in text(drv, "#mlip-fetch-academic-note")
    assert drv.execute_script("return document.querySelector('#mlip-fetch-go').disabled")
    js_set(drv, "#mlip-fetch-accept", checked=True)
    assert not drv.execute_script("return document.querySelector('#mlip-fetch-go').disabled")
    click(drv, "#mlip-fetch-dlg [data-close]")
    U.no_errors(drv)
