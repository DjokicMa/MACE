"""Batch Setup with spin-orbit coupling in a real browser (1.1.6, queue 11): a CIF folder with PbTe and
HfO, single points with SOC on.

* STUTSC is named "refused for Hf in 1 of 2 files" with the CLI's reason (soc_ecp.py's table, as Setup
  grays it out), and grayed out when every file holds Hf;
* the batch's STUTSC is refused for HfO in the CLI's words; the file's Override dialog has the SOC
  controls, and INPSOC for that file alone writes its deck while PbTe keeps STUTSC.

MACE_GUI_BROWSER picks the browser (Firefox by default)."""
import json
import time

import pytest

import guiharness as H
import p4util as U
from test_setup_soc_api import HFO_CIF, PBTE_CIF, SOC_READY

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")

B = "#batch-host "


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("batchsoc")
    root = H.examples_copy(tmp)
    d = root / "soc"
    d.mkdir()
    (d / "PbTe.cif").write_text(PBTE_CIF)
    (d / "HfO.cif").write_text(HFO_CIF)
    a = U.start_app(tmp, root=root)
    try:
        yield a
    finally:
        U.stop_app(a)


def _js(drv, code, *args):
    return drv.execute_script(code, *args)


def _settings_with_soc(drv):
    _js(drv, "window.__mace.run('explorer.batch', {dir: 'soc'})")
    H.wait(drv, f"return document.querySelectorAll('{B}[data-part=rows] tbody tr').length === 2", 90)
    _js(drv, f"const s = document.querySelector('{B}[data-b=calc]'); s.value = 'SP'; s.dispatchEvent(new Event('change', {{bubbles: true}}))")
    _js(drv, f"document.querySelector('{B}[data-go=\"settings\"]').click()")
    H.wait(drv, "return !!document.querySelector('#bat-soc-on') && !document.querySelector('#bat-soc-on').disabled", 300)
    # the answers that let a SOC deck be written (an explicit basis set), as the API tests use
    _js(drv, "Object.assign(window.__mace.doc().batch.answers, arguments[0])", SOC_READY)
    _js(drv, "document.querySelector('#bat-soc-on').click()")
    H.wait(drv, "return !!document.querySelector('#bat-soc-route')", 30)


def test_batch_soc_stutsc_grayed_and_a_files_override(app):
    drv = app[0]
    U.reset(drv)
    _settings_with_soc(drv)
    # STUTSC named for Hf, with the CLI's reason under the select
    H.wait(drv, "const o = document.querySelector('#bat-soc-route option[value=stutsc]'); return o && /refused for Hf in 1 of 2 files/.test(o.textContent)", 60)
    why = _js(drv, "return document.querySelector('#bat-soc-why-stutsc').textContent")
    assert why.startswith("Hf: CRYSTAL's internal STUTSC for Hf is not the published ECP60MDF"), why
    assert _js(drv, "return document.querySelector('#bat-soc-route').getAttribute('aria-describedby')") == "bat-soc-why-stutsc"
    assert not _js(drv, "return document.querySelector('#bat-soc-route option[value=stutsc]').disabled")
    _js(drv, "const s = document.querySelector('#bat-soc-route'); s.value = 'stutsc'; s.dispatchEvent(new Event('change', {bubbles: true}))")
    assert _js(drv, "return window.__mace.doc().batch.patches") == {"soc": True, "soc_ecp": "stutsc"}
    # Preview: HfO refused in the CLI's words, with an Override… button all the same
    _js(drv, f"document.querySelector('{B}[data-go=\"preview\"]').click()")
    row = f"{B}[data-part=plan] tbody tr"
    H.wait(drv, f"return document.querySelectorAll('{row}').length === 2", 900)
    H.wait(drv, f"return !window.__mace.doc().batch.planning", 900)
    rows = _js(drv, f"return [...document.querySelectorAll('{row}')].map((r) => r.innerText)")
    hf = next(r for r in rows if "HfO.cif" in r)
    assert "STUTSC for Hf is not the published ECP60MDF" in hf, rows
    _js(drv, f"document.querySelector('{B}[data-override=\"soc/HfO.cif\"]').click()")
    dlg = "[...document.querySelectorAll('dialog[open]')].pop()"
    H.wait(drv, f"const d = {dlg}; return d && d.querySelector('#bat-ov-soc-on')", 10)
    assert _js(drv, f"return {dlg}.querySelector('#bat-ov-soc-on').checked")
    assert _js(drv, f"return {dlg}.querySelector('#bat-ov-soc-route').value") == "stutsc"
    # STUTSC is grayed out for this file (its Hf), with the reason
    assert "Hf:" in _js(drv, f"return {dlg}.querySelector('#bat-ov-why-stutsc').textContent")
    _js(drv, f"const s = {dlg}.querySelector('#bat-ov-soc-route'); s.value = 'inpsoc'; s.dispatchEvent(new Event('change', {{bubbles: true}}))")
    _js(drv, f"[...{dlg}.querySelectorAll('button')].find((b) => /Apply Override/.test(b.textContent)).click()")
    H.wait(drv, "const b = window.__mace.doc().batch; return b.overridePatches['soc/HfO.cif'] && b.plan && !b.planning && "
                "b.plan.files.every((f) => f.status === 'ok')", 900)
    assert _js(drv, "return window.__mace.doc().batch.overridePatches") == {"soc/HfO.cif": {"soc_ecp": None}}
    plan = _js(drv, "return window.__mace.doc().batch.plan.files.map((f) => [f.rel, f.status, !!f.override])")
    assert sorted(plan) == [["soc/HfO.cif", "ok", True], ["soc/PbTe.cif", "ok", False]], plan
    U.no_errors(drv)


def test_batch_stutsc_grayed_when_every_file_has_hf(app):
    drv = app[0]
    U.reset(drv)
    _js(drv, "window.__mace.run('explorer.batch', {dir: 'soc'})")
    H.wait(drv, f"return document.querySelectorAll('{B}[data-part=rows] tbody tr').length === 2", 90)
    # HfO alone
    _js(drv, "const b = window.__mace.doc().batch; b.selected = new Set(['soc/HfO.cif'])")
    _js(drv, f"const s = document.querySelector('{B}[data-b=calc]'); s.value = 'SP'; s.dispatchEvent(new Event('change', {{bubbles: true}}))")
    _js(drv, f"document.querySelector('{B}[data-go=\"settings\"]').click()")
    H.wait(drv, "return !!document.querySelector('#bat-soc-on') && !document.querySelector('#bat-soc-on').disabled", 300)
    _js(drv, "document.querySelector('#bat-soc-on').click()")
    H.wait(drv, "const o = document.querySelector('#bat-soc-route option[value=stutsc]'); return o && o.disabled && /refused for Hf\\)/.test(o.textContent)", 60)
    U.no_errors(drv)
