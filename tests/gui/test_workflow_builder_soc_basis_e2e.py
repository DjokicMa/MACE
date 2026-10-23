"""Switching a later SP step to spin-orbit coupling in the New Workflow dialog (1.1.6): the step's inherited
internal basis set becomes the matching external library, the heavy atoms take the official SOC basis set,
the group says why, the internal sets and the functionals without a two-component form are grayed out, and
the projected deck in the review is a SOC deck."""
import shutil

import pytest

import guiharness as H
import submit_util as SU
from test_setup_soc_api import PBTE_CIF
from test_workflow_builder_step_options_e2e import _click

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")

FORM = ".wfb-form[data-step='SP_2']"


@pytest.fixture
def app(tmp_path):
    if not (H.REPO_ROOT / "test" / "CIFs").exists():
        pytest.skip("test/ data corpus not present")
    from fake_slurm import FakeSlurm
    root = tmp_path / "proj"
    (root / "cifs").mkdir(parents=True)
    for n in ("A", "B"):
        (root / "cifs" / f"{n}.cif").write_text(PBTE_CIF)
    fake = FakeSlurm(tmp_path / "fs")
    env = SU.fake_env(fake, tmp_path)
    env.pop("XDG_CACHE_HOME")
    env.update(MACE_GUI_TEST_HOOKS="1", MACE_GUI_NO_WARMUP="1", MACE_SITE_CONFIG="none")
    proc, base, _ = H.start_server(root, tmp_path / "xdg", **env)
    drv = H.firefox(1440, 900)
    try:
        H.load(drv, base)
        yield {"drv": drv}
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def test_soc_on_picks_a_basis_the_soc_deck_can_take(app):
    drv = app["drv"]
    drv.execute_script("window.__mace.run('explorer.workflow', {dir: 'cifs'})")
    H.wait(drv, "const s = window.__maceWfBuilder && window.__maceWfBuilder.state(); return s && s.inputs && s.inputs.files.length === 2", 60)
    _click(drv, "[data-act='next']")
    _click(drv, "[data-template='opt_sp']")
    H.wait(drv, "return document.querySelectorAll('.wfb-chain .wfb-node').length === 2")
    _click(drv, "[data-act='next']")
    _click(drv, "[data-form='SP_2']")
    H.wait(drv, f"return document.querySelector(\"{FORM} [data-soc='soc']\")", 120)
    _click(drv, f"{FORM} [data-soc='soc']")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); const x = s.draft.steps.SP_2; return !s.formBusy.SP_2 && x.soc && x.soc.soc_basis === 'pob-tzvp-rev2-soc' && x.answers.basis_set === 'TZVP-REV2'", 120)
    H.wait(drv, f"return document.querySelector(\"{FORM} select[data-field='basis_set']\")", 60)
    note = drv.execute_script(f"const e = document.querySelector(\"{FORM} [data-part='soc-basis']\"); return e && e.textContent")
    assert note and "TZVP-REV2" in note and "pob-TZVP-rev2-SOC" in note, note
    off = drv.execute_script(f"return [...document.querySelector(\"{FORM} select[data-field='basis_set']\").options].filter((o) => o.disabled).map((o) => o.value)")
    assert "POB-TZVP-REV2" in off and "TZVP-REV2" not in off, off
    fn_off = drv.execute_script(f"return [...document.querySelector(\"{FORM} select[data-field='functional']\").options].filter((o) => o.disabled).map((o) => o.value)")
    assert "HSE06" in fn_off and "PBE0" not in fn_off, fn_off
    errors = drv.execute_script(f"return [...document.querySelectorAll(\"{FORM} [data-part='soc-error']\")].map((e) => e.textContent)")
    assert not errors, errors
    # switched off again: the basis set goes back to the parent's
    _click(drv, f"{FORM} [data-soc='soc']")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); const x = s.draft.steps.SP_2; return !s.formBusy.SP_2 && !x.soc && !('basis_set' in (x.answers || {}))", 120)
    _click(drv, f"{FORM} [data-soc='soc']")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); const x = s.draft.steps.SP_2; return !s.formBusy.SP_2 && x.soc && x.answers.basis_set === 'TZVP-REV2'", 120)
    _click(drv, "[data-act='next']")
    H.wait(drv, "return window.__maceWfBuilder.state().sec === 3")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.defaults && !s.defaultsBusy", 120)
    _click(drv, "[data-act='next']")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return (s.preview && !s.previewing) || s.previewError", 300)
    p = drv.execute_script("return window.__maceWfBuilder.state().preview")
    assert p and p["code"] == 0, drv.execute_script("return window.__maceWfBuilder.state().previewError")
    sp = {s["step"]: s for s in p["projected"]}["SP_2"]["deck_text"]
    assert "TWOCOMPON" in sp and "STUTSC" in sp
    assert not H.errors(drv), H.errors(drv)
