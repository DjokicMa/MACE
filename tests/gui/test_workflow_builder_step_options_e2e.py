"""A later step's settings in the New Workflow dialog (1.1.6): the SP step's basis set and SCF convergence
are changed in its form (as the CLI planner's customization levels change them), the menu options carry the
CLI's numbers, and the projected SP deck in the review has both."""
import shutil
import time

import pytest

import guiharness as H
import submit_util as SU

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")

TEST_DATA = H.REPO_ROOT / "test"


@pytest.fixture
def app(tmp_path):
    cif = TEST_DATA / "CIFs" / "1_dia_opt_rev1.cif"
    if not cif.exists():
        pytest.skip("test/ data corpus not present")
    from fake_slurm import FakeSlurm
    root = tmp_path / "proj"
    (root / "cifs").mkdir(parents=True)
    for n in ("A", "B"):
        shutil.copy(cif, root / "cifs" / f"{n}.cif")
    fake = FakeSlurm(tmp_path / "fs")
    env = SU.fake_env(fake, tmp_path)
    env.pop("XDG_CACHE_HOME")
    env.update(MACE_GUI_TEST_HOOKS="1", MACE_GUI_NO_WARMUP="1", MACE_SITE_CONFIG="none")
    proc, base, _ = H.start_server(root, tmp_path / "xdg", **env)
    drv = H.firefox(1440, 900)
    try:
        H.load(drv, base)
        yield {"drv": drv, "base": base, "root": root, "fake": fake}
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def _click(drv, css, timeout=30):
    from selenium.common.exceptions import StaleElementReferenceException
    for _ in range(10):
        el = H.wait(drv, f"const e = document.querySelector({css!r}); return e && !e.disabled ? e : null", timeout)
        try:
            drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
            el.click()
            return el
        except StaleElementReferenceException:
            time.sleep(0.2)
    raise AssertionError(f"{css} kept being replaced")


def _set(drv, find, value):
    H.wait(drv, f"return {find}", 300)
    assert drv.execute_script(f"const e = {find}; if (!e) return false; e.value = arguments[0]; "
                              "e.dispatchEvent(new Event('change', {bubbles: true})); return true", value)


def test_sp_basis_set_and_convergence(app):
    drv = app["drv"]
    drv.execute_script("window.__mace.run('explorer.workflow', {dir: 'cifs'})")
    H.wait(drv, "const s = window.__maceWfBuilder && window.__maceWfBuilder.state(); return s && s.inputs && s.inputs.files.length === 2", 60)
    _click(drv, "[data-act='next']")
    _click(drv, "[data-template='opt_sp']")
    H.wait(drv, "return document.querySelectorAll('.wfb-chain .wfb-node').length === 2")
    _click(drv, "[data-act='next']")
    _click(drv, "[data-form='SP_2']")
    H.wait(drv, "return document.querySelector(\".wfb-form[data-step='SP_2'] select[data-field='tolerance']\")", 120)
    # the editable rows of a later SCF step (spin polarization too); the rest stay the parent's (locked)
    rows = drv.execute_script("return [...document.querySelectorAll(\".wfb-form[data-step='SP_2'] .su-row:not(.locked)\")].map((r) => r.dataset.row)")
    assert {"functional", "dispersion", "basis_set", "tolerance", "dft_grid", "maxcycle", "spin"} <= set(rows), rows
    # the menus show the CLI's numbers; the value stays the name
    opts = drv.execute_script("return [...document.querySelector(\".wfb-form[data-step='SP_2'] select[data-field='tolerance']\").options].map((o) => [o.value, o.textContent])")
    assert opts[0][1].startswith("1 - ") and opts[0][1][4:].startswith(opts[0][0]), opts
    # each level shows the values it sets (d12_constants' presets)
    assert opts[1][1] == "2 - Tight (TOLINTEG: 8 8 8 9 24, TOLDEE: 9)", opts
    # no Method row, no basis-set type row, no functional family: they follow from the functional and the basis set
    ids = drv.execute_script("return [...document.querySelectorAll(\".wfb-form[data-step='SP_2'] .su-row\")].map((r) => r.dataset.row)")
    assert not {"method", "basis_type", "basis_keep"} & set(ids) and not any("category" in i for i in ids), ids
    _set(drv, "document.querySelector(\".wfb-form[data-step='SP_2'] select[data-field='basis_set']\")", "POB-DZVP-REV2")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return !s.formBusy.SP_2 && s.draft.steps.SP_2 && s.draft.steps.SP_2.answers.basis_set === 'POB-DZVP-REV2'", 120)
    _set(drv, "document.querySelector(\".wfb-form[data-step='SP_2'] select[data-field='tolerance']\")", "Tight")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return !s.formBusy.SP_2 && s.draft.steps.SP_2.answers.tolerance === 'Tight'", 120)
    sent = drv.execute_script("return window.__maceWfBuilder.draft().steps.SP_2")
    assert sent["answers"] == {"basis_set": "POB-DZVP-REV2", "tolerance": "Tight"} and "method" not in sent, sent
    # review: the projected SP deck has both. Resources shows no check mark before you have been there
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.defaults && !s.defaultsBusy", 120)
    assert drv.execute_script("return !document.querySelector(\".wfb-tab[data-sec='3'] .wfb-num\").classList.contains('ok')")
    _click(drv, "[data-act='next']")
    H.wait(drv, "return window.__maceWfBuilder.state().sec === 3")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.defaults && !s.defaultsBusy", 120)
    _click(drv, "[data-act='next']")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.preview && !s.previewing", 300)
    p = drv.execute_script("return window.__maceWfBuilder.state().preview")
    assert p["code"] == 0, p.get("cli_log")
    sp = {s["step"]: s for s in p["projected"]}["SP_2"]["deck_text"]
    assert "POB-DZVP-REV2" in sp and "8 8 8 9 24" in sp
    assert not H.errors(drv), H.errors(drv)
