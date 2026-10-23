"""The relaxation time in the New Workflow dialog, clicked through (1.1.6): a diamond SP deck, the chain
SP → BAND → DOSS → TRANSPORT; TRANSPORT's "From deformation potentials" puts a TAU step before it (5 strained
SPs for a cubic crystal) whose options are edited and reset; Default takes the TAU step out again; without
BAND and DOSS the TAU form warns and its button puts them back; the preview projects the 5 strained decks."""
import shutil
import time

import pytest

import guiharness as H
import submit_util as SU

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")

DECK = H.REPO_ROOT / "test" / "SP" / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.d12"


@pytest.fixture
def app(tmp_path):
    if not DECK.exists():
        pytest.skip("test/ data corpus not present")
    from fake_slurm import FakeSlurm
    root = tmp_path / "proj"
    (root / "decks").mkdir(parents=True)
    shutil.copy(DECK, root / "decks" / DECK.name)
    fake = FakeSlurm(tmp_path / "fs")
    env = SU.fake_env(fake, tmp_path)
    env.pop("XDG_CACHE_HOME")
    env.update(MACE_GUI_TEST_HOOKS="1", MACE_GUI_NO_WARMUP="1", MACE_SITE_CONFIG="none")
    proc, base, _ = H.start_server(root, tmp_path / "xdg", **env)
    drv = H.firefox(1440, 900)
    try:
        H.load(drv, base)
        yield drv
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def _click(drv, css, timeout=60):
    from selenium.common.exceptions import StaleElementReferenceException
    for _ in range(20):
        el = H.wait(drv, f"const e = document.querySelector({css!r}); return e && !e.disabled ? e : null", timeout)
        try:
            drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
            el.click()
            return
        except StaleElementReferenceException:
            time.sleep(0.2)
    raise AssertionError(f"{css} kept being replaced")


def _set(drv, css, value):
    H.wait(drv, f"return document.querySelector({css!r})", 180)
    drv.execute_script(f"const e = document.querySelector({css!r}); e.value = arguments[0]; "
                       "e.dispatchEvent(new Event('change', {bubbles: true}))", value)


def _seq(drv):
    return drv.execute_script("return window.__maceWfBuilder.state().draft.sequence")


def _form(drv, k, cond="true"):
    """Wait for step k's form to have loaded and ``cond`` (JS on r, the form) to hold."""
    return H.wait(drv, f"const s = window.__maceWfBuilder.state(), r = s.forms[{k!r}]; "
                       f"return !s.formBusy[{k!r}] && r && ({cond}) ? r : null", 180)


def test_relaxation_time_from_deformation_potentials(app):
    drv = app
    drv.execute_script("window.__mace.run('explorer.workflow', {dir: 'decks'})")
    H.wait(drv, "const s = window.__maceWfBuilder && window.__maceWfBuilder.state(); return s && s.inputs && s.inputs.files.length === 1", 60)
    _click(drv, "[data-act='next']")
    H.wait(drv, "return window.__maceWfBuilder.state().sec === 1")
    for t in ("BAND", "DOSS", "TRANSPORT"):
        _set(drv, "#wfb-add", t)
        H.wait(drv, f"return window.__maceWfBuilder.state().draft.sequence.includes({t!r})")
    assert _seq(drv) == ["SP", "BAND", "DOSS", "TRANSPORT"]
    _click(drv, "[data-act='next']")
    _click(drv, "[data-form='TRANSPORT_4']")
    _set(drv, "select[data-relax='mode']", "auto")
    H.wait(drv, "return window.__maceWfBuilder.state().draft.sequence.includes('TAU')")
    # TAU goes right before TRANSPORT, whose settings move with it to its new key
    assert _seq(drv) == ["SP", "BAND", "DOSS", "TAU", "TRANSPORT"]
    assert drv.execute_script("return window.__maceWfBuilder.state().formStep") == "TRANSPORT_5"
    assert drv.execute_script("return window.__maceWfBuilder.draft().steps.TRANSPORT_5.relaxation_time") == "auto"
    H.wait(drv, "const e = document.querySelector(\"select[data-relax='mode']\"); return e && e.value === 'auto'", 180)

    # the TAU step's form: diamond is cubic, so x only: 4 strains and the unstrained run
    _click(drv, "[data-form='TAU_4']")
    r = _form(drv, "TAU_4")
    assert (r["mode"], r["n_runs"], r["axes"], r["parent"], r["transport"]) == ("tau", 5, ["x"], "SP_1", "TRANSPORT_5")
    assert "5 strained single points per structure" in drv.find_element("css selector", ".wfb-form[data-step='TAU_4']").text
    assert not drv.execute_script("return document.querySelector(\"[data-part='tau-masses']\")")
    _set(drv, "select[data-tau='carrier']", "hole")
    _form(drv, "TAU_4", "r.options.carrier === 'hole'")
    _set(drv, "input[data-tau='axes']", "x z")
    _form(drv, "TAU_4", "r.n_runs === 9")
    drv.execute_script("document.querySelector(\"input[data-tau='strict']\").click()")
    _form(drv, "TAU_4", "r.options.strict === true")
    assert drv.execute_script("return window.__maceWfBuilder.draft().steps.TAU_4.tau") == \
        {"carrier": "hole", "axes": ["x", "z"], "strict": True}
    _click(drv, "[data-tau-reset='axes']")
    _form(drv, "TAU_4", "r.n_runs === 5")
    assert drv.execute_script("return window.__maceWfBuilder.draft().steps.TAU_4.tau") == {"carrier": "hole", "strict": True}
    # an emptied number or list is the default again: never a fallback of 0 fs or no strains
    _set(drv, "input[data-tau='fallback_fs']", "7")
    _form(drv, "TAU_4", "r.options.fallback_fs === 7")
    _set(drv, "input[data-tau='fallback_fs']", "")
    _form(drv, "TAU_4", "r.options.fallback_fs === r.defaults.fallback_fs && r.options.fallback_fs > 0")
    _set(drv, "input[data-tau='strains']", "")
    _form(drv, "TAU_4", "r.n_runs === 5")
    assert drv.execute_script("return window.__maceWfBuilder.draft().steps.TAU_4.tau") == {"carrier": "hole", "strict": True}
    assert not drv.execute_script("return window.__maceWfBuilder.state().formError.TAU_4")
    # the chain shows the count of the loaded form
    drv.execute_script("window.__maceWfBuilder.go(1)")
    H.wait(drv, "return [...document.querySelectorAll('.wfb-chain .wfb-node')].some((n) => /5 strained SPs/.test(n.textContent))")

    # Default takes the TAU step out; TRANSPORT moves back to its key
    drv.execute_script("window.__maceWfBuilder.go(2)")
    _click(drv, "[data-form='TRANSPORT_5']")
    _set(drv, "select[data-relax='mode']", "default")
    H.wait(drv, "return !window.__maceWfBuilder.state().draft.sequence.includes('TAU')")
    assert _seq(drv) == ["SP", "BAND", "DOSS", "TRANSPORT"]
    assert "TAU_4" not in drv.execute_script("return window.__maceWfBuilder.draft().steps")
    # Fixed: a value, no TAU step
    _set(drv, "select[data-relax='mode']", "fixed")
    _set(drv, "input[data-relax='fs']", "25")
    H.wait(drv, "const s = window.__maceWfBuilder.draft().steps.TRANSPORT_4; return s && s.relaxation_time === 25")
    assert _seq(drv) == ["SP", "BAND", "DOSS", "TRANSPORT"]

    # from deformation potentials without BAND and DOSS: the warning, and its button puts them back
    _set(drv, "select[data-relax='mode']", "auto")
    H.wait(drv, "return window.__maceWfBuilder.state().draft.sequence.includes('TAU')")
    drv.execute_script("window.__maceWfBuilder.go(1)")
    for name in ("BAND", "DOSS"):
        _click(drv, f"[data-remove='{_seq(drv).index(name)}']")
        H.wait(drv, f"return !window.__maceWfBuilder.state().draft.sequence.includes({name!r})")
    assert _seq(drv) == ["SP", "TAU", "TRANSPORT"]
    _click(drv, "[data-act='next']")
    _click(drv, "[data-form='TAU_2']")
    H.wait(drv, "return document.querySelector(\"[data-part='tau-masses']\")", 180)
    note = drv.find_element("css selector", ".wfb-form[data-step='TAU_2'] .wfb-note2").text
    assert "BAND" not in note, note
    _click(drv, "[data-add-masses]")
    H.wait(drv, "const q = window.__maceWfBuilder.state().draft.sequence; return q.includes('BAND') && q.includes('DOSS')")
    assert _seq(drv) == ["SP", "BAND", "DOSS", "TAU", "TRANSPORT"]
    assert drv.execute_script("return window.__maceWfBuilder.draft().steps.TRANSPORT_5.relaxation_time") == "auto"

    # the preview: the spec has the relaxation time, the projection the 5 strained decks
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.defaults && !s.defaultsBusy", 180)
    drv.execute_script("window.__maceWfBuilder.go(4)")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.preview && !s.previewing", 600)
    p = drv.execute_script("return window.__maceWfBuilder.state().preview")
    assert p["code"] == 0, p.get("cli_log")
    taus = [x for x in p["projected"] if x["step"] == "TAU_4"]
    # 4 strained decks and the unstrained one
    assert len(taus) == 5 and sum("ELASTIC" in (x["deck_text"] or "") for x in taus) == 4, [x["deck_name"] for x in taus]
    assert '"relaxation_time": "auto"' in p["spec"]["text"]
    assert not H.errors(drv), H.errors(drv)
