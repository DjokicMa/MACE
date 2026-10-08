"""Clean Geometry's method picker in headless Firefox (software GL; plan:
Clean tiers T8, A17, A19): the options and their hints, Recheck, the
unrelated-model note, symmetrize-first off by default, the charge prefill
and the required charge of a charged deck, the host-geometry warning, the
Run button (field changes start nothing; a second Run replaces the first and
the model is loaded once), Run Longer past 120 s, the readout and citation,
and closing the dialog cancels the job. The ML method is the fake worker
(no torch); xtb is a fake 6.6.1 until Recheck sees 6.7.1."""
import os
import sys
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
from selenium.webdriver.common.keys import Keys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preopt"))
import preopt_fixtures as F  # noqa: E402
import test_clean_tiers_api as A  # noqa: E402


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cleantiers_e2e")
    root = H.examples_copy(tmp)
    (root / "decks").mkdir()
    (root / "decks" / "charged.d12").write_text(A._deck(scf="CHARGED\n"))
    (root / "decks" / "defect.d12").write_text(A._deck(edits="SUPERCEL\n2 0 0\n0 2 0\n0 0 2\nATOMREMO\n1\n1\n"))
    xtbdir = tmp / "fakebin"
    A._fake_xtb(xtbdir, "6.6.1")
    py = F.make_fake_python(tmp / "venv")
    a = U.start_app(tmp, root=root, MACE_GUI_NO_WARMUP="1", MACE_MLIP_PYTHON=str(py),
                    FAKE_HELLO_LOG=str(tmp / "hello.log"), FAKE_MODE_FILE=str(tmp / "mode"),
                    PATH=f"{xtbdir}:{os.environ.get('PATH', '/usr/bin:/bin')}", XDG_DATA_HOME=str(tmp / "data"))
    try:
        yield a + (tmp, xtbdir)
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    return app


def dialog_open(drv):
    H.wait(drv, "return !!document.querySelector('dialog.builder-dialog[data-builder=\"clean\"][open]')", 10)
    H.wait(drv, "const s = document.querySelector('.builder-dialog [data-k=\"method\"]'); return s && s.options.length >= 5", 30)
    time.sleep(0.2)


def q(drv, k, prop="value"):
    return drv.execute_script(f"const e = document.querySelector('.builder-dialog [data-k=\"{k}\"]'); return e ? e.{prop} : null")


def set_field(drv, k, value):
    drv.execute_script(f"""const e = document.querySelector('.builder-dialog [data-k="{k}"]');
        e.value = arguments[0]; e.dispatchEvent(new Event('input', {{bubbles: true}})); e.dispatchEvent(new Event('change', {{bubbles: true}}));""", value)


def clean_posts(drv):
    return drv.execute_script("return performance.getEntriesByType('resource').filter((r) => /api\\/clean$/.test(r.name)).length")


def open_clean(drv, path=None):
    if path:
        U.open_doc(drv, path)
    drv.execute_script("window.__mace.run('build.clean')")
    dialog_open(drv)


def hellos(tmp):
    p = tmp / "hello.log"
    return len(p.read_text().splitlines()) if p.exists() else 0


def test_picker_states_hints_and_recheck(fresh):
    drv, tmp, xtbdir = fresh[0], fresh[6], fresh[7]
    open_clean(drv, "structures/NaCl.cif")
    opts = drv.execute_script("""return [...document.querySelector('.builder-dialog [data-k="method"]').options]
        .map((o) => ({v: o.value, t: o.textContent.trim(), d: o.disabled}))""")
    by = {o["v"]: o for o in opts}
    assert [o["v"] for o in opts] == ["springs", "gfnff", "mlip:acesuit-mace-mpa-0-medium", "mlip:chgnet-0.3.0", "auto"]
    assert by["springs"]["t"] == "Springs: rough, always available" and q(drv, "method") == "springs"
    assert by["gfnff"]["d"] and by["mlip:acesuit-mace-mpa-0-medium"]["d"] and not by["mlip:chgnet-0.3.0"]["d"]
    assert by["mlip:acesuit-mace-mpa-0-medium"]["t"].startswith("ML potential: MACE-MPA-0 (ACEsuit mace-torch)")
    hints = q(drv, "unavailable", "textContent")
    assert "Found xtb 6.6.1; GFN-FF for periodic cells needs 6.7+" in hints
    assert 'conda install -c conda-forge "xtb>=6.7"' in hints and "mace mlip fetch acesuit-mace-mpa-0-medium" in hints
    chips = drv.execute_script("return [...document.querySelectorAll('.builder-dialog .clean-cmd')].map((c) => c.textContent)")
    assert 'conda install -c conda-forge "xtb>=6.7"' in chips and "mace mlip fetch acesuit-mace-mpa-0-medium" in chips
    assert not any("," in c or " or " in c for c in chips), chips      # commands only, never the prose
    assert q(drv, "symfirst", "checked") is False
    assert q(drv, "springs-note", "hidden") is False
    U.shot(drv, "clean_tiers_picker")
    A._fake_xtb(xtbdir, "6.7.1")
    try:
        drv.execute_script("document.querySelector('.builder-dialog [data-k=\"recheck\"]').click()")
        H.wait(drv, """const o = [...document.querySelector('.builder-dialog [data-k="method"]').options].find((x) => x.value === 'gfnff');
            return o && !o.disabled && o.textContent.includes('6.7.1')""", 30)
    finally:
        A._fake_xtb(xtbdir, "6.6.1")
        drv.execute_script("window.MACE.api('clean/methods/recheck', {body: {}})")
    U.keys(drv, Keys.ESCAPE)


def test_ml_runs_only_on_run_and_loads_the_model_once(fresh):
    drv, tmp = fresh[0], fresh[6]
    open_clean(drv, "structures/MgO.cif")
    H.wait(drv, "const f = window.__mace.builderFrame(); return f && !f.goDisabled", 60)   # the springs preview
    n0 = hellos(tmp)
    set_field(drv, "method", "mlip:chgnet-0.3.0")
    assert q(drv, "unrelated", "hidden") is False
    assert "not part of, and unrelated to, this MACE tool" in q(drv, "unrelated", "textContent")
    assert q(drv, "limit") == "120" and q(drv, "limit", "max") == "600"
    assert "about 1% too large" in q(drv, "relax-note", "textContent")
    assert q(drv, "ml-charge-note", "hidden") is False and q(drv, "uhf-note", "hidden") is True
    drv.execute_script("performance.clearResourceTimings()")
    set_field(drv, "charge", "1")
    set_field(drv, "charge", "0")
    drv.execute_script("document.querySelector('.builder-dialog [data-k=\"adv\"]').open = true")
    set_field(drv, "fmax", "0.05")
    time.sleep(1.5)                                          # past the 400 ms debounce
    assert clean_posts(drv) == 0
    drv.execute_script("document.querySelector('.builder-dialog [data-k=\"run\"]').click()")
    H.wait(drv, "return performance.getEntriesByType('resource').filter((r) => /api\\/clean$/.test(r.name)).length === 1", 30)
    drv.execute_script("document.querySelector('.builder-dialog [data-k=\"run\"]').click()")   # replaces the first
    H.wait(drv, "const f = window.__mace.builderFrame(); return f && !f.goDisabled", 90)
    assert clean_posts(drv) == 2
    ro = drv.execute_script("return document.querySelector('.builder-dialog .bld-readout').textContent")
    font = drv.execute_script("""const p = document.querySelector('.builder-dialog [data-k="ran"]');
        return getComputedStyle(p.firstElementChild || p).fontFamily""")
    assert "mono" not in font.lower(), font                    # a sentence, not a number
    assert "Ran: ML potential: CHGNet 0.3.0" in ro and "Starting geometry only" in ro and "Copy citation" in ro, ro
    assert hellos(tmp) - n0 <= 1
    drv.execute_script("document.querySelector('.builder-dialog [data-k=\"copy-cite\"]').click()")
    copied = drv.execute_script("return document.querySelector('.builder-dialog').dataset.copied")
    assert copied.startswith("B. Deng et al.") and "Training data: MPtrj" in copied
    U.shot(drv, "clean_tiers_ml_result")
    rows = U.ed_state(drv)["rows"]
    drv.execute_script("document.querySelector('.builder-dialog .bld-go').click()")
    H.wait(drv, f"const e = window.__mace.editor.doc(); return e.rows.length === {len(rows) + 1} && e.pending === 0", 30)
    U.no_errors(drv)


def test_run_longer_goes_past_120_s_for_ml(fresh):
    drv = fresh[0]
    open_clean(drv, "structures/Si.cif")
    set_field(drv, "method", "mlip:chgnet-0.3.0")
    drv.execute_script("document.querySelector('.builder-dialog [data-k=\"longer-row\"]').hidden = false")
    drv.execute_script("document.querySelector('.builder-dialog [data-k=\"longer\"]').click()")
    assert q(drv, "limit") == "240"
    H.wait(drv, "const f = window.__mace.builderFrame(); return f && !f.goDisabled", 90)
    U.keys(drv, Keys.ESCAPE)


def test_charge_prefill_from_a_defect_document(fresh):
    drv = fresh[0]
    drv.execute_async_script("""const done = arguments[0];
        window.MACE.api('edit/from_structure', {body: {name: 'MgO_vac', cell: [[4.21,0,0],[0,4.21,0],[0,0,4.21]],
          numbers: [12,12,12,12,8,8,8], frac: [[0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0],[.5,.5,.5],[0,0,.5],[0,.5,0]],
          meta: {hints: {charge: 2}}}}).then((r) => window.__mace.openUntitled(r, 'MgO_vac')).then(() => done(1))""")
    H.wait(drv, "return window.__mace.viewport.displayed() > 0", 10)
    open_clean(drv)
    assert q(drv, "charge") == "2"
    U.keys(drv, Keys.ESCAPE)


def test_charged_deck_requires_the_charge(fresh):
    drv = fresh[0]
    open_clean(drv, "decks/charged.d12")
    assert q(drv, "charge") == ""
    set_field(drv, "method", "mlip:chgnet-0.3.0")
    H.wait(drv, "return /Enter the charge: the deck has CHARGED/.test(window.__mace.builderFrame().error)", 10)
    assert q(drv, "run", "disabled") is True
    U.shot(drv, "clean_tiers_charged_deck")
    set_field(drv, "charge", "1")
    assert q(drv, "run", "disabled") is False
    U.keys(drv, Keys.ESCAPE)


def test_host_geometry_warning(fresh):
    drv = fresh[0]
    open_clean(drv, "decks/defect.d12")
    w = q(drv, "host-warning", "textContent")
    assert w.startswith("This document is the host geometry of defect.d12 (SUPERCEL, ATOMREMO)"), w
    assert "bld-warn" in q(drv, "host-warning", "className")         # shown as a warning, like the readout's
    U.keys(drv, Keys.ESCAPE)


def test_closing_the_dialog_cancels_the_job(fresh):
    drv, base, tmp = fresh[0], fresh[1], fresh[6]
    open_clean(drv, "structures/LiF.cif")
    set_field(drv, "method", "mlip:chgnet-0.3.0")
    (tmp / "mode").write_text("hang")
    try:
        drv.execute_script("document.querySelector('.builder-dialog [data-k=\"run\"]').click()")
        H.wait(drv, "return /Cleaning/.test(document.querySelector('.builder-dialog .bld-readout').textContent)", 30)
        U.keys(drv, Keys.ESCAPE)
        end = time.time() + 30
        while time.time() < end:
            ts = [t for t in U.req(base, "api/tasks")[2]["tasks"] if t["kind"] == "clean"]
            if ts and ts[-1]["state"] in ("canceled", "done", "failed"):
                break
            time.sleep(0.2)
        assert ts[-1]["state"] == "canceled", ts[-1]
    finally:
        (tmp / "mode").write_text("")
