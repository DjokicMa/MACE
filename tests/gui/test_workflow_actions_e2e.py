"""The Workflows view's actions (Phase 5) in headless Firefox and Chromium at
100 % and 110 % zoom, with the logging fake SLURM and the real job callback:
a student builds OPT → SP for two decks in New Workflow and submits; one OPT
finishes (its completion step submits the SP) and the grid updates by itself
over the event stream; Pause and Resume; the other OPT runs out of memory and
is recovered from its step (preview, then resubmit); Cancel lists the exact
jobs it will scancel and cancels only those. Clone opens the builder
pre-filled. The account column shows what the job scripts get: on this
(non-MSU) cluster, where the student is not in MACE's built-in account, no
account line at all.

MACE_WF_SHOTS=<dir> saves screenshots at 110 % there."""
import json
import os
import re
import shutil
import sqlite3
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U
import submit_util as SU

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")

TEST_DATA = H.REPO_ROOT / "test"
DECK = TEST_DATA / "OPT" / "1_dia_opt_rev1.d12"
OUT = TEST_DATA / "OPT" / "1_dia_opt_rev1.out"
OOM = TEST_DATA / "FAILED_QA" / "3_dia3_opt_killed_signal9_oom.out"
CASES = [("firefox", 1.0), ("firefox", 1.1), ("chromium", 1.0), ("chromium", 1.1)]


@pytest.fixture(params=CASES, ids=[f"{b}-{int(z * 100)}" for b, z in CASES])
def app(request, tmp_path):
    browser, zoom = request.param
    if not (DECK.exists() and OOM.exists()):
        pytest.skip("test/ data corpus not present")
    from fake_slurm import FakeSlurm
    root = tmp_path / "proj"
    (root / "decks").mkdir(parents=True)
    for n in ("dia0", "dia1"):
        shutil.copy(DECK, root / "decks" / f"{n}.d12")
    # another university's cluster: the student's accounts do not include MACE's built-in one
    fake = FakeSlurm(tmp_path / "fs", {"assoc": ["physics"], "default_account": "physics"})
    env = SU.fake_env(fake, tmp_path)
    env.pop("XDG_CACHE_HOME")
    env.update(MACE_GUI_TEST_HOOKS="1", MACE_GUI_NO_WARMUP="1", MACE_SITE_CONFIG="none",
               MACE_HOSTNAME="login1.cluster.example.edu")
    proc, base, _ = H.start_server(root, tmp_path / "xdg", **env)
    cli_env = dict(SU.fake_env(fake, tmp_path), MACE_SITE_CONFIG="none", MACE_HOSTNAME="login1.cluster.example.edu",
                   XDG_CACHE_HOME=str(tmp_path / "xdg" / "cache"))
    drv = None
    try:
        if browser == "firefox":
            drv = H.firefox(1440, 900, prefs={"layout.css.devPixelsPerPx": str(zoom)})
        else:
            drv = H.chromium(1440, 900, args=(f"--force-device-scale-factor={zoom}",))
        H.load(drv, base)
        yield {"drv": drv, "base": base, "root": root, "fake": fake, "zoom": zoom, "browser": browser, "env": cli_env}
    finally:
        if drv is not None:
            drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def _shot(app, name):
    out = os.environ.get("MACE_WF_SHOTS")
    if out and app["zoom"] == 1.1:
        Path(out).mkdir(parents=True, exist_ok=True)
        time.sleep(0.3)
        app["drv"].save_screenshot(str(Path(out) / f"actions-{app['browser']}-110-{name}.png"))


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


def _cell(drv, mid, step):
    return drv.execute_script(f"const b = document.querySelector(\".wfv-node[data-mid='{mid}'][data-step='{step}']\");"
                              "return b ? b.innerText : ''")


def _wait_cell(drv, mid, step, text, timeout=60):
    H.wait(drv, f"const b = document.querySelector(\".wfv-node[data-mid='{mid}'][data-step='{step}']\");"
                f"return b && b.innerText.includes({text!r})", timeout)


def _view(drv, expr):
    return drv.execute_script(f"const v = window.__mace.workflowView(); const s = v && v.state(); return s ? ({expr}) : null")


def _confirm(drv, timeout=120):
    """The confirmation dialog's title, text and list; then Yes."""
    H.wait(drv, "return document.querySelector('#setup-confirm[open] [data-r=\"yes\"]')", timeout)
    txt = drv.execute_script("return document.querySelector('#setup-confirm').innerText")
    pre = drv.execute_script("const p = document.querySelector('#setup-confirm pre'); return p ? p.innerText : ''")
    return txt, pre


def _yes(drv):
    _click(drv, "#setup-confirm[open] [data-r='yes']")
    H.wait(drv, "return !document.querySelector('#setup-confirm[open]')", 30)


def _rows(db):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute("SELECT * FROM calculations ORDER BY created_at")]
    finally:
        con.close()


def _idle(drv, timeout=300):
    H.wait(drv, "const v = window.__mace.workflowView(); return v && !v.state().busy", timeout)


def test_actions_on_a_running_workflow(app):
    drv, fake, root = app["drv"], app["fake"], app["root"]
    from array_helpers import callback

    # ---- build and submit OPT → SP for two decks
    drv.execute_script("window.__mace.run('explorer.workflow', {dir: 'decks'})")
    H.wait(drv, "const s = window.__maceWfBuilder && window.__maceWfBuilder.state(); return s && s.inputs && s.inputs.files.length === 2", 60)
    _click(drv, "[data-act='next']")
    _click(drv, "[data-template='opt_sp']")
    H.wait(drv, "return document.querySelectorAll('.wfb-chain .wfb-node').length === 2")
    _click(drv, "[data-act='next']")
    _click(drv, "[data-act='next']")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.defaults && !s.defaultsBusy", 180)
    # the account: what the job scripts get here, which is no account line (never MACE's built-in MSU one)
    acct = H.wait(drv, "return document.querySelector(\"tr[data-step='OPT_1'] input[data-res='account']\")", 60)
    assert acct.get_attribute("value") == ""
    assert acct.get_attribute("placeholder") == "SLURM default"
    grid = drv.execute_script("return document.querySelector('.wfb-grid').innerText")
    assert "mendoza" not in grid and "no account line" in grid
    assert drv.execute_script("return document.querySelector('[data-part=\"account-warning\"]')") is None
    _shot(app, "0-account")
    _click(drv, "[data-act='next']")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.preview && !s.previewing", 300)
    p = drv.execute_script("return window.__maceWfBuilder.state().preview")
    assert p["code"] == 0, p.get("cli_log")
    assert all("-A " not in (f.get("script_text") or "") for f in p["first_step"])
    assert "SLURM's default account" in drv.execute_script("return document.querySelector('.wfb-sec').innerText")
    _click(drv, "[data-act='submit']")
    H.wait(drv, "return !document.querySelector('#wfb-dialog')", 300)
    assert len(fake.calls("sbatch")) == 2
    assert not any("#SBATCH -A" in json.dumps(c) for c in fake.calls("sbatch"))
    H.wait(drv, "return document.querySelector(\".wfv-node[data-mid='dia0'][data-step='OPT_1']\")", 60)
    st = _view(drv, "s.status")
    wid, folder = st["workflow_id"], st["folder_rel"]
    assert folder == "decks"
    db = root / "decks" / f".mace_context_{wid}" / "materials.db"
    # Phase 3's copyable command is gone; the buttons replace it
    assert drv.execute_script("return document.querySelector('.wfv-detail .wfv-cmd')") is None
    assert drv.execute_script("return [...document.querySelectorAll('.wfv-wf-actions button')].map(b => b.innerText.trim())") == \
        ["Pause", "Cancel Workflow…", "Clone…"]
    _shot(app, "1-submitted")

    # ---- dia0's OPT finishes: its completion step submits the SP; the grid updates over the event stream
    rows = {r["material_id"]: r for r in _rows(db)}
    opt0 = rows["dia0"]
    work = Path(opt0["input_file"]).parent
    shutil.copy(OUT, work / f"{Path(opt0['input_file']).stem}.out")
    exports = dict(re.findall(r'^export (MACE_WORKFLOW_ID|MACE_CONTEXT_DIR|MACE_ISOLATION_MODE)="([^"]*)"',
                              (work / f"{Path(opt0['input_file']).stem}.sh").read_text(), re.M))
    fake.set_record(opt0["slurm_job_id"], state="COMPLETED")
    events = drv.execute_script("return window.__maceWorkflowEvents || 0")
    cb = callback(app["env"], work, db, mode="completion", extra_env=dict(exports, SLURM_JOB_ID=opt0["slurm_job_id"]))
    assert cb.returncode == 0, cb.stdout[-3000:] + cb.stderr[-2000:]
    fake.remove_record(opt0["slurm_job_id"])
    assert len(fake.calls("sbatch")) == 3                      # the SP
    time.sleep(15.2)                                           # past the shared reader's refresh floor
    U.req(app["base"], "api/jobs?scope=folder&refresh=1")
    H.wait(drv, f"return (window.__maceWorkflowEvents || 0) > {events}", 60)
    _wait_cell(drv, "dia0", "OPT_1", "Completed")
    _wait_cell(drv, "dia0", "SP_2", "Queued")
    _shot(app, "2-live")

    # ---- pause, then resume
    _click(drv, "[data-wact='pause']")
    _idle(drv)
    H.wait(drv, "const v = window.__mace.workflowView().state(); return v.status && v.status.control.state === 'paused'", 60)
    assert "Paused." in drv.execute_script("return document.querySelector('.wfv-detail').innerText")
    labels = drv.execute_script("return [...document.querySelectorAll('.wfv-wf-actions button')].map(b => b.innerText.trim())")
    assert labels == ["Resume", "Cancel Workflow…", "Clone…"]
    _shot(app, "3-paused")
    _click(drv, "[data-wact='resume']")
    _idle(drv)
    H.wait(drv, "const v = window.__mace.workflowView().state(); return v.status && v.status.control.state === 'active'", 60)

    # ---- dia1's OPT runs out of memory: Recover from its step
    opt1 = rows["dia1"]
    work1 = Path(opt1["input_file"]).parent
    out1 = work1 / f"{Path(opt1['input_file']).stem}.out"
    shutil.copy(OOM, out1)
    fake.remove_record(opt1["slurm_job_id"])
    con = sqlite3.connect(db)
    with con:
        con.execute("UPDATE calculations SET status='failed', output_file=? WHERE calc_id=?", (str(out1), opt1["calc_id"]))
    con.close()
    _click(drv, ".wfv-refresh")
    _wait_cell(drv, "dia1", "OPT_1", "Failed")
    _click(drv, ".wfv-node[data-mid='dia1'][data-step='OPT_1']")
    _click(drv, "[data-wact='recover']")
    txt, pre = _confirm(drv)
    assert "Recover dia1" in txt and "--calc-id " + opt1["calc_id"] in txt and "--yes" in txt
    assert "keeps workflow " + wid in pre
    assert len(fake.calls("sbatch")) == 3                      # the preview submitted nothing
    _shot(app, "4-recover")
    _yes(drv)
    _idle(drv)
    assert len(fake.calls("sbatch")) == 4
    H.wait(drv, "const b = document.querySelector(\".wfv-node[data-mid='dia1'][data-step='OPT_1']\");"
                "return b && b.innerText.includes('×2')", 60)

    # ---- cancel: the dialog lists exactly the jobs it cancels
    live = sorted(r["slurm_job_id"] for r in _rows(db) if r["status"] in ("submitted", "queued", "running", "pending"))
    assert len(live) == 2
    _click(drv, "[data-wact='cancel']")
    txt, pre = _confirm(drv)
    assert "Cancel Workflow?" in txt and "Cancel Workflow and 2 Jobs" in txt
    assert sorted(re.findall(r"^(\d+)\s", pre, re.M)) == live
    assert not fake.calls("scancel")
    _shot(app, "5-cancel")
    _yes(drv)
    _idle(drv)
    (call,) = fake.calls("scancel")
    assert sorted(a for a in call["argv"] if a.isdigit()) == live
    H.wait(drv, "const v = window.__mace.workflowView().state(); return v.status && v.status.control.state === 'canceled'", 60)
    assert "Canceled." in drv.execute_script("return document.querySelector('.wfv-detail').innerText")
    labels = drv.execute_script("return [...document.querySelectorAll('.wfv-wf-actions button')].map(b => b.innerText.trim())")
    assert labels == ["Resume", "Clone…"]
    _shot(app, "6-canceled")

    # ---- clone: the builder, pre-filled with the same decks, steps and settings
    _click(drv, "[data-wact='clone']")
    H.wait(drv, "const s = window.__maceWfBuilder && window.__maceWfBuilder.state(); return s && s.inputs && s.inputs.files.length === 2", 60)
    s = drv.execute_script("const s = window.__maceWfBuilder.state(); return {seq: s.draft.sequence, label: s.draft.label, tmpl: s.draft.template}")
    assert s["seq"] == ["OPT", "SP"] and s["tmpl"] == "opt_sp" and s["label"].endswith("-copy")
    assert "A copy of an existing workflow" in drv.execute_script("return document.querySelector('[data-part=\"clone-note\"]').innerText")
    assert drv.execute_script("return document.querySelector('#wfb-title').innerText") == "New Workflow (a copy)"
    _shot(app, "7-clone")
    drv.execute_script("window.__maceWfBuilder.dialog.close()")
    assert len(fake.calls("sbatch")) == 4
    assert not H.errors(drv), H.errors(drv)


@pytest.mark.parametrize("slurm", ["no-accounting", "none"])
def test_builtin_account_off_msu(tmp_path, slurm):
    """Off MSU HPCC the account column shows what the job scripts get (1.1.5's
    rule, site.resolve_account): where SLURM is here but its accounting cannot
    be read, no account line (SLURM's default account) and no warning; where
    there is no SLURM at all (a workstation writing MSU HPCC's scripts), the
    built-in account is kept, and the builder says another cluster may refuse it."""
    if not DECK.exists():
        pytest.skip("test/ data corpus not present")
    from fake_slurm import FakeSlurm
    from mace.submission import site
    root = tmp_path / "proj"
    (root / "decks").mkdir(parents=True)
    shutil.copy(DECK, root / "decks" / "dia0.d12")
    fake = FakeSlurm(tmp_path / "fs")
    (fake.bindir / "sacctmgr").unlink()
    if slurm == "none":
        for tool in site.SLURM_CLIENTS:
            if (fake.bindir / tool).exists():
                (fake.bindir / tool).unlink()
    env = SU.fake_env(fake, tmp_path)
    env.pop("XDG_CACHE_HOME")
    env.update(MACE_GUI_TEST_HOOKS="1", MACE_GUI_NO_WARMUP="1", MACE_SITE_CONFIG="none",
               MACE_HOSTNAME="login1.cluster.example.edu")
    if slurm == "none" and any(site.slurm_here(t, env) for t in site.SLURM_CLIENTS):
        pytest.skip("SLURM client commands are on this computer's PATH")
    proc, base, _ = H.start_server(root, tmp_path / "xdg", **env)
    drv = None
    try:
        drv = H.chromium(1440, 900)
        H.load(drv, base)
        drv.execute_script("window.__mace.run('explorer.workflow', {dir: 'decks'})")
        H.wait(drv, "const s = window.__maceWfBuilder && window.__maceWfBuilder.state(); return s && s.inputs && s.inputs.files.length === 1", 60)
        drv.execute_script("window.__maceWfBuilder.go(3)")
        H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.defaults && !s.defaultsBusy", 180)
        # read in one go: the section is drawn again when an answer lands, so a found element may be gone
        value, placeholder = H.wait(drv, "const i = document.querySelector(\"tr[data-step='OPT_1'] input[data-res='account']\");"
                                         " return i ? [i.value, i.placeholder] : null", 60)
        if slurm == "no-accounting":
            assert value == "" and placeholder == "SLURM default"
            assert drv.execute_script("return document.querySelector('[data-part=\"account-warning\"]')") is None
        else:
            assert value                                       # what the job scripts get
            warn = H.wait(drv, "const w = document.querySelector('[data-part=\"account-warning\"]'); return w && w.innerText", 30)
            assert "may refuse it" in warn and value in warn
        assert not H.errors(drv), H.errors(drv)
    finally:
        if drv is not None:
            drv.quit()
        proc.terminate()
        proc.wait(timeout=15)
