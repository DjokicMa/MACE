"""The Workflows document (Phase 3, read only) in headless Firefox and
Chromium at 100 % and 110 % zoom, with the logging fake SLURM: a CLI-written
workflow is listed, its grid shows every material's steps in plain words, a
running step its time left; a step opens its output; a job that ends updates
the grid by itself (workflow.changed on the event stream, no Refresh).

MACE_WF_SHOTS=<dir> saves screenshots at 110 % there."""
import os
import shutil
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U
import submit_util as SU
from test_workflow_api import OLD, TEST_DATA, WID, _db, _plan, _set_status

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")

CASES = [("firefox", 1.0), ("firefox", 1.1), ("chromium", 1.0), ("chromium", 1.1)]


@pytest.fixture(params=CASES, ids=[f"{b}-{int(z * 100)}" for b, z in CASES])
def app(request, tmp_path):
    browser, zoom = request.param
    src = TEST_DATA / "OPT" / "1_dia_opt_rev1.out"
    if not src.exists():
        pytest.skip("test/ data corpus not present")
    from fake_slurm import FakeSlurm
    root = tmp_path / "proj"
    wf = root / "perov"
    run = wf / "workflow_outputs" / WID / "step_001_OPT" / "A"
    run.mkdir(parents=True)
    lines = src.read_text(errors="replace").splitlines(True)
    heads = [i for i, l in enumerate(lines) if "OPTIMIZATION - POINT" in l]
    (run / "A.out").write_text("".join(lines[:heads[-1] + 3]))
    shutil.copy(src.with_suffix(".d12"), run / "A.d12")
    (run / "A.sh").write_text("#!/bin/bash\n#SBATCH --ntasks=32\n")
    _plan(wf, WID, ["OPT", "SP", "BAND", "DOSS"], ["A", "B", "C"], label="perovskites")
    db = _db(wf, WID, [
        {"id": "a1", "mat": "A", "type": "OPT", "status": "running", "job": "5101", "input": str(run / "A.d12")},
        {"id": "b1", "mat": "B", "type": "OPT", "status": "completed", "completed": "2026-10-15T10:30:00"},
        {"id": "b2", "mat": "B", "type": "SP", "status": "submitted", "job": "5102"},
        {"id": "c1", "mat": "C", "type": "OPT", "status": "failed", "job": "5099"},
    ])
    _plan(root, OLD, ["OPT", "SP"], ["X"])
    fake = FakeSlurm(tmp_path / "fs", {"sacct": {"5101": "COMPLETED"}})
    fake.add_record(id="5101", name="A", state="RUNNING", user=SU.USER, workdir=str(run), elapsed="1:00:00",
                    limit="4:00:00")
    fake.add_record(id="5102", name="B", state="PENDING", user=SU.USER, workdir=str(wf), reason="Priority")
    env = SU.fake_env(fake, tmp_path)
    env.pop("XDG_CACHE_HOME")
    env.update(MACE_GUI_TEST_HOOKS="1", MACE_GUI_NO_WARMUP="1", MACE_SITE_CONFIG="none")
    proc, base, _ = H.start_server(root, tmp_path / "xdg", **env)
    drv = None
    try:
        if browser == "firefox":
            drv = H.firefox(1440, 900, prefs={"layout.css.devPixelsPerPx": str(zoom)})
        else:
            drv = H.chromium(1440, 900, args=(f"--force-device-scale-factor={zoom}",))
        H.load(drv, base)
        yield {"drv": drv, "base": base, "root": root, "db": db, "fake": fake, "zoom": zoom, "browser": browser}
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
        app["drv"].save_screenshot(str(Path(out) / f"{app['browser']}-110-{name}.png"))


def _click(drv, css, timeout=15):
    el = H.wait(drv, f"return document.querySelector({css!r})", timeout)
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
    el.click()
    return el


def _cell(drv, mid, step):
    return drv.execute_script(
        "const b = document.querySelector(`.wfv-node[data-mid='${arguments[0]}'][data-step='${arguments[1]}']`);"
        "return b ? b.innerText.replace(/\\s+/g, ' ').trim() : null", mid, step)


def test_workflows_view(app):
    drv = app["drv"]
    try:
        drv.execute_script("window.__mace.run('view.workflows')")
        H.wait(drv, "return document.querySelectorAll('#workflow-host .wfv-item').length === 2")
        first = drv.execute_script("return document.querySelector('.wfv-item').innerText")
        assert "perovskites" in first and "In progress" in first and "OPT → SP → BAND → DOSS" in first
        H.wait(drv, "return document.querySelector('.wfv-grid')")
        assert "Running" in _cell(drv, "A", "OPT_1") and "left" in _cell(drv, "A", "OPT_1")
        assert _cell(drv, "B", "OPT_1").startswith("Completed") and _cell(drv, "B", "SP_2").startswith("Queued")
        assert _cell(drv, "C", "OPT_1").startswith("Failed") and _cell(drv, "C", "SP_2").startswith("Blocked")
        assert _cell(drv, "A", "SP_2").startswith("Not started")
        head = drv.execute_script("return document.querySelector('.wfv-title').innerText")
        assert "perovskites" in head and "In progress" in head
        # nothing scrolls sideways at this zoom, and the grid shows every material's row
        assert drv.execute_script("const v = document.querySelector('.wfv'); return v.scrollWidth <= v.clientWidth + 1")
        assert drv.execute_script("const w = document.querySelector('.wfv-grid-wrap'); return w.clientHeight >= w.scrollHeight - 1")
        _shot(app, "1-grid")

        # a step: its job, progress and files
        _click(drv, ".wfv-node[data-mid='A'][data-step='OPT_1']")
        H.wait(drv, "return document.querySelector('.wfv-node-detail table')")
        det = drv.execute_script("return document.querySelector('.wfv-node-detail').innerText")
        assert "5101" in det and "Time left" in det and "Progress" in det and "Point" in det
        assert "Open Output" in det and "Open Input Deck" in det and "Open Job Script" in det and "Job Details" in det
        aria = drv.execute_script("return document.querySelector(\".wfv-node[data-mid='A'][data-step='OPT_1']\").getAttribute('aria-label')")
        assert aria.startswith("A, OPT (step 1): Running")
        _shot(app, "2-step")
        _click(drv, ".wfv-node-detail [data-open][data-kind='out']")
        H.wait(drv, "return window.__mace.docs().find(d => d.kind === 'text' && d.title === 'A.out') || null")

        # the legend
        drv.execute_script("window.__mace.run('view.workflows')")
        H.wait(drv, "return document.querySelector('#tab-workflows.active .wfv-grid')")
        _click(drv, ".wfv-legend summary")
        legend = drv.execute_script("return document.querySelector('.wfv-legend').innerText")
        assert "Not submitted" in legend and "Being recovered" in legend

        # the 1.1.4 plan: listed, nothing started
        _click(drv, f".wfv-item[data-key='::{OLD}']")
        H.wait(drv, f"return document.querySelector('.wfv-meta') && document.querySelector('.wfv-meta').innerText.includes('{OLD}')")
        title = drv.execute_script("return document.querySelector('.wfv-title').innerText")
        assert "Unnamed workflow" in title and "Not started" in title
        _click(drv, f".wfv-item[data-key='perov::{WID}']")
        H.wait(drv, "return document.querySelector(\".wfv-node[data-mid='A'][data-step='OPT_1']\")")

        # live: the OPT job ends and its callback records it; the next poll updates the grid
        app["fake"].remove_record("5101")
        _set_status(app["db"], "a1", "completed")
        time.sleep(15.2)                                   # past the shared reader's refresh floor
        U.req(app["base"], "api/jobs?scope=folder&refresh=1")
        H.wait(drv, "return (window.__maceWorkflowEvents || 0) >= 1", 30)
        H.wait(drv, "const b = document.querySelector(\".wfv-node[data-mid='A'][data-step='OPT_1']\");"
                    "return b && b.innerText.includes('Completed')", 30)
        assert _cell(drv, "A", "SP_2").startswith(("Starting", "Not submitted"))
        _shot(app, "3-live")
        assert not H.errors(drv), H.errors(drv)
    except BaseException:
        H.save_failure(drv, f"workflow-{app['browser']}-{app['zoom']}")
        raise
