"""Runs and files that are still being written, in the GUI (PLAN_analysis
C18-d, C18-e, C18-f): the run document follows new points, the chip says
what is known, polling slows but never stops until CRYSTAL's end marker,
Follow in the text view, a replaced output, a BAND.DAT written in two
halves, and Write decks refused while the run is going.

A real output copied from the corpus is written into the served folder by
this test. The page's poll intervals are scaled down by a test variable
(window.__maceLiveScale); everything under test is clicked for real. The
SLURM-aware "Running (job ...)" chip needs PLAN_submit's jobs.py and is
covered when that lands."""
import os
import shutil
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U
import runcorpus as RC

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
from selenium.webdriver.common.keys import Keys  # noqa: E402
from selenium.webdriver.support.ui import Select  # noqa: E402



def _cut_after_point(data: bytes, n: int) -> int:
    """The offset where point n+1 starts (points 1..n written whole); the end
    of the points when there is no point n+1."""
    k = data.find(f"OPTIMIZATION - POINT {n + 1:4d}".encode())
    if k < 0:
        return data.find(b" * OPT END")
    return data.rfind(b"\n", 0, data.rfind(b"\n", 0, k)) + 1      # the line of stars before it


def file_row(drv, rel):
    row = drv.find_element("css selector", f'#file-list .file[data-path="{rel}"]')
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", row)
    return row


def chip(drv):
    return drv.execute_script("const c = document.querySelector('#o-live-chip'); return c ? c.textContent : ''")


def live_polls(drv):
    return drv.execute_script("return performance.getEntriesByType('resource').filter((r) => r.name.includes('api/out/live')).length")


@pytest.fixture
def app(tmp_path):
    src = RC.need(RC.DIA_OPT)
    root = H.examples_copy(tmp_path)
    (root / "live").mkdir()
    data = src.read_bytes()
    body = data[: data.find(b" * OPT END")]                    # everything before the end of the run
    run = root / "live" / "run_opt.out"
    run.write_bytes(data[: _cut_after_point(data, 2)])
    a = U.start_app(tmp_path, root=root)
    try:
        yield a, run, data, body
    finally:
        try:
            U.stop_app(a)
        except Exception:
            pass


def test_live_run_follows_stalls_resumes_and_finishes(app):
    pytest.importorskip("ase", reason="the server reads and writes structures with ase")
    (drv, base, root, xdg, proc, _), run, data, body = app
    drv.execute_script("window.__maceLiveScale = 0.1")             # 15 s -> 1.5 s, 60 s -> 6 s
    file_row(drv, "live/run_opt.out").click()
    H.wait(drv, "return document.querySelectorAll('#o-energy circle.pt').length >= 2", 60)
    H.wait(drv, "return /Live · updated/.test(document.querySelector('#o-live-chip').textContent)", 15)
    n0 = drv.execute_script("return window.MACE.state.traj.frames.length")
    # select an earlier frame: it must stay selected while points arrive
    drv.execute_script("const f = document.querySelector('#o-frame'); f.value = 0; f.dispatchEvent(new Event('input'))")
    H.wait(drv, "return window.MACE.state.frame === 0", 10)
    with open(run, "ab") as fh:                                     # two more points
        fh.write(body[run.stat().st_size:_cut_after_point(data, 4)])
    H.wait(drv, "return window.MACE.state.traj.frames.length >= arguments[0] + 2", 20, n0)
    assert drv.execute_script("return window.MACE.state.frame") == 0
    assert drv.execute_script("return document.querySelectorAll('#o-energy circle.pt').length") >= n0 + 2
    # stall: Live window 1 minute (Settings, real clicks) and an old mtime -> "No end marker"; polling continues
    drv.execute_script("document.body.focus()")
    U.keys(drv, Keys.CONTROL, ",")
    H.wait(drv, "return document.querySelector('#settings') && document.querySelector('#settings').open")
    drv.find_element("css selector", "#settings [data-sec='files']").click()
    box = drv.find_element("css selector", "#set-files-liveMinutes")
    box.clear()
    box.send_keys("1", Keys.TAB)
    H.wait(drv, "return window.__mace.settings.get('files.liveMinutes') === 1", 10)
    drv.execute_async_script("window.__mace.settings.flush().then(arguments[0])")
    U.keys(drv, Keys.ESCAPE)
    old = time.time() - 600
    os.utime(run, (old, old))
    H.wait(drv, "return /No end marker; last write/.test(document.querySelector('#o-live-chip').textContent)", 30)
    p0 = live_polls(drv)
    time.sleep(7)
    assert live_polls(drv) > p0                                     # the slow poll goes on
    # resume: a point appended to the stalled run shows within one slow poll
    n1 = drv.execute_script("return window.MACE.state.traj.frames.length")
    end = data.rfind(b"\n", 0, data.find(b"EEEEEEEEEE TERMINATION")) + 1
    with open(run, "ab") as fh:                                     # OPT END and the final geometry
        fh.write(data[run.stat().st_size:end])
    H.wait(drv, "return window.MACE.state.traj.frames.length > arguments[0]", 20, n1)
    # finished: the end marker turns the chip to Finished and polling stops
    with open(run, "ab") as fh:
        fh.write(data[end:])
    H.wait(drv, "return document.querySelector('#o-live-chip').textContent === 'Finished'", 20)
    p1 = live_polls(drv)
    time.sleep(4)
    assert live_polls(drv) == p1
    U.no_errors(drv)


def test_text_view_follows_and_offers_reload_when_replaced(app):
    (drv, base, root, xdg, proc, _), run, data, body = app
    drv.execute_script("window.__maceLiveScale = 0.1")
    U.req(base, "api/files")
    drv.execute_async_script("Promise.resolve(window.__mace.openAsText(arguments[0])).then(() => arguments[1]())", "live/run_opt.out")
    H.wait(drv, "return window.__mace.textView && window.__mace.textView.follow && window.__mace.textView.follow()", 20)
    n0 = drv.execute_script("return window.__mace.textView.total()")
    with open(run, "ab") as fh:
        fh.write(b" APPENDED LINE ONE\n APPENDED LINE TWO\n")
    H.wait(drv, "return window.__mace.textView.total() >= arguments[0] + 2", 15, n0)
    assert drv.execute_script("return window.__mace.textView.lineAt(window.__mace.textView.total() - 1)") == " APPENDED LINE TWO"
    run.write_bytes(data[:2000])                                   # a new run truncates the output
    H.wait(drv, "return !document.querySelector('.tv-replaced').hidden", 15)
    assert "replaced" in drv.execute_script("return document.querySelector('.tv-replaced').textContent")
    drv.find_element("css selector", ".tv-replaced [data-act=reload]").click()
    H.wait(drv, "return document.querySelector('.tv-replaced').hidden", 10)
    U.no_errors(drv)


def test_band_dat_written_in_two_halves(tmp_path):
    band = RC.CORPUS / "BAND" / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band.BAND.DAT"
    RC.need(band)
    pytest.importorskip("plotly")
    root = H.examples_copy(tmp_path)
    data = band.read_bytes()
    target = root / "half.BAND.DAT"
    target.write_bytes(data[: len(data) // 2])
    a = U.start_app(tmp_path, root=root)
    try:
        drv = a[0]
        file_row(drv, "half.BAND.DAT").click()
        H.wait(drv, "return /still being written/.test((document.querySelector('.pv-note') || {}).textContent || '')", 30)
        assert not drv.execute_script("return [...document.querySelectorAll('#toasts .toast2.error')].length")
        target.write_bytes(data)
        old = time.time() - 120
        os.utime(target, (old, old))
        drv.find_element("css selector", ".pv-note [data-act=retry-read]").click()
        H.wait(drv, "return window.__mace.plotView && window.__mace.plotView.state() && window.__mace.plotView.state().drawn", 30)
        U.no_errors(drv)
    finally:
        U.stop_app(a)
    # a stable truncated file keeps its error (not "still being written")
    root2 = H.examples_copy(tmp_path / "b")
    t2 = root2 / "cut.BAND.DAT"
    t2.write_bytes(data[: len(data) // 2])
    old = time.time() - 120
    os.utime(t2, (old, old))
    proc, base, _ = H.start_server(root2, tmp_path / "b" / "xdg", MACE_GUI_NO_WARMUP="1")
    try:
        st, _, out = U.req(base, "api/bands?path=cut.BAND.DAT")
        assert st != 409 and not (isinstance(out, dict) and out.get("still_writing")), (st, out)
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def test_write_decks_refused_while_the_run_goes_on(tmp_path):
    pytest.importorskip("ase")
    root = H.examples_copy(tmp_path)
    src = root / "outputs"
    proc, base, _ = H.start_server(root, tmp_path / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1")
    try:
        old = time.time() - 3600
        f9 = src / "Si_sp.f9"
        f9.write_bytes(b"\0" * 4096)
        os.utime(f9, (old, old))
        st, _, r = U.post(base, "api/setup/run", {"source": {"path": "outputs/Si_sp.out"}, "calc_type": "BAND", "wait": 120})
        assert st == 200 and r.get("json_text"), r
        body = {"source": {"path": "outputs/Si_sp.out"}, "calc_type": "BAND", "json_text": r["json_text"],
                "overwrite": True, "write_decks": True}
        # an f9 written 2 s ago: the run is still copying it
        f9.write_bytes(b"\0" * 8192)
        st, _, w = U.post(base, "api/setup/save", body)
        assert st == 409 and "still going" in w["error"], (st, w)
        # a finished run whose scratch folder still holds a larger fort.9 (the job-end copy runs)
        os.utime(f9, (old, old))
        scratch = tmp_path / "scratch" / "Si_sp"
        scratch.mkdir(parents=True)
        (scratch / "fort.9").write_bytes(b"\0" * 65536)
        (src / ".Si_sp.scratch").write_text(str(scratch) + "\n")
        st, _, w = U.post(base, "api/setup/save", body)
        assert st == 409 and "still going" in w["error"], (st, w)
        # finished, copy complete: Write decks proceeds as before
        (src / ".Si_sp.scratch").unlink()
        st, _, w = U.post(base, "api/setup/save", body)
        assert st in (200, 202), (st, w)
    finally:
        proc.terminate()
        proc.wait(timeout=15)
