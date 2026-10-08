"""Run Analysis, the run card and the shared plot options in headless Firefox
(PLAN_analysis C3, C4, C7). Real outputs copied from the corpus; real clicks
on the controls under test (plotly markers are clicked at their pixel
positions); a fresh Firefox profile per module with the clipboard granted
for the Copy as TSV paste."""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U
import runcorpus as RC

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402

HA = 27.211386245988
BAND = RC.CORPUS / "BAND" / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band.BAND.DAT"
SP = RC.CORPUS / "SP" / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out"


def _other_opt():
    """A real cell OPT with more atoms than 1_dia, other elements than carbon and a
    final frame (found by property)."""
    for p in sorted((RC.CORPUS / "OPT").glob("*.out")):
        if p.stat().st_size > 1_500_000 or "1_dia" in p.name or "MOLECULE" in p.name:
            continue
        t = p.read_text(errors="replace")
        m = re.search(r"ATOMS IN THE UNIT CELL:\s*(\d+)", t)
        zs = set(re.findall(r"^\s+\d+ [TF]\s+(\d+) ", t[m.end():m.end() + 20000], re.M)) if m else set()
        if m and 2 < int(m.group(1)) < 40 and zs - {"6"} and "OPT END - CONVERGED" in t and "FINAL OPTIMIZED GEOMETRY" in t:
            return p
    pytest.skip("no second OPT in the corpus")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    RC.need(RC.DIA_OPT)
    tmp = tmp_path_factory.mktemp("runplot")
    root = H.examples_copy(tmp)
    RC.copy_into(root / "runs", RC.DIA_OPT, RC.DIA_OPT.with_suffix(".d12"), BAND, SP, _other_opt())
    shutil.copy2(RC.DIA_OPT, root / "runs" / "dia_copy.out")
    RC.copy_into(root / "failed", *sorted((RC.CORPUS / "FAILED_QA").glob("*.out")))
    a = U.start_app(tmp, root=root)
    a[0].execute_script("window.__mace.settings.set('reduceMotion', 'on')")
    try:
        yield a
    finally:
        U.stop_app(a)


def file_row(drv, rel):
    row = drv.find_element("css selector", f'#file-list .file[data-path="{rel}"]')
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", row)
    return row


def open_run(drv, rel):
    drv.execute_script("window.__mace.closeAll()")
    from selenium.common.exceptions import StaleElementReferenceException
    for attempt in range(5):         # the list is drawn again once after the file details load
        try:
            file_row(drv, rel).click()
            break
        except StaleElementReferenceException:
            time.sleep(0.5)
    H.wait(drv, "return document.querySelectorAll('#o-energy circle.pt, #o-energy path.pt').length >= 2", 60)


def open_analysis(drv):
    # the run card lives in the Timeline panel: wait until it shows
    H.wait(drv, "const b = document.querySelector('#o-e-plot'); return b && b.offsetParent !== null", 30)
    b = drv.find_element("css selector", "#o-e-plot")
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", b)
    b.click()
    H.wait(drv, "return window.__mace.runView && window.__mace.runView.state() && window.__mace.runView.state().drawn", 60)


def tab(drv, name):
    drv.find_element("css selector", f"#runplot-host [data-tab={name}]").click()
    H.wait(drv, "const s = window.__mace.runView.state(); return s && s.drawn && s.drawn.endsWith('|' + arguments[0])", 30, name)


def traces(drv):
    return drv.execute_script("return window.__mace.runView.figure().traces")


def click_point(drv, meta, x):
    """A real click on the marker of trace `meta` at data x."""
    pos = drv.execute_script("""
        const el = window.__mace.runView.el(), fl = el._fullLayout;
        const t = el._fullData.find((d) => d.meta === arguments[0]); const i = t.x.indexOf(arguments[1]);
        const xa = fl[(t.xaxis || 'x').replace('x', 'xaxis')], ya = fl[(t.yaxis || 'y').replace('y', 'yaxis')];
        const y = t.type === 'bar' ? t.y[i] / 2 : t.y[i]; const r = el.getBoundingClientRect();
        const yy = ya.type === 'log' ? Math.log10(y) : y;
        return [xa._offset + xa.l2p(t.x[i]) - r.width / 2, ya._offset + ya.l2p(yy) - r.height / 2];""", meta, x)
    el = drv.find_element("css selector", "#runplot-host .rp-plot")
    ActionChains(drv).move_to_element_with_offset(el, int(pos[0]), int(pos[1])).pause(0.2).click().perform()


def test_convergence_tab_and_linking(app):
    drv = app[0]
    open_run(drv, "runs/1_dia_opt_rev1.out")
    open_analysis(drv)
    tab(drv, "convergence")
    tr = [t for t in traces(drv) if t["type"] == "scatter"]
    assert len(tr) == 4
    lay = drv.execute_script("return window.__mace.runView.figure().layout")
    thr = sorted(round(s["y0"], 6) for s in lay["shapes"] if str(s.get("name", "")).startswith("thr-"))
    assert thr == [0.0003, 0.00045, 0.0012, 0.0018]
    assert "internal a.u." in lay["yaxis"]["title"]["text"]
    assert lay["xaxis"].get("dtick") == 1                                # whole optimization points only
    # the criteria strip sits below the plot: the plot's own x axis is not covered by it
    box = drv.execute_script("""const p = document.querySelector('#runplot-host .rp-plot .main-svg').getBoundingClientRect();
        const b = document.querySelector('#runplot-host .rp-below').getBoundingClientRect(); return [p.bottom, b.top, b.height]""")
    assert box[2] > 0 and box[0] <= box[1] + 1, box
    assert all("from the .out" in a["text"] for a in lay["annotations"] if "(" in a["text"] and "TOLDEE" not in a["text"] and "restart" not in a["text"] and "met" not in a["text"].lower())
    mx = next(t for t in tr if t["meta"] == "max_grad0")
    assert mx["marker"]["symbol"][mx["x"].index(3)] == "circle"          # POINT 3: all four YES, filled
    md = next(t for t in tr if t["meta"] == "max_disp0")
    assert 1 not in md["x"]                                             # POINT 1 prints no displacement
    click_point(drv, "max_grad0", 2)
    H.wait(drv, "return window.MACE.state.frame === 1 && document.querySelector('#o-frame').value === '1'", 10)
    drv.execute_script("window.MACE.selectFrame('runs/1_dia_opt_rev1.out', 3)")
    tab(drv, "energy")
    H.wait(drv, "const s = window.__mace.runView.figure().layout.shapes.find((x) => x.name === 'selected'); return s && s.x0 === 4", 10)
    drv.execute_script("window.MACE.selectFrame('runs/1_dia_opt_rev1.out', 4)")
    H.wait(drv, "const s = window.__mace.runView.figure().layout.shapes.find((x) => x.name === 'selected'); return s && s.x0 === 4", 10)
    fin = [t for t in traces(drv) if str(t["meta"]).startswith("fin")]
    assert fin and fin[0]["x"] == [4]
    U.no_errors(drv)


def test_energy_log_view_segments_and_below_reference(app):
    drv = app[0]
    open_run(drv, "runs/1_dia_opt_rev1.out")
    open_analysis(drv)
    tab(drv, "energy")
    # Units: Ha (the Units menu), then Log |dE| (Log Y)
    drv.find_element("css selector", "#runplot-host .pk-units").click()
    drv.find_element("css selector", "#runplot-host .pk-menu [data-v='Ha']").click()
    if not drv.execute_script("return window.__mace.runView.state().opts.energy.ylog"):
        drv.find_element("css selector", "#runplot-host .pk-logy").click()
    H.wait(drv, "return window.__mace.runView.state().opts.energy.ylog && window.__mace.runView.figure().traces.some((t) => t.meta === 'de0')", 10)
    tr = {t["meta"]: t for t in traces(drv)}
    de = tr["de0"]
    assert de["x"] == [2, 3, 4]
    assert [round(v, 9) for v in de["y"]] == [4.549e-5, 1.012e-5, 2.783e-5]
    assert de["marker"]["symbol"][-1] == "triangle-up"                 # the energy rose across the restart
    seg = tr["seg0"]
    assert seg["x"] == [1, 2] and [round(v, 10) for v in seg["y"]] == [round(5.5608004e-5, 10), round(1.0118060e-5, 10)]
    lay = drv.execute_script("return window.__mace.runView.figure().layout")
    td = [s for s in lay["shapes"] if s.get("name") == "toldee"][0]
    assert abs(td["y0"] - 1e-7) < 1e-12
    fr = [s for s in lay["shapes"] if s.get("name") == "finalrun"][0]
    assert 3 < fr["x0"] < 4
    # linear, relative to the final energy, one reference: the points below it get the red marker and the chip
    drv.find_element("css selector", "#runplot-host .pk-logy").click()
    drv.find_element("css selector", "#runplot-host .pk-more").click()
    from selenium.webdriver.support.ui import Select
    Select(drv.find_element("css selector", "#runplot-host .pk-pop [data-k=relTo]")).select_by_value("final")
    drv.find_element("css selector", "#runplot-host .pk-pop [data-k=single]").click()
    U.keys(drv, Keys.ESCAPE)
    H.wait(drv, "return /below/.test(document.querySelector('#runplot-host .pv-chips').textContent)", 10)
    tr = {t["meta"]: t for t in traces(drv)}
    e = tr["e0_0"]
    below = [x for x, s in zip(e["x"], e["marker"]["symbol"]) if s == "triangle-down"]
    assert 3 in below
    chip = drv.execute_script("return document.querySelector('#runplot-host .rp-below-chip').textContent")
    assert chip == f"{len(below)} point{'s' if len(below) != 1 else ''} below {'its' if len(below) == 1 else 'their'} reference energy"
    U.no_errors(drv)


def test_scf_tab(app):
    drv = app[0]
    open_run(drv, "runs/1_dia_opt_rev1.out")
    open_analysis(drv)
    drv.execute_script("window.MACE.selectFrame('runs/1_dia_opt_rev1.out', 0)")
    tab(drv, "scf")
    tr = {t["meta"]: t for t in traces(drv)}
    assert tr["cycles"]["y"] == [6, 5, 5, 3]
    assert len(tr["detail"]["x"]) == 7
    ann = drv.execute_script("return window.__mace.runView.figure().layout.annotations.map((a) => a.text)")
    assert any("TOLDEE 1e-7 Ha (from the .out)" in a for a in ann), ann
    sel = drv.execute_script("return window.__mace.runView.figure().layout.shapes.find((x) => x.name === 'selected')")
    assert sel and sel["yref"] == "y domain"          # the point axis only, not across the SCF-cycle plot below
    click_point(drv, "cycles", 3)
    H.wait(drv, "return /printed only for the first point/.test(document.querySelector('#runplot-host .pv-note').textContent)", 10)
    assert "SCFOUT.LOG" in drv.execute_script("return document.querySelector('#runplot-host .pv-note').textContent")
    U.no_errors(drv)


def test_cell_tab_equals_the_parser(app):
    drv, base = app[0], app[1]
    open_run(drv, "runs/1_dia_opt_rev1.out")
    open_analysis(drv)
    tab(drv, "cell")
    vol = {t["meta"]: t for t in traces(drv)}["cell-volume"]
    st, _, conv = U.req(base, "api/out/convergence?path=runs/1_dia_opt_rev1.out")
    assert vol["y"] == [p["cell"]["volume"] for p in conv["points"]]


def test_convergence_tsv_equals_the_cli(app):
    drv, base, root = app[0], app[1], app[2]
    open_run(drv, "runs/1_dia_opt_rev1.out")
    open_analysis(drv)
    tab(drv, "convergence")
    t = drv.execute_script("return window.__mace.runView.table()")
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    r = subprocess.run([sys.executable, str(H.MACE_CLI), "--no-banner", "analyze", "--convergence",
                        str(root / "runs" / "1_dia_opt_rev1.out"), "--json"], capture_output=True, text=True, env=env, timeout=120)
    pts = json.loads(r.stdout)["points"]
    assert t["rows"] == [[p["point"], p["frame"], p["max_grad"], p["rms_grad"], p["max_disp"], p["rms_disp"]] for p in pts]


def test_compare_runs_per_atom(app):
    drv = app[0]
    other = _other_opt().name
    open_run(drv, "runs/1_dia_opt_rev1.out")
    drv.execute_script("document.body.focus()")
    # Analysis › Run › Compare Runs… (the menu, real clicks)
    drv.find_element("css selector", ".menu-top[data-menu='" + str(drv.execute_script(
        "return [...document.querySelectorAll('.menu-top')].findIndex((b) => b.textContent.trim() === 'Analysis')")) + "']").click()
    H.wait(drv, "return !!document.querySelector('.menu .menu-item[data-sub]')", 5)
    sub = [b for b in drv.find_elements("css selector", ".menu .menu-item[data-sub]") if b.text.strip().startswith("Run")][0]
    sub.click()
    H.wait(drv, "return !!document.querySelector(`.menu .menu-item[data-cmd-id='analysis.compareRuns']`)", 5)
    drv.find_element("css selector", ".menu .menu-item[data-cmd-id='analysis.compareRuns']").click()
    H.wait(drv, "return document.querySelector('#compare-runs') && document.querySelector('#compare-runs').open", 10)
    from selenium.webdriver.support.ui import Select
    Select(drv.find_element("css selector", "#cmp-run-pick")).select_by_value(f"runs/{other}")
    drv.find_element("css selector", "#compare-runs [data-v=ok]").click()
    H.wait(drv, "const s = window.__mace.runView.state(); return s && s.other && s.drawn", 60)
    s = drv.execute_script("return window.__mace.runView.state()")
    assert s["opts"]["energy"]["per"] == "atom"
    lay = drv.execute_script("return window.__mace.runView.figure().layout")
    assert "/atom" in lay["yaxis"]["title"]["text"]
    names = {t["name"] for t in traces(drv)}
    assert any("1_dia_opt_rev1.out" in n for n in names) and any(other in n for n in names)
    drv.find_element("css selector", "#runplot-host .pk-more").click()
    opt = drv.find_element("css selector", "#runplot-host .pk-pop [data-k=relTo] option[value=absolute]")
    assert opt.get_attribute("disabled") and "different compositions" in opt.get_attribute("title")
    U.keys(drv, Keys.ESCAPE)
    t = drv.execute_script("return window.__mace.runView.table()")
    assert len(t["columns"]) == 3                                   # point + one value column per run
    U.no_errors(drv)


def test_run_card_and_problems(app):
    drv = app[0]
    open_run(drv, "runs/1_dia_opt_rev1.out")
    H.wait(drv, "return /Converged: all four criteria met at point 4/.test((document.querySelector('#inspector .rc-stop') || {}).textContent || '')", 60)
    H.wait(drv, "return /Calculation/.test(document.querySelector('#inspector .rc-kv').textContent)", 60)
    drv.execute_script("window.__mace.closeAll()")
    p = next(x for x in sorted(Path(app[2], "failed").glob("*.out")) if "neighb" in x.name)
    file_row(drv, f"failed/{p.name}").click()
    H.wait(drv, "return /Geometry error \\(atoms too close\\)/.test(document.querySelector('.bottom-pane.problems') ? document.querySelector('.bottom-pane.problems').textContent : '')", 60)
    U.no_errors(drv)


def test_export_png_and_svg_download(app):
    """Export > PNG and SVG save files. They used to fetch() Plotly's data: URL,
    which the page's CSP (connect-src 'self') refuses: nothing was saved."""
    drv, downloads = app[0], app[5]
    open_run(drv, "runs/1_dia_opt_rev1.out")
    open_analysis(drv)
    for f in downloads.iterdir():
        f.unlink()
    for v in ("png", "svg"):
        drv.find_element("css selector", "#runplot-host .pk-export").click()
        drv.find_element("css selector", f"#runplot-host .pk-menu [data-v={v}]").click()
    end = time.time() + 20
    while time.time() < end and len([f for f in downloads.iterdir() if not f.name.endswith(".part")]) < 2:
        time.sleep(0.2)
    names = sorted(f.name for f in downloads.iterdir())
    assert any(n.endswith(".png") for n in names) and any(n.endswith(".svg") for n in names), names
    png = next(f for f in downloads.iterdir() if f.name.endswith(".png")).read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    U.no_errors(drv)


def test_copy_image_is_not_refused_by_the_page(app):
    """Copy Image to Clipboard turns the viewport into a Blob without fetch():
    a refusal may come from the browser's clipboard, never a NetworkError."""
    drv = app[0]
    drv.execute_script("window.__mace.closeAll()")
    file_row(drv, "runs/1_dia_opt_rev1.d12").click()
    H.wait(drv, "return !!(window.__mace.doc() && window.__mace.doc().kind === 'structure')", 30)
    drv.execute_script("document.querySelectorAll('#toasts .toast2').forEach((t) => t.remove())")
    drv.execute_async_script("const d = arguments[0]; window.__mace.copyImage().then(d, d)")
    msgs = drv.execute_script("return [...document.querySelectorAll('#toasts .msg')].map((m) => m.textContent)")
    assert msgs and not any("NetworkError" in m for m in msgs), msgs
    U.no_errors(drv)
