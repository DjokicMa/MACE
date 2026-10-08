"""A run's Atoms, Coordination and Measurements panels in headless Firefox
(PLAN_analysis C5, C6): they show the frame on screen, read-only, under a
banner; the Timeline slider (real keys) moves them; Edit This Step gives the
same coordination; a distance measured on the run is tracked along it in
Run Analysis. A real cell optimization copied from the corpus (found by
property), a fresh profile, and the folder is unchanged at the end."""
import hashlib
import re
import time

import pytest

import guiharness as H
import p4util as U
import runcorpus as RC

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")
from selenium.webdriver.common.keys import Keys  # noqa: E402


def _cell_opt():
    """A converged cell OPT with 4 to 40 atoms, more than one element and a final frame."""
    for p in sorted((RC.CORPUS / "OPT").glob("*.out")):
        if p.stat().st_size > 1_500_000 or "MOLECULE" in p.name:
            continue
        t = p.read_text(errors="replace")
        m = re.search(r"ATOMS IN THE UNIT CELL:\s*(\d+)", t)
        zs = set(re.findall(r"^\s+\d+ [TF]\s+(\d+) ", t[m.end():m.end() + 20000], re.M)) if m else set()
        if m and 4 <= int(m.group(1)) <= 40 and len(zs) > 1 and "OPT END - CONVERGED" in t and "FINAL OPTIMIZED GEOMETRY" in t \
                and t.count("OPTIMIZATION - POINT") >= 3:
            return p
    pytest.skip("no cell OPT of 4-40 atoms in the corpus")


def _md5(d):
    return {p.relative_to(d).as_posix(): hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    RC.need(RC.DIA_OPT)
    tmp = tmp_path_factory.mktemp("runpanels")
    root = tmp / "root"
    src = _cell_opt()
    RC.copy_into(root / "runs", src)
    a = U.start_app(tmp, root=root)
    a[0].execute_script("window.__mace.settings.set('reduceMotion', 'on')")
    try:
        yield a + (src.name,)
    finally:
        U.stop_app(a)


def bottom(drv, k):
    drv.find_element("css selector", f"#bottom-tabs [data-bottom={k}]").click()
    H.wait(drv, "return document.querySelector(`.bottom-pane.${arguments[0]}`).classList.contains('active')", 5, k)


def banner(drv):
    return drv.execute_script("const b = document.querySelector('.bottom-pane.atoms .run-ro-t'); return b ? b.textContent : ''")


def first_rows(drv, n=3):
    return drv.execute_script(
        "return [...document.querySelectorAll('.bottom-pane.atoms .atoms-table tbody tr[data-i]')].slice(0, arguments[0]).map((r) => [...r.cells].map((c) => c.textContent))", n)


def click_file(drv, rel):
    """A real click on the Explorer row; the list is drawn again once after
    the page loads its file details, so a row found just before can be gone."""
    from selenium.common.exceptions import StaleElementReferenceException
    for attempt in range(5):
        try:
            drv.find_element("css selector", f'#file-list .file[data-path="{rel}"]').click()
            return
        except StaleElementReferenceException:
            time.sleep(0.5)
    raise AssertionError(f"{rel}: the Explorer row kept being redrawn")


def test_run_panels_follow_the_frame_read_only(app):
    drv, base, root, xdg, proc, downloads, name = app
    before = _md5(root)
    rel = f"runs/{name}"
    click_file(drv, rel)
    H.wait(drv, "const d = window.__mace.doc(); return d && d.kind === 'trajectory'", 60)
    st, _, summ = U.req(base, f"api/out/summary?path={rel}")
    n_frames = len(summ["labels"]) if "labels" in summ else summ.get("n_steps")
    # Atoms: the final frame, read-only, under the banner; never "Open a structure"
    bottom(drv, "atoms")
    H.wait(drv, "return /read-only/.test(document.querySelector('.bottom-pane.atoms .run-ro-t')?.textContent || '')", 30)
    assert banner(drv).startswith("Final geometry")
    pane = drv.find_element("css selector", ".bottom-pane.atoms").text
    assert "Open a structure" not in pane
    st, _, last = U.req(base, f"api/out/step?path={rel}&step={n_frames - 1}")
    n_atoms = len(last["numbers"])
    rows = drv.execute_script("return document.querySelectorAll('.bottom-pane.atoms .atoms-table tbody tr[data-i]').length")
    assert rows == n_atoms
    assert float(first_rows(drv, 1)[0][3]) == pytest.approx(last["frac"][0][0], abs=1e-5)
    # editing is refused with the reason, not done
    cell = drv.find_element("css selector", ".bottom-pane.atoms .atoms-table tbody tr[data-i='0'] td[data-c='3']")
    from selenium.webdriver.common.action_chains import ActionChains
    ActionChains(drv).double_click(cell).perform()
    H.wait(drv, "return [...document.querySelectorAll('#toasts .msg')].some((m) => /read-only. Use Edit This Step/.test(m.textContent))", 5)
    assert not drv.find_elements("css selector", ".bottom-pane.atoms input.at-edit")
    # the Timeline's slider, real keys: Home goes to frame 0 = POINT 1
    bottom(drv, "timeline")
    slider = drv.find_element("css selector", "#o-frame")
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", slider)
    slider.click()
    slider.send_keys(Keys.HOME)
    H.wait(drv, "return window.MACE.state.frame === 0", 10)
    bottom(drv, "atoms")
    H.wait(drv, "return /^Frame 0 of \\d+ · POINT 1/.test(document.querySelector('.bottom-pane.atoms .run-ro-t').textContent)", 10)
    assert banner(drv).startswith(f"Frame 0 of {n_frames - 1} · POINT 1")
    st, _, f0 = U.req(base, f"api/out/step?path={rel}&step=0")
    H.wait(drv, "const c = document.querySelector('.bottom-pane.atoms .atoms-table tbody tr[data-i=\"0\"] td[data-c=\"3\"]'); return c && Math.abs(Number(c.textContent) - arguments[0]) < 1e-5", 10, f0["frac"][0][0])
    # Frac/Cart on the banner
    drv.find_element("css selector", ".bottom-pane.atoms .run-ro [data-cm=cart]").click()
    H.wait(drv, "const c = document.querySelector('.bottom-pane.atoms .atoms-table tbody tr[data-i=\"0\"] td[data-c=\"3\"]'); return c && Math.abs(Number(c.textContent) - arguments[0]) < 1e-4", 10, f0["cart"][0][0])
    # Coordination on frame 0 equals Edit This Step's on the same frame
    bottom(drv, "coordination")
    H.wait(drv, "return document.querySelectorAll('.bottom-pane.coordination .coord-table tbody tr').length > 0", 20)
    run_rows = drv.execute_script("return [...document.querySelectorAll('.bottom-pane.coordination .coord-table tbody tr')].map((r) => [...r.cells].slice(0, 5).map((c) => c.textContent))")
    # Measurements: atoms 1 and 2 typed in, then Measure (a real click)
    bottom(drv, "measurements")
    box = drv.find_element("css selector", ".bottom-pane.measurements .run-ms-add")
    box.send_keys("1 2")
    drv.find_element("css selector", ".bottom-pane.measurements .run-ms-go").click()
    H.wait(drv, "return document.querySelectorAll('.bottom-pane.measurements .run-ms-table tbody tr').length === 1", 5)
    row = drv.find_element("css selector", ".bottom-pane.measurements .run-ms-table tbody tr").text
    assert "frame 0" in row and "Distance" in row
    # Edit This Step (the banner's button) opens frame 0 as a structure document
    bottom(drv, "atoms")
    drv.find_element("css selector", ".bottom-pane.atoms .run-ro-edit").click()
    H.wait(drv, "const d = window.__mace.doc(); return d && d.kind === 'structure'", 30)
    bottom(drv, "coordination")
    H.wait(drv, "return document.querySelectorAll('.bottom-pane.coordination .coord-table tbody tr').length > 0", 20)
    ed_rows = drv.execute_script("return [...document.querySelectorAll('.bottom-pane.coordination .coord-table tbody tr')].map((r) => [...r.cells].slice(0, 5).map((c) => c.textContent))")
    assert ed_rows[0][2] == run_rows[0][2] and ed_rows[0][4] == run_rows[0][4]      # CN and nearest distance of site 1
    assert _md5(root) == before
    U.no_errors(drv)


def test_track_a_measurement_along_the_run(app):
    """C6-a: one bond tracked; a value per frame; the final frame's equals the Measurements row there."""
    drv, base, root, xdg, proc, downloads, name = app
    rel = f"runs/{name}"
    drv.execute_script("window.__mace.closeAll()")
    click_file(drv, rel)
    H.wait(drv, "const d = window.__mace.doc(); return d && d.kind === 'trajectory'", 60)
    bottom(drv, "atoms")
    H.wait(drv, "return /^Final geometry/.test(document.querySelector('.bottom-pane.atoms .run-ro-t')?.textContent || '')", 30)
    # pick atoms 1 and 2 in the Atoms table (click, Ctrl+click), then Measure Selected
    r0 = drv.find_element("css selector", ".bottom-pane.atoms .atoms-table tbody tr[data-i='0'] td[data-c='1']")
    r1 = drv.find_element("css selector", ".bottom-pane.atoms .atoms-table tbody tr[data-i='1'] td[data-c='1']")
    from selenium.webdriver.common.action_chains import ActionChains
    r0.click()
    ActionChains(drv).key_down(Keys.CONTROL).click(r1).key_up(Keys.CONTROL).perform()
    bottom(drv, "measurements")
    btn = drv.find_element("css selector", ".bottom-pane.measurements .run-ms-sel")
    H.wait(drv, "return !document.querySelector('.bottom-pane.measurements .run-ms-sel').disabled", 5)
    assert btn.text.startswith("Measure Selected (2)")
    btn.click()
    H.wait(drv, "return document.querySelectorAll('.bottom-pane.measurements .run-ms-table tbody tr').length === 1", 5)
    final_val = float(drv.find_element("css selector", ".bottom-pane.measurements .run-ms-table tbody tr td.num").text.split()[0])
    drv.find_element("css selector", ".bottom-pane.measurements .run-ms-track").click()
    H.wait(drv, "const s = window.__mace.runView && window.__mace.runView.state(); return s && s.tab === 'tracked' && s.tracks.length && s.tracks[0].values", 60)
    vals = drv.execute_script("return window.__mace.runView.state().tracks[0].values")
    st, _, summ = U.req(base, f"api/out/summary?path={rel}")
    n_frames = len(summ["labels"]) if "labels" in summ else summ.get("n_steps")
    assert len(vals) == n_frames and all(v is not None for v in vals)
    assert vals[-1] == pytest.approx(final_val, abs=1e-3)       # the row shows 3 decimals
    U.no_errors(drv)
