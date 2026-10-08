"""Phase 3 budgets (spec §8.8) and the section 9 resource rows for the editor.

Opt-in like the Phase 1/2 bench: MACE_GUI_BENCH=1 nice -n 10 pytest -m guibench tests/gui/bench
(serially, never with -n). The rows go into the same bench_report.json as
test_bench_shell.py (same ROWS, same row()); this module rewrites the file
with every row when it finishes.

Fixtures are generated into tmp dirs with ase (MgO supercells and a
synthetic triclinic cell), never committed:
  mgo_5x5x5.cif (1000 atoms), mgo_10x10x10.cif (8000), mgo_11x11x11.cif
  (10,648, for the bond row) and mgo_slice_5000.cif (5000, a 5x5x25 cell:
  the editing limit). MACE_GUI_BENCH_BIG=1 adds mgo_22x22x22.cif (85,184)
  after checking MemAvailable >= 4 GB.

The browser renders in software (llvmpipe): the orbit frame rows are
information there. MACE_GUI_BENCH_GPU1=1 runs the orbit headful on GPU1
(PRIME offload) and gates it at 16.7 ms; GPU0 must stay untouched.
"""
import json
import math
import os
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import guiharness as H  # noqa: E402
import test_bench_shell as B  # noqa: E402  (the shared ROWS, row(), p95(), resource rows)

pytestmark = [pytest.mark.guibench,
              pytest.mark.skipif(os.environ.get("MACE_GUI_BENCH") != "1",
                                 reason="GUI benchmarks are opt-in: MACE_GUI_BENCH=1")]
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")

row, p95, ROWS = B.row, B.p95, B.ROWS


@pytest.fixture(scope="module")
def report(tmp_path_factory, request):
    out = Path(os.environ.get("MACE_GUI_BENCH_OUT") or tmp_path_factory.mktemp("bench3"))
    yield out
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench_report.json").write_text(json.dumps(
        {"phase": "1-5: every bench module that ran (rows are shared)", "written_by": "test_bench_editor", "load_average": [round(x, 2) for x in os.getloadavg()], "rows": list(ROWS.values()),
         # a test that stops on an exception records no failing row: count it too
         "tests_failed": request.session.testsfailed,
         "all_pass": all(r["pass"] for r in ROWS.values()) and not request.session.testsfailed},
        indent=2) + "\n")
    print(f"\nbench report: {out / 'bench_report.json'}")


def mgo(n):
    from ase.build import bulk
    return bulk("MgO", "rocksalt", a=4.212, cubic=True).repeat(n)


def write_fixtures(root: Path):
    import ase.io
    ase.io.write(root / "mgo_5x5x5.cif", mgo((5, 5, 5)))
    ase.io.write(root / "mgo_10x10x10.cif", mgo((10, 10, 10)))
    ase.io.write(root / "mgo_11x11x11.cif", mgo((11, 11, 11)))
    ase.io.write(root / "mgo_slice_5000.cif", mgo((5, 5, 25)))
    if os.environ.get("MACE_GUI_BENCH_BIG") == "1" and mem_available_gb() >= 4:
        ase.io.write(root / "mgo_22x22x22.cif", mgo((22, 22, 22)))


def mem_available_gb():
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bench3srv")
    root, xdg = H.examples_copy(tmp), tmp / "xdg"
    write_fixtures(root)
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_TEST_HOOKS="1")
    drv = H.firefox(1280, 800)
    drv.set_script_timeout(180)
    sampler = H.ResourceSampler().watch("server", proc.pid).watch("browser", drv.service.process.pid).start()
    try:
        H.load(drv, base)
        yield drv, base, root, proc
        B.resource_rows(sampler.stop())
    finally:
        sampler.stop_ev.set()
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def fresh(drv):
    drv.execute_script("window.__mace.closeAll(); window.__mace.settings.resetAll(); window.__mace.run('tool.select');"
                       "performance.clearResourceTimings(); performance.setResourceTimingBufferSize(100000);"
                       "performance.clearMeasures(); performance.clearMarks();")
    time.sleep(0.3)


def open_doc(drv, path, displayed=None, timeout=120):
    drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then(() => d())", path)
    cond = f"=== {displayed}" if displayed else "> 0"
    H.wait(drv, f"return window.__mace.viewport && window.__mace.viewport.displayed() {cond}", timeout)
    time.sleep(0.5)


def no_boundary(drv):
    drv.execute_script("window.__mace.settings.set('showBoundary', false)")
    time.sleep(0.3)


def wait_idle(drv, timeout=120):
    H.wait(drv, "const e = window.__mace.editor.doc(); return e && e.pending === 0 && !e.waiting", timeout)


def _status(pid, key):
    with open(f"/proc/{pid}/status") as fh:
        for line in fh:
            if line.startswith(key + ":"):
                return int(line.split()[1])
    return 0


def _rss_mb(pid, key="VmRSS"):
    return _status(pid, key) / 1024


def worker_cap():
    return B.worker_cap()


# --------------------------------------------------------------------------- rows

def test_bonds_worker_10k(report, app):
    """§8.8: bond finding in the worker on the cell-list grid, 10k atoms ≤40 ms."""
    import numpy as np
    from scipy.spatial import cKDTree
    drv = app[0]
    fresh(drv)
    no_boundary(drv)
    open_doc(drv, "mgo_11x11x11.cif", 10648)
    bench = "window.__mace.bench.editor()"
    drv.execute_async_script(f"const d = arguments[0]; {bench}.bonds(1).then(d)")                 # warm-up
    r = drv.execute_async_script(f"const d = arguments[0]; {bench}.bonds(5).then(d)")
    med = statistics.median(r["times"])
    row("bonds_worker_10k_ms", 40, round(med, 1), med <= 40, unit="ms")
    row("bonds_worker_10k_p95_ms", "info", round(p95(r["times"]), 1), True, unit="ms")
    row("bonds_worker_10k_in_worker_ms", "info", round(statistics.median(r["workerMs"]), 1), True, unit="ms")
    # the numpy oracle: the same rule (d < (ri + rj) * tol, metal-metal pairs only if asked)
    a = drv.execute_script(f"""const a = {bench}.bondInputs();
        return {{cart: Array.from(a.cart), radii: Array.from(a.radii), metal: Array.from(a.metal), tol: a.tol, mm: a.metalMetal}}""")
    cart = np.asarray(a["cart"]).reshape(-1, 3)
    radii, metal = np.asarray(a["radii"]), np.asarray(a["metal"], bool)
    pairs = np.array(sorted(cKDTree(cart).query_pairs(2 * radii.max() * a["tol"])))
    d = np.linalg.norm(cart[pairs[:, 0]] - cart[pairs[:, 1]], axis=1)
    keep = (d < (radii[pairs[:, 0]] + radii[pairs[:, 1]]) * a["tol"]) & (d > 0.1)
    if not a["mm"]:
        keep &= ~(metal[pairs[:, 0]] & metal[pairs[:, 1]])
    oracle = int(keep.sum())
    row("bonds_worker_10k_pairs_match_oracle", True, r["pairs"] == oracle, r["pairs"] == oracle)
    # Firefox has no Long Tasks API: the main thread only posts and receives here
    lt = drv.execute_script("return PerformanceObserver.supportedEntryTypes.includes('longtask')")
    row("long_tasks_api_available", "info", lt, True)
    assert r["pairs"] == oracle, (r["pairs"], oracle)


def test_lod_levels(report, app):
    drv = app[0]
    fresh(drv)
    no_boundary(drv)
    for path, n, want in (("mgo_5x5x5.cif", 1000, 1), ("mgo_10x10x10.cif", 8000, 2)):
        open_doc(drv, path, n)
        lod = drv.execute_script("return window.__mace.viewport.lod()")
        row(f"lod_level_{n}", want, lod, lod == want)
    drv.execute_script("window.__mace.activate(0); window.__mace.settings.set('quality', 'low')")
    time.sleep(0.8)
    lod = drv.execute_script("return window.__mace.viewport.lod()")
    row("lod_quality_low_1000", 1, lod, lod == 1)            # 1000 <= 2000 * 0.5: still full detail
    drv.execute_script("window.__mace.settings.set('quality', 'auto')")
    if (app[2] / "mgo_22x22x22.cif").exists():
        open_doc(drv, "mgo_22x22x22.cif", 85184, timeout=300)
        lod = drv.execute_script("return window.__mace.viewport.lod()")
        row("lod_level_85184", 3, lod, lod == 3)
        # §8.8 Phase 4: >= 30 fps at 85k; software GL here (info), gated in the GPU1 run
        drv.set_script_timeout(180)
        deltas = drv.execute_async_script("const d = arguments[0]; window.__mace.bench.editor().orbit(3000).then(d)")
        row("orbit_frame_p95_ms_85k", "info (software GL; GPU1 run: <= 33)", round(p95(deltas), 1), True, unit="ms")


def test_box_select_8000(report, app):
    drv = app[0]
    fresh(drv)
    no_boundary(drv)
    open_doc(drv, "mgo_10x10x10.cif", 8000)
    canvas = H.viewer_canvas(drv)
    w, h = drv.execute_script("const r = arguments[0].getBoundingClientRect(); return [r.width, r.height]", canvas)
    mains = []
    for k in range(5):
        drv.execute_script("window.__maceBox = null")
        H.drag(drv, canvas, [(w * 0.3, h * 0.3), (w * (0.55 + 0.02 * k), h * 0.6)])
        H.wait(drv, "return window.__maceBox !== null", 20)
        mains.append(drv.execute_script("return window.__maceBox"))
    main = statistics.median([m["mainMs"] for m in mains])
    row("box_select_main_ms_8000", 16, round(main, 2), main <= 16, unit="ms")
    row("box_select_worker_ms_8000", "info", round(statistics.median([m["workerMs"] for m in mains]), 1), True, unit="ms")
    row("box_select_hits_8000", "info", mains[-1]["hits"], mains[-1]["hits"] > 0)


def test_edit_to_render_1000(report, app):
    drv = app[0]
    fresh(drv)
    open_doc(drv, "mgo_5x5x5.cif")
    drv.execute_script("window.__mace.run('build.symModeP1')")
    wait_idle(drv)
    # the GL draw call inside the edit's paint, timed apart: in software GL (llvmpipe, every run but
    # the opt-in GPU1 one) rasterizing 1000 spheres alone takes 35-40 ms on a loaded workstation
    drv.execute_script("""const v = window.MACE.state.viewers['viewer-main']; window.__maceRaster = [];
        if (!v.__timed) { const o = v.paintNow.bind(v); v.paintNow = function () { const a = performance.now(); const r = o();
          window.__maceRaster.push(performance.now() - a); return r; }; v.__timed = true; }""")
    paint, confirm, raster = [], [], []
    for k in range(10):
        el = ["Ca", "Mg"][k % 2]
        t0 = drv.execute_script("return performance.now()")
        drv.execute_script("""performance.clearMeasures('ui:edit'); performance.mark('ui:edit:start');
            window.__mace.editor.E.api.edit(window.__mace.editor.doc(), 'replace', {indices: [0], element: arguments[0]})""", el)
        H.wait(drv, "return performance.getEntriesByName('ui:edit').length > 0", 10)
        paint.append(drv.execute_script("return performance.getEntriesByName('ui:edit')[0].duration"))
        raster.append(drv.execute_script("const r = window.__maceRaster; return r.length ? r[r.length - 1] : 0"))
        wait_idle(drv)
        confirm.append(drv.execute_script("return performance.now()") - t0)
    med = statistics.median(paint)
    gpu = os.environ.get("MACE_GUI_BENCH_GPU1") == "1"
    own = statistics.median([p - r for p, r in zip(paint, raster)])
    # §8.1 "edit to re-render <= 50 ms" is a GPU budget: gated whole on the GPU1 run; in software GL
    # the app's own part (model rebuild, panels; everything but the raster call) is gated instead
    row("edit_to_render_ms_1000", 50 if gpu else "info (software GL; GPU1 run: <= 50)", round(med, 1),
        med <= 50 if gpu else True, unit="ms")
    row("edit_to_render_app_ms_1000", 50, round(own, 1), own <= 50, unit="ms")
    row("edit_raster_ms_1000", "info (llvmpipe)" if not gpu else "info", round(statistics.median(raster), 1), True, unit="ms")
    row("edit_load_average", "info", round(os.getloadavg()[0], 2), True)
    row("edit_confirmed_ms_1000", "info", round(statistics.median(confirm), 1), True, unit="ms")


def _drag_move(drv, atom, seconds):
    """A scripted Move drag: a pointer move every 16 ms for `seconds`."""
    p = drv.execute_script("return window.__mace.viewport.atomScreen(arguments[0])", atom)
    canvas = H.viewer_canvas(drv)
    t = H._Ticks(drv)
    t.move(canvas, *H._offset(drv, canvas, p["x"], p["y"]))
    t.down(0)
    for k in range(1, int(seconds * 1000 / 16)):
        t.move(canvas, *H._offset(drv, canvas, p["x"] + 30 * math.sin(k / 10), p["y"] + 15 * math.cos(k / 13)), ms=16)
    t.up(0)
    t.perform()


def _symmetry_drag(drv, path, seconds=1.5):
    """A grab (G) of atom 5 followed for `seconds`, a pointer move every 16 ms,
    then a click: the same requests as a drag, whatever is in front of it."""
    from selenium.webdriver.common.action_chains import ActionChains
    fresh(drv)
    open_doc(drv, path)
    drv.execute_script("window.__mace.run('build.symModeP1'); window.__mace.editor.select([4], 'replace')")
    wait_idle(drv)
    canvas = H.viewer_canvas(drv)
    w, h = drv.execute_script("const r = arguments[0].getBoundingClientRect(); return [r.width, r.height]", canvas)
    cx, cy = w / 2, h / 2
    H.hover_at(drv, cx, cy)
    drv.execute_script("document.querySelector('#viewer-main').closest('.viewer-wrap').focus()")
    t0 = H.now_ms(drv)
    ActionChains(drv).send_keys("g").perform()
    t = H._Ticks(drv)
    for k in range(1, int(seconds * 1000 / 16)):
        t.move(canvas, *H._offset(drv, canvas, cx + 30 * math.sin(k / 10), cy + 15 * math.cos(k / 13)), ms=16)
    t.perform()
    H.click_at(drv, cx + 30 * math.sin(seconds * 1000 / 160), cy)
    wait_idle(drv, 300)
    time.sleep(0.5)
    reqs = sorted(H.requests_since(drv, t0, "/api/symmetry"), key=lambda r: r["start"])
    span = drv.execute_script("return performance.now()") - t0
    timing = drv.execute_script("""return performance.getEntriesByType('resource').filter((r) => r.name.includes('/api/symmetry'))
        .map((r) => (r.serverTiming || []).filter((t) => t.name === 'symm').map((t) => t.duration)[0]).filter((x) => x !== undefined)""")
    return reqs, span, timing


def test_symmetry_requests_during_drag(report, app):
    """§8.8: at most one /api/symmetry per 150 ms of a 1.5 s drag, never two
    at once; spglib's time from Server-Timing. The 150 ms server budget is
    for the unit cell (as the HPCC check); a 1000-atom supercell with its
    symmetry broken by the drag is recorded too: spglib needs seconds there."""
    drv = app[0]
    for path, tag in (("structures/MgO.cif", ""), ("mgo_5x5x5.cif", "_1000")):
        reqs, span, timing = _symmetry_drag(drv, path)
        overlap = any(b["start"] < a["end"] - 1 for a, b in zip(reqs, reqs[1:]))
        ok = 1 <= len(reqs) <= math.ceil(span / 150) and not overlap
        row(f"symmetry_requests_per_150ms{tag}", f"1..{math.ceil(span / 150)}, none overlapping", len(reqs), ok)
        if timing:
            med = statistics.median(timing)
            if tag:
                row(f"symmetry_server_ms{tag}_drag", "info (spglib on a broken supercell)", round(med, 1), True, unit="ms")
            else:
                row("symmetry_server_ms", 150, round(med, 1), med <= 150, unit="ms")


def test_symmetry_server_5000(report, app):
    import re
    from mace.gui import structures
    base, root = app[1], app[2]
    from ase.io import read
    a = read(root / "mgo_slice_5000.cif")
    body = json.dumps({"cell": a.cell.array.tolist(), "frac": a.get_scaled_positions().tolist(),
                       "numbers": a.numbers.tolist(), "symprec": 1e-3, "key": "bench5000"}).encode()
    symm, wait = [], []
    for k in range(6):
        req = urllib.request.Request(base + "api/symmetry", data=body, headers={
            "X-Requested-With": "mace-gui", "X-Mace-Token": H.TOKEN, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            st = r.headers.get("Server-Timing", "")
        if k == 0:
            continue                                                # warm-up
        symm.append(float(re.search(r"symm;dur=([\d.]+)", st).group(1)))
        wait.append(float(re.search(r"wait;dur=([\d.]+)", st).group(1)))
    del structures
    row("symmetry_server_ms_5000", "info", round(statistics.median(symm), 1), True, unit="ms")
    row("symmetry_wait_ms_5000", "info", round(statistics.median(wait), 1), True, unit="ms")


def test_server_compute_during_drag_5000(report, app):
    """§9: during a 3 s Move drag at the editing limit the server's spglib
    work stays inside the compute cap (threads and CPU cores)."""
    drv, base, root, proc = app
    fresh(drv)
    no_boundary(drv)
    open_doc(drv, "mgo_slice_5000.cif", 5000)
    drv.execute_script("window.__mace.run('build.symModeP1'); window.__mace.run('tool.move'); window.__mace.editor.select([4], 'replace')")
    wait_idle(drv)
    info = json.loads(urllib.request.urlopen(urllib.request.Request(
        base + "api/resources", headers={"X-Mace-Token": H.TOKEN, "X-Requested-With": "mace-gui"})).read())
    cap = info["workers_effective"]
    threads0 = _status(proc.pid, "Threads")
    peak, stop = [threads0], threading.Event()

    def sample():
        while not stop.wait(0.05):
            peak.append(_status(proc.pid, "Threads"))
    th = threading.Thread(target=sample, daemon=True)
    cpu0, t0 = B._cpu_seconds(proc.pid), time.time()
    th.start()
    _drag_move(drv, 4, 3.0)
    wall = time.time() - t0
    cores = (B._cpu_seconds(proc.pid) - cpu0) / wall
    stop.set()
    th.join()
    wait_idle(drv, 300)
    limit_threads = threads0 + cap + 2
    row("server_threads_peak_during_drag", f"<= {limit_threads}", max(peak), max(peak) <= limit_threads)
    row("server_cpu_cores_during_drag", f"<= {cap}", round(cores, 2), cores <= cap)


def test_rapid_nudges_undo(report, app):
    """30 nudges 16 ms apart on three atoms in turn, then 5 undos: the page
    and the server agree with the serial result."""
    drv, base = app[0], app[1]
    fresh(drv)
    open_doc(drv, "mgo_5x5x5.cif")
    drv.execute_script("window.__mace.run('build.symModeP1')")
    wait_idle(drv)
    f0 = drv.execute_script("return window.__mace.editor.doc().structure.frac.slice(0, 3).map((f) => f.slice())")
    drv.execute_async_script("""const done = arguments[0], E = window.__mace.editor.E, ed = window.__mace.editor.doc();
        let k = 0;
        const step = () => {
          E.api.select(ed, [k % 3], 'replace');
          E.api.edit(ed, 'move', {indices: [k % 3], frac_delta: [0.01, 0, 0]}, {merge: `nudge:${k % 3}`});
          if (++k < 30) setTimeout(step, 16); else done();
        };
        step();""")
    for _ in range(5):
        drv.execute_script("window.__mace.run('edit.undo')")
        time.sleep(0.016)
    wait_idle(drv)
    got = drv.execute_script("return window.__mace.editor.doc().structure.frac.slice(0, 3)")
    sid = drv.execute_script("return window.__mace.editor.doc().srv.session")
    srv = json.loads(urllib.request.urlopen(urllib.request.Request(
        base + f"api/edit/state?session={sid}", headers={"X-Mace-Token": H.TOKEN, "X-Requested-With": "mace-gui"})).read())
    # serial oracle: 30 single-atom steps (no two in a row share a merge key), 5 undone
    want = [list(f) for f in f0]
    for k in range(25):
        want[k % 3][0] += 0.01
    ok = all(abs(((g[0] - w[0]) + 0.5) % 1 - 0.5) < 1e-9 for g, w in zip(got, want)) and \
        all(abs(((s[0] - w[0]) + 0.5) % 1 - 0.5) < 1e-9 for s, w in zip(srv["frac"][:3], want))
    row("rapid_nudges_undo_ok", True, ok, ok)
    assert ok, (got, srv["frac"][:3], want)


def test_atoms_table_and_viewers(report, app):
    drv = app[0]
    fresh(drv)
    no_boundary(drv)
    open_doc(drv, "mgo_10x10x10.cif", 8000)
    drv.execute_script("document.querySelector('.bottom-tab[data-bottom=\"atoms\"]').click()")
    H.wait(drv, "return document.querySelectorAll('.atoms-table tbody tr').length > 0", 30)
    most = drv.execute_async_script("""const done = arguments[0], sc = document.querySelector('.at-scroll');
        let most = 0, top = 0;
        const step = () => {
          most = Math.max(most, document.querySelectorAll('.atoms-table tr').length);
          top += 1500; sc.scrollTop = top;
          if (top < sc.scrollHeight) requestAnimationFrame(() => requestAnimationFrame(step)); else done(most);
        };
        step();""")
    row("atoms_table_dom_rows_max_8000", 80, most, most <= 80)
    # hidden viewers: 20 edits in a structure document while a run's viewers are hidden
    fresh(drv)
    drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then(() => d())", "outputs/Ag2Br3_opt_unfinished.out")
    time.sleep(0.5)
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.symModeP1'); window.__paintsBy = {};"
                       "document.addEventListener('mace-painted', (e) => { window.__paintsBy[e.detail] = (window.__paintsBy[e.detail] || 0) + 1; })")
    for k in range(20):
        drv.execute_script("window.__mace.editor.E.api.edit(window.__mace.editor.doc(), 'move', {indices: [arguments[0]], frac_delta: [0.001, 0, 0]})", k % 8)
    wait_idle(drv)
    time.sleep(0.3)
    paints = drv.execute_script("return window.__paintsBy")
    hidden = sum(v for k, v in paints.items() if k != "viewer-main")
    row("hidden_viewer_paints", 0, hidden, hidden == 0)
    # a freed viewer comes back in ≤300 ms
    drv.execute_script("window.__maceViewerIdleMs = 300")
    drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then(() => d())", "materials.db")
    H.wait(drv, "return window.MACE.state.viewers['viewer-main'].freed === true", 10)
    drv.execute_script("window.__maceRebuildMs = null; window.__mace.activate(1)")
    H.wait(drv, "return window.__maceRebuildMs !== null && window.__maceRebuildMs !== undefined", 10)
    ms = drv.execute_script("return window.__maceRebuildMs")
    row("freed_viewer_rebuild_ms", 300, round(ms, 1), ms <= 300, unit="ms")
    drv.execute_script("window.__maceViewerIdleMs = 0")
    workers = drv.execute_script("const E = window.__mace.editor.E; return E && E.worker ? E.worker.started : 0")
    row("browser_workers", 1, workers, workers == 1)


def test_orbit_frames_5000(report, app):
    drv = app[0]
    fresh(drv)
    no_boundary(drv)
    open_doc(drv, "mgo_slice_5000.cif", 5000)
    deltas = drv.execute_async_script("const d = arguments[0]; window.__mace.bench.editor().orbit(3000).then(d)")
    v = p95(deltas)
    gpu = os.environ.get("MACE_GUI_BENCH_GPU1") == "1"
    row("orbit_frame_p95_ms_5000", 16.7 if gpu else "info (software GL)", round(v, 1), v <= 16.7 if gpu else True, unit="ms")
    row("orbit_frame_median_ms_5000", "info", round(statistics.median(deltas), 1), True, unit="ms")


@pytest.mark.skipif(os.environ.get("MACE_GUI_BENCH_GPU1") != "1", reason="the GPU1 orbit run is opt-in: MACE_GUI_BENCH_GPU1=1")
def test_orbit_frames_5000_gpu1(report, app):
    """§8.7/§9.3, once at sign-off: the orbit headful on GPU1 (PRIME offload),
    gated at 16.7 ms; GPU0 (the display's) must stay untouched."""
    from mace.gui.cli import browser_gpu_env
    if not os.environ.get("DISPLAY"):
        pytest.skip("needs the workstation's X display")
    used, _ = H.gpu_snapshot()
    total = {0: 0, 1: 0}
    import subprocess as sp
    out = sp.run(["nvidia-smi", "--query-gpu=index,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                 capture_output=True, text=True, timeout=10).stdout
    util = {}
    for ln in out.strip().splitlines():
        i, tot, u = [x.strip() for x in ln.split(",")]
        total[int(i)], util[int(i)] = int(tot), int(u)
    free1 = total.get(1, 0) - used.get(1, 0)
    row("gpu1_free_mib_before_orbit", ">= 1024", free1, True, unit="MiB")
    row("gpu1_utilization_before_orbit", "info", util.get(1), True, unit="%")
    if free1 < 1024:
        pytest.skip(f"GPU1 has {free1} MiB free (needs 1024)")
    env = {k: v for k, v in os.environ.items() if k not in ("MOZ_HEADLESS", "LIBGL_ALWAYS_SOFTWARE", "GALLIUM_DRIVER")}
    env.update(browser_gpu_env(1))
    gpu0_before = used.get(0, 0)
    drv = H.firefox(1280, 800, env=env)
    try:
        drv.set_script_timeout(180)
        H.load(drv, app[1])
        drv.execute_script("window.__mace.settings.set('showBoundary', false)")
        open_doc(drv, "mgo_slice_5000.cif", 5000)
        renderer = H.webgl_renderer(drv)
        deltas = drv.execute_async_script("const d = arguments[0]; window.__mace.bench.editor().orbit(3000).then(d)")
        big = None
        if (app[2] / "mgo_22x22x22.cif").exists():       # with MACE_GUI_BENCH_BIG=1: the 85k orbit (§8.8 Phase 4)
            open_doc(drv, "mgo_22x22x22.cif", 85184, timeout=300)
            big = drv.execute_async_script("const d = arguments[0]; window.__mace.bench.editor().orbit(3000).then(d)")
        after, procs = H.gpu_snapshot()
        ours = {drv.service.process.pid, *H.descendants(drv.service.process.pid)}
    finally:
        drv.quit()
    v = p95(deltas)
    row("orbit_renderer_gpu1", "NVIDIA", renderer, "NVIDIA" in renderer.upper() or "GEFORCE" in renderer.upper() or "RTX" in renderer.upper())
    row("orbit_frame_p95_ms_5000_gpu1", 16.7, round(v, 1), v <= 16.7, unit="ms")
    if big:
        vb = p95(big)
        row("orbit_frame_p95_ms_85k_gpu1", 33, round(vb, 1), vb <= 33, unit="ms")
    d0 = after.get(0, 0) - gpu0_before
    row("gpu0_memory_delta_mib_gpu1_run", 32, d0, d0 <= 32, unit="MiB")
    on0 = [p for g, p, _ in procs if g == 0 and p in ours]
    row("gpu0_processes_of_ours_gpu1_run", 0, len(on0), not on0)


def test_idle_with_dirty_document(report, app):
    """Phase 1/2 idle budgets still hold with a modified document open (drafts are local)."""
    drv = app[0]
    fresh(drv)
    drv.execute_script("window.__mace.settings.set('autosaveDrafts', 1)")
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.symModeP1');"
                       "window.__mace.editor.E.api.edit(window.__mace.editor.doc(), 'replace', {indices: [0], element: 'Ca'})")
    wait_idle(drv)
    time.sleep(1.5)
    drv.execute_script("""window.__idle = {raf: 0, req: performance.getEntriesByType('resource').length};
      const orig = window.requestAnimationFrame; window.__origRaf = orig;
      window.requestAnimationFrame = (f) => { window.__idle.raf++; return orig(f); };""")
    time.sleep(5)
    idle = drv.execute_script("""const r = {raf: window.__idle.raf,
        req: performance.getEntriesByType('resource').length - window.__idle.req};
      window.requestAnimationFrame = window.__origRaf; return r""")
    row("idle_animation_frames_5s_dirty", 0, idle["raf"], idle["raf"] == 0)
    row("idle_requests_5s_dirty", 0, idle["req"], idle["req"] == 0)
    drv.execute_script("window.__mace.settings.set('autosaveDrafts', 10); window.__mace.closeAll()")


def test_zz_server_rss_fill(report, tmp_path):
    """§9.2: the one memory budget. A server with the smallest cap (64 MB)
    gets its static cache filled and 16 sessions of the 5000-atom slice with
    18 edits each (288 history entries of about 126 KB: past the 32 MB history
    cap). Peak RSS stays within the warm baseline + the cap + 15 %, and the
    ledger never counts past the cap. Runs last: its server is its own."""
    import gzip  # noqa: F401
    from concurrent.futures import ThreadPoolExecutor
    import ase.io
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    ase.io.write(root / "mgo_slice_5000.cif", mgo((5, 5, 25)))
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_CACHE_MB="64", **({"MALLOC_ARENA_MAX": os.environ["BENCH_ARENAS"]} if os.environ.get("BENCH_ARENAS") else {}))
    hdr = {"X-Mace-Token": H.TOKEN, "X-Requested-With": "mace-gui"}

    def call(path, data=None, extra=None):
        h = dict(hdr, **(extra or {}))
        body = None
        if data is not None:
            body = json.dumps(data).encode()
            h["Content-Type"] = "application/json"
        with urllib.request.urlopen(urllib.request.Request(base + path, data=body, headers=h), timeout=300) as r:
            raw = r.read()
            return json.loads(raw) if r.headers.get("Content-Type", "").startswith("application/json") else raw
    try:
        info = call("api/info")
        cap_mb = info["resources"]["cache_mb"]
        # warm-up: the readers, ase, spglib and the working set of serving
        # 5000-atom edits from 4 threads at once (the fill's concurrency),
        # with nothing kept afterwards: that is the interpreter's, not a cache
        def warm(_):
            s = call("api/structure?path=mgo_slice_5000.cif")
            r = call("api/edit/open", {"path": "mgo_slice_5000.cif", "expected": s["file"], "locked": False})
            call("api/edit/op", {"session": r["session"], "base_rev": r["rev"], "op": "move",
                                 "args": {"indices": [0], "frac_delta": [0.001, 0, 0]}})
            call("api/edit/close", {"session": r["session"]})
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(warm, range(4)))
        time.sleep(1.0)
        base_rss = _rss_mb(proc.pid)
        row("server_rss_baseline_mb_fill", "info", round(base_rss, 1), True, unit="MB")
        # the static cache: every asset, gzip and identity
        static = H.REPO_ROOT / "mace" / "gui" / "static"
        for f in sorted(static.rglob("*")):
            if f.is_file():
                rel = f.relative_to(static).as_posix()
                for enc in ("gzip", "identity"):
                    try:
                        call(f"static/{info['static']}/{rel}", extra={"Accept-Encoding": enc})
                    except Exception:
                        pass
        ledger_max = [0.0]

        def session(k):
            s = call("api/structure?path=mgo_slice_5000.cif")
            r = call("api/edit/open", {"path": "mgo_slice_5000.cif", "expected": s["file"], "locked": False})
            sid, rev = r["session"], r["rev"]
            for e in range(18):
                r = call("api/edit/op", {"session": sid, "base_rev": rev, "op": "move",
                                         "args": {"indices": [e], "frac_delta": [0.001, 0, 0]}})
                if "rev" not in r:
                    break
                rev = r["rev"]
                st = call("api/resources")["stores"]
                ledger_max[0] = max(ledger_max[0], st["total_mb"])
            return True
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(session, range(16)))
        stores = call("api/resources")["stores"]
        peak = _rss_mb(proc.pid, "VmHWM")
        row("server_rss_after_fill_mb", "info", round(_rss_mb(proc.pid), 1), True, unit="MB")
        limit = base_rss + cap_mb * 1.15
        row("server_peak_rss_mb_fill", f"<= {round(limit)} (baseline {round(base_rss)} + {cap_mb} MB + 15%)",
            round(peak, 1), peak <= limit, unit="MB")
        total_max = max(ledger_max[0], stores["max_total_mb"])
        row("ledger_total_max_mb", cap_mb, total_max, total_max <= cap_mb, unit="MB")
        row("history_bytes_total_max_mb", cap_mb / 2, stores["max_history_mb"], stores["max_history_mb"] <= cap_mb / 2, unit="MB")
    finally:
        proc.terminate()
        proc.wait(timeout=15)
    failed = [r for r in ROWS.values() if not r["pass"]]
    assert not failed, json.dumps(failed, indent=1)
