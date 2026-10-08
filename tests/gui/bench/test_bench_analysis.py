"""Phase 5 budgets (plan T14, spec §8.8) and the §9 rows for Setup, the
database, windowed text, plots, FREQ modes, polyhedra and SVG export.

Opt-in like the other benches: MACE_GUI_BENCH=1 nice -n 10 pytest -m guibench tests/gui/bench
(serially, never with -n). The rows go into the same bench_report.json
(test_bench_shell's ROWS and row()); this module rewrites it at the end with
"phase": 5.

Fixtures are generated into tmp dirs and deleted at teardown, never committed:
  big10k.db: 10k materials (dbgen: MaterialDatabase's schema, made-up rows);
  big100k.db: 100k materials, 300k calculations, 1M properties, only with
    MACE_GUI_BENCH_BIG=1 and 2 GB free in /tmp (its size goes in the report);
  big_matdump.out: the Phase 4 34 MB output (test_outindex.make_big);
  MgO 8x8x8 (4096 atoms) and 10x10x10 (8000 atoms) CIFs written by ase;
  synthetic BAND.DAT (test_gui_results.write_band);
  the 2_dia2 FREQ output copied from test/FREQ when the corpus is present.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import guiharness as H  # noqa: E402
import p4util as U  # noqa: E402
import test_bench_shell as B  # noqa: E402  (the shared ROWS, row(), p95())

pytestmark = [pytest.mark.guibench,
              pytest.mark.skipif(os.environ.get("MACE_GUI_BENCH") != "1",
                                 reason="GUI benchmarks are opt-in: MACE_GUI_BENCH=1")]
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
pytest.importorskip("plotly")

ROWS, row = B.ROWS, B.row
BIG = os.environ.get("MACE_GUI_BENCH_BIG") == "1"
# the Phase 5 rows' server (with big10k.db); the shell bench's MACE_GUI_BENCH_TUNNEL serves the plain demo set,
# so its reload and boot sizes compare with the local column
TUNNEL = os.environ.get("MACE_GUI_BENCH_TUNNEL_DB") or os.environ.get("MACE_GUI_BENCH_TUNNEL")
FREQ = H.REPO_ROOT / "test" / "FREQ" / "2_dia2_opt_rev1_freq_B3LYP-D3-D3_optimized.out"


@pytest.fixture(scope="module")
def report(tmp_path_factory, request):
    out = Path(os.environ.get("MACE_GUI_BENCH_OUT") or tmp_path_factory.mktemp("bench5"))
    yield out
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench_report.json").write_text(json.dumps(
        {"phase": "1-5: every bench module that ran (rows are shared)", "written_by": "test_bench_analysis", "load_average": [round(x, 2) for x in os.getloadavg()], "rows": list(ROWS.values()), "tests_failed": request.session.testsfailed,
         "all_pass": all(r["pass"] for r in ROWS.values()) and not request.session.testsfailed}, indent=2) + "\n")
    print(f"\nbench report: {out / 'bench_report.json'}")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    from ase.io import read, write
    import dbgen
    import test_outindex as T
    from test_gui_results import write_band
    tmp = tmp_path_factory.mktemp("bench5app")
    root = H.examples_copy(tmp)
    dbgen.make(root / "big10k.db", 10000, calcs_per=3, props_per=10)
    if U.tmp_free_ok(2) and U.mem_ok(4):
        T.make_big(root / "big_matdump.out")
    (root / "results").mkdir()
    write_band(root / "results")
    (root / "big").mkdir()
    m = read(str(root / "structures" / "MgO.cif"))
    write(str(root / "big" / "MgO_888.cif"), m.repeat((8, 8, 8)))
    write(str(root / "big" / "MgO_1000.cif"), m.repeat((10, 10, 10)))
    if FREQ.is_file():
        (root / "freq").mkdir()
        shutil.copy2(FREQ, root / "freq" / FREQ.name)
    a = U.start_app(tmp, width=1600, height=1000, root=root)
    sampler = H.ResourceSampler().watch("server", a[4].pid).start()
    try:
        yield a + (sampler,)
    finally:
        sampler.stop_ev.set()
        U.stop_app(a)
        for p in (root / "big10k.db", root / "big100k.db", root / "big_matdump.out"):
            p.unlink(missing_ok=True)
        shutil.rmtree(root / "big", ignore_errors=True)


def _res(drv, pattern, since=0):
    return drv.execute_script("""return performance.getEntriesByType('resource').filter((r) => r.startTime >= arguments[1] && r.name.includes(arguments[0]))
        .map((r) => ({name: r.name, size: r.transferSize || r.encodedBodySize, start: r.startTime, end: r.responseEnd}))""", pattern, since)


def _measure(drv, name, start, end):
    return drv.execute_script("""try { performance.clearMeasures(arguments[0]); performance.measure(arguments[0], arguments[1], arguments[2]);
        const m = performance.getEntriesByName(arguments[0], 'measure'); return m.length ? m[m.length - 1].duration : null; } catch (e) { return null; }""", name, start, end)


def _db_rows(drv, base, path, suffix, column="local"):
    U.reset(drv, 1600, 1000)
    drv.execute_script("performance.clearMarks('db:open:start'); performance.clearMarks('db:rows:rendered')")
    t0 = drv.execute_script("return performance.now()")
    U.open_doc(drv, path, wait_atoms=False)
    H.wait(drv, "return document.querySelectorAll('#tab-db .dbv-tr[data-id]').length > 0", 60)
    first = _measure(drv, "db:first", "db:open:start", "db:rows:rendered")
    budget = 200 if column == "local" else 400
    row(f"db_first_page_ms_{suffix}", "<= 200 local / <= 400 tunnel", round(first, 1) if first else None, first is not None and first <= budget, column, unit="ms")
    pages = [r for r in _res(drv, "api/db/page", t0)]
    biggest = max((r["size"] for r in pages), default=None)
    row("db_page_bytes", "<= 30 KB", biggest, biggest is not None and biggest <= 30 * 1024, column, unit="B")
    time.sleep(0.5)
    s = drv.execute_async_script("const d = arguments[arguments.length - 1]; window.__mace.dbView.benchScroll(3000).then(d)")
    # one 60 Hz frame; the browser's rAF intervals jitter by a fraction of a ms around 16.67, a dropped frame is 33
    row(f"db_scroll_frame_p95_ms_{suffix}", "<= 16.7 (+0.5 rAF jitter)", round(s["p95"], 2), s["p95"] <= 16.7 + 0.5, column, unit="ms")
    row(f"db_dom_rows_max_{suffix}", "<= 80", s["maxRows"], s["maxRows"] <= 80, column)
    # the filter: formula LIKE on the whole table, timed after the debounce
    drv.execute_script("performance.clearMarks('db:filter:start'); performance.clearMarks('db:filter:end');"
                       "const q = document.querySelector('#tab-db .dbv-q'); q.value = 'Mg'; q.dispatchEvent(new Event('input', {bubbles: true}))")
    H.wait(drv, "return performance.getEntriesByName('db:filter:end').length > 0", 30)
    f = _measure(drv, "db:filter", "db:filter:start", "db:filter:end")
    row(f"db_filter_ms_{suffix}", "<= 200", round(f, 1) if f else None, f is not None and f <= (200 if column == "local" else 400), column, unit="ms")
    log = U.req(base, "api/test/dblog")[2]["log"]
    rows_q = [q for q in log if q.lstrip().upper().startswith("SELECT") and " FROM MATERIALS" in q.upper()
              and "COUNT(" not in q.upper() and "LIMIT" not in q.upper()]
    row("db_queries_paged", True, not rows_q, not rows_q, column)
    return first


def test_db_rows(report, app):
    drv, base, root = app[0], app[1], app[2]
    _db_rows(drv, base, "big10k.db", "10k")
    if BIG:
        if not U.tmp_free_ok(2):
            row("db_first_page_ms_100k", "<= 200", "skipped: /tmp free under 2 GB", True, unit="ms")
            return
        import dbgen
        p = dbgen.make(root / "big100k.db", 100000, calcs_per=3, props_per=10)
        try:
            row("big_materials_db_mb", "info", round(p.stat().st_size / 2**20, 1), True, unit="MB")
            drv.execute_async_script("const d = arguments[0]; Promise.resolve(window.MACE.loadFiles()).then(() => d())")
            time.sleep(0.5)
            _db_rows(drv, base, "big100k.db", "100k")
        finally:
            p.unlink(missing_ok=True)
    else:
        row("db_first_page_ms_100k", "<= 200", "opt-in: MACE_GUI_BENCH_BIG=1", True, unit="ms")


def test_text_window_rows(report, app):
    base, root = app[1], app[2]
    if not (root / "big_matdump.out").is_file():
        pytest.skip("the 34 MB output needs 2 GB free in /tmp and 4 GB of memory")
    t0 = time.perf_counter()
    st, h, r = U.req(base, "api/text?path=big_matdump.out&offset=0")
    ms = (time.perf_counter() - t0) * 1000
    n = len(json.dumps(r).encode())
    row("text_first_window_ms_34mb", "<= 300", round(ms, 1), st == 200 and ms <= 300, unit="ms")
    row("text_first_bytes", "<= 300 KB", n, n <= 300 * 1024, unit="B")


def test_plotly_lazy(report, app):
    drv, base = app[0], app[1]
    H.load(drv, base)
    _plotly_rows(drv, "local")


def _plotly_rows(drv, column):
    U.reset(drv, 1600, 1000)
    early = _res(drv, "plotly")
    U.open_doc(drv, "results/mgo_band.BAND.DAT", wait_atoms=False)
    H.wait(drv, "const v = window.__mace.plotView, s = v && v.state(); return s && s.drawn", 60)
    first = _res(drv, "plotly")
    size = sum(r["size"] for r in first)
    t1 = drv.execute_script("return performance.now()")
    U.open_doc(drv, "outputs/Si_sp.out", wait_atoms=False)
    U.open_doc(drv, "results/mgo_band.BAND.DAT", wait_atoms=False)
    time.sleep(0.5)
    again = sum(r["size"] for r in _res(drv, "plotly", t1))
    ok = not early and 0 < size <= 1.4 * 2**20 * 1.05 and again == 0
    row("plotly_lazy", "not before the first plot; <= 1.4 MB gz; 0 B again", {"before": len(early), "bytes": size, "second": again}, ok, column)


def _task_row(base, tid):
    return next((x for x in U.req(base, "api/tasks")[2]["tasks"] if x["id"] == tid), None)


def _task(base, body, route="api/setup/run"):
    st, _, r = U.post(base, route, body)
    if st == 200:
        return r, []
    tid = r["task"]["id"]
    seen = []
    while True:
        t = _task_row(base, tid) or {}
        msg = t.get("message")
        if msg and msg not in seen:
            seen.append(msg)
        st3, _, res = U.req(base, f"api/tasks/{tid}/result")
        if st3 == 200:
            return res, seen
        time.sleep(0.02)


def test_setup_rows(report, app):
    base = app[1]
    body = {"source": {"path": "structures/LiF.cif"}, "calc_type": "OPT", "preview": True, "wait": 0}
    t0 = time.perf_counter()
    r, stages = _task(base, body)
    ms = (time.perf_counter() - t0) * 1000
    row("deck_preview_ms", "report (about 300-600)", round(ms), bool(r.get("deck")), unit="ms")
    row("deck_preview_progress_events", ">= 2 stages", len(stages), len(stages) >= 2)
    t0 = time.perf_counter()
    st, _, r2 = U.post(base, "api/setup/defaults", {"source": {"path": "structures/LiF.cif"}, "calc_type": "OPT", "wait": 30})
    hit = (time.perf_counter() - t0) * 1000
    row("cli_defaults_cache_hit_ms", "<= 20", round(hit, 1), st == 200 and hit <= 20, unit="ms")
    # cancel: a fake slow CLI child (test hooks; the same process-group kill as a real one)
    st, _, r = U.post(base, "api/test/cli", {"seconds": 30, "doc": "bench-cancel"})
    tid = r["task"]["id"]
    end = time.time() + 10
    while time.time() < end and not (_task_row(base, tid) or {}).get("pid"):
        time.sleep(0.05)
    t0 = time.time()
    U.post(base, f"api/tasks/{tid}/cancel", {})
    while time.time() - t0 < 10:
        t = _task_row(base, tid) or {}
        if t.get("state") == "canceled":
            break
        time.sleep(0.05)
    took = time.time() - t0
    row("deck_preview_cancel_s", "<= 2.5", round(took, 2), took <= 2.5, unit="s")
    # a burst of 10 previews (other structures and calc types): workers stay within the cap
    cap = B.worker_cap()
    for k, (src, calc) in enumerate([(s, c) for s in ("MgO", "NaCl", "LiF", "Si", "diamond") for c in ("SP", "FREQ")]):
        U.post(base, "api/setup/run", {"source": {"path": f"structures/{src}.cif"}, "calc_type": calc, "preview": True, "wait": 0,
                                        "doc": f"burst{k}"})
    time.sleep(3)
    gate = U.req(base, "api/test/gate")[2]
    row("cpu_workers_used", f"<= {gate['cap']} (effective_workers)", gate["max_running"], gate["max_running"] <= gate["cap"] <= cap)


def test_freq_mode_frames(report, app):
    drv = app[0]
    if not FREQ.is_file():
        row("freq_mode_frame_ms", "report; <= 30 fps; 0 hidden", "skipped: test/ corpus not present", True)
        pytest.skip("test/ corpus not present")
    U.reset(drv, 1600, 1000)
    drv.execute_script("window.__mace.settings.set('reduceMotion', 'off')")
    drv.execute_async_script("const done = arguments[1]; window.__mace.openModes(arguments[0]).then(() => done(1))", f"freq/{FREQ.name}")
    H.wait(drv, "const v = window.__mace.freqView, s = v && v.state(); return s && s.mode", 60)
    H.wait(drv, "return window.__mace.freqView.playing()", 10)
    f0 = drv.execute_script("return window.__maceFreqFrames")
    time.sleep(2.0)
    fps = (drv.execute_script("return window.__maceFreqFrames") - f0) / 2.0
    drv.execute_script("Object.defineProperty(document, 'hidden', {configurable: true, get: () => true}); document.dispatchEvent(new Event('visibilitychange'))")
    time.sleep(0.3)
    g0 = drv.execute_script("return window.__maceFreqFrames")
    time.sleep(1.0)
    hidden = drv.execute_script("return window.__maceFreqFrames") - g0
    drv.execute_script("delete document.hidden; document.dispatchEvent(new Event('visibilitychange'))")
    row("freq_mode_fps", "<= 30", round(fps, 1), fps <= 31, unit="fps")
    row("freq_mode_frames_hidden", 0, hidden, hidden == 0)


CORE = "const core = await import(document.querySelector('script[src*=\"/js/shell.js\"]').src.replace('shell.js', 'geom/core.js'));"


def test_polyhedra_and_svg(report, app):
    drv = app[0]
    U.reset(drv, 1600, 1000)
    U.open_doc(drv, "big/MgO_888.cif")
    H.wait(drv, "return window.__mace.viewState().displayedAtoms > 4000", 60)
    ms = drv.execute_async_script("""const done = arguments[arguments.length - 1];
      (async () => {""" + CORE + """
        const W = await import(document.querySelector('script[src*="/js/shell.js"]').src.replace('shell.js', 'geom/client.js'));
        const s = window.__mace.doc().structure, d = window.__mace.viewport.scene().display;
        const args = () => ({ cell: s.cell, frac: Float64Array.from(s.frac.flat()), z: Uint8Array.from(s.symbols.map((x) => core.zOf(x))),
          radii: Float32Array.from(s.radii), metal: Uint8Array.from(s.metal.map(Number)), tol: 1.15, dispCart: d.cart.slice(), dispBase: Uint32Array.from(d.base) });
        await W.job("polyhedra", args());
        const t = [];
        for (let k = 0; k < 5; k++) t.push((await W.job("polyhedra", args())).ms);
        return t.sort((a, b) => a - b)[2];
      })().then(done, (e) => done(String(e)));""")
    row("polyhedra_4096_worker_ms", "<= 100", round(ms, 1) if isinstance(ms, (int, float)) else ms,
        isinstance(ms, (int, float)) and ms <= 100, unit="ms")
    # main-thread work to draw them, on a structure under the LOD limit (MgO 4x4x4 + boundary)
    drv.execute_script("window.__mace.settings.set('showPolyhedra', true)")
    try:
        U.open_doc(drv, "structures/MgO.cif")
        drv.execute_script("window.__mace.run('view.range3')")
        H.wait(drv, "const p = window.__mace.viewport.polyhedra(); return p && p.on && p.meshes === 1 && p.mainMs !== undefined", 30)
        main = drv.execute_script("return window.__mace.viewport.polyhedra().mainMs")
        row("polyhedra_main_thread_ms", "<= 8", round(main, 2), main <= 8, unit="ms")
    finally:
        drv.execute_script("window.__mace.settings.set('showPolyhedra', false); window.__mace.run('view.range1')")
    drv.execute_script("window.__mace.settings.set('showBoundary', false)")
    try:
        U.open_doc(drv, "big/MgO_1000.cif")
        H.wait(drv, "return window.__mace.viewState().displayedAtoms === 8000", 60)
        time.sleep(1.0)
        r = drv.execute_async_script("""const done = arguments[arguments.length - 1]; const t0 = performance.now();
          window.__maceExportHook = () => {};
          window.__mace.exportSvg({}).then((n) => done({ n, s: (performance.now() - t0) / 1000 }), (e) => done({ error: String(e) }));""")
        row("svg_export_8000_s", "<= 1.5", round(r.get("s", -1), 3), r.get("n") == 8000 and r["s"] <= 1.5, unit="s")
    finally:
        drv.execute_script("window.__mace.settings.set('showBoundary', true)")


def test_resource_rows_last(report, app):
    """Spec §9: the server's peak RSS (with the CLI children it ran), workers, GPUs."""
    proc, sampler = app[4], app[6]
    res = sampler.stop()
    rss = res["peak_rss_mb"].get("server")
    row("peak_rss_mb_p5", "<= 600", rss, (rss or 0) <= 600, unit="MB")
    cap = B.worker_cap()
    workers = res["max_children"].get("server", 0)
    # the server's children: the pool workers plus the CLI runs the gate admits (nice(1) keeps them one process each)
    row("cpu_workers_used_p5", f"<= {cap} pool + {cap} CLI", workers, workers <= 2 * cap)
    for g, d in sorted(res["gpu_delta_mib"].items()):
        row(f"gpu{g}_delta_mib_p5", 32 if g == 0 else "report", d, d <= 32 if g == 0 else True, unit="MiB")
    failed = [r for r in ROWS.values() if not r["pass"]]
    assert not failed, json.dumps(failed, indent=1)


@pytest.mark.skipif(not TUNNEL, reason="the tunnel column needs MACE_GUI_BENCH_TUNNEL_DB or MACE_GUI_BENCH_TUNNEL (a mace gui URL through ssh -L)")
def test_tunnel_rows(report):
    """Plan T15: the same rows through `ssh -J hpcc -L` to a dev node (module Python),
    the Setup and database latencies there, and Run Preflight with CRYSTAL's TESTPDIM."""
    base = TUNNEL.split("?")[0]
    drv = H.firefox(1600, 1000)
    try:
        drv.get(TUNNEL)
        H.wait(drv, "return document.documentElement.dataset.appReady === '1'", 60)
        _db_rows(drv, base, "big10k.db", "10k", "tunnel")
    finally:
        drv.quit()
    # plotly over the tunnel: the demo set with results/ (the shell bench's server), loaded once, then cached
    shell = os.environ.get("MACE_GUI_BENCH_TUNNEL")
    if shell:
        drv = H.firefox(1600, 1000)
        try:
            drv.get(shell)
            H.wait(drv, "return document.documentElement.dataset.appReady === '1'", 60)
            _plotly_rows(drv, "tunnel")
        finally:
            drv.quit()
    t0 = time.perf_counter()
    for _ in range(5):
        st, _, r = U.req(base, "api/db/page?path=big10k.db&sort=formula&dir=asc&offset=5000")
    ms = (time.perf_counter() - t0) * 200
    row("db_page_request_ms", "report", round(ms, 1), st == 200, "tunnel", unit="ms")
    t0 = time.perf_counter()
    r, stages = _task(base, {"source": {"path": "structures/MgO.cif"}, "calc_type": "OPT", "preview": True, "wait": 0})
    cold = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    r2, _ = _task(base, {"source": {"path": "structures/MgO.cif"}, "calc_type": "OPT", "preview": True, "wait": 0})
    warm = (time.perf_counter() - t0) * 1000
    row("setup_preview_ms_cold", "report", round(cold), bool(r.get("deck")), "tunnel", unit="ms")
    row("setup_preview_ms_cached", "report", round(warm), r2.get("deck") == r.get("deck"), "tunnel", unit="ms")
    # Run Preflight: row 1's deck and a follow-up SP deck ACCEPTED by the serial crystal, a broken one REJECTED
    sp, _ = _task(base, {"source": {"path": "outputs/Si_sp.out"}, "calc_type": "SP", "preview": True, "wait": 0})
    broken = r["deck"].replace("POB-TZVP-REV2", "DEF2-MSVP")
    lines = broken.splitlines()
    # Mg (Z 12) as Pb (Z 82): CRYSTAL23's internal def2-mSVP has no Pb, so the basis input is short of an element
    lines = [("82" + ln[2:] if ln.startswith("12 ") else ln) for ln in lines]
    broken = "\n".join(lines) + "\n"
    out = {}
    for name, deck_name, text in (("row1", r["deck_name"], r["deck"]), ("sp", sp["deck_name"], sp["deck"]), ("broken", "broken_pb.d12", broken)):
        res, _ = _task(base, {"deck_name": deck_name, "deck": text, "wait": 0}, "api/setup/preflight")
        out[name] = {"status": res.get("status"), "reason": res.get("reason"), "crystal": res.get("crystal")}
    ok = out["row1"]["status"] == "accepted" and out["sp"]["status"] == "accepted" and out["broken"]["status"] == "rejected"
    row("hpcc_preflight", "row 1 + SP accepted, broken rejected", out, ok, "tunnel")
    failed = [x for x in ROWS.values() if not x["pass"]]
    assert not failed, json.dumps(failed, indent=1)
