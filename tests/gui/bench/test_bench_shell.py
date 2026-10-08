"""GUI performance budgets (spec §8.1, §8.7, the Phase 1 and Phase 2 rows of §8.8).

Opt-in: MACE_GUI_BENCH=1 pytest -m guibench tests/gui/bench
Writes bench_report.json (one row per metric: budget, measured local,
measured tunnel, pass) into $MACE_GUI_BENCH_OUT or the test's tmp dir,
including the section 9 resource rows: CPU workers, peak RSS, and the
per-GPU memory delta (the browser renders in software; GPU0 must stay ~0).
Run it serially and niced: nice -n 10 pytest ... (never pytest -n).
For the tunnel column, point MACE_GUI_BENCH_TUNNEL at a `mace gui` URL
(with its one-time ?login= link) reached through `ssh -L`; the browser rows run there too.
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import guiharness as H  # noqa: E402

pytestmark = [pytest.mark.guibench,
              pytest.mark.skipif(os.environ.get("MACE_GUI_BENCH") != "1",
                                 reason="GUI benchmarks are opt-in: MACE_GUI_BENCH=1")]
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")

ROWS = {}   # metric -> row


def row(metric, budget, value, ok, column="local", unit=""):
    r = ROWS.setdefault(metric, {"metric": metric, "budget": budget, "unit": unit,
                                 "local": None, "tunnel": None, "pass": True})
    # a row written twice (two modules' resource rows) keeps the failing value, so the report shows why it failed
    if not ok or r["pass"] or r[column] is None:
        r[column] = value
    r["pass"] = bool(r["pass"] and ok)
    return ok


def p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]


@pytest.fixture(scope="module")
def report(tmp_path_factory, request):
    out = Path(os.environ.get("MACE_GUI_BENCH_OUT") or tmp_path_factory.mktemp("bench"))
    yield out
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench_report.json").write_text(json.dumps(
        {"phase": "1-5: every bench module that ran (rows are shared)", "written_by": "test_bench_shell", "load_average": [round(x, 2) for x in os.getloadavg()], "rows": list(ROWS.values()),
         # a test that stops on an exception records no failing row: count it too
         "tests_failed": request.session.testsfailed,
         "all_pass": all(r["pass"] for r in ROWS.values()) and not request.session.testsfailed},
        indent=2) + "\n")
    print(f"\nbench report: {out / 'bench_report.json'}")


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("benchsrv")
    root, xdg = H.examples_copy(tmp), tmp / "xdg"
    (root / "results").mkdir()
    sys.path.insert(0, str(H.REPO_ROOT / "tests"))
    from test_gui_results import write_band
    write_band(root / "results")
    proc, base, _ = H.start_server(root, xdg)
    yield proc, base, root, xdg
    proc.terminate()
    proc.wait(timeout=15)


def test_startup_until_url(report, tmp_path):
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    times = []
    for _ in range(5):
        proc, _, t = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1")
        times.append(t)
        proc.terminate()
        proc.wait(timeout=15)
    med = statistics.median(times)
    assert row("start_until_url_s", 0.6, round(med, 3), med <= 0.6, unit="s"), times


def _cpu_seconds(pid):
    with open(f"/proc/{pid}/stat") as fh:
        f = fh.read().rsplit(")", 1)[1].split()
    return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")


def _browser_rows(drv, url, column, server_pid=None):
    """The Phase 1 browser budgets against one served page."""
    t0 = time.perf_counter()
    drv.get(url)
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'")
    nav = drv.execute_script("""
      const fcp = performance.getEntriesByName('first-contentful-paint')[0];
      const ready = performance.getEntriesByName('mace:ready')[0];
      const res = performance.getEntriesByType('resource');
      const shell = res.filter((r) => /\\.(js|css)(\\?|$)/.test(r.name) && !/vendor\\//.test(r.name));
      const three = res.find((r) => r.name.includes('3Dmol'));
      return {fcp: fcp ? fcp.startTime : null, ready: ready ? ready.startTime : null,
              shellBytes: shell.reduce((s, r) => s + (r.transferSize || r.encodedBodySize), 0),
              shellFiles: shell.length, threeStart: three ? three.startTime : null,
              plotly: res.some((r) => r.name.includes('plotly'))}""")
    # spec §8.7: timings that depend on drawing are from a software renderer here
    renderer = H.webgl_renderer(drv)
    row("webgl_renderer", "software (llvmpipe)", renderer, True, column)
    row("first_paint_ms", 1000 if column == "local" else 1500, round(nav["fcp"] or -1),
        nav["fcp"] is not None and nav["fcp"] <= (1000 if column == "local" else 1500), column, "ms")
    row("shell_ready_ms", 1200 if column == "local" else 2000, round(nav["ready"]),
        nav["ready"] <= (1200 if column == "local" else 2000), column, "ms")
    row("shell_js_css_gzip_kb", 150, round(nav["shellBytes"] / 1024, 1), nav["shellBytes"] <= 150 * 1024, column, "KB")
    row("3dmol_deferred_after_first_paint", True,
        nav["threeStart"] is None or (nav["fcp"] is not None and nav["threeStart"] >= nav["fcp"] - 1),
        nav["threeStart"] is None or (nav["fcp"] is not None and nav["threeStart"] >= nav["fcp"] - 1), column)
    # documents: open a structure and an OPT trajectory; plotly stays unrequested
    for path in ("structures/MgO.cif", "outputs/Ag2Br3_opt_unfinished.out"):
        drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then(() => d())", path)
    time.sleep(0.5)
    plotly_early = drv.execute_script("return performance.getEntriesByType('resource').some((r) => r.name.includes('plotly'))")
    row("plotly_not_requested_before_a_plot", True, not plotly_early, not plotly_early, column)
    # tab switch, 20 times
    drv.execute_script("performance.clearMeasures('ui:tab')")
    for i in range(20):
        drv.execute_async_script("const d = arguments[1]; window.__mace.activate(arguments[0]).then(() =>"
                                 "requestAnimationFrame(() => requestAnimationFrame(() => d())))", i % 2)
    tabs = drv.execute_script("return performance.getEntriesByName('ui:tab').map((m) => m.duration)")
    row("tab_switch_p95_ms", 100, round(p95(tabs), 1), len(tabs) >= 20 and p95(tabs) <= 100, column, "ms")
    # menus and the palette (Phase 2 budgets, measured from now on)
    lat = []
    for i in range(20):
        lat.append(drv.execute_async_script("""const d = arguments[0]; const t = performance.now();
          const b = document.querySelectorAll('.menu-top')[2];
          b.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true, button: 0}));
          requestAnimationFrame(() => { const x = performance.now() - t;
            document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape'}));
            document.querySelector('.menu') && document.querySelector('.menu').dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
            d(x); });"""))
    row("menu_open_p95_ms", 100, round(p95(lat), 1), p95(lat) <= 100, column, "ms")
    lat = []
    for i in range(20):
        lat.append(drv.execute_async_script("""const d = arguments[0]; const t = performance.now();
          window.__mace.palette.open('');
          requestAnimationFrame(() => { const x = performance.now() - t; window.__mace.palette.close(); d(x); });"""))
    row("palette_open_p95_ms", 100, round(p95(lat), 1), p95(lat) <= 100, column, "ms")
    filt = drv.execute_script("""const t = []; for (const q of ['s','su','sup','supe','super','show b','gamma','view along']) {
        const a = performance.now(); window.__mace.palette.results(q); t.push(performance.now() - a); } return t""")
    row("palette_filter_ms", 8, round(max(filt), 2), max(filt) <= 8, column, "ms")
    # idle: no animation frames, no requests, (locally) no server CPU
    drv.execute_script("""window.__idle = {raf: 0, req: performance.getEntriesByType('resource').length};
      const orig = window.requestAnimationFrame; window.__origRaf = orig;
      window.requestAnimationFrame = (f) => { window.__idle.raf++; return orig(f); };""")
    cpu0 = _cpu_seconds(server_pid) if server_pid else None
    time.sleep(5)
    idle = drv.execute_script("""const r = {raf: window.__idle.raf,
        req: performance.getEntriesByType('resource').length - window.__idle.req};
      window.requestAnimationFrame = window.__origRaf; return r""")
    row("idle_animation_frames_5s", 0, idle["raf"], idle["raf"] == 0, column)
    row("idle_requests_5s", 0, idle["req"], idle["req"] == 0, column)
    if server_pid:
        used = _cpu_seconds(server_pid) - cpu0
        row("idle_server_cpu_percent", 1.0, round(used / 5 * 100, 2), used / 5 * 100 <= 1.0, column, "%")
    connected = drv.execute_script("return document.querySelector('#st-conn').textContent")
    row("sse_connected", True, "Connected" in connected, "Connected" in connected, column)
    # a reload (F5: a normal reload, not WebDriver's cache-bypassing refresh)
    # re-downloads nothing but the page and its one boot request
    drv.execute_script("window.__before_reload = 1; location.reload()")
    H.wait(drv, "return !window.__before_reload && document.documentElement.dataset.appReady === '1'")
    time.sleep(0.5)
    sent = drv.execute_script("""const n = performance.getEntriesByType('navigation')[0];
      return (n ? n.transferSize : 0) + performance.getEntriesByType('resource').filter((r) => !/3Dmol/.test(r.name))
        .reduce((s, r) => s + r.transferSize, 0)""")
    row("reload_transfer_kb", 10, round(sent / 1024, 2), sent <= 10 * 1024, column, "KB")
    # what the reloaded page fetched of the lazy modules, before anything opens (a plot document
    # loads js/lazy/lazy_util.js for its stylesheet: that is on demand, not at boot)
    drv.execute_script("""window.__bootLazy = performance.getEntriesByType('resource')
        .filter((r) => ['/js/lazy/', '/js/editor/', '/js/viewport/', '/js/geom/', 'dialogs.css', 'editor.css'].some((p) => r.name.includes(p))).length""")
    # the band plot loads plotly (lazily) and still works
    drv.execute_async_script("const d = arguments[0]; window.__mace.run('analysis.band'); setTimeout(d, 50)")
    # (Phase 5: the band plot is a document of its own, js/docs/plot.js)
    H.wait(drv, "const v = window.__mace.plotView, s = v && v.state(); return !!(s && s.drawn && document.querySelector('#plot-host .main-svg'))", 30)
    lazy = drv.execute_script("return performance.getEntriesByType('resource').some((r) => r.name.includes('plotly'))")
    row("plotly_loads_when_a_plot_opens", True, lazy, lazy, column)
    _phase2_rows(drv, column)
    return time.perf_counter() - t0


def _raf_latency(drv, script, n=20):
    """Event -> next animation frame, n times (spec §8.1 measures menus that way)."""
    out = []
    for i in range(n):
        out.append(drv.execute_async_script("""const done = arguments[arguments.length - 1], i = arguments[0];
          const t = performance.now(); (function () { """ + script + """ })();
          requestAnimationFrame(() => { const x = performance.now() - t;
            document.querySelectorAll('.menu').forEach((m) => m.remove());
            if (window.__mace.palette.isOpen()) window.__mace.palette.close();
            done(x); });""", i))
    return out


def _phase2_rows(drv, column):
    """Phase 2 (§8.8): palette, menu and context menu open ≤50 ms (p95 ≤100),
    palette filter ≤8 ms per keystroke on 500 commands + 2000 atoms,
    Settings and the shortcut sheet loaded on first open."""
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.common.keys import Keys
    lazy = drv.execute_script("""return window.__bootLazy !== undefined ? window.__bootLazy : performance.getEntriesByType('resource')
        .filter((r) => ['/js/lazy/', '/js/editor/', '/js/viewport/', '/js/geom/', 'dialogs.css', 'editor.css'].some((p) => r.name.includes(p))).length""")
    row("lazy_modules_absent_at_boot", True, lazy == 0, lazy == 0, column)
    drv.execute_async_script("const d = arguments[0]; window.__mace.open('structures/MgO.cif').then(() => setTimeout(d, 400))")
    # menus: all nine, 20 opens in turn
    lat = _raf_latency(drv, """const b = document.querySelectorAll('.menu-top')[i % 9];
        b.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true, button: 0}));""", 9 * 20)
    row("menu_open_p95_ms", 100, round(p95(lat), 1), p95(lat) <= 100, column, "ms")
    row("menu_open_median_ms", 50, round(statistics.median(lat), 1), statistics.median(lat) <= 50, column, "ms")
    # the palette: from the API and from a Ctrl+Alt+K key press through the dispatcher
    lat = _raf_latency(drv, "window.__mace.palette.open('');")
    lat += _raf_latency(drv, """document.body.dispatchEvent(new KeyboardEvent('keydown',
        {key: 'k', code: 'KeyK', ctrlKey: true, altKey: true, bubbles: true, cancelable: true}));""")
    row("palette_open_p95_ms", 100, round(p95(lat), 1), p95(lat) <= 100, column, "ms")
    row("palette_open_median_ms", 50, round(statistics.median(lat), 1), statistics.median(lat) <= 50, column, "ms")
    # context menus: atom (a real hit test at Mg1) and explorer row
    lat = _raf_latency(drv, """const p = window.__mace.atomScreen(0), r = document.querySelector('#viewer-main canvas').getBoundingClientRect();
        window.__mace.contextAt(r.left + p.x, r.top + p.y);""")
    lat += _raf_latency(drv, """const f = document.querySelector('#file-list .file');
        f.dispatchEvent(new MouseEvent('contextmenu', {bubbles: true, cancelable: true, clientX: 100, clientY: 200}));""")
    row("context_menu_open_p95_ms", 100, round(p95(lat), 1), p95(lat) <= 100, column, "ms")
    row("context_menu_open_median_ms", 50, round(statistics.median(lat), 1), statistics.median(lat) <= 50, column, "ms")
    # the palette filter, rank + DOM, on the stress set
    times = drv.execute_script("return window.__mace.palette.bench(500, 2000)")
    worst = max(t["ms"] for t in times)
    row("palette_filter_ms", 8, round(worst, 2), worst <= 8, column, "ms")
    row("palette_dom_rows_max", 40, max(t["dom"] for t in times), max(t["dom"] for t in times) <= 40, column)
    # and with real typing, from the ui:filter measures
    drv.execute_script("performance.clearMeasures('ui:filter'); window.__mace.palette.open('')")
    box = drv.find_element("css selector", "#palette input")
    for ch in "show bonds supercell":
        box.send_keys(ch)
    real = drv.execute_script("return performance.getEntriesByName('ui:filter').map((m) => m.duration)")
    drv.execute_script("window.__mace.palette.close()")
    row("palette_filter_real_p95_ms", 8, round(p95(real), 2), bool(real) and p95(real) <= 8, column, "ms")
    # a tooltip on keyboard focus shows at once
    lat = _raf_latency(drv, """const bs = document.querySelectorAll('#toolbar [data-cmd]'); document.body.focus();
        const b = bs[i % bs.length]; b.focus({focusVisible: true});
        if (document.querySelector('#tooltip').hidden) b.dispatchEvent(new FocusEvent('focusin', {bubbles: true}));""")
    row("tooltip_show_ms", 50, round(p95(lat), 1), p95(lat) <= 50, column, "ms")
    # Settings and the shortcut sheet: the first open fetches them, a second does not
    for name, opener, dlg in (("settings", "window.__mace.run('file.settings')", "settings"),
                              ("cheatsheet", "window.__mace.run('help.shortcuts')", "cheatsheet")):
        first = drv.execute_async_script("""const d = arguments[1]; const t = performance.now(); """ + opener + """;
          (function wait() { const el = document.getElementById('""" + dlg + """');
            if (el && el.open) requestAnimationFrame(() => d(performance.now() - t)); else setTimeout(wait, 2); })();""", dlg)
        row(f"{name}_first_open_ms", "info (target 250)", round(first, 1), True, column, "ms")
        drv.execute_script(f"document.getElementById('{dlg}').close()")
        # static fetches only: Settings asks /api/info for fresh resource numbers on each open, by design
        static = "return performance.getEntriesByType('resource').filter((r) => !r.name.includes('/api/')).length"
        n0 = drv.execute_script(static)
        again = []
        for _ in range(10):
            again.append(drv.execute_async_script("""const d = arguments[0]; const t = performance.now(); """ + opener + """;
              (function wait() { const el = document.getElementById('""" + dlg + """');
                if (el && el.open) requestAnimationFrame(() => { const x = performance.now() - t; el.close(); d(x); }); else setTimeout(wait, 1); })();"""))
        more = drv.execute_script(static) - n0
        if name == "settings":
            row("settings_reopen_p95_ms", 100, round(p95(again), 1), p95(again) <= 100, column, "ms")
        row(f"{name}_reopen_requests", 0, more, more == 0, column)
    # one key press through the chord index, in microseconds (information)
    us = drv.execute_script("""const n = 2000, t = performance.now();
        for (let k = 0; k < n; k++) document.body.dispatchEvent(new KeyboardEvent('keydown', {key: 'q', code: 'KeyQ', ctrlKey: true, altKey: true, bubbles: true}));
        return (performance.now() - t) / n * 1000""")
    row("keydown_dispatch_us", "info", round(us, 1), True, column, "µs")


def worker_cap():
    """Spec section 9: at most a quarter of the usable cores, 1 to 8."""
    n = len(os.sched_getaffinity(0))
    slurm = os.environ.get("SLURM_CPUS_ON_NODE")
    return max(1, min(n // 4, 8, int(slurm) if slurm and slurm.isdigit() else 8))


def resource_rows(res):
    """Spec section 9.4: CPU workers, peak RSS, per-GPU memory deltas (GPU0 ~0)."""
    cap = worker_cap()
    workers = res["max_children"].get("server", 0)
    row("cpu_workers_used", f"<= {cap}", workers, workers <= cap)
    rss = res["peak_rss_mb"]
    row("server_peak_rss_mb", 512, rss.get("server"), (rss.get("server") or 0) <= 512, unit="MB")
    row("browser_peak_pss_mb", None, rss.get("browser"), True, unit="MB")
    for g, d in sorted(res["gpu_delta_mib"].items()):
        # GPU0 drives the display: the run must leave it alone; the others
        # are shared with training jobs, so their delta is only recorded
        # (other users' listed processes, such as a remote-desktop client, are left out; the raw delta is recorded too)
        row(f"gpu{g}_memory_delta_mib", 32 if g == 0 else None, d, d <= 32 if g == 0 else True, unit="MiB")
    for g, d in sorted(res.get("gpu_delta_raw_mib", {}).items()):
        row(f"gpu{g}_memory_delta_raw_mib", "info", d, True, unit="MiB")
    ours = res["gpu_processes_of_ours"]
    row("gpu_processes_of_ours", 0, len(ours), not ours)


def test_browser_budgets(report, server):
    proc, base, _, _ = server
    drv = H.firefox(1920, 1080)
    sampler = H.ResourceSampler().watch("server", proc.pid).watch("browser", drv.service.process.pid).start()
    try:
        _browser_rows(drv, H.login_url(base), "local", proc.pid)
        tunnel = os.environ.get("MACE_GUI_BENCH_TUNNEL")
        if tunnel:
            _browser_rows(drv, tunnel, "tunnel")
        resource_rows(sampler.stop())
    finally:
        sampler.stop_ev.set()
        drv.quit()
    failed = [r for r in ROWS.values() if not r["pass"]]
    assert not failed, json.dumps(failed, indent=1)


def test_sse_heartbeat_timing(report, server):
    import socket
    _, base, _, _ = server
    from urllib.parse import urlsplit
    u = urlsplit(base)
    host, port, prefix = u.hostname, u.port, u.path
    s = socket.create_connection((host, int(port)), timeout=10)
    s.sendall(f"GET {prefix}api/events?interval=0.5 HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
              f"X-Mace-Token: {H.TOKEN}\r\n\r\n".encode())
    t0, data, beats = time.time(), b"", []
    while len(beats) < 4 and time.time() - t0 < 6:
        chunk = s.recv(4096)
        data += chunk
        if chunk.count(b"event: heartbeat"):
            beats.append(time.time() - t0)
    s.close()
    gaps = [b - a for a, b in zip(beats, beats[1:])]
    ok = len(beats) >= 4 and all(0.3 <= g <= 0.8 for g in gaps)
    assert row("sse_heartbeat_interval_s", 0.5, round(statistics.mean(gaps), 3) if gaps else None, ok, unit="s"), beats
