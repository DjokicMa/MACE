"""Phase 4 budgets (spec §8.8) and the §9 rows for the pool, the tasks and
the file tools.

Opt-in like the other benches: MACE_GUI_BENCH=1 nice -n 10 pytest -m guibench tests/gui/bench
(serially, never with -n). The rows go into the same bench_report.json
(test_bench_shell's ROWS and row()); this module rewrites the file when it
finishes.

Fixtures are generated into tmp dirs and deleted at teardown, never
committed:
  folder_2000/: 2000 small CIFs written by ase from the six demo structures
    with small random strains, plus 50 copies of the small demo .out files;
  big_matdump.out (34 MB, the demo OPT output with one point repeated; see
    test_outindex.make_big);
  the 85k-atom MgO 22x22x22 supercell is built through the API (it is the
    thing measured), only with MACE_GUI_BENCH_BIG=1.
The big ones check MemAvailable >= 4 GB and /tmp free >= 2 GB first and skip
with a message naming the check.
"""
import gzip
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import guiharness as H  # noqa: E402
import p4util as U  # noqa: E402
import test_bench_shell as B  # noqa: E402  (the shared ROWS, row(), p95())

pytestmark = [pytest.mark.guibench,
              pytest.mark.skipif(os.environ.get("MACE_GUI_BENCH") != "1",
                                 reason="GUI benchmarks are opt-in: MACE_GUI_BENCH=1")]
pytest.importorskip("ase")
np = pytest.importorskip("numpy")

ROWS, row = B.ROWS, B.row
BIG = os.environ.get("MACE_GUI_BENCH_BIG") == "1"


@pytest.fixture(scope="module")
def report(tmp_path_factory, request):
    out = Path(os.environ.get("MACE_GUI_BENCH_OUT") or tmp_path_factory.mktemp("bench4"))
    yield out
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench_report.json").write_text(json.dumps(
        {"phase": "1-5: every bench module that ran (rows are shared)", "written_by": "test_bench_files", "load_average": [round(x, 2) for x in os.getloadavg()], "rows": list(ROWS.values()), "tests_failed": request.session.testsfailed,
         "all_pass": all(r["pass"] for r in ROWS.values()) and not request.session.testsfailed}, indent=2) + "\n")
    print(f"\nbench report: {out / 'bench_report.json'}")


def room(mem_gb=4, tmp_gb=2):
    if not U.mem_ok(mem_gb):
        return f"MemAvailable under {mem_gb} GB"
    if not U.tmp_free_ok(tmp_gb):
        return f"/tmp free under {tmp_gb} GB"
    return ""


def write_folder_2000(root: Path):
    import warnings
    import ase.io
    d = root / "folder_2000"
    d.mkdir()
    rng = np.random.default_rng(2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = [ase.io.read(str(p)) for p in sorted((H.EXAMPLES / "structures").glob("*.cif"))]
        for k in range(2000):
            a = base[k % len(base)].copy()
            strain = np.eye(3) + rng.normal(0, 0.01, (3, 3))
            a.set_cell(a.cell.array @ strain, scale_atoms=True)
            ase.io.write(str(d / f"s{k:04d}.cif"), a, format="cif")
    outs = sorted(p for p in (H.EXAMPLES / "outputs").glob("*.out") if p.stat().st_size < 100_000)
    for k in range(50):
        shutil.copy(outs[k % len(outs)], d / f"o{k:02d}.out")


def tree_cpu(pid):
    """utime+stime (s) of a process and all its descendants."""
    tck = os.sysconf("SC_CLK_TCK")
    total = 0.0
    for p in [pid] + H.descendants(pid):
        try:
            with open(f"/proc/{p}/stat") as fh:
                f = fh.read().rsplit(")", 1)[1].split()
            total += (int(f[11]) + int(f[12])) / tck
        except OSError:
            pass
    return total


def threads_of(pid):
    with open(f"/proc/{pid}/status") as fh:
        for line in fh:
            if line.startswith("Threads:"):
                return int(line.split()[1])
    return 0


def rss_mb(pid, key="VmRSS"):
    with open(f"/proc/{pid}/status") as fh:
        for line in fh:
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 1024
    return 0.0


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bench4srv")
    root, xdg = H.examples_copy(tmp), tmp / "xdg"
    write_folder_2000(root)
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_TEST_HOOKS="1", MACE_GUI_NO_WARMUP="1")
    yield proc, base, root, tmp
    proc.terminate()
    proc.wait(timeout=15)
    shutil.rmtree(root / "folder_2000", ignore_errors=True)


def cap_of(base):
    return U.req(base, "api/resources")[2]["budget"]["cap"]


def wait_result(base, st, r, timeout=600):
    if st == 200:
        return r
    tid = r["task"]["id"]
    end = time.time() + timeout
    while time.time() < end:
        st2, _, r2 = U.req(base, f"api/tasks/{tid}/result")
        if st2 == 200:
            return r2
        if st2 != 409:
            raise AssertionError((st2, r2))
        time.sleep(0.05)
    raise AssertionError("timed out")


def distorted_mgo(n=2, seed=1):
    from ase.build import bulk
    a = bulk("MgO", "rocksalt", a=4.212, cubic=True).repeat(n)
    a.positions += np.random.default_rng(seed).normal(0, 0.08, a.positions.shape)
    return a


def session_of(base, atoms, name):
    st, _, s = U.post(base, "api/edit/from_structure", {"name": name, "cell": atoms.cell.array.tolist(),
                                                       "frac": atoms.get_scaled_positions(wrap=False).tolist(),
                                                       "numbers": atoms.numbers.tolist(), "locked": False})
    assert st == 200, s
    return s


# --------------------------------------------------------------------------- rows
def _details_timed(base):
    """One untimed pass (it starts the workers and imports ase and spglib in
    each), the parse cache cleared, then the timed pass."""
    U.req(base, "api/files?details=1", timeout=1800)
    U.post(base, "api/test/parse_cache/clear", {})
    t0 = time.perf_counter()
    st, _, r = U.req(base, "api/files?details=1", timeout=1800)
    return time.perf_counter() - t0, r


def test_batch_parse_2000_and_speedup(report, srv, tmp_path):
    """/api/files?details=1 on 2050 files with warm workers: the pool at 1
    worker, then at the cap."""
    proc, base, root, _ = srv
    cap = cap_of(base)
    t_cap, r = _details_timed(base)
    n = sum(1 for k in r["details"] if k.startswith("folder_2000/"))
    row("batch_parse_2000_s", "info", round(t_cap, 2), n >= 2050, unit="s")
    t0 = time.perf_counter()
    U.req(base, "api/files?details=1", timeout=1200)
    row("batch_parse_2000_cache_hit_s", "info", round(time.perf_counter() - t0, 3), True, unit="s")
    # the same folder on a 1-worker server
    xdg = tmp_path / "xdg1"
    p1, b1, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1", MACE_GUI_MAX_WORKERS="1", MACE_GUI_TEST_HOOKS="1")
    try:
        t_one, _ = _details_timed(b1)
    finally:
        p1.terminate()
        p1.wait(timeout=15)
    speed = t_one / t_cap if t_cap else 0
    row(f"batch_parse_speedup_{cap}", f">= {0.6 * cap:.1f}", round(speed, 2), speed >= 0.6 * cap)
    row("batch_parse_2000_s_1worker", "info", round(t_one, 2), True, unit="s")


def test_pool_resource_rows(report, srv):
    proc, base, _, _ = srv
    pool = U.req(base, "api/test/pool")[2]["pool"]
    pids = pool["pids"] if pool else []
    if not pids:
        U.post(base, "api/test/task", {"seconds": 0.2, "steps": 1, "wait": 5})
        pids = U.req(base, "api/test/pool")[2]["pool"]["pids"]
    cap = cap_of(base)
    row("cpu_workers_used", f"> 0, <= {cap}", len(pids), 0 < len(pids) <= cap)
    nices = []
    for p in pids:
        with open(f"/proc/{p}/stat") as fh:
            nices.append(int(fh.read().rsplit(")", 1)[1].split()[16]))
    row("pool_worker_nice", 10, min(nices), all(x == 10 for x in nices))
    th = max(threads_of(p) for p in pids)
    row("pool_threads_per_worker", "<= 2", th, th <= 2)
    peak = max(rss_mb(p, "VmHWM") for p in pids)                 # the high-water mark
    row("pool_worker_peak_rss_mb", "<= 250", round(peak, 1), peak <= 250, unit="MB")
    row("pool_rss_total_peak_mb", "info", round(sum(rss_mb(p, "VmHWM") for p in pids), 1), True, unit="MB")


def test_clean_in_pool_and_parallel_speedup(report, srv):
    proc, base, _, _ = srv
    cap = cap_of(base)
    sess = [session_of(base, distorted_mgo(2, seed=k), f"mgo_d{k}") for k in range(4)]
    kw = {"time_limit": 60, "keep_symmetry": False}
    t0 = time.perf_counter()
    reports = []
    for s in sess:
        st, _, r = U.post(base, "api/clean", {"session": s["session"], "base_rev": s["rev"], **kw})
        r = wait_result(base, st, r)
        reports.append(r["report"])
        U.post(base, "api/edit/clean_discard", {"session": s["session"], "proposal_id": r["proposal_id"]})
    serial = time.perf_counter() - t0
    server_pid = U.req(base, "api/test/pool")[2]["server_pid"]
    in_pool = all(r["worker_pid"] != server_pid and r["worker_nice"] == 10 for r in reports)
    row("clean_in_pool", True, in_pool, in_pool)
    res = [None] * 4

    def one(k):
        s = sess[k]
        st, _, r = U.post(base, "api/clean", {"session": s["session"], "base_rev": s["rev"], **kw})
        res[k] = wait_result(base, st, r)
    t0 = time.perf_counter()
    th = [threading.Thread(target=one, args=(k,)) for k in range(4)]
    for t in th:
        t.start()
    for t in th:
        t.join(600)
    par = time.perf_counter() - t0
    speed = serial / par if par else 0
    want = 0.6 * min(4, cap)
    row("clean_parallel_speedup_4", f">= {want:.1f}", round(speed, 2), speed >= want)


def _cancel_latency(base, path, body):
    sse = H.sse_events(base)
    try:
        sse.wait(lambda e: e[0] == "hello", 10)
        st, _, r = U.post(base, path, body)
        if st != 202:
            return None
        tid = r["task"]["id"]
        sse.wait(lambda e: e[0] == "task" and e[1]["id"] == tid and e[1]["state"] == "running", 60)
        time.sleep(0.3)
        t0 = time.time()
        U.post(base, f"api/tasks/{tid}/cancel", {})
        ev = sse.wait(lambda e: e[0] == "task" and e[1]["id"] == tid and e[1]["state"] in ("canceled", "done", "failed"), 30)
        return ev[2] - t0 if ev[1]["state"] == "canceled" else None
    finally:
        sse.close()


def test_cancel_latency_clean_and_symmetrize(report, srv):
    from ase.build import make_supercell
    proc, base, _, _ = srv
    big = make_supercell(distorted_mgo(1, 7), np.diag([5, 5, 5]))
    big.positions += np.random.default_rng(4).normal(0, 0.2, big.positions.shape)
    s = session_of(base, big, "mgo_slow")
    lat = _cancel_latency(base, "api/clean", {"session": s["session"], "base_rev": s["rev"], "time_limit": 60, "keep_symmetry": False})
    row("cancel_latency_clean_s", "<= 2.5", None if lat is None else round(lat, 3), lat is not None and lat <= 2.5, unit="s")
    # the sweep must still be running when the cancel arrives: 8000 distorted atoms (13 spglib
    # runs of about a second each), from a file in the served folder (a session stops at 5000)
    proc_, base_, root, _ = srv
    import ase.io
    ase.io.write(str(root / "mgo_sweep_8000.cif"), make_supercell(distorted_mgo(1, 8), np.diag([10, 10, 10])), format="cif")
    try:
        lat = _cancel_latency(base, "api/symmetrize/sweep", {"source": {"path": "mgo_sweep_8000.cif"}})
    finally:
        (root / "mgo_sweep_8000.cif").unlink()
    row("cancel_latency_symmetrize_s", "<= 2.5", None if lat is None else round(lat, 3),
        lat is not None and lat <= 2.5, unit="s")    # None: no running task was canceled (a failed measurement)


def test_parse_cache_hit_and_payload(report, srv):
    proc, base, root, _ = srv
    from ase.build import bulk
    import ase.io
    ase.io.write(str(root / "mgo_1000.cif"), bulk("MgO", "rocksalt", a=4.212, cubic=True).repeat(5))
    U.req(base, "api/structure?path=mgo_1000.cif&packed=1")
    durs = []
    for _ in range(10):
        st, h, _ = U.req(base, "api/structure?path=mgo_1000.cif&packed=1")
        t = h.get("Server-Timing", "")
        if "cache;desc=hit" in t:
            durs.append(float(t.split("parse;dur=")[1].split(",")[0]))
    hit = B.p95(durs) if durs else None
    row("parse_cache_hit_ms", "< 5", hit, hit is not None and hit < 5, unit="ms")
    st, h, body = U.req(base, "api/structure?path=mgo_1000.cif&packed=1", headers={"Accept-Encoding": "gzip"})
    raw = json.dumps(body).encode()
    per = len(gzip.compress(raw, 6)) / 1000
    row("payload_bytes_per_atom_gz_1000", "<= 40", round(per, 1), per <= 40, unit="B/atom")


def test_drop_parse_dir_unchanged(report, srv):
    proc, base, root, _ = srv
    before = U.tree_md5(root)
    data = (H.EXAMPLES / "structures" / "NaCl.cif").read_bytes()
    st, _, r = U.req(base, "api/parse?box=1", raw=data, headers={"Content-Type": "application/octet-stream", "X-File-Name": "NaCl_drop.cif"})
    ok = st == 200 and U.tree_md5(root) == before and U.req(base, "api/test/parse_tmp")[2]["files"] == []
    row("drop_parse_dir_unchanged", True, ok, ok)


def test_out_34mb_summary(report, srv):
    why = room()
    if why:
        pytest.skip(f"the 34 MB output needs room: {why}")
    proc, base, root, _ = srv
    import test_outindex as T
    p = root / "big_matdump.out"
    T.make_big(p)
    try:
        rss0 = rss_mb(proc.pid)
        t0 = time.perf_counter()
        st, h, r = U.req(base, "api/out/summary?path=big_matdump.out")
        cold = (time.perf_counter() - t0) * 1000
        raw = json.dumps(r).encode()
        row("out_34mb_summary_cold_ms", "<= 2500", round(cold, 1), st == 200 and cold <= 2500, unit="ms")
        row("out_34mb_summary_bytes", "<= 300 KB", len(raw), len(raw) <= 300 * 1024, unit="B")
        t0 = time.perf_counter()
        U.req(base, "api/out/summary?path=big_matdump.out")
        hit = (time.perf_counter() - t0) * 1000
        row("out_34mb_summary_hit_ms", "<= 50", round(hit, 1), hit <= 50, unit="ms")
        st, _, s = U.req(base, f"api/out/step?path=big_matdump.out&step={r['n_steps'] // 2}")
        n = len(json.dumps(s).encode())
        row("out_step_bytes", "<= 40 KB", n, n <= 40 * 1024, unit="B")
        delta = rss_mb(proc.pid) - rss0
        row("out_34mb_server_rss_delta_mb", "<= 64", round(delta, 1), delta <= 64, unit="MB")
    finally:
        p.unlink(missing_ok=True)


def test_server_cpu_cores_mixed(report, srv):
    """Two pool Cleans with inline /api/symmetry and small builds from four
    threads for 10 s: the server and its workers stay within the cap."""
    proc, base, _, _ = srv
    cap = cap_of(base)
    U.post(base, "api/test/gate/reset", {})
    sess = [session_of(base, distorted_mgo(3, seed=20 + k), f"mgo_mix{k}") for k in range(2)]
    stop = threading.Event()
    mgo = U.req(base, "api/structure?path=structures/MgO.cif")[2]
    body = {"cell": mgo["cell"], "numbers": mgo["numbers"], "frac": mgo["frac"]}

    def loop(k):
        i = 0
        while not stop.is_set():
            if i % 2:
                U.post(base, "api/symmetry", {**body, "key": f"mix{k}-{i}"})
            else:
                U.post(base, "api/build/supercell", {"source": {"path": "structures/MgO.cif"}, "params": {"n": [2, 2, 2]}, "preview": True})
            i += 1
    c0, t0 = tree_cpu(proc.pid), time.perf_counter()
    for s in sess:
        U.post(base, "api/clean", {"session": s["session"], "base_rev": s["rev"], "time_limit": 10, "keep_symmetry": False})
    th = [threading.Thread(target=loop, args=(k,)) for k in range(4)]
    for t in th:
        t.start()
    time.sleep(10)
    stop.set()
    for t in th:
        t.join(60)
    cores = (tree_cpu(proc.pid) - c0) / (time.perf_counter() - t0)
    row("server_cpu_cores_mixed", f"<= {cap + 0.5}", round(cores, 2), cores <= cap + 0.5)
    g = U.req(base, "api/test/gate")[2]
    row("budget_max_running", f"<= {cap}", g["max_running"], g["max_running"] <= cap)


def test_ledger_rows(report, srv):
    """Runs after the batch parse, the Cleans and the 34 MB summary: the
    ledger's own high-water marks."""
    proc, base, _, _ = srv
    st = U.req(base, "api/resources")[2]["stores"]
    cap = st["limit_mb"]
    row("ledger_total_max_mb", f"<= {cap}", st["max_total_mb"], st["max_total_mb"] <= cap, unit="MB")
    pin = st.get("max_held_pinned_mb", 0.0)
    row("held_pinned_max_mb", f"<= {cap / 4}", pin, pin <= cap / 4, unit="MB")


def test_pool_idle_shutdown(report, srv):
    proc, base, _, _ = srv
    U.post(base, "api/test/task", {"seconds": 0.1, "steps": 1, "wait": 5})
    pids = U.req(base, "api/test/pool")[2]["pool"]["pids"]
    t0 = time.time()
    while time.time() - t0 < 90 and any(os.path.exists(f"/proc/{p}") for p in pids):
        time.sleep(0.5)
    gone = time.time() - t0
    row("pool_idle_shutdown_s", "<= 65", round(gone, 1), gone <= 65, unit="s")


@pytest.mark.skipif(not BIG, reason="the 85k-atom rows are opt-in: MACE_GUI_BENCH_BIG=1")
def test_supercell_85k(report, srv):
    why = room()
    if why:
        pytest.skip(f"the 85k build needs room: {why}")
    proc, base, _, _ = srv
    c0 = tree_cpu(proc.pid)
    sse = H.sse_events(base)
    try:
        sse.wait(lambda e: e[0] == "hello", 10)
        t0 = time.time()
        st, h, r = U.post(base, "api/build/supercell", {"source": {"path": "structures/MgO.cif"}, "params": {"n": [22, 22, 22]},
                                                         "open_as": {"name": "MgO_22"}})
        assert st in (200, 202), r
        if st == 202:
            tid = r["task"]["id"]
            done = sse.wait(lambda e: e[0] == "task" and e[1]["id"] == tid and e[1]["state"] in ("done", "failed"), 120)
            took = done[2] - t0
            prog = [e for e in sse.of("task", lambda d: d["id"] == tid) if e[1]["state"] == "running" and 0 < e[1]["progress"] < 1]
            # progress travels at most 4 times a second: a build done in under a second may show none
            row("supercell_85k_progress_events", ">= 2 (or done in < 1 s)", len(prog), len(prog) >= 2 or done[2] - t0 < 1.0)
            wait_result(base, 202, r)
        else:
            took = time.time() - t0
        row("supercell_85k_build_s", "<= 3.0", round(took, 2), took <= 3.0, unit="s")
        cores = (tree_cpu(proc.pid) - c0) / max(1e-6, time.time() - t0)
        row("server_cpu_cores_during_build", f"<= {cap_of(base)}", round(cores, 2), cores <= cap_of(base))
        # the cancel: the biggest build the limits allow (30³ cells, 216,000 atoms), so it is still running when it lands
        t0b = time.time()
        st, _, r = U.post(base, "api/build/supercell", {"source": {"path": "structures/MgO.cif"}, "params": {"n": [30, 30, 30]},
                                                         "open_as": {"name": "MgO_30"}})
        if st == 202:
            tid = r["task"]["id"]
            sse.wait(lambda e: e[0] == "task" and e[1]["id"] == tid and e[1]["state"] in ("running", "done"), 60)
            t1 = time.time()
            U.post(base, f"api/tasks/{tid}/cancel", {})
            ev = sse.wait(lambda e: e[0] == "task" and e[1]["id"] == tid and e[1]["state"] in ("canceled", "done"), 30)
            lat = ev[2] - t1
            if ev[1]["state"] == "canceled":
                row("supercell_216k_cancel_latency_s", "<= 2.5", round(lat, 3), lat <= 2.5, unit="s")
            else:
                # done before the cancel landed: the whole build is shorter than a cancel round trip
                # (the cancel path itself is test_tasks_api's and the Clean/symmetrize rows)
                took = ev[2] - t0b
                row("supercell_216k_cancel_latency_s", "<= 2.5 (or the build ends first, < 2.5 s)", f"done in {took:.2f} s",
                    took <= 2.5, unit="s")
        else:
            row("supercell_216k_cancel_latency_s", "<= 2.5", None, False, unit="s")      # answered inline: nothing to cancel
    finally:
        sse.close()
