"""F12 / F7-b / F7-d: the GUI's /api/cluster routes on the real `mace gui`
with FakeSlurm on its PATH: answers, the shared cache's rate limit, the
batched sacct of open Job documents, own-jobs-only, and the served-root rule
for job files."""
import json
import os
import shutil
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cluster"))
from cluster_fixtures import TEST_DATA, fake  # noqa: E402

from mace.cluster import query  # noqa: E402


def _opt_job(d: Path, name: str, point: int = 5):
    src = sorted((TEST_DATA / "OPT").glob("Ag1Cl5_sym*.out")) if (TEST_DATA / "OPT").exists() else []
    if not src:
        pytest.skip("test/ data corpus not present")
    d.mkdir(parents=True, exist_ok=True)
    lines = src[0].read_text(errors="replace").splitlines(True)
    heads = [i for i, l in enumerate(lines) if "OPTIMIZATION - POINT" in l]
    (d / f"{name}.out").write_text("".join(lines[:heads[point]]))
    shutil.copy(src[0].with_suffix(".d12"), d / f"{name}.d12")
    return src[0], lines, heads


def _job_line(jid, name, d, limit="02:00:00", run="00:28:00"):
    return (f"JobId={jid} JobName={name} UserId=user1(1001) JobState=RUNNING Reason=None RunTime={run} "
            f"TimeLimit={limit} Partition=general-long NodeList=cpu-001 NumNodes=1 NumCPUs=32 WorkDir={d} "
            f"StdOut={d}/{name}-{jid}.o Account=acct1 QOS=normal")


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("clusterapi")
    root = H.examples_copy(tmp)
    home = tmp / "home"
    home.mkdir()
    fs = fake(tmp)
    jobs = {}
    for k in range(5):
        d = root / "runs" / f"m{k}"
        _opt_job(d, f"m{k}_opt")
        jid = str(4001 + k)
        fs.add_record(id=jid, state="RUNNING", name=f"m{k}_opt", workdir=str(d), cpus=32, time_limit="2:00:00",
                      elapsed="0:28:00")
        jobs[jid] = _job_line(jid, f"m{k}_opt", d)
    other = home / "elsewhere"                      # inside an allowed root (HOME), outside the served folder
    _opt_job(other, "o_opt")
    fs.add_record(id="4100", state="RUNNING", name="o_opt", workdir=str(other), cpus=32)
    jobs["4100"] = _job_line("4100", "o_opt", other)
    far = tmp / "far"                               # outside every allowed root
    _opt_job(far, "f_opt")
    fs.add_record(id="4200", state="RUNNING", name="f_opt", workdir=str(far), cpus=32)
    jobs["4200"] = _job_line("4200", "f_opt", far)
    fs.update(job_text=jobs, sacct_rows={})
    env = {k: v for k, v in fs.env.items() if k in ("PATH", "FAKE_SLURM_STATE", "FAKE_SLURM_LOG",
                                                    "FAKE_SLURM_HOOK_LOG", "USER", "LOGNAME")}
    env.update(HOME=str(home), MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1", MACE_CLUSTER_TEST_POLL_SECONDS="1",
               MACE_SITE_CONFIG="none")
    md5 = U.tree_md5(root)
    proc, base, _ = H.start_server(root, tmp / "xdg", **env)
    yield {"base": base, "root": root, "fs": fs, "xdg": tmp / "xdg", "md5": md5, "home": home}
    proc.terminate()
    proc.wait(timeout=15)


def _get(s, route, **q):
    qs = "&".join(f"{k}={v}" for k, v in q.items())
    return U.req(s["base"], f"api/cluster/{route}" + (f"?{qs}" if qs else ""))


def _fetch_log(s, key):
    cache = Path(s["xdg"]) / "cache" / "mace" / "cluster" / "cpu-cluster" / f"{key}.json"
    try:
        return json.loads(cache.read_text()).get("log") or []
    except (OSError, ValueError):
        return []


def test_routes_answer(server):
    """F12-a: each route."""
    st, _, caps = _get(server, "capabilities")
    assert st == 200 and caps["slurm"] and caps["cluster_name"] == "cpu-cluster"
    for route in ("overview", "nodes", "reservations", "limits", "quotas"):
        st, _, r = _get(server, route)
        assert st == 200, (route, r)
    st, _, r = _get(server, "nodes", group="partition", all="1")
    assert any(p["name"] == "visible-restricted" and not p["usable"] for p in r["partitions"])
    st, _, r = U.post(server["base"], "api/cluster/fit", {"cpus": 24, "mem_per_cpu": "4G", "time": "1-00:00:00"})
    assert st == 200 and r["partitions"]
    st, _, r = U.post(server["base"], "api/cluster/predict", {"cpus": 24, "time": "1-00:00:00"})
    assert st == 200 and r["predict"]["accepted"] is True
    st, _, r = U.post(server["base"], "api/cluster/predict", {"cpus": 25, "time": "1-00:00:00"})
    assert st == 200 and r["predict"].get("throttled")          # one prediction per 10 s per server
    assert U.tree_md5(server["root"]) == server["md5"]


def test_nodes_rate_limit(server):
    """F12-a: 200 requests in 10 s -> 1 scontrol show nodes (FakeSlurm's log
    and the cache's fetch log agree)."""
    fs = server["fs"]
    before = len([c for c in fs.calls("scontrol") if "nodes" in c["argv"]])
    before_log = len(_fetch_log(server, "nodes"))
    t0 = time.time()
    for _ in range(200):
        assert _get(server, "nodes")[0] == 200
    assert time.time() - t0 < 10
    calls = len([c for c in fs.calls("scontrol") if "nodes" in c["argv"]]) - before
    logged = len(_fetch_log(server, "nodes")) - before_log
    assert calls <= 1 and calls == logged, (calls, logged)


def _events(base, stop):
    """Hold an /api/events stream open (a connected page)."""
    h = {"X-Mace-Token": H.TOKEN}
    r = urllib.request.Request(base + "api/events?interval=1", headers=h)
    with urllib.request.urlopen(r, timeout=60) as resp:
        while not stop.is_set():
            if not resp.readline():
                break


def test_job_documents_share_one_sacct(server):
    """F7-d: five Job documents open over the poll ticks: every sacct call
    carries all five ids; one scontrol show job per open; no sstat until
    Show Live Use, which makes one."""
    fs = server["fs"]
    stop = threading.Event()
    t = threading.Thread(target=_events, args=(server["base"], stop), daemon=True)
    t.start()
    try:
        ids = [str(4001 + k) for k in range(5)]
        n_show = len([c for c in fs.calls("scontrol") if "job" in c["argv"]])
        for jid in ids:
            st, _, r = _get(server, "job", id=jid)
            assert st == 200, r
            assert r["in_root"] and r["progress"]["done"] == 5
        assert len([c for c in fs.calls("scontrol") if "job" in c["argv"]]) - n_show == 5
        n_sacct = len(fs.calls("sacct"))
        # the 1 s test poll: at least 2 ticks (a loaded machine runs fewer per second), never one per job
        t0 = time.time()
        while len(fs.calls("sacct")) - n_sacct < 2 and time.time() - t0 < 15:
            time.sleep(0.25)
        sacct = fs.calls("sacct")[n_sacct:]
        assert 2 <= len(sacct) <= 2 + int(time.time() - t0) + 1, (len(sacct), time.time() - t0)
        for c in sacct:
            j = c["argv"][c["argv"].index("-j") + 1].split(",")
            assert set(ids) <= set(j), j
        assert fs.calls("sstat") == []
        st, _, r = U.post(server["base"], "api/cluster/job/live", {"id": "4001"})
        assert st == 200 and len(fs.calls("sstat")) == 1
    finally:
        stop.set()
        for jid in [str(4001 + k) for k in range(5)]:
            U.post(server["base"], "api/cluster/job/close", {"id": jid})


def test_not_yours_and_bad_ids(server):
    """F7-b."""
    st, _, r = _get(server, "job", id="777")
    assert st == 404 and r["error"] == "Not one of your jobs"
    st, _, r = _get(server, "job", id="12_3%3Brm")
    assert st == 400


def test_job_outside_the_served_folder(server):
    """F12-b: inside an allowed root -> progress, files "other_root", no text
    URL; outside every allowed root -> "outside", nothing read; /api/text
    refuses the .out."""
    st, _, r = _get(server, "job", id="4100")
    assert st == 200
    assert r["progress"] and r["progress"]["done"] == 5
    out = [e for e in r["file_entries"] if e["kind"] == "out"][0]
    assert out["access"] == "other_root" and "rel" not in out and out["open_dir"].startswith("~")
    st, _, r = _get(server, "job", id="4200")
    assert st == 200
    out = [e for e in r["file_entries"] if e["kind"] == "out"][0]
    assert out["access"] == "outside" and r["progress"] is None
    st, _, _ = U.req(server["base"], "api/text?path=" + urllib.request.quote(str(server["home"] / "elsewhere" /
                                                                                  "o_opt.out")))
    assert st in (400, 403, 404)
    assert U.tree_md5(server["root"]) == server["md5"]


def test_open_in_new_window_command_is_quoted(tmp_path, monkeypatch):
    """The "Open in a New Window Rooted at" command survives a folder name
    with a space or a ';' (one argument; ~ still expands)."""
    import shlex
    from mace.gui import cluster_api
    home = tmp_path / "home"
    d = home / "my runs; x"
    d.mkdir(parents=True)
    (d / "a.out").write_text("")
    monkeypatch.setenv("HOME", str(home))

    class St:
        root = tmp_path / "served"

    class Svc:
        def path_policy(self, real):
            return "other_root"
    out = cluster_api._job_payload(St, Svc(), {"files": {"out": str(d / "a.out")}, "access": {}})
    cmd = out["file_entries"][0]["command"]
    assert cmd == "mace gui --dir ~/'my runs; x'"
    assert shlex.split(cmd)[-1] == "~/my runs; x"


def test_network_bound_refuses_file_routes(tmp_path, monkeypatch):
    """Under --insecure-network the job, scan, estimate and predict routes
    are off (checked on the route functions with a non-loopback address)."""
    from mace.gui import cluster_api

    class St:
        root = tmp_path
        clients = {}

    class Srv:
        server_address = ("10.1.2.3", 8765)

    class Hd:
        state = St()
        server = Srv()
        headers = {}
        out = None

        def _json(self, obj, status=200):
            self.out = (status, obj)

        def _error(self, status, msg):
            self.out = (status, {"error": msg})
    monkeypatch.setenv("PATH", str(tmp_path))
    query._which_cache.clear()
    h = Hd()
    cluster_api.get(h, "job", {"id": ["1"]})
    assert h.out == (403, {"error": cluster_api.NETWORK_OFF})
    cluster_api.get(h, "software", {})                # it would run the login profile's module command
    assert h.out == (403, {"error": cluster_api.NETWORK_OFF})
    for route in ("scan", "estimate", "predict", "job/live"):
        cluster_api.post(h, route, {})
        assert h.out == (403, {"error": cluster_api.NETWORK_OFF}), route


def test_nothing_read_outside(tmp_path, monkeypatch):
    """F12-b: a job folder outside every allowed root is never opened (an
    open() spy in the process), and a symlink in the served folder is judged
    by its target."""
    from mace.cluster import jobinfo
    from mace.gui import cluster_api
    fs = fake(tmp_path)
    from cluster_fixtures import use_env
    use_env(monkeypatch, fs)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    root = tmp_path / "served"
    root.mkdir()
    far = tmp_path / "far"
    _opt_job(far, "f_opt")
    (far / ".f_opt.scratch").write_text(str(tmp_path / "scr"))
    fs.add_record(id="4200", state="RUNNING", name="f_opt", workdir=str(far))
    fs.update(job_text={"4200": _job_line("4200", "f_opt", far)})

    class St:
        clients = {}
    St.root = root
    svc = cluster_api.ClusterService(St())
    import builtins
    opened = []
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: opened.append(str(f)) or real_open(f, *a, **k))
    r = jobinfo.job("4200", path_policy=svc.path_policy, scratch_ok=svc.scratch_ok)
    monkeypatch.setattr(builtins, "open", real_open)
    assert r["access"]["out"] == "outside" and "progress" not in r or r.get("progress") is None
    assert not [p for p in opened if p.startswith(str(far))], opened
    (root / "link.out").symlink_to(far / "f_opt.out")
    assert svc.path_policy(os.path.realpath(root / "link.out")) == "outside"
    (home / "x").mkdir()
    (home / "x" / "a.out").write_text("x")
    (root / "link2.out").symlink_to(home / "x" / "a.out")
    assert svc.path_policy(os.path.realpath(root / "link2.out")) == "other_root"
    (home / ".ssh").mkdir()
    assert svc.path_policy(str(home / ".ssh")) == "outside"
