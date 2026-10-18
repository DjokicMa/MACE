"""The Workflows view's actions (GUI_WORKFLOW_DESIGN.md sections 4.7 and 5,
Phase 5) on a real `mace gui` with the logging fake SLURM. Every action is a
CLI command run in the workflow's folder:

POST workflow/control         pause / resume / cancel  -> mace workflow --pause|--resume|--cancel ID
                              (cancel lists the jobs first; the confirmed list must still be the list)
POST workflow/recover/preview -> mace recover --db DB --calc-id ID --dry-run --json
POST workflow/recover/run     -> the same with --yes
POST workflow/process         -> mace workflow --process ID
GET  workflow/clone           the builder's draft of an existing workflow

Writing routes need root_gen; anything that changes SLURM's state is refused
while the server is open to the network."""
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import guiharness as H
import submit_util as SU

TEST_DATA = H.REPO_ROOT / "test"
DECK = TEST_DATA / "OPT" / "1_dia_opt_rev1.d12"
OUT = TEST_DATA / "OPT" / "1_dia_opt_rev1.out"
OOM = TEST_DATA / "FAILED_QA" / "3_dia3_opt_killed_signal9_oom.out"


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    if not (DECK.exists() and OOM.exists()):
        pytest.skip("test/ data corpus not present")
    s = SU.Served(tmp_path_factory.mktemp("wfact"), MACE_SITE_CONFIG="none")
    yield s
    s.stop()


def _gen(srv):
    return srv.req("api/info")[2]["root_gen"]


def _task(srv, route, body, timeout=600):
    r = srv.req(f"api/{route}", {**body, "wait": 60})
    if r[0] in (400, 403, 404, 409):
        return r[0], r[2]
    return srv.task_result(r, timeout=timeout)


def _decks(srv, folder, names=("dia0", "dia1")):
    d = srv.root / folder
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        shutil.copy(DECK, d / f"{n}.d12")
    return d


UI_DRAFT = {"label": "acts", "template": "opt_sp", "sequence": ["OPT", "SP"],
            "steps": {"OPT_1": {"resources": {"ntasks": 12}},
                      "SP_2": {"answers": {"functional": "PBE0"}, "method": {"functional": "PBE0", "dispersion": False}}},
            "execution": {"max_jobs": "40"}}


def _launch(srv, folder, names=("dia0", "dia1")):
    """A workflow started from the builder (preview, then run)."""
    _decks(srv, folder, names)
    draft = {"inputs": {"type": "d12", "files": [f"{folder}/{n}.d12" for n in names]}, "label": "acts",
             "template": "opt_sp", "sequence": ["OPT", "SP"],
             "steps": {"OPT_1": {"resources": {"ntasks": 12}},
                       "SP_2": {"method": {"functional": "PBE0", "dispersion": False}}},
             "execution": {"max_jobs": 40}}
    st, prev = _task(srv, "workflow/preview", {"root_gen": _gen(srv), "draft": draft, "ui_draft": UI_DRAFT})
    assert st == 200 and prev["code"] == 0, prev.get("cli_log")
    st, res = srv.task_result(srv.req("api/workflow/run", {"root_gen": _gen(srv), "preview_id": prev["preview_id"]}))
    assert st == 200 and res["code"] == 0, res
    wid = res["workflow_id"]
    return wid, srv.root / folder / f".mace_context_{wid}" / "materials.db"


def _rows(db):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        return {r["calc_id"]: dict(r) for r in con.execute("SELECT * FROM calculations")}
    finally:
        con.close()


def _status(srv, folder, wid):
    st, _, r = srv.req(f"api/workflow/status?id={wid}&folder={folder}&eta=0")
    assert st == 200, r
    return r


def _node(st, mid, step):
    m = [x for x in st["materials"] if x["material_id"] == mid][0]
    return [n for n in m["nodes"] if n["step"] == step][0]


def _ctl(srv, folder, wid, action, **extra):
    return _task(srv, "workflow/control", {"root_gen": _gen(srv), "folder": folder, "id": wid, "action": action, **extra})


# ---------------------------------------------------------------- guards
def test_writing_routes_need_root_gen_and_a_real_workflow(srv):
    wid, _ = _launch(srv, "guards")
    for route, body in (("workflow/control", {"action": "pause"}), ("workflow/process", {}),
                        ("workflow/recover/preview", {"calc_id": "x"})):
        st, _, r = srv.req(f"api/{route}", {"folder": "guards", "id": wid, **body})
        assert st == 400 and "root_gen" in r["error"], (route, r)
        st, _, r = srv.req(f"api/{route}", {"root_gen": int(_gen(srv)) + 5, "folder": "guards", "id": wid, **body})
        assert st == 409 and "folder changed" in r["error"], (route, r)
    st, r = _ctl(srv, "guards", "workflow_; rm -rf ~", "pause")
    assert st == 400 and "not a workflow id" in r["error"]
    st, r = _ctl(srv, "../..", wid, "pause")
    assert st == 403
    st, r = _ctl(srv, "guards", "workflow_20990101_000000_ffff", "pause")
    assert st == 404
    st, r = _ctl(srv, "guards", wid, "explode")
    assert st == 400 and "pause, resume or cancel" in r["error"]
    assert not list((srv.root / "guards" / "workflow_configs").glob("workflow_control_*"))


# ---------------------------------------------------------------- pause / resume / cancel
def test_pause_resume_and_cancel_with_the_exact_jobs(srv):
    wid, db = _launch(srv, "ctl")
    ctl_file = srv.root / "ctl" / "workflow_configs" / f"workflow_control_{wid[len('workflow_'):]}.json"
    st, r = _ctl(srv, "ctl", wid, "pause")
    assert st == 200 and r["ok"] and r["state"] == "paused", r
    assert r["argv"] == ["mace", "workflow", "--pause", wid, "--json"]
    assert json.loads(ctl_file.read_text())["state"] == "paused"
    assert _status(srv, "ctl", wid)["control"]["state"] == "paused"
    st, r = _ctl(srv, "ctl", wid, "resume")
    assert st == 200 and r["ok"] and r["state"] == "active" and r["steps_started"] == 0, r
    assert json.loads(ctl_file.read_text())["state"] == "active"

    ids = sorted(x["slurm_job_id"] for x in _rows(db).values())
    srv.clear_calls()
    # 1. the list: nothing canceled yet
    st, r = _ctl(srv, "ctl", wid, "cancel")
    assert st == 200 and r["confirm"] is True and r["squeue"] is True, r
    assert sorted(j["job_id"] for j in r["jobs"]) == ids
    assert {j["material_id"] for j in r["jobs"]} == {"dia0", "dia1"}
    assert not srv.calls("scancel") and json.loads(ctl_file.read_text())["state"] == "active"
    # 2. a confirmation of another list: refused, nothing canceled
    st, r = _ctl(srv, "ctl", wid, "cancel", confirm=True, jobs=ids[:1])
    assert st == 200 and r["ok"] is False and r["changed"] is True, r
    assert sorted(j["job_id"] for j in r["jobs"]) == ids
    assert not srv.calls("scancel") and json.loads(ctl_file.read_text())["state"] == "active"
    # 3. the confirmed list: scancel exactly those
    st, r = _ctl(srv, "ctl", wid, "cancel", confirm=True, jobs=ids)
    assert st == 200 and r["ok"] and sorted(r["canceled"]) == ids, r
    (call,) = srv.calls("scancel")
    assert sorted(a for a in call["argv"] if a.isdigit()) == ids
    assert json.loads(ctl_file.read_text())["state"] == "canceled"
    st = _status(srv, "ctl", wid)
    assert st["control"]["state"] == "canceled"


# ---------------------------------------------------------------- process now
def test_process_now_starts_the_step_a_missing_callback_did_not(srv):
    wid, db = _launch(srv, "proc", names=("dia0",))
    (opt,) = _rows(db).values()
    work = Path(opt["input_file"]).parent
    shutil.copy(OUT, work / f"{Path(opt['input_file']).stem}.out")
    srv.fake.remove_record(opt["slurm_job_id"])
    con = sqlite3.connect(db)
    with con:
        con.execute("UPDATE calculations SET status='completed', completed_at='2026-10-15T08:00:00', output_file=? "
                    "WHERE calc_id=?", (str(work / f"{Path(opt['input_file']).stem}.out"), opt["calc_id"]))
    con.close()
    assert _node(_status(srv, "proc", wid), "dia0", "SP_2")["state"] in ("waiting", "stale")
    # paused: refused by the CLI, nothing submitted
    _ctl(srv, "proc", wid, "pause")
    srv.clear_calls()
    st, r = _task(srv, "workflow/process", {"root_gen": _gen(srv), "folder": "proc", "id": wid})
    assert st == 200 and r["ok"] is False and "paused" in r["error"], r
    assert not srv.calls("sbatch")
    _ctl(srv, "proc", wid, "resume")                     # resume starts it by itself
    assert len(srv.calls("sbatch")) == 1
    # a second time: nothing left to start
    st, r = _task(srv, "workflow/process", {"root_gen": _gen(srv), "folder": "proc", "id": wid})
    assert st == 200 and r["ok"] and r["steps_started"] == 0, r
    assert r["argv"] == ["mace", "workflow", "--process", wid, "--json"]
    assert _node(_status(srv, "proc", wid), "dia0", "SP_2")["state"] == "queued"


# ---------------------------------------------------------------- recover
def test_recover_one_failed_node_with_preview(srv):
    wid, db = _launch(srv, "rec", names=("dia0", "dia1"))
    rows = _rows(db)
    opt = [x for x in rows.values() if x["material_id"] == "dia0"][0]
    work = Path(opt["input_file"]).parent
    out = work / f"{Path(opt['input_file']).stem}.out"
    shutil.copy(OOM, out)
    srv.fake.remove_record(opt["slurm_job_id"])
    con = sqlite3.connect(db)
    with con:
        con.execute("UPDATE calculations SET status='failed', output_file=? WHERE calc_id=?", (str(out), opt["calc_id"]))
    con.close()
    node = _node(_status(srv, "rec", wid), "dia0", "OPT_1")
    assert node["state"] == "failed" and node["calc_id"] == opt["calc_id"]

    # only a calculation of this workflow
    st, r = _task(srv, "workflow/recover/preview", {"root_gen": _gen(srv), "folder": "rec", "id": wid,
                                                    "calc_id": "not_one_of_its_calcs"})
    assert st == 400 and "not a failed step of" in r["error"], r
    other = [x for x in rows.values() if x["material_id"] == "dia1"][0]
    st, r = _task(srv, "workflow/recover/preview", {"root_gen": _gen(srv), "folder": "rec", "id": wid,
                                                    "calc_id": other["calc_id"]})
    assert st == 400 and "not a failed step of" in r["error"], r

    srv.clear_calls()
    st, p = _task(srv, "workflow/recover/preview", {"root_gen": _gen(srv), "folder": "rec", "id": wid,
                                                    "calc_id": opt["calc_id"]})
    assert st == 200 and p["ready"] == 1, p
    assert p["argv"][:2] == ["mace", "recover"] and "--dry-run" in p["argv"]
    assert p["argv"][p["argv"].index("--calc-id") + 1] == opt["calc_id"]
    assert p["run_argv"][-2:] == ["--yes", "--json"] and "--dry-run" not in p["run_argv"]
    assert "keeps workflow " + wid in p["text"]
    assert not srv.calls("sbatch")
    # paused: the run is refused (a paused workflow submits no recovery job)
    _ctl(srv, "rec", wid, "pause")
    st, r = _task(srv, "workflow/recover/run", {"root_gen": _gen(srv), "preview_id": p["preview_id"]})
    assert st == 409 and "paused" in r["error"], r
    _ctl(srv, "rec", wid, "resume")
    st, p = _task(srv, "workflow/recover/preview", {"root_gen": _gen(srv), "folder": "rec", "id": wid,
                                                    "calc_id": opt["calc_id"]})
    st, r = _task(srv, "workflow/recover/run", {"root_gen": _gen(srv), "preview_id": p["preview_id"]})
    assert st == 200 and [x["status"] for x in r["results"]] == ["submitted"], r
    assert len(srv.calls("sbatch")) == 1
    node = _node(_status(srv, "rec", wid), "dia0", "OPT_1")
    assert node["attempts"] == 2 and node["state"] == "queued", node
    # single use
    st, r = _task(srv, "workflow/recover/run", {"root_gen": _gen(srv), "preview_id": p["preview_id"]})
    assert st == 409 and "preview again" in r["error"]


# ---------------------------------------------------------------- clone
def test_clone_of_a_builder_workflow_is_its_draft(srv):
    wid, _ = _launch(srv, "cl1", names=("dia0",))
    st, _, r = srv.req(f"api/workflow/clone?id={wid}&folder=cl1")
    assert st == 200 and r["available"] and r["source"] == "builder", r
    assert r["kind"] == "d12" and r["targets"] == ["cl1/dia0.d12"]
    d = r["draft"]
    assert d["sequence"] == ["OPT", "SP"] and d["template"] == "opt_sp"
    assert d["steps"]["SP_2"]["answers"] == {"functional": "PBE0"}
    assert d["steps"]["OPT_1"]["resources"] == {"ntasks": 12}
    assert d["label"] == "acts-copy" and d["labelAuto"] is False
    st, _, s = srv.req(f"api/workflow/status?id={wid}&folder=cl1&eta=0")
    assert s["clone"]["available"] and s["clone"]["source"] == "builder"


def _cli(srv, folder, *args):
    env = {k: v for k, v in srv.env.items()}
    env.update(PATH=srv.env["PATH"], MACE_SITE_CONFIG="none", XDG_CACHE_HOME=str(srv.tmp / "xdg" / "cache"))
    p = subprocess.run([sys.executable, str(SU.MACE_CLI), "--no-banner", *args], cwd=str(srv.root / folder),
                       env=env, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return p.stdout


def test_clone_of_a_cli_spec_and_of_a_bare_plan(srv):
    # a spec written by hand and launched from the command line: no builder draft
    d = _decks(srv, "cl2", ("dia0",))
    (d / "spec.json").write_text(json.dumps({
        "version": 1, "label": "byhand", "inputs": {"type": "d12", "files": ["dia0.d12"]}, "template": "opt_sp",
        "steps": {"OPT_1": {"resources": {"walltime": "4:00:00"}},
                  "SP_2": {"overrides": {"method_modifications": {"new_functional": "HSE06", "use_dispersion": True}}}}}))
    w = json.loads(_cli(srv, "cl2", "workflow", "--spec", "spec.json", "--write-plan", "--json"))
    _cli(srv, "cl2", "submit", "--progress-plan", w["plan_file"], "dia0.d12")
    st, _, r = srv.req(f"api/workflow/clone?id={w['workflow_id']}&folder=cl2")
    assert st == 200 and r["available"] and r["source"] == "spec", r
    sp = r["draft"]["steps"]["SP_2"]
    assert sp["method"] == {"functional": "HSE06", "dispersion": True}
    assert sp["answers"] == {"functional": "HSE06", "dispersion": "Yes"}
    assert r["draft"]["steps"]["OPT_1"]["resources"] == {"walltime": "4:00:00"}
    assert r["draft"]["label"] == "byhand-copy"

    # `mace submit --progress`: a plan and no spec; the steps, structures and job resources
    _decks(srv, "cl3", ("dia0",))
    _cli(srv, "cl3", "submit", "--progress", "opt_sp", "dia0.d12")
    (plan,) = (srv.root / "cl3" / "workflow_configs").glob("workflow_plan_*.json")
    wid = json.loads(plan.read_text())["workflow_id"]
    st, _, r = srv.req(f"api/workflow/clone?id={wid}&folder=cl3")
    assert st == 200 and r["available"] and r["source"] == "plan", r
    assert r["targets"] == ["cl3/dia0.d12"] and r["draft"]["sequence"] == ["OPT", "SP"]
    assert r["notes"] and "command line" in r["notes"][0]
    assert r["draft"]["steps"]["OPT_1"]["resources"]["ntasks"] == 32

    # the structures are gone: no clone, and why
    (srv.root / "cl3" / "dia0.d12").unlink()
    st, _, r = srv.req(f"api/workflow/clone?id={wid}&folder=cl3")
    assert st == 200 and r["available"] is False and "dia0.d12" in r["reason"], r
    st, _, s = srv.req(f"api/workflow/status?id={wid}&folder=cl3&eta=0")
    assert s["clone"]["available"] is False and s["clone"]["reason"]


# ---------------------------------------------------------------- network
def test_slurm_changing_actions_refused_on_network_bind(tmp_path):
    import re
    if not DECK.exists():
        pytest.skip("test/ data corpus not present")
    fake = SU.FakeSlurm(tmp_path / "fs")
    env = SU.fake_env(fake, tmp_path)
    env.pop("XDG_CACHE_HOME")
    env["MACE_SITE_CONFIG"] = "none"
    root = tmp_path / "proj"
    (root / "net").mkdir(parents=True)
    shutil.copy(DECK, root / "net" / "dia0.d12")
    # a workflow started from the command line
    cli_env = dict(env, XDG_CACHE_HOME=str(tmp_path / "cache"))
    p = subprocess.run([sys.executable, str(SU.MACE_CLI), "--no-banner", "submit", "--progress", "opt_sp", "dia0.d12"],
                       cwd=str(root / "net"), env=cli_env, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    (plan,) = (root / "net" / "workflow_configs").glob("workflow_plan_*.json")
    wid = json.loads(plan.read_text())["workflow_id"]
    port = H.free_port()
    henv = H.env(tmp_path / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1", **env)
    proc = H.ServerProcess([sys.executable, str(SU.MACE_CLI), "--no-banner", "gui", "--dir", str(root), "--port",
                            str(port), "--token", H.TOKEN, "--host", "0.0.0.0", "--insecure-network",
                            "--allow-host", "127.0.0.1"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, env=henv, start_new_session=True)
    try:
        base = None
        for _ in range(60):
            m = re.search(r"(http://[^/\s]+:\d+/s/[A-Za-z0-9_-]+/)", proc.stdout.readline())
            if m:
                base = m.group(1).replace("0.0.0.0", "127.0.0.1")
                break
        assert base
        s = SU.Served.__new__(SU.Served)
        s.base, s.proc, s.root, s.fake, s.tmp = base, proc, root, fake, tmp_path
        gen = _gen(s)
        st, _, lst = s.req("api/workflow/list")
        assert lst["network"] is True
        for action in ("resume", "cancel"):
            st, _, r = s.req("api/workflow/control", {"root_gen": gen, "folder": "net", "id": wid, "action": action})
            assert st == 409 and "open to the network" in r["error"], (action, r)
        st, _, r = s.req("api/workflow/process", {"root_gen": gen, "folder": "net", "id": wid})
        assert st == 409 and "open to the network" in r["error"]
        st, _, r = s.req("api/workflow/recover/run", {"root_gen": gen, "preview_id": "wfrec-0"})
        assert st == 409 and "open to the network" in r["error"]
        # pausing changes nothing in SLURM: allowed
        st, r = _task(s, "workflow/control", {"root_gen": gen, "folder": "net", "id": wid, "action": "pause"})
        assert st == 200 and r["ok"] and r["state"] == "paused", r
        assert not [c for c in fake.calls() if c["tool"] in ("scancel",)]
        assert len([c for c in fake.calls() if c["tool"] == "sbatch"]) == 1
    finally:
        proc.terminate()
        proc.wait(timeout=30)
