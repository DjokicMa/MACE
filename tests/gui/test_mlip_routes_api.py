"""Settings › ML potentials through the real `mace gui`: the status, the plan,
an install as a server task through mlip_install (a fake base Python whose
venv's pip only logs: no network), its log, Clean's tiers refreshed without
a restart, a cancel that removes the half-built folder, one install at a
time, the license accept the fetch needs, and the route security every
other task route has (POST only, X-Requested-With, the token)."""
import json
import os
import sys
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preopt"))
import preopt_fixtures as F  # noqa: E402


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mlipapi")
    root, xdg = H.examples_copy(tmp), tmp / "xdg"
    base, fenv = F.make_fake_base_python(tmp / "fakepy", tmp / "pip.log")
    ptmp = tmp / "ptmp"
    ptmp.mkdir()
    env = dict(MACE_GUI_NO_WARMUP="1", MACE_GUI_MAX_WORKERS="2", PATH="/usr/bin:/bin", TMPDIR=str(ptmp),
               XDG_DATA_HOME=str(tmp / "data"), FAKE_PIP_SLEEP="orb-models:60", **fenv)
    proc, url, _ = H.start_server(root, xdg, **env)
    yield {"base": url, "tmp": tmp, "python": str(base), "xdg": xdg, "log": tmp / "pip.log"}
    proc.terminate()
    proc.wait(timeout=20)


def result_of(base, st, r, timeout=120):
    if st != 202:                     # answered at once: done (200) or failed (422)
        return st, r
    tid = r["task"]["id"]
    end = time.time() + timeout
    while time.time() < end:
        st2, _, r2 = U.req(base, f"api/tasks/{tid}/result")
        if st2 == 200:
            return 200, r2
        if st2 == 409 and r2.get("task", {}).get("state") in ("queued", "running"):
            time.sleep(0.1)
            continue
        return st2, r2
    raise AssertionError("the task did not finish")


def task_record(base, tid):
    st, _, r = U.req(base, "api/tasks")
    return next(t for t in r["tasks"] if t["id"] == tid)


def test_status_before_anything_is_set_up(served):
    st, _, r = U.req(served["base"], "api/mlip/status")
    assert st == 200 and r["python"] is None and r["host_ok"] and r["loopback"]
    assert r["default_dir"] == str(served["tmp"] / "data" / "mace-tool" / "mlip" / "venv")
    ids = [m["id"] for m in r["models"]]
    assert "acesuit-mace-mpa-0-medium" in ids
    assert all(m["status"] == ("needs_ack" if m["academic"] else "needs_setup") for m in r["models"])
    assert {p["package"]: p["license"] for p in r["packages"]}["mace-torch"] == "MIT"
    assert r["default_backends"] == ["mace-torch"]


def test_routes_are_post_only_and_need_the_header_and_the_token(served):
    base = served["base"]
    st, _, r = U.req(base, "api/mlip/install")
    assert st == 405
    st, _, r = U.post(base, "api/mlip/install", {"backends": ["chgnet"]}, headers={"X-Requested-With": "x"})
    assert st == 403
    st, _, r = U.post(base, "api/mlip/install", {"backends": ["chgnet"]}, token=False)
    assert st in (401, 403)
    st, _, r = U.post(base, "api/mlip/plan", {"backends": "chgnet"})
    assert st == 400
    st, _, r = U.post(base, "api/mlip/plan", {"backends": ["not-a-package"]})
    assert st == 400 and "unknown package" in r["error"]


def test_plan_install_log_and_tiers_refresh(served):
    base = served["base"]
    st, _, plan = U.post(base, "api/mlip/plan", {"backends": ["chgnet"], "python": served["python"]})
    assert st == 200 and plan["problems"] == [] and plan["pins"] == ["chgnet==0.4.2"] and plan["torch"] == "cpu"
    st, _, r = U.post(base, "api/mlip/install", {"backends": ["chgnet"], "python": served["python"],
                                                  "fetch": "acesuit-mace-mpa-0-medium"})
    assert st == 400 and "accept its weights license" in r["error"]
    st, _, r = U.post(base, "api/mlip/install", {"backends": ["chgnet"], "python": served["python"]})
    tid = r["task"]["id"] if st == 202 else None
    st, r = result_of(base, st, r)
    assert st == 200 and r["ok"], r
    venv_py = str(Path(plan["dir"]) / "bin" / "python")
    assert r["python"] == venv_py
    if tid:
        log = task_record(base, tid)["log"]
        assert "== Upgrading pip" in log and "--index-url https://download.pytorch.org/whl/cpu" in log
    calls = [json.loads(x) for x in served["log"].read_text().splitlines()]
    assert not any(c["pythonpath"] for c in calls)
    rec = json.loads((served["xdg"] / "mace" / "mlip.json").read_text())
    assert rec["python"] == venv_py
    # Clean's tiers see it without a restart
    end = time.time() + 60
    ok = False
    while time.time() < end and not ok:
        st, _, cm = U.req(base, "api/clean/methods")
        ok = any(m["id"] == "mlip:chgnet-0.3.0" and m["available"] for m in cm.get("methods", []))
        time.sleep(0.3)
    assert ok
    st, _, s = U.req(base, "api/mlip/status")
    assert s["python"] == venv_py and s["source"].startswith("recorded in")
    assert {m["id"]: m["status"] for m in s["models"]}["chgnet-0.3.0"] == "ready"


def test_cancel_removes_the_half_built_folder_and_one_install_at_a_time(served):
    base = served["base"]
    d = served["tmp"] / "orbenv"
    st, _, r = U.post(base, "api/mlip/install", {"dir": str(d), "backends": ["orb-models"], "python": served["python"]})
    assert st == 202, r
    tid = r["task"]["id"]
    end = time.time() + 30
    while time.time() < end and "orb-models" not in task_record(base, tid)["log"]:
        time.sleep(0.2)
    st2, _, r2 = U.post(base, "api/mlip/install", {"dir": str(served["tmp"] / "other"), "backends": ["chgnet"],
                                                    "python": served["python"]})
    assert st2 == 409 and r2["task"] == tid
    U.post(base, f"api/tasks/{tid}/cancel", {})
    end = time.time() + 30
    while time.time() < end and task_record(base, tid)["state"] not in ("canceled", "failed", "done"):
        time.sleep(0.2)
    assert task_record(base, tid)["state"] == "canceled"
    assert not d.exists()


def test_fetch_and_record_need_what_they_need(served):
    base = served["base"]
    st, _, r = U.post(base, "api/mlip/fetch", {"id": "acesuit-mace-mpa-0-medium"})
    assert st == 400 and "accept" in r["error"]
    st, _, r = U.post(base, "api/mlip/fetch", {"id": "no-such-model", "accept_license": True})
    assert st == 400 and "no ML model" in r["error"]
    st, _, r = U.post(base, "api/mlip/record", {"python": str(served["tmp"] / "nope" / "python")})
    st, r = result_of(base, st, r)
    assert st == 422 and "Not recorded" in r["error"]
    st, _, r = U.post(base, "api/mlip/record", {})
    assert st == 400


def test_academic_and_gated_models_in_the_status_and_their_fetch_gates(served):
    base = served["base"]
    st, _, s = U.req(base, "api/mlip/status")
    rows = {m["id"]: m for m in s["models"]}
    off = rows["acesuit-mace-off24-medium"]
    assert off["academic"] and off["status"] == "needs_ack" and off["academic_ack"] is None and off["fetchable"]
    uma = rows["meta-uma-s-1p1"]
    assert uma["gated"]["page"] == "https://huggingface.co/facebook/UMA" and uma["env"] == "fairchem"
    assert s["default_dirs"]["fairchem"].endswith("mace-tool/mlip/venv-fairchem")
    assert {p["package"]: p["env"] for p in s["packages"]}["fairchem-core"] == "fairchem"
    st, _, r = U.post(base, "api/mlip/fetch", {"id": "acesuit-mace-off24-medium", "accept_license": True})
    assert st == 400 and "academic, non-commercial use only" in r["error"]       # the license accept is not enough
    st, _, r = U.post(base, "api/mlip/plan", {"backends": ["mace-torch", "fairchem-core"], "python": served["python"]})
    assert st == 400 and "separate environments" in r["error"]
