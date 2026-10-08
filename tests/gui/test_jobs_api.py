"""The Jobs panel's server side (PLAN_submit D6, D7, D25, D40): own jobs only,
folder scope, array grouping, finished events, the poll's rate, and a cancel
that never touches another user's job, a whole array (recorded by MACE or
not), a task range or the job MACE GUI runs in."""
import json
import threading
import time
import urllib.request

import pytest

import submit_util as SU


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    s = SU.Served(tmp_path_factory.mktemp("jobs"), {"sacct": {}})
    yield s
    s.stop()


def _jobs(srv, scope="folder", refresh=True):
    st, _, r = srv.req(f"api/jobs?scope={scope}&refresh={'1' if refresh else '0'}")
    assert st == 200
    return r


def test_scope_and_arrays(srv):
    d = srv.root / "j" / "decks"
    d.mkdir(parents=True)
    (d / "mat1.d12").write_text(SU.SP_D12)
    srv.fake.add_record(id="801", name="mat1", state="RUNNING", user=SU.USER, workdir=str(d), elapsed="1:02",
                        limit="2:00:00", cpus=32, nodelist="n007")
    srv.fake.add_record(id="802", name="elsewhere", state="PENDING", user=SU.USER, workdir="/somewhere/else",
                        reason="Priority")
    srv.fake.add_record(id="803", name="theirs", state="RUNNING", user="someone_else", workdir=str(d))
    srv.fake.add_record(id="900_0", array="900", task=0, name="mat1", state="RUNNING", user=SU.USER, workdir=str(d))
    srv.fake.add_record(id="900_1", array="900", task=1, name="mat1", state="PENDING", user=SU.USER, workdir=str(d))
    (d / "mat1-801.o").write_text("job log\n")
    time.sleep(15.2)                      # past the shared reader's 15 s refresh floor
    r = _jobs(srv)
    ids = {j["id"]: j for j in r["jobs"]}
    assert set(ids) == {"801", "900_0", "900_1"}
    assert ids["801"]["log_rel"] == "j/decks/mat1-801.o"
    assert ids["900_1"]["log_rel"] is None                 # no log yet: Show Log is off
    assert ids["801"]["deck_rel"] == "j/decks/mat1.d12" and ids["801"]["out_rel"] == "j/decks/mat1.out"
    assert ids["801"]["node"] == "n007" and ids["801"]["cpus"] == "32" and ids["801"]["limit"] == "2:00:00"
    assert ids["900_1"]["array_id"] == "900" and ids["900_1"]["task"] == "1"
    assert r["counts"] == {"running": 2, "pending": 1}
    r = _jobs(srv, "all")
    assert "802" in {j["id"] for j in r["jobs"]} and "803" not in {j["id"] for j in r["jobs"]}
    for jid in ("801", "802", "803", "900_0", "900_1"):
        srv.fake.remove_record(jid)


def test_cancel_guards(srv):
    d = srv.root / "c"
    d.mkdir()
    srv.fake.add_record(id="7100", name="other", state="PENDING", user="someone_else", workdir=str(d))
    srv.fake.add_record(id="7200_0", array="7200", task=0, name="t", state="PENDING", user=SU.USER, workdir=str(d))
    srv.fake.add_record(id="7200_1", array="7200", task=1, name="t", state="PENDING", user=SU.USER, workdir=str(d))
    srv.fake.add_record(id="7300", name="plain", state="PENDING", user=SU.USER, workdir=str(d))
    srv.clear_calls()
    st, _, r = srv.req("api/jobs/cancel", {"ids": ["7100", "7200", "7200_[0-1]", "x; rm", "7300", "7200_1"]})
    assert st == 200
    why = {x["id"]: x["reason"] for x in r["refused"]}
    assert "not one of your queued jobs" in why["7100"]
    assert "job array" in why["7200"] and "one by one" in why["7200"]          # unrecorded array: squeue -r
    assert "7200_[0-1]" in why and "x; rm" in why
    assert r["canceled"] == ["7300", "7200_1"]
    assert [c["argv"] for c in srv.calls("scancel")] == [["scancel", "7300", "7200_1"]]
    srv.fake.remove_record("7100")
    srv.fake.remove_record("7200")


def test_cancel_recorded_array_bare_id(srv):
    d = srv.root / "rec" / "decks"
    for n in ("a", "b"):
        srv.decks("rec/decks", [n])
    st, prev = srv.preview(["rec/decks"], options={"array": True})
    st, res = srv.run(prev["preview_id"])
    arr = res["job_ids"][0].split("_")[0]
    srv.clear_calls()
    st, _, r = srv.req("api/jobs/cancel", {"ids": [arr]})
    assert r["canceled"] == [] and "job array" in r["refused"][0]["reason"]
    st, _, r = srv.req("api/jobs/cancel", {"ids": [f"{arr}_1"]})
    assert r["canceled"] == [f"{arr}_1"]
    assert [c["argv"] for c in srv.calls("scancel")] == [["scancel", f"{arr}_1"]]
    srv.fake.remove_record(arr)


def test_cancel_refuses_the_guis_own_job(tmp_path):
    s = SU.Served(tmp_path, SLURM_JOB_ID="7777")
    try:
        s.fake.add_record(id="7777", name="mace-gui", state="RUNNING", user=SU.USER, workdir=str(s.root))
        st, _, r = s.req("api/jobs/cancel", {"ids": ["7777"]})
        assert r["canceled"] == [] and r["refused"][0]["reason"] == "This is the job MACE GUI runs in"
        assert not s.fake.calls("scancel")
        jobs = _jobs(s)["jobs"]
        assert jobs[0]["own_gui_job"] is True
    finally:
        s.stop()


def test_cancel_with_an_unreadable_database_asks_slurm(tmp_path):
    """materials.db that cannot be read: the array check falls back to SLURM,
    and a plain job SLURM lists as such still cancels."""
    s = SU.Served(tmp_path)
    try:
        (s.root / "materials.db").write_bytes(b"not a database" * 64)
        s.fake.add_record(id="7600", name="plain", state="RUNNING", user=SU.USER, workdir=str(s.root))
        s.clear_calls()
        st, _, r = s.req("api/jobs/cancel", {"ids": ["7600"]})
        assert st == 200 and r["canceled"] == ["7600"], r
        assert [c["argv"] for c in s.calls("scancel")] == [["scancel", "7600"]]
    finally:
        s.stop()


def test_without_slurm(tmp_path):
    root = tmp_path / "p"
    root.mkdir()
    proc, base, _ = SU.H.start_server(root, tmp_path / "xdg", MACE_GUI_NO_WARMUP="1", PATH="/usr/bin:/bin",
                                      MACE_GUI_TEST_HOOKS="1")
    try:
        import p4util as U
        st, _, r = U.req(base, "api/jobs?refresh=1")
        assert st == 200 and r["slurm"] is False and r["jobs"] == []
        st, _, caps = U.req(base, "api/submit/capabilities")
        assert caps["sbatch"] is False and caps["squeue"] is False
        st, _, r = U.req(base, "api/jobs/cancel", {"ids": ["5"]})
        assert r["refused"][0]["reason"] == "no SLURM on this machine"
    finally:
        proc.terminate()
        proc.wait(timeout=30)


class Events:
    def __init__(self, base):
        self.events = []
        req = urllib.request.Request(base + "api/events?interval=1", headers={"X-Mace-Token": SU.H.TOKEN})
        self.resp = urllib.request.urlopen(req, timeout=120)
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()

    def _read(self):
        ev = None
        try:
            for raw in self.resp:
                line = raw.decode().rstrip("\n")
                if line.startswith("event: "):
                    ev = line[7:]
                elif line.startswith("data: ") and ev:
                    self.events.append((ev, json.loads(line[6:])))
        except Exception:
            pass

    def of(self, kind):
        return [d for e, d in self.events if e == kind]

    def close(self):
        try:
            self.resp.close()
        except Exception:
            pass


def test_job_finished_event_once(tmp_path):
    s = SU.Served(tmp_path, {"sacct": {"8100": "FAILED"}})
    ev = Events(s.base)
    try:
        d = s.decks("f/decks", ["mat1"])
        s.fake.add_record(id="8100", name="mat1", state="RUNNING", user=SU.USER, workdir=str(d))
        _jobs(s)                                        # seen running
        s.fake.remove_record("8100")
        time.sleep(15.2)
        _jobs(s)                                        # gone: one sacct, one event
        _jobs(s, refresh=False)
        time.sleep(1.5)
        fin = ev.of("jobs.finished")
        assert len(fin) == 1 and fin[0]["id"] == "8100" and fin[0]["state"] == "FAILED"
        assert fin[0]["out_rel"] == "f/decks/mat1.out"
        assert len([c for c in s.fake.calls("sacct")]) == 1
    finally:
        ev.close()
        s.stop()


def test_poll_rate(tmp_path):
    """The panel polls every cluster.pollSeconds (30 here); calls inside the
    15 s refresh floor share one squeue."""
    s = SU.Served(tmp_path)
    try:
        st, _, r = s.req("api/settings", {"settings": {"cluster.pollSeconds": 30}})
        assert r["saved"] == ["cluster.pollSeconds"]
        s.clear_calls()
        t0 = time.time()
        while time.time() - t0 < 65:
            _jobs(s, refresh=False)
            time.sleep(5)
        polls = [c for c in s.fake.calls("squeue") if "-u" in c["argv"]]
        assert 2 <= len(polls) <= 4, len(polls)
        s.clear_calls()
        _jobs(s, refresh=True)
        _jobs(s, refresh=True)
        assert len([c for c in s.fake.calls("squeue") if "-u" in c["argv"]]) <= 1
        st, _, r = s.req("api/settings", {"settings": {"cluster.pollSeconds": 5}})
        assert "cluster.pollSeconds" in r["rejected"]
    finally:
        s.stop()


def test_show_log_of_an_array_task(tmp_path):
    """An array task's log is .mace_arrays/<array>/logs/<tag>-A_T.o (job_array.wrapper_text), not beside the deck."""
    import json as _json
    from mace.gui import jobs
    d = tmp_path / "decks"
    (d / ".mace_arrays" / "arr1" / "logs").mkdir(parents=True)
    (d / "mat1.d12").write_text(SU.SP_D12)
    (d / ".mace_arrays" / "arr1" / "submitted.json").write_text(_json.dumps(
        {"array_job_id": "950", "members": [{"task": 0, "job": "mat1", "dir": str(d)}]}))
    (d / ".mace_arrays" / "arr1" / "logs" / "scf-950_0.o").write_text("log\n")
    rows = [{"id": "950_0", "name": "wrapper", "state": "RUNNING", "elapsed": "0:01", "limit": "1:00", "nodes": "1",
             "cpus": "1", "reason": "n1", "workdir": str(d / ".mace_arrays" / "arr1")}]
    out = jobs.describe(tmp_path, rows, "folder", [], jobs.manifests_in(tmp_path, max_age=0))
    assert out[0]["name"] == "mat1" and out[0]["deck_rel"] == "decks/mat1.d12"
    assert out[0]["log_rel"] == "decks/.mace_arrays/arr1/logs/scf-950_0.o"


# ---------------------------------------------------------------- C19-b, C18.7: the jobs File History asks about
def test_queued_and_running_decks_are_protected(tmp_path):
    """saving.JOB_STATE_PROVIDER and setup.JOB_STATE_PROVIDER are the Jobs
    panel's: a deck whose job is queued or running is not written over, and
    the job's state is known for its output (the wavefunction copy waits)."""
    pytest.importorskip("ase", reason="the server reads and writes structures with ase")
    import shutil
    import guiharness as H
    import p4util as U
    s = SU.Served(tmp_path, {"sacct": {}})
    try:
        (s.root / "structures").mkdir()
        shutil.copyfile(H.EXAMPLES / "structures" / "MgO.cif", s.root / "structures" / "MgO.cif")
        d = s.decks("f/decks", ["mat9"])
        s.fake.add_record(id="8300", name="mat9", state="RUNNING", user=SU.USER, workdir=str(d))
        _jobs(s)
        st, _, r = U.post(s.base, "api/edit/open", {"path": "structures/MgO.cif"})
        assert st == 200, r
        st, _, out = U.post(s.base, "api/structure/save", {"session": r["session"], "path": "f/decks/mat9.d12", "overwrite": True})
        assert st == 409 and "mat9 is running with this deck (job 8300)" in out["error"], out
        assert (d / "mat9.d12").read_text() == SU.SP_D12
    finally:
        s.stop()


def test_job_state_providers_read_the_folder_rows(tmp_path):
    from pathlib import Path
    from mace.gui import jobs
    d = tmp_path / "decks"
    d.mkdir()
    (d / "mat1.d12").write_text(SU.SP_D12)
    j = jobs.Jobs(tmp_path, lambda *a: None)
    rows = [{"id": "77", "name": "mat1", "state": "PENDING", "elapsed": "0:00", "limit": "1:00:00", "nodes": "1",
             "cpus": "1", "reason": "Priority", "workdir": str(d)}]
    j.own.available = lambda: True
    j.own.get = lambda max_age=60, fresh=False: (rows, time.time())
    assert "mat1 is queued (job 77)" in j.deck_reason(d / "mat1.d12")
    assert j.deck_reason(d / "other.d12") is None
    assert j.out_state(d / "mat1.out") == "PENDING" and j.out_state(Path(d / "x.out")) is None
