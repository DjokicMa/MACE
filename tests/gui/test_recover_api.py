"""Recover in the Jobs panel: the preview is `mace recover --dry-run` (the
same text a shell prints), the run is the argv the preview named (`mace
recover --yes`), both the CLI itself; refused on a network bind
(test_submit_api.test_refused_on_network_bind). Real failed output from the
test/ corpus (an OOM-killed OPT)."""
import re
import shutil
import subprocess
import sys

import pytest

import submit_util as SU
from conftest import find_data


def _seed(root, job_id="12345", name="3_dia3_opt_rev1", settings=None):
    w = root / "w"
    w.mkdir(parents=True, exist_ok=True)
    shutil.copy(find_data("OPT/3_dia3_opt_rev1.d12"), w / f"{name}.d12")
    (w / f"{name}.sh").write_text(find_data("OPT/3_dia3_opt_rev1.sh").read_text().replace("3_dia3_opt_rev1", name))
    shutil.copy(find_data("FAILED_QA/3_dia3_opt_killed_signal9_oom.out"), w / f"{name}.out")
    code = f"""
import sys
sys.path.insert(0, {str(SU.REPO)!r})
from mace.database.materials import MaterialDatabase
db = MaterialDatabase({str(root / 'materials.db')!r})
if not db.get_material({name!r}):
    db.create_material({name!r}, "C")
cid = db.create_calculation(material_id={name!r}, calc_type="OPT", input_file={str(w / (name + '.d12'))!r},
                            work_dir={str(w)!r}, settings={settings!r})
db.update_calculation_status(cid, "failed", slurm_job_id={job_id!r}, output_file={str(w / (name + '.out'))!r})
print(cid)
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    return w, r.stdout.split()[-1]


def _norm(text):
    return re.sub(r"_recovery_\d{8}_\d{6}", "_recovery_<T>", text)


@pytest.fixture()
def srv(tmp_path):
    s = SU.Served(tmp_path, {"sacct": {}})
    yield s
    s.stop()


def test_preview_is_the_dry_run_and_the_run_uses_its_argv(srv):
    w, cid = _seed(srv.root)
    srv.fake.update(sacct_rows={"": [f"12345|3_dia3_opt_rev1|OUT_OF_MEMORY|00:10:00|{w}"]})
    st, _, rec = srv.req("api/jobs/recent")
    assert st == 200
    (job,) = rec["jobs"]
    assert job["recoverable"] is True and job["tracked"]["calc_id"] == cid

    srv.clear_calls()
    st, prev = srv.task_result(srv.req("api/jobs/recover/preview", {"ids": ["12345"], "wait": 120}))
    assert st == 200, prev
    base = ["mace", "recover", "--db", "materials.db", "--calc-id", cid]
    assert prev["argv"] == base + ["--dry-run", "--json"]
    assert prev["run_argv"] == base + ["--yes", "--json"]
    assert prev["ready"] == 1 and prev["stale"] == 0
    assert "Dry run: nothing is changed or submitted." in prev["text"]
    assert "+#SBATCH --mem-per-cpu=7GB" in prev["text"]
    assert not srv.calls("sbatch")
    # the same text `mace recover --dry-run` prints in a shell
    twin = SU.twin_submit(srv.tmp, srv.fake, srv.root, ["--no-banner", "recover", "--db", "materials.db",
                                                         "--calc-id", cid, "--dry-run"])
    assert twin.returncode == 0, twin.stdout + twin.stderr
    assert _norm(prev["text"]) == _norm(twin.stdout)
    assert not srv.calls("sbatch")

    st, res = srv.task_result(srv.req("api/jobs/recover/run", {"preview_id": prev["preview_id"], "wait": 120}))
    assert st == 200, res
    assert res["argv"] == prev["run_argv"]
    assert res["exit"] == 0 and [r["status"] for r in res["results"]] == ["submitted"], res
    (sb,) = srv.calls("sbatch")
    assert "#SBATCH --mem-per-cpu=7GB" in sb["script_text"]
    # a preview is used once
    st, _, again = srv.req("api/jobs/recover/run", {"preview_id": prev["preview_id"]})
    assert st == 409 and "preview again" in again["error"]


def test_recover_failed_scopes_the_folder_and_untracked_ids_are_refused(srv):
    _, cid = _seed(srv.root)
    st, prev = srv.task_result(srv.req("api/jobs/recover/preview", {"all": True, "wait": 120}))
    assert st == 200, prev
    assert prev["argv"] == ["mace", "recover", "--dir", ".", "--dry-run", "--json"]
    # the run resubmits what the preview showed, not what fails meanwhile
    assert prev["run_argv"] == ["mace", "recover", "--dir", ".", "--calc-id", cid, "--yes", "--json"]
    assert prev["ready"] == 1
    st, _, r = srv.req("api/jobs/recover/preview", {"ids": ["999"]})
    assert st == 400 and "Not tracked" in r["error"]
    st, _, r = srv.req("api/jobs/recover/preview", {"ids": ["1; rm -rf"]})
    assert st == 400
    assert not srv.calls("sbatch")


def test_recover_failed_runs_only_the_previewed_calculations(srv):
    _, cid = _seed(srv.root)
    st, prev = srv.task_result(srv.req("api/jobs/recover/preview", {"all": True, "wait": 120}))
    assert st == 200 and prev["ready"] == 1, prev
    # another calculation fails after the preview was shown
    _, later = _seed(srv.root, job_id="12346", name="later_opt")
    st, res = srv.task_result(srv.req("api/jobs/recover/run", {"preview_id": prev["preview_id"], "wait": 120}))
    assert st == 200, res
    assert [(r["calc_id"], r["status"]) for r in res["results"]] == [(cid, "submitted")], res
    assert len(srv.calls("sbatch")) == 1


def test_a_held_workflow_step_is_not_offered_for_recovery(srv):
    """The Jobs panel runs `mace recover`, which skips the steps of a paused
    or canceled workflow (the Workflows view refuses the same)."""
    import json
    wid = "workflow_20261015_000000_abcd"
    _, cid = _seed(srv.root, settings={"workflow_id": wid, "workflow_step": 1})
    ctl = srv.root / "workflow_configs" / f"workflow_control_{wid[len('workflow_'):]}.json"
    ctl.parent.mkdir(parents=True, exist_ok=True)
    ctl.write_text(json.dumps({"version": 1, "workflow_id": wid, "state": "canceled"}))
    for body in ({"all": True}, {"ids": ["12345"]}):
        st, prev = srv.task_result(srv.req("api/jobs/recover/preview", dict(body, wait=120)))
        assert st == 200, prev
        assert prev["ready"] == 0 and prev["stale"] == 0, prev
        assert f"workflow {wid} is canceled: resume it first" in prev["text"]
    st, res = srv.task_result(srv.req("api/jobs/recover/run", {"preview_id": prev["preview_id"], "wait": 120}))
    assert not srv.calls("sbatch")
