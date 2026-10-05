"""Job ids of SLURM job arrays: parsing, ownership and task-scoped cancel.

Shapes below are the ones measured on SLURM 26.05 (MSU HPCC, 2026-10-04):
squeue folds pending tasks into ``A_[2-3%2]`` and, once the pending set has a
gap, ``A_[2,4-5%2]``; sacct without -P cuts the JobID column to ``A_[2+``; a
job submitted from inside an array task inherits every SLURM_ARRAY_* and
MACE_ARRAY_* variable but has a SLURM_JOB_ID of its own.
"""
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from fake_slurm import FakeSlurm
from mace.queue import slurm_ids
from mace.queue.slurm_ids import (
    Truncated, expand_queue_id, own_job_ids, own_rows, safe_scancel, safe_scancel_many,
    scancel_refusal, split_job_id, split_squeue_line, verified_array_task)


# Plain squeue lines in the formats MACE asks for (-o %i,%T,%S and %i,%T,%S,%j).
PLAIN_LINES = [
    "JOBID,STATE,START_TIME",
    "12345678,RUNNING,2026-10-04T22:32:58",
    "12345679,PENDING,N/A",
    "12345680,CONFIGURING,2026-10-04T22:40:00",
    "12345681,COMPLETING,2026-10-04T21:00:00",
]


@pytest.mark.parametrize("line", PLAIN_LINES)
def test_split_equals_plain_split_on_plain_lines(line):
    assert split_squeue_line(line, 3) == line.split(",")[:2] + [",".join(line.split(",")[2:])]
    assert split_squeue_line(line, 3)[:2] == line.split(",")[:2]


def test_split_keeps_comma_brackets_whole():
    assert split_squeue_line("4242_[1,4-6%3],PENDING,N/A", 3) == ["4242_[1,4-6%3]", "PENDING", "N/A"]
    # the old split(',') cut it, reading "4-6%3]" as the state
    assert "4242_[1,4-6%3],PENDING,N/A".split(",")[1] == "4-6%3]"


def test_split_ends_a_cut_task_list_at_the_cut_mark():
    assert split_squeue_line("4242_[1,3,5,...,PENDING,N/A", 3) == [
        "4242_[1,3,5,...", "PENDING", "N/A"]
    assert split_squeue_line("4242_[1,3,5,...,PENDING,N/A,a,b", 4) == [
        "4242_[1,3,5,...", "PENDING", "N/A", "a,b"]


def test_split_keeps_a_job_name_with_commas_whole():
    assert split_squeue_line("4242_[2,4-5%2],PENDING,N/A,my,job", 4) == [
        "4242_[2,4-5%2]", "PENDING", "N/A", "my,job"]


@pytest.mark.parametrize("token,expected", [
    ("4242_[2-3%2]", ["4242_2", "4242_3"]),
    ("4242_[123]", ["4242_123"]),
    ("4242_[1,4-6%3]", ["4242_1", "4242_4", "4242_5", "4242_6"]),
    ("18417197_[2,4-5%2]", ["18417197_2", "18417197_4", "18417197_5"]),
    ("4242_[0-6:2]", ["4242_0", "4242_2", "4242_4", "4242_6"]),
    ("4242_7", ["4242_7"]),
    ("12345678", ["12345678"]),
])
def test_expand_queue_id(token, expected):
    assert expand_queue_id(token) == expected


@pytest.mark.parametrize("token", ["4242_[1,3,5,...", "4242_[1,3,5,7,9,11...]", "18417197_[2+",
                                   "4242_[1,3", "4242_[x-y]"])
def test_expand_truncated_gives_marker(token):
    assert expand_queue_id(token) == Truncated(re.match(r"\d+", token).group(0))


def test_split_job_id():
    assert split_job_id("4242_7") == ("4242", "7")
    assert split_job_id("12345") == ("12345", None)
    assert split_job_id(None) == ("", None)


def test_slurm_env_only_adds_bitstr_len():
    base = {"PATH": "/bin", "HOME": "/h"}
    assert slurm_ids.slurm_env(base) == {"PATH": "/bin", "HOME": "/h", "SLURM_BITSTR_LEN": "0"}


TASK_ENV = {"SLURM_JOB_ID": "9999", "SLURM_ARRAY_JOB_ID": "4242", "SLURM_ARRAY_TASK_ID": "1",
            "SLURM_ARRAY_TASK_COUNT": "4", "MACE_ARRAY_RAW_ID": "9999",
            "MACE_ARRAY_TASK": "4242_1"}


def test_own_ids_of_the_task_itself():
    assert own_job_ids(TASK_ENV) == {"9999", "4242_1"}
    assert verified_array_task(TASK_ENV) == "4242_1"


def test_child_job_that_inherited_the_task_env_is_not_the_task():
    """Measured: the child keeps all SLURM_ARRAY_* (TASK_COUNT too) and
    MACE_ARRAY_*; only SLURM_JOB_ID is its own."""
    child = dict(TASK_ENV, SLURM_JOB_ID="10001")
    assert own_job_ids(child) == {"10001"}
    assert verified_array_task(child) is None


def test_array_task_without_mace_wrapper_is_not_trusted():
    """A non-MACE array task (no MACE_ARRAY_RAW_ID): its A_T is never taken
    as a row id, since its children carry the same SLURM_ARRAY_* variables."""
    env = {k: v for k, v in TASK_ENV.items() if not k.startswith("MACE_")}
    assert own_job_ids(env) == {"9999"}


def test_last_task_raw_id_equal_to_array_id():
    env = dict(TASK_ENV, SLURM_JOB_ID="4242", MACE_ARRAY_RAW_ID="4242")
    assert own_job_ids(env) == {"4242", "4242_1"}


def test_plain_job_own_ids():
    assert own_job_ids({"SLURM_JOB_ID": "777"}) == {"777"}
    assert own_job_ids({}) == set()


def _db(tmp_path):
    from mace.database.materials import MaterialDatabase
    return MaterialDatabase(str(tmp_path / "materials.db"))


def test_own_rows_finds_the_row_through_the_manifest(tmp_path):
    db = _db(tmp_path)
    calc_id = db.create_calculation(material_id="m", calc_type="SP", input_file="x.d12",
                                    work_dir=str(tmp_path))
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("# MACE job array manifest v1: task\tdir\tjob\tscript\tcalc_id\n"
                        f"0\t/d\ta\t/d/a.sh\tother\n1\t{tmp_path}\tm\t{tmp_path}/m.sh\t{calc_id}\n")
    env = dict(TASK_ENV, MACE_ARRAY_CALC_ID=calc_id, MACE_ARRAY_MANIFEST=str(manifest))
    assert [r["calc_id"] for r in own_rows(db, env)] == [calc_id]
    # a child job with the same inherited variables finds nothing
    assert own_rows(db, dict(env, SLURM_JOB_ID="10001")) == []
    # a manifest naming another calc for this task: not trusted
    env2 = dict(env, MACE_ARRAY_CALC_ID="other")
    assert own_rows(db, env2) == []


def test_scancel_refusals(tmp_path):
    db = _db(tmp_path)
    calc_id = db.create_calculation(material_id="m", calc_type="SP", work_dir=str(tmp_path))
    db.update_calculation_status(calc_id, "submitted", slurm_job_id="4242_3")
    assert "task range" in scancel_refusal("4242_[1-3]")
    assert "4242_3" in scancel_refusal("4242", db=db)
    assert scancel_refusal("4242_3", db=db) is None
    assert scancel_refusal("42", db=db) is None      # 42_ is not a prefix of 4242_3
    assert scancel_refusal("5000", db=db) is None


def test_scancel_refuses_untracked_array_ids_from_submitted_json(tmp_path):
    arr = tmp_path / ".mace_arrays" / "arr_x_0"
    arr.mkdir(parents=True)
    (arr / "submitted.json").write_text(json.dumps({"array_job_id": "4300"}))
    assert "4300" in scancel_refusal("4300", search_dirs=[tmp_path])
    # found from a material dir below the submission dir as well
    sub = tmp_path / "mat"
    sub.mkdir()
    assert scancel_refusal("4300", search_dirs=[sub]) is not None
    assert scancel_refusal("4301", search_dirs=[tmp_path]) is None


def test_safe_scancel_runs_exactly_scancel_id(tmp_path, monkeypatch):
    fake = FakeSlurm(tmp_path)
    monkeypatch.setenv("PATH", fake.env["PATH"])
    monkeypatch.setenv("FAKE_SLURM_STATE", fake.env["FAKE_SLURM_STATE"])
    monkeypatch.setenv("FAKE_SLURM_LOG", fake.env["FAKE_SLURM_LOG"])
    db = _db(tmp_path)
    calc_id = db.create_calculation(material_id="m", calc_type="SP", work_dir=str(tmp_path))
    db.update_calculation_status(calc_id, "submitted", slurm_job_id="4242_3")
    assert safe_scancel("4242_3", db=db).returncode == 0
    assert safe_scancel("5000", db=db).returncode == 0
    refused = safe_scancel("4242", db=db)
    assert refused.returncode == 1 and "refused" in refused.stderr
    res, ref = safe_scancel_many(["5001", "4242", "4242_1"], db=db)
    assert [r[0] for r in ref] == ["4242"]
    assert [c["argv"] for c in fake.calls("scancel")] == [
        ["scancel", "4242_3"], ["scancel", "5000"], ["scancel", "5001", "4242_1"]]


# --- .o id lookup -----------------------------------------------------------

def _old_find_slurm_job_id(out_file_path):
    """completion_checker.find_slurm_job_id before job arrays (reference)."""
    base = out_file_path.stem
    parent = out_file_path.parent
    for f in parent.glob(f"{base}-*.o"):
        m = re.search(r'-(\d+)\.o$', f.name)
        if m:
            return m.group(1)
    for f in parent.glob(f"{base}.o*"):
        m = re.search(r'\.o(\d+)$', f.name)
        if m:
            return m.group(1)
    for f in parent.glob(f"*{base}*.o"):
        m = re.search(r'[-_](\d{6,})\.o$', f.name)
        if m:
            return m.group(1)
    return None


@pytest.mark.parametrize("names", [
    ["X-12345678.o"], ["X.o12345678"], ["pre_X_12345678.o"], [],
    ["X-111.o", "X-222.o"], ["X-abc.o", "X.o99"],
])
def test_find_job_id_unchanged_without_array_logs(tmp_path, names):
    from mace.completion_checker import find_slurm_job_id
    out = tmp_path / "X.out"
    out.write_text("")
    for n in names:
        (tmp_path / n).write_text("")
    assert find_slurm_job_id(out) == _old_find_slurm_job_id(out)


def test_find_job_id_never_reads_a_task_index(tmp_path):
    from mace.completion_checker import find_slurm_job_id
    out = tmp_path / "X.out"
    out.write_text("")
    (tmp_path / "X-4242_123456.o").write_text("")
    assert _old_find_slurm_job_id(out) == "123456"  # the old misreading
    assert find_slurm_job_id(out) == "4242_123456"


@pytest.mark.parametrize("newer", ["X-111.o", "X-4242_3.o"])
def test_find_job_id_newest_log_wins_when_a_task_log_exists(tmp_path, newer):
    from mace.completion_checker import find_slurm_job_id
    out = tmp_path / "X.out"
    out.write_text("")
    for name in ("X-111.o", "X-4242_3.o"):
        (tmp_path / name).write_text("")
    now = time.time()
    older = "X-4242_3.o" if newer == "X-111.o" else "X-111.o"
    os.utime(tmp_path / older, (now - 100, now - 100))
    os.utime(tmp_path / newer, (now, now))
    expect = "111" if newer == "X-111.o" else "4242_3"
    assert find_slurm_job_id(out) == expect


def test_find_job_id_through_a_task_log_link(tmp_path):
    from mace.completion_checker import find_slurm_job_id
    logs = tmp_path / ".mace_arrays" / "t" / "logs"
    logs.mkdir(parents=True)
    (logs / "t-4242_2.o").write_text("")
    out = tmp_path / "X.out"
    out.write_text("")
    os.symlink(logs / "t-4242_2.o", tmp_path / "X-4242_2.o")
    assert find_slurm_job_id(out) == "4242_2"


def test_running_jobs_expand_pending_brackets(tmp_path, monkeypatch):
    from mace.completion_checker import get_running_jobs
    fake = FakeSlurm(tmp_path, {"records": [
        {"id": "4242_0", "array": "4242", "task": 0, "state": "RUNNING"},
        {"id": "4242_2", "array": "4242", "task": 2, "state": "PENDING"},
        {"id": "4242_4", "array": "4242", "task": 4, "state": "PENDING"},
        {"id": "5000", "state": "RUNNING"}], "throttle": {"4242": 2}})
    for k in ("PATH", "FAKE_SLURM_STATE", "FAKE_SLURM_LOG"):
        monkeypatch.setenv(k, fake.env[k])
    assert get_running_jobs() == {"4242_0", "4242_2", "4242_4", "5000"}
    (call,) = fake.calls("squeue")
    assert call["argv"] == ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i"]
    assert call["env"] == {"SLURM_BITSTR_LEN": "0"}


# --- queue counts through the manager ---------------------------------------

def _manager(tmp_path, monkeypatch, records, **state):
    fake = FakeSlurm(tmp_path / "slurm", dict(records=records, **state))
    for k in ("PATH", "FAKE_SLURM_STATE", "FAKE_SLURM_LOG"):
        monkeypatch.setenv(k, fake.env[k])
    monkeypatch.chdir(tmp_path)
    from mace.queue.manager import EnhancedCrystalQueueManager
    mgr = EnhancedCrystalQueueManager(d12_dir=str(tmp_path), enable_tracking=True,
                                      db_path=str(tmp_path / "materials.db"))
    return mgr, fake


ARRAY_RECORDS = [
    {"id": "4242_0", "array": "4242", "task": 0, "state": "RUNNING"},
    {"id": "4242_1", "array": "4242", "task": 1, "state": "PENDING"},
    {"id": "4242_4", "array": "4242", "task": 4, "state": "PENDING"},
    {"id": "4242_5", "array": "4242", "task": 5, "state": "PENDING"},
    {"id": "777", "state": "PENDING"},
]


def test_check_queue_counts_every_task(tmp_path, monkeypatch):
    mgr, fake = _manager(tmp_path, monkeypatch, ARRAY_RECORDS, throttle={"4242": 3})
    assert mgr.check_queue() == (1, 4)
    (call,) = fake.calls("squeue")
    assert "4242_[1,4-5%3],PENDING,N/A" in subprocess.run(
        call["argv"], env=dict(fake.env, SLURM_BITSTR_LEN="0"),
        capture_output=True, text=True).stdout


def test_check_queue_status_keeps_plain_rows_and_sees_bracket_tasks(tmp_path, monkeypatch):
    mgr, fake = _manager(tmp_path, monkeypatch, ARRAY_RECORDS, throttle={"4242": 3})
    calc_id = mgr.db.create_calculation(material_id="m", calc_type="SP",
                                        input_file=str(tmp_path / "m.d12"), work_dir=str(tmp_path))
    mgr.db.update_calculation_status(calc_id, "submitted", slurm_job_id="777")
    mgr.check_queue_status()
    assert mgr.db.get_calculation(calc_id)["status"] == "submitted"
    assert not fake.calls("sacct"), "a queued job is never looked up in sacct"


def test_monitor_health_counts_tasks(tmp_path, monkeypatch):
    fake = FakeSlurm(tmp_path / "slurm", {"records": ARRAY_RECORDS, "throttle": {"4242": 3}})
    for k in ("PATH", "FAKE_SLURM_STATE", "FAKE_SLURM_LOG"):
        monkeypatch.setenv(k, fake.env[k])
    from mace.queue.monitor import MaterialMonitor
    mon = MaterialMonitor.__new__(MaterialMonitor)
    health = mon._check_queue_status()
    assert health["total_jobs"] == 5
    assert health["by_status"] == {"RUNNING": 1, "PENDING": 4}


@pytest.mark.parametrize("logs,expected", [
    (["X-4242_3.o"], "4242_3"),
    (["X-12345.o"], "12345"),
])
def test_populate_scan_reads_task_ids(tmp_path, logs, expected):
    """The completed-job scan records A_T from a task's <name>-A_T.o (it
    used to drop it: 'A_T' is not all digits). Real CRYSTAL output."""
    import shutil
    from conftest import find_data
    from mace.database.populate_completed_jobs import scan_for_completed_calculations
    src = find_data("SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out")
    shutil.copy(src, tmp_path / "X.out")
    for name in logs:
        (tmp_path / name).write_text("")
    (calc,) = scan_for_completed_calculations(tmp_path)
    assert calc["slurm_job_id"] == expected
