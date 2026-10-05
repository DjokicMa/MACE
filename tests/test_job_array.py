"""mace.submission.job_array: grouping by request, the manifest and wrapper,
held submission, and the wrapper run under bash the way SLURM runs a task."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from fake_slurm import FakeSlurm
from mace.submission import job_array as ja
from mace.submission.job_array import (ArrayMember, ArrayRefusal, ArraySubmitError,
                                       plan_groups, prepare_arrays, resource_key,
                                       sbatch_directives, submit_prepared)

TEMPLATE = REPO_ROOT / "mace" / "submission" / "submitcrystal23.sh"


def script(directives, body="export JOB={job}\nexport scratch=$SCRATCH/crys23\necho run\n",
           job="x"):
    return "#!/bin/bash --login\n" + "".join(f"#SBATCH {d}\n" for d in directives) + \
        body.format(job=job)


BASE = ["-J {job}", "-o {job}-%J.o", "--cpus-per-task=1", "--ntasks=32",
        "-A mendoza_q # or general", "-N 1", "-t 7-00:00:00", "--mem-per-cpu=5G"]


def member(tmp_path, job, directives=None, body=None, subdir=None):
    d = tmp_path / (subdir or "decks")
    d.mkdir(parents=True, exist_ok=True)
    text = script([x.format(job=job) for x in (directives or BASE)],
                  **({"body": body} if body else {}), job=job)
    p = d / f"{job}.sh"
    p.write_text(text)
    return ArrayMember(dir=d, job=job, script=p)


def test_directives_leading_block_only():
    text = ("#!/bin/bash --login\n# a comment\n\n#SBATCH -t 1:00:00\n#SBATCH --mem=5G\n"
            "export JOB=x\n#SBATCH --mem=99G\n")
    assert [l.strip() for l in sbatch_directives(text)] == ["#SBATCH -t 1:00:00",
                                                            "#SBATCH --mem=5G"]
    assert ja.request_of(text) == {"--time": "1:00:00", "--mem": "5G"}


@pytest.mark.parametrize("a,b", [
    (["-t 7-00:00:00"], ["--time=7-00:00:00"]),
    (["-A mendoza_q # or general"], ["--account mendoza_q"]),
    (["-n 32", "-N 1"], ["--nodes=1", "--ntasks=32"]),
    (["-J a", "-o a-%J.o", "--mem=5G"], ["-J b", "-o b-%J.o", "--mem=5G"]),
])
def test_same_request(a, b):
    assert resource_key(script(a)) == resource_key(script(b))


@pytest.mark.parametrize("a,b", [
    (["--mem=5G"], ["--mem=5120M"]),
    (["--mem=5G"], ["--mem=5G", "-e err.e"]),
    (["--export=ALL"], ["--export=NONE"]),
    (["-t 1:00:00"], ["-t 2:00:00"]),
    (["--ntasks=32"], ["--ntasks=16"]),
])
def test_different_request(a, b):
    assert resource_key(script(a)) != resource_key(script(b))


def test_grouping_identical_requests_only(tmp_path):
    ms = [member(tmp_path, f"m{i}") for i in range(3)]
    ms.append(member(tmp_path, "big", BASE[:-1] + ["--mem-per-cpu=10G"]))
    arrays, singles = plan_groups(ms)
    assert [[m.job for m in g.members] for g in arrays] == [["m0", "m1", "m2"]]
    assert [(m.job, "normal job" in r) for m, r in singles] == [("big", True)]


@pytest.mark.parametrize("extra,what", [(["-D /elsewhere"], "working directory"),
                                        (["--chdir=/elsewhere"], "working directory"),
                                        (["--dependency=afterok:1"], "waits on")])
def test_single_path_members(tmp_path, extra, what):
    ms = [member(tmp_path, "m0"), member(tmp_path, "m1"), member(tmp_path, "m2", BASE + extra)]
    arrays, singles = plan_groups(ms)
    assert [m.job for m in arrays[0].members] == ["m0", "m1"]
    assert singles[0][0].job == "m2" and what in singles[0][1]


def test_refuses_a_nested_array(tmp_path):
    ms = [member(tmp_path, "m0"), member(tmp_path, "m1", BASE + ["--array=1-10"])]
    with pytest.raises(ArrayRefusal, match="job array itself"):
        plan_groups(ms)


def test_refuses_two_decks_sharing_a_scratch_dir(tmp_path):
    a = member(tmp_path, "same", subdir="d1")
    b = member(tmp_path, "same", subdir="d2")
    with pytest.raises(ArrayRefusal, match="same scratch"):
        plan_groups([a, b])


@pytest.mark.parametrize("bad", ["it's", "tab\there"])
def test_refuses_awkward_paths(tmp_path, bad):
    with pytest.raises(ArrayRefusal):
        plan_groups([member(tmp_path, "m0", subdir=bad), member(tmp_path, "m1")])


def test_late_sbatch_line_neither_groups_nor_reaches_the_wrapper(tmp_path):
    late = member(tmp_path, "late", body="export JOB={job}\nexport scratch=$SCRATCH/crys23\n"
                                         "#SBATCH --mem=99G\necho run\n")
    arrays, singles = plan_groups([member(tmp_path, "m0"), late])
    assert len(arrays) == 1 and not singles
    (p,) = prepare_arrays(arrays, tmp_path, "sp", max_tasks=10)
    assert "99G" not in p.wrapper.read_text()


def test_wrapper_manifest_and_throttle(tmp_path):
    ms = [member(tmp_path, f"m{i}") for i in range(4)]
    arrays, _ = plan_groups(ms)
    (p,) = prepare_arrays(arrays, tmp_path, "sp", max_running=2, max_tasks=10)
    text = p.wrapper.read_text()
    assert "#SBATCH --array=0-3%2" in text
    assert "#SBATCH -J mace_arr_sp_0" in text
    assert "#SBATCH -e" not in text
    assert "#SBATCH -A mendoza_q # or general" in text
    assert "-J m0" not in text and "m0-%J.o" not in text
    assert subprocess.run(["bash", "-n", str(p.wrapper)]).returncode == 0
    rows = ja.read_manifest(p.manifest)
    assert [(r["task"], r["job"], r["calc_id"]) for r in rows] == [
        (str(i), f"m{i}", "-") for i in range(4)]
    assert rows[1]["script"] == str((tmp_path / "decks" / "m1.sh").resolve())


def test_wrapper_has_its_own_stderr_only_when_the_request_does(tmp_path):
    ms = [member(tmp_path, f"m{i}", BASE + ["-e {job}.e"]) for i in range(2)]
    arrays, _ = plan_groups(ms)
    (p,) = prepare_arrays(arrays, tmp_path, "sp", max_tasks=10)
    text = p.wrapper.read_text()
    assert "#SBATCH -e " in text and "-%A_%a.e" in text
    assert "m0.e" not in text


def test_chunking_at_the_task_limit(tmp_path):
    ms = [member(tmp_path, f"m{i}") for i in range(5)]
    arrays, _ = plan_groups(ms)
    prepared = prepare_arrays(arrays, tmp_path, "sp", max_tasks=2)
    assert [len(p.members) for p in prepared] == [2, 2, 1]
    assert len({p.tag for p in prepared}) == 3


def test_task_limit_read_from_the_cluster(tmp_path, monkeypatch):
    fake = FakeSlurm(tmp_path / "slurm", {"max_array_tasks": 3})
    for k in ("PATH", "FAKE_SLURM_STATE", "FAKE_SLURM_LOG"):
        monkeypatch.setenv(k, fake.env[k])
    assert ja.max_tasks_per_array() == 3
    monkeypatch.setenv("PATH", "/nonexistent")
    assert ja.max_tasks_per_array() == ja.DEFAULT_MAX_TASKS


def _fake_env(monkeypatch, fake):
    for k in ("PATH", "FAKE_SLURM_STATE", "FAKE_SLURM_LOG", "FAKE_SLURM_HOOK_LOG"):
        monkeypatch.setenv(k, fake.env[k])
    monkeypatch.setenv("USER", "tester")


def test_held_submit_record_release_order(tmp_path, monkeypatch):
    fake = FakeSlurm(tmp_path / "slurm")
    _fake_env(monkeypatch, fake)
    ms = [member(tmp_path, f"m{i}") for i in range(3)]
    arrays, _ = plan_groups(ms)
    prepared = prepare_arrays(arrays, tmp_path, "sp", max_tasks=10)
    seen = []

    def record(subs):
        seen.append([c["argv"][:2] for c in fake.calls()])
        seen.append([p.task_id(i) for p in subs for i in range(len(p.members))])

    (p,) = submit_prepared(prepared, record)
    calls = [c["argv"] for c in fake.calls() if c["tool"] in ("sbatch", "scontrol")]
    assert calls == [["scontrol", "show", "assoc_mgr", "users=tester", "accounts=mendoza_q",
                      "flags=assoc"],
                     ["sbatch", "--hold", p.wrapper.name], ["scontrol", "release", "4242"]]
    assert ["scontrol", "release"] not in seen[0]
    assert seen[1] == ["4242_0", "4242_1", "4242_2"]
    data = json.loads((p.array_dir / "submitted.json").read_text())
    assert data["array_job_id"] == "4242" and data["released"] is True
    (sb,) = fake.calls("sbatch")
    assert sb["cwd"] == str(p.array_dir)


def test_partial_wave_is_removed_again(tmp_path, monkeypatch):
    fake = FakeSlurm(tmp_path / "slurm", {"fail_sbatch_call": 2})
    _fake_env(monkeypatch, fake)
    ms = [member(tmp_path, f"m{i}") for i in range(4)]
    arrays, _ = plan_groups(ms)
    prepared = prepare_arrays(arrays, tmp_path, "sp", max_tasks=2)
    recorded = []
    with pytest.raises(ArraySubmitError, match="QOSMaxSubmitJobPerUserLimit"):
        submit_prepared(prepared, recorded.append)
    assert not recorded
    tools = [(c["tool"], c["argv"][1:]) for c in fake.calls()
             if c["tool"] in ("sbatch", "scancel", "squeue", "scontrol")
             and c["argv"][1:3] != ["show", "assoc_mgr"]]
    assert tools[:2] == [("sbatch", ["--hold", prepared[0].wrapper.name]),
                         ("sbatch", ["--hold", prepared[1].wrapper.name])]
    assert ("squeue", ["-h", "-r", "-j", "4242", "-o", "%T,%r"]) in tools
    assert tools[-1] == ("scancel", ["4242"])
    assert not fake.calls("scontrol") or all(c["argv"][1] != "release" for c in fake.calls("scontrol"))


def test_no_sbatch_at_all_when_the_tasks_exceed_free_submit_slots(tmp_path, monkeypatch):
    fake = FakeSlurm(tmp_path / "slurm", {"assoc_mgr":
        "ClusterName=c Account=mendoza_q UserName=tester(1) ID=1\n"
        "    MaxJobs=520(3) MaxSubmitJobs=1000(998) MaxWallPJ="})
    _fake_env(monkeypatch, fake)
    ms = [member(tmp_path, f"m{i}") for i in range(3)]
    arrays, _ = plan_groups(ms)
    prepared = prepare_arrays(arrays, tmp_path, "sp", max_tasks=10)
    with pytest.raises(ArrayRefusal, match="MaxSubmitJobs is 1000 and 998"):
        submit_prepared(prepared)
    assert not fake.calls("sbatch")
    (q,) = [c for c in fake.calls("scontrol")]
    assert q["argv"] == ["scontrol", "show", "assoc_mgr", "users=tester",
                         "accounts=mendoza_q", "flags=assoc"]


def test_recording_failure_removes_the_held_arrays(tmp_path, monkeypatch):
    fake = FakeSlurm(tmp_path / "slurm")
    _fake_env(monkeypatch, fake)
    arrays, _ = plan_groups([member(tmp_path, f"m{i}") for i in range(2)])
    prepared = prepare_arrays(arrays, tmp_path, "sp", max_tasks=10)

    def record(_):
        raise RuntimeError("database is locked")

    with pytest.raises(ArraySubmitError, match="could not be recorded"):
        submit_prepared(prepared, record)
    assert [c["argv"] for c in fake.calls("scancel")] == [["scancel", "4242"]]


def test_abort_leaves_an_array_that_has_started(tmp_path, monkeypatch):
    fake = FakeSlurm(tmp_path / "slurm", {"records": [
        {"id": "4242_0", "array": "4242", "task": 0, "state": "RUNNING", "reason": "None"},
        {"id": "4242_1", "array": "4242", "task": 1, "state": "PENDING",
         "reason": "JobHeldUser"}]})
    _fake_env(monkeypatch, fake)
    assert ja.abort_held_arrays(["4242"]) == []
    assert not fake.calls("scancel")


def test_sbatch_shim(tmp_path):
    env, marker = ja.sbatch_shim()
    gen = tmp_path / "gen.sh"
    gen.write_text("echo '#!/bin/bash' > $1.sh\nsbatch $1.sh\n")
    res = subprocess.run(["bash", str(gen), "x"], cwd=tmp_path, env=env,
                         capture_output=True, text=True)
    assert ja.generator_outcome(res, marker) == ("shim", None)
    gen.write_text("echo '#!/bin/bash' > $1.sh\n")
    marker.unlink(missing_ok=True)
    res = subprocess.run(["bash", str(gen), "x"], cwd=tmp_path, env=env,
                         capture_output=True, text=True)
    assert ja.generator_outcome(res, marker) == ("generated", None)
    res = subprocess.CompletedProcess([], 0, "Submitted batch job 777\n", "")
    assert ja.generator_outcome(res, marker) == ("submitted", "777")
    ja.cleanup_shim(marker)
    assert not marker.parent.exists()
    assert not list(tmp_path.glob(".mace_array_shim*"))


# --- the wrapper under bash ---------------------------------------------------

def _wrapper_env(tmp_path, task, extra=None):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {"PATH": os.environ["PATH"], "HOME": str(home), "USER": "tester",
           "SLURM_ARRAY_TASK_ID": str(task), "SLURM_ARRAY_JOB_ID": "4242",
           "SLURM_JOB_ID": "9001", "SCRATCH": "", "LANG": "C"}
    env.update(extra or {})
    return env


def test_wrapper_runs_the_task_script_in_its_directory(tmp_path):
    stub = ("#!/bin/bash\necho \"submit=$SLURM_SUBMIT_DIR pwd=$(pwd) task=$MACE_ARRAY_TASK "
            "raw=$MACE_ARRAY_RAW_ID calc=$MACE_ARRAY_CALC_ID args=$#\"\nexit 3\n")
    ms = []
    for i in range(2):
        d = tmp_path / f"mat{i}"
        d.mkdir()
        p = d / f"mat{i}.sh"
        p.write_text("#!/bin/bash --login\n#SBATCH -J x\n#SBATCH --ntasks=1\n"
                     f"export JOB=mat{i}\nexport scratch=$SCRATCH/crys23\n" + stub)
        ms.append(ArrayMember(dir=d, job=f"mat{i}", script=p, calc_id=f"calc{i}"))
    arrays, _ = plan_groups(ms)
    (p,) = prepare_arrays(arrays, tmp_path, "sp", max_tasks=10)
    res = subprocess.run(["bash", str(p.wrapper)], env=_wrapper_env(tmp_path, 1),
                         capture_output=True, text=True, cwd=str(tmp_path))
    assert res.returncode == 3, res.stdout + res.stderr
    d1 = (tmp_path / "mat1").resolve()
    assert f"submit={d1} pwd={d1} task=4242_1 raw=9001 calc=calc1 args=0" in res.stdout
    link = tmp_path / "mat1" / "mat1-4242_1.o"
    assert link.is_symlink()
    assert os.readlink(link) == str(p.array_dir.resolve() / "logs" / f"{p.tag}-4242_1.o")
    res = subprocess.run(["bash", str(p.wrapper)], env=_wrapper_env(tmp_path, 7),
                         capture_output=True, text=True, cwd=str(tmp_path))
    assert res.returncode == 1 and "no manifest line for task 7" in res.stdout


def test_real_job_script_in_a_task_falls_back_to_its_own_scratch(tmp_path):
    """The generated submitcrystal23 script, run as task 1 with $SCRATCH
    empty (seen on a real node): the scratch guard falls back per material,
    so two tasks never share a directory."""
    decks = tmp_path / "decks"
    decks.mkdir()
    ms = []
    for name in ("matA", "matB"):
        subprocess.run(["bash", str(TEMPLATE), name], cwd=decks, check=True,
                       capture_output=True)
        (decks / f"{name}.d12").write_text("TEST\nEND\n")
        ms.append(ArrayMember(dir=decks, job=name, script=decks / f"{name}.sh"))
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    (fakebin / "mpirun").write_text("#!/bin/sh\necho \"CRYSTAL ran in $(pwd)\"\n"
                                    "touch fort.9\n")
    (fakebin / "mpirun").chmod(0o755)
    arrays, _ = plan_groups(ms)
    (p,) = prepare_arrays(arrays, tmp_path, "d12", max_tasks=10)
    env = _wrapper_env(tmp_path, 1, {"PATH": f"{fakebin}:{os.environ['PATH']}",
                                     "EBROOTCRYSTAL": str(tmp_path / "crystal"),
                                     "SLURM_NTASKS": "1", "MACE_LOCAL_RUN": "1",
                                     "TMPDIR": str(tmp_path / "tmpdir")})
    env.pop("SLURM_JOB_ID")  # no callback: the LOCAL RUN guard ends the script
    res = subprocess.run(["bash", str(p.wrapper)], env=env, capture_output=True, text=True,
                         cwd=str(tmp_path))
    assert res.returncode == 0, res.stdout + res.stderr
    want = f"{decks.resolve()}/.mace_scratch/crys23/matB"
    assert (decks / ".matB.scratch").read_text().strip() == want
    assert f"CRYSTAL ran in {want}" in (decks / "matB.out").read_text()
    assert not (decks / "matA.out").exists()


ASSOC = """ClusterName=c Account=acct UserName= Partition= Priority=0 ID=1
    ParentAccount=root(1) Lineage=/acct/ DefAssoc=No
    MaxJobs=520(98) MaxJobsAccrue= MaxSubmitJobs=1000(4362) MaxWallPJ=
ClusterName=c Account=acct UserName=tester(5) Partition= Priority=0 ID=2
    ParentAccount= Lineage=/acct/0-tester/ DefAssoc=Yes
    MaxJobs=520(49) MaxJobsAccrue= MaxSubmitJobs=1000(33) MaxWallPJ=
ClusterName=c Account=other UserName=tester(5) Partition= Priority=0 ID=3
    ParentAccount= Lineage=/other/0-tester/ DefAssoc=No
    MaxJobs=520(0) MaxJobsAccrue= MaxSubmitJobs=1000(990) MaxWallPJ=
ClusterName=c Account=acct UserName=someone(6) Partition= Priority=0 ID=4
    MaxJobs=520(0) MaxJobsAccrue= MaxSubmitJobs=0(0) MaxWallPJ=
"""


@pytest.mark.parametrize("account,expected", [("acct", (1000, 33)), (None, (1000, 33)),
                                              ("other", (1000, 990)), ("none", None)])
def test_submit_slots_are_the_users_own_association(account, expected):
    """The layout of `scontrol show assoc_mgr ... flags=assoc` (SLURM 26.05):
    the account line (more in use than its limit) and other users' lines
    are not this user's limit."""
    assert ja.parse_submit_slots(ASSOC, "tester", account) == expected


def test_combined_name_line_keeps_its_request(tmp_path):
    """sbatch reads every option of a line: one that names the job and asks
    for resources loses only the name in the wrapper, never the request."""
    combined = ["-J {job} -t 2-00:00:00 --mem=40G", "-o {job}-%J.o -e {job}.e",
                "--ntasks=32", "--constraint='a|b'"]
    ms = [member(tmp_path, f"m{i}", combined) for i in range(2)]
    arrays, singles = plan_groups(ms)
    assert len(arrays) == 1 and not singles
    assert dict(arrays[0].key)["--time"] == "2-00:00:00"
    (p,) = prepare_arrays(arrays, tmp_path, "sp", max_tasks=10)
    text = p.wrapper.read_text()
    assert "#SBATCH --time=2-00:00:00 --mem=40G\n" in text
    assert "#SBATCH --ntasks=32\n" in text and "#SBATCH --constraint='a|b'\n" in text
    assert "-J m0" not in text and "m0-%J.o" not in text and "m0.e" not in text
    # the wrapper asks for exactly what the grouping compared
    assert resource_key(text.replace("#SBATCH --array=0-1\n", "")) == arrays[0].key
    assert "--time=2-00:00:00" in ja.describe_request(p.group)


def test_wrapper_logs_are_relative_to_the_array_dir(tmp_path):
    """sbatch splits #SBATCH lines on whitespace: the log directives name the
    log relative to the array directory (where the wrapper is submitted), so
    a space in the path cannot break them; the links in the material
    directories still work."""
    base = tmp_path / "my decks"
    stub = "#!/bin/bash\necho ran in $(pwd)\n"
    ms = []
    for i in range(2):
        d = base / f"mat {i}"
        d.mkdir(parents=True)
        p = d / f"mat{i}.sh"
        p.write_text("#!/bin/bash --login\n#SBATCH -J x\n#SBATCH -o x-%J.o\n#SBATCH -e x.e\n"
                     f"#SBATCH --ntasks=1\nexport JOB=mat{i}\nexport scratch=$SCRATCH/crys23\n"
                     + stub)
        ms.append(ArrayMember(dir=d, job=f"mat{i}", script=p))
    arrays, _ = plan_groups(ms)
    (p,) = prepare_arrays(arrays, base, "sp", max_tasks=10)
    text = p.wrapper.read_text()
    for flag, ext in (("-o", "o"), ("-e", "e")):
        (line,) = [l for l in text.splitlines() if l.startswith(f"#SBATCH {flag} ")]
        assert line == f"#SBATCH {flag} logs/{p.tag}-%A_%a.{ext}"
    assert (p.array_dir / "logs").is_dir()
    res = subprocess.run(["bash", str(p.wrapper)], env=_wrapper_env(tmp_path, 1),
                         capture_output=True, text=True, cwd=str(p.array_dir))
    assert res.returncode == 0, res.stdout + res.stderr
    assert f"ran in {(base / 'mat 1').resolve()}" in res.stdout
    link = base / "mat 1" / "mat1-4242_1.o"
    assert os.readlink(link) == str(p.array_dir.resolve() / "logs" / f"{p.tag}-4242_1.o")
