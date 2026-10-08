"""`mace site check` (PLAN_G G6, A6) through the real mace_cli with the fake
SLURM: a per-item report, a dry submission with sbatch --test-only through
the real generator, an exit code (0 ok, 1 warnings, 2 failed) and its
temporary folder removed every time."""
import json
from pathlib import Path

import pytest

from test_site_generic_cluster import _cluster
from test_site_jobscript import _env, _mace, _toml

SINFO = "".join(f"n{i:03d}|idle|2|16|1|32|256000|epyc,ib|(null)|short*\n" for i in range(3))
ITEMS = ["Settings", "SLURM", "Account", "Partition", "QOS/constraint", "Resources", "CRYSTAL", "MPI launch",
         "Completion Python", "Scratch", "Job script"]


def _check(tmp, env, *args):
    tmpdir = tmp / "checktmp"
    tmpdir.mkdir(exist_ok=True)
    env = dict(env, TMPDIR=str(tmpdir))
    r = _mace(tmp, env, "site", "check", *args)
    return r, tmpdir


def _items(r):
    return {i["item"]: i for i in json.loads(r.stdout)["items"]}


def test_ok_cluster(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    fake.update(sinfo=SINFO)
    _toml(tmp_path, '[slurm]\naccount = "grp_x"\n')
    r, tmpdir = _check(tmp_path, env, "--json")
    rep = json.loads(r.stdout)
    assert [i["item"] for i in rep["items"]] == ITEMS
    items = _items(r)
    assert items["Job script"]["status"] == "ok", items["Job script"]
    assert "Job 5999 to start at" in items["Job script"]["detail"]
    assert items["Account"]["status"] == "ok" and items["Partition"]["status"] == "ok"
    # fits; the default partition's 2-day MaxTime is below the 7-day default walltime: a warning
    assert items["Resources"]["status"] == "warn" and "fits short" in items["Resources"]["detail"]
    assert "allows at most 2-00:00:00" in items["Resources"]["detail"]
    calls = [c for c in fake.calls("sbatch") if "--test-only" in c["argv"]]
    assert len(calls) == 2 and all("#SBATCH -A grp_x" in c["script_text"] for c in calls)
    assert not any(c for c in fake.calls("sbatch") if "--test-only" not in c["argv"])      # no real job
    assert list(tmpdir.iterdir()) == []                                                    # removed
    assert r.returncode == rep["exit"]
    text, _ = _check(tmp_path, env)
    assert "  [ ok ] Account" in text.stdout and "mace site check:" in text.stdout


@pytest.mark.parametrize("stderr,fix", [
    ("Batch job submission failed: Invalid account or account/partition combination specified", "[slurm] account"),
    ("Batch job submission failed: Invalid qos specification", "[slurm] qos"),
    ("Batch job submission failed: Requested node configuration is not available", "[resources]"),
    ("Batch job submission failed: Invalid feature specification", "[slurm] constraint"),
])
def test_test_only_refusals_name_the_key(tmp_path, stderr, fix):
    env, fake = _cluster(tmp_path, owner=False)
    fake.update(fail_test_only_with=stderr)
    r, tmpdir = _check(tmp_path, env, "--json")
    assert r.returncode == 2
    item = _items(r)["Job script"]
    assert item["status"] == "fail" and fix in item["fix"], item
    assert list(tmpdir.iterdir()) == []
    text, _ = _check(tmp_path, env)
    assert "[FAIL] Job script" in text.stdout and "Fix: " in text.stdout


def test_no_slurm_but_pbs(tmp_path):
    stubs = tmp_path / "pbs"
    stubs.mkdir()
    (stubs / "qsub").write_text("#!/bin/bash\nexit 0\n")
    (stubs / "qsub").chmod(0o755)
    env = _env(tmp_path, PATH=f"{stubs}:/usr/bin:/bin")
    r, tmpdir = _check(tmp_path, env, "--json")
    assert r.returncode == 2
    item = _items(r)["SLURM"]
    assert item["status"] == "fail" and "qsub found" in item["detail"]
    assert "PBS/Torque and LSF support is planned for MACE 1.2 or later" in item["detail"]
    assert _items(r)["Account"]["status"] == "skip"
    assert list(tmpdir.iterdir()) == []


def test_not_a_member_and_missing_partition(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    _toml(tmp_path, '[slurm]\naccount = "grp_z"\npartition = "nope"\n')
    r, _ = _check(tmp_path, env, "--json")
    items = _items(r)
    assert items["Account"]["status"] == "fail" and '[slurm] account = "grp_x"' in items["Account"]["fix"]
    assert items["Partition"]["status"] == "fail" and "nope does not exist" in items["Partition"]["detail"]


def test_partition_and_account_options(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    r, _ = _check(tmp_path, env, "--json", "--partition", "long", "--account", "grp_y", "--kind", "scf")
    calls = [c for c in fake.calls("sbatch") if "--test-only" in c["argv"]]
    assert len(calls) == 1
    assert "#SBATCH -A grp_y" in calls[0]["script_text"] and "#SBATCH -p long" in calls[0]["script_text"]


def test_required_qos_missing_from_associations(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    fake.update(qos=["normal", "debug"])
    _toml(tmp_path, '[slurm]\nqos = "high"\n')
    r, _ = _check(tmp_path, env, "--json")
    item = _items(r)["QOS/constraint"]
    assert item["status"] == "fail" and "high is not one of yours" in item["detail"]
    assert '[slurm] qos = "debug"' in item["fix"] or '[slurm] qos = "normal"' in item["fix"]


def test_too_big_for_the_nodes(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    fake.update(sinfo="".join(f"n{i}|idle|1|8|2|16|64000|x|(null)|short*\n" for i in range(2)))
    r, _ = _check(tmp_path, env, "--json")
    item = _items(r)["Resources"]
    assert item["status"] == "fail" and "largest node has 8 cores" in item["detail"]
    assert "[resources] ntasks = 8" in item["fix"]


def test_group_writable_group_file_fails_item_one(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    g = tmp_path / "lab"
    g.mkdir()
    (g / "site.toml").write_text('[slurm]\naccount = "grp_x"\n')
    (g / "site.toml").chmod(0o666)
    env["MACE_SITE_GROUP_CONFIG"] = str(g / "site.toml")
    r, tmpdir = _check(tmp_path, env, "--json")
    item = _items(r)["Settings"]
    assert item["status"] == "fail" and str(g / "site.toml") in item["detail"] and "0666" in item["detail"]
    assert _items(r)["Job script"]["status"] == "skip"
    assert list(tmpdir.iterdir()) == []


def test_srun_mpi_list(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    _toml(tmp_path, '[mpi]\nlauncher = "srun"\nsrun_mpi = "pmix"\n')
    r, _ = _check(tmp_path, env, "--json")
    assert _items(r)["MPI launch"]["status"] == "fail"                 # the fake lists none, pmi2, cray_shasta
    _toml(tmp_path, '[mpi]\nlauncher = "srun"\nsrun_mpi = "pmi2"\n')
    r, _ = _check(tmp_path, env, "--json")
    assert _items(r)["MPI launch"]["status"] == "ok"


def test_no_sbatch_keeps_the_script(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    r, tmpdir = _check(tmp_path, env, "--json", "--no-sbatch")
    rep = json.loads(r.stdout)
    kept = Path(rep["kept"])
    assert (kept / "site_check.sh").is_file() and (kept / "site_check_band.sh").is_file()
    assert not [c for c in fake.calls("sbatch")]
    import shutil
    shutil.rmtree(kept)


def test_json_schema_is_stable(tmp_path):
    env, fake = _cluster(tmp_path, owner=False)
    r, _ = _check(tmp_path, env, "--json")
    rep = json.loads(r.stdout)
    assert set(rep) == {"items", "summary", "kept", "exit"}
    assert set(rep["summary"]) == {"ok", "warn", "fail", "skip"}
    for i in rep["items"]:
        assert set(i) == {"item", "status", "detail", "fix"} and i["status"] in ("ok", "warn", "fail", "skip")
