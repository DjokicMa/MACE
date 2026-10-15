"""Two workflows in one folder keep their own job-script templates.

Each workflow's databases, storage and locks live in its own
``.mace_context_<id>/``, but the job-script templates the engine builds every
later step from were only kept in the folder's shared ``workflow_scripts/``,
named by step (``submitcrystal23_sp_2.sh``). A second workflow planned in the
same folder overwrote the first one's templates, and the first workflow's
later steps then ran with the second one's resources.

Run through the real CLI and the real job callback, with the fake SLURM of
tests/fake_slurm.py and a real CRYSTAL OPT output (corpus).
"""
import json
import re
import shutil

import pytest

from array_helpers import callback, rows
from conftest import find_data
from fake_slurm import FakeSlurm
from test_array_default_path_golden import _env, _mace


def _ok(p):
    assert p.returncode == 0, p.stdout[-5000:] + p.stderr[-3000:]
    return p


def _plan_with_sp_cores(template, workflow_id, ntasks):
    plan = json.loads(json.dumps(template))
    plan["workflow_id"] = workflow_id
    sp = plan["step_configurations"]["SP_2"]["slurm_config"]
    sp.setdefault("resources", {})["ntasks"] = ntasks
    for script in sp["scripts"].values():
        script["resources"]["ntasks"] = ntasks
    return plan


def _ntasks(script_text):
    return re.findall(r"#SBATCH --ntasks=(\d+)", script_text)


def _seed_plan(tmp_path, env, deck):
    """A plan to edit: quick-start's, built in another folder."""
    seed = tmp_path / "seed"
    (seed / "decks").mkdir(parents=True)
    shutil.copy(deck, seed / "decks" / "dia.d12")
    _ok(_mace(seed, env, "workflow", "--quick-start", "--d12-dir", "decks",
              "--workflow", "opt_sp"))
    (plan_file,) = (seed / "workflow_configs").glob("workflow_plan_*.json")
    return json.loads(plan_file.read_text())


def _write_plan(root, template, deck, name, wid, cores, **extra):
    (root / name).mkdir(parents=True)
    shutil.copy(deck, root / name / f"{name}.d12")
    plan = _plan_with_sp_cores(template, wid, cores)
    plan["input_directory"] = str(root / name)
    plan["input_files"] = {"d12": [str(root / name / f"{name}.d12")], "cif": []}
    plan.update(extra)
    (root / f"plan_{name}.json").write_text(json.dumps(plan))


@pytest.mark.parametrize("planned_before_fix", [False, True])
def test_second_workflow_does_not_change_the_first_ones_resources(tmp_path,
                                                                  planned_before_fix):
    """planned_before_fix: workflow one was planned by a MACE that kept no
    copy of its own (its context has no workflow_scripts/); it must keep
    progressing from the shared templates it was planned with."""
    out = find_data("OPT/1_dia_opt_rev1.out")
    deck = find_data("OPT/1_dia_opt_rev1.d12")
    fake = FakeSlurm(tmp_path)
    env = _env(tmp_path, fake)
    template = _seed_plan(tmp_path, env, deck)

    root = tmp_path / "shared_folder"
    _write_plan(root, template, deck, "one", "workflow_20991231_000001", 16)
    _write_plan(root, template, deck, "two", "workflow_20991231_000002", 32)
    before = len(fake.calls("sbatch"))
    _ok(_mace(root, env, "workflow", "--execute", "plan_one.json"))
    own = root / ".mace_context_workflow_20991231_000001" / "workflow_scripts"
    assert _ntasks((own / "submitcrystal23_sp_2.sh").read_text()) == ["16"]
    if planned_before_fix:
        shutil.rmtree(own)
    _ok(_mace(root, env, "workflow", "--execute", "plan_two.json"))
    assert len(fake.calls("sbatch")) == before + 2
    # the folder's shared copy is the latest workflow's, as before
    assert _ntasks((root / "workflow_scripts" / "submitcrystal23_sp_2.sh").read_text()) == ["32"]

    # workflow one's OPT finishes after workflow two was planned
    db = root / ".mace_context_workflow_20991231_000001" / "materials.db"
    (opt,) = [r for r in rows(db).values() if r["calc_type"] == "OPT"]
    work = shutil.os.path.dirname(opt["input_file"])
    stem = shutil.os.path.splitext(shutil.os.path.basename(opt["input_file"]))[0]
    shutil.copy(out, f"{work}/{stem}.out")
    script = open(f"{work}/{stem}.sh").read()
    exports = dict(re.findall(r'^export (MACE_WORKFLOW_ID|MACE_CONTEXT_DIR|MACE_ISOLATION_MODE)'
                              r'="([^"]*)"', script, re.M))
    assert exports["MACE_WORKFLOW_ID"] == "workflow_20991231_000001"
    fake.set_record(opt["slurm_job_id"], state="COMPLETED")
    before = len(fake.calls("sbatch"))
    p = _ok(callback(env, work, db, mode="completion",
                     extra_env=dict(exports, SLURM_JOB_ID=opt["slurm_job_id"])))
    sp_rows = [r for r in rows(db).values() if r["calc_type"].startswith("SP")]
    assert len(sp_rows) == 1, p.stdout[-4000:]
    (sb,) = fake.calls("sbatch")[before:]
    assert _ntasks(sb["script_text"]) == ["16"], (
        "workflow one's SP must run with its own 16 cores, not workflow two's 32")


def test_shared_mode_keeps_no_copy_of_its_own(tmp_path):
    from array_helpers import OPT_D12
    deck = tmp_path / "deck.d12"
    deck.write_text(OPT_D12)
    fake = FakeSlurm(tmp_path)
    env = _env(tmp_path, fake)
    template = _seed_plan(tmp_path, env, deck)
    root = tmp_path / "shared_folder"
    _write_plan(root, template, deck, "one", "workflow_20991231_000001", 16,
                isolation_mode="shared")
    _ok(_mace(root, env, "workflow", "--execute", "plan_one.json"))
    assert _ntasks((root / "workflow_scripts" / "submitcrystal23_sp_2.sh").read_text()) == ["16"]
    assert not list(root.glob(".mace_context_*/workflow_scripts"))


def test_submit_progress_keeps_its_own_copy(tmp_path):
    from array_helpers import OPT_D12
    fake = FakeSlurm(tmp_path)
    env = _env(tmp_path, fake)
    (tmp_path / "decks").mkdir()
    (tmp_path / "decks" / "a_opt.d12").write_text(OPT_D12)
    _ok(_mace(tmp_path, env, "submit", "--progress", "opt_sp", "decks/a_opt.d12"))
    (plan_file,) = (tmp_path / "decks" / "workflow_configs").glob("workflow_plan_*.json")
    wid = json.loads(plan_file.read_text())["workflow_id"]
    own = tmp_path / "decks" / f".mace_context_{wid}" / "workflow_scripts"
    shared = tmp_path / "decks" / "workflow_scripts"
    assert sorted(p.name for p in own.glob("*.sh")) == sorted(p.name for p in shared.glob("*.sh")) \
        == ["submitcrystal23_opt_1.sh", "submitcrystal23_sp_2.sh"]
    for p in own.glob("*.sh"):
        assert p.read_text() == (shared / p.name).read_text()
