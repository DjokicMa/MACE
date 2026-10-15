"""A plan with execution_settings.auto_progression false is followed.

The executor printed "Auto progression: disabled" and every job callback went
on to submit the next steps anyway. With it off, a callback now records the
completion and stops, printing how to continue; `mace engine --action
process` (run by hand) submits the next steps.

Through the real CLI and job callback, fake SLURM, real OPT output (corpus).
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from array_helpers import MACE_CLI, callback, rows
from conftest import REPO_ROOT, find_data
from fake_slurm import FakeSlurm
from test_array_default_path_golden import _env, _mace
from test_workflow_isolation_scripts import _ok, _seed_plan, _write_plan

WID = "workflow_20991231_000004"


def _complete_opt(root, env, fake, out):
    db = root / f".mace_context_{WID}" / "materials.db"
    (opt,) = [r for r in rows(db).values() if r["calc_type"] == "OPT"]
    work = Path(opt["input_file"]).parent
    stem = Path(opt["input_file"]).stem
    shutil.copy(out, work / f"{stem}.out")
    exports = dict(re.findall(r'^export (MACE_WORKFLOW_ID|MACE_CONTEXT_DIR|MACE_ISOLATION_MODE)'
                              r'="([^"]*)"', (work / f"{stem}.sh").read_text(), re.M))
    fake.set_record(opt["slurm_job_id"], state="COMPLETED")
    p = _ok(callback(env, work, db, mode="completion",
                     extra_env=dict(exports, SLURM_JOB_ID=opt["slurm_job_id"])))
    return db, opt, p


@pytest.mark.parametrize("auto", [True, False])
def test_callback_follows_auto_progression(tmp_path, auto):
    out = find_data("OPT/1_dia_opt_rev1.out")
    deck = find_data("OPT/1_dia_opt_rev1.d12")
    fake = FakeSlurm(tmp_path)
    env = _env(tmp_path, fake)
    template = _seed_plan(tmp_path, env, deck)
    template.setdefault("execution_settings", {})["auto_progression"] = auto
    root = tmp_path / "wf"
    _write_plan(root, template, deck, "one", WID, 16)
    _ok(_mace(root, env, "workflow", "--execute", "plan_one.json"))

    before = len(fake.calls("sbatch"))
    db, opt, p = _complete_opt(root, env, fake, out)
    r = rows(db)
    assert r[opt["calc_id"]]["status"] == "completed", p.stdout[-4000:]
    sps = [x for x in r.values() if x["calc_type"] == "SP"]
    if auto:
        assert len(sps) == 1 and len(fake.calls("sbatch")) == before + 1, p.stdout[-4000:]
        return
    assert not sps and len(fake.calls("sbatch")) == before, p.stdout[-4000:]
    assert "auto_progression" in p.stdout and "mace engine --action process" in p.stdout

    # continuing by hand, as the message says
    q = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "engine", "--action",
                        "process", "--db", str(db), "--work-dir", str(root)],
                       cwd=str(root), env=env, capture_output=True, text=True, timeout=600)
    assert q.returncode == 0, q.stdout[-4000:] + q.stderr[-3000:]
    sps = [x for x in rows(db).values() if x["calc_type"] == "SP"]
    assert len(sps) == 1 and sps[0]["slurm_job_id"], q.stdout[-4000:]
    assert len(fake.calls("sbatch")) == before + 1
