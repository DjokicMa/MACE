"""`mace recover` leaves the steps of a paused or canceled workflow alone
(1.1.6 checkpoint review): the job callback records their failures and
submits no recovery job, and `mace recover` (the CLI, the Jobs panel's
Recover and Recover Failed…) does the same until the workflow is resumed.

Real `mace_cli` subprocesses, tests/fake_slurm.py, a real OPT deck and a
real OOM-killed output from the test/ corpus."""
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

from array_helpers import rows
from conftest import REPO_ROOT, find_data
from test_workflow_spec_cli_e2e import _json, _ok, _setup
from test_array_default_path_golden import _mace


def _failed_opt(tmp_path):
    """A spec workflow whose first step (OPT) ran out of memory."""
    out, deck, fake, env, root = _setup(tmp_path)
    (root / "spec.json").write_text(json.dumps({
        "version": 1, "inputs": {"type": "d12", "files": [deck.name]}, "template": "opt_sp"}))
    w = _json(_ok(_mace(root, env, "workflow", "--spec", "spec.json", "--write-plan", "--json")))
    wid = w["workflow_id"]
    _ok(_mace(root, env, "submit", "--progress-plan", w["plan_file"], deck.name))
    db = root / f".mace_context_{wid}" / "materials.db"
    (opt,) = rows(db).values()
    work = Path(opt["input_file"]).parent
    oom = work / f"{Path(opt['input_file']).stem}.out"
    shutil.copy(find_data("FAILED_QA/3_dia3_opt_killed_signal9_oom.out"), oom)
    fake.remove_record(opt["slurm_job_id"])
    con = sqlite3.connect(db)
    with con:
        con.execute("UPDATE calculations SET status='failed', output_file=? WHERE calc_id=?",
                    (str(oom), opt["calc_id"]))
    con.close()
    return fake, env, root, wid, db, opt


def test_recover_skips_the_steps_of_a_held_workflow_until_it_is_resumed(tmp_path):
    fake, env, root, wid, db, opt = _failed_opt(tmp_path)
    assert len(fake.calls("sbatch")) == 1
    for verb, state in (("--cancel", "canceled"), ("--pause", "paused")):
        args = ["workflow", verb, wid, "--json"] + (["--yes"] if verb == "--cancel" else [])
        _ok(_mace(root, env, *args))
        p = _mace(root, env, "recover", "--yes")
        out = p.stdout + p.stderr
        assert p.returncode == 0, out
        assert f"workflow {wid} is {state}: resume it first" in out, out
        assert "0 to resubmit" in out, out
        assert len(fake.calls("sbatch")) == 1, out
        dry = _ok(_mace(root, env, "recover", "--dry-run")).stdout
        assert "keeps workflow" not in dry and f"is {state}: resume it first" in dry, dry
        doc = _json(_ok(_mace(root, env, "recover", "--dry-run", "--json")))
        (e,) = doc["entries"]
        assert e["state"] == "blocked" and e["held"] == state
        assert rows(db)[opt["calc_id"]]["status"] == "failed"

    # resumed: the same command resubmits it, in its workflow
    _ok(_mace(root, env, "workflow", "--resume", wid, "--json"))
    out = _ok(_mace(root, env, "recover", "--yes")).stdout
    assert "Resubmitted 1 of 1 calculation(s)." in out, out
    assert len(fake.calls("sbatch")) == 2
    new = [r for r in rows(db).values() if r["calc_id"] != opt["calc_id"]]
    assert len(new) == 1 and json.loads(new[0]["settings_json"])["workflow_id"] == wid


def test_the_resubmission_checks_the_hold_again(tmp_path):
    """The workflow is paused between the plan and the run: the worker (the
    step that holds the queue lock) refuses it."""
    fake, env, root, wid, db, opt = _failed_opt(tmp_path)
    _ok(_mace(root, env, "workflow", "--pause", wid, "--json"))
    work = Path(opt["input_file"]).parent
    script = Path(opt["input_file"]).with_suffix(".sh").read_text()
    exports = {}
    for var in ("MACE_WORKFLOW_ID", "MACE_CONTEXT_DIR", "MACE_ISOLATION_MODE"):
        for line in script.splitlines():
            if line.startswith(f"export {var}="):
                exports[var] = line.split("=", 1)[1].strip('"')
    wenv = dict(env, PYTHONPATH=str(REPO_ROOT), **exports)
    p = subprocess.run([sys.executable, "-m", "mace.recovery.resubmit", "--worker", "--db", str(db),
                        "--calc-id", opt["calc_id"], "--error-type", "memory_error"],
                       cwd=str(work), env=wenv, capture_output=True, text=True, timeout=600)
    (line,) = [x for x in p.stdout.splitlines() if x.startswith("MACE_RECOVER_RESULT ")]
    res = json.loads(line[len("MACE_RECOVER_RESULT "):])
    assert res["status"] == "skipped" and f"workflow {wid} is paused" in res["why"], p.stdout + p.stderr
    assert len(fake.calls("sbatch")) == 1
    assert rows(db)[opt["calc_id"]]["status"] == "failed"


def test_dry_run_names_the_hold_instead_of_promising_progression():
    from mace.recovery.resubmit import render_preview
    e = {"calc_id": "c1", "fix": "f", "message": "", "workflow_id": "workflow_20261015_000000_abcd",
         "preview": {"ok": True, "changes": [], "notes": [], "commands": []}}
    assert "the workflow goes on when the new run completes" in render_preview(1, e)
    held = dict(e, held="paused")
    text = render_preview(1, held)
    assert "goes on" not in text and "workflow_20261015_000000_abcd is paused" in text
