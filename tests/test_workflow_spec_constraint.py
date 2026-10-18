"""A spec step's ``resources.constraint`` reaches every job script of that
step (1.1.6 checkpoint review): the plan's templates, the first
submission (the workflow's own template copy), a later step submitted by the
job callback, and a recovery of that step by `mace recover`.

Real `mace_cli` subprocesses and job callback, tests/fake_slurm.py, a real
OPT output and a real OOM-killed SP output from the test/ corpus."""
import json
import re
import shutil
import sqlite3
from pathlib import Path

from array_helpers import rows
from conftest import find_data
from test_workflow_spec_cli_e2e import _complete, _json, _ok, _setup, _mace


def _directives(text):
    return re.findall(r"^#SBATCH (?:--constraint=|-C )(.*)$", text, re.M)


def _template_directives(text):
    return re.findall(r"^echo '#SBATCH (?:--constraint=|-C )(.*)' >> \$1\.sh$", text, re.M)


def test_spec_constraint_reaches_every_job_script(tmp_path):
    out, deck, fake, env, root = _setup(tmp_path)
    (root / "spec.json").write_text(json.dumps({
        "version": 1, "inputs": {"type": "d12", "files": [deck.name]}, "template": "opt_sp",
        "steps": {"OPT_1": {"resources": {"constraint": "amd20"}},
                  "SP_2": {"resources": {"constraint": "intel18|amd24", "memory": "48G"}}}}))
    w = _json(_ok(_mace(root, env, "workflow", "--spec", "spec.json", "--write-plan", "--json")))
    wid = w["workflow_id"]
    scripts = root / "workflow_scripts"
    assert _template_directives((scripts / "submitcrystal23_opt_1.sh").read_text()) == ["amd20"]
    assert _template_directives((scripts / "submitcrystal23_sp_2.sh").read_text()) == ["intel18|amd24"]

    # the first submission: the workflow's own template copy
    _ok(_mace(root, env, *w["launch"][1:]))
    (first,) = fake.calls("sbatch")
    assert _directives(first["script_text"]) == ["amd20"]

    # the job callback submits the SP
    db = root / f".mace_context_{wid}" / "materials.db"
    _complete(root, env, fake, db, "OPT", out)
    sbs = fake.calls("sbatch")
    assert len(sbs) == 2 and _directives(sbs[1]["script_text"]) == ["intel18|amd24"]

    # the SP runs out of memory; mace recover resubmits it with its constraint
    (sp,) = [r for r in rows(db).values() if r["calc_type"] == "SP"]
    oom = Path(sp["input_file"]).with_suffix(".out")
    shutil.copy(find_data("FAILED_QA/T7_sp_killed_signal9_scf_converged.out"), oom)
    fake.remove_record(sp["slurm_job_id"])
    con = sqlite3.connect(db)
    with con:
        con.execute("UPDATE calculations SET status='failed', output_file=? WHERE calc_id=?",
                    (str(oom), sp["calc_id"]))
    con.close()
    rec = _ok(_mace(root, env, "recover", "--yes")).stdout
    assert "Resubmitted 1 of 1 calculation(s)." in rec, rec
    sbs = fake.calls("sbatch")
    assert len(sbs) == 3 and _directives(sbs[2]["script_text"]) == ["intel18|amd24"], sbs[2]["script_text"]

    # the check mode names it as the spec's
    chk = _json(_ok(_mace(root, env, "workflow", "--spec", "spec.json", "--json")))
    assert chk["steps"]["SP_2"]["resources"]["constraint"] == "intel18|amd24"


def test_spec_constraint_replaces_the_sites(tmp_path):
    """A site constraint stays for the steps the spec leaves alone; a step
    with its own gets that one only (one directive, never two)."""
    out, deck, fake, env, root = _setup(tmp_path)
    env["MACE_SLURM_CONSTRAINT"] = "epyc"
    (root / "spec.json").write_text(json.dumps({
        "version": 1, "inputs": {"type": "d12", "files": [deck.name]}, "template": "opt_sp",
        "steps": {"OPT_1": {"resources": {"constraint": "amd20"}}}}))
    _ok(_mace(root, env, "workflow", "--spec", "spec.json", "--write-plan", "--json"))
    scripts = root / "workflow_scripts"
    assert _template_directives((scripts / "submitcrystal23_opt_1.sh").read_text()) == ["amd20"]
    assert _template_directives((scripts / "submitcrystal23_sp_2.sh").read_text()) == ["epyc"]
