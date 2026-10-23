"""Existing job scripts MACE wrote: from an older MACE, or from a customized
template.

A <deck>.sh an older MACE (1.0.0 - the 1.1.5 preview) wrote with its default
resources carries no 1.1.5 marker. `mace submit --track` submitted it as an
"edited" script (without the 1.1.5 job-script fixes), and with --walltime,
-i, --progress or --array - and untracked with --walltime - it stopped the
run as a conflict. It is now recognized and written again. A script edited
by hand still is the user's.

A script a workflow plan's customized template copy wrote (its cores and
memory) carries a valid marker: --walltime wrote it again from the plain
template and dropped the plan's sizes. Now only the walltime is set in it.

The fixtures in tests/data/older_job_scripts/ are the output of the
generators of 1.1.4 and of the 1.1.5 preview (git show v1.1.4: / v1.1.5:).
Through the real mace_cli with tests/fake_slurm.py.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from fake_slurm import FakeSlurm
from test_array_default_path_golden import OPT_D12, SP_D12
from test_site_jobscript import _env, _mace

OLDER = Path(__file__).resolve().parent / "data" / "older_job_scripts"
TEMPLATE = REPO_ROOT / "mace" / "submission" / "submitcrystal23.sh"


def _setup(tmp_path, older=None):
    fake = FakeSlurm(tmp_path / "fs")
    # MSU HPCC, whose plain scripts the templates write (elsewhere a SLURM
    # without readable accounting gets no -A line: test_site_account_unreadable)
    env = _env(tmp_path, PATH=fake.env["PATH"], FAKE_SLURM_STATE=fake.env["FAKE_SLURM_STATE"],
               FAKE_SLURM_LOG=fake.env["FAKE_SLURM_LOG"], MACE_HOSTNAME="dev-amd20.hpcc.msu.edu")
    d = tmp_path / "d"
    d.mkdir()
    (d / "x.d12").write_text(SP_D12)
    if older:
        shutil.copy(OLDER / older, d / "x.sh")
    return fake, env, d


def _sbatch_lines(text):
    return [l for l in text.splitlines() if l.startswith("#SBATCH")]


@pytest.mark.parametrize("older", ["scf_1.1.4.sh", "scf_1.1.5_preview.sh"])
def test_track_writes_an_older_generated_script_again(tmp_path, older):
    fake, env, d = _setup(tmp_path, older)
    p = _mace(d, env, "submit", "--track", "x.d12")
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-3000:]
    assert "x.sh was written by an older MACE; regenerating it with 1.1.5's fixes" in out, out[-3000:]
    assert "edited" not in out
    (call,) = fake.calls("sbatch")
    assert ".mace_submit_dir" in call["script_text"]


@pytest.mark.parametrize("args", [("--track", "--walltime", "1:00:00"), ("--walltime", "1:00:00")])
def test_walltime_writes_an_older_generated_script_again(tmp_path, args):
    fake, env, d = _setup(tmp_path, "scf_1.1.4.sh")
    p = _mace(d, env, "submit", *args, "x.d12")
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-3000:]
    assert "would not reach" not in out
    (call,) = fake.calls("sbatch")
    assert "#SBATCH -t 1:00:00" in call["script_text"]
    assert ".mace_submit_dir" in call["script_text"]


def test_an_older_script_edited_by_hand_is_still_the_users(tmp_path):
    fake, env, d = _setup(tmp_path, "scf_1.1.4.sh")
    sh = d / "x.sh"
    sh.write_text(sh.read_text().replace("module purge", "module purge\nmodule load mine"))
    p = _mace(d, env, "submit", "--track", "--walltime", "1:00:00", "x.d12")
    out = p.stdout + p.stderr
    assert p.returncode != 0 and "would not reach" in out, out[-3000:]
    assert fake.calls("sbatch") == []
    assert "module load mine" in sh.read_text()


def _plan_script(tmp_path, d, env):
    """x.sh written by a workflow plan's customized copy of the template."""
    ws = tmp_path / "workflow_scripts"
    ws.mkdir()
    copy = ws / "submitcrystal23_sp_2.sh"
    copy.write_text(TEMPLATE.read_text().replace("--ntasks=32", "--ntasks=64")
                    .replace("--mem-per-cpu=5G", "--mem-per-cpu=9G"))
    subprocess.run(["bash", str(copy), "x"], cwd=d, env=env, check=True, capture_output=True)
    from mace.submission.job_script_age import generated_unmodified
    assert generated_unmodified(d / "x.sh")


@pytest.mark.parametrize("track", [False, True])
def test_walltime_keeps_a_plan_scripts_cores_and_memory(tmp_path, track):
    fake, env, d = _setup(tmp_path)
    _plan_script(tmp_path, d, env)
    p = _mace(d, env, "submit", *(["--track"] if track else []), "--walltime", "1:00:00", "x.d12")
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-3000:]
    assert "keeps its cores and memory" in out, out[-3000:]
    (call,) = fake.calls("sbatch")
    lines = _sbatch_lines(call["script_text"])
    assert "#SBATCH --ntasks=64" in lines and "#SBATCH --mem-per-cpu=9G" in lines, lines
    assert "#SBATCH -t 1:00:00" in lines, lines
    from mace.submission.job_script_age import generated_unmodified
    assert generated_unmodified(d / "x.sh")          # stamped again: still MACE's


def test_walltime_still_writes_a_plain_script_again(tmp_path):
    fake, env, d = _setup(tmp_path)
    subprocess.run(["bash", str(TEMPLATE), "x"], cwd=d, env=env, check=True, capture_output=True)
    p = _mace(d, env, "submit", "--walltime", "1:00:00", "x.d12")
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-3000:]
    assert "keeps its cores and memory" not in out
    (call,) = fake.calls("sbatch")
    assert "#SBATCH -t 1:00:00" in _sbatch_lines(call["script_text"])


def test_track_reuses_a_plan_script_as_it_is(tmp_path):
    fake, env, d = _setup(tmp_path)
    _plan_script(tmp_path, d, env)
    p = _mace(d, env, "submit", "--track", "x.d12")
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-3000:]
    (call,) = fake.calls("sbatch")
    lines = _sbatch_lines(call["script_text"])
    assert "#SBATCH --ntasks=64" in lines and "#SBATCH --mem-per-cpu=9G" in lines, lines


# the 1.1.5 preview's template (git show 28e85db8:mace/submission/submitcrystal23.sh), frozen
BASELINE = Path(__file__).parent / "data" / "older_job_scripts" / "submitcrystal23_1.1.5_preview_template.sh"


@pytest.mark.parametrize("source", ["scf_1.1.4.sh", "scf_1.1.5_preview.sh", "baseline"])
def test_progress_leaves_an_older_generated_script_and_writes_its_own(tmp_path, source):
    """--progress runs the deck in its workflow's step folder with a job
    script written there; the older one beside the deck is left as it is."""
    if source == "baseline":
        fake, env, d = _setup(tmp_path)
        # laid out as in a preview install (mace/submission next to mace/queue/manager.py), so the
        # template pins its own tree as it does for a real user
        tpl = tmp_path / "preview" / "mace" / "submission" / "submitcrystal23.sh"
        tpl.parent.mkdir(parents=True)
        (tmp_path / "preview" / "mace" / "queue").mkdir()
        (tmp_path / "preview" / "mace" / "queue" / "manager.py").write_text("")
        shutil.copy(BASELINE, tpl)
        subprocess.run(["bash", str(tpl), "x"], cwd=d, env=env, check=True, capture_output=True)
        assert "MACE-generated" not in (d / "x.sh").read_text()
    else:
        fake, env, d = _setup(tmp_path, source)
    (d / "x.d12").write_text(OPT_D12)
    old = (d / "x.sh").read_bytes()
    p = _mace(d, env, "submit", "--progress", "opt_sp", "x.d12")
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-3000:]
    assert "would not reach" not in out and "regenerating it" not in out, out[-3000:]
    assert (d / "x.sh").read_bytes() == old
    calls = fake.calls("sbatch")
    assert calls and ".mace_submit_dir" in calls[0]["script_text"], out[-3000:]
    assert "/workflow_outputs/" in calls[0]["argv"][-1] and calls[0]["argv"][-1].endswith("/x_opt.sh")


@pytest.mark.parametrize("opts", [("--track", "--array")])
def test_an_edited_script_is_not_told_to_drop_progress_or_array(tmp_path, opts):
    fake, env, d = _setup(tmp_path, "scf_1.1.4.sh")
    sh = d / "x.sh"
    sh.write_text(sh.read_text().replace("module purge", "module purge\nmodule load mine"))
    p = _mace(d, env, "submit", *opts, "x.d12")
    out = p.stdout + p.stderr
    assert p.returncode == 1, out[-3000:]
    assert "--overwrite-sh" in out and "drop the options" not in out, out[-3000:]
    assert fake.calls("sbatch") == []
