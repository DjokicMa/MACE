"""`mace submit --track` honors -i and never overwrites an edited job script.

The tracked path always regenerated <name>.sh from the template: `-i` was
ignored (no prompt, template resources) and a hand-edited <name>.sh was
silently replaced. It now follows the rules of the untracked path:

* -i asks for the resources and the job scripts carry them;
* an edited <name>.sh is submitted as written (MACE adds only the line that
  points its completion step at the database);
* with resource options (-i, --walltime) an edited script stops the
  submission, before anything is recorded or queued;
* a script MACE wrote and nobody changed is written again, as before;
* --overwrite-sh replaces any script.

Runs the real `mace_cli` against tests/fake_slurm.py.
"""
import re
from pathlib import Path

import pytest

from fake_slurm import FakeSlurm
from test_array_default_path_golden import SP_D12
from test_site_jobscript import _env, _mace


def _setup(tmp_path, nosubmit_first=True):
    fake = FakeSlurm(tmp_path / "fs")
    env = _env(tmp_path, PATH=fake.env["PATH"], FAKE_SLURM_STATE=fake.env["FAKE_SLURM_STATE"],
               FAKE_SLURM_LOG=fake.env["FAKE_SLURM_LOG"])
    d = tmp_path / "d"
    d.mkdir()
    (d / "x.d12").write_text(SP_D12)
    if nosubmit_first:
        p = _mace(d, env, "submit", "--nosubmit", "x.d12")
        assert p.returncode == 0, p.stdout + p.stderr
    return env, fake, d


def _edit(d):
    text = (d / "x.sh").read_text().replace("--ntasks=32", "--ntasks=8")
    (d / "x.sh").write_text(text)
    return text


def test_interactive_resources_reach_the_tracked_job(tmp_path):
    env, fake, d = _setup(tmp_path, nosubmit_first=False)
    p = _mace(d, env, "submit", "--track", "-i", "x.d12", stdin="y\n4\n\n\n\nn\n")
    out = p.stdout + p.stderr
    assert p.returncode == 0, out
    assert "Customize resources" in out
    (call,) = fake.calls("sbatch")
    assert "#SBATCH --ntasks=4" in call["script_text"]
    assert 'export MACE_DB_PATH="' in call["script_text"]
    assert not list(Path(__file__).resolve().parent.parent.glob("mace/submission/*_custom_*.sh"))


def test_edited_script_is_submitted_as_written(tmp_path):
    env, fake, d = _setup(tmp_path)
    edited = _edit(d)
    p = _mace(d, env, "submit", "--track", "x.d12")
    out = p.stdout + p.stderr
    assert p.returncode == 0, out
    (call,) = fake.calls("sbatch")
    assert "#SBATCH --ntasks=8" in call["script_text"]
    kept = (d / "x.sh").read_text()
    assert re.sub(r'export MACE_DB_PATH="[^"]*"\n', "", kept) == edited      # only the database line added
    assert "existing" in out and "x.sh" in out


@pytest.mark.parametrize("opts,stdin", [(["--walltime", "0:45:00"], ""), (["-i"], "n\n")])
def test_edited_script_with_resource_options_stops(tmp_path, opts, stdin):
    env, fake, d = _setup(tmp_path)
    edited = _edit(d)
    p = _mace(d, env, "submit", "--track", *opts, "x.d12", stdin=stdin)
    out = p.stdout + p.stderr
    assert p.returncode == 1, out
    assert "--overwrite-sh" in out and "Nothing was submitted" in out
    assert "Customize resources" not in out
    assert not fake.calls("sbatch")
    assert (d / "x.sh").read_text() == edited
    assert not (d / "materials.db").exists()


def test_unchanged_mace_script_is_written_again(tmp_path):
    env, fake, d = _setup(tmp_path)
    p = _mace(d, env, "submit", "--track", "--walltime", "0:45:00", "x.d12")
    assert p.returncode == 0, p.stdout + p.stderr
    (call,) = fake.calls("sbatch")
    assert "#SBATCH -t 0:45:00" in call["script_text"]


def test_overwrite_sh_replaces_an_edited_script(tmp_path):
    env, fake, d = _setup(tmp_path)
    _edit(d)
    p = _mace(d, env, "submit", "--track", "--overwrite-sh", "x.d12")
    assert p.returncode == 0, p.stdout + p.stderr
    (call,) = fake.calls("sbatch")
    assert "#SBATCH --ntasks=32" in call["script_text"]


def test_progress_does_not_replace_an_edited_script(tmp_path):
    """--progress runs the deck in its workflow's step folder with the
    plan's job script written there (it carries the workflow context): an
    edited <name>.sh beside the deck is neither used nor replaced, and does
    not stop the submission."""
    env, fake, d = _setup(tmp_path)
    edited = _edit(d)
    p = _mace(d, env, "submit", "--progress", "opt_sp", "x.d12")
    out = p.stdout + p.stderr
    assert p.returncode == 0, out
    assert "would not reach" not in out and "--overwrite-sh" not in out
    (call,) = fake.calls("sbatch")
    assert "/workflow_outputs/" in call["argv"][-1]
    assert (d / "x.sh").read_text() == edited


def test_tracked_array_does_not_replace_an_edited_script(tmp_path):
    env, fake, d = _setup(tmp_path)
    edited = _edit(d)
    p = _mace(d, env, "submit", "--track", "--array", "x.d12")
    assert p.returncode == 1, p.stdout + p.stderr
    assert not fake.calls("sbatch")
    assert (d / "x.sh").read_text() == edited


@pytest.mark.parametrize("array", [False, True])
def test_tracked_submit_from_outside_reads_the_decks_folder_file(tmp_path, array):
    # review fix: the decks' folder's mace-folder.toml, as the callback there reads it
    env, fake, d = _setup(tmp_path, nosubmit_first=False)
    (d / "mace-folder.toml").write_text('[submit]\nntasks = 12\n')
    p = _mace(tmp_path, env, "submit", "--track", *(["--array"] if array else []),
              str(d / "x.d12") if not array else str(d))
    assert p.returncode == 0, p.stdout + p.stderr
    (call,) = fake.calls("sbatch")
    assert "#SBATCH --ntasks=12" in call["script_text"]
