"""Resource options never silently miss an existing <name>.sh.

`mace submit` reuses an existing <name>.sh (--overwrite-sh replaces it). With
-i, --walltime or a node exclusion the options went into a newly generated
script only, so a reused one ran with its old resources and nothing said
so. Now:

* a script MACE wrote and nobody changed (its last line holds the cksum of
  the lines above) is written again with the options;
* a script that was edited, or written before 1.1.5, stops the submission
  before anything is queued, with a message naming --overwrite-sh;
* without resource options an existing script is reused as before.

Runs the real `mace_cli` against tests/fake_slurm.py.
"""
import re
from pathlib import Path

import pytest

from fake_slurm import FakeSlurm
from test_array_default_path_golden import BAND_D3, SP_D12
from test_site_jobscript import _env, _mace

from mace.submission.job_script_age import generated_unmodified


def _setup(tmp_path, decks=("x.d12",)):
    fake = FakeSlurm(tmp_path / "fs")
    env = _env(tmp_path, PATH=fake.env["PATH"], FAKE_SLURM_STATE=fake.env["FAKE_SLURM_STATE"],
               FAKE_SLURM_LOG=fake.env["FAKE_SLURM_LOG"])
    d = tmp_path / "d"
    d.mkdir()
    for name in decks:
        (d / name).write_text(BAND_D3 if name.endswith(".d3") else SP_D12)
        if name.endswith(".d3"):
            (d / name).with_suffix(".f9").write_text("wavefunction")
    for name in decks:
        p = _mace(tmp_path, env, "submit", "--nosubmit", f"d/{name}")
        assert p.returncode == 0, p.stdout + p.stderr
    return env, fake, d


def _t(text):
    return re.search(r"(?m)^#SBATCH -t (\S+)", text).group(1)


@pytest.mark.parametrize("deck", ["x.d12", "x_band.d3"])
def test_unchanged_mace_script_is_written_again_with_the_walltime(tmp_path, deck):
    env, fake, d = _setup(tmp_path, (deck,))
    sh = d / (Path(deck).stem + ".sh")
    assert generated_unmodified(sh)
    p = _mace(tmp_path, env, "submit", "--walltime", "0:45:00", f"d/{deck}")
    out = p.stdout + p.stderr
    assert p.returncode == 0, out
    assert "again with the requested resources" in out
    (call,) = fake.calls("sbatch")
    assert _t(call["script_text"]) == "0:45:00"
    assert generated_unmodified(sh)


@pytest.mark.parametrize("opts,stdin", [(["--walltime", "0:45:00"], ""), (["-i"], "n\n"),
                                        (["--exclude-nodes", "node[1-2]"], "")])
def test_edited_script_stops_the_submission(tmp_path, opts, stdin):
    env, fake, d = _setup(tmp_path, ("x.d12", "y.d12"))
    edited = (d / "x.sh").read_text().replace("--ntasks=32", "--ntasks=8")
    (d / "x.sh").write_text(edited)
    p = _mace(tmp_path, env, "submit", *opts, "d", stdin=stdin)
    out = p.stdout + p.stderr
    assert p.returncode == 1, out
    assert "x.sh" in out and "--overwrite-sh" in out and "Nothing was submitted" in out
    assert not fake.calls("sbatch")
    assert (d / "x.sh").read_text() == edited
    if opts[0] == "-i":
        assert "Customize resources" not in out          # stopped before the prompts


def test_edited_script_with_overwrite_sh_is_replaced(tmp_path):
    env, fake, d = _setup(tmp_path)
    (d / "x.sh").write_text((d / "x.sh").read_text().replace("--ntasks=32", "--ntasks=8"))
    p = _mace(tmp_path, env, "submit", "--overwrite-sh", "--walltime", "0:45:00", "d/x.d12")
    assert p.returncode == 0, p.stdout + p.stderr
    (call,) = fake.calls("sbatch")
    assert _t(call["script_text"]) == "0:45:00" and "--ntasks=32" in call["script_text"]


def test_edited_script_without_options_is_reused_as_before(tmp_path):
    env, fake, d = _setup(tmp_path)
    edited = (d / "x.sh").read_text().replace("--ntasks=32", "--ntasks=8")
    (d / "x.sh").write_text(edited)
    p = _mace(tmp_path, env, "submit", "d/x.d12")
    assert p.returncode == 0, p.stdout + p.stderr
    (call,) = fake.calls("sbatch")
    assert call["script_text"] == edited


def test_script_from_before_the_marker_counts_as_edited(tmp_path):
    env, fake, d = _setup(tmp_path)
    text = (d / "x.sh").read_text()
    (d / "x.sh").write_text(text[:text.rindex("# MACE-generated")])
    p = _mace(tmp_path, env, "submit", "--walltime", "0:45:00", "d/x.d12")
    assert p.returncode == 1
    assert not fake.calls("sbatch")


def test_database_path_line_of_the_tracking_manager_is_not_an_edit(tmp_path):
    env, fake, d = _setup(tmp_path)
    sh = d / "x.sh"
    sh.write_text(sh.read_text().replace("export JOB=x\n", 'export JOB=x\nexport MACE_DB_PATH="/w/materials.db"\n'))
    assert generated_unmodified(sh)


def test_array_submission_with_an_edited_script_stops(tmp_path):
    env, fake, d = _setup(tmp_path, ("x.d12", "y.d12"))
    (d / "y.sh").write_text((d / "y.sh").read_text() + "# my note\n")
    p = _mace(tmp_path, env, "submit", "--array", "--walltime", "0:45:00", "d")
    assert p.returncode == 1, p.stdout + p.stderr
    assert not fake.calls("sbatch")
