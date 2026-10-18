"""`mace submit` (untracked) exits non-zero when sbatch did not take a job.

It printed "Submission complete" and exited 0 even when every sbatch call
failed. It now says how many jobs were not submitted and exits 1, also when
only some failed (`mace submit --track` already exited 1).

Runs the real `mace_cli` against tests/fake_slurm.py.
"""
from pathlib import Path

import pytest

from fake_slurm import FakeSlurm
from test_array_default_path_golden import BAND_D3, SP_D12
from test_site_jobscript import _env, _mace

REJECT = "sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified"


def _setup(tmp_path, state, decks):
    fake = FakeSlurm(tmp_path / "fs", state)
    env = _env(tmp_path, PATH=fake.env["PATH"], FAKE_SLURM_STATE=fake.env["FAKE_SLURM_STATE"],
               FAKE_SLURM_LOG=fake.env["FAKE_SLURM_LOG"])
    d = tmp_path / "decks"
    d.mkdir()
    for name in decks:
        (d / name).write_text(BAND_D3 if name.endswith(".d3") else SP_D12)
        if name.endswith(".d3"):
            (d / name).with_suffix(".f9").write_text("wavefunction")
    return env, fake


@pytest.mark.parametrize("decks", [["a.d12", "b.d12"], ["a_band.d3", "b_band.d3"]])
def test_every_sbatch_failing_exits_1(tmp_path, decks):
    env, fake = _setup(tmp_path, {"fail": {"sbatch": REJECT}}, decks)
    p = _mace(tmp_path, env, "submit", "decks")
    out = p.stdout + p.stderr
    assert p.returncode == 1, out
    assert "Submission complete" not in out
    assert "none of the 2 job(s) was submitted" in out


@pytest.mark.parametrize("decks", [["a.d12", "b.d12"], ["a_band.d3", "b_band.d3"]])
def test_some_sbatch_failing_exits_1_with_counts(tmp_path, decks):
    env, fake = _setup(tmp_path, {"fail_sbatch_call": 2}, decks)
    p = _mace(tmp_path, env, "submit", "decks")
    out = p.stdout + p.stderr
    assert p.returncode == 1, out
    assert "Submission complete" not in out
    assert "1 of 2 job(s) not submitted" in out


def test_all_submitted_exits_0(tmp_path):
    env, fake = _setup(tmp_path, {}, ["a.d12", "b.d12"])
    p = _mace(tmp_path, env, "submit", "decks")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "Submission complete" in p.stdout


def test_mixed_folder_failure_in_the_d12_part_still_submits_the_d3_and_exits_1(tmp_path):
    # (not a submit-limit refusal: that stops submitting for the run and
    # holds the rest, tests/test_queue_budget.py)
    env, fake = _setup(tmp_path, {"fail_sbatch_call": 1,
                                  "fail_sbatch_text": "Batch job submission failed: Invalid account or "
                                                      "account/partition combination specified"},
                       ["a.d12", "a_band.d3"])
    p = _mace(tmp_path, env, "submit", "decks", stdin="3\n")
    out = p.stdout + p.stderr
    assert p.returncode == 1, out
    assert [Path(c["script"]).name for c in fake.calls("sbatch")][-1] == "a_band.sh"
