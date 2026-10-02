"""A two-component restart deck with no density matrix beside it is stopped
before it costs a job: by `mace preflight` and by `mace submit`.

GUESSPSO/GUESSPNOSO (manual p. 173) read fort.20, which submitcrystal23.sh
stages from $JOB.f20, or for GUESSPSO from the job's own non-empty $JOB.f9.
CRYSTAL stops on GUESSPSO with nothing in fort.20 (measured on HPCC).
"""
import subprocess
import sys

import pytest

from conftest import REPO_ROOT
from mace.submission import preflight as pf
from test_submit_progress_cli_e2e import D12, fake_slurm  # noqa: F401

MACE_CLI = REPO_ROOT / "mace_cli"
GUESSPSO_D12 = D12.replace("SHRINK\n8 8\n", "SHRINK\n8 8\nTWOCOMPON\nSOC\nGUESSPSO\nEND\n")
GUESSPNOSO_D12 = GUESSPSO_D12.replace("GUESSPSO", "GUESSPNOSO")
assert "GUESSPSO" in GUESSPSO_D12


def _deck(tmp_path, text, f20=None, f9=None, name="mat_sp"):
    deck = tmp_path / f"{name}.d12"
    deck.write_text(text)
    if f20 is not None:
        deck.with_suffix(".f20").write_bytes(f20)
    if f9 is not None:
        deck.with_suffix(".f9").write_bytes(f9)
    return deck


@pytest.mark.parametrize("text,f20,f9,missing", [
    (GUESSPSO_D12, None, None, "no mat_sp.f20 or a non-empty mat_sp.f9"),
    (GUESSPSO_D12, None, b"", "no mat_sp.f20 or a non-empty mat_sp.f9"),
    (GUESSPSO_D12, b"", None, "no mat_sp.f20 or a non-empty mat_sp.f9"),
    (GUESSPSO_D12, b"m", None, None),
    (GUESSPSO_D12, None, b"m", None),
    (GUESSPNOSO_D12, None, b"m", "no mat_sp.f20 next to the deck"),
    (GUESSPNOSO_D12, b"m", None, None),
    (D12, None, None, None),
])
def test_restart_file_problem(tmp_path, text, f20, f9, missing):
    problem = pf.restart_file_problem(_deck(tmp_path, text, f20, f9))
    if missing is None:
        assert problem is None
    else:
        assert missing in problem
        if "GUESSPSO" in text:
            assert "CRYSTAL stops on GUESSPSO with nothing in fort.20 (measured)" in problem


def test_a_guesspso_in_the_title_is_not_a_restart(tmp_path):
    assert pf.restart_file_problem(_deck(tmp_path, "GUESSPSO\n" + D12.split("\n", 1)[1])) is None


class _Runner:
    def __init__(self):
        self.ran = False

    def __call__(self, workdir):
        self.ran = True
        return pf.RunOutcome(returncode=0)


def test_preflight_fails_a_restart_deck_with_nothing_to_stage(tmp_path):
    runner = _Runner()
    result = pf.preflight_deck(_deck(tmp_path, GUESSPSO_D12), runner=runner)
    assert result.status == pf.FAIL and "mat_sp.f20" in result.reason
    assert not runner.ran


def test_preflight_cannot_check_a_staged_restart_deck(tmp_path):
    runner = _Runner()
    result = pf.preflight_deck(_deck(tmp_path, GUESSPSO_D12, f20=b"m"), runner=runner)
    assert result.status == pf.ERROR and "GUESSPSO reads fort.20" in result.reason
    assert not runner.ran


def _submit(workdir, env, *args):
    return subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "submit", *args],
                          cwd=str(workdir), env=env, capture_output=True, text=True,
                          timeout=300)


def test_mace_submit_does_not_submit_a_restart_deck_with_nothing_to_stage(tmp_path, fake_slurm):
    deck = _deck(tmp_path, GUESSPSO_D12)
    proc = _submit(tmp_path, fake_slurm, str(deck))
    out = proc.stdout + proc.stderr
    assert "Not submitting mat_sp.d12: GUESSPSO" in out
    assert "999001" not in out


def test_mace_submit_track_does_not_submit_it_either(tmp_path, fake_slurm):
    deck = _deck(tmp_path, GUESSPSO_D12)
    proc = _submit(tmp_path, fake_slurm, "--track", str(deck))
    out = proc.stdout + proc.stderr
    assert proc.returncode == 1 and "Submitted 0/1 tracked calculation(s)" in out
    assert "999001" not in out


def test_mace_submit_submits_a_staged_restart_deck(tmp_path, fake_slurm):
    deck = _deck(tmp_path, GUESSPSO_D12, f20=b"m")
    proc = _submit(tmp_path, fake_slurm, str(deck))
    assert "999001" in proc.stdout + proc.stderr
