"""`mace submit --array` leaves out a two-component restart deck with no
density matrix beside it, as the single-job paths do (test_soc_guess_preflight).

GUESSPSO/GUESSPNOSO (manual p. 173) read fort.20, staged from $JOB.f20 or,
for GUESSPSO, the job's own non-empty $JOB.f9; CRYSTAL stops on GUESSPSO with
nothing in fort.20 (measured on HPCC). In an array such a deck would cost a
task; it is not submitted, and the other decks go into the array as before.
"""
import json

from array_helpers import SP_D12, rows
from test_array_submit_cli import _mace, _ok, _setup, _wrappers

GUESSPSO_D12 = SP_D12.replace("SHRINK\n8 8\n", "SHRINK\n8 8\nTWOCOMPON\nSOC\nGUESSPSO\nEND\n")
assert "GUESSPSO" in GUESSPSO_D12


def _decks(tmp_path, staged=False):
    fake, env, d = _setup(tmp_path, ["m0", "m1"])
    (d / "m2.d12").write_text(GUESSPSO_D12)
    if staged:
        (d / "m2.f20").write_bytes(b"m")
    return fake, env, d


def _members(d):
    (w,) = _wrappers(d)
    data = json.loads((w.parent / "submitted.json").read_text())
    return w, sorted(m["job"] for m in data["members"])


def test_untracked_array_leaves_out_a_restart_deck_with_nothing_to_stage(tmp_path):
    fake, env, d = _decks(tmp_path)
    p = _mace(tmp_path, env, "submit", "--array", "decks")
    out = p.stdout + p.stderr
    assert "Not submitting m2.d12: GUESSPSO" in out
    w, members = _members(d)
    assert members == ["m0", "m1"]
    assert "#SBATCH --array=0-1\n" in w.read_text()
    assert not (d / "m2.sh").exists() or "m2" not in w.read_text()


def test_tracked_array_leaves_it_out_too(tmp_path):
    fake, env, d = _decks(tmp_path)
    p = _mace(tmp_path, env, "submit", "--array", "--track", "decks")
    out = p.stdout + p.stderr
    assert "Not submitting m2.d12: GUESSPSO" in out
    jobs = sorted(x["slurm_job_id"] for x in rows(tmp_path / "materials.db").values()
                  if x.get("slurm_job_id"))
    assert jobs == ["4242_0", "4242_1"]
    _w, members = _members(d)
    assert "m2" not in members


def test_a_staged_restart_deck_goes_into_the_array(tmp_path):
    fake, env, d = _decks(tmp_path, staged=True)
    p = _ok(_mace(tmp_path, env, "submit", "--array", "decks"))
    assert "Not submitting" not in p.stdout + p.stderr
    _w, members = _members(d)
    assert members == ["m0", "m1", "m2"]
