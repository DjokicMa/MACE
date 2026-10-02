"""`mace submit DIR` with a RELATIVE directory (or sub/deck.d12).

crystal.py changes into the deck's directory before generating the job script,
but then looked for the script at <relative dir>/<job>.sh from INSIDE that
directory - a path that does not exist - and reported "Failed to generate/find
script" for every deck, submitting nothing. Found on HPCC with
`mace submit phonopy_<stem>_222 --walltime 1:00:00`, the command
`mace phonopy generate` prints. Absolute paths and the current directory were
unaffected.
"""
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"

DECK = "t\nCRYSTAL\n0 0 0\n225\n4.2\n1\n12 0 0 0\nBASISSET\nPOB-TZVP-REV2\nEND\n"


@pytest.fixture
def env(tmp_path):
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    log = tmp_path / "sbatch.log"
    (bindir / "sbatch").write_text(
        f"#!/bin/bash\necho \"$PWD $*\" >> {log}\necho 'Submitted batch job 999002'\n")
    for name in ("squeue", "sacct", "scancel"):
        (bindir / name).write_text("#!/bin/bash\nexit 0\n")
    for f in bindir.iterdir():
        f.chmod(0o755)
    e = dict(os.environ)
    e["PATH"] = f"{bindir}:{e['PATH']}"
    e["MACE_NO_BANNER"] = "1"
    return e


@pytest.mark.parametrize("target", ["sub", "sub/", "sub/mat.d12"])
def test_relative_target_is_submitted(tmp_path, env, target):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "mat.d12").write_text(DECK)
    proc = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "submit", target],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    out = proc.stdout + proc.stderr
    assert "Failed to generate/find script" not in out, out
    assert "999002" in out, out
    assert (sub / "mat.sh").is_file()
    assert (tmp_path / "sbatch.log").read_text().split() == [str(sub), "mat.sh"]
