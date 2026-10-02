"""`mace submit DIR` on a directory with both .d12 and .d3 decks, no answer on stdin.

The 'Select option (1-3)' question raised an EOFError traceback when stdin
was empty or closed. It must stop with a plain message and exit 1, without
writing or submitting anything.
"""
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"


def _env():
    return {k: v for k, v in os.environ.items()
            if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME")}


def _mixed(tmp_path):
    (tmp_path / "mat_sp.d12").write_text("mat\nCRYSTAL\nEND\n")
    (tmp_path / "mat_band.d3").write_text("BAND\nEND\n")


@pytest.mark.parametrize("stdin", ["", None])
@pytest.mark.parametrize("flag", ["--nosubmit", "--track"])
def test_mixed_directory_without_an_answer_stops_cleanly(tmp_path, stdin, flag):
    _mixed(tmp_path)
    kw = dict(input=stdin) if stdin is not None else dict(stdin=subprocess.DEVNULL)
    r = subprocess.run([sys.executable, str(MACE_CLI), "submit", str(tmp_path), flag],
                       cwd=tmp_path, capture_output=True, text=True, timeout=120,
                       env=_env(), **kw)
    out = r.stdout + r.stderr
    assert "Traceback" not in out and "EOFError" not in out, out
    assert r.returncode == 1, out
    assert "No answer" in out, out
    assert not list(tmp_path.glob("*.sh")), out


def test_mixed_directory_answer_still_works(tmp_path):
    _mixed(tmp_path)
    r = subprocess.run([sys.executable, str(MACE_CLI), "submit", str(tmp_path), "--nosubmit"],
                       cwd=tmp_path, input="2\n", capture_output=True, text=True,
                       timeout=120, env=_env())
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "mat_band.sh").exists()
    assert not (tmp_path / "mat_sp.sh").exists()
