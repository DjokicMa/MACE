"""`mace ions --help` shows worked examples, as `mace opt2d12` and `mace defects` do.

Its help listed the options only, with nothing on how they combine (an
ion cluster from cif2d12 decks, an explicit net charge, per-atom shell
charges checked against a parent output) or where the records are explained.
"""
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"


def _help():
    env = {k: v for k, v in os.environ.items()
           if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    r = subprocess.run([sys.executable, str(MACE_CLI), "ions", "--help"], env=env,
                       cwd=str(REPO_ROOT), stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def _examples():
    text = _help()
    assert "Examples:" in text, text
    return [line.strip() for line in text.split("Examples:", 1)[1].splitlines()
            if line.strip().startswith("mace ions ")]


@pytest.mark.parametrize("flags", [
    ("--ion-charges auto", "-o "),
    ("--net-charge",),
    ("--site-states", "--net-charge"),
    ("--parent-out", "--site-states"),
])
def test_an_example_uses(flags):
    examples = _examples()
    assert any(all(f in ex for f in flags) for ex in examples), examples


def test_every_example_is_a_valid_command_line(tmp_path):
    """mace ions accepts each example's options (argparse exits 2 on a bad one).

    The files an example names are created empty in a scratch directory, so
    the run gets past the options and stops at the empty deck.
    """
    import shlex
    env = {k: v for k, v in os.environ.items()
           if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    for ex in _examples():
        argv = shlex.split(ex.split(" #", 1)[0])
        assert argv[:2] == ["mace", "ions"], ex
        for a in argv[2:]:
            if a.endswith((".d12", ".out", ".json")):
                (tmp_path / a).parent.mkdir(parents=True, exist_ok=True)
                (tmp_path / a).write_text("")
        r = subprocess.run([sys.executable, str(MACE_CLI), *argv[1:]], env=env,
                           cwd=str(tmp_path), stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=120)
        assert r.returncode in (0, 1), (ex, r.stdout + r.stderr)
        assert "usage:" not in r.stderr and "Traceback" not in r.stderr, (ex, r.stderr)


def test_points_at_the_charged_systems_docs():
    text = " ".join(_help().split())
    assert "Crystal_d12/README.md" in text and "Charged systems and ion clusters" in text, text
    readme = (REPO_ROOT / "Crystal_d12" / "README.md").read_text()
    assert "## Charged systems and ion clusters" in readme
