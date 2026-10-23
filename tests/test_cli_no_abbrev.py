"""The 1.1.6 tool parsers take full option names only (allow_abbrev=False).

An abbreviation (`--forc` for --force, `--output` for --output-dir) let mace
mcp's calc_run pass on options its filter never saw; a prefix is now an
unknown argument, as through `mace_cli` in a subprocess.
"""
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"


def _mace(*args):
    env = {k: v for k, v in os.environ.items()
           if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", *args], env=env, cwd=str(REPO_ROOT),
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout + r.stderr


@pytest.mark.parametrize("argv, prefix", [
    (("calc", "sp", "x.xyz", "-b", "uff", "--forc"), "--forc"),
    (("calc", "sp", "x.xyz", "-b", "uff", "--output=elsewhere"), "--output=elsewhere"),
    (("calc", "sp", "x.xyz", "-b", "mlip", "--model-f=m.model"), "--model-f=m.model"),
    (("analyze", "pxrd", "x.cif", "--jso"), "--jso"),
    (("mlip", "list", "--jso"), "--jso"),
    (("analyze", "elastic", "x.out", "--jso"), "--jso"),
    (("tau", "estimate", "--m-star", "0.3", "--temp", "300"), "--temp"),
    (("mcp", "--allow-outside", "--list-tools"), "--allow-outside"),
])
def test_option_prefixes_are_refused(argv, prefix):
    rc, text = _mace(*argv)
    assert rc == 2, text
    assert "unrecognized arguments" in text and prefix in text, text
