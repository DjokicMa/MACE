"""A config file whose top-level "functional" is a 3c method gets its basis.

opt2d12 set a 3c method's basis only when the config named it under
method_modifications. A config with "functional": "HSESOL3C" at the top
level (the way cif2d12 and --save-options name the functional) on a parent
with an explicit basis wrote "BASISSET / EXTERNAL (from original D12)",
a deck CRYSTAL stops on.
"""
import os
import re
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"
PARENT = REPO_ROOT / "tests" / "data" / "opt_restart" / "tqc_agbr"


def _env():
    return {k: v for k, v in os.environ.items()
            if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME")}


@pytest.mark.parametrize("functional, basis", [("HSESOL3C", "SOLDEF2MSVP"),
                                               ("PBEH3C", "def2-mSVP")])
def test_top_level_3c_functional_writes_its_basis(tmp_path, functional, basis):
    for ext in (".d12", ".out"):
        shutil.copy(str(PARENT) + ext, tmp_path / f"p{ext}")
    (tmp_path / "c.json").write_text(
        '{"calculation_type": "SP", "functional": "%s"}' % functional)
    r = subprocess.run([sys.executable, str(MACE_CLI), "opt2d12", "--out-file", "p.out",
                        "--d12-file", "p.d12", "--config-file", "c.json",
                        "--non-interactive", "--output-dir", "o"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=300,
                       env=_env(), stdin=subprocess.DEVNULL)
    log = r.stdout + r.stderr
    assert r.returncode == 0, log[-3000:]
    decks = list((tmp_path / "o").glob("*.d12"))
    assert len(decks) == 1, log[-3000:]
    text = decks[0].read_text()
    assert "from original D12" not in text
    assert re.search(rf"^BASISSET\n{re.escape(basis)}\n", text, re.M), text[:800]
    assert re.search(rf"^{functional}$", text, re.M), text[:800]
    # no explicit basis blocks after the geometry: the 3c basis is internal
    assert not re.search(r"^99 0$", text, re.M), text
