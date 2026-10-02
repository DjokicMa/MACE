"""`mace convert --options_file X` without --batch, nothing on stdin.

Without --batch, convert asks its setup questions (the options file is where
the answers get saved). With empty or closed stdin the first question raised
an EOFError traceback from menu_nav. It must stop with a plain message that
points at --batch, exit 1, and write no deck.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"
CIF = REPO_ROOT / "tests" / "data" / "1_dia_opt_BULK_OPTGEOM_symm.cif"
CONFIG = REPO_ROOT / "Crystal_d12" / "example_configs" / "standard_dft_opt.json"


def _env():
    return {k: v for k, v in os.environ.items()
            if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME")}


@pytest.mark.parametrize("stdin", ["", "\n\n", None])
def test_options_file_without_batch_and_no_answers(tmp_path, stdin):
    cifs = tmp_path / "cifs"
    cifs.mkdir()
    shutil.copy(CIF, cifs)
    opts = tmp_path / "opts.json"
    opts.write_text(CONFIG.read_text())
    kw = dict(input=stdin) if stdin is not None else dict(stdin=subprocess.DEVNULL)
    r = subprocess.run([sys.executable, str(MACE_CLI), "convert", "--no-banner",
                        "--options_file", str(opts), "--cif_dir", str(cifs),
                        "--output_dir", str(tmp_path / "out")],
                       cwd=tmp_path, capture_output=True, text=True, timeout=300,
                       env=_env(), **kw)
    out = r.stdout + r.stderr
    assert "Traceback" not in out and "EOFError" not in out, out[-3000:]
    assert r.returncode == 1, out[-3000:]
    assert "--batch" in out, out[-3000:]
    assert not list(tmp_path.glob("out/*.d12"))
    assert json.loads(opts.read_text()) == json.loads(CONFIG.read_text())  # untouched
