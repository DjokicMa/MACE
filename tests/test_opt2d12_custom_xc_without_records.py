"""A config naming CUSTOM-XC for a parent without custom records fails cleanly.

CUSTOM-XC is MACE's label for a parent's own EXCHANGE/CORRELAT/HYBRID
records. For a parent without them the writer raised half way through the
deck, leaving the truncated file behind. opt2d12 now stops before writing,
says why, and exits non-zero.
"""
import json
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, TEST_DATA
from d12_constants import CUSTOM_FUNCTIONAL

MACE_CLI = REPO_ROOT / "mace_cli"


def _parent(tmp_path, dft_lines, after=(), grid=True):
    src = TEST_DATA / "OPT" / "1_dia_opt_rev1.out"
    if not src.exists():
        pytest.skip("test/ corpus not present (gitignored, ~12GB)")
    deck = []
    for line in src.with_suffix(".d12").read_text().splitlines():
        if line.strip() == "B3LYP-D3":
            deck += dft_lines
        elif line.strip() == "XLGRID" and not grid:
            continue
        elif line.strip() == "ENDDFT":
            deck += [line, *after]
        else:
            deck.append(line)
    (tmp_path / "parent.d12").write_text("\n".join(deck) + "\n")
    (tmp_path / "parent.out").write_text(src.read_text(errors="replace"))
    return tmp_path


def _run(cwd, *extra, stdin="", ok=True):
    cmd = [sys.executable, str(MACE_CLI), "opt2d12", "--out-file", "parent.out",
           "--d12-file", "parent.d12", *extra]
    result = subprocess.run(cmd, cwd=cwd, input=stdin, capture_output=True,
                            text=True, timeout=300)
    log = result.stdout + result.stderr
    decks = [p for p in cwd.glob("*.d12") if p.name != "parent.d12"]
    if not ok:
        return result.returncode, decks, log
    assert result.returncode == 0, log[-1500:]
    assert len(decks) == 1, (sorted(p.name for p in cwd.iterdir()), log[-1500:])
    return decks[0].read_text().splitlines(), log


def test_custom_xc_config_without_records_fails_cleanly(tmp_path):
    parent = _parent(tmp_path, ["PBE0"])
    (parent / "cfg.json").write_text(json.dumps({"calculation_type": "SP",
                                                 "functional": CUSTOM_FUNCTIONAL}))
    code, decks, log = _run(parent, "--config-file", "cfg.json", "--non-interactive", ok=False)
    assert code != 0
    assert decks == []
    assert "Traceback" not in log
    assert "this parent's DFT block has none" in log
