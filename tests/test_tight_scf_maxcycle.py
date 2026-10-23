"""FREQ (and ELASTIC/EOS/QHA/CPHF) decks get at least 200 SCF cycles.

These decks tighten TOLDEE past the parent's (FREQ to 11) but used to keep the
parent's MAXCYCLE, so a metallic or semimetal FREQ run stopped with "SCF ENDED
- TOO MANY CYCLES" (graphene's, at 100). A generated deck's MAXCYCLE is the
parent's, raised to 200 when it is lower; a higher one is kept. A parent with
no MAXCYCLE gets opt2d12's usual 800, never less than it wrote before the
floor. Every route writes these decks through opt2d12, so the tests run the
real ``mace_cli opt2d12`` on the repo's Si parent (MAXCYCLE 60 as shipped).
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))

import test_opt2d12_prompt_defaults as prompts  # noqa: E402
from d12_constants import (SCF_MAXCYCLE_FALLBACK, TIGHT_SCF_MIN_MAXCYCLE,  # noqa: E402
                           tight_scf_maxcycle)

MACE_CLI = REPO_ROOT / "mace_cli"
SI = REPO_ROOT / "tests" / "fixtures" / "phonopy_crystal" / "Si"


def _parent(tmp_path, maxcycle):
    """The Si parent with its SCF MAXCYCLE set to ``maxcycle`` (None: no record)."""
    lines = (SI / "parent.d12").read_text().splitlines()
    i = lines.index("MAXCYCLE")
    if maxcycle is None:
        del lines[i:i + 2]
    else:
        lines[i + 1] = str(maxcycle)
    (tmp_path / "parent.d12").write_text("\n".join(lines) + "\n")
    shutil.copy(SI / "parent.out", tmp_path / "parent.out")


def _opt2d12(tmp_path, *args, stdin=""):
    env = dict(os.environ, MACE_NO_BANNER="1")
    r = subprocess.run([sys.executable, str(MACE_CLI), "opt2d12", "--out-file", "parent.out",
                        "--d12-file", "parent.d12", *args],
                       cwd=tmp_path, env=env, input=stdin, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, (r.stdout + r.stderr)[-1500:]
    decks = [p for p in tmp_path.glob("*.d12") if p.name != "parent.d12"]
    assert len(decks) == 1, sorted(p.name for p in tmp_path.iterdir())
    return decks[0].read_text().splitlines()


def _scf_maxcycle(lines):
    """The SCF block's MAXCYCLE (the last one: OPTGEOM/CPHF blocks have their own)."""
    i = len(lines) - 1 - lines[::-1].index("MAXCYCLE")
    return int(lines[i + 1])


CASES = [(40, 200), (100, 200), (300, 300), (800, 800), (None, 800)]


@pytest.mark.parametrize("parent, expected", CASES)
def test_helper(parent, expected):
    assert TIGHT_SCF_MIN_MAXCYCLE == 200
    assert SCF_MAXCYCLE_FALLBACK == 800
    assert tight_scf_maxcycle(parent) == expected


@pytest.mark.parametrize("parent, expected", CASES)
def test_non_interactive_freq(tmp_path, parent, expected):
    _parent(tmp_path, parent)
    child = _opt2d12(tmp_path, "--calc-type", "FREQ", "--non-interactive")
    assert "FREQCALC" in child
    assert _scf_maxcycle(child) == expected


@pytest.mark.parametrize("parent, expected", CASES)
def test_config_file_freq(tmp_path, parent, expected):
    """The workflow engine's FREQ step (and the GUI preview): a config file."""
    _parent(tmp_path, parent)
    (tmp_path / "cfg.json").write_text(json.dumps({"calculation_type": "FREQ"}))
    child = _opt2d12(tmp_path, "--config-file", "cfg.json", stdin="y\n" + "\n" * 17)
    assert "FREQCALC" in child
    assert _scf_maxcycle(child) == expected


@pytest.mark.parametrize("parent, expected", [(100, 200), (800, 800), (None, 800)])
def test_interactive_freq(tmp_path, parent, expected):
    """The opt2d12 menu: blank answers, calculation type FREQ."""
    _parent(tmp_path, parent)
    child = prompts._opt2d12(tmp_path, "parent", {"exact settings": "n", "calculation type": "3"})
    assert "FREQCALC" in child
    assert _scf_maxcycle(child) == expected


@pytest.mark.parametrize("calc, mode", [("ELASTIC", "ELASTCON"), ("QHA", "QHA"), ("CPHF", "CPHF")])
def test_module_decks(tmp_path, calc, mode):
    _parent(tmp_path, 100)
    child = _opt2d12(tmp_path, "--calc-type", calc, "--non-interactive",
                     "--elastic-settings", json.dumps({"mode": mode}))
    assert mode in child
    assert _scf_maxcycle(child) == 200


def test_sp_child_keeps_the_parent_value(tmp_path):
    """An SP deck does not tighten TOLDEE, so it keeps the parent's 40."""
    _parent(tmp_path, 40)
    child = _opt2d12(tmp_path, "--calc-type", "SP", "--non-interactive")
    assert _scf_maxcycle(child) == 40


@pytest.mark.parametrize("calc", ["SP", "FREQ"])
def test_parent_without_maxcycle_keeps_the_fallback(tmp_path, calc):
    """A parent with SCF records but no MAXCYCLE: FREQ gets what SP (and the
    1.1.6 preview's FREQ) got, 800, not the 200 floor."""
    _parent(tmp_path, None)
    child = _opt2d12(tmp_path, "--calc-type", calc, "--non-interactive")
    assert _scf_maxcycle(child) == SCF_MAXCYCLE_FALLBACK
