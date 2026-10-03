"""opt2d12 --config-file reads legacy functional spellings as plan steps do.

Workflow plans have their functional names normalised when they are loaded
(mace/workflow/plan_functionals.py): "HSEsol-3C" becomes HSESOL3C, "M06-2X"
M062X, "HF" RHF, and an unknown name is refused. An opt2d12 config file -
written by hand, by --save-options, or saved by the planner's Expert mode
under workflow_configs/expert_*_configs and handed to opt2d12 by the engine -
was used as written: "M06-2X" went into the deck, which CRYSTAL23 rejects,
and "HSEsol-3C" missed the 3c basis rule, so the parent's explicit basis was
kept as "BASISSET / EXTERNAL (from original D12)". The same rule now applies
to the config, and an unknown name stops the file with a clear message.
"""
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

import test_numbered_calc_keeps_parent_method as base
from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"
PARENT = REPO_ROOT / "tests" / "data" / "opt_restart" / "tqc_agbr"


def _env():
    return {k: v for k, v in os.environ.items()
            if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME")}


def _opt2d12(tmp_path, config):
    for ext in (".d12", ".out"):
        shutil.copy(str(PARENT) + ext, tmp_path / f"p{ext}")
    (tmp_path / "c.json").write_text(json.dumps(config))
    r = subprocess.run([sys.executable, str(MACE_CLI), "opt2d12", "--out-file", "p.out",
                        "--d12-file", "p.d12", "--config-file", "c.json",
                        "--non-interactive", "--output-dir", "o"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=300,
                       env=_env(), stdin=subprocess.DEVNULL)
    decks = list((tmp_path / "o").glob("*.d12")) if (tmp_path / "o").exists() else []
    return r, (decks[0].read_text() if len(decks) == 1 else None)


def _dft(text):
    lines = text.splitlines()
    return lines[lines.index("DFT") + 1:lines.index("ENDDFT")]


@pytest.mark.parametrize("config,keyword,basis", [
    ({"functional": "HSEsol-3C"}, "HSESOL3C", "SOLDEF2MSVP"),
    ({"functional": "PBEh-3C"}, "PBEH3C", "def2-mSVP"),
    ({"method_modifications": {"new_functional": "HSE-3C"}}, "HSE3C", "def2-mSVP"),
])
def test_legacy_3c_spelling_gets_the_keyword_and_its_basis(tmp_path, config, keyword, basis):
    r, deck = _opt2d12(tmp_path, dict(config, calculation_type="SP"))
    assert deck, (r.stdout + r.stderr)[-3000:]
    assert keyword in _dft(deck)
    assert re.search(rf"^BASISSET\n{re.escape(basis)}\n", deck, re.M), deck[:800]
    assert "from original D12" not in deck


@pytest.mark.parametrize("config", [
    {"functional": "M06-2X"},
    {"method_modifications": {"new_functional": "M06-2X"}},
    {"method_modifications": {"functional": "M06-2X"}},
])
def test_legacy_spelling_is_written_as_the_crystal23_keyword(tmp_path, config):
    r, deck = _opt2d12(tmp_path, dict(config, calculation_type="SP"))
    assert deck, (r.stdout + r.stderr)[-3000:]
    assert "M062X" in _dft(deck) and "M06-2X" not in deck


def test_valid_name_is_unchanged(tmp_path):
    r, deck = _opt2d12(tmp_path, {"calculation_type": "SP", "functional": "PBE0"})
    assert deck and "PBE0" in _dft(deck)


def test_unknown_functional_is_refused(tmp_path):
    r, deck = _opt2d12(tmp_path, {"calculation_type": "SP", "functional": "B2PLYP"})
    log = r.stdout + r.stderr
    assert deck is None, log[-3000:]
    assert "functional 'B2PLYP' is not a CRYSTAL23 keyword" in log, log[-3000:]


def test_engine_expert_config_is_normalised(tmp_path):
    """A planner Expert FREQ config saved with a legacy spelling."""
    pytest.importorskip("numpy")
    eng = base._engine(tmp_path, PARENT.with_suffix(".d12").read_text(),
                       PARENT.with_suffix(".out").read_text())
    cfg_dir = tmp_path / "workflow_configs" / "expert_freq_configs"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "mat_freq_expert_config.json").write_text(json.dumps(
        {"calculation_type": "FREQ", "functional": "HSEsol-3C"}))
    name, deck = base._only_deck(base._numbered(eng, "FREQ", True))
    assert "HSESOL3C" in _dft(deck) and "FREQCALC" in deck
    assert re.search(r"^BASISSET\nSOLDEF2MSVP\n", deck, re.M), deck[:800]
