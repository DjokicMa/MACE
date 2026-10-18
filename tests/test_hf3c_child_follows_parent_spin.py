"""An HF-3c child of a spin-polarized parent is unrestricted (UHF).

A follow-up keeps what the user did not change, and the DFT 3c methods keep
the parent's SPIN. HF3C and HFSOL3C dropped it: the deck was written
"BASISSET / MINIX / HF3C / END" with no UHF, a closed-shell HF run. UHF is
the open-shell Hartree-Fock keyword of the Hamiltonian section (CRYSTAL23
manual p. 131); the HF-3c block sits in the same section, closed by its own
END (sec. 5.3.1, p. 158-159). A closed-shell parent's child stays RHF.
"""
import os
import re
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, TEST_DATA

MACE_CLI = REPO_ROOT / "mace_cli"
SPIN_PARENT = "1_dia_opt_rev1"               # DFT / SPIN / B3LYP-D3, SPINLOCK
CLOSED_PARENT = "1_dia_opt_BULK_OPTGEOM"     # no SPIN


def _env():
    return {k: v for k, v in os.environ.items()
            if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME")}


def _copy(tmp_path, stem):
    src = TEST_DATA / "OPT" / f"{stem}.out"
    if not src.exists():
        pytest.skip("test/ corpus not present (gitignored, ~12GB)")
    shutil.copy(src, tmp_path / "p.out")
    shutil.copy(src.with_suffix(".d12"), tmp_path / "p.d12")


def _opt2d12(cwd, *args):
    r = subprocess.run([sys.executable, str(MACE_CLI), "opt2d12", *args],
                       cwd=cwd, capture_output=True, text=True, timeout=300,
                       env=_env(), stdin=subprocess.DEVNULL)
    return r, r.stdout + r.stderr


def _child(tmp_path, functional, config_key="functional"):
    cfg = ({"calculation_type": "SP", "functional": functional} if config_key == "functional"
           else {"calculation_type": "SP", "method_modifications": {"new_functional": functional}})
    import json
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    r, log = _opt2d12(tmp_path, "--out-file", "p.out", "--d12-file", "p.d12",
                      "--config-file", "c.json", "--non-interactive", "--output-dir", "o")
    assert r.returncode == 0, log[-3000:]
    decks = list((tmp_path / "o").glob("*.d12"))
    assert len(decks) == 1, log[-3000:]
    return decks[0]


def _hamiltonian(text):
    """The lines after the basis-set record (the Hamiltonian section)."""
    lines = text.splitlines()
    return lines[lines.index("BASISSET") + 2:]


@pytest.mark.parametrize("functional, basis", [("HF3C", "MINIX"), ("HFSOL3C", "SOLMINIX")])
@pytest.mark.parametrize("config_key", ["functional", "new_functional"])
def test_spin_parent_gives_an_unrestricted_hf3c_child(tmp_path, functional, basis, config_key):
    _copy(tmp_path, SPIN_PARENT)
    text = _child(tmp_path, functional, config_key).read_text()
    ham = _hamiltonian(text)
    assert re.search(rf"^BASISSET\n{basis}\nUHF\n{functional}\nEND\n", text, re.M), text
    assert "SPIN" not in ham, "SPIN is a DFT-block keyword; HF takes UHF"
    assert "DFT" not in ham
    # the parent's SPINLOCK needs an open-shell Hamiltonian: UHF is one
    assert "SPINLOCK" in ham, text


def test_closed_shell_parent_gives_a_restricted_hf3c_child(tmp_path):
    _copy(tmp_path, CLOSED_PARENT)
    text = _child(tmp_path, "HF3C").read_text()
    assert re.search(r"^BASISSET\nMINIX\nHF3C\nEND\n", text, re.M), text
    assert "UHF" not in _hamiltonian(text)


def test_unrestricted_hf3c_deck_round_trips(tmp_path):
    """The UHF HF-3c child as a parent in its turn: still HF3C, still UHF."""
    _copy(tmp_path, SPIN_PARENT)
    child = _child(tmp_path, "HF3C")
    nxt = tmp_path / "next"
    nxt.mkdir()
    shutil.copy(child, nxt / "c.d12")
    r, log = _opt2d12(nxt, "--d12-file", "c.d12", "--calc-type", "SP", "--non-interactive")
    assert r.returncode == 0, log[-3000:]
    decks = [p for p in nxt.glob("*.d12") if p.name != "c.d12"]
    assert len(decks) == 1, log[-3000:]
    text = decks[0].read_text()
    assert re.search(r"^BASISSET\nMINIX\nUHF\nHF3C\nEND\n", text, re.M), text
