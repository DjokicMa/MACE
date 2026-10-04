"""cif2d12 custom setup asks for the functional from the menus again.

1.1.2 added a prompt for a parent functional CRYSTAL23 does not accept
(opt2d12). It was keyed on "no functional in the options", which is true of
every CIF, so `mace convert` answered 'n' to "use these default settings"
showed "The parent's functional could not be identified" in place of the
category and functional menus, rejected menu numbers, and a blank answer
wrote HSE06 without D3 (the D3 default followed a parent that does not
exist). As in 1.1.1, a CIF gets the menus and D3 defaults to yes.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

pytest.importorskip("ase", reason="CIF parsing needs ase (absent in CI)")

MACE_CLI = REPO_ROOT / "mace_cli"
CIF = Path(__file__).parent / "data" / "1_dia_opt_BULK_OPTGEOM_symm.cif"
# 'n' to the defaults, then Enter for: symmetry defaults, dimensionality,
# calculation type, optimization type, convergence level, MAXTRADIUS, method
BEFORE_CATEGORY = ["n"] + [""] * 7


def _convert(tmp_path, answers):
    cif_dir = tmp_path / "cif"
    cif_dir.mkdir()
    (cif_dir / CIF.name).write_bytes(CIF.read_bytes())
    stdin = "\n".join(answers + [""] * 80) + "\n"
    result = subprocess.run(
        [sys.executable, str(MACE_CLI), "convert", "--no-banner",
         "--cif_dir", str(cif_dir), "--output_dir", str(tmp_path / "out")],
        cwd=tmp_path, input=stdin, capture_output=True, text=True, timeout=300)
    log = result.stdout + result.stderr
    assert result.returncode == 0, log[-3000:]
    decks = sorted((tmp_path / "out").glob("*.d12"))
    assert len(decks) == 1, log[-3000:]
    return decks[0], log


def test_blank_answers_give_hse06_d3_from_the_menus(tmp_path):
    deck, log = _convert(tmp_path, BEFORE_CATEGORY)
    assert "could not be identified" not in log
    assert "Available functional categories:" in log
    assert "Add D3 dispersion correction to HSE06? [Y/n]" in log
    assert "_HSE06-D3_" in deck.name
    assert "HSE06-D3" in deck.read_text().splitlines()


def test_menu_numbers_choose_the_functional(tmp_path):
    # Hybrid category, B3LYP, D3 yes
    deck, log = _convert(tmp_path, BEFORE_CATEGORY + ["3", "1", "y"])
    assert "not a CRYSTAL23 functional keyword" not in log
    assert "B3LYP-D3" in deck.read_text().splitlines()
