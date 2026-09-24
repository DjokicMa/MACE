"""A parent without a grid keyword gets a child without one.

CRYSTAL23 then uses its default grid, and its output says which one ("DFT
INTEGRATION GRID INCREASED TO XLGRID"). MACE read that line and wrote the
grid into the child, and the grid prompt offered it as the parent's; the
parent wrote no grid, so the child should not either.
"""
import subprocess
import sys

import pytest

import d12_interactive
import menu_nav
from conftest import REPO_ROOT, TEST_DATA
from d12_parsers import CrystalInputParser

MACE_CLI = REPO_ROOT / "mace_cli"


def _parse(tmp_path, dft_lines, title="t", after=(), opt=(), basis=("BASISSET", "POB-TZVP")):
    deck = tmp_path / "p.d12"
    geometry = ["CRYSTAL", "0 0 0", "227", "3.567", "1", "6 0.125 0.125 0.125"]
    optgeom = ["OPTGEOM", *opt, "ENDOPT"] if opt else []
    deck.write_text("\n".join([title, *geometry, *optgeom, "END", *basis, "DFT",
                               *dft_lines, "ENDDFT", *after, "TOLDEE", "7",
                               "SHRINK", "8 8", "END"]) + "\n")
    return CrystalInputParser(str(deck)).parse()



def test_no_grid_keyword_is_read_as_the_default_grid(tmp_path):
    assert _parse(tmp_path, ["PBE0"])["dft_grid"] == "DEFAULT"


def test_grid_prompt_defaults_to_the_parents_default_grid(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    monkeypatch.setattr(menu_nav, "_REAL_INPUT", lambda prompt="": "")
    assert d12_interactive.configure_dft_grid_with_defaults("PBE0", "DEFAULT") == "DEFAULT"


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


def _run(cwd, *extra, stdin=""):
    cmd = [sys.executable, str(MACE_CLI), "opt2d12", "--out-file", "parent.out",
           "--d12-file", "parent.d12", *extra]
    result = subprocess.run(cmd, cwd=cwd, input=stdin, capture_output=True,
                            text=True, timeout=300)
    log = result.stdout + result.stderr
    decks = [p for p in cwd.glob("*.d12") if p.name != "parent.d12"]
    assert result.returncode == 0, log[-1500:]
    assert len(decks) == 1, (sorted(p.name for p in cwd.iterdir()), log[-1500:])
    return decks[0].read_text().splitlines(), log


PATHS = [
    (("--non-interactive",), ""),
    (("--non-interactive", "--calc-type", "SP"), "n\n" + "\n" * 39),  # engine planless
    ((), "n\n1\n" + "\n" * 60),  # interactive, blank answers
]


@pytest.mark.parametrize("extra, stdin", PATHS)
def test_opt2d12_writes_no_grid_for_a_parent_without_one(tmp_path, extra, stdin):
    lines, _ = _run(_parent(tmp_path, ["PBE0"], grid=False), *extra, stdin=stdin)
    block = lines[lines.index("DFT") + 1:lines.index("ENDDFT")]
    assert not any(r.endswith("GRID") or r == "DEFAULT" for r in block), block
