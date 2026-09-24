"""The interactive "modify settings" flow defaults to the parent's settings.

The planless engine path (MACE_PLANLESS_PROGRESSION=1) drives that flow with
blank answers, so every default that ignored the parent reset the child:

* the SP/OPT tolerance menu always defaulted to Standard (7 7 7 7 14 / 7), so
  tight parents lost their TOLINTEG and TOLDEE;
* the internal-basis wrapper waited for an "Enter your choice" prompt that the
  menu never shows ("Select internal basis set"), and its fixed numbers do not
  fit a menu that is filtered per element set, so the basis fell back to
  POB-TZVP-REV2.

The corpus tests run the real ``mace_cli opt2d12`` with the engine's planless
stdin and skip when ``test/`` is absent. The other tests run everywhere.
"""
import shutil
import subprocess
import sys

import pytest

import d12_constants
import d12_interactive
from conftest import REPO_ROOT, TEST_DATA
from d12_interactive import _tolerance_preset_for, select_basis_set_with_defaults

MACE_CLI = REPO_ROOT / "mace_cli"


# ------------------------------------------------------------ tolerance default

@pytest.mark.parametrize("tolerances,expected", [
    ({"TOLINTEG": "7 7 7 7 14", "TOLDEE": 7}, "1"),
    ({"TOLINTEG": "8 8 8 9 24", "TOLDEE": 9}, "2"),
    ({"TOLINTEG": "9 9 9 11 38", "TOLDEE": 11}, "3"),
    ({"TOLINTEG": "8  8 8  9 24", "TOLDEE": "9"}, "2"),   # raw spacing, str TOLDEE
    ({"tolinteg": [9, 9, 9, 11, 38], "toldee": 11}, "3"),
    ({"TOLINTEG": "8 8 8 9 24", "TOLDEE": 7}, "keep"),    # matches no preset
    ({"TOLINTEG": "7 7 7 8 16", "TOLDEE": 8}, "keep"),
    ({}, "1"),
    (None, "1"),
])
def test_tolerance_default_follows_the_parent(tolerances, expected):
    assert _tolerance_preset_for(tolerances) == expected


# ---------------------------------------------------------------- basis default

def _pick_internal(monkeypatch, current_basis):
    """Run the wrapper on a filtered, renumbered internal menu, answering blank."""
    menu = {"1": "STO-3G", "2": "POB-DZVP-REV2", "3": "POB-TZVP-REV2", "4": "POB-TZVP"}
    seen = {}

    def fake_select_basis_set(elements, method, functional, shared_mode):
        # What d12_constants.select_basis_set does for the internal menu.
        default = next(n for n, v in menu.items() if v == "POB-TZVP-REV2")
        choice = d12_constants.get_user_input("Select internal basis set", menu, default)
        return {"basis_set_type": "INTERNAL", "basis_set": menu[choice]}

    def blank_answer(prompt, options, default):
        seen["default"] = default
        return default

    monkeypatch.setattr(d12_interactive, "select_basis_set", fake_select_basis_set)
    monkeypatch.setattr(d12_constants, "get_user_input", blank_answer)
    result = select_basis_set_with_defaults(
        [], "DFT", "PBE0", current_basis_type="INTERNAL", current_basis=current_basis)
    return result["basis_set"], seen["default"]


def test_internal_basis_default_is_the_parents_entry(monkeypatch):
    assert _pick_internal(monkeypatch, "POB-TZVP") == ("POB-TZVP", "4")


def test_internal_basis_match_ignores_case(monkeypatch):
    assert _pick_internal(monkeypatch, "pob-dzvp-rev2")[0] == "POB-DZVP-REV2"


def test_unknown_internal_basis_keeps_the_menus_default(monkeypatch):
    assert _pick_internal(monkeypatch, "def2-QZVP") == ("POB-TZVP-REV2", "3")


def test_get_user_input_is_restored(monkeypatch):
    original = d12_constants.get_user_input
    monkeypatch.setattr(d12_interactive, "select_basis_set",
                        lambda *a: {"basis_set_type": "INTERNAL", "basis_set": "X"})
    select_basis_set_with_defaults([], current_basis_type="INTERNAL", current_basis="X")
    assert d12_constants.get_user_input is original


# --------------------------------------------------- real planless SP generation

TIGHT_PARENTS = [
    "OPT/Ag1Cl3_sym_CRYSTAL_OPT_symm_PBE-D3_POB-TZVP-REV2_opt_B3LYP-D3-D3_optimized",
    "OPT/4LG_2x2_AA_opt_HSESOL3C_optimized",
]
POB_TZVP_PARENT = "OPT/1_dia_opt_BULK_OPTGEOM"   # internal BASISSET POB-TZVP


def _copy_parent(stem, tmp_path):
    src = TEST_DATA / f"{stem}.out"
    if not src.exists():
        pytest.skip("test/ corpus not present (gitignored, ~12GB)")
    shutil.copy(src, tmp_path)
    shutil.copy(src.with_suffix(".d12"), tmp_path)
    return src.stem


def _planless_sp(tmp_path, name):
    """The engine's planless SP call: the interactive flow on blank answers."""
    result = subprocess.run(
        [sys.executable, str(MACE_CLI), "opt2d12", "--out-file", f"{name}.out",
         "--d12-file", f"{name}.d12", "--non-interactive", "--calc-type", "SP"],
        cwd=tmp_path, input="n\n" + "\n" * 19, capture_output=True, text=True,
        timeout=300)
    assert result.returncode == 0, (result.stdout + result.stderr)[-1500:]
    decks = [p for p in tmp_path.glob("*.d12") if p.name != f"{name}.d12"]
    assert len(decks) == 1, sorted(p.name for p in tmp_path.iterdir())
    return decks[0].read_text().splitlines()


def _after(lines, key):
    return " ".join(lines[lines.index(key) + 1].split())


@pytest.mark.parametrize("stem", TIGHT_PARENTS)
def test_planless_sp_keeps_tight_tolerances(tmp_path, stem):
    name = _copy_parent(stem, tmp_path)
    parent = (tmp_path / f"{name}.d12").read_text().splitlines()
    child = _planless_sp(tmp_path, name)
    assert _after(child, "TOLINTEG") == _after(parent, "TOLINTEG") == "8 8 8 9 24"
    assert _after(child, "TOLDEE") == "9"


def test_planless_sp_keeps_internal_basis(tmp_path):
    name = _copy_parent(POB_TZVP_PARENT, tmp_path)
    child = _planless_sp(tmp_path, name)
    assert _after(child, "BASISSET") == "POB-TZVP"


def test_planless_sp_keeps_non_preset_tolerances(tmp_path):
    """A parent matching no preset keeps its values on a blank answer, rather
    than going through the custom menu (whose own blank default is Standard)."""
    name = _copy_parent(TIGHT_PARENTS[0], tmp_path)
    deck = tmp_path / f"{name}.d12"
    lines = deck.read_text().splitlines()
    i = lines.index("TOLDEE", lines.index("TOLINTEG"))    # the SCF one, not OPTGEOM's
    assert lines[i + 1].strip() == "9"
    lines[i + 1] = "8"
    deck.write_text("\n".join(lines) + "\n")
    child = _planless_sp(tmp_path, name)
    assert _after(child, "TOLINTEG") == "8 8 8 9 24"
    assert _after(child, "TOLDEE") == "8"
