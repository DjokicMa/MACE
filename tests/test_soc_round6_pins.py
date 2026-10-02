"""Decks frozen before the round-6 SOC options (soc_ecp, soc_basis, soc_guess).

Every new option is off by default, so these decks must not change unless a
listed deck change says so. The hashes were taken from decks written by
cloud/soc-round5 (c9837e3b) and are updated only by such a change: the
geometry comment of an ECP atom (atom number 2xx) in an opt2d12 child was
"X" before the element symbol was written there.

The second part pins which files of MACE's basis libraries CRYSTAL could
read as they stand (basis_file_check.basis_block_problem).
"""
import hashlib
import os

import pytest

from conftest import REPO_ROOT

import soc_ecp as S
from basis_file_check import basis_block_problem, library_files
from test_soc_decks import (PBTE, SOC_PARENT, _Parser, _NoTTY, cif2d12,  # noqa: F401
                            opt2d12, opt2d12_soc_parent)


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


PINS = {
    "soc_deck_pbte": "52e7e9b1d1dc034ccbd22eafb83e88a1b7c8f46471851239d8a76a400decb668",
    "opt2d12_soc_child": "397a28854e9914cca28af0507d07350cab0e1801d5abd5fa3599ae8b799004b5",
    "cif2d12_soc_au": "6491de160c3dada85f346eb45e6b01f013e18e6ed516a1ab5cd3be59f2556c55",
    "opt2d12_scalar_child": "a816e774a163a29f6c5117fdf45d97cdd3bf79f975cd2db70eb26c325852f85a",
}


def test_soc_deck_default_is_pinned():
    assert _sha(S.soc_deck(PBTE, log=lambda m: None)) == PINS["soc_deck_pbte"]


def test_opt2d12_soc_child_of_a_soc_parent_is_pinned(opt2d12_soc_parent):
    status, decks = opt2d12_soc_parent()
    assert status == 0 and len(decks) == 1
    assert _sha(decks[0].read_text()) == PINS["opt2d12_soc_child"]


def test_cif2d12_soc_deck_is_pinned(cif2d12):
    ok, out = cif2d12("external", soc=True)
    assert ok is True
    assert _sha(out.read_text()) == PINS["cif2d12_soc_au"]


def test_opt2d12_scalar_child_of_an_ecp_parent_is_pinned(opt2d12):
    status, decks = opt2d12(["pbte"])
    assert status == 0
    text = decks["pbte"].read_text()
    assert _sha(text) == PINS["opt2d12_scalar_child"]
    # The geometry lines of the two ECP atoms (282 Pb, 252 Te).
    geometry = [ln for ln in text.split("\n") if " Biso " in ln]
    assert [ln.rsplit(" ", 1)[-1] for ln in geometry] == ["X", "X"]


# --- basis libraries ----------------------------------------------------------

LIBRARIES = REPO_ROOT / "Crystal_d12" / "basis_sets"
# Files CRYSTAL cannot read as they stand (the reason is in the file itself):
# 219, 220 and 227 hold a placeholder sentence where NSHL belongs and "X" for
# every shell charge; 204 announces 2 shells and holds 1; 294 announces 11 and
# holds 9 (in all three libraries).
UNREADABLE = {
    "full.basis.triplezeta": {"294": "it announces 11 shells but holds 9"},
    "full.basis.doublezeta": {"294": "it announces 11 shells but holds 9"},
    "stuttgart": {
        "204": "it announces 2 shells but holds 1",
        "219": "no shell count in its first line; shell charge 'X' is not a number",
        "220": "no shell count in its first line; shell charge 'X' is not a number",
        "227": "no shell count in its first line; shell charge 'X' is not a number",
        "294": "it announces 11 shells but holds 9",
    },
}


@pytest.mark.parametrize("library", sorted(UNREADABLE))
def test_every_library_file_crystal_can_read_passes_the_check(library):
    directory = LIBRARIES / library
    problems = {name: basis_block_problem((directory / name).read_text())
                for name in library_files(str(directory))}
    assert {n: p for n, p in problems.items() if p} == UNREADABLE[library]
    assert len(problems) > 80
    # 231.SDF and __init__.py are not per-element files
    assert not any(n in problems for n in ("231.SDF", "__init__.py"))
    assert set(os.listdir(directory)) - set(problems) <= {"231.SDF", "__init__.py",
                                                          "__pycache__"}


@pytest.mark.parametrize("text,problem", [
    ("8 2\n0 0 1 2.0 1.0\n1.0 1.0\n\n0 1 1 6.0 1.0  \n0.5 1.0 1.0\n", None),
    ("282 1\nHAYWSC\n0 0 1 2.0 1.0\n1.0 1.0\n", None),
    ("282 1\nINPSOC\nINTERNAL 1.0\n2. 0 1 0 0 0 0\n1.0 2.0 0.5 0\n0 0 1 2.0 1.0\n1.0 1.0\n", None),
    ("282 1\nINPUT\n2. 0 2 0 0 0 0\n1.0 2.0 0\n0 0 1 2.0 1.0\n1.0 1.0\n",
     "'1.0 1.0' is not a shell record"),
    ("8 3\n0 0 1 2.0 1.0\n1.0 1.0\n", "it announces 3 shells but holds 1"),
    ("8 1\n0 0 2 2.0 1.0\n1.0 1.0\n", "its last shell announces 2 primitives and has fewer"),
    ("", "it is empty"),
], ids=["blank lines", "library keyword", "inpsoc", "short ecp", "shells", "primitives", "empty"])
def test_basis_block_problem(text, problem):
    got = basis_block_problem(text)
    if problem is None:
        assert got is None
    else:
        assert got is not None and problem in got
