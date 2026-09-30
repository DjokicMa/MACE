"""What CrystalInputParser reads from a deck MACE wrote, in the form the deck
writer (CRYSTALOptToD12.write_d12_file) takes it.

An RHF deck (no Hamiltonian keyword, "RHF [default]", manual p. 123) and the
HF-3c / HFsol-3c decks (manual sec. 5.3.1, pp. 158, 162) were read with no
functional, and written back with a DFT block; SMEAR was read under a key the
writer does not take.
"""
import contextlib
import io
import textwrap

import pytest

from d12_parsers import CrystalInputParser


def parse(path):
    with contextlib.redirect_stdout(io.StringIO()):
        return CrystalInputParser(str(path)).parse()


# ------------------------------------------------------ what the parse reads

def _parse_text(tmp_path, text):
    deck = tmp_path / "deck.d12"
    deck.write_text(textwrap.dedent(text).lstrip("\n"))
    return parse(deck)


DIAMOND = """
    dia
    CRYSTAL
    0 0 0
    227
    3.56700000
    1
    6 1.250000000000E-01 1.250000000000E-01 1.250000000000E-01 Biso 1.000000 C
    {block}BASISSET
    {basis}
    {hamiltonian}TOLINTEG
    7 7 7 7 14
    TOLDEE
    7
    SHRINK
    8 16
    SCFDIR
    MAXCYCLE
    800
    FMIXING
    30
    DIIS
    HISTDIIS
    100
    PPAN
    END
"""


def _diamond(block="", basis="POB-TZVP-REV2", hamiltonian="DFT\nPBE\nENDDFT\n"):
    return textwrap.dedent(DIAMOND).lstrip("\n").format(
        block=block, basis=basis, hamiltonian=hamiltonian)


@pytest.mark.parametrize("hamiltonian,functional", [
    ("", "RHF"),
    ("UHF\n", "UHF"),
    ("HF3C\nEND\n", "HF3C"),
    ("HFSOL3C\nEND\n", "HFSOL3C"),
])
def test_hartree_fock_decks_read_their_hamiltonian(tmp_path, hamiltonian, functional):
    p = _parse_text(tmp_path, _diamond(basis="MINIX", hamiltonian=hamiltonian))
    assert p["functional"] == functional
    assert p["method"] == "HF"


def test_smear_is_read_under_the_writers_key(tmp_path):
    p = _parse_text(tmp_path, _diamond().replace("SCFDIR\n", "SMEAR\n0.005000\nSCFDIR\n"))
    assert p["smearing"] is True and p["smearing_width"] == 0.005
