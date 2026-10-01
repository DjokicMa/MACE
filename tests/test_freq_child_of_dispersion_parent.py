"""No FREQ child of a phonon-dispersion (SCELPHONO) FREQ parent.

A FREQ child repeats its FREQ parent's FREQCALC settings except RESTART
(child_freq_settings), SCELPHONO and DISPERSION included. SCELPHONO is a
geometry-input keyword that builds the phonon supercell (manual sec. 4.21,
p. 73); the geometry CRYSTAL prints after it is the supercell, primitive-cell
atoms first (p. 74), and DISPERSION runs on that supercell (sec. 8.8, p. 232).
So the child starts from the supercell, and its inherited SCELPHONO would
expand the supercell again (2x2x2 of a 2x2x2: a 4x4x4 MgO). opt2d12 refuses
such a child and says to build it from the OPT/SP the dispersion run started
from. Children of other types (SP, OPT) are not touched here.

The parent deck is a synthetic MgO deck read by the real CrystalInputParser;
the output parser is replaced by the dictionary it returns.
"""
import contextlib
import copy
import io
import json
import sys

import pytest

import CRYSTALOptToD12 as M
from d12_parsers import CrystalInputParser

PARENT_DECK = """mgo_freq
CRYSTAL
0 0 0
225
4.21
2
12 0.0 0.0 0.0
8 0.5 0.5 0.5
{scel}FREQCALC
{disp}NUMDERIV
2
ENDFREQ
BASISSET
POB-TZVP-REV2
DFT
PBE0
ENDDFT
TOLINTEG
9 9 9 11 38
TOLDEE
11
SHRINK
8 8
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

SCEL = "SCELPHONO\n2 0 0\n0 2 0\n0 0 2\n"
DISP = "DISPERSION\n"

OUT = {"dimensionality": "CRYSTAL", "spacegroup": 225, "origin_setting": "0 0 0",
       "conventional_cell": [4.21, 4.21, 4.21, 90.0, 90.0, 90.0],
       "coordinates": [{"atom_number": "12", "x": "0.0", "y": "0.0", "z": "0.0",
                        "is_unique": True},
                       {"atom_number": "8", "x": "0.5", "y": "0.5", "z": "0.5",
                        "is_unique": True}],
       "functional": "PBE0", "calculation_type": "FREQ"}


@pytest.fixture
def child(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(M, "CrystalOutputParser",
                        lambda p: type("P", (), {"parse": lambda self: copy.deepcopy(OUT)})())

    def run(calc_type, dispersion=True):
        deck = PARENT_DECK.format(scel=SCEL if dispersion else "",
                                  disp=DISP if dispersion else "")
        (tmp_path / "mgo.d12").write_text(deck)
        (tmp_path / "mgo.out").write_text("")
        with contextlib.redirect_stdout(io.StringIO()):
            parsed = CrystalInputParser("mgo.d12").parse()
        assert ("scelphono" in parsed["freq_settings"]) == dispersion
        (tmp_path / "c.json").write_text(json.dumps({"calculation_type": calc_type}))
        ok, _ = M.process_files("mgo.out", "mgo.d12", config_file="c.json",
                                unattended=True)
        out = capsys.readouterr()
        decks = [p for p in tmp_path.glob("*.d12") if p.name != "mgo.d12"]
        return ok, out.out + out.err, decks
    return run


def test_freq_child_of_a_scelphono_parent_is_refused(child):
    ok, log, decks = child("FREQ")
    assert ok is False
    assert decks == []
    assert "SCELPHONO" in log and "supercell" in log
    assert "expand it again" in log or "expand again" in log


def test_freq_child_of_a_gamma_only_freq_parent_is_still_written(child):
    ok, log, decks = child("FREQ", dispersion=False)
    assert ok is True, log
    lines = decks[0].read_text().splitlines()
    assert "SCELPHONO" not in lines and "FREQCALC" in lines


def test_sp_child_of_a_scelphono_parent_is_not_refused_here(child):
    ok, log, decks = child("SP")
    assert ok is True, log
    assert "SCELPHONO" not in decks[0].read_text().splitlines()
