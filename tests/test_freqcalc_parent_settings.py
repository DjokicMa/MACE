"""CrystalInputParser reads a deck's whole FREQCALC block, and an opt2d12 FREQ
child of a FREQ parent repeats it.

Only NUMDERIV was read from the block. Every other record the FREQ writer
(d12_calc_freq.write_frequency_section) writes - INTENS, INTCPHF, INTRAMAN,
IRSPEC, RAMSPEC, TEMPERAT, ... - was lost, so a FREQ parent that asked for IR
intensities got a child written with NOINTENS, and no FREQ deck MACE wrote
could be written back from its parse. Measured on the corpus: 75 of its 92
FREQ decks carry FREQCALC records other than NOINTENS and NUMDERIV (one of
them NOUSESYMM, which the writer never writes; it is reported as
``freq_unparsed``).

The block is now read into the writer's own keys. RESTART is read too but not
repeated in a child: it restarts the parent's own frequency run from the
FREQINFO.DAT that run wrote (manual sec. 8.2, p. 219).
"""
import contextlib
import io
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import CRYSTALOptToD12 as opt2d12
from conftest import REPO_ROOT
from d12_parsers import CrystalInputParser

MACE_CLI = REPO_ROOT / "mace_cli"
AGBR = Path(__file__).parent / "data" / "opt_restart" / "tqc_agbr"   # a real AgBr run

DECK = """
    dia
    CRYSTAL
    0 0 0
    227
    3.56700000
    1
    6 1.250000000000E-01 1.250000000000E-01 1.250000000000E-01 Biso 1.000000 C
    {block}BASISSET
    POB-TZVP-REV2
    DFT
    PBE
    ENDDFT
    TOLINTEG
    9 9 9 11 38
    TOLDEE
    11
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


def _parse(tmp_path, block):
    deck = tmp_path / "deck.d12"
    deck.write_text(textwrap.dedent(DECK).lstrip("\n").format(block=block))
    with contextlib.redirect_stdout(io.StringIO()):
        return CrystalInputParser(str(deck)).parse()


def test_freqcalc_block_is_read_into_the_writers_settings(tmp_path):
    block = ("FREQCALC\nRESTART\nNUMDERIV\n1\nTEMPERAT\n20 0 400\nINTENS\nINTRAMAN\n"
             "INTCPHF\nFMIXING2\n60\nENDCPHF\nIRSPEC\nRANGE\n0.0 4000.0\nEND\n"
             "RAMSPEC\nRANGE\n0.0 4000.0\nVOIGT\n0.5\nEND\nENDFREQ\n")
    p = _parse(tmp_path, block)
    assert p["calculation_type"] == "FREQ"
    assert p["freq_settings"] == {
        "restart": True, "numderiv": 1, "temprange": (20, 0, 400), "intensities": True,
        "raman": True, "cphf_settings": {"fmixing2": 60}, "irspec": True,
        "spec_range": [0.0, 4000.0], "ramspec": True, "raman_voigt": 0.5,
    }
    assert "freq_unparsed" not in p


def test_ir_intensities_through_cphf(tmp_path):
    p = _parse(tmp_path, "FREQCALC\nINTENS\nINTCPHF\nFMIXING\n60\nTOLALPHA\n4\nENDCPHF\n"
                         "DIELISO\n5.7\nENDFREQ\n")
    assert p["freq_settings"] == {"intensities": True, "ir_method": "CPHF",
                                  "cphf_settings": {"fmixing": 60, "tolalpha": 4},
                                  "dielectric_constant": 5.7}


def test_a_default_freqcalc_block_is_no_settings(tmp_path):
    p = _parse(tmp_path, "FREQCALC\nNOINTENS\nENDFREQ\n")
    assert p["calculation_type"] == "FREQ"
    assert p["freq_settings"] == {}


def test_numderiv_is_read_as_before(tmp_path):
    p = _parse(tmp_path, "FREQCALC\nNUMDERIV\n2\nNOINTENS\nENDFREQ\n")
    assert p["freq_settings"] == {"numderiv": 2}


def test_a_freqcalc_record_the_writer_never_writes_is_reported(tmp_path):
    p = _parse(tmp_path, "FREQCALC\nNOUSESYMM\nNOINTENS\nENDFREQ\n")
    assert p["freq_settings"] == {}
    assert p["freq_unparsed"] == ["NOUSESYMM"]


def test_the_unread_records_never_reach_a_childs_settings():
    from d12_parsers import DECK_TEXT_KEYS
    assert "freq_unparsed" in DECK_TEXT_KEYS


def test_a_child_does_not_restart_the_parents_frequency_run():
    parent = {"restart": True, "intensities": True, "numderiv": 1}
    assert opt2d12.child_freq_settings(parent) == {"intensities": True, "numderiv": 1}
    assert parent["restart"] is True                     # not mutated
    assert opt2d12.child_freq_settings({}) == {}


# ------------------------------------------------ real opt2d12 FREQ children

def _freq_child(tmp_path, freqcalc, *extra):
    """opt2d12 --calc-type FREQ from the real AgBr run, its deck made a FREQ deck."""
    lines = AGBR.with_suffix(".d12").read_text().split("\n")
    start, end = lines.index("OPTGEOM"), lines.index("ENDOPT")
    (tmp_path / "agbr_freq.d12").write_text(
        "\n".join(lines[:start] + freqcalc.split("\n") + lines[end + 1:]))
    shutil.copy(AGBR.with_suffix(".out"), tmp_path / "agbr_freq.out")
    result = subprocess.run(
        [sys.executable, str(MACE_CLI), "opt2d12", "--out-file", "agbr_freq.out",
         "--d12-file", "agbr_freq.d12", "--non-interactive", *extra],
        cwd=tmp_path, input="", capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, (result.stdout + result.stderr)[-1500:]
    kids = [p for p in tmp_path.glob("*.d12") if p.name != "agbr_freq.d12"]
    assert len(kids) == 1, sorted(p.name for p in tmp_path.iterdir())
    lines = kids[0].read_text().split("\n")
    return lines[lines.index("FREQCALC"):lines.index("ENDFREQ") + 1]


def test_freq_child_of_a_freq_parent_keeps_its_freqcalc_records(tmp_path):
    """Before: FREQCALC / NOINTENS / ENDFREQ."""
    child = _freq_child(tmp_path, "FREQCALC\nINTENS\nINTCPHF\nENDCPHF\nENDFREQ",
                        "--calc-type", "FREQ")
    assert child == ["FREQCALC", "INTENS", "INTCPHF", "ENDCPHF", "ENDFREQ"]


def test_freq_child_keeps_the_parents_spectra_but_not_its_restart(tmp_path):
    child = _freq_child(tmp_path, "FREQCALC\nRESTART\nINTENS\nINTRAMAN\nINTCPHF\nENDCPHF\n"
                                  "IRSPEC\nRANGE\n0.0 4000.0\nEND\nENDFREQ",
                        "--calc-type", "FREQ")
    assert child == ["FREQCALC", "INTENS", "INTRAMAN", "INTCPHF", "ENDCPHF",
                     "IRSPEC", "RANGE", "0.0 4000.0", "END", "ENDFREQ"]


def test_freq_settings_of_a_config_file_still_replace_the_parents(tmp_path):
    (tmp_path / "c.json").write_text('{"calculation_type": "FREQ", "freq_settings": {"numderiv": 2}}')
    child = _freq_child(tmp_path, "FREQCALC\nINTENS\nINTCPHF\nENDCPHF\nENDFREQ",
                        "--config-file", "c.json")
    assert child == ["FREQCALC", "NUMDERIV", "2", "NOINTENS", "ENDFREQ"]


def test_a_config_file_without_freq_settings_keeps_the_parents(tmp_path):
    (tmp_path / "c.json").write_text('{"calculation_type": "FREQ"}')
    child = _freq_child(tmp_path, "FREQCALC\nINTENS\nINTCPHF\nENDCPHF\nENDFREQ",
                        "--config-file", "c.json")
    assert child == ["FREQCALC", "INTENS", "INTCPHF", "ENDCPHF", "ENDFREQ"]
