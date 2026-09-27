"""A deck title holding a keyword must not decide the calculation type.

MACE titles decks after the file name, so an SP of a material called
"..._BULK_OPTGEOM_TZ_opt_..." carries OPTGEOM in its title, and CRYSTAL echoes
that title into the .out (properties runs echo the SCF title too). Several
places used to search the whole file for the keyword and read such SP, FREQ,
BAND and DOSS files as geometry optimizations.

The corpus tests use the real title-trap files under test/ and skip when the
corpus is absent; the text tests at the top use lines taken from those files
and always run.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA
from mace.utils.calc_detection import (
    deck_records, deck_records_for, is_frequency_output,
    is_optimization_output, is_transport_output)

TRAP_TITLE_LINE = (" ./3,4^2T7_CA_BULK_OPTGEOM_TZ_opt_B3LYP-D3-D3_optimized_rev1_sp_"
                   "B3LYP-D3-D3_optim\n")


# ---------------------------------------------------------------------------
# Text level (always runs)
# ---------------------------------------------------------------------------
def test_d12_title_is_not_a_record():
    deck = ("./x_BULK_OPTGEOM_TZ_opt_sp\r\nCRYSTAL\r\n0 0 0\r\n227\r\n"
            "SHRINK\r\nEND\r\n")
    records = deck_records(deck)
    assert 'OPTGEOM' not in records
    assert {'CRYSTAL', 'SHRINK', 'END'} <= records


def test_d12_keyword_record_is_found_with_crlf_and_padding():
    deck = "1_dia_opt\r\nCRYSTAL\r\n  OPTGEOM  \r\nFULLOPTG\r\nENDOPT\r\n"
    assert 'OPTGEOM' in deck_records(deck)


def test_d3_band_title_is_not_a_record():
    deck = "BAND\nmat_BULK_OPTGEOM_doss_BOLTZTRA\n14 16 1000 1 259 1 0\nEND\n"
    records = deck_records(deck, is_d3=True)
    assert 'BAND' in records
    assert not records & {'OPTGEOM', 'DOSS', 'BOLTZTRA'}
    assert 'MAT_BULK_OPTGEOM_DOSS_BOLTZTRA' not in records


def test_d3_first_line_is_a_record():
    assert 'NEWK' in deck_records("NEWK\n12 24\n1 0\nDOSS\n", is_d3=True)
    assert 'NEWK' not in deck_records_for(Path("x.d12"), "NEWK\n12 24\n")
    assert 'NEWK' in deck_records_for(Path("x.d3"), "NEWK\n12 24\n")


def test_echoed_title_is_not_an_optimization_marker():
    assert not is_optimization_output(TRAP_TITLE_LINE * 2)
    assert not is_frequency_output(" ./mat_FREQCALC_FREQUENCY_CALCULATION\n")
    assert not is_transport_output(" ./mat_BOLTZTRA_transport\n")


@pytest.mark.parametrize("line", [
    " * OPT END - CONVERGED * E(AU):  -7.621718784637E+01  POINTS    3 *\n",
    " *                             OPTIMIZATION STARTS                             *\n",
    " FINAL OPTIMIZED GEOMETRY - DIMENSIONALITY OF THE SYSTEM      3\n",
    " CELL OPTIMIZATION - POINT    1\n",
    " INFORMATION **** NEW DEFAULT **** OPTGEOM OPTIMIZES BOTH ATOMIC COORDINATES AND CELL PARAMETERS\n",
    " INFORMATION **** FULLOPTG **** FULL OPTIMIZATION (ATOMS AND CELL PARAMETERS)\n",
])
def test_crystal_printed_optimization_lines_are_markers(line):
    assert is_optimization_output(TRAP_TITLE_LINE + line)


def test_crystal_printed_frequency_and_transport_lines_are_markers():
    assert is_frequency_output("                       FREQUENCY CALCULATION\n")
    assert is_frequency_output(
        " INFORMATION **** READM **** FREQCALC - MULLIKEN POPULATION ANALYSIS NOT PERFORMED\n")
    assert is_transport_output(
        "           THERMOELECTRIC AND ELECTRONIC TRANSPORT PROPERTIES CALCULATION\n")
    assert is_transport_output(
        " TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT BOLTZTRA    TELAPSE     1168.30 TCPU     1111.24\n")


# ---------------------------------------------------------------------------
# Real corpus
# ---------------------------------------------------------------------------
def _corpus():
    if not TEST_DATA.is_dir():
        pytest.skip("test/ data corpus not present (gitignored, ~12GB)")


def _traps(pattern):
    """Corpus files matching ``pattern`` whose title holds OPTGEOM, or skip."""
    _corpus()
    found = sorted(TEST_DATA.glob(pattern))
    if not found:
        pytest.skip(f"no title-trap file matching {pattern!r}")
    return found


def _deck_type(deck: Path):
    """What the deck's records say it is (the reference for the sweeps)."""
    lines = [l.strip().upper() for l in deck.read_text(errors="ignore").splitlines()]
    if deck.suffix == '.d12':
        body = set(lines[1:])
        return 'OPT' if 'OPTGEOM' in body else 'FREQ' if 'FREQCALC' in body else 'SP'
    body = set(lines)
    if 'BOLTZTRA' in body:
        return 'TRANSPORT'
    if 'ECH3' in body or 'POT3' in body:
        return 'CHARGE+POTENTIAL'
    return 'DOSS' if 'DOSS' in body else 'BAND'


def _outputs_with_decks():
    _corpus()
    pairs = []
    for out in sorted(TEST_DATA.rglob("*.out")):
        for ext in ('.d12', '.d3'):
            deck = out.with_suffix(ext)
            if deck.exists():
                pairs.append((out, _deck_type(deck)))
                break
    if not pairs:
        pytest.skip("no .out with a sibling deck under test/")
    return pairs


@pytest.mark.parametrize("pattern,expected", [
    ("SP/*BULK_OPTGEOM*_sp_*.d12", "SP"),
    ("FREQ/*BULK_OPTGEOM*_freq_*.d12", "FREQ"),
    ("BAND/*BULK_OPTGEOM*_band.d3", "BAND"),
])
def test_queue_manager_reads_records_not_the_title(tmp_path, pattern, expected):
    """The content fallback (a file name without a type token)."""
    from mace.queue.manager import EnhancedCrystalQueueManager
    mgr = EnhancedCrystalQueueManager(
        d12_dir=str(tmp_path), db_path=str(tmp_path / "materials.db"),
        enable_tracking=False, organize_outputs=False)
    for deck in _traps(pattern):
        anon = tmp_path / f"mystery{deck.suffix}"
        shutil.copy(deck, anon)
        assert mgr.determine_calc_type_from_file(anon) == expected, deck.name


@pytest.mark.parametrize("pattern", ["SP/*BULK_OPTGEOM*_sp_*.d12",
                                     "FREQ/*BULK_OPTGEOM*_freq_*.d12"])
def test_formula_update_does_not_treat_title_trap_deck_as_opt(pattern):
    from mace.utils.formula_extractor import _updates_formula_as_opt_deck
    for deck in _traps(pattern):
        assert not _updates_formula_as_opt_deck(deck, deck.read_text()), deck.name


def test_formula_update_still_treats_real_opt_decks_as_opt():
    from mace.utils.formula_extractor import _updates_formula_as_opt_deck
    _corpus()
    decks = [d for d in sorted(TEST_DATA.rglob("*.d12")) if _deck_type(d) == 'OPT']
    assert decks
    for deck in decks:
        assert _updates_formula_as_opt_deck(deck, deck.read_text(errors="ignore")), deck.name


@pytest.mark.parametrize("pattern,expected", [
    ("SP/*BULK_OPTGEOM*_sp_*.out", "SP"),
    ("FREQ/*BULK_OPTGEOM*_freq_*.out", "FREQ"),
    ("BAND/*BULK_OPTGEOM*_band.out", "SP"),
    ("DOSS/*BULK_OPTGEOM*_doss.out", "SP"),
])
def test_title_trap_outputs_are_not_optimizations(extractor, pattern, expected):
    from CrystalOutToCif import CrystalOutToCifConverter
    for out in _traps(pattern):
        content = out.read_text(errors="ignore")
        assert 'OPTGEOM' in content  # the trap: only in the echoed title
        assert not is_optimization_output(content), out.name
        geo = extractor._extract_geometry_optimization(content)
        assert geo.get('calculation_type') != 'geometry_optimization', out.name
        assert CrystalOutToCifConverter.detect_calculation_type(None, content) == expected


def test_out_to_cif_script_reports_sp_for_title_trap(tmp_path):
    """The standalone script, run as a user would (no PYTHONPATH)."""
    out = _traps("SP/*BULK_OPTGEOM*_sp_*.out")[0]
    env = {k: v for k, v in __import__('os').environ.items() if k != 'PYTHONPATH'}
    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "Crystal_d12" / "CrystalOutToCif.py"),
         str(out), "--output-dir", str(tmp_path), "--verbose"],
        capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=300)
    assert "Detected calculation type: SP" in r.stdout, r.stdout + r.stderr


def test_every_corpus_output_matches_its_deck():
    """Markers against the deck records, over every .out with a sibling deck."""
    wrong = []
    for out, kind in _outputs_with_decks():
        content = out.read_text(errors="ignore")
        if is_optimization_output(content) != (kind == 'OPT'):
            wrong.append((out.name, kind, 'opt'))
        if is_frequency_output(content) != (kind == 'FREQ'):
            wrong.append((out.name, kind, 'freq'))
        if is_transport_output(content) != (kind == 'TRANSPORT'):
            wrong.append((out.name, kind, 'transport'))
    assert not wrong, wrong[:10]


def test_opt_runs_that_died_early_are_still_optimizations():
    """Killed/aborted OPTs, including one that stopped before any OPT step."""
    for name in ("1_dia_b3lyp_fermi_not_in_interval.out",
                 "3_dia3_neighb_atoms_too_close.out",
                 "3_dia3_opt_killed_signal9_oom.out"):
        out = TEST_DATA / "FAILED_QA" / name
        if not out.exists():
            pytest.skip(f"{out} not present")
        assert is_optimization_output(out.read_text(errors="ignore")), name
    sp = TEST_DATA / "FAILED_QA" / "T7_sp_killed_signal9_scf_converged.out"
    if sp.exists():
        assert not is_optimization_output(sp.read_text(errors="ignore"))
