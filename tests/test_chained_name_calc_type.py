"""A chained MACE file name must not decide the calculation type.

MACE carries every earlier step into a follow-up's name, so an SP of an OPT is
"X_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized.d12" and its properties decks
add "_band", "_doss", "_charge+potential". Name substrings such as "_opt" or
"optim" therefore read almost every chained SP, FREQ and properties deck as an
OPT: `mace submit` recorded them as OPT and sent .d3 decks to the CRYSTAL SCF
script. The records a deck holds, and the lines a run prints, are the type.

The text tests at the top always run; the corpus tests check every file under
test/ and skip when the corpus is absent.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA
from mace.utils.calc_detection import (
    calc_type_from_filename, d3_calc_type, deck_records, is_band_output,
    is_charge_potential_output, is_doss_output, is_frequency_output,
    is_optimization_output, is_transport_output)

CHAIN = "1LiFSI-1EC-conf1_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized"

SP_DECK = f"{CHAIN}_sp\nMOLECULE\n1\n2\n3 0.0 0.0 0.0\nEND\nDFT\nHSESOL3C\nEND\nSHRINK\n1 1\nEND\n"
OPT_DECK = f"{CHAIN}\nMOLECULE\n1\n2\n3 0.0 0.0 0.0\nOPTGEOM\nFULLOPTG\nENDOPT\nEND\nEND\n"
FREQ_DECK = f"{CHAIN}_freq\nMOLECULE\n1\n2\n3 0.0 0.0 0.0\nFREQCALC\nEND\nEND\nEND\n"
CP_DECK = "ECH3\n100\nPOT3\n100\n5\nEND\n"
TRANSPORT_DECK = "NEWK\n12 12 12\n1 0\nBOLTZTRA\nTRANGE\n100 800 50\nEND\nEND\n"
DOSS_DECK = "NEWK\n12 12 12\n1 0\nDOSS\n0 1000 4 11 1 14 0\nEND\n"
BAND_DECK = f"BAND\n{CHAIN}_sp_band\n14 16 1000 1 259 1 0\n0 0 0 8 8 8\nEND\n"

# Lines copied from real outputs under test/
CPU_TIME = "    TOTAL CPU TIME =    12.345\n"
FREQ_BANNER = ("\n ******************************************************\n"
               "                 FREQUENCY CALCULATION\n"
               " ******************************************************\n")
DOSS_LINES = (" TOTAL AND PROJECTED DENSITY OF STATES - FOURIER LEGENDRE METHOD\n"
              " FROM BAND    4 TO BAND   11 ENERGY RANGE -0.36749E+00  0.73499E+00\n"
              " TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT DOSS        TELAPSE        1.46 TCPU        0.80\n")
BAND_LINES = (" *  BAND STRUCTURE                                                             *\n"
              " *  FROM BAND   1 TO BAND  36                                                  *\n"
              " TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT BAND        TELAPSE        1.68 TCPU        0.92\n")
CP_LINES = (" TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT ECH3 START  TELAPSE        1.06 TCPU        0.51\n"
            " TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT POT3        TELAPSE      566.08 TCPU      513.93\n")


def _manager(tmp_path):
    from mace.queue.manager import EnhancedCrystalQueueManager
    return EnhancedCrystalQueueManager(
        d12_dir=str(tmp_path), db_path=str(tmp_path / "materials.db"),
        enable_tracking=False, organize_outputs=False)


# ---------------------------------------------------------------------------
# Text level (always runs)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,expected", [
    (f"{CHAIN}.d12", "OPT"),
    (f"{CHAIN}_sp_HSESOL3C_optimized.d12", "SP"),
    (f"{CHAIN}_freq_HSESOL3C_optimized_temp.out", "FREQ"),
    (f"{CHAIN}_CRYSTAL_SP_symm_HSESOL3C_SOLDEF2MSVP_band.d3", "BAND"),
    (f"{CHAIN}_sp_HSESOL3C_optimized_charge+potential.d3", "CHARGE+POTENTIAL"),
    ("C1-RCSR-ana_optimized_TRANSPORT.d3", "TRANSPORT"),
    ("mat_opt2.out", "OPT"),
    ("mat_doss3.d3", "DOSS"),
    (f"{CHAIN}_sp_HSESOL3C_optimized_matdump.d3", "MATDUMP"),
    ("x_optimized.d12", None),          # "_optimized" is not "_opt"
    ("C1-RCSR-sph_optimized.d12", None),  # "_sph" is not "_sp"
])
def test_last_type_token_of_a_name(name, expected):
    assert calc_type_from_filename(name) == expected


@pytest.mark.parametrize("name,deck,expected", [
    (f"{CHAIN}_sp_HSESOL3C_optimized.d12", SP_DECK, "SP"),
    (f"{CHAIN}_freq_HSESOL3C_optimized.d12", FREQ_DECK, "FREQ"),
    (f"{CHAIN}.d12", OPT_DECK, "OPT"),
    # the deck decides even against its own name
    ("4LG_FSI_TopBottom_2x2_ABAB_SLAB_OPT_FSI.d12", SP_DECK, "SP"),
    (f"{CHAIN}_sp_HSESOL3C_optimized_charge+potential.d3", CP_DECK, "CHARGE+POTENTIAL"),
    ("C1-RCSR-ana_optimized_TRANSPORT.d3", TRANSPORT_DECK, "TRANSPORT"),
    (f"{CHAIN}_sp_HSESOL3C_optimized_doss.d3", DOSS_DECK, "DOSS"),
    (f"{CHAIN}_sp_HSESOL3C_optimized_band.d3", BAND_DECK, "BAND"),
    # a .d3 with no known record is still a properties deck, never SP/OPT
    (f"{CHAIN}_sp_HSESOL3C_optimized.d3", "NEWK\n12 12\n1 0\nEND\n", "BAND"),
    ("mystery_doss.d3", "NEWK\n12 12\n1 0\nEND\n", "DOSS"),
    # BASISSET with 60/64 prtrec records is a matrix dump, whatever the name
    (f"{CHAIN}_sp_HSESOL3C_optimized.d3", "BASISSET\n2\n60 60\n64 60\nEND\n",
     "MATDUMP"),
    ("mystery_matdump.d3", "NEWK\n12 12\n1 0\nEND\n", "MATDUMP"),
])
def test_queue_manager_reads_the_deck_not_the_name(tmp_path, name, deck, expected):
    path = tmp_path / name
    path.write_text(deck)
    assert _manager(tmp_path).determine_calc_type_from_file(path) == expected


def test_unreadable_deck_falls_back_to_the_name(tmp_path):
    mgr = _manager(tmp_path)
    assert mgr.determine_calc_type_from_file(
        tmp_path / f"{CHAIN}_freq_x_optimized.d12") == "FREQ"
    assert mgr.determine_calc_type_from_file(tmp_path / "missing_charge.d3") == "CHARGE+POTENTIAL"
    assert mgr.determine_calc_type_from_file(tmp_path / "missing.d12") == "SP"


@pytest.mark.parametrize("deck,expected", [
    (CP_DECK, "CHARGE+POTENTIAL"),
    ("ECHG\n0 100\nCOORDINA\n0 0 0\n1 0 0\n0 1 0\nEND\n", "CHARGE+POTENTIAL"),
    ("POTC\n0 1 0\n0.0 0.0 0.0\nEND\n", "CHARGE+POTENTIAL"),
    (TRANSPORT_DECK, "TRANSPORT"),
    (DOSS_DECK, "DOSS"),
    (BAND_DECK, "BAND"),
])
def test_completion_checker_reads_every_d3_record_mace_writes(tmp_path, deck, expected):
    from mace.completion_checker import _detect_calc_type_from_d3
    d3 = tmp_path / "x.d3"
    d3.write_text(deck)
    assert _detect_calc_type_from_d3(d3) == expected
    assert d3_calc_type(deck_records(deck, is_d3=True)) == expected


@pytest.mark.parametrize("name,text,has_opt_end,expected", [
    ("61_crb2_opt_tier7.freq.out", FREQ_BANNER + CPU_TIME, False, "FREQ"),
    ("4LG_FSI_TopBottom_2x2_ABAB_SLAB_OPT_FSI.out", CPU_TIME, False, "SP"),
    (f"{CHAIN}_sp_HSESOL3C_optimized.out", CPU_TIME, False, "SP"),
    ("mat.out", " * OPT END - CONVERGED * E(AU): -1.0 POINTS 3 *\n" + CPU_TIME, True, "OPT"),
    ("JOB.out", DOSS_LINES, False, "DOSS"),
    ("JOB.out", BAND_LINES, False, "BAND"),
    ("JOB.out", CP_LINES, False, "CHARGE+POTENTIAL"),
    # a properties run that stopped early: only its name says what it was
    (f"{CHAIN}_sp_HSESOL3C_optimized_doss.out", CPU_TIME, False, "DOSS"),
])
def test_completed_subtype_reads_the_output_before_the_name(tmp_path, name, text,
                                                           has_opt_end, expected):
    from mace.completion_checker import determine_completed_subtype
    out = tmp_path / name
    out.write_text(text)
    lines = text.splitlines(keepends=True)
    assert determine_completed_subtype(out, lines, has_opt_end=has_opt_end) == expected


def test_sibling_d3_still_decides_a_properties_output(tmp_path):
    from mace.completion_checker import determine_completed_subtype
    (tmp_path / "x_band.d3").write_text(CP_DECK)
    out = tmp_path / "x_band.out"
    out.write_text(CPU_TIME)
    assert determine_completed_subtype(out, [CPU_TIME]) == "CHARGE+POTENTIAL"


def test_properties_output_tells_are_distinct():
    assert is_doss_output(DOSS_LINES) and not is_band_output(DOSS_LINES)
    assert is_band_output(BAND_LINES) and not is_doss_output(BAND_LINES)
    assert is_charge_potential_output(CP_LINES)
    for text in (DOSS_LINES, BAND_LINES, CP_LINES):
        assert not is_frequency_output(text) and not is_optimization_output(text)
        assert not is_transport_output(text)


def test_detector_does_not_call_a_freq_output_sp(tmp_path):
    from mace.recovery.detector import CrystalErrorDetector
    out = tmp_path / f"{CHAIN}_freq_HSESOL3C_optimized.out"
    out.write_text(FREQ_BANNER + CPU_TIME + " CRYSTAL ENDS\n")
    det = CrystalErrorDetector(str(tmp_path), str(tmp_path / "d.db"))
    result = det.analyze_output_file(out)
    assert result["status"] == "completed"
    assert result["calc_type"] == "FREQ"
    assert result["completion_type"] == "frequency_complete"


def test_doss_output_is_not_a_band_structure(tmp_path, extractor):
    out = tmp_path / "x_doss.out"
    text = "INDIRECT ENERGY BAND GAP:   6.0112 eV\n" + DOSS_LINES
    out.write_text(text)
    assert extractor._extract_band_structure_properties(text, out) == {}
    assert extractor._extract_band_structure_properties(BAND_LINES, out)["calculation_type"] == "BAND"


def test_settings_record_charge_potential_and_transport_property_types():
    from mace.utils.settings_extractor import _extract_property_parameters
    assert _extract_property_parameters(CP_DECK)["property_types"] == ["ECH3", "POT3"]
    assert _extract_property_parameters(TRANSPORT_DECK)["property_types"] == ["NEWK", "BOLTZTRA"]
    # the band title (a material name) holds no property keyword
    assert _extract_property_parameters(
        "BAND\nmat_doss_BOLTZTRA\n14 16 1000 1 259 1 0\nEND\n")["property_types"] == ["BAND"]


def test_fresh_database_optgeom_flag_ignores_the_title():
    from mace.database.utils.create_fresh_database import _deck_has_optgeom
    assert not _deck_has_optgeom("x_BULK_OPTGEOM_TZ_opt_sp\n" + SP_DECK.split("\n", 1)[1])
    assert _deck_has_optgeom(OPT_DECK)


def test_populate_scan_records_the_completion_checker_type(tmp_path):
    from mace.database.populate_completed_jobs import scan_for_completed_calculations
    (tmp_path / "61_crb2_opt_tier7.freq.out").write_text(FREQ_BANNER + CPU_TIME)
    (tmp_path / "4LG_FSI_SLAB_OPT_FSI.out").write_text(CPU_TIME)
    found = {Path(c["output_file"]).name: c["calc_type"]
             for c in scan_for_completed_calculations(tmp_path)}
    assert found == {"61_crb2_opt_tier7.freq.out": "FREQ", "4LG_FSI_SLAB_OPT_FSI.out": "SP"}


@pytest.mark.parametrize("wanted,expected", [
    ("SP", True), ("sp2", True), ("OPT", False), ("FREQ", False)])
def test_analyze_filter_type_matches_by_run_type(tmp_path, wanted, expected):
    from mace.utils.property_extractor import _output_matches_calc_type
    out = tmp_path / f"{CHAIN}_sp_HSESOL3C_optimized.out"
    out.write_text(CPU_TIME)
    assert _output_matches_calc_type(out, wanted) is expected


# ---------------------------------------------------------------------------
# Real corpus (skips without test/)
# ---------------------------------------------------------------------------
def _corpus():
    if not TEST_DATA.is_dir():
        pytest.skip("test/ data corpus not present (gitignored, ~12GB)")


def _true_type(deck: Path) -> str:
    content = deck.read_text(errors="ignore")
    if deck.suffix == ".d3":
        return d3_calc_type(deck_records(content, is_d3=True))
    records = deck_records(content)
    return "OPT" if "OPTGEOM" in records else "FREQ" if "FREQCALC" in records else "SP"


def _decks():
    _corpus()
    decks = sorted(TEST_DATA.rglob("*.d12")) + sorted(TEST_DATA.rglob("*.d3"))
    return [(d, _true_type(d)) for d in decks if _true_type(d)]


def _completed_pairs():
    _corpus()
    pairs = []
    for out in sorted(TEST_DATA.rglob("*.out")):
        for ext in (".d12", ".d3"):
            deck = out.with_suffix(ext)
            if deck.exists():
                pairs.append((out, _true_type(deck)))
                break
    return pairs


def test_queue_manager_types_every_corpus_deck_by_its_records(tmp_path):
    mgr = _manager(tmp_path)
    wrong = [(d.name, t, mgr.determine_calc_type_from_file(d))
             for d, t in _decks() if mgr.determine_calc_type_from_file(d) != t]
    assert not wrong, wrong[:5]


def test_every_corpus_ech3_pot3_deck_is_charge_potential():
    from mace.completion_checker import _detect_calc_type_from_d3
    _corpus()
    decks = sorted((TEST_DATA / "ECH3POT3").glob("*.d3"))
    if not decks:
        pytest.skip("no test/ECH3POT3 decks")
    assert {_detect_calc_type_from_d3(d) for d in decks} == {"CHARGE+POTENTIAL"}


def test_mace_check_sorts_every_completed_corpus_output_by_its_deck():
    from mace.completion_checker import CALC_TYPE_TO_BUCKET, categorize_output_file
    wrong = []
    for out, true_type in _completed_pairs():
        category, _ = categorize_output_file(out)
        if category.startswith("complete") and category != CALC_TYPE_TO_BUCKET[true_type]:
            wrong.append((out.name, true_type, category))
    assert not wrong, wrong[:5]


def test_detector_types_every_completed_corpus_output_by_its_deck(tmp_path):
    from mace.recovery.detector import CrystalErrorDetector
    det = CrystalErrorDetector(str(tmp_path), str(tmp_path / "d.db"))
    wrong = []
    for out, true_type in _completed_pairs():
        result = det.analyze_output_file(out)
        if result["status"] == "completed" and result["calc_type"] != true_type:
            wrong.append((out.name, true_type, result["calc_type"]))
    assert not wrong, wrong[:5]


def test_no_corpus_doss_output_is_read_as_a_band_structure(extractor):
    doss = [o for o, t in _completed_pairs() if t == "DOSS"]
    band = [o for o, t in _completed_pairs() if t == "BAND"]
    assert doss and band
    for out in doss:
        assert extractor._extract_band_structure_properties(
            out.read_text(errors="ignore"), out) == {}, out.name
    for out in band:
        assert extractor._extract_band_structure_properties(
            out.read_text(errors="ignore"), out), out.name


def test_mace_analyze_filter_type_keeps_only_that_type(tmp_path):
    """Real invocation: `mace analyze --extract-properties DIR --filter-type SP`."""
    _corpus()
    picks = {}
    for out, true_type in _completed_pairs():
        if true_type in ("OPT", "SP", "FREQ") and true_type not in picks \
                and out.stat().st_size < 2_000_000:
            picks[true_type] = out
    if len(picks) < 3:
        pytest.skip("need an OPT, SP and FREQ output with decks")
    work = tmp_path / "outs"
    work.mkdir()
    for out in picks.values():
        shutil.copy2(out, work / out.name)
        shutil.copy2(out.with_suffix(".d12"), work / out.with_suffix(".d12").name)
    env = {**os.environ, "MACE_NO_BANNER": "1"}
    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "mace_cli"), "analyze", "--extract-properties",
         str(work), "--filter-type", "SP", "--db-path", str(tmp_path / "m.db")],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=600)
    text = r.stdout + r.stderr
    assert "3 output files, 1 match filters" in text, text[-2000:]
    assert picks["SP"].name in text
    assert picks["OPT"].name not in text.split("match filters", 1)[1]


def test_tracked_submit_of_chained_decks_records_their_real_type(monkeypatch, tmp_path):
    """The `mace submit --track` path (SLURM mocked) on real chained decks."""
    from mace.queue.manager import EnhancedCrystalQueueManager
    _corpus()
    want = {"SP": None, "FREQ": None, "CHARGE+POTENTIAL": None, "BAND": None}
    for deck, true_type in _decks():
        if true_type in want and want[true_type] is None and "_opt_" in deck.name:
            want[true_type] = deck
    if None in want.values():
        pytest.skip("corpus lacks a chained deck of every type")
    monkeypatch.chdir(tmp_path)
    mgr = EnhancedCrystalQueueManager(
        d12_dir=str(tmp_path), db_path=str(tmp_path / "materials.db"),
        enable_tracking=True, organize_outputs=False)
    mgr.is_workflow_context = False
    submitted = {}

    def fake_submit(input_file, work_dir, calc_type, submit_script_override=None):
        submitted[Path(input_file).name] = calc_type
        return f"JOB{len(submitted)}"

    mgr.submit_to_slurm = fake_submit
    for true_type, deck in want.items():
        local = tmp_path / deck.name
        shutil.copy2(deck, local)
        calc_id = mgr.submit_calculation(local)
        assert mgr.db.get_calculation(calc_id)["calc_type"] == true_type, deck.name
        assert submitted[deck.name] == true_type
    assert mgr._get_submit_script_for_calc_type("CHARGE+POTENTIAL") == \
        mgr._get_submit_script_for_calc_type("BAND")


@pytest.mark.parametrize("deck,expected", [
    (DOSS_DECK, "DOSS"), (TRANSPORT_DECK, "TRANSPORT"),
    (CP_DECK, "CHARGE+POTENTIAL"), (BAND_DECK, "BAND")])
def test_fresh_database_fallback_records_the_d3_type(tmp_path, deck, expected):
    import inspect
    import mace.database.utils.create_fresh_database as cfd
    d3 = tmp_path / "x.d3"
    d3.write_text(deck)
    cls = next(c for _, c in inspect.getmembers(cfd, inspect.isclass)
               if c.__module__ == cfd.__name__ and hasattr(c, "_extract_basic_settings"))
    settings = cls._extract_basic_settings(object.__new__(cls), d3)
    assert settings["property_type"] == expected


@pytest.mark.parametrize("name", ["MoS2_band.out", "X_dos.out", "Fe_cp.out",
                                  "Cu_charge_2.out", "Ag_transport_film.out"])
def test_finished_scf_named_like_a_properties_run_is_sp(tmp_path, name):
    """A material name ending in a properties token does not make a finished
    SCF a BAND/DOSS/... run; the name only decides for a properties run that
    stopped before printing what it was."""
    from mace.completion_checker import determine_completed_subtype
    src = TEST_DATA / "SP" / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out"
    if not src.exists():
        pytest.skip("test/ corpus not present")
    out = tmp_path / name
    shutil.copy(src, out)
    lines = out.read_text(errors="ignore").splitlines(keepends=True)
    assert determine_completed_subtype(out, lines) == "SP"
