"""MATDUMP deck generation end to end, and its registration across MACE.

The generator-level guards live here: the 0-D refusal, the wavefunction
provenance refusal, and the fact that nothing is written when a refusal fires.
"A truncated deck is worse than a missing one" is already this codebase's
contract for the other D3 writers, and MATDUMP keeps it.

The happy path runs against the REAL corpus material
``test/SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized`` and the deck it produces is
the one that was actually run on CRYSTAL23 (1247 overlap cells, 2494 Fock
blocks, ENDPROP, no errors). The refusal paths use kilobyte-sized distilled
output so they still run in the corpus-less CI.
"""
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, find_data

sys.path.insert(0, str(REPO_ROOT / "Crystal_d3"))

from d3_config import get_default_d3_config, validate_d3_config  # noqa: E402

MACE_CLI = REPO_ROOT / "mace_cli"


def _run_opt2d3(*args):
    """Invoke the real CLI as a subprocess.

    The real invocation path, not an isolated import: import order and the
    passthrough wiring are part of what is under test.
    """
    return subprocess.run(
        [sys.executable, str(MACE_CLI), "--no-banner", "opt2d3", *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )


def _corpus_material(tmp_path):
    """The real diamond SP material, copied somewhere writable."""
    out = find_data("SP/1_dia*sp*.out", must_contain="MAX G-VECTOR INDEX")
    staged = tmp_path / out.name
    shutil.copy2(out, staged)
    for suffix in (".d12", ".f9"):
        sibling = out.with_suffix(suffix)
        if sibling.exists():
            shutil.copy2(sibling, tmp_path / sibling.name)
    return staged


def _distilled(tmp_path, maxg=1247, dimension="CRYSTAL", stem="mat_sp"):
    """A minimal but real-shaped parent, with a wavefunction beside it."""
    out = tmp_path / f"{stem}.out"
    lines = [
        " *******************************************************************************",
        " TYPE OF CALCULATION :  RESTRICTED CLOSED SHELL",
        " NO.OF VECTORS CREATED 6999 STARS 1021 RMAX   101.51010 BOHR",
        " NUMBER OF AO                36  EXCHANGE OVERLAP TOL        (T3) 10**   -8",
        " ATOMS IN THE ASYMMETRIC UNIT    1 - ATOMS IN THE UNIT CELL:    2",
    ]
    if dimension == "MOLECULE":
        lines.append(" MOLECULAR CALCULATION")
    if maxg is not None:
        lines.append(
            f" MAX G-VECTOR INDEX FOR 1- AND 2-ELECTRON INTEGRALS{maxg:>4}")
    out.write_text("\n".join(lines) + "\n")
    (tmp_path / f"{stem}.d12").write_text(
        f"title\n{dimension}\n0 0 0\n227\n3.54\n1\n6 0.125 0.125 0.125\nEND\n")
    (tmp_path / f"{stem}.f9").write_bytes(b"\x00" * 64)
    return out


# --------------------------------------------------------------------------
# Acceptance criterion 1: the generated deck, from the real material
# --------------------------------------------------------------------------


def test_generated_deck_from_the_real_corpus_material(tmp_path):
    """The deck MACE writes is the deck that was run on CRYSTAL23.

    Derived N = 1247, byte-exact against the manual's record form.
    """
    staged = _corpus_material(tmp_path)
    result = _run_opt2d3("--input", str(staged), "--calc-type", "MATDUMP")
    assert result.returncode == 0, result.stdout + result.stderr

    decks = list(tmp_path.glob("*_matdump.d3"))
    assert len(decks) == 1, [p.name for p in tmp_path.iterdir()]
    assert decks[0].read_text() == "BASISSET\n2\n60 1247\n64 1247\nEND"


def test_generation_reports_the_derived_n_and_credits_the_author(tmp_path):
    """Attribution is a build requirement, so it is an assertion."""
    staged = _corpus_material(tmp_path)
    result = _run_opt2d3("--input", str(staged), "--calc-type", "MATDUMP")
    combined = result.stdout + result.stderr
    assert "N = 1247" in combined
    assert "William Comaskey" in combined
    assert "CITATION: TODO" in combined


def test_explicit_n_overrides_the_derivation(tmp_path):
    staged = _corpus_material(tmp_path)
    result = _run_opt2d3("--input", str(staged), "--calc-type", "MATDUMP",
                         "--n-rvectors", "321")
    assert result.returncode == 0, result.stdout + result.stderr
    deck = next(tmp_path.glob("*_matdump.d3"))
    assert deck.read_text() == "BASISSET\n2\n60 321\n64 321\nEND"


def test_explicit_n_above_the_vector_pool_is_refused_and_writes_nothing(tmp_path):
    staged = _corpus_material(tmp_path)
    result = _run_opt2d3("--input", str(staged), "--calc-type", "MATDUMP",
                         "--n-rvectors", "20000")
    assert list(tmp_path.glob("*_matdump.d3")) == []
    assert "pool" in (result.stdout + result.stderr)


def test_generation_credits_the_author_without_the_corpus(tmp_path):
    """Attribution is a build requirement, so it is asserted in CI too, not
    only on a developer machine that happens to have the 12 GB corpus."""
    staged = _distilled(tmp_path, stem="mat_sp")
    result = _run_opt2d3("--input", str(staged), "--calc-type", "MATDUMP")
    combined = result.stdout + result.stderr
    assert "William Comaskey" in combined
    assert "CITATION: TODO" in combined
    assert "N = 1247" in combined
    assert (tmp_path / "mat_matdump.d3").read_text() == (
        "BASISSET\n2\n60 1247\n64 1247\nEND")


def test_explicit_n_override_without_the_corpus(tmp_path):
    staged = _distilled(tmp_path, stem="mat_sp")
    result = _run_opt2d3("--input", str(staged), "--calc-type", "MATDUMP",
                         "--n-rvectors", "321")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "mat_matdump.d3").read_text() == (
        "BASISSET\n2\n60 321\n64 321\nEND")


def test_explicit_n_above_the_pool_writes_nothing_without_the_corpus(tmp_path):
    staged = _distilled(tmp_path, stem="mat_sp")
    result = _run_opt2d3("--input", str(staged), "--calc-type", "MATDUMP",
                         "--n-rvectors", "20000")
    assert list(tmp_path.glob("*_matdump.d3")) == []
    assert "pool" in (result.stdout + result.stderr)


def test_a_dump_at_or_above_cell_1000_still_generates_a_deck(tmp_path):
    """Layer 1 is stock CRYSTAL and ships unconditionally.

    lcao2wannier may not be installed at all, so a defect in that optional
    package must not stop MACE writing a valid CRYSTAL deck - it only warns.
    The corpus version of this check lives in test_wannier_driver.py; this one
    runs in the corpus-less CI.
    """
    staged = _distilled(tmp_path, maxg=1247, stem="mat_sp")
    result = _run_opt2d3("--input", str(staged), "--calc-type", "MATDUMP")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "mat_matdump.d3").read_text() == (
        "BASISSET\n2\n60 1247\n64 1247\nEND")
    assert "1000" in (result.stdout + result.stderr)


# --------------------------------------------------------------------------
# The 0-D refusal
# --------------------------------------------------------------------------


def test_a_molecule_parent_is_refused_and_no_deck_is_written(tmp_path):
    """CRYSTAL does not error here, so MACE must.

    MEASURED on a real 10-atom EC MOLECULE deck from test/SP/: the dump exits 0
    with ENDPROP, no ERROR and no WARNING, and emits 11.5 MB in which only cell
    1 is real - cells 2..N are all-zero blocks carrying uninitialised lattice
    indices such as (100150200) and (*********).

    Reintroducing the bug - dropping the dimensionality check and trusting
    CRYSTAL to complain - makes this assertion fail.
    """
    staged = _distilled(tmp_path, maxg=1, dimension="MOLECULE", stem="ec_sp")
    result = _run_opt2d3("--input", str(staged), "--calc-type", "MATDUMP")
    combined = result.stdout + result.stderr
    assert list(tmp_path.glob("*_matdump.d3")) == [], "a deck was written anyway"
    assert "MOLECULE" in combined


def test_the_dimensionality_guard_fires_independently_of_the_count(tmp_path):
    """Isolate the 0-D check from the N == 1 check.

    A molecule in a box is still 0-D but does NOT report a count of 1 - the
    measured case gave 27, the 3x3x3 simple-cubic shell closure - so the two
    guards must be independent. With only the N == 1 rule, this deck would be
    accepted and dumped.

    Reintroducing the bug - deleting the dimensionality check and relying on the
    count alone - makes this assertion fail.
    """
    staged = _distilled(tmp_path, maxg=27, dimension="MOLECULE", stem="box_sp")
    result = _run_opt2d3("--input", str(staged), "--calc-type", "MATDUMP")
    assert list(tmp_path.glob("*_matdump.d3")) == [], "a 0-D deck was written"
    assert "MOLECULE" in (result.stdout + result.stderr)


def test_a_real_corpus_molecule_material_is_refused(tmp_path):
    """The same guard, against a real 0-D material from the corpus."""
    deck = find_data("SP/*MOLECULE*.d12", must_contain="MOLECULE")
    out = deck.with_suffix(".out")
    if not out.exists():
        pytest.skip("no .out beside the MOLECULE deck")
    staged = tmp_path / out.name
    shutil.copy2(out, staged)
    shutil.copy2(deck, tmp_path / deck.name)
    (tmp_path / f"{staged.stem}.f9").write_bytes(b"\x00" * 64)

    result = _run_opt2d3("--input", str(staged), "--calc-type", "MATDUMP")
    assert list(tmp_path.glob("*_matdump.d3")) == []
    assert "MOLECULE" in (result.stdout + result.stderr)


# --------------------------------------------------------------------------
# Wavefunction provenance - the GUESSP hard constraint, finally enforced
# --------------------------------------------------------------------------


def test_a_mismatched_wavefunction_pair_is_refused(tmp_path):
    """N comes from the .out; the matrices come from the .f9.

    _copy_wavefunction accepts ANY file matching its naming patterns and checks
    only that one exists, so an `X_sp.out` can be paired with the `X.f9` left by
    an earlier OPT at a different geometry. For MATDUMP that is a wrong model,
    not a degraded one.

    Reintroducing the bug - removing the provenance check - makes this assertion
    fail, because the generator would happily write a deck at the .out's N using
    the OPT's wavefunction.
    """
    staged = _distilled(tmp_path, stem="mat_sp")
    (tmp_path / "mat_sp.f9").unlink()
    # An OPT wavefunction for the same base name, a different calculation.
    (tmp_path / "mat.f9").write_bytes(b"\x00" * 64)

    result = _run_opt2d3("--input", str(staged), "--calc-type", "MATDUMP")
    combined = result.stdout + result.stderr
    assert list(tmp_path.glob("*_matdump.d3")) == []
    assert "mat.f9" in combined
    assert "geometry" in combined and "basis" in combined


def test_the_matching_wavefunction_is_accepted(tmp_path):
    staged = _distilled(tmp_path, stem="mat_sp")
    result = _run_opt2d3("--input", str(staged), "--calc-type", "MATDUMP")
    assert result.returncode == 0, result.stdout + result.stderr
    # The deck is named from the base name, which strips the _sp suffix - the
    # existing D3 convention, unchanged by MATDUMP.
    assert [p.name for p in tmp_path.glob("*_matdump.d3")] == ["mat_matdump.d3"]


def test_a_bare_fort9_is_accepted(tmp_path):
    staged = _distilled(tmp_path, stem="mat_sp")
    (tmp_path / "mat_sp.f9").unlink()
    (tmp_path / "fort.9").write_bytes(b"\x00" * 64)
    result = _run_opt2d3("--input", str(staged), "--calc-type", "MATDUMP")
    assert result.returncode == 0, result.stdout + result.stderr


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


def test_the_cli_accepts_the_calc_type():
    """Reintroducing the bug - omitting MATDUMP from the argparse choices -
    makes this fail with 'invalid choice', which is what it did before."""
    result = _run_opt2d3("--calc-type", "MATDUMP", "--input", "/nonexistent.out")
    assert "invalid choice" not in (result.stdout + result.stderr)


def test_the_calc_type_name_survives_maces_own_parsers():
    """The name must be letters only.

    Two idioms parse a calc type: the engine's ``^([A-Z]+(?:\\+[A-Z]+)?)(_|)(\\d*)$``
    and the ``rstrip('0123456789')`` base-type idiom. 'WANNIER90' parses as
    instance 90 of 'WANNIER' and 'W90' rstrips to 'W'; 'MATDUMP' is safe.
    """
    import re

    pattern = re.compile(r"^([A-Z]+(?:\+[A-Z]+)?)(?:_|)(\d*)$")
    assert pattern.match("MATDUMP").groups() == ("MATDUMP", "")
    assert pattern.match("MATDUMP2").groups() == ("MATDUMP", "2")
    assert "MATDUMP2".rstrip("0123456789") == "MATDUMP"


def test_a_matdump_config_validates():
    """validate_d3_config rejected MATDUMP outright before this change, which
    was a hard gate on any saved configuration."""
    valid, errors = validate_d3_config({"calculation_type": "MATDUMP",
                                        "n_rvectors": "auto"})
    assert valid, errors
    valid, errors = validate_d3_config({"calculation_type": "MATDUMP"})
    assert valid, errors


def test_a_matdump_config_rejects_a_nonsense_count():
    valid, errors = validate_d3_config({"calculation_type": "MATDUMP",
                                        "n_rvectors": -3})
    assert not valid and any("n_rvectors" in e for e in errors)


def test_the_default_config_never_carries_a_numeric_n():
    """There is no safe constant: measured values for one element on one lattice
    span 87..2731 depending on basis and TOLINTEG."""
    assert get_default_d3_config("MATDUMP")["n_rvectors"] == "auto"


def test_the_calc_type_is_in_the_d3_family():
    from mace.workflow.common.constants import D3_CALC_TYPES, CALC_TYPES

    assert "MATDUMP" in D3_CALC_TYPES
    assert CALC_TYPES["MATDUMP"]["depends_on"] == ["SP", "OPT"]


def test_a_matdump_output_does_not_register_as_a_new_material():
    """The _matdump suffix must be a calc-type token, not part of the id.

    Reintroducing the bug - omitting 'matdump' from the suffix stop-word regex -
    makes the id keep the suffix ('diamond_matdump'), which registers every dump
    as a brand-new material. That is a documented prior bug for the other D3
    types, and it is why this is tested on a name where no OTHER stop word can
    end the scan first: '1_dia_sp_matdump' would pass either way, because '_sp'
    already stops it.
    """
    from mace.database.materials import create_material_id_from_file

    assert create_material_id_from_file("diamond_matdump.out") == "diamond"
    assert create_material_id_from_file("Ti6O11_matdump2.out") == "Ti6O11"
    assert (create_material_id_from_file("1_dia_sp_matdump.out")
            == create_material_id_from_file("1_dia_sp.out"))


def test_the_completion_checker_classifies_a_matdump_output(tmp_path):
    from mace.completion_checker import (CALC_TYPE_TO_BUCKET,
                                         categorize_output_file)
    from mace.utils.calc_detection import calc_type_from_filename

    assert CALC_TYPE_TO_BUCKET["MATDUMP"] == "completematdump"
    assert calc_type_from_filename("1_dia_sp_matdump") == "MATDUMP"
    assert calc_type_from_filename("1_dia_sp_matdump.out") == "MATDUMP"

    # A finished dump with its deck beside it, and one without: the deck's
    # records decide the first, the printed matrix header the second.
    dump = ("   OVERLAP MATRIX - CELL N.   1(  0  0  0)\n"
            "   OVERLAP MATRIX - CELL N.1000( -4  1 -3)\n"
            "    TOTAL CPU TIME =      0.97\n")
    with_deck = tmp_path / "1_dia_sp_matdump.out"
    with_deck.write_text(dump)
    (tmp_path / "1_dia_sp_matdump.d3").write_text(
        "BASISSET\n2\n60 1247\n64 1247\nEND\n")
    assert categorize_output_file(with_deck)[0] == "completematdump"

    alone = tmp_path / "diamond_dump.out"
    alone.write_text(dump)
    assert categorize_output_file(alone)[0] == "completematdump"


def test_the_matdump_records_need_the_prtrec_pair():
    """BASISSET alone is the .d12 internal-basis keyword, not a matrix dump."""
    from mace.utils.calc_detection import (PROPERTY_CALC_TYPES, d3_calc_type,
                                           deck_records, is_matdump_output)

    deck = "BASISSET\n2\n60 1247\n64 1247\nEND\n"
    assert d3_calc_type(deck_records(deck, is_d3=True)) == "MATDUMP"
    assert d3_calc_type({"BASISSET", "END"}) is None
    assert d3_calc_type({"60 1247", "64 1247", "END"}) is None
    assert d3_calc_type({"BASISSET", "600 12", "END"}) is None
    # Runs on the properties script like BAND/DOSS
    assert "MATDUMP" in PROPERTY_CALC_TYPES
    assert not is_matdump_output("== SCF ENDED - CONVERGENCE ON ENERGY\n")


def test_the_queue_manager_classifies_a_matdump_deck(tmp_path):
    """Without this the completion callback would fan BAND/DOSS out of a matrix
    dump, which is the bug its own comment records for TRANSPORT."""
    from mace.queue.manager import EnhancedCrystalQueueManager

    deck = tmp_path / "1_dia_sp_matdump.d3"
    deck.write_text("BASISSET\n2\n60 1247\n64 1247\nEND\n")
    manager = EnhancedCrystalQueueManager.__new__(EnhancedCrystalQueueManager)
    assert manager.determine_calc_type_from_file(deck) == "MATDUMP"


def test_the_d3_sniffer_does_not_fire_on_a_crystal_basis_record(tmp_path):
    """The BASISSET collision, covered WITHOUT the corpus so CI runs it.

    A crystal deck's BASISSET record names the internal basis set on the next
    line (POB-TZVP-REV2, SOLDEF2MSVP, ...). A properties matrix-dump record
    instead carries the NPR count and the 60/64 prtrec pairs. Only the second
    shape may match.
    """
    from mace.completion_checker import _detect_calc_type_from_d3

    for basis in ("POB-TZVP", "POB-TZVP-REV2", "SOLDEF2MSVP"):
        deck = tmp_path / "crystal_deck.d3"
        deck.write_text(
            "title\nCRYSTAL\n0 0 0\n227\n3.54\n1\n"
            "6 0.125 0.125 0.125\n"
            f"BASISSET\n{basis}\n"
            "DFT\nEXCHANGE\nPBE\nEND\nSHRINK\n8 8\nEND\n")
        assert _detect_calc_type_from_d3(deck) != "MATDUMP", basis


def test_the_d3_sniffer_does_not_fire_on_a_real_d12(tmp_path):
    """BASISSET is TWO different keywords.

    In a crystal deck it names the internal basis set - all 295 .d12 files in
    the corpus contain it. In a properties deck it requests matrix printing.
    Keying the MATDUMP sniffer on the bare word would reclassify every ordinary
    SP and OPT deck, so it matches the properties record GRAMMAR instead.

    Reintroducing the bug - sniffing for `'BASISSET' in content` - makes this
    assertion fail on the first corpus deck it sees.
    """
    from mace.completion_checker import _detect_calc_type_from_d3

    deck = find_data("SP/*.d12", must_contain="BASISSET")
    staged = tmp_path / "probe.d3"
    shutil.copy2(deck, staged)
    assert _detect_calc_type_from_d3(staged) != "MATDUMP"


def test_the_d3_sniffer_fires_on_a_real_matdump_deck(tmp_path):
    from mace.completion_checker import _detect_calc_type_from_d3

    deck = tmp_path / "probe.d3"
    deck.write_text("BASISSET\n2\n60 1247\n64 1247\nEND\n")
    assert _detect_calc_type_from_d3(deck) == "MATDUMP"


def test_matdump_extracts_no_properties():
    """The dump is tens to hundreds of MB of matrix text. Ingesting any of it
    would be costly and useless, so the type is registered with an empty
    requirement rather than left unknown - an unknown type is never considered
    satisfied and is endlessly re-suggested."""
    from mace.database.analysis.missing_data import MissingDataAnalyzer

    assert MissingDataAnalyzer.CALC_TYPE_PROPERTIES["MATDUMP"]["required"] == []


def test_the_opt2d3_help_names_the_calc_type_and_its_author():
    result = subprocess.run(
        [sys.executable, str(MACE_CLI), "--no-banner", "opt2d3", "--help"],
        capture_output=True, text=True, cwd=str(REPO_ROOT))
    combined = result.stdout + result.stderr
    assert "MATDUMP" in combined
    assert "William Comaskey" in combined
    assert "CITATION: TODO" in combined
