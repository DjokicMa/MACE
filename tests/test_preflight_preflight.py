"""Pre-flight validation of .d12 decks with CRYSTAL's TESTPDIM.

``mace/submission/preflight.py`` inserts TESTPDIM into a COPY of a deck and runs
CRYSTAL on it. TESTPDIM "stops after processing of the full input (all four input
blocks) and performing symmetry analysis" (manual p.129), which is late enough to
catch the failure that motivated this module: a hexagonal SLAB deck carrying the
3D space-group number 191 in the record CRYSTAL reads as a layer group aborts
right after the title and writes NO fort.87, so MACE's fort.87-based error
classification sees nothing at all.

Every test here fails against the behaviour that was specified before review and
passes against what shipped. The specific earlier behaviour each one rejects is
named in its own docstring; the recurring ones are:

* the verdict helper searched a 160-character, whitespace-collapsed prefix of
  out.txt for the success marker - which a CRYSTAL output can never contain,
  because it opens with an asterisk banner, so NO deck could ever pass;
* it also joined the whole file into one string, letting the marker match across
  a line boundary;
* any occurrence of "TESTPDIM" in a failure's tail relabelled that failure a
  harness error, which would excuse the very decks this module exists to catch;
* a deck that merely cannot be pre-flighted (already stops early, no final END,
  needs fort.34/fort.20) was reported FAIL - i.e. as a bad deck;
* ``--static-only`` had no defined status, so it either exited 2 or printed a
  green "pass" for a deck CRYSTAL never read;
* rewriting every line terminator to the first one found, and reading with
  ``errors="ignore"``, so the bytes validated were not the bytes on disk.

CRYSTAL is never invoked here. The run step is exercised through the injected
runner seam; the discovery tests only inspect the environment.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA, find_data

from mace.submission import preflight as pf

MACE_CLI = REPO_ROOT / "mace_cli"

# Two real CRYSTAL outputs that end in a genuine fatal ERROR line. These are the
# only ' ERROR **** ' instances in the whole test/ corpus.
REAL_ERROR_OUTS = {
    "FAILED_QA/3_dia3_neighb_atoms_too_close.out":
        "ERROR **** NEIGHB **** USE KEYWORD SMALLDIST TO RUN ANYWAY",
    "FAILED_QA/1_dia_b3lyp_fermi_not_in_interval.out":
        "ERROR **** ZERO **** FERMI ENERGY NOT IN INTERVAL",
}


@pytest.fixture(autouse=True)
def _clear_module_probe_cache():
    """The ``module load`` probe is cached; without this the discovery tests
    would read whichever answer the first-collected test happened to produce,
    making them pass or fail on collection order."""
    pf.reset_module_probe_cache()
    yield
    pf.reset_module_probe_cache()


def _corpus_decks():
    """Every .d12 under test/, or a loud skip. Never an empty green sweep."""
    if not TEST_DATA.is_dir():
        pytest.skip("test/ data corpus not present (gitignored, ~12GB)")
    decks = sorted(TEST_DATA.rglob("*.d12"))
    if not decks:
        pytest.skip(f"no .d12 files under {TEST_DATA}")
    return decks


# ==========================================================================
# Text surgery - exactness
# ==========================================================================
def test_split_lines_round_trips_every_corpus_deck():
    """``"".join(split_lines(t)) == t``, byte for byte, on real decks.

    Rejects the earlier ``nl.join(lines)`` approach, which rewrote every
    terminator in the deck to the first one it found.
    """
    decks = _corpus_decks()
    assert len(decks) >= 400, f"corpus looks truncated: {len(decks)} decks"
    for deck in decks:
        text = pf.read_deck(deck)
        assert "".join(pf.split_lines(text)) == text, deck


def test_insert_testpdim_is_byte_exact_on_every_corpus_deck():
    """Inserting TESTPDIM changes exactly one thing: one added record.

    Deleting the inserted line from the result must reproduce the original text
    byte for byte. Measured over the corpus: 401 decks, all reproduce exactly.
    Rejects both terminator normalisation and ``errors="ignore"`` (which
    silently deletes undecodable bytes before they are ever validated).
    """
    decks = _corpus_decks()
    assert len(decks) >= 400, f"corpus looks truncated: {len(decks)} decks"
    for deck in decks:
        original = pf.read_deck(deck)
        prepared = pf.insert_testpdim(original)
        lines = pf.split_lines(prepared)
        idx = [i for i, ln in enumerate(lines) if ln.strip() == "TESTPDIM"]
        assert len(idx) == 1, f"{deck}: expected one TESTPDIM, got {len(idx)}"
        del lines[idx[0]]
        assert "".join(lines) == original, deck


def test_testpdim_lands_immediately_before_the_final_end():
    """The record must be inside block 3, right before the deck's final END.

    Measured over the corpus: in all 401 decks the record before the final END
    is PPAN, so after insertion the tail must read PPAN / TESTPDIM / END.
    """
    decks = _corpus_decks()
    for deck in decks:
        prepared = pf.insert_testpdim(pf.read_deck(deck))
        records = [ln.strip() for ln in prepared.splitlines() if ln.strip()]
        assert records[-1].upper().startswith("END"), deck
        assert records[-2] == "TESTPDIM", f"{deck}: tail is {records[-3:]}"


def test_crlf_deck_stays_crlf():
    """A CRLF deck must come back CRLF, including the inserted record.

    test/OPT/1_dia_opt_BULK_OPTGEOM.d12 is one of the three real CRLF decks in
    the corpus. The earlier design normalised to a single terminator, which is
    exactly the "silently repaired by the validator, reappears at submission"
    failure this module exists to prevent.
    """
    deck = TEST_DATA / "OPT" / "1_dia_opt_BULK_OPTGEOM.d12"
    if not deck.is_file():
        pytest.skip("CRLF reference deck not present")
    original = pf.read_deck(deck)
    assert pf.line_terminators(original) == {"\r\n"}, "reference deck is not CRLF"
    prepared = pf.insert_testpdim(original)
    assert pf.line_terminators(prepared) == {"\r\n"}
    assert "\r\nTESTPDIM\r\n" in prepared


def test_read_deck_round_trips_bytes_that_are_not_utf8(tmp_path):
    """A stray non-UTF-8 byte survives read->write unchanged.

    Rejects ``errors="ignore"``, which deletes such a byte, so the deck that got
    validated would not be the deck that gets submitted.
    """
    raw = b"title \xff\nCRYSTAL\nEND\n"
    src = tmp_path / "odd.d12"
    src.write_bytes(raw)
    pf.write_deck(tmp_path / "copy.d12", pf.read_deck(src))
    assert (tmp_path / "copy.d12").read_bytes() == raw


def test_deck_without_final_end_is_refused_not_rewritten(tmp_path):
    """No final END means we cannot tell where block 3 ends -> refuse."""
    deck = tmp_path / "noend.d12"
    deck.write_text("title\nCRYSTAL\n0 0 0\n1\nPPAN\n")
    with pytest.raises(pf.DeckStructureError):
        pf.insert_testpdim(deck.read_text())


def test_free_text_title_is_never_mistaken_for_a_stop_record():
    """A deck TITLED "STOP" still pre-flights; only records count."""
    text = "STOP\nCRYSTAL\n0 0 0\n1\nPPAN\nEND\n"
    assert pf.early_stop_record(text) is None
    assert "\nTESTPDIM\nEND\n" in pf.insert_testpdim(text)


@pytest.mark.parametrize("record", ["STOP", "TESTGEOM", "TEST", "TESTPDIM"])
def test_early_stop_records_are_detected(record):
    """TESTPDIM inserted after one of these would never be reached."""
    text = f"title\nCRYSTAL\n0 0 0\n1\n{record}\nPPAN\nEND\n"
    assert pf.early_stop_record(text) == record
    with pytest.raises(pf.DeckStructureError):
        pf.insert_testpdim(text)


# ==========================================================================
# Verdict
# ==========================================================================
def test_success_marker_is_found_far_past_the_first_160_characters():
    """The marker must be found wherever it is in the file.

    THE regression test for this module. The reviewed spec normalised out.txt
    with ``" ".join(s.split())[:160]`` before searching, and a real CRYSTAL
    output opens with an asterisk banner - so the marker could never be found
    and every single deck, good or bad, was reported FAIL.
    """
    banner = "\n".join([" " + "*" * 78] * 40)
    out = banner + "\n TESTPDIM TEST - SIZE OF DENSITY MATRIX EVALUATED\n"
    assert pf.has_success_marker(out)
    status, reason, _ = pf.classify_run(out)
    assert status == pf.PASS, reason


def test_success_marker_does_not_match_across_a_line_break():
    """Joining the whole file into one string let two innocent lines add up to
    the marker. The search is per line, so this must NOT pass."""
    out = " SOMETHING ABOUT THE SIZE OF DENSITY\n MATRIX EVALUATED HERE\n"
    assert not pf.has_success_marker(out)
    assert pf.classify_run(out)[0] == pf.FAIL


def test_silent_abort_is_a_failure_not_a_pass():
    """No marker, no ERROR line, no fort.87 - the measured layer-group abort.

    This is the shape of the bug the module was written for: CRYSTAL MPI_Aborts
    after the title, writes no fort.87, and a fort.87-driven classifier sees
    nothing. Silence must never be reported as success.
    """
    out = " " + "*" * 78 + "\n a title line\n"
    status, reason, _ = pf.classify_run(out, "", returncode=1)
    assert status == pf.FAIL
    assert "no TESTPDIM completion marker" in reason


@pytest.mark.parametrize("rel,expected", sorted(REAL_ERROR_OUTS.items()))
def test_real_crystal_error_output_is_classified_fail(rel, expected):
    """Against the real CRYSTAL outputs in test/FAILED_QA/, not a fixture."""
    path = TEST_DATA / rel
    if not path.is_file():
        pytest.skip(f"{rel} not present")
    text = path.read_text(errors="ignore")
    status, reason, detail = pf.classify_run(text, "", returncode=0)
    assert status == pf.FAIL
    assert reason == expected
    assert detail, "a failing deck must carry the output tail for the user"


def test_out_txt_error_beats_fort87():
    """scan_basis.sh:28-29 greps out.txt for ERROR first and only falls back to
    fort.87; this keeps that order, because the abort that motivated the module
    writes an out.txt and no fort.87 at all."""
    out = " ERROR **** LoadBa **** UNIT CELL NOT NEUTRAL\n"
    status, reason, _ = pf.classify_run(out, "something else entirely\n")
    assert status == pf.FAIL
    assert reason == "ERROR **** LoadBa **** UNIT CELL NOT NEUTRAL"


def test_fort87_is_used_when_out_txt_says_nothing():
    status, reason, _ = pf.classify_run(" a quiet output\n", " SGINFO SOMETHING WRONG\n")
    assert status == pf.FAIL
    assert reason == "fort.87: SGINFO SOMETHING WRONG"


def test_error_naming_testpdim_is_a_harness_error_not_a_deck_verdict():
    """If CRYSTAL rejects our own inserted record that is our bug, not the
    user's deck."""
    out = " ERROR **** TESTPDIM **** UNKNOWN KEYWORD\n"
    status, reason, _ = pf.classify_run(out)
    assert status == pf.ERROR
    assert "TESTPDIM" in reason


def test_informational_line_naming_testpdim_does_not_excuse_a_bad_deck():
    """CRYSTAL echoes keyword names in ordinary informational lines - the real
    corpus has " INFORMATION **** PPAN **** MULLIKEN POPULATION ANALYSIS ...".

    The earlier spec relabelled any failure whose tail merely CONTAINED the
    string "TESTPDIM" as a harness error, which would have quietly excused the
    bad decks this module was built to catch. Only an ERROR-shaped line that
    names TESTPDIM counts.
    """
    out = (" INFORMATION **** TESTPDIM **** SIZE OF DENSITY MATRIX WILL BE EVALUATED\n"
           " ERROR **** SGINFO **** WRONG LAYER GROUP\n")
    status, reason, detail = pf.classify_run(out)
    assert status == pf.FAIL, f"{reason} / {detail}"
    assert reason == "ERROR **** SGINFO **** WRONG LAYER GROUP"


# ==========================================================================
# Per-deck orchestration
# ==========================================================================
class FakeRunner:
    """Stands in for CRYSTAL: writes a canned out.txt (and optional fort.87)."""

    def __init__(self, out="", fort87=None, returncode=0, timed_out=False):
        self.out, self.fort87 = out, fort87
        self.returncode, self.timed_out = returncode, timed_out
        self.inputs = []

    def __call__(self, workdir):
        workdir = Path(workdir)
        self.inputs.append((workdir / "INPUT").read_bytes())
        (workdir / "out.txt").write_text(self.out)
        if self.fort87 is not None:
            (workdir / "fort.87").write_text(self.fort87)
        return pf.RunOutcome(returncode=self.returncode, timed_out=self.timed_out)


GOOD_OUT = (" " + "*" * 70 + "\n" * 3 +
            " TESTPDIM TEST - SIZE OF DENSITY MATRIX EVALUATED\n")


def test_preflight_deck_never_touches_the_users_file():
    """The deck under test/ must be byte-identical afterwards, and the copy
    CRYSTAL saw must be the deck plus exactly one TESTPDIM record."""
    deck = find_data("OPT/*.d12")
    before = deck.read_bytes()
    runner = FakeRunner(out=GOOD_OUT)
    result = pf.preflight_deck(deck, runner=runner)
    assert result.status == pf.PASS, result.reason
    assert deck.read_bytes() == before
    seen = runner.inputs[0].decode("utf-8", "surrogateescape")
    assert pf.insert_testpdim(pf.read_deck(deck)) == seen


def test_preflight_deck_reports_fail_for_a_rejected_deck():
    deck = find_data("OPT/*.d12")
    runner = FakeRunner(out=" ERROR **** SGINFO **** WRONG LAYER GROUP\n", returncode=1)
    result = pf.preflight_deck(deck, runner=runner)
    assert result.status == pf.FAIL
    assert "SGINFO" in result.reason


def test_uncheckable_deck_is_error_not_fail(tmp_path):
    """A deck that merely cannot be pre-flighted is NOT a bad deck.

    The earlier spec mapped every DeckStructureError to FAIL, which conflated
    "this deck is scientifically bad" (exit 1) with "the harness could not check
    it" (exit 2) - and for a deck containing TESTGEOM or STOP, both of which are
    perfectly valid CRYSTAL, that verdict would simply have been wrong. The
    manual says nothing at all about a deck lacking a final END, so that is
    refused too rather than judged.
    """
    cases = {
        "stops.d12": "title\nCRYSTAL\n0 0 0\nTESTGEOM\nPPAN\nEND\n",
        "noend.d12": "title\nCRYSTAL\n0 0 0\nPPAN\n",
        "ext.d12": "title\nEXTERNAL\nPPAN\nEND\n",
    }
    runner = FakeRunner(out=GOOD_OUT)
    for name, text in cases.items():
        deck = tmp_path / name
        deck.write_text(text)
        result = pf.preflight_deck(deck, runner=runner)
        assert result.status == pf.ERROR, f"{name}: {result.status} {result.reason}"
        # The reason must say WHY, since that is the only field the report shows.
        assert len(result.reason) > 20, f"{name}: unhelpful reason {result.reason!r}"
    assert runner.inputs == [], "CRYSTAL must not be run for an uncheckable deck"


def test_guessp_deck_from_the_corpus_is_error():
    """test/SP/*_graphene*.d12 restart from a density matrix (GUESSP -> fort.20,
    manual p.114). A throwaway directory has none, so refuse rather than judge."""
    deck = find_data("SP/*.d12", must_contain="GUESSP")
    runner = FakeRunner(out=GOOD_OUT)
    result = pf.preflight_deck(deck, runner=runner)
    assert result.status == pf.ERROR
    assert "fort.20" in result.reason
    assert runner.inputs == []


def test_mixed_terminator_deck_is_refused_not_normalised(tmp_path):
    deck = tmp_path / "mixed.d12"
    deck.write_bytes(b"title\r\nCRYSTAL\n0 0 0\r\nPPAN\nEND\n")
    result = pf.preflight_deck(deck, runner=FakeRunner(out=GOOD_OUT))
    assert result.status == pf.ERROR
    assert "mixed line terminators" in result.reason


def test_timeout_is_an_error_not_a_failure():
    deck = find_data("OPT/*.d12")
    result = pf.preflight_deck(deck, runner=FakeRunner(out="", timed_out=True))
    assert result.status == pf.ERROR
    assert "timed out" in result.reason


def test_static_only_reports_skipped_never_pass():
    """``runner=None`` must never produce a PASS.

    The earlier spec had --static-only either exit 2 (contradicting its own
    tests) or print a green "pass" for a structurally-clean deck - which for the
    hexagonal SLAB deck carrying space group 191 would have been a silent false
    assurance about the exact deck this module exists to catch.
    """
    deck = find_data("OPT/*.d12")
    result = pf.preflight_deck(deck, runner=None)
    assert result.status == pf.SKIPPED
    assert result.status != pf.PASS
    assert "CRYSTAL was not run" in result.reason
    assert pf.exit_code([result]) == 0


def test_huge_out_txt_is_capped_but_keeps_the_marker_in_the_tail(tmp_path, monkeypatch):
    """A runaway out.txt must not be read whole, and truncation must not lose
    the verdict: the success marker is at the END of a TESTPDIM run."""
    monkeypatch.setattr(pf, "MAX_OUT_BYTES", 4096)
    big = tmp_path / "out.txt"
    big.write_text(("filler line\n" * 2000) +
                   " TESTPDIM TEST - SIZE OF DENSITY MATRIX EVALUATED\n")
    assert big.stat().st_size > 4096
    text = pf._read_capped(big)
    assert len(text) < big.stat().st_size
    assert pf.has_success_marker(text)


def test_crystal_runner_feeds_input_on_stdin_and_captures_out_txt(tmp_path):
    """The real subprocess wiring, with a stand-in for the binary.

    Mirrors scan_basis.sh:27 (`crystal < INPUT > out.txt 2>&1`), which is the
    only in-repo precedent for invoking the serial binary. FakeRunner cannot
    catch a mistake here because it bypasses subprocess entirely.
    """
    fake = tmp_path / "crystal"
    fake.write_text("#!/bin/sh\n"
                    "grep -q '^TESTPDIM$' && "
                    "echo ' TESTPDIM TEST - SIZE OF DENSITY MATRIX EVALUATED' "
                    "|| echo ' ERROR **** MAIN **** NO TESTPDIM ON STDIN'\n")
    fake.chmod(0o755)

    deck = tmp_path / "deck.d12"
    deck.write_text("title\nCRYSTAL\n0 0 0\n1\nPPAN\nEND\n")
    keep = tmp_path / "kept"
    result = pf.preflight_deck(deck, runner=pf.CrystalRunner(str(fake), timeout=60),
                               keep_dir=keep)
    assert result.status == pf.PASS, f"{result.reason} / {result.detail}"
    assert result.kept is not None and (result.kept / "out.txt").is_file()
    assert "SIZE OF DENSITY MATRIX EVALUATED" in (result.kept / "out.txt").read_text()


# ==========================================================================
# Batch + exit codes
# ==========================================================================
def test_exit_codes():
    """0 nothing wrong / 1 a deck is bad / 2 a deck could not be checked.
    A real FAIL outranks an ERROR so it cannot be masked."""
    p = pf.PreflightResult(Path("a.d12"), pf.PASS)
    s = pf.PreflightResult(Path("b.d12"), pf.SKIPPED)
    f = pf.PreflightResult(Path("c.d12"), pf.FAIL)
    e = pf.PreflightResult(Path("d.d12"), pf.ERROR)
    assert pf.exit_code([]) == 0
    assert pf.exit_code([p, s]) == 0
    assert pf.exit_code([p, e]) == 2
    assert pf.exit_code([p, e, f]) == 1


def test_preflight_decks_preserves_order_when_parallel(tmp_path):
    decks = []
    for i in range(6):
        d = tmp_path / f"deck_{i}.d12"
        d.write_text("title\nCRYSTAL\n0 0 0\nPPAN\nEND\n")
        decks.append(d)
    results = pf.preflight_decks(decks, runner=FakeRunner(out=GOOD_OUT), jobs=4)
    assert [r.deck.name for r in results] == [d.name for d in decks]
    assert {r.status for r in results} == {pf.PASS}


@pytest.mark.skipif(not MACE_CLI.is_file(), reason="mace_cli not found")
def test_parallel_batch_survives_the_mace_queue_shadowing_stdlib_queue(tmp_path):
    """`mace preflight -j N` must work under the real CLI import stack.

    mace_cli does sys.path.insert(0, "<repo>/mace"), and mace/queue/ then
    shadows the stdlib `queue` module. concurrent.futures.thread imports
    `queue`, so building a ThreadPoolExecutor under the CLI dies with
    "module 'queue' has no attribute 'SimpleQueue'" - a crash no in-process
    unit test can see, because pytest never puts mace/ on sys.path.
    """
    fake = tmp_path / "crystal"
    fake.write_text("#!/bin/sh\n"
                    "echo ' TESTPDIM TEST - SIZE OF DENSITY MATRIX EVALUATED'\n")
    fake.chmod(0o755)
    decks = tmp_path / "decks"
    decks.mkdir()
    for i in range(4):
        (decks / f"d{i}.d12").write_text("title\nCRYSTAL\n0 0 0\n1\nPPAN\nEND\n")
    proc = subprocess.run(
        [sys.executable, str(MACE_CLI), "preflight", str(decks),
         "--crystal-bin", str(fake), "-j", "3"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)
    combined = proc.stdout + proc.stderr
    assert "SimpleQueue" not in combined, combined[-2000:]
    assert "Traceback (most recent call last)" not in combined, combined[-2000:]
    assert proc.returncode == 0, f"exit {proc.returncode}\n{combined[-2000:]}"
    assert "4 passed" in combined, combined[-2000:]


def test_collect_decks_directory_is_top_level_unless_recursive(tmp_path):
    """Matches `mace submit`, which resolves a directory with os.listdir."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.d12").write_text("x")
    (tmp_path / "sub" / "b.d12").write_text("x")
    assert [d.name for d in pf.collect_decks([str(tmp_path)])] == ["a.d12"]
    assert [d.name for d in pf.collect_decks([str(tmp_path)], recursive=True)] == \
        ["a.d12", "b.d12"]


# ==========================================================================
# Binary discovery
# ==========================================================================
def _fake_binary(tmp_path, name="crystal"):
    path = tmp_path / name
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def test_explicit_path_wins(tmp_path, monkeypatch):
    binary = _fake_binary(tmp_path)
    monkeypatch.setenv("MACE_CRYSTAL_BIN", "/nonexistent/crystal")
    assert pf.find_crystal_binary(str(binary)) == str(binary)


def test_mace_crystal_bin_env(tmp_path, monkeypatch):
    binary = _fake_binary(tmp_path)
    monkeypatch.setenv("MACE_CRYSTAL_BIN", str(binary))
    monkeypatch.delenv("EBROOTCRYSTAL", raising=False)
    assert pf.find_crystal_binary() == str(binary)


def test_ebrootcrystal_bin_layout(tmp_path, monkeypatch):
    """$EBROOTCRYSTAL/bin/crystal - the layout scan_basis.sh:27 uses."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    binary = _fake_binary(bindir)
    monkeypatch.delenv("MACE_CRYSTAL_BIN", raising=False)
    monkeypatch.setenv("EBROOTCRYSTAL", str(tmp_path))
    assert pf.find_crystal_binary() == str(binary)


def test_absent_crystal_returns_none_without_raising(monkeypatch):
    """No binary anywhere -> None, never an exception.

    subprocess.run is made to raise so the module-load probe cannot succeed;
    the assertion is only that discovery degrades to None.
    """
    monkeypatch.delenv("MACE_CRYSTAL_BIN", raising=False)
    monkeypatch.delenv("EBROOTCRYSTAL", raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(pf.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no shell")))
    assert pf.find_crystal_binary() is None


def test_module_probe_is_cached_but_discovery_is_not(tmp_path, monkeypatch):
    """The shell probe runs at most once per process, but find_crystal_binary
    itself is never cached - otherwise the discovery tests above would each read
    whichever answer ran first, and the environment could change under us."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        raise OSError("no shell")

    monkeypatch.delenv("MACE_CRYSTAL_BIN", raising=False)
    monkeypatch.delenv("EBROOTCRYSTAL", raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(pf.subprocess, "run", fake_run)
    assert pf.find_crystal_binary() is None
    assert pf.find_crystal_binary() is None
    assert len(calls) == 1, f"probe ran {len(calls)} times"

    # Not cached at the top level: a binary appearing in the environment is seen.
    binary = _fake_binary(tmp_path)
    monkeypatch.setenv("MACE_CRYSTAL_BIN", str(binary))
    assert pf.find_crystal_binary() == str(binary)


def test_module_probe_purges_first():
    """submitcrystal23.sh:27 and scan_basis.sh:4 both `module purge` before
    loading. Without it a CRYSTAL module already loaded on a login node can win
    and point the validator at a different build."""
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        raise OSError("no shell")

    original = pf.subprocess.run
    pf.subprocess.run = fake_run
    try:
        pf._module_probe()
    finally:
        pf.subprocess.run = original
    script = seen["cmd"][-1]
    assert script.index("module purge") < script.index("module load")
    assert pf.CRYSTAL_MODULE in script


# ==========================================================================
# Module hygiene + CLI
# ==========================================================================
def test_module_imports_with_stdlib_only():
    """The module must import on a bare cluster python (no rich, no numpy).

    Run in a fresh interpreter so nothing another test already imported can
    mask a hard third-party dependency.
    """
    code = ("import sys; sys.path.insert(0, %r);"
            "import mace.submission.preflight as p;"
            "print(p.TESTPDIM_RECORD)" % str(REPO_ROOT))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=120,
                          env={**os.environ, "PYTHONPATH": ""})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "TESTPDIM"


@pytest.mark.skipif(not MACE_CLI.is_file(), reason="mace_cli not found")
def test_cli_static_only_exits_zero(tmp_path):
    """`mace preflight <dir> --static-only` on clean decks: exit 0, and the
    report must say CRYSTAL was not run rather than 'pass'."""
    deck = tmp_path / "clean.d12"
    deck.write_text("title\nCRYSTAL\n0 0 0\n1\nPPAN\nEND\n")
    proc = subprocess.run(
        [sys.executable, str(MACE_CLI), "preflight", str(tmp_path), "--static-only"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)
    combined = proc.stdout + proc.stderr
    assert "Traceback (most recent call last)" not in combined, combined[-2000:]
    assert proc.returncode == 0, f"exit {proc.returncode}\n{combined[-2000:]}"
    assert "CRYSTAL was not run" in combined, combined[-2000:]
    # The count of decks CRYSTAL accepted must be zero: it was never asked.
    assert "0 passed" in combined, combined[-2000:]
    assert "1 structure-only" in combined, combined[-2000:]


@pytest.mark.skipif(not MACE_CLI.is_file(), reason="mace_cli not found")
def test_cli_no_decks_exits_two(tmp_path):
    """Nothing to check is 'could not check' (2), not success (0)."""
    proc = subprocess.run(
        [sys.executable, str(MACE_CLI), "preflight", str(tmp_path), "--static-only"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)
    assert proc.returncode == 2, proc.stdout + proc.stderr


@pytest.mark.skipif(not MACE_CLI.is_file(), reason="mace_cli not found")
def test_cli_reports_missing_crystal(tmp_path):
    """Without a binary and without --static-only the run must NOT quietly
    succeed: exit 2, and the message must point at the way out."""
    deck = tmp_path / "clean.d12"
    deck.write_text("title\nCRYSTAL\n0 0 0\n1\nPPAN\nEND\n")
    env = {k: v for k, v in os.environ.items()
           if k not in ("MACE_CRYSTAL_BIN", "EBROOTCRYSTAL")}
    env["PATH"] = str(tmp_path / "empty")
    proc = subprocess.run(
        [sys.executable, str(MACE_CLI), "preflight", str(tmp_path)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300, env=env)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 2, f"exit {proc.returncode}\n{combined[-2000:]}"
    assert "No CRYSTAL binary found" in combined, combined[-2000:]
    assert "--static-only" in combined, combined[-2000:]


@pytest.mark.skipif(not MACE_CLI.is_file(), reason="mace_cli not found")
def test_cli_help_lists_preflight():
    """`mace --help` advertises it and `mace preflight --help` explains it."""
    top = subprocess.run([sys.executable, str(MACE_CLI), "--help"],
                         cwd=str(REPO_ROOT), capture_output=True, text=True,
                         timeout=300)
    assert "preflight" in top.stdout, top.stdout[-2000:]
    sub = subprocess.run([sys.executable, str(MACE_CLI), "preflight", "--help"],
                         cwd=str(REPO_ROOT), capture_output=True, text=True,
                         timeout=300)
    assert sub.returncode == 0, (sub.stdout + sub.stderr)[-2000:]
    assert "TESTPDIM" in sub.stdout, sub.stdout[-2000:]
