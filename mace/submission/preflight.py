"""Pre-flight validation: let CRYSTAL itself vet a .d12 before it costs a job.

MACE can write a deck that is structurally perfect and still unrunnable — the
measured case is a hexagonal SLAB carrying the 3D space-group number 191 in the
record CRYSTAL reads as a *layer* group (valid range 1-80). That deck MPI_Aborts
right after the title without writing any fort.87, so MACE's fort.87-based error
classification sees nothing at all. Nothing short of running CRYSTAL catches it.

The cheap way to run CRYSTAL without running the calculation is TESTPDIM
(manual page 129, "The program stops after processing of the full input (all
four input blocks) and performing symmetry analysis. The size of the Fock/KS
and density matrices in direct space is printed. No input data are required.").
It is a block-3 keyword (manual keyword index: "TESTPDIM / stop after symmetry
analysis"), and the manual's own advice for a full neighbourhood analysis is
"a complete input deck must be read in (blocks 1-3), and the keyword TESTPDIM
inserted in block 3, to stop execution after the symmetry analysis."

Crucially TESTPDIM stops *after* the basis is loaded, so it also catches a basis
that has no functions for one of the deck's elements. Measured on CRYSTAL23: a
good deck finishes in ~2 s and prints "TESTPDIM TEST - SIZE OF DENSITY MATRIX
EVALUATED"; a deck whose basis lacks the element still reports
"ERROR **** LoadBa **** UNIT CELL NOT NEUTRAL" in ~1 s.

Design rules this module holds to:

* The user's deck is never modified. TESTPDIM is inserted into a copy written
  to a throwaway directory, mirroring how submitcrystal23.sh copies
  ``$DIR/$JOB.d12`` to ``$scratch/$JOB/INPUT``.
* Silence is never success. A run is a PASS only if the TESTPDIM completion
  marker is actually present in out.txt. Anything else is a FAIL or an ERROR.
* "This deck is bad" (FAIL) and "this deck could not be checked" (ERROR) are
  different answers with different exit codes, and are never conflated.
* A structure-only run never reports PASS. It reports SKIPPED, labelled
  "structural check only - CRYSTAL was not run", so it cannot read as
  reassurance about a deck CRYSTAL never saw.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------
PASS = "pass"
FAIL = "fail"
ERROR = "error"
SKIPPED = "skipped"

#: Block-3 record that stops CRYSTAL after the symmetry analysis (manual p.129).
TESTPDIM_RECORD = "TESTPDIM"

#: The distinctive part of the line CRYSTAL23 prints when TESTPDIM completes.
#: Measured verbatim this session: "TESTPDIM TEST - SIZE OF DENSITY MATRIX
#: EVALUATED". Only the trailing phrase is matched so that a change in how the
#: "TESTPDIM TEST -" prefix is spaced/dashed cannot turn a good deck into a
#: FAIL. This string appears nowhere in the ~12 GB test/ corpus of ordinary
#: CRYSTAL outputs, so it cannot collide with normal SCF output.
SUCCESS_MARKER = "SIZE OF DENSITY MATRIX EVALUATED"

#: Records that make a deck stop early on their own, so a TESTPDIM inserted
#: before the final END would never be reached (or would be redundant).
#: STOP - "Execution stops immediately" (manual p.129);
#: TESTGEOM - "Execution stops after reading the geometry input block" (p.77);
#: TEST[RUN] - "stop after integrals classification and disk storage estimate";
#: TESTPDIM - already present.
EARLY_STOP_RECORDS = ("STOP", "TESTGEOM", "TESTRUN", "TEST", "TESTPDIM")

#: Matches the module both submitcrystal23.sh:28 and
#: tests/basis_coverage/scan_basis.sh:5 load.
CRYSTAL_MODULE = "CRYSTAL/23-intel-2023a"

#: scan_basis.sh:27 allows the serial binary 30 s for a one-atom SCF. A real
#: deck's symmetry analysis is bigger; 120 s is generous and still bounded.
DEFAULT_TIMEOUT = 120

#: out.txt is normally a few hundred KB for a TESTPDIM run, but a deck that
#: loops before stopping could produce far more, and this may run over 1000
#: decks. Read at most this much, head + tail, so one runaway cannot exhaust RAM.
MAX_OUT_BYTES = 4 * 1024 * 1024

_TRUNCATION_NOTE = "... [out.txt truncated by pre-flight] ..."

#: ' ERROR **** <ROUTINE> **** <MESSAGE>' is the shape CRYSTAL uses for a fatal
#: input/setup error. Confirmed in the corpus by
#: test/FAILED_QA/3_dia3_neighb_atoms_too_close.out ("ERROR **** NEIGHB ****
#: USE KEYWORD SMALLDIST TO RUN ANYWAY") and
#: test/FAILED_QA/1_dia_b3lyp_fermi_not_in_interval.out ("ERROR **** ZERO ****
#: FERMI ENERGY NOT IN INTERVAL").
_ERROR_LINE_RE = re.compile(r"ERROR\s+\*{4}")

#: An error that names TESTPDIM itself is our insertion misfiring, not the
#: user's deck being bad. Deliberately anchored to the ERROR shape: CRYSTAL
#: echoes keyword names in ordinary informational lines (e.g. " INFORMATION
#: **** PPAN **** MULLIKEN POPULATION ANALYSIS ..."), so a bare "TESTPDIM
#: appears somewhere in the tail" test would let a genuinely bad deck be
#: excused as a harness problem.
_TESTPDIM_ERROR_RE = re.compile(r"ERROR\s+\*{4}.*TESTPDIM")

_LINE_SPLIT_RE = re.compile(r"(\r\n|\r|\n)")


class DeckStructureError(Exception):
    """The deck cannot be prepared for a TESTPDIM run (not a verdict on it)."""


@dataclass
class RunOutcome:
    """What a :class:`CrystalRunner` reports back."""

    returncode: int
    timed_out: bool = False


@dataclass
class PreflightResult:
    """One deck's verdict."""

    deck: Path
    status: str
    reason: str = ""
    detail: str = ""
    kept: Optional[Path] = None


# --------------------------------------------------------------------------
# Text surgery - exact, never normalising
# --------------------------------------------------------------------------
def split_lines(text: str) -> List[str]:
    """Split ``text`` into lines that each keep their OWN terminator.

    ``"".join(split_lines(t)) == t`` for every ``t``. Unlike
    ``str.splitlines(keepends=True)`` this splits only on CR, LF and CRLF, so a
    form feed or U+2028 inside a deck is left alone rather than silently
    becoming a line break.
    """
    tokens = _LINE_SPLIT_RE.split(text)
    lines = []
    for i in range(0, len(tokens), 2):
        body = tokens[i]
        sep = tokens[i + 1] if i + 1 < len(tokens) else ""
        lines.append(body + sep)
    return lines


def line_terminators(text: str) -> set:
    """The distinct line terminators present in ``text``."""
    return set(_LINE_SPLIT_RE.findall(text))


def read_deck(path) -> str:
    """Read a deck without altering a single byte.

    ``errors="surrogateescape"`` round-trips bytes that are not valid UTF-8
    instead of deleting them, so a deck carrying a stray byte is validated as
    written rather than as silently repaired. The same codec on the write side
    puts those bytes back verbatim. ``newline=""`` disables universal-newline
    translation so CRLF decks stay CRLF.
    """
    with open(path, "r", encoding="utf-8", errors="surrogateescape", newline="") as fh:
        return fh.read()


def write_deck(path, text: str) -> None:
    """Inverse of :func:`read_deck` - byte-for-byte for anything it produced."""
    with open(path, "w", encoding="utf-8", errors="surrogateescape", newline="") as fh:
        fh.write(text)


def _is_end_record(body: str) -> bool:
    """True for a block terminator.

    Manual, END (block 1): "Processing of geometry input block stops when the
    first three characters of the string are ''END''. Any character can follow:
    ENDGEOM, ENDGINP, etc etc." The same three-character rule terminates the
    later blocks (keyword index: "END / terminate processing of block3 input").
    """
    return body.strip().upper().startswith("END")


def find_final_end(lines: Sequence[str]) -> int:
    """Index of the deck's last non-blank record, which must be an END.

    Raises :class:`DeckStructureError` if the last record is not an END.
    Verified over all 401 .d12 files under ``test/``: in every one the final
    record is END and the record immediately before it is PPAN.
    """
    for idx in range(len(lines) - 1, -1, -1):
        body = _LINE_SPLIT_RE.split(lines[idx])[0]
        if not body.strip():
            continue
        if _is_end_record(body):
            return idx
        raise DeckStructureError(
            "deck does not end with an END record (last record is "
            f"{body.strip()!r}); cannot tell where block 3 ends"
        )
    raise DeckStructureError("deck is empty")


def early_stop_record(text: str) -> Optional[str]:
    """The first standalone early-stop keyword in ``text``, or None.

    The first line of a CRYSTAL deck is a free-text title, so it is exempt: a
    deck titled "STOP" is not a deck that stops.
    """
    for line in split_lines(text)[1:]:
        body = _LINE_SPLIT_RE.split(line)[0].strip().upper()
        if body in EARLY_STOP_RECORDS:
            return body
    return None


def uncheckable_reason(text: str) -> Optional[str]:
    """Why this deck cannot be pre-flighted in an isolated directory, or None.

    * EXTERNAL / DLVINPUT - manual, "Geometry input from external geometry
      editor. Keywords: EXTERNAL, DLVINPUT ... The complete geometry input data
      are read from file fort.34." (EXTPRT is what writes that fort.34.) The
      throwaway directory has no fort.34, so a failure would say nothing about
      the deck. No deck under ``test/`` uses either keyword.
    * GUESSP - manual, "The density matrix from a previous run, P0 (direct
      lattice), is read from disk, and used as SCF guess", "read as file
      fort.20". The manual does not say at which point fort.20 is opened
      relative to the symmetry analysis, so whether TESTPDIM reaches it is
      UNVERIFIED; refusing to check is the conservative answer. Two decks under
      ``test/`` use it (both under test/SP/).
    """
    for line in split_lines(text)[1:]:
        body = _LINE_SPLIT_RE.split(line)[0].strip().upper()
        if body in ("EXTERNAL", "DLVINPUT"):
            return (f"deck reads its geometry from fort.34 ({body}); "
                    "a throwaway pre-flight directory has no fort.34")
        if body == "GUESSP":
            return ("deck restarts from a previous density matrix (GUESSP reads "
                    "fort.20); a throwaway pre-flight directory has no fort.20")
    return None


def insert_testpdim(text: str) -> str:
    """Return ``text`` with a TESTPDIM record inserted before its final END.

    Every other byte is preserved exactly, including each line's own terminator
    and a missing terminator on the last line. The inserted record reuses the
    terminator of the END it precedes.

    Raises :class:`DeckStructureError` if the deck already stops early or does
    not end with an END record - in both cases the deck is not necessarily bad,
    it just cannot be pre-flighted this way.
    """
    stopper = early_stop_record(text)
    if stopper is not None:
        raise DeckStructureError(
            f"deck already contains the early-stop record {stopper!r}; "
            "a pre-flight TESTPDIM would be unreachable or redundant"
        )
    lines = split_lines(text)
    idx = find_final_end(lines)
    sep = _LINE_SPLIT_RE.split(lines[idx])
    terminator = sep[1] if len(sep) > 1 else ""
    if not terminator:
        # Final END has no newline of its own; borrow the deck's terminator.
        present = line_terminators(text)
        terminator = present.pop() if len(present) == 1 else "\n"
    lines.insert(idx, TESTPDIM_RECORD + terminator)
    return "".join(lines)


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------
def _collapse(value: str, limit: int = 160) -> str:
    """One-line, whitespace-collapsed, length-capped - for the report only."""
    return " ".join(value.split())[:limit]


def has_success_marker(out_text: str) -> bool:
    """True if any single line of ``out_text`` carries :data:`SUCCESS_MARKER`.

    Deliberately per line and uncapped. Searching a truncated prefix would make
    the marker unfindable (a CRYSTAL output opens with an asterisk banner);
    searching the whole file collapsed into one string would let the marker
    match across a line boundary.
    """
    for line in out_text.splitlines():
        if SUCCESS_MARKER in " ".join(line.split()):
            return True
    return False


def first_error_line(out_text: str) -> Optional[str]:
    """First ' ERROR **** ... ' line of ``out_text``, collapsed, or None."""
    for line in out_text.splitlines():
        if _ERROR_LINE_RE.search(line):
            return _collapse(line)
    return None


def _tail(text: str, lines: int = 20) -> str:
    kept = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(kept[-lines:])


def classify_run(out_text: str, fort87_text: str = "", returncode: int = 0):
    """Classify one TESTPDIM run. Returns ``(status, reason, detail)``.

    Order of questions:

    1. Did TESTPDIM actually complete? Only this can produce a PASS.
    2. Did CRYSTAL print a fatal ERROR line? This mirrors
       tests/basis_coverage/scan_basis.sh:28-29, which greps out.txt for ERROR
       first and only falls back to fort.87 second - the right order here too,
       because the measured layer-group abort writes NO fort.87 at all.
    3. Otherwise, does fort.87 say something?
    4. Otherwise it failed silently. Silence is not success.

    An ERROR line that names TESTPDIM is the harness misfiring, not a verdict
    on the deck, so it is reported as ERROR rather than FAIL.
    """
    if has_success_marker(out_text):
        return PASS, "TESTPDIM completed", ""

    err_line = first_error_line(out_text)
    if err_line:
        if _TESTPDIM_ERROR_RE.search(err_line):
            return (ERROR, "CRYSTAL rejected the inserted TESTPDIM record",
                    err_line)
        return FAIL, err_line, _tail(out_text)

    fort87 = _collapse(_first_nonblank(fort87_text))
    if fort87:
        return FAIL, f"fort.87: {fort87}", _tail(out_text)

    return (FAIL,
            f"no TESTPDIM completion marker and no error text (exit {returncode})",
            _tail(out_text))


def _first_nonblank(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line
    return ""


# --------------------------------------------------------------------------
# Finding and running CRYSTAL
# --------------------------------------------------------------------------
_MODULE_PROBE_CACHE = {}


def reset_module_probe_cache() -> None:
    """Forget the cached ``module load`` probe (used by the tests)."""
    _MODULE_PROBE_CACHE.clear()


def _module_probe() -> Optional[str]:
    """``module purge && module load CRYSTAL/23`` in a login shell, read $EBROOTCRYSTAL.

    ``module purge`` first, matching submitcrystal23.sh:27 and scan_basis.sh:4 -
    without it a CRYSTAL module already loaded on a login node can win and point
    the validator at a different build, which is the worst outcome this feature
    can produce. Cached, so a 1000-deck batch spawns at most one shell.
    """
    if "root" in _MODULE_PROBE_CACHE:
        return _MODULE_PROBE_CACHE["root"]
    root = None
    script = (
        "module purge >/dev/null 2>&1 && "
        f"module load {CRYSTAL_MODULE} >/dev/null 2>&1 && "
        'printf %s "$EBROOTCRYSTAL"'
    )
    try:
        proc = subprocess.run(["bash", "--login", "-c", script],
                              capture_output=True, text=True, timeout=60)
        candidate = (proc.stdout or "").strip()
        if candidate:
            root = candidate
    except (OSError, subprocess.SubprocessError):
        root = None
    _MODULE_PROBE_CACHE["root"] = root
    return root


def _usable(path: Optional[str]) -> Optional[str]:
    if path and os.path.isfile(path) and os.access(path, os.X_OK):
        return path
    return None


def find_crystal_binary(explicit: Optional[str] = None) -> Optional[str]:
    """Locate the SERIAL ``crystal`` binary, or None.

    Serial, not Pcrystal: a TESTPDIM run is a single short symmetry analysis and
    must not need mpirun. tests/basis_coverage/scan_basis.sh:27 runs exactly
    this binary (``"$EBROOTCRYSTAL/bin/crystal"``) and its 12 result CSVs show
    it works on the cluster.

    Search order - explicit path, ``$MACE_CRYSTAL_BIN``, ``$EBROOTCRYSTAL/bin``
    (already set inside a job that loaded the module), ``crystal`` on PATH, then
    a ``module load`` probe as a last resort. Never cached at this level, so the
    answer always reflects the current environment.
    """
    if explicit:
        return _usable(os.path.expanduser(explicit))

    found = _usable(os.environ.get("MACE_CRYSTAL_BIN"))
    if found:
        return found

    root = os.environ.get("EBROOTCRYSTAL")
    if root:
        found = _usable(os.path.join(root, "bin", "crystal"))
        if found:
            return found

    found = _usable(shutil.which("crystal"))
    if found:
        return found

    root = _module_probe()
    if root:
        return _usable(os.path.join(root, "bin", "crystal"))
    return None


class CrystalRunner:
    """Runs ``crystal < INPUT > out.txt 2>&1`` in a directory, with a timeout.

    Mirrors tests/basis_coverage/scan_basis.sh:27. Callable so it can be swapped
    for a fake in tests; :func:`preflight_deck` takes it as a keyword-only
    argument with no default so nothing can accidentally spawn CRYSTAL.
    """

    def __init__(self, binary: str, timeout: int = DEFAULT_TIMEOUT):
        self.binary = binary
        self.timeout = timeout

    def __call__(self, workdir: Path) -> RunOutcome:
        workdir = Path(workdir)
        with open(workdir / "INPUT", "rb") as stdin, \
                open(workdir / "out.txt", "wb") as stdout:
            try:
                proc = subprocess.run([self.binary], stdin=stdin, stdout=stdout,
                                      stderr=subprocess.STDOUT, cwd=str(workdir),
                                      timeout=self.timeout)
            except subprocess.TimeoutExpired:
                return RunOutcome(returncode=-1, timed_out=True)
            except OSError as exc:
                raise DeckStructureError(f"could not run {self.binary}: {exc}")
        return RunOutcome(returncode=proc.returncode)


def _read_capped(path: Path) -> str:
    """Read a file, keeping head AND tail if it is larger than the cap.

    Binary + explicit decode, not text mode: seeking a text stream to anything
    other than a cookie ``tell()`` returned is undefined, and the tail is what
    carries the success marker. Head is kept too because an ERROR line can be
    anywhere.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    try:
        with open(path, "rb") as fh:
            if size <= MAX_OUT_BYTES:
                raw = fh.read()
                return raw.decode("utf-8", "replace")
            half = MAX_OUT_BYTES // 2
            head = fh.read(half)
            fh.seek(size - half)
            tail = fh.read()
    except OSError:
        return ""
    return (head.decode("utf-8", "replace") + "\n" + _TRUNCATION_NOTE + "\n"
            + tail.decode("utf-8", "replace"))


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def preflight_deck(deck, *, runner, keep_dir=None) -> PreflightResult:
    """Pre-flight one deck. The deck file itself is only ever read.

    ``runner=None`` means "structure only": the deck is parsed and prepared but
    CRYSTAL is never consulted, so the verdict is SKIPPED, never PASS.
    """
    deck = Path(deck)
    try:
        text = read_deck(deck)
    except OSError as exc:
        return PreflightResult(deck, ERROR, "unreadable", str(exc))

    terminators = line_terminators(text)
    if len(terminators) > 1:
        return PreflightResult(
            deck, ERROR, "mixed line terminators",
            "CR, LF and CRLF are mixed in this deck; refusing to normalise it, "
            "because the copy CRYSTAL saw would then differ from the file you submit")

    # These are all "the harness cannot check this deck", never "this deck is
    # bad" - the explanation IS the reason, so it reaches the report table.
    blocked = uncheckable_reason(text)
    if blocked:
        return PreflightResult(deck, ERROR, blocked)

    try:
        prepared = insert_testpdim(text)
    except DeckStructureError as exc:
        return PreflightResult(deck, ERROR, str(exc))

    if runner is None:
        return PreflightResult(deck, SKIPPED,
                               "structural check only - CRYSTAL was not run", "")

    workdir = Path(tempfile.mkdtemp(prefix="mace_preflight_"))
    try:
        write_deck(workdir / "INPUT", prepared)
        try:
            outcome = runner(workdir)
        except DeckStructureError as exc:
            return PreflightResult(deck, ERROR, str(exc))
        out_text = _read_capped(workdir / "out.txt")
        if outcome.timed_out:
            return PreflightResult(
                deck, ERROR, "TESTPDIM timed out",
                "a symmetry analysis should finish in seconds; raise --timeout "
                "if this deck is genuinely huge\n" + _tail(out_text))
        fort87 = _read_capped(workdir / "fort.87")
        status, reason, detail = classify_run(out_text, fort87, outcome.returncode)
        kept = None
        if keep_dir is not None:
            kept = _keep(workdir, Path(keep_dir), deck)
        return PreflightResult(deck, status, reason, detail, kept)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _keep(workdir: Path, keep_dir: Path, deck: Path) -> Optional[Path]:
    """Copy a run directory out of tmp so a failure can be inspected."""
    target = keep_dir / deck.stem
    suffix = 1
    while target.exists():
        suffix += 1
        target = keep_dir / f"{deck.stem}_{suffix}"
    try:
        keep_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(workdir, target)
        return target
    except OSError:
        return None


def preflight_decks(decks: Iterable, *, runner, jobs: int = 1,
                    keep_dir=None, show_progress: bool = False) -> List[PreflightResult]:
    """Pre-flight many decks, optionally in parallel. Order is preserved."""
    decks = [Path(d) for d in decks]
    if not decks:
        return []

    if jobs <= 1 or runner is None:
        items = decks
        if show_progress:
            items = _progress(decks, "Pre-flight")
        return [preflight_deck(d, runner=runner, keep_dir=keep_dir) for d in items]

    # Plain threading, NOT concurrent.futures: mace_cli puts mace/ on sys.path,
    # and mace/queue/ shadows the stdlib `queue` module that
    # concurrent.futures.thread imports, so a ThreadPoolExecutor built under the
    # CLI dies with "module 'queue' has no attribute 'SimpleQueue'". Workers
    # take the next index under a lock and are daemons, so Ctrl-C returns
    # immediately instead of draining a queue of pending decks.
    results: List[Optional[PreflightResult]] = [None] * len(decks)
    lock = threading.Lock()
    done = threading.Condition()
    state = {"next": 0, "done": 0}

    def _worker():
        while True:
            with lock:
                idx = state["next"]
                if idx >= len(decks):
                    return
                state["next"] = idx + 1
            try:
                result = preflight_deck(decks[idx], runner=runner, keep_dir=keep_dir)
            except Exception as exc:  # one bad deck must not sink the batch
                result = PreflightResult(decks[idx], ERROR, "pre-flight raised",
                                         repr(exc))
            results[idx] = result
            with done:
                state["done"] += 1
                done.notify_all()

    threads = [threading.Thread(target=_worker, daemon=True)
               for _ in range(min(jobs, len(decks)))]
    for thread in threads:
        thread.start()

    def _completions():
        seen = 0
        while seen < len(decks):
            with done:
                while state["done"] <= seen:
                    done.wait()
            seen += 1
            yield seen

    stream = _completions()
    if show_progress:
        stream = _progress(stream, "Pre-flight", total=len(decks))
    for _ in stream:
        pass
    for thread in threads:
        thread.join()
    return [r for r in results if r is not None]


# --------------------------------------------------------------------------
# Reporting - the only part that needs the MACE ui layer
# --------------------------------------------------------------------------
def _ui():
    """The MACE ui facade if importable, else None (module stays stdlib-only)."""
    try:
        from mace.utils import ui
        return ui
    except ImportError:
        try:
            from utils import ui  # mace_cli puts mace/ on sys.path
            return ui
        except ImportError:
            return None


def _progress(items, description, total=None):
    ui = _ui()
    if ui is None:
        return iter(items)
    return ui.progress(items, total=total, description=description, unit="deck")


def _say(kind: str, text: str) -> None:
    ui = _ui()
    if ui is None:
        stream = sys.stderr if kind in ("warn", "err") else sys.stdout
        print(text, file=stream)
        return
    getattr(ui, kind)(text)


#: How each status is introduced in the report. SKIPPED spells out that CRYSTAL
#: never ran, so it can never be misread as a pass.
_STATUS_LABEL = {
    PASS: "passed - CRYSTAL accepted the deck through symmetry analysis",
    FAIL: "FAILED - CRYSTAL rejected the deck",
    ERROR: "could not be checked",
    SKIPPED: "structural check only - CRYSTAL was not run",
}


def summarize(results: Sequence[PreflightResult]) -> None:
    """Print the grouped report."""
    ui = _ui()
    counts = {s: [r for r in results if r.status == s]
              for s in (FAIL, ERROR, SKIPPED, PASS)}

    for status in (FAIL, ERROR, SKIPPED):
        group = counts[status]
        if not group:
            continue
        if ui is not None:
            ui.rule(f"{len(group)} {_STATUS_LABEL[status]}")
            ui.table(["Deck", "Reason"],
                     [[r.deck.name, _collapse(r.reason, 110)] for r in group])
        else:
            print(f"-- {len(group)} {_STATUS_LABEL[status]}")
            for r in group:
                print(f"   {r.deck.name}: {_collapse(r.reason, 110)}")
        kept = [r for r in group if r.kept]
        for r in kept:
            _say("info", f"kept {r.deck.name} run directory: {r.kept}")

    # The summary goes to stderr (ui.ok/warn/err); flush the tables first so a
    # piped/logged run does not show the verdict above the table it summarises.
    sys.stdout.flush()
    total = len(results)
    n_pass, n_fail = len(counts[PASS]), len(counts[FAIL])
    n_err, n_skip = len(counts[ERROR]), len(counts[SKIPPED])
    parts = [f"{n_pass} passed", f"{n_fail} failed",
             f"{n_err} not checked", f"{n_skip} structure-only"]
    line = f"{total} deck(s): " + ", ".join(parts)
    if n_fail:
        _say("err", line)
    elif n_err:
        _say("warn", line)
    else:
        _say("ok", line)


def exit_code(results: Sequence[PreflightResult]) -> int:
    """0 = nothing wrong, 1 = a deck is bad, 2 = a deck could not be checked.

    A FAIL outranks an ERROR: a deck CRYSTAL actually rejected is the
    actionable finding, and it must not be masked by an unrelated deck the
    harness could not reach.
    """
    if any(r.status == FAIL for r in results):
        return 1
    if any(r.status == ERROR for r in results):
        return 2
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def collect_decks(paths: Sequence[str], recursive: bool = False) -> List[Path]:
    """Resolve paths to .d12 files.

    A directory yields its top-level .d12 files, matching how `mace submit`
    resolves a directory (os.listdir, not a walk); ``--recursive`` walks it.
    """
    decks: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            decks.append(p)
        elif p.is_dir():
            it = sorted(p.rglob("*.d12")) if recursive else sorted(p.glob("*.d12"))
            decks.extend(d for d in it if d.is_file())
        else:
            _say("err", f"Error: {raw} is not a valid file or directory")
    seen, unique = set(), []
    for d in decks:
        key = str(d.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mace preflight",
        description=("Validate .d12 decks by running CRYSTAL with TESTPDIM, which "
                     "stops after the full input is read and the symmetry analysis "
                     "is done (manual p.129). Catches decks that are structurally "
                     "fine but that CRYSTAL rejects."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit codes:
  0  every deck CRYSTAL saw was accepted
  1  at least one deck was REJECTED by CRYSTAL
  2  at least one deck could not be checked (or no decks / no CRYSTAL)

Examples:
  mace preflight mat.d12                 # one deck
  mace preflight decks/ -j 8             # a directory, 8 at a time
  mace preflight decks/ -r --keep bad/   # recurse, keep run dirs to inspect
  mace preflight decks/ --static-only    # no CRYSTAL: structure check only
""")
    parser.add_argument("paths", nargs="*", default=["."],
                        help="D12 files and/or directories (default: .)")
    parser.add_argument("-j", "--jobs", type=int, default=1, metavar="N",
                        help="run N decks concurrently (default: 1)")
    parser.add_argument("-t", "--timeout", type=int, default=DEFAULT_TIMEOUT,
                        metavar="SECS",
                        help=f"per-deck timeout (default: {DEFAULT_TIMEOUT}s)")
    parser.add_argument("-k", "--keep", metavar="DIR",
                        help="copy each run directory into DIR for inspection")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="recurse into subdirectories")
    parser.add_argument("--static-only", action="store_true",
                        help="never run CRYSTAL; report structure-only (never 'pass')")
    parser.add_argument("--crystal-bin", metavar="PATH",
                        help="path to the serial crystal binary")
    parser.add_argument("--no-banner", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))

    decks = collect_decks(args.paths or ["."], recursive=args.recursive)
    if not decks:
        _say("err", "No D12 files found to pre-flight")
        return 2

    if args.static_only:
        runner = None
    else:
        binary = find_crystal_binary(args.crystal_bin)
        if binary is None:
            _say("err", "No CRYSTAL binary found - nothing was checked")
            _say("info", "Set MACE_CRYSTAL_BIN, pass --crystal-bin PATH, load the "
                         f"{CRYSTAL_MODULE} module, or use --static-only")
            return 2
        _say("info", f"Using {binary}")
        runner = CrystalRunner(binary, timeout=args.timeout)

    keep_dir = Path(args.keep) if args.keep else None
    results = preflight_decks(decks, runner=runner, jobs=max(1, args.jobs),
                              keep_dir=keep_dir, show_progress=len(decks) > 1)
    summarize(results)
    return exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
