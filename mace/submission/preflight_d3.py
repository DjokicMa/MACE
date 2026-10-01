"""Pre-flight for .d3 decks: a MATDUMP deck is checked without CRYSTAL.

`mace preflight` vets a .d12 by running the serial ``crystal`` with TESTPDIM.
That means nothing for a .d3, which is a ``properties`` deck, and
``properties`` has no TESTPDIM. A MATDUMP deck can still be checked from what
is on disk, and that is all this module does:

* Record grammar, from the manual: BASISSET (p.310) takes NPR, the number of
  printing options, then NPR prtrec records (p.71: position in the LPRINT
  vector and its value). Print options 60 (overlap) and 64 (Fock/KS) are the
  matrix dump (p.310; Appendix C p.442). MACE writes
  ``BASISSET / 2 / 60 N / 64 N / END``; both counts must be the same N.
* N against the bounds MACE derives from the parent SCF output (see
  Crystal_d3/d3_matdump.py, where each bound is explained and measured): above
  CRYSTAL's vector pool is refused, a MOLECULE parent is refused, below the
  derived count is accepted but reported as a truncation.
* The wavefunction the job script stages: submit_prop.sh runs
  ``cp $DIR/$JOB.f9 $scratch/$JOB/fort.9``, so ``<deck stem>.f9`` must sit
  next to the deck.

Every other .d3 kind is reported as not checked. Nothing here runs CRYSTAL, so
a good MATDUMP deck is reported as structure-only, never as passed.

Stdlib only, like preflight.py; d3_matdump.py is stdlib only as well.
"""

import importlib.util
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_D3_DIR = Path(__file__).resolve().parents[2] / "Crystal_d3"

# A prtrec record as MACE writes it: one "position value" pair per line.
_INT_RE = re.compile(r"[+-]?\d+")

# Print options of the matrix dump (manual p.310).
_OVERLAP, _FOCK = 60, 64


def _matdump_module():
    """Crystal_d3/d3_matdump.py, the one place N's bounds are defined."""
    loaded = sys.modules.get("d3_matdump")
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(
        "d3_matdump", _D3_DIR / "d3_matdump.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {_D3_DIR / 'd3_matdump.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["d3_matdump"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("d3_matdump", None)
        raise
    return module


def _records(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def d3_kind(text: str) -> Optional[str]:
    """The properties calculation a .d3 drives, or None if not recognised."""
    records = {r.upper() for r in _records(text)}
    try:
        from mace.utils.calc_detection import d3_calc_type
    except ImportError:
        return "MATDUMP" if "BASISSET" in records else None
    return d3_calc_type(records)


def parse_matdump_records(text: str) -> Tuple[Optional[int], str]:
    """N from a MATDUMP deck, or (None, why the records are not valid)."""
    records = _records(text)
    if not records or records[0].upper() != "BASISSET":
        return None, ("the deck must open with BASISSET; pre-flight checks only "
                      "the MATDUMP deck MACE writes (BASISSET / NPR / prtrec / END)")
    if len(records) < 2 or not _INT_RE.fullmatch(records[1]):
        return None, ("BASISSET must be followed by NPR, the number of printing "
                      "options (manual p.310)")
    npr = int(records[1])
    if "END" not in (r.upper() for r in records[2:]):
        return None, "no END record closes the deck"
    end = next(i for i, r in enumerate(records) if i >= 2 and r.upper() == "END")
    body = records[2:end]
    pairs = []
    for record in body:
        fields = record.split()
        if len(fields) != 2 or not all(_INT_RE.fullmatch(f) for f in fields):
            return None, (f"prtrec record {record!r} is not a pair of integers "
                          "(position, value; manual p.71)")
        pairs.append((int(fields[0]), int(fields[1])))
    if len(pairs) != npr:
        return None, (f"NPR is {npr} but {len(pairs)} prtrec record(s) follow; "
                      "CRYSTAL reads exactly NPR, so the END would be consumed "
                      "as a record (manual p.310)")
    codes = dict(pairs)
    missing = [str(c) for c in (_OVERLAP, _FOCK) if c not in codes]
    if missing:
        return None, ("a matrix dump needs print options 60 (overlap) and 64 "
                      f"(Fock/KS); missing {', '.join(missing)}")
    if codes[_OVERLAP] != codes[_FOCK]:
        return None, (f"60 asks for {codes[_OVERLAP]} cells and 64 for "
                      f"{codes[_FOCK]}; the conversion needs the same N for both")
    return codes[_OVERLAP], ""


def parent_candidates(deck: Path) -> List[Path]:
    """Parent SCF outputs that `mace opt2d3` could have built this deck from.

    The generator names the deck ``<base>_matdump.d3`` where ``<base>`` is the
    parent's stem with one of _opt/_sp/_OPT/_SP removed.
    """
    stem = deck.stem
    base = stem[:-len("_matdump")] if stem.lower().endswith("_matdump") else stem
    found = []
    for parent_stem in (f"{base}_sp", f"{base}_opt", f"{base}_SP", f"{base}_OPT",
                        base):
        for suffix in (".out", ".log"):
            path = deck.with_name(parent_stem + suffix)
            if path.is_file() and path != deck:
                found.append(path)
    return found


def check_d3_deck(deck: Path, text: str):
    """(status, reason, detail) for a .d3. Statuses are preflight.py's."""
    from mace.submission.preflight import ERROR, REFUSED, SKIPPED

    kind = d3_kind(text)
    if kind != "MATDUMP":
        return (ERROR,
                f"not checked: this is a {kind or 'unrecognised'} properties deck, "
                "and pre-flight checks only MATDUMP .d3 decks (properties has no "
                "TESTPDIM, so nothing short of running it vets the others)", "")

    n, why = parse_matdump_records(text)
    if n is None:
        return REFUSED, f"MATDUMP record: {why}", ""
    if n < 1:
        return REFUSED, f"MATDUMP N = {n}: N must be a positive cell count", ""

    # The wavefunction submit_prop.sh stages.
    f9 = deck.with_suffix(".f9")
    if not f9.is_file():
        bare = deck.with_name("fort.9")
        extra = (f"; a bare {bare.name} is there, but the job script copies "
                 f"{f9.name} (cp $DIR/$JOB.f9 ... /fort.9), so rename it"
                 if bare.is_file() else "")
        return REFUSED, f"no {f9.name} next to the deck{extra}", ""
    if f9.stat().st_size == 0:
        return REFUSED, f"{f9.name} is empty", ""

    parents = parent_candidates(deck)
    if not parents:
        return (ERROR,
                f"MATDUMP N = {n} could not be bounded: no parent SCF output "
                f"found next to the deck (looked for <base>_sp/_opt .out)", "")
    if len(parents) > 1:
        return (ERROR,
                f"MATDUMP N = {n} could not be bounded: more than one possible "
                f"parent ({', '.join(p.name for p in parents)}); pre-flight will "
                "not guess which one wrote the .f9", "")
    parent = parents[0]

    try:
        md = _matdump_module()
    except Exception as exc:  # pragma: no cover - a broken checkout
        return ERROR, f"could not load Crystal_d3/d3_matdump.py: {exc}", ""
    try:
        out_text = parent.read_text(errors="replace")
    except OSError as exc:
        return ERROR, f"cannot read the parent {parent.name}: {exc}", ""

    pool = md.parse_vector_pool_size(out_text)
    if pool is None:
        return (ERROR,
                f"MATDUMP N = {n} could not be bounded: {parent.name} has no "
                "'NO.OF VECTORS CREATED' line (CRYSTAL's vector pool)", "")
    try:
        derived = md.parse_max_gvector_index(out_text)
    except md.MatdumpRefusal as refusal:
        return ERROR, f"{parent.name}: {str(refusal).splitlines()[0]}", ""

    if derived == 1:
        return (REFUSED,
                f"the parent {parent.name} is a MOLECULE (one direct-lattice "
                "vector): H(R)/S(R) over lattice vectors is meaningless, and "
                "CRYSTAL does not error on it", "")
    if n > pool:
        return (REFUSED,
                f"MATDUMP N = {n} is above CRYSTAL's vector pool ({pool}, "
                f"{parent.name}); past the pool properties prints fabricated "
                "cell headers that corrupt the on-site block", "")

    notes = [f"N = {n}"]
    if derived is None:
        notes.append(f"{parent.name} gives no derived count, so only the pool "
                     f"({pool}) was checked")
    elif n < derived:
        notes.append(f"below the {derived} vectors the SCF used: the model "
                     "will be TRUNCATED")
    else:
        notes.append(f"derived {derived}, pool {pool}")
    return (SKIPPED,
            f"MATDUMP checked without CRYSTAL: records OK, {'; '.join(notes)}; "
            f"{f9.name} present (parent {parent.name})", "")
