#!/usr/bin/env python3
"""
d3_matdump.py - MATDUMP: direct-lattice H(R)/S(R) matrix dump for Wannier90

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group

MATDUMP generates the CRYSTAL ``properties`` deck that prints the Fock/KS and
overlap matrices in the direct-lattice representation, which is the input the
LCAO->Wannier90 bridge consumes.

ATTRIBUTION
-----------
The LCAO->Wannier90 method and the ``lcao2wannier`` package that consumes this
dump are **William Comaskey's work**. MACE only generates the CRYSTAL input and
orchestrates the run; it does not implement, and must never reimplement, any
part of the conversion (Fourier transform to H(k)/S(k), the generalized
eigenproblem in a non-orthogonal AO basis, SCDM projections, or the analytic
momentum-shifted GTO overlaps). See AUTHORSHIP.md.

THE DECK (CRYSTAL23 manual section 14, "BASISSET - Printing of basis set and
data from SCF", properties program; Appendix C p. 442 maps code 60 -> OVERLAP N
and code 64 -> FGRED N)::

    BASISSET
    2
    60 N
    64 N
    END

``N`` is the number of direct-lattice R-vectors to print. It is the one
genuinely dangerous parameter in this calculation: a too-small ``N`` does not
error, it yields a *wrong model*. Everything in this module exists to stop MACE
from guessing it.

MEASURED (stock CRYSTAL/23-intel-2023a, MSU HPCC dev-amd20, serial
``properties < INPUT`` with the corpus ``fort.9`` staged), using the real MACE
corpus material ``test/SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized`` whose parent
SCF reports ``MAX G-VECTOR INDEX ... 1247``:

* at the derived ``N = 1247``: 1247 overlap cells, 2494 Fock cells (collinear
  spin -> 2 per cell), ``ENDPROP``, no errors, 34,275,372 B, 0.97 s.
* at ``N = 60`` (the value used in the original hand-written reference deck)
  ``lcao2wannier`` **aborts**: ``STATUS: BAD``, 80/100 k-points not positive
  definite, ``cond(S) = inf``. Twenty times too small, and only the downstream
  package noticed.

That is the whole case for deriving ``N`` rather than defaulting it.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Attribution strings, surfaced in help text and driver messages ---------

MATDUMP_METHOD_CREDIT = (
    "LCAO->Wannier90 method and the lcao2wannier package: William Comaskey"
)

MATDUMP_CREDIT_BLOCK = (
    "  The matrix dump feeds the LCAO->Wannier90 bridge. That method and its\n"
    "  implementation (the lcao2wannier package) are William Comaskey's work.\n"
    "  MACE only generates the CRYSTAL deck and orchestrates the run.\n"
    "  CITATION: TODO - ask William Comaskey which citation he wants (the\n"
    "  package, a paper, or both) before a citation string is published here.\n"
    "  Do not invent one."
)

# --- Refusals ---------------------------------------------------------------


class MatdumpRefusal(Exception):
    """MACE refuses to generate or convert a matrix dump, with a stated reason.

    Every refusal in this module is deliberate: the alternative is a deck that
    runs to completion and produces a physically wrong model.
    """


# --- Parsing the parent SCF output -----------------------------------------

# The count of direct-lattice vectors CRYSTAL itself used at this run's
# TOLINTEG. The value is written in a fixed I4 field with NO guaranteed
# separating space: real corpus lines include both " INTEGRALS 103" and
# " INTEGRALS1111". A whitespace split silently fails on the second form, so
# the separator must be \s* and never \s+.
_MAXG_RE = re.compile(
    r"^\s*MAX G-VECTOR INDEX FOR 1- AND 2-ELECTRON INTEGRALS\s*(\S+)\s*$",
    re.MULTILINE,
)

# The fixed pool of direct-lattice vectors CRYSTAL creates. Constant at 6999 in
# every output in the MACE corpus, but parsed rather than hard-coded.
_VECTOR_POOL_RE = re.compile(r"NO\.OF VECTORS CREATED\s+(\d+)\s+STARS")

_NUMBER_OF_AO_RE = re.compile(r"NUMBER OF AO\s+(\d+)")

# CRYSTAL prints this in both SCF and properties outputs.
_UNRESTRICTED_RE = re.compile(r"TYPE OF CALCULATION\s*:\s*UNRESTRICTED")
_RESTRICTED_RE = re.compile(r"TYPE OF CALCULATION\s*:\s*RESTRICTED")


def parse_max_gvector_index(text: str) -> Optional[int]:
    """Number of direct-lattice R-vectors CRYSTAL used, from an .out file.

    Returns None when the line is absent. Raises MatdumpRefusal when the I4
    field has overflowed to ``****`` (>= 10000), because the true value is then
    unrecoverable and guessing it is exactly the failure this module prevents.

    The value is constant within a single output file (measured across the 707
    outputs in ``test/``: no file carries two different values), so any
    occurrence would do; the last is taken for definiteness.
    """
    matches = _MAXG_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1]
    if not raw.isdigit():
        raise MatdumpRefusal(
            f"CRYSTAL wrote {raw!r} for the direct-lattice vector count "
            f"(MAX G-VECTOR INDEX FOR 1- AND 2-ELECTRON INTEGRALS).\n"
            "That is an I4 field overflow, which happens at 10000 and above, so\n"
            "the true count cannot be recovered from this output.\n"
            "Supply N explicitly with --n-rvectors if you know it."
        )
    return int(raw)


def parse_vector_pool_size(text: str) -> Optional[int]:
    """Size of CRYSTAL's direct-lattice vector pool ('NO.OF VECTORS CREATED').

    This is a hard ceiling on any usable N. MEASURED: asking ``properties`` for
    more cells than the pool holds does NOT error - it prints the extra headers
    with indices read out of uninitialised memory (bcc Fe at N=7005 gave
    ``N.7000(  0  0  0)``, ``N.7003(***  0  0)``, ``N.7005(  0  0***)``), and
    several of those fabricated headers parse as R=(0,0,0). Downstream that
    overwrites the genuine on-site S(0) block in an R-keyed dict and S(k) goes
    singular at every k. Over-large N is not merely wasteful; it is corrupting.
    """
    matches = _VECTOR_POOL_RE.findall(text)
    if not matches:
        return None
    return int(matches[-1])


def parse_number_of_ao(text: str) -> Optional[int]:
    """Number of atomic orbitals, used only to predict the dump's size."""
    match = _NUMBER_OF_AO_RE.search(text)
    return int(match.group(1)) if match else None


def parse_dimensionality(text: str) -> int:
    """Periodicity of the parent calculation, from its output banner."""
    if "SLAB CALCULATION" in text or "SLAB GROUP" in text:
        return 2
    if "POLYMER CALCULATION" in text:
        return 1
    if "MOLECULAR CALCULATION" in text:
        return 0
    return 3


def parse_deck_dimensionality(deck_text: str) -> Optional[int]:
    """Periodicity from a CRYSTAL .d12 deck's own first structural keyword.

    Preferred over the output banner for the 0-D guard: the deck states the
    dimensionality unambiguously, while the output has to be inferred from
    printed prose.
    """
    keywords = {
        "MOLECULE": 0,
        "POLYMER": 1,
        "SLAB": 2,
        "CRYSTAL": 3,
        "EXTERNAL": None,
        "DLVINPUT": None,
    }
    for line in deck_text.splitlines():
        token = line.strip().upper()
        if token in keywords:
            return keywords[token]
    return None


# Spin treatment of the parent run. This decides which properties binary is
# required, so it is kept separate from everything else.
SPIN_CLOSED = "closed"
SPIN_COLLINEAR = "collinear"
SPIN_SOC = "soc"


def detect_spin_treatment(out_text: str, deck_text: str = "") -> str:
    """Classify the parent run as closed-shell, collinear spin, or 2c SOC.

    MEASURED: CRYSTAL prints ``TYPE OF CALCULATION : RESTRICTED CLOSED SHELL``
    or ``... UNRESTRICTED OPEN SHELL`` in 706 of the 707 outputs in ``test/``.

    SOC has no example in the MACE corpus (zero occurrences of ``SPIN-ORBIT``
    or ``SPINOR`` in 707 outputs), so it is detected from the deck's own ``SOC``
    keyword. That path is therefore untested against a corpus artifact and says
    so rather than pretending otherwise.
    """
    for line in deck_text.splitlines():
        if line.strip().upper() == "SOC":
            return SPIN_SOC
    if "SPIN-ORBIT" in out_text or "SPINOR" in out_text:
        return SPIN_SOC
    if _UNRESTRICTED_RE.search(out_text):
        return SPIN_COLLINEAR
    if _RESTRICTED_RE.search(out_text):
        return SPIN_CLOSED
    # No banner at all: assume the cheapest requirement rather than refusing a
    # run on a parse failure. The capability gate only ever *adds* a refusal,
    # so guessing "closed" here can never wrongly block a legitimate dump.
    return SPIN_CLOSED


def fock_blocks_per_cell(spin: str) -> int:
    """Fock/KS blocks printed per R-vector for each spin treatment.

    MEASURED on the corpus diamond (collinear): N=1247 gave 1247 overlap and
    2494 Fock blocks, i.e. 2 per cell. SOC's 8 (the 2x2 spinor structure, real
    and imaginary parts) is read from lcao2wannier's parser, not measured here.
    """
    return {SPIN_CLOSED: 1, SPIN_COLLINEAR: 2, SPIN_SOC: 8}.get(spin, 2)


# --- Deriving N -------------------------------------------------------------


def derive_n_rvectors(out_text: str, source: str = "the parent SCF output") -> int:
    """Derive N from the parent SCF output, or refuse with a stated reason.

    N is taken from ``MAX G-VECTOR INDEX FOR 1- AND 2-ELECTRON INTEGRALS``, the
    count of direct-lattice vectors CRYSTAL itself used at this run's TOLINTEG.

    What this value is, stated exactly: it is an upper bound on the support of
    S(R) and F(R) in ``fort.9``. ``fort.9`` can only hold what survived the
    SCF's integral screening at that index bound, so ``properties`` cannot print
    a nonzero block beyond it. N = MAXG is therefore the maximal-information
    choice, and any larger N (up to the vector pool) adds only zero blocks.

    It is NOT a claim that the physical H(R)/S(R) support ends there. The
    Wannier model inherits CRYSTAL's own integral truncation, and how converged
    that is with respect to TOLINTEG has not been measured here.

    Refuses, rather than defaulting, in every case where the value is not
    trustworthy - the measured range for one element on one lattice spans
    87..2731 depending on basis and TOLINTEG, so there is no safe constant.
    """
    maxg = parse_max_gvector_index(out_text)

    if maxg is None:
        raise MatdumpRefusal(
            f"Could not find the direct-lattice vector count in {source}.\n"
            "MATDUMP needs the line:\n"
            "  MAX G-VECTOR INDEX FOR 1- AND 2-ELECTRON INTEGRALS<count>\n"
            "which CRYSTAL writes in essentially every output. Its absence means\n"
            "this is not a CRYSTAL output, or it is truncated.\n"
            "Supply N explicitly with --n-rvectors if you know the right value.\n"
            "MACE will not guess it: a too-small N produces a wrong model, not\n"
            "an error."
        )

    if maxg == 1:
        raise MatdumpRefusal(
            f"{source} reports exactly one direct-lattice vector, which means the\n"
            "parent calculation is a MOLECULE (0-D). H(R)/S(R) over direct-lattice\n"
            "vectors is meaningless with no periodicity.\n"
            "MEASURED: CRYSTAL does NOT error on this - it exits 0 with ENDPROP and\n"
            "emits N blocks of which only the first is real; the rest are all-zero\n"
            "matrices carrying uninitialised lattice indices. MACE is the only guard."
        )

    if maxg % 2 == 0:
        raise MatdumpRefusal(
            f"Parsed a direct-lattice vector count of {maxg} from {source}, but the\n"
            "count must be odd: the R-vector set is closed under negation (cell 1 is\n"
            "(0,0,0), then +/- pairs). An even value means the line was misparsed,\n"
            "not that CRYSTAL changed. Refusing rather than dumping at a wrong N."
        )

    pool = parse_vector_pool_size(out_text)
    if pool is not None and maxg > pool:
        raise MatdumpRefusal(
            f"The derived N ({maxg}) exceeds CRYSTAL's direct-lattice vector pool\n"
            f"({pool}, from 'NO.OF VECTORS CREATED'). Beyond the pool, properties\n"
            "prints headers with indices read from uninitialised memory instead of\n"
            "erroring, and several of them parse as R=(0,0,0), which corrupts the\n"
            "on-site block downstream. Refusing."
        )

    return maxg


def validate_n_rvectors(n: int, out_text: str, explicit: bool = False) -> int:
    """Check a user-supplied N against the invariants N must satisfy."""
    if n < 1:
        raise MatdumpRefusal(
            f"N must be a positive number of direct-lattice vectors; got {n}."
        )

    pool = parse_vector_pool_size(out_text)
    if pool is None:
        raise MatdumpRefusal(
            "Could not read CRYSTAL's direct-lattice vector pool size\n"
            "('NO.OF VECTORS CREATED') from the parent output, so N cannot be\n"
            "bounded. Refusing rather than dumping past the pool, where properties\n"
            "emits uninitialised lattice indices instead of an error."
        )
    if n > pool:
        raise MatdumpRefusal(
            f"N={n} exceeds CRYSTAL's direct-lattice vector pool ({pool}, from\n"
            "'NO.OF VECTORS CREATED').\n"
            "MEASURED: properties does not error past the pool. It prints the extra\n"
            "cells with garbage indices - bcc Fe at N=7005 gave N.7000(  0  0  0),\n"
            "N.7003(***  0  0), N.7005(  0  0***) - and those fabricated (0,0,0)\n"
            "headers overwrite the genuine on-site overlap block downstream, making\n"
            "S(k) singular at every k. Over-large N is corrupting, not just wasteful."
        )

    if explicit:
        derived = parse_max_gvector_index(out_text)
        if derived is not None and n < derived:
            # Allowed - the user may know the true support is tighter than the
            # bound - but never silent.
            pass
    return n


# --- Size prediction --------------------------------------------------------


def predict_dump_bytes(n: int, n_ao: Optional[int], spin: str) -> Optional[int]:
    """Predict the properties output size, in bytes.

    Cost is linear in N and quadratic in the number of AOs; zero blocks print at
    the same width as data, so the law does not soften for a sparse dump.

    CALIBRATED on the corpus diamond (n_ao=36, collinear, so 3 blocks per cell)
    at two values of N on real hardware: N=60 -> 1,660,173 B measured vs
    1,678,320 B predicted; N=1247 -> 34,275,372 B measured vs 34,884,468 B
    predicted. Both within 2%. One system, one basis: treat it as an estimate
    for sizing a warning, not as a guarantee.
    """
    if not n_ao:
        return None
    blocks = 1 + fock_blocks_per_cell(spin)
    elements = n_ao * (n_ao + 1) // 2
    return n * blocks * elements * 14


def format_bytes(num: Optional[int]) -> str:
    if num is None:
        return "unknown"
    value = float(num)
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


# --- Capability detection ---------------------------------------------------

# Byte markers that distinguish a properties binary able to print 2-component
# SOC matrices from one that cannot. MEASURED by inspecting both binaries with
# strings/readelf (never executing them): the markers live in one packed
# character array in .data on an unstripped build.
_SOC_MARKERS = (b"FOCK MATRIX (REAL PART)", b"ALPHA_ALPHA ELECTRONS")


def properties_binary_supports_soc(binary_path: Path) -> Optional[bool]:
    """Whether a properties binary can emit 2-component SOC matrices.

    Returns None when the binary cannot be read at all, so "unknown" is never
    conflated with "incapable".

    This is a heuristic on a third-party binary: it scans a data blob in an
    unstripped build. No symbol-level discriminator exists (the ``*wannier*``
    symbols are present in both builds and belong to CRYSTAL's own LOCALWF cube
    plotting, not to this feature).
    """
    try:
        blob = Path(binary_path).read_bytes()
    except OSError:
        return None
    return any(marker in blob for marker in _SOC_MARKERS)


def capability_refusal(spin: str, binary_path: Optional[Path]) -> Optional[str]:
    """Refusal message when the available properties binary is not capable.

    MEASURED, and this corrects an earlier assumption that cost real time:
    **stock CRYSTAL23 already emits collinear spin-resolved matrices.** The
    corpus material ``1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized`` is a SPIN /
    UNRESTRICTED OPEN SHELL deck, and on the stock HPCC module it dumped 1247
    overlap and 2494 Fock blocks under ALPHA/BETA headers, which lcao2wannier
    parses natively. Only 2-component SOC needs the development binary.

    So this gate fires for SOC parents only. Refusing a collinear spin dump
    would refuse a calculation that demonstrably works.
    """
    if spin != SPIN_SOC:
        return None

    if binary_path is None:
        return None

    capable = properties_binary_supports_soc(binary_path)
    if capable is None or capable:
        return None

    return (
        f"MATDUMP refused: this is a 2-component spin-orbit (SOC) calculation, and\n"
        f"the available CRYSTAL properties binary cannot print SOC matrices.\n"
        f"\n"
        f"  binary checked : {binary_path}\n"
        f"  missing        : 'FOCK MATRIX (REAL PART)' / 'ALPHA_ALPHA ELECTRONS'\n"
        f"                   (complex matrices and the 2x2 spinor block labels)\n"
        f"\n"
        f"This is needed for SOC only. Scalar closed-shell systems, AND collinear\n"
        f"spin-polarized systems, work on a stock CRYSTAL23 build - measured: a\n"
        f"SPIN/UNRESTRICTED deck dumps ALPHA and BETA blocks on the stock module.\n"
        f"\n"
        f"The SOC-capable properties/Pproperties are development builds. Request\n"
        f"them from the CRYSTAL23 developers directly. MACE never bundles,\n"
        f"downloads or redistributes them."
    )


# --- The deck ---------------------------------------------------------------


def write_matdump_deck(n: int) -> str:
    """The CRYSTAL properties matrix-dump deck, verified against CRYSTAL23.

    Exactly the record from the manual: NPR=2 printing options, then the prtrec
    pairs 60 (overlap) and 64 (Fock/KS), each with the R-vector count.

    Do not "improve" this. A SETPRINT-based variant was tried and fails with
    ``ERROR **** BASE **** FORMAT ERROR IN INPUT DECK``, and dropping one prtrec
    pair while leaving NPR at 2 makes properties consume the END and fail the
    same way.
    """
    if n < 1:
        raise MatdumpRefusal(f"N must be at least 1; got {n}.")
    return "\n".join(["BASISSET", "2", f"60 {n}", f"64 {n}", "END"])


# --- Post-dump verification -------------------------------------------------


def count_dumped_cells(out_path: Path) -> Tuple[int, int]:
    """Count the overlap and Fock cell headers actually emitted by a dump.

    Read line by line: a production dump is tens to hundreds of MB and must not
    be slurped. Opened in binary because CRYSTAL outputs can contain NUL bytes
    (50 files in the MACE corpus do, which is how an earlier survey lost them).
    """
    overlap = fock = 0
    with open(out_path, "rb") as handle:
        for raw in handle:
            if b"OVERLAP MATRIX - CELL" in raw:
                overlap += 1
            elif b"FOCK MATRIX" in raw and b"- CELL" in raw:
                fock += 1
    return overlap, fock


def verify_dump(out_path: Path, requested_n: int) -> Dict[str, Any]:
    """Compare a finished dump against what the deck asked for.

    The authoritative check is a count: the number of ``OVERLAP MATRIX - CELL``
    headers emitted must equal the N the deck requested. That is exact.

    There is deliberately NO "the outermost cells are all zero" heuristic here.
    It was measured to false-pass: on the diamond dump there is a run of 27
    consecutive all-zero cells ending at cell 297 with genuine data at 298, 302,
    314 and 318, so a tail window of ~10 declares a dump truncated at 297
    complete and discards 21 cells of real data. The support is not a contiguous
    prefix, so no fixed window is safe.
    """
    overlap, fock = count_dumped_cells(out_path)
    return {
        "requested_n": requested_n,
        "overlap_cells": overlap,
        "fock_blocks": fock,
        "complete": overlap == requested_n,
        "fock_blocks_per_cell": (fock // overlap) if overlap else 0,
    }
