"""Whether a MACE/CRYSTAL23 parent can seed phonopy's displaced supercells.

Standard library only: nothing here imports phonopy.

opt2d12 writes the displaced-supercell decks (calculation type PHONOPY,
choice 8 / --calc-type PHONOPY: Crystal_d12/d12_calc_phonopy.py) with its
own deck writer, from the parent's settings and the answers to its
questions::

    <title>
    EXTERNAL                      geometry from fort.34 (manual p.21, format pp.450-451)
    BASISSET / <name>             internal library: replaces the END of block 1 (p.28)
      - or -
    END / <basis records> / 99 0 / ... / END     explicit basis and block 2
    <block 3 as opt2d12 writes it>
    GRADCAL                       analytical gradient after the SCF (p.111)
    END

This module reads the parent deck and output: the refusals (a parent that
is not a finished 3D OPT or SP run, or carries atom-indexed records that do
not carry over to a P1 supercell), the SHRINK of a supercell, and the
method records NAC compares.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Refusal codes (stable names; tests refer to them)
# ---------------------------------------------------------------------------
REFUSE_NOT_3D = "NOT_3D"
REFUSE_ABNORMAL_TERMINATION = "ABNORMAL_TERMINATION"
REFUSE_SCF_NOT_CONVERGED = "SCF_NOT_CONVERGED"
REFUSE_OPT_NOT_CONVERGED = "OPT_NOT_CONVERGED"
REFUSE_FREQ_GEOMETRY = "FREQ_GEOMETRY"
REFUSE_CHARGE_RECORDS = "CHARGE_RECORDS"
REFUSE_SPIN_LOCKS = "SPIN_LOCKS"
REFUSE_GHOSTS = "GHOSTS"
REFUSE_UNLOCKED_MAGNETISM = "UNLOCKED_MAGNETISM"
REFUSE_UNKNOWN_KEYWORD = "UNKNOWN_KEYWORD"
REFUSE_DECK_STRUCTURE = "DECK_STRUCTURE"

WARN_SP_PARENT = "SP_PARENT"

#: |Mulliken spin| above which an atom counts as magnetic (electrons).
MAGNETIC_SPIN_THRESHOLD = 0.1

_DIMENSION_KEYWORDS = {"CRYSTAL": 3, "SLAB": 2, "POLYMER": 1, "HELIX": 1, "MOLECULE": 0}

# Block-3 sub-blocks whose contents are the method itself: copied verbatim and
# not scanned keyword by keyword. Each one closes with a record starting "END".
_SUB_BLOCKS = {"DFT", "DFTD3", "GCP"}

#: Top-level block-3 keywords a displaced supercell may carry: what MACE's
#: d12_writer emits plus the common SCF/DFT controls that are not tied to atom
#: labels. Anything else (EIGSHIFT, SPINEDIT, FDOCCUP, ATOMONO, ...) may number
#: atoms of the primitive cell and is refused by name.
BLOCK3_WHITELIST = frozenset({
    # SCF / integrals / k-points
    "SHRINK", "TOLINTEG", "TOLDEE", "TOLPSEUD", "MAXCYCLE", "FMIXING", "DIIS",
    "NODIIS", "HISTDIIS", "DIISALLK", "THREDIIS", "THRDIIS", "TWSCDIIS",
    "BROYDEN", "ANDERSON", "LEVSHIFT", "SMEAR", "SCFDIR", "BIPOSIZE",
    "EXCHSIZE", "PPAN", "GUESSP", "NOSYMADA", "SYMADAPT", "CMPLXFAC",
    "ILASIZE", "MADELIND", "POLEORDR", "NOBIPOLA", "NOBIPCOU", "NOBIPEXC",
    "BIPOLAR", "NEIGHBOR", "NOMULLIK", "MULPOPAN", "TOLSCF", "GRADCAL",
    "UHF", "RHF", "ROHF", "MPP", "FULLTIME", "SETINF", "SETPRINT", "MEMOPRT",
    "STOREIJ", "NOFMWF", "SAVEWF", "SAVEPRED", "NOSHIFT", "EXCHPERM",
    "BIESPLIT", "MONSPLIT", "MAXNEIGHB", "NOINVERS",
    # method selection (sub-blocks and composite methods)
    "DFT", "DFTD3", "GCP", "GCPAUTO", "HF3C", "HFSOL3C", "PBEH3C", "HSE3C",
    "B973C", "PBESOL03C", "HSESOL3C",
})

_KEYWORD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_+-]*$")
_TERMINATION_RE = re.compile(r"^\s*EEEEEEEEEE TERMINATION", re.M)
_MULLIKEN_ROW_RE = re.compile(r"^\s*(\d+)\s+([A-Za-z]{1,2})\s+(\d+)\s+(-?\d+\.\d+)")
# the header of a Mulliken alpha-beta block (not SPINLOCK's "... LOCKED TO" echo)
_MULLIKEN_BLOCK_HEAD_RE = re.compile(r"ALPHA-BETA ELECTRONS[ \t]*\r?\n[ \t]*MULLIKEN POPULATION ANALYSIS")


class DeckSplitError(ValueError):
    """The parent deck could not be split into block 1 / basis / block 3."""


@dataclass
class ParentDeck:
    """A parent deck cut at its basis set."""

    lines: List[str]
    title: str
    dimension_keyword: Optional[str]
    block1: List[str]
    basis_kind: str                      # "internal" or "explicit"
    basis_lines: List[str]               # BASISSET + name, or records through block-2 END
    block3: List[str]                    # without the final END
    dropped_keywords: List[str] = field(default_factory=list)

    @property
    def basis_label(self) -> Optional[str]:
        if self.basis_kind == "internal":
            return self.basis_lines[1].strip()
        return None

    def keyword_lines(self) -> List[str]:
        """Upper-cased single-token keyword records outside the title."""
        return [_kw(ln) for ln in self.lines[1:] if _kw(ln)]


def _kw(line: str) -> Optional[str]:
    body = line.strip()
    if _KEYWORD_RE.match(body):
        return body.upper()
    return None


def _is_end(line: str) -> bool:
    k = _kw(line)
    return bool(k) and k.startswith("END")


def _is_99_0(line: str) -> bool:
    """The whole-line record that closes an explicit basis. Token match only:
    "99 0" inside an ECP or exponent record is not it."""
    return line.split() == ["99", "0"]


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_parent_deck(text: str) -> ParentDeck:
    """Cut a parent deck at its basis set.

    * internal library: the top-level ``BASISSET`` record and its name line;
    * explicit basis: the first whole-line ``99 0``. The basis starts after the
      last END/ENDGEOM record before it (basis records are numeric, so no END
      sits inside them); the records after ``99 0`` through the block-2 END
      belong to the basis part;
    * block 3: the rest. Its last non-blank record must be END; it is removed.

    OPTGEOM...ENDOPT, FREQCALC...ENDFREQ and the like are all in block 1 and go
    with it; DFT...ENDDFT, DFTD3...END and GCP...END are in block 3 and are kept.
    """
    lines = text.splitlines()
    if len(lines) < 3:
        raise DeckSplitError("deck is too short")
    title = lines[0]
    dimension = _kw(lines[1]) if len(lines) > 1 else None

    bs_idx = next((i for i in range(1, len(lines)) if _kw(lines[i]) == "BASISSET"), None)
    e99_idx = next((i for i in range(1, len(lines)) if _is_99_0(lines[i])), None)

    if bs_idx is not None and (e99_idx is None or bs_idx < e99_idx):
        if bs_idx + 1 >= len(lines) or not lines[bs_idx + 1].strip():
            raise DeckSplitError("BASISSET is not followed by a basis-set name")
        block1 = lines[1:bs_idx]
        basis_kind = "internal"
        basis_lines = lines[bs_idx:bs_idx + 2]
        rest_start = bs_idx + 2
    elif e99_idx is not None:
        end1 = None
        for i in range(e99_idx - 1, 0, -1):
            if _is_end(lines[i]):
                end1 = i
                break
        if end1 is None:
            raise DeckSplitError("no END closing the geometry block before the basis set")
        end2 = next((i for i in range(e99_idx + 1, len(lines)) if _is_end(lines[i])), None)
        if end2 is None:
            raise DeckSplitError("no END closing the basis-set block after '99 0'")
        block1 = lines[1:end1]
        basis_kind = "explicit"
        basis_lines = lines[end1 + 1:end2 + 1]
        rest_start = end2 + 1
    else:
        raise DeckSplitError("no BASISSET record and no '99 0' record: cannot find the basis set")

    rest = lines[rest_start:]
    last = len(rest) - 1
    while last >= 0 and not rest[last].strip():
        last -= 1
    if last < 0 or _kw(rest[last]) != "END":
        raise DeckSplitError("the deck does not end with an END record")
    block3 = rest[:last]

    return ParentDeck(
        lines=lines, title=title, dimension_keyword=dimension, block1=block1,
        basis_kind=basis_kind, basis_lines=basis_lines, block3=block3,
        dropped_keywords=_block1_keywords(block1),
    )


def _block1_keywords(block1: Sequence[str]) -> List[str]:
    """Block-1 keywords the template drops, OPTGEOM/FREQCALC contents folded
    into the name of their block."""
    out: List[str] = []
    inside: Optional[str] = None
    for ln in block1[1:]:            # block1[0] is the dimensionality keyword
        k = _kw(ln)
        if not k:
            continue
        if inside:
            if k.startswith("END"):
                inside = None
            continue
        if k in ("OPTGEOM", "FREQCALC", "ELASTCON", "EOS", "PREOPTGEOM"):
            inside = k
            out.append(k)
            continue
        if k.startswith("END"):
            continue
        out.append(k)
    return out


# ---------------------------------------------------------------------------
# Block-3 scanning
# ---------------------------------------------------------------------------

def _top_level(block3: Sequence[str]) -> List[Tuple[int, str]]:
    """(index, KEYWORD) of every top-level keyword record of block 3."""
    out = []
    inside = None
    for i, ln in enumerate(block3):
        k = _kw(ln)
        if not k:
            continue
        if inside:
            if k.startswith("END"):
                inside = None
            continue
        out.append((i, k))
        if k in _SUB_BLOCKS:
            inside = k
    return out


# Top-level keywords whose NEXT record is a free-text argument rather than a
# keyword (so it must not be mistaken for one).
_TEXT_ARGUMENT = {"SMEAR": 1}


def unknown_block3_keywords(block3: Sequence[str]) -> List[str]:
    found = []
    skip_until = -1
    for i, k in _top_level(block3):
        if i <= skip_until:
            continue
        if k not in BLOCK3_WHITELIST and not k.startswith("END"):
            found.append(k)
        skip_until = i + _TEXT_ARGUMENT.get(k, 0)
    return found


# ---------------------------------------------------------------------------
# Output checks
# ---------------------------------------------------------------------------

def normal_termination(out_text: str) -> bool:
    return bool(_TERMINATION_RE.search(out_text)) and "ERROR ****" not in out_text


def scf_converged(out_text: str) -> bool:
    ends = [ln for ln in out_text.splitlines() if "SCF ENDED" in ln]
    return bool(ends) and "CONVERGENCE" in ends[-1]


def output_dimensionality(out_text: str) -> Optional[int]:
    for marker, dim in ((" CRYSTAL CALCULATION", 3), (" SLAB CALCULATION", 2),
                        (" POLYMER CALCULATION", 1), (" MOLECULAR CALCULATION", 0),
                        (" HELIX CALCULATION", 1)):
        # the literal first: a scan of the whole text per marker otherwise
        if marker in out_text and re.search(r"^" + marker + r"\s*$", out_text, re.M):
            return dim
    m = re.search(r"^\s*([0-3])D - ", out_text, re.M) if "D - " in out_text else None
    if m:
        return int(m.group(1))
    return None


def final_mulliken_spins(out_text: str) -> Optional[List[Tuple[int, str, float]]]:
    """(atom, symbol, spin) from the last ALPHA-BETA Mulliken block, or None
    for a closed-shell (or NOMULLIK) run.

    The block's own header is "ALPHA-BETA ELECTRONS" followed by "MULLIKEN
    POPULATION ANALYSIS" on the next line. SPINLOCK echoes "ALPHA-BETA
    ELECTRONS LOCKED TO n FOR m SCF CYCLES" near the top of the output; that
    line is not a Mulliken block, so a SPINLOCK run that printed no block
    (killed, or NOMULLIK) has no spins rather than rows read from its input
    echo."""
    idx = len(out_text)
    while True:
        idx = out_text.rfind("ALPHA-BETA ELECTRONS", 0, idx)
        if idx < 0:
            return None
        if _MULLIKEN_BLOCK_HEAD_RE.match(out_text, idx):
            break
    rows = []
    started = False
    for ln in out_text[idx:].splitlines()[1:]:
        if "SHELL POPULATION" in ln or "OVERLAP POPULATION" in ln:
            break
        m = _MULLIKEN_ROW_RE.match(ln)
        if m:
            started = True
            rows.append((int(m.group(1)), m.group(2), float(m.group(4))))
        elif started and ln.strip() and not ln.startswith(" " * 10):
            break
    return rows or None


def is_opt_parent(deck: ParentDeck, out_text: str) -> bool:
    return "OPTGEOM" in deck.dropped_keywords or "OPTOPTOPT" in out_text


def is_freq_parent(deck: ParentDeck, out_text: str) -> bool:
    return ("FREQCALC" in deck.dropped_keywords
            or "FREQUENCY CALCULATION" in out_text)


def run_refusals(out_text: str, is_opt: bool, is_freq: bool,
                 why: str = "would give spurious imaginary modes",
                 freq_hint: str = " (a FREQ output can still be used for NAC with collect --nac-from)"
                 ) -> List[Tuple[str, str]]:
    """Why this output is not the finished OPT or SP run a follow-up needs, as (code, message):
    phonopy's displaced supercells and the mechanical-property decks (mace opt2d12 choices 4-8)
    alike (mace.workflow.common.constants.FINISHED_RUN_TYPES)."""
    reasons: List[Tuple[str, str]] = []
    if not normal_termination(out_text):
        reasons.append((REFUSE_ABNORMAL_TERMINATION,
                        "parent output did not terminate normally (no EEEEEEEEEE TERMINATION, "
                        "or an ERROR record)"))
    if not scf_converged(out_text):
        reasons.append((REFUSE_SCF_NOT_CONVERGED, "parent's last SCF did not converge"))
    if is_opt and "OPT END - CONVERGED" not in out_text:
        reasons.append((REFUSE_OPT_NOT_CONVERGED,
                        "parent is an optimization without 'OPT END - CONVERGED': its last "
                        f"geometry is not a minimum and {why}"))
    if is_freq:
        reasons.append((REFUSE_FREQ_GEOMETRY,
                        "parent is a FREQCALC run; use the OPT (or SP) output it was "
                        f"derived from as the geometry source{freq_hint}"))
    return reasons


def refusal_reasons(deck: ParentDeck, out_text: str) -> List[Tuple[str, str]]:
    """Every reason this parent cannot seed a phonopy run, as (code, message)."""
    reasons: List[Tuple[str, str]] = []
    keys = set(deck.keyword_lines())

    dim = _DIMENSION_KEYWORDS.get(deck.dimension_keyword or "")
    out_dim = output_dimensionality(out_text)
    if (dim is not None and dim != 3) or (out_dim is not None and out_dim != 3):
        reasons.append((REFUSE_NOT_3D,
                        "parent is not a 3D crystal; phonopy's CRYSTAL interface writes 3D "
                        "EXTERNAL cells only. For slabs/polymers use CRYSTAL's own "
                        "FREQCALC + SCELPHONO + DISPERSI (manual pp.73, 232)"))
    reasons += run_refusals(out_text, is_opt_parent(deck, out_text), is_freq_parent(deck, out_text))

    try:
        import sys as _sys
        _cd12 = str(Path(__file__).resolve().parents[2] / "Crystal_d12")
        if _cd12 not in _sys.path:
            _sys.path.insert(0, _cd12)
        from charge_records import extract_charge_records
        charge = extract_charge_records("\n".join(deck.lines))
        charged = [k for k, v in charge.items() if v]
    except ImportError:  # pragma: no cover - repo layout always has it
        charged = [k for k in ("CHEMOD", "CHARGED", "DOPING") if k in keys]
    if charged:
        reasons.append((REFUSE_CHARGE_RECORDS,
                        f"parent carries {', '.join(charged)}: atom labels and electron "
                        "counts do not scale to a supercell (not supported yet)"))
    locks = sorted(keys & {"ATOMSPIN", "SPINLOCK", "BETALOCK"})
    if locks:
        reasons.append((REFUSE_SPIN_LOCKS,
                        f"parent uses {', '.join(locks)}: atom-indexed spin settings do not "
                        "carry over to a P1 supercell (not supported yet)"))
    if "GHOSTS" in keys:
        reasons.append((REFUSE_GHOSTS, "parent uses GHOSTS (atom labels of the primitive cell)"))

    if not locks:
        spins = final_mulliken_spins(out_text)
        if spins:
            big = [(a, s, v) for a, s, v in spins if abs(v) > MAGNETIC_SPIN_THRESHOLD]
            if big:
                a, s, v = max(big, key=lambda t: abs(t[2]))
                reasons.append((REFUSE_UNLOCKED_MAGNETISM,
                                f"parent is magnetic (Mulliken spin {v:+.3f} on atom {a} {s}) "
                                "with no ATOMSPIN/SPINLOCK: the magnetic state is not "
                                "reproducible in a P1 supercell"))

    stray = [i for i, k in _top_level(deck.block3) if k == "END"]
    if stray:
        reasons.append((REFUSE_DECK_STRUCTURE,
                        f"block 3 has an END before its last record ({len(stray)} extra): "
                        "CRYSTAL stops reading there, and in a displaced deck it would stop "
                        "before GRADCAL and write no forces. Remove the extra END"))
    # Keywords refused by name above are not refused a second time as unknown
    # (a SPINLOCK parent was reported as SPIN_LOCKS and as UNKNOWN_KEYWORD).
    unknown = [k for k in unknown_block3_keywords(deck.block3)
               if k not in ("ATOMSPIN", "SPINLOCK", "BETALOCK", "DOPING")]
    if unknown:
        reasons.append((REFUSE_UNKNOWN_KEYWORD,
                        f"block 3 keyword(s) not known to be safe in a supercell: "
                        f"{', '.join(dict.fromkeys(unknown))}"))
    return reasons


def warnings_for(deck: ParentDeck, out_text: str) -> List[Tuple[str, str]]:
    warns = []
    if not is_opt_parent(deck, out_text) and not is_freq_parent(deck, out_text):
        warns.append((WARN_SP_PARENT,
                      "parent is a single point: its geometry is assumed to be relaxed "
                      "(forces at a non-stationary geometry give wrong low modes)"))
    return warns


# ---------------------------------------------------------------------------
# Supercell k-point mesh
# ---------------------------------------------------------------------------

def is_diagonal(dim: Sequence[Sequence[int]]) -> bool:
    return all(dim[i][j] == 0 for i in range(3) for j in range(3) if i != j)


def scaled_mesh(old: Sequence[int], isp: int, diag: Sequence[int]) -> Tuple[List[int], int, str]:
    """SHRINK of a diagonal supercell: the parent's mesh divided by the supercell
    size in each direction, rounded up where it does not divide (so the k-point
    density never drops below the parent's), and its Gilat net scaled with it.
    Returns (mesh, ISP, the note opt2d12 prints and the manifest keeps)."""
    old = [int(k) for k in old]
    new = [max(1, math.ceil(k / d)) for k, d in zip(old, diag)]
    new_isp = max(max(new), math.ceil(int(isp) * max(new) / max(old)))
    note = f"SHRINK {' '.join(map(str, old))} (ISP {isp}) -> {' '.join(map(str, new))} (ISP {new_isp})"
    inexact = [f"{k}/{d}" for k, d in zip(old, diag) if k % d]
    if inexact:
        note += ("; supercell size does not divide the parent mesh "
                 f"({', '.join(inexact)}), so the sampling is equivalent but not identical")
    return new, new_isp, note


def method_lines(deck: ParentDeck) -> Dict[str, List[str]]:
    """The records that define the method and basis: the basis part and the
    DFT/DFTD3/GCP sub-blocks (and composite-method keywords) of block 3."""
    out: Dict[str, List[str]] = {"basis": list(deck.basis_lines)}
    inside = None
    for ln in deck.block3:
        k = _kw(ln)
        if inside:
            out[inside].append(ln)
            if k and k.startswith("END"):
                inside = None
            continue
        if k in _SUB_BLOCKS:
            inside = k
            out.setdefault(k, []).append(ln)
        elif k in {"HF3C", "HFSOL3C", "PBEH3C", "HSE3C", "B973C", "PBESOL03C", "HSESOL3C",
                   "UHF", "RHF", "ROHF"}:
            out.setdefault("composite", []).append(ln)
    return out


def read_parent(d12_path: Path, out_path: Path) -> Tuple[ParentDeck, str]:
    text = Path(d12_path).read_text(errors="replace")
    out_text = Path(out_path).read_text(errors="replace")
    return split_parent_deck(text), out_text
