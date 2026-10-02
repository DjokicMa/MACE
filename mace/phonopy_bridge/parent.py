"""Read a MACE/CRYSTAL23 parent into a phonopy supercell template.

Standard library only: nothing here imports phonopy.

The displaced-supercell deck is the one phonopy's CRYSTAL interface writes
around a TEMPLATE file::

    <title>
    EXTERNAL                      geometry from fort.34 (manual p.21, format pp.450-451)
    BASISSET / <name>             internal library: replaces the END of block 1 (p.28)
      - or -
    END / <basis records> / 99 0 / ... / END     explicit basis and block 2
    <block 3 of the parent, edited only as listed below>      } the TEMPLATE
    TOLDEE / N                    phonopy's writer appends TOLDEE / 10 here
    GRADCAL                       analytical gradient after the SCF (p.111)
    END

The TEMPLATE (basis part + block 3) has no final END, no TOLDEE and no GRADCAL.
Measured on CRYSTAL23 with phonopy 4.7.2: a TEMPLATE ending with END gives
``... END / TOLDEE 10 / GRADCAL / END``; CRYSTAL stops at the first END, the
SCF finishes with return code 0 and no forces are written.

Block 1 of the parent (its geometry, OPTGEOM, FREQCALC, ...) is dropped: the
geometry comes from the parent's output via phonopy. Everything from the basis
on is copied byte for byte, except for these edits, each reported:

    (a) the final END is removed (the deck's END follows GRADCAL);
    (b) TOLDEE is removed from block 3 and written once after it, as
        max(10, parent, requested) - phonopy's own writer always writes 10;
    (c) GUESSP is dropped (no density matrix exists for a displaced supercell);
    (d) SHRINK is divided by the supercell size in each direction (diagonal
        supercells; rounded up where the size does not divide the parent's
        value, so the k-point density never drops below the parent's);
    (e) BIPOSIZE/EXCHSIZE 110000000 are added when the parent has neither and
        the supercell has more than 5 atoms (d12_writer's own rule);
    (f) GRADCAL is removed from block 3 if the parent had it, and written once
        after TOLDEE, followed by END.

The method, the basis label and every other tolerance are never changed.
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

#: d12_writer's buffer rule (BIPOSIZE/EXCHSIZE for more than 5 atoms).
BUFFER_SIZE = "110000000"
BUFFER_ATOM_THRESHOLD = 5

#: Default TOLDEE for the displaced supercells (phonopy's own template uses 10).
DEFAULT_TOLDEE = 10

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
        if re.search(r"^" + marker + r"\s*$", out_text, re.M):
            return dim
    m = re.search(r"^\s*([0-3])D - ", out_text, re.M)
    if m:
        return int(m.group(1))
    return None


def final_mulliken_spins(out_text: str) -> Optional[List[Tuple[int, str, float]]]:
    """(atom, symbol, spin) from the last ALPHA-BETA Mulliken block, or None
    for a closed-shell (or NOMULLIK) run."""
    idx = out_text.rfind("ALPHA-BETA ELECTRONS")
    if idx < 0:
        return None
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
    if not normal_termination(out_text):
        reasons.append((REFUSE_ABNORMAL_TERMINATION,
                        "parent output did not terminate normally (no EEEEEEEEEE TERMINATION, "
                        "or an ERROR record)"))
    if not scf_converged(out_text):
        reasons.append((REFUSE_SCF_NOT_CONVERGED, "parent's last SCF did not converge"))
    if is_opt_parent(deck, out_text) and "OPT END - CONVERGED" not in out_text:
        reasons.append((REFUSE_OPT_NOT_CONVERGED,
                        "parent is an optimisation without 'OPT END - CONVERGED': its last "
                        "geometry is not a minimum and would give spurious imaginary modes"))
    if is_freq_parent(deck, out_text):
        reasons.append((REFUSE_FREQ_GEOMETRY,
                        "parent is a FREQCALC run; use the OPT (or SP) output it was "
                        "derived from as the geometry source (a FREQ output can still be "
                        "used for NAC with collect --nac-from)"))

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

    unknown = unknown_block3_keywords(deck.block3)
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
# Template
# ---------------------------------------------------------------------------

@dataclass
class Template:
    lines: List[str]                     # the TEMPLATE: basis part + edited block 3,
                                         # without TOLDEE, GRADCAL and the final END
    edits: List[str]
    toldee: int


def _value_index(block3: Sequence[str], key: str) -> Optional[int]:
    for i, k in _top_level(block3):
        if k == key:
            return i
    return None


def _parse_int(line: str) -> Optional[int]:
    try:
        return int(line.split()[0])
    except (ValueError, IndexError):
        return None


def parent_toldee(block3: Sequence[str]) -> Optional[int]:
    i = _value_index(block3, "TOLDEE")
    if i is None or i + 1 >= len(block3):
        return None
    return _parse_int(block3[i + 1])


def is_diagonal(dim: Sequence[Sequence[int]]) -> bool:
    return all(dim[i][j] == 0 for i in range(3) for j in range(3) if i != j)


def scale_shrink(block3: List[str], diag: Sequence[int]) -> Tuple[List[str], Optional[str]]:
    """Scale SHRINK by the supercell multiplicities (ceil per direction)."""
    i = _value_index(block3, "SHRINK")
    if i is None or i + 1 >= len(block3):
        return block3, None
    first = block3[i + 1].split()
    try:
        is_, isp = int(first[0]), int(first[1])
    except (ValueError, IndexError):
        return block3, None
    consumed = 2
    if is_ == 0:
        try:
            old = [int(x) for x in block3[i + 2].split()[:3]]
        except (ValueError, IndexError):
            return block3, None
        consumed = 3
    else:
        old = [is_] * 3
    new = [max(1, math.ceil(k / d)) for k, d in zip(old, diag)]
    new_isp = max(max(new), math.ceil(isp * max(new) / max(old)))
    if len(set(new)) == 1:
        repl = [f"{new[0]} {new_isp}"]
    else:
        repl = [f"0 {new_isp}", f"{new[0]} {new[1]} {new[2]}"]
    note = f"SHRINK {' '.join(map(str, old))} (ISP {isp}) -> {' '.join(map(str, new))} (ISP {new_isp})"
    inexact = [f"{k}/{d}" for k, d in zip(old, diag) if k % d]
    if inexact:
        note += ("; supercell size does not divide the parent mesh "
                 f"({', '.join(inexact)}), so the sampling is equivalent but not identical")
    out = list(block3[:i + 1]) + repl + list(block3[i + consumed:])
    return out, note


def template_lines(deck: ParentDeck, n_super_atoms: int, toldee: Optional[int] = None,
                   scale_shrink_dims: Optional[Sequence[int]] = None) -> Template:
    """The TEMPLATE: basis part + block 3 of the parent with edits (a)-(f)."""
    edits = ["(a) final END of the parent removed (the deck's END follows GRADCAL)"]
    block3 = list(deck.block3)

    # (b) TOLDEE: out of block 3, into the deck tail
    old = parent_toldee(block3)
    target = max(DEFAULT_TOLDEE if toldee is None else toldee, old or 0)
    ti = _value_index(block3, "TOLDEE")
    if ti is not None:
        del block3[ti:ti + 2 if old is not None else ti + 1]
    if old is None:
        edits.append(f"(b) TOLDEE {target} written before GRADCAL (parent had none)")
    elif target != old:
        edits.append(f"(b) TOLDEE {old} -> {target}, moved before GRADCAL")
    else:
        edits.append(f"(b) TOLDEE {old} kept, moved before GRADCAL")

    # (c) GUESSP
    gi = [i for i, k in _top_level(block3) if k == "GUESSP"]
    for i in reversed(gi):
        del block3[i]
    if gi:
        edits.append("(c) GUESSP dropped (no density matrix for a displaced supercell)")

    # (d) SHRINK
    if scale_shrink_dims is not None:
        block3, note = scale_shrink(block3, scale_shrink_dims)
        if note:
            edits.append(f"(d) {note}")

    # (f) GRADCAL: out of block 3, into the deck tail
    gr = [i for i, k in _top_level(block3) if k == "GRADCAL"]
    for i in reversed(gr):
        del block3[i]
    gradcal_note = ("(f) GRADCAL moved before the final END" if gr
                    else "(f) GRADCAL and END appended")

    # (e) buffers
    keys = {k for _, k in _top_level(block3)}
    if not ({"BIPOSIZE", "EXCHSIZE"} & keys) and n_super_atoms > BUFFER_ATOM_THRESHOLD:
        block3 += ["BIPOSIZE", BUFFER_SIZE, "EXCHSIZE", BUFFER_SIZE]
        edits.append(f"(e) BIPOSIZE/EXCHSIZE {BUFFER_SIZE} added ({n_super_atoms} atoms in "
                     "the supercell, parent had neither)")
    edits.append(gradcal_note)

    return Template(lines=list(deck.basis_lines) + block3, edits=edits, toldee=target)


def _check_template(deck: ParentDeck, template: Template) -> None:
    """A top-level END, TOLDEE or GRADCAL in the block-3 part would stop or
    duplicate what the deck tail adds."""
    body = template.lines[len(deck.basis_lines):]
    bad = [k for _, k in _top_level(body) if k in ("END", "TOLDEE", "GRADCAL")]
    if bad:
        raise ValueError(f"template block 3 must not contain {', '.join(dict.fromkeys(bad))}")


def template_text(template: Template) -> str:
    """The TEMPLATE file phonopy's CRYSTAL writer (phonopy-init --crystal -d)
    inserts after ``EXTERNAL / ENDGEOM``: explicit basis sets only."""
    return "\n".join(template.lines) + "\n"


def deck_text(title: str, deck: ParentDeck, template: Template) -> str:
    """The full displaced-supercell deck: header, TEMPLATE, then the tail
    phonopy's own writer appends (TOLDEE, GRADCAL, END)."""
    _check_template(deck, template)
    head = [title, "EXTERNAL"]
    if deck.basis_kind == "explicit":
        head.append("END")
    tail = ["TOLDEE", str(template.toldee), "GRADCAL", "END"]
    return "\n".join(head + template.lines + tail) + "\n"


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
