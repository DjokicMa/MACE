"""Build defect deck sets by editing a host deck's text.

The pristine deck is the host deck plus the supercell records; each defect deck
is the pristine deck plus the defect, charge and spin records. Nothing else in
the host is re-rendered: the method, basis set, grids, tolerances and every
other line stay byte-identical, so a defect set always runs the host's method
(never a downgraded one).

Where the records go, and why (CRYSTAL23 manual pages):

* ``SUPERCEL`` / ``SUPERCON`` (block 1, pp.74-77) after the geometry records
  and any other block-1 keywords (e.g. SLABCUT), before the block-1 terminator
  (END, BASISSET or OPTGEOM). SUPERCON refers to the crystallographic cell and
  is 3D only; SUPERCEL refers to the primitive cell (2x2 for slabs).
* ``ATOMREMO`` (p.42), ``ATOMSUBS`` (p.43), ``ATOMINSE`` (p.41) right after the
  supercell records, so their labels and coordinates refer to the supercell.
  Under KEEPSYMM these act on every symmetry-equivalent atom (pp.41-44), so a
  host with KEEPSYMM in block 1 is refused.
* ``DOPING`` (p.103) and ``SPINLOCK`` (p.128) in the SCF block, immediately
  before a trailing ``PPAN`` that directly precedes the final END, otherwise
  before the final END. This reproduces the decks measured on HPCC byte for
  byte (``MAXCYCLE / DOPING / SPINLOCK / PPAN / END``).

The electron count is printed for every charge state, with the spin choice it
leads to; nothing is chosen silently.

This module is standard library only.
"""

import hashlib
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from mace.defects import DefectError, NeedsPristineOutput

#: Lock length written with SPINLOCK. Measured on HPCC (FACTS.md): "SPIN +
#: SPINLOCK 1 30 works" for odd ATOMREMO + DOPING states. MACE's deck writer
#: default (DEFAULT_SPINLOCK_CYCLES = 50) is deliberately not used here: the
#: measured defect runs all used 30.
DEFAULT_SPINLOCK_CYCLES = 30

#: Records that end block 1 (manual: "terminated by END/ENDGEOM or STOP").
_BLOCK1_TERMINATORS = {"END", "ENDG", "ENDGEOM", "BASISSET", "OPTGEOM"}

#: Calculations other than a single point. The defect set is pristine SP + one
#: SP per charge state; anything else is refused with the reason.
_REFUSED_CALCS = {
    "OPTGEOM": "the host is a geometry optimisation; its deck holds the INITIAL geometry, "
               "and a charged supercell must not relax its cell. Build from an SP of the "
               "optimised geometry: mace opt2d12 --calc-type SP",
    "FREQCALC": "the host is a frequency calculation; build from an SP deck "
                "(mace opt2d12 --calc-type SP)",
    "ELASTCON": "the host is an elastic-constant calculation; build from an SP deck",
    "EOS": "the host is an equation-of-state calculation; build from an SP deck",
    "SCELPHONO": "the host already builds a phonon supercell (SCELPHONO, p.73); build from "
                 "an SP deck of the primitive cell",
}

#: Geometry edits that would collide with the ones this builder writes.
_REFUSED_EDITS = {
    "SUPERCEL": "the host already has a supercell (SUPERCEL); build from the primitive host",
    "SUPERCON": "the host already has a supercell (SUPERCON); build from the primitive host",
    "ATOMREMO": "the host already removes atoms (ATOMREMO); build from the pristine host",
    "ATOMSUBS": "the host already substitutes atoms (ATOMSUBS); build from the pristine host",
    "ATOMINSE": "the host already inserts atoms (ATOMINSE); build from the pristine host",
    "GHOSTS": "the host already has ghost atoms (GHOSTS); build from the pristine host",
}

_CHARGE_RECORDS = {
    "CHEMOD": "CHEMOD",
    "CHARGED": "CHARGED",
    "DOPING": "DOPING",
}

_SPIN_LOCK_RECORDS = ("SPINLOCK", "ATOMSPIN", "BETALOCK")

_3C_FUNCTIONALS = {"PBEH3C", "HSE3C", "B973C", "PBESOL03C", "HSESOL3C"}
_3C_HF = {"HF3C", "HFSOL3C"}


# ---------------------------------------------------------------------------
# Imports from Crystal_d12 (read-only use)
# ---------------------------------------------------------------------------

def _crystal_d12_on_path() -> None:
    root = Path(__file__).resolve().parents[2] / "Crystal_d12"
    if root.is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _element_tables():
    _crystal_d12_on_path()
    from d12_constants import ELEMENT_SYMBOLS, SYMBOL_TO_NUMBER, SPACEGROUP_SYMBOLS
    return ELEMENT_SYMBOLS, SYMBOL_TO_NUMBER, SPACEGROUP_SYMBOLS


def element_number(symbol: str) -> int:
    """Atomic number of an element symbol (case-insensitive); DefectError if unknown."""
    _, to_number, _ = _element_tables()
    for sym, z in to_number.items():
        if sym.lower() == symbol.strip().lower():
            return z
    raise DefectError(f"unknown element {symbol!r}")


def element_symbol(z: int) -> str:
    symbols, _, _ = _element_tables()
    if z == 0:
        return "XX"
    return symbols.get(z % 100, f"Z{z}")


def _kw(line: str) -> str:
    return line.strip().upper()


def _ints(line: str) -> Optional[List[int]]:
    try:
        return [int(t) for t in line.split()]
    except ValueError:
        return None


def _floats(line: str) -> Optional[List[float]]:
    try:
        return [float(t) for t in line.split()]
    except ValueError:
        return None


def format_charge(value: float) -> str:
    """Shell charge as CRYSTAL decks write it: 2.0, 3.0, 4.25."""
    text = f"{value:.6f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def doping_value(q: float) -> str:
    """The DOPING record value for charge q: electrons added = -q (p.103)."""
    return repr(float(-q))


def q_tag(q: float) -> str:
    """Job-name tag for a charge state: q0, qp1, qm2."""
    if q == 0:
        return "q0"
    n = abs(q)
    num = str(int(n)) if float(n).is_integer() else str(n).replace(".", "")
    return ("qp" if q > 0 else "qm") + num


def fmt_q(q: float) -> str:
    """A charge for messages: 0, +1, -2."""
    return "0" if q == 0 else f"{q:+g}"


# ---------------------------------------------------------------------------
# Host classification
# ---------------------------------------------------------------------------

@dataclass
class HostInfo:
    """What the builder needs to know about a host deck."""
    lines: List[str]
    trailing_newline: bool
    geometry_kind: str               # CRYSTAL or SLAB
    dimensionality: int              # 3 or 2
    slabcut: bool
    spacegroup: str                  # number or symbol as written
    ifhr: int
    lattice_params: List[float]
    natoms_index: int                # line index of the atom-count record
    atoms: List[Tuple[int, float, float, float]]  # (conventional Z, x, y, z)
    block1_end: int                  # index of the block-1 terminator line
    basis_kind: str                  # "internal" or "explicit"
    basis_name: Optional[str]
    basis_anchor: int                # index where basis-set keyword records go
    block3_start: int                # first line of block 3 (general information / DFT / SCF)
    explicit_basis: Dict[int, List[Tuple[int, float]]]  # conv Z -> [(LAT, CHE)]
    method: str                      # "DFT", "HF" or "3c"
    functional: Optional[str]
    spin: bool                       # SPIN in the DFT block, or UHF
    has: Dict[str, bool] = field(default_factory=dict)  # PPAN, SMEAR, LEVSHIFT, GUESSP, ...
    spin_records: List[str] = field(default_factory=list)
    dispersion: bool = False

    @property
    def elements(self) -> List[int]:
        return sorted({z % 100 for z, *_ in self.atoms})

    @property
    def is_p1(self) -> bool:
        sg = self.spacegroup.strip().upper().replace(" ", "")
        return sg in ("1", "P1")

    def text(self) -> str:
        return "\n".join(self.lines) + ("\n" if self.trailing_newline else "")


def _geometry_atom_count(lines: List[str]) -> Optional[int]:
    """Number of atoms in a deck's geometry block (local copy of the
    charge_records helper, honouring the IFSO > 1 origin-shift record)."""
    dims = {"CRYSTAL": 3, "SLAB": 2, "POLYMER": 2, "MOLECULE": 2}
    for i, line in enumerate(lines[1:], start=1):
        key = line.strip().upper()
        if key in dims:
            j = i + 1
            if key == "CRYSTAL":
                flags = _ints(lines[j]) if j < len(lines) else None
                j += 3
                if flags and len(flags) >= 3 and flags[2] > 1:
                    j += 1
            elif key in ("SLAB", "POLYMER"):
                j += 2
            else:
                j += 1
            try:
                return int(lines[j].split()[0])
            except (ValueError, IndexError):
                return None
    return None


def block1_insert_index(lines: List[str]) -> int:
    """Index of the line that ends block 1 (supercell and defect records go before it).

    Scans, after the geometry records and the atom list, for the first END,
    ENDG(EOM), BASISSET or OPTGEOM; other block-1 keywords in between (e.g.
    SLABCUT and its records) stay before the insertion point. The atom count
    read along the way is cross-checked against the record count of a local
    copy of charge_records' helper; on disagreement the deck is refused.
    """
    kind = _kw(lines[1]) if len(lines) > 1 else ""
    if kind == "CRYSTAL":
        flags = _ints(lines[2]) if len(lines) > 2 else None
        if not flags or len(flags) < 3:
            raise DefectError("line 3 is not the CRYSTAL 'IFLAG IFHR IFSO' record")
        j = 4 + (1 if flags[2] > 1 else 0) + 1
    elif kind == "SLAB":
        j = 4
    else:
        raise DefectError(f"cannot find block 1 of a {kind or 'blank'} geometry")
    try:
        natoms = int(lines[j].split()[0])
    except (ValueError, IndexError):
        raise DefectError(f"line {j + 1} should hold the number of atoms: {lines[j] if j < len(lines) else ''!r}")
    first_atom = j + 1
    for k in range(first_atom, first_atom + natoms):
        vals = lines[k].split() if k < len(lines) else []
        if len(vals) < 4 or _ints(vals[0]) is None or _floats(" ".join(vals[1:4])) is None:
            raise DefectError(f"atom record {k - first_atom + 1} of {natoms} is not 'Z x y z': "
                              f"{lines[k] if k < len(lines) else '<end of deck>'!r}")
    check = _geometry_atom_count(lines)
    if check != natoms:
        raise DefectError(f"atom count disagreement ({natoms} records read, {check} from the "
                          "atom-count record); refusing to edit a deck whose geometry block "
                          "cannot be read unambiguously")
    for k in range(first_atom + natoms, len(lines)):
        if _kw(lines[k]) in _BLOCK1_TERMINATORS:
            return k
    raise DefectError("no END, BASISSET or OPTGEOM after the geometry: block 1 never ends")


def _parse_explicit_basis(lines: List[str], start: int) -> Tuple[Dict[int, List[Tuple[int, float]]], int]:
    """Read an inline basis from ``start`` up to its ``99 0``; return ({conv Z: [(LAT, CHE)]}, index of 99 0)."""
    basis: Dict[int, List[Tuple[int, float]]] = {}
    i = start
    while i < len(lines):
        head = _ints(lines[i])
        if head is None or len(head) != 2:
            raise DefectError(f"explicit basis: expected 'NAT NSHELL' at line {i + 1}: {lines[i]!r}")
        nat, nshell = head
        if nat == 99 and nshell == 0:
            return basis, i
        i += 1
        if nat > 200:
            # Effective core potential follows the atom record (p.84-85).
            ecp = _kw(lines[i])
            i += 1
            if ecp == "INPUT":
                counts = lines[i].split()
                try:
                    i += 1 + sum(int(c) for c in counts[1:])
                except ValueError:
                    raise DefectError(f"explicit basis: unreadable ECP INPUT record at line {i + 1}")
        shells = []
        for _ in range(nshell):
            rec = lines[i].split()
            if len(rec) < 5:
                raise DefectError(f"explicit basis: expected 'ITYB LAT NG CHE SCAL' at line {i + 1}: {lines[i]!r}")
            ityb, lat, ng = int(rec[0]), int(rec[1]), int(rec[2])
            shells.append((lat, float(rec[3])))
            i += 1
            if ityb == 0:
                i += ng
        basis[nat] = shells
    raise DefectError("explicit basis set has no closing '99 0'")


def classify_host(text: str, *, insert: bool = False, host_spin: str = "refuse") -> Tuple[HostInfo, List[str]]:
    """Read a host deck; return (HostInfo, notes) or raise DefectError with the reason.

    ``insert`` is True when an interstitial will be written (BOHR / FRACTION in
    block 1 would change the units of the ATOMINSE coordinates). ``host_spin``
    "keep" accepts SPINLOCK/ATOMSPIN/BETALOCK in the host.
    """
    lines = text.splitlines()
    notes: List[str] = []
    if len(lines) < 4:
        raise DefectError("the host deck is too short to be a CRYSTAL input")
    kind = _kw(lines[1])
    if kind == "MOLECULE":
        raise DefectError("MOLECULE host: a cluster has no defect supercell. For a charged "
                          "molecule or ion cluster write the electron count with DOPING "
                          "(the charged-systems records), not with mace defects")
    if kind in ("POLYMER", "HELIX"):
        raise DefectError(f"{kind} host: only 3D (CRYSTAL) and 2D (SLAB, or CRYSTAL + SLABCUT) "
                          "hosts are supported")
    if kind in ("EXTERNAL", "DLVINPUT"):
        raise DefectError(f"{kind} host: the geometry is in a separate file (fort.34), so the "
                          "supercell and defect records cannot be checked against it; write the "
                          "geometry into the deck first")
    if kind not in ("CRYSTAL", "SLAB"):
        raise DefectError(f"line 2 is {lines[1].strip()!r}, not CRYSTAL or SLAB")

    # Whole-deck refusals (title excluded).
    keys = [_kw(l) for l in lines]
    for i, key in enumerate(keys[1:], start=1):
        if key in _REFUSED_CALCS:
            raise DefectError(f"{key}: {_REFUSED_CALCS[key]}")
        if key in _REFUSED_EDITS:
            raise DefectError(f"{key}: {_REFUSED_EDITS[key]}")
        if key in _CHARGE_RECORDS:
            raise DefectError(f"{key}: the host is already charged or carries charge records "
                              "(CHEMOD/CHARGED/DOPING); build from a neutral pristine host")

    b1 = block1_insert_index(lines)
    for key in keys[2:b1]:
        if key == "KEEPSYMM":
            raise DefectError("KEEPSYMM in block 1: ATOMREMO, ATOMSUBS, ATOMINSE and GHOSTS would act "
                              "on every symmetry-equivalent atom (manual pp.41-44, 54, 82), removing "
                              "or replacing all of them, not one; remove KEEPSYMM from the host")
        if insert and key in ("BOHR", "FRACTION", "BOHRANGS"):
            raise DefectError(f"{key} in block 1 changes the units of the ATOMINSE coordinates "
                              "(p.41, Angstrom by default); remove it from the host to insert an atom")
    slabcut = "SLABCUT" in keys[2:b1]

    # Geometry records
    if kind == "CRYSTAL":
        flags = _ints(lines[2]) or [0, 0, 0]
        spacegroup = lines[3].strip()
        j = 4 + (1 if flags[2] > 1 else 0)
        params = _floats(lines[j]) or []
        nat_idx = j + 1
        ifhr = flags[1]
    else:
        spacegroup = lines[2].strip()
        params = _floats(lines[3]) or []
        nat_idx = 4
        ifhr = 0
    natoms = int(lines[nat_idx].split()[0])
    atoms = []
    for k in range(nat_idx + 1, nat_idx + 1 + natoms):
        v = lines[k].split()
        atoms.append((int(v[0]), float(v[1]), float(v[2]), float(v[3])))

    # Basis
    term = keys[b1]
    explicit: Dict[int, List[Tuple[int, float]]] = {}
    if term == "BASISSET":
        basis_kind, basis_name, anchor = "internal", lines[b1 + 1].strip(), b1 + 2
        block3 = anchor
    elif term in ("END", "ENDG", "ENDGEOM") and b1 + 1 < len(lines) and keys[b1 + 1] == "BASISSET":
        basis_kind, basis_name, anchor = "internal", lines[b1 + 2].strip(), b1 + 3
        block3 = anchor
    elif term in ("END", "ENDG", "ENDGEOM"):
        explicit, idx99 = _parse_explicit_basis(lines, b1 + 1)
        basis_kind, basis_name, anchor = "explicit", None, idx99 + 1
        block3 = None
        for k in range(anchor, len(lines)):
            if keys[k] in ("END", "ENDB", "ENDBS"):
                block3 = k + 1
                break
        if block3 is None:
            raise DefectError("explicit basis set: no END after '99 0'")
    else:
        raise DefectError(f"block 1 ends with {term}, which this builder does not handle")

    rest = keys[block3:]
    has = {k: (k in rest) for k in ("PPAN", "SMEAR", "LEVSHIFT", "GUESSP", "BIPOSIZE", "EXCHSIZE",
                                     "TWOCOMPON", "DFTD3", "GCPAUTO", "GCP")}
    spin_records = [k for k in _SPIN_LOCK_RECORDS if k in rest]

    # Method family
    functional = None
    if "DFT" in rest:
        method = "DFT"
        d = rest.index("DFT")
        for k in rest[d + 1:]:
            if k in ("END", "ENDDFT"):
                break
            if k in ("SPIN",) or not re.match(r"^[A-Z]", k):
                continue
            if functional is None and k not in ("XLGRID", "XXLGRID", "LGRID", "OLDGRID", "RADIAL",
                                                 "ANGULAR", "EXCHANGE", "CORRELAT", "NONLOCAL",
                                                 "HYBRID", "SAVEPRED", "BECKE", "SAVIN"):
                functional = k
        d_end = next((k for k in range(d + 1, len(rest)) if rest[k] in ("END", "ENDDFT")), len(rest))
        spin = "SPIN" in rest[d + 1:d_end]
        if functional and functional.replace("-D3", "").replace("-", "") in _3C_FUNCTIONALS:
            method = "3c"
    else:
        method = "HF"
        spin = "UHF" in rest
        for k in rest:
            if k in _3C_HF:
                method, functional = "3c", k
    dispersion = bool(functional and ("-D3" in functional or "D3" in functional.split("-")[-1])) \
        or has["DFTD3"] or has["GCPAUTO"] or has["GCP"]

    info = HostInfo(lines=lines, trailing_newline=text.endswith("\n"), geometry_kind=kind,
                    dimensionality=2 if (kind == "SLAB" or slabcut) else 3, slabcut=slabcut,
                    spacegroup=spacegroup, ifhr=ifhr, lattice_params=params, natoms_index=nat_idx,
                    atoms=atoms, block1_end=b1, basis_kind=basis_kind, basis_name=basis_name,
                    basis_anchor=anchor, block3_start=block3, explicit_basis=explicit,
                    method=method, functional=functional, spin=spin, has=has,
                    spin_records=spin_records, dispersion=dispersion)

    if spin_records and host_spin != "keep":
        raise DefectError(f"{', '.join(spin_records)} in the host: a fixed spin state set for the "
                          "host cell is not the spin state of each charged supercell. Remove it, or "
                          "pass --host-spin keep and give --spin-state q=NSPIN for every charge")
    if has["TWOCOMPON"]:
        raise DefectError("TWOCOMPON (two-component/SOC) host: defect sets are one-component only")
    if has["SMEAR"]:
        notes.append("WARNING: the host uses SMEAR - total energies are free energies and the "
                     "Fermi level is smeared; formation energies inherit that")
    if has["GUESSP"]:
        notes.append("note: the host has GUESSP; it is kept. The job script drops it when no "
                     "fort.20 can be staged (submitcrystal23.sh), so new supercell jobs cold start")
    return info, notes


def host_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def _det(m: Sequence[Sequence[int]]) -> int:
    if len(m) == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def supercell_records(dim: int, keyword: str, matrix: Sequence[Sequence[int]]) -> List[str]:
    """SUPERCEL / SUPERCON records (pp.74-77): 3x3 integer matrix for 3D, 2x2 for 2D, det != 0."""
    keyword = keyword.upper()
    if keyword not in ("SUPERCEL", "SUPERCON"):
        raise DefectError(f"unknown supercell keyword {keyword}")
    if keyword == "SUPERCON" and dim != 3:
        raise DefectError("SUPERCON refers to the 3D crystallographic cell (p.75); a 2D host "
                          "(SLAB, or CRYSTAL + SLABCUT) takes SUPERCEL with a 2x2 matrix")
    n = 3 if dim == 3 else 2
    if len(matrix) != n or any(len(r) != n for r in matrix):
        raise DefectError(f"a {dim}D supercell needs a {n}x{n} matrix, got {matrix}")
    for row in matrix:
        for v in row:
            if int(v) != v:
                raise DefectError(f"supercell matrix entries must be integers, got {v}")
    if _det(matrix) == 0:
        raise DefectError(f"supercell matrix {matrix} is singular")
    return [keyword] + [" ".join(str(int(v)) for v in row) for row in matrix]


def centring_multiplicity(host: HostInfo) -> int:
    """Lattice points in the crystallographic cell (P 1, A/B/C/I 2, F 4, R 3 in hexagonal axes)."""
    if host.geometry_kind != "CRYSTAL":
        return 1
    sg = host.spacegroup.strip()
    symbol = sg
    if re.fullmatch(r"\d+", sg):
        _, _, groups = _element_tables()
        symbol = groups.get(int(sg), "P")
    letter = symbol.strip()[:1].upper()
    if letter == "R":
        return 1 if host.ifhr == 1 else 3
    return {"P": 1, "A": 2, "B": 2, "C": 2, "I": 2, "F": 4}.get(letter, 1)


def supercell_multiplicity(host: HostInfo, keyword: str, matrix) -> int:
    """Number of primitive cells in the supercell."""
    m = abs(_det(matrix))
    if keyword.upper() == "SUPERCON":
        m *= centring_multiplicity(host)
    return m


def defect_records(kind: str, *, label: Optional[int] = None, z_conv: Optional[int] = None,
                   cart: Optional[Sequence[float]] = None) -> List[str]:
    """Block-1 defect records: ATOMREMO (p.42), ATOMSUBS (p.43), ATOMINSE (p.41)."""
    if kind == "vacancy":
        return ["ATOMREMO", "1", str(label)]
    if kind == "substitution":
        return ["ATOMSUBS", "1", f"{label} {z_conv}"]
    if kind == "interstitial":
        x, y, z = cart
        return ["ATOMINSE", "1", f"{z_conv} {x:.6f} {y:.6f} {z:.6f}"]
    raise DefectError(f"no block-1 records for a {kind}")


# ---------------------------------------------------------------------------
# Electron count and spin
# ---------------------------------------------------------------------------

@dataclass
class ElectronCount:
    exact: Optional[int]
    parity: Optional[int]   # 0 even, 1 odd, None unknown
    source: str

    def shifted(self, delta: int) -> "ElectronCount":
        exact = None if self.exact is None else self.exact + delta
        parity = None if self.parity is None else (self.parity + delta) % 2
        return ElectronCount(exact, parity, self.source)

    def describe(self) -> str:
        if self.exact is not None:
            return f"N={self.exact} {'odd' if self.exact % 2 else 'even'}"
        if self.parity is None:
            return "N unknown"
        return f"N {'odd' if self.parity else 'even'}"


def supercell_electrons(host: HostInfo, multiplicity: int, pristine_n: Optional[int]) -> ElectronCount:
    """Electrons in the pristine supercell.

    From the pristine output when there is one; otherwise its parity: the
    supercell holds ``multiplicity`` primitive cells with the same count, so an
    even multiplicity gives an even count, and a P1 host gives the parity of
    the sum of Z (ECP cores and default shell charges are even or neutral).
    Otherwise unknown until the pristine output exists.
    """
    if pristine_n is not None:
        return ElectronCount(pristine_n, pristine_n % 2, "pristine output")
    if multiplicity % 2 == 0:
        return ElectronCount(None, 0, f"{multiplicity} primitive cells (even)")
    if host.is_p1:
        total = sum(z % 100 for z, *_ in host.atoms) * multiplicity
        return ElectronCount(None, total % 2, "sum of Z over a P1 host")
    return ElectronCount(None, None, "needs the pristine output")


def default_nspin(parity: int) -> int:
    return 1 if parity else 0


# ---------------------------------------------------------------------------
# Deck assembly
# ---------------------------------------------------------------------------

@dataclass
class Deck:
    name: str
    role: str                 # pristine or defect
    q: float                  # defect charge (0 for pristine)
    doping: float             # DOPING value written (0 = none)
    nspin: Optional[int]      # SPINLOCK value written (None = none)
    restricted: bool
    recipe: str
    text: str
    n_electrons: Optional[int] = None
    odd: Optional[bool] = None

    def manifest(self) -> dict:
        return {"name": self.name, "role": self.role, "q": self.q, "doping": self.doping,
                "nspin": self.nspin, "restricted": self.restricted, "odd": self.odd,
                "n_electrons": self.n_electrons, "recipe": self.recipe}


class DeckEditor:
    """Text edits on a classified host. Each method returns a new list of lines."""

    def __init__(self, host: HostInfo):
        self.host = host

    # -- segments --------------------------------------------------------
    def segments(self):
        h = self.host
        lines = h.lines
        return (list(lines[:h.block1_end]), list(lines[h.block1_end:h.basis_anchor]),
                list(lines[h.basis_anchor:h.block3_start]), list(lines[h.block3_start:]))

    @staticmethod
    def join(parts, trailing=True) -> str:
        out = []
        for p in parts:
            out.extend(p)
        return "\n".join(out) + ("\n" if trailing else "")

    # -- block 3 edits ---------------------------------------------------
    @staticmethod
    def final_end(block3: List[str]) -> int:
        for i in range(len(block3) - 1, -1, -1):
            if _kw(block3[i]) == "END":
                return i
        raise DefectError("the host deck has no final END")

    @classmethod
    def scf_insert_index(cls, block3: List[str]) -> int:
        end = cls.final_end(block3)
        if end > 0 and _kw(block3[end - 1]) == "PPAN":
            return end - 1
        return end

    @classmethod
    def ensure_ppan(cls, block3: List[str]) -> Tuple[List[str], bool]:
        if any(_kw(l) == "PPAN" for l in block3):
            return block3, False
        end = cls.final_end(block3)
        return block3[:end] + ["PPAN"] + block3[end:], True

    @staticmethod
    def add_spin(block3: List[str]) -> Tuple[List[str], str]:
        """Make the Hamiltonian open shell: SPIN after DFT, or UHF for HF."""
        keys = [_kw(l) for l in block3]
        if "DFT" in keys:
            d = keys.index("DFT")
            end = next((k for k in range(d + 1, len(keys)) if keys[k] in ("END", "ENDDFT")), len(keys))
            if "SPIN" in keys[d + 1:end]:
                return block3, ""
            return block3[:d + 1] + ["SPIN"] + block3[d + 1:], "added SPIN to the DFT block"
        if "UHF" in keys:
            return block3, ""
        if "RHF" in keys:
            i = keys.index("RHF")
            return block3[:i] + ["UHF"] + block3[i + 1:], "RHF -> UHF"
        return ["UHF"] + block3, "added UHF"

    @staticmethod
    def replace_shrink(block3: List[str], tokens: Sequence[int]) -> List[str]:
        keys = [_kw(l) for l in block3]
        if "SHRINK" not in keys:
            raise DefectError("--shrink given but the host has no SHRINK record")
        i = keys.index("SHRINK")
        first = _ints(block3[i + 1]) or []
        old_len = 3 if (first and first[0] == 0) else 2
        if len(tokens) == 1:
            new = [f"{tokens[0]} {2 * tokens[0]}"]
        elif len(tokens) == 2:
            new = [f"{tokens[0]} {tokens[1]}"]
        elif len(tokens) == 3:
            new = [f"0 {2 * max(tokens)}", " ".join(str(t) for t in tokens)]
        else:
            raise DefectError("--shrink takes IS, 'IS ISP' or 'ka kb kc'")
        return block3[:i + 1] + new + block3[i + old_len:]

    @staticmethod
    def set_host_spinlock(block3: List[str], nspin: int) -> Tuple[List[str], bool]:
        keys = [_kw(l) for l in block3]
        if "SPINLOCK" not in keys:
            return block3, False
        i = keys.index("SPINLOCK")
        rec = block3[i + 1].split()
        cycles = rec[1] if len(rec) > 1 else str(DEFAULT_SPINLOCK_CYCLES)
        return block3[:i + 1] + [f"{nspin} {cycles}"] + block3[i + 2:], True


def host_shrink(host: HostInfo) -> Optional[List[int]]:
    block3 = host.lines[host.block3_start:]
    keys = [_kw(l) for l in block3]
    if "SHRINK" not in keys:
        return None
    i = keys.index("SHRINK")
    first = _ints(block3[i + 1]) or []
    if first and first[0] == 0 and i + 2 < len(block3):
        return _ints(block3[i + 2])
    return first[:1] if first else None


def suggest_shrink(host: HostInfo, matrix) -> Optional[str]:
    """Equivalent-density SHRINK for the supercell (printed, never written unless --shrink)."""
    mesh = host_shrink(host)
    if not mesh:
        return None
    dim = len(matrix)
    lengths = [math.sqrt(sum(v * v for v in row)) for row in matrix]
    if len(mesh) == 1:
        mesh = mesh * dim
    new = [max(1, math.ceil(k / L)) for k, L in zip(mesh, lengths)]
    if len(set(new)) == 1:
        return f"{new[0]}"
    return " ".join(str(n) for n in new)


@dataclass
class SetOptions:
    charges: List[float]
    spin_states: Dict[float, int] = field(default_factory=dict)
    spinlock_cycles: int = DEFAULT_SPINLOCK_CYCLES
    host_spin: str = "refuse"
    shrink: Optional[List[int]] = None


@dataclass
class DefectSite:
    kind: str                 # vacancy, substitution, interstitial
    element: str              # removed / replaced / inserted element
    new_element: Optional[str] = None
    label: Optional[int] = None
    cart: Optional[Tuple[float, float, float]] = None

    @property
    def delta_z(self) -> int:
        z_old = element_number(self.element)
        if self.kind == "vacancy":
            return -z_old
        if self.kind == "substitution":
            return element_number(self.new_element) - z_old
        return z_old    # interstitial: element is the inserted one

    def describe(self) -> str:
        if self.kind == "vacancy":
            return f"V_{self.element}"
        if self.kind == "substitution":
            return f"{self.new_element}_{self.element}"
        return f"{self.element}_i"


def _conv_number(host: HostInfo, z: int) -> int:
    """Conventional atomic number the host's basis uses for element z."""
    if host.basis_kind == "explicit":
        convs = [c for c in host.explicit_basis if c % 100 == z]
        if len(convs) > 1:
            raise DefectError(f"the explicit basis has several sets for {element_symbol(z)} "
                              f"({convs}); cannot tell which one the new atom should use")
        if convs:
            return convs[0]
    return z


def check_element_coverage(host: HostInfo, z: int) -> None:
    """A new element must have a basis: explicit -> present inline; internal -> library covers it."""
    sym = element_symbol(z)
    if host.basis_kind == "explicit":
        if not any(c % 100 == z for c in host.explicit_basis):
            raise DefectError(f"{sym} is not in the host's explicit basis set: add a {sym} basis "
                              "block (before '99 0') to the host deck first")
        return
    _crystal_d12_on_path()
    from d12_constants import check_basis_set_compatibility
    ok, missing = check_basis_set_compatibility(host.basis_name, [z], "INTERNAL")
    if not ok:
        raise DefectError(f"the internal basis {host.basis_name} has no {sym} "
                          f"(missing Z={missing}); choose a host basis that covers it")


def build_supercell_set(host: HostInfo, site: DefectSite, keyword: str, matrix, opts: SetOptions,
                        electrons: ElectronCount, name: str, title_note: str = "") -> Tuple[List[Deck], List[str], List[str]]:
    """Pristine + defect decks for one supercell.

    Returns (decks, messages, added_records). ``electrons`` is the pristine
    supercell's count (exact or parity).
    """
    if electrons.parity is None and opts.charges:
        raise NeedsPristineOutput("the electron parity of the supercell is unknown: the supercell "
                                  "holds an odd number of primitive cells of a symmetric host")
    msgs: List[str] = []
    added: List[str] = []
    ed = DeckEditor(host)
    head, basis, basis_tail, block3 = ed.segments()
    sc = supercell_records(host.dimensionality, keyword, matrix)

    block3_p = list(block3)
    block3_p, ppan_added = ed.ensure_ppan(block3_p)
    if ppan_added:
        added.append("PPAN")
        msgs.append("added PPAN to every deck of the set (pristine included): the electron count "
                    "is read from the Mulliken 'NO. OF ELECTRONS' line, printed only with PPAN")
    if opts.shrink:
        block3_p = ed.replace_shrink(block3_p, opts.shrink)
        added.append("SHRINK " + " ".join(str(t) for t in opts.shrink))

    title = lambda role: (f"{name} {role}{title_note}")[:80]
    decks = [Deck(name=f"{name}_pris", role="pristine", q=0, doping=0.0, nspin=None,
                  restricted=not host.spin, recipe="pristine",
                  text=ed.join([[title(f"pristine {sc[0]}")], head[1:], sc,
                                basis, basis_tail, block3_p], host.trailing_newline),
                  n_electrons=electrons.exact, odd=bool(electrons.parity))]

    if site.kind in ("substitution", "interstitial"):
        z_new = element_number(site.new_element if site.kind == "substitution" else site.element)
        check_element_coverage(host, z_new)
    rec1: List[str] = []
    if site.kind == "vacancy":
        rec1 = defect_records("vacancy", label=site.label)
    elif site.kind == "substitution":
        rec1 = defect_records("substitution", label=site.label, z_conv=_conv_number(host, z_new))
    elif site.kind == "interstitial":
        rec1 = defect_records("interstitial", z_conv=_conv_number(host, z_new), cart=site.cart)
        msgs.append("note: ATOMINSE (Cartesian Angstrom, current = supercell frame, p.41) is untested on HPCC")

    n_def0 = electrons.shifted(site.delta_z)
    for q in opts.charges:
        if not float(q).is_integer():
            raise DefectError(f"charge states are integers; got {q}")
        n_q = n_def0.shifted(-int(q))
        parity = n_q.parity
        if q in opts.spin_states:
            nspin = opts.spin_states[q]
            if nspin % 2 != parity:
                raise DefectError(f"--spin-state q={fmt_q(q)}={nspin}: NSPIN must be "
                                  f"{'odd' if parity else 'even'} for {n_q.describe()} "
                                  "(manual p.128: NSPIN odd when the electron count is odd)")
        elif opts.host_spin == "keep":
            raise DefectError(f"--host-spin keep needs --spin-state for every charge (missing q={fmt_q(q)})")
        else:
            nspin = default_nspin(parity)
        locked = bool(parity) or nspin != 0

        b3 = list(block3_p)
        tail = list(basis_tail)
        scf: List[str] = []
        recipe = site.kind
        spin_note = ""
        if q != 0:
            scf += ["DOPING", doping_value(q)]

        if locked:
            if opts.host_spin == "keep" and "SPINLOCK" in host.spin_records:
                b3, _ = DeckEditor.set_host_spinlock(b3, nspin)
                spin_note = f"host SPINLOCK set to {nspin}"
            else:
                b3, how = ed.add_spin(b3)
                scf += ["SPINLOCK", f"{nspin} {opts.spinlock_cycles}"]
                spin_note = (how + ", " if how else "") + f"SPINLOCK {nspin} {opts.spinlock_cycles}"
        at = ed.scf_insert_index(b3)
        b3 = b3[:at] + scf + b3[at:]
        head_d = head[1:] + sc + rec1
        text = ed.join([[title(f"{site.describe()} q={fmt_q(q)}")], head_d, basis, tail, b3],
                       host.trailing_newline)
        decks.append(Deck(name=f"{name}_{q_tag(q)}", role="defect", q=q,
                          doping=-q,
                          nspin=nspin if locked else None,
                          restricted=not (host.spin or locked), recipe=recipe, text=text,
                          n_electrons=n_q.exact, odd=bool(parity)))
        what = spin_note or ("closed shell" if not host.spin else "host SPIN kept, unlocked")
        msgs.append(f"q={fmt_q(q)}: {n_q.describe()} -> {what}"
                    + (f"; DOPING {doping_value(q)}" if q != 0 else ""))

    return decks, msgs, added

