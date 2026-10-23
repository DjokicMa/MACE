"""What the ORBITALS run's CRYSTAL output (.out) adds to the Molden files.

The Molden files carry no k vector, no shrinking factor, no spin multiplicity
and atoms wrapped to [-0.5, 0.5); the .out has:

* ``*** K POINTS COORDINATES (OBLIQUE COORDINATES IN UNITS OF IS =  4)`` with
  entries ``3-R(  2  0  0)``: k index -> oblique k and real/complex;
* the atom table (``ATOM N.AT. SHELL X(A) Y(A) Z(A)``) with CRYSTAL's internal
  atom images, which zone-boundary Bloch phases need;
* ``File opened : <name>.molden`` lines, band edges, the Fermi energy, the
  electron counts (valence count with ECPs) and the calculation type.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_KHEAD = re.compile(r"K POINTS COORDINATES \(OBLIQUE COORDINATES IN UNITS OF IS\s*=\s*(\d+)\)")
_KENTRY = re.compile(r"(\d+)-([RC])\(\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\)")
_SHRINK = re.compile(r"SHRINK FACTORS\(MONK\.\)\s+(\d+)\s+(\d+)\s+(\d+)")
_OPENED = re.compile(r"File opened(?: \(complex\))?\s*:\s*(\S+\.molden)")
_ATOMHEAD = re.compile(r"ATOM\s+N\.AT\.\s+SHELL\s+X\(A\)\s+Y\(A\)\s+Z\(A\)")
_ATOMROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+([A-Z]{1,2})\s+\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")
_TOP = re.compile(r"TOP OF VALENCE BANDS -\s+BAND\s+(\d+);\s+K\s+(\d+);\s+EIG\s+(\S+)")
_BOTTOM = re.compile(r"BOTTOM OF VIRTUAL BANDS -\s+BAND\s+(\d+);\s+K\s+(\d+);\s+EIG\s+(\S+)")
# only the NEWK step's "EFERMI(AU)" line: the parent-SCF summary "N. OF SCF CYCLES ... FERMI ENERGY
# -0.100E+01" is a 3-digit placeholder (-1.0 for an insulating molecule), not a Fermi level
_EFERMI = re.compile(r"EFERMI\(AU\)\s+(-?\d+\.\d+(?:E[-+]\d+)?)")
_NELEC = re.compile(r"N\. OF ELECTRONS PER CELL\s+(\d+)")
_NCORE = re.compile(r"CORE ELECTRONS PER CELL\s+(\d+)")
_NAO = re.compile(r"NUMBER OF AO\s+(\d+)")
_CALC = re.compile(r"TYPE OF CALCULATION :\s+(.+?)\s*$", re.M)
# a spin-polarized run's net spin (CRYSTAL23: "SPIN POLARIZED SYSTEM: SUMMED SPIN DENSITY   -0.00000000")
_SPINDENS = re.compile(r"SUMMED SPIN DENSITY\s+(-?\d+\.\d+)")


_PRINTING = re.compile(r"PRINTING OF (?:CRYSTALLINE|MOLECULAR) ORBITALS")


def has_orbitals_section(text: str) -> bool:
    """The .out ran ORBITALS (periodic: CRYSTALLINE, 0D: MOLECULAR ORBITALS)."""
    return _PRINTING.search(text) is not None


@dataclass
class KPoint:
    index: int                 # 1-based position in the .out list (the nnnnn of the file names)
    oblique: Tuple[int, int, int]
    shrink: Tuple[int, int, int]
    real: bool

    @property
    def frac(self) -> np.ndarray:
        """k in fractions of the reciprocal lattice vectors."""
        return np.array([o / s if s else 0.0 for o, s in zip(self.oblique, self.shrink)], float)

    def label(self) -> str:
        f = self.frac
        if not np.any(f):
            return "Γ"
        return "(" + ", ".join(_frac_text(x) for x in f) + ")"


def _frac_text(x: float) -> str:
    from fractions import Fraction
    fr = Fraction(x).limit_denominator(48)
    return str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"


@dataclass
class OrbitalsOut:
    path: str
    kpoints: List[KPoint] = field(default_factory=list)
    shrink: Optional[Tuple[int, int, int]] = None
    atoms_ang: Optional[np.ndarray] = None       # (nat, 3) Cartesian Angstrom, CRYSTAL's internal images
    atom_symbols: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    electrons: Optional[int] = None
    core_electrons: Optional[int] = None
    nao: Optional[int] = None
    calc_type: Optional[str] = None
    spin_density: Optional[float] = None         # a spin-polarized run's summed spin density (n_alpha - n_beta)
    vbm: Optional[Tuple[int, int, float]] = None   # band, k, eig (Ha)
    cbm: Optional[Tuple[int, int, float]] = None
    efermi: Optional[float] = None
    has_orbitals: bool = False
    errors: List[str] = field(default_factory=list)

    @property
    def valence_electrons(self) -> Optional[int]:
        """Electrons the orbitals describe (ECP cores excluded; all-electron cores included)."""
        return self.electrons

    def kpoint(self, index: int) -> Optional[KPoint]:
        for k in self.kpoints:
            if k.index == index:
                return k
        return None


def parse_out(path) -> OrbitalsOut:
    path = Path(path)
    text = path.read_text(errors="replace")
    return parse_out_text(text, str(path))


def parse_out_text(text: str, path: str = "") -> OrbitalsOut:
    res = OrbitalsOut(path=path)
    res.has_orbitals = has_orbitals_section(text)
    for ln in text.splitlines():
        if "ERROR" in ln and "****" in ln:
            res.errors.append(ln.strip())
    # the last k list printed (NEWK's) is the one ORBITALS uses
    heads = list(_KHEAD.finditer(text))
    if heads:
        m = heads[-1]
        IS = int(m.group(1))
        shrink = None
        sh = list(_SHRINK.finditer(text[:m.start()]))
        if sh:
            shrink = tuple(int(x) for x in sh[-1].groups())
        if shrink is None:
            shrink = (IS, IS, IS)
        # oblique coordinates are in units of IS along every direction
        res.shrink = shrink
        body = text[m.end():]
        block = []
        for ln in body.splitlines()[1:]:
            if not ln.strip():
                break
            if not _KENTRY.search(ln):
                break
            block.append(ln)
        for e in _KENTRY.finditer("\n".join(block)):
            res.kpoints.append(KPoint(index=int(e.group(1)), oblique=tuple(int(e.group(i)) for i in (3, 4, 5)),
                                      shrink=(IS, IS, IS), real=e.group(2) == "R"))
    # atom table: the last one printed
    lines = text.splitlines()
    head_idx = [i for i, ln in enumerate(lines) if _ATOMHEAD.search(ln)]
    if head_idx:
        rows, syms = [], []
        for ln in lines[head_idx[-1] + 1:]:
            if ln.strip().startswith("*") and rows:
                break
            r = _ATOMROW.match(ln)
            if r:
                syms.append(r.group(3).capitalize())
                rows.append([float(r.group(i)) for i in (4, 5, 6)])
        if rows:
            res.atoms_ang = np.array(rows)
            res.atom_symbols = syms
    res.files = list(dict.fromkeys(_OPENED.findall(text)))
    m = _NELEC.search(text)
    res.electrons = int(m.group(1)) if m else None
    m = _NCORE.search(text)
    res.core_electrons = int(m.group(1)) if m else None
    m = _NAO.search(text)
    res.nao = int(m.group(1)) if m else None
    m = _CALC.search(text)
    res.calc_type = m.group(1).strip() if m else None
    m = _SPINDENS.search(text)
    res.spin_density = float(m.group(1)) if m else None
    # band edges / Fermi level printed after the NEWK k list
    tail = text[heads[-1].end():] if heads else text
    tops = [(int(a), int(b), float(c)) for a, b, c in _TOP.findall(tail)]
    bots = [(int(a), int(b), float(c)) for a, b, c in _BOTTOM.findall(tail)]
    if tops:
        res.vbm = max(tops, key=lambda t: t[2])
    if bots:
        res.cbm = min(bots, key=lambda t: t[2])
    ef = _EFERMI.findall(tail)
    if ef:
        try:
            res.efermi = float(ef[-1])
        except ValueError:
            res.efermi = None
    return res


_ROOT_RE = re.compile(r"^(?P<root>.+?)(?:_(?P<n>\d{5})_K(?P<k>\d+)_(?P<part>real|complex))?\.molden$")


def split_name(name: str) -> Dict[str, Optional[str]]:
    """``si_orb_00003_K200_real.molden`` -> root, k index (int), part; molecules have no k."""
    m = _ROOT_RE.match(Path(name).name)
    if not m:
        return {"root": None, "n": None, "part": None, "label": None}
    return {"root": m.group("root"), "n": int(m.group("n")) if m.group("n") else None,
            "part": m.group("part"), "label": m.group("k")}


def find_out_for(molden_path) -> Optional[Path]:
    """The CRYSTAL output that wrote this Molden file (same folder), or None."""
    p = Path(molden_path)
    name = p.name
    info = split_name(name)
    root = info["root"] or p.stem
    cands = []
    for o in sorted(p.parent.glob("*.out")):
        stem = o.stem
        score = 0
        if root == stem or root == stem + "_orb" or root.startswith(stem):
            score = 2
        cands.append((score, o))
    cands.sort(key=lambda t: -t[0])
    for score, o in cands:
        try:
            if o.stat().st_size > 50_000_000:
                continue
            txt = o.read_text(errors="replace")
        except OSError:
            continue
        if ("File opened : " + name) in txt or ("File opened (complex): " + name) in txt:
            return o
    for score, o in cands:
        if score == 2:
            try:
                if has_orbitals_section(o.read_text(errors="replace")):
                    return o
            except OSError:
                continue
    return None
