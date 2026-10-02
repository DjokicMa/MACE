"""Read what a defect analysis needs from CRYSTAL23 outputs and POTC.DAT files.

Private to mace.defects. The Mulliken parsing here duplicates what the
charged-systems work reads; the two are to be unified later.

Measured facts this module relies on (FACTS.md, HPCC, CRYSTAL23):

* DOPING prints no message, and the header "N. OF ELECTRONS PER CELL" still
  shows the count BEFORE doping. The real count is the Mulliken
  "NO. OF ELECTRONS" line (ALPHA+BETA block, printed with PPAN). The header is
  never used as the electron count.
* POTC.DAT (``POTC / 0 0 0``): point index, x y z in bohr, V0, field
  components; atomic units, 6 significant figures. The planar average
  (``POTC / 2 41 0``) is an XMGR text file with ``#`` and ``@`` lines.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

_SCF_END = re.compile(r"== SCF ENDED - (.*?)\s+E\(AU\)\s+(\S+)\s+CYCLES\s+(\d+)")
_MULL = re.compile(r"MULLIKEN POPULATION ANALYSIS - NO\. OF ELECTRONS\s+(\S+)")
_ATOM_ROW = re.compile(r"^\s*(\d+)\s+([A-Za-z]{1,2})\s+(\d+)\s+(-?\d+\.\d+)")
_CONT_ROW = re.compile(r"^\s+-?\d+\.\d+(\s+-?\d+\.\d+)*\s*$")
_TOP = re.compile(r"TOP OF VALENCE BANDS -\s+BAND\s+(\d+);\s+K\s+(\d+);\s+EIG\s+(\S+)")
_BOTTOM = re.compile(r"BOTTOM OF VIRTUAL BANDS -\s+BAND\s+(\d+);\s+K\s+(\d+);\s+EIG\s+(\S+)")


def scf_energy(text: str) -> Tuple[Optional[float], str]:
    """(E in Ha, "") for an SCF converged on energy, else (None, reason)."""
    found = None
    for found in _SCF_END.finditer(text):
        pass
    if found is None:
        if "TOO MANY CYCLES" in text:
            return None, "SCF not converged (TOO MANY CYCLES)"
        return None, "no '== SCF ENDED' line (still running, or failed)"
    status = found.group(1)
    if "TOO MANY CYCLES" in status or "CONVERGENCE ON ENERGY" not in status:
        return None, f"SCF ended without convergence ({status.strip()})"
    return float(found.group(2)), ""


def scf_cycles(text: str) -> Optional[int]:
    """Number of SCF cycles on the last '== SCF ENDED' line."""
    found = None
    for found in _SCF_END.finditer(text):
        pass
    return int(found.group(3)) if found else None


def header_electrons(text: str) -> Optional[int]:
    """'N. OF ELECTRONS PER CELL' - the count BEFORE DOPING (FACTS.md); a cross-check only."""
    m = None
    for m in re.finditer(r"N\. OF ELECTRONS PER CELL\s+(\d+)", text):
        pass
    return int(m.group(1)) if m else None


@dataclass
class Mulliken:
    total: float
    atoms: List[Tuple[int, str, int, float]]   # (label, symbol, Z, population)

    def charge(self, label: int) -> Optional[float]:
        """Net charge Z - population (ghost XX rows: -population)."""
        for lab, _, z, pop in self.atoms:
            if lab == label:
                return z - pop
        return None

    def population(self, label: int) -> Optional[float]:
        for lab, _, _, pop in self.atoms:
            if lab == label:
                return pop
        return None


def _mulliken_block(lines: List[str], start: int) -> Optional[Mulliken]:
    m = _MULL.search(lines[start])
    if not m:
        return None
    atoms = []
    started = False
    for line in lines[start + 1:]:
        a = _ATOM_ROW.match(line)
        if a:
            started = True
            atoms.append((int(a.group(1)), a.group(2).upper(), int(a.group(3)), float(a.group(4))))
            continue
        if not line.strip() or (started and _CONT_ROW.match(line)) or "ATOM" in line and not started:
            continue
        if started:
            break
    return Mulliken(float(m.group(1)), atoms)


def _blocks(text: str, title: str) -> List[Mulliken]:
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        if title in line and i + 1 < len(lines) and _MULL.search(lines[i + 1]):
            blk = _mulliken_block(lines, i + 1)
            if blk:
                out.append(blk)
    return out


def mulliken(text: str) -> Optional[Mulliken]:
    """The last ALPHA+BETA Mulliken block (ghost XX rows included)."""
    blocks = _blocks(text, "ALPHA+BETA ELECTRONS")
    return blocks[-1] if blocks else None


def mulliken_spin(text: str) -> Optional[Mulliken]:
    """The last ALPHA-BETA Mulliken block; its total is the cell's net spin."""
    blocks = _blocks(text, "ALPHA-BETA ELECTRONS")
    return blocks[-1] if blocks else None


def electron_count(text: str) -> Optional[float]:
    """Electrons in the cell from the Mulliken 'NO. OF ELECTRONS' line (after DOPING)."""
    m = mulliken(text)
    return m.total if m else None


@dataclass
class BandEdges:
    vbm: Optional[float] = None          # Ha
    cbm: Optional[float] = None
    vbm_band: Optional[int] = None
    vbm_k: Optional[int] = None
    cbm_band: Optional[int] = None
    cbm_k: Optional[int] = None

    @property
    def gap(self) -> Optional[float]:
        if self.vbm is None or self.cbm is None:
            return None
        return self.cbm - self.vbm


def band_edges(text: str) -> BandEdges:
    """Top of the valence bands and bottom of the virtual bands of the last SCF cycle printed.

    Several TOP lines can appear in one cycle (alpha and beta for spin
    outputs, or different k points); the highest VBM and the lowest CBM are
    taken, with their band and k index.
    """
    lines = text.splitlines()
    last = None
    for i, line in enumerate(lines):
        if _TOP.search(line):
            last = i
    if last is None:
        return BandEdges()
    start = last
    while start > 0 and "TTTTTTTTTT" not in lines[start - 1]:
        start -= 1
    end = last
    while end + 1 < len(lines) and "TTTTTTTTTT" not in lines[end + 1]:
        end += 1
    edges = BandEdges()
    for line in lines[start:end + 1]:
        t = _TOP.search(line)
        if t:
            e = float(t.group(3))
            if edges.vbm is None or e > edges.vbm:
                edges.vbm, edges.vbm_band, edges.vbm_k = e, int(t.group(1)), int(t.group(2))
        b = _BOTTOM.search(line)
        if b:
            e = float(b.group(3))
            if edges.cbm is None or e < edges.cbm:
                edges.cbm, edges.cbm_band, edges.cbm_k = e, int(b.group(1)), int(b.group(2))
    return edges


def efermi(text: str) -> Optional[float]:
    """The last 'EFERMI(AU)' value (conducting / fractionally occupied states)."""
    m = None
    for m in re.finditer(r"EFERMI\(AU\)\s+(\S+)", text):
        pass
    return float(m.group(1)) if m else None


def spheropole(text: str) -> Optional[float]:
    """The last '::: EXT EL-SPHEROPOLE' energy term (Ha)."""
    m = None
    for m in re.finditer(r"::: EXT EL-SPHEROPOLE\s+(\S+)", text):
        pass
    return float(m.group(1)) if m else None


@dataclass
class OutputSummary:
    name: str
    energy: Optional[float] = None
    reason: str = ""
    cycles: Optional[int] = None
    electrons: Optional[float] = None
    header_electrons: Optional[int] = None
    spin: Optional[float] = None
    edges: BandEdges = field(default_factory=BandEdges)
    efermi: Optional[float] = None
    spheropole: Optional[float] = None
    mulliken: Optional[Mulliken] = None
    mulliken_spin: Optional[Mulliken] = None


def summarise(name: str, text: str) -> OutputSummary:
    e, reason = scf_energy(text)
    mul = mulliken(text)
    spin = mulliken_spin(text)
    return OutputSummary(name=name, energy=e, reason=reason, cycles=scf_cycles(text),
                         electrons=mul.total if mul else None,
                         header_electrons=header_electrons(text),
                         spin=spin.total if spin else None, edges=band_edges(text),
                         efermi=efermi(text), spheropole=spheropole(text),
                         mulliken=mul, mulliken_spin=spin)


# ---------------------------------------------------------------------------
# POTC.DAT
# ---------------------------------------------------------------------------

def read_potc_atoms(text: str) -> List[Tuple[int, Tuple[float, float, float], float]]:
    """POTC / 0 0 0: [(point, (x, y, z) bohr, V0 au)]."""
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0])
            xyz = (float(parts[1]), float(parts[2]), float(parts[3]))
            v = float(parts[4])
        except ValueError:
            continue
        out.append((idx, xyz, v))
    return out


def read_potc_planar(text: str) -> List[Tuple[float, float]]:
    """POTC / 2 41 0: [(z bohr, V au)] from the XMGR text, '#' and '@' lines skipped."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("@"):
            continue
        parts = s.split()
        try:
            out.append((float(parts[0]), float(parts[1])))
        except (ValueError, IndexError):
            continue
    return out
