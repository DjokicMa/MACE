"""Read Molden files, including CRYSTAL23's periodic "modified Molden".

The rules come from the files CRYSTAL23 writes with the ORBITALS keyword
(checked on molecules, polymers, slabs and bulk crystals):

* ``[Cell]`` is garbage (bulk) or absent (slab); only ``[CellAxes]`` (Angstrom,
  lattice vectors as rows, an all-zero row for a non-periodic direction) is used.
* ``[Atoms] (Fractional)`` holds fractional coordinates along the periodic axes
  and Cartesian Angstrom along the non-periodic ones. ``(Angs)`` / ``(AU)`` are
  plain Cartesian.
* ``sp`` shells are written as separate ``s`` and ``p`` shells (a combined
  ``sp`` shell is still accepted).
* contraction coefficients multiply raw primitives ``r^l exp(-a r^2)``.
* ``[5D]`` / ``[7F]`` (spherical d and f) are always written.
* BETA coefficients are numbered ``nbf+1 .. 2*nbf`` (a generic Molden reader
  would read them as zeros); indices are taken modulo ``nbf``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

BOHR_ANG = 0.529177210903  # Angstrom per bohr

_LMAP = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}
_SECTION = re.compile(r"^\s*\[([^\]]+)\](.*)$")


class MoldenError(ValueError):
    """The file is not a Molden file MACE can use."""


@dataclass
class Shell:
    atom: int            # 0-based atom index
    l: int
    exps: np.ndarray     # (nprim,)
    coefs: np.ndarray    # (nprim,), multiply raw primitives r^l exp(-a r^2)


@dataclass
class MoldenData:
    path: str
    symbols: List[str]
    Z: np.ndarray                       # nuclear charge as written (full Z, even with an ECP)
    coords_raw: np.ndarray              # (nat, 3) as written
    atoms_mode: str                     # "angs", "bohr" or "fractional"
    axes_ang: Optional[np.ndarray]      # (3, 3) lattice vectors as rows, Angstrom, or None
    shells: List[Shell]
    spherical_d: bool
    spherical_f: bool
    energies: np.ndarray                # (nmo,) Hartree
    occupations: np.ndarray             # (nmo,)
    spins: List[str]                    # "ALPHA", "BETA" or "ALPHA+BETA" per MO
    coef: np.ndarray                    # (nbf, nmo) float
    nbf: int
    sections: List[str] = field(default_factory=list)

    @property
    def periodic_dims(self) -> List[int]:
        if self.axes_ang is None:
            return []
        return [i for i in range(3) if np.linalg.norm(self.axes_ang[i]) > 1e-8]

    @property
    def dimensionality(self) -> int:
        return len(self.periodic_dims)

    def positions_bohr(self) -> np.ndarray:
        """Cartesian atom positions (bohr) as the file writes them."""
        if self.atoms_mode == "bohr":
            return self.coords_raw.copy()
        if self.atoms_mode == "angs":
            return self.coords_raw / BOHR_ANG
        pos = np.zeros_like(self.coords_raw)
        per = self.periodic_dims
        ax = self.axes_ang / BOHR_ANG if self.axes_ang is not None else np.zeros((3, 3))
        for i in range(3):
            if i in per:
                pos += self.coords_raw[:, i:i + 1] * ax[i][None, :]
            else:
                pos[:, i] += self.coords_raw[:, i] / BOHR_ANG
        return pos


def _num(tok: str) -> float:
    return float(tok.replace("D", "E").replace("d", "e"))


def _split_sections(lines):
    out = []
    cur = None
    for ln in lines:
        m = _SECTION.match(ln)
        if m:
            cur = [m.group(1).strip().upper(), m.group(2).strip(), []]
            out.append(cur)
        elif cur is not None:
            cur[2].append(ln)
    return out


def _parse_gto(lines) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
    shells = []
    atom = None
    i, n = 0, len(lines)
    while i < n:
        p = lines[i].split()
        i += 1
        if not p:
            continue
        head = p[0].lower()
        if head.isdigit() and (len(p) == 1 or (len(p) == 2 and p[1] == "0")):
            atom = int(head) - 1
            continue
        if atom is None or (head not in _LMAP and head != "sp"):
            raise MoldenError(f"unexpected [GTO] line: {lines[i - 1].strip()!r}")
        nprim = int(p[1])
        rows = []
        for _ in range(nprim):
            if i >= n:
                raise MoldenError("[GTO] ends inside a shell")
            rows.append([_num(v) for v in lines[i].split()])
            i += 1
        arr = np.array(rows, float)
        if head == "sp":
            shells.append((atom, 0, arr[:, 0].copy(), arr[:, 1].copy()))
            shells.append((atom, 1, arr[:, 0].copy(), arr[:, 2].copy()))
        else:
            shells.append((atom, _LMAP[head], arr[:, 0].copy(), arr[:, 1].copy()))
    return shells


def _nfunc(l: int, sph_d: bool, sph_f: bool) -> int:
    if (l == 2 and sph_d) or (l == 3 and sph_f) or l >= 4:
        return 2 * l + 1
    return (l + 1) * (l + 2) // 2


def _parse_mo(lines, nbf: int):
    ene, occ, spin, cols = [], [], [], []
    cur = None
    in_coef = False
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if "=" in s:
            key, val = (t.strip() for t in s.split("=", 1))
            key = key.lower()
            if cur is None or in_coef:
                cur = np.zeros(nbf)
                cols.append(cur)
                ene.append(np.nan)
                occ.append(0.0)
                spin.append("ALPHA+BETA")
                in_coef = False
            if key == "ene":
                ene[-1] = _num(val)
            elif key == "occup":
                occ[-1] = _num(val)
            elif key == "spin":
                spin[-1] = val.upper().replace(" ", "")
            continue
        if cur is None:
            continue
        in_coef = True
        p = s.split()
        idx = int(p[0]) - 1
        # CRYSTAL numbers BETA coefficients nbf+1 .. 2*nbf; any other index outside 1..nbf means
        # the basis was read with the wrong function count (e.g. Cartesian vs spherical f)
        if nbf <= idx < 2 * nbf and spin[-1] == "BETA":
            idx -= nbf
        elif not 0 <= idx < nbf:
            raise MoldenError(f"MO coefficient index {idx + 1} is outside 1..{nbf} (the [GTO] section "
                              f"gives {nbf} basis functions; check the [5D]/[7F] flags)")
        cur[idx] = _num(p[1])
    coef = np.column_stack(cols) if cols else np.zeros((nbf, 0))
    return np.array(ene), np.array(occ), spin, coef


def read_molden(path) -> MoldenData:
    """Parse a Molden / CRYSTAL periodic Molden file."""
    path = Path(path)
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        raise MoldenError(f"cannot read {path}: {exc}") from exc
    lines = text.splitlines()
    if not any(ln.strip().lower().startswith("[molden format]") for ln in lines[:5]):
        raise MoldenError(f"{path.name} is not a Molden file (no [Molden Format] header)")
    secs = _split_sections(lines)
    names = [s[0] for s in secs]
    by = {}
    for name, hdr, body in secs:
        by.setdefault(name, (hdr, body))

    sph_d = any(n in ("5D", "5D7F", "5D10F") for n in names)
    # Molden convention: [5D] alone means 5D and 7F; [5D10F] is 5D with Cartesian f
    sph_f = any(n in ("5D", "7F", "5D7F") for n in names)

    if "ATOMS" not in by:
        raise MoldenError(f"{path.name} has no [Atoms] section")
    hdr, body = by["ATOMS"]
    h = hdr.upper()
    mode = "fractional" if "FRACT" in h else ("angs" if "ANG" in h else "bohr")
    syms, zs, xyz = [], [], []
    for ln in body:
        p = ln.split()
        if len(p) < 6:
            continue
        syms.append(p[0])
        zs.append(int(float(p[2])))
        xyz.append([_num(v) for v in p[3:6]])
    if not syms:
        raise MoldenError(f"{path.name}: [Atoms] is empty")

    axes = None
    if "CELLAXES" in by:
        hdr, body = by["CELLAXES"]
        rows = [[_num(v) for v in ln.split()[:3]] for ln in body if len(ln.split()) >= 3]
        if len(rows) >= 3:
            axes = np.array(rows[:3], float)
            if "AU" in hdr.upper() or "BOHR" in hdr.upper():
                axes = axes * BOHR_ANG
    if mode == "fractional" and axes is None:
        raise MoldenError(f"{path.name}: fractional atoms but no [CellAxes]")

    if "GTO" not in by:
        raise MoldenError(f"{path.name} has no [GTO] section")
    raw_shells = _parse_gto(by["GTO"][1])
    shells = [Shell(a, l, e, c) for a, l, e, c in raw_shells]
    if any(s.atom >= len(syms) for s in shells):
        raise MoldenError(f"{path.name}: [GTO] names an atom that [Atoms] lacks")
    shells.sort(key=lambda s: s.atom)          # stable: keeps file order within an atom
    nbf = sum(_nfunc(s.l, sph_d, sph_f) for s in shells)

    if "MO" not in by:
        raise MoldenError(f"{path.name} has no [MO] section")
    ene, occ, spin, coef = _parse_mo(by["MO"][1], nbf)
    if coef.shape[1] == 0:
        raise MoldenError(f"{path.name}: [MO] holds no orbitals")

    return MoldenData(path=str(path), symbols=syms, Z=np.array(zs), coords_raw=np.array(xyz, float),
                      atoms_mode=mode, axes_ang=axes, shells=shells, spherical_d=sph_d,
                      spherical_f=sph_f, energies=ene, occupations=occ, spins=spin, coef=coef,
                      nbf=nbf, sections=names)
