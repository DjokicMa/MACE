"""One ORBITALS run: its Molden files, k points, levels, self-check and grids.

Typical use::

    run = OrbitalRun.from_path("si_orb_00001_K000_real.molden")   # or the .out, or the folder
    run.kpoints()                                  # k list with files and usability
    kd = run.load(1)                               # k index 1 (Γ); runs the self-check
    kd.levels("ALPHA+BETA")                        # energies, occupations, HOMO/LUMO, groups
    grid, data = run.grid(1, "ALPHA+BETA", [kd.homo()], part="re", cells=(2, 2, 2))
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .basis import Basis, bloch_overlap, orthonormality_error
from .grid import GridSpec, build_box, evaluate
from .molden import BOHR_ANG, MoldenData, MoldenError, read_molden
from .outfile import KPoint, OrbitalsOut, find_out_for, parse_out, split_name

HARTREE_EV = 27.211386245988
DEGENERACY_TOL = 1e-4      # Hartree; Ene is printed with 6 decimals
CHECK_OK = 1e-3            # max |C^H S C - 1| for a usable file
CHECK_FAIL = 1e-2          # above this the file is broken (CRYSTAL complex 2D/3D k, ILOC 1)

QUALITY_SPACING = {"preview": 0.40, "normal": 0.25, "fine": 0.15}   # bohr

PARTS = ("re", "im", "abs2")


class OrbitalsError(ValueError):
    """An ORBITALS file set or request MACE cannot use."""


@dataclass
class KEntry:
    index: int
    label: str
    frac: np.ndarray
    real: bool
    files: Dict[str, Path]               # "real" / "complex" -> path
    usable: bool = True
    reason: str = ""

    def to_dict(self) -> dict:
        return {"index": self.index, "label": self.label, "frac": [float(x) for x in self.frac],
                "real": self.real, "files": {k: v.name for k, v in self.files.items()},
                "usable": self.usable, "reason": self.reason}


@dataclass
class Level:
    index: int          # 1-based MO number within its spin channel
    column: int         # column in the file's [MO] list
    energy: float       # Hartree
    occupation: float
    group: int          # degenerate-set id within the channel


@dataclass
class KData:
    entry: KEntry
    md: MoldenData
    basis: Basis
    C: np.ndarray                         # (nbf, nmo) real or complex, gauge-fixed
    spins: List[str]
    check: Dict[str, float] = field(default_factory=dict)   # spin -> max |C^H S C - 1| over occupied
    status: str = "unchecked"             # ok / warn / fail / unchecked
    max_occ: float = 2.0
    efermi: Optional[float] = None        # Hartree, from the .out (metals)
    S: Optional[np.ndarray] = None        # S(k) of the self-check, reused to check each drawn orbital
    basis_norm_error: float = 0.0         # max |S_mu,mu(R=0) - 1| over the basis functions

    def drawn_check(self, cols: Sequence[int]) -> float:
        """max |C^H S C - 1| over exactly the columns about to be drawn (inf without S)."""
        if self.S is None:
            return float("nan")
        err = orthonormality_error(self.C[:, list(cols)], self.S)
        return err if np.isfinite(err) else float("inf")

    def channel(self, spin: str) -> List[int]:
        return [j for j, s in enumerate(self.md.spins) if s == spin]

    def levels(self, spin: str) -> List[Level]:
        cols = self.channel(spin)
        out, group = [], 0
        prev = None
        for n, j in enumerate(cols, start=1):
            e = float(self.md.energies[j])
            if prev is not None and abs(e - prev) > DEGENERACY_TOL:
                group += 1
            prev = e
            out.append(Level(n, j, e, float(self.md.occupations[j]), group))
        return out

    def homo(self, spin: Optional[str] = None) -> Optional[int]:
        """1-based index (in the channel) of the highest level with occupation above half."""
        spin = spin or self.spins[0]
        lv = self.levels(spin)
        if self.efermi is not None and self.metallic(spin):
            # CRYSTAL's Occup of a metal is not a 0..1 occupation (bcc Fe: up to 2.13 per spin
            # channel at Γ); the Fermi level from the .out decides what is occupied
            occ = [l for l in lv if l.energy <= self.efermi]
        else:
            occ = [l for l in lv if l.occupation > 0.5 * self.max_occ]
        return occ[-1].index if occ else None

    def metallic(self, spin: Optional[str] = None) -> bool:
        spin = spin or self.spins[0]
        return any(not (abs(l.occupation) < 0.02 * self.max_occ or abs(l.occupation - self.max_occ) < 0.02 * self.max_occ)
                   for l in self.levels(spin))

    def resolve(self, spin: str, mo) -> int:
        """'HOMO', 'LUMO', 'HOMO-1', 'LUMO+2' or a number -> 1-based index in the channel."""
        if isinstance(mo, (int, np.integer)):
            n = int(mo)
        else:
            s = str(mo).strip().upper()
            m = re.fullmatch(r"(HOMO|LUMO)\s*([+-]\s*\d+)?", s)
            if m:
                h = self.homo(spin)
                if h is None:
                    raise OrbitalsError(f"no occupied level in spin channel {spin}")
                n = h + (1 if m.group(1) == "LUMO" else 0) + (int(m.group(2).replace(" ", "")) if m.group(2) else 0)
            elif s.isdigit():
                n = int(s)
            else:
                raise OrbitalsError(f"cannot read orbital {mo!r}: use a number, HOMO, LUMO, HOMO-1, LUMO+1 ...")
        nch = len(self.channel(spin))
        if not 1 <= n <= nch:
            raise OrbitalsError(f"orbital {n} is outside 1..{nch} for spin {spin}")
        return n

    def degenerate_set(self, spin: str, n: int) -> List[int]:
        lv = self.levels(spin)
        g = lv[n - 1].group
        return [l.index for l in lv if l.group == g]

    def summary(self) -> dict:
        out = {"k": self.entry.to_dict(), "nbf": self.basis.nbf, "status": self.status,
               "check": {k: float(v) for k, v in self.check.items()}, "spins": []}
        for sp in self.spins:
            lv = self.levels(sp)
            h = self.homo(sp)
            out["spins"].append({
                "spin": sp, "homo": h, "lumo": (h + 1 if h is not None and h < len(lv) else None),
                "metallic": self.metallic(sp),
                "levels": [{"n": l.index, "e": l.energy, "e_ev": l.energy * HARTREE_EV,
                            "occ": l.occupation, "group": l.group} for l in lv],
            })
        return out


def _gauge_fix(C: np.ndarray) -> np.ndarray:
    """Make the largest coefficient of each orbital real and positive (repeatable +/- colors)."""
    C = C.copy()
    j = np.argmax(np.abs(C), axis=0)
    piv = C[j, np.arange(C.shape[1])]
    with np.errstate(invalid="ignore", divide="ignore"):
        ph = np.where(np.abs(piv) > 0, piv / np.abs(piv), 1.0)
    C = C / ph[None, :]
    if np.iscomplexobj(C) and np.abs(C.imag).max(initial=0.0) == 0.0:
        C = C.real
    return C


class OrbitalRun:
    """The Molden files of one ORBITALS run plus its .out (when found)."""

    def __init__(self, files: Sequence[Path], out: Optional[OrbitalsOut] = None, dimensionality: Optional[int] = None):
        self.files = sorted(Path(f) for f in files)
        if not self.files:
            raise OrbitalsError("no .molden files")
        self.out = out
        self.root = split_name(self.files[0].name)["root"] or self.files[0].stem
        self.molecule = all(split_name(f.name)["n"] is None for f in self.files)
        self._cache: Dict[int, KData] = {}
        self._dim = dimensionality
        self._entries = self._build_entries()

    # ---------------- discovery ----------------
    @classmethod
    def from_path(cls, path, out_path=None) -> "OrbitalRun":
        """A run from a .molden file, the ORBITALS .out or a folder; ``out_path`` overrides the .out search."""
        p = Path(path)
        if out_path is not None and p.suffix.lower() != ".out":
            run = cls.from_path(p)
            return cls(run.files, parse_out(out_path))
        if p.is_dir():
            mold = sorted(p.glob("*.molden"))
            if not mold:
                raise OrbitalsError(f"no .molden files in {p}")
            roots = {split_name(f.name)["root"] for f in mold}
            if len(roots) > 1:
                raise OrbitalsError(f"{p} holds several ORBITALS runs ({', '.join(sorted(r for r in roots if r))}); "
                                    "give one .molden file")
            return cls._from_molden(mold[0])
        if p.suffix.lower() == ".out":
            out = parse_out(p)
            files = [p.parent / n for n in out.files if (p.parent / n).exists()]
            if not files:
                # only this job's files: Si_orb*.molden would also take Si_orbitals_orb_*
                from mace.utils.calc_detection import is_orbitals_molden_name
                files = sorted(f for f in p.parent.glob(f"{p.stem}_orb*.molden")
                               if is_orbitals_molden_name(f.name, p.stem))
            if not files:
                raise OrbitalsError(f"{p.name} lists no Molden file that exists next to it"
                                    + (f" ({out.errors[0]})" if out.errors else ""))
            return cls(files, out)
        if p.suffix.lower() == ".molden":
            return cls._from_molden(p)
        raise OrbitalsError(f"{p.name}: give a .molden file, the ORBITALS .out or a folder")

    @classmethod
    def _from_molden(cls, p: Path) -> "OrbitalRun":
        if not p.exists():
            raise OrbitalsError(f"{p} does not exist")
        info = split_name(p.name)
        root = info["root"] or p.stem
        if info["n"] is None:
            files = [p]
        else:
            rx = re.compile(re.escape(root) + r"_\d{5}_K\d+_(real|complex)\.molden$")
            files = [f for f in p.parent.glob(f"{root}_*.molden") if rx.match(f.name)]
        outp = find_out_for(p)
        out = parse_out(outp) if outp else None
        return cls(files, out)

    def _build_entries(self) -> List[KEntry]:
        if self.molecule:
            return [KEntry(1, "molecule", np.zeros(3), True, {"real": self.files[0]})]
        groups: Dict[int, Dict[str, Path]] = {}
        labels: Dict[int, str] = {}
        for f in self.files:
            info = split_name(f.name)
            if info["n"] is None:
                continue
            groups.setdefault(info["n"], {})[info["part"]] = f
            labels[info["n"]] = info["label"]
        dim = self.dimensionality()
        entries = []
        for n in sorted(groups):
            parts = groups[n]
            kp = self.out.kpoint(n) if self.out else None
            if kp is not None:
                frac, real, label = kp.frac, kp.real, kp.label()
            else:
                lab = labels[n]
                real = "complex" not in parts
                if set(lab) == {"0"}:
                    frac, label = np.zeros(3), "Γ"
                else:
                    frac, label = None, f"K{lab}"
            e = KEntry(n, label, frac if frac is not None else np.full(3, np.nan), real, parts)
            if frac is None:
                e.usable, e.reason = False, "the k vector is unknown: the ORBITALS .out is not next to the files"
            elif not real and dim >= 2:
                e.usable = False
                e.reason = ("complex k point in a 2D/3D system: CRYSTAL23 writes coefficients that fail the "
                            "orthonormality check here, so the picture would be wrong")
            elif not real and "complex" not in parts:
                e.usable, e.reason = False, "the _complex file (imaginary part) is missing"
            entries.append(e)
        return entries

    # ---------------- info ----------------
    def dimensionality(self) -> int:
        if self._dim is None:
            if self.molecule:
                self._dim = 0
            else:
                md = read_molden(self.files[0])
                self._dim = md.dimensionality
        return self._dim

    def kpoints(self) -> List[KEntry]:
        return list(self._entries)

    def entry(self, index: int) -> KEntry:
        for e in self._entries:
            if e.index == index:
                return e
        raise OrbitalsError(f"no k point {index} in this run (have {[e.index for e in self._entries]})")

    def total_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.files if f.exists())

    # ---------------- loading ----------------
    def load(self, index: int = 1, check: bool = True, allow_unusable: bool = False) -> KData:
        if index in self._cache:
            return self._cache[index]
        e = self.entry(index)
        if not e.usable and not allow_unusable:
            raise OrbitalsError(f"k point {index} {e.label}: {e.reason}")
        md = read_molden(e.files["real"])
        C = md.coef.astype(float)
        if not e.real and "complex" in e.files:
            mi = read_molden(e.files["complex"])
            if mi.coef.shape != md.coef.shape:
                raise OrbitalsError(f"{e.files['complex'].name} does not match {e.files['real'].name}")
            C = C + 1j * mi.coef
        pos = self._internal_positions(md)
        basis = Basis(md, pos)
        spins = list(dict.fromkeys(md.spins))
        # molecules: 2.0 closed shell / 1.0 per spin; periodic files: 1.0 per band (polyyne's 1D files say 2.0)
        max_occ = 2.0 if (self.molecule and spins == ["ALPHA+BETA"]) else 1.0
        occs = md.occupations
        if np.all((np.abs(occs) < 1e-6) | (np.abs(occs - 2.0) < 1e-6)) and np.any(np.abs(occs - 2.0) < 1e-6):
            max_occ = 2.0
        efermi = self.out.efermi if self.out is not None else None
        kd = KData(e, md, basis, _gauge_fix(C), spins, max_occ=max_occ, efermi=efermi)
        if check:
            self._self_check(kd)
        self._cache[index] = kd
        return kd

    def _internal_positions(self, md: MoldenData) -> np.ndarray:
        """File positions moved to CRYSTAL's internal atom images (from the .out atom table).

        The file wraps atoms to [-0.5, 0.5); CRYSTAL's own image can differ by a
        lattice vector, which changes Bloch phases at zone-boundary k. The .out
        table has only 3 decimals, so it picks the lattice vector and the file
        keeps the precision.
        """
        pos = md.positions_bohr()
        per = md.periodic_dims
        if not per or self.out is None or self.out.atoms_ang is None:
            return pos
        ref = self.out.atoms_ang / BOHR_ANG
        if len(ref) != len(pos):
            return pos
        A = md.axes_ang / BOHR_ANG
        M = np.array([A[i] for i in per])                       # (nper, 3)
        out = pos.copy()
        for a in range(len(pos)):
            d = ref[a] - pos[a]
            f = np.linalg.lstsq(M.T, d, rcond=None)[0]
            n = np.round(f)
            resid = d - n @ M
            if np.linalg.norm(resid) < 0.05 / BOHR_ANG:       # 3-decimal table: 0.05 A tolerance
                out[a] = pos[a] + n @ M
        return out

    def _self_check(self, kd: KData) -> None:
        basis = kd.basis
        ns, SR = basis.overlap_cells()
        S = bloch_overlap(ns, SR, kd.entry.frac if basis.periodic else np.zeros(3))
        kd.S = S
        # every basis function must be normalized on site (catches a wrong d/f convention even
        # where those functions do not mix into the occupied orbitals)
        i0 = [i for i, n in enumerate(ns) if not np.any(n)]
        kd.basis_norm_error = float(np.abs(np.diag(SR[i0[0]]) - 1.0).max()) if i0 else 0.0
        worst = kd.basis_norm_error
        for sp in kd.spins:
            cols = kd.channel(sp)
            h = kd.homo(sp)
            occ = cols[:h] if h else []
            if not occ:
                occ = cols[: max(1, len(cols) // 2)]
            # occupied plus the first two virtuals (the LUMO is a default view)
            extra = [j for j in cols if j not in occ][:2]
            err = orthonormality_error(kd.C[:, occ + extra], S)
            kd.check[sp] = err
            worst = max(worst, err)
        if not np.isfinite(worst):
            worst = float("inf")
        kd.status = "ok" if worst < CHECK_OK else ("warn" if worst < CHECK_FAIL else "fail")

    # ---------------- grids ----------------
    def grid(self, index: int, spin: str, mos: Sequence, part: str = "re", cells=(1, 1, 1),
             quality: str = "normal", spacing: Optional[float] = None, degenerate_sum: bool = False,
             max_points: Optional[int] = 4_000_000, grid: Optional[GridSpec] = None,
             allow_failed: bool = False) -> Tuple[GridSpec, np.ndarray, dict]:
        """Values on a grid: (GridSpec, data (n0,n1,n2) float, meta).

        ``part``: "re" (Re psi, signed), "im" (Im psi, complex k only) or "abs2"
        (|psi|^2). Several ``mos`` (or ``degenerate_sum``) give the summed
        |psi|^2; a signed sum of degenerate partners is never formed (their
        mixing is arbitrary).
        """
        if part not in PARTS:
            raise OrbitalsError(f"part must be one of {', '.join(PARTS)}")
        kd = self.load(index)
        if kd.status == "fail" and not allow_failed:
            raise OrbitalsError(f"k point {kd.entry.label}: the file fails the orthonormality self-check "
                                f"(max |C^H S C - 1| = {max(kd.check.values()):.2g}); refusing to draw it")
        if spin not in kd.spins:
            alias = {"A": "ALPHA", "B": "BETA", "ALPHA": "ALPHA", "BETA": "BETA"}.get(str(spin).upper())
            if alias in kd.spins:
                spin = alias
            elif len(kd.spins) == 1:
                spin = kd.spins[0]
            else:
                raise OrbitalsError(f"spin {spin!r} not in {kd.spins}")
        ns = sorted({kd.resolve(spin, m) for m in mos})
        if degenerate_sum:
            ns = sorted({x for n in ns for x in kd.degenerate_set(spin, n)})
        if part == "im" and kd.entry.real:
            raise OrbitalsError("Im psi is zero at Γ and real k points; use re or abs2")
        summed = len(ns) > 1
        if summed and part != "abs2":
            part = "abs2"
        cols = [kd.channel(spin)[n - 1] for n in ns]
        # the load-time check covers the occupied orbitals plus two virtuals; check the
        # orbitals being drawn themselves (a high f-rich virtual can be wrong on its own)
        drawn = kd.drawn_check(cols)
        if np.isnan(drawn):
            drawn_status = kd.status
        else:
            drawn_status = "ok" if drawn < CHECK_OK else ("warn" if drawn < CHECK_FAIL else "fail")
        if drawn_status == "fail" and not allow_failed:
            raise OrbitalsError(f"k point {kd.entry.label}: orbital(s) {', '.join(map(str, ns))} fail the "
                                f"orthonormality self-check (max |C^H S C - 1| = {drawn:.2g}); refusing to draw")
        rank = {"ok": 0, "unchecked": 0, "warn": 1, "fail": 2}
        status = max((kd.status, drawn_status), key=lambda x: rank.get(x, 0))
        if grid is None:
            sp = spacing if spacing is not None else QUALITY_SPACING.get(quality, 0.25)
            cells = tuple(cells) if kd.basis.periodic else (1, 1, 1)
            grid = build_box(kd.basis, cells, sp, max_points=max_points)
        psi = evaluate(kd.basis, kd.C[:, cols], grid, kd.entry.frac if kd.basis.periodic else None)
        if part == "re":
            data = psi[0].real
        elif part == "im":
            data = psi[0].imag
        else:
            data = (np.abs(psi) ** 2).sum(axis=0)
        lv = kd.levels(spin)
        meta = {"k": kd.entry.to_dict(), "spin": spin, "mos": ns, "part": part,
                "energies": [lv[n - 1].energy for n in ns], "occupations": [lv[n - 1].occupation for n in ns],
                "signed": part in ("re", "im"), "status": status,
                "check": {k: float(v) for k, v in kd.check.items()},
                "check_drawn": None if np.isnan(drawn) else float(drawn),
                "homo": kd.homo(spin), "cells": list(cells), "degenerate_sum": bool(degenerate_sum)}
        return grid, np.ascontiguousarray(data), meta

    def atoms_bohr(self, index: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        kd = self.load(index, check=False, allow_unusable=True)
        return kd.md.Z.copy(), kd.basis.pos.copy()


def default_isovalue(data: np.ndarray, signed: bool, fraction: float = 0.9) -> float:
    """Isovalue enclosing ``fraction`` of sum |psi|^2 on the grid (psi or |psi|^2 data)."""
    w = np.abs(data).ravel() if not signed else (data.ravel() ** 2)
    v = np.abs(data).ravel()
    if w.sum() <= 0:
        return 0.0
    order = np.argsort(v)[::-1]
    cum = np.cumsum(w[order])
    i = int(np.searchsorted(cum, fraction * cum[-1]))
    return float(v[order[min(i, len(order) - 1)]])
