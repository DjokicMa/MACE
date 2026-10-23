"""Dielectric response of a FREQCALC run with INTENS + INTCPHF (CRYSTAL23 manual 9.1, 10.2): the
atomic Born charge tensors, eps_inf and eps_0, and the LO-TO splitting at Gamma.

Read from the output:

* ``ATOMIC BORN CHARGE TENSOR`` - Z*_ij of each atom (e) and its "dynamic charge" Tr(Z*)/3;
* ``BORN CHARGE VECTOR IN THE BASIS OF NORMAL MODES`` - Z_n,i (e m_e^-1/2) of each mode n;
* ``SUM TENSOR OF THE VIBRATIONAL CONTRIBUTIONS TO THE STATIC DIELECTRIC TENSOR`` - sum over modes
  of f_n,ij = 4 pi / V Z_n,i Z_n,j / nu_n^2;
* the mode table's frequencies (cm^-1; its EIGV = omega_n^2 in hartree^2 tells the real ones);
* eps_inf: the CPHF dielectric tensor of the run (analysis.parse's ``dielectric``).

Derived: eps_0 = eps_inf + the vibrational sum (and its principal values), and the LO frequencies
for a wave vector direction q: the eigenvalues of the dynamical matrix in the basis of the TO modes
with its non-analytic term, omega_n^2 delta_nm + 4 pi / V (Z_n . q)(Z_m . q) / (q . eps_inf . q)
(atomic units; Cochran & Cowley, J. Phys. Chem. Solids 23, 447 (1962); Gonze & Lee, PRB 55, 10355
(1997)). The runs in the corpus print no LO frequencies (CRYSTAL does with its own LO-TO options):
these are computed from what the run prints. The generalized Lyddane-Sachs-Teller relation,
prod(omega_LO^2) / prod(omega_TO^2) = q.eps_0.q / q.eps_inf.q over the polar modes, checks them.

Each LO goes with its own TO mode, not with the TO that happens to sort beside it: a mode with
Z_n . q = 0 (non-polar along q, as an A mode along x in a 6mm crystal) has no non-analytic term and
keeps its frequency, so it is left out of the matrix; the non-analytic term is of rank one, so the
polar modes' LO frequencies interlace their TO frequencies (TO_1 <= LO_1 <= TO_2 <= ...) and the
k-th LO goes with the k-th polar TO. Ag2Br2 (P6_3mc) along x: E1 136.3 -> 161.6 cm^-1, the A mode
at 147.0 unchanged (by sort order it read 136.3 -> 145.7 and 147.0 -> 161.6).

eps_0 is checked against the crystal family like eps_inf (optics.symmetry_mismatch).

Numpy only.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

from . import analysis as A
from . import tensor as T

from mace.constants import BOHR_TO_ANGSTROM as BOHR_ANG, HARTREE_TO_CM1 as HARTREE_CM  # noqa: E402
_BORN_HEAD = re.compile(r"^\s*ATOMIC BORN CHARGE TENSOR", re.M)
_ATOM = re.compile(r"^\s*ATOM\s+(\d+)\s+(\S+)\s+DYNAMIC CHARGE\s+(" + A._FLOAT + r")\s*$", re.M)
_MODE_VEC_HEAD = re.compile(r"^\s*BORN CHARGE VECTOR IN THE BASIS OF NORMAL MODES", re.M)
_SUM_HEAD = re.compile(r"^\s*SUM TENSOR OF THE VIBRATIONAL CONTRIBUTIONS TO THE STATIC DIELECTRIC", re.M)
_MODES_HEAD = re.compile(r"^\s*MODES\s+EIGV\s+FREQUENCIES", re.M)
_MODE_ROW = re.compile(r"^\s*(\d+)-\s*(\d+)\s+(" + A._FLOAT + r")\s+(" + A._FLOAT + r")\s+(" + A._FLOAT + r")\s+\((\S+)\s*\)")


def _rows3(lines: List[str], ncol: int = 3) -> List[List[float]]:
    out = []
    for ln in lines:
        v = ln.split()
        if len(v) == ncol + 1 and v[0].isdigit():
            try:
                out.append([A._f(x) for x in v[1:]])
            except ValueError:
                break
        elif out:
            break
    return out


def born_charges(text: str) -> List[Dict[str, Any]]:
    """[{atom, symbol, isotropic, tensor}] of the last ATOMIC BORN CHARGE TENSOR block."""
    m = A._last(_BORN_HEAD, text)
    if not m:
        return []
    end = text.find("+++", m.end())
    block = text[m.end():end if end > 0 else m.end() + 200000]
    atoms = list(_ATOM.finditer(block))
    out = []
    for i, a in enumerate(atoms):
        stop = atoms[i + 1].start() if i + 1 < len(atoms) else len(block)
        lines = block[a.end():stop].splitlines()
        t = _rows3(lines[next((k for k, ln in enumerate(lines) if ln.split() == ["1", "2", "3"]), 0) + 1:])
        if len(t) == 3:
            out.append({"atom": int(a.group(1)), "symbol": a.group(2).capitalize(), "isotropic": A._f(a.group(3)), "tensor": t})
    return out


def mode_table(text: str) -> List[Dict[str, Any]]:
    """Each mode of the last mode table: {mode, eigv_au, freq_cm, irrep}."""
    m = A._last(_MODES_HEAD, text)
    if not m:
        return []
    out = []
    for ln in text[m.end():].splitlines()[2:]:
        r = _MODE_ROW.match(ln)
        if not r:
            if out:
                break
            continue
        for k in range(int(r.group(1)), int(r.group(2)) + 1):
            out.append({"mode": k, "eigv_au": A._f(r.group(3)), "freq_cm": A._f(r.group(4)), "irrep": r.group(6)})
    return out


def mode_charges(text: str) -> Dict[int, List[float]]:
    m = A._last(_MODE_VEC_HEAD, text)
    if not m:
        return {}
    out = {}
    for ln in text[m.end():].splitlines()[2:]:
        v = ln.split()
        if len(v) == 4 and v[0].isdigit():
            out[int(v[0])] = [A._f(x) for x in v[1:]]
        elif out and ln.strip():
            break
    return out


def vibrational_sum(text: str) -> Optional[List[List[float]]]:
    m = A._last(_SUM_HEAD, text)
    if not m:
        return None
    rows = []
    for ln in text[m.end():].splitlines()[1:]:
        v = ln.split()
        if len(v) == 3:
            try:
                rows.append([A._f(x) for x in v])
            except ValueError:
                break
            if len(rows) == 3:
                return rows
        elif rows:
            break
    return None


#: |Z_n . q| below this fraction of the largest mode Born charge: non-polar along q (the corpus's
#: symmetry-forbidden components are 1e-17 or less of it, its weakest polar ones 1e-1)
POLAR_TOL = 1e-6


def lo_to(modes: List[Dict[str, Any]], zvec: Dict[int, List[float]], eps_inf, volume_bohr3: float, q) -> Dict[str, Any]:
    """TO and LO frequencies (cm^-1) of each mode for the wave vector direction q, mode by mode (to_cm,
    lo_cm, modes, irreps and polar aligned, ascending in TO): a non-polar mode keeps LO = TO."""
    import numpy as np
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    E = np.asarray(eps_inf, dtype=float)
    ks = sorted((m for m in modes if m["mode"] in zvec and m["eigv_au"] > 0 and m["freq_cm"] > 1.0),
                key=lambda m: (m["freq_cm"], m["mode"]))
    if not ks:
        return {"q": q.tolist(), "to_cm": [], "lo_cm": []}
    w2 = np.array([(m["freq_cm"] / HARTREE_CM) ** 2 for m in ks])     # the printed frequencies (EIGV has 4 digits)
    Zq = np.array([np.dot(zvec[m["mode"]], q) for m in ks])
    zmax = max(float(np.linalg.norm(zvec[m["mode"]])) for m in ks)
    polar = np.abs(Zq) > POLAR_TOL * zmax if zmax > 0 else np.zeros(len(ks), dtype=bool)
    lo2 = w2.copy()
    if polar.any():
        zp = Zq[polar]
        D = np.diag(w2[polar]) + 4 * math.pi / volume_bohr3 * np.outer(zp, zp) / float(q @ E @ q)
        lo2[polar] = np.sort(np.linalg.eigvalsh(D))           # interlacing: the k-th LO of the k-th polar TO
    lo = np.sqrt(np.clip(lo2, 0.0, None)) * HARTREE_CM
    to = np.sqrt(w2) * HARTREE_CM
    return {"q": [round(float(x), 6) + 0.0 for x in q], "to_cm": [float(x) for x in to], "lo_cm": [float(x) for x in lo],
            "modes": [m["mode"] for m in ks], "irreps": [m["irrep"] for m in ks], "polar": [bool(p) for p in polar]}


def dielectric_response(text: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Born charges, eps_inf, eps_0 and the LO-TO splitting along x, y, z and the principal axes of
    eps_inf ({} when the output has no Born charges)."""
    import numpy as np
    born = born_charges(text)
    if not born:
        return {}
    out: Dict[str, Any] = {"born_charges": born,
                           "charge_sum": [[round(sum(b["tensor"][i][j] for b in born), 6) + 0.0 for j in range(3)] for i in range(3)]}
    de = data.get("dielectric")
    vib = vibrational_sum(text)
    dim = data.get("dimensionality")
    if de and dim == 3:
        from .views import tensor3
        f0 = de["components"][0]["frequency_ev"]
        einf = np.array(tensor3([r for r in de["components"] if r["frequency_ev"] == f0], "epsilon"))
        out["epsilon_inf"] = einf.tolist()
        out["epsilon_inf_principal"] = [float(x) for x in np.linalg.eigvalsh(einf)]
        if vib:
            e0 = einf + np.array(vib)
            out.update(epsilon_vib=vib, epsilon_0=e0.tolist(), epsilon_0_principal=[float(x) for x in np.linalg.eigvalsh(e0)],
                       epsilon_0_iso=float(np.trace(e0)) / 3, epsilon_inf_iso=float(np.trace(einf)) / 3)
            from .optics import symmetry_mismatch
            mm = symmetry_mismatch(e0, data.get("crystal_family"))
            if mm:
                out["epsilon_0_symmetry_mismatch"] = mm
                out["warnings"] = [f"eps_0 is {mm['found']} (principal values "
                                   + ", ".join(f"{x:.4g}" for x in out["epsilon_0_principal"])
                                   + f") but the space group {data.get('space_group') or ''} is {data['crystal_family'].lower()}, "
                                   f"so it must be {mm['expected']}: eps_0 and the LO-TO splitting inherit the CPHF tensor's error"]
        modes, zvec = mode_table(text), mode_charges(text)
        vol = data.get("volume_ang3")
        if modes and zvec and vol:
            vb = vol / BOHR_ANG ** 3
            dirs = [("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1])]
            w, v = np.linalg.eigh(einf)
            for k in range(3):
                lab = T.direction_label(v[:, k], lattice=T.is_cubic(data))
                if all(abs(abs(float(np.dot(v[:, k], d))) - 1) > 1e-3 for _, d in dirs):
                    dirs.append((f"ε axis {lab}", v[:, k].tolist()))
            out["lo_to"] = []
            for name, d in dirs:
                r = lo_to(modes, zvec, einf, vb, d)
                r["direction"] = name
                if vib:
                    qq = np.asarray(r["q"])
                    r["lst_ratio_eps"] = float(qq @ (einf + np.array(vib)) @ qq / (qq @ einf @ qq))
                    if r["to_cm"]:
                        r["lst_ratio_freq"] = float(np.prod(np.square(r["lo_cm"])) / np.prod(np.square(r["to_cm"])))
                out["lo_to"].append(r)
            out["modes"] = [{**m, "born_vector": zvec.get(m["mode"])} for m in modes]
            out["lo_to_source"] = "computed"
    return out
