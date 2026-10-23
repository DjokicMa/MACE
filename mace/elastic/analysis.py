"""Read the tensors of an ELASTCON / ELAPIEZO / PIEZOCON / PIEZOCP /
PHOTOELA run and the fits of an EOS run, and derive the moduli.

What CRYSTAL23 prints (manual ch. 13, pp. 292-303, and ch. 12, p. 286):

* ``SYMMETRIZED ELASTIC CONSTANTS FOR <SYSTEM> CASE, IN GPa`` - the upper
  triangle of the 6x6 stiffness tensor C (Voigt order xx yy zz yz xz xy);
  1D and 2D runs give C in energy units (their volume is not defined);
* ``ELASTIC MODULI (COMPLIANCE TENSOR), IN TPa^-1`` - S = C^-1;
* ``SEISMIC VELOCITIES BY CHRISTOFFEL EQUATION (km/s)``;
* ``PIEZOELECTRIC CONSTANTS FOR <SYSTEM> CASE, IN C/m^2`` - direct e (3x6);
* ``PIEZOELECTRIC STRAIN (CONVERSE) CONSTANTS (pC/N = pm/V)`` - d (3x6);
* ``ELASTO-OPTIC (PHOTOELASTIC) POCKELS TENSOR`` - p (6x6, not symmetric);
* ``PIEZO-OPTIC TENSOR (IN BREWSTERS ...)`` - pi (6x6);
* ``SORTING VOLUMES/ENERGIES`` and ``FITTING USING ALL POINTS`` - EOS;
* ``THERMAL PROPERTIES FROM HELMHOLTZ FREE ENERGY`` and ``LINEAR THERMAL
  EXPANSION OF CONVENTIONAL LATTICE PARAMETERS`` - QHA (manual 9.3, p. 260).

Derived (3D, from C and S in GPa; Hill, Proc. Phys. Soc. A 65, 349 (1952)):

* Voigt  K_V = [(C11+C22+C33) + 2(C12+C13+C23)] / 9
         G_V = [(C11+C22+C33) - (C12+C13+C23) + 3(C44+C55+C66)] / 15
* Reuss  K_R = 1 / [(S11+S22+S33) + 2(S12+S13+S23)]
         G_R = 15 / [4(S11+S22+S33) - 4(S12+S13+S23) + 3(S44+S55+S66)]
* Hill   X_H = (X_V + X_R) / 2
* E = 9KG / (3K + G),  nu = (3K - 2G) / (2(3K + G))  (Hill moduli)
* universal anisotropy A^U = 5 G_V/G_R + K_V/K_R - 6
  (Ranganathan & Ostoja-Starzewski, PRL 101, 055504 (2008)); 0 = isotropic
* Born stability: C positive definite, all eigenvalues > 0
  (Mouhat & Coudert, PRB 90, 224104 (2014))
* Debye temperature from the mean sound velocity (Anderson, J. Phys.
  Chem. Solids 24, 909 (1963)): v_t = sqrt(G/rho), v_l = sqrt((K + 4G/3)/rho),
  v_m = [(2/v_t^3 + 1/v_l^3)/3]^(-1/3), theta_D = (h/k_B) (3n/(4 pi V))^(1/3) v_m

2D (a slab: xx, yy, xy): the layer moduli of the in-plane block,
E_x = (C11 C22 - C12^2)/C22, E_y = (C11 C22 - C12^2)/C11, nu_xy = C12/C22,
in N/m when the run gives C in hartree and the cell area is known
(C_2D = C / A). Stability: C11 > 0, C11 C22 > C12^2, C66 > 0.

Standard library and numpy only.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mace.constants import HARTREE_TO_J as HARTREE_J  # noqa: E402
H_PLANCK = 6.62607015e-34
K_BOLTZMANN = 1.380649e-23

_FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"
_FLOAT_RE = re.compile(_FLOAT)

_ELASTIC_HEAD = re.compile(r"^\s*(SYMMETRIZED\s+)?ELASTIC CONSTANTS(?:\s+FOR\s+(.+?)\s+CASE)?\s*,?\s*IN\s+(\S+)\s*$", re.M)
_COMPLIANCE_HEAD = re.compile(r"^\s*ELASTIC MODULI \(COMPLIANCE TENSOR\)\s*,?\s*IN\s+(\S+)\s*$", re.M)
_PIEZO_E_HEAD = re.compile(r"^\s*(SYMMETRIZED\s+)?PIEZOELECTRIC CONSTANTS(?:\s+FOR\s+(.+?)\s+CASE)?\s*,?\s*IN\s+(\S+)\s*$", re.M)
_PIEZO_D_HEAD = re.compile(r"^\s*PIEZOELECTRIC STRAIN \(CONVERSE\) CONSTANTS\s*\((.+?)\)\s*$", re.M)
_POCKELS_HEAD = re.compile(r"^\s*ELASTO-OPTIC \(PHOTOELASTIC\) POCKELS TENSOR\s*$", re.M)
_PIEZO_OPTIC_HEAD = re.compile(r"^\s*PIEZO-OPTIC TENSOR\s*(?:\((.+)\))?\s*$", re.M)
_VRH_HEAD = re.compile(r"ACCORDING TO VOIGT-REUSS-HILL\s*$", re.M)
_SEISMIC_HEAD = re.compile(r"^\s*SEISMIC VELOCITIES BY CHRISTOFFEL EQUATION", re.M)
_SEISMIC_ROW = re.compile(r"^\s*\[\s*(" + _FLOAT + r")\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s*\]\s+("
                          + _FLOAT + r")\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s*$")
_EOS_SORT_HEAD = re.compile(r"^\s*SORTING VOLUMES/ENERGIES\s*$", re.M)
_EOS_FIT_HEAD = re.compile(r"^\s*\++\s*FITTING USING ALL POINTS\s*\++\s*$", re.M)
_EOS_FIT_ROW = re.compile(r"^\s*(MURNAGHAN 1944|BIRCH-MURNAGHAN 1947|POIRIER-TARANTOLA 1998|VINET 1987)\s+("
                          + _FLOAT + r")\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")")
_VOLUME_DENSITY = re.compile(r"PRIMITIVE CELL - CENTRING CODE\s+\S+\s+VOLUME=\s*(" + _FLOAT + r")\s*-\s*DENSITY\s+("
                             + _FLOAT + r")\s*g/cm\^3")
_N_ATOMS = re.compile(r"N\. OF ATOMS PER CELL\s+(\d+)")
_CELL_HEAD = re.compile(r"^\s*A\s+B\s+C\s+ALPHA\s+BETA\s+GAMMA\s*$", re.M)

_DIELECTRIC_HEAD = re.compile(r"^\s*POLARIZABILITY \(ALPHA\), DIELECTRIC \(EPSILON\) AND FIRST-ORDER ELECTRIC\s*$", re.M)
_DIELECTRIC_ROW = re.compile(r"^\s*(" + _FLOAT + r")\s+([XYZ]{2})\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s+("
                             + _FLOAT + r")\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s*$")
# 0D-2D CPHF/KS (manual 10.2): the volume is not defined, so CRYSTAL prints the polarizability only
_POLARIZ_HEAD = re.compile(r"^\s*POLARIZABILITY TENSOR\s+ALPHA \(INDEPENDENT COMPONENTS\)\s*$", re.M)
_POLARIZ_ROW = re.compile(r"^\s*(" + _FLOAT + r")\s+([XYZ]{2})\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s*$")
_PRINCIPAL = re.compile(r"AA\s+(" + _FLOAT + r")\s+BB\s+(" + _FLOAT + r")\s+CC\s+(" + _FLOAT + r")")
_HYPER_HEADS = {"beta": re.compile(r"^\s*FIRST HYPERPOLARIZABILITY \(BETA\) AND SECOND ELECTRIC SUSCEPTIBILITY", re.M),
                "gamma": re.compile(r"^\s*SECOND HYPERPOLARIZABILITY \(GAMMA\) AND THIRD ELECTRIC SUSCEPTIBILITY", re.M)}
# Fortran's E format drops the E of a three-digit exponent: "-8.9595+158" is -8.9595E+158, a diverged
# response (real CRYSTAL23 output, test/FREQ Ag2Cl1Se1)
_HFLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+|[-+]\d{3})?"
_OVERFLOW = re.compile(r"^([-+]?(?:\d+\.\d*|\.\d+|\d+))([-+]\d{3})$")
_HYPER_ROW = re.compile(r"^\s*([XYZ]{3,4})\s+(" + _HFLOAT + r")\s+\(\s*(" + _HFLOAT + r")\s*\)\s+(" + _HFLOAT
                        + r")\s+(" + _HFLOAT + r")\s+(" + _HFLOAT + r")\s*$")
#: the independent (Kleinman) components CRYSTAL prints: XXX ... ZZZ, XXXX ... ZZZZ
HYPER_COMPONENTS = {"beta": 10, "gamma": 15}
#: |value| (a.u., symmetrized or not) above which the CPHF/CPKS response has diverged: converged runs
#: print beta below ~1e2 and gamma below ~1e5; the corpus's diverged ones 1e8 to 1e19, or overflow
HYPER_DIVERGED_AU = {"beta": 1e6, "gamma": 1e10}
_FAMILY = re.compile(r"^\s*CRYSTAL FAMILY\s*:\s*([A-Z]+)", re.M)
_SPACE_GROUP = re.compile(r"^\s*SPACE GROUP \([A-Z]+\)\s*:\s*(.+?)\s*$", re.M)
_CPHF_DIIS = re.compile(r"DIIS TEST:\s*(" + _FLOAT + r")\s+AT CPHF\s+CYCLE\s+\d+")
_CPHF_FIELD = re.compile(r"ELECTRIC FIELD APPLIED ALONG CARTESIAN DIRECTION\s+([XYZ])")
_CPHF_CYCLE = re.compile(r"^\s*CYCLE\s+(\d+)\s+ALPHA\s+\S+\s+EPSILON\s+\S+\s+DELTA\s+(\S+)", re.M)
_CPHF_MAXCYC = re.compile(r"MAX NUMBER OF CPHF/CPKS CYCLES\s+(\d+)")
#: the largest CPHF DIIS test of a stable run: the corpus's well-behaved runs stay below 10, the ones
#: whose tensors break the crystal symmetry or diverge reach 1e4 to 1e8
CPHF_DIIS_UNSTABLE = 1e3
_QHA_HEAD = re.compile(r"^\s*THERMAL PROPERTIES FROM HELMHOLTZ FREE ENERGY\s*$", re.M)
_QHA_LATTICE_HEAD = re.compile(r"^\s*LINEAR THERMAL EXPANSION OF CONVENTIONAL LATTICE PARAMETERS\s*$", re.M)
QHA_COLUMNS = ("temperature_k", "volume_ang3", "bulk_modulus_t_gpa", "alpha_v_per_k", "cp_minus_cv_j_mol_k",
               "bulk_modulus_s_gpa", "sp_minus_sv_j_mol_k")
QHA_LATTICE_COLUMNS = ("temperature_k", "a_ang", "alpha_a_per_k", "b_ang", "alpha_b_per_k", "c_ang", "alpha_c_per_k")

# The size and modulus units of a QHA (and its EOS fit) by dimensionality: CRYSTAL23 prints the
# length "L (Ang)" and "B (GPa x Ang^2)" for a polymer, the surface "S (Ang^2)" and "Bt (GPa x Ang)" for
# a slab (measured: trans-polyacetylene and h-BN of the 1.1.6 example suite); the keys stay
# volume_ang3 / bulk_modulus_*_gpa
SIZE_UNITS = {3: {"size": "V", "size_unit": "A^3", "modulus_unit": "GPa"},
              2: {"size": "S", "size_unit": "A^2", "modulus_unit": "GPa x A"},
              1: {"size": "L", "size_unit": "A", "modulus_unit": "GPa x A^2"}}

EOS_NAMES = {"MURNAGHAN 1944": "murnaghan", "BIRCH-MURNAGHAN 1947": "birch_murnaghan",
             "POIRIER-TARANTOLA 1998": "poirier_tarantola", "VINET 1987": "vinet"}


def _f(s: str) -> float:
    return float(s.replace("D", "E").replace("d", "e"))


def _fx(s: str) -> float:
    """_f, and Fortran's overflowed "-8.9595+158" (no E) as -8.9595E+158."""
    m = _OVERFLOW.match(s)
    return float(m.group(1) + "E" + m.group(2)) if m else _f(s)


def cphf_diagnostics(text: str) -> Dict[str, Any]:
    """The convergence of the first CPHF/CPKS run (the one before the first dielectric block), as
    CRYSTAL prints it: the cycles of each field direction, the cycle limit and the largest DIIS test.
    ``unstable``: the DIIS test went above CPHF_DIIS_UNSTABLE; ``unconverged``: the directions that hit
    the cycle limit. {} when the output prints no CPHF cycles."""
    end = _DIELECTRIC_HEAD.search(text)
    head = text[:end.start()] if end else text
    fields = list(_CPHF_FIELD.finditer(head))
    if not fields:
        return {}
    cycles: Dict[str, int] = {}
    for i, f in enumerate(fields):
        stop = fields[i + 1].start() if i + 1 < len(fields) else len(head)
        last = None
        for last in _CPHF_CYCLE.finditer(head, f.end(), stop):
            pass
        if last:
            cycles[f.group(1).lower()] = int(last.group(1))
    diis = [_f(m.group(1)) for m in _CPHF_DIIS.finditer(head, fields[0].start())]
    m = _CPHF_MAXCYC.search(text)
    out: Dict[str, Any] = {"cycles": cycles}
    if m:
        out["max_cycles"] = int(m.group(1))
        out["unconverged"] = [k for k, n in cycles.items() if n >= out["max_cycles"]]
    if diis:
        out["max_diis_test"] = max(diis)
        out["unstable"] = bool(out["max_diis_test"] > CPHF_DIIS_UNSTABLE)
    return out


def _bar_rows(text: str, start: int, max_rows: int = 8) -> List[List[float]]:
    """The ``| ... |`` rows after ``start`` (blank lines before them skipped)."""
    rows: List[List[float]] = []
    for line in text[start:].splitlines()[1:]:
        s = line.strip()
        if not s:
            if rows:
                break
            continue
        if not s.startswith("|"):
            if rows:
                break
            continue
        inner = s.strip("|")
        rows.append([_f(x) for x in _FLOAT_RE.findall(inner)])
        if len(rows) >= max_rows:
            break
    return rows


def _square(rows: List[List[float]]) -> Optional[List[List[float]]]:
    """A full matrix from upper-triangle rows (n, n-1, ..., 1 values) or full rows."""
    if not rows:
        return None
    n = len(rows[0])
    if n == 0:
        return None
    if all(len(r) == n for r in rows):
        return [list(r) for r in rows]
    if len(rows) == n and all(len(r) == n - i for i, r in enumerate(rows)):
        m = [[0.0] * n for _ in range(n)]
        for i, r in enumerate(rows):
            for k, v in enumerate(r):
                j = i + k
                m[i][j] = v
                m[j][i] = v
        return m
    return None


def _last(regex, text: str):
    found = None
    for found in regex.finditer(text):
        pass
    return found


def _number_table(text: str, start: int, ncol: int, max_rows: int = 2000) -> List[List[float]]:
    """Rows of exactly ``ncol`` numbers after ``start`` (headers and blank lines
    before them skipped; the table ends at the first other line after rows)."""
    rows: List[List[float]] = []
    for line in text[start:].splitlines()[1:]:
        vals = line.split()
        if len(vals) == ncol:
            try:
                rows.append([_f(v) for v in vals])
                if len(rows) >= max_rows:
                    break
                continue
            except ValueError:
                pass
        if rows and line.strip():
            break
    return rows


def _first_cell(text: str) -> Optional[Tuple[float, ...]]:
    """(a, b, c, alpha, beta, gamma) of the first printed primitive cell (the input geometry)."""
    for m in _CELL_HEAD.finditer(text):
        nxt = text[m.end():].lstrip("\n").splitlines()
        if nxt:
            vals = _FLOAT_RE.findall(nxt[0])
            if len(vals) == 6:
                return tuple(_f(v) for v in vals)
    return None


def dimensionality(text: str) -> Optional[int]:
    for marker, dim in ((" CRYSTAL CALCULATION", 3), (" SLAB CALCULATION", 2),
                        (" POLYMER CALCULATION", 1), (" MOLECULAR CALCULATION", 0)):
        if re.search(r"^" + marker + r"\s*$", text, re.M):
            return dim
    return None


def parse(text: str) -> Dict[str, Any]:
    """Every tensor and EOS table in a CRYSTAL output ({} when there is none).

    Keys (each only when printed): ``elastic`` {matrix, unit, system,
    symmetrized}, ``compliance`` {matrix, unit}, ``seismic`` [{direction,
    vp, vs1, vs2}], ``piezo_e`` {matrix, unit, system}, ``piezo_d`` {matrix,
    unit}, ``pockels`` {matrix}, ``piezo_optic`` {matrix, unit}, ``eos``
    {points: [[V, E]], fits: {name: {volume_ang3, energy_au, bulk_modulus_gpa,
    bulk_modulus_prime}}}, ``qha`` {temperature_k, volume_ang3, bulk_modulus_t_gpa,
    alpha_v_per_k, ..., lattice}, plus ``dimensionality``, ``volume_ang3``,
    ``density_g_cm3``, ``n_atoms``, ``cell`` of the input geometry.
    """
    out: Dict[str, Any] = {}
    full = text
    # PRINT adds the fits of every number of points and order after "FINAL
    # RESULTS END" (each with its own compliance block): the results are the
    # tables before that line (real CRYSTAL23 ELASTCON output)
    cut = text.find("FINAL RESULTS END")
    if cut > 0:
        text = text[:cut]
    m = _last(_ELASTIC_HEAD, text)
    el_end = 0
    if m:
        mat = _square(_bar_rows(text, m.start()))
        if mat:
            out["elastic"] = {"matrix": mat, "unit": m.group(3), "system": (m.group(2) or "").strip() or None,
                              "symmetrized": bool(m.group(1))}
            el_end = m.end()
    m = None
    for m in _COMPLIANCE_HEAD.finditer(text, el_end):
        break                                   # the compliance that follows the constants
    if m:
        mat = _square(_bar_rows(text, m.start()))
        if mat:
            out["compliance"] = {"matrix": mat, "unit": m.group(1)}
    m = _last(_VRH_HEAD, text)
    if m:
        for line in text[m.end():].splitlines()[1:8]:
            vals = _FLOAT_RE.findall(line)
            if len(vals) == 8:
                out["crystal_vrh"] = dict(zip(("K_V", "G_V", "K_R", "G_R", "K_H", "G_H", "E_H", "v_H"),
                                              (_f(v) for v in vals)))
                break
    m = _last(_PIEZO_E_HEAD, text)
    if m:
        rows = _bar_rows(text, m.start(), 3)
        if rows and all(len(r) == len(rows[0]) for r in rows):
            out["piezo_e"] = {"matrix": rows, "unit": m.group(3), "system": (m.group(2) or "").strip() or None}
    m = _last(_PIEZO_D_HEAD, text)
    if m:
        rows = _bar_rows(text, m.start(), 3)
        if rows and all(len(r) == len(rows[0]) for r in rows):
            out["piezo_d"] = {"matrix": rows, "unit": m.group(1).strip()}
    m = _last(_POCKELS_HEAD, text)
    if m:
        rows = _bar_rows(text, m.start(), 6)
        if rows and all(len(r) == len(rows[0]) for r in rows):
            out["pockels"] = {"matrix": rows, "unit": "dimensionless"}
    m = _last(_PIEZO_OPTIC_HEAD, text)
    if m:
        rows = _bar_rows(text, m.start(), 6)
        if rows and all(len(r) == len(rows[0]) for r in rows):
            out["piezo_optic"] = {"matrix": rows, "unit": "Brewster (TPa^-1)"}
    m = _last(_SEISMIC_HEAD, text)
    if m:
        vel = []
        started = False
        for line in text[m.end():].splitlines()[1:]:
            r = _SEISMIC_ROW.match(line)
            if r:
                started = True
                v = [_f(x) for x in r.groups()]
                vel.append({"direction": v[:3], "vp": v[3], "vs1": v[4], "vs2": v[5]})
            elif started and line.strip() and not line.strip().isdigit():
                # a page number or blank line inside the table is skipped
                if not re.fullmatch(r"\s*\d+\s*", line):
                    break
        if vel:
            out["seismic"] = vel
    eos: Dict[str, Any] = {}
    m = _last(_EOS_SORT_HEAD, text)
    if m:
        pts = []
        for line in text[m.end():].splitlines()[1:]:
            vals = _FLOAT_RE.findall(line)
            if len(vals) == 2 and "." in vals[0]:
                pts.append([_f(vals[0]), _f(vals[1])])
            elif pts and line.strip():
                break
        if pts:
            eos["points"] = pts
    m = _last(_EOS_FIT_HEAD, text)
    if m:
        fits = {}
        # the table follows the module's citation lines (real CRYSTAL23 output)
        for line in text[m.end():].splitlines()[1:40]:
            if "POLYNOMIAL FITTING" in line or "THERMODYNAMIC FUNCTIONS" in line:
                break
            r = _EOS_FIT_ROW.match(line)
            if r:
                fits[EOS_NAMES[r.group(1)]] = {"volume_ang3": _f(r.group(2)), "energy_au": _f(r.group(3)),
                                               "bulk_modulus_gpa": _f(r.group(4)),
                                               "bulk_modulus_prime": _f(r.group(5))}
        if fits:
            eos["fits"] = fits
    if eos:
        out["eos"] = eos
    # CPHF/KS (manual 10.2): the first polarizability / dielectric block is
    # the input geometry's (a PHOTOELA run prints one per strained cell after it)
    m = _DIELECTRIC_HEAD.search(full)
    if m:
        rows = []
        for line in full[m.end():].splitlines()[1:40]:
            r = _DIELECTRIC_ROW.match(line)
            if r:
                v = [_f(x) for x in (r.group(1), *r.group(3, 4, 5, 6, 7))]
                rows.append({"frequency_ev": v[0], "component": r.group(2), "alpha": v[1], "alpha_im": v[2],
                             "epsilon": v[3], "epsilon_im": v[4], "chi1": v[5]})
            elif rows and line.strip():
                break
        if rows:
            diel: Dict[str, Any] = {"components": rows, "units": {"alpha": "bohr^3", "epsilon": "dimensionless"}}
            tail = full[m.end():m.end() + 4000]
            pa = tail.find("TENSOR IN PRINCIPAL AXES SYSTEM")
            if pa >= 0:
                p = _PRINCIPAL.search(tail, pa)
                if p:
                    diel["principal_epsilon"] = [_f(x) for x in p.groups()]
            ri = tail.find("REFRACTIVE INDICES")
            if ri >= 0:
                p = _PRINCIPAL.search(tail, ri)
                if p:
                    diel["refractive_indices"] = [_f(x) for x in p.groups()]
            out["dielectric"] = diel
    m = None if "dielectric" in out else _POLARIZ_HEAD.search(full)
    if m:
        rows = []
        for line in full[m.end():].splitlines()[1:40]:
            r = _POLARIZ_ROW.match(line)
            if r:
                rows.append({"frequency_ev": _f(r.group(1)), "component": r.group(2), "alpha": _f(r.group(3)),
                             "alpha_im": _f(r.group(4))})
            elif rows and line.strip():
                break
        if rows:
            out["polarizability"] = {"components": rows, "units": {"alpha": "bohr^3"}}
    for name, head in _HYPER_HEADS.items():
        m = _last(head, full)
        if not m:
            continue
        rows = []
        overflow = False
        for line in full[m.end():].splitlines()[1:120]:
            r = _HYPER_ROW.match(line)
            if r:
                overflow = overflow or any(_OVERFLOW.match(x) for x in r.groups()[1:])
                v = [_fx(x) for x in r.groups()[1:]]
                v = [x if math.isfinite(x) else None for x in v]
                rows.append({"component": r.group(1), "value": v[0], "unsymmetrized": v[1], "chi": v[2],
                             "mks": v[3], "cgs": v[4]})
            elif rows and line.strip() and not line.strip().startswith("*"):
                break
        if rows:
            # a conclusion (zero: centrosymmetric) needs every independent component, each finite and
            # converged: a row CRYSTAL printed as an overflow is a diverged response, never a zero
            sizes = [abs(x) for r in rows for x in (r["value"], r["unsymmetrized"]) if x is not None]
            largest = max(sizes) if sizes else None
            diverged = bool(overflow or any(r["value"] is None or r["unsymmetrized"] is None for r in rows)
                            or (largest is not None and largest >= HYPER_DIVERGED_AU[name]))
            out[name] = {"components": rows, "n_independent": HYPER_COMPONENTS[name],
                         "complete": len({r["component"] for r in rows}) == HYPER_COMPONENTS[name],
                         "diverged": diverged, "overflow": overflow, "largest_au": largest,
                         "units": ({"value": "a.u.", "chi": "a.u.", "mks": "pm/V (d, SHG)", "cgs": "1e-7 esu"}
                                   if name == "beta" else
                                   {"value": "a.u.", "chi": "a.u.", "mks": "1e-22 m^2/V^2 (THG)", "cgs": "esu"})}
    # QHA (manual 9.3): the zero-pressure table is the first one
    m = _QHA_HEAD.search(text)
    if m:
        rows = _number_table(text, m.end(), len(QHA_COLUMNS))
        if rows:
            qha: Dict[str, Any] = {k: [r[i] for r in rows] for i, k in enumerate(QHA_COLUMNS)}
            ml = _QHA_LATTICE_HEAD.search(text, m.end())
            if ml:
                lrows = _number_table(text, ml.end(), len(QHA_LATTICE_COLUMNS))
                if lrows:
                    qha["lattice"] = {k: [r[i] for r in lrows] for i, k in enumerate(QHA_LATTICE_COLUMNS)}
            out["qha"] = qha
    if not out:
        return out
    text = full
    out["dimensionality"] = dimensionality(text)
    m = _FAMILY.search(text)
    if m:
        out["crystal_family"] = m.group(1)
    m = _SPACE_GROUP.search(text)
    if m:
        out["space_group"] = m.group(1)
    for key in ("qha", "eos"):
        if key in out and out["dimensionality"] in SIZE_UNITS:
            out[key]["units"] = dict(SIZE_UNITS[out["dimensionality"]])
    m = _VOLUME_DENSITY.search(text)          # the first: the input (equilibrium) cell
    if m:
        out["volume_ang3"] = _f(m.group(1))
        out["density_g_cm3"] = _f(m.group(2))
    m = _N_ATOMS.search(text)
    if m:
        out["n_atoms"] = int(m.group(1))
    cell = _first_cell(text)
    if cell:
        out["cell"] = list(cell)
    comp = composition(text)
    if comp:
        out.update(comp)
    return out


_ATOM_ROW = re.compile(r"^\s*\d+\s+[TF]\s+(\d+)\s+([A-Za-z]{1,2})\s+" + _FLOAT + r"\s+" + _FLOAT + r"\s+" + _FLOAT + r"\s*$")


def composition(text: str) -> Dict[str, Any]:
    """The input cell's composition from its first atom table (ATOMS IN THE UNIT CELL, every atom of
    the primitive cell): {formula, formula_units, composition}; {} when the output has none."""
    m = re.search(r"ATOMS IN THE ASYMMETRIC UNIT\s+\d+\s+-\s+ATOMS IN THE UNIT CELL:\s+(\d+)", text)
    if not m:
        return {}
    n = int(m.group(1))
    counts: Dict[str, int] = {}
    got = 0
    for line in text[m.end():].splitlines()[1:n + 6]:
        r = _ATOM_ROW.match(line)
        if r:
            el = r.group(2).capitalize()
            counts[el] = counts.get(el, 0) + 1
            got += 1
            if got == n:
                break
    if got != n or not counts:
        return {}
    from mace.utils.formula import reduced_formula
    formula, z = reduced_formula(counts)
    return {"formula": formula, "formula_units": z, "composition": counts}


# ------------------------------------------------------------------ moduli
def _inv(m: Sequence[Sequence[float]]):
    import numpy as np
    return np.linalg.inv(np.array(m, dtype=float))


def moduli_3d(c: Sequence[Sequence[float]], s: Optional[Sequence[Sequence[float]]] = None,
              density_g_cm3: Optional[float] = None, n_atoms: Optional[int] = None,
              volume_ang3: Optional[float] = None) -> Dict[str, Any]:
    """Voigt-Reuss-Hill moduli of a 3D stiffness tensor C in GPa.

    ``s``: the compliance in GPa^-1 (default C^-1). Density, atoms per
    cell and cell volume give the sound velocities and the Debye
    temperature.
    """
    import numpy as np
    C = np.array(c, dtype=float)
    C = (C + C.T) / 2.0
    S = np.array(s, dtype=float) if s is not None else np.linalg.inv(C)
    eig = np.linalg.eigvalsh(C)
    kv = ((C[0, 0] + C[1, 1] + C[2, 2]) + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9.0
    gv = ((C[0, 0] + C[1, 1] + C[2, 2]) - (C[0, 1] + C[0, 2] + C[1, 2]) + 3 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15.0
    kr = 1.0 / ((S[0, 0] + S[1, 1] + S[2, 2]) + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
    gr = 15.0 / (4 * (S[0, 0] + S[1, 1] + S[2, 2]) - 4 * (S[0, 1] + S[0, 2] + S[1, 2]) + 3 * (S[3, 3] + S[4, 4] + S[5, 5]))
    kh, gh = (kv + kr) / 2.0, (gv + gr) / 2.0
    out: Dict[str, Any] = {
        "bulk_modulus_voigt_gpa": kv, "bulk_modulus_reuss_gpa": kr, "bulk_modulus_hill_gpa": kh,
        "shear_modulus_voigt_gpa": gv, "shear_modulus_reuss_gpa": gr, "shear_modulus_hill_gpa": gh,
        "eigenvalues_gpa": [float(x) for x in eig],
        "min_eigenvalue_gpa": float(eig.min()),
        "born_stable": bool(eig.min() > 0),
    }
    if 3 * kh + gh != 0:
        out["youngs_modulus_gpa"] = 9 * kh * gh / (3 * kh + gh)
        out["poisson_ratio"] = (3 * kh - 2 * gh) / (2 * (3 * kh + gh))
    if gh:
        out["pugh_ratio"] = kh / gh
    if gr and kr:
        out["universal_anisotropy"] = 5 * gv / gr + kv / kr - 6
    # directional Young's moduli along x, y, z: 1/S11, 1/S22, 1/S33
    out["youngs_modulus_xyz_gpa"] = [float(1.0 / S[i, i]) if S[i, i] else None for i in range(3)]
    if density_g_cm3 and gh > 0 and kh > 0:
        rho = density_g_cm3 * 1000.0                         # kg/m^3
        vt = math.sqrt(gh * 1e9 / rho)
        vl = math.sqrt((kh + 4.0 * gh / 3.0) * 1e9 / rho)
        vm = (((2.0 / vt ** 3) + (1.0 / vl ** 3)) / 3.0) ** (-1.0 / 3.0)
        out.update(sound_velocity_transverse_km_s=vt / 1000.0, sound_velocity_longitudinal_km_s=vl / 1000.0,
                   sound_velocity_mean_km_s=vm / 1000.0)
        if n_atoms and volume_ang3:
            n_per_m3 = n_atoms / (volume_ang3 * 1e-30)
            out["debye_temperature_k"] = (H_PLANCK / K_BOLTZMANN) * (3.0 * n_per_m3 / (4.0 * math.pi)) ** (1.0 / 3.0) * vm
    return {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in out.items()}


def moduli_2d(c: Sequence[Sequence[float]], unit: str, area_ang2: Optional[float] = None) -> Dict[str, Any]:
    """In-plane moduli of a slab from the xx, yy, xy block of C."""
    import numpy as np
    C = np.array(c, dtype=float)
    if C.shape == (6, 6):
        idx = [0, 1, 5]
        C = C[np.ix_(idx, idx)]
    C = (C + C.T) / 2.0
    scale, out_unit = 1.0, unit
    if str(unit).upper().startswith("HARTREE") and area_ang2:
        scale, out_unit = HARTREE_J / (area_ang2 * 1e-20), "N/m"
    C2 = C * scale
    c11, c22, c12, c66 = C2[0, 0], C2[1, 1], C2[0, 1], C2[2, 2]
    out: Dict[str, Any] = {"unit": out_unit, "c2d": C2.tolist(),
                           "born_stable": bool(c11 > 0 and c11 * c22 > c12 ** 2 and c66 > 0)}
    if c22:
        out["youngs_modulus_x"] = (c11 * c22 - c12 ** 2) / c22
        out["poisson_ratio_xy"] = c12 / c22
    if c11:
        out["youngs_modulus_y"] = (c11 * c22 - c12 ** 2) / c11
    out["shear_modulus_xy"] = c66
    out["layer_modulus"] = (c11 + c22 + 2 * c12) / 4.0
    return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in out.items()}


def analyze(text: str, epsilon=None) -> Dict[str, Any]:
    """parse() plus the derived moduli (``moduli``) for the run's periodicity and the directional
    properties, anisotropy indices and estimates of ``tensor.derived`` (``derived``). ``epsilon``: the
    relative static permittivity for the piezoelectric coupling (one, three or nine values).
    ``warnings``: what makes results unphysical (warnings()), shown by the report and the GUI."""
    data = _analyze(text, epsilon)
    w = warnings(data)
    if w:
        data["warnings"] = w
    return data


def warnings(data: Dict[str, Any]) -> List[str]:
    """The results analyze() flags: a mechanically unstable C, a dielectric tensor that breaks the
    crystal symmetry or has no real index, an unstable CPHF, a diverged or incomplete beta/gamma."""
    out: List[str] = []
    mod = data.get("moduli") or {}
    if mod.get("born_stable") is False:
        low = f" (smallest eigenvalue of C {mod['min_eigenvalue_gpa']:.4g} GPa)" if "min_eigenvalue_gpa" in mod else ""
        out.append(f"C is not positive definite{low}: the crystal is mechanically unstable, so the moduli derived from it "
                   "(directional extremes, sound velocities, anisotropy, Pugh, hardness, thermal conductivity, piezoelectric "
                   "coupling) are unphysical and are not stored in the database")
    out += (data.get("optics") or {}).get("warnings") or []
    out += (data.get("born") or {}).get("warnings") or []
    for name, title in (("beta", "first hyperpolarizability beta"), ("gamma", "second hyperpolarizability gamma")):
        h = data.get(name)
        if not h:
            continue
        if h.get("diverged"):
            out.append(f"The {title} diverged (" + ("CRYSTAL printed overflowed values" if h.get("overflow") else
                                                    f"values up to {h['largest_au']:.3g} a.u.")
                       + "): no SHG analysis and no conclusion about inversion symmetry")
        elif h.get("complete") is False:
            out.append(f"The {title} table is incomplete ({len(h['components'])} of {h['n_independent']} independent "
                       "components read): no conclusion is drawn from it")
    return out


def _analyze(text: str, epsilon=None) -> Dict[str, Any]:
    data = parse(text)
    if data.get("qha"):
        from . import qha as Q                     # the Grueneisen table, F, S, C_v, C_p along V(T)
        data["qha"].update(Q.summary(text, data["qha"]))
    if "ATOMIC BORN CHARGE TENSOR" in text and data:
        from . import born as B                    # Born charges, eps_0, LO-TO of a FREQ run with INTCPHF
        resp = B.dielectric_response(text, data)
        if resp:
            data["born"] = resp
    if data.get("dielectric") or data.get("beta"):
        from . import optics as O                  # principal axes, n(omega), SHG d_eff
        opt = O.summary(text, data)
        if opt:
            data["optics"] = opt
    eos = data.get("eos") or {}
    if eos.get("fits") and len(eos.get("points") or []) > 1:
        from .views import eos_quality             # each fit's residuals from the computed points
        eos["quality"] = eos_quality(eos)
    el = data.get("elastic")
    if not el:
        if data.get("piezo_e") or data.get("piezo_d"):
            _derive(data, epsilon)
        return data
    dim = data.get("dimensionality") or 3
    try:
        if dim == 3 and len(el["matrix"]) == 6:
            # S = C^-1 from the constants (3 decimals in GPa), not the printed
            # compliance (4 decimals in TPa^-1, too coarse for the Reuss bound)
            data["moduli"] = moduli_3d(el["matrix"], None, data.get("density_g_cm3"), data.get("n_atoms"),
                                       data.get("volume_ang3"))
        elif dim == 2:
            cell = data.get("cell")
            area = None
            if cell:
                a, b, _c, _al, _be, ga = cell
                area = a * b * math.sin(math.radians(ga))
            data["moduli"] = moduli_2d(el["matrix"], el.get("unit", ""), area)
    except Exception as exc:                 # a singular tensor (an unconverged run)
        data["moduli_error"] = f"{type(exc).__name__}: {exc}"
        return data
    _derive(data, epsilon)
    return data


def _derive(data: Dict[str, Any], epsilon=None) -> None:
    from . import tensor as T
    try:
        der = T.derived(data, epsilon)
    except ValueError:
        raise                                 # a bad --epsilon
    except Exception as exc:                 # a tensor the directional analysis cannot use
        data["derived_error"] = f"{type(exc).__name__}: {exc}"
        return
    if der:
        data["derived"] = der


def analyze_file(path, epsilon=None) -> Dict[str, Any]:
    return analyze(Path(path).read_text(errors="replace"), epsilon)


# ------------------------------------------------------------------ report
def _ext(e: Dict[str, Any], unit: str = "", digits: int = 4) -> str:
    """'min V along [..] to max V along [..]' of tensor.extremes (with the second direction when there is one)."""
    if not e or not e.get("min"):
        return "-"
    u = f" {unit}" if unit else ""

    def val(v):                               # None: infinite (1/0 of a singular or unstable tensor)
        return f"{v:.{digits}g}" if v is not None else "undefined"

    def one(x):
        return f"{val(x['value'])}{u} along {x['label']}" + (f", m {x['label2']}" if x.get("label2") else "")
    if e.get("isotropic"):
        return f"{val(e['max']['value'])}{u} along every direction"
    return f"{one(e['min'])} to {one(e['max'])}"


def _derived_lines(der: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    if der.get("unstable"):
        lines.append("UNPHYSICAL (C not positive definite): the derived numbers below are those of an unstable tensor")
    d = der.get("directional") or {}
    if d:
        lines.append("Directional (min to max over directions; G and nu over every m perpendicular to n)")
        for k, title in (("youngs", "Young's modulus E(n)"), ("compressibility", "Linear compressibility beta(n)"),
                         ("shear", "Shear modulus G(n, m)"), ("poisson", "Poisson's ratio nu(n, m)")):
            if d.get(k):
                lines.append(f"  {title}: {_ext(d[k], d[k].get('unit', ''))}")
        for name, s in (d.get("sound") or {}).items():
            lines.append(f"  Sound velocity {name}: phase {_ext(s, 'km/s')}; group {_ext(s.get('group') or {}, 'km/s')}")
    idx = der.get("indices") or {}
    if idx:
        parts = [f"A^U {idx['universal']:.4f}" if idx.get("universal") is not None else "",
                 f"Zener A {idx['zener']:.4f}" if idx.get("zener") is not None else "",
                 f"Chung-Buessem A_C {idx['chung_buessem']:.4f}" if idx.get("chung_buessem") is not None else ""]
        for k, t in (("e_ratio", "E_max/E_min"), ("g_ratio", "G_max/G_min"), ("nu_ratio", "nu_max/nu_min")):
            if idx.get(k) is not None:
                parts.append(f"{t} {idx[k]:.4g}")
        lines.append("Anisotropy: " + ", ".join(p for p in parts if p))
    ch = der.get("character") or {}
    if ch.get("pugh_ratio") is not None:
        lines.append(f"Pugh's ratio K/G {ch['pugh_ratio']:.3f} ({'ductile' if ch['ductile'] else 'brittle'}; "
                     f"ductile above {ch['pugh_threshold']})")
    if ch.get("cauchy"):
        lines.append("Cauchy relations (GPa): " + ", ".join(f"{c['relation'].replace('−', '-')} = {c['value']:.2f}" for c in ch["cauchy"]))
    hv = ch.get("hardness") or {}
    if hv.get("chen_gpa") is not None:
        chen = (f"Chen 0 GPa (its formula gives {hv['chen_formula_gpa']:.2f} GPa: a soft solid outside the fit)"
                if hv.get("chen_clamped") else f"Chen {hv['chen_gpa']:.1f} GPa")
        lines.append(f"Vickers hardness: {chen} ({hv['references']['chen']}); "
                     f"Tian {hv['tian_gpa']:.1f} GPa ({hv['references']['tian']})")
    th = der.get("thermal") or {}
    if th.get("clarke_w_mk") is not None:
        lines.append(f"Minimum thermal conductivity: Clarke {th['clarke_w_mk']:.3f} W/(m K), "
                     f"Cahill {th['cahill_w_mk']:.3f} W/(m K)")
    pz = der.get("piezo") or {}
    if pz.get("longitudinal") and not pz.get("zero"):
        k = pz["kind"]
        mx = pz["longitudinal"]["max"]
        lines.append(f"Longitudinal piezoelectric {k}_L(n): largest |{k}_L| {abs(mx['value']):.4g} {pz['unit']} along {mx['label']} "
                     f"({k}_L there {mx.get('signed', mx['value']):.4g}); along x, y, z: "
                     + ", ".join(f"{x:.4g}" for x in pz.get("axes") or []))
        cp = pz.get("coupling") or {}
        if cp.get("available"):
            lines.append(f"  Coupling k^2 along x, y, z: {', '.join(f'{x:.4g}' for x in cp['k2_axes'])}; largest "
                         f"{cp['k2_max']['value']:.4g} (k {cp['k_max']:.4g}) along {cp['k2_max']['label']} "
                         f"({'the given' if cp['epsilon_source'] == 'given' else 'the electronic eps_inf: an upper bound with'} permittivity)")
        elif k == "d":
            lines.append("  Coupling k^2 needs the static permittivity eps^T (--epsilon)")
    t2 = der.get("two_d") or {}
    if t2.get("anisotropy_su") is not None:
        lines.append(f"2D anisotropy A^SU {t2['anisotropy_su']:.4f} (Li et al. 2020; 0 = isotropic)")
    return lines


def _matrix_lines(m: Sequence[Sequence[float]], fmt: str = "{:10.3f}") -> List[str]:
    return ["  " + "".join(fmt.format(x) for x in row) for row in m]


def report(data: Dict[str, Any], name: str = "") -> str:
    """A plain-text summary for ``mace analyze elastic``."""
    lines: List[str] = []
    if name:
        lines.append(name)
    if not data:
        return (name + ": " if name else "") + "no elastic, piezoelectric, photoelastic or EOS results in this output"
    for w in data.get("warnings") or []:
        lines.append(f"WARNING: {w}")
    el = data.get("elastic")
    if el:
        lines.append(f"Elastic constants C ({el['unit']}{', ' + el['system'].lower() if el.get('system') else ''})")
        lines += _matrix_lines(el["matrix"])
    comp = data.get("compliance")
    if comp:
        lines.append(f"Compliance S ({comp['unit']})")
        lines += _matrix_lines(comp["matrix"], "{:11.5f}")
    mod = data.get("moduli")
    if mod and "bulk_modulus_hill_gpa" in mod:
        lines.append("Moduli (GPa)          Voigt      Reuss       Hill")
        lines.append(f"  Bulk K          {mod['bulk_modulus_voigt_gpa']:10.2f} {mod['bulk_modulus_reuss_gpa']:10.2f} "
                     f"{mod['bulk_modulus_hill_gpa']:10.2f}")
        lines.append(f"  Shear G         {mod['shear_modulus_voigt_gpa']:10.2f} {mod['shear_modulus_reuss_gpa']:10.2f} "
                     f"{mod['shear_modulus_hill_gpa']:10.2f}")
        if "youngs_modulus_gpa" in mod:
            lines.append(f"  Young's E (Hill) {mod['youngs_modulus_gpa']:.2f} GPa, Poisson's ratio {mod['poisson_ratio']:.3f}, "
                         f"Pugh K/G {mod.get('pugh_ratio', float('nan')):.2f}")
        if "universal_anisotropy" in mod:
            lines.append(f"  Universal anisotropy A^U {mod['universal_anisotropy']:.3f} (0 = isotropic)")
        vrh = data.get("crystal_vrh")
        if vrh:
            lines.append(f"  CRYSTAL's own VRH: K_H {vrh['K_H']:.2f}, G_H {vrh['G_H']:.2f}, E_H {vrh['E_H']:.2f} GPa, "
                         f"v_H {vrh['v_H']:.3f}")
        lines.append(f"  Born stability: {'stable' if mod['born_stable'] else 'UNSTABLE'} "
                     f"(smallest eigenvalue of C {mod['min_eigenvalue_gpa']:.2f} GPa)")
        if "debye_temperature_k" in mod:
            lines.append(f"  Sound velocities (km/s): transverse {mod['sound_velocity_transverse_km_s']:.3f}, "
                         f"longitudinal {mod['sound_velocity_longitudinal_km_s']:.3f}, mean {mod['sound_velocity_mean_km_s']:.3f}")
            lines.append(f"  Debye temperature {mod['debye_temperature_k']:.0f} K")
    elif mod and "layer_modulus" in mod:
        u = mod["unit"]
        lines.append(f"In-plane moduli ({u}): E_x {mod.get('youngs_modulus_x', float('nan')):.3f}, "
                     f"E_y {mod.get('youngs_modulus_y', float('nan')):.3f}, nu_xy {mod.get('poisson_ratio_xy', float('nan')):.3f}, "
                     f"G_xy {mod['shear_modulus_xy']:.3f}, layer modulus {mod['layer_modulus']:.3f}")
        lines.append(f"  Born stability: {'stable' if mod['born_stable'] else 'UNSTABLE'}")
    lines += _derived_lines(data.get("derived") or {})
    if data.get("moduli_error"):
        lines.append(f"Moduli not computed: {data['moduli_error']}")
    if data.get("seismic"):
        lines.append("Seismic velocities (km/s)   direction        Vp     Vs1     Vs2")
        for v in data["seismic"]:
            d = " ".join(f"{x:6.3f}" for x in v["direction"])
            lines.append(f"  [{d}]          {v['vp']:7.3f} {v['vs1']:7.3f} {v['vs2']:7.3f}")
    for key, title in (("piezo_e", "Direct piezoelectric tensor e"), ("piezo_d", "Converse piezoelectric tensor d"),
                       ("pockels", "Elasto-optic (Pockels) tensor p"), ("piezo_optic", "Piezo-optic tensor pi")):
        t = data.get(key)
        if t:
            lines.append(f"{title} ({t.get('unit', '')})")
            lines += _matrix_lines(t["matrix"])
    de = data.get("dielectric")
    if de:
        lines.append("Polarizability and dielectric tensor (CPHF/KS, input geometry)")
        lines.append("  E (eV)  comp   alpha (bohr^3)   epsilon    chi(1)")
        for r in de["components"]:
            lines.append(f"  {r['frequency_ev']:6.3f}  {r['component']:4s} {r['alpha']:14.4f} {r['epsilon']:10.4f} {r['chi1']:9.4f}")
        if de.get("principal_epsilon"):
            lines.append("  principal epsilon " + " ".join(f"{x:.4f}" for x in de["principal_epsilon"])
                         + ("; refractive indices " + " ".join(f"{x:.4f}" for x in de["refractive_indices"])
                            if de.get("refractive_indices") else ""))
    opt = data.get("optics") or {}
    pr = opt.get("principal")
    if pr and pr.get("unphysical"):
        lines.append(f"  principal epsilon {', '.join(f'{x:.4f}' for x in pr['epsilon'])}: unphysical (a non-positive value)")
    elif pr:
        lines.append(f"  principal axes {', '.join(pr['axis_labels'])}: n {', '.join(f'{x:.4f}' for x in pr['n'])} "
                     f"({pr['optical_class']}, birefringence {pr['birefringence']:.4g}"
                     + (f"; n_e - n_o {pr['birefringence_signed']:.4g}, optically {pr['sign']}, optic axis {pr['optic_axis_label']}"
                        if pr["optical_class"] == "uniaxial" else "")
                     + (f"; 2V {pr['optic_axial_angle_deg']:.2f} deg" if pr["optical_class"] == "biaxial" else "") + ")"
                     + (f" - SYMMETRY MISMATCH: a {pr['expected_class']} crystal" if pr.get("symmetry_mismatch") else ""))
    dy = opt.get("dynamic")
    if dy:
        lines.append(f"  n(omega) of {', '.join(dy['components'])} over {len(dy['frequency_ev'])} frequencies"
                     + (" with kappa and absorption (Im eps)" if dy["absorbing"] else " (eps real: no absorption)"))
    sh = opt.get("shg") or {}
    if sh.get("zero") is False:
        lines.append(f"SHG d_eff (pm/V): d11, d22, d33 {', '.join(f'{x:.4g}' for x in sh['d_axes'])}; largest d_L "
                     f"{abs(sh['longitudinal']['max']['value']):.4g} along {sh['longitudinal']['max']['label']}; largest |d(n)| "
                     f"{sh['magnitude']['max']['value']:.4g} along {sh['magnitude']['max']['label']}; powder <d_L^2>^1/2 "
                     f"{sh['powder_longitudinal']:.4g}, <|d|^2>^1/2 {sh['powder_magnitude']:.4g}")
    po = data.get("polarizability")
    if po:
        lines.append("Polarizability tensor (CPHF/KS, input geometry; no dielectric tensor: the volume of a "
                     "molecule, polymer or slab is not defined)")
        lines.append("  E (eV)  comp   alpha (bohr^3)")
        for r in po["components"]:
            lines.append(f"  {r['frequency_ev']:6.3f}  {r['component']:4s} {r['alpha']:14.4f}")
    for name, title in (("beta", "First hyperpolarizability beta"), ("gamma", "Second hyperpolarizability gamma")):
        h = data.get(name)
        if h:
            big = [r for r in h["components"] if r["value"] is None or abs(r["value"]) > 1e-10]
            vals = ", ".join(f"{r['component']} {r['value']:.4g}" if r["value"] is not None else f"{r['component']} overflow"
                             for r in big)
            if h.get("diverged"):
                lines.append(f"{title} (a.u.): DIVERGED, not analyzed" + (f": {vals}" if vals else ""))
            elif h.get("complete") is False:
                lines.append(f"{title} (a.u.): incomplete table ({len(h['components'])} of {h['n_independent']} components)"
                             + (f": {vals}" if vals else ", no conclusion"))
            else:
                lines.append(f"{title} (a.u.): " + (vals or ("all components zero (centrosymmetric)" if name == "beta"
                                                             else "all components zero")))
    bo = data.get("born")
    if bo:
        lines.append("Born effective charges (e): atom, Tr(Z*)/3, diagonal")
        for b in bo["born_charges"]:
            lines.append(f"  {b['atom']:4d} {b['symbol']:3s} {b['isotropic']:9.4f}   " + " ".join(f"{b['tensor'][i][i]:8.4f}" for i in range(3)))
        if bo.get("epsilon_0"):
            lines.append("  eps_inf principal " + " ".join(f"{x:.4f}" for x in bo["epsilon_inf_principal"]) + "; eps_0 principal "
                         + " ".join(f"{x:.4f}" for x in bo["epsilon_0_principal"]))
        for r in bo.get("lo_to") or []:
            pairs = [(t, l) for t, l in zip(r["to_cm"], r["lo_cm"]) if l - t > 0.05]
            lines.append(f"  LO-TO along {r['direction']}: " + (", ".join(f"{t:.1f} -> {l:.1f}" for t, l in pairs) or "no splitting")
                         + " cm-1 (computed: TO -> LO)")
    q = data.get("qha")
    if q:
        u = q.get("units") or SIZE_UNITS[3]
        lines.append("Quasi-harmonic thermal properties (P = 0)")
        lines.append(f"      T (K)   {u['size']} ({u['size_unit']})   K_T ({u['modulus_unit']})  alpha_{u['size']} (1/K)"
                     f"   K_S ({u['modulus_unit']})")
        n = len(q["temperature_k"])
        for i in sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1}):
            lines.append(f"  {q['temperature_k'][i]:9.2f} {q['volume_ang3'][i]:9.4f} {q['bulk_modulus_t_gpa'][i]:10.3f} "
                         f"{q['alpha_v_per_k'][i]:14.5e} {q['bulk_modulus_s_gpa'][i]:10.3f}")
        th = q.get("thermo")
        if th:
            lines.append(f"Along V(T) (per mole of cells; order-{th['polynomial_order']} polynomial through "
                         f"{len(th['volumes_ang3'])} volumes)")
            lines.append("      T (K)  F - F(T0) (kJ/mol)  S (J/mol K)  Cv (J/mol K)  Cp (J/mol K)  gamma_th")
            m = len(th["temperature_k"])
            for i in sorted({0, m // 4, m // 2, (3 * m) // 4, m - 1}):
                g = th["gamma_th"][i]
                lines.append(f"  {th['temperature_k'][i]:9.2f} {th['f_kj_mol'][i] - th['f_kj_mol'][0]:19.5f} "
                             f"{th['entropy_j_mol_k'][i]:12.4f} {th['cv_j_mol_k'][i]:13.4f} {th['cp_j_mol_k'][i]:13.4f} "
                             + (f"{g:9.3f}" if g is not None else "        -"))
        if q.get("gruneisen"):
            gg = q["gruneisen"]["gamma"]
            lines.append(f"  CRYSTAL's mode Grueneisen parameter: {gg[0]:.3f} to {gg[-1]:.3f}")
    eos = data.get("eos")
    if eos:
        if eos.get("fits"):
            u = eos.get("units") or SIZE_UNITS[3]
            lines.append(f"Equation of state        {u['size']}0 ({u['size_unit']})        E0 (Ha)   B0 ({u['modulus_unit']})    B0'")
            for k, f in eos["fits"].items():
                lines.append(f"  {k:20s} {f['volume_ang3']:10.4f} {f['energy_au']:16.8f} {f['bulk_modulus_gpa']:9.2f} "
                             f"{f['bulk_modulus_prime']:6.2f}")
        if eos.get("points"):
            lines.append(f"  {len(eos['points'])} E(V) points")
        for k, q in (eos.get("quality") or {}).items():
            lines.append(f"  {k:20s} residuals: RMS {q['rms_mev']:.4g} meV, largest {q['max_abs_mev']:.4g} meV ({q['quality']}; "
                         f"RMS/energy spread {q['rms_relative']:.2g})")
    return "\n".join(lines)
