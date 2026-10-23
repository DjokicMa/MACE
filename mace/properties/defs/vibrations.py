"""Derived properties of a FREQ run's phonons:

* phonon_debye_temperature: Debye temperatures from the moments of the
  phonon density of states (FREQCALC with PDOS; the PDOS blocks of the
  run's fort.25, read by mace.plotting.f25): Theta(n) = (h c / k_B)
  [(n + 3)/3 <nu^n>]^(1/n), Theta(0) = (h c / k_B) exp(1/3 + <ln nu>)
  (Barron, Berg and Morrison, Proc. R. Soc. A 242, 478, 1957), each equal
  to Theta_D for a Debye spectrum; n = 2 is the high-temperature heat
  capacity's;
* phonon_sound_velocities: the slopes of the three acoustic branches at
  Gamma along each path segment that starts or ends there (FREQCALC with
  DISPERSI and BANDS: the BAND blocks of the fort.25), their Debye
  average v_m = [(1/3)(1/v_L^3 + 2/v_T^3)]^(-1/3) over the directions,
  and the Debye temperature it gives, Theta = (h v_m / k_B)(3 n / 4 pi)^(1/3)
  with n the atoms per volume of the run's cell;
* mean_square_displacement: <u^2> of each atom of a molecule from its
  normal modes (the classical-amplitude modes CRYSTAL prints, which the
  GUI's mode viewer reads), at 0 K and 300 K: <u_i^2> = sum over the
  vibrations of (1/2) coth(hbar w / 2 k_B T) |x_i|^2, x the mode's
  turning-point displacement (sum m x^2 = hbar / w). A periodic cell's
  modes are Gamma's only: its acoustic and dispersive modes, which carry
  most of <u^2>, are missing, so it gets none.

The Debye temperature of an ELASTIC run's sound velocities is the
extractor's debye_temperature_elastic_k; the thermodynamics at every
temperature are thermodynamics_vs_temperature.
"""
from __future__ import annotations

import contextlib
import io
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mace.properties.registry import Output, Sibling, Skip, derived

HC_K = 1.438776877           # h c / k_B in cm K (second radiation constant)
C_CM_S = 2.99792458e10
from mace.constants import BOHR_TO_ANGSTROM as BOHR_A, HARTREE_TO_CM1  # noqa: E402
BOHR_M = BOHR_A * 1e-10
HBAR = 1.054571817e-34
KB = 1.380649e-23
H = 6.62607015e-34
AMU_ME = 1822.888486209
HARTREE_PER_CM = 1.0 / HARTREE_TO_CM1


def phonon_f25(out: Path) -> Optional[Path]:
    """The FREQ run's fort.25 (``<stem>.f25``, ``<stem>.fort.25``) when it holds phonon blocks (PDOS or BAND)."""
    out = Path(out)
    for p in (out.with_suffix(".f25"), out.with_name(out.stem + ".fort.25")):
        if p.is_file():
            from mace.plotting import f25
            try:
                if set(f25.file_kinds(p)) & {"PDOS", "BAND"}:
                    return p
            except (OSError, ValueError):
                return None
    return None


# ------------------------------------------------------------------ the phonon DOS
def pdos(path) -> Tuple[List[float], List[float]]:
    """(frequencies in cm-1, the total phonon DOS) of the last complete PDOS set of a phonon fort.25."""
    from mace.plotting import f25
    blocks = [b for b in f25.read_blocks(path, kinds=("PDOS",)) if b.kind == "PDOS"]
    if not blocks:
        return [], []
    chosen, _, _ = f25.select_run(blocks, -1, "phonon")
    b = chosen[-1]
    n = b.nrow * b.ncol
    return [b.dum + i * b.dk for i in range(n)], list(b.values[:n])


def debye_moments(nu: Sequence[float], g: Sequence[float], orders=(-2, -1, 0, 1, 2)) -> Dict[int, float]:
    """{n: Theta_D(n) in K} of the spectrum g(nu) (nu in cm-1; the positive frequencies, trapezoids)."""
    pts = [(x, y) for x, y in zip(nu, g) if x > 0 and y > 0]
    if len(pts) < 2:
        return {}

    def integ(f):
        s = 0.0
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            s += 0.5 * (f(x0) * y0 + f(x1) * y1) * (x1 - x0)
        return s
    norm = integ(lambda x: 1.0)
    out = {}
    for n in orders:
        if n == 0:
            out[0] = HC_K * math.exp(1.0 / 3.0 + integ(math.log) / norm)
        elif n > -3:
            m = integ(lambda x, n=n: x ** n) / norm
            out[n] = HC_K * ((n + 3) / 3.0 * m) ** (1.0 / n)
    return out


@derived("phonon_debye_temperature", calc_types=("FREQ",), category="thermodynamics", group="Phonons",
         version="1",
         siblings=(Sibling("f25", find=phonon_f25, description="the run's fort.25 (<stem>.f25) with PDOS blocks"),),
         outputs=(
             Output("debye_temperature_phonon_k", "K", label="Θ<sub>D</sub>(2) (phonon DOS)", precision=5,
                    description="Debye temperature of the second moment of the phonon DOS: the high-temperature "
                                "heat capacity's"),
             Output("debye_temperature_phonon_log_k", "K", label="Θ<sub>D</sub>(0) (phonon DOS)", precision=5,
                    description="Debye temperature of the logarithmic moment of the phonon DOS"),
             Output("phonon_debye_moments", "JSON", "json", label="Θ<sub>D</sub>(n) of the phonon DOS",
                    description="Theta_D(n) for n = -2, -1, 0, 1, 2, the highest frequency's temperature "
                                "h c nu_max / k_B, and the DOS grid"),
         ),
         source="the PDOS blocks of the run's fort.25 (FREQCALC PDOS), via mace.plotting.f25")
def _phonon_debye(ctx):
    """Debye temperatures from the moments of the phonon DOS."""
    from mace.plotting import f25
    try:
        nu, g = pdos(ctx.sibling("f25"))
    except (OSError, ValueError, f25.F25Error) as e:
        raise Skip(f"unreadable fort.25: {e}")
    if not nu:
        raise Skip("no phonon DOS (PDOS) in the fort.25")
    th = debye_moments(nu, g)
    if not th:
        raise Skip("the phonon DOS has no positive frequencies")
    top = max((x for x, y in zip(nu, g) if y > 0), default=None)
    return {"debye_temperature_phonon_k": th[2], "debye_temperature_phonon_log_k": th[0],
            "phonon_debye_moments": {"theta_k": {str(k): round(v, 3) for k, v in sorted(th.items())},
                                     "theta_max_k": round(HC_K * top, 3) if top else None,
                                     "points": len(nu), "step_cm": round(nu[1] - nu[0], 6) if len(nu) > 1 else None,
                                     "file": Path(ctx.sibling("f25")).name}}


# ------------------------------------------------------------------ acoustic slopes
def _label(p) -> str:
    return "Γ" if not any(p) else "(" + ",".join(str(v) for v in p) + ")"


def acoustic_velocities(path, kmax_fraction: float = 0.12, gamma_tol: float = 5.0) -> List[Dict[str, Any]]:
    """One record per path segment of a phonon fort.25 that starts or ends at Gamma: the toward-point,
    and the three acoustic branches' group velocities at Gamma (km/s, ascending), each the least-squares
    slope through the origin over the points within ``kmax_fraction`` of the segment (at least two).

    The k distance: f25.phonon_path_k, the one the band plot uses (a phonon fort.25's DK is the segment's
    length in bohr^-1, 2 pi / a for diamond's Gamma-X; a later segment's first point is one step past its start)."""
    from mace.plotting import f25
    blocks = [b for b in f25.read_blocks(path, kinds=("BAND",)) if b.kind == "BAND"]
    if not blocks:
        return []
    segs, _, _ = f25.select_run(blocks, -1, "phonon")
    pos, ticks = f25.phonon_path_k(segs)
    nsub = max(b.ncol for b in segs)
    out = []
    for i, b in enumerate(segs):
        if len(b.ints) < 6 or b.ncol < 3 or b.nrow < 3:
            continue
        a, z = tuple(b.ints[:3]), tuple(b.ints[3:6])
        if any(a) and any(z):
            continue
        cols = [b.values[j * b.nrow:(j + 1) * b.nrow] for j in range(b.ncol)]
        rel = [x - ticks[i] for x in pos[i]]          # the distance from the segment's start
        if any(a):                                     # ends at Gamma: walk it backward
            ks_all = [b.dk - r for r in rel][::-1]
            cols = cols[::-1]
            toward = a
        else:
            ks_all = rel
            toward = z
            if ks_all[0] > 1e-9 and i:                 # the start dropped: Gamma is the previous segment's last point
                ks_all = [0.0] + ks_all
                cols = [segs[i - 1].values[-segs[i - 1].nrow:]] + cols
        if abs(ks_all[0]) > 1e-9 or max(abs(v) for v in sorted(cols[0])[:3]) > gamma_tol:
            continue
        L = float(b.dk)
        m = max(2, int(kmax_fraction * (nsub - 1)))
        ks = ks_all[1:m + 1]
        vel = []
        for branch in range(3):
            ws = [sorted(c)[branch] for c in cols[1:m + 1]]
            slope = sum(w * k for w, k in zip(ws, ks)) / sum(k * k for k in ks)          # cm-1 bohr
            vel.append(2 * math.pi * C_CM_S * slope * BOHR_M / 1e3)                    # km/s
        out.append({"from": "Γ", "to": _label(toward), "velocities_km_s": sorted(vel), "points": m,
                    "segment_length_bohr_inv": L})
    return out


def debye_average(records: List[Dict[str, Any]]) -> Optional[float]:
    """v_m = [mean over directions and branches of v^-3]^(-1/3) (km/s)."""
    inv = [v ** -3 for r in records for v in r["velocities_km_s"] if v > 0]
    if not inv:
        return None
    return (sum(inv) / len(inv)) ** (-1.0 / 3.0)


_CELL = re.compile(r"PRIMITIVE CELL - CENTRING CODE[^\n]*?VOLUME=\s*([\d.]+)[^\n]*\n(?:[^\n]*\n){0,8}?"
                   r"\s*ATOMS IN THE ASYMMETRIC UNIT\s+\d+\s*-\s*ATOMS IN THE UNIT CELL:\s*(\d+)")


def atom_density(text: str, atoms=None, volume=None) -> Optional[float]:
    """Atoms per Å³ of the cell the run computed (a SCELPHONO run's supercell): the atom count and the
    volume CRYSTAL prints together for it (PRIMITIVE CELL ... VOLUME= and ATOMS IN THE UNIT CELL), else
    ``atoms / volume`` as the caller gives them."""
    found = _CELL.findall(text or "")
    if found:
        v, n = found[-1]
        if float(v) > 0 and int(n) > 0:
            return int(n) / float(v)
    if atoms and volume:
        return float(atoms) / float(volume)
    return None


def debye_from_sound(vm_km_s: float, atoms_per_A3: float) -> float:
    """Theta_D = (h v_m / k_B)(3 n / 4 pi)^(1/3) (K), n atoms per Å³."""
    n = atoms_per_A3 * 1e30
    return H * vm_km_s * 1e3 / KB * (3 * n / (4 * math.pi)) ** (1.0 / 3.0)


@derived("phonon_sound_velocities", calc_types=("FREQ",), category="mechanical", group="Phonons", version="1",
         siblings=(Sibling("f25", find=phonon_f25, description="the run's fort.25 (<stem>.f25) with phonon bands"),),
         outputs=(
             Output("sound_velocity_phonon_ta_min_km_s", "km/s", label="v<sub>TA</sub> min (phonon bands)",
                    precision=4, description="slowest transverse acoustic velocity at Γ over the path's directions"),
             Output("sound_velocity_phonon_la_max_km_s", "km/s", label="v<sub>LA</sub> max (phonon bands)",
                    precision=4, description="fastest longitudinal acoustic velocity at Γ over the path's directions"),
             Output("sound_velocity_phonon_mean_km_s", "km/s", label="v<sub>m</sub> (phonon bands)", precision=4,
                    description="Debye average [mean of v^-3]^(-1/3) of the acoustic velocities over the path's "
                                "directions through Γ"),
             Output("debye_temperature_sound_k", "K", label="Θ<sub>D</sub> (acoustic slopes)", precision=5,
                    description="(h v_m / k_B)(3 n / 4 pi)^(1/3), n the atoms per volume of one cell (a "
                                "SCELPHONO run's supercell: its atoms over its volume)"),
             Output("phonon_sound_velocities", "JSON", "json", label="Acoustic velocities per direction",
                    description="per path segment through Γ: the direction, the three acoustic velocities (km/s) "
                                "and the points fitted"),
         ),
         source="the BAND blocks of the run's fort.25 (FREQCALC DISPERSI + BANDS), via mace.plotting.f25; "
                "the atom count and volume of the computed cell (a SCELPHONO run's supercell) the .out prints")
def _phonon_sound(ctx):
    """Acoustic velocities at Γ from the phonon bands; their Debye temperature."""
    from mace.plotting import f25
    try:
        recs = acoustic_velocities(ctx.sibling("f25"))
    except (OSError, ValueError, f25.F25Error) as e:
        raise Skip(f"unreadable fort.25: {e}")
    if not recs:
        raise Skip("no phonon band segment through Γ in the fort.25")
    vm = debye_average(recs)
    vals: Dict[str, Any] = {
        "sound_velocity_phonon_ta_min_km_s": min(r["velocities_km_s"][0] for r in recs),
        "sound_velocity_phonon_la_max_km_s": max(r["velocities_km_s"][2] for r in recs),
        "sound_velocity_phonon_mean_km_s": vm,
        "phonon_sound_velocities": {"directions": [{**r, "velocities_km_s": [round(v, 5) for v in r["velocities_km_s"]]}
                                                   for r in recs], "file": Path(ctx.sibling("f25")).name}}
    # the atoms per volume of one cell: a SCELPHONO run's atoms_in_unit_cell is the supercell's, its
    # primitive_cell_volume the primitive cell's (8x the density of a 2x2x2: 2x the temperature)
    rho = atom_density(ctx.content, ctx.value("atoms_in_unit_cell"), ctx.value("primitive_cell_volume"))
    if vm and rho and (ctx.dimensionality or 3) == 3:
        vals["debye_temperature_sound_k"] = debye_from_sound(vm, rho)
    return vals


# ------------------------------------------------------------------ mean-square displacement
_MASS_HEAD = "ATOMS ISOTOPIC MASS (AMU) FOR FREQUENCY CALCULATION"
_MASS = re.compile(r"(\d+)\s+([A-Z][A-Za-z]?)\s+(\d+\.\d+)")


def masses(text: str) -> List[float]:
    """The isotopic masses (amu) of the FREQ run's atoms, in order (the last table printed)."""
    i = text.rfind(_MASS_HEAD)
    if i < 0:
        return []
    block = text[i + len(_MASS_HEAD):]
    rows = []
    started = False
    for line in block.splitlines()[1:]:
        found = _MASS.findall(line)
        if found:
            started = True
            rows += found
        elif started:
            break
    rows.sort(key=lambda r: int(r[0]))
    return [float(r[2]) for r in rows]


def normal_modes(path) -> Dict[str, Any]:
    """The modes and displacements the GUI's mode viewer reads (vibmode_viewer.Crystal23FreqParser)."""
    from mace.plotting.engines.vibmode_viewer import Crystal23FreqParser
    with contextlib.redirect_stdout(io.StringIO()):
        p = Crystal23FreqParser(str(path))
    return {"modes": p.modes, "displacements": p.displacements}


def msd(modes, displacements, mass_amu: Sequence[float], temperatures=(0.0, 300.0), zero_cm: float = 1.0
        ) -> Dict[str, Any]:
    """<u^2> per atom (Å^2) at each temperature: sum over the real vibrations (|nu| > zero_cm, not
    imaginary) of (1/2) coth(h c nu / 2 k_B T) |x_i|^2, each mode's displacement x (bohr) scaled so that
    sum m x^2 = hbar / omega exactly (CRYSTAL prints it so, to its 4 decimals)."""
    import numpy as np
    m = np.asarray(mass_amu, float) * AMU_ME
    nat = len(m)
    used, skipped_imag, skipped_zero, norm_dev = 0, 0, 0, 0.0
    u2 = {T: np.zeros(nat) for T in temperatures}
    for md in modes:
        nu = float(md["freq"])
        x = displacements.get(md["mode"])
        if x is None:
            continue
        if md.get("imaginary") or nu < 0:
            skipped_imag += 1
            continue
        if abs(nu) <= zero_cm:
            skipped_zero += 1
            continue
        x = np.asarray(x, float).reshape(nat, 3)
        w = nu * HARTREE_PER_CM
        s = float((m[:, None] * x ** 2).sum()) * w
        if s <= 0:
            continue
        norm_dev = max(norm_dev, abs(s - 1.0))
        x2 = (x ** 2).sum(axis=1) / s                    # sum m x^2 w = 1 (hbar)
        used += 1
        for T in temperatures:
            c = 1.0 if T <= 0 else 1.0 / math.tanh(HC_K * nu / (2.0 * T))
            u2[T] += 0.5 * c * x2
    return {"u2_A2": {T: (v * BOHR_A ** 2).tolist() for T, v in u2.items()}, "modes_used": used,
            "modes_imaginary": skipped_imag, "modes_zero": skipped_zero, "normalization_deviation": norm_dev}


@derived("mean_square_displacement", calc_types=("FREQ",), category="frequency", group="Vibrations", version="1",
         outputs=(
             Output("msd_mean_300k_ang2", "Å²", label="⟨u<sup>2</sup>⟩ mean (300 K)", precision=4,
                    description="mean over a molecule's atoms of the mean-square displacement at 300 K (its "
                                "vibrations; translations and rotations left out)"),
             Output("msd_max_300k_ang2", "Å²", label="⟨u<sup>2</sup>⟩ max (300 K)", precision=4,
                    description="largest mean-square displacement of an atom at 300 K"),
             Output("msd_mean_0k_ang2", "Å²", label="⟨u<sup>2</sup>⟩ mean (0 K)", precision=4,
                    description="mean over the atoms of the zero-point mean-square displacement"),
             Output("mean_square_displacements", "JSON", "json", label="⟨u<sup>2</sup>⟩ per atom",
                    description="per atom: element, <u^2> at 0 K and 300 K (Å²), U_iso = <u^2>/3 and "
                                "B = 8 pi^2 U_iso at 300 K; the modes used and left out"),
         ),
         source=".out: NORMAL MODES NORMALIZED TO CLASSICAL AMPLITUDES and the isotopic masses (the GUI's mode "
                "viewer's parser)")
def _msd(ctx):
    """Mean-square displacement of each atom of a molecule from its normal modes."""
    if (ctx.dimensionality if ctx.dimensionality is not None else 3) != 0:
        raise Skip("a periodic cell's printed modes are Γ's only: the acoustic and dispersive modes that carry "
                   "most of <u²> are missing")
    if "NORMALIZED TO CLASSICAL AMPLITUDES" not in ctx.content:
        raise Skip("the output prints no normal modes")
    mass = masses(ctx.content)
    if not mass:
        raise Skip("no isotopic masses in the output")
    nm = normal_modes(ctx.output_file)
    if not nm["displacements"]:
        raise Skip("no normal-mode displacements could be read")
    first = next(iter(nm["displacements"].values()))
    if len(first) != len(mass):
        raise Skip(f"{len(first)} displaced atoms, {len(mass)} masses")
    r = msd(nm["modes"], nm["displacements"], mass)
    if not r["modes_used"]:
        raise Skip("no real vibration")
    els = re.findall(r"\d+\s+([A-Z][A-Za-z]?)\s+\d+\.\d+", ctx.content[ctx.content.rfind(_MASS_HEAD):][:20000])
    u0, u3 = r["u2_A2"][0.0], r["u2_A2"][300.0]
    atoms = [{"atom": i + 1, "element": els[i] if i < len(els) else "?", "u2_0K_A2": round(u0[i], 6),
              "u2_300K_A2": round(u3[i], 6), "U_iso_300K_A2": round(u3[i] / 3, 6),
              "B_300K_A2": round(8 * math.pi ** 2 * u3[i] / 3, 5)} for i in range(len(mass))]
    detail = {"atoms": atoms, "temperatures_k": [0, 300], "modes_used": r["modes_used"],
              "modes_left_out_zero": r["modes_zero"], "modes_left_out_imaginary": r["modes_imaginary"],
              "normalization_deviation": round(r["normalization_deviation"], 6)}
    if r["modes_imaginary"]:
        detail["warning"] = f"{r['modes_imaginary']} imaginary mode(s) left out: <u²> is not that of a minimum"
    return {"msd_mean_300k_ang2": sum(u3) / len(u3), "msd_max_300k_ang2": max(u3),
            "msd_mean_0k_ang2": sum(u0) / len(u0), "mean_square_displacements": detail}
