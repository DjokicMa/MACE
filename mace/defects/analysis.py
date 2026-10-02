"""Formation energies and charge-transition levels for a built defect set.

Every result is for ions at the pristine positions ("unrelaxed").

3D (FACTS.md, measured on MgO V_O): raw E(q) - E(0) of a charged supercell
carries a constant offset C per electron removed. Fractional DOPING runs
(-0.5, -1.0, -1.5) give dE/dN = eps_F + C, so

    C_i = (E_{i+1} - E_{i-1}) / (N_{i+1} - N_{i-1}) - EFERMI_i

at each interior point of the calibration ladder, and

    E_corr(q) = E_raw(q) + q * C

only for q inside the calibrated range (never extrapolated). Without a
calibration only q = 0 is reported. C tracks 2*(EXT EL-SPHEROPOLE)/N_e
(reported as a cross-check only; the relation is a hypothesis). Results are
labelled "uncorrected for image charges": no dielectric correction exists yet.

2D (FACTS.md, hBN V_N): fractional slopes match eps_F directly (no offset),
and E(q) - E(0) converges as A + B/L with L = sqrt(|a x b|), fitted by least
squares over the supercell sizes. A charged sheet has no vacuum level.

Formation energy (E_F measured from the VBM):

    E_f(q; E_F) = E_corr(q) - E_pristine - sum_i n_i mu_i + q (E_VBM + E_F)

with n_i > 0 for atoms added. mu_i are absolute energies of the same method
and basis. Potential alignment (POTC) is always reported and only applied with
--apply-alignment, as q * (-dV), dV = <V_defect - V_pristine> over far atoms.
"""

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mace.constants import ANGSTROM_TO_BOHR, HARTREE_TO_EV
from mace.defects import DefectError
from mace.defects import outputs as out
from mace.defects.deck_edit import element_number, fmt_q
from mace.defects.structure import Geometry, match_atoms, neighbours, parse_geometry, pbc_distance

#: |Delta N_e + q| above this excludes a job ("DOPING not applied").
ELECTRON_TOL = 1e-4
#: |Mulliken spin - NSPIN| above this is a warning: the cell did not end in the locked state.
SPIN_TOL = 0.1
#: |spin| below this lets an unrestricted integer state serve as a calibration endpoint.
CLOSED_SHELL_SPIN = 0.01
#: Spread of the calibration constant above this (Ha) is a warning.
C_SPREAD_WARN = 0.005
#: Spread of the far-atom potential difference above this (eV) is a warning.
ALIGN_STD_WARN = 0.05

UNRELAXED = "unrelaxed (ions at the pristine positions)"
UNCORRECTED = "uncorrected for image charges"
NO_CALIBRATION_3D = (
    "3D charged states are not reported without --calibrate: raw E(q) - E(0) of a charged "
    "supercell carries a constant offset per electron (measured on MgO V_O: C = 0.444 Ha, "
    "which put eps(+/0) 15.7 eV above the VBM inside a 7.25 eV gap). Rebuild with "
    "--calibrate and run the calibration decks")


def parse_energy(text: str, what: str = "value") -> float:
    """'-75.1Ha', '-2043.6 eV' -> Ha. Units are required."""
    m = re.fullmatch(r"\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*(eV|Ha|ha|EV|HA|hartree)\s*", str(text))
    if not m:
        raise DefectError(f"{what} {text!r} needs a unit: e.g. -75.1Ha or -2043.6eV")
    value = float(m.group(1))
    return value / HARTREE_TO_EV if m.group(2).lower() == "ev" else value


def parse_mu(items: Sequence[str]) -> Dict[str, float]:
    mu = {}
    for item in items:
        if "=" not in item:
            raise DefectError(f"--mu {item!r}: expected EL=value[eV|Ha]")
        el, val = item.split("=", 1)
        element_number(el)
        mu[el.strip().capitalize()] = parse_energy(val, f"chemical potential of {el}")
    return mu


def _ev(x: Optional[float]) -> Optional[float]:
    return None if x is None else x * HARTREE_TO_EV


@dataclass
class JobResult:
    name: str
    role: str
    q: float
    doping: float
    nspin: Optional[int]
    restricted: bool
    supercell: int
    summary: Optional[out.OutputSummary]
    excluded: str = ""
    warnings: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.summary is not None and self.summary.energy is not None and not self.excluded

    @property
    def energy(self) -> Optional[float]:
        return self.summary.energy if self.summary else None

    def as_dict(self) -> dict:
        s = self.summary
        return {"name": self.name, "role": self.role, "q": self.q, "doping": self.doping,
                "nspin": self.nspin, "energy_ha": s.energy if s else None,
                "electrons": s.electrons if s else None, "spin": s.spin if s else None,
                "status": "ok" if self.ok else "excluded", "reason": self.excluded or (s.reason if s else "no output"),
                "warnings": list(self.warnings)}


def _defect_counts(defect: dict) -> Dict[str, int]:
    kind, el = defect["kind"], defect["element"].capitalize()
    if kind in ("vacancy", "ghost"):
        return {el: -1}
    if kind == "substitution":
        return {el: -1, defect["new_element"].capitalize(): 1}
    return {el: 1}


def check_jobs(manifest: dict, texts: Dict[str, str]) -> List[JobResult]:
    """Parse each job's output and apply the electron-count and spin checks."""
    results = []
    delta_z = manifest["defect"].get("delta_electrons")
    lock_cycles = manifest.get("spinlock_cycles")
    for sc_index, sc in enumerate(manifest["supercells"]):
        pris_name = sc.get("pristine_job")
        pris_text = texts.get(pris_name)
        pris = out.summarise(pris_name, pris_text) if pris_text else None
        for job in sc.get("jobs", []):
            text = texts.get(job["name"])
            summary = out.summarise(job["name"], text) if text is not None else None
            res = JobResult(job["name"], job["role"], job["q"], job["doping"], job.get("nspin"),
                            job.get("restricted", False), sc_index, summary)
            warns = []
            if summary is None:
                res.excluded = "no output"
            elif summary.energy is None:
                res.excluded = summary.reason
            elif job["role"] != "pristine":
                n = summary.electrons
                on_pristine = job.get("recipe", "").endswith("_pristine")
                if n is None:
                    res.excluded = "no Mulliken 'NO. OF ELECTRONS' line (PPAN missing?)"
                elif pris is not None and pris.electrons is not None and delta_z is not None:
                    expected = pris.electrons + (0 if on_pristine else delta_z) - job["q"]
                    if abs(n - expected) > ELECTRON_TOL:
                        res.excluded = (f"DOPING not applied: {n:g} electrons, expected {expected:g} "
                                        f"(pristine {pris.electrons:g}, q={fmt_q(job['q'])})")
                elif job["doping"] and summary.header_electrons is not None:
                    # Fallback (ECP on the defect atom): the header is the pre-DOPING count.
                    if abs((n - summary.header_electrons) - job["doping"]) > ELECTRON_TOL:
                        res.excluded = (f"DOPING not applied: Mulliken {n:g} vs header "
                                        f"{summary.header_electrons} (DOPING {job['doping']})")
                if not res.excluded and job.get("nspin") is not None and summary.spin is not None:
                    if abs(summary.spin - job["nspin"]) > SPIN_TOL:
                        warns.append(_spin_warning(summary, job["nspin"], lock_cycles))
            res.warnings = tuple(warns)
            results.append(res)
    return results


def _spin_warning(summary: out.OutputSummary, nspin: int, lock_cycles: Optional[int]) -> str:
    """Say whether the SCF ended under the lock or after it was released.

    Measured (MgO N_O q=0, SPINLOCK 1 30): the SCF converged in 15 cycles,
    still locked, and the final Mulliken spin was 0.84, not 1.
    """
    head = f"final Mulliken spin {summary.spin:g} differs from the locked NSPIN {nspin}"
    if summary.cycles is not None and lock_cycles is not None:
        if summary.cycles <= lock_cycles:
            return (f"{head}, although the SCF converged in {summary.cycles} cycles with the lock "
                    f"still on ({lock_cycles}): the final density is not in the locked spin state; "
                    "check the state before using this energy")
        return (f"{head}: the state drifted after SPINLOCK was released (lock {lock_cycles} cycles, "
                f"SCF {summary.cycles})")
    return f"{head}: the cell did not end in the locked spin state; check it before using this energy"


# ---------------------------------------------------------------------------
# 3D calibration
# ---------------------------------------------------------------------------

def _level(summary: out.OutputSummary) -> Optional[float]:
    if summary.efermi is not None:
        return summary.efermi
    return summary.edges.vbm


def _usable_endpoint(job: JobResult) -> bool:
    """An integer state can close the ladder if it is restricted or ends with |spin| < 0.01."""
    if not job.ok:
        return False
    if job.restricted:
        return True
    if job.summary.spin is None:
        return job.nspin is None
    return abs(job.summary.spin) < CLOSED_SHELL_SPIN


def calibrate(jobs: Sequence[JobResult], on: str) -> List[dict]:
    """Calibration constants, one per ladder side (removal: d < 0, addition: d > 0)."""
    cal = [j for j in jobs if j.role == "calibration" and j.ok]
    ends = [j for j in jobs if j.role == "endpoint" and j.ok]
    results = []
    for sign in (-1, 1):
        side = [j for j in cal if j.doping * sign > 0]
        if not side:
            continue
        points: Dict[float, Tuple[JobResult, str]] = {j.doping: (j, "calibration") for j in side}
        notes = []
        for d_end in (0.0, 2.0 * sign):
            q_end = -d_end
            if on == "pristine":
                cands = [j for j in jobs if j.role == "pristine"] if d_end == 0 else []
            else:
                cands = [j for j in jobs if j.role == "defect" and j.q == q_end]
            usable = [j for j in cands if _usable_endpoint(j)]
            explicit = [j for j in ends if j.doping == d_end]
            if usable:
                points[d_end] = (usable[0], "charge state" if d_end else "neutral")
            elif explicit:
                points[d_end] = (explicit[0], "restricted endpoint deck")
            elif cands:
                notes.append(f"d={d_end:+.1f}: {cands[0].name} is spin-polarised (|spin| >= "
                             f"{CLOSED_SHELL_SPIN}) and there is no restricted endpoint deck; not used")
        ds = sorted(points)
        pts = [(d, points[d][0]) for d in ds]
        values = []
        for i in range(1, len(pts) - 1):
            (_, j0), (d1, j1), (_, j2) = pts[i - 1], pts[i], pts[i + 1]
            n0, n2 = j0.summary.electrons, j2.summary.electrons
            level = _level(j1.summary)
            if level is None or n0 == n2:
                continue
            slope = (j2.energy - j0.energy) / (n2 - n0)
            c = slope - level
            sph = j1.summary.spheropole
            values.append({"at": j1.name, "doping": d1, "C": c, "slope": slope, "level": level,
                           "spheropole_crosscheck": (2 * sph / j1.summary.electrons) if sph else None})
        if not values:
            notes.append("no interior point with neighbours on both sides: no calibration")
            results.append({"side": "removal" if sign < 0 else "addition", "C": None, "values": [],
                            "notes": notes})
            continue
        cs = [v["C"] for v in values]
        single = len(values) == 1
        if single and len(pts) == 3:
            notes.append(f"only the fractional points exist: single central value at d={values[0]['doping']:+.1f}")
        spread = max(cs) - min(cs)
        if spread > C_SPREAD_WARN:
            notes.append(f"WARNING: C varies by {spread:.4f} Ha across the ladder (> {C_SPREAD_WARN})")
        qs = [-d + 0.0 for d in ds]
        results.append({"side": "removal" if sign < 0 else "addition", "C": sum(cs) / len(cs),
                        "spread": spread, "values": values, "points": [j.name for _, j in pts],
                        "q_range": [min(qs), max(qs)], "single_central": single, "notes": notes})
    return results


def _in_range(q: float, cals: Sequence[dict]) -> Optional[float]:
    if q == 0:
        return 0.0
    for c in cals:
        if c.get("C") is None:
            continue
        lo, hi = c["q_range"]
        if lo - 1e-9 <= q <= hi + 1e-9:
            return c["C"]
    return None


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def potential_alignment(pris_geom: Geometry, site_cart, potc_pris: str, potc_def: str,
                        far_fraction: float = 0.25) -> dict:
    """<V_def - V_pris> (eV) over the far atoms, matched by bohr coordinates under PBC.

    Far atoms: the farthest quarter (at least one) of the matched atoms from
    the defect site.
    """
    lat_b = pris_geom.lattice * ANGSTROM_TO_BOHR
    g_b = Geometry(lat_b, [], pris_geom.dimensionality)
    site_b = [x * ANGSTROM_TO_BOHR for x in site_cart]
    pris = out.read_potc_atoms(potc_pris)
    dfc = out.read_potc_atoms(potc_def)
    pairs = []
    for _, xyz, v in dfc:
        best = None
        for _, xyz_p, vp in pris:
            d = pbc_distance(g_b, xyz, xyz_p)
            if d < 0.1 and (best is None or d < best[0]):
                best = (d, vp)
        if best:
            pairs.append((pbc_distance(g_b, site_b, xyz), v - best[1]))
    if not pairs:
        raise DefectError("no POTC points of the defect cell match the pristine cell")
    pairs.sort()
    n = max(1, int(round(len(pairs) * far_fraction)))
    far = [dv for _, dv in pairs[-n:]]
    mean = sum(far) / len(far) * HARTREE_TO_EV
    std = (sum((x * HARTREE_TO_EV - mean) ** 2 for x in far) / len(far)) ** 0.5
    res = {"dV_ev": mean, "std_ev": std, "n_far": len(far), "n_matched": len(pairs),
           "min_far_distance_bohr": pairs[-n][0]}
    if std > ALIGN_STD_WARN:
        res["warning"] = (f"far-atom spread {std:.3f} eV > {ALIGN_STD_WARN}: no plateau, the cell is "
                          "likely too small (measured: 2x2x2 MgO is too small for a plateau)")
    return res


def alignment_term(q: float, dv_ev: float) -> float:
    """Energy (eV) added to E_f(q) when alignment is applied: q * (-dV)."""
    return q * (-dv_ev)


# ---------------------------------------------------------------------------
# Levels
# ---------------------------------------------------------------------------

def transition_levels(a_q: Dict[float, float], gap: Optional[float]) -> List[dict]:
    """eps(q1/q2) between adjacent charges from A_q = E_f(q; E_F = 0) (eV)."""
    qs = sorted(a_q, reverse=True)
    out_levels = []
    for q1, q2 in zip(qs, qs[1:]):
        eps = (a_q[q2] - a_q[q1]) / (q1 - q2)
        item = {"transition": f"{fmt_q(q1)}/{fmt_q(q2)}", "eV_above_VBM": eps}
        if gap is not None and not (0.0 <= eps <= gap):
            item["flag"] = f"outside the gap [0, {gap:.2f}] eV"
        out_levels.append(item)
    return out_levels


def fit_inverse_size(sizes: Sequence[float], values: Sequence[float]) -> dict:
    """Least-squares A + B/L."""
    L = np.asarray(sizes, float)
    y = np.asarray(values, float)
    X = np.vstack([np.ones_like(L), 1.0 / L]).T
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - (a + b / L)
    return {"A": float(a), "B": float(b), "residuals": [float(r) for r in resid]}


# ---------------------------------------------------------------------------
# Site report
# ---------------------------------------------------------------------------

def site_report(manifest: dict, pris_text: str, pris_sum: out.OutputSummary,
                jobs: Sequence[JobResult], texts: Dict[str, str]) -> List[dict]:
    geom_p = parse_geometry(pris_text)
    label = manifest.get("site", {}).get("label")
    kind = manifest["defect"]["kind"]
    # an inserted atom has no pristine label: the report centres on its position
    if geom_p is None or pris_sum.mulliken is None or (label is None and kind != "interstitial"):
        return []
    if kind == "interstitial":
        center = manifest["defect"].get("cart")
        if center is None:
            return []
        exclude = ()
    else:
        center = geom_p.atom(label).cart
        exclude = (label,)
    nn = neighbours(geom_p, center, 1e9, exclude=exclude)
    if not nn:
        return []
    cutoff = 1.2 * nn[0][1]
    shell = [lab for lab, _ in neighbours(geom_p, center, cutoff, exclude=exclude)]
    rows = []
    for job in jobs:
        if job.role != "defect" or not job.ok or job.summary.mulliken is None:
            continue
        geom_d = parse_geometry(texts[job.name])
        if geom_d is None:
            continue
        to_pris = match_atoms(geom_d, geom_p)
        from_pris = {p: d for d, p in to_pris.items()}
        entry = {"job": job.name, "q": job.q, "atoms": []}
        watch = list(shell)
        if kind in ("ghost", "substitution"):
            watch = [label] + watch
        for p_lab in watch:
            d_lab = from_pris.get(p_lab)
            if d_lab is None:
                continue
            entry["atoms"].append({
                "pristine_label": p_lab, "label": d_lab,
                "charge_pristine": pris_sum.mulliken.charge(p_lab),
                "charge": job.summary.mulliken.charge(d_lab),
                "spin": job.summary.mulliken_spin.population(d_lab) if job.summary.mulliken_spin else None,
            })
        if kind == "interstitial":
            new = [a.label for a in geom_d.atoms if a.label not in to_pris]
            for d_lab in new:
                entry["atoms"].append({"pristine_label": None, "label": d_lab,
                                       "charge": job.summary.mulliken.charge(d_lab)})
        rows.append(entry)
    return rows


# ---------------------------------------------------------------------------
# The whole set
# ---------------------------------------------------------------------------

def analyze(manifest: dict, texts: Dict[str, str], *, mu: Optional[Dict[str, float]] = None,
            vbm: Optional[float] = None, potc: Optional[Dict[str, str]] = None,
            apply_alignment: bool = False) -> dict:
    """Report for a built set. ``texts`` maps job name -> output text; ``potc``
    maps job name -> POTC.DAT text (atoms)."""
    mu = mu or {}
    potc = potc or {}
    jobs = check_jobs(manifest, texts)
    dim = manifest["dimensionality"]
    report = {
        "dimensionality": dim,
        "defect": manifest["defect"],
        "site": manifest.get("site"),
        "labels": [UNRELAXED] + ([UNCORRECTED] if dim == 3 else []),
        "image_charge_correction": "none" if dim == 3 else "A + B/L extrapolation (empirical)",
        "jobs": [j.as_dict() for j in jobs],
        "notes": [],
    }
    counts = _defect_counts(manifest["defect"])
    missing_mu = [el for el in counts if el not in mu]
    if mu and missing_mu:
        raise DefectError(f"--mu is missing {', '.join(missing_mu)}")
    report["notes"].append("chemical potentials must be absolute energies from the same method and basis")
    mu_term = -sum(n * mu[el] for el, n in counts.items()) if not missing_mu else None

    if dim == 3:
        _analyze_3d(manifest, texts, jobs, report, mu_term, vbm, potc, apply_alignment)
    else:
        _analyze_2d(manifest, texts, jobs, report, mu_term, vbm)
    return report


def _vbm_info(pris: out.OutputSummary, override: Optional[float], report: dict) -> Tuple[Optional[float], Optional[float]]:
    e = pris.edges
    info = {"vbm_ev": _ev(e.vbm), "cbm_ev": _ev(e.cbm), "gap_ev": _ev(e.gap),
            "vbm_band": e.vbm_band, "vbm_k": e.vbm_k,
            "note": "VBM from the SCF k-mesh of the pristine supercell (band and k index above); "
                    "override with --vbm"}
    if override is not None:
        info["vbm_override_ev"] = _ev(override)
    report["pristine"] = info
    vbm = override if override is not None else e.vbm
    return vbm, e.gap


def _analyze_3d(manifest, texts, jobs, report, mu_term, vbm_override, potc, apply_alignment):
    pris_job = next((j for j in jobs if j.role == "pristine"), None)
    if pris_job is None or not pris_job.ok:
        report["notes"].append("the pristine job has no converged output: nothing to analyse")
        return
    vbm, gap = _vbm_info(pris_job.summary, vbm_override, report)
    if vbm is None:
        report["notes"].append("no VBM in the pristine output (metal or locked spin): give --vbm")
        return
    cal = calibrate(jobs, manifest.get("calibration", {}).get("on", "defect") if manifest.get("calibration") else "defect")
    report["calibration"] = cal
    for c in cal:
        report["notes"] += c.get("notes", [])
    defects = {j.q: j for j in jobs if j.role == "defect" and j.ok}
    e_pris = pris_job.energy
    align = {}
    site_label = manifest.get("site", {}).get("label")
    pris_text = texts.get(pris_job.name)
    pris_geom = parse_geometry(pris_text) if pris_text else None
    if pris_job.name in potc and pris_geom is not None:
        center = (manifest["defect"].get("cart") if manifest["defect"]["kind"] == "interstitial"
                  else pris_geom.atom(site_label).cart if site_label else None)
        for q, j in defects.items():
            if j.name in potc and center is not None:
                try:
                    align[q] = potential_alignment(pris_geom, center, potc[pris_job.name], potc[j.name])
                except DefectError as exc:
                    align[q] = {"error": str(exc)}
    report["alignment"] = {fmt_q(q): a for q, a in align.items()}
    report["alignment_applied"] = bool(apply_alignment)
    states, a_q = [], {}
    for q in sorted(defects):
        j = defects[q]
        c = _in_range(q, cal)
        entry = {"q": q, "job": j.name, "E_raw_ha": j.energy}
        if c is None:
            entry["status"] = ("not calibrated (outside the calibrated range)" if cal and any(x.get("C") is not None for x in cal)
                               else "not reported (no calibration)")
            states.append(entry)
            continue
        e_corr = j.energy + q * c
        e_f = (e_corr - e_pris + q * vbm) * HARTREE_TO_EV
        if apply_alignment and q in align and "dV_ev" in align[q]:
            e_f += alignment_term(q, align[q]["dV_ev"])
        entry.update({"E_corr_ha": e_corr, "C_ha": c})
        a_q[q] = e_f
        states.append(entry)
    if 0 not in a_q:
        report["notes"].append("no converged q=0 defect job")
    if not any(x.get("C") is not None for x in cal):
        report["notes"].append(NO_CALIBRATION_3D)
    ref = a_q.get(0.0, 0.0)
    for entry in states:
        q = entry["q"]
        if q in a_q:
            if mu_term is not None:
                entry["formation_energy_ev_at_vbm"] = a_q[q] + mu_term * HARTREE_TO_EV
            else:
                entry["formation_energy_rel_q0_ev_at_vbm"] = a_q[q] - ref
    report["states"] = states
    report["levels"] = transition_levels(a_q, _ev(gap))
    if mu_term is None:
        report["notes"].append("no --mu: formation energies are relative to q=0 (E_F at the VBM)")
    report["site_report"] = site_report(manifest, pris_text, pris_job.summary, jobs, texts) if pris_text else []


def _analyze_2d(manifest, texts, jobs, report, mu_term, vbm_override):
    sizes = []
    for i, sc in enumerate(manifest["supercells"]):
        pris = next((j for j in jobs if j.supercell == i and j.role == "pristine"), None)
        if pris is None or not pris.ok:
            continue
        geom = parse_geometry(texts[pris.name])
        if geom is None:
            continue
        area = float(np.linalg.norm(np.cross(geom.lattice[0], geom.lattice[1])))
        L = math.sqrt(area)
        states = {j.q: j for j in jobs if j.supercell == i and j.role == "defect" and j.ok}
        sizes.append({"L": L, "pristine": pris, "states": states})
    if not sizes:
        report["notes"].append("no converged pristine output")
        return
    sizes.sort(key=lambda s: s["L"])
    vbms = [s["pristine"].summary.edges.vbm for s in sizes if s["pristine"].summary.edges.vbm is not None]
    largest = sizes[-1]["pristine"].summary
    vbm, gap = _vbm_info(largest, vbm_override, report)
    report["pristine"]["vbm_spread_ev"] = _ev(max(vbms) - min(vbms)) if vbms else None
    report["pristine"]["L_angstrom"] = [s["L"] for s in sizes]
    report["notes"].append("L = sqrt(|a x b|) of the supercell (Angstrom); E(q) - E(0) fitted as A + B/L")
    report["notes"].append("a charged sheet is not compensated in the vacuum: there is no vacuum level "
                           "in a charged 2D cell (FACTS.md)")
    qs = sorted({q for s in sizes for q in s["states"] if q != 0})
    fits, a_q = {}, {0.0: 0.0}
    for q in qs:
        pts = [(s["L"], s["states"][q].energy - s["states"][0].energy) for s in sizes
               if q in s["states"] and 0 in s["states"]]
        if len(pts) < 2:
            fits[fmt_q(q)] = {"error": f"{len(pts)} size(s) - at least 2 are needed"}
            continue
        f = fit_inverse_size([p[0] for p in pts], [p[1] for p in pts])
        f["points"] = [{"L": p[0], "dE_ha": p[1]} for p in pts]
        f["A_ev"], f["B_ev_A"] = f["A"] * HARTREE_TO_EV, f["B"] * HARTREE_TO_EV
        f["residuals_ev"] = [r * HARTREE_TO_EV for r in f["residuals"]]
        if len(pts) < 3:
            f["warning"] = "only 2 sizes: the fit has no residual to judge it by"
        fits[fmt_q(q)] = f
        a_q[q] = f["A"]
    report["fits"] = fits
    if vbm is None:
        report["notes"].append("no VBM in the largest pristine output: give --vbm")
        return
    # E_f(q; E_F=0) relative to q=0, extrapolated: A_q + q*E_VBM
    rel = {q: (a + q * vbm) * HARTREE_TO_EV for q, a in a_q.items()}
    report["levels"] = transition_levels(rel, _ev(gap))
    states = []
    for q, v in sorted(rel.items()):
        entry = {"q": q, "formation_energy_rel_q0_ev_at_vbm": v}
        states.append(entry)
    if mu_term is not None and 0 in sizes[-1]["states"]:
        e0 = sizes[-1]["states"][0].energy - sizes[-1]["pristine"].energy + mu_term
        report["neutral_formation_energy_ev"] = e0 * HARTREE_TO_EV
        report["notes"].append("neutral formation energy from the largest supercell (not extrapolated)")
    report["states"] = states
