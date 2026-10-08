"""``preoptimize``: the one pre-optimization pipeline of the CLI and the GUI.

1. Prepare: optionally symmetrize first (``symmetrize_tol``, off by
   default). The prepared start is the reference of every later check, and
   its space group (spglib number and Hall number) is ``sg_ref``.
2. Springs baseline, always. With ``method == "springs"`` it *is* the
   result, run with the caller's fmax, steps and time limit: the very call
   the GUI's Clean made before the other tiers existed. For any other method
   it runs with the spring defaults and is the fallback.
3. The tier list: the method asked for, or Auto's order (``classify``).
4. Refusals before a run (size, elements, vacuum, charge, memory).
5. The tier runs from the prepared start (not from the spring result).
6. Sanity checks against the prepared start: energy rose, |dV| > 10 %, a new
   or worse close contact, any space-group change (a gain is never allowed:
   CRYSTAL OPT keeps the group it is given and could not restore the lost
   distortion), not converged, non-finite numbers.
7. The first tier result that passes wins; otherwise the spring result,
   with the reasons (``fell_back``).

Two callbacks, never one: ``canceled()`` is a pure predicate (calculators
poll it every 50 ms), ``progress(fraction, text)`` reports. Only the hook
``clean()`` calls once per FIRE step counts steps.

An FF/ML geometry is a starting geometry only: CRYSTAL still optimizes it.
Nothing here reads or writes OPTGEOM, the functional, the basis or the
tolerances.
"""
from __future__ import annotations

import datetime
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from . import proc

TIER_DEFAULTS = {
    "springs": {"fmax": 1e-3, "steps": 2000, "time_limit": 10.0},
    "gfnff": {"fmax": 0.01, "steps": 500, "time_limit": 60.0},
    "mlip": {"fmax": 0.02, "steps": 300, "time_limit": 120.0},
}
MAX_TIME_LIMIT = {"springs": 120.0, "gfnff": 600.0, "mlip": 600.0, "auto": 600.0}

MAX_ATOMS_ANY = 2000
MAX_ATOMS_GFNFF = 1000
MAX_Z_GFNFF = 86
ML_MAX_ATOMS_CPU = 300
ML_MAX_ATOMS_CPU_LONG = 1500     # CLI, with an explicit --time-limit
ML_MIN_RAM_GB = 2.0
VACUUM_GAP = 5.0                 # Å: an axis with a gap this wide is a vacuum axis
GFNFF_MIN_VACUUM = 10.0          # Å, atom center to center: thinner vacuum is refused for GFN-FF
CONTACT_SCALE = 0.75
DV_LIMIT = 0.10

SPRINGS_LABEL = "Springs (rough spring model)"


class NoResult(RuntimeError):
    """``fallback=False`` and the tier did not give an accepted result."""

    def __init__(self, message, result=None):
        super().__init__(message)
        self.result = result


def method_kind(method: str) -> str:
    if method in ("springs", "gfnff", "auto"):
        return method
    if method.startswith("mlip:") and len(method) > 5:
        return "mlip"
    raise ValueError(f"unknown method {method!r}: use springs, gfnff, mlip:<id> or auto")


def defaults_for(method: str) -> Dict[str, float]:
    kind = method_kind(method)
    return dict(TIER_DEFAULTS["mlip" if kind in ("mlip", "auto") else kind])


# --------------------------------------------------------------------------- geometry checks

def spacegroup(atoms, symprec: float = 1e-3) -> Optional[Dict[str, Any]]:
    """spglib's group of ``atoms``: number, Hall number, symbol and the
    number of operations in this cell; None without a group."""
    import spglib
    try:
        ds = spglib.get_symmetry_dataset((atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers),
                                         symprec=symprec)
    except Exception:
        return None
    if ds is None:
        return None
    get = (lambda k: ds[k]) if isinstance(ds, dict) else (lambda k: getattr(ds, k))
    return {"number": int(get("number")), "hall": int(get("hall_number")),
            "international": str(get("international")), "n_ops": int(len(get("rotations")))}


def sg_name(g: Optional[Dict[str, Any]]) -> str:
    return g["international"] if g else "no group"


def same_group(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return a["number"] == b["number"] and a["hall"] == b["hall"]


def group_change(before, after) -> Optional[str]:
    """None, "gain", "loss" or "change" (same operation count, other group)."""
    if same_group(before, after):
        return None
    nb = before["n_ops"] if before else 1
    na = after["n_ops"] if after else 1
    return "gain" if na > nb else "loss" if na < nb else "change"


def _gaps(atoms):
    """Per lattice vector: (the widest empty slab along the normal of the
    opposite cell face minus the covalent radii of the two atoms that bound
    it, the same slab measured between the atom centers)."""
    from mace.gui.structures import _covalent
    cell = atoms.cell.array
    rad = _covalent(atoms.numbers)
    vol = abs(np.linalg.det(cell))
    out = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        cr = np.cross(cell[j], cell[k])
        n = cr / np.linalg.norm(cr)
        d = vol / np.linalg.norm(cr)          # the cell's height along n
        if len(atoms) == 0:
            out.append((d, d))
            continue
        x = (atoms.positions @ n) % d
        order = np.argsort(x)
        xs, rs = x[order], rad[order]
        nxt = np.roll(np.arange(len(xs)), -1)
        gaps = (xs[nxt] - xs) % d
        if len(xs) == 1:
            gaps = np.array([d])
        free = gaps - rs - rs[nxt]
        out.append((float(max(free.max(), 0.0)), float(gaps.max())))
    return out


def vacuum_gaps(atoms) -> List[float]:
    """Per lattice vector, the widest empty slab (minus covalent radii).
    Every GUI and CLI cell is periodic in 3D, so slabs and molecules are
    cells with vacuum; this finds it geometrically."""
    return [g for g, _ in _gaps(atoms)]


def vacuum_axes(atoms) -> List[int]:
    return [i for i, g in enumerate(vacuum_gaps(atoms)) if g >= VACUUM_GAP]


def close_contacts(start, after, scale: float = CONTACT_SCALE) -> List[str]:
    """Pairs closer than ``scale`` x the sum of their covalent radii in
    ``after`` that were not that short in ``start`` (or got 0.01 Å shorter)."""
    from ase.neighborlist import neighbor_list
    from mace.gui.structures import _covalent
    rad = _covalent(after.numbers)
    ii, jj, SS, dd = neighbor_list("ijSd", after, list(scale * rad))
    out = []
    cell0 = start.cell.array
    for i, j, s, d in zip(ii, jj, SS, dd):
        if i > j or (i == j and tuple(s) < (0, 0, 0)):     # each pair once
            continue
        lim = scale * (rad[i] + rad[j])
        d0 = float(np.linalg.norm(start.positions[j] + s @ cell0 - start.positions[i]))
        if d0 >= lim or d < d0 - 0.01:
            sym = after.get_chemical_symbols()
            out.append(f"{sym[i]}{i + 1}-{sym[j]}{j + 1} {d:.2f} Å")
    return out


# --------------------------------------------------------------------------- Auto

RDA_FALLBACK_K = 1.25    # bonded when d < k (r_i + r_j), ASE's covalent radii (inside RDA's scoring window)


def _rda_fixed(atoms, k: float = RDA_FALLBACK_K):
    """ASE's rank determination (RDA) for the bonds d < k (r_i + r_j): the
    component histogram (0D, 1D, 2D, 3D). (0, 0, 0, 1) when it cannot run."""
    try:
        from ase.data import covalent_radii
        from ase.geometry.dimensionality.rank_determination import RDA
        from ase.neighborlist import neighbor_list
        ii, jj, SS = neighbor_list("ijS", atoms, list(k * covalent_radii[atoms.numbers]), self_interaction=False)
        calc = RDA(len(atoms))
        for i, j, s in zip(ii, jj, SS):
            calc.insert_bond(int(i), int(j), tuple(int(x) for x in s))
        return tuple(int(x) for x in calc.check())
    except Exception:
        return (0, 0, 0, 1)


RDA_KMAX = 3.0    # the k scan stops here: ASE's score f(k) is 0.994 at k = 3


def _rda_scan(atoms, kmax: float = RDA_KMAX):
    """ASE's interval analysis (``analyze_dimensionality(atoms, "RDA")``:
    bonds d < k (r_i + r_j) inserted in order of k, an interval per change
    of the component histogram, scored by ASE's f(k), merged by type) with
    the scan capped at ``kmax``. ASE scans on until every periodic image is
    joined, which across the vacuum of a slab means k ~ 8 and minutes (32,
    64, 128-atom MgO slabs: 17, 62, 227 s); past k = 3 the score left is
    0.6 %, so the top interval is the same. Returns (histogram, dimtype)."""
    from ase.data import covalent_radii
    from ase.geometry.dimensionality.interval_analysis import build_kinterval, merge_intervals
    from ase.geometry.dimensionality.rank_determination import RDA
    from ase.neighborlist import neighbor_list
    rs = covalent_radii[atoms.numbers]
    ii, jj, SS, dd = neighbor_list("ijSd", atoms, list(kmax * rs), self_interaction=False)
    keep = (ii < jj) | ((ii == jj) & [tuple(s) > (0, 0, 0) for s in SS])
    ii, jj, SS, dd = ii[keep], jj[keep], SS[keep], dd[keep]
    ks = dd / (rs[ii] + rs[jj])
    calc = RDA(len(atoms))
    hprev = calc.check()
    comp_prev, cdim_prev = calc.get_components()
    end = [0, 0, 0, 0]
    end[int(sum(atoms.pbc))] = 1
    end = tuple(end)
    intervals, kprev = [], 0.0
    for n in np.argsort(ks, kind="stable"):
        k = float(ks[n])
        calc.insert_bond(int(ii[n]), int(jj[n]), tuple(int(x) for x in SS[n]))
        h = calc.check()
        if h == hprev:
            continue
        comp, cdim = calc.get_components()
        if k != kprev:
            intervals.append(build_kinterval(kprev, k, hprev, comp_prev, cdim_prev))
        kprev, hprev, comp_prev, cdim_prev = k, h, comp, cdim
        if h == end:
            break
    intervals.append(build_kinterval(kprev, float("inf"), hprev, comp_prev, cdim_prev))
    top = sorted(merge_intervals(intervals), reverse=True, key=lambda x: (x.score, x.b - x.a))[0]
    return tuple(int(x) for x in top.h), str(top.dimtype)


def classify(atoms, charge: int = 0) -> Dict[str, Any]:
    """Auto's view of a structure: its RDA dimensionality (ASE's own bonding,
    a scan over covalent-radius intervals; it works for metals, where the
    spring model sees no bonds), 0D components and C-H content."""
    from mace.gui.structures import _covalent
    try:
        h, dimtype = _rda_scan(atoms)
    except Exception:
        # never guess "3D" from a failure: the same rank determination at one
        # bond scale still sees an ion pair in a box as 0D
        h = _rda_fixed(atoms)
        dimtype = "".join(str(i) for i, e in enumerate(h) if e) + "D"
    organic = False
    z = atoms.numbers
    if 6 in z and 1 in z:
        from ase.neighborlist import neighbor_list
        rad = _covalent(z)
        cut = [1.25 * r if n in (1, 6) else 0.0 for r, n in zip(rad, z)]
        ii, jj = neighbor_list("ij", atoms, cut)
        organic = bool(np.any((z[ii] == 6) & (z[jj] == 1)))
    zero_d = h[0] > 0
    reasons = []
    if charge:
        reasons.append(f"charge {charge:+d}")
    if zero_d:
        reasons.append("molecules or ions in a box (0D components)")
    if organic:
        reasons.append("organic (C-H bonds)")
    return {"dimtype": dimtype, "components": list(h), "organic": organic, "zero_d": zero_d,
            "molecular_or_charged": bool(reasons), "reasons": reasons}


def auto_order(cls: Dict[str, Any], charge: int) -> List[str]:
    from . import registry
    default, fallback = registry.default_id(), registry.fallback_id()
    if cls["molecular_or_charged"]:
        order = ["gfnff"]
        if charge == 0:
            order.append(f"mlip:{default}")
        return order
    order = [f"mlip:{default}"]
    if fallback and fallback != default:
        order.append(f"mlip:{fallback}")
    return order + ["gfnff"]


# --------------------------------------------------------------------------- tiers

@dataclass
class Tier:
    """One runnable pre-optimization method (resolved for this run)."""
    method: str
    kind: str
    label: str
    ready: bool = True
    reason: str = ""
    factory: Optional[Callable[..., Any]] = None     # (atoms, **opts) -> ase Calculator
    info: Dict[str, Any] = field(default_factory=dict)


def resolve_tier(method: str, *, threads: int = 1, device: str = "cpu", mlip_lease=None,
                 canceled=None, nice: int = 0, recheck: bool = False) -> Tier:
    """The tier behind ``method``, with its readiness from detection."""
    kind = method_kind(method)
    if kind == "gfnff":
        from . import detect, gfnff
        st = detect.xtb_status()
        info = {"code": "xtb", "version": st.get("version"), "license": "LGPL-3.0",
                "citation": gfnff.CITATIONS, "label": f"GFN-FF (xtb {st.get('version') or '?'})"}
        if st.get("module"):
            # the module name is what this machine offered at run time, not code
            info["module"] = st["module"]
            info["source"] = f"xtb {st.get('version') or '?'} (module {st['module']}, loaded automatically)"
        if st["status"] != "ready":
            return Tier(method, kind, "GFN-FF", False, st.get("reason") or st["status"], info=info)
        path, xenv = st["path"], st.get("env")

        def factory(atoms, charge=0, uhf=0, **_):
            return gfnff.GFNFF(xtb=path, charge=charge, uhf=uhf, canceled=canceled, nice=nice, env=xenv)
        return Tier(method, kind, info["label"], True, "", factory, info)
    if kind == "mlip":
        from . import detect, registry
        mid = method.split(":", 1)[1]
        model = registry.get(mid)
        if model is None:
            return Tier(method, kind, f"ML potential {mid}", False, f"no ML model {mid!r} in the registry")
        info = registry.provenance_info(model)
        label = f"ML potential: {model['label']}"
        if mlip_lease is not None and (mlip_lease.get("model") != mid or mlip_lease.get("error")):
            why = mlip_lease.get("error") or (f"the GUI loads one ML model per Clean ({mlip_lease.get('model')}); "
                                              f"pick {model['label']} as the method to run it")
            return Tier(method, kind, label, False, why, info=info)
        if mlip_lease is None:
            st = detect.mlip_status(model, recheck=recheck)
            if st["status"] != "ready":
                return Tier(method, kind, label, False, st.get("reason") or st["status"], info=info)
            info.update({"versions": st.get("versions", {}), "weights_sha256": st.get("weights_sha256")})
        from . import mlip_client

        def factory(atoms, **_):
            calc = mlip_client.make_calculator(model, threads=threads, device=device, lease=mlip_lease,
                                               canceled=canceled, elements=set(int(z) for z in atoms.numbers),
                                               nice=nice)
            hello = calc.info.get("hello") or {}
            info["versions"] = hello.get("versions") or info.get("versions", {})
            info["device_text"] = calc.info.get("device") or mlip_client.device_mapping(device)[2]
            return calc
        return Tier(method, kind, label, True, "", factory, info)
    raise ValueError(f"no tier for {method!r}")


# --------------------------------------------------------------------------- the result

@dataclass
class PreoptResult:
    atoms: Any
    method_requested: str
    method_ran: str
    label_ran: str
    attempts: List[Dict[str, Any]]
    fell_back: bool
    report: Dict[str, Any]
    params: Dict[str, Any]
    sg_input: Optional[Dict[str, Any]]
    sg_ref: Optional[Dict[str, Any]]
    sg_final: Optional[Dict[str, Any]]
    symmetrize: Optional[Dict[str, Any]] = None
    tier_info: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    classification: Optional[Dict[str, Any]] = None
    nothing_to_clean: bool = False

    @property
    def summary(self) -> str:
        return describe(self)


def _attempt(method, status, reason="", **kw) -> Dict[str, Any]:
    a = {"method": method, "status": status, "reason": reason, "steps": None, "converged": None,
         "dE_eV": None, "dV_pct": None, "sg_before": None, "sg_after": None, "sg_change": None,
         "max_shift_A": None, "seconds": None}
    a.update(kw)
    return a


def _short(method: str) -> str:
    k = method_kind(method) if method != "springs" else "springs"
    return {"springs": "springs", "gfnff": "GFN-FF", "auto": "Auto"}.get(k, "ML " + method.split(":", 1)[-1])


def describe(r: PreoptResult) -> str:
    """One line: which method ran and why, e.g. "Auto: GFN-FF refused (Z > 86);
    ML rejected (energy rose); kept the spring result"."""
    rep = r.report or {}
    steps = rep.get("steps")
    how = (f"converged in {steps} step{'' if steps == 1 else 's'}" if rep.get("converged")
           else f"stopped after {steps} steps (not converged)" if steps is not None else "unchanged")
    lead = "Auto: " if r.method_requested == "auto" else ""
    if r.nothing_to_clean:
        return lead + "nothing to clean (no bonded pairs); the structure is unchanged"
    if r.method_ran == "springs" and r.method_requested != "springs":
        tried = [f"{_short(a['method'])} {a['status']}" + (f" ({a['reason']})" if a["reason"] else "")
                 for a in r.attempts if a["method"] != "springs"]
        return lead + "; ".join(tried + ["kept the spring result"])
    return f"{lead}{r.label_ran}, {how}"


# --------------------------------------------------------------------------- the pipeline

def _hook(canceled, progress, max_steps, time_limit, lo=0.0, hi=1.0, name=""):
    """clean()'s once-per-step hook: count, report, and the cancel check."""
    t0, n = time.monotonic(), [0]
    span = hi - lo

    def should_stop():
        n[0] += 1
        if progress is not None:
            frac = max(n[0] / max(max_steps, 1), (time.monotonic() - t0) / max(time_limit, 1e-6))
            progress(lo + span * min(frac, 1.0), f"{name}{n[0]} steps")
        return bool(canceled()) if canceled is not None else False
    return should_stop


def _finite(atoms) -> bool:
    return bool(np.all(np.isfinite(atoms.positions)) and np.all(np.isfinite(atoms.cell.array)))


def _max_shift(a, b) -> float:
    d = b.positions - a.positions
    return float(np.sqrt((d ** 2).sum(axis=1)).max()) if len(a) else 0.0


def _gain_text(before, after, by="the pre-optimization") -> str:
    return (f"space group raised from {sg_name(before)} to {sg_name(after)} by {by}: CRYSTAL OPT would keep "
            f"{sg_name(after)} and could never restore the distortion")


def refusal(tier: Tier, atoms, *, charge: int, auto: bool, relax_cell: bool, route: str,
            device: str, explicit_time_limit: bool, leased: bool) -> Optional[str]:
    n = len(atoms)
    if n == 0:
        return "no atoms"
    if n > MAX_ATOMS_ANY:
        return f"{n} atoms (more than {MAX_ATOMS_ANY})"
    if tier.kind == "gfnff":
        if n > MAX_ATOMS_GFNFF:
            return f"{n} atoms (GFN-FF here takes at most {MAX_ATOMS_GFNFF})"
        heavy = sorted({int(z) for z in atoms.numbers if z > MAX_Z_GFNFF})
        if heavy:
            from ase.data import chemical_symbols
            return f"Z > {MAX_Z_GFNFF} ({', '.join(chemical_symbols[z] for z in heavy)}): GFN-FF gives wrong energies there"
    both = _gaps(atoms)
    vac = [i for i, (g, _) in enumerate(both) if g >= VACUUM_GAP]
    if tier.kind == "gfnff":
        thin = [i for i in vac if both[i][1] < GFNFF_MIN_VACUUM]
        if thin:
            return (f"vacuum of {min(both[i][1] for i in thin):.1f} Å along {'abc'[thin[0]]} (atom center to "
                    f"center): GFN-FF needs at least {GFNFF_MIN_VACUUM:g} Å between periodic images")
    if relax_cell and vac and tier.kind in ("gfnff", "mlip"):
        return "cell relaxation of a cell with vacuum (a slab or a molecule in a box)"
    if tier.kind == "mlip":
        on_cpu = not str(device).startswith("cuda")
        limit = ML_MAX_ATOMS_CPU
        if route == "cli" and (explicit_time_limit or not on_cpu):
            limit = ML_MAX_ATOMS_CPU_LONG
        if on_cpu and n > limit or n > ML_MAX_ATOMS_CPU_LONG:
            return (f"{n} atoms: too large for an interactive ML pre-optimization (about 2-4 s per step at 216 "
                    f"atoms); use springs, a smaller cell, or the CLI" if route == "gui" else
                    f"{n} atoms: more than {limit} on the CPU (give --time-limit for up to "
                    f"{ML_MAX_ATOMS_CPU_LONG}, or --device cuda:N)")
        if charge and auto:
            return "charged system (ML potentials take no charge input)"
        if not leased:
            mem = proc.mem_available_gb()
            if mem is not None and mem < ML_MIN_RAM_GB:
                return f"only {mem:.1f} GB of RAM available (an ML model needs about 2 GB)"
    return None


def _first_line(text) -> str:
    return (str(text or "").strip().splitlines() or [""])[0]


def _tier_failure(tier: Tier, exc: BaseException) -> str:
    if tier.kind == "gfnff":
        from . import gfnff
        if isinstance(exc, gfnff.GFNFFError):
            return str(exc)
    if tier.kind == "mlip":
        from . import mlip_client
        if isinstance(exc, mlip_client.MlipError):
            return f"{exc.kind}: {exc}"
    return (str(exc).strip().splitlines() or [type(exc).__name__])[0]


def preoptimize(atoms, method: str = "springs", *, charge: int = 0, uhf: int = 0, keep_symmetry: bool = True,
                symprec: float = 1e-3, symmetrize_tol: Optional[float] = None, relax_cell: bool = False,
                fmax: Optional[float] = None, steps: Optional[int] = None, time_limit: Optional[float] = None,
                threads: Optional[int] = None, device: str = "cpu", canceled: Optional[Callable[[], bool]] = None,
                progress: Optional[Callable[[float, str], None]] = None, mlip_lease=None, fallback: bool = True,
                route: str = "cli", explicit_time_limit: bool = False, nice: int = 0,
                resolve: Optional[Callable[..., Tier]] = None) -> PreoptResult:
    """Pre-optimize a copy of ``atoms`` (see the module doc). Raises
    ``mace.gui.clean.NothingToClean`` for springs without bonds,
    ``proc.Canceled`` on a cancel, ``NoResult`` with ``fallback=False``."""
    from mace.gui import clean as C
    kind = method_kind(method)
    resolve = resolve or resolve_tier
    threads = threads or proc.default_threads()
    canceled = canceled or (lambda: False)
    proc.sweep()

    # 1. prepare
    start = atoms.copy()
    start.calc = None
    sg_input = spacegroup(atoms, symprec)
    sym_info = None
    notes: List[str] = []
    warnings: List[str] = []
    if symmetrize_tol:
        from ase.spacegroup.symmetrize import refine_symmetry
        v0 = start.get_volume()
        refine_symmetry(start, symprec=float(symmetrize_tol), verbose=False)
        start.set_constraint()
        g = spacegroup(start, symprec)
        sym_info = {"tol": float(symmetrize_tol), "before": sg_input, "after": g,
                    "dV_pct": 100.0 * (start.get_volume() - v0) / v0}
        if not same_group(sg_input, g):
            ch = group_change(sg_input, g)
            verb = "raised" if ch == "gain" else "changed"
            line = (f"{verb} from {sg_name(sg_input)} to {sg_name(g)} by symmetrize-first ({symmetrize_tol:g} Å); "
                    f"CRYSTAL OPT will keep {sg_name(g)} and cannot restore a lower-symmetry distortion.")
            sym_info["text"] = line
            warnings.append(line)
    sg_ref = spacegroup(start, symprec)
    v_start = start.get_volume()

    params = {"charge": int(charge), "uhf": int(uhf), "keep_symmetry": bool(keep_symmetry), "symprec": symprec,
              "symmetrize_tol": symmetrize_tol, "relax_cell": bool(relax_cell), "threads": threads,
              "device": device}
    attempts: List[Dict[str, Any]] = []

    # 2. the springs baseline
    sd = TIER_DEFAULTS["springs"]
    vac = vacuum_axes(start) if relax_cell else []
    b_relax = relax_cell
    if vac and kind == "springs":
        warnings.append("the cell has vacuum (a slab or a molecule in a box): relaxing it with the spring model "
                        "shrinks the vacuum")
    elif vac:
        # the tiers refuse a cell relaxation with vacuum; the fallback must not do it either
        b_relax = False
        notes.append("springs: the cell has vacuum, so the spring fallback keeps the cell fixed")
    if kind == "springs":
        b_fmax, b_steps, b_limit = (fmax if fmax is not None else sd["fmax"]), int(steps or sd["steps"]), \
            float(time_limit if time_limit is not None else sd["time_limit"])
        lo, hi = 0.0, 1.0
    else:
        b_fmax, b_steps, b_limit = sd["fmax"], sd["steps"], sd["time_limit"]
        lo, hi = 0.0, 0.1
    t0 = time.monotonic()
    nothing = False
    try:
        base = C.clean(start, keep_symmetry=keep_symmetry, relax_cell=b_relax, symprec=symprec, fmax=b_fmax,
                       max_steps=b_steps, time_limit=b_limit,
                       should_stop=_hook(canceled, progress, b_steps, b_limit, lo, hi,
                                         "" if kind == "springs" else "springs: "))
    except C.NothingToClean as exc:
        if kind == "springs":
            raise
        nothing = True
        out = start.copy()
        base = {"atoms": out, "converged": True, "steps": 0, "energy_before": 0.0, "energy_after": 0.0,
                "timed_out": False, "n_bonds": 0, "method": "springs"}
        notes.append(f"springs: {exc}; the fallback is the unchanged structure")
    if canceled():
        raise proc.Canceled("canceled")
    b_atoms = base["atoms"]
    g_b = spacegroup(b_atoms, symprec)
    ch_b = group_change(sg_ref, g_b)
    attempts.append(_attempt("springs", "ok", "nothing to clean" if nothing else "", steps=base["steps"],
                             converged=base["converged"], dE_eV=base["energy_after"] - base["energy_before"],
                             dV_pct=100.0 * (b_atoms.get_volume() - v_start) / v_start,
                             sg_before=sg_name(sg_ref), sg_after=sg_name(g_b), sg_change=ch_b,
                             max_shift_A=_max_shift(start, b_atoms), seconds=time.monotonic() - t0))
    springs_gain = f"springs: {_gain_text(sg_ref, g_b, 'the spring model')}" if ch_b == "gain" else None

    def result_from(a_atoms, rep, ran, label, info, fell, ptier):
        res = PreoptResult(atoms=a_atoms, method_requested=method, method_ran=ran, label_ran=label,
                           attempts=attempts, fell_back=fell, report=rep,
                           params={**params, **ptier}, sg_input=sg_input, sg_ref=sg_ref,
                           sg_final=spacegroup(a_atoms, symprec), symmetrize=sym_info, tier_info=info,
                           notes=notes, warnings=warnings, nothing_to_clean=nothing and ran == "springs")
        if progress is not None:
            progress(1.0, "done")
        return res

    springs_params = {"fmax": b_fmax, "steps": b_steps, "time_limit": b_limit, "relax_cell": b_relax}
    if springs_gain and kind != "springs":
        notes.append(springs_gain)               # provenance; a warning only if the spring result is written
    if kind == "springs":
        if springs_gain:
            warnings.append(springs_gain)        # shown only when the spring result is the one written
        return result_from(b_atoms, base, "springs", SPRINGS_LABEL, {}, False, springs_params)

    # 3. the tier list
    cls = None
    if kind == "auto":
        cls = classify(start, charge)
        order = auto_order(cls, charge)
    else:
        order = [method]
        if kind == "mlip" and charge:
            warnings.append("ML potentials take no charge input; they see a neutral cell")

    reasons = []
    n_tiers = len(order)
    for k, m in enumerate(order):
        if canceled():
            raise proc.Canceled("canceled")
        try:
            tier = resolve(m, threads=threads, device=device, mlip_lease=mlip_lease, canceled=canceled, nice=nice)
        except Exception as exc:          # a broken registry or detection never takes the springs down
            tier = Tier(m, method_kind(m), m, False, str(exc))
        if not tier.ready:
            why = _first_line(tier.reason)
            if why != (tier.reason or "").strip():
                notes.append(f"{m}: {tier.reason}")
            attempts.append(_attempt(m, "refused", f"not installed: {why}" if kind == "auto" else why))
            reasons.append(attempts[-1]["reason"])
            continue
        why = refusal(tier, start, charge=charge, auto=kind == "auto", relax_cell=relax_cell, route=route,
                      device=device, explicit_time_limit=explicit_time_limit, leased=mlip_lease is not None)
        if why:
            attempts.append(_attempt(m, "refused", why))
            reasons.append(why)
            continue
        d = TIER_DEFAULTS[tier.kind]
        t_fmax = fmax if fmax is not None else d["fmax"]
        t_steps = int(steps or d["steps"])
        t_limit = float(time_limit if time_limit is not None else d["time_limit"])
        span_lo, span_hi = 0.1 + 0.9 * k / n_tiers, 0.1 + 0.9 * (k + 1) / n_tiers
        t1 = time.monotonic()
        try:
            res = C.clean(start, keep_symmetry=keep_symmetry, relax_cell=relax_cell, symprec=symprec, fmax=t_fmax,
                          max_steps=t_steps, time_limit=t_limit, method=m,
                          calc_opts={"charge": int(charge), "uhf": int(uhf)}, factory=tier.factory,
                          should_stop=_hook(canceled, progress, t_steps, t_limit, span_lo, span_hi,
                                            f"{_short(m)}: "))
        except proc.Canceled:
            raise
        except Exception as exc:
            if canceled():
                raise proc.Canceled("canceled")
            from . import mlip_client
            status = "refused" if isinstance(exc, mlip_client.MlipError) and exc.kind == "elements" else "failed"
            full = _tier_failure(tier, exc)
            if _first_line(full) != full.strip():
                notes.append(f"{m}: {full}")                  # the worker's stderr tail: provenance, not the readout
            attempts.append(_attempt(m, status, _first_line(full), seconds=time.monotonic() - t1))
            reasons.append(attempts[-1]["reason"])
            continue
        if canceled():
            raise proc.Canceled("canceled")
        a1 = res["atoms"]
        e0, e1 = float(res["energy_before"]), float(res["energy_after"])
        g1 = spacegroup(a1, symprec) if _finite(a1) else None
        ch = group_change(sg_ref, g1)
        dv = 100.0 * (a1.get_volume() - v_start) / v_start if _finite(a1) else float("nan")
        att = _attempt(m, "ok", steps=res["steps"], converged=res["converged"], dE_eV=e1 - e0, dV_pct=dv,
                       sg_before=sg_name(sg_ref), sg_after=sg_name(g1), sg_change=ch,
                       max_shift_A=_max_shift(start, a1) if _finite(a1) else None, seconds=time.monotonic() - t1)
        # 6. sanity checks, all against the prepared start
        bad = None
        if not _finite(a1) or not (math.isfinite(e0) and math.isfinite(e1)):
            bad = "non-finite positions, cell or energy"
        elif e1 > e0 + max(1e-4, 1e-6 * abs(e0)):
            bad = f"energy rose by {e1 - e0:.3g} eV"
        elif abs(dv) > 100 * DV_LIMIT:
            bad = f"volume changed by {dv:+.1f} % (more than {100 * DV_LIMIT:g} %)"
        else:
            cc = close_contacts(start, a1)
            if cc:
                bad = f"close contact {cc[0]}" + (f" (+{len(cc) - 1} more)" if len(cc) > 1 else "")
            elif ch == "gain" or (ch and keep_symmetry):
                bad = (_gain_text(sg_ref, g1) if ch == "gain" else
                       f"space group changed from {sg_name(sg_ref)} to {sg_name(g1)}")
                if ch == "gain" and not symmetrize_tol:
                    # a rattled cell relaxing toward its ideal group lands here too
                    bad += "; if the lower symmetry is only noise, symmetrize first"
            elif not res["converged"]:
                bad = ("not converged (hit the time limit)" if res.get("timed_out")
                       else "not converged (hit the step limit)")
        if bad:
            att.update(status="rejected", reason=bad)
            attempts.append(att)
            reasons.append(bad)
            continue
        if ch == "loss":
            warnings.append(f"space group lowered from {sg_name(sg_ref)} to {sg_name(g1)} (--no-symmetry)")
        attempts.append(att)
        rep = {k2: v for k2, v in res.items() if k2 != "atoms"}
        info = dict(tier.info)
        r = result_from(a1, rep, m, tier.label, info, False, {"fmax": t_fmax, "steps": t_steps, "time_limit": t_limit})
        r.classification = cls
        return r

    # 7. nothing passed: the spring result
    if not fallback:
        raise NoResult("; ".join(reasons) or "no tier ran", None)
    if springs_gain:
        warnings.append(springs_gain)
    r = result_from(b_atoms, base, "springs", SPRINGS_LABEL, {}, True, springs_params)
    r.classification = cls
    return r


# --------------------------------------------------------------------------- provenance

def provenance(r: PreoptResult, *, input_name: Optional[str] = None, input_sha256: Optional[str] = None,
               output_mode: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The provenance record of a run (the CLI's JSON sidecar, the GUI's
    document provenance). Basenames only: no absolute paths, no host names."""
    try:
        from mace import __version__ as mv
    except Exception:
        mv = None
    info = r.tier_info or {}
    p = r.params
    out = {
        "tool": "mace preopt", "mace_version": mv, "input": input_name, "input_sha256": input_sha256,
        "method_requested": r.method_requested, "method_ran": r.method_ran, "label": r.label_ran,
        "fell_back": r.fell_back, "summary": describe(r), "attempts": r.attempts,
        "code": info.get("code"), "code_version": info.get("version"), "code_module": info.get("module"), "code_license": info.get("code_license",
                                                                                              info.get("license")),
        "model": info.get("model_id"), "model_label": info.get("model_label"),
        "weights_license": info.get("weights_license"), "weights_sha256": info.get("weights_sha256"),
        "training_data": info.get("training_data"), "citation": info.get("citation"),
        "data_citation": info.get("data_citation"), "versions": info.get("versions"),
        "device": info.get("device_text") or p.get("device"), "threads": p.get("threads"),
        "charge": p["charge"],
        "uhf": p["uhf"] if r.method_ran != "gfnff" else f"{p['uhf']} (recorded only; GFN-FF ignores it)",
        "keep_symmetry": p["keep_symmetry"], "symprec": p["symprec"], "symmetrize_tol": p["symmetrize_tol"],
        "symmetrize": ({"before": sg_name(r.symmetrize["before"]), "after": sg_name(r.symmetrize["after"]),
                        "dV_pct": r.symmetrize["dV_pct"], "text": r.symmetrize.get("text")}
                       if r.symmetrize else None),
        "relax_cell": p["relax_cell"], "fmax": p.get("fmax"), "steps": p.get("steps"),
        "time_limit": p.get("time_limit"),
        "space_group": {"input": sg_name(r.sg_input), "reference": sg_name(r.sg_ref), "result": sg_name(r.sg_final)},
        "converged": (r.report or {}).get("converged"), "steps_taken": (r.report or {}).get("steps"),
        "warnings": r.warnings, "notes": r.notes, "output_cif": output_mode,
        "starting_geometry_only": True,
        "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if r.classification:
        out["auto"] = r.classification
    return out


def citation_text(r: PreoptResult) -> str:
    """The "Copy citation" text: citations plus training-data attributions."""
    info = r.tier_info or {}
    lines = []
    for c in info.get("citation") or []:
        lines.append(c["text"] + (f", doi:{c['doi']}" if c.get("doi") else ""))
    for c in info.get("data_citation") or []:
        lines.append(c["text"] + (f", doi:{c['doi']}" if c.get("doi") else ""))
    if info.get("training_data"):
        lines.append("Training data: " + "; ".join(info["training_data"]))
    return "\n".join(lines)
