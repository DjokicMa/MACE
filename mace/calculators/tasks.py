"""The task drivers: single point, optimization, frequencies, MD and charges.

``run(atoms, spec)`` is the one entry point of the CLI (``mace calc run``)
and the GUI's Calculators panel (a pool job): it resolves the backend,
checks it is ready, builds its calculator, runs the task with the caller's
``progress(fraction, text)`` and pure ``canceled()`` callbacks and an
optional wall-clock limit, closes the calculator, and returns a
``CalcResult``. ``write_outputs`` writes the files of a result.

Every driver is plain ase: FIRE/BFGS/LBFGS with an optional cell filter and
``FixSymmetry``; central finite differences for the Hessian (Gamma-point
phonons of the cell as given, or molecular vibrations), harmonic
thermochemistry; Velocity Verlet (NVE), Langevin (NVT) and Berendsen (NPT)
dynamics. A force field or ML geometry is a result of that model, not of
CRYSTAL: an optimized structure written for CRYSTAL carries the
``# mace-preopt:`` marker, so ``mace convert`` warns before a non-OPT deck.
"""
from __future__ import annotations

import datetime
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from . import base, registry

Progress = Optional[Callable[[float, str], None]]


@dataclass
class CalcResult:
    task: str
    backend: str
    method: str
    atoms: Any = None                          # the final structure (opt, md), else the input
    energy: Optional[float] = None             # eV
    forces: Optional[np.ndarray] = None        # eV/Å
    stress: Optional[np.ndarray] = None        # eV/Å^3, Voigt
    charges: Optional[np.ndarray] = None       # e
    converged: Optional[bool] = None
    steps: Optional[int] = None
    frequencies: Optional[np.ndarray] = None   # cm^-1 (negative: imaginary)
    modes: Optional[np.ndarray] = None         # (3N, N, 3) mass-weighted normal modes, Cartesian displacements
    thermo: Dict[str, Any] = field(default_factory=dict)
    md: Dict[str, Any] = field(default_factory=dict)
    trajectory: List[Any] = field(default_factory=list)   # opt/md frames kept in memory
    log: List[Dict[str, float]] = field(default_factory=list)
    report: Dict[str, Any] = field(default_factory=dict)  # backend-specific (UFF types, xtb version, ...)
    warnings: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    files: Dict[str, str] = field(default_factory=dict)
    stopped: Optional[str] = None              # "time limit", "canceled" (partial result kept)

    @property
    def fmax(self) -> Optional[float]:
        if self.forces is None:
            return None
        return float(np.sqrt((np.asarray(self.forces) ** 2).sum(axis=1)).max()) if len(self.forces) else 0.0

    def summary(self) -> str:
        lab = base.TASK_LABELS.get(self.task, self.task)
        parts = [f"{lab} with {self.provenance.get('method_label') or self.method}"]
        if self.energy is not None:
            n = len(self.atoms) if self.atoms is not None else 0
            parts.append(f"E = {self.energy:.6f} eV" + (f" ({self.energy / n:.6f} eV/atom)" if n else ""))
        if self.task == "opt":
            parts.append(f"{'converged' if self.converged else 'not converged'} in {self.steps} steps"
                         + (f", fmax {self.fmax:.4f} eV/Å" if self.fmax is not None else ""))
        if self.frequencies is not None:
            imag = int((self.frequencies < -20.0).sum())
            parts.append(f"{len(self.frequencies)} modes" + (f", {imag} imaginary (< -20 cm^-1)" if imag else ""))
        if self.task == "md" and self.md:
            parts.append(f"{self.md.get('steps_done')} steps, <T> = {self.md.get('mean_temperature_K', 0):.1f} K")
        if self.charges is not None:
            parts.append(f"charges sum {float(np.sum(self.charges)):+.4f} e")
        if self.stopped:
            parts.append(f"stopped ({self.stopped})")
        return "; ".join(parts)

    def as_dict(self, full: bool = True) -> Dict[str, Any]:
        """JSON-ready (the CLI's report file, the GUI's answer)."""
        a = self.atoms
        out: Dict[str, Any] = {
            "task": self.task, "backend": self.backend, "method": self.method, "summary": self.summary(),
            "energy_eV": self.energy, "energy_per_atom_eV": (self.energy / len(a)) if (self.energy is not None and a is not None and len(a)) else None,
            "fmax_eV_A": self.fmax, "converged": self.converged, "steps": self.steps,
            "stress_eV_A3": None if self.stress is None else [float(x) for x in self.stress],
            "pressure_GPa": None if self.stress is None else float(-np.mean(np.asarray(self.stress)[:3]) * 160.21766208),
            "charges": None if self.charges is None else [round(float(x), 6) for x in self.charges],
            "frequencies_cm1": None if self.frequencies is None else [round(float(x), 3) for x in self.frequencies],
            "thermo": self.thermo or None, "md": self.md or None, "report": self.report,
            "warnings": list(self.warnings), "provenance": self.provenance, "seconds": round(self.seconds, 3),
            "files": dict(self.files), "stopped": self.stopped,
        }
        if a is not None:
            out["natoms"] = len(a)
            out["formula"] = a.get_chemical_formula()
            out["symbols"] = a.get_chemical_symbols()
            if full:
                out["structure"] = {"cell": a.cell.array.tolist(), "positions": a.positions.tolist(),
                                    "numbers": [int(z) for z in a.numbers], "pbc": [bool(p) for p in a.pbc]}
        if full and self.forces is not None:
            out["forces_eV_A"] = np.round(np.asarray(self.forces), 6).tolist()
        if self.log:
            out["log"] = self.log[-2000:]
        return out


# --------------------------------------------------------------------------- helpers

class _Clock:
    def __init__(self, spec, canceled, progress):
        self.t0 = time.monotonic()
        self.limit = spec.time_limit
        self.canceled = canceled or (lambda: False)
        self.progress = progress
        self.last = 0.0

    def stop_reason(self) -> Optional[str]:
        if self.canceled():
            return "canceled"
        if self.limit and time.monotonic() - self.t0 > self.limit:
            return "time limit"
        return None

    def report(self, frac: float, text: str, force: bool = False) -> None:
        if self.progress is None:
            return
        now = time.monotonic()
        if force or now - self.last > 0.25:
            self.last = now
            self.progress(max(0.0, min(1.0, frac)), text)


def periodicity(atoms) -> int:
    """Periodic axes with real material along them. A slab, polymer or
    molecule given with its pbc (a SLAB/POLYMER/MOLECULE deck, an extended
    XYZ with pbc="T T F", the GUI's non-3D documents) has its periodic axes;
    in a cell periodic in 3D, an axis with a vacuum gap of 5 Å or more counts
    as non-periodic (a molecule or slab in a box)."""
    pbc = [bool(p) for p in atoms.pbc]
    if not any(pbc) or atoms.cell.rank < 3:
        return 0
    if not all(pbc):
        return sum(pbc)
    try:
        from mace.preopt.core import vacuum_axes
        vac = set(vacuum_axes(atoms))
    except Exception:
        vac = set()
    return sum(1 for i, p in enumerate(pbc) if p and i not in vac)


def _optimizer(name: str):
    from ase import optimize
    name = (name or "fire").lower()
    cls = {"fire": optimize.FIRE, "bfgs": optimize.BFGS, "lbfgs": optimize.LBFGS}.get(name)
    if cls is None:
        raise base.CalcError(f"optimizer is fire, bfgs or lbfgs, not {name!r}", kind="usage")
    return cls


def _cell_filter(atoms, mask=None):
    try:
        from ase.filters import FrechetCellFilter
        return FrechetCellFilter(atoms, mask=mask)
    except ImportError:                 # ase < 3.23
        from ase.constraints import ExpCellFilter
        return ExpCellFilter(atoms, mask=mask)


def _props(atoms, calc, want_stress: bool):
    e = float(atoms.get_potential_energy())
    f = np.asarray(atoms.get_forces(), float)
    s = None
    if want_stress:
        try:
            s = np.asarray(atoms.get_stress(voigt=True), float)
        except Exception:
            s = None
    q = None
    try:
        if "charges" in getattr(calc, "results", {}):
            q = np.asarray(calc.results["charges"], float)
    except Exception:
        q = None
    return e, f, s, q


def _check_finite(e, f):
    if not math.isfinite(e) or not np.all(np.isfinite(f)):
        raise base.CalcError("the calculator returned NaN or Inf", kind="nan")


# --------------------------------------------------------------------------- drivers

def single_point(atoms, calc, spec, clock) -> CalcResult:
    a = atoms.copy()
    a.calc = calc
    clock.report(0.1, "computing", force=True)
    e, f, s, q = _props(a, calc, periodicity(a) == 3)
    _check_finite(e, f)
    return CalcResult("sp", spec.backend, spec.method, atoms=a, energy=e, forces=f, stress=s, charges=q)


def optimize(atoms, calc, spec, clock) -> CalcResult:
    a = atoms.copy()
    a.calc = calc
    warnings = []
    if spec.keep_symmetry:
        from ase.constraints import FixSymmetry
        a.set_constraint(FixSymmetry(a))
    obj = a
    if spec.relax_cell:
        from mace.preopt import lowdim
        if lowdim.is_low_dim(a):
            mask = lowdim.strain_mask(a) if any(a.pbc) else None
            if mask is None:
                warnings.append("a molecule keeps its box (no cell to relax)" if not any(a.pbc) else
                                "the periodic directions do not lie along Cartesian axes: the cell is kept fixed")
            else:
                obj = _cell_filter(a, mask)
                warnings.append("only the periodic directions of the cell relax")
        elif periodicity(a) < 3:
            warnings.append("the cell has vacuum (a slab or a molecule in a box): the cell is kept fixed")
        else:
            obj = _cell_filter(a)
    opt = _optimizer(spec.optimizer)(obj, logfile=None)
    frames, log = [a.copy()], []
    steps, converged, stopped = 0, False, None
    fmax = float(spec.fmax)
    for converged in opt.irun(fmax=fmax, steps=int(spec.steps)):
        e = float(a.get_potential_energy())
        fm = float(np.sqrt((obj.get_forces() ** 2).sum(axis=1)).max())
        log.append({"step": steps, "energy_eV": e, "fmax_eV_A": fm})
        if not math.isfinite(e) or not math.isfinite(fm):
            raise base.CalcError(f"the optimization blew up at step {steps} (non-finite energy or forces)", kind="nan")
        if steps % 5 == 0:
            frames.append(a.copy())
        clock.report(min(0.95, steps / max(1, spec.steps)), f"step {steps}, E {e:.5f} eV, fmax {fm:.4f} eV/Å")
        if converged:
            break
        stopped = clock.stop_reason()
        if stopped:
            break
        steps += 1
    steps = int(getattr(opt, "nsteps", steps))
    final = a.copy()
    final.set_constraint()
    frames.append(final.copy())
    final.calc = calc
    e, f, s, q = _props(final, calc, periodicity(final) == 3)
    r = CalcResult("opt", spec.backend, spec.method, atoms=final, energy=e, forces=f, stress=s, charges=q,
                   converged=bool(converged), steps=steps, trajectory=frames, log=log, warnings=warnings,
                   stopped=stopped)
    r.report["optimizer"] = spec.optimizer
    r.report["relax_cell"] = bool(spec.relax_cell and obj is not a)
    r.report["keep_symmetry"] = bool(spec.keep_symmetry)
    if len(atoms):
        v0, v1 = atoms.get_volume() if atoms.cell.rank == 3 else None, final.get_volume() if final.cell.rank == 3 else None
        if v0 and v1:
            r.report["volume_change_pct"] = round(100.0 * (v1 - v0) / v0, 4)
    return r


def _freq_factor() -> float:
    """cm^-1 per sqrt(eigenvalue in eV/(Å^2 amu))."""
    from ase import units
    return float(np.sqrt(units._e / (units._amu * 1e-20)) / (2 * np.pi * units._c * 100.0))


def frequencies(atoms, calc, spec, clock) -> CalcResult:
    a = atoms.copy()
    a.calc = calc
    n = len(a)
    h = float(spec.displacement)
    if not h > 0:
        raise base.CalcError("the displacement must be positive", kind="usage")
    e0, f0, s0, q0 = _props(a, calc, periodicity(a) == 3)
    _check_finite(e0, f0)
    H = np.zeros((3 * n, 3 * n))
    total = 6 * n
    done = 0
    for i in range(n):
        for x in range(3):
            fs = []
            for sgn in (1, -1):
                b = a.copy()
                b.positions[i, x] += sgn * h
                b.calc = calc
                f = np.asarray(b.get_forces(), float)
                if not np.all(np.isfinite(f)):
                    raise base.CalcError("non-finite forces in a displaced structure", kind="nan")
                fs.append(f)
                done += 1
            H[3 * i + x] = -(fs[0] - fs[1]).ravel() / (2 * h)
            why = clock.stop_reason()
            if why:
                raise base.CalcError(f"frequencies stopped ({why}) after {done} of {total} displacements; "
                                     "nothing partial is kept", kind="canceled" if why == "canceled" else "timeout")
            clock.report(done / total, f"displacement {done} of {total}")
    H = 0.5 * (H + H.T)
    m = np.repeat(a.get_masses(), 3)
    Hm = H / np.sqrt(np.outer(m, m))
    w, v = np.linalg.eigh(Hm)
    f = _freq_factor()
    freq = np.sign(w) * np.sqrt(np.abs(w)) * f
    modes = (v / np.sqrt(m)[:, None]).T.reshape(3 * n, n, 3)
    per = periodicity(a)
    n_zero = 3 if per == 3 else (6 if n > 2 else 5)
    if per in (1, 2):
        n_zero = 3
    thermo, warns = _thermo(freq, n_zero, spec.temperature)
    r = CalcResult("freq", spec.backend, spec.method, atoms=a, energy=e0, forces=f0, stress=s0, charges=q0,
                   frequencies=freq, modes=modes, thermo=thermo, warnings=warns)
    r.report["displacement_A"] = h
    r.report["what"] = ("Gamma-point phonons of this cell (use a supercell for other q points)" if per == 3
                        else "vibrations of the structure as given (translations and rotations near zero)")
    r.report["fmax_at_reference_eV_A"] = float(np.sqrt((f0 ** 2).sum(axis=1)).max())
    if r.report["fmax_at_reference_eV_A"] > 0.05:
        r.warnings.append(f"the structure is not at a minimum (fmax {r.report['fmax_at_reference_eV_A']:.3f} eV/Å): "
                          "optimize it first for meaningful frequencies")
    return r


def _thermo(freq_cm, n_zero: int, T: float):
    """Harmonic ZPE, vibrational free energy, entropy and heat capacity from
    the modes above the n_zero smallest (by magnitude); imaginary modes are
    left out and reported."""
    from ase import units
    order = np.argsort(np.abs(freq_cm))
    keep = np.sort(order[n_zero:]) if len(freq_cm) > n_zero else np.array([], int)
    real = freq_cm[keep]
    warns = []
    imag = real[real < 0]
    if len(imag):
        warns.append(f"{len(imag)} imaginary mode(s) (down to {imag.min():.1f} cm^-1) left out of the thermochemistry")
    nu = real[real > 1e-6]
    e = nu * units.invcm                     # eV
    zpe = float(0.5 * e.sum())
    out = {"zpe_eV": zpe, "temperature_K": float(T), "modes_used": int(len(nu)), "modes_left_out": int(n_zero)}
    if T > 0 and len(e):
        kT = units.kB * T
        x = e / kT
        F = float((kT * np.log1p(-np.exp(-x))).sum())
        U = float((e / np.expm1(x)).sum())
        S = (U - F) / T
        Cv = float((units.kB * x * x * np.exp(x) / np.expm1(x) ** 2).sum())
        out.update(vib_free_energy_eV=zpe + F, vib_energy_eV=zpe + U, vib_entropy_eV_K=S, vib_heat_capacity_eV_K=Cv)
    return out, warns


def molecular_dynamics(atoms, calc, spec, clock) -> CalcResult:
    from ase import units
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
    a = atoms.copy()
    a.calc = calc
    per = periodicity(a)
    rng = np.random.default_rng(spec.seed)
    T = float(spec.temperature)
    if T > 0:
        MaxwellBoltzmannDistribution(a, temperature_K=T, rng=rng)
        Stationary(a)
        if per == 0:
            ZeroRotation(a)
    dt = float(spec.timestep) * units.fs
    ens = (spec.ensemble or "nvt").lower()
    warns = []
    if ens == "nve":
        from ase.md.verlet import VelocityVerlet
        dyn = VelocityVerlet(a, dt)
    elif ens == "nvt":
        from ase.md.langevin import Langevin
        dyn = Langevin(a, dt, temperature_K=T, friction=float(spec.friction) / units.fs, rng=rng)
    elif ens == "npt":
        if per < 3:
            raise base.CalcError("NPT needs a cell periodic in 3D without vacuum", kind="usage")
        from ase.md.nptberendsen import NPTBerendsen
        comp = float((spec.options or {}).get("compressibility_per_bar", 4.57e-5))
        dyn = NPTBerendsen(a, dt, temperature_K=T, pressure_au=float(spec.pressure) * units.bar,
                           taut=100 * units.fs, taup=1000 * units.fs, compressibility_au=comp / units.bar)
    else:
        raise base.CalcError(f"ensemble is nve, nvt or npt, not {spec.ensemble!r}", kind="usage")
    nsteps = int(spec.md_steps)
    every = max(1, int(spec.interval))
    frames, log = [a.copy()], []
    temps = []
    stopped = None
    done = 0
    for k in range(nsteps):
        dyn.run(1)
        done = k + 1
        ep = float(a.get_potential_energy())
        ek = float(a.get_kinetic_energy())
        if not (math.isfinite(ep) and math.isfinite(ek)):
            raise base.CalcError(f"the dynamics blew up at step {done} (non-finite energy)", kind="nan")
        tk = ek / (1.5 * units.kB * len(a)) if len(a) else 0.0
        temps.append(tk)
        if done % every == 0 or done == nsteps:
            row = {"step": done, "time_fs": done * float(spec.timestep), "epot_eV": ep, "ekin_eV": ek,
                   "etot_eV": ep + ek, "temperature_K": tk}
            if per == 3:
                row["volume_A3"] = float(a.get_volume())
                try:
                    row["pressure_GPa"] = float(-np.mean(a.get_stress(include_ideal_gas=True)[:3]) / units.GPa)
                except Exception:
                    pass
            log.append(row)
            f = a.copy()
            f.set_velocities(a.get_velocities())
            frames.append(f)
        clock.report(done / nsteps, f"step {done} of {nsteps}, T {tk:.0f} K, Epot {ep:.4f} eV")
        stopped = clock.stop_reason()
        if stopped:
            break
    final = a.copy()
    final.calc = calc
    e, fr, s, q = _props(final, calc, per == 3)
    tail = temps[len(temps) // 5:] or temps
    md = {"ensemble": ens, "timestep_fs": float(spec.timestep), "steps_requested": nsteps, "steps_done": done,
          "temperature_target_K": T, "mean_temperature_K": float(np.mean(tail)) if tail else None,
          "frames": len(frames), "interval": every, "seed": spec.seed}
    if ens == "nvt":
        md["friction_per_fs"] = float(spec.friction)
    if ens == "npt":
        md["pressure_bar"] = float(spec.pressure)
    if log:
        drift = log[-1]["etot_eV"] - log[0]["etot_eV"]
        md["etot_drift_eV"] = float(drift)
        if ens == "nve" and len(a) and abs(drift) / len(a) > 0.01:
            warns.append(f"total energy drifted {drift:.3f} eV in NVE: use a smaller time step")
    return CalcResult("md", spec.backend, spec.method, atoms=final, energy=e, forces=fr, stress=s, charges=q,
                      md=md, trajectory=frames, log=log, warnings=warns, stopped=stopped, steps=done)


# --------------------------------------------------------------------------- the entry point

def setup(spec: base.CalcSpec, recheck: bool = False) -> Dict[str, Any]:
    """Is ``spec`` runnable here? {ready, reason, hint, backend, method, status}."""
    be = registry.get(spec.backend)
    m = be.method(spec.method)
    if spec.task not in be.capabilities.tasks:
        return {"ready": False, "reason": f"{be.label} does not do {base.TASK_LABELS.get(spec.task, spec.task)}",
                "hint": "", "backend": be.name, "method": m.id, "status": "unsupported"}
    st = be.status(m.id, recheck=recheck, options=spec.options)
    return {"ready": bool(st.get("available")), "reason": st.get("reason", ""), "hint": st.get("hint", ""),
            "commands": st.get("commands") or [], "backend": be.name, "method": m.id, "status": st.get("status"),
            "version": st.get("version")}


DRIVERS = {"sp": single_point, "opt": optimize, "freq": frequencies, "md": molecular_dynamics}


def run(atoms, spec: base.CalcSpec, *, progress: Progress = None, canceled: Optional[Callable[[], bool]] = None,
        check_ready: bool = True) -> CalcResult:
    """Run ``spec.task`` on a copy of ``atoms``. Raises CalcError (with
    ``kind``) when it cannot run or fails, base.Canceled on a cancel before
    any result exists."""
    be = registry.get(spec.backend)
    m = be.method(spec.method)
    spec.method = m.id
    if spec.task not in base.TASKS:
        raise base.CalcError(f"task is one of {', '.join(base.TASKS)}, not {spec.task!r}", kind="usage")
    if spec.task not in be.capabilities.tasks:
        raise base.CalcError(f"{be.label} does not do {base.TASK_LABELS[spec.task].lower()}", kind="usage")
    if len(atoms) == 0:
        raise base.CalcError("the structure has no atoms", kind="usage")
    if check_ready:
        st = be.status(m.id, options=spec.options)
        if not st.get("available"):
            raise base.CalcError(f"{m.label}: {st.get('reason') or st.get('status')}", kind="unavailable",
                                 hint=st.get("hint", ""))
    clock = _Clock(spec, canceled, progress)
    t0 = time.monotonic()
    warns: List[str] = []
    eqeq_uff = be.name == "uff" and (spec.options or {}).get("charges") == "eqeq"     # its EQeq charges take it
    takes_cs = False
    if be.name == "mlip":                  # OrbMol-v2, UMA (omol): the charge goes to the model
        from . import mlip as _mlip
        mm = _mlip.model(spec.method) if spec.method else None
        takes_cs = bool(mm and mm.get("charge_spin"))
    if spec.charge and not be.capabilities.needs_charge_input and not eqeq_uff and not takes_cs:
        warns.append(f"{be.label} takes no charge input: the charge {spec.charge:+d} is recorded only")
    if spec.task == "charges" and be.name == "eqeq":
        clock.report(0.05, "EQeq: building the charge matrix", force=True)
        res = be.charges(atoms, spec, canceled=canceled)
        r = CalcResult("charges", be.name, m.id, atoms=atoms.copy(), charges=np.asarray(res["charges"]),
                       report={k: v for k, v in res.items() if k != "charges"}, warnings=list(res.get("warnings") or []))
    else:
        clock.report(0.0, f"starting {m.label}", force=True)
        calc = be.calculator(atoms, spec, canceled=canceled)
        try:
            if spec.task == "charges":
                r = single_point(atoms, calc, spec, clock)
                r.task = "charges"
                if r.charges is None:
                    raise base.CalcError(f"{m.label} reports no atomic charges", kind="usage")
            else:
                r = DRIVERS[spec.task](atoms, calc, spec, clock)
            rep = getattr(be, "report", None)
            if rep is not None:
                try:
                    r.report.update(rep(calc) or {})
                except Exception:
                    pass
        except _runtime_failures() as exc:
            # the backend's own runtime failure (xtb's SCF did not converge, DFTB+ stopped, the ML worker
            # failed): a message for the user, not a traceback; the cause stays chained for --debug
            raise _backend_failure(be, m, spec, calc, exc) from exc
        finally:
            close = getattr(calc, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass
    r.backend, r.method = be.name, m.id
    for a in [r.atoms] + list(r.trajectory):     # no calculator (a socket, a worker) travels with a result
        if a is not None:
            a.calc = None
    r.warnings = warns + r.warnings
    r.seconds = time.monotonic() - t0
    r.provenance = provenance(be, m.id, spec, r)
    clock.report(1.0, "done", force=True)
    return r


def _runtime_failures() -> tuple:
    """The exceptions a backend raises when its calculation fails at run time (not a MACE bug)."""
    from ase.calculators.calculator import CalculatorError    # DFTB+ and tblite: CalculationFailed, ReadError
    from mace.preopt.gfnff import GFNFFError
    from mace.preopt.mlip_client import MlipError
    return (GFNFFError, MlipError, CalculatorError)


# the log files a backend leaves in its private directory (removed when the calculator closes)
_LOG_FILES = ("xtb.out", "xtb.err", "dftb.out", "dftb.err", "detailed.out", "dftb_in.hsd", "worker.log")


def _keep_logs(calc, backend: str) -> Optional[Path]:
    """Copy a failed backend's log files out of its private directory into a kept one; None when it has none."""
    import shutil
    import tempfile
    from mace.preopt import proc
    dirs = []
    for obj in (calc, getattr(calc, "owner", None)):     # a boxed calculator passes the lookups to the one inside
        for attr in ("mace_workdir", "workdir"):
            d = getattr(obj, attr, None) if obj is not None else None
            if d and Path(d).is_dir() and Path(d) not in dirs:
                dirs.append(Path(d))
    logs = [d / n for d in dirs for n in _LOG_FILES if (d / n).is_file()]
    if not logs:
        return None
    try:
        keep = Path(tempfile.mkdtemp(prefix=f"failed-{backend}-", dir=str(proc.temp_root())))
        for f in logs:
            shutil.copy2(f, keep / f.name)
    except (OSError, RuntimeError):
        return None
    return keep


def _log_errors(folder: Path, n: int = 3) -> List[str]:
    """The program's own error lines in the kept logs (DFTB+: ERROR! then "-> why")."""
    import re
    out: List[str] = []
    for f in sorted(folder.iterdir()):
        if f.suffix not in (".out", ".err"):
            continue
        for ln in f.read_text(errors="replace").splitlines():
            s = ln.strip()
            if re.match(r"(->|ERROR\b|\[ERROR\])", s) and s.rstrip("!") != "ERROR":
                s = s.lstrip("->").strip()
                if s and s not in out:
                    out.append(s)
    return out[-n:]


def _backend_failure(be, m, spec: base.CalcSpec, calc, exc: Exception) -> base.CalcError:
    """The CalcError for a backend's runtime failure: what failed, the backend's own message, where its log is."""
    import re
    from ase.calculators.calculator import CalculatorError
    what = base.TASK_LABELS.get(spec.task, spec.task).lower()
    msg = str(exc).strip() or type(exc).__name__
    kind = getattr(exc, "kind", None) if be.name == "mlip" else None
    if kind:
        msg = f"{kind}: {msg}"
    logs = _keep_logs(calc, be.name)
    if logs is not None and isinstance(exc, CalculatorError):
        # ase's text names the command and the private directory (gone by now): the exit code and the
        # program's own error lines say more
        code = re.search(r"error code (-?\d+)", msg)
        errs = _log_errors(logs)
        if code or errs:
            msg = (f"{be.label} stopped" + (f" (exit code {code.group(1)})" if code else "")
                   + (": " + " | ".join(errs) if errs else ""))
    text = f"{m.label} failed during the {what}: {msg}"
    if logs is not None:
        text += f"\nIts log files were kept in {logs}"
    return base.CalcError(text, kind=kind or "failed")


def provenance(be, method: str, spec: base.CalcSpec, r: CalcResult) -> Dict[str, Any]:
    try:
        from mace import __version__ as mv
    except Exception:
        mv = None
    p = be.provenance(method)
    p.update({"tool": "mace calc", "mace_version": mv, "task": spec.task,
              "settings": {k: v for k, v in spec.as_dict().items() if k not in ("backend", "method", "task")},
              "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    for k in ("device_text", "versions", "code_version"):
        if r.report.get(k) is not None:
            p[k] = r.report[k]
    return p


# --------------------------------------------------------------------------- files

def tag(spec: base.CalcSpec) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9.+-]+", "-", f"{spec.backend}-{spec.method}")


def marker_method(r: CalcResult) -> str:
    return f"mlip:{r.method}" if r.backend == "mlip" else f"calc:{r.backend}:{r.method}"


def write_charges_cif(path: Path, atoms, charges, header: List[str]) -> None:
    """A P1 CIF with ``_atom_site_charge`` (what RASPA and Zeo++ read)."""
    from collections import Counter
    c = atoms.cell.cellpar()
    lines = [*header, f"data_{path.stem}", "_symmetry_space_group_name_H-M    'P 1'", "_symmetry_Int_Tables_number       1",
             f"_cell_length_a    {c[0]:.6f}", f"_cell_length_b    {c[1]:.6f}", f"_cell_length_c    {c[2]:.6f}",
             f"_cell_angle_alpha {c[3]:.6f}", f"_cell_angle_beta  {c[4]:.6f}", f"_cell_angle_gamma {c[5]:.6f}",
             "loop_", "_symmetry_equiv_pos_as_xyz", "x,y,z", "loop_", "_atom_site_label", "_atom_site_type_symbol",
             "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z", "_atom_site_charge"]
    seen: Counter = Counter()
    frac = atoms.get_scaled_positions(wrap=True)
    for s, f, q in zip(atoms.get_chemical_symbols(), frac, charges):
        seen[s] += 1
        lines.append(f"{s}{seen[s]:<5d} {s:3s} {f[0]:.6f} {f[1]:.6f} {f[2]:.6f} {float(q):+.6f}")
    path.write_text("\n".join(lines) + "\n")


def write_outputs(r: CalcResult, outdir: Path, stem: str, spec: base.CalcSpec, *, force: bool = False,
                  spacegroup=None, kinds=None, check: Optional[Callable[[Path], None]] = None) -> Dict[str, str]:
    """Write the result's files; returns {kind: file name}. Refuses to
    overwrite unless ``force``. ``check(path)`` (the GUI's check_write) runs
    for every file before anything is written and may raise."""
    outdir = Path(outdir)
    t = tag(spec)
    base_name = f"{stem}_{t}_{r.task}"
    planned = {"report": outdir / f"{base_name}.json"}
    if r.task == "opt" and r.atoms is not None:
        if r.atoms.pbc.all():
            planned["structure"] = outdir / f"{base_name}.cif"
        else:                           # a slab, polymer or molecule: exact (pbc kept) and boxed for CRYSTAL
            planned["structure"] = outdir / f"{base_name}.extxyz"
            planned["structure_cif"] = outdir / f"{base_name}.cif"
        planned["trajectory"] = outdir / f"{base_name}.traj"
    if r.task == "md":
        planned["trajectory"] = outdir / f"{base_name}.traj"
        planned["frames"] = outdir / f"{base_name}.extxyz"
        planned["log"] = outdir / f"{base_name}_log.csv"
    if r.task == "freq":
        planned["modes"] = outdir / f"{base_name}_modes.extxyz"
    if r.charges is not None and r.atoms is not None and r.atoms.pbc.all() and r.atoms.cell.rank == 3:
        planned["charges_cif"] = outdir / f"{stem}_{t}_charges.cif"
    elif r.charges is not None and r.atoms is not None:
        planned["charges_xyz"] = outdir / f"{stem}_{t}_charges.extxyz"
    if check is not None:
        for p in planned.values():
            check(p)
    exists = [p.name for p in planned.values() if p.exists()]
    if exists and not force:
        raise FileExistsError(f"{', '.join(exists)} already exist in {outdir} (pass --force to overwrite)")
    outdir.mkdir(parents=True, exist_ok=True)
    from ase.io import write
    label = r.provenance.get("method_label") or r.method
    lic = r.provenance.get("license")
    if "structure" in planned:
        from mace.preopt import cifout, marker
        header = [f"# Optimized with {label} ({lic}) by mace calc. A force-field/ML geometry, not a CRYSTAL "
                  "result: run a CRYSTAL OPT from it before SP/FREQ/BAND/DOSS.",
                  marker.line(marker_method(r), r.method if r.backend == "mlip" else None,
                              academic=bool(r.provenance.get("academic")))]
        p = planned["structure"]
        if p.suffix == ".cif":
            mode = cifout.write(p, r.atoms, name=stem, header=header, spacegroup=spacegroup, kinds=kinds,
                                keep_group=bool(spec.keep_symmetry))
            r.report["cif"] = mode
        else:
            from mace.gui.structures import SAVE_VACUUM, boxed
            out = r.atoms.copy()
            out.calc, out.info = None, {}
            write(str(p), out, format="extxyz")
            dim = ("molecule", "polymer", "slab")[sum(bool(x) for x in r.atoms.pbc)]
            r.report["cif"] = cifout.write(planned["structure_cif"], boxed(r.atoms), name=stem, header=header,
                                           keep_group=False,
                                           p1_reason=f"a {dim}: written in a box with {SAVE_VACUUM:g} Å of vacuum")
    if "trajectory" in planned and r.trajectory:
        from ase.io.trajectory import Trajectory
        with Trajectory(str(planned["trajectory"]), "w") as tr:
            for f in r.trajectory:
                tr.write(f)
    if "frames" in planned and r.trajectory:
        write(str(planned["frames"]), r.trajectory, format="extxyz")
    if "log" in planned and r.log:
        keys = list(r.log[0].keys())
        rows = [",".join(keys)] + [",".join(f"{row.get(k, ''):.8g}" if isinstance(row.get(k), float) else str(row.get(k, ''))
                                            for k in keys) for row in r.log]
        planned["log"].write_text("\n".join(rows) + "\n")
    if "modes" in planned and r.modes is not None:
        frames = []
        for k, (nu, mode) in enumerate(zip(r.frequencies, r.modes)):
            f = r.atoms.copy()
            f.calc = None
            f.info = {"mode": k, "frequency_cm1": float(nu)}
            nrm = np.linalg.norm(mode)
            f.new_array("mode", mode / nrm if nrm > 0 else mode)
            frames.append(f)
        write(str(planned["modes"]), frames, format="extxyz")
    if "charges_xyz" in planned:
        out = r.atoms.copy()
        out.calc, out.info = None, {}
        out.set_initial_charges(np.asarray(r.charges, float))
        write(str(planned["charges_xyz"]), out, format="extxyz")
    if "charges_cif" in planned:
        write_charges_cif(planned["charges_cif"], r.atoms, r.charges,
                          [f"# Partial charges from {label} by mace calc ({lic})."])
    files = {k: p.name for k, p in planned.items() if k != "report"}
    files["report"] = planned["report"].name
    r.files = files
    planned["report"].write_text(json.dumps(r.as_dict(), indent=2, default=_json_default) + "\n")
    return files


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
