"""DFTB+: density-functional tight binding through ase's calculator.

The engine is the external ``dftb+`` program (LGPL-3.0): ``MACE_DFTB`` if
set, else ``dftb+`` on PATH, else a ``dftbplus`` module of this machine's
module system (``mace/utils/envmodules.py``, loaded for that program only).
MACE drives it as a force engine (one DFTB+ single point per ase call, in a
private directory); optimization, frequencies and dynamics are MACE's ase
drivers, so every backend behaves the same.

Methods:

* ``skf``: SCC-DFTB with a Slater-Koster set the user has (``--skf-dir``,
  ``MACE_DFTB_SKF`` or ase's ``DFTB_PREFIX``). MACE never ships or
  downloads Slater-Koster files: the sets from dftb.org (mio, 3ob, matsci,
  pbc, ...) carry their own licenses (3ob and mio: CC BY-SA 4.0 with a
  citation requirement). When the set's README lists Hubbard derivatives
  (3ob), the run is DFTB3 with them and the set's damping exponent, as the
  set prescribes; the maximum angular momenta come from the README or the
  .skf files.
* ``gfn2`` / ``gfn1``: DFTB+'s built-in xTB Hamiltonian (DFTB+ 22 or newer,
  no Slater-Koster files).

The system charge goes to DFTB+; spin-polarized DFTB+ needs spin constants
per element, which MACE does not guess, so ``uhf`` must be 0 here (use xTB
for open shells).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from . import base

LICENSE = "LGPL-3.0 (the external dftb+ program; not bundled)"
CITATIONS = [{"text": "B. Hourahine et al., DFTB+, a software package for efficient approximate density functional "
                      "theory based atomistic simulations, J. Chem. Phys. 152, 124101 (2020)",
              "doi": "10.1063/1.5143190"}]
HINT = ("install DFTB+ (conda install -c conda-forge dftbplus), or export MACE_DFTB=/path/to/dftb+; "
        "for the skf method also a Slater-Koster set from dftb.org: export MACE_DFTB_SKF=/path/to/set")
_VERSION = re.compile(r"DFTB\+\s+(?:release\s+(\d+\.\d+(?:\.\d+)?)|development version\s*\(commit:\s*(\w+)\))", re.I)
_STATUS: Dict[str, Dict[str, Any]] = {}


def binary() -> Dict[str, Any]:
    """{path, env, source, module, version}; path None when not found."""
    key = os.environ.get("MACE_DFTB", "") + "|" + os.environ.get("PATH", "")
    if key in _STATUS:
        return _STATUS[key]
    out: Dict[str, Any] = {"path": None, "env": None, "source": None, "module": None, "version": None, "reason": ""}
    env_path = os.environ.get("MACE_DFTB")
    if env_path:
        p = os.path.abspath(os.path.expanduser(env_path))
        if os.path.isfile(p) and os.access(p, os.X_OK):
            out.update(path=p, source="MACE_DFTB")
        else:
            out["reason"] = f"MACE_DFTB={env_path} is not an executable file"
    else:
        p = shutil.which("dftb+")
        if p:
            out.update(path=p, source="PATH")
        else:
            try:
                from mace.utils import envmodules
                info = envmodules.discover("dftbplus", "dftb+", ())
                if info.get("binary"):
                    out.update(path=info["binary"], env=dict(info.get("env") or {}), source="module",
                               module=info.get("module"))
            except Exception:
                pass
            if not out["path"]:
                out["reason"] = "dftb+ was not found on PATH"
    if out["path"]:
        out["version"] = _version(out["path"], out["env"])
    _STATUS[key] = out
    return out


def _version(path: str, env: Optional[Dict[str, str]]) -> Optional[str]:
    from mace.preopt import proc
    wd = proc.private_dir("dftbver")
    try:
        p = subprocess.run([path, "--version"], cwd=str(wd), stdin=subprocess.DEVNULL, capture_output=True,
                           timeout=10, env=proc.child_env({**os.environ, **(env or {})}, OMP_NUM_THREADS="1"))
        m = _VERSION.search(p.stdout.decode(errors="replace") + p.stderr.decode(errors="replace"))
        return (m.group(1) or f"development {m.group(2)}") if m else None
    except (OSError, subprocess.TimeoutExpired):
        return None
    finally:
        proc.remove(wd)


def skf_dir(options: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    o = options or {}
    for v in (o.get("skf_dir"), os.environ.get("MACE_DFTB_SKF"), os.environ.get("DFTB_PREFIX")):
        if v:
            return Path(os.path.expanduser(str(v)))
    return None


def read_set_readme(d: Path) -> Dict[str, Any]:
    """Hubbard derivatives, the damping exponent and angular momenta from a
    dftb.org set's README (3ob lists them); {} when there is none."""
    for name in ("README", "README.md", "README.txt"):
        p = d / name
        if p.is_file():
            text = p.read_text(errors="replace")
            break
    else:
        return {}
    out: Dict[str, Any] = {}
    m = re.search(r"Hubbard derivatives.*?\n((?:\s*[A-Z][a-z]?\s*=\s*-?\d+\.\d+\s*\n)+)", text, re.S)
    if m:
        out["hubbard_derivs"] = {k: float(v) for k, v in re.findall(r"([A-Z][a-z]?)\s*=\s*(-?\d+\.\d+)", m.group(1))}
    m = re.search(r"zeta\s*=\s*(\d+(?:\.\d+)?)", text)
    if m:
        out["damping_exponent"] = float(m.group(1))
    m = re.search(r"Maximum angular momenta.*?\n((?:\s*[A-Z][a-z]?\s*=\s*\"[spdf]\"\s*\n)+)", text, re.S)
    if m:
        out["max_l"] = dict(re.findall(r"([A-Z][a-z]?)\s*=\s*\"([spdf])\"", m.group(1)))
    return out


def _kpts(atoms, spec, per: int):
    k = (spec.options or {}).get("kpts")
    if k:
        k = [int(x) for x in (k.split(",") if isinstance(k, str) else k)]
        return tuple(k) if len(k) == 3 else (k[0],) * 3
    pbc = [bool(x) for x in atoms.pbc]
    if not any(pbc):
        return None                                                    # a molecule: a cluster, no k-points
    density = float((spec.options or {}).get("kpoint_density", 20.0))   # Å: n_i = ceil(density / |a_i|)
    lens = atoms.cell.lengths()
    if all(pbc) and per == 0:
        return (1, 1, 1)                                               # a molecule in a 3D box
    if all(pbc) and per < 3:
        # a slab or polymer in a 3D box: one k-point along its vacuum axes, as with its pbc
        from mace.preopt.core import vacuum_axes
        vac = set(vacuum_axes(atoms))
        pbc = [i not in vac for i in range(3)]
    # one k-point along a non-periodic axis (a slab or polymer with its pbc, boxed for DFTB+)
    return tuple(int(max(1, np.ceil(density / L))) if p else 1 for L, p in zip(lens, pbc))


class DFTBBackend(base.Backend):
    name = "dftb"
    label = "DFTB+"
    family = "tight-binding"
    license = LICENSE
    capabilities = base.Capabilities(frozenset({"sp", "opt", "freq", "md", "charges"}), frozenset({0, 1, 2, 3}),
                                     frozenset({"energy", "forces", "stress", "charges"}), needs_charge_input=True)
    citation = CITATIONS

    def methods(self) -> List[base.Method]:
        return [base.Method("skf", "SCC-DFTB with your Slater-Koster set (DFTB3 for 3ob)", default=True,
                            license=LICENSE + "; Slater-Koster files: their own license",
                            citation=list(CITATIONS), note="Cite the Slater-Koster set too (its README)"),
                base.Method("gfn2", "GFN2-xTB Hamiltonian in DFTB+", license=LICENSE, citation=list(CITATIONS)),
                base.Method("gfn1", "GFN1-xTB Hamiltonian in DFTB+", license=LICENSE, citation=list(CITATIONS))]

    def status(self, method: Optional[str] = None, recheck: bool = False, options=None) -> Dict[str, Any]:
        if recheck:
            _STATUS.clear()
        b = binary()
        if not b["path"]:
            return {"available": False, "status": "not_found", "reason": b["reason"], "hint": HINT,
                    "commands": ['conda install -c conda-forge dftbplus'], "version": None}
        if (method or self.default_method()) == "skf":
            d = skf_dir(options)
            if d is None or not d.is_dir() or not any(d.glob("*-*.skf")):
                why = "no Slater-Koster set given" if d is None else f"{d} has no .skf files"
                return {"available": False, "status": "needs_skf", "reason": why, "version": b["version"],
                        "hint": "download a set from dftb.org (e.g. 3ob, mio, matsci, pbc), then "
                                "export MACE_DFTB_SKF=/path/to/set or pass --skf-dir",
                        "commands": ["export MACE_DFTB_SKF=/path/to/set"]}
        return {"available": True, "status": "ready", "reason": "", "hint": "", "version": b["version"],
                "source": b["source"], "module": b["module"]}

    def calculator(self, atoms, spec: base.CalcSpec, canceled: Optional[Callable[[], bool]] = None):
        from ase.calculators.dftb import Dftb
        from mace.preopt import proc
        from .tasks import periodicity
        m = self.method(spec.method).id
        o = spec.options or {}
        if int(spec.uhf):
            raise base.CalcError("spin-polarized DFTB+ needs per-element spin constants, which MACE does not "
                                 "guess: run open shells with xTB, or set uhf 0", kind="usage")
        b = binary()
        if not b["path"]:
            raise base.CalcError(f"DFTB+: {b['reason']}", kind="unavailable", hint=HINT)
        per = periodicity(atoms)
        params: Dict[str, Any] = {}
        if m == "skf":
            d = skf_dir(o)
            st = self.status("skf", options=o)
            if not st["available"]:
                raise base.CalcError(f"DFTB+: {st['reason']}", kind="unavailable", hint=st.get("hint", ""))
            els = sorted(set(atoms.get_chemical_symbols()))
            missing = [f"{a}-{c}" for a in els for c in els if not (d / f"{a}-{c}.skf").is_file()]
            if missing:
                raise base.CalcError(f"the Slater-Koster set {d} has no {', '.join(missing[:6])}"
                                     + (" ..." if len(missing) > 6 else "") + ".skf", kind="unavailable")
            info = read_set_readme(d)
            params.update(Hamiltonian_SCC="Yes", Hamiltonian_SCCTolerance=float(o.get("scc_tolerance", 1e-7)),
                          Hamiltonian_MaxSCCIterations=int(o.get("max_scc", 200)),
                          Hamiltonian_Charge=float(spec.charge))
            if info.get("max_l"):
                params["Hamiltonian_MaxAngularMomentum_"] = ""
                for e in els:
                    if e in info["max_l"]:
                        params[f"Hamiltonian_MaxAngularMomentum_{e}"] = f'"{info["max_l"][e]}"'
            hd = info.get("hubbard_derivs") or {}
            if hd and all(e in hd for e in els) and not o.get("no_third_order"):
                params["Hamiltonian_ThirdOrderFull"] = "Yes"
                params["Hamiltonian_HubbardDerivs_"] = ""
                for e in els:
                    params[f"Hamiltonian_HubbardDerivs_{e}"] = hd[e]
                if "H" in els:
                    params["Hamiltonian_HCorrection_"] = "Damping"
                    params["Hamiltonian_HCorrection_Exponent"] = info.get("damping_exponent", 4.0)
                # ase writes parser version 1, whose conversion breaks HubbardDerivs; 8 reads ase's keywords
                # (Analysis/CalculateForces) and the DFTB3 block as written here
                params["ParserOptions_ParserVersion"] = 8
            slako = str(d) + "/"
        else:
            params.update(Hamiltonian_="xTB", Hamiltonian_Method="GFN2-xTB" if m == "gfn2" else "GFN1-xTB",
                          Hamiltonian_Charge=float(spec.charge))
            slako = None
        for k, v in (o.get("hsd") or {}).items():          # expert: raw ase Dftb keywords
            params[str(k)] = v
        wd = proc.private_dir("dftb")
        threads = int(spec.threads or proc.default_threads())
        envs = " ".join(f"{k}={v}" for k, v in {**(b["env"] or {}), "OMP_NUM_THREADS": threads}.items()
                        if k in ("OMP_NUM_THREADS", "PATH", "LD_LIBRARY_PATH"))
        cmd = f"env {envs} {b['path']} > PREFIX.out 2> PREFIX.err"
        calc = Dftb(directory=str(wd), label="dftb", command=cmd, kpts=_kpts(atoms, spec, per),
                    **({"slako_dir": slako} if slako else {}), **params)
        calc.mace_workdir = wd
        calc.mace_info = {"code_version": b["version"], "source": b["source"], "module": b["module"],
                          "kpts": _kpts(atoms, spec, per), "third_order": params.get("Hamiltonian_ThirdOrderFull") == "Yes",
                          "skf_dir": str(skf_dir(o)) if m == "skf" else None}
        orig_close = getattr(calc, "close", None)

        def close():
            if orig_close:
                try:
                    orig_close()
                except Exception:
                    pass
            proc.remove(wd)
        calc.close = close
        from mace.preopt import lowdim
        if lowdim.is_low_dim(atoms) and any(atoms.pbc):
            calc.mace_info["boxed_vacuum_A"] = lowdim.VACUUM
            return lowdim.Boxed(calc)                 # DFTB+ needs a 3D cell: vacuum along the open axes
        return calc

    @staticmethod
    def report(calc) -> Dict[str, Any]:
        return dict(getattr(calc, "mace_info", {}) or {})
