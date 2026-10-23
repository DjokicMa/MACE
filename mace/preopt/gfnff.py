"""GFN-FF through the ``xtb`` program (6.7 or newer), as an ase calculator.

One xtb run per force call (``xtb POSCAR --gfnff --grad``), about 0.13 s:

* energy from the last ``TOTAL ENERGY ... Eh`` line of xtb's output;
* forces from ``gradient`` (Turbomole ``$grad``, Eh/Bohr);
* stress from ``gradlatt`` (dE/dL per lattice vector, Eh/Bohr):
  sigma = sym(L^T G) / V, in ase's Voigt order [xx, yy, zz, yz, xz, xy].

The topology is rebuilt for every call: xtb 6.7.1 reuses a topology file
that is wrong once an atom crosses a cell face (Si moved across a face came
out 30 eV off), so ``gfnff_topo`` and its companions are deleted first.

Each call runs in this calculator's private work directory (0700, removed by
``close()``), in its own session, with one OpenMP thread, its own TMPDIR
(xtb's own temp files go with the work directory) and the coreutils
``timeout --kill-after=2`` wrapper, so an xtb orphaned by a killed parent
still stops after the per-call timeout. ``canceled()`` (pure) is polled
every 50 ms; a cancel stops the process group within 2 s.

xtb is LGPL-3.0 and only ever run as a separate program; it is not bundled.

``method`` (1.1.6, for ``mace calc``; default ``gfnff``, the Clean tier as
before): ``gfn2`` or ``gfn1`` run GFN2-xTB / GFN1-xTB instead. A periodic
cell goes through xtb's tblite library (``--tblite``, energy, forces and the
lattice gradient); a molecule (``periodic=False``, any method) is written as
an xyz file and run as a molecule. Both xTB methods also leave the atomic charges in
xtb's ``charges`` file, which become the ``charges`` result.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Callable, Dict, List, Optional

import numpy as np

from . import proc

PER_CALL_TIMEOUT = 60.0
_TOPOLOGY = ("gfnff_topo", "gfnff_charges", "gfnff_adjacency")
_RESULTS = ("gradient", "gradlatt", "energy")
_IGNORED = ("Could not read topology file",)

METHODS = ("gfnff", "gfn2", "gfn1")
XTB_CITATIONS = {
    "gfn2": [{"text": "C. Bannwarth, S. Ehlert and S. Grimme, GFN2-xTB: an accurate and broadly parametrized "
                      "self-consistent tight-binding quantum chemical method, J. Chem. Theory Comput. 15, 1652 (2019)",
              "doi": "10.1021/acs.jctc.8b01176"}],
    "gfn1": [{"text": "S. Grimme, C. Bannwarth and P. Shushkov, A robust and accurate tight-binding quantum chemical "
                      "method for structures, vibrational frequencies, and noncovalent interactions of large molecular "
                      "systems parametrized for all spd-block elements (Z = 1-86), J. Chem. Theory Comput. 13, 1989 "
                      "(2017)", "doi": "10.1021/acs.jctc.7b00118"}],
}

CITATIONS = [
    {"text": "S. Spicher and S. Grimme, Robust atomistic modeling of materials, organometallic, and biochemical "
             "systems, Angew. Chem. Int. Ed. 59, 15665 (2020)", "doi": "10.1002/anie.202004239"},
    {"text": "C. Bannwarth et al., Extended tight-binding quantum chemistry methods, WIREs Comput. Mol. Sci. 11, "
             "e1493 (2021)", "doi": "10.1002/wcms.1493"},
]


class GFNFFError(RuntimeError):
    """xtb could not do this structure (or crashed)."""


def parse_total_energy(stdout: str) -> float:
    """The last ``TOTAL ENERGY  <x> Eh`` of xtb's output, in Eh."""
    m = re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+(?:[EeDd][-+]?\d+)?)\s+Eh", stdout)
    if not m:
        raise GFNFFError("xtb printed no TOTAL ENERGY")
    return float(m[-1].replace("D", "E").replace("d", "e"))


def parse_turbomole_block(text: str, n: int) -> np.ndarray:
    """The gradient rows of the last ``cycle`` block of a Turbomole
    ``$grad``/``$gradlatt`` file: ``n`` coordinate rows, then ``n`` gradient
    rows (Fortran ``D`` exponents allowed). Returns an n x 3 array."""
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if "cycle" in ln]
    if not starts:
        raise GFNFFError("no cycle block in the gradient file")
    body = [ln for ln in lines[starts[-1] + 1:] if ln.strip() and not ln.lstrip().startswith("$")]
    if len(body) < 2 * n:
        raise GFNFFError(f"the gradient block has {len(body)} rows, expected {2 * n}")
    rows = body[n:2 * n]
    try:
        return np.array([[float(x.replace("D", "E").replace("d", "e")) for x in r.split()[:3]] for r in rows])
    except ValueError as exc:
        raise GFNFFError(f"unreadable gradient row: {exc}")


def stress_from_gradlatt(L: np.ndarray, G: np.ndarray, V: float) -> np.ndarray:
    """sigma = sym(L^T G)/V in Voigt order [xx, yy, zz, yz, xz, xy] (eV/Å^3
    for L in Å, G in eV/Å, V in Å^3). G[i] = dE/dL[i], the derivative of the
    energy with respect to lattice vector i at fixed fractional coordinates."""
    s = np.asarray(L).T @ np.asarray(G) / V
    s = 0.5 * (s + s.T)
    return np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])


def error_lines(text: str) -> List[str]:
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith(("-1-", "-2-", "-3-")) and not any(k in s for k in _IGNORED):
            out.append(s)
    return out


def _base():
    try:
        from ase.calculators.calculator import Calculator, all_changes
    except ImportError:
        return object, ()
    return Calculator, all_changes


_Calculator, _ALL = _base()


class GFNFF(_Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress", "charges"]

    def __init__(self, xtb: str = "xtb", charge: int = 0, uhf: int = 0,
                 canceled: Optional[Callable[[], bool]] = None, per_call_timeout: float = PER_CALL_TIMEOUT,
                 nice: int = 0, env: Optional[Dict[str, str]] = None, method: str = "gfnff",
                 periodic: bool = True, threads: int = 1, **kw):
        super().__init__(**kw)
        if method not in METHODS:
            raise ValueError(f"unknown xtb method {method!r}: use {', '.join(METHODS)}")
        self.xtb, self.charge, self.uhf = xtb, int(charge), int(uhf)
        self.method, self.periodic, self.threads = method, bool(periodic), max(1, int(threads))
        # an automatically loaded xtb module's variables (PATH, LD_LIBRARY_PATH): for xtb's process only
        self.extra_env = dict(env or {})
        self.canceled = canceled
        self.per_call_timeout = float(per_call_timeout)
        self.nice = int(nice)
        self.workdir = proc.private_dir("gfnff")
        self.ncalls = 0
        self.argv_seen: List[str] = []
        proc.raise_stack_limit()

    def close(self) -> None:
        if self.workdir is not None:
            proc.remove(self.workdir)
            self.workdir = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def command(self, fn: str) -> List[str]:
        if self.method == "gfnff":
            how = ["--gfnff"]
        else:
            how = ["--gfn", self.method[3:]] + (["--tblite"] if self.periodic else [])
        cmd = [self.xtb, fn] + how + ["--grad", "--chrg", str(self.charge), "--uhf", str(self.uhf)]
        wrapper = shutil.which("timeout")
        if wrapper:
            cmd = [wrapper, "--kill-after=2", f"{self.per_call_timeout:g}"] + cmd
        if self.nice > 0 and shutil.which("nice"):
            cmd = [shutil.which("nice"), "-n", str(self.nice)] + cmd
        return cmd

    def calculate(self, atoms=None, properties=("energy",), system_changes=_ALL):
        from ase.io import write
        from ase.units import Bohr, Hartree
        super().calculate(atoms, properties, system_changes)
        if self.workdir is None:
            raise GFNFFError("the GFN-FF calculator was closed")
        a = self.atoms
        wd = str(self.workdir)
        for f in _TOPOLOGY + _RESULTS + ("charges",):        # a fresh topology for every call
            p = os.path.join(wd, f)
            if os.path.exists(p):
                os.remove(p)
        tmp = os.path.join(wd, "tmp")
        os.makedirs(tmp, exist_ok=True)
        if not self.periodic:
            fn = "mol.xyz"
            write(os.path.join(wd, fn), a, format="xyz")
        else:
            fn = "POSCAR"
            write(os.path.join(wd, fn), a, format="vasp", direct=True, sort=False)
        k = str(self.threads)
        env = proc.child_env({**os.environ, **self.extra_env}, OMP_NUM_THREADS=k, OMP_STACKSIZE="512M", OMP_MAX_ACTIVE_LEVELS="1",
                             MKL_NUM_THREADS=k, OPENBLAS_NUM_THREADS=k, TMPDIR=tmp)
        cmd = self.command(fn)
        self.argv_seen = cmd
        with open(os.path.join(wd, "xtb.out"), "w") as out, open(os.path.join(wd, "xtb.err"), "w") as err:
            p = subprocess.Popen(cmd, cwd=wd, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                 start_new_session=True)
            try:
                rc = proc.wait_child(p, self.canceled, self.per_call_timeout + 5.0)
            except TimeoutError:
                raise GFNFFError(f"xtb gave no answer within {self.per_call_timeout:g} s")
        self.ncalls += 1
        stdout = open(os.path.join(wd, "xtb.out"), errors="replace").read()
        stderr = open(os.path.join(wd, "xtb.err"), errors="replace").read()
        grad = os.path.join(wd, "gradient")
        if rc != 0 or not os.path.exists(grad):
            errs = error_lines(stdout + "\n" + stderr)
            why = "segfault" if rc in (-11, 139) else f"exit code {rc}" if rc else "no gradient written"
            if self.method != "gfnff":
                name = "GFN2-xTB" if self.method == "gfn2" else "GFN1-xTB"
                raise GFNFFError(f"{name} failed: xtb {why}" + (": " + " | ".join(errs[-3:]) if errs else "")
                                 + (" (a crash in xtb's periodic tblite path: try GFN1-xTB, or the tblite "
                                    "Python package)" if why == "segfault" and self.periodic else ""))
            raise GFNFFError("GFN-FF can't type this structure (common for MgO-like oxides and halides): "
                             f"xtb {why}" + (": " + " | ".join(errs[-3:]) if errs else ""))
        e = parse_total_energy(stdout) * Hartree
        n = len(a)
        g = parse_turbomole_block(open(grad).read(), n) * Hartree / Bohr
        self.results["energy"] = self.results["free_energy"] = e
        self.results["forces"] = -g
        chg = os.path.join(wd, "charges")
        if os.path.exists(chg):
            try:
                q = np.array([float(x.split()[0]) for x in open(chg).read().split("\n") if x.strip()])
                if q.shape == (n,):
                    self.results["charges"] = q
            except (ValueError, IndexError):
                pass
        lat = os.path.join(wd, "gradlatt")
        if not self.periodic:
            if "stress" in properties:
                raise GFNFFError("a molecule has no stress")
        elif os.path.exists(lat):
            G = parse_turbomole_block(open(lat).read(), 3) * Hartree / Bohr
            self.results["stress"] = stress_from_gradlatt(a.cell.array, G, a.get_volume())
        elif "stress" in properties:
            raise GFNFFError("xtb wrote no lattice gradient (gradlatt): no stress for a cell relaxation")
