"""xTB: GFN2-xTB, GFN1-xTB and GFN-FF.

The engine is the external ``xtb`` program (LGPL-3.0, 6.7 or newer), found
exactly as Clean's GFN-FF tier finds it (``MACE_XTB``, PATH, or this
machine's module system: ``mace.preopt.detect.xtb_status``) and run by the
same calculator (``mace.preopt.gfnff.GFNFF``, one private directory, one
process group per call, cancelable). Periodic GFN1/GFN2 go through xtb's
tblite library (``--tblite``); a molecule (no periodic axis without vacuum)
is run as a molecule.

``options["engine"] = "tblite"`` uses the tblite Python package (LGPL-3.0)
in-process instead, when it is installed in MACE's own interpreter. It is
never the default: a crash inside a library takes the whole process down,
while a crashed xtb program only fails its own call.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from . import base

METHODS = {"gfn2": "GFN2-xTB", "gfn1": "GFN1-xTB", "gfnff": "GFN-FF"}
LICENSE = "LGPL-3.0 (the external xtb program; not bundled)"


def _tblite_available() -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec("tblite") is not None and importlib.util.find_spec("tblite.ase") is not None
    except (ImportError, ValueError):
        return False


class XTBBackend(base.Backend):
    name = "xtb"
    label = "xTB (GFN2-xTB, GFN1-xTB, GFN-FF)"
    family = "tight-binding"
    license = LICENSE
    capabilities = base.Capabilities(frozenset({"sp", "opt", "freq", "md", "charges"}), frozenset({0, 1, 2, 3}),
                                     frozenset({"energy", "forces", "stress", "charges"}), needs_charge_input=True)

    def methods(self) -> List[base.Method]:
        from mace.preopt import gfnff
        return [base.Method("gfn2", "GFN2-xTB", default=True, license=LICENSE,
                            citation=gfnff.XTB_CITATIONS["gfn2"] + gfnff.CITATIONS[1:]),
                base.Method("gfn1", "GFN1-xTB", license=LICENSE, citation=gfnff.XTB_CITATIONS["gfn1"] + gfnff.CITATIONS[1:]),
                base.Method("gfnff", "GFN-FF (force field)", license=LICENSE, citation=list(gfnff.CITATIONS))]

    def status(self, method: Optional[str] = None, recheck: bool = False, options=None) -> Dict[str, Any]:
        from mace.preopt import detect
        try:
            st = detect.xtb_status(recheck=recheck)
        except Exception as exc:            # detection never breaks a listing
            st = {"status": "broken", "reason": str(exc), "hint": detect.XTB_HINT}
        ok = st.get("status") == "ready"
        out = {"available": ok, "status": st.get("status"), "reason": st.get("reason", ""),
               "hint": st.get("hint", "") if not ok else "", "version": st.get("version"),
               "commands": detect.hint_commands(st.get("hint", "")) if not ok else [],
               "source": st.get("source"), "module": st.get("module")}
        if not ok and method in ("gfn1", "gfn2") and _tblite_available():
            out.update(available=True, status="ready", reason="", hint="", commands=[],
                       source="tblite (Python package)", note="xtb not found; the tblite package runs it in-process")
        return out

    def calculator(self, atoms, spec: base.CalcSpec, canceled: Optional[Callable[[], bool]] = None):
        from mace.preopt import detect, gfnff, proc
        from .tasks import periodicity
        m = self.method(spec.method).id
        o = spec.options or {}
        engine = str(o.get("engine", "xtb")).lower()
        st = detect.xtb_status()
        if engine == "tblite" or (st.get("status") != "ready" and m != "gfnff" and _tblite_available()):
            if m == "gfnff":
                raise base.CalcError("GFN-FF runs through the xtb program only", kind="usage")
            if not _tblite_available():
                raise base.CalcError("the tblite Python package is not installed in this interpreter "
                                     "(conda install -c conda-forge tblite-python)", kind="unavailable")
            from tblite.ase import TBLite
            calc = TBLite(method=METHODS[m], charge=float(spec.charge), multiplicity=int(spec.uhf) + 1, verbosity=0,
                          **({"electronic_temperature": float(o["etemp"])} if o.get("etemp") else {}))
            calc.mace_engine = "tblite"
            return calc
        if st.get("status") != "ready":
            raise base.CalcError(f"xtb: {st.get('reason') or st.get('status')}", kind="unavailable",
                                 hint=detect.xtb_hint(st))
        per = periodicity(atoms)
        n = len(atoms)
        timeout = float(o.get("per_call_timeout") or (60.0 if m == "gfnff" else max(120.0, 1.0 * n)))
        threads = int(spec.threads or proc.default_threads())
        calc = gfnff.GFNFF(xtb=st["path"], charge=int(spec.charge), uhf=int(spec.uhf), canceled=canceled,
                           per_call_timeout=timeout, env=st.get("env"), method=m, periodic=per > 0,
                           threads=threads)
        calc.mace_engine = "xtb"
        calc.mace_info = {"code_version": st.get("version"), "source": st.get("source"), "module": st.get("module"),
                          "periodic": per > 0, "threads": threads}
        from mace.preopt import lowdim
        if per == 0 and any(atoms.pbc):
            # a molecule in a periodic box runs without the cell: made whole first (it may be wrapped across a face)
            return lowdim.Whole(calc)
        if lowdim.is_low_dim(atoms) and any(atoms.pbc):
            # a slab or polymer with its pbc: xtb gets a copy with lowdim.VACUUM Å along the open axes
            calc.mace_info["boxed_vacuum_A"] = lowdim.VACUUM
            return lowdim.Boxed(calc)
        return calc

    @staticmethod
    def report(calc) -> Dict[str, Any]:
        out = {"engine": getattr(calc, "mace_engine", None)}
        out.update(getattr(calc, "mace_info", {}) or {})
        if getattr(calc, "ncalls", None) is not None:
            out["xtb_calls"] = calc.ncalls
        return out
