"""The common interface of every calculator backend.

A backend is one code (UFF, EQeq, xTB, DFTB+, an ML potential family) behind
the same small surface, so the task drivers (``tasks.py``), the CLI (``mace
calc``) and the GUI's Calculators panel never branch on the code:

* ``name``, ``label``, ``family`` (forcefield, charges, tight-binding, mlip),
  ``license`` and ``capabilities`` (the tasks and periodicities it takes);
* ``methods()``: what the backend offers (xTB: gfn2, gfn1, gfnff; an ML
  family: its models), each a ``Method``;
* ``status(method)``: is it installed and ready here, and if not, why and
  what to run (never installs or downloads anything itself);
* ``calculator(atoms, spec)``: an ase calculator for one run (``close()`` it
  when done; the task drivers do), or ``charges(atoms, spec)`` for a charge
  model, which has no energy surface.

1.1.7 generalizes the code backends (``mace/backends``); this interface is
kept to what a ``Backend.capabilities`` / ``write_input``-free calculator
needs so it can sit behind that protocol unchanged: no CRYSTAL concepts, no
scheduler, plain dicts and dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, NamedTuple, Optional

TASKS = ("sp", "opt", "freq", "md", "charges")
TASK_LABELS = {"sp": "Single point", "opt": "Geometry optimization", "freq": "Frequencies (finite differences)",
               "md": "Molecular dynamics", "charges": "Partial charges"}


class CalcError(RuntimeError):
    """A calculation that could not run or failed (the message is for the user)."""

    def __init__(self, message: str, kind: str = "failed", hint: str = ""):
        super().__init__(message)
        self.kind = kind
        self.hint = hint


class Canceled(Exception):
    """The caller's cancel predicate turned true."""


class Capabilities(NamedTuple):
    tasks: FrozenSet[str]                 # subset of TASKS
    periodicities: FrozenSet[int]         # 0 (molecule) .. 3 (bulk)
    properties: FrozenSet[str]            # energy, forces, stress, charges
    needs_charge_input: bool = False      # takes the system charge (xTB, DFTB+, EQeq)


@dataclass
class Method:
    id: str                               # e.g. "gfn2", "acesuit-mace-mpa-0-medium", "uff4mof"
    label: str
    default: bool = False
    note: str = ""                        # e.g. "MACE-MP (mace-torch), unrelated to this MACE tool"
    license: str = ""
    citation: List[Dict[str, str]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "label": self.label, "default": self.default, "note": self.note,
                "license": self.license, "citation": list(self.citation),
                **self.extra}


@dataclass
class CalcSpec:
    """Everything one run needs (the CLI's flags, the GUI panel's fields)."""
    backend: str
    method: Optional[str] = None
    task: str = "sp"
    device: str = "auto"                  # auto | cpu | cuda:N  (auto: GPU0 when there is one)
    threads: Optional[int] = None
    charge: int = 0
    uhf: int = 0                          # unpaired electrons (xTB, DFTB+); spin multiplicity = uhf + 1
    charge_set: bool = False              # charge/uhf given (CLI flag, GUI field), not the defaults: a model that
    uhf_set: bool = False                 # takes a charge and spin otherwise reads the structure's file (chargespin)
    # opt
    fmax: float = 0.02                    # eV/Å
    steps: int = 500
    relax_cell: bool = False
    keep_symmetry: bool = False
    optimizer: str = "fire"               # fire | bfgs | lbfgs
    # freq
    displacement: float = 0.01            # Å, central differences
    temperature: float = 298.15           # K, thermochemistry and MD
    # md
    ensemble: str = "nvt"                 # nve | nvt | npt
    timestep: float = 1.0                 # fs
    md_steps: int = 1000
    friction: float = 0.01                # 1/fs, Langevin
    pressure: float = 1.01325             # bar, NPT
    interval: int = 10                    # steps between saved frames
    seed: Optional[int] = None
    time_limit: Optional[float] = None    # s, wall clock for the whole run
    options: Dict[str, Any] = field(default_factory=dict)   # backend-specific (kpts, skf_dir, charges, ...)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class Backend:
    """Base class: subclasses set the class attributes and override the hooks."""
    name = ""
    label = ""
    family = ""
    license = ""
    capabilities = Capabilities(frozenset(), frozenset(), frozenset())
    citation: List[Dict[str, str]] = []

    def methods(self) -> List[Method]:
        return [Method(self.name, self.label, default=True, license=self.license, citation=list(self.citation))]

    def default_method(self) -> str:
        ms = self.methods()
        return next((m.id for m in ms if m.default), ms[0].id)

    def method(self, mid: Optional[str]) -> Method:
        mid = mid or self.default_method()
        for m in self.methods():
            if m.id == mid:
                return m
        raise CalcError(f"{self.label}: no method {mid!r}; known: {', '.join(m.id for m in self.methods())}",
                        kind="usage")

    def status(self, method: Optional[str] = None, recheck: bool = False,
               options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """{available, status, reason, hint, version}; never raises.
        ``options``: the run's backend options (a DFTB+ Slater-Koster folder)."""
        return {"available": True, "status": "ready", "reason": "", "hint": "", "version": None}

    def calculator(self, atoms, spec: CalcSpec, canceled: Optional[Callable[[], bool]] = None):
        raise CalcError(f"{self.label} gives no energy surface", kind="usage")

    def charges(self, atoms, spec: CalcSpec, canceled: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
        """{charges: [...], ...} for a charge model; energy backends that
        report charges (xTB, DFTB+) answer through their calculator."""
        raise CalcError(f"{self.label} computes no partial charges on its own", kind="usage")

    def provenance(self, method: Optional[str] = None) -> Dict[str, Any]:
        m = self.method(method)
        return {"backend": self.name, "backend_label": self.label, "method": m.id, "method_label": m.label,
                "license": m.license or self.license,
                "citation": list(m.citation or self.citation), "note": m.note}

    def describe(self, recheck: bool = False) -> Dict[str, Any]:
        """The row ``mace calc list`` and the GUI's backend list show."""
        out = {"name": self.name, "label": self.label, "family": self.family, "license": self.license,
               "tasks": sorted(self.capabilities.tasks, key=TASKS.index),
               "periodicities": sorted(self.capabilities.periodicities),
               "properties": sorted(self.capabilities.properties),
               "needs_charge_input": self.capabilities.needs_charge_input,
               "default_method": None, "methods": []}
        try:
            out["default_method"] = self.default_method()
            for m in self.methods():
                row = m.as_dict()
                try:
                    st = self.status(m.id, recheck=recheck)
                except Exception as exc:          # a broken probe never breaks the listing
                    st = {"available": False, "status": "broken", "reason": str(exc), "hint": ""}
                row.update(available=bool(st.get("available")), status=st.get("status"),
                           reason=st.get("reason", ""), hint=st.get("hint", ""), version=st.get("version"),
                           commands=list(st.get("commands") or []))
                out["methods"].append(row)
        except Exception as exc:
            out["error"] = str(exc)
        out["available"] = any(m.get("available") for m in out["methods"])
        return out
