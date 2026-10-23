"""The calculator backends by name: a plain dict, imported lazily.

``get("uff")`` imports ``uff.py`` on first use and caches one instance;
nothing heavy is imported until a backend's ``calculator()`` runs (xtb and
dftb+ are external programs, the ML potentials live in their own
interpreter). ``register`` adds or replaces one (tests; a 1.1.7 backend
may register itself the same way).
"""
from __future__ import annotations

import importlib
from typing import Dict, List

from .base import Backend, CalcError

# name -> (module, class); the order is the order of every listing
BUILTIN = {
    "uff": ("uff", "UFFBackend"),
    "eqeq": ("eqeq", "EQeqBackend"),
    "xtb": ("xtb", "XTBBackend"),
    "dftb": ("dftb", "DFTBBackend"),
    "mlip": ("mlip", "MLIPBackend"),
}
ALIASES = {"uff4mof": ("uff", "uff4mof"), "gfn2": ("xtb", "gfn2"), "gfn1": ("xtb", "gfn1"), "gfnff": ("xtb", "gfnff"),
           "gfn2-xtb": ("xtb", "gfn2"), "gfn1-xtb": ("xtb", "gfn1"), "gfn-ff": ("xtb", "gfnff"),
           "dftb+": ("dftb", None), "mace": ("mlip", None)}

_EXTRA: Dict[str, Backend] = {}
_CACHE: Dict[str, Backend] = {}


def register(backend: Backend) -> None:
    _EXTRA[backend.name] = backend
    _CACHE.pop(backend.name, None)


def names() -> List[str]:
    return list(BUILTIN) + [n for n in _EXTRA if n not in BUILTIN]


def get(name: str) -> Backend:
    name = str(name or "").lower()
    if name in _EXTRA:
        return _EXTRA[name]
    if name in _CACHE:
        return _CACHE[name]
    if name not in BUILTIN:
        raise CalcError(f"no calculator backend {name!r}; known: {', '.join(names())}", kind="usage")
    mod, cls = BUILTIN[name]
    be = getattr(importlib.import_module(f"{__package__}.{mod}"), cls)()
    _CACHE[name] = be
    return be


def resolve(backend: str, method=None):
    """(backend name, method) from what a user typed: "uff4mof", "gfn2",
    "mlip:<model id>", "xtb" + "gfn2", ..."""
    b = str(backend or "").strip().lower()
    if b.startswith("mlip:") and len(b) > 5:
        return "mlip", backend.split(":", 1)[1]
    if b in ALIASES and b not in BUILTIN:
        name, m = ALIASES[b]
        return name, method or m
    return b, method


def all_backends() -> List[Backend]:
    return [get(n) for n in names()]
