"""The Cluster view's Software tab: the modules MACE would load on this
machine, read-only.

* CRYSTAL (Pcrystal and Pproperties): Submit's ``site.crystal_module()``,
  the same resolution ``mace site --json`` prints and Submit writes into the
  job scripts it generates (module list, source, note, rejected candidates).
* xtb: ``envmodules.discover`` with Clean's own minimum version (the newest
  module at or above it that loads and provides ``xtb``).

Both helpers belong to other parts of MACE (Submit's site settings and the
module discovery); this module only asks them. Without a module system, or
without those helpers, there is nothing to show and the tab stays hidden.
Nothing here loads a module into this process: the helpers probe in a
subshell and cache what they find.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


def _envmodules():
    try:
        # through sys.modules (not the package attribute), so a stand-in installed there is used
        import importlib
        return importlib.import_module("mace.utils.envmodules")
    except ImportError:
        return None


def _crystal_module() -> Optional[Callable[..., Any]]:
    try:
        from mace.submission.site import crystal_module
        return crystal_module
    except ImportError:
        return None


def _xtb_minimum():
    try:
        from mace.preopt import detect
    except ImportError:
        return None
    return getattr(detect, "XTB_MIN", None)


def module_system() -> Optional[str]:
    """"lmod", "tcl" or None, from markers only (no subprocess)."""
    em = _envmodules()
    if em is None or not hasattr(em, "module_system"):
        return None
    try:
        return em.module_system()
    except Exception:
        return None


def available() -> bool:
    """True when there is a module system and at least one helper to ask."""
    if module_system() is None:
        return False
    em = _envmodules()
    return _crystal_module() is not None or (em is not None and hasattr(em, "discover")
                                            and _xtb_minimum() is not None)


def rows(refresh: bool = False) -> Dict[str, Any]:
    """{module_system, rows}: one row per code, each {code, binaries,
    modules, module, source, note, rejected} or {code, error}."""
    system = module_system()
    out: List[Dict[str, Any]] = []
    if system is None:
        return {"module_system": None, "rows": out}
    cm = _crystal_module()
    if cm is not None:
        try:
            rec = cm(refresh=refresh).record()
            out.append({"code": "CRYSTAL", "binaries": "Pcrystal, Pproperties", **rec})
        except Exception as exc:                  # a probe that failed is shown, never raised
            out.append({"code": "CRYSTAL", "binaries": "Pcrystal, Pproperties",
                        "error": f"{type(exc).__name__}: {exc}"})
    em = _envmodules()
    minimum = _xtb_minimum()
    if em is not None and hasattr(em, "discover") and minimum is not None:
        try:
            info = em.discover("xtb", "xtb", minimum, use_cache=not refresh)
            mod = info.get("module")
            out.append({"code": "xtb", "binaries": "xtb", "modules": [mod] if mod else [], "module": mod,
                        "source": ("cached" if info.get("cached") else "discovered") if mod else "not found here",
                        "note": info.get("reason") or "",
                        "minimum": ".".join(str(x) for x in minimum),
                        "available": list(info.get("available") or [])})
        except Exception as exc:
            out.append({"code": "xtb", "binaries": "xtb", "error": f"{type(exc).__name__}: {exc}"})
    return {"module_system": system, "rows": out}
