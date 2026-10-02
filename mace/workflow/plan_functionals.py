"""Functional names stored in a workflow plan, checked against CRYSTAL23.

Plans saved by MACE 1.1.1 stored the planner menus' prose spellings
("HSEsol-3C", "HF-3C", "M06-2X", "HF", ...) and could append "-D3" to any
non-3c choice. 1.1.1 never read an SP step's ``custom_functional``, so those
names did no harm; since 1.1.2 the engine writes them into the deck, and
CRYSTAL23 rejects them. This module turns a legacy spelling into the CRYSTAL23
keyword, drops a "-D3" suffix CRYSTAL23 has no parameters for (with a warning),
and refuses a name it does not know. A name that is already valid is returned
exactly as written.
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Crystal_d12"))
try:
    from d12_constants import (  # noqa: E402
        D3_FUNCTIONALS, crystal23_functional_keyword, mace_functional_name,
    )
except ImportError:  # no keyword tables: plans are used as written
    D3_FUNCTIONALS = None

# Prose spellings the 1.1.1 planner menus offered -> CRYSTAL23 keyword.
LEGACY_SPELLINGS = {
    "HF": "RHF",
    "HF-3C": "HF3C",
    "HFSOL-3C": "HFSOL3C",
    "PBEH-3C": "PBEH3C",
    "HSE-3C": "HSE3C",
    "B97-3C": "B973C",
    "PBESOL0-3C": "PBESOL03C",
    "HSESOL-3C": "HSESOL3C",
    "M06-2X": "M062X",
    "M06-L": "M06L",
    "M06-HF": "M06HF",
}

# Plan keys that hold a functional name, inside these step sub-dicts.
FUNCTIONAL_KEYS = ("custom_functional", "new_functional")
METHOD_SECTIONS = ("method_modifications", "method_settings")


class UnknownFunctionalError(ValueError):
    """A plan names a functional CRYSTAL23 does not accept."""


def _base_name(name: str) -> Optional[str]:
    """The name MACE or CRYSTAL23 uses for ``name`` (no -D3), else None."""
    return (mace_functional_name(name)
            or crystal23_functional_keyword(name, allow_hf=True))


def _d3_ok(base: str) -> bool:
    return base in D3_FUNCTIONALS or crystal23_functional_keyword(f"{base}-D3") is not None


def _is_valid(name: str) -> bool:
    if name.upper().endswith("-D3"):
        if crystal23_functional_keyword(name) is not None:
            return True
        base = _base_name(name[:-3])
        return base is not None and _d3_ok(base)
    return _base_name(name) is not None


def normalize_functional(name: Any) -> Tuple[Any, Optional[str]]:
    """Return ``(name CRYSTAL23 accepts, message or None)``.

    A valid name comes back unchanged. Raises UnknownFunctionalError for a
    name that is neither valid nor a known legacy spelling.
    """
    if D3_FUNCTIONALS is None or not isinstance(name, str) or not name.strip():
        return name, None
    raw = name.strip()
    if _is_valid(raw):
        return name, None
    base, d3 = raw, False
    if base.upper().endswith("-D3"):
        base, d3 = base[:-3], True
    base = LEGACY_SPELLINGS.get(base.upper(), base)
    keyword = _base_name(base)
    if keyword is None:
        raise UnknownFunctionalError(
            f"functional '{name}' is not a CRYSTAL23 keyword MACE knows; "
            f"re-plan this step with a functional from the planner's menu")
    if d3 and _d3_ok(keyword):
        return f"{keyword}-D3", f"functional '{name}' written as '{keyword}-D3'"
    if d3:
        why = ("a 3c method carries its own dispersion correction"
               if keyword.upper().endswith("3C")
               else f"CRYSTAL23 has no -D3 parameters for {keyword}")
        return keyword, (f"functional '{name}': {why}, so the -D3 suffix is "
                         f"dropped and '{keyword}' is written")
    return keyword, f"functional '{name}' written as '{keyword}'"


def normalize_step_functionals(step: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Normalise one step config in place; return (messages, errors).

    An unknown name is left as written and reported in ``errors``.
    """
    messages: List[str] = []
    errors: List[str] = []
    if not isinstance(step, dict):
        return messages, errors
    for section in METHOD_SECTIONS:
        mods = step.get(section)
        if not isinstance(mods, dict):
            continue
        for key in FUNCTIONAL_KEYS:
            if key not in mods:
                continue
            try:
                new, msg = normalize_functional(mods[key])
            except UnknownFunctionalError as exc:
                errors.append(str(exc))
                continue
            mods[key] = new
            if msg:
                messages.append(msg)
    return messages, errors


def normalize_plan_functionals(plan: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Normalise every step of a workflow plan in place; return (messages, errors)."""
    messages: List[str] = []
    errors: List[str] = []
    if not isinstance(plan, dict):
        return messages, errors
    for key, step in (plan.get("step_configurations") or {}).items():
        msgs, errs = normalize_step_functionals(step)
        messages += [f"{key}: {m}" for m in msgs]
        errors += [f"{key}: {e}" for e in errs]
    return messages, errors


def step_functional_errors(step: Dict[str, Any]) -> List[str]:
    """Errors for the functional names in one step config (does not modify it)."""
    import copy
    return normalize_step_functionals(copy.deepcopy(step))[1]
