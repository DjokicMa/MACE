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
    # The rule lives with opt2d12, which applies it to --config-file configs
    from d12_constants import (  # noqa: E402
        LEGACY_FUNCTIONAL_SPELLINGS as LEGACY_SPELLINGS, UnknownFunctionalError,
        normalize_functional,
    )
except ImportError:  # no keyword tables: plans are used as written
    LEGACY_SPELLINGS = {}

    class UnknownFunctionalError(ValueError):
        """A plan names a functional CRYSTAL23 does not accept."""

    def normalize_functional(name: Any, hint: str = "") -> Tuple[Any, Optional[str]]:
        return name, None

# Plan keys that hold a functional name, inside these step sub-dicts.
FUNCTIONAL_KEYS = ("custom_functional", "new_functional")
METHOD_SECTIONS = ("method_modifications", "method_settings")


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
