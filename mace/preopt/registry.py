"""The ML-potential registry: which models MACE offers, under what license.

The license allow-list is code, not data: ``ALLOWED_LICENSES`` below. The
JSON (``mlip_registry.json``, shipped) has no license list of its own, and
``load`` rejects unknown top-level keys, so adding one to the JSON cannot
widen what is allowed. An entry whose code or weights license is not in the
constant is rejected, and so is the whole registry with it.

The preopt default is MACE-MPA-0 medium (MIT code and weights) and its
fallback CHGNet 0.3.0 (BSD, weights inside the pip wheel).

Scopes (1.1.6). Each model serves ``preopt`` (Clean and ``mace preopt``),
``calc`` (``mace calc`` and the GUI's Calculators panel) or both (no
``scopes`` key: both). ``load()``, ``get()``, ``ids()``, ``default_id()``
and ``fallback_id()`` answer for the preopt scope unless told otherwise, so
the pre-optimization tiers see exactly the two models above. The calc scope
adds MACE-MP-0b3 (ACEsuit mace-torch), ORB-v3 (orb-models) and MatterSim v1
(mattersim).

Commercial-safe by default. Every model listed, defaulted to, chosen by
``auto`` or in a preset allows commercial use (``ALLOWED_LICENSES``; the
FAIR Chemistry License of Meta's UMA grants commercial use with its
Acceptable Use Policy, and its weights are gated: the user's own Hugging Face
access fetches them, after MACE shows the terms and records the accept).
Academic-only models (MACE-OFF, Academic Software License) are an explicit
opt-in only (1.1.6): marked ``academic``, never the default or the
fallback, left out of ``load()`` and of ``mace mlip list`` unless asked,
fetched from upstream only after the user acknowledges academic,
non-commercial use (recorded with a date beside the weights, ACADEMIC.json),
never run without that record, and every result made with one is tagged
"(academic license)" (``tag``). ``REMOVED`` names models MACE dropped.
A model file the user already has runs through ``mace calc --model-file``
(mace/calculators/mlip.py) and never gets an entry here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ALLOWED_LICENSES = frozenset({"MIT", "BSD-3-Clause", "BSD-3-Clause-LBNL", "Apache-2.0",
                              "FAIR-Chemistry-License-1.0"})
#: weights licenses that forbid commercial use: only on an entry marked
#: ``academic``, never the default or the fallback, never listed unless asked,
#: and fetched or run only after the user acknowledged academic use
ACADEMIC_LICENSES = frozenset({"ASL"})
BACKENDS = ("mace_mp", "chgnet", "orb", "mattersim", "fairchem")
SCOPES = ("preopt", "calc")
PATH = Path(__file__).with_name("mlip_registry.json")
_TOP_KEYS = {"schema", "models"}
_MODEL_KEYS = {"id", "label", "unrelated_note", "backend", "default", "fallback", "code", "weights",
               "training_data", "citation", "data_citation", "defaults", "limits", "scopes",
               "variant", "family", "elements", "env", "academic", "gated", "requires_python", "charge_spin",
               "dims", "gpu_dims", "tasks", "task_default", "role", "stress"}

#: models MACE does not offer, and why: a command, config or saved choice
#: naming one gets this instead of "no model"
REMOVED: Dict[str, str] = {}

_CACHE: Optional[List[Dict[str, Any]]] = None


class RegistryError(ValueError):
    pass


def validate(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(data, dict):
        raise RegistryError("the registry is not a JSON object")
    extra = set(data) - _TOP_KEYS
    if extra:
        raise RegistryError(f"unknown top-level key(s) {sorted(extra)} in the ML registry")
    if data.get("schema") != 1:
        raise RegistryError(f"unknown registry schema {data.get('schema')!r}")
    models = data.get("models")
    if not isinstance(models, list) or not models:
        raise RegistryError("the registry lists no models")
    seen = set()
    for m in models:
        if not isinstance(m, dict):
            raise RegistryError("a registry entry is not an object")
        mid = m.get("id")
        if not isinstance(mid, str) or not mid or not all(c.isalnum() or c in "-._" for c in mid):
            raise RegistryError(f"bad model id {mid!r}")
        if mid in seen:
            raise RegistryError(f"model id {mid} twice")
        seen.add(mid)
        extra = set(m) - _MODEL_KEYS
        if extra:
            raise RegistryError(f"{mid}: unknown key(s) {sorted(extra)}")
        if m.get("backend") not in BACKENDS:
            raise RegistryError(f"{mid}: unknown backend {m.get('backend')!r}")
        code, weights = m.get("code") or {}, m.get("weights") or {}
        scopes = m.get("scopes", list(SCOPES))
        if not isinstance(scopes, list) or not scopes or any(x not in SCOPES for x in scopes):
            raise RegistryError(f"{mid}: scopes must be a list of {', '.join(SCOPES)}")
        if code.get("license") not in ALLOWED_LICENSES:
            raise RegistryError(f"{mid}: code license {code.get('license')!r} is not allowed "
                                f"(only {', '.join(sorted(ALLOWED_LICENSES))})")
        lic = weights.get("license")
        if m.get("academic"):
            if m.get("academic") is not True or lic not in ACADEMIC_LICENSES:
                raise RegistryError(f"{mid}: an academic entry needs an academic weights license "
                                    f"({', '.join(sorted(ACADEMIC_LICENSES))}), not {lic!r}")
            if m.get("default") or m.get("fallback"):
                raise RegistryError(f"{mid}: an academic-only model is never the default or the fallback")
            if "url" not in weights:
                raise RegistryError(f"{mid}: an academic-only model is fetched from upstream (weights.url)")
        elif lic not in ALLOWED_LICENSES:
            raise RegistryError(f"{mid}: weights license {lic!r} is not allowed "
                                f"(only {', '.join(sorted(ALLOWED_LICENSES))})")
        gated = m.get("gated")
        if gated is not None:
            if not isinstance(gated, dict) or gated.get("provider") != "huggingface" or not gated.get("page", "").startswith("https://"):
                raise RegistryError(f"{mid}: gated is {{provider: huggingface, page, terms}}")
            for k in ("hf_repo", "hf_file", "hf_revision", "sha256", "size_bytes"):
                if not weights.get(k):
                    raise RegistryError(f"{mid}: gated weights need {k}")
            if "url" in weights:
                raise RegistryError(f"{mid}: gated weights come from the user's own Hugging Face access, not a URL")
        cs = m.get("charge_spin")
        if cs is not None and (not isinstance(cs, list) or not all(isinstance(x, str) for x in cs)):
            raise RegistryError(f"{mid}: charge_spin lists the tasks that need a charge and spin ('*': always)")
        for k in ("dims", "gpu_dims"):
            if k in m and (not isinstance(m[k], list) or any(d not in (0, 1, 2, 3) for d in m[k])):
                raise RegistryError(f"{mid}: {k} is a list of dimensionalities 0-3")
        if "tasks" in m:
            td = m.get("task_default") or {}
            if not isinstance(m["tasks"], dict) or set(td) != {"0", "1", "2", "3"} or not set(td.values()) <= set(m["tasks"]):
                raise RegistryError(f"{mid}: tasks need a task_default for dimensionalities 0-3 among them")
        if "url" in weights:
            sha = str(weights.get("sha256", ""))
            if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise RegistryError(f"{mid}: the weights need a sha256")
            if not str(weights["url"]).startswith("https://") or not weights.get("filename"):
                raise RegistryError(f"{mid}: the weights need an https URL and a file name")
            if "/" in weights["filename"] or weights["filename"].startswith("."):
                raise RegistryError(f"{mid}: bad weights file name")
        if not m.get("citation"):
            raise RegistryError(f"{mid}: no citation")
        env = m.get("env", "default")
        if not isinstance(env, str) or not env.isalnum() or env != env.lower():
            raise RegistryError(f"{mid}: env is a lowercase word")
    for scope in SCOPES:
        if sum(bool(m.get("default")) for m in models if scope in m.get("scopes", SCOPES) and not m.get("academic")) != 1:
            raise RegistryError(f"exactly one model must be the default (scope {scope})")
    return models


def in_scope(m: Dict[str, Any], scope: Optional[str]) -> bool:
    return scope is None or scope in m.get("scopes", SCOPES)


def load(path: Optional[Path] = None, scope: Optional[str] = "preopt",
         include_academic: bool = False) -> List[Dict[str, Any]]:
    """The validated model list of ``scope`` (None: every model), cached
    for the shipped file. The whole file is validated whatever the scope.
    Academic-only models are left out unless ``include_academic`` (a listing
    asked for them; ``get`` finds them by id)."""
    global _CACHE
    if path is None and _CACHE is not None:
        models = _CACHE
    else:
        with open(path or PATH) as fh:
            models = validate(json.load(fh))
        if path is None:
            _CACHE = models
    return [m for m in models if in_scope(m, scope) and (include_academic or not m.get("academic"))]


def get(mid: str, scope: Optional[str] = "preopt") -> Optional[Dict[str, Any]]:
    """The model named ``mid`` (an academic-only one too: it is gated where
    it is fetched and run, by the user's recorded acknowledgment)."""
    return next((m for m in load(scope=scope, include_academic=True) if m["id"] == mid), None)


def is_academic(m: Optional[Dict[str, Any]]) -> bool:
    return bool(m and m.get("academic"))


def tag(m: Dict[str, Any]) -> str:
    """How every result made with ``m`` names it: an academic-only model's
    label always says so."""
    return f"{m['label']} (academic license)" if is_academic(m) else m["label"]


def default_id(scope: Optional[str] = "preopt") -> str:
    return next(m["id"] for m in load(scope=scope) if m.get("default") and not m.get("academic"))


def fallback_id(scope: Optional[str] = "preopt") -> Optional[str]:
    return next((m["id"] for m in load(scope=scope) if m.get("fallback")), None)


def ids(scope: Optional[str] = "preopt") -> List[str]:
    return [m["id"] for m in load(scope=scope)]


def unknown(mid: str, scope: Optional[str] = "preopt", prefix: str = "") -> str:
    """Why ``mid`` names no model of ``scope``: the removal note of a model
    MACE dropped, else the ids it knows (each with ``prefix``)."""
    if mid in REMOVED:
        return REMOVED[mid]
    return f"no ML model {mid!r}; known: " + ", ".join(prefix + i for i in ids(scope=scope))


def provenance_info(m: Dict[str, Any]) -> Dict[str, Any]:
    """What a result records about the model (provenance, citation copy)."""
    return {"code": m["code"]["package"], "version": m["code"].get("pin"), "code_license": m["code"]["license"],
            "license": m["code"]["license"], "model_id": m["id"], "model_label": tag(m),
            "weights_license": m["weights"]["license"], "weights_sha256": m["weights"].get("sha256"),
            "training_data": list(m.get("training_data") or []), "citation": list(m.get("citation") or []),
            "data_citation": list(m.get("data_citation") or []), "unrelated_note": m.get("unrelated_note"),
            "limits": m.get("limits"), "label": f"ML potential: {tag(m)}",
            "family": m.get("family"), "academic": is_academic(m)}
