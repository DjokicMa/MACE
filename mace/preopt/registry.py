"""The ML-potential registry: which models MACE offers, under what license.

The license allow-list is code, not data: ``ALLOWED_LICENSES`` below. The
JSON (``mlip_registry.json``, shipped) has no license list of its own, and
``load`` rejects unknown top-level keys, so adding one to the JSON cannot
widen what is allowed. An entry whose code or weights license is not in the
constant is rejected, and so is the whole registry with it.

In 1.1.5 the registry holds MACE-MPA-0 medium (MIT code and weights, the
default) and CHGNet 0.3.0 (BSD, weights inside the pip wheel, the offline
fallback). Models under the Academic Software License and Meta's
FAIR/OMat24 models are not listed and not offered.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ALLOWED_LICENSES = frozenset({"MIT", "BSD-3-Clause", "BSD-3-Clause-LBNL", "Apache-2.0"})
BACKENDS = ("mace_mp", "chgnet")
PATH = Path(__file__).with_name("mlip_registry.json")
_TOP_KEYS = {"schema", "models"}
_MODEL_KEYS = {"id", "label", "unrelated_note", "backend", "default", "fallback", "code", "weights",
               "training_data", "citation", "data_citation", "defaults", "limits"}

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
        for what, lic in (("code", code.get("license")), ("weights", weights.get("license"))):
            if lic not in ALLOWED_LICENSES:
                raise RegistryError(f"{mid}: {what} license {lic!r} is not allowed "
                                    f"(only {', '.join(sorted(ALLOWED_LICENSES))})")
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
    if sum(bool(m.get("default")) for m in models) != 1:
        raise RegistryError("exactly one model must be the default")
    return models


def load(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """The validated model list (cached for the shipped file)."""
    global _CACHE
    if path is None and _CACHE is not None:
        return _CACHE
    with open(path or PATH) as fh:
        models = validate(json.load(fh))
    if path is None:
        _CACHE = models
    return models


def get(mid: str) -> Optional[Dict[str, Any]]:
    return next((m for m in load() if m["id"] == mid), None)


def default_id() -> str:
    return next(m["id"] for m in load() if m.get("default"))


def fallback_id() -> Optional[str]:
    return next((m["id"] for m in load() if m.get("fallback")), None)


def ids() -> List[str]:
    return [m["id"] for m in load()]


def provenance_info(m: Dict[str, Any]) -> Dict[str, Any]:
    """What a result records about the model (provenance, citation copy)."""
    return {"code": m["code"]["package"], "version": m["code"].get("pin"), "code_license": m["code"]["license"],
            "license": m["code"]["license"], "model_id": m["id"], "model_label": m["label"],
            "weights_license": m["weights"]["license"], "weights_sha256": m["weights"].get("sha256"),
            "training_data": list(m.get("training_data") or []), "citation": list(m.get("citation") or []),
            "data_citation": list(m.get("data_citation") or []), "unrelated_note": m.get("unrelated_note"),
            "limits": m.get("limits"), "label": f"ML potential: {m['label']}"}
