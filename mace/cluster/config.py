"""Area F's site settings: the ``[cluster]``, ``[quota]``, ``[scan]`` and
``[devnodes]`` tables of ``site.toml``.

One site file for MACE: ``$XDG_CONFIG_HOME/mace/site.toml`` (default
``~/.config/mace/site.toml``); ``MACE_SITE_CONFIG=<path>`` replaces it and
``MACE_SITE_CONFIG=none`` ignores it. Precedence: site.toml > built-in. When
``mace.submission.site`` (the site loader) is present it reads the file;
otherwise the file is read here with ``tomllib``. Every key is optional.

Validation errors name the file, the line, the key and the value
(``site.toml:12: [quota] warn_percent = 120: must be a whole number from 50
to 99``); a bad key keeps its built-in value. Unknown keys inside these
tables give one warning line each.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_GROUP_REGEX = (r"^(intel|amd|arm|xeon|epyc|zen|skylake|cascade|icelake|sapphire|grace)"
                       r"[-_]?[0-9a-z]*$")

BUILTIN: Dict[str, Dict[str, Any]] = {
    "cluster": {"group_feature_regex": DEFAULT_GROUP_REGEX, "node_query": "auto"},
    "quota": {"command": None, "format": "table", "paths": [], "warn_percent": 80,
              "critical_percent": 90, "alias_env_allow": [],
              "research_roots": ["/mnt/research"], "research_paths": []},
    "scan": {"entries_per_second": 2000, "max_seconds": 600, "max_entries": 1_200_000,
             "sizes": "top", "leftover_paths": []},
    "devnodes": {"hosts": [], "pick_by": "load"},
}

# Environment names a quota alias may never set, even when listed in
# [quota] alias_env_allow (they change which code runs).
REFUSED_ENV_PREFIXES = ("LD_", "DYLD_", "PYTHON", "PERL5", "RUBY")
REFUSED_ENV_NAMES = ("PATH", "BASH_ENV", "ENV", "IFS", "SHELLOPTS")


def refused_env_name(name: str) -> bool:
    return name in REFUSED_ENV_NAMES or any(name.startswith(p) for p in REFUSED_ENV_PREFIXES)


@dataclass
class ClusterConfig:
    values: Dict[str, Dict[str, Any]]
    sources: Dict[str, str] = field(default_factory=dict)     # "quota.warn_percent" -> "site.toml" | "built-in"
    path: Optional[Path] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def get(self, table: str, key: str) -> Any:
        return self.values[table][key]


def site_path(env: Optional[Dict[str, str]] = None) -> Optional[Path]:
    """The site.toml in effect, or None (``MACE_SITE_CONFIG=none``)."""
    env = os.environ if env is None else env
    explicit = (env.get("MACE_SITE_CONFIG") or "").strip()
    if explicit:
        return None if explicit.lower() == "none" else Path(explicit).expanduser()
    base = env.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "mace" / "site.toml"


def _read_toml(path: Path) -> Tuple[Dict[str, Any], Optional[str], str]:
    """(data, error, text)."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        return {}, None, ""
    except OSError as exc:
        return {}, f"{path}: cannot read: {exc}", ""
    try:
        import tomllib
    except ImportError:                                   # pragma: no cover - Python < 3.11
        return {}, f"{path}: needs Python 3.11 or newer (tomllib) to read", text
    try:
        return tomllib.loads(text), None, text
    except Exception as exc:                              # tomllib.TOMLDecodeError
        return {}, f"{path}: {exc}", text


def _line_of(text: str, table: str, key: str) -> int:
    """1-based line of ``key =`` inside ``[table]`` (0 when not found)."""
    current = None
    for n, raw in enumerate(text.splitlines(), 1):
        s = raw.strip()
        m = re.match(r"^\[\s*([A-Za-z0-9_.-]+)\s*\]", s)
        if m:
            current = m.group(1)
            continue
        if current == table and re.match(rf"^{re.escape(key)}\s*=", s):
            return n
    return 0


def _int_range(v, lo, hi) -> Optional[str]:
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        return f"must be a whole number from {lo} to {hi}"
    return None


def _str_list(v) -> Optional[str]:
    if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
        return "must be a list of non-empty strings"
    return None


def _check(table: str, key: str, v: Any) -> Optional[str]:
    if table == "cluster":
        if key == "group_feature_regex":
            if not isinstance(v, str):
                return "must be a regular expression string"
            try:
                re.compile(v, re.I)
            except re.error as exc:
                return f"is not a valid regular expression ({exc})"
            return None
        if key == "node_query":
            return None if v in ("auto", "scontrol", "sinfo") else 'must be "auto", "scontrol" or "sinfo"'
    if table == "quota":
        if key == "command":
            return _str_list(v) if v is not None else None
        if key == "format":
            return None if v in ("table", "lfs", "quota", "json") else 'must be "table", "lfs", "quota" or "json"'
        if key in ("paths", "research_paths"):
            return _str_list(v) if v != [] else None
        if key == "research_roots":
            return _str_list(v)
        if key in ("warn_percent", "critical_percent"):
            return _int_range(v, 50, 99)
        if key == "alias_env_allow":
            p = _str_list(v) if v != [] else None
            if p:
                return p
            bad = [x for x in v if refused_env_name(x)]
            if bad:
                return f"lists {', '.join(bad)}, which a quota alias may never set"
            return None
    if table == "scan":
        if key == "entries_per_second":
            return _int_range(v, 500, 10000)
        if key == "max_seconds":
            return _int_range(v, 60, 3600)
        if key == "max_entries":
            return _int_range(v, 1000, 36_000_000)
        if key == "sizes":
            return None if v in ("top", "all", "off") else 'must be "top", "all" or "off"'
        if key == "leftover_paths":
            return _str_list(v) if v != [] else None
    if table == "devnodes":
        if key == "hosts":
            if v == []:
                return None
            p = _str_list(v)
            if p:
                return p
            bad = [h for h in v if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,252}$", h)]
            return f"has a host name that is not plain: {bad[0]!r}" if bad else None
        if key == "pick_by":
            return None if v in ("load", "memory") else 'must be "load" or "memory"'
    return None


def _site_loader_data(env) -> Optional[Tuple[Dict[str, Any], Optional[Path], str]]:
    """The tables through the site loader (mace.submission.site) when present."""
    try:
        from mace.submission import site as site_mod      # Area D's loader, when merged
    except Exception:
        return None
    loader = getattr(site_mod, "load_tables", None)
    if loader is None:
        return None
    try:
        data, path, text = loader(env)
    except Exception:
        return None
    return data, path, text


def load(env: Optional[Dict[str, str]] = None) -> ClusterConfig:
    env = os.environ if env is None else env
    values = {t: dict(v) for t, v in BUILTIN.items()}
    sources = {f"{t}.{k}": "built-in" for t, kv in BUILTIN.items() for k in kv}
    cfg = ClusterConfig(values=values, sources=sources)
    via = _site_loader_data(env)
    if via is not None:
        data, path, text = via
        err = None
    else:
        path = site_path(env)
        if path is None:
            return cfg
        data, err, text = _read_toml(path)
    cfg.path = path
    if err:
        cfg.errors.append(err)
        return cfg
    name = path.name if path else "site.toml"
    for table in BUILTIN:
        tab = data.get(table)
        if tab is None:
            continue
        if not isinstance(tab, dict):
            cfg.errors.append(f"{name}: [{table}] must be a table")
            continue
        for key, v in tab.items():
            if key not in BUILTIN[table]:
                cfg.warnings.append(f"{name}:{_line_of(text, table, key)}: [{table}] {key} is not a known key; ignored")
                continue
            why = _check(table, key, v)
            if why:
                cfg.errors.append(f"{name}:{_line_of(text, table, key)}: [{table}] {key} = {v!r}: {why}")
                continue
            values[table][key] = v
            sources[f"{table}.{key}"] = "site.toml"
    q = values["quota"]
    if q["critical_percent"] <= q["warn_percent"]:
        cfg.errors.append(f"{name}:{_line_of(text, 'quota', 'critical_percent')}: [quota] critical_percent = "
                          f"{q['critical_percent']} must be above warn_percent = {q['warn_percent']}; "
                          f"using the built-in 80 and 90")
        q["warn_percent"], q["critical_percent"] = 80, 90
        sources["quota.warn_percent"] = sources["quota.critical_percent"] = "built-in"
    s = values["scan"]
    if s["max_entries"] > s["entries_per_second"] * s["max_seconds"]:
        cfg.errors.append(f"{name}:{_line_of(text, 'scan', 'max_entries')}: [scan] max_entries = {s['max_entries']} "
                          f"is more than entries_per_second x max_seconds = "
                          f"{s['entries_per_second']} x {s['max_seconds']} = "
                          f"{s['entries_per_second'] * s['max_seconds']}; using that product")
        s["max_entries"] = s["entries_per_second"] * s["max_seconds"]
        sources["scan.max_entries"] = "derived"
    return cfg


def as_dict(cfg: ClusterConfig) -> Dict[str, Any]:
    """JSON view (for `mace cluster --json` and the GUI's Settings note)."""
    return {"path": str(cfg.path) if cfg.path else None, "values": cfg.values,
            "sources": cfg.sources, "errors": cfg.errors, "warnings": cfg.warnings}
