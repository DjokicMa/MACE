"""MCP resources: the shipped option templates and the MCP guide.

``mace://templates/d12/<name>`` - a D12 options file (Crystal_d12/example_configs),
``mace://templates/d3/<name>``  - a D3 configuration (Crystal_d3/example_configs),
``mace://docs/mcp``             - docs/mcp.md, what each tool does.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[2]
TEMPLATE_DIRS = {"d12": REPO / "Crystal_d12" / "example_configs", "d3": REPO / "Crystal_d3" / "example_configs"}
GUIDE = REPO / "docs" / "mcp.md"


def template_files(kind: str) -> List[Path]:
    d = TEMPLATE_DIRS[kind]
    return sorted(p for p in d.glob("*.json") if p.is_file()) if d.is_dir() else []


def template_summary(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"name": path.stem, "error": str(exc)}
    conf = data.get("configuration", data) if isinstance(data, dict) else {}
    conf = conf if isinstance(conf, dict) else {}
    return {"name": path.stem,
            "calculation_type": conf.get("calculation_type") or (data.get("calculation_type") if isinstance(data, dict) else None),
            "description": conf.get("description") or (data.get("description") if isinstance(data, dict) else "") or ""}


def listing(config) -> List[Dict[str, Any]]:
    out = []
    for kind in ("d12", "d3"):
        for p in template_files(kind):
            s = template_summary(p)
            desc = s.get("description") or f"{kind.upper()} {s.get('calculation_type') or ''} options".strip()
            out.append({"uri": f"mace://templates/{kind}/{p.stem}", "name": f"{kind}-{p.stem}",
                        "title": f"{kind.upper()} template {p.stem}", "description": desc,
                        "mimeType": "application/json"})
    if GUIDE.is_file():
        out.append({"uri": "mace://docs/mcp", "name": "mcp-guide", "title": "MACE MCP tools",
                    "description": "What each MACE tool does and how confirmations work",
                    "mimeType": "text/markdown"})
    return out


def read(config, uri: str) -> List[Dict[str, Any]]:
    if uri == "mace://docs/mcp" and GUIDE.is_file():
        return [{"uri": uri, "mimeType": "text/markdown", "text": GUIDE.read_text()}]
    prefix = "mace://templates/"
    if uri.startswith(prefix):
        kind, _, name = uri[len(prefix):].partition("/")
        if kind in TEMPLATE_DIRS:
            for p in template_files(kind):
                if p.stem == name:
                    return [{"uri": uri, "mimeType": "application/json", "text": p.read_text()}]
    raise KeyError(uri)


def template_path(kind: str, name: str) -> Path:
    """The template file of ``name`` (a stem, or a stem with .json)."""
    stem = name[:-5] if name.endswith(".json") else name
    for p in template_files(kind):
        if p.stem == stem:
            return p
    known = ", ".join(p.stem for p in template_files(kind))
    raise ValueError(f"no {kind.upper()} template {name!r}; known: {known}")
