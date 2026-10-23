"""Research spaces: shared project folders next to ``$HOME`` and ``$SCRATCH``.

Many clusters give each research group a folder (``<group>`` under ``/mnt/research``, a symlink
into the shared file system, on some clusters) with its own space and file
quota. This module finds the ones you can use, without listing the whole
parent folder (one such parent holds over a thousand groups):

* a candidate is a group you belong to (its folder under a research root), a
  name the site's quota tool lists, or a path in ``[quota] research_paths``;
* it counts when it exists as a folder you can read and enter.

``annotate(rows)`` then labels the quota rows the site tool printed: ``home``,
``scratch``, or a research space (with its folder), so the GUI and
``mace cluster quota`` show Quota % and Files % per space. ``spaces(rows)``
lists every usable research space, with its numbers when the tool had a row
for it and without them when it did not. Standard library only.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

DEFAULT_ROOTS = ("/mnt/research",)
HOME_LABEL = re.compile(r"^home\b", re.I)


def group_names() -> List[str]:
    """The names of the groups you belong to (empty where the platform has no groups)."""
    try:
        import grp
    except ImportError:                         # Windows
        return []
    gids = set()
    try:
        gids.update(os.getgroups())
        gids.add(os.getgid())
    except (AttributeError, OSError):
        return []
    out: List[str] = []
    for gid in sorted(gids):
        try:
            out.append(grp.getgrgid(gid).gr_name)
        except KeyError:
            continue
    return out


def _usable(path: str) -> bool:
    return os.path.isdir(path) and os.access(path, os.R_OK | os.X_OK)


def roots(cfg=None) -> List[str]:
    """The parent folders of research spaces: ``[quota] research_roots`` (a
    list), else /mnt/research. Only those that exist."""
    vals: Sequence[str] = DEFAULT_ROOTS
    if cfg is not None:
        try:
            vals = cfg.get("quota", "research_roots") or DEFAULT_ROOTS
        except Exception:
            vals = DEFAULT_ROOTS
    return [r for r in (os.path.expanduser(v).rstrip("/") or "/" for v in vals) if os.path.isdir(r)]


def extra_paths(cfg=None) -> List[str]:
    if cfg is None:
        return []
    try:
        return [os.path.expanduser(p) for p in (cfg.get("quota", "research_paths") or [])]
    except Exception:
        return []


def usable_spaces(cfg=None, names: Iterable[str] = (), groups: Optional[Sequence[str]] = None
                  ) -> List[Dict[str, Any]]:
    """[{name, path, real}] for each research space you can enter, sorted by name."""
    groups = group_names() if groups is None else list(groups)
    cands: Dict[str, str] = {}
    parents = roots(cfg)
    for parent in parents:
        for n in list(groups) + [x for x in names if x]:
            if not n or "/" in n or n.startswith("."):
                continue
            p = os.path.join(parent, n)
            if _usable(p):
                cands.setdefault(os.path.realpath(p), p)
    for p in extra_paths(cfg):
        if _usable(p):
            cands.setdefault(os.path.realpath(p), p)
    out = [{"name": os.path.basename(p.rstrip("/")), "path": p, "real": real} for real, p in cands.items()]
    return sorted(out, key=lambda r: r["name"].lower())


def space_kind(label: str) -> str:
    """'home' or 'scratch' for the site tool's label of that name, else ''."""
    low = (label or "").strip().lower()
    if HOME_LABEL.match(low):
        return "home"
    if low.startswith("scratch"):
        return "scratch"
    return ""


def annotate(rows: List[Dict[str, Any]], cfg=None, scratch: Sequence[str] = (),
             groups: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Add ``space`` ('home', 'scratch', 'research' or '') and the folder
    (``path``) to the quota rows of the site tool, in place. A row labeled
    with an absolute path (the tool's --verbose form) keeps the path and gets
    the folder's name as its label. Returns the rows."""
    names = []
    for r in rows:
        lab = (r.get("label") or "").strip()
        if lab and not lab.startswith("/") and not space_kind(lab) and r.get("kind") != "filesystem":
            names.append(lab)
    spaces = usable_spaces(cfg, names=names, groups=groups)
    by_name = {s["name"]: s for s in spaces}
    by_real = {s["real"]: s for s in spaces}
    home = os.path.realpath(os.path.expanduser("~"))
    for r in rows:
        if r.get("kind") == "filesystem" or r.get("source") in ("filesystem",):
            continue
        lab = (r.get("label") or "").strip()
        if lab.startswith("/"):
            real = os.path.realpath(lab)
            r["path"] = r.get("path") or lab
            hit = by_real.get(real)
            if real == home:
                r["space"], r["label"] = "home", "home"
            elif hit:
                r["space"], r["label"] = "research", hit["name"]
            elif any(real == os.path.realpath(s) for s in scratch):
                r["space"], r["label"] = "scratch", "scratch"
            else:
                r["space"] = ""
                r["label"] = os.path.basename(lab.rstrip("/")) or lab
            r["real"] = real
            continue
        kind = space_kind(lab)
        if kind == "home":
            r["space"], r["path"] = "home", r.get("path") or os.path.expanduser("~")
        elif kind == "scratch":
            r["space"], r["path"] = "scratch", r.get("path") or (scratch[0] if scratch else None)
        elif lab in by_name:
            r["space"], r["path"] = "research", r.get("path") or by_name[lab]["path"]
        else:
            r.setdefault("space", "")
        if r.get("path"):
            r["real"] = os.path.realpath(r["path"])
    return rows


def spaces(rows: List[Dict[str, Any]], cfg=None, groups: Optional[Sequence[str]] = None
           ) -> List[Dict[str, Any]]:
    """Every research space you can use, one record each: name, path, whether
    the tool had a row for it (``has_quota``), and the Quota % / Files %."""
    named = [r["label"] for r in rows if r.get("space") == "research"]
    out = []
    for s in usable_spaces(cfg, names=named, groups=groups):
        row = next((r for r in rows if r.get("space") == "research"
                    and (r.get("label") == s["name"] or (r.get("path") and os.path.realpath(r["path"]) == s["real"]))),
                   None)
        out.append({"name": s["name"], "path": s["path"], "real": s["real"], "has_quota": bool(row),
                    "space_pct": row.get("space_pct") if row else None,
                    "files_pct": row.get("files_pct") if row else None,
                    "space_used_b": row.get("space_used_b") if row else None,
                    "space_limit_b": row.get("space_limit_b") if row else None,
                    "files_used": row.get("files_used") if row else None,
                    "files_limit": row.get("files_limit") if row else None,
                    "level": row.get("level", "ok") if row else "none"})
    return out
