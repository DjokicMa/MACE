"""Named folders the GUI shortens paths with, and Open Folder starts from.

``root_aliases()`` is the one list (shared rule S4 of the 1.1.5 polish
plans): the home folder (as written in $HOME and resolved, which differ when
home is a symbolic link) and the cluster scratch/work folders named by
$SCRATCH, $WORK and $PSCRATCH, each only when set, non-empty and an existing
directory. Nothing site-specific is hard-coded: a site without those
variables gets HOME alone. The page writes "~" for HOME and "$SCRATCH/..."
etc. for the others, the longest matching path winning."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional

ALIAS_VARS = ("SCRATCH", "WORK", "PSCRATCH")


def _home_literal(env: Mapping[str, str]) -> str:
    h = env.get("HOME", "")
    if h:
        return h
    try:
        return str(Path.home())
    except (RuntimeError, KeyError):
        return ""


def root_aliases(env: Optional[Mapping[str, str]] = None) -> List[Dict[str, str]]:
    """[{name, path}] in order HOME (literal, then resolved), SCRATCH, WORK,
    PSCRATCH; absolute paths without a trailing slash, no duplicates."""
    env = os.environ if env is None else env
    out: List[Dict[str, str]] = []
    seen = set()

    def add(name: str, raw: str, resolve_too: bool) -> None:
        raw = (raw or "").strip()
        if not raw or not os.path.isabs(raw):
            return
        p = Path(raw)
        try:
            if not p.is_dir():
                return
            real = str(p.resolve())
        except OSError:
            return
        for cand in ([os.path.normpath(raw), real] if resolve_too else [real]):
            if cand not in seen and cand != "/":
                seen.add(cand)
                out.append({"name": name, "path": cand})

    add("HOME", _home_literal(env), True)
    for var in ALIAS_VARS:
        add(var, env.get(var, ""), True)
    return out
