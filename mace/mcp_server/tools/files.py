"""files_list: what is in the served folder, with MACE's file kinds."""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any, Dict

MAX_LIST = 2000


def register(reg) -> None:
    reg.tool("files_list", (
        "List files in a folder of the served folder with MACE's kind for each (cif, d12, d3, out, band, dos, "
        "cube, db, ...). Hidden files and folders are skipped. Use it to find structures, decks and outputs."),
        {"properties": {
            "directory": {"type": "string", "default": ".", "description": "folder, relative to the served folder"},
            "pattern": {"type": "string", "default": "*", "description": "file-name pattern, e.g. *.out"},
            "recursive": {"type": "boolean", "default": False},
            "kinds_only": {"type": "boolean", "default": False,
                           "description": "only files MACE knows (structures, decks, outputs, data, databases)"},
            "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIST, "default": 300}}},
        title="List files", group="files")(files_list)


def files_list(ctx, args: Dict[str, Any]) -> Dict[str, Any]:
    from mace.gui.filekind import kind_by_name
    top = ctx.path(args.get("directory") or ".", kind="dir")
    pattern = args.get("pattern") or "*"
    limit = int(args.get("limit") or 300)
    rows, truncated = [], False

    def walk(d: Path):
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError:
            return
        for e in entries:
            if e.name.startswith("."):
                continue
            yield e
            if args.get("recursive") and e.is_dir(follow_symlinks=False):
                yield from walk(Path(e.path))

    for e in walk(top):
        ctx.check()
        try:
            is_file = e.is_file()
        except OSError:
            continue
        if not is_file or not fnmatch.fnmatch(e.name, pattern):
            continue
        kind = kind_by_name(e.name)
        if args.get("kinds_only") and not kind:
            continue
        if len(rows) >= limit:
            truncated = True
            break
        try:
            st = e.stat()
            size, mtime = st.st_size, int(st.st_mtime)
        except OSError:
            size, mtime = None, None
        rows.append({"path": ctx.show(e.path), "kind": kind or None, "size": size, "mtime": mtime})
    return {"directory": ctx.show(top), "files": rows, "count": len(rows), "truncated": truncated}
