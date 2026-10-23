"""The Plot Gallery's routes: many outputs' plots of one kind in panes.

Everything is ``mace.plotting.gallery`` (the module ``mace plotting
--gallery`` runs): this module only turns the page's choice of set into
files inside the served folder and paths back into relative ones.

* ``POST gallery/collect`` {kind, source, db?}: the set's files (capped, with
  the message), their titles and the caption table;
* ``GET gallery/pane`` ?kind&path&emin&emax: one pane's description (plotly
  draws it in the page; the figure export draws the same with matplotlib);
* ``POST gallery/export`` {kind, paths, cols, shared, emin, emax, format}: a
  task that runs ``mace plotting --gallery`` in a folder of the server's cache
  (never the browsed folder); the page downloads the file through
  ``plot-image``;
* ``POST gallery/csv`` {paths, db?}: the caption table as CSV.

A source is ``{"type": "folder", "folder"}``, ``{"type": "paths",
"paths"}`` (the Explorer's selection), ``{"type": "batch", "batch"}`` (a
Batch Setup folder: its decks' folder and the files it was made from) or
``{"type": "db", "db", "filters"}`` (the Database view's filtered
materials, matched to the files below the database's folder).
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mace.plotting import gallery as G

MAX_PATHS = G.MAX_ITEMS         # panes per page by default (the CLI's --max default too)
SELECT_MAX = 5000               # files in a selection (paths): paged, so far more than a page
PAGE_SIZES = (25, 50, 100, 200, 500)
PAGE_MAX = 500
DB_MATERIALS_MAX = 5000
_PANES: "OrderedDict[Tuple, Dict[str, Any]]" = OrderedDict()
_PANES_MAX = 64
_lock = threading.Lock()


def _ewin(get) -> Tuple[float, float]:
    try:
        lo = float(get("emin") if get("emin") not in (None, "") else G.EWINDOW[0])
        hi = float(get("emax") if get("emax") not in (None, "") else G.EWINDOW[1])
    except (TypeError, ValueError):
        raise ValueError("the energy window is two numbers (eV)")
    if not lo < hi or abs(lo) > 200 or abs(hi) > 200:
        raise ValueError("the energy window runs from a lower to a higher energy, within ±200 eV")
    return lo, hi


def _kind(k: Any) -> str:
    if k not in G.KINDS:
        raise ValueError(f"the gallery kind is one of {', '.join(G.KINDS)}")
    return str(k)


def _rel_or_none(st, p: Optional[Path]) -> Optional[str]:
    if p is None:
        return None
    try:
        return st.rel(Path(p))
    except ValueError:
        return None


def _paths(st, rels: Any, limit: int = 0) -> List[Path]:
    limit = limit or SELECT_MAX
    if not isinstance(rels, list) or not rels:
        raise ValueError("paths is a list of files")
    if len(rels) > limit:
        raise ValueError(f"at most {limit} files")
    return [st.resolve(str(r)) for r in rels]


def _db_of(st, body: Dict[str, Any]) -> Optional[Path]:
    rel = body.get("db")
    if not rel:
        return None
    p = st.resolve(str(rel))
    return p if p.is_file() else None


def _batch_inputs(st, rel: str) -> List[Path]:
    """A Batch Setup folder: its decks' folder and the files it was made from."""
    b = st.resolve(rel)
    if b.is_file() and b.name == "batch.json":
        b = b.parent
    bj = b / "batch.json"
    if not bj.is_file():
        raise ValueError(f"{st.rel(b) or 'this folder'} is not a Batch Setup folder (no batch.json)")
    try:
        rec = json.loads(bj.read_text())
    except (OSError, ValueError):
        raise ValueError("the batch.json cannot be read")
    out = [b / "decks"] if (b / "decks").is_dir() else [b]
    for r in rec.get("selection") or []:
        try:
            p = st.resolve(str(r))
        except (PermissionError, FileNotFoundError, ValueError):
            continue
        if p.exists():
            out.append(p)
    return out


def _db_materials(st, db: Path, filters: Dict[str, Any]) -> List[str]:
    from . import db as gui_db
    f = filters if isinstance(filters, dict) else {}
    flt = gui_db.Filters.from_query(lambda k: None if f.get(k) is None else str(f.get(k)))
    ids: List[str] = []
    after = None
    while len(ids) < DB_MATERIALS_MAX:
        page = gui_db.page(db, "material_id", "asc", after, gui_db.PAGE_MAX, flt)
        ids += [r["material_id"] for r in page["rows"]]
        after = page.get("next")
        if not after:
            break
    return ids


def _inside(st):
    """A file the Explorer would list: a symbolic link out of the served folder is left out, never read."""
    from .filekind import rel_inside
    return lambda p: rel_inside(st, p) is not None


def _page_args(body: Dict[str, Any]) -> Tuple[int, int]:
    """(page size, page) of a collect request: ``page_size`` panes per page (default 200, at most 500)
    and ``page`` from 1."""
    try:
        size = int(body.get("page_size") or MAX_PATHS)
        page = int(body.get("page") or 1)
    except (TypeError, ValueError):
        raise ValueError("page and page_size are whole numbers")
    if not 1 <= size <= PAGE_MAX:
        raise ValueError(f"page_size is from 1 to {PAGE_MAX}")
    if page < 1:
        raise ValueError("page is from 1")
    return size, page


def collect(st, body: Dict[str, Any]) -> Dict[str, Any]:
    kind = _kind(body.get("kind"))
    src = body.get("source") or {}
    typ = src.get("type") or "folder"
    db = _db_of(st, body)
    wanted: Optional[set] = None
    if typ == "folder":
        inputs = [st.resolve(str(src.get("folder") or "."))]
        label = st.rel(inputs[0]) or "this folder"
    elif typ == "paths":
        inputs = _paths(st, src.get("paths"))
        label = f"{len(inputs)} selected" if len(inputs) != 1 else st.rel(inputs[0])
    elif typ == "batch":
        inputs = _batch_inputs(st, str(src.get("batch") or ""))
        label = f"Batch {st.resolve(str(src.get('batch'))).name}"
    elif typ == "db":
        db = st.resolve(str(src.get("db") or body.get("db") or ""))
        if not db.is_file():
            raise FileNotFoundError(st.rel(db))
        wanted = set(_db_materials(st, db, src.get("filters") or {}))
        inputs = [db.parent]
        label = f"{len(wanted)} materials of {db.name}"
    else:
        raise ValueError("the source is a folder, a selection (paths), a batch or a database query (db)")
    cap, page = _page_args(body)
    if wanted is not None:
        got = G.collect(inputs, kind, cap=G.SCAN_CAP, inside=_inside(st))
        caps = G.captions(got["items"], db=db, stop=st.root)
        keep = [p for p, r in zip(got["items"], caps["rows"]) if r["material_id"] in wanted]
        got = G._paged(keep, len(keep), cap, page, kind)
        if not got["items"]:
            got["message"] = (f"No {G.LABELS[kind].lower()} files below {st.rel(db.parent) or 'this folder'} belong to the "
                              f"{len(wanted)} materials of this query.")
    else:
        got = G.collect(inputs, kind, cap=cap, inside=_inside(st), page=page)
    items = got["items"]
    rels = [st.rel(p) for p in items]
    caps = G.captions(items, db=db, stop=st.root, rel=lambda p: st.rel(p), start=got["offset"] + 1)
    return {"kind": kind, "label": G.LABELS[kind], "source_label": label,
            "items": [{"path": r, "name": p.name, "title": t} for r, p, t in zip(rels, items, G.titles(items))],
            "total": got["total"], "capped": got["capped"], "cap": cap, "message": got["message"],
            "page": got["page"], "pages": got["pages"], "page_size": cap, "offset": got["offset"],
            "page_sizes": list(PAGE_SIZES),
            "captions": {**caps, "databases": [_rel_or_none(st, Path(d)) or Path(d).name for d in caps["databases"]]},
            "ewindow": list(G.EWINDOW), "max_cols": G.MAX_COLS,
            "cli_inputs": [_short(st, i) for i in inputs] if typ != "db" else rels,
            "command": "mace " + " ".join(_quote(a) for a in G.command(kind, [_short(st, i) for i in inputs] if typ != "db"
                                                                      else rels, 3, f"gallery_{kind}.svg",
                                                                      page=got["page"], page_size=cap))}


def _short(st, p: Path) -> str:
    return st.rel(p) or "."


def _quote(a: str) -> str:
    import shlex
    return shlex.quote(a)


def pane(st, get) -> Dict[str, Any]:
    kind = _kind(get("kind"))
    p = st.resolve(get("path"))
    if not p.is_file():
        raise FileNotFoundError(st.rel(p))
    ewin = _ewin(get)
    # Settings › Plots › DOS colors: which projections a DOS pane draws (mace plotting --dos-colors)
    dos_colors = get("dos_colors") or "auto"
    if dos_colors not in ("auto", "total", "element", "orbital", "element-orbital"):
        raise ValueError("dos_colors is auto, total, element, orbital or element-orbital")
    # Settings › Plots › DOS smoothing: the Gaussian sigma in eV, 0 off (mace plotting --dos-smooth)
    try:
        dos_smooth = float(get("dos_smooth") or 0)
    except (TypeError, ValueError):
        raise ValueError("dos_smooth is a number of eV")
    if not 0 <= dos_smooth <= 1:
        raise ValueError("dos_smooth is from 0 (off) to 1 eV")
    # Settings › Plots › Fermi shift (with the folder file's fermi_shift) and the energy zero, as Export
    # and Copy CLI Command pass them (mace plotting --fermi-shift/--energy-reference)
    from .plots import energy_reference
    shift, reference = energy_reference(get)
    s = p.stat()
    key = (str(p), s.st_mtime_ns, s.st_size, kind, ewin, dos_colors, dos_smooth, shift, reference)
    with _lock:
        hit = _PANES.get(key)
        if hit is not None:
            _PANES.move_to_end(key)
    if hit is None:
        hit = G.pane(p, kind, ewin, shift, reference, dos_colors=dos_colors, dos_smooth=dos_smooth)
        with _lock:
            _PANES[key] = hit
            while len(_PANES) > _PANES_MAX:
                _PANES.popitem(last=False)
    out = {**hit, "path": st.rel(p)}
    if hit.get("partner"):
        out["partner"] = _rel_or_none(st, Path(hit["partner"]))
    return out


def _spin_look(body: Dict[str, Any]) -> Dict[str, Any]:
    """The spin look of Settings › Plots (scheme, custom pair, spin down's line, the closed-shell band
    color), checked."""
    from mace.plotting import presets
    scheme = str(body.get("spin_colors") or "auto")
    if scheme != "auto" and scheme not in presets.SPIN_SCHEMES and scheme != "custom":
        raise ValueError("spin_colors is auto, custom or one of " + ", ".join(presets.SPIN_SCHEMES))
    out = {"spin_colors": scheme}
    for k in ("spin_up_color", "spin_down_color", "band_color"):
        v = body.get(k)
        if v not in (None, ""):
            if not (isinstance(v, str) and presets._HEXCOLOR.match(v)):
                raise ValueError(f"{k} is a #rrggbb color")
            out[k] = v.lower()
    bstyle = str(body.get("band_style") or "auto")
    if bstyle not in presets.BAND_STYLES:
        raise ValueError("band_style is one of " + ", ".join(presets.BAND_STYLES))
    out["band_style"] = bstyle
    dash = str(body.get("spin_down_style") or "dash")
    if dash not in ("dash", "dot", "dashdot", "solid"):
        raise ValueError("spin_down_style is dash, dot, dashdot or solid")
    out["spin_down_style"] = dash
    return out


def export_job(st, body: Dict[str, Any], ctx) -> Dict[str, Any]:
    """``mace plotting --gallery`` in a cache folder; the figure's name."""
    from . import clirun
    kind = _kind(body.get("kind"))
    paths = _paths(st, body.get("paths"), PAGE_MAX)
    fmt = str(body.get("format") or "svg")
    if fmt not in G.FIGURE_FORMATS:
        raise ValueError(f"the format is one of {', '.join(G.FIGURE_FORMATS)}")
    try:
        cols = int(body.get("cols") or 3)
    except (TypeError, ValueError):
        raise ValueError("columns is a number")
    cols = max(1, min(cols, G.MAX_COLS))
    shared = body.get("shared", True) is not False
    ewin = _ewin(lambda k: body.get(k))
    # the figure in the GUI's plot style and DOS colors (Settings), as the screen draws it
    from mace.plotting import presets
    style = str(body.get("style") or "default")
    if style not in presets.PRESETS:
        raise ValueError("the plot style is one of " + ", ".join(presets.ORDER))
    dos_colors = str(body.get("dos_colors") or "auto")
    if dos_colors not in presets.DOS_MODES:
        raise ValueError("dos_colors is one of " + ", ".join(presets.DOS_MODES))
    try:
        dos_smooth = float(body.get("dos_smooth") or 0)
    except (TypeError, ValueError):
        raise ValueError("dos_smooth is a number of eV")
    if not 0 <= dos_smooth <= 1:
        raise ValueError("dos_smooth is from 0 (off) to 1 eV")
    from .plots import energy_reference
    shift, reference = energy_reference(lambda k: body.get(k))
    look = {"style": style, "dos_colors": dos_colors, "dos_smooth": dos_smooth, **_spin_look(body),
            "shift": shift, "reference": reference}
    # the child reads the served folder's mace-folder.toml: with one, every value is passed, the
    # defaults too, so its [plots] values do not replace what the grid shows
    from . import folderprefs
    env = folderprefs.cli_env(st.root)
    look["explicit"] = env.get(folderprefs.F.ENV, "none") != "none"
    stamp = "|".join(f"{p}:{p.stat().st_mtime_ns}:{p.stat().st_size}" for p in paths)
    key = hashlib.sha1(f"gallery|{kind}|{cols}|{shared}|{ewin}|{fmt}|{sorted(look.items())}|{stamp}".encode()).hexdigest()[:16]
    name = f"gallery_{kind}.{fmt}"
    shown = "mace " + " ".join(_quote(a) for a in G.command(kind, [st.rel(p) for p in paths], cols, name, shared, ewin, **look))
    work = st.disk.acquire(key)
    ok = False
    try:
        target = work / name
        if not target.is_file():
            if ctx is not None:
                ctx.progress(0.1, f"Drawing {len(paths)} panes with mace plotting --gallery")
            args = G.command(kind, [str(p) for p in paths], cols, str(target), shared, ewin, **look)
            args += ["--max", str(len(paths))]
            r = clirun.run(clirun.mace_argv("--no-banner", *args), work, ctx, timeout=900, env=env)
            (work / "log.txt").write_text(r.log)
            if r.returncode != 0 or not target.is_file():
                tail = (r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ["no output"])[-1][-400:]
                raise RuntimeError(f"mace plotting --gallery failed: {tail}")
        ok = True
        return {"key": key, "name": name, "command": shown, "size": target.stat().st_size}
    finally:
        st.disk.release(key, remove=not ok)


def csv_text(st, body: Dict[str, Any]) -> str:
    paths = _paths(st, body.get("paths"), PAGE_MAX)
    try:
        start = max(0, int(body.get("offset") or 0)) + 1
    except (TypeError, ValueError):
        start = 1
    caps = G.captions(paths, db=_db_of(st, body), stop=st.root, rel=lambda p: st.rel(p), start=start)
    return G.captions_csv(caps["rows"])


def handle(handler, method: str, route: str, arg):
    """The gallery/* routes; ``arg`` is the query (GET) or the body (POST)."""
    st = handler.state
    if method == "GET" and route == "gallery/pane":
        return handler._json(pane(st, lambda k: handler._q(arg, k)))
    if method == "POST" and route == "gallery/collect":
        return handler._json(collect(st, arg))
    if method == "POST" and route == "gallery/csv":
        text = csv_text(st, arg).encode()
        return handler._send(200, text, "text/csv; charset=utf-8",
                             {"Cache-Control": "no-store", "Content-Disposition": 'attachment; filename="gallery_captions.csv"'})
    if method == "POST" and route == "gallery/export":
        from . import tasks as gui_tasks
        kind = _kind(arg.get("kind"))
        n = len(_paths(st, arg.get("paths"), PAGE_MAX))
        if str(arg.get("format") or "svg") not in G.FIGURE_FORMATS:
            raise ValueError(f"the format is one of {', '.join(G.FIGURE_FORMATS)}")
        _ewin(lambda k: arg.get(k))
        t = st.tasks.submit("plot", f"Plot gallery: {n} {G.LABELS[kind].lower()} panes", "", {},
                            client=handler.headers.get("X-Mace-Client"), inline=lambda ctx: export_job(st, arg, ctx))
        return gui_tasks.answer(handler, st.tasks, t, 0.3 if arg.get("detach") else 600)
    return handler._error(404, f"unknown API route {route}")
