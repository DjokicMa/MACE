"""An edited band k-path: one representation for opt2d3, its config JSON and mace gui.

A path is a list of *segments*, in the order they are drawn or computed::

    {"from": "GAMMA", "to": "Y"}                                  # a point of the computed path
    {"from": "Y", "to": "P", "k_to": [0.25, 0.25, 0.0]}            # a point you added (fractional k)
    {"from": "K", "to": "GAMMA", "hidden": true}                   # kept for the editor, not computed

Labels are CRYSTAL/seekpath ASCII names (GAMMA, F_0, DELTA_0, X'), the form
the BAND title records them in. Fractional coordinates are in the basis of the
deck's own segment records (CRYSTAL's primitive reciprocal cell), and are
given only for points that are not on the computed path: a path that names
only computed points carries over to any material of the same lattice type,
as an automatic path does.

Two consecutive segments that do not meet (one ends at A, the next starts at
L) are a discontinuity, written "A|L" on the plot and "-|-" in the BAND title.
A run of segments without one is a *piece*.

The same operations exist in mace/gui/static/js/kpath.js; tests compare the
two on the same edits.
"""
from __future__ import annotations

import copy
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

Segment = Dict[str, Any]

GREEK = {"Γ": "GAMMA", "Δ": "DELTA", "Σ": "SIGMA", "Λ": "LAMBDA", "Π": "PI"}
MAX_SEGMENTS = 20          # BAND's NLINE (manual: "max 20")
_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_']{0,15}$")


# ---------------------------------------------------------------- labels
def canonical_label(label: Any) -> str:
    """'Γ', 'G', 'gamma' -> 'GAMMA'; 'F₀' / 'F<sub>0</sub>' / '$F_{0}$' -> 'F_0'; else as given."""
    s = str(label or "").strip()
    s = re.sub(r"<sub>(.*?)</sub>", r"_\1", s)
    s = s.replace("$", "").replace("{", "").replace("}", "")
    s = s.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
    for g, name in GREEK.items():
        s = s.replace(g, name)
    if s.upper() in ("G", "GAMMA"):
        return "GAMMA"
    return s


def label_problem(label: Any) -> Optional[str]:
    s = canonical_label(label)
    if not s:
        return "a point needs a label"
    if not _LABEL.match(s):
        return f"{label!r} is not a usable label (letters, digits, _ and ', up to 16 characters)"
    return None


# ---------------------------------------------------------------- structure
def _k(seg: Segment, end: str) -> Optional[List[float]]:
    k = seg.get(f"k_{end}")
    return [float(v) for v in k] if k is not None else None


def _same_k(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> bool:
    if a is None or b is None:
        return True                     # a label without coordinates: the label decides
    return all(abs(float(x) - float(y)) < 1e-6 for x, y in zip(a, b))


def joined(a: Segment, b: Segment) -> bool:
    """True when segment b starts where segment a ends (no discontinuity between them)."""
    return (canonical_label(a["to"]) == canonical_label(b["from"])
            and _same_k(_k(a, "to"), _k(b, "from")))


def pieces(segs: Sequence[Segment]) -> List[List[int]]:
    """Indices of segments grouped into continuous runs."""
    out: List[List[int]] = []
    for i, s in enumerate(segs):
        if out and joined(segs[out[-1][-1]], s):
            out[-1].append(i)
        else:
            out.append([i])
    return out


def visible(segs: Sequence[Segment]) -> List[Segment]:
    return [s for s in segs if not s.get("hidden")]


def path_tokens(segs: Sequence[Segment]) -> List[str]:
    """Labels in order with "|" at each discontinuity (the BAND title's form)."""
    segs = visible(segs)
    out: List[str] = []
    for i, s in enumerate(segs):
        if i == 0:
            out += [canonical_label(s["from"]), canonical_label(s["to"])]
        elif joined(segs[i - 1], s):
            out.append(canonical_label(s["to"]))
        else:
            out += ["|", canonical_label(s["from"]), canonical_label(s["to"])]
    return out


def from_tokens(tokens: Sequence[str], ks: Optional[Sequence[Sequence[float]]] = None) -> List[Segment]:
    """Segments from a token list with "|" breaks; ``ks`` (one 6-vector per
    segment) adds each segment's fractional coordinates."""
    segs: List[Segment] = []
    run: List[str] = []
    for t in list(tokens) + ["|"]:
        if t == "|":
            for a, b in zip(run, run[1:]):
                segs.append({"from": canonical_label(a), "to": canonical_label(b)})
            run = []
        else:
            run.append(t)
    if ks is not None and len(ks) == len(segs):
        for s, k in zip(segs, ks):
            s["k_from"] = [float(v) for v in k[:3]]
            s["k_to"] = [float(v) for v in k[3:6]]
    return segs


def normalize(band_path: Any) -> List[Segment]:
    """A JSON band_path as a list of clean segment dicts (raises ValueError on a bad shape)."""
    if not isinstance(band_path, list):
        raise ValueError("band_path is a list of segments ({\"from\": ..., \"to\": ...})")
    out = []
    for n, s in enumerate(band_path, 1):
        if not isinstance(s, dict) or "from" not in s or "to" not in s:
            raise ValueError(f"segment {n} needs \"from\" and \"to\" labels")
        seg: Segment = {"from": canonical_label(s["from"]), "to": canonical_label(s["to"])}
        for end in ("from", "to"):
            k = s.get(f"k_{end}")
            if k is not None:
                if not (isinstance(k, (list, tuple)) and len(k) == 3):
                    raise ValueError(f"segment {n}: k_{end} is three fractional coordinates")
                try:
                    seg[f"k_{end}"] = [float(v) for v in k]
                except (TypeError, ValueError):
                    raise ValueError(f"segment {n}: k_{end} is three numbers")
        if s.get("hidden"):
            seg["hidden"] = True
        out.append(seg)
    return out


# ---------------------------------------------------------------- operations
# Each returns a new list; the input is not changed.
def move(segs: Sequence[Segment], i: int, j: int) -> List[Segment]:
    out = copy.deepcopy(list(segs))
    if 0 <= i < len(out) and 0 <= j < len(out):
        out.insert(j, out.pop(i))
    return out


def reverse(segs: Sequence[Segment], i: int) -> List[Segment]:
    out = copy.deepcopy(list(segs))
    s = out[i]
    r: Segment = {"from": s["to"], "to": s["from"]}
    if "k_to" in s:
        r["k_from"] = s["k_to"]
    if "k_from" in s:
        r["k_to"] = s["k_from"]
    if s.get("hidden"):
        r["hidden"] = True
    out[i] = r
    return out


def remove(segs: Sequence[Segment], i: int) -> List[Segment]:
    out = copy.deepcopy(list(segs))
    del out[i]
    return out


def toggle_hidden(segs: Sequence[Segment], i: int) -> List[Segment]:
    out = copy.deepcopy(list(segs))
    if out[i].get("hidden"):
        out[i].pop("hidden")
    else:
        out[i]["hidden"] = True
    return out


def remove_point(segs: Sequence[Segment], i: int, end: str, join: bool) -> List[Segment]:
    """Remove the point at the ``end`` ("from"/"to") of segment i.

    Inside a piece the point is shared by two segments. ``join`` (a path to
    compute) replaces them by one straight segment between their outer
    points ("Remove H: joins L→A"). Without it (a computed path on display,
    where no data runs between those points) both are hidden, leaving a jump
    L|A ("Remove H: hides L→H and H→A"), as reversible as any hidden segment.
    A piece's end point takes its one segment (removed, or hidden).
    """
    out = copy.deepcopy(list(segs))
    for k in removal(out, i, end):
        if not join:
            out[k]["hidden"] = True
    affected = removal(out, i, end)
    if not join:
        return out
    if len(affected) == 2:
        a, b = affected
        merged: Segment = {"from": out[a]["from"], "to": out[b]["to"]}
        if "k_from" in out[a]:
            merged["k_from"] = out[a]["k_from"]
        if "k_to" in out[b]:
            merged["k_to"] = out[b]["k_to"]
        out[a:b + 1] = [merged]
    else:
        del out[affected[0]]
    return out


def removal(segs: Sequence[Segment], i: int, end: str) -> List[int]:
    """The segments removing the point at ``end`` of segment i touches: the two that
    meet there (in order), or segment i alone at a piece's end."""
    j = i + 1 if end == "to" else i - 1
    a, b = (i, j) if end == "to" else (j, i)
    if 0 <= a < len(segs) and 0 <= b < len(segs) and joined(segs[a], segs[b]):
        return [a, b]
    return [i]


def main_piece(segs: Sequence[Segment]) -> int:
    """Index (in pieces()) of the longest piece; the first of equals."""
    ps = pieces(segs)
    return max(range(len(ps)), key=lambda n: (len(ps[n]), -n)) if ps else 0


def _regroup(segs: Sequence[Segment], order: List[List[int]]) -> List[Segment]:
    return [copy.deepcopy(segs[i]) for piece in order for i in piece]


def jumps_first(segs: Sequence[Segment]) -> List[Segment]:
    """The pieces after a discontinuity before the main piece."""
    ps = pieces(segs)
    m = main_piece(segs)
    return _regroup(segs, [p for n, p in enumerate(ps) if n != m] + [ps[m]]) if ps else []


def jumps_last(segs: Sequence[Segment]) -> List[Segment]:
    ps = pieces(segs)
    m = main_piece(segs)
    return _regroup(segs, [ps[m]] + [p for n, p in enumerate(ps) if n != m]) if ps else []


def continuous_only(segs: Sequence[Segment]) -> List[Segment]:
    """Only the main piece: every jump ("|") and what follows it is dropped."""
    ps = pieces(segs)
    return _regroup(segs, [ps[main_piece(segs)]]) if ps else []


def move_piece(segs: Sequence[Segment], n: int, m: int) -> List[Segment]:
    """Move piece n to position m among the pieces."""
    ps = pieces(segs)
    if not (0 <= n < len(ps) and 0 <= m < len(ps)):
        return copy.deepcopy(list(segs))
    ps.insert(m, ps.pop(n))
    return _regroup(segs, ps)


def add_point(segs: Sequence[Segment], label: str, k: Sequence[float],
              from_label: Optional[str] = None, from_k: Optional[Sequence[float]] = None) -> List[Segment]:
    """A segment from the path's last point (or ``from_label``) to a new point at fractional ``k``."""
    out = copy.deepcopy(list(segs))
    seg: Segment = {"to": canonical_label(label), "k_to": [float(v) for v in k]}
    if from_label is not None:
        seg["from"] = canonical_label(from_label)
        if from_k is not None:
            seg["k_from"] = [float(v) for v in from_k]
    elif out:
        last = out[-1]
        seg["from"] = last["to"]
        if "k_to" in last:
            seg["k_from"] = list(last["k_to"])
    else:
        seg["from"] = "GAMMA"
        seg["k_from"] = [0.0, 0.0, 0.0]
    out.append({"from": seg["from"], "to": seg["to"],
                **({"k_from": seg["k_from"]} if "k_from" in seg else {}), "k_to": seg["k_to"]})
    return out


OPS = {"move": move, "reverse": reverse, "remove": remove, "toggle_hidden": toggle_hidden,
       "remove_point": remove_point, "jumps_first": jumps_first, "jumps_last": jumps_last,
       "continuous_only": continuous_only, "move_piece": move_piece, "add_point": add_point}


def apply_ops(segs: Sequence[Segment], ops: Sequence[Sequence[Any]]) -> List[Segment]:
    """Apply [["reverse", 2], ["move", 0, 3], ...] in order (for tests and scripts)."""
    out = copy.deepcopy(list(segs))
    for op in ops:
        out = OPS[op[0]](out, *op[1:])
    return out


# ---------------------------------------------------------------- points and checks
def points(segs: Sequence[Segment]) -> Dict[str, List[float]]:
    """Label -> fractional k of every point with coordinates (first occurrence wins)."""
    out: Dict[str, List[float]] = {}
    for s in segs:
        for end in ("from", "to"):
            k = _k(s, end)
            lab = canonical_label(s[end])
            if k is not None and lab not in out:
                out[lab] = k
    return out


# The k components a slab (2D) or a polymer (1D) must give as 0: CRYSTAL23
# manual p. 309, BAND note 3, "in two-(one-) dimensional cases I3, J3 (I2, I3,
# J2, J3) are formally input as zero".
_AXES = "xyz"


def out_of_plane_problem(label: str, k: Optional[Sequence[float]], dimensionality: int) -> Optional[str]:
    """Why the fractional ``k`` of point ``label`` cannot be written for a
    ``dimensionality``-D system (a slab or a polymer), or None."""
    if k is None or dimensionality not in (1, 2):
        return None
    bad = [i for i in range(dimensionality, 3) if abs(float(k[i])) > 1e-9]
    if not bad:
        return None
    kind = "a slab (2D)" if dimensionality == 2 else "a polymer (1D)"
    comps = ", ".join(f"k_{_AXES[i]} = {float(k[i]):g}" for i in bad)
    need = "k_z" if dimensionality == 2 else "k_y and k_z"
    return (f"{label} has {comps}: in {kind} {need} must be 0 (CRYSTAL23 manual p. 309, "
            f"BAND note 3: the non-periodic components are formally input as zero)")


def validate(band_path: Any, computed: Optional[Sequence[Segment]] = None,
             need_k: bool = False, dimensionality: int = 3) -> Tuple[List[str], List[str]]:
    """(errors, warnings) for an edited path.

    Errors stop the deck: no visible segment, more than 20 segments (BAND's
    NLINE), a bad label or coordinate, a zero-length segment, a point with no
    coordinates where they are needed, a non-zero non-periodic k component
    of a slab (``dimensionality`` 2) or a polymer (1). Warnings: computed
    high-symmetry points the edited path no longer visits.
    """
    errors: List[str] = []
    warnings: List[str] = []
    try:
        segs = normalize(band_path)
    except ValueError as exc:
        return [str(exc)], []
    shown = visible(segs)
    if not shown:
        return ["At least one segment must stay visible"], []
    if len(shown) > MAX_SEGMENTS:
        errors.append(f"BAND takes at most {MAX_SEGMENTS} segments; this path has {len(shown)}")
    known = points(computed or []) if computed else {}
    known_labels = {canonical_label(s[e]) for s in (computed or []) for e in ("from", "to")}
    for n, s in enumerate(shown, 1):
        for end in ("from", "to"):
            p = label_problem(s[end])
            if p:
                errors.append(f"segment {n}: {p}")
            k = _k(s, end)
            if k is not None and not all(math.isfinite(v) and abs(v) <= 10 for v in k):
                errors.append(f"segment {n}: the coordinates of {s[end]} must be numbers between -10 and 10")
            elif out_of_plane_problem(s[end], k, dimensionality):
                errors.append(f"segment {n}: {out_of_plane_problem(s[end], k, dimensionality)}")
            if need_k and k is None and canonical_label(s[end]) not in known:
                errors.append(f"segment {n}: {s[end]} is not a point of the computed path; give its coordinates")
        ka = _k(s, "from") if "k_from" in s else known.get(canonical_label(s["from"]))
        kb = _k(s, "to") if "k_to" in s else known.get(canonical_label(s["to"]))
        if canonical_label(s["from"]) == canonical_label(s["to"]) and (ka is None or kb is None or _same_k(ka, kb)):
            errors.append(f"segment {n} ({s['from']} → {s['to']}) has no length")
        elif ka is not None and kb is not None and _same_k(ka, kb):
            errors.append(f"segment {n} ({s['from']} → {s['to']}) has no length")
    if computed:
        used = {canonical_label(s[e]) for s in shown for e in ("from", "to")}
        skipped = [lab for lab in _ordered_labels(computed) if lab not in used and lab in known_labels]
        if skipped:
            warnings.append("The path no longer visits " + ", ".join(skipped)
                            + " (high-symmetry points of the computed path); band extrema there would be missed")
    return errors, warnings


def _ordered_labels(segs: Sequence[Segment]) -> List[str]:
    seen: List[str] = []
    for s in segs:
        for e in ("from", "to"):
            lab = canonical_label(s[e])
            if lab not in seen:
                seen.append(lab)
    return seen


def resolve(band_path: Any, computed: Sequence[Segment]) -> List[Segment]:
    """The visible segments with fractional k on every end: from the segment
    itself, else from the computed segment with the same two labels (either
    direction), else from the computed point of that label."""
    segs = visible(normalize(band_path))
    by_pair = {}
    for c in computed:
        if "k_from" in c and "k_to" in c:
            a, b = canonical_label(c["from"]), canonical_label(c["to"])
            by_pair.setdefault((a, b), (c["k_from"], c["k_to"]))
            by_pair.setdefault((b, a), (c["k_to"], c["k_from"]))
    known = points(computed)
    out = []
    for s in segs:
        a, b = canonical_label(s["from"]), canonical_label(s["to"])
        pair = by_pair.get((a, b))
        ka = s.get("k_from") or (pair[0] if pair else known.get(a))
        kb = s.get("k_to") or (pair[1] if pair else known.get(b))
        r: Segment = {"from": a, "to": b}
        if ka is not None:
            r["k_from"] = list(ka)
        if kb is not None:
            r["k_to"] = list(kb)
        out.append(r)
    return out


def same_path(band_path: Any, computed: Sequence[Segment]) -> bool:
    """True when ``band_path`` is the computed path unchanged (then nothing is rewritten)."""
    try:
        segs = normalize(band_path)
    except ValueError:
        return False
    if any(s.get("hidden") for s in segs) or len(segs) != len(computed):
        return False
    for s, c in zip(segs, computed):
        if (canonical_label(s["from"]), canonical_label(s["to"])) != (canonical_label(c["from"]), canonical_label(c["to"])):
            return False
        for end in ("from", "to"):
            k = s.get(f"k_{end}")
            if k is not None and not _same_k(k, c.get(f"k_{end}")):
                return False
    return True


# ---------------------------------------------------------------- BAND decks
def parse_band_deck(text: str) -> Optional[Dict[str, Any]]:
    """A BAND deck as {"title_head", "source", "header": [7 ints/str], "shrink",
    "segments": [...], "labels_mode": bool}; None when it is not one.

    Segments carry fractional k (ints / ISS) when the deck has coordinates;
    their labels come from the title when it names every segment, else from
    the coordinates."""
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    if len(lines) < 3 or lines[0].strip() != "BAND":
        return None
    title = lines[1]
    head = lines[2].split()
    if len(head) < 7:
        return None
    try:
        nline, iss = int(head[0]), int(head[1])
    except ValueError:
        return None
    recs = [ln.split() for ln in lines[3:3 + nline]]
    parts = title.split(" - ")
    path_part = parts[-1].strip() if len(parts) >= 3 else ""
    tokens = [t for t in path_part.split("-") if t.strip()] if path_part else []
    tokens = [t.strip() for t in tokens]
    # a jump written the way the plot writes it ("H|P" between two labels) is the same jump as "-|-"
    expanded: List[str] = []
    for t in tokens:
        pieces = [x.strip() for x in t.split("|")] if "|" in t and t != "|" else [t]
        for n, piece in enumerate(pieces):
            if n:
                expanded.append("|")
            if piece:
                expanded.append(piece)
    tokens = expanded
    named = from_tokens(tokens) if tokens and all(t == "|" or _LABEL.match(t) for t in tokens) else []
    labels_mode = iss == 0
    segs: List[Segment] = []
    if labels_mode:
        segs = [{"from": canonical_label(r[0]), "to": canonical_label(r[1])} for r in recs if len(r) >= 2]
    else:
        ks = []
        for r in recs:
            if len(r) < 6:
                return None
            ks.append([int(v) / iss for v in r[:6]])
        if len(named) == len(ks):
            segs = named
            for s, k in zip(segs, ks):
                s["k_from"], s["k_to"] = k[:3], k[3:6]
        else:
            names: Dict[Tuple[float, ...], str] = {}

            def name(k):
                key = tuple(round(v, 6) for v in k)
                if key not in names:
                    names[key] = "GAMMA" if not any(key) else f"K{len([n for n in names.values() if n != 'GAMMA']) + 1}"
                return names[key]
            segs = [{"from": name(k[:3]), "to": name(k[3:6]), "k_from": k[:3], "k_to": k[3:6]} for k in ks]
    head_part = " - ".join(parts[:-1]) if path_part and named else title
    return {"title": title, "title_head": head_part, "header": head, "shrink": iss,
            "segments": segs, "labels_mode": labels_mode, "named": bool(named) or labels_mode}


def write_band_deck(computed_deck: str, band_path: Any, scale, dimensionality: int = 3) -> str:
    """The BAND deck for ``band_path``, from the deck of the computed path.

    ``scale`` is d3_kpoints.scale_kpoint_segments (fractional -> integers and
    the ISS that makes them exact). A path that keeps CRYSTAL's label form
    (no added coordinates, labels deck) stays in labels; anything else is
    written as coordinates. The title names the edited path and says it was
    edited."""
    deck = parse_band_deck(computed_deck)
    if deck is None:
        raise ValueError("the computed deck is not a BAND deck")
    computed = deck["segments"]
    errors, _ = validate(band_path, computed, need_k=not deck["labels_mode"], dimensionality=dimensionality)
    if errors:
        raise ValueError("; ".join(errors))
    segs = resolve(band_path, computed)
    head = list(deck["header"])
    labels_only = deck["labels_mode"] and not any("k_from" in s or "k_to" in s for s in visible(normalize(band_path)))
    tokens = path_tokens(segs)
    title = f"{deck['title_head']} (edited) - {'-'.join(tokens)}"
    if labels_only:
        head[0] = str(len(segs))
        crystal = lambda lab: "G" if lab == "GAMMA" else lab    # noqa: E731
        recs = [f"{crystal(s['from'])} {crystal(s['to'])}" for s in segs]
    else:
        missing = [s["from"] if "k_from" not in s else s["to"] for s in segs if "k_from" not in s or "k_to" not in s]
        if missing:
            raise ValueError(f"no coordinates for {', '.join(dict.fromkeys(missing))}: give them in band_path")
        frac = [list(s["k_from"]) + list(s["k_to"]) for s in segs]
        ints, iss = scale(frac, deck["shrink"] if deck["shrink"] > 0 else 16)
        head[0], head[1] = str(len(segs)), str(iss)
        recs = [f"{r[0]} {r[1]} {r[2]}  {r[3]} {r[4]} {r[5]}" for r in ints]
    return "\n".join(["BAND", title, " ".join(head)] + recs + ["END"])


# ---------------------------------------------------------------- the opt2d3 prompt
HELP = """  r N        reverse segment N            d N        remove segment N
  m N P      move segment N to place P    p LABEL    remove point LABEL (its neighbors are joined)
  a LABEL x y z   add a point at the end (fractional coordinates of the deck's cell)
  c          continuous only (drop every jump "|" and what follows it)
  jf / jl    jumps first / jumps last     u          undo
  reset      back to the computed path    Enter      done"""


def describe(segs: Sequence[Segment]) -> str:
    """Numbered segments, "|" between pieces: '1 GAMMA→X  2 X→M | 3 K→GAMMA'."""
    out = []
    for n, s in enumerate(segs):
        if n and not joined(segs[n - 1], s):
            out.append("|")
        out.append(f"{n + 1} {s['from']}→{s['to']}")
    return "  ".join(out)


def edit_interactively(computed: Sequence[Segment], ask, say,
                       dimensionality: int = 3) -> Optional[List[Segment]]:
    """The opt2d3 k-path editor: ``ask(prompt) -> str``, ``say(text)``.
    Returns the edited path, or None when it is the computed one.
    ``dimensionality`` 2 (slab) or 1 (polymer) refuses an added point with a
    non-zero non-periodic k component."""
    segs = copy.deepcopy(list(computed))
    history: List[List[Segment]] = []
    say("Computed path: " + " ".join(path_tokens(segs)))
    say(HELP)
    while True:
        say("  " + describe(segs))
        line = ask("k-path edit (Enter when done): ").strip()
        if not line:
            errors, warnings = validate(segs, computed, dimensionality=dimensionality)
            for w in warnings:
                say("  Note: " + w)
            if errors:
                for e in errors:
                    say("  " + e)
                continue
            return None if same_path(segs, computed) else segs
        parts = line.split()
        cmd, args = parts[0].lower(), parts[1:]
        try:
            if cmd in ("h", "help", "?"):
                say(HELP)
                continue
            if cmd == "u":
                if history:
                    segs = history.pop()
                else:
                    say("  Nothing to undo")
                continue
            before = copy.deepcopy(segs)
            n = lambda i: int(args[i]) - 1     # noqa: E731
            if cmd == "r":
                segs = reverse(segs, n(0))
            elif cmd == "d":
                segs = remove(segs, n(0))
            elif cmd == "m":
                segs = move(segs, n(0), n(1))
            elif cmd == "p":
                lab = canonical_label(args[0])
                inner = [i for i, s in enumerate(segs[:-1]) if canonical_label(s["to"]) == lab and joined(s, segs[i + 1])]
                ends = [(i, e) for i, s in enumerate(segs) for e in ("from", "to") if canonical_label(s[e]) == lab]
                if inner:
                    segs = remove_point(segs, inner[0], "to", join=True)
                elif ends:
                    segs = remove_point(segs, ends[0][0], ends[0][1], join=True)
                else:
                    say(f"  {args[0]} is not on the path")
                    continue
            elif cmd == "a":
                p = label_problem(args[0])
                if p:
                    say("  " + p)
                    continue
                if len(args) < 4:
                    raise ValueError
                k = [float(v) for v in args[1:4]]
                p = out_of_plane_problem(args[0], k, dimensionality)
                if p:
                    say("  " + p)
                    continue
                segs = add_point(segs, args[0], k)
            elif cmd == "c":
                segs = continuous_only(segs)
            elif cmd == "jf":
                segs = jumps_first(segs)
            elif cmd == "jl":
                segs = jumps_last(segs)
            elif cmd == "reset":
                segs = copy.deepcopy(list(computed))
            else:
                say(f"  Unknown command {parts[0]!r} (h for help)")
                continue
        except (IndexError, ValueError):
            say("  That command needs its numbers (h for help)")
            segs = before
            continue
        if not segs:
            say("  At least one segment must stay")
            segs = before
            continue
        history.append(before)
