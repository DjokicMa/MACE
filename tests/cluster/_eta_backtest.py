"""Shared helpers for the ETA backtests (test-only; no package code).

The corpus in test/ holds conformer and symmetry families and OPT/SP
near-duplicates (an SP of an optimized geometry repeats the optimization's
first SCF). A split by basename leaks such families into both halves, so the
held-out scores read too high. ``family_of`` groups the outputs by material
name and by identical opening SCF energies; ``half`` puts a whole family in
one half. Coverage figures carry a 95 % bootstrap interval over families.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_DATA = REPO_ROOT / "test"

# the material name ends where the calculation's own words start
_CUT = re.compile(r"(_sym_|_mp-|_MOLECULE|_BULK|_SLAB|_rev\d|_opt|_OPT|_sp\b|_freq|_CRYSTAL|_2x2)")
_CONF = re.compile(r"[-_]conf\d+", re.I)
_CYC_E = re.compile(r"^\s*CYC\s+\d+\s+ETOT\(AU\)\s+(\S+)")


def need_corpus(*paths):
    for p in paths or (TEST_DATA,):
        if not Path(p).exists():
            pytest.skip("test/ data corpus not present")


def name_key(path) -> str:
    """Material part of a corpus basename: no calc suffix, no -confN."""
    base = Path(path).name
    base = re.sub(r"\.out$", "", base)
    base = _CONF.sub("", base)
    if base.startswith("4LG"):
        # 4LG_<adsorbate/stacking variants>: one family of layered slabs and their fragments
        return "4lg_fsi" if "FSI" in base else "4lg"
    m = _CUT.search(base)
    key = base[:m.start()] if m and m.start() > 0 else base
    for suf in ("_temp", "_sp", "_opt", "_freq"):
        key = key.replace(suf, "")
    return key.lower().replace(".", ",")


def scf_fingerprint(text: str) -> Optional[Tuple[float, ...]]:
    """The first five SCF total energies rounded to 1e-8 (None if fewer)."""
    vals = []
    for line in text.splitlines():
        m = _CYC_E.match(line)
        if m:
            try:
                vals.append(round(float(m.group(1).replace("D", "E")), 8))
            except ValueError:
                return None
            if len(vals) == 5:
                return tuple(vals)
    return None


def family_of(paths: Iterable, texts: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """path -> family key. Same material name, or the same opening SCF
    energies as another output (union-find), share a family."""
    paths = [str(p) for p in paths]
    parent = {p: p for p in paths}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    by_name: Dict[str, str] = {}
    by_fp: Dict[Tuple, str] = {}
    for p in paths:
        k = name_key(p)
        if k in by_name:
            union(p, by_name[k])
        else:
            by_name[k] = p
        text = texts[p] if texts and p in texts else Path(p).read_text(errors="replace")
        fp = scf_fingerprint(text)
        if fp is not None:
            if fp in by_fp:
                union(p, by_fp[fp])
            else:
                by_fp[fp] = p
    groups: Dict[str, List[str]] = {}
    for p in paths:
        groups.setdefault(find(p), []).append(p)
    out = {}
    for members in groups.values():
        key = min(name_key(m) for m in members)
        for m in members:
            out[m] = key
    return out


def half(family: str) -> str:
    return "held" if int(hashlib.sha1(family.encode()).hexdigest(), 16) % 2 else "dev"


def corpus_families(dirs: Sequence[str] = ("OPT", "SP", "FREQ")) -> Dict[str, str]:
    """Family of every corpus .out across the given folders (so an SP lands
    with the optimization it repeats)."""
    paths = []
    for d in dirs:
        paths += sorted((TEST_DATA / d).glob("*.out"))
    return family_of(paths)


def subsample(ks: Sequence[int], cap: int = 30) -> List[int]:
    """At most ``cap`` evenly spaced items (long runs would dominate)."""
    ks = list(ks)
    if len(ks) <= cap:
        return ks
    return [ks[round(i * (len(ks) - 1) / (cap - 1))] for i in range(cap)]


# ------------------------------------------------------------------ statistics
def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def bootstrap_ci(values: Sequence[float], groups: Sequence[str], n: int = 2000,
                 seed: int = 0) -> Tuple[float, float, float]:
    """(point, low, high): mean of ``values`` and its 95 % interval from
    resampling whole groups (families) with replacement."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    by: Dict[str, List[float]] = {}
    for v, g in zip(values, groups):
        by.setdefault(g, []).append(float(v))
    keys = sorted(by)
    point = sum(values) / len(values)
    if all(len(by[k]) == 1 for k in keys):
        lo, hi = wilson(int(round(sum(values))), len(values))
        return point, lo, hi
    rng = random.Random(seed)
    sums = [(sum(by[k]), len(by[k])) for k in keys]
    stats = []
    for _ in range(n):
        s = c = 0.0
        for _ in keys:
            a, b = sums[rng.randrange(len(keys))]
            s += a
            c += b
        stats.append(s / c)
    stats.sort()
    return point, stats[int(0.025 * n)], stats[int(0.975 * n) - 1]


def family_mean_ci(means: Sequence[float], n: int = 2000, seed: int = 0) -> Tuple[float, float, float]:
    """Mean of per-family values (each family weighted 1) and its 95 %
    percentile-bootstrap interval."""
    if not means:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    k = len(means)
    stats = sorted(sum(means[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return sum(means) / k, stats[int(0.025 * n)], stats[int(0.975 * n) - 1]


def floor05(x: float) -> float:
    """Gate floor: rounded down to 0.05."""
    return math.floor(x * 20 + 1e-9) / 20


def quantile(xs: Sequence[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = q * (len(xs) - 1)
    lo = int(math.floor(i))
    hi = min(len(xs) - 1, lo + 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def summarize(rows: Sequence[dict]) -> dict:
    """Scores of backtest rows. Each row: family, run, ranged (bool),
    hit (bool|None), capped (bool), low, high, true, thit (bool|None).
    Coverage counts every returned range; informative coverage only the
    trend-bounded ones; width = high/low per range (low >= 1)."""
    out = {"n": len(rows), "runs": len({r["run"] for r in rows}), "families": len({r["family"] for r in rows})}
    ranged = [r for r in rows if r["ranged"]]
    out["unknown"] = 1 - len(ranged) / len(rows) if rows else float("nan")
    out["coverage"] = bootstrap_ci([r["hit"] for r in ranged], [r["family"] for r in ranged])
    inf = [r for r in ranged if not r["capped"]]
    out["informative"] = bootstrap_ci([r["hit"] for r in inf], [r["family"] for r in inf])
    out["capped_share"] = (sum(r["capped"] for r in ranged) / len(ranged)) if ranged else 0.0
    tr = [r for r in ranged if r.get("thit") is not None]
    out["time"] = bootstrap_ci([r["thit"] for r in tr], [r["family"] for r in tr])
    widths = [r["high"] / max(1.0, r["low"]) for r in ranged if r["high"] is not None]
    out["width_p50"], out["width_p90"] = quantile(widths, 0.5), quantile(widths, 0.9)
    fam: Dict[str, List[int]] = {}
    for r in ranged:
        fam.setdefault(r["family"], []).append(r["hit"])
    fm = [sum(v) / len(v) for v in fam.values()]
    out["per_family"] = family_mean_ci(fm)
    out["too_short"] = sum(1 for r in ranged if not r["hit"] and r["high"] is not None and r["true"] > r["high"])
    out["too_long"] = sum(1 for r in ranged if not r["hit"] and r["true"] < r["low"])
    return out


def fmt(s: dict) -> str:
    def ci(t):
        return f"{t[0]:.0%} ({t[1]:.0%}-{t[2]:.0%})"
    return (f"{s['n']} snapshots, {s['runs']} runs, {s['families']} families; unknown {s['unknown']:.0%}; "
            f"coverage {ci(s['coverage'])}; informative {ci(s['informative'])}; per family {ci(s['per_family'])}; "
            f"capped {s['capped_share']:.0%}; time {ci(s['time'])}; width p50 {s['width_p50']:.1f} "
            f"p90 {s['width_p90']:.1f}; misses short/long {s['too_short']}/{s['too_long']}")


# ------------------------------------------------------------------ corpus cases
_POINT_RE = re.compile(r"OPTIMIZATION - POINT\s+(\d+)\s*$")
_TELAPSE_RE = re.compile(r"^\s*T{10,}\s+(\S+)\s+TELAPSE\s+([\d.]+)")
_cache: Dict[str, object] = {}


def last_telapse(lines: Sequence[str], upto: Optional[int] = None) -> Optional[float]:
    upto = len(lines) if upto is None else upto
    for i in range(upto - 1, -1, -1):
        m = _TELAPSE_RE.match(lines[i])
        if m:
            return float(m.group(2))
    return None


def opt_runs() -> List[dict]:
    """Completed multi-point optimizations of the corpus (test/OPT and the
    optimizations filed under test/SP), one per SCF fingerprint. Each:
    path, family, lines, heads (line index of every POINT header), n
    (points), t_end (last TELAPSE), deck (text of the .d12 sibling)."""
    if "opt" in _cache:
        return _cache["opt"]          # type: ignore[return-value]
    fam = corpus_families()
    runs, seen = [], set()
    for d in ("OPT", "SP"):
        for f in sorted((TEST_DATA / d).glob("*.out")):
            text = f.read_text(errors="replace")
            if "OPT END" not in text or "OPTIMIZATION - POINT" not in text:
                continue
            fp = scf_fingerprint(text)
            if fp in seen:
                continue
            seen.add(fp)
            lines = text.splitlines()
            heads = [i for i, l in enumerate(lines) if _POINT_RE.search(l)]
            deck = f.with_suffix(".d12")
            runs.append({"path": str(f), "family": fam[str(f)], "lines": lines, "heads": heads,
                         "n": None, "t_end": last_telapse(lines),
                         "deck": deck.read_text(errors="replace") if deck.is_file() else ""})
    _cache["opt"] = runs
    return runs


def sp_runs() -> List[dict]:
    """Single-point SCFs of the corpus (test/SP without optimization points
    and without an OPT run of the same opening SCF). Each: path, family,
    lines, ci (CYC line indexes of the SCF), n (cycles), t_end (TELAPSE
    after SCF ENDED), t_last, guessp, deck."""
    if "sp" in _cache:
        return _cache["sp"]          # type: ignore[return-value]
    fam = corpus_families()
    opt_fps = {scf_fingerprint(f.read_text(errors="replace")) for f in (TEST_DATA / "OPT").glob("*.out")}
    runs = []
    for f in sorted((TEST_DATA / "SP").glob("*.out")):
        text = f.read_text(errors="replace")
        if "OPTIMIZATION - POINT" in text or scf_fingerprint(text) in opt_fps:
            continue
        lines = text.splitlines()
        ci = [i for i, l in enumerate(lines) if _CYC_E.match(l)]
        end = next((i for i, l in enumerate(lines) if "== SCF ENDED" in l), None)
        if end is None or len(ci) < 5:
            continue
        tl = [(i, float(m.group(2))) for i, l in enumerate(lines) for m in [_TELAPSE_RE.match(l)] if m]
        t_end = next((t for i, t in tl if i > end), tl[-1][1])
        deck = f.with_suffix(".d12")
        runs.append({"path": str(f), "family": fam[str(f)], "lines": lines, "ci": ci, "n": len(ci),
                     "t_end": t_end, "t_last": tl[-1][1], "end": end,
                     "guessp": "SCF GUESS FROM A PREVIOUS RUN DENSITY MATRIX" in text,
                     "deck": deck.read_text(errors="replace") if deck.is_file() else ""})
    _cache["sp"] = runs
    return runs
