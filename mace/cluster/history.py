"""Runtime history of finished calculations, and estimates for a new deck.

Sources (each optional): the folder's ``materials.db`` (opened read-only),
the finished ``.out`` files under the folder (at most 2000, cached by file
stamp in ``$XDG_CACHE_HOME/mace/cluster/history-<hash>.json``), and one
batched ``sacct`` per 100 known job ids (Elapsed, NCPUS, MaxRSS, NodeList).

Features of a run: calc type, functional family, spin, atoms per cell,
atomic orbitals (``NUMBER OF AO``), k points in the IBZ, cores (the job
script's ``--ntasks``/``-n``), node group (sacct NodeList -> the node's CPU
type feature; "unknown" when not known), SCF cycles, OPT points, wall time
(CRYSTAL's ``END TELAPSE``) and MaxRSS.

Model: per-SCF-cycle time of the k (<= 7) nearest runs of the same calc type
and functional family, scaled by ``(nAO/nAO_i)^3 * (nk/nk_i) *
(cores_i/cores)^alpha``, times the neighbors' median cycle count. ``alpha``
is fit per calc type when the same system ran at two or more core counts,
else both 0.6 and 1.0 feed the range (no perfect strong scaling assumed).
Same-node-group neighbors are preferred; with fewer than 3 of them the
others are used and the range widened by the measured group ratio (or x2).
Output: the median and the 10-90 % range of the neighbor predictions; fewer
than 3 neighbors gives "Not enough similar runs". Right-sizing is a
suggestion only: walltime = p90 x 1.3 rounded up to the hour (capped at the
queue limit), memory only from plausible MaxRSS, cores only from the same
system at several core counts.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mace.cluster import config as config_mod
from mace.cluster import query

MAX_FILES = 2000
K_NEIGHBORS = 7
MIN_NEIGHBORS = 3
ALPHA_RANGE = (0.6, 1.0)
GROUP_WIDEN_DEFAULT = 2.0
INDEX_VERSION = 2

_ATOMS = re.compile(r"N\. OF ATOMS PER CELL\s+(\d+)")
_AO = re.compile(r"NUMBER OF AO\s+(\d+)")
_NK = re.compile(r"NUMBER OF K POINTS IN THE IBZ\s+(\d+)")
_END = re.compile(r"T{10,}\s+END\s+TELAPSE\s+([\d.]+)")
_CYCLES = re.compile(r"TOTAL ENERGY\((?:DFT|HF)\)\(AU\)\(\s*(\d+)\)")
_POINT = re.compile(r"OPTIMIZATION - POINT\s+(\d+)")
_NTASKS = re.compile(r"^#SBATCH\s+(?:--ntasks[= ]|-n\s*)(\d+)", re.M)

FAMILIES = {
    "hybrid": ("B3LYP", "PBE0", "HSE06", "HSESOL", "B97", "M06", "SCAN0", "B1WC", "LC-", "CAM", "WB97", "PW1PW",
               "HISS", "RSHXLDA", "B3PW", "MN15", "HSE"),
    "3c": ("HF3C", "PBEH3C", "HSE3C", "B973C", "PBESOL03C", "HSESOL3C"),
    "gga": ("PBE", "PBESOL", "BLYP", "PW91", "WCGGA", "SOGGA", "B97D", "REVPBE", "SCAN"),
    "hf": ("RHF", "UHF", "HF"),
    "lda": ("LDA", "SVWN", "VBH"),
}


def functional_family(deck_text: str) -> str:
    recs = [l.strip().upper() for l in (deck_text or "").splitlines()[1:]]
    if "DFT" not in recs:
        return "hf"
    i = recs.index("DFT")
    block = recs[i + 1:i + 30]
    for r in block:
        if r == "ENDDFT" or r == "END":
            break
        for fam in ("3c", "hybrid", "gga", "lda"):
            if any(r == k or r.startswith(k) for k in FAMILIES[fam]):
                return fam
        if r in ("EXCHANGE", "CORRELAT"):
            return "gga"
    return "dft"


def deck_calc_type(deck_text: str, out_text: str = "") -> str:
    from mace.utils import calc_detection as cd
    if out_text:
        if cd.is_frequency_output(out_text):
            return "FREQ"
        if cd.is_optimization_output(out_text):
            return "OPT"
        return "SP"
    return cd.d12_calc_type(cd.deck_records(deck_text))


def cores_of_script(sh: Path) -> Optional[int]:
    try:
        m = _NTASKS.search(sh.read_text(errors="replace"))
        return int(m.group(1)) if m else None
    except OSError:
        return None


def features_of_out(out: Path) -> Optional[Dict[str, Any]]:
    """Features of one finished .out (None when it did not finish)."""
    try:
        text = out.read_text(errors="replace")
    except OSError:
        return None
    m_end = _END.search(text)
    if not m_end:
        return None
    deck = out.with_suffix(".d12")
    deck_text = deck.read_text(errors="replace") if deck.is_file() else ""
    m_at, m_ao, m_nk = _ATOMS.search(text), _AO.search(text), _NK.search(text)
    cycles = sum(int(x) for x in _CYCLES.findall(text))
    sh = out.with_suffix(".sh")
    rec = {
        "file": str(out), "stem": out.stem, "calc_type": deck_calc_type(deck_text, text),
        "family": functional_family(deck_text) if deck_text else "unknown",
        "spin": bool(re.search(r"^\s*SPIN\s*$", deck_text, re.M)),
        "atoms": int(m_at.group(1)) if m_at else None, "nao": int(m_ao.group(1)) if m_ao else None,
        "nk": int(m_nk.group(1)) if m_nk else None,
        "cores": cores_of_script(sh) if sh.is_file() else None,
        "cycles": cycles or None, "points": len(_POINT.findall(text)) or None,
        "wall_s": float(m_end.group(1)), "group": "unknown", "max_rss_mb": None, "job_id": None,
    }
    return rec


def _index_cache(folder: Path) -> Path:
    h = hashlib.sha1(str(folder.resolve()).encode()).hexdigest()[:16]
    return query.cache_root() / f"history-{h}.json"


def _db_records(db_path: Path) -> List[Dict[str, Any]]:
    """Finished calculations of a materials.db (read-only; a writer may hold it)."""
    from mace.cluster.progress import ro_uri
    out = []
    for uri in (ro_uri(db_path), ro_uri(db_path, "immutable=1")):
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=30)
            conn.execute("PRAGMA busy_timeout=30000")
            try:
                rows = conn.execute("SELECT calc_id, calc_type, slurm_job_id, output_file, work_dir, status "
                                    "FROM calculations WHERE status = 'completed'").fetchall()
            finally:
                conn.close()
            for r in rows:
                out.append({"calc_id": r[0], "calc_type": r[1], "job_id": r[2], "output_file": r[3],
                            "work_dir": r[4]})
            return out
        except sqlite3.Error:
            continue
    return out


def index(folder, db_path: Optional[Path] = None, use_sacct: bool = True,
          max_files: int = MAX_FILES) -> List[Dict[str, Any]]:
    """History records of a folder (see the module doc)."""
    folder = Path(folder)
    cache_f = _index_cache(folder)
    try:
        cache = json.loads(cache_f.read_text())
        if cache.get("version") != INDEX_VERSION:
            cache = {}
    except (OSError, ValueError):
        cache = {}
    stamps = cache.get("files", {})
    outs: List[Path] = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if f.endswith(".out"):
                outs.append(Path(root) / f)
                if len(outs) >= max_files:
                    break
        if len(outs) >= max_files:
            break
    db_path = db_path or (folder / "materials.db")
    jobs_by_out = {}
    if db_path and Path(db_path).is_file():
        for r in _db_records(Path(db_path)):
            if r.get("output_file"):
                p = Path(r["output_file"])
                if not p.is_absolute() and r.get("work_dir"):
                    p = Path(r["work_dir"]) / p
                jobs_by_out[str(p.resolve())] = r.get("job_id")
                if p.is_file() and p not in outs and len(outs) < max_files:
                    outs.append(p)
    records, new = [], {}
    for out in outs:
        try:
            st = out.stat()
        except OSError:
            continue
        key = str(out.resolve())
        stamp = f"{st.st_size}:{st.st_mtime_ns}"
        hit = stamps.get(key)
        rec = hit["rec"] if hit and hit.get("stamp") == stamp else features_of_out(out)
        new[key] = {"stamp": stamp, "rec": rec}
        if rec:
            rec = dict(rec)
            rec["job_id"] = jobs_by_out.get(key) or rec.get("job_id")
            records.append(rec)
    try:
        cache_f.parent.mkdir(parents=True, exist_ok=True)
        query._atomic_write(cache_f, {"version": INDEX_VERSION, "files": new})
    except OSError:
        pass
    if use_sacct:
        _add_sacct(records)
    return records


def _add_sacct(records: List[Dict[str, Any]]) -> None:
    """Elapsed/NCPUS/MaxRSS/NodeList for records with a job id (100 per call)."""
    from mace.cluster import reader
    from mace.cluster import nodes as nodes_mod
    ids = [r["job_id"] for r in records if r.get("job_id") and reader.valid_job_id(str(r["job_id"]))]
    if not ids or not query.available("sacct"):
        return
    groups = {}
    nrec = query.read_cache("nodes").get("value") or []
    regex = config_mod.load().get("cluster", "group_feature_regex")
    for n in nrec:
        groups[n["name"]] = nodes_mod.group_of(n, regex)
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        argv = ["sacct", "-u", reader.user(), "-j", ",".join(chunk), "-X", "-P", "-n",
                "-o", "JobID,Elapsed,NCPUS,MaxRSS,NodeList,State"]
        res = query.run("history-sacct-" + hashlib.sha1(",".join(chunk).encode()).hexdigest()[:12], argv,
                        ttl=86400, refresh_floor=3600)
        for line in (res.value or "").splitlines():
            f = line.split("|")
            if len(f) < 6:
                continue
            for r in records:
                if str(r.get("job_id")) == f[0]:
                    if f[2].isdigit():
                        r["cores"] = r.get("cores") or int(f[2])
                    first = f[4].split(",")[0]
                    first = re.sub(r"\[(\d+).*", r"\1", first)
                    r["group"] = groups.get(first, "unknown")


# ------------------------------------------------------------------ deck features
def deck_features(deck: Path, records: Sequence[Dict[str, Any]] = ()) -> Dict[str, Any]:
    deck = Path(deck)
    text = deck.read_text(errors="replace")
    feats = {"stem": deck.stem, "calc_type": deck_calc_type(text), "family": functional_family(text),
             "spin": bool(re.search(r"^\s*SPIN\s*$", text, re.M)), "atoms": None, "nao": None, "nk": None,
             "cores": cores_of_script(deck.with_suffix(".sh")) if deck.with_suffix(".sh").is_file() else None,
             "parent": None}
    # the same system ran before: the deck's own .out (a rerun) or the parent
    # whose stem the follow-up's name starts with (MACE names follow-ups that way)
    cands = [deck.with_suffix(".out")] + sorted(deck.parent.glob("*.out"), key=lambda p: -len(p.stem))
    for c in cands:
        if c.is_file() and (c.stem == deck.stem or deck.stem.startswith(c.stem + "_")):
            try:
                t = c.read_text(errors="replace")
            except OSError:
                continue
            m_at, m_ao, m_nk = _ATOMS.search(t), _AO.search(t), _NK.search(t)
            if m_ao:
                feats.update(atoms=int(m_at.group(1)) if m_at else None, nao=int(m_ao.group(1)),
                             nk=int(m_nk.group(1)) if m_nk else None, parent=str(c))
                break
    return feats


# ------------------------------------------------------------------ model
def _pct(xs: Sequence[float], q: float) -> float:
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _usable(r: Dict[str, Any]) -> bool:
    return bool(r.get("wall_s") and r.get("cycles") and r.get("nao") and r.get("cores"))


def fit_alpha(records: Sequence[Dict[str, Any]], calc_type: str) -> Optional[float]:
    """Core-scaling exponent from runs of the same system at >= 2 core counts."""
    by: Dict[Tuple, List[Tuple[int, float]]] = {}
    for r in records:
        if r.get("calc_type") != calc_type or not _usable(r):
            continue
        by.setdefault((r["nao"], r.get("nk"), r.get("atoms")), []).append((r["cores"], r["wall_s"] / r["cycles"]))
    slopes = []
    for runs in by.values():
        cs = {}
        for c, t in runs:
            cs.setdefault(c, []).append(t)
        if len(cs) < 2:
            continue
        xs = [math.log(c) for c in cs]
        ys = [math.log(statistics.median(v)) for v in cs.values()]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx > 0:
            slopes.append(-sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx)
    if not slopes:
        return None
    return min(1.2, max(0.2, statistics.median(slopes)))


def estimate_from(feats: Dict[str, Any], records: Sequence[Dict[str, Any]], cores: Optional[int] = None,
                  group: Optional[str] = None, exclude: Optional[str] = None) -> Dict[str, Any]:
    cores = cores or feats.get("cores")
    out: Dict[str, Any] = {"n": 0, "neighbors": [], "wall": None, "memory_per_core_mb": None,
                           "suggestion": None, "reasons": [], "features": feats, "cores": cores}
    ct, fam = feats.get("calc_type"), feats.get("family")
    pool = [r for r in records if r.get("calc_type") == ct and r.get("family") == fam and _usable(r)
            and r.get("file") != exclude]
    if not cores:
        out["reasons"].append("The core count is unknown (no job script); give --cores")
        out["message"] = "Not enough information: the core count is unknown"
        return out
    if not feats.get("nao"):
        out["reasons"].append("No finished run of this system: the number of atomic orbitals is unknown, "
                              "so the closest runs by type are used and the range widens")

    def dist(r):
        d = 0.0
        for k in ("nao", "nk", "atoms"):
            a, b = feats.get(k), r.get(k)
            if a and b:
                d += abs(math.log(a / b))
            else:
                d += 1.0
        return d
    pool.sort(key=dist)
    same = [r for r in pool if group and r.get("group") == group]
    widen = 1.0
    if group and len(same) >= MIN_NEIGHBORS:
        neigh = same[:K_NEIGHBORS]
    else:
        neigh = pool[:K_NEIGHBORS]
        groups = {r.get("group") for r in neigh}
        if len(groups) > 1 or "unknown" in groups or group is None or not same:
            ratio = _group_ratio(pool)
            widen = ratio if ratio else GROUP_WIDEN_DEFAULT
            out["reasons"].append(f"Runs on other or unknown node types are used; the range is widened x{widen:.1f}")
    out["n"] = len(neigh)
    if len(neigh) < MIN_NEIGHBORS:
        out["message"] = f"Not enough similar runs ({len(neigh)} found; {MIN_NEIGHBORS} needed)"
        return out
    alpha = fit_alpha(records, ct)
    alphas = [alpha] if alpha is not None else list(ALPHA_RANGE)
    if alpha is None:
        out["reasons"].append(f"Core scaling unknown (no system ran at two core counts): alpha "
                              f"{ALPHA_RANGE[0]}..{ALPHA_RANGE[1]} both used")
    else:
        out["reasons"].append(f"Core scaling fitted from history: alpha {alpha:.2f}")
    cyc = statistics.median([r["cycles"] for r in neigh])
    preds = []
    for r in neigh:
        per = r["wall_s"] / r["cycles"]
        s = 1.0
        if feats.get("nao"):
            s *= (feats["nao"] / r["nao"]) ** 3
        if feats.get("nk") and r.get("nk"):
            s *= feats["nk"] / r["nk"]
        for a in alphas:
            preds.append(per * s * (r["cores"] / cores) ** a * cyc)
    med = statistics.median(preds)
    p10, p90 = _pct(preds, 0.1), _pct(preds, 0.9)
    if not feats.get("nao"):
        p10, p90 = p10 / 2, p90 * 2
    p10, p90 = p10 / widen, p90 * widen
    out["wall"] = {"median": med, "p10": p10, "p90": p90}
    out["neighbors"] = [{"stem": r["stem"], "wall_s": r["wall_s"], "cores": r["cores"], "nao": r["nao"],
                         "nk": r.get("nk"), "group": r.get("group")} for r in neigh]
    wt = math.ceil(p90 * 1.3 / 3600.0) * 3600
    out["suggestion"] = {"walltime_s": wt, "walltime": _hms(wt), "mem_per_cpu_mb": None, "cores": None}
    rss = [r["max_rss_mb"] * (feats["nao"] / r["nao"]) ** 2 for r in neigh if r.get("max_rss_mb") and feats.get("nao")]
    if rss:
        out["memory_per_core_mb"] = max(rss) * 1.25 / cores
        out["suggestion"]["mem_per_cpu_mb"] = math.ceil(out["memory_per_core_mb"])
    return out


def _group_ratio(pool: Sequence[Dict[str, Any]]) -> Optional[float]:
    by: Dict[str, List[float]] = {}
    for r in pool:
        g = r.get("group")
        if g and g != "unknown":
            by.setdefault(g, []).append(r["wall_s"] / r["cycles"] / ((r["nao"] or 1) ** 3) * r["cores"])
    if len(by) < 2:
        return None
    meds = sorted(statistics.median(v) for v in by.values())
    return max(1.0, meds[-1] / meds[0])


def _hms(s: float) -> str:
    s = int(s)
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m = rem // 60
    return f"{d}-{h:02d}:{m:02d}:00" if d else f"{h}:{m:02d}:00"


def coverage(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Leave-one-out coverage per calc type: how often the p10-p90 range of
    the others holds the true wall time."""
    out: Dict[str, Dict[str, Any]] = {}
    for r in records:
        if not _usable(r):
            continue
        feats = {k: r.get(k) for k in ("calc_type", "family", "spin", "atoms", "nao", "nk", "cores")}
        est = estimate_from(feats, records, cores=r["cores"], exclude=r["file"])
        c = out.setdefault(r["calc_type"], {"n": 0, "hit": 0})
        if est.get("wall"):
            c["n"] += 1
            c["hit"] += est["wall"]["p10"] <= r["wall_s"] <= est["wall"]["p90"]
    for v in out.values():
        v["coverage"] = v["hit"] / v["n"] if v["n"] else None
    return out


def estimate(deck_path, cores: Optional[int] = None, history: Optional[Path] = None,
             db_path: Optional[Path] = None, use_sacct: bool = True, group: Optional[str] = None,
             queue_limit_s: Optional[int] = None) -> Dict[str, Any]:
    deck_path = Path(deck_path)
    folder = Path(history) if history else deck_path.parent
    records = index(folder, db_path=db_path, use_sacct=use_sacct)
    feats = deck_features(deck_path, records)
    est = estimate_from(feats, records, cores=cores, group=group)
    cov = coverage(records).get(feats.get("calc_type") or "", {})
    est["history"] = {"records": len(records), "folder": str(folder), "loo": cov}
    if est.get("wall") and cov.get("n", 0) >= 5 and (cov.get("coverage") or 0) < 0.5:
        est["hidden"] = True
        est["message"] = f"Estimates hidden for {feats['calc_type']}: history too scattered"
        est["wall"] = None
        est["suggestion"] = None
    if est.get("suggestion") and queue_limit_s:
        if est["suggestion"]["walltime_s"] > queue_limit_s:
            est["suggestion"]["walltime_s"] = queue_limit_s
            est["suggestion"]["walltime"] = _hms(queue_limit_s)
    return est
