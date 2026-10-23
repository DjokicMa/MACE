"""Reference nets for identifying topologies (RCSR three-letter symbols).

The reference set, tried in this order (the first entry of a name wins):

- nets you import (``import_file``): RCSR's 3dall.txt (http://rcsr.net/data/3dall.txt,
  which already carries every vertex's coordination sequence) or a Systre .cgd file
  such as RCSRnets.cgd (sequences computed here), stored in the MACE config folder;
- RCSR's list of 3-periodic nets, bundled with MACE (rcsr_data/3dall.txt.xz, about
  3600 nets; source, date, checksum and terms in rcsr_data/NOTICE.txt);
- a built-in set of common nets (BUILTIN_DEFS below) whose coordination sequences
  MACE computes itself from their textbook embeddings (the diamond, CsCl, NbO, PtS,
  SrSi2, sodalite, quartz, ... positions), which also covers the 2-periodic hcb, sql,
  kgm and hxl.

The Reticular Chemistry Structure Resource: M. O'Keeffe, M. A. Peskov, S. J. Ramsden
and O. M. Yaghi, Acc. Chem. Res. 41, 1782 (2008), doi:10.1021/ar800124u (CITATION).

Matching compares the set of distinct coordination sequences (10 shells)
of a net's vertices.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# name: (space group, cell parameters, [(label, (x, y, z))], periodic dims, edge type pairs or None)
# edges join the closest vertices (of the listed label pairs, when given)
BUILTIN_DEFS: Dict[str, Tuple] = {
    "pcu": (221, (1, 1, 1, 90, 90, 90), [("A", (0, 0, 0))], 3, None),
    "bcu": (229, (1, 1, 1, 90, 90, 90), [("A", (0, 0, 0))], 3, None),
    "fcu": (225, (1, 1, 1, 90, 90, 90), [("A", (0, 0, 0))], 3, None),
    "dia": (227, (1, 1, 1, 90, 90, 90), [("A", (0, 0, 0))], 3, None),          # origin choice 1
    "srs": (214, (1, 1, 1, 90, 90, 90), [("A", (0.125, 0.125, 0.125))], 3, None),
    "nbo": (229, (1, 1, 1, 90, 90, 90), [("A", (0, 0.5, 0.5))], 3, None),
    "sod": (229, (1, 1, 1, 90, 90, 90), [("A", (0.25, 0, 0.5))], 3, None),
    "pts": (131, (1, 1, 1.41421, 90, 90, 90), [("A", (0, 0.5, 0)), ("B", (0, 0, 0.25))], 3, [("A", "B")]),
    "lon": (194, (1, 1, 1.63299, 90, 90, 120), [("A", (1 / 3, 2 / 3, 0.0625))], 3, None),
    "qtz": (180, (4.999, 4.999, 5.457, 90, 90, 120), [("A", (0.5, 0, 0))], 3, None),
    "ftw": (221, (1, 1, 1, 90, 90, 90), [("A", (0, 0, 0)), ("B", (0, 0.5, 0.5))], 3, [("A", "B")]),
    "she": (229, (1, 1, 1, 90, 90, 90), [("A", (0.25, 0, 0.5)), ("B", (0.25, 0.25, 0.25))], 3, [("A", "B")]),
    "scu": (123, (1.633, 1.633, 1.1547, 90, 90, 90), [("A", (0.5, 0, 0.5)), ("B", (0, 0, 0))], 3, [("A", "B")]),
    "ths": (141, (1, 1, 3.4641, 90, 90, 90), [("A", (0, 0, 0.41667))], 3, None),
    "tbo": (225, (1, 1, 1, 90, 90, 90), [("A", (1 / 6, 1 / 6, 1 / 6)), ("B", (0, 0.25, 0.25))], 3, [("A", "B")]),
    "pto": (223, (1, 1, 1, 90, 90, 90), [("A", (0.25, 0.25, 0.25)), ("B", (0.25, 0, 0.5))], 3, [("A", "B")]),
    "acs": (194, (1.4142, 1.4142, 1.1547, 90, 90, 120), [("A", (1 / 3, 2 / 3, 0.25))], 3, None),
    "hcb": (191, (1, 1, 5, 90, 90, 120), [("A", (1 / 3, 2 / 3, 0))], 2, None),
    "sql": (123, (1, 1, 5, 90, 90, 90), [("A", (0, 0, 0))], 2, None),
    "kgm": (191, (1, 1, 5, 90, 90, 120), [("A", (0.5, 0, 0))], 2, None),
    "hxl": (191, (1, 1, 5, 90, 90, 120), [("A", (0, 0, 0))], 2, None),
}


def config_dir() -> Path:
    base = os.environ.get("MACE_CONFIG_DIR") or os.path.join(os.environ.get("XDG_CONFIG_HOME") or
                                                             os.path.expanduser("~/.config"), "mace")
    return Path(base)


def imported_path() -> Path:
    return config_dir() / "rcsr_nets.json"


CITATION = ("M. O'Keeffe, M. A. Peskov, S. J. Ramsden and O. M. Yaghi, The Reticular Chemistry Structure "
            "Resource (RCSR) Database of, and Symbols for, Crystal Nets, Acc. Chem. Res. 41, 1782-1789 (2008), "
            "doi:10.1021/ar800124u")
CITATION_SHORT = "RCSR (rcsr.net): O'Keeffe, Peskov, Ramsden, Yaghi, Acc. Chem. Res. 41, 1782 (2008), doi:10.1021/ar800124u"
DOWNLOAD_URL = "http://rcsr.net/data/3dall.txt"
BUNDLED_DIR = Path(__file__).resolve().parent / "rcsr_data"
BUNDLED_FILE = BUNDLED_DIR / "3dall.txt.xz"
BUNDLED_DATE = "2026-10-22"              # downloaded; server Last-Modified 2026-01-28 (rcsr_data/NOTICE.txt)
BUNDLED_SOURCE = f"RCSR 3dall.txt (bundled, {BUNDLED_DATE})"


def build_reference(name: str, defn=None):
    """The PeriodicNet of a built-in definition."""
    from ase.spacegroup import crystal
    from .topology import PeriodicNet
    from .structure import Structure
    sg, cellpar, sites, dims, pairs = defn or BUILTIN_DEFS[name]
    labels = [lab for lab, _ in sites]
    zs = list(range(1, len(sites) + 1))
    at = crystal([int(z) for z in zs], basis=[p for _, p in sites], spacegroup=sg, cellpar=list(cellpar),
                 symprec=1e-4)
    pbc = (True, True, dims == 3) if dims == 2 else (True, True, True)
    st = Structure(at.cell.array, at.get_scaled_positions(), at.numbers, pbc)
    lab = [labels[z - 1] for z in at.numbers]
    cut = 1.5 * float(max(cellpar[:3]))
    ii, jj, sh, d = st.neighbor_pairs(cut)
    if pairs:
        allowed = {tuple(p) for p in pairs} | {tuple(reversed(p)) for p in pairs}
        m = np.array([(lab[a], lab[b]) in allowed for a, b in zip(ii, jj)], bool)
        ii, jj, sh, d = ii[m], jj[m], sh[m], d[m]
    dmin = d.min()
    m = d < dmin * 1.02
    edges = [(int(a), int(b), tuple(int(x) for x in s)) for a, b, s in zip(ii[m], jj[m], sh[m])]
    return PeriodicNet(len(lab), edges, st.frac, st.cell, lab, pbc)


_BUILTIN_CACHE: Optional[List[Dict]] = None


def builtin() -> List[Dict]:
    """[{name, dims, cs: [[...10], ...], cn: [...], td10, source}] computed once."""
    global _BUILTIN_CACHE
    if _BUILTIN_CACHE is None:
        from .topology import vertex_classes, td10
        out = []
        for name in BUILTIN_DEFS:
            net = build_reference(name)
            cl = vertex_classes(net)
            out.append({"name": name, "dims": BUILTIN_DEFS[name][3], "cs": [g["cs"] for g in cl],
                        "cn": [g["cn"] for g in cl], "td10": td10(cl), "source": "built-in"})
        _BUILTIN_CACHE = out
    return _BUILTIN_CACHE


_BUNDLED_CACHE: Optional[List[Dict]] = None


def bundled_text() -> str:
    """The bundled RCSR 3dall.txt, decompressed ("" if it or Python's lzma is missing)."""
    try:
        import lzma
        return lzma.decompress(BUNDLED_FILE.read_bytes()).decode("utf-8", errors="replace")
    except Exception:                      # no lzma, or the file missing or damaged: the built-in nets only
        return ""


def bundled() -> List[Dict]:
    """RCSR's 3-periodic nets shipped with MACE, parsed once per process."""
    global _BUNDLED_CACHE
    if _BUNDLED_CACHE is None:
        nets = parse_3dall(bundled_text())
        for n in nets:
            n["source"] = BUNDLED_SOURCE
        _BUNDLED_CACHE = nets
    return _BUNDLED_CACHE


def imported() -> List[Dict]:
    p = imported_path()
    if not p.is_file():
        return []
    try:
        return json.loads(p.read_text()).get("nets", [])
    except (OSError, ValueError):
        return []


def references(reference=None) -> List[Dict]:
    """Imported nets, then the bundled RCSR list, then the built-in set; a name
    already given by an earlier source is skipped, so your import wins."""
    if reference is not None:
        return list(reference)
    out, seen = [], set()
    for ref in imported() + bundled() + builtin():
        key = (ref.get("name"), int(ref.get("dims", 3)))
        if key not in seen:
            seen.add(key)
            out.append(ref)
    return out


def match(vertex_classes: Sequence[Dict], dims: int = 3, reference=None, limit: int = 3) -> List[Dict]:
    """References whose set of coordination sequences equals the net's."""
    key = {tuple(c["cs"][:10]) for c in vertex_classes}
    if not key:
        return []
    out, seen = [], set()
    for ref in references(reference):
        if int(ref.get("dims", 3)) != int(dims):
            continue
        rk = {tuple(c[:10]) for c in ref["cs"]}
        if rk == key and ref["name"] not in seen:
            seen.add(ref["name"])
            out.append({"name": ref["name"], "source": ref.get("source", "?"), "td10": ref.get("td10"),
                        "by": "coordination sequences (10 shells)"})
            if len(out) >= limit:
                break
    return out


# ---------------------------------------------------------------------------
# importing a downloaded RCSR file
# ---------------------------------------------------------------------------
def parse_3dall(text: str) -> List[Dict]:
    """RCSR's 3dall.txt: name, vertex count, and every vertex's coordination
    sequence (10 shells) with td10."""
    nets = []
    blocks = re.split(r"(?m)^start\s*$", text)
    for b in blocks:
        lines = [l.rstrip() for l in b.strip("\n").split("\n")]
        if len(lines) < 10:
            continue
        try:
            i = 0
            i += 1                                  # serial number
            name = lines[i].strip()
            i += 2                                  # name, embedding type
            for _ in range(4):                      # other symbols, names, other names, keywords
                n = int(lines[i].split()[0])
                i += 1 + n
            n = int(lines[i].split()[0])            # references
            i += 1 + n
            i += 2                                  # space group, cell
            nv = int(lines[i].split()[0])
            i += 1
            cn = []
            for _ in range(nv):
                cn.append(int(lines[i].split()[1]))
                i += 6
            ne = int(lines[i].split()[0])
            i += 1 + 5 * ne
            # then 5 lines of tiling data, then one coordination-sequence line per vertex
            cs = []
            j = i
            while j < len(lines) and len(cs) < nv:
                parts = lines[j].split()
                if len(parts) == 11 and all(p.lstrip("-").isdigit() for p in parts):
                    cs.append([int(p) for p in parts])
                j += 1
            if len(cs) != nv:
                continue
            tot = sum(sum(c[:10]) + 1 for c in cs) / nv
            nets.append({"name": name, "dims": 3, "cn": cn, "cs": [c[:10] for c in cs], "td10": tot,
                         "source": "RCSR 3dall.txt"})
        except (ValueError, IndexError):
            continue
    return nets


def parse_cgd(text: str, k: int = 10) -> List[Dict]:
    """A Systre .cgd file (CRYSTAL blocks with NAME, GROUP, CELL, NODE, EDGE):
    coordination sequences computed here (slow for thousands of nets)."""
    from ase.spacegroup import Spacegroup
    from .topology import PeriodicNet, vertex_classes, td10
    from .structure import Structure
    out = []
    for blk in re.findall(r"(?ms)^CRYSTAL\s*$(.*?)^END\s*$", text):
        try:
            d = {"nodes": [], "edges": []}
            for line in blk.splitlines():
                t = line.split("#")[0].split()
                if not t:
                    continue
                key = t[0].upper()
                if key == "NAME":
                    d["name"] = t[1]
                elif key == "GROUP":
                    d["group"] = t[1]
                elif key == "CELL":
                    d["cell"] = [float(x) for x in t[1:]]
                elif key == "NODE":
                    d["nodes"].append((int(t[2]), [float(x) for x in t[3:]]))
                elif key == "EDGE":
                    d["edges"].append([float(x) for x in t[1:]])
            dims = 2 if len(d.get("cell", [])) == 3 else 3
            if dims != 3 or "group" not in d:
                continue
            sg = Spacegroup(d["group"].replace("(", "").replace(")", ""))
            frac, cn = [], []
            for c, p in d["nodes"]:
                sites, _ = sg.equivalent_sites([p], symprec=1e-3)
                for s_ in sites:
                    frac.append(s_)
                    cn.append(c)
            frac = np.array(frac) % 1.0
            from ase.geometry import cellpar_to_cell
            cell = cellpar_to_cell(d["cell"])
            st = Structure(cell, frac, np.ones(len(frac), int))
            # edges: symmetry images of each EDGE's two ends, matched to nodes
            edges = set()
            for e in d["edges"]:
                a, b = np.array(e[:3]), np.array(e[3:6])
                for R, t in zip(sg.get_rotations(), sg.get_translations()):
                    pa, pb = R @ a + t, R @ b + t
                    ia = int(np.argmin(np.linalg.norm(((frac - pa) + 0.5) % 1 - 0.5, axis=1)))
                    ib = int(np.argmin(np.linalg.norm(((frac - pb) + 0.5) % 1 - 0.5, axis=1)))
                    sa = np.rint(pa - frac[ia]).astype(int)
                    sb = np.rint(pb - frac[ib]).astype(int)
                    edges.add((ia, ib, tuple(int(x) for x in (sb - sa))))
            net = PeriodicNet(len(frac), list(edges), frac, cell)
            cl = vertex_classes(net, k)
            out.append({"name": d.get("name", "?"), "dims": 3, "cn": [g["cn"] for g in cl], "cs": [g["cs"] for g in cl],
                        "td10": td10(cl), "source": "Systre cgd"})
        except Exception:
            continue
    return out


def import_file(path, dest: Optional[Path] = None) -> Dict:
    """Read a downloaded RCSR 3dall.txt or .cgd file into MACE's reference set."""
    p = Path(path)
    text = p.read_text(errors="replace")
    nets = parse_cgd(text) if ("CRYSTAL" in text[:5000] and "EDGE" in text) else parse_3dall(text)
    if not nets:
        raise ValueError(f"no nets found in {p.name} (expected RCSR 3dall.txt or a Systre .cgd file)")
    dest = dest or imported_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"source": p.name, "count": len(nets), "nets": nets}))
    return {"imported": len(nets), "path": str(dest), "source": p.name}


def status() -> Dict:
    imp = imported()
    return {"builtin": [n["name"] for n in builtin()], "bundled": len(bundled()), "bundled_source": BUNDLED_SOURCE,
            "imported": len(imp), "imported_path": str(imported_path()), "download": DOWNLOAD_URL,
            "citation": CITATION}
