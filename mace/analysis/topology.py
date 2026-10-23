"""Underlying nets of crystal structures: simplification, coordination
sequences, point symbols, interpenetration, and identification against
RCSR nets (rcsr.py).

A periodic net is stored as its quotient graph: vertices in the cell and
edges (u, v, t), meaning u in cell 0 is bonded to v in cell t.

Simplification (from the bonded structure):

- ``atoms``: every atom is a vertex; terminal (1-connected) vertices are
  removed and 2-connected ones replaced by an edge joining their two
  neighbors, repeatedly. For zeolites and other T-X-T frameworks this gives
  the T-atom net.
- ``sbu`` (metal-organic frameworks, "single node" representation): metal
  atoms with the non-carbon atoms that bond only to metals (oxo, hydroxo,
  halide bridges) form the inorganic building units; the remaining
  connected organic pieces are linkers. A unit becomes a vertex at its
  centroid; a linker bonded to three or more units becomes a vertex too, a
  linker bonded to two becomes an edge. An inorganic unit that is itself
  infinite (a rod) falls back to one vertex per metal atom.
- ``auto``: ``sbu`` when the structure has metals and carbon, else ``atoms``.
Molecules not bonded to the framework (solvent, guests) are dropped first.

Invariants: the coordination sequence of each vertex to 10 shells and the
topological density td10 (RCSR's convention: 1 + the first 10 shells, the
average over vertices), and the point symbol (shortest rings at each
angle). Equal coordination sequences are necessary for two nets to be the
same, and in practice nearly sufficient, but not a proof (a few distinct
nets share them); matching is reported as by coordination sequence.

Interpenetration: each connected component of the quotient graph repeats
along the lattice spanned by its cycle vectors; the index of that lattice
in the full one is the number of disjoint copies it splits into. The
degree of interpenetration is the sum over the 3-periodic components; a
component with rank below the structure's periodicity is a layer or a
chain (stacked, or interwoven, which is not classified further).
"""
from __future__ import annotations

import itertools
import math
from collections import defaultdict, deque
from functools import reduce
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .structure import Structure

# elements treated as metals for building units (everything else is not)
NONMETALS = {"H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "Ge", "As", "Se", "Br", "Kr",
             "Sb", "Te", "I", "Xe", "At", "Rn"}


# ---------------------------------------------------------------------------
# the periodic graph
# ---------------------------------------------------------------------------
class PeriodicNet:
    def __init__(self, n: int, edges: Sequence[Tuple[int, int, Sequence[int]]], frac=None, cell=None,
                 labels=None, pbc=(True, True, True)):
        self.n = int(n)
        self.frac = np.zeros((n, 3)) if frac is None else np.asarray(frac, float).reshape(n, 3)
        self.cell = np.eye(3) if cell is None else np.asarray(cell, float)
        self.labels = list(labels) if labels is not None else [str(i) for i in range(n)]
        self.pbc = tuple(bool(x) for x in pbc)
        es = set()
        for u, v, t in edges:
            u, v, t = int(u), int(v), tuple(int(x) for x in t)
            if u == v and not any(t):
                continue
            if (u, v, t) > (v, u, tuple(-x for x in t)):
                u, v, t = v, u, tuple(-x for x in t)
            es.add((u, v, t))
        self.edges = sorted(es)
        self.adj: List[List[Tuple[int, Tuple[int, int, int]]]] = [[] for _ in range(n)]
        for u, v, t in self.edges:
            self.adj[u].append((v, t))
            self.adj[v].append((u, tuple(-x for x in t)))

    def degree(self) -> List[int]:
        return [len(a) for a in self.adj]

    # -- invariants --------------------------------------------------------------
    def coordination_sequence(self, v: int, k: int = 10) -> List[int]:
        start = (v, (0, 0, 0))
        seen = {start}
        frontier = [start]
        out = []
        for _ in range(k):
            nxt = []
            for (u, t) in frontier:
                for w, s in self.adj[u]:
                    node = (w, (t[0] + s[0], t[1] + s[1], t[2] + s[2]))
                    if node not in seen:
                        seen.add(node)
                        nxt.append(node)
            out.append(len(nxt))
            frontier = nxt
            if not nxt:
                out += [0] * (k - len(out))
                break
        return out

    def _ring(self, v: int, a, b, maxlen: int) -> Optional[int]:
        """Shortest cycle through the angle a-v-b (a, b lifted neighbors)."""
        if a == b:
            return None
        banned = (v, (0, 0, 0))
        target = b
        seen = {a, banned}
        q = deque([(a, 0)])
        while q:
            node, dist = q.popleft()
            if dist + 2 >= maxlen:
                continue
            u, t = node
            for w, s in self.adj[u]:
                nn = (w, (t[0] + s[0], t[1] + s[1], t[2] + s[2]))
                if nn == target:
                    return dist + 3
                if nn not in seen:
                    seen.add(nn)
                    q.append((nn, dist + 1))
        return None

    def point_symbol(self, v: int, maxring: int = 12) -> str:
        nbrs = [(w, s) for w, s in self.adj[v]]
        rings = []
        for a, b in itertools.combinations(range(len(nbrs)), 2):
            r = self._ring(v, nbrs[a], nbrs[b], maxring)
            rings.append(r if r is not None else 0)
        cnt = defaultdict(int)
        for r in rings:
            cnt[r] += 1
        parts = []
        for r in sorted(cnt):
            name = str(r) if r else "inf"
            parts.append(f"{name}^{cnt[r]}" if cnt[r] > 1 else name)
        return ".".join(parts)

    # -- components and interpenetration -------------------------------------------
    def components(self) -> List[Dict]:
        comp = [-1] * self.n
        off: List[Optional[np.ndarray]] = [None] * self.n
        out = []
        for s in range(self.n):
            if comp[s] >= 0:
                continue
            cid = len(out)
            comp[s] = cid
            off[s] = np.zeros(3, int)
            members = [s]
            cyc = []
            q = deque([s])
            while q:
                u = q.popleft()
                for w, t in self.adj[u]:
                    o = off[u] + np.array(t)
                    if comp[w] < 0:
                        comp[w] = cid
                        off[w] = o
                        members.append(w)
                        q.append(w)
                    else:
                        c = o - off[w]
                        if c.any():
                            cyc.append(c)
            rank = int(np.linalg.matrix_rank(np.array(cyc))) if cyc else 0
            out.append({"vertices": members, "rank": rank, "cycles": cyc,
                        "copies": _lattice_index(cyc) if rank == 3 else None})
        return out

    def subnet(self, vertices: Sequence[int]) -> "PeriodicNet":
        idx = {v: i for i, v in enumerate(vertices)}
        e = [(idx[u], idx[v], t) for u, v, t in self.edges if u in idx and v in idx]
        return PeriodicNet(len(vertices), e, self.frac[list(vertices)], self.cell,
                           [self.labels[v] for v in vertices], self.pbc)

    # -- simplification -----------------------------------------------------------------
    def simplified(self, keep_two: bool = False) -> "PeriodicNet":
        """Remove 1-connected vertices and contract 2-connected ones, repeatedly."""
        alive = [True] * self.n
        adj = [list(a) for a in self.adj]
        changed = True
        while changed:
            changed = False
            for v in range(self.n):
                if not alive[v]:
                    continue
                d = len(adj[v])
                if d <= 1:
                    for w, t in adj[v]:
                        adj[w] = [(x, s) for x, s in adj[w] if not (x == v and s == tuple(-y for y in t))]
                    adj[v] = []
                    alive[v] = False
                    changed = True
                elif d == 2 and not keep_two:
                    (a, sa), (b, sb) = adj[v]
                    if a == v or b == v:
                        continue
                    adj[a] = [(x, s) for x, s in adj[a] if not (x == v and s == tuple(-y for y in sa))]
                    adj[b] = [(x, s) for x, s in adj[b] if not (x == v and s == tuple(-y for y in sb))]
                    new = tuple(y - x for x, y in zip(sa, sb))     # a (at sa) to b (at sb), relative to a
                    if not (a == b and not any(new)):
                        if (b, new) not in adj[a]:
                            adj[a].append((b, new))
                            adj[b].append((a, tuple(-x for x in new)))
                    adj[v] = []
                    alive[v] = False
                    changed = True
        keep = [v for v in range(self.n) if alive[v]]
        idx = {v: i for i, v in enumerate(keep)}
        edges = []
        for v in keep:
            for w, t in adj[v]:
                edges.append((idx[v], idx[w], t))
        return PeriodicNet(len(keep), edges, self.frac[keep], self.cell, [self.labels[v] for v in keep], self.pbc)


def _lattice_index(vectors: Sequence[np.ndarray]) -> int:
    """|Z^3 / L| for the lattice L the integer vectors span (rank 3)."""
    M = [list(map(int, v)) for v in vectors]
    # integer row reduction (Hermite form) to three rows
    rows = [r for r in M if any(r)]
    basis = []
    for col in range(3):
        piv = [r for r in rows if r[col] != 0]
        rest = [r for r in rows if r[col] == 0]
        while len(piv) > 1:
            piv.sort(key=lambda r: abs(r[col]))
            p = piv[0]
            newp = [p]
            for r in piv[1:]:
                q = r[col] // p[col]
                rr = [x - q * y for x, y in zip(r, p)]
                if rr[col] != 0:
                    newp.append(rr)
                elif any(rr):
                    rest.append(rr)
            piv = newp
        if piv:
            basis.append(piv[0])
        rows = rest
    if len(basis) < 3:
        return 0
    return abs(int(round(np.linalg.det(np.array(basis, float)))))


# ---------------------------------------------------------------------------
# from a structure
# ---------------------------------------------------------------------------
def bonds(st: Structure, tolerance: float = 1.15, metal_metal: Optional[bool] = None) -> List[Tuple[int, int, Tuple]]:
    """The bonds the viewer draws (mace.gui.structures.bond_pairs, the twin of geom/core.js's bond
    rule): d < tolerance * (R_i + R_j) with the bond radii, longer than 0.1 Å; periodic images
    only along periodic directions. Metal-metal contacts count as bonds in a metal or alloy and
    not in a compound, unless ``metal_metal`` says otherwise."""
    from mace.gui.structures import bond_pairs
    w = st.wrapped()
    return bond_pairs(w.cell, w.cart, w.numbers, w.pbc, tolerance=tolerance, metal_metal=metal_metal)


def atom_net(st: Structure, tolerance: float = 1.15, metal_metal: bool = True) -> PeriodicNet:
    w = st.wrapped()
    return PeriodicNet(st.n, bonds(w, tolerance, metal_metal), w.frac, w.cell, w.symbols, w.pbc)


def _drop_molecules(net: PeriodicNet) -> Tuple[PeriodicNet, int]:
    comps = net.components()
    dim = sum(net.pbc)
    framework = [c for c in comps if c["rank"] > 0]
    if not framework or dim == 0:
        return net, 0
    keep = sorted(v for c in framework for v in c["vertices"])
    return net.subnet(keep), len(comps) - len(framework)


def sbu_net(st: Structure, atoms: PeriodicNet) -> Tuple[PeriodicNet, Dict]:
    """The single-node representation of a metal-organic framework."""
    sym = atoms.labels
    n = atoms.n
    metal = [s not in NONMETALS for s in sym]
    carbon_bonded = [False] * n
    for u, v, t in atoms.edges:
        if sym[v] == "C":
            carbon_bonded[u] = True
        if sym[u] == "C":
            carbon_bonded[v] = True
    inorganic = [metal[i] or (sym[i] not in ("C", "H") and not carbon_bonded[i]
                              and any(metal[w] for w, _ in atoms.adj[i])) for i in range(n)]
    # connected pieces of each kind, with offsets so a piece is whole
    piece = [-1] * n
    off = [None] * n
    pieces = []
    for s in range(n):
        if piece[s] >= 0 or sym[s] == "H":
            continue
        kind = inorganic[s]
        pid = len(pieces)
        piece[s] = pid
        off[s] = np.zeros(3, int)
        members, cyc = [s], []
        q = deque([s])
        while q:
            u = q.popleft()
            for w, t in atoms.adj[u]:
                if inorganic[w] != kind or sym[w] == "H":
                    continue
                o = off[u] + np.array(t)
                if piece[w] < 0:
                    piece[w] = pid
                    off[w] = o
                    members.append(w)
                    q.append(w)
                elif piece[w] == pid and (o - off[w]).any():
                    cyc.append(o - off[w])
        pieces.append({"inorganic": kind, "members": members, "infinite": bool(cyc)})
    rods = [p for p in pieces if p["inorganic"] and p["infinite"]]
    for p in rods:
        # a rod building unit: one vertex per metal atom (points of extension
        # are not attempted); its other atoms join a bonded metal
        mem = p["members"]
        p["members"] = []
        for m in mem:
            if metal[m]:
                piece[m] = len(pieces)
                off[m] = np.zeros(3, int)
                pieces.append({"inorganic": True, "members": [m], "infinite": False})
        for m in mem:
            if metal[m]:
                continue
            for w, t in atoms.adj[m]:
                if metal[w] and w in mem:
                    piece[m] = piece[w]
                    off[m] = -np.array(t)          # m sits at -t from that metal
                    pieces[piece[w]]["members"].append(m)
                    break
            else:
                piece[m] = -1
    # links between pieces
    links = defaultdict(set)       # piece -> {(other piece, shift between piece origins)}
    for u, v, t in atoms.edges:
        pu, pv = piece[u], piece[v]
        if pu < 0 or pv < 0 or pu == pv and not (np.array(t) + off[u] - off[v]).any():
            continue
        if pu == pv:
            continue
        s = tuple(int(x) for x in (np.array(t) + off[u] - off[v]))
        links[pu].add((pv, s))
        links[pv].add((pu, tuple(-x for x in s)))
    # vertices: inorganic units and organic pieces linking >= 3 units
    centroid = {}
    for pid, p in enumerate(pieces):
        if not p["members"]:
            continue
        f = np.array([atoms.frac[m] + off[m] for m in p["members"]])
        centroid[pid] = f.mean(0)
    is_vertex = {}
    for pid, p in enumerate(pieces):
        if not p["members"]:
            continue
        if p["inorganic"]:
            is_vertex[pid] = True
        else:
            k = len({x for x in links[pid] if pieces[x[0]]["inorganic"]})
            is_vertex[pid] = k >= 3
    verts = [pid for pid in sorted(is_vertex) if is_vertex[pid]]
    vidx = {p: i for i, p in enumerate(verts)}
    edges = []
    for pid in verts:
        for q, s in links[pid]:
            if q in vidx:
                edges.append((vidx[pid], vidx[q], s))
    for pid, p in enumerate(pieces):
        if not p["members"] or is_vertex.get(pid, True):
            continue
        ends = sorted({x for x in links[pid] if x[0] in vidx})
        if len(ends) == 2:
            (a, sa), (b, sb) = ends
            # linker origin at 0: unit a sits at -sa relative... links store shift from linker to unit
            edges.append((vidx[a], vidx[b], tuple(y - x for x, y in zip(sa, sb))))
    frac = np.array([centroid[p] for p in verts]) if verts else np.zeros((0, 3))
    labels = []
    for p in verts:
        from collections import Counter
        c = Counter(sym[m] for m in pieces[p]["members"])
        labels.append("".join(f"{k}{v if v > 1 else ''}" for k, v in sorted(c.items())))
    info = {"inorganic_units": sum(1 for p in verts if pieces[p]["inorganic"]),
            "organic_nodes": sum(1 for p in verts if not pieces[p]["inorganic"]),
            "linkers_as_edges": sum(1 for pid, p in enumerate(pieces) if p["members"] and not is_vertex.get(pid, True)),
            "rod_units": len(rods)}
    return PeriodicNet(len(verts), edges, frac, atoms.cell, labels, atoms.pbc), info


def underlying_net(st: Structure, mode: str = "auto", tolerance: float = 1.15,
                   metal_metal: Optional[bool] = None) -> Tuple[PeriodicNet, Dict]:
    """``metal_metal``: bond metal atoms to each other; by default yes for a
    framework's building units and a pure metal, no in an inorganic compound
    (CsCl is the bcu net of Cs-Cl bonds)."""
    sym = set(st.symbols)
    has_metal = any(s not in NONMETALS for s in sym)
    if mode == "auto":
        mode = "sbu" if ("C" in sym and has_metal) else "atoms"
    if metal_metal is None:
        metal_metal = mode == "sbu" or all(s not in NONMETALS for s in sym)
    atoms = atom_net(st, tolerance, metal_metal)
    atoms, dropped = _drop_molecules(atoms)
    info = {"mode": mode, "molecules_dropped": dropped, "atoms": atoms.n}
    if mode == "sbu":
        net, extra = sbu_net(st, atoms)
        info.update(extra)
    elif mode in ("atoms", "all"):
        net = atoms
    else:
        raise ValueError("mode is auto, atoms, sbu or all")
    if mode != "all":
        net = net.simplified()
    return net, info


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def vertex_classes(net: PeriodicNet, k: int = 10) -> List[Dict]:
    """Vertices grouped by coordination sequence (one BFS per symmetry-distinct
    vertex when spglib finds the net's symmetry)."""
    reps = list(range(net.n))
    eq = list(range(net.n))
    if net.n and all(net.pbc):
        try:
            import spglib
            deg = net.degree()
            ds = spglib.get_symmetry_dataset((net.cell, net.frac % 1.0, [d + 1 for d in deg]), symprec=1e-2)
            if ds is not None:
                eq = [int(x) for x in (ds.equivalent_atoms if hasattr(ds, "equivalent_atoms") else ds["equivalent_atoms"])]
        except Exception:
            pass
    cs_of = {}
    for r in sorted(set(eq)):
        cs_of[r] = net.coordination_sequence(r, k)
    groups: Dict[Tuple, Dict] = {}
    for v in range(net.n):
        cs = tuple(cs_of[eq[v]]) if eq[v] in cs_of else tuple(net.coordination_sequence(v, k))
        g = groups.setdefault(cs, {"cs": list(cs), "cn": cs[0] if cs else 0, "vertices": [], "labels": set()})
        g["vertices"].append(v)
        g["labels"].add(net.labels[v])
    out = []
    for cs, g in groups.items():
        g["count"] = len(g["vertices"])
        g["labels"] = sorted(g["labels"])
        out.append(g)
    out.sort(key=lambda g: (g["cn"], g["cs"]))
    return out


def td10(classes: Sequence[Dict]) -> float:
    tot = sum(g["count"] for g in classes)
    if not tot:
        return 0.0
    return sum(g["count"] * (1 + sum(g["cs"][:10])) for g in classes) / tot


def analyze(st: Structure, mode: str = "auto", tolerance: float = 1.15, rings: bool = True,
            reference=None, metal_metal: Optional[bool] = None) -> Dict:
    """Net, invariants, interpenetration and the RCSR match of a structure."""
    from . import rcsr
    net, info = underlying_net(st, mode, tolerance, metal_metal)
    dim = sum(st.pbc)
    comps = net.components()
    nets = []
    seen_keys = {}
    total_copies = 0
    for c in comps:
        sub = net.subnet(c["vertices"])
        classes = vertex_classes(sub)
        key = tuple(sorted(tuple(g["cs"]) for g in classes))
        entry = {"vertices": len(c["vertices"]), "periodicity": c["rank"], "copies": c["copies"],
                 "vertex_classes": [{"cn": g["cn"], "cs": g["cs"], "count": g["count"], "labels": g["labels"]}
                                    for g in classes],
                 "td10": td10(classes)}
        if rings and sub.n:
            for g, ec in zip(classes, entry["vertex_classes"]):
                ec["point_symbol"] = sub.point_symbol(g["vertices"][0])
        entry["matches"] = rcsr.match(entry["vertex_classes"], dims=c["rank"], reference=reference)
        entry["topology"] = entry["matches"][0]["name"] if entry["matches"] else None
        if c["rank"] == dim and dim > 0:
            total_copies += c["copies"] or 1
        entry["same_as"] = seen_keys.get(key)
        seen_keys.setdefault(key, len(nets))
        nets.append(entry)
    framework = [n for n in nets if n["periodicity"] == dim]
    lower = [n for n in nets if 0 < n["periodicity"] < dim]
    interp = total_copies if framework else None
    shown = framework or lower
    summary = {"interpenetration": interp,
               "topologies": sorted({n["topology"] or "unknown" for n in shown}),
               "lower_dimensional_components": len(lower)}
    if interp and interp > 1:
        summary["note"] = f"{interp}-fold interpenetrated"
    elif lower:
        kinds = sorted({n["periodicity"] for n in lower})
        what = " and ".join({1: "chains", 2: "layers"}[k] for k in kinds)
        summary["note"] = f"{len(lower)} {what} per cell" + ("" if framework else f" ({dim}D crystal of lower-dimensional nets)")
    out = {"mode": info["mode"], "simplification": info, "vertices": net.n, "edges": len(net.edges),
           "nets": nets, "summary": summary, "dimensionality": dim, "citation": rcsr.CITATION}
    out["net_frac"] = net.frac.tolist()
    out["net_edges"] = [[u, v, list(t)] for u, v, t in net.edges]
    out["net_labels"] = net.labels
    return out
