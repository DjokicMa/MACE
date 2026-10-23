"""Crystal morphology (Materials Studio's Morphology tools):

- BFDH (Bravais-Friedel-Donnay-Harker): a face's distance from the center
  is proportional to 1/d_hkl, with d the spacing of the first reflection
  along (hkl) the space group allows (systematic absences halve it);
- Wulff construction from surface energies you give per family: distance
  proportional to gamma_hkl (J/m^2 or any consistent unit).

Each family is expanded by the crystal's Laue group, the shape is the
intersection of the half-spaces n.x <= distance (scipy HalfspaceIntersection),
and the result is a triangle mesh with the area fraction of each family.
Attachment-energy morphology (which needs a force field) is not modeled.
"""
from __future__ import annotations

import itertools
import math
from functools import reduce
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .structure import Structure


def _gcd3(h):
    return reduce(math.gcd, [abs(int(x)) for x in h])


def _ops(st: Structure):
    from .refine import symmetry_ops
    rot, tr, _ = symmetry_ops(st)
    return rot, tr


def family_members(h: Sequence[int], rotations) -> List[Tuple[int, int, int]]:
    rots = list(rotations) + [-np.asarray(r) for r in rotations]
    return sorted({tuple(int(v) for v in np.asarray(h) @ np.asarray(R)) for R in rots})


def bfdh_distance(st: Structure, h, rotations, translations) -> float:
    from .refine import absent
    rec = np.linalg.inv(st.cell).T
    g = np.asarray(h) @ rec
    d = 1.0 / np.linalg.norm(g)
    for n in range(1, 7):
        if not absent(np.array([np.asarray(h) * n]), rotations, translations)[0]:
            return n / d                                # 1 / d_eff
    return 6 / d


def wulff(st: Structure, energies: Optional[Dict[Tuple[int, int, int], float]] = None, max_index: int = 2,
          method: Optional[str] = None) -> Dict:
    """The crystal shape: BFDH when no energies are given, else Wulff."""
    st.require_3d("Morphology")
    from .pxrd import conventional
    st = conventional(st)
    from scipy.spatial import ConvexHull, HalfspaceIntersection
    rot, tr = _ops(st)
    rec = np.linalg.inv(st.cell).T
    method = method or ("wulff" if energies else "bfdh")
    fams: Dict[Tuple[int, int, int], Dict] = {}
    if method == "wulff":
        if not energies:
            raise ValueError("Wulff needs surface energies: {(h, k, l): gamma}")
        for h, gam in energies.items():
            h = tuple(int(x) for x in h)
            if not any(h) or gam <= 0:
                raise ValueError(f"bad family {h}: gamma must be positive")
            mem = family_members(h, rot)
            fams[max(mem)] = {"hkl": list(max(mem)), "members": mem, "distance": float(gam), "gamma": float(gam)}
    else:
        r = range(-max_index, max_index + 1)
        for h in itertools.product(r, r, r):
            if not any(h) or _gcd3(h) != 1:
                continue
            mem = family_members(h, rot)
            key = max(mem)
            if key in fams:
                continue
            fams[key] = {"hkl": list(key), "members": mem, "distance": bfdh_distance(st, key, rot, tr)}
    hs, owner = [], []
    keys = list(fams)
    for fi, k in enumerate(keys):
        for m in fams[k]["members"]:
            n = np.asarray(m) @ rec
            n = n / np.linalg.norm(n)
            hs.append(np.r_[n, -fams[k]["distance"]])
            owner.append(fi)
    hs = np.array(hs)
    try:
        hi = HalfspaceIntersection(hs, np.zeros(3))
    except Exception as exc:
        raise ValueError(f"the planes do not enclose a shape ({exc}); add more families") from exc
    verts = hi.intersections
    hull = ConvexHull(verts)
    scale = float(np.linalg.norm(verts, axis=1).max())
    tri, tri_fam = [], []
    area = np.zeros(len(keys))
    for simp, eq in zip(hull.simplices, hull.equations):
        normal = eq[:3]
        j = int(np.argmax(hs[:, :3] @ normal))
        fi = owner[j]
        a, b, c = verts[simp]
        nrm = np.cross(b - a, c - a)
        if nrm @ normal < 0:
            simp = simp[[0, 2, 1]]
        area[fi] += 0.5 * np.linalg.norm(nrm)
        tri.append([int(x) for x in simp])
        tri_fam.append(fi)
    tot = area.sum() or 1.0
    families = []
    for fi, k in enumerate(keys):
        if area[fi] <= 1e-9 * tot:
            continue
        f = fams[k]
        families.append({"hkl": f["hkl"], "multiplicity": len(f["members"]), "distance": f["distance"] / scale,
                         "area_fraction": float(area[fi] / tot), "index": fi,
                         **({"gamma": f["gamma"]} if "gamma" in f else {}),
                         **({"d_eff": 1.0 / f["distance"]} if method == "bfdh" else {})})
    families.sort(key=lambda f: -f["area_fraction"])
    ext = verts.max(0) - verts.min(0)
    return {"method": method, "vertices": (verts / scale).tolist(), "triangles": tri, "triangle_family": tri_fam,
            "families": families, "aspect": (ext / ext.max()).tolist(),
            "volume": float(hull.volume / scale ** 3), "area": float(tot / scale ** 2)}


def parse_energies(text: str) -> Dict[Tuple[int, int, int], float]:
    """"100:1.2, 111:0.9, 1-10:1.05" -> {(1,0,0): 1.2, ...} (a minus sign
    before a digit negates it)."""
    out = {}
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        hkl, _, g = part.partition(":")
        digits = [int(x) for x in __import__("re").findall(r"-?\d", hkl.replace(" ", ""))]
        if len(digits) != 3:
            raise ValueError(f"bad family {hkl!r}: write it as 100, 1-10 or 1 0 0")
        out[tuple(digits)] = float(g)
    return out
