"""Band edges, band widths, spin splitting and Fermi velocities of a BAND run,
from its BAND.DAT (or BAND fort.25) as mace/transport/mass.py reads it
(``mass.read_band``: per spin channel the bands along the path, E - E_F in
hartree, and the path's segments with their labels and fractional k).

Every value is along the computed k path, not over the whole Brillouin zone.
A band is occupied when its maximum along the path is at most
``mass.OCCUPIED_TOL`` (1 mHa) above E_F, the rule of mace tau's masses
(CRYSTAL puts E_F at the valence-band top of the SCF k mesh, and a path
point can lie a few meV above it). The gap is CBM - VBM over both spin
channels; a gap below property_keys.METAL_GAP_EV (or a band overlap) makes
the run gapless along the path.

* metal or semimetal (a gapless run): decided by the total DOS at E_F per
  atom, both spins, averaged over E_F +- 0.05 eV, of the DOSS run of the
  same parent (its DOSS.DAT in the run's folder or in a folder beside it):
  below property_keys.SEMIMETAL_DOS_PER_ATOM (0.05 states/eV/atom) a
  semimetal (graphene's Dirac point: 0.002), else a metal (bcc Fe: 0.89).
  Without a DOSS run the class is "metal or semimetal (no DOS to tell)".
  The extractor's electronic_classification of the run uses the same value
  and threshold (advanced_electronic_analyzer.fermi_dos_per_atom);

* gap character: direct when the smallest vertical gap (same k, same spin)
  equals the fundamental gap within 2 * mass.EDGE_TOL; the k points of the
  VBM, the CBM and the smallest vertical gap, named by the path's labels
  ("Γ") or by where they lie on a segment ("0.76 of Γ–X"), with their
  fractional coordinates in the details;
* band widths: the valence (conduction) manifold is the connected set of
  occupied (empty) bands reached from the VBM (CBM) through bands whose
  energy ranges along the path overlap (or leave less than
  MANIFOLD_GAP_EV between them); its width is VBM - its bottom
  (its top - CBM). A manifold that reaches the first (last) band the file
  holds may go on beyond it: then no width is stored (the details say so).
  The dispersion of the single band at each edge is stored too. The
  conduction manifold's width stays in the details only: the empty bands of
  a Gaussian basis run on into high virtual states with no gap between them
  (159 eV for a 3,4^2T11-CA cell), so it is no property of the material;
* spin splitting (spin-polarized runs): the alpha - beta difference of each
  band edge, and the largest |E_alpha - E_beta| of the bands within 1 eV of
  E_F at any k;
* Fermi velocity (metals and semimetals): v_F = |dE/dk| / hbar at every point where a band
  crosses E_F along the path (the chord of the two points around it; a
  band that touches E_F at a k point, graphene's Dirac point, takes each
  side's slope), with k in bohr^-1 (2 pi included) and E in hartree, so the
  slope is in atomic units of velocity (2.18769e6 m/s).
"""
from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mace.properties.registry import Output, Sibling, Skip, derived

from mace.constants import HARTREE_TO_EV as HARTREE_EV  # noqa: E402
AU_VELOCITY_M_S = 2.18769126364e6
NEAR_FERMI_EV = 1.0
# narrower gaps between band ranges along the path do not end a manifold (the path samples a few lines of
# the zone; off them the bands' ranges close such gaps: SOD's O 2p band shows a 30 meV one on H-Γ-P-N)
MANIFOLD_GAP_EV = 0.1
SPINS = ("alpha", "beta")


def _band_file(output_file: Path) -> Optional[Path]:
    from mace.properties.defs.effective_masses import _band_file as find
    return find(output_file)


def _wrap(f) -> Tuple[float, ...]:
    """Fractional k modulo a reciprocal lattice vector (in (-1/2, 1/2])."""
    import numpy as np
    g = np.asarray(f, float)
    g = g - np.round(g)
    g[np.isclose(g, -0.5, atol=1e-4)] = 0.5
    return tuple(round(float(v), 4) + 0.0 for v in g)


def _frac_text(f) -> str:
    from fractions import Fraction
    return "(" + ",".join(str(Fraction(float(v)).limit_denominator(48)) for v in f) + ")"


def _kpoints(bp) -> Dict[int, Dict[str, Any]]:
    """row -> {label, frac, vertex, segment}: a vertex row takes the path's label, an inner row its place
    on the segment ("0.76 of Γ–X", the form of mass.band_masses)."""
    rows: Dict[int, Dict[str, Any]] = {}
    for seg in bp.segments:
        n = len(seg.idx)
        for j, r in enumerate(seg.idx):
            t = j / (n - 1) if n > 1 else 0.0
            frac = None
            if seg.start is not None and seg.end is not None:
                frac = [float(v) for v in seg.start + t * (seg.end - seg.start)]
            vertex = j in (0, n - 1)
            if vertex:
                label = seg.labels[0] if j == 0 else seg.labels[1]
            else:
                label = f"{t:.2f} of {seg.name}"
            if r not in rows or (vertex and not rows[r]["vertex"]):
                rows[r] = {"label": label, "frac": frac, "vertex": vertex, "segment": seg.name}
    return rows


def _key(rows, r):
    k = rows.get(r) or {}
    if k.get("frac") is not None:
        return ("k",) + _wrap(k["frac"])
    if k.get("vertex"):
        return ("label", k["label"])
    return ("row", r)


def _where(rows, r) -> Dict[str, Any]:
    k = rows.get(r) or {"label": f"row {r}", "frac": None}
    out = {"label": k["label"], "row": int(r)}
    if k.get("frac") is not None:
        out["frac"] = [round(v, 6) for v in k["frac"]]
        out["frac_text"] = _frac_text(k["frac"])
    return out


def _at_vertex(rows, values, r: int, top: bool) -> int:
    """Row ``r`` of an edge, or the high-symmetry row the band reaches within mass.EDGE_TOL of it: a band
    flat to the file's precision ties on several rows, and the first of them is an arbitrary inner point
    ("0.99 of H–Γ" for Γ)."""
    import numpy as np
    from mace.transport import mass
    if (rows.get(r) or {}).get("vertex"):
        return r
    v = np.asarray(values, float)
    near = [int(i) for i in np.flatnonzero(np.abs(v - v[r]) <= mass.EDGE_TOL) if (rows.get(int(i)) or {}).get("vertex")]
    if not near:
        return r
    return max(near, key=lambda i: v[i]) if top else min(near, key=lambda i: v[i])


def _manifold(intervals: List[Tuple[float, float, Any]], start: Any, down: bool, tol: float):
    """The bands connected to band ``start`` through overlapping energy ranges, going down (valence) or up
    (conduction): (far edge, bands). ``intervals``: (min, max, band) per band."""
    if down:
        order = sorted(intervals, key=lambda t: -t[1])
        top = next(t for t in intervals if t[2] == start)
        low, members = top[0], {start}
        for lo, hi, b in order:
            if b in members or hi > top[1] + tol:
                continue
            if hi >= low - tol:
                low = min(low, lo)
                members.add(b)
            else:
                break
        return low, members
    order = sorted(intervals, key=lambda t: t[0])
    bot = next(t for t in intervals if t[2] == start)
    high, members = bot[1], {start}
    for lo, hi, b in order:
        if b in members or lo < bot[0] - tol:
            continue
        if lo <= high + tol:
            high = max(high, hi)
            members.add(b)
        else:
            break
    return high, members


def _crossings(bp, blk, spin: Optional[str], first_band: int) -> List[Dict[str, Any]]:
    """Every point where a band of ``blk`` crosses (or touches) E_F along the path, with v_F."""
    import numpy as np
    from mace.transport import mass
    tol = mass.EDGE_TOL
    rows = _kpoints(bp)
    out: List[Dict[str, Any]] = []
    for seg in bp.segments:
        if len(seg.idx) < 2 or not seg.dk:
            continue
        sub = blk[seg.idx]
        cand = np.flatnonzero((sub.min(axis=0) <= tol) & (sub.max(axis=0) >= -tol))
        for b in cand:
            col = sub[:, b]
            s = np.where(np.abs(col) <= tol, 0, np.sign(col)).astype(int)
            n = len(col)

            def rec(i0, i1, how):
                slope = abs(float(col[i1] - col[i0])) / (abs(i1 - i0) * seg.dk)
                lo, hi = min(i0, i1), max(i0, i1)
                if s[lo] != 0 and s[hi] != 0:      # interpolate where E crosses 0
                    t = float(col[lo] / (col[lo] - col[hi]))
                    r = seg.idx[lo] if t < 0.5 else seg.idx[hi]
                else:
                    r = seg.idx[lo] if s[lo] == 0 else seg.idx[hi]
                out.append({"spin": spin, "band": int(b) + first_band, "segment": seg.name, **_where(rows, r),
                            "kind": how, "v_au": slope, "v_m_s": slope * AU_VELOCITY_M_S})
            i = 0
            while i < n - 1:
                if s[i] * s[i + 1] == -1:
                    rec(i, i + 1, "crossing")
                    i += 1
                    continue
                if s[i] == 0:
                    a = i
                    while i + 1 < n and s[i + 1] == 0:
                        i += 1
                    z = i
                    left = a - 1 if a > 0 else None
                    right = z + 1 if z < n - 1 else None
                    if left is not None and right is not None and s[left] == -s[right]:
                        rec(left, right, "crossing")
                    else:
                        for nb, at in ((left, a), (right, z)):
                            if nb is not None:
                                rec(at, nb, "touch")
                i += 1
            if s[-1] == 0 and s[-2] != 0:          # the loop stops before a lone zero at the segment's end
                rec(n - 1, n - 2, "touch")
    return out


@functools.lru_cache(maxsize=16)
def _analyze(path: str, mtime: float, size: int, first_band: int) -> Dict[str, Any]:
    import numpy as np
    from mace.transport import mass
    from mace.utils import property_keys
    from mace.utils.property_keys import METAL_GAP_EV
    bp = mass.read_band(Path(path))
    rows = _kpoints(bp)
    blocks = list(bp.blocks)
    spin_polarized = len(blocks) == 2
    identical = spin_polarized and blocks[0].shape == blocks[1].shape and bool(
        np.allclose(blocks[0], blocks[1], atol=2e-5))        # mass.band_masses' rule
    names = list(SPINS) if spin_polarized else [None]
    tol = mass.OCCUPIED_TOL
    per: Dict[str, Dict[str, Any]] = {}
    for name, blk in zip(names, blocks):
        mx, mn = blk.max(axis=0), blk.min(axis=0)
        occ = mx <= tol
        rec: Dict[str, Any] = {"bands": int(blk.shape[1]), "occupied_bands": int(occ.sum())}
        if occ.any():
            # bands degenerate at the edge: the top one is the band at the VBM (its dispersion is stored)
            top = float(mx[occ].max())
            b = int(np.flatnonzero(occ & (mx >= top - mass.EDGE_TOL))[-1])
            r = _at_vertex(rows, blk[:, b], int(np.argmax(blk[:, b])), True)
            rec["vbm"] = {"energy_ev": top * HARTREE_EV, "band": b + first_band, **_where(rows, r)}
            rec["_vbm"] = (top, r, b)
        if (~occ).any():
            bot = float(mn[~occ].min())
            b = int(np.flatnonzero(~occ & (mn <= bot + mass.EDGE_TOL))[0])
            r = _at_vertex(rows, blk[:, b], int(np.argmin(blk[:, b])), False)
            rec["cbm"] = {"energy_ev": bot * HARTREE_EV, "band": b + first_band, **_where(rows, r)}
            rec["_cbm"] = (bot, r, b)
        if occ.any() and (~occ).any():
            gap = (rec["_cbm"][0] - rec["_vbm"][0]) * HARTREE_EV
            rec["gap_ev"] = max(gap, 0.0)
            rec["metallic"] = gap < METAL_GAP_EV
            vert = blk[:, ~occ].min(axis=1) - blk[:, occ].max(axis=1)
            r = int(np.argmin(vert))
            rec["direct_gap_ev"] = max(float(vert[r]) * HARTREE_EV, 0.0)
            rec["direct_gap_at"] = _where(rows, _at_vertex(rows, vert, r, False))
        rec["_intervals"] = [(float(mn[b]), float(mx[b]), int(b)) for b in range(blk.shape[1])]
        rec["_occ"] = occ
        per[name or "total"] = rec
    chans = [per[n or "total"] for n in names]
    have_v = [c for c in chans if "_vbm" in c]
    have_c = [c for c in chans if "_cbm" in c]
    res: Dict[str, Any] = {"file": Path(path).name, "spin_polarized": spin_polarized, "spins_identical": identical,
                           "first_band": first_band, "num_kpoints": int(blocks[0].shape[0]),
                           "segments": [s.name for s in bp.segments], "notes": list(bp.notes)}
    if not have_v or not have_c:
        res["character"] = None
        res["reason"] = ("no empty band in the file (the bands written stop below E_F)" if have_v
                         else "no occupied band in the file")
    else:
        cv = max(have_v, key=lambda c: c["_vbm"][0])
        cc = min(have_c, key=lambda c: c["_cbm"][0])
        gap = (cc["_cbm"][0] - cv["_vbm"][0]) * HARTREE_EV
        res["gap_ev"] = max(gap, 0.0)
        res["vbm"] = dict(cv["vbm"], spin=next(n for n in names if per[n or "total"] is cv))
        res["cbm"] = dict(cc["cbm"], spin=next(n for n in names if per[n or "total"] is cc))
        res["gapless"] = gap < METAL_GAP_EV
        if res["gapless"]:
            # metal or semimetal: the DOS at E_F decides (classify)
            res["character"] = property_keys.GAPLESS_NO_DOS
        else:
            best = min((c for c in chans if "direct_gap_ev" in c), key=lambda c: c["direct_gap_ev"])
            res["direct_gap_ev"] = best["direct_gap_ev"]
            res["direct_gap_at"] = best["direct_gap_at"]
            direct = best["direct_gap_ev"] - gap <= 2 * mass.EDGE_TOL * HARTREE_EV
            res["character"] = "direct" if direct else "indirect"
            if direct:      # name the edges where the vertical gap is (a degenerate VBM can sit elsewhere too)
                res["vbm"].update(best["direct_gap_at"])
                res["cbm"].update(best["direct_gap_at"])
    # band widths (both spins pooled; per spin in the details)
    first, last = 0, int(blocks[0].shape[1]) - 1
    widths: Dict[str, Any] = {}
    for label, chs in [("all", chans)] + ([(n, [per[n]]) for n in SPINS] if spin_polarized and not identical else []):
        # (min, max, (channel, band)) of every band of the channels pooled
        ivals = [(lo, hi, (ci, b)) for ci, c in enumerate(chs) for lo, hi, b in c["_intervals"]]
        w: Dict[str, Any] = {}
        vch = [c for c in chs if "_vbm" in c]
        cch = [c for c in chs if "_cbm" in c]
        metal = bool(res.get("gapless")) or any(c.get("metallic") for c in chs)
        if vch and cch and not metal:
            cv = max(vch, key=lambda c: c["_vbm"][0])
            cc = min(cch, key=lambda c: c["_cbm"][0])
            iv = next(i for i, c in enumerate(chs) if c is cv)
            ic = next(i for i, c in enumerate(chs) if c is cc)
            occ_iv = [t for t in ivals if chs[t[2][0]]["_occ"][t[2][1]]]
            emp_iv = [t for t in ivals if not chs[t[2][0]]["_occ"][t[2][1]]]
            gtol = MANIFOLD_GAP_EV / HARTREE_EV
            low, vm = _manifold(occ_iv, (iv, cv["_vbm"][2]), True, gtol)
            high, cm = _manifold(emp_iv, (ic, cc["_cbm"][2]), False, gtol)
            vb = sorted({b for _, b in vm})
            cb = sorted({b for _, b in cm})
            w["valence"] = {"width_ev": (cv["_vbm"][0] - low) * HARTREE_EV, "bottom_ev": low * HARTREE_EV,
                            "bands": [vb[0] + first_band, vb[-1] + first_band],
                            "complete": not (vb[0] == first and first_band > 1)}
            w["conduction"] = {"width_ev": (high - cc["_cbm"][0]) * HARTREE_EV, "top_ev": high * HARTREE_EV,
                               "bands": [cb[0] + first_band, cb[-1] + first_band], "complete": cb[-1] != last}
            if not w["valence"]["complete"]:
                w["valence"]["note"] = "reaches the first band the file holds: the manifold may go on below it"
            if not w["conduction"]["complete"]:
                w["conduction"]["note"] = "reaches the last band the file holds: the manifold may go on above it"
            lo_v, hi_v, _ = next(t for t in cv["_intervals"] if t[2] == cv["_vbm"][2])
            lo_c, hi_c, _ = next(t for t in cc["_intervals"] if t[2] == cc["_cbm"][2])
            w["vbm_band_dispersion_ev"] = (hi_v - lo_v) * HARTREE_EV
            w["cbm_band_dispersion_ev"] = (hi_c - lo_c) * HARTREE_EV
        widths[label] = w
    res["widths"] = widths
    # spin splitting
    if spin_polarized:
        a, b = per["alpha"], per["beta"]
        sp: Dict[str, Any] = {}
        if "_vbm" in a and "_vbm" in b and not a.get("metallic") and not b.get("metallic"):
            sp["vbm_mev"] = (a["_vbm"][0] - b["_vbm"][0]) * HARTREE_EV * 1000
        if "_cbm" in a and "_cbm" in b and not a.get("metallic") and not b.get("metallic"):
            sp["cbm_mev"] = (a["_cbm"][0] - b["_cbm"][0]) * HARTREE_EV * 1000
        A, B = blocks
        if A.shape == B.shape:
            near = (np.abs(A) <= NEAR_FERMI_EV / HARTREE_EV) | (np.abs(B) <= NEAR_FERMI_EV / HARTREE_EV)
            if near.any():
                d = np.where(near, np.abs(A - B), 0.0)
                r, c = np.unravel_index(int(np.argmax(d)), d.shape)
                sp["max_near_fermi_mev"] = float(d[r, c]) * HARTREE_EV * 1000
                sp["max_at"] = dict(_where(rows, r), band=int(c) + first_band)
        if "gap_ev" in a:
            sp["gap_alpha_ev"] = a["gap_ev"]
        if "gap_ev" in b:
            sp["gap_beta_ev"] = b["gap_ev"]
        res["spin_splitting"] = sp
    # Fermi velocities (a metal or semimetal along the path)
    if res.get("gapless"):
        cr: List[Dict[str, Any]] = []
        for name, blk in zip(names, blocks):
            if identical and name == "beta":
                break
            cr += _crossings(bp, blk, None if identical else name, first_band)
        res["fermi_crossings"] = cr
    res["channels"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in per.items()}
    return res


def analyze(band_file: Path, first_band: int = 1) -> Dict[str, Any]:
    """The band-edge analysis of a BAND.DAT/BAND f25 (cached per file); ``first_band``: the band the
    file's first column holds (band_start of the run)."""
    p = Path(band_file)
    st = p.stat()
    return _analyze(str(p), st.st_mtime, st.st_size, int(first_band or 1))


def _run(ctx) -> Dict[str, Any]:
    if (ctx.dimensionality or 3) == 0:
        raise Skip("a molecule has no band structure")
    try:
        return analyze(ctx.sibling("BAND"), ctx.value("band_start") or 1)
    except (OSError, ValueError) as e:
        raise Skip(f"the band file could not be read: {e}")


def _dos_file(output_file: Path) -> Optional[Path]:
    from mace.utils.property_extractor import CrystalPropertyExtractor
    return CrystalPropertyExtractor._parent_dos_file(Path(output_file))


def classify(r: Dict[str, Any], dos_file: Optional[Path], n_atoms: Optional[int]) -> Dict[str, Any]:
    """The analysis ``r`` of a gapless run with its class: metal or semimetal by N(E_F) per atom of
    ``dos_file`` (property_keys.classify_gapless), else GAPLESS_NO_DOS; a gapped one as it is."""
    from mace.utils import property_keys as K
    from mace.utils.advanced_electronic_analyzer import fermi_dos_per_atom
    if not r.get("gapless"):
        return r
    r = dict(r)
    info: Dict[str, Any] = {"threshold": K.SEMIMETAL_DOS_PER_ATOM, "window_ev": K.FERMI_DOS_WINDOW_EV,
                            "unit": "states/eV/atom", "atoms": n_atoms,
                            "file": Path(dos_file).name if dos_file else None}
    n_ef = None
    if dos_file is None:
        info["reason"] = "no DOSS run of the same parent beside the run"
    elif not n_atoms:
        info["reason"] = "no atom count in the output"
    else:
        n_ef = fermi_dos_per_atom(dos_file, n_atoms)
        if n_ef is None:
            info["reason"] = "the DOSS.DAT could not be read, or its energies do not reach E_F"
    info["per_atom"] = n_ef
    r["character"] = K.classify_gapless(n_ef) or K.GAPLESS_NO_DOS
    r["fermi_dos"] = info
    return r


_SIB = (Sibling("BAND", find=_band_file, description="the run's BAND.DAT (or BAND fort.25)"),)
_SRC = "BAND.DAT (or BAND fort.25) via mace.transport.mass.read_band, the reader of mace tau's masses"


@derived("band_gap_character", calc_types=("BAND",), category="band_structure", group="Band gap",
         siblings=_SIB + (Sibling("DOSS", required=False, find=_dos_file,
                                  description="the DOSS.DAT of the same parent (in the run's folder or a folder "
                                              "beside it): metal or semimetal of a gapless run"),),
         version="2", source=_SRC + "; N(E_F) of the DOSS.DAT via advanced_electronic_analyzer.fermi_dos_per_atom",
         outputs=(Output("band_path_gap_type", "category", "text", label="Gap type (path)",
                         description="direct or indirect along the band path (both spins); a gapless run (gap "
                                     "< 0.01 eV, or bands overlapping): semimetal when the total DOS at E_F is "
                                     "below 0.05 states/eV/atom, else metal, or 'metal or semimetal (no DOS to "
                                     "tell)' without a DOSS run"),
                  Output("band_path_vbm_k", "category", "text", label="VBM k point",
                         description="k point of the valence-band maximum: the path label, or where it lies "
                                     "on a segment"),
                  Output("band_path_cbm_k", "category", "text", label="CBM k point",
                         description="k point of the conduction-band minimum: the path label, or where it lies "
                                     "on a segment"),
                  Output("band_path_direct_gap_k", "category", "text", label="Smallest direct gap at",
                         description="k point of the smallest vertical (same k, same spin) gap"),
                  Output("band_path_gap_alpha_ev", "eV", label="Gap (α spin)", precision=5,
                         description="gap of the alpha channel alone along the path (spin-polarized runs)"),
                  Output("band_path_gap_beta_ev", "eV", label="Gap (β spin)", precision=5,
                         description="gap of the beta channel alone along the path (spin-polarized runs)"),
                  Output("band_path_fermi_dos_per_atom", "states/(eV·atom)",
                         label="N(E<sub>F</sub>) per atom (metal or semimetal)", precision=4,
                         description="total DOS at E_F per atom, both spins, averaged over E_F ± 0.05 eV, of the "
                                     "DOSS run of the same parent: what decides metal or semimetal of a gapless "
                                     "run"),
                  Output("band_path_semimetal_threshold", "states/(eV·atom)",
                         label="Semimetal threshold of N(E<sub>F</sub>)", precision=4,
                         description="N(E_F) per atom below which a gapless run is a semimetal"),
                  Output("band_edges", "JSON", "json", blob=True, label="Band edges (details)",
                         description="per spin: VBM, CBM (energy, band, k label and fractional k), gap, smallest "
                                     "vertical gap and where; band widths; spin splitting; Fermi-level crossings; "
                                     "N(E_F) of a gapless run")))
def _band_gap_character(ctx):
    """Direct or indirect gap, with the k points of the VBM and CBM; metal or semimetal of a gapless run."""
    from mace.utils.advanced_electronic_analyzer import atoms_per_cell
    r = _run(ctx)
    if r.get("character") is None:
        raise Skip(r.get("reason") or "no band edges")
    r = classify(r, ctx.sibling("DOSS"), atoms_per_cell(ctx.properties, ctx.content))
    vals: Dict[str, Any] = {"band_path_gap_type": r["character"], "band_edges": r}
    if r.get("gapless"):
        fd = r["fermi_dos"]
        vals["band_path_semimetal_threshold"] = fd["threshold"]
        vals["band_path_fermi_dos_per_atom"] = fd["per_atom"]
    else:
        vals["band_path_vbm_k"] = r["vbm"]["label"]
        vals["band_path_cbm_k"] = r["cbm"]["label"]
        vals["band_path_direct_gap_k"] = r["direct_gap_at"]["label"]
    sp = r.get("spin_splitting") or {}
    vals["band_path_gap_alpha_ev"] = sp.get("gap_alpha_ev")
    vals["band_path_gap_beta_ev"] = sp.get("gap_beta_ev")
    return vals


@derived("band_widths", calc_types=("BAND",), category="band_structure", group="Band widths",
         siblings=_SIB, version="1", source=_SRC,
         outputs=(Output("valence_band_width_ev", "eV", label="Valence band width", precision=5,
                         description="VBM - bottom of the valence manifold (the occupied bands connected to the "
                                     "VBM through overlapping energy ranges), along the path, both spins"),
                  Output("vbm_band_dispersion_ev", "eV", label="VBM band dispersion", precision=5,
                         description="max - min along the path of the band at the VBM"),
                  Output("cbm_band_dispersion_ev", "eV", label="CBM band dispersion", precision=5,
                         description="max - min along the path of the band at the CBM")))
def _band_widths(ctx):
    """Valence band width and the dispersion of the edge bands along the path (a gapped run)."""
    r = _run(ctx)
    if r.get("character") is None or r.get("gapless"):
        raise Skip("no gap along the path: the valence and conduction manifolds are not separate")
    w = (r.get("widths") or {}).get("all") or {}
    if not w:
        raise Skip("no band edges")
    vals = {"vbm_band_dispersion_ev": w.get("vbm_band_dispersion_ev"),
            "cbm_band_dispersion_ev": w.get("cbm_band_dispersion_ev")}
    if w["valence"]["complete"]:
        vals["valence_band_width_ev"] = w["valence"]["width_ev"]
    return vals


@derived("band_spin_splitting", calc_types=("BAND",), category="band_structure", group="Spin splitting",
         siblings=_SIB, version="1", source=_SRC,
         outputs=(Output("vbm_spin_splitting_mev", "meV", label="VBM spin splitting (α − β)", precision=4,
                         description="VBM of the alpha channel - VBM of the beta channel"),
                  Output("cbm_spin_splitting_mev", "meV", label="CBM spin splitting (α − β)", precision=4,
                         description="CBM of the alpha channel - CBM of the beta channel"),
                  Output("band_max_spin_splitting_mev", "meV", label="Largest spin splitting near E<sub>F</sub>",
                         precision=4,
                         description="largest |E_alpha - E_beta| of a band (by index) within 1 eV of E_F at any "
                                     "k of the path")))
def _band_spin_splitting(ctx):
    """Exchange splitting of the band edges (spin-polarized runs)."""
    r = _run(ctx)
    if not r.get("spin_polarized"):
        raise Skip("a closed-shell run (one spin channel)")
    sp = r.get("spin_splitting") or {}
    return {"vbm_spin_splitting_mev": sp.get("vbm_mev"), "cbm_spin_splitting_mev": sp.get("cbm_mev"),
            "band_max_spin_splitting_mev": sp.get("max_near_fermi_mev")}


@derived("fermi_velocity", calc_types=("BAND",), category="band_structure", group="Fermi velocity",
         siblings=_SIB, version="1", source=_SRC,
         outputs=(Output("fermi_velocity_mean_m_s", "m/s", label="v<sub>F</sub> (mean)", precision=4,
                         description="mean |dE/dk|/ħ over the points where bands cross E_F along the path"),
                  Output("fermi_velocity_min_m_s", "m/s", label="v<sub>F</sub> (min)", precision=4,
                         description="smallest v_F of the E_F crossings"),
                  Output("fermi_velocity_max_m_s", "m/s", label="v<sub>F</sub> (max)", precision=4,
                         description="largest v_F of the E_F crossings"),
                  Output("fermi_level_crossings", "dimensionless", "int", label="E<sub>F</sub> crossings (path)",
                         precision=6, description="points where a band crosses or touches E_F along the path "
                                                  "(per spin channel of a magnetic run)"),
                  Output("fermi_velocity_crossings", "JSON", "json", blob=True, label="v<sub>F</sub> per crossing",
                         description="each E_F crossing: spin, band, segment, k label and fractional k, v_F")))
def _fermi_velocity(ctx):
    """Fermi velocities of a metal or semimetal along the band path."""
    r = _run(ctx)
    if not r.get("gapless"):
        raise Skip("not a metal along the path (no band crosses E_F)")
    cr = r.get("fermi_crossings") or []
    vs = [c["v_m_s"] for c in cr if math.isfinite(c["v_m_s"])]
    if not vs:
        raise Skip("no band crosses E_F between two path points")
    return {"fermi_velocity_mean_m_s": sum(vs) / len(vs), "fermi_velocity_min_m_s": min(vs),
            "fermi_velocity_max_m_s": max(vs), "fermi_level_crossings": len(cr),
            "fermi_velocity_crossings": {"unit": "m/s", "crossings": cr}}
