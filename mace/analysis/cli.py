"""`mace analyze <tool> ...`: the structure analysis tools and the results of the
mechanical-property and phonopy runs on the command line.

    mace analyze porosity FILE [--probe N2] [--engine native|zeopp|both]
    mace analyze pxrd FILE [--source CuKa] [--range 5 90] [--out pattern.xy]
    mace analyze peaks PATTERN
    mace analyze index PATTERN [--structure FILE | --cell a b c alpha beta gamma]
    mace analyze refine PATTERN --structure FILE [--method pawley|rietveld]
    mace analyze topology FILE [--mode auto|atoms|sbu]
    mace analyze rcsr status | import 3dall.txt
    mace analyze polyhedra FILE
    mace analyze bonds FILE [--rdf]
    mace analyze morphology FILE [--energies "100:1.2,111:0.9"]
    mace analyze planes FILE --hkl 1 1 1 | --cleavage
    mace analyze zeopp
    mace analyze elastic RUN.out [--json] [--plot NAME] [--compare OUT]   (mace.elastic.cli)
    mace analyze phonopy DIR [--nac-from FREQ.out] [--band ...] [--check]  (mace.phonopy_bridge.cli)

Every tool takes --json for machine-readable output. The structure tools
(porosity, pxrd, topology, polyhedra, bonds, morphology, planes) take several
files too: one line per structure, and a combined summary with --csv PATH
(one row per structure) or --json (a list of {file, tool, params, result}).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

TOOLS = ("porosity", "pxrd", "peaks", "index", "refine", "topology", "rcsr", "polyhedra", "bonds", "morphology",
         "planes", "zeopp", "elastic", "phonopy")


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(type(o).__name__)


def _dump(obj) -> None:
    print(json.dumps(obj, indent=1, default=_default))


def _load(path, frame=-1):
    from .structure import load
    return load(path, frame)


def _f(x, n=3):
    return "-" if x is None else f"{x:.{n}f}"


# ---------------------------------------------------------------------------
def _porosity(a) -> int:
    from . import porosity, zeopp
    st = _load(a.file, a.frame)
    over = {}
    for item in a.radius or []:
        k, _, v = item.partition("=")
        over[k.strip()] = float(v)
    probes = list(porosity.PROBES) if a.probe == "all" else [a.probe]
    results = []
    for pr in probes:
        r = {}
        if a.engine in ("native", "both"):
            r["native"] = porosity.analyze(st, probe=pr, radii=a.radii, radius_overrides=over, spacing=a.spacing,
                                           render=bool(a.cube))
            if a.cube:
                _write_cubes(st, r["native"], a.cube if len(probes) == 1 else f"{a.cube}_{pr}")
                r["native"].pop("render", None)
        if a.engine in ("zeopp", "both"):
            if not zeopp.find():
                if a.engine == "zeopp":
                    print(zeopp.INSTALL_HINT, file=sys.stderr)
                    return 2
                r["zeopp_error"] = zeopp.INSTALL_HINT
            else:
                r["zeopp"] = zeopp.analyze(st, probe=pr, radii=a.radii, radius_overrides=over)
        results.append(r)
    if a.json:
        _dump(results if len(results) > 1 else results[0])
        return 0
    print(f"{st.name}: {st.formula()}  cell {st.volume:.2f} A^3  density {st.density:.4f} g/cm^3  radii {a.radii}")
    for r in results:
        for eng in ("native", "zeopp"):
            x = r.get(eng)
            if not x:
                continue
            p = x["probe"]
            print(f"\n[{eng}] probe {p['name']} (r = {p['radius']:.3f} A)")
            print(f"  Di {x['Di']:.3f}  Df {x['Df']:.3f}  Dif {x['Dif']:.3f} A   Df along a/b/c "
                  + " / ".join(_f(v) for v in x["Df_axes"]))
            for key, lab in (("AV", "accessible volume (probe center)"), ("NAV", "non-accessible volume (probe center)"),
                             ("POAV", "probe-occupiable accessible volume"), ("PONAV", "probe-occupiable non-accessible")):
                v = x[key]
                print(f"  {lab:38s} {_f(v['A3'], 2):>10} A^3  fraction {_f(v['fraction'], 4)}  {_f(v['cm3_g'], 4)} cm^3/g")
            for key, lab in (("ASA", "accessible surface area"), ("NASA", "non-accessible surface area")):
                v = x[key]
                print(f"  {lab:38s} {_f(v['A2'], 2):>10} A^2  {_f(v['m2_cm3'], 1)} m^2/cm^3  {_f(v['m2_g'], 1)} m^2/g")
            if eng == "native":
                print(f"  geometric void fraction (probe 0)      {x['void_fraction_geometric']:.4f}")
                print(f"  channels {x['n_channels']} (dimensionality {x['channel_dimensionality']}), pockets {x['n_pockets']}"
                      f"   grid {'x'.join(map(str, x['grid']['shape']))} at {x['grid']['spacing']:.3f} A")
                for w in x.get("warnings", []):
                    print(f"  note: {w}")
        if r.get("zeopp_error"):
            print(f"\n[zeo++] {r['zeopp_error']}")
    return 0


def _write_cubes(st, res, prefix) -> None:
    import base64
    rd = res["render"]
    shape = rd["shape"]
    from mace.constants import BOHR_TO_ANGSTROM as bohr
    for key in ("accessible", "inaccessible"):
        arr = np.frombuffer(base64.b64decode(rd[key]), "<f4").reshape(shape)[:-1, :-1, :-1]
        n = arr.shape
        lines = [f"MACE pore field ({key}): >0 where the probe sphere fits", f"probe {res['probe']['name']}, level 0",
                 f"{st.n:5d} " + " ".join(f"{x / bohr:12.6f}" for x in rd["origin"])]
        for i in range(3):
            v = np.asarray(rd["voxels"][i]) / bohr
            lines.append(f"{n[i]:5d} {v[0]:12.6f} {v[1]:12.6f} {v[2]:12.6f}")
        # the atoms where the grid has them (a slab's or molecule's box has its own origin)
        for z, c in zip(st.numbers, (st.cart + np.asarray(st.meta.get("origin", (0.0, 0.0, 0.0)), float)) / bohr):
            lines.append(f"{int(z):5d} {0.0:12.6f} {c[0]:12.6f} {c[1]:12.6f} {c[2]:12.6f}")
        flat = arr.reshape(-1)
        body = "\n".join(" ".join(f"{x:13.5E}" for x in flat[i:i + 6]) for i in range(0, len(flat), 6))
        Path(f"{prefix}_{key}.cube").write_text("\n".join(lines) + "\n" + body + "\n")
        print(f"wrote {prefix}_{key}.cube")


def _range(a):
    return (float(a.range[0]), float(a.range[1])) if a.range else (5.0, 90.0)


def _pxrd(a) -> int:
    from . import pxrd
    st = _load(a.file, a.frame)
    r = pxrd.simulate(st, a.source, _range(a), a.step, a.b_iso, a.U, a.V, a.W, a.eta, a.size_nm)
    if a.no_standardize:
        r["peaks"] = pxrd.reflections(st, a.source, _range(a), a.b_iso, standardize=False)
    if a.out:
        Path(a.out).write_text("# 2theta intensity (MACE simulated, %s)\n" % r["source"]
                               + "\n".join(f"{x:.4f} {y:.4f}" for x, y in zip(r["two_theta"], r["intensity"])) + "\n")
    if a.plot:
        _plot_pattern(a.plot, r["two_theta"], r["intensity"], None, None, r["peaks"], f"{st.name} ({r['source']})")
    if a.json:
        _dump(r)
        return 0
    print(f"{st.name}: {st.formula()}  {r['source']} ({r['wavelength']:.6f} A)  2theta {r['range'][0]}-{r['range'][1]}")
    print(f"{'2theta':>9} {'d (A)':>9} {'I':>7}  {'hkl':12s} mult")
    for p in r["peaks"]:
        if p["intensity"] >= a.min_intensity:
            print(f"{p['two_theta']:9.3f} {p['d']:9.4f} {p['intensity']:7.1f}  {str(tuple(p['hkl'])):12s} {p['multiplicity']}")
    if a.out:
        print(f"wrote {a.out}")
    return 0


def _plot_pattern(path, x, y, yc, diff, peaks, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, lw=0.8, color="k", label="observed" if yc is not None else "simulated")
    if yc is not None:
        ax.plot(x, yc, lw=0.8, color="tab:red", label="calculated")
        off = min(np.min(y), np.min(yc)) - 0.1 * (np.max(y) - np.min(y))
        ax.plot(x, np.asarray(diff) + off, lw=0.6, color="tab:blue", label="difference")
    if peaks:
        top = np.max(y)
        for p in peaks:
            ax.axvline(p["two_theta"], ymax=0.03, color="tab:green", lw=0.8)
    ax.set_xlabel("2θ (°)")              # plot_labels.with_unit("2θ", "deg"), as the GUI's PXRD chart
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def _peaks(a) -> int:
    from . import pxrd
    pat = pxrd.read_pattern(a.pattern)
    lam = pxrd.source(a.source)[0]
    r = pxrd.find_peaks(pat["two_theta"], pat["intensity"], prominence=a.prominence)
    rows = [{**p, "d": float(pxrd.d_spacing(p["two_theta"], lam))} for p in r["peaks"]]
    if a.json:
        _dump({"peaks": rows, "source": a.source})
        return 0
    print(f"{len(rows)} peaks in {Path(a.pattern).name} ({a.source})")
    print(f"{'2theta':>9} {'d (A)':>9} {'height':>10} {'FWHM':>7}")
    for p in sorted(rows, key=lambda p: p["two_theta"]):
        print(f"{p['two_theta']:9.3f} {p['d']:9.4f} {p['height']:10.1f} {_f(p.get('fwhm')):>7}")
    return 0


def _cell_arg(vals):
    if not vals:
        return None
    from ase.geometry import cellpar_to_cell
    return cellpar_to_cell([float(v) for v in vals])


def _index(a) -> int:
    from . import indexing, pxrd
    if a.lines:
        tt = [float(x) for x in a.lines.split(",")]
    else:
        pat = pxrd.read_pattern(a.pattern)
        found = pxrd.find_peaks(pat["two_theta"], pat["intensity"], prominence=a.prominence)["peaks"]
        tt = [p["two_theta"] for p in pxrd.strip_ka2(found, a.source)]
    if a.structure or a.cell:
        st = _load(a.structure) if a.structure else None
        r = indexing.index_with_cell(tt, _cell_arg(a.cell), a.source, structure=st, tol_deg=a.tol)
        if a.json:
            _dump(r)
            return 0
        L = r["lattice"]
        print(f"indexed {r['indexed']} of {r['total']} lines; refined cell a {L['a']:.5f} b {L['b']:.5f} c {L['c']:.5f} "
              f"alpha {L['alpha']:.3f} beta {L['beta']:.3f} gamma {L['gamma']:.3f}; M20 {r['figures_of_merit']['M20']:.1f}")
        for ln in r["lines"]:
            print(f"  {ln['two_theta']:9.3f}  {str(tuple(ln['hkl'])) if ln['hkl'] else 'not indexed':14s} {_f(ln['delta'], 4)}")
        return 0
    r = indexing.index_ab_initio(tt, a.source, systems=tuple(a.systems), tol_deg=a.tol)
    if a.json:
        _dump(r)
        return 0
    if not r["solutions"]:
        print("no cell found (cubic, tetragonal, hexagonal and orthorhombic were tried)")
        return 1
    print(f"{len(tt)} lines; best cells (M20, lines indexed):")
    for s in r["solutions"]:
        L = s["lattice"]
        print(f"  {s['system']:13s} {s['centering']}  a {L['a']:.5f} b {L['b']:.5f} c {L['c']:.5f}  "
              f"alpha {L['alpha']:.2f} beta {L['beta']:.2f} gamma {L['gamma']:.2f}  V {s['volume']:.2f}  "
              f"M20 {s['M20']:.1f}  {s['indexed']}/{s['total']}")
    print(r["note"])
    return 0


def _refine(a) -> int:
    from . import pxrd, refine
    pat = pxrd.read_pattern(a.pattern)
    st = _load(a.structure) if a.structure else None
    excl = []
    for e in a.exclude or []:
        lo, _, hi = e.partition("-")
        excl.append((float(lo), float(hi)))
    r = refine.fit(pat, a.method, structure=st, cell=_cell_arg(a.cell), wavelength=a.source,
                   two_theta_range=tuple(a.range) if a.range else None, exclude=excl, n_background=a.background,
                   refine_positions=a.positions, preferred_orientation=a.po, lattice_symmetry="triclinic" if a.triclinic else None)
    if a.out:
        Path(a.out).write_text("2theta,observed,calculated,background,difference\n" + "\n".join(
            f"{x:.4f},{o:.3f},{c:.3f},{b:.3f},{d:.3f}" for x, o, c, b, d in
            zip(r["two_theta"], r["observed"], r["calculated"], r["background"], r["difference"])) + "\n")
    if a.plot:
        _plot_pattern(a.plot, r["two_theta"], r["observed"], r["calculated"], r["difference"], r["reflections"],
                      f"{a.method.capitalize()} fit, Rwp {100 * r['statistics']['Rwp']:.2f}%")
    if a.json:
        _dump(r)
        return 0
    L, E, S = r["lattice"], r["lattice_esd"], r["statistics"]
    print(f"{a.method.capitalize()} fit of {Path(a.pattern).name} ({r['source']}): "
          f"{'converged' if r['converged'] else 'not converged'} after {r['nfev']} evaluations")
    for k in ("a", "b", "c", "alpha", "beta", "gamma", "volume"):
        e = E.get(k)
        print(f"  {k:6s} {L[k]:12.6f}" + (f" ({e:.6f})" if e and e > 0 else ""))
    print(f"  zero {r['zero']:.4f} deg   U {r['profile']['U']:.5f} V {r['profile']['V']:.5f} W {r['profile']['W']:.5f} "
          f"eta {r['profile']['eta']:.3f}")
    if a.method == "rietveld":
        print(f"  scale {r['scale']:.5g}  B {r['B']:.3f} A^2")
        if "frac" in r:
            for s, f in zip(r["symbols"], r["frac"]):
                print(f"  {s:3s} {f[0]:9.5f} {f[1]:9.5f} {f[2]:9.5f}")
    print(f"  Rp {100 * S['Rp']:.2f}%  Rwp {100 * S['Rwp']:.2f}%  Rexp {100 * S['Rexp']:.2f}%  GOF {S['GOF']:.3f}")
    for p in (a.out, ):
        if p:
            print(f"wrote {p}")
    return 0


def _topology(a) -> int:
    from . import topology
    st = _load(a.file, a.frame)
    r = topology.analyze(st, a.mode, a.tolerance)
    if a.json:
        _dump(r)
        return 0
    s = r["summary"]
    print(f"{st.name}: {st.formula()}  simplification {r['mode']}  net: {r['vertices']} vertices, {r['edges']} edges")
    info = r["simplification"]
    if info.get("molecules_dropped"):
        print(f"  {info['molecules_dropped']} molecule(s) not bonded to the framework left out")
    for i, n in enumerate(r["nets"]):
        name = n["topology"] or "not in the reference set"
        alt = ", ".join(m["name"] for m in n["matches"][1:])
        print(f"  component {i + 1}: {n['periodicity']}-periodic, topology {name}" + (f" (also {alt})" if alt else "")
              + (f", splits into {n['copies']} copies" if n.get("copies") and n["copies"] > 1 else "")
              + f", td10 {n['td10']:.1f}")
        for c in n["vertex_classes"]:
            print(f"    {c['count']:3d} x {c['cn']}-c  CS {' '.join(map(str, c['cs']))}  point symbol {{{c.get('point_symbol', '')}}}"
                  f"  ({', '.join(c['labels'][:4])})")
    ip = s["interpenetration"]
    print(f"  interpenetration: {ip if ip is not None else 'n/a (no 3-periodic net)'}" + (f"   {s['note']}" if s.get("note") else ""))
    from . import rcsr
    if not rcsr.imported() and not rcsr.bundled():
        print("  (only the built-in nets are known: `mace analyze rcsr import 3dall.txt` adds the RCSR list)")
    print(f"  net symbols and reference nets from RCSR; please cite {rcsr.CITATION}")
    return 0


def _rcsr(a) -> int:
    from . import rcsr
    if a.action == "import":
        if not a.path:
            print("usage: mace analyze rcsr import FILE   (RCSR 3dall.txt from http://rcsr.net/data/3dall.txt, or a .cgd)")
            return 2
        r = rcsr.import_file(a.path)
        print(f"imported {r['imported']} nets from {r['source']} into {r['path']}") if not a.json else _dump(r)
        return 0
    st = rcsr.status()
    if a.json:
        _dump(st)
        return 0
    print(f"built-in nets ({len(st['builtin'])}): {', '.join(st['builtin'])}")
    print(f"bundled RCSR nets: {st['bundled']}" + (f" ({st['bundled_source']})" if st["bundled"] else ""))
    print(f"imported nets: {st['imported']}" + (f" ({st['imported_path']}; used before the bundled list)" if st["imported"]
                                                else f"; a newer {st['download']} can be added with "
                                                     "`mace analyze rcsr import 3dall.txt`"))
    print(f"please cite RCSR: {st['citation']}")
    return 0


def _polyhedra(a) -> int:
    from . import polyhedra
    st = _load(a.file, a.frame)
    r = polyhedra.analyze(st, a.tolerance)
    if a.json:
        _dump(r)
        return 0
    if not r["polyhedra"]:
        print("no coordination polyhedra (no cation with 4-12 anion ligands)")
        return 0
    print(f"{'atom':>8} {'CN':>3} {'shape':22s} {'CShM':>6} {'<d> A':>7} {'V A^3':>8} {'Delta':>8} {'<lambda>':>9} {'sigma^2':>8} {'ECoN':>6}")
    for p in r["polyhedra"]:
        print(f"{p['element'] + str(p['index'] + 1):>8} {p['cn']:3d} {str(p.get('shape', '-')):22s} {_f(p.get('cshm'), 2):>6} "
              f"{p['bond_mean']:7.4f} {_f(p['volume'], 3):>8} {p['distortion_index']:8.5f} {_f(p.get('quadratic_elongation'), 4):>9} "
              f"{_f(p.get('angle_variance'), 2):>8} {p['econ']:6.3f}")
    return 0


def _bonds(a) -> int:
    from . import geometry
    st = _load(a.file, a.frame)
    if a.rdf:
        r = geometry.rdf(st, a.rmax, pairs=a.pair)
        if a.json:
            _dump(r)
            return 0
        for x, g in zip(r["r"], r["g"]):
            print(f"{x:7.3f} {g:9.4f}")
        return 0
    r = geometry.analyze(st, a.tolerance, a.bins)
    if a.json:
        _dump(r)
        return 0
    print(f"{r['n_bonds']} bonds, {r['n_angles']} angles (tolerance {a.tolerance})")
    print(f"{'bond':10s} {'n':>5} {'min':>8} {'mean':>8} {'max':>8} {'std':>7}")
    for k, v in r["bonds"].items():
        print(f"{k:10s} {v['n']:5d} {v['min']:8.4f} {v['mean']:8.4f} {v['max']:8.4f} {v['std']:7.4f}")
    print(f"{'angle':14s} {'n':>5} {'min':>8} {'mean':>8} {'max':>8} {'std':>7}")
    for k, v in r["angles"].items():
        print(f"{k:14s} {v['n']:5d} {v['min']:8.2f} {v['mean']:8.2f} {v['max']:8.2f} {v['std']:7.2f}")
    return 0


def _morphology(a) -> int:
    from . import morphology
    st = _load(a.file, a.frame)
    en = morphology.parse_energies(a.energies) if a.energies else None
    r = morphology.wulff(st, en, a.max_index)
    if a.obj:
        lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in r["vertices"]]
        lines += [f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}" for t in r["triangles"]]
        Path(a.obj).write_text("# MACE crystal morphology\n" + "\n".join(lines) + "\n")
    if a.json:
        _dump(r)
        return 0
    print(f"{st.name}: {r['method'].upper()} morphology  aspect {' : '.join(f'{x:.2f}' for x in r['aspect'])}")
    print(f"{'(hkl)':12s} {'mult':>4} {'area %':>8} {'distance':>9}")
    for f in r["families"]:
        print(f"{str(tuple(f['hkl'])):12s} {f['multiplicity']:4d} {100 * f['area_fraction']:8.2f} {f['distance']:9.4f}")
    if a.obj:
        print(f"wrote {a.obj}")
    return 0


def _planes(a) -> int:
    from . import planes
    st = _load(a.file, a.frame)
    if a.cleavage or not a.hkl:
        r = planes.cleavage(st, a.max_index, a.tolerance)
        if a.json:
            _dump(r)
            return 0
        print(f"{st.name}: bonds broken per nm^2 at the best cut ({r['model']})")
        print(f"{'(hkl)':12s} {'d (A)':>8} {'bonds/nm^2':>11} {'at offset':>9}")
        for f in r["families"]:
            print(f"{str(tuple(f['hkl'])):12s} {f['d']:8.4f} {f['min_per_nm2']:11.3f} {f['best_offset']:9.3f}")
        return 0
    r = _plane(st, a)
    hkl, d, poly, on, bb = r["hkl"], r["d"], r["planes"], r["atoms_on_plane"], r
    if a.json:
        _dump(r)
        return 0
    sym = st.symbols
    print(f"({' '.join(map(str, hkl))}) d = {d:.5f} A; {len(poly)} plane(s) at offset {a.offset} cross the cell")
    print(f"  atoms within {a.tol} A of the plane: {', '.join(sym[i] + str(i + 1) for i in on) or 'none'}")
    print(f"  bonds broken here {r['broken_bonds_per_nm2']:.3f}/nm^2; fewest {bb['min_per_nm2']:.3f}/nm^2 at offset {bb['best_offset']:.3f}")
    return 0


def _plane(st, a):
    from . import planes
    hkl = [int(x) for x in a.hkl]
    d = planes.d_spacing(st.cell, hkl)
    bb = planes.broken_bonds(st, hkl, tolerance=a.tolerance)
    poly = planes.planes_in_box(st.cell, hkl, a.offset)
    on = planes.atoms_on_plane(st, hkl, a.offset, a.tol)
    return {"hkl": hkl, "d": d, "offset": a.offset, "planes": poly, "atoms_on_plane": on,
            "broken_bonds_per_nm2": bb["profile"][int(round((a.offset % 1) * len(bb["profile"]))) % len(bb["profile"])] if bb["profile"] else 0.0,
            "best_offset": bb["best_offset"], "min_per_nm2": bb["min_per_nm2"]}


def _zeopp(a) -> int:
    from . import zeopp
    p = zeopp.find()
    if a.json:
        _dump({"found": bool(p), "binary": p, "hint": None if p else zeopp.INSTALL_HINT})
        return 0
    print(f"Zeo++: {p}" if p else zeopp.INSTALL_HINT)
    return 0 if p else 1


# ---------------------------------------------------------------------------
# several structures: `mace analyze <tool> A.cif B.cif ...` (and the GUI's batch summary)
# ---------------------------------------------------------------------------
MULTI = ("porosity", "pxrd", "topology", "polyhedra", "bonds", "morphology", "planes")
# options that write one file or need one structure: refused with several
ONE_FILE = {"porosity": (("cube", "--cube"),), "pxrd": (("out", "--out"), ("plot", "--plot")),
            "bonds": (("rdf", "--rdf"),), "morphology": (("obj", "--obj"),)}


def _num(x, n=4):
    return None if x is None else round(float(x), n)


def summary_row(tool: str, result: dict, file: str, params: Optional[dict] = None) -> dict:
    """One structure's key numbers for a summary table (CSV): the same
    columns for `mace analyze <tool> FILE...` and the GUI's batch runs."""
    r, p = result or {}, params or {}
    row = {"file": file}
    if tool == "porosity":
        pr, g = r.get("probe") or {}, r.get("grid") or {}
        row.update({"probe": pr.get("name"), "probe_radius": _num(pr.get("radius")), "radii": r.get("radii"),
                    "spacing": _num(g.get("spacing")), "Di": _num(r.get("Di")), "Df": _num(r.get("Df")),
                    "Dif": _num(r.get("Dif")), "AV_fraction": _num((r.get("AV") or {}).get("fraction"), 5),
                    "POAV_A3": _num((r.get("POAV") or {}).get("A3"), 2), "POAV_cm3_g": _num((r.get("POAV") or {}).get("cm3_g"), 5),
                    "ASA_m2_g": _num((r.get("ASA") or {}).get("m2_g"), 2), "NASA_m2_g": _num((r.get("NASA") or {}).get("m2_g"), 2),
                    "density": _num(r.get("density_g_cm3"), 5), "n_channels": r.get("n_channels"),
                    "channel_dim": r.get("channel_dimensionality"), "n_pockets": r.get("n_pockets")})
    elif tool == "pxrd":
        floor = float(p.get("min_intensity", 0.5) or 0.0)
        peaks = [x for x in r.get("peaks") or [] if x.get("intensity", 0) >= floor]
        top = max(peaks, key=lambda x: x["intensity"]) if peaks else None
        row.update({"source": r.get("source"), "wavelength": _num(r.get("wavelength"), 6), "n_peaks": len(peaks),
                    "strongest_two_theta": _num(top["two_theta"]) if top else None,
                    "strongest_d": _num(top["d"]) if top else None,
                    "strongest_hkl": " ".join(map(str, top["hkl"])) if top and top.get("hkl") else None})
    elif tool == "topology":
        s = r.get("summary") or {}
        row.update({"mode": r.get("mode"), "topology": ", ".join(s.get("topologies") or []) or None,
                    "interpenetration": s.get("interpenetration"), "vertices": r.get("vertices"), "edges": r.get("edges")})
    elif tool == "polyhedra":
        row.update({"n_polyhedra": len(r.get("polyhedra") or []),
                    "shapes": "; ".join(f"{x['element']} CN{x['cn']} {x.get('shape') or '-'} x{x['count']}"
                                        for x in r.get("summary") or []) or None})
    elif tool == "bonds":
        row.update({"n_bonds": r.get("n_bonds"), "n_angles": r.get("n_angles")})
    elif tool == "morphology":
        fam = sorted(r.get("families") or [], key=lambda f: -f.get("area_fraction", 0))
        row.update({"method": r.get("method"), "aspect": ":".join(f"{x:.2f}" for x in r.get("aspect") or []) or None,
                    "top_facet": " ".join(map(str, fam[0]["hkl"])) if fam else None,
                    "top_facet_area_pct": _num(100 * fam[0]["area_fraction"], 2) if fam else None})
    elif tool in ("planes", "cleavage"):
        if r.get("families") is not None:           # a cleavage ranking: the best family
            fam = sorted(r["families"], key=lambda f: f.get("min_per_nm2", float("inf")))
            f = fam[0] if fam else {}
            row.update({"hkl": " ".join(map(str, f["hkl"])) if f else None, "d": _num(f.get("d")),
                        "min_bonds_per_nm2": _num(f.get("min_per_nm2")), "best_offset": _num(f.get("best_offset"))})
        else:
            bb = r.get("broken") or r
            row.update({"hkl": " ".join(map(str, r.get("hkl") or [])) or None, "d": _num(r.get("d")),
                        "min_bonds_per_nm2": _num(bb.get("min_per_nm2")), "best_offset": _num(bb.get("best_offset"))})
    return row


def _params(a) -> dict:
    keys = {"porosity": ("probe", "radii", "radius", "spacing"),
            "pxrd": ("source", "range", "step", "b_iso", "U", "V", "W", "eta", "size_nm", "min_intensity", "no_standardize"),
            "topology": ("mode", "tolerance"), "polyhedra": ("tolerance",), "bonds": ("tolerance", "bins"),
            "morphology": ("energies", "max_index"),
            "planes": ("hkl", "offset", "tol", "cleavage", "max_index", "tolerance")}[a.tool]
    out = {k: getattr(a, k, None) for k in keys}
    out["frame"] = a.frame
    return out


def _result(a, st) -> dict:
    """One structure's result for a run over several files (what --json prints for one file)."""
    from . import geometry, morphology, planes, polyhedra, porosity, pxrd, topology
    t = a.tool
    if t == "porosity":
        over = {k.strip(): float(v) for k, _, v in (x.partition("=") for x in a.radius or [])}
        return porosity.analyze(st, probe=a.probe, radii=a.radii, radius_overrides=over, spacing=a.spacing)
    if t == "pxrd":
        r = pxrd.simulate(st, a.source, _range(a), a.step, a.b_iso, a.U, a.V, a.W, a.eta, a.size_nm)
        if a.no_standardize:
            r["peaks"] = pxrd.reflections(st, a.source, _range(a), a.b_iso, standardize=False)
        return r
    if t == "topology":
        return topology.analyze(st, a.mode, a.tolerance)
    if t == "polyhedra":
        return polyhedra.analyze(st, a.tolerance)
    if t == "bonds":
        return geometry.analyze(st, a.tolerance, a.bins)
    if t == "morphology":
        return morphology.wulff(st, morphology.parse_energies(a.energies) if a.energies else None, a.max_index)
    if a.cleavage or not a.hkl:
        return planes.cleavage(st, a.max_index, a.tolerance)
    return _plane(st, a)


def _fmt_cell(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _many(a) -> int:
    """The tool on each file in turn: a line per structure, then the combined summary."""
    import csv
    for attr, flag in ONE_FILE.get(a.tool, ()):
        if getattr(a, attr, None):
            print(f"mace analyze {a.tool}: {flag} takes one structure file, not {len(a.file)}", file=sys.stderr)
            return 2
    if a.tool == "porosity" and (a.probe == "all" or a.engine != "native"):
        print(f"mace analyze porosity: --probe all and --engine zeopp|both take one structure file, not {len(a.file)}",
              file=sys.stderr)
        return 2
    params = _params(a)
    runs, rows, failed = [], [], 0
    for path in a.file:
        try:
            r = _result(a, _load(path, a.frame))
            runs.append({"file": path, "tool": a.tool, "params": params, "result": r})
            rows.append(summary_row(a.tool, r, path, params))
            if not a.json:
                row = rows[-1]
                print(f"{path}: " + "  ".join(f"{k} {_fmt_cell(v)}" for k, v in row.items() if k != "file"), flush=True)
        except Exception as exc:                  # one structure failing does not stop the others
            failed += 1
            msg = str(exc).strip() or type(exc).__name__
            runs.append({"file": path, "tool": a.tool, "params": params, "error": msg})
            rows.append({"file": path, "error": msg})
            print(f"{path}: error: {msg}", file=sys.stderr, flush=True)
    if a.csv:
        cols: List[str] = []
        for row in rows:
            cols += [k for k in row if k not in cols and k != "error"]
        if failed:
            cols.append("error")
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in cols})
        if not a.json:
            print(f"wrote {a.csv} ({len(rows)} structures)")
    if a.json:
        _dump(runs)
    elif failed:
        print(f"{failed} of {len(a.file)} structures failed", file=sys.stderr)
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mace analyze", description="Structure analysis tools", allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="tool", required=True)

    def common(p, file=True):
        if file:
            p.add_argument("file", nargs="+", help="structure file(s) (CIF, D12, CRYSTAL output, POSCAR, XYZ, ...)")
            p.add_argument("--frame", type=int, default=-1, help="step of a CRYSTAL output (default: last)")
            p.add_argument("--csv", metavar="PATH", help="write a summary CSV, one row per structure (several files)")
        p.add_argument("--json", action="store_true", help="machine-readable output")

    p = sub.add_parser("porosity", help="pore sizes, volumes, surface areas")
    common(p)
    p.add_argument("--probe", default="N2", help="N2, CO2, CH4, H2, H2O, Ar, He, all, or a radius in Angstrom")
    p.add_argument("--radii", default="ccdc", choices=("ccdc", "uff"))
    p.add_argument("--radius", action="append", metavar="EL=R", help="override one element's radius")
    p.add_argument("--spacing", type=float, default=0.15, help="grid spacing in Angstrom")
    p.add_argument("--engine", default="native", choices=("native", "zeopp", "both"))
    p.add_argument("--cube", metavar="PREFIX", help="write the accessible/inaccessible pore fields as .cube files")
    p.set_defaults(fn=_porosity)

    def src(p):
        p.add_argument("--source", default="CuKa", help="CuKa, CuKa1, MoKa, CoKa, CrKa, FeKa, AgKa or a wavelength")

    p = sub.add_parser("pxrd", help="simulated powder pattern")
    common(p)
    src(p)
    p.add_argument("--range", nargs=2, type=float, metavar=("MIN", "MAX"))
    p.add_argument("--step", type=float, default=0.02)
    p.add_argument("--b-iso", type=float, default=0.0, help="isotropic displacement B (A^2)")
    p.add_argument("--U", type=float, default=0.0)
    p.add_argument("--V", type=float, default=0.0)
    p.add_argument("--W", type=float, default=0.01, help="Caglioti W (FWHM^2 at 0, deg^2)")
    p.add_argument("--eta", type=float, default=0.5, help="pseudo-Voigt Lorentzian fraction")
    p.add_argument("--size-nm", type=float, help="crystallite size for Scherrer broadening")
    p.add_argument("--min-intensity", type=float, default=0.5)
    p.add_argument("--no-standardize", action="store_true", help="Miller indices of the cell as given")
    p.add_argument("--out", help="write the profile as 2theta intensity")
    p.add_argument("--plot", help="write a PNG")
    p.set_defaults(fn=_pxrd)

    p = sub.add_parser("peaks", help="peaks of a measured pattern")
    p.add_argument("pattern")
    common(p, file=False)
    src(p)
    p.add_argument("--prominence", type=float, default=0.02, help="relative to the strongest peak")
    p.set_defaults(fn=_peaks)

    p = sub.add_parser("index", help="Miller indices for measured lines")
    p.add_argument("pattern", nargs="?")
    common(p, file=False)
    src(p)
    p.add_argument("--lines", help="comma-separated 2theta values instead of a pattern")
    p.add_argument("--structure", help="index with this structure's cell and absences")
    p.add_argument("--cell", nargs=6, type=float, metavar=("a", "b", "c", "alpha", "beta", "gamma"))
    p.add_argument("--systems", nargs="+", default=["cubic", "tetragonal", "hexagonal", "orthorhombic"])
    p.add_argument("--tol", type=float, default=0.05, help="2theta tolerance (deg)")
    p.add_argument("--prominence", type=float, default=0.02)
    p.set_defaults(fn=_index)

    p = sub.add_parser("refine", help="Pawley or Rietveld fit of a measured pattern")
    p.add_argument("pattern")
    common(p, file=False)
    src(p)
    p.add_argument("--structure", help="structure model (required for Rietveld)")
    p.add_argument("--cell", nargs=6, type=float, metavar=("a", "b", "c", "alpha", "beta", "gamma"))
    p.add_argument("--method", default="pawley", choices=("pawley", "rietveld"))
    p.add_argument("--positions", action="store_true", help="Rietveld: refine atomic coordinates")
    p.add_argument("--po", nargs=3, type=int, metavar=("h", "k", "l"), help="March-Dollase preferred orientation")
    p.add_argument("--range", nargs=2, type=float, metavar=("MIN", "MAX"))
    p.add_argument("--exclude", action="append", metavar="LO-HI")
    p.add_argument("--background", type=int, default=6, help="Chebyshev terms")
    p.add_argument("--triclinic", action="store_true", help="refine all six cell parameters")
    p.add_argument("--out", help="write 2theta, observed, calculated, background, difference (CSV)")
    p.add_argument("--plot", help="write a PNG")
    p.set_defaults(fn=_refine)

    p = sub.add_parser("topology", help="underlying net, RCSR symbol, interpenetration")
    common(p)
    p.add_argument("--mode", default="auto", choices=("auto", "atoms", "sbu", "all"))
    p.add_argument("--tolerance", type=float, default=1.15, help="bond cutoff factor on covalent radii")
    p.set_defaults(fn=_topology)

    p = sub.add_parser("rcsr", help="the reference nets: status, or import a newer RCSR file")
    p.add_argument("action", choices=("status", "import"))
    p.add_argument("path", nargs="?")
    common(p, file=False)
    p.set_defaults(fn=_rcsr)

    p = sub.add_parser("polyhedra", help="coordination polyhedra and their distortion")
    common(p)
    p.add_argument("--tolerance", type=float, default=1.15)
    p.set_defaults(fn=_polyhedra)

    p = sub.add_parser("bonds", help="bond-length and angle distributions, or g(r)")
    common(p)
    p.add_argument("--tolerance", type=float, default=1.15)
    p.add_argument("--bins", type=int, default=40)
    p.add_argument("--rdf", action="store_true", help="radial distribution g(r) instead")
    p.add_argument("--rmax", type=float, default=10.0)
    p.add_argument("--pair", help="g(r) of one pair, e.g. Si-O")
    p.set_defaults(fn=_bonds)

    p = sub.add_parser("morphology", help="BFDH or Wulff crystal shape")
    common(p)
    p.add_argument("--energies", help='surface energies, e.g. "100:1.2,111:0.9" (Wulff); none = BFDH')
    p.add_argument("--max-index", type=int, default=2)
    p.add_argument("--obj", help="write the shape as a Wavefront .obj")
    p.set_defaults(fn=_morphology)

    p = sub.add_parser("planes", help="a Miller plane, or cleavage ranking")
    common(p)
    p.add_argument("--hkl", nargs=3, type=int)
    p.add_argument("--offset", type=float, default=0.0, help="plane position: h.x = offset (+ n)")
    p.add_argument("--tol", type=float, default=0.3, help="distance for atoms on the plane (A)")
    p.add_argument("--cleavage", action="store_true", help="rank low-index families by bonds broken")
    p.add_argument("--max-index", type=int, default=2)
    p.add_argument("--tolerance", type=float, default=1.15)
    p.set_defaults(fn=_planes)

    p = sub.add_parser("zeopp", help="is Zeo++ available")
    common(p, file=False)
    p.set_defaults(fn=_zeopp)

    # the results of the runs mace opt2d12 sets up (choices 4-8): their own options and reports
    from mace.elastic import cli as elastic_cli
    p = sub.add_parser("elastic", help="tensors, moduli and EOS fits of an ELASTIC, EOS, QHA or CPHF run",
                       formatter_class=argparse.RawDescriptionHelpFormatter, epilog=elastic_cli.EPILOG)
    elastic_cli.add_analyze_arguments(p)
    p.set_defaults(fn=elastic_cli.analyze)

    from mace.phonopy_bridge import cli as phonopy_cli
    p = sub.add_parser("phonopy", help="a phonopy set's forces -> force constants, bands, DOS, thermal properties",
                       formatter_class=argparse.RawDescriptionHelpFormatter, epilog=phonopy_cli.COLLECT_EPILOG)
    phonopy_cli.add_collect_arguments(p)
    p.set_defaults(fn=phonopy_cli.analyze)
    for q in sub.choices.values():
        q.allow_abbrev = False                          # full option names only
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    try:
        if isinstance(getattr(a, "file", None), list):
            if len(a.file) > 1 or a.csv:
                return _many(a)
            a.file = a.file[0]
        return int(a.fn(a) or 0)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"mace analyze {a.tool}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
