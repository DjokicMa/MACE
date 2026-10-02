"""``mace defects {build,potc,analyze}``.

build    pristine supercell + one deck per charge state (+ calibration decks)
         from a single-point host deck; submit them with ``mace submit DIR``
potc     stage POTC (electrostatic potential at the atoms) property runs for
         potential alignment into DIR/potc; submit with ``mace submit DIR/potc``
analyze  checks, calibration, formation energies and transition levels
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

from mace.defects import DefectError, NeedsPristineOutput
from mace.defects import deck_edit as de
from mace.defects import outputs as out

MANIFEST = "defects_manifest.json"
MANIFEST_VERSION = 1


def _p(msg: str = "") -> None:
    print(msg)


# ---------------------------------------------------------------------------
# argument helpers
# ---------------------------------------------------------------------------

def _ints(text: str, what: str) -> List[int]:
    vals = text.replace("x", " ").replace("X", " ").replace(",", " ").replace(";", " ").split()
    try:
        return [int(v) for v in vals]
    except ValueError:
        raise DefectError(f"{what} {text!r}: integers expected")


def _floats(text: str, what: str) -> List[float]:
    try:
        return [float(v) for v in text.replace(",", " ").split()]
    except ValueError:
        raise DefectError(f"{what} {text!r}: numbers expected")


def _diag(vals: List[int], dim: int) -> List[List[int]]:
    if len(vals) == 1:
        vals = vals * dim
    if len(vals) != dim:
        raise DefectError(f"a {dim}D host needs {dim} supercell factors, got {vals}")
    return [[vals[i] if i == j else 0 for j in range(dim)] for i in range(dim)]


def _supercells(args, dim: int) -> List[dict]:
    given = [x for x in (args.supercell, args.supercon, args.supercell_matrix, args.sizes) if x]
    if len(given) != 1:
        raise DefectError("give exactly one of --supercell, --supercon, --supercell-matrix, --sizes")
    if args.supercell:
        return [{"keyword": "SUPERCEL", "matrix": _diag(_ints(args.supercell, "--supercell"), dim)}]
    if args.supercon:
        return [{"keyword": "SUPERCON", "matrix": _diag(_ints(args.supercon, "--supercon"), dim)}]
    if args.supercell_matrix:
        vals = _ints(args.supercell_matrix, "--supercell-matrix")
        n = 3 if dim == 3 else 2
        if len(vals) != n * n:
            raise DefectError(f"--supercell-matrix needs {n * n} integers for a {dim}D host")
        return [{"keyword": "SUPERCEL", "matrix": [vals[i * n:(i + 1) * n] for i in range(n)]}]
    if dim != 2:
        raise DefectError("--sizes is for 2D hosts (the A + B/L fit); a 3D set takes one supercell")
    return [{"keyword": "SUPERCEL", "matrix": _diag([n], 2), "size": n}
            for n in _ints(args.sizes, "--sizes")]


def _defect(args) -> de.DefectSite:
    kinds = [k for k in ("vacancy", "ghost", "substitute", "insert") if getattr(args, k)]
    if len(kinds) != 1:
        raise DefectError("give exactly one of --vacancy, --ghost, --substitute, --insert")
    if args.vacancy:
        return de.DefectSite("vacancy", _el(args.vacancy))
    if args.ghost:
        return de.DefectSite("ghost", _el(args.ghost))
    if args.substitute:
        if "=" not in args.substitute:
            raise DefectError("--substitute takes OLD=NEW, e.g. O=N")
        old, new = args.substitute.split("=", 1)
        return de.DefectSite("substitution", _el(old), new_element=_el(new))
    if "@" not in args.insert:
        raise DefectError("--insert takes EL@x,y,z")
    el, xyz = args.insert.split("@", 1)
    coords = _floats(xyz, "--insert")
    if len(coords) != 3:
        raise DefectError("--insert needs three coordinates")
    site = de.DefectSite("interstitial", _el(el))
    site.cart = tuple(coords)  # resolved to Cartesian later
    return site


def _el(sym: str) -> str:
    z = de.element_number(sym)
    return de.element_symbol(z)


def _spin_states(items: List[str]) -> Dict[float, int]:
    out_ = {}
    for item in items or []:
        if "=" not in item:
            raise DefectError(f"--spin-state {item!r}: expected q=NSPIN, e.g. +1=1")
        q, n = item.split("=", 1)
        out_[float(q)] = int(n)
    return out_


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _anion(host: de.HostInfo) -> str:
    syms = [de.element_symbol(z) for z in host.elements]
    return max(syms, key=lambda s: (de.PAULING.get(s, 0.0), -de.element_number(s)))


def _pristine_text(path: Optional[Path]) -> Optional[str]:
    if path is not None and path.is_file():
        return path.read_text(errors="replace")
    return None


def build(args) -> int:
    from mace.defects import structure as st

    if args.resume:
        out_dir = Path(args.resume)
        manifest_path = out_dir / MANIFEST
        if not manifest_path.is_file():
            raise DefectError(f"{manifest_path} not found: --resume takes a directory written by "
                              "mace defects build")
        old = json.loads(manifest_path.read_text())
        saved = old["options"]
        for key, value in saved.items():
            setattr(args, key, value)
        host_text = old["host_text"]
        host_path = old["host"]
        args.resume = str(out_dir)
    else:
        if not args.host:
            raise DefectError("give the host deck (HOST.d12) or --resume DIR")
        host_path = args.host
        host_text = Path(args.host).read_text()
        out_dir = Path(args.output or (Path(args.host).stem + "_defects"))
    site = _defect(args)
    host, notes = de.classify_host(host_text, insert=site.kind == "interstitial", host_spin=args.host_spin)
    for n in notes:
        _p(n)
    _p(f"host: {host_path} - {host.dimensionality}D {host.geometry_kind}"
       f"{' + SLABCUT' if host.slabcut else ''}, {host.method} {host.functional or ''}, "
       f"basis {host.basis_name or 'explicit'}{', SPIN' if host.spin else ''}")
    supercells = _supercells(args, host.dimensionality)
    charges = sorted(set(_floats(args.charges, "--charges")))
    opts = de.SetOptions(
        charges=charges, spin_states=_spin_states(args.spin_state),
        spinlock_cycles=args.spinlock_cycles, host_spin=args.host_spin,
        shrink=_ints(args.shrink, "--shrink") if args.shrink else None,
        calibrate=(_floats(args.calibrate, "--calibrate") if args.calibrate else None),
        calibrate_on=args.calibrate_on, calibrate_endpoints=args.calibrate_endpoints, force=args.force)
    if args.calibrate_on == "pristine" and args.calibrate:
        _p("note: --calibrate-on pristine is untested as the calibration source of a defect set")
    if args.pristine_out and len(supercells) > 1:
        raise DefectError("--pristine-out is for a single supercell; with --sizes use --resume DIR")
    prefix = args.name or Path(host_path).stem
    site_frac = _floats(args.site_frac, "--site-frac") if args.site_frac else None
    if site_frac is not None and len(site_frac) != 3:
        raise DefectError("--site-frac needs three coordinates")
    if site.kind != "interstitial" and args.cart:
        raise DefectError("--cart is for --insert")

    all_decks: List[de.Deck] = []
    sc_entries = []
    pending = False
    site_out = None
    added_all: List[str] = []
    for sc in supercells:
        name = f"{prefix}{sc['size']}" if "size" in sc else prefix
        pris_job = f"{name}_pris"
        pris_path = Path(args.pristine_out) if args.pristine_out else out_dir / f"{pris_job}.out"
        pris_text = _pristine_text(pris_path)
        geom = st.parse_geometry(pris_text) if pris_text else None
        n_super = None
        if pris_text:
            n_super = out.electron_count(pris_text) or out.header_electrons(pris_text)
            n_super = int(round(n_super)) if n_super is not None else None
        mult = de.supercell_multiplicity(host, sc["keyword"], sc["matrix"])
        electrons = de.supercell_electrons(host, mult, n_super)
        sc_site = de.DefectSite(site.kind, site.element, site.new_element)
        reason = ""
        try:
            if site.kind == "interstitial":
                if geom is None:
                    raise NeedsPristineOutput("the inserted atom is checked against the supercell atoms")
                sc_site.cart = tuple(float(x) for x in st.insertion_point(
                    host, geom, site.cart, args.cart, sc["keyword"], sc["matrix"]))
                src = "Cartesian" if args.cart else "fractional (host cell)"
                site_out = {"label": None, "source": src, "cart": list(sc_site.cart)}
            else:
                label, src, snotes = st.resolve_site(host, geom, site.element, label=args.site_label,
                                                     frac=site_frac, keyword=sc["keyword"], matrix=sc["matrix"])
                for n in snotes:
                    _p(n)
                if label is None:
                    raise NeedsPristineOutput(src)
                sc_site.label = label
                site_out = {"label": label, "source": src}
                if site.kind == "ghost" and host.basis_kind == "explicit" and geom is not None:
                    anion = _anion(host)
                    sc_site.far_label = st.far_atom(geom, label, anion)
                    far_z = de.element_number(anion)
                    convs = [c for c in host.explicit_basis if c % 100 == far_z]
                    far_shells = host.explicit_basis[convs[0]] if len(convs) == 1 else None
                else:
                    far_shells = None
            if site.kind == "interstitial":
                far_shells = None
            decks, msgs, added = de.build_supercell_set(host, sc_site, sc["keyword"], sc["matrix"], opts,
                                                        electrons, name, far_shells=far_shells)
            status = "built"
        except NeedsPristineOutput as exc:
            reason = str(exc)
            no_q = de.SetOptions(charges=[], shrink=opts.shrink)
            decks, msgs, added = de.build_supercell_set(host, sc_site, sc["keyword"], sc["matrix"], no_q,
                                                        electrons, name)
            status = "awaiting_pristine"
            pending = True
        _p(f"\nsupercell {' '.join(de.supercell_records(host.dimensionality, sc['keyword'], sc['matrix']))}"
           f" ({mult} primitive cells): pristine {electrons.describe()} [{electrons.source}]")
        sug = de.suggest_shrink(host, sc["matrix"])
        if sug and not opts.shrink:
            _p(f"  SHRINK kept from the host; the equivalent k-point density for this supercell is "
               f"about {sug} (--shrink to change it)")
        if not host.has.get("BIPOSIZE") and not host.has.get("EXCHSIZE") and mult * len(host.atoms) > 5:
            _p("  note: the host has no BIPOSIZE/EXCHSIZE; large supercells may need them "
               "(MACE writes 110000000 above 5 atoms)")
        for m in msgs:
            _p("  " + m)
        if status != "built":
            _p(f"  only the pristine deck is written: {reason}")
        for a in added:
            if a not in added_all:
                added_all.append(a)
        all_decks += decks
        sc_entries.append({"keyword": sc["keyword"], "matrix": sc["matrix"], "size": sc.get("size"),
                           "multiplicity": mult, "status": status, "reason": reason,
                           "pristine_job": f"{name}_pris", "n_electrons": electrons.exact,
                           "parity": electrons.parity, "parity_source": electrons.source,
                           "jobs": [d.manifest() for d in decks]})

    delta_e = None
    involved = [site.element] + ([site.new_element] if site.new_element else [])
    if not any(de.has_ecp(host, de.element_number(e)) for e in involved):
        delta_e = site.delta_z
    manifest = {
        "version": MANIFEST_VERSION,
        "host": str(host_path),
        "host_sha256": de.host_sha256(host_text),
        "host_text": host_text,
        "dimensionality": host.dimensionality,
        "defect": {"kind": site.kind, "element": site.element, "new_element": site.new_element,
                   "cart": site_out.get("cart") if site_out else None, "delta_electrons": delta_e},
        "site": site_out or {"label": None, "source": "awaiting pristine output"},
        "status": "awaiting_pristine" if pending else "built",
        "charges": charges,
        "spin_states": {de.fmt_q(k): v for k, v in opts.spin_states.items()},
        "spinlock_cycles": opts.spinlock_cycles,
        "calibration": ({"ladder": de.calibration_ladder(charges, opts.calibrate), "on": opts.calibrate_on,
                         "restricted": True, "endpoints": opts.calibrate_endpoints,
                         "differences_from_host": "SPIN, SPINLOCK and UHF removed (restricted)"}
                        if opts.calibrate else None),
        "added_records": added_all,
        "supercells": sc_entries,
        "jobs": [dict(d.manifest(), supercell=i) for i, sc in enumerate(sc_entries) for d in
                 [x for x in all_decks if x.name in {j["name"] for j in sc["jobs"]}]],
        "options": {k: getattr(args, k) for k in _SAVED_OPTIONS},
    }
    new_jobs = _write(out_dir, all_decks, manifest, args.force)
    # absolute paths in the printed commands: `mace submit sub/dir/x.d12` with a
    # relative directory does not find the job script it generates
    shown = out_dir.resolve()
    _p(f"\nwrote {len(all_decks)} deck(s) and {MANIFEST} to {out_dir}/")
    if pending:
        _p(f"next: mace submit {shown}")
        _p(f"      then: mace defects build --resume {out_dir}")
        _p("      (--resume works as soon as the pristine .out shows its geometry tables - seconds "
           "after the job starts - no need to wait for the SCF)")
    elif args.resume and new_jobs:
        _p("next: submit the new decks (mace submit takes one target):")
        _p(f"      for j in {' '.join(new_jobs)}; do mace submit {shown}/$j.d12; done")
    else:
        _p(f"next: mace submit {shown}")
    _p(f"then: mace defects analyze {out_dir}")
    return 0


_SAVED_OPTIONS = ("vacancy", "ghost", "substitute", "insert", "cart", "site_label", "site_frac", "supercell",
                  "supercon", "supercell_matrix", "sizes", "charges", "spin_state", "spinlock_cycles",
                  "host_spin", "calibrate", "calibrate_on", "calibrate_endpoints", "pristine_out", "shrink",
                  "name")


def _write(out_dir: Path, decks: List[de.Deck], manifest: dict, force: bool) -> List[str]:
    """Write the decks and the manifest; return the names of decks that did not exist before."""
    conflicts = []
    for d in decks:
        path = out_dir / f"{d.name}.d12"
        if path.exists() and path.read_text() != d.text:
            conflicts.append(path.name)
    if conflicts and not force:
        raise DefectError(f"{out_dir} already has different decks: {', '.join(conflicts)} "
                          "(--force to overwrite)")
    out_dir.mkdir(parents=True, exist_ok=True)
    new = []
    for d in decks:
        path = out_dir / f"{d.name}.d12"
        if not path.exists():
            new.append(d.name)
        path.write_text(d.text)
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return new


# ---------------------------------------------------------------------------
# potc
# ---------------------------------------------------------------------------

def potc(args) -> int:
    d = Path(args.dir)
    manifest = _load_manifest(d)
    jobs = [j for sc in manifest["supercells"] for j in sc["jobs"] if j["role"] in ("pristine", "defect")]
    if args.jobs:
        want = set(args.jobs.split(","))
        jobs = [j for j in jobs if j["name"] in want]
    missing = [j["name"] for j in jobs if not (d / f"{j['name']}.f9").is_file()
               or (d / f"{j['name']}.f9").stat().st_size == 0]
    if missing:
        raise DefectError("no (or an empty) .f9 for: " + ", ".join(missing) +
                          " - POTC reads the converged wave function; run those jobs first")
    pdir = d / "potc"
    pdir.mkdir(exist_ok=True)
    planar = args.planar
    if planar and manifest["dimensionality"] != 2:
        raise DefectError("--planar is for 2D sets")
    for j in jobs:
        base = f"{j['name']}_potc"
        (pdir / f"{base}.d3").write_text("POTC\n0 0 0\nEND\n")
        shutil.copyfile(d / f"{j['name']}.f9", pdir / f"{base}.f9")
        if planar:
            zm, zp = planar
            basez = f"{j['name']}_potcz"
            (pdir / f"{basez}.d3").write_text(f"POTC\n2 41 0\n{zm} {zp}\nEND\n")
            shutil.copyfile(d / f"{j['name']}.f9", pdir / f"{basez}.f9")
    _p(f"staged {len(jobs)} POTC run(s) in {pdir}/ (the job outputs and .f9 files in {d} are untouched)")
    _p(f"next: mace submit {pdir.resolve()}")
    _p(f"then: mace defects analyze {d}")
    return 0


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

def _load_manifest(d: Path) -> dict:
    path = d / MANIFEST
    if not path.is_file():
        raise DefectError(f"{path} not found (build the set with mace defects build)")
    return json.loads(path.read_text())


def analyze(args) -> int:
    from mace.defects import analysis as an
    d = Path(args.dir)
    manifest = _load_manifest(d)
    texts = {}
    for sc in manifest["supercells"]:
        for j in sc["jobs"]:
            p = d / f"{j['name']}.out"
            if p.is_file():
                texts[j["name"]] = p.read_text(errors="replace")
    potc_texts = {}
    for sc in manifest["supercells"]:
        for j in sc["jobs"]:
            p = d / "potc" / f"{j['name']}_potc.POTC.DAT"
            if p.is_file():
                potc_texts[j["name"]] = p.read_text(errors="replace")
    mu = an.parse_mu(args.mu or [])
    if args.mu_file:
        data = json.loads(Path(args.mu_file).read_text())
        mu.update(an.parse_mu([f"{k}={v}" for k, v in data.items() if not k.startswith("_")]))
    vbm = an.parse_energy(args.vbm, "--vbm") if args.vbm else None
    report = an.analyze(manifest, texts, mu=mu, vbm=vbm, potc=potc_texts, apply_alignment=args.apply_alignment)
    _print_report(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str) + "\n")
        _p(f"wrote {args.json}")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["q", "job", "status", "E_raw_ha", "E_corr_ha", "formation_energy_ev_at_vbm",
                        "formation_energy_rel_q0_ev_at_vbm"])
            for s in report.get("states", []):
                w.writerow([s.get("q"), s.get("job"), s.get("status", "ok"), s.get("E_raw_ha"),
                            s.get("E_corr_ha"), s.get("formation_energy_ev_at_vbm"),
                            s.get("formation_energy_rel_q0_ev_at_vbm")])
        _p(f"wrote {args.csv}")
    return 0


def _print_report(r: dict) -> None:
    from mace.defects.deck_edit import fmt_q
    _p(f"{r['dimensionality']}D defect set - " + "; ".join(r["labels"]))
    _p(f"image-charge correction: {r['image_charge_correction']}")
    for j in r["jobs"]:
        line = f"  {j['name']:<24} {j['role']:<11} q={fmt_q(j['q']):<3} {j['status']}"
        if j["status"] != "ok":
            line += f" ({j['reason']})"
        _p(line)
        for w in j["warnings"]:
            _p(f"      WARNING: {w}")
    p = r.get("pristine")
    if p and p.get("vbm_ev") is not None:
        gap = f"{p['gap_ev']:.3f}" if p.get("gap_ev") is not None else "?"
        _p(f"pristine: VBM {p['vbm_ev']:.3f} eV (band {p['vbm_band']}, k {p['vbm_k']}; SCF k-mesh), gap {gap} eV"
           + (f"; --vbm {p['vbm_override_ev']:.3f} eV used" if p.get("vbm_override_ev") is not None else ""))
    for c in r.get("calibration", []) or []:
        if c.get("C") is not None:
            _p(f"calibration ({c['side']}): C = {c['C']:.5f} Ha (spread {c['spread']:.5f}) over "
               f"{', '.join(c['points'])}; q range {c['q_range']}")
            for v in c["values"]:
                x = v.get("spheropole_crosscheck")
                _p(f"    at {v['at']}: C = {v['C']:.5f} Ha" + (f"  (2*sph/N = {x:.5f}, cross-check only)" if x else ""))
    for f, v in (r.get("fits") or {}).items():
        if "A_ev" in v:
            _p(f"fit q={f}: A = {v['A_ev']:.3f} eV, B = {v['B_ev_A']:.2f} eV*A, residuals "
               + ", ".join(f"{x:.3f}" for x in v["residuals_ev"]) + " eV")
        else:
            _p(f"fit q={f}: {v.get('error')}")
    for s in r.get("states", []):
        bits = [f"q={fmt_q(s['q'])}"]
        for key in ("formation_energy_ev_at_vbm", "formation_energy_rel_q0_ev_at_vbm"):
            if s.get(key) is not None:
                bits.append(f"{key} = {s[key]:.3f}")
        if s.get("status"):
            bits.append(s["status"])
        _p("  " + "  ".join(bits))
    for lv in r.get("levels", []):
        _p(f"  eps({lv['transition']}) = {lv['eV_above_VBM']:.3f} eV above the VBM"
           + (f"  [{lv['flag']}]" if lv.get("flag") else ""))
    for q, a in (r.get("alignment") or {}).items():
        if "dV_ev" in a:
            _p(f"  alignment q={q}: dV = {a['dV_ev']:.3f} eV (std {a['std_ev']:.3f}, {a['n_far']} far atoms)"
               + ("" if r.get("alignment_applied") else " - reported, not applied")
               + (f"  {a['warning']}" if a.get("warning") else ""))
    for row in r.get("site_report", []) or []:
        bits = []
        for a in row["atoms"]:
            ch = a.get("charge")
            cp = a.get("charge_pristine")
            sp = a.get("spin")
            bits.append(f"{a['label']}:{ch:+.3f}" + (f"({cp:+.3f})" if cp is not None else "")
                        + (f" s={sp:+.3f}" if sp is not None else ""))
        _p(f"  site {row['job']} (Mulliken charge, pristine in brackets): " + "  ".join(bits))
    for n in r.get("notes", []):
        _p(f"note: {n}")


# ---------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mace defects", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action")

    b = sub.add_parser("build", help="write the pristine and charge-state decks")
    b.add_argument("host", nargs="?", help="single-point host deck (.d12)")
    g = b.add_argument_group("defect (one of)")
    g.add_argument("--vacancy", metavar="EL", help="remove an atom (ATOMREMO)")
    g.add_argument("--ghost", metavar="EL", help="vacancy keeping the basis functions (GHOSTS)")
    g.add_argument("--substitute", metavar="OLD=NEW", help="replace an atom (ATOMSUBS)")
    g.add_argument("--insert", metavar="EL@x,y,z", help="add an atom (ATOMINSE); fractional in the host "
                   "cell, or Cartesian Angstrom in the supercell frame with --cart")
    g.add_argument("--cart", action="store_true", help="--insert coordinates are Cartesian Angstrom")
    s = b.add_argument_group("site")
    s.add_argument("--site-label", type=int, metavar="N", help="supercell atom label (CRYSTAL numbering)")
    s.add_argument("--site-frac", metavar="x,y,z", help="site in the host deck's cell")
    c = b.add_argument_group("supercell (one of)")
    c.add_argument("--supercell", metavar="NxNxN", help="SUPERCEL, diagonal (primitive cell)")
    c.add_argument("--supercon", metavar="NxNxN", help="SUPERCON, diagonal (crystallographic cell, 3D)")
    c.add_argument("--supercell-matrix", metavar="'a b c d e f g h i'", help="SUPERCEL, full matrix")
    c.add_argument("--sizes", metavar="3,4,5,6", help="2D: one n x n set per size, for the A + B/L fit")
    q = b.add_argument_group("charge and spin")
    q.add_argument("--charges", default="0", metavar="0,1,2", help="charge states (default 0); "
                   "use --charges=-1,0,1 when the list starts with a minus")
    q.add_argument("--spin-state", action="append", metavar="q=NSPIN",
                   help="n_alpha - n_beta for one charge (repeatable); default 0 for even, 1 for odd")
    q.add_argument("--spinlock-cycles", type=int, default=de.DEFAULT_SPINLOCK_CYCLES,
                   help=f"SPINLOCK cycles (default {de.DEFAULT_SPINLOCK_CYCLES}, as measured)")
    q.add_argument("--host-spin", choices=("refuse", "keep"), default="refuse",
                   help="host SPINLOCK/ATOMSPIN/BETALOCK: refuse (default) or keep")
    k = b.add_argument_group("calibration (3D)")
    k.add_argument("--calibrate", nargs="?", const="0.5,1.0,1.5", metavar="LADDER",
                   help="fractional-DOPING calibration decks (default ladder 0.5,1.0,1.5 on the side of "
                        "the requested charges)")
    k.add_argument("--calibrate-on", choices=("defect", "pristine"), default="defect",
                   help="cell the calibration decks are built on (default defect; pristine untested)")
    k.add_argument("--calibrate-endpoints", action="store_true",
                   help="also write restricted endpoint decks (DOPING 0 and +-2)")
    o = b.add_argument_group("other")
    o.add_argument("--pristine-out", metavar="FILE", help="pristine supercell output (labels, electron count)")
    o.add_argument("--shrink", metavar="'IS [ISP]' | 'ka kb kc'", help="replace the host's SHRINK in every deck")
    o.add_argument("--name", metavar="PREFIX", help="job-name prefix (default: host file stem)")
    o.add_argument("-o", "--output", metavar="DIR", help="output directory (default HOST_defects)")
    o.add_argument("--force", action="store_true", help="overwrite different decks; allow GHOSTS on an ECP atom")
    o.add_argument("--resume", metavar="DIR", help="finish a set that was waiting for its pristine output")

    p = sub.add_parser("potc", help="stage POTC runs for potential alignment into DIR/potc")
    p.add_argument("dir")
    p.add_argument("--jobs", metavar="a,b", help="only these jobs")
    p.add_argument("--planar", nargs=2, type=float, metavar=("ZM", "ZP"),
                   help="2D: also the planar-averaged potential between ZM and ZP (bohr)")

    a = sub.add_parser("analyze", help="formation energies and transition levels")
    a.add_argument("dir")
    a.add_argument("--mu", action="append", metavar="EL=val[eV|Ha]", help="chemical potential (absolute)")
    a.add_argument("--mu-file", metavar="FILE", help='JSON {"O": "-75.1Ha"}')
    a.add_argument("--vbm", metavar="val[eV|Ha]", help="override the pristine VBM")
    a.add_argument("--apply-alignment", action="store_true", help="add q*(-dV) from POTC to E_f")
    a.add_argument("--json", metavar="FILE")
    a.add_argument("--csv", metavar="FILE")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = make_parser()
    args = ap.parse_args(argv)
    if not args.action:
        ap.print_help()
        return 0
    try:
        return {"build": build, "potc": potc, "analyze": analyze}[args.action](args)
    except DefectError as exc:
        print(f"mace defects {args.action}: refused - {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
