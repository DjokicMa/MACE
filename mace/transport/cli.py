"""``mace tau``: a physically documented relaxation time for BOLTZTRA.

estimate  tau and the mobility from deformation-potential theory, given m*,
          E1 and C (or from a mobility you have, --mobility)
strain    the strained single-point decks of an SP parent (ELASTIC, one axis)
fit       E1 and C from those outputs (+ the unstrained parent), then tau;
          m* from --m-star, or (default, --m-star auto) from the BAND and
          DOSS outputs beside them or given (--band, --dos; mass.py), C
          optionally from an ELASTCON output (--c-from)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

EPILOG = """\
Method: deformation-potential theory, longitudinal acoustic phonons
(Bardeen & Shockley 1950; 2D: Takagi 1994 / Qiao 2014; 1D: Beleznay 2003).
tau = mu m*/e is the constant relaxation time for RELAXTIM; in 3D tau(E)
is printed too. It is the acoustic-phonon limit: other scattering lowers tau.

Examples:
  mace tau estimate --dim 3 --m-star 0.26 --e1 9.5 --c 165 --temperature 300
  mace tau estimate --dim 2 --m-star 0.17 --m-other 1.12 --e1 2.72 --c 28.94
  mace tau estimate --mobility 1400 --m-star 0.26          # tau from a measured mobility
  mace tau strain mat_sp.d12 --axis x                      # strains -1%, -0.5%, +0.5%, +1%
  mace submit .                                            # run the strained decks
  mace tau fit mat_sp_dp_x_*.out --carrier electron --m-star 0.26
  mace tau fit mat_sp_dp_x_*.out --carrier hole         # m* from the BAND/DOSS outputs beside them
  mace tau fit mat_sp_dp_x_*.out --carrier hole --band mat_band.BAND.DAT --dos mat_doss.DOSS.DAT \\
               --c-from mat_elastcon.out --save tau.json
  mace tau estimate --carrier electron --axis x --band mat_band.f25 --e1 9.5 --c 165
Masses: tau takes the density-of-states mass m_d, the mobility the band
mass m_c along the transport axis. --m-star auto fits both: m_c from the
band curvature along the path segment through the edge that runs along
the axis, m_d from the band directions through the edge or the DOS. Each
mass printed says where it came from; an explicit --m-star (and --m-dos)
wins.
Then: RELAXTIM <tau fs> in the TRANSPORT deck, or rescale a finished run:
  mace analyze --extract-properties DIR --kappa-lattice K --tau-fs <tau fs>
"""


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mace tau", description="Relaxation time tau for Boltzmann transport.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EPILOG, allow_abbrev=False)
    sub = p.add_subparsers(dest="action", required=True)
    e = sub.add_parser("estimate", help="tau from m*, E1 and C (or from a mobility)",
                       formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EPILOG)
    e.add_argument("--dim", type=int, choices=(1, 2, 3), default=3)
    e.add_argument("--m-star", "--mstar", help="effective mass along transport (m_e), or auto (from --band/--dos)")
    e.add_argument("--m-other", type=float, help="2D: the in-plane mass across transport (m_e; default m*)")
    e.add_argument("--m-dos", type=float, help="the density-of-states mass m_d (m_e; default m*, 2D sqrt(m* m_other))")
    e.add_argument("--carrier", choices=("electron", "hole"), help="with --m-star auto: which band edge")
    e.add_argument("--axis", choices=("x", "y", "z"), help="with --m-star auto: the transport direction")
    e.add_argument("--band", help="BAND.DAT or BAND .f25 for --m-star auto")
    e.add_argument("--dos", help="DOSS.DAT or DOSS .f25 for --m-star auto")
    e.add_argument("--mass-window", type=float, default=40.0, help="meV from the band edge fitted (default 40)")
    e.add_argument("--e1", type=float, help="deformation potential (eV)")
    e.add_argument("--c", type=float, help="elastic constant: GPa (3D), N/m (2D), J/m (1D)")
    e.add_argument("--temperature", "-T", type=float, default=300.0)
    e.add_argument("--mobility", type=float, help="cm^2/(V s): tau = mu m*/e instead of the DP formula")
    e.add_argument("--json", action="store_true")
    e.add_argument("--save", help="write the result (with provenance) as JSON")
    s = sub.add_parser("strain", help="strained SP decks for the deformation potential")
    s.add_argument("parent", help="the SP deck (.d12) of the optimized structure")
    s.add_argument("--axis", default="x", choices=("x", "y", "z"))
    s.add_argument("--strains", default="-0.01,-0.005,0.005,0.01", help="comma-separated (default ±0.5%%, ±1%%)")
    s.add_argument("--output-dir")
    s.add_argument("--force", action="store_true", help="replace decks that exist")
    f = sub.add_parser("fit", help="E1 and C from the strained outputs, then tau",
                       formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EPILOG)
    f.add_argument("outputs", nargs="+", help="the outputs of the strain set (unstrained one included)")
    f.add_argument("--carrier", choices=("electron", "hole"), required=True)
    f.add_argument("--m-star", "--mstar", default="auto",
                   help="effective mass along transport (m_e), or auto (default): from the BAND/DOSS outputs "
                        "beside the outputs, or --band/--dos")
    f.add_argument("--m-other", type=float, help="2D: the in-plane mass across transport (m_e)")
    f.add_argument("--m-dos", type=float, help="the density-of-states mass m_d (m_e)")
    f.add_argument("--band", "--band-dat", dest="band", help="BAND.DAT or BAND .f25 to fit m* from")
    f.add_argument("--dos", help="DOSS.DAT or DOSS .f25 to fit m_d from")
    f.add_argument("--mass-window", type=float, default=40.0, help="meV from the band edge fitted (default 40)")
    f.add_argument("--c-from", help="an ELASTCON output: use its relaxed-ion C_ii (3D) instead of the fitted C")
    f.add_argument("--e1", type=float, help="use this E1 (eV) instead of the fitted one")
    f.add_argument("--temperature", "-T", type=float, default=300.0)
    f.add_argument("--json", action="store_true")
    f.add_argument("--save", help="write the result (with provenance) as JSON")
    for q in sub.choices.values():
        q.allow_abbrev = False                          # full option names only
    return p


def resolve_masses(m_star, carrier: Optional[str], axis: Optional[str], dim: int, outputs=(), band=None,
                   dos=None, m_dos: Optional[float] = None, window_mev: float = 40.0, inside=None):
    """(m_c, m_d or None, sources, the automatic record or None). A number for ``m_star`` wins (with
    ``m_dos`` when given); "auto"/None fits them from ``band``/``dos`` or the files found beside
    ``outputs`` (mace.transport.mass; ``inside``: the folder the search stays in). Raises mass.MassError
    (a ValueError) when there is no mass."""
    from mace.transport import dpt, mass
    if m_star not in (None, "") and str(m_star).strip().lower() != "auto":
        try:
            m = float(m_star)
        except ValueError:
            raise dpt.TauError(f"m* is a number (m_e) or auto, not {m_star!r}")
        src = {"m_star": "given (--m-star)"}
        if m_dos is not None:
            src["m_dos"] = "given (--m-dos)"
        return m, m_dos, src, None
    if carrier not in ("electron", "hole"):
        raise dpt.TauError("--m-star auto needs --carrier electron|hole (which band edge)")
    am = mass.auto_masses(carrier, axis, dim, [Path(o) for o in outputs], Path(band) if band else None,
                          Path(dos) if dos else None, window_mev, inside)
    src = dict(am["sources"])
    md = am["m_dos"]
    if m_dos is not None:
        md, src["m_dos"] = m_dos, "given (--m-dos)"
    return am["m_star"], md, src, am


def c_from_elastcon(path: str, axis: str) -> float:
    from mace.elastic.analysis import analyze_file
    data = analyze_file(path)
    el = data.get("elastic")
    if not el or not str(el.get("unit", "")).upper().startswith("GPA"):
        raise SystemExit(f"Error: {path} has no 3D elastic constants in GPa")
    k = "xyz".index(axis)
    return float(el["matrix"][k][k])


def _emit(res, args) -> None:
    from mace.transport import dpt
    if args.save:
        Path(args.save).write_text(json.dumps(res, indent=2) + "\n")
    if args.json:
        print(json.dumps(res))
    else:
        if res.get("masses"):
            from mace.transport import mass
            print(mass.report(res["masses"]))
        if "fit" in res:
            ft = res["fit"]
            print(f"Fit ({ft['axis']}, {ft['carrier']}, {ft['dimensionality']}D, {len(ft['strains'])} points):")
            print(f"  E1 = {ft['E1_eV']:.4f} eV (R^2 {ft['E1_r2']:.4f}),  edge reference: {ft['edge_reference']}")
            print(f"  C  = {ft['C']:.4g} {ft['C_unit']} (R^2 {ft['C_r2']:.5f}, clamped ions)")
            for w in ft.get("warnings", []):
                print(f"  Warning: {w}")
        print(dpt.report(res))
        if args.save:
            print(f"Saved {args.save}")


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    from mace.transport import dpt, mass, strain
    try:
        if args.action == "estimate":
            if args.m_star in (None, "") and not (args.band or args.dos):
                print("Error: --m-star is required (a number, or auto with --band/--dos)", file=sys.stderr)
                return 2
            m, md, msrc, am = resolve_masses(args.m_star, args.carrier, args.axis or "x", args.dim,
                                             band=args.band, dos=args.dos, m_dos=args.m_dos,
                                             window_mev=args.mass_window)
            if args.mobility is not None:
                res = dpt.estimate_from_mobility(args.mobility, m, msrc)
            else:
                if args.e1 is None or args.c is None:
                    print("Error: --e1 and --c are required (or --mobility)", file=sys.stderr)
                    return 2
                res = dpt.estimate(args.dim, m, args.e1, args.c, args.temperature, args.m_other, msrc,
                                   carrier=args.carrier, axis=args.axis, m_dos=md)
            if am:
                res["masses"] = am
            _emit(res, args)
            return 0
        if args.action == "strain":
            strains = [float(x) for x in args.strains.split(",") if x.strip()]
            written = strain.write_strain_set(Path(args.parent), args.axis, strains,
                                              Path(args.output_dir) if args.output_dir else None, args.force)
            for p in written:
                print(f"Wrote {p}")
            stem = Path(args.parent).stem
            print(f"Run them (mace submit), then: mace tau fit {stem}_dp_{args.axis}_*.out "
                  "--carrier electron|hole  (m* from the BAND/DOSS outputs beside them, or --m-star M)")
            return 0
        # fit
        points = [strain.read_point(Path(p)) for p in args.outputs]
        ft = strain.fit(points, args.carrier)
        sources = {"E1": f"linear fit of the {'CBM' if args.carrier == 'electron' else 'VBM'} over "
                         f"{len(ft['strains'])} strained SP runs ({ft['axis']})",
                   "C": "quadratic fit of the total energy over the same runs (clamped ions)"}
        e1 = ft["E1_eV"]
        if args.e1 is not None:
            e1, sources["E1"] = args.e1, "given (--e1)"
        c = ft["C"]
        if args.c_from:
            if ft["dimensionality"] != 3:
                print("Error: --c-from reads 3D elastic constants in GPa", file=sys.stderr)
                return 2
            c, sources["C"] = c_from_elastcon(args.c_from, ft["axis"]), f"C_{ft['axis']*2} of {Path(args.c_from).name} (ELASTCON)"
        m, md, msrc, am = resolve_masses(args.m_star, args.carrier, ft["axis"], ft["dimensionality"],
                                         args.outputs, args.band, args.dos, args.m_dos, args.mass_window)
        sources.update(msrc)
        res = dpt.estimate(ft["dimensionality"], m, e1, c, args.temperature, args.m_other, sources,
                           carrier=args.carrier, axis=ft["axis"], m_dos=md)
        if am:
            res["masses"] = am
        res["fit"] = ft
        _emit(res, args)
        return 0
    except (dpt.TauError, strain.StrainError, mass.MassError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
