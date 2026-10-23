"""The command line of ``mace analyze elastic`` (mace/analysis/cli.py registers it):
the tensors and EOS fits of a finished ELASTIC, EOS, QHA or CPHF run (the decks are
mace opt2d12's, choices 4-7 / --calc-type), with the derived moduli
(``mace.elastic.analysis``); ``--json`` for scripts and the GUI; ``--plot`` draws the
run's plots (``mace.elastic.plot``: the directional Young's modulus and linear
compressibility, E(V) and P(V), V(T), thermal expansion, B(T), heat capacity, eps(w)),
the figures the GUI's results document shows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EPILOG = """\
Examples:
  mace analyze elastic mat_opt_elastcon_PBE0_optimized.out
  mace analyze elastic run.out --json
  mace analyze elastic run_elastcon.out --plot youngs               # E(n) as a 3D surface (png)
  mace analyze elastic run_elastcon.out --plot compressibility --shape planes --format svg
  mace analyze elastic run_elastcon.out --plot shear --plot poisson --plot sound
  mace analyze elastic run_elapiezo.out --plot piezo --epsilon 6.8 --json
  mace analyze elastic run_eos.out --plot all -o figures            # E(V) and P(V)
  mace analyze elastic run_qha.out --plot expansion
  mace analyze elastic diamond_eos.out --compare lonsdaleite_eos.out --plot enthalpy

Plots (--plot; the GUI's results document draws the same, Copy CLI Command prints the line):
  youngs, compressibility   ELASTIC: directional Young's modulus E(n) and linear
                            compressibility beta(n) (--shape surface, or planes: the
                            xy, xz and yz cuts; a slab: its in-plane polar plot)
  shear, poisson            ELASTIC: shear modulus G(n, m) and Poisson's ratio nu(n, m), their
                            minimum and maximum over m perpendicular to n (ELATE's envelopes;
                            --shape surface or planes)
  sound                     ELASTIC: Christoffel phase and group velocities of the quasi-
                            longitudinal and two quasi-shear waves, xy, xz and yz cuts
  piezo                     ELAPIEZO/PIEZOCON: the longitudinal piezoelectric coefficient
                            d_L(n) (e_L(n) without d), signed (--shape surface or planes)
  eos, pv                   EOS: E(V) points with the four fits, P(V) of the fits
  bp, vp, residuals         EOS: B(P) and V/V0(P) of each fit, each fit's residuals (RMS)
  enthalpy                  EOS: H(P) = E + PV per formula unit; with --compare, H - H_ref of the
                            other phases and the transition pressures (--fit picks the fit)
  volume, vratio, expansion, bulk, heat
                            QHA: V(T), V/V0 and a/a0, alpha(T), B_T and B_S, Cp - Cv (with
                            Cp and Cv along V(T) and each volume's harmonic Cv when the
                            output has them)
  gamma, free, entropy, fv  QHA: the Grueneisen parameter (CRYSTAL's and alpha B V / Cv),
                            F(T), S(T), and F(V) at several temperatures with the minima
                            traced (each volume's harmonic blocks or the PVT table)
  epsilon                   CPHF DYNAMIC: the dielectric tensor at each frequency
  refractive, absorption    CPHF DYNAMIC: n(w) and kappa(w) of each diagonal component; the
                            absorption coefficient 2 w kappa / c (with DAMPING)
  indicatrix                CPHF: the refractive-index ellipsoid of eps_inf
  shg                       CPHF with beta: |d_ijk n_j n_k| (--shape surface or planes)
"""


def add_analyze_arguments(a) -> None:
    """The options of ``mace analyze elastic``."""
    a.add_argument("outputs", nargs="+", help="CRYSTAL output files")
    a.add_argument("--json", action="store_true", help="one JSON object per file")
    a.add_argument("--compare", action="append", metavar="OUT",
                   help="EOS: another phase's EOS output (repeatable, same composition): --plot enthalpy draws H - H_ref "
                        "per formula unit and the transition pressures; --json adds them")
    a.add_argument("--fit", choices=("murnaghan", "birch_murnaghan", "poirier_tarantola", "vinet"),
                   help="EOS: the fit the enthalpy comparison uses (default birch_murnaghan)")
    a.add_argument("--epsilon", metavar='"E" | "EXX EYY EZZ"',
                   help="the static stress-free relative permittivity for the piezoelectric coupling k^2 (one, three or "
                        "nine values; default: the run's electronic eps_inf when it has one, an upper bound)")
    from .views import PLOTS, SHAPES
    a.add_argument("--plot", action="append", metavar="NAME", choices=[*PLOTS, "all"],
                   help="draw this plot (repeatable): " + ", ".join(PLOTS) + ", or all (every plot the run has)")
    a.add_argument("--shape", choices=SHAPES, default="surface",
                   help="youngs, compressibility, shear, poisson and piezo of a 3D run: a 3D surface (default) or "
                        "polar cuts in the xy, xz and yz planes (a slab draws its in-plane polar plot)")
    a.add_argument("--format", choices=("png", "svg", "pdf"), action="append",
                   help="figure format, repeatable (default: png)")
    a.add_argument("-o", "--output-dir", dest="plot_dir", help="where the figures go (default: beside the output)")
    from mace.plotting.presets import ORDER
    a.add_argument("--style", choices=list(ORDER), help="plot style, as mace plotting --style (default: default)")


def analyze(args) -> int:
    """``mace analyze elastic RUN.out ...``"""
    return _analyze(args)


def _analyze(args) -> int:
    from mace.elastic import analysis as A
    rc = 0
    for path in args.outputs:
        p = Path(path)
        if not p.is_file():
            print(f"Error: {path}: no such file", file=sys.stderr)
            rc = 1
            continue
        try:
            data = A.analyze_file(p, args.epsilon)
        except ValueError as e:
            print(f"Error: --epsilon: {e}", file=sys.stderr)
            return 2
        compare = []
        for c in args.compare or []:
            cp = Path(c)
            if not cp.is_file():
                print(f"Error: --compare {c}: no such file", file=sys.stderr)
                return 1
            compare.append((cp.name, A.analyze_file(cp)))
        found = []
        if compare and data.get("eos"):
            from . import views as V
            try:
                for label, other in compare:
                    t = V.transitions(data, other, args.fit)
                    found.append({"phase": label, "formula": other.get("formula"),
                                  **{k: t[k] for k in ("fit", "p_range_gpa", "overlap", "crossings")}})
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1
        if args.json:
            print(json.dumps({"file": str(p), **data, **({"transitions": found} if compare else {})}))
        else:
            print(A.report(data, p.name))
            for f in found:
                ps = ", ".join(f"{c['pressure_gpa']:.2f} GPa ({f['phase'] if c['stable_above'] == 'other' else p.name} stable above)"
                               for c in f["crossings"])
                print(f"Enthalpy against {f['phase']} ({f['fit']}): " + (f"transition at {ps}" if ps else
                      f"no crossing between {f['p_range_gpa'][0]:.1f} and {f['p_range_gpa'][1]:.1f} GPa"))
            print()
        if not data:
            rc = rc or 1
        if args.plot and data:
            from . import plot as P
            try:
                for f in P.plot_file(p, args.plot, args.shape, args.plot_dir, tuple(args.format or ["png"]), args.style,
                                     compare, args.fit):
                    print(f"  Generated: {f}", file=sys.stderr if args.json else sys.stdout)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                rc = rc or 1
    return rc
