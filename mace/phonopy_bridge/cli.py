"""`mace phonopy generate | collect` - argument parsing and reporting."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional

INSTALL_HINT = ("phonopy is not installed. Install the optional dependency with\n"
                "    pip install 'phonopy<5'\n"
                "(see requirements-optional.txt).")

GENERATE_EPILOG = """
Writes one EXTERNAL deck per displaced supercell (<stem>_ph<dim>_disp-NNN.d12
with its geometry in .ext, staged as fort.34 by the job script), plus
phonopy_disp.yaml and mace_phonopy.json. Then:

    mace submit DIR --walltime <printed>    (never --track or --progress)
    mace phonopy collect DIR [--nac-from FREQ.out]

The parent must be a finished 3D OPT (OPT END - CONVERGED) or SP run. The
method, basis and tolerances of the parent are kept. The deck is phonopy's own
layout: EXTERNAL, the parent's basis and SCF block (no END, TOLDEE or GRADCAL in
it - written to TEMPLATE for an explicit basis), then TOLDEE (at least 10),
GRADCAL and END. GUESSP is dropped, buffers are added for large cells, and
SHRINK is divided by the supercell size in each direction (rounded up when it
does not divide; kept for a non-diagonal --dim or with --keep-shrink). Every
edit is printed.
"""

COLLECT_EPILOG = """
Checks every output first (present, normal termination, SCF converged, one
force row per atom, printed geometry == its displaced cell) and stops on any
problem; --check stops there. Then writes FORCE_SETS, phonopy_params.yaml,
band.yaml, mesh.yaml, total_dos.dat and thermal_properties.yaml (--plot adds
band_dos.pdf). Frequencies are in THz (x 33.35641 for cm-1).
"""


# Walltime estimate for one displaced-supercell job at mace submit's default 32
# tasks. Calibrated on the heaviest case measured on CRYSTAL23: diamond,
# B3LYP-D3 + SPIN + XLGRID, 2x2x2 (16 atoms) in P1 with SHRINK 6 (216 k-points
# before time reversal) took 3160 s on 64 tasks, i.e. about 6300 s on 32. The
# cost is taken as k-points x atoms^3; 3x3x3 at SHRINK 4 (54 atoms) did not
# finish its SCF within 2 h on 64 tasks, which this puts at about 20 h on 32
# (30 h with the margin below).
_SECONDS_PER_K_ATOM3 = 2 * 3160.0 / (216 * 16 ** 3)
# Margin over that calibration: the 32-task time is extrapolated from the
# 64-task run, and a job killed at its walltime leaves no forces, so a low
# estimate costs the whole set while a high one only queue time (mace submit's
# own default is 7 days). Lighter methods and --cry-sym (measured: Si PBE0
# 4x4x4 at 1.7 h, --cry-sym about 3x cheaper) are not shortened for the same
# reason.
WALLTIME_MARGIN = 1.5
MIN_WALLTIME_HOURS = 2
MAX_WALLTIME_HOURS = 7 * 24          # mace submit's default (submitcrystal23.sh)
FALLBACK_WALLTIME = "1-00:00:00"


def format_walltime(hours: int) -> str:
    """SLURM -t value as mace submit --walltime passes it on: H:MM:SS or D-HH:MM:SS."""
    if hours < 24:
        return f"{hours}:00:00"
    return f"{hours // 24}-{hours % 24:02d}:00:00"


def suggest_walltime(n_kpoints: int, n_atoms: int):
    """(walltime, capped) for one job; capped when the estimate exceeds 7 days."""
    seconds = WALLTIME_MARGIN * _SECONDS_PER_K_ATOM3 * n_kpoints * n_atoms ** 3
    hours = max(MIN_WALLTIME_HOURS, math.ceil(seconds / 3600))
    capped = hours > MAX_WALLTIME_HOURS
    return format_walltime(min(hours, MAX_WALLTIME_HOURS)), capped


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mace phonopy",
                                description="Finite-displacement phonons with phonopy")
    sub = p.add_subparsers(dest="action")

    g = sub.add_parser("generate", help="write the displaced-supercell decks",
                       epilog=GENERATE_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    g.add_argument("parent_out", help="finished parent output (OPT or SP)")
    g.add_argument("--d12", help="parent deck (default: same stem as the output)")
    g.add_argument("--dim", required=True, nargs="+",
                   help='supercell: "N N N" or 9 integers (row-major matrix)')
    g.add_argument("--output-dir", help="default: phonopy_<stem>_<dim>/")
    g.add_argument("--force", action="store_true", help="write into a non-empty --output-dir")
    g.add_argument("--amplitude", type=float, default=None,
                   help="displacement in Angstrom (default: phonopy's 0.01)")
    g.add_argument("--pm", choices=["auto", "yes", "no"], default="auto",
                   help="plus/minus displacements (default: auto)")
    g.add_argument("--toldee", type=int, default=None,
                   help="TOLDEE for the supercells; can only raise the parent's (default: max(10, parent))")
    g.add_argument("--cry-sym", action="store_true",
                   help="write each displaced cell's symmetry into its .ext (phonopy's "
                        "CRY_SYM): fewer k-points, but phonopy warns some space groups "
                        "(e.g. R-3m) do not work")
    g.add_argument("--keep-shrink", action="store_true",
                   help="keep the parent's SHRINK (default: divide it by the supercell size, "
                        "rounding up; diagonal --dim only)")
    g.add_argument("--scale-shrink", action="store_true", help=argparse.SUPPRESS)

    c = sub.add_parser("collect", help="check outputs, then forces -> phonons",
                       epilog=COLLECT_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    c.add_argument("directory")
    c.add_argument("--check", action="store_true", help="only check the outputs")
    c.add_argument("--band", default="auto",
                   help='"auto" (SeeK-path, needs seekpath) or phonopy BAND points, e.g. '
                        '"0 0 0  1/2 0 1/2, 1/2 1/2 1/2  0 0 0"; "none" to skip')
    c.add_argument("--band-labels", help="labels for --band points, space separated")
    c.add_argument("--band-points", type=int, default=51)
    c.add_argument("--mesh", default="100.0", help='length (default 100.0) or "a b c"')
    c.add_argument("--tmin", type=float, default=0.0)
    c.add_argument("--tmax", type=float, default=1000.0)
    c.add_argument("--tstep", type=float, default=10.0)
    c.add_argument("--nac-from", help="Gamma FREQCALC output with INTENS + INTCPHF")
    c.add_argument("--plot", action="store_true", help="write band_dos.pdf")
    return p


def _require_phonopy() -> bool:
    try:
        import phonopy  # noqa: F401
    except ImportError:
        print(INSTALL_HINT, file=sys.stderr)
        return False
    return True


def _inside_workflow_outputs(path: Path) -> bool:
    return "workflow_outputs" in path.resolve().parts


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def cmd_generate(a) -> int:
    import numpy as np

    from mace import __version__ as mace_version

    from . import parent as P
    from . import structure as S
    from . import writer as W

    out_path = Path(a.parent_out)
    d12_path = Path(a.d12) if a.d12 else out_path.with_suffix(".d12")
    for f in (out_path, d12_path):
        if not f.is_file():
            print(f"Error: {f} not found", file=sys.stderr)
            return 1
    try:
        dim = S.normalize_dim(" ".join(a.dim).split())
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    tag = S.dim_tag(dim)
    stem = out_path.stem
    if a.keep_shrink and a.scale_shrink:
        print("Error: --keep-shrink and --scale-shrink contradict each other", file=sys.stderr)
        return 1
    diagonal = P.is_diagonal(dim.tolist())
    scale_shrink = diagonal and not a.keep_shrink

    outdir = Path(a.output_dir) if a.output_dir else Path(f"phonopy_{stem}_{tag}")
    if _inside_workflow_outputs(outdir):
        print(f"Error: {outdir} is inside a workflow_outputs tree; the workflow engine would "
              "pick these jobs up. Choose another --output-dir.", file=sys.stderr)
        return 1
    if outdir.exists() and any(outdir.iterdir()) and not a.force:
        print(f"Error: {outdir} is not empty (use --force to write into it)", file=sys.stderr)
        return 1

    try:
        deck, out_text = P.read_parent(d12_path, out_path)
    except P.DeckSplitError as exc:
        print(f"Error: {d12_path}: {exc}", file=sys.stderr)
        return 1
    reasons = P.refusal_reasons(deck, out_text)
    if reasons:
        print(f"Refusing {out_path.name}:", file=sys.stderr)
        for code, msg in reasons:
            print(f"  [{code}] {msg}", file=sys.stderr)
        return 1
    warns = P.warnings_for(deck, out_text)

    cell, conv = S.read_parent_cell(out_path)
    fold = S.folded_warning(cell)
    if fold:
        warns.append(fold)
    phonon = S.make_phonopy(cell, dim)
    S.generate(phonon, a.amplitude, a.pm)
    conv_super = S.supercell_conv_numbers(phonon, conv)
    n_super = len(phonon.supercell.numbers)
    template = P.template_lines(
        deck, n_super, toldee=a.toldee,
        scale_shrink_dims=[int(v) for v in np.diag(dim)] if scale_shrink else None)
    if not diagonal and not a.keep_shrink:
        warns.append(("SHRINK_KEPT", "non-diagonal --dim: SHRINK kept as in the parent (it "
                      "cannot be divided per direction); set it in the decks by hand"))

    try:
        P._check_template(deck, template)
    except ValueError as exc:       # refusal_reasons should have caught it already
        print(f"Error: {d12_path}: {exc}", file=sys.stderr)
        return 1

    outdir.mkdir(parents=True, exist_ok=True)
    if a.cry_sym:
        warns += [("CRY_SYM", m) for m in S.cry_sym_warnings(cell)]
    names = W.write_set(outdir, stem, tag, deck, template, phonon, conv_super,
                        cry_sym=a.cry_sym)
    if a.cry_sym:
        (outdir / W.CRY_SYM).write_text("")   # phonopy-init --crystal -d reads it too
    template_file = None
    if deck.basis_kind == "explicit":
        # phonopy-init --crystal -d reads this; it puts EXTERNAL / ENDGEOM in front,
        # which only an explicit basis can follow.
        template_file = W.TEMPLATE
        (outdir / W.TEMPLATE).write_text(P.template_text(template))
    with W.working_directory(outdir):
        phonon.save(W.DISP_YAML)
    n_cells = int(round(abs(np.linalg.det(dim))))
    kp = _kpoints(template.lines)
    nk = kp[0] * kp[1] * kp[2] if kp else None
    if nk:
        walltime, capped = suggest_walltime(nk, n_super)
    else:
        walltime, capped = FALLBACK_WALLTIME, False

    import phonopy
    W.write_manifest(outdir, {
        "parent_out": str(out_path.resolve()), "parent_out_sha256": W.sha256(out_path),
        "parent_d12": str(d12_path.resolve()), "parent_d12_sha256": W.sha256(d12_path),
        "dim": dim.tolist(), "amplitude": a.amplitude if a.amplitude is not None else 0.01,
        "pm": a.pm, "toldee": template.toldee, "scale_shrink": bool(scale_shrink),
        "template": template_file, "cry_sym": bool(a.cry_sym), "walltime": walltime,
        "edits": template.edits, "dropped_block1_keywords": deck.dropped_keywords,
        "basis": deck.basis_kind, "method_lines": P.method_lines(deck),
        "conv_numbers": [int(c) for c in conv], "conv_numbers_supercell": conv_super,
        "n_atoms_supercell": n_super, "jobs": names,
        "warnings": [w[1] for w in warns],
        "mace_version": mace_version, "phonopy_version": phonopy.__version__,
    })

    # ---- report
    print(f"Parent: {out_path} ({deck.basis_kind} basis"
          + (f" {deck.basis_label}" if deck.basis_label else "") + ")")
    for _, msg in warns:
        print(f"  Warning: {msg}")
    if deck.dropped_keywords:
        print("Dropped with block 1 (geometry comes from the output): "
              + ", ".join(deck.dropped_keywords))
    print("Deck edits relative to the parent:")
    for e in template.edits:
        print(f"  {e}")
    print(f"Supercell {tag}: {n_super} atoms ({n_cells} cells), {len(names)} displaced "
          "deck(s), " + ("symmetry of each displaced cell (--cry-sym)" if a.cry_sym
                         else "P1 (no symmetry)"))
    if nk:
        print(f"  cost per job ~ {nk} k-points (before time reversal) x {n_super} atoms"
              f" = {nk * n_super} k-point x atom")
        if not scale_shrink and n_cells > 1:
            print(f"  The supercell keeps the parent's SHRINK: {n_cells}x the k-point density "
                  "the parent used.")
    if nk:
        print(f"Walltime per job ~ {walltime} at mace submit's 32 tasks (k-points x atoms^3, "
              "calibrated on a hybrid-DFT diamond supercell measured on CRYSTAL23, with a 1.5x "
              "margin; lighter methods and --cry-sym run faster). A job that hits its walltime leaves no forces, and "
              "collect reports it.")
        if capped:
            print("  WARNING: the estimate is beyond 7 days, the longest mace submit asks "
                  "for: use --cry-sym, a smaller supercell or fewer k-points (SHRINK).")
    else:
        print(f"Walltime per job: no SHRINK found to estimate from; {walltime} suggested.")
    print(f"Wrote {len(names)} deck(s) + .ext to {outdir}/")
    print("Next:")
    print(f"  mace submit {outdir} --walltime {walltime}    (do NOT add --track)")
    print(f"  mace phonopy collect {outdir}")
    return 0


def _kpoints(lines: List[str]) -> Optional[List[int]]:
    for i, ln in enumerate(lines):
        if ln.strip().upper() == "SHRINK" and i + 1 < len(lines):
            try:
                first = [int(x) for x in lines[i + 1].split()[:2]]
                if first[0] == 0:
                    return [int(x) for x in lines[i + 2].split()[:3]]
                return [first[0]] * 3
            except (ValueError, IndexError):
                return None
    return None


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

def cmd_collect(a) -> int:
    from . import collect as C
    from .nac import NacError

    d = Path(a.directory)
    if not (d / "mace_phonopy.json").is_file():
        print(f"Error: {d} has no mace_phonopy.json (not made by mace phonopy generate)",
              file=sys.stderr)
        return 1
    band = None if a.band.lower() == "none" else a.band
    if band == "auto":
        try:
            import seekpath  # noqa: F401
        except ImportError:
            print("Error: --band auto needs seekpath (pip install seekpath); give the path "
                  "with --band or skip it with --band none", file=sys.stderr)
            return 1
    try:
        res = C.collect(d, check_only=a.check, band=band, band_labels=a.band_labels,
                        band_points=a.band_points, mesh=a.mesh, tmin=a.tmin, tmax=a.tmax,
                        tstep=a.tstep, nac_from=a.nac_from, plot=a.plot)
    except NacError as exc:
        print(f"Error: NAC refused: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    bad = [c for c in res.checks if c.problems]
    for c in res.checks:
        state = "ok" if not c.problems else "PROBLEM"
        print(f"  {c.name}.out: {state}")
        for p in c.problems:
            print(f"      - {p}")
    if bad:
        print(f"{len(bad)} of {len(res.checks)} output(s) not usable; nothing written.",
              file=sys.stderr)
        return 1
    if a.check:
        print(f"All {len(res.checks)} output(s) usable.")
        return 0
    for ln in res.drifts:
        print(f"  {ln}")
    print("Wrote: " + ", ".join(res.files))
    if res.min_frequency is not None:
        print(f"Lowest mesh frequency: {res.min_frequency:.4f} THz "
              f"({res.min_frequency * C.CM1_PER_THZ:.1f} cm-1)")
    if res.imaginary:
        worst = min(res.imaginary, key=lambda t: t[1])
        print(f"WARNING: {len(res.imaginary)} q-point(s) away from Gamma with a mode below "
              f"{C.IMAGINARY_THRESHOLD} THz (lowest {worst[1]:.3f} THz at q={worst[0]}): "
              "the structure is dynamically unstable or the supercell is too small. Thermal "
              "properties skip imaginary modes.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = _parser()
    a = p.parse_args(argv)
    if a.action is None:
        p.print_help()
        return 2
    if not _require_phonopy():
        return 2
    if a.action == "generate":
        return cmd_generate(a)
    return cmd_collect(a)


if __name__ == "__main__":
    sys.exit(main())
