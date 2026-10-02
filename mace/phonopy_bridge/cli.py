"""`mace phonopy generate` - argument parsing and reporting."""

from __future__ import annotations

import argparse
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

    mace submit DIR --walltime HH:MM:SS     (never --track or --progress)

The parent must be a finished 3D OPT (OPT END - CONVERGED) or SP run. The
method, basis and tolerances of the parent are kept; only TOLDEE (raised to at
least 10), GUESSP (dropped), buffers, GRADCAL and END are edited, and SHRINK
with --scale-shrink. Every edit is printed.
"""

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
    g.add_argument("--scale-shrink", action="store_true",
                   help="divide SHRINK by the supercell size (diagonal --dim only)")

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
        dim = S.normalise_dim(" ".join(a.dim).split())
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    tag = S.dim_tag(dim)
    stem = out_path.stem
    if a.scale_shrink and not P.is_diagonal(dim.tolist()):
        print("Error: --scale-shrink needs a diagonal --dim", file=sys.stderr)
        return 1

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
        scale_shrink_dims=[int(v) for v in np.diag(dim)] if a.scale_shrink else None)

    outdir.mkdir(parents=True, exist_ok=True)
    names = W.write_set(outdir, stem, tag, deck, template, phonon, conv_super)
    import contextlib
    with contextlib.chdir(outdir):
        phonon.save(W.DISP_YAML)
    import phonopy
    W.write_manifest(outdir, {
        "parent_out": str(out_path.resolve()), "parent_out_sha256": W.sha256(out_path),
        "parent_d12": str(d12_path.resolve()), "parent_d12_sha256": W.sha256(d12_path),
        "dim": dim.tolist(), "amplitude": a.amplitude if a.amplitude is not None else 0.01,
        "pm": a.pm, "toldee": template.toldee, "scale_shrink": bool(a.scale_shrink),
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
    n_cells = int(round(abs(np.linalg.det(dim))))
    print(f"Supercell {tag}: {n_super} atoms ({n_cells} cells), {len(names)} displaced "
          f"deck(s), P1 (no symmetry)")
    kp = _kpoints(template.lines)
    if kp:
        nk = kp[0] * kp[1] * kp[2]
        print(f"  cost per job ~ {nk} k-points (before time reversal) x {n_super} atoms"
              f" = {nk * n_super} k-point x atom")
        if not a.scale_shrink and n_cells > 1:
            print(f"  The supercell keeps the parent's SHRINK: {n_cells}x the k-point density "
                  "the parent used. --scale-shrink keeps the parent's density with about "
                  f"1/{n_cells} of the k-points per job.")
    print(f"Wrote {len(names)} deck(s) + .ext to {outdir}/")
    print("Next:")
    print(f"  mace submit {outdir} --walltime 24:00:00    (adjust; do NOT add --track)")
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


def main(argv: Optional[List[str]] = None) -> int:
    p = _parser()
    a = p.parse_args(argv)
    if a.action is None:
        p.print_help()
        return 2
    if not _require_phonopy():
        return 2
    return cmd_generate(a)


if __name__ == "__main__":
    sys.exit(main())
