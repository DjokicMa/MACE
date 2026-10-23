"""`mace supercell`: the GUI's Build › Supercell from the command line (1.1.6).

A supercell of any structure MACE reads (CIF, P1 or 3D .d12, CRYSTAL .out,
fort.34 / .gui, POSCAR, XYZ / extended XYZ). A slab, polymer or molecule
repeats along its periodic directions; along a non-periodic one it stays 1
unless you ask for more there, and then the copies are spaced by a Vacuum
mode, as the GUI's Inspector › Display › Vacuum spaces them:

  bbox     the atoms' extent + 1 Å on each side (default)
  file     the vacuum the file defines (a CRYSTAL output's 500 Å, an extended XYZ's box)
  crystal  CRYSTAL's 500 Å
  GAP      a number: the atoms' extent + GAP Å (custom)
  open     nothing repeats along a non-periodic direction (asking for more is an error)

The result is written as a CIF (a slab, polymer or molecule in a box with
15 Å of vacuum, the format having no pbc) or, with --format extxyz, as an
extended XYZ that keeps the periodicity.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

USAGE_EPILOG = """examples:
  mace supercell MgO.cif 2 2 2
  mace supercell graphene_slab_sp.out 3 3 1
  mace supercell graphene_slab_sp.out 1 1 2 --vacuum 12 --format extxyz
  mace supercell water.xyz 2 1 1 --vacuum crystal
  mace supercell hBN.cif --matrix "2 1 0  -1 1 0  0 0 1"
"""


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mace supercell", description=__doc__.split("\n\n")[1].replace("\n", " "),
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=USAGE_EPILOG)
    p.add_argument("input", help="a structure file (.cif, .d12, .out, .f34/.gui, POSCAR, .xyz, .extxyz)")
    p.add_argument("n", nargs="*", type=int, metavar="N",
                   help="n1 n2 n3 (default 2 along periodic directions, 1 along non-periodic ones)")
    p.add_argument("--matrix", help='a 3x3 integer matrix, rows = new a, b, c in old units ("2 0 0 0 2 0 0 0 1")')
    p.add_argument("--vacuum", default="bbox",
                   help="spacing of repeats along a non-periodic direction: bbox (default), file, crystal, open, "
                        "or a number of Å (custom)")
    p.add_argument("--format", choices=("cif", "extxyz"), default="cif",
                   help="cif (default; a slab, polymer or molecule gets a 15 Å vacuum box) or extxyz (keeps pbc)")
    p.add_argument("-o", "--output", help="the output file (default <stem>_<n1>x<n2>x<n3>.<format> beside the input)")
    p.add_argument("--force", action="store_true", help="overwrite an existing output")
    return p


def _vacuum_mode(text: str):
    t = str(text).strip().lower()
    if t in ("bbox", "open", "crystal"):
        return t, 10.0
    if t in ("file", "cell"):
        return "cell", 10.0
    try:
        gap = float(t)
    except ValueError:
        raise ValueError(f"--vacuum {text!r}: bbox, file, crystal, open or a number of Å") from None
    if not 0 <= gap <= 1000:
        raise ValueError("--vacuum: a custom vacuum is 0 to 1000 Å")
    return "gap", gap


def build(path: Path, n: Optional[List[int]] = None, matrix=None, vacuum: str = "bbox"):
    """(ase.Atoms of the supercell, its label, {repeats, vacuum vectors}) for a structure file."""
    import numpy as np
    from ase import Atoms
    from mace.gui import builders as B
    from mace.gui import structures as S
    p = S.read_structure(Path(path))
    pbc = [bool(x) for x in p["pbc"]]
    mode, gap = _vacuum_mode(vacuum)
    if matrix is not None:
        try:
            m = B.check_matrix(np.asarray([float(x) for x in str(matrix).replace(",", " ").split()]).reshape(3, 3))
        except ValueError as exc:
            raise ValueError(f"--matrix: {exc}") from None
    else:
        n = list(n or [])
        if n and len(n) != 3:
            raise ValueError("give n1 n2 n3 (three whole numbers) or --matrix")
        n = n or [2 if pbc[k] else 1 for k in range(3)]
        if min(n) < 1:
            raise ValueError("n1 n2 n3 are whole numbers from 1")
        m = B.check_matrix(np.diag(n))
    asked = [k for k in range(3) if not pbc[k] and m[k, k] > 1]
    if not any(pbc) and not asked:
        raise ValueError("a molecule repeats only at a vacuum spacing: give N1 N2 N3 with more than 1 "
                         "along x, y or z (e.g. 2 1 1)")
    vecs = None
    if asked:
        if mode == "open":
            raise ValueError(f"the open vacuum repeats nothing along {', '.join('abc'[k] for k in asked)}; "
                             "pick bbox, file, crystal or a number of Å")
        vecs = [S.vacuum_vector(p["cell"], p["cart"], k, mode, gap, p.get("vacuum_vectors")).tolist()
                if k in asked else None for k in range(3)]
    r = B.supercell(cell=np.asarray(p["cell"], float), frac=np.asarray(p["frac"], float), numbers=p["numbers"],
                    pbc=tuple(pbc), matrix=m, vacuum=vecs)
    atoms = Atoms(numbers=[int(z) % 100 or int(z) for z in r["numbers"]], cell=r["cell"], scaled_positions=r["frac"],
                  pbc=r["pbc"])
    if not all(pbc):          # the file's vacuum vectors, a repeated axis's the new period (an extxyz writes them)
        atoms.info.update(S.repeated_info(S.nonperiodic_info(p), r["info"]["repeats"], r["info"]["spacing"]))
    full = np.asarray(r["info"]["matrix"])
    diag = not np.count_nonzero(full - np.diag(np.diag(full)))
    label = "x".join(str(int(full[k, k])) for k in range(3)) if diag else f"sc{int(r['info']['det'])}"
    return atoms, label, {"repeats": r["info"].get("repeats", [1, 1, 1]), "vacuum": vecs, "mode": mode}


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    src = Path(args.input)
    if not src.is_file():
        sys.stderr.write(f"mace supercell: error: {src} is not a file\n")
        return 2
    try:
        import ase  # noqa: F401
    except ImportError:
        sys.stderr.write("mace supercell needs ase: pip install ase\n")
        return 1
    try:
        atoms, label, info = build(src, args.n, args.matrix, args.vacuum)
    except ValueError as exc:
        sys.stderr.write(f"mace supercell: error: {exc}\n")
        return 2
    out = Path(args.output) if args.output else src.with_name(f"{src.stem}_{label}.{args.format}")
    if out.exists() and not args.force:
        sys.stderr.write(f"mace supercell: error: {out} exists (--force to overwrite)\n")
        return 2
    import io
    import warnings
    import ase.io
    from mace.gui import structures as S
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if args.format == "cif":
            raw = io.BytesIO()                       # ase's CIF writer wants bytes
            ase.io.write(raw, S.boxed(atoms), format="cif")
            text = raw.getvalue().decode()
        else:
            buf = io.StringIO()
            ase.io.write(buf, S.vacuum_cell(atoms), format="extxyz")
            text = buf.getvalue()
    out.write_text(text)
    dim = S._DIM_NAMES[int(sum(bool(x) for x in atoms.pbc))]
    msg = f"{out}: {len(atoms)} atoms ({label}, {dim})"
    if info["vacuum"]:
        msg += "; repeated along " + ", ".join(
            f"{'abc'[k]} at {float(sum(x * x for x in v)) ** 0.5:.2f} Å" for k, v in enumerate(info["vacuum"]) if v)
        msg += f" ({'custom' if info['mode'] == 'gap' else info['mode']} vacuum)"
    if dim != "CRYSTAL" and args.format == "cif":
        msg += f"; the CIF has a {S.SAVE_VACUUM:g} Å vacuum box (--format extxyz keeps the pbc)"
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
