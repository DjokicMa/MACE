"""``mace preopt``: pre-optimize a structure into a starting geometry.

Input: a CIF, a .d12 deck (CRYSTAL, or a group-1 SLAB, POLYMER or MOLECULE
deck: 1.1.6), extended XYZ (its pbc kept, e.g. pbc="T T F" for a slab), XYZ
(a molecule), POSCAR/CONTCAR/.vasp or a CRYSTAL .out (its last geometry; the
way to start from a SLAB, POLYMER or MOLECULE deck in a group other than 1,
which MACE does not expand). A slab, polymer or molecule keeps its
periodicity: nothing wraps or sees an image along a non-periodic axis, only
the periodic cell directions relax, and a molecule keeps its box. Its result
is also written as ``<stem>_preopt.extxyz`` (the exact structure and pbc);
the CIF holds it in a box with 15 Å of vacuum, P1.

Writes ``OUTDIR/<stem>_preopt.cif`` (the input's space group and setting
when the run kept it, else P1 with a warning) and ``<stem>_preopt.json``
(provenance). Geometry only: OPTGEOM, the functional, the basis, the
tolerances, SPIN and the charge/spin keywords live in the user's options
and are never read or rewritten here.

Exit codes: 0 the requested method's result (auto: whatever ran) was
written; 3 an explicit gfnff/mlip run was refused, failed or rejected and the
spring result was written instead; 1 nothing was written; 2 usage error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

FOLLOW_UP = ("This is a starting geometry, not a result. Run a CRYSTAL OPT from it before any SP, FREQ, BAND or "
             "DOSS: `mace convert` on {cif} with your options (calculation type OPT).")


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, f"mace preopt: error: {message}\n")


def _parser() -> argparse.ArgumentParser:
    p = _Parser(prog="mace preopt", description="Pre-optimize a structure into a starting geometry for CRYSTAL.")
    p.add_argument("input")
    p.add_argument("--method", default="springs")
    p.add_argument("-o", "--output-dir", dest="outdir")
    p.add_argument("--force", action="store_true")
    p.add_argument("--relax-cell", action="store_true")
    p.add_argument("--no-symmetry", action="store_true")
    p.add_argument("--symprec", type=float, default=1e-3)
    p.add_argument("--symmetrize", type=float, metavar="TOL")
    p.add_argument("--charge", type=int)
    p.add_argument("--uhf", type=int, default=0)
    p.add_argument("--bonds", choices=("shown", "lenient", "none"), default="shown")
    p.add_argument("--fmax", type=float)
    p.add_argument("--steps", type=int)
    p.add_argument("--time-limit", type=float)
    p.add_argument("--threads", type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-fallback", action="store_true")
    p.add_argument("--json", action="store_true")
    return p


def _err(msg: str) -> None:
    sys.stderr.write(msg.rstrip() + "\n")


def read_input(path: Path):
    """(atoms, spacegroup or None, kinds or None, deck scan or None)."""
    import warnings
    suffix = path.suffix.lower()
    if suffix == ".cif":
        from mace.utils.cif_symbols import read_cif
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atoms = read_cif(path)
        # Fd-3m without listed operators: the origin cif2d12 writes, one rule.
        from mace.gui.structures import cif_deck_origin
        atoms = cif_deck_origin(path, atoms)
        return atoms, atoms.info.get("spacegroup"), atoms.arrays.get("spacegroup_kinds"), None
    if suffix == ".d12":
        from . import deckscan
        scan = deckscan.scan(path.read_text(errors="replace"))
        if scan["geometry_edits"]:
            return None, None, None, scan
        from mace.gui.structures import d12_atoms
        atoms, _ = d12_atoms(path)
        return atoms, atoms.info.get("spacegroup"), atoms.arrays.get("spacegroup_kinds"), scan
    low = path.name.lower()
    if suffix in (".xyz", ".extxyz", ".vasp", ".poscar") or low.startswith(("poscar", "contcar")):
        from ase.io import read
        fmt = "vasp" if suffix in (".vasp", ".poscar") or low.startswith(("poscar", "contcar")) else "extxyz"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atoms = read(str(path), format=fmt, index=-1)
        if atoms.cell.rank == 0:
            atoms.pbc = False                  # a plain XYZ: a molecule, no box
        if any(atoms.pbc) and atoms.cell.rank < 3:
            raise ValueError(f"{path.name}: periodic along {''.join('abc'[i] for i in range(3) if atoms.pbc[i])} "
                             "but its cell has fewer than three vectors")
        return atoms, None, None, None
    if suffix in (".out", ".o", ".log"):
        # a CRYSTAL output: its last geometry, through the GUI's reader (as mace calc and the GUI's
        # Clean read it): a SLAB, POLYMER or MOLECULE run in any group keeps its pbc
        from mace.calculators.structio import atoms_from_payload
        from mace.gui.structures import read_structure
        return atoms_from_payload(read_structure(path, -1)), None, None, None
    raise ValueError(f"{path.name}: mace preopt reads .cif, .d12, .xyz/.extxyz, POSCAR and CRYSTAL .out files")


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        from . import core
        kind = core.method_kind(args.method)
    except ValueError as exc:
        _err(f"mace preopt: error: {exc}")
        return 2
    if args.method.startswith("mlip:"):
        from . import registry
        if registry.get(args.method[5:]) is None:
            _err(f"mace preopt: error: {registry.unknown(args.method[5:], prefix='mlip:')}")
            return 2
    try:
        from . import mlip_client
        mlip_client.device_mapping(args.device)
    except ValueError as exc:
        _err(f"mace preopt: error: {exc}")
        return 2
    for name, v in (("--fmax", args.fmax), ("--time-limit", args.time_limit), ("--symprec", args.symprec),
                    ("--symmetrize", args.symmetrize)):
        if v is not None and not v > 0:
            _err(f"mace preopt: error: {name} must be positive")
            return 2
    if (args.steps is not None and args.steps < 1) or (args.threads is not None and args.threads < 1):
        _err("mace preopt: error: --steps and --threads must be at least 1")
        return 2
    try:
        import ase  # noqa: F401
        import spglib  # noqa: F401
    except ImportError:
        _err("mace preopt needs ase and spglib: pip install ase spglib")
        return 1

    src = Path(args.input)
    if not src.is_file():
        _err(f"mace preopt: {src} is not a file")
        return 1
    try:
        atoms, sg, kinds, scan = read_input(src)
    except Exception as exc:
        _err(f"mace preopt: cannot read {src.name}: {exc}")
        return 1
    from . import cifout, deckscan, marker
    if scan and scan["geometry_edits"]:
        _err(deckscan.refusal_text(src.name, scan["geometry_edits"]))
        return 1
    charge = args.charge
    if charge is None:
        if scan and scan["charge_records"] and kind != "springs":
            _err(deckscan.charged_text(scan["charge_records"]))
            return 1
        charge = 0

    outdir = Path(args.outdir) if args.outdir else src.resolve().parent
    stem = src.stem
    cif_out, json_out = outdir / f"{stem}_preopt.cif", outdir / f"{stem}_preopt.json"
    low_dim = not all(bool(x) for x in atoms.pbc)
    xyz_out = outdir / f"{stem}_preopt.extxyz" if low_dim else None
    exists = [p.name for p in (cif_out, json_out, xyz_out) if p is not None and p.exists()]
    if exists and not args.force:
        _err(f"mace preopt: {', '.join(exists)} already exist in {outdir} (pass --force to overwrite)")
        return 1

    tty = sys.stderr.isatty()
    last = [0.0]

    def progress(frac, text):
        now = time.monotonic()
        if tty and (now - last[0] > 0.5 or frac >= 1.0):
            last[0] = now
            sys.stderr.write(f"\r  {text[:60]:60s} {100 * frac:5.1f}%")
            if frac >= 1.0:
                sys.stderr.write("\n")
            sys.stderr.flush()

    from mace.gui import clean as C
    try:
        r = core.preoptimize(atoms, args.method, charge=charge, uhf=args.uhf, keep_symmetry=not args.no_symmetry,
                             symprec=args.symprec, symmetrize_tol=args.symmetrize, relax_cell=args.relax_cell,
                             fmax=args.fmax, steps=args.steps, time_limit=args.time_limit, threads=args.threads,
                             device=args.device, progress=progress, fallback=not args.no_fallback, route="cli",
                             explicit_time_limit=args.time_limit is not None, bonds_held=args.bonds)
    except C.NothingToClean as exc:
        _err(f"mace preopt: {exc}; nothing was written")
        return 1
    except core.NoResult as exc:
        _err(f"mace preopt: {args.method} gave no accepted result ({exc}); nothing was written (--no-fallback)")
        return 1
    except KeyboardInterrupt:
        _err("mace preopt: interrupted; nothing was written")
        return 130

    outdir.mkdir(parents=True, exist_ok=True)
    info = r.tier_info or {}
    lic = info.get("code_license") or info.get("license")
    if r.method_ran == "springs":
        label, lic = "the MACE spring model", None
    else:
        label = r.label_ran
        if info.get("weights_license") and info.get("weights_license") != lic:
            lic = f"code {lic}, weights {info['weights_license']}"
    header = [marker.comment(label, lic), marker.line(r.method_ran, info.get("model_id"),
                                                      academic=bool(info.get("academic")) and r.method_ran != "springs")]
    if info.get("unrelated_note"):
        header.insert(1, f"# {info['unrelated_note']}")
    keep = core.same_group(r.sg_final, r.sg_ref)
    if low_dim:
        from mace.gui.structures import SAVE_VACUUM, boxed
        from ase.io import write as awrite
        dim = ("molecule", "polymer", "slab")[sum(bool(x) for x in r.atoms.pbc)]
        mode = cifout.write(cif_out, boxed(r.atoms), name=stem, header=header, keep_group=False,
                            p1_reason=f"a {dim}: written in a box with {SAVE_VACUUM:g} Å of vacuum")
        out_atoms = r.atoms.copy()
        out_atoms.calc = None
        out_atoms.info = {}
        awrite(str(xyz_out), out_atoms, format="extxyz")
    else:
        mode = cifout.write(cif_out, r.atoms, name=stem, header=header, spacegroup=sg, kinds=kinds, keep_group=keep,
                            p1_reason=None)
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    prov = core.provenance(r, input_name=src.name, input_sha256=sha, output_mode=mode)
    prov["output"] = cif_out.name
    if xyz_out is not None:
        prov["output_extxyz"] = xyz_out.name
    json_out.write_text(json.dumps(prov, indent=2, default=str) + "\n")

    say = sys.stderr.write if args.json else sys.stdout.write
    say(f"Ran: {r.summary}\n")
    if r.method_ran == "gfnff" and info.get("source"):
        say(f"Used {info['source']}\n")
    if len(r.attempts) > 1 or r.fell_back:
        for a in r.attempts:
            extra = f" - {a['reason']}" if a["reason"] else ""
            say(f"  {a['method']}: {a['status']}{extra}\n")
    for w in r.warnings:
        say(f"warning: {w}\n")
    if mode["mode"] == "p1" and low_dim:
        say(f"note: {mode['reason']}; {xyz_out.name} keeps the structure and its periodicity exactly\n")
    elif mode["mode"] == "p1":
        raised = (r.symmetrize and not core.same_group(r.sg_ref, r.sg_input) and r.sg_ref
                  and r.sg_ref["number"] > 1 and keep)
        if raised:
            # symmetrize-first gave the group the user asked for, but the CIF can only be P1 here
            # (the input's own group was P1 or is gone): say how the deck gets that group
            g = core.sg_name(r.sg_ref)
            say(f"warning: wrote a P1 CIF ({mode['reason']}) of a structure in {g}: cif2d12's default "
                f"symmetry_handling CIF would build a P1 deck from it; set symmetry_handling SPGLIB in your "
                f"options for a deck in {g}\n")
        else:
            say(f"warning: wrote a P1 CIF ({mode['reason']}): cif2d12 will build a P1 deck; check "
                "symmetry_handling in your options\n")
    say(f"Wrote {cif_out}" + (f", {xyz_out.name}" if xyz_out is not None else "") + f" and {json_out.name}\n")
    follow = FOLLOW_UP.format(cif=cif_out.name)
    if src.suffix.lower() == ".d12":
        follow += f" Use the options you saved for {src.name}."
    if r.symmetrize and r.symmetrize.get("text"):
        follow += " Note: the space group was " + r.symmetrize["text"]
    say(follow + "\n")
    if r.method_ran != "springs" and info.get("citation"):
        say("Cite: " + "; ".join(c["text"] for c in info["citation"]) + "\n")
    if args.json:
        sys.stdout.write(json.dumps(prov, indent=2, default=str) + "\n")
    if kind in ("gfnff", "mlip") and r.method_ran == "springs":
        _err(f"mace preopt: {args.method} did not give an accepted result; the spring result was written instead")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
