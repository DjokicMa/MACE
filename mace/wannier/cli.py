"""``mace wannier`` - drive the LCAO->Wannier90 conversion of a matrix dump.

Deliberately a user-invoked step rather than an automatic workflow progression.
Layers 2 and 3 (lcao2wannier, wannier90.x) may simply be absent on the cluster
that ran the dump, so the MACE-managed chain ends at MATDUMP and conversion is
asked for explicitly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from mace.wannier.driver import (
    LCAO2WANNIER_CREDIT,
    Lcao2WannierUnavailable,
    convert,
    describe_missing_wannier90,
    find_lcao2wannier,
)

try:
    from mace.utils import ui
except Exception:  # pragma: no cover - standalone fallback
    class _Shim:
        @staticmethod
        def info(msg=""):
            print(msg)

        @staticmethod
        def warn(msg=""):
            print(msg, file=sys.stderr)

        @staticmethod
        def err(msg=""):
            print(msg, file=sys.stderr)

        @staticmethod
        def ok(msg=""):
            print(msg)

        @staticmethod
        def rule(msg=""):
            print(f"--- {msg} ---")

    ui = _Shim()


EPILOG = f"""
Converts a MATDUMP matrix dump into Wannier90 hand-off files
(.win .nnkp .eig .amn .mmn).

{LCAO2WANNIER_CREDIT}.
The LCAO->Wannier90 method and the lcao2wannier package are his work. MACE
generates the CRYSTAL deck and orchestrates the run; it implements none of the
conversion. CITATION: ask William Comaskey which citation he wants before
publishing a Wannier model produced this way.

lcao2wannier is an optional dependency (pip install lcao2wannier); wannier90.x
is user-supplied upstream software. MACE bundles neither.

Examples:
  mace wannier --input material_matdump.out
  mace wannier --input material_matdump.out --seed material --threads 8
  mace wannier --input material_matdump.out --wannier90 /path/to/wannier90.x
"""


def _emit(text: str, emitter) -> None:
    for line in str(text).splitlines():
        emitter(line) if line.strip() else print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mace wannier",
        description="Wannier90 hand-off from a CRYSTAL matrix dump (MATDUMP).",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True,
                        help="MATDUMP properties output containing H(R) and S(R)")
    parser.add_argument("--seed", help="output basename (default: the dump's stem)")
    parser.add_argument("--output-dir", help="output directory "
                                             "(default: <seed>.wannier beside the dump)")
    parser.add_argument("--stage", default="all",
                        choices=["all", "prepare", "overlaps", "hybrid", "localize"],
                        help="lcao2wannier run stage (default: all)")
    parser.add_argument("--threads", type=int,
                        help="CPU budget; omit inside a SLURM job, where "
                             "lcao2wannier reads the allocation itself")
    parser.add_argument("--wannier90",
                        help="path to wannier90.x, to localize after the hand-off")
    parser.add_argument("--l2w-arg", action="append", dest="l2w_args", metavar="ARG",
                        help="pass one raw argument through to lcao2wannier "
                             "(repeatable). MACE does not validate or interpret it, "
                             "and never synthesizes one: his defaults for --method, "
                             "--mmn-method, --hybrid-trust and --k-grid are "
                             "deliberately left alone.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    available, version = find_lcao2wannier()
    if available:
        ui.info(f"lcao2wannier {version or 'unknown version'} "
                f"({LCAO2WANNIER_CREDIT})")

    if args.stage == "localize" and not args.wannier90:
        _emit(describe_missing_wannier90(), ui.err)
        return 2

    try:
        result = convert(
            parent=Path(args.input),
            seed=args.seed,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            stage=args.stage,
            threads=args.threads,
            wannier90=Path(args.wannier90) if args.wannier90 else None,
            extra_args=args.l2w_args,
        )
    except Lcao2WannierUnavailable as exc:
        # Every refusal reaches the user as its own text, never a traceback.
        _emit(str(exc), ui.err)
        return 2

    # His diagnostics go to the user whole. He is the one who knows what they
    # mean, and summarizing them is how a reason gets lost.
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, file=sys.stderr,
              end="" if result.stderr.endswith("\n") else "\n")

    print()
    ui.rule("Wannier90 hand-off")
    ui.info(f"output directory: {result.output_dir}")
    for seed, present in sorted(result.produced.items()):
        missing = [s for s in (".win", ".nnkp", ".eig", ".amn", ".mmn")
                   if s not in present]
        if missing:
            ui.warn(f"  {seed}: missing {', '.join(missing)}")
        else:
            ui.ok(f"  {seed}: .win .nnkp .eig .amn .mmn")
    if not result.produced:
        ui.warn("  no hand-off files were written")

    if result.audit == "fail":
        ui.err("")
        ui.err("The conversion's own self-audit REFUSED this model:")
        _emit(result.audit_text, ui.err)
        ui.err("")
        ui.err("lcao2wannier reports a violation without aborting, so the files")
        ui.err("above may exist despite the refusal. Treat this calculation as")
        ui.err("FAILED and fix what the audit names before using the result.")
        return 1

    if result.audit == "unknown":
        ui.warn("")
        ui.warn("No self-audit verdict was found in the conversion output, so the")
        ui.warn("model is NOT verified. Read the full output above before using it.")

    if not result.ok:
        ui.err("")
        ui.err(f"lcao2wannier exited {result.returncode} without a complete hand-off.")
        return result.returncode or 1

    ui.ok("")
    ui.ok("Hand-off complete. Next: run wannier90.x on the .win, or re-run with")
    ui.ok("--wannier90 PATH to have lcao2wannier localize for you.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
