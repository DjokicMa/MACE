"""``mace wannier`` - drive the LCAO->Wannier90 conversion of a matrix dump.

Deliberately a user-invoked step rather than an automatic workflow progression.
The conversion (the bundled lcao2wannier) is a separate, memory-hungry step and
Layer 3 (wannier90.x) may simply be absent on the cluster that ran the dump, so
the MACE-managed chain ends at MATDUMP and conversion is asked for explicitly.

This is also the bundled package's command-line entry: MACE runs it as
``python -m mace.wannier.lcao2wannier`` and ``--l2w-arg`` passes any of its own
options through. There is no second console script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from mace.wannier.driver import (
    LCAO2WANNIER_CREDIT,
    LCAO2WANNIER_VERSION,
    Lcao2WannierUnavailable,
    convert,
    describe_missing_wannier90,
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

lcao2wannier is bundled with MACE (MIT; its local fixes are listed in
mace/wannier/lcao2wannier/VENDORED.md). wannier90.x is user-supplied upstream
software; MACE never bundles it.

Examples:
  mace wannier --input material_matdump.out
  mace wannier --input material_matdump.out --seed material --threads 8
  mace wannier --input material_matdump.out --wannier90 /path/to/wannier90.x
"""


def _emit(text: str, emitter) -> None:
    for line in str(text).splitlines():
        emitter(line) if line.strip() else print()


def _echo_child(line: str, stream: str = "stdout") -> None:
    """Pass one line of his output through as it arrives, unaltered."""
    print(line, file=sys.stderr if stream == "stderr" else sys.stdout,
          flush=True)


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

    ui.info(f"lcao2wannier {LCAO2WANNIER_VERSION}, bundled "
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
            echo=lambda line, stream="stdout": _echo_child(line, stream),
            progress=ui.info,
        )
    except Lcao2WannierUnavailable as exc:
        # Every refusal reaches the user as its own text, never a traceback.
        _emit(str(exc), ui.err)
        return 2

    # His diagnostics go to the user whole. He is the one who knows what they
    # mean, and summarizing them is how a reason gets lost.
    # Already printed line by line while it ran when streamed.
    if result.stdout and not result.streamed:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr and not result.streamed:
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

    if result.audit == "marginal":
        ui.warn("")
        ui.warn("lcao2wannier QUALIFIED this model: its overlap-conditioning check")
        ui.warn("rated the matrices moderately ill-conditioned. His verdict is not")
        ui.warn("a refusal - but it is the downstream symptom of too few surviving")
        ui.warn("R-vectors, which is the one failure this calc type exists to catch.")
        ui.warn("His report, whole:")
        ui.warn("")
        _emit(result.conditioning_text or result.audit_text, ui.warn)
        ui.warn("")
        ui.warn("If you want a cleaner model, re-run the parent SCF at a tighter")
        ui.warn("TOLINTEG (which raises the R-vector count CRYSTAL keeps) and dump")
        ui.warn("again. MATDUMP already uses the largest N that run supports.")

    if result.audit == "unknown":
        ui.warn("")
        ui.warn("No self-audit verdict was found in the conversion output, so the")
        ui.warn("model is NOT verified. Read the full output above before using it.")

    if not result.ok:
        ui.err("")
        ui.err(f"lcao2wannier exited {result.returncode} without a complete hand-off.")
        return result.returncode or 1

    ui.ok("")
    if result.qualified:
        # Never an unqualified "complete" over a caveat he raised.
        ui.ok("Hand-off written, WITH the conditioning caveat above. Next: run")
        ui.ok("wannier90.x on the .win, or re-run with --wannier90 PATH to have")
        ui.ok("lcao2wannier localize for you - and check the spreads against the")
        ui.ok("caveat before trusting the interpolated bands.")
    else:
        ui.ok("Hand-off complete. Next: run wannier90.x on the .win, or re-run with")
        ui.ok("--wannier90 PATH to have lcao2wannier localize for you.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
