# Copyright (c) 2025 Computational Materials Science Team (lcao2wannier, author William Comaskey).
# MIT License, see LICENSE in this directory.
# Vendored into MACE from lcao2wannier 1.0.0; local changes are listed in VENDORED.md.
"""Canonical ``lcao2wannier`` command-line driver.

The scientific workflow remains in :mod:`lcao2wannier.workflow`. This module
resolves the public interface before invoking that workflow, so validation,
capability checks, and dry runs cannot start a calculation or create outputs.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterator, Sequence


NAMED_STAGE_MAP = {"all": "all", "prepare": "1", "overlaps": "2", "hybrid": "all"}
NUMERIC_STAGE_LABELS = {
    "1": "legacy stage 1",
    "2": "legacy stage 2",
    "3": "legacy stage 3 (H(R) postprocess)",
    "4": "legacy stage 4 (band plot)",
}
HANDOFF_SUFFIXES = (".win", ".nnkp", ".eig", ".amn", ".mmn")
INCOMPLETE_SUFFIX = ".lcao2wannier.incomplete"
BLAS_THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)

_ADVANCED_HELP = """
Advanced numerical controls accepted and forwarded unchanged:
  --window E_MIN E_MAX            outer window in eV relative to E_F
  --fermi-energy EV               override the reported Fermi energy
  --mmn-method METHOD             analytic (default), midpoint, lowdin, or
                                  lowdin_no_berry overlap algorithm
  --gto-cutoff BOHR               analytic-GTO pair cutoff (default: 16 Bohr)
  --memory {fast,low}             parser/matrix memory strategy (default: fast)
  --solve-nbands N                lowest bands solved at each k point
  --spin {alpha,beta,both}        collinear channel policy
  --proj-threshold P              projectability threshold (default: 0.9)
  --projection-method {weight,scdm} (default: weight)
  --extended, --include-tm-p      expand automatic target channels
  --hybrid-p-froz P               derived automatically (manual default: 0.95)
  --hybrid-p-floor P              admission floor (default: 0.10)
  --hybrid-p-froz-band P          derived automatically (manual default: 0.85)
  --hybrid-trust {auto,localization,manual} (default: auto)
  --hybrid-thr-floor P            auto threshold clip (default: 0.55)
  --hybrid-trust-margin P         auto band-gate margin (default: 0.015)
  --hybrid-pool-cap FACTOR        pool size floor as a multiple of num_wann;
                                  grows only to cover the seed span
                                  (default: 1.5; name is historical)
  --hybrid-fid-emax EV            fidelity ceiling (default: 8 eV above E_F)
  --hybrid-froz-emax auto|none|EV frozen ceiling: freeze every pool state below it
                                  (default: auto, walked to fit num_wann)
  --hybrid-dis-niter N            maximum sweeps (default: 2000)
  --hybrid-dis-tol TOL            absolute tolerance (default: 1e-9 Ang^2)
  --hybrid-dis-rel-tol TOL        relative tolerance (default: 1e-10)
  --hybrid-dis-check-every N      sweeps between checks (default: 1)
  --hybrid-dis-mix FACTOR         fixed relaxation (default: 1.0)
  --hybrid-dis-mix-schedule PEAK opening relaxation schedule
  --hybrid-dis-taper FACTOR       schedule taper factor (default: 0.5)
  --hybrid-dis-taper-on {check,stall} (default: check)
  --hybrid-dis-taper-reset-history
                                  reset convergence history after taper
  --hybrid-dis-relax-late FACTOR  late-sweep relaxation
  --hybrid-require-capacity       fail on insufficient frozen capacity
  --hybrid-guiding-centres        write SCDM anchor centres to .win
  --augment-coverage-min P        augmentation trigger (default: 0.70)
  --augment-retention-floor P     retention floor (default: 0.40)
  --hybrid-report                 print projectability anatomy
  --dump-masks PATH.npz           write per-state hybrid masks
  --dump-handoff PATH.npz         write isolated Bloch hand-off
  --hybrid-report-json PATH.json  write projectability anatomy JSON
  --hybrid-dis-log PATH           stream convergence data
  --hybrid-gauge-rank-tol TOL     scale-free SCDM tolerance (default: 1e-10)
  --hybrid-gauge-min-sv SIGMA     calibrated absolute floor (default: off)
  --hybrid-shell-floor-budget P   shell-floor fraction budget (default: 0.01)
  --hybrid-shell-p-floor P|auto   shell-floor threshold (default: auto)
  --baseline {dis-froz-proj,dis-froz-proj-svd}
                                  stock-Wannier90 comparison baseline
  --dis-proj-min P, --dis-proj-max P
                                  baseline thresholds (defaults: 0.01/0.95)
  --pdwf-first-radial-only        one radial per target channel
  --pdwf-p-high P|auto, --pdwf-p-low P|auto
                                  PDWF thresholds (default: auto = the band
                                  gate and junk cliff the hybrid rule derives)
  --auto-window                   Omega_I-optimal disentanglement windows
                                  (implied by --method window)
  --spread-window-assist          refine windows for localization
  --min-froz-window LO HI         required coverage (default: -6 3 eV rel E_F)
  --assist-proj-floor P           diffuse-band exclusion threshold
  --conditioning {none,svd,soft} MMN conditioning (default: none)
  --conditioning-knee P           soft-conditioning knee
  --no-internal-nnkp              require external -pp preprocessing
  --no-prune, --prune-threshold TOL
                                  R-vector controls (threshold default: 0)
  --bands-plot                    request Wannier90 band output
  --projections PROJ [...]        explicit projection cards
  --frozen-conduction {auto,off}  PDWF policy (default: auto)
  --conduction-pmin P             conduction floor (default: 0.70)
  --window-mode {manifold,spread} policy (default: manifold)
  --no-parallel                   legacy alias for --parallel serial
  --hr-file PATH                  legacy stage-3 input
  --output PATH                   legacy stage-3 output
  --no-hermitize, --no-time-reversal
                                  legacy stage-3 symmetry switches
  --symm-threshold TOL            stage-3 threshold (default: 1e-9)
  --kpath TYPE, --npts N          stage-4 path (defaults: auto, 60)
  --energy-range LO HI            stage-4 plot range (default: -20 25 eV)
  --output-plot PATH              legacy stage-4 plot file
  --no-pdwf                       omit PDWF coloring in stage 4
  --custom-kpath SPEC             legacy stage-4 custom path
  --force                         accept a large direct-method target

Numeric stages retain the legacy meanings and method default (auto). The
compatibility script lcao_to_wannier90.py retains its original required
arguments. Named stages use the canonical hybrid default. Stages 3 and 4 are
legacy postprocess/plot operations; they are not aliases for hybrid.
"""


class _HelpfulParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = _HelpfulParser(
        prog="lcao2wannier", add_help=False, allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Prepare Wannier90 hand-off files from a CRYSTAL/LCAO calculation.\n\n"
            "Examples:\n"
            "  lcao2wannier --input material.out --seed material --k-grid 8 8 8\n"
            "  lcao2wannier --input material.out --seed material --threads 8 "
            "--wannier90 /path/to/wannier90.x"
        ), epilog=_ADVANCED_HELP,
    )
    parser.add_argument("-h", "--help", action="store_true", help="show this complete reference and exit")
    io_group = parser.add_argument_group("Input and output")
    io_group.add_argument("--input", "-i", metavar="PATH", help="CRYSTAL output file; required for stages that prepare data")
    io_group.add_argument("--seed", metavar="NAME", help="output basename")
    io_group.add_argument("--seedname", "-s", metavar="NAME", help="deprecated compatibility alias for --seed")
    io_group.add_argument("--output-dir", metavar="PATH", help="output directory (default: ./SEED.wannier for named stages; current directory for numeric stages)")
    io_group.add_argument("--wannier90", metavar="PATH", help="external consumer; localizes after all/hybrid or at stage localize")
    selection = parser.add_argument_group("Automatic selection")
    selection.add_argument("--method", choices=("hybrid", "window", "pdwf", "auto", "projectability", "direct"), help="method (default: hybrid for named stages; auto for numeric stages)")
    selection.add_argument("--num-wann", type=int, metavar="N", help="override automatically selected target size")
    selection.add_argument("--k-grid", nargs=3, type=int, metavar=("NX", "NY", "NZ"), help="Monkhorst-Pack grid; default from CRYSTAL SHRINK FACT")
    stages = parser.add_argument_group("Run stages")
    stages.add_argument("--stage", default="all", choices=("all", "prepare", "overlaps", "hybrid", "localize", "1", "2", "3", "4"), help="all (default): hybrid hand-off; prepare: .win/.nnkp; overlaps: raw .eig/.amn/.mmn; hybrid: isolated hand-off; localize: consumer only; 1-4: legacy meanings")
    resources = parser.add_argument_group("Resources")
    resources.add_argument("--threads", type=int, metavar="N", help="CPU budget per rank; allocated without nested BLAS workers")
    resources.add_argument("--parallel", choices=("auto", "serial", "threads", "mpi"), default="auto", help="frontend mechanism (default: auto; frontend MPI unsupported)")
    resources.add_argument("--mpi-ranks", type=int, default=1, metavar="N", help="MPI ranks for external --wannier90 only (default: 1)")
    diagnostics = parser.add_argument_group("Diagnostics")
    diagnostics.add_argument("--check", action="store_true", help="inspect capabilities/input/consumer without science or output writes")
    diagnostics.add_argument("--dry-run", action="store_true", help="print resolved plan without running or writing output")
    verbosity = diagnostics.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", action="store_true", help="show resolved plan and full progress")
    verbosity.add_argument("--quiet", action="store_true", help="suppress routine progress; retain warnings/errors")
    diagnostics.add_argument("--version", action="store_true", help="show version, revision, and capabilities")
    return parser


def _source_revision() -> str:
    root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"], text=True, capture_output=True, check=False)
    except OSError:
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _version_text() -> str:
    try:
        from . import __version__
    except ImportError:
        __version__ = "unknown"
    return (f"lcao2wannier {__version__}\nsource revision: {_source_revision()}\n"
            "capabilities: frontend=serial,threads; frontend-mpi=no; wannier90-mpi=external")


def _scheduler_budget() -> int:
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        value = os.environ.get(name)
        if value:
            try:
                parsed = int(value)
            except ValueError:
                continue
            if parsed > 0:
                return parsed
    return max(1, os.cpu_count() or 1)


def _resolve_resources(parser: argparse.ArgumentParser, args: argparse.Namespace) -> tuple[str, int]:
    if args.threads is not None and args.threads <= 0:
        parser.error("--threads must be a positive integer")
    if args.mpi_ranks <= 0:
        parser.error("--mpi-ranks must be a positive integer")
    if args.parallel == "mpi":
        parser.error("LCAO frontend does not support MPI; use --parallel threads (or auto). External Wannier90 MPI uses --wannier90 with --mpi-ranks.")
    if args.parallel == "serial" and args.threads not in (None, 1):
        parser.error("--parallel serial conflicts with --threads greater than 1")
    for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "MV2_COMM_WORLD_SIZE"):
        try:
            enclosing_ranks = int(os.environ.get(name, "1"))
        except ValueError:
            continue
        if enclosing_ranks > 1:
            parser.error(
                f"refusing MPI-launched frontend: {name}={enclosing_ranks}; run "
                "lcao2wannier once outside the existing MPI launch and use "
                "--mpi-ranks only for the external Wannier90 consumer"
            )
    if args.mpi_ranks > 1:
        try:
            allocated_tasks = int(os.environ.get("SLURM_NTASKS", "0"))
        except ValueError:
            allocated_tasks = 0
        if allocated_tasks and args.mpi_ranks > allocated_tasks:
            parser.error(
                f"--mpi-ranks {args.mpi_ranks} exceeds SLURM_NTASKS={allocated_tasks}"
            )
    workers = 1 if args.parallel == "serial" else (args.threads or _scheduler_budget())
    return ("serial" if workers == 1 else "threads"), workers


def _resolve_seed(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str | None:
    if args.seed and args.seedname and args.seed != args.seedname:
        parser.error("conflicting --seed and --seedname values")
    seed = args.seed or args.seedname
    if seed is not None:
        if seed in ("", ".", "..") or Path(seed).name != seed or "/" in seed or "\\" in seed:
            parser.error("--seed must be a basename without directory components")
    return seed


def _resolved_output_dir(stage: str, seed: str | None, value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    if stage in NUMERIC_STAGE_LABELS:
        return Path.cwd().resolve()
    return (Path.cwd() / f"{seed or 'lcao2wannier'}.wannier").resolve()


def _stage_description(stage: str) -> str:
    labels = {
        "all": "all (prepare + overlaps + hybrid hand-off)",
        "prepare": "prepare (legacy stage 1)",
        "overlaps": "overlaps (raw parent emission; legacy stage 2 orchestration)",
        "hybrid": "hybrid (legacy stage all; recomputes prerequisites)",
        "localize": "localize (external Wannier90 consumer only)",
    }
    return labels.get(stage, NUMERIC_STAGE_LABELS.get(stage, stage))


def _configure_frontend_environment(workers: int) -> None:
    os.environ["LCAO2WANNIER_WORKERS"] = str(workers)
    for name in BLAS_THREAD_VARIABLES:
        os.environ[name] = "1"
    if workers == 1:
        os.environ["LCAO_SERIAL_MMN"] = "1"
    else:
        os.environ.pop("LCAO_SERIAL_MMN", None)


def _resolve_executable(value: str) -> str | None:
    path = Path(value).expanduser()
    if path.parent != Path("."):
        absolute = path.resolve()
        return str(absolute) if absolute.is_file() and os.access(absolute, os.X_OK) else None
    resolved = shutil.which(value)
    return str(Path(resolved).resolve()) if resolved else None


def _resolve_mpi_launcher() -> str | None:
    configured = os.environ.get("MPIEXEC")
    return ((_resolve_executable(configured) if configured else None)
            or _resolve_executable("mpiexec") or _resolve_executable("mpirun"))


def _print_plan(args: argparse.Namespace, seed: str | None, output_dir: Path, mechanism: str, workers: int, extras: Sequence[str]) -> None:
    method = args.method or ("auto" if args.stage in NUMERIC_STAGE_LABELS else "hybrid")
    print("lcao2wannier resolved plan")
    print(f"  Stage: {_stage_description(args.stage)}")
    print(f"  Method: {method}")
    print(f"  Input: {Path(args.input).expanduser().resolve() if args.input else 'not supplied'}")
    print(f"  Seed: {seed or 'not supplied'}")
    print(f"  Output directory: {output_dir}")
    plural = "es" if workers != 1 else ""
    print(f"  Frontend: {mechanism} ({workers} worker process{plural}, 1 BLAS thread each)")
    print(f"  Wannier90: {args.wannier90 or 'not requested'}")
    print(f"  Wannier90 MPI ranks: {args.mpi_ranks}")
    if args.k_grid:
        print(f"  K grid: {' '.join(str(n) for n in args.k_grid)}")
    if args.num_wann is not None:
        print(f"  Number of Wannier functions: {args.num_wann}")
    if extras:
        print(f"  Advanced arguments: {' '.join(extras)}")


def _print_capability_check(args: argparse.Namespace, output_dir: Path) -> int:
    ok = True
    print("lcao2wannier capability check")
    if args.input:
        input_path = Path(args.input).expanduser().resolve()
        input_ok = input_path.is_file() and os.access(input_path, os.R_OK)
        print(f"  Input: {input_path} ({'readable' if input_ok else 'missing or unreadable'})")
        ok = ok and input_ok
    else:
        print("  Input: not supplied")
    print(f"  Output directory: {output_dir} (not created by --check)")
    print("  Frontend serial: supported")
    print("  Frontend threads: supported")
    print("  Frontend MPI: unsupported")
    print("  Wannier90 MPI: supported through --mpi-ranks and an MPI launcher")
    if args.wannier90:
        resolved = _resolve_executable(args.wannier90)
        print(f"  Wannier90: {resolved or 'missing or not executable'}")
        ok = ok and resolved is not None
    if args.mpi_ranks > 1:
        launcher = _resolve_mpi_launcher()
        print(f"  MPI launcher: {launcher or 'missing'}")
        ok = ok and launcher is not None
    return 0 if ok else 1


def _validate_common_values(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.num_wann is not None and args.num_wann <= 0:
        parser.error("--num-wann must be a positive integer")
    if args.k_grid and any(n <= 0 for n in args.k_grid):
        parser.error("--k-grid values must be positive integers")


def _same_file(left: Path, right: Path) -> bool:
    """Compare planned and existing paths, including symlinks/hard links."""
    if left.resolve() == right.resolve():
        return True
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def _validate_no_input_collision(
    parser: argparse.ArgumentParser, input_path: Path, output_dir: Path, seed: str,
    advanced: argparse.Namespace | None = None,
    external_consumer: bool = False,
) -> None:
    planned_seeds = (seed, f"{seed}_alpha", f"{seed}_beta")
    outputs: list[Path] = []
    for planned_seed in planned_seeds:
        for suffix in HANDOFF_SUFFIXES:
            outputs.append(output_dir / f"{planned_seed}{suffix}")
        outputs.append(output_dir / f"{planned_seed}{INCOMPLETE_SUFFIX}")
        outputs.append(output_dir / f"{planned_seed}_disentangle.log")
        if external_consumer:
            for suffix in (".wout", ".werr", ".chk", "_hr.dat", "_centres.xyz",
                           "_band.dat", "_band.gnu", "_band.kpt", "_band.labelinfo.dat"):
                outputs.append(output_dir / f"{planned_seed}{suffix}")

    source_hash = hashlib.sha256(str(input_path.resolve()).encode()).hexdigest()[:16]
    outputs.append(output_dir / f"{input_path.name}.{source_hash}.parsecache.pkl")

    if advanced is not None:
        def output_path(value: str) -> Path:
            path = Path(value).expanduser()
            return path.resolve() if path.is_absolute() else (output_dir / path).resolve()

        for name in ('dump_masks', 'dump_handoff', 'hybrid_report_json',
                     'hybrid_dis_log', 'output', 'output_plot'):
            value = getattr(advanced, name, None)
            if not value:
                continue
            base = output_path(value)
            outputs.append(base)
            if name in ('dump_masks', 'dump_handoff') and base.suffix != '.npz':
                outputs.append(Path(str(base) + '.npz'))
            root, ext = os.path.splitext(str(base))
            outputs.extend((Path(f"{root}_alpha{ext}"), Path(f"{root}_beta{ext}")))

        if getattr(advanced, 'stage', None) == 3:
            hr_file = getattr(advanced, 'hr_file', None) or f"{seed}_hr.dat"
            default_output = getattr(advanced, 'output', None) or f"{hr_file}_postprocessed"
            outputs.append(output_path(default_output))
        if getattr(advanced, 'stage', None) == 4 and not getattr(advanced, 'output_plot', None):
            outputs.append(output_dir / f"{seed}_bands.png")

    for output in outputs:
        if _same_file(input_path, output):
            parser.error(
                f"input/output collision: {input_path} would be overwritten by {output}"
            )


def _validate_input_structure(
    parser: argparse.ArgumentParser, input_path: Path, stage: str,
) -> None:
    required = {
        "CRYSTAL banner": False,
        "direct lattice": False,
        "basis size": False,
        "k grid": False,
        "overlap matrix": False,
        "Fock matrix": False,
    }
    try:
        with input_path.open("r", errors="replace") as stream:
            for line in stream:
                upper = line.upper()
                required["CRYSTAL banner"] |= "CRYSTAL" in upper
                required["direct lattice"] |= "DIRECT LATTICE VECTOR COMPONENTS" in upper
                required["basis size"] |= "NUMBER OF AO" in upper
                required["k grid"] |= "SHRINK. FACT.(MONKH.)" in upper
                required["overlap matrix"] |= "OVERLAP MATRIX - CELL" in upper
                required["Fock matrix"] |= "FOCK MATRIX" in upper and "CELL" in upper
                if stage == "3" and required["CRYSTAL banner"]:
                    return
                if all(required.values()):
                    return
    except OSError as exc:
        parser.error(f"cannot inspect input file {input_path}: {exc}")
    if stage == "3":
        missing = "CRYSTAL banner"
    else:
        missing = ", ".join(name for name, found in required.items() if not found)
    parser.error(f"input is not a complete supported CRYSTAL matrix output; missing: {missing}")


def _missing_handoff(output_dir: Path, seed: str, suffixes: Sequence[str]) -> list[str]:
    return [seed + suffix for suffix in suffixes if not (output_dir / (seed + suffix)).is_file()]


def _is_raw_parent_handoff(output_dir: Path, seed: str) -> bool:
    try:
        head = (output_dir / f"{seed}.win").read_text(errors="replace")[:512]
    except OSError:
        return False
    return "lcao2wannier stage: raw hybrid parent pool" in head


def _is_incomplete_generation(output_dir: Path, seed: str) -> bool:
    return (output_dir / f"{seed}{INCOMPLETE_SUFFIX}").exists()


def _resolve_localize_seeds(
    parser: argparse.ArgumentParser, output_dir: Path, seed: str,
) -> list[str]:
    base_missing = _missing_handoff(output_dir, seed, HANDOFF_SUFFIXES)
    if (not base_missing
            and not _is_incomplete_generation(output_dir, seed)
            and not _is_raw_parent_handoff(output_dir, seed)):
        return [seed]
    spin_seeds = [f"{seed}_alpha", f"{seed}_beta"]
    if all(not _missing_handoff(output_dir, candidate, HANDOFF_SUFFIXES)
           and not _is_incomplete_generation(output_dir, candidate)
           and not _is_raw_parent_handoff(output_dir, candidate)
           for candidate in spin_seeds):
        return spin_seeds
    if not base_missing and _is_raw_parent_handoff(output_dir, seed):
        parser.error(
            f"{seed} contains raw hybrid parent data, not a localizable "
            "hand-off; run --stage hybrid first")
    if not base_missing and _is_incomplete_generation(output_dir, seed):
        marker = output_dir / f"{seed}{INCOMPLETE_SUFFIX}"
        parser.error(
            f"{marker} marks an interrupted refresh, so this is not a complete hand-off; "
            "rerun --stage hybrid before localization")
    raw_spin = [candidate for candidate in spin_seeds
                if not _missing_handoff(output_dir, candidate, HANDOFF_SUFFIXES)
                and _is_raw_parent_handoff(output_dir, candidate)]
    if raw_spin:
        parser.error(
            "raw hybrid parent data are not localizable ("
            + ", ".join(raw_spin) + "); run --stage hybrid first")
    missing = base_missing
    parser.error(
        "--stage localize requires a complete hand-off for SEED, or complete "
        "SEED_alpha and SEED_beta hand-offs; missing base files: " + ", ".join(missing)
    )


def _mark_generation_started(output_dir: Path, seeds: Sequence[str], stage: str) -> None:
    for seed in seeds:
        marker = output_dir / f"{seed}{INCOMPLETE_SUFFIX}"
        marker.write_text(
            f"stage={stage}\n"
            "This marker prevents localization of files left by an interrupted refresh.\n"
        )


def _mark_handoff_complete(
    parser: argparse.ArgumentParser, output_dir: Path, seeds: Sequence[str],
) -> None:
    for seed in seeds:
        missing = _missing_handoff(output_dir, seed, HANDOFF_SUFFIXES)
        if missing:
            parser.error(
                f"workflow returned success without a complete {seed} hand-off; "
                "missing: " + ", ".join(missing)
            )
        if _is_raw_parent_handoff(output_dir, seed):
            parser.error(
                f"workflow returned raw parent files for {seed}; hybrid hand-off is incomplete"
            )
        try:
            (output_dir / f"{seed}{INCOMPLETE_SUFFIX}").unlink()
        except FileNotFoundError:
            pass


def _validate_handoff(parser: argparse.ArgumentParser, output_dir: Path, seed: str, *, stage: str) -> None:
    suffixes = (".win", ".nnkp") if stage == "overlaps" else HANDOFF_SUFFIXES
    missing = [seed + suffix for suffix in suffixes if not (output_dir / (seed + suffix)).is_file()]
    if missing and stage == "overlaps":
        spin_seeds = (f"{seed}_alpha", f"{seed}_beta")
        if all(not _missing_handoff(output_dir, candidate, suffixes)
               for candidate in spin_seeds):
            return
    if missing:
        label = "prepared files" if stage == "overlaps" else "hand-off files"
        parser.error(f"--stage {stage} requires {label}; missing: " + ", ".join(missing))


def _validate_consumer(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.wannier90 and _resolve_executable(args.wannier90) is None:
        parser.error(f"--wannier90 executable not found or not executable: {args.wannier90}")
    if args.mpi_ranks > 1 and _resolve_mpi_launcher() is None:
        parser.error("--mpi-ranks requires MPIEXEC, mpiexec, or mpirun")


def _validate_run(
    parser: argparse.ArgumentParser, args: argparse.Namespace, seed: str | None,
    output_dir: Path,
) -> None:
    if seed is None:
        parser.error("--seed NAME is required for this operation")
    if args.stage == "localize":
        if not args.wannier90:
            parser.error("--stage localize requires --wannier90 PATH")
        _validate_consumer(parser, args)
        assert seed is not None
        _resolve_localize_seeds(parser, output_dir, seed)
        return
    if not args.input:
        parser.error("--input PATH is required for this stage")
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file() or not os.access(input_path, os.R_OK):
        parser.error(f"input file not found or unreadable: {input_path}")
    _validate_input_structure(parser, input_path, args.stage)
    assert seed is not None
    if args.stage in ("overlaps", "2"):
        _validate_handoff(parser, output_dir, seed, stage="overlaps")
    if args.stage in ("prepare", "overlaps") and args.wannier90:
        parser.error("--wannier90 is valid with --stage all, hybrid, or localize")
    if args.mpi_ranks > 1 and not args.wannier90:
        parser.error("--mpi-ranks greater than 1 requires --wannier90 PATH")
    _validate_consumer(parser, args)


def _legacy_arguments(args: argparse.Namespace, seed: str, input_path: Path, method: str, workers: int, extras: Sequence[str]) -> list[str]:
    result = ["--stage", NAMED_STAGE_MAP.get(args.stage, args.stage), "--input", str(input_path), "--seedname", seed, "--method", method, "--lcao-workers", str(workers), "--lcao-cache-dir", "."]
    if args.parallel == "serial" or workers == 1:
        result.append("--no-parallel")
    if args.stage == "overlaps":
        result.append("--raw-overlaps")
    if (args.stage not in NUMERIC_STAGE_LABELS
            and not any(token == "--mmn-method" or token.startswith("--mmn-method=")
                        for token in extras)):
        result.extend(("--mmn-method", "analytic"))
    if args.num_wann is not None:
        result.extend(("--num-wann", str(args.num_wann)))
    if args.k_grid:
        result.extend(("--k-grid", *(str(n) for n in args.k_grid)))
    result.extend(extras)
    return result


_INTERNAL_OPTIONS = ("--lcao-workers", "--lcao-cache-dir", "--lcao-parse-only", "--raw-overlaps")


def _validate_advanced_arguments(
    parser: argparse.ArgumentParser, legacy_args: Sequence[str], extras: Sequence[str],
) -> argparse.Namespace:
    for token in extras:
        if any(token == option or token.startswith(option + "=") for option in _INTERNAL_OPTIONS):
            parser.error(f"{token.split('=', 1)[0]} is an internal option and cannot be overridden")
    from . import workflow
    try:
        parsed = workflow.main([*legacy_args, "--lcao-parse-only"])
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 2
        if code:
            raise
    assert isinstance(parsed, argparse.Namespace)
    return parsed


def _validate_advanced_prerequisites(
    parser: argparse.ArgumentParser, args: argparse.Namespace,
    advanced: argparse.Namespace, output_dir: Path, seed: str,
) -> None:
    if args.stage == "3":
        value = getattr(advanced, 'hr_file', None) or f"{seed}_hr.dat"
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = output_dir / path
        path = path.resolve()
        if not path.is_file() or not os.access(path, os.R_OK):
            parser.error(f"legacy stage 3 requires readable H(R) input: {path}")


class _WarningStream(io.TextIOBase):
    def __init__(self, target: io.TextIOBase):
        self.target, self.pending = target, ""

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        self.pending += value
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if any(token in line.lower() for token in ("warning", "error", "failed", "abort", "⚠")):
                self.target.write(line + "\n")
        return len(value)

    def flush(self) -> None:
        if self.pending and any(token in self.pending.lower() for token in ("warning", "error", "failed", "abort", "⚠")):
            self.target.write(self.pending)
        self.pending = ""
        self.target.flush()


@contextmanager
def _quiet_output(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    from contextlib import redirect_stdout
    stream = _WarningStream(sys.stdout)
    with redirect_stdout(stream):
        yield
    stream.flush()


def _run_wannier90(parser: argparse.ArgumentParser, executable: str, seed: str, output_dir: Path, ranks: int, threads: int) -> int:
    resolved = _resolve_executable(executable)
    if resolved is None:
        parser.error(f"--wannier90 executable not found or not executable: {executable}")
    command = [resolved, seed]
    if ranks > 1:
        launcher = _resolve_mpi_launcher()
        if launcher is None:
            parser.error("--mpi-ranks requires MPIEXEC, mpiexec, or mpirun")
        command = [launcher, "-n", str(ranks), *command]
    environment = os.environ.copy()
    for name in BLAS_THREAD_VARIABLES:
        environment[name] = str(threads)
    print(f"Running Wannier90: {' '.join(command)} (cwd: {output_dir})")
    return subprocess.run(command, cwd=output_dir, env=environment, check=False).returncode


def _print_complete_help(parser: argparse.ArgumentParser) -> None:
    parser.print_help()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, extras = parser.parse_known_args(argv)
    if args.help:
        _print_complete_help(parser)
        return 0
    if args.version:
        print(_version_text())
        return 0
    if "--no-parallel" in extras:
        if args.parallel not in ("auto", "serial") or args.threads not in (None, 1):
            parser.error("--no-parallel conflicts with --parallel threads or --threads greater than 1")
        args.parallel = "serial"
    seed = _resolve_seed(parser, args)
    mechanism, workers = _resolve_resources(parser, args)
    _validate_common_values(parser, args)
    # This must precede the parse-only import of the numerical workflow:
    # BLAS/OpenMP libraries read their limits at import initialization.
    _configure_frontend_environment(workers)
    output_dir = _resolved_output_dir(args.stage, seed, args.output_dir)
    capability_only = (
        args.check and args.input is None and seed is None and not extras
        and args.wannier90 is None and args.stage == "all"
    )
    if capability_only:
        return _print_capability_check(args, output_dir)
    _validate_run(parser, args, seed, output_dir)
    method = args.method or ("auto" if args.stage in NUMERIC_STAGE_LABELS else "hybrid")
    if args.stage == "overlaps" and method != "hybrid":
        parser.error("named --stage overlaps uses the hybrid-selected raw pool; omit --method or use --method hybrid")
    if args.stage == "localize" and extras:
        parser.error("advanced scientific options are not valid with --stage localize")
    input_path = Path(args.input).expanduser().resolve() if args.input else Path()
    legacy_args = (
        _legacy_arguments(args, seed, input_path, method, workers, extras)
        if args.stage != "localize" else []
    )
    advanced_args = None
    if legacy_args:
        advanced_args = _validate_advanced_arguments(parser, legacy_args, extras)
        assert seed is not None
        _validate_no_input_collision(
            parser, input_path, output_dir, seed, advanced_args,
            external_consumer=bool(args.wannier90))
        _validate_advanced_prerequisites(
            parser, args, advanced_args, output_dir, seed)
    if args.check:
        return _print_capability_check(args, output_dir)
    if args.verbose or args.dry_run:
        _print_plan(args, seed, output_dir, mechanism, workers, extras)
    if args.dry_run:
        return 0
    assert seed is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "localize":
        for actual_seed in _resolve_localize_seeds(parser, output_dir, seed):
            status = _run_wannier90(
                parser, args.wannier90, actual_seed, output_dir,
                args.mpi_ranks, workers,
            )
            if status:
                return status
        return 0
    if args.stage in ("all", "hybrid", "prepare", "overlaps", "1", "2"):
        _mark_generation_started(
            output_dir, (seed, f"{seed}_alpha", f"{seed}_beta"), args.stage)
    previous = Path.cwd()
    try:
        os.chdir(output_dir)
        from . import workflow
        with _quiet_output(args.quiet):
            actual_seeds = workflow.main(legacy_args)
    finally:
        os.chdir(previous)
    if args.stage in ("all", "hybrid", "2"):
        _mark_handoff_complete(parser, output_dir, actual_seeds or [seed])
    if args.wannier90:
        for actual_seed in actual_seeds or [seed]:
            status = _run_wannier90(
                parser, args.wannier90, actual_seed, output_dir,
                args.mpi_ranks, workers,
            )
            if status:
                return status
        return 0
    if args.quiet:
        print(f"lcao2wannier hand-off complete: {output_dir}")
    return 0


def cli_main() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli_main()
