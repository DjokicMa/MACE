"""The one classification of a failed CRYSTAL run's output.

completion_checker (its bucket folders), the recovery detector and the queue
manager (whose error type picks the recovery handler) all read the error
lines through this table, so one output gets one class everywhere.

The classes are tried in the order below, each over the whole output, and the
first class with a matching line wins - not the first matching line. An OOM
kill comes first: mpirun prints its BAD TERMINATION banner just before
"KILLED BY SIGNAL: 9", and the banner alone is only a generic runtime error.
CRYSTAL's own errors come next ("ERROR **** ZERO **** FERMI ENERGY NOT IN
INTERVAL" is followed by the MPI_Abort it causes), and the generic MPI, SLURM
and Fortran runtime lines come last.

Each class names the error type the queue manager records (the key of the
recovery handler in recovery_config.yaml), the completion_checker bucket and,
when no handler exists for it, why the run is not recovered automatically.
Matching is case-insensitive.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union


@dataclass(frozen=True)
class ErrorClass:
    key: str
    patterns: Tuple[str, ...]
    error_type: str          # queue manager / recovery-config name
    bucket: str              # completion_checker category
    description: str
    no_recovery: str = ""    # why there is no automatic recovery ("" = there is one)


ERROR_CLASSES: Tuple[ErrorClass, ...] = (
    # Written by the job script's scratch guard before CRYSTAL runs.
    ErrorClass("scratch", ("no writable scratch directory",), "scratch_error", "scratch",
               "No writable scratch directory (CRYSTAL not run)",
               "a person has to make a scratch or submit directory writable"),
    # The OOM killer: SLURM's handler line, or MPI reporting a rank SIGKILLed
    # (job 12091685: SLURM still recorded COMPLETED).
    ErrorClass("oom", ("out-of-memory handler", "KILLED BY SIGNAL: 9", "INSUFFICIENT MEMORY",
                       "OUT OF MEMORY", "MEMORY ALLOCATION"),
               "memory_error", "memory", "Out of memory error"),
    ErrorClass("time_limit", ("DUE TO TIME LIMIT", "TIME LIMIT EXCEEDED"),
               "timeout_error", "time", "Time limit exceeded"),
    ErrorClass("disk", ("error during write", "DISK FULL", "NO SPACE LEFT", "DISK QUOTA EXCEEDED"),
               "disk_space_error", "quota", "Disk quota/write error"),
    # OPTGEOM MAXCYCLE reached; the MPI_Abort after it is no runtime error.
    ErrorClass("opt_unconverged", ("OPT END - FAILED",), "opt_unconverged_error",
               "opt_unconverged", "Geometry optimization not converged (OPT END - FAILED)"),
    # SCF divergence: TOO MANY CYCLES, or a blow-up ending in the Fermi stop
    # (diamond + B3LYP-D3 after 206 cycles).
    ErrorClass("scf", ("TOO MANY CYCLES", "SCF NOT CONVERGED", "CONVERGENCE NOT ACHIEVED",
                       "FERMI ENERGY NOT IN INTERVAL"),
               "convergence_error", "too_many_scf", "SCF convergence failure"),
    # "DISTANCE BETWEEN ATOMS n m TOO SMALL" + "ERROR **** NEIGHB ****"
    ErrorClass("atoms_too_close", ("**** NEIGHB ****", "ATOMS TOO CLOSE",
                                   "SMALL DISTANCE BETWEEN ATOMS"),
               "geometry_error", "geometry_small_dist", "Geometry error (atoms too close)",
               "the geometry has to be fixed (or SMALLDIST set) by a person"),
    ErrorClass("shrink", ("ANISOTROPIC SHRINKING FACTOR", "SHRINK FACTOR TOO SMALL",
                          "TOO SMALL SHRINK FACTOR"),
               "shrink_error", "shrink_error", "SHRINK parameter error"),
    ErrorClass("linear_basis", ("BASIS SET LINEARLY DEPENDENT",), "basis_error", "linear_basis",
               "Linear dependency in basis set",
               "the basis set or geometry has to be changed by a person"),
    ErrorClass("geometry_failed", ("GEOMETRY OPTIMIZATION FAILED",), "geometry_error", "potential",
               "Geometry optimization failed",
               "the geometry has to be checked by a person"),
    # ("Permission denied" alone is not matched: a cp or rm warning in a
    # finished run's output says it too.)
    ErrorClass("io", ("I/O ERROR",), "io_error", "potential",
               "Input/output error",
               "file permissions or paths have to be fixed by a person"),
    # A crash with no cause of its own; the recovery treats it as memory.
    ErrorClass("segfault", ("segmentation fault",), "memory_error", "potential",
               "Segmentation fault/runtime error"),
    # Generic MPI / SLURM / Fortran runtime lines: what stopped the run is
    # not in them (the queue manager reads CRYSTAL's fort.87 for it).
    ErrorClass("runtime", ("=   bad termination of", "abort(1) on node", "srun: error:",
                           "slurmstepd: error: ***", "forrtl: error (78):",
                           "Stack trace terminated abnormally."),
               "system_error", "potential", "Segmentation fault/runtime error",
               "the output names no cause; check CRYSTAL's fort.87 or run the deck serially"),
)

BY_KEY = {c.key: c for c in ERROR_CLASSES}


@dataclass(frozen=True)
class Match:
    cls: ErrorClass
    pattern: str
    line: str

    def __getattr__(self, name):
        return getattr(self.cls, name)


def classify(text: Union[str, Iterable[str]],
             classes: Iterable[ErrorClass] = ERROR_CLASSES) -> Optional[Match]:
    """The highest-priority class with a line in ``text`` (a string or its
    lines), with the pattern and the first line it matched; None if none."""
    lines = text.splitlines() if isinstance(text, str) else list(text)
    lowered = [l.lower() for l in lines]
    for cls in classes:
        for pattern in cls.patterns:
            p = pattern.lower()
            for raw, low in zip(lines, lowered):
                if p in low:
                    return Match(cls, pattern, raw.strip())
    return None


def bucket_patterns(order: Iterable[str]) -> dict:
    """{completion_checker bucket: [patterns]} in the bucket order given."""
    out = {b: [] for b in order}
    for cls in ERROR_CLASSES:
        out.setdefault(cls.bucket, []).extend(cls.patterns)
    return out
