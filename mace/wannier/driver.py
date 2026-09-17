"""Thin driver for William Comaskey's ``lcao2wannier`` package.

Thin is the design, not a shortcut. Everything scientific in the LCAO->Wannier90
bridge - the Fourier transform to H(k)/S(k), the generalized eigenproblem in a
non-orthogonal AO basis, SCDM projections for ``.amn``, and the analytic
momentum-shifted GTO overlaps for ``.mmn`` - is his. MACE resolves the module,
builds an argument list, runs it, and surfaces its diagnostics unaltered.

What this module deliberately does NOT do:

* It does not pass ``--method``, ``--mmn-method`` or ``--hybrid-trust``. Those
  are already his validated defaults for named stages. Restating them is a
  no-op today and pins MACE against his future retuning.
* It does not pass ``--k-grid``. He derives the Monkhorst-Pack mesh from the
  parent's own ``SHRINK`` factors, which is the mesh the SCF actually used. A
  MACE-chosen grid would silently decouple the Wannier mesh from the SCF mesh -
  the same class of silent-wrong-model hazard as a guessed R-vector count.
* It does not pass ``--quiet``. MEASURED: ``--quiet`` filters stdout to lines
  containing warning/error/failed/abort, and his self-audit banner
  ("STATUS: FAIL - wannier90 will reject this setup:") and its violation
  bullets contain none of those tokens. Quiet mode keeps the symptom and drops
  every reason.
* It never invokes ``wannier90.x`` itself. His package localizes given
  ``--wannier90 PATH``; MACE delegates.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

LCAO2WANNIER_CREDIT = (
    "Wannier90 hand-off via lcao2wannier (William Comaskey)"
)

# CITATION: TODO - ask William Comaskey which citation he wants (the package, a
# paper, or both). Nothing is written here until he answers; a provisional
# citation string would misattribute a publication on his behalf, which is worse
# than having none.
LCAO2WANNIER_CITATION = None

# The five files that constitute a completed hand-off.
HANDOFF_SUFFIXES = (".win", ".nnkp", ".eig", ".amn", ".mmn")

# CRYSTAL writes the direct-lattice cell index in an I4 field, so from index
# 1000 on the header has no separating space: " OVERLAP MATRIX - CELL N.1000(".
# lcao2wannier v1.0's header patterns are `CELL N\.\s+\d+\(` - the \s+ cannot
# match - so every cell from 1000 on is dropped, silently, and the run reports
# a plausible R-vector count.
#
# MEASURED on the real MACE corpus material
# test/SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized (derived N = 1247):
# 1247 overlap headers in the file, 999 matched by the stock regex, 248 dropped
# without a word. The run then reported "Unique R-vectors for H: 999" and
# proceeded. The one-character upstream fix is `\s+` -> `\s*` in all three
# header patterns; it belongs in his package, not in a MACE monkey-patch.
CELL_INDEX_PARSE_LIMIT = 1000

# Versions known to carry the >=1000 defect. Checked by version rather than
# blindly, so the guard retires itself when he ships the fix.
CELL_INDEX_BUG_VERSIONS = ("1.0.0",)

_OVERLAP_HEADER = b"OVERLAP MATRIX - CELL"

# His self-audit's POSITIVE verdict. Success requires FINDING this, never merely
# failing to find a failure marker: if the banner is reworded, absence-of-FAIL
# would silently start reporting refused models as successes.
# He prints a STATUS line from TWO independent self-checks, and MACE must read
# both. MEASURED in lcao2wannier 1.0.0:
#
#   wannier_checks.py:179/181  ->  "STATUS: PASS - satisfies Wannier90
#                                   disentanglement rules" | "STATUS: FAIL - ..."
#   conditioning.py:138        ->  "STATUS: GOOD" | "MARGINAL" | "BAD"
#
# The first version of this matched PASS and FAIL|BAD, and nothing else. GOOD and
# MARGINAL fell through the classifier entirely, so a run whose overlap matrices
# he had just rated MARGINAL ("could benefit from more R-vectors") was reported
# by MACE as an unqualified success - discarding the one negative signal that
# points straight back at a too-small R-vector count, the failure mode this
# whole feature exists to eliminate. Both end-to-end validation runs hit exactly
# that (worst cond(S) = 2.530e+04, STATUS: MARGINAL, ConversionResult.ok True).
#
# All four verdicts are now matched explicitly. An unmatched STATUS line means
# he reworded the banner, and that must surface as "unknown" - never as a pass.
_AUDIT_PASS_RE = re.compile(r"STATUS:\s*[^\n]*(PASS|GOOD)")
_AUDIT_MARGINAL_RE = re.compile(r"STATUS:\s*[^\n]*MARGINAL")
_AUDIT_FAIL_RE = re.compile(r"STATUS:\s*[^\n]*(FAIL|BAD)")

# His conditioning report frames the STATUS line. Captured whole so the caveat
# reaches the user with his own remedy attached rather than as a bare word.
_CONDITIONING_BANNER = "Overlap Matrix Conditioning Check"


class Lcao2WannierUnavailable(Exception):
    """The optional dependency is absent or unusable. Never a traceback."""


@dataclass
class ConversionResult:
    """Outcome of one lcao2wannier invocation."""

    returncode: int
    stdout: str
    stderr: str
    command: List[str]
    output_dir: Path
    seed: str
    produced: Dict[str, List[str]] = field(default_factory=dict)
    audit: str = "unknown"   # "pass" | "marginal" | "fail" | "unknown"
    audit_text: str = ""
    conditioning_text: str = ""

    @property
    def ok(self) -> bool:
        """Success, defined positively on all three axes.

        rc == 0 is not enough on its own: his disentanglement self-audit is
        deliberately non-fatal ("a violation is reported, not fatal, so the user
        still gets files"), so a refused model exits 0 with all five files
        present.
        """
        return (
            self.returncode == 0
            and bool(self.produced)
            and all(len(v) == len(HANDOFF_SUFFIXES) for v in self.produced.values())
            and self.audit in ("pass", "marginal")
        )

    @property
    def qualified(self) -> bool:
        """Usable, but he attached a caveat to it.

        A MARGINAL conditioning verdict is not a refusal - his own text says
        "results may be acceptable" - so it must not be reported as a failure.
        It is also not a clean pass, and reporting it as one throws away the
        only warning a user gets that the R-vector count was too tight.
        """
        return self.ok and self.audit == "marginal"


# --- Locating the dependency ------------------------------------------------


def find_lcao2wannier() -> Tuple[bool, Optional[str]]:
    """Is ``lcao2wannier`` importable, and at what version?

    Uses ``find_spec`` rather than an import: his ``__init__`` is lazy (it does
    not pull numpy), so this stays sub-millisecond and cannot fail on a broken
    numerical stack. That also means importability is NOT proof of usability -
    only the ``--dry-run`` pre-flight settles that.
    """
    try:
        spec = importlib.util.find_spec("lcao2wannier")
    except (ImportError, ValueError):
        return False, None
    if spec is None:
        return False, None
    try:
        import lcao2wannier  # noqa: F401

        return True, getattr(lcao2wannier, "__version__", None)
    except Exception:
        return True, None


def describe_missing_dependency() -> str:
    """The message a user sees when the optional dependency is not installed."""
    return (
        "lcao2wannier is not installed, so MACE cannot run the Wannier90\n"
        "conversion.\n"
        "\n"
        "  install with : pip install lcao2wannier\n"
        "\n"
        "This is an OPTIONAL dependency. Nothing else in MACE needs it: the\n"
        "MATDUMP calculation itself is stock CRYSTAL23, the deck it generated is\n"
        "already valid, and the matrix dump it produced can be converted by hand\n"
        "or on another machine.\n"
        "\n"
        f"{LCAO2WANNIER_CREDIT}.\n"
        "The LCAO->Wannier90 method and the package are his work; MACE only\n"
        "generates the CRYSTAL input and orchestrates the run."
    )


def describe_missing_wannier90() -> str:
    """The message a user sees when they ask to localize with no wannier90.x."""
    return (
        "Localization was requested but no wannier90.x was given.\n"
        "\n"
        "  supply one with : --wannier90 /path/to/wannier90.x\n"
        "\n"
        "wannier90.x is user-supplied upstream software (Wannier90 3.x). MACE\n"
        "never bundles or downloads it.\n"
        "\n"
        "Without it the hand-off still completes: lcao2wannier writes the .win,\n"
        ".nnkp, .eig, .amn and .mmn files and needs no `wannier90 -pp` round\n"
        "trip, so you can run wannier90.x on them yourself, later or elsewhere."
    )


# --- Guarding the parent dump ----------------------------------------------


def count_overlap_cells(parent: Path) -> int:
    """Number of ``OVERLAP MATRIX - CELL`` headers in a dump.

    Streamed in binary: a production dump is tens to hundreds of MB, and CRYSTAL
    outputs can contain NUL bytes.
    """
    count = 0
    with open(parent, "rb") as handle:
        for raw in handle:
            if _OVERLAP_HEADER in raw:
                count += 1
    return count


def check_parent_dump(parent: Path, version: Optional[str]) -> Optional[str]:
    """Refuse a conversion the installed lcao2wannier would silently truncate.

    The defect is in the optional dependency, so the guard lives here and not in
    the Layer 1 deck generator: a user with no lcao2wannier, or with a patched
    or newer one, must never be refused a perfectly valid CRYSTAL dump. The
    check is by version, so it retires itself once the fix ships upstream.
    """
    if version is not None and version not in CELL_INDEX_BUG_VERSIONS:
        return None

    try:
        cells = count_overlap_cells(parent)
    except OSError as exc:
        return f"Cannot read the matrix dump {parent}: {exc}"

    if cells < CELL_INDEX_PARSE_LIMIT:
        return None

    version_text = version or "an unknown version"
    return (
        f"Refusing to convert {parent.name}: it contains {cells} direct-lattice\n"
        f"cells, and lcao2wannier {version_text} cannot read past cell 999.\n"
        "\n"
        "CRYSTAL writes the cell index in an I4 field, so from 1000 on the header\n"
        "runs together with no separating space:\n"
        "\n"
        "   OVERLAP MATRIX - CELL N. 999(  4  3 -6)\n"
        "   OVERLAP MATRIX - CELL N.1000( -4  1 -3)     <- no space\n"
        "\n"
        "The package's header patterns require whitespace there, so every cell\n"
        f"from 1000 on is dropped SILENTLY - {cells - CELL_INDEX_PARSE_LIMIT + 1} "
        f"of {cells} here - and the run\n"
        "then reports a plausible R-vector count and proceeds. The resulting\n"
        "Wannier model is wrong, with nothing in the output saying so.\n"
        "\n"
        "The dump itself is correct and complete; only the reader is affected.\n"
        "The upstream fix is one character, `\\s+` -> `\\s*` after `N\\.` in the\n"
        "overlap and both Fock header patterns. Report it to William Comaskey\n"
        "rather than patching around it here.\n"
        "\n"
        "To proceed anyway you would have to re-dump with fewer R-vectors, which\n"
        "means knowingly truncating the model - MACE will not do that for you."
    )


# --- Building and running ---------------------------------------------------


def build_command(
    parent: Path,
    seed: str,
    output_dir: Path,
    stage: str = "all",
    threads: Optional[int] = None,
    wannier90: Optional[Path] = None,
    extra_args: Optional[Sequence[str]] = None,
    dry_run: bool = False,
) -> List[str]:
    """Assemble the lcao2wannier argument list.

    Invoked as ``python -m lcao2wannier``, never the console script: the script
    is absent from PATH whenever the package lives in a venv or conda env that
    is not activated, while ``-m`` always resolves to the same interpreter the
    guarded import used.

    Only ``--input``, ``--seed``, ``--output-dir`` and ``--stage`` are derived by
    MACE. ``--threads`` is passed only when a CPU budget is actually known; his
    own scheduler detection already reads SLURM_CPUS_PER_TASK, so under a MACE
    SLURM job omitting it is correct.
    """
    command = [
        sys.executable,
        "-m",
        "lcao2wannier",
        "--input",
        str(Path(parent).resolve()),
        "--seed",
        seed,
        "--output-dir",
        str(Path(output_dir).resolve()),
        "--stage",
        stage,
    ]
    if threads:
        command += ["--threads", str(int(threads))]
    if wannier90:
        command += ["--wannier90", str(Path(wannier90))]
    if dry_run:
        command.append("--dry-run")
    if extra_args:
        # Appended last and unvalidated: this is the user's opt-in escape hatch
        # for his advanced numerical knobs. MACE must not synthesize any of them.
        command += list(extra_args)
    return command


def sanitize_seed(name: str) -> str:
    """Reduce a name to a bare basename he will accept.

    He validates and exits 2 on a seed containing a path separator or a dot
    segment, so do it here and report a usable name instead.
    """
    seed = Path(str(name)).name
    seed = seed.replace(os.sep, "_")
    if seed in ("", ".", ".."):
        raise Lcao2WannierUnavailable(
            f"{name!r} does not reduce to a usable output basename."
        )
    return seed


def _classify_audit(stdout: str) -> Tuple[str, str]:
    """Read his self-audit verdicts out of captured stdout.

    Positive-marker logic: a PASS/GOOD must be found. Anything else is 'fail' if
    a failure banner is present, 'marginal' if he flagged the overlap matrices
    as moderately ill-conditioned, and otherwise 'unknown' - reported as
    not-verified, never as a pass.

    Severity wins over count: a run that prints both "PASS" (disentanglement)
    and "MARGINAL" (conditioning) is marginal, not a pass. Conflating them is
    what let a model he had qualified be reported as an unqualified success.
    """
    lines = stdout.splitlines()
    verdict_lines = [ln for ln in lines
                     if _AUDIT_PASS_RE.search(ln)
                     or _AUDIT_MARGINAL_RE.search(ln)
                     or _AUDIT_FAIL_RE.search(ln)]
    text = "\n".join(verdict_lines)
    if any(_AUDIT_FAIL_RE.search(ln) for ln in verdict_lines):
        return "fail", text
    if any(_AUDIT_MARGINAL_RE.search(ln) for ln in verdict_lines):
        return "marginal", text
    if verdict_lines:
        return "pass", text
    return "unknown", text


def _conditioning_report(stdout: str) -> str:
    """His overlap-conditioning block, verbatim, or "" if he did not print one.

    Reproduced whole rather than summarized: the numbers (worst/median condition
    number, how many k-points were not positive definite) and his own remedy are
    what tell a user whether to re-run at a larger R-vector count, and he is the
    one who knows what they mean.
    """
    lines = stdout.splitlines()
    start = None
    for index, line in enumerate(lines):
        if _CONDITIONING_BANNER in line:
            start = index
            break
    if start is None:
        return ""
    block = [lines[start]]
    for line in lines[start + 1:]:
        block.append(line)
        # His block is fenced by a rule of '=' on both sides; the closing one
        # comes after the STATUS line and any remedy text.
        if set(line.strip()) == {"="} and len(block) > 2:
            break
    return "\n".join(block)


def _collect_outputs(output_dir: Path, seed: str) -> Dict[str, List[str]]:
    """Which of the five hand-off files exist, per emitted seed.

    A collinear run emits ``<seed>_alpha`` and ``<seed>_beta`` rather than
    ``<seed>``, so the seeds are discovered from the ``.win`` files on disk.

    Never judge success by globbing for ``*.lcao2wannier.incomplete`` markers:
    MEASURED, a SUCCESSFUL non-collinear run leaves ``<seed>_alpha`` and
    ``<seed>_beta`` markers behind, and ``--stage prepare`` leaves all three.
    """
    produced: Dict[str, List[str]] = {}
    for win in sorted(Path(output_dir).glob("*.win")):
        stem = win.stem
        if stem != seed and not stem.startswith(f"{seed}_"):
            continue
        present = [suffix for suffix in HANDOFF_SUFFIXES
                   if (Path(output_dir) / f"{stem}{suffix}").exists()]
        produced[stem] = present
    return produced


def convert(
    parent: Path,
    seed: Optional[str] = None,
    output_dir: Optional[Path] = None,
    stage: str = "all",
    threads: Optional[int] = None,
    wannier90: Optional[Path] = None,
    extra_args: Optional[Sequence[str]] = None,
    skip_preflight: bool = False,
) -> ConversionResult:
    """Run the conversion, in two phases, and report his diagnostics intact.

    Phase 1 is the identical argument list plus ``--dry-run``. That is the only
    cheap probe that actually proves the package can run: ``--version`` and a
    bare ``--check`` both exit 0 even with numpy broken, because his package
    ``__init__`` is lazy. ``--dry-run`` forces the numerical import, validates
    that the input really is a CRYSTAL matrix dump, and writes nothing.

    Phase 2 is the same list without it.
    """
    parent = Path(parent)
    if not parent.exists():
        raise Lcao2WannierUnavailable(f"Matrix dump not found: {parent}")

    available, version = find_lcao2wannier()
    if not available:
        raise Lcao2WannierUnavailable(describe_missing_dependency())

    refusal = check_parent_dump(parent, version)
    if refusal:
        raise Lcao2WannierUnavailable(refusal)

    seed = sanitize_seed(seed or parent.stem)
    output_dir = Path(output_dir) if output_dir else parent.parent / f"{seed}.wannier"
    output_dir.mkdir(parents=True, exist_ok=True)

    base = dict(parent=parent, seed=seed, output_dir=output_dir, stage=stage,
                threads=threads, wannier90=wannier90, extra_args=extra_args)

    if not skip_preflight:
        probe = build_command(dry_run=True, **base)
        completed = subprocess.run(probe, capture_output=True, text=True)
        if completed.returncode != 0:
            # His argparse refusals are a single clean line on stderr with exit
            # code 2. Surface it verbatim - do not reformat it, and never turn
            # it into a traceback.
            detail = (completed.stderr or completed.stdout).strip()
            if completed.returncode == 2:
                raise Lcao2WannierUnavailable(
                    f"lcao2wannier refused these arguments:\n  {detail}"
                )
            raise Lcao2WannierUnavailable(
                "lcao2wannier is installed but could not run. Its numerical\n"
                "stack (numpy/scipy) is most likely missing or broken.\n\n"
                f"{detail}"
            )

    command = build_command(dry_run=False, **base)
    completed = subprocess.run(command, capture_output=True, text=True)

    audit, audit_text = _classify_audit(completed.stdout)
    conditioning_text = _conditioning_report(completed.stdout)
    result = ConversionResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        command=command,
        output_dir=output_dir,
        seed=seed,
        produced=_collect_outputs(output_dir, seed),
        audit=audit,
        audit_text=audit_text,
        conditioning_text=conditioning_text,
    )

    # Persist his full diagnostics beside the outputs. Everything except his
    # argparse refusals goes to stdout, including the audit block.
    try:
        (output_dir / f"{seed}.lcao2wannier.log").write_text(
            completed.stdout + completed.stderr
        )
    except OSError:
        pass

    return result
