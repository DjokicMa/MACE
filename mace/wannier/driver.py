"""Thin driver for William Comaskey's ``lcao2wannier`` package.

The package is bundled with MACE as ``mace.wannier.lcao2wannier`` (MIT; see
VENDORED.md there for the source version and every local change). Thin is
still the design, not a shortcut. Everything scientific in the LCAO->Wannier90
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

import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

LCAO2WANNIER_CREDIT = (
    "Wannier90 hand-off via lcao2wannier (William Comaskey)"
)

# No citation has been designated yet; the open question is tracked in
# AUTHORSHIP.md. A provisional citation string would misattribute a publication
# on his behalf, which is worse than having none.
LCAO2WANNIER_CITATION = None

# The bundled copy, run as ``python -m`` so its own CLI is what executes.
LCAO2WANNIER_MODULE = "mace.wannier.lcao2wannier"

# Read from the package, not restated: VENDORED.md records the same value.
from mace.wannier.lcao2wannier import __version__ as LCAO2WANNIER_VERSION  # noqa: E402

# The directory that holds the ``mace`` package, put on the child's PYTHONPATH
# so ``-m mace.wannier.lcao2wannier`` resolves to this copy whatever the cwd.
_MACE_ROOT = Path(__file__).resolve().parents[2]

# The five files that constitute a completed hand-off.
HANDOFF_SUFFIXES = (".win", ".nnkp", ".eig", ".amn", ".mmn")

# CRYSTAL writes the direct-lattice cell index in an I4 field, so from index
# 1000 on the header has no separating space: " OVERLAP MATRIX - CELL N.1000(".
# Stock lcao2wannier 1.0.0 required whitespace there and silently dropped every
# such cell (MEASURED on the corpus diamond, derived N = 1247: 999 read, 248
# dropped, no warning), so MACE used to refuse those dumps. The bundled copy
# carries the one-character fix (`\s+` -> `\s*`, see VENDORED.md), so there is
# nothing left to refuse.

# His self-audit's POSITIVE verdict. Success requires FINDING this, never merely
# failing to find a failure marker: if the banner is reworded, absence-of-FAIL
# would silently start reporting refused models as successes.
# He prints a STATUS line from TWO independent self-checks, and MACE must read
# both. MEASURED in lcao2wannier 1.0.0 (line numbers are the bundled copy's):
#
#   wannier_checks.py:181/183  ->  "STATUS: PASS - satisfies Wannier90
#                                   disentanglement rules" | "STATUS: FAIL - ..."
#   conditioning.py:140        ->  "STATUS: GOOD" | "MARGINAL" | "BAD"
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


# --- 2-component dumps --------------------------------------------------------

# Fock-block headers in a dump. A 2c dump from the development properties
# prints each block as REAL and IMAG parts under the spinor labels; the stock
# build on a 2c fort.9 prints scalar blocks only, without an error (MEASURED on
# a real 2c-SOC Bi2 fort.9: 124 REAL + 124 IMAG blocks with ALPHA_ALPHA /
# ALPHA_BETA / BETA_BETA from the development build, 62 scalar
# "FOCK MATRIX - CELL" blocks from the stock one).
_FOCK_SCALAR = b"FOCK MATRIX - CELL"
_FOCK_REAL = b"FOCK MATRIX (REAL PART)"
_FOCK_IMAG = b"FOCK MATRIX (IMAG PART)"
_SPINOR_LABEL = b"ALPHA_ALPHA ELECTRONS"

_D3_MATDUMP = Path(__file__).resolve().parents[2] / "Crystal_d3" / "d3_matdump.py"


def _d3_matdump():
    """Crystal_d3/d3_matdump.py, where MACE's 2-component detection lives."""
    import importlib.util

    loaded = sys.modules.get("d3_matdump")
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location("d3_matdump", _D3_MATDUMP)
    module = importlib.util.module_from_spec(spec)
    sys.modules["d3_matdump"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("d3_matdump", None)
        raise
    return module


def dump_fock_blocks(dump: Path) -> Dict[str, int]:
    """Count the Fock-block headers in a dump, line by line (dumps are large)."""
    counts = {"scalar": 0, "real": 0, "imag": 0, "spinor": 0}
    with open(dump, "rb") as handle:
        for raw in handle:
            if _FOCK_SCALAR in raw:
                counts["scalar"] += 1
            elif _FOCK_REAL in raw:
                counts["real"] += 1
            elif _FOCK_IMAG in raw:
                counts["imag"] += 1
            elif _SPINOR_LABEL in raw:
                counts["spinor"] += 1
    return counts


def find_scf_parents(dump: Path) -> List[Path]:
    """The SCF outputs beside a dump that `mace opt2d3` names it after.

    opt2d3 writes ``<base>_matdump.d3`` (so the dump is ``<base>_matdump.out``)
    where ``<base>`` is the parent's stem without _opt/_sp/_OPT/_SP.
    """
    dump = Path(dump)
    stem = dump.stem
    base = stem[:-len("_matdump")] if stem.lower().endswith("_matdump") else stem
    found = []
    for parent_stem in (f"{base}_sp", f"{base}_opt", f"{base}_SP", f"{base}_OPT",
                        base):
        for suffix in (".out", ".log"):
            candidate = dump.with_name(parent_stem + suffix)
            if candidate.is_file() and candidate != dump:
                found.append(candidate)
    return found


def scf_is_two_component(scf_output: Path) -> bool:
    """Whether an SCF run was 2-component: its .out's own lines, or a TWOCOMPON
    block in the .d12 beside it (manual sec. 6.2, p.170). The deck's first line
    is its title and is never read as a keyword."""
    md = _d3_matdump()
    try:
        out_text = Path(scf_output).read_text(errors="replace")
    except OSError:
        out_text = ""
    if md.output_is_two_component(out_text):
        return True
    deck = Path(scf_output).with_suffix(".d12")
    try:
        deck_text = deck.read_text(errors="replace") if deck.is_file() else ""
    except OSError:
        deck_text = ""
    body = deck_text.split("\n", 1)[1] if "\n" in deck_text else ""
    return any(line.strip().upper() == "TWOCOMPON" for line in body.splitlines())


def two_component_refusal(dump: Path, scf_outputs: Sequence[Path]) -> Optional[str]:
    """Why a dump must not be converted, or None.

    A 2c parent needs a dump with REAL and IMAG Fock parts AND the spinor
    labels. A dump with all three is accepted whatever the parent; a dump from
    a 2c parent without them is what the stock build prints on a 2c fort.9.
    """
    blocks = dump_fock_blocks(dump)
    if blocks["real"] and blocks["imag"] and blocks["spinor"]:
        return None
    two_c = [p for p in scf_outputs if scf_is_two_component(p)]
    if not two_c:
        return None
    found = (f"{blocks['scalar']} scalar 'FOCK MATRIX - CELL', "
             f"{blocks['real']} REAL PART, {blocks['imag']} IMAG PART blocks, "
             f"{blocks['spinor']} 'ALPHA_ALPHA ELECTRONS' labels")
    return (
        f"Refusing to convert {Path(dump).name}: its parent SCF\n"
        f"({', '.join(p.name for p in two_c)}) is a 2-component (TWOCOMPON) run,\n"
        f"but the dump does not hold 2-component matrices.\n"
        f"\n"
        f"  found    : {found}\n"
        f"  needed   : FOCK MATRIX (REAL PART) and (IMAG PART) blocks under the\n"
        f"             ALPHA_ALPHA / ALPHA_BETA / BETA_BETA spinor labels\n"
        f"\n"
        f"This is what the STOCK CRYSTAL23 properties prints on a 2-component\n"
        f"fort.9, without any error (measured: only scalar 'FOCK MATRIX - CELL'\n"
        f"blocks). Converting it would build a model from an incomplete\n"
        f"Hamiltonian. Re-run the MATDUMP deck with the CRYSTAL23 development\n"
        f"properties/Pproperties (request it from the CRYSTAL23 developers; MACE\n"
        f"never bundles or redistributes it)."
    )


class Lcao2WannierUnavailable(Exception):
    """The conversion cannot run (bad input, broken numpy/scipy). Never a traceback."""


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
    # True when every line was already passed to the caller's echo as it
    # arrived, so the caller must not print stdout/stderr a second time.
    streamed: bool = False

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


# --- The bundled package ----------------------------------------------------


def _subprocess_env() -> Dict[str, str]:
    """The child's environment: this process's, with MACE importable.

    mace_cli finds the package because it runs from the repository root; a
    child started from any other cwd would not, and would report the bundled
    copy as missing.
    """
    env = os.environ.copy()
    parts = [str(_MACE_ROOT)]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


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

    Invoked as ``python -m mace.wannier.lcao2wannier`` - the bundled copy's own
    command-line entry, in the same interpreter that is running MACE. There is
    no separate console script to find on PATH.

    Only ``--input``, ``--seed``, ``--output-dir`` and ``--stage`` are derived by
    MACE. ``--threads`` is passed only when a CPU budget is actually known; his
    own scheduler detection already reads SLURM_CPUS_PER_TASK, so under a MACE
    SLURM job omitting it is correct.
    """
    command = [
        sys.executable,
        "-m",
        LCAO2WANNIER_MODULE,
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


def _run_streaming(command: List[str], env: Dict[str, str],
                   echo: Callable[[str, str], None]) -> Tuple[int, str, str]:
    """Run ``command``, passing each output line to ``echo`` as it arrives.

    A conversion runs for 10-15 minutes and prints its own stages (parsing,
    the eigenproblems, each file written); capturing to the end left the user
    with nothing to look at. Both streams are still kept whole, for the audit
    and the log. ``PYTHONUNBUFFERED`` makes the child flush line by line,
    which a Python writing to a pipe otherwise does not.
    """
    env = dict(env, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1,
                            env=env)
    captured: Dict[str, List[str]] = {"stdout": [], "stderr": []}

    def pump(handle, name):
        for line in handle:
            captured[name].append(line)
            echo(line.rstrip("\n"), name)
        handle.close()

    readers = [threading.Thread(target=pump, args=(proc.stdout, "stdout"),
                                daemon=True),
               threading.Thread(target=pump, args=(proc.stderr, "stderr"),
                                daemon=True)]
    for reader in readers:
        reader.start()
    returncode = proc.wait()
    for reader in readers:
        reader.join()
    return returncode, "".join(captured["stdout"]), "".join(captured["stderr"])


def convert(
    parent: Path,
    seed: Optional[str] = None,
    output_dir: Optional[Path] = None,
    stage: str = "all",
    threads: Optional[int] = None,
    wannier90: Optional[Path] = None,
    extra_args: Optional[Sequence[str]] = None,
    skip_preflight: bool = False,
    echo: Optional[Callable[[str, str], None]] = None,
    progress: Optional[Callable[[str], None]] = None,
    scf_output: Optional[Path] = None,
) -> ConversionResult:
    """Run the conversion, in two phases, and report his diagnostics intact.

    Phase 1 is the identical argument list plus ``--dry-run``. That is the only
    cheap probe that actually proves the package can run: ``--version`` and a
    bare ``--check`` both exit 0 even with numpy broken, because his package
    ``__init__`` is lazy. ``--dry-run`` forces the numerical import, validates
    that the input really is a CRYSTAL matrix dump, and writes nothing.

    Dumps past cell 999 are no longer refused: the bundled parser reads
    CRYSTAL's I4 cell index (see the I4 note near the top of this module).

    Phase 2 is the same list without it. With ``echo`` its output is passed
    on line by line while it runs (``echo(line, "stdout"|"stderr")``);
    ``progress`` receives MACE's own one-line stage messages.

    Before either phase, a dump whose parent SCF (``scf_output``, or the
    outputs beside the dump that opt2d3 names it after) was 2-component but
    which lacks the 2-component Fock blocks is refused (see
    ``two_component_refusal``).
    """
    parent = Path(parent)
    if not parent.exists():
        raise Lcao2WannierUnavailable(f"Matrix dump not found: {parent}")

    say = progress or (lambda message: None)

    if scf_output is not None:
        if not Path(scf_output).is_file():
            raise Lcao2WannierUnavailable(f"SCF output not found: {scf_output}")
        scf_outputs = [Path(scf_output)]
    else:
        scf_outputs = find_scf_parents(parent)
    refusal = two_component_refusal(parent, scf_outputs)
    if refusal:
        raise Lcao2WannierUnavailable(refusal)
    if not scf_outputs:
        blocks = dump_fock_blocks(parent)
        if blocks["scalar"] and not blocks["real"]:
            say("Note: no parent SCF output was found beside the dump, so a "
                "2-component parent (whose stock dump would be incomplete) "
                "could not be ruled out; pass --scf-output to check it.")

    seed = sanitize_seed(seed or parent.stem)
    output_dir = Path(output_dir) if output_dir else parent.parent / f"{seed}.wannier"
    output_dir.mkdir(parents=True, exist_ok=True)

    base = dict(parent=parent, seed=seed, output_dir=output_dir, stage=stage,
                threads=threads, wannier90=wannier90, extra_args=extra_args)

    if not skip_preflight:
        say("Checking the dump and arguments (lcao2wannier dry run, "
            "writes nothing)...")
        probe = build_command(dry_run=True, **base)
        completed = subprocess.run(probe, capture_output=True, text=True,
                                   env=_subprocess_env())
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
                "The bundled lcao2wannier could not run. Its numerical stack\n"
                "(numpy/scipy, both MACE requirements) is most likely missing\n"
                "or broken.\n\n"
                f"{detail}"
            )

    command = build_command(dry_run=False, **base)
    say(f"Converting {parent.name} (stage {stage}); lcao2wannier's progress "
        "follows. A large dump takes 10-15 minutes.")
    if echo is not None:
        returncode, stdout, stderr = _run_streaming(command, _subprocess_env(),
                                                    echo)
    else:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   env=_subprocess_env())
        returncode, stdout, stderr = (completed.returncode, completed.stdout,
                                      completed.stderr)

    audit, audit_text = _classify_audit(stdout)
    conditioning_text = _conditioning_report(stdout)
    result = ConversionResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        command=command,
        output_dir=output_dir,
        seed=seed,
        produced=_collect_outputs(output_dir, seed),
        audit=audit,
        audit_text=audit_text,
        conditioning_text=conditioning_text,
        streamed=echo is not None,
    )

    # Persist his full diagnostics beside the outputs. Everything except his
    # argparse refusals goes to stdout, including the audit block.
    try:
        (output_dir / f"{seed}.lcao2wannier.log").write_text(
            stdout + stderr
        )
    except OSError:
        pass

    return result
