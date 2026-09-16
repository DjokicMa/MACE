"""The lcao2wannier driver (Layer 2).

``lcao2wannier`` is William Comaskey's package and an OPTIONAL dependency. The
tests that need it use ``pytest.importorskip`` so the corpus-less CI stays
green without it; everything that can be asserted without it - the argument
list, the audit polarity, the refusals - runs everywhere.

MACE's job here is narrow: build an argument list, run it, and report what he
says. So most of these tests assert what MACE does NOT do.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, find_data

from mace.wannier.driver import (
    CELL_INDEX_PARSE_LIMIT,
    ConversionResult,
    HANDOFF_SUFFIXES,
    LCAO2WANNIER_CITATION,
    Lcao2WannierUnavailable,
    _classify_audit,
    build_command,
    check_parent_dump,
    convert,
    count_overlap_cells,
    describe_missing_dependency,
    describe_missing_wannier90,
    find_lcao2wannier,
    sanitize_seed,
)

MACE_CLI = REPO_ROOT / "mace_cli"


def _dump(tmp_path, cells, name="material_matdump.out"):
    """A matrix dump with a given number of cell headers.

    Only the headers matter to the guard under test, and CRYSTAL's I4 field is
    reproduced exactly - that is the whole point at index >= 1000.
    """
    lines = [" CRYSTAL"]
    for i in range(1, cells + 1):
        lines.append(f" OVERLAP MATRIX - CELL N.{i:>4}(  0  0  0)")
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------
# The argument list - defaults are respected by OMITTING them
# --------------------------------------------------------------------------


def test_command_passes_only_what_mace_derives(tmp_path):
    command = build_command(tmp_path / "d.out", "seed", tmp_path / "out")
    assert command[:3] == [sys.executable, "-m", "lcao2wannier"]
    for flag in ("--input", "--seed", "--output-dir", "--stage"):
        assert flag in command


def test_command_never_restates_his_validated_defaults(tmp_path):
    """--method hybrid, --mmn-method analytic and --hybrid-trust auto are
    already his defaults for named stages. Restating them is a no-op today and
    pins MACE against his future retuning."""
    command = build_command(tmp_path / "d.out", "seed", tmp_path / "out")
    for flag in ("--method", "--mmn-method", "--hybrid-trust"):
        assert flag not in command


def test_command_never_chooses_the_k_grid(tmp_path):
    """He derives the Monkhorst-Pack mesh from the parent's own SHRINK factors,
    which is the mesh the SCF actually used. A MACE-chosen grid would silently
    decouple the Wannier mesh from the SCF mesh - the same silent-wrong-model
    class as a guessed R-vector count, in a second place."""
    command = build_command(tmp_path / "d.out", "seed", tmp_path / "out")
    assert "--k-grid" not in command


def test_command_never_passes_quiet(tmp_path):
    """--quiet is how the reason gets swallowed.

    MEASURED: it filters stdout to lines containing warning/error/failed/abort,
    and his self-audit banner ("STATUS: FAIL - wannier90 will reject this
    setup:") and its violation bullets contain none of those tokens. Quiet mode
    keeps the symptom and drops every reason.
    """
    command = build_command(tmp_path / "d.out", "seed", tmp_path / "out",
                            threads=8, wannier90=tmp_path / "w90.x")
    assert "--quiet" not in command


def test_threads_are_passed_only_when_known(tmp_path):
    """His scheduler detection already reads SLURM_CPUS_PER_TASK, so under a
    MACE SLURM job omitting --threads is correct."""
    assert "--threads" not in build_command(tmp_path / "d.out", "s", tmp_path)
    assert "--threads" in build_command(tmp_path / "d.out", "s", tmp_path, threads=8)


def test_user_passthrough_arguments_come_last_and_unmodified(tmp_path):
    command = build_command(tmp_path / "d.out", "s", tmp_path,
                            extra_args=["--proj-threshold", "0.8"])
    assert command[-2:] == ["--proj-threshold", "0.8"]


def test_seed_is_reduced_to_a_bare_basename():
    """He validates and exits 2 on a seed with a path separator."""
    assert sanitize_seed("a/b/material_matdump") == "material_matdump"
    with pytest.raises(Lcao2WannierUnavailable):
        sanitize_seed("..")


# --------------------------------------------------------------------------
# The audit verdict - polarity is the whole point
# --------------------------------------------------------------------------


def test_a_pass_verdict_is_recognised():
    verdict, text = _classify_audit(
        "  STATUS: ✓ PASS — satisfies Wannier90 disentanglement rules\n")
    assert verdict == "pass"
    assert "PASS" in text


def test_a_fail_verdict_is_recognised_and_kept_verbatim():
    verdict, text = _classify_audit(
        "  STATUS: ✗ FAIL — wannier90 will reject this setup:\n"
        "    • dis_froz_max too high\n")
    assert verdict == "fail"
    assert "FAIL" in text


def test_a_bad_conditioning_verdict_is_a_failure():
    """MEASURED live: the corpus diamond dumped at N=60 - the hand-written value
    this feature started from - produced 'STATUS: BAD', cond(S)=inf at 80 of 100
    k-points. That is a refusal and must not read as anything else."""
    assert _classify_audit("  STATUS: ✗ BAD\n")[0] == "fail"


def test_an_absent_verdict_is_unknown_and_never_a_pass():
    """Success requires FINDING a PASS, never merely failing to find a FAIL.

    Reintroducing the bug - defining success as "no FAIL marker in stdout" -
    makes this assertion fail, and would silently start reporting refused models
    as successes the moment he rewords the banner.
    """
    assert _classify_audit("everything looks fine to me\n")[0] == "unknown"


def test_success_requires_all_three_conditions(tmp_path):
    """rc == 0 alone is not success.

    His disentanglement audit is deliberately non-fatal - "a violation is
    reported, not fatal, so the user still gets files" - so a refused model
    exits 0 with all five files present.
    """
    complete = {"seed": list(HANDOFF_SUFFIXES)}
    base = dict(returncode=0, stdout="", stderr="", command=[],
                output_dir=tmp_path, seed="seed")

    assert ConversionResult(produced=complete, audit="pass", **base).ok
    assert not ConversionResult(produced=complete, audit="fail", **base).ok
    assert not ConversionResult(produced=complete, audit="unknown", **base).ok
    assert not ConversionResult(produced={"seed": [".win"]}, audit="pass", **base).ok
    assert not ConversionResult(produced={}, audit="pass", **base).ok
    assert not ConversionResult(
        produced=complete, audit="pass",
        **{**base, "returncode": 1}).ok


# --------------------------------------------------------------------------
# The >= 1000 cell-index guard
# --------------------------------------------------------------------------


def test_cells_are_counted_from_the_real_measured_dump_shape(tmp_path):
    assert count_overlap_cells(_dump(tmp_path, 1247)) == 1247


def test_a_dump_past_cell_999_is_refused_on_the_affected_version(tmp_path):
    """MEASURED on the real corpus material at its derived N = 1247: the file
    holds 1247 overlap headers, lcao2wannier 1.0.0's regex matches 999, and 248
    are dropped silently while the run reports a plausible R-vector count.

    Reintroducing the bug - converting anyway - makes this assertion fail.
    """
    message = check_parent_dump(_dump(tmp_path, 1247), "1.0.0")
    assert message is not None
    assert "1247" in message and "248" in message
    assert "CELL N.1000(" in message
    # It must name the cause and the upstream fix, not just refuse.
    assert "William Comaskey" in message


def test_a_dump_below_the_limit_converts_on_the_affected_version(tmp_path):
    assert check_parent_dump(_dump(tmp_path, CELL_INDEX_PARSE_LIMIT - 1), "1.0.0") is None


def test_the_guard_retires_itself_on_a_fixed_version(tmp_path):
    """Version-checked, not blind: MACE does not pin lcao2wannier, so a
    version-blind refusal would keep firing after he ships the one-character
    fix, with nothing in the design describing how it would ever be retired."""
    assert check_parent_dump(_dump(tmp_path, 1247), "1.1.0") is None


def test_layer_one_is_not_gated_on_this_layer_two_defect(tmp_path):
    """The deck generator must still produce a deck at N >= 1000.

    Layer 1 is stock CRYSTAL and ships unconditionally; lcao2wannier may not be
    installed at all. Refusing a valid CRYSTAL dump because of a bug in an
    optional third-party package would break the one thing Phase 1 claims - that
    it is useful on its own, for someone who runs the rest by hand.
    """
    out = find_data("SP/1_dia*sp*.out", must_contain="MAX G-VECTOR INDEX")
    import shutil

    staged = tmp_path / out.name
    shutil.copy2(out, staged)
    shutil.copy2(out.with_suffix(".f9"), tmp_path / f"{out.stem}.f9")

    result = subprocess.run(
        [sys.executable, str(MACE_CLI), "--no-banner", "opt2d3",
         "--input", str(staged), "--calc-type", "MATDUMP"],
        capture_output=True, text=True, cwd=str(REPO_ROOT))
    deck = next(tmp_path.glob("*_matdump.d3"))
    assert deck.read_text() == "BASISSET\n2\n60 1247\n64 1247\nEND"
    # But it must warn, so the user is not surprised at conversion time.
    assert "1000" in (result.stdout + result.stderr)


# --------------------------------------------------------------------------
# Missing dependencies produce messages, never tracebacks
# --------------------------------------------------------------------------


def test_the_missing_dependency_message_is_actionable():
    message = describe_missing_dependency()
    assert "pip install lcao2wannier" in message
    assert "OPTIONAL" in message
    assert "William Comaskey" in message
    # It must say the deck is still usable: Layer 1 is unaffected.
    assert "already valid" in message


def test_the_missing_wannier90_message_is_actionable():
    message = describe_missing_wannier90()
    assert "--wannier90" in message
    assert "user-supplied" in message
    # The hand-off completes without it; that is the useful part.
    assert ".mmn" in message


def test_no_citation_string_is_invented():
    """The spec's OPEN says to ask him which citation he wants and not to invent
    one. A provisional string in shipped documentation would misattribute a
    publication on his behalf, which is worse than having none."""
    assert LCAO2WANNIER_CITATION is None


def test_a_missing_dump_is_refused_without_a_traceback(tmp_path):
    with pytest.raises(Lcao2WannierUnavailable):
        convert(tmp_path / "nope.out")


def test_the_cli_reports_a_missing_dependency_cleanly(tmp_path):
    """The real invocation path, in an interpreter without the package."""
    if find_lcao2wannier()[0]:
        pytest.skip("lcao2wannier is installed in this interpreter")
    dump = _dump(tmp_path, 60)
    result = subprocess.run(
        [sys.executable, str(MACE_CLI), "--no-banner", "wannier",
         "--input", str(dump)],
        capture_output=True, text=True, cwd=str(REPO_ROOT))
    combined = result.stdout + result.stderr
    assert result.returncode == 2
    assert "Traceback" not in combined
    assert "pip install lcao2wannier" in combined


def test_localize_without_a_wannier90_path_is_refused_before_running(tmp_path):
    dump = _dump(tmp_path, 60)
    result = subprocess.run(
        [sys.executable, str(MACE_CLI), "--no-banner", "wannier",
         "--input", str(dump), "--stage", "localize"],
        capture_output=True, text=True, cwd=str(REPO_ROOT))
    combined = result.stdout + result.stderr
    assert result.returncode == 2
    assert "Traceback" not in combined
    assert "--wannier90" in combined


def test_the_wannier_help_credits_the_author():
    result = subprocess.run(
        [sys.executable, str(MACE_CLI), "--no-banner", "wannier", "--help"],
        capture_output=True, text=True, cwd=str(REPO_ROOT))
    combined = result.stdout + result.stderr
    assert "William Comaskey" in combined
    assert "CITATION: TODO" in combined
    assert "optional dependency" in combined


# --------------------------------------------------------------------------
# With the optional dependency actually installed
# --------------------------------------------------------------------------


def test_a_structurally_invalid_parent_is_refused_with_one_clean_line(tmp_path):
    """His argparse refusals are a single line on stderr with exit code 2, and
    MACE surfaces that verbatim rather than turning it into a traceback."""
    pytest.importorskip("lcao2wannier")
    junk = tmp_path / "junk.out"
    junk.write_text("hello\n")
    with pytest.raises(Lcao2WannierUnavailable) as exc:
        convert(junk, seed="j", output_dir=tmp_path / "out")
    assert "Traceback" not in str(exc.value)
    assert "lcao2wannier" in str(exc.value)


def test_a_bad_wannier90_path_is_refused_by_his_own_preflight(tmp_path):
    pytest.importorskip("lcao2wannier")
    dump = _dump(tmp_path, 60)
    with pytest.raises(Lcao2WannierUnavailable) as exc:
        convert(dump, seed="m", output_dir=tmp_path / "out",
                wannier90=Path("/nonexistent/wannier90.x"))
    assert "Traceback" not in str(exc.value)
