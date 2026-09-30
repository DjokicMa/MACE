"""The lcao2wannier driver (Layer 2).

``lcao2wannier`` is William Comaskey's package, bundled with MACE as
``mace.wannier.lcao2wannier`` (MIT). Everything here runs everywhere, including
the corpus-less CI; the one test that needs a real 2c-SOC dump lives in
test_wannier_vendored.py and is gated on MACE_W90_REFDATA.

MACE's job here is narrow: build an argument list, run it, and report what he
says. So most of these tests assert what MACE does NOT do.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, find_data

import mace.wannier.driver as driver
from mace.wannier.driver import (
    ConversionResult,
    HANDOFF_SUFFIXES,
    LCAO2WANNIER_CITATION,
    LCAO2WANNIER_MODULE,
    Lcao2WannierUnavailable,
    _classify_audit,
    _conditioning_report,
    build_command,
    convert,
    describe_missing_wannier90,
    sanitize_seed,
)

MACE_CLI = REPO_ROOT / "mace_cli"


def _dump(tmp_path, cells, name="material_matdump.out"):
    """A matrix dump with a given number of cell headers.

    Only the headers matter to the tests using it, and CRYSTAL's I4 field is
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
    # The bundled copy, never whatever "lcao2wannier" happens to be installed.
    assert command[:3] == [sys.executable, "-m", "mace.wannier.lcao2wannier"]
    assert LCAO2WANNIER_MODULE == "mace.wannier.lcao2wannier"
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


# His conditioning check prints its own STATUS line, from a different module
# (conditioning.py:140) than the disentanglement audit (wannier_checks.py:181),
# with a THREE-valued verdict: GOOD / MARGINAL / BAD. The first version of this
# classifier matched PASS and FAIL|BAD and nothing else, so MARGINAL and GOOD
# both fell through it entirely - and MARGINAL is the one negative signal MACE
# was discarding. Both end-to-end validation runs hit it (worst cond(S) =
# 2.530e+04, "could benefit from more R-vectors") and were reported as
# unqualified successes.

REAL_MARGINAL_REPORT = """\

======================================================================
Overlap Matrix Conditioning Check
======================================================================
  R-vectors in Fourier sum:     999
  Orbital space dimension:      36 x 36
  K-points sampled:             100

  Worst condition number:       2.530e+04  at k = (0.000, 0.000, 0.000)
  Median condition number:      1.204e+03
  Min eigenvalue of S(k):       3.101e-04  at k = (0.500, 0.500, 0.500)

  K-points not positive def.:   0 / 100 (0.0%)
  K-points needing regulariz.:  0 / 100 (0.0%)

  STATUS: \u26a0 MARGINAL

  The overlap matrices show moderate ill-conditioning.
  Results may be acceptable but could benefit from more R-vectors.
======================================================================
  STATUS: \u2713 PASS \u2014 satisfies Wannier90 disentanglement rules
"""


def test_a_marginal_conditioning_verdict_is_not_swallowed():
    """Reintroducing the bug - matching only PASS and FAIL|BAD - classifies
    this stdout as a clean 'pass' and drops the MARGINAL line from audit_text
    entirely, which is exactly what shipped."""
    verdict, text = _classify_audit(REAL_MARGINAL_REPORT)
    assert verdict == "marginal"
    assert "MARGINAL" in text
    # Severity wins over count: the PASS on the same stdout must not mask it.
    assert "PASS" in text


def test_a_good_conditioning_verdict_is_a_pass():
    """GOOD was unmatched too - harmless, but the marker set must cover all
    three conditioning verdicts explicitly rather than by accident."""
    assert _classify_audit("  STATUS: \u2713 GOOD\n")[0] == "pass"


def test_a_bad_verdict_still_outranks_a_marginal_one():
    assert _classify_audit(
        "  STATUS: \u26a0 MARGINAL\n  STATUS: \u2717 BAD\n")[0] == "fail"


def test_a_marginal_run_is_usable_but_qualified(tmp_path):
    """His own text says "results may be acceptable", so MARGINAL must not be
    reported as a failure - and must not be reported as a clean success either.
    """
    complete = {"seed": list(HANDOFF_SUFFIXES)}
    base = dict(returncode=0, stdout="", stderr="", command=[],
                output_dir=tmp_path, seed="seed")
    marginal = ConversionResult(produced=complete, audit="marginal", **base)
    assert marginal.ok
    assert marginal.qualified
    clean = ConversionResult(produced=complete, audit="pass", **base)
    assert clean.ok and not clean.qualified


def test_the_conditioning_report_is_reproduced_whole():
    """The numbers and his remedy are what tell a user whether to re-run at a
    larger R-vector count; a bare word 'MARGINAL' tells them nothing."""
    block = _conditioning_report(REAL_MARGINAL_REPORT)
    assert "Overlap Matrix Conditioning Check" in block
    assert "2.530e+04" in block
    assert "could benefit from more R-vectors" in block
    # and it stops at his closing rule rather than swallowing the rest
    assert "disentanglement" not in block


def test_no_conditioning_report_yields_an_empty_block():
    assert _conditioning_report("STATUS: \u2713 PASS\n") == ""


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
    assert ConversionResult(produced=complete, audit="marginal", **base).ok
    assert not ConversionResult(produced=complete, audit="fail", **base).ok
    assert not ConversionResult(produced=complete, audit="unknown", **base).ok
    assert not ConversionResult(produced={"seed": [".win"]}, audit="pass", **base).ok
    assert not ConversionResult(produced={}, audit="pass", **base).ok
    assert not ConversionResult(
        produced=complete, audit="pass",
        **{**base, "returncode": 1}).ok


# --------------------------------------------------------------------------
# The >= 1000 cell-index guard is gone
# --------------------------------------------------------------------------


class _Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def test_a_dump_past_cell_999_is_no_longer_refused(tmp_path, monkeypatch):
    """MACE used to refuse dumps with >= 1000 cells because stock lcao2wannier
    1.0.0 dropped every cell from N.1000 on (MEASURED on the corpus diamond at
    N = 1247: 999 read, 248 dropped). That refusal existed only because of the
    reader; the bundled copy reads the I4 index (test_wannier_vendored.py), so
    the dump must now reach the package."""
    calls = []
    monkeypatch.setattr(driver.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or _Completed())
    convert(_dump(tmp_path, 1247), seed="m", output_dir=tmp_path / "out")
    assert len(calls) == 2                      # preflight, then the run
    assert "--dry-run" in calls[0] and "--dry-run" not in calls[1]


def test_the_child_can_import_the_bundled_copy_from_any_cwd(tmp_path, monkeypatch):
    """mace_cli finds the package because it runs from the repository root. The
    conversion child does not, so the driver puts that root on its PYTHONPATH -
    otherwise it would die with "No module named mace"."""
    envs = []
    monkeypatch.setattr(driver.subprocess, "run",
                        lambda cmd, **kw: envs.append(kw.get("env")) or _Completed())
    monkeypatch.setenv("PYTHONPATH", "/some/user/path")
    convert(_dump(tmp_path, 60), seed="m", output_dir=tmp_path / "out")
    for env in envs:
        parts = env["PYTHONPATH"].split(os.pathsep)
        assert parts[0] == str(REPO_ROOT)
        assert "/some/user/path" in parts     # the user's own entries survive


def test_layer_one_is_not_gated_on_this_layer_two_defect(tmp_path):
    """The deck generator must still produce a deck at N >= 1000.

    Layer 1 is stock CRYSTAL and ships unconditionally. The dump may be
    converted elsewhere, by hand, with any reader; refusing a valid CRYSTAL
    dump because of a reader's bug would break the one thing Phase 1 claims -
    that it is useful on its own, for someone who runs the rest by hand. It
    still says so, because a stock lcao2wannier 1.0.0 drops cells >= 1000.
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


def test_a_broken_numerical_stack_is_reported_not_raised(tmp_path, monkeypatch):
    """The package itself always ships; what can still be missing is numpy or
    scipy. The --dry-run preflight is what exposes that, and it must reach the
    user as a message naming them, not as the child's traceback alone."""
    monkeypatch.setattr(
        driver.subprocess, "run",
        lambda cmd, **kw: _Completed(1, "", "ModuleNotFoundError: No module named 'scipy'"))
    with pytest.raises(Lcao2WannierUnavailable) as exc:
        convert(_dump(tmp_path, 60), seed="m", output_dir=tmp_path / "out")
    assert "numpy/scipy" in str(exc.value)
    assert "scipy" in str(exc.value).splitlines()[-1]


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


def test_the_cli_runs_the_bundled_copy_from_outside_the_repository(tmp_path):
    """The real invocation path, started from a directory that is not the repo
    root: the bundled package must still be found and must be the one that
    answers (its own clean refusal of a non-dump), never an import error."""
    junk = tmp_path / "junk.out"
    junk.write_text("hello\n")
    result = subprocess.run(
        [sys.executable, str(MACE_CLI), "--no-banner", "wannier",
         "--input", str(junk), "--output-dir", str(tmp_path / "out")],
        capture_output=True, text=True, cwd=str(tmp_path))
    combined = result.stdout + result.stderr
    assert result.returncode == 2, combined
    assert "Traceback" not in combined
    assert "No module named" not in combined
    assert "refused these arguments" in combined


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
    assert "lcao2wannier" in combined and "cite" in combined.lower()
    # user-facing text must not carry developer instructions
    assert "do not invent" not in combined.lower()
    assert "CITATION: TODO" not in combined
    # bundled now, not something to install separately
    assert "bundled" in combined
    assert "pip install lcao2wannier" not in combined


# --------------------------------------------------------------------------
# Through the bundled package itself
# --------------------------------------------------------------------------


def test_a_structurally_invalid_parent_is_refused_with_one_clean_line(tmp_path):
    """His argparse refusals are a single line on stderr with exit code 2, and
    MACE surfaces that verbatim rather than turning it into a traceback."""
    junk = tmp_path / "junk.out"
    junk.write_text("hello\n")
    with pytest.raises(Lcao2WannierUnavailable) as exc:
        convert(junk, seed="j", output_dir=tmp_path / "out")
    assert "Traceback" not in str(exc.value)
    assert "lcao2wannier" in str(exc.value)


def test_a_bad_wannier90_path_is_refused_by_his_own_preflight(tmp_path):
    dump = _dump(tmp_path, 60)
    with pytest.raises(Lcao2WannierUnavailable) as exc:
        convert(dump, seed="m", output_dir=tmp_path / "out",
                wannier90=Path("/nonexistent/wannier90.x"))
    assert "Traceback" not in str(exc.value)


# --------------------------------------------------------------------------
# What `mace wannier` REPORTS for each verdict
# --------------------------------------------------------------------------
#
# The classifier and the report are separate failures. Before this, a MARGINAL
# run reached cli.py as audit == "pass" and printed an unqualified
# "Hand-off complete." with exit 0 - VERIFIED end to end against the real 2c-SOC
# bismuth parent: his conditioning block was in stdout and MACE said nothing
# about it.


def _result(tmp_path, **kwargs):
    from mace.wannier.driver import ConversionResult

    base = dict(returncode=0, stdout="", stderr="", command=[],
                output_dir=tmp_path, seed="seed",
                produced={"seed": list(HANDOFF_SUFFIXES)})
    base.update(kwargs)
    return ConversionResult(**base)


def _run_cli(monkeypatch, capsys, result):
    import mace.wannier.cli as cli

    monkeypatch.setattr(cli, "convert", lambda **kwargs: result)
    code = cli.main(["--input", "dump_matdump.out"])
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def test_the_cli_never_reports_a_marginal_model_as_an_unqualified_success(
        tmp_path, monkeypatch, capsys):
    """VERIFIED end to end: driving the real lcao2wannier 1.0.0 on the real
    2c-SOC bismuth parent at a lowered marginal threshold reproduced his
    "STATUS: MARGINAL" banner, and MACE now prints his report whole with the
    remedy and a qualified closing line.

    Reintroducing the bug - letting MARGINAL classify as "pass" - prints
    "Hand-off complete." and this fails.
    """
    result = _result(
        tmp_path, audit="marginal",
        audit_text="  STATUS: ⚠ MARGINAL",
        conditioning_text=REAL_MARGINAL_REPORT.strip())
    code, text = _run_cli(monkeypatch, capsys, result)
    assert code == 0                       # his verdict is a caveat, not a refusal
    assert "Hand-off complete." not in text
    assert "QUALIFIED" in text
    assert "could benefit from more R-vectors" in text   # his own remedy
    assert "2.530e+04" in text                            # his own numbers
    assert "TOLINTEG" in text                             # what to do about it


def test_the_cli_still_reports_a_clean_pass_plainly(tmp_path, monkeypatch,
                                                    capsys):
    code, text = _run_cli(monkeypatch, capsys, _result(tmp_path, audit="pass"))
    assert code == 0
    assert "Hand-off complete." in text
    assert "QUALIFIED" not in text


def test_the_cli_still_fails_a_refused_model(tmp_path, monkeypatch, capsys):
    result = _result(tmp_path, audit="fail",
                     audit_text="  STATUS: ✗ FAIL — wannier90 will reject this setup:")
    code, text = _run_cli(monkeypatch, capsys, result)
    assert code == 1
    assert "REFUSED" in text
    assert "Hand-off complete." not in text
