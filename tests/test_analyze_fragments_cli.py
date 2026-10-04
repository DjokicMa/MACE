"""`mace analyze --extract-properties DIR --fragments`, through mace_cli.

The CHANGELOG gave the form `mace analyze --fragments`, which only prints the
usage line: `mace analyze` needs --extract-properties. The working form
computes the per-fragment Mulliken charges for a neutral run too (a charged
run gets them anyway). The output is a synthetic excerpt in CRYSTAL23 format,
modeled on a real HSEsol-3c/SOLDEF2MSVP run: a hand-built, neutral Li+ FSI-
ethylene carbonate molecule (20 atoms, 138 electrons; Mulliken Li +0.80,
FSI -0.80, EC 0.00).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from conftest import REPO_ROOT

OUT = Path(__file__).resolve().parent / "data" / "charged_runs" / "li_fsi_ec_neutral.out"


def _analyze(tmp_path, *args):
    shutil.copy(OUT, tmp_path / OUT.name)
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR")}
    return subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "analyze", *args,
                           "--db-path", str(tmp_path / "materials.db")],
                          cwd=tmp_path, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          timeout=300, env=env)


def test_extract_properties_with_fragments_reports_a_neutral_run(tmp_path):
    run = _analyze(tmp_path, "--extract-properties", ".", "--fragments")
    assert run.returncode == 0, run.stdout + run.stderr
    assert "Fragments (net charge +0, 138 electrons)" in run.stdout
    assert "FSI-" in run.stdout and "Li+" in run.stdout


def test_single_output_file_form_works_too(tmp_path):
    run = _analyze(tmp_path, "--extract-properties", "--output-file", OUT.name, "--fragments")
    assert run.returncode == 0, run.stdout + run.stderr
    assert "Fragments (net charge +0" in run.stdout


def test_without_fragments_a_neutral_run_gets_none(tmp_path):
    run = _analyze(tmp_path, "--extract-properties", ".")
    assert run.returncode == 0, run.stdout + run.stderr
    assert "Fragments" not in run.stdout


def test_the_help_and_changelog_give_the_working_form():
    help_text = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "analyze", "--help"],
                               capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL).stdout
    assert "--fragments" in help_text
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text()
    # The entry lives under [Unreleased] until a release dates it: read from
    # [Unreleased] through the newest released section.
    after = changelog.split("## [Unreleased]", 1)[1]
    flat = " ".join("\n## [".join(after.split("\n## [", 2)[:2]).split())
    assert "mace analyze --extract-properties <dir> --fragments" in flat
    assert "`mace analyze --fragments`" not in flat
