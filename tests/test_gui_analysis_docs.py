"""docs/gui.md and `mace analyze --help` match the analysis and database
features (PLAN_analysis C12): the CLI help lists --convergence and the
database-build flags (subprocess, the real entry point), the guide no longer
calls the database read-only, and its guarantee sentence is the dialog's."""
import os
import subprocess
import sys

from conftest import REPO_ROOT

DOC = (REPO_ROOT / "docs" / "gui.md").read_text()


def test_analyze_help_lists_the_new_flags():
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    p = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "analyze", "--help"],
                       capture_output=True, text=True, env=env, timeout=120)
    out = p.stdout + p.stderr
    for flag in ("--convergence", "--register", "--dry-run", "--skip-unchanged", "--backup", "--keep-backups",
                 "--progress-json"):
        assert flag in out, flag


def test_database_is_no_longer_called_read_only():
    assert "read-only in this version" not in DOC
    html = (REPO_ROOT / "mace" / "gui" / "static" / "index.html").read_text()
    assert "read-only in this version" not in html


def test_guarantee_sentence_is_the_dialogs():
    sys.path.insert(0, str(REPO_ROOT))
    from mace.gui import dbbuild
    assert dbbuild.GUARANTEE in DOC


def test_sections_are_there():
    for h in ("## Analyzing a run", "### Plot options", "### Building a database from a folder",
              "### What the GUI writes into your folder", "### Runs and files that are still being written"):
        assert h in DOC, h
    assert "](#what-the-gui-writes-into-your-folder)" in DOC
