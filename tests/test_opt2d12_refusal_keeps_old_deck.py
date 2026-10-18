"""A rerun whose new deck is refused must leave the older deck of that name.

Some refusals come after the deck is written (the charge-record checks read
the written deck), and opt2d12 then deleted it. But the new deck had already
replaced an older one of the same name - a deck from an earlier run, maybe
already submitted - so the old deck was lost too. The old deck is now put
back on any refusal, and nothing is left behind on success.
"""
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, TEST_DATA

MACE_CLI = REPO_ROOT / "mace_cli"
PARENT = "1_dia_opt_BULK_OPTGEOM"
DECK = f"{PARENT}_sp_B3LYP-D3_optimized.d12"
OLD = "an older deck of the same name\n"


@pytest.fixture
def parent(tmp_path):
    src = TEST_DATA / "OPT" / f"{PARENT}.out"
    if not src.exists():
        pytest.skip("test/ corpus not present (gitignored, ~12GB)")
    shutil.copy(src, tmp_path)
    shutil.copy(src.with_suffix(".d12"), tmp_path)
    return tmp_path


def _opt2d12(cwd, *extra):
    return subprocess.run(
        [sys.executable, str(MACE_CLI), "opt2d12", "--out-file", f"{PARENT}.out",
         "--calc-type", "SP", "--non-interactive", *extra],
        cwd=cwd, input="", capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize("extra", [
    ("--chemod", "yes"),                      # CHEMOD without a charge change
    ("--fragment-bond-scale", "1.4"),         # same refusal, other flag
])
def test_refused_rerun_keeps_the_old_deck(parent, extra):
    (parent / DECK).write_text(OLD)
    result = _opt2d12(parent, *extra)
    combined = result.stdout + result.stderr
    assert result.returncode != 0, combined[-1500:]
    assert "only take effect with a charge change" in combined, combined[-1500:]
    assert (parent / DECK).exists(), "the older deck was deleted"
    assert (parent / DECK).read_text() == OLD
    assert sorted(p.name for p in parent.iterdir()) == sorted(
        [f"{PARENT}.out", f"{PARENT}.d12", DECK])


def test_refused_first_run_leaves_no_deck(parent):
    result = _opt2d12(parent, "--chemod", "yes")
    assert result.returncode != 0
    assert sorted(p.name for p in parent.iterdir()) == sorted([f"{PARENT}.out", f"{PARENT}.d12"])


def test_successful_rerun_replaces_the_old_deck_and_leaves_no_backup(parent):
    (parent / DECK).write_text(OLD)
    result = _opt2d12(parent)
    assert result.returncode == 0, (result.stdout + result.stderr)[-1500:]
    assert (parent / DECK).read_text() != OLD
    assert sorted(p.name for p in parent.iterdir()) == sorted(
        [f"{PARENT}.out", f"{PARENT}.d12", DECK])
