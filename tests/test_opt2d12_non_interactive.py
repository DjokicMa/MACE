"""`mace opt2d12 --calc-type X --non-interactive` must work with nothing on stdin.

That is the form MACE's own help documents, and in v1.1.1 it died with
EOFError at the first settings prompt: --non-interactive combined with
--calc-type still walks the interactive settings flow, because the workflow
engine drives that flow by piping scripted answers to stdin. The fix keeps the
extracted settings once the answers run out. These tests pin both halves:
the bare command succeeds, and answers that ARE supplied are still honoured, so
the fallback cannot quietly change what the engine generates.
"""
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, TEST_DATA

MACE_CLI = REPO_ROOT / "mace_cli"
PARENT = "1_dia_opt_BULK_OPTGEOM"


@pytest.fixture
def parent(tmp_path):
    src = TEST_DATA / "OPT" / f"{PARENT}.out"
    if not src.exists():
        pytest.skip("test/ corpus not present (gitignored, ~12GB)")
    shutil.copy(src, tmp_path)
    shutil.copy(src.with_suffix(".d12"), tmp_path)
    return tmp_path


def _run(cwd, calc_type, stdin_text):
    return subprocess.run(
        [sys.executable, str(MACE_CLI), "opt2d12", "--out-file", f"{PARENT}.out",
         "--calc-type", calc_type, "--non-interactive"],
        cwd=cwd, input=stdin_text, capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize("calc_type,block,absent", [
    ("SP", None, ("OPTGEOM", "FREQCALC")),
    ("OPT", "OPTGEOM", ("FREQCALC",)),
    ("FREQ", "FREQCALC", ("OPTGEOM",)),
])
def test_documented_command_succeeds_with_nothing_on_stdin(parent, calc_type, block, absent):
    result = _run(parent, calc_type, "")
    combined = result.stdout + result.stderr
    assert "EOFError" not in combined, combined[-1500:]
    assert result.returncode == 0, combined[-1500:]
    assert "No answers on stdin" in combined
    deck = parent / f"{PARENT}_{calc_type.lower()}_B3LYP-D3_optimized.d12"
    assert deck.exists(), sorted(p.name for p in parent.iterdir())
    body = deck.read_text().splitlines()[1:]          # skip the title line
    if block:
        assert block in body, f"{calc_type} deck lacks {block}"
    for other in absent:
        assert other not in body, f"{calc_type} deck carries a stray {other}"


def test_supplied_answers_are_still_honoured(parent):
    """The fallback fires only at end-of-file, so scripted answers - the
    workflow engine's mechanism - are consumed exactly as before."""
    engine_answers = "n\n" + "\n" * 19               # mace/workflow/engine.py
    result = _run(parent, "SP", engine_answers)
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined[-1500:]
    assert "No answers on stdin" not in combined


def test_output_dir_is_honoured_and_stays_out_of_the_title(parent):
    """--output-dir was parsed and the directory created, but the path never
    reached the writer, so the deck landed in the current directory. The
    directory must also not leak into the CRYSTAL title line."""
    dest = parent / "dest"
    result = subprocess.run(
        [sys.executable, str(MACE_CLI), "opt2d12", "--out-file", f"{PARENT}.out",
         "--calc-type", "SP", "--non-interactive", "--output-dir", "dest"],
        cwd=parent, input="", capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, (result.stdout + result.stderr)[-1500:]
    name = f"{PARENT}_sp_B3LYP-D3_optimized.d12"
    assert (dest / name).exists(), sorted(p.name for p in parent.rglob("*.d12"))
    assert not (parent / name).exists(), "deck also written to the current directory"
    assert (dest / name).read_text().splitlines()[0] == name[:-len(".d12")]
