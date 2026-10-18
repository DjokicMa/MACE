"""`mace opt2d12 --calc-type X --non-interactive` must work with nothing on stdin.

That is the form MACE's own help documents, and in v1.1.1 it died with
EOFError at the first settings prompt: --non-interactive combined with
--calc-type still walks the interactive settings flow, because the workflow
engine drives that flow by piping scripted answers to stdin. The fix keeps the
extracted settings once the answers run out. These tests pin both halves:
the bare command succeeds, and answers that ARE supplied are still honored, so
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


def test_supplied_answers_are_still_honored(parent):
    """The fallback fires only at end-of-file, so scripted answers - the
    workflow engine's mechanism - are consumed exactly as before."""
    engine_answers = "n\n" + "\n" * 19               # mace/workflow/engine.py
    result = _run(parent, "SP", engine_answers)
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined[-1500:]
    assert "No answers on stdin" not in combined


def test_output_dir_is_honored_and_stays_out_of_the_title(parent):
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


# --non-interactive never waits for an answer. Answers are read only when
# they are already on stdin (the workflow engine's piped answers); a pipe that
# stays open with nothing on it (a GUI or script that never closes the child's
# stdin), or a terminal, is not someone to ask. In 1.1.6 previews the run
# blocked on "Use these exact settings for the new calculation?" forever.

CONFIRM = "Use these exact settings for the new calculation?"


def _popen(cwd, args, **kw):
    return subprocess.Popen(
        [sys.executable, str(MACE_CLI), "opt2d12", "--out-file", f"{PARENT}.out", *args],
        cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)


def _finish(proc, timeout=120):
    """Wait for the run with its stdin left as it is (communicate() would
    close a pipe, which is end of input, not an open pipe)."""
    import threading
    chunks = []
    reader = threading.Thread(target=lambda: chunks.append(proc.stdout.read()), daemon=True)
    reader.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        reader.join(10)
        pytest.fail(f"opt2d12 --non-interactive waited for an answer:\n{''.join(chunks)[-800:]}")
    reader.join(30)
    return "".join(chunks)


@pytest.mark.parametrize("calc_type", ["SP", "FREQ"])
def test_open_empty_pipe_is_not_waited_on(parent, calc_type):
    proc = _popen(parent, ["--calc-type", calc_type, "--non-interactive"], stdin=subprocess.PIPE)
    try:
        out = _finish(proc)
    finally:
        proc.stdin.close()
    assert proc.returncode == 0, out[-1500:]
    assert CONFIRM not in out, out[-1500:]
    assert "No answers on stdin" in out
    assert (parent / f"{PARENT}_{calc_type.lower()}_B3LYP-D3_optimized.d12").exists()


def test_directory_mode_open_empty_pipe_is_not_waited_on(parent):
    proc = subprocess.Popen(
        [sys.executable, str(MACE_CLI), "opt2d12", "--directory", ".", "--calc-type", "SP",
         "--non-interactive"],
        cwd=parent, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True)
    try:
        out = _finish(proc)
    finally:
        proc.stdin.close()
    assert proc.returncode == 0, out[-1500:]
    assert CONFIRM not in out, out[-1500:]
    assert (parent / f"{PARENT}_sp_B3LYP-D3_optimized.d12").exists()


def test_answers_on_a_pipe_left_open_are_still_read(parent):
    """The engine's answers, written to a pipe the writer keeps open: read
    as before (the deck equals the one from the same answers and a closed
    pipe), and the run does not wait once they are used up."""
    engine_answers = "n\n" + "\n" * 39
    closed = _run(parent, "SP", engine_answers)
    assert closed.returncode == 0, (closed.stdout + closed.stderr)[-1500:]
    deck = parent / f"{PARENT}_sp_B3LYP-D3_optimized.d12"
    expected = deck.read_text()
    deck.unlink()

    proc = _popen(parent, ["--calc-type", "SP", "--non-interactive"], stdin=subprocess.PIPE)
    proc.stdin.write(engine_answers)
    proc.stdin.flush()
    try:
        out = _finish(proc)
    finally:
        proc.stdin.close()
    assert proc.returncode == 0, out[-1500:]
    assert "No answers on stdin" not in out
    assert deck.read_text() == expected


def test_a_terminal_is_not_asked(parent):
    pty = pytest.importorskip("pty")
    import os
    master, slave = pty.openpty()
    try:
        proc = _popen(parent, ["--calc-type", "SP", "--non-interactive"], stdin=slave)
        os.close(slave)
        out = _finish(proc)
    finally:
        os.close(master)
    assert proc.returncode == 0, out[-1500:]
    assert CONFIRM not in out, out[-1500:]
    assert (parent / f"{PARENT}_sp_B3LYP-D3_optimized.d12").exists()
