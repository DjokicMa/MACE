"""Applying one saved --config-file template to many OPT parents, through the
real CLI (`mace_cli opt2d12` as a subprocess) on copies of real corpus files.

The report: settings saved with `--save-options --options-file sp_template.json`
could be applied to one structure (`echo y | mace opt2d12 --out-file ...`), but
`--directory DIR --config-file sp_template.json` asked "Apply these settings
from config file?" for every file, and with nothing on stdin logged
"EOF when reading a line" per file, wrote no deck and exited 0. A template
saved from a parent with an EXTERNAL basis also failed on every other
structure ("External basis set path not configured properly").
"""
import os
import select
import shutil
import subprocess
import sys
import time

import pytest

from conftest import REPO_ROOT, TEST_DATA

MACE_CLI = REPO_ROOT / "mace_cli"

EXT = "Ag1Br1_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized"
INT = "Ag1Cl3_sym_CRYSTAL_OPT_symm_PBE-D3_POB-TZVP-REV2_opt_B3LYP-D3-D3_optimized"
DIA = "1_dia_opt_rev1"
MOL3C = "EC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized"
PARENTS = (EXT, INT, DIA)


def _deck(directory, stem, functional="B3LYP-D3"):
    return directory / f"{stem}_sp_{functional}_optimized.d12"


def _copy(stems, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        src = TEST_DATA / "OPT" / f"{stem}.out"
        if not src.exists():
            pytest.skip("test/ corpus not present (gitignored, ~12GB)")
        shutil.copy(src, dest)
        shutil.copy(src.with_suffix(".d12"), dest)
    return dest


def _cli(args, cwd, stdin=subprocess.DEVNULL, input_text=None):
    kwargs = dict(cwd=cwd, capture_output=True, text=True, timeout=300)
    if input_text is None:
        kwargs["stdin"] = stdin
    else:
        kwargs["input"] = input_text
    return subprocess.run([sys.executable, str(MACE_CLI), "opt2d12", *args], **kwargs)


def _save_template(tmp_path, stem):
    """A template saved as the report did: --calc-type SP --non-interactive
    --save-options, with nothing on stdin."""
    work = _copy([stem], tmp_path / f"save_{stem[:8]}")
    template = tmp_path / f"{stem[:8]}_template.json"
    result = _cli(["--out-file", f"{stem}.out", "--d12-file", f"{stem}.d12",
                   "--calc-type", "SP", "--non-interactive", "--output-dir", "saved",
                   "--save-options", "--options-file", str(template)], work)
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
    return template


@pytest.fixture
def ext_template(tmp_path):
    return _save_template(tmp_path, EXT)


def test_directory_batch_with_nothing_on_stdin_writes_every_deck(tmp_path, ext_template):
    parents = _copy(PARENTS, tmp_path / "opts")
    result = _cli(["--directory", ".", "--config-file", str(ext_template),
                   "--output-dir", "sp"], parents)
    out = result.stdout + result.stderr
    assert result.returncode == 0, out[-3000:]
    assert "EOF when reading a line" not in out
    assert "Apply these settings from config file?" not in out
    assert "3 written, 0 failed" in out
    for stem in PARENTS:
        assert _deck(parents / "sp", stem).exists(), stem


def test_several_out_files_match_the_directory_batch(tmp_path, ext_template):
    """--out-file with several paths (what a shell glob expands to): each
    .out is paired with its own .d12 and the decks are the directory batch's."""
    parents = _copy(PARENTS, tmp_path / "opts")
    by_dir = _cli(["--directory", ".", "--config-file", str(ext_template),
                   "--output-dir", "by_dir"], parents)
    outs = sorted(p.name for p in parents.glob("*.out"))
    by_list = _cli(["--out-file", *outs, "--config-file", str(ext_template),
                    "--yes", "--output-dir", "by_list"], parents)
    assert by_dir.returncode == 0 and by_list.returncode == 0, by_list.stderr[-2000:]
    for stem in PARENTS:
        assert (_deck(parents / "by_dir", stem).read_bytes()
                == _deck(parents / "by_list", stem).read_bytes()), stem


def test_batch_matches_one_file_at_a_time(tmp_path, ext_template):
    """The batch writes what the single-file config path writes for each file."""
    parents = _copy(PARENTS, tmp_path / "opts")
    result = _cli(["--directory", ".", "--config-file", str(ext_template),
                   "--output-dir", "batch"], parents)
    assert result.returncode == 0
    for stem in PARENTS:
        one = _cli(["--out-file", f"{stem}.out", "--d12-file", f"{stem}.d12",
                    "--config-file", str(ext_template), "--output-dir", "one"],
                   parents, input_text="y\n")
        assert one.returncode == 0, (one.stdout + one.stderr)[-2000:]
        assert (_deck(parents / "batch", stem).read_bytes()
                == _deck(parents / "one", stem).read_bytes()), stem


def test_a_failed_file_is_reported_and_fails_the_run(tmp_path):
    """A 3c template on a lead structure: SOLDEF2MSVP has no Pb, so that file
    fails with the reason, the others are written, and the exit status says so."""
    template = _save_template(tmp_path, MOL3C)
    lead = "TiPbO3_mp-19845_sg221_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized"
    parents = _copy([DIA, lead], tmp_path / "opts")
    result = _cli(["--directory", ".", "--config-file", str(template),
                   "--output-dir", "sp"], parents)
    out = result.stdout + result.stderr
    assert result.returncode == 1, out[-3000:]
    assert "1 written, 1 failed" in out
    assert f"{lead}.out: basis set 'SOLDEF2MSVP' has no basis for Pb" in out
    assert _deck(parents / "sp", DIA, "HSESOL3C").exists()
    assert not _deck(parents / "sp", lead, "HSESOL3C").exists()
    assert not list((parents / "sp").glob("*.tmp*"))


@pytest.mark.parametrize("stdin", ["devnull", "closed"])
def test_single_file_needs_nothing_on_stdin(tmp_path, ext_template, stdin):
    parents = _copy([INT], tmp_path / "opts")
    args = [sys.executable, str(MACE_CLI), "opt2d12", "--out-file", f"{INT}.out",
            "--d12-file", f"{INT}.d12", "--config-file", str(ext_template),
            "--output-dir", "sp"]
    if stdin == "devnull":
        result = subprocess.run(args, cwd=parents, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=300)
    else:
        read_end, write_end = os.pipe()
        os.close(write_end)
        with os.fdopen(read_end) as closed:
            result = subprocess.run(args, cwd=parents, stdin=closed,
                                    capture_output=True, text=True, timeout=300)
    out = result.stdout + result.stderr
    assert result.returncode == 0, out[-2000:]
    assert "EOF" not in out
    assert _deck(parents / "sp", INT).exists()


def test_piped_yes_still_works_for_one_file(tmp_path, ext_template):
    """The workflow engine's form: 'y' piped to a single --out-file."""
    parents = _copy([INT], tmp_path / "opts")
    result = _cli(["--out-file", f"{INT}.out", "--d12-file", f"{INT}.d12",
                   "--config-file", str(ext_template), "--output-dir", "sp"],
                  parents, input_text="y\n")
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
    assert _deck(parents / "sp", INT).exists()


def _run_at_a_terminal(args, cwd, timeout=300):
    """Run with a pty for stdin/stdout, answering every [Y/n] with 'y'.
    Returns (exit status, output, answers given)."""
    master, slave = os.openpty()
    env = dict(os.environ, NO_COLOR="1")
    proc = subprocess.Popen(args, cwd=cwd, stdin=slave, stdout=slave, stderr=slave,
                            env=env, close_fds=True)
    os.close(slave)
    output, answered = b"", 0
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            ready, _, _ = select.select([master], [], [], 0.2)
            if ready:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                output += chunk
                while output.count(b"[Y/n]") > answered:
                    os.write(master, b"y\n")
                    answered += 1
            elif proc.poll() is not None:
                break
        else:
            proc.kill()
            pytest.fail("run at a terminal did not finish")
    finally:
        os.close(master)
    return proc.wait(timeout=30), output.decode(errors="replace"), answered


def test_at_a_terminal_a_batch_asks_once(tmp_path, ext_template):
    parents = _copy(PARENTS, tmp_path / "opts")
    status, out, answered = _run_at_a_terminal(
        [sys.executable, str(MACE_CLI), "opt2d12", "--directory", ".",
         "--config-file", str(ext_template), "--output-dir", "sp"], parents)
    assert status == 0, out[-3000:]
    assert out.count("Apply to all 3 files? [Y/n]") == 1
    assert "Apply these settings from config file?" not in out
    assert answered == 1
    assert "3 written, 0 failed" in out


def test_at_a_terminal_yes_skips_the_question(tmp_path, ext_template):
    parents = _copy(PARENTS, tmp_path / "opts")
    status, out, answered = _run_at_a_terminal(
        [sys.executable, str(MACE_CLI), "opt2d12", "--directory", ".", "-y",
         "--config-file", str(ext_template), "--output-dir", "sp"], parents)
    assert status == 0, out[-3000:]
    assert answered == 0
    assert "Apply to all" not in out


def test_at_a_terminal_one_file_still_asks(tmp_path, ext_template):
    parents = _copy([INT], tmp_path / "opts")
    status, out, answered = _run_at_a_terminal(
        [sys.executable, str(MACE_CLI), "opt2d12", "--out-file", f"{INT}.out",
         "--d12-file", f"{INT}.d12", "--config-file", str(ext_template),
         "--output-dir", "sp"], parents)
    assert status == 0, out[-3000:]
    assert out.count("Apply these settings from config file? [Y/n]") == 1
    assert answered == 1
