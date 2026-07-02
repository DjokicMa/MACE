"""End-to-end tests that drive the REAL `mace_cli` binary as a subprocess.

These exist because the unit/handler tests import the plotting code in isolation,
which missed a real-world crash: under the full `mace` CLI the cube engine's
`import plotly.express` pulled xarray->pandas->pyarrow and double-registered a
pyarrow extension type (ArrowKeyError) — a failure that only appears when the
whole CLI import stack is loaded the way a user runs it.

So: invoke `python mace_cli plotting ...` exactly as a user would, with the full
import stack, and assert it exits cleanly and produces output. Slower than the
unit tests, but they catch integration/import-order bugs the unit tests cannot.

All tests use real corpus inputs via find_data and skip if the gitignored
test/ corpus is absent.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import find_data

REPO = Path(__file__).resolve().parent.parent
MACE_CLI = REPO / "mace_cli"

pytestmark = pytest.mark.skipif(not MACE_CLI.is_file(), reason="mace_cli not found")


def _run_plotting(args, timeout=300):
    """Run `python mace_cli plotting <args>` from the repo root, as a user would."""
    proc = subprocess.run(
        [sys.executable, str(MACE_CLI), "plotting", *args],
        cwd=str(REPO), capture_output=True, text=True, timeout=timeout,
    )
    return proc


def _assert_clean(proc):
    combined = proc.stdout + proc.stderr
    # The bug we are guarding against printed a Traceback + ArrowKeyError.
    assert "Traceback (most recent call last)" not in combined, combined[-2000:]
    assert "ArrowKeyError" not in combined, combined[-2000:]
    assert proc.returncode == 0, f"exit {proc.returncode}\n{combined[-2000:]}"


def test_mace_cli_cube_renders(tmp_path):
    """The exact path that crashed: a cube isosurface through the full CLI."""
    cube = find_data("ECH3POT3/*DENS*.CUBE")
    src = tmp_path / "in"
    src.mkdir()
    local = src / "sample_DENS.CUBE"
    shutil.copy(cube, local)
    out = tmp_path / "out"
    proc = _run_plotting(["--cube", str(local), "--iso", "0.01", "--no-atoms",
                          "-o", str(out)])
    _assert_clean(proc)
    assert list(out.glob("*.html")), "no cube HTML produced"


def test_mace_cli_cube_interactive_session(tmp_path):
    """Replicates the reported interactive session verbatim (menu -> isosurface).

    A one-cube directory yields a 2-option menu; the stdin drives:
    pick cube (1), view isosurface (1), then accept defaults through to exit.
    """
    cube = find_data("ECH3POT3/*DENS*.CUBE")
    workdir = tmp_path / "work"
    workdir.mkdir()
    shutil.copy(cube, workdir / "sample_DENS.CUBE")
    proc = subprocess.run(
        [sys.executable, str(MACE_CLI), "plotting", "-d", str(workdir)],
        cwd=str(REPO), capture_output=True, text=True, timeout=300,
        input="1\n1\n\n\n\n\n\n\n\n\n",   # menu=cube, view=iso, then defaults -> exit
    )
    _assert_clean(proc)
    assert list(workdir.glob("*.html")), "no HTML produced by interactive run"


def test_mace_cli_freq_renders(tmp_path):
    """FREQ vibrational mode render through the full CLI."""
    out = tmp_path / "out"
    f = find_data("FREQ/*MOLECULE*.out", "NORMAL MODES NORMALIZED")
    proc = _run_plotting(["--freq", str(f), "--frames", "3", "-o", str(out)])
    _assert_clean(proc)
    assert list(out.rglob("*.html")), "no FREQ HTML produced"


def test_mace_cli_ir_spectrum_renders(tmp_path):
    """IR spectrum render through the full CLI (matplotlib path)."""
    out = tmp_path / "out"
    f = find_data("FREQ/*IRSPEC.DAT")
    proc = _run_plotting(["--ir", str(f), "-o", str(out)])
    _assert_clean(proc)
    assert list(out.glob("*.png")), "no IR png produced"


def test_mace_cli_raman_spectrum_renders(tmp_path):
    """Raman spectrum render through the full CLI (matplotlib path)."""
    out = tmp_path / "out"
    f = find_data("FREQ/*RAMSPEC.DAT")
    proc = _run_plotting(["--raman", "--raman-mode", "all", str(f), "-o", str(out)])
    _assert_clean(proc)
    assert list(out.glob("*.png")), "no Raman png produced"


def test_mace_cli_plotting_help_is_clean():
    """`mace plotting --help` must render without importing/among-crashing."""
    proc = _run_plotting(["--help"], timeout=120)
    assert proc.returncode == 0
    assert "Traceback (most recent call last)" not in (proc.stdout + proc.stderr)
    assert "--cube" in proc.stdout and "--freq" in proc.stdout and "--ir" in proc.stdout


def test_missing_spectra_file_is_clean_error(tmp_path):
    """A nonexistent IRSPEC path must produce a clean per-file error, not the
    raw FileNotFoundError traceback that spectra (alone among handler kinds)
    used to leak."""
    proc = _run_plotting(["--ir", str(tmp_path / "definitely_missing_IRSPEC.DAT"),
                          "-o", str(tmp_path)], timeout=120)
    combined = proc.stdout + proc.stderr
    assert "Traceback (most recent call last)" not in combined, combined[-2000:]
    assert "Error reading" in combined


def test_no_files_exit_code_propagates(tmp_path):
    """`mace plotting --band` over a dir with no band data returns exit 1
    (plotting_main's status used to be discarded, so scripts always saw 0)."""
    proc = _run_plotting(["--band", "-d", str(tmp_path), "-o", str(tmp_path)],
                         timeout=120)
    assert proc.returncode == 1, (proc.stdout + proc.stderr)[-1000:]


def test_bad_iso_value_is_usage_error(tmp_path):
    """Malformed --iso must be an argparse-style usage error (exit 2, clean
    message), not a ValueError traceback."""
    proc = _run_plotting(["--cube", "--iso", "abc",
                          str(tmp_path / "x.CUBE"), "-o", str(tmp_path)],
                         timeout=120)
    combined = proc.stdout + proc.stderr
    assert "Traceback (most recent call last)" not in combined, combined[-2000:]
    assert "--iso expects comma-separated numbers" in combined
    assert proc.returncode == 2
