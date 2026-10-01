"""The bundled lcao2wannier's optional pieces degrade cleanly when absent.

Two compiled Fortran kernels (``_spread_fortran``, ``_disentangle_fortran``)
are optional speed-ups that are never committed, and matplotlib is only used
by the band plot. Without them the package must import, report the kernels as
absent, and fall back (pure Python; a text summary instead of a plot) rather
than raise. Each check runs in a fresh interpreter with the piece blocked, so
a locally built kernel or an installed matplotlib cannot mask the fallback.
"""
import subprocess
import sys

from conftest import REPO_ROOT

PKG = "mace.wannier.lcao2wannier"


def _run(code):
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def test_the_package_runs_without_the_fortran_kernels():
    out = _run(
        "import sys\n"
        f"sys.modules['{PKG}._spread_fortran'] = None\n"
        f"sys.modules['{PKG}._disentangle_fortran'] = None\n"
        f"from {PKG} import spread, hybrid\n"
        "print(spread.have_fortran(), hybrid.have_disentangle_fortran())\n")
    assert out.split() == ["False", "False"]


def test_the_band_plot_falls_back_without_matplotlib():
    out = _run(
        "import sys\n"
        "sys.modules['matplotlib'] = None\n"
        f"from {PKG}.band_plot import plot_band_structure\n"
        "print(plot_band_structure(None, 'unused.png'))\n")
    assert out.strip() == "False"


def test_compiled_kernels_are_never_committed():
    """VENDORED.md: compiled *.so files must not be committed."""
    import pytest

    inside = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                            cwd=str(REPO_ROOT), capture_output=True, text=True)
    if inside.returncode != 0:
        pytest.skip("not a git checkout")
    for name in ("_spread_fortran.cpython-311-x86_64-linux-gnu.so",
                 "_disentangle_fortran.cpython-312-darwin.so"):
        proc = subprocess.run(
            ["git", "check-ignore", "-q", f"mace/wannier/lcao2wannier/{name}"],
            cwd=str(REPO_ROOT), capture_output=True, text=True)
        assert proc.returncode == 0, f"{name} is not ignored"
