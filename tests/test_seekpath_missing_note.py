"""Importing the k-path modules without seekpath prints nothing.

"Note: seekpath library not installed. Install with: pip install seekpath"
was printed at import, for every opt2d3 run (a slab, which never uses the
library, included). The static route still names the library where it would
be used.
"""
import subprocess
import sys

from conftest import REPO_ROOT

BLOCK = ("import sys; sys.modules['seekpath'] = None; "
         f"sys.path[:0] = [{str(REPO_ROOT / 'Crystal_d3')!r}, {str(REPO_ROOT / 'Crystal_d12')!r}]; ")


def _run(code):
    return subprocess.run([sys.executable, "-c", BLOCK + code],
                          capture_output=True, text=True, timeout=120)


def test_import_without_seekpath_is_silent():
    r = _run("import seekpath_interface, d3_kpoints; print('imported')")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "imported"
    assert "seekpath library not installed" not in r.stdout + r.stderr


def test_static_route_still_names_the_library():
    r = _run("import d3_kpoints; d3_kpoints.get_seekpath_full_kpath(63, 'C')")
    assert r.returncode == 0, r.stderr
    assert "Install the 'seekpath' library" in r.stdout
