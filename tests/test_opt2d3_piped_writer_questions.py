"""opt2d3 runs answered through a pipe ask their remaining questions there too.

Two questions still required a terminal although the run's settings were
being answered on stdin (as 1.1.1 read them):

* no wavefunction next to the output: "Continue anyway?" was never asked,
  the run aborted;
* several outputs with shared settings: each file's own questions (the ECHG
  map plane, POTC points) were refused with "no terminal to ask", so no deck
  was written for any file.

Runs given their settings (--config-file) with no terminal still do not ask.
"""
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, TEST_DATA

MACE_CLI = REPO_ROOT / "mace_cli"
PARENT = TEST_DATA / "OPT" / "1_dia_opt_rev1.out"
PLANE = "F\n0 0 0\n1 0 0\n0 1 0\n"


@pytest.fixture(autouse=True)
def _corpus():
    if not PARENT.exists():
        pytest.skip("test/ corpus not present (gitignored)")


def _opt2d3(cwd, target, stdin):
    return subprocess.run([sys.executable, str(MACE_CLI), "opt2d3", "--input", target,
                           "--calc-type", "CHARGE"], cwd=cwd, input=stdin,
                          capture_output=True, text=True, timeout=300)


def test_missing_wavefunction_is_asked_on_the_pipe(tmp_path):
    shutil.copy(PARENT, tmp_path / "p.out")              # no p.f9
    r = _opt2d3(tmp_path, "p.out", "y\n2\n0\n50\ny\n" + PLANE + "\n" * 10)
    log = r.stdout + r.stderr
    assert "no terminal to confirm" not in log, log[-3000:]
    assert r.returncode == 0, log[-3000:]
    assert (tmp_path / "p_charge.d3").read_text().split()[0] == "ECHG"


def test_missing_wavefunction_no_still_writes_nothing(tmp_path):
    shutil.copy(PARENT, tmp_path / "p.out")
    r = _opt2d3(tmp_path, "p.out", "n\n" + "\n" * 10)
    assert not list(tmp_path.glob("*.d3")), r.stdout + r.stderr


def test_shared_settings_ask_each_files_plane_on_the_pipe(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    for name in ("p1", "p2"):
        shutil.copy(PARENT, d / f"{name}.out")
        (d / f"{name}.f9").write_bytes(b"x")
    # shared: yes; ECHG, order 0, 50 points, plane for the reference file;
    # don't save; then the plane for each file
    r = _opt2d3(tmp_path, "in", "y\n2\n0\n50\ny\n" + PLANE + "n\n" + PLANE + PLANE + "\n" * 5)
    log = r.stdout + r.stderr
    assert "no terminal to ask" not in log, log[-3000:]
    assert "Completed processing 2 of 2 files" in log, log[-3000:]
    for name in ("p1", "p2"):
        assert (d / f"{name}_charge.d3").read_text().split()[0] == "ECHG"


@pytest.mark.parametrize("unit", ["", "C"], ids=["fractional", "cartesian"])
def test_a_115_era_enter_for_the_plane_question_is_refused_cleanly(tmp_path, unit):
    """1.1.5 asked "Define charge density map plane? [Y/n]" before the unit;
    its Enter now answers the unit, and the old unit answer reached "Point A":
    a ValueError or IndexError traceback. A point that is not three numbers is
    refused, with no deck and no traceback."""
    shutil.copy(PARENT, tmp_path / "p.out")
    r = _opt2d3(tmp_path, "p.out", "y\n2\n0\n50\n\n" + unit + "\n0 0 0\n1 0 0\n0 1 0\n" + "\n" * 10)
    log = r.stdout + r.stderr
    assert "Traceback" not in log, log[-3000:]
    assert r.returncode != 0 and not list(tmp_path.glob("*.d3"))
    assert "Point A is three numbers" in log, log[-3000:]


def test_potential_type_takes_only_1_or_2(monkeypatch):
    """CHARGE+POTENTIAL: a 1.1.5-era "y" for the removed plane question landed
    on "Select type (1-2)" and chose POTC; it is refused."""
    monkeypatch.syspath_prepend(str(REPO_ROOT / "Crystal_d3"))
    import d3_interactive as D
    monkeypatch.setattr(D, "_nav_read", lambda *a, **k: "y")
    with pytest.raises(SystemExit):
        D.configure_potential_calculation()
