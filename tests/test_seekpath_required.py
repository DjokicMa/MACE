"""seekpath is required for a 3D cell's SeeK-path band path.

Without the seekpath package a 3D SeeK-path BAND path (opt2d3 format 4, a
plan's or a saved configuration's ``seekpath_full``) and an opt2d12 SeeK-path
phonon dispersion fell back, silently apart from a note, to the static HPKOT
tables. Those are SeeK-path's numbers in SeeK-path's standard cell, which for
many lattices is not CRYSTAL's cell, so the deck could name and visit other
points. Now that deck stops with the install instructions, unless the static
path is asked for explicitly (``--static-kpath`` or MACE_STATIC_KPATH=1).

Not affected: a slab or a polymer (seekpath is never used for them, so no
install message either), and the label, k-point vector, literature and
custom paths, which do not use seekpath.

Every route runs as a subprocess (opt2d3 / opt2d12 through mace_cli), with
seekpath blocked by a ``seekpath`` package on PYTHONPATH that raises
ImportError, as an environment without it does.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

pytest.importorskip("numpy")

MACE_CLI = REPO_ROOT / "mace_cli"
DATA = Path(__file__).parent / "data"
PTO = DATA / "opt_restart" / "tqb_pto.out"            # 3D, Pm-3m
SLAB = DATA / "low_dim_groups" / "graphene_lg80.out"  # slab, p6/mmm
POLYMER = DATA / "low_dim_groups" / "polyyne_rg28.out"

INSTALL = "pip install seekpath"
MODULE_HINT = "module load Python/3.11.3-GCCcore-12.3.0"


@pytest.fixture
def blocked(tmp_path):
    """A PYTHONPATH entry whose seekpath package fails to import."""
    shim = tmp_path / "no_seekpath"
    (shim / "seekpath").mkdir(parents=True)
    (shim / "seekpath" / "__init__.py").write_text(
        "raise ImportError(\"No module named 'seekpath'\")\n")
    return shim


def _env(shim=None, static=False):
    env = dict(os.environ)
    for k in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH",
              "MACE_STATIC_KPATH"):
        env.pop(k, None)
    if shim is not None:
        env["PYTHONPATH"] = str(shim)
    if static:
        env["MACE_STATIC_KPATH"] = "1"
    return env


def _stage(tmp_path, src, name="parent"):
    d = tmp_path / name
    d.mkdir()
    shutil.copy(src, d / src.name)
    if src.with_suffix(".d12").exists():
        shutil.copy(src.with_suffix(".d12"), d / src.with_suffix(".d12").name)
    (d / src.with_suffix(".f9").name).write_text("wavefunction placeholder\n")
    return d / src.name


def _opt2d3(parent, out_dir, answers=None, config=None, extra=(), env=None):
    cmd = [sys.executable, str(MACE_CLI), "opt2d3", "--input", str(parent),
           "--calc-type", "BAND", "--output-dir", str(out_dir), *extra]
    if config is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        cfg = Path(out_dir).parent / (Path(out_dir).name + "_config.json")
        cfg.write_text(json.dumps({"version": "1.0", "type": "d3_configuration",
                                   "calculation_type": "BAND", "configuration": config}))
        cmd += ["--config-file", str(cfg)]
    r = subprocess.run(cmd, input="\n".join(answers or []) + "\n" * 10,
                       capture_output=True, text=True, env=env, timeout=180)
    decks = sorted(Path(out_dir).glob("*_band.d3")) if Path(out_dir).exists() else []
    return r, decks


SEEKPATH_CONFIG = {"calculation_type": "BAND", "path": "auto", "bands": "auto",
                   "shrink": "auto", "labels": "none", "auto_path": True,
                   "n_points": 1000, "path_method": "coordinates", "seekpath_full": True}

# opt2d3's answers: automatic path (1), format, shrink (Enter), all bands, no print
FORMAT = {"labels": ["1", "1", "", "1", "n"], "vectors": ["1", "2", "", "1", "n"],
          "literature": ["1", "3", "", "1", "n"], "seekpath": ["1", "4", "", "1", "n"]}


# ------------------------------------------------------------------ opt2d3

def test_3d_seekpath_band_without_seekpath_stops_with_the_install_message(tmp_path, blocked):
    r, decks = _opt2d3(_stage(tmp_path, PTO), tmp_path / "out", FORMAT["seekpath"],
                       env=_env(blocked))
    text = r.stdout + r.stderr
    assert r.returncode != 0, text[-2000:]
    assert decks == []
    assert INSTALL in text and MODULE_HINT in text and "--static-kpath" in text
    assert "Traceback" not in text


def test_seekpath_older_than_2_1_counts_as_missing(tmp_path):
    """seekpath 2.0.x returns no 'rotation_matrix': the library route failed
    with KeyError and fell back to the static path with a warning."""
    shim = tmp_path / "old_seekpath"
    (shim / "seekpath").mkdir(parents=True)
    (shim / "seekpath" / "__init__.py").write_text("__version__ = '2.0.1'\n")
    r, decks = _opt2d3(_stage(tmp_path, PTO), tmp_path / "out", FORMAT["seekpath"],
                       env=_env(shim))
    text = r.stdout + r.stderr
    assert r.returncode != 0 and decks == [], text[-2000:]
    assert "2.0.1" in text and "pip install --upgrade seekpath" in text


def test_3d_seekpath_config_without_seekpath_stops(tmp_path, blocked):
    r, decks = _opt2d3(_stage(tmp_path, PTO), tmp_path / "out", config=dict(SEEKPATH_CONFIG),
                       env=_env(blocked))
    text = r.stdout + r.stderr
    assert r.returncode != 0 and decks == [], text[-2000:]
    assert INSTALL in text and "Traceback" not in text


def test_static_kpath_flag_writes_the_static_deck(tmp_path, blocked):
    parent = _stage(tmp_path, PTO)
    r, decks = _opt2d3(parent, tmp_path / "flag", FORMAT["seekpath"],
                       extra=["--static-kpath"], env=_env(blocked))
    assert r.returncode == 0 and len(decks) == 1, (r.stdout + r.stderr)[-2000:]
    deck = decks[0].read_text().splitlines()
    assert deck[0] == "BAND" and deck[-1] == "END"
    # The static cP2 entry (SeeK-path's GAMMA-X-M-GAMMA-R-X|R-M)
    assert "GAMMA-X-M-GAMMA-R-X-|-R-M" in deck[1], deck[1]
    assert "static" in r.stdout.lower()
    # the environment variable is the same opt-in
    r2, decks2 = _opt2d3(parent, tmp_path / "env", FORMAT["seekpath"],
                         env=_env(blocked, static=True))
    assert r2.returncode == 0 and len(decks2) == 1, (r2.stdout + r2.stderr)[-2000:]
    assert decks2[0].read_text() == decks[0].read_text()


def test_static_kpath_flag_with_a_config_file(tmp_path, blocked):
    r, decks = _opt2d3(_stage(tmp_path, PTO), tmp_path / "out", config=dict(SEEKPATH_CONFIG),
                       extra=["--static-kpath"], env=_env(blocked))
    assert r.returncode == 0 and len(decks) == 1, (r.stdout + r.stderr)[-2000:]


@pytest.mark.parametrize("parent", [SLAB, POLYMER], ids=["slab", "polymer"])
def test_slab_and_polymer_seekpath_band_need_no_seekpath(tmp_path, blocked, parent):
    r, decks = _opt2d3(_stage(tmp_path, parent), tmp_path / "out", FORMAT["seekpath"],
                       env=_env(blocked))
    text = r.stdout + r.stderr
    assert r.returncode == 0 and len(decks) == 1, text[-2000:]
    assert INSTALL not in text and "Install" not in text and "--static-kpath" not in text


@pytest.mark.parametrize("fmt", ["labels", "vectors", "literature"])
def test_paths_that_do_not_use_seekpath_are_unchanged(tmp_path, blocked, fmt):
    parent = _stage(tmp_path, PTO)
    r, decks = _opt2d3(parent, tmp_path / "blocked", FORMAT[fmt], env=_env(blocked))
    text = r.stdout + r.stderr
    assert r.returncode == 0 and len(decks) == 1, text[-2000:]
    assert INSTALL not in text and "--static-kpath" not in text
    r2, decks2 = _opt2d3(parent, tmp_path / "free", FORMAT[fmt], env=_env())
    assert r2.returncode == 0 and decks2[0].read_text() == decks[0].read_text()


@pytest.mark.parametrize("answers", [
    ["3", "GAMMA X M GAMMA"],                                       # custom labels
    ["4", "y", "2", "C", "0 0 0 8 0 0", "C", "8 0 0 8 8 0"],         # k-point vectors
], ids=["custom-labels", "custom-vectors"])
def test_custom_paths_need_no_seekpath(tmp_path, blocked, answers):
    r, decks = _opt2d3(_stage(tmp_path, PTO), tmp_path / "out", answers, env=_env(blocked))
    text = r.stdout + r.stderr
    assert r.returncode == 0 and len(decks) == 1, text[-2000:]
    assert INSTALL not in text and "--static-kpath" not in text


def test_with_seekpath_the_library_path_is_written(tmp_path):
    pytest.importorskip("seekpath")
    r, decks = _opt2d3(_stage(tmp_path, PTO), tmp_path / "out", FORMAT["seekpath"],
                       env=_env())
    assert r.returncode == 0 and len(decks) == 1, (r.stdout + r.stderr)[-2000:]
    assert "Using seekpath library" in r.stdout
    assert INSTALL not in r.stdout + r.stderr


# ----------------------------------------------------------------- opt2d12

def _dispersion_config(**bands):
    return {"calculation_type": "FREQ",
            "freq_settings": {"dispersion": True, "scelphono": [2, 2, 2],
                              "bands": dict({"path_method": "coordinates", "path": "auto",
                                             "shrink": 16, "npoints": 100}, **bands)}}


def _opt2d12(tmp_path, config, extra=(), env=None):
    parent = _stage(tmp_path, PTO, "d12")
    cfg = tmp_path / "freq.json"
    cfg.write_text(json.dumps(config))
    r = subprocess.run([sys.executable, str(MACE_CLI), "opt2d12", "--out-file", parent.name,
                        "--config-file", str(cfg), "--yes", *extra],
                       cwd=parent.parent, input="", capture_output=True, text=True,
                       env=env, timeout=300)
    kids = [p for p in parent.parent.glob("*.d12") if p.name != parent.with_suffix(".d12").name]
    return r, kids


SEEKPATH_BANDS = {"seekpath_full": True, "format": "seekpath"}


def test_opt2d12_seekpath_dispersion_without_seekpath_stops(tmp_path, blocked):
    r, kids = _opt2d12(tmp_path, _dispersion_config(**SEEKPATH_BANDS), env=_env(blocked))
    text = r.stdout + r.stderr
    assert r.returncode != 0 and kids == [], text[-2000:]
    assert INSTALL in text and MODULE_HINT in text and "--static-kpath" in text
    assert "Traceback" not in text


def test_opt2d12_static_kpath_writes_the_static_dispersion(tmp_path, blocked):
    r, kids = _opt2d12(tmp_path, _dispersion_config(**SEEKPATH_BANDS),
                       extra=["--static-kpath"], env=_env(blocked))
    assert r.returncode == 0 and len(kids) == 1, (r.stdout + r.stderr)[-2000:]
    deck = kids[0].read_text()
    assert "BANDS" in deck and "SCELPHONO" in deck


def test_opt2d12_literature_dispersion_needs_no_seekpath(tmp_path, blocked):
    r, kids = _opt2d12(tmp_path, _dispersion_config(literature_path=True, format="literature"),
                       env=_env(blocked))
    text = r.stdout + r.stderr
    assert r.returncode == 0 and len(kids) == 1, text[-2000:]
    assert INSTALL not in text


# ------------------------------------------------------------------- setup

def test_setup_dependency_check_requires_seekpath(blocked):
    code = ("import sys; sys.path.insert(0, %r); import setup_mace; "
            "ok = setup_mace.MACESetup.check_dependencies(object.__new__(setup_mace.MACESetup)); "
            "print('RESULT', ok)" % str(REPO_ROOT))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=_env(blocked), timeout=120)
    assert "seekpath (missing)" in r.stdout, r.stdout + r.stderr
    assert "RESULT False" in r.stdout
