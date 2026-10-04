"""A seekpath that fails for a 3D cell stops the deck, as a missing one does.

With seekpath installed, a library error for a 3D cell (opt2d3 format 4, a
saved ``seekpath_full`` configuration, an opt2d12 SeeK-path phonon
dispersion) printed "seekpath library failed (...), falling back to static
data" and wrote the static HPKOT path - SeeK-path's standard cell, which for
many lattices is not CRYSTAL's - the silent fallback that a missing seekpath
no longer gets. That deck now stops with the reason, in the style of the
missing-seekpath stop, unless the static path was asked for (--static-kpath
or MACE_STATIC_KPATH=1). A polymer's path, which does not need seekpath,
still falls back.

The failing seekpath is a ``seekpath`` package on PYTHONPATH (version 2.1.0)
whose get_path raises.
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
POLYMER = DATA / "low_dim_groups" / "polyyne_rg28.out"
REASON = "synthetic seekpath failure"
SEEKPATH_ANSWERS = ["1", "4", "", "1", "n"]   # automatic path, SeeK-path, Enter, all bands, no print


@pytest.fixture
def failing(tmp_path):
    shim = tmp_path / "failing_seekpath"
    (shim / "seekpath").mkdir(parents=True)
    (shim / "seekpath" / "__init__.py").write_text(
        "__version__ = '2.1.0'\n"
        "def get_path(*args, **kwargs):\n"
        f"    raise ValueError({REASON!r})\n"
        "def get_explicit_k_path(*args, **kwargs):\n"
        f"    raise ValueError({REASON!r})\n")
    return shim


def _env(shim, static=False):
    env = {k: v for k, v in os.environ.items()
           if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH",
                        "MACE_STATIC_KPATH")}
    env["PYTHONPATH"] = str(shim)
    env["MACE_NO_BANNER"] = "1"
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


def _opt2d3(parent, out_dir, env, answers=SEEKPATH_ANSWERS, config=None, extra=()):
    cmd = [sys.executable, str(MACE_CLI), "opt2d3", "--input", str(parent),
           "--calc-type", "BAND", "--output-dir", str(out_dir), *extra]
    if config is not None:
        cfg = Path(out_dir).parent / (Path(out_dir).name + "_config.json")
        cfg.write_text(json.dumps({"version": "1.0", "type": "d3_configuration",
                                   "calculation_type": "BAND", "configuration": config}))
        cmd += ["--config-file", str(cfg)]
    r = subprocess.run(cmd, input="\n".join(answers) + "\n" * 10, capture_output=True,
                       text=True, env=env, timeout=180)
    decks = sorted(Path(out_dir).glob("*_band.d3")) if Path(out_dir).exists() else []
    return r, r.stdout + r.stderr, decks


def _assert_stopped(r, text, written):
    assert r.returncode != 0, text[-2000:]
    assert written == [], text[-2000:]
    assert REASON in text, text[-2000:]
    assert "--static-kpath" in text and "MACE_STATIC_KPATH=1" in text, text[-2000:]
    assert "No deck was written" in text, text[-2000:]
    assert "falling back to static data" not in text, text[-2000:]
    assert "Traceback" not in text, text[-2000:]


def test_opt2d3_seekpath_band_stops_when_the_library_fails(tmp_path, failing):
    r, text, decks = _opt2d3(_stage(tmp_path, PTO), tmp_path / "out", _env(failing))
    _assert_stopped(r, text, decks)


def test_opt2d3_seekpath_config_stops_when_the_library_fails(tmp_path, failing):
    config = {"calculation_type": "BAND", "path": "auto", "bands": "auto", "shrink": "auto",
              "labels": "none", "auto_path": True, "n_points": 1000,
              "path_method": "coordinates", "seekpath_full": True}
    r, text, decks = _opt2d3(_stage(tmp_path, PTO), tmp_path / "out", _env(failing),
                             answers=[], config=config)
    _assert_stopped(r, text, decks)


@pytest.mark.parametrize("how", ["flag", "env"])
def test_opt2d3_static_kpath_opt_in_writes_the_static_path(tmp_path, failing, how):
    extra = ["--static-kpath"] if how == "flag" else []
    r, text, decks = _opt2d3(_stage(tmp_path, PTO), tmp_path / "out",
                             _env(failing, static=(how == "env")), extra=extra)
    assert r.returncode == 0 and len(decks) == 1, text[-2000:]
    assert "GAMMA-X-M-GAMMA-R-X-|-R-M" in decks[0].read_text().splitlines()[1]
    assert REASON in text   # the warning still says why


def test_opt2d3_polymer_still_falls_back(tmp_path, failing):
    r, text, decks = _opt2d3(_stage(tmp_path, POLYMER), tmp_path / "out", _env(failing))
    assert r.returncode == 0 and len(decks) == 1, text[-2000:]


def _opt2d12(tmp_path, env, extra=()):
    parent = _stage(tmp_path, PTO, "d12")
    cfg = tmp_path / "freq.json"
    cfg.write_text(json.dumps({
        "calculation_type": "FREQ",
        "freq_settings": {"dispersion": True, "scelphono": [2, 2, 2],
                          "bands": {"path_method": "coordinates", "path": "auto",
                                    "shrink": 16, "npoints": 100,
                                    "seekpath_full": True, "format": "seekpath"}}}))
    r = subprocess.run([sys.executable, str(MACE_CLI), "opt2d12", "--out-file", parent.name,
                        "--config-file", str(cfg), "--yes", *extra],
                       cwd=parent.parent, input="", capture_output=True, text=True,
                       env=env, timeout=300)
    kids = [p for p in parent.parent.glob("*.d12") if p.name != parent.with_suffix(".d12").name]
    return r, r.stdout + r.stderr, kids


def test_opt2d12_seekpath_dispersion_stops_when_the_library_fails(tmp_path, failing):
    r, text, kids = _opt2d12(tmp_path, _env(failing))
    _assert_stopped(r, text, kids)


def test_opt2d12_static_kpath_writes_the_dispersion(tmp_path, failing):
    r, text, kids = _opt2d12(tmp_path, _env(failing), extra=["--static-kpath"])
    assert r.returncode == 0 and len(kids) == 1, text[-2000:]
    assert "SCELPHONO" in kids[0].read_text()
