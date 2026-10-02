"""`opt2d3 --batch --config-file X` without --calc-type takes the config's type.

The batch path asked "Select type (1-6)" before loading the config, so with
nothing on stdin the run hit that menu (EOFError) and the config's
calculation_type was never used. It now reads the type from the config and
asks nothing; without a type anywhere and no answer it stops cleanly.
"""
import json
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, TEST_DATA

MACE_CLI = REPO_ROOT / "mace_cli"
PARENT = TEST_DATA / "OPT" / "1_dia_opt_rev1.out"
CONFIG = REPO_ROOT / "Crystal_d3" / "example_configs" / "doss_total_only.json"


def _batch(tmp_path, config):
    if not PARENT.exists():
        pytest.skip("test/ corpus not present (gitignored)")
    for name in ("p1", "p2"):
        shutil.copy(PARENT, tmp_path / f"{name}.out")
        (tmp_path / f"{name}.f9").write_bytes(b"x")
    return subprocess.run([sys.executable, str(MACE_CLI), "opt2d3", "--batch",
                           "--config-file", str(config)], cwd=tmp_path,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          timeout=300)


def test_batch_takes_the_calc_type_from_the_config(tmp_path):
    r = _batch(tmp_path, CONFIG)
    log = r.stdout + r.stderr
    assert "Select type (1-6)" not in log, log[-3000:]
    assert "Traceback" not in log, log[-3000:]
    assert r.returncode == 0, log[-3000:]
    for name in ("p1", "p2"):
        assert "DOSS" in (tmp_path / f"{name}_doss.d3").read_text().split()


def test_batch_without_any_calc_type_stops_cleanly(tmp_path):
    data = json.loads(CONFIG.read_text())
    data.pop("calculation_type")
    data["configuration"].pop("calculation_type")
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(data))
    r = _batch(tmp_path, cfg)
    log = r.stdout + r.stderr
    assert "Traceback" not in log and "EOFError" not in log, log[-3000:]
    assert r.returncode == 1, log[-3000:]
    assert "--calc-type" in log
