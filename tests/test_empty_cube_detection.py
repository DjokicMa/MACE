"""An ECH3/POT3 run that ended normally with an empty cube is not complete.

Measured on HPCC: the stock properties ran ECH3 on a two-component molecule's
fort.9 with exit code 0 and wrote a 0-byte DENS_CUBE.DAT, which the job
script copied back as <job>_DENS.CUBE. The .out here is a real corpus
ECH3+POT3 run (EC molecule).
"""
import shutil
from pathlib import Path

import pytest

from mace.utils import calc_detection as cd

DATA = Path(__file__).resolve().parent / "data" / "charge_potential"
STEM = "ec_charge+potential"


def _run(tmp_path, sizes):
    d = tmp_path / "run"
    d.mkdir(parents=True, exist_ok=True)
    for ext in (".out", ".d3"):
        shutil.copy(DATA / f"{STEM}{ext}", d / f"{STEM}{ext}")
    for suffix, size in sizes.items():
        (d / f"{STEM}{suffix}").write_text("x" * size)
    return d / f"{STEM}.out"


GOOD = {"_DENS.CUBE": 100, "_POT.CUBE": 100, "_SPIN.CUBE": 100}


def test_complete_with_non_empty_cubes(tmp_path):
    from mace.completion_checker import categorize_output_file
    out = _run(tmp_path, GOOD)
    assert cd.empty_cube_reason(out) is None
    assert categorize_output_file(out) == ("completecharge_potential", STEM)


@pytest.mark.parametrize("empty", ["_DENS.CUBE", "_POT.CUBE", "_SPIN.CUBE"])
def test_an_empty_cube_is_a_failure(tmp_path, empty):
    from mace.completion_checker import ERROR_DESCRIPTIONS, categorize_output_file
    out = _run(tmp_path, dict(GOOD, **{empty: 0}))
    why = cd.empty_cube_reason(out)
    assert f"{STEM}{empty} is 0 bytes" in why
    assert categorize_output_file(out) == ("cube_empty", STEM)
    assert "empty" in ERROR_DESCRIPTIONS["cube_empty"]


def test_recovery_detector_and_queue_manager_flag_it(tmp_path):
    from mace.queue.manager import EnhancedCrystalQueueManager as QM
    from mace.recovery.detector import CrystalErrorDetector
    det = CrystalErrorDetector(base_dir=str(tmp_path), enable_tracking=False)
    res = det.analyze_output_file(_run(tmp_path / "a", GOOD))
    assert (res["status"], res["calc_type"]) == ("completed", "CHARGE+POTENTIAL")
    bad = _run(tmp_path / "b", dict(GOOD, **{"_DENS.CUBE": 0}))
    res = det.analyze_output_file(bad)
    assert (res["status"], res["error_type"], res["recoverable"]) == ("error", "cube_empty", False)
    qm = QM.__new__(QM)
    error_type, message = QM.analyze_calculation_error(qm, {"output_file": str(bad), "calc_id": "x"})
    assert error_type == "cube_empty" and "0 bytes" in message


def test_gui_batch_knows_the_bucket():
    from mace.gui import batch
    assert "cube_empty" in batch.ERROR_BUCKETS and "cube_empty" in batch.FAILED_ROUTE
