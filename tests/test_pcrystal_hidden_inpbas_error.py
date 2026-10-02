"""A Pcrystal run hides "UNIT CELL NOT NEUTRAL": its .out ends in MPI_Abort only.

Real CRYSTAL23 outputs (HPCC, 2026-10-01; tests/data/charged_runs/): a
periodic box with an explicit basis and a non-neutral CHEMOD but no CHARGED.
The serial run prints

    ERROR **** INPBAS **** UNIT CELL NOT NEUTRAL

and the same deck under Pcrystal (8 ranks) prints only

    Abort(1) on node 5 (rank 5 in comm 0): application called MPI_Abort(...)

Neither may ever count as a finished calculation, wherever MACE classifies an
output, and the queue manager's failure message must point at the hidden
error rather than say "unknown".
"""
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "data" / "charged_runs"
PCRYSTAL = DATA / "box_not_neutral_pcrystal.out"
SERIAL = DATA / "box_not_neutral_serial.out"


def test_the_fixtures_are_what_the_docstring_says():
    par, ser = PCRYSTAL.read_text(), SERIAL.read_text()
    assert "UNIT CELL NOT NEUTRAL" not in par and "MPI_Abort" in par
    assert "ERROR **** INPBAS **** UNIT CELL NOT NEUTRAL" in ser
    for text in (par, ser):
        assert "TOTAL CPU TIME" not in text and "OPT END" not in text


@pytest.mark.parametrize("out", [PCRYSTAL, SERIAL], ids=["pcrystal", "serial"])
def test_completion_checker_does_not_call_it_complete(out):
    from mace.completion_checker import categorize_output_file
    category, _ = categorize_output_file(out)
    assert not category.startswith("complete")
    assert category not in ("ongoing", "unknown")
    assert category == "potential"


@pytest.mark.parametrize("out", [PCRYSTAL, SERIAL], ids=["pcrystal", "serial"])
def test_error_detector_reports_an_error(out, tmp_path):
    from mace.recovery.detector import CrystalErrorDetector
    det = CrystalErrorDetector(base_dir=str(tmp_path), enable_tracking=False)
    result = det.analyze_output_file(out)
    assert result["status"] == "error", result
    assert result["completion_type"] is None


@pytest.mark.parametrize("out", [PCRYSTAL, SERIAL], ids=["pcrystal", "serial"])
def test_preflight_never_passes_it(out):
    from mace.submission.preflight import FAIL, classify_run
    status, reason, _ = classify_run(out.read_text(), "", 1)
    assert status == FAIL
    if out is SERIAL:
        assert "UNIT CELL NOT NEUTRAL" in reason


@pytest.fixture
def mgr(tmp_path):
    from mace.queue.manager import EnhancedCrystalQueueManager
    return EnhancedCrystalQueueManager(
        d12_dir=str(tmp_path), db_path=str(tmp_path / "materials.db"),
        enable_tracking=True, organize_outputs=False)


def test_queue_manager_sorts_the_pcrystal_output_as_failed(mgr, tmp_path, monkeypatch):
    """check_completed_or_failed_job: an MPI_Abort-only output is a failure."""
    work = tmp_path / "job"
    work.mkdir()
    (work / "box.out").write_text(PCRYSTAL.read_text())
    seen = {}
    monkeypatch.setattr(mgr.db, "update_calculation_status",
                        lambda calc_id, status, **kw: seen.setdefault("status", status))
    monkeypatch.setattr(mgr, "handle_completed_calculation", lambda *a: seen.setdefault("handler", "completed"))
    monkeypatch.setattr(mgr, "handle_failed_calculation", lambda *a: seen.setdefault("handler", "failed"))
    mgr.check_completed_or_failed_job({"calc_id": "c1", "work_dir": str(work),
                                       "input_file": str(work / "box.d12")})
    assert seen == {"status": "failed", "handler": "failed"}


def test_queue_manager_message_points_at_the_hidden_error(mgr):
    error_type, message = mgr.analyze_calculation_error({"output_file": str(PCRYSTAL)})
    assert error_type == "unknown_error"  # recovery routing unchanged
    assert "MPI_Abort" in message and "serial" in message
    assert "UNIT CELL NOT NEUTRAL" in message


def test_queue_manager_message_quotes_the_serial_error(mgr):
    error_type, message = mgr.analyze_calculation_error({"output_file": str(SERIAL)})
    assert error_type == "unknown_error"
    assert message == "CRYSTAL: ERROR **** INPBAS **** UNIT CELL NOT NEUTRAL"
