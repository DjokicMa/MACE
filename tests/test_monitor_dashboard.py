"""Regression tests for ``MaterialMonitor._dashboard_rows`` (ui status-dashboard rows).

Covers the defensive ``.get()`` hardening: a sparse/partial status dict must not
``KeyError``, and the normal full-status path must still produce all five subsystem
rows plus the overall verdict and subtitle. No database is opened — the instance is
built via ``__new__`` because ``_dashboard_rows`` and ``_state_for`` use no instance
state.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mace.queue.monitor import MaterialMonitor


def _monitor():
    # Bypass __init__ (which would open a DB / read config). _dashboard_rows and
    # _state_for are stateless, so a bare instance is sufficient and DB-free.
    return MaterialMonitor.__new__(MaterialMonitor)


def _full_status():
    return {
        "timestamp": "2026-06-21T12:00:00",
        "database": {"status": "healthy", "accessible": True, "size_mb": 12.3,
                     "stats": {"total_materials": 7,
                               "calculations_by_status": {"completed": 5}},
                     "issues": []},
        "queue": {"status": "healthy", "total_jobs": 3, "by_status": {"R": 2},
                  "issues": []},
        "files": {"status": "warning", "total_materials": 7, "total_files": 40,
                  "total_size_mb": 100.0, "organization_score": 88.0,
                  "issues": ["3 missing .out"]},
        "errors": {"status": "healthy", "recent_count": 0, "error_rate": 0.0,
                   "critical_errors": 0, "trending_up": [], "issues": []},
        "performance": {"status": "healthy", "success_rate": 99.0,
                        "avg_job_time": 1.2, "queue_throughput": 5.0, "issues": []},
    }


def test_dashboard_rows_full_status_happy_path():
    rows, overall, subtitle = _monitor()._dashboard_rows(_full_status())
    assert [r.subsystem for r in rows] == [
        "DATABASE", "QUEUE", "FILES", "ERRORS", "PERFORMANCE"]
    assert overall == "WARNING"                 # files is 'warning'
    assert "2026-06-21T12:00:00" in subtitle
    assert any("7 materials" in r.detail for r in rows)   # data preserved


@pytest.mark.parametrize("status", [
    {},                                   # completely empty
    {"timestamp": "t"},                   # only a timestamp
    {"database": {}},                     # subsystem present but empty
    {"database": {"status": "error"}},    # partial subsystem
])
def test_dashboard_rows_sparse_status_does_not_keyerror(status):
    """A sparse status dict (missing subsystems/keys) must not KeyError — every
    subsystem extraction and the overall/subtitle computation uses ``.get()``."""
    rows, overall, subtitle = _monitor()._dashboard_rows(status)
    assert len(rows) == 5                       # always five subsystem rows
    assert overall in ("HEALTHY", "WARNING", "CRITICAL")
    assert "Last Updated:" in subtitle


def test_dashboard_rows_error_status_is_critical():
    s = _full_status()
    s["errors"]["status"] = "error"
    _, overall, _ = _monitor()._dashboard_rows(s)
    assert overall == "CRITICAL"
