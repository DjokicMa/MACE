"""A manual submission must never turn itself into a workflow.

Real incident (2026-07-27, reported from normal use): a user ran
``mace opt2d12`` / ``mace opt2d3`` by hand, then ``mace submit``. Every job
script MACE writes ends with the queue-manager completion callback, and that
callback

  1. adopted every loose ``*.d12``/``*.d3`` in the tree it had no DB record for
     (``process_new_d12_files``) — including the deck the user had just
     generated but not yet submitted, and
  2. ran ``plan_next_calculation`` -> ``WorkflowEngine.execute_workflow_step``,
     whose no-plan ``else`` branches emit the built-in defaults
     (OPT -> SP, SP -> BAND + DOSS),

so a single hand-run calculation silently became a 4-step workflow inside a
synthesized ``workflow_outputs/workflow_<timestamp>/`` directory. The HPCC QA
tree still shows the fingerprint: six such directories under ``phase2_batch``.

Progression is a WORKFLOW feature. It requires a plan; without one MACE now
records the calculation and stops. ``mace submit --progress`` is the opt-in
(it writes a real plan first — see test_submit_progress_plan.py).
"""
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------
# Engine: the choke point. No plan -> no generation.
# --------------------------------------------------------------------------
def _engine_without_init(monkeypatch, calc, plan=None):
    """A WorkflowEngine with its DB/filesystem setup bypassed."""
    from mace.workflow import engine as engine_mod

    eng = engine_mod.WorkflowEngine.__new__(engine_mod.WorkflowEngine)
    eng.base_work_dir = Path(".")
    eng.allow_planless_progression = False

    class _DB:
        def get_calculation(self, calc_id):
            return calc

        def get_calculations_by_status(self, material_id=None, status=None):
            return [calc]

    eng.db = _DB()
    monkeypatch.setattr(eng, "_cleanup_failed_workflow_dirs", lambda: None, raising=False)
    monkeypatch.setattr(eng, "_load_workflow_plan", lambda wid: plan, raising=False)
    return eng


def _completed(calc_type, settings=None):
    return {
        "calc_id": f"mat_{calc_type}_1",
        "material_id": "mat",
        "calc_type": calc_type,
        "status": "completed",
        "settings_json": json.dumps(settings) if settings else None,
    }


@pytest.mark.parametrize("calc_type", ["OPT", "SP"])
def test_planless_completion_generates_nothing(monkeypatch, calc_type):
    """The bug: OPT -> SP and SP -> BAND+DOSS fired with no plan in sight."""
    monkeypatch.delenv("MACE_WORKFLOW_ID", raising=False)
    monkeypatch.delenv("MACE_PLANLESS_PROGRESSION", raising=False)
    eng = _engine_without_init(monkeypatch, _completed(calc_type))

    def _boom(*a, **kw):  # pragma: no cover - only runs on regression
        raise AssertionError("planless progression generated a calculation")

    for name in ("generate_sp_from_opt", "generate_band_from_sp",
                 "generate_doss_from_sp", "generate_numbered_calculation"):
        monkeypatch.setattr(eng, name, _boom, raising=False)

    assert eng.execute_workflow_step("mat", f"mat_{calc_type}_1") == []


def test_plan_present_still_progresses(monkeypatch):
    """The gate keys on the PLAN, not on the calculation being manual: a deck
    submitted by hand under `--progress` carries a workflow_id and must
    progress exactly like a workflow-submitted one."""
    monkeypatch.delenv("MACE_WORKFLOW_ID", raising=False)
    calc = _completed("OPT", {"workflow_id": "workflow_20260727_120000"})
    plan = {"workflow_sequence": ["OPT", "SP"], "step_configurations": {}}
    eng = _engine_without_init(monkeypatch, calc, plan=plan)

    monkeypatch.setattr(eng, "_calculation_already_exists", lambda m, t: False, raising=False)
    monkeypatch.setattr(eng, "_check_dependencies_met",
                        lambda *a, **kw: (True, None), raising=False)
    monkeypatch.setattr(eng, "generate_numbered_calculation",
                        lambda parent, ct: f"mat_{ct}_new", raising=False)

    assert eng.execute_workflow_step("mat", "mat_OPT_1") == ["mat_SP_new"]


def test_escape_hatch_restores_planless_progression(monkeypatch):
    """The old behavior stays reachable for anyone who relied on it."""
    monkeypatch.setenv("MACE_PLANLESS_PROGRESSION", "1")
    from mace.workflow import engine as engine_mod

    eng = engine_mod.WorkflowEngine.__new__(engine_mod.WorkflowEngine)
    eng.base_work_dir = Path(".")
    eng.allow_planless_progression = engine_mod._planless_progression_allowed()
    assert eng.allow_planless_progression is True

    calc = _completed("OPT")

    class _DB:
        def get_calculation(self, calc_id):
            return calc

        def get_calculations_by_status(self, material_id=None, status=None):
            return [calc]

    eng.db = _DB()
    monkeypatch.setattr(eng, "_cleanup_failed_workflow_dirs", lambda: None, raising=False)
    monkeypatch.setattr(eng, "_load_workflow_plan", lambda wid: None, raising=False)
    monkeypatch.setattr(eng, "_calculation_already_exists", lambda m, t: False, raising=False)
    monkeypatch.setattr(eng, "generate_sp_from_opt", lambda cid: "mat_SP_default", raising=False)

    assert eng.execute_workflow_step("mat", "mat_OPT_1") == ["mat_SP_default"]


# --------------------------------------------------------------------------
# Queue manager: don't even reach the engine, and don't adopt loose decks.
# --------------------------------------------------------------------------
class _StatsDB:
    """Just enough DB for the callback's closing status line."""

    def get_database_stats(self):
        return {"total_materials": 0, "calculations_by_status": {}}


def _manager_without_init():
    from mace.queue.manager import EnhancedCrystalQueueManager

    mgr = EnhancedCrystalQueueManager.__new__(EnhancedCrystalQueueManager)
    mgr.enable_tracking = True
    mgr.workflow_enabled = True
    mgr.auto_submit_followups = True
    mgr.is_workflow_context = False
    mgr.db = _StatsDB()
    return mgr


def test_manual_completion_does_not_plan_next_calculation(monkeypatch):
    monkeypatch.delenv("MACE_WORKFLOW_ID", raising=False)
    mgr = _manager_without_init()
    calc = _completed("OPT")

    class _DB:
        def get_calculation(self, cid):
            return calc

        def get_calculation_by_slurm_id(self, cid):
            return None

    mgr.db = _DB()
    for name in ("extract_and_store_input_settings", "update_file_records",
                 "extract_and_store_properties", "update_material_information"):
        monkeypatch.setattr(mgr, name, lambda *a, **kw: None, raising=False)

    def _boom(*a, **kw):  # pragma: no cover - only runs on regression
        raise AssertionError("manual completion triggered workflow progression")

    monkeypatch.setattr(mgr, "plan_next_calculation", _boom, raising=False)

    mgr.handle_completed_calculation("mat_OPT_1")  # must not raise


def test_workflow_completion_still_plans_next(monkeypatch):
    monkeypatch.delenv("MACE_WORKFLOW_ID", raising=False)
    mgr = _manager_without_init()
    calc = _completed("OPT", {"workflow_id": "workflow_20260727_120000"})

    class _DB:
        def get_calculation(self, cid):
            return calc

        def get_calculation_by_slurm_id(self, cid):
            return None

    mgr.db = _DB()
    for name in ("extract_and_store_input_settings", "update_file_records",
                 "extract_and_store_properties", "update_material_information"):
        monkeypatch.setattr(mgr, name, lambda *a, **kw: None, raising=False)

    called = []
    monkeypatch.setattr(mgr, "plan_next_calculation",
                        lambda mid, cid: called.append((mid, cid)), raising=False)

    mgr.handle_completed_calculation("mat_OPT_1")
    assert called == [("mat", "mat_OPT_1")]


def test_env_workflow_id_counts_as_workflow(monkeypatch):
    """Workflow job scripts export MACE_WORKFLOW_ID; recovery successors can
    reach the callback before their settings are written."""
    monkeypatch.setenv("MACE_WORKFLOW_ID", "workflow_20260727_120000")
    mgr = _manager_without_init()
    assert mgr._belongs_to_workflow(_completed("OPT")) is True


def test_completion_callback_does_not_adopt_loose_decks(monkeypatch):
    """The adoption half of the bug: the callback submitted every untracked
    deck in the tree, including the SP the user had just generated by hand."""
    monkeypatch.delenv("MACE_WORKFLOW_ID", raising=False)
    mgr = _manager_without_init()
    mgr.throttler = None
    mgr.lock_manager = None
    monkeypatch.setattr(mgr, "check_queue_status", lambda: None, raising=False)

    def _boom(*a, **kw):  # pragma: no cover - only runs on regression
        raise AssertionError("completion callback auto-submitted loose inputs")

    monkeypatch.setattr(mgr, "process_new_d12_files", _boom, raising=False)

    mgr.run_callback_check("completion")  # must not raise


def test_submit_new_mode_still_submits(monkeypatch):
    """`--callback-mode submit_new` and `mace manager` are explicit user asks:
    keeping the queue fed is exactly their job, so they are untouched."""
    mgr = _manager_without_init()
    mgr.throttler = None
    mgr.lock_manager = None
    called = []
    monkeypatch.setattr(mgr, "process_new_d12_files",
                        lambda: called.append(True), raising=False)

    mgr.run_callback_check("submit_new")
    assert called == [True]
