"""Dependency-driven workflow progression.

Real-world failure (testMACE1.1, 2026-07-02): progression was strict
sequence-order, so FREQ — whose only real dependency is OPT — sat behind
BAND/DOSS, and when every DOSS failed, FREQ / CHARGE+POTENTIAL / OPT2 were
never generated at all.

Now: after every completion, execute_workflow_step runs a dependency-driven
sweep (_check_and_trigger_pending_calculations with skip_geometry_steps=True)
that triggers any later planned step whose REAL dependency is satisfied:

  - FREQ fires as soon as OPT completes (parallel to SP/BAND/DOSS)
  - BAND/DOSS/TRANSPORT/CHARGE+POTENTIAL fire once their wavefunction
    provider (SP or OPT) completes
  - OPT-chain steps (OPT2, OPT3...) stay strictly sequence-ordered so
    "highest completed OPT" geometry selection can't race
"""
import json

import pytest

from mace.workflow.engine import WorkflowEngine

SEQ = ["OPT", "SP", "BAND", "DOSS", "FREQ", "CHARGE+POTENTIAL", "OPT2"]


class _FakeDB:
    def __init__(self, calcs):
        self.calcs = calcs

    def get_calculations_by_status(self, material_id=None, **kw):
        return self.calcs

    def update_workflow_state(self, *a, **k):
        pass


def _calc(calc_type, status="completed", cid=None):
    return {"calc_id": cid or f"{calc_type.lower()}_1", "material_id": "mat",
            "calc_type": calc_type, "status": status,
            "settings_json": json.dumps({"workflow_id": "wf_test"})}


@pytest.fixture
def engine():
    eng = WorkflowEngine.__new__(WorkflowEngine)
    eng.triggered = []
    eng._use_new_d3_generation = lambda: True
    eng.generate_freq_from_opt = lambda src, t: eng.triggered.append((t, src)) or f"id_{t}"
    eng.generate_d3_calculation_new = lambda src, t: eng.triggered.append((t, src)) or f"id_{t}"
    eng.generate_property_calculation = lambda src, t: eng.triggered.append(("LEGACY_" + t, src)) or f"id_{t}"
    eng.generate_numbered_calculation = lambda src, t: eng.triggered.append((t, src)) or f"id_{t}"
    eng.generate_calculation_from_cif = lambda mid, t: eng.triggered.append((t, "CIF")) or f"id_{t}"
    return eng


def test_sweep_after_opt_fires_freq_only(engine):
    """OPT just completed (SP already created by the sequence-order branch):
    the sweep must fire FREQ (dep=OPT) — and nothing that needs SP results,
    and no OPT2 (geometry steps stay sequence-ordered)."""
    engine.db = _FakeDB([_calc("OPT")])
    existing = {"OPT", "SP"}  # SP was just created by the normal branch
    engine._calculation_already_exists = lambda mid, t: t in existing

    new_ids = engine._check_and_trigger_pending_calculations(
        "mat", SEQ, skip_geometry_steps=True)

    fired = [t for t, _ in engine.triggered]
    assert fired == ["FREQ"], f"expected only FREQ, got {fired}"
    # FREQ's source is the completed OPT (highest completed OPT)
    assert engine.triggered[0][1] == "opt_1"
    assert new_ids == ["id_FREQ"]


def test_sweep_after_sp_fires_all_wavefunction_consumers(engine):
    """SP completed: BAND, DOSS and CHARGE+POTENTIAL all fire off the SP
    wavefunction via the NEW d3 path; OPT2 still held back."""
    engine.db = _FakeDB([_calc("OPT"), _calc("SP")])
    existing = {"OPT", "SP", "FREQ"}  # FREQ already launched after OPT
    engine._calculation_already_exists = lambda mid, t: t in existing

    engine._check_and_trigger_pending_calculations(
        "mat", SEQ, skip_geometry_steps=True)

    fired = {t for t, _ in engine.triggered}
    assert fired == {"BAND", "DOSS", "CHARGE+POTENTIAL"}
    # all sourced from the SP calc (the wavefunction provider), new-gen path
    assert all(src == "sp_1" for _, src in engine.triggered)
    assert not any(t.startswith("LEGACY_") for t, _ in engine.triggered)


def test_sweep_without_skip_still_triggers_opt_chain(engine):
    """Full (non-sweep) mode retains the old semantics: OPT2 is triggered
    when its dependency is met — used for explicit recovery sweeps."""
    engine.db = _FakeDB([_calc("OPT")])
    existing = {"OPT", "SP", "BAND", "DOSS", "FREQ", "CHARGE+POTENTIAL"}
    engine._calculation_already_exists = lambda mid, t: t in existing

    engine._check_and_trigger_pending_calculations("mat", SEQ)

    assert [t for t, _ in engine.triggered] == ["OPT2"]


def test_execute_workflow_step_runs_dependency_sweep(engine):
    """execute_workflow_step must invoke the sweep (skip_geometry_steps=True)
    and merge its calc IDs into the returned list."""
    completed = _calc("OPT", cid="opt_1")
    engine.db = _FakeDB([completed])
    engine.db.get_calculation = lambda cid: completed
    engine._cleanup_failed_workflow_dirs = lambda: None
    engine.get_workflow_sequence = lambda wid: SEQ
    engine._find_calc_position_in_sequence = lambda *a: 0
    engine._get_next_steps_from_sequence = lambda *a: []  # normal branch: nothing

    seen = {}

    def fake_sweep(material_id, planned_sequence, skip_geometry_steps=False):
        seen.update(material_id=material_id, seq=planned_sequence,
                    skip=skip_geometry_steps)
        return ["id_FREQ"]

    engine._check_and_trigger_pending_calculations = fake_sweep

    new_ids = engine.execute_workflow_step("mat", "opt_1")

    assert seen == {"material_id": "mat", "seq": SEQ, "skip": True}
    assert "id_FREQ" in new_ids


def test_no_plan_defaults_are_idempotent(engine, monkeypatch):
    """Manual (no-plan) submissions: re-firing progression for a completed OPT
    must NOT generate another SP once one exists (real-world: every completion
    callback re-fired the unguarded default and produced sp2 + 4x duplicate
    band/doss jobs); same guard for the SP -> BAND+DOSS default."""
    monkeypatch.delenv("MACE_WORKFLOW_ID", raising=False)
    engine._cleanup_failed_workflow_dirs = lambda: None
    engine.get_workflow_sequence = lambda wid: None
    engine.generate_sp_from_opt = (
        lambda cid: engine.triggered.append(("SP", cid)) or "sp_new")
    engine.generate_doss_from_sp = (
        lambda cid: engine.triggered.append(("DOSS", cid)) or "doss_new")
    engine.generate_band_from_sp = (
        lambda cid: engine.triggered.append(("BAND", cid)) or "band_new")

    # OPT completed, SP already submitted -> re-fire generates NOTHING
    opt = _calc("OPT", cid="opt_1"); opt["settings_json"] = None
    engine.db = _FakeDB([opt, _calc("SP", status="submitted")])
    engine.db.get_calculation = lambda cid: opt
    engine.execute_workflow_step("mat", "opt_1")
    assert engine.triggered == []

    # OPT completed, no SP yet -> generated exactly once
    engine.db = _FakeDB([opt])
    engine.db.get_calculation = lambda cid: opt
    engine.execute_workflow_step("mat", "opt_1")
    assert engine.triggered == [("SP", "opt_1")]

    # SP completed, BAND pending + DOSS absent -> only DOSS generated
    engine.triggered.clear()
    sp = _calc("SP", cid="sp_1"); sp["settings_json"] = None
    engine.db = _FakeDB([opt, sp, _calc("BAND", status="pending")])
    engine.db.get_calculation = lambda cid: sp
    engine.execute_workflow_step("mat", "sp_1")
    assert engine.triggered == [("DOSS", "sp_1")]


def test_wavefunction_selector_skips_empty_f9(engine, tmp_path):
    """Real incident (phase-3 QA): an OOM-killed SP left a 0-byte .f9; BAND
    and DOSS generated from it aborted instantly. The selector must skip
    empty wavefunctions and fall back to the previous valid one (OPT)."""
    opt_out = tmp_path / "mat_opt.out"
    opt_out.write_text("done\n")
    (tmp_path / "mat_opt.f9").write_bytes(b"WAVEFUNCTION")
    sp_out = tmp_path / "mat_sp.out"
    sp_out.write_text("done\n")
    (tmp_path / "mat_sp.f9").write_bytes(b"")  # OOM-killed: empty

    opt = _calc("OPT", cid="opt_1")
    opt["output_file"] = str(opt_out)
    opt["end_time"] = "2026-07-09T00:10:00"
    sp = _calc("SP", cid="sp_1")
    sp["output_file"] = str(sp_out)
    sp["end_time"] = "2026-07-09T00:20:00"
    engine.db = _FakeDB([opt, sp])

    picked = engine._find_most_recent_wavefunction_calc("mat")

    assert picked == "opt_1", (
        f"selector picked {picked}: an empty f9 must be skipped")
