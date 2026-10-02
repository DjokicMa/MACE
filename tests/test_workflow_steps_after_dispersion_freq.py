"""Workflow steps after a FREQ step are never built from the FREQ run.

opt2d12 refuses an SP/OPT/FREQ child of a phonon-dispersion (SCELPHONO) FREQ
run, whose output geometry is the supercell, and advises building it from
the OPT or SP the dispersion run started from. The planner accepts plans
such as OPT -> FREQ(dispersion) -> SP -> OPT2 -> FREQ2; this pins that the
engine already builds every such later step from the highest completed OPT
(or SP), so the refusal is never reached on the workflow route.
"""
import json

import pytest

from mace.workflow.engine import WorkflowEngine

SEQ = ["OPT", "FREQ", "SP", "OPT2", "FREQ2", "BAND"]


class _FakeDB:
    def __init__(self, calcs):
        self.calcs = {c["calc_id"]: c for c in calcs}

    def get_calculation(self, cid):
        return self.calcs.get(cid)

    def get_calculations_by_status(self, material_id=None, **kw):
        return list(self.calcs.values())

    def update_workflow_state(self, *a, **k):
        pass


def _calc(calc_type, cid, end):
    return {"calc_id": cid, "material_id": "mat", "calc_type": calc_type,
            "status": "completed", "end_time": end,
            "settings_json": json.dumps({"workflow_id": "wf_test"})}


@pytest.mark.parametrize("done", [["OPT", "FREQ"], ["OPT", "FREQ", "SP"],
                                  ["OPT", "FREQ", "SP", "OPT2", "FREQ2"]])
def test_no_step_takes_the_freq_run_as_its_parent(done):
    eng = WorkflowEngine.__new__(WorkflowEngine)
    calcs = [_calc(t, f"{t.lower()}_id", f"2026-01-0{i + 1}") for i, t in enumerate(done)]
    eng.db = _FakeDB(calcs)
    eng.get_workflow_sequence = lambda wf: SEQ
    eng._cleanup_failed_workflow_dirs = lambda: None
    eng._calculation_already_exists = lambda mid, t: t in done
    eng._use_new_d3_generation = lambda: True
    eng._find_most_recent_wavefunction_calc = lambda mid: None
    triggered = []
    for name in ("generate_numbered_calculation", "generate_freq_from_opt",
                 "generate_d3_calculation_new", "generate_property_calculation"):
        setattr(eng, name, lambda src, t, name=name: triggered.append((t, src)) or f"new_{t}")

    eng.execute_workflow_step("mat", calcs[-1]["calc_id"])

    assert triggered, "nothing was generated"
    freq_ids = {c["calc_id"] for c in calcs if c["calc_type"].startswith("FREQ")}
    assert not [t for t in triggered if t[1] in freq_ids], triggered
