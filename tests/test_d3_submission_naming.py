"""Regression test: the SLURM script generated for a D3 calculation must
reference the deck that is actually on disk.

Real-world failure (testMACE1.1 workflow, 2026-07-02): the engine wrote the
deck as ``1_dia_band.d3`` / ``1_dia_band.f9`` but generated the submit script
with ``JOB=1_dia`` (bare material_id). submit_prop.sh does
``cp $DIR/$JOB.d3 INPUT`` / ``cp $DIR/$JOB.f9 fort.9``, so both cp's failed,
Pproperties launched with an empty INPUT, and every workflow BAND/DOSS died
with MPI_Abort — which in turn dammed FREQ and everything downstream.

The invariant pinned here: JOB in the generated script == the .d3 stem.
"""
import json
import re
from pathlib import Path

import pytest

from mace.workflow.engine import WorkflowEngine


# The three template lines the naming contract lives in (mirrors
# mace/submission/submit_prop.sh; scaffolding only — no CRYSTAL output).
_TEMPLATE = """#!/bin/bash --login
#SBATCH -J $1
export JOB=$1
export DIR=$SLURM_SUBMIT_DIR
cp $DIR/$JOB.d3  $scratch/$JOB/INPUT
cp $DIR/$JOB.f9  $scratch/$JOB/fort.9
mpirun Pproperties 2>&1 >& $DIR/${JOB}.out
"""


class _FakeDB:
    def __init__(self):
        self.created = None

    def get_calculation(self, calc_id):
        return {
            "calc_id": calc_id,
            "material_id": "1_dia",
            "settings_json": json.dumps({"workflow_id": "wf_test"}),
        }

    def create_calculation(self, **kwargs):
        self.created = kwargs
        return "calc_d3_1"

    def update_calculation_status(self, *args, **kwargs):
        pass


@pytest.fixture
def engine(tmp_path):
    eng = WorkflowEngine.__new__(WorkflowEngine)
    eng.base_work_dir = tmp_path
    eng.db = _FakeDB()
    scripts = tmp_path / "workflow_scripts"
    scripts.mkdir()
    (scripts / "submit_prop_band.sh").write_text(_TEMPLATE)
    (scripts / "submit_prop_doss.sh").write_text(_TEMPLATE)
    eng._submit_calculation_to_slurm = lambda script, work_dir: "424242"
    return eng


@pytest.mark.parametrize("calc_type,stem", [
    ("BAND", "1_dia_band"),
    ("DOSS", "1_dia_doss"),
])
def test_d3_submit_script_job_matches_deck_on_disk(engine, tmp_path, calc_type, stem):
    # Flattened layout: the deck sits directly in the material's step dir
    # (no BAND1/DOSS1 instance level).
    work_dir = tmp_path / f"step_003_{calc_type}" / f"1_dia_{calc_type.lower()}"
    work_dir.mkdir(parents=True)
    d3_file = work_dir / f"{stem}.d3"
    d3_file.write_text("BAND\ndummy deck\nEND\n")
    (work_dir / f"{stem}.f9").write_bytes(b"\x00")

    calc_id = engine._create_and_submit_d3_calculation(
        "1_dia", calc_type, d3_file, work_dir, "parent_sp_1"
    )
    assert calc_id == "calc_d3_1"

    # The script must exist, be named after the deck stem, and its JOB must
    # resolve to files that are actually on disk (the bug: JOB=1_dia).
    script = work_dir / f"{stem}.sh"
    assert script.exists(), f"expected {script.name}; found {[p.name for p in work_dir.glob('*.sh')]}"
    content = script.read_text()
    m = re.search(r"^export JOB=(\S+)$", content, re.M)
    assert m, "no JOB= line in generated script"
    job = m.group(1)
    assert (work_dir / f"{job}.d3").exists(), \
        f"JOB={job} but {job}.d3 is not on disk (deck is {d3_file.name})"
    assert (work_dir / f"{job}.f9").exists()


def test_generate_d3_places_deck_flat_in_step_dir(engine, tmp_path, monkeypatch):
    """generate_d3_calculation_new must place the deck directly in the
    material's step dir — the old extra BAND1/DOSS1 instance level was a
    needless single-child subdirectory (numbered repeats are disambiguated by
    the material-dir suffix and the step folder instead)."""
    import mace.workflow.engine as eng_mod

    wf_out = tmp_path / "workflow_outputs" / "workflow_test"
    wf_out.mkdir(parents=True)

    db = engine.db
    db.get_calculation = lambda cid: {
        "calc_id": cid, "material_id": "1_dia", "status": "completed",
        "output_file": str(tmp_path / "1_dia_sp.out"),
        "settings_json": json.dumps({"workflow_id": "workflow_test"}),
    }
    engine._find_most_recent_wavefunction_calc = lambda mid: "wf_calc_1"
    engine.get_workflow_output_base = lambda calc: wf_out
    fake_script = tmp_path / "CRYSTALOptToD3.py"
    fake_script.write_text("# stub\n")
    engine.script_paths = {"crystal_to_d3": str(fake_script)}

    def fake_run(cmd, **kwargs):
        # Emulate CRYSTALOptToD3.py: drop a *_band.d3 (+.f9) into --output-dir
        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        (out_dir / "1_dia_sp_band.d3").write_text("BAND\nEND\n")
        (out_dir / "1_dia_sp_band.f9").write_bytes(b"\x00")

        class R:
            returncode = 0
            stdout = stderr = ""
        return R()

    monkeypatch.setattr(eng_mod.subprocess, "run", fake_run)

    captured = {}

    def fake_submit(material_id, calc_type, d3_file, final_dir, parent_id):
        captured.update(d3=Path(d3_file), final_dir=Path(final_dir))
        return "calc_ok"

    engine._create_and_submit_d3_calculation = fake_submit

    calc_id = engine.generate_d3_calculation_new("src_sp_1", "BAND")
    assert calc_id == "calc_ok"

    step_dir = wf_out / "step_003_BAND" / "1_dia_band"
    assert captured["final_dir"] == step_dir, \
        f"deck dir {captured['final_dir']} (expected flat {step_dir})"
    assert captured["d3"].parent == step_dir
    assert captured["d3"].exists()
    assert not (step_dir / "BAND1").exists(), "redundant instance dir recreated"
