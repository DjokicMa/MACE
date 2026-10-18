"""A builder TRANSPORT or CHARGE+POTENTIAL step runs the deck `mace opt2d3`
writes with the same settings on the same parent.

The route a workflow step takes, each piece the real one: the step's form
(Calculation Setup's opt2d3 form, gui/setup.py) answers the CLI's questions
and gives its settings file; the builder writes it into the spec
(workflow_build.spec_files); `mace workflow --spec`'s plan copies it for the
step (spec.build_plan); and the engine, at the SP's completion, runs
CRYSTALOptToD3 with it (WorkflowEngine.generate_d3_calculation_new). The
other route is `mace opt2d3 -i SP.out -t TYPE --config-file FILE`, run as a
subprocess. Parents are real CRYSTAL23 SP outputs from the test/ corpus.
"""
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from mace.gui import setup as S
from mace.gui import workflow_build as wb
from mace.workflow.engine import WorkflowEngine
from mace.workflow.spec import build_plan, load_spec

pytest.importorskip("numpy")

CORPUS = REPO_ROOT / "test" / "SP"
PARENTS = {
    # diamond, an insulator: TRANSPORT's μ relative to its Fermi energy
    "dia": "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized",
    # silver, a metal
    "ag": "Ag1_sym_CRYSTAL_OPT_symm_PBE-D3_POB-TZVP-REV2_opt_B3LYP-D3-D3_optimized_sp_B3LYP-D3-D3_optimized",
}
# the rows each run changes, by the question's label (a repeated question: and its block)
CHANGES = {
    "TRANSPORT": {"Minimum temperature (K) [100]": "200", "Maximum temperature (K) [800]": "600",
                  "Temperature step (K) [50]": "100", "Relaxation time (fs) [10]": "20"},
    # the potential at the atoms (POTC) instead of its 3D grid, and a finer charge density grid
    "CHARGE+POTENTIAL": {"Potential calculation type": "POTC",
                         "Number of grid points per direction [100] (charge density)": "150"},
}
EXPECT = {"TRANSPORT": ["TRANGE", "200.0 600.0 100.0"], "CHARGE+POTENTIAL": ["ECH3", "150", "POTC"]}


def _env():
    env = dict(os.environ)
    for k in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME"):
        env.pop(k, None)
    return env


def _stage(d, name):
    d.mkdir(parents=True)
    for ext in (".out", ".d12"):
        shutil.copy(CORPUS / f"{name}{ext}", d / f"{name}{ext}")
    (d / f"{name}.f9").write_text("wavefunction placeholder\n")
    return d / f"{name}.out"


def _form_settings(tmp_path, parent, calc):
    """The step form's settings file: Setup's opt2d3 form, these rows changed."""
    cache = {}

    def run(answers):
        return S.run_setup(S.Request({"answers": answers, "preview": False}), "opt2d3", calc,
                           tmp_path / f"setup{len(answers)}", parent, parent.name, ("t", calc),
                           lambda k: cache.get(pickle.dumps(k)),
                           lambda k, v, n: cache.__setitem__(pickle.dumps(k), v))
    ids = {f["label"]: f["id"] for f in run({})["form"]}
    r = run({ids[label]: value for label, value in CHANGES[calc].items()})
    assert {f["id"] for f in r["form"] if f["changed"]} == {ids[x] for x in CHANGES[calc]}
    return r["json_text"]


def _plan(tmp_path, parent, calc, settings_text):
    """The builder's spec for OPT -> SP -> calc, and the plan `mace workflow --spec` writes."""
    folder = tmp_path / "decks"
    folder.mkdir()
    deck = folder / "mat_opt.d12"
    shutil.copy(parent.with_suffix(".d12"), deck)
    inp = wb.Inputs("d12", folder, [deck], ["mat_opt"])
    spec_dir = folder / wb.SPEC_DIR / "x"
    key = f"{calc}_3"
    files = wb.spec_files(inp, {"sequence": ["OPT", "SP", calc], "steps": {key: {"settings_text": settings_text}}},
                          spec_dir)
    wb.write_files(spec_dir, files)
    work = tmp_path / "work"
    work.mkdir()
    _plan_file, wid, plan = build_plan(load_spec(spec_dir / "spec.json"), work)
    cfg = plan["step_configurations"][key]
    assert json.loads(Path(cfg["expert_config_file"]).read_text()) == json.loads(settings_text)
    return work, wid


def _engine_deck(work, wid, parent, calc):
    """The engine generating the step from the completed SP, as at its completion."""
    eng = WorkflowEngine.__new__(WorkflowEngine)
    eng.base_work_dir = work

    class _DB:
        def get_calculation(self, cid):
            return {"calc_id": cid, "material_id": "mat", "status": "completed", "output_file": str(parent),
                    "settings_json": json.dumps({"workflow_id": wid})}
    eng.db = _DB()
    eng._find_most_recent_wavefunction_calc = lambda mid: "sp_1"
    out_base = work / "workflow_outputs" / wid
    out_base.mkdir(parents=True)
    eng.get_workflow_output_base = lambda calc_row: out_base
    eng.script_paths = {"crystal_to_d3": str(REPO_ROOT / "Crystal_d3" / "CRYSTALOptToD3.py")}
    made = {}

    def submit(material_id, calc_type, d3_file, work_dir, parent_calc_id):
        made["deck"] = Path(d3_file).read_text()
        return "calc_new"
    eng._create_and_submit_d3_calculation = submit
    assert eng.generate_d3_calculation_new("sp_1", calc) == "calc_new"
    return made["deck"]


def _cli_deck(tmp_path, parent_name, calc, settings_text):
    parent = _stage(tmp_path / "cli", parent_name)
    cfg = tmp_path / "cli_settings.json"
    cfg.write_text(settings_text)
    r = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "opt2d3", "-i", str(parent),
                        "-t", calc, "--config-file", str(cfg)],
                       cwd=parent.parent, capture_output=True, text=True, env=_env(), timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    decks = list(parent.parent.glob("*.d3"))
    assert len(decks) == 1, r.stdout[-2000:]
    return decks[0].read_text()


@pytest.mark.parametrize("calc", ["TRANSPORT", "CHARGE+POTENTIAL"])
@pytest.mark.parametrize("which", sorted(PARENTS))
def test_builder_step_runs_the_opt2d3_deck(tmp_path, which, calc):
    name = PARENTS[which]
    if not (CORPUS / f"{name}.out").exists():
        pytest.skip("test/ data corpus not present")
    settings = _form_settings(tmp_path, _stage(tmp_path / "form", name), calc)
    work, wid = _plan(tmp_path, tmp_path / "form" / f"{name}.out", calc, settings)
    workflow = _engine_deck(work, wid, _stage(tmp_path / "engine", name), calc)
    manual = _cli_deck(tmp_path, name, calc, settings)
    assert workflow == manual
    lines = [ln.strip() for ln in workflow.splitlines()]
    for want in EXPECT[calc]:
        assert want in lines, workflow
