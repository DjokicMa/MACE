"""The workflow builder's later SCF steps change more than the method (1.1.6).

A later step (SP, OPT2, FREQ) inherits its parent's settings for each structure, as the CLI planner's
customization levels do, and can change what those levels change: the functional, the basis set, the DFT
grid, the SCF settings and the k-point mesh on every SCF step, OPT's type and convergence, FREQ's template,
NUMDERIV and IR.
The GUI never writes a setting itself: the server asks Calculation Setup (the CLI) what the answers change
on a stand-in parent and writes those keys into the workflow spec; ``mace workflow --spec`` and the engine
do the rest, so the projected deck is the deck the engine writes.
"""
import json
import shutil

import pytest

import guiharness as H
import submit_util as SU
from test_workflow_builder_api import CIF, _cifs, _draft, _preview, _run, _task

pytestmark = pytest.mark.skipif(not CIF.exists(), reason="test/ data corpus not present")


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    s = SU.Served(tmp_path_factory.mktemp("wfsteps"), MACE_SITE_CONFIG="none")
    yield s
    s.stop()


def _chain(folder, steps, seq=("OPT", "SP", "FREQ")):
    d = _draft(folder, template=None, sequence=list(seq), steps=steps)
    return d


def _projected(prev):
    return {p["step"]: p for p in prev["projected"]}


def test_a_later_step_form_lets_the_planners_settings_change(srv):
    _cifs(srv, "f1")
    draft = _chain("f1", {})
    scf = {"dft_grid", "tolerance", "maxcycle", "fmixing", "scf_method", "levshift", "levshift_values", "spinlock",
           "smearing", "smear_width", "spin", "k_points_for_all_files"}
    for step, editable in (("SP_2", {"functional", "dispersion", "basis_set"} | scf),
                           ("FREQ_3", {"functional", "dispersion", "basis_set", "freq_template",
                                       "numderiv", "ir"} | scf)):
        st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": step})
        assert st == 200, r
        assert set(r["editable"]) == editable
        ids = {f["id"] for f in r["form"]}
        assert editable - {"levshift_values", "smear_width"} <= ids | {"basis_set"}
    # the k-point mesh is editable too (test_workflow_builder_step_kmesh_api.py), and checked
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "SP_2",
                                              "answers": {"k_points_for_all_files": "8 x"}})
    assert st == 400 and "the k-point mesh is" in r["error"]
    # an external basis set and a Hartree-Fock method are the first step's
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "SP_2", "answers": {"functional": "UHF"}})
    assert st == 400 and "Hartree-Fock" in r["error"]


def test_basis_set_and_convergence_reach_the_projected_deck(srv):
    _cifs(srv, "f2")
    steps = {"SP_2": {"answers": {"tolerance": "Tight", "basis_set": "POB-DZVP-REV2"}}}
    st, prev = _preview(srv, _chain("f2", steps, ("OPT", "SP")))
    assert st == 200 and prev["code"] == 0, (prev.get("refused"), (prev.get("cli_log") or "")[-1500:])
    spec = json.loads(prev["spec"]["text"])
    ov = spec["steps"]["SP_2"]["overrides"]
    assert ov["basis_modifications"] == {"new_basis": "POB-DZVP-REV2"}
    assert ov["tolerance_modifications"]["custom_tolerances"]["TOLINTEG"] == "8 8 8 9 24" or \
        ov["tolerance_modifications"]["custom_tolerances"]["TOLINTEG"].split()[-1] != "14"
    sp = _projected(prev)["SP_2"]["deck_text"]
    parent = prev["first_step"][0]["deck_text"]
    assert "POB-DZVP-REV2" in sp or "BASISSET" in sp
    # the parent's own deck is untouched
    assert "POB-TZVP-REV2" in parent
    tol = ov["tolerance_modifications"]["custom_tolerances"]
    assert f"TOLINTEG\n{tol['TOLINTEG']}" in sp


def test_opt_and_freq_steps_take_their_own_settings(srv):
    _cifs(srv, "f3")
    steps = {"OPT2_2": {"answers": {"opt_type": "CELLONLY", "opt_convergence": "Tight"}},
             "FREQ_3": {"answers": {"numderiv": "Two displacements per atom", "ir": True}}}
    st, prev = _preview(srv, _chain("f3", steps, ("OPT", "OPT2", "FREQ")))
    assert st == 200 and prev["code"] == 0, (prev.get("refused"), (prev.get("cli_log") or "")[-1500:])
    spec = json.loads(prev["spec"]["text"])
    assert spec["steps"]["OPT2_2"]["overrides"]["optimization_type"] == "CELLONLY"
    assert spec["steps"]["FREQ_3"]["overrides"]["frequency_settings"]["numderiv"] == 2
    proj = _projected(prev)
    assert "CELLONLY" in proj["OPT2_2"]["deck_text"]
    assert "NUMDERIV" in proj["FREQ_3"]["deck_text"] and "INTENS" in proj["FREQ_3"]["deck_text"]


def test_settings_changed_on_a_step_run_with_the_workflow(srv):
    _cifs(srv, "f4")
    steps = {"SP_2": {"answers": {"tolerance": "Tight"}}}
    st, prev = _preview(srv, _chain("f4", steps, ("OPT", "SP")))
    assert st == 200 and prev["code"] == 0
    st, res = _run(srv, prev["preview_id"])
    assert st == 200 and res["code"] == 0 and res["script_drift"] == [], res
    plan = json.loads((srv.root / res["plan_rel"]).read_text())
    cfg = plan["step_configurations"]["SP_2"]
    assert cfg["tolerance_modifications"]["custom_tolerances"]


def test_a_workflow_can_wait_for_you_between_steps(srv):
    """"Start each next step by itself" off is the spec's execution.auto_progression false, the CLI's pause."""
    _cifs(srv, "f5")
    draft = _chain("f5", {}, ("OPT", "SP"))
    draft["execution"] = {"auto_progression": False}
    st, prev = _preview(srv, draft)
    assert st == 200 and prev["code"] == 0, prev.get("refused")
    assert json.loads(prev["spec"]["text"])["execution"]["auto_progression"] is False
    assert json.loads(prev["plan"]["text"])["execution_settings"]["auto_progression"] is False


def test_a_chosen_cap_of_200_is_kept(srv):
    _cifs(srv, "f6")
    draft = _chain("f6", {}, ("OPT", "SP"))
    draft["execution"] = {"max_jobs": 200}
    st, prev = _preview(srv, draft)
    assert st == 200 and prev["code"] == 0, prev.get("refused")
    ex = json.loads(prev["plan"]["text"])["execution_settings"]
    assert ex["max_concurrent_jobs"] == 200 and ex["max_jobs_explicit"] is True


def test_the_ready_made_chains_carry_the_numbers_of_the_clis_template_menu(srv, tmp_path):
    from mace.workflow.planner import WorkflowPlanner
    cli = {k: i for i, k in enumerate(WorkflowPlanner(work_dir=tmp_path / "plan").workflow_templates, 1)}
    st, _, r = srv.req("api/workflow/templates")
    assert st == 200 and r["templates"]
    assert {t["name"]: t["num"] for t in r["templates"]} == {t["name"]: cli[t["name"]] for t in r["templates"]}
    assert cli["full_electronic"] == 3                       # the planner's default answer
