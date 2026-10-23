"""The workflow builder's TRANSPORT relaxation time and TAU step (1.1.6).

TRANSPORT's form carries a Relaxation time group (Default 10 fs / Fixed /
From deformation potentials); the last puts a TAU step before TRANSPORT.
The TAU step's form says how many strained single points each structure
gets (from the parent deck's symmetry) and takes the step's options. Both
become the workflow spec's overrides; the preview runs the CLI, so the plan
has the TAU step and the projection shows its strained decks (from stand-in
parents) and the TRANSPORT deck written after them.
"""
import json

import pytest

import submit_util as SU
from test_workflow_builder_api import CIF, _cifs, _draft, _preview, _task

pytestmark = pytest.mark.skipif(not CIF.exists(), reason="test/ data corpus not present")


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    s = SU.Served(tmp_path_factory.mktemp("wftau"), MACE_SITE_CONFIG="none")
    yield s
    s.stop()


SEQ = ["OPT", "SP", "BAND", "DOSS", "TAU", "TRANSPORT"]


def test_templates_list_the_tau_step(srv):
    st, _, r = srv.req("api/workflow/templates")
    assert st == 200 and r["calc_types"]["TAU"]["depends_on"] == ["SP"]
    assert "deformation potentials" in r["calc_types"]["TAU"]["help"]


def test_transport_form_has_the_relaxation_time(srv):
    _cifs(srv, "t1")
    st, r = _task(srv, "workflow/step-form", {"draft": _draft("t1", template=None, sequence=SEQ, steps={}),
                                              "step": "TRANSPORT_6"})
    assert st == 200, r
    assert r["relaxation"] == {"value": "auto", "tau_step": "TAU_5", "default_fs": 10}
    st, r = _task(srv, "workflow/step-form", {"draft": _draft("t1", template=None, sequence=SEQ[:4] + ["TRANSPORT"],
                                                              steps={"TRANSPORT_5": {"relaxation_time": 25}}),
                                              "step": "TRANSPORT_5"})
    assert st == 200 and r["relaxation"]["value"] == 25 and r["relaxation"]["tau_step"] is None


def test_tau_form_counts_the_strained_runs(srv):
    _cifs(srv, "t2")
    draft = _draft("t2", template=None, sequence=SEQ, steps={})
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "TAU_5"})
    assert st == 200, r
    assert r["mode"] == "tau" and r["axes"] == ["x"] and r["n_runs"] == 5        # diamond: cubic
    assert r["parent"] == "SP_2" and r["transport"] == "TRANSPORT_6" and r["masses"] == ["BAND", "DOSS"]
    assert r["options"] == r["defaults"]
    draft["steps"] = {"TAU_5": {"tau": {"axes": ["x", "z"], "carrier": "hole"}}}
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "TAU_5"})
    assert st == 200 and r["n_runs"] == 9 and r["options"]["carrier"] == "hole"
    draft["steps"] = {"TAU_5": {"tau": {"carrier": "both"}}}
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "TAU_5"})
    assert st == 400 and "carrier" in r["error"]


def test_preview_writes_the_spec_plan_and_projects_the_strained_decks(srv):
    _cifs(srv, "t3", names=("A",))
    steps = {"TAU_5": {"tau": {"carrier": "hole", "strict": False}},
             "TRANSPORT_6": {"relaxation_time": "auto"}}
    st, prev = _preview(srv, _draft("t3", names=("A",), template=None, sequence=SEQ, steps=steps))
    assert st == 200 and prev["code"] == 0, (prev.get("refused"), (prev.get("cli_log") or "")[-2000:])
    spec = json.loads(prev["spec"]["text"])
    assert spec["sequence"] == SEQ
    assert spec["steps"]["TAU_5"]["overrides"] == {"carrier": "hole", "strict": False}
    assert spec["steps"]["TRANSPORT_6"]["overrides"] == {"relaxation_time": "auto"}
    plan = json.loads(prev["plan"]["text"])
    assert plan["step_configurations"]["TAU_5"]["tau"]["carrier"] == "hole"
    assert any("submitcrystal23_tau_5.sh" in w["rel"] for w in prev["workflow_scripts"])
    taus = [p for p in prev["projected"] if p["step"] == "TAU_5"]
    assert len(taus) == 5 and all(p["parent"] == "SP_2" for p in taus)
    assert any("ELASTIC" in (p["deck_text"] or "") for p in taus)
    assert sorted(p["deck_name"].rsplit("_dp_", 1)[1] for p in taus) == \
        sorted(f"x_{t}.d12" for t in ("m1pct", "m0_5pct", "p0pct", "p0_5pct", "p1pct"))
    tr = [p for p in prev["projected"] if p["step"] == "TRANSPORT_6"]
    assert len(tr) == 1 and "BOLTZTRA" in tr[0]["deck_text"]


def test_a_relaxation_time_on_another_step_is_refused(srv):
    _cifs(srv, "t4", names=("A",))
    st, prev = _preview(srv, _draft("t4", names=("A",), steps={"BAND_3": {"relaxation_time": 20}}))
    assert st == 400 and "TRANSPORT step's setting" in prev["error"]
