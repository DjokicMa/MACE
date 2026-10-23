"""A later workflow step's k-point mesh is inherited and can be overridden (1.1.6).

Marcus: "Of course by defualt things should be inherited but we should always be allowed to customize the  (spelling-ok: quoted)
workflow."

The k-point mesh row of a later SCF step was the one row that stayed locked. It now shows its parent step's
SHRINK record as inherited ("SHRINK 12 24") and takes the step's own: "IS ISP", "IS1 IS2 IS3" per axis or
"auto" (MACE's mesh from each structure's cell). The answer reaches Calculation Setup as opt2d12's
k_points_for_all_files patch (so the form's deck is the CLI's), the spec as kpoint_modifications, the engine's
projected deck as its SHRINK, and the step after inherits it. A mesh the space group cannot have is raised to
agree, and the row says so.
"""
import json

import pytest

import submit_util as SU
from test_workflow_builder_api import CIF, _cifs, _draft, _preview, _task

pytestmark = pytest.mark.skipif(not CIF.exists(), reason="test/ data corpus not present")
KMESH = "k_points_for_all_files"


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    s = SU.Served(tmp_path_factory.mktemp("wfkmesh"), MACE_SITE_CONFIG="none")
    yield s
    s.stop()


def _shrink(deck):
    lines = [line.strip() for line in deck.splitlines()]
    i = lines.index("SHRINK")
    return lines[i + 1] if not lines[i + 1].startswith("0 ") else f"{lines[i + 1]} / {lines[i + 2]}"


def _row(form):
    return {f["id"]: f for f in form["form"]}[KMESH]


def test_a_later_steps_kmesh_row_is_inherited_and_editable(srv):
    _cifs(srv, "k1")
    draft = _draft("k1", template=None, sequence=["OPT", "OPT2", "SP"], steps={})
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "OPT2_2"})
    assert st == 200, r
    assert KMESH in r["editable"]
    row = _row(r)
    # diamond's OPT deck: SHRINK 12 24, shown in the deck's words and as the value an answer takes
    assert row["label"] == "k-point mesh (SHRINK)" and row["section"] == "k-points and SCF"
    assert row["default"] == "12 24" and row["value"] == "12 24" and row["default_label"] == "SHRINK 12 24"
    assert row["suggestions"] == ["auto", "12 24"] and not row.get("answered") and not row.get("note")
    assert "GUESSP" in row["hint"] and "IS ISP" in row["hint"]
    assert _shrink(r["deck"]) == "12 24"


def test_opt2_changes_shrink_and_the_sp_after_inherits_it(srv):
    _cifs(srv, "k2")
    steps = {"OPT2_2": {"answers": {KMESH: "9 18"}}}
    draft = _draft("k2", template=None, sequence=["OPT", "OPT2", "SP"], steps=steps)
    st, form = _task(srv, "workflow/step-form", {"draft": draft, "step": "OPT2_2", "answers": steps["OPT2_2"]["answers"]})
    assert st == 200, form
    row = _row(form)
    assert row["value"] == "9 18" and row["default"] == "12 24" and row["changed"] and not row.get("note")
    assert _shrink(form["deck"]) == "9 18" and "fresh SCF guess" not in form["deck"].splitlines()[0]
    # the SP after OPT2 inherits OPT2's mesh
    st, sp = _task(srv, "workflow/step-form", {"draft": draft, "step": "SP_3"})
    assert st == 200, sp
    assert _row(sp)["default"] == "9 18" and _row(sp)["default_label"] == "SHRINK 9 18"
    st, prev = _preview(srv, draft)
    assert st == 200 and prev["code"] == 0, (prev.get("refused"), (prev.get("cli_log") or "")[-1500:])
    spec = json.loads(prev["spec"]["text"])["steps"]
    assert spec["OPT2_2"]["overrides"] == {"kpoint_modifications": {"new_k_points": "9 18"}}
    assert "kpoint_modifications" not in ((spec.get("SP_3") or {}).get("overrides") or {})
    assert _shrink(prev["first_step"][0]["deck_text"]) == "12 24"
    proj = {p["step"]: p["deck_text"] for p in prev["projected"]}
    assert _shrink(proj["OPT2_2"]) == "9 18" and _shrink(proj["SP_3"]) == "9 18"


def test_auto_and_a_mesh_the_symmetry_cannot_have_say_what_the_deck_gets(srv):
    _cifs(srv, "k3")
    draft = _draft("k3", template=None, sequence=["OPT", "SP"], steps={})
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "SP_2", "answers": {KMESH: "auto"}})
    assert st == 200, r
    written = _shrink(r["deck"])
    assert _row(r)["note"] == f"auto: SHRINK {written} for A, from its cell; each structure gets its own"
    # Fd-3m: one factor along the three primitive vectors
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "SP_2", "answers": {KMESH: "9 9 6"}})
    assert st == 200, r
    assert _shrink(r["deck"]) == "9 18"
    assert _row(r)["note"].startswith("written as SHRINK 9 18: directions the symmetry makes equivalent")
    st, prev = _preview(srv, dict(draft, steps={"SP_2": {"answers": {KMESH: "auto"}}}))
    assert st == 200 and prev["code"] == 0, (prev.get("refused"), (prev.get("cli_log") or "")[-1500:])
    assert json.loads(prev["spec"]["text"])["steps"]["SP_2"]["overrides"] == \
        {"kpoint_modifications": {"new_k_points": "auto"}}
    assert _shrink({p["step"]: p["deck_text"] for p in prev["projected"]}["SP_2"]) == written


def test_a_bad_mesh_is_refused_with_the_forms_it_takes(srv):
    _cifs(srv, "k4")
    draft = _draft("k4", template=None, sequence=["OPT", "SP"], steps={})
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "SP_2", "answers": {KMESH: "8 x"}})
    assert st == 400 and "the k-point mesh is" in r["error"] and '"IS ISP"' in r["error"]
