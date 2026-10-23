"""A later workflow step's spin polarization is inherited and can be overridden (1.1.6).

Marcus: "Only spin polarization stays locked. Why? [...] Imagine a case where to get a tough geometry to
converge we dont add spin in the preopt, then for the opt2 and onward want spin".

The Spin polarized row of a later SCF step shows its parent step's spin treatment and changes it for that
step alone: the spec gets scf_modifications {"SPIN": ...}, the engine's deck (the projection) and the step
form's own deck carry SPIN, and the step after inherits it. A change starts the step from a fresh SCF guess
(the row's note, the deck title, the spec check's warning). A SOC step treats spin itself: its row stays
inherited.
"""
import json

import pytest

import submit_util as SU
from test_workflow_builder_api import CIF, _cifs, _draft, _preview, _task

pytestmark = pytest.mark.skipif(not CIF.exists(), reason="test/ data corpus not present")


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    s = SU.Served(tmp_path_factory.mktemp("wfspin"), MACE_SITE_CONFIG="none")
    yield s
    s.stop()


def _spin(deck):
    return "SPIN" in [line.strip() for line in deck.splitlines()]


def test_a_later_steps_spin_row_is_inherited_and_editable(srv):
    _cifs(srv, "s1")
    draft = _draft("s1", template=None, sequence=["OPT", "OPT2", "SP"], steps={})
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "OPT2_2"})
    assert st == 200, r
    assert "spin" in r["editable"]
    row = {f["id"]: f for f in r["form"]}["spin"]
    # the workflow's default stays spin-polarized, inherited from OPT
    assert row["value"] == "Yes" and row["default"] == "Yes" and not row.get("answered")
    assert "fresh SCF guess" in row["hint"] and not row.get("note")


def test_marcus_chain_closed_shell_preopt_then_spin(srv):
    _cifs(srv, "s2")
    steps = {"OPT_1": {"answers": {"spin": "No"}}, "OPT2_2": {"answers": {"spin": "Yes"}}}
    draft = _draft("s2", template=None, sequence=["OPT", "OPT2", "SP"], steps=steps)
    st, form = _task(srv, "workflow/step-form", {"draft": draft, "step": "OPT2_2", "answers": steps["OPT2_2"]["answers"]})
    assert st == 200, form
    row = {f["id"]: f for f in form["form"]}["spin"]
    assert row["default"] == "No" and row["value"] == "Yes" and row["changed"]
    assert row["note"] == "starts from a fresh guess: the parent's wavefunction has no spin"
    assert _spin(form["deck"]) and form["deck"].splitlines()[0].endswith("fresh SCF guess")
    # the SP after OPT2 inherits its spin
    st, sp_form = _task(srv, "workflow/step-form", {"draft": draft, "step": "SP_3"})
    assert st == 200, sp_form
    sp_row = {f["id"]: f for f in sp_form["form"]}["spin"]
    assert sp_row["value"] == "Yes" and sp_row["default"] == "Yes" and not sp_row.get("note")
    st, prev = _preview(srv, draft)
    assert st == 200 and prev["code"] == 0, (prev.get("refused"), (prev.get("cli_log") or "")[-1500:])
    spec = json.loads(prev["spec"]["text"])["steps"]
    assert spec["OPT2_2"]["overrides"] == {"scf_modifications": {"SPIN": True}}
    assert "SP_3" not in spec or "scf_modifications" not in (spec["SP_3"].get("overrides") or {})
    assert not _spin(prev["first_step"][0]["deck_text"])
    proj = {p["step"]: p["deck_text"] for p in prev["projected"]}
    assert _spin(proj["OPT2_2"]) and proj["OPT2_2"].splitlines()[0].endswith(
        " - SPIN on (closed-shell parent), fresh SCF guess")
    assert _spin(proj["SP_3"]) and "fresh SCF guess" not in proj["SP_3"].splitlines()[0]
    st, d = _task(srv, "workflow/defaults", {"draft": draft})
    assert st == 200, d
    assert any("starts from a fresh SCF guess" in w for w in d["steps"]["OPT2_2"]["warnings"])
    assert d["steps"]["SP_3"]["warnings"] == []


def test_spin_off_for_one_step_drops_spin_and_spinlock(srv):
    _cifs(srv, "s3")
    answers = {"spin": "No"}
    draft = _draft("s3", template=None, sequence=["OPT", "SP"], steps={"SP_2": {"answers": answers}})
    st, form = _task(srv, "workflow/step-form", {"draft": draft, "step": "SP_2", "answers": answers})
    assert st == 200, form
    assert {f["id"]: f for f in form["form"]}["spin"]["note"] == \
        "starts from a fresh guess: the parent's wavefunction is spin-polarized"
    st, prev = _preview(srv, draft)
    assert st == 200 and prev["code"] == 0, (prev.get("refused"), (prev.get("cli_log") or "")[-1500:])
    assert json.loads(prev["spec"]["text"])["steps"]["SP_2"]["overrides"] == {"scf_modifications": {"SPIN": False}}
    sp = {p["step"]: p["deck_text"] for p in prev["projected"]}["SP_2"]
    assert not _spin(sp) and "SPINLOCK" not in sp
    assert sp.splitlines()[0].endswith(" - SPIN off (spin-polarized parent), fresh SCF guess")
    assert _spin(prev["first_step"][0]["deck_text"])


def test_a_soc_steps_spin_row_stays_inherited(srv):
    _cifs(srv, "s4")
    draft = _draft("s4", template=None, sequence=["OPT", "SP"], steps={})
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "SP_2", "patches": {"soc": True}})
    assert st == 200, r
    assert "spin" not in r["editable"]
    assert "treats spin itself" in {f["id"]: f for f in r["form"]}["spin"]["hint"]
