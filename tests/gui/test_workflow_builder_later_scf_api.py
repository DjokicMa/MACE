"""A later workflow step's DFT grid and SCF block are inherited, and each can be overridden (1.1.6).

Marcus: "why are following steps locked like i cant switch DFT grid, its greyed out and says inherited  (british-ok: quoted)
from OPT (step 1) [...] in case i do a OPT 1 and then OPT2 and so on, i may want to change that. And
especially not being able to change stuff like max cycles and fmixing".

A later SP, OPT2 or FREQ step shows its parent step's DFT grid, tolerances, MAXCYCLE, FMIXING, mixing method,
LEVSHIFT, SPINLOCK and smearing, inherited by default; a value changed on the step is written into the spec
(grid_modifications, scf_modifications), read by the engine and written into the deck by CRYSTALOptToD12.
The step form's deck (opt2d12 answered interactively) and the projected deck (the engine's config path) carry
the same SCF records. An explicit MAXCYCLE on a FREQ step wins over the 200-cycle floor, with a warning.
"""
import json
import re

import pytest

import submit_util as SU
from test_workflow_builder_api import CIF, _cifs, _draft, _preview, _task

pytestmark = pytest.mark.skipif(not CIF.exists(), reason="test/ data corpus not present")

SCF_IDS = {"dft_grid", "tolerance", "maxcycle", "fmixing", "scf_method", "spinlock", "smearing"}


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    s = SU.Served(tmp_path_factory.mktemp("wfscf"), MACE_SITE_CONFIG="none")
    yield s
    s.stop()


def _record(deck, word):
    m = re.search(rf"^{word}\n(.+)$", deck, re.M)
    return m.group(1).strip() if m else None


def _projected(prev):
    return {p["step"]: p for p in prev["projected"]}


def test_a_later_step_shows_the_parents_scf_settings_and_lets_each_change(srv):
    _cifs(srv, "g1")
    draft = _draft("g1", template=None, sequence=["OPT", "OPT2", "SP", "FREQ"], steps={})
    for step in ("OPT2_2", "SP_3", "FREQ_4"):
        st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": step})
        assert st == 200, r
        assert SCF_IDS <= set(r["editable"]), (step, r["editable"])
        rows = {f["id"]: f for f in r["form"]}
        assert SCF_IDS - {"spinlock"} <= set(rows), sorted(rows)
        # the inherited values are the parent step's (the workflow's OPT deck: XLGRID, MAXCYCLE 800)
        assert rows["dft_grid"]["value"] == "XLGRID"
        assert str(rows["maxcycle"]["value"]) == "800"
        assert rows["scf_method"]["value"] == "DIIS" and {o["head"] for o in rows["scf_method"]["options"]} >= {"DIIS", "Broyden"}
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "FREQ_4"})
    assert "at least 200" in {f["id"]: f for f in r["form"]}["maxcycle"]["hint"]


def test_scf_overrides_reach_the_spec_and_the_projected_deck(srv):
    _cifs(srv, "g2")
    answers = {"dft_grid": "XXLGRID", "maxcycle": "350", "fmixing": "40", "scf_method": "Broyden"}
    draft = _draft("g2", template=None, sequence=["OPT", "SP"], steps={"SP_2": {"answers": answers}})
    st, form = _task(srv, "workflow/step-form", {"draft": draft, "step": "SP_2", "answers": answers})
    assert st == 200, form
    st, prev = _preview(srv, draft)
    assert st == 200 and prev["code"] == 0, (prev.get("refused"), (prev.get("cli_log") or "")[-1500:])
    ov = json.loads(prev["spec"]["text"])["steps"]["SP_2"]["overrides"]
    assert ov["grid_modifications"] == {"new_grid": "XXLGRID"}
    assert ov["scf_modifications"] == {"MAXCYCLE": 350, "FMIXING": 40, "METHOD": "BROYDEN"}
    sp = _projected(prev)["SP_2"]["deck_text"]
    assert "XXLGRID" in sp and _record(sp, "MAXCYCLE") == "350" and _record(sp, "FMIXING") == "40"
    assert re.search(r"^BROYDEN$", sp, re.M)
    # the step form's own deck (opt2d12 asked interactively) has the same SCF block as the engine's
    for word in ("MAXCYCLE", "FMIXING", "TOLDEE", "TOLINTEG"):
        assert _record(form["deck"], word) == _record(sp, word), word
    assert ("XXLGRID" in form["deck"]) and re.search(r"^BROYDEN$", form["deck"], re.M)
    # the parent step is untouched
    opt = prev["first_step"][0]["deck_text"]
    assert "XXLGRID" not in opt and _record(opt, "MAXCYCLE") == "800"


def test_an_explicit_maxcycle_on_a_freq_step_wins_over_the_floor_with_a_warning(srv):
    _cifs(srv, "g3")
    steps = {"FREQ_2": {"answers": {"maxcycle": "150"}}}
    draft = _draft("g3", template=None, sequence=["OPT", "FREQ"], steps=steps)
    st, form = _task(srv, "workflow/step-form", {"draft": draft, "step": "FREQ_2", "answers": steps["FREQ_2"]["answers"]})
    assert st == 200, form
    assert "below the 200" in ({f["id"]: f for f in form["form"]}["maxcycle"].get("note") or "")
    st, prev = _preview(srv, draft)
    assert st == 200 and prev["code"] == 0, (prev.get("refused"), (prev.get("cli_log") or "")[-1500:])
    assert json.loads(prev["spec"]["text"])["steps"]["FREQ_2"]["overrides"]["scf_modifications"] == {"MAXCYCLE": 150}
    assert _record(_projected(prev)["FREQ_2"]["deck_text"], "MAXCYCLE") == "150"
    st, d = _task(srv, "workflow/defaults", {"draft": draft})
    assert st == 200, d
    assert any("MAXCYCLE 150 is below the 200" in w for w in d["steps"]["FREQ_2"]["warnings"])


def test_an_inherited_maxcycle_on_a_freq_step_keeps_the_floor(srv):
    _cifs(srv, "g4")
    st, prev = _preview(srv, _draft("g4", template=None, sequence=["OPT", "FREQ"], steps={}))
    assert st == 200 and prev["code"] == 0, prev.get("refused")
    assert int(_record(_projected(prev)["FREQ_2"]["deck_text"], "MAXCYCLE")) >= 200


def test_level_shifting_and_smearing_switch_on_for_one_step(srv):
    _cifs(srv, "g5")
    answers = {"levshift": "Yes", "smearing": "Yes"}
    draft = _draft("g5", template=None, sequence=["OPT", "SP"], steps={"SP_2": {"answers": answers}})
    st, form = _task(srv, "workflow/step-form", {"draft": draft, "step": "SP_2", "answers": answers})
    assert st == 200, form
    rows = {f["id"]: f for f in form["form"]}
    assert rows["levshift_values"]["value"] == "5 1"          # the CLI's default for a new LEVSHIFT
    st, prev = _preview(srv, draft)
    assert st == 200 and prev["code"] == 0, (prev.get("refused"), (prev.get("cli_log") or "")[-1500:])
    scf = json.loads(prev["spec"]["text"])["steps"]["SP_2"]["overrides"]["scf_modifications"]
    assert scf["LEVSHIFT"] == [5, 1] and scf["SMEAR"] == pytest.approx(0.01)
    sp = _projected(prev)["SP_2"]["deck_text"]
    assert _record(sp, "LEVSHIFT").split() == ["5", "1"] and float(_record(sp, "SMEAR")) == pytest.approx(0.01)
    assert "LEVSHIFT" not in prev["first_step"][0]["deck_text"]
