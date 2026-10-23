"""A later workflow step switches between internal and external basis sets in either direction (1.1.6).

Marcus: "Why does the workflow seemingly let me change the basis set: SP_3: a later step takes an internal
basis set; an external one is chosen on the first step".

The Basis set row of a later step offered the bundled external libraries, and choosing one was refused. CRYSTAL
takes either kind on any deck, and opt2d12 writes either (BASISSET <name>, or the library's records ending
"99 0"); the spec's basis_modifications now carries {"new_basis": <library>, "basis_set_type": "EXTERNAL"}
next to the internal {"new_basis": <name>}, and the engine hands it to opt2d12. A chain external -> internal
-> external projects three decks with the right basis input each.
"""
import json
import re

import pytest

import submit_util as SU
from test_workflow_builder_api import CIF, _cifs, _draft, _preview, _task

pytestmark = pytest.mark.skipif(not CIF.exists(), reason="test/ data corpus not present")


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    s = SU.Served(tmp_path_factory.mktemp("wfbasis"), MACE_SITE_CONFIG="none")
    yield s
    s.stop()


def _internal(deck, name):
    return re.search(rf"^BASISSET\n{re.escape(name)}$", deck, re.M) and not re.search(r"^99 0$", deck, re.M)


def _external(deck):
    return re.search(r"^99 0$", deck, re.M) and not re.search(r"^BASISSET$", deck, re.M)


def test_external_internal_external_chain(srv):
    _cifs(srv, "b1")
    steps = {"OPT_1": {"answers": {"basis_set": "TZVP-REV2"}},
             "SP_2": {"answers": {"basis_set": "POB-DZVP-REV2"}},
             "SP2_3": {"answers": {"basis_set": "DZVP-REV2"}}}
    draft = _draft("b1", template=None, sequence=["OPT", "SP", "SP2"], steps=steps)
    for k in ("SP_2", "SP2_3"):
        st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": k, "answers": steps[k]["answers"]})
        assert st == 200, r
    st, prev = _preview(srv, draft)
    assert st == 200 and prev["code"] == 0, (prev.get("refused"), (prev.get("cli_log") or "")[-1500:])
    spec = json.loads(prev["spec"]["text"])["steps"]
    assert spec["SP_2"]["overrides"]["basis_modifications"] == {"new_basis": "POB-DZVP-REV2"}
    assert spec["SP2_3"]["overrides"]["basis_modifications"] == {"new_basis": "full.basis.doublezeta",
                                                                 "basis_set_type": "EXTERNAL"}
    opt = prev["first_step"][0]["deck_text"]
    proj = {p["step"]: p["deck_text"] for p in prev["projected"]}
    assert _external(opt), opt
    assert _internal(proj["SP_2"], "POB-DZVP-REV2"), proj["SP_2"]
    assert _external(proj["SP2_3"]), proj["SP2_3"]
    # the double-zeta library's carbon, not the triple-zeta parent's: fewer shells than TZVP-REV2's
    assert proj["SP2_3"].count("\n0 ") < opt.count("\n0 ")


def test_internal_then_external(srv):
    _cifs(srv, "b2")
    steps = {"SP_2": {"answers": {"basis_set": "TZVP-REV2"}}}
    st, prev = _preview(srv, _draft("b2", template=None, sequence=["OPT", "SP"], steps=steps))
    assert st == 200 and prev["code"] == 0, (prev.get("refused"), (prev.get("cli_log") or "")[-1500:])
    assert _internal(prev["first_step"][0]["deck_text"], "POB-TZVP-REV2")
    assert _external({p["step"]: p["deck_text"] for p in prev["projected"]}["SP_2"])
