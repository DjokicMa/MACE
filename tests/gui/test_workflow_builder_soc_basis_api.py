"""A later SP step switched to spin-orbit coupling gets a basis set a SOC deck can take (1.1.6).

Marcus: "with SP and so on, if I switch to SOC, it should be easier to manage the basis set stuff, like it
complains that its not listed as external basis set since it was stuck on inherited?"

A SOC deck needs explicit basis input (no BASISSET). A later SP step that inherits an internal basis set
(the workflow's default POB-TZVP-REV2) shows, in its Spin-orbit coupling group, what its SOC deck is written
with: the external library with the same shells (TZVP-REV2) and, when every ECP atom of the structures has
one, the official pob-TZVP-rev2-SOC basis set; an element without one is named, and the ECPs get INPSOC. Left
on the internal set, the form and the preview say which row to change instead of the CLI's settings-file
words. A functional the two-component SCF does not take (HSE06) is named with the ones it does.
"""
import json
import re

import pytest

import submit_util as SU
from test_setup_soc_api import PBTE_CIF
from test_workflow_builder_api import CIF, _draft, _preview, _task

pytestmark = pytest.mark.skipif(not CIF.exists(), reason="test/ data corpus not present")

LATE_CIF = PBTE_CIF.replace("data_PbTe", "data_LaTe").replace("Pb1 Pb", "La1 La")


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    s = SU.Served(tmp_path_factory.mktemp("wfsocb"), MACE_SITE_CONFIG="none")
    yield s
    s.stop()


def _write(srv, folder, text):
    d = srv.root / folder
    d.mkdir(parents=True, exist_ok=True)
    for n in ("A", "B"):
        (d / f"{n}.cif").write_text(text)


def _soc_form(srv, folder, answers=None, soc=None, functional=None, first=None):
    soc = soc or {"soc": True}
    answers = dict(answers or {}, **({"functional": functional} if functional else {}))
    steps = {"SP_2": {"soc": soc, "answers": answers}}
    if functional:
        steps["SP_2"]["method"] = {"functional": functional, "dispersion": False}
    if first:
        steps["OPT_1"] = {"answers": first}
    draft = _draft(folder, template=None, sequence=["OPT", "SP"], steps=steps)
    body = {"draft": draft, "step": "SP_2", "answers": answers, "patches": soc}
    return draft, _task(srv, "workflow/step-form", body)


def test_soc_offers_the_matching_external_basis_and_the_official_soc_set(srv):
    _write(srv, "p1", PBTE_CIF)
    draft, (st, r) = _soc_form(srv, "p1")
    assert st == 200, r
    g = r["soc_group"]["basis"]
    assert g["parent_kind"] == "INTERNAL" and g["basis_set"] == "TZVP-REV2" and g["soc_basis"] == "pob-tzvp-rev2-soc"
    assert "same shells as POB-TZVP-REV2" in g["note"] and ("Pb, Te" in g["note"] or "Te, Pb" in g["note"])
    assert "PBE0" in r["soc_group"]["functionals"] and "HSE06" not in r["soc_group"]["functionals"]
    # left on the inherited internal set: the row to change, not the CLI's settings-file words
    msgs = [p["message"] for p in r["problems"] if p.get("field") == "soc"]
    assert any("needs an external basis set" in m and "pick TZVP-REV2" in m for m in msgs), msgs
    assert not any("basis_set_type" in m for m in msgs), msgs
    st, prev = _preview(srv, draft)
    assert st != 200 and "pick TZVP-REV2" in prev["error"], prev


def test_the_suggested_choices_write_the_soc_deck(srv):
    _write(srv, "p2", PBTE_CIF)
    soc = {"soc": True, "soc_basis": "pob-tzvp-rev2-soc"}
    draft, (st, r) = _soc_form(srv, "p2", answers={"basis_set": "TZVP-REV2"}, soc=soc)
    assert st == 200, r
    assert not [p for p in r["problems"] if p.get("severity") == "error"], r["problems"]
    assert "TWOCOMPON" in r["deck"] and "STUTSC" in r["deck"]
    st, prev = _preview(srv, draft)
    assert st == 200 and prev["code"] == 0, (prev.get("error"), prev.get("refused"))
    ov = json.loads(prev["spec"]["text"])["steps"]["SP_2"]["overrides"]
    assert ov["basis_modifications"] == {"new_basis": "full.basis.triplezeta", "basis_set_type": "EXTERNAL"}
    sp = {p["step"]: p["deck_text"] for p in prev["projected"]}["SP_2"]
    assert "TWOCOMPON" in sp and not re.search(r"^BASISSET$", sp, re.M)


def test_an_element_without_a_soc_basis_is_named(srv):
    """La has an ECP in TZVP-REV2 (the first step's external basis here) and no pob-TZVP-rev2-SOC set."""
    _write(srv, "p3", LATE_CIF)
    _, (st, r) = _soc_form(srv, "p3", first={"basis_set": "TZVP-REV2"})
    assert st == 200, r
    g = r["soc_group"]["basis"]
    assert g["parent_kind"] == "EXTERNAL" and g["basis_set"] is None
    assert g["missing"] == ["La"] and g["soc_basis"] is None
    assert "La has no pob-TZVP-rev2-SOC basis set" in g["note"] and "INPSOC" in g["note"]


def test_a_functional_without_a_two_component_form_is_named(srv):
    _write(srv, "p4", PBTE_CIF)
    _, (st, r) = _soc_form(srv, "p4", answers={"basis_set": "TZVP-REV2"}, functional="HSE06")
    assert st == 200, r
    msgs = [p["message"] for p in r["problems"] if p.get("field") == "soc"]
    assert any(m.startswith("HSE06 has no two-component form") and "PBE0" in m for m in msgs), msgs
