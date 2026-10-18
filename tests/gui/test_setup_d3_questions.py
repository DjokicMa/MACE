"""The opt2d3 form of the properties runs with no named fields (TRANSPORT,
CHARGE, POTENTIAL, CHARGE+POTENTIAL): its rows are the CLI's own questions.

They were listed only after a first answer, so Calculation Setup and the
workflow builder showed these runs with no row to change. CHARGE+POTENTIAL
asks the same question in its charge and its potential block ("Select type
(1-2) [1]:", "Number of grid points per direction [100]:"); each one is its
own row, answered only where it is asked.
"""
import json
import pickle
import re
import shutil
from pathlib import Path

import pytest

from mace.gui import setup as S

CORPUS = Path(__file__).resolve().parents[2] / "test" / "SP"
PARENT = "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized"


@pytest.fixture
def parent(tmp_path):
    src = CORPUS / f"{PARENT}.out"
    if not src.exists():
        pytest.skip("test/ data corpus not present")
    d = tmp_path / "parent"
    d.mkdir()
    for ext in (".out", ".d12"):
        shutil.copy(CORPUS / f"{PARENT}{ext}", d / f"{PARENT}{ext}")
    (d / f"{PARENT}.f9").write_text("wavefunction placeholder\n")
    return d / f"{PARENT}.out"


def _setup(tmp_path, parent, calc, answers=None):
    cache = {}
    return S.run_setup(S.Request({"answers": answers or {}, "preview": False}), "opt2d3", calc,
                       tmp_path / f"w_{calc}_{len(answers or {})}", parent, parent.name, ("t", calc),
                       lambda k: cache.get(pickle.dumps(k)),
                       lambda k, v, n: cache.__setitem__(pickle.dumps(k), v))


def test_transport_lists_its_questions_before_any_answer(tmp_path, parent):
    r = _setup(tmp_path, parent, "TRANSPORT")
    rows = {f["label"]: f for f in r["form"]}
    assert rows["Minimum temperature (K) [100]"]["value"] == "100", sorted(rows)
    assert rows["Relaxation time (fs) [10]"]["kind"] == "value"
    assert not any(f["changed"] for f in r["form"])
    assert all(f["section"] == "Other questions" for f in r["form"])
    # a menu is named by its heading and shows its options' names; a yes/no shows Yes or No
    mu = rows["Chemical potential options"]
    assert mu["kind"] == "menu" and mu["value"] == mu["default"] == "Relative to Fermi energy (automatic)"
    assert rows["Use smearing for metallic systems? [y/N]"]["value"] == "No"


def test_transport_answers_by_its_rows(tmp_path, parent):
    ids = {f["label"]: f["id"] for f in _setup(tmp_path, parent, "TRANSPORT")["form"]}
    r = _setup(tmp_path, parent, "TRANSPORT", {ids["Minimum temperature (K) [100]"]: "200",
                                               ids["Maximum temperature (K) [800]"]: "600"})
    cfg = json.loads(r["json_text"])["configuration"]
    assert cfg["temperature_range"][:2] == [200.0, 600.0]
    rows = {f["id"]: f for f in r["form"]}
    assert rows[ids["Minimum temperature (K) [100]"]]["changed"]
    assert not rows[ids["Temperature step (K) [50]"]]["changed"]


def test_charge_potential_repeats_are_rows_of_their_own(tmp_path, parent):
    form = _setup(tmp_path, parent, "CHARGE+POTENTIAL")["form"]
    types = [f for f in form if f["question"].startswith("Select type (1-2)")]
    grids = [f for f in form if f["question"].startswith("Number of grid points")]
    assert len(types) == 2 and len(grids) == 2, [f["id"] for f in form]
    assert len({f["id"] for f in form}) == len(form)
    # the label says which block asks it
    assert [f["label"] for f in types] == ["Charge density type", "Potential calculation type"]
    assert [f["label"] for f in grids] == ["Number of grid points per direction [100] (charge density)",
                                           "Number of grid points per direction [100] (electrostatic potential)"]
    assert [f["value"] for f in types] == ["ECH3", "POT3"]
    assert len(types[1]["options"]) == 2


def test_charge_potential_answers_only_the_block_asked(tmp_path, parent):
    form = _setup(tmp_path, parent, "CHARGE+POTENTIAL")["form"]
    types = [f["id"] for f in form if f["question"].startswith("Select type (1-2)")]
    grids = [f["id"] for f in form if f["question"].startswith("Number of grid points")]
    # the potential at the atoms (POTC) instead of its 3D grid; the charge grid finer
    r = _setup(tmp_path, parent, "CHARGE+POTENTIAL", {types[1]: "2", grids[0]: "150"})
    cfg = json.loads(r["json_text"])["configuration"]
    assert cfg["charge_config"]["type"] == "ECH3" and cfg["charge_config"]["n_points"] == 150
    assert cfg["potential_config"]["type"] == "POTC"
    changed = {f["id"] for f in r["form"] if f["changed"]}
    assert changed == {types[1], grids[0]}


def test_a_menu_row_answered_by_its_option_name(tmp_path, parent):
    """The form's select sends the option's name ("POTC"), not the CLI's number: it is typed as the
    number of its menu line (opt2d3 refuses a potential type that is not 1 or 2)."""
    form = _setup(tmp_path, parent, "CHARGE+POTENTIAL")["form"]
    types = [f["id"] for f in form if f["question"].startswith("Select type (1-2)")]
    r = _setup(tmp_path, parent, "CHARGE+POTENTIAL", {types[0]: "ECH3", types[1]: "POTC"})
    cfg = json.loads(r["json_text"])["configuration"]
    assert cfg["charge_config"]["type"] == "ECH3" and cfg["potential_config"]["type"] == "POTC"


def test_a_question_asked_once_keeps_its_plain_id_and_rule():
    """Unique questions keep the id and the answer rule they always had."""
    rules = S.answer_rules("opt2d3", {"q:Minimum temperature (K) [100]:": "200"})
    assert rules[0] == {"match": re.escape("Minimum temperature (K) [100]:"), "answer": "200"}
