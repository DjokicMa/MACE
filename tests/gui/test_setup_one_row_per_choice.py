"""Calculation Setup shows one row for each thing a user chooses (1.1.6).

The CLI asks the method (Hartree-Fock or DFT), then the functional's family, then the functional; the
basis set's type, then the set (and, for a follow-up, whether the parent's is kept). The functional
alone says the first three and the basis set alone says the others, so the form has one Functional row
and one Basis set row, and works the other answers out: the decks are the ones the CLI writes for the same
choices. Every menu option carries the number the CLI gives it, and a basis directory is shown by its
name, not its path.
"""
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

import guiharness as H

pytest.importorskip("numpy")
pytest.importorskip("ase")

sys.path.insert(0, str(H.REPO_ROOT))
from mace.gui import setup as S  # noqa: E402

DERIVED = {"method", "basis_type", "basis_keep"}


def _cache():
    store = {}
    return (lambda k: store.get(json.dumps(k, default=str)),
            lambda k, v, n: store.__setitem__(json.dumps(k, default=str), v), store)


def _run(work, path, rel, mode, calc, answers=None, flags=None, extra=None, cache=None):
    get, put, _ = cache or _cache()
    body = {"answers": answers or {}, "patches": {}, "preview": True, "json_text": None, **(extra or {})}
    if flags:
        body["flags"] = flags
    return S.run_setup(S.Request(body), mode, calc, work, path, rel, S.source_key(mode, path), get, put, None,
                       schema_info=S.schema())


@pytest.fixture(scope="module")
def mgo(tmp_path_factory):
    d = tmp_path_factory.mktemp("onerow")
    shutil.copytree(H.EXAMPLES / "structures", d / "structures")
    shutil.copytree(H.EXAMPLES / "outputs", d / "outputs")
    (d / "outputs" / "Si_sp.f9").write_text("placeholder\n")
    return d


def _form(r):
    return {f["id"]: f for f in r["form"]}


def test_no_row_for_what_another_row_says(tmp_path, mgo):
    r = _run(tmp_path / "a", mgo / "structures" / "MgO.cif", "structures/MgO.cif", "cif2d12", "OPT")
    form = _form(r)
    assert not DERIVED & set(form), sorted(form)
    assert {"functional", "basis_set"} <= set(form)
    assert not any("category" in f["label"].lower() or "category" in f["id"].lower() for f in r["form"])
    # FREQ asks the functional's family a second time: still no row for it
    r = _run(tmp_path / "b", mgo / "structures" / "MgO.cif", "structures/MgO.cif", "cif2d12", "FREQ")
    assert not any("category" in f["id"].lower() or "category" in f.get("question", "").lower() for f in r["form"])
    # a follow-up has the basis row too, at the parent's
    r = _run(tmp_path / "c", mgo / "outputs" / "Si_sp.out", "outputs/Si_sp.out", "opt2d12", "SP")
    form = _form(r)
    assert not DERIVED & set(form) and form["basis_set"]["value"] == S.KEEP_BASIS
    assert form["basis_set"]["options"][0]["head"] == S.KEEP_BASIS


def test_the_functional_row_says_the_method_too(tmp_path, mgo):
    cif = (mgo / "structures" / "MgO.cif", "structures/MgO.cif", "cif2d12", "OPT")
    old = _run(tmp_path / "old", *cif, answers={"method": "Hartree-Fock (HF)"})
    new = _run(tmp_path / "new", *cif, answers={"functional": "RHF"})
    assert json.loads(new["json_text"])["method_type"] == "HF" and new["deck"] == old["deck"]
    form = _form(new)
    assert form["functional"]["value"] == "RHF"
    assert "dispersion" not in form and "dft_grid" not in form              # a Hartree-Fock deck has neither
    hf = [o for o in form["functional"]["options"] if o.get("family") == "hf"]
    assert [o["head"] for o in hf][:2] == ["RHF", "UHF"] and [o["num"] for o in hf][:2] == ["1", "2"]
    # DFT back again
    back = _run(tmp_path / "back", *cif, answers={"functional": "PBE0"})
    assert json.loads(back["json_text"])["method_type"] == "DFT" and "dispersion" in _form(back)
    # a 3c method sets its own basis set: no basis row, no D3 row
    c3 = _run(tmp_path / "c3", *cif, answers={"functional": "HSESOL3C"})
    assert not {"basis_set", "dispersion", "dft_grid"} & set(_form(c3))
    assert "SOLDEF2MSVP" in json.dumps(json.loads(c3["json_text"]))


def test_the_basis_set_row_says_the_type_too(tmp_path, mgo):
    cif = (mgo / "structures" / "MgO.cif", "structures/MgO.cif", "cif2d12", "OPT")
    old = _run(tmp_path / "old", *cif, answers={"basis_type": "EXTERNAL", "basis_set": "TZVP-REV2"})
    new = _run(tmp_path / "new", *cif, answers={"basis_set": "TZVP-REV2"})
    assert json.loads(new["json_text"])["basis_set_type"] == "EXTERNAL" and new["deck"] == old["deck"]
    row = _form(new)["basis_set"]
    assert row["value"] == "TZVP-REV2"
    groups = {o["group"] for o in row["options"]}
    assert len(groups) == 2 and {"DZVP-REV2", "TZVP-REV2"} <= {o["head"] for o in row["options"]}
    internal = _run(tmp_path / "int", *cif, answers={"basis_set": "POB-DZVP-REV2"})
    assert json.loads(internal["json_text"])["basis_set_type"] == "INTERNAL"
    # a follow-up: keep the parent's, or pick another (the type and "keep?" questions are answered for you)
    si = (mgo / "outputs" / "Si_sp.out", "outputs/Si_sp.out", "opt2d12", "SP")
    keep = _run(tmp_path / "keep", *si)
    other = _run(tmp_path / "other", *si, answers={"basis_set": "POB-DZVP-REV2"})
    assert json.loads(keep["json_text"])["use_original_external_basis"] is True
    got = json.loads(other["json_text"])
    assert (got["basis_set"], got["basis_set_type"], got["use_original_external_basis"]) == ("POB-DZVP-REV2", "INTERNAL", False)
    assert "POB-DZVP-REV2" in other["deck"] and "EXTERNAL" not in other["deck"]
    assert _form(other)["basis_set"]["changed"] and not _form(keep)["basis_set"]["changed"]


def test_menu_options_carry_the_clis_numbers(tmp_path, mgo):
    r = _run(tmp_path / "n", mgo / "structures" / "MgO.cif", "structures/MgO.cif", "cif2d12", "OPT")
    form = _form(r)
    tol = form["tolerance"]["options"]
    assert [o["num"] for o in tol][:3] == ["1", "2", "3"]
    # the functional is numbered as the CLI lists it inside its family
    fn = form["functional"]["options"]
    hyb = [o for o in fn if o["group"] == "Hybrid Functionals" or "Hybrid" in o["group"]]
    assert hyb and hyb[0]["num"] == "1" and [o["num"] for o in hyb] == [str(i) for i in range(1, len(hyb) + 1)]
    assert all("num" in o for o in form["symmetry"]["options"])


def test_a_basis_directory_is_shown_by_its_name(tmp_path, capsys, monkeypatch):
    sys.path.insert(0, str(H.REPO_ROOT / "Crystal_d12"))
    import d12_constants

    answers = iter(["1", "2"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    import menu_nav
    monkeypatch.setattr(menu_nav, "_REAL_INPUT", lambda prompt="": next(answers))
    config = d12_constants.select_basis_set([6, 8])
    out = capsys.readouterr().out
    menu = out.split("Select external basis set")[1]
    lines = [ln for ln in menu.splitlines() if re.match(r"^\d: ", ln)]
    assert [ln.split(" - ")[0] for ln in lines] == ["1: DZVP-REV2", "2: TZVP-REV2", "3: Custom path", "4: TZVP-REV2 (2022 update)"]
    assert lines[1].endswith("/full.basis.triplezeta/") and type(config["basis_set"]) is str
    assert config["basis_set"].endswith("/full.basis.triplezeta/") and config["basis_set_path"] == config["basis_set"]


def test_a_long_basis_path_reads_as_its_family_or_its_last_folders(tmp_path):
    sys.path.insert(0, str(H.REPO_ROOT / "Crystal_d12"))
    import basis_library as BL
    assert BL.display_name("full.basis.triplezeta") == "TZVP-REV2"
    assert BL.display_name(BL.BASIS_SETS_DIR + "/full.basis.doublezeta/") == "DZVP-REV2"
    mine = "/home/someone/a/very/long/project/folder/of/theirs/basis_sets/my_pob_tzvp/"
    assert BL.display_name(mine) == ".../basis_sets/my_pob_tzvp/"
    assert S._short_value(mine) == ".../basis_sets/my_pob_tzvp/" and S._short_value("PBE0") == "PBE0"


def test_static_kpath_is_a_switch_of_the_band_command(tmp_path, mgo):
    src = (mgo / "outputs" / "Si_sp.out", "outputs/Si_sp.out", "opt2d3", "BAND")
    default = _run(tmp_path / "d", *src)
    static = _run(tmp_path / "s", *src, extra={"static_kpath": True})
    assert "--static-kpath" not in default["command"] and "--static-kpath" in static["command"]
    assert "SeeKPath" in default["deck"].splitlines()[1]
    assert json.loads(default["json_text"])["configuration"]["seekpath_full"] is True
    assert static["deck"] and static["deck"].startswith("BAND")      # with seekpath installed the path is SeeK-path's either way
    assert S.schema()["static_kpath"]["switch"] == "--static-kpath"


def test_the_path_format_belongs_to_the_automatic_path(tmp_path, mgo):
    src = (mgo / "outputs" / "Si_sp.out", "outputs/Si_sp.out", "opt2d3", "BAND")
    auto = _run(tmp_path / "a", *src)
    assert "band_format" in _form(auto) and _form(auto)["band_format"]["value"].startswith("SeeK-path")
    tmpl = _run(tmp_path / "t", *src, answers={"band_path": "Template selection"})
    assert "band_format" not in _form(tmpl)
