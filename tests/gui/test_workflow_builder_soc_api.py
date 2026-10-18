"""Spin-orbit coupling in the workflow builder (1.1.6): SOC is a per-step
setting of SP steps. The draft's ``steps[key].soc`` becomes that step's
``overrides`` in the spec (the CLI's keys: soc, soc_ecp, soc_basis, smear),
the CLI turns the spec into the plan, and the engine hands the keys to
opt2d12, which writes the SOC deck. Other steps refuse it with the CLI's
reason. The projected SP deck is byte for byte the deck ``mace opt2d12
--out-file PARENT.out --config-file SOC.json --yes`` writes from the same
parent."""
import json
import subprocess
import sys

import pytest

import guiharness as H
import submit_util as SU

sys.path.insert(0, str(H.REPO_ROOT))
from mace.gui import workflow_build as wb  # noqa: E402
from mace.workflow.spec import soc_refusal  # noqa: E402
from test_setup_soc_api import PBTE_CIF, SOC_READY  # noqa: E402

PY = sys.executable
SOC = {"soc": True, "soc_ecp": "inpsoc", "smear": 0.005}


def _inputs(tmp_path, kind="cif", names=("PbTe",)):
    d = tmp_path / "in"
    d.mkdir(exist_ok=True)
    files = []
    for n in names:
        f = d / f"{n}.{kind}"
        f.write_text(PBTE_CIF if kind == "cif" else "x\n")
        files.append(f)
    return wb.Inputs(kind, d, files, list(names))


def _steps(files):
    return json.loads(files["spec.json"]).get("steps", {})


# ---------------------------------------------------------------- draft -> spec (no CLI)
def test_soc_on_an_sp_step_is_its_overrides(tmp_path):
    inp = _inputs(tmp_path)
    files = wb.spec_files(inp, {"sequence": ["OPT", "SP", "BAND"], "steps": {
        "SP_2": {"soc": dict(SOC, soc_basis="pob-tzvp-rev2-soc"), "method": {"functional": "PBE0"}}}})
    assert _steps(files)["SP_2"]["overrides"] == {
        # no dispersion named: none written, the step keeps its parent's D3 (engine + opt2d12)
        "method_modifications": {"new_functional": "PBE0"},
        "soc": True, "soc_ecp": "inpsoc", "soc_basis": "pob-tzvp-rev2-soc", "smear": 0.005}
    # empty values are left out; SOC switched off writes nothing
    files = wb.spec_files(inp, {"sequence": ["OPT", "SP"], "steps": {
        "SP_2": {"soc": {"soc": True, "soc_ecp": None, "smear": ""}}}})
    assert _steps(files)["SP_2"]["overrides"] == {"soc": True}
    files = wb.spec_files(inp, {"sequence": ["OPT", "SP"], "steps": {"SP_2": {"soc": {"soc": False, "smear": 0.01}}}})
    assert "SP_2" not in _steps(files)


def test_soc_group_names_stutsc_refused_for_hf(tmp_path):
    """The step form's SOC group names the structures whose element the CLI refuses STUTSC for
    (soc_ecp.py's table and words), so the builder grays STUTSC out as Setup does."""
    pytest.importorskip("ase")      # reading a CIF's elements needs ase (its absence: the test below)
    from test_setup_soc_api import HFO_CIF
    import soc_ecp
    inp = _inputs(tmp_path, names=("PbTe",))
    (inp.folder / "HfO.cif").write_text(HFO_CIF)
    both = wb.Inputs("cif", inp.folder, inp.files + [inp.folder / "HfO.cif"], ["PbTe", "HfO"])
    g = wb.soc_group(both, ["OPT", "SP"], 1)
    assert g["stutsc"]["files"] == 2 and g["stutsc"]["all"] is False
    assert g["stutsc"]["unread"] == []
    assert [(x["symbol"], x["why"]) for x in g["stutsc"]["refused"]] == [("Hf", soc_ecp.STUTSC_NOT_THE_PUBLISHED_ECP[72])]
    hf = wb.Inputs("cif", inp.folder, [inp.folder / "HfO.cif"], ["HfO"])
    assert wb.soc_group(hf, ["OPT", "SP"], 1)["stutsc"]["all"] is True
    assert wb.soc_group(inp, ["OPT", "SP"], 1)["stutsc"]["refused"] == []
    assert wb.soc_group(inp, ["OPT", "SP"], 0)["stutsc"] is None        # OPT takes no SOC


def test_soc_group_says_stutsc_unchecked_without_ase(tmp_path, monkeypatch):
    """Without ase a CIF's elements cannot be read: the SOC group lists the file as unread with the
    words to install ase, never a silent "nothing refused" that would let STUTSC through for Hf."""
    from collections import OrderedDict
    from test_setup_soc_api import HFO_CIF
    from mace.gui import setup as gui_setup
    for m in ["ase"] + [m for m in sys.modules if m.startswith("ase.")]:
        monkeypatch.setitem(sys.modules, m, None)       # import ase (and ase.io) fails as where it is not installed
    monkeypatch.setattr(gui_setup, "_ELEMENTS_CACHE", OrderedDict())
    inp = _inputs(tmp_path, names=("PbTe",))
    (inp.folder / "HfO.cif").write_text(HFO_CIF)
    hf = wb.Inputs("cif", inp.folder, [inp.folder / "HfO.cif"], ["HfO"])
    st = wb.soc_group(hf, ["OPT", "SP"], 1)["stutsc"]
    assert st["refused"] == [] and st["all"] is False
    assert [u["file"] for u in st["unread"]] == ["HfO.cif"]
    assert "install ase" in st["unread"][0]["why"], st["unread"]


@pytest.mark.parametrize("seq,step", [(["OPT", "SP"], "OPT_1"), (["OPT", "SP", "OPT2"], "OPT2_3"),
                                      (["OPT", "SP", "FREQ"], "FREQ_3"), (["OPT", "SP", "BAND"], "BAND_3")])
def test_soc_on_a_step_that_is_not_a_single_point_is_refused(tmp_path, seq, step):
    with pytest.raises(wb.BuildError) as e:
        wb.spec_files(_inputs(tmp_path), {"sequence": seq, "steps": {step: {"soc": {"soc": True}}}})
    assert soc_refusal(seq[int(step.rsplit("_", 1)[1]) - 1]) in str(e.value)


def test_soc_values_and_restarts_are_refused_in_the_clis_words(tmp_path):
    inp = _inputs(tmp_path)
    with pytest.raises(wb.BuildError, match="smear must be a width in hartree"):
        wb.spec_files(inp, {"sequence": ["OPT", "SP"], "steps": {"SP_2": {"soc": {"soc": True, "smear": -1}}}})
    with pytest.raises(wb.BuildError, match="soc_ecp must be"):
        wb.spec_files(inp, {"sequence": ["OPT", "SP"], "steps": {"SP_2": {"soc": {"soc": True, "soc_ecp": "x"}}}})
    # a restart needs a parent known to be symmetry-free or two-component: by hand in the spec only
    with pytest.raises(wb.BuildError, match="from scratch"):
        wb.spec_files(inp, {"sequence": ["OPT", "SP"], "steps": {"SP_2": {"soc": {"soc": True,
                                                                                  "soc_guess": "guesspso"}}}})
    with pytest.raises(wb.BuildError, match="as they are"):
        wb.spec_files(_inputs(tmp_path, "d12", ("sp",)), {"sequence": ["SP", "BAND"],
                                                         "steps": {"SP_1": {"soc": {"soc": True}}}})


def test_a_cif_start_sp_gets_soc_through_its_conversion_settings(tmp_path):
    """The first step of a CIF start: SOC goes in with the form's patches
    (the CLI works out the conversion settings), not as a second override."""
    inp = _inputs(tmp_path)
    draft = {"sequence": ["SP", "BAND"], "steps": {"SP_1": {"soc": dict(SOC)}}}
    files = wb.spec_files(inp, draft, None, {"soc": True, "smear": 0.005})
    assert _steps(files)["SP_1"]["overrides"] == {"soc": True, "smear": 0.005}
    assert wb.soc_patches(draft["steps"]["SP_1"]) == SOC


def test_form_answers_take_soc_patches_on_sp_steps_only(tmp_path):
    inp = _inputs(tmp_path)
    wb.check_form_answers(inp, ["OPT", "SP"], 1, {"patches": dict(SOC), "answers": {"functional": "PBE0"}})
    with pytest.raises(wb.BuildError, match="inherits"):
        wb.check_form_answers(inp, ["OPT", "SP"], 1, {"patches": {"k_points": "8 8 8"}})
    with pytest.raises(wb.BuildError) as e:
        wb.check_form_answers(inp, ["OPT", "SP", "FREQ"], 2, {"patches": {"soc": True}})
    assert soc_refusal("FREQ") in str(e.value)
    with pytest.raises(wb.BuildError) as e:
        wb.check_form_answers(inp, ["OPT", "SP"], 0, {"patches": {"soc": True}})
    assert soc_refusal("OPT") in str(e.value)


# ---------------------------------------------------------------- on a real server, with the CLI
@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    s = SU.Served(tmp_path_factory.mktemp("wfsoc"), MACE_SITE_CONFIG="none")
    yield s
    s.stop()


def _task(srv, route, body, timeout=900):
    return srv.task_result(srv.req(f"api/{route}", {**body, "wait": 60}), timeout=timeout)


def _pbte(srv, folder):
    d = srv.root / folder
    d.mkdir(parents=True, exist_ok=True)
    (d / "PbTe.cif").write_text(PBTE_CIF)
    return d


def _draft(folder, steps, sequence=("OPT", "SP")):
    return {"inputs": {"type": "cif", "files": [f"{folder}/PbTe.cif"]}, "label": "soc",
            "sequence": list(sequence), "steps": steps, "execution": {}}


@H.needs_ase
def test_sp_step_form_previews_the_soc_deck(srv):
    _pbte(srv, "form")
    draft = _draft("form", {"OPT_1": {"answers": dict(SOC_READY)}, "SP_2": {"soc": dict(SOC)}})
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "SP_2", "patches": dict(SOC)})
    assert st == 200, r
    soc = r["soc"]
    assert soc["applies"] and soc["values"] == SOC and not soc["refusal"], soc
    assert "TWOCOMPON" in r["deck"] and "INPSOC" in r["deck"] and "SMEAR" in r["deck"]
    assert any(c.startswith("SOC deck:") for c in soc["changes"])
    # the OPT step's form: SOC does not apply there
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "OPT_1", "answers": dict(SOC_READY)})
    assert st == 200 and r["soc"]["applies"] is False, r


@H.needs_ase
def test_the_default_chain_shows_the_clis_soc_refusal(srv):
    """The workflow defaults (an internal basis set) cannot make a SOC deck:
    the step form says why, in the CLI's words."""
    _pbte(srv, "refuse")
    draft = _draft("refuse", {"SP_2": {"soc": {"soc": True}}})
    st, r = _task(srv, "workflow/step-form", {"draft": draft, "step": "SP_2", "patches": {"soc": True}})
    assert st == 200, r
    assert r["soc"]["refusal"] and "BASISSET" in r["soc"]["refusal"], r["soc"]
    assert any(p.get("field") == "soc" and p["severity"] == "error" for p in r.get("problems") or [])


@H.needs_ase
def test_check_names_the_steps_that_cannot_follow_a_soc_run(srv):
    _pbte(srv, "chk")
    draft = _draft("chk", {"OPT_1": {"answers": dict(SOC_READY)}, "SP_2": {"soc": dict(SOC)}},
                   ("OPT", "SP", "FREQ"))
    st, r = _task(srv, "workflow/defaults", {"draft": draft})
    assert st == 200, r
    assert "follows the SOC step SP" in " ".join(r["steps"]["FREQ_3"]["warnings"])
    assert r["steps"]["SP_2"]["settings"]["soc"] is True and r["steps"]["SP_2"]["settings"]["smear"] == 0.005


@H.needs_ase
def test_the_check_builds_a_soc_sp2_after_an_sp(srv):
    """OPT -> SP -> SP2 with SOC on SP2 (the release's SOC chain): the CLI
    builds the plan (a numbered repeat is configured as its base type)."""
    _pbte(srv, "sp2")
    draft = _draft("sp2", {"OPT_1": {"answers": dict(SOC_READY)}, "SP2_3": {"soc": dict(SOC)}},
                   ("OPT", "SP", "SP2"))
    st, r = _task(srv, "workflow/defaults", {"draft": draft})
    assert st == 200, r
    assert r["steps"]["SP2_3"]["settings"]["soc"] is True


@H.needs_ase
def test_the_check_refuses_orbitals_after_a_soc_sp(srv):
    """ORBITALS reading a SOC SP's wavefunction (SOC set the way the builder
    writes it: the step's overrides) is refused by the check."""
    _pbte(srv, "orb")
    draft = _draft("orb", {"OPT_1": {"answers": dict(SOC_READY)}, "SP_2": {"soc": dict(SOC)}},
                   ("OPT", "SP", "ORBITALS"))
    st, r = _task(srv, "workflow/defaults", {"draft": draft})
    assert st != 200 or not r.get("ok", True), r
    assert "ORBITALS_3" in json.dumps(r), r


@H.needs_ase
def test_projected_soc_sp_deck_is_the_clis_deck(srv, tmp_path):
    _pbte(srv, "par")
    draft = _draft("par", {"OPT_1": {"answers": dict(SOC_READY)}, "SP_2": {"soc": dict(SOC)}})
    st, r = _task(srv, "workflow/preview", {"root_gen": srv.req("api/info")[2]["root_gen"], "draft": draft})
    assert st == 200 and not r.get("code"), r.get("refused") or r.get("cli_log", "")[-2000:]
    sp = next(s for s in r["projected"] if s["step"] == "SP_2")
    assert "TWOCOMPON" in sp["deck_text"] and "INPSOC" in sp["deck_text"]
    opt_deck = r["first_step"][0]["deck_text"]
    assert "OPTGEOM" in opt_deck and "TWOCOMPON" not in opt_deck
    # the CLI on its own, from the same stand-in parent the projection made from that deck
    mat = r["first_step"][0]["material_id"]
    out = wb.stand_in(tmp_path / "parent", mat, opt_deck)
    cfg = tmp_path / "soc_sp.json"
    cfg.write_text(json.dumps({"calculation_type": "SP", **SOC}))
    outdir = tmp_path / "cli"
    outdir.mkdir()
    p = subprocess.run([PY, str(H.REPO_ROOT / "mace_cli"), "opt2d12", "--out-file", str(out), "--config-file",
                        str(cfg), "--yes", "--output-dir", str(outdir)], capture_output=True, text=True,
                       timeout=600, cwd=tmp_path)
    decks = sorted(outdir.glob("*.d12"))
    assert p.returncode == 0 and decks, p.stdout[-2000:] + p.stderr[-2000:]
    body = lambda t: t.split("\n", 1)[1]           # line 1 is the title: the deck's own name
    assert body(decks[0].read_text()) == body(sp["deck_text"])
