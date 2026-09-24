"""Engine-generated SP/FREQ/OPT2 decks must keep the parent's method.

generate_numbered_calculation used to re-derive the parent's functional with a
substring match against the functional list, first hit wins, and wrote that
name into the step config, where it overrode the parent deck:
  - PBE0, PBESOL, PBESOL0, LC-wPBE and PBEH3C all became PBE (-D3)
  - every UHF parent became RHF, losing UHF and its SPINLOCK
  - a parent with no detectable functional (every HSEsol-3c deck) got no
    config at all, so "Use these exact settings? y" kept the parent's OPT type,
    the SP/FREQ step wrote an OPT deck and then failed to find its output.
The planless SP path also answered "2" (DFT) at "Select method type", turning
an HF parent into HSE06-D3.

The engine now passes no functional: CRYSTALOptToD12 reads the method from the
parent .d12, and only the plan's method_modifications may override it.
"""
import json
import shutil
from pathlib import Path

import pytest

from conftest import TEST_DATA
from mace.workflow.engine import WorkflowEngine

WF_ID = "workflow_keepmethod"

PLAN_FREQ_STEP = {
    "calculation_type": "FREQ",
    "inherit_base_settings": True,
    "frequency_settings": {"mode": "GAMMA", "numderiv": 2, "intensities": True,
                           "ir_method": "CPHF", "raman": True},
}


class _FakeDB:
    def __init__(self, calc):
        self.calc = calc

    def get_calculation(self, cid):
        return self.calc

    def get_calculations_by_status(self, *a, **k):
        return []

    def __getattr__(self, name):
        raise RuntimeError(f"engine went past deck generation: db.{name}")


class _Stop(Exception):
    pass


def _engine(tmp_path, d12_text, out_text, plan_steps=None):
    src = tmp_path / "parent"
    src.mkdir()
    (src / "parent.d12").write_text(d12_text)
    (src / "parent.out").write_text(out_text)
    settings = None
    if plan_steps is not None:
        cfg = tmp_path / "workflow_configs"
        cfg.mkdir()
        (cfg / f"workflow_plan_{WF_ID.replace('workflow_', '')}.json").write_text(
            json.dumps({"workflow_sequence": ["OPT", "SP", "FREQ", "OPT2"],
                        "step_configurations": plan_steps}))
        settings = json.dumps({"workflow_id": WF_ID, "workflow_step": 1})
    calc = {"calc_id": "c1", "material_id": "mat", "status": "completed",
            "calc_type": "OPT", "output_file": str(src / "parent.out"),
            "input_file": str(src / "parent.d12"), "settings_json": settings}
    eng = WorkflowEngine(db_path=str(tmp_path / "m.db"), base_work_dir=str(tmp_path),
                         auto_submit=False)
    eng.db = _FakeDB(calc)
    return eng


def _capture(eng, run_script=False):
    """Record what the engine hands CRYSTALOptToD12, optionally run it for
    real, then stop the engine before it moves files or submits."""
    seen = {}
    real = eng.run_script_in_isolated_directory

    def intercept(script_path, work_dir, args=None, input_data=None):
        seen["args"] = list(args or [])
        seen["stdin"] = input_data
        cfgs = [a for i, a in enumerate(seen["args"])
                if i and seen["args"][i - 1] == "--config-file"]
        seen["config"] = json.loads(Path(cfgs[-1]).read_text()) if cfgs else None
        if run_script:
            before = {p.name for p in Path(work_dir).glob("*.d12")}
            ok, so, se = real(script_path, work_dir, args, input_data)
            gen = sorted(p for p in Path(work_dir).glob("*.d12") if p.name not in before)
            seen["ok"], seen["log"] = ok, so + se
            seen["decks"] = {p.name: p.read_text() for p in gen}
        raise _Stop()

    eng.run_script_in_isolated_directory = intercept
    return seen


def _numbered(eng, target, run_script=False):
    seen = _capture(eng, run_script)
    with pytest.raises(_Stop):
        eng.generate_numbered_calculation("c1", target)
    return seen


DECK = "dia_synth\nCRYSTAL\n0 0 0\n227\n3.56\n1\n6 0 0 0\nEND\n{ham}\nSHRINK\n12 24\nEND\n"


# ---------------------------------------------------------------------------
# What the engine passes (no corpus needed; the script is not run)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ham", ["DFT\nSPIN\nPBE0\nXLGRID\nENDDFT", "DFT\nSPIN\nPBESOL0\nENDDFT",
                                 "DFT\nLC-wPBE\nENDDFT", "UHF", "DFT\nSPIN\nHSESOL3C\nENDDFT"])
@pytest.mark.parametrize("target,base,stdin", [("SP", "SP", "y\n"),
                                               ("FREQ", "FREQ", "y\n" + "\n" * 17),
                                               ("OPT2", "OPT", "y\n")])
def test_no_plan_names_only_the_target_type(tmp_path, ham, target, base, stdin):
    eng = _engine(tmp_path, DECK.format(ham=ham), "dummy output\n")
    seen = _numbered(eng, target)
    assert seen["config"] == {"calculation_type": base}
    assert seen["stdin"] == stdin


def test_plan_functional_switch_keeps_the_plans_freq_settings(tmp_path):
    """A plan that changes the FREQ functional used to lose frequency_settings."""
    step = dict(PLAN_FREQ_STEP, method_modifications={"new_functional": "B3LYP",
                                                      "use_dispersion": True})
    eng = _engine(tmp_path, DECK.format(ham="DFT\nSPIN\nPBE0\nENDDFT"), "dummy\n",
                  {"FREQ_3": step})
    cfg = _numbered(eng, "FREQ")["config"]
    assert cfg["functional"] == "B3LYP-D3" and cfg["dispersion"] is True
    assert cfg["freq_settings"]["numderiv"] == 2
    assert cfg["frequency_settings"]["raman"] is True


def test_plan_without_method_change_passes_no_functional(tmp_path):
    eng = _engine(tmp_path, DECK.format(ham="DFT\nSPIN\nPBE0\nENDDFT"), "dummy\n",
                  {"SP_2": {"calculation_type": "SP", "inherit_settings": True},
                   "FREQ_3": PLAN_FREQ_STEP})
    for target in ("SP", "FREQ"):
        cfg = _numbered(eng, target)["config"]
        assert "functional" not in cfg, cfg
    assert cfg["freq_settings"]["ir_method"] == "CPHF"


def test_plan_opt2_settings_reach_the_config(tmp_path):
    opt = {"TOLDEG": 3e-05, "TOLDEX": 0.00012, "TOLDEE": 7, "MAXCYCLE": 800}
    eng = _engine(tmp_path, DECK.format(ham="UHF"), "dummy\n",
                  {"OPT2_4": {"calculation_type": "OPT", "optimization_type": "FULLOPTG",
                              "optimization_settings": opt, "inherit_settings": True}})
    cfg = _numbered(eng, "OPT2")["config"]
    assert cfg["optimization_settings"] == opt
    assert cfg["optimization_type"] == "FULLOPTG"
    assert "functional" not in cfg


def test_planless_sp_leaves_the_method_type_at_its_default(tmp_path, monkeypatch):
    monkeypatch.setenv("MACE_PLANLESS_PROGRESSION", "1")
    eng = _engine(tmp_path, DECK.format(ham="UHF"), "dummy\n")
    seen = _capture(eng)
    with pytest.raises(_Stop):
        eng.generate_sp_from_opt("c1")
    # "n" (modify settings), then only blank answers: every prompt takes
    # its parent-based default. Enough of them that no parent runs out.
    assert seen["stdin"].startswith("n\n")
    assert set(seen["stdin"][2:]) == {"\n"} and len(seen["stdin"]) - 2 >= 30


# ---------------------------------------------------------------------------
# Real decks, real script (corpus-gated)
# ---------------------------------------------------------------------------
DIA = TEST_DATA / "OPT" / "1_dia_opt_rev1"
SLAB = TEST_DATA / "OPT" / "4LG_2x2_AA_opt_HSESOL3C_optimized"


def _need(stem):
    if not stem.with_suffix(".d12").exists() or not stem.with_suffix(".out").exists():
        pytest.skip("test/ corpus not present (gitignored, ~12GB)")


def _dia_with(ham_lines):
    """1_dia_opt_rev1 with its B3LYP-D3 block swapped for ``ham_lines``; the
    D3 markers are removed from its .out so nothing re-imports dispersion."""
    lines = DIA.with_suffix(".d12").read_text().splitlines()
    i, j = lines.index("DFT"), lines.index("ENDDFT")
    lines[0] = "dia_synth"
    d12 = "\n".join(lines[:i] + ham_lines + lines[j + 1:]) + "\n"
    out = (DIA.with_suffix(".out").read_text().replace("DFT-D3", "DFT-Dx")
           .replace("GRIMME D3", "GRIMME Dx").replace("D3 DISPERSION", "Dx DISPERSION"))
    return d12, out


def _only_deck(seen):
    assert seen["ok"], seen["log"][-2000:]
    assert len(seen["decks"]) == 1, list(seen["decks"])
    return next(iter(seen["decks"].items()))


@pytest.mark.parametrize("fx", ["PBE0", "PBESOL0", "LC-wPBE", "PBE0-D3"])
@pytest.mark.parametrize("target", ["SP", "FREQ"])
def test_real_script_keeps_the_parent_functional(tmp_path, fx, target):
    _need(DIA)
    d12, out = _dia_with(["DFT", "SPIN", fx, "XLGRID", "ENDDFT"])
    name, deck = _only_deck(_numbered(_engine(tmp_path, d12, out), target, True))
    body = deck.splitlines()
    assert body[body.index("DFT") + 2] == fx, deck
    assert f"_{fx}_" in name


def test_real_script_keeps_uhf(tmp_path):
    _need(DIA)
    d12, out = _dia_with(["UHF"])
    name, deck = _only_deck(_numbered(_engine(tmp_path, d12, out), "SP", True))
    body = deck.splitlines()
    assert "UHF" in body and "RHF" not in body and "DFT" not in body, deck
    assert "_UHF_" in name


def test_real_planless_sp_keeps_hartree_fock(tmp_path, monkeypatch):
    _need(DIA)
    monkeypatch.setenv("MACE_PLANLESS_PROGRESSION", "1")
    d12, out = _dia_with(["UHF"])
    eng = _engine(tmp_path, d12, out)
    seen = _capture(eng, run_script=True)
    with pytest.raises(_Stop):
        eng.generate_sp_from_opt("c1")
    name, deck = _only_deck(seen)
    body = deck.splitlines()
    assert "UHF" in body and "DFT" not in body and "HSE06-D3" not in body, deck


@pytest.mark.parametrize("target,block", [("SP", None), ("FREQ", "FREQCALC")])
def test_real_hsesol3c_parent_gets_an_sp_or_freq_deck(tmp_path, target, block):
    """Nothing is detected for an HSEsol-3c deck, which used to mean no config
    and an OPT deck in place of the SP/FREQ one."""
    _need(SLAB)
    eng = _engine(tmp_path, SLAB.with_suffix(".d12").read_text(),
                  SLAB.with_suffix(".out").read_text())
    name, deck = _only_deck(_numbered(eng, target, True))
    body = deck.splitlines()
    assert "HSESOL3C" in body, deck
    assert "OPTGEOM" not in body
    assert f"_{target.lower()}_HSESOL3C_" in name
    if block:
        assert block in body
