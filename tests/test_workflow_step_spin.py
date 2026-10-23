"""A later workflow step's spin polarization over its parent's (1.1.6).

Marcus: "Only spin polarization stays locked. Why? Of course by defualt things should be inherited but  (spelling-ok: quoted)
we should always be allowed to customize the workflow. Imagine a case where to get a tough geometry to
converge we dont add spin in the preopt, then for the opt2 and onward want spin".

The step's ``scf_modifications`` take ``"SPIN": true/false`` (spec, plan and opt2d12's config file); a step
without it keeps its parent's, so after a closed-shell OPT a spin-polarized OPT2 hands SPIN on to the SP after
it. A deck that turns spin on or off starts from a fresh SCF guess, never the parent's density matrix: the
GUESSP manual entry (p. 114) asks for the same symmetry, atoms, basis functions and shells and says nothing
of the spin, and CRYSTAL23's fort.20 reader stops on a matrix of the other spin treatment ("ERROR ****
READFG **** RESTART NOT POSSIBLE - OPEN-CLOSED SHELL INCOMPATIBILITY", measured on HPCC for a GUESSPNOSO deck
from a closed-shell parent). opt2d12 drops a carried GUESSP and says so in its log and the deck title, the
engine stages no fort.20 (guessp_chain.refusal), and the spec check says so for the step. A SOC step treats
spin itself (SPIN is refused there), and its GUESSPNOSO restart still needs a spin-polarized parent.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, find_data
from mace.workflow import guessp_chain
from mace.workflow.spec import SpecError, build_plan, load_spec, resolve_defaults, step_spin_states

MACE_CLI = REPO_ROOT / "mace_cli"
SPIN_PARENT = REPO_ROOT / "tests" / "data" / "opt_restart" / "tqc_agbr"
sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))
import d12_constants as C  # noqa: E402


@pytest.fixture(autouse=True)
def _plain_site(monkeypatch, tmp_path):
    monkeypatch.setenv("MACE_SITE_CONFIG", "none")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def _env():
    return {k: v for k, v in os.environ.items()
            if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}


def _opt2d12(tmp_path, parent, config, d12_text=None, out_text=None):
    d12, out = Path(str(parent) + ".d12"), Path(str(parent) + ".out")
    (tmp_path / "p.d12").write_text(d12_text if d12_text is not None else d12.read_text())
    (tmp_path / "p.out").write_text(out_text if out_text is not None else out.read_text(errors="replace"))
    (tmp_path / "c.json").write_text(json.dumps(config))
    r = subprocess.run([sys.executable, str(MACE_CLI), "opt2d12", "--out-file", "p.out", "--d12-file", "p.d12",
                        "--config-file", "c.json", "--non-interactive", "--output-dir", "o"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=300, env=_env(),
                       stdin=subprocess.DEVNULL)
    decks = list((tmp_path / "o").glob("*.d12")) if (tmp_path / "o").is_dir() else []
    return r.returncode, (decks[0].read_text() if decks else None), r.stdout + r.stderr


def _records(deck):
    return [line.strip() for line in deck.splitlines()]


# ------------------------------------------------------------------ the key
def test_spin_is_a_checked_scf_modification():
    assert C.check_scf_modifications({"SPIN": True, "SPINLOCK": 2}) == {"SPIN": True, "SPINLOCK": 2}
    assert C.check_scf_modifications({"SPIN": False}) == {"SPIN": False}
    for bad, msg in (({"SPIN": "yes"}, "SPIN is true"), ({"SPIN": 1}, "SPIN is true"),
                     ({"SPIN": False, "SPINLOCK": 2}, "SPINLOCK 2 needs a spin-polarized deck")):
        with pytest.raises(ValueError, match=msg):
            C.check_scf_modifications(bad)
    assert C.describe_scf_modifications({"SPIN": True}) == "spin-polarized (SPIN)"
    assert C.describe_scf_modifications({"SPIN": False}) == "closed shell (no SPIN)"


def test_switching_spin_sets_the_writers_flags_and_hartree_fock():
    opts = {"functional": "B3LYP", "spin_polarized": True, "spinlock": 2, "spinlock_explicit": True}
    C.apply_scf_modifications(opts, {"SPIN": False})
    assert opts["spin_polarized"] is False and opts["is_spin_polarized"] is False
    assert opts["spinlock"] == 0 and opts["spinlock_explicit"] is False
    hf = {"functional": "RHF", "method": "HF"}
    C.set_spin_polarized(hf, True)
    assert hf["functional"] == "UHF" and C.is_spin_polarized(hf)
    C.set_spin_polarized(hf, False)
    assert hf["functional"] == "RHF" and not C.is_spin_polarized(hf)
    assert C.spin_state_change({"spin_polarized": False}, {"spin_polarized": True}) == "on"
    assert C.spin_state_change({"functional": "UHF"}, {"functional": "RHF"}) == "off"
    assert C.spin_state_change({"spin_polarized": True}, {"spin_polarized": True}) is None


# ------------------------------------------------------------------ opt2d12's deck, both directions
def test_closed_shell_parent_to_a_spin_polarized_deck(tmp_path):
    """Diamond OPT that ran closed shell (RESTRICTED CLOSED SHELL in its .out): SPIN on for the child."""
    parent = find_data("OPT/1_dia_opt_BULK_OPTGEOM.out", "RESTRICTED CLOSED SHELL").with_suffix("")
    assert "SPIN" not in _records(Path(str(parent) + ".d12").read_text())
    code, deck, log = _opt2d12(tmp_path, parent, {"calculation_type": "OPT", "scf_modifications": {"SPIN": True}})
    assert code == 0 and deck, log[-3000:]
    recs = _records(deck)
    dft = recs[recs.index("DFT"):recs.index("ENDDFT")]
    assert "SPIN" in dft
    # MACE's spin-polarized default for a neutral, even-electron cell: SPIN alone (no SPINLOCK, no ATOMSPIN)
    assert "SPINLOCK" not in recs and "ATOMSPIN" not in recs and "GUESSP" not in recs
    assert recs[0].endswith(" - SPIN on (closed-shell parent), fresh SCF guess"), recs[0]
    assert "Spin polarization on (the parent ran closed shell): the SCF starts from a fresh guess" in log
    assert "OPEN-CLOSED SHELL INCOMPATIBILITY" in log


def test_spin_polarized_parent_to_a_closed_shell_deck_drops_spinlock_and_guessp(tmp_path):
    """The spin-polarized diamond OPT (SPIN, SPINLOCK 0 50), its deck also asking for GUESSP."""
    parent = find_data("OPT/1_dia_opt_rev1.out", "SPIN POLARIZED DFT").with_suffix("")
    text = Path(str(parent) + ".d12").read_text()
    assert "SPINLOCK" in _records(text)
    with_guessp = text.replace("\nSCFDIR\n", "\nGUESSP\nSCFDIR\n")
    code, deck, log = _opt2d12(tmp_path, parent, {"calculation_type": "SP", "scf_modifications": {"SPIN": False}},
                               d12_text=with_guessp)
    assert code == 0 and deck, log[-3000:]
    recs = _records(deck)
    assert "SPIN" not in recs and "SPINLOCK" not in recs and "UHF" not in recs
    assert "GUESSP" not in recs
    assert recs[0].endswith(" - SPIN off (spin-polarized parent), fresh SCF guess"), recs[0]
    assert "the parent deck's GUESSP is dropped" in log


def test_a_child_that_keeps_the_parents_spin_keeps_its_guessp_and_title(tmp_path):
    parent = find_data("OPT/1_dia_opt_rev1.out", "SPIN POLARIZED DFT").with_suffix("")
    text = Path(str(parent) + ".d12").read_text().replace("\nSCFDIR\n", "\nGUESSP\nSCFDIR\n")
    code, deck, log = _opt2d12(tmp_path, parent, {"calculation_type": "SP", "scf_modifications": {"SPIN": True}},
                               d12_text=text)
    assert code == 0 and deck, log[-3000:]
    recs = _records(deck)
    assert "SPIN" in recs and "GUESSP" in recs and "fresh SCF guess" not in recs[0]


def test_an_odd_electron_parent_cannot_go_closed_shell(tmp_path):
    out = SPIN_PARENT.with_suffix(".out").read_text(errors="replace")
    odd = re.sub(r"(N\. OF ELECTRONS PER CELL\s+)\d+", r"\g<1>223", out, count=1)
    if odd == out:
        odd = out.replace("\n", "\n N. OF ELECTRONS PER CELL   223  EXCHANGE PSEUDO OVP (F(G))  (T4) 10**   -7\n", 1)
    code, deck, log = _opt2d12(tmp_path, SPIN_PARENT, {"calculation_type": "SP",
                                                       "scf_modifications": {"SPIN": False}}, out_text=odd)
    assert code != 0 and deck is None
    assert "223 electrons per cell" in log and "keep SPIN" in log


def _uhf_deck(parent):
    """The parent's deck run as UHF: its DFT block replaced by the UHF record (the .out stays the real one)."""
    text = Path(str(parent) + ".d12").read_text()
    new = re.sub(r"(?ms)^DFT\n.*?^ENDDFT\n", "UHF\n", text, count=1)
    assert new != text and "UHF" in _records(new)
    return new


@pytest.mark.parametrize("config", [
    {"calculation_type": "SP", "method_modifications": {"new_functional": "PBE0"}},
    {"calculation_type": "SP", "functional": "B3LYP"}])
def test_a_functional_switch_off_a_uhf_parent_keeps_spin(tmp_path, config):
    """Review 1.1.6 E1: a UHF parent's spin is its method name, not the spin flag. A functional switch with no
    SPIN in the config wrote a closed-shell DFT deck titled "SPIN off", and refused an odd-electron parent.
    The Ag fcc OPT has 19 electrons per cell."""
    parent = find_data("OPT/Ag1_sym_CRYSTAL_OPT_symm_PBE-D3_POB-TZVP-REV2_opt_B3LYP-D3-D3_optimized.out",
                       "N. OF ELECTRONS PER CELL    19").with_suffix("")
    code, deck, log = _opt2d12(tmp_path, parent, config, d12_text=_uhf_deck(parent))
    assert code == 0 and deck, log[-3000:]
    recs = _records(deck)
    dft = recs[recs.index("DFT"):recs.index("ENDDFT")]
    assert "SPIN" in dft and "UHF" not in recs
    assert "SPIN off" not in recs[0] and "fresh SCF guess" not in recs[0], recs[0]
    assert "Spin polarization: kept from the parent (spin-polarized" in log
    assert "Spin polarization off" not in log and "closed-shell deck" not in log


def test_a_uhf_parent_still_goes_closed_shell_when_the_step_says_so(tmp_path):
    parent = find_data("OPT/1_dia_opt_rev1.out", "SPIN POLARIZED DFT").with_suffix("")
    code, deck, log = _opt2d12(tmp_path, parent, {"calculation_type": "SP",
                                                  "method_modifications": {"new_functional": "PBE0"},
                                                  "scf_modifications": {"SPIN": False}},
                               d12_text=_uhf_deck(parent))
    assert code == 0 and deck, log[-3000:]
    recs = _records(deck)
    assert "SPIN" not in recs and "UHF" not in recs and "DFT" in recs
    assert recs[0].endswith(" - SPIN off (spin-polarized parent), fresh SCF guess"), recs[0]
    assert "Spin polarization: kept from the parent" not in log


def test_spinlock_on_a_closed_shell_step_is_refused(tmp_path):
    """Review 1.1.6 E2: a SPINLOCK on a closed-shell step was dropped from the deck while the log said
    "SCF settings changed: SPINLOCK 2". It is refused unless the step turns SPIN on."""
    parent = find_data("OPT/1_dia_opt_BULK_OPTGEOM.out", "RESTRICTED CLOSED SHELL").with_suffix("")
    code, deck, log = _opt2d12(tmp_path, parent, {"calculation_type": "SP", "scf_modifications": {"SPINLOCK": 2}})
    assert code != 0 and deck is None, log[-3000:]
    assert "SPINLOCK 2 needs a spin-polarized deck" in log and '"SPIN": true' in log
    assert "SCF settings changed: SPINLOCK 2" not in log
    code, deck, log = _opt2d12(tmp_path, parent, {"calculation_type": "SP",
                                                  "scf_modifications": {"SPINLOCK": 2, "SPIN": True}})
    assert code == 0 and deck, log[-3000:]
    recs = _records(deck)
    assert "SPIN" in recs and recs[recs.index("SPINLOCK") + 1].split()[0] == "2"


def test_the_spec_refuses_a_spinlock_on_a_closed_shell_step(tmp_path):
    steps = {"OPT_1": {"overrides": {"is_spin_polarized": False}},
             "SP_3": {"overrides": {"scf_modifications": {"SPINLOCK": 2}}}}
    with pytest.raises(SpecError, match="SP_3 runs closed shell like OPT2_2"):
        resolve_defaults(load_spec(_cif_spec(tmp_path, steps)), tmp_path)
    steps["OPT2_2"] = {"overrides": {"scf_modifications": {"SPIN": True}}}
    w = resolve_defaults(load_spec(_cif_spec(tmp_path, steps)), tmp_path)
    assert w["SP_3"]["settings"]["scf_modifications"] == {"SPINLOCK": 2}


# ------------------------------------------------------------------ GUESSP across a spin change
def test_the_engine_never_restarts_across_a_spin_change():
    """guessp_chain.refusal, which the engine's opt-in GUESSP restart asks first: both directions."""
    head = "x\nCRYSTAL\n0 0 0\n227\n3.56\n1\n6 0.125 0.125 0.125\nEND\nBASISSET\nPOB-TZVP-REV2\n"
    tail = "B3LYP\nXLGRID\nENDDFT\nTOLDEE\n7\nSHRINK\n8 8\nSCFDIR\nEND\n"
    spin, closed = head + "DFT\nSPIN\n" + tail, head + "DFT\n" + tail
    assert guessp_chain.refusal(spin, spin) is None and guessp_chain.refusal(closed, closed) is None
    for src, tgt in ((closed, spin), (spin, closed)):
        why = guessp_chain.refusal(src, tgt)
        assert why and "OPEN-CLOSED SHELL INCOMPATIBILITY" in why and "fresh guess" in why


# ------------------------------------------------------------------ spec -> plan -> engine config
def _cif_spec(tmp_path, steps, seq=("OPT", "OPT2", "SP")):
    cif = find_data("CIFs/1_dia_opt_rev1.cif")
    shutil.copy(cif, tmp_path / "A.cif")
    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"version": 1, "inputs": {"type": "cif", "files": ["A.cif"]},
                             "sequence": list(seq), "steps": steps}))
    return p


MARCUS_CHAIN = {"OPT_1": {"overrides": {"is_spin_polarized": False}},
                "OPT2_2": {"overrides": {"scf_modifications": {"SPIN": True}}}}


def test_marcus_chain_round_trips_spec_plan_and_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = load_spec(_cif_spec(tmp_path, MARCUS_CHAIN))
    assert spec["steps"]["OPT2_2"]["overrides"]["scf_modifications"] == {"SPIN": True}
    # the closed-shell preoptimization, then spin from OPT2 on: SP inherits it
    assert step_spin_states(spec) == [False, True, True]
    _, wid, plan = build_plan(spec, tmp_path)
    assert plan["cif_conversion_config"]["is_spin_polarized"] is False
    assert plan["step_configurations"]["OPT2_2"]["scf_modifications"] == {"SPIN": True}
    assert "scf_modifications" not in plan["step_configurations"]["SP_3"]
    w = resolve_defaults(spec, tmp_path)
    assert w["OPT2_2"]["warnings"] == [
        "OPT2_2: spin-polarized (SPIN) after the closed-shell OPT_1: it starts from a fresh SCF guess, not "
        "OPT_1's wavefunction (CRYSTAL cannot restart across a spin change)"]
    assert w["SP_3"]["warnings"] == [] and w["OPT_1"]["warnings"] == []


def test_the_default_stays_spin_polarized_and_inherited(tmp_path):
    spec = load_spec(_cif_spec(tmp_path, {}))
    assert step_spin_states(spec) == [True, True, True]
    off = load_spec(_cif_spec(tmp_path, {"SP_3": {"overrides": {"scf_modifications": {"SPIN": False}}}}))
    assert step_spin_states(off) == [True, True, False]


def test_a_soc_step_keeps_its_own_spin_rules(tmp_path):
    seq = ("OPT", "SP")
    with pytest.raises(SpecError, match="two-component .SOC. step treats spin itself"):
        load_spec(_cif_spec(tmp_path, {"SP_2": {"overrides": {"soc": True, "scf_modifications": {"SPIN": True}}}},
                            seq))
    # GUESSPNOSO needs an unrestricted 1c parent (manual p. 173): a closed-shell parent is refused early
    steps = {"OPT_1": {"overrides": {"is_spin_polarized": False}},
             "SP_2": {"overrides": {"soc": True, "soc_guess": "guesspnoso"}}}
    with pytest.raises(SpecError, match="GUESSPNOSO starts from an unrestricted"):
        resolve_defaults(load_spec(_cif_spec(tmp_path, steps, seq)), tmp_path)


def test_the_planners_advanced_setup_asks_for_the_steps_spin(tmp_path, monkeypatch):
    from mace.workflow.planner import WorkflowPlanner
    planner = WorkflowPlanner(work_dir=tmp_path)
    for reply, want in (("y", {"SPIN": True}), ("n", {"SPIN": False}), ("", None)):
        monkeypatch.setattr("builtins.input", lambda prompt="", r=reply: r)
        cfg = {"scf_modifications": {"SPINLOCK": 2}} if reply == "n" else {}
        planner._ask_step_spin(cfg)
        assert cfg.get("scf_modifications") == want, (reply, cfg)


# ------------------------------------------------------------------ the real CLI path
def _ok(p):
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-4000:]
    return p


def test_mace_workflow_spec_writes_the_chain_through_the_engine(tmp_path):
    """mace workflow --spec (non-spin OPT -> spin OPT2 -> SP) on a corpus CIF, launched with fake SLURM, then
    the engine writes OPT2 and SP from stand-in outputs (the GUI preview's projection)."""
    pytest.importorskip("ase", reason="the workflow starts from a CIF, which mace convert reads with ase")
    from fake_slurm import FakeSlurm
    from test_array_default_path_golden import _mace
    fake = FakeSlurm(tmp_path / "fs")
    env = dict(fake.env, HOME=str(tmp_path / "home"), USER="tester", XDG_CACHE_HOME=str(tmp_path / "cache"),
               MACE_SITE_CONFIG="none")
    env.pop("MACE_HOME", None)
    env.pop("PYTHONPATH", None)
    (tmp_path / "home").mkdir(exist_ok=True)
    root = tmp_path / "copy"
    folder = root / "cifs"
    folder.mkdir(parents=True)
    (root / ".mace_preview_mirror").write_text("preview\n")
    _cif_spec(folder, MARCUS_CHAIN)
    check = json.loads(_ok(_mace(folder, env, "workflow", "--spec", "spec.json", "--json")).stdout)
    assert check["steps"]["OPT2_2"]["settings"]["scf_modifications"] == {"SPIN": True}
    assert any("fresh SCF guess" in w for w in check["steps"]["OPT2_2"]["warnings"])
    w = json.loads(_ok(_mace(folder, env, "workflow", "--spec", "spec.json", "--write-plan", "--json")).stdout)
    _ok(_mace(folder, env, *w["launch"][1:]))
    first = [p for p in folder.rglob("*.d12") if "opt" in p.name.lower()]
    assert first and "SPIN" not in _records(first[0].read_text())
    p = _ok(subprocess.run([sys.executable, "-m", "mace.workflow.projection", "--folder", str(folder),
                            "--workflow-id", w["workflow_id"], "--json"], cwd=str(folder),
                           env=dict(env, PYTHONPATH=str(REPO_ROOT)), capture_output=True, text=True, timeout=600))
    steps = {s["step"]: s for s in json.loads(p.stdout)["steps"]}
    opt2, sp = (Path(steps[k]["input_file"]).read_text() for k in ("OPT2_2", "SP_3"))
    assert "SPIN" in _records(opt2) and "OPTGEOM" in _records(opt2)
    assert opt2.splitlines()[0].endswith(" - SPIN on (closed-shell parent), fresh SCF guess")
    # SP inherits OPT2's spin, from a spin-polarized parent: no note
    assert "SPIN" in _records(sp) and "fresh SCF guess" not in sp.splitlines()[0]
