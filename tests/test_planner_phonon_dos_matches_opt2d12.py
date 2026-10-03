"""The planner's phonon DOS and "both" steps write opt2d12's deck.

Planner -> Advanced FREQ -> phonon calculations -> full dispersion asks for an
output type. "2: Phonon density of states" and "3: Both band structure and
DOS" stored only a label ("dos", "both") that nothing reads, so the deck was
FREQCALC NOINTENS DISPERSION ENDFREQ: no PDOS, and for "both" no BANDS either.
On default answers they now write what opt2d12 writes:

- DOS: "7: Phonon density of states" (PDOS 2000 cm-1, 200 bins, projected);
- both: "8: Custom settings" in phonon dispersion mode with no INTERPHESS
  (the planner's own default), which is template 6's automatic BANDS path
  followed by template 7's PDOS.
"""
import pytest

import test_freq_child_intensities_follow_the_mode as phonon
import test_numbered_calc_keeps_parent_method as base
import test_opt2d12_prompt_defaults as prompts
from mace.workflow.planner import WorkflowPlanner

OPT2D12_DOS = {"Select template": "7"}
OPT2D12_BOTH = {"Select template": "8", "Select mode (1-2)": "2",
                "Fourier interpolation (INTERPHESS)": "n"}


def _workflow_step(tmp_path, monkeypatch, output_type):
    planner = WorkflowPlanner(work_dir=tmp_path / "plan")

    def reply(prompt=""):
        if "customization level" in prompt:
            return "2"
        if "Select mode [1]" in prompt:
            return "2"          # phonon calculations
        if "Select output type" in prompt:
            return output_type
        return ""

    monkeypatch.setattr("builtins.input", reply)
    return planner.configure_frequency_step("FREQ", 3)


def _both_routes(tmp_path, monkeypatch, manual_answers, output_type):
    pytest.importorskip("numpy")
    parent_dir = tmp_path / "manual"
    parent_dir.mkdir()
    name = phonon._agbr_parent(parent_dir, phonon.IR_PARENT)
    manual = prompts._opt2d12(parent_dir, name, answers={"exact settings": "n",
                                                         "calculation type": "3",
                                                         **manual_answers})
    step = _workflow_step(tmp_path, monkeypatch, output_type)
    (tmp_path / "wf").mkdir()
    eng = base._engine(tmp_path / "wf", (parent_dir / f"{name}.d12").read_text(),
                       (parent_dir / f"{name}.out").read_text(), {"FREQ_3": step})
    seen = base._numbered(eng, "FREQ", run_script=True)
    assert seen["ok"], seen["log"][-1500:]
    (deck,) = seen["decks"].values()
    return manual, deck.rstrip("\n").split("\n")


def _same_deck(manual, workflow):
    assert workflow[1:] == manual[1:]
    # The first line is the title, named after each route's parent file.
    assert workflow[0].split(" - ", 1)[1:] == manual[0].split(" - ", 1)[1:]


def test_planner_phonon_dos_step_writes_opt2d12_template_7(tmp_path, monkeypatch):
    manual, workflow = _both_routes(tmp_path, monkeypatch, OPT2D12_DOS, "2")
    block = phonon._block(manual)
    assert block[:3] == ["FREQCALC", "NOINTENS", "DISPERSION"]
    assert "PDOS" in block and "BANDS" not in block and "INTERPHESS" not in manual
    _same_deck(manual, workflow)


def test_planner_phonon_both_step_writes_bands_and_pdos(tmp_path, monkeypatch):
    manual, workflow = _both_routes(tmp_path, monkeypatch, OPT2D12_BOTH, "3")
    block = phonon._block(manual)
    assert block[:3] == ["FREQCALC", "NOINTENS", "DISPERSION"]
    assert "BANDS" in block and "PDOS" in block and "INTERPHESS" not in manual
    assert block.index("BANDS") < block.index("PDOS")
    _same_deck(manual, workflow)


def test_both_is_template_6_bands_then_template_7_pdos(tmp_path):
    """opt2d12's custom "both" deck is the two templates' records together."""
    pytest.importorskip("numpy")
    decks = {}
    for key, answers in (("bands", {"Select template": "6"}), ("dos", OPT2D12_DOS),
                         ("both", OPT2D12_BOTH)):
        d = tmp_path / key
        d.mkdir()
        name = phonon._agbr_parent(d, phonon.IR_PARENT)
        decks[key] = phonon._block(prompts._opt2d12(
            d, name, answers={"exact settings": "n", "calculation type": "3", **answers}))
    pdos = decks["dos"][decks["dos"].index("PDOS"):-1]
    assert decks["both"] == decks["bands"][:-1] + pdos + ["ENDFREQ"]


def test_unknown_output_type_falls_back_to_the_band_structure_step(tmp_path, monkeypatch):
    """"Invalid choice - defaulting to band structure" stored no BANDS path."""
    band = _workflow_step(tmp_path / "a", monkeypatch, "1")["frequency_settings"]
    other = _workflow_step(tmp_path / "b", monkeypatch, "9")["frequency_settings"]
    assert other == band and "pdos" not in band
