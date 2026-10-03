"""A FREQ child keeps a FREQ parent's IR/Raman intensities only where they apply.

Two things went wrong once FREQ children kept the parent's intensities:

- opt2d12's template menu offered "0: Keep the parent's FREQCALC settings" as
  its default for a parent with only NUMDERIV, so the NUMDERIV question (whose
  default is the parent's NUMDERIV) was skipped and an answer to it was lost.
  The keep option is now offered only for a parent with intensity or spectrum
  settings.
- a config naming no intensity setting got the parent's intensities whatever
  its FREQ mode, so a phonon dispersion child of an IR parent wrote INTENS /
  INTCPHF. Intensities are Gamma-point properties (CRYSTAL23 manual,
  FREQCALC): they are now kept only when the child runs in the parent's mode.

The workflow's phonon band structure step (Advanced, phonon, Enter-through)
now writes what opt2d12's "6: Phonon band structure" template writes on
Enter-through: NOINTENS, DISPERSION and its automatic BANDS path, no
INTERPHESS.
"""
import shutil

import pytest

import test_freqcalc_parent_settings as agbr
import test_numbered_calc_keeps_parent_method as base
import test_opt2d12_prompt_defaults as prompts
from conftest import TEST_DATA
from mace.workflow.planner import WorkflowPlanner

IR_PARENT = "FREQCALC\nINTENS\nINTCPHF\nENDCPHF\nENDFREQ"
AG1CL1_FREQ = "FREQ/Ag1Cl1_sym_*_freq_B3LYP-D3-D3_optimized.out"


def _block(lines):
    return lines[lines.index("FREQCALC"):lines.index("ENDFREQ") + 1]


def _agbr_parent(tmp_path, freqcalc):
    """The real AgBr run, its deck made a FREQ deck with ``freqcalc``."""
    lines = agbr.AGBR.with_suffix(".d12").read_text().split("\n")
    start, end = lines.index("OPTGEOM"), lines.index("ENDOPT")
    (tmp_path / "agbr_freq.d12").write_text(
        "\n".join(lines[:start] + freqcalc.split("\n") + lines[end + 1:]))
    shutil.copy(agbr.AGBR.with_suffix(".out"), tmp_path / "agbr_freq.out")
    return "agbr_freq"


def _ag1cl1_parent(tmp_path):
    found = sorted(TEST_DATA.glob(AG1CL1_FREQ))
    if not found:
        pytest.skip("test/ corpus not present (gitignored, ~12GB)")
    shutil.copy(found[0], tmp_path)
    shutil.copy(found[0].with_suffix(".d12"), tmp_path)
    return found[0].stem


# ---------------------------------------------- a NUMDERIV-only FREQ parent

def test_numderiv_only_parent_is_not_offered_the_keep_template(tmp_path):
    name = _agbr_parent(tmp_path, "FREQCALC\nNUMDERIV\n2\nNOINTENS\nENDFREQ")
    child = prompts._opt2d12(tmp_path, name, answers={"exact settings": "n",
                                                      "calculation type": "3"})
    assert _block(child) == ["FREQCALC", "NUMDERIV", "2", "NOINTENS", "ENDFREQ"]


def test_numderiv_only_parent_answer_zero_drops_it(tmp_path):
    name = _agbr_parent(tmp_path, "FREQCALC\nNUMDERIV\n2\nNOINTENS\nENDFREQ")
    child = prompts._opt2d12(tmp_path, name, answers={"exact settings": "n",
                                                      "calculation type": "3",
                                                      "Select method (0-2)": "0"})
    assert _block(child) == ["FREQCALC", "NOINTENS", "ENDFREQ"]


def test_keep_template_offered_only_for_intensity_or_spectrum_settings(monkeypatch, capsys):
    import d12_calc_freq
    import menu_nav

    def blank(prompt=""):
        print(prompt)
        return ""

    monkeypatch.setattr("builtins.input", blank)
    monkeypatch.setattr(menu_nav, "_REAL_INPUT", blank)
    got = d12_calc_freq.get_advanced_frequency_settings(2, {"numderiv": 2})
    out = capsys.readouterr().out
    assert "Keep the parent's FREQCALC" not in out and "[2]" in out
    assert got["numderiv"] == 2 and not got.get("intensities")

    got = d12_calc_freq.get_advanced_frequency_settings(2, {"numderiv": 2, "intensities": True})
    assert "Keep the parent's FREQCALC" in capsys.readouterr().out
    assert got == {"numderiv": 2, "intensities": True}


# --------------------------------------- a config in another FREQ mode

DISPERSION_CONFIG = ('{"calculation_type": "FREQ", "freq_settings": '
                     '{"mode": "DISPERSION", "dispersion": true, "scelphono": [2, 2, 2]}}')


def test_dispersion_config_does_not_get_the_parents_intensities(tmp_path):
    (tmp_path / "c.json").write_text(DISPERSION_CONFIG)
    child = agbr._freq_child(tmp_path, IR_PARENT, "--config-file", "c.json")
    assert child == ["FREQCALC", "NOINTENS", "DISPERSION", "ENDFREQ"]


def test_gamma_config_still_gets_the_parents_intensities(tmp_path):
    (tmp_path / "c.json").write_text(
        '{"calculation_type": "FREQ", "freq_settings": {"mode": "GAMMA"}}')
    child = agbr._freq_child(tmp_path, IR_PARENT, "--config-file", "c.json")
    assert child == ["FREQCALC", "INTENS", "INTCPHF", "ENDCPHF", "ENDFREQ"]


# ------------------- workflow phonon band step == opt2d12 template 6 deck

def _workflow_band_step(tmp_path, monkeypatch):
    planner = WorkflowPlanner(work_dir=tmp_path / "plan")

    def reply(prompt=""):
        if "customization level" in prompt:
            return "2"
        if "Select mode [1]" in prompt:
            return "2"          # phonon calculations; Enter for the rest
        return ""

    monkeypatch.setattr("builtins.input", reply)
    return planner.configure_frequency_step("FREQ", 3)


def _both_routes(tmp_path, monkeypatch, parent_dir, name):
    pytest.importorskip("numpy")
    manual = prompts._opt2d12(parent_dir, name, answers={"exact settings": "n",
                                                         "calculation type": "3",
                                                         "Select template": "6"})
    step = _workflow_band_step(tmp_path, monkeypatch)
    eng = base._engine(tmp_path / "wf", (parent_dir / f"{name}.d12").read_text(),
                       (parent_dir / f"{name}.out").read_text(), {"FREQ_3": step})
    seen = base._numbered(eng, "FREQ", run_script=True)
    assert seen["ok"], seen["log"][-1500:]
    (deck,) = seen["decks"].values()
    return manual, deck.rstrip("\n").split("\n")


def _assert_same_dispersion_deck(manual, workflow):
    assert "INTENS" not in manual and "INTCPHF" not in manual
    assert _block(manual)[:3] == ["FREQCALC", "NOINTENS", "DISPERSION"]
    assert "BANDS" in _block(manual) and "INTERPHESS" not in manual
    # The first line is the title, named after each route's parent file.
    assert workflow[1:] == manual[1:]
    assert workflow[0].split(" - ", 1)[1:] == manual[0].split(" - ", 1)[1:]


def test_dispersion_child_of_an_ir_parent_same_on_both_routes(tmp_path, monkeypatch):
    parent_dir = tmp_path / "manual"
    parent_dir.mkdir()
    name = _agbr_parent(parent_dir, IR_PARENT)
    (tmp_path / "wf").mkdir()
    _assert_same_dispersion_deck(*_both_routes(tmp_path, monkeypatch, parent_dir, name))


def test_corpus_ag1cl1_dispersion_child_same_on_both_routes(tmp_path, monkeypatch):
    parent_dir = tmp_path / "manual"
    parent_dir.mkdir()
    name = _ag1cl1_parent(parent_dir)
    assert _block((parent_dir / f"{name}.d12").read_text().split("\n")) == IR_PARENT.split("\n")
    (tmp_path / "wf").mkdir()
    _assert_same_dispersion_deck(*_both_routes(tmp_path, monkeypatch, parent_dir, name))
