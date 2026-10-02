"""The legacy WorkflowExecutor step generator applies the step configuration.

WorkflowExecutor.run_crystal_opt_conversion (reached only from the
executor's own monitor loop, which nothing in MACE starts) passed its step
configuration as --options-file, which CRYSTALOptToD12 only uses as the file
--save-options writes. The configuration was never read: a FREQ step wrote
whatever the parent's type was, and the run waited on prompts with no
terminal. It now passes --config-file with --non-interactive.
"""
import shutil
from pathlib import Path

import pytest

AGBR = Path(__file__).parent / "data" / "opt_restart" / "tqc_agbr"   # a real AgBr OPT run


def _executor(tmp_path):
    from mace.workflow.executor import WorkflowExecutor
    ex = WorkflowExecutor.__new__(WorkflowExecutor)
    ex.work_dir = tmp_path / "work"
    ex.work_dir.mkdir()
    ex.temp_dir = tmp_path / "tmp"
    ex.temp_dir.mkdir()
    return ex


@pytest.mark.parametrize("expert", [False, True])
def test_freq_step_config_reaches_the_deck(tmp_path, expert):
    src = tmp_path / "parent"
    src.mkdir()
    for ext in (".out", ".d12"):
        shutil.copy(AGBR.with_suffix(ext), src / f"agbr{ext}")
    out_dir = tmp_path / "next"
    out_dir.mkdir()
    freq = {"mode": "GAMMA", "numderiv": 2, "intensities": False}
    if expert:
        step = {"expert_mode": True,
                "crystal_opt_config": {"calculation_type": "FREQ", "freq_settings": freq}}
    else:
        step = {"inherit_settings": True, "frequency_settings": freq}
    _executor(tmp_path).run_crystal_opt_conversion(
        str(src / "agbr.out"), str(src / "agbr.d12"), out_dir, "FREQ", step)
    decks = list(out_dir.glob("*.d12"))
    assert len(decks) == 1, sorted(p.name for p in out_dir.iterdir())
    lines = decks[0].read_text().split("\n")
    assert "OPTGEOM" not in lines
    assert lines[lines.index("FREQCALC") + 1:lines.index("FREQCALC") + 3] == ["NUMDERIV", "2"]
