"""The planner's free-text questions stop a piped run cleanly too.

The sequence builder's "Next calculation", the Advanced BAND custom path and
"Use all bands?", and the free-form SLURM directive question read stdin with
a bare input(): piped answers out of step were taken as they came (any word
as a directive or a k-path, anything but "n" as "use all bands", an unknown
calculation type asked again for as long as the pipe had lines), and the end
of the input was an EOFError traceback. They now refuse answers that cannot
be meant for them, stop after three in a row with no terminal, and stop with
a message when the input ends. The shared readers (yes/no, menus, numbers)
end the same way when the input runs out.
"""
import builtins
import os
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"
CIF = REPO_ROOT / "tests" / "data" / "1_dia_opt_BULK_OPTGEOM_symm.cif"


def _env():
    return {k: v for k, v in os.environ.items()
            if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME")}


def _planner_run(tmp_path, answers):
    shutil.copy(CIF, tmp_path / CIF.name)
    r = subprocess.run([sys.executable, str(MACE_CLI), "workflow", "--interactive"],
                       cwd=tmp_path, input=answers, capture_output=True, text=True,
                       timeout=180, env=_env())
    return r, r.stdout + r.stderr


def test_sequence_builder_end_of_input_is_not_a_traceback(tmp_path):
    # input type, directory, template 9 (custom), one step, then nothing
    r, log = _planner_run(tmp_path, "\n\n9\nOPT\n")
    assert r.returncode == 1, log[-3000:]
    assert "Traceback" not in log and "EOFError" not in log, log[-3000:]
    assert "input ended" in log, log[-3000:]


def test_sequence_builder_stops_after_three_unknown_types(tmp_path):
    r, log = _planner_run(tmp_path, "\n\n9\n" + "XX\n" * 20)
    assert r.returncode == 1, log[-3000:]
    assert "not a terminal" in log and "Traceback" not in log, log[-3000:]
    assert log.count("Unknown calculation type") == 3, log[-3000:]


def _feed(monkeypatch, replies, tty=False):
    it = iter(replies)

    def fake(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None
    monkeypatch.setattr(builtins, "input", fake)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: tty, raising=False)


def _planner():
    from mace.workflow.planner import WorkflowPlanner
    return WorkflowPlanner.__new__(WorkflowPlanner)


@pytest.mark.parametrize("bad", ["1", "n", "2:00:00"])
def test_custom_band_path_refuses_answers_that_are_not_labels(monkeypatch, bad):
    _feed(monkeypatch, ["2", bad, bad, bad])
    with pytest.raises(SystemExit) as exc:
        _planner()._get_advanced_d3_config("BAND")
    assert "not a terminal" in str(exc.value)


def test_custom_band_path_at_a_terminal_is_asked_again(monkeypatch):
    _feed(monkeypatch, ["2", "1", "n", "1", "1", "G X | M G", "", ""], tty=True)
    cfg = _planner()._get_advanced_d3_config("BAND")
    assert cfg["path"] == ["G", "X", "|", "M", "G"]


def test_use_all_bands_refuses_a_non_answer(monkeypatch):
    _feed(monkeypatch, ["2", "G X M G", "maybe", "maybe", "maybe"])
    with pytest.raises(SystemExit) as exc:
        _planner()._get_advanced_d3_config("BAND")
    assert "not a terminal" in str(exc.value)


@pytest.mark.parametrize("answers", [["2"], ["2", "G X M G"], ["2", "G X M G", "n", "3"]])
def test_advanced_band_end_of_input_stops_cleanly(monkeypatch, answers):
    _feed(monkeypatch, answers)
    with pytest.raises(SystemExit) as exc:
        _planner()._get_advanced_d3_config("BAND")
    assert "input ended" in str(exc.value)


def test_slurm_directive_refuses_answers_that_are_not_directives(monkeypatch):
    _feed(monkeypatch, ["y", "5G", "2:00:00"])
    with pytest.raises(SystemExit) as exc:
        _planner().get_additional_customizations("OPT")
    assert "not a terminal" in str(exc.value)


def test_slurm_directives_still_taken(monkeypatch):
    _feed(monkeypatch, ["1", "--constraint=amd20", "6", "-C amd22", ""])
    got = [c["directive"] for c in _planner().get_additional_customizations("OPT")]
    assert got == ["--constraint=intel18", "--constraint=amd20", "-C amd22"]


def test_slurm_directive_end_of_input_stops_cleanly(monkeypatch):
    _feed(monkeypatch, ["1"])
    with pytest.raises(SystemExit) as exc:
        _planner().get_additional_customizations("OPT")
    assert "input ended" in str(exc.value)


def _shared_readers():
    from mace.workflow.planner import yes_no_prompt
    p = _planner()
    return [
        lambda: yes_no_prompt("Go?", "yes"),
        lambda: p._ask_menu_choice("Pick [1]: ", ["1", "2"], "1"),
        lambda: p.get_safe_integer_input("N: ", 5),
        lambda: p.get_safe_float_input("X: ", 1.0),
        lambda: p.get_safe_choice_input("Pick", ["1", "2"], "1"),
        lambda: p.get_safe_memory_input("Mem: ", "5G"),
        lambda: p.get_safe_walltime_input("Time: ", "1:00:00"),
        lambda: p.get_safe_float_list_input("L: ", [1.0]),
    ]


@pytest.mark.parametrize("index", range(8))
def test_shared_readers_stop_cleanly_at_end_of_input(monkeypatch, index):
    _feed(monkeypatch, [])
    with pytest.raises(SystemExit) as exc:
        _shared_readers()[index]()
    assert "input ended" in str(exc.value)
