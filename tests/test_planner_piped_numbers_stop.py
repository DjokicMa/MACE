"""The planner's number, menu and resource questions stop a piped run too.

The three-strike stop reached the planner's yes/no questions and its
get_user_input menu, but not its get_safe_* readers (numbers, memory,
walltime, choices) or its level and basis/grid/functional menus. Piped
answers out of step were swallowed there while the pipe had lines, and
the run ended in an EOFError traceback at "Enter level (0-3)".
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


def test_blank_piped_planner_stops_at_the_level_menu(tmp_path):
    shutil.copy(CIF, tmp_path / CIF.name)
    r = subprocess.run([sys.executable, str(MACE_CLI), "workflow", "--interactive"],
                       cwd=tmp_path, input="\n" * 40, capture_output=True, text=True,
                       timeout=180, env=_env())
    log = r.stdout + r.stderr
    assert r.returncode == 1, log[-3000:]
    assert "not a terminal" in log, log[-3000:]
    assert "Traceback" not in log, log[-3000:]
    assert log.count("Please enter a valid number") == 3, log[-3000:]


def _feed(monkeypatch, replies, tty):
    it = iter(replies)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(it))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: tty, raising=False)


def _readers():
    from mace.workflow.planner import WorkflowPlanner
    p = WorkflowPlanner.__new__(WorkflowPlanner)
    return [
        ("integer", lambda: p.get_safe_integer_input("N: ", 5), "7"),
        ("float", lambda: p.get_safe_float_input("X: ", 1.0), "2.5"),
        ("float list", lambda: p.get_safe_float_list_input("L: ", [1.0]), "1 2"),
        ("memory", lambda: p.get_safe_memory_input("Mem: ", "5G"), "4G"),
        ("walltime", lambda: p.get_safe_walltime_input("Time: ", "1:00:00"), "2:00:00"),
        ("choice", lambda: p.get_safe_choice_input("Pick", ["1", "2"], "1"), "2"),
    ]


@pytest.mark.parametrize("index", range(6))
def test_no_terminal_stops_after_three_bad_answers(monkeypatch, index):
    name, ask, _ = _readers()[index]
    _feed(monkeypatch, ["zz", "zz", "zz", ""], tty=False)
    with pytest.raises(SystemExit) as exc:
        ask()
    assert "not a terminal" in str(exc.value), name


@pytest.mark.parametrize("index", range(6))
def test_a_terminal_is_asked_until_it_answers(monkeypatch, index):
    name, ask, good = _readers()[index]
    _feed(monkeypatch, ["zz"] * 6 + [good], tty=True)
    assert ask() is not None, name


@pytest.mark.parametrize("answers", [["9", "9", "9"], ["1", "7", "7", "7"]])
def test_advanced_band_menus_refuse_off_menu_answers(monkeypatch, answers):
    """An off-menu k-path option or path format was taken silently: "3" for
    the path option meant a custom path, "7" for the format meant labels."""
    from mace.workflow.planner import WorkflowPlanner
    p = WorkflowPlanner.__new__(WorkflowPlanner)
    _feed(monkeypatch, answers + [""] * 10, tty=False)
    with pytest.raises(SystemExit) as exc:
        p._get_advanced_d3_config("BAND")
    assert "not a terminal" in str(exc.value)
