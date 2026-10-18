"""Every question stops a piped run after three invalid answers in a row.

The stop existed only in d12_constants.yes_no_prompt. The other yes/no
questions (planner, opt2d3, plotting) and the numbered menus re-asked for as
long as the pipe had lines: answers piped in an older order were swallowed
one by one until a blank took the default and the run went on out of step.
At a terminal the question still asks until it gets an answer.
"""
import builtins
import os
import shutil
import subprocess
import sys

import pytest

import menu_nav
from conftest import REPO_ROOT, TEST_DATA

MACE_CLI = REPO_ROOT / "mace_cli"


def _env():
    return {k: v for k, v in os.environ.items()
            if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME")}


def _feed(monkeypatch, replies, tty):
    it = iter(replies)
    feeder = lambda prompt="": next(it)
    monkeypatch.setattr(builtins, "input", feeder)
    monkeypatch.setattr(menu_nav, "_REAL_INPUT", feeder)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: tty, raising=False)


def _readers():
    import d12_constants
    import d12_interactive
    import d3_interactive
    from mace.plotting import prompts as plot_prompts
    from mace.workflow import planner
    return [
        ("d12 get_user_input", lambda: d12_constants.get_user_input("Grid", {"1": "a", "2": "b"}, "1"), "2"),
        ("d12 get_valid_input", lambda: d12_constants.get_valid_input("Pick: ", ["1", "2"], "1"), "2"),
        ("d12 get_user_choice", lambda: d12_interactive.get_user_choice("Type", [("1", "a"), ("2", "b")], "1"), "2"),
        ("d3 yes_no_prompt", lambda: d3_interactive.yes_no_prompt("Plane?", "yes"), "n"),
        ("nav_int", lambda: menu_nav.nav_int("Order: ", default=0, choices=[0, 1]), "1"),
        ("nav_float", lambda: menu_nav.nav_float("Width: ", default=0.1), "0.2"),
        ("nav_choice", lambda: menu_nav.nav_choice("Pick: ", choices=["1", "2"], default="1"), "2"),
        ("planner get_user_input", lambda: planner.get_user_input("Input type", {"1": "a", "2": "b"}, "1"), "2"),
        ("planner yes_no_prompt", lambda: planner.yes_no_prompt("Go?", "yes"), "n"),
        ("plotting yes_no_prompt", lambda: plot_prompts.yes_no_prompt("Go?", "yes"), "n"),
        ("plotting select_option", lambda: plot_prompts.select_option("Pick", ["a", "b"], 1), "2"),
        ("plotting get_float_input", lambda: plot_prompts.get_float_input("Width", 0.1), "0.2"),
    ]


@pytest.mark.parametrize("index", range(12))
def test_no_terminal_stops_after_three_bad_answers(monkeypatch, index):
    name, ask, _ = _readers()[index]
    _feed(monkeypatch, ["zz", "zz", "zz", ""], tty=False)
    with pytest.raises(SystemExit) as exc:
        ask()
    assert "not a terminal" in str(exc.value), name


@pytest.mark.parametrize("index", range(12))
def test_two_bad_answers_are_still_asked_again(monkeypatch, index):
    name, ask, good = _readers()[index]
    _feed(monkeypatch, ["zz", "zz", good], tty=False)
    assert ask() not in (None,), name


@pytest.mark.parametrize("index", range(12))
def test_a_terminal_is_asked_until_it_answers(monkeypatch, index):
    name, ask, good = _readers()[index]
    _feed(monkeypatch, ["zz"] * 6 + [good], tty=True)
    ask()


def test_planner_stops_on_piped_answers_out_of_step(tmp_path):
    r = subprocess.run([sys.executable, str(MACE_CLI), "workflow", "--interactive"],
                       cwd=tmp_path, input="x\ny\nz\n" + "\n" * 5, capture_output=True,
                       text=True, timeout=120, env=_env())
    log = r.stdout + r.stderr
    assert r.returncode == 1, log[-3000:]
    assert "not a terminal" in log and "Traceback" not in log, log[-3000:]


def test_opt2d3_echg_stops_on_1_1_1_ordered_answers(tmp_path):
    parent = TEST_DATA / "OPT" / "1_dia_opt_rev1.out"
    if not parent.exists():
        pytest.skip("test/ corpus not present (gitignored)")
    shutil.copy(parent, tmp_path / "p.out")
    (tmp_path / "p.f9").write_bytes(b"x")
    # 1.1.1 order: ECHG, order, then straight to the plane - the coordinates
    # land on the plane's "Coordinate type (F)ractional or (C)artesian"
    r = subprocess.run([sys.executable, str(MACE_CLI), "opt2d3", "--input", "p.out",
                        "--calc-type", "CHARGE"], cwd=tmp_path,
                       input="2\n0\n50\n0 0 0\n1 0 0\n0 1 0\n" + "\n" * 20,
                       capture_output=True, text=True, timeout=300, env=_env())
    log = r.stdout + r.stderr
    assert r.returncode != 0, log[-3000:]
    assert "neither FRACTION nor ANGSTROM" in log, log[-3000:]
    assert not list(tmp_path.glob("*.d3"))
