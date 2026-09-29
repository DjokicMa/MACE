"""Pins every copy of the plain-text ``ui`` fallback shim and of ``yes_no_prompt``.

The copies are not all alike, and each caller's behaviour is what these tests
fix in place, so any later consolidation has to keep every one of them exactly:

* The rich-less ``ui`` fallback. Six sub-tools carry the same small shim
  (CRYSTALOptToD12, NewCifToD12, CRYSTALOptToD3, the workflow planner and
  executor, the database explorer); mace_cli carries a larger one with a
  different ``print``/``rule``/``table`` and extra ``banner``/``credits``.
  The shim is taken from each file's own fallback branch, so the test runs the
  code a user gets when ``from mace.utils import ui`` fails.
* ``yes_no_prompt``, which exists four times with four behaviours: the prompt
  suffix differs, which answers are accepted differs, and an unrecognised
  answer re-asks in three of them but counts as "no" in d3_interactive.
"""
import ast
import builtins
import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

SUBTOOL_SHIM_FILES = [
    "Crystal_d12/CRYSTALOptToD12.py",
    "Crystal_d12/NewCifToD12.py",
    "Crystal_d3/CRYSTALOptToD3.py",
    "mace/workflow/planner.py",
    "mace/workflow/executor.py",
    "mace/database/interactive/interactive.py",
]


def _fallback_ui(rel):
    """Run the file's ``except`` branch of ``try: from mace.utils import ui`` and
    return the ``ui`` object it builds."""
    tree = ast.parse((REPO / rel).read_text())
    for node in tree.body:
        if (isinstance(node, ast.Try) and len(node.body) == 1
                and isinstance(node.body[0], ast.ImportFrom)
                and node.body[0].module == "mace.utils"
                and [a.name for a in node.body[0].names] == ["ui"]):
            code = ast.Module(body=node.handlers[0].body, type_ignores=[])
            ns = {"__name__": "_shim_under_test"}
            exec(compile(code, rel, "exec"), ns)
            return ns["ui"]
    raise AssertionError(f"no guarded `from mace.utils import ui` in {rel}")


def _mace_cli_fallback_ui():
    """Run mace_cli's ``if not UI_AVAILABLE:`` block and return its ``ui``."""
    tree = ast.parse((REPO / "mace_cli").read_text())
    for node in tree.body:
        if (isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp)
                and isinstance(node.test.op, ast.Not)
                and getattr(node.test.operand, "id", None) == "UI_AVAILABLE"):
            code = ast.Module(body=node.body, type_ignores=[])
            ns = {"__name__": "_shim_under_test", "sys": sys, "os": os}
            exec(compile(code, "mace_cli", "exec"), ns)
            return ns["ui"]
    raise AssertionError("no `if not UI_AVAILABLE:` fallback in mace_cli")


# ------------------------------------------------------------ ui fallback ---

@pytest.mark.parametrize("rel", SUBTOOL_SHIM_FILES)
def test_subtool_ui_fallback_behaviour(rel, capsys):
    ui = _fallback_ui(rel)

    ui.ok("[bold]Done[/bold] [OK]")
    ui.info("[#aabbcc]info[/] [Errno 2]")
    ui.print("[red]p[/red] [1, 2, 3]")
    assert capsys.readouterr() == ("Done [OK]\ninfo [Errno 2]\np [1, 2, 3]\n", "")

    ui.warn("[yellow]w[/yellow]")
    ui.err("[/]e")
    assert capsys.readouterr() == ("", "w\ne\n")

    ui.rule("[bold]Title[/bold]")
    ui.rule()
    assert capsys.readouterr().out == "Title\n\n"

    ui.table(["a", "b"], [[1, "[b]x[/b]"], [2, "y"]], title="[b]T[/b]")
    # Only the title is stripped; cells are printed as given, unpadded.
    assert capsys.readouterr().out == "T\n1  [b]x[/b]\n2  y\n"
    ui.table(["a"], [["z"]])
    assert capsys.readouterr().out == "z\n"

    it = iter([1, 2])
    assert ui.progress(it, total=2, description="x") is it
    assert ui.badge("done") == "DONE"
    assert ui.badge(3) == "3"


def test_mace_cli_ui_fallback_behaviour(capsys):
    ui = _mace_cli_fallback_ui()

    ui.ok("[bold]Done[/bold] [OK]")
    ui.info("[#aabbcc]info[/]")
    ui.print("[red]p[/red]", 5, "[x]", sep="|", style="bold", markup=True)
    assert capsys.readouterr() == ("Done [OK]\ninfo\np|5|\n", "")
    ui.print("to-err", file=sys.stderr)
    ui.warn("[yellow]w[/yellow]")
    ui.err("e")
    assert capsys.readouterr() == ("", "to-err\nw\ne\n")

    ui.rule("[bold]Title[/bold]")
    ui.rule()
    assert capsys.readouterr().out == "-- Title " + "-" * 69 + "\n" + "-" * 80 + "\n"

    ui.table(["name", "n"], [["[b]ab[/b]", 1], ["c", 22]], title="[b]T[/b]")
    assert capsys.readouterr().out == (
        "T\nname  n \n----  --\nab    1 \nc     22\n")

    it = iter([1])
    assert ui.progress(it, total=1) is it
    assert ui.badge("run") == "RUN"


def test_mace_cli_ui_fallback_banner(capsys, monkeypatch):
    ui = _mace_cli_fallback_ui()
    monkeypatch.delenv("MACE_NO_BANNER", raising=False)
    ui.banner("9.9")
    assert capsys.readouterr().out == "MACE v9.9 - Mendoza Automated CRYSTAL Engine\n"
    monkeypatch.setenv("MACE_NO_BANNER", "1")
    ui.banner("9.9")
    assert capsys.readouterr().out == ""


# ----------------------------------------------------------- yes_no_prompt ---

def _nav_script(monkeypatch, module, answers):
    """Replace ``module._nav_read`` with a scripted reader; return the calls."""
    calls = []
    answers = list(answers)

    def fake(prompt="", valid_set=None):
        calls.append((prompt, valid_set))
        answer = answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(module, "_nav_read", fake)
    return calls


def _input_script(monkeypatch, answers):
    """Replace ``builtins.input`` with a scripted reader; return the prompts."""
    prompts = []
    answers = list(answers)

    def fake(prompt=""):
        prompts.append(prompt)
        answer = answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(builtins, "input", fake)
    return prompts


# d12_constants: "<prompt> [Y/n] " (no colon), y/yes/n/no only, lower-cased but
# not stripped, re-asks on anything else; default must be exactly yes/no.
D12_CASES = [
    ("yes", [""], True), ("no", [""], False),
    ("yes", ["y"], True), ("yes", ["Y"], True), ("yes", ["YES"], True),
    ("no", ["yes"], True), ("yes", ["n"], False), ("yes", ["No"], False),
    ("yes", ["maybe", "n"], False), ("no", ["1", "y"], True),
    ("yes", ["true", "no"], False), ("yes", [" y", "y"], True),
]


@pytest.mark.parametrize("default,answers,expected", D12_CASES)
def test_yes_no_prompt_d12_constants(default, answers, expected, monkeypatch, capsys):
    import d12_constants
    capsys.readouterr()  # drop anything printed on first import
    calls = _nav_script(monkeypatch, d12_constants, answers)
    assert d12_constants.yes_no_prompt("Go?", default) is expected
    suffix = " [Y/n] " if default == "yes" else " [y/N] "
    assert calls == [("Go?" + suffix, {"yes": True, "y": True, "no": False, "n": False})
                     ] * len(answers)
    out = capsys.readouterr().out
    assert out == "Please respond with 'yes' or 'no' (or 'y' or 'n').\n" * (len(answers) - 1)


def test_yes_no_prompt_d12_constants_default_and_eof(monkeypatch):
    import d12_constants
    with pytest.raises(ValueError, match="Invalid default value: Yes"):
        d12_constants.yes_no_prompt("Go?", "Yes")
    _nav_script(monkeypatch, d12_constants, [EOFError()])
    with pytest.raises(EOFError):
        d12_constants.yes_no_prompt("Go?")


# d3_interactive: "<prompt> [Y/n]: ", stripped and lower-cased, asks ONCE: an
# unrecognised answer is "no", whatever the default.
D3_CASES = [
    ("yes", [""], True), ("no", [""], False), ("Yes", [""], True),
    ("maybe", [""], False),
    ("yes", ["y"], True), ("yes", [" Y "], True), ("no", ["YES"], True),
    ("yes", ["n"], False), ("yes", ["no"], False),
    ("yes", ["maybe"], False), ("yes", ["1"], False), ("yes", ["true"], False),
]


@pytest.mark.parametrize("default,answers,expected", D3_CASES)
def test_yes_no_prompt_d3_interactive(default, answers, expected, monkeypatch, capsys):
    import d3_interactive
    capsys.readouterr()  # drop anything printed on first import
    calls = _nav_script(monkeypatch, d3_interactive, answers)
    assert d3_interactive.yes_no_prompt("Go?", default) is expected
    suffix = " [Y/n]: " if default.lower() == "yes" else " [y/N]: "
    assert calls == [("Go?" + suffix, {"y", "yes", "n", "no"})]
    assert capsys.readouterr() == ("", "")


def test_yes_no_prompt_d3_interactive_eof(monkeypatch):
    import d3_interactive
    _nav_script(monkeypatch, d3_interactive, [EOFError()])
    with pytest.raises(EOFError):
        d3_interactive.yes_no_prompt("Go?")


# mace/plotting/prompts.py and the planner: plain input(), "<prompt> [Y/n]: ",
# stripped and lower-cased, also accept true/1 and false/0, re-ask on anything
# else. They differ only in how the re-ask is reported.
INPUT_CASES = [
    ("yes", [""], True), ("no", [""], False), ("NO", [""], False),
    ("maybe", [""], False),
    ("yes", ["y"], True), ("yes", [" Y "], True), ("no", ["yes"], True),
    ("no", ["true"], True), ("no", ["1"], True),
    ("yes", ["n"], False), ("yes", ["No"], False), ("yes", ["false"], False),
    ("yes", ["0"], False),
    ("yes", ["maybe", "n"], False), ("no", ["2", "", ], False),
]


def _load_plotting_prompts():
    # Loaded from its file: importing the mace.plotting package pulls in the
    # plotting front end, which this helper does not need.
    path = REPO / "mace" / "plotting" / "prompts.py"
    spec = importlib.util.spec_from_file_location("_plotting_prompts_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("default,answers,expected", INPUT_CASES)
def test_yes_no_prompt_plotting(default, answers, expected, monkeypatch, capsys):
    prompts_mod = _load_plotting_prompts()
    prompts = _input_script(monkeypatch, answers)
    assert prompts_mod.yes_no_prompt("Go?", default) is expected
    suffix = " [Y/n]: " if default.lower() == "yes" else " [y/N]: "
    assert prompts == ["Go?" + suffix] * len(answers)
    bad = [a.strip().lower() for a in answers[:-1]]
    assert capsys.readouterr() == (
        "".join(f"  Invalid response '{b}'. Please enter yes/no (y/n).\n" for b in bad), "")


@pytest.mark.parametrize("default,answers,expected", INPUT_CASES)
def test_yes_no_prompt_planner(default, answers, expected, monkeypatch, capsys):
    from mace.workflow import planner
    capsys.readouterr()  # drop anything printed on first import
    warned = []
    monkeypatch.setattr(planner.ui, "warn", lambda m: warned.append(m))
    prompts = _input_script(monkeypatch, answers)
    assert planner.yes_no_prompt("Go?", default) is expected
    suffix = " [Y/n]: " if default.lower() == "yes" else " [y/N]: "
    assert prompts == ["Go?" + suffix] * len(answers)
    bad = [a.strip().lower() for a in answers[:-1]]
    assert warned == [f"⚠️  Invalid response '{b}'. Please enter yes/no (y/n)."
                      for b in bad]
    assert capsys.readouterr() == ("", "")


def test_yes_no_prompt_input_based_eof(monkeypatch):
    from mace.workflow import planner
    for fn in (_load_plotting_prompts().yes_no_prompt, planner.yes_no_prompt):
        _input_script(monkeypatch, [EOFError()])
        with pytest.raises(EOFError):
            fn("Go?")
