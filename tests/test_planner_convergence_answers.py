"""Planner convergence menus: yes/no answers keep their 1.1.1 meaning.

Up to 1.1.1 the planner asked "Use tight convergence? [y/N]" (SP) and "Use
tighter convergence for refined optimization? [Y/n]" (OPT). Both became 0-3
level menus, and any answer outside the menu, a habitual or scripted 'y'
included, was taken silently as 0 (keep the previous step's values). Now
y/yes gives the old tight values, n/no gives Standard, each with a note, and
any other answer is asked again.

The questions are answered through a real piped stdin in a subprocess.
"""
import json
import subprocess
import sys

import pytest

from conftest import REPO_ROOT
from d12_constants import opt_convergence, scf_tolerances

SCRIPT = r"""
import json, sys, tempfile
sys.path.insert(0, sys.argv[1])
from mace.workflow.planner import WorkflowPlanner
p = WorkflowPlanner(work_dir=tempfile.mkdtemp())
if sys.argv[2] == "sp":
    p._get_method_modifications = lambda: {"inherit_functional": True}
    p._get_basis_modifications = lambda: {"inherit_basis": True}
    cfg = p._get_basic_sp_config()
else:
    cfg = p._get_basic_opt_config()
print("RESULT=" + json.dumps(cfg))
"""


def _run(kind, stdin):
    r = subprocess.run([sys.executable, "-c", SCRIPT, str(REPO_ROOT), kind],
                       input=stdin, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    # The result follows the last prompt on its line (input() adds no newline)
    line = [l for l in r.stdout.splitlines() if "RESULT=" in l][-1]
    return json.loads(line.split("RESULT=", 1)[1]), r.stdout


OLD_OPT_TIGHT = {"TOLDEG": 1.5e-5, "TOLDEX": 6e-5, "TOLDEE": 8, "MAXCYCLE": 1000}


@pytest.mark.parametrize("answer", ["y", "yes", "Y"])
def test_sp_yes_is_the_old_tight_convergence(answer):
    cfg, out = _run("sp", f"{answer}\n")
    assert cfg["tolerance_modifications"] == {"custom_tolerances": scf_tolerances("3")}
    assert scf_tolerances("3") == {"TOLINTEG": "9 9 9 11 38", "TOLDEE": 11}
    assert "now a 0-3 menu" in out


@pytest.mark.parametrize("answer", ["n", "no"])
def test_sp_no_is_standard(answer):
    cfg, out = _run("sp", f"{answer}\n")
    assert cfg["tolerance_modifications"] == {"custom_tolerances": scf_tolerances("1")}
    assert "now a 0-3 menu" in out


def test_sp_other_answer_is_asked_again():
    cfg, out = _run("sp", "tight\n5\n2\n")
    assert cfg["tolerance_modifications"] == {"custom_tolerances": scf_tolerances("2")}
    assert out.count("Choose convergence level (0-3) [0]:") == 3


def test_sp_blank_still_keeps_the_previous_tolerances():
    cfg, _ = _run("sp", "\n")
    assert "tolerance_modifications" not in cfg


@pytest.mark.parametrize("answer", ["y", "yes"])
def test_opt_yes_is_the_old_tight_convergence(answer):
    cfg, out = _run("opt", f"1\n{answer}\n")
    assert cfg["optimization_type"] == "FULLOPTG"
    assert cfg["optimization_settings"] == OLD_OPT_TIGHT
    assert "now a 0-3 menu" in out


@pytest.mark.parametrize("answer", ["n", "no"])
def test_opt_no_is_standard(answer):
    cfg, _ = _run("opt", f"1\n{answer}\n")
    assert cfg["optimization_settings"] == opt_convergence("1", upper=True)


def test_opt_other_answer_is_asked_again():
    cfg, out = _run("opt", "1\nsure\n3\n")
    assert cfg["optimization_settings"] == opt_convergence("3", upper=True)
    assert out.count("Choose convergence level (0-3, default 0):") == 2


def test_opt_blank_still_keeps_the_previous_opt():
    cfg, _ = _run("opt", "1\n\n")
    assert "optimization_settings" not in cfg
