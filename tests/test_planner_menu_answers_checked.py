"""Planner menus that took any answer: the OPT type and the CHARGE+POTENTIAL output.

"Choose optimization type (0-3, default 0)" took any answer silently as 0
(keep the previous OPT's type), and the advanced CHARGE+POTENTIAL "Select
output type" looked the answer up in its table and raised KeyError on
anything off the menu. Both now ask again, and stop a piped run after three
invalid answers. Driven through a real piped stdin in a subprocess.
"""
import json
import subprocess
import sys

from conftest import REPO_ROOT

SCRIPT = r"""
import json, sys, tempfile
sys.path.insert(0, sys.argv[1])
from mace.workflow.planner import WorkflowPlanner
p = WorkflowPlanner(work_dir=tempfile.mkdtemp())
if sys.argv[2] == "opt":
    cfg = p._get_basic_opt_config()
else:
    cfg = p._get_advanced_d3_config("CHARGE+POTENTIAL")
print("RESULT=" + json.dumps(cfg))
"""


def _run(kind, stdin):
    r = subprocess.run([sys.executable, "-c", SCRIPT, str(REPO_ROOT), kind],
                       input=stdin, capture_output=True, text=True, timeout=120)
    out = r.stdout + r.stderr
    lines = [l for l in r.stdout.splitlines() if "RESULT=" in l]
    cfg = json.loads(lines[-1].split("RESULT=", 1)[1]) if lines else None
    return r.returncode, cfg, out


def test_opt_type_asks_again_on_an_off_menu_answer():
    rc, cfg, out = _run("opt", "5\n2\n\n")
    assert rc == 0, out[-2000:]
    assert cfg["optimization_type"] == "ATOMONLY"


def test_opt_type_blank_keeps_the_previous_type():
    rc, cfg, out = _run("opt", "\n\n")
    assert rc == 0, out[-2000:]
    assert "optimization_type" not in cfg


def test_opt_type_stops_a_piped_run_out_of_step():
    rc, cfg, out = _run("opt", "y\nFULLOPTG\n9\n\n\n")
    assert rc == 1 and cfg is None, out[-2000:]
    assert "not a terminal" in out


def test_charge_potential_output_type_off_menu_is_asked_again():
    rc, cfg, out = _run("cp", "9\n2\n" + "\n" * 30)
    assert "Traceback" not in out and "KeyError" not in out, out[-2000:]
    assert rc == 0, out[-2000:]
    assert cfg["option_type"] == 3


def test_charge_potential_output_type_stops_a_piped_run_out_of_step():
    rc, cfg, out = _run("cp", "9\nx\n0\n" + "\n" * 30)
    assert "Traceback" not in out and "KeyError" not in out, out[-2000:]
    assert rc == 1, out[-2000:]
    assert "not a terminal" in out
