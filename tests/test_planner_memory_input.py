"""The workflow planner's "Memory per CPU" answer, as `mace submit -i` takes it.

WorkflowPlanner.get_safe_memory_input appended G to a bare number ("4096"
became --mem-per-cpu=4096G, a job SLURM can never schedule) and accepted
"4K". It now takes <n>G or <n>M with an optional B and asks again otherwise,
with the same help text. Answers that were valid before give the same
#SBATCH line as before (planner -> executor job script, golden values).
"""
import builtins
import os
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT


def _feed(monkeypatch, replies, tty):
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


def _sbatch_mem_line(monkeypatch, answer):
    """The job script line the planner's custom resources give, through the executor."""
    from mace.submission.crystal import get_default_d12_resources
    from mace.workflow.executor import WorkflowExecutor
    _feed(monkeypatch, ["", answer, "", ""], tty=False)     # cores, memory, walltime, account
    res = _planner().get_custom_resources(get_default_d12_resources(), "OPT")
    ex = WorkflowExecutor.__new__(WorkflowExecutor)
    ex.work_dir = Path(os.getcwd())
    tmpl = (REPO_ROOT / "mace" / "submission" / "submitcrystal23.sh").read_text()
    out = ex.apply_script_customizations(tmpl, {"resources": res})
    return [l for l in out.splitlines() if "SBATCH" in l and "--mem" in l]


@pytest.mark.parametrize("answer, line", [
    ("5G", "echo '#SBATCH --mem-per-cpu=5G' >> $1.sh"),
    ("4000MB", "echo '#SBATCH --mem-per-cpu=4000MB' >> $1.sh"),
    ("4GB", "echo '#SBATCH --mem-per-cpu=4GB' >> $1.sh"),
    ("4000m", "echo '#SBATCH --mem-per-cpu=4000M' >> $1.sh"),
])
def test_valid_answers_give_the_same_sbatch_line_as_before(monkeypatch, answer, line):
    assert _sbatch_mem_line(monkeypatch, answer) == [line]


@pytest.mark.parametrize("bad", ["4096", "4K", "4KB", "4T", "4 G", "G"])
def test_a_terminal_is_asked_again(monkeypatch, capsys, bad):
    _feed(monkeypatch, [bad, "5G"], tty=True)
    assert _planner().get_safe_memory_input("Memory per CPU [5G]: ", "5G") == "5G"
    out = capsys.readouterr()
    text = out.out + out.err
    assert f"Invalid memory format '{bad}'" in text
    assert "Valid formats: 4G, 4GB, 4000M, 4000MB" in text


def test_piped_bare_numbers_stop_the_run(monkeypatch):
    _feed(monkeypatch, ["4096", "4096", "4096", "5G"], tty=False)
    with pytest.raises(SystemExit) as exc:
        _planner().get_safe_memory_input("Memory per CPU [5G]: ", "5G")
    assert "not a terminal" in str(exc.value)


def test_blank_keeps_the_default(monkeypatch):
    _feed(monkeypatch, [""], tty=False)
    assert _planner().get_safe_memory_input("Memory per CPU [5G]: ", "5G") == "5G"
