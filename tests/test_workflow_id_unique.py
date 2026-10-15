"""Workflows started in the same second get different ids.

Workflow ids had one-second resolution (``workflow_<YYYYmmdd_HHMMSS>``): two
launches in the same second (two `mace submit --progress` runs, a script
looping over folders' plans) shared one id, so the second plan file
overwrote the first and both workflows used one context. An id now ends in
four random hex digits; every reader takes the id whole.
"""
import json
import re
import subprocess
import sys

from array_helpers import MACE_CLI, OPT_D12
from fake_slurm import FakeSlurm
from test_array_default_path_golden import _env

ID = re.compile(r"^workflow_\d{8}_\d{6}_[0-9a-f]{4}$")


def _forced_suffixes(monkeypatch, suffixes):
    """Every id is minted in the same second, from these suffixes in turn."""
    from mace.workflow import context
    it = iter(suffixes)
    monkeypatch.setattr(context, "_id_suffix", lambda: next(it))
    monkeypatch.setattr(context, "_id_stamp", lambda: "20991231_000000")
    monkeypatch.setattr(context, "_ISSUED_IDS", set())


def test_ids_minted_in_one_second_differ(monkeypatch):
    from mace.workflow.context import new_workflow_id
    from mace.workflow.manual_progress import _mint_workflow_id
    ids = [new_workflow_id() for _ in range(40)] + [_mint_workflow_id() for _ in range(40)]
    assert all(ID.match(i) for i in ids), ids[:3]
    assert len(set(ids)) == len(ids)
    # the same suffix drawn again in the same second: a new one is drawn
    _forced_suffixes(monkeypatch, ["aaaa", "aaaa", "aaaa", "bbbb", "bbbb", "cccc"])
    ids = [new_workflow_id() for _ in range(3)]
    assert ids == ["workflow_20991231_000000_aaaa", "workflow_20991231_000000_bbbb",
                   "workflow_20991231_000000_cccc"]


def test_ids_reserved_in_a_folder_differ_across_processes(tmp_path, monkeypatch):
    """Another process (no shared memory) minted the same id in the folder:
    its reservation, plan file or context keeps this one from reusing it."""
    from mace.workflow import context
    from mace.workflow.context import new_workflow_id
    # ids already taken in the folder, three ways
    (tmp_path / "workflow_configs").mkdir()
    (tmp_path / "workflow_configs" / "workflow_plan_20991231_000000_aaaa.json").write_text("{}")
    (tmp_path / ".mace_context_workflow_20991231_000000_bbbb").mkdir()
    _forced_suffixes(monkeypatch, ["aaaa", "bbbb", "cccc", "cccc", "dddd", "eeee"])
    first = new_workflow_id(tmp_path)
    assert first == "workflow_20991231_000000_cccc"
    # a fresh process: nothing remembered in memory, the reservation file holds
    monkeypatch.setattr(context, "_ISSUED_IDS", set())
    assert new_workflow_id(tmp_path) == "workflow_20991231_000000_dddd"
    # a folder that is no workflow folder gets nothing written
    plain = tmp_path / "plain"
    plain.mkdir()
    new_workflow_id(plain)
    assert list(plain.iterdir()) == []
    # old ids keep resolving: the reservation is outside the plan-file pattern
    assert sorted(p.name for p in (tmp_path / "workflow_configs").glob("workflow_plan_*.json")) == [
        "workflow_plan_20991231_000000_aaaa.json"]


def test_plans_built_in_one_second_do_not_collide(tmp_path, monkeypatch):
    from mace.workflow import manual_progress
    monkeypatch.chdir(tmp_path)
    deck = tmp_path / "a_opt.d12"
    deck.write_text(OPT_D12)
    made = [manual_progress.build_progress_plan(tmp_path, "materials.db", [deck], "opt_sp")
            for _ in range(3)]
    ids = [wid for _, wid, _ in made]
    assert len(set(ids)) == 3, ids
    plans = sorted((tmp_path / "workflow_configs").glob("workflow_plan_*.json"))
    assert len(plans) == 3
    for plan_file, wid, _ in made:
        assert plan_file.name == f"workflow_plan_{wid[len('workflow_'):]}.json"
        assert json.loads(plan_file.read_text())["workflow_id"] == wid


def test_burst_of_submit_progress_runs(tmp_path):
    """Three `mace submit --progress` runs started together in one folder:
    three plans, three contexts, each submission stamped with its own id, and
    the engine finds each plan by its id."""
    fake = FakeSlurm(tmp_path)
    env = _env(tmp_path, fake)
    decks = tmp_path / "decks"
    decks.mkdir()
    for n in range(3):
        (decks / f"m{n}_opt.d12").write_text(OPT_D12)
    procs = [subprocess.Popen([sys.executable, str(MACE_CLI), "--no-banner", "submit",
                               "--progress", "opt_sp", f"decks/m{n}_opt.d12"],
                              cwd=str(tmp_path), env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) for n in range(3)]
    for p in procs:
        out, _ = p.communicate(timeout=600)
        assert p.returncode == 0, out[-4000:]
    plans = list((decks / "workflow_configs").glob("workflow_plan_*.json"))
    ids = {json.loads(p.read_text())["workflow_id"] for p in plans}
    assert len(plans) == 3 and len(ids) == 3, [p.name for p in plans]
    assert {c.name for c in decks.glob(".mace_context_*")} == {f".mace_context_{i}" for i in ids}

    from mace.workflow.engine import WorkflowEngine
    eng = WorkflowEngine.__new__(WorkflowEngine)
    eng.base_work_dir = decks
    for wid in ids:
        assert eng._load_workflow_plan(wid)["workflow_id"] == wid


def test_quick_start_plan_file_and_execution_share_one_id(tmp_path):
    """Quick-start saves its plan without an id and executes it: the plan
    file's name and the execution must use one id (they agreed before only
    when both were minted in the same second)."""
    from test_array_default_path_golden import _mace
    fake = FakeSlurm(tmp_path)
    env = _env(tmp_path, fake)
    (tmp_path / "decks").mkdir()
    (tmp_path / "decks" / "a_opt.d12").write_text(OPT_D12)
    p = _mace(tmp_path, env, "workflow", "--quick-start", "--d12-dir", "decks",
              "--workflow", "opt_sp")
    assert p.returncode == 0, p.stdout[-4000:]
    (plan,) = (tmp_path / "workflow_configs").glob("workflow_plan_*.json")
    (ctx,) = tmp_path.glob(".mace_context_*")
    wid = "workflow_" + plan.stem[len("workflow_plan_"):]
    assert ID.match(wid) and ctx.name == f".mace_context_{wid}"
