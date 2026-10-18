"""The Workflows view finds every workflow up to two levels down, however
many files the folders around it hold (1.1.6 checkpoint review): the scan's
cap counts directories only, and every folder of a level is checked for
plans before the cap stops the walk."""
from pathlib import Path

from mace.gui import workflow as gw

PLAN = "workflow_plan_20261015_000000_abcd.json"


def _plan(folder: Path) -> Path:
    (folder / "workflow_configs").mkdir(parents=True)
    (folder / "workflow_configs" / PLAN).write_text("{}")
    return folder


def _files(folder: Path, n: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"calc_{i:05d}.out").write_text("")


def test_a_sibling_folder_of_many_files_hides_no_workflow(tmp_path):
    _files(tmp_path / "a_calcs", gw.SCAN_MAX_DIRS + 100)
    wf = _plan(tmp_path / "b_wf")
    assert gw.workflow_folders(tmp_path) == [wf]


def test_a_served_folder_of_many_files_hides_no_workflow(tmp_path):
    _files(tmp_path, gw.SCAN_MAX_DIRS + 100)
    wf = _plan(tmp_path / "z_wf")
    deeper = _plan(tmp_path / "z_more" / "inner")
    assert gw.workflow_folders(tmp_path) == [wf, deeper]


def test_the_cap_still_bounds_the_walk_but_checks_the_level_it_reached(tmp_path, monkeypatch):
    monkeypatch.setattr(gw, "SCAN_MAX_DIRS", 5)
    for i in range(8):
        (tmp_path / f"d{i}").mkdir()
    last = _plan(tmp_path / "d7")
    # a folder past the cap one level further down is not looked at
    _plan(tmp_path / "d0" / "deep")
    found = gw.workflow_folders(tmp_path)
    assert last in found and tmp_path / "d0" / "deep" not in found
