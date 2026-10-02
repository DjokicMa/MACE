"""opt2d3 --config-file finds the shipped example configs, and fails loudly.

A bare example name ("band_auto_everything.json") was looked up in the
working directory only, and a harness that builds
"<code>/Crystal_d3/example_configs/<name>" from a RELATIVE code path, then runs
opt2d3 in the job directory, pointed at nothing. Both printed "Failed to load
configuration" and exited 0, so a caller saw success and no deck.

The parent is a real CRYSTAL23 slab output (graphene, layer group 80) from
tests/data/low_dim_groups, with a placeholder wavefunction file.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from d3_config import resolve_d3_config_path

CONFIGS = REPO_ROOT / "Crystal_d3" / "example_configs"
PARENT = REPO_ROOT / "tests" / "data" / "low_dim_groups" / "graphene_lg80"
NAME = "band_auto_everything.json"


def test_bare_example_name_is_found_from_any_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_d3_config_path(NAME) == str(CONFIGS / NAME)


def test_relative_example_configs_path_from_another_directory_is_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    given = os.path.join("x", "Crystal_d3", "example_configs", NAME)
    assert not os.path.exists(given)
    assert resolve_d3_config_path(given) == str(CONFIGS / NAME)


def test_an_existing_relative_path_is_the_file_used(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mine").mkdir()
    (tmp_path / "mine" / NAME).write_text("{}")
    got = resolve_d3_config_path(os.path.join("mine", NAME))
    assert got == str(tmp_path / "mine" / NAME)


def test_other_missing_paths_are_not_reduced_to_their_file_name(tmp_path, monkeypatch):
    """Only example_configs/<name> is mapped; elsewhere/<name> could be a
    different file than the shipped one."""
    monkeypatch.chdir(tmp_path)
    given = os.path.join("elsewhere", NAME)
    assert resolve_d3_config_path(given) == given
    assert resolve_d3_config_path("nope.json") == "nope.json"


def _job_dir(tmp_path, tag):
    d = tmp_path / tag
    d.mkdir()
    for suffix in (".out", ".d12"):
        shutil.copy(PARENT.with_suffix(suffix), d)
    (d / "graphene_lg80.f9").write_text("placeholder wavefunction\n")
    return d


def _opt2d3(cwd, *args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d3", "--calc-type", "BAND", *args],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, cwd=cwd)


@pytest.mark.parametrize("single", [True, False], ids=["one-file", "directory"])
def test_missing_config_exits_non_zero(tmp_path, single):
    d = _job_dir(tmp_path, "missing")
    args = ["--input", "graphene_lg80.out", "--output-dir", "."] if single else []
    run = _opt2d3(d, *args, "--config-file", "no_such_config.json")
    assert run.returncode != 0, run.stdout + run.stderr
    assert "Failed to load configuration" in run.stdout + run.stderr
    assert not list(d.glob("*.d3"))


def test_bare_name_writes_the_same_deck_as_the_absolute_path(tmp_path):
    bare = _job_dir(tmp_path, "bare")
    absolute = _job_dir(tmp_path, "absolute")
    run = _opt2d3(bare, "--input", "graphene_lg80.out", "--output-dir", ".",
                  "--config-file", NAME)
    assert run.returncode == 0, run.stdout + run.stderr
    assert str(CONFIGS / NAME) in run.stdout          # says which file it loaded
    run = _opt2d3(absolute, "--input", "graphene_lg80.out", "--output-dir", ".",
                  "--config-file", str(CONFIGS / NAME))
    assert run.returncode == 0, run.stdout + run.stderr
    deck = Path(bare / "graphene_lg80_band.d3").read_bytes()
    assert deck == Path(absolute / "graphene_lg80_band.d3").read_bytes()
