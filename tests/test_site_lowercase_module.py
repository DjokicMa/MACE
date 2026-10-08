"""A cluster whose CRYSTAL modulefile is named in lower case (``crystal/23.1``,
common for hand-written modulefiles): site.toml accepts it, discovery finds
it when the built-in module does not load, and a completion callback inside a
job keeps it. The real `mace_cli` runs as a subprocess on a fake cluster."""
from pathlib import Path

import pytest

from fake_modules import FakeModules
from fake_slurm import FakeSlurm
from test_array_default_path_golden import SP_D12
from test_site_jobscript import _deck, _env, _mace, _toml

from mace.submission import site


@pytest.fixture(autouse=True)
def _fresh():
    site._MEMO.clear()
    site._SLURM_MEMO.clear()
    yield
    site._MEMO.clear()
    site._SLURM_MEMO.clear()


def _cluster(tmp):
    fake = FakeSlurm(tmp / "fs", {"assoc": ["chem_lab"], "default_account": "chem_lab"})
    fm = FakeModules(tmp / "fm", {"crystal/23.1": {"path": True}}, python=False)
    base = _env(tmp, PATH=fake.env["PATH"], FAKE_SLURM_STATE=fake.env["FAKE_SLURM_STATE"],
                FAKE_SLURM_LOG=fake.env["FAKE_SLURM_LOG"], MACE_HOSTNAME="login1.cluster.example.edu")
    return fm.env(base)


def _script(tmp, env):
    d = tmp / "decks"
    _deck(d, "m.d12", SP_D12)
    r = _mace(d, env, "submit", "--nosubmit", "m.d12")
    assert r.returncode == 0, r.stdout + r.stderr
    return (d / "m.sh").read_text(), r


def test_lowercase_module_is_discovered(tmp_path):
    env = _cluster(tmp_path)
    text, r = _script(tmp_path, env)
    assert "module load crystal/23.1" in text
    assert "module load CRYSTAL/" not in text
    assert "using crystal/23.1" in r.stdout + r.stderr


def test_lowercase_module_in_site_toml(tmp_path):
    env = _cluster(tmp_path)
    _toml(tmp_path, '[modules]\ncrystal = "crystal/23.1"\n')
    r = _mace(tmp_path, env, "site")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "crystal/23.1" in r.stdout and "not a valid" not in r.stdout + r.stderr
    text, _ = _script(tmp_path, env)
    assert "module load crystal/23.1" in text


def test_site_toml_still_refuses_a_non_crystal_last_entry(tmp_path):
    env = _cluster(tmp_path)
    _toml(tmp_path, '[modules]\ncrystal = ["crystal/23.1", "foss/2023a"]\n')
    r = _mace(tmp_path, env, "site")
    assert r.returncode == 2
    assert "must start with CRYSTAL/ or crystal/" in r.stdout + r.stderr


def test_callback_inside_a_job_keeps_the_lowercase_module(tmp_path):
    s = site.Site()
    env = {"SLURM_JOB_ID": "77", "LOADEDMODULES": "gcc/12:crystal/23.1"}
    res = site._inherited(s, env, None)
    assert res.modules[-1] == "crystal/23.1"
    assert site.is_crystal_module("CRYSTAL/23-intel-2023a") and site.is_crystal_module("crystal/23.1")
    assert not site.is_crystal_module("foss/2023a")
