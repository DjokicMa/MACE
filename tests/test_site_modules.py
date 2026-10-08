"""The CRYSTAL module MACE writes into job scripts (PLAN_submit D29): bounded
to the preferred module's major version, ranked, cached, and the same module
for the job script, `mace run`'s probe and `mace preflight`'s probe.

A fake module system (tests/fake_modules.py: a `module()` function through
BASH_ENV) stands in for Lmod; every module subprocess MACE starts is counted
through MACE_MODULES_SPY."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from fake_modules import FIXTURE_A, FakeModules
from test_array_default_path_golden import SP_D12, BAND_D3

from mace.submission import site
from mace.utils import envmodules

MACE_CLI = REPO_ROOT / "mace_cli"
S8_NAMES = ["CRYSTAL/17-intel-2023a-1.0.2", "CRYSTAL/17-MPP-intel-2023a-1.0.2", "CRYSTAL/23_v1.0.1_binary",
            "CRYSTAL/23-intel-2023a", "CRYSTAL/23-MPP-intel-2023a"]


def _base_env(tmp):
    env = {k: v for k, v in os.environ.items()
           if k not in ("MACE_HOME", "PYTHONPATH", "MACE_SLURM_ACCOUNT", "MACE_SLURM_PARTITION",
                        "MACE_CRYSTAL_MODULE", "MACE_SITE_CONFIG", "LMOD_CMD", "LMOD_VERSION", "MODULESHOME",
                        "MODULEPATH", "LOADEDMODULES", "BASH_ENV", "SLURM_JOB_ID")}
    env.update(XDG_CONFIG_HOME=str(tmp / "xdg"), XDG_CACHE_HOME=str(tmp / "cache"), HOME=str(tmp / "home"),
               MACE_NO_BANNER="1", MACE_MODULES_NO_PROFILE="1")
    (tmp / "home").mkdir(parents=True, exist_ok=True)
    return env


def _submit(tmp, env, deck="m.d12", text=SP_D12, extra=()):
    d = tmp / "decks"
    d.mkdir(exist_ok=True)
    (d / deck).write_text(text)
    p = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "submit", "--nosubmit", "--overwrite-sh",
                        *extra, deck], cwd=d, env=env, capture_output=True, text=True, timeout=300)
    sh = d / (Path(deck).stem + ".sh")
    return p, (sh.read_text() if sh.exists() else None)


def _module_lines(script):
    return [l for l in script.splitlines() if l.startswith(("module load", "# MACE:", "[ -n \"$EBROOTCRYSTAL\" ]"))]


@pytest.fixture(autouse=True)
def _fresh_memo():
    site._MEMO.clear()
    yield
    site._MEMO.clear()


def test_rank_s8_names():
    ranked = envmodules.rank_candidates(S8_NAMES, "CRYSTAL/23-intel-2023a")
    assert ranked[0] == "CRYSTAL/23-intel-2023a"
    plain_23 = [n for n in ranked if envmodules.version_key(n) == (23,)]
    assert plain_23[0] == "CRYSTAL/23-intel-2023a"
    assert set(plain_23[-2:]) == {"CRYSTAL/23_v1.0.1_binary", "CRYSTAL/23-MPP-intel-2023a"}
    assert envmodules.version_key("CRYSTAL/23-intel-2023a") == (23,)
    assert envmodules.version_key("xtb/6.10.0-x") > envmodules.version_key("xtb/6.7.1")


def test_fixture_a_discovers_the_best_23(tmp_path):
    fm = FakeModules(tmp_path / "fm", FIXTURE_A)
    env = fm.env(_base_env(tmp_path))
    p, sh = _submit(tmp_path, env)
    assert p.returncode == 0, p.stdout + p.stderr
    lines = _module_lines(sh)
    assert "module load CRYSTAL/23-intel-2023b" in lines
    assert lines[0].startswith("# MACE: CRYSTAL/23-intel-2023a does not load")
    assert "module load CRYSTAL/23-intel-2023a" not in sh
    assert p.stdout.count("does not load on") == 1           # one notice
    # the probes of `mace run` and `mace preflight` load the same module
    fm.reset_calls()
    code = ("import sys; sys.path.insert(0, %r)\n"
            "from mace.submission import local_run, preflight\n"
            "print(local_run.crystal_root()); print(preflight._module_probe())" % str(REPO_ROOT))
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    root = str(fm.folders["CRYSTAL/23-intel-2023b"])
    assert r.stdout.splitlines() == [str(("module", root)), root]


def test_fixture_b_never_switches_major(tmp_path):
    fm = FakeModules(tmp_path / "fm", {"CRYSTAL/24-intel-x": {"root": True}, "CRYSTAL/17-x": {"root": True}})
    env = fm.env(_base_env(tmp_path))
    p, sh = _submit(tmp_path, env)
    base = _base_env(tmp_path / "plain")
    (tmp_path / "plain").mkdir(exist_ok=True)
    p0, sh0 = _submit(tmp_path / "plain", base)
    assert p.returncode == 0 and sh is not None
    norm = lambda s, t: s.replace(str(t), "<T>")
    assert norm(sh, tmp_path) == norm(sh0, tmp_path / "plain")       # line unchanged
    assert "does not load" not in p.stdout
    d = site.describe(env)
    assert "MACE does not switch major versions" in d["crystal"]["note"]
    assert d["crystal"]["other_major"] == "CRYSTAL/24-intel-x"
    # opt in: crystal_maximum = 24
    site._MEMO.clear()
    (tmp_path / "xdg" / "mace").mkdir(parents=True, exist_ok=True)
    (tmp_path / "xdg" / "mace" / "site.toml").write_text('[modules]\ncrystal_maximum = "24"\n')
    p2, sh2 = _submit(tmp_path, env)
    assert "module load CRYSTAL/24-intel-x" in sh2


def test_fixture_c_only_17_keeps_the_line(tmp_path):
    fm = FakeModules(tmp_path / "fm", {"CRYSTAL/17-x": {"root": True}})
    p, sh = _submit(tmp_path, fm.env(_base_env(tmp_path)))
    assert "module load CRYSTAL/23-intel-2023a" in sh and "CRYSTAL/17" not in sh


def test_fixture_d_spack_style_root_from_path(tmp_path):
    fm = FakeModules(tmp_path / "fm", {"CRYSTAL/23-foss": {"path": True}})
    env = fm.env(_base_env(tmp_path))
    p, sh = _submit(tmp_path, env)
    lines = _module_lines(sh)
    assert "module load CRYSTAL/23-foss" in lines and lines[-1] == site.ROOT_FROM_PATH_LINE or \
        site.ROOT_FROM_PATH_LINE in lines
    i = lines.index("module load CRYSTAL/23-foss")
    assert lines[i + 1] == site.ROOT_FROM_PATH_LINE
    # the generated script's module block, run under the fake modules, finds the binary
    block = "module purge\n" + "\n".join(l for l in lines if not l.startswith("#")) + \
        '\ntest -x "$EBROOTCRYSTAL/bin/Pcrystal" && echo FOUND\n'
    r = subprocess.run(["bash", "-c", block], env=env, capture_output=True, text=True)
    assert "FOUND" in r.stdout, r.stdout + r.stderr


def test_fixture_e_hierarchy_list(tmp_path):
    fm = FakeModules(tmp_path / "fm", {"foss/2023a": {}, "CRYSTAL/23-foss-2023a": {"root": True, "needs": "foss/2023a"}})
    env = fm.env(_base_env(tmp_path))
    (tmp_path / "xdg" / "mace").mkdir(parents=True)
    (tmp_path / "xdg" / "mace" / "site.toml").write_text('[modules]\ncrystal = ["foss/2023a", "CRYSTAL/23-foss-2023a"]\n')
    p, sh = _submit(tmp_path, env)
    assert p.returncode == 0, p.stdout + p.stderr
    assert _module_lines(sh)[:2] == ["module load foss/2023a", "module load CRYSTAL/23-foss-2023a"]
    assert "# MACE:" not in sh


def test_pin_and_none_start_no_module_subprocess(tmp_path):
    fm = FakeModules(tmp_path / "fm", FIXTURE_A)
    p, sh = _submit(tmp_path, fm.env(_base_env(tmp_path), MACE_CRYSTAL_MODULE="CRYSTAL/99-pinned"))
    # no CRYSTAL probe (the Python modules are checked on their own, once)
    crystal_calls = lambda: [c for c in fm.calls() if "CRYSTAL" in c]
    assert "module load CRYSTAL/99-pinned" in sh and crystal_calls() == []
    p, sh = _submit(tmp_path, fm.env(_base_env(tmp_path), MACE_CRYSTAL_MODULE="none"))
    assert "module load CRYSTAL/23-intel-2023a" in sh and crystal_calls() == []


def test_second_run_uses_the_cache_and_a_deleted_root_reprobes(tmp_path):
    fm = FakeModules(tmp_path / "fm", FIXTURE_A)
    env = fm.env(_base_env(tmp_path))
    _submit(tmp_path, env)
    assert fm.calls()
    fm.reset_calls()
    p, sh = _submit(tmp_path, env)
    assert "module load CRYSTAL/23-intel-2023b" in sh and fm.calls() == []      # cache hit: zero probes
    os.unlink(fm.folders["CRYSTAL/23-intel-2023b"] / "bin" / "Pcrystal")
    p, sh = _submit(tmp_path, env)
    assert fm.calls(), "a cached module whose Pcrystal is gone is probed again"
    assert "module load CRYSTAL/23-intel-2023b" not in sh


def test_qualifying_default_module_keeps_bytes_and_stdout(tmp_path):
    fm = FakeModules(tmp_path / "fm", {"CRYSTAL/23-intel-2023a": {"root": True}})
    p, sh = _submit(tmp_path, fm.env(_base_env(tmp_path)))
    (tmp_path / "plain").mkdir()
    p0, sh0 = _submit(tmp_path / "plain", _base_env(tmp_path / "plain"))
    assert sh.replace(str(tmp_path), "<T>") == sh0.replace(str(tmp_path / "plain"), "<T>")
    assert p.stdout.replace(str(tmp_path), "<T>") == p0.stdout.replace(str(tmp_path / "plain"), "<T>")


def test_d3_gets_the_same_module(tmp_path):
    fm = FakeModules(tmp_path / "fm", FIXTURE_A)
    p, sh = _submit(tmp_path, fm.env(_base_env(tmp_path)), deck="m_band.d3", text=BAND_D3)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "module load CRYSTAL/23-intel-2023b" in sh


def test_no_module_system_starts_nothing(tmp_path, monkeypatch):
    env = _base_env(tmp_path)
    env["MACE_MODULES_SPY"] = str(tmp_path / "spy")
    before = envmodules.PROBES[0]
    res = site.crystal_module(env)
    assert res.source == "unchecked" and res.modules == site.DEFAULTS["crystal"]
    assert envmodules.PROBES[0] == before and not (tmp_path / "spy").exists()


def test_preferred_without_a_version_uses_the_built_in_major(tmp_path):
    """HPCC H10: `crystal = "CRYSTAL/does-not-exist"` falls back to a 23.x build."""
    fm = FakeModules(tmp_path / "fm", FIXTURE_A)
    env = fm.env(_base_env(tmp_path))
    (tmp_path / "xdg" / "mace").mkdir(parents=True)
    (tmp_path / "xdg" / "mace" / "site.toml").write_text('[modules]\ncrystal = "CRYSTAL/does-not-exist"\n')
    p, sh = _submit(tmp_path, env)
    assert "module load CRYSTAL/23-intel-2023b" in sh and "does-not-exist" in _module_lines(sh)[0]
