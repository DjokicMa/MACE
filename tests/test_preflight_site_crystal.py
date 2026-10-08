"""`mace preflight` tests the CRYSTAL build the job scripts and `mace run` use.

Its module probe hard-coded `module purge &&` and ignored the site's
``[modules] purge = false``, so where a login module decides which CRYSTAL a
module load gives (here PrgEnv-cray), preflight probed another build than
`mace run` (local_run.crystal_root) and the job script. With ``[crystal]
bin_dir`` set but no serial ``crystal`` there, it silently looked elsewhere.

A fake ``module`` function (in the login shell's profile) records every call;
fake CRYSTAL installs record which binary ran.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"
DECK = """MgO
CRYSTAL
0 0 0
225
4.21
2
12 0.0 0.0 0.0
8 0.5 0.5 0.5
END
BASISSET
POB-TZVP-REV2
DFT
PBE0
ENDDFT
SHRINK
8 8
END
"""


def _world(tmp_path, site_toml):
    """Two CRYSTAL installs: a module load gives crysCray when PrgEnv-cray is
    loaded at login (a purge unloads it), else crysGnu."""
    for build in ("crysCray", "crysGnu"):
        b = tmp_path / build / "bin"
        b.mkdir(parents=True)
        for exe in ("crystal", "Pcrystal", "Pproperties"):
            p = b / exe
            p.write_text(f'#!/bin/bash\necho "{build}/{exe}" >> "{tmp_path}/runs.log"\n'
                         'cat >/dev/null\necho " TESTPDIM  END - TERMINATION"\n')
            p.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    (home / ".bash_profile").write_text(
        "module() {\n"
        f'  echo "$*" >> "{tmp_path}/module_calls.log"\n'
        '  case "$1" in\n'
        "    purge) unset EBROOTCRYSTAL; LOADEDMODULES= ;;\n"
        '    load) case "$2" in CRYSTAL/*) case ":$LOADEDMODULES:" in\n'
        f'            *:PrgEnv-cray:*) export EBROOTCRYSTAL="{tmp_path}/crysCray" ;;\n'
        f'            *) export EBROOTCRYSTAL="{tmp_path}/crysGnu" ;; esac ;; esac\n'
        '          export LOADEDMODULES="$LOADEDMODULES:$2" ;;\n'
        "  esac\n"
        "}\n"
        "export -f module\n")
    (tmp_path / "site.toml").write_text(site_toml)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SLURM_", "MACE_", "LMOD", "XDG_"))
           and k not in ("PYTHONPATH", "EBROOTCRYSTAL", "BASH_ENV", "MODULEPATH", "MODULESHOME",
                         "COLORTERM", "TERM", "NO_COLOR")}
    env.update(HOME=str(home), PATH="/usr/bin:/bin", LOADEDMODULES="PrgEnv-cray",
               MACE_SITE_CONFIG=str(tmp_path / "site.toml"), MACE_SITE_GROUP_CONFIG="none",
               MACE_CRYSTAL_MODULE="CRYSTAL/23", XDG_CONFIG_HOME=str(tmp_path / "xdg"),
               XDG_CACHE_HOME=str(tmp_path / "cache"), MACE_NO_BANNER="1")
    return env


def _probes(env, cwd):
    code = ("import sys; sys.path.insert(0, %r)\n"
            "from mace.submission import local_run, preflight\n"
            "print(local_run.crystal_root()[1]); print(preflight._module_probe())" % str(REPO_ROOT))
    r = subprocess.run([sys.executable, "-c", code], env=env, cwd=cwd, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


def _preflight(env, cwd, *extra):
    (cwd / "mgo.d12").write_text(DECK)
    return subprocess.run([sys.executable, str(MACE_CLI), "preflight", "mgo.d12", "-j", "1", *extra],
                          env=env, cwd=cwd, capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize("purge, build", [("false", "crysCray"), ("true", "crysGnu")])
def test_preflight_probes_the_build_mace_run_uses(tmp_path, purge, build):
    env = _world(tmp_path, f"[modules]\npurge = {purge}\n")
    run_root, probe_root = _probes(env, tmp_path)
    assert run_root == probe_root == str(tmp_path / build)
    calls = (tmp_path / "module_calls.log").read_text().splitlines()
    assert ("purge" in calls) == (purge == "true"), calls

    r = _preflight(env, tmp_path)
    assert f"Using {tmp_path / build / 'bin' / 'crystal'}" in r.stdout + r.stderr, r.stdout + r.stderr
    assert (tmp_path / "runs.log").read_text().split() == [f"{build}/crystal"]


def test_bin_dir_mode_runs_no_module_probe(tmp_path):
    env = _world(tmp_path, f'[crystal]\nbin_dir = "{tmp_path / "crysGnu" / "bin"}"\n')
    env.pop("MACE_CRYSTAL_MODULE")
    assert _probes(env, tmp_path)[1] == "None"
    assert not (tmp_path / "module_calls.log").exists()
    r = _preflight(env, tmp_path)
    assert f"Using {tmp_path / 'crysGnu' / 'bin' / 'crystal'}" in r.stdout + r.stderr


def test_bin_dir_without_a_serial_crystal_says_so(tmp_path):
    env = _world(tmp_path, f'[crystal]\nbin_dir = "{tmp_path / "crysGnu" / "bin"}"\n')
    env.pop("MACE_CRYSTAL_MODULE")
    (tmp_path / "crysGnu" / "bin" / "crystal").unlink()
    # a crystal elsewhere (EBROOTCRYSTAL, PATH) is another build: not used
    env["EBROOTCRYSTAL"] = str(tmp_path / "crysCray")
    env["PATH"] = f"{tmp_path / 'crysCray' / 'bin'}:{env['PATH']}"
    r = _preflight(env, tmp_path)
    out = " ".join((r.stdout + r.stderr).split())
    assert r.returncode == 2, out
    assert f"[crystal] bin_dir is {tmp_path / 'crysGnu' / 'bin'}, but it has no executable serial 'crystal'" in out
    assert not (tmp_path / "runs.log").exists()
    assert not (tmp_path / "module_calls.log").exists()


def test_gui_fast_path_follows_bin_dir_and_names_the_problem(tmp_path, monkeypatch):
    env = _world(tmp_path, f'[crystal]\nbin_dir = "{tmp_path / "crysGnu" / "bin"}"\n')
    for k in list(os.environ):
        if k.startswith(("MACE_", "XDG_")) or k == "EBROOTCRYSTAL":
            monkeypatch.delenv(k, raising=False)
    for k in ("HOME", "MACE_SITE_CONFIG", "MACE_SITE_GROUP_CONFIG", "XDG_CONFIG_HOME", "XDG_CACHE_HOME"):
        monkeypatch.setenv(k, env[k])
    monkeypatch.setenv("EBROOTCRYSTAL", str(tmp_path / "crysCray"))
    from mace.submission import site
    site._MEMO.clear()
    from mace.gui import preflight as gpre
    from mace.submission import preflight as pf
    assert gpre.fast_crystal() == str(tmp_path / "crysGnu" / "bin" / "crystal")
    (tmp_path / "crysGnu" / "bin" / "crystal").unlink()
    site._MEMO.clear()
    assert gpre.fast_crystal() is None
    assert pf.find_crystal_binary() is None
    assert "has no executable serial 'crystal'" in pf.site_bin_dir_problem()
    site._MEMO.clear()
    # the Preflight panel names the problem instead of "CRYSTAL not found"
    from mace.gui import clirun
    monkeypatch.setattr(clirun, "run", lambda *a, **k: clirun.Result(0, "structure-only\n", "", 0.1, 0))
    work = tmp_path / "work"
    work.mkdir()
    r = gpre.run("mgo.d12", DECK, work)
    site._MEMO.clear()
    assert r["status"] == "static"
    assert r["problems"][0]["severity"] == "warning"
    assert "has no executable serial 'crystal'" in r["problems"][0]["message"]
