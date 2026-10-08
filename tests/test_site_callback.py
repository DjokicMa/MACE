"""Completion callbacks never probe modules (PLAN_submit D29b).

Inside a SLURM job (SLURM_JOB_ID set) the follow-up's job script loads the
CRYSTAL module the parent job loaded (LOADEDMODULES), with zero module
subprocesses even when MODULEPATH differs (a compute node) and the cache
misses; the new calculation row records that module in settings_json. A job
whose .out says ``bin/Pcrystal: No such file or directory`` makes the
callback drop that module's cached choice (a file edit only)."""
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from conftest import REPO_ROOT
from fake_modules import FIXTURE_A, FakeModules
from fake_slurm import FakeSlurm
from test_array_default_path_golden import SP_D12

MANAGER = REPO_ROOT / "mace" / "queue" / "manager.py"


def _env(tmp, fm, fake, **extra):
    base = {k: v for k, v in fake.env.items()
            if not k.startswith(("MACE_SLURM_", "MACE_CRYSTAL", "MACE_SITE")) and k not in ("MACE_HOME", "PYTHONPATH")}
    (tmp / "home").mkdir(parents=True, exist_ok=True)
    base.update(XDG_CONFIG_HOME=str(tmp / "xdg"), XDG_CACHE_HOME=str(tmp / "cache"), HOME=str(tmp / "home"),
                USER="tester", PYTHONPATH=str(REPO_ROOT), MACE_HOME=str(REPO_ROOT))
    return fm.env(base, **extra)


SUBMIT = """
import json, sys
from pathlib import Path
from mace.queue.manager import EnhancedCrystalQueueManager
deck, parent = Path(sys.argv[1]), (sys.argv[2] if len(sys.argv) > 2 else None)
m = EnhancedCrystalQueueManager(d12_dir=str(deck.parent), enable_tracking=True, organize_outputs=False,
                                db_path=str(deck.parent / 'materials.db'))
cid = m.submit_calculation(deck, prerequisite_calc_id=parent)
row = m.db.get_calculation(cid)
print('ROW=' + json.dumps({'calc_id': cid, 'settings': json.loads(row.get('settings_json') or '{}')}))
"""


def _submit(env, deck, parent=None):
    r = subprocess.run([sys.executable, "-c", SUBMIT, str(deck)] + ([parent] if parent else []), env=env,
                       cwd=str(deck.parent), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    line = next(l for l in r.stdout.splitlines() if l.startswith("ROW="))
    return json.loads(line[4:]), r.stdout


def test_callback_uses_the_parents_module_without_probing(tmp_path):
    fm = FakeModules(tmp_path / "fm", FIXTURE_A)
    fake = FakeSlurm(tmp_path / "fs")
    d = tmp_path / "calc"
    d.mkdir()
    (d / "parent_sp.d12").write_text(SP_D12)
    (d / "child_sp.d12").write_text(SP_D12)
    # the login side: a probe resolves 23-intel-2023b and records it
    login = _env(tmp_path, fm, fake)
    parent, _ = _submit(login, d / "parent_sp.d12")
    assert parent["settings"]["crystal_module"] == "CRYSTAL/23-intel-2023b"
    assert fm.calls()
    fm.reset_calls()
    # inside the parent's job on a compute node: another MODULEPATH (cache miss), LOADEDMODULES of the job
    job = _env(tmp_path, fm, fake, SLURM_JOB_ID="1", MODULEPATH="/compute/modulefiles",
               LOADEDMODULES="CRYSTAL/23-intel-2023b:Python/3.11.3-GCCcore-12.3.0")
    child, out = _submit(job, d / "child_sp.d12", parent["calc_id"])
    assert fm.calls() == []                                  # zero module subprocesses
    assert "module load CRYSTAL/23-intel-2023b" in (d / "child_sp.sh").read_text()
    assert child["settings"]["crystal_module"] == parent["settings"]["crystal_module"]
    assert child["settings"]["crystal_module_source"] == "inherited from job 1"
    assert "mixed in this chain" not in out


def test_mixed_chain_is_reported(tmp_path):
    fm = FakeModules(tmp_path / "fm", FIXTURE_A)
    fake = FakeSlurm(tmp_path / "fs")
    d = tmp_path / "calc"
    d.mkdir()
    (d / "parent_sp.d12").write_text(SP_D12)
    (d / "child_sp.d12").write_text(SP_D12)
    parent, _ = _submit(_env(tmp_path, fm, fake), d / "parent_sp.d12")
    job = _env(tmp_path, fm, fake, SLURM_JOB_ID="2", MACE_CRYSTAL_MODULE="CRYSTAL/23-MPP-intel-2023a")
    child, out = _submit(job, d / "child_sp.d12", parent["calc_id"])
    assert ("loads CRYSTAL/23-MPP-intel-2023a, its parent ran with CRYSTAL/23-intel-2023b: "
            "results of different CRYSTAL builds are mixed in this chain.") in out


def test_missing_binary_forgets_the_cached_module(tmp_path):
    fm = FakeModules(tmp_path / "fm", FIXTURE_A)
    fake = FakeSlurm(tmp_path / "fs")
    login = _env(tmp_path, fm, fake)
    code = "from mace.submission import site; print(site.crystal_module().module)"
    r = subprocess.run([sys.executable, "-c", code], env=login, capture_output=True, text=True, timeout=120)
    assert r.stdout.strip() == "CRYSTAL/23-intel-2023b"
    cache = tmp_path / "cache" / "mace" / "modules.json"
    assert "23-intel-2023b" in cache.read_text()
    d = tmp_path / "job"
    d.mkdir()
    (d / "m.out").write_text("bash: /x/bin/Pcrystal: No such file or directory\n")
    fm.reset_calls()
    job = _env(tmp_path, fm, fake, SLURM_JOB_ID="7", JOB="m", DIR=str(d), SLURM_SUBMIT_DIR=str(d),
               LOADEDMODULES="CRYSTAL/23-intel-2023b")
    subprocess.run([sys.executable, str(MANAGER), "--callback-mode", "completion", "--db-path",
                    str(d / "materials.db")], cwd=str(d), env=job, capture_output=True, text=True, timeout=300)
    assert "23-intel-2023b" not in cache.read_text()
    assert fm.calls() == []
