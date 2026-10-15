"""The temporary generators of `mace submit -i/--walltime/--exclude`
(crystal.py's submitcrystal23_custom_<pid>.sh, properties.py's
submit_prop_custom_<pid>.sh) are written to a private temporary folder, not
into the install's mace/submission/: a group or site-wide install is
read-only to its users, a checkout was littered, and two runs in one process
wrote the same file. The job scripts they write stay byte for byte the same,
pinned to the install as before."""
import importlib
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from test_site_jobscript import _deck, _env, _mace

SUB = REPO_ROOT / "mace" / "submission"
KINDS = [("mace.submission.crystal", "submitcrystal23.sh", "get_default_d12_resources"),
         ("mace.submission.properties", "submit_prop.sh", "get_default_d3_resources")]


def _module(name):
    return importlib.import_module(name)


def _resources(mod, getter, walltime="0:30:00"):
    res = dict(getattr(mod, getter)())
    res["walltime"] = walltime
    return res


def _run(generator, cwd, job="m"):
    """Run a generator as MACE does, without MACE_HOME (the pin must come from
    the install, not the environment); returns the job script's bytes."""
    cwd.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("MACE_HOME", "PYTHONPATH", "BASH_ENV")}
    subprocess.run(["bash", str(generator), job, "100"], cwd=cwd, env=env, check=True, capture_output=True)
    return (cwd / f"{job}.sh").read_bytes()


def _fake_install(tmp, template):
    """A minimal install (its generator in <root>/mace/submission/, the queue
    manager the pin looks for) whose submission folder is read-only."""
    root = tmp / "install"
    (root / "mace" / "queue").mkdir(parents=True)
    (root / "mace" / "queue" / "manager.py").write_text("")
    sub = root / "mace" / "submission"
    sub.mkdir()
    shutil.copy2(SUB / template, sub / template)
    sub.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    return root, sub / template


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a non-root POSIX user")
@pytest.mark.parametrize("modname,template,getter", KINDS)
def test_custom_generator_from_a_read_only_install(tmp_path, modname, template, getter):
    mod = _module(modname)
    root, shipped = _fake_install(tmp_path, template)
    try:
        custom = mod.create_custom_slurm_script(shipped, _resources(mod, getter))
        try:
            assert not custom.is_relative_to(root) and not custom.is_relative_to(REPO_ROOT)
            out = _run(custom, tmp_path / "job")
        finally:
            mod.remove_custom_slurm_script(custom)
        assert not custom.exists() and not custom.parent.exists()
        pin = [ln for ln in out.decode().splitlines() if ln.startswith("MACE_SUBMIT_HOME=")]
        assert pin == [f"MACE_SUBMIT_HOME='{root}'"]
        # byte for byte what the same generator writes from inside the install
        shipped.parent.chmod(0o755)
        beside = shipped.parent / custom.name
        custom2 = mod.create_custom_slurm_script(shipped, _resources(mod, getter))
        try:
            custom_text = custom2.read_text()
        finally:
            mod.remove_custom_slurm_script(custom2)
        text = shipped.read_text()
        gen_line = next(ln for ln in text.splitlines() if ln.startswith("MACE_GEN_DIR="))
        beside.write_text("\n".join(gen_line if ln.startswith("MACE_GEN_DIR=") else ln
                                    for ln in custom_text.split("\n")))
        assert _run(beside, tmp_path / "beside") == out
    finally:
        shipped.parent.chmod(0o755)


@pytest.mark.parametrize("modname,template,getter", KINDS)
def test_two_custom_generators_in_one_process_do_not_collide(tmp_path, modname, template, getter):
    mod = _module(modname)
    a = mod.create_custom_slurm_script(SUB / template, _resources(mod, getter, "0:30:00"))
    b = mod.create_custom_slurm_script(SUB / template, _resources(mod, getter, "1:00:00"))
    try:
        assert a != b and a.exists() and b.exists()
        assert "#SBATCH -t 0:30:00" in a.read_text() and "#SBATCH -t 1:00:00" in b.read_text()
        assert not list(SUB.glob("*_custom_*.sh"))
        mod.remove_custom_slurm_script(a)
        assert not a.exists() and b.exists()
        assert b"#SBATCH -t 1:00:00" in _run(b, tmp_path / "b")
    finally:
        mod.remove_custom_slurm_script(a)
        mod.remove_custom_slurm_script(b)


@pytest.mark.parametrize("bad_site", [False, True])
def test_submit_leaves_no_generator_behind(tmp_path, bad_site):
    """In the user's temp folder (TMPDIR) or the install, after a run that
    wrote its jobs, and after one a site setting stopped (sys.exit(2))."""
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()
    extra = {"TMPDIR": str(tmpdir)}
    if bad_site:
        extra["MACE_CRYSTAL_MODULE"] = "not a module!"
    env = _env(tmp_path, **extra)
    d = tmp_path / "decks"
    _deck(d, "m.d12")
    p = _mace(d, env, "submit", "--nosubmit", "--walltime", "0:30:00", "m.d12")
    if bad_site:
        assert p.returncode == 2, p.stdout + p.stderr
    else:
        assert p.returncode == 0, p.stdout + p.stderr
        assert "#SBATCH -t 0:30:00" in (d / "m.sh").read_text()
        assert f"MACE_SUBMIT_HOME='{REPO_ROOT}'" in (d / "m.sh").read_text()
    assert list(tmpdir.iterdir()) == []
    assert not list(SUB.glob("*_custom_*.sh"))
    assert not [p for p in d.iterdir() if "_custom_" in p.name]
