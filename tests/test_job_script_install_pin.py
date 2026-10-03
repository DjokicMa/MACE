"""A job's completion callback runs the MACE install that submitted the job.

Job scripts start with `#!/bin/bash --login`, which re-reads ~/.bashrc. A
`export MACE_HOME=...` there (the form setup_mace.py --add-to-path writes, with
PYTHONPATH prepended) used to decide which manager.py - and which `mace`
package - ran a job's follow-up and recovery steps, so a second install (a
side-by-side release test, a shared group copy) had its callbacks run by the
one the dotfile names, with nothing in the log saying so.

The scripts are generated the way MACE generates them (running the
submitcrystal23.sh / submit_prop.sh generators) and then run under a real
login shell whose profile exports another install, with `python` and `mpirun`
replaced by stubs that record what would have run.
"""
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
GENERATORS = {"crystal": REPO / "mace" / "submission" / "submitcrystal23.sh",
              "prop": REPO / "mace" / "submission" / "submit_prop.sh"}
NO_HPCC_USER = "mace_no_such_user_zz"

PYTHON_STUB = """#!/bin/bash
# Stand-in for the module python: record which queue manager the callback ran.
{ echo "MANAGER=$1"; echo "PYTHONPATH=$PYTHONPATH"; echo "MACE_HOME=$MACE_HOME"; } > "$DIR/$JOB.callback"
"""
MPIRUN_STUB = """#!/bin/bash
echo "stub crystal run" > fort.9
"""


def _fake_install(root: Path) -> Path:
    """Another MACE install: just enough for the callback to find manager.py."""
    (root / "mace" / "queue").mkdir(parents=True)
    (root / "mace" / "queue" / "manager.py").write_text("raise SystemExit('wrong install')\n")
    return root


def _gen_env(**extra):
    env = {k: v for k, v in os.environ.items()
           if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME")}
    env.update(extra)
    return env


def _generate(generator: Path, cwd: Path, job: str, env) -> Path:
    subprocess.run(["bash", str(generator), job], cwd=cwd, check=True,
                   capture_output=True, text=True, env=env)
    script = cwd / f"{job}.sh"
    assert script.is_file()
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    return script


def _run_login(tmp_path: Path, script: Path, login_home: Path):
    """Run the job under `bash --login` with a profile that exports
    `login_home` the way ~/.bashrc does on HPCC."""
    stubs = tmp_path / "stubs"
    stubs.mkdir(exist_ok=True)
    for name, body in (("python", PYTHON_STUB), ("mpirun", MPIRUN_STUB)):
        f = stubs / name
        f.write_text(body)
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    (home / ".bash_profile").write_text(
        f'export PATH="{stubs}:$PATH"\n'
        f'export MACE_HOME={login_home}\n'
        'export PYTHONPATH="$MACE_HOME:$PYTHONPATH"\n')
    scratch = tmp_path / "scr"
    scratch.mkdir(exist_ok=True)
    env = {"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(home),
           "SLURM_SUBMIT_DIR": str(script.parent), "SLURM_NTASKS": "1",
           "EBROOTCRYSTAL": str(tmp_path), "SCRATCH": str(scratch),
           "USER": NO_HPCC_USER}
    r = subprocess.run(["bash", "--login", str(script)], cwd=script.parent, env=env,
                       capture_output=True, text=True, timeout=60)
    rec = script.parent / f"{script.stem}.callback"
    seen = dict(l.split("=", 1) for l in rec.read_text().splitlines()) if rec.is_file() else {}
    return r, seen


@pytest.mark.parametrize("kind,deck", [("crystal", "mat.d12"), ("prop", "mat.d3")])
def test_callback_runs_the_submitting_install_not_the_login_one(tmp_path, kind, deck):
    other = _fake_install(tmp_path / "other_install")
    submit = tmp_path / "submit"
    submit.mkdir()
    (submit / deck).write_text("mat\nEND\n")
    # Generated from this install while the shell names the other one.
    script = _generate(GENERATORS[kind], submit, "mat", _gen_env(MACE_HOME=str(other)))
    r, seen = _run_login(tmp_path, script, other)
    out = r.stdout + r.stderr
    assert seen.get("MANAGER") == str(REPO / "mace" / "queue" / "manager.py"), out
    assert seen["MACE_HOME"] == str(REPO), out
    assert seen["PYTHONPATH"].split(":")[0] == str(REPO), out
    # The log says the login shell's install was overridden.
    assert f"running the submitting install {REPO}" in r.stdout, out
    assert f"MACE_HOME={other}" in r.stdout, out


def test_copied_generator_pins_the_mace_home_of_the_process_running_it(tmp_path):
    """A workflow runs a copy of the generator from workflow_scripts/ or the
    step directory; that copy pins the MACE_HOME of the MACE running it."""
    submitting = _fake_install(tmp_path / "submitting_install")
    login = _fake_install(tmp_path / "login_install")
    wf = tmp_path / "workflow_scripts"
    wf.mkdir()
    copy = wf / "submitcrystal23_opt_1.sh"
    shutil.copy(GENERATORS["crystal"], copy)
    submit = tmp_path / "step"
    submit.mkdir()
    (submit / "mat.d12").write_text("mat\nEND\n")
    script = _generate(copy, submit, "mat", _gen_env(MACE_HOME=str(submitting)))
    assert f"MACE_SUBMIT_HOME='{submitting}'" in script.read_text()
    r, seen = _run_login(tmp_path, script, login)
    assert seen.get("MANAGER") == str(submitting / "mace" / "queue" / "manager.py"), \
        r.stdout + r.stderr


def test_without_a_known_install_the_script_is_unpinned(tmp_path):
    """A copy run with no MACE_HOME has nothing to pin: the job resolves the
    queue manager as before (from the login shell)."""
    login = _fake_install(tmp_path / "login_install")
    copy = tmp_path / "submitcrystal23_copy.sh"
    shutil.copy(GENERATORS["crystal"], copy)
    submit = tmp_path / "step"
    submit.mkdir()
    (submit / "mat.d12").write_text("mat\nEND\n")
    script = _generate(copy, submit, "mat", _gen_env())
    assert "MACE_SUBMIT_HOME=" not in script.read_text()
    r, seen = _run_login(tmp_path, script, login)
    assert seen.get("MANAGER") == str(login / "mace" / "queue" / "manager.py"), \
        r.stdout + r.stderr


def test_a_submitting_install_missing_on_the_node_is_reported(tmp_path):
    """When the pinned install cannot be read on the compute node the job
    keeps the login shell's and says so, rather than finding no manager."""
    gone = _fake_install(tmp_path / "gone_install")
    login = _fake_install(tmp_path / "login_install")
    copy = tmp_path / "submitcrystal23_copy.sh"
    shutil.copy(GENERATORS["crystal"], copy)
    submit = tmp_path / "step"
    submit.mkdir()
    (submit / "mat.d12").write_text("mat\nEND\n")
    script = _generate(copy, submit, "mat", _gen_env(MACE_HOME=str(gone)))
    shutil.rmtree(gone)
    r, seen = _run_login(tmp_path, script, login)
    assert seen.get("MANAGER") == str(login / "mace" / "queue" / "manager.py"), \
        r.stdout + r.stderr
    assert f"submitting install {gone} is not readable" in r.stdout


def test_mace_cli_submit_pins_its_own_install(tmp_path):
    """The real path: `mace submit --nosubmit` from this checkout, with the
    shell's MACE_HOME naming another install, writes a job pinned to this one."""
    other = _fake_install(tmp_path / "other_install")
    (tmp_path / "mat.d12").write_text("mat\nCRYSTAL\nEND\n")
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "submit",
                        str(tmp_path / "mat.d12"), "--nosubmit"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=120,
                       stdin=subprocess.DEVNULL,
                       env=_gen_env(MACE_HOME=str(other)))
    assert r.returncode == 0, r.stdout + r.stderr
    text = (tmp_path / "mat.sh").read_text()
    assert f"MACE_SUBMIT_HOME='{REPO}'" in text


def test_running_mace_exports_its_own_install_to_the_generators(tmp_path, monkeypatch):
    """MACE sets MACE_HOME (and PYTHONPATH) to its own install for what it
    starts, so a workflow's generator copies pin the MACE actually running."""
    from mace.utils.install_home import pin_install_home
    env = {"MACE_HOME": "/somewhere/else", "PYTHONPATH": "/somewhere/else"}
    assert pin_install_home(env) == REPO
    assert env["MACE_HOME"] == str(REPO)
    assert env["PYTHONPATH"] == f"{REPO}:/somewhere/else"
    pin_install_home(env)                       # idempotent
    assert env["PYTHONPATH"] == f"{REPO}:/somewhere/else"
    env = {}
    pin_install_home(env)
    assert env == {"MACE_HOME": str(REPO), "PYTHONPATH": str(REPO)}
