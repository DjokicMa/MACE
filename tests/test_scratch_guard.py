"""The job scripts' scratch guard: an empty or unusable $SCRATCH must not turn
into CRYSTAL reading an empty INPUT ("END OF DATA IN INPUT DECK", seen on
agx-000 where $SCRATCH was empty inside the job).

The job scripts are generated the way MACE generates them - by running the
submitcrystal23.sh / submit_prop.sh generators - and then run with bash, with
`mpirun` replaced by a stub that records what CRYSTAL would have read.
"""
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from mace.recovery import opt_restart

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = {"crystal": REPO / "mace" / "submission" / "submitcrystal23.sh",
             "prop": REPO / "mace" / "submission" / "submit_prop.sh"}
ROOT_USER = hasattr(os, "geteuid") and os.geteuid() == 0

DECK = ("diamond\nCRYSTAL\n0 0 0\n227\n3.567\n1\n6 0.125 0.125 0.125\nEND\n"
        "6 0\nEND\nDFT\nPBE\nEND\nSHRINK\n4 4\nEND\n")

MPIRUN_STUB = """#!/bin/bash
# Stand-in for mpirun: record where CRYSTAL would run and what it would read.
{ echo "CWD=$(pwd)"; echo "INPUT_BYTES=$(wc -c < INPUT 2>/dev/null || echo missing)"; } > "$DIR/$JOB.stub"
echo "stub crystal run" > fort.9
"""


def _generate(tmp_path: Path, kind: str, job: str) -> Path:
    """Run the generator like the queue manager does (`bash <template> <job>`)."""
    subprocess.run(["bash", str(TEMPLATES[kind]), job], cwd=tmp_path, check=True,
                   capture_output=True, text=True)
    script = tmp_path / f"{job}.sh"
    assert script.is_file()
    return script


def _run(tmp_path: Path, script: Path, env_extra: dict, drop=("SCRATCH", "TMPDIR")):
    stubs = tmp_path / "stubs"
    stubs.mkdir(exist_ok=True)
    mpirun = stubs / "mpirun"
    mpirun.write_text(MPIRUN_STUB)
    mpirun.chmod(mpirun.stat().st_mode | stat.S_IEXEC)
    # A minimal PATH: no mace_cli / python queue manager gets picked up.
    env = {"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp_path),
           "SLURM_SUBMIT_DIR": str(script.parent), "SLURM_NTASKS": "1",
           "EBROOTCRYSTAL": str(tmp_path)}
    env.update(env_extra)
    for k in drop:
        if k not in env_extra:
            env.pop(k, None)
    return subprocess.run(["bash", str(script)], cwd=script.parent, env=env,
                          capture_output=True, text=True, timeout=60)


def _stub(submit: Path, job: str) -> dict:
    f = submit / f"{job}.stub"
    if not f.is_file():
        return {}
    return dict(l.split("=", 1) for l in f.read_text().splitlines() if "=" in l)


@pytest.fixture
def submit(tmp_path):
    d = tmp_path / "submit"
    d.mkdir()
    (d / "diamond.d12").write_text(DECK)
    return d


NO_HPCC_USER = "mace_no_such_user_zz"      # /mnt/scratch/<this> never exists


def test_scratch_set_and_writable_is_used_as_before(tmp_path, submit):
    scratch = tmp_path / "scr"
    scratch.mkdir()
    script = _generate(submit, "crystal", "diamond")
    r = _run(tmp_path, script, {"SCRATCH": str(scratch), "USER": NO_HPCC_USER})
    assert r.returncode == 0, r.stdout + r.stderr
    job_dir = scratch / "crys23" / "diamond"
    assert _stub(submit, "diamond")["CWD"] == str(job_dir)
    assert int(_stub(submit, "diamond")["INPUT_BYTES"]) == len(DECK)
    assert "instead of" not in r.stdout
    assert (submit / ".diamond.scratch").read_text().strip() == str(job_dir)
    assert (submit / "diamond.f9").is_file()


@pytest.mark.parametrize("how", ["empty", "unset"])
def test_empty_scratch_falls_back_to_the_submit_directory(tmp_path, submit, how):
    script = _generate(submit, "crystal", "diamond")
    env = {"USER": NO_HPCC_USER}
    if how == "empty":
        env["SCRATCH"] = ""
    r = _run(tmp_path, script, env)
    assert r.returncode == 0, r.stdout + r.stderr
    job_dir = submit / ".mace_scratch" / "crys23" / "diamond"
    stub = _stub(submit, "diamond")
    assert stub["CWD"] == str(job_dir)                      # never /crys23
    assert int(stub["INPUT_BYTES"]) == len(DECK)            # CRYSTAL gets the deck
    assert "$SCRATCH is empty" in r.stdout
    assert f"using {submit}/.mace_scratch/crys23/diamond instead of /crys23/diamond" in r.stdout
    assert (submit / ".diamond.scratch").read_text().strip() == str(job_dir)


def test_empty_scratch_prefers_the_hpcc_scratch_root(tmp_path, submit, monkeypatch):
    """/mnt/scratch/$USER is what MSU HPCC sets $SCRATCH to; it is tried first.
    Exercised with a real user dir only where it exists (on HPCC)."""
    user = os.environ.get("USER") or ""
    root = Path("/mnt/scratch") / user
    if not user or not root.is_dir() or not os.access(root, os.W_OK):
        pytest.skip("no writable /mnt/scratch/$USER here (not on HPCC)")
    job = f"mace_guard_test_{os.getpid()}"
    (submit / f"{job}.d12").write_text(DECK)
    script = _generate(submit, "crystal", job)
    r = _run(tmp_path, script, {"SCRATCH": "", "USER": user})
    try:
        assert r.returncode == 0, r.stdout + r.stderr
        assert _stub(submit, job)["CWD"] == str(root.resolve() / "crys23" / job)
    finally:
        shutil.rmtree(root.resolve() / "crys23" / job, ignore_errors=True)


@pytest.mark.skipif(ROOT_USER, reason="root can write anywhere")
def test_unwritable_scratch_falls_back(tmp_path, submit):
    scratch = tmp_path / "readonly"
    scratch.mkdir()
    scratch.chmod(0o555)
    try:
        script = _generate(submit, "crystal", "diamond")
        r = _run(tmp_path, script, {"SCRATCH": str(scratch), "USER": NO_HPCC_USER})
    finally:
        scratch.chmod(0o755)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"cannot write to {scratch}/crys23/diamond" in r.stdout
    assert _stub(submit, "diamond")["CWD"] == str(submit / ".mace_scratch" / "crys23" / "diamond")


@pytest.mark.skipif(ROOT_USER, reason="root can write anywhere")
def test_tmpdir_is_the_last_resort(tmp_path, submit):
    script = _generate(submit, "crystal", "diamond")
    tmpdir = tmp_path / "nodelocal"
    tmpdir.mkdir()
    submit.chmod(0o555)                        # submit dir not writable either
    try:
        r = _run(tmp_path, script, {"SCRATCH": "", "USER": NO_HPCC_USER,
                                    "TMPDIR": str(tmpdir)})
    finally:
        submit.chmod(0o755)
    assert f"using {tmpdir}/crys23/diamond" in r.stdout, r.stdout + r.stderr
    assert (tmpdir / "crys23" / "diamond" / "INPUT").read_text() == DECK


@pytest.mark.skipif(ROOT_USER, reason="root can write anywhere")
def test_nothing_writable_fails_fast_without_running_crystal(tmp_path, submit):
    script = _generate(submit, "crystal", "diamond")
    submit.chmod(0o555)
    try:
        r = _run(tmp_path, script, {"SCRATCH": "", "USER": NO_HPCC_USER})
    finally:
        submit.chmod(0o755)
    assert r.returncode == 1
    assert "ERROR: no writable scratch directory" in r.stdout
    assert "Not running CRYSTAL" in r.stdout
    assert not (submit / "diamond.stub").exists()          # mpirun never ran


def test_properties_script_keeps_its_prop_subdirectory(tmp_path, submit):
    (submit / "dia_band.d3").write_text("BAND\nEND\n")
    (submit / "dia_band.f9").write_text("f9")
    script = _generate(submit, "prop", "dia_band")
    r = _run(tmp_path, script, {"SCRATCH": "", "USER": NO_HPCC_USER})
    assert r.returncode == 0, r.stdout + r.stderr
    assert _stub(submit, "dia_band")["CWD"] == str(submit / ".mace_scratch" / "crys23" / "prop" / "dia_band")
    assert int(_stub(submit, "dia_band")["INPUT_BYTES"]) > 0


def test_restart_finds_optinfo_left_in_the_previous_runs_directory(tmp_path, submit):
    """A killed run on a node that needed the fallback left OPTINFO.DAT under
    the submit directory; the rerun lands on a healthy node with $SCRATCH set.
    RESTART must still see the optimization history."""
    prev = submit / ".mace_scratch" / "crys23" / "diamond"
    prev.mkdir(parents=True)
    (prev / "OPTINFO.DAT").write_text("optinfo")
    (prev / "fort.20").write_text("density")
    (submit / ".diamond.scratch").write_text(f"{prev}\n")
    deck = DECK.replace("END\nDFT", "OPTGEOM\nRESTART\nENDOPT\nEND\nDFT", 1)
    (submit / "diamond.d12").write_text(deck)
    scratch = tmp_path / "scr"
    scratch.mkdir()
    script = _generate(submit, "crystal", "diamond")
    r = _run(tmp_path, script, {"SCRATCH": str(scratch), "USER": NO_HPCC_USER})
    assert r.returncode == 0, r.stdout + r.stderr
    new = scratch / "crys23" / "diamond"
    assert (new / "OPTINFO.DAT").read_text() == "optinfo"
    assert "RESTART: continuing the optimization from OPTINFO.DAT" in r.stdout
    assert "RESTART" in (new / "INPUT").read_text()         # kept, not dropped
    assert (submit / ".diamond.scratch").read_text().strip() == str(new)


def test_recovery_looks_where_the_job_ran(tmp_path, submit, monkeypatch):
    """The recovery (opt_restart.job_scratch_dir) must find OPTINFO.DAT in the
    directory the guard chose, not in "/crys23" or nowhere."""
    script = _generate(submit, "crystal", "diamond")
    r = _run(tmp_path, script, {"SCRATCH": "", "USER": NO_HPCC_USER})
    assert r.returncode == 0, r.stdout + r.stderr
    job_dir = Path(_stub(submit, "diamond")["CWD"])
    (job_dir / "OPTINFO.DAT").write_text("x")
    monkeypatch.setenv("SCRATCH", "")
    text = script.read_text()
    found = opt_restart.job_scratch_dir(text, "diamond", submit_dir=submit)
    assert found == job_dir
    assert opt_restart.optinfo_state(found) == "present"


def test_recovery_rebuilds_an_empty_scratch_like_the_job(tmp_path, monkeypatch):
    root = tmp_path / "mnt_scratch"
    (root / "someone").mkdir(parents=True)
    monkeypatch.setattr(opt_restart, "HPCC_SCRATCH_ROOT", root)
    monkeypatch.setenv("USER", "someone")
    monkeypatch.setenv("SCRATCH", "")
    script = "export JOB=mat_opt\nexport scratch=$SCRATCH/crys23\n"
    assert opt_restart.job_scratch_dir(script, "mat_opt") == \
        (root / "someone").resolve() / "crys23" / "mat_opt"


def test_guard_is_the_same_in_both_templates_and_safe_inside_the_echo():
    blocks = []
    for path in TEMPLATES.values():
        text = path.read_text()
        start = text.index("# SCRATCH GUARD")
        end = text.index('> "$DIR/.$JOB.scratch" 2>/dev/null\n', start)
        block = text[start:end]
        # The template body is one single-quoted echo, and the workflow
        # executor replaces every "$1" in it with the material name.
        assert "'" not in block
        assert "$1" not in block
        blocks.append(block)
        # The recovery and the workflow engine read/rewrite the first
        # `export scratch=$SCRATCH/...` line; it must still be the first one.
        first = next(l for l in text.splitlines() if l.lstrip().startswith(("export scratch=", "scratch=")))
        assert first.startswith("export scratch=$SCRATCH/crys23")
    assert blocks[0] == blocks[1]


def test_workflow_copies_carry_the_guard(tmp_path):
    """The workflow executor builds its per-step scripts from the same
    generators; its customization must leave the guard intact."""
    from mace.workflow.executor import WorkflowExecutor
    ex = WorkflowExecutor.__new__(WorkflowExecutor)
    ex.work_dir = tmp_path
    out = ex.customize_slurm_script(TEMPLATES["crystal"].read_text(), "wf_1", "OPT", 1,
                                    "diamond", tmp_path)
    assert "# SCRATCH GUARD" in out
    assert "export scratch=$SCRATCH/wf_1/step_001_OPT" in out
    gen = tmp_path / "gen.sh"
    gen.write_text(out)
    (tmp_path / "run").mkdir()
    subprocess.run(["bash", str(gen)], cwd=tmp_path / "run", check=True, capture_output=True)
    assert subprocess.run(["bash", "-n", str(tmp_path / "run" / "diamond.sh")]).returncode == 0
