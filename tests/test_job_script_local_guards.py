"""Guards in the generated job scripts for running them outside SLURM.

`mace run` runs the normal job script (from submitcrystal23.sh / submit_prop.sh)
on the node it is on, with bash instead of sbatch. Three blocks in the
generators make that safe, and none of them can fire inside a SLURM job:

* no submit directory: `bash job.sh` outside SLURM read and wrote `/`
  (DIR=$SLURM_SUBMIT_DIR was empty), exited 0 and ran the completion callback
  from $HOME (measured on an HPCC dev node). It now stops with exit 1.
* LOCAL RUN: with MACE_LOCAL_RUN set (by `mace run` only) the script skips the
  completion callback, which would submit follow-up jobs from a dev node.
* serial: with MACE_RUN_SERIAL set the serial crystal/properties binaries run
  instead of mpirun + Pcrystal/Pproperties.

The scripts are generated the way MACE generates them and run with bash, with
stand-ins for mpirun, the CRYSTAL binaries and the callback's python.
"""
import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = {"crystal": REPO / "mace" / "submission" / "submitcrystal23.sh",
             "prop": REPO / "mace" / "submission" / "submit_prop.sh"}

DECK = ("diamond\nCRYSTAL\n0 0 0\n227\n3.567\n1\n6 0.125 0.125 0.125\nEND\n"
        "6 0\nEND\nDFT\nPBE\nEND\nSHRINK\n4 4\nEND\n")

MPIRUN = """#!/bin/bash
echo "mpirun $*" >> "$STUB_LOG"
shift 2
"$@"
"""

# A CRYSTAL binary: records its name and what it read on stdin.
BINARY = """#!/bin/bash
echo "$(basename "$0") stdin=$(wc -c)" >> "$STUB_LOG"
echo "stub fort.9" > fort.9
echo " EEEEEEEEEE TERMINATION  DATE 02 10 2026"
echo "    TOTAL CPU TIME =        1.000"
"""

PYTHON = """#!/bin/bash
echo "python $*" >> "$STUB_LOG"
"""


def _exe(path: Path, text: str):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def env(tmp_path):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    _exe(stubs / "mpirun", MPIRUN)
    _exe(stubs / "python", PYTHON)
    ebroot = tmp_path / "ebroot"
    (ebroot / "bin").mkdir(parents=True)
    for name in ("Pcrystal", "Pproperties", "crystal", "properties"):
        _exe(ebroot / "bin" / name, BINARY)
    home = tmp_path / "home"
    home.mkdir()
    return {"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(home),
            "EBROOTCRYSTAL": str(ebroot), "STUB_LOG": str(tmp_path / "stub.log"),
            "SCRATCH": str(tmp_path / "scratch"), "USER": "mace_no_such_user_zz"}


@pytest.fixture
def submit(tmp_path):
    d = tmp_path / "submit"
    d.mkdir()
    (d / "dia.d12").write_text(DECK)
    (d / "dia_band.d3").write_text("BAND\nEND\n")
    (d / "dia_band.f9").write_text("f9")
    return d


def _generate(submit: Path, kind: str, job: str) -> Path:
    subprocess.run(["bash", str(TEMPLATES[kind]), job], cwd=submit, check=True,
                   capture_output=True, text=True)
    script = submit / f"{job}.sh"
    syntax = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr
    return script


def _run(script: Path, env: dict, cwd: Path, **extra):
    e = dict(env)
    e.update(extra)
    return subprocess.run(["bash", str(script)], cwd=cwd, env=e, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=60)


def _log(env) -> str:
    p = Path(env["STUB_LOG"])
    return p.read_text() if p.exists() else ""


JOBS = [("crystal", "dia", ".d12"), ("prop", "dia_band", ".d3")]


# --- no submit directory -----------------------------------------------------

@pytest.mark.parametrize("kind, job, ext", JOBS)
def test_script_without_a_submit_directory_stops(tmp_path, env, submit, kind, job, ext):
    script = _generate(submit, kind, job)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    r = _run(script, env, elsewhere)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "SLURM_SUBMIT_DIR is not set" in r.stdout
    assert f"mace run {job}{ext}" in r.stdout
    assert _log(env) == ""                              # no CRYSTAL, no callback
    assert list(Path(env["HOME"]).iterdir()) == []
    assert not Path(env["SCRATCH"]).exists()            # nothing staged


@pytest.mark.parametrize("kind, job, ext", JOBS)
def test_script_with_a_submit_directory_runs_as_before(env, submit, kind, job, ext):
    script = _generate(submit, kind, job)
    r = _run(script, env, submit, SLURM_SUBMIT_DIR=str(submit), SLURM_NTASKS="2")
    assert r.returncode == 0, r.stdout + r.stderr
    log = _log(env)
    assert "mpirun -n 2" in log
    assert "python " in log                             # the callback still runs
    assert (submit / f"{job}.out").is_file()


# --- LOCAL RUN: no completion callback -----------------------------------------

@pytest.mark.parametrize("kind, job, ext", JOBS)
def test_local_run_skips_the_completion_callback(env, submit, kind, job, ext):
    script = _generate(submit, kind, job)
    r = _run(script, env, submit, SLURM_SUBMIT_DIR=str(submit), SLURM_NTASKS="2",
             MACE_LOCAL_RUN="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "mace run: no completion callback" in r.stdout
    log = _log(env)
    assert "mpirun -n 2" in log                         # CRYSTAL did run
    assert "python " not in log                         # the callback did not
    assert (submit / f"{job}.out").is_file()


@pytest.mark.parametrize("kind, job, ext", JOBS)
def test_local_run_flag_inside_a_slurm_job_keeps_the_callback(env, submit, kind, job, ext):
    """sbatch exports the user's environment; a stray MACE_LOCAL_RUN must not
    make a real batch job skip its callback."""
    script = _generate(submit, kind, job)
    r = _run(script, env, submit, SLURM_SUBMIT_DIR=str(submit), SLURM_NTASKS="2",
             MACE_LOCAL_RUN="1", SLURM_JOB_ID="123")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "no completion callback" not in r.stdout
    assert "python " in _log(env)


@pytest.mark.parametrize("local", [False, True])
def test_scratch_fallback_cleanup_still_runs_in_a_local_run(env, submit, local):
    script = _generate(submit, "crystal", "dia")
    extra = {"SLURM_SUBMIT_DIR": str(submit), "SLURM_NTASKS": "1", "SCRATCH": ""}
    if local:
        extra["MACE_LOCAL_RUN"] = "1"
    r = _run(script, env, submit, **extra)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "removed it after the run" in r.stdout
    assert not (submit / ".mace_scratch" / "crys23" / "dia").exists()


def test_new_blocks_have_no_text_the_workflow_rewrites_would_touch():
    """The blocks live inside single-quoted echo sections: no apostrophes, and
    no literal $1 (the engine replaces every $1 with the material name)."""
    for path in TEMPLATES.values():
        text = path.read_text()
        for marker, end in (("# NO SUBMIT DIRECTORY", "fi\n"), ("# LOCAL RUN", "fi\n")):
            start = text.index(marker)
            block = text[start:text.index(end, start) + len(end)]
            assert "'" not in block and "$1" not in block, block
