"""A job submitted from the GUI gets what a shell gives it (PLAN_submit D22,
D42): sbatch sees the environment `mace gui` was started with, not the GUI's
own (no NO_COLOR, MACE_NO_BANNER, MACE_GUI_*, thread caps or shim PATH), at
the server's own niceness; inside an allocation the allocation's SLURM_*
variables (all but SLURM_CONF) stay out of the job."""
import os
import subprocess
import sys

import pytest

import submit_util as SU

IGNORE = {"PWD", "OLDPWD", "_", "SHLVL"}


def _sbatch_env(fake):
    (call,) = [c for c in fake.calls("sbatch")]
    return call["full_env"], call.get("nice")


def _nice_of(pid):
    with open(f"/proc/{pid}/stat") as fh:
        return int(fh.read().rsplit(")", 1)[1].split()[16])


def test_env_equals_shell_and_not_niced(tmp_path):
    s = SU.Served(tmp_path / "gui", {"record_env": True})
    try:
        s.decks("e/decks", ["a"])
        st, prev = s.preview(["e/decks/a.d12"])
        st, res = s.run(prev["preview_id"])
        assert res["code"] == 0, res
        gui_env, gui_nice = _sbatch_env(s.fake)
        server_nice = _nice_of(s.proc.pid)
        server_env = SU.H.env(s.tmp / "xdg", MACE_GUI_TEST_HOOKS="1", MACE_GUI_NO_WARMUP="1", **s.env)
    finally:
        s.stop()
    # the same deck submitted from a shell with the server's login environment
    shell = tmp_path / "shell"
    (shell / "e" / "decks").mkdir(parents=True)
    (shell / "e" / "decks" / "a.d12").write_text(SU.SP_D12)
    fake2 = SU.FakeSlurm(tmp_path / "fs2", {"record_env": True})
    env = {k: v for k, v in server_env.items() if not k.startswith("MACE_GUI_")}
    env.update(FAKE_SLURM_STATE=fake2.env["FAKE_SLURM_STATE"], FAKE_SLURM_LOG=fake2.env["FAKE_SLURM_LOG"],
               PATH=fake2.env["PATH"])
    r = subprocess.run([sys.executable, str(SU.MACE_CLI), "submit", "decks"], cwd=shell / "e", env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    shell_env, _ = _sbatch_env(fake2)
    differ = {k for k in set(gui_env) | set(shell_env) if gui_env.get(k) != shell_env.get(k)} - IGNORE
    differ -= {"FAKE_SLURM_STATE", "FAKE_SLURM_LOG", "PATH"}         # the two fakes' own files
    assert not differ, {k: (gui_env.get(k), shell_env.get(k)) for k in differ}
    assert gui_env["PATH"].split(os.pathsep)[1:] == shell_env["PATH"].split(os.pathsep)[1:]
    for k in ("NO_COLOR", "MACE_NO_BANNER", "MPLBACKEND", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS"):
        assert gui_env.get(k) == server_env.get(k), k             # nothing the GUI adds for its own children
    assert not [k for k in gui_env if k.startswith("MACE_GUI_")]
    assert "shim" not in gui_env["PATH"]
    assert gui_nice == server_nice


def test_inside_allocation(tmp_path):
    alloc = {"SLURM_JOB_ID": "4321", "SLURM_MEM_PER_NODE": "8000", "SLURM_CPUS_ON_NODE": "4",
             "SLURM_CONF": "/etc/slurm/slurm.conf"}
    s = SU.Served(tmp_path, {"record_env": True}, **alloc)
    try:
        s.decks("e/decks", ["a"])
        st, _, caps = s.req("api/submit/capabilities")
        assert caps["in_slurm_job"] and caps["slurm_job_id"] == "4321" and "4321" in caps["own_job_ids"]
        st, prev = s.preview(["e/decks/a.d12"])
        st, res = s.run(prev["preview_id"])
        assert res["code"] == 0
        env, _ = _sbatch_env(s.fake)
        assert {k for k in env if k.startswith("SLURM_")} == {"SLURM_CONF"}
        # Run on This Node: the CLI's own refusal inside a job
        st, prev = s.preview(["e/decks/a.d12"], where="local")
        assert prev["code"] != 0 and "inside a SLURM job" in prev["refused"]
    finally:
        s.stop()
