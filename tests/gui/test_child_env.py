"""The Python children the GUI starts never import from the MACE checkout by
accident (mace/queue shadows the standard library's queue): a neutral
working directory, and an inherited PYTHONPATH without the checkout
(childenv.py). Covers the CLI children (clirun: deck previews, plots, Setup,
preflight), the process pool's workers (builders, Clean, symmetrize, FREQ
parsing) and the server started the way a user with MACE's own folder as the
working directory and the checkout on PYTHONPATH starts it.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

import guiharness as H
import p4util as U
from mace.gui import childenv, clirun

REPO = H.REPO_ROOT


# --- unit -------------------------------------------------------------------------------

def test_clean_pythonpath_drops_mace_dir_and_relative_entries(tmp_path):
    other = str(tmp_path / "site-packages")
    value = os.pathsep.join([str(REPO), "", str(REPO / "mace"), str(REPO / "mace" / "queue"), "relative/dir",
                             other, ".", other])
    assert childenv.clean_pythonpath(value) == os.pathsep.join([str(REPO), other])
    assert childenv.clean_pythonpath(f"{REPO / 'mace'}{os.pathsep}") is None
    assert childenv.clean_pythonpath("") is None and childenv.clean_pythonpath(None) is None


def test_clean_sys_path_drops_mace_dir_only(tmp_path):
    other = str(tmp_path)
    got = childenv.clean_sys_path([str(REPO), str(REPO / "mace"), "", "rel", other, str(REPO / "mace" / "queue"),
                                   str(REPO / "Crystal_d12")])
    assert got == [str(REPO), other, str(REPO / "Crystal_d12")]


def test_neutral_cwd_is_outside_the_checkout():
    assert not childenv.inside(childenv.neutral_cwd())


def test_cli_env_cleans_pythonpath_and_keeps_module_paths(monkeypatch, tmp_path):
    bundle = str(tmp_path / "bundle" / "lib" / "python3.11" / "site-packages")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(REPO / "mace"), "", bundle]))
    monkeypatch.delenv("MACE_GUI_TEST_HOOKS", raising=False)
    env = clirun.cli_env()
    assert env["PYTHONPATH"] == bundle            # an HPCC module bundle stays importable
    monkeypatch.setenv("PYTHONPATH", f"{REPO / 'mace'}{os.pathsep}")
    assert "PYTHONPATH" not in clirun.cli_env()


def test_a_cli_child_from_inside_the_checkout_imports_the_stdlib_queue(monkeypatch):
    """The real CLI as a child, its working directory mace/ (where queue/ is),
    the checkout and that directory on the parent's PYTHONPATH."""
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(REPO), ""]))
    r = clirun.run(clirun.mace_argv("--no-banner", "version"), REPO / "mace", timeout=120)
    assert r.returncode == 0, r.log
    probe = [sys.executable, "-c", "import queue, sys; print(queue.__file__)"]
    r = clirun.run(probe, Path(childenv.neutral_cwd()), timeout=60)
    assert r.returncode == 0 and "mace" not in r.stdout and r.stdout.strip().endswith("queue.py"), r.log


# --- the real server, started from inside the checkout --------------------------------

@pytest.fixture(scope="module")
def served_from_mace_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("childenv")
    root, xdg = H.examples_copy(tmp), tmp / "xdg"
    env = H.env(xdg, MACE_GUI_TEST_HOOKS="1", MACE_GUI_NO_WARMUP="1",
                PYTHONPATH=os.pathsep.join([str(REPO), ""]))
    proc = subprocess.Popen(
        [sys.executable, str(H.MACE_CLI), "--no-banner", "gui", "--dir", str(root), "--port", "0",
         "--token", H.TOKEN],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
        cwd=str(REPO / "mace"), start_new_session=True)
    import re
    base, lines = None, []
    for _ in range(40):
        line = proc.stdout.readline()
        if not line:
            break
        lines.append(line)
        m = re.search(H.URL_RE, line)
        if m:
            base = m.group(1)
            break
    try:
        assert base, "".join(lines)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_pool_worker_is_clean_when_the_server_runs_in_mace_dir(served_from_mace_dir):
    base = served_from_mace_dir
    st, _, w = U.post(base, "api/test/task", {"job": "probe", "wait": 30})
    assert st == 200, w
    assert "/mace/queue" not in (w["queue_file"] or ""), w
    assert not w["mace_dir_on_sys_path"] and not w["mace_subdir_on_sys_path"], w
    assert not w["cwd_inside_repo"], w


def test_a_pool_builder_runs_when_the_server_runs_in_mace_dir(served_from_mace_dir):
    """A real pool job (a supercell over the inline size) from that server."""
    pytest.importorskip("ase")
    base = served_from_mace_dir
    from test_builders_api import result_of
    st, _, r = U.post(base, "api/build/supercell", {"source": {"path": "structures/MgO.cif"},
                                                    "params": {"n": [6, 6, 6]}})
    r = result_of(base, st, r)
    assert "error" not in r, r
