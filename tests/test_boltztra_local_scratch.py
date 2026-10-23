"""BOLTZTRA's scratch files go to node-local storage (1.1.6).

BOLTZTRA writes fort.70.peN, one a rank (4.5 GB each for diamond at NEWK IS 48, the Standard TRANSPORT
mesh), which nothing reads after the run. On MSU HPCC's shared scratch file system that run took 62
minutes, 2622 s of them in the SMAT stage, against 1:47 with the files on the node's /tmp (dev-amd20,
CRYSTAL23, 4 ranks), with byte-identical SIGMA.DAT and SEEBECK.DAT. ISP (the Gilat net) changed neither
the time nor the results. The properties job script makes the files links to a directory on the node
($TMPDIR, else /tmp) when it has enough room, and removes it after the run.

The job script is generated the way MACE generates it and run with bash, `mpirun` a stub that writes
the files as CRYSTAL does (tests/test_scratch_guard.py's harness).
"""
import os

import pytest

from test_scratch_guard import NO_HPCC_USER, _generate, _run

MPIRUN = """#!/bin/bash
# Stand-in for Pproperties: write a rank's BOLTZTRA scratch file and say where it went.
echo data > fort.70.pe0
{ echo "LINK=$(readlink fort.70.pe0)"; echo "CWD=$(pwd)"; } > "$DIR/$JOB.stub"
echo " EEEEEEEEEE TERMINATION"
"""

TRANSPORT = "NEWK\n48 96\n1 0\nBOLTZTRA\nTRANGE\n100.0 800.0 50.0\nMURANGE\n-4.1 -0.1 0.01\nTDFRANGE\n-7.1 2.9 0.01\nEND\nEND\n"
DOSS = "NEWK\n12 24\n1 0\nDOSS\n0 1500 -1 -1 2 14 0\n-0.6 0.3\nEND\n"


def _job(tmp_path, deck, env_extra):
    submit = tmp_path / "submit"
    submit.mkdir()
    (submit / "dia_t.d3").write_text(deck)
    (submit / "dia_t.f9").write_text("wavefunction\n")
    script = _generate(submit, "prop", "dia_t")
    scratch = tmp_path / "scr"
    scratch.mkdir()
    r = _run(tmp_path, script, {"SCRATCH": str(scratch), "USER": NO_HPCC_USER, **env_extra})    # MPIRUN as mpirun
    stub = dict(l.split("=", 1) for l in (submit / "dia_t.stub").read_text().splitlines()) \
        if (submit / "dia_t.stub").is_file() else {}
    return r, stub, scratch / "crys23" / "prop" / "dia_t"


@pytest.fixture(autouse=True)
def _our_stub(monkeypatch):
    """_run writes test_scratch_guard.MPIRUN_STUB as mpirun: ours here."""
    import test_scratch_guard as G
    monkeypatch.setattr(G, "MPIRUN_STUB", MPIRUN)


def test_boltztra_scratch_files_go_to_the_node_and_are_removed(tmp_path):
    node = tmp_path / "node_tmp"
    node.mkdir()
    r, stub, job_dir = _job(tmp_path, TRANSPORT, {"TMPDIR": str(node), "MACE_BOLTZTRA_GB": "0"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert stub["CWD"] == str(job_dir)                         # CRYSTAL runs in the usual scratch directory
    assert stub["LINK"].startswith(str(node / "mace_boltztra_dia_t_"))     # its fort.70 on the node
    assert "BOLTZTRA: its scratch files (fort.70.peN) on the node" in r.stdout
    assert list(node.iterdir()) == []                          # removed after the run
    assert not os.path.lexists(job_dir / "fort.70.pe0")        # and the link with it


def test_too_little_room_on_the_node_keeps_them_in_scratch(tmp_path):
    node = tmp_path / "node_tmp"
    node.mkdir()
    r, stub, _ = _job(tmp_path, TRANSPORT, {"TMPDIR": str(node), "MACE_BOLTZTRA_GB": "999999999"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert stub["LINK"] == ""                                  # a plain file in the scratch directory
    assert "no node-local directory with 999999999 GB free" in r.stdout


def test_other_properties_runs_are_unchanged(tmp_path):
    node = tmp_path / "node_tmp"
    node.mkdir()
    r, stub, _ = _job(tmp_path, DOSS, {"TMPDIR": str(node), "MACE_BOLTZTRA_GB": "0"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert stub["LINK"] == "" and "BOLTZTRA" not in r.stdout
    assert list(node.iterdir()) == []
