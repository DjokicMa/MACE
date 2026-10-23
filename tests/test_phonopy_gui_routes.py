"""The Phonopy dialog's server side (mace/gui/phonopy_routes.py) and the
SPINLOCK fix of phonopy_bridge.parent.final_mulliken_spins.

The plot data come from phonopy's own files (tests/fixtures/phonopy_crystal/
NaCl_collected, written by the collect step, now mace analyze phonopy); the job
table from the
real displaced-supercell outputs of the NaCl fixture. phonopy itself is not
needed: the routes read files and run the CLI.
"""
import json
import os
import shutil
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from mace.gui import mechanics_routes as MR
from mace.gui import phonopy_routes as R
from mace.phonopy_bridge.parent import final_mulliken_spins

FIX = REPO_ROOT / "tests" / "fixtures" / "phonopy_crystal"
STRAIN = REPO_ROOT / "tests" / "data" / "hpcc_116" / "strain"


class St:
    """The server state the routes use: the served folder and its path rules."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def resolve(self, rel, must_exist=True):
        t = (self.root / rel).resolve()
        if t != self.root and self.root not in t.parents:
            raise PermissionError("outside the served directory")
        if must_exist and not t.exists():
            raise FileNotFoundError(rel)
        return t

    def rel(self, p):
        return Path(p).resolve().relative_to(self.root).as_posix()

    def rel_new(self, p):
        p = Path(p).resolve()
        if p != self.root and self.root not in p.parents:
            raise PermissionError("outside the served directory")
        return p.relative_to(self.root).as_posix()

    def out_dir(self, p):
        from mace.gui.server import GuiState
        return GuiState.out_dir(self, p)


def test_band_yaml_segments_labels_and_jumps():
    b = R.read_band_yaml(FIX / "NaCl_collected" / "band.yaml")
    assert len(b["distances"]) == 30 and len(b["bands"]) == 6 and len(b["bands"][0]) == 30
    labels = [t["label"] for t in b["ticks"]]
    # X-U then K-Gamma: the jump is one tick labeled U|K
    assert labels == ["$\\Gamma$", "X", "U|K", "$\\Gamma$", "L", "W", "X"]
    assert b["ticks"][0]["x"] == 0.0 and b["ticks"][-1]["x"] == b["distances"][-1]
    assert min(b["bands"][0]) == pytest.approx(0.0, abs=1e-3)          # acoustic branch at Gamma


def test_dos_and_thermal_properties():
    d = R.read_total_dos(FIX / "NaCl_collected" / "total_dos.dat")
    assert len(d["frequency"]) == len(d["dos"]) > 50
    t = R.read_thermal(FIX / "NaCl_collected" / "thermal_properties.yaml")
    assert t["temperature"] == [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
    assert t["units"]["heat_capacity"] == "J/(K mol)" and t["zero_point_energy"] > 0
    assert t["heat_capacity"][0] == 0.0 and 40 < t["heat_capacity"][3] < 50     # NaCl, near 3R x 2 at 300 K


def _set(tmp_path):
    d = tmp_path / "phonopy_parent_444"
    d.mkdir()
    jobs = ["parent_ph444_disp-001", "parent_ph444_disp-002", "parent_ph444_disp-003"]
    (d / R.MANIFEST).write_text(json.dumps({"parent_out": str(tmp_path / "parent.out"), "jobs": jobs,
                                            "dim": [[4, 0, 0], [0, 4, 0], [0, 0, 4]], "walltime": "2-02:00:00",
                                            "n_atoms_supercell": 128, "warnings": ["w"]}))
    shutil.copy(FIX / "NaCl" / "parent.out", tmp_path / "parent.out")
    shutil.copy(FIX / "NaCl" / "supercell-001.o", d / f"{jobs[0]}.out")
    text = (FIX / "NaCl" / "supercell-002.o").read_text()
    (d / f"{jobs[1]}.out").write_text(text[:text.index("CARTESIAN FORCES")])   # cut before its forces
    import os
    import time
    old = time.time() - 3600
    os.utime(d / f"{jobs[1]}.out", (old, old))
    for f in ("band.yaml", "total_dos.dat", "thermal_properties.yaml"):
        shutil.copy(FIX / "NaCl_collected" / f, d / f)
    return d


def test_status_reads_each_job(tmp_path):
    d = _set(tmp_path)
    s = R.status(St(tmp_path), d)
    states = {j["name"][-3:]: j["state"] for j in s["jobs"]}
    assert states == {"001": "done", "002": "failed", "003": "not run"}
    assert s["counts"] == {"done": 1, "failed": 1, "not run": 1} and not s["ready"]
    assert s["parent_out"] == "parent.out" and s["walltime"] == "2-02:00:00"
    assert s["results"] == ["band.yaml", "total_dos.dat", "thermal_properties.yaml"]
    assert R.sets(St(tmp_path), tmp_path)[0]["dir"] == "phonopy_parent_444"
    data = R.data(d)
    assert set(data) >= {"band", "dos", "thermal"} and data["cm1_per_thz"] == pytest.approx(33.35641)


def test_commands_are_the_cli_s(tmp_path):
    d = _set(tmp_path)
    st = St(tmp_path)
    argv, cwd, out = R.generate_argv(st, {"parent": "parent.out", "dim": "2 2 2", "amplitude": 0.02, "cry_sym": True})
    assert argv[:7] == ["--no-banner", "opt2d12", "--out-file", str(tmp_path / "parent.out"), "--calc-type",
                        "PHONOPY", "--non-interactive"]
    assert argv[7:11] == ["--dim", "2", "2", "2"] and "--cry-sym" in argv and "--amplitude" in argv
    assert out == tmp_path and cwd == tmp_path          # the set's folder (opt2d12 names it) goes here
    argv, _, out = R.generate_argv(st, {"parent": "parent.out", "dim": "2 2 2", "output_dir": "sets"})
    assert argv[argv.index("--output-dir") + 1] == str(tmp_path / "sets") and out == tmp_path / "sets"
    argv, _ = R.collect_argv(st, {"dir": d.name, "band": "none", "mesh": "6 6 6", "nac_from": "parent.out"})
    assert argv[:4] == ["--no-banner", "analyze", "phonopy", str(d)]
    assert argv[argv.index("--band") + 1] == "none" and argv[argv.index("--mesh") + 1] == "6 6 6"
    for bad in ({"parent": "parent.out", "dim": "2 2"}, {"parent": "parent.out", "dim": "2 2 2", "amplitude": 1},
                {"parent": "../x.out", "dim": "2 2 2"}):
        with pytest.raises((ValueError, PermissionError, FileNotFoundError)):
            R.generate_argv(st, bad)
    with pytest.raises(PermissionError):
        R.generate_argv(st, {"parent": "parent.out", "dim": "2 2 2", "output_dir": "../../elsewhere"})


def test_tau_routes_estimate_series_and_fit(tmp_path):
    for f in STRAIN.glob("*.out"):
        shutil.copy(f, tmp_path / f.name)
    st = St(tmp_path)
    ser = MR.series(st, tmp_path)["series"]
    assert len(ser) == 1 and ser[0]["name"] == "dia_sp_dp_x" and len(ser[0]["outputs"]) == 5
    r = MR.fit(st, {"outputs": ser[0]["outputs"], "carrier": "hole", "m_star": 0.5})
    assert r["result"]["fit"]["C"] == pytest.approx(1147, rel=0.01) and "Bardeen" in r["report"]
    with pytest.raises(ValueError, match="m\\*"):
        MR.fit(st, {"outputs": ser[0]["outputs"], "carrier": "hole"})
    e = MR.estimate({"dim": 2, "m_star": 0.17, "m_other": 1.12, "e1": 2.72, "c": 28.94})
    assert 1.05e3 < e["result"]["mobility_cm2_Vs"] < 1.2e3


def test_elastic_routes_modes_and_argv(tmp_path):
    m = MR.modes()
    assert [x["mode"] for x in m["modes"]] == ["ELASTCON", "ELAPIEZO", "PIEZOCON", "PIEZOCP", "PHOTOELA", "EOS", "QHA", "CPHF"]
    shutil.copy(FIX / "Si" / "parent.out", tmp_path / "si.out")
    argv, cwd = MR.generate_argv(St(tmp_path), {"parent": "si.out", "settings": {"mode": "EOS", "range": [0.9, 1.1, 8]}})
    assert argv[:8] == ["--no-banner", "opt2d12", "--out-file", str(tmp_path / "si.out"), "--calc-type", "EOS",
                        "--non-interactive", "--elastic-settings"]
    assert json.loads(argv[argv.index("--elastic-settings") + 1]) == {"mode": "EOS", "range": [0.9, 1.1, 8]}
    with pytest.raises(ValueError):
        MR.generate_argv(St(tmp_path), {"parent": "si.out", "settings": {"mode": "ELASTCON", "numderiv": "x"}})


def test_final_mulliken_spins_skips_the_spinlock_echo():
    """A SPINLOCK run echoes "ALPHA-BETA ELECTRONS LOCKED TO 0 FOR 3 SCF CYCLES".
    With its Mulliken block gone (a killed run), 1.1.5 read the neighbor table
    after that echo as spins (1.54 ... 3.54 "muB" on atom 1 C), and phonopy
    generate refused the parent as magnetic."""
    text = (STRAIN / "dia_sp_dp_x_p1pct.out").read_text()
    assert "ALPHA-BETA ELECTRONS LOCKED TO" in text
    rows = final_mulliken_spins(text)
    assert [(a, s) for a, s, _ in rows] == [(1, "C"), (2, "C")] and all(abs(v) < 1e-3 for *_, v in rows)
    killed = text[:text.rfind("ALPHA-BETA ELECTRONS\n")]
    assert "ALPHA-BETA ELECTRONS" in killed
    assert final_mulliken_spins(killed) is None


class _Handler:
    """The request handler the routes answer through: (status, body)."""

    def __init__(self, root):
        self.state = St(root)
        self.state.root_gen = 0
        self.headers = {}

    def _json(self, obj, status=200):
        return status, obj

    def _error(self, status, msg):
        return status, {"error": msg}


def test_lists_leave_out_symlinks_out_of_the_folder(tmp_path):
    """A strain output or a phonopy set that is a symlink leading out of the served folder is left out
    of tau/series and phonopy/sets; the rest of the list still answers (it was a 400 naming the
    absolute path outside)."""
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    for f in STRAIN.glob("*.out"):
        shutil.copy(f, root / f.name)
    shutil.copy(STRAIN / "dia_sp_dp_x_p1pct.out", outside / "far_dp_x_p1pct.out")
    os.symlink(outside / "far_dp_x_p1pct.out", root / "dia_sp_dp_x_p2pct.out")
    d = _set(root)
    (outside / "far").mkdir()
    shutil.copy(d / R.MANIFEST, outside / "far" / R.MANIFEST)
    os.symlink(outside / "far", root / "linkset")
    (outside / "nested").mkdir()
    (outside / "nested" / "inner").mkdir()
    shutil.copy(d / R.MANIFEST, outside / "nested" / "inner" / R.MANIFEST)
    os.symlink(outside / "nested", root / "linknest")
    st = St(root)
    ser = MR.series(st, root)["series"]
    assert len(ser) == 1 and len(ser[0]["outputs"]) == 5 and "dia_sp_dp_x_p2pct.out" not in ser[0]["outputs"]
    assert [s["dir"] for s in R.sets(st, root)] == ["phonopy_parent_444"]
    code, body = R.handle(_Handler(root), "GET", "phonopy/sets", {"dir": ["."]})
    assert code == 200 and str(outside) not in json.dumps(body), body


def test_bad_input_is_a_400_not_a_500(tmp_path):
    for f in STRAIN.glob("*.out"):
        shutil.copy(f, tmp_path / f.name)
    shutil.copy(FIX / "Si" / "parent.out", tmp_path / "si.out")
    shutil.copy(FIX / "Si" / "parent.d12", tmp_path / "si_sp.d12")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / R.MANIFEST).write_text("[1, 2]")
    h = _Handler(tmp_path)
    assert MR.handle(h, "GET", "elastic/analyze", {"path": ["sub"]})[0] == 400
    assert MR.handle(h, "POST", "tau/fit", {"outputs": ["sub"] * 3, "carrier": "hole", "m_star": 0.3})[0] == 400
    for strains in ([{"a": 1}], "0.01,abc", "nan,0.01", [None]):
        code, body = MR.handle(h, "POST", "tau/strain", {"parent": "si_sp.d12", "strains": strains, "root_gen": 0})
        assert code == 400, (strains, body)
    assert not list(tmp_path.glob("si_sp_dp_*"))
    assert R.handle(h, "GET", "phonopy/status", {"dir": ["sub"]})[0] == 400
    assert [s["dir"] for s in R.sets(h.state, tmp_path)] == []
    from mace.gui import volume as V
    for bad in ("1e400", "-1e400", "inf", "nan"):
        with pytest.raises(ValueError, match="whole numbers"):
            V._plane_ints([bad, 0, 1])
