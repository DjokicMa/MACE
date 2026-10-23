"""`mace analyze <tool>` through the real mace_cli (a subprocess, as users run
it): the structure analysis tools answer, with --json, without disturbing
the older `mace analyze --extract-properties/--convergence` forms."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("scipy")
pytest.importorskip("spglib")
pytest.importorskip("ase")

REPO = Path(__file__).resolve().parent.parent
CLI = REPO / "mace_cli"
SI = REPO / "examples" / "gui" / "structures" / "Si.cif"
NACL = REPO / "examples" / "gui" / "structures" / "NaCl.cif"


def mace(*args, cwd=None, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    env["MPLBACKEND"] = "Agg"
    env.update(env_extra or {})
    r = subprocess.run([sys.executable, str(CLI), "--no-banner", "analyze", *map(str, args)], cwd=cwd, env=env,
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)
    return r


def test_pxrd_json_and_xy(tmp_path):
    out = tmp_path / "si.xy"
    r = mace("pxrd", SI, "--range", 20, 60, "--out", out, "--json")
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert d["peaks"][0]["hkl"] == [1, 1, 1] and abs(d["peaks"][0]["two_theta"] - 28.442) < 0.01
    lines = [l for l in out.read_text().splitlines() if not l.startswith("#")]
    assert len(lines) == len(d["two_theta"])



def test_pxrd_of_an_extended_xyz_saved_as_xyz(tmp_path):
    """An extended XYZ named .xyz (what ase writes) keeps its cell: its pattern is the CIF's. A plain XYZ
    is still a molecule (no pattern)."""
    from ase.io import read, write
    write(str(tmp_path / "si.xyz"), read(str(SI)), format="extxyz")
    a = mace("pxrd", tmp_path / "si.xyz", "--range", 20, 60, "--json")
    b = mace("pxrd", SI, "--range", 20, 60, "--json")
    assert a.returncode == 0 and b.returncode == 0, a.stderr + b.stderr
    pa, pb = json.loads(a.stdout)["peaks"], json.loads(b.stdout)["peaks"]
    assert [p["hkl"] for p in pa] == [p["hkl"] for p in pb]
    assert [round(p["two_theta"], 4) for p in pa] == [round(p["two_theta"], 4) for p in pb]
    (tmp_path / "water.xyz").write_text("3\nwater\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.469\nH 0.0 -0.757 -0.469\n")
    r = mace("pxrd", tmp_path / "water.xyz", "--json")
    assert r.returncode != 0 and "Traceback" not in r.stderr, r.stderr

def test_porosity_text_and_cube_files(tmp_path):
    r = mace("porosity", NACL, "--probe", "0.2", "--spacing", "0.3", "--cube", tmp_path / "nacl", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "Di " in r.stdout and "accessible surface area" in r.stdout
    cube = (tmp_path / "nacl_accessible.cube").read_text().splitlines()
    assert cube[2].split()[0] == "8"                      # NaCl conventional cell: 8 atoms


def test_topology_and_rcsr_status():
    r = mace("topology", SI, "--json")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["summary"]["topologies"] == ["dia"]
    r = mace("rcsr", "status", env_extra={"MACE_CONFIG_DIR": str(REPO / "nonexistent-config")})
    assert r.returncode == 0 and "built-in nets" in r.stdout and "3dall.txt" in r.stdout


def test_planes_cleavage_polyhedra_bonds_morphology():
    r = mace("planes", SI, "--cleavage", "--json")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["families"][0]["hkl"] == [1, 1, 1]
    r = mace("planes", SI, "--hkl", 1, 1, 1, "--offset", 0.125)
    assert r.returncode == 0 and "d = 3.13" in r.stdout
    r = mace("polyhedra", NACL)
    assert r.returncode == 0 and "octahedron" in r.stdout
    r = mace("bonds", SI)
    assert r.returncode == 0 and "Si-Si" in r.stdout
    r = mace("morphology", NACL, "--energies", "100:1.0,111:1.4")
    assert r.returncode == 0 and "(1, 0, 0)" in r.stdout


def test_peaks_index_refine_on_a_pattern(tmp_path):
    from mace.analysis import refine, structure
    pat = refine.synthetic_pattern(structure.load(SI), two_theta_range=(20, 90))
    f = tmp_path / "si.xy"
    f.write_text("\n".join(f"{x:.4f} {y:.1f}" for x, y in zip(pat["two_theta"], pat["intensity"])) + "\n")
    r = mace("peaks", f)
    assert r.returncode == 0 and "peaks in si.xy" in r.stdout
    r = mace("index", f, "--json")
    assert r.returncode == 0, r.stderr
    best = json.loads(r.stdout)["solutions"][0]
    assert best["system"] == "cubic" and abs(best["lattice"]["a"] - 5.431) < 0.005
    r = mace("refine", f, "--structure", SI, "--method", "rietveld", "--out", tmp_path / "fit.csv")
    assert r.returncode == 0, r.stderr
    assert "Rwp" in r.stdout and (tmp_path / "fit.csv").read_text().startswith("2theta,observed")


def test_bad_input_is_a_clean_error(tmp_path):
    r = mace("pxrd", tmp_path / "missing.cif")
    assert r.returncode == 2 and "no such file" in r.stderr and "Traceback" not in r.stderr


def test_older_analyze_forms_still_answer():
    r = mace()
    assert r.returncode == 0 and "--extract-properties" in r.stdout
