"""`mace analyze <tool> FILE...` on several structures, through mace_cli as a
subprocess: a line per structure, a combined CSV (one row each) and JSON
(a list of {file, tool, params, result}), one file's output unchanged, a
file that fails reported while the others go on (exit code 1), and the
options that write one file refused with several."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("ase")
pytest.importorskip("scipy")

REPO = Path(__file__).resolve().parents[1]
CLI = REPO / "mace_cli"


def mace(*args, cwd):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(MPLBACKEND="Agg", NO_COLOR="1")
    return subprocess.run([sys.executable, str(CLI), "analyze", *map(str, args)], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def structures(tmp_path_factory):
    from ase.build import bulk
    from ase.io import write
    from ase.spacegroup import crystal
    d = tmp_path_factory.mktemp("batch")
    write(str(d / "Si.cif"), bulk("Si", "diamond", a=5.431, cubic=True))
    write(str(d / "MgO.cif"), bulk("MgO", "rocksalt", a=4.212, cubic=True))
    write(str(d / "SOD.cif"), crystal(["Si", "O"], basis=[(0.25, 0.5, 0.0), (0.1467, 0.5, 0.8533)], spacegroup=229,
                                      cellpar=[8.965] * 3 + [90] * 3))
    return d


FILES = ["Si.cif", "MgO.cif", "SOD.cif"]


def test_porosity_over_three_structures(structures):
    d = structures
    p = mace("porosity", *FILES, "--probe", "He", "--spacing", "0.3", "--csv", "pores.csv", cwd=d)
    assert p.returncode == 0, p.stderr
    lines = [ln for ln in p.stdout.splitlines() if ln.split(":")[0] in FILES]
    assert [ln.split(":")[0] for ln in lines] == FILES and "Di " in lines[0]
    rows = list(csv.DictReader((d / "pores.csv").open()))
    assert [r["file"] for r in rows] == FILES
    assert list(rows[0]) == ["file", "probe", "probe_radius", "radii", "spacing", "Di", "Df", "Dif", "AV_fraction", "POAV_A3",
                             "POAV_cm3_g", "ASA_m2_g", "NASA_m2_g", "density", "n_channels", "channel_dim", "n_pockets"]
    assert all(r["probe"] == "He" and float(r["spacing"]) == pytest.approx(0.3) for r in rows)
    assert rows[2]["n_pockets"] == "2" and rows[2]["n_channels"] == "0"         # sodalite's two cages, closed to He
    # the same numbers as a run on one file
    one = json.loads(mace("porosity", "SOD.cif", "--probe", "He", "--spacing", "0.3", "--json", cwd=d).stdout)
    assert float(rows[2]["Di"]) == pytest.approx(one["native"]["Di"], abs=1e-4)
    assert float(rows[2]["POAV_A3"]) == pytest.approx(one["native"]["POAV"]["A3"], abs=0.01)


def test_pxrd_json_and_one_file_unchanged(structures):
    d = structures
    p = mace("pxrd", *FILES, "--json", "--csv", "peaks.csv", cwd=d)
    assert p.returncode == 0, p.stderr
    runs = json.loads(p.stdout)
    assert [r["file"] for r in runs] == FILES and all(r["tool"] == "pxrd" and "result" in r for r in runs)
    assert runs[0]["params"]["source"] == "CuKa"
    rows = list(csv.DictReader((d / "peaks.csv").open()))
    assert len(rows) == 3 and rows[0]["strongest_hkl"] == "1 1 1" and int(rows[0]["n_peaks"]) > 3
    # one file: the output it always had (a table, or the result itself as JSON)
    single = mace("pxrd", "Si.cif", cwd=d)
    assert single.returncode == 0 and single.stdout.startswith("Si: Si8  CuKa") and "2theta" in single.stdout
    assert json.loads(mace("pxrd", "Si.cif", "--json", cwd=d).stdout)["peaks"] == runs[0]["result"]["peaks"]


def test_a_failing_file_does_not_stop_the_others(structures):
    d = structures
    p = mace("topology", "Si.cif", "missing.cif", "MgO.cif", "--csv", "topo.csv", cwd=d)
    assert p.returncode == 1
    assert "missing.cif: error" in p.stderr and "1 of 3 structures failed" in p.stderr
    rows = list(csv.DictReader((d / "topo.csv").open()))
    assert [r["file"] for r in rows] == ["Si.cif", "missing.cif", "MgO.cif"]
    assert rows[0]["topology"] == "dia" and rows[1]["error"] and not rows[0]["error"]


def test_one_file_options_are_refused_with_several(structures):
    d = structures
    for args in (("pxrd", "--out", "x.xy"), ("porosity", "--cube", "pore"), ("porosity", "--probe", "all")):
        p = mace(args[0], "Si.cif", "MgO.cif", *args[1:], cwd=d)
        assert p.returncode == 2 and "one structure file" in p.stderr, (args, p.stderr)


def test_summary_row_reads_gui_results():
    # the GUI's planes result keeps the bond profile under "broken"; cleavage lists families
    from mace.analysis.cli import summary_row
    r = summary_row("planes", {"hkl": [1, 1, 1], "d": 3.1355, "broken": {"min_per_nm2": 7.1, "best_offset": 0.25}}, "Si.cif")
    assert r == {"file": "Si.cif", "hkl": "1 1 1", "d": 3.1355, "min_bonds_per_nm2": 7.1, "best_offset": 0.25}
    c = summary_row("cleavage", {"families": [{"hkl": [1, 0, 0], "d": 2, "min_per_nm2": 9, "best_offset": 0},
                                               {"hkl": [1, 1, 1], "d": 3, "min_per_nm2": 7, "best_offset": 0.1}]}, "x")
    assert c["hkl"] == "1 1 1" and c["min_bonds_per_nm2"] == 7
