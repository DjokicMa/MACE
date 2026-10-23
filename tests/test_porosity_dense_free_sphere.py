"""A dense crystal has no free path: Df (and Dif, the Df axes, Di) are 0,
never negative, in the batch summary and in `mace analyze porosity --json`.
The Voronoi edges of MgO and NaCl run through overlapping atom spheres, so
their clearance is negative; the largest free sphere used to come out as
about -0.46 A with the He probe."""
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("ase")
pytest.importorskip("scipy")

REPO = Path(__file__).resolve().parents[1]
CLI = REPO / "mace_cli"
EXAMPLES = REPO / "examples" / "gui" / "structures"
FILES = ["MgO.cif", "NaCl.cif"]


def mace(*args, cwd):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(MPLBACKEND="Agg", NO_COLOR="1")
    return subprocess.run([sys.executable, str(CLI), "analyze", *map(str, args)], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def dense(tmp_path_factory):
    d = tmp_path_factory.mktemp("dense")
    for f in FILES:
        shutil.copy(EXAMPLES / f, d / f)
    return d


def test_batch_summary_has_no_negative_diameters(dense):
    p = mace("porosity", *FILES, "--probe", "He", "--spacing", "0.3", "--csv", "pores.csv", cwd=dense)
    assert p.returncode == 0, p.stderr
    assert " -0." not in p.stdout, p.stdout
    rows = list(csv.DictReader((dense / "pores.csv").open()))
    assert [r["file"] for r in rows] == FILES
    for r in rows:
        assert float(r["Df"]) == 0.0, r
        assert float(r["Dif"]) == 0.0, r
        assert float(r["Di"]) >= 0.0, r


@pytest.mark.parametrize("name", FILES)
def test_single_file_json_free_sphere_is_zero(dense, name):
    p = mace("porosity", name, "--probe", "He", "--spacing", "0.3", "--json", cwd=dense)
    assert p.returncode == 0, p.stderr
    r = json.loads(p.stdout)["native"]
    assert r["Df_method"] == "voronoi"
    assert r["Df"] == 0.0 and r["Dif"] == 0.0 and r["Df_axes"] == [0.0, 0.0, 0.0]
    assert r["Di"] >= 0.0 and r["Df_grid"] >= 0.0
    assert any("fits nowhere" in w for w in r["warnings"])
