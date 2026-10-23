"""Optics of CPHF runs (1.1.6, mace/elastic/optics.py): principal axes and values of eps_inf, the
index ellipsoid and birefringence, n(omega), kappa and the absorption of a DYNAMIC run, and the SHG
d_eff and directional |d(n)| (mace analyze elastic --plot indicatrix|refractive|absorption|shg).

The real diamond CPHF run (isotropic, centrosymmetric) and outputs in CRYSTAL's format with a chosen
tensor (tests/cphf_synthetic.py), checked against hand values: n = sqrt(eps); a uniaxial crystal's
n_o, n_e and sign; n + i kappa = sqrt(eps' + i eps''); for a -43m d14 tensor, the largest |d(n)| and
d_L along <111> are 2 d14/sqrt(3) and the powder averages are d14 sqrt(36/105) and d14 sqrt(4/5)
(<x^2 y^2 z^2> = 1/105, <x^2 y^2> = 1/15 over the sphere).
"""
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT
from cphf_synthetic import cphf_output
from mace.elastic import analysis as A
from mace.elastic import optics as O
from mace.elastic import views as V

MECH = REPO_ROOT / "tests" / "data" / "hpcc_116" / "mechanics"
MACE_CLI = REPO_ROOT / "mace_cli"


def test_diamond_is_isotropic_and_centrosymmetric():
    d = A.analyze_file(MECH / "dia_cphf.out")
    pr = d["optics"]["principal"]
    assert pr["optical_class"] == "isotropic" and pr["n"] == [pytest.approx(math.sqrt(5.387))] * 3
    assert pr["n"][0] == pytest.approx(d["dielectric"]["refractive_indices"][0], abs=1e-4)     # CRYSTAL's own n
    assert d["optics"]["shg"] == {"zero": True, "unit": "pm/V"}
    assert [v["key"] for v in V.views(d, (MECH / "dia_cphf.out").read_text())] == ["indicatrix"]


def test_principal_axes_of_a_rotated_biaxial_tensor():
    # principal values 3, 4, 5 along axes rotated 30 degrees about z
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    E = R @ np.diag([3.0, 4.0, 5.0]) @ R.T
    pr = O.principal(E)
    assert pr["epsilon"] == pytest.approx([3, 4, 5]) and pr["optical_class"] == "biaxial"
    assert abs(np.dot(pr["axes"][0], R[:, 0])) == pytest.approx(1, abs=1e-6)
    nx, ny, nz = math.sqrt(3), 2.0, math.sqrt(5)
    assert pr["birefringence"] == pytest.approx(nz - nx)
    vz = 2 * math.degrees(math.atan(nz / nx * math.sqrt((ny ** 2 - nx ** 2) / (nz ** 2 - ny ** 2))))     # 104.5: obtuse about z
    assert pr["optic_axial_angle_deg"] == pytest.approx(180 - vz) and pr["sign"] == "negative"
    surf = O.indicatrix(pr)
    pts = np.stack([np.array(surf[k]).ravel() for k in "xyz"], axis=1)
    # every point of the glyph is on the ellipsoid x.R diag(1/n^2) R^T.x = 1
    M = R @ np.diag([1 / 3, 1 / 4, 1 / 5]) @ R.T
    assert np.einsum("ni,ij,nj->n", pts, M, pts) == pytest.approx(1, abs=1e-4)


def test_a_uniaxial_crystal_with_shg_by_hand():
    d14 = 10.0
    t = cphf_output({"XX": 4.0, "YY": 4.0, "ZZ": 4.5}, {"XYZ": d14})
    d = A.analyze(t)
    pr = d["optics"]["principal"]
    assert pr["optical_class"] == "uniaxial" and pr["n_o"] == 2.0 and pr["n_e"] == pytest.approx(math.sqrt(4.5))
    # a tetragonal crystal (P-42m): Cartesian labels, its [111] is not the Cartesian x+y+z
    assert pr["sign"] == "positive" and pr["optic_axis_label"] == "z" and "warnings" not in d
    sh = d["optics"]["shg"]
    assert sh["d_axes"] == [0.0, 0.0, 0.0]
    for k in ("longitudinal", "magnitude"):
        assert sh[k]["max"]["label"] == "x+y+z" and sh[k]["max"]["value"] == pytest.approx(2 * d14 / math.sqrt(3), rel=1e-6)
    assert sh["powder_longitudinal"] == pytest.approx(d14 * math.sqrt(36 / 105), rel=1e-3)
    assert sh["powder_magnitude"] == pytest.approx(d14 * math.sqrt(4 / 5), rel=1e-3)
    assert [v["key"] for v in V.views(d, t)] == ["indicatrix", "shg"]
    f = V.figure(d, "shg", t, "planes")
    assert [k["key"] for k in f["planes"][0]["curves"]] == ["magnitude", "longitudinal"]
    f = V.figure(d, "shg", t, "surface")
    assert max(max(r) for r in f["surface"]["value"]) == pytest.approx(2 * d14 / math.sqrt(3), rel=2e-3)
    f = V.figure(d, "indicatrix", t)
    assert f["glyph"] and V.table(f)["columns"] == ["x", "y", "z", "n"] and "uniaxial, Δn = 0.1213" in f["note"]
    assert "optically positive, optic axis z" in A.report(d) and "SHG d_eff (pm/V)" in A.report(d)


def test_a_dynamic_run_s_refractive_index_and_absorption():
    eps = {0.0: {"XX": 4.0, "YY": 4.0, "ZZ": 4.5}, 1.0: {"XX": 4.2, "YY": 4.2, "ZZ": 4.8}}
    im = {0.0: {}, 1.0: {"XX": 0.5, "YY": 0.5, "ZZ": 0.6}}
    t = cphf_output(eps, None, (0.0, 1.0), im)
    d = A.analyze(t)
    xx = d["optics"]["dynamic"]["components"]["xx"]
    m = math.hypot(4.2, 0.5)
    n, k = math.sqrt((m + 4.2) / 2), math.sqrt((m - 4.2) / 2)
    assert xx["n"][1] == pytest.approx(n, rel=1e-6) and xx["kappa"][1] == pytest.approx(k, rel=1e-6)
    assert xx["absorption_cm"][1] == pytest.approx(2 * 1.0 * k / 1.973269804e-5, rel=1e-6)          # 2 omega kappa / c
    assert [v["key"] for v in V.views(d, t)] == ["epsilon", "refractive", "absorption", "indicatrix"]
    f = V.figure(d, "refractive", t)
    assert [s["name"] for s in f["series"]][:2] == ["n<sub>xx</sub>", "κ<sub>xx</sub>"]
    # real eps only: n(omega) without kappa, no absorption tab
    t2 = cphf_output(eps, None, (0.0, 1.0))
    d2 = A.analyze(t2)
    assert [v["key"] for v in V.views(d2, t2)] == ["epsilon", "refractive", "indicatrix"]
    assert all(s["name"].startswith("n<sub>") for s in V.figure(d2, "refractive", t2)["series"])


def test_mace_elastic_analyze_optics_json_and_plots(tmp_path):
    (tmp_path / "uni_cphf.out").write_text(cphf_output({"XX": 4.0, "YY": 4.0, "ZZ": 4.5}, {"XYZ": 10.0}))
    env = {"MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "analyze", "elastic", "uni_cphf.out", "--json"],
                       capture_output=True, text=True, cwd=tmp_path, env=env, timeout=300)
    assert r.returncode == 0, r.stderr
    js = json.loads(r.stdout)
    assert js["optics"] == json.loads(json.dumps(A.analyze_file(tmp_path / "uni_cphf.out")["optics"]))
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "analyze", "elastic", "uni_cphf.out", "--plot", "indicatrix",
                        "--plot", "shg", "--format", "svg"], capture_output=True, text=True, cwd=tmp_path, env=env, timeout=300)
    assert r.returncode == 0, r.stderr
    for name in ("indicatrix_surface", "shg_surface"):
        assert (tmp_path / f"uni_cphf.{name}.svg").stat().st_size > 10000, name
