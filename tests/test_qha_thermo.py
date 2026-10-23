"""QHA thermodynamics beyond the zero-pressure table (1.1.6, mace/elastic/qha.py): CRYSTAL's
Grueneisen table, each volume's harmonic block, F, S, C_v and C_p along V(T), the thermodynamic
Grueneisen parameter, and the F(V) family from the PVT relation (mace analyze elastic --plot
gamma|free|entropy|fv|vratio).

On tests/data/hpcc_116/mechanics/dia_qha_full.out: the real diamond QHA run (4 volumes, Gamma only)
with each volume's harmonic block and the PVT relation kept at T = 10 K and every 50 K (README).
Checked against what CRYSTAL prints itself and against hand formulas:

* F of each volume = CRYSTAL's EL+E0+ET+PV-TS line minus its PV line;
* C_v of a volume = the Einstein heat capacity of its three F2g modes (frequencies printed);
* S = -dF/dT along V(T) (the envelope theorem);
* the minima of the F(V) curves sit at the zero-pressure V(T) of the HELMHOLTZ table;
* gamma_th approaches CRYSTAL's mode gamma (a single optical mode).
"""
import math
import re
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT
from mace.elastic import analysis as A
from mace.elastic import qha as Q
from mace.elastic import views as V

FULL = REPO_ROOT / "tests" / "data" / "hpcc_116" / "mechanics" / "dia_qha_full.out"
MACE_CLI = REPO_ROOT / "mace_cli"


@pytest.fixture(scope="module")
def text():
    return FULL.read_text()


def test_each_volume_s_harmonic_block_against_crystal_s_own_lines(text):
    blocks = Q.harmonic_blocks(text)
    assert [round(b["volume_ang3"], 4) for b in blocks] == [10.8017, 11.1357, 11.4766, 11.8244]
    assert blocks[0]["temperature_k"] == [10.0] + [50.0 * k for k in range(1, 13)]
    # F = (EL+E0+ET+PV-TS) - PV, both as CRYSTAL prints them, for the first volume at 600 K
    i = text.index("AT (T =  600.00 K", text.index("10.801674 - DENSITY"))
    part = text[i:i + 800]
    g = float(re.search(r"EL\+E0\+ET\+PV-TS:\s*(\S+)", part).group(1))
    pv = float(re.search(r"PV\s*:\s*(\S+)", part).group(1))
    b = blocks[0]
    f = b["el_au"] + b["e0_au"] + b["et_au"][-1] - b["ts_au"][-1]
    assert f == pytest.approx(g - pv, abs=2e-12)
    # C_v of three degenerate modes at 1420.3364 cm-1 (Einstein), per mole of cells
    x = 6.62607015e-34 * 2.99792458e10 * 1420.3364 / (1.380649e-23 * 600.0)
    cv = 3 * 8.314462618 * x * x * math.exp(x) / (math.exp(x) - 1) ** 2
    assert b["cv_j_mol_k"][-1] == pytest.approx(cv, rel=1e-5)


def test_f_s_cv_cp_and_gamma_along_v_of_t(text):
    d = A.analyze(text)
    q, th = d["qha"], d["qha"]["thermo"]
    assert th["temperature_k"] == [10.0] + [50.0 * k for k in range(1, 13)] and th["polynomial_order"] == 2
    # the envelope theorem: S = -dF/dT along the equilibrium volume
    T, F = th["temperature_k"], th["f_kj_mol"]
    for t in (400.0, 500.0):
        k = T.index(t)
        dfdt = (F[k + 1] - F[k - 1]) / (T[k + 1] - T[k - 1]) * 1000.0
        assert -dfdt == pytest.approx(th["entropy_j_mol_k"][k], rel=0.08)
    k = T.index(600.0)
    i = q["temperature_k"].index(600.0)
    assert th["cp_j_mol_k"][k] == pytest.approx(th["cv_j_mol_k"][k] + q["cp_minus_cv_j_mol_k"][i])
    assert th["volume_ang3"][k] == q["volume_ang3"][i]
    # one optical mode: the thermodynamic gamma is the mode gamma (0.919) within the printed precision of alpha
    assert th["gamma_th"][k] == pytest.approx(q["gruneisen"]["gamma"][-1], rel=0.05)
    assert th["gamma_th"][0] is None                                  # alpha prints as 0 at 10 K
    assert "Along V(T)" in A.report(d) and "mode Grueneisen parameter: 0.919" in A.report(d)


def test_the_f_of_v_family_has_its_minima_at_v_of_t(text):
    d = A.analyze(text)
    fam = Q.free_energy_family(text)
    assert fam["source"] == "pvt" and len(fam["curves"]) == 7 and len(fam["points"]) == 7
    for c in fam["curves"]:
        i = d["qha"]["temperature_k"].index(c["temperature_k"])
        assert c["interior"] and c["v_min"] == pytest.approx(d["qha"]["volume_ang3"][i], abs=5e-4)
    # without the PVT table: a polynomial through the computed volumes, minima at the same V(T)
    cut = text[:text.index(" PRESSURE-VOLUME-TEMPERATURE RELATION")]
    fam2 = Q.free_energy_family(cut)
    assert fam2["source"] == "harmonic"
    for c in fam2["curves"]:
        i = d["qha"]["temperature_k"].index(c["temperature_k"])
        assert c["v_min"] == pytest.approx(d["qha"]["volume_ang3"][i], abs=2e-3)
    f = V.figure(d, "fv", text)
    assert [s["key"] for s in f["series"]][-2:] == ["minima", "computed"] and f["y_label"] == "F − F<sub>min</sub> (meV/cell)"
    assert V.table(f)["columns"] == ["series", "V (Å^3)", "F - F_min (meV/cell)"]


def test_views_figures_and_the_trimmed_output_without_them(text):
    d = A.analyze(text)
    assert [v["key"] for v in V.views(d, text)] == ["volume", "vratio", "expansion", "bulk", "heat", "gamma", "free", "entropy", "fv"]
    f = V.figure(d, "heat", text)
    assert [s["name"] for s in f["series"]][:3] == ["C<sub>p</sub>", "C<sub>v</sub> at V(T)", "C<sub>p</sub> − C<sub>v</sub>"]
    f = V.figure(d, "vratio", text)
    assert f["series"][0]["y"][0] == 1.0 and f["series"][0]["y"][-1] == pytest.approx(11.2326 / 11.2261, rel=1e-6)
    short = (FULL.parent / "dia_qha.out").read_text()
    ds = A.analyze(short)
    assert "thermo" not in ds["qha"] and [v["key"] for v in V.views(ds, short)] == ["volume", "vratio", "expansion", "bulk", "heat"]
    from mace.elastic import properties as P
    props = P.all_properties(d)
    assert props["qha_gamma_mode_300k"]["value"] == 0.919 and props["qha_cp_300k"]["unit"] == "J/(mol K)"


def test_mace_elastic_analyze_draws_the_qha_plots(tmp_path):
    shutil.copy(FULL, tmp_path / FULL.name)
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "analyze", "elastic", FULL.name, "--plot", "gamma",
                        "--plot", "free", "--plot", "entropy", "--plot", "fv", "--plot", "vratio", "--plot", "heat", "--format", "svg"],
                       capture_output=True, text=True, cwd=tmp_path, timeout=300,
                       env={"MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)})
    assert r.returncode == 0, r.stderr
    for name in ("gamma", "free", "entropy", "fv", "vratio", "heat"):
        assert (tmp_path / f"dia_qha_full.{name}.svg").stat().st_size > 10000, name


def test_the_300_k_row_is_one_rule_for_the_database_and_the_gui(text):
    """The *_300k values and the GUI's γ (mode) tile read the same row (properties.nearest_300k): the
    temperature nearest 300 K, none when no row is within 15 K (the tile showed the last row)."""
    from mace.elastic import properties as P
    assert P.nearest_300k([10, 290, 310, 600]) == 1 and P.nearest_300k([10, 290, 302]) == 2
    assert P.nearest_300k([10, 100, 600]) is None and P.nearest_300k([]) is None and P.nearest_300k(None) is None
    doc = V.document(text, FULL)
    g = doc["data"]["qha"]["gruneisen"]
    i = doc["gruneisen_300k"]
    assert g["temperature_k"][i] == 300.0 and g["gamma"][i] == P.all_properties(doc["data"])["qha_gamma_mode_300k"]["value"]
