"""EOS analyses (1.1.6): B(P) and V/V0(P) of each fit, each fit's residuals and RMS, the enthalpy
H(P) = E + PV per formula unit and the transition pressure between two phases of one composition
(mace analyze elastic --plot bp|vp|residuals|enthalpy, --compare; GET elastic/figure?view=enthalpy&compare=).

The real diamond EOS run of tests/data/hpcc_116/mechanics against a made-up denser carbon phase
(tests/eos_synthetic.py); the transition pressure is checked against an independent root search
(scipy) on the Birch-Murnaghan curves, the bulk modulus against B0 and B' at P = 0.
"""
import json
import math
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT
from eos_synthetic import GPA_A3_HA, bm_energy, eos_output
from mace.elastic import analysis as A
from mace.elastic import views as V

MECH = REPO_ROOT / "tests" / "data" / "hpcc_116" / "mechanics"
MACE_CLI = REPO_ROOT / "mace_cli"
HP = dict(v0=10.80, e0=-76.21522625, b0=480.0, bp=3.7)


def _bm_p(v, v0, b0, bp):
    x = (v0 / v) ** (1 / 3)
    return 1.5 * b0 * (x ** 7 - x ** 5) * (1 + 0.75 * (bp - 4) * (x * x - 1))


def test_bulk_modulus_and_volume_along_pressure():
    d = A.analyze_file(MECH / "dia_eos.out")
    for name, f in d["eos"]["fits"].items():
        v0, b0, bp = f["volume_ang3"], f["bulk_modulus_gpa"], f["bulk_modulus_prime"]
        assert V.eos_bulk(name, v0, f) == pytest.approx(b0, rel=1e-6)
        # dB/dP at P = 0 is B'
        h = 1e-3 * v0
        db = V.eos_bulk(name, v0 - h, f) - V.eos_bulk(name, v0 + h, f)
        dp = V.eos_pressure(name, v0 - h, f) - V.eos_pressure(name, v0 + h, f)
        assert db / dp == pytest.approx(bp, rel=1e-3)
    f = V.figure(d, "vp")
    bm = next(s for s in f["series"] if s["key"] == "birch_murnaghan")
    k = min(range(len(bm["x"])), key=lambda i: abs(bm["x"][i]))
    assert abs(bm["y"][k] - 1) < 2e-3 and f["y_label"] == "V/V<sub>0</sub>"
    f = V.figure(d, "bp")
    assert f["x_label"] == "P (GPa)" and V.table(f)["columns"] == ["series", "P (GPa)", "B (GPa)"]


def test_residuals_and_their_rms():
    d = A.analyze_file(MECH / "dia_eos.out")
    pts, q = d["eos"]["points"], d["eos"]["quality"]
    for name, f in d["eos"]["fits"].items():
        res = [(e - V.eos_energy(name, v, f)) * V.HARTREE_MEV for v, e in pts]
        assert q[name]["residuals_mev"] == pytest.approx(res, abs=1e-6)
        assert q[name]["rms_mev"] == pytest.approx(math.sqrt(sum(r * r for r in res) / len(res)), rel=1e-5)
        assert q[name]["quality"] == "good"          # a well-converged run: RMS ~0.01 meV over ~100 meV
    f = V.figure(d, "residuals")
    assert f["zero_line"] and f["series"][1]["name"].startswith("Birch-Murnaghan 1947 (RMS ")
    assert "residuals: RMS" in A.report(d)


def test_enthalpy_and_the_transition_pressure_against_an_independent_root_search():
    scipy = pytest.importorskip("scipy.optimize")
    ref = A.analyze_file(MECH / "dia_eos.out")
    hp = A.analyze(eos_output(**HP))
    assert hp["formula"] == ref["formula"] == "C" and hp["formula_units"] == 2
    t = V.transitions(ref, hp)
    assert t["fit"] == "birch_murnaghan" and len(t["crossings"]) == 1 and t["crossings"][0]["stable_above"] == "other"
    fa = ref["eos"]["fits"]["birch_murnaghan"]

    def h(p, v0, e0, b0, bp):
        v = scipy.brentq(lambda v: _bm_p(v, v0, b0, bp) - p, 0.7 * v0, 1.3 * v0, xtol=1e-12)
        return (bm_energy(v, v0, e0, b0, bp) + p * v * GPA_A3_HA) / 2

    a = (fa["volume_ang3"], fa["energy_au"], fa["bulk_modulus_gpa"], fa["bulk_modulus_prime"])
    b = (HP["v0"], HP["e0"], HP["b0"], HP["bp"])
    pt = scipy.brentq(lambda p: h(p, *b) - h(p, *a), 0.0, 45.0, xtol=1e-9)
    assert t["crossings"][0]["pressure_gpa"] == pytest.approx(pt, abs=0.02)          # ~26.9 GPa
    # the single-phase curve: H = E0 at P = 0 (relative 0), dH/dP = V per formula unit
    f = V.figure(ref, "enthalpy", name="dia_eos.out")
    bm = next(s for s in f["series"] if s["key"] == "birch_murnaghan")
    k = min(range(len(bm["x"])), key=lambda i: abs(bm["x"][i]))
    slope = (bm["y"][k + 1] - bm["y"][k - 1]) / (bm["x"][k + 1] - bm["x"][k - 1])           # eV/(f.u. GPa)
    assert slope == pytest.approx(fa["volume_ang3"] / 2 / 160.21766, rel=2e-3)
    assert abs(bm["y"][k] - slope * bm["x"][k]) < 2e-5
    f = V.figure(ref, "enthalpy", compare=[("c_hp.out", hp)], name="dia_eos.out")
    assert f["title"] == "ΔH(P)" and f["markers"] == [t["crossings"][0]["pressure_gpa"]]
    assert [s["key"] for s in f["series"]] == ["reference", "c_hp.out"] and "P<sub>t</sub> = 26.9" in f["note"]



def test_the_same_run_twice_has_no_transition():
    """dH = 0 at every pressure: each grid point counted as a crossing before (400 of them)."""
    ref = A.analyze_file(MECH / "dia_eos.out")
    t = V.transitions(ref, ref)
    assert t["overlap"] and t["same"] and t["crossings"] == []
    f = V.figure(ref, "enthalpy", compare=[("again.out", ref)], name="dia_eos.out")
    assert f["markers"] == [] and "the same enthalpy (within 0.01 meV/f.u.) at every pressure: no transition" in f["note"]
    # a real difference within the tolerance band near the crossing still gives one transition
    hp = A.analyze(eos_output(**HP))
    assert len(V.transitions(ref, hp)["crossings"]) == 1 and not V.transitions(ref, hp)["same"]

def test_phases_of_another_composition_are_refused():
    ref = A.analyze_file(MECH / "dia_eos.out")
    bn = A.analyze(eos_output(11.8, -79.5, 380.0, 3.8, atoms=(("5", "B"), ("7", "N"))))
    with pytest.raises(ValueError, match="differ in composition"):
        V.transitions(ref, bn)


def _cli(*args, cwd):
    return subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "analyze", "elastic", *args],
                          capture_output=True, text=True, cwd=cwd, timeout=300,
                          env={"MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin", "HOME": str(cwd)})


def test_mace_elastic_analyze_compare_json_text_and_plots(tmp_path):
    shutil.copy(MECH / "dia_eos.out", tmp_path / "dia_eos.out")
    (tmp_path / "c_hp_eos.out").write_text(eos_output(**HP))
    (tmp_path / "bn_eos.out").write_text(eos_output(11.8, -79.5, 380.0, 3.8, atoms=(("5", "B"), ("7", "N"))))
    r = _cli("dia_eos.out", "--compare", "c_hp_eos.out", "--json", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    js = json.loads(r.stdout)
    t = V.transitions(A.analyze_file(tmp_path / "dia_eos.out"), A.analyze_file(tmp_path / "c_hp_eos.out"))
    assert js["transitions"][0]["crossings"] == t["crossings"] and js["eos"]["quality"]["vinet"]["quality"] == "good"
    r = _cli("dia_eos.out", "--compare", "c_hp_eos.out", "--fit", "vinet", cwd=tmp_path)
    assert "Enthalpy against c_hp_eos.out (vinet): transition at" in r.stdout
    r = _cli("dia_eos.out", "--compare", "bn_eos.out", cwd=tmp_path)
    assert r.returncode == 1 and "differ in composition" in r.stderr
    r = _cli("dia_eos.out", "--compare", "c_hp_eos.out", "--plot", "enthalpy", "--plot", "bp", "--plot", "vp",
             "--plot", "residuals", "--format", "svg", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    for name in ("enthalpy", "bp", "vp", "residuals"):
        assert (tmp_path / f"dia_eos.{name}.svg").stat().st_size > 10000, name
