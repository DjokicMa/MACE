"""Tensors, moduli and EOS fits of real CRYSTAL23 runs (mace/elastic/analysis.py).

The outputs are diamond (B3LYP-D3) runs of decks MACE wrote, run on HPCC
with CRYSTAL/23-intel-2023a (tests/data/hpcc_116/README.md): ELASTCON
(NUCHESS, HESSNUM2, PRINT), ELAPIEZO (PRINT; diamond is centrosymmetric, so
CRYSTAL prints no piezoelectric tensor), PHOTOELA and EOS (PRINT). The
derived moduli are checked against the Voigt-Reuss-Hill table CRYSTAL itself
prints, and the ELASTCON bulk modulus against the EOS fit of the same
structure.
"""
import json
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT
from mace.elastic import analysis as A
from mace.utils.property_extractor import CrystalPropertyExtractor

MECH = REPO_ROOT / "tests" / "data" / "hpcc_116" / "mechanics"
MACE_CLI = REPO_ROOT / "mace_cli"


def _data(name):
    return A.analyze_file(MECH / f"{name}.out")


def test_elastcon_tensor_compliance_and_seismic_velocities():
    d = _data("dia_elastcon")
    el = d["elastic"]
    assert el["unit"] == "GPa" and el["system"] == "CUBIC" and el["symmetrized"]
    c = el["matrix"]
    assert (c[0][0], c[0][1], c[3][3]) == (1150.388, 122.664, 618.258)
    assert all(c[i][j] == c[j][i] for i in range(6) for j in range(6))
    # the compliance printed with the final constants, not a later per-fit block
    assert d["compliance"]["unit"] == "TPa^-1" and d["compliance"]["matrix"][0][0] == 0.8875
    assert len(d["seismic"]) == 7 and d["seismic"][6] == {"direction": [1.0, 1.0, 1.0], "vp": 18.983, "vs1": 12.382, "vs2": 12.382}
    assert d["dimensionality"] == 3 and d["n_atoms"] == 2 and d["volume_ang3"] == pytest.approx(11.135747)


def test_moduli_equal_crystal_s_own_vrh_table():
    d = _data("dia_elastcon")
    m, v = d["moduli"], d["crystal_vrh"]
    for ours, theirs in (("bulk_modulus_voigt_gpa", "K_V"), ("shear_modulus_voigt_gpa", "G_V"),
                         ("bulk_modulus_reuss_gpa", "K_R"), ("shear_modulus_reuss_gpa", "G_R"),
                         ("bulk_modulus_hill_gpa", "K_H"), ("shear_modulus_hill_gpa", "G_H"),
                         ("youngs_modulus_gpa", "E_H")):
        assert m[ours] == pytest.approx(v[theirs], abs=0.02), ours
    assert m["poisson_ratio"] == pytest.approx(v["v_H"], abs=5e-4)
    assert m["born_stable"] and m["min_eigenvalue_gpa"] == pytest.approx(618.258)
    # Debye temperature of diamond from the elastic constants (exp. ~2230 K)
    assert 2100 < m["debye_temperature_k"] < 2450


def test_elastcon_bulk_modulus_agrees_with_the_eos_fit():
    k = _data("dia_elastcon")["moduli"]["bulk_modulus_hill_gpa"]
    eos = _data("dia_eos")["eos"]
    assert set(eos["fits"]) == {"murnaghan", "birch_murnaghan", "poirier_tarantola", "vinet"}
    bm = eos["fits"]["birch_murnaghan"]
    assert bm == {"volume_ang3": 11.1422, "energy_au": -76.21722625, "bulk_modulus_gpa": 464.52, "bulk_modulus_prime": 3.61}
    assert len(eos["points"]) == 7 and eos["points"][3] == [11.135747, -76.2172261633]
    assert k == pytest.approx(bm["bulk_modulus_gpa"], rel=0.005)


def test_elapiezo_of_a_centrosymmetric_crystal_has_no_piezo_tensor():
    d = _data("dia_elapiezo")
    assert d["elastic"]["matrix"][0][0] == pytest.approx(1150.401)
    assert "piezo_e" not in d and "piezo_d" not in d
    text = (MECH / "dia_elapiezo.out").read_text()
    assert "NO PIEZOELECTRICITY FOR CENTROSYMMETRIC CRYSTALS" in text


def test_an_output_without_results_is_empty():
    text = (REPO_ROOT / "tests" / "fixtures" / "phonopy_crystal" / "Si" / "parent.out").read_text()
    assert A.analyze(text) == {}
    assert "no elastic" in A.report({}, "x.out")


def test_the_property_extractor_files_the_moduli():
    ex = CrystalPropertyExtractor.__new__(CrystalPropertyExtractor)
    props = ex._extract_mechanical_properties((MECH / "dia_elastcon.out").read_text())
    assert props["bulk_modulus_hill_gpa"] == pytest.approx(465.24, abs=0.01)
    assert props["elastic_born_stable"] is True and props["elastic_tensor"]["unit"] == "GPa"
    props = ex._extract_mechanical_properties((MECH / "dia_eos.out").read_text())
    assert props["eos_bulk_modulus_gpa"] == 464.52 and props["eos_equilibrium_volume_ang3"] == 11.1422
    from mace.utils import property_keys as PK
    assert all(PK.KEYS[k].category == "mechanical" for k in props if not k.startswith(("dielectric", "refractive")))


def test_mace_elastic_analyze_json_and_text():
    env = dict(os.environ, MACE_NO_BANNER="1")
    r = subprocess.run([sys.executable, str(MACE_CLI), "analyze", "elastic", str(MECH / "dia_elastcon.out"), "--json"],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert d["moduli"]["bulk_modulus_hill_gpa"] == pytest.approx(465.24, abs=0.01)
    r = subprocess.run([sys.executable, str(MACE_CLI), "analyze", "elastic", str(MECH / "dia_eos.out")],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0 and "birch_murnaghan" in r.stdout and "464.52" in r.stdout


def test_2d_moduli_of_a_slab_block():
    """In-plane block of a 6x6 C given in hartree, normalized by the cell area."""
    c = [[0.0] * 6 for _ in range(6)]
    c[0][0] = c[1][1] = 1.0
    c[0][1] = c[1][0] = 0.2
    c[5][5] = 0.4
    m = A.moduli_2d(c, "HARTREE", area_ang2=10.0)
    assert m["unit"] == "N/m" and m["born_stable"]
    s = A.HARTREE_J / 10e-20
    assert m["youngs_modulus_x"] == pytest.approx((1.0 - 0.04) * s)
    assert m["poisson_ratio_xy"] == pytest.approx(0.2)


def test_photoela_pockels_and_piezo_optic_tensors():
    """Diamond: p11 -0.280, p12 0.072, p44 -0.188 (exp. about -0.25 to -0.43, 0.04, -0.17)."""
    d = _data("dia_photoela")
    p, pi = d["pockels"]["matrix"], d["piezo_optic"]["matrix"]
    assert (p[0][0], p[0][1], p[3][3]) == (-0.28, 0.072, -0.188)
    assert (pi[0][0], pi[0][1], pi[3][3]) == (-0.261, 0.082, -0.304)
    assert d["piezo_optic"]["unit"].startswith("Brewster")
    # PHOTOELA computes the elastic constants it needs as well
    assert d["elastic"]["matrix"][0][0] == pytest.approx(1150.37)
    ex = CrystalPropertyExtractor.__new__(CrystalPropertyExtractor)
    props = ex._extract_mechanical_properties((MECH / "dia_photoela.out").read_text())
    assert props["photoelastic_tensor"]["matrix"][0][0] == -0.28


def test_cphf_dielectric_tensor_and_hyperpolarizabilities():
    """CPHF with FOURTH on diamond: eps_inf 5.387 (exp. 5.7), beta zero by symmetry, gamma_xxxx 1.0941e4 a.u."""
    d = _data("dia_cphf")
    de = d["dielectric"]
    xx = next(r for r in de["components"] if r["component"] == "XX")
    assert (xx["alpha"], xx["epsilon"], xx["chi1"]) == (26.2346, 5.387, 4.387)
    assert de["principal_epsilon"] == [5.387002, 5.387002, 5.387002]
    assert de["refractive_indices"] == [2.320992, 2.320992, 2.320992]
    assert all(abs(r["value"]) < 1e-20 for r in d["beta"]["components"]) and len(d["beta"]["components"]) == 10
    g = {r["component"]: r["value"] for r in d["gamma"]["components"]}
    assert g["XXXX"] == 1.0941e4 and g["XXYY"] == 5.2224e3 and len(g) == 15
    ex = CrystalPropertyExtractor.__new__(CrystalPropertyExtractor)
    props = ex._extract_mechanical_properties((MECH / "dia_cphf.out").read_text())
    assert props["dielectric_constant_electronic"] == pytest.approx(5.387002)
    assert props["refractive_index_electronic"] == pytest.approx(2.320992)
    assert "hyperpolarizability_gamma" in props


def test_qha_thermal_properties():
    """Diamond, Gamma-only QHA (4 volumes, TEMPERAT 60 10 600): the zero-pressure tables."""
    d = _data("dia_qha")
    q = d["qha"]
    assert len(q["temperature_k"]) == 60 and q["temperature_k"][0] == 10.0 and q["temperature_k"][-1] == 600.0
    i = q["temperature_k"].index(310.0)
    assert (q["volume_ang3"][i], q["bulk_modulus_t_gpa"][i], q["alpha_v_per_k"][i]) == (11.2263, 458.885, 4.7334e-07)
    assert len(q["lattice"]["a_ang"]) == 60 and q["lattice"]["a_ang"][0] == 3.55437062
    ex = CrystalPropertyExtractor.__new__(CrystalPropertyExtractor)
    props = ex._extract_mechanical_properties((MECH / "dia_qha.out").read_text())
    assert props["qha_bulk_modulus_300k_gpa"] == pytest.approx(458.885, abs=0.2)


def test_elapiezo_piezoelectric_tensors_of_cubic_bn():
    """Zincblende BN (PBE0, POB-TZVP-REV2, optimized by MACE's own OPT deck; ELAPIEZO + RESTART):
    C11 839.5, C12 183.3, C44 486.3 GPa (exp. about 820, 190, 480); e14 -0.760 C/m^2, d14 -1.563 pC/N."""
    d = _data("bn_elapiezo")
    c = d["elastic"]["matrix"]
    assert (c[0][0], c[0][1], c[3][3]) == (839.53, 183.336, 486.325)
    e, dd = d["piezo_e"], d["piezo_d"]
    assert e["unit"] == "C/m^2" and e["system"] == "CUBIC" and dd["unit"] == "pC/N = pm/V"
    assert [e["matrix"][i][3 + i] for i in range(3)] == [-0.76, -0.76, -0.76]
    assert [dd["matrix"][i][3 + i] for i in range(3)] == [-1.563, -1.563, -1.563]
    assert sum(abs(x) for row in e["matrix"] for x in row) == pytest.approx(3 * 0.76)
    assert d["moduli"]["bulk_modulus_hill_gpa"] == pytest.approx(d["crystal_vrh"]["K_H"], abs=0.02)
    assert "RESTART" in (MECH / "bn_elapiezo.d12").read_text().split()
