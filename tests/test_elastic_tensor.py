"""Directional and derived elastic properties (1.1.6, mace/elastic/tensor.py; ELATE conventions,
Gaillac, Pullumbi & Coudert, J. Phys.: Condens. Matter 28, 275201 (2016)).

Checked against hand values and closed forms, not against the code's own formulas:

* G(n, m) and nu(n, m) of a triclinic tensor: the exact min/max over m (2x2 eigenvalues) against a
  brute-force scan of m around n with the compliance rotated into the (n, m, n x m) frame;
* a cubic crystal (diamond, the real HPCC ELASTCON run): G_min = (C11 - C12)/2 along <110>/<1-10>,
  G_max = C44, nu(<110>, <1-10>) = -(S12 + S0/2)/(S11 - S0/2) and nu(<110>, <001>) = -S12/(S11 - S0/2)
  with S0 = S11 - S12 - S44/2 (Nye 1985), A^U = 6/5 (sqrt(A) - 1/sqrt(A))^2 (Ranganathan 2008);
* the Christoffel phase velocities equal the table CRYSTAL prints (SEISMIC VELOCITIES);
* an isotropic C gives flat E, beta, G, nu and sound velocities with their textbook values;
* the hardness and minimum thermal conductivity formulas by hand; BN's d_L along [111] = d14/sqrt(3);
* a slab: the in-plane E, nu, G and A^SU of an isotropic and an orthotropic block;
* the real diamond ELASTCON output with other constants in its table (_elastcon): an unstable C
  flagged and its derived numbers not stored, a soft solid's Chen hardness clamped at 0, and the
  Cartesian direction labels of a tetragonal crystal.
"""
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT
from mace.elastic import analysis as A
from mace.elastic import tensor as T
from mace.elastic import views as V

MECH = REPO_ROOT / "tests" / "data" / "hpcc_116" / "mechanics"
MACE_CLI = REPO_ROOT / "mace_cli"


def _iso(lam=100.0, mu=80.0):
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[[0, 1, 2], [0, 1, 2]] = lam + 2 * mu
    C[[3, 4, 5], [3, 4, 5]] = mu
    return C


def _iso_data(lam=100.0, mu=80.0, rho=3.0):
    C = _iso(lam, mu)
    d = {"elastic": {"matrix": C.tolist(), "unit": "GPa", "system": "CUBIC"}, "dimensionality": 3,
         "density_g_cm3": rho, "n_atoms": 2, "volume_ang3": 20.0}
    d["moduli"] = A.moduli_3d(C, None, rho, 2, 20.0)
    return d


def _rotated(St, n, m):
    """S'_1111, S'_1122, S'_1212 in the frame (n, m, n x m): the brute-force reference."""
    R = np.array([n, m, np.cross(n, m)])
    Sp = np.einsum("ai,bj,ck,dl,ijkl->abcd", R, R, R, R, St)
    return Sp[0, 0, 0, 0], Sp[0, 0, 1, 1], Sp[0, 1, 0, 1]


def test_shear_and_poisson_extremes_over_m_match_a_brute_force_scan():
    rng = np.random.default_rng(7)
    B = rng.normal(size=(6, 6))
    C = B @ B.T * 40 + np.diag([300, 280, 260, 90, 80, 70])          # triclinic, positive definite
    St = T.voigt_to_tensor(np.linalg.inv(C), True)
    dirs = T.sphere_grid(15.0)[::7]
    got = T.shear_poisson_extremes(St, dirs)
    for k, n in enumerate(dirs):
        u, v = T.perp_basis(n[None])
        g, nu = [], []
        for chi in np.radians(np.arange(0.0, 180.0, 0.05)):
            m = math.cos(chi) * u[0] + math.sin(chi) * v[0]
            s1111, s1122, s1212 = _rotated(St, n, m)
            g.append(1 / (4 * s1212))
            nu.append(-s1122 / s1111)
        assert got["g_min"][k] == pytest.approx(min(g), rel=1e-6) and got["g_max"][k] == pytest.approx(max(g), rel=1e-6)
        assert got["nu_min"][k] == pytest.approx(min(nu), abs=1e-7) and got["nu_max"][k] == pytest.approx(max(nu), abs=1e-7)
        # the m returned gives that value
        for key, val in (("g_min", 1 / (4 * _rotated(St, n, got["m_g_min"][k])[2])),
                         ("nu_max", -_rotated(St, n, got["m_nu_max"][k])[1] / _rotated(St, n, got["m_nu_max"][k])[0])):
            assert got[key][k] == pytest.approx(val, rel=1e-9)


def test_diamond_shear_poisson_and_anisotropy_against_the_cubic_closed_forms():
    d = A.analyze_file(MECH / "dia_elastcon.out")
    C = np.array(d["elastic"]["matrix"])
    S = np.linalg.inv((C + C.T) / 2)
    s11, s12, s44 = S[0, 0], S[0, 1], S[3, 3]
    s0 = s11 - s12 - s44 / 2
    der = d["derived"]
    G, nu = der["directional"]["shear"], der["directional"]["poisson"]
    assert G["min"]["value"] == pytest.approx((C[0, 0] - C[0, 1]) / 2, rel=1e-5)          # 513.86 GPa
    assert G["min"]["label"] == "[1 0 1]" and G["min"]["label2"] == "[1 0 −1]"
    assert G["max"]["value"] == pytest.approx(C[3, 3], rel=1e-5)                          # C44 618.258 GPa
    assert nu["min"]["value"] == pytest.approx(-(s12 + s0 / 2) / (s11 - s0 / 2), rel=1e-4)   # 0.0042, <110>/<1-10>
    assert nu["max"]["value"] == pytest.approx(-s12 / (s11 - s0 / 2), rel=1e-4)              # 0.106, <110>/<001>
    assert nu["max"]["label2"] == "[0 1 0]" and not nu["auxetic"]
    A_z = 2 * C[3, 3] / (C[0, 0] - C[0, 1])
    idx = der["indices"]
    assert idx["zener"] == pytest.approx(A_z, rel=1e-5)
    assert idx["universal"] == pytest.approx(1.2 * (math.sqrt(A_z) - 1 / math.sqrt(A_z)) ** 2, rel=1e-4)
    assert idx["chung_buessem"] == pytest.approx(3 * (A_z - 1) ** 2 / (3 * (A_z - 1) ** 2 + 25 * A_z), rel=1e-3)
    assert idx["g_ratio"] == pytest.approx(A_z, rel=1e-5)          # diamond: G_max/G_min = C44 / ((C11 - C12)/2)
    # literature: diamond's Hill Poisson's ratio ~0.07 and A^U ~0.04 (Zener ~1.2)
    assert 0.05 < d["moduli"]["poisson_ratio"] < 0.08 and 0.02 < idx["universal"] < 0.06
    ch = der["character"]
    assert ch["cauchy"] == [{"relation": "C12 − C44", "value": pytest.approx(C[0, 1] - C[3, 3], abs=1e-3)}]
    assert ch["pugh_ratio"] == pytest.approx(d["moduli"]["bulk_modulus_hill_gpa"] / d["moduli"]["shear_modulus_hill_gpa"], rel=1e-5)
    assert not ch["ductile"]


def test_hardness_and_minimum_thermal_conductivity_by_hand():
    d = A.analyze_file(MECH / "dia_elastcon.out")
    m = d["moduli"]
    K, G, E = m["bulk_modulus_hill_gpa"], m["shear_modulus_hill_gpa"], m["youngs_modulus_gpa"]
    k = G / K
    hv = d["derived"]["character"]["hardness"]
    assert hv["chen_gpa"] == pytest.approx(2 * (k * k * G) ** 0.585 - 3, rel=1e-5)        # ~102 GPa
    assert hv["tian_gpa"] == pytest.approx(0.92 * k ** 1.137 * G ** 0.708, rel=1e-5)      # ~105 GPa
    th = d["derived"]["thermal"]
    rho, V, N = 3579.0, 11.135747e-30, 2
    ma = rho * V / N
    assert th["mean_atomic_mass_amu"] == pytest.approx(12.0, abs=0.01)                    # carbon
    assert th["clarke_w_mk"] == pytest.approx(0.87 * 1.380649e-23 * ma ** (-2 / 3) * math.sqrt(E * 1e9) * rho ** (1 / 6), rel=1e-5)
    cahill = (math.pi / 6) ** (1 / 3) * 1.380649e-23 * (N / V) ** (2 / 3) * (m["sound_velocity_longitudinal_km_s"]
                                                                          + 2 * m["sound_velocity_transverse_km_s"]) * 1e3
    assert th["cahill_w_mk"] == pytest.approx(cahill, rel=1e-5)
    assert th["debye_temperature_k"] == pytest.approx(m["debye_temperature_k"], rel=1e-5)


@pytest.mark.parametrize("name", ["dia_elastcon", "bn_elapiezo"])
def test_christoffel_phase_velocities_equal_crystal_s_seismic_table(name):
    d = A.analyze_file(MECH / f"{name}.out")
    got = T.compliance_of(d)
    ch = T.christoffel(got[1], [r["direction"] for r in d["seismic"]], d["density_g_cm3"])
    for r, v in zip(d["seismic"], ch["phase"]):
        assert v == pytest.approx([r["vp"], r["vs1"], r["vs2"]], abs=2e-3)
    # the group velocity's projection on n is the phase velocity
    n = T._unit([r["direction"] for r in d["seismic"]])
    assert np.einsum("nlm,nl->nm", ch["group_vector"], n) == pytest.approx(ch["phase"], rel=1e-9)


def test_an_isotropic_tensor_gives_flat_surfaces():
    lam, mu, rho = 100.0, 80.0, 3.0
    der = T.derived(_iso_data(lam, mu, rho))
    want = {"youngs": mu * (3 * lam + 2 * mu) / (lam + mu), "compressibility": 1000 / (3 * lam + 2 * mu),
            "shear": mu, "poisson": lam / (2 * (lam + mu))}
    for k, v in want.items():
        e = der["directional"][k]
        assert e["isotropic"] and e["min"]["value"] == pytest.approx(v, rel=1e-5) and e["max"]["value"] == pytest.approx(v, rel=1e-5)
    s = der["directional"]["sound"]
    for k, v in (("qL", math.sqrt((lam + 2 * mu) / rho)), ("qS1", math.sqrt(mu / rho)), ("qS2", math.sqrt(mu / rho))):
        assert s[k]["isotropic"] and s[k]["max"]["value"] == pytest.approx(v, rel=1e-5)
        assert s[k]["group"]["max"]["value"] == pytest.approx(v, rel=1e-5)
    assert der["indices"]["universal"] == pytest.approx(0, abs=1e-9) and der["indices"]["zener"] == pytest.approx(1)
    assert der["indices"]["e_ratio"] == pytest.approx(1) and der["indices"]["nu_ratio"] == pytest.approx(1)
    # the plots are flat too: every point of the G and nu envelopes, and every cut, at the same value
    d = _iso_data(lam, mu, rho)
    d["derived"] = der
    for view in ("shear", "poisson"):
        f = V.figure(d, view, "", "surface")
        for env in ("min", "max"):
            vals = np.array(f["envelopes"][env]["value"], dtype=float)
            assert vals == pytest.approx(want[view], rel=1e-4)
    f = V.figure(d, "sound", "", "surface")
    assert f["shape"] == "planes" and np.array(f["planes"][1]["curves"][0]["r"]) == pytest.approx(math.sqrt((lam + 2 * mu) / rho), rel=1e-5)


def test_piezoelectric_longitudinal_coefficient_and_coupling_of_bn():
    d = A.analyze_file(MECH / "bn_elapiezo.out")
    d14 = d["piezo_d"]["matrix"][0][3]                                  # -1.563 pC/N (cubic -43m: d14 only)
    pz = d["derived"]["piezo"]
    assert pz["kind"] == "d" and pz["axes"] == [0.0, 0.0, 0.0]
    assert pz["longitudinal"]["max"]["label"] == "[1 1 1]"
    assert pz["longitudinal"]["max"]["signed"] == pytest.approx(d14 / math.sqrt(3), rel=1e-5)   # 3 d14 n1 n2 n3
    assert pz["coupling"] == {"available": False}                     # ELAPIEZO prints no permittivity
    eps = 6.8
    d2 = A.analyze_file(MECH / "bn_elapiezo.out", "6.8")
    cp = d2["derived"]["piezo"]["coupling"]
    n = np.ones(3) / math.sqrt(3)
    E111 = float(T.youngs(T.compliance_of(d2)[0], [n])[0])
    k2 = (d14 / math.sqrt(3) * 1e-12) ** 2 * E111 * 1e9 / (8.8541878128e-12 * eps)
    assert cp["available"] and cp["epsilon_source"] == "given" and cp["k2_axes"] == [0.0, 0.0, 0.0]
    assert cp["k2_max"]["label"] == "[1 1 1]" and cp["k2_max"]["value"] == pytest.approx(k2, rel=1e-4)
    with pytest.raises(ValueError):
        A.analyze_file(MECH / "bn_elapiezo.out", "1 2")
    # the e tensor alone (a PIEZOCON run): e_L(n) = e_ijk n_i n_j n_k, e14 2 n1 n2 n3 * 3 along [111]
    e14 = d["piezo_e"]["matrix"][0][3]
    pe = T.piezo({"piezo_e": d["piezo_e"]})
    assert pe["kind"] == "e" and pe["longitudinal"]["max"]["signed"] == pytest.approx(2 * e14 / math.sqrt(3), rel=1e-5)


def test_piezo_figure_cuts_through_the_largest_coefficient():
    d = A.analyze_file(MECH / "bn_elapiezo.out")
    f = V.figure(d, "piezo", "", "planes")
    # -43m: zero in the xy, xz and yz planes, so the cut through z and [111] only
    assert [c["plane"] for c in f["planes"]] == ["(1 −1 0)"] and f["planes"][0]["axes"] == ["[1 1 0]", "[0 0 1]"]
    r = f["planes"][0]["r"]
    assert max(abs(x) for x in r) == pytest.approx(abs(d["piezo_d"]["matrix"][0][3]) / math.sqrt(3), rel=1e-3)
    s = V.figure(d, "piezo", "", "surface")
    assert s["negative"] and min(min(row) for row in s["surface"]["value"]) < 0 < max(max(row) for row in s["surface"]["value"])
    assert V.table(s)["columns"] == ["n_x", "n_y", "n_z", "d_L (pC/N)"]


def test_a_slab_in_plane_shear_poisson_and_2d_anisotropy():
    # isotropic in-plane block (C66 = (C11 - C12)/2), N/m
    c11, c12 = 100.0, 30.0
    d = {"elastic": {"matrix": [[c11, c12, 0], [c12, c11, 0], [0, 0, (c11 - c12) / 2]], "unit": "N/m"}, "dimensionality": 2}
    d["moduli"] = A.moduli_2d(d["elastic"]["matrix"], "N/m")
    der = T.derived(d)
    for k, v in (("youngs", (c11 ** 2 - c12 ** 2) / c11), ("poisson", c12 / c11), ("shear", (c11 - c12) / 2)):
        assert der["directional"][k]["isotropic"] and der["directional"][k]["max"]["value"] == pytest.approx(v, rel=1e-5)
    assert der["two_d"]["anisotropy_su"] == pytest.approx(0, abs=1e-9)
    # orthotropic: E along x = (C11 C22 - C12^2)/C22, nu_xy = C12/C22, G at 0 degrees = C66; A^SU by hand
    C = np.array([[200.0, 40.0, 0], [40.0, 100.0, 0], [0, 0, 30.0]])
    d = {"elastic": {"matrix": C.tolist(), "unit": "N/m"}, "dimensionality": 2}
    d["moduli"] = A.moduli_2d(C, "N/m")
    der = T.derived(d)
    St = T.voigt_to_tensor(np.linalg.inv(C), True)
    x = [[1.0, 0.0, 0.0]]
    assert T.youngs(St, x)[0] == pytest.approx((200 * 100 - 1600) / 100)
    assert T.poisson(St, x, [[0.0, 1.0, 0.0]])[0] == pytest.approx(40 / 100)
    assert T.shear(St, x, [[0.0, 1.0, 0.0]])[0] == pytest.approx(30.0)
    S = np.linalg.inv(C)
    kv, gv = (200 + 100 + 80) / 4, (200 + 100 - 80 + 120) / 8
    kr, gr = 1 / (S[0, 0] + S[1, 1] + 2 * S[0, 1]), 2 / (S[0, 0] + S[1, 1] - 2 * S[0, 1] + S[2, 2])
    assert der["two_d"]["anisotropy_su"] == pytest.approx(math.sqrt((kv / kr - 1) ** 2 + 2 * (gv / gr - 1) ** 2), rel=1e-5)
    assert [v["key"] for v in V.views(d)] == ["youngs", "compressibility", "shear", "poisson"]
    f = V.figure(d, "shear", "", "surface")
    assert f["shape"] == "planes" and f["planes"][0]["r"][0] == pytest.approx(30.0, rel=1e-4)


def test_mace_elastic_analyze_json_epsilon_and_new_plots(tmp_path):
    env = {"MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    out = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "analyze", "elastic", str(MECH / "bn_elapiezo.out"),
                          "--json", "--epsilon", "6.8"], capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout)
    assert data["derived"]["piezo"]["coupling"]["epsilon_source"] == "given"
    assert data["derived"] == json.loads(json.dumps(A.analyze_file(MECH / "bn_elapiezo.out", "6.8")["derived"]))
    bad = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "analyze", "elastic", str(MECH / "bn_elapiezo.out"),
                          "--epsilon", "x"], capture_output=True, text=True, env=env, timeout=300)
    assert bad.returncode == 2 and "--epsilon" in bad.stderr
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "analyze", "elastic", str(MECH / "bn_elapiezo.out"),
                        "--plot", "shear", "--plot", "poisson", "--plot", "sound", "--plot", "piezo", "--format", "svg",
                        "-o", str(tmp_path)], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr
    for name in ("shear_surface", "poisson_surface", "sound_planes", "piezo_surface"):
        p = tmp_path / f"bn_elapiezo.{name}.svg"
        assert p.is_file() and p.stat().st_size > 10000, name
    # the text report names the new numbers
    t = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "analyze", "elastic", str(MECH / "dia_elastcon.out")],
                       capture_output=True, text=True, env=env, timeout=300).stdout
    for want in ("Shear modulus G(n, m): 513.9 GPa along [1 0 1], m [1 0 −1]", "Zener A 1.2032", "Pugh's ratio K/G 0.810 (brittle",
                 "Vickers hardness: Chen", "Minimum thermal conductivity: Clarke", "Sound velocity qL: phase 17.93 km/s"):
        assert want in t, want


def test_the_property_list_names_every_derived_scalar():
    from mace.elastic import properties as P
    props = P.all_properties(A.analyze_file(MECH / "bn_elapiezo.out", "6.8"))
    for name, unit in (("shear_modulus_min", "GPa"), ("poisson_ratio_min", ""), ("anisotropy_zener", ""),
                       ("vickers_hardness_chen_gpa", "GPa"), ("min_thermal_conductivity_cahill", "W/(m K)"),
                       ("sound_velocity_qL_max_km_s", "km/s"), ("piezo_d_longitudinal_max", "pC/N"),
                       ("piezo_coupling_k2_max", ""), ("cauchy_c12_minus_c44", "GPa")):
        assert props[name]["unit"] == unit, name
    assert props["poisson_ratio_min"]["value"] < 0 and props["ductile_pugh"]["value"] is False


def _elastcon(C, system="CUBIC", family=None):
    """The real diamond ELASTCON output with the constants C (upper triangle printed, GPa) and the
    system in its SYMMETRIZED ELASTIC CONSTANTS table (and the crystal family line, when given)."""
    text = (MECH / "dia_elastcon.out").read_text()
    head = " SYMMETRIZED ELASTIC CONSTANTS FOR CUBIC        CASE, IN GPa"
    i = text.index(head)
    j = text.index(" ELASTIC MODULI (COMPLIANCE TENSOR)", i)
    rows = [" |" + " " * 9 * k + "".join(f"{C[k][m]:9.3f}" for m in range(k, 6)) + " |" for k in range(6)]
    text = text[:i] + f" SYMMETRIZED ELASTIC CONSTANTS FOR {system:15s}CASE, IN GPa\n\n" + "\n".join(rows) + "\n\n\n" + text[j:]
    if family:
        text = text.replace(" CRYSTAL FAMILY                       :  CUBIC       ", f" CRYSTAL FAMILY                       :  {family:12s}", 1)
    return text


def _cubic(c11, c12, c44):
    C = np.zeros((6, 6))
    C[:3, :3] = c12
    np.fill_diagonal(C, [c11] * 3 + [c44] * 3)
    return C.tolist()


def test_an_unstable_tensor_is_flagged_and_its_derived_numbers_not_stored(tmp_path):
    """C44 < 0: before, E_min -3.8e8 GPa, the hardness, sound velocities and the rest (28 values) went into the
    database as if the crystal were stable."""
    from mace.elastic import properties as P
    text = _elastcon(_cubic(1150.388, 122.664, -20.0))
    d = A.analyze(text)
    assert d["moduli"]["born_stable"] is False and d["derived"]["unstable"]
    assert d["derived"]["directional"]["youngs"]["min"]["value"] < 0                 # unphysical, still shown
    assert d["warnings"][0].startswith("C is not positive definite (smallest eigenvalue of C -20 GPa)")
    rep = A.report(d)
    assert "WARNING: C is not positive definite" in rep and "UNPHYSICAL (C not positive definite)" in rep
    assert P.elastic_properties(d) == {}
    out = tmp_path / "unstable_elastcon.out"
    out.write_text(text)
    from mace.utils.property_extractor import CrystalPropertyExtractor
    import contextlib
    import io
    x = CrystalPropertyExtractor(db_path=None, enable_tracking=False)
    with contextlib.redirect_stdout(io.StringIO()):
        props = x.extract_all_properties(out)
    assert props["elastic_born_stable"] is False and props["bulk_modulus_hill_gpa"] == d["moduli"]["bulk_modulus_hill_gpa"]
    for k in ("youngs_modulus_gpa", "poisson_ratio", "debye_temperature_elastic_k", "youngs_modulus_min", "vickers_hardness_tian_gpa",
              "pugh_ratio"):
        assert k not in props, k
    # the stable diamond keeps them all
    assert "warnings" not in A.analyze_file(MECH / "dia_elastcon.out")


def test_a_soft_solid_s_chen_hardness_is_clamped_at_zero():
    """K = 100, G = 20 GPa: Chen's 2 (k^2 G)^0.585 - 3 = -1.24 GPa, stored before as a hardness."""
    from mace.elastic import properties as P
    d = A.analyze(_elastcon(_cubic(126.6667, 86.6667, 20.0)))
    assert d["moduli"]["bulk_modulus_hill_gpa"] == pytest.approx(100, abs=1e-3)
    assert d["moduli"]["shear_modulus_hill_gpa"] == pytest.approx(20, abs=1e-3)
    hv = d["derived"]["character"]["hardness"]
    assert hv["chen_gpa"] == 0 and hv["chen_clamped"]
    assert hv["chen_formula_gpa"] == pytest.approx(2 * (0.2 ** 2 * 20) ** 0.585 - 3, abs=1e-3)       # -1.24
    assert hv["tian_gpa"] == pytest.approx(0.92 * 0.2 ** 1.137 * 20 ** 0.708, rel=1e-4)
    props = P.elastic_properties(d)
    assert "vickers_hardness_chen_gpa" not in props and props["vickers_hardness_tian_gpa"]["value"] > 0
    assert "Chen 0 GPa (its formula gives -1.24 GPa: a soft solid outside the fit)" in A.report(d)
    assert T.hardness(400.0, 500.0) == {"chen_gpa": pytest.approx(2 * (1.25 ** 2 * 500) ** 0.585 - 3), "tian_gpa": pytest.approx(
        0.92 * 1.25 ** 1.137 * 500 ** 0.708)}


def test_directions_are_cartesian_unless_the_crystal_is_cubic():
    """A tetragonal crystal's Cartesian x+z is not its lattice [101] (c != a): before, every direction was
    labeled [u v w], and an extreme off the symmetry directions was snapped to small integers."""
    assert T.direction_label([0.6, 0.8, 0.0]) == "(0.600, 0.800, 0.000)"                # was [3 4 0]
    assert T.direction_label([0.6, 0.8, 0.0], lattice=True) == "(0.600, 0.800, 0.000)"
    s = 1 / math.sqrt(2)
    assert T.direction_label([s, 0, -s]) == "x−z" and T.direction_label([s, 0, -s], lattice=True) == "[1 0 −1]"
    assert T.direction_label([0, 0, -1]) == "z" and T.direction_label([2, 1, 0]) == "2x+y"
    assert T.direction_label([0.70, 0.71, 0.0]) == "(0.702, 0.712, 0.000)"                  # 0.6 degrees off: no snapping
    C = np.array(_cubic(300.0, 100.0, 80.0))
    C[2, 2], C[0, 2], C[2, 0], C[1, 2], C[2, 1], C[5, 5] = 200.0, 60.0, 60.0, 60.0, 60.0, 120.0
    d = A.analyze(_elastcon(C.tolist(), "TETRAGONAL (I)", "TETRAGONAL"))
    labels = []
    for q in d["derived"]["directional"].values():
        for k in ("min", "max"):
            for key in ("label", "label2"):
                if isinstance(q, dict) and key in (q.get(k) or {}):
                    labels.append(q[k][key])
    assert labels and not [x for x in labels if "[" in x]
    E = d["derived"]["directional"]["youngs"]
    assert E["min"]["label"] == "z" and E["min"]["value"] == pytest.approx(1 / np.linalg.inv(C)[2, 2], rel=1e-5)
    # diamond (cubic): lattice directions, as before
    dia = A.analyze_file(MECH / "dia_elastcon.out")["derived"]["directional"]["youngs"]
    assert (dia["min"]["label"], dia["max"]["label"]) == ("[1 0 0]", "[1 1 1]")
