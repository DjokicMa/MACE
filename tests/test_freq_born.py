"""Born charges, eps_0 and the LO-TO splitting of FREQ runs with INTENS + INTCPHF (1.1.6,
mace/elastic/born.py; the FREQ view's Dielectric button opens them as a results document).

Real B3LYP-D3 outputs of the corpus (tests/data/intcphf, README there): zinc-blende AgBr (cubic),
Ag2Br2 (P6_3mc) and Ag2SeCl (P4mm, a CPHF that breaks the crystal symmetry). Checked against what
CRYSTAL prints and against identities:

* the Born charges and dynamic charges as printed; their sum over atoms vanishes (charge neutrality);
* eps_0 = eps_inf + CRYSTAL's sum of the vibrational contributions;
* the computed LO frequencies obey the generalized Lyddane-Sachs-Teller relation
  prod(omega_LO^2/omega_TO^2) = q.eps_0.q / q.eps_inf.q along every direction;
* AgBr's single polar mode: omega_LO = omega_TO sqrt(eps_0/eps_inf) (139.82 -> 165.3 cm-1);
* each LO with its own TO mode: Ag2Br2 along x, E1 136.3 -> 161.6 cm-1 and the A mode (polar along
  z only) unchanged at 147.0 (by sort order it read 147.0 -> 161.6);
* Ag2SeCl's eps_inf is biaxial (CRYSTAL's 2V = 62.86 deg) in a tetragonal crystal: flagged, and its
  class-dependent results not stored; its beta table overflows: diverged, not "centrosymmetric".
"""
import json
import math
import subprocess
import sys

import pytest

from conftest import REPO_ROOT
from mace.elastic import analysis as A
from mace.elastic import born as B
from mace.elastic import views as V

DATA = REPO_ROOT / "tests" / "data" / "intcphf"
MACE_CLI = REPO_ROOT / "mace_cli"


def test_agbr_born_charges_eps0_and_lo_to():
    d = A.analyze_file(DATA / "agbr_freq.out")
    b = d["born"]
    assert [(x["symbol"], x["isotropic"]) for x in b["born_charges"]] == [("Ag", 1.108165), ("Br", -1.108165)]
    assert b["born_charges"][0]["tensor"][0][0] == 1.1082 and abs(b["charge_sum"][0][0]) < 1e-6
    assert b["epsilon_inf"][0][0] == 3.0421 and b["epsilon_vib"][0][0] == 1.20758
    assert b["epsilon_0"][0][0] == pytest.approx(3.0421 + 1.20758)
    for r in b["lo_to"]:
        assert r["direction"] in ("x", "y", "z")
        assert r["to_cm"] == pytest.approx([139.819] * 3, abs=0.01)
        assert sum(r["polar"]) == 1                                             # one F2 component along each axis
        lo = sorted(r["lo_cm"])
        assert lo[-1] == pytest.approx(139.819 * math.sqrt((3.0421 + 1.20758) / 3.0421), rel=2e-4)     # 165.3
        assert lo[:2] == pytest.approx([139.819] * 2, abs=0.01)
    assert "warnings" not in d
    assert V.kind_of(d) == "FREQ"
    doc = V.document((DATA / "agbr_freq.out").read_text(), DATA / "agbr_freq.out")
    assert doc["kind"] == "FREQ" and doc["views"] == [] and {r["label"]: r["value"] for r in doc["settings"]}["Module"] == "FREQ"


def test_ag2secl_lst_relation_and_the_symmetry_mismatch():
    d = A.analyze_file(DATA / "ag2secl_freq.out")
    b = d["born"]
    assert [x["symbol"] for x in b["born_charges"]] == ["Ag", "Ag", "Se", "Cl"]
    assert max(abs(v) for row in b["charge_sum"] for v in row) < 1e-3
    assert len(b["lo_to"]) == 3
    for r in b["lo_to"]:
        assert r["lst_ratio_freq"] == pytest.approx(r["lst_ratio_eps"], rel=2e-4)
        assert all(lo >= to - 1e-6 for lo, to in zip(r["lo_cm"], r["to_cm"]))
    z = next(r for r in b["lo_to"] if r["direction"] == "z")
    assert round(max(z["lo_cm"]), 1) == 231.8
    # eps_inf as CRYSTAL prints it (biaxial, 2V 62.86 deg) ...
    pr = d["optics"]["principal"]
    assert pr["optical_class"] == "biaxial" and pr["sign"] == "negative"
    assert pr["optic_axial_angle_deg"] == pytest.approx(62.86192, abs=0.01)                # CRYSTAL's 2V
    assert pr["n"] == pytest.approx(sorted(d["dielectric"]["refractive_indices"]), abs=5e-6)          # CRYSTAL prints 7 digits
    # ... but P4mm is tetragonal, so eps_xx = eps_yy: the CPHF tensor breaks the crystal symmetry
    assert d["crystal_family"] == "TETRAGONAL" and d["space_group"] == "P 4 M M"
    assert pr["symmetry_mismatch"] and pr["expected_class"] == "uniaxial"
    assert b["epsilon_0_symmetry_mismatch"] == {"expected": "uniaxial", "found": "biaxial"}
    cphf = d["optics"]["cphf"]                     # CRYSTAL's own convergence record flags it too
    assert cphf["unstable"] and cphf["max_diis_test"] == pytest.approx(9.5073e7) and cphf["cycles"] == {"x": 3, "y": 2, "z": 4}
    w = d["warnings"]
    assert any(x.startswith("eps_inf is biaxial") and "P 4 M M is tetragonal, so it must be uniaxial" in x for x in w)
    assert any(x.startswith("eps_0 is biaxial") for x in w) and any("DIIS test reached 9.51e+07" in x for x in w)
    rep = A.report(d)
    assert "WARNING: eps_inf is biaxial" in rep and "SYMMETRY MISMATCH: a uniaxial crystal" in rep
    assert "LO-TO along z: 87.6 -> 87.7, 131.4 -> 150.0, 208.6 -> 231.8 cm-1" in rep
    # the class-dependent results and the LO-TO splitting (q.eps_inf.q) are not stored
    from mace.elastic import properties as P
    props = P.all_properties(d)
    assert props["born_charge_min"]["value"] == -1.02875
    assert not {"refractive_index_principal_1", "birefringence", "optic_axial_angle_deg", "lo_to_splitting_max"} & set(props)


def test_an_unstable_cphf_stores_no_class_dependent_optics():
    """Ag1Cl5 is monoclinic, so any eps_inf passes the symmetry check, but its CPHF was unstable
    (CRYSTAL's DIIS test far past a stable run's) and eps reaches the hundreds: its principal indices,
    birefringence, 2V and LO-TO splitting are shown with a warning and not stored."""
    out = REPO_ROOT / "test" / "FREQ" / ("Ag1Cl5_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_"
                                         "B3LYP-D3-D3_optimized_freq_B3LYP-D3-D3_optimized.out")
    if not out.exists():
        pytest.skip("needs the test/ corpus")
    d = A.analyze_file(out)
    assert d["optics"]["cphf"]["unstable"] and not d["optics"]["principal"].get("symmetry_mismatch")
    assert any("DIIS test reached" in w and "are not stored" in w for w in d["warnings"])
    from mace.elastic import properties as P
    props = P.all_properties(d)
    assert not {"refractive_index_principal_1", "birefringence", "optic_axial_angle_deg", "lo_to_splitting_max"} & set(props)


def test_ag2secl_overflowed_beta_is_diverged_not_centrosymmetric():
    """CRYSTAL prints -8.9595E+158 as "-8.9595+158": read before as two numbers, the row did not match,
    the table stopped after XXX and XXY (both 0) and the report said "centrosymmetric" for P4mm."""
    d = A.analyze_file(DATA / "ag2secl_freq.out")
    beta = d["beta"]
    assert [r["component"] for r in beta["components"]] == ["XXX", "XXY", "XXZ", "XYY", "XYZ", "XZZ", "YYY", "YYZ", "YZZ", "ZZZ"]
    assert beta["complete"] and beta["overflow"] and beta["diverged"]
    xxz = beta["components"][2]
    assert xxz["value"] == pytest.approx(-8.9595e158) and xxz["unsymmetrized"] == pytest.approx(-9.2144e158)
    assert d["optics"]["shg"] == {"zero": None, "diverged": True, "unit": "pm/V"}
    rep = A.report(d)
    assert "centrosymmetric" not in rep and "First hyperpolarizability beta (a.u.): DIVERGED, not analyzed" in rep
    assert "shg" not in [v["key"] for v in V.views(d, (DATA / "ag2secl_freq.out").read_text())]
    from mace.elastic import properties as P
    assert not [k for k in P.all_properties(d) if k.startswith("shg_")]
    # a partial table never concludes zero either
    part = A.parse((DATA / "agbr_freq.out").read_text())["beta"]
    part = {**part, "components": [r for r in part["components"] if r["component"] != "XYZ"], "complete": False}
    from mace.elastic import optics as O
    assert O.shg(part) == {"zero": None, "incomplete": True, "unit": "pm/V"}


def test_ag2br2_lo_to_pairs_each_mode_with_its_own_lo():
    d = A.analyze_file(DATA / "ag2br2_freq.out")
    b = d["born"]
    assert "warnings" not in d and d["optics"]["principal"]["optical_class"] == "uniaxial"
    assert not d["optics"]["cphf"]["unstable"]
    lo_to = {r["direction"]: r for r in b["lo_to"]}
    x = lo_to["x"]
    pairs = {(m, i): (round(t, 1), round(l, 1)) for m, i, t, l in zip(x["modes"], x["irreps"], x["to_cm"], x["lo_cm"])}
    assert [p for p in pairs.values() if p[1] - p[0] > 0.05] == [(136.3, 161.6)]            # E1: 25.3 cm-1
    assert pairs[(11, "A")] == (147.0, 147.0)                                              # non-polar along x
    assert [pairs[k] for k in pairs if k[1] == "E1"] == [(136.3, 161.6), (136.3, 136.3)]
    z = lo_to["z"]
    assert [(round(t, 1), round(l, 1)) for t, l, p in zip(z["to_cm"], z["lo_cm"], z["polar"]) if p] == [(147.0, 172.3)]
    for r in b["lo_to"]:
        assert r["lst_ratio_freq"] == pytest.approx(r["lst_ratio_eps"], rel=2e-4)
    from mace.elastic import properties as P
    assert P.all_properties(d)["lo_to_splitting_max"]["value"] == pytest.approx(25.34, abs=0.01)    # was 25.23, the x one 14.6
    assert "LO-TO along x: 136.3 -> 161.6 cm-1" in A.report(d)


def test_mace_elastic_analyze_json_has_the_born_charges():
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "analyze", "elastic", str(DATA / "agbr_freq.out"), "--json"],
                       capture_output=True, text=True, timeout=300, env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"})
    assert r.returncode == 0, r.stderr
    js = json.loads(r.stdout)
    assert js["born"] == json.loads(json.dumps(A.analyze_file(DATA / "agbr_freq.out")["born"]))
    # the flags too: the CLI and the GUI's document share analyze()
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "analyze", "elastic", str(DATA / "ag2secl_freq.out"), "--json"],
                       capture_output=True, text=True, timeout=300, env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"})
    assert r.returncode == 0, r.stderr
    doc = V.document((DATA / "ag2secl_freq.out").read_text(), DATA / "ag2secl_freq.out")
    assert json.loads(r.stdout)["warnings"] == doc["data"]["warnings"] and len(doc["data"]["warnings"]) == 4


def test_the_freq_parser_flags_the_born_charges(tmp_path):
    from mace.gui import freq as F
    text = (DATA / "agbr_freq.out").read_bytes()
    start = text.index(b"FREQUENCY") if b"FREQUENCY" in text else 0
    got = F.parse_job(None, path=str(DATA / "agbr_freq.out"), start=start, end=len(text), tmpdir=str(tmp_path))
    assert got["born"] is True
    got = F.parse_job(None, path=str(DATA / "agbr_freq.out"), start=start, end=text.index(b"ATOMIC BORN CHARGE"), tmpdir=str(tmp_path))
    assert got["born"] is False
