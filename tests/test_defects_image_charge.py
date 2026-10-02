"""Image-charge term and dielectric constants for charged 3D defect cells."""
import math

import numpy as np
import pytest

from mace.defects import DefectError
from mace.defects import analysis as an
from mace.defects import image_charge as ic
from mace.defects import outputs as out

BOHR = 0.529177210903
FCC = np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
BCC = np.array([[-.5, .5, .5], [.5, -.5, .5], [.5, .5, -.5]])

# Excerpts of CRYSTAL23 MgO outputs (PBE0/POB-TZVP-REV2): CPHF, and FREQCALC INTENS INTCPHF.
CPHF = """
 DIELECTRIC TENSOR

 TENSOR IN PRINCIPAL AXES SYSTEM
 AA  2.865435E+00 BB  2.865435E+00 CC  2.865435E+00

 REFRACTIVE INDICES 
 AA  1.692760E+00 BB  1.692760E+00 CC  1.692760E+00
"""
VIB = """
 SUM TENSOR OF THE VIBRATIONAL CONTRIBUTIONS TO THE STATIC DIELECTRIC     TENSOR

           6.47740   -0.00000   -0.00000
          -0.00000    6.47740    0.00000
          -0.00000    0.00000    6.47740

 INTEGRATED IR INTENSITIES, IN UNITS OF KM/MOL, ARE COMPUTED UNDER THE
"""


def test_madelung_of_simple_cubic_and_its_scaling():
    assert ic.madelung(np.eye(3)) == pytest.approx(-2.837297, abs=1e-6)
    assert ic.madelung(2.5 * np.eye(3)) == pytest.approx(-2.837297 / 2.5, abs=1e-6)


def test_madelung_of_fcc_and_bcc_point_lattices():
    """Per V^(1/3): fcc 2.8883, bcc 2.8885 (point charges in a neutralising background)."""
    for lat, ref in ((FCC, 2.88828), (BCC, 2.88846)):
        v = ic.madelung(lat)
        assert v * abs(np.linalg.det(lat)) ** (1 / 3) == pytest.approx(-ref, abs=1e-5)
    # independent of the cell chosen for the same lattice
    skew = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]]) @ FCC
    assert ic.madelung(skew) == pytest.approx(ic.madelung(FCC), abs=1e-9)


def test_madelung_of_the_measured_mgo_supercells():
    a = 4.212 / BOHR
    assert ic.madelung(3 * a * FCC) == pytest.approx(-0.192007, abs=1e-6)
    assert ic.madelung(2 * a * np.eye(3)) == pytest.approx(-0.178233, abs=1e-6)
    assert ic.madelung(5 * a * FCC) == pytest.approx(-0.115204, abs=1e-6)


def test_madelung_refuses_a_flat_cell():
    with pytest.raises(DefectError):
        ic.madelung([[1, 0, 0], [0, 1, 0], [1, 1, 0]])


def test_lany_zunger_makov_payne():
    eps = 2.865
    f = 1 - (1 - 1 / eps) / 3
    assert ic.lany_zunger_factor(eps) == pytest.approx(f)
    assert ic.lany_zunger_factor(1.0) == 1.0
    e1 = ic.makov_payne(1, eps, -0.2)
    assert e1 == pytest.approx(f * 0.2 / (2 * eps)) and e1 > 0
    assert ic.makov_payne(-2, eps, -0.2) == pytest.approx(4 * e1)
    with pytest.raises(DefectError):
        ic.makov_payne(1, 0.0, -0.2)


def test_dielectric_from_cphf_and_freq_outputs():
    d = out.dielectric(CPHF)
    assert d.eps_inf == pytest.approx(2.865435) and d.eps_0 is None
    f = out.dielectric(CPHF + VIB)
    assert f.eps_0 == pytest.approx(2.865435 + 6.4774)
    assert out.dielectric("no tensor here").eps_inf is None


def test_dielectric_choice():
    c = an.dielectric_choice(text=CPHF, source="mgo_cphf.out")
    assert c["kind"] == "eps_inf" and c["eps"] == pytest.approx(2.865435) and "warning" not in c
    s = an.dielectric_choice(text=CPHF + VIB, source="mgo_freq.out", static=True)
    assert s["kind"] == "eps_0" and s["eps"] == pytest.approx(9.342835)
    assert an.dielectric_choice(value=2.865) == {"eps": 2.865, "kind": "given", "source": "--eps"}
    with pytest.raises(DefectError, match="INTCPHF"):
        an.dielectric_choice(text=CPHF, source="x.out", static=True)
    with pytest.raises(DefectError, match="DIELECTRIC TENSOR"):
        an.dielectric_choice(text="nothing", source="x.out")
    with pytest.raises(DefectError, match="not both"):
        an.dielectric_choice(value=2.0, text=CPHF)
    with pytest.raises(DefectError, match="positive"):
        an.dielectric_choice(value=-1.0)
    aniso = CPHF.replace("CC  2.865435E+00", "CC  3.500000E+00")
    a = an.dielectric_choice(text=aniso, source="x.out")
    assert a["eps"] == pytest.approx((2 * 2.865435 + 3.5) / 3) and "anisotropic" in a["warning"]
    assert math.isclose(out.dielectric(aniso).anisotropy, (3.5 - 2.865435) / a["eps"])


def test_makov_payne_keeps_the_sign_of_the_madelung_potential():
    """An elongated cell (aspect ratio about 4 or more) has v_M > 0: its
    spurious point-charge energy q^2 v_M / (2 eps) is positive, so the
    correction -q^2 v_M / (2 eps) must be negative, not |v_M|."""
    elongated = np.diag([1.0, 1.0, 5.0]) * 10.0
    v = ic.madelung(elongated)
    assert v > 0
    eps = 2.865
    f = ic.lany_zunger_factor(eps)
    assert ic.makov_payne(1, eps, v) == pytest.approx(-f * v / (2 * eps))
    assert ic.makov_payne(1, eps, v) < 0
