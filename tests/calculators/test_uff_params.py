"""The UFF and UFF4MOF parameter tables (mace/calculators/uff_params.py)
against the published values: the original 127 UFF types (Rappé et al.
1992, as tabulated identically by RDKit and lammps-interface) and the
UFF4MOF additions (Addicoat 2014, Coupry 2016)."""
import pytest

from mace.calculators import uff_params as P

# rows as RDKit tabulates Table 1 of Rappé et al. (Code/ForceField/UFF/Params.cpp):
# r1, theta0, x1, D1, zeta, Z1, Vi, Uj, Xi, Hard, Radius
PAPER = {
    "H_": (0.354, 180, 2.886, 0.044, 12, 0.712, 0, 0, 4.528, 6.9452, 0.371),
    "C_3": (0.757, 109.47, 3.851, 0.105, 12.73, 1.912, 2.119, 2, 5.343, 5.063, 0.759),
    "C_R": (0.729, 120, 3.851, 0.105, 12.73, 1.912, 0, 2, 5.343, 5.063, 0.759),
    "N_R": (0.699, 120, 3.66, 0.069, 13.407, 2.544, 0, 2, 6.899, 5.88, 0.715),
    "O_3": (0.658, 104.51, 3.5, 0.06, 14.085, 2.3, 0.018, 2, 8.741, 6.682, 0.669),
    "O_2": (0.634, 120, 3.5, 0.06, 14.085, 2.3, 0, 2, 8.741, 6.682, 0.669),
    "Si3": (1.117, 109.47, 4.295, 0.402, 12.175, 2.323, 1.225, 1.25, 4.168, 3.487, 1.176),
    "S_3+2": (1.064, 92.1, 4.035, 0.274, 13.969, 2.703, 0.484, 1.25, 6.928, 4.486, 1.047),
    "Zn3+2": (1.193, 109.47, 2.763, 0.124, 12, 1.308, 0, 0.7, 5.106, 4.285, 1.4),
    "Cu3+1": (1.302, 109.47, 3.495, 0.005, 12, 1.756, 0, 0.7, 4.2, 4.22, 1.434),
}


def test_counts():
    assert len(P.UFF_DATA) == 127
    assert P.UFF_TYPES <= set(P.UFF4MOF_DATA)
    assert all(len(v) == len(P.FIELDS) for v in P.UFF4MOF_DATA.values())


@pytest.mark.parametrize("t", sorted(PAPER))
def test_paper_rows(t):
    assert P.UFF_DATA[t] == pytest.approx(PAPER[t])


def test_uff4mof_types():
    for t in ("Zn3f2", "Zn4+2", "Cu4+2", "Zr8f4", "O_3_f", "O_2_z", "Al6+3", "Mg6f3"):
        assert t in P.UFF4MOF_DATA and t not in P.UFF_DATA
    assert P.UFF4MOF_DATA["Cu4+2"][:2] == (1.28, 90.0)
    # the one UFF row UFF4MOF revises, used only in UFF4MOF mode
    assert P.UFF_DATA["Pt4+2"][0] == 1.364 and P.UFF4MOF_DATA["Pt4+2"][0] == 1.125
    for t in P.UFF_TYPES - {"Pt4+2"}:
        assert P.UFF4MOF_DATA[t] == P.UFF_DATA[t]
