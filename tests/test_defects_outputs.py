"""mace.defects.outputs: the lines a defect analysis reads, and the ones it must not.

Excerpts are synthetic (built line by line in the CRYSTAL23 format, with
numbers from the HPCC defect runs).
"""
import re
from pathlib import Path

import pytest

from _defects_fixtures import fake_output

from mace.defects import outputs as out


def test_electron_count_is_the_mulliken_line_not_the_pre_doping_header():
    # MgO V_O q=+1: the header still says 632, DOPING -1 left 631 (measured on CRYSTAL23)
    text = fake_output(energy=-8735.8147109285, header=632, electrons=631.0, spin=1.0)
    assert out.electron_count(text) == 631.0
    assert out.header_electrons(text) == 632
    s = out.summarise("v1", text)
    assert s.electrons == 631.0 and s.header_electrons == 632 and s.spin == 1.0


def test_scf_energy_and_unconverged_runs():
    assert out.scf_energy(fake_output(energy=-8810.8448478966))[0] == pytest.approx(-8810.8448478966)
    assert out.scf_cycles(fake_output(energy=-1.0, cycles=15)) == 15
    e, why = out.scf_energy(fake_output(energy=-1.0, converged=False))
    assert e is None and "TOO MANY CYCLES" in why
    e, why = out.scf_energy("SCF still running\n")
    assert e is None and "no '== SCF ENDED'" in why


def test_the_last_vbm_block_is_used_with_its_band_and_k():
    early = fake_output(vbm=-0.33675830, cbm=-0.09853323)
    late = fake_output(vbm=-0.21811894, cbm=0.048263216, vbm_k=6)
    e = out.band_edges(early + late)
    assert e.vbm == pytest.approx(-0.21811894) and e.cbm == pytest.approx(0.048263216)
    assert e.vbm_band == 320 and e.vbm_k == 6
    assert e.gap * 27.211386245988 == pytest.approx(7.2486, abs=1e-3)


def test_spin_outputs_take_the_highest_vbm_and_lowest_cbm():
    block = "\n".join([
        " TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT FDIK        TELAPSE       43.59 TCPU       42.05",
        "    ALPHA      ELECTRONS",
        " INSULATING STATE",
        " TOP OF VALENCE BANDS -    BAND     78; K    1; EIG -2.8933201E-01 AU",
        " BOTTOM OF VIRTUAL BANDS - BAND     79; K    1; EIG -1.1692619E-02 AU",
        "",
        "    BETA       ELECTRONS",
        " INSULATING STATE",
        " TOP OF VALENCE BANDS -    BAND     77; K    2; EIG -2.5000000E-01 AU",
        " BOTTOM OF VIRTUAL BANDS - BAND     78; K    1; EIG -2.0000000E-02 AU",
        " TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT PDIG        TELAPSE       46.14 TCPU       44.59",
    ])
    e = out.band_edges(block)
    assert e.vbm == pytest.approx(-0.25) and (e.vbm_band, e.vbm_k) == (77, 2)
    assert e.cbm == pytest.approx(-0.020)


def test_efermi_and_spheropole_are_the_last_values():
    text = fake_output(efermi=-0.10420524, sph=1.0) + fake_output(efermi=-0.11251191, sph=140.63210845155)
    assert out.efermi(text) == pytest.approx(-0.11251191)
    assert out.spheropole(text) == pytest.approx(140.63210845155)


def test_mulliken_rows_include_ghosts_and_the_spin_block():
    text = fake_output(electrons=632.0, mulliken_atoms=[(1, "XX", 0, 0.234), (2, "O", 8, 9.327),
                                                         (33, "MG", 12, 10.656)],
                       spin=1.0, spin_atoms=[(1, "XX", 0, 0.50), (2, "O", 8, -0.002)])
    m = out.mulliken(text)
    assert [a[:3] for a in m.atoms] == [(1, "XX", 0), (2, "O", 8), (33, "MG", 12)]
    assert m.charge(1) == pytest.approx(-0.234) and m.charge(33) == pytest.approx(1.344)
    s = out.mulliken_spin(text)
    assert s.total == 1.0 and s.population(1) == pytest.approx(0.5)


POTC_ATOMS = """    64  POINT  COORDINATES      V0           V1(Z)        V1(X)        V1(Y)
  1  0.0000  0.0000  0.0000 -2.29034E+01  3.66901E-15  8.22527E-15 -4.62898E-15
  2  3.9798  3.9798  0.0000 -2.29034E+01 -5.53413E-15 -5.78924E-11 -5.78934E-11
 33  3.9798  0.0000  0.0000 -4.80000E+01  0.00000E+00  0.00000E+00  0.00000E+00
"""

POTC_PLANAR = """#
@ XAXIS LABEL "Z (AU)"
@ YAXIS LABEL "ELECTROSTATIC PROPERTIES (AU)"
@ LEGEND STRING 0 "V"
 -2.0000000E+01 -2.8421709E-14  0.0000000E+00  3.6488471E-51 -2.9036603E-52
 -1.9000000E+01  9.9475983E-14  0.0000000E+00  3.2340772E-46 -2.5735969E-47
"""


def test_potc_files():
    atoms = out.read_potc_atoms(POTC_ATOMS)
    assert atoms[0] == (1, (0.0, 0.0, 0.0), -22.9034)
    assert atoms[2][0] == 33 and len(atoms) == 3
    planar = out.read_potc_planar(POTC_PLANAR)
    assert planar == [(-20.0, pytest.approx(-2.8421709e-14)), (-19.0, pytest.approx(9.9475983e-14))]


def test_dispersion_and_gcp_terms_are_added_to_the_scf_energy():
    # The '== SCF ENDED' E(AU) leaves out D3 and gCP (corpus: SCF + D3 + gCP equals CRYSTAL's
    # 'TOTAL ENERGY + DISP + GCP' line in all 340 dispersion outputs checked). Synthetic numbers.
    text = fake_output(energy=-1000.25, d3=-0.0125, gcp=0.0875, electrons=100.0)
    s = out.summarise("c3", text)
    assert s.scf_energy == pytest.approx(-1000.25)
    assert s.d3 == pytest.approx(-0.0125) and s.gcp == pytest.approx(0.0875)
    assert s.energy == pytest.approx(-1000.25 - 0.0125 + 0.0875)
    assert s.energy_terms == "SCF + D3 + gCP"
    d3_only = out.summarise("d3", fake_output(energy=-10.0, d3=-0.5))
    assert d3_only.energy == pytest.approx(-10.5) and d3_only.gcp is None
    assert d3_only.energy_terms == "SCF + D3"
    plain = out.summarise("p", fake_output(energy=-10.0))
    assert plain.energy == pytest.approx(-10.0) and plain.energy_terms == "SCF"


def test_dispersion_terms_must_belong_to_the_last_scf():
    # terms printed after an earlier SCF only: the final total is unknown, not the bare SCF value
    text = fake_output(energy=-10.0, d3=-0.5) + fake_output(energy=-10.2)
    s = out.summarise("x", text)
    assert s.energy is None and "D3/gCP" in s.reason
    assert out.dispersion_terms(fake_output(energy=-10.0) + fake_output(energy=-10.2, d3=-0.5)) \
        == (pytest.approx(-0.5), None)


CORPUS = Path(__file__).resolve().parents[1] / "test"
_PRINTED_TOTAL = re.compile(r"TOTAL ENERGY \+ DISP(?: \+ GCP)? \(AU\)\s+(\S+)")


@pytest.mark.skipif(not CORPUS.is_dir(), reason="test/ corpus not present")
def test_corpus_totals_match_crystals_disp_gcp_line():
    """SCF + D3 + gCP equals the printed 'TOTAL ENERGY + DISP (+ GCP)' after the last SCF."""
    checked = 0
    for path in sorted(CORPUS.glob("*/*.out")):
        text = path.read_text(errors="replace")
        if "D3 DISPERSION ENERGY" not in text and "GCP ENERGY" not in text:
            continue
        e, reason, d3, gcp = out.total_energy(text)
        if e is None:
            continue
        last = list(re.finditer(r"== SCF ENDED", text))[-1]
        printed = _PRINTED_TOTAL.search(text, last.end())
        if printed is None:
            continue
        assert e == pytest.approx(float(printed.group(1)), abs=1e-8), path.name
        assert e != pytest.approx(out.scf_energy(text)[0], abs=1e-6), path.name
        checked += 1
    assert checked > 0
