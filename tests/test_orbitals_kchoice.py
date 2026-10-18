"""The ORBITALS k-point choice (mace/orbitals/kchoice.py) on real SCF outputs.

The outputs in tests/data/orbitals/scf/ are the HPCC campaign's SCF runs
(CRYSTAL23, 2026-10-16): Si diamond (CBM at K 10 = (2 2 0)/6, a complex
point), H2O and the two-component HI (molecules), trans-polyyne (1D, both
edges at K 5 = (4 0 0)/8, the zone boundary), the MgO(001) slab (both edges
at Γ) and bcc Fe (a metal: no edge lines).
"""
from fractions import Fraction
from pathlib import Path

import pytest

from mace.orbitals import kchoice as K

DATA = Path(__file__).resolve().parent / "data" / "orbitals" / "scf"
SPIN_INSULATOR = Path(__file__).resolve().parent / "data" / "charged_runs" / "odd_hsesol3c_3d_crash.out"


def _out(name):
    return (DATA / f"{name}.out").read_text()


def _d12(name):
    p = DATA / f"{name}.d12.txt"      # .txt: not a committed sample deck
    return p.read_text() if p.is_file() else None


# ------------------------------------------------------------------ parsing
def test_k_list_of_si():
    is_scf, points = K.parse_kpoints(_out("si"))
    assert is_scf == 6 and len(points) == 16
    assert points[1].oblique == (0, 0, 0) and points[1].is_gamma and points[1].is_real
    assert points[10].oblique == (2, 2, 0) and points[10].kind == "C"
    assert points[10].frac == (Fraction(1, 3), Fraction(1, 3), Fraction(0))
    assert points[13].kind == "R" and points[13].in_mesh(2)       # (3 3 0)/6 = (1/2 1/2 0)
    assert not points[10].in_mesh(2)


def test_band_edges_agree_with_the_defect_reader():
    from mace.defects.outputs import band_edges
    for name in ("si", "polyyne", "mgoslab"):
        text = _out(name)
        edges = {e.edge: e for e in K.parse_band_edges(text)}
        be = band_edges(text)
        assert (edges["VBM"].band, edges["VBM"].k_index) == (be.vbm_band, be.vbm_k)
        assert (edges["CBM"].band, edges["CBM"].k_index) == (be.cbm_band, be.cbm_k)


def test_spin_channels_give_an_edge_pair_each():
    edges = K.parse_band_edges(SPIN_INSULATOR.read_text())
    assert [(e.channel, e.edge, e.band) for e in edges] == [
        ("alpha", "VBM", 71), ("alpha", "CBM", 72), ("beta", "VBM", 70), ("beta", "CBM", 71)]


def test_a_metal_has_no_edges():
    assert K.parse_band_edges(_out("fe")) == []


def test_edges_of_an_earlier_cycle_are_not_taken_for_a_metal():
    # insulating at first, conducting in the last cycles: no edges
    text = ("TTTTTTTTTTTTTTT FDIK\n TOP OF VALENCE BANDS -    BAND      5; K    3; EIG -2.0E-01 AU\n"
            " BOTTOM OF VIRTUAL BANDS - BAND      6; K    3; EIG  1.0E-01 AU\nTTTTTTTTTTTTTTT PDIG\n"
            " POSSIBLY CONDUCTING STATE - EF(AU)  -1.0E-01\n")
    assert K.parse_band_edges(text) == []


def test_parent_isp():
    assert K.parent_isp(_out("si"), _d12("si")) == 12
    assert K.parent_isp(_out("si")) == 12                  # the Gilat net line of the .out
    assert K.parent_isp(_out("polyyne"), _d12("polyyne")) == 16
    assert K.parent_isp(_out("h2o")) is None
    assert K.parent_isp("", "T\nCRYSTAL\nSHRINK\n0 24\n4 4 2\nEND\n") == 24


def test_dimensionality():
    assert [K.dimensionality_of(_out(n)) for n in ("h2o", "polyyne", "mgoslab", "si", "fe")] == [0, 1, 2, 3, 3]
    assert K.dimensionality_of("title\nSLAB\n12\n") == 2


# ------------------------------------------------------------------ the default
def test_si_band_edges_take_is_2_with_a_note():
    c = K.choose_kmesh(_out("si"), d12_text=_d12("si"))
    assert (c.is_, c.isp, c.mode, c.dimensionality) == (2, 12, "band_edges", 3)
    assert c.newk_lines() == ["2 12"]
    cbm = next(e for e in c.edges if e.edge == "CBM")
    assert (cbm.band, cbm.k_index, cbm.oblique, cbm.kind) == (15, 10, (2, 2, 0), "C")
    assert cbm.frac == (Fraction(1, 3), Fraction(1, 3), Fraction(0))
    assert len(c.notes) == 1 and "K 10" in c.notes[0] and "nearest real points" in c.notes[0]


def test_a_molecule_has_no_mesh_line():
    c = K.choose_kmesh(_out("h2o"))
    assert (c.is_, c.isp, c.mode, c.dimensionality) == (None, None, "molecule", 0)
    assert c.newk_lines() == []


def test_edges_at_the_zone_boundary_take_is_2_without_a_note():
    c = K.choose_kmesh(_out("polyyne"), d12_text=_d12("polyyne"))
    assert (c.is_, c.isp, c.dimensionality) == (2, 16, 1)
    assert c.notes == []
    assert {e.oblique for e in c.edges} == {(4, 0, 0)}


def test_edges_at_gamma_take_is_1():
    c = K.choose_kmesh(_out("mgoslab"))
    assert (c.is_, c.isp, c.dimensionality) == (1, 12, 2)
    assert c.notes == []


def test_a_metal_takes_gamma_with_a_note():
    c = K.choose_kmesh(_out("fe"))
    assert (c.is_, c.isp) == (1, 16)
    assert "No band edges" in c.notes[0]


# ------------------------------------------------------------------ the other modes
def test_gamma_and_all_real_modes():
    assert K.choose_kmesh(_out("si"), "gamma").newk_lines() == ["1 12"]
    c = K.choose_kmesh(_out("si"), "all_real")
    assert (c.is_, c.mode, c.has_complex_points) == (2, "all_real", False)


def test_custom_is_with_complex_points_is_refused_in_2d_and_3d():
    for name in ("si", "mgoslab"):
        with pytest.raises(K.KChoiceError, match="complex k points are not usable"):
            K.choose_kmesh(_out(name), "custom", custom_is=4)
    # no override: CRYSTAL23's complex-k files in 2D/3D are unusable, so the deck is never written
    with pytest.raises(TypeError):
        K.choose_kmesh(_out("si"), "custom", custom_is=4, allow_complex=True)
    with pytest.raises(K.KChoiceError, match="IS 1"):
        K.choose_kmesh(_out("si"), "custom", custom_is=4)


def test_custom_is_in_1d_may_have_complex_points():
    c = K.choose_kmesh(_out("polyyne"), "custom", custom_is=4)
    assert (c.is_, c.has_complex_points) == (4, True)
    assert "validated" in c.notes[0]


def test_custom_is_1_and_2_are_fine_everywhere():
    assert K.choose_kmesh(_out("si"), "custom", custom_is=2).is_ == 2
    assert K.custom_is_problem(1, 3) is None


@pytest.mark.parametrize("bad", [0, -1, "x", None])
def test_bad_custom_is(bad):
    with pytest.raises(K.KChoiceError):
        K.choose_kmesh(_out("si"), "custom", custom_is=bad)


def test_unknown_mode():
    with pytest.raises(K.KChoiceError, match="unknown ORBITALS k mode"):
        K.choose_kmesh(_out("si"), "wannier")


# ------------------------------------------------------------------ files
def test_file_estimate_counts_and_sizes():
    est = K.estimate_orbital_files(None, 0, nbf=24)
    assert (est.n_files, est.exact, est.bytes_per_file) == (1, True, 24 * 24 * 22)
    # the SCF's own mesh: exact, from its k list (Si IS 6: 3 real, 13 complex)
    est = K.estimate_orbital_files(6, 3, nbf=36, scf_text=_out("si"))
    assert (est.n_real, est.n_complex, est.n_files, est.exact) == (3, 13, 29, True)
    # another mesh: an upper bound
    est = K.estimate_orbital_files(2, 3, nbf=36)
    assert (est.n_files, est.exact) == (8, False)
    assert K.estimate_orbital_files(1, 3).n_files == 1
    assert K.estimate_orbital_files(2, 2, nbf=10, spin=True).bytes_per_file == 2 * 100 * 22


def test_file_estimate_warns_for_many_or_big_files():
    assert K.estimate_orbital_files(2, 3, nbf=36).warnings == []
    many = K.estimate_orbital_files(8, 3, nbf=36)
    assert many.n_files == 512 and "1M-file quota" in many.warnings[0]
    big = K.estimate_orbital_files(2, 3, nbf=3000)
    assert any("GB of Molden files" in w for w in big.warnings)


def test_shared_messages():
    assert "two-component" in K.SOC_REFUSAL and "one-component" in K.SOC_REFUSAL
    assert "crash" in K.SOC_REFUSAL
    assert "Wannier90" in K.WANNIER_ROUTE


def test_loads_by_path_without_the_mace_package(tmp_path):
    """The deck writer loads kchoice by path when it runs as a standalone script."""
    import subprocess
    import sys
    path = Path(K.__file__)
    code = ("import importlib.util, sys; sys.modules['mace'] = None; "      # import mace fails
            f"spec = importlib.util.spec_from_file_location('kc', {str(path)!r}); m = importlib.util.module_from_spec(spec); "
            "sys.modules['kc'] = m; spec.loader.exec_module(m); "
            f"print(m.choose_kmesh(open({str(DATA / 'si.out')!r}).read()).newk_lines())")
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "['2 12']"
