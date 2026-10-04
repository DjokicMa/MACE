"""mace.defects.structure: supercell labels come from CRYSTAL's own tables.

The fixture is a 64-row excerpt of the pristine MgO 2x2x2 (SUPERCON) output
run on HPCC (geometry tables only).
"""
import numpy as np
import pytest

from _defects_fixtures import DATA, mgo_host

from mace.defects import DefectError
from mace.defects import deck_edit as de
from mace.defects import structure as st

MGO_222 = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
TEXT = (DATA / "mgo_pris_geometry.out").read_text()


def _host(text=None):
    h, _ = de.classify_host(text or mgo_host())
    return h


def test_tables_are_read():
    g = st.parse_geometry(TEXT)
    assert len(g.atoms) == 64 and g.dimensionality == 3 and g.n_electrons_header == 640
    assert g.lattice.tolist() == [[8.424, 0, 0], [0, 8.424, 0], [0, 0, 8.424]]
    assert g.atom(1).element == "O" and g.atom(64).element == "Mg"
    assert g.atom(1).asym is True and g.atom(1).frac.tolist() == [0, 0, 0]


def test_fractional_site_of_the_host_cell_maps_to_the_supercell_label():
    g = st.parse_geometry(TEXT)
    # (1,1,1) of the host's conventional cell is the O at the supercell's body center
    label, src, notes = st.resolve_site(_host(), g, "O", frac=(1, 1, 1), keyword="SUPERCON", matrix=MGO_222)
    assert label == 22 and src == "fractional"
    assert g.atom(22).cart.tolist() == pytest.approx([4.212, 4.212, 4.212])
    label, _, _ = st.resolve_site(_host(), g, "Mg", frac=(0.5, 0.5, 0.5), keyword="SUPERCON", matrix=MGO_222)
    assert g.atom(label).element == "Mg"
    assert g.atom(label).cart.tolist() == pytest.approx([2.106, 2.106, 2.106])
    with pytest.raises(DefectError, match="no O within 0.3"):
        st.resolve_site(_host(), g, "O", frac=(0.5, 0.5, 0.5), keyword="SUPERCON", matrix=MGO_222)


def test_fractional_site_needs_supercon_on_a_centered_host():
    g = st.parse_geometry(TEXT)
    with pytest.raises(DefectError, match="centered host"):
        st.resolve_site(_host(), g, "O", frac=(1, 1, 1), keyword="SUPERCEL", matrix=MGO_222)


def test_element_only_site():
    g = st.parse_geometry(TEXT)
    label, src, notes = st.resolve_site(_host(), g, "O")
    assert label == 1 and src == "element" and "equivalent" in notes[0]
    # without an output there is no label to give yet
    label, src, _ = st.resolve_site(_host(), None, "O")
    assert label is None and "pristine output" in src
    two_o = mgo_host().replace("2\n8  0.0 0.0 0.0\n", "3\n8  0.0 0.0 0.0\n8  0.25 0.25 0.25\n")
    with pytest.raises(DefectError, match="2 inequivalent sites") as exc:
        st.resolve_site(_host(two_o), g, "O")
    assert "host atom 1: O" in str(exc.value) and "host atom 2: O" in str(exc.value)


def test_site_label_is_checked_when_there_is_an_output():
    g = st.parse_geometry(TEXT)
    assert st.resolve_site(_host(), g, "O", label=22)[0] == 22
    with pytest.raises(DefectError, match="is Mg, not O"):
        st.resolve_site(_host(), g, "O", label=64)
    label, src, notes = st.resolve_site(_host(), None, "O", label=7)
    assert label == 7 and "unchecked" in src and "not checked" in notes[0]


def test_pbc_matching_after_atomremo_renumbering():
    g = st.parse_geometry(TEXT)
    removed = st.Geometry(g.lattice, [st.Atom(a.label - 1, a.z, a.symbol, a.cart + 8.424 * (a.label % 2))
                                      for a in g.atoms[1:]], 3)
    m = st.match_atoms(removed, g)
    assert len(m) == 63 and all(p == d + 1 for d, p in m.items())
    near = st.neighbors(g, g.atom(1).cart, 2.2, exclude=(1,))
    assert len(near) == 6 and all(abs(d - 2.106) < 1e-6 for _, d in near)
    assert st.far_atom(g, 1, "O") == 22


def test_partial_output_is_accepted_once_the_tables_are_complete():
    head = TEXT.split(" N. OF ELECTRONS PER CELL")[0]
    g = st.parse_geometry(head)
    assert g is not None and len(g.atoms) == 64 and g.n_electrons_header is None
    cut = TEXT[:TEXT.index("   40    12 MG")]
    assert st.parse_geometry(cut) is None
    assert st.parse_geometry("nothing yet\n") is None


def test_min_image_in_2d_ignores_the_vacuum_direction():
    lat = np.array([[2.5, 0, 0], [-1.25, 2.165, 0], [0, 0, 500]])
    v = st.min_image(np.array([2.4, 0, 30.0]), lat, 2)
    assert v.tolist() == pytest.approx([-0.1, 0, 30.0])
