"""Without the seekpath library, the static path uses the computed cell too.

extract_lattice_parameters_from_output returned the conventional cell of
the input echo: an optimisation's starting cell (the lattice variant, tI1
or tI2, came from it), and for a SUPERCEL output the input cell, while
fort.9 holds the supercell, so the static path was in the reciprocal basis
of the primitive cell. It now takes the CRYSTALLOGRAPHIC CELL printed with
the final optimised geometry, and a SUPERCEL output's static points are
mapped to the supercell basis, f_super = E f_prim, as the library route
gives them. A cell echoed in rhombohedral axes is read in hexagonal axes.
"""
import contextlib
import io
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
d3_kpoints = pytest.importorskip("d3_kpoints")

import crystal_out_text as cot  # noqa: E402
from test_kpath_computed_geometry import (  # noqa: E402
    DIA_PRIM, DIA_SUPER, _opt_output, _supercell_output)

CORPUS = Path(__file__).resolve().parents[1] / "test"


@pytest.fixture
def static(monkeypatch):
    monkeypatch.setattr(d3_kpoints, "SEEKPATH_LIBRARY_AVAILABLE", False)

    def run(sg, lattice, out, **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            segments, info = d3_kpoints.get_seekpath_full_kpath(sg, lattice, str(out), **kw)
            labels = d3_kpoints.get_seekpath_labels(sg, lattice, str(out))
        return segments, info, labels
    return run


def test_optimisation_takes_the_final_cell(tmp_path, static):
    out = tmp_path / "opt.out"
    out.write_text(_opt_output())
    params = d3_kpoints.extract_lattice_parameters_from_output(str(out))
    assert (params["a"], params["c"]) == (3.89, 6.38)
    assert params["W"] == [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    _, info, labels = static(139, "I", out)
    assert info["lookup_key"] == "tI2"
    assert labels == d3_kpoints.seekpath_data["tI2"]["labels"]


def test_unconverged_optimisation_keeps_the_input_cell(tmp_path, static):
    out = tmp_path / "opt.out"
    out.write_text(_opt_output(final=False))
    params = d3_kpoints.extract_lattice_parameters_from_output(str(out))
    assert (params["a"], params["c"]) == (4.0, 3.0)
    assert static(139, "I", out)[1]["lookup_key"] == "tI1"


def test_single_point_keeps_the_input_echo(tmp_path):
    """No final geometry: the conventional cell of the input echo, as before,
    plus the printed W."""
    out = tmp_path / "sp.out"
    out.write_text(cot.echo("F D 3 M", (3.54481, 3.54481, 3.54481, 90, 90, 90), DIA_PRIM)
                   + cot.block(DIA_PRIM, [(6, "C", .125, .125, .125), (6, "C", -.125, -.125, -.125)],
                               w=((-1, 1, 1), (1, -1, 1), (1, 1, -1)),
                               conventional=(3.54481, 3.54481, 3.54481, 90, 90, 90)))
    params = d3_kpoints.extract_lattice_parameters_from_output(str(out))
    assert params["a"] == 3.54481 and params["alpha"] == 90.0
    assert params["W"] == [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
    assert "expansion" not in params


# --------------------------------------------------------------------------
# SUPERCEL: f_super = E f_prim
# --------------------------------------------------------------------------

def test_supercel_static_points_are_in_the_supercell_basis(tmp_path, static):
    out = tmp_path / "super.out"
    out.write_text(_supercell_output())
    segments, info, labels = static(227, "F", out)
    prim = d3_kpoints.seekpath_data["cF2"]["segments"]
    assert info["expansion"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    np.testing.assert_allclose(segments, 2 * np.array(prim), atol=1e-12)
    assert labels == d3_kpoints.seekpath_data["cF2"]["labels"]
    # opt2d12 keeps the primitive basis
    primitive, info, _ = static(227, "F", out, cell="primitive")
    np.testing.assert_allclose(primitive, prim, atol=1e-12)
    assert "expansion" not in info


def test_non_diagonal_expansion_rows_as_printed():
    """E1, E2, E3 are rows: a_super_i = sum_j E_ij a_prim_j, so f_super = E f_prim.
    The transpose gives other points."""
    e = [[2, 1, 0], [0, 1, 0], [0, 0, 1]]
    seg = [[0.5, 0.0, 0.5, 0.5, 0.25, 0.75]]
    got = d3_kpoints.expand_segments(seg, e)
    assert got == [[1.0, 0.0, 0.5, 1.25, 0.25, 0.75]]
    assert got != d3_kpoints.expand_segments(seg, np.array(e).T.tolist())
    # a point's Cartesian vector is unchanged: f_super . B_super = f_prim . B_prim
    a_prim = np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]) * 3.5
    a_super = np.array(e) @ a_prim
    b_prim, b_super = (2 * np.pi * np.linalg.inv(m).T for m in (a_prim, a_super))
    np.testing.assert_allclose(np.array(got[0][3:]) @ b_super, np.array(seg[0][3:]) @ b_prim,
                               atol=1e-12)


def _supercell_text(e, a=3.54481):
    """A diamond SUPERCEL output with expansion E, atoms in the supercell."""
    e = np.array(e, dtype=float)
    a_prim = np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]) * a
    a_super = e @ a_prim
    base = [np.array([.125, .125, .125]), np.array([-.125, -.125, -.125])]
    inv_e = np.linalg.inv(e)
    atoms = []
    rng = range(-3, 4)
    for t in ((i, j, k) for i in rng for j in rng for k in rng):
        for x in base:
            f = (x + np.array(t)) @ inv_e        # prim fractions -> super fractions
            if np.all(f > -0.5 - 1e-9) and np.all(f <= 0.5 + 1e-9):
                if not any(np.allclose(f, g) for _, _, *g in atoms):
                    atoms.append((6, "C", *f))
    assert len(atoms) == 2 * round(abs(np.linalg.det(e)))
    lengths = np.linalg.norm(a_super, axis=1)

    def angle(i, j):
        return np.degrees(np.arccos(a_super[i] @ a_super[j] / lengths[i] / lengths[j]))
    params = (*lengths, angle(1, 2), angle(0, 2), angle(0, 1))
    prim = (a / np.sqrt(2),) * 3 + (60.0,) * 3
    return (cot.echo("F D 3 M", (a, a, a, 90, 90, 90), prim) + cot.supercel(e)
            + cot.block(params, atoms, centring="1/1"))


def _segment_equivalent(ends, targets, rotations, tol, expansion=None):
    """One operation (with time reversal) and one reciprocal lattice vector
    map both targets onto the ends.

    With ``expansion`` the points are supercell fractions compared in the
    primitive cell's basis, under the primitive cell's rotations; the
    lattice vector is then the supercell's (E d integer), as BAND of the
    supercell folds every such vector onto the same bands.
    """
    (fa, fb), (ta, tb) = ends, targets
    e = np.eye(3) if expansion is None else np.array(expansion, dtype=float)
    inv = np.linalg.inv(e)
    fa, fb, ta, tb = (inv @ np.array(v, dtype=float) for v in (fa, fb, ta, tb))
    for w in rotations:
        for s in (1, -1):
            da = e @ (s * (ta @ w) - fa)
            db = e @ (s * (tb @ w) - fb)
            g = np.round(da)
            if np.all(np.abs(da - g) < tol) and np.all(np.abs(db - g) < tol):
                return True
    return False


DIAMOND = (np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]) * 3.54481,
           [[.125, .125, .125], [-.125, -.125, -.125]], [6, 6])


@pytest.mark.parametrize("e", [[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                               [[2, 1, 0], [0, 1, 0], [0, 0, 1]],
                               [[1, 1, 0], [0, 1, 1], [1, 0, 2]]])
def test_static_supercell_path_equals_the_library_route(tmp_path, monkeypatch, e):
    pytest.importorskip("seekpath")
    spglib = pytest.importorskip("spglib")
    si = pytest.importorskip("seekpath_interface")
    out = tmp_path / "super.out"
    out.write_text(_supercell_text(e))
    with contextlib.redirect_stdout(io.StringIO()):
        library, info = d3_kpoints.get_seekpath_full_kpath(227, "F", str(out))
        st = si.parse_crystal_structure(str(out))
    assert info["source"] == "seekpath_library" and st["supercell"] == "supercel"
    monkeypatch.setattr(d3_kpoints, "SEEKPATH_LIBRARY_AVAILABLE", False)
    with contextlib.redirect_stdout(io.StringIO()):
        static, _ = d3_kpoints.get_seekpath_full_kpath(227, "F", str(out))
    rotations = spglib.get_symmetry(DIAMOND, symprec=1e-5)["rotations"]
    assert len(library) == len(static)
    for lib, sta in zip(library, static):
        assert _segment_equivalent((lib[:3], lib[3:]), (sta[:3], sta[3:]), rotations, 1e-5, e), \
            (lib, sta)
    # the transposed expansion would not be
    if not np.allclose(e, np.array(e).T):
        wrong = d3_kpoints.expand_segments(d3_kpoints.seekpath_data["cF2"]["segments"],
                                           np.array(e).T.tolist())
        assert not all(_segment_equivalent((lib[:3], lib[3:]), (w[:3], w[3:]), rotations, 1e-5, e)
                       for lib, w in zip(library, wrong))


def test_primitive_basis_library_route_is_exact(tmp_path):
    """cell="primitive" maps the library's points back with E^-1, exactly."""
    pytest.importorskip("seekpath")
    out = tmp_path / "super.out"
    out.write_text(_supercell_text([[2, 1, 0], [0, 1, 0], [0, 0, 1]]))
    with contextlib.redirect_stdout(io.StringIO()):
        sup, info = d3_kpoints.get_seekpath_full_kpath(227, "F", str(out))
        prim, pinfo = d3_kpoints.get_seekpath_full_kpath(227, "F", str(out), cell="primitive")
    e = np.array(info["expansion"])
    for s, p in zip(sup, prim):
        np.testing.assert_allclose(e @ np.array(p[:3]), s[:3], atol=1e-12)
        np.testing.assert_allclose(e @ np.array(p[3:]), s[3:], atol=1e-12)
    iss = pinfo["shrink_factor"]
    assert all(abs(v * iss - round(v * iss)) < 1e-9 for seg in prim for v in seg)


@pytest.mark.skipif(not DIA_SUPER.is_file(), reason="test/ corpus not present")
def test_corpus_supercel222_static_equals_library(monkeypatch):
    pytest.importorskip("seekpath")
    spglib = pytest.importorskip("spglib")
    si = pytest.importorskip("seekpath_interface")
    with contextlib.redirect_stdout(io.StringIO()):
        library, _ = d3_kpoints.get_seekpath_full_kpath(227, "F", str(DIA_SUPER))
        st = si.parse_crystal_structure(str(DIA_SUPER))
    monkeypatch.setattr(d3_kpoints, "SEEKPATH_LIBRARY_AVAILABLE", False)
    with contextlib.redirect_stdout(io.StringIO()):
        static, _ = d3_kpoints.get_seekpath_full_kpath(227, "F", str(DIA_SUPER))
    rotations = spglib.get_symmetry(DIAMOND, symprec=1e-5)["rotations"]
    for lib, sta in zip(library, static):
        assert _segment_equivalent((lib[:3], lib[3:]), (sta[:3], sta[3:]), rotations, 1e-5,
                                   st["expansion"])


# --------------------------------------------------------------------------
# R in rhombohedral axes
# --------------------------------------------------------------------------

def test_rhombohedral_axes_are_read_in_hexagonal_axes(tmp_path, static):
    """Bi given with IFHR = 1 (a = 4.7458, alpha = 57.23): CRYSTAL23 echoes
    that cell as the CONVENTIONAL CELL and prints the hexagonal
    CRYSTALLOGRAPHIC CELL 4.54573318 / 11.86222949 (measured)."""
    out = tmp_path / "bi.out"
    out.write_text(cot.echo("R -3 M", (4.7458, 4.7458, 4.7458, 57.23, 57.23, 57.23)))
    params = d3_kpoints.static_cell_parameters(str(out), "R")
    assert abs(params["a"] - 4.54573318) < 1e-5 and abs(params["c"] - 11.86222949) < 1e-5
    assert (params["alpha"], params["beta"], params["gamma"]) == (90.0, 90.0, 120.0)
    # the same cell in hexagonal axes gives the same key
    hexa = tmp_path / "bi_hex.out"
    hexa.write_text(cot.echo("R -3 M", (4.54573, 4.54573, 11.86223, 90, 90, 120)))
    assert static(166, "R", out)[1]["lookup_key"] == static(166, "R", hexa)[1]["lookup_key"] == "hR1"
    # other lattices are not converted
    assert d3_kpoints.static_cell_parameters(str(out), "P")["a"] == 4.7458


@pytest.mark.parametrize("alpha", [30.0, 57.23, 80.0, 95.0, 110.0])
def test_hexagonal_axes_conversion(alpha):
    a = 5.0
    h = d3_kpoints.hexagonal_axes({"a": a, "b": a, "c": a, "alpha": alpha, "beta": alpha,
                                   "gamma": alpha})
    # rhombohedral vectors from the hexagonal cell (obverse): (2a+b+c)/3, ...
    hexa = np.array([[h["a"], 0, 0], [-h["a"] / 2, h["a"] * np.sqrt(3) / 2, 0], [0, 0, h["c"]]])
    rho = np.array([[2, 1, 1], [-1, 1, 1], [-1, -2, 1]]) / 3 @ hexa
    np.testing.assert_allclose(np.linalg.norm(rho, axis=1), a, atol=1e-10)
    cos = rho[0] @ rho[1] / a / a
    assert abs(np.degrees(np.arccos(cos)) - alpha) < 1e-8


def test_opt2d12_dispersion_path_stays_primitive(monkeypatch):
    """opt2d12 writes a SUPERCEL parent's child in the input space group's
    cell, so its phonon-dispersion path stays in the primitive basis."""
    d12_calc_freq = pytest.importorskip("d12_calc_freq")
    monkeypatch.setattr(d3_kpoints, "SEEKPATH_LIBRARY_AVAILABLE", False)
    with contextlib.redirect_stdout(io.StringIO()):
        path = d12_calc_freq.get_auto_phonon_path(
            space_group=227, shrink=8, format_type="seekpath", lattice_type="F",
            band_settings={}, optimization_section=_supercell_output())
    prim = d3_kpoints.seekpath_data["cF2"]["segments"]
    assert path == [[int(round(v * 8)) for v in seg] for seg in prim]
