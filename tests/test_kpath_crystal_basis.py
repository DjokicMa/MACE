"""SeeK-path's band-path points are written in CRYSTAL's reciprocal basis.

BAND reads a segment end as I1/ISS b1 + I2/ISS b2 + I3/ISS b3, with b the
reciprocal vectors of CRYSTAL's primitive cell (CRYSTAL23 manual p.309).
SeeK-path gives its points in the reciprocal basis of its own standardised
primitive cell. Until this was fixed the numbers were copied across, which
put C- and A-centred orthorhombic points off the special points and named
the wrong points in triclinic, monoclinic and other cells whose axes
SeeK-path reorders.

Every test checks Cartesian vectors: a written point (CRYSTAL basis) must be
the point SeeK-path names (its basis, rotated back into the input frame), or
- where SeeK-path's own numbers are kept - a symmetry image of it, segment by
segment.
"""
import contextlib
import io
import itertools
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
spglib = pytest.importorskip("spglib")
seekpath = pytest.importorskip("seekpath")
si = pytest.importorskip("seekpath_interface")

# CRYSTAL's primitive vectors (rows) in terms of the conventional ones, as
# its outputs print them ("DIRECT LATTICE VECTORS CARTESIAN COMPONENTS"):
# C (a/2, b/2, 0), (-a/2, b/2, 0), c; A a, (0, b/2, c/2), (0, -b/2, c/2).
CRYSTAL_PRIMITIVE = {
    "P": np.eye(3),
    "C": np.array([[0.5, 0.5, 0], [-0.5, 0.5, 0], [0, 0, 1]]),
    "A": np.array([[1, 0, 0], [0, 0.5, 0.5], [0, -0.5, 0.5]]),
    "F": np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]),
    "I": np.array([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]]),
}


def _lattice(a, b, c, al, be, ga):
    return si.cell_params_to_vectors(a, b, c, al, be, ga)


def _crystal_structure(sg, centring, params):
    """Structure of group ``sg`` (standard setting) in CRYSTAL's primitive cell."""
    first_hall = next(h for h in range(1, 531) if spglib.get_spacegroup_type(h).number == sg)
    sym = spglib.get_symmetry_from_database(first_hall)
    conv = _lattice(*params)
    p = CRYSTAL_PRIMITIVE[centring]
    cell = p @ conv
    to_prim = np.linalg.inv(p)
    pos, types = [], []
    for kind, x in enumerate(([0.1234, 0.2345, 0.3456], [0.4321, 0.0765, 0.2011])):
        for rot, tr in zip(sym["rotations"], sym["translations"]):
            f = ((rot @ np.array(x) + tr) @ to_prim) % 1.0
            if not any(t == kind + 1 and np.allclose((f - q + 0.5) % 1 - 0.5, 0, atol=1e-6)
                       for q, t in zip(pos, types)):
                pos.append(f)
                types.append(kind + 1)
    return cell, np.array(pos), types


def _written_points(cell, pos, types):
    """(seekpath result, CRYSTAL-basis result, [(label, written fractional)] per segment)."""
    raw = si.get_seekpath_bandpath(cell, pos, types)
    with contextlib.redirect_stdout(io.StringIO()):
        moved = si.to_crystal_basis(raw, cell, pos, types)
        segments, _, info = si.convert_to_mace_format(moved)
    iss = info["shrink_factor"]
    written = [((a, np.array(seg[:3]) / iss), (b, np.array(seg[3:]) / iss))
               for (a, b), seg in zip(raw["path"], segments)]
    return raw, moved, written, info


def _cartesian_named(raw, label, cell):
    """Cartesian vector (input frame) of the point SeeK-path names ``label``."""
    rot = np.array(raw["rotation_matrix"])
    b_sp = 2 * np.pi * np.linalg.inv(np.array(raw["primitive_lattice"])).T
    return (np.array(raw["point_coords"][label]) @ b_sp) @ rot


def _segment_equivalent(ends, targets, rotations, tol):
    """One operation (with time reversal) and one G map both targets onto the ends."""
    (fa, fb), (ta, tb) = ends, targets
    for w in rotations:
        for s in (1, -1):
            da = s * (ta @ w) - fa
            db = s * (tb @ w) - fb
            g = np.round(da)
            if np.all(np.abs(da - g) < tol) and np.all(np.abs(db - g) < tol):
                return True
    return False


CASES = {
    # C-centred orthorhombic, Cmce, a > b (oC2) and a < b (oC1)
    "oC2_Cmce": (64, "C", (11.49, 10.08, 13.89, 90, 90, 90)),
    "oC1_Cmce": (64, "C", (5.1, 7.4, 3.0, 90, 90, 90)),
    # A-centred orthorhombic, Aea2 and Amm2, in each axis order
    **{f"oA_Aea2_{i}": (41, "A", abc + (90, 90, 90))
       for i, abc in enumerate(itertools.permutations((13.8, 6.39, 4.95)))},
    **{f"oA_Amm2_{i}": (38, "A", abc + (90, 90, 90))
       for i, abc in enumerate(itertools.permutations((3.0, 5.1, 7.4)))},
    # triclinic, two reductions
    "aP_1": (2, "P", (5.1, 3.0, 7.4, 100, 95, 80)),
    "aP_2": (2, "P", (3.0, 5.1, 7.4, 80, 95, 100)),
    "aP_P1": (1, "P", (7.4, 5.1, 3.0, 70, 110, 96)),
    # primitive orthorhombic in every axis order: SeeK-path takes a < b < c
    # for Pmmm, whose setting does not fix the axes; Pnma's does
    **{f"oP_Pmmm_{i}": (47, "P", abc + (90, 90, 90))
       for i, abc in enumerate(itertools.permutations((3.0, 5.1, 7.4)))},
    **{f"oP_Pnma_{i}": (62, "P", abc + (90, 90, 90))
       for i, abc in enumerate(itertools.permutations((3.0, 5.1, 7.4)))},
    # monoclinic with a > c and an obtuse beta
    "mP_P21c": (14, "P", (7.4, 5.1, 3.0, 90, 115, 90)),
    "mC_C2c": (15, "C", (7.4, 5.1, 9.0, 90, 105, 90)),
    "oF_Fddd": (70, "F", (7.4, 3.0, 5.1, 90, 90, 90)),
    "oI_Imma": (74, "I", (7.4, 3.0, 5.1, 90, 90, 90)),
    "tI_I41amd": (141, "I", (3.8, 3.8, 9.5, 90, 90, 90)),
    "cF_Fd-3m": (227, "F", (5.43, 5.43, 5.43, 90, 90, 90)),
    "hP_P63mmc": (194, "P", (3.2, 3.2, 5.2, 90, 90, 120)),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_written_points_are_seekpaths_points(name):
    sg, centring, params = CASES[name]
    cell, pos, types = _crystal_structure(sg, centring, params)
    raw, moved, written, info = _written_points(cell, pos, types)
    b_cr = 2 * np.pi * np.linalg.inv(cell).T
    # a lattice-dependent point may move to the nearest k/ISS, by at most this
    tol = info["max_point_shift"] + 1e-6
    rotations = spglib.get_symmetry((cell, pos, types), symprec=1e-5)["rotations"]
    rewritten = moved["point_coords"] is not raw["point_coords"]
    for (a, fa), (b, fb) in written:
        if rewritten:
            # exactly the point SeeK-path names, as a Cartesian vector
            for label, f in ((a, fa), (b, fb)):
                np.testing.assert_allclose(f @ b_cr, _cartesian_named(raw, label, cell), atol=tol,
                                           err_msg=f"{name}: {label}")
        # either way the segment is SeeK-path's segment up to symmetry
        targets = tuple(np.linalg.solve(b_cr.T, _cartesian_named(raw, lab, cell)) for lab in (a, b))
        assert _segment_equivalent((fa, fb), targets, rotations, 1e-3 + tol), (name, a, b)


@pytest.mark.parametrize("name", ["oC2_Cmce", "oA_Aea2_0", "aP_1"])
def test_old_copy_was_wrong(name):
    """The numbers SeeK-path gives, read in CRYSTAL's basis, miss its points here."""
    sg, centring, params = CASES[name]
    cell, pos, types = _crystal_structure(sg, centring, params)
    raw = si.get_seekpath_bandpath(cell, pos, types)
    b_cr = 2 * np.pi * np.linalg.inv(cell).T
    rotations = spglib.get_symmetry((cell, pos, types), symprec=1e-5)["rotations"]
    wrong = 0
    for a, b in raw["path"]:
        ends = tuple(np.array(raw["point_coords"][lab]) for lab in (a, b))
        targets = tuple(np.linalg.solve(b_cr.T, _cartesian_named(raw, lab, cell)) for lab in (a, b))
        wrong += not _segment_equivalent(ends, targets, rotations, 1e-6)
    assert wrong


def test_transform_direction():
    """f_crystal = N^-1 f_seekpath, with SeeK-path's cell = N CRYSTAL's cell.

    For a Cmce cell with a > b, N maps CRYSTAL's (a/2, b/2, 0),
    (-a/2, b/2, 0), c onto SeeK-path's cell; each point's CRYSTAL
    coordinates f satisfy N f = f_sp and give SeeK-path's Cartesian vector.
    """
    cell, pos, types = _crystal_structure(*CASES["oC2_Cmce"])
    raw = si.get_seekpath_bandpath(cell, pos, types)
    n = si.crystal_cell_matrix(raw, cell)
    assert abs(abs(np.linalg.det(n)) - 1) < 1e-9
    with contextlib.redirect_stdout(io.StringIO()):
        moved = si.to_crystal_basis(raw, cell, pos, types)
    for label, f_sp in raw["point_coords"].items():
        f = np.array(moved["point_coords"][label])
        np.testing.assert_allclose(n @ f, f_sp, atol=1e-9)
        b_cr = 2 * np.pi * np.linalg.inv(cell).T
        np.testing.assert_allclose(f @ b_cr, _cartesian_named(raw, label, cell), atol=1e-8)
    # reciprocal basis travels with the points: B = N^T B_sp
    np.testing.assert_allclose(np.array(moved["reciprocal_primitive_lattice"]),
                               n.T @ np.array(raw["reciprocal_primitive_lattice"]), atol=1e-12)


def test_triclinic_matrix_is_not_orthogonal_and_still_exact():
    cell, pos, types = _crystal_structure(*CASES["aP_1"])
    raw = si.get_seekpath_bandpath(cell, pos, types)
    n = si.crystal_cell_matrix(raw, cell)
    b_cr = 2 * np.pi * np.linalg.inv(cell).T
    with contextlib.redirect_stdout(io.StringIO()):
        moved = si.to_crystal_basis(raw, cell, pos, types)
    for label in raw["point_coords"]:
        f = np.array(moved["point_coords"][label])
        np.testing.assert_allclose(f @ b_cr, _cartesian_named(raw, label, cell), atol=1e-8)
        # the transpose instead of the inverse would not be the same point
        # unless N happens to be orthogonal
    if not np.allclose(n @ n.T, np.eye(3)):
        f_sp = np.array(raw["point_coords"]["X"])
        assert not np.allclose((n.T @ f_sp) @ b_cr, _cartesian_named(raw, "X", cell), atol=1e-6)


@pytest.mark.parametrize("name", ["cF_Fd-3m", "hP_P63mmc"])
def test_unchanged_where_seekpaths_numbers_were_right(name):
    """Cells SeeK-path only relabels by a symmetry keep its own numbers."""
    cell, pos, types = _crystal_structure(*CASES[name])
    raw = si.get_seekpath_bandpath(cell, pos, types)
    with contextlib.redirect_stdout(io.StringIO()):
        moved = si.to_crystal_basis(raw, cell, pos, types)
    assert moved["point_coords"] is raw["point_coords"]


# --------------------------------------------------------------------------
# The test/ corpus: every SP and OPT output, against a cell matrix found
# independently of SeeK-path's rotation (lattice metric and atom positions)
# --------------------------------------------------------------------------

CORPUS = Path(__file__).resolve().parents[1] / "test"


def _independent_n(cell, pos, nums, raw, rng=2, tol=1e-3):
    """Integer N with SeeK-path's crystal = N (CRYSTAL's), from metric and atoms."""
    a = np.array(cell)
    asp = np.array(raw["primitive_lattice"])
    gsp = asp @ asp.T
    psp = np.array(raw["primitive_positions"])
    tsp = np.array(raw["primitive_types"])
    pos, nums = np.array(pos), np.array(nums)
    vecs = [np.array(v) for v in itertools.product(range(-rng, rng + 1), repeat=3) if any(v)]
    lengths = [np.linalg.norm(v @ a) for v in vecs]
    rows = [[v for v, length in zip(vecs, lengths) if abs(length - np.sqrt(gsp[i, i])) < tol * 10]
            for i in range(3)]
    for n0 in rows[0]:
        for n1 in rows[1]:
            if abs((n0 @ a) @ (n1 @ a) - gsp[0, 1]) > 1e-2:
                continue
            for n2 in rows[2]:
                n = np.array([n0, n1, n2], float)
                if np.linalg.det(n) < 0.5 or abs(np.linalg.det(n) - 1) > 1e-6:
                    continue
                if not np.allclose(n @ a @ a.T @ n.T, gsp, atol=1e-2):
                    continue
                xs = (pos @ np.linalg.inv(n)) % 1.0
                for j in np.where(tsp == nums[0])[0]:
                    m = (xs + psp[j] - xs[0]) % 1.0
                    if all(np.any(np.all(np.abs((d := psp[tsp == t] - x) - np.round(d)) < 1e-3, axis=1))
                           for x, t in zip(m, nums)):
                        return n
    return None


def _corpus_outputs():
    if not CORPUS.is_dir():
        return []
    return sorted(list((CORPUS / "SP").glob("*.out")) + list((CORPUS / "OPT").glob("*.out")))


@pytest.mark.skipif(not CORPUS.is_dir(), reason="test/ corpus not present")
def test_corpus_band_paths_are_seekpaths_points():
    seen, checked, mismatches, unmatched = set(), 0, [], []
    for out in _corpus_outputs():
        with contextlib.redirect_stdout(io.StringIO()):
            st = si.parse_crystal_structure(str(out))
        if st is None:
            continue
        cell, pos, nums = np.array(st["cell"]), np.array(st["positions"]), list(st["numbers"])
        key = (tuple(np.round(cell.ravel(), 3)), len(nums))
        if key in seen:
            continue
        seen.add(key)
        try:
            raw, _, written, info = _written_points(cell, pos, nums)
        except Exception:
            continue
        n = _independent_n(cell, pos, nums, raw)
        if n is None:
            unmatched.append(out.name)
            continue
        checked += 1
        rotations = spglib.get_symmetry((cell, pos, nums), symprec=1e-5)["rotations"]
        to_crystal = np.linalg.inv(n)
        tol = 2e-3
        for (a, fa), (b, fb) in written:
            targets = tuple(to_crystal @ np.array(raw["point_coords"][lab]) for lab in (a, b))
            if not _segment_equivalent((fa, fb), targets, rotations, tol):
                mismatches.append((out.name, raw["bravais_lattice_extended"], a, b))
    assert checked >= 50, checked
    assert not unmatched, unmatched
    assert not mismatches, mismatches


# --------------------------------------------------------------------------
# Without the seekpath library: the static oA entries
# --------------------------------------------------------------------------

OUT_TEMPLATE = (" SPACE GROUP (NONCENTROSYMMETRIC)    :  A B A 2\n"
                " LATTICE PARAMETERS  (ANGSTROMS AND DEGREES) - CONVENTIONAL CELL\n"
                "        A           B           C        ALPHA        BETA       GAMMA\n"
                " {:11.5f} {:11.5f} {:11.5f} {:11.5f} {:11.5f} {:11.5f}\n\n")


@pytest.mark.parametrize("name", sorted(k for k in CASES if k.startswith("oA_")))
def test_static_oa_path_is_seekpaths_points(name, tmp_path, monkeypatch):
    d3_kpoints = pytest.importorskip("d3_kpoints")
    monkeypatch.setattr(d3_kpoints, "SEEKPATH_LIBRARY_AVAILABLE", False)
    sg, centring, params = CASES[name]
    cell, pos, types = _crystal_structure(sg, centring, params)
    raw = si.get_seekpath_bandpath(cell, pos, types)
    out = tmp_path / "oa.out"
    out.write_text(OUT_TEMPLATE.format(*params))
    with contextlib.redirect_stdout(io.StringIO()):
        segments, info = d3_kpoints.get_seekpath_full_kpath(sg, "A", str(out))
        labels = d3_kpoints.get_seekpath_labels(sg, "A", str(out))
    assert info["lookup_key"][:3] == raw["bravais_lattice_extended"]
    edges = d3_kpoints._edges_of(labels)
    assert len(edges) == len(segments)
    b_cr = 2 * np.pi * np.linalg.inv(cell).T
    rotations = spglib.get_symmetry((cell, pos, types), symprec=1e-5)["rotations"]

    def named(label):
        # a primed point is the time-reversed copy, -k; fractional in CRYSTAL's basis
        sign = -1 if label.endswith("'") else 1
        return np.linalg.solve(b_cr.T, sign * _cartesian_named(raw, label.rstrip("'"), cell))

    for (a, b), seg in zip(edges, segments):
        # SeeK-path may pick a symmetry-equivalent cell of its own; the
        # parameter is written to six decimals
        assert _segment_equivalent((np.array(seg[:3]), np.array(seg[3:])),
                                   (named(a), named(b)), rotations, 1e-5), (name, a, b)
