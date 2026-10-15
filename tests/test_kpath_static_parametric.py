"""Static band paths take the cell's own lattice-dependent points.

Without the seekpath library, the static ``seekpath_data`` entries of the
lattices whose SeeK-path (HPKOT) points depend on the cell held the values
of one cell: tI1 Z = (0.341718, 0.341718, -0.341718) for every
body-centered tetragonal crystal with c < a, hR1 H_2 at 0.744565, and so on.
For tI1, tI2, hR1, hR2, oF1 and the C-centered monoclinic entry (mS1 =
SeeK-path's mC1 with its primed copy) the points are now evaluated for the
cell CRYSTAL prints (hpkot_static) and given in CRYSTAL's reciprocal basis,
with the ISS the library route takes, so both routes write the same deck.
Cells not in SeeK-path's setting, or of another variant, keep the old entry
with a warning that says why.
"""
import contextlib
import io
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
d3_kpoints = pytest.importorskip("d3_kpoints")
hs = pytest.importorskip("hpkot_static")

import crystal_out_text as cot  # noqa: E402

CORPUS = Path(__file__).resolve().parents[1] / "test"
CONVERTED = ["tI1", "tI2", "hR1", "hR2", "oF1", "oC1", "oC2", "mC1"]

# CRYSTAL's primitive cell in terms of the conventional one (rows),
# (W^T)^-1 with W as CRYSTAL prints it
CRYSTAL_PRIMITIVE = {
    "C": np.array([[0.5, 0.5, 0], [-0.5, 0.5, 0], [0, 0, 1]]),
    "F": np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]),
    "I": np.array([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]]),
    "R": np.array([[2, 1, 1], [-1, 1, 1], [-1, -2, 1]]) / 3,
}


def _out(tmp_path, params, sg):
    out = tmp_path / f"cell_{sg}.out"
    out.write_text(cot.echo("X", params, centro=sg in d3_kpoints.CENTROSYMMETRIC_SPACE_GROUPS))
    return out


def _static(monkeypatch, sg, lattice, out):
    monkeypatch.setattr(d3_kpoints, "SEEKPATH_LIBRARY_AVAILABLE", False)
    monkeypatch.setenv("MACE_STATIC_KPATH", "1")  # the static route, asked for
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        segments, info = d3_kpoints.get_seekpath_full_kpath(sg, lattice, str(out))
        labels = d3_kpoints.get_seekpath_labels(sg, lattice, str(out))
    return segments, info, labels, printed.getvalue()


# --------------------------------------------------------------------------
# The transcription, against SeeK-path's own files
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key", CONVERTED)
def test_entries_are_seekpaths_files(key):
    seekpath = pytest.importorskip("seekpath")
    data = Path(seekpath.__file__).parent / "hpkot" / "band_path_data" / key
    entry = hs.HPKOT_POINTS[key]
    params = [tuple(line.split(None, 1)) for line in
              (data / "k_vector_parameters.txt").read_text().strip().splitlines()]
    assert entry["parameters"] == params
    assert entry["points"].strip().splitlines() == \
        [" ".join(line.split()) for line in (data / "points.txt").read_text().strip().splitlines()]
    path = [tuple(line.split()) for line in (data / "path.txt").read_text().strip().splitlines()]
    assert [tuple(e.split("-")) for e in entry["path"].split()] == path


def test_hpkot_p_matrices_are_seekpaths():
    pytest.importorskip("seekpath")
    from seekpath.hpkot.spg_mapping import get_P_matrix
    for family, p in hs.HPKOT_P.items():
        np.testing.assert_allclose(p, get_P_matrix(family)[0], atol=1e-15)


def test_ti1_parameter_at_a4_c3():
    """eta = (1 + c^2/a^2)/4 = (1 + 9/16)/4 = 0.390625 (the old entry: 0.341718)."""
    params = {"a": 4.0, "b": 4.0, "c": 3.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    points = hs.evaluate("tI1", params)
    assert points["Z"] == [0.390625, 0.390625, -0.390625]
    assert points["Z_0"] == [-0.390625, 0.609375, 0.390625]
    crystal, key = hs.crystal_points("tI1", params)
    assert key == "tI1" and crystal["Z"] == points["Z"] and crystal["Z'"] == [-0.390625, -0.390625, 0.390625]


# --------------------------------------------------------------------------
# Against the seekpath library, structure by structure
# --------------------------------------------------------------------------

CELLS = [
    (139, "I", (3.8, 3.8, 9.5, 90, 90, 90)), (139, "I", (4.0, 4.0, 3.0, 90, 90, 90)),
    (82, "I", (4.0, 4.0, 3.0, 90, 90, 90)), (82, "I", (4.0, 4.0, 7.0, 90, 90, 90)),
    (141, "I", (3.78, 3.78, 9.51, 90, 90, 90)), (87, "I", (5.0, 5.0, 3.3, 90, 90, 90)),
    (166, "R", (4.5, 4.5, 11.9, 90, 90, 120)), (166, "R", (5.0, 5.0, 4.0, 90, 90, 120)),
    (167, "R", (6.58, 6.58, 17.7, 90, 90, 120)), (160, "R", (5.0, 5.0, 12.0, 90, 90, 120)),
    (161, "R", (5.0, 5.0, 5.0, 90, 90, 120)), (148, "R", (7.7, 7.7, 20.0, 90, 90, 120)),
    (69, "F", (3.0, 7.0, 8.0, 90, 90, 90)), (70, "F", (3.0, 7.0, 8.0, 90, 90, 90)),
    (22, "F", (3.0, 7.0, 8.0, 90, 90, 90)), (43, "F", (3.0, 7.0, 8.0, 90, 90, 90)),
    (12, "C", (10.6, 3.45, 9.33, 90, 120.43, 90)), (15, "C", (9.0, 4.0, 12.0, 90, 95, 90)),
    (5, "C", (9.0, 4.0, 12.0, 90, 95, 90)), (9, "C", (10.6, 3.45, 9.33, 90, 120.43, 90)),
    (8, "C", (6.0, 4.0, 5.0, 90, 100, 90)),
    # C-centered orthorhombic: oC1 (a <= b) and oC2 (a > b, graphite AB slip's 7.36 x 6.22),
    # with and without inversion
    (63, "C", (3.0, 5.1, 7.4, 90, 90, 90)), (63, "C", (5.1, 3.0, 7.4, 90, 90, 90)),
    (65, "C", (7.36, 6.22, 4.25, 90, 90, 90)), (65, "C", (3.75, 7.72, 2.49, 90, 90, 90)),
    (36, "C", (4.0, 6.0, 5.0, 90, 90, 90)), (36, "C", (6.0, 4.0, 5.0, 90, 90, 90)),
]


def _structure(sg, centering, params):
    spglib = pytest.importorskip("spglib")
    si = pytest.importorskip("seekpath_interface")
    first_hall = next(h for h in range(1, 531) if spglib.get_spacegroup_type(h).number == sg)
    sym = spglib.get_symmetry_from_database(first_hall)
    p = CRYSTAL_PRIMITIVE[centering]
    cell = p @ si.cell_params_to_vectors(*params)
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


@pytest.mark.parametrize("sg,centering,params", CELLS)
def test_static_equals_the_library_route(tmp_path, monkeypatch, sg, centering, params):
    """Same ISS and the same integers as the library route, or segment by
    segment a symmetry image of its segments (where the library keeps
    SeeK-path's own numbers)."""
    pytest.importorskip("seekpath")
    spglib = pytest.importorskip("spglib")
    si = pytest.importorskip("seekpath_interface")
    cell, pos, types = _structure(sg, centering, params)
    segments, info, labels, printed = _static(monkeypatch, sg, centering, _out(tmp_path, params, sg))
    assert info["source"] == "seekpath_data_parametric", printed
    assert "WARNING" not in printed.replace("WARNING: Non-centrosymmetric path not available", "")
    assert "Install 'seekpath'" not in printed
    iss = info["shrink_factor"]
    static_ints, _ = d3_kpoints.scale_kpoint_segments(segments, iss)

    with contextlib.redirect_stdout(io.StringIO()):
        raw = si.get_seekpath_bandpath(cell, pos, types)
        moved = si.to_crystal_basis(raw, cell, pos, types)
        library, _, linfo = si.convert_to_mace_format(moved)
    assert raw["bravais_lattice_extended"] == hs.hpkot_key(info["lookup_key"])
    assert linfo["shrink_factor"] == iss
    # the static entry may add the primed copy; the library path is its start
    edges = d3_kpoints._edges_of(labels)[:len(library)]
    assert [tuple(e) for e in edges] == [tuple(e) for e in raw["path"]]
    if static_ints[:len(library)] == library:
        return
    rotations = spglib.get_symmetry((cell, pos, types), symprec=1e-5)["rotations"]
    original = {lab: np.array(seg[:3]) / iss for (lab, _), seg in zip(edges, library)}
    original.update({lab: np.array(seg[3:]) / iss for (_, lab), seg in zip(edges, library)})
    moved_pts = {lab: np.array(seg[:3]) / iss for (lab, _), seg in zip(edges, static_ints)}
    moved_pts.update({lab: np.array(seg[3:]) / iss for (_, lab), seg in zip(edges, static_ints)})
    assert si._same_path_by_symmetry(original, moved_pts, edges, rotations, True)


@pytest.mark.parametrize("sg,centering,params", CELLS)
def test_points_within_1e6_of_seekpaths(sg, centering, params):
    """Before the ISS choice, each point is SeeK-path's point in CRYSTAL's
    basis to 1e-6 - or, where SeeK-path standardized to a cell related to
    CRYSTAL's by a symmetry operation, an image of it under that symmetry."""
    pytest.importorskip("seekpath")
    spglib = pytest.importorskip("spglib")
    si = pytest.importorskip("seekpath_interface")
    cell, pos, types = _structure(sg, centering, params)
    raw = si.get_seekpath_bandpath(cell, pos, types)
    m = np.linalg.inv(si.crystal_cell_matrix(raw, cell))
    lookup = raw["bravais_lattice_extended"]
    lookup = "mS1" if lookup == "mC1" else lookup
    points, _ = hs.crystal_points(lookup, dict(zip(("a", "b", "c", "alpha", "beta", "gamma"), params)))
    rotations = spglib.get_symmetry((cell, pos, types), symprec=1e-5)["rotations"]
    for label, f_sp in raw["point_coords"].items():
        target = m @ np.array(f_sp)
        ours = np.array(points[label])
        if np.allclose(ours, target, atol=1e-6):
            continue
        assert any(np.allclose(d := s * (ours @ w) - target, np.round(d), atol=1e-6)
                   for w in rotations for s in (1, -1)), label


def test_primed_points_are_seekpaths_without_time_reversal():
    """SeeK-path appends X' = -X when there is neither inversion nor time reversal."""
    pytest.importorskip("seekpath")
    si = pytest.importorskip("seekpath_interface")
    cell, pos, types = _structure(82, "I", (4.0, 4.0, 3.0, 90, 90, 90))
    raw = si.get_seekpath_bandpath(cell, pos, types, with_time_reversal=False)
    assert not raw["has_inversion_symmetry"]
    primed = {k: v for k, v in raw["point_coords"].items() if k.endswith("'")}
    assert primed
    for label, coords in primed.items():
        np.testing.assert_allclose(coords, -np.array(raw["point_coords"][label[:-1]]), atol=1e-12)
    points, _ = hs.crystal_points("tI1_noinv", {"a": 4.0, "b": 4.0, "c": 3.0, "alpha": 90.0,
                                                "beta": 90.0, "gamma": 90.0})
    for label in primed:
        np.testing.assert_allclose(points[label], -np.array(points[label[:-1]]), atol=1e-12)


# --------------------------------------------------------------------------
# Refusals keep the old entry and say why
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sg,lattice,params,reason", [
    # axes not in a < b < c
    (69, "F", (3.0, 8.0, 7.0, 90, 90, 90), "not in SeeK-path's order a < b < c"),
    # C2/m whose a-c mesh SeeK-path reduces
    (12, "C", (6.0, 4.0, 5.0, 90, 130, 90), "a-c mesh is not reduced"),
    # an mC2 cell: the static entry is mC1's
    (12, "C", (5.0, 9.0, 6.0, 90, 105, 90), "SeeK-path takes mC2 for this cell, not mC1"),
])
def test_refusal_keeps_the_entry_with_a_warning(tmp_path, monkeypatch, sg, lattice, params, reason):
    segments, info, _, printed = _static(monkeypatch, sg, lattice, _out(tmp_path, params, sg))
    assert info["source"] == "seekpath_data"
    assert segments == d3_kpoints.seekpath_data[info["lookup_key"]]["segments"]
    assert "WARNING: the static" in printed and "Install the 'seekpath' library" in printed
    assert reason in printed


@pytest.mark.parametrize("key", ["mC1"])
def test_entries_whose_labels_do_not_name_their_segments_are_not_converted(key):
    params = {"a": 11.49, "b": 10.08, "c": 13.89, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    segments, reason, _ = d3_kpoints.parametric_static_segments(key, params)
    assert segments is None and "do not name its segments" in reason


def test_unconverted_lattices_still_warn(tmp_path, monkeypatch):
    """aP, mP, oP, oI, oF3, ... keep the warning (oS/oC is converted since 1.1.5)."""
    for sg, lattice, params in [(2, "P", (5.1, 3.0, 7.4, 100, 95, 80)),
                                (14, "P", (7.4, 5.1, 3.0, 90, 115, 90)),
                                (62, "P", (5.1, 7.4, 3.0, 90, 90, 90)),
                                (71, "I", (3.0, 5.1, 7.4, 90, 90, 90))]:
        _, info, _, printed = _static(monkeypatch, sg, lattice, _out(tmp_path, params, sg))
        assert info["source"] == "seekpath_data" and "WARNING: the static" in printed, sg


def test_cubic_and_hexagonal_unchanged(tmp_path, monkeypatch):
    for sg, lattice, params, key in [(227, "F", (5.43,) * 3 + (90,) * 3, "cF2"),
                                     (194, "P", (3.2, 3.2, 5.2, 90, 90, 120), "hP2")]:
        segments, info, _, _ = _static(monkeypatch, sg, lattice, _out(tmp_path, params, sg))
        assert info["source"] == "seekpath_data"
        assert segments == d3_kpoints.seekpath_data[key]["segments"]


def test_without_an_output_nothing_changes(monkeypatch):
    monkeypatch.setattr(d3_kpoints, "SEEKPATH_LIBRARY_AVAILABLE", False)
    for sg, lattice, key in [(139, "I", "tI1"), (166, "R", "hR1"), (69, "F", "oF1"), (12, "C", "mS1")]:
        with contextlib.redirect_stdout(io.StringIO()):
            segments, info = d3_kpoints.get_seekpath_full_kpath(sg, lattice)
        assert info["source"] == "seekpath_data" and info["lookup_key"] == key
        assert segments is d3_kpoints.seekpath_data[key]["segments"]


# --------------------------------------------------------------------------
# Rhombohedral axes, supercells, W from the output
# --------------------------------------------------------------------------

def test_rhombohedral_axes_give_the_hexagonal_result(tmp_path, monkeypatch):
    rho = tmp_path / "rho.out"
    rho.write_text(cot.echo("R -3 M", (4.7458, 4.7458, 4.7458, 57.23, 57.23, 57.23)))
    hexa = tmp_path / "hex.out"
    p = d3_kpoints.hexagonal_axes({"a": 4.7458, "b": 4.7458, "c": 4.7458, "alpha": 57.23,
                                   "beta": 57.23, "gamma": 57.23})
    hexa.write_text(cot.echo("R -3 M", (p["a"], p["b"], p["c"], 90, 90, 120)))
    a = _static(monkeypatch, 166, "R", rho)
    b = _static(monkeypatch, 166, "R", hexa)
    assert a[1]["source"] == "seekpath_data_parametric"
    np.testing.assert_allclose(a[0], b[0], atol=1e-5)


def test_supercell_maps_the_converted_points(tmp_path, monkeypatch):
    e = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
    out = tmp_path / "super.out"
    out.write_text(cot.echo("I 4/M M M", (4.0, 4.0, 3.0, 90, 90, 90)) + cot.supercel(e)
                   + cot.block((4.0, 4.0, 3.0, 90, 90, 90), [(20, "CA", 0, 0, 0)], centering="1/1"))
    plain = _out(tmp_path, (4.0, 4.0, 3.0, 90, 90, 90), 139)
    segments, info, _, _ = _static(monkeypatch, 139, "I", out)
    base, _, _, _ = _static(monkeypatch, 139, "I", plain)
    assert info["source"] == "seekpath_data_parametric"
    np.testing.assert_allclose(segments, d3_kpoints.expand_segments(base, e), atol=1e-12)


def test_optimized_supercell_maps_the_host_path(tmp_path, monkeypatch):
    """An OPT of a SUPERCEL-built cell (the defect-supercell case): the final
    block is the supercell, so its cell is not the host's conventional cell.
    The path is the host's, expanded to the supercell's basis, as for the
    SP of the same supercell - not a path for the supercell's cell read as
    the host's group."""
    e = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
    head = cot.echo("I 4/M M M", (4.0, 4.0, 3.0, 90, 90, 90)) + cot.supercel(e)
    super_params = (5.0, 3.2, 3.2, 100.0, 110.0, 120.0)   # the supercell's own cell
    sp = tmp_path / "super_sp.out"
    sp.write_text(head + cot.block(super_params, [(20, "CA", 0, 0, 0)], centering="1/1"))
    opt = tmp_path / "super_opt.out"
    opt.write_text(head + cot.block(super_params, [(20, "CA", 0, 0, 0)], centering="1/1")
                   + cot.opt_step(super_params, [(20, "CA", 0, 0, 0)], centering="1/1")
                   + cot.final_block(super_params, [(20, "CA", 0, 0, 0)], centering="1/1"))
    params = d3_kpoints.extract_lattice_parameters_from_output(str(opt))
    assert params["expansion"] == e
    assert (params["a"], params["c"]) == (4.0, 3.0)
    a = _static(monkeypatch, 139, "I", sp)
    b = _static(monkeypatch, 139, "I", opt)
    assert b[1].get("expansion") == e
    np.testing.assert_allclose(b[0], a[0], atol=1e-12)


@pytest.mark.skipif(not CORPUS.is_dir(), reason="test/ corpus not present")
def test_table_a1_is_the_printed_w():
    """Every centered corpus output prints Table A.1's W for its centering."""
    si = pytest.importorskip("seekpath_interface")
    import re
    seen = set()
    for out in sorted(CORPUS.glob("*/*.out")):
        text = out.read_text(errors="ignore")
        if "TRANSFORMATION MATRIX PRIMITIVE-CRYSTALLOGRAPHIC" not in text:
            continue
        symbol = re.search(r"SPACE GROUP.*?:\s*(\S)", text).group(1)
        for block in si._geometry_blocks(text.split("\n")):
            if block["W"] is not None:
                np.testing.assert_array_equal(block["W"], hs.CRYSTAL_W[symbol], err_msg=out.name)
                seen.add(symbol)
    assert {"C", "F", "I", "R"} <= seen


@pytest.mark.skipif(not CORPUS.is_dir(), reason="test/ corpus not present")
def test_corpus_static_route_writes_the_library_deck(monkeypatch):
    """Centered corpus outputs of the converted lattices: the static deck is
    the library deck (same ISS and integers). A group without inversion may
    add the primed copy after it; a centrosymmetric one writes exactly the
    library's path (C 2/M did not, before mS1 stopped being the Cc path;
    the corpus has no I 41/A M D output)."""
    pytest.importorskip("seekpath")
    import re
    names = {"I 4/M M M": 139, "I 4/M": 87, "R -3 C": 167, "R -3 M": 166, "R -3": 148,
             "F M M M": 69, "C 2": 5, "C M": 8, "C 2/M": 12, "C M C A": 64, "C M C E": 64}
    centrosymmetric = {"I 4/M M M", "I 4/M", "R -3 C", "R -3 M", "R -3", "F M M M", "C 2/M", "C M C A", "C M C E"}
    checked = 0
    for out in sorted(CORPUS.glob("*/*.out")):
        text = out.read_text(errors="ignore")
        match = re.search(r"SPACE GROUP.*?:\s*(.*?)\s*$", text, re.M)
        if not match or match.group(1) not in names or "PRIMITIVE CELL - CENTRING" not in text:
            continue
        sg, lattice = names[match.group(1)], match.group(1)[0]
        with contextlib.redirect_stdout(io.StringIO()):
            monkeypatch.setattr(d3_kpoints, "SEEKPATH_LIBRARY_AVAILABLE", True)
            lib, linfo = d3_kpoints.get_seekpath_full_kpath(sg, lattice, str(out))
            monkeypatch.setattr(d3_kpoints, "SEEKPATH_LIBRARY_AVAILABLE", False)
            monkeypatch.setenv("MACE_STATIC_KPATH", "1")  # the static route, asked for
            sta, sinfo = d3_kpoints.get_seekpath_full_kpath(sg, lattice, str(out))
            lib_ints, _ = d3_kpoints.scale_kpoint_segments(lib, linfo["shrink_factor"])
            sta_ints, _ = d3_kpoints.scale_kpoint_segments(sta, sinfo["shrink_factor"])
        assert sinfo["source"] == "seekpath_data_parametric", out.name
        assert sinfo["shrink_factor"] == linfo["shrink_factor"], out.name
        assert sta_ints[:len(lib_ints)] == lib_ints, out.name
        if match.group(1) in centrosymmetric:
            assert len(sta_ints) == len(lib_ints), out.name
        checked += 1
    assert checked >= 22


@pytest.mark.parametrize("params,variant", [
    ((4.0, 5.0, 6.0, 90, 90, 90), "oF3"),       # 1/a^2 < 1/b^2 + 1/c^2
    ((3.0, 8.0, 9.0, 90, 90, 90), "oF1"),       # 1/a^2 > 1/b^2 + 1/c^2
    ((8.0, 9.0, 3.0, 90, 90, 90), "oF1"),       # the same cell, axes in another order
])
def test_of_variant_is_seekpaths(tmp_path, monkeypatch, params, variant):
    """The oF variant was chosen by the shortest axis alone, so every a < b < c
    cell was oF1 - (4, 5, 6) included, which SeeK-path takes as oF3 (the
    static route then wrote the oF1 path with a refusal warning)."""
    assert d3_kpoints.determine_orthorhombic_f_variant(*params[:3]) == variant
    assert hs.hpkot_variant("oF", dict(zip(("a", "b", "c", "alpha", "beta", "gamma"),
                                           sorted(params[:3]) + [90, 90, 90]))) == variant
    _, info, _, _ = _static(monkeypatch, 69, "F", _out(tmp_path, params, 69))
    assert info["lookup_key"] == variant



def test_oc_variant_is_seekpaths():
    """C-centered orthorhombic: oC1 when a <= b, else oC2 (hpkot.get_path). The static
    route named every one "oS1" (oC1) and wrote its numbers in SeeK-path's cell: for
    graphite AB slip (a 7.36 > b 6.22, oC2) a "Y" at (1/2, 1/2, 0) of CRYSTAL's basis,
    outside the first zone, and F_0, DELTA_0 for another cell."""
    assert d3_kpoints.determine_orthorhombic_s_variant(3.0, 5.1, 7.4) == "oS1"
    assert d3_kpoints.determine_orthorhombic_s_variant(7.36, 6.22, 4.25) == "oC2"
    params = {"a": 7.36, "b": 6.22, "c": 4.25, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    points, key = hs.crystal_points("oC2", params)
    zeta = (1 + 6.22 ** 2 / 7.36 ** 2) / 4
    assert key == "oC2" and points["Y"] == [0.5, -0.5, 0.0]
    np.testing.assert_allclose(points["F_0"], [1 - zeta, -zeta, 0], atol=1e-12)
    np.testing.assert_allclose(points["DELTA_0"], [zeta, zeta, 0], atol=1e-12)


def _in_first_zone(f, rec):
    import itertools
    k = np.asarray(f) @ rec
    for n in itertools.product(range(-2, 3), repeat=3):
        if any(n):
            g = np.array(n) @ rec
            if k @ g > g @ g / 2 + 2e-3 * max(1.0, g @ g):     # ISS rounding of a boundary point
                return False
    return True


@pytest.mark.skipif(not CORPUS.is_dir(), reason="test/ corpus not present")
@pytest.mark.parametrize("static", [False, True])
def test_corpus_band_paths_lie_in_the_first_zone(monkeypatch, static):
    """Every point of the SeeK-path path opt2d3 writes for a 3D corpus output (the library
    route, and the static route asked for) lies in the first Brillouin zone of the cell
    CRYSTAL runs: the coordinates are in CRYSTAL's reciprocal basis. (A triclinic static
    path is in SeeK-path's reduced cell and is not checked; it warns.) """
    pytest.importorskip("seekpath")
    import CRYSTALOptToD3 as d3gen
    import seekpath_interface as si
    checked = 0
    for out in sorted(set(CORPUS.glob("OPT/*.out")) | set(CORPUS.glob("SP/*.out"))):
        if "MOLECULE" in out.name:
            continue
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if d3_kpoints.output_dimensionality(str(out)) != 3:
                continue
            g = d3gen.D3Generator(str(out), "BAND")
            sg, lat = g.structure_info.get("space_group"), g.structure_info.get("lattice_type")
            st = si.parse_crystal_structure(str(out))
            monkeypatch.setattr(d3_kpoints, "SEEKPATH_LIBRARY_AVAILABLE", not static)
            if static:
                monkeypatch.setenv("MACE_STATIC_KPATH", "1")
            try:
                frac, info = d3_kpoints.get_seekpath_full_kpath(sg, lat, str(out))
            finally:
                monkeypatch.delenv("MACE_STATIC_KPATH", raising=False)
        if st is None or not frac or info.get("lookup_key", "")[:2] == "aP":
            continue
        rec = 2 * np.pi * np.linalg.inv(np.asarray(st["cell"], float)).T
        for seg in frac:
            for f in (seg[:3], seg[3:]):
                assert _in_first_zone(f, rec), (out.name, info.get("lookup_key") or info.get("bravais_lattice_extended"), f)
        checked += 1
    assert checked >= 40
