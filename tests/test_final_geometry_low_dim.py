"""The final geometry of a slab or molecule OPT is read.

The final-cell pattern needed ``VOLUME=`` and ``DENSITY`` on the PRIMITIVE
CELL line and the final-positions pattern an ``X/A Y/B Z/C`` header, as a
3D cell prints them. A slab prints neither VOLUME nor DENSITY and its
coordinates as ``X/A Y/B Z(ANGSTROM)``; a molecule has no cell and
``X(ANGSTROM) Y(ANGSTROM) Z(ANGSTROM)``. So no final cell or positions were
stored for them, and the plain primitive_* names kept the initial cell.

Since 1.1.6 the final geometry is stored once, under the plain names (no
final_* copy), and a slab's non-periodic c (the formal 500 A) is not stored.
"""
from conftest import find_data


def _structure(extractor, pattern):
    return extractor._extract_structural_properties(find_data(pattern).read_text(errors="ignore"))


def test_slab_final_cell_and_positions(extractor):
    p = _structure(extractor, "OPT/4LG_2x2_ABAB_opt_HSESOL3C_optimized.out")
    final = {k: p[f"primitive_{k}"] for k in ("a", "b", "alpha", "beta", "gamma")}
    assert final == {"a": 8.49711071, "b": 8.49675461, "alpha": 90.0, "beta": 90.0,
                     "gamma": 120.001135}
    assert "primitive_c" not in p and "initial_primitive_c" not in p   # the formal 500 A
    assert not [k for k in p if k.startswith("final_")]
    assert len(p["atomic_positions"]) == 96
    first = p["atomic_positions"][0]
    assert (first["element"], first["x"], first["z"]) == ("C", 1.557389736222E-01, 8.474314119562)
    assert p["atomic_position_axes"] == "X/A Y/B Z(ANGSTROM)"


def test_molecule_final_positions(extractor):
    p = _structure(extractor, "OPT/1LiFSI-1DEC-conf1_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_"
                              "HSESOL3C_optimized.out")
    assert len(p["atomic_positions"]) == 28
    assert p["atomic_positions"][0]["x"] == 5.527227200000E+01
    assert p["atomic_position_axes"] == "X(ANGSTROM) Y(ANGSTROM) Z(ANGSTROM)"
    assert "primitive_a" not in p


def test_bulk_final_geometry_unchanged(extractor):
    p = _structure(extractor, "OPT/1_dia_opt_rev1.out")
    assert p["primitive_a"] == 2.50656004 and p["primitive_cell_volume"] == 11.135747
    assert p["primitive_c"] == p["initial_primitive_c"] or p["primitive_c"] > 0   # 3D: c is stored
    assert len(p["atomic_positions"]) == 2
    assert "atomic_position_axes" not in p


def test_slab_cell_measure_is_an_area(extractor):
    # CRYSTAL's initial table heads the last column VOLUME, but for a slab it
    # is the area of the 2D cell (8.49647 * 8.49644 * sin 120 = 62.518 A^2)
    for pattern in ("OPT/4LG_2x2_ABAB_opt_HSESOL3C_optimized.out",
                    "SP/4LG_2x2_AA_opt_HSESOL3C_optimized.out",
                    "SP/4LG_FSI_2x2_AA_opt_sp.out"):
        p = _structure(extractor, pattern)
        assert not any(k.endswith("primitive_cell_volume") for k in p), pattern
        assert 62.4 < p["initial_primitive_cell_area"] < 62.6
    p = _structure(extractor, "OPT/4LG_2x2_ABAB_opt_HSESOL3C_optimized.out")
    assert p["initial_primitive_cell_area"] == 62.51804
    assert extractor._get_property_unit("initial_primitive_cell_area") == "Å²"
    assert extractor._get_property_unit("primitive_cell_length") == "Å"


def test_bulk_keeps_its_volume(extractor):
    p = _structure(extractor, "OPT/1_dia_opt_rev1.out")
    assert p["initial_primitive_cell_volume"] == 11.248833
    assert not any(k.endswith(("_cell_area", "_cell_length")) for k in p)


def test_slab_final_cell_area(extractor):
    import math
    p = _structure(extractor, "OPT/4LG_2x2_ABAB_opt_HSESOL3C_optimized.out")
    area = 8.49711071 * 8.49675461 * math.sin(math.radians(120.001135))
    assert p["primitive_cell_area"] == round(area, 6)
    # the same formula gives CRYSTAL's printed initial area
    assert 8.49647 * 8.49644 * math.sin(math.radians(120.00023)) == \
        __import__("pytest").approx(p["initial_primitive_cell_area"], abs=1e-4)


def test_no_doubled_prefixes(extractor):
    from mace.utils.db_export_values import STRUCTURE_FIELDS
    p = _structure(extractor, "OPT/4LG_2x2_ABAB_opt_HSESOL3C_optimized.out")
    assert not [k for k in p if k.startswith(("final_", "initial_initial_"))]
    assert len(p["atomic_positions"]) == 96
    p = _structure(extractor, "OPT/1_dia_opt_rev1.out")
    assert not [k for k in p if k.startswith(("final_", "initial_initial_"))]
    assert p["initial_atoms_count"] == len(p["atomic_positions"]) == 2
    # an existing database's doubled rows are left out of exports
    assert {"final_final_atomic_positions", "initial_initial_atomic_positions"} <= set(STRUCTURE_FIELDS)
