"""The seekpath-library band path keeps ISS small enough for fort.25.

BAND writes each segment end as integers in units of 1/ISS (manual p.309),
and fort.25 writes them again with format 6I3 (App. D, p.447), so an end
coordinate of 1000 or more prints as asterisks there. The library route
took ISS from the largest denominator of every SeeK-path point - including
points not on the path - after rounding lattice-dependent coordinates to a
denominator below 10**6, so ISS reached ~10**6 (FACTS: decks with ISS
867098-995110 ran; BAND.DAT is unaffected, fort.25 is not).

Now:
- a point SeeK-path defines by fixed fractions (HPKOT points.txt) is
  rational; its denominators set ISS to their lowest common multiple (its
  first multiple from 4, the smallest ISS scale_kpoint_segments writes),
  and it is written exactly;
- a lattice-dependent (parametric) point is written as k/ISS with
  ISS <= 999, ISS a multiple of that LCM, choosing the ISS that moves the
  parametric points least, provided each moves less than half the k-point
  spacing of the path at NSUB = 10000 (MACE's largest default); otherwise
  the old exact representation is kept;
- whether a point is parametric comes from SeeK-path's own definitions,
  not from whether its value happens to be close to p/q (an old heuristic
  that multiplied ISS by such a q).
"""
from fractions import Fraction

import numpy as np
import pytest

seekpath_interface = pytest.importorskip("seekpath_interface")
pytest.importorskip("seekpath")

from d3_kpoints import scale_kpoint_segments  # noqa: E402


def _written(result):
    segments, labels, info = seekpath_interface.convert_to_mace_format(result, 16)
    return segments, info["shrink_factor"]


def _max_displacement(result, segments, iss):
    B = np.array(result["reciprocal_primitive_lattice"])
    worst = 0.0
    for (a, b), seg in zip(result["path"], segments):
        for label, ints in ((a, seg[:3]), (b, seg[3:])):
            d = np.array(ints) / iss - np.array(result["point_coords"][label])
            worst = max(worst, float(np.linalg.norm(d @ B)))
    return worst


def _path_length(result):
    B = np.array(result["reciprocal_primitive_lattice"])
    pc = result["point_coords"]
    return sum(float(np.linalg.norm((np.array(pc[b]) - np.array(pc[a])) @ B))
               for a, b in result["path"])


@pytest.mark.parametrize("sg,cell,iss", [
    (221, (4.0, 4.0, 4.0, 90, 90, 90), 4),      # cP2: halves only (2, raised to 4)
    (229, (3.3, 3.3, 3.3, 90, 90, 90), 4),      # cI1: P = (1/4, 1/4, 1/4)
    (225, (5.6, 5.6, 5.6, 90, 90, 90), 8),      # cF2: U, K = 3/8, 5/8
    (194, (3.2, 3.2, 5.2, 90, 90, 120), 6),     # hP2: K = 1/3, A = 1/2
    (14, (5.1, 6.3, 7.4, 90, 104, 90), 4),      # mP1: path points are halves;
                                                #  the parametric H, M are off the path
])
def test_rational_paths_get_the_lowest_common_denominator(sg, cell, iss, seekpath_path):
    result = seekpath_path(sg, cell)
    segments, got = _written(result)
    assert got == iss
    for (a, b), seg in zip(result["path"], segments):
        for label, ints in ((a, seg[:3]), (b, seg[3:])):
            exact = [Fraction(c).limit_denominator(100) for c in result["point_coords"][label]]
            assert [Fraction(i, got) for i in ints] == exact, label


@pytest.mark.parametrize("sg,cell", [
    (12, (9.31415, 3.02718, 3.11935, 90, 120.01234, 90)),    # mC1
    (63, (3.07123, 6.38461, 7.49227, 90, 90, 90)),           # oC1
    (139, (5.17213, 5.17213, 7.56547, 90, 90, 90)),          # tI2
    (166, (4.14321, 4.14321, 28.63614, 90, 90, 120)),        # hR1 (Bi2Se3-like)
    (70, (5.17212, 6.40912, 7.56547, 90, 90, 90)),           # oF3
    (71, (7.50465, 6.40912, 3.06708, 90, 90, 90)),           # oI3
    (38, (3.07123, 6.38461, 7.49227, 90, 90, 90)),           # oA1
])
def test_parametric_points_fit_fort25(sg, cell, seekpath_path):
    result = seekpath_path(sg, cell)
    segments, iss = _written(result)
    assert iss <= 999
    assert max(abs(c) for seg in segments for c in seg) <= 999
    # every point within half the k-point spacing at NSUB = 10000
    assert _max_displacement(result, segments, iss) <= 0.5 * _path_length(result) / 10000


def test_rational_points_stay_exact_next_to_parametric_ones(seekpath_path):
    """tI2 (I4/mmm): X, P, N are fixed fractions; S, S_0, R, G are not."""
    result = seekpath_path(139, (5.17213, 5.17213, 7.56547, 90, 90, 90))
    segments, iss = _written(result)
    exact = {"GAMMA", "X", "P", "N", "M"}
    for (a, b), seg in zip(result["path"], segments):
        for label, ints in ((a, seg[:3]), (b, seg[3:])):
            if label in exact:
                ref = [Fraction(c).limit_denominator(100) for c in result["point_coords"][label]]
                assert [Fraction(i, iss) for i in ints] == ref, label


def test_parametric_points_are_seekpaths_not_near_fractions():
    """Which points are lattice-dependent comes from SeeK-path's definitions
    (hpkot/band_path_data/<lattice>/points.txt), not from the value: the old
    test (exact_shrink_step) took any coordinate within 1e-6 of p/q, q <= 12,
    as a fixed point, so a parametric C_0 that happened to sit at 4/9 made ISS
    a multiple of 9 and was then held exact instead of approximated."""
    assert seekpath_interface.parametric_point_names("oC1") == {"SIGMA_0", "C_0", "A_0", "E_0"}
    assert seekpath_interface.parametric_point_names("cF2") == set()
    assert "H_2" in seekpath_interface.parametric_point_names("hR1")
    result = {
        "bravais_lattice_extended": "oC1",
        "point_coords": {"GAMMA": [0.0, 0.0, 0.0], "Y": [-0.5, 0.5, 0.0],
                         "C_0": [-0.4444444, 0.5555556, 0.0],
                         "SIGMA_0": [0.4444444, 0.4444444, 0.0]},
        "path": [("GAMMA", "Y"), ("Y", "C_0"), ("SIGMA_0", "GAMMA")],
        "reciprocal_primitive_lattice": [[1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "has_inversion_symmetry": True,
    }
    assert seekpath_interface.rational_iss_step(result) == 2


def test_library_iss_survives_the_deck_round_trip():
    """CRYSTALOptToD3 divides the library's integers by its ISS and scales
    them again with scale_kpoint_segments; an ISS that already represents
    every point must come back unchanged (an odd 785 was raised to 786 and
    every point rounded again)."""
    odd = [[0.0, 0.0, 0.0, 331 / 785, 454 / 785, 0.0]]
    assert scale_kpoint_segments(odd, 785) == ([[0, 0, 0, 331, 454, 0]], 785)


def test_phonon_bands_use_the_library_iss():
    """opt2d12's phonon BANDS block (AgBr R-3c, real output) writes the
    library's ISS and points, as opt2d3 does; it rescaled them to the band
    shrink instead, rounding the lattice-dependent points again."""
    import io
    from pathlib import Path
    from d12_calc_freq import write_frequency_section

    out = Path(__file__).parent / "data" / "opt_restart" / "tqc_agbr.out"
    settings = {"dispersion": True,
                "bands": {"path_method": "coordinates", "path": "auto", "seekpath_full": True,
                          "format": "seekpath", "shrink": 16, "npoints": 100}}
    buf = io.StringIO()
    write_frequency_section(buf, settings, "trigonal", 167, out.read_text())
    lines = buf.getvalue().splitlines()
    i = lines.index("BANDS")
    iss = int(lines[i + 1].split()[0])
    records = [[int(v) for v in ln.split()] for ln in lines[i + 3:i + 3 + int(lines[i + 2])]]
    segments, _, info = seekpath_interface.get_accurate_bandpath(str(out))
    assert iss == info["shrink_factor"] <= 999
    assert records == segments
