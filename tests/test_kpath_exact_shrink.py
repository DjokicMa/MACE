"""Special k-points must be written exactly: the shrink is raised to fit them.

BAND (properties, manual p.311) and the phonon BANDS block write each segment
end as integers in units of 1/ISS. Hexagonal K = (1/3, 1/3, 0) and
H = (1/3, 1/3, 1/2) are only whole numbers of 1/ISS steps when ISS is a
multiple of 3; with the usual ISS = 16 they were rounded to 5/16 = 0.3125.
The shrink is already raised when it is too small for a path; it is now also
raised to the next common multiple of the points' denominators.
"""
import io
from fractions import Fraction

import pytest

from d3_kpoints import (get_band_path_from_symmetry,
                        get_kpoint_coordinates_from_labels,
                        scale_kpoint_segments)
from d12_calc_freq import write_frequency_section

def _assert_exact(frac_segments, int_segments, shrink):
    for frac, ints in zip(frac_segments, int_segments):
        for f, i in zip(frac, ints):
            assert Fraction(i, shrink) == Fraction(f).limit_denominator(1000), (frac, ints, shrink)


@pytest.mark.parametrize("shrink_in,shrink_out", [(16, 18), (4, 6), (8, 12), (18, 18), (12, 12)])
def test_hexagonal_path_scales_to_exact_thirds(shrink_in, shrink_out):
    labels = get_band_path_from_symmetry(194, "P")
    frac = get_kpoint_coordinates_from_labels(labels, 194, "P")
    ints, shrink = scale_kpoint_segments(frac, shrink_in)
    assert shrink == shrink_out
    _assert_exact(frac, ints, shrink)


def test_quarter_points_are_not_rounded_at_shrink_6():
    """W = (1/2, 1/4, 3/4): 6/4 = 1.5 was rounded to 2 (1/3 instead of 1/4)."""
    frac = [[0.5, 0.25, 0.75, 0.0, 0.0, 0.0]]
    ints, shrink = scale_kpoint_segments(frac, 6)
    assert shrink == 8
    assert ints == [[4, 2, 6, 0, 0, 0]]


def test_seekpath_library_conversion_keeps_thirds_exact():
    """convert_to_mace_format (seekpath library route) had the same rounding."""
    seekpath_interface = pytest.importorskip("seekpath_interface")
    result = {
        "point_coords": {"GAMMA": [0.0, 0.0, 0.0], "M": [0.5, 0.0, 0.0],
                         "K": [1 / 3, 1 / 3, 0.0]},
        "path": [("GAMMA", "M"), ("M", "K"), ("K", "GAMMA")],
        "has_inversion_symmetry": True,
    }
    segments, _, info = seekpath_interface.convert_to_mace_format(result, shrink_factor=16)
    assert info["shrink_factor"] == 18
    assert segments[1] == [9, 0, 0, 6, 6, 0]


def test_phonon_bands_header_matches_exact_hexagonal_points():
    """The phonon BANDS block writes the raised ISS with the exact points."""
    settings = {"dispersion": True,
                "bands": {"path_method": "coordinates", "path": "auto",
                          "format": "vectors", "shrink": 16, "npoints": 100}}
    buf = io.StringIO()
    write_frequency_section(buf, settings, "hexagonal", 194)
    lines = buf.getvalue().splitlines()
    i = lines.index("BANDS")
    assert lines[i + 1] == "18 100"
    records = [[int(v) for v in ln.split()] for ln in lines[i + 3:i + 3 + int(lines[i + 2])]]
    assert [6, 6, 0] in [r[:3] for r in records] + [r[3:] for r in records]

