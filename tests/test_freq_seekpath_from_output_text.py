"""An opt2d12 phonon deck with an automatic SeeK-path band path.

opt2d12 hands write_frequency_section the parent's CRYSTAL output as text
(the parser's ``optimization_content``), but the SeeK-path helpers in
d3_kpoints take the path of a .out file. The text was used as a file name
and every such deck stopped with ``OSError: [Errno 36] File name too long``.

The snippet is copied verbatim from tests/data/opt_restart/tqb_pto.out
(a real CRYSTAL23 output, PbTiO3 in Pm-3m, space group 221).
"""
import io

import d12_calc_freq
from d12_calc_freq import write_frequency_section
from d3_kpoints import get_seekpath_full_kpath, scale_kpoint_segments

PTO_OUTPUT = """\
 CRYSTAL CALCULATION
 (INPUT ACCORDING TO THE INTERNATIONAL TABLES FOR X-RAY CRYSTALLOGRAPHY)
 CRYSTAL FAMILY                       :  CUBIC
 CRYSTAL CLASS  (GROTH - 1921)        :  CUBIC HEXAKISOCTAHEDRAL

 SPACE GROUP (CENTROSYMMETRIC)        :  P M 3 M

 LATTICE PARAMETERS  (ANGSTROMS AND DEGREES) - CONVENTIONAL CELL
        A           B           C        ALPHA        BETA       GAMMA
     3.94650     3.94650     3.94650    90.00000    90.00000    90.00000

"""


def _seekpath_bands():
    return {
        "dispersion": True,
        "bands": {
            "path_method": "coordinates",
            "path": "auto",
            "seekpath_full": True,
            "format": "seekpath",
            "shrink": 16,
            "npoints": 100,
        },
    }


def test_seekpath_phonon_path_from_output_text(tmp_path):
    """The deck is written, with the path the same output gives as a file."""
    buf = io.StringIO()
    write_frequency_section(buf, _seekpath_bands(), "cubic", 221, PTO_OUTPUT)
    lines = buf.getvalue().splitlines()

    out = tmp_path / "pto.out"
    out.write_text(PTO_OUTPUT)
    frac, _ = get_seekpath_full_kpath(221, "P", str(out))
    expected, shrink = scale_kpoint_segments(frac, 16)

    i = lines.index("BANDS")
    assert lines[i + 1] == f"{shrink} 100"
    assert int(lines[i + 2]) == len(expected)
    written = [[int(v) for v in ln.split()] for ln in lines[i + 3:i + 3 + len(expected)]]
    assert written == expected


def test_seekpath_auto_phonon_path_from_output_text():
    """get_auto_phonon_path takes the same text and must not crash on it."""
    segments = d12_calc_freq.get_auto_phonon_path(
        "cubic", 221, shrink=16, format_type="seekpath",
        lattice_type="P", optimization_section=PTO_OUTPUT)
    assert segments and all(len(s) == 6 for s in segments)


def test_auto_phonon_path_rounds_fractional_coordinates():
    """k-point coordinates are rounded to the nearest 1/shrink, not truncated.

    The static SeeK-path for body-centred tetragonal (tI) has parametric
    points such as 0.683436: 0.683436 * 16 = 10.93 is 11, truncation wrote 10.
    scale_kpoint_segments (d3_kpoints) already rounds.
    """
    frac, _ = get_seekpath_full_kpath(139, "I")
    segments = d12_calc_freq.get_auto_phonon_path(
        "tetragonal", 139, shrink=16, format_type="seekpath", lattice_type="I")
    assert segments == [[round(v * 16) for v in seg] for seg in frac]
    assert any(v * 16 % 1 > 0.5 for seg in frac for v in seg if v > 0)
