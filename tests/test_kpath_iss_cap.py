"""ISS never exceeds what fort.25 can hold, on either k-path route.

choose_band_iss moves the lattice-dependent points to the nearest k/ISS
with ISS <= FORT25_MAX_ISS when that moves them by less than half the
path's k-point spacing at NSUB 10000. For a long-axis cell no such ISS
exists (a C2/m cell of about 9.7 x 3.7 x 20.8 A, beta 91.7: mC1), and the
points used to be written exactly, with ISS ~10**6 (959108 on the library
route, 588382 on the static one). BAND writes the ends again in fort.25 in
a fixed-width integer format (manual App. D p.447: 6I3; measured 6I5,
measured on CRYSTAL23), so ends of 10**5 and more overflow there. ISS now stays at most
999, taking the ISS that moves the points least, with a warning that gives
the move.
"""
import contextlib
import io

import pytest

np = pytest.importorskip("numpy")
d3_kpoints = pytest.importorskip("d3_kpoints")
si = pytest.importorskip("seekpath_interface")

import crystal_out_text as cot  # noqa: E402

# a long-c C-centred monoclinic cell (synthetic numbers)
LONG_MC1 = (9.668, 3.749, 20.767, 90.0, 91.689, 90.0)


def _written_max(segments, iss):
    ints, _ = d3_kpoints.scale_kpoint_segments(segments, iss)
    return max(abs(c) for seg in ints for c in seg)


def test_static_route_caps_iss(tmp_path, monkeypatch):
    out = tmp_path / "long_mc1.out"
    out.write_text(cot.echo("X", LONG_MC1, centro=True))
    monkeypatch.setattr(d3_kpoints, "SEEKPATH_LIBRARY_AVAILABLE", False)
    monkeypatch.setenv("MACE_STATIC_KPATH", "1")  # the static route, asked for
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        segments, info = d3_kpoints.get_seekpath_full_kpath(12, "C", str(out))
    assert info["source"] == "seekpath_data_parametric", printed.getvalue()
    iss = info["shrink_factor"]
    assert 4 <= iss <= si.FORT25_MAX_ISS
    assert _written_max(segments, iss) <= si.FORT25_MAX_ISS
    # the move is small: well under a thousandth of a reciprocal vector (1/A)
    assert 0 < info["max_point_shift"] < 2e-3
    assert "WARNING: no ISS up to 999" in printed.getvalue()


def test_library_route_caps_iss(seekpath_path):
    result = seekpath_path(12, LONG_MC1)
    assert result["bravais_lattice_extended"] == "mC1"
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        segments, _, info = si.convert_to_mace_format(result, 16)
    iss = info["shrink_factor"]
    assert 4 <= iss <= si.FORT25_MAX_ISS
    assert max(abs(c) for seg in segments for c in seg) <= si.FORT25_MAX_ISS
    assert 0 < info["max_point_shift"] < 2e-3
    assert "WARNING: no ISS up to 999" in printed.getvalue()


def test_fitting_cells_keep_their_iss_and_print_no_warning(seekpath_path):
    """tI2 fits within tolerance: unchanged, no warning."""
    result = seekpath_path(139, (5.17213, 5.17213, 7.56547, 90, 90, 90))
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        iss, shift = si.choose_band_iss(result)
    assert iss <= 999 and "WARNING" not in printed.getvalue()
    assert shift <= 0.5 * sum(
        float(np.linalg.norm((np.array(result["point_coords"][b]) - np.array(result["point_coords"][a]))
                             @ np.array(result["reciprocal_primitive_lattice"])))
        for a, b in result["path"]) / si.REFERENCE_NSUB
