"""K-path tables in d12_constants, checked against their sources.

- SPACEGROUP_TO_PATH gives F- and I-centred cubic groups the fcc and bcc paths.
- Every label on every HIGH_SYMMETRY_PATHS path has coordinates
  (test_d12_constants_tables.py::test_every_path_label_has_coordinates).
"""
import sys

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))

import d12_constants as C  # noqa: E402

F_CUBIC = {196, 202, 203, 209, 210, 216, 219, 225, 226, 227, 228}
I_CUBIC = {197, 199, 204, 206, 211, 214, 217, 220, 229, 230}


@pytest.mark.parametrize("n", range(195, 231))
def test_cubic_groups_get_the_path_of_their_centring(n):
    expected = "cubic_fc" if n in F_CUBIC else "cubic_bc" if n in I_CUBIC else "cubic_simple"
    assert C.SPACEGROUP_TO_PATH[n] == expected
    assert C.SPACEGROUP_SYMBOLS[n][0] == {"cubic_fc": "F", "cubic_bc": "I",
                                          "cubic_simple": "P"}[expected]


def test_monoclinic_points_are_the_manuals_table_14_1():
    """CRYSTAL23 manual Table 14.1 (p. 311), P monoclinic."""
    coords = C.HIGH_SYMMETRY_PATHS["monoclinic"]["coordinates"]
    assert coords == {"G": [0.0, 0.0, 0.0], "A": [0.5, -0.5, 0.0], "B": [0.5, 0.0, 0.0],
                      "C": [0.0, 0.5, 0.5], "D": [0.5, 0.0, 0.5], "E": [0.5, -0.5, 0.5],
                      "Y": [0.0, 0.5, 0.0], "Z": [0.0, 0.0, 0.5]}


@pytest.mark.parametrize("key,d3", [("monoclinic", "monoclinic_simple"),
                                    ("triclinic", "triclinic")])
def test_low_symmetry_paths_match_the_band_path_code(key, d3):
    """The same path and points as Crystal_d3/d3_kpoints.py uses for BAND."""
    sys.path.insert(0, str(REPO_ROOT / "Crystal_d3"))
    import d3_kpoints
    ours = C.HIGH_SYMMETRY_PATHS[key]
    assert ["GAMMA" if p == "G" else p for p in ours["labels"]] == d3_kpoints.BAND_PATHS[d3]
    theirs = d3_kpoints.KPOINT_COORDINATES[d3]
    assert {("GAMMA" if k == "G" else k): v for k, v in ours["coordinates"].items()} == theirs
