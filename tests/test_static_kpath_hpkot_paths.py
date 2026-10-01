"""Static SeeK-path entries write SeeK-path's own path.

Without the seekpath library, ``get_seekpath_full_kpath`` writes the segments
of a ``seekpath_data`` entry and titles the deck "SeeKPath". The entries
checked here held a different path (Setyawan-Curtarolo's, or one that ran on
through SeeK-path's jumps). Expected edges and points are SeeK-path 2.2.2's
``hpkot/band_path_data/<lattice>/path.txt`` and ``points.txt`` (HPKOT,
Hinuma et al., Comp. Mat. Sci. 128, 140 (2017)); a non-centrosymmetric
entry adds the primed copy SeeK-path appends without time reversal.
Parametric points (lattice-dependent) are checked for their name only.
"""
from fractions import Fraction

import pytest

from d3_kpoints import seekpath_data


def _edges(labels):
    return [(a, b) for a, b in zip(labels, labels[1:]) if "|" not in (a, b)]


def _split(path):
    return [tuple(e.split("-")) for e in path.split()]


def _primed(edges):
    def p(label):
        return label if label == "GAMMA" else label + "'"
    return edges + [(p(a), p(b)) for a, b in edges]


def _pts(text):
    out = {}
    for line in text.strip().splitlines():
        name, *xyz = line.split()
        out[name] = tuple(Fraction(v) for v in xyz)
    return out


CF_POINTS = _pts("""
GAMMA 0 0 0
X 1/2 0 1/2
L 1/2 1/2 1/2
W 1/2 1/4 3/4
W_2 3/4 1/4 1/2
K 3/8 3/8 3/4
U 5/8 1/4 5/8
""")

HR1_POINTS = _pts("""
GAMMA 0 0 0
T 1/2 1/2 1/2
L 1/2 0 0
F 1/2 0 1/2
""")

HR2_POINTS = _pts("""
GAMMA 0 0 0
T 1/2 -1/2 1/2
L 1/2 0 0
F 1/2 -1/2 0
""")

OA1_POINTS = _pts("""
GAMMA 0 0 0
Y -1/2 1/2 0
T -1/2 1/2 1/2
Z 0 0 1/2
S 0 1/2 0
R 0 1/2 1/2
""")

OA2_POINTS = _pts("""
GAMMA 0 0 0
Y 1/2 1/2 0
T 1/2 1/2 1/2
Z 0 0 1/2
S 0 1/2 0
R 0 1/2 1/2
""")

CF1 = _split("GAMMA-X X-U K-GAMMA GAMMA-L L-W W-X X-W_2")

EXPECTED = {
    "cF1": (CF1, CF_POINTS),
    "cF1_noinv": (_primed(CF1), CF_POINTS),
    "cF2": (_split("GAMMA-X X-U K-GAMMA GAMMA-L L-W W-X"), CF_POINTS),
    # R-3, R-3m, R-3c: the entries held Setyawan-Curtarolo's rhombohedral
    # paths (B, B1, Q, P1, X, Z, ...), names SeeK-path does not use.
    "hR1": (_split("GAMMA-T T-H_2 H_0-L L-GAMMA GAMMA-S_0 S_2-F F-GAMMA"), HR1_POINTS),
    "hR2": (_split("GAMMA-L L-T T-P_0 P_2-GAMMA GAMMA-F"), HR2_POINTS),
    # Amm2, Aem2, Ama2, Aea2: the segments were SeeK-path's, but the title
    # ran on through the jumps C_0|SIGMA_0 (F_0|DELTA_0) and A_0|E_0
    # (B_0|G_0), naming more edges than the deck has segments.
    "oA1_noinv": (_primed(_split(
        "GAMMA-Y Y-C_0 SIGMA_0-GAMMA GAMMA-Z Z-A_0 E_0-T T-Y GAMMA-S S-R R-Z Z-T")), OA1_POINTS),
    "oA2_noinv": (_primed(_split(
        "GAMMA-Y Y-F_0 DELTA_0-GAMMA GAMMA-Z Z-B_0 G_0-T T-Y GAMMA-S S-R R-Z Z-T")), OA2_POINTS),
}


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_static_entry_is_seekpaths_path(key):
    edges, points = EXPECTED[key]
    entry = seekpath_data[key]
    assert _edges(entry["labels"]) == edges
    assert len(entry["segments"]) == len(edges)
    for (a, b), seg in zip(edges, entry["segments"]):
        for label, xyz in ((a, seg[:3]), (b, seg[3:])):
            exact = points.get(label.rstrip("'"))
            if exact is None:
                continue  # parametric point
            sign = -1 if label.endswith("'") else 1
            assert all(abs(v - sign * float(e)) < 1e-12 for v, e in zip(xyz, exact)), (key, label, xyz)


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_expected_paths_are_seekpaths_files(key):
    """The tables above are SeeK-path's own (where seekpath is installed)."""
    pytest.importorskip("seekpath")
    from seekpath.hpkot.tools import get_path_data
    _, points_def, path = get_path_data(key[:3])
    edges, points = EXPECTED[key]
    assert edges[:len(path)] == [tuple(e) for e in path]
    for name, exact in points.items():
        assert tuple(Fraction(v) for v in points_def[name]) == exact, name
