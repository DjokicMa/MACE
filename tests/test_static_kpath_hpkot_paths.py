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

CF1 = _split("GAMMA-X X-U K-GAMMA GAMMA-L L-W W-X X-W_2")

EXPECTED = {
    "cF1": (CF1, CF_POINTS),
    "cF1_noinv": (_primed(CF1), CF_POINTS),
    "cF2": (_split("GAMMA-X X-U K-GAMMA GAMMA-L L-W W-X"), CF_POINTS),
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
