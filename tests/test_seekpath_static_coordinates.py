"""The static SeeK-path table (no seekpath library) puts its points where
SeeK-path does.

Coordinates are fractions of the primitive reciprocal lattice vectors - the
convention of SeeK-path (Hinuma et al., Comp. Mat. Sci. 128, 140 (2017)) and
of CRYSTAL's BAND input (CRYSTAL23 manual, Tables 14.1-14.2, pp. 311-312):
simple cubic X = (0, 1/2, 0), M = (1/2, 1/2, 0), R = (1/2, 1/2, 1/2).

``seekpath_data`` defines several keys twice, and the definitions that win
had every coordinate multiplied by 2 (cubic-I by 4, hexagonal by 6): simple
cubic X at (0, 1, 0), a reciprocal lattice vector - the same point as Gamma.
A band path through such points runs Gamma -> Gamma.
"""
import sys
from pathlib import Path

import pytest

_D3 = str(Path(__file__).resolve().parent.parent / "Crystal_d3")
if _D3 not in sys.path:
    sys.path.insert(0, _D3)

from d3_kpoints import get_seekpath_full_kpath, seekpath_data  # noqa: E402


def _points(entry):
    """{label: {coordinates}} from an entry's labels and segments.

    Two label layouts are used: one pair per segment, or a continuous path
    broken by "|".
    """
    segs = entry["segments"]
    labels = entry.get("labels") or []
    flat = [l for l in labels if l != "|"]
    if len(flat) == 2 * len(segs):
        pairs = [(flat[2 * i], flat[2 * i + 1]) for i in range(len(segs))]
    else:
        pairs, run = [], []
        for label in labels + ["|"]:
            if label == "|":
                pairs += list(zip(run[:-1], run[1:]))
                run = []
            else:
                run.append(label)
    if len(pairs) != len(segs):
        return None
    points = {}
    for (a, b), seg in zip(pairs, segs):
        points.setdefault(a, set()).add(tuple(seg[:3]))
        points.setdefault(b, set()).add(tuple(seg[3:]))
    return points


def test_simple_cubic_points_are_hpkot():
    for key in ("cP1", "cP2"):
        pts = _points(seekpath_data[key])
        assert pts["X"] == {(0.0, 0.5, 0.0)}, key
        assert pts["M"] == {(0.5, 0.5, 0.0)}, key
        assert pts["R"] == {(0.5, 0.5, 0.5)}, key
    assert _points(seekpath_data["cP1"])["X_1"] == {(0.5, 0.0, 0.0)}


def test_simple_cubic_fallback_path_leaves_gamma():
    """What a Pm-3m .d3/.d12 band path gets without the seekpath library."""
    segments, info = get_seekpath_full_kpath(221, "P")
    assert info["lookup_key"] in ("cP1", "cP2")
    assert segments[0] == [0.0, 0.0, 0.0, 0.0, 0.5, 0.0]


@pytest.mark.parametrize("key", sorted(seekpath_data))
def test_no_path_end_other_than_gamma_is_a_reciprocal_lattice_vector(key):
    for seg in seekpath_data[key]["segments"]:
        for end in (seg[:3], seg[3:]):
            if any(end):
                assert not all(abs(x - round(x)) < 1e-9 for x in end), (key, seg)


@pytest.mark.parametrize("key, point, coords", [
    ("cI1", "H", (0.5, -0.5, 0.5)),
    ("cI1", "P", (0.25, 0.25, 0.25)),
    ("cI1", "N", (0.0, 0.0, 0.5)),
    ("hP2", "M", (0.5, 0.0, 0.0)),
    ("hP2", "K", (1 / 3, 1 / 3, 0.0)),
    ("hP2", "A", (0.0, 0.0, 0.5)),
    ("tP1", "A", (0.5, 0.5, 0.5)),
    ("oP1", "R", (0.5, 0.5, 0.5)),
])
def test_manual_table_14_points(key, point, coords):
    """CRYSTAL23 manual Tables 14.1-14.2 (pp. 311-312), which SeeK-path
    agrees with for these lattices."""
    got = _points(seekpath_data[key])[point]
    assert any(all(abs(a - b) < 1e-9 for a, b in zip(c, coords)) for c in got), got


def _label_edges(labels):
    """[(start, end), ...] of a "|"-broken continuous label path."""
    edges = []
    for prev, cur in zip(labels, labels[1:]):
        if "|" not in (prev, cur):
            edges.append((prev, cur))
    return edges


def test_op1_noinv_labels_name_its_segments_in_order():
    """oP1_noinv listed Y-T before X-U in its labels while its segments (and
    SeeK-path's path) run X-U, then Y-T - for both the plain and the primed
    half - so those two segments carried each other's labels."""
    entry = seekpath_data["oP1_noinv"]
    edges = _label_edges(entry["labels"])
    assert len(edges) == len(entry["segments"])
    assert edges[9:12] == [("X", "U"), ("Y", "T"), ("S", "R")]
    assert edges[21:24] == [("X'", "U'"), ("Y'", "T'"), ("S'", "R'")]
    # SeeK-path oP1: X = (1/2, 0, 0), U = (1/2, 0, 1/2), Y = (0, 1/2, 0),
    # T = (0, 1/2, 1/2); the primed points are their negatives.
    assert entry["segments"][9] == [0.5, 0.0, 0.0, 0.5, 0.0, 0.5]
    assert entry["segments"][10] == [0.0, 0.5, 0.0, 0.0, 0.5, 0.5]
    assert entry["segments"][21] == [-0.5, 0.0, 0.0, -0.5, 0.0, -0.5]
    assert entry["segments"][22] == [0.0, -0.5, 0.0, 0.0, -0.5, -0.5]
