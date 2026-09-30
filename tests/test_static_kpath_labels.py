"""Static SeeK-path labels name the segments they are written with.

Without the seekpath library, ``get_seekpath_full_kpath`` writes an entry's
segments and ``get_seekpath_labels`` its labels, which go on the BAND title
line. The title is read back as a continuous path broken by "|" (each "|"
joins two endpoints into one node, property_extractor._kpath_labels_from_d3),
so the labels must name exactly one edge per segment, in segment order.

Several entries listed a continuous path where SeeK-path's path jumps, so the
title had more edges than the deck had segments and the band-plot nodes were
misnamed; the aP2 entry named (1/2,1/2,0) and (1/2,0,1/2) N and M, while
SeeK-path calls them V and U. Expected edges are SeeK-path's own path files
(seekpath 2.2.2, hpkot/band_path_data/<lattice>/path.txt), with the primed
copy SeeK-path appends when time reversal is not used.
"""
import pytest

from d3_kpoints import get_seekpath_full_kpath, get_seekpath_labels, seekpath_data


def _edges(labels):
    return [(a, b) for a, b in zip(labels, labels[1:]) if "|" not in (a, b)]


def _split(path):
    return [tuple(e.split("-")) for e in path.split()]


def _primed(edges):
    def p(label):
        return label if label == "GAMMA" else label + "'"
    return edges + [(p(a), p(b)) for a, b in edges]


MC1 = _split("GAMMA-C C_2-Y_2 Y_2-GAMMA GAMMA-M_2 M_2-D D_2-A A-GAMMA L_2-GAMMA GAMMA-V_2")
OF2 = _split("GAMMA-T T-Z Z-Y Y-GAMMA GAMMA-LAMBDA_0 Q_0-Z T-G_0 H_0-Y GAMMA-L")
CF2 = _split("GAMMA-X X-U K-GAMMA GAMMA-L L-W W-X")

EXPECTED = {
    "aP3": _split("GAMMA-X Y-GAMMA GAMMA-Z R_2-GAMMA GAMMA-T_2 U_2-GAMMA GAMMA-V_2"),
    "tI1": _split("GAMMA-X X-M M-GAMMA GAMMA-Z Z_0-M X-P P-N N-GAMMA"),
    "tI2": _split("GAMMA-X X-P P-N N-GAMMA GAMMA-M M-S S_0-GAMMA X-R G-M"),
    "oF1": _split("GAMMA-Y Y-T T-Z Z-GAMMA GAMMA-SIGMA_0 U_0-T Y-C_0 A_0-Z GAMMA-L"),
    "oF3": _split("GAMMA-Y Y-C_0 A_0-Z Z-B_0 D_0-T T-G_0 H_0-Y T-GAMMA GAMMA-Z GAMMA-L"),
    "oF2": _primed(OF2),
    "mS1": _primed(MC1),
    "cF2_noinv": _primed(CF2),
}


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_labels_are_seekpaths_path_for_the_segments(key):
    entry = seekpath_data[key]
    assert _edges(entry["labels"]) == EXPECTED[key]
    assert len(EXPECTED[key]) == len(entry["segments"])


@pytest.mark.parametrize("key", sorted(EXPECTED) + ["aP2"])
def test_each_label_names_one_point(key):
    entry = seekpath_data[key]
    seen = {}
    for (a, b), seg in zip(_edges(entry["labels"]), entry["segments"]):
        for label, xyz in ((a, seg[:3]), (b, seg[3:])):
            xyz = tuple(round(v, 4) for v in xyz)
            assert seen.setdefault(label, xyz) == xyz, (label, seen[label], xyz)
    assert seen["GAMMA"] == (0.0, 0.0, 0.0)


def test_ap2_uses_seekpath_names():
    """SeeK-path aP2: V = (1/2, 1/2, 0), U = (1/2, 0, 1/2)."""
    entry = seekpath_data["aP2"]
    names = {}
    for (a, b), seg in zip(_edges(entry["labels"]), entry["segments"]):
        names[tuple(seg[:3])] = a
        names[tuple(seg[3:])] = b
    assert names[(0.5, 0.5, 0.0)] == "V"
    assert names[(0.5, 0.0, 0.5)] == "U"
    assert names[(0.5, 0.5, 0.5)] == "R"


@pytest.mark.parametrize("sg,lattice", [(2, "P"), (87, "I"), (139, "I"), (69, "F"),
                                        (216, "F"), (5, "C"), (8, "C")])
def test_title_labels_match_the_written_segments(sg, lattice):
    """P-1, I4/m, I4/mmm, Fmmm, F-43m, C2 and Cm without the seekpath library."""
    segments, _ = get_seekpath_full_kpath(sg, lattice)
    labels = get_seekpath_labels(sg, lattice)
    assert len(_edges(labels)) == len(segments)
