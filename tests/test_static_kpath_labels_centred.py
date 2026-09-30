"""Static SeeK-path labels of the centred lattices name their segments.

Continues test_static_kpath_labels.py for the entries the static lookup
(get_extended_bravais + "_noinv") reaches for the non-centrosymmetric
rhombohedral, body-centred tetragonal and face-centred orthorhombic groups,
the body-centred orthorhombic groups and the C-centred orthorhombic groups
(oS1). Their labels ran on where SeeK-path's path jumps (H_2-H_0, Z-Z_0,
SIGMA_0-U_0, F_2-SIGMA_0, C_0-SIGMA_0, ...), so the BAND title had more edges
than the deck had segments and the band-plot nodes were misnamed. Expected
edges are SeeK-path's paths (seekpath 2.2.2, hpkot/band_path_data), with the
primed copy it appends when time reversal is not used; oS1 is SeeK-path's
oC1 path with its primed copy.
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


OI1 = _split("GAMMA-X X-F_2 SIGMA_0-GAMMA GAMMA-Y_0 U_0-X GAMMA-R R-W W-S S-GAMMA GAMMA-T T-W")

EXPECTED = {
    "hR1_noinv": _primed(_split("GAMMA-T T-H_2 H_0-L L-GAMMA GAMMA-S_0 S_2-F F-GAMMA")),
    "hR2_noinv": _primed(_split("GAMMA-L L-T T-P_0 P_2-GAMMA GAMMA-F")),
    "tI1_noinv": _primed(_split("GAMMA-X X-M M-GAMMA GAMMA-Z Z_0-M X-P P-N N-GAMMA")),
    "tI2_noinv": _primed(_split("GAMMA-X X-P P-N N-GAMMA GAMMA-M M-S S_0-GAMMA X-R G-M")),
    "oF1_noinv": _primed(_split("GAMMA-Y Y-T T-Z Z-GAMMA GAMMA-SIGMA_0 U_0-T Y-C_0 A_0-Z GAMMA-L")),
    "oF3_noinv": _primed(_split("GAMMA-Y Y-C_0 A_0-Z Z-B_0 D_0-T T-G_0 H_0-Y T-GAMMA GAMMA-Z GAMMA-L")),
    "oI1": OI1,
    "oI1_noinv": _primed(OI1),
    "oS1": _primed(_split("GAMMA-Y Y-C_0 SIGMA_0-GAMMA GAMMA-Z Z-A_0 E_0-T T-Y GAMMA-S S-R R-Z Z-T")),
}


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_labels_are_seekpaths_path_for_the_segments(key):
    entry = seekpath_data[key]
    assert _edges(entry["labels"]) == EXPECTED[key]
    assert len(EXPECTED[key]) == len(entry["segments"])


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_each_label_names_one_point(key):
    entry = seekpath_data[key]
    seen = {}
    for (a, b), seg in zip(_edges(entry["labels"]), entry["segments"]):
        for label, xyz in ((a, seg[:3]), (b, seg[3:])):
            xyz = tuple(round(v, 4) for v in xyz)
            assert seen.setdefault(label, xyz) == xyz, (label, seen[label], xyz)
    assert seen["GAMMA"] == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("sg,lattice", [(160, "R"), (161, "R"), (122, "I"), (79, "I"),
                                        (43, "F"), (22, "F"), (71, "I"), (44, "I"),
                                        (63, "C"), (36, "C")])
def test_title_labels_match_the_written_segments(sg, lattice):
    """R3m, R3c, I-42d, I4, Fdd2, F222, Immm, Imm2, Cmcm and Cmc2_1 without
    the seekpath library."""
    segments, _ = get_seekpath_full_kpath(sg, lattice)
    labels = get_seekpath_labels(sg, lattice)
    assert len(_edges(labels)) == len(segments)
