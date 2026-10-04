"""The static SeeK-path table defines each lattice once.

``seekpath_data`` in d3_kpoints is a dict literal, and a key written twice
silently keeps its last definition: the earlier copies were never read, but
they looked like the table's data (an older oP1 with its segments' labels
in another order, tI1/tI2 labels that ran on over SeeK-path's jumps, ...).
Those copies are removed.

aP1, mS1 and oS1 were each written twice as well, the later copy being the
path without inversion (primed points), so it won: on the static route (no
seekpath library) C2/m, Cmcm and every other centrosymmetric C-centred group
got the primed Cc / Cmc2_1 path, 18 and 22 segments, where the library route
writes SeeK-path's 9 and 11. Those copies are now mS1_noinv and oS1_noinv,
which the lookup takes for groups without inversion, as for every other
lattice; the second aP1 (labels that named none of its segments) is gone.
"""
import ast
import contextlib
import io
from pathlib import Path

import pytest

import d3_kpoints

SOURCE = Path(d3_kpoints.__file__).read_text()


def _definitions():
    for node in ast.parse(SOURCE).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "seekpath_data":
            keys = [ast.literal_eval(k) for k in node.value.keys]
            values = [ast.get_source_segment(SOURCE, v) for v in node.value.values]
            return list(zip(keys, values))
    raise AssertionError("seekpath_data not found")


def test_no_lattice_is_defined_twice():
    keys = [k for k, _ in _definitions()]
    repeated = {k for k in keys if keys.count(k) > 1}
    assert repeated == set()


def _static(monkeypatch, sg, lattice):
    monkeypatch.setattr(d3_kpoints, "SEEKPATH_LIBRARY_AVAILABLE", False)
    with contextlib.redirect_stdout(io.StringIO()):
        return d3_kpoints.get_seekpath_full_kpath(sg, lattice)


@pytest.mark.parametrize("sg,lattice,key,n", [
    (12, "C", "mS1", 9),          # C2/m
    (15, "C", "mS1", 9),          # C2/c
    (63, "C", "oS1", 11),         # Cmcm
    (9, "C", "mS1_noinv", 18),    # Cc keeps its primed copy
    (36, "C", "oS1_noinv", 22),   # Cmc2_1 keeps its primed copy
])
def test_centered_groups_take_the_path_for_their_inversion(monkeypatch, sg, lattice, key, n):
    segments, info = _static(monkeypatch, sg, lattice)
    assert info["lookup_key"] == key
    assert len(segments) == n
    primed = [lab for lab in d3_kpoints.get_seekpath_labels(sg, lattice) if lab.endswith("'")]
    assert bool(primed) == key.endswith("_noinv")


def test_centrosymmetric_static_path_is_the_library_path(monkeypatch):
    """C2/m from an output: the static route writes the library route's deck."""
    pytest.importorskip("seekpath")
    si = pytest.importorskip("seekpath_interface")
    import numpy as np
    a, b, c, beta = 11.0, 4.0, 6.0, 105.0
    br = np.radians(beta)
    conv = np.array([[a, 0, 0], [0, b, 0], [c * np.cos(br), 0, c * np.sin(br)]])
    cell = np.array([(conv[0] - conv[1]) / 2, (conv[0] + conv[1]) / 2, conv[2]])
    raw = si.get_seekpath_bandpath(cell, [[0.1, 0.1, 0.3]], [14])
    assert raw["bravais_lattice_extended"] == "mC1"
    edges = [tuple(e) for e in raw["path"]]
    static = d3_kpoints.seekpath_data["mS1"]
    labels = [lab for lab in static["labels"]]
    static_edges = [(x, y) for x, y in zip(labels, labels[1:]) if "|" not in (x, y)]
    assert static_edges == edges
