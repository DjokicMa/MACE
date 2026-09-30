"""The static SeeK-path table defines each lattice once.

``seekpath_data`` in d3_kpoints is a dict literal, and a key written twice
silently keeps its last definition: the earlier copies were never read, but
they looked like the table's data (an older oP1 with its segments' labels
in another order, tI1/tI2 labels that ran on over SeeK-path's jumps, ...).
Those copies are removed.

Three keys are still written twice, and the two definitions are different
paths, so neither copy is a stale duplicate of the other: mS1 and oS1 first
hold the centrosymmetric mC1 and oC1 paths and then the paths with primed
points for Cc and Cmc2_1, and aP1 first holds a centrosymmetric default and
then the P1 path with primed points. The later definitions are the ones in
use; making the first ones reachable would change the decks written for
C2/m, Cmcm and the other centrosymmetric C-centred groups.
"""
import ast
from pathlib import Path

import d3_kpoints

SOURCE = Path(d3_kpoints.__file__).read_text()
DIFFERENT_PATHS_UNDER_ONE_KEY = {"aP1", "mS1", "oS1"}


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
    assert repeated == DIFFERENT_PATHS_UNDER_ONE_KEY


def test_the_remaining_repeats_are_different_paths():
    by_key = {}
    for key, value in _definitions():
        by_key.setdefault(key, []).append(value)
    for key in DIFFERENT_PATHS_UNDER_ONE_KEY:
        first, last = by_key[key]
        assert "without inversion" in last
        assert "without inversion" not in first
        # the definition in use is the last one
        assert d3_kpoints.seekpath_data[key] == eval(last)  # noqa: S307
