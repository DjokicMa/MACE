"""Before/after comparison through the server (plan T10, B7).

Position matching recovers a shuffled structure exactly (100 random
permutations of a 2x2x2 MgO), mismatched counts and species are refused with
the reason, the symmetry row names both ends, and Compare with Input finds
the .d12 beside an output. Fixtures are written by ase into a temp dir."""
import sys
import time

import numpy as np
import pytest

import guiharness as H
import p4util as U

sys.path.insert(0, str(H.REPO_ROOT))
ase = pytest.importorskip("ase")
from ase.io import read, write  # noqa: E402

from mace.gui import structures  # noqa: E402


def payload(atoms):
    return {"cell": atoms.cell.array.tolist(), "frac": atoms.get_scaled_positions(wrap=False).tolist(),
            "cart": atoms.positions.tolist(), "numbers": atoms.numbers.tolist(), "pbc": [True, True, True]}


@pytest.fixture(scope="module")
def mgo():
    return read(str(H.EXAMPLES / "structures" / "MgO.cif")).repeat((2, 2, 2))


def test_position_matching_recovers_100_shuffles(mgo):
    rng = np.random.default_rng(7)
    a = payload(mgo)
    n = len(mgo)
    for _ in range(100):
        perm = rng.permutation(n)
        b = mgo[perm]
        # a small rattle and a lattice translation of some atoms: minimum image still pairs them
        b.positions += rng.normal(0, 0.01, (n, 3))
        k = rng.integers(0, n)
        b.positions[k] += b.cell.array[0]
        order, worst, how = structures.match_atoms(a, payload(b))
        assert how == "position"
        assert [int(perm[j]) for j in order] == list(range(n))     # every atom back on its own
        assert worst < 0.06
    c = structures.compare_payloads(a, payload(mgo[rng.permutation(n)]))
    assert c["match"] == "position" and c["max_mismatch"] < 1e-6 and c["max_displacement"] < 1e-6


def test_index_order_is_kept_when_species_agree(mgo):
    a = payload(mgo)
    b = mgo.copy()
    b.positions[0] += [0.1, 0, 0]
    c = structures.compare_payloads(a, payload(b))
    assert c["match"] == "index" and c["order"] is None and abs(c["max_displacement"] - 0.1) < 1e-6


def test_count_and_species_mismatch_are_refused(mgo):
    small = read(str(H.EXAMPLES / "structures" / "MgO.cif"))
    with pytest.raises(structures.MatchError, match=r"^8 atoms vs 64: compare a supercell with Build › Supercell first$"):
        structures.match_atoms(payload(small), payload(mgo))
    lif = small.copy()
    lif.numbers = [3 if z == 12 else 9 for z in lif.numbers]
    with pytest.raises(structures.MatchError, match="need the same atoms: Mg4 O4 vs F4 Li4"):
        structures.match_atoms(payload(small), payload(lif))


@pytest.fixture(scope="module")
def served(tmp_path_factory, mgo):
    tmp = tmp_path_factory.mktemp("cmp")
    root = H.examples_copy(tmp)
    (root / "cmp").mkdir()
    rng = np.random.default_rng(3)
    write(str(root / "cmp" / "big.cif"), mgo)
    write(str(root / "cmp" / "shuffled.cif"), mgo[rng.permutation(len(mgo))])
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_NO_WARMUP="1")
    yield root, base
    proc.terminate()
    proc.wait(timeout=15)


def test_compare_route_matches_refuses_and_names_symmetry(served):
    root, base = served
    st, _, r = U.req(base, "api/compare?a=cmp/big.cif&b=cmp/shuffled.cif")
    assert st == 200, r
    c = r["comparison"]
    assert c["match"] == "position" and c["max_mismatch"] < 1e-5
    assert c["symmetry_change"] == {"before": "Fm-3m (225)", "after": "Fm-3m (225)"}
    # the after payload comes back in the matched order: atom i is atom i
    assert r["b"]["numbers"] == r["a"]["numbers"]
    st, _, r = U.req(base, "api/compare?a=structures/MgO.cif&b=cmp/big.cif")
    assert st == 400 and r["error"] == "8 atoms vs 64: compare a supercell with Build › Supercell first"


def test_compare_with_input_uses_the_d12(served):
    root, base = served
    st, _, r = U.req(base, "api/compare/input?path=outputs/Ag2Br3_opt_unfinished.out")
    assert st == 200, r
    inp = r["input"]
    assert (inp["kind"], inp["path"]) == ("d12", "outputs/Ag2Br3_opt_unfinished.d12"), r
    # the deck is the 30-atom centered cell; CRYSTAL ran the 10-atom primitive one, and
    # the output's first frame is checked to be that deck (formula, volume per atom, group)
    assert inp.get("via") == "frame0" and "centered cell (30 atoms)" in inp["note"]
    assert len(r["a"]["numbers"]) == len(r["b"]["numbers"]) and "symmetry_change" in r["comparison"]
    # an output with no deck beside it falls back to its first frame
    (root / "outputs" / "Si_alone.out").write_bytes((root / "outputs" / "Si_sp.out").read_bytes())
    st, _, r = U.req(base, "api/compare/input?path=outputs/Si_alone.out")
    assert st == 200 and r["input"]["kind"] == "frame0", r


def test_trajectory_compare_symmetry_row(served):
    root, base = served
    st, _, r = U.req(base, "api/trajectory/compare?path=outputs/Ag2Br3_opt_unfinished.out&i=0&j=-1&sym=1")
    assert st == 200 and set(r["symmetry_change"]) == {"before", "after"}
    assert r["match"] == "index"


def test_same_species_order_but_shuffled_atoms_are_matched_by_position(mgo):
    """Two files that list Mg then O can still list the Mg atoms in another order:
    pairing them by index would report half-cell "displacements"."""
    a = payload(mgo)
    nmg = int((mgo.numbers == 12).sum())
    perm = list(range(len(mgo)))
    perm[0], perm[1] = perm[1], perm[0]                 # two Mg swapped, the species sequence unchanged
    b = mgo[perm]
    assert list(b.numbers) == list(mgo.numbers)
    c = structures.compare_payloads(a, payload(b))
    assert c["match"] == "position" and c["max_displacement"] < 1e-6, (c["match"], c["max_displacement"])
    assert nmg > 1


def test_position_matching_works_across_cartesian_orientations(mgo):
    """A file in another Cartesian frame (the same fractional positions) pairs atom for atom."""
    a = payload(mgo)
    b = mgo.copy()
    b.rotate(37, "z", rotate_cell=True)
    rng = np.random.default_rng(11)
    perm = rng.permutation(len(b))
    pb = payload(b[perm])
    order, worst, how = structures.match_atoms(a, pb)
    assert how == "position" and worst < 1e-6
    assert [int(perm[j]) for j in order] == list(range(len(b)))


def test_large_same_order_structures_skip_position_matching(monkeypatch, mgo):
    monkeypatch.setattr(structures, "MATCH_MAX_PER_ELEMENT", 4)
    a = payload(mgo)
    order, worst, how = structures.match_atoms(a, payload(mgo))
    assert how == "index"
    with pytest.raises(structures.MatchError, match="too many atoms of one element"):
        structures.match_atoms(a, payload(mgo[np.random.default_rng(1).permutation(len(mgo))]))
