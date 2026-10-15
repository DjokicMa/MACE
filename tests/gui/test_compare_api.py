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


# ---------------------------------------------------------------- molecules and mixed periodicity
EC = "EC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized"


def _ec():
    d = H.REPO_ROOT / "test" / "OPT"
    if not (d / f"{EC}.out").is_file() or not (d / f"{EC}.cif").is_file():
        pytest.skip("test/ corpus not present")
    return structures.read_structure(d / f"{EC}.cif"), structures.read_structure(d / f"{EC}.out")


def test_molecule_cif_against_its_output_does_not_move():
    """The CIF (a 40 Å box, periodic) and the MOLECULE output (its bounding box, not
    periodic) hold the same geometry: Cartesian, centered and rotated onto each
    other, nothing moves and no cell is compared."""
    cif, out = _ec()
    assert all(cif["pbc"]) and not any(out["pbc"])
    for a, b in ((cif, out), (out, cif)):
        c = structures.compare_payloads(a, b)
        assert c["max_displacement"] < 1e-3, c["max_displacement"]
        assert c["cell"] == []
        assert c["bonds"] and all(abs(r["delta"]) < 1e-3 for r in c["bonds"])


def test_molecule_one_atom_moved_is_the_only_displacement():
    _, out = _ec()
    moved = {k: (list(v) if k != "cart" else [list(r) for r in v]) for k, v in out.items() if k in ("cell", "frac", "cart", "numbers", "pbc")}
    moved["cart"][3][0] += 0.5
    for b in (moved, {**moved, "cart": [[x + 7.0, y - 3.0, z] for x, y, z in moved["cart"]]}):
        c = structures.compare_payloads(out, b, match="index")
        lengths = [d["length"] for d in c["displacements"]]
        assert abs(lengths[3] - 0.5) < 0.02, lengths
        assert max(lengths[:3] + lengths[4:]) < 0.02, lengths


def _slab_and_cif(shift_atom=0):
    """A small 2D slab (periodic along a and b) and its 3D CIF (a 20 Å c
    box) with one atom wrapped by +a: the same structure."""
    cell = [[4.0, 0.0, 0.0], [1.0, 3.8, 0.0], [0.0, 0.0, 20.0]]
    frac = [[0.10, 0.10, 0.50], [0.60, 0.20, 0.55], [0.30, 0.70, 0.45], [0.85, 0.55, 0.52], [0.45, 0.45, 0.60]]
    cart = (np.asarray(frac) @ np.asarray(cell)).tolist()
    nums = [12, 8, 12, 8, 8]
    slab = {"cell": cell, "frac": None, "cart": cart, "numbers": nums, "pbc": [True, True, False]}
    moved = [list(r) for r in cart]
    moved[shift_atom] = [x + y for x, y in zip(moved[shift_atom], cell[0])]
    cif = {"cell": cell, "frac": None, "cart": moved, "numbers": nums, "pbc": [True, True, True]}
    return slab, cif


def test_slab_against_its_cif_with_a_wrapped_atom_does_not_move():
    """Mixed periodicity (slab vs 3D CIF): an atom wrapped by +a along an axis
    periodic in both is taken at its nearest image, not a 4 Å displacement."""
    for k in range(5):
        slab, cif = _slab_and_cif(k)
        for a, b in ((slab, cif), (cif, slab)):
            c = structures.compare(a, b)
            assert c["max_displacement"] < 1e-6, (k, c["max_displacement"])
            assert c["rms_displacement"] < 1e-6


def test_slab_against_its_cif_keeps_a_real_move():
    slab, cif = _slab_and_cif(2)
    cif["cart"][4][2] += 0.3                     # along the non-periodic c
    cif["cart"][1][0] += 0.2
    c = structures.compare(slab, cif)
    lengths = [d["length"] for d in c["displacements"]]
    assert max(lengths[i] for i in (0, 2, 3)) < 0.1, lengths
    assert lengths[4] > 0.15 and lengths[1] > 0.1, lengths
