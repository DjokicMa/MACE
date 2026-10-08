"""`mace gui` structure editor: edits, symmetry-aware edits, undo/redo, the
CIF -> cif2d12 deck round trip, and the editor API through the real server."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO_ROOT

pytest.importorskip("ase")
pytest.importorskip("spglib")

from mace.gui import editor as E  # noqa: E402
from mace.gui import structures as S  # noqa: E402

EXAMPLES = REPO_ROOT / "examples" / "gui"
CONFIG = json.loads((REPO_ROOT / "Crystal_d12" / "example_configs" /
                     "standard_dft_opt.json").read_text())


def _mgo():
    return E.EditSession(E.atoms_from_payload(S.read_structure(EXAMPLES / "structures" / "MgO.cif")),
                         "MgO")


def test_p1_add_delete_move_replace_with_index_maps_and_undo_redo():
    s = _mgo()
    r = s.apply("add", {"element": "Li", "frac": [0.25, 0.25, 0.25]})
    assert len(r["numbers"]) == 9 and r["index_map"] == list(range(8)) and r["can_undo"]
    r = s.apply("delete", {"indices": [0]})
    assert len(r["numbers"]) == 8 and r["index_map"][0] is None and r["index_map"][1] == 0
    p0 = np.array(s.atoms.positions[0])
    s.apply("move", {"indices": [0], "delta": [0.1, 0, 0]})
    assert s.atoms.positions[0] == pytest.approx(p0 + [0.1, 0, 0])
    r = s.apply("replace", {"indices": [0], "element": "Ca"})
    assert r["symbols"][0] == "Ca"
    for _ in range(4):
        r = s.apply("undo", {})
    assert r["reduced_formula"] == "MgO" and not r["can_undo"] and r["can_redo"]
    r = s.apply("redo", {})
    assert len(r["numbers"]) == 9
    with pytest.raises(E.EditError):
        s.apply("add", {"element": "Q", "frac": [0, 0, 0.5]})
    with pytest.raises(E.EditError, match="already sits"):
        s.apply("add", {"element": "O", "frac": [0.5, 0.5, 0.5]})


def test_ghost_and_dummy_atoms_are_refused():
    with pytest.raises(E.EditError, match="ghost"):
        E._number(0)
    assert E._number(47) == 47 and E._number("ag") == 47


def test_symmetry_aware_edits_act_on_whole_orbits():
    s = _mgo()
    r = s.apply("set_locked", {"locked": True})
    assert r["locked"] and r["locked_group"] == 192
    r = s.apply("add", {"element": "Li", "frac": [0.25, 0.25, 0.25]})     # 8c
    assert r["symbols"].count("Li") == 8 and r["symmetry"]["number"] == 225
    o = r["symbols"].index("O")
    r = s.apply("replace", {"indices": [o], "element": "S"})
    assert r["symbols"].count("S") == 4 and "O" not in r["symbols"]
    r = s.apply("delete", {"indices": [r["symbols"].index("Li")]})
    assert "Li" not in r["symbols"] and r["reduced_formula"] == "MgS"
    # moving Mg off 4a would make it a 32f-type orbit: two-step confirm
    with pytest.raises(E.NeedsConfirm) as info:
        s.apply("move", {"indices": [0], "frac_delta": [0.02, 0.02, 0.02]})
    assert info.value.details == {"old_multiplicity": 4, "new_multiplicity": 32}
    r = s.apply("move", {"indices": [0], "frac_delta": [0.02, 0.02, 0.02], "confirm": True})
    assert r["symbols"].count("Mg") == 32 and r["symmetry"]["number"] == 225


def test_locked_move_keeps_the_group_when_the_orbit_does_not_grow():
    from ase.spacegroup import crystal
    atoms = crystal(["Ti", "O"], [(0, 0, 0), (0.3, 0.3, 0)], spacegroup=136,
                    cellpar=[4.59, 4.59, 2.96, 90, 90, 90])      # rutile, O on 4f (x, x, 0)
    s = E.EditSession(atoms, "rutile")
    s.apply("set_locked", {"locked": True})
    o = int(np.where(s.atoms.numbers == 8)[0][0])
    r = s.apply("move", {"indices": [o], "frac_delta": [0.005, 0.005, 0]})
    assert r["symmetry"]["number"] == 136 and len(r["numbers"]) == 6
    o_frac = [f for f, z in zip(r["frac"], r["numbers"]) if z == 8]
    assert o_frac[0][0] % 1 == pytest.approx(0.305, abs=1e-6)   # still on (x, x, 0)


def test_cell_supercell_primitive_symmetrize_and_recapture():
    s = _mgo()
    r = s.apply("set_cell", {"cellpar": [4.4, 4.4, 4.4, 90, 90, 90], "scale_atoms": True})
    assert r["lattice"]["a"] == pytest.approx(4.4) and r["symmetry"]["number"] == 225
    s.apply("set_locked", {"locked": True})
    r = s.apply("primitive", {})
    assert len(r["numbers"]) == 2 and "re-captured" in r["op_summary"]
    r = s.apply("supercell", {"n": [2, 2, 2]})
    assert len(r["numbers"]) == 16 and r["index_map"] is None
    s.apply("set_locked", {"locked": False})
    s.atoms.positions[0] += [0.004, 0, 0]
    s.apply("set_symprec", {"symprec": 0.01})
    r = s.apply("symmetrize", {})
    assert r["symmetry"]["number"] == 225 and len(r["numbers"]) == 8


def test_new_cell_from_space_group_and_clear_overlap_error():
    atoms = E.new_cell([5.431] * 3 + [90] * 3, [{"element": "Si", "frac": [0, 0, 0]}], 227)
    assert len(atoms) == 8
    with pytest.raises(E.EditError, match="space group 225"):
        E.new_cell([4.2] * 3 + [90] * 3, [{"element": "Mg", "frac": [0, 0, 0]},
                                          {"element": "O", "frac": [0.5, 0.5, 0]}], 225)
    p1 = E.new_cell([3, 3, 3, 90, 90, 90], [{"element": "Na", "frac": [0, 0, 0]}])
    assert len(p1) == 1


def test_busy_session_refuses_edits():
    s = _mgo()
    s.busy = True
    with pytest.raises(E.Busy):
        s.apply("delete", {"indices": [0]})


# --- the CIF round trip: saved CIF -> cif2d12 deck = the structure edited ------

def _fingerprint(atoms, rmax=4.5):
    from ase.neighborlist import neighbor_list
    i, j, d = neighbor_list("ijd", atoms, rmax)
    sym = atoms.get_chemical_symbols()
    out = {}
    for a, b, x in zip(i, j, d):
        out.setdefault(tuple(sorted((sym[a], sym[b]))), []).append(round(float(x), 3))
    n = len(atoms)
    return {k: sorted(v) for k, v in out.items()}, atoms.get_volume() / n, \
        atoms.get_chemical_formula("hill", empirical=True)


def _deck_atoms(deck_text, tmp_path):
    path = tmp_path / "deck.d12"
    path.write_text(deck_text)
    p = S.read_structure(path)
    return E.atoms_from_payload(p), p["symmetry"]["number"]


def _monoclinic():
    from ase.spacegroup import crystal
    return crystal(["Zn", "O"], [(0.1, 0.2, 0.3), (0.35, 0.15, 0.4)], spacegroup=14,
                   cellpar=[5.1, 6.2, 7.3, 90, 101.5, 90])


@pytest.mark.parametrize("label, make, sg", [
    ("MgO", lambda: S.read_structure(EXAMPLES / "structures" / "MgO.cif"), 225),
    ("Si", lambda: S.read_structure(EXAMPLES / "structures" / "Si.cif"), 227),
    ("hBN", lambda: S.read_structure(EXAMPLES / "structures" / "hBN.cif"), 194),
    ("Ag2Br3", lambda: S.read_structure(EXAMPLES / "outputs" / "Ag2Br3_opt_unfinished.out"), 167),
    ("ZnO-P21c", None, 14),
])
def test_symmetrized_cif_round_trips_through_cif2d12(tmp_path, label, make, sg):
    from mace.gui import decks
    atoms = _monoclinic() if make is None else E.atoms_from_payload(make())
    s = E.EditSession(atoms, label)
    cif = tmp_path / f"{label}.cif"
    cif.write_text(s.cif_text("symmetrized"))
    prev = decks.preview("cif2d12", CONFIG, cif)
    deck_atoms, deck_sg = _deck_atoms(prev["deck"], tmp_path)
    assert deck_sg == sg == S.symmetry(atoms.cell.array, atoms.get_scaled_positions(),
                                       atoms.numbers)["number"]
    fa, va, fma = _fingerprint(s.atoms)
    fb, vb, fmb = _fingerprint(deck_atoms)
    assert fma == fmb and va == pytest.approx(vb, rel=1e-4)
    assert fa.keys() == fb.keys()
    for k in fa:   # the same neighbor shells, as a multiset per atom
        ma, mb = np.array(sorted(fa[k])), np.array(sorted(fb[k]))
        na, nb = len(s.atoms), len(deck_atoms)
        assert len(ma) / na == pytest.approx(len(mb) / nb)
        assert set(np.round(ma, 2)) == set(np.round(mb, 2))


def test_p1_cif_keeps_every_atom(tmp_path):
    s = _mgo()
    s.apply("add", {"element": "Li", "frac": [0.25, 0.25, 0.25]})
    import ase.io
    cif = tmp_path / "x.cif"
    cif.write_text(s.cif_text("p1"))
    back = ase.io.read(cif)
    assert len(back) == 9
    assert np.allclose(back.get_scaled_positions(), s.atoms.get_scaled_positions(), atol=1e-6)


# --- the editor API through the real server ------------------------------------------

@pytest.fixture(scope="module")
def server(tmp_path_factory):
    from test_gui_examples import _start
    root = tmp_path_factory.mktemp("guied") / "demo"
    shutil.copytree(EXAMPLES, root)
    proc, base, _ = _start(root)
    yield root, base
    proc.terminate()
    proc.wait(timeout=15)


def _post(base, route, body):
    from test_gui_examples import _json
    return _json(base, "api/edit/" + route, data=body)


def test_editor_api_edit_undo_save_and_deck_preview(server, tmp_path):
    from test_gui_examples import _cli, _json
    root, base = server
    st, r = _post(base, "open", {"path": "structures/MgO.cif"})
    assert st == 200 and r["reduced_formula"] == "MgO", r
    sid = r["session"]
    st, r = _post(base, "op", {"session": sid, "op": "replace",
                               "args": {"indices": [0], "element": "Ca"}})
    assert st == 200 and r["symbols"][0] == "Ca"
    st, r = _post(base, "op", {"session": sid, "op": "undo", "args": {}})
    assert r["symbols"][0] == "Mg"
    st, g = _json(base, "api/edit/state?session=" + sid)
    assert st == 200 and g["symbols"][0] == "Mg"
    st, r = _post(base, "op", {"session": sid, "op": "set_locked", "args": {"locked": True}})
    st, r = _post(base, "op", {"session": sid, "op": "move",
                               "args": {"indices": [0], "frac_delta": [0.02, 0.02, 0.02]}})
    assert st == 200 and r["needs_confirm"] and r["new_multiplicity"] == 32
    st, r = _post(base, "op", {"session": sid, "op": "nope", "args": {}})
    assert st == 400
    st, r = _post(base, "op", {"session": "missing", "op": "undo", "args": {}})
    assert st == 404
    # preview and save use the same stem, so the decks are byte-equal
    st, prev = _post(base, "deck_preview", {"session": sid, "name": "MgO_edit.cif",
                                            "json": CONFIG, "cif_mode": "symmetrized"})
    assert st == 200, prev
    assert prev["deck_name"].startswith("MgO_edit_") and "--cif_dir ." in prev["command"]
    st, saved = _post(base, "save_cif", {"session": sid, "name": "MgO_edit.cif",
                                         "mode": "symmetrized"})
    assert st == 200 and saved["saved"] == "MgO_edit.cif"
    st, again = _post(base, "save_cif", {"session": sid, "name": "MgO_edit.cif"})
    assert st == 400 and "exists" in again["error"]
    assert _post(base, "save_cif", {"session": sid, "name": "../x.cif"})[0] == 400
    cifs = tmp_path / "cifs"
    cifs.mkdir()
    shutil.copy2(root / "MgO_edit.cif", cifs)
    (tmp_path / "opts.json").write_text(json.dumps(CONFIG))
    _cli("convert", "--no-banner", "--batch", "--options_file", str(tmp_path / "opts.json"),
         "--cif_dir", str(cifs), "--output_dir", str(tmp_path / "decks"), cwd=tmp_path)
    assert (tmp_path / "decks" / prev["deck_name"]).read_text() == prev["deck"]


def test_editor_api_new_cell_and_clean(server):
    _, base = server
    st, r = _post(base, "new", {"name": "si_new", "cellpar": [5.431] * 3 + [90] * 3,
                                "spacegroup": 227, "sites": [{"element": "Si", "frac": [0, 0, 0]}]})
    assert st == 200 and len(r["numbers"]) == 8 and r["symmetry"]["number"] == 227
    sid = r["session"]
    _post(base, "op", {"session": sid, "op": "move",
                       "args": {"indices": [0], "delta": [0.15, 0.0, 0.0]}})
    st, c = _post(base, "clean", {"session": sid, "keep_symmetry": False})
    assert st == 200, c
    assert c["comparison"]["max_displacement"] > 0.05
    assert c["structure"]["symmetry"]["number"] == 227 and c["structure"]["can_undo"]
    st, r = _post(base, "op", {"session": sid, "op": "undo", "args": {}})
    assert r["symmetry"]["number"] != 227
    st, r = _post(base, "new", {"cellpar": [3.6] * 3 + [90] * 3, "spacegroup": 225,
                                "sites": [{"element": "Cu", "frac": [0, 0, 0]}]})
    st, c = _post(base, "clean", {"session": r["session"]})
    assert st == 200 and c["nothing_to_clean"]
    st, r = _post(base, "new", {"cellpar": [4.2] * 3 + [90] * 3, "spacegroup": 225,
                                "sites": [{"element": "Mg", "frac": [0, 0, 0]},
                                          {"element": "O", "frac": [0.5, 0.5, 0]}]})
    assert st == 400 and "same position" in r["error"]


def test_locked_cell_change_recaptures_the_lower_group():
    # cubic MgO stretched to orthorhombic is Fmmm; the cubic operations no
    # longer hold, so a move along z must grow Mg 4a -> 8 (Fmmm), not 24
    s = _mgo()
    s.apply("set_locked", {"locked": True})
    r = s.apply("set_cell", {"cellpar": [4.21, 4.5, 5.0, 90, 90, 90], "scale_atoms": True})
    assert r["locked_group"] == 32 and "re-captured" in r["op_summary"]
    with pytest.raises(E.NeedsConfirm) as info:
        s.apply("move", {"indices": [0], "frac_delta": [0, 0, 0.05]})
    assert info.value.details == {"old_multiplicity": 4, "new_multiplicity": 8}
    s.apply("undo", {})
    assert s._locked_group() == 192


def test_locked_clean_result_recaptures_the_group():
    s = _mgo()
    s.apply("set_locked", {"locked": True})
    rattled = s.atoms.copy()
    rattled.positions[0] += [0.1, 0.05, 0.0]
    r = s.apply_clean(rattled, "Clean")
    assert r["locked_group"] < 192 and "re-captured" in r["op_summary"]
    s.apply("undo", {})
    assert s._locked_group() == 192


def test_symmetrized_cif_block_name_is_one_cif_token():
    from ase.build import bulk
    s = E.EditSession(bulk("MgO", "rocksalt", a=4.21, cubic=True), "my run (2)")
    first = s.cif_text("symmetrized").splitlines()[0]
    assert first.startswith("data_") and " " not in first and "(" not in first


@pytest.mark.parametrize("bad", ["nan", "-1", "0", "inf", "abc"])
def test_symprec_is_validated_at_open(bad):
    with pytest.raises(E.EditError, match="tolerance"):
        E.check_symprec(bad)
    assert E.check_symprec("1e-3") == 1e-3


def test_editor_api_rejects_a_bad_open_tolerance(server):
    _, base = server
    st, r = _post(base, "open", {"path": "structures/MgO.cif", "symprec": -1})
    assert st == 400 and "tolerance" in r["error"]


# --- Phase 3: labeled, numbered history; merges; one memory budget ------------------------

def _mgo_n(n):
    """MgO repeated n (a tuple), generated here: 8 atoms per cubic cell."""
    from ase.build import bulk
    return E.EditSession(bulk("MgO", "rocksalt", a=4.212, cubic=True).repeat(n), "MgO")


def test_history_labels_revs_and_jump():
    s = _mgo()
    r0 = s.rev
    r = s.apply("replace", {"indices": [0], "element": "Ca"})
    assert [h["label"] for h in r["history"]] == ["Opened MgO", "Replace Mg→Ca (1 atom)"]
    assert r["position"] == 1 and r["rev"] > r0 and r["history"][0]["rev"] == r0
    r = s.apply("delete", {"indices": [1, 2, 3]})
    r = s.apply("move", {"indices": [0], "delta": [0.1, 0, 0]})
    labels = [h["label"] for h in r["history"]]
    assert labels[2:] == ["Delete 3 Atoms", "Move 1 Atom"]
    assert all(h["n_atoms"] == n for h, n in zip(r["history"], [8, 8, 5, 5]))
    revs = [h["rev"] for h in r["history"]]
    assert revs == sorted(revs) and len(set(revs)) == 4
    # one jump back to the opened state: the rows after it stay redoable
    r = s.jump(0)
    assert r["position"] == 0 and r["rev"] == r0 and r["symbols"].count("Mg") == 4
    assert [h["label"] for h in r["history"]] == labels and r["can_redo"] and not r["can_undo"]
    r = s.jump(2)
    assert r["rev"] == revs[2] and len(r["numbers"]) == 5
    r = s.apply("redo", {})
    assert r["rev"] == revs[3]
    # a new edit after a jump drops the redo rows
    s.jump(1)
    r = s.apply("replace", {"indices": [4], "element": "S"})
    assert [h["label"] for h in r["history"]][-1] == "Replace O→S (1 atom)" and len(r["history"]) == 3
    with pytest.raises(E.EditError):
        s.jump(7)


def test_nudges_merge_only_into_named_rev():
    s = _mgo()
    key = "nudge:0"
    r = s.apply("move", {"indices": [0], "delta": [0.01, 0, 0]}, merge=key)
    first = r["rev"]
    r = s.apply("move", {"indices": [0], "delta": [0.01, 0, 0]}, merge=key, merge_into=first)
    r = s.apply("move", {"indices": [0], "delta": [0.01, 0, 0]}, merge=key, merge_into=r["rev"])
    assert [h["label"] for h in r["history"]] == ["Opened MgO", "Move 1 Atom (×3)"]
    assert r["rev"] != first and s.atoms.positions[0][0] == pytest.approx(0.03)
    # a stale merge_into, or another key, records a new step
    r = s.apply("move", {"indices": [0], "delta": [0.01, 0, 0]}, merge=key, merge_into=first)
    r = s.apply("move", {"indices": [0], "delta": [0.01, 0, 0]}, merge="nudge:1", merge_into=r["rev"])
    assert [h["label"] for h in r["history"]] == ["Opened MgO", "Move 1 Atom (×3)", "Move 1 Atom", "Move 1 Atom"]
    # a request carrying several nudges (the browser summed the waiting ones) counts them all
    r = s.apply("move", {"indices": [0], "delta": [0.02, 0, 0]}, merge=key, merge_count=2)
    r = s.apply("move", {"indices": [0], "delta": [0.01, 0, 0]}, merge=key, merge_into=r["rev"], merge_count=3)
    assert r["history"][-1]["label"] == "Move 1 Atom (×5)"
    s.apply("undo", {})
    # undoing the merged step undoes all three nudges at once
    for _ in range(3):
        r = s.apply("undo", {})
    assert s.atoms.positions[0][0] == pytest.approx(0.0) and not r["can_undo"]


def test_set_locked_record_false_is_not_an_undo_step():
    s = _mgo()
    rev = s.rev
    r = s.apply("set_locked", {"locked": True}, record=False)
    assert r["locked"] and r["locked_group"] == 192 and r["rev"] == rev and not r["can_undo"]
    assert len(r["history"]) == 1 and r["ops"]["rotations"] and r["ops_key"]
    # the operations are sent once: a payload for a known key leaves them out
    r = s.apply("replace", {"indices": [0], "element": "Ca"}, known_ops_key=r["ops_key"])
    assert "ops" not in r and r["symbols"].count("Ca") == 4
    assert r["label"] == "Replace Mg→Ca (4 atoms, orbit 4a)"
    r = s.apply("set_locked", {"locked": False}, record=False)
    r = s.apply("undo", {})
    assert not r["locked"] and r["symbols"].count("Ca") == 0       # the mode stays P1 across undo
    r = s.apply("redo", {})
    assert not r["locked"]
    with pytest.raises(E.EditError):
        s.apply("delete", {"indices": [0]}, record=False)


def test_orbit_label_and_multiplicity_use_the_conventional_cell():
    """spglib counts an orbit's atoms in the cell it is given; the label and
    the Wyckoff multiplicity count them in the conventional cell, as the
    tables do: MgO's Mg is 4a in a primitive cell and in a supercell too,
    and the operations stay 192."""
    from ase.build import bulk
    from mace.gui import structures as S
    for atoms, n_mg in [(bulk("MgO", "rocksalt", a=4.212), 1),
                        (bulk("MgO", "rocksalt", a=4.212, cubic=True).repeat(2), 32)]:
        y = S.symmetry(atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers)
        assert y["number"] == 225 and y["n_operations_conventional"] == 192
        assert {(w, m) for w, m in zip(y["wyckoffs"], y["multiplicity"])} == {("a", 4), ("b", 4)}
        s = E.EditSession(atoms, "MgO")
        s.apply("set_locked", {"locked": True}, record=False)
        r = s.apply("replace", {"indices": [0], "element": "Ca"})
        assert r["history"][-1]["label"] == f"Replace Mg→Ca ({n_mg} atom{'s' * (n_mg != 1)}, orbit 4a)"


def test_replace_label_names_orbit():
    s = _mgo()
    s.apply("set_locked", {"locked": True}, record=False)
    r = s.apply("replace", {"indices": [0], "element": "Ca"})
    assert r["history"][-1]["label"] == "Replace Mg→Ca (4 atoms, orbit 4a)"
    r = s.apply("add", {"element": "Li", "frac": [0.25, 0.25, 0.25]})
    assert r["history"][-1]["label"] == "Add Li (8 atoms, orbit 8c)"
    r = s.apply("set_cell", {"cellpar": [4.3] * 3 + [90] * 3})
    assert r["history"][-1]["label"] == "Edit Cell"


def test_history_byte_cap_drops_oldest_and_flags(monkeypatch):
    from mace.gui import memory
    s = _mgo()
    s.undo_limit = 5
    flags = [s.apply("move", {"indices": [0], "delta": [0.001, 0, 0]})["history_trimmed"] for _ in range(7)]
    assert flags == [False] * 5 + [True, True]          # each payload says whether steps were dropped
    r = s.payload()
    assert r["history_trimmed"] is False                  # and only once
    assert len(r["history"]) == 6 and r["history"][0]["label"] == "Move 1 Atom"   # 5 steps + the current state
    s.undo_limit = 200
    monkeypatch.setattr(memory, "SESSION_HISTORY_CAP", 3 * s.history[-1]["nbytes"] + 10)
    r = s.apply("move", {"indices": [0], "delta": [0.001, 0, 0]})
    assert r["history_trimmed"] is True and len(s.history) == 3
    assert s.history_nbytes == sum(e["nbytes"] for e in s.history)
    r = s.apply("move", {"indices": [0], "delta": [0.001, 0, 0]})
    assert len(s.history) == 3


@pytest.mark.parametrize("repeat", [(5, 5, 5), (5, 5, 25)])
def test_snapshot_nbytes_matches_tracemalloc(repeat):
    """The ledger's estimate of a history entry is within 20% of what Python
    really holds for it (1000 and 5000 atoms, generated here): 50 snapshots
    taken the way an edit takes them."""
    import gc
    import tracemalloc
    s = _mgo_n(repeat)
    s.apply("set_locked", {"locked": True}, record=False)
    older = s._state()        # holds the operations, which every later snapshot shares
    gc.collect()
    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        entries = []
        for k in range(50):
            s.atoms.positions[k] += 0.001
            st = s._state()
            entries.append(s._entry(st, dict(s.meta), older))
            older = st
        gc.collect()
        grown = tracemalloc.get_traced_memory()[0] - base
    finally:
        tracemalloc.stop()
    est = sum(e["nbytes"] for e in entries)
    assert abs(est - grown) <= 0.2 * grown, (est, grown)
    # deviation D7's trigger: diffs would be needed above 160 KB per 5000-atom entry
    assert entries[-1]["nbytes"] <= 160_000


def test_history_cap_holds_while_other_sessions_are_busy(tmp_path, monkeypatch):
    """The bench found the history 0.1 MB over its cap: the ledger skips busy
    sessions, so with every other session in the middle of an edit nothing
    could be trimmed. An edit now trims its own oldest steps last, and the
    peak is recorded after that: the cap is never passed."""
    from mace.gui import server
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("MACE_GUI_CACHE_MB", "64")
    monkeypatch.setattr(E.EditSession, "symmetry", lambda self: {"available": False, "note": "test"})
    st = server.GuiState(tmp_path, "t")
    try:
        led = st.ledger
        busy = [st.sessions.add(_mgo_n((5, 5, 25))) for _ in range(2)]
        for s in busy:
            for j in range(100):
                s.apply("move", {"indices": [j], "delta": [0.001, 0, 0]})
        for s in busy:
            s.lock.acquire()                     # in the middle of their own edits
        try:
            mine = st.sessions.add(_mgo_n((5, 5, 25)))
            for j in range(150):
                mine.apply("move", {"indices": [j], "delta": [0.001, 0, 0]})
                assert st.sessions.history_bytes() <= led.history_cap()
        finally:
            for s in busy:
                s.lock.release()
        assert led.max_history <= led.history_cap()
    finally:
        st.close()


def test_ledger_shares_one_budget(tmp_path, monkeypatch):
    """Static cache + history + live sessions under one cache_mb: history at
    most half, the total at most the limit, and eviction in the stated order."""
    from mace.gui import server
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("MACE_GUI_CACHE_MB", "64")
    # spglib is not what this measures (5000 atoms take a second each)
    monkeypatch.setattr(E.EditSession, "symmetry", lambda self: {"available": False, "note": "test"})
    st = server.GuiState(tmp_path, "t")
    try:
        led = st.ledger
        assert led.limit == 64 * 1024 * 1024
        for name in ["js/shell.js", "css/shell.css", "js/actions.js", "app.js", "index.html", "editor.js"]:
            st.static_bytes(name, True)
            st.static_bytes(name, False)
        static_before = st.static_used()
        sessions = []
        for k in range(4):
            s = st.sessions.add(_mgo_n((5, 5, 25)))
            sessions.append(s)
            for j in range(80):
                s.apply("move", {"indices": [j], "delta": [0.001, 0, 0]})
                led.enforce()
                assert led.total() <= led.limit
                assert st.sessions.history_bytes() <= led.history_cap()
        # the least recently used session lost its oldest steps first
        assert len(sessions[0].history) < len(sessions[-1].history)
        assert sessions[-1].history[0]["meta"]["label"] == "Opened MgO"
        assert st.static_used() == static_before          # static files were not needed
        assert led.max_history <= led.history_cap() and led.max_total <= led.limit
        # a smaller limit: history first, then idle sessions, then static files
        led.set_limit(4 * 1024 * 1024)
        assert st.sessions.history_bytes() <= 2 * 1024 * 1024 and led.total() <= led.limit
        assert st.static_used() == static_before          # history went first
        rep = led.report()
        assert rep["limit_mb"] == 4.0 and "history_mb" in rep and "static_mb" in rep
        # idle sessions go, least recently used first; one used just now stays
        now = __import__("time").monotonic()
        for s in sessions[:-1]:
            s.last_used = now - 120
        led.set_limit(300_000)          # below what the four sessions alone hold
        assert st.sessions.history_bytes() == 0
        assert st.sessions.ids() == [sessions[-1].id]       # never one used in the last minute
        assert st.static_used() <= led.static_cap() and led.total() <= led.limit
    finally:
        st.close()


def test_stale_base_rev_raises():
    s = _mgo()
    rev = s.rev
    s.apply("replace", {"indices": [0], "element": "Ca"}, base_rev=rev)
    with pytest.raises(E.StaleRev) as info:
        s.apply("replace", {"indices": [1], "element": "Ca"}, base_rev=rev)
    assert info.value.rev == s.rev and s.atoms.get_chemical_symbols().count("Ca") == 1
    with pytest.raises(E.StaleRev):
        s.jump(0, base_rev=rev)


def test_store_evicts_least_recently_used(monkeypatch):
    monkeypatch.setattr(E, "MAX_SESSIONS", 3)
    store = E.SessionStore()
    a, b, c = (store.add(_mgo()) for _ in range(3))
    store.get(a.id)                      # a is used again: b is now the oldest
    d = store.add(_mgo())
    assert set(store.ids()) == {a.id, c.id, d.id}
    with pytest.raises(KeyError):
        store.get(b.id)
    # revisions never repeat, not even across sessions
    assert len({a.rev, c.rev, d.rev}) == 3


def test_move_wrap_counts_wrapped_atoms():
    s = _mgo()
    r = s.apply("move", {"indices": [0, 4], "frac_delta": [-0.05, 0, 0], "wrap": True})
    assert r["wrapped"] == 1                          # Mg1 at x=0 left the cell, O5 at 0.5 did not
    f = np.array(r["frac"])
    assert np.all((f >= 0) & (f < 1)) and f[0][0] == pytest.approx(0.95)
    r = s.apply("move", {"indices": [0], "frac_delta": [0.1, 0, 0]})          # wrap off: the default
    assert r["wrapped"] == 0 and r["frac"][0][0] == pytest.approx(1.05)


def test_project_keeps_group_and_orbit_size():
    """Keep Fm-3m: a move of O off 4b is projected onto what 4b allows (nothing)."""
    s = _mgo()
    s.apply("set_locked", {"locked": True}, record=False)
    o = s.atoms.get_chemical_symbols().index("O")
    with pytest.raises(E.NeedsConfirm):
        s.apply("move", {"indices": [o], "frac_delta": [0.03, 0.01, 0]})
    r = s.apply("move", {"indices": [o], "frac_delta": [0.03, 0.01, 0], "project": True})
    assert r["symmetry"]["number"] == 225 and len(r["numbers"]) == 8
    # rutile O on 4f (x, x, 0): only the (1, 1, 0) part of a move survives
    from ase.spacegroup import crystal
    s = E.EditSession(crystal(["Ti", "O"], [(0, 0, 0), (0.3, 0.3, 0)], spacegroup=136,
                              cellpar=[4.59, 4.59, 2.96, 90, 90, 90]), "rutile", locked=True)
    o = int(np.where(s.atoms.numbers == 8)[0][0])
    f0 = s.atoms.get_scaled_positions()[o]
    r = s.apply("move", {"indices": [o], "frac_delta": [0.02, 0.0, 0.01], "project": True})
    assert r["symmetry"]["number"] == 136 and len(r["numbers"]) == 6
    assert np.array(r["frac"][o]) - f0 == pytest.approx([0.01, 0.01, 0.0], abs=1e-9)


def test_set_positions_p1_and_sym():
    s = _mgo()
    r = s.apply("set_positions", {"indices": [4], "frac": [[0.55, 0.0, 0.0]]})
    assert r["frac"][4] == pytest.approx([0.55, 0, 0]) and r["label"] == "Set Coordinates"
    r = s.apply("set_positions", {"indices": [4], "cart": [[1.0, 2.0, 3.0]]})
    assert r["cart"][4] == pytest.approx([1.0, 2.0, 3.0])
    with pytest.raises(E.EditError):
        s.apply("set_positions", {"indices": [4, 5], "frac": [[0.1, 0.1, 0.1]]})
    s = _mgo()
    s.apply("set_locked", {"locked": True}, record=False)
    o = s.atoms.get_chemical_symbols().index("O")
    with pytest.raises(E.NeedsConfirm):
        s.apply("set_positions", {"indices": [o], "frac": [[0.55, 0.0, 0.0]]})
    r = s.apply("set_positions", {"indices": [o], "frac": [[0.55, 0.0, 0.0]], "confirm": True})
    assert r["symbols"].count("O") > 4


def test_set_cell_matrix():
    s = _mgo()
    m = [[4.3, 0, 0], [0, 4.3, 0], [0.2, 0, 4.4]]
    r = s.apply("set_cell", {"matrix": m, "scale_atoms": True})
    assert np.allclose(r["cell"], m) and r["frac"][4] == pytest.approx(s.atoms.get_scaled_positions(wrap=False)[4])
    with pytest.raises(E.EditError):
        s.apply("set_cell", {"matrix": [[1, 0, 0], [2, 0, 0], [0, 0, 1]]})


def test_resources_route_reports_each_store(server):
    from test_gui_examples import _json
    _, base = server
    st, r = _json(base, "api/resources")
    assert st == 200 and r["stores"]["limit_mb"] == 512.0
    assert {"static_mb", "history_mb", "sessions_mb", "total_mb"} <= set(r["stores"])
    assert r["compute"]["cap"] >= 1 and r["cache_used_mb"] == pytest.approx(r["stores"]["total_mb"], abs=0.2)


def test_p1_cif_fast_path_matches_ase(tmp_path):
    """A big P1 CIF skips ase's O(N²) duplicate check (8000 atoms took 190 s)
    and gives the same atoms; anything with symmetry or partial sites does not."""
    import warnings
    import ase.io
    p = tmp_path / "mgo_p1.cif"
    ase.io.write(p, _mgo_n((5, 5, 5)).atoms)
    fast = S._p1_cif_atoms(p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slow = ase.io.read(p, format="cif")
    assert fast is not None and len(fast) == len(slow) == 1000
    assert np.allclose(fast.positions, slow.positions) and (fast.numbers == slow.numbers).all()
    assert np.allclose(fast.cell.array, slow.cell.array)
    assert S._p1_cif_atoms(EXAMPLES / "structures" / "MgO.cif") is None            # Fm-3m: ase's path
    part = tmp_path / "partial.cif"
    part.write_text(p.read_text().replace("1.0000\n", "0.5000\n", 1))
    assert "0.5000" in part.read_text() and S._p1_cif_atoms(part) is None


def test_heavy_symmetry_calls_run_one_at_a_time():
    """Section 9.2: spglib on a big supercell needs tens of MB while it runs,
    so heavy calls take turns; a small one still runs beside a heavy one."""
    import threading
    import time
    from mace.gui.compute import ComputeGate
    gate = ComputeGate(4)
    lock, now, peak = threading.Lock(), {"heavy": 0, "light": 0}, {"heavy": 0, "both": 0}

    def job(kind):
        def fn():
            with lock:
                now[kind] += 1
                peak["heavy"] = max(peak["heavy"], now["heavy"])
                peak["both"] = max(peak["both"], now["heavy"] + now["light"] if now["heavy"] else 0)
            time.sleep(0.05)
            with lock:
                now[kind] -= 1
        return fn
    threads = [threading.Thread(target=gate.run, args=(None, job("heavy")), kwargs={"heavy": True}) for _ in range(4)]
    threads += [threading.Thread(target=gate.run, args=(None, job("light"))) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak["heavy"] == 1 and peak["both"] >= 2
    assert gate.stats()["running"] == 0 and gate.heavy_running == 0


def test_history_cap_holds_with_concurrent_edits(tmp_path, monkeypatch):
    """Four sessions editing at once (four request threads) never take the
    shared history past its cap, not even between one commit and its trim."""
    import threading
    from mace.gui import server
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("MACE_GUI_CACHE_MB", "64")
    monkeypatch.setattr(E.EditSession, "symmetry", lambda self: {"available": False, "note": "test"})
    st = server.GuiState(tmp_path, "t")
    try:
        led = st.ledger
        sessions = [st.sessions.add(_mgo_n((5, 5, 25))) for _ in range(4)]

        def work(s):
            for j in range(70):
                with s.lock:
                    s.apply("move", {"indices": [j], "delta": [0.001, 0, 0]})
        threads = [threading.Thread(target=work, args=(s,)) for s in sessions]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert st.sessions.history_bytes() <= led.history_cap()
        assert led.max_history <= led.history_cap(), (led.max_history, led.history_cap())
    finally:
        st.close()


def test_a_gated_call_inside_a_held_slot_does_not_wait_for_another():
    """A task holding the only slot (one worker on a small allocation) that
    reads a CIF (gated symmetry) runs that call in its own slot instead of
    waiting forever; other threads still wait for the slot."""
    import threading
    import time
    from mace.gui.compute import ComputeGate
    gate = ComputeGate(1)
    out, other = {}, {}
    started = threading.Event()

    def task():
        with gate.slot(pool=False):
            out["inner"] = gate.run(None, lambda: "symmetry")
            with gate.slot():
                out["nested"] = True
            started.wait(2)
            time.sleep(0.2)
            other["blocked"] = "ran" not in other

    def outsider():
        started.set()
        other["ran"] = gate.run(None, lambda: True)
    t = threading.Thread(target=task, daemon=True)        # a deadlock fails the test, never hangs it
    t.start()
    time.sleep(0.05)
    o = threading.Thread(target=outsider, daemon=True)
    o.start()
    t.join(5)
    o.join(5)
    assert not t.is_alive() and not o.is_alive()
    assert out == {"inner": "symmetry", "nested": True}
    assert other["blocked"] is True and other["ran"] is True       # the outsider waited its turn
    assert gate.stats()["running"] == 0
