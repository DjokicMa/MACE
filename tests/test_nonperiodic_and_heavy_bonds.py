"""Slabs, polymers and molecules in the editor, Clean and Supercell (1.1.6), and the
bond radii of the heavy elements.

Before 1.1.6 the structure editor (and with it Move, Add, Bond and Clean) refused
anything but a 3D crystal, and a builder or a dropped file of lower dimension
opened view-only. A session now keeps the structure's ``pbc``: nothing wraps or
has images along a non-periodic direction, Sym mode stays 3D-only, Clean keeps
no space group there, and Supercell repeats along a non-periodic direction only
at an explicit multiple and a vacuum spacing. Heavy cations (Ag+, Pb2+, Tl+, ...)
are bonded at their ionic distances, and every element up to Og has a radius.
"""
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT

pytest.importorskip("ase")
pytest.importorskip("spglib")

from ase import Atoms  # noqa: E402
from ase.build import bulk  # noqa: E402

from mace.gui import builders as B  # noqa: E402
from mace.gui import clean as C  # noqa: E402
from mace.gui import editor as E  # noqa: E402
from mace.gui import structures as S  # noqa: E402

EXAMPLES = REPO_ROOT / "examples" / "gui"
SLAB_OUT = EXAMPLES / "outputs" / "graphene_slab_sp.out"
CORE_JS = REPO_ROOT / "mace" / "gui" / "static" / "js" / "geom" / "core.js"


def _pairs(atoms, scale=1.15):
    return S._neighbor_pairs(atoms.cell.array, atoms.positions, list(atoms.numbers), [True] * 3, scale=scale,
                             max_pairs=100000)


def _bonded(atoms, a, b, scale=1.15):
    syms = atoms.get_chemical_symbols()
    return any({syms[i], syms[j]} == {a, b} for i, j, _ in _pairs(atoms, scale))


# ---------------------------------------------------------------- heavy-element bonds

def test_heavy_cations_bond_at_ionic_distances():
    agcl = bulk("AgCl", "rocksalt", a=5.78)                  # Ag-Cl 2.89 A: 1.17 x the covalent sum
    assert _bonded(agcl, "Ag", "Cl")
    pto = Atoms("PbTiO3", cell=np.eye(3) * 3.97, pbc=True,
                scaled_positions=[[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
    assert _bonded(pto, "Pb", "O") and _bonded(pto, "Ti", "O")   # Pb-O 2.81 A (A site, 12-fold)
    tlcl = Atoms("TlCl", cell=np.eye(3) * 3.84, pbc=True, scaled_positions=[[0, 0, 0], [.5, .5, .5]])
    assert _bonded(tlcl, "Tl", "Cl")                         # Tl-Cl 3.33 A (CsCl type)
    # and only the first shell: AgCl's second Ag-Cl shell (5.0 A) is not a bond
    syms = agcl.repeat((2, 2, 2)).get_chemical_symbols()
    a2 = agcl.repeat((2, 2, 2))
    d = [a2.get_distance(i, j, mic=True) for i, j, _ in _pairs(a2) if {syms[i], syms[j]} == {"Ag", "Cl"}]
    assert d and max(d) < 3.0


def test_metals_stay_unbonded_and_light_elements_unchanged():
    # metal-metal pairs are bonds only in a structure of metals only (1.1.6, as the viewer draws them):
    # fcc Ag has its 12 Ag-Ag at the covalent radii (2.89 A < 1.15 x 2 x 1.45 A), not Ag's ionic bond radius
    ag = bulk("Ag", "fcc", a=4.09)
    assert len(_pairs(ag)) == 6 and not _pairs(bulk("Ag", "fcc", a=4.09 * 1.2))
    mgo = bulk("MgO", "rocksalt", a=4.212)
    assert S._bond_radii([12, 8]).tolist() == S._covalent([12, 8]).tolist()
    assert _bonded(mgo, "Mg", "O") and not _bonded(mgo, "Mg", "Mg")


def test_every_element_has_a_radius():
    rad = S._covalent(list(range(1, 119)))
    assert np.all(rad > 0.25), [z for z, r in zip(range(1, 119), rad) if r <= 0.25]
    assert S._covalent([292])[0] == S._covalent([92])[0]      # CRYSTAL's ECP numbering
    assert S._bond_radii([282])[0] == pytest.approx(1.975)


def test_js_tables_match_the_server():
    """geom/core.js draws bonds with the same radii as structures.py tabulates them."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed (the GUI's own tests check core.js in Firefox)")
    code = (f"import('{CORE_JS.as_uri()}').then((G) => {{ const r = []; for (let z = 1; z <= 118; z++) "
            "r.push(G.bondRadius(z)); console.log(JSON.stringify({r, sym: G.SYMBOLS.length, "
            "mr: G.metalRadius(G.bondRadius(47))})); });")
    out = subprocess.run([node, "--input-type=module", "-e", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    js = json.loads(out.stdout)
    assert js["sym"] == 119
    assert np.allclose(js["r"], S._bond_radii(list(range(1, 119))), atol=1e-6)
    assert js["mr"] == pytest.approx(S._covalent([47])[0])


# ---------------------------------------------------------------- the editor off 3D

def _slab_session():
    p = S.read_structure(SLAB_OUT)
    assert p["pbc"] == [True, True, False]
    return E.EditSession(E.atoms_from_payload(p), "graphene", locked=True), p


def test_slab_session_keeps_its_periodicity():
    s, p = _slab_session()
    assert not s.locked                                       # Sym mode is 3D only: the session opens in P1
    out = s.payload()
    assert out["pbc"] == [True, True, False] and out["dimensionality"] == "SLAB"
    assert out["origin"] == pytest.approx(p["origin"])
    assert out.get("vacuum_vectors") and out["vacuum_vectors"][2][2] == pytest.approx(500.0)   # CRYSTAL's formal c
    z0 = s.atoms.positions[0, 2]
    # a move far along c never wraps (the box is a drawing aid), along a it does
    r = s.apply("move", {"indices": [0], "delta": [0.0, 0.0, 30.0], "wrap": True})
    assert s.atoms.positions[0, 2] == pytest.approx(z0 + 30.0)
    assert r["pbc"] == [True, True, False]
    s.apply("move", {"indices": [0], "frac_delta": [1.25, 0.0, 0.0], "wrap": True})
    f = s.atoms.get_scaled_positions(wrap=False)[0]
    assert 0 <= f[0] < 1 and s.atoms.positions[0, 2] == pytest.approx(z0 + 30.0)
    # add and delete
    n = len(s.atoms)
    s.apply("add", {"element": "H", "cart": (s.atoms.positions[1] + [0, 0, 1.1]).tolist()})
    assert len(s.atoms) == n + 1
    s.apply("delete", {"indices": [n]})
    assert len(s.atoms) == n
    with pytest.raises(E.EditError, match="3D crystal"):
        s.apply("set_locked", {"locked": True}, record=False)
    with pytest.raises(E.EditError, match="3D crystal"):
        s.apply("primitive", {})
    s.apply("undo", {})
    assert len(s.atoms) == n + 1


def test_slab_supercell_repeats_c_only_at_a_vacuum_spacing():
    s, _ = _slab_session()
    n = len(s.atoms)
    s.apply("supercell", {"n": [2, 2, 1]})
    assert len(s.atoms) == 4 * n and list(s.atoms.pbc) == [True, True, False]
    with pytest.raises(E.EditError, match="n3 must be 1"):
        s.apply("supercell", {"n": [1, 1, 2]})
    z = s.atoms.positions[:, 2]
    h = float(z.max() - z.min())
    s.apply("supercell", {"n": [1, 1, 3], "vacuum": [None, None, [0, 0, h + 10.0]]})
    z3 = np.sort(s.atoms.positions[:, 2])
    assert len(z3) == 12 * n and z3[-1] - z3[0] == pytest.approx(3 * h + 20.0)
    assert list(s.atoms.pbc) == [True, True, False]


def test_molecule_supercell_and_estimate():
    cell = np.diag([3.0, 2.0, 0.5])
    frac = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    with pytest.raises(ValueError, match="no periodic direction"):
        B.supercell(cell=cell, frac=frac, numbers=[8, 1, 1], pbc=(False,) * 3, matrix=np.diag([2, 1, 1]))
    r = B.supercell(cell=cell, frac=frac, numbers=[8, 1, 1], pbc=(False,) * 3, matrix=np.diag([2, 1, 1]),
                    vacuum=[[8.0, 0, 0], None, None])
    cart = np.asarray(r["frac"]) @ np.asarray(r["cell"])
    assert len(r["numbers"]) == 6 and r["pbc"] == [False] * 3
    assert sorted(set(np.round(cart[:, 0], 6))) == [0.0, 3.0, 8.0, 11.0]
    assert r["info"]["matrix"][0][0] == 2


def test_molecule_session_moves_adds_and_cleans():
    water = Atoms("OH2", positions=[[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]])
    p = S._payload(np.diag([1.514, 0.586, 0.5]), None, water.positions - water.positions.min(0), [8, 1, 1],
                   (False, False, False), {"kind": "xyz", "path": "water.xyz"})
    s = E.EditSession(E.atoms_from_payload(p), "water")
    s.apply("move", {"indices": [1], "delta": [0.08, 0.0, 0.0], "wrap": True})
    assert s.payload()["dimensionality"] == "MOLECULE"
    # the springs bond O-H only: the box (1.5 x 0.6 x 0.5 A) is no period
    assert sorted((i, j) for i, j, _ in C.bonded_pairs(s.atoms)) == [(0, 1), (0, 2)]
    res = C.clean(s.atoms, keep_symmetry=True, relax_cell=True, time_limit=5)
    assert "no space group to keep" in " ".join(res["notes"]) and "no cell" in " ".join(res["notes"])
    d = res["atoms"].get_distance(0, 1)
    assert abs(d - (0.66 + 0.31)) < abs(water.get_distance(0, 1) + 0.0 - (0.66 + 0.31)) + 0.35
    s.apply_clean(res["atoms"], "Clean")
    assert list(s.atoms.pbc) == [False] * 3 and "mace_origin" in s.atoms.info


def test_slab_clean_relaxes_the_periodic_directions_only():
    s, _ = _slab_session()
    c0 = s.atoms.cell.array.copy()
    res = C.clean(s.atoms, keep_symmetry=False, relax_cell=True, time_limit=3)
    c1 = res["atoms"].cell.array
    assert np.allclose(c1[2], c0[2]) and "only the periodic directions" in " ".join(res["notes"])


def test_saving_boxes_a_slab_for_cif_and_poscar():
    s, _ = _slab_session()
    a = S.boxed(s.atoms)
    assert all(a.pbc) and not any(k.startswith("mace_") for k in a.info)
    z = s.atoms.positions[:, 2]
    assert a.cell.array[2][2] == pytest.approx(float(z.max() - z.min()) + S.SAVE_VACUUM)
    za = a.get_scaled_positions()[:, 2]
    assert za.min() > 0.05 and za.max() < 0.95
    text = s.cif_text("p1")
    assert "_cell_length_c" in text
    with pytest.raises(E.EditError, match="P1"):
        s.cif_text("symmetrized")


def test_extxyz_partial_pbc_opens_as_a_slab(tmp_path):
    import ase.io
    g = Atoms("C2", cell=[[2.46, 0, 0], [-1.23, 2.13, 0], [0, 0, 20.0]], pbc=[True, True, False],
              scaled_positions=[[0, 0, 0.5], [1 / 3, 2 / 3, 0.5]])
    ase.io.write(tmp_path / "g.extxyz", g, format="extxyz")
    p = S.read_any(tmp_path / "g.extxyz", "extxyz")
    assert p["pbc"] == [True, True, False] and p["dimensionality"] == "SLAB"
    assert p["vacuum_vectors"][2] == pytest.approx([0, 0, 20.0])


def _min_distance(atoms):
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, 99.0)
    return float(d.min())


def test_repeated_slab_supercells_never_overlap():
    """After a supercell along c at the file's 500 Å the old 500 Å stayed as the file's vacuum vector, so a
    second 1x1x2 put copies on copies (105 pairs under 0.1 Å). The repeated axis's vector becomes the new
    block's period, and a file vector that no longer holds the atoms is not used."""
    s, p = _slab_session()
    n, d0 = len(s.atoms), _min_distance(s.atoms)
    file_vec = p["vacuum_vectors"]
    s.apply("supercell", {"n": [1, 1, 2], "vacuum": file_vec})
    assert s.atoms.info["mace_vacuum_vectors"][2] == pytest.approx([0, 0, 1000.0])
    vv = s.payload()["vacuum_vectors"]
    s.apply("supercell", {"n": [1, 1, 2], "vacuum": vv})
    assert len(s.atoms) == 4 * n and _min_distance(s.atoms) == pytest.approx(d0)
    z = np.sort(np.unique(np.round(s.atoms.positions[:, 2], 3)))
    assert np.diff(z).min() > 400                                     # four layers 500 Å apart
    # the rule (structures.vacuum_vector, core.js vacuumAxis): a file vector shorter than the atoms' extent
    # is not the spacing; the cell-mode fallback is
    cart = s.atoms.positions
    v = S.vacuum_vector(s.atoms.cell.array, cart, 2, "cell", file_vectors=[None, None, [0, 0, 500.0]])
    assert float(v[2]) > float(np.ptp(cart[:, 2]))
    assert S.vacuum_vector(s.atoms.cell.array, cart, 2, "cell", file_vectors=[None, None, [0, 0, 1700.0]])[2] == 1700.0
    assert S.vacuum_vector(s.atoms.cell.array, cart, 2, "cell", file_vectors=[None, None, [0, 0, -1700.0]])[2] > 0


def test_a_reversed_or_short_vacuum_spacing_is_refused():
    s, _ = _slab_session()
    with pytest.raises(E.EditError, match="copies would overlap"):
        s.apply("supercell", {"n": [1, 1, 2], "vacuum": [None, None, [0, 0, -20.0]]})
    cell, frac = np.diag([3.0, 2.0, 0.5]), [[0, 0, 0], [1, 0, 0], [0, 1, 0]]      # a molecule 3 Å long along x
    for bad in ([2.0, 0, 0], [-8.0, 0, 0]):
        with pytest.raises(ValueError, match="copies would overlap"):
            B.supercell(cell=cell, frac=frac, numbers=[8, 1, 1], pbc=(False,) * 3, matrix=np.diag([2, 1, 1]),
                        vacuum=[bad, None, None])


def test_js_vacuum_axis_takes_a_file_vector_only_when_it_holds_the_atoms(tmp_path):
    if not shutil.which("node"):
        pytest.skip("node not installed")
    cell = [[2.46, 0, 0], [-1.23, 2.13, 0], [0, 0, 1000.5]]
    cart = [[0, 0, 0], [0, 0, 1000.5]]
    script = tmp_path / "v.mjs"
    script.write_text(f"import {{ vacuumAxis }} from {json.dumps(CORE_JS.as_uri())};\n"
                      f"const a = vacuumAxis({cell}, {cart}, 2, {{ mode: 'cell', file: [null, null, [0, 0, 500]] }});\n"
                      f"const b = vacuumAxis({cell}, {cart}, 2, {{ mode: 'cell', file: [null, null, [0, 0, 2001]] }});\n"
                      "console.log(JSON.stringify([a.source, a.vec[2], b.source, b.vec[2]]));\n")
    r = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    src_a, za, src_b, zb = json.loads(r.stdout)
    assert src_a != "file" and za > 1000.5 and (src_b, zb) == ("file", 2001)
    pv = S.vacuum_vector(cell, cart, 2, "cell", file_vectors=[None, None, [0, 0, 500]])
    assert za == pytest.approx(float(pv[2]))


def test_a_saved_slab_extxyz_opens_as_the_same_slab(tmp_path):
    """Save As / Export extXYZ of a slab wrote the drawing box (the slab's thickness) as Lattice: reopened,
    the file's vacuum was the thickness and its repeats overlapped. The vacuum vectors are written."""
    from mace.gui import saving
    s, p = _slab_session()

    class Sess:
        atoms, name, doc_meta = s.atoms, "g", {}
    for fmt in ("extxyz", "xyz"):
        f = tmp_path / f"g.{fmt}"
        f.write_bytes(saving.render(Sess, fmt))
        q = S.read_any(f, "extxyz")                   # Save As .xyz writes the extended header too
        assert q["pbc"] == [True, True, False] and q["dimensionality"] == "SLAB"
        assert q["vacuum_vectors"][2] == pytest.approx([0, 0, 500.0])
        assert np.asarray(q["cell"])[2] == pytest.approx(np.asarray(p["cell"])[2])        # the drawing box again
        assert q["origin"] == pytest.approx(p["origin"])
        s2 = E.EditSession(E.atoms_from_payload(q), "g2")
        s2.apply("supercell", {"n": [1, 1, 3], "vacuum": q["vacuum_vectors"]})
        assert _min_distance(s2.atoms) > 1.0


def _mace(*args, cwd):
    return subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", *args], capture_output=True,
                          text=True, timeout=600, cwd=str(cwd))


def test_mace_supercell_extxyz_round_trip_keeps_the_vacuum(tmp_path):
    import ase.io
    shutil.copy(SLAB_OUT, tmp_path / "g.out")
    r = _mace("supercell", "g.out", "2", "2", "1", "--format", "extxyz", "-o", "s1.extxyz", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    a = ase.io.read(tmp_path / "s1.extxyz")
    assert list(a.pbc) == [True, True, False] and a.cell.array[2][2] == pytest.approx(500.0)
    q = S.read_any(tmp_path / "s1.extxyz", "extxyz")
    assert q["dimensionality"] == "SLAB" and q["vacuum_vectors"][2] == pytest.approx([0, 0, 500.0])
    r = _mace("supercell", "s1.extxyz", "1", "1", "3", "--vacuum", "file", "--format", "extxyz", "-o", "s2.extxyz",
              cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    b = ase.io.read(tmp_path / "s2.extxyz")
    assert len(b) == 3 * len(a) and _min_distance(b) == pytest.approx(_min_distance(a))
    assert b.cell.array[2][2] == pytest.approx(1500.0)


# ---------------------------------------------------------------- polyhedra: one rule, two languages

def _structures_for_polyhedra():
    from ase.spacegroup import crystal
    pto = Atoms("BaTiO3", cell=np.eye(3) * 4.0, pbc=True,
                scaled_positions=[[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
    ch4 = Atoms("CH4", positions=[[0, 0, 0], [.63, .63, .63], [-.63, -.63, .63], [-.63, .63, -.63], [.63, -.63, -.63]],
                cell=np.eye(3) * 12, pbc=True)
    return {"BaTiO3": pto, "SiC": bulk("SiC", "zincblende", a=4.36), "ZnS": bulk("ZnS", "zincblende", a=5.41),
            "Si": bulk("Si", "diamond", a=5.43), "MgO": bulk("MgO", "rocksalt", a=4.212), "CH4": ch4,
            "AgCl": bulk("AgCl", "rocksalt", a=5.78),
            "SrTiO3": crystal(["Sr", "Ti", "O"], [(0, 0, 0), (.5, .5, .5), (.5, .5, 0)], spacegroup=221, cellpar=[3.905] * 3 + [90] * 3),
            "CsPbI3": crystal(["Pb", "I", "Cs"], [(0, 0, 0), (.5, 0, 0), (.5, .5, .5)], spacegroup=221, cellpar=[6.289] * 3 + [90] * 3),
            "rutile": crystal(["Ti", "O"], [(0, 0, 0), (.3048, .3048, 0)], spacegroup=136, cellpar=[4.594, 4.594, 2.959, 90, 90, 90]),
            # forsterite (Pnma): Mg and Si 2.7 Å apart, a cation-cation contact that is no vertex
            "Mg2SiO4": crystal(["Mg", "Mg", "Si", "O", "O", "O"], [(0, 0, 0), (.2774, .25, .9915), (.094, .25, .4262), (.0913, .25, .7657),
                                                                   (.4474, .25, .2215), (.1628, .0331, .2777)],
                               spacegroup=62, cellpar=[10.207, 5.98, 4.756, 90, 90, 90])}


def test_polyhedron_centers_rule():
    """Cations coordinated by anions, as VESTA draws them: BO6 octahedra of a perovskite without its A site
    (SrO12, BaO12 share faces with the smaller TiO6) or any anion; never an intermetallic's or C-H carbon."""
    got = {k: sorted(set(a.get_chemical_symbols()[i] for i in S.polyhedron_centers(a)))
           for k, a in _structures_for_polyhedra().items()}
    assert got == {"BaTiO3": ["Ti"], "SiC": ["Si"], "ZnS": ["Zn"], "Si": [], "MgO": ["Mg"], "CH4": [],
                   "AgCl": ["Ag"], "SrTiO3": ["Ti"], "CsPbI3": ["Pb"], "rutile": ["Ti"], "Mg2SiO4": ["Mg", "Si"]}


def test_polyhedron_ligands_and_spacers():
    st = _structures_for_polyhedra()
    for name, cation, cn in (("SrTiO3", "Ti", 6), ("rutile", "Ti", 6), ("CsPbI3", "Pb", 6), ("Mg2SiO4", "Si", 4), ("Mg2SiO4", "Mg", 6)):
        a = st[name]
        nums, sym = [int(z) for z in a.numbers], a.get_chemical_symbols()
        adj = S.bonded_shells(a, shifts=True)
        for i in S.polyhedron_centers(a):
            if sym[i] != cation:
                continue
            lig = S.polyhedron_ligands(nums, adj, i)
            assert len(lig) == cn and {sym[j] for j, _, _ in lig} == ({"I"} if name == "CsPbI3" else {"O"}), (name, i, lig)
    a = st["SrTiO3"]
    nums = [int(z) for z in a.numbers]
    adj = S.bonded_shells(a, shifts=True)
    assert S.polyhedron_spacers(nums, adj) == {38}                     # Sr: the A site
    assert sorted(set(nums[i] for i in S.polyhedron_candidates(nums, adj))) == [22, 38]
    # the user's choice draws the A site too, and an empty choice none
    assert sorted(a.get_chemical_symbols()[i] for i in S.polyhedron_centers(a, only=[38, 22])) == ["Sr", "Ti"]
    assert S.polyhedron_centers(a, only=[]) == []


def test_perovskite_polyhedra_analysis_lists_bo6_only():
    from mace.analysis import polyhedra
    from mace.analysis.structure import Structure
    for name in ("SrTiO3", "rutile"):
        a = _structures_for_polyhedra()[name]
        st = Structure(cell=a.cell.array, frac=a.get_scaled_positions(), numbers=a.numbers)
        r = polyhedra.analyze(st, shapes=False)
        assert {(p["element"], p["cn"], tuple(p["ligands"])) for p in r["polyhedra"]} == {("Ti", 6, ("O",))}, (name, r["polyhedra"])


def test_polyhedron_centers_match_core_js():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    cases = {k: {"cell": a.cell.array.tolist(), "frac": a.get_scaled_positions().ravel().tolist(),
                 "z": [int(x) for x in a.numbers], "radii": S._bond_radii(a.numbers).tolist(),
                 "metal": [int(x) for x in S._is_metal(a.numbers)]} for k, a in _structures_for_polyhedra().items()}
    code = (f"import('{CORE_JS.as_uri()}').then((G) => {{ const cases = JSON.parse(process.argv[1]); const out = {{}};"
            "for (const [k, c] of Object.entries(cases)) { const adj = G.bondedShells(c.cell, Float64Array.from(c.frac), "
            "Float32Array.from(c.radii), Uint8Array.from(c.metal), 1.15); const ctr = G.polyhedronCenters(c.z, adj);"
            "out[k] = [...ctr.keys()].filter((i) => ctr[i]); } console.log(JSON.stringify(out)); });")
    out = subprocess.run([node, "--input-type=module", "-e", code, json.dumps(cases)], capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 0, out.stderr
    js = json.loads(out.stdout)
    for k, a in _structures_for_polyhedra().items():
        assert js[k] == S.polyhedron_centers(a), k


def test_polyhedra_draw_once_inside_the_range():
    """core.js polyhedra with the drawn range: a corner Ti of rutile draws one octahedron, not one at each of the
    eight corners its boundary copies sit on; every vertex is returned once (the O both octahedra share too)."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    a = _structures_for_polyhedra()["rutile"]
    case = {"cell": a.cell.array.tolist(), "frac": a.get_scaled_positions().ravel().tolist(), "z": [int(x) for x in a.numbers],
            "radii": S._bond_radii(a.numbers).tolist(), "metal": [int(x) for x in S._is_metal(a.numbers)]}
    code = (f"import('{CORE_JS.as_uri()}').then((G) => {{ const c = JSON.parse(process.argv[1]);"
            "const s = { cell: c.cell, frac: [], pbc: [true, true, true] }; for (let i = 0; i < c.z.length; i++) s.frac.push(c.frac.slice(3 * i, 3 * i + 3));"
            "const d = G.displayList(s, [1, 1, 1], true); const args = [c.cell, Float64Array.from(c.frac), Uint8Array.from(c.z), "
            "Float32Array.from(c.radii), Uint8Array.from(c.metal), 1.15, d.cart, Uint32Array.from(d.base), null, [true, true, true]];"
            "const all = G.polyhedra(...args), once = G.polyhedra(...args, { shift: d.shift, lo: [0, 0, 0], hi: [1, 1, 1] });"
            "console.log(JSON.stringify({ n: d.n, all: all.count, once: once.count, verts: once.vertZ.length, centers: once.centers,"
            " cand: once.candidates, auto: once.auto, ligands: once.ligands })); });")
    out = subprocess.run([node, "--input-type=module", "-e", code, json.dumps(case)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    r = json.loads(out.stdout)
    assert r["all"] == 9 and r["once"] == 2, r                       # 8 corner copies + the body center, then 2
    assert r["verts"] == 11 and r["centers"] == [22] and r["auto"] == [22] and r["ligands"] == [8], r


# ---------------------------------------------------------------- fort.34 screw axes (3_1 / 3_2)

def _f34_text(lat, rots, trans, numbers, pos, row=True, declared=None):
    lines = ["   3   1   1"] + ["".join(f"{x:20.12E}" for x in v) for v in lat] + [f"{len(rots):4d}"]
    for R, t in zip(rots, trans):
        M = R.T if row else R
        lines += ["".join(f"{x:20.12E}" for x in r) for r in M] + ["".join(f"{x:20.12E}" for x in t)]
    lines.append(f"{len(numbers):4d}")
    lines += [f"{z:4d}" + "".join(f"{x:20.12E}" for x in p) for z, p in zip(numbers, pos)]
    if declared:
        lines.append(f"{declared:4d}{len(rots):4d}")
    return "\n".join(lines) + "\n"


def _trigonal_se():
    """A P3_1 (144) cell: one atom in the asymmetric unit, three in the cell. (In P3_121 the 2-fold axes
    already refuse the wrong reading; with the screw alone both readings close, as 144 and 145.)"""
    import spglib
    from ase.spacegroup import crystal
    a = crystal("Se", [(0.2254, 0.1, 0.3)], spacegroup=144, cellpar=[4.366, 4.366, 4.954, 90, 90, 120])
    lat = a.cell.array
    sym = spglib.get_symmetry((lat, a.get_scaled_positions(), a.numbers))
    rots = [lat.T @ R @ np.linalg.inv(lat.T) for R in sym["rotations"]]
    trans = [t @ lat for t in sym["translations"]]
    return lat, rots, trans, (np.array([[0.2254, 0.1, 0.3]]) @ lat)


@pytest.mark.parametrize("row", [True, False])
def test_fort34_screw_axis_handedness_follows_the_declared_group(tmp_path, row):
    from mace.utils import fort34
    lat, rots, trans, pos = _trigonal_se()
    p = tmp_path / "se.f34"
    p.write_text(_f34_text(lat, rots, trans, [34], pos, row=row, declared=144))
    f = fort34.read(p)
    assert len(f["numbers"]) == 3 and f["convention"] == ("row" if row else "column")
    assert not f.get("convention_ambiguous")
    assert fort34._spacegroup_number(f["cart"], f["numbers"], lat) == 144
    # without the group record the two enantiomorphs (144, 145) cannot be told apart: said, not hidden
    p.write_text(_f34_text(lat, rots, trans, [34], pos, row=row))
    f = fort34.read(p)
    assert f["convention_ambiguous"] and sorted(f["groups"].values()) == [144, 145]
    s = S.read_structure(p)
    assert "enantiomorphs" in s["source"]["warning"]


def test_slab_cell_edit_scales_in_plane_and_keeps_heights():
    s, _ = _slab_session()
    s.apply("add", {"element": "H", "cart": (s.atoms.positions[0] + [0, 0, 1.2]).tolist()})
    z0, f0 = s.atoms.positions[:, 2].copy(), s.atoms.get_scaled_positions(wrap=False)[:, :2].copy()
    lat = s.atoms.cell.cellpar()
    s.apply("set_cell", {"cellpar": [lat[0] * 1.1, lat[1] * 1.1, lat[2] * 3, lat[3], lat[4], lat[5]]})
    assert np.allclose(s.atoms.positions[:, 2], z0)                       # c is the drawing box: no stretch
    assert np.allclose(s.atoms.get_scaled_positions(wrap=False)[:, :2], f0)  # a, b scale the atoms
