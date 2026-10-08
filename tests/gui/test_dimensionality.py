"""Molecules, slabs and polymers (plan E10): the vacuum hint, the orientation
CRYSTAL's SLAB / POLYMER / MOLECULE decks need, and Setup writing such decks
through the real CLI (the answer driver, cif2d12) from a molecule file.

The skewed-cell cases are small generic systems built here (a water molecule
in a monoclinic box, a polyethylene-like chain along c of a triclinic cell, a
graphene-like layer with its vacuum along a): after orienting, the deck the
CLI writes puts every atom where it truly is, up to one rigid motion (Kabsch
RMSD < 1e-6 Å, all distances kept). Corpus CIFs (copied, neutral names) check
the hint on real files; skipped without the corpus."""
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U
import setup_routes as R

np = pytest.importorskip("numpy")
ase = pytest.importorskip("ase")
CIFS = H.REPO_ROOT / "test" / "CIFs"


def _kabsch_rmsd(P, Q):
    P = P - P.mean(0)
    Q = Q - Q.mean(0)
    U_, S, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(U_ @ Vt))
    D = np.diag([1, 1, d])
    Rm = U_ @ D @ Vt
    return float(np.sqrt(((P @ Rm - Q) ** 2).sum(1).mean()))


def _dists(X):
    return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)


def water_monoclinic():
    from ase import Atoms
    from ase.geometry import cellpar_to_cell
    cell = cellpar_to_cell([15.0, 16.0, 17.0, 90, 100, 90])
    pos = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2399872, 0.927, 0.0]]) + [6.0, 7.0, 8.0]
    return Atoms("OH2", positions=pos, cell=cell, pbc=True)


def chain_triclinic():
    """A zig-zag CH2 chain along c of a triclinic cell, vacuum along a and b."""
    from ase import Atoms
    from ase.geometry import cellpar_to_cell
    c_len = 2.55
    cell = cellpar_to_cell([18.0, 19.0, c_len, 80, 85, 75])
    cdir = cell[2] / np.linalg.norm(cell[2])
    side = np.cross(cdir, [1.0, 0.0, 0.0])
    side /= np.linalg.norm(side)
    up = np.cross(cdir, side)
    o = 0.5 * cell[0] + 0.5 * cell[1]
    pts = [o + 0.0 * cdir + 0.42 * side, o + 0.5 * c_len * cdir - 0.42 * side]
    hs = []
    for k, p in enumerate(pts):
        s = 1 if k == 0 else -1
        hs += [p + s * 0.63 * side + 0.89 * up, p + s * 0.63 * side - 0.89 * up]
    return Atoms("C2H4", positions=np.array(pts + hs), cell=cell, pbc=True)


def layer_vacuum_along_a():
    """A graphene-like layer spanning b and c, with 20 Å of vacuum along a (skewed)."""
    from ase import Atoms
    a_g = 2.46
    b = np.array([0.0, a_g, 0.0])
    c = np.array([0.0, -a_g / 2, a_g * np.sqrt(3) / 2])
    a = np.array([20.0, 1.5, 0.7])
    cell = np.array([a, b, c])
    frac = np.array([[0.5, 1 / 3, 2 / 3], [0.5, 2 / 3, 1 / 3]])
    return Atoms("C2", positions=frac @ cell, cell=cell, pbc=True)


def test_hint_and_orientation_are_rigid():
    from mace.gui import structures as S
    for atoms, dim in ((water_monoclinic(), "MOLECULE"), (chain_triclinic(), "POLYMER"), (layer_vacuum_along_a(), "SLAB")):
        h = S.dimensionality_hint(atoms.cell.array, atoms.positions, [True] * 3)
        assert h["dim"] == dim, (dim, h)
        r = S.standardize_for_dim(atoms.numbers, atoms.positions, atoms.cell.array, [True] * 3, dim)
        assert _kabsch_rmsd(r["cart"], atoms.positions) < 1e-9
        assert np.abs(_dists(r["cart"]) - _dists(atoms.positions)).max() < 1e-9
        cell = r["cell"]
        if dim == "MOLECULE":
            assert np.allclose(cell, np.diag(np.diag(cell)))
        if dim == "SLAB":
            assert abs(np.dot(cell[2], cell[0])) < 1e-9 and abs(np.dot(cell[2], cell[1])) < 1e-9
            assert abs(cell[0][1]) < 1e-12 and abs(cell[0][2]) < 1e-12 and abs(cell[1][2]) < 1e-12
        if dim == "POLYMER":
            assert np.allclose(cell, np.diag(np.diag(cell)))
    # a slab whose vacuum is not along one cell vector is refused, with the reason
    w = water_monoclinic()
    with pytest.raises(S.OrientError, match="vacuum is not along one cell vector"):
        S.standardize_for_dim(w.numbers, w.positions, w.cell.array, [True] * 3, "SLAB")


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    from ase.io import write
    tmp = tmp_path_factory.mktemp("dimensionality")
    root = H.examples_copy(tmp)
    d = root / "dim"
    d.mkdir()
    write(str(d / "water_mono.cif"), water_monoclinic())
    write(str(d / "chain.cif"), chain_triclinic())
    write(str(d / "layer_a.cif"), layer_vacuum_along_a())
    w = water_monoclinic()
    w.pbc = False
    w.cell = None
    write(str(d / "water.xyz"), w, format="xyz")
    if CIFS.is_dir():
        shutil.copyfile(CIFS / "1LiFSI-1DEC-conf1_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized.cif", d / "mol_box.cif")
        shutil.copyfile(CIFS / "4LG_2x2_AA_opt_HSESOL3C_optimized.cif", d / "slab_box.cif")
        shutil.copyfile(CIFS / "3,4^2T1-CA_rev1.cif", d / "molecular_crystal.cif")
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1")
    yield root, base, tmp
    proc.terminate()
    proc.wait(timeout=15)


def setup_run(base, body):
    st, _, r = U.post(base, "api/setup/run", {"preview": True, "wait": 300, **body}, timeout=400)
    assert st == 200, r
    assert r.get("deck"), r.get("refused") or r.get("error") or (r.get("log") or "")[-2000:]
    return r


def deck_cart(deck_text, tmp=None):
    """The Cartesian positions CRYSTAL builds from a P1 SLAB / POLYMER / MOLECULE deck's
    geometry block (layer group 1: a b gamma; rod group 1: a): MOLECULE x y z; POLYMER
    x a, y, z; SLAB x A + y B, z."""
    lines = [x.strip() for x in deck_text.splitlines()]
    dim = next(x for x in lines[:4] if x in ("SLAB", "POLYMER", "MOLECULE"))
    k = lines.index(dim)
    group = int(lines[k + 1].split()[0])
    assert group == 1, f"P1 decks only here, not group {group}"
    if dim == "MOLECULE":
        cell_vals, k = [], k + 2
    else:
        cell_vals, k = [float(v) for v in lines[k + 2].split()], k + 3
    n = int(lines[k])
    rows = [[float(v) for v in lines[k + 1 + i].split()[1:4]] for i in range(n)]
    X = np.array(rows)
    if dim == "MOLECULE":
        return X, dim
    if dim == "POLYMER":
        return np.column_stack([X[:, 0] * cell_vals[0], X[:, 1], X[:, 2]]), dim
    a, b, gamma = cell_vals[0], cell_vals[1], np.radians(cell_vals[2])
    A, B = np.array([a, 0, 0]), np.array([b * np.cos(gamma), b * np.sin(gamma), 0])
    return np.outer(X[:, 0], A) + np.outer(X[:, 1], B) + np.outer(X[:, 2], [0, 0, 1]), dim


@pytest.mark.parametrize("name,dim,atoms", [("water_mono.cif", "MOLECULE", water_monoclinic),
                                            ("chain.cif", "POLYMER", chain_triclinic),
                                            ("layer_a.cif", "SLAB", layer_vacuum_along_a)])
def test_cli_deck_puts_atoms_where_they_are(served, tmp_path, name, dim, atoms):
    root, base, _ = served
    r = setup_run(base, {"source": {"path": f"dim/{name}"}, "calc_type": "SP", "answers": {"dimensionality": dim, "symmetry": "P1"}})
    lines = [x.strip() for x in r["deck"].splitlines()]
    assert dim in lines[:4], lines[:6]
    cart, _ = deck_cart(r["deck"])
    ref = atoms()
    true = ref.positions
    assert len(cart) == len(true)
    if dim == "MOLECULE":
        assert _kabsch_rmsd(cart, true) < 1e-5, _kabsch_rmsd(cart, true)
        assert np.abs(_dists(cart) - _dists(true)).max() < 1e-5
        return
    # periodic: the CLI may shift and wrap atoms along the periodic directions (any image is the same
    # crystal), so every minimum-image distance must be the true one
    lines = [x.strip() for x in r["deck"].splitlines()]
    k = lines.index(dim)
    vals = [float(v) for v in lines[k + 2].split()]
    if dim == "POLYMER":
        deck_vecs = [np.array([vals[0], 0, 0])]
        true_vecs = [ref.cell.array[2]]
    else:
        g = np.radians(vals[2])
        deck_vecs = [np.array([vals[0], 0, 0]), np.array([vals[1] * np.cos(g), vals[1] * np.sin(g), 0])]
        true_vecs = [ref.cell.array[1], ref.cell.array[2]]
    assert np.allclose(sorted(np.linalg.norm(v) for v in deck_vecs), sorted(np.linalg.norm(v) for v in true_vecs), atol=1e-5)

    def mic(X, vecs):
        shifts = [np.zeros(3)]
        for v in vecs:
            shifts = [s + n * v for s in shifts for n in (-1, 0, 1)]
        D = np.full((len(X), len(X)), np.inf)
        for s in shifts:
            D = np.minimum(D, np.linalg.norm(X[:, None, :] - X[None, :, :] + s, axis=-1))
        return D
    assert np.abs(mic(cart, deck_vecs) - mic(true, true_vecs)).max() < 1e-5


def test_molecule_file_sets_up_as_molecule_like_the_cli(served, tmp_path):
    """An xyz (no cell) opens to Setup as MOLECULE, and the preview is the CLI's deck on
    the same oriented CIF with the same answers (manual route: mace convert, asked)."""
    root, base, _ = served
    r = setup_run(base, {"source": {"path": "dim/water.xyz"}, "calc_type": "OPT", "name": "water"})        # the CLI menu default
    assert "MOLECULE" in [x.strip() for x in r["deck"].splitlines()[:4]]
    # the same oriented CIF, made here by the same code, through the CLI asked by hand
    from ase.io import read
    from mace.gui import structures as S, saving
    from mace.gui.server import _SessLike
    from ase import Atoms
    w = read(str(root / "dim/water.xyz"))
    o = S.standardize_for_dim(w.numbers, w.positions, np.zeros((3, 3)), [False] * 3, "MOLECULE")
    cif = saving.render(_SessLike(Atoms(numbers=w.numbers, cell=o["cell"], positions=o["cart"], pbc=True), "water"), "cif", "p1")
    m = tmp_path / "m"
    (m / "cif").mkdir(parents=True)
    (m / "cif" / "water.cif").write_bytes(cif)
    answers = [("use these default settings? [Y/n]", "n"), ("default symmetry settings? [Y/n]", ""),
               ("Select system dimensionality", "4")]
    proc = R.run_cli_manual(["convert", "--cif_dir", "cif", "--output_dir", "m_out"], answers, m)
    decks = sorted((m / "m_out").glob("*.d12"))
    assert proc.returncode == 0 and decks, (proc.stdout + proc.stderr)[-3000:]
    import difflib
    assert r["deck_name"] == decks[0].name
    assert r["deck"] == decks[0].read_text(), "\n".join(difflib.unified_diff(decks[0].read_text().splitlines(),
                                                                           r["deck"].splitlines(), "cli", "gui", lineterm=""))


@pytest.mark.skipif(not CIFS.is_dir(), reason="test/ corpus not present (gitignored)")
def test_corpus_hints(served):
    root, base, _ = served
    for name, dim in (("mol_box.cif", "MOLECULE"), ("slab_box.cif", "SLAB"), ("molecular_crystal.cif", "CRYSTAL")):
        st, _, s = U.req(base, f"api/structure?path=dim/{name}")
        assert st == 200 and s["dim_hint"]["dim"] == dim, (name, s.get("dim_hint"))


def test_wrapped_molecule_and_chain_are_made_whole():
    """A CIF stores a molecule (or a chain's cross-section) wrapped into its box: across the
    box's faces the atoms sit on opposite sides. MOLECULE and POLYMER decks get them whole
    again (lattice translations only), so every bond keeps its length; the box is sized by
    the whole molecule, not by the box it was wrapped in."""
    from mace.gui import structures as S
    cell = np.eye(3) * 15.0
    water = np.array([[0.0, 0.0, 0.1], [0.757, 0.0, -0.486], [-0.757, 0.0, -0.486]])
    wrapped = np.mod(water, 15.0)
    assert np.linalg.norm(wrapped[1] - wrapped[0]) > 10            # broken as stored
    r = S.standardize_for_dim([8, 1, 1], wrapped, cell, [True] * 3, "MOLECULE")
    assert np.abs(_dists(r["cart"]) - _dists(water)).max() < 1e-9
    assert r["cell"][0, 0] < 20                                    # extent + 2 x gap, not the wrapped span
    # the same molecule in the middle of its box: the same deck geometry as before this fix
    mid = S.standardize_for_dim([8, 1, 1], water + 7.5, cell, [True] * 3, "MOLECULE")
    assert np.allclose(mid["cart"], r["cart"]) and np.allclose(mid["cell"], r["cell"])
    # a zigzag chain along a, wrapped across b
    cell = np.diag([2.5, 15.0, 15.0])
    chain = np.array([[0.0, 0.2, 7.0], [1.25, -0.6, 7.0]])
    r = S.standardize_for_dim([6, 6], np.mod(chain, [2.5, 15.0, 15.0]), cell, [True] * 3, "POLYMER")
    assert abs(np.linalg.norm(r["cart"][1] - r["cart"][0]) - np.linalg.norm(chain[1] - chain[0])) < 1e-9
    assert r["cell"][1, 1] < 10                                    # 0.8 A across + the gap
