"""`mace transform`: the GUI's Rotate and Move gizmos from the command line (1.1.6). Run through mace_cli as a
subprocess: an exact turn about a Cartesian axis, a cell axis and a lattice direction through the picked atoms'
centroid (made whole across the cell's faces, as the GUI does), exact displacements in Å and in fractions of
the cell, picking atoms by index and element, and clear errors."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase.io import read  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MGO = REPO / "examples" / "gui" / "structures" / "MgO.cif"


def mace(*args, cwd):
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "transform", *map(str, args)],
                       cwd=str(cwd), env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout + r.stderr


def payload(path):
    sys.path.insert(0, str(REPO))
    from mace.gui.structures import read_structure
    return read_structure(Path(path))


def wrapped_diff(a, b, cell):
    f = (np.asarray(a) - np.asarray(b)) @ np.linalg.inv(cell)
    f -= np.round(f)
    return np.abs(f @ cell).max()


def test_turn_about_z_through_the_made_whole_centroid(tmp_path):
    p0 = payload(MGO)
    cell, cart = np.array(p0["cell"]), np.array(p0["cart"])
    out = tmp_path / "rot.cif"
    rc, text = mace(MGO, "--atoms", "1,5", "--rotate", 90, "--axis", "z", "-o", out, cwd=tmp_path)
    assert rc == 0, text
    assert "turned 2 atoms by 90° about z" in text and f"Wrote {out}" in text
    got = np.array(payload(out)["cart"])
    f = np.array(p0["frac"])
    d = f[4] - f[0]
    p_a, p_b = cart[0], (f[0] + d - np.round(d)) @ cell
    piv = (p_a + p_b) / 2
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    want = [piv + R @ (p_a - piv), piv + R @ (p_b - piv)]
    # the CIF keeps the atom order; each turned atom is found where it should be
    for w in want:
        assert min(wrapped_diff(g, w, cell) for g in got) < 1e-5
    for i in (1, 2, 3, 5, 6, 7):
        assert min(wrapped_diff(g, cart[i], cell) for g in got) < 1e-5


def test_lattice_direction_and_cell_axis_turns_match_rodrigues(tmp_path):
    from mace.structure_transform import axis_vector, rotation_matrix, transform
    p0 = payload(MGO)
    cell = np.array(p0["cell"])
    u = axis_vector("1 1 0", cell, [True] * 3)
    assert np.allclose(u, [2 ** -0.5, 2 ** -0.5, 0])
    assert np.allclose(axis_vector("111", cell, [True] * 3), np.ones(3) / 3 ** 0.5)
    assert np.allclose(axis_vector("c", cell, [True] * 3), [0, 0, 1])
    R = rotation_matrix(u, 120)
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
    r = transform(p0, atoms="all", rotate=180, axis="c", about="origin", wrap=False)
    got, c0 = np.array(r["cart"]), np.array(p0["cart"])
    assert np.allclose(got[:, :2], -c0[:, :2]) and np.allclose(got[:, 2], c0[:, 2])


def test_translations_in_angstrom_and_fractions_and_element_pick(tmp_path):
    p0 = payload(MGO)
    f0 = np.array(p0["frac"])
    o_idx = [i for i, s in enumerate(p0["symbols"]) if s == "O"]
    out = tmp_path / "moved.cif"
    rc, text = mace(MGO, "--atoms", "O", "--translate-frac", 0.1, 0, 0, "--no-wrap", "-o", out, cwd=tmp_path)
    assert rc == 0, text
    f1 = np.array(payload(out)["frac"])
    # read back: the O atoms moved by 0.1 a (ase may wrap a CIF on reading: compare modulo 1)
    moved = {tuple(np.round((f0[i] + [0.1, 0, 0]) % 1, 6)) for i in o_idx}
    have = {tuple(np.round(x % 1, 6)) for x in f1}
    assert moved <= have, (moved, have)
    out2 = tmp_path / "moved2.cif"
    rc, text = mace(MGO, "--atoms", "2-3", "--translate", 0.25, -0.1, 0, "-o", out2, cwd=tmp_path)
    assert rc == 0, text
    c0, cell = np.array(p0["cart"]), np.array(p0["cell"])
    c2 = np.array(payload(out2)["cart"])
    for i in (1, 2):
        w = c0[i] + [0.25, -0.1, 0]
        assert min(wrapped_diff(g, w, cell) for g in c2) < 1e-5


@pytest.mark.parametrize("args, message", [
    (("--rotate", "30"), "--rotate needs --axis"),
    (("--atoms", "99", "--translate", "0", "0", "1"), "the structure has atoms 1 to 8"),
    (("--atoms", "Zr", "--translate", "0", "0", "1"), "no such atom index or element"),
    (("--rotate", "30", "--axis", "0 0 0"), "not all zero"),
    ((), "nothing to do"),
    (("--rotate", "30", "--axis", "z", "--about", "1,2"), "--about"),
])
def test_errors_are_clear(tmp_path, args, message):
    rc, text = mace(MGO, *args, "-o", tmp_path / "x.cif", cwd=tmp_path)
    assert rc == 2 and message in text, text
    assert not (tmp_path / "x.cif").exists()


def test_refuses_to_overwrite_without_force(tmp_path):
    out = tmp_path / "o.cif"
    out.write_text("keep")
    rc, text = mace(MGO, "--translate", 0, 0, 1, "-o", out, cwd=tmp_path)
    assert rc == 2 and "--force" in text and out.read_text() == "keep"
    rc, text = mace(MGO, "--translate", 0, 0, 1, "-o", out, "--force", cwd=tmp_path)
    assert rc == 0 and out.read_text() != "keep"


def test_help_names_the_command():
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME")}
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "transform", "-h"], env=env, capture_output=True, text=True,
                       stdin=subprocess.DEVNULL, timeout=120)
    assert r.returncode == 0 and "Usage: mace transform" in r.stdout + r.stderr


def test_a_slab_is_written_boxed_to_cif_and_poscar_and_with_its_vacuum_to_extxyz(tmp_path):
    """A slab moved along c was written to CIF unboxed (_cell_length_c 0.5: the drawing box): CIF and POSCAR
    get a box with vacuum (as mace supercell), an extended XYZ keeps the pbc and the file's 500 Å."""
    import shutil
    import ase.io
    from mace.gui import structures as S
    shutil.copy(REPO / "examples" / "gui" / "outputs" / "graphene_slab_sp.out", tmp_path / "g.out")
    for out in ("g_t.cif", "POSCAR_t.vasp", "g_t.extxyz"):
        code, text = mace("g.out", "--translate", 0, 0, 0.5, "-o", out, cwd=tmp_path)
        assert code == 0, text
        a = ase.io.read(tmp_path / out)
        if out.endswith(".extxyz"):
            assert list(a.pbc) == [True, True, False] and a.cell.array[2][2] == pytest.approx(500.0)
        else:
            assert a.cell.array[2][2] == pytest.approx(S.SAVE_VACUUM)
            z = a.get_scaled_positions()[:, 2]
            assert 0.05 < z.min() and z.max() < 0.95


# ------------------------------------------------------------------------------------------------- --about (1.1.6)
# Ti1, O2, O3 picked: O2 across the a face and O3 across the b face of Ti1 (made whole at x = -0.07, y = -0.15);
# C4 and H5 stay. No coordinate sits half a cell from another, so every image is unambiguous.
PIVOT_FRAC = [[0.10, 0.20, 0.30], [0.93, 0.25, 0.32], [0.15, 0.85, 0.28], [0.60, 0.55, 0.50], [0.70, 0.45, 0.65]]
PIVOT_CELL = np.diag([5.0, 6.0, 7.0])
WHOLE = np.array([[0.10, 0.20, 0.30], [-0.07, 0.25, 0.32], [0.15, -0.15, 0.28]]) @ PIVOT_CELL
M_TI, M_O = 47.867, 15.999
ABOUT = {
    "centroid": (WHOLE.mean(axis=0), "the centroid"),
    "com": ((M_TI * WHOLE[0] + M_O * WHOLE[1] + M_O * WHOLE[2]) / (M_TI + 2 * M_O), "the center of mass"),
    "cell": (np.array([2.5, 3.0, 3.5]), "the cell center"),
    "origin": (np.zeros(3), "the origin"),
    # C4, not picked: its image nearest the three's centroid (frac 0.06, 0.10, 0.30)
    "atom:4": (np.array([-0.40, 0.55, 0.50]) @ PIVOT_CELL, "C4"),
    "atom:1": (WHOLE[0], "Ti1"),
    "1.5,-2,0.25": (np.array([1.5, -2.0, 0.25]), "the point"),
}


@pytest.fixture
def pivot_cif(tmp_path):
    from ase import Atoms
    from ase.io import write
    path = tmp_path / "pivot.cif"
    write(str(path), Atoms("TiO2CH", scaled_positions=PIVOT_FRAC, cell=PIVOT_CELL, pbc=True))
    return path


@pytest.mark.parametrize("about", list(ABOUT))
def test_about_each_pivot_turns_the_atoms_about_it(tmp_path, pivot_cif, about):
    """mace transform --about centroid|com|cell|origin|atom:N|x,y,z, as the GUI's Rotate panel: 90° about z."""
    p0 = payload(pivot_cif)
    assert p0["symbols"] == ["Ti", "O", "O", "C", "H"]
    out = tmp_path / "turned.cif"
    rc, text = mace(pivot_cif, "--atoms", "1-3", "--rotate", 90, "--axis", "z", "--about", about, "-o", out, cwd=tmp_path)
    assert rc == 0, text
    piv, name = ABOUT[about]
    assert f"turned 3 atoms by 90° about z" in text and f"through {name} at" in text, text
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    want = [piv + R @ (p - piv) for p in WHOLE]
    got = np.array(payload(out)["cart"])
    assert len(got) == 5
    for k in range(3):
        assert wrapped_diff(got[k], want[k], PIVOT_CELL) < 1e-5, (about, k, got[k], want[k])
    c0 = np.array(p0["cart"])
    for k in (3, 4):
        assert wrapped_diff(got[k], c0[k], PIVOT_CELL) < 1e-5


@pytest.mark.parametrize("about, message", [("atom:6", "the structure has atoms 1 to 5"), ("atom:0", "atoms 1 to 5"),
                                            ("mass", "atom:N"), ("1,2", "--about")])
def test_about_errors_are_clear(tmp_path, pivot_cif, about, message):
    rc, text = mace(pivot_cif, "--rotate", 30, "--axis", "z", "--about", about, "-o", tmp_path / "x.cif", cwd=tmp_path)
    assert rc == 2 and message in text, text
    assert not (tmp_path / "x.cif").exists()


def test_help_lists_every_pivot():
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "transform", "-h"], env=env, capture_output=True, text=True,
                       stdin=subprocess.DEVNULL, timeout=120)
    text = " ".join((r.stdout + r.stderr).split())
    assert r.returncode == 0
    assert "--about centroid|com|cell|origin|atom:N|X,Y,Z" in text
    for word in ("com (their center of mass)", "cell (the cell's center", "atom:N (atom N, 1-based", "--about atom:12"):
        assert word in text, word


def test_about_cell_of_a_slab_is_the_middle_of_its_drawing_box(tmp_path):
    """A slab's cell center: the middle of a and b, at the atoms' mid-height along the open c (not c/2 of the
    0.5 Å drawing cell, nor 250 Å of CRYSTAL's 500 Å vacuum)."""
    import re
    import shutil
    shutil.copy(REPO / "examples" / "gui" / "outputs" / "graphene_slab_sp.out", tmp_path / "g.out")
    p0 = payload(tmp_path / "g.out")
    a, b = np.array(p0["cell"][0]), np.array(p0["cell"][1])
    rc, text = mace("g.out", "--atoms", "1", "--rotate", 180, "--axis", "z", "--about", "cell", "-o", "g_t.extxyz", cwd=tmp_path)
    assert rc == 0, text
    m = re.search(r"through the cell center at \(([-\d.]+), ([-\d.]+), ([-\d.]+)\) Å", text)
    assert m, text
    assert [float(x) for x in m.groups()] == pytest.approx(list(((a + b) / 2)[:2]) + [0.0], abs=1e-4)
    got = read(str(tmp_path / "g_t.extxyz"))
    piv = (a + b) / 2
    want = 2 * piv - np.array(p0["cart"][0])
    d = got.positions[0] - want
    f = np.linalg.solve(np.array([a, b, [0, 0, 1]]).T, d)
    f[:2] -= np.round(f[:2])
    assert np.abs(f).max() < 1e-5
