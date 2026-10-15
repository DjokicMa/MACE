"""structures.compare(): the two displacements the OPT compare view offers.

"internal" (``displacements``): the change of fractional coordinates in the
final cell, zero when only the cell changes; "cart" (``displacements_cart``):
r_final - r_initial in Angstrom with the cell change included, each final atom
at the image nearest its start, the mean shift removed. AB graphite (atoms on
special positions, c shrinking) is the case the maintainer hit: the
Displacement map drew nothing and its scale read 0.0000 to 0.0000 A."""
import numpy as np
import pytest

pytest.importorskip("ase")

from mace.gui import structures as S
import runcorpus as RC


def _struct(cell, frac, numbers=None):
    cell = np.asarray(cell, float)
    frac = np.asarray(frac, float)
    return {"cell": cell.tolist(), "cart": (frac @ cell).tolist(), "frac": frac.tolist(),
            "numbers": numbers or [6] * len(frac), "pbc": [True, True, True]}


def _hex(a, c):
    return [[a, 0, 0], [-a / 2, a * 3 ** 0.5 / 2, 0], [0, 0, c]]


# AB graphite: two layers at z = 1/4 and 3/4, every atom on a special position
GRAPHITE = [[0, 0, 0.25], [0, 0, 0.75], [1 / 3, 2 / 3, 0.25], [2 / 3, 1 / 3, 0.75]]


def test_cell_only_change_is_zero_internal_and_real_cartesian():
    c = S.compare(_struct(_hex(2.452, 6.7), GRAPHITE), _struct(_hex(2.4528, 6.1752), GRAPHITE))
    # zero up to rounding (another numpy/ase can leave ~1e-16)
    assert c["max_displacement"] < 1e-12 and all(d["length"] < 1e-12 for d in c["displacements"])
    assert c["cell_changed"] is True
    v = np.array([d["vector"] for d in c["displacements_cart"]])
    # the mean shift is removed: the layers move toward each other by half the interlayer change
    assert np.allclose(v.mean(axis=0), 0, atol=1e-12)
    half = (6.7 - 6.1752) / 4                                     # each layer by a quarter of Δc
    assert np.allclose(np.abs(v[:, 2]), half, atol=1e-3)
    assert np.sign(v[0, 2]) == -np.sign(v[1, 2])                  # the two layers close in
    assert c["max_displacement_cart"] == pytest.approx(np.linalg.norm(v, axis=1).max())
    assert c["rms_displacement_cart"] == pytest.approx(np.sqrt((np.linalg.norm(v, axis=1) ** 2).mean()))


def test_no_cell_change_cartesian_is_internal_minus_the_mean():
    cell = np.diag([5.0, 5.0, 5.0])
    f0 = np.array([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]])
    f1 = f0 + [[0.01, 0, 0], [0, 0.02, 0]]
    c = S.compare(_struct(cell, f0), _struct(cell, f1))
    assert c["cell_changed"] is False
    vi = np.array([d["vector"] for d in c["displacements"]])
    vc = np.array([d["vector"] for d in c["displacements_cart"]])
    assert np.allclose(vi, [[0.05, 0, 0], [0, 0.1, 0]])
    assert np.allclose(vc, vi - vi.mean(axis=0))


def test_wrap_across_the_cell_is_not_a_displacement():
    """Minimum image: an atom leaving through a face and coming back on the other
    side moved 0.02 in fractional terms, not 0.98."""
    cell = np.diag([4.0, 4.0, 4.0])
    f0 = np.array([[0.99, 0.5, 0.5], [0.5, 0.5, 0.5]])
    f1 = np.array([[0.01, 0.5, 0.5], [0.5, 0.5, 0.5]])
    c = S.compare(_struct(cell, f0), _struct(cell, f1))
    assert c["displacements"][0]["length"] == pytest.approx(0.08)
    v = np.array([d["vector"] for d in c["displacements_cart"]])
    assert np.allclose(v, [[0.04, 0, 0], [-0.04, 0, 0]])         # 0.08 Å apart, mean removed
    assert c["max_displacement_cart"] == pytest.approx(0.04)


def test_origin_does_not_matter_for_cartesian():
    """The same strain with the structure moved in the cell gives the same
    Cartesian displacements (the mean shift is what an origin choice changes)."""
    a0, a1 = _hex(2.452, 6.7), _hex(2.4528, 6.1752)
    sh = np.array([0.1, 0.2, 0.05])
    c1 = S.compare(_struct(a0, GRAPHITE), _struct(a1, GRAPHITE))
    c2 = S.compare(_struct(a0, np.array(GRAPHITE) + sh), _struct(a1, np.array(GRAPHITE) + sh))
    v1 = np.array([d["vector"] for d in c1["displacements_cart"]])
    v2 = np.array([d["vector"] for d in c2["displacements_cart"]])
    assert np.allclose(v1, v2, atol=1e-12)


def test_real_corpus_opts():
    """Ag1Br1: only the cell relaxes (internal zero, Cartesian not); Ag2Cl1: the atoms relax too."""
    fixed = RC.CORPUS / "OPT" / "Ag1Br1_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized.out"
    moved = RC.CORPUS / "OPT" / "Ag2Cl1_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized.out"
    for p in (fixed, moved):
        if not p.is_file():
            pytest.skip("test/ corpus not present (gitignored)")
    c = S.trajectory(fixed)["comparison"]
    assert c["cell_changed"] and c["max_displacement"] < 1e-6 and c["max_displacement_cart"] > 1e-3
    c = S.trajectory(moved)["comparison"]
    assert c["cell_changed"] and c["max_displacement"] > 1e-2 and c["max_displacement_cart"] > 1e-2


def test_molecule_has_no_cell_choice():
    """A molecule takes the aligned Cartesian path (rigid motion removed): no cell, so no
    displacements_cart and no "relative to the cell" choice in the GUI."""
    x0 = np.array([[0, 0, 0], [1.1, 0, 0], [0, 1.1, 0], [0, 0, 1.1]], float)
    x1 = x0 + [3.0, -1.0, 2.0]                                       # a rigid shift only
    x1[1, 0] += 0.05                                                 # and one bond longer
    mol = lambda x: {"cell": np.eye(3).tolist(), "cart": x.tolist(), "frac": None, "numbers": [6, 1, 1, 1],
                     "pbc": [False, False, False]}
    c = S.compare(mol(x0), mol(x1))
    assert c["aligned"] == "molecule" and c["cell"] == [] and "displacements_cart" not in c
    assert not c.get("cell_changed")
    assert max(d["length"] for d in c["displacements"]) < 0.06      # the shift is not a displacement
