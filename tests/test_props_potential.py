"""CHARGE+POTENTIAL derived properties (mace/properties/defs/charge_potential.py,
mace/properties/volumetric.py): the potential along the stacking axis, the
vacuum levels and work functions of a slab, the cell dipole and its
correction, and MACE's own Bader partition.

Checked against CRYSTAL's own numbers: the dipole moment it prints for the
parent SCF of the same molecule or slab, the Fermi energy; the Bader
partition against an analytic zero-flux partition (steepest-ascent paths
of the analytic density) and against the symmetry of real molecules."""
import contextlib
import io
import itertools
import math
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO_ROOT, TEST_DATA
from mace.properties import registry
from mace.properties import volumetric as V
from mace.utils.property_extractor import CrystalPropertyExtractor

ECH = TEST_DATA / "ECH3POT3"


def _corpus(name: str) -> Path:
    p = ECH / name
    if not p.is_file():
        pytest.skip("test/ corpus not present")
    return p


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    from mace import examples_suite as ES
    root = tmp_path_factory.mktemp("props_pot") / "suite"
    ES.copy_suite(root)
    return root


def derived(out: Path, only):
    """The derived properties ``only`` of one run, as the extractor runs them."""
    x = CrystalPropertyExtractor(db_path=None, enable_tracking=False)
    x.run_derived = False
    with contextlib.redirect_stdout(io.StringIO()):
        props = x.extract_all_properties(Path(out))
    text = Path(out).read_text(errors="replace")
    vals, rep = registry.run_derived(Path(out), text, {k: v for k, v in props.items() if not k.startswith("_")},
                                     calc_type=props.get("calculation_type"), only=only,
                                     dimensionality=CrystalPropertyExtractor._dimensionality(text))
    return vals, rep


def printed_dipole(out: Path):
    from mace.properties.defs.scf_summary import printed_dipole as pd
    return pd(out.read_text(errors="replace"))


# ------------------------------------------------------------------ Bader against an analytic partition
def _promolecule(cell, atoms, terms, pbc):
    imgs = [np.array(s) @ cell for s in itertools.product(*[(-1, 0, 1) if p else (0,) for p in pbc])]

    def rho_grad(x):
        r = np.zeros(len(x))
        g = np.zeros_like(x)
        for R, tt in zip(atoms, terms):
            for s in imgs:
                d = x - (R + s)
                dist = np.linalg.norm(d, axis=1) + 1e-12
                for c, a in tt:
                    v = c * np.exp(-a * dist)
                    r += v
                    g += (-a * v / dist)[:, None] * d
        return r, g
    return rho_grad, imgs


def _exact_labels(pts, rho_grad, atoms, imgs, step=0.02, stop=0.05, max_steps=3000):
    x = pts.copy()
    lab = -np.ones(len(x), int)
    act = np.arange(len(x))
    for _ in range(max_steps):
        if not act.size:
            break
        _, gr = rho_grad(x[act])
        x[act] += step * gr / (np.linalg.norm(gr, axis=1, keepdims=True) + 1e-30)
        best = np.full(len(act), np.inf)
        bi = np.zeros(len(act), int)
        for i, R in enumerate(atoms):
            for s in imgs:
                d = np.linalg.norm(x[act] - (R + s), axis=1)
                u = d < best
                best[u], bi[u] = d[u], i
        done = best < stop
        lab[act[done]] = bi[done]
        act = act[~done]
    return lab


@pytest.mark.parametrize("skew", [False, True])
def test_bader_matches_the_analytic_zero_flux_partition(skew):
    """Three cusped 'atoms' of unequal size in a periodic (skewed) cell: each atom's grid population
    from volumetric.bader against the populations of the same grid points assigned by integrating the
    analytic density's gradient to a nucleus. Agreement to 0.5 % of each population."""
    L, N = 10.0, 36
    cell = np.eye(3) * L if not skew else np.array([[L, 0, 0], [3.5, 9.0, 0], [2.0, 2.5, 8.5]])
    atoms = np.array([[0.30, 0.5, 0.5], [0.65, 0.52, 0.47], [0.5, 0.15, 0.8]]) @ cell
    terms = [[(10.0, 3.0), (1.0, 1.2)], [(4.0, 2.0), (0.6, 1.0)], [(6.0, 2.5), (0.8, 1.1)]]
    rho_grad, imgs = _promolecule(cell, atoms, terms, (True, True, True))
    fr = np.stack(np.meshgrid(*[np.arange(N) / N] * 3, indexing="ij"), -1).reshape(-1, 3)
    pts = fr @ cell
    rho, _ = rho_grad(pts)
    vox = cell / N
    g = V.Grid(rho.reshape(N, N, N), vox, np.zeros(3), [True] * 3, abs(np.linalg.det(vox)), cell, [6, 4, 5],
               [6.0, 4.0, 5.0], atoms, "density", Path("synthetic"))
    r = V.bader(g, valence=[0.0, 0.0, 0.0])
    lab = _exact_labels(pts, rho_grad, atoms, imgs)
    assert (lab >= 0).all()
    exact = np.bincount(lab, weights=rho, minlength=3) * g.dv
    pops = np.asarray(r["populations"]) + np.asarray(r["core_corrections"])   # the grid's own sums
    np.testing.assert_allclose(pops, exact, rtol=5e-3)
    assert r["maxima"] == 3 and r["nonnuclear_maxima"] == 0


def test_bader_of_two_equal_atoms_splits_them_equally():
    L, N = 8.0, 32
    cell = np.eye(3) * L
    atoms = np.array([[0.3, 0.5, 0.5], [0.7, 0.5, 0.5]]) @ cell
    rho_grad, _ = _promolecule(cell, atoms, [[(5.0, 2.0), (0.5, 0.9)]] * 2, (True, True, True))
    pts = np.stack(np.meshgrid(*[np.arange(N) / N] * 3, indexing="ij"), -1).reshape(-1, 3) @ cell
    rho, _ = rho_grad(pts)
    vox = cell / N
    g = V.Grid(rho.reshape(N, N, N), vox, np.zeros(3), [True] * 3, abs(np.linalg.det(vox)), cell, [1, 1],
               [1.0, 1.0], atoms, "density", Path("synthetic"))
    # on the grid alone the mirror plane's points are split half and half: exactly equal
    r = V.bader(g, valence=[0.0, 0.0], passes=0)
    assert r["tie_points"] > 0
    assert r["populations"][0] == pytest.approx(r["populations"][1], rel=1e-12)
    # the near-grid refinement follows rounded trajectories: equal to its accuracy
    r = V.bader(g, valence=[0.0, 0.0])
    assert r["populations"][0] == pytest.approx(r["populations"][1], rel=5e-3)


def test_core_corrections_close_the_grid_electron_count():
    """The grid sum of a real ECH3 cube misses or over-counts part of each nuclear cusp; the Slater
    core model accounts for it: EC (46 e) and the FSI/graphene slab (754 e) to 0.03 %."""
    for name, ne in (("EC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized"
                      "_charge+potential_DENS.CUBE", 46),
                     ("4LG_FSI_TopMiddle_2x2_ABAB_charge+potential_DENS.CUBE", 754)):
        g = V.load(_corpus(name))
        n_grid = float(g.data.sum()) * g.dv
        corrected = n_grid - float(V.core_corrections(g).sum())
        assert abs(n_grid - ne) > 0.09                      # the raw sum is off
        assert corrected == pytest.approx(ne, rel=3e-4)


def test_bader_charges_of_a_real_molecule_are_symmetric_and_sensible():
    """LiFSI on CRYSTAL's 0.23 bohr grid: Li about +0.92; the two S and the two F of FSI agree (they
    did not before the cusp correction: S 3.07/3.60, F -0.29/-0.63); the charges sum to zero."""
    out = _corpus("LiFSI_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized"
                  "_charge+potential.out")
    vals, rep = derived(out, ["bader_charges"])
    assert rep["bader_charges"]["status"] == "stored", rep
    atoms = vals["bader_charges"]["atoms"]
    q = {}
    for a in atoms:
        q.setdefault(a["element"], []).append(a["charge"])
    assert 0.88 < q["Li"][0] < 0.96
    assert abs(q["S"][0] - q["S"][1]) < 0.15 and 3.3 < min(q["S"]) and max(q["S"]) < 3.7
    assert abs(q["F"][0] - q["F"][1]) < 0.03 and -0.75 < min(q["F"]) and max(q["F"]) < -0.55
    assert abs(sum(a["charge"] for a in atoms)) < 0.05
    assert vals["bader_charge_max"] == pytest.approx(max(max(v) for v in q.values()))
    assert vals["bader_nonnuclear_maxima"] == 0


# ------------------------------------------------------------------ dipoles
@pytest.mark.parametrize("stem,sp", [
    ("EC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized",
     "EC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized"),
    ("LiFSI_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized",
     "LiFSI_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized"),
])
def test_molecule_cell_dipole_matches_the_dipole_crystal_prints(stem, sp):
    """A molecule's cell dipole is CRYSTAL's DIPOLE MOMENT of the parent SCF (SP/ beside ECH3POT3/), as
    printed; the density cube's own dipole agrees with it on these fine grids: within 2 % in magnitude
    and 0.15 D per component (EC 5.29 vs 5.34 D, LiFSI 7.01 vs 6.95 D)."""
    out = _corpus(stem + "_charge+potential.out")
    ref = printed_dipole(TEST_DATA / "SP" / (sp + ".out"))
    vals, rep = derived(out, ["cell_dipole"])
    assert rep["cell_dipole"]["status"] == "stored", rep
    assert vals["cell_dipole"]["source"] == sp + ".out"
    mag = math.sqrt(sum(v * v for v in ref.values()))
    assert vals["cell_dipole_debye"] == pytest.approx(mag, rel=1e-9)
    for ax, v in ref.items():
        assert vals[f"cell_dipole_{ax}_debye"] == pytest.approx(v, rel=1e-9)
    from mace.properties.defs import charge_potential as CP
    g = V.load(out.with_name(stem + "_charge+potential_DENS.CUBE"))
    text = out.read_text(errors="replace")
    d = V.dipole(g, CP.valence(text, g), CP.electrons(text))
    assert d["magnitude_debye"] == pytest.approx(mag, rel=0.02)
    for k, ax in enumerate("xyz"):
        assert d["vector_debye"][k] == pytest.approx(ref[ax], abs=0.15)


_LIDEC = "1LiFSI-1DEC-{}_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized"


@pytest.mark.parametrize("conf,printed,spacing", [("conf1", 7.3832, 0.312), ("conf2", 8.4192, 0.478)])
def test_coarse_cube_dipole_is_never_stored(conf, printed, spacing, tmp_path):
    """On a coarse grid the cube dipole is wrong (LiFSI-1DEC conf1 4.97 D, conf2 4.13 D with the nuclear
    cusps 9.7 e off, for CRYSTAL's 7.38 and 8.42 D): with the parent SCF beside the run its printed
    dipole is stored; without it the property is skipped, with the grid's numbers."""
    stem = _LIDEC.format(conf)
    out = _corpus(stem + "_charge+potential.out")
    vals, rep = derived(out, ["cell_dipole"])
    assert rep["cell_dipole"]["status"] == "stored", rep
    assert vals["cell_dipole"]["source"] == stem + ".out"
    assert vals["cell_dipole_debye"] == pytest.approx(printed, abs=1e-4)
    # the run alone, its parent SCF not beside it
    for suffix in ("_charge+potential.out", "_charge+potential_DENS.CUBE"):
        shutil.copy(ECH / (stem + suffix), tmp_path / (stem + suffix))
    vals, rep = derived(tmp_path / (stem + "_charge+potential.out"), ["cell_dipole"])
    assert rep["cell_dipole"]["status"] == "skipped" and "cell_dipole_debye" not in vals
    assert "too coarse" in rep["cell_dipole"]["reason"] and f"spacing {spacing:.3f} bohr" in rep["cell_dipole"]["reason"]


def test_fine_cube_dipole_without_the_parent(tmp_path):
    """EC's 0.157 bohr cube (nuclear-cusp correction -0.097 e) without its parent SCF: the cube's dipole is
    stored and says it is the cube's (5.29 D; CRYSTAL prints 5.34 D)."""
    stem = "EC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized"
    for suffix in ("_charge+potential.out", "_charge+potential_DENS.CUBE"):
        shutil.copy(_corpus(stem + suffix), tmp_path / (stem + suffix))
    vals, rep = derived(tmp_path / (stem + "_charge+potential.out"), ["cell_dipole"])
    assert rep["cell_dipole"]["status"] == "stored", rep
    assert vals["cell_dipole"]["source"].endswith("_DENS.CUBE") and vals["cell_dipole"]["spacing_bohr"] < 0.16
    assert vals["cell_dipole_debye"] == pytest.approx(5.3403, rel=0.02)


@pytest.mark.parametrize("name,opt,tol", [("4LG_FSI_TopMiddleBottom_2x2_ABAB", "4LG_FSI_TopMiddleBottom_2x2_ABAB_opt", 0.01),
                                          ("4LG_FSI_TopMiddle_2x2_ABAB", "4LG_FSI_TopMiddle_2x2_ABAB_opt", 0.05)])
def test_slab_dipole_and_work_functions(name, opt, tol):
    """A slab's dipole from its vacuum step (p = -A dU / 4 pi) against CRYSTAL's printed dipole of the
    OPT the properties ran from (0.1504 vs 0.1501 D; -1.026 vs -0.995 D); the work functions are the
    vacuum levels minus the .out's Fermi energy and differ by the step."""
    out = _corpus(name + "_charge+potential.out")
    ref = printed_dipole(TEST_DATA / "OPT" / (opt + ".out"))["z"]
    vals, rep = derived(out, ["cell_dipole", "potential_profile"])
    assert rep["cell_dipole"]["status"] == "stored" and rep["potential_profile"]["status"] == "stored", rep
    assert vals["cell_dipole_z_debye"] == pytest.approx(ref, rel=tol)
    assert vals["cell_dipole_correction_ev"] == pytest.approx(vals["potential_vacuum_step_ev"])
    assert vals["work_function_top_ev"] - vals["work_function_bottom_ev"] == pytest.approx(
        vals["potential_vacuum_step_ev"])
    ef = vals["potential_planar_average"]["fermi_ev"]
    assert vals["work_function_top_ev"] == pytest.approx(vals["vacuum_level_top_ev"] - ef)
    assert ef == pytest.approx(-0.174 * 27.211386245988 if "Bottom" in name else -0.166 * 27.211386245988)
    curve = vals["potential_planar_average"]
    assert curve["axis"] == "c" and len(curve["z_A"]) == len(curve["planar_ev"]) == len(curve["macroscopic_ev"])
    assert 2.5 < curve["window_A"] < 3.8                   # the layers' spacing (graphene and FSI)
    assert all(curve["vacuum"][s]["converged"] for s in ("top", "bottom"))


def test_graphene_work_function_is_minus_the_fermi_energy(suite):
    """A symmetric monolayer: both vacuum levels are 0 (CRYSTAL's potential vanishes far from a neutral
    slab), so Φ = -E_F = 0.150 Ha = 4.0817 eV, the step and the dipole vanish."""
    out = suite / "3-slab-2d" / "graphene" / "CHARGE+POTENTIAL" / "graphene_charge+potential.out"
    vals, rep = derived(out, ["potential_profile", "cell_dipole", "bader_charges"])
    assert vals["work_function_top_ev"] == pytest.approx(0.150 * 27.211386245988, abs=2e-3)
    assert vals["work_function_bottom_ev"] == pytest.approx(vals["work_function_top_ev"], abs=1e-3)
    assert abs(vals["potential_vacuum_step_ev"]) < 1e-3 and abs(vals["cell_dipole_z_debye"]) < 1e-3
    # the suite's cubes keep every 2nd point: no grid sums on them
    assert rep["bader_charges"]["status"] == "skipped" and "thinned" in rep["bader_charges"]["reason"]


def test_bulk_profile_and_skips(suite):
    out = suite / "2-dense-3d" / "diamond" / "CHARGE+POTENTIAL" / "diamond_charge+potential.out"
    vals, rep = derived(out, ["potential_profile", "cell_dipole"])
    curve = vals["potential_planar_average"]
    assert curve["periodic"] is True and "vacuum" not in curve and "work_function_ev" not in vals
    assert rep["cell_dipole"]["status"] == "skipped"
    mol = suite / "5-molecule-0d" / "EC" / "CHARGE+POTENTIAL" / "EC_charge+potential.out"
    vals, rep = derived(mol, ["potential_profile", "cell_dipole", "bader_charges"])
    assert rep["potential_profile"]["status"] == "skipped"
    # the suite's coarse 0.31 bohr grid gives 5.90 D: CRYSTAL's 5.34 D of the SP run beside it is stored
    assert vals["cell_dipole_debye"] == pytest.approx(5.3403, abs=1e-4)
    assert vals["cell_dipole"]["source"] == "EC_sp.out"
    assert abs(sum(a["charge"] for a in vals["bader_charges"]["atoms"])) < 0.05


def test_macroscopic_average_of_a_periodic_profile_is_flat():
    """The running mean over one period of a periodic profile is its mean everywhere."""
    z = np.arange(120) * 0.05
    prof = np.sin(2 * np.pi * z / 1.5) + 0.3
    m = V.macroscopic_average(z, prof, 1.5, periodic=True)
    np.testing.assert_allclose(m, 0.3, atol=1e-9)
    assert V.layer_period([0.0, 0.1, 3.35, 3.4, 6.7, 6.75]) == pytest.approx(3.35, abs=0.06)
    assert V.layer_period([1.0, 1.2]) is None


def test_extraction_through_the_cli_stores_the_work_function(suite, tmp_path):
    """`mace analyze --extract-properties` (subprocess) stores the slab's work function and profile."""
    run = tmp_path / "g"
    shutil.copytree(suite / "3-slab-2d" / "graphene" / "CHARGE+POTENTIAL", run)
    env = {k: v for k, v in __import__("os").environ.items() if k not in ("PYTHONPATH", "MACE_HOME")}
    p = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "analyze", "--extract-properties",
                        str(run), "--db-path", str(tmp_path / "m.db")], capture_output=True, text=True, env=env,
                       timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    conn = sqlite3.connect(tmp_path / "m.db")
    rows = dict(conn.execute("SELECT property_name, property_value FROM properties"))
    assert rows["work_function_ev"] == pytest.approx(4.0817, abs=2e-3)
    blobs = {r[0] for r in conn.execute("SELECT property_name FROM property_blobs")}
    assert "potential_planar_average" in blobs
