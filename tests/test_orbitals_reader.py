"""mace.orbitals against real CRYSTAL23 ORBITALS output (tests/data/orbitals, MSU HPCC 2026-10-16).

Reference grids (tests/data/orbitals/ref) are CRYSTAL's own PBAN/ECH3 cubes,
subsampled to keep the fixtures small.
"""
from pathlib import Path

import numpy as np
import pytest

from mace.orbitals import api
from mace.orbitals.cube import cube_meta, read_cube
from mace.orbitals.dataset import OrbitalRun, OrbitalsError, default_isovalue
from mace.orbitals.molden import MoldenError, read_molden
from mace.orbitals.outfile import parse_out, split_name

DATA = Path(__file__).parent / "data" / "orbitals"


def _corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


# ---------------------------------------------------------------- parsing
def test_molecule_parse_h2o():
    md = read_molden(DATA / "h2o" / "h2o_c.molden")
    assert md.dimensionality == 0
    assert md.symbols == ["O", "H", "H"]
    assert md.spherical_d and md.spherical_f
    assert md.coef.shape == (md.nbf, len(md.energies))
    assert md.occupations[:5].tolist() == [2.0] * 5 and md.occupations[5] == 0.0


def test_beta_coefficients_are_numbered_after_alpha():
    md = read_molden(DATA / "o2" / "o2_c.molden")
    spins = set(md.spins)
    assert spins == {"ALPHA", "BETA"}
    beta = [j for j, s in enumerate(md.spins) if s == "BETA"]
    # beta numbering nbf+1..2nbf lands on rows 0..nbf-1, not dropped
    assert np.abs(md.coef[:, beta]).max() > 0.5
    run = OrbitalRun.from_path(DATA / "o2" / "o2_c.molden")
    kd = run.load(1)
    assert kd.status == "ok" and kd.check["BETA"] < 1e-5 and kd.check["ALPHA"] < 1e-5
    # triplet: alpha has two more occupied orbitals than beta
    assert kd.homo("ALPHA") - kd.homo("BETA") == 2


def test_periodic_cellaxes_and_mixed_atoms():
    md = read_molden(DATA / "slab" / "slab_f_00003_K200_real.molden")
    assert md.periodic_dims == [0, 1]           # [CellAxes] zero row along z
    assert md.atoms_mode == "fractional"
    pos = md.positions_bohr()
    # z is Cartesian Angstrom in the file (-2.105 A), not a fraction
    assert pos[2, 2] == pytest.approx(-2.105 / 0.529177210903)


def test_not_molden_is_refused(tmp_path):
    p = tmp_path / "x.molden"
    p.write_text("hello\n")
    with pytest.raises(MoldenError):
        read_molden(p)


def test_k_from_out_not_label():
    out = parse_out(DATA / "si_f" / "si_f.out")
    assert [k.index for k in out.kpoints] == list(range(1, 9))
    k3 = out.kpoint(3)
    assert k3.real and np.allclose(k3.frac, [0.5, 0, 0])
    assert not out.kpoint(4).real
    assert out.has_orbitals and out.electrons == 28
    assert split_name("si_f_00004_K110_complex.molden") == {"root": "si_f", "n": 4, "part": "complex", "label": "110"}
    assert split_name("h2o_c.molden")["n"] is None


# ---------------------------------------------------------------- self-check
@pytest.mark.parametrize("folder,name,k,tol", [
    ("h2o", "h2o_c.molden", 1, 1e-5),
    ("o2", "o2_c.molden", 1, 1e-5),
    ("si_g", "si_g_00001_K000_real.molden", 1, 1e-3),     # incl. 2 virtuals (TOLINTEG-limited)
    ("si_f", "si_f_00003_K200_real.molden", 3, 1e-3),
    ("polyyne", "py_f_00001_K000_real.molden", 1, 1e-4),
    ("polyyne", "py_f_00003_K200_real.molden", 3, 1e-4),
    ("polyyne", "py_f_00002_K100_real.molden", 2, 1e-5),   # complex k in 1D validates
    ("slab", "slab_f_00003_K200_real.molden", 3, 1e-4),     # needs CRYSTAL's atom images
    ("fe", "fe_f_00001_K000_real.molden", 1, 1e-3),         # spin + f shell, metal (E_F decides occupied)
])
def test_orthonormal(folder, name, k, tol):
    run = OrbitalRun.from_path(DATA / folder / name)
    kd = run.load(k)
    assert kd.status == "ok"
    assert max(kd.check.values()) < tol


def test_slab_zone_boundary_needs_internal_atom_images():
    run = OrbitalRun.from_path(DATA / "slab" / "slab_f_00003_K200_real.molden")
    kd = run.load(3)
    assert max(kd.check.values()) < 1e-4
    # the file's own wrapping (O and Mg at -0.5) breaks the zone-boundary phases
    from mace.orbitals.basis import Basis, bloch_overlap, orthonormality_error
    b = Basis(kd.md)                       # positions as written
    ns, SR = b.overlap_cells()
    S = bloch_overlap(ns, SR, kd.entry.frac)
    occ = [j for j in range(kd.C.shape[1]) if kd.md.occupations[j] > 0.5]
    assert orthonormality_error(kd.C[:, occ], S) > 0.1


def test_complex_k_3d_is_refused_and_fails_the_check():
    run = OrbitalRun.from_path(DATA / "si_f" / "si_f_00003_K200_real.molden")
    e = run.entry(4)
    assert not e.real and not e.usable and "complex" in e.reason
    with pytest.raises(OrbitalsError):
        run.load(4)
    kd = run.load(4, allow_unusable=True)
    assert kd.status == "fail"
    with pytest.raises(OrbitalsError):
        run.grid(4, "ALPHA+BETA", ["HOMO"], allow_failed=False)


def test_complex_k_1d_is_offered():
    run = OrbitalRun.from_path(DATA / "polyyne" / "py_f_00002_K100_real.molden")
    e = run.entry(2)
    assert not e.real and e.usable


# ---------------------------------------------------------------- grids vs CRYSTAL
def test_h2o_homo_matches_pban():
    run = OrbitalRun.from_path(DATA / "h2o" / "h2o_c.molden")
    g, ref, *_ = read_cube(DATA / "ref" / "h2o_pban5.cube")
    _, d, meta = run.grid(1, "ALPHA+BETA", ["HOMO"], part="abs2", grid=g)
    assert meta["mos"] == [5]
    assert _corr(2 * d, ref) > 0.9999
    assert (2 * d).sum() / ref.sum() == pytest.approx(1.0, abs=1e-3)


def test_h2o_occupied_sum_matches_ech3():
    run = OrbitalRun.from_path(DATA / "h2o" / "h2o_c.molden")
    g, ref, *_ = read_cube(DATA / "ref" / "h2o_ech3.cube")
    _, d, _ = run.grid(1, "ALPHA+BETA", [1, 2, 3, 4, 5], part="abs2", grid=g)
    assert _corr(2 * d, ref) > 0.9999
    assert (2 * d).sum() / ref.sum() == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("bands,ref", [([11], "si_pban11.cube"), ([12, 13, 14], "si_pban1214.cube")])
def test_si_gamma_band_sum_matches_pban(bands, ref):
    run = OrbitalRun.from_path(DATA / "si_g" / "si_g_00001_K000_real.molden")
    g, r, *_ = read_cube(DATA / "ref" / ref)
    _, d, _ = run.grid(1, "ALPHA+BETA", bands, part="abs2", grid=g)
    assert _corr(2 * d, r) > 0.9999
    assert (2 * d).sum() / r.sum() == pytest.approx(1.0, abs=2e-3)


def test_si_triplet_degenerate_set_and_summed_view():
    run = OrbitalRun.from_path(DATA / "si_g" / "si_g_00001_K000_real.molden")
    kd = run.load(1)
    assert kd.homo() == 14
    assert kd.degenerate_set("ALPHA+BETA", 14) == [12, 13, 14]
    _, d, meta = run.grid(1, "ALPHA+BETA", ["HOMO"], degenerate_sum=True, part="re", quality="preview")
    # a degenerate set is only ever summed as |psi|^2
    assert meta["mos"] == [12, 13, 14] and meta["part"] == "abs2" and not meta["signed"]
    assert d.min() >= 0


def test_bloch_norm_and_zone_boundary_sign_alternation():
    run = OrbitalRun.from_path(DATA / "polyyne" / "py_f_00003_K200_real.molden")
    g, d, meta = run.grid(3, "ALPHA+BETA", ["HOMO"], part="re", cells=(2, 1, 1), quality="preview")
    n = g.shape[0] // 2
    # k = 1/2: psi(r + a) = -psi(r)
    assert np.allclose(d[n:], -d[:n], atol=1e-6 * np.abs(d).max() + 1e-9)
    g, d2, _ = run.grid(3, "ALPHA+BETA", ["HOMO"], part="abs2", cells=(1, 1, 1))
    assert d2.sum() * g.voxel_volume == pytest.approx(1.0, abs=5e-3)


def test_complex_1d_re_im_quarter_period_and_uniform_density():
    run = OrbitalRun.from_path(DATA / "polyyne" / "py_f_00002_K100_real.molden")
    g, re_, _ = run.grid(2, "ALPHA+BETA", ["HOMO"], part="re", cells=(4, 1, 1), quality="preview")
    _, a2, _ = run.grid(2, "ALPHA+BETA", ["HOMO"], part="abs2", cells=(4, 1, 1), quality="preview")
    _, im, _ = run.grid(2, "ALPHA+BETA", ["HOMO"], part="im", cells=(4, 1, 1), quality="preview")
    assert np.abs(im).max() > 0.1 * np.abs(re_).max()
    n = g.shape[0] // 4
    # |psi|^2 has the lattice period
    assert np.allclose(a2[:n], a2[n:2 * n], atol=1e-6 * a2.max())
    with pytest.raises(OrbitalsError):
        run.grid(1, "ALPHA+BETA", ["HOMO"], part="im")


def test_default_isovalue_encloses_ninety_percent():
    run = OrbitalRun.from_path(DATA / "h2o" / "h2o_c.molden")
    _, d, _ = run.grid(1, "ALPHA+BETA", ["HOMO"], part="re", quality="normal")
    iso = default_isovalue(d, signed=True)
    w = d ** 2
    assert w[np.abs(d) >= iso].sum() / w.sum() == pytest.approx(0.9, abs=0.02)


# ---------------------------------------------------------------- cubes / api
def test_make_cube_caches_next_to_the_job(tmp_path):
    import shutil
    for f in (DATA / "h2o").iterdir():
        shutil.copy(f, tmp_path / f.name)
    run = api.open_run(tmp_path / "h2o_c.molden")
    path, meta = api.make_cube(run, 1, "ab", ["HOMO"], part="re", quality="preview")
    assert path.parent == tmp_path / "h2o_c_orbitals"
    assert path.name == "h2o_c_ab_MO5_re_preview.cube" and not meta["cached"]
    g, d, Z, pos, c1, c2 = read_cube(path)
    m = cube_meta(c2)
    assert m["mo"] == "5" and m["part"] == "re" and m["spin"] == "ALPHA+BETA"
    assert list(Z) == [8, 1, 1]
    again, meta2 = api.make_cube(run, 1, "ab", ["HOMO"], part="re", quality="preview")
    assert again == path and meta2["cached"]
    # the Molden file is untouched
    assert (tmp_path / "h2o_c.molden").read_bytes() == (DATA / "h2o" / "h2o_c.molden").read_bytes()


def test_describe_lists_k_points_and_levels():
    run = api.open_run(DATA / "polyyne" / "py_f_00001_K000_real.molden")
    info = api.describe(run)
    assert info["dimensionality"] == 1
    assert [k["index"] for k in info["kpoints"]] == [1, 2, 3]
    sp = info["k"]["spins"][0]
    assert sp["homo"] == 6 and sp["lumo"] == 7
    assert info["k"]["status"] == "ok"


def test_quota_warning():
    assert api.quota_warning(3, 10_000_000) is None
    assert "1M-file" in api.quota_warning(500, 10)


def test_cube_folder_names():
    from mace.orbitals.cube import cache_dir_for
    assert cache_dir_for("/w", "h2o_c").name == "h2o_c_orbitals"
    assert cache_dir_for("/w", "si_orb").name == "si_orbitals"
    # MACE's own step name: <material>_orbitals.d3 -> root <material>_orbitals_orb
    assert cache_dir_for("/w", "mgo_orbitals_orb").name == "mgo_orbitals"
