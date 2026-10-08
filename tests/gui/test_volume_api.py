"""The Volume document's server side (plan E5): real corpus cubes copied
under neutral names (skipped without the gitignored corpus) and small
synthetic cubes written here.

Statistics come from the full grid (numpy on CubeFile.data); the decimated
grid fits the quality budget; a slice is CubeFile.data's plane exactly; the
MEP range is the CLI's rule on the same surface; a difference equals
perform_cube_arithmetic(..., align_grids=True); every route keeps the
server's path and token rules; Save Result As never writes over a file."""
import gzip
import os
import shutil
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

np = pytest.importorskip("numpy")
pytest.importorskip("plotly", reason="the cube engine imports plotly")
CORPUS = H.REPO_ROOT / "test" / "ECH3POT3"
MOL = "DEC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized_charge+potential"
SLAB = "4LG_FSI_TopMiddle_2x2_ABAB_charge+potential"
# the slab triple on different grids (792 and 604 points along c, origins 15 Bohr apart): complex, substrate, fragment
TRIPLE = ("4LG_FSI_TopMiddleBottom_2x2_ABAB_opt_charge+potential_DENS.CUBE",
          "4LG_FSI_TopMiddleBottom_2x2_ABAB_graphene_opt_charge+potential_DENS.CUBE",
          "4LG_FSI_TopMiddleBottom_2x2_ABAB_FSI_opt_charge+potential_DENS.CUBE")
needs_corpus = pytest.mark.skipif(not (CORPUS / f"{MOL}_DENS.CUBE").is_file(), reason="test/ corpus not present (gitignored)")


def write_cube(path: Path, data, voxel=0.5, origin=(0.0, 0.0, 0.0), comment2="0 0 0 0 0 0", atoms=((8, (1.0, 1.0, 1.0)),)):
    nx, ny, nz = data.shape
    with open(path, "w") as fh:
        fh.write("synthetic cube\n" + comment2 + "\n")
        fh.write(f"{len(atoms):5d} {origin[0]:12.6f} {origin[1]:12.6f} {origin[2]:12.6f}\n")
        for n, v in zip((nx, ny, nz), np.eye(3) * voxel):
            fh.write(f"{n:5d} {v[0]:12.6f} {v[1]:12.6f} {v[2]:12.6f}\n")
        for z, pos in atoms:
            fh.write(f"{z:5d} 0.000000 {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
        flat = data.ravel()
        for i in range(0, len(flat), 6):
            fh.write(" ".join(f"{x:13.5E}" for x in flat[i:i + 6]) + "\n")


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    if not (CORPUS / f"{MOL}_DENS.CUBE").is_file():
        pytest.skip("test/ corpus not present (gitignored)")
    tmp = tmp_path_factory.mktemp("volume_api")
    root = H.examples_copy(tmp)
    v = root / "vol"
    v.mkdir()
    for t in ("DENS", "POT", "SPIN"):
        shutil.copyfile(CORPUS / f"{MOL}_{t}.CUBE", v / f"mol_{t}.CUBE")
    shutil.copyfile(CORPUS / f"{MOL}.out", v / "mol.out")
    shutil.copyfile(CORPUS / f"{MOL}_DENS.CUBE", v / "DENS_CUBE.DAT")
    for name, new in zip(TRIPLE, ("complex_DENS.CUBE", "graphene_DENS.CUBE", "fragment_DENS.CUBE")):
        shutil.copyfile(CORPUS / name, v / new)
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1", MACE_GUI_MAX_WORKERS="2")
    yield root, base, tmp
    proc.terminate()
    proc.wait(timeout=15)
    shutil.rmtree(v, ignore_errors=True)


def result(base, path, data=None, timeout=180):
    st, _, r = U.post(base, path, data) if data is not None else U.req(base, path)
    if st == 202:
        tid = r["task"]["id"]
        end = time.time() + timeout
        while time.time() < end:
            st, _, r = U.req(base, f"api/tasks/{tid}/result")
            if st == 200:
                return r
            time.sleep(0.1)
        raise AssertionError("the task never finished")
    assert st == 200, (st, r)
    return r


def _cube(path):
    from mace.gui import volume
    return volume.read(str(path), skip_vacuum=True)


@needs_corpus
def test_load_stats_on_the_full_grid_and_the_budget(served):
    root, base, _ = served
    for name, typ in (("vol/mol_DENS.CUBE", "density"), ("vol/DENS_CUBE.DAT", "density"), ("vol/mol_SPIN.CUBE", "spin"),
                      ("vol/complex_DENS.CUBE", "density")):
        m = result(base, f"api/volume?path={name}&quality=low&wait=60")
        assert m["type"] == typ, (name, m["type"])
        assert np.prod(m["shape"]) <= 250_000 * 1.05, m["shape"]
        c = _cube(root / name)
        d = np.asarray(c.data, float)
        assert m["stats"]["min"] == float(d.min()) and m["stats"]["max"] == float(d.max())
        st, h, raw = U.req(base, f"api/volume/grid?key={m['grid_key']}")
        assert st == 200
    # the integral: the full grid's sum x voxel volume, with its label rules
    m = result(base, "api/volume?path=vol/mol_DENS.CUBE&quality=low&wait=60")
    c = _cube(root / "vol/mol_DENS.CUBE")
    want = float(np.asarray(c.data, float).sum() * abs(np.linalg.det(c.voxel_vectors)))
    assert abs(m["stats"]["integral"] - want) / want < 1e-9
    assert m["stats"]["ecp"] is False and "valence count" in m["stats"]["integral_note"]
    # the slab: the periodic a and b repeat their first plane, counted once
    m = result(base, "api/volume?path=vol/complex_DENS.CUBE&quality=low&wait=60")
    assert m["stats"]["duplicate_planes"] == ["a", "b"] and m["stats"]["integral"] > 0


def test_integral_label_cases(tmp_path):
    """Cut off at a face: no number; a periodic grid with its endpoint repeated: counted once."""
    from mace.gui import volume
    g = np.zeros((10, 10, 10))
    g[3:7, 3:7, 3:7] = 1.0
    write_cube(tmp_path / "ok_DENS.CUBE", g)
    s = volume.stats(volume.read(str(tmp_path / "ok_DENS.CUBE"), skip_vacuum=True), tmp_path / "ok_DENS.CUBE")
    assert abs(s["integral"] - 64 * 0.125) < 1e-9 and s["ecp"] is None
    g2 = g.copy()
    g2[0, 5, 5] = 0.9
    write_cube(tmp_path / "cut_DENS.CUBE", g2)
    s = volume.stats(volume.read(str(tmp_path / "cut_DENS.CUBE"), skip_vacuum=True))
    assert s["integral"] is None and "cut off" in s["integral_note"]
    per = np.random.default_rng(1).random((9, 9, 9))
    per[-1, :, :] = per[0, :, :]
    per[:, -1, :] = per[:, 0, :]
    per[:, :, -1] = per[:, :, 0]
    write_cube(tmp_path / "per_DENS.CUBE", per, comment2=f"{8 * 0.5} {8 * 0.5} {8 * 0.5} 90 90 90")
    c = volume.read(str(tmp_path / "per_DENS.CUBE"), skip_vacuum=True)
    s = volume.stats(c)
    assert s["duplicate_planes"] == ["a", "b", "c"]
    assert abs(s["integral"] - np.asarray(c.data)[:-1, :-1, :-1].sum() * 0.125) < 1e-9


def test_raw_crystal_names_and_signed_orbitals(tmp_path):
    from mace.gui import volume, filekind
    g = np.zeros((8, 8, 8))
    g[2, 2, 2], g[5, 5, 5] = 0.6, -0.6
    write_cube(tmp_path / "t_WAN_CUBE.DAT", g)
    assert filekind.kind_by_name("t_WAN_CUBE.DAT") == "cube" and filekind.cube_header_ok(tmp_path / "t_WAN_CUBE.DAT")
    c = volume.read(str(tmp_path / "t_WAN_CUBE.DAT"))
    assert c.data_type == "orbital"
    meta, grid = volume.payload(c, tmp_path / "t_WAN_CUBE.DAT")
    assert meta["signed"] and meta["level"] == 0.2 and meta["label"].startswith("Orbital")


@needs_corpus
def test_slice_is_the_grid_plane(served):
    root, base, _ = served
    c = _cube(root / "vol/complex_DENS.CUBE")
    k = 300
    r = result(base, f"api/volume/slice?path=vol/complex_DENS.CUBE&axis=c&index={k}")
    assert np.array_equal(np.asarray(r["values"]), np.round(c.data[:, :, k], 10))
    # the hexagonal plane resampled on Cartesian axes: a parallelogram inside its bounding box (NaN
    # outside: two thirds of the box are inside for a 120-degree cell), its values those of the plane
    cart = np.asarray([[np.nan if x is None else x for x in row] for row in r["cart"]], float)
    inside = np.isfinite(cart).mean()
    assert 0.6 < inside < 0.7, inside
    plane = c.data[:, :, k]
    assert np.nanmin(cart) >= plane.min() - 1e-12 and np.nanmax(cart) <= plane.max() + 1e-12


@needs_corpus
def test_mep_range_equals_the_cli_rule(served):
    """The GUI's auto range (vertices -> interpolate_cube_at_vertices -> the CLI's percentiles)
    equals plot_dual_cube_isosurface's on the same files and level within 2 %."""
    pytest.importorskip("skimage", reason="the CLI side needs scikit-image")
    root, base, _ = served
    from mace.gui import volume
    from mace.plotting.engines import crystal_cubeviz_plotly as E
    import contextlib
    import io
    dens, pot = volume.read(str(root / "vol/mol_DENS.CUBE")), volume.read(str(root / "vol/mol_POT.CUBE"))
    level = 0.02
    with contextlib.redirect_stdout(io.StringIO()):
        fig = E.plot_dual_cube_isosurface(dens, pot, level)
    mesh = next(t for t in fig.data if t.type == "mesh3d")
    cli = (mesh.cmin, mesh.cmax)
    # the route as the page calls it: the density's file and level
    r = result(base, "api/volume/mep_range", {"path": "vol/mol_POT.CUBE", "shape": "vol/mol_DENS.CUBE", "level": level})
    assert r["method"] == "cli-mesh" and abs(r["vmin"] - cli[0]) <= 0.02 * abs(cli[0]) and abs(r["vmax"] - cli[1]) <= 0.02 * abs(cli[1]), (r, cli)
    assert r["signed"] and r["type"] == "potential" and abs(r["vmin"] + r["vmax"]) < 1e-12
    # the vertex route (a difference made in the GUI has no file): the CLI's mesh gives the CLI's range
    verts = np.column_stack([mesh.x, mesh.y, mesh.z]).ravel().tolist()
    r = result(base, "api/volume/mep_range", {"path": "vol/mol_POT.CUBE", "vertices": verts})
    assert r["method"] == "vertices" and abs(r["vmax"] - cli[1]) <= 0.02 * abs(cli[1]), (r, cli)
    # without scikit-image: the potential at the density's near-level grid points, same sign rule
    old = E.HAS_SKIMAGE
    E.HAS_SKIMAGE = False
    try:
        f = volume.mep_range(str(root / "vol/mol_POT.CUBE"), shape_path=str(root / "vol/mol_DENS.CUBE"), level=level)
    finally:
        E.HAS_SKIMAGE = old
    assert f["method"] == "near-iso" and f["vmin"] == -f["vmax"] and abs(f["vmax"] - cli[1]) <= 0.25 * cli[1], (f, cli)
    chk = result(base, "api/volume/check", {"a": "vol/mol_DENS.CUBE", "b": "vol/complex_DENS.CUBE"})
    assert chk["problem"] and "different" in chk["problem"]


def test_same_grid_difference_is_the_cli_arithmetic(tmp_path):
    """Grids of one shape: perform_cube_arithmetic itself, bit for bit."""
    from mace.gui import volume
    from mace.plotting.engines import crystal_cubeviz_plotly as E
    import contextlib
    import io
    rng = np.random.default_rng(3)
    a, b = rng.random((6, 7, 8)), rng.random((6, 7, 8))
    write_cube(tmp_path / "a_DENS.CUBE", a)
    write_cube(tmp_path / "b_DENS.CUBE", b)
    paths = [str(tmp_path / "a_DENS.CUBE"), str(tmp_path / "b_DENS.CUBE")]
    got, _ = volume.combination(paths, [1, -1])
    cubes = [volume.read(p, skip_vacuum=True) for p in paths]
    with contextlib.redirect_stdout(io.StringIO()):
        want, _ = E.perform_cube_arithmetic(cubes, [1.0, -1.0], "linear")
    assert np.array_equal(got, want)


def test_alignment_is_exact_on_a_skewed_cell(tmp_path):
    """One physical density sampled on two grids of a triclinic cell (different sizes and a
    shifted origin): aligned, their difference is ~0 everywhere (the projection onto voxel
    directions mace plotting's align_cube_grids uses would shear it)."""
    from mace.gui import volume
    V = np.array([[0.30, 0.0, 0.0], [-0.12, 0.26, 0.0], [0.05, 0.04, 0.28]])
    center = np.array([4.0, 4.0, 4.0])

    def sample(origin, n):
        ii, jj, kk = np.meshgrid(*[np.arange(x) for x in n], indexing="ij")
        R = np.asarray(origin) + ii[..., None] * V[0] + jj[..., None] * V[1] + kk[..., None] * V[2]
        return np.exp(-((R - center) ** 2).sum(-1) / (2 * 1.2 ** 2))

    def write(path, origin, data):
        nx, ny, nz = data.shape
        with open(path, "w") as fh:
            fh.write("t\n0 0 0 0 0 0\n")
            fh.write(f"    1 {origin[0]:12.6f} {origin[1]:12.6f} {origin[2]:12.6f}\n")
            for n, v in zip(data.shape, V):
                fh.write(f"{n:5d} {v[0]:12.6f} {v[1]:12.6f} {v[2]:12.6f}\n")
            fh.write("    1 0.000000 4.000000 4.000000 4.000000\n")
            flat = data.ravel()
            for i in range(0, len(flat), 6):
                fh.write(" ".join(f"{x:13.5E}" for x in flat[i:i + 6]) + "\n")
    oa = np.array([0.0, 0.0, 0.0])
    ob = oa + 5 * V[0] + 3 * V[1] + 4 * V[2]           # whole voxels: copied, not interpolated
    write(tmp_path / "a_DENS.CUBE", oa, sample(oa, (40, 40, 40)))
    write(tmp_path / "b_DENS.CUBE", ob, sample(ob, (30, 32, 31)))
    d, t = volume.combination([str(tmp_path / "a_DENS.CUBE"), str(tmp_path / "b_DENS.CUBE")], [1, -1])
    # inside both grids the difference vanishes (to the file's 5 significant digits)
    sub = d[5:5 + 30, 3:3 + 32, 4:4 + 31]
    assert np.abs(sub).max() < 1e-4, np.abs(sub).max()


@needs_corpus
def test_difference_of_the_slab_triple_and_save_never_overwrites(served):
    """complex − substrate − fragment on the real triple (grids of 792 and 604 points along c):
    the GUI's aligned sum equals an independent per-point affine reference."""
    root, base, _ = served
    from mace.gui import volume
    from scipy.interpolate import RegularGridInterpolator
    paths = ["vol/complex_DENS.CUBE", "vol/graphene_DENS.CUBE", "vol/fragment_DENS.CUBE"]
    t0 = time.time()
    m = result(base, "api/volume/combine", {"paths": paths, "coefficients": [1, -1, -1], "quality": "low", "wait": 120})
    assert time.time() - t0 < 120
    assert m["name"] == "complex_DENS.CUBE − graphene_DENS.CUBE − fragment_DENS.CUBE" and m["type"] == "density_difference"
    got, t = volume.combination([str(root / p) for p in paths], [1, -1, -1])
    assert m["full_shape"] == list(got.shape)
    cubes = [volume.read(str(root / p), skip_vacuum=True) for p in paths]
    rng = np.random.default_rng(7)
    pts = np.column_stack([rng.integers(0, n, 400) for n in got.shape])
    for i, j, k in pts:
        r = t.origin + i * t.voxel_vectors[0] + j * t.voxel_vectors[1] + k * t.voxel_vectors[2]
        want = 0.0
        for c, coef in zip(cubes, (1, -1, -1)):
            f = np.linalg.solve(np.asarray(c.voxel_vectors).T, r - c.origin)
            ip = RegularGridInterpolator([np.arange(n) for n in c.data.shape], c.data, bounds_error=False, fill_value=0.0)
            want += coef * float(ip(f)[0])
        assert abs(got[i, j, k] - want) <= 1e-9 * max(1.0, abs(want)), (i, j, k, got[i, j, k], want)
    st, _, r = U.post(base, "api/volume/save", {"paths": paths, "coefficients": [1, -1, -1], "name": "vol/diff.CUBE"})
    assert st == 200, r
    saved = volume.read(str(root / "vol/diff.CUBE"), skip_vacuum=True)
    assert saved.data.shape == got.shape and np.abs(np.asarray(saved.data) - got).max() < 1e-4 * np.abs(got).max()
    for name in ("vol/diff.CUBE", "vol/mol_DENS.CUBE", "vol/mol.out"):
        st, _, r = U.post(base, "api/volume/save", {"paths": paths, "coefficients": [1, -1, -1], "name": name})
        assert st in (400, 409), (name, st, r)
    (root / "vol/diff.CUBE").unlink()


@needs_corpus
def test_routes_keep_the_path_and_token_rules(served, tmp_path):
    root, base, tmp = served
    outside = tmp / "outside_DENS.CUBE"
    shutil.copyfile(root / "vol/mol_DENS.CUBE", outside)
    link = root / "vol" / "link_DENS.CUBE"
    if not link.exists():
        os.symlink(outside, link)
    for q in ("../outside_DENS.CUBE", str(outside), "vol/link_DENS.CUBE", "vol/../../outside_DENS.CUBE"):
        for route in ("volume", "volume/slice"):
            st, _, r = U.req(base, f"api/{route}?path={q}")
            assert st in (400, 403, 404), (route, q, st)
    st, _, _ = U.req(base, "api/volume?path=vol/mol_DENS.CUBE", token=False)
    assert st == 403
    st, _, r = U.post(base, "api/volume/combine", {"paths": ["vol/mol_DENS.CUBE"] * 9, "coefficients": [1] * 9})
    assert st == 400 and "2 to 8" in r["error"]
    st, _, r = U.post(base, "api/volume/combine", {"paths": ["vol/mol_DENS.CUBE"] * 2, "coefficients": [1, float("nan")]})
    assert st == 400
    st, _, r = U.post(base, "api/volume/save", {"paths": ["vol/mol_DENS.CUBE", "vol/mol_SPIN.CUBE"], "coefficients": [1, -1],
                                               "name": "../escape.CUBE"})
    assert st in (400, 403) and not (tmp / "escape.CUBE").exists()
    link.unlink()


def test_round_off_negatives_do_not_make_a_density_signed(tmp_path):
    """-1e-15 in a density is round-off, not a negative lobe; a real difference stays signed."""
    from mace.gui import volume
    rng = np.random.default_rng(5)
    a = rng.random((6, 6, 6)) + 0.1
    a[0, 0, 0] = -1.1e-15
    write_cube(tmp_path / "a_DENS.CUBE", a)
    meta, _ = volume.payload(volume.read(str(tmp_path / "a_DENS.CUBE")), tmp_path / "a_DENS.CUBE", "low")
    assert meta["type"] == "density" and not meta["signed"]
    b = a.copy()
    b[1, 1, 1] = -0.05
    write_cube(tmp_path / "b_DENS.CUBE", b)
    meta, _ = volume.payload(volume.read(str(tmp_path / "b_DENS.CUBE")), tmp_path / "b_DENS.CUBE", "low")
    assert meta["signed"]
