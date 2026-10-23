"""mace.analysis (1.1.6): each tool checked against an independent reference.

- porosity: dense crystals have no room for N2; sodalite and HKUST-1 against
  Zeo++ 0.3 (-ha, the same CCDC radii, N2 r = 1.82 A); a single atom's
  surface area against the analytic sphere; Zeo++ itself when installed;
- PXRD: Si and NaCl peak positions and relative intensities with Cu K-alpha1
  (the values pymatgen's XRDCalculator gives for the same cells);
- Pawley and Rietveld recover a known cell, zero shift, B and an oxygen
  coordinate from synthetic patterns with Poisson noise; indexing finds the
  cubic, tetragonal, hexagonal and orthorhombic cells;
- topology: textbook nets (dia, pcu, bcu, fcu, sod), Cu2O as two
  interpenetrating dia nets, HKUST-1 as tbo; the built-in reference
  sequences are the published ones;
- polyhedra, distributions, morphology, planes on rutile, Si, NaCl, Cu.
"""
import math
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("spglib")
ase = pytest.importorskip("ase")
from ase.build import bulk  # noqa: E402
from ase.spacegroup import crystal  # noqa: E402

from mace.analysis import (geometry, indexing, morphology, planes, polyhedra, porosity, pxrd, rcsr, refine,  # noqa: E402
                           structure, topology, zeopp)

DATA = Path(__file__).parent / "data" / "analysis"
HKUST1 = DATA / "HKUST1_COD4002052.cif"


def st_of(atoms):
    return structure.from_atoms(atoms)


def sodalite():
    # IZA SOD framework (pure SiO2, DLS-optimized): Im-3m, a = 8.965 A
    return st_of(crystal(["Si", "O"], basis=[(0.25, 0.5, 0.0), (0.1467, 0.5, 0.8533)], spacegroup=229,
                         cellpar=[8.965] * 3 + [90] * 3))


def lta():
    # IZA LTA framework (pure SiO2, DLS-optimized): Pm-3m, a = 11.919 A; the alpha cage sits at the
    # body center, the beta (sodalite) cage at the corners, joined to the alpha cages by 6-rings only
    return st_of(crystal(["O", "O", "O", "Si"], basis=[(0, 0.2122, 0.5), (0.1103, 0.1103, 0.3384), (0, 0.2967, 0.2967),
                                                     (0, 0.1823, 0.3684)], spacegroup=221, cellpar=[11.919] * 3 + [90] * 3))


def pore_fields(r):
    """The two render fields as the browser gets them: (render, accessible, inaccessible)."""
    import base64
    rd = r["render"]
    dec = lambda k: np.frombuffer(base64.b64decode(rd[k]), "<f4").reshape(rd["shape"])
    return rd, dec("accessible"), dec("inaccessible")


def surface_points(F, rd):
    """Where a render field crosses its level along the grid edges (the vertices marching cubes
    places), in Angstrom through the same origin and voxel vectors viewport/pores.js gives 3Dmol."""
    pts = []
    for ax in range(3):
        n = F.shape[ax]
        a, b = np.take(F, range(n - 1), axis=ax), np.take(F, range(1, n), axis=ax)
        m = (a > rd["level"]) != (b > rd["level"])
        idx = np.argwhere(m).astype(float)
        idx[:, ax] += (a[m] - rd["level"]) / (a[m] - b[m])
        pts.append(idx)
    return np.asarray(rd["origin"]) + np.concatenate(pts) @ np.asarray(rd["voxels"])


def rutile():
    return st_of(crystal(["Ti", "O"], basis=[(0, 0, 0), (0.3048, 0.3048, 0)], spacegroup=136,
                         cellpar=[4.594, 4.594, 2.959, 90, 90, 90]))


# ---------------------------------------------------------------- porosity
@pytest.mark.parametrize("atoms", [bulk("Si", "diamond", a=5.431, cubic=True), bulk("NaCl", "rocksalt", a=5.64, cubic=True)])
def test_dense_crystal_has_no_room_for_nitrogen(atoms):
    r = porosity.analyze(st_of(atoms), probe="N2", spacing=0.2, surface_samples=200)
    assert r["AV"]["fraction"] == 0 and r["POAV"]["fraction"] == 0 and r["NAV"]["fraction"] == 0
    assert r["ASA"]["A2"] == 0 and r["Df"] < 2 * 1.82
    assert r["n_channels"] == 0


def test_sodalite_against_zeopp():
    r = porosity.analyze(sodalite(), probe="N2", spacing=0.15, render=True)
    # Zeo++ 0.3 -ha: Di 5.8205, Df 2.1322, NAV 0.0185, PONAV 0.3248; the beta cages are closed to N2.
    # The probe-occupiable volume is larger here (0.345): Zeo++ -volpo leaves out part of the
    # crevices between touching atoms; a brute-force Monte Carlo of the definition (any probe
    # center within r_p) agrees with this module to 1% (checked on CHA during development)
    assert r["Di"] == pytest.approx(5.8205, abs=0.02)
    assert r["Df"] == pytest.approx(2.1322, abs=0.01)
    assert r["NAV"]["fraction"] == pytest.approx(0.0185, abs=0.002)
    assert 0.3248 < r["PONAV"]["fraction"] < 0.36
    assert r["AV"]["fraction"] == 0 and r["n_pockets"] == 2 and r["n_channels"] == 0
    rd = r["render"]
    assert rd["has_inaccessible"] and not rd["has_accessible"]
    n = int(np.prod(rd["shape"]))
    import base64
    assert len(base64.b64decode(rd["inaccessible"])) == 4 * n


def test_sodalite_is_open_to_helium():
    r = porosity.analyze(sodalite(), probe=1.0, spacing=0.2, surface_samples=200)
    assert r["Df"] > 2.0 and r["AV"]["fraction"] > 0 and r["channel_dimensionality"] == 3


@pytest.mark.skipif(not HKUST1.is_file(), reason="fixture missing")
def test_hkust1_against_zeopp():
    st = structure.load(HKUST1)
    r = porosity.analyze(st, probe="N2", spacing=0.3, surface_samples=300)
    # Zeo++ 0.3 -ha, CCDC radii: Di 13.2814, Df 6.7011, AV fraction 0.2664, ASA 1894 m2/cm3
    assert r["Di"] == pytest.approx(13.2814, abs=0.03)
    assert r["Df"] == pytest.approx(6.7011, abs=0.08)
    assert r["AV"]["fraction"] == pytest.approx(0.2664, abs=0.01)
    assert r["ASA"]["m2_cm3"] == pytest.approx(1894.3, rel=0.04)
    assert r["channel_dimensionality"] == 3 and r["density_g_cm3"] == pytest.approx(0.8863, abs=0.001)


def test_single_atom_surface_area_is_the_sphere():
    st = structure.Structure(np.eye(3) * 20, [[0.5, 0.5, 0.5]], [18])
    r = porosity.analyze(st, probe=1.0, spacing=0.4)
    assert r["ASA"]["A2"] == pytest.approx(4 * math.pi * (1.88 + 1.0) ** 2, rel=1e-3)
    assert r["void_fraction_geometric"] == pytest.approx(1 - 4 / 3 * math.pi * 1.88 ** 3 / 8000, abs=2e-3)


def test_probe_presets_and_radii():
    assert porosity.probe_radius("CO2")[0] == pytest.approx(1.65)
    assert porosity.probe_radius("h2o")[1] == "H2O"
    assert porosity.probe_radius(1.4)[0] == 1.4
    with pytest.raises(ValueError):
        porosity.probe_radius("XeF6")
    st = sodalite()
    uff = porosity.atom_radii(st, "uff")
    assert uff[st.symbols.index("O")] == pytest.approx(3.5 / 2 ** (1 / 6) / 2)


def test_lta_against_zeopp():
    r = porosity.analyze(lta(), probe="N2", spacing=0.15, surface_samples=1000)
    # Zeo++ 0.3 -ha, CCDC radii, N2 r = 1.82 A: Di 10.2383, Df 3.8136, AV 0.11068, NAV 0.00298,
    # ASA 1073.0 and NASA 88.94 m2/cm3; one 3D channel system (the alpha cages through the 8-rings)
    # and one pocket (the beta cage). The grid carries about one spacing of error on the volumes;
    # the surface areas agree within 5%.
    assert r["Di"] == pytest.approx(10.2383, abs=0.04)
    assert r["Df"] == pytest.approx(3.8136, abs=0.01)
    assert r["AV"]["fraction"] == pytest.approx(0.11068, abs=0.002)
    assert r["NAV"]["fraction"] == pytest.approx(0.00298, abs=0.0005)
    assert r["ASA"]["m2_cm3"] == pytest.approx(1073.0, rel=0.05)
    assert r["NASA"]["m2_cm3"] == pytest.approx(88.94, rel=0.03)
    assert r["channel_dimensionality"] == 3 and r["n_channels"] == 1 and r["n_pockets"] == 1


def test_lta_reproduces_the_iza_database_spheres():
    # The IZA Database of Zeolite Structures lists for LTA a largest included sphere of 11.05 A and a
    # largest free sphere of 4.21 A along a, b and c; its convention (oxygen only, r = 1.32 A) gives them
    r = porosity.analyze(lta(), probe=0.0, radii={"O": 1.32, "Si": 0.01}, spacing=0.2, surface_samples=50)
    assert r["Di"] == pytest.approx(11.05, abs=0.02)
    assert r["Df"] == pytest.approx(4.21, abs=0.02)
    assert r["Df_axes"] == pytest.approx([4.21] * 3, abs=0.02)


@pytest.mark.parametrize("probe,beta_open", [("N2", False), ("He", False), (0.6, True)])
def test_lta_cages_accessible_or_not(probe, beta_open):
    # the alpha cages are a 3D channel system for all three; the beta cage's 6-rings let a sphere of
    # r = 0.6 A through but not He or N2 (Zeo++ 0.3 -ha: NAV 0 at r = 0.6 A, 0.031 at 0.7 A)
    r = porosity.analyze(lta(), probe=probe, spacing=0.2, surface_samples=200, render=True)
    rd, acc, nac = pore_fields(r)
    mid = tuple((n - 1) // 2 for n in rd["shape"])      # the grid includes the far faces: this is (1/2, 1/2, 1/2)
    assert acc[mid] > 0 and nac[mid] <= 0
    corners = [(i, j, k) for i in (0, -1) for j in (0, -1) for k in (0, -1)]
    assert all((acc[c] > 0) == beta_open and (nac[c] > 0) == (not beta_open) for c in corners)
    assert (r["n_pockets"] == 0) == beta_open and r["channel_dimensionality"] == 3


def test_pore_surfaces_sit_on_their_cages():
    # the surfaces drawn are the ones behind the numbers, in the right place of the cell: closed
    # (inaccessible) only around the beta cages at the corners, accessible only away from them
    st = lta()
    r = porosity.analyze(st, probe="N2", spacing=0.2, surface_samples=200, render=True)
    rd, acc, nac = pore_fields(r)
    assert rd["has_accessible"] and rd["has_inaccessible"]
    a = 11.919
    corner = lambda p: np.linalg.norm((p / a - np.rint(p / a)) * a, axis=1)
    pa, pn = surface_points(acc, rd), surface_points(nac, rd)
    assert len(pa) > 1000 and len(pn) > 500
    assert corner(pn).max() < 3.6            # the beta cage's probe-occupiable volume, radius about 3.2 A
    assert corner(pa).min() > 4.0            # the alpha side stops at the 6-rings, about 4.5 A from a corner
    # every vertex is on the pore wall: on an atom's surface, or in a crevice the probe touches
    d = porosity.SurfaceDistance(st.wrapped(), porosity.atom_radii(st))
    for p in (pa, pn):
        dd = d(p)
        assert dd.min() > -0.35 and np.percentile(dd, 99) < r["probe"]["radius"]
    # the meshes the 3D view draws (porosity.pore_meshes, smoothed): in the same places, on the wall
    ma, mn = mesh_of(r, "accessible"), mesh_of(r, "inaccessible")
    assert corner(mn["p"]).max() < 3.6 and corner(ma["p"]).min() > 4.0
    for m in (ma, mn):
        dd = d(m["p"])
        assert dd.min() > -0.35 and np.percentile(dd, 99) < r["probe"]["radius"]
    # the sodalite cages: closed to N2, drawn as pockets, nothing as accessible
    rs = porosity.analyze(sodalite(), probe="N2", spacing=0.2, surface_samples=200, render=True)
    rd, acc, nac = pore_fields(rs)
    assert not rd["has_accessible"] and (acc <= 0).all() and len(surface_points(nac, rd)) > 500
    assert rd["meshes"]["accessible"] is None and rd["meshes"]["inaccessible"]["n_pockets"] == 2


def mesh_of(r, kind):
    """A pore mesh as the browser gets it: positions (A), normals, faces, pocket per vertex."""
    import base64
    m = r["render"]["meshes"][kind]
    dec = lambda k, t: np.frombuffer(base64.b64decode(m[k]), t)
    return {"p": dec("positions", "<f4").reshape(-1, 3).astype(float), "n": dec("normals", "i1").reshape(-1, 3) / 127.0,
            "f": dec("faces", "<u4").reshape(-1, 3).astype(int), "pocket": dec("pocket", "<u2") if "pocket" in m else None,
            "n_pockets": m.get("n_pockets", 0)}


def pockets_of(m):
    """Each pocket's faces."""
    return [m["f"][m["pocket"][m["f"][:, 0]] == k] for k in range(m["n_pockets"])]


def closed_volume(p, f):
    """The volume a closed, outward-wound triangle surface encloses (divergence theorem)."""
    return float(np.einsum("ij,ij->i", p[f[:, 0]], np.cross(p[f[:, 1]], p[f[:, 2]])).sum() / 6)


def assert_closed_sphere(f):
    """Every edge in exactly two triangles, each way once (one orientation), and a sphere's
    Euler characteristic: a closed shell with no holes and no handles."""
    from mace.analysis.isosurface import edges_shared
    assert set(edges_shared(f).tolist()) == {2}
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    assert len(np.unique(e, axis=0)) == len(e)
    assert len(np.unique(f)) - len(edges_shared(f)) + len(f) == 2


def sodalite_primitive(left_handed=False):
    """Sodalite in its primitive (rhombohedral, 109.47 degree) cell: one cage per cell."""
    st = sodalite()
    a = st.cell[0, 0]
    P = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]) * a / 2
    if left_handed:
        P = P[[1, 0, 2]]
    f = np.mod(st.cart @ np.linalg.inv(P) + 1e-6, 1.0) - 1e-6
    _, keep = np.unique(np.round(f, 4) % 1.0, axis=0, return_index=True)
    return structure.Structure(P, f[keep], st.numbers[keep], (True, True, True))


@pytest.mark.parametrize("which", ["SOD", "LTA", "SOD primitive", "SOD primitive, left-handed"])
def test_pocket_meshes_are_closed_and_whole(which):
    # the pockets are drawn closed, each once and whole in the image around its own center (a
    # cage on a cell corner around that corner), enclosing the volume the numbers give (PONAV),
    # also in a non-orthogonal and a left-handed cell
    st = {"SOD": sodalite, "LTA": lta, "SOD primitive": sodalite_primitive,
          "SOD primitive, left-handed": lambda: sodalite_primitive(True)}[which]()
    r = porosity.analyze(st, probe="N2", spacing=0.2, surface_samples=200, render=True)
    m = mesh_of(r, "inaccessible")
    cages = {"SOD": 2, "LTA": 1}.get(which, 1)
    assert m["n_pockets"] == cages
    total = 0.0
    centers = []
    for f in pockets_of(m):
        assert_closed_sphere(f)
        v = closed_volume(m["p"], f)
        assert v > 0                                   # wound outward, whatever the cell's handedness
        total += v
        centers.append(m["p"][np.unique(f)].mean(0))
    assert total == pytest.approx(r["PONAV"]["A3"], rel=0.05)
    # the normals the browser shades with agree with the winding (by area: a sliver's own normal is noise)
    gn = np.cross(m["p"][m["f"][:, 1]] - m["p"][m["f"][:, 0]], m["p"][m["f"][:, 2]] - m["p"][m["f"][:, 0]])
    area = np.linalg.norm(gn, axis=1)
    assert area[(gn * m["n"][m["f"]].sum(1)).sum(1) > 0].sum() > 0.995 * area.sum()
    # each cage around a lattice point (the corner) or the body center, never cut by a cell face
    a = 8.965 if which != "LTA" else 11.919
    pts = np.array([[0, 0, 0], [a / 2] * 3])
    for c in centers:
        assert np.linalg.norm(pts - c, axis=1).min() < 0.05
    assert len({tuple(np.round(c, 1)) for c in centers}) == cages


def test_channel_mesh_coarsens_over_the_triangle_budget():
    # over the budget the channels are drawn from a coarser grid, still on the pore wall; the
    # pockets stay on the full grid, closed
    st = lta()
    r = porosity.analyze(st, probe="N2", spacing=0.2, surface_samples=200, render=True)
    rd, acc, nac = pore_fields(r)
    full = r["render"]["meshes"]
    out = porosity.pore_meshes(acc, nac, st.pbc, np.array(rd["voxels"]), rd["origin"], max_triangles=60_000)
    assert full["step"] == 1 and out["step"] == 2
    assert out["accessible"]["n_triangles"] < 0.5 * full["accessible"]["n_triangles"]
    assert out["inaccessible"]["n_triangles"] == full["inaccessible"]["n_triangles"]
    m = mesh_of({"render": {"meshes": out}}, "accessible")
    dd = porosity.SurfaceDistance(st.wrapped(), porosity.atom_radii(st))(m["p"])
    assert dd.min() > -0.35 and np.percentile(dd, 99) < r["probe"]["radius"]
    for f in pockets_of(mesh_of({"render": {"meshes": out}}, "inaccessible")):
        assert_closed_sphere(f)


def feature_sums(r):
    """(channels, pockets) of r["features"] and their summed volumes and areas."""
    F = r["features"]
    ch = [f for f in F if f["kind"] == "channel"]
    pk = [f for f in F if f["kind"] == "pocket"]
    tot = lambda L, k: sum(f[k] for f in L)
    return ch, pk, tot


@pytest.mark.parametrize("which,probe", [("SOD", "He"), ("SOD", "N2"), ("LTA", "N2"), ("LTA", 0.6)])
def test_features_add_up_to_the_totals(which, probe):
    # every channel and pocket on its own (the GUI's per-feature table): their volumes and areas add
    # up to AV, POAV, ASA (channels) and NAV, PONAV, NASA (pockets); each has its own included sphere
    import base64
    r = porosity.analyze({"SOD": sodalite, "LTA": lta}[which](), probe=probe, spacing=0.25, render=True)
    ch, pk, tot = feature_sums(r)
    assert len(ch) == r["n_channels"] and len(pk) == r["n_pockets"]
    assert [f["id"] for f in r["features"]] == list(range(len(r["features"])))
    near = lambda a, b: pytest.approx(b, rel=1e-9, abs=1e-6) == a
    assert near(tot(ch, "volume_A3"), r["AV"]["A3"]) and near(tot(pk, "volume_A3"), r["NAV"]["A3"])
    assert near(tot(ch, "occupiable_A3"), r["POAV"]["A3"]) and near(tot(pk, "occupiable_A3"), r["PONAV"]["A3"])
    assert near(tot(ch, "area_A2"), r["ASA"]["A2"]) and near(tot(pk, "area_A2"), r["NASA"]["A2"])
    assert all(f["accessible"] == (f["kind"] == "channel") for f in r["features"])
    assert all(f["dim"] == c["dim"] for f, c in zip(ch, r["channels"]))
    assert all(0 < f["Di"] <= r["Di"] + 1e-3 and f["n_points"] > 0 for f in r["features"])
    assert max(f["Di"] for f in r["features"]) == pytest.approx(r["Di"], abs=0.02)
    a = 8.965 if which == "SOD" else 11.919
    wrap = lambda x: np.abs((np.asarray(x) / a + 0.5) % 1.0 - 0.5) * a     # distance to the nearest lattice point, per axis
    if which == "SOD":
        # the two sodalite cages: one around the corner, one at the body center
        assert sorted(round(float(np.linalg.norm(wrap(f["centroid_cart"]))), 1) for f in pk) == [0.0, round(0.5 * a * 3 ** 0.5, 1)]
        assert all(np.linalg.norm(wrap(f["centroid_cart"]) - wrap(f["Di_center_cart"])) < 0.3 for f in pk)
    else:
        # the alpha cage (the channel system) at the body center: Di is its 11 A sphere
        assert np.allclose(wrap(ch[0]["Di_center_cart"]), [a / 2] * 3, atol=0.3) and ch[0]["Di"] > 10
        for f in pk:                     # the beta cage, closed to N2, at the corner
            assert np.linalg.norm(wrap(f["centroid_cart"])) < 0.3
    # the pore meshes number each vertex with its feature (uint16, 65535: none)
    for k, fs in (("accessible", ch), ("inaccessible", pk)):
        m = r["render"]["meshes"][k]
        if not fs:
            assert m is None
            continue
        ids = np.frombuffer(base64.b64decode(m["feature"]), "<u2")
        assert len(ids) == m["n_vertices"] and (ids != porosity.NO_FEATURE).all()
        assert m["features"] == sorted(f["id"] for f in fs) and set(ids.tolist()) == set(m["features"])


def test_features_of_a_slab_and_without_render():
    # no Voronoi network (a slab): one feature per grid region, still adding up; no render data needed
    slab = st_of(bulk("Si", "diamond", a=5.431, cubic=True))
    cell = slab.cell.copy()
    cell[2] *= 3
    s = structure.Structure(cell, slab.frac * [1, 1, 1 / 3], slab.numbers, (True, True, False))
    r = porosity.analyze(s, probe="N2", spacing=0.3, surface_samples=100)
    ch, pk, tot = feature_sums(r)
    assert len(ch) == r["n_channels"] >= 1 and len(pk) == r["n_pockets"]
    assert tot(ch, "volume_A3") == pytest.approx(r["AV"]["A3"], rel=1e-9)
    assert tot(ch, "occupiable_A3") == pytest.approx(r["POAV"]["A3"], rel=1e-9)
    assert tot(ch, "area_A2") == pytest.approx(r["ASA"]["A2"], rel=1e-9)
    # a pore mesh without the feature lookup (pore_meshes on its own) has no feature arrays
    st = lta()
    q = porosity.analyze(st, probe="N2", spacing=0.3, surface_samples=50, render=True)
    rd, acc, nac = pore_fields(q)
    out = porosity.pore_meshes(acc, nac, st.pbc, np.array(rd["voxels"]), rd["origin"])
    assert "feature" not in out["accessible"] and "feature" in q["render"]["meshes"]["accessible"]


def test_marching_tetrahedra_closes_and_orients():
    from mace.analysis.isosurface import marching_tetrahedra
    x = np.linspace(-1, 1, 33)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    ball = 0.6 - np.sqrt(X ** 2 + (1.3 * Y) ** 2 + Z ** 2)
    verts, faces, _ = marching_tetrahedra(ball)
    assert_closed_sphere(faces)
    p = verts * (x[1] - x[0]) - 1
    assert closed_volume(p, faces) == pytest.approx(4 / 3 * math.pi * 0.6 ** 3 / 1.3, rel=0.02)
    torus = 0.25 - np.sqrt((np.sqrt(X ** 2 + Y ** 2) - 0.6) ** 2 + Z ** 2)
    from mace.analysis.isosurface import edges_shared
    verts, faces, _ = marching_tetrahedra(torus)
    assert set(edges_shared(faces).tolist()) == {2} and len(verts) - len(edges_shared(faces)) + len(faces) == 0


def crystal_low_dim(kind):
    """Lower-dimensional structures as CRYSTAL outputs give them: a 0.5 A dummy vector along a
    non-periodic direction and atoms outside the [0, 1) range of it."""
    if kind == "slab":           # graphene
        cell = [[2.46, 0, 0], [-1.23, 2.1304, 0], [0, 0, 0.5]]
        return structure.Structure(cell, [[0, 0, 0], [1 / 3, 2 / 3, 0]], [6, 6], (True, True, False))
    if kind == "polymer":        # a carbon chain along a, far from the dummy box
        cell = [[2.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]
        return structure.Structure(cell, [[0, 12.0, 10.0], [0.5, 12.6, 10.0]], [6, 6], (True, False, False))
    cell = np.eye(3) * 0.5       # a CO2 molecule twenty boxes away
    return structure.Structure(cell, [[40, 40, 40], [40 + 1.16 / 0.5, 40, 40], [40 - 1.16 / 0.5, 40, 40]], [6, 8, 8],
                               (False, False, False))


@pytest.mark.parametrize("kind", ["slab", "polymer", "molecule"])
def test_lower_dimensional_structures_are_boxed_around_their_atoms(kind):
    st = crystal_low_dim(kind)
    r = porosity.analyze(st, probe="N2", spacing=0.25, surface_samples=200, render=True)
    # Di is bounded by the box (it was 1e50 A: the search ran off into the open vacuum)
    assert 0 < r["Di"] < 40 and 0 < r["AV"]["fraction"] < 1 and r["void_fraction_geometric"] < 1
    assert any("box" in w for w in r["warnings"]) and r["channels"] and r["channels"][0]["open"]
    # the accessible surface wraps the atoms where they are (Cartesian, as the viewer draws them)
    rd, acc, _ = pore_fields(r)
    pts = surface_points(acc, rd)
    cart = st.frac @ st.cell
    if not all(st.pbc):
        # periodic images along the periodic axes
        for k in range(3):
            if st.pbc[k]:
                cart = np.concatenate([cart + s * st.cell[k] for s in (-2, -1, 0, 1, 2)])
    near = np.min(np.linalg.norm(pts[:, None, :] - cart[None, :, :], axis=2), axis=1)
    R = porosity.atom_radii(st).max()
    assert len(pts) > 50 and near.max() < R + 0.5 and near.min() > 1.0


def test_slab_vacuum_is_open():
    slab = st_of(bulk("Si", "diamond", a=5.431, cubic=True))
    cell = slab.cell.copy()
    cell[2] *= 3
    s = structure.Structure(cell, slab.frac * [1, 1, 1 / 3], slab.numbers, (True, True, False))
    r = porosity.analyze(s, probe="N2", spacing=0.3, surface_samples=100)
    # the atoms touch the bottom face: the box is rebuilt with 5 A of vacuum on each side (14 A)
    assert r["AV"]["fraction"] > 0.15 and r["channels"] and r["channels"][0]["open"]
    assert any("2D" in w for w in r["warnings"])


@pytest.mark.skipif(not zeopp.find(), reason="Zeo++ is not installed")
def test_native_agrees_with_installed_zeopp():
    st = sodalite()
    z = zeopp.analyze(st, probe="N2")
    n = porosity.analyze(st, probe="N2", spacing=0.15)
    assert n["Di"] == pytest.approx(z["Di"], abs=0.03)
    assert n["Df"] == pytest.approx(z["Df"], abs=0.03)
    # LTA: the same accessible and closed volumes, channels and pockets
    z = zeopp.analyze(lta(), probe="N2", samples=20000, surface_samples=500)
    n = porosity.analyze(lta(), probe="N2", spacing=0.2, surface_samples=500)
    assert n["AV"]["fraction"] == pytest.approx(z["AV"]["fraction"], abs=0.003)
    assert n["NAV"]["fraction"] == pytest.approx(z["NAV"]["fraction"], abs=0.001)
    assert n["channel_dimensionality"] == z["channel_dimensionality"] == 3 and n["n_pockets"] == z["n_pockets"] == 1


# ---------------------------------------------------------------- PXRD
SI_REF = [(28.442, 100.0), (47.302, 66.7), (56.122, 39.6), (69.129, 10.7), (76.375, 16.3), (88.029, 23.5)]
NACL_REF = [(27.367, 8.1), (31.704, 100.0), (45.449, 65.6), (53.870, 1.9), (56.474, 21.2), (66.229, 9.3), (75.294, 24.7),
            (83.994, 18.2)]


@pytest.mark.parametrize("atoms,ref", [(bulk("Si", "diamond", a=5.431, cubic=True), SI_REF),
                                       (bulk("NaCl", "rocksalt", a=5.640, cubic=True), NACL_REF)])
def test_pattern_matches_reference(atoms, ref):
    peaks = [p for p in pxrd.reflections(st_of(atoms), "CuKa1", (10, 90)) if p["intensity"] > 1.5]
    assert len(peaks) == len(ref)
    for p, (tt, i) in zip(peaks, ref):
        assert p["two_theta"] == pytest.approx(tt, abs=0.002)
        assert p["intensity"] == pytest.approx(i, abs=max(1.0, 0.03 * i))


def test_miller_indices_and_multiplicity_in_the_conventional_cell():
    peaks = pxrd.reflections(st_of(bulk("Cu", "fcc", a=3.615)), "CuKa1", (40, 100))
    assert [tuple(p["hkl"]) for p in peaks[:3]] == [(1, 1, 1), (2, 0, 0), (2, 2, 0)]
    assert [p["multiplicity"] for p in peaks[:3]] == [8, 6, 12]


def test_simulated_profile_and_peak_search_round_trip():
    st = st_of(bulk("Si", "diamond", a=5.431, cubic=True))
    sim = pxrd.simulate(st, "CuKa1", (20, 90), 0.01, W=0.004)
    found = pxrd.find_peaks(sim["two_theta"], sim["intensity"], prominence=0.03)["peaks"]
    got = sorted(p["two_theta"] for p in found)
    assert len(got) == 6
    for a, (b, _) in zip(got, SI_REF):
        assert a == pytest.approx(b, abs=0.01)



@pytest.mark.parametrize("src, two_theta", [("MoKa", 155.0), ("AgKa", 150.0)])
def test_ka2_line_is_drawn_at_high_angle(src, two_theta):
    """At high angle Kalpha2 lies degrees above Kalpha1 (Ag near 150 deg: about 4.7 deg); the profile
    draws it whole, at half the Kalpha1 line's height, not clipped by the Kalpha1 line's window."""
    l1, l2, ratio, _ = pxrd.source(src)
    d = l1 / (2 * math.sin(math.radians(two_theta / 2)))
    t2 = math.degrees(2 * math.asin(l2 / (2 * d)))
    assert t2 - two_theta > 3.0
    x = np.arange(140.0, 170.0, 0.005)
    y = pxrd.profile([{"two_theta": two_theta, "d": d, "intensity": 1.0}], x, src, W=0.01)
    i1, i2 = int(np.argmin(abs(x - two_theta))), int(np.argmin(abs(x - t2)))
    assert y[i1] == pytest.approx(100.0, rel=1e-3)
    assert y[i2] == pytest.approx(ratio * 100.0, rel=0.02), (y[i2], t2)
    area2 = y[(x > t2 - 1.0) & (x < t2 + 1.0)].sum()
    area1 = y[(x > two_theta - 1.0) & (x < two_theta + 1.0)].sum()
    assert area2 == pytest.approx(ratio * area1, rel=0.02)

def test_read_pattern_text_and_xrdml(tmp_path):
    txt = "# comment\n2theta,I\n" + "\n".join(f"{10 + 0.1 * i:.2f},{100 + i}" for i in range(50))
    p = pxrd.read_pattern_text(txt, "x.csv")
    assert len(p["two_theta"]) == 50 and p["intensity"][0] == 100
    xml = ('<?xml version="1.0"?><xrdMeasurements xmlns="http://www.xrdml.com/XRDMeasurement/1.5"><xrdMeasurement>'
           '<scan><dataPoints><positions axis="2Theta" unit="deg"><startPosition>10</startPosition>'
           '<endPosition>20</endPosition></positions><counts unit="counts">' + " ".join(str(i) for i in range(11)) +
           '</counts></dataPoints></scan></xrdMeasurement></xrdMeasurements>')
    q = pxrd.read_pattern_text(xml, "s.xrdml")
    assert q["two_theta"][0] == 10 and q["two_theta"][-1] == 20 and len(q["intensity"]) == 11


# ---------------------------------------------------------------- refinement and indexing
def test_pawley_recovers_cell_and_zero():
    si = st_of(bulk("Si", "diamond", a=5.431, cubic=True))
    pat = refine.synthetic_pattern(si, zero=0.03)
    r = refine.fit(pat, "pawley", cell=st_of(bulk("Si", "diamond", a=5.40, cubic=True)).cell)
    assert r["lattice"]["a"] == pytest.approx(5.431, abs=2e-4)
    assert r["zero"] == pytest.approx(0.03, abs=0.005)
    assert r["n_cell_parameters"] == 1 and r["statistics"]["GOF"] < 1.3


def test_rietveld_recovers_cell_b_and_oxygen_position():
    pat = refine.synthetic_pattern(rutile(), b_iso=0.6)
    start = st_of(crystal(["Ti", "O"], basis=[(0, 0, 0), (0.295, 0.295, 0)], spacegroup=136,
                          cellpar=[4.57, 4.57, 2.97, 90, 90, 90]))
    r = refine.fit(pat, "rietveld", structure=start, refine_positions=True)
    assert r["lattice"]["a"] == pytest.approx(4.594, abs=5e-4) and r["lattice"]["c"] == pytest.approx(2.959, abs=5e-4)
    assert r["n_cell_parameters"] == 2
    ox = [f for f, s in zip(r["frac"], r["symbols"]) if s == "O"][0]
    assert ox[0] == pytest.approx(0.3048, abs=2e-3) and ox[0] == pytest.approx(ox[1], abs=1e-9)
    assert r["B"] == pytest.approx(0.6, abs=0.15) and r["statistics"]["GOF"] < 1.3


def test_strain_basis_follows_the_point_group():
    for atoms, k in [(bulk("Si", "diamond", a=5.431, cubic=True), 1), (bulk("ZnO", "wurtzite", a=3.25, c=5.207), 2)]:
        st = st_of(atoms)
        rot, _, _ = refine.symmetry_ops(st)
        assert len(refine.strain_basis(st.cell, rot)) == k


@pytest.mark.parametrize("atoms,system,abc", [
    (bulk("Si", "diamond", a=5.431, cubic=True), "cubic", (5.431, 5.431, 5.431)),
    (crystal(["Ti", "O"], basis=[(0, 0, 0), (0.3048, 0.3048, 0)], spacegroup=136, cellpar=[4.594, 4.594, 2.959, 90, 90, 90]),
     "tetragonal", (4.594, 4.594, 2.959)),
    (bulk("ZnO", "wurtzite", a=3.25, c=5.207), "hexagonal", (3.25, 3.25, 5.207)),
    (crystal(["Cu"], basis=[(0, 0, 0)], spacegroup=47, cellpar=[3.1, 4.3, 5.7, 90, 90, 90]), "orthorhombic", (3.1, 4.3, 5.7)),
])
def test_ab_initio_indexing(atoms, system, abc):
    pk = pxrd.reflections(st_of(atoms), "CuKa1", (10, 100))
    tt = np.array(sorted(p["two_theta"] for p in pk if p["intensity"] > 1))[:20]
    tt = tt + np.random.default_rng(3).normal(0, 0.005, size=len(tt))
    s = indexing.index_ab_initio(tt, "CuKa1")["solutions"][0]
    assert s["system"] == system and s["indexed"] == s["total"]
    got = sorted([s["lattice"]["a"], s["lattice"]["b"], s["lattice"]["c"]])
    assert got == pytest.approx(sorted(abc), abs=2e-3)


def test_indexing_with_a_structure_cell():
    st = rutile()
    tt = [p["two_theta"] for p in pxrd.reflections(st, "CuKa", (20, 80))]
    r = indexing.index_with_cell(tt, None, "CuKa", structure=st)
    assert r["indexed"] == r["total"] and r["lattice"]["a"] == pytest.approx(4.594, abs=1e-3)


# ---------------------------------------------------------------- topology
@pytest.mark.parametrize("atoms,name", [
    (bulk("Si", "diamond", a=5.431, cubic=True), "dia"), (bulk("NaCl", "rocksalt", a=5.64, cubic=True), "pcu"),
    (bulk("CsCl", "cesiumchloride", a=4.12), "bcu"), (bulk("Cu", "fcc", a=3.615), "fcu"),
])
def test_textbook_nets(atoms, name):
    r = topology.analyze(st_of(atoms))
    assert r["summary"]["topologies"] == [name] and r["summary"]["interpenetration"] == 1


def test_sodalite_net_and_point_symbol():
    r = topology.analyze(sodalite())
    n = r["nets"][0]
    assert n["topology"] == "sod" and n["vertex_classes"][0]["point_symbol"] == "4^2.6^4"


def test_cuprite_is_two_interpenetrating_diamond_nets():
    cu2o = st_of(crystal(["O", "Cu"], basis=[(0, 0, 0), (0.25, 0.25, 0.25)], spacegroup=224, cellpar=[4.27] * 3 + [90] * 3,
                         setting=2))
    r = topology.analyze(cu2o)
    assert r["summary"]["interpenetration"] == 2 and r["summary"]["topologies"] == ["dia"]


@pytest.mark.skipif(not HKUST1.is_file(), reason="fixture missing")
def test_hkust1_is_tbo():
    r = topology.analyze(structure.load(HKUST1))
    assert r["mode"] == "sbu" and r["summary"]["topologies"] == ["tbo"]
    cls = {c["cn"]: c for c in r["nets"][0]["vertex_classes"]}
    assert cls[3]["count"] == 32 and cls[4]["count"] == 24 and cls[3]["point_symbol"] == "6^3"


# published coordination sequences (O'Keeffe et al.) the built-in embeddings must reproduce
KNOWN_CS = {
    "dia": [[4, 12, 24, 42, 64, 92, 124, 162, 204, 252]],
    "pcu": [[6, 18, 38, 66, 102, 146, 198, 258, 326, 402]],
    "srs": [[3, 6, 12, 24, 35, 48, 69, 86, 108, 138]],
    "fcu": [[12, 42, 92, 162, 252, 362, 492, 642, 812, 1002]],
    "hcb": [[3, 6, 9, 12, 15, 18, 21, 24, 27, 30]],
    "sql": [[4, 8, 12, 16, 20, 24, 28, 32, 36, 40]],
}


def test_builtin_reference_sequences():
    ref = {n["name"]: n for n in rcsr.builtin()}
    for name, cs in KNOWN_CS.items():
        assert ref[name]["cs"] == cs, name
    assert ref["dia"]["td10"] == 981 and ref["srs"]["td10"] == 530


def test_rcsr_import_of_a_3dall_style_file(tmp_path):
    # a file in RCSR's 3dall.txt layout (made up here: two nets with this module's own sequences)
    blk = lambda i, name, cs: "\n".join([
        "start", str(i), name, "1a", " 0  !number of other symbols", " 0  !number of names", " 0  !number of other names",
        " 0  !number of keywords", " 0  !number of references", "Pm-3m   221", "  1.0 1.0 1.0 90.0 90.0 90.0", "1",
        "V1  " + str(cs[0]), "  0 0 0", "  0, 0, 0", "  1 a", "  m-3m", "  48", "1", "E1  2", "  0.5 0 0", "  1/2, 0, 0",
        "  3 d", "  4/mmm", "1", "1", "1", "[x]", "unk", "  " + "  ".join(map(str, cs)) + "  " + str(1 + sum(cs)),
        "  x", "1", "1"])
    text = "\n".join([blk(1, "zzz", [6, 18, 38, 66, 102, 146, 198, 258, 326, 402]), blk(2, "yyy", [3, 3, 3, 3, 3, 3, 3, 3, 3, 3])])
    nets = rcsr.parse_3dall(text)
    assert [n["name"] for n in nets] == ["zzz", "yyy"] and nets[0]["td10"] == 1561
    dest = tmp_path / "nets.json"
    rcsr.import_file(_write(tmp_path / "3dall.txt", text), dest)
    vc = [{"cs": [6, 18, 38, 66, 102, 146, 198, 258, 326, 402]}]
    import json
    assert [m["name"] for m in rcsr.match(vc, reference=json.loads(dest.read_text())["nets"])] == ["zzz"]


def _write(p, text):
    p.write_text(text)
    return p


# ---------------------------------------------------------------- VESTA / Materials Studio tools
def test_rutile_octahedron():
    p = polyhedra.analyze(rutile())["polyhedra"][0]
    assert p["element"] == "Ti" and p["cn"] == 6 and p["shape"] == "octahedron"
    assert p["bond_min"] == pytest.approx(1.9487, abs=1e-3) and p["bond_max"] == pytest.approx(1.9803, abs=1e-3)
    assert 1.0 < p["quadratic_elongation"] < 1.02 and 5.9 < p["econ"] <= 6.0


def test_ideal_tetrahedron_scores_zero():
    p = polyhedra.analyze(sodalite())["polyhedra"][0]
    assert p["shape"] == "tetrahedron" and p["cshm"] < 0.05
    assert p["angle_variance"] < 1.0


def test_bond_and_angle_distributions():
    g = geometry.analyze(rutile())
    assert g["bonds"]["O-Ti"]["n"] == 12 and g["bonds"]["O-Ti"]["mean"] == pytest.approx(1.9592, abs=1e-3)
    assert set(g["angles"]) == {"O-Ti-O", "Ti-O-Ti"}
    assert sum(g["bond_histogram"]["counts"]) == 12


def test_morphology_bfdh_and_wulff():
    m = morphology.wulff(st_of(bulk("Si", "diamond", a=5.431, cubic=True)))
    assert [f["hkl"] for f in m["families"]] == [[1, 1, 1]]          # BFDH: an octahedron
    w = morphology.wulff(st_of(bulk("Cu", "fcc", a=3.615)), energies={(1, 1, 1): 1.0, (1, 0, 0): 1.15})
    fam = {tuple(f["hkl"]): f["area_fraction"] for f in w["families"]}
    assert set(fam) == {(1, 1, 1), (1, 0, 0)} and fam[(1, 1, 1)] > fam[(1, 0, 0)]
    assert sum(fam.values()) == pytest.approx(1.0)
    assert morphology.parse_energies("100:1.2, 1-10:0.9") == {(1, 0, 0): 1.2, (1, -1, 0): 0.9}


def test_cleavage_ranking():
    si = planes.cleavage(st_of(bulk("Si", "diamond", a=5.431, cubic=True)))
    assert si["families"][0]["hkl"] == [1, 1, 1]
    nacl = planes.cleavage(st_of(bulk("NaCl", "rocksalt", a=5.64, cubic=True)))
    assert nacl["families"][0]["hkl"] == [1, 0, 0]


def test_cleavage_ignores_float_noise_in_the_positions():
    # the GUI's structure round trip moves atoms by ~1e-16; -1e-17 wrapped to exactly 1.0
    # and bonds() (wrapping again) gave image shifts off by a cell, ranking (100) first
    w = structure.Structure(np.eye(3), [[-1e-17, 0.5, 1 - 1e-17]], [14]).wrapped()
    assert np.all(w.frac < 1) and np.array_equal(w.wrapped().frac, w.frac)
    base = st_of(bulk("Si", "diamond", a=5.431, cubic=True))
    rng = np.random.default_rng(116)
    for eps in np.logspace(-17, -10, 15):
        s = structure.Structure(base.cell, base.frac + eps * rng.uniform(-1, 1, base.frac.shape), base.numbers)
        fam = {tuple(f["hkl"]): f["min_per_nm2"] for f in planes.cleavage(s)["families"]}
        # bonds cut per surface atom x atoms per nm^2: 1 x 4/(sqrt(3) a^2), 1 x 4/(sqrt(2) a^2), 2 x 2/a^2
        a2 = 5.431 ** 2 / 100
        assert fam[(1, 1, 1)] == pytest.approx(4 / math.sqrt(3) / a2, rel=1e-6), eps
        assert fam[(1, 1, 0)] == pytest.approx(4 / math.sqrt(2) / a2, rel=1e-6), eps
        assert fam[(1, 0, 0)] == pytest.approx(4 / a2, rel=1e-6), eps
        assert min(fam, key=fam.get) == (1, 1, 1), eps


def test_plane_polygon_and_spacing():
    cell = np.eye(3) * 4.0
    assert planes.d_spacing(cell, (1, 1, 1)) == pytest.approx(4 / math.sqrt(3))
    hexa = planes.polygon(cell, (1, 1, 1), 1.5)
    assert len(hexa) == 6
    tri = planes.polygon(cell, (1, 1, 1), 0.5)
    assert len(tri) == 3


def test_layered_crystal_reads_as_layers():
    hbn = structure.load(Path(__file__).parent.parent / "examples" / "gui" / "structures" / "hBN.cif")
    r = topology.analyze(hbn)
    assert r["summary"]["interpenetration"] is None and r["summary"]["topologies"] == ["hcb"]
    assert "layers" in r["summary"]["note"] and all(n["periodicity"] == 2 for n in r["nets"])
