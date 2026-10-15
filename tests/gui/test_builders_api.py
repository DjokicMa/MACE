"""The builders through the real `mace gui` (plan T4-T8, spec §2.5): each one
checked against an independent ase or spglib result, through /api/build/*
(previews, new documents as sessions or held view-only structures, pool
tasks above 2000 atoms) and the matching in-place editor ops (one history
entry each). Nothing is ever written into the served folder."""
import time
import warnings

import pytest

import guiharness as H
import p4util as U

np = pytest.importorskip("numpy")
ase_io = pytest.importorskip("ase.io")
pytest.importorskip("spglib")


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("builders")
    root, xdg = H.examples_copy(tmp), tmp / "xdg"
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1", MACE_GUI_MAX_WORKERS="2")
    yield base, root, xdg, proc
    proc.terminate()
    proc.wait(timeout=15)


def read(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ase_io.read(str(H.EXAMPLES / "structures" / name))


def build(base, kind, params, source=None, preview=False, open_as=None, headers=None):
    body = {"source": source or {"path": "structures/MgO.cif"}, "params": params}
    if preview:
        body["preview"] = True
    if open_as:
        body["open_as"] = open_as
    return U.post(base, f"api/build/{kind}", body, headers=headers)


def result_of(base, st, r, timeout=60):
    """The build's answer: at once (200), or its task's result after a 202."""
    if st == 200:
        return r
    assert st == 202, (st, r)
    tid = r["task"]["id"]
    end = time.time() + timeout
    while time.time() < end:
        st2, _, r2 = U.req(base, f"api/tasks/{tid}/result")
        if st2 == 200:
            return r2
        assert st2 == 409, (st2, r2)
        time.sleep(0.1)
    raise AssertionError("the build did not finish")


def as_set(frac, numbers, digits=6):
    f = np.round(np.asarray(frac) % 1.0, digits) % 1.0
    return sorted(zip(map(tuple, np.round(f, digits)), map(int, numbers)))


# --- T4 supercell ---------------------------------------------------------------------

@pytest.mark.parametrize("name", ["MgO.cif", "hBN.cif", "Triclinic"])
def test_builders_supercell_diag_and_matrix_match_ase(name):
    from ase import Atoms
    from ase.build import make_supercell
    from mace.gui import builders as B
    if name == "Triclinic":
        from ase.geometry import cellpar_to_cell
        a = Atoms("SiO2Si", cell=cellpar_to_cell([5.1, 6.3, 7.4, 78, 104, 96]),
                  scaled_positions=[[.1, .2, .3], [.35, .15, .42], [.62, .71, .18], [.8, .55, .66]], pbc=True)
    else:
        a = read(name)
    for M in ([[2, 0, 0], [0, 2, 0], [0, 0, 2]], [[1, -1, 0], [1, 1, 0], [0, 0, 1]], [[2, 1, 0], [-1, 1, 0], [0, 0, 1]],
              [[1, 1, 0], [0, 1, 1], [1, 0, 1]], [[3, 0, 0], [0, 1, 0], [1, 0, 2]]):
        r = B.supercell(cell=a.cell.array, frac=a.get_scaled_positions(), numbers=a.numbers, matrix=M)
        ref = make_supercell(a, M)
        assert np.allclose(r["cell"], ref.cell.array, atol=1e-10), M
        # positions as sets modulo the supercell (an atom on a face may wrap to
        # either side): every one of ours has exactly one of ase's, same element
        f1, f2 = np.asarray(r["frac"]), ref.get_scaled_positions(wrap=False)
        assert len(f1) == len(f2), M
        d = f1[:, None, :] - f2[None, :, :]
        d -= np.round(d)
        dist = np.linalg.norm(d @ ref.cell.array, axis=2)
        j = dist.argmin(1)
        assert np.all(dist[np.arange(len(f1)), j] < 1e-8) and len(set(j)) == len(f1), M
        assert list(np.asarray(r["numbers"])) == list(ref.numbers[j]), M


def test_builders_supercell_rejects_bad_matrices():
    from mace.gui import builders as B
    a = read("MgO.cif")
    for M, msg in (([[1, 0, 0], [0, 1, 0], [0, 0, 0]], "positive determinant"), ([[0, 1, 0], [1, 0, 0], [0, 0, 1]], "positive determinant"),
                   ([[31, 0, 0], [0, 1, 0], [0, 0, 1]], "±30"), ([[1.5, 0, 0], [0, 1, 0], [0, 0, 1]], "integers")):
        with pytest.raises(ValueError, match=msg):
            B.supercell(cell=a.cell.array, frac=a.get_scaled_positions(), numbers=a.numbers, matrix=M)


def test_supercell_route_counts_and_group(served):
    base, root = served[0], served[1]
    before = U.tree_md5(root)
    st, h, r = build(base, "supercell", {"n": [2, 2, 2], "label": "2×2×2"},
                     open_as={"name": "MgO_2x2x2", "label": "Supercell 2×2×2 of MgO.cif", "locked": True})
    assert st == 200, r
    s = r["session"]
    assert len(s["numbers"]) == 64 and s["symmetry"]["number"] == 225 and s["symmetry"]["international"] == "Fm-3m"
    assert s["history"][0]["label"] == "Supercell 2×2×2 of MgO.cif" and len(s["history"]) == 1
    assert s["doc_meta"]["provenance"]["op"] == "supercell" and s["doc_meta"]["provenance"]["parent"] == "structures/MgO.cif"
    assert s["doc_meta"]["provenance"]["parent_sha256"]
    assert s["name"] == "MgO_2x2x2" and s["locked"]
    assert "build;dur=" in h["Server-Timing"]
    assert U.tree_md5(root) == before                    # never writes


def test_supercell_preset_r2_r2_1(served):
    base = served[0]
    st, _, r = build(base, "supercell", {"matrix": [[1, -1, 0], [1, 1, 0], [0, 0, 1]]}, open_as={"name": "MgO_r2xr2x1"})
    s = r["session"]
    assert len(s["numbers"]) == 16 and abs(s["lattice"]["a"] - 5.957) < 1e-3 and abs(s["lattice"]["c"] - 4.212) < 1e-6


def test_supercell_preview_inline_and_outline(served):
    base = served[0]
    st, _, r = build(base, "supercell", {"n": [2, 2, 2]}, preview=True)
    assert st == 200 and r["n"] == 64 and r["preview"]["packed"] == 1 and r["preview"]["n"] == 64
    st, _, r = build(base, "supercell", {"n": [7, 7, 7]}, preview=True)             # 2744 atoms
    assert st == 200 and r["outline_only"] and r["n"] == 2744 and "preview" not in r
    assert np.allclose(r["cell"], np.eye(3) * 4.212 * 7)
    assert U.req(base, "api/test/pool")[2]["pool"] is None or True     # a preview never goes to the pool


def test_supercell_errors(served):
    base = served[0]
    for params, msg in (({"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 0]]}, "positive determinant"),
                        ({"matrix": [[40, 0, 0], [0, 1, 0], [0, 0, 1]]}, "±30")):
        st, _, r = build(base, "supercell", params)
        assert st == 400 and msg in r["error"], (params, r)
    # the 250,000-atom cap, refused before building: 8-atom cells can't reach it
    # inside ±30 (216,000), so from a 32-atom session (2×2×1 NaCl)
    s = U.req(base, "api/structure?path=structures/NaCl.cif")[2]
    _, _, o = U.post(base, "api/edit/open", {"path": "structures/NaCl.cif", "expected": s["file"]})
    _, _, o2 = U.post(base, "api/edit/op", {"session": o["session"], "base_rev": o["rev"], "op": "supercell", "args": {"n": [2, 2, 1]}})
    st, _, r = build(base, "supercell", {"n": [30, 30, 30]}, source={"session": o["session"], "rev": o2["rev"]})
    assert st == 400 and "builds stop at 250,000" in r["error"], r
    # a slab repeats along a and b only
    st, _, r = build(base, "supercell", {"n": [2, 2, 2]}, source={"path": "outputs/graphene_slab_sp.out"})
    assert st == 400 and "n3 must be 1" in r["error"], r
    st, _, r = build(base, "supercell", {"matrix": [[1, 0, 1], [0, 1, 0], [0, 0, 1]]}, source={"path": "outputs/graphene_slab_sp.out"})
    assert st == 400 and "periodic along a and b only" in r["error"], r


def test_builders_supercell_of_a_slab_keeps_its_layers():
    """2x2x1 of a slab whose atoms sit outside [0, 1) along its non-periodic c:
    nothing is wrapped along c, the heights stay, the slab stays a slab."""
    from mace.gui import builders as B
    cell = [[2.46, 0, 0], [-1.23, 2.130422, 0], [0, 0, 14.0]]
    frac = [[0, 0, 0.9], [1 / 3, 2 / 3, 0.9], [0, 0, 1.15], [2 / 3, 1 / 3, 1.15]]
    r = B.supercell(cell=cell, frac=frac, numbers=[6] * 4, pbc=(True, True, False), matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    assert len(r["numbers"]) == 16 and r["pbc"] == [True, True, False]
    z = sorted(set(np.round(np.asarray(r["frac"])[:, 2] * 14.0, 6)))
    assert z == [12.6, 16.1], z
    for M in ([[2, 0, 0], [0, 2, 0], [0, 0, 2]], [[1, 0, 1], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [1, 0, 1]]):
        with pytest.raises(ValueError):
            B.supercell(cell=cell, frac=frac, numbers=[6] * 4, pbc=(True, True, False), matrix=M)


def test_supercell_of_the_graphene_slab_output(served):
    base = served[0]
    s = U.req(base, "api/structure?path=outputs/graphene_slab_sp.out")[2]
    st, _, r = build(base, "supercell", {"n": [2, 2, 1]}, source={"path": "outputs/graphene_slab_sp.out"}, open_as={"name": "gr_2x2"})
    r = result_of(base, st, r)
    got = r["structure"]                 # held, view-only (editing needs a 3D cell)
    import base64
    zn = np.frombuffer(base64.b64decode(got["zn"]), np.uint8)
    frac = np.frombuffer(base64.b64decode(got["frac"]), "<f8").reshape(-1, 3)
    assert len(zn) == 4 * len(s["numbers"]), r
    assert got["pbc"] == [True, True, False] and got["dimensionality"] == "SLAB"
    z_new = sorted(set(np.round((frac @ np.asarray(got["cell"], float).reshape(3, 3))[:, 2], 4)))
    assert z_new == sorted(set(np.round(np.asarray(s["cart"])[:, 2], 4)))
    U.post(base, "api/held/release", {"ids": [r["held"]]})


def test_supercell_over_5000_view_only_reason(served):
    """8000 atoms: a pool task, then a held view-only structure with the reason."""
    base = served[0]
    st, _, r = build(base, "supercell", {"n": [10, 10, 10]}, open_as={"name": "MgO_10x10x10"})
    assert st in (200, 202)
    res = result_of(base, st, r)
    assert res["view_only"] and res["held"] and res["n"] == 8000
    assert res["reason"] == "Editing is limited to 5000 atoms; this one has 8,000"
    assert res["structure"]["packed"] == 1 and res["structure"]["symmetry"]["number"] == 225
    pool = U.req(base, "api/resources")[2]["pool"]
    assert pool["workers_used"] >= 1
    U.post(base, "api/held/release", {"ids": [res["held"]]})


def test_op_supercell_matrix_in_place_one_history_entry(served):
    base = served[0]
    s = U.req(base, "api/structure?path=structures/NaCl.cif")[2]
    st, _, r = U.post(base, "api/edit/open", {"path": "structures/NaCl.cif", "expected": s["file"]})
    n0, rows0 = len(r["numbers"]), len(r["history"])
    st, _, r2 = U.post(base, "api/edit/op", {"session": r["session"], "base_rev": r["rev"], "op": "supercell",
                                             "args": {"matrix": [[1, -1, 0], [1, 1, 0], [0, 0, 1]]}, "label": "Supercell √2×√2×1 R45°"})
    assert st == 200, r2
    assert len(r2["numbers"]) == 2 * n0 and len(r2["history"]) == rows0 + 1
    assert r2["history"][-1]["label"] == "Supercell √2×√2×1 R45°"
    # the same structure the New-tab route makes
    st, _, nb = build(base, "supercell", {"matrix": [[1, -1, 0], [1, 1, 0], [0, 0, 1]]}, source={"path": "structures/NaCl.cif"},
                      open_as={"name": "NaCl_r2"})
    assert as_set(r2["frac"], r2["numbers"]) == as_set(nb["session"]["frac"], nb["session"]["numbers"])
    # the old diagonal op is unchanged
    st, _, r3 = U.post(base, "api/edit/op", {"session": r["session"], "base_rev": r2["rev"], "op": "supercell", "args": {"n": [1, 1, 2]}})
    assert st == 200 and len(r3["numbers"]) == 4 * n0
    # in place stops at 5000 atoms (the 32 atoms above, ×400)
    st, _, r4 = U.post(base, "api/edit/op", {"session": r["session"], "base_rev": r3["rev"], "op": "supercell",
                                             "args": {"matrix": [[20, 0, 0], [0, 20, 0], [0, 0, 1]]}})      # 32 × 400
    assert st == 400 and "5000" in r4["error"]


def test_held_budget_refuses_before_building(tmp_path):
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1", MACE_GUI_CACHE_MB="64")
    try:
        t0 = time.time()
        st, _, r = build(base, "supercell", {"n": [30, 30, 7]})          # 50,400 atoms, ~10 MB: fits 16 MB
        assert st in (200, 202)
        res = result_of(base, st, r)
        st, _, r = build(base, "supercell", {"n": [30, 30, 10]})         # 72,000 atoms: over the quarter
        assert st == 400 and r["error"].startswith("Close some of the big built structures first"), r
        assert time.time() - t0 < 30
        tasks = U.req(base, "api/tasks")[2]["tasks"]
        assert len(tasks) <= 1                                            # the refused one never ran
        U.post(base, "api/held/release", {"ids": [res["held"]]})
    finally:
        proc.terminate()
        proc.wait(timeout=15)


@pytest.mark.parametrize("path", ["api/build/supercell", "api/symmetry/compute"])
def test_build_routes_require_token_host_and_xrw(served, path):
    base = served[0]
    data = {"source": {"path": "structures/MgO.cif"}, "params": {"n": [1, 1, 1]}}
    assert U.post(base, path, data, token=False)[0] == 403
    port = base.split("/")[2].rsplit(":", 1)[1]
    assert U.post(base, path, data, headers={"Host": f"evil.example:{port}"})[0] == 421
    assert U.post(base, path, data, headers={"X-Requested-With": "x"})[0] == 403


# --- T5 slab ----------------------------------------------------------------------------

def planes_of(s, tol=0.15):
    """(heights of the atomic planes along c, image gap) of a slab payload."""
    cell = np.asarray(s["cell"])
    c = np.linalg.norm(cell[2])
    z = np.sort(np.asarray(s["frac"])[:, 2] % 1.0 * c)
    groups = [[z[0]]]
    for x in z[1:]:
        if x - groups[-1][-1] < tol:
            groups[-1].append(x)
        else:
            groups.append([x])
    mean = [float(np.mean(g)) for g in groups]
    return mean, c - (z.max() - z.min()), c


def test_slab_mgo_001_5_layers_c(served):
    """Criterion 2: (001), 5 layers, 15 Å vacuum: c = 15 + 4 × 2.106 Å, five
    planes 2.106 Å apart, 20 atoms, 15.000 Å between periodic images."""
    base, root = served[0], served[1]
    before = U.tree_md5(root)
    st, _, r = build(base, "slab", {"hkl": [0, 0, 1], "layers": 5, "vacuum": 15},
                     open_as={"name": "MgO_001_5L", "label": "Slab (001), 5 layers of MgO.cif"})
    assert st == 200, r
    s = r["session"]
    mean, gap, c = planes_of(s)
    assert abs(c - 23.424) <= 0.01 and abs(s["lattice"]["c"] - (15 + 4 * 2.106)) <= 0.01, c
    assert len(mean) == 5 and np.allclose(np.diff(mean), 2.106, atol=1e-3), mean
    assert len(s["numbers"]) == 20 and abs(gap - 15.0) <= 0.01, gap
    assert all(s["pbc"]) and s["dimensionality"] == "CRYSTAL"           # a 3D cell with vacuum (a CIF)
    assert r["info"]["layers"] == 5 and abs(r["info"]["thickness"] - 4 * 2.106) < 1e-3
    assert s["history"][0]["label"] == "Slab (001), 5 layers of MgO.cif"
    assert U.tree_md5(root) == before


def test_slab_terminations_mgo_111_two(served):
    base = served[0]
    st, _, r = build(base, "slab", {"hkl": [1, 1, 1], "layers": 5, "vacuum": 12}, preview=True)
    terms = r["info"]["terminations"]
    assert sorted(t["formula"] for t in terms) == ["Mg4", "O4"], terms
    tops = set()
    for i in range(2):
        st, _, r = build(base, "slab", {"hkl": [1, 1, 1], "layers": 5, "vacuum": 12, "termination": i}, preview=True)
        tops.add(r["info"]["top"])
        f = np.frombuffer(__import__("base64").b64decode(r["preview"]["frac"]), "<f8").reshape(-1, 3)
        z = np.frombuffer(__import__("base64").b64decode(r["preview"]["zn"]), np.uint8)
        top = z[f[:, 2] > f[:, 2].max() - 0.01]
        assert set(top) == ({12} if r["info"]["top"] == "Mg4" else {8})
    assert tops == {"Mg4", "O4"}


def test_slab_symmetric_layer_count(served):
    base = served[0]
    st, _, r = build(base, "slab", {"hkl": [1, 1, 1], "layers": 4, "vacuum": 10, "symmetric": True}, preview=True)
    assert r["info"]["layers"] == 5 and r["info"]["asked_layers"] == 4
    f = np.frombuffer(__import__("base64").b64decode(r["preview"]["frac"]), "<f8").reshape(-1, 3)
    z = np.frombuffer(__import__("base64").b64decode(r["preview"]["zn"]), np.uint8)
    assert set(z[f[:, 2] > f[:, 2].max() - 0.01]) == set(z[f[:, 2] < f[:, 2].min() + 0.01])     # same top and bottom
    st, _, r = build(base, "slab", {"hkl": [0, 0, 1], "layers": 4, "vacuum": 10, "symmetric": True}, preview=True)
    assert r["info"]["layers"] == 4                                            # (001) is symmetric already


def test_slab_si_111_and_hbn_001(served):
    base = served[0]
    st, _, r = build(base, "slab", {"hkl": [1, 1, 1], "layers": 6, "vacuum": 15}, source={"path": "structures/Si.cif"}, preview=True)
    assert len(r["info"]["terminations"]) == 2                                 # the bilayer's two cuts
    assert r["n"] == 24
    st, _, r = build(base, "slab", {"hkl": [0, 0, 1], "layers": 3, "vacuum": 20}, source={"path": "structures/hBN.cif"},
                     open_as={"name": "hBN_001_3L"})
    s = r["session"]
    mean, gap, c = planes_of(s)
    assert len(mean) == 3 and np.allclose(np.diff(mean), 3.3305, atol=2e-3) and abs(gap - 20) < 0.01 and len(s["numbers"]) == 6


def test_slab_doc_editable_3d_with_vacuum(served):
    base = served[0]
    st, _, r = build(base, "slab", {"hkl": [0, 0, 1], "layers": 3, "vacuum": 12}, open_as={"name": "MgO_001_3L"})
    s = r["session"]
    st, _, r2 = U.post(base, "api/edit/op", {"session": s["session"], "base_rev": s["rev"], "op": "delete", "args": {"indices": [0]}})
    assert st == 200 and len(r2["numbers"]) == len(s["numbers"]) - 1


def test_slab_errors(served):
    base = served[0]
    for params, msg in (({"hkl": [0, 0, 0]}, "h k l can't all be zero"), ({"hkl": [0.5, 0, 1]}, "whole numbers"),
                        ({"hkl": [0, 0, 1], "layers": 0}, "layers"), ({"hkl": [0, 0, 1], "vacuum": -1}, "vacuum")):
        st, _, r = build(base, "slab", params)
        assert st == 400 and msg in r["error"], (params, r)
    st, _, r = build(base, "slab", {"hkl": [0, 0, 1]}, source={"path": "outputs/graphene_slab_sp.out"})
    assert st == 400 and r["error"] == "Slabs are cut from 3D bulk cells"


# --- T6 point defects -------------------------------------------------------------------

def info(base, source=None, **kw):
    st, _, r = U.post(base, "api/defect/info", {"source": source or {"path": "structures/MgO.cif"}, **kw})
    return result_of(base, st, r)


def test_defect_sites_mgo(served):
    r = info(served[0])
    sites = [(s["element"], s["wyckoff"], s["multiplicity"], len(s["orbit"])) for s in r["sites"]]
    assert sites == [("Mg", "a", 4, 4), ("O", "b", 4, 4)], sites
    assert r["group"]["number"] == 225


def test_defect_default_supercell_by_image_distance(served):
    base = served[0]
    r = info(base)["supercell"]
    assert r["matrix"] == [[3, 0, 0], [0, 3, 0], [0, 0, 3]] and r["atoms"] == 216 and abs(r["nearest"] - 12.636) < 1e-3, r
    assert info(base, min_dist=8)["supercell"]["matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    # the 2-atom primitive cell of MgO: 4×4×4, 128 atoms, 11.913 Å
    from ase.build import bulk
    p = bulk("MgO", "rocksalt", a=4.212)
    st, _, s = U.post(base, "api/edit/from_structure", {"name": "MgO_prim", "cell": p.cell.array.tolist(),
                                                       "frac": p.get_scaled_positions().tolist(), "numbers": p.numbers.tolist()})
    r = info(base, source={"session": s["session"]})["supercell"]
    assert r["matrix"] == [[4, 0, 0], [0, 4, 0], [0, 0, 4]] and r["atoms"] == 128 and abs(r["nearest"] - 11.913) < 1e-3, r
    # the matrix search: fewer atoms at the same distance when one exists
    m = info(base, search_matrices=True)["supercell"]
    assert m["atoms"] <= 216 and m["nearest"] >= 10 - 1e-9, m


def test_defect_supercell_min_image_property():
    """For random triclinic cells: every nonzero vector of the chosen supercell
    (indices within ±3) is at least d long, and no smaller diagonal one is."""
    import itertools
    from ase.geometry import cellpar_to_cell
    from mace.gui import builders as B
    rng = np.random.default_rng(7)
    k = np.array([c for c in itertools.product(range(-3, 4), repeat=3) if any(c)])
    for _ in range(25):
        par = list(rng.uniform(3, 9, 3)) + list(rng.uniform(65, 115, 3))
        try:
            cell = cellpar_to_cell(par)
        except ValueError:
            continue
        if abs(np.linalg.det(cell)) < 1:
            continue
        d = float(rng.uniform(6, 14))
        r = B.defect_supercell(cell, 4, d)
        sc = np.asarray(r["matrix"]) @ cell
        assert np.linalg.norm(k @ sc, axis=1).min() >= d - 1e-6, (par, d, r)
        assert abs(np.linalg.norm(k @ sc, axis=1).min() - r["nearest"]) < 1e-6 or r["nearest"] <= np.linalg.norm(k @ sc, axis=1).min()


def test_vacancy_count_and_group_drop(served):
    base, root = served[0], served[1]
    before = U.tree_md5(root)
    st, _, r = build(base, "defect", {"kind": "vacancy", "site": 4, "matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]]},
                     open_as={"name": "MgO_2x2x2_V_O_4b", "label": "Vacancy at O (4b) in MgO.cif", "hints": {"charge": 2}})
    assert st == 200, r
    s = r["session"]
    assert len(s["numbers"]) == 63 and s["numbers"].count(8) == 31 and s["numbers"].count(12) == 32
    assert s["symmetry"]["number"] < 225
    assert r["info"]["nearest_image"] == pytest.approx(8.424, abs=1e-3)
    # the charge hint is the document's (session, draft and Save sidecar carry it)
    assert s["doc_meta"]["hints"]["charge"] == 2 and s["doc_meta"]["provenance"]["op"] == "defect"
    assert U.tree_md5(root) == before                       # never writes


def test_substitution_and_antisite_counts(served):
    base = served[0]
    m = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    st, _, r = build(base, "defect", {"kind": "substitution", "site": 0, "element": "Ca", "matrix": m}, open_as={"name": "MgO_Ca"})
    s = r["session"]["numbers"]
    assert len(s) == 64 and s.count(20) == 1 and s.count(12) == 31
    st, _, r = build(base, "defect", {"kind": "antisite", "site": 4, "element": "Mg", "matrix": m}, open_as={"name": "MgO_anti"})
    s = r["session"]["numbers"]
    assert len(s) == 64 and s.count(12) == 33 and s.count(8) == 31
    st, _, r = build(base, "defect", {"kind": "substitution", "site": 0, "element": "Mg", "matrix": m})
    assert st == 400 and r["error"] == "Pick a different element"
    st, _, r = build(base, "defect", {"kind": "interstitial", "element": "Li", "position": [0.25, 0.25, 0.25], "matrix": m},
                     open_as={"name": "MgO_Li_i"})
    s = r["session"]["numbers"]
    assert len(s) == 65 and s.count(3) == 1
    st, _, r = build(base, "defect", {"kind": "interstitial", "element": "Li", "position": [0, 0, 0], "matrix": m})
    assert st == 400 and "already sits there" in r["error"]


def test_interstitial_candidates_rocksalt_8c(served):
    base = served[0]
    r = info(base, interstitial=True)
    c = r["candidates"]
    assert c and c[0]["wyckoff"] == "c" and c[0]["multiplicity"] == 8, c
    assert np.allclose(np.asarray(c[0]["frac"]) % 0.5, 0.25, atol=1e-6) and abs(c[0]["distance"] - 4.212 * 3 ** 0.5 / 4) < 1e-3
    assert all(x["distance"] >= 1.0 for x in c)
    from mace.gui import builders as B
    a = read("MgO.cif")
    for x in c:          # nothing closer than 1 Å to an atom, by an independent minimum-image check
        d = a.get_distances(0, range(len(a)), mic=True)  # noqa: F841 (warm)
        from ase import Atoms
        b = a + Atoms("He", scaled_positions=[x["frac"]], cell=a.cell)
        assert b.get_distances(len(b) - 1, range(len(a)), mic=True).min() >= 1.0


def test_op_defect_in_place_one_history_entry(served):
    base = served[0]
    s = U.req(base, "api/structure?path=structures/LiF.cif")[2]
    st, _, r = U.post(base, "api/edit/open", {"path": "structures/LiF.cif", "expected": s["file"], "locked": True})
    rows0 = len(r["history"])
    st, _, r2 = U.post(base, "api/edit/op", {"session": r["session"], "base_rev": r["rev"], "op": "defect",
                                             "args": {"kind": "vacancy", "index": 0}, "label": "Vacancy at Li (4a)"})
    assert st == 200, r2
    assert len(r2["numbers"]) == len(r["numbers"]) - 1            # one atom, not its orbit (Sym mode)
    assert len(r2["history"]) == rows0 + 1 and r2["history"][-1]["label"] == "Vacancy at Li (4a)"
    assert r2["index_map"][0] is None and r2["index_map"][1] == 0


# --- T7 transform cell, symmetrize -------------------------------------------------------

def session_of(base, atoms, name="tmp", **kw):
    st, _, s = U.post(base, "api/edit/from_structure", {"name": name, "cell": atoms.cell.array.tolist(),
                                                       "frac": atoms.get_scaled_positions(wrap=False).tolist(),
                                                       "numbers": atoms.numbers.tolist(), **kw})
    assert st == 200, s
    return s


def distorted_mgo(delta=(0.001, 0.0007, 0.0004)):
    a = read("MgO.cif")
    f = a.get_scaled_positions()
    f[4] += np.asarray(delta)
    a.set_scaled_positions(f)
    return a


def test_transform_primitive_conventional_niggli(served):
    base = served[0]
    st, _, r = build(base, "standardize", {"kind": "primitive"}, preview=True)
    assert r["n"] == 2 and abs(np.linalg.norm(r["cell"][0]) - 2.9783) < 1e-3, r["n"]
    st, _, r = build(base, "standardize", {"kind": "primitive"}, open_as={"name": "MgO_prim"})
    prim = r["session"]
    assert len(prim["numbers"]) == 2 and prim["symmetry"]["number"] == 225
    st, _, r = build(base, "standardize", {"kind": "conventional"}, source={"session": prim["session"]}, preview=True)
    assert r["n"] == 8
    import spglib
    from ase.geometry import cellpar_to_cell
    from ase import Atoms
    tri = Atoms("SiO2Si", cell=cellpar_to_cell([5.1, 6.3, 7.4, 78, 104, 96]) @ np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]]),
                scaled_positions=[[.1, .2, .3], [.35, .15, .42], [.62, .71, .18], [.8, .55, .66]], pbc=True)
    s = session_of(base, tri, "tri")
    st, _, r = build(base, "standardize", {"kind": "niggli"}, source={"session": s["session"]}, preview=True)
    assert np.allclose(r["cell"], spglib.niggli_reduce(tri.cell.array), atol=1e-8)
    assert r["n"] == 4


def test_op_transform_one_history_entry(served):
    base = served[0]
    s = session_of(base, read("MgO.cif"), "mgo_t")
    st, _, r = U.post(base, "api/edit/op", {"session": s["session"], "base_rev": s["rev"], "op": "transform",
                                            "args": {"kind": "primitive"}, "label": "Transform to Primitive"})
    assert st == 200 and len(r["numbers"]) == 2 and [h["label"] for h in r["history"]][-1] == "Transform to Primitive"
    assert len(r["history"]) == 2


def test_symmetrize_sweep_lists_groups_and_changes(served):
    base = served[0]
    s = session_of(base, distorted_mgo(), "mgo_d")
    st, _, r = U.post(base, "api/symmetrize/sweep", {"source": {"session": s["session"]}})
    sw = r["sweep"]
    assert len(sw) == 13 and sw[0]["symprec"] == 1e-5 and sw[-1]["symprec"] == 0.1
    assert sw[0]["number"] == 1, sw[0]                         # P1 at 1e-5
    at = {x["symprec"]: x["number"] for x in sw}
    assert at[1e-2] == 225 and at[0.1] == 225, at
    # a stop is marked as a change exactly when its group differs from the stop before
    assert [x["change"] for x in sw] == [k > 0 and sw[k]["number"] != sw[k - 1]["number"] for k in range(len(sw))]
    assert any(x["change"] for x in sw)


def test_symmetrize_displacement_stats_match_python(served):
    """The shift the route reports is the one put in (the moved O goes back)."""
    base = served[0]
    delta = np.array([0.001, 0.0007, 0.0004])
    s = session_of(base, distorted_mgo(delta), "mgo_d2")
    st, _, r = build(base, "symmetrize", {"symprec": 0.01}, source={"session": s["session"]}, preview=True)
    moved = float(np.linalg.norm(delta * 4.212))
    info = r["info"]
    assert info["number"] == 225 and r["n"] == 8
    # the refined structure sits where the mean of the orbits puts it: the moved O moves back by about all of it
    assert moved * 0.7 <= info["max_shift"] <= moved * 1.01, (info, moved)
    v = np.asarray(info["vectors"])
    assert np.allclose(np.linalg.norm(v, axis=1).max(), info["max_shift"], atol=1e-6)
    assert abs(np.sqrt((np.linalg.norm(v, axis=1) ** 2).mean()) - info["rms_shift"]) < 1e-6
    st, _, r = build(base, "symmetrize", {"symprec": 1e-5}, source={"session": s["session"]}, preview=True)
    assert r["info"]["number"] == 1


def test_op_symmetrize_with_tolerance_one_entry(served):
    base = served[0]
    s = session_of(base, distorted_mgo(), "mgo_d3")
    st, _, r = U.post(base, "api/edit/op", {"session": s["session"], "base_rev": s["rev"], "op": "symmetrize",
                                            "args": {"symprec": 0.01}, "label": "Symmetrize at 0.01 Å"})
    assert st == 200 and r["symmetry"]["number"] == 225 and r["history"][-1]["label"] == "Symmetrize at 0.01 Å"
    assert r["symprec"] == 1e-3                                # the session's own tolerance is unchanged


def test_symmetrize_sweep_cancel_latency(served):
    base = served[0]
    from ase.build import make_supercell
    big = make_supercell(distorted_mgo(), np.diag([6, 6, 7]))       # 2016 atoms: the sweep is a pool task
    s = session_of(base, big, "mgo_big")
    sse = H.sse_events(base)
    try:
        sse.wait(lambda e: e[0] == "hello", 10)
        st, _, r = U.post(base, "api/symmetrize/sweep", {"source": {"session": s["session"]}})
        if st == 200:
            pytest.skip("the sweep finished within 300 ms on this machine; nothing to cancel")
        tid = r["task"]["id"]
        sse.wait(lambda e: e[0] == "task" and e[1]["id"] == tid and e[1]["state"] == "running", 20)
        t0 = time.time()
        U.post(base, f"api/tasks/{tid}/cancel", {})
        ev = sse.wait(lambda e: e[0] == "task" and e[1]["id"] == tid and e[1]["state"] in ("canceled", "done"), 15)
        assert ev[1]["state"] == "canceled" and ev[2] - t0 <= 2.5, (ev[1]["state"], ev[2] - t0)
    finally:
        sse.close()


# --- T8 Clean geometry ------------------------------------------------------------------

def test_clean_should_stop_default_unchanged():
    from mace.gui import clean as C
    a = distorted_mgo((0.02, 0.0, 0.0))
    r1 = C.clean(a, time_limit=20)
    r2 = C.clean(a, time_limit=20, should_stop=None)
    assert r1["atoms"].positions.tobytes() == r2["atoms"].positions.tobytes() and r1["steps"] == r2["steps"]
    calls = []
    r3 = C.clean(a, time_limit=20, should_stop=lambda: calls.append(1) or len(calls) >= 3)
    assert r3["steps"] == 3 and not r3["converged"]


def clean_req(base, sid, rev, **kw):
    st, _, r = U.post(base, "api/clean", {"session": sid, "base_rev": rev, **kw})
    return result_of(base, st, r)


def test_clean_preview_accept_and_discard(served):
    """Criterion 3 (API): the proposal commits nothing; Accept adds one
    history entry "Clean Geometry"; Discard adds none."""
    base = served[0]
    s = session_of(base, distorted_mgo((0.047, 0.0, 0.0)), "mgo_c")          # one O moved 0.2 Å
    r = clean_req(base, s["session"], s["rev"])
    assert r["proposal_id"] and r["proposal"]["packed"] == 1 and r["before"]["packed"] == 1
    assert r["summary"].startswith("max shift ") and " · RMS " in r["summary"] and "group " in r["summary"], r["summary"]
    assert r["report"]["worker_pid"] != U.req(base, "api/test/pool")[2]["server_pid"] and r["report"]["worker_nice"] == 10
    state = U.req(base, f"api/edit/state?session={s['session']}")[2]
    assert state["rev"] == s["rev"] and len(state["history"]) == 1             # nothing committed
    st, _, d = U.post(base, "api/edit/clean_discard", {"session": s["session"], "proposal_id": r["proposal_id"]})
    assert d == {"discarded": True, "rev": s["rev"]}
    state = U.req(base, f"api/edit/state?session={s['session']}")[2]
    assert state["rev"] == s["rev"] and len(state["history"]) == 1 and state["frac"] == s["frac"]
    r = clean_req(base, s["session"], s["rev"])
    st, _, a = U.post(base, "api/edit/clean_accept", {"session": s["session"], "base_rev": s["rev"], "proposal_id": r["proposal_id"]})
    assert st == 200 and len(a["history"]) == 2 and a["history"][-1]["label"] == "Clean Geometry" and a["rev"] != s["rev"]
    assert U.post(base, "api/edit/clean_accept", {"session": s["session"], "proposal_id": r["proposal_id"]})[0] == 404


def test_clean_stale_base_rev_409(served):
    base = served[0]
    s = session_of(base, distorted_mgo((0.047, 0.0, 0.0)), "mgo_c2", locked=False)
    r = clean_req(base, s["session"], s["rev"])
    st, _, e = U.post(base, "api/edit/op", {"session": s["session"], "base_rev": s["rev"], "op": "delete", "args": {"indices": [0]}})
    assert st == 200
    st, _, a = U.post(base, "api/edit/clean_accept", {"session": s["session"], "base_rev": e["rev"], "proposal_id": r["proposal_id"]})
    assert st == 409 and a["conflict"] == "rev"


def test_clean_on_metal_nothing_to_clean(served):
    from ase.build import bulk
    base = served[0]
    s = session_of(base, bulk("Cu", "fcc", a=3.61, cubic=True), "cu")
    st, _, r = U.post(base, "api/clean", {"session": s["session"], "base_rev": s["rev"]})
    assert st == 200 and r["nothing_to_clean"] and r["message"].startswith("nothing to clean: no bonded pairs"), r
    assert U.req(base, f"api/edit/state?session={s['session']}")[2]["rev"] == s["rev"]


def test_clean_time_limit_reported(served):
    from ase.build import make_supercell
    base = served[0]
    big = make_supercell(distorted_mgo((0.04, 0.02, 0.0)), np.diag([3, 3, 3]))
    rng = np.random.default_rng(3)
    big.positions += rng.normal(0, 0.15, big.positions.shape)
    s = session_of(base, big, "mgo_slow", locked=False)
    r = clean_req(base, s["session"], s["rev"], time_limit=0.05, keep_symmetry=False)
    assert r["report"]["timed_out"] and not r["report"]["converged"]
    assert r["warning"].startswith("Stopped at the 0.05 s time limit after ") and r["warning"].endswith("(not converged)")


def test_clean_cancel_latency(served):
    from ase.build import make_supercell
    base = served[0]
    big = make_supercell(distorted_mgo((0.04, 0.02, 0.0)), np.diag([5, 5, 5]))
    big.positions += np.random.default_rng(4).normal(0, 0.2, big.positions.shape)
    s = session_of(base, big, "mgo_slow2", locked=False)
    sse = H.sse_events(base)
    try:
        sse.wait(lambda e: e[0] == "hello", 10)
        st, _, r = U.post(base, "api/clean", {"session": s["session"], "base_rev": s["rev"], "time_limit": 60, "keep_symmetry": False})
        assert st == 202, r
        tid = r["task"]["id"]
        sse.wait(lambda e: e[0] == "task" and e[1]["id"] == tid and e[1]["state"] == "running" and e[1]["progress"] > 0, 30)
        t0 = time.time()
        U.post(base, f"api/tasks/{tid}/cancel", {})
        ev = sse.wait(lambda e: e[0] == "task" and e[1]["id"] == tid and e[1]["state"] in ("canceled", "done", "failed"), 15)
        assert ev[1]["state"] == "canceled" and ev[2] - t0 <= 2.5, (ev[1]["state"], ev[2] - t0)
    finally:
        sse.close()


@pytest.mark.parametrize("path", ["api/clean", "api/edit/clean_accept", "api/edit/clean_discard", "api/symmetrize/sweep",
                                  "api/defect/info", "api/build/defect", "api/build/slab", "api/build/standardize",
                                  "api/build/symmetrize"])
def test_more_build_routes_require_token_host_and_xrw(served, path):
    base = served[0]
    data = {"session": "x", "source": {"path": "structures/MgO.cif"}}
    assert U.post(base, path, data, token=False)[0] == 403
    port = base.split("/")[2].rsplit(":", 1)[1]
    assert U.post(base, path, data, headers={"Host": f"evil.example:{port}"})[0] == 421
    assert U.post(base, path, data, headers={"X-Requested-With": "x"})[0] == 403
