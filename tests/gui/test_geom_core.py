"""The pure geometry module (static/js/geom/core.js) and its worker, run in
headless Firefox (no node toolchain): bonds against a numpy oracle, the
camera matrix against 3Dmol, measurements against ase, symmetry orbits and
site projections against the server's editor, LOD limits, view directions
and the coordinate-field expressions."""
import json
import math
import time

import numpy as np
import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
pytest.importorskip("spglib")

from ase.build import bulk  # noqa: E402


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("geom")
    root, xdg = H.examples_copy(tmp), tmp / "xdg"
    gen = root / "generated"
    gen.mkdir()
    import ase.io
    mgo = bulk("MgO", "rocksalt", a=4.212, cubic=True)
    ase.io.write(gen / "mgo_5x5x5.cif", mgo.repeat((5, 5, 5)))
    ase.io.write(gen / "mgo_10x10x10.cif", mgo.repeat((10, 10, 10)))
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1")
    drv = H.firefox(1280, 800)
    try:
        H.load(drv, base)
        yield drv, base, root
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def core(drv, body, *args, timeout=60):
    """Run `body` (an async function body with `G` = core.js, `args`) in the page."""
    drv.set_script_timeout(timeout)
    r = drv.execute_async_script("""
        const done = arguments[arguments.length - 1], args = arguments[0];
        const v = document.querySelector('script[src*="/js/shell.js"]').getAttribute('src').split('/')[1];
        import(`./static/${v}/js/geom/core.js`).then(async (G) => {
          const f = new Function('G', 'args', 'return (async () => {' + arguments[1] + '})()');
          return f(G, args);
        }).then((r) => done(JSON.stringify({ok: r === undefined ? null : r})), (e) => done(JSON.stringify({err: String(e && e.stack || e)})));
    """, list(args), body)
    out = json.loads(r)
    assert "err" not in out, out["err"]
    return out["ok"]


def structure(drv, path):
    return drv.execute_async_script(
        "const d = arguments[1]; window.MACE.api('structure?path=' + encodeURIComponent(arguments[0])).then(d)", path)


def pairs_oracle(cart, radii, metal, tol=1.15, metal_metal=False):
    """Every pair i<j of the given points with 0.1 < d < tol (ri + rj): numpy, chunked."""
    cart, radii, metal = np.asarray(cart), np.asarray(radii), np.asarray(metal, bool)
    out = set()
    for s in range(0, len(cart), 400):
        d = np.linalg.norm(cart[s:s + 400, None, :] - cart[None, :, :], axis=2)
        cut = tol * (radii[s:s + 400, None] + radii[None, :])
        ok = (d > 0.1) & (d < cut)
        if not metal_metal:
            ok &= ~(metal[s:s + 400, None] & metal[None, :])
        for i, j in zip(*np.nonzero(ok)):
            if s + i < j:
                out.add((int(s + i), int(j)))
    return out


@pytest.mark.parametrize("path", ["structures/MgO.cif", "structures/Si.cif", "structures/hBN.cif",
                                  "structures/NaCl.cif", "structures/LiF.cif", "structures/diamond.cif",
                                  "outputs/Ag2Br3_opt_unfinished.out", "generated/mgo_5x5x5.cif",
                                  "generated/mgo_10x10x10.cif"])
def test_cell_list_bonds_match_reference(page, path):
    drv = page[0]
    s = structure(drv, path)
    # the drawn atoms, as the viewer lays them out (boundary images on)
    shown = drv.execute_script("""const s = arguments[0];
        return window.MACE.displayAtoms(s, {boundary: true, supercell: false}).map((a) => [a.i, ...a.cart])""", s)
    cart = [a[1:] for a in shown]
    idx = [a[0] for a in shown]
    radii = [s["radii"][i] for i in idx]
    metal = [s["metal"][i] for i in idx]
    got = core(drv, """const p = G.cellListBonds(Float64Array.from(args[0].flat()), Float32Array.from(args[1]),
                                                 Uint8Array.from(args[2]), 1.15, false);
                       return Array.from(p)""", cart, radii, [1 if m else 0 for m in metal])
    got = {(got[k], got[k + 1]) for k in range(0, len(got), 2)}
    want = pairs_oracle(cart, np.float32(radii).astype(float), metal)
    assert got == want, (path, len(got), len(want))
    if len(cart) <= 4000:    # the old O(N²) bondsFor finds the same bonds
        old = drv.execute_script("""const s = arguments[0], atoms = window.MACE.displayAtoms(s, {boundary: true, supercell: false});
            return window.MACE.bondsFor(atoms, s, 1.15).reduce((t, b) => t + b.length, 0) / 2""", s)
        assert old == len(got), (path, old, len(got))


@pytest.mark.parametrize("path", ["structures/MgO.cif", "structures/Si.cif", "outputs/Ag2Br3_opt_unfinished.out",
                                  "generated/mgo_5x5x5.cif"])
def test_periodic_bonds_match_ase_neighbor_list(page, path):
    from ase import Atoms
    from ase.neighborlist import neighbor_list
    drv = page[0]
    s = structure(drv, path)
    got = core(drv, """const s = args[0];
        const b = G.periodicBonds(s.cell, Float64Array.from(s.frac.flat()), Float32Array.from(s.radii),
                                  Uint8Array.from(s.metal.map(Number)), 1.15, false);
        return {i: Array.from(b.i), j: Array.from(b.j), img: Array.from(b.img)}""", s)
    have = set()
    for k in range(len(got["i"])):
        i, j, img = got["i"][k], got["j"][k], tuple(got["img"][3 * k:3 * k + 3])
        have.add((i, j, img))
        have.add((j, i, tuple(-x for x in img)))
    atoms = Atoms(numbers=[n % 100 for n in s["numbers"]], cell=s["cell"], scaled_positions=s["frac"], pbc=True)
    radii = np.float32(s["radii"]).astype(float)
    ii, jj, SS, dd = neighbor_list("ijSd", atoms, 1.15 * 2 * radii.max())
    metal = np.asarray(s["metal"], bool)
    want = set()
    for i, j, S, d in zip(ii, jj, SS, dd):     # D = positions[j] - positions[i] + S.cell, as given
        if d <= 0.1 or d >= 1.15 * (radii[i] + radii[j]) or (metal[i] and metal[j]):
            continue
        want.add((int(i), int(j), tuple(int(x) for x in S)))
    assert have == want, (path, len(have), len(want))


def test_dlt_matches_model_to_screen(page):
    drv = page[0]
    drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then(() => d())", "structures/MgO.cif")
    time.sleep(0.8)
    for proj in ["orthographic", "perspective"]:
        drv.execute_script("window.__mace.settings.set('projection', arguments[0])", proj)
        time.sleep(0.3)
        err = core(drv, """const v = window.MACE.state.viewers['viewer-main'], r = v.getCanvas().getBoundingClientRect();
            const scr = (x, y, z) => { const p = v.modelToScreen({x, y, z}); return [p.x - r.left, p.y - r.top]; };
            const cal = [];
            for (const x of [-2, 6]) for (const y of [-2, 6]) for (const z of [-2, 6]) cal.push([x, y, z, ...scr(x, y, z)]);
            cal.push([2.1, 1, 3, ...scr(2.1, 1, 3)]);
            const P = G.dltCalibrate(cal);
            let worst = 0;
            for (let k = 0; k < 1000; k++) {
              const p = [Math.random() * 8 - 2, Math.random() * 8 - 2, Math.random() * 8 - 2];
              const [u, w] = G.project(P, ...p), [su, sv] = scr(...p);
              worst = Math.max(worst, Math.hypot(u - su, w - sv));
            }
            return worst""")
        assert err <= 0.5, (proj, err)
    drv.execute_script("window.__mace.settings.set('projection', 'orthographic'); window.__mace.closeAll()")


def test_point_in_polygon_and_box(page):
    r = core(page[0], """
        const sq = [[0, 0], [10, 0], [10, 10], [0, 10]], L = [[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]];
        const uv = Float32Array.from([1, 1, 5, 5, 9, 9, 20, 20, 7, 7]);
        return {a: G.pointInPolygon(5, 5, sq), b: G.pointInPolygon(11, 5, sq), c: G.pointInPolygon(7, 7, L),
                d: G.pointInPolygon(2, 7, L), box: Array.from(G.boxSelect(uv, {x0: 10, y0: 10, x1: 0, y1: 0})),
                lasso: Array.from(G.lassoSelect(uv, L))}""")
    assert r == {"a": True, "b": False, "c": False, "d": True, "box": [0, 1, 2, 4], "lasso": [0]}


def test_measure_math_against_ase(page):
    from ase import Atoms
    rng = np.random.default_rng(7)
    atoms = Atoms("MgOSiC", cell=[[4.0, 0, 0], [0.9, 5.0, 0], [0.4, -0.7, 6.0]],
                  scaled_positions=rng.random((4, 3)), pbc=True)
    frac = atoms.get_scaled_positions(wrap=False).tolist()
    r = core(page[0], """const [cell, frac] = args;
        const img = (a, b) => G.minImage(cell, frac[a], frac[b]).image;
        // each atom at its image nearest the one before it, as ase's mic does per vector
        const d = G.measure(cell, frac, [{i: 0, image: [0, 0, 0]}, {i: 1, image: img(0, 1)}]);
        const i10 = img(1, 0), i12 = img(1, 2);
        const ang = G.measure(cell, frac, [{i: 0, image: i10}, {i: 1, image: [0, 0, 0]}, {i: 2, image: i12}]);
        const i23 = img(2, 3).map((x, k) => x + i12[k]);
        const dih = G.measure(cell, frac, [{i: 0, image: i10}, {i: 1, image: [0, 0, 0]}, {i: 2, image: i12}, {i: 3, image: i23}]);
        return [d, ang, dih]""", atoms.cell.array.tolist(), frac)
    assert r[0] == pytest.approx(atoms.get_distance(0, 1, mic=True), abs=1e-9)
    assert r[1] == pytest.approx(atoms.get_angle(0, 1, 2, mic=True), abs=1e-7)
    assert abs(((r[2] - atoms.get_dihedral(0, 1, 2, 3, mic=True)) + 180) % 360 - 180) < 1e-6
    mgo = bulk("MgO", "rocksalt", a=4.212, cubic=True)
    o = mgo.get_chemical_symbols().index("O")
    d = core(page[0], "return G.minImage(args[0], args[1][0], args[1][args[2]]).length",
             mgo.cell.array.tolist(), mgo.get_scaled_positions().tolist(), o)
    assert d == pytest.approx(mgo.get_distance(0, o, mic=True)) and round(d, 3) == 2.106


def test_lod_level_boundaries(page):
    r = core(page[0], """const out = {};
        for (const q of ['low', 'medium', 'high', 'auto'])
          out[q] = [1999, 2000, 2001, 19999, 20000, 20001, 199999, 200000, 200001, 999, 1000, 1001, 3999, 4000, 4001]
            .map((n) => G.lodLevel(n, q));
        return out""")
    assert r["medium"][:9] == [1, 1, 2, 2, 2, 3, 3, 3, 4] == r["auto"][:9]
    assert r["low"][9:12] == [1, 1, 2] and r["low"][:3] == [2, 2, 2]
    assert r["high"][12:15] == [1, 1, 2] and r["high"][:3] == [1, 1, 1]


def test_along_directions_triclinic(page):
    cell = [[4.0, 0.0, 0.0], [5 * math.cos(math.radians(105)), 5 * math.sin(math.radians(105)), 0.0], [0.6, -0.9, 5.8]]
    r = core(page[0], """const C = args[0], out = {};
        for (const w of ['a', 'b', 'c', 'a*', 'b*', 'c*']) {
          const d = G.alongDirection(C, w), up = G.upVector(C, d), q = G.viewQuaternion(d, up);
          out[w] = {d, up, dz: G.qrotate(q, d), upz: G.qrotate(q, up),
                    axes: C.map((v) => G.qrotate(q, G.unit3(v)))};
        }
        out.uvw = G.alongDirection(C, [1, 1, 0]);
        return out""", cell)
    C = np.array(cell)
    for k, w in enumerate("abc"):
        assert np.allclose(r[w]["d"], C[k] / np.linalg.norm(C[k]))
        assert np.allclose(r[w]["dz"], [0, 0, 1], atol=1e-9) and np.allclose(r[w]["upz"], [0, 1, 0], atol=1e-9)
        # a* is perpendicular to b and c: those two lie in the screen plane
        others = [m for m in range(3) if m != k]
        assert all(abs(r[w + "*"]["axes"][m][2]) < 1e-9 for m in others)
        assert r[w + "*"]["axes"][k][2] > 0
        assert abs(np.dot(r[w]["up"], r[w]["d"])) < 1e-12
    assert np.allclose(r["uvw"], (C[0] + C[1]) / np.linalg.norm(C[0] + C[1]))


def test_eval_expr(page):
    r = core(page[0], """return ['1/3', '0.5+0.01', '-x', '-(1/4)', '2*(1+0.5)', '1/0', '', '.5', '3e-2', '1+', 'abc']
        .map((t) => G.evalExpr(t))""")
    assert r[0]["ok"] and r[0]["value"] == pytest.approx(1 / 3, abs=1e-15)
    assert r[1]["value"] == pytest.approx(0.51)
    assert not r[2]["ok"] and "not a number" in r[2]["error"]
    assert r[3]["value"] == -0.25 and r[4]["value"] == 3.0
    assert not r[5]["ok"] and "zero" in r[5]["error"]
    assert not r[6]["ok"] and r[7]["value"] == 0.5 and r[8]["value"] == 0.03
    assert not r[9]["ok"] and not r[10]["ok"]
    r = core(page[0], """return [G.parseIndexList('1-4, 9, 12-', 14), G.parseIndexList('0-2', 8), G.parseIndexList('x', 8),
        [[0.1, 0.6, 0.7], [0.3, 0.2, 1.6], [0.2, 0.2, -0.4]].map((f) => G.parseFracRange('z > 0.5').test(f)),
        G.parseFracRange('x < 0.25 and y >= 0.2').test([0.2, 0.2, 0]), G.parseFracRange('q > 1').ok]""")
    assert r[0] == {"ok": True, "indices": [0, 1, 2, 3, 8, 11, 12, 13]}
    assert not r[1]["ok"] and not r[2]["ok"]
    assert r[3] == [True, True, True] and r[4] is True and r[5] is False


def test_orbit_apply_matches_spglib(page):
    from mace.gui import editor as E
    from mace.gui import structures as S
    s = E.EditSession(E.atoms_from_payload(S.read_structure(H.EXAMPLES / "structures" / "MgO.cif")), "MgO", locked=True)
    ops = {"rotations": s.ops[0].tolist(), "translations": s.ops[1].tolist()}
    cell = s.atoms.cell.array.tolist()
    for f in ([0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.1, 0.2, 0.3]):
        want = s._images(f)
        got = core(page[0], "return G.orbitApply(args[0], args[1], args[2], 1e-3)", ops, f, cell)
        assert len(got) == len(want), f
        for p in want:
            assert any(np.allclose((np.array(q) - p + 0.5) % 1 - 0.5, 0, atol=1e-9) for q in got)


def test_project_onto_site_matches_server(page):
    from ase.spacegroup import crystal
    from mace.gui import editor as E
    s = E.EditSession(crystal(["Ti", "O"], [(0, 0, 0), (0.3, 0.3, 0)], spacegroup=136,
                              cellpar=[4.59, 4.59, 2.96, 90, 90, 90]), "rutile", locked=True)
    o = int(np.where(s.atoms.numbers == 8)[0][0])
    f0 = s.atoms.get_scaled_positions()[o]
    ops = {"rotations": s.ops[0].tolist(), "translations": s.ops[1].tolist()}
    for df in ([0.02, 0.0, 0.01], [0.0, 0.03, 0.0], [0.01, -0.01, 0.02]):
        want = s._project(f0, np.array(df))
        got = core(page[0], """const [ops, f0, df, cell] = args;
            return {p: G.projectOntoSite(ops, f0, df, cell, 2e-3), leaves: G.leavesSite(ops, f0, df, cell, 2e-3)}""",
                   ops, f0.tolist(), df, s.atoms.cell.array.tolist())
        assert np.allclose(got["p"], want, atol=1e-12)
        assert got["leaves"] == (not np.allclose(want, df))


def test_worker_supersedes_older_job(page):
    drv = page[0]
    drv.set_script_timeout(30)
    r = drv.execute_async_script("""const done = arguments[0];
        const v = document.querySelector('script[src*="/js/shell.js"]').getAttribute('src').split('/')[1];
        import(`./static/${v}/js/geom/client.js`).then(async (W) => {
          const n = 20000, cart = new Float64Array(3 * n);
          for (let i = 0; i < 3 * n; i++) cart[i] = Math.random() * 50;
          const P = Float64Array.from([10, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0]);
          const jobs = [0, 1, 2].map((k) => W.job('boxSelect', {P, cart: cart.slice(), rect: {x0: 0, y0: 0, x1: 100 * (k + 1), y1: 100 * (k + 1)}}));
          const out = await Promise.all(jobs);
          const bonds = await W.job('bonds', {cart: Float64Array.from([0, 0, 0, 1, 0, 0, 5, 5, 5]), radii: Float32Array.from([0.7, 0.7, 0.7]),
                                              metal: new Uint8Array(3), tol: 1.15, metalMetal: false});
          done({sup: out.map((o) => !!o.superseded), last: out[2].hits ? out[2].hits.length : -1, bonds: Array.from(bonds.pairs),
                stats: {...W.stats}});
        }, (e) => done({err: String(e)}));""")
    assert "err" not in r, r
    assert r["sup"] == [True, True, False] and r["last"] > 0
    assert r["bonds"] == [0, 1] and r["stats"]["started"] == 1
