"""Fractional display ranges, bonded atoms beyond the range and the
dimensionality of bonded networks (polish A4), in core.js run in headless
Firefox, against numpy / ase oracles on real structures from test/.

- displayList with {lo, hi}: the set of (atom, cell shift) equals a numpy
  enumeration of VESTA's boundary rule; the old [n1, n2, n3] form is the
  same set as {lo: 0, hi: n} (and its code path is unchanged).
- components(): each atom's dimension equals ase's rank determination
  (ase.geometry.dimensionality RDA) over ase's own neighbor list at the
  same cutoffs (tolerance x (r_i + r_j), no metal-metal bonds).
- bondedImages "molecules": exactly the molecules the range touches, each
  complete (BFS oracle over ase's bonds); "neighbors": exactly one shell;
  an extended 3D network never grows and says so."""
import json
import math
import time

import numpy as np
import pytest

import guiharness as H
import realdata

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")

from ase import Atoms  # noqa: E402
from ase.geometry.dimensionality.rank_determination import RDA  # noqa: E402
from ase.neighborlist import neighbor_list  # noqa: E402

TOL = 1.15
FILES = ["agte.cif", "tipbo3.cif", "molcrystal.cif"]
RANGES = [[1, 1, 1], [2, 3, 1], {"lo": [-0.1] * 3, "hi": [1.1] * 3}, {"lo": [-0.5, 0, 0], "hi": [1.5, 1, 2]}]


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("georange")
    root = H.examples_copy(tmp)
    realdata.copy(root / "real", "agte.cif", "tipbo3.cif", "molcrystal.cif", "molecule.cif", "slab_opt.out")
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_NO_WARMUP="1")
    drv = H.firefox(1280, 800)
    try:
        H.load(drv, base)
        yield drv
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def run(drv, body, *args, timeout=120):
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


_cache = {}


def payload(drv, path):
    if path not in _cache:
        _cache[path] = drv.execute_async_script(
            "const d = arguments[1]; window.MACE.api('structure?path=' + encodeURIComponent(arguments[0])).then(d)", path)
    return _cache[path]


def oracle_range(s, rng, boundary, tol=0.02):
    pbc = s.get("pbc") or [True] * 3
    if isinstance(rng, list):
        lo, hi = [0, 0, 0], rng
    else:
        lo, hi = rng["lo"], rng["hi"]
    out = set()
    for i, f0 in enumerate(s["frac"]):
        f = [f0[k] - math.floor(f0[k] + 1e-9) if pbc[k] else f0[k] for k in range(3)]
        axes = []
        for k in range(3):
            if not pbc[k]:
                axes.append([0])
            else:
                axes.append(range(math.floor(lo[k] - tol) - 1, math.floor(hi[k] + tol) + 2))
        for a in axes[0]:
            for b in axes[1]:
                for c in axes[2]:
                    g = [f[0] + a, f[1] + b, f[2] + c]
                    ok = True
                    for k in range(3):
                        if not pbc[k]:
                            continue
                        if boundary:
                            ok &= lo[k] - tol <= g[k] <= hi[k] + tol
                        else:
                            ok &= lo[k] - 1e-9 <= g[k] < hi[k] - 1e-9
                    if ok:
                        out.add((i, round(g[0] - f0[0]), round(g[1] - f0[1]), round(g[2] - f0[2])))
    return out


def js_set(r):
    return {(r["base"][k], *r["shift"][3 * k:3 * k + 3]) for k in range(len(r["base"]))}


DISP = """const s = args[0], d = G.displayList(s, args[1], args[2]);
  return {base: Array.from(d.base), shift: Array.from(d.shift), image: Array.from(d.image)};"""


@pytest.mark.parametrize("name", FILES)
@pytest.mark.parametrize("rng", RANGES, ids=["1x1x1", "2x3x1", "cell+10%", "mixed"])
@pytest.mark.parametrize("boundary", [True, False])
def test_display_range_matches_oracle(page, name, rng, boundary):
    s = payload(page, f"real/{name}")
    got = js_set(run(page, DISP, s, rng, boundary))
    assert got == oracle_range(s, rng, boundary)


@pytest.mark.parametrize("name", FILES)
def test_old_form_equals_whole_cells_from_zero(page, name):
    s = payload(page, f"real/{name}")
    for n in ([1, 1, 1], [2, 2, 2], [3, 1, 2]):
        for b in (True, False):
            old = js_set(run(page, DISP, s, n, b))
            new = js_set(run(page, DISP, s, {"lo": [0, 0, 0], "hi": n}, b))
            assert old == new, (name, n, b)


def ase_bonds(s, tol=TOL):
    """(i, j, S) from ase's neighbor list on the payload's own positions and radii, no metal-metal pairs."""
    a = Atoms(numbers=[z % 100 or z for z in s["numbers"]], cell=s["cell"], scaled_positions=s["frac"],
              pbc=s.get("pbc") or True)
    radii = np.asarray(s["radii"]) * tol
    i, j, S = neighbor_list("ijS", a, radii, self_interaction=False)
    metal = np.asarray(s["metal"], bool)
    pos = a.positions
    keep = []
    for x, y, sh in zip(i, j, S):
        if metal[x] and metal[y]:
            continue
        d = np.linalg.norm(pos[y] + sh @ a.cell.array - pos[x])
        if d * d <= 0.01:
            continue
        keep.append((int(x), int(y), tuple(int(v) for v in sh)))
    return keep


def ase_dims(s):
    bonds = ase_bonds(s)
    rda = RDA(len(s["numbers"]))
    for x, y, sh in bonds:
        if x < y or (x == y and sh > (0, 0, 0)):
            rda.insert_bond(x, y, sh)
    rda.check()
    comps, dims = rda.get_components()
    return [dims[c] for c in comps]


COMP = """const s = args[0], c = G.components(s, args[1]);
  return {comp: Array.from(c.comp), dims: c.dims, axes: c.axes};"""


# TiPbO3 (sg 87) at the drawn bond tolerance, without metal-metal bonds: Ti-O chains along c
# (1D), as ase's rank determination also finds; AgTe: whatever the oracle says
@pytest.mark.parametrize("name,want", [("tipbo3.cif", {1}), ("molcrystal.cif", {0}), ("molecule.cif", {0}),
                                       ("agte.cif", None)])
def test_components_dimension_matches_ase(page, name, want):
    s = payload(page, f"real/{name}")
    r = run(page, COMP, s, TOL)
    per_atom = [r["dims"][c] for c in r["comp"]]
    if all(s.get("pbc") or [True]):
        assert per_atom == ase_dims(s), name
    if want is not None:
        assert set(per_atom) == want, (name, set(per_atom))
    else:
        # recorded: what the oracle says for AgTe at this tolerance
        assert set(per_atom) == set(ase_dims(s))


def test_slab_final_frame_is_a_layer(page):
    fr = page.execute_async_script("""const d = arguments[1];
        window.MACE.api('traj?' + new URLSearchParams({path: arguments[0]})).then(d, (e) => d({err: String(e)}))""",
                                   "real/slab_opt.out")
    if not isinstance(fr, dict) or "frames" not in fr:
        # the OPT payload's last geometry, through the structure route
        s = payload(page, "real/slab_opt.out")
    else:
        s = fr["frames"][-1]
    assert s["pbc"][:2] == [True, True] and not s["pbc"][2]
    r = run(page, COMP, s, TOL)
    dims = {r["dims"][c] for c in r["comp"]}
    assert max(dims) == 2, dims
    axes = {r["axes"][c] for c in r["comp"] if r["dims"][c] == 2}
    assert axes == {"a, b"}, axes
    # the range on c is ignored: 1x1x3 draws what 1x1x1 does
    one = js_set(run(page, DISP, s, {"lo": [0, 0, 0], "hi": [1, 1, 3]}, True))
    assert one == js_set(run(page, DISP, s, {"lo": [0, 0, 0], "hi": [1, 1, 1]}, True))


BONDED = """const s = args[0], d = G.displayList(s, args[1], true), c = G.components(s, args[3]);
  const t0 = performance.now(), r = G.bondedImages(s, d, args[2], c, args[4] || 50000);
  return {base: Array.from(r.base), shift: Array.from(r.shift), image: Array.from(r.image), extended: r.extended,
    allExtended: r.allExtended, truncated: r.truncated, ms: performance.now() - t0, n0: d.n};"""


def bfs_molecules(start, bonds):
    adj = {}
    for x, y, sh in bonds:
        adj.setdefault(x, []).append((y, sh))
    seen, todo = set(start), list(start)
    while todo:
        i, a, b, c = todo.pop()
        for j, sh in adj.get(i, []):
            k = (j, a + sh[0], b + sh[1], c + sh[2])
            if k not in seen:
                seen.add(k)
                todo.append(k)
    return seen


@pytest.mark.parametrize("rng", [[1, 1, 1], {"lo": [-0.1] * 3, "hi": [1.1] * 3}], ids=["1x1x1", "cell+10%"])
def test_whole_molecules_on_a_molecular_crystal(page, rng):
    s = payload(page, "real/molcrystal.cif")
    r = run(page, BONDED, s, rng, "molecules", TOL)
    start = oracle_range(s, rng, True)
    want = bfs_molecules(start, ase_bonds(s))
    assert js_set(r) == want
    assert r["extended"] is None and not r["truncated"]
    assert all(v == 2 for v in r["image"][r["n0"]:])


def test_bonded_neighbors_one_shell(page):
    s = payload(page, "real/molcrystal.cif")
    r = run(page, BONDED, s, [1, 1, 1], "neighbors", TOL)
    start = oracle_range(s, [1, 1, 1], True)
    shell = set(start)
    for x, y, sh in ase_bonds(s):
        for (i, a, b, c) in start:
            if i == x:
                shell.add((y, a + sh[0], b + sh[1], c + sh[2]))
    assert js_set(r) == shell


@pytest.mark.parametrize("rng", [[1, 1, 1], {"lo": [-0.1] * 3, "hi": [1.1] * 3}], ids=["1x1x1", "cell+10%"])
def test_extended_network_never_grows(page, rng):
    s = payload(page, "real/tipbo3.cif")
    r = run(page, BONDED, s, rng, "molecules", TOL)
    assert js_set(r) == oracle_range(s, rng, True)
    assert r["extended"] and r["extended"]["dim"] >= 1 and r["allExtended"] and not r["truncated"], r["extended"]
    assert r["ms"] < 200, r["ms"]


def test_cap_reports_truncation(page):
    s = payload(page, "real/molcrystal.cif")
    n0 = run(page, BONDED, s, [1, 1, 1], "off", TOL)["n0"]
    r = run(page, BONDED, s, [1, 1, 1], "molecules", TOL, n0 + 5)
    assert r["truncated"] and len(r["base"]) == n0 + 5
