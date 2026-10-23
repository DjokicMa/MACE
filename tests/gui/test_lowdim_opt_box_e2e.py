"""The optimization view of a polymer, a slab and a molecule draws the structure
view's box (its Vacuum mode around each frame's atoms), never CRYSTAL's dummy
vectors, in all four looks (Overlay, Two-tone, Side by side, Displacement map)
and with whole cells repeated along the periodic axes (1.1.6).

The bug: Display Range 3×1×1 on polyacetylene_opt.out drew a skewed sliver off
to the side of the atoms. The optimization view passes its cells as
[n1, n2, n3], which the vacuum box (viewport/vacuum.js vacuumBox) took for one
count, so the periodic edges came out NaN. The FREQ modes view of a molecule
draws its Vacuum box too, as the structure view does. Firefox, the bundled
example suite."""
import lzma
import time

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")

SUITE = H.REPO_ROOT / "examples" / "suite"
RUNS = {
    "polymer": "4-polymer-1d/polyacetylene/OPT/polyacetylene_opt.out",
    "slab": "3-slab-2d/graphene/OPT/graphene_opt.out",
    "molecule": "5-molecule-0d/EC/OPT/EC_opt.out",
}
MOLECULE_FREQ = "5-molecule-0d/EC/FREQ/EC_freq.out"
LOOKS = ("overlay", "twotone", "side", "dmap")


def unpack(root, rel):
    src = SUITE / (rel + ".xz")
    if not src.is_file():
        pytest.skip("the bundled example suite is not here")
    dst = root / "suite" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(lzma.decompress(src.read_bytes()))
    return "suite/" + rel


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lowdimopt")
    root = H.examples_copy(tmp)
    paths = {k: unpack(root, rel) for k, rel in RUNS.items()}
    paths["freq"] = unpack(root, MOLECULE_FREQ)
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1")
    drv = None
    try:
        drv = H.browser("firefox", width=1400, height=900)
        drv.set_script_timeout(120)
        H.load(drv, base)
        drv.execute_script("window.__mace.settings.set('reduceMotion', 'on')")
        yield drv, paths
    finally:
        if drv is not None:
            drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def open_run(drv, path):
    drv.execute_script("window.__mace.closeAll()")
    drv.execute_async_script("const done = arguments[1]; window.__mace.open(arguments[0]).then(() => done())", path)
    H.wait(drv, "return !!(window.MACE.state.traj && window.MACE.state.cmp && window.MACE.state.viewers['viewer-before'])", 60)
    time.sleep(0.5)


def set_look(drv, look):
    drv.execute_script("window.MACE.setLook(arguments[0])", look)
    H.wait(drv, "return window.MACE.look() === arguments[0] && window.MACE.state.viewers['viewer-before'].getModel() !== null", 10, look)
    time.sleep(0.6)


# every drawn atom of each optimization viewer and each box drawn there; per box: its edges (a, b, c from the
# origin corner), and whether each atom is inside it: within 1e-3 Å along a non-periodic axis, 0.3 Å along a
# periodic one (an atom on the cell's face, a step's unwrapped coordinates); arguments[0]: the pbc
DRAWN = """
const out = {}, pbc = arguments[0] || [true, true, true];
for (const id of ['viewer-before', 'viewer-after']) {
  const v = window.MACE.state.viewers[id];
  if (!v || !v.getModel() || id === 'viewer-after' && window.MACE.look() !== 'side') continue;
  const atoms = [];
  for (const m of (v.models || [v.getModel()]).filter(Boolean))
    for (const a of m.selectedAtoms({})) atoms.push([a.x, a.y, a.z]);
  const boxes = (v.cellCorners || []).map((c) => {
    const o = c[0], E = [1, 2, 3].map((k) => [0, 1, 2].map((j) => c[k][j] - o[j]));
    const det = E[0][0] * (E[1][1] * E[2][2] - E[1][2] * E[2][1]) - E[0][1] * (E[1][0] * E[2][2] - E[1][2] * E[2][0])
      + E[0][2] * (E[1][0] * E[2][1] - E[1][1] * E[2][0]);
    const len = E.map((e) => Math.hypot(...e));
    const frac = (p) => {
      if (!(Math.abs(det) > 1e-9)) return null;
      const d = [0, 1, 2].map((j) => p[j] - o[j]);
      const col = (k, r) => { const M = E.map((e) => e.slice()); M[k] = r; return M; };
      const D = (M) => M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
        + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
      return [0, 1, 2].map((k) => D(col(k, d)) / det);
    };
    return { finite: c.flat().every(Number.isFinite), len, inside: atoms.map((p) => {
      const f = frac(p), tol = (k) => (pbc[k] ? 0.3 : 1e-3) / (len[k] || 1);
      return !!f && f.every((x, k) => x > -tol(k) && x < 1 + tol(k)); }) };
  });
  out[id] = { n: atoms.length, boxes };
}
return out;
"""


def check_boxes(drv, what):
    views = drv.execute_script(DRAWN, drv.execute_script("return window.MACE.state.traj.frames[0].pbc"))
    assert views.get("viewer-before"), what
    for vid, v in views.items():
        assert v["n"] > 0, (what, vid)
        assert v["boxes"], (what, vid, "no box drawn")
        for b in v["boxes"]:
            assert b["finite"], (what, vid, b["len"])
        # every drawn atom (both geometries, every repeated cell) lies inside a box drawn in its view
        for k in range(v["n"]):
            assert any(b["inside"][k] for b in v["boxes"]), (what, vid, k, [b["len"] for b in v["boxes"]])
    return views


@pytest.mark.parametrize("name", list(RUNS))
def test_every_look_boxes_the_atoms(app, name):
    drv, paths = app
    open_run(drv, paths[name])
    for n in (1, 3):
        drv.execute_script("window.__mace.run(arguments[0])", f"view.range{n}")
        time.sleep(0.6)
        cells = drv.execute_script("return window.__mace.optCells()")
        pbc = drv.execute_script("return window.MACE.state.traj.frames[0].pbc")
        assert cells == [n if p else 1 for p in pbc]
        for look in LOOKS:
            views = check_boxes(drv, f"{name} {look} {cells}")
            box = views["viewer-before"]["boxes"][-1]
            a = drv.execute_script("const c = window.MACE.state.traj.frames[0].cell; return c.map((r) => Math.hypot(...r))")
            for k in range(3):
                if pbc[k]:
                    # a periodic axis spans the cells repeated along it
                    assert abs(box["len"][k] - cells[k] * a[k]) < 0.05, (name, look, k, box["len"], a)
                else:
                    # a non-periodic one the Bounding box (the atoms + 1 Å each side), not CRYSTAL's 0.5 Å dummy vector
                    assert box["len"][k] >= 2.0 - 1e-6 and box["len"][k] < 20, (name, look, k, box["len"])
            if look != LOOKS[-1]:
                set_look(drv, LOOKS[LOOKS.index(look) + 1])
        set_look(drv, "overlay")
    assert not H.errors(drv), H.errors(drv)


def test_custom_and_open_vacuum(app):
    drv, paths = app
    open_run(drv, paths["polymer"])
    drv.execute_script("window.__mace.run('view.range3')")
    time.sleep(0.6)
    set_look(drv, "overlay")
    ext = drv.execute_script("""const s = window.MACE.state.traj.frames[window.MACE.state.frame];
      return [1, 2].map((k) => { const xs = s.cart.map((p) => p[k]); return Math.max(...xs) - Math.min(...xs); })""")
    for mode, gap in (("gap", 12), ("open", None), ("bbox", None)):
        drv.execute_script("""const s = document.getElementById('opt-vacuum-mode'); s.value = arguments[0];
          s.dispatchEvent(new Event('change'));
          if (arguments[1] !== null) { const g = document.getElementById('opt-vacuum-gap'); g.value = String(arguments[1]);
            g.dispatchEvent(new Event('change')); }""", mode, gap)
        time.sleep(0.8)
        if mode == "gap":
            box = check_boxes(drv, mode)["viewer-before"]["boxes"][-1]
            assert abs(box["len"][1] - (ext[0] + 12)) < 0.05 and abs(box["len"][2] - (ext[1] + 12)) < 0.05, (box, ext)
        elif mode == "open":
            # no walls along b and c: the outline is the a edges in the atoms' mid-plane
            boxes = drv.execute_script(DRAWN, [True, False, False])["viewer-before"]["boxes"]
            assert boxes and all(b["finite"] and b["len"][1] < 1e-6 and b["len"][2] < 1e-6 for b in boxes), boxes
        else:
            check_boxes(drv, mode)
    assert not H.errors(drv), H.errors(drv)


def test_freq_molecule_draws_its_vacuum_box(app):
    drv, paths = app
    drv.execute_script("window.__mace.closeAll()")
    drv.execute_async_script("const done = arguments[1]; window.__mace.open(arguments[0]).then(() => done())", paths["freq"])
    H.wait(drv, "const v = window.__mace.freqView, s = v && v.state(); return s && s.mode", 60)
    st = drv.execute_script("return window.__mace.freqView.state()")
    assert st["display"]["cell"] is True, st["display"]
    assert not H.errors(drv), H.errors(drv)
