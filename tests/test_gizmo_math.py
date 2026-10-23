"""The Rotate and Move gizmos' geometry (mace/gui/static/js/viewport/gizmo_math.js, camera.js), run in node:
exact pointer-to-plane, pointer-to-axis and pointer-to-ring math under a perspective projection, the
modifier rules of a rotate drag (snap, Ctrl inverting it, Shift snapping every angle, Alt keeping one axis,
Tab picking the other), the turn applied to the view or to atoms (the same as the camera's own free drag),
the readout both ways round, typed axes, the selection made whole across the cell's faces, and the custom
angle snap."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

JS = Path(__file__).resolve().parents[1] / "mace" / "gui" / "static" / "js"

pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node not installed")

SCRIPT = r"""
import * as M from %(math)s;
import * as G from %(core)s;
import { rotationPrefs, invertedSnap, dragOrientation } from %(camera)s;
const out = {};
// a perspective camera: model -> screen (v down), P[11] = 1
const f = 900, cx = 400, cy = 300, q = G.qaxis([0.3, -0.5, 0.8], 37), t = [0.4, -0.2, -40];
const K = [[f, 0, cx], [0, -f, cy], [0, 0, 1]];
const R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]].map((e) => G.qrotate(q, e));     // columns: images of x, y, z
const Rm = [0, 1, 2].map((r) => [R[0][r], R[1][r], R[2][r]]);
// camera space z toward the viewer: depth = -(R p + t)_z
const rows = [0, 1, 2].map((r) => [...Rm[r], t[r]]);
const flip = [rows[0], rows[1], rows[2].map((x) => -x)];
const KP = [0, 1, 2].map((r) => [0, 1, 2, 3].map((c) => K[r][0] * flip[0][c] + K[r][1] * flip[1][c] + K[r][2] * flip[2][c]));
const P = [].concat(...KP).map((x) => x / KP[2][3]);
const proj = (p) => G.project(P, p[0], p[1], p[2]);
const toward = G.qrotate(G.qconj(q), [0, 0, 1]);
const o = [1.2, -0.7, 2.1];
// planeHit: a plane point's screen image comes back to its coordinates
{
  const e1 = G.unit3([1, 0.2, -0.1]), e2 = G.unit3(G.cross3([0.2, 0.3, 1], e1));
  const p = G.add3(o, G.add3(G.scale3(e1, 1.7), G.scale3(e2, -2.3)));
  const s = proj(p);
  out.planeHit = M.planeHit(P, o, e1, e2, s[0], s[1]);
}
// axisDrag: a drag between the images of o + 0.5 e and o + 3.25 e is 2.75 along e
{
  const e = G.unit3([0.6, -0.3, 0.74]);
  const s0 = proj(G.add3(o, G.scale3(e, 0.5))), s1 = proj(G.add3(o, G.scale3(e, 3.25)));
  out.axisDrag = M.axisDrag(P, o, e, toward, s0[0], s0[1], s1[0], s1[1]);
  // planeDrag: the in-plane displacement between two plane points
  const e1 = [1, 0, 0], e2 = [0, 1, 0];
  const a = G.add3(o, [0.3, 0.4, 0]), b = G.add3(o, [-1.1, 2.0, 0]);
  const sa = proj(a), sb = proj(b);
  out.planeDrag = M.planeDrag(P, o, e1, e2, sa[0], sa[1], sb[0], sb[1]);
}
// ringAngle: the pointer on the ring about n at 63° (counterclockwise seen from n's tip)
{
  const n = G.unit3([0.2, 0.4, 0.9]), [e1, e2] = M.planeBasis(n), th = 63 * Math.PI / 180, r = 2.5;
  const p = G.add3(o, G.add3(G.scale3(e1, r * Math.cos(th)), G.scale3(e2, r * Math.sin(th))));
  const s = proj(p), s0 = proj(G.add3(o, G.scale3(e1, r)));
  out.ringAngle = M.ringAngle(P, o, n, toward, s[0], s[1], { x: s0[0], y: s0[1], model: G.add3(o, G.scale3(e1, r)) })
    - M.ringAngle(P, o, n, toward, s0[0], s0[1], { x: s0[0], y: s0[1] });
  out.basis = G.cross3(e1, e2).map((x, k) => x - n[k]);
  // edge-on: the linear fallback gives the right sign and size for a small turn
  const m = G.unit3(G.sub3([0, 0, 1], G.scale3(toward, G.dot3([0, 0, 1], toward))));   // in the screen plane
  const [f1] = M.planeBasis(m), g = G.add3(o, G.scale3(f1, r));
  const g2 = G.add3(o, G.qrotate(G.qaxis(m, 2), G.scale3(f1, r)));
  const a = proj(g), b = proj(g2);
  out.edgeOn = M.ringAngle(P, o, m, toward, b[0], b[1], { x: a[0], y: a[1], model: g });
}
out.unwrap = [M.unwrap(170, 0, -175), M.unwrap(-170, 0, 175), M.unwrap(10, 0, 20)];
out.resolve = {
  free: M.resolveAngles([88.3, 43.2], { snap: false, step: 45 }).degs,
  snap: M.resolveAngles([88.3, 43.2], { snap: true, step: 45 }).degs,
  shift: M.resolveAngles([88.3, 43.2], { snap: false, step: 45, snapAll: true }).degs,
  alt: M.resolveAngles([88.3, 43.2], { step: 45, isolate: true }),
  altTab: M.resolveAngles([88.3, 43.2], { step: 45, isolate: true, pick: 1 }),
  altTab2: M.resolveAngles([88.3, 43.2], { step: 45, isolate: true, pick: 2 }).degs,
};
// a free rotate drag of the view is the camera's own free drag (the same 0.4°/px, the same axes)
{
  const q0 = G.qaxis([0.1, 0.9, -0.3], 71), qi = G.qconj(q0), dx = 130, dy = -55;
  const qm = M.composeTurn([{ axis: G.qrotate(qi, [0, 1, 0]), deg: dx * M.DEG_PER_PX }, { axis: G.qrotate(qi, [1, 0, 0]), deg: dy * M.DEG_PER_PX }]);
  const a = M.turnedView(q0, qm), b = dragOrientation(q0, dx, dy, { mode: "free", step: 0 });
  out.sameAsCamera = Math.min(G.norm3(G.sub3(a, b)) + Math.abs(a[3] - b[3]), G.norm3(G.add3(a, b)) + Math.abs(a[3] + b[3]));
  // atoms turned by qm in front of the camera at q0 land where the turned view shows the unturned ones
  const pts = [[1, 2, 3], [-2, 0.5, 1]], piv = [0.5, 0.5, 0.5];
  const moved = M.turnAbout(pts, piv, qm);
  out.atomsMatchView = Math.max(...pts.map((p, k) => G.norm3(G.sub3(G.qrotate(q0, G.sub3(moved[k], piv)), G.qrotate(a, G.sub3(p, piv))))));
}
out.both = [M.bothWays(30), M.bothWays(-30), M.bothWays(0), M.bothWays(400), M.bothWays(-90)];
{
  const cell = [[4, 0, 0], [0, 4, 0], [0, 0, 6]];
  out.axes = { x: M.axisVector("x"), c: M.axisVector("c", cell), uvw: M.axisVector("110", cell), uvw2: M.axisVector("[1 -1 0]", cell),
    cartesian: M.axisVector("0 0 2", null), astar: M.axisVector("a*", [[4, 0, 0], [2, 3, 0], [0, 0, 5]]), bad: M.axisVector("0 0 0", cell),
    nocell: M.axisVector("a", null), view: M.axisVector("view", null, G.qaxis([1, 0, 0], 90)) };
  // a molecule across the a face: made whole, its centroid between the halves (not mid-cell)
  const s = { cell, pbc: [true, true, true], frac: [[0.05, 0.5, 0.5], [0.95, 0.5, 0.5], [0.5, 0.5, 0.5]] };
  out.whole = M.wholeSelection(s, new Set([0, 1]));
}
{
  const get = (o) => (k) => o[k];
  out.prefs = [rotationPrefs(get({ "viewport.angleSnap": "custom", "viewport.snapStep": 22.5 })),
    rotationPrefs(get({ "viewport.angleSnap": "45", "viewport.snapStep": 22.5 })),
    rotationPrefs(get({ "viewport.angleSnap": "off" })),
    rotationPrefs(get({ "viewport.angleSnap": "custom", "viewport.snapStep": 500 }))];
  out.inverted = [invertedSnap(out.prefs[0]).step, invertedSnap(out.prefs[2]).step, M.snapStepOf(out.prefs[2]), M.snapStepOf(out.prefs[1])];
}
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def got(tmp_path_factory):
    script = tmp_path_factory.mktemp("gizmo") / "run.mjs"
    script.write_text(SCRIPT % {"math": json.dumps((JS / "viewport" / "gizmo_math.js").as_uri()),
                                "core": json.dumps((JS / "geom" / "core.js").as_uri()),
                                "camera": json.dumps((JS / "viewport" / "camera.js").as_uri())})
    r = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_pointer_meets_plane_axis_and_ring_exactly_in_perspective(got):
    assert got["planeHit"] == pytest.approx([1.7, -2.3], abs=1e-6)
    assert got["axisDrag"] == pytest.approx(2.75, abs=1e-6)
    assert got["planeDrag"] == pytest.approx([-1.4, 1.6, 0.0], abs=1e-6)
    assert got["ringAngle"] == pytest.approx(63.0, abs=1e-6)
    assert max(abs(x) for x in got["basis"]) < 1e-12           # e1 x e2 = n: positive turns are right-handed
    assert got["edgeOn"] == pytest.approx(2.0, rel=0.02)        # the ring seen edge-on: the tangent fallback


def test_unwrap_values(got):
    assert got["unwrap"] == pytest.approx([185.0, -185.0, 20.0])


def test_modifiers_snap_isolate_and_pick(got):
    r = got["resolve"]
    assert r["free"] == pytest.approx([88.3, 43.2])
    assert r["snap"] == [90, 45]                                 # the snap (or Ctrl in free mode)
    assert r["shift"] == [90, 45]                                # Shift: every angle snaps, whatever the mode
    assert r["alt"]["degs"] == [90, 0] and r["alt"]["kept"] == 0  # Alt: the larger turn only, snapped
    assert r["altTab"]["degs"] == [0, 45] and r["altTab"]["kept"] == 1   # Tab: the other one
    assert r["altTab2"] == [90, 0]                               # and round again


def test_turn_matches_the_camera_and_moves_atoms_as_the_view(got):
    assert got["sameAsCamera"] < 1e-12
    assert got["atomsMatchView"] < 1e-9


def test_readout_both_ways_round(got):
    assert got["both"] == [["+30.0°", "−330.0°"], ["−30.0°", "+330.0°"], ["0.0°", "0.0°"], ["+40.0°", "−320.0°"], ["−90.0°", "+270.0°"]]


def test_typed_axes(got):
    a = got["axes"]
    assert a["x"] == [1, 0, 0] and a["c"] == pytest.approx([0, 0, 1])
    assert a["uvw"] == pytest.approx([2 ** -0.5, 2 ** -0.5, 0]) and a["uvw2"] == pytest.approx([2 ** -0.5, -(2 ** -0.5), 0])
    assert a["cartesian"] == pytest.approx([0, 0, 1])
    # a* is perpendicular to b and c
    import numpy as np
    assert abs(np.dot(a["astar"], [2, 3, 0])) < 1e-12 and abs(a["astar"][2]) < 1e-12
    assert a["bad"] is None and a["nocell"] is None
    assert a["view"] == pytest.approx([0, 1, 0], abs=1e-12)    # turned 90° about x: model +y faces the viewer


def test_selection_made_whole_across_the_boundary(got):
    w = got["whole"]
    assert w["sel"] == [0, 1]
    assert w["frac"][1] == pytest.approx([-0.05, 0.5, 0.5])       # the image next to the first atom
    assert w["pivot"] == pytest.approx([0.0, 2.0, 3.0])            # on the a face, not mid-cell


def test_custom_angle_snap_and_its_inversion(got):
    p = got["prefs"]
    assert p[0]["step"] == 22.5 and p[0]["custom"] == 22.5
    assert p[1]["step"] == 45 and p[2]["step"] == 0 and p[2]["custom"] == 15
    assert p[3]["step"] == 15                                    # out of range: the 15° default
    assert got["inverted"] == [0, 15, 15, 45]
