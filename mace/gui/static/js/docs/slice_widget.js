/* The Volume document's slice widget (1.1.6): where the slice is taken from, at a glance.
 *
 * A plane is a lattice plane of the grid (a, b or c: the grid's own planes, exact values from the
 * server), a Miller plane (hkl) (normal h a* + k b* + l c*) or the plane normal to a direction [uvw]
 * (u a + v b + w c), at a position from 0 to 1 across the box along its normal, and tilted and twisted
 * about its own in-plane axes. The geometry is the server's (mace/gui/volume.py plane_geometry: the
 * same arithmetic, so the outline drawn here is the polygon the server samples), and the box is the
 * grid's (a periodic axis spans its period, a non-periodic one its points).
 *
 * The mini unit-cell preview draws the box from the 3D view's own direction with the plane cut through
 * it; dragging the plane moves it along its normal, Shift-drag (or a right-drag) turns it, the wheel
 * and the arrow keys step it. While a plane moves, the map is sampled here from the grid the 3D view
 * already holds (samplePlane); the full grid's values come from the server when it stops. */

import { legible } from "../viewport/contrast.js";

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => Math.hypot(a[0], a[1], a[2]);
const unit = (a) => { const n = norm(a) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };
export const PLANE_MAX_INDEX = 12;

/** The inverse of a 3x3 matrix given as rows (null when it is singular). */
export function inv3(m) {
  const [a, b, c] = m;
  const A = cross(b, c), B = cross(c, a), C = cross(a, b), det = dot(a, A);
  if (!det || !Number.isFinite(det)) return null;
  // the rows of inv(m) are the columns of [A B C] / det
  return [[A[0] / det, B[0] / det, C[0] / det], [A[1] / det, B[1] / det, C[1] / det], [A[2] / det, B[2] / det, C[2] / det]];
}
/** Row vector x matrix (rows). */
const vm = (v, m) => [0, 1, 2].map((j) => v[0] * m[0][j] + v[1] * m[1][j] + v[2] * m[2][j]);
const transpose = (m) => [0, 1, 2].map((j) => m.map((r) => r[j]));

/** The box the plane cuts (Å): the server's (meta.box), else the full grid's extent. */
export function boxOf(meta) {
  if (meta.box && meta.box.edges) return meta.box;
  const o = meta.cell_origin || meta.origin, V = meta.full_voxels, n = meta.full_shape, pbc = meta.pbc || [false, false, false];
  return { origin: o, edges: V.map((r, k) => r.map((x) => x * (pbc[k] ? n[k] : Math.max(1, n[k] - 1)))), pbc, period: n.slice() };
}
const corners = (box) => {
  const out = [], E = box.edges;
  for (const i of [0, 1]) for (const j of [0, 1]) for (const k of [0, 1]) out.push(add(box.origin, add(add(mul(E[0], i), mul(E[1], j)), mul(E[2], k))));
  return out;
};
/** Rodrigues: v turned by deg about the unit axis k. */
function rotate(v, k, deg) {
  const t = (deg * Math.PI) / 180;
  if (!t) return v;
  const c = Math.cos(t), s = Math.sin(t);
  return add(add(mul(v, c), mul(cross(k, v), s)), mul(k, dot(k, v) * (1 - c)));
}
/** Where the plane cuts the box: its corners in order around the plane. */
export function boxPolygon(box, n, p, e1, e2) {
  const C = corners(box), scale = Math.max(1e-9, ...box.edges.flat().map(Math.abs)), pts = [];
  // corner index i*4 + j*2 + k; an edge joins two corners that differ in one index
  for (let a = 0; a < 8; a++) for (const bit of [4, 2, 1]) {
    if (a & bit) continue;
    const c0 = C[a], c1 = C[a | bit], s0 = dot(sub(c0, p), n), s1 = dot(sub(c1, p), n);
    let q = null;
    if (Math.abs(s0) < 1e-9 * scale) q = c0;
    else if (Math.abs(s1) < 1e-9 * scale) q = c1;
    else if (s0 * s1 < 0) q = add(c0, mul(sub(c1, c0), s0 / (s0 - s1)));
    if (q && !pts.some((x) => norm(sub(x, q)) < 1e-6 * scale)) pts.push(q);
  }
  if (pts.length < 3) return [];
  const m = mul(pts.reduce(add, [0, 0, 0]), 1 / pts.length);
  return pts.map((x) => ({ x, a: Math.atan2(dot(sub(x, m), e2), dot(sub(x, m), e1)) })).sort((u, w) => u.a - w.a).map((u) => u.x);
}
/** A lattice plane of the grid: the plane at grid index `index` along a, b or c (the server's slice). */
export function axisGeometry(meta, axis, index) {
  const ax = "abc".indexOf(axis), others = [0, 1, 2].filter((k) => k !== ax), V = meta.full_voxels, N = meta.full_shape;
  const o = meta.cell_origin || meta.origin, p0 = add(o, mul(V[ax], index));
  const u = mul(V[others[0]], N[others[0]]), w = mul(V[others[1]], N[others[1]]);
  const n = unit(cross(u, w)), e1 = unit(u), e2 = cross(n, e1);
  const box = boxOf(meta), C = corners(box).map((c) => dot(sub(c, box.origin), n));
  const dmin = Math.min(...C), dmax = Math.max(...C), d = dot(sub(p0, box.origin), n);
  return { mode: "axis", axis, index, n: N[ax], normal: n, point: p0, e1, e2, d, dmin, dmax,
    polygon: [p0, add(p0, u), add(add(p0, u), w), add(p0, w)], pos: N[ax] > 1 ? index / (N[ax] - 1) : 0 };
}
/** A (hkl) or [uvw] plane (mace/gui/volume.py plane_geometry, the same arithmetic). */
export function planeGeometry(box, { mode, ints, pos = 0.5, tilt = 0, twist = 0 }) {
  const E = box.edges, o = box.origin;
  const R = inv3(E);
  if (!R || !ints || !ints.some((x) => x)) return null;
  const nvec = vm(ints, mode === "hkl" ? transpose(R) : E), nlen = norm(nvec);
  const nhat = mul(nvec, 1 / nlen);
  const proj = corners(box).map((c) => dot(sub(c, o), nhat));
  // a hair inside the box: at 0 or 1 an oblique plane only touches a corner or an edge (volume.py PLANE_END_EPS)
  const dmin = Math.min(...proj), dmax = Math.max(...proj), eps = 1e-3 * (dmax - dmin);
  const d = Math.min(Math.max(dmin + pos * (dmax - dmin), dmin + eps), dmax - eps);
  const center = add(o, mul(add(add(E[0], E[1]), E[2]), 0.5));
  const point = add(center, mul(nhat, d - dot(sub(center, o), nhat)));
  let k = 0, best = Infinity;
  for (let i = 0; i < 3; i++) { const t = Math.abs(dot(E[i], nhat)) / Math.max(1e-12, norm(E[i])); if (t < best - 1e-12) { best = t; k = i; } }
  const e1 = unit(sub(E[k], mul(nhat, dot(E[k], nhat)))), e2 = cross(nhat, e1);
  const n2 = rotate(nhat, e1, tilt), e2b = rotate(e2, e1, tilt);
  const n3 = rotate(n2, e2b, twist), e1c = rotate(e1, e2b, twist);
  const out = { mode, ints: ints.slice(), pos, tilt, twist, normal: n3, point, e1: e1c, e2: e2b, d, dmin, dmax,
    position_A: d - dmin, polygon: boxPolygon(box, n3, point, e1c, e2b) };
  if (mode === "hkl") { out.spacing_A = 1 / nlen; out.miller_t = d * nlen; }
  return out;
}
/** The plane sampled from the decimated grid the 3D view holds (Float32Array, C order), for a quick map
 * while the plane moves: rows of z along e2, columns along e1, null outside the polygon. */
export function samplePlane(meta, data, geom, maxSide = 140) {
  const poly = geom.polygon;
  if (!poly || poly.length < 3 || !data) return null;
  const p = geom.point, e1 = geom.e1, e2 = geom.e2;
  const U = poly.map((x) => dot(sub(x, p), e1)), W = poly.map((x) => dot(sub(x, p), e2));
  const u0 = Math.min(...U), u1 = Math.max(...U), w0 = Math.min(...W), w1 = Math.max(...W);
  const step = Math.max(u1 - u0, w1 - w0) / Math.max(2, maxSide - 1) || 1;
  const xs = [], ys = [];
  for (let x = u0; x <= u1 + 0.5 * step; x += step) xs.push(x);
  for (let y = w0; y <= w1 + 0.5 * step; y += step) ys.push(y);
  const box = boxOf(meta), Bi = inv3(box.edges), pbc = box.pbc || meta.pbc || [false, false, false];
  const Vi = inv3(meta.voxels), [nx, ny, nz] = meta.shape, go = meta.origin;
  if (!Bi || !Vi) return null;
  // the polygon's edges, for the inside test (convex)
  const m = U.length;
  let area = 0;
  for (let i = 0; i < m; i++) area += U[i] * W[(i + 1) % m] - U[(i + 1) % m] * W[i];
  const sg = area >= 0 ? 1 : -1;
  const z = ys.map((y) => xs.map((x) => {
    for (let i = 0; i < m; i++) {
      const j = (i + 1) % m;
      if (sg * ((U[j] - U[i]) * (y - W[i]) - (W[j] - W[i]) * (x - U[i])) < -1e-9) return null;
    }
    let r = add(p, add(mul(e1, x), mul(e2, y)));
    // a periodic axis wraps into the box
    const f = vm(sub(r, box.origin), Bi);
    for (let k = 0; k < 3; k++) if (pbc[k]) f[k] -= Math.floor(f[k]);
    r = add(box.origin, vm(f, box.edges));
    const g = vm(sub(r, go), Vi);
    const lim = [nx - 1, ny - 1, nz - 1];
    for (let k = 0; k < 3; k++) {
      if (g[k] < -1e-6 || g[k] > lim[k] + 1e-6) { if (!pbc[k]) return null; }
      g[k] = Math.max(0, Math.min(lim[k], g[k]));
    }
    const i0 = Math.min(Math.floor(g[0]), Math.max(0, nx - 2)), j0 = Math.min(Math.floor(g[1]), Math.max(0, ny - 2)), k0 = Math.min(Math.floor(g[2]), Math.max(0, nz - 2));
    const fx = g[0] - i0, fy = g[1] - j0, fz = g[2] - k0;
    const i1 = Math.min(i0 + 1, nx - 1), j1 = Math.min(j0 + 1, ny - 1), k1 = Math.min(k0 + 1, nz - 1);
    const at = (i, j, k) => data[(i * ny + j) * nz + k];
    return at(i0, j0, k0) * (1 - fx) * (1 - fy) * (1 - fz) + at(i1, j0, k0) * fx * (1 - fy) * (1 - fz)
      + at(i0, j1, k0) * (1 - fx) * fy * (1 - fz) + at(i0, j0, k1) * (1 - fx) * (1 - fy) * fz
      + at(i1, j1, k0) * fx * fy * (1 - fz) + at(i1, j0, k1) * fx * (1 - fy) * fz
      + at(i0, j1, k1) * (1 - fx) * fy * fz + at(i1, j1, k1) * fx * fy * fz;
  }));
  return { x: xs, y: ys, z, outline: [U, W], preview: true };
}
/** The 3D view's rotation (3Dmol getView()'s quaternion) as rows: right, up, toward the viewer. */
export function viewRows(q) {
  const [x, y, z, w] = q && q.length === 4 && q.every(Number.isFinite) ? q : [-0.43, 0.22, 0.11, 0.87];
  return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]].map(unit);
}
/** A short words-only description of a plane (the slice map's heading, the preview's label). */
export function planeLabel(g) {
  if (!g) return "";
  if (g.mode === "axis") return `Plane ${g.axis} = ${g.index}`;
  const ix = g.ints.map((x) => (x < 0 ? `−${-x}` : String(x))).join(" ");
  const turned = g.tilt || g.twist ? ` turned ${+g.tilt.toFixed(1)}° / ${+g.twist.toFixed(1)}°` : "";
  return `${g.mode === "hkl" ? `(${ix})` : `⊥ [${ix}]`}${turned}`;
}

/** The mini unit-cell preview: `S()` gives {box, geom, quat, canRotate}; `change(delta, final)` gets
 * {pos} (a step across the box), {tilt, twist} (degrees) or {} with final = true when a drag ends. */
export function createSlicePreview(canvas, { S, change, css }) {
  let proj = null, drag = null;
  const g = () => canvas.getContext("2d");
  function layout() {
    const st = S();
    if (!st || !st.box) return null;
    const rows = viewRows(st.quat), C = corners(st.box);
    const center = mul(C.reduce(add, [0, 0, 0]), 1 / 8);
    const P = (x) => { const r = sub(x, center); return [dot(rows[0], r), -dot(rows[1], r), dot(rows[2], r)]; };
    const pc = C.map(P), W = canvas.clientWidth || 220, Hh = canvas.clientHeight || 160;
    const xs = pc.map((c) => c[0]), ys = pc.map((c) => c[1]);
    const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 1e-6);
    const s = (Math.min(W, Hh) - 44) / span, cx = W / 2 - s * (Math.max(...xs) + Math.min(...xs)) / 2, cy = Hh / 2 - s * (Math.max(...ys) + Math.min(...ys)) / 2;
    const to = (x) => { const c = P(x); return [cx + s * c[0], cy + s * c[1], c[2]]; };
    return { st, rows, C, to, s, W, H: Hh, center };
  }
  function draw() {
    const L = layout(), dpr = window.devicePixelRatio || 1, W = canvas.clientWidth || 220, Hh = canvas.clientHeight || 160;
    if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(Hh * dpr)) { canvas.width = Math.round(W * dpr); canvas.height = Math.round(Hh * dpr); }
    const c = g();
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, W, Hh);
    proj = L;
    if (!L) return;
    const { st, C, to } = L, sc = C.map(to), zc = sc.reduce((n, x) => n + x[2], 0) / 8;
    // the inspector's surface under it: every mark at least 3:1 on it (a faint back edge 2:1)
    const ground = css("--surface-2"), edge = legible(css("--text-2"), ground), faint = legible(css("--text-3"), ground, 2), accent = legible(css("--accent"), ground);
    // the box: its 12 edges, those behind the middle dashed
    for (let a = 0; a < 8; a++) for (const bit of [4, 2, 1]) {
      if (a & bit) continue;
      const p = sc[a], q = sc[a | bit], back = (p[2] + q[2]) / 2 < zc - 1e-6;
      c.beginPath(); c.moveTo(p[0], p[1]); c.lineTo(q[0], q[1]);
      c.setLineDash(back ? [3, 3] : []); c.strokeStyle = back ? faint : edge; c.lineWidth = back ? 1 : 1.5; c.stroke();
    }
    c.setLineDash([]);
    // a, b, c at the far end of each edge from the origin corner, in the axis colors
    ["a", "b", "c"].forEach((name, k) => {
      const end = sc[[4, 2, 1][k]], o = sc[0], dx = end[0] - o[0], dy = end[1] - o[1], l = Math.hypot(dx, dy) || 1;
      if (l < 8) return;                       // an edge that points at the viewer: no room for its letter
      c.font = "600 12px system-ui, sans-serif"; c.fillStyle = legible(css(`--axis-${name}`), ground); c.textAlign = "center"; c.textBaseline = "middle";
      c.fillText(name, end[0] + (dx / l) * 9, end[1] + (dy / l) * 9);
    });
    // the plane: filled, outlined, with its normal
    const poly = st.geom && st.geom.polygon ? st.geom.polygon.map(to) : [];
    if (poly.length >= 3) {
      c.beginPath(); poly.forEach((p, i) => (i ? c.lineTo(p[0], p[1]) : c.moveTo(p[0], p[1]))); c.closePath();
      c.globalAlpha = drag ? 0.5 : 0.35; c.fillStyle = accent; c.fill(); c.globalAlpha = 1;
      c.lineWidth = 2; c.strokeStyle = accent; c.stroke();
      const m = mul(st.geom.polygon.reduce(add, [0, 0, 0]), 1 / st.geom.polygon.length);
      const len = 0.18 * Math.max(...st.box.edges.map(norm));
      const a = to(m), b = to(add(m, mul(st.geom.normal, len)));
      c.beginPath(); c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); c.strokeStyle = accent; c.lineWidth = 1.5; c.stroke();
      const ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
      c.beginPath(); c.moveTo(b[0], b[1]); c.lineTo(b[0] - 7 * Math.cos(ang - 0.4), b[1] - 7 * Math.sin(ang - 0.4));
      c.lineTo(b[0] - 7 * Math.cos(ang + 0.4), b[1] - 7 * Math.sin(ang + 0.4)); c.closePath(); c.fillStyle = accent; c.fill();
    }
    c.font = "11px system-ui, sans-serif"; c.fillStyle = edge; c.textAlign = "right"; c.textBaseline = "top";
    c.fillText(planeLabel(st.geom), W - 6, 4);
  }
  /** The plane normal's direction on the canvas (px per Å) and the box's extent along it (Å). */
  function normalOnScreen() {
    if (!proj || !proj.st.geom) return null;
    const n = proj.st.geom.normal, sx = dot(proj.rows[0], n) * proj.s, sy = -dot(proj.rows[1], n) * proj.s;
    return { sx, sy, len: Math.hypot(sx, sy), extent: proj.st.geom.dmax - proj.st.geom.dmin };
  }
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    if (!proj) return;
    e.preventDefault();
    canvas.focus({ preventScroll: true });
    canvas.setPointerCapture(e.pointerId);
    drag = { x: e.clientX, y: e.clientY, turn: e.shiftKey || e.button === 2 };
    draw();
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY;
    if (!dx && !dy) return;
    if (drag.turn) { if (proj.st.canRotate) change({ tilt: -dy * 0.5, twist: dx * 0.5 }, false); return; }
    const ns = normalOnScreen();
    // along the normal as it is drawn; a normal that points at the viewer: up and down
    const step = ns && ns.len > 0.25 * proj.s ? (dx * ns.sx + dy * ns.sy) / (ns.len * ns.len * Math.max(1e-9, ns.extent))
      : -dy / Math.max(40, proj.H);
    change({ pos: step }, false);
  });
  const end = (e) => { if (!drag) return; drag = null; try { canvas.releasePointerCapture(e.pointerId); } catch (er) { /* gone */ } change({}, true); draw(); };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);
  canvas.addEventListener("wheel", (e) => { if (!proj) return; e.preventDefault(); change({ pos: e.deltaY < 0 ? 0.01 : -0.01, wheel: true }, true); }, { passive: false });
  canvas.addEventListener("keydown", (e) => {
    const k = e.key, big = e.shiftKey ? 5 : 1;
    if (k === "ArrowUp" || k === "ArrowDown") change({ pos: (k === "ArrowUp" ? 0.01 : -0.01) * big, key: true }, true);
    else if ((k === "ArrowLeft" || k === "ArrowRight") && proj && proj.st.canRotate) change({ twist: (k === "ArrowRight" ? 1 : -1) * big }, true);
    else if ((k === "PageUp" || k === "PageDown") && proj && proj.st.canRotate) change({ tilt: (k === "PageUp" ? 1 : -1) * big }, true);
    else return;
    e.preventDefault();
  });
  return { draw, get dragging() { return !!drag; }, normalOnScreen };
}
