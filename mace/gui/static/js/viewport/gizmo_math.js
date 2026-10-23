/* The geometry of the viewport's Rotate and Move gizmos (spec §2.1, 1.1.6):
 * pure functions, no DOM, so node runs them in the tests.
 *
 * Screen points are CSS px from the canvas's top left (v grows down); P is the
 * camera's 3x4 projection (geom/core.js dltCalibrate, P[11] = 1) and "toward"
 * the model direction that points at the viewer. Angles are in degrees; a
 * positive turn about an axis is counterclockwise seen from the axis's tip
 * (the right-hand rule). */
import * as G from "../geom/core.js";

/** Degrees per px of a free drag: the camera's own (viewport/camera.js), so the gizmo turns as an orbit does. */
export const DEG_PER_PX = 0.4;

/** The snap step of the drags: an Angle snap preset (15, 30, 45, 90), the custom angle, or 15°. */
export function snapStepOf(prefs) {
  if (prefs && prefs.step > 0) return prefs.step;
  return prefs && prefs.custom > 0 ? prefs.custom : 15;
}
export const snapTo = (a, step) => (step > 0 ? Math.round(a / step) * step : a);

/** Unit vectors [e1, e2] across the plane normal to n, with e1 × e2 = n̂ (a turn about n takes e1 toward e2). */
export function planeBasis(n) {
  const u = G.unit3(n);
  const ref = Math.abs(u[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const e1 = G.unit3(G.cross3(ref, u));
  return [e1, G.cross3(u, e1)];
}

/** Where the pointer (x, y) meets the plane o + a·e1 + b·e2, exactly under P (either projection): [a, b] (Å), or null edge-on. */
export function planeHit(P, o, e1, e2, x, y) {
  const r2 = [P[8], P[9], P[10]];
  const A = [P[0] - x * r2[0], P[1] - x * r2[1], P[2] - x * r2[2]];
  const B = [P[4] - y * r2[0], P[5] - y * r2[1], P[6] - y * r2[2]];
  const ra = x - P[3] - G.dot3(A, o), rb = y - P[7] - G.dot3(B, o);
  const a11 = G.dot3(A, e1), a12 = G.dot3(A, e2), a21 = G.dot3(B, e1), a22 = G.dot3(B, e2);
  const det = a11 * a22 - a12 * a21;
  const scale = Math.max(1e-12, Math.hypot(a11, a12) * Math.hypot(a21, a22));
  if (Math.abs(det) < 1e-9 * scale) return null;
  return [(ra * a22 - a12 * rb) / det, (a11 * rb - ra * a21) / det];
}

/** Screen point of a model point: [u, v, w]. */
export const screen = (P, p) => G.project(P, p[0], p[1], p[2]);

/** A ring of radius r (Å) about axis n through o, sampled N times: [{x, y, front}]; front: on the viewer's side of o. */
export function ringPoints(P, o, n, r, toward, N = 96) {
  const [e1, e2] = planeBasis(n), out = [];
  for (let k = 0; k <= N; k++) {
    const t = (2 * Math.PI * k) / N, off = G.add3(G.scale3(e1, r * Math.cos(t)), G.scale3(e2, r * Math.sin(t)));
    const s = screen(P, G.add3(o, off));
    out.push({ x: s[0], y: s[1], front: G.dot3(off, toward) >= -1e-9 * r });
  }
  return out;
}

/** Distance (px) from (x, y) to segment ab. */
export function segDist(x, y, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1], L = dx * dx + dy * dy;
  const t = L > 0 ? Math.max(0, Math.min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / L)) : 0;
  return Math.hypot(x - a[0] - t * dx, y - a[1] - t * dy);
}
/** Nearest distance from (x, y) to a ring's polyline; frontOnly skips the far half. */
export function ringDist(pts, x, y, frontOnly = false) {
  let best = Infinity;
  for (let k = 1; k < pts.length; k++) {
    const a = pts[k - 1], b = pts[k];
    if (frontOnly && !(a.front && b.front)) continue;
    best = Math.min(best, segDist(x, y, [a.x, a.y], [b.x, b.y]));
  }
  return best;
}

/** True when (x, y) is inside the parallelogram corner, corner+u, corner+u+v, corner+v (screen px). */
export function inQuad(x, y, quad) {
  let sign = 0;
  for (let k = 0; k < 4; k++) {
    const a = quad[k], b = quad[(k + 1) % 4];
    const c = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
    if (Math.abs(c) < 1e-12) continue;
    if (!sign) sign = Math.sign(c);
    else if (Math.sign(c) !== sign) return false;
  }
  return true;
}

/** The angle (deg) of the pointer about axis n through o, for a ring drag: exact on the ring's plane when it faces
 *  the viewer enough; edge-on, from the drag along the ring's screen tangent at the grab point (start = {x, y}). */
export function ringAngle(P, o, n, toward, x, y, start) {
  const nn = G.unit3(n);
  if (Math.abs(G.dot3(nn, toward)) > 0.25) {
    const [e1, e2] = planeBasis(nn), h = planeHit(P, o, e1, e2, x, y);
    if (h) return (Math.atan2(h[1], h[0]) * 180) / Math.PI;
  }
  // edge-on: a small turn dθ moves the grab point by dθ·(n × r): its screen image is the tangent (px per radian)
  const g = start.model;
  if (!g) return 0;
  const r = G.sub3(g, o), tip = G.add3(g, G.scale3(G.cross3(nn, r), 1e-3));
  const s0 = screen(P, g), s1 = screen(P, tip);
  const t = [(s1[0] - s0[0]) / 1e-3, (s1[1] - s0[1]) / 1e-3], L = t[0] * t[0] + t[1] * t[1];
  if (L < 1e-9) return 0;
  return (((x - start.x) * t[0] + (y - start.y) * t[1]) / L) * (180 / Math.PI);
}

/** The model point on the ring (radius r about n through o) nearest the pointer, for the edge-on fallback. */
export function ringGrabPoint(P, o, n, r, toward, x, y) {
  let best = null, bd = Infinity;
  const [e1, e2] = planeBasis(n);
  for (let k = 0; k < 96; k++) {
    const t = (2 * Math.PI * k) / 96, p = G.add3(o, G.add3(G.scale3(e1, r * Math.cos(t)), G.scale3(e2, r * Math.sin(t))));
    const s = screen(P, p), d = Math.hypot(s[0] - x, s[1] - y) - (G.dot3(G.sub3(p, o), toward) >= 0 ? 0.5 : 0);
    if (d < bd) { bd = d; best = p; }
  }
  return best;
}

/** The angle (deg) of a screen point about a screen center, counterclockwise as the viewer sees it (v grows down). */
export const screenAngle = (cx, cy, x, y) => (-Math.atan2(y - cy, x - cx) * 180) / Math.PI;
/** a1 - a0 brought to the turn nearest prev (so a drag can go past ±180°). */
export function unwrap(prev, a0, a1) {
  let d = a1 - a0;
  while (d - prev > 180) d -= 360;
  while (d - prev < -180) d += 360;
  return d;
}

/** The displacement (Å) along unit axis e of a drag from (x0, y0) to (x, y), the axis through o: exact, on the plane
 *  that holds the axis and faces the viewer most; along the view direction, from the axis's screen image. */
export function axisDrag(P, o, e, toward, x0, y0, x, y) {
  const u = G.unit3(e);
  const nrm = G.sub3(toward, G.scale3(u, G.dot3(toward, u)));
  if (G.norm3(nrm) > 0.05) {
    const w = G.unit3(G.cross3(G.unit3(nrm), u));
    const h0 = planeHit(P, o, u, w, x0, y0), h1 = planeHit(P, o, u, w, x, y);
    if (h0 && h1) return h1[0] - h0[0];
  }
  const s0 = screen(P, o), s1 = screen(P, G.add3(o, u));
  const t = [s1[0] - s0[0], s1[1] - s0[1]], L = t[0] * t[0] + t[1] * t[1];
  return L > 1e-9 ? ((x - x0) * t[0] + (y - y0) * t[1]) / L : 0;
}
/** The displacement (Å) in the plane o + a·e1 + b·e2 of a drag from (x0, y0) to (x, y); edge-on, null. */
export function planeDrag(P, o, e1, e2, x0, y0, x, y) {
  const h0 = planeHit(P, o, e1, e2, x0, y0), h1 = planeHit(P, o, e1, e2, x, y);
  if (!h0 || !h1) return null;
  return G.add3(G.scale3(e1, h1[0] - h0[0]), G.scale3(e2, h1[1] - h0[1]));
}

/** The angles a rotate drag applies, from the raw ones (deg, from the reference orientation):
 *   snap: the drag snaps (the Angle snap setting, inverted while Ctrl is held);
 *   snapAll (Shift): every angle snaps to the step, whatever the mode (88°, 43° -> 90°, 45°);
 *   isolate (Alt): only one angle is kept, snapped: the largest, or with pick (Tab) the next (-> 90°, 0° or 0°, 45°).
 *  Returns {degs, kept} (kept: the index isolate keeps, else -1). */
export function resolveAngles(raw, { snap = false, step = 15, snapAll = false, isolate = false, pick = 0 } = {}) {
  if (isolate && raw.length) {
    const order = raw.map((_, i) => i).sort((i, j) => Math.abs(raw[j]) - Math.abs(raw[i]) || i - j);
    const keep = order[((pick % raw.length) + raw.length) % raw.length];
    return { degs: raw.map((d, i) => (i === keep ? snapTo(d, step) : 0)), kept: keep };
  }
  return { degs: raw.map((d) => (snap || snapAll ? snapTo(d, step) : d)), kept: -1 };
}

/** The model-space turn of a list of {axis (model), deg}: the first applied first. */
export function composeTurn(comps) {
  let q = [0, 0, 0, 1];
  for (const c of comps) if (c.deg) q = G.qmul(G.qaxis(c.axis, c.deg), q);
  return q;
}
/** Points turned by the model-space quaternion qm about pivot. */
export function turnAbout(points, pivot, qm) {
  return points.map((p) => G.add3(pivot, G.qrotate(qm, G.sub3(p, pivot))));
}
/** The view quaternion after turning the model by qm (model space) in front of a camera at q0. */
export const turnedView = (q0, qm) => G.qmul(q0, qm);

/** An angle and the same turn the other way round: "+30.0°", "−330.0°" (0: "0.0°", "0.0°"). */
export function bothWays(deg, digits = 1) {
  const a = ((deg % 360) + 360) % 360, eps = 0.5 * Math.pow(10, -digits);
  const fmt = (x) => `${x < -eps ? "−" : x > eps ? "+" : ""}${Math.abs(x).toFixed(digits)}°`;
  if (a < eps || 360 - a < eps) return [fmt(0), fmt(0)];
  return deg >= 0 ? [fmt(a), fmt(a - 360)] : [fmt(a - 360), fmt(a)];
}

/** A rotation axis typed or picked: x, y, z; a, b, c, a*, b*, c* (needs a cell); "view" (toward the viewer, at q);
 *  "h" / "v" (the screen's vertical and horizontal axes at q: a horizontal / vertical turn); or three numbers:
 *  a lattice direction [u v w] with a cell, a Cartesian vector without one. Returns a unit vector or null. */
export function axisVector(key, cell, q) {
  const k = String(key || "").trim();
  const qi = q ? G.qconj(q) : [0, 0, 0, 1];
  if (k === "x" || k === "y" || k === "z") return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]["xyz".indexOf(k)];
  if (k === "view") return G.qrotate(qi, [0, 0, 1]);
  if (k === "h") return G.qrotate(qi, [0, 1, 0]);
  if (k === "v") return G.qrotate(qi, [1, 0, 0]);
  const usable = cell && cell.every((r) => G.norm3(r) > 1e-9) && Math.abs(G.dot3(cell[0], G.cross3(cell[1], cell[2]))) > 1e-9;
  if (/^[abc]\*?$/.test(k)) {
    if (!usable) return null;
    try { return G.alongDirection(cell, k); } catch (e) { return null; }
  }
  const t = k.replace(/^\[|\]$/g, "");
  let v = G.parseTriple(t);
  // "110", "1-10": one digit per index, the crystallographer's shorthand
  if (!v && /^(-?\d){3}$/.test(t.replace(/\s/g, ""))) v = t.replace(/\s/g, "").match(/-?\d/g).map(Number);
  if (!v || G.norm3(v) < 1e-12) return null;
  return usable ? G.alongDirection(cell, v) : G.unit3(v);
}

/** The selected atoms of structure s ({frac, cell, pbc}), made whole across the cell's faces: each placed at its
 *  image nearest an atom already placed (periodic axes only; above 400 atoms, nearest the first), so a molecule
 *  the boundary splits moves and turns as one piece. {sel (sorted), frac, cart, pivot (the centroid)}. */
export function wholeSelection(s, selection) {
  const sel = [...selection].sort((a, b) => a - b), pbc = s.pbc || [true, true, true];
  if (!sel.length) return { sel, frac: [], cart: [], pivot: [0, 0, 0] };
  const out = [s.frac[sel[0]].slice()];
  for (let k = 1; k < sel.length; k++) {
    const f = s.frac[sel[k]];
    let best = null, bl = Infinity;
    const refs = out.length > 400 ? [out[0]] : out;
    for (const r of refs) {
      const d = f.map((x, m) => (pbc[m] ? x - r[m] - Math.round(x - r[m]) : x - r[m]));
      const v = G.cartOf(s.cell, d), L = G.dot3(v, v);
      if (L < bl) { bl = L; best = r.map((x, m) => x + d[m]); }
    }
    out.push(best);
  }
  const cart = out.map((f) => G.cartOf(s.cell, f));
  const pivot = cart.reduce((t, p) => G.add3(t, p), [0, 0, 0]).map((x) => x / cart.length);
  return { sel, frac: out, cart, pivot };
}

/* ---------------------------------------------------------------- the pivot of a turn of the atoms (1.1.6) */
/** The Rotate panel's Pivot choices (Settings › Viewport › Rotate pivot; a document keeps its own, with
 *  "atom:N", a picked atom, 1-based): mace transform --about takes the same names. */
export const PIVOTS = [["centroid", "Selection centroid"], ["com", "Center of mass"], ["cell", "Cell center"], ["origin", "Origin"]];
/** A stored pivot (settings.js and settings.py ROTATE_PIVOT agree on this form). */
export const PIVOT_RE = /^(centroid|com|cell|origin|atom:[1-9][0-9]{0,6})$/;
/** {mode, atom (0-based, mode "atom")} of a stored pivot; anything else reads as the centroid. */
export function parsePivot(v) {
  const t = String(v || "");
  if (!PIVOT_RE.test(t)) return { mode: "centroid", atom: -1 };
  if (t.startsWith("atom:")) return { mode: "atom", atom: Number(t.slice(5)) - 1 };
  return { mode: t, atom: -1 };
}
/** An atom's name: its symbol and 1-based index ("Ti1"), as the viewer's Labels › Both. */
export function atomName(s, i) {
  const sym = (s.symbols && s.symbols[i]) || G.SYMBOLS[(s.numbers && s.numbers[i]) % 100] || "X";
  return `${sym}${i + 1}`;
}
/** The pivot (Å) of a turn of the selection w (wholeSelection of structure s), and its name:
 *   centroid  the selection's centroid (made whole across the cell's faces);
 *   com       its center of mass (the same images, standard atomic masses: G.ATOMIC_MASS);
 *   cell      the middle of the cell (½, ½, ½); along a slab's, polymer's or molecule's non-periodic axes, the
 *             middle of the drawing box under vacuum (viewport/vacuum.js of; default the bounding box);
 *   origin    (0, 0, 0);
 *   atom:N    atom N (1-based; it need not be selected): where the selection has it, else its image nearest the
 *             selection's centroid (periodic axes only).
 *  {point, mode, atom, name (undo step: "Centroid", "Ti1"), bad (atom:N out of range: the centroid instead)}.
 *  structure_transform.py pivot_point is the same rule. */
export function pivotPoint(s, w, pivot, vacuum = null) {
  const p = parsePivot(pivot), n = (s.frac || s.cart || []).length;
  const centroid = { point: w.pivot.slice(), mode: "centroid", atom: -1, name: "Centroid" };
  if (p.mode === "com") {
    let m = 0, t = [0, 0, 0];
    w.sel.forEach((i, k) => {
      const z = (s.numbers && s.numbers[i]) % 100, mi = G.ATOMIC_MASS[z] || 1;
      m += mi; t = G.add3(t, G.scale3(w.cart[k], mi));
    });
    return m > 0 ? { point: G.scale3(t, 1 / m), mode: "com", atom: -1, name: "Center of Mass" } : centroid;
  }
  if (p.mode === "cell") {
    if (!s.cell) return { ...centroid, bad: true };
    const c = G.boxCorners(s, { lo: [0, 0, 0], hi: [1, 1, 1], vacuum: vacuum || { mode: "bbox" } });
    return { point: G.scale3(G.add3(c[0], c[7]), 0.5), mode: "cell", atom: -1, name: "Cell Center" };
  }
  if (p.mode === "origin") return { point: [0, 0, 0], mode: "origin", atom: -1, name: "Origin" };
  if (p.mode === "atom") {
    const i = p.atom;
    if (!(i >= 0 && i < n)) return { ...centroid, bad: true };
    const k = w.sel.indexOf(i);
    let point;
    if (k >= 0) point = w.cart[k].slice();
    else if (s.frac && s.cell && w.frac.length) {
      const pbc = s.pbc || [true, true, true];
      const fc = w.frac.reduce((t, f) => G.add3(t, f), [0, 0, 0]).map((x) => x / w.frac.length);
      const d = s.frac[i].map((x, m) => (pbc[m] ? x - fc[m] - Math.floor(x - fc[m] + 0.5) : x - fc[m]));
      point = G.cartOf(s.cell, G.add3(fc, d));
    } else point = (s.cart && s.cart[i] ? s.cart[i] : G.cartOf(s.cell, s.frac[i])).slice();
    return { point, mode: "atom", atom: i, name: atomName(s, i) };
  }
  return centroid;
}
