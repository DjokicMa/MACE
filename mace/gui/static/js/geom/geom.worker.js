/* The geometry worker (spec §8.3.1): bonds on a cell-list grid, box and
 * lasso hit tests from the camera matrix the page sends, distance shells,
 * bonded neighbors and fragments, and measurements, off the main thread.
 *
 * Every job carries an id and a kind. Jobs that arrive while others wait are
 * looked at together: of several of one kind only the newest runs, and the
 * older ones answer {superseded: true} (a drag sends one box per frame).
 * Results are typed arrays, transferred, not copied. */
import * as G from "./core.js";

const queue = [];
let scheduled = false;

self.onmessage = (e) => {
  queue.push(e.data);
  if (!scheduled) { scheduled = true; setTimeout(pump, 0); }
};

function pump() {
  scheduled = false;
  const jobs = queue.splice(0);
  const newest = new Map();
  for (const j of jobs) newest.set(j.kind, j);
  for (const j of jobs) {
    if (newest.get(j.kind) !== j) { self.postMessage({ id: j.id, kind: j.kind, superseded: true }); continue; }
    const t0 = performance.now();
    try {
      const { result, transfer } = run(j.kind, j.args);
      self.postMessage({ id: j.id, kind: j.kind, result, ms: performance.now() - t0 }, transfer);
    } catch (err) {
      self.postMessage({ id: j.id, kind: j.kind, error: String(err && err.message || err) });
    }
  }
}

function run(kind, a) {
  if (kind === "bonds") {
    const pairs = G.cellListBonds(a.cart, a.radii, a.metal, a.tol, a.metalMetal);
    return { result: { pairs }, transfer: [pairs.buffer] };
  }
  if (kind === "periodicBonds" || kind === "neighbors" || kind === "fragment") {
    const b = G.periodicBonds(a.cell, a.frac, a.radii, a.metal, a.tol, a.metalMetal);
    if (kind === "periodicBonds") return { result: b, transfer: [b.i.buffer, b.j.buffer, b.img.buffer] };
    const adj = G.adjacency(b, a.radii.length), sel = new Set(a.selection);
    const out = Uint32Array.from(kind === "neighbors" ? G.neighborsOf(adj, sel) : G.fragmentOf(adj, sel));
    return { result: { indices: out }, transfer: [out.buffer] };
  }
  if (kind === "coordination") {
    const rows = G.coordination(a.cell, a.frac, a.z, a.radii, a.metal, a.tol, a.sites);
    return { result: { rows }, transfer: [] };
  }
  if (kind === "polyhedra") {
    const r = G.polyhedra(a.cell, a.frac, a.z, a.radii, a.metal, a.tol, a.dispCart, a.dispBase, a.only);
    return { result: r, transfer: r.meshes.flatMap((m) => [m.pos.buffer, m.nrm.buffer, m.edge.buffer]) };
  }
  if (kind === "boxSelect" || kind === "lassoSelect") {
    const uv = G.projectAll(a.P, a.cart);
    const hits = kind === "boxSelect" ? G.boxSelect(uv, a.rect) : G.lassoSelect(uv, a.poly);
    return { result: { hits }, transfer: [hits.buffer] };
  }
  if (kind === "pick") {
    // the front-most atom whose drawn disc holds the point (large structures)
    const uv = G.projectAll(a.P, a.cart);
    let best = -1, bestW = Infinity;
    for (let i = 0; i < uv.length / 2; i++) {
      const du = uv[2 * i] - a.x, dv = uv[2 * i + 1] - a.y, r = a.rpx[i];
      if (du * du + dv * dv > r * r) continue;
      const x = a.cart[3 * i], y = a.cart[3 * i + 1], z = a.cart[3 * i + 2];
      const w = a.depth[0] * x + a.depth[1] * y + a.depth[2] * z;
      if (w < bestW) { bestW = w; best = i; }
    }
    return { result: { index: best }, transfer: [] };
  }
  if (kind === "within") {
    // atoms within r of each selected atom (or of their centroid), periodic images included
    const n = a.frac.length / 3, C = a.cell, out = new Set();
    const fr = (i) => [a.frac[3 * i], a.frac[3 * i + 1], a.frac[3 * i + 2]];
    let centers = a.selection.map(fr);
    if (a.centroid && centers.length) centers = [centers.reduce((s, f) => G.add3(s, f), [0, 0, 0]).map((x) => x / centers.length)];
    for (const c of centers) for (let j = 0; j < n; j++) {
      const d = a.periodic ? G.minImage(C, c, fr(j)).length : G.norm3(G.cartOf(C, G.sub3(fr(j), c)));
      if (d <= a.r) out.add(j);
    }
    const indices = Uint32Array.from(out);
    return { result: { indices }, transfer: [indices.buffer] };
  }
  if (kind === "measure") {
    const values = Float64Array.from(a.items.map((atoms) => G.measure(a.cell, a.frac, atoms)));
    return { result: { values }, transfer: [values.buffer] };
  }
  throw new Error(`no job ${kind}`);
}
