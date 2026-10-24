/* Periodic overlays over the display range (1.1.6).
 *
 * The atoms and bonds are drawn over the display range (the N × M × L cells picker, the display-range
 * presets), and so are a periodic cube's isosurfaces (periodic_surface.js); the Structure Analysis
 * overlays of one cell (pores, Miller planes, a net) were drawn in that one cell only. They are drawn in
 * each cell of the range's cell box now: the overlay's mesh is built once, for the unit cell, and each
 * other cell draws the same built mesh moved by its lattice vector (a clone sharing its geometry and
 * material, as 3Dmol draws a model's symmetry copies), so 27 cells cost one build. The non-periodic axes
 * of a slab or a polymer have no copies. */

/** The most cells an overlay is drawn in (a range beyond it: the first ones, and the Inspector says so). */
export const TILES_MAX = 64;

const rowsOf = (c) => (c && c.length === 9 ? [[c[0], c[1], c[2]], [c[3], c[4], c[5]], [c[6], c[7], c[8]]] : c);

/** The cells (integer triples) of a display box ({lo, hi}, whole cells, as display_state.boxOf draws the cell box):
 * lo..hi-1 along each periodic axis, the cell itself along a non-periodic one; no box: the cell. */
export function tileCells(box, pbc) {
  const per = pbc || [true, true, true];
  const lo = [0, 1, 2].map((k) => (per[k] && box && box.lo ? Math.round(Number(box.lo[k]) || 0) : 0));
  const hi = [0, 1, 2].map((k) => (per[k] && box && box.hi ? Math.max(lo[k] + 1, Math.round(Number(box.hi[k]) || 1)) : lo[k] + 1));
  const out = [];
  for (let a = lo[0]; a < hi[0]; a++) for (let b = lo[1]; b < hi[1]; b++) for (let c = lo[2]; c < hi[2]; c++) out.push([a, b, c]);
  return out;
}

/** The cells an overlay is drawn in and their Cartesian shifts (Å, rows of ``cell``): {cells, shifts, total, capped}. */
export function tileShifts(cell, pbc, box, max = TILES_MAX) {
  const C = rowsOf(cell), all = C ? tileCells(box, pbc) : [[0, 0, 0]];
  const cells = all.slice(0, Math.max(1, max));
  const shifts = cells.map((i) => (C ? [0, 1, 2].map((j) => i[0] * C[0][j] + i[1] * C[1][j] + i[2] * C[2][j]) : [0, 0, 0]));
  return { cells, shifts, total: all.length, capped: all.length > cells.length };
}

/** The same tiles without the ones a polygon on a far face of the cell would double: a plane at fractional 1
 * along a periodic axis is the next cell's plane at 0, so it is drawn in the last cell of the range only. */
export function farFaceTiles(tiles, frac, pbc, tol = 1e-6) {
  const per = pbc || [true, true, true];
  const far = [0, 1, 2].filter((k) => per[k] && frac.length && frac.every((f) => Math.abs(f[k] - 1) < tol));
  if (!far.length) return tiles;
  const last = far.map((k) => Math.max(...tiles.cells.map((c) => c[k])));
  const keep = tiles.cells.map((c) => far.every((k, i) => c[k] === last[i]));
  return { ...tiles, cells: tiles.cells.filter((_, i) => keep[i]), shifts: tiles.shifts.filter((_, i) => keep[i]) };
}

/** The tiles one shape draws in: set before its first build (shape_cache.keepBuilt adds the copies). Its bounding
 * sphere grows to hold every copy (the slab and fog of clip.js, zoomTo). */
export function tile(s, shifts) {
  if (!s || s.maceTiles || !shifts || !(shifts.length > 1 || shifts.some((t) => t[0] || t[1] || t[2]))) return s;
  s.maceTiles = shifts;
  const bs = s.boundingSphere;
  // a custom shape's is left empty by 3Dmol: the box of its vertices
  if (bs && bs.center && !(bs.radius > 0) && s.geo && s.geo.geometryGroups) {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const g of s.geo.geometryGroups) {
      const A = g.vertexArray;
      for (let i = 0; A && i < g.vertices; i++) for (let k = 0; k < 3; k++) { const x = A[3 * i + k]; if (x < lo[k]) lo[k] = x; if (x > hi[k]) hi[k] = x; }
    }
    if (hi[0] >= lo[0]) {
      bs.center.set((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2);
      bs.radius = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2;
    }
  }
  if (bs && bs.center && bs.radius > 0) {
    const m = [0, 1, 2].map((k) => shifts.reduce((a, t) => a + t[k], 0) / shifts.length);
    const far = Math.max(...shifts.map((t) => Math.hypot(t[0] - m[0], t[1] - m[1], t[2] - m[2])));
    bs.center.set(bs.center.x + m[0], bs.center.y + m[1], bs.center.z + m[2]);
    bs.radius += far;
  }
  return s;
}

/** A built shape's 3D object drawn at every shift: its meshes (and lines) moved to the first shift, and a copy
 * of each, sharing its geometry and material, at every other. */
export function cloneTiles(obj, shifts) {
  if (!obj || !shifts || !shifts.length) return obj;
  const base = (obj.children || []).slice();
  const place = (m, t) => {
    m.matrix.setPosition({ x: t[0], y: t[1], z: t[2] });
    m.matrixAutoUpdate = false;
    m.matrixWorldNeedsUpdate = true;
    m.maceTile = t;
  };
  base.forEach((m) => place(m, shifts[0]));
  for (let i = 1; i < shifts.length; i++) {
    for (const m of base) {
      const c = m.clone();
      place(c, shifts[i]);
      obj.add(c);
    }
  }
  return obj;
}
