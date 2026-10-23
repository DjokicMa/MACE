/* Isosurfaces that join across the cell's faces (1.1.6).
 *
 * A periodic cube's surface is drawn once per cell of the display range. 3Dmol's addIsosurface
 * marched one cell's grid and smoothed it as an open mesh, so every cell's surface stopped short
 * of the cell's faces (the Laplacian smoothing pulled its border inward, and the cells were set a
 * voxel too far apart): a gap at every cell boundary of a supercell.
 *
 * gui/volume.py sends a periodic axis as whole voxels of one period plus the closing plane (meta.wrap).
 * Here that grid is padded with the next cells' planes along each wrapping axis, marched and smoothed
 * with 3Dmol's own MarchingCube, and cut back to the cell: a triangle belongs to the cell its
 * unsmoothed centroid lies in ([0, period) on each wrapping axis, integers: exact), so the copies of
 * a triangle in two neighboring cells are kept once, and the border vertices of neighbors coincide.
 * The padding is wider than the smoothing reaches, so the cut mesh is the same as one smoothed over
 * the whole crystal. The mesh is made once per level in the grid's index space; each cell draws it
 * moved by its lattice vector (no march per cell).
 */

/** Planes added on each side of a wrapping axis: the smoothing's reach (one voxel a pass) and the
 * border's own ring. */
export const margin = (smoothness) => Math.max(0, Math.round(Number(smoothness) || 0)) + 3;

/** The cell's mesh in the grid's index space: ``pos`` (x, y, z per vertex; the grid's first point
 * is 0) and ``faces`` (3 vertex indices per triangle). ``shape``: the grid's points per axis;
 * ``wrap``: the axes that hold one period plus the closing plane; ``MC``: 3Dmol's MarchingCube. */
export function periodicMesh(data, shape, wrap, iso, smoothness, MC) {
  const per = [0, 1, 2].map((k) => (wrap[k] ? shape[k] - 1 : shape[k]));
  const pad = [0, 1, 2].map((k) => (wrap[k] ? margin(smoothness) : 0));
  const N = [0, 1, 2].map((k) => (wrap[k] ? per[k] + 2 * pad[k] + 1 : shape[k]));
  // the grid point each padded point copies: a wrapping axis' index modulo its period
  const src = [0, 1, 2].map((k) => Int32Array.from({ length: N[k] }, (_, j) => (wrap[k] ? (((j - pad[k]) % per[k]) + per[k]) % per[k] : j)));
  const [nx, ny, nz] = shape, [X, Y, Z] = N, ISDONE = MC.ISDONE || 2;
  const bits = new Uint8Array(X * Y * Z);
  // inside: above a positive level, below a negative one (3Dmol's rule)
  for (let i = 0, q = 0; i < X; i++) {
    const si = src[0][i] * ny;
    for (let j = 0; j < Y; j++) {
      const sij = (si + src[1][j]) * nz;
      for (let k = 0; k < Z; k++, q++) {
        const v = data[sij + src[2][k]];
        if ((iso >= 0 ? v - iso : iso - v) > 0) bits[q] = ISDONE;
      }
    }
  }
  const verts = [], faces = [];
  MC.march(bits, verts, faces, { fulltable: true, voxel: false, unitCube: { x: 1, y: 1, z: 1 }, origin: { x: 0, y: 0, z: 0 }, nX: X, nY: Y, nZ: Z });
  const n = verts.length;
  // the grid points the vertices sit on before the smoothing moves them (the cut's exact positions)
  const raw = new Int32Array(3 * n);
  for (let i = 0; i < n; i++) { raw[3 * i] = verts[i].x; raw[3 * i + 1] = verts[i].y; raw[3 * i + 2] = verts[i].z; }
  if (smoothness > 0) MC.laplacianSmooth(smoothness, verts, faces);
  const lo = pad.map((p) => 3 * p), hi = [0, 1, 2].map((k) => 3 * (pad[k] + per[k]));
  const keep = [], map = new Int32Array(n).fill(-1);
  let m = 0;
  for (let f = 0; f + 2 < faces.length; f += 3) {
    const a = faces[f], b = faces[f + 1], c = faces[f + 2];
    let ok = true;
    for (let k = 0; k < 3 && ok; k++) {
      if (!wrap[k]) continue;
      const s = raw[3 * a + k] + raw[3 * b + k] + raw[3 * c + k];
      ok = s >= lo[k] && s < hi[k];
    }
    if (!ok) continue;
    for (const v of [a, b, c]) {
      if (map[v] < 0) map[v] = m++;
      keep.push(map[v]);
    }
  }
  const pos = new Float64Array(3 * m);
  for (let i = 0; i < n; i++) {
    const t = map[i];
    if (t < 0) continue;
    pos[3 * t] = verts[i].x - pad[0]; pos[3 * t + 1] = verts[i].y - pad[1]; pos[3 * t + 2] = verts[i].z - pad[2];
  }
  return { pos, faces: Uint32Array.from(keep) };
}

/** The mesh's vertices in Cartesian space (Å): the grid's origin ``o`` plus the index times its voxel
 * vectors ``V`` (rows), as 3Dmol's VolumeData matrix places them. */
export function placed(mesh, o, V) {
  const P = mesh.pos, n = P.length / 3, out = new Array(n);
  for (let i = 0; i < n; i++) {
    const a = P[3 * i], b = P[3 * i + 1], c = P[3 * i + 2];
    out[i] = { x: o[0] + a * V[0][0] + b * V[1][0] + c * V[2][0], y: o[1] + a * V[0][1] + b * V[1][1] + c * V[2][1],
      z: o[2] + a * V[0][2] + b * V[1][2] + c * V[2][2] };
  }
  return out;
}

/** A volume to color a cut mesh by (the potential of Color by Potential), read across the cell's faces: the
 * cut mesh reaches up to a voxel past its cell, where 3Dmol's VolumeData.getVal answers 0 (a pale seam).
 * The grid point nearest a position (as getVal), its index wrapped along ``meta.wrap``. */
export function wrappedVolume(vd) {
  const m = vd.__meta || {}, wrap = m.wrap || [false, false, false];
  if (!wrap.some(Boolean)) return vd;
  const V = m.voxels, o = m.origin, n = m.shape, [a, b, c] = V;
  const det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
  // r = g V (rows): g = r V⁻¹
  const Vi = [[b[1] * c[2] - b[2] * c[1], -(a[1] * c[2] - a[2] * c[1]), a[1] * b[2] - a[2] * b[1]],
    [-(b[0] * c[2] - b[2] * c[0]), a[0] * c[2] - a[2] * c[0], -(a[0] * b[2] - a[2] * b[0])],
    [b[0] * c[1] - b[1] * c[0], -(a[0] * c[1] - a[1] * c[0]), a[0] * b[1] - a[1] * b[0]]].map((r) => r.map((x) => x / det));
  return {
    getVal(x, y, z) {
      const r = [x - o[0], y - o[1], z - o[2]], g = [0, 0, 0];
      for (let k = 0; k < 3; k++) {
        let i = Math.round(r[0] * Vi[0][k] + r[1] * Vi[1][k] + r[2] * Vi[2][k]);
        if (wrap[k]) { const p = n[k] - 1; i = ((i % p) + p) % p; }
        if (i < 0 || i >= n[k]) return 0;
        g[k] = i;
      }
      return vd.data[(g[0] * n[1] + g[1]) * n[2] + g[2]];
    },
  };
}

/** A 3Dmol shape of the mesh moved to the grid origin ``o`` (``spec``: as addIsosurface takes it, the
 * potential's voldata colors too), or null when the level has no surface here. */
export function addMeshShape(vw, mesh, o, V, spec) {
  if (!mesh.faces.length) return null;
  const vertexArr = placed(mesh, o, V);
  const sh = vw.addShape(spec);
  sh.addCustom({ vertexArr, faceArr: Array.from(mesh.faces), normalArr: [] });
  sh.updateStyle(spec);                  // as addIsosurface: the potential's colors on the vertices
  // the bounding sphere zoomTo fits (addIsosurface sets it the same way: custom shapes leave it empty)
  const c = { x: 0, y: 0, z: 0 };
  for (const p of vertexArr) { c.x += p.x; c.y += p.y; c.z += p.z; }
  c.x /= vertexArr.length; c.y /= vertexArr.length; c.z /= vertexArr.length;
  let r = 0;
  for (const p of vertexArr) r = Math.max(r, (p.x - c.x) ** 2 + (p.y - c.y) ** 2 + (p.z - c.z) ** 2);
  sh.boundingSphere.center.set(c.x, c.y, c.z);
  sh.boundingSphere.radius = Math.sqrt(r);
  return sh;
}
