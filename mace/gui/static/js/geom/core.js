/* Pure geometry for the viewport and the editor (spec §8.3): typed-array
 * structures, frac <-> Cartesian, the reciprocal cell, minimum-image
 * measurements, O(N) cell-list bonds, symmetry orbits, the camera projection
 * used for picking, level of detail, view directions and the expression
 * parser of the coordinate fields. No DOM: the geometry worker imports this
 * module, and the main thread calls the same functions for small structures. */
export { fracToCart, cartToFrac, parseTriple } from "../rank.js";

/* Element data: ase.data.covalent_radii (Cordero et al. 2008; 0.2 Å for
 * Z = 0 and above Cm, as ase has it) and the symbols, by atomic number. */
export const SYMBOLS = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge " +
  "As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm " +
  "Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr").split(" ");
export const COVALENT = Float32Array.from([0.2, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, 1.66, 1.41,
  1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03, 1.76, 1.7, 1.6, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, 1.22, 1.2,
  1.19, 1.2, 1.2, 1.16, 2.2, 1.95, 1.9, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38,
  1.39, 1.4, 2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.9, 1.87, 1.87,
  1.75, 1.7, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.4, 1.5, 1.5, 2.6, 2.21, 2.15, 2.06, 2,
  1.96, 1.9, 1.87, 1.8, 1.69, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]);
/* Everything not in this set counts as a metal for drawing bonds (the
 * server's structures._NONMETALS): metal-metal pairs get none, as VESTA. */
const NONMETALS = new Set([1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 32, 33, 34, 35, 36, 51, 52, 53, 54, 85, 86]);
export const isMetal = (z) => !NONMETALS.has(z);
const Z_OF = new Map(SYMBOLS.map((s, z) => [s.toLowerCase(), z]));
/** "fe", "Fe" -> 26; anything else -> 0. */
export function zOf(sym) {
  const z = Z_OF.get(String(sym || "").trim().toLowerCase());
  return z && z <= 103 ? z : 0;
}

/* ---------------------------------------------------------------- structures */
/** A /api/structure or edit payload as typed arrays. */
export function fromPayload(s) {
  const n = s.numbers.length;
  const t = { n, cell: new Float64Array(9), frac: new Float64Array(3 * n), cart: new Float64Array(3 * n),
    z: new Uint8Array(n), radii: new Float32Array(n), metal: new Uint8Array(n) };
  for (let k = 0; k < 3; k++) for (let m = 0; m < 3; m++) t.cell[3 * k + m] = s.cell[k][m];
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) { t.frac[3 * i + k] = s.frac[i][k]; t.cart[3 * i + k] = s.cart[i][k]; }
    const z = s.numbers[i] % 100 || s.numbers[i];
    t.z[i] = z;
    t.radii[i] = s.radii ? s.radii[i] : COVALENT[z] || 0.2;
    t.metal[i] = s.metal ? (s.metal[i] ? 1 : 0) : (isMetal(z) ? 1 : 0);
  }
  return t;
}

/** base64 -> bytes. */
export function b64bytes(text) {
  const s = atob(text), out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}
/** A packed payload (spec §8.4.5: cell, z and frac as base64 typed arrays)
 * as the usual structure payload: the typed arrays are decoded exactly
 * (little-endian Float64), and symbols, radii, metal and the Cartesian
 * positions are computed here. Unpacked payloads come back as they are. */
export function decodePacked(p) {
  if (!p || !p.packed) return p;
  const n = p.n;
  const zb = p.zn16 ? new Uint16Array(b64bytes(p.zn16).buffer) : b64bytes(p.zn);
  const fb = new Float64Array(b64bytes(p.frac).buffer);
  const cell = [[p.cell[0], p.cell[1], p.cell[2]], [p.cell[3], p.cell[4], p.cell[5]], [p.cell[6], p.cell[7], p.cell[8]]];
  const cb = p.cart ? new Float64Array(b64bytes(p.cart).buffer) : null;
  const numbers = new Array(n), frac = new Array(n), cart = new Array(n), symbols = new Array(n), radii = new Array(n), metal = new Array(n);
  for (let i = 0; i < n; i++) {
    const z = zb[i], zz = z % 100 || z;
    numbers[i] = z;
    const f = [fb[3 * i], fb[3 * i + 1], fb[3 * i + 2]];
    frac[i] = f;
    cart[i] = cb ? [cb[3 * i], cb[3 * i + 1], cb[3 * i + 2]]
      : [f[0] * cell[0][0] + f[1] * cell[1][0] + f[2] * cell[2][0], f[0] * cell[0][1] + f[1] * cell[1][1] + f[2] * cell[2][1],
        f[0] * cell[0][2] + f[1] * cell[1][2] + f[2] * cell[2][2]];
    symbols[i] = SYMBOLS[zz] || "X";
    radii[i] = COVALENT[zz] || 0.2;
    metal[i] = isMetal(zz);
  }
  const out = { ...p, cell, numbers, frac, cart, symbols, radii, metal };
  delete out.packed; delete out.zn; delete out.zn16; delete out.n;
  return out;
}

/** The payload fields an edit changes, recomputed from cell, frac and numbers
 * (what the browser draws before the server has answered). */
export function payloadFrom(cell, frac, numbers, base = {}) {
  const C = cell.map((r) => r.slice());
  const cart = frac.map((f) => [0, 1, 2].map((k) => f[0] * C[0][k] + f[1] * C[1][k] + f[2] * C[2][k]));
  const zs = numbers.map((z) => z % 100 || z);
  return { ...base, cell: C, frac: frac.map((f) => f.slice()), cart, numbers: numbers.slice(),
    symbols: zs.map((z) => SYMBOLS[z] || "X"), radii: zs.map((z) => COVALENT[z] || 0.2),
    metal: zs.map((z) => isMetal(z)), lattice: latticeOf(C) };
}

/* ---------------------------------------------------------------- linear algebra */
export const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm3 = (a) => Math.sqrt(dot3(a, a));
export const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add3 = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale3 = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const unit3 = (a) => { const n = norm3(a) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };
const rows = (cell) => (cell.length === 9 ? [[cell[0], cell[1], cell[2]], [cell[3], cell[4], cell[5]], [cell[6], cell[7], cell[8]]] : cell);

/** The inverse of a 3x3 matrix (rows), or null when singular. */
export function inv3(m) {
  const [a, b, c] = rows(m);
  const det = dot3(a, cross3(b, c));
  if (Math.abs(det) < 1e-12) return null;
  const r0 = cross3(b, c), r1 = cross3(c, a), r2 = cross3(a, b);
  // inverse = [r0 r1 r2]^T / det (columns)
  return [[r0[0] / det, r1[0] / det, r2[0] / det], [r0[1] / det, r1[1] / det, r2[1] / det], [r0[2] / det, r1[2] / det, r2[2] / det]];
}
/** Reciprocal cell rows a*, b*, c* (no 2π): a*·a = 1, a*·b = 0, ... */
export function reciprocal(cell) {
  const inv = inv3(cell);
  return inv ? [[inv[0][0], inv[1][0], inv[2][0]], [inv[0][1], inv[1][1], inv[2][1]], [inv[0][2], inv[1][2], inv[2][2]]] : null;
}
export function cartOf(cell, f) {
  const C = rows(cell);
  return [0, 1, 2].map((k) => f[0] * C[0][k] + f[1] * C[1][k] + f[2] * C[2][k]);
}
export function fracOf(cell, x) {
  const R = reciprocal(cell);
  return [dot3(R[0], x), dot3(R[1], x), dot3(R[2], x)];
}
export function latticeOf(cell) {
  const C = rows(cell);
  const [a, b, c] = C.map(norm3);
  const ang = (u, v, nu, nv) => (Math.acos(Math.max(-1, Math.min(1, dot3(u, v) / (nu * nv)))) * 180) / Math.PI;
  return { a, b, c, alpha: ang(C[1], C[2], b, c), beta: ang(C[0], C[2], a, c), gamma: ang(C[0], C[1], a, b),
    volume: Math.abs(dot3(C[0], cross3(C[1], C[2]))) };
}
/** The cell from a b c α β γ (ase's convention: a along x, b in the xy plane). */
export function cellFromParams(a, b, c, al, be, ga) {
  const r = Math.PI / 180, ca = Math.cos(al * r), cb = Math.cos(be * r), cg = Math.cos(ga * r), sg = Math.sin(ga * r);
  const cx = cb, cy = (ca - cb * cg) / sg, cz2 = 1 - cx * cx - cy * cy;
  if (!(a > 0 && b > 0 && c > 0) || !(cz2 > 1e-12)) return null;
  return [[a, 0, 0], [b * cg, b * sg, 0], [c * cx, c * cy, c * Math.sqrt(cz2)]];
}

/** The shortest Cartesian vector between two fractional positions, with the
 * lattice translation that gives it (checked over the 27 neighbors of the
 * rounded difference, so it is exact for skewed cells too). */
export function minImage(cell, fa, fb) {
  const C = rows(cell);
  const d = [fb[0] - fa[0], fb[1] - fa[1], fb[2] - fa[2]];
  const r = d.map(Math.round);
  let best = null, bestImg = null, bestLen = Infinity;
  for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) for (let k = -1; k <= 1; k++) {
    const img = [-r[0] + i, -r[1] + j, -r[2] + k];
    const v = cartOf(C, [d[0] + img[0], d[1] + img[1], d[2] + img[2]]);
    const L = dot3(v, v);
    // a tie (an atom exactly half a cell away) goes to the smallest shift, the atom's own cell first
    const size = Math.abs(img[0]) + Math.abs(img[1]) + Math.abs(img[2]);
    if (L < bestLen - 1e-9 || (Math.abs(L - bestLen) <= 1e-9 && size < bestImg.reduce((t, x) => t + Math.abs(x), 0))) {
      bestLen = L; best = v; bestImg = img;
    }
  }
  return { vec: best, image: bestImg, length: Math.sqrt(bestLen) };
}
export const distance = (p, q) => norm3(sub3(p, q));
/** The angle at b, in degrees. */
export function angle(a, b, c) {
  const u = sub3(a, b), v = sub3(c, b);
  return (Math.acos(Math.max(-1, Math.min(1, dot3(u, v) / (norm3(u) * norm3(v) || 1)))) * 180) / Math.PI;
}
/** The dihedral a-b-c-d in degrees, -180 to 180 (IUPAC sign). */
export function dihedral(a, b, c, d) {
  const b0 = sub3(a, b), b1 = sub3(c, b), b2 = sub3(d, c);
  const n1 = unit3(b1);
  const v = sub3(b0, scale3(n1, dot3(b0, n1))), w = sub3(b2, scale3(n1, dot3(b2, n1)));
  const x = dot3(v, w), y = dot3(cross3(n1, v), w);
  return (Math.atan2(y, x) * 180) / Math.PI;
}
/** A measurement's value from its atoms [{i, image:[n1,n2,n3]}] in a structure
 * {cell, frac}: 2 atoms distance (Å), 3 angle (°), 4 dihedral (°). */
export function measure(cell, frac, atoms) {
  const pts = atoms.map(({ i, image }) => cartOf(cell, [0, 1, 2].map((k) => frac[i][k] + (image ? image[k] : 0))));
  if (pts.length === 2) return distance(pts[0], pts[1]);
  if (pts.length === 3) return angle(pts[0], pts[1], pts[2]);
  if (pts.length === 4) return dihedral(pts[0], pts[1], pts[2], pts[3]);
  return NaN;
}

/* ---------------------------------------------------------------- bonds */
/* A cell list (spec §8.3.1): buckets as large as the longest bond cutoff,
 * each atom checked against its 27 neighbor buckets: O(N), never O(N²). */
function cutoffOf(radii) {
  let r = 0;
  for (let i = 0; i < radii.length; i++) if (radii[i] > r) r = radii[i];
  return 2 * r;
}
/** Bonds between drawn atoms (Cartesian, no periodicity): pairs i<j with
 * 0.1 < d < tol·(ri+rj), no metal–metal pair unless allowed, as the old
 * O(N²) bondsFor. Returns Uint32Array [i0, j0, i1, j1, ...]. */
export function cellListBonds(cart, radii, metal, tol, metalMetal = false) {
  const n = radii.length;
  if (!n) return new Uint32Array(0);
  const cut = tol * cutoffOf(radii) || 1;
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) {
    const v = cart[3 * i + k];
    if (v < lo[k]) lo[k] = v;
    if (v > hi[k]) hi[k] = v;
  }
  // never more buckets than atoms (a sparse cloud would otherwise allocate a huge grid)
  let size = cut;
  const dims = () => [0, 1, 2].map((k) => Math.max(1, Math.floor((hi[k] - lo[k]) / size) + 1));
  let g = dims();
  while (g[0] * g[1] * g[2] > 4 * n + 64) { size *= 1.5; g = dims(); }
  const nb = g[0] * g[1] * g[2];
  const cellOf = new Int32Array(n), count = new Int32Array(nb + 1);
  for (let i = 0; i < n; i++) {
    const a = Math.min(g[0] - 1, Math.floor((cart[3 * i] - lo[0]) / size));
    const b = Math.min(g[1] - 1, Math.floor((cart[3 * i + 1] - lo[1]) / size));
    const c = Math.min(g[2] - 1, Math.floor((cart[3 * i + 2] - lo[2]) / size));
    const id = (a * g[1] + b) * g[2] + c;
    cellOf[i] = id; count[id + 1]++;
  }
  for (let k = 0; k < nb; k++) count[k + 1] += count[k];
  const order = new Int32Array(n), fill = count.slice(0, nb);
  for (let i = 0; i < n; i++) order[fill[cellOf[i]]++] = i;
  let out = new Uint32Array(Math.max(64, 8 * n)), m = 0;
  for (let i = 0; i < n; i++) {
    const id = cellOf[i], a = Math.floor(id / (g[1] * g[2])), b = Math.floor(id / g[2]) % g[1], c = id % g[2];
    const xi = cart[3 * i], yi = cart[3 * i + 1], zi = cart[3 * i + 2], ri = radii[i], mi = metal[i];
    for (let da = -1; da <= 1; da++) {
      const A = a + da; if (A < 0 || A >= g[0]) continue;
      for (let db = -1; db <= 1; db++) {
        const B = b + db; if (B < 0 || B >= g[1]) continue;
        for (let dc = -1; dc <= 1; dc++) {
          const Cc = c + dc; if (Cc < 0 || Cc >= g[2]) continue;
          const cell2 = (A * g[1] + B) * g[2] + Cc;
          for (let p = count[cell2]; p < count[cell2 + 1]; p++) {
            const j = order[p];
            if (j <= i) continue;
            if (!metalMetal && mi && metal[j]) continue;
            const dx = cart[3 * j] - xi, dy = cart[3 * j + 1] - yi, dz = cart[3 * j + 2] - zi;
            const d2 = dx * dx + dy * dy + dz * dz, lim = tol * (ri + radii[j]);
            if (d2 > 0.01 && d2 < lim * lim) {
              if (m + 2 > out.length) { const o = new Uint32Array(out.length * 2); o.set(out); out = o; }
              out[m++] = i; out[m++] = j;
            }
          }
        }
      }
    }
  }
  return out.slice(0, m);
}
/** Bonds of the periodic structure (the bond graph for fragments, neighbors
 * and Add Atom): {i, j, img} with atom j at image img bonded to atom i, each
 * bond once. Buckets in fractional space, wrapped at the cell faces. */
export function periodicBonds(cell, frac, radii, metal, tol, metalMetal = false) {
  const C = rows(cell), R = reciprocal(C), n = radii.length;
  const I = [], J = [], IMG = [];
  if (!n || !R) return { i: new Uint32Array(0), j: new Uint32Array(0), img: new Int8Array(0) };
  const cut = tol * cutoffOf(radii) || 1;
  const h = R.map((r) => 1 / norm3(r));                     // plane spacings
  const g = h.map((hk) => Math.max(1, Math.min(64, Math.floor(hk / cut))));
  const reach = h.map((hk, k) => Math.ceil(cut / (hk / g[k])));
  const nb = g[0] * g[1] * g[2];
  const w = new Float64Array(3 * n), cellOf = new Int32Array(n), count = new Int32Array(nb + 1);
  for (let i = 0; i < n; i++) {
    const idx = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
      let f = frac[3 * i + k] - Math.floor(frac[3 * i + k]);
      if (f >= 1) f = 0;
      w[3 * i + k] = f;
      idx[k] = Math.min(g[k] - 1, Math.floor(f * g[k]));
    }
    cellOf[i] = (idx[0] * g[1] + idx[1]) * g[2] + idx[2];
    count[cellOf[i] + 1]++;
  }
  for (let k = 0; k < nb; k++) count[k + 1] += count[k];
  const order = new Int32Array(n), fill = count.slice(0, nb);
  for (let i = 0; i < n; i++) order[fill[cellOf[i]]++] = i;
  const shiftOf = (i) => [0, 1, 2].map((k) => w[3 * i + k] - frac[3 * i + k]);   // wrapped = frac + shift
  for (let i = 0; i < n; i++) {
    const id = cellOf[i], a = [Math.floor(id / (g[1] * g[2])), Math.floor(id / g[2]) % g[1], id % g[2]];
    const fi = [w[3 * i], w[3 * i + 1], w[3 * i + 2]];
    const si = shiftOf(i);
    for (let da = -reach[0]; da <= reach[0]; da++) for (let db = -reach[1]; db <= reach[1]; db++) for (let dc = -reach[2]; dc <= reach[2]; dc++) {
      const t = [a[0] + da, a[1] + db, a[2] + dc];
      const img = t.map((x, k) => Math.floor(x / g[k]));
      const cid = ((t[0] - img[0] * g[0]) * g[1] + (t[1] - img[1] * g[1])) * g[2] + (t[2] - img[2] * g[2]);
      for (let p = count[cid]; p < count[cid + 1]; p++) {
        const j = order[p];
        // each bond once: j > i, or the atom's own image on the positive side
        if (j < i || (j === i && !(img[0] > 0 || (img[0] === 0 && (img[1] > 0 || (img[1] === 0 && img[2] > 0)))))) continue;
        if (!metalMetal && metal[i] && metal[j]) continue;
        const df = [w[3 * j] + img[0] - fi[0], w[3 * j + 1] + img[1] - fi[1], w[3 * j + 2] + img[2] - fi[2]];
        const v = cartOf(C, df), d2 = dot3(v, v), lim = tol * (radii[i] + radii[j]);
        if (d2 > 0.01 && d2 < lim * lim) {
          // the image in terms of the unwrapped positions the caller gave
          const sj = shiftOf(j);
          I.push(i); J.push(j); IMG.push(img[0] + sj[0] - si[0], img[1] + sj[1] - si[1], img[2] + sj[2] - si[2]);
        }
      }
    }
  }
  return { i: Uint32Array.from(I), j: Uint32Array.from(J), img: Int8Array.from(IMG.map(Math.round)) };
}
/** Neighbors of each atom from periodicBonds: Map i -> [{j, img}]. */
export function adjacency(bonds, n) {
  const adj = Array.from({ length: n }, () => []);
  for (let k = 0; k < bonds.i.length; k++) {
    const i = bonds.i[k], j = bonds.j[k], img = [bonds.img[3 * k], bonds.img[3 * k + 1], bonds.img[3 * k + 2]];
    adj[i].push({ j, img });
    if (i !== j) adj[j].push({ j: i, img: img.map((x) => -x) });
  }
  return adj;
}
/** The atoms bonded to any atom of `sel` (Set of indices). */
export function neighborsOf(adj, sel) {
  const out = new Set();
  for (const i of sel) for (const { j } of adj[i] || []) if (!sel.has(j)) out.add(j);
  return out;
}
/** The connected fragment(s) holding `start` (minimum-image bond graph). */
export function fragmentOf(adj, start) {
  const seen = new Set(start), todo = [...start];
  while (todo.length) {
    const i = todo.pop();
    for (const { j } of adj[i] || []) if (!seen.has(j)) { seen.add(j); todo.push(j); }
  }
  return seen;
}

/* ---------------------------------------------------------------- coordination (plan B9) */
/** Each atom's bonded neighbors {j, img, d, vec} from periodicBonds with metal-metal
 * pairs searched too; an atom with any metal/non-metal bond keeps only those (so
 * Mg in MgO has its six O, not its twelve Mg), an all-metal one keeps its metal
 * shell (Cu: twelve). Self-image bonds count in both directions. */
export function bondedShells(cell, frac, radii, metal, tol) {
  const C = rows(cell), n = radii.length;
  const b = periodicBonds(C, frac, radii, metal, tol, true);
  const adj = Array.from({ length: n }, () => []);
  const f = (i) => [frac[3 * i], frac[3 * i + 1], frac[3 * i + 2]];
  for (let k = 0; k < b.i.length; k++) {
    const i = b.i[k], j = b.j[k], img = [b.img[3 * k], b.img[3 * k + 1], b.img[3 * k + 2]];
    const vec = cartOf(C, add3(sub3(f(j), f(i)), img)), d = norm3(vec);
    adj[i].push({ j, img, d, vec });
    adj[j].push({ j: i, img: img.map((x) => -x), d, vec: scale3(vec, -1) });
  }
  for (let i = 0; i < n; i++) {
    const hetero = adj[i].some((x) => !metal[i] !== !metal[x.j]);
    if (hetero) adj[i] = adj[i].filter((x) => !(metal[i] && metal[x.j]));
    adj[i].sort((x, y) => x.d - y.d);
  }
  return adj;
}
/** Hoppe's effective coordination number: the weighted mean bond length
 * l_av (iterated from the shortest) and ECoN = Σ exp(1 - (l/l_av)^6). */
export function econ(lengths) {
  if (!lengths.length) return { econ: 0, lav: 0 };
  let lav = Math.min(...lengths);
  for (let it = 0; it < 50; it++) {
    let num = 0, den = 0;
    for (const l of lengths) { const w = Math.exp(1 - Math.pow(l / lav, 6)); num += l * w; den += w; }
    const next = num / den;
    if (Math.abs(next - lav) < 1e-7) { lav = next; break; }
    lav = next;
  }
  let e = 0;
  for (const l of lengths) e += Math.exp(1 - Math.pow(l / lav, 6));
  return { econ: e, lav };
}
/** Coordination of the atoms `sites`: CN by the bond rule, shells (element and
 * distance, 0.005 Å bins) of every neighbor within 1.25 × the nearest bond, and ECoN
 * over those. z: atomic numbers. */
export function coordination(cell, frac, z, radii, metal, tol, sites, adj = null) {
  const C = rows(cell);
  adj = adj || bondedShells(C, frac, radii, metal, tol);
  const n = radii.length, R = reciprocal(C);
  const f = (i) => [frac[3 * i], frac[3 * i + 1], frac[3 * i + 2]];
  return sites.map((i) => {
    const bonded = adj[i] || [];
    const dmin = bonded.length ? bonded[0].d : 0;
    const shells = [], lens = [];
    if (dmin > 0 && R) {
      const rc = dmin * 1.25;
      const reach = R.map((r) => Math.ceil(rc * norm3(r)) + 1);
      const fi = f(i);
      for (let j = 0; j < n; j++) {
        const d0 = sub3(f(j), fi).map((x) => x - Math.round(x));
        for (let a = -reach[0]; a <= reach[0]; a++) for (let b = -reach[1]; b <= reach[1]; b++) for (let c = -reach[2]; c <= reach[2]; c++) {
          const d = norm3(cartOf(C, [d0[0] + a, d0[1] + b, d0[2] + c]));
          if (d < 0.1 || d > rc) continue;
          lens.push(d);
          const key = `${z[j]}:${Math.round(d / 0.005)}`;
          const sh = shells.find((x) => x.key === key);
          if (sh) sh.count++; else shells.push({ key, z: z[j], d, count: 1 });
        }
      }
      shells.sort((x, y) => x.d - y.d || x.z - y.z);
    }
    const byEl = new Map();
    for (const x of bonded) byEl.set(z[x.j], (byEl.get(z[x.j]) || 0) + 1);
    const e = econ(lens.length ? lens : bonded.map((x) => x.d));
    return { i, cn: bonded.length, nearest: dmin, mean: bonded.length ? bonded.reduce((t, x) => t + x.d, 0) / bonded.length : 0,
      neighbors: [...byEl].map(([zz, count]) => ({ z: zz, count })), shells: shells.map(({ z: zz, d, count }) => ({ z: zz, d, count })),
      econ: e.econ, lav: e.lav };
  });
}
/** The convex hull of a few points (a coordination polyhedron, 4-12 vertices):
 * every plane through three points with all others on one side is a facet;
 * coplanar points merge into one polygon (a square face of a cube stays one
 * face), returned in order around its normal and fanned into triangles.
 * Exact for these sizes and simpler than an incremental quickhull. */
export function convexHull(pts, eps = 1e-6) {
  const n = pts.length, facets = [];
  if (n < 4) return { facets, triangles: [] };
  const scaleL = Math.max(...pts.map((p) => norm3(p))) || 1, tolD = eps * scaleL * 10;
  const seen = new Set();
  for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) for (let c = b + 1; c < n; c++) {
    let nrm = cross3(sub3(pts[b], pts[a]), sub3(pts[c], pts[a]));
    const L = norm3(nrm);
    if (L < 1e-9 * scaleL * scaleL) continue;
    nrm = scale3(nrm, 1 / L);
    const off = dot3(nrm, pts[a]);
    let pos = 0, neg = 0;
    const on = [];
    for (let k = 0; k < n; k++) {
      const s = dot3(nrm, pts[k]) - off;
      if (s > tolD) pos++; else if (s < -tolD) neg++; else on.push(k);
    }
    if (pos && neg) continue;
    if (pos) { nrm = scale3(nrm, -1); }                  // outward: every other point behind the plane
    const key = on.slice().sort((x, y) => x - y).join(",");
    if (seen.has(key)) continue;
    seen.add(key);
    // order the facet's points around its centroid
    const cen = on.reduce((t, k) => add3(t, pts[k]), [0, 0, 0]).map((x) => x / on.length);
    const u = unit3(sub3(pts[on[0]], cen)), w = cross3(nrm, u);
    const ring = on.slice().sort((x, y) => {
      const px = sub3(pts[x], cen), py = sub3(pts[y], cen);
      return Math.atan2(dot3(px, w), dot3(px, u)) - Math.atan2(dot3(py, w), dot3(py, u));
    });
    facets.push({ ring, normal: nrm });
  }
  const triangles = [];
  for (const fct of facets) for (let k = 1; k + 1 < fct.ring.length; k++) triangles.push([fct.ring[0], fct.ring[k], fct.ring[k + 1]]);
  return { facets, triangles };
}
/* Pauling electronegativities by atomic number (0: none; the noble gases are never centers). */
export const PAULING = [0,
  2.2, 0, 0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98, 0, 0.93, 1.31, 1.61, 1.9, 2.19, 2.58, 3.16, 0,
  0.82, 1, 1.36, 1.54, 1.63, 1.66, 1.55, 1.83, 1.88, 1.91, 1.9, 1.65, 1.81, 2.01, 2.18, 2.55, 2.96, 3,
  0.82, 0.95, 1.22, 1.33, 1.6, 2.16, 1.9, 2.2, 2.28, 2.2, 1.93, 1.69, 1.78, 1.96, 2.05, 2.1, 2.66, 2.6,
  0.79, 0.89, 1.1, 1.12, 1.13, 1.14, 1.13, 1.17, 1.2, 1.2, 1.1, 1.22, 1.23, 1.24, 1.25, 1.1, 1.27,
  1.3, 1.5, 2.36, 1.9, 2.2, 2.2, 2.28, 2.54, 2, 1.62, 2.33, 2.02, 2, 2.2, 2.2,
  0.7, 0.9, 1.1, 1.3, 1.5, 1.38, 1.36, 1.28, 1.3, 1.3];
/** Coordination polyhedra (plan B9; the Polyhedral style, 1.1.7): for each drawn atom whose base atom
 * is a center (CN 4-12, every bonded neighbor more electronegative: Ti and Pb in TiPbO3, Mg in MgO,
 * S in a sulfonyl group; never Si in Si or C in graphene), its neighbors' hull, translated to where
 * the atom is drawn. One mesh per center element: flat-shaded triangles
 * (positions and normals, three vertices each) and the edges as pairs. */
export function polyhedra(cell, frac, z, radii, metal, tol, dispCart, dispBase, only = null) {
  const C = rows(cell), n = radii.length;
  const adj = bondedShells(C, frac, radii, metal, tol);
  let center = new Uint8Array(n);
  const en = (zz) => PAULING[zz] || 0;
  for (let i = 0; i < n; i++) {
    const nb = adj[i];
    if (nb.length >= 4 && nb.length <= 12 && en(z[i]) > 0 && nb.every((x) => en(z[x.j]) > en(z[i]))) center[i] = 1;
  }
  if (only && only.length) { const set = new Set(only); for (let i = 0; i < n; i++) center[i] = center[i] && set.has(z[i]) ? 1 : 0; }
  const hulls = new Map();
  const hullOf = (i) => {
    if (!hulls.has(i)) {
      const pts = adj[i].map((x) => x.vec);
      const h = convexHull(pts);
      const edges = new Set();
      for (const fct of h.facets) for (let k = 0; k < fct.ring.length; k++) {
        const a = fct.ring[k], b = fct.ring[(k + 1) % fct.ring.length];
        edges.add(a < b ? `${a},${b}` : `${b},${a}`);
      }
      hulls.set(i, { pts, h, edges: [...edges].map((e) => e.split(",").map(Number)) });
    }
    return hulls.get(i);
  };
  const byZ = new Map();
  let count = 0;
  for (let k = 0; k < dispBase.length; k++) {
    const i = dispBase[k];
    if (!center[i]) continue;
    const { pts, h, edges } = hullOf(i);
    if (!h.triangles.length) continue;
    const o = [dispCart[3 * k], dispCart[3 * k + 1], dispCart[3 * k + 2]];
    if (!byZ.has(z[i])) byZ.set(z[i], { z: z[i], pos: [], nrm: [], edge: [], n: 0 });
    const m = byZ.get(z[i]);
    for (const fct of h.facets) for (let t = 1; t + 1 < fct.ring.length; t++) {
      for (const v of [fct.ring[0], fct.ring[t], fct.ring[t + 1]]) {
        m.pos.push(o[0] + pts[v][0], o[1] + pts[v][1], o[2] + pts[v][2]);
        m.nrm.push(fct.normal[0], fct.normal[1], fct.normal[2]);
      }
    }
    for (const [a, b] of edges) m.edge.push(o[0] + pts[a][0], o[1] + pts[a][1], o[2] + pts[a][2], o[0] + pts[b][0], o[1] + pts[b][1], o[2] + pts[b][2]);
    m.n++; count++;
  }
  const meshes = [...byZ.values()].map((m) => ({ z: m.z, n: m.n, pos: Float32Array.from(m.pos), nrm: Float32Array.from(m.nrm), edge: Float32Array.from(m.edge) }));
  const faces = [...hulls.entries()].filter(([i]) => center[i]).map(([i, x]) => ({ i, cn: x.pts.length, faces: x.h.facets.length }));
  return { meshes, count, faces };
}

/* ---------------------------------------------------------------- symmetry */
const wrap1 = (x) => { const y = x - Math.floor(x); return y >= 1 - 1e-12 ? 0 : y; };
function fracDist(C, a, b) {
  const d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]].map((x) => x - Math.round(x));
  return norm3(cartOf(C, d));
}
/** The distinct images of a fractional position under the operations
 * {rotations, translations} (wrapped into [0, 1)), as the server's _images. */
export function orbitApply(ops, f, cell, tol = 1e-3) {
  const C = rows(cell), out = [];
  for (let k = 0; k < ops.rotations.length; k++) {
    const Rm = ops.rotations[k], t = ops.translations[k];
    const p = [0, 1, 2].map((r) => wrap1(Rm[r][0] * f[0] + Rm[r][1] * f[1] + Rm[r][2] * f[2] + t[r]));
    if (!out.some((q) => fracDist(C, p, q) < tol)) out.push(p);
  }
  return out;
}
/** The operations that leave a site where it is (R·f0 + t = f0 mod 1). */
export function siteStabilizer(ops, f0, cell, tol = 1e-3) {
  const C = rows(cell), out = [];
  for (let k = 0; k < ops.rotations.length; k++) {
    const Rm = ops.rotations[k], t = ops.translations[k];
    const p = [0, 1, 2].map((r) => Rm[r][0] * f0[0] + Rm[r][1] * f0[1] + Rm[r][2] * f0[2] + t[r]);
    if (fracDist(C, p, f0) < tol) out.push(Rm);
  }
  return out;
}
/** The part of a fractional displacement that keeps the site's symmetry:
 * the mean of R·df over its stabilizer ("Keep Fm-3m"). */
export function projectOntoSite(ops, f0, df, cell, tol = 1e-3) {
  const stab = siteStabilizer(ops, f0, cell, tol);
  if (!stab.length) return df.slice();
  const s = [0, 0, 0];
  for (const Rm of stab) for (let r = 0; r < 3; r++) s[r] += Rm[r][0] * df[0] + Rm[r][1] * df[1] + Rm[r][2] * df[2];
  return s.map((x) => x / stab.length);
}
/** Does moving a site by df leave the subspace its symmetry allows? */
export function leavesSite(ops, f0, df, cell, tol = 1e-3) {
  const p = projectOntoSite(ops, f0, df, cell, tol);
  return norm3(cartOf(cell, sub3(df, p))) > tol;
}

/* ---------------------------------------------------------------- projection */
/** A 3x4 camera matrix (11 numbers, the last element 1) from ≥6 model
 * points and where they land on screen: [[x, y, z, u, v], ...]. A direct
 * linear transform, exact for orthographic and perspective cameras. */
export function dltCalibrate(pairs) {
  const N = Array.from({ length: 11 }, () => new Float64Array(11)), B = new Float64Array(11);
  for (const [X, Y, Z, u, v] of pairs) {
    const r1 = [X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z], r2 = [0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z];
    for (const [r, rhs] of [[r1, u], [r2, v]]) for (let i = 0; i < 11; i++) {
      B[i] += r[i] * rhs;
      for (let j = 0; j < 11; j++) N[i][j] += r[i] * r[j];
    }
  }
  return solve(N, B);
}
function solve(A, b) {
  const n = b.length, M = A.map((r) => Float64Array.from(r)), y = Float64Array.from(b);
  for (let i = 0; i < n; i++) {
    let p = i;
    for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r;
    [M[i], M[p]] = [M[p], M[i]]; [y[i], y[p]] = [y[p], y[i]];
    if (Math.abs(M[i][i]) < 1e-300) continue;
    for (let r = i + 1; r < n; r++) {
      const f = M[r][i] / M[i][i];
      if (!f) continue;
      for (let c = i; c < n; c++) M[r][c] -= f * M[i][c];
      y[r] -= f * y[i];
    }
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let c = i + 1; c < n; c++) s -= M[i][c] * x[c];
    x[i] = M[i][i] ? s / M[i][i] : 0;
  }
  return x;
}
/** Screen point [u, v] of a model point, and w (the perspective divisor:
 * larger is farther for a perspective camera, constant for orthographic). */
export function project(P, x, y, z) {
  const w = P[8] * x + P[9] * y + P[10] * z + 1;
  return [(P[0] * x + P[1] * y + P[2] * z + P[3]) / w, (P[4] * x + P[5] * y + P[6] * z + P[7]) / w, w];
}
/** Every point at once: Float32Array [u0, v0, u1, v1, ...]. */
export function projectAll(P, cart, out = null) {
  const n = cart.length / 3, uv = out || new Float32Array(2 * n);
  for (let i = 0; i < n; i++) {
    const x = cart[3 * i], y = cart[3 * i + 1], z = cart[3 * i + 2];
    const w = P[8] * x + P[9] * y + P[10] * z + 1;
    uv[2 * i] = (P[0] * x + P[1] * y + P[2] * z + P[3]) / w;
    uv[2 * i + 1] = (P[4] * x + P[5] * y + P[6] * z + P[7]) / w;
  }
  return uv;
}
/** Even-odd rule: is (x, y) inside the polygon [[x, y], ...]? */
export function pointInPolygon(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
/** Indices whose projected point is inside a rectangle {x0, y0, x1, y1} or a polygon. */
export function boxSelect(uv, r) {
  const x0 = Math.min(r.x0, r.x1), x1 = Math.max(r.x0, r.x1), y0 = Math.min(r.y0, r.y1), y1 = Math.max(r.y0, r.y1);
  const out = [];
  for (let i = 0; i < uv.length / 2; i++) {
    const u = uv[2 * i], v = uv[2 * i + 1];
    if (u >= x0 && u <= x1 && v >= y0 && v <= y1) out.push(i);
  }
  return Uint32Array.from(out);
}
export function lassoSelect(uv, poly) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of poly) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  const out = [];
  for (let i = 0; i < uv.length / 2; i++) {
    const u = uv[2 * i], v = uv[2 * i + 1];
    if (u >= x0 && u <= x1 && v >= y0 && v <= y1 && pointInPolygon(u, v, poly)) out.push(i);
  }
  return Uint32Array.from(out);
}

/* ---------------------------------------------------------------- level of detail */
/** The drawing level for N displayed atoms (spec §8.2.4): 1 up to 2k, 2 up
 * to 20k, 3 up to 200k, 4 above; Quality Low/Medium/High scales the limits
 * by 0.5/1/2 (Auto is Medium). */
export function lodLevel(n, quality = "auto") {
  const f = { low: 0.5, medium: 1, high: 2, auto: 1 }[quality] || 1;
  if (n <= 2000 * f) return 1;
  if (n <= 20000 * f) return 2;
  if (n <= 200000 * f) return 3;
  return 4;
}

/* ---------------------------------------------------------------- view directions */
/** A unit view direction d (pointing toward the viewer): a, b or c, a*, b*
 * or c* (rows of the reciprocal cell), or a typed [u, v, w] (u·a + v·b + w·c). */
export function alongDirection(cell, which) {
  const C = rows(cell);
  if (Array.isArray(which)) return unit3(cartOf(C, which));
  const k = "abc".indexOf(which[0]);
  if (k < 0) throw new Error(`no direction ${which}`);
  return unit3(which.endsWith("*") ? reciprocal(C)[k] : C[k]);
}
/** "Up" for a view along d: the part of c perpendicular to d, or of b when
 * c is within 5° of d. */
export function upVector(cell, d) {
  const C = rows(cell);
  for (const v of [C[2], C[1], C[0]]) {
    const u = unit3(v);
    if (Math.abs(dot3(u, d)) < Math.cos((5 * Math.PI) / 180)) return unit3(sub3(u, scale3(d, dot3(u, d))));
  }
  return [0, 1, 0];
}
/** The view quaternion [x, y, z, w] that turns d to +z (toward the viewer)
 * and up to +y. */
export function viewQuaternion(d, up) {
  const z = unit3(d), y = unit3(sub3(up, scale3(z, dot3(up, z)))), x = cross3(y, z);
  // rows of the rotation are x, y, z (model -> camera)
  const m = [x, y, z];
  const tr = m[0][0] + m[1][1] + m[2][2];
  let q;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    q = [(m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, 0.25 * s];
  } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
    const s = Math.sqrt(1 + m[0][0] - m[1][1] - m[2][2]) * 2;
    q = [0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s];
  } else if (m[1][1] > m[2][2]) {
    const s = Math.sqrt(1 + m[1][1] - m[0][0] - m[2][2]) * 2;
    q = [(m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s];
  } else {
    const s = Math.sqrt(1 + m[2][2] - m[0][0] - m[1][1]) * 2;
    q = [(m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s, (m[1][0] - m[0][1]) / s];
  }
  const n = Math.hypot(...q);
  return q.map((v) => v / n);
}
/** The default view of a cell (spec carry-over a, decision D5): c up on
 * screen, looking from an azimuth of 25° about c and 12° above the ab plane,
 * so a comes toward the lower left and b goes right (as VESTA and
 * CrystalMaker show a crystal). mode "a" / "c" looks straight along a or c
 * instead. Returns a view quaternion, or null for a cell with no lattice (a
 * molecule), which keeps the camera it has. */
export function standardView(cell, pbc = [true, true, true], mode = "standard", azimuth = 25, elevation = 12) {
  if (!cell || !pbc || !pbc.some(Boolean)) return null;
  const C = rows(cell);
  if (C.some((r) => !(norm3(r) > 1e-9))) return null;
  if (mode === "a" || mode === "c") {
    const d = alongDirection(C, mode);
    return viewQuaternion(d, upVector(C, d));
  }
  const ec = unit3(C[2]);
  // a's part perpendicular to c; b's when a is (nearly) along c
  let u = sub3(C[0], scale3(ec, dot3(C[0], ec)));
  if (norm3(u) < 1e-6 * norm3(C[0])) u = sub3(C[1], scale3(ec, dot3(C[1], ec)));
  u = unit3(u);
  const w = cross3(ec, u);
  const phi = (azimuth * Math.PI) / 180, el = (elevation * Math.PI) / 180;
  const flat = add3(scale3(u, Math.cos(phi)), scale3(w, Math.sin(phi)));
  const d = unit3(add3(scale3(flat, Math.cos(el)), scale3(ec, Math.sin(el))));
  return viewQuaternion(d, ec);
}
/** Quaternion helpers ([x, y, z, w]). */
export function qmul(a, b) {
  return [a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1], a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3], a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
}
export function qaxis(axis, deg) {
  const u = unit3(axis), h = ((deg * Math.PI) / 180) / 2, s = Math.sin(h);
  return [u[0] * s, u[1] * s, u[2] * s, Math.cos(h)];
}
export function qrotate(q, p) {
  const [x, y, z, w] = q;
  const ix = w * p[0] + y * p[2] - z * p[1], iy = w * p[1] + z * p[0] - x * p[2];
  const iz = w * p[2] + x * p[1] - y * p[0], iw = -x * p[0] - y * p[1] - z * p[2];
  return [ix * w + iw * -x + iy * -z - iz * -y, iy * w + iw * -y + iz * -x - ix * -z, iz * w + iw * -z + ix * -y - iy * -x];
}
export const qconj = (q) => [-q[0], -q[1], -q[2], q[3]];
export function qslerp(a, b, t) {
  let c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const bb = c < 0 ? b.map((x) => -x) : b;
  c = Math.abs(c);
  if (c > 0.9995) { const r = a.map((x, k) => x + t * (bb[k] - x)); const n = Math.hypot(...r); return r.map((x) => x / n); }
  const th = Math.acos(c), s = Math.sin(th), wa = Math.sin((1 - t) * th) / s, wb = Math.sin(t * th) / s;
  return a.map((x, k) => wa * x + wb * bb[k]);
}

/* ---------------------------------------------------------------- typed input */
/** A coordinate field's expression: numbers, + - * /, parentheses and a
 * minus sign before a number or a parenthesis ("1/3", "0.5+0.01", "-(1/4)").
 * Returns {ok, value} or {ok: false, error}. */
export function evalExpr(text) {
  const s = String(text ?? "").replace(/\s+/g, "");
  if (!s) return { ok: false, error: "Type a number" };
  let p = 0;
  const fail = (msg) => { throw new Error(msg); };
  const number = () => {
    const m = /^(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?/.exec(s.slice(p));
    if (!m) fail(p < s.length && /[A-Za-z]/.test(s[p]) ? `"${s[p]}" is not a number: use numbers, + − × ÷ and parentheses`
      : "Expected a number");
    p += m[0].length;
    return Number(m[0]);
  };
  const primary = () => {
    if (s[p] === "(") { p++; const v = expr(); if (s[p] !== ")") fail("A parenthesis is not closed"); p++; return v; }
    if (s[p] === "-" || s[p] === "+") {
      const sign = s[p] === "-" ? -1 : 1;
      p++;
      if (s[p] === "(") return sign * primary();
      if (!/[\d.]/.test(s[p] || "")) fail(s[p] && /[A-Za-z]/.test(s[p]) ? `"${s[p]}" is not a number: use numbers, + − × ÷ and parentheses` : "Expected a number");
      return sign * number();
    }
    return number();
  };
  const term = () => {
    let v = primary();
    while (s[p] === "*" || s[p] === "/") {
      const op = s[p++], r = primary();
      if (op === "/" && r === 0) fail("Division by zero");
      v = op === "*" ? v * r : v / r;
    }
    return v;
  };
  const expr = () => {
    let v = term();
    while (s[p] === "+" || s[p] === "-") { const op = s[p++]; const r = term(); v = op === "+" ? v + r : v - r; }
    return v;
  };
  try {
    const v = expr();
    if (p !== s.length) fail(`Unexpected "${s[p]}"`);
    if (!Number.isFinite(v)) fail("Not a finite number");
    return { ok: true, value: v };
  } catch (e) { return { ok: false, error: e.message }; }
}
/** "1-4, 9, 12-" (1-based, inclusive; "12-" runs to the end) -> 0-based indices. */
export function parseIndexList(text, n) {
  const out = new Set();
  for (const part of String(text || "").split(/[,\s]+/).filter(Boolean)) {
    const m = /^(\d+)?(-)?(\d+)?$/.exec(part);
    if (!m || (!m[1] && !m[3])) return { ok: false, error: `"${part}" is not an index or a range` };
    const a = m[1] ? Number(m[1]) : 1, b = m[2] ? (m[3] ? Number(m[3]) : n) : a;
    if (a < 1 || b > n || a > b) return { ok: false, error: `${part}: atoms are numbered 1 to ${n}` };
    for (let i = a; i <= b; i++) out.add(i - 1);
  }
  return { ok: true, indices: [...out].sort((x, y) => x - y) };
}
/** "z > 0.5", "x < 0.25 and y >= 0.1": a test on fractional coordinates
 * (wrapped into [0, 1)). Axes x y z or a b c. */
export function parseFracRange(text) {
  const conds = [];
  for (const part of String(text || "").split(/\s*(?:,|&&|\band\b)\s*/i).filter(Boolean)) {
    const m = /^\s*([xyzabc])\s*(<=|>=|<|>|=)\s*(-?\d*\.?\d+)\s*$/i.exec(part);
    if (!m) return { ok: false, error: `"${part}": write it like z > 0.5` };
    conds.push(["xyzabc".indexOf(m[1].toLowerCase()) % 3, m[2], Number(m[3])]);
  }
  if (!conds.length) return { ok: false, error: "Write a condition like z > 0.5" };
  const test = (f) => conds.every(([k, op, v]) => {
    const x = wrap1(f[k]);
    return op === "<" ? x < v : op === ">" ? x > v : op === "<=" ? x <= v : op === ">=" ? x >= v : Math.abs(x - v) < 1e-6;
  });
  return { ok: true, test };
}
/** Shown with `decimals` places; never "-0.000". */
export function fmtNum(x, decimals) {
  const s = Number(x).toFixed(decimals);
  return /^-0\.?0*$/.test(s) ? s.slice(1) : s;
}

/* ---------------------------------------------------------------- what is drawn */
/** The drawn atoms: each atom wrapped into the cell along periodic axes,
 * repeated over a display range [n1, n2, n3], plus the images on the cell
 * faces (VESTA's boundary atoms). Returns typed arrays: base (the atom each
 * drawn one is a copy of), shift (its cell offset), image (not the n=0
 * copy), cart. Non-periodic axes are drawn as they are. */
export function displayList(s, range = [1, 1, 1], boundary = true, tol = 0.02) {
  if (!Array.isArray(range)) return displayRange(s, range, boundary, tol);
  const C = rows(s.cell), pbc = s.pbc || [true, true, true], n = s.frac.length;
  const R = [0, 1, 2].map((k) => (pbc[k] ? Math.max(1, range[k] | 0) : 1));
  const base = [], shift = [], img = [], cart = [];
  for (let i = 0; i < n; i++) {
    const f0 = s.frac[i];
    const f = [0, 1, 2].map((k) => (pbc[k] ? f0[k] - Math.floor(f0[k] + 1e-9) : f0[k]));
    const lo = [0, 1, 2].map((k) => (!pbc[k] ? 0 : boundary ? -1 : 0));
    const hi = [0, 1, 2].map((k) => (!pbc[k] ? 0 : boundary ? R[k] : R[k] - 1));
    for (let a = lo[0]; a <= hi[0]; a++) for (let b = lo[1]; b <= hi[1]; b++) for (let c = lo[2]; c <= hi[2]; c++) {
      const sh = [a, b, c], g = [f[0] + a, f[1] + b, f[2] + c];
      let inside = true;
      for (let k = 0; k < 3; k++) if (pbc[k] && (g[k] < -tol || g[k] > R[k] + tol)) inside = false;
      if (!inside) continue;
      base.push(i);
      // the offset from the atom's own (unwrapped) fractional position
      shift.push(Math.round(g[0] - f0[0]), Math.round(g[1] - f0[1]), Math.round(g[2] - f0[2]));
      img.push(a === 0 && b === 0 && c === 0 ? 0 : 1);
      cart.push(g[0] * C[0][0] + g[1] * C[1][0] + g[2] * C[2][0], g[0] * C[0][1] + g[1] * C[1][1] + g[2] * C[2][1],
        g[0] * C[0][2] + g[1] * C[1][2] + g[2] * C[2][2]);
    }
  }
  return { base: Int32Array.from(base), shift: Int16Array.from(shift), image: Uint8Array.from(img),
    cart: Float64Array.from(cart), n: base.length };
}

/* ---------------------------------------------------------------- display range, bonded atoms (polish A4) */
export const RANGE_MIN = -5, RANGE_MAX = 10, RANGE_SPAN = 0.05;
/** {lo, hi} clamped to [-5, 10] with hi - lo >= 0.05 on every axis. */
export function cleanRange(r) {
  const lo = [0, 1, 2].map((k) => Math.min(RANGE_MAX - RANGE_SPAN, Math.max(RANGE_MIN, Number(r.lo[k]) || 0)));
  const hi = [0, 1, 2].map((k) => Math.min(RANGE_MAX, Math.max(lo[k] + RANGE_SPAN, Number.isFinite(Number(r.hi[k])) ? Number(r.hi[k]) : 1)));
  return { lo, hi };
}
/** The old [n1, n2, n3] form when the range is whole cells from 0 (so the
 * default draws exactly as before), else null. */
export function integerRange(r) {
  const ok = [0, 1, 2].every((k) => r.lo[k] === 0 && Number.isInteger(r.hi[k]) && r.hi[k] >= 1);
  return ok ? r.hi.slice() : null;
}
/* The fractional form of displayList: VESTA's boundary from lo to hi on each
 * periodic axis. Boundary on keeps lo - tol <= g <= hi + tol (the atoms on
 * the range's faces, edges and corners); off keeps lo <= g < hi (each site
 * once per cell for whole ranges). Non-periodic axes ignore the range. */
function displayRange(s, range, boundary, tol) {
  const { lo, hi } = cleanRange(range);
  const C = rows(s.cell), pbc = s.pbc || [true, true, true], n = s.frac.length;
  const a0 = [0, 1, 2].map((k) => (pbc[k] ? Math.floor(lo[k] - tol) : 0));
  const a1 = [0, 1, 2].map((k) => (pbc[k] ? Math.floor(hi[k] + tol) : 0));
  const keep = (g, k) => (boundary ? g >= lo[k] - tol && g <= hi[k] + tol : g >= lo[k] - 1e-9 && g < hi[k] - 1e-9);
  const base = [], shift = [], img = [], cart = [];
  for (let i = 0; i < n; i++) {
    const f0 = s.frac[i];
    const f = [0, 1, 2].map((k) => (pbc[k] ? f0[k] - Math.floor(f0[k] + 1e-9) : f0[k]));
    for (let a = a0[0]; a <= a1[0]; a++) for (let b = a0[1]; b <= a1[1]; b++) for (let c = a0[2]; c <= a1[2]; c++) {
      const g = [f[0] + a, f[1] + b, f[2] + c];
      let inside = true;
      for (let k = 0; k < 3; k++) if (pbc[k] && !keep(g[k], k)) inside = false;
      if (!inside) continue;
      base.push(i);
      shift.push(Math.round(g[0] - f0[0]), Math.round(g[1] - f0[1]), Math.round(g[2] - f0[2]));
      img.push(a === 0 && b === 0 && c === 0 ? 0 : 1);
      cart.push(g[0] * C[0][0] + g[1] * C[1][0] + g[2] * C[2][0], g[0] * C[0][1] + g[1] * C[1][1] + g[2] * C[2][1],
        g[0] * C[0][2] + g[1] * C[1][2] + g[2] * C[2][2]);
    }
  }
  return { base: Int32Array.from(base), shift: Int16Array.from(shift), image: Uint8Array.from(img),
    cart: Float64Array.from(cart), n: base.length };
}
/** The 8 corners of the cell box from box.lo to box.hi (whole cells; non-periodic axes one cell), plus the origin. */
export function boxCorners(s, box) {
  const C = rows(s.cell), pbc = s.pbc || [true, true, true], o = s.origin || [0, 0, 0];
  const lo = [0, 1, 2].map((k) => (pbc[k] ? box.lo[k] : 0)), hi = [0, 1, 2].map((k) => (pbc[k] ? box.hi[k] : 1));
  const at = (f) => [0, 1, 2].map((j) => o[j] + f[0] * C[0][j] + f[1] * C[1][j] + f[2] * C[2][j]);
  // the order the cell outline's edges index: 0, a, b, c, ab, ac, bc, abc
  return [[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [lo[0], hi[1], lo[2]], [lo[0], lo[1], hi[2]], [hi[0], hi[1], lo[2]],
    [hi[0], lo[1], hi[2]], [lo[0], hi[1], hi[2]], [hi[0], hi[1], hi[2]]].map(at);
}
/** The rotation center of the default view: the middle of the drawn cell box
 * (box.lo to box.hi) along each periodic axis and of the atoms (cart: flat
 * or [x, y, z] rows, n of them) along each other one. A crystal turns about
 * its cell's center (a displayed supercell: the center of what is drawn), a
 * slab about its in-plane cell center at the slab's mid-height, a molecule
 * (or a structure without a cell) about the middle of its atoms. */
export function pivotOf(s, box, cart, n = null) {
  const flat = !!(cart && cart.length && typeof cart[0] === "number");
  const count = n ?? (cart ? (flat ? cart.length / 3 : cart.length) : 0);
  const at = (k) => (flat ? [cart[3 * k], cart[3 * k + 1], cart[3 * k + 2]] : cart[k]);
  const pbc = s.pbc || [true, true, true];
  const R = s.cell && pbc.some(Boolean) ? reciprocal(s.cell) : null;
  if (!R) {
    if (!count) return [0, 0, 0];
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let k = 0; k < count; k++) {
      const p = at(k);
      for (let j = 0; j < 3; j++) { if (p[j] < lo[j]) lo[j] = p[j]; if (p[j] > hi[j]) hi[j] = p[j]; }
    }
    return [0, 1, 2].map((j) => (lo[j] + hi[j]) / 2);
  }
  const o = s.origin || [0, 0, 0], b = box || { lo: [0, 0, 0], hi: [1, 1, 1] };
  const f = [0, 1, 2].map((j) => {
    if (pbc[j]) return (b.lo[j] + b.hi[j]) / 2;
    let lo = Infinity, hi = -Infinity;
    for (let k = 0; k < count; k++) { const x = dot3(R[j], sub3(at(k), o)); if (x < lo) lo = x; if (x > hi) hi = x; }
    return count ? (lo + hi) / 2 : 0.5;
  });
  return add3(o, cartOf(s.cell, f));
}
/** About how many atoms a range draws (the Inspector's estimate). */
export function estimateCount(s, range, boundary) {
  const pbc = s.pbc || [true, true, true];
  const r = Array.isArray(range) ? { lo: [0, 0, 0], hi: range } : cleanRange(range);
  let v = s.frac.length;
  for (let k = 0; k < 3; k++) if (pbc[k]) v *= Math.max(r.hi[k] - r.lo[k], 0.05) + (boundary ? 0.04 : 0);
  return Math.round(v);
}

/* Integer rank of a set of integer 3-vectors (Gaussian elimination). */
function rank3(vecs) {
  const m = vecs.map((v) => v.slice());
  let r = 0;
  for (let col = 0; col < 3 && r < m.length; col++) {
    let p = -1;
    for (let i = r; i < m.length; i++) if (Math.abs(m[i][col]) > 1e-9) { p = i; break; }
    if (p < 0) continue;
    [m[r], m[p]] = [m[p], m[r]];
    for (let i = 0; i < m.length; i++) {
      if (i === r || Math.abs(m[i][col]) < 1e-12) continue;
      const f = m[i][col] / m[r][col];
      for (let k = 0; k < 3; k++) m[i][k] -= f * m[r][k];
    }
    r++;
  }
  return r;
}
/** Connected components of the periodic bond graph and their dimensionality
 * (Larsen et al. 2019, the rank determination behind ase.geometry.dimensionality):
 * a BFS gives each atom an image offset in its component; a bond that closes
 * a cycle with a nonzero net lattice shift adds that shift to the component's
 * set, whose rank is its dimension (0 molecule, 1 chain, 2 layer, 3 network).
 * Returns {comp: Int32Array (component of each atom), offset: Int32Array
 * (3 per atom), dims: [dim per component], axes: [names], adj}. */
export function components(s, tol = 1.15, metalMetal = false) {
  const n = s.frac.length, pbc = s.pbc || [true, true, true];
  const radii = s.radii ? Float32Array.from(s.radii) : Float32Array.from(s.numbers.map((z) => COVALENT[z % 100 || z] || 0.2));
  const metal = s.metal ? Uint8Array.from(s.metal.map(Number)) : Uint8Array.from(s.numbers.map((z) => (isMetal(z % 100 || z) ? 1 : 0)));
  const frac = Float64Array.from(s.frac.flat());
  const b = periodicBonds(s.cell, frac, radii, metal, tol, metalMetal);
  const adj = Array.from({ length: n }, () => []);
  for (let k = 0; k < b.i.length; k++) {
    const img = [b.img[3 * k], b.img[3 * k + 1], b.img[3 * k + 2]];
    // a bond across a non-periodic axis (a slab's c, a molecule's box) is not a bond
    if ([0, 1, 2].some((m) => !pbc[m] && img[m] !== 0)) continue;
    const i = b.i[k], j = b.j[k];
    adj[i].push({ j, img });
    if (i !== j) adj[j].push({ j: i, img: img.map((x) => -x) });
    else adj[i].push({ j: i, img: img.map((x) => -x) });
  }
  const comp = new Int32Array(n).fill(-1), offset = new Int32Array(3 * n), dims = [], axes = [];
  for (let s0 = 0; s0 < n; s0++) {
    if (comp[s0] >= 0) continue;
    const id = dims.length, cyc = [];
    comp[s0] = id;
    const todo = [s0];
    while (todo.length) {
      const i = todo.pop(), oi = [offset[3 * i], offset[3 * i + 1], offset[3 * i + 2]];
      for (const { j, img } of adj[i]) {
        const want = [oi[0] + img[0], oi[1] + img[1], oi[2] + img[2]];
        if (comp[j] < 0) {
          comp[j] = id; offset[3 * j] = want[0]; offset[3 * j + 1] = want[1]; offset[3 * j + 2] = want[2];
          todo.push(j);
        } else {
          const d = [want[0] - offset[3 * j], want[1] - offset[3 * j + 1], want[2] - offset[3 * j + 2]];
          if (d[0] || d[1] || d[2]) cyc.push(d);
        }
      }
    }
    const r = rank3(cyc);
    dims.push(r);
    axes.push(r === 0 ? "" : axesOf(cyc, r));
  }
  return { comp, offset, dims, axes, adj };
}
function axesOf(cyc, r) {
  if (r === 3) return "a, b, c";
  const used = [0, 1, 2].filter((k) => cyc.some((v) => v[k] !== 0));
  if (used.length === r) return used.map((k) => "abc"[k]).join(", ");
  return "oblique";
}

/** Atoms outside the range bonded to drawn ones (VESTA): mode "neighbors"
 * adds one shell; "molecules" completes every discrete (0-dimensional)
 * molecule that has a drawn atom. Extended components (chains, layers, 3D
 * networks) never grow: reported as extended {dim, axes}. The images are
 * appended with image = 2; a safety cap stops at `cap` drawn atoms
 * (truncated). Returns a display list like displayList's, plus
 * {added, extended, allExtended, truncated}. */
export function bondedImages(s, disp, mode, comps, cap = 50000) {
  const none = { ...disp, added: 0, extended: null, allExtended: false, truncated: false };
  if (!mode || mode === "off" || !disp.n) return none;
  const C = rows(s.cell), { comp, dims, axes, adj } = comps;
  const key = (i, a, b, c) => `${i}:${a},${b},${c}`;
  const have = new Set();
  for (let k = 0; k < disp.n; k++) have.add(key(disp.base[k], disp.shift[3 * k], disp.shift[3 * k + 1], disp.shift[3 * k + 2]));
  const base = Array.from(disp.base), shift = Array.from(disp.shift), image = Array.from(disp.image), cart = Array.from(disp.cart);
  let truncated = false;
  const add = (j, sh) => {
    const kk = key(j, sh[0], sh[1], sh[2]);
    if (have.has(kk)) return false;
    if (base.length >= cap) { truncated = true; return false; }
    have.add(kk);
    const f = s.frac[j], g = [f[0] + sh[0], f[1] + sh[1], f[2] + sh[2]];
    base.push(j); shift.push(sh[0], sh[1], sh[2]); image.push(2);
    cart.push(g[0] * C[0][0] + g[1] * C[1][0] + g[2] * C[2][0], g[0] * C[0][1] + g[1] * C[1][1] + g[2] * C[2][1],
      g[0] * C[0][2] + g[1] * C[1][2] + g[2] * C[2][2]);
    return true;
  };
  let maxDim = -1, maxAxes = "";
  const seenComp = new Set();
  for (let k = 0; k < disp.n; k++) {
    const c = comp[disp.base[k]];
    if (seenComp.has(c)) continue;
    seenComp.add(c);
    if (dims[c] > maxDim) { maxDim = dims[c]; maxAxes = axes[c]; }
  }
  const allExtended = [...seenComp].every((c) => dims[c] > 0);
  if (mode === "neighbors") {
    for (let k = 0; k < disp.n && !truncated; k++) {
      const i = disp.base[k], sh = [disp.shift[3 * k], disp.shift[3 * k + 1], disp.shift[3 * k + 2]];
      for (const { j, img } of adj[i]) add(j, [sh[0] + img[0], sh[1] + img[1], sh[2] + img[2]]);
    }
  } else if (mode === "molecules") {
    const todo = [];
    for (let k = 0; k < disp.n; k++) {
      if (dims[comp[disp.base[k]]] !== 0) continue;
      todo.push([disp.base[k], disp.shift[3 * k], disp.shift[3 * k + 1], disp.shift[3 * k + 2]]);
    }
    while (todo.length && !truncated) {
      const [i, a, b, c] = todo.pop();
      for (const { j, img } of adj[i]) {
        const sh = [a + img[0], b + img[1], c + img[2]];
        if (add(j, sh)) todo.push([j, ...sh]);
      }
    }
  }
  return { base: Int32Array.from(base), shift: Int16Array.from(shift), image: Uint8Array.from(image),
    cart: Float64Array.from(cart), n: base.length, added: base.length - disp.n,
    extended: maxDim > 0 ? { dim: maxDim, axes: maxAxes } : null, allExtended, truncated };
}
