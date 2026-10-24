/* Mesh quality and mesh style for every surface the 3D view draws (1.1.6):
 * the pore surfaces, cube isosurfaces (density, spin, potential, orbitals,
 * differences), the crystal shape, Miller planes and a net.
 *
 * Quality: Draft (screening many materials), Standard (the look before
 * presets, the default), High, Publication (dense triangulation for a
 * figure; seconds and more memory). The pore surfaces, the heaviest the
 * view draws (two walls each, over every cell of a supercell), have a tier
 * below Draft, Preview, which is their default whatever the other surfaces
 * use (Settings › Viewport › Pore surface mesh quality, pores.quality).
 * Each preset is a set of knobs the Inspector shows under Advanced; the server's copy of the pore and grid
 * numbers is mace/analysis/mesh_quality.py (tests/test_mesh_quality.py keeps
 * the two in step); the wireframe spacing, isosurface smoothing and sphere
 * detail are the page's own.
 *
 * Style: Solid (at the overlay's opacity: below 1 it is translucent),
 * Wireframe, Solid + edges, Points (the vertices as dots); Smooth or Flat
 * shading; the sides drawn. styled() turns triangle parts into 3Dmol shapes
 * in any of them (keepBuilt: a style change makes new shapes, an unchanged
 * one is never rebuilt). */
import { keepBuilt } from "./shape_cache.js";

export const PRESETS = ["draft", "standard", "high", "publication"];
export const DEFAULT = "standard";
export const KEY = "mesh.quality";
// the pore surfaces' presets (one tier below Draft) and their default, a setting of their own
export const PORE_PRESETS = ["preview", ...PRESETS];
export const PORE_DEFAULT = "preview";
export const PORE_KEY = "pores.quality";
export const PRESET_LABEL = { preview: "Preview", draft: "Draft", standard: "Standard", high: "High", publication: "Publication" };
export const PRESET_HELP = {
  preview: "Pore surfaces only: the coarsest, for many cells or slow graphics",
  draft: "Coarse and fast: for screening many materials",
  standard: "The default surfaces",
  high: "Finer surfaces for a close look",
  publication: "Dense triangulation for a figure: can take seconds and more memory",
};
export const presetOf = (v) => (PRESETS.includes(v) ? v : DEFAULT);
/** A pore surface's preset (Preview too); ``d``: an unknown one's. */
export const porePresetOf = (v, d = DEFAULT) => (PORE_PRESETS.includes(v) ? v : d);

/* the pore surfaces (porosity.py _render_fields / pore_meshes), as mesh_quality.py PORES */
export const PORES = {
  preview: { grid: 24, upsample: 1, smooth: 1.4, max_triangles: 60000 },
  draft: { grid: 40, upsample: 1, smooth: 0.7, max_triangles: 150000 },
  standard: { grid: 72, upsample: 1, smooth: 0.7, max_triangles: 600000 },
  high: { grid: 96, upsample: 2, smooth: 0.8, max_triangles: 1200000 },
  publication: { grid: 128, upsample: 3, smooth: 0.9, max_triangles: 2400000 },
};
export const PORE_LIMITS = { grid: [16, 160], upsample: [1, 4], smooth: [0, 2], max_triangles: [20000, 3000000] };
/* a cube's grid (gui/volume.py; points null: Settings › Volume quality) and 3Dmol's smoothing passes */
export const VOLUME = {
  draft: { points: 250000, upsample: 1, smoothness: 1 },
  standard: { points: null, upsample: 1, smoothness: 1 },
  high: { points: 2000000, upsample: 1, smoothness: 2 },
  publication: { points: 8000000, upsample: 2, smoothness: 3 },
};
export const VOLUME_LIMITS = { points: [50000, 8000000], upsample: [1, 3], smoothness: [0, 6] };
export const LEGACY_POINTS = { low: 250000, medium: 1000000, high: 2000000 };
/* a CRYSTAL23 ORBITALS grid (gui/orbitals.py): the reader's spacing */
export const ORBITAL = { draft: "preview", standard: "normal", high: "fine", publication: "fine" };
/* the wireframe's and the points' vertex spacing on a pore surface (Å), and a net's sphere detail (3Dmol's 2) */
export const WIRE_STEP = { preview: 1.2, draft: 0.9, standard: 0.6, high: 0.45, publication: 0.35 };
export const SPHERE = { preview: 1, draft: 1, standard: undefined, high: 3, publication: 4 };
export const KNOB_LABEL = {
  grid: ["Render grid", "Points per cell axis, at most"],
  upsample: ["Refine", "The render grid may be this many times finer than the analysis grid (pores) or the cube's (volumes)"],
  smooth: ["Smoothing", "Gaussian smoothing of the surface's field, in grid voxels"],
  max_triangles: ["Triangle cap", "The surfaces' triangle budget (a coarser grid beyond it)"],
  points: ["Grid points", "The cube's points sent to the view, at most"],
  smoothness: ["Smoothing passes", "Laplacian smoothing of the isosurface"],
};

/** {quality, <knob>...} of a spec, the knobs clamped into their limits (kind: "pores" | "volume"). */
export function knobs(kind, spec = {}) {
  const table = kind === "pores" ? PORES : VOLUME, lim = kind === "pores" ? PORE_LIMITS : VOLUME_LIMITS;
  const q = kind === "pores" ? porePresetOf(spec && spec.quality) : presetOf(spec && spec.quality), out = { quality: q, ...table[q] };
  for (const [k, [lo, hi]] of Object.entries(lim)) {
    const v = spec ? spec[k] : undefined;
    if (v === undefined || v === null || v === "" || !Number.isFinite(Number(v))) continue;
    out[k] = Math.min(hi, Math.max(lo, Number(v)));
    if (k !== "smooth") out[k] = Math.round(out[k]);
  }
  return out;
}
/** Same surfaces? (two pore specs or a spec and the quality a mesh was made at) */
export function sameMesh(a, b) {
  const x = knobs("pores", a || {}), y = knobs("pores", b || {});
  return Object.keys(PORE_LIMITS).every((k) => Math.abs(x[k] - y[k]) < 1e-9);
}
/** The volume request's {quality, points?, upsample?}: Standard keeps the Settings' Volume quality. */
export function volumeQuery(spec = {}, legacy = "medium") {
  const q = presetOf(spec.quality);
  const out = { quality: q === "standard" ? (legacy in LEGACY_POINTS ? legacy : "medium") : q === "draft" ? "low" : q };
  if (spec.points != null && spec.points !== "") out.points = knobs("volume", spec).points;
  if (spec.upsample != null && spec.upsample !== "") out.upsample = knobs("volume", spec).upsample;
  return out;
}
/** About how many triangles and bytes the surfaces take at ``spec``, from a mesh made at ``ref`` with ``tri`` triangles. */
export function estimate(spec, ref, tri) {
  const a = knobs("pores", spec), b = knobs("pores", ref || {});
  const g = (k) => Math.min(k.grid, 1e9);
  const t = Math.min(a.max_triangles, Math.round((tri || 0) * (g(a) / g(b)) ** 2));
  return { triangles: t, bytes: t * 28 };
}
export const fmtCount = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)} M` : n >= 1e3 ? `${Math.round(n / 1e3)} k` : String(n));

/* ---------------------------------------------------------------- style */
export const STYLES = [["solid", "Solid"], ["wireframe", "Wireframe"], ["edges", "Solid + edges"], ["points", "Points"]];
export const SHADINGS = [["smooth", "Smooth"], ["flat", "Flat"]];
// a pore surface: its far walls and faint near walls (as before), the far walls only, or both sides alike
export const PORE_SIDES = [["auto", "Far + faint near walls"], ["single", "Far walls only"], ["double", "Both sides"]];
export const SIDES = [["double", "Both sides"], ["single", "Front only"]];
const STYLE_IDS = STYLES.map((s) => s[0]);

/** {style, shading, sides} of an overlay's options (``wire`` and "wireframe" as before). */
export function look(o = {}, sides = "double") {
  let style = STYLE_IDS.includes(o.style) ? o.style : "solid";
  if (o.wire && style === "solid") style = "wireframe";
  return { style, shading: o.shading === "flat" ? "flat" : "smooth", sides: ["auto", "single", "double"].includes(o.sides) ? o.sides : sides };
}

/** A color a shade darker (edges over a surface of that color): "#rrggbb" or "rgb()" in, "#rrggbb" out. */
export function darker(c, k = 0.35) {
  const s = String(c || "").trim();
  let rgb = null;
  let m = /^#([0-9a-f]{6})$/i.exec(s);
  if (m) rgb = [0, 2, 4].map((i) => parseInt(m[1].slice(i, i + 2), 16));
  else if ((m = /^#([0-9a-f]{3})$/i.exec(s))) rgb = [...m[1]].map((x) => parseInt(x + x, 16));
  else if ((m = /rgba?\(\s*([\d.]+)[ ,]+([\d.]+)[ ,]+([\d.]+)/i.exec(s))) rgb = [m[1], m[2], m[3]].map(Number);
  if (!rgb) return s;              // a named or token color: kept as it is
  return "#" + rgb.map((x) => Math.round(x * (1 - k)).toString(16).padStart(2, "0")).join("");
}

// 3Dmol keeps a group's faces as 16-bit indices
const CHUNK = 60000;
/** Parts with every triangle on its own vertices and its face's normal (Flat shading), at most CHUNK vertices each. */
export function flatParts(parts) {
  const out = [];
  for (const p of parts) {
    let cur = null;
    const open = () => { cur = { ...p, vertexArr: [], normalArr: [], faceArr: [], colorArr: p.colorArr ? [] : undefined }; out.push(cur); };
    const V = p.vertexArr, N = p.normalArr, F = p.faceArr;
    for (let f = 0; f + 2 < F.length; f += 3) {
      if (!cur || cur.vertexArr.length > CHUNK - 3) open();
      const a = V[F[f]], b = V[F[f + 1]], c = V[F[f + 2]];
      let nx = (b.y - a.y) * (c.z - a.z) - (b.z - a.z) * (c.y - a.y);
      let ny = (b.z - a.z) * (c.x - a.x) - (b.x - a.x) * (c.z - a.z);
      let nz = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
      // the side the smooth normals point to
      const s = N && N.length ? [F[f], F[f + 1], F[f + 2]].reduce((t, i) => t + N[i].x * nx + N[i].y * ny + N[i].z * nz, 0) : 1;
      const l = (Math.hypot(nx, ny, nz) || 1) * (s < 0 ? -1 : 1);
      nx /= l; ny /= l; nz /= l;
      for (let k = 0; k < 3; k++) {
        const i = F[f + k];
        cur.faceArr.push(cur.vertexArr.length);
        cur.vertexArr.push(V[i]);
        cur.normalArr.push({ x: nx, y: ny, z: nz });
        if (cur.colorArr) cur.colorArr.push(p.colorArr[i]);
      }
    }
  }
  return out;
}
/** The vertices as small dots (tetrahedra of radius ``r`` Å), merged on a ``step`` Å grid, at most CHUNK vertices per part. */
export function pointParts(parts, step = 0.6, r = 0.1) {
  const T = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]].map((d) => d.map((x) => x / Math.sqrt(3)));
  const out = [];
  for (const p of parts) {
    const seen = new Set();
    let cur = null;
    const open = () => { cur = { ...p, vertexArr: [], normalArr: [], faceArr: [], colorArr: p.colorArr ? [] : undefined }; out.push(cur); };
    p.vertexArr.forEach((v, i) => {
      const key = `${Math.floor(v.x / step)},${Math.floor(v.y / step)},${Math.floor(v.z / step)}`;
      if (seen.has(key)) return;
      seen.add(key);
      if (!cur || cur.vertexArr.length > CHUNK - 4) open();
      const o = cur.vertexArr.length;
      for (const d of T) {
        cur.vertexArr.push({ x: v.x + r * d[0], y: v.y + r * d[1], z: v.z + r * d[2] });
        cur.normalArr.push({ x: d[0], y: d[1], z: d[2] });
        if (cur.colorArr) cur.colorArr.push(p.colorArr[i]);
      }
      cur.faceArr.push(o, o + 1, o + 2, o, o + 3, o + 1, o, o + 2, o + 3, o + 1, o + 3, o + 2);
    });
  }
  return out.filter((p) => p.faceArr.length);
}
/** A 3Dmol shape's triangles as parts (one per geometry group), with its vertex colors. */
export function partsOfShape(s) {
  const out = [];
  for (const g of (s && s.geo && s.geo.geometryGroups) || []) {
    const n = g.vertices, P = g.vertexArray, N = g.normalArray, C = g.colorArray, F = g.faceArray;
    if (!n || !F) continue;
    const vertexArr = [], normalArr = [], colorArr = [];
    for (let i = 0; i < n; i++) {
      vertexArr.push({ x: P[3 * i], y: P[3 * i + 1], z: P[3 * i + 2] });
      normalArr.push(N && N.length >= 3 * n ? { x: N[3 * i], y: N[3 * i + 1], z: N[3 * i + 2] } : { x: 0, y: 0, z: 1 });
      if (C) colorArr.push({ r: C[3 * i], g: C[3 * i + 1], b: C[3 * i + 2] });
    }
    out.push({ vertexArr, normalArr, faceArr: Array.from(F.subarray(0, g.faceidx)), colorArr: C ? colorArr : undefined });
  }
  return out;
}
/** One shape of ``parts`` (each a 3Dmol group); its own vertex colors when the parts have them. */
export function addParts(vw, parts, spec, after) {
  if (!parts.length) return null;
  const own = !!parts[0].colorArr;
  const one = (p) => ({ vertexArr: p.vertexArr, normalArr: p.normalArr, faceArr: p.faceArr, ...(own ? { colorArr: p.colorArr } : { color: p.color || spec.color }) });
  const s = vw.addCustom({ ...spec, ...one(parts[0]) });
  for (let i = 1; i < parts.length; i++) s.addCustom(one(parts[i]));
  // a shape color repaints every vertex: vertex colors keep their own
  if (own) s.color = undefined;
  return keepBuilt(s, after);
}
/** Shapes of ``parts`` in a look ({style, shading, sides}): ``color``, ``opacity``, ``step`` (points), ``wire`` (the
 * parts a wireframe draws, default the same), ``after`` (keepBuilt's callback for the filled shapes). */
export function styled(vw, parts, lk, { color, opacity = 1, step = 0.6, wire = null, after = null } = {}) {
  const $3 = window.$3Dmol, side = lk.sides === "single" ? $3.FrontSide : $3.DoubleSide, out = [];
  const add = (s, pass) => { if (s) { if (pass) s.macePass = pass; out.push(s); } };
  if (lk.style === "points") {
    add(addParts(vw, pointParts(parts, step), { color, opacity: Math.max(opacity, 0.85), side: $3.DoubleSide }, after));
    return out;
  }
  if (lk.style === "wireframe") {
    add(addParts(vw, wire || parts, { color, opacity, wireframe: true, side: $3.DoubleSide }));
    return out;
  }
  add(addParts(vw, lk.shading === "flat" ? flatParts(parts) : parts, { color, opacity, side }, after));
  if (lk.style === "edges") add(addParts(vw, (wire || parts).map((p) => ({ ...p, colorArr: undefined, color: undefined })),
    { color: darker(color), opacity: 1, wireframe: true, side: $3.DoubleSide }), "edges");
  return out;
}
