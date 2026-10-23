/* Grayscale print patterns (1.1.6): in a black-and-white print two elements of the same lightness
 * draw as the same gray. Settings › Viewport › Print patterns puts a procedural pattern on each atom's
 * visible surface (slashes, cross-hatching, stipples, ...) so every species stays distinct:
 *   off   none (the default)
 *   auto  only elements whose drawn colors are too close in lightness to tell apart (contrast < 1.35)
 *   all   every drawn element its own pattern
 *
 * 3Dmol draws spheres without textures, so the patterns are a 2D layer over each 3D view, drawn after
 * every paint (as the selection rings are): back to front, each atom's disc first clears what is
 * behind it and then gets its element's pattern, and each bond in front clears its band, so only the
 * visible part of an atom is patterned. Export Image and Export Animation draw the same layer at the
 * picture's size (overlayCanvas); the SVG export draws them as <pattern> tiles (svgPatterns).
 * `mace animate --patterns` draws the same patterns (mace/gui/animexport.py PATTERNS, the same order). */
import { luminance, ratio, rgbOf } from "./contrast.js";

export const PATTERNS = ["slash", "cross", "stipple", "backslash", "horizontal", "grid", "vertical", "dots"];
export const LABELS = { slash: "Slashes", cross: "Cross-hatching", stipple: "Stipples", backslash: "Back slashes",
  horizontal: "Horizontal lines", grid: "Grid", vertical: "Vertical lines", dots: "Dots" };
const SIMILAR = 1.35;           // two colors closer than this contrast ratio read as one gray
const MAX_ATOMS = 4000;         // more drawn atoms: no layer (the picture is too busy for patterns to read)

/** {element: pattern} for the drawn elements (most frequent first): "auto" patterns all but one element of
 * each group of similar lightness; "all" patterns every element. */
export function assign(symbols, colorOf, mode) {
  const out = {};
  if (mode !== "auto" && mode !== "all") return out;
  const count = new Map();
  for (const s of symbols) count.set(s, (count.get(s) || 0) + 1);
  const els = [...count.keys()].sort((a, b) => count.get(b) - count.get(a) || (a < b ? -1 : 1));
  let next = 0;
  if (mode === "all") { els.forEach((e) => { out[e] = PATTERNS[next++ % PATTERNS.length]; }); return out; }
  const col = new Map(els.map((e) => [e, colorOf(e)]));
  const done = new Set();
  for (const e of els) {
    if (done.has(e)) continue;
    // its group: every element as light as it (it keeps the plain surface: the most frequent of them)
    const group = els.filter((x) => !done.has(x) && ratio(col.get(x), col.get(e)) < SIMILAR);
    group.forEach((x, k) => { done.add(x); if (k) out[x] = PATTERNS[next++ % PATTERNS.length]; });
  }
  return out;
}

/** A tile of pattern `id` in `ink`, `px` device pixels a side. */
function tile(id, ink, px) {
  const c = document.createElement("canvas");
  c.width = c.height = px;
  const g = c.getContext("2d"), w = Math.max(1, px / 8);
  g.strokeStyle = ink; g.fillStyle = ink; g.lineWidth = w; g.lineCap = "square";
  const line = (x0, y0, x1, y1) => { g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke(); };
  const dot = (x, y, r) => { g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.fill(); };
  if (id === "slash" || id === "cross") { line(0, px, px, 0); line(-px / 2, px / 2, px / 2, -px / 2); line(px / 2, px * 1.5, px * 1.5, px / 2); }
  if (id === "backslash" || id === "cross") { line(0, 0, px, px); line(-px / 2, px / 2, px / 2, px * 1.5); line(px / 2, -px / 2, px * 1.5, px / 2); }
  if (id === "horizontal" || id === "grid") line(0, px / 2, px, px / 2);
  if (id === "vertical" || id === "grid") line(px / 2, 0, px / 2, px);
  if (id === "stipple") { dot(px * 0.25, px * 0.25, px * 0.09); dot(px * 0.75, px * 0.7, px * 0.09); }
  if (id === "dots") dot(px / 2, px / 2, px * 0.2);
  return c;
}
/** The pattern's ink on an atom of this color: dark on a light surface, light on a dark one. */
const inkOn = (color) => (luminance(color) > 0.35 ? "rgba(0,0,0,0.62)" : "rgba(255,255,255,0.78)");

/** The same patterns as SVG <pattern> definitions (Export Image › SVG): {defs, idOf: {element: id}} for a map of
 * assign(); each tile is `px` user units a side, drawn as tile() draws it (rgb + opacity: SVG 1.1 has no rgba). */
export function svgPatterns(map, colorOf, px = 7) {
  const defs = new Map(), idOf = {};
  const f = (x) => (Math.round(x * 100) / 100).toString();
  for (const [sym, id] of Object.entries(map)) {
    const light = luminance(colorOf(sym)) > 0.35;
    const key = `pp-${id}-${light ? "d" : "l"}`;
    idOf[sym] = key;
    if (defs.has(key)) continue;
    const ink = light ? 'stroke="rgb(0,0,0)" stroke-opacity="0.62"' : 'stroke="rgb(255,255,255)" stroke-opacity="0.78"';
    const fill = light ? 'fill="rgb(0,0,0)" fill-opacity="0.62"' : 'fill="rgb(255,255,255)" fill-opacity="0.78"';
    const w = Math.max(0.5, px / 8), parts = [];
    const line = (x0, y0, x1, y1) => parts.push(`<line x1="${f(x0)}" y1="${f(y0)}" x2="${f(x1)}" y2="${f(y1)}" ${ink} stroke-width="${f(w)}" stroke-linecap="square"/>`);
    const dot = (x, y, r) => parts.push(`<circle cx="${f(x)}" cy="${f(y)}" r="${f(r)}" ${fill}/>`);
    if (id === "slash" || id === "cross") { line(0, px, px, 0); line(-px / 2, px / 2, px / 2, -px / 2); line(px / 2, px * 1.5, px * 1.5, px / 2); }
    if (id === "backslash" || id === "cross") { line(0, 0, px, px); line(-px / 2, px / 2, px / 2, px * 1.5); line(px / 2, -px / 2, px * 1.5, px / 2); }
    if (id === "horizontal" || id === "grid") line(0, px / 2, px, px / 2);
    if (id === "vertical" || id === "grid") line(px / 2, 0, px / 2, px);
    if (id === "stipple") { dot(px * 0.25, px * 0.25, px * 0.09); dot(px * 0.75, px * 0.7, px * 0.09); }
    if (id === "dots") dot(px / 2, px / 2, px * 0.2);
    defs.set(key, `<pattern id="${key}" patternUnits="userSpaceOnUse" width="${f(px)}" height="${f(px)}">${parts.join("")}</pattern>`);
  }
  return { defs: [...defs.values()].join(""), idOf };
}

/** What a viewer draws now, in its own canvas fractions: [{x, y (0 to 1), r (fraction of the width), z, sym}]
 * and the bonds in front ({x1, y1, x2, y2, w, z}). Translucent atoms (selection shells, ghosts) are left out. */
export function sceneOf(v) {
  if (!v || !v.getModelList || !v.modelToScreen || !v.WIDTH || !v.HEIGHT) return null;
  const vdw = (window.$3Dmol && window.$3Dmol.GLModel && window.$3Dmol.GLModel.vdwRadii) || {};
  const atoms = [], bonds = [];
  for (const m of v.getModelList()) {
    if (!m || (m.isHidden && m.isHidden())) continue;
    const list = m.selectedAtoms({});
    for (const a of list) {
      const sp = a.style && a.style.sphere;
      if (!sp || sp.hidden || (sp.opacity !== undefined && sp.opacity < 0.9)) continue;
      const r = sp.radius !== undefined ? sp.radius : (vdw[a.elem] || vdw[String(a.elem).toUpperCase()] || 1.5) * (sp.scale !== undefined ? sp.scale : 1);
      atoms.push({ a, r, sym: a.elem });
    }
    for (const a of list) {
      const st = a.style && a.style.stick;
      if (!st || st.hidden || !a.bonds) continue;
      a.bonds.forEach((j) => { const b = list[j]; if (b && j > a.index) bonds.push({ a, b, w: 2 * (st.radius !== undefined ? st.radius : 0.25) }); });
    }
  }
  if (!atoms.length || atoms.length > MAX_ATOMS) return { atoms: [], bonds: [], skipped: atoms.length > MAX_ATOMS };
  // the projection, measured: three unit steps from the scene's middle give the screen's right and up
  // in model space (toward the viewer = right x up) and the scale, whatever the camera
  const off = v.canvasOffset ? v.canvasOffset() : { left: 0, top: 0 }, W = v.WIDTH, H = v.HEIGHT;
  const c = [0, 1, 2].map((k) => atoms.reduce((n, x) => n + [x.a.x, x.a.y, x.a.z][k], 0) / atoms.length);
  const P = v.modelToScreen([{ x: c[0], y: c[1], z: c[2] }, { x: c[0] + 1, y: c[1], z: c[2] }, { x: c[0], y: c[1] + 1, z: c[2] }, { x: c[0], y: c[1], z: c[2] + 1 }]);
  const right = [1, 2, 3].map((k) => P[k].x - P[0].x), down = [1, 2, 3].map((k) => P[k].y - P[0].y);
  const s = Math.sqrt((right.reduce((n, x) => n + x * x, 0) + down.reduce((n, x) => n + x * x, 0)) / 2);
  if (!(s > 0)) return { atoms: [], bonds: [] };
  const up = down.map((x) => -x);
  const toward = [right[1] * up[2] - right[2] * up[1], right[2] * up[0] - right[0] * up[2], right[0] * up[1] - right[1] * up[0]];
  const depth = (a) => a.x * toward[0] + a.y * toward[1] + a.z * toward[2];
  const sc = v.modelToScreen(atoms.map((x) => ({ x: x.a.x, y: x.a.y, z: x.a.z })));
  const out = atoms.map((x, k) => ({ x: (sc[k].x - off.left) / W, y: (sc[k].y - off.top) / H, r: (x.r * s) / W, z: depth(x.a), sym: x.sym }));
  const ends = bonds.length ? v.modelToScreen(bonds.flatMap((b) => [{ x: b.a.x, y: b.a.y, z: b.a.z }, { x: b.b.x, y: b.b.y, z: b.b.z }])) : [];
  const rOf = new Map(atoms.map((x) => [x.a, (x.r * s) / W]));
  const bout = bonds.map((b, k) => ({ x1: (ends[2 * k].x - off.left) / W, y1: (ends[2 * k].y - off.top) / H, x2: (ends[2 * k + 1].x - off.left) / W,
    y2: (ends[2 * k + 1].y - off.top) / H, w: (b.w * s) / W, z: (depth(b.a) + depth(b.b)) / 2, r1: rOf.get(b.a) || 0, r2: rOf.get(b.b) || 0 }));
  return { atoms: out, bonds: bout, aspect: H / W };
}

/** Draw the pattern layer of scene `S` into a 2D context of `width` x `height` device pixels. `scale`: device
 * pixels per CSS pixel of the view (the tiles keep their size on screen, and grow with an export). */
export function paint(g, S, map, { width, height, scale = 1, colorOf }) {
  g.save();
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.clearRect(0, 0, width, height);
  if (!S || !S.atoms.length || !Object.keys(map).length) { g.restore(); return 0; }
  const tiles = new Map(), px = Math.max(4, Math.round(7 * scale));
  const patternFor = (sym) => {
    const id = map[sym];
    if (!id) return null;
    const ink = inkOn(colorOf(sym)), key = `${id}|${ink}`;
    if (!tiles.has(key)) tiles.set(key, g.createPattern(tile(id, ink, px), "repeat"));
    return tiles.get(key);
  };
  const items = S.atoms.map((a) => ({ kind: "a", z: a.z, a })).concat(S.bonds.map((b) => ({ kind: "b", z: b.z + 1e-6, b })));
  items.sort((p, q) => p.z - q.z);
  let n = 0;
  for (const it of items) {
    if (it.kind === "b") {
      // a bond in front clears the band it covers
      // (only where it shows: between the two atoms' discs, inside which it is hidden by their surfaces)
      const b = it.b, x1 = b.x1 * width, y1 = b.y1 * height, x2 = b.x2 * width, y2 = b.y2 * height, L = Math.hypot(x2 - x1, y2 - y1);
      const t1 = (b.r1 * width * 0.95) / (L || 1), t2 = 1 - (b.r2 * width * 0.95) / (L || 1);
      if (!(L > 0) || t2 <= t1) continue;
      g.globalCompositeOperation = "destination-out";
      g.lineWidth = Math.max(1, b.w * width); g.lineCap = "butt";
      g.beginPath(); g.moveTo(x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1); g.lineTo(x1 + (x2 - x1) * t2, y1 + (y2 - y1) * t2); g.stroke();
      g.globalCompositeOperation = "source-over";
      continue;
    }
    const a = it.a, x = a.x * width, y = a.y * height, r = a.r * width;
    if (r < 1.5 || x + r < 0 || y + r < 0 || x - r > width || y - r > height) continue;
    g.globalCompositeOperation = "destination-out";
    g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.fill();
    g.globalCompositeOperation = "source-over";
    const p = patternFor(a.sym);
    if (!p) continue;
    g.save();
    g.beginPath(); g.arc(x, y, r * 0.97, 0, 2 * Math.PI); g.clip();
    g.translate(x, y);                     // the pattern sits on the atom (it moves with it)
    g.fillStyle = p; g.fillRect(-r, -r, 2 * r, 2 * r);
    g.restore();
    n++;
  }
  g.restore();
  return n;
}

const fallbackAtom = () => getComputedStyle(document.documentElement).getPropertyValue("--fallback-atom").trim();
const colorOf = (sym) => {
  const M = window.MACE;
  const c = M && M.elementColor ? M.elementColor(sym) : "";
  return rgbOf(c) ? c : fallbackAtom();
};
const modeNow = () => { const M = window.MACE, m = M && M.setting ? M.setting("viewport.printPatterns") : "off"; return m === "auto" || m === "all" ? m : "off"; };
/** {element: pattern} for a viewer now (the legend and the tests read it). */
export function mapFor(v, mode = modeNow()) {
  const S = sceneOf(v);
  return S ? assign(S.atoms.map((a) => a.sym), colorOf, mode) : {};
}

/** The layer as a canvas of width x height device pixels (Export Image, Export Animation), or null when off. */
export function overlayCanvas(v, width, height, mode = modeNow(), scale = 1) {
  if (mode === "off") return null;
  const S = sceneOf(v);
  if (!S || !S.atoms.length) return null;
  const map = assign(S.atoms.map((a) => a.sym), colorOf, mode);
  if (!Object.keys(map).length) return null;
  const c = document.createElement("canvas");
  c.width = width; c.height = height;
  const n = paint(c.getContext("2d"), S, map, { width, height, scale, colorOf });
  return n ? c : null;
}

/* ---------------------------------------------------------------- the live layer over each 3D view */
const layers = new Map();          // viewer id -> {cv, n, map}
function layerOf(v) {
  const cv0 = v.getCanvas(), wrap = cv0 && cv0.parentElement;
  if (!wrap) return null;
  let L = layers.get(v.viewerId);
  if (!L || !L.cv.isConnected) {
    const cv = document.createElement("canvas");
    cv.className = "vp-patterns";
    cv.setAttribute("aria-hidden", "true");
    cv.style.cssText = "position:absolute;pointer-events:none;z-index:1";
    wrap.appendChild(cv);
    L = { cv, n: 0, map: {} };
    layers.set(v.viewerId, L);
  }
  return L;
}
function clear(id) {
  const L = layers.get(id);
  if (!L) return;
  const g = L.cv.getContext("2d");
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.clearRect(0, 0, L.cv.width, L.cv.height);
  L.n = 0; L.map = {};
}
function repaint(id) {
  const M = window.MACE, v = M && M.state && M.state.viewers ? M.state.viewers[id] : null;
  const mode = modeNow();
  if (!v || mode === "off") { clear(id); return; }
  const L = layerOf(v);
  if (!L) return;
  // over the canvas, in its own px (the UI scale zooms both), at the screen's resolution (as the selection rings)
  const cv = v.getCanvas(), cr = cv.getBoundingClientRect(), wr = cv.parentElement.getBoundingClientRect();
  const z = window.maceZoom ? window.maceZoom.of(cv) : 1, dpr = (window.devicePixelRatio || 1) * z;
  L.cv.style.left = (cr.left - wr.left) / z + "px"; L.cv.style.top = (cr.top - wr.top) / z + "px";
  L.cv.style.width = cr.width / z + "px"; L.cv.style.height = cr.height / z + "px";
  const W = Math.round((cr.width * dpr) / z), Hh = Math.round((cr.height * dpr) / z);
  if (L.cv.width !== W || L.cv.height !== Hh) { L.cv.width = W; L.cv.height = Hh; }
  const S = sceneOf(v);
  L.map = S ? assign(S.atoms.map((a) => a.sym), colorOf, mode) : {};
  L.n = paint(L.cv.getContext("2d"), S, L.map, { width: W, height: Hh, scale: dpr, colorOf });
}
let installed = false;
/** Follow every 3D view's paints (idempotent). */
export function install() {
  if (installed) return;
  installed = true;
  document.addEventListener("mace-painted", (e) => { try { repaint(e.detail); } catch (er) { /* a viewer mid-teardown */ } });
}
/** What the layer over a view holds (the tests). */
export function stateOf(id) {
  const L = layers.get(id);
  return { mode: modeNow(), patterned: L ? L.n : 0, map: L ? { ...L.map } : {} };
}
if (typeof window !== "undefined") window.__macePatterns = { stateOf, mapFor, assign, PATTERNS };
