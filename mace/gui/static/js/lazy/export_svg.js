/* SVG export (plan B10), loaded on first use: MACE's own orthographic
 * projector. Every drawn atom becomes one <circle> (the same atoms the
 * viewer shows: the display range, boundary atoms and images included),
 * every drawn bond two <line> halves in their atoms' colors, the cell its
 * twelve edges; painted back to front by depth along the view direction.
 * The projection is the viewer's own (camera.calibrate, fitted to 3Dmol's
 * modelToScreen), so the picture lines up with the screen. Perspective is
 * refused: an SVG circle cannot stand for a perspective sphere honestly.
 * Large structures are written in chunks of 2000 atoms, yielding between
 * them, so the page never blocks for long. No external references. */
import * as G from "../geom/core.js";

const CHUNK = 2000;
const yieldNow = () => new Promise((r) => setTimeout(r, 0));
const f1 = (x) => (Math.round(x * 10) / 10).toString();
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

export const refuseReason = (projection) => (projection === "perspective" ? "SVG export is orthographic; press 5" : "");

function shade(hex, k) {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(hex).trim());
  if (!m) return String(hex);          // not a #rrggbb color: drawn as given
  const n = parseInt(m[1], 16);
  // k > 1 lightens toward white, k < 1 darkens
  return "#" + [n >> 16 & 255, n >> 8 & 255, n & 255].map((v) => Math.round(k > 1 ? v + (255 - v) * (k - 1) : v * k)
    .toString(16).padStart(2, "0")).join("");
}

/** One SVG from groups of {id, dx, atoms:[{u, v, r, z, fill, opacity, title}], lines:[{u1, v1, u2, v2, z, stroke, width, opacity}]}. */
/** Shaded (plan E9): one radial gradient per atom color, lit from the upper left; Flat (the default) keeps
 * plain fills. Fog and ambient occlusion stay raster only. */
function gradients(groups) {
  const colors = new Set();
  for (const g of groups) for (const a of g.atoms) colors.add(a.fill);
  const defs = [...colors].filter((c) => /^#?[0-9a-f]{6}$/i.test(String(c).trim())).map((c) => {
    const id = gradId(c);
    return `<radialGradient id="${id}" cx="0.35" cy="0.35" r="0.75" fx="0.3" fy="0.3"><stop offset="0" stop-color="${shade(c, 1.35)}"/>` +
      `<stop offset="0.55" stop-color="${c}"/><stop offset="1" stop-color="${shade(c, 0.55)}"/></radialGradient>`;
  });
  return defs.length ? `<defs>${defs.join("")}</defs>` : "";
}
const gradId = (c) => "g" + String(c).replace(/[^0-9a-f]/gi, "").toLowerCase();

export async function svgDocument({ width, height, background, groups, title = "MACE structure", shading = "flat" }) {
  const shaded = shading === "shaded";
  const out = [`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    `<title>${esc(title)}</title>`];
  if (shaded) out.push(gradients(groups));
  if (background) out.push(`<rect width="100%" height="100%" fill="${esc(background)}"/>`);
  for (const g of groups) {
    out.push(`<g id="${esc(g.id)}"${g.dx ? ` transform="translate(${f1(g.dx)} 0)"` : ""}>`);
    // lines and atoms painted together, far to near (z grows toward the viewer)
    const items = [];
    for (const l of g.lines) items.push([l.z, 0, l]);
    for (const a of g.atoms) items.push([a.z, 1, a]);
    items.sort((p, q) => p[0] - q[0] || p[1] - q[1]);
    for (let k = 0; k < items.length; k++) {
      const [, kind, o] = items[k];
      if (kind) out.push(`<circle cx="${f1(o.u)}" cy="${f1(o.v)}" r="${f1(Math.max(0.5, o.r))}" fill="${shaded && /^#?[0-9a-f]{6}$/i.test(String(o.fill).trim()) ? `url(#${gradId(o.fill)})` : o.fill}" stroke="${shade(o.fill, 0.55)}" stroke-width="${f1(Math.max(0.4, o.r * 0.06))}"${o.opacity < 1 ? ` fill-opacity="${o.opacity}" stroke-opacity="${o.opacity}"` : ""}>${o.title ? `<title>${esc(o.title)}</title>` : ""}</circle>`);
      else out.push(`<line x1="${f1(o.u1)}" y1="${f1(o.v1)}" x2="${f1(o.u2)}" y2="${f1(o.v2)}" stroke="${o.stroke}" stroke-width="${f1(o.width)}" stroke-linecap="round"${o.opacity < 1 ? ` stroke-opacity="${o.opacity}"` : ""}/>`);
      if (k % (2 * CHUNK) === 2 * CHUNK - 1) await yieldNow();
    }
    out.push("</g>");
  }
  out.push("</svg>");
  return out.join("\n");
}

/** The structure viewer's picture as one group: X = {scene, camera, M, viewState, setting}. */
export async function sceneGroup(X, { cell = true } = {}) {
  const sc = X.scene, d = sc && sc.display, doc = sc && sc.doc;
  if (!d || !doc || !doc.structure) throw new Error("No structure is drawn");
  const s = doc.structure, cal = X.camera.calibrate(), P = cal.P, k = cal.s;
  const uv = G.projectAll(P, d.cart), T = cal.toward;
  const depth = (x, y, z) => x * T[0] + y * T[1] + z * T[2];
  const atoms = new Array(d.n), lines = [];
  const R = sc.drawnRadii, colorOf = new Map();
  const fill = (sym) => { if (!colorOf.has(sym)) colorOf.set(sym, X.M.elementColor(sym)); return colorOf.get(sym); };
  // the Magnetic style colors each atom by its spin moment (a colorfunc of its site), as the view does
  const spec = X.M.colorSpec ? X.M.colorSpec() : null, byAtom = spec && spec.colorfunc;
  for (let i = 0; i < d.n; i++) {
    const b = d.base[i], sym = s.symbols[b];
    const f = byAtom ? byAtom({ serial: b + 1, properties: { base: b } }) : fill(sym);
    atoms[i] = { u: uv[2 * i], v: uv[2 * i + 1], r: R[i] * k, z: depth(d.cart[3 * i], d.cart[3 * i + 1], d.cart[3 * i + 2]), fill: f, opacity: 1, title: `${sym}${b + 1}` };
    if (i % CHUNK === CHUNK - 1) await yieldNow();
  }
  const p = sc.pairs, bw = 2 * (Number(X.setting("bondRadius")) || 0.12) * k;
  if (p) for (let q = 0; q < p.length; q += 2) {
    const a = p[q], b = p[q + 1], A = atoms[a], B = atoms[b];
    const mu = (A.u + B.u) / 2, mv = (A.v + B.v) / 2, mz = (A.z + B.z) / 2;
    lines.push({ u1: A.u, v1: A.v, u2: mu, v2: mv, z: (A.z + mz) / 2 - 1e-3, stroke: A.fill, width: bw, opacity: 1 });
    lines.push({ u1: mu, v1: mv, u2: B.u, v2: B.v, z: (B.z + mz) / 2 - 1e-3, stroke: B.fill, width: bw, opacity: 1 });
  }
  if (cell && X.viewState.cell) {
    const rg = X.range;
    const corners = G.boxCorners(s, Array.isArray(rg) ? { lo: [0, 0, 0], hi: rg } : rg);
    const cu = G.projectAll(P, Float64Array.from(corners.flat()));
    const color = X.M.cssVar("--cell");
    for (const [a, b] of [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]]) {
      lines.push({ u1: cu[2 * a], v1: cu[2 * a + 1], u2: cu[2 * b], v2: cu[2 * b + 1], z: (depth(...corners[a]) + depth(...corners[b])) / 2,
        stroke: color, width: Math.max(1, 0.06 * k), opacity: 1 });
    }
  }
  return { id: "structure", atoms, lines };
}

/** Atoms of a 3Dmol viewer (the comparison panes): [{x, y, z, r(Å), fill, opacity, title}] projected with modelToScreen. */
export function viewerGroup(v, id, items, rect) {
  const atoms = items.map((a) => {
    const p = v.modelToScreen({ x: a.x, y: a.y, z: a.z });
    const q = v.modelToScreen({ x: a.x + a.r * a.right[0], y: a.y + a.r * a.right[1], z: a.z + a.r * a.right[2] });
    return { u: p.x - rect.left, v: p.y - rect.top, r: Math.hypot(q.x - p.x, q.y - p.y), z: a.depth, fill: a.fill, opacity: a.opacity, title: a.title };
  });
  return { id, atoms, lines: [] };
}
