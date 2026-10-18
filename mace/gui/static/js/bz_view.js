/* The Brillouin zone view of the k-path editor: the first zone of the run's cell (from
 * GET api/bz, Crystal_d3/d3_bz.py), its reciprocal axes b1 b2 b3, the high-symmetry points
 * (labeled as on the band plot) and the path being edited, as arrows in the direction they run;
 * hidden segments are faint and dashed, a jump is a dashed connector. 3Dmol shapes in one small
 * viewer (window.MACE.viewer, so the UI scale and the theme are handled as for every 3D view);
 * drag to rotate, scroll to zoom.
 *
 * Linked both ways with the strip: the editor calls update() on every draw and highlight() on a
 * hover or a preview; the view reports the segment under the pointer (onHover) and a click on a
 * segment (onSelect) or on a point (onPick: in Calculation Setup it adds that point to the path). */
import * as K from "./kpath.js";

let serial = 0;

// the views alive now (each follows the theme until destroy())
let live = 0;
/** How many zone views follow the theme (for the tests: closed views must let go). */
export const listening = () => live;
// the tests reach it without importing the module again (a second URL would be a second module)
if (typeof window !== "undefined") window.__maceBz = { listening };

export function createBZ({ api, path, onHover, onSelect, onPick, pickable = false }) {
  const el = document.createElement("div");
  el.className = "kp-bz";
  const vid = `kp-bz-${++serial}`;
  el.innerHTML = `<div class="kp-bz-h"><span class="kp-sub">Brillouin zone</span><span class="kp-bz-note muted"></span></div>
    <div class="kp-bz-gl" id="${vid}" role="img" aria-label="The first Brillouin zone with the k-path"></div>
    ${pickable ? '<div class="kp-bz-pts" role="group" aria-label="Add a point of the zone"></div>' : ""}`;
  const note = (t) => { el.querySelector(".kp-bz-note").textContent = t; };
  let data = null, segs = [], hl = null, v = null, loading = null, failed = null, fitted = false;
  const M = () => window.MACE;
  const css = (n) => (M() && M().cssVar ? M().cssVar(n) : getComputedStyle(document.documentElement).getPropertyValue(n).trim());

  function load() {
    if (!loading) loading = Promise.all([api(`bz?path=${encodeURIComponent(path)}`), M().load3Dmol()]).then(([d]) => {
      data = d;
      draw();
    }).catch((e) => { failed = e.message || String(e); note(failed); });
    return loading;
  }
  function viewer() {
    if (v || !el.isConnected) return v;
    v = M().viewer(vid);
    if (v) { try { v.setHoverDuration && v.setHoverDuration(40); } catch (e) { /* older 3Dmol */ } }
    return v;
  }
  const P = (c) => ({ x: c[0], y: c[1], z: c[2] });
  const kOf = (s, end) => {
    const k = s[`k_${end}`];
    if (k && data) return K.canonical(s[end]) in data.points && data.points[K.canonical(s[end])].k.every((x, n) => Math.abs(x - k[n]) < 1e-6)
      ? data.points[K.canonical(s[end])].cart : k.map((_, n) => k[0] * data.bz.recip[0][n] + k[1] * data.bz.recip[1][n] + k[2] * data.bz.recip[2][n]);
    const p = data && data.points[K.canonical(s[end])];
    return p ? p.cart : null;
  };
  function draw() {
    if (!data || !viewer()) return;
    const bz = data.bz, V = bz.vertices;
    const span = Math.max(1e-6, ...V.map((p) => Math.hypot(...p)));
    const r = span * 0.011;
    v.removeAllShapes(); v.removeAllLabels();
    v.setBackgroundColor(css("--surface-2") || "#f7f8fa");
    const edge = css("--text-3"), text = css("--text"), accent = css("--accent"), warn = css("--warning"), faint = css("--border-strong");
    // the zone: its edges, its faces faintly
    for (const [i, j] of bz.edges) v.addCylinder({ start: P(V[i]), end: P(V[j]), radius: r * 0.6, color: edge, fromCap: 1, toCap: 1 });
    for (const f of bz.faces || []) {
      if (f.length < 3) continue;
      const vertexArr = f.map((i) => P(V[i])), faceArr = [];
      for (let k = 1; k + 1 < f.length; k++) faceArr.push(0, k, k + 1);
      try { v.addCustom({ vertexArr, normalArr: vertexArr.map(() => ({ x: 0, y: 0, z: 1 })), faceArr, color: accent, opacity: 0.05 }); } catch (e) { /* no faces */ }
    }
    // the reciprocal axes
    bz.recip.slice(0, bz.dim).forEach((b, n) => {
      const e = b.map((x) => x * 0.62);
      v.addArrow({ start: { x: 0, y: 0, z: 0 }, end: P(e), radius: r * 0.5, radiusRatio: 2.2, mid: 0.85, color: faint });
      v.addLabel(`b${"₁₂₃"[n]}`, { position: P(e.map((x) => x * 1.08)), fontSize: 11, fontColor: edge, backgroundOpacity: 0, inFront: true, showBackground: false });
    });
    // the path: arrows in the order and direction it runs
    const on = new Set(), hset = hl ? new Set(hl.indices) : null;
    let prev = null;
    segs.forEach((s, i) => {
      const a = kOf(s, "from"), b = kOf(s, "to");
      if (!a || !b) return;
      const lit = hset && hset.has(i) && hl.kind !== "keep", dim = hset && hl.kind === "keep" && !hset.has(i);
      if (s.hidden) {
        v.addCylinder({ start: P(a), end: P(b), radius: r * 0.7, color: lit ? warn : accent, dashed: true, dashLength: span * 0.03, gapLength: span * 0.025, opacity: lit ? 0.9 : 0.35,
          clickable: true, callback: () => onSelect && onSelect(i), hoverable: true, hover_callback: () => onHover && onHover(i), unhover_callback: () => onHover && onHover(null) });
        return;
      }
      if (prev && !K.joined(prev.s, s)) v.addCylinder({ start: P(prev.b), end: P(a), radius: r * 0.45, color: edge, dashed: true, dashLength: span * 0.025, gapLength: span * 0.02 });
      v.addArrow({ start: P(a), end: P(b), radius: lit ? r * 2.0 : r * 1.4, radiusRatio: 2.4, mid: 0.82, color: lit ? warn : accent, opacity: dim ? 0.25 : 1,
        clickable: true, callback: () => onSelect && onSelect(i), hoverable: true, hover_callback: () => onHover && onHover(i), unhover_callback: () => onHover && onHover(null) });
      on.add(K.canonical(s.from)); on.add(K.canonical(s.to));
      prev = { s, b };
    });
    // the points: those on the path labeled bold; the zone's other points small (a click adds one in Setup)
    for (const [lab, p] of Object.entries(data.points)) {
      const used = on.has(lab);
      v.addSphere({ center: P(p.cart), radius: used ? r * 2.6 : r * 1.8, color: p.inside === false ? warn : used ? text : faint,
        clickable: pickable, callback: pickable ? () => onPick && onPick(lab, p) : undefined });
      v.addLabel(K.plain(lab), { position: P(p.cart.map((x) => x + span * 0.02)), fontSize: used ? 15 : 12, fontColor: used ? text : edge,
        backgroundOpacity: 0, showBackground: false, inFront: true, bold: used });
    }
    // fitted at the size it is shown at (the panel can be drawn before it has its width)
    const gl = el.querySelector(".kp-bz-gl"), size = `${gl.clientWidth}x${gl.clientHeight}`;
    if (gl.clientWidth > 0 && fitted !== size) { v.resize(); fit(); fitted = size; }
    v.render();
    const missing = segs.filter((s) => !kOf(s, "from") || !kOf(s, "to")).length;
    const out = Object.entries(data.points).filter(([lab, p]) => on.has(lab) && p.inside === false).map(([lab]) => K.plain(lab));
    note(`${data.output}${missing ? ` · ${missing} segment${missing === 1 ? "" : "s"} without coordinates not drawn` : ""}` +
      `${out.length ? ` · ${out.join(", ")} outside the first zone (the same states one reciprocal vector over)` : ""}`);
    const row = el.querySelector(".kp-bz-pts");
    if (row && !row.childElementCount) {
      row.innerHTML = `<span class="kp-sub">Add a point:</span>` + Object.entries(data.points).map(([lab, p]) =>
        `<button type="button" data-bz-pick="${lab}" data-tip="Add a segment to ${K.plain(lab)} (${p.k.map((x) => +x.toFixed(4)).join(" ")}); a click on the point in the zone does the same">${K.display(lab)}</button>`).join("");
      row.addEventListener("click", (e) => { const b = e.target.closest("[data-bz-pick]"); if (b && onPick) onPick(b.dataset.bzPick, data.points[b.dataset.bzPick]); });
    }
  }
  function fit() {
    const pts = data.bz.vertices.concat(Object.values(data.points).map((p) => p.cart));
    // turned first (c* tilted toward the viewer), then fitted: the zone sits in the middle of the view
    v.setView([0, 0, 0, 0, 0, 0, 0, 1]);                  // the start: look down z, nothing turned
    v.rotate(-62, "x"); v.rotate(28, "y");
    const m = v.addModel();
    m.addAtoms(pts.map((p, k) => ({ elem: "H", x: p[0], y: p[1], z: p[2], index: k, bonds: [] })));
    m.setStyle({}, {});
    v.zoomTo({ model: m });
    v.removeModel(m);
    v.zoom(0.85);
  }
  const api_ = {
    el,
    /** The path as the editor shows it (hidden segments included, their index is the strip's). */
    update(nextSegs) { segs = nextSegs || []; if (!data && !failed) load(); else if (data) { if (!el.isConnected) return; viewer(); draw(); } },
    highlight(indices, kind = "hover") { const next = indices ? { indices, kind } : null; if (JSON.stringify(next) === JSON.stringify(hl)) return; hl = next; draw(); },
    attach(slot) { if (!slot) return; slot.appendChild(el); if (v) { v.resize(); v.render(); } if (data) requestAnimationFrame(() => draw()); },
    redraw() { draw(); },
    // what a pointer on a segment or a point of the zone does (the tests drive the link with these)
    hoverSegment(i) { if (onHover) onHover(i); },
    selectSegment(i) { if (onSelect) onSelect(i); },
    pickPoint(lab) { if (onPick && data && data.points[lab]) onPick(lab, data.points[lab]); },
    state: () => ({ loaded: !!data, failed, segments: segs.length, hl, points: data ? Object.keys(data.points) : [], viewer: !!v }),
    /** Its document closed (or Setup made a new one): stop following the theme, let the viewer go. */
    destroy() {
      if (gone) return;
      gone = true;
      live--;
      document.removeEventListener("mace-theme", onTheme);
      el.remove();
      try { if (v && v.clear) v.clear(); } catch (e) { /* already gone */ }
      v = null; data = null;
    },
  };
  let gone = false;
  const onTheme = () => { if (data && el.isConnected) draw(); };
  document.addEventListener("mace-theme", onTheme);
  live++;
  el.__bz = api_;            // the tests reach the view through its element
  return api_;
}
