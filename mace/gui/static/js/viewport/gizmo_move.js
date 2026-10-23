/* The Move tool's gizmo (spec §2.1, 1.1.6): an AutoCAD-style 3D Move handle
 * at the selection's centroid, drawn while the Move tool is active on an
 * editable structure with atoms selected.
 *   Three arrows (x red, y green, z blue; or the cell's a, b, c, picked in the
 *     panel): drag one to move along that axis only, exactly under the
 *     camera's projection (a perspective view too).
 *   Three squares between the arrows: drag one to move in that plane only.
 *   The center ring: move in the screen plane (the plain grab).
 *   Ctrl while dragging snaps: 0.05 of a cell axis (cell frame), 0.1 Å (Cartesian).
 * The panel takes an exact displacement, Δx Δy Δz in Å or Δa Δb Δc as fractions
 * of the cell axes: the atoms preview it as typed, Apply (or Enter) moves them,
 * one undo step. The grab itself (G, X/Y/Z, typed offsets, Sym mode, the live
 * space group) stays js/editor/tools.js's. */
import * as G from "../geom/core.js";
import * as M from "./gizmo_math.js";
import { haloText, vpColor, gizmoPanel } from "./gizmo_overlay.js";
import * as profile from "../profile.js";

const HIT_PX = 7, ARROW_PX = 78, CENTER_PX = 9;
const COLORS = ["--axis-a", "--axis-b", "--axis-c"];
export const CART_SNAP = 0.1, CELL_SNAP = 0.05;

export function createMoveGizmo(c) {
  const { V } = c;
  let frame = profile.ui("moveFrame", "cart") === "cell" ? "cell" : "cart", hover = null, panel = null;
  const canvasEl = () => document.getElementById("viewer-main");
  const font = (w = 700, px = 12) => `${w} ${px}px ${vpColor("--font-sans", "system-ui, sans-serif")}`;

  /** A usable lattice (three independent periodic vectors), else the cell frame falls back to Cartesian. */
  function cellOf(s) {
    const pbc = s.pbc || [true, true, true], C = s.cell;
    if (!C || !pbc.every(Boolean)) return null;
    const vol = G.dot3(C[0], G.cross3(C[1], C[2]));
    return Math.abs(vol) > 1e-9 ? C : null;
  }
  const frameOf = (s) => (frame === "cell" && cellOf(s) ? "cell" : "cart");
  /** The frame's unit axes and their names. */
  function axesOf(s) {
    if (frameOf(s) === "cell") return s.cell.map((r, k) => ({ key: "abc"[k], e: G.unit3(r), len: G.norm3(r), col: COLORS[k] }));
    return [[1, 0, 0], [0, 1, 0], [0, 0, 1]].map((e, k) => ({ key: "xyz"[k], e, len: 1, col: COLORS[k] }));
  }

  let wholeCache = null;
  /** The selection made whole (cached until the structure or the selection changes). */
  function wholeOf(e) {
    const key = `${c.selVersion()}:${e.selection.size}`;
    if (!wholeCache || wholeCache.s !== e.structure || wholeCache.key !== key) wholeCache = { s: e.structure, key, w: M.wholeSelection(e.structure, e.selection) };
    return wholeCache.w;
  }
  /** The handles on screen for editable doc e, the atoms displaced by d (Å) so far. */
  function handles(e, d = [0, 0, 0]) {
    const cam = V.cam(), el = canvasEl();
    if (!cam || !el || !e || !e.selection.size) return null;
    const cal = cam.calibrate(), s = e.structure;
    const whole = wholeOf(e), o = G.add3(whole.pivot, d);
    const ps = M.screen(cal.P, o);
    const ppa = Math.abs(M.screen(cal.P, G.add3(o, cal.right))[0] - ps[0]) || cal.s;
    const L = ARROW_PX / ppa;
    const axes = axesOf(s).map((a) => {
      const tip = M.screen(cal.P, G.add3(o, G.scale3(a.e, L)));
      return { ...a, kind: "axis", tip: [tip[0], tip[1]], base: [ps[0], ps[1]] };
    });
    const planes = [[0, 1, 2], [1, 2, 0], [2, 0, 1]].map(([i, j, k]) => {
      const a = 0.26 * L, b = 0.24 * L, e1 = axes[i].e, e2 = axes[j].e;
      const corner = (u, v) => { const p = M.screen(cal.P, G.add3(o, G.add3(G.scale3(e1, u), G.scale3(e2, v)))); return [p[0], p[1]]; };
      return { key: axes[i].key + axes[j].key, kind: "plane", e1, e2, col: axes[k].col, i, j,
        quad: [corner(a, a), corner(a + b, a), corner(a + b, a + b), corner(a, a + b)] };
    });
    return { cal, o, origin: whole.pivot, ps: [ps[0], ps[1]], L, axes, planes, frame: frameOf(s) };
  }
  function hit(h, x, y) {
    if (!h) return null;
    if (Math.hypot(x - h.ps[0], y - h.ps[1]) <= CENTER_PX) return { kind: "free", key: "free" };
    let best = null, bd = HIT_PX;
    for (const a of h.axes) {
      const dx = a.tip[0] - a.base[0], dy = a.tip[1] - a.base[1];
      const from = [a.base[0] + 0.2 * dx, a.base[1] + 0.2 * dy], to = [a.tip[0] + 0.12 * dx, a.tip[1] + 0.12 * dy];
      const dd = M.segDist(x, y, from, to);
      if (dd < bd) { bd = dd; best = a; }
    }
    if (best) return best;
    for (const p of h.planes) if (M.inQuad(x, y, p.quad)) return p;
    return null;
  }
  /** The displacement (Å) of a handle drag from (x0, y0) to (x, y), the handle's origin o; snap: Ctrl held. */
  function dragDelta(handle, e, o, x0, y0, x, y, snap) {
    const cal = V.cam().calibrate(), s = e.structure, cell = frameOf(s) === "cell";
    if (handle.kind === "axis") {
      let t = M.axisDrag(cal.P, o, handle.e, cal.toward, x0, y0, x, y);
      if (snap) t = cell ? Math.round(t / handle.len / CELL_SNAP) * CELL_SNAP * handle.len : Math.round(t / CART_SNAP) * CART_SNAP;
      return G.scale3(handle.e, t);
    }
    if (handle.kind === "plane") {
      // the plane's two axes (not orthogonal in a skewed cell): solve on the plane they span
      const [e1, e2] = [handle.e1, handle.e2];
      const h0 = M.planeHit(cal.P, o, e1, e2, x0, y0), h1 = M.planeHit(cal.P, o, e1, e2, x, y);
      if (!h0 || !h1) return null;
      let u = h1[0] - h0[0], v = h1[1] - h0[1];
      if (snap) {
        const ax = axesOf(s);
        const st = (t, a) => (cell ? Math.round(t / a.len / CELL_SNAP) * CELL_SNAP * a.len : Math.round(t / CART_SNAP) * CART_SNAP);
        u = st(u, ax[handle.i]); v = st(v, ax[handle.j]);
      }
      return G.add3(G.scale3(e1, u), G.scale3(e2, v));
    }
    return null;
  }

  /* ---------------------------------------------------------------- drawing */
  function draw(g, e, state = {}) {
    const h = handles(e, state.d || [0, 0, 0]);
    if (!h) return false;
    const bg = vpColor("--viewport-bg"), text = vpColor("--text"), accent = vpColor("--accent");
    const act = state.active ? state.active.key : null, hv = hover ? hover.key : null;
    for (const p of h.planes) {
      const on = act === p.key || (!act && hv === p.key);
      g.save();
      g.fillStyle = vpColor(p.col, vpColor("--fallback-atom")); g.strokeStyle = g.fillStyle;
      g.globalAlpha = on ? 0.55 : act ? 0.12 : 0.28;
      g.beginPath(); p.quad.forEach(([x, y], k) => (k ? g.lineTo(x, y) : g.moveTo(x, y))); g.closePath(); g.fill();
      g.globalAlpha = on ? 1 : 0.6; g.lineWidth = 1; g.stroke();
      g.restore();
    }
    for (const a of h.axes) {
      const on = act === a.key || (!act && hv === a.key), col = vpColor(a.col, vpColor("--fallback-atom"));
      arrow(g, a.base, a.tip, col, on ? 4 : 2.5, act && !on ? 0.4 : 1);
      const dx = a.tip[0] - a.base[0], dy = a.tip[1] - a.base[1], n = Math.hypot(dx, dy) || 1;
      haloText(g, a.key, a.tip[0] + (dx / n) * 12, a.tip[1] + (dy / n) * 12, col, bg, { font: font(700, 12) });
    }
    g.save();
    g.strokeStyle = act === "free" || (!act && hv === "free") ? accent : text;
    g.lineWidth = act === "free" || (!act && hv === "free") ? 3 : 2;
    g.beginPath(); g.arc(h.ps[0], h.ps[1], CENTER_PX - 2, 0, 2 * Math.PI); g.stroke();
    g.restore();
    // the displacement so far, from where the atoms were
    if (state.d && G.norm3(state.d) > 1e-9) {
      const p0 = M.screen(h.cal.P, h.origin);
      g.save(); g.setLineDash([4, 4]); g.strokeStyle = text; g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(p0[0], p0[1]); g.lineTo(h.ps[0], h.ps[1]); g.stroke(); g.restore();
      const s = e.structure, comps = frameOf(s) === "cell" ? G.fracOf(s.cell, state.d) : state.d, names = frameOf(s) === "cell" ? "abc" : "xyz";
      const unit = frameOf(s) === "cell" ? "" : " Å";
      const txt = [0, 1, 2].map((k) => `Δ${names[k]} ${G.fmtNum(comps[k], frameOf(s) === "cell" ? 4 : 3)}${unit}`).join("  ");
      haloText(g, txt, h.ps[0] + 16, h.ps[1] + 22, text, bg, { align: "left", font: font(600, 12) });
    }
    return true;
  }
  function arrow(g, a, b, col, w, alpha) {
    const dx = b[0] - a[0], dy = b[1] - a[1], n = Math.hypot(dx, dy);
    g.save();
    g.globalAlpha = alpha; g.strokeStyle = col; g.fillStyle = col; g.lineWidth = w; g.lineCap = "round";
    if (n < 4) { g.beginPath(); g.arc(b[0], b[1], 4, 0, 2 * Math.PI); g.fill(); g.restore(); return; }   // seen end-on
    const ux = dx / n, uy = dy / n, head = Math.min(12, 0.3 * n);
    g.beginPath(); g.moveTo(a[0] + ux * CENTER_PX, a[1] + uy * CENTER_PX); g.lineTo(b[0] - ux * head * 0.8, b[1] - uy * head * 0.8); g.stroke();
    g.beginPath(); g.moveTo(b[0], b[1]);
    g.lineTo(b[0] - ux * head - uy * head * 0.45, b[1] - uy * head + ux * head * 0.45);
    g.lineTo(b[0] - ux * head + uy * head * 0.45, b[1] - uy * head - ux * head * 0.45);
    g.closePath(); g.fill();
    g.restore();
  }

  /* ---------------------------------------------------------------- the panel: an exact displacement */
  function renderPanel(e, state = {}) {
    const host = c.panelHost();
    if (!host) return;
    if (!panel || !panel.isConnected) {
      panel = gizmoPanel("move", "Move by", "Drag an arrow to move along it only, a square to move in its plane only, the center ring "
        + "freely · Ctrl (held): snap (0.05 of a cell axis, 0.1 Å) · or type a displacement and press Enter · G, X Y Z and typed offsets "
        + "work as before · Esc or a right click cancels", profile);
      panel.querySelector(".gz-body").innerHTML = `<div class="gz-row"><span class="segment gz-frame" role="radiogroup" aria-label="Displacement frame">
            <button type="button" role="radio" data-f="cart" title="Δx Δy Δz in Å">Cartesian</button><button type="button" role="radio" data-f="cell" title="Δa Δb Δc as fractions of the cell axes">Cell</button></span></div>
        ${[0, 1, 2].map((k) => `<div class="gz-row"><label class="gz-f"><span class="gz-n" data-k="${k}">Δ${"xyz"[k]}</span><input class="gz-d" data-k="${k}" inputmode="decimal" spellcheck="false" autocomplete="off" placeholder="0"><span class="gz-u gz-unit">Å</span></label></div>`).join("")}
        <div class="gz-row"><button type="button" class="gz-btn primary" data-act="apply">Apply</button>
          <button type="button" class="gz-btn" data-act="clear" title="Clear the fields and the preview">Clear</button></div>
        <span class="gz-err" role="alert"></span>`;
      host.appendChild(panel);
      wire();
    }
    const s = e ? e.structure : null, f = s ? frameOf(s) : "cart";
    panel.querySelectorAll(".gz-frame button").forEach((b) => {
      b.setAttribute("aria-checked", String(b.dataset.f === f));
      b.disabled = b.dataset.f === "cell" && !(s && cellOf(s));
    });
    panel.querySelectorAll(".gz-n").forEach((n) => { n.textContent = `Δ${(f === "cell" ? "abc" : "xyz")[+n.dataset.k]}`; });
    panel.querySelectorAll(".gz-unit").forEach((u) => { u.textContent = f === "cell" ? "" : "Å"; });
    panel.querySelectorAll(".gz-d").forEach((i) => i.setAttribute("aria-label", `Displacement ${f === "cell" ? "Δ" + "abc"[+i.dataset.k] + " (fraction)" : "Δ" + "xyz"[+i.dataset.k] + " (Å)"}`));
    // a pointer grab under way shows its displacement in the fields (not while they are typed in)
    if (state.d && !panel.contains(document.activeElement) && !state.typed) {
      const comps = f === "cell" ? G.fracOf(s.cell, state.d) : state.d;
      panel.querySelectorAll(".gz-d").forEach((i) => { i.value = Math.abs(comps[+i.dataset.k]) > 1e-12 ? G.fmtNum(comps[+i.dataset.k], f === "cell" ? 5 : 4) : ""; });
    } else if (!state.d && !panel.contains(document.activeElement)) panel.querySelectorAll(".gz-d").forEach((i) => { i.value = ""; });
  }
  /** The typed displacement: {vec, frame} (Å or fractions), null when every field is blank, or {error}. */
  function typedVec() {
    const vals = [...panel.querySelectorAll(".gz-d")].map((i) => i.value.trim());
    if (vals.every((v) => v === "")) return null;
    const out = [];
    for (let k = 0; k < 3; k++) {
      if (vals[k] === "") { out.push(0); continue; }
      const r = G.evalExpr(vals[k]);
      if (!r.ok || !Number.isFinite(r.value)) return { error: `Δ${"xyz"[k]}: ${r.error || "not a number"}` };
      out.push(r.value);
    }
    return { vec: out, frame };
  }
  function wire() {
    const err = panel.querySelector(".gz-err");
    panel.querySelectorAll(".gz-frame button").forEach((b) => b.addEventListener("click", () => {
      frame = b.dataset.f; profile.setUi("moveFrame", frame);
      panel.querySelectorAll(".gz-d").forEach((i) => { i.value = ""; });
      c.typed(null);
      c.refresh();
    }));
    const changed = () => {
      const t = typedVec();
      err.textContent = t && t.error ? t.error : "";
      panel.querySelectorAll(".gz-d").forEach((i) => i.removeAttribute("aria-invalid"));
      if (t && t.error) return;
      c.typed(t);
    };
    panel.querySelectorAll(".gz-d").forEach((i) => {
      i.addEventListener("input", changed);
      i.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter") { ev.preventDefault(); ev.stopPropagation(); changed(); c.apply(); }
        else if (ev.key === "Escape") { ev.preventDefault(); ev.stopPropagation(); clear(); V.focusViewport(); }
      });
    });
    panel.querySelector('[data-act="apply"]').addEventListener("click", () => { changed(); c.apply(); });
    panel.querySelector('[data-act="clear"]').addEventListener("click", clear);
  }
  function clear() {
    if (!panel) return;
    panel.querySelectorAll(".gz-d").forEach((i) => { i.value = ""; });
    panel.querySelector(".gz-err").textContent = "";
    c.typed(null);
  }

  return {
    handles, hit, dragDelta, draw, renderPanel, clear,
    setHover(h) { const k = h ? h.key : null; if (k !== (hover ? hover.key : null)) { hover = h; return true; } return false; },
    get frame() { return frame; }, setFrame(f) { frame = f === "cell" ? "cell" : "cart"; profile.setUi("moveFrame", frame); },
    get panel() { return panel; },
    /** The handles' screen points, for the tests. */
    screenHandles(e) {
      const h = handles(e);
      return h && { center: h.ps, axes: Object.fromEntries(h.axes.map((a) => [a.key, { base: a.base, tip: a.tip }])),
        planes: Object.fromEntries(h.planes.map((p) => [p.key, p.quad])), frame: h.frame };
    },
  };
}
