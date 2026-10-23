/* The Rotate tool (R; spec §2.1, 1.1.6): an AutoCAD-style 3D Rotate gizmo.
 *
 * While the tool is active, compass wheels sit on the viewport: one wheel per
 * Cartesian axis (x red, y green, z blue, drawn in 3D: the far half faint) and
 * an outer wheel in the screen plane. What turns is the Target: the selected
 * atoms (an edit, about the Pivot) when the structure is editable and atoms
 * are selected, else the view (about its center), or as picked in the panel.
 * The Pivot (1.1.6, the panel; Settings › Viewport › Rotate pivot is the
 * default, each document keeps its own): the selection's centroid, its center
 * of mass, the cell's center, the origin or a picked atom (Pick, then click an
 * atom). A marker shows it and the wheels sit on it (gizmo_math.js pivotPoint).
 *   Drag a wheel: turn about that axis only (the outer wheel: in-plane, about
 *     the line of sight). Drag anywhere else: a free, out-of-plane turn (a
 *     horizontal drag about the screen's vertical, a vertical drag about its
 *     horizontal; in Turntable mode about the structure's up).
 *   The angles are measured from the orientation the drag began at; snapping
 *     is to whole steps of that reference (Settings › Viewport › Angle snap,
 *     Custom angle; the panel's Snap and Step).
 *   Ctrl (⌘): invert the snap while held (a snapping drag turns freely, a free one snaps).
 *   Shift: snap every angle (≈88°, ≈43° -> 90°, 45°).
 *   Alt: keep one axis only, snapped: the larger turn (-> 90°, 0°); pressing Alt again (or Tab) switches to the
 *     other (-> 0°, 45°). (Not Alt+Tab: the system's window switcher.)
 *   Shift+Alt: in-plane only (a free drag turns about the line of sight, by the pointer's sweep about the center).
 *   Esc or a right click cancels the drag; letting go applies it (one undo step for atoms).
 * The panel: the target, an axis (x, y, z, a, b, c, a*, b*, c*, the line of sight, the screen's axes or a
 * typed [u v w]) and an exact angle to apply either way; the snap on/off and its step; the compass dials of the
 * live readout, each angle given both ways round (+30.0° / −330.0°). A click without a drag selects, as in
 * the Select tool (after Pick: makes that atom the pivot). */
import * as G from "../geom/core.js";
import * as M from "./gizmo_math.js";
import { turntableUp, rotationPrefs, ANGLE_STEPS } from "./camera.js";
import { strokeRing, haloText, vpColor, gizmoPanel } from "./gizmo_overlay.js";
import * as DS from "./display_state.js";
import { of as vacuumOf } from "./vacuum.js";
import * as profile from "../profile.js";
import { onChange as settingsChanged } from "../settings.js";

const HIT_PX = 7;
const AXES = [["x", [1, 0, 0], "--axis-a"], ["y", [0, 1, 0], "--axis-b"], ["z", [0, 0, 1], "--axis-c"]];
const AXIS_CHOICES = [["x", "x"], ["y", "y"], ["z", "z"], ["a", "a"], ["b", "b"], ["c", "c"], ["a*", "a*"], ["b*", "b*"], ["c*", "c*"],
  ["view", "Line of sight (in-plane)"], ["h", "Screen vertical"], ["v", "Screen horizontal"], ["uvw", "[u v w]…"]];
const PIVOT_KEY = "viewport.rotatePivot";
const SINGLE_HINT = "One atom turns in place; pick another pivot";

export function createRotateTool(c) {
  const { E, S, V } = c;
  let rot = null, hover = null, last = null, panel = null, typed = { axis: "z", uvw: "1 1 0", angle: "90" };
  let targetPref = profile.ui("rotateTarget", "auto");
  let picking = false;          // Pick: the next click on an atom makes it the pivot
  const lastAtom = new WeakMap(); // document -> the atom picked last in it (Picked atom again after another pivot)
  // a new default (Settings › Viewport › Rotate pivot): documents without their own follow it at once
  settingsChanged((k) => { if (k === PIVOT_KEY && panel && !panel.hidden) { gmCache = null; renderPanel(); c.schedule(); S.toolChip(); } });
  const prefs = () => rotationPrefs(S.setting);
  const ed = () => V.ed();
  const doc = () => { const sc = V.scene(); return sc && sc.doc ? sc.doc : null; };
  const canvasEl = () => document.getElementById("viewer-main");
  const font = (w = 600, px = 12) => `${w} ${px}px ${vpColor("--font-sans", "system-ui, sans-serif")}`;

  /* ---------------------------------------------------------------- what turns */
  function selectionUsable() {
    const e = ed();
    return !!(e && e.selection.size && !e.editBlock());
  }
  function target() {
    if (targetPref === "view") return "view";
    return selectionUsable() ? "selection" : "view";
  }
  function setTarget(t) {
    targetPref = t === "selection" ? "auto" : t;
    profile.setUi("rotateTarget", targetPref);
    c.schedule(); renderPanel(); S.toolChip();
  }
  let wholeCache = null;
  /** The selection made whole (cached until the structure or the selection changes). */
  function selectedAtoms(e) {
    const key = `${c.selVersion()}:${e.selection.size}`;
    if (!wholeCache || wholeCache.s !== e.structure || wholeCache.key !== key) wholeCache = { s: e.structure, key, w: M.wholeSelection(e.structure, e.selection) };
    return wholeCache.w;
  }

  /* ---------------------------------------------------------------- the pivot (1.1.6) */
  /** The document's pivot choice: its own, else Settings › Viewport › Rotate pivot. */
  const pivotPref = () => String(DS.value(doc(), PIVOT_KEY) || "centroid");
  function setPivot(v) {
    const d = doc();
    if (!d || !M.PIVOT_RE.test(String(v))) return;
    const p = M.parsePivot(v);
    if (p.mode === "atom") lastAtom.set(d, p.atom);
    DS.set(d, PIVOT_KEY, v);
    picking = false;
    gmCache = null;
    c.schedule(); renderPanel(); S.toolChip();
  }
  function setPicking(on) {
    picking = !!on;
    c.schedule(); renderPanel(); S.toolChip();
  }
  /** The pivot of a turn of the selection now ({point, mode, atom, name, bad}; gizmo_math.js pivotPoint). */
  function pivotOf(e) {
    const d = doc(), s = e.structure;
    return M.pivotPoint(s, selectedAtoms(e), pivotPref(), d ? vacuumOf(d, s) : null);
  }

  /* ---------------------------------------------------------------- the gizmo's geometry */
  let gmCache = null;
  /** The gizmo now (cached while the view, the target, the selection and the viewer's size stay the same). */
  function geometry(fixed = null) {
    const cam = V.cam(), d = doc(), el = canvasEl();
    if (!cam || !d || !el) return null;
    if (fixed) return build(cam, d, el, fixed);
    const e = ed(), key = [cam.getView().join(","), target(), e ? `${c.selVersion()}:${e.selection.size}` : "", pivotPref(), el.offsetWidth, el.offsetHeight,
      panel ? `${panel.className}${panel.hidden}${panel.offsetHeight}` : ""].join("|");
    if (gmCache && gmCache.key === key && gmCache.s === d.structure) return gmCache.gm;
    const gm = build(cam, d, el, null);
    gmCache = { key, s: d.structure, gm };
    return gm;
  }
  function build(cam, d, el, fixed) {
    const cal = fixed ? fixed.cal : cam.calibrate();
    const t = fixed ? fixed.target : target();
    // the selection turns about the Pivot (the wheels sit on it), the view about its center
    const piv = fixed ? fixed.piv : t === "selection" ? pivotOf(ed()) : null;
    const pivot = fixed ? fixed.pivot : piv ? piv.point : cal.center;
    const ps = M.screen(cal.P, pivot);
    const w = el.offsetWidth, h = el.offsetHeight;
    let R = Math.max(44, Math.min(120, 0.27 * Math.min(w, h)));
    // the wheels stay clear of the panel (bottom right) when they can: the outer one ends short of its left edge
    if (panel && !panel.hidden && panel.isConnected) {
      const pr = panel.getBoundingClientRect(), cr = el.getBoundingClientRect(), z = window.maceZoom ? window.maceZoom.of(el) : 1;
      const left = (pr.left - cr.left) / z, top = (pr.top - cr.top) / z;
      if (ps[0] < left && ps[1] + R * 1.22 > top) R = Math.max(44, Math.min(R, (left - ps[0] - 10) / 1.22));
    }
    const ppa = M.screen(cal.P, G.add3(pivot, cal.right))[0] - ps[0];
    const rA = R / (Math.abs(ppa) > 1e-6 ? Math.abs(ppa) : cal.s);
    const rings = AXES.map(([k, n, col]) => ({ key: k, n, col, pts: M.ringPoints(cal.P, pivot, n, rA, cal.toward) }));
    return { cal, target: t, pivot, piv, ps, R, Rv: R * 1.22, rA, rings, w, h };
  }
  function hitTest(gm, x, y) {
    let best = null, bd = HIT_PX;
    for (const frontOnly of [true, false]) {
      for (const r of gm.rings) {
        const d = M.ringDist(r.pts, x, y, frontOnly);
        if (d < bd) { bd = d; best = r.key; }
      }
      if (best) return best;
    }
    if (Math.abs(Math.hypot(x - gm.ps[0], y - gm.ps[1]) - gm.Rv) < HIT_PX) return "view";
    return "free";
  }

  /* ---------------------------------------------------------------- a drag */
  const mods = { ctrl: false, shift: false, alt: false };
  function readMods(ev) {
    mods.ctrl = !!(ev.ctrlKey || ev.metaKey); mods.shift = !!ev.shiftKey; mods.alt = !!ev.altKey;
  }
  function onModKey(ev) {
    if (!rot) return;
    if (["Control", "Meta", "Shift", "Alt", "AltGraph"].includes(ev.key)) {
      readMods(ev);
      if (ev.key === "Alt") {
        // no menu bar or mnemonics from Alt during the drag; pressing it again switches the axis it keeps
        ev.preventDefault(); ev.stopPropagation();
        if (ev.type === "keydown" && !ev.repeat && rot.altPresses++ > 0) rot.pick++;
      }
      update();
    }
  }
  function onKey(ev) {
    if (!rot) return "pass";
    if (["Control", "Meta", "Shift", "Alt", "AltGraph"].includes(ev.key)) return "pass";
    // Tab (without Alt: Alt+Tab is the system's window switcher) also switches the axis Alt keeps
    if (ev.key === "Tab") { rot.pick++; update(); return "consume"; }
    if (ev.key === "Escape") { cancel(); return "consume"; }
    if (ev.key === "Enter" || ev.code === "NumpadEnter") { finish(); return "consume"; }
    return "swallow";
  }
  function begin(ev, x, y) {
    const gm = geometry();
    if (!gm) return false;
    const cam = V.cam(), view = cam.getView(), q0 = view.slice(4, 8), qi = G.qconj(q0), p = prefs();
    const kind = hitTest(gm, x, y);
    const e = ed();
    const atoms = gm.target === "selection" ? selectedAtoms(e) : null;
    rot = { kind, x0: x, y0: y, x, y, moved: false, view0: view, q0, gm, e, atoms, pick: 0, altPresses: ev.altKey ? 1 : 0, qm: [0, 0, 0, 1], comps: [],
      right: G.qrotate(qi, [1, 0, 0]), up: G.qrotate(qi, [0, 1, 0]), toward: G.qrotate(qi, [0, 0, 1]),
      hAxis: p.mode === "axes" ? (c.keptUp = turntableUp(q0, c.keptUp)) : G.qrotate(qi, [0, 1, 0]),
      sweep0: M.screenAngle(gm.ps[0], gm.ps[1], x, y), prev: 0, prevSweep: 0 };
    if (kind !== "free" && kind !== "view") {
      const n = AXES.find((a) => a[0] === kind)[1];
      const model = M.ringGrabPoint(gm.cal.P, gm.pivot, n, gm.rA, gm.cal.toward, x, y);
      rot.ring = { n, start: { x, y, model } };
      rot.ring.a0 = M.ringAngle(gm.cal.P, gm.pivot, n, gm.cal.toward, x, y, rot.ring.start);
    }
    readMods(ev);
    S.setLayer({ name: "rotateActive", onKey });
    addEventListener("keydown", onModKey, true);
    addEventListener("keyup", onModKey, true);
    if (gm.target === "view") cam.stopAnim();
    c.schedule(); S.toolChip(); renderPanel();
    return true;
  }
  /** The raw angles of the drag so far: [{name, axis (model), deg}]. */
  function rawComps() {
    const r = rot, gm = r.gm;
    const sweep = () => {
      r.prevSweep = M.unwrap(r.prevSweep, r.sweep0, M.screenAngle(gm.ps[0], gm.ps[1], r.x, r.y));
      return [{ name: "In-plane", axis: r.toward, deg: r.prevSweep }];
    };
    if (r.kind === "view") return sweep();
    if (r.kind === "free") {
      if (mods.shift && mods.alt) return sweep();
      return [{ name: "Horizontal", axis: r.hAxis, deg: (r.x - r.x0) * M.DEG_PER_PX },
        { name: "Vertical", axis: r.right, deg: (r.y - r.y0) * M.DEG_PER_PX }];
    }
    const a = M.ringAngle(gm.cal.P, gm.pivot, r.ring.n, gm.cal.toward, r.x, r.y, r.ring.start);
    r.prev = M.unwrap(r.prev, r.ring.a0, a);
    return [{ name: `About ${r.kind}`, axis: r.ring.n, deg: r.prev, key: r.kind }];
  }
  function snapState() {
    const p = prefs();
    return { on: (p.step > 0) !== mods.ctrl, step: M.snapStepOf(p) };
  }
  function update() {
    if (!rot) return;
    const raw = rawComps(), sn = snapState();
    const lock = rot.kind === "free" && mods.shift && mods.alt;
    const r = M.resolveAngles(raw.map((x) => x.deg), { snap: sn.on, step: sn.step, snapAll: mods.shift && !lock,
      isolate: mods.alt && !mods.shift, pick: rot.pick });
    rot.comps = raw.map((x, k) => ({ ...x, raw: x.deg, deg: r.degs[k], kept: r.kept === k }));
    rot.mode = lock ? "in-plane only" : r.kept >= 0 ? `one axis (${raw[r.kept].name.toLowerCase()})` : mods.shift ? "snap all" : sn.on ? `snap ${fmtStep(sn.step)}` : "free";
    rot.qm = M.composeTurn(rot.comps);
    apply(rot.qm);
    last = { target: rot.gm.target, comps: rot.comps, mode: rot.mode, pivot: rot.gm.piv ? pivotLabel(rot.gm.piv) : "" };
    c.schedule(); S.toolChip(); renderReadout();
  }
  function apply(qm) {
    const r = rot;
    if (r.gm.target === "view") {
      V.cam().setView([...r.view0.slice(0, 4), ...M.turnedView(r.q0, qm)]);
      return;
    }
    const s = r.e.structure, cart = M.turnAbout(r.atoms.cart, r.gm.pivot, qm);
    r.cart = cart;
    const df = new Map(r.atoms.sel.map((i, k) => [i, G.fracOf(s.cell, G.sub3(cart[k], r.atoms.cart[k]))]));
    r.df = df;
    c.preview(r.e, r.atoms.sel, (i) => df.get(i) || [0, 0, 0]);
  }
  function end() {
    removeEventListener("keydown", onModKey, true);
    removeEventListener("keyup", onModKey, true);
    S.setLayer(null);
    rot = null;
    c.schedule(); S.toolChip(); renderPanel();
  }
  function cancel() {
    const r = rot;
    if (!r) return;
    if (r.gm.target === "view") V.cam().setView(r.view0);
    else c.clearPreview();
    last = null;
    end();
  }
  function finish() {
    const r = rot;
    if (!r) return;
    const turned = Math.abs(r.qm[3]) < 1 - 1e-12;
    if (r.gm.target === "selection") {
      c.clearPreview();
      if (turned && r.df) commitAtoms(r.e, r.atoms.sel, r.df, r.gm.piv);
    }
    end();
  }
  /** One undo step, named after the pivot ("Rotate 3 Atoms about Ti1"): each atom where it was plus its share of
   *  the turn (df: frac displacements), so every atom stays in the cell image it had (Settings' wrapIntoCell
   *  still applies). */
  function commitAtoms(e, sel, df, piv) {
    const s = e.structure, n = sel.length;
    const frac = sel.map((i) => s.frac[i].map((x, k) => x + df.get(i)[k]));
    const about = piv && piv.name ? ` about ${piv.name}` : "";
    return E.edit(e, "set_positions", { indices: sel, frac }, { label: `Rotate ${n} Atom${n === 1 ? "" : "s"}${about}` });
  }

  /* ---------------------------------------------------------------- the pointer (tools.js registers it) */
  const tool = {
    down(ev, x, y) {
      if (ev.button !== 0 || rot) return false;
      return begin(ev, x, y);
    },
    move(ev, x, y, drag) {
      if (!rot) return;
      rot.x = x; rot.y = y; rot.moved = rot.moved || (drag ? drag.moved : true);
      readMods(ev);
      update();
    },
    hover(ev, x, y) {
      if (rot) return;
      const gm = geometry();
      const h = gm ? hitTest(gm, x, y) : null;
      if (h !== hover) {
        hover = h;
        const wrap = canvasEl() && canvasEl().closest(".viewer-wrap");
        if (wrap && !picking) wrap.dataset.gizmo = h && h !== "free" ? "ring" : "";
        c.schedule();
      }
    },
    up(ev, x, y, drag) {
      if (!rot) return;
      if (!drag || !drag.moved) {
        // a click: select as the Select tool does (in an editable structure)
        cancelQuiet();
        const e = ed(), k = V.scene().pickAt(x, y);
        if (picking) {
          // Pick: the atom clicked (any atom, selected or not) is the pivot; the selection stays
          if (k >= 0) setPivot(`atom:${V.scene().display.base[k] + 1}`);
          return;
        }
        if (!e) return;
        const mod = S.isMac ? ev.metaKey : ev.ctrlKey;
        const mode = ev.shiftKey && mod ? "subtract" : ev.shiftKey ? "add" : mod ? "toggle" : "replace";
        if (k < 0) { if (mode === "replace") E.select(e, [], "replace"); return; }
        E.select(e, [V.scene().display.base[k]], mode);
        return;
      }
      if (Number.isFinite(x) && (x !== rot.x || y !== rot.y)) { rot.x = x; rot.y = y; readMods(ev); update(); }
      finish();
    },
    right() { if (rot) { cancel(); return true; } return false; },
  };
  function cancelQuiet() {
    if (!rot) return;
    if (rot.gm.target === "view") V.cam().setView(rot.view0);
    end();
  }

  /* ---------------------------------------------------------------- drawing */
  function draw(g) {
    const gm = rot ? { ...geometry(rot.gm), ps: rot.gm.ps } : geometry();
    if (!gm) return false;
    const bg = vpColor("--viewport-bg"), text = vpColor("--text"), text2 = vpColor("--text-2");
    const accent = vpColor("--accent");
    const [cx, cy] = gm.ps;
    const active = rot ? rot.kind : null;
    // the trackball disc: drag inside it (or anywhere off a wheel) for a free turn
    g.save();
    g.globalAlpha = active === "free" ? 0.12 : 0.06;
    g.fillStyle = accent;
    g.beginPath(); g.arc(cx, cy, gm.R, 0, 2 * Math.PI); g.fill();
    g.restore();
    // the outer wheel: in-plane, about the line of sight
    g.save();
    g.strokeStyle = active === "view" || hover === "view" ? accent : text2;
    g.lineWidth = active === "view" || hover === "view" ? 3.5 : 2;
    g.globalAlpha = 0.9;
    g.beginPath(); g.arc(cx, cy, gm.Rv, 0, 2 * Math.PI); g.stroke();
    g.restore();
    // the axis wheels, far ones first
    const rings = gm.rings.slice().sort((a, b) => Math.abs(G.dot3(a.n, gm.cal.toward)) - Math.abs(G.dot3(b.n, gm.cal.toward)));
    for (const r of rings) {
      const col = vpColor(r.col, vpColor("--fallback-atom"));
      const on = active === r.key || (!rot && hover === r.key);
      strokeRing(g, r.pts, col, on ? 3.5 : 2.2, { highlight: on, alpha: rot && active !== r.key ? 0.45 : 1 });
      // the axis letter on the ring's near half, where it is farthest from the other wheels (and just outside)
      if (!rot) {
        const others = gm.rings.filter((o) => o !== r).flatMap((o) => o.pts.filter((p, k) => k % 3 === 0));
        let best = null, bd = -1;
        for (let k = 0; k < r.pts.length; k += 2) {
          const p = r.pts[k];
          if (!p.front) continue;
          let d = Infinity;
          for (const o of others) d = Math.min(d, Math.hypot(p.x - o.x, p.y - o.y));
          if (d > bd + 0.5) { bd = d; best = p; }
        }
        if (best) {
          const dx = best.x - cx, dy = best.y - cy, n = Math.hypot(dx, dy) || 1;
          haloText(g, r.key, best.x + (dx / n) * 10, best.y + (dy / n) * 10, col, bg, { font: font(700, 12) });
        }
      }
    }
    // the turn so far: a sector from the reference mark on the wheel being dragged, and the angle
    if (rot && rot.comps.length) {
      const comp = rot.comps.find((x) => x.kept) || rot.comps[0];
      if (rot.kind === "view" || (rot.kind === "free" && rot.comps[0].name === "In-plane")) {
        const a0 = rot.sweep0 * Math.PI / 180, a = comp.deg * Math.PI / 180;
        sector(g, cx, cy, gm.Rv, -a0, -(a0 + a), accent);
        tick(g, cx, cy, gm.Rv, -a0, text);
        const am = -(a0 + a);
        haloText(g, M.bothWays(comp.deg)[0], cx + (gm.Rv + 22) * Math.cos(am), cy + (gm.Rv + 22) * Math.sin(am), text, bg, { font: font(700, 13) });
      } else if (rot.ring) {
        const pts = arcPoints(gm, rot.ring.n, rot.ring.start.model, comp.deg);
        const col = vpColor(AXES.find((x) => x[0] === rot.kind)[2], accent);
        g.save();
        g.fillStyle = col; g.globalAlpha = 0.22;
        g.beginPath(); g.moveTo(cx, cy); for (const p of pts) g.lineTo(p[0], p[1]); g.closePath(); g.fill();
        g.globalAlpha = 1; g.strokeStyle = col; g.lineWidth = 2;
        g.beginPath(); g.moveTo(cx, cy); g.lineTo(pts[0][0], pts[0][1]); g.moveTo(cx, cy);
        g.lineTo(pts[pts.length - 1][0], pts[pts.length - 1][1]); g.stroke();
        g.restore();
        const e = pts[pts.length - 1];
        haloText(g, M.bothWays(comp.deg)[0], e[0] + (e[0] - cx) * 0.25, e[1] + (e[1] - cy) * 0.25, text, bg, { font: font(700, 13) });
      } else {
        // a free drag: the angles above the wheels (below them when the top of the view is too close)
        const lines = rot.comps.map((x) => `${x.name} ${M.bothWays(x.deg)[0]}`), n = lines.length;
        const above = cy - gm.Rv - 12 - 16 * (n - 1) > 44;
        lines.forEach((t, k) => haloText(g, t, cx, above ? cy - gm.Rv - 12 - 16 * (n - 1 - k) : cy + gm.Rv + 14 + 16 * k, text, bg,
          { font: font(700, 13) }));
      }
    }
    // the center: the selection's pivot gets a marker and its name, the view's center a dot
    if (gm.piv) pivotMarker(g, cx, cy, gm.piv, accent, text, bg);
    else { g.save(); g.fillStyle = text; g.beginPath(); g.arc(cx, cy, 2.5, 0, 2 * Math.PI); g.fill(); g.restore(); }
    return true;
  }
  /** The pivot: a small crosshair in a ring (a diamond for a picked atom), in the text color on a halo (apart
   *  from the colored wheels), its name beside it. */
  function pivotMarker(g, cx, cy, piv, accent, text, bg) {
    g.save();
    g.lineCap = "round";
    for (const [col, w] of [[bg, 4.5], [text, 2]]) {
      g.strokeStyle = col; g.lineWidth = w;
      g.beginPath();
      if (piv.mode === "atom") { g.moveTo(cx, cy - 7); g.lineTo(cx + 7, cy); g.lineTo(cx, cy + 7); g.lineTo(cx - 7, cy); g.closePath(); }
      else g.arc(cx, cy, 6, 0, 2 * Math.PI);
      g.moveTo(cx - 10, cy); g.lineTo(cx - 3, cy); g.moveTo(cx + 3, cy); g.lineTo(cx + 10, cy);
      g.moveTo(cx, cy - 10); g.lineTo(cx, cy - 3); g.moveTo(cx, cy + 3); g.lineTo(cx, cy + 10);
      g.stroke();
    }
    g.fillStyle = accent; g.beginPath(); g.arc(cx, cy, 1.8, 0, 2 * Math.PI); g.fill();
    g.restore();
    haloText(g, pivotLabel(piv), cx + 13, cy + 13, text, bg, { align: "left", font: font(600, 11) });
  }
  function sector(g, cx, cy, R, from, to, color) {
    g.save();
    g.fillStyle = color; g.globalAlpha = 0.2;
    g.beginPath(); g.moveTo(cx, cy); g.arc(cx, cy, R, from, to, to < from); g.closePath(); g.fill();
    g.globalAlpha = 1; g.strokeStyle = color; g.lineWidth = 3;
    g.beginPath(); g.arc(cx, cy, R, from, to, to < from); g.stroke();
    g.restore();
  }
  function tick(g, cx, cy, R, a, color) {
    g.save(); g.strokeStyle = color; g.lineWidth = 2;
    g.beginPath(); g.moveTo(cx + (R - 8) * Math.cos(a), cy + (R - 8) * Math.sin(a)); g.lineTo(cx + (R + 8) * Math.cos(a), cy + (R + 8) * Math.sin(a));
    g.stroke(); g.restore();
  }
  /** Screen points along the wheel about n from the grab point, through deg. */
  function arcPoints(gm, n, from, deg) {
    const r0 = G.sub3(from, gm.pivot), out = [], steps = Math.max(2, Math.ceil(Math.abs(deg) / 4));
    for (let k = 0; k <= steps; k++) {
      const p = G.add3(gm.pivot, G.qrotate(G.qaxis(n, (deg * k) / steps), r0));
      const s = M.screen(gm.cal.P, p);
      out.push([s[0], s[1]]);
    }
    return out;
  }

  /* ---------------------------------------------------------------- the panel */
  const fmtStep = (s) => `${+s.toFixed(3)}°`;
  /** The pivot as the marker and the chip name it: "centroid", "center of mass", ..., or the atom ("Ti1"). */
  const pivotLabel = (piv) => (piv.mode === "atom" ? piv.name : (M.PIVOTS.find((x) => x[0] === piv.mode) || [0, "centroid"])[1].replace("Selection ", "").toLowerCase());
  function renderPanel() {
    const host = c.panelHost();
    if (!host) return;
    if (!panel || !panel.isConnected) {
      panel = gizmoPanel("rotate", "Rotate", "Drag a wheel to turn about its axis only (the outer wheel: in-plane), anywhere else to turn freely. "
        + "Ctrl (held): invert the snap · Shift: snap every angle · Alt: keep one axis, snapped (Alt again or Tab: the other) · Shift+Alt: in-plane only · "
        + "Esc or a right click cancels · letting go applies", profile);
      panel.querySelector(".gz-body").innerHTML = `<div class="gz-dials" aria-live="polite"></div>
        <div class="gz-row"><span class="segment gz-target" role="radiogroup" aria-label="Rotate target">
            <button type="button" role="radio" data-t="view">View</button><button type="button" role="radio" data-t="selection">Selection</button></span></div>
        <div class="gz-row"><label class="gz-f">Pivot <select class="gz-pivot" aria-label="Rotate pivot">${M.PIVOTS.map(([v, t]) => `<option value="${v}">${t}</option>`).join("")}<option value="atom">Picked atom</option></select></label>
          <button type="button" class="gz-btn gz-pick" aria-pressed="false" title="Then click an atom: the selection turns about it (it need not be selected)">Pick</button>
          <span class="gz-k gz-pivot-atom" hidden></span></div>
        <span class="gz-hint gz-pivot-hint" role="status" hidden></span>
        <div class="gz-row"><label class="gz-f">Axis <select class="gz-axis" aria-label="Rotation axis">${AXIS_CHOICES.map(([v, t]) => `<option value="${v}">${t}</option>`).join("")}</select></label>
          <input class="gz-uvw" aria-label="Axis [u v w]" spellcheck="false" autocomplete="off" placeholder="u v w" hidden></div>
        <div class="gz-row"><label class="gz-f">Angle <input class="gz-angle" inputmode="decimal" aria-label="Rotation angle (degrees)" spellcheck="false" autocomplete="off"><span class="gz-u">°</span></label>
          <button type="button" class="gz-btn primary" data-act="ccw" title="Turn by the angle (counterclockwise seen from the axis's tip)">↺ Apply</button>
          <button type="button" class="gz-btn" data-act="cw" title="Turn by the angle the other way" aria-label="Apply the other way">↻</button></div>
        <div class="gz-row"><label class="gz-f gz-check"><input type="checkbox" class="gz-snap" aria-label="Snap"> Snap</label>
          <label class="gz-f">Step <input class="gz-step" inputmode="decimal" aria-label="Snap step (degrees)" spellcheck="false" autocomplete="off"><span class="gz-u">°</span></label></div>
        <span class="gz-err" role="alert"></span>`;
      host.appendChild(panel);
      wirePanel();
      panel.querySelector(".gz-collapse").addEventListener("click", () => c.schedule());
    }
    const t = rot ? rot.gm.target : target();
    panel.querySelectorAll(".gz-target button").forEach((b) => {
      b.setAttribute("aria-checked", String(b.dataset.t === t));
      b.disabled = b.dataset.t === "selection" && !selectionUsable();
      b.title = b.dataset.t === "selection" ? (selectionUsable() ? "Turn the selected atoms about the pivot (an edit)" : "Select atoms in an editable structure first") : "Turn the view about its center";
    });
    renderPivot(t);
    const ax = panel.querySelector(".gz-axis"), uvw = panel.querySelector(".gz-uvw"), ang = panel.querySelector(".gz-angle");
    if (document.activeElement !== ax) ax.value = typed.axis;
    uvw.hidden = typed.axis !== "uvw";
    if (document.activeElement !== uvw) uvw.value = typed.uvw;
    if (document.activeElement !== ang) ang.value = typed.angle;
    const p = prefs(), snap = panel.querySelector(".gz-snap"), step = panel.querySelector(".gz-step");
    snap.checked = p.step > 0;
    if (document.activeElement !== step) step.value = String(+M.snapStepOf(p).toFixed(3));
    renderReadout();
  }
  /** The Pivot row: the document's choice, the picked atom's name, Pick's state, and the single-atom hint. */
  function renderPivot(t) {
    const sel = panel.querySelector(".gz-pivot"), pick = panel.querySelector(".gz-pick");
    const name = panel.querySelector(".gz-pivot-atom"), hint = panel.querySelector(".gz-pivot-hint");
    const p = M.parsePivot(pivotPref()), e = ed(), d = doc();
    sel.value = picking ? "atom" : p.mode;
    pick.setAttribute("aria-pressed", String(picking));
    const piv = e && e.selection.size ? pivotOf(e) : null;
    const s = (e && e.structure) || (d && d.structure);
    // the picked atom by name (its symbol and number, as Labels › Both): "Picked atom (Ti1)"
    const atomTxt = p.mode !== "atom" || !s ? "" : p.atom < (s.frac || s.cart || []).length ? M.atomName(s, p.atom) : `atom ${p.atom + 1}?`;
    sel.querySelector('option[value="atom"]').textContent = atomTxt ? `Picked atom (${atomTxt})` : "Picked atom";
    name.textContent = picking ? "Click an atom…" : "";
    name.hidden = !picking;
    let why = "";
    // one atom about itself (its centroid, its center of mass or itself picked) would turn in place
    if (t === "selection" && e && e.selection.size === 1 && (p.mode === "centroid" || p.mode === "com" || (p.mode === "atom" && e.selection.has(p.atom)))) why = SINGLE_HINT;
    else if (t === "selection" && piv && piv.bad) why = p.mode === "atom" ? "The picked atom is gone: turning about the centroid" : "No cell: turning about the centroid";
    hint.textContent = why;
    hint.hidden = !why;
    const wrap = canvasEl() && canvasEl().closest(".viewer-wrap");
    if (wrap && (picking || wrap.dataset.gizmo === "pick")) wrap.dataset.gizmo = picking ? "pick" : "";
  }
  function wirePanel() {
    const q = (s) => panel.querySelector(s), err = q(".gz-err");
    panel.querySelectorAll(".gz-target button").forEach((b) => b.addEventListener("click", () => setTarget(b.dataset.t)));
    q(".gz-pivot").addEventListener("change", (ev) => {
      const v = ev.target.value;
      // Picked atom: the atom picked before, else Pick
      if (v === "atom") {
        const p = M.parsePivot(pivotPref());
        if (p.mode === "atom") return;
        const d = doc(), last = d && lastAtom.has(d) ? lastAtom.get(d) : -1;
        if (last >= 0) setPivot(`atom:${last + 1}`); else setPicking(true);
        return;
      }
      setPivot(v);
    });
    q(".gz-pick").addEventListener("click", () => setPicking(!picking));
    q(".gz-axis").addEventListener("change", (ev) => { typed.axis = ev.target.value; renderPanel(); if (typed.axis === "uvw") q(".gz-uvw").focus(); });
    q(".gz-uvw").addEventListener("input", (ev) => { typed.uvw = ev.target.value; });
    q(".gz-angle").addEventListener("input", (ev) => { typed.angle = ev.target.value; });
    const keys = (ev) => {
      if (ev.key === "Enter") { ev.preventDefault(); ev.stopPropagation(); applyTyped(1); }
      else if (ev.key === "Escape") { ev.preventDefault(); ev.stopPropagation(); V.focusViewport(); }
    };
    q(".gz-angle").addEventListener("keydown", keys);
    q(".gz-uvw").addEventListener("keydown", keys);
    q('[data-act="ccw"]').addEventListener("click", () => applyTyped(1));
    q('[data-act="cw"]').addEventListener("click", () => applyTyped(-1));
    q(".gz-snap").addEventListener("change", (ev) => {
      const p = prefs(), step = M.snapStepOf(p);
      S.set("viewport.angleSnap", ev.target.checked ? snapKey(step) : "off");
      renderPanel();
    });
    const stepIn = q(".gz-step");
    const takeStep = () => {
      const r = G.evalExpr(stepIn.value);
      // settings.py viewport.snapStep: 0.1 to 180 (a smaller step would not be kept)
      if (!r.ok || !(r.value >= 0.1 && r.value <= 180)) { stepIn.setAttribute("aria-invalid", "true"); err.textContent = "Step: a number of degrees from 0.1 to 180"; return; }
      stepIn.removeAttribute("aria-invalid"); err.textContent = "";
      setStep(r.value);
    };
    stepIn.addEventListener("change", takeStep);
    stepIn.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") { ev.preventDefault(); ev.stopPropagation(); takeStep(); }
      else if (ev.key === "Escape") { ev.preventDefault(); ev.stopPropagation(); V.focusViewport(); }
    });
  }
  const snapKey = (step) => (ANGLE_STEPS.includes(step) ? String(step) : "custom");
  /** A new snap step: the custom angle always; the Angle snap follows it while it is on. */
  function setStep(step) {
    S.set("viewport.snapStep", step);
    if (prefs().step > 0) S.set("viewport.angleSnap", snapKey(step));
    renderPanel();
  }
  /** The panel's Apply (dir 1) or its reverse (dir -1): the typed angle about the picked axis. */
  function applyTyped(dir) {
    const err = panel.querySelector(".gz-err");
    const a = G.evalExpr(String(typed.angle || ""));
    if (!a.ok || !Number.isFinite(a.value)) { err.textContent = "Angle: a number of degrees"; return null; }
    const d = doc(), cam = V.cam();
    if (!d || !cam) return null;
    const s = d.structure, q = cam.getView().slice(4, 8);
    const pbc = s.pbc || [true, true, true];
    const cell = s.cell && pbc.some(Boolean) ? s.cell : null;
    const axis = M.axisVector(typed.axis === "uvw" ? typed.uvw : typed.axis, cell, q);
    if (!axis) { err.textContent = typed.axis === "uvw" ? "Axis: three numbers, not all zero" : "That axis needs a cell"; return null; }
    err.textContent = "";
    const deg = dir * a.value, qm = G.qaxis(axis, deg), t = target();
    last = { target: t, comps: [{ name: `About ${typed.axis === "uvw" ? `[${typed.uvw.trim()}]` : AXIS_CHOICES.find((x) => x[0] === typed.axis)[1]}`, deg, raw: deg }], mode: "typed",
      pivot: t === "selection" ? pivotLabel(pivotOf(ed())) : "" };
    renderReadout();
    if (t === "view") return cam.turnModel(axis, deg).then(() => c.schedule());
    const e = ed(), at = selectedAtoms(e), piv = pivotOf(e);
    const cart = M.turnAbout(at.cart, piv.point, qm), ecell = e.structure.cell;
    return commitAtoms(e, at.sel, new Map(at.sel.map((i, k) => [i, G.fracOf(ecell, G.sub3(cart[k], at.cart[k]))])), piv);
  }

  /* ---------------------------------------------------------------- the readout: compass dials */
  function renderReadout() {
    if (!panel) return;
    const host = panel.querySelector(".gz-dials");
    const r = rot ? { comps: rot.comps, mode: rot.mode, target: rot.gm.target, pivot: rot.gm.piv ? pivotLabel(rot.gm.piv) : "" } : last;
    if (!r || !r.comps || !r.comps.length) { host.innerHTML = ""; host.hidden = true; return; }
    host.hidden = false;
    const head = `<span class="gz-k">${rot ? "Turning" : "Last turn"} · ${r.target === "view" ? "view" : `selection${r.pivot ? ` about ${r.pivot}` : ""}`} · ${r.mode}</span>`;
    host.innerHTML = head + r.comps.map((x) => {
      const [a, b] = M.bothWays(x.deg), rawTxt = Math.abs(x.raw - x.deg) > 0.05 ? ` <span class="gz-k">(drag ${M.bothWays(x.raw)[0]})</span>` : "";
      return `<span class="gz-dial" data-name="${x.name}">${dial(x.deg)}<span><b>${x.name}</b> ${a} <span class="gz-k">/ ${b}</span>${rawTxt}</span></span>`;
    }).join("");
  }
  /** A small compass: the reference mark at the top, the needle at the angle, the turn shaded (counterclockwise positive). */
  function dial(deg) {
    const a = ((deg % 360) + 360) % 360, rad = (a * Math.PI) / 180, R = 11;
    const x = Math.sin(-rad) * R, y = -Math.cos(rad) * R;
    const large = a > 180 ? 1 : 0;
    const arc = a < 0.05 || a > 359.95 ? "" : `<path class="gz-sec" d="M0 0 L0 ${-R} A${R} ${R} 0 ${deg >= 0 ? large : 1 - large} ${deg >= 0 ? 0 : 1} ${x.toFixed(3)} ${y.toFixed(3)} Z"/>`;
    return `<svg class="gz-compass" viewBox="-14 -14 28 28" aria-hidden="true"><circle r="${R}"/>${arc}<line class="gz-ref" x1="0" y1="${-R - 2}" x2="0" y2="${-R + 3}"/><line class="gz-needle" x1="0" y1="0" x2="${x.toFixed(3)}" y2="${y.toFixed(3)}"/></svg>`;
  }

  /* ---------------------------------------------------------------- the chip */
  function chip() {
    const t = rot ? rot.gm.target : target(), e = rot ? rot.e : ed();
    const n = e ? e.selection.size : 0;
    const who = t === "view" ? "the view" : `${n} atom${n === 1 ? "" : "s"}`;
    if (picking && !rot) return `Rotate · click an atom for the pivot · Pick again to stop`;
    if (rot && rot.comps.length) {
      const parts = rot.comps.map((x) => `${x.name} ${M.bothWays(x.deg).join(" / ")}`);
      const gtxt = c.groupText && t === "selection" ? c.groupText() : "";
      const about = rot.gm.piv ? ` about ${pivotLabel(rot.gm.piv)}` : "";
      return `Rotate ${who}${about} · ${parts.join(" · ")} · ${rot.mode}${gtxt ? ` · ${gtxt}` : ""}`;
    }
    const g = t === "selection" ? geometry() : null, about = g && g.piv ? ` about ${pivotLabel(g.piv)}` : "";
    return `Rotate ${who}${about} · drag a wheel or inside it · Esc to exit`;
  }

  return {
    tool, draw, chip, renderPanel, cancel,
    get active() { return !!rot; },
    get state() { return rot ? { kind: rot.kind, target: rot.gm.target, comps: rot.comps.map((x) => ({ name: x.name, deg: x.deg, raw: x.raw })), mode: rot.mode } : null; },
    get last() { return last; },
    geometry: () => { const g = geometry(); return g && { target: g.target, center: g.ps.slice(0, 2), R: g.R, Rv: g.Rv,
      pivot: g.piv ? { mode: g.piv.mode, atom: g.piv.atom, name: g.piv.name, label: pivotLabel(g.piv), point: g.pivot.slice(), bad: !!g.piv.bad } : null,
      rings: Object.fromEntries(g.rings.map((r) => [r.key, r.pts.filter((p) => p.front).map((p) => [p.x, p.y])])) }; },
    applyTyped, setTarget, setTyped: (o) => { Object.assign(typed, o); renderPanel(); }, setStep,
    setPivot, setPicking, get pivot() { return pivotPref(); }, get picking() { return picking; },
    hidePanel() {
      if (picking) { picking = false; const w = canvasEl() && canvasEl().closest(".viewer-wrap"); if (w && w.dataset.gizmo === "pick") w.dataset.gizmo = ""; }
      if (panel) panel.hidden = true;
    },
    showPanel() { if (panel) panel.hidden = false; },
  };
}
