/* The structure viewer's camera (spec §2.1, §6): orbit as a trackball on
 * the view quaternion (or, Settings › Viewport › Rotation, as a turntable,
 * optionally in whole angle steps, Ctrl held inverting the snap), pan in screen space, zoom toward the
 * cursor or the view center, view
 * along a direction with a defined "up", 90°/15° turns, framing, and the
 * projection matrix the picker, box and lasso use (calibrated from 3Dmol's
 * own modelToScreen, so it is exact for both projections).
 *
 * 3Dmol's view is [x, y, z, zoom, qx, qy, qz, qw]: the model translation
 * (minus the point at the screen center), the camera distance, and the
 * rotation from model to camera space (x right, y up, z toward the viewer). */
import * as G from "../geom/core.js";

const DEG_PER_PX = 0.4;
const X = [1, 0, 0], Y = [0, 1, 0], Z = [0, 0, 1];

/* ---------------------------------------------------------------- rotation modes
 * Settings › Viewport › Rotation and Angle snap, shared by the structure view
 * (this camera) and the 3Dmol run, mode and volume views (app.js):
 *   free: the trackball, about the screen axis across the drag;
 *   axes: a turntable: a horizontal drag turns about the model's "up" (the
 *         direction that was up on screen when the turntable took it, kept
 *         while it stays upright on screen), a vertical drag about the screen
 *         x axis. Neither moves up's image off the vertical: no roll drift.
 *   angle snap (15/30/45/90°): the drag's horizontal and vertical turns are
 *         summed from where it began and applied in whole steps. */
export const ANGLE_STEPS = [15, 30, 45, 90];
/** Custom (1.1.6): Angle snap "custom" snaps by Settings › Viewport › Custom angle (viewport.snapStep, any
 *  angle up to 180°); custom is that angle (15° unset), the step Ctrl snaps by while the snap is off. */
export function rotationPrefs(get) {
  const snap = get("viewport.angleSnap"), step = Number(snap), own = Number(get("viewport.snapStep"));
  const custom = Number.isFinite(own) && own > 0 && own <= 180 ? own : 15;
  return { mode: get("viewport.rotation") === "axes" ? "axes" : "free",
    step: ANGLE_STEPS.includes(step) ? step : snap === "custom" ? custom : 0, custom };
}
/** The prefs with the snap inverted (Ctrl held during a drag): a snapping drag turns freely, a free one snaps by custom. */
export const invertedSnap = (p) => ({ ...p, step: p.step > 0 ? 0 : (p.custom || 15) });
/** The model's "up" for a turntable at orientation q: the kept one while its image is still vertical on
 *  screen (no roll since), else the model direction that is up on screen now. */
export function turntableUp(q, up) {
  if (up) {
    const s = G.qrotate(q, up);
    if (Math.abs(s[0]) < 1e-3 && Math.hypot(s[1], s[2]) > 0.5) return up;
  }
  return G.qrotate(G.qconj(q), Y);
}
const snapAngle = (a, step) => (step > 0 ? Math.round(a / step) * step : a);
/** The orientation after a whole drag of (dx, dy) px from q0 (rotationPrefs: mode, step; up: turntableUp). */
export function dragOrientation(q0, dx, dy, { mode = "free", step = 0, up = Y } = {}) {
  const yaw = snapAngle(dx * DEG_PER_PX, step), pitch = snapAngle(dy * DEG_PER_PX, step);
  if (mode === "axes") return G.qmul(G.qaxis(X, pitch), G.qmul(q0, G.qaxis(up, yaw)));
  return G.qmul(G.qaxis(X, pitch), G.qmul(G.qaxis(Y, yaw), q0));
}
/** A turn by deg about screen axis "x", "y" or "z"; a turntable turns "y" about its up instead. */
export function turnOrientation(q, axis, deg, { mode = "free", up = null } = {}) {
  if (mode === "axes" && axis === "y") return G.qmul(q, G.qaxis(turntableUp(q, up), deg));
  return G.qmul(G.qaxis({ x: X, y: Y, z: Z }[axis], deg), q);
}
/** The view along the cell axis or Cartesian axis (a, b, c, a*, b*, c*, x, y, z, either sign) nearest to
 *  the current one, its "up" the in-plane axis nearest the current up (else the current up, made
 *  perpendicular): { q, label }. cell: the lattice rows, or null for a molecule (x, y, z only). */
export function nearestAxisView(q, cell) {
  const qi = G.qconj(q), toward = G.qrotate(qi, Z), upNow = G.qrotate(qi, Y);
  // the cell's axes first: of two the same (a cubic cell's a and x), the cell's name
  const cands = [];
  if (cell) for (const w of ["a", "b", "c", "a*", "b*", "c*"]) { try { cands.push([w, G.alongDirection(cell, w)]); } catch (e) { /* a flat cell */ } }
  cands.push(["x", X], ["y", Y], ["z", Z]);
  const both = cands.flatMap(([n, d]) => [[n, d], [`−${n}`, d.map((x) => -x)]]);
  let best = both[0], bd = -2;
  for (const c of both) { const s = G.dot3(c[1], toward); if (s > bd + 1e-9) { bd = s; best = c; } }
  const d = best[1];
  let up = null, bu = 0.3;
  for (const [, c] of both) {
    if (Math.abs(G.dot3(c, d)) > 0.02) continue;
    const s = G.dot3(c, upNow);
    if (s > bu + 1e-9) { bu = s; up = c; }
  }
  if (!up) up = G.sub3(upNow, G.scale3(d, G.dot3(upNow, d)));
  if (G.norm3(up) < 1e-6) up = Math.abs(d[2]) < 0.9 ? Z : Y;
  return { q: G.viewQuaternion(d, up), label: best[0] };
}

export function createCamera(v, hooks = {}) {
  let cal = null, anim = 0, radius = 10;
  // a drag in a snapping or turntable mode: its start, the px moved since, and the turntable's up;
  // turnFrom: the orientation any orbit drag began at (the live readout's reference)
  let drag = null, up = null, turnFrom = null;
  const prefs = () => (hooks.rotation ? hooks.rotation() : { mode: "free", step: 0 });
  const el = () => v.getCanvas();
  const rect = () => el().getBoundingClientRect();
  const reduced = () => (hooks.reducedMotion ? hooks.reducedMotion() : false);
  const changed = () => {
    cal = null;
    document.dispatchEvent(new CustomEvent("mace-camera", { detail: v }));
  };

  /* 3Dmol updates its matrices when it paints, and the shell paints once per
   * frame: bring them up to date now, so the projection is of this view. */
  function sync() {
    try {
      if (v.setSlabAndFog) v.setSlabAndFog();
      if (v.scene && v.scene.updateMatrixWorld) v.scene.updateMatrixWorld(true);
    } catch (e) { /* a viewer not drawn yet */ }
  }
  /** Where a model point lands, in CSS px from the canvas's top left. */
  function screenOf(p) {
    sync();
    const r = rect(), s = v.modelToScreen({ x: p[0], y: p[1], z: p[2] });
    return [s.x - r.left, s.y - r.top];
  }

  /** The projection matrix and the screen frame at the view center, rebuilt
   * only after the view changed. */
  function calibrate() {
    if (cal) return cal;
    const view = v.getView(), c = [-view[0], -view[1], -view[2]], R = Math.max(radius, 2);
    const q = view.slice(4, 8), qi = G.qconj(q);
    const right = G.qrotate(qi, [1, 0, 0]), up = G.qrotate(qi, [0, 1, 0]), toward = G.qrotate(qi, [0, 0, 1]);
    const pts = [];
    for (const a of [-1, 1]) for (const b of [-1, 1]) for (const d of [-1, 1]) pts.push([c[0] + a * R, c[1] + b * R, c[2] + d * R]);
    pts.push([c[0] + 0.3 * R, c[1] - 0.2 * R, c[2] + 0.7 * R]);
    sync();
    const r0 = rect();
    const P = G.dltCalibrate(pts.map((p) => { const q = v.modelToScreen({ x: p[0], y: p[1], z: p[2] }); return [...p, q.x - r0.left, q.y - r0.top]; }));
    const c0 = G.project(P, ...c), c1 = G.project(P, ...G.add3(c, right));
    const s = Math.hypot(c1[0] - c0[0], c1[1] - c0[1]) || 1;      // px per Å at the center
    cal = { P, view, center: c, centerPx: [c0[0], c0[1]], s, right, up, toward, q };
    return cal;
  }

  const setView = (view) => { v.setView(view); v.render(); changed(); };
  const getView = () => v.getView();

  function stopAnim() { if (anim) cancelAnimationFrame(anim); anim = 0; }
  /** Turn to quaternion q (and optionally a center): animated over --dur-3. */
  function animateTo(q, center = null, ms = null) {
    stopAnim();
    const from = getView(), q0 = from.slice(4, 8);
    const dur = reduced() ? 0 : (ms ?? (hooks.duration ? hooks.duration() : 200));
    const c0 = from.slice(0, 3), c1 = center ? center.map((x) => -x) : c0;
    if (!dur) { setView([...c1, from[3], ...q]); return Promise.resolve(); }
    const t0 = performance.now();
    return new Promise((resolve) => {
      const step = (now) => {
        const t = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - t, 3);
        const qq = G.qslerp(q0, q, e), cc = c0.map((x, k) => x + (c1[k] - x) * e);
        v.setView([...cc, from[3], ...qq]);
        v.render();
        if (t < 1) anim = requestAnimationFrame(step);
        else { anim = 0; changed(); resolve(); }
      };
      anim = requestAnimationFrame(step);
    });
  }

  return {
    calibrate, screenOf, getView, setView, stopAnim,
    invalidate() { cal = null; },
    setRadius(r) { radius = r; cal = null; },
    /** Trackball: a drag of (dx, dy) px turns the model about the screen axis
     * (dy, dx, 0); with roll, about the view axis instead (Shift). In the
     * turntable or with an angle snap the drag is summed from its start
     * (endOrbit closes it) and the orientation computed from there. */
    orbit(dx, dy, roll = false, invert = false) {
      stopAnim();
      const view = getView(), q = view.slice(4, 8), p = invert ? invertedSnap(prefs()) : prefs();
      if (!turnFrom) turnFrom = q;
      if (p.mode !== "free" || p.step) {
        // Ctrl pressed or let go mid-drag (invert): the rest of the drag is summed from where the view is
        if (!drag || drag.roll !== roll || drag.invert !== invert) {
          if (p.mode === "axes") up = turntableUp(q, up);
          drag = { q0: q, dx: 0, dy: 0, roll, invert, step: p.step };
        }
        drag.dx += dx; drag.dy += dy;
        const target = roll ? G.qmul(G.qaxis(Z, snapAngle(-drag.dx * DEG_PER_PX, p.step)), drag.q0)
          : dragOrientation(drag.q0, drag.dx, drag.dy, { ...p, up });
        setView([...view.slice(0, 4), ...target]);
        return;
      }
      drag = null;
      let dq;
      if (roll) dq = G.qaxis([0, 0, 1], -dx * DEG_PER_PX);
      else {
        const len = Math.hypot(dx, dy);
        if (len < 1e-9) return;
        dq = G.qaxis([dy / len, dx / len, 0], len * DEG_PER_PX);
      }
      setView([...view.slice(0, 4), ...G.qmul(dq, q)]);
    },
    /** The drag ended: the next one starts from the view as it is. */
    endOrbit() { drag = null; turnFrom = null; },
    /** The live readout of an orbit drag, from where it began: {total (deg), parts: [{name, deg}] (a summed
     *  drag: the horizontal and vertical turns, or the in-plane one), step}; null between drags. */
    orbitReadout() {
      if (!turnFrom) return null;
      const r = G.qmul(getView().slice(4, 8), G.qconj(turnFrom));
      const total = (2 * Math.acos(Math.min(1, Math.abs(r[3]))) * 180) / Math.PI;
      let parts = null;
      if (drag && drag.roll) parts = [{ name: "In-plane", deg: snapAngle(-drag.dx * DEG_PER_PX, drag.step) }];
      else if (drag) parts = [{ name: "Horizontal", deg: snapAngle(drag.dx * DEG_PER_PX, drag.step) },
        { name: "Vertical", deg: snapAngle(drag.dy * DEG_PER_PX, drag.step) }];
      return { total, parts, step: drag ? drag.step : 0 };
    },
    /** Move the model with the pointer: (dx, dy) px in screen space. */
    pan(dx, dy) {
      stopAnim();
      const c = calibrate(), view = getView();
      const d = G.add3(G.scale3(c.right, dx / c.s), G.scale3(c.up, -dy / c.s));
      setView([view[0] + d[0], view[1] + d[1], view[2] + d[2], ...view.slice(3)]);
    },
    /** Zoom by factor (>1 closer) keeping the model point under (x, y) where it is. */
    zoomAt(factor, x, y) {
      stopAnim();
      const c = calibrate();
      const anchor = G.add3(c.center, G.add3(G.scale3(c.right, (x - c.centerPx[0]) / c.s),
        G.scale3(c.up, -(y - c.centerPx[1]) / c.s)));
      v.zoom(factor);
      cal = null;
      const after = screenOf(anchor), view = getView(), c2 = calibrate();
      const d = G.add3(G.scale3(c2.right, (x - after[0]) / c2.s), G.scale3(c2.up, -(y - after[1]) / c2.s));
      setView([view[0] + d[0], view[1] + d[1], view[2] + d[2], ...view.slice(3)]);
    },
    zoom(factor) { stopAnim(); v.zoom(factor); v.render(); changed(); },
    /** Turn about model point p from now on: p slides to the view's center (animated), zoom and orientation kept. */
    centerOn(p) { return animateTo(getView().slice(4, 8), p); },
    /** Look along d (toward the viewer) with up as up; animated. */
    along(d, upv) { return animateTo(G.viewQuaternion(d, upv)); },
    /** Turn to the nearest cell or Cartesian axis (nearestAxisView); animated. Its label. */
    snapNearest(cell) { const r = nearestAxisView(getView().slice(4, 8), cell); return animateTo(r.q).then(() => r.label); },
    /** The turntable's up (model space), for the tests. */
    turntableUp: () => up,
    /** Turn the view about a screen axis ("x", "y" or "z": in-plane) by deg. */
    turn(axis, deg, animate = true) {
      const view = getView(), q = view.slice(4, 8), p = prefs();
      if (p.mode === "axes" && axis === "y") up = turntableUp(q, up);
      const target = turnOrientation(q, axis, deg, { mode: p.mode, up });
      if (!animate) { setView([...view.slice(0, 4), ...target]); return Promise.resolve(); }
      return animateTo(target);
    },
    /** Turn the model by deg about a model-space axis through the view's center (the Rotate tool's typed turn). */
    turnModel(axis, deg, animate = true) {
      const view = getView(), target = G.qmul(view.slice(4, 8), G.qaxis(axis, deg));
      if (!animate || reduced()) { setView([...view.slice(0, 4), ...target]); return Promise.resolve(); }
      return animateTo(target);
    },
    /** Center on these model points and zoom so they fill the view. */
    frame(points, margin = 1.6) {
      if (!points.length) return;
      const n = points.length;
      const c = points.reduce((s, p) => G.add3(s, p), [0, 0, 0]).map((x) => x / n);
      const R = Math.max(1, ...points.map((p) => G.norm3(G.sub3(p, c)))) + margin;
      const view = getView();
      setView([-c[0], -c[1], -c[2], ...view.slice(3)]);
      const cc = calibrate(), r = rect();
      const want = 0.42 * Math.min(r.width, r.height);
      const factor = want / (R * cc.s);
      if (Number.isFinite(factor) && factor > 0) this.zoom(factor);
    },
    setProjection(mode) { if (v.setProjection) { v.setProjection(mode); v.render(); changed(); } },
  };
}
