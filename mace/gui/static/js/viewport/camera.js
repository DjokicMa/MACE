/* The structure viewer's camera (spec §2.1, §6): orbit as a trackball on
 * the view quaternion, pan in screen space, zoom toward the cursor, view
 * along a direction with a defined "up", 90°/15° turns, framing, and the
 * projection matrix the picker, box and lasso use (calibrated from 3Dmol's
 * own modelToScreen, so it is exact for both projections).
 *
 * 3Dmol's view is [x, y, z, zoom, qx, qy, qz, qw]: the model translation
 * (minus the point at the screen center), the camera distance, and the
 * rotation from model to camera space (x right, y up, z toward the viewer). */
import * as G from "../geom/core.js";

const DEG_PER_PX = 0.4;

export function createCamera(v, hooks = {}) {
  let cal = null, anim = 0, radius = 10;
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
     * (dy, dx, 0); with roll, about the view axis instead (Shift). */
    orbit(dx, dy, roll = false) {
      stopAnim();
      const view = getView(), q = view.slice(4, 8);
      let dq;
      if (roll) dq = G.qaxis([0, 0, 1], -dx * DEG_PER_PX);
      else {
        const len = Math.hypot(dx, dy);
        if (len < 1e-9) return;
        dq = G.qaxis([dy / len, dx / len, 0], len * DEG_PER_PX);
      }
      setView([...view.slice(0, 4), ...G.qmul(dq, q)]);
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
    along(d, up) { return animateTo(G.viewQuaternion(d, up)); },
    /** Turn the view about a screen axis ("x", "y" or "z": in-plane) by deg. */
    turn(axis, deg, animate = true) {
      const view = getView(), q = view.slice(4, 8);
      const ax = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] }[axis];
      const target = G.qmul(G.qaxis(ax, deg), q);
      if (!animate) { setView([...view.slice(0, 4), ...target]); return Promise.resolve(); }
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
