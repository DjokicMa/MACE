/* Clipping planes and depth cueing for every 3Dmol viewer (polish A2).
 *
 * 3Dmol sets its slab (the near and far clipping planes) only in zoomTo(),
 * i.e. at the first fit: a larger display range, a pan or a zoom afterwards
 * cut atoms off. Its fog was never configured, so 3Dmol's default (fog from
 * 40% of the slab) washed out most of a large cell.
 *
 * installClip(v) wraps the viewer's setSlabAndFog, which 3Dmol calls before
 * every paint (show()): first it sets slabNear/slabFar so the bounding sphere
 * of what is drawn lies inside them for the current view, then it calls the
 * original (which turns them into the camera's near/far and updates the
 * projection in the same paint), then it sets the fog from the depth of the
 * drawn object, not of the slab, so a pan does not change the shading.
 *
 * The bounding sphere {c, R} (model space, atom radii included) is computed
 * from the viewer's models (every frame of an animation) and shapes, and kept
 * until a model or shape is added or removed, so a paint does O(1) work.
 * 3Dmol internals used (vendored 3Dmol 2.5.5): setSlabAndFog, slabNear,
 * slabFar, scene.fog, config.disableFog, getModelList, model.frames,
 * shapes[].boundingSphere. */

export const MARGIN = 2;          // Å beyond the bounding sphere
export const DEFAULT_DEPTH_CUE = 0.15;

const MUTATORS = ["addModel", "addModels", "addModelsAsFrames", "addAsOneMolecule", "removeModel", "removeAllModels",
  "addShape", "addSphere", "addBox", "addArrow", "addCylinder", "addCurve", "addLine", "addCustom", "addUnitCell",
  "removeShape", "removeAllShapes", "removeUnitCell"];

/** Bounding sphere of points ([x,y,z] arrays or a flat array), each with radius r (a number or radii[k]). */
export function boundsOf(points, radii = 0, extra = 0) {
  const flat = points.length && typeof points[0] === "number";
  const n = flat ? points.length / 3 : points.length;
  if (!n) return null;
  const at = (k, j) => (flat ? points[3 * k + j] : points[k][j]);
  let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
  for (let k = 0; k < n; k++) {
    const x = at(k, 0), y = at(k, 1), z = at(k, 2);
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; if (z < z0) z0 = z; if (z > z1) z1 = z;
  }
  const c = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2];
  let R = 0;
  for (let k = 0; k < n; k++) {
    const r = typeof radii === "number" ? radii : radii[k] || 0;
    const d = Math.hypot(at(k, 0) - c[0], at(k, 1) - c[1], at(k, 2) - c[2]) + r;
    if (d > R) R = d;
  }
  return { c, R: R + extra };
}

/** The union of two bounding spheres (either may be null). */
export function union(a, b) {
  if (!a) return b;
  if (!b) return a;
  const d = Math.hypot(b.c[0] - a.c[0], b.c[1] - a.c[1], b.c[2] - a.c[2]);
  if (d + b.R <= a.R) return a;
  if (d + a.R <= b.R) return b;
  const R = (d + a.R + b.R) / 2, t = d > 1e-12 ? (R - a.R) / d : 0;
  return { c: [0, 1, 2].map((j) => a.c[j] + (b.c[j] - a.c[j]) * t), R };
}

function vdw(elem) {
  const t = (window.$3Dmol && window.$3Dmol.GLModel && window.$3Dmol.GLModel.vdwRadii) || {};
  return t[elem] || 1.6;
}
/** The radius an atom is drawn with (Å), read from its style; never smaller than what 3Dmol draws. */
function atomRadius(a) {
  const st = a.style || {};
  if (st.sphere) {
    if (st.sphere.radius) return st.sphere.radius;
    return (st.sphere.scale || 1) * vdw(a.elem);
  }
  return st.stick ? (st.stick.radius || 0.25) : 0.3;
}

/** What the viewer draws: every atom of every model (each frame of an animation) and every shape. */
export function autoBounds(v) {
  const pts = [], rad = [];
  for (const m of (v.getModelList ? v.getModelList() : [])) {
    const frames = Array.isArray(m.frames) && m.frames.length ? m.frames : [m.selectedAtoms({})];
    for (const atoms of frames) {
      if (!Array.isArray(atoms)) continue;
      for (const a of atoms) if (a && Number.isFinite(a.x)) { pts.push([a.x, a.y, a.z]); rad.push(atomRadius(a)); }
    }
  }
  let b = boundsOf(pts, rad);
  for (const s of v.shapes || []) {
    const bs = s && s.boundingSphere;
    if (bs && bs.center && Number.isFinite(bs.center.x) && bs.radius > 0)
      b = union(b, { c: [bs.center.x, bs.center.y, bs.center.z], R: bs.radius });
  }
  return b;
}

/** z of a model-space point in the rotated frame (view = 3Dmol's getView()). */
function viewZ(view, p) {
  const x = p[0] + view[0], y = p[1] + view[1], z = p[2] + view[2];
  const [qx, qy, qz, qw] = [view[4], view[5], view[6], view[7]];
  // rotate (x, y, z) by the unit quaternion q: z component of q p q*
  const ix = qw * x + qy * z - qz * y, iy = qw * y + qz * x - qx * z, iz = qw * z + qx * y - qy * x, iw = -qx * x - qy * y - qz * z;
  return iz * qw + iw * -qz + ix * -qy - iy * -qx;
}

/** Wrap v.setSlabAndFog (once). opts.depthCue(): the fog strength 0..1 (0 = off);
 * opts.beforePaint(v): called first at every paint (the Rendering values, O(1)). */
export function installClip(v, opts = {}) {
  if (!v || v.maceClip || typeof v.setSlabAndFog !== "function") return;
  const orig = v.setSlabAndFog.bind(v);
  let cache = null;
  const invalidate = () => { cache = null; };
  for (const name of MUTATORS) {
    const f = v[name];
    if (typeof f !== "function") continue;
    v[name] = function (...args) { cache = null; return f.apply(this, args); };
  }
  const depthCue = () => {
    const s = typeof v.maceDepthCue === "number" ? v.maceDepthCue : opts.depthCue ? opts.depthCue() : DEFAULT_DEPTH_CUE;
    return Number.isFinite(s) ? Math.min(1, Math.max(0, s)) : DEFAULT_DEPTH_CUE;
  };
  v.maceClip = { invalidate, bounds: () => v.maceBounds || cache || (cache = autoBounds(v)), last: null };
  v.setSlabAndFog = function () {
    if (opts.beforePaint) opts.beforePaint(v);
    const b = v.maceClip.bounds();
    if (!b || !this.scene || !this.rotationGroup) return orig();
    const view = this.getView();
    const zc = viewZ(view, b.c), R = b.R;
    // a point at rotated z lies at distance e - z from the camera; 3Dmol keeps it
    // when e + slabNear <= e - z <= e + slabFar
    this.slabNear = -zc - R - MARGIN;
    this.slabFar = -zc + R + MARGIN;
    orig();
    const e = this.camera.position.z - this.rotationGroup.position.z;
    const front = e - zc - R, back = e - zc + R, s = depthCue();
    if (s <= 0) {
      this.config.disableFog = true;
      this.scene.fog.near = this.scene.fog.far = back + 1;
    } else {
      this.config.disableFog = false;
      this.scene.fog.near = front;
      this.scene.fog.far = front + (back - front) / s;
    }
    v.maceClip.last = { zc, R, e, front, back, s, near: this.camera.near, far: this.camera.far };
  };
}
