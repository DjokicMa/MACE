/* Big 3D overlays without the lag (1.1.6): a pore mesh, an isosurface, polyhedra.
 *
 * 3Dmol (vendored 2.5.5) builds every shape's mesh again on each viewer.render()
 * (GLShape.globj: finalizeGeo, initTypedArrays, the colors rewritten, a new material
 * and Mesh), and MACE calls render() on a hover, a click, a zoom step, a label...
 * keepBuilt(shape) builds it once and keeps that mesh while what it is drawn with
 * (hidden, opacity, wireframe, side, line width, color, vertices) stays the same;
 * removeShape/removeAllShapes (removegl) or another group start it over.
 *
 * 3Dmol also draws every translucent mesh twice per paint: a depth pre-pass (the
 * material at opacity 0) and then the color. A surface MACE draws without writing
 * depth (pores.js) gets nothing from that pre-pass, so skipDepthPass(viewer) leaves
 * it out for those materials only (3Dmol internal: Renderer.unrollBufferMaterial,
 * which sets each object's transparentDepth).
 *
 * A shape drawn over the display range's cells (cell_tiles.js: ``maceTiles``) is built
 * once too: each build adds the copies of its mesh at the other cells. */

import { cloneTiles } from "./cell_tiles.js";

const colorKey = (c) => (c == null ? "" : typeof c === "object" ? `${c.r},${c.g},${c.b}` : String(c));
const signature = (s) => [s.hidden, s.opacity, s.wireframe, s.side, s.linewidth, colorKey(s.color),
  s.geo ? s.geo.groups : 0, s.geo ? s.geo.vertices : 0, s.maceTiles ? s.maceTiles.length : 1].join("|");

/** The shape, its mesh built once (and moved to each of its ``maceTiles``); ``after(obj, shape)`` runs on each
 * build (e.g. a material tweak). */
export function keepBuilt(s, after) {
  if (!s || s.maceKept) return s;
  const globj = s.globj;
  s.maceKept = true;
  s.globj = function (group, ...rest) {
    const sig = signature(this), r = this.renderedShapeObj;
    if (r && r.parent === group && this.maceSig === sig) return;
    globj.call(this, group, ...rest);
    this.maceSig = sig;
    if (this.maceTiles && this.renderedShapeObj) cloneTiles(this.renderedShapeObj, this.maceTiles);
    if (after && this.renderedShapeObj) after(this.renderedShapeObj, this);
  };
  return s;
}

/** No depth pre-pass for translucent meshes that write no depth (once per viewer). */
export function skipDepthPass(vw) {
  const R = vw && vw.renderer;
  if (!R || R.maceSkipDepth || typeof R.unrollBufferMaterial !== "function") return;
  const unroll = R.unrollBufferMaterial;
  R.maceSkipDepth = true;
  R.unrollBufferMaterial = function (o) {
    unroll.call(this, o);
    const m = o && o.object && o.object.material;
    if (m && m.transparent && m.depthWrite === false) o.transparentDepth = null;
  };
}
