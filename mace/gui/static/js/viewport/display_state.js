/* What a structure document draws (polish A3): the Inspector's "Range &
 * boundary" and "Rendering" groups. Settings › Viewport holds the defaults;
 * a document stores only its overrides (docprefs.js, kind "display"), so a
 * changed default applies at once to every document that has not changed
 * that value, and Reset makes a document follow the default again. The
 * Display section's Cell, Bonds, Labels, atom style and color scheme stay
 * global; a document's Style preset can override the drawing and colors.
 * Nothing here reaches a deck, a CIF or a settings file. */
import * as docprefs from "../docprefs.js";
import * as settings from "../settings.js";

import { LIGHTING, STYLES, LIGHTING_KEYS, STYLE_KEYS } from "./presets.js";

/* range: Range & boundary; render: the lighting (its preset is renderPreset, the
 * name it had when it was the only preset, so saved documents keep theirs); style:
 * how atoms are drawn and colored (stylePreset); vacuum (1.1.6): a slab's, polymer's
 * or molecule's vacuum mode and custom gap (viewport/vacuum.js). */
/* bonds (1.1.6): the bond rule (geom/core.js bondRule), set per document by Build › Calculate Bonds… (Recalculate). */
/* rotate (1.1.6): the Rotate tool's pivot (viewport/gizmo_rotate.js; "atom:N" a picked atom of this document). */
/* POLY_KEYS (1.1.6): the polyhedra's centers, face opacity, edges and center atom, in the Style group (its Reset
 * and Save as Default) but set by no style preset, so changing one never makes the style Custom. */
export const POLY_KEYS = ["polyCenters", "polyOpacity", "polyEdges", "polyCenterAtom"];
export const GROUPS = {
  range: ["displayLo", "displayHi", "showBoundary", "bondedOutside"],
  render: ["renderPreset", ...LIGHTING_KEYS],
  style: ["stylePreset", ...STYLE_KEYS, ...POLY_KEYS],
  vacuum: ["viewport.vacuumMode", "viewport.vacuumGap"],
  rotate: ["viewport.rotatePivot"],
  bonds: ["bondTolerance", "bondMin", "bondMetalMetal", "bondPairs"],
};
export const KEYS = new Set([...GROUPS.range, ...GROUPS.render, ...GROUPS.style, ...GROUPS.vacuum, ...GROUPS.rotate, ...GROUPS.bonds]);
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

/* The presets of each dropdown ({id: values}, viewport/presets.js): Default lighting is
 * exactly the look before 1.1.5 (one light at 3Dmol's direction, no ambient, no outline)
 * with subtle depth cueing; Default style is the ball and stick drawn before 1.1.7. */
export const PRESETS = Object.fromEntries(LIGHTING.map((p) => [p.id, p.values]));
export const STYLE_PRESETS = Object.fromEntries(STYLES.map((p) => [p.id, p.values]));

export function defaults() {
  const o = {};
  for (const k of KEYS) o[k] = settings.get(k);
  return o;
}
/** The effective values: the defaults with the document's overrides on top. */
export function of(doc) {
  const d = defaults();
  if (!doc) return d;
  // a copy: reading never changes what is stored (it marked a saved document Custom)
  const o = { ...docprefs.get("display", doc) };
  // saved before the Style dropdown (1.1.7): a document's own atom size, bond radius or
  // polyhedra and no style preset is a Custom style, not Default
  if (!("stylePreset" in o) && STYLE_KEYS.some((k) => k in o)) o.stylePreset = "custom";
  return { ...d, ...o };
}
/** One effective value, as of(doc)[k] (the 3D view reads a dozen of them at every paint: no copies). */
export function value(doc, k) {
  if (!doc) return settings.get(k);
  const o = docprefs.get("display", doc);
  if (k in o) return o[k];
  if (k === "stylePreset" && STYLE_KEYS.some((x) => x in o)) return "custom";
  return settings.get(k);
}
export function overrides(doc) { return doc ? docprefs.get("display", doc) : {}; }
export const isOverride = (doc, k) => !!doc && k in docprefs.get("display", doc);

/** Set one value (several: an object); one equal to the default is removed, not stored. */
export function set(doc, key, value) {
  if (!doc) return;
  const pairs = typeof key === "object" ? key : { [key]: value };
  const o = docprefs.get("display", doc);
  for (const [k, v] of Object.entries(pairs)) {
    if (!KEYS.has(k)) throw new Error(`not a display key: ${k}`);
    if (same(v, settings.get(k))) delete o[k]; else o[k] = v;
  }
  docprefs.put("display", doc, o, Object.keys(pairs));
}
/** Clear a group's overrides (or the given keys). */
export function reset(doc, groupOrKeys) {
  if (!doc) return;
  const keys = Array.isArray(groupOrKeys) ? groupOrKeys : GROUPS[groupOrKeys] || [];
  const o = docprefs.get("display", doc);
  keys.forEach((k) => delete o[k]);
  docprefs.put("display", doc, o, keys);
}
/** Write the document's values of a group into Settings (the defaults); its overrides then go. */
export function saveAsDefault(doc, group) {
  const v = of(doc);
  // a document's own centers are its elements: the default stays "auto"
  const keys = (GROUPS[group] || []).filter((k) => k !== "polyCenters");
  for (const k of keys) settings.set(k, v[k]);
  reset(doc, keys);
}

/* ---------------------------------------------------------------- the range */
/** {lo, hi} of the document. */
export function rangeOf(doc) {
  const v = of(doc);
  return { lo: (v.displayLo || [0, 0, 0]).slice(), hi: (v.displayHi || [1, 1, 1]).slice() };
}
/** What displayList takes: the old [n1, n2, n3] for whole cells from 0, else {lo, hi}. */
export function drawRange(doc) {
  const r = rangeOf(doc);
  const whole = [0, 1, 2].every((k) => r.lo[k] === 0 && Number.isInteger(r.hi[k]) && r.hi[k] >= 1);
  return whole ? r.hi.slice() : r;
}
/** The cell box drawn around a range: whole cells from round(lo) to round(hi), at least one. With `vacuum`
 * ({mode, gap, file}, 1.1.6) the non-periodic axes follow the vacuum mode (geom/core.js boxCorners). */
export function boxOf(range, vacuum = null) {
  const v = vacuum ? { vacuum } : {};
  if (Array.isArray(range)) return { lo: [0, 0, 0], hi: range.slice(), ...v };
  const lo = range.lo.map((x) => Math.round(x));
  const hi = range.hi.map((x, k) => Math.max(lo[k] + 1, Math.round(x)));
  return { lo, hi, ...v };
}
/** The toolbar's label: 2×2×2, or −0.1–1.1 (one axis' bounds when all agree). */
export function rangeLabel(doc) {
  const r = rangeOf(doc);
  if ([0, 1, 2].every((k) => r.lo[k] === 0 && Number.isInteger(r.hi[k]))) return r.hi.join("×");
  const f = (x) => String(+x.toFixed(2)).replace("-", "−");
  const axis = (k) => `${f(r.lo[k])}–${f(r.hi[k])}`;
  return [0, 1, 2].every((k) => axis(k) === axis(0)) ? axis(0) : [0, 1, 2].map(axis).join(", ");
}
export function setCells(doc, n) {
  const hi = Array.isArray(n) ? n.slice() : [n, n, n];
  set(doc, { displayLo: [0, 0, 0], displayHi: hi });
}
