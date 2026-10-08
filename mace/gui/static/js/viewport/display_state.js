/* What a structure document draws (polish A3): the Inspector's "Range &
 * boundary" and "Rendering" groups. Settings › Viewport holds the defaults;
 * a document stores only its overrides (docprefs.js, kind "display"), so a
 * changed default applies at once to every document that has not changed
 * that value, and Reset makes a document follow the default again. The
 * Display section's Cell, Bonds, Labels, style and colors stay global.
 * Nothing here reaches a deck, a CIF or a settings file. */
import * as docprefs from "../docprefs.js";
import * as settings from "../settings.js";

export const GROUPS = {
  range: ["displayLo", "displayHi", "showBoundary", "bondedOutside", "showPolyhedra", "atomScale", "bondRadius"],
  render: ["renderPreset", "depthCue", "lightStrength", "lightDirection", "ambient", "outline", "outlineWidth",
    "outlineColor", "ambientOcclusion", "background", "backgroundColor"],
};
export const KEYS = new Set([...GROUPS.range, ...GROUPS.render]);
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

/* The presets of the Rendering group: Default is exactly the look before 1.1.5
 * (one light at 3Dmol's direction, no ambient, no outline) with subtle depth cueing. */
export const PRESETS = {
  default: { lightStrength: 1, lightDirection: "front", ambient: 0, depthCue: 0.15, outline: false, ambientOcclusion: 0, background: "theme" },
  flat: { lightStrength: 0.8, lightDirection: "front", ambient: 0.35, depthCue: 0, outline: false, ambientOcclusion: 0, background: "theme" },
  publication: { lightStrength: 1, lightDirection: "front", ambient: 0.2, depthCue: 0, outline: true, outlineWidth: 0.1,
    ambientOcclusion: 0.5, background: "white" },
  presentation: { lightStrength: 1.2, lightDirection: "front", ambient: 0.15, depthCue: 0.25, outline: false,
    ambientOcclusion: 0, background: "black" },
};

export function defaults() {
  const o = {};
  for (const k of KEYS) o[k] = settings.get(k);
  return o;
}
/** The effective values: the defaults with the document's overrides on top. */
export function of(doc) {
  const d = defaults();
  if (!doc) return d;
  return { ...d, ...docprefs.get("display", doc) };
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
  const keys = GROUPS[group] || [];
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
/** The cell box drawn around a range: whole cells from round(lo) to round(hi), at least one. */
export function boxOf(range) {
  if (Array.isArray(range)) return { lo: [0, 0, 0], hi: range.slice() };
  const lo = range.lo.map((x) => Math.round(x));
  const hi = range.hi.map((x, k) => Math.max(lo[k] + 1, Math.round(x)));
  return { lo, hi };
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
