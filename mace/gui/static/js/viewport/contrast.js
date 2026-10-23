/* Contrast on whatever a mark is drawn over (1.1.6): the tokens of tokens.css keep 3:1 on their own
 * surfaces (tests/gui/test_tokens_contrast.py), but a 3D view can have a custom background, and a
 * small canvas draws on the inspector's surface. legible() keeps a mark's color when it already
 * stands out (WCAG contrast at least `min`), else moves it toward black or white until it does, so a
 * cell, a slice plane, a zone edge or a label never vanishes on the background it is drawn on. */

/** [r, g, b] (0-255) of "#rgb", "#rrggbb", "rgb(...)" or "rgba(...)"; null for anything else. */
export function rgbOf(c) {
  const s = String(c || "").trim();
  let m = /^#([0-9a-f]{3})$/i.exec(s);
  if (m) return m[1].split("").map((x) => parseInt(x + x, 16));
  m = /^#([0-9a-f]{6})([0-9a-f]{2})?$/i.exec(s);
  if (m) return [0, 2, 4].map((k) => parseInt(m[1].slice(k, k + 2), 16));
  m = /^rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)/i.exec(s);
  if (m) return [m[1], m[2], m[3]].map((x) => Math.max(0, Math.min(255, Math.round(Number(x)))));
  return null;
}
const hex = (rgb) => "#" + rgb.map((x) => Math.round(x).toString(16).padStart(2, "0")).join("");
/** WCAG relative luminance (0 black to 1 white). */
export function luminance(c) {
  const v = Array.isArray(c) ? c : rgbOf(c);
  if (!v) return 0.5;
  const f = (x) => { x /= 255; return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4; };
  return 0.2126 * f(v[0]) + 0.7152 * f(v[1]) + 0.0722 * f(v[2]);
}
/** WCAG contrast ratio of two colors (1 to 21). */
export function ratio(a, b) {
  const x = luminance(a), y = luminance(b);
  return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05);
}
/** `fg` when it stands out on `bg` (contrast >= min), else fg mixed toward black or white (whichever
 * the background is farther from) in 10 % steps until it does. Unreadable colors pass through. */
export function legible(fg, bg, min = 3) {
  const a = rgbOf(fg), b = rgbOf(bg);
  if (!a || !b || ratio(a, b) >= min) return fg;
  const to = luminance(b) > 0.4 ? [0, 0, 0] : [255, 255, 255];
  for (let t = 0.1; t <= 1.0001; t += 0.1) {
    const c = a.map((x, k) => x + (to[k] - x) * t);
    if (ratio(c, b) >= min) return hex(c);
  }
  return hex(to);
}
