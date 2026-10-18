/* Plot presets (1.1.6): the GUI side of mace/plotting/presets.py. The data (the presets, the spin
 * pair, the DOS rules) is SHARED, generated from that module into plotpresets_data.js; the
 * functions here are the same as presets.py's (tests/test_plot_presets.py runs both on the same
 * inputs). The series colors a preset draws in the GUI come from tokens.css ([data-plot-preset]
 * blocks, per light or dark plot); everything else from here. No DOM: node imports this module. */
import { SHARED } from "./plotpresets_data.js";

export { SHARED };
export const FAMILIES = SHARED.families;
export const FAMILY_LABELS = { band: "Band & DOS", spectra: "Spectra", maps: "Maps & volumes", scf: "SCF & run charts" };
export const DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"];

/** A user's own preset (room for saved presets): `preset` over a copy of `base`. */
export function register(id, preset, base = "default") {
  if (!/^[a-z][a-z0-9_-]{0,31}$/.test(String(id || ""))) throw new Error("a preset id is lowercase letters, digits, - and _");
  SHARED.presets[id] = { ...JSON.parse(JSON.stringify(SHARED.presets[base])), ...JSON.parse(JSON.stringify(preset)) };
  if (!SHARED.order.includes(id)) SHARED.order.push(id);
  return SHARED.presets[id];
}
/** The preset id `pid` as `family` draws it: default for an unknown id or a family it does not cover. */
export function resolveId(pid, family = "band") {
  const p = SHARED.presets[pid || "default"];
  return p && p.families.includes(family) ? pid : "default";
}
export const get = (pid, family = "band") => SHARED.presets[resolveId(pid, family)];
const HEX = /^#[0-9a-fA-F]{6}$/;
/** The spin pair {up, down} (presets.py spin_colors): `scheme` (a SHARED.spin_schemes id) when given, else the
 *  preset's own; `up`/`down` the custom colors (they apply with scheme "custom" or "auto"). */
export function spinColors(pid, family = "band", tone = "light", scheme = null, up = null, down = null) {
  const p = get(pid, family), S = SHARED.spin_schemes;
  tone = tone === "dark" ? "dark" : "light";
  const pair = S[scheme] || S[p.spin_scheme] || p.spin || S[SHARED.default_spin_scheme];
  const out = { ...(pair[tone] || pair.light) };
  if (!scheme || scheme === "auto" || scheme === "custom") {
    if (up && HEX.test(up)) out.up = up.toLowerCase();
    if (down && HEX.test(down)) out.down = down.toLowerCase();
  }
  return out;
}
/** The color of a run's one set of bands and its one-sided DOS total (presets.py band_color): `custom`
 *  (#rrggbb), else the preset's "bands", else `ink` (the plot's text color). */
export function bandColor(pid, family = "band", tone = "light", custom = null, ink = "#000000") {
  if (custom && HEX.test(custom)) return custom.toLowerCase();
  const own = get(pid, family).bands;
  if (own) return own[tone === "dark" ? "dark" : "light"] || own.light;
  return ink;
}
export const STYLE_OPTIONS = SHARED.style_options;
export const STYLE_DEFAULTS = Object.fromEntries(SHARED.style_options.map((o) => [o.name, o.default]));
/** mace plotting's flags for a style (presets.py style_flags): only the values that are not the defaults;
 *  explicit: the defaults too, so a mace-folder.toml's [plots] values do not replace them. */
export function styleFlags(look, explicit = false) {
  const v = { ...STYLE_DEFAULTS };
  for (const [k, x] of Object.entries(look || {})) if (x !== null && x !== undefined) v[k] = x;
  const out = [];
  if (v.style !== "default" || explicit) out.push(`--style=${v.style}`);
  const scheme = v.spin_colors || "auto";
  if (scheme !== "auto" || explicit) out.push(`--spin-colors=${scheme}`);
  if (scheme === "auto" || scheme === "custom") {
    for (const k of ["spin_up_color", "spin_down_color"]) if (v[k] && HEX.test(v[k])) out.push(`--${k.replace(/_/g, "-")}=${v[k].toLowerCase()}`);
  }
  if (v.band_color && HEX.test(v.band_color)) out.push(`--band-color=${v.band_color.toLowerCase()}`);
  if (v.spin_down_style === "solid") out.push("--spin-down-solid");
  else if (v.spin_down_style !== "dash" || explicit) out.push(`--spin-down-style=${v.spin_down_style}`);
  if (v.dos_colors !== "auto" || explicit) out.push(`--dos-colors=${v.dos_colors}`);
  if (v.dos_spin !== "auto" || explicit) out.push(`--dos-spin=${v.dos_spin}`);
  if (v.dos_total === false) out.push("--no-dos-total");
  else if (explicit) out.push("--dos-total");
  const sm = smoothSigma(v.dos_smooth);
  if (sm || explicit) out.push(`--dos-smooth=${sm}`);
  return out;
}
/** The DOS smoothing width (eV) of a style value (presets.py smooth_sigma): 0 (off) unless a positive number. */
export function smoothSigma(v) {
  return typeof v === "number" && Number.isFinite(v) && v > 0 ? v : 0;
}

/* ---------------------------------------------------------------- colors (as presets.py) */
const rgbOf = (h) => { h = String(h).replace("#", ""); if (h.length === 3) h = [...h].map((c) => c + c).join(""); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255); };
const hexOf = (rgb) => "#" + rgb.map((c) => Math.max(0, Math.min(255, Math.floor(c * 255 + 0.5))).toString(16).padStart(2, "0")).join("");
export function luminance(h) {
  const ch = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  const [r, g, b] = rgbOf(h).map(ch);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
export function contrast(a, b) { const la = luminance(a), lb = luminance(b); return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05); }
/* python's colorsys.rgb_to_hls / hls_to_rgb */
function toHls([r, g, b]) {
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), l = (mn + mx) / 2;
  if (mn === mx) return [0, l, 0];
  const s = l <= 0.5 ? (mx - mn) / (mx + mn) : (mx - mn) / (2 - mx - mn);
  const rc = (mx - r) / (mx - mn), gc = (mx - g) / (mx - mn), bc = (mx - b) / (mx - mn);
  let h = r === mx ? bc - gc : g === mx ? 2 + rc - bc : 4 + gc - rc;
  h = ((h / 6) % 1 + 1) % 1;
  return [h, l, s];
}
function fromHls([h, l, s]) {
  if (s === 0) return [l, l, l];
  const m2 = l <= 0.5 ? l * (1 + s) : l + s - l * s, m1 = 2 * l - m2;
  const v = (hue) => { hue = ((hue % 1) + 1) % 1; if (hue < 1 / 6) return m1 + (m2 - m1) * hue * 6; if (hue < 0.5) return m2; if (hue < 2 / 3) return m1 + (m2 - m1) * (2 / 3 - hue) * 6; return m1; };
  return [v(h + 1 / 3), v(h), v(h - 1 / 3)];
}
/** `color` darkened (light background) or lightened (dark one) until it has `need`:1 on `bg`. */
export function readable(color, bg = "#ffffff", need = 2) {
  if (contrast(color, bg) >= need) return String(color).toLowerCase();
  let [h, l, s] = toHls(rgbOf(color));
  const darkBg = luminance(bg) < 0.18;
  for (let i = 0; i < 100; i++) {
    l = darkBg ? Math.min(1, l + 0.01) : Math.max(0, l - 0.01);
    const c = hexOf(fromHls([h, l, s]));
    if (contrast(c, bg) >= need) return c;
  }
  return darkBg ? "#ffffff" : "#000000";
}
const RANK = { s: 0, p: 1 / 3, d: 2 / 3, f: 1 };
/** One orbital's shade of an element color (element × orbital): s strongest, f faintest still readable. */
export function shade(color, orbital, bg = "#ffffff", need = 2, rank = null) {
  if (rank === null && !(orbital in RANK)) return readable(color, bg, need);
  const [h, , s] = toHls(rgbOf(color)), darkBg = luminance(bg) < 0.18;
  let edge = darkBg ? 1 : 0;
  for (let i = 0; i <= 100; i++) {
    const l = darkBg ? i / 100 : (100 - i) / 100;
    if (contrast(hexOf(fromHls([h, l, s])), bg) >= need) { edge = l; break; }
  }
  // spread down (up) from the edge, never to black or white: the hue stays visible
  const t = rank === null ? RANK[orbital] : rank, span = SHARED.shade_span;
  const l = darkBg ? edge + (Math.min(0.9, edge + span) - edge) * (1 - t) : edge - (edge - Math.max(0.15, edge - span)) * (1 - t);
  return hexOf(fromHls([h, l, s]));
}

/* ---------------------------------------------------------------- DOS projections (as presets.py) */
const ORB_RE = /^\s*([A-Z][a-z]?)\s*(?:\((all|s|p|d|f)\))?\s*$/;
export function parseProjection(label) {
  if (String(label).trim() === "Total") return { element: null, orbital: "total" };
  const m = ORB_RE.exec(String(label));
  return m ? { element: m[1], orbital: m[2] || "all" } : { element: null, orbital: null };
}
/** "Mg (s)" -> "Mg s", "Mg (all)" -> "Mg", "Total" -> "Total". */
export function legendName(label) {
  const p = parseProjection(label);
  if (!p.element) return String(label).trim();
  return p.orbital === "all" ? p.element : `${p.element} ${p.orbital}`;
}
export function autoDosMode(projections) {
  const elements = [...new Set(projections.filter((p) => p.element).map((p) => p.element))];
  const A = SHARED.auto_dos;
  if (!elements.length) return "total";
  if (elements.length <= A.orbital_max_elements) return "orbital";
  const orbLines = projections.filter((p) => p.element && p.orbital !== "all" && p.orbital !== null).length;
  if (elements.length <= A.element_orbital_max_elements && orbLines <= A.max_lines) return "element-orbital";
  return "element";
}
const DARK_GRAYS = ["#ffffff", "#bdbdbd", "#8c8c8c", "#e0e0e0", "#a6a6a6", "#d0d0d0", "#b0b0b0", "#9a9a9a", "#eeeeee", "#949494"];
const RANK_STEPS = [0, 1 / 3, 2 / 3, 1];
const DASH_STEPS = ["solid", "dash", "dot", "dashdot"];
/** Each projection's weight between lo and hi (eV): the sum of |up| and |down| there (presets.py dos_weights). */
export function dosWeights(energy, projections, lo, hi) {
  const out = {};
  for (const p of projections) {
    let w = 0;
    energy.forEach((e, i) => {
      if (e === null || e < lo || e > hi) return;
      w += Math.abs(p.up[i] || 0);
      if (p.down) w += Math.abs(p.down[i] || 0);
    });
    out[p.label] = w;
  }
  return out;
}
/* each orbital projection's place among its element's orbitals: by weight, largest first; else s, p, d, f */
function places(parsed, weights) {
  if (weights && !Object.keys(weights).length) weights = null;
  const order = { s: 0, p: 1, d: 2, f: 3 }, byEl = {}, out = {};
  parsed.forEach((p) => { if (p.element && p.orbital in order) (byEl[p.element] = byEl[p.element] || []).push(p); });
  for (const items of Object.values(byEl)) {
    const sorted = [...items].sort((a, b) => (weights ? (weights[b.label] || 0) - (weights[a.label] || 0) : 0) || order[a.orbital] - order[b.orbital]);
    sorted.forEach((p, k) => { out[p.label] = k; });
  }
  return out;
}
/** Which projections a DOS draws and how: {mode, lines: [{label, name, color, dash, show, total}]}
 *  (presets.py dos_styles; `palette` is the preset's series colors as the plot draws them). With
 *  `weights` an element's orbitals take their shade and line style by weight (the dominant one strongest
 *  and solid); `spin`: both spins drawn, spin down dashed, so every projection is solid. */
export function dosStyles(labels, { mode = "auto", elementColors = {}, preset = "default", showTotal = true, bg = "#ffffff", palette = null, spin = false, weights = null } = {}) {
  const P = get(preset, "band"), pal = palette && palette.length ? palette : P.palette || SHARED.presets.default.palette;
  const gray = !!P.dashes, light = luminance(bg) > 0.18;
  if (weights && !Object.keys(weights).length) weights = null;
  const parsed = labels.map((label) => ({ ...parseProjection(label), label }));
  let used = !mode || mode === "auto" ? autoDosMode(parsed) : mode;
  if (!SHARED.dos_modes.includes(used)) used = "element";
  const hasAll = new Set(parsed.filter((p) => p.element && p.orbital === "all").map((p) => p.element));
  const elements = [...new Set(parsed.filter((p) => p.element).map((p) => p.element))];
  const place = places(parsed, weights), ecol = {}, edash = {}, eplace = {};
  elements.forEach((el, k) => { ecol[el] = readable((elementColors || {})[el] || pal[k % pal.length], bg); edash[el] = DASHES[k % DASHES.length]; eplace[el] = k; });
  const lines = [];
  let unnamed = 0;
  parsed.forEach((p) => {
    const line = { label: p.label, name: legendName(p.label), total: p.orbital === "total", color: null, dash: "solid", show: false };
    const el = p.element, orb = p.orbital, n = place[p.label] || 0, step = RANK_STEPS[Math.min(n, 3)];
    if (line.total) { line.show = used === "total" || showTotal; line.color = light ? "#000000" : "#ffffff"; }
    else if (!el) { line.show = used !== "total"; line.color = pal[unnamed++ % pal.length]; }
    else if (used === "element") {
      line.show = orb === "all" || !hasAll.has(el);
      line.color = orb === "all" || !spin ? ecol[el] : shade(ecol[el], orb, bg, 2, step);
      line.dash = orb === "all" || spin ? "solid" : weights ? DASH_STEPS[Math.min(n, 3)] : SHARED.orbital_dash[orb] || "solid";
    } else if (used === "orbital") {
      line.show = orb !== "all";
      const base = SHARED.orbital_colors[orb] || "#555555";
      if (spin && elements.length > 1) line.color = shade(base, orb, bg, 2, RANK_STEPS[Math.min(eplace[el], 3)]);
      else { line.color = readable(base, bg); line.dash = elements.length > 1 ? edash[el] : "solid"; }
    } else if (used === "element-orbital") {
      line.show = orb !== "all";
      if (weights) { line.color = shade(ecol[el], orb, bg, 2, step); line.dash = spin ? "solid" : DASH_STEPS[Math.min(n, 3)]; }
      else { line.color = shade(ecol[el], orb, bg); line.dash = spin ? "solid" : SHARED.orbital_dash[orb] || "solid"; }
    }
    if (gray && !line.total && line.show) {
      const m = lines.filter((x) => x.show && !x.total).length;
      line.color = light ? P.palette[(m + 1) % P.palette.length] : spin ? DARK_GRAYS[(m + 1) % DARK_GRAYS.length] : "#d9d9d9";
      line.dash = spin ? "solid" : P.dashes[m % P.dashes.length];
    }
    lines.push(line);
  });
  return { mode: used, lines };
}

/** The k-th series of a figure in a preset: {dash, marker} (Grayscale tells series apart by them). */
/* ---------------------------------------------------------------- the DOS layout (1.1.6 design; as presets.py)
   Shared by bandfig.js (the band/DOS figure), the gallery's DOS panes and mace plotting's figures. */
export const DOS_LAYOUT = SHARED.dos_layout;
const round3 = (x) => Math.floor(x * 1000 + 0.5) / 1000;
/** `#rrggbb` at opacity `a` (3 places) as an rgba() color; another color as it is. */
export function rgba(color, a) {
  if (typeof color !== "string" || !HEX.test(color)) return color;
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(color.slice(i, i + 2), 16));
  return `rgba(${r},${g},${b},${round3(a)})`;
}
/** The light fills under the spin up and spin down totals, in the active pair's colors. */
export function spinFills(pair, pid = "default", family = "band") {
  const a = round3((get(pid, family).dos.total_fill || 0.18) * DOS_LAYOUT.spin_fill);
  return { up: rgba(pair.up, a), down: rgba(pair.down, a) };
}
/** The light fill under a closed-shell run's total, in its neutral band color. */
export function neutralFill(color, pid = "default", family = "band") {
  return rgba(color, round3((get(pid, family).dos.total_fill || 0.18) * DOS_LAYOUT.neutral_fill));
}
function hueChroma(color) {
  const [r, g, b] = rgbOf(color), mx = Math.max(r, g, b), mn = Math.min(r, g, b), c = mx - mn;
  if (c === 0) return [0, 0];
  let h;
  if (mx === r) h = (((g - b) / c) % 6 + 6) % 6;
  else if (mx === g) h = (b - r) / c + 2;
  else h = (r - g) / c + 4;
  return [h * 60, c];
}
/** The projection colors, largest first: the preset's "proj" colors clear of the active spin pair's hues
 *  (`avoid`), readable on `bg`; null: the preset keeps dosStyles' colors (Print grayscale). */
export function projectionPalette(pid = "default", avoid = null, bg = "#ffffff", family = "band") {
  const pool = get(pid, family).dos.proj;
  if (!pool || !pool.length) return null;
  const gap = DOS_LAYOUT.hue_gap, chroma = DOS_LAYOUT.chroma;
  const spins = (avoid || []).filter((c) => typeof c === "string" && HEX.test(c)).map(hueChroma).filter(([, c]) => c >= chroma).map(([h]) => h);
  const clear = (c) => {
    const [h, ch] = hueChroma(c);
    return ch < chroma || spins.every((s) => Math.min(Math.abs(h - s), 360 - Math.abs(h - s)) >= gap);
  };
  const out = pool.filter(clear);
  return (out.length ? out : pool.slice()).map((c) => readable(c, bg));
}
/** The projections a DOS draws by default (presets.py pick_projections): {order, shown}. */
export function pickProjections(lines, weights, { nmin = DOS_LAYOUT.pick_min, nmax = DOS_LAYOUT.pick_max, frac = DOS_LAYOUT.pick_frac } = {}) {
  const cand = lines.filter((l) => l.show && !l.total).map((l) => l.label);
  const w = Object.fromEntries(cand.map((l) => [l, +((weights || {})[l] || 0)]));
  const tl = lines.find((l) => l.total);
  const tot = (tl ? +((weights || {})[tl.label] || 0) : 0) || cand.reduce((a, l) => a + w[l], 0);
  const order = cand.map((l, k) => [l, k]).sort((a, b) => w[b[0]] - w[a[0]] || a[1] - b[1]).map(([l]) => l);
  const shown = order.filter((l, k) => k < nmin || w[l] >= frac * tot).slice(0, nmax);
  return { order, shown };
}
/** `y` convolved with a normalized Gaussian of width `sigma` (eV), out to 4 sigma (presets.py gauss_smooth). */
export function gaussSmooth(y, energy, sigma) {
  const vals = y.map((v) => (v === null || v === undefined ? 0 : +v));
  if (!sigma || sigma <= 0 || vals.length < 3) return vals;
  const steps = [];
  for (let i = 0; i < energy.length - 1; i++) steps.push(energy[i + 1] - energy[i]);
  steps.sort((a, b) => a - b);
  const m = Math.floor(steps.length / 2), de = steps.length % 2 ? steps[m] : (steps[m - 1] + steps[m]) / 2;
  if (!de || de <= 0) return vals;
  const n = Math.ceil(4 * sigma / de);
  let k = [];
  for (let j = 0; j < 2 * n + 1; j++) k.push(Math.exp(-0.5 * ((j - n) * de / sigma) ** 2));
  let ks = 0;
  for (const v of k) ks += v;
  k = k.map((v) => v / ks);
  const size = vals.length, out = [];
  for (let i = 0; i < size; i++) {
    let acc = 0;
    for (let j = Math.max(0, n - i); j < Math.min(2 * n + 1, size - i + n); j++) acc += vals[i + j - n] * k[j];
    out.push(acc);
  }
  return out;
}
/** The energy ranges in [lo, hi] the DOS run did not compute (outside `window`): [[y0, y1], ...]. */
export function dosWindowGaps(window, lo, hi) {
  if (!window || window.length !== 2) return [];
  const out = [];
  if (window[0] > lo) out.push([lo, Math.min(window[0], hi)]);
  if (window[1] < hi) out.push([Math.max(window[1], lo), hi]);
  return out.filter((g) => g[1] > g[0]);
}
/** Where a DOS-alone legend goes: "top" or "bottom" (inside, right) where no curve enters a box fw x fh
 *  (fractions of the plot), else "outside". curves: [[xs, ys], ...] in data coordinates. */
export function legendSpot(curves, xr, yr, fw, fh) {
  const x0 = Math.min(...xr), x1 = Math.max(...xr), y0 = Math.min(...yr), y1 = Math.max(...yr), xcut = x1 - fw * (x1 - x0);
  for (const [spot, lo, hi] of [["top", y1 - fh * (y1 - y0), y1], ["bottom", y0, y0 + fh * (y1 - y0)]]) {
    const hit = curves.some(([xs, ys]) => xs.some((x, i) => x !== null && ys[i] !== null && ys[i] >= lo && ys[i] <= hi && x > xcut));
    if (!hit) return spot;
  }
  return "outside";
}
/** A tick step that puts about three ticks between 0 and m. */
export function niceStep(m) {
  for (const s of [0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100, 200, 250, 500]) if (m / s <= 3.2) return s;
  return 1000;
}
/** A mirrored DOS axis's ticks: magnitudes on both sides of zero ({vals, text}). */
export function mirroredTicks(lo, hi) {
  const step = niceStep(Math.max(Math.abs(lo), Math.abs(hi))), k0 = Math.ceil(lo / step - 1e-9), k1 = Math.floor(hi / step + 1e-9);
  const vals = [];
  for (let k = k0; k <= k1; k++) vals.push(Math.floor(k * step * 1e6 + 0.5) / 1e6);
  return { vals, text: vals.map((v) => `${Math.abs(v)}`) };
}

export function seriesStyle(pid, family, k) {
  const p = get(pid, family);
  return { dash: p.dashes ? p.dashes[k % p.dashes.length] : "solid", marker: p.markers ? p.markers[k % p.markers.length] : null };
}

/* ---------------------------------------------------------------- map colormaps */
/* VESTA-like section map: blue, cyan, green, yellow, red (its default rainbow look) */
export const VESTA_SCALE = [[0, "#0000ff"], [0.25, "#00ffff"], [0.5, "#00ff00"], [0.75, "#ffff00"], [1, "#ff0000"]];
const VESTA_DIVERGING = [[0, "#00a0c8"], [0.5, "#ffffff"], [1, "#e6c200"]];
const GRAY_DIVERGING = [[0, "#ffffff"], [0.5, "#bfbfbf"], [1, "#000000"]];
/** A map's plotly colorscale for a preset: {colorscale, reversescale, contours: "filled"|"lines"|"both"}.
 *  Signed data (spin, a difference) draws + blue and − red (plotly.js RdBu: low blue, high red, reversed),
 *  as mace plotting and the 3D view; VESTA-like + yellow, − cyan. */
export function mapScale(pid, signed) {
  const p = get(pid, "maps"), m = p.maps;
  if (signed) {
    if (m.diverging === "vesta") return { colorscale: VESTA_DIVERGING, reversescale: false, contours: m.contours };
    if (m.diverging === "gray") return { colorscale: GRAY_DIVERGING, reversescale: false, contours: m.contours };
    return { colorscale: m.diverging, reversescale: true, contours: m.contours };
  }
  if (m.sequential === "vesta") return { colorscale: VESTA_SCALE, reversescale: false, contours: m.contours };
  return { colorscale: m.sequential, reversescale: m.sequential === "Greys", contours: m.contours };   // british-ok: a colormap name
}

/* ---------------------------------------------------------------- a figure in a preset */
/** Every plot builder passes its figure through here before plotly draws it (the screen and the PNG/SVG
 *  exports alike; app.js styleFigure gives `tok`, a token as the family's plots draw it). Default: unchanged.
 *  Publication: the axes, frame and ticks in ink (black on a light plot), outside ticks, larger tick labels,
 *  no decorative grid (an axis that asks for its grid, the band structure's high-symmetry k-points, keeps it,
 *  in ink), lines a quarter heavier. Presentation: larger text, thicker lines and frame. Print grayscale:
 *  Publication's frame, and the series of spectra and run charts told apart by line style and markers.
 *  mace plotting draws its band/DOS figures through the same rules (mace/plotting/plotly_bands.py style_figure,
 *  tests/test_plotly_bands.py compares the two). */
export function styleFigure(traces, layout, family, pid, tok) {
  pid = resolveId(pid, family);
  if (!layout || pid === "default") return { traces, layout };
  const pr = get(pid, family);
  const inky = pid === "publication" || pid === "grayscale";
  const ink = (inky && tok("--plot-ink")) || tok("--text"), base = (layout.font && layout.font.size) || 12;
  const fd = pr.font_delta || 0, outside = pr.ticks === "outside", nogrid = pr.grid === "none";
  layout.font = { ...(layout.font || {}), color: ink, size: base + fd };
  for (const [k, ax] of Object.entries(layout)) {
    if (!/^[xy]axis\d*$/.test(k) || !ax || typeof ax !== "object") continue;
    if (ax.showline !== false && !ax.overlaying) Object.assign(ax, { showline: true, linecolor: ink, linewidth: +(1.0 * pr.frame).toFixed(2) });
    if (ax.ticks !== "" && outside) Object.assign(ax, { ticks: "outside", ticklen: Math.round(4 * pr.frame), tickwidth: +(0.9 * pr.frame).toFixed(2), tickcolor: ink });
    if (!(ax.tickfont && ax.tickfont.size)) ax.tickfont = { ...(ax.tickfont || {}), color: ink, size: base + fd + 1 };   // a sized one (the peak labels) stays
    if (nogrid) { if (ax.showgrid === true) Object.assign(ax, { gridcolor: ink, gridwidth: 0.6 }); else ax.showgrid = false; }
    if (ax.zeroline && inky) Object.assign(ax, { zerolinecolor: ink, zerolinewidth: 0.8 });
    if (ax.title && typeof ax.title === "object") ax.title = { ...ax.title, font: { ...(ax.title.font || {}), color: ink, size: base + fd + 2 } };
  }
  const gray = !!pr.dashes && (family === "spectra" || family === "scf");
  let n = 0;
  for (const t of traces || []) {
    if (!t) continue;
    if (t.line && typeof t.line.width === "number" && t.line.width > 0 && t.mode !== "markers") t.line = { ...t.line, width: +(t.line.width * pr.line).toFixed(2) };
    if (gray && t.type === "scatter" && !t.meta && /lines/.test(t.mode || "lines") && Array.isArray(t.x) && t.x.length > 1) {
      t.line = { ...(t.line || {}), color: tok(`--plot-${(n % 10) + 1}`), dash: pr.dashes[n % pr.dashes.length] };
      if (/markers/.test(t.mode || "")) t.marker = { ...(t.marker || {}), symbol: pr.markers[n % pr.markers.length], color: t.line.color };
      n += 1;
    }
    if (t.marker && typeof t.marker.size === "number" && fd > 2) t.marker = { ...t.marker, size: Math.round(t.marker.size * 1.4) };
    // a legend group's title (the band/DOS legend's Spin and Projections) at the legend's size
    if (t.legendgrouptitle && t.legendgrouptitle.font) t.legendgrouptitle = { ...t.legendgrouptitle, font: { ...t.legendgrouptitle.font, size: base + fd, ...(inky ? { color: ink } : {}) } };
    // Print grayscale: marks in ink (a spectrum's mode markers are the accent otherwise)
    if (pr.dashes && t.marker && t.meta === "peaks") t.marker = { ...t.marker, color: ink };
  }
  // every legend (a band + DOS figure has two)
  for (const k of Object.keys(layout)) {
    if (/^legend\d*$/.test(k) && layout[k]) {
      layout[k] = { ...layout[k], font: { ...(layout[k].font || {}), color: ink, size: base + fd } };
      if (layout[k].title && layout[k].title.font) layout[k].title = { ...layout[k].title, font: { ...layout[k].title.font, size: base + fd, ...(inky ? { color: ink } : {}) } };
    }
  }
  // the figure's words (the DOS's spin labels, its smoothing and no-data notes) grow with the text
  if (Array.isArray(layout.annotations)) layout.annotations = layout.annotations.map((a) => (a && a.font && typeof a.font.size === "number" ? { ...a, font: { ...a.font, size: a.font.size + fd } } : a));
  // the larger tick labels: room for the top one (a range's last tick, "6" of ±6 eV) above the frame
  if (layout.margin && typeof layout.margin.t === "number" && layout.margin.t < 16 + 2 * fd) layout.margin = { ...layout.margin, t: 16 + 2 * fd };
  if (fd > 2 && layout.margin && typeof layout.margin.l === "number") layout.margin = { ...layout.margin, l: layout.margin.l + 3 * fd, b: (layout.margin.b || 40) + 2 * fd };
  return { traces, layout };
}
