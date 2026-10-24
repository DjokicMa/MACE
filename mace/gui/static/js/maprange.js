/* The color range of a 2D map (a fort.25 section, a cube slice): robust limits, so the spikes at the
 * nuclei of a density or potential map do not take the whole color scale. The 1st and 99th
 * percentiles of the values (numpy's linear percentile), symmetric about 0 for a signed map (spin,
 * a difference, a potential: a diverging scale centered on zero). mace plotting's cube slices use the
 * same rule (mace/plotting/engines/crystal_cubeviz_plotly.py robust_range; tests/test_gui_map_range.py
 * runs both). No DOM. */
export function percentile(sorted, q) {
  if (!sorted.length) return NaN;
  const pos = (q / 100) * (sorted.length - 1), lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}
/** [lo, hi] of `values` (an array, or an array of rows): robust (1st to 99th percentile), or min to max
 *  with `full`; `signed`: symmetric about 0, the larger of the two ends. */
export function robustRange(values, { signed = false, full = false, lo = 1, hi = 99 } = {}) {
  const flat = [];
  const walk = (v) => { if (Array.isArray(v)) v.forEach(walk); else if (typeof v === "number" && Number.isFinite(v)) flat.push(v); };
  walk(values);
  if (!flat.length) return [0, 1];
  flat.sort((a, b) => a - b);
  let a = full ? flat[0] : percentile(flat, lo), b = full ? flat[flat.length - 1] : percentile(flat, hi);
  if (signed) { const m = Math.max(Math.abs(a), Math.abs(b)) || 1; a = -m; b = m; }
  if (!(b > a)) { const c = a; a = c - (Math.abs(c) || 1) * 0.5; b = c + (Math.abs(c) || 1) * 0.5; }
  return [a, b];
}

/* A surface colored by a potential (the Volume view's color bar): what the ends of its color range mean and
 * where its ticks go. Sign words only where the range holds both signs; a range of one sign says so ("all
 * positive") and names its ends lowest/highest (crystal_cubeviz_plotly.potential_scale_words, the CLI's
 * caption, says the same: tests/test_volume_colorbar.py runs both). `chem`: chemistry colors (red low). */
export function potentialWords(vmin, vmax, chem = true) {
  const [lowC, highC] = chem ? ["red", "blue"] : ["blue", "red"];
  if (vmin < 0 && vmax > 0) return { spans: true, low: `${lowC} = negative (electron-rich)`, high: `${highC} = positive (electron-poor)`, note: "" };
  const note = vmin >= 0 && vmax > 0 ? "all positive" : vmax <= 0 && vmin < 0 ? "all negative" : "zero";
  return { spans: false, low: `${lowC} = lowest`, high: `${highC} = highest`, note };
}
/** Round tick values in [lo, hi] (about `n` of them, 1-2-5 steps; zero among them when the range spans it). */
export function niceTicks(lo, hi, n = 5) {
  if (!(Number.isFinite(lo) && Number.isFinite(hi)) || !(hi > lo)) return Number.isFinite(lo) ? [lo] : [];
  const raw = (hi - lo) / Math.max(1, n - 1), mag = 10 ** Math.floor(Math.log10(raw)), r = raw / mag;
  const step = (r <= 1 ? 1 : r <= 2 ? 2 : r <= 5 ? 5 : 10) * mag;
  const out = [];
  for (let k = Math.ceil(lo / step - 1e-9); k * step <= hi + step * 1e-9; k++) out.push(Math.abs(k) < 1e-12 ? 0 : Number((k * step).toPrecision(12)));
  return out;
}
/** A tick's text: 0, 0.05, −0.25, 1.2×10⁻⁴ (a real minus sign and superscripts). */
export function tickText(v) {
  if (v === 0) return "0";
  const a = Math.abs(v), sgn = v < 0 ? "−" : "";
  if (a >= 1e-3 && a < 1e4) return sgn + String(Number(a.toPrecision(3)));
  const e = Math.floor(Math.log10(a)), m = Number((a / 10 ** e).toPrecision(3));
  const sup = String(e).replace("-", "⁻").replace(/\d/g, (c) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[c]);
  return `${sgn}${m === 1 ? "" : `${m}×`}10${sup}`;
}
