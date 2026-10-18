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
