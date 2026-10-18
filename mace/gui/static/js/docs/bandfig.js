/* The band structure / DOS figure (plot.js draws it, the Publication Figure's mace plotting draws
 * the same one): the plotly traces and layout from the arrays mace plotting's readers return
 * (mace/gui/results.py band_data, dos_data). No DOM here: plot.js passes what it reads from the page
 * (the tokens, the spin pair, the settings, the pane's width) in `env`, and node runs this module for
 * tests/test_plotly_bands.py, which compares it with mace/plotting/plotly_bands.py, its port for
 * mace plotting. A change here is a change there.
 *
 * env = { css(name): a token as the band plots draw it in the look's preset, look: the style (lookOf: the
 *   presets.py STYLE_OPTIONS names), tone: "light" | "dark" (the plot's background), paper: {paper_bgcolor,
 *   plot_bgcolor}, presets: plotpresets.js (or null), elementColors, width: the pane's width in px, marginR:
 *   the right margin as drawn, height: the pane's height in px (where a DOS-alone legend fits; default 420),
 *   fermiKey: an "E_F" legend entry for the dotted line (mace plotting's files,
 *   which have no note line above them) } */
import { ENERGY_UNITS, UNIT_LABEL, legendTop, legendSide } from "../plotbase.js";
import * as K from "../kpath.js";
import * as PX from "../plotpresets.js";

export const PAL = (css) => Array.from({ length: 10 }, (_, k) => css(`--plot-${k + 1}`));
// mace plotting's reader names the zone center Γ (a bare G of MACE's default paths too, B1): GAMMA is left over
// from an edited path; a bare G stays G (seekpath's tI2/mC2 G is not the center)
export const kLabel = (l) => String(l || "").replace(/GAMMA/g, "Γ");
const DASH_WORD = { dash: "dashed", dot: "dotted", dashdot: "dash-dot", solid: "solid" };
export const dashWord = (d) => DASH_WORD[d] || d;

/** A plot's style (presets.py STYLE_OPTIONS names): its own choice (`o`, the plot's options) over `base`
 *  (the user's defaults: Settings, or mace plotting's flags). */
export function lookOf(o, base, PP) {
  const out = {};
  for (const opt of PP.STYLE_OPTIONS) {
    const v = o ? o[opt.name] : undefined;
    out[opt.name] = v !== null && v !== undefined ? v : base && base[opt.name] !== null && base[opt.name] !== undefined ? base[opt.name] : opt.default;
  }
  return out;
}
/** The look's preset id (band family) and spin pair {up, down, dash}. */
export function lookParts(env) {
  const PP = env.presets, look = env.look || {};
  const preset = PP ? PP.resolveId(look.style || "default", "band") : "default";
  const spin = PP ? { ...PP.spinColors(preset, "band", env.tone || "light", look.spin_colors, look.spin_up_color, look.spin_down_color), dash: look.spin_down_style || "dash" } : null;
  // a run that is not spin-polarized: its one set of bands and one-sided DOS total in a neutral color
  if (spin) spin.bands = PP.bandColor(preset, "band", env.tone || "light", look.band_color, (env.css && (env.css("--plot-ink") || env.css("--text"))) || undefined);   // undefined: bandColor's own black
  return { preset, spin, look };
}

/* How a DOS is drawn (Settings › Plots): which projections and in which colors (DOS colors: auto, total,
   by element, by orbital, element × orbital, plotpresets.js dosStyles, as mace plotting --dos-colors) and
   how the two spins sit: "mirrored" (spin down on the left), "overlaid" (both on the right, down in spin
   down's line style) or "one" (one-sided). Spin DOS Auto (B2, one rule with mace plotting): a
   spin-polarized run mirrored, a closed-shell run one-sided. With both spins drawn every projection is
   solid (spin down is the dashed line); an element's orbitals are shaded by their weight in the window. */
export function dosLook(dos, env, o = {}) {
  const css = env.css, pal = PAL(css), PP = env.presets, { preset, look } = lookParts(env);
  const labels = dos.projections.map((p) => p.label);
  const bg = css("--plot-bg") || "#ffffff";
  const want = look.dos_spin || "auto";
  const side = want === "auto" ? (dos.spin_polarized ? "mirrored" : "one") : want === "overlaid" ? (dos.spin_polarized ? "overlaid" : "one") : "mirrored";
  const ef = dos.fermi || 0, shift = o.shift || 0;
  const emin = Number.isFinite(o.emin) ? o.emin : -6, emax = Number.isFinite(o.emax) ? o.emax : 6;
  const weights = PP ? PP.dosWeights(dos.energy, dos.projections, ef + shift + emin, ef + shift + emax) : {};
  const out = PP ? PP.dosStyles(labels, { mode: look.dos_colors || "auto", elementColors: env.elementColors || {},
    preset, showTotal: look.dos_total !== false, bg, palette: pal.slice(1).concat(pal[0]),
    spin: side !== "one", weights })
    : { mode: "total", lines: labels.map((l, k) => ({ label: l, name: l, total: l === "Total", color: l === "Total" ? css("--text") : pal[(k + 1) % pal.length], dash: "solid", show: true })) };
  out.side = side;
  out.byLabel = Object.fromEntries(out.lines.map((l) => [l.label, l]));
  // the projections drawn by default (o.hidden not set): the 2-4 carrying the DOS in the window; the
  // rest are hidden, listed in Plot options › Projections (pickProjections, one rule with mace plotting)
  const pk = PX.pickProjections(out.lines, weights);
  out.order = pk.order;
  out.hidden = Array.isArray(o.hidden) ? o.hidden.slice() : pk.order.filter((l) => !pk.shown.includes(l));
  // DOS colors Auto: the projections in the preset's projection colors, largest first, clear of the
  // spin pair's hues when both spins are drawn (Print grayscale keeps its grays)
  if (PP && (look.dos_colors || "auto") === "auto") {
    const sp = lookParts(env).spin, ppal = PX.projectionPalette(preset, side !== "one" && sp ? [sp.up, sp.down] : null, bg);
    if (ppal) pk.order.forEach((l, k) => { out.byLabel[l].color = ppal[k % ppal.length]; out.byLabel[l].dash = "solid"; });
  }
  return out;
}
// an alpha = beta run's DOS is the summed total (results.py dos_data spin_summed): drawn with both
// spins, each side holds half of it; a closed-shell (one-block) run drawn mirrored on request: spin
// down is spin up's mirror image
export const upOf = (dos, p, side) => (dos.spin_summed && side && side !== "one" ? p.up.map((v) => v / 2) : p.up);
export const downOf = (dos, p) => (dos.spin_polarized && p.down ? p.down : p.up.map((v) => (dos.spin_summed ? -v / 2 : -v)));
export function dosRange(dos, lo, hi, opts, look) {
  let m = 0;
  const projs = dos.projections
    .filter((p) => !(((look ? look.hidden : opts.hidden) || []).includes(p.label) || (look && look.byLabel[p.label] && !look.byLabel[p.label].show)))
    .map((p) => [upOf(dos, p, look ? look.side : null), downOf(dos, p)]);
  dos.energy.forEach((e, i) => {
    if (e < lo || e > hi) return;
    projs.forEach(([up, down]) => {
      if (opts.spin !== "down") m = Math.max(m, Math.abs(up[i] || 0));
      if ((!look || look.side !== "one") && opts.spin !== "up") m = Math.max(m, Math.abs(down[i] || 0));
    });
  });
  m = m > 0 ? m * 1.08 : 1;
  const side = look ? look.side : dos.spin_polarized ? "mirrored" : "one";
  if (side === "mirrored") return opts.spin === "both" ? [-m, m] : opts.spin === "down" ? [-m, 0] : [0, m];
  return [0, m];
}
// Band and DOS lines keep every point (line.simplify: false, as mace plotting's): plotly simplifies at the
// 1x layout size, so a 300 dpi export showed straight facets along curved bands (1.1.6 review M3)
/* A Fermi shift s moves the data by -s: the energy zero is E_F + s (or E_vac + s on the vacuum scale)
   and the dotted line, which stays put, marks E_F + s (mace plotting's matplotlib energy_axis_label) */
export const shiftText = (shift) => (shift ? ` ${shift > 0 ? "+" : "−"} ${Math.abs(shift).toFixed(2)} eV` : "");
export const energyTitle = (ref, shift) => {
  const zero = ref === "vacuum" ? "E<sub>vac</sub>" : "E<sub>F</sub>";
  return shift ? `E − (${zero}${shiftText(shift)})` : `E − ${zero}`;
};
export function joinBands(x, bands, shift) {
  const xs = [], ys = [], idx = [];
  bands.forEach((b, k) => { for (let i = 0; i < x.length; i++) { xs.push(x[i]); ys.push(b[i] === null ? null : b[i] - shift); idx.push(k); } xs.push(null); ys.push(null); idx.push(-1); });
  return { xs, ys, idx };
}
/* k-path tick labels that never overlap: ticks at the same k (a writer that repeats a break's
   point) become one "A|B" label; a label too close to its neighbor for both to fit (seekpath's
   Y and F₀ 0.04 Å⁻¹ apart) drops to a second row instead of overlapping or turning the axis's
   labels sideways. Widths are estimated from the text at the axis font size. */
export const TICK_FONT = 12;
export function labelWidth(html) {
  const sub = (html.match(/<sub>(.*?)<\/sub>/g) || []).join("").replace(/<[^>]+>/g, "").length;
  const main = html.replace(/<sub>.*?<\/sub>/g, "").replace(/<[^>]+>/g, "").length;
  return (main * 0.62 + sub * 0.45) * TICK_FONT + 6;
}
export function kTicks(ticks, labels, range, widthPx) {
  const vals = [], text = [];
  ticks.forEach((t, i) => {
    const lab = kLabel(labels[i]), k = vals.length - 1;
    if (k >= 0 && Math.abs(t - vals[k]) < 1e-5) { if (text[k] !== lab) text[k] = `${text[k]}|${lab}`; return; }
    vals.push(t); text.push(lab);
  });
  const [x0, x1] = range[0] <= range[1] ? range : [range[1], range[0]], span = x1 - x0 || 1;
  const right = [-Infinity, -Infinity];
  let rows = 1;
  const out = text.map((lab, i) => {
    if (vals[i] < x0 - 1e-9 || vals[i] > x1 + 1e-9 || !lab) return lab;
    const px = ((vals[i] - x0) / span) * widthPx, half = labelWidth(lab) / 2;
    let row = px - half >= right[0] + 4 ? 0 : px - half >= right[1] + 4 ? 1 : right[0] <= right[1] ? 0 : 1;
    right[row] = px + half;
    if (row) rows = 2;
    return row ? ` <br>${lab}` : lab;     // plotly drops an empty first line: a no-break space holds it
  });
  return { vals, text: out, rows };
}
/** The band panel's k ticks for a pane `width` px wide whose right margin is `r`. */
export function bandTickLayout(band, range, dos, width, r = 14) {
  const w = Math.max(200, ((width || 900) - 58 - r) * (dos ? 0.7 : 1));
  const t = kTicks(band.ticks, band.labels, range, w);
  return { "xaxis.tickvals": t.vals, "xaxis.ticktext": t.text, "margin.b": 42 + (t.rows - 1) * 15 };
}

/* ------------------------------------------------------------ the k-path as edited (kpath_editor.js)
   The computed path comes from the tick labels ("A|L" is a jump); each segment is the data
   between its two ticks. An edited path lays those slices out again in its order, reversed
   where it says so, with a gap at each jump; hidden segments are left out. */
export function pathSegments(band) {
  const labs = band.labels.map((l, i) => l || `P${i + 1}`), tokens = [];
  labs.forEach((l) => { const [a, b] = String(l).split("|"); if (b !== undefined) tokens.push(a, "|", b); else tokens.push(a); });
  const segs = K.fromTokens(tokens);
  return segs.length === band.ticks.length - 1 ? segs : null;
}
export function sliceOf(band, n) {
  // CRYSTAL: the tick's point ends a segment; after a jump the next segment starts one point (and a null) later
  const near = (t) => { let best = -1, bd = Infinity; band.x.forEach((x, i) => { if (x !== null && Math.abs(x - t) < bd) { bd = Math.abs(x - t); best = i; } }); return best; };
  let a = near(band.ticks[n]);
  const b = near(band.ticks[n + 1]);
  if (n > 0 && String(band.labels[n]).includes("|")) { a += 1; while (a < b && band.x[a] === null) a += 1; }
  return [a, b];
}
/** The band data along the edited path `edit` (null: the computed one, unchanged). */
export function editedBand(band, edit) {
  if (!band || !edit) return band;
  const computed = pathSegments(band);
  if (!computed) return band;
  const used = new Set(), x = [], up = band.up.map(() => []), down = band.down ? band.down.map(() => []) : null;
  const ticks = [], labels = [];
  let x0 = 0, prev = null;
  const segX = {};
  edit.forEach((s, ei) => {
    if (s.hidden) return;
    let n = computed.findIndex((c, i) => !used.has(i) && K.canonical(c.from) === K.canonical(s.from) && K.canonical(c.to) === K.canonical(s.to)), rev = false;
    if (n < 0) { n = computed.findIndex((c, i) => !used.has(i) && K.canonical(c.from) === K.canonical(s.to) && K.canonical(c.to) === K.canonical(s.from)); rev = n >= 0; }
    if (n < 0) return;                         // not a segment of these data (a preset of another path)
    used.add(n);
    const [a, b] = sliceOf(band, n);
    let idx = []; for (let i = a; i <= b; i++) if (band.x[i] !== null) idx.push(i);
    if (rev) idx = idx.reverse();
    const xa = band.x[idx[0]];
    const cont = prev && K.joined(prev, s);
    if (prev && !cont) {                      // a jump: a gap, one tick for both labels
      x.push(null); up.forEach((u) => u.push(null)); if (down) down.forEach((u) => u.push(null));
      labels[labels.length - 1] = `${labels[labels.length - 1]}|${K.display(s.from)}`;
    }
    if (!prev) { ticks.push(0); labels.push(K.display(s.from)); }
    idx.forEach((i, k) => {
      if (cont && k === 0) return;            // the shared point is already there
      x.push(x0 + Math.abs(band.x[i] - xa));
      band.up.forEach((col, j) => up[j].push(col[i]));
      if (down) band.down.forEach((col, j) => down[j].push(col[i]));
    });
    const xs0 = x0;
    x0 += Math.abs(band.x[idx[idx.length - 1]] - xa);
    segX[ei] = [xs0, x0];
    ticks.push(x0); labels.push(K.display(s.to));
    prev = s;
  });
  return x.length ? { ...band, x, up, down, ticks, labels, edited: true, segX } : band;
}

/* ------------------------------------------------------------ the legend (1.1.6 design)
   Two titled groups: Spin (neutral keys: spin up solid, spin down in its line style; a closed-shell
   run "α = β (closed shell)") and Projections (color swatches only). A band plot alone: one line above
   the frame; a DOS alone: inside the plot where it covers no curve, else right of it; band + DOS: right
   of the plot. Keys are legend-only entries (meta "key-..."): a click on them hides nothing. */
export const GROUP_TITLE = (css) => ({ size: 11, color: css("--text-2") });
/** The Spin group's keys: [{name, dash}] for a figure that draws both spins (o.spin: one of them) or one
 *  closed-shell set (`closed`: its words). */
export function spinKeys(spin2, o, spinDash, closed) {
  if (!spin2) return [{ name: closed || "α = β (closed shell)", dash: "solid", closed: true }];
  const out = [];
  if (o.spin !== "down") out.push({ name: "spin up (α)", dash: "solid" });
  if (o.spin !== "up") out.push({ name: "spin down (β)", dash: spinDash });
  return out;
}
/** The DOS-alone legend's box as a fraction of the plot {fw, fh}: from its entries and their words. */
export function legendBox(names, groups, width, height) {
  const chars = Math.max(0, ...names.map((n) => String(n).replace(/<[^>]+>/g, "").length));
  const wpx = 46 + 6.6 * chars, hpx = 10 + 19 * (names.length + groups);
  const pw = Math.max(100, (width || 520) - 72), ph = Math.max(100, (height || 420) - 52);
  return { fw: Math.min(0.9, wpx / pw), fh: Math.min(0.9, hpx / ph) };
}

/** The figure: {traces, layout, legendBase, absShown, dosMode}. `band`, `dos`: the readers' data
 *  (either may be null); `o`: the plot's options (plot.js DEFAULTS.bands). */
export function bandFigure(band, dos, o, env) {
  const css = env.css, pal = PAL(css), parts = lookParts(env);
  // the spin pair and spin down's line style (the look's Spin colors, the plot style's scheme by default; β dashed)
  const spin = parts.spin || { up: css("--accent"), down: pal[0], dash: "dash", bands: css("--plot-ink") || css("--text") };
  let absShown = false, dosMode = null;
  // the API's energies as they come (results.py, mace plotting's readers): E - E_F, or with a POTC run
  // E - E_vac with E_F at the vacuum shift (fermi), as mace plotting draws them; nothing is recomputed
  // here. The dotted line marks E_F; a Fermi shift s moves the data by -s, so the axis reads E - (E_F + s)
  const ref = (band || dos).reference, ef = (band || dos).fermi || 0, shift = o.shift || 0;
  // in the chosen unit (plotkit)
  const f = ENERGY_UNITS[o.unit] || 1, unit = UNIT_LABEL[o.unit] || "eV";
  const tf = (y) => (y === null ? null : y * f);
  const logE = !!o.ylog && !dos;                     // a DOS plot: Log Y is the DOS axis
  const lg = (ys) => (logE ? ys.map((y) => (y === null || y === 0 ? null : Math.abs(y))) : ys);
  absShown = logE;
  const grid = css("--line"), traces = [], shapes = [], annotations = [], keys = [];
  const ink = css("--plot-ink") || css("--text");
  const yr = o.yr || [(ef + o.emin) * f, (ef + o.emax) * f];   // the window is around E_F (mace plotting's too)
  // E_F dotted: a dashed line would read as spin down
  const fermiLine = { color: css("--text-2"), width: 1.2, dash: "dot" };
  const lay = {
    // the legends sit outside the data, in the top and right margins that fitLegends sizes to the
    // legends as drawn (a DOS alone: inside the plot when it covers no curve)
    margin: { l: 58, r: 14, t: 10, b: 42, autoexpand: false }, showlegend: true, hovermode: "closest", dragmode: "zoom",
    legend: legendTop(), ...(env.paper || {}),
    font: { color: css("--text"), size: 12, family: css("--font-sans") },
    modebar: { bgcolor: "rgba(0,0,0,0)", color: css("--text-2"), activecolor: css("--accent") },
    yaxis: { title: { text: `${logE ? "|" : ""}${energyTitle(ref, shift)}${logE ? "|" : ""} (${unit})` },
      type: logE ? "log" : "linear", range: logE ? undefined : yr, autorange: logE ? true : undefined,
      gridcolor: grid, zeroline: false, linecolor: grid, mirror: true, showline: true,
      showspikes: true, spikemode: "across", spikethickness: 1, spikecolor: css("--text-3"), spikedash: "dot" },
  };
  const look = dos ? dosLook(dos, env, o) : null, side = look ? look.side : null;
  const spin2 = !!((band && band.spin_polarized) || (look && side !== "one"));
  if (band) {
    lay.xaxis = { domain: dos ? [0, 0.7] : [0, 1],
      range: [band.ticks[0], band.ticks[band.ticks.length - 1]], showgrid: true, gridcolor: css("--border-strong"), zeroline: false,
      showline: true, mirror: true, linecolor: grid, showspikes: true, spikemode: "across", spikethickness: 1,
      spikecolor: css("--text-3"), spikedash: "dot", tickangle: 0 };
    // spin up solid and spin down dashed (Settings › Plots › Spin down line), in the spin pair of the plot style
    const accent = spin.up, downCol = spin.down;
    if (o.xr) lay.xaxis.range = o.xr;
    const tl = bandTickLayout(band, lay.xaxis.range, dos, env.width, env.marginR);
    lay.xaxis.tickvals = tl["xaxis.tickvals"]; lay.xaxis.ticktext = tl["xaxis.ticktext"]; lay.margin.b = tl["margin.b"];
    // beta over alpha, dashed (as mace plotting draws them): where the two coincide the line reads
    // alternately in both colors, so a spin-polarized run never looks closed-shell; a non-magnetic run
    // (alpha = beta within 10 meV) comes as one set and is drawn once whichever spin is chosen. The
    // legend's Spin keys name them (below)
    if (o.spin !== "down" || !band.spin_polarized) {
      const up = joinBands(band.x, band.up, shift);
      up.ys = lg(up.ys.map(tf));
      traces.push({ x: up.xs, y: up.ys, customdata: up.idx, xaxis: "x", yaxis: "y", mode: "lines", type: "scatter",
        name: band.spin_polarized ? "spin up (α)" : band.spin_identical ? "α = β (non-magnetic)" : "bands",
        showlegend: false, meta: "up", legendrank: 1,
        line: { color: band.spin_polarized ? accent : spin.bands, width: o.width, simplify: false }, hoverinfo: "none" });
    }
    if (band.spin_polarized && band.down && o.spin !== "up") {
      const dn = joinBands(band.x, band.down, shift);
      dn.ys = lg(dn.ys.map(tf));
      traces.push({ x: dn.xs, y: dn.ys, customdata: dn.idx, xaxis: "x", yaxis: "y", mode: "lines", type: "scatter",
        name: "spin down (β)", showlegend: false, meta: "down", legendrank: 2,
        line: { color: downCol, width: o.width, dash: spin.dash, simplify: false }, hoverinfo: "none" });
    }
    if (!logE) shapes.push({ type: "line", xref: "x domain", x0: 0, x1: 1, yref: "y", y0: ef * f, y1: ef * f, editable: true, name: "fermi",
      line: { ...fermiLine } });
  }
  const PP = env.presets;
  const smooth = PX.smoothSigma(parts.look && parts.look.dos_smooth);
  let curves = [];
  if (dos) {
    const ax = band ? "xaxis2" : "xaxis", xr = band ? "x2" : "x";
    const dlog = !!o.ylog;
    lay[ax] = { domain: band ? [0.73, 1] : [0, 1], title: { text: dlog ? "|DOS| (states/eV, log)" : "DOS (states/eV)" }, gridcolor: grid,
      zeroline: true, zerolinecolor: ink, zerolinewidth: 1.2, showline: true, mirror: true, linecolor: grid,
      type: dlog ? "log" : "linear", range: dlog ? undefined : dosRange(dos, shift + yr[0] / f, shift + yr[1] / f, o, look), autorange: dlog ? true : undefined };
    if (dlog) absShown = true;
    if (!band && o.xr) lay[ax].range = o.xr;
    // a mirrored DOS: magnitudes on both sides of zero
    if (side === "mirrored" && !dlog && lay[ax].range[0] < 0 && lay[ax].range[1] > 0) {
      const tk = PX.mirroredTicks(lay[ax].range[0], lay[ax].range[1]);
      lay[ax].tickvals = tk.vals; lay[ax].ticktext = tk.text;
    }
    dosMode = look.mode;
    const dl = "legend", both = side !== "one";
    const fills = PX.spinFills(spin, parts.preset), nfill = PX.neutralFill(spin.bands, parts.preset);
    const ys = dos.energy.map((e) => tf(e - shift));
    const sm = (xs) => (smooth ? PX.gaussSmooth(xs, dos.energy, smooth) : xs);
    const dx = (xs) => (o.ylog ? xs.map((v) => (v === null || v === 0 ? null : Math.abs(v))) : xs);
    const pos = (xs) => xs.map((v) => (v === null ? null : Math.abs(v)));
    // the total first (its fill under the projections), then the projections drawn (look.hidden: the rest)
    const drawn = dos.projections.map((p, k) => [p, k]).filter(([p]) => {
      const st = look.byLabel[p.label];
      return st && st.show && !look.hidden.includes(p.label);
    });
    drawn.sort((a, b) => (look.byLabel[b[0].label].total ? 1 : 0) - (look.byLabel[a[0].label].total ? 1 : 0));
    drawn.forEach(([p, k]) => {
      const st = look.byLabel[p.label], total = st.total, lw = total ? o.width + 0.2 : o.width - 0.2;
      // the total in the spin pair (both spins) or the neutral band color, lightly filled in its own color; the projections in their DOS colors
      const col = total ? (both ? spin.up : spin.bands) : st.color, dcol = total && both ? spin.down : col;
      const dash = st.dash || "solid", name = st.name;
      if (o.spin !== "down") {
        const x = dx(sm(upOf(dos, p, side)));
        traces.push({ x, y: ys, xaxis: xr, yaxis: "y", mode: "lines", type: "scatter", meta: "dos",
          name: both ? `${name} ↑` : name, legendgroup: p.label, legend: dl, showlegend: false, legendrank: 10 + k,
          line: { color: col, width: lw, dash, simplify: false },
          fill: total ? "tozerox" : "none", fillcolor: total ? (both ? fills.up : nfill) : undefined, hoverinfo: "none" });
        curves.push([x, ys]);
      }
      // spin down: mirrored (projections in their color, dashed; the total filled in spin down's color) or
      // overlaid on the right in spin down's line style
      if (both && o.spin !== "up") {
        const x = dx(sm(side === "overlaid" ? pos(downOf(dos, p)) : downOf(dos, p)));
        traces.push({ x, y: ys, xaxis: xr, yaxis: "y", mode: "lines", type: "scatter",
          meta: "dos", name: `${name} ↓`, legendgroup: p.label, legend: dl, showlegend: false, legendrank: 10 + k,
          line: { color: dcol, width: lw, dash: dash === "solid" ? spin.dash : dash, simplify: false },
          opacity: side === "overlaid" && dash !== "solid" ? 0.65 : undefined,
          fill: total && side === "mirrored" ? "tozerox" : "none", fillcolor: total && side === "mirrored" ? fills.down : undefined, hoverinfo: "none" });
        curves.push([x, ys]);
      }
    });
    // the Projections group: a swatch each (the total's in its neutral fill), largest first
    const rank = Object.fromEntries((look.order || []).map((l, i) => [l, i]));
    const sw = drawn.slice().sort((a, b) => {
      const ta = look.byLabel[a[0].label].total, tb = look.byLabel[b[0].label].total;
      if (ta !== tb) return ta ? -1 : 1;
      return (rank[a[0].label] ?? 999) - (rank[b[0].label] ?? 999) || a[1] - b[1];
    });
    sw.forEach(([p], i) => {
      const st = look.byLabel[p.label];
      const fill = st.total ? (both ? PX.rgba(ink, 0.12) : PX.rgba(spin.bands, 0.22)) : st.color;
      const edge = st.total ? (both ? css("--border-strong") : spin.bands) : st.color;
      keys.push({ x: [null], y: [null], xaxis: "x", yaxis: "y", mode: "markers", type: "scatter", name: st.name, meta: "key-proj",
        legend: "legend", legendgroup: "proj", legendrank: 100 + i, customdata: [p.label], hoverinfo: "skip", showlegend: true,
        marker: { symbol: "square", size: 11, color: fill, line: { color: edge, width: 1 } },
        legendgrouptitle: i === 0 ? { text: "Projections", font: GROUP_TITLE(css) } : undefined });
    });
    // which half is which: words over a mirrored DOS
    if (side === "mirrored" && o.spin === "both" && !dlog) {
      // each beside the zero line, on its own side: they never overlap in a narrow DOS panel
      const r = lay[ax].range, z = (0 - r[0]) / (r[1] - r[0]);
      annotations.push({ xref: `${xr} domain`, yref: "paper", x: z, y: 1, xanchor: "left", xshift: 3, yanchor: "bottom", showarrow: false,
        text: "<b>↑ up</b>", font: { size: 11, color: spin.up }, name: "spin-up-label" });
      annotations.push({ xref: `${xr} domain`, yref: "paper", x: z, y: 1, xanchor: "right", xshift: -3, yanchor: "bottom", showarrow: false,
        text: "<b>down ↓</b>", font: { size: 11, color: spin.down }, name: "spin-down-label" });
      lay.margin.t = Math.max(lay.margin.t, 24);
    }
    if (!logE) shapes.push({ type: "line", xref: `${xr} domain`, x0: 0, x1: 1, yref: "y", y0: ef * f, y1: ef * f, editable: !band, name: "fermi2",
      line: { ...fermiLine } });
    // CRYSTAL computed the DOS between dos.window's energies only: beyond them is no data (not zero
    // states), shaded lightly and, on a DOS alone, said so
    const y0 = Math.min(yr[0], yr[1]), y1 = Math.max(yr[0], yr[1]);
    const win = dos.window ? dos.window.map((w) => (w - shift) * f) : null;
    PX.dosWindowGaps(win, y0, y1).forEach(([g0, g1]) => {
      shapes.push({ type: "rect", xref: `${xr} domain`, x0: 0, x1: 1, yref: "y", y0: g0, y1: g1, line: { width: 0 },
        fillcolor: PX.rgba(ink, 0.05), layer: "below", name: "no-data" });
      if (!band && g1 - g0 >= 0.06 * (y1 - y0)) {
        const low = g0 <= y0;
        annotations.push({ xref: `${xr} domain`, yref: "y", x: 0.5, y: low ? g1 : g0, xanchor: "center", yanchor: low ? "top" : "bottom",
          showarrow: false, text: PX.DOS_LAYOUT.no_data, font: { size: 9, color: css("--text-3") }, name: "no-data" });
      }
    });
    // the smoothing is said in the plot (display only)
    if (smooth) {
      annotations.push({ xref: `${xr} domain`, yref: "y domain", x: 0.99, y: 0.01, xanchor: "right", yanchor: "bottom", showarrow: false,
        text: band ? `smoothed, σ = ${smooth} eV` : `Gaussian smoothing σ = ${smooth} eV (display only)`,
        font: { size: 10, color: css("--text-2") }, name: "dos-smooth" });
    }
  }
  // the Spin group (neutral keys: line style is the message)
  const closedWords = band && !band.spin_polarized && band.spin_identical ? "α = β (non-magnetic)" : "α = β (closed shell)";
  const sk = spinKeys(spin2, o, spin.dash, closedWords);
  const grouped = !!dos;
  keys.unshift(...sk.map((k, i) => ({ x: [null], y: [null], xaxis: "x", yaxis: "y", mode: "lines", type: "scatter", name: k.name, meta: "key-spin",
      legend: "legend", legendgroup: grouped ? "spin" : undefined, legendrank: 1 + i, hoverinfo: "skip", showlegend: true,
      line: { color: k.closed ? spin.bands : ink, width: 2, dash: k.dash },
      legendgrouptitle: grouped && i === 0 ? { text: "Spin", font: GROUP_TITLE(css) } : undefined })));
  // mace plotting's files have no note line above them: the dotted line gets a legend entry of its own
  if (env.fermiKey && !logE) {
    keys.push({ x: [null], y: [null], xaxis: "x", yaxis: "y", mode: "lines", type: "scatter", meta: "key-fermi",
      name: `E<sub>F</sub>${shiftText(shift)}`,
      legend: "legend", line: { ...fermiLine }, hoverinfo: "skip", legendrank: 5000 });
  }
  traces.push(...keys);
  // where the legend goes
  if (!dos) {
    lay.legend = legendTop({ title: { text: "Spin", side: "left", font: GROUP_TITLE(css) } });
  } else if (band) {
    lay.legend = legendSide({ traceorder: "grouped", tracegroupgap: 8 });
  } else {
    const groups = new Set(keys.map((k) => k.legendgroup).filter(Boolean)).size;
    const box = legendBox(keys.map((k) => k.name), groups, env.width, env.height);
    const spot = lay.xaxis.range ? PX.legendSpot(curves, lay.xaxis.range, yr, box.fw, box.fh) : "outside";
    if (spot === "outside") lay.legend = legendSide({ traceorder: "grouped", tracegroupgap: 8 });
    else lay.legend = { orientation: "v", x: 0.99, xanchor: "right", y: spot === "top" ? 0.99 : smooth ? 0.08 : 0.01, yanchor: spot === "top" ? "top" : "bottom",
      font: { size: 11 }, traceorder: "grouped", tracegroupgap: 8, bgcolor: PX.rgba(css("--plot-bg") || "#ffffff", 0.88), bordercolor: grid, borderwidth: 1 };
  }
  lay.shapes = shapes;
  if (annotations.length) lay.annotations = annotations;
  return { traces, layout: lay, legendBase: { t: lay.margin.t, r: lay.margin.r }, absShown, dosMode };
}
