/* Plot documents (spec §1.2, plan B5): a band structure, a DOS, both on one
 * energy axis, an IR or a Raman spectrum, drawn by plotly from the package
 * (loaded on the first plot only, §8.5) from the arrays mace plotting's own
 * readers return. Each document has its Plot options in the inspector, with
 * their own undo (Mod+Z on the plot) and Reset; the Fermi line can be dragged
 * to shift the energy reference; hovering a band reads out k, E − E_F and the
 * band. Export: PNG/SVG from plotly, the publication figure from mace
 * plotting itself (a task), and the plotted data as CSV. */
import * as PK from "../plotkit.js";
import * as K from "../kpath.js";
import * as KE from "../kpath_editor.js";
import { createBZ } from "../bz_view.js";
import * as BF from "./bandfig.js";
import { cliCommand } from "./clicmd.js";
import { robustRange } from "../maprange.js";

const MERGE_MS = 800;

export function createPlotView(host, optsHost, H) {
  const { api, q, esc, icon, toast, loadPlotly, apiTask, download } = H;
  // a token as the plots draw it: the theme's, or the forced plot surface's (Settings › Appearance › Plot background)
  // ... in the plot family's preset (Settings › Appearance › Plot style): band & DOS, spectra, maps, SCF & run charts
  const familyOf = (d) => (!d ? "band" : d.kind === "map" ? "maps" : d.kind === "convergence" ? "scf" : d.kind === "ir" || d.kind === "raman" ? "spectra" : "band");
  // a plot's own Plot style (Plot options › Style) over its family's (Settings › Appearance › Plot style)
  const ownPreset = (fam) => { const s = doc && doc.opts && doc.opts.style, pp = H.presets ? H.presets() : null; return s && pp ? pp.resolveId(s, fam) : null; };
  const presetOf = (fam) => ownPreset(fam) || (H.plotPreset ? H.plotPreset(fam) : "default");
  const tokenOf = (fam) => (n) => { const pid = ownPreset(fam); return pid && H.plotVarIn ? H.plotVarIn(n, pid) : H.plotVar ? H.plotVar(n, fam) : H.cssVar(n); };
  const cssVar = (n) => tokenOf(familyOf(doc))(n), paper = () => (H.plotPaper ? H.plotPaper() : { paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)" });
  host.innerHTML = `<div class="pv" tabindex="-1">
    <div class="pv-head"><div class="pv-title"></div><div class="pv-chips"></div><div class="pv-acts"></div></div>
    <div class="pv-note muted small"></div>
    <div class="pv-plot" tabindex="0" aria-label="Plot"></div>
    <div class="pv-readout mono" aria-live="off"></div>
    <img class="pv-img" alt="" hidden>
    <iframe class="pv-frame" sandbox="allow-scripts" title="Plot" hidden></iframe></div>`;
  const $ = (s) => host.querySelector(s);
  /* the title is cut to one line when the pane is narrow: its tooltip has the whole of it */
  const setTitle = (t) => { const e = $(".pv-title"); e.textContent = t; e.title = t; };
  const el = $(".pv-plot");
  let doc = null, building = false, kit = null;

  /* plotly's "responsive" follows window resizes only. A pane that changed size while the plot was
     hidden (another tab showing, the inspector or bottom panel toggled, a splitter dragged) refits
     when it shows again, at most once per frame. */
  let fitQueued = false;
  function fit() {
    const fl = el._fullLayout;
    if (!window.Plotly || !fl || building || !el.clientWidth || !el.clientHeight) return;
    if (Math.abs(fl.width - el.clientWidth) > 4 || Math.abs(fl.height - el.clientHeight) > 4) window.Plotly.Plots.resize(el);
  }
  if (typeof ResizeObserver === "function") {
    new ResizeObserver(() => {
      if (fitQueued) return;
      fitQueued = true;
      requestAnimationFrame(() => { fitQueued = false; fit(); });
    }).observe(el);
  }

  /* ------------------------------------------------------------ options and their history */
  const DEFAULTS = {
    bands: { emin: -6, emax: 6, shift: 0, spin: "both", width: 1.5, hidden: null, kpath: null,
      // the plot's own style (Plot options › Style; null: Settings), by the presets.py STYLE_OPTIONS names
      style: null, spin_colors: null, spin_up_color: null, spin_down_color: null, band_color: null, spin_down_style: null, dos_colors: null, dos_spin: null, dos_total: null, dos_smooth: null },
    spectrum: { xmin: null, xmax: null, normalize: false, component: "total", average: false, width: 1.5, style: null },
    // BOLTZTRA transport data (plan E8): the temperature block and the tensor component shown
    // view "zt": the figure of merit ZTₑ and, with a lattice κ_L (kl: W/(m·K) or a T,kappa_L CSV in the
    // folder) the total ZT, at the run's relaxation time or tau (fs): the numbers of mace analyze --kappa-lattice
    transport: { T: null, component: "avg", width: 1.5, view: "quantity", kl: "", tau: null, style: null },
    // Compare Convergence (plan E12 fallback): energy per atom above the lowest final energy of the set
    convergence: { width: 1.6, log: false, style: null },
    // a fort.25 2D map (plan E6): which map (density, spin), log scale, contour count, contour style (null: the
    // plot style's), color range (robust: the 1st to 99th percentile, symmetric about 0 for a signed map, so a
    // nuclear spike does not take the whole scale; full: min to max; custom: zmin to zmax)
    map: { which: 0, log: true, contours: 20, contour_style: null, crange: "robust", zmin: null, zmax: null, style: null },
    // a spectrum broadened from a FREQ output's modes (plan E4): width and line shape; fwhm null = the run's DAMPFAC or 8
    modes: { xmin: null, xmax: null, normalize: false, component: "total", average: false, width: 1.5, fwhm: null, shape: "lorentzian", eta: 1, style: null },
  };
  // the shared options (plotkit): log axes, units, ranges; new documents take the Settings defaults
  const setting = (k, dflt) => { try { const v = H.setting ? H.setting(k) : undefined; return v === undefined || v === null || v === "" ? dflt : v; } catch (e) { return dflt; } };
  const isSpec = (d) => d.kind === "ir" || d.kind === "raman";
  // the plots of plan E (a fort.25 map, Compare Convergence, BOLTZTRA data) keep their own options only
  const isOwn = (d) => d.kind === "map" || d.kind === "convergence" || d.kind === "transport";
  const defaultsFor = (d) => {
    if (isOwn(d)) return { ...(d.kind === "map" ? DEFAULTS.map : d.kind === "convergence" ? DEFAULTS.convergence : DEFAULTS.transport) };
    const own = d.source === "modes" ? { ...DEFAULTS.modes, shape: H.lineShape ? H.lineShape(d.kind) : "lorentzian" } : isSpec(d) ? DEFAULTS.spectrum : DEFAULTS.bands;
    // band/DOS: Settings' energy window and Fermi shift (a folder's mace-folder.toml [plots] may set them)
    const win = own === DEFAULTS.bands ? setting("plots.energyWindow", null) : null;
    const band = Array.isArray(win) && win.length === 2 ? { emin: Number(win[0]), emax: Number(win[1]), shift: Number(setting("plots.fermiShift", 0)) || 0 } : {};
    return { ...own, ...band, ylog: false, xlog: false, xr: null, yr: null,
      unit: isSpec(d) ? "cm-1" : setting("energyUnit", "eV"), width: Number(setting("plots.lineWidth", 1.5)) || 1.5 };   // Settings › Plots › Line width (1.5)
  };
  const key = (d) => `mace.gui.plot.${H.dirHash()}.${d.path}${d.source ? `|${d.source}|${d.kind}` : ""}`;
  function load(d) {
    if (d.opts) return;
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(key(d)) || "null"); } catch (e) { saved = null; }
    d.opts = { ...defaultsFor(d), ...((saved && saved.opts) || {}) };
    // the 1.1.6 preview's map option "style" was the contour style (now contour_style; "style" is the Plot style)
    if (d.kind === "map" && ["filled", "lines", "both"].includes(d.opts.style)) { d.opts.contour_style = d.opts.style; d.opts.style = null; }
    if ("dosColors" in d.opts) { if (d.opts.dosColors && !d.opts.dos_colors) d.opts.dos_colors = d.opts.dosColors; delete d.opts.dosColors; }   // the 1.1.6 preview's name
    d.hist = (saved && saved.hist) || [{ label: "Opened", opts: { ...d.opts }, t: Date.now() }];
    d.pos = saved && Number.isInteger(saved.pos) ? saved.pos : d.hist.length - 1;
  }
  function persist(d) {
    try { localStorage.setItem(key(d), JSON.stringify({ opts: d.opts, hist: d.hist.slice(-60), pos: Math.min(d.pos, 59) })); } catch (e) { /* private */ }
  }
  function setOpt(d, patch, label, mergeKey = null) {
    Object.assign(d.opts, patch);
    d.hist = d.hist.slice(0, d.pos + 1);
    const last = d.hist[d.hist.length - 1];
    if (mergeKey && last && last.mergeKey === mergeKey && Date.now() - last.t < MERGE_MS) d.hist[d.hist.length - 1] = { label, opts: { ...d.opts }, t: Date.now(), mergeKey };
    else d.hist.push({ label, opts: { ...d.opts }, t: Date.now(), mergeKey });
    d.pos = d.hist.length - 1;
    persist(d);
    redraw();
    renderOptions();
    if (kit) kit.refresh();
  }
  function undo(d) { if (d.pos <= 0) return null; d.pos -= 1; d.opts = { ...d.hist[d.pos].opts }; persist(d); redraw(); renderOptions(); return d.hist[d.pos + 1].label; }
  function redo(d) { if (d.pos >= d.hist.length - 1) return null; d.pos += 1; d.opts = { ...d.hist[d.pos].opts }; persist(d); redraw(); renderOptions(); return d.hist[d.pos].label; }
  function reset(d) { setOpt(d, defaultsFor(d), "Reset plot options"); }

  /* ------------------------------------------------------------ data */
  async function fetchData(d) {
    if (d.data) return d.data;
    if (d.kind === "transport") {
      d.data = { transport: await api("transport?" + q({ path: d.path })) };
      return d.data;
    }
    if (d.kind === "map") {
      d.data = { map: await api("map?" + q({ path: d.path })) };
      return d.data;
    }
    if (d.kind === "convergence") {
      d.data = { convergence: await api("convergence?" + d.paths.map((p) => "path=" + encodeURIComponent(p)).join("&")) };
      return d.data;
    }
    if (d.source === "modes") {
      const o = d.opts || {};
      const s0 = await api("spectrum?" + q({ path: d.path, source: "modes", kind: d.kind, fwhm: o.fwhm ?? "", shape: o.shape || "lorentzian", eta: o.eta ?? 1 }));
      d.data = { spectra: [s0], peaks: s0.peaks || [], out: s0.out };
      return d.data;
    }
    if (d.kind === "ir" || d.kind === "raman") {
      const s0 = await api("spectrum?" + q({ path: d.path }));
      d.data = { spectra: [s0], peaks: s0.peaks || [], out: s0.out || null };
      if (s0.group && s0.group.length > 1) d.data.group = s0.group;
      if (d.opts && d.opts.average && d.data.group) d.data.average = await api("spectrum?" + q({ path: d.path, average: 1 })).catch(() => null);
      return d.data;
    }
    const first = await api((d.kind === "dos" ? "dos" : "bands") + "?" + q({ path: d.path }));
    let band = d.kind === "dos" ? null : first, dos = d.kind === "dos" ? first : null;
    if (first.partner) {
      try {
        const other = await api((d.kind === "dos" ? "bands" : "dos") + "?" + q({ path: first.partner }));
        if (d.kind === "dos") band = other; else dos = other;
      } catch (e) { /* the partner is optional */ }
    }
    d.data = { band, dos };
    return d.data;
  }

  /* ------------------------------------------------------------ figures */
  const PAL = (css = cssVar) => BF.PAL(css);
  // what the energy axis reads: E - E_F, or E - E_vac with a POTC run (the API's reference)
  // the readout's energy: E − E_F, or with a Fermi shift E − (E_F + s) as the axis title (bandfig.js energyTitle)
  const eLabel = (d) => BF.energyTitle(((d.data && (d.data.band || d.data.dos)) || {}).reference, (d.opts && d.opts.shift) || 0)
    .replace(/<sub>(.*?)<\/sub>/g, "_$1");
  const kLabel = BF.kLabel;

  /* the user's defaults for a plot's style (Settings › Appearance › Plot style, Settings › Plots), by the
     presets.py STYLE_OPTIONS names; a plot's own choices (its options of the same names) go over them */
  function settingsLook() {
    return { style: H.plotPreset ? H.plotPreset("band") : "default", spin_colors: setting("plots.spinColors", "auto"),
      spin_up_color: setting("plots.spinUp", ""), spin_down_color: setting("plots.spinDown", ""),
      band_color: setting("plots.bandColor", ""),
      spin_down_style: setting("plots.spinDownDash", "dash"), dos_colors: setting("plots.dosColors", "auto"),
      dos_spin: setting("plots.spinDos", "auto"), dos_total: setting("plots.dosTotal", true) !== false,
      dos_smooth: Number(setting("plots.dosSmooth", 0)) || 0 };
  }
  /* the figure's page-side inputs (bandfig.js env): the plot's look, its preset's tokens, the pane's width */
  function envOf(o, fermiKey = false) {
    const PP = H.presets ? H.presets() : null;
    const look = PP ? BF.lookOf(o, settingsLook(), PP) : settingsLook();
    const pid = PP ? PP.resolveId(look.style, "band") : "default";
    const css = o && o.style && H.plotVarIn ? (n) => H.plotVarIn(n, pid) : tokenOf("band");
    return { css, look, tone: H.plotTone ? H.plotTone() : "light", paper: paper(), presets: PP,
      elementColors: H.elementColors ? H.elementColors() : {},
      width: el.clientWidth || 900, height: el.clientHeight || 420, marginR: el._fullLayout && el._fullLayout.margin ? el._fullLayout.margin.r : 14, fermiKey };
  }
  // How a DOS is drawn (bandfig.js dosLook): the plot's look, its DOS colors
  const dosLook = (dos, css, o = {}) => BF.dosLook(dos, envOf(o), o);
  const dosRange = BF.dosRange;
  function bandTickLayout(band, range, dos, r = el._fullLayout && el._fullLayout.margin ? el._fullLayout.margin.r : 14) {
    return BF.bandTickLayout(band, range, dos, el.clientWidth || 900, r);
  }
  /* ------------------------------------------------------------ the k-path as edited (kpath_editor.js)
     The computed path comes from the tick labels ("A|L" is a jump); each segment is the data
     between its two ticks. An edited path (opts.kpath) lays those slices out again in its order,
     reversed where it says so, with a gap at each jump; hidden segments are left out. */
  const pathSegments = BF.pathSegments;
  function bandView(d) {
    const band = d.data && d.data.band, edit = d.opts && d.opts.kpath;
    if (!band || !edit) return band;
    const sig = JSON.stringify(edit);
    if (d.view && d.view.sig === sig && d.view.band === band) return d.view.out;
    const out = BF.editedBand(band, edit);
    d.view = { sig, band, out };
    return out;
  }
  /** Where each segment of the editor's list is drawn: {index: [x0, x1]} (hidden ones are not). */
  function segRanges(d) {
    const v = bandView(d);
    if (v && v.segX) return v.segX;
    const out = {};
    if (v) v.ticks.slice(0, -1).forEach((t, n) => { out[n] = [t, v.ticks[n + 1]]; });
    return out;
  }
  /* what an editor control would do, on the plot before it is used: the segments a point's ×
     hides (or the one under the pointer, in the strip or the Brillouin zone view) shaded; the part
     Continuous only keeps, with the rest grayed */
  function plotPreview(p) {
    if (!doc || !el.layout || !el.layout.shapes) return;
    const base = el.layout.shapes.filter((x) => !String(x.name || "").startsWith("kp-"));
    const add = [];
    if (p) {
      const R = segRanges(doc), set = new Set(p.indices);
      const rect = (x0, x1, fill, op) => ({ type: "rect", xref: "x", yref: "y domain", x0, x1, y0: 0, y1: 1, layer: "below", line: { width: 0 },
        fillcolor: fill, opacity: op, name: `kp-${p.kind}` });
      for (const [i, r] of Object.entries(R)) {
        const on = set.has(+i);
        if (p.kind === "keep" ? !on : on) add.push(rect(r[0], r[1], p.kind === "keep" ? cssVar("--text-3") : cssVar(p.kind === "affected" ? "--warning" : "--accent"), p.kind === "hover" ? 0.12 : 0.2));
      }
    }
    window.Plotly.relayout(el, { shapes: [...base, ...add] });
  }
  function bandFigure(d) {
    const fig = BF.bandFigure(bandView(d), d.data.dos, d.opts, envOf(d.opts));
    d.absShown = fig.absShown;
    if (fig.dosMode) d.dosMode = fig.dosMode;
    return { traces: fig.traces, layout: fig.layout, legendBase: fig.legendBase };
  }
  /* The band and DOS legends sit in the top and right margins (bandFigure). Plotly's own margin
     push leaves a legend where the previous size put it when a resize changes its height (it does
     not redraw while replotting), so the margins are set here from the legends as drawn: a legend
     never covers the data or the tick labels at any pane size or UI scale, and the exports, drawn
     from this layout, are the same. */
  let legendBase = null;
  function fitLegends() {
    if (!legendBase || building) return Promise.resolve();
    // the k labels' rows depend on the band panel's width, which the right margin changes
    return PK.fitLegends(el, legendBase, (r) => (doc && doc.data && doc.data.band && el.layout.xaxis
      ? bandTickLayout(doc.data.band, el.layout.xaxis.range, doc.data.dos, r) : {}));
  }
  function irLabel(shown, fallback) {
    const r = [...shown].filter((c) => c.startsWith("reflectance")).length;
    return r === 0 ? fallback || "Absorbance" : r === shown.size ? "Reflectance" : "Absorbance / reflectance";
  }
  function spectrumFigure(d) {
    const o = d.opts, css = tokenOf("spectra"), pal = PAL(css), grid = css("--line");
    const traces = [];
    const list = o.average && d.data.average ? [d.data.average] : d.data.spectra;
    const fx = PK.WAVENUMBER_UNITS[o.unit] || 1, ul = PK.UNIT_LABEL[o.unit] || "cm⁻¹";
    d.absShown = false;
    const shown = new Set();
    list.forEach((s, k) => {
      // Raman's components, or a crystal's IR absorbances and reflectances (IRSPEC.DAT columns 3-9)
      const keys = Object.keys(s.columns);
      const comps = o.component === "all" && (d.kind === "raman" || keys.length > 1) ? keys : [s.columns[o.component] ? o.component : keys[0]];
      comps.forEach((c) => shown.add(c));
      comps.forEach((c, j) => {
        let y = s.columns[c] || s.columns[Object.keys(s.columns)[0]];
        if (o.normalize) { const m = Math.max(...y.map(Math.abs)) || 1; y = y.map((v) => v / m); }
        if (o.ylog) { const a = PK.absIfNeeded(y); y = a.ys; if (a.abs) d.absShown = true; }
        traces.push({ x: s.x.map((v) => v * fx), y, mode: "lines", type: "scatter", name: list.length > 1 || (o.average && d.data.average) ? `${s.name} · ${c}` : c,
          line: { color: pal[(k + j) % pal.length], width: o.width }, hovertemplate: `%{x:.2f} ${ul}: %{y:.4g}<extra>${esc(c)}</extra>` });
      });
    });
    const xs = d.data.spectra[0].x.map((v) => v * fx);
    // the legend in a column right of the plot, the peaks' wavenumbers on a top axis (fitLegends sizes both margins)
    const lay = { margin: { l: 64, r: 14, t: 10, b: 44, autoexpand: false }, showlegend: traces.length > 1 || !!(o.average && d.data.average), hovermode: "x unified",
      legend: PK.legendSide(), ...paper(),
      font: { color: css("--text"), size: 12, family: css("--font-sans") },
      modebar: { bgcolor: "rgba(0,0,0,0)", color: css("--text-2"), activecolor: css("--accent") },
      xaxis: { title: { text: `${o.unit === "cm-1" || !o.unit ? "Wavenumber" : "Energy"} (${ul})` },
        type: o.xlog ? "log" : "linear",
        range: o.xr || (o.xlog ? undefined : [(o.xmin ?? Math.min(...d.data.spectra[0].x)) * fx, (o.xmax ?? Math.max(...d.data.spectra[0].x)) * fx]),
        autorange: o.xlog && !o.xr ? true : undefined, gridcolor: grid, showline: true, mirror: true, linecolor: grid, zeroline: false },
      yaxis: { title: { text: o.normalize ? "Normalized" : d.kind === "ir" ? irLabel(shown, d.data.spectra[0].ylabel) : "Raman activity" },
        type: o.ylog ? "log" : "linear", range: o.yr || undefined,
        gridcolor: grid, showline: true, mirror: true, linecolor: grid, zeroline: false } };
    if (d.data.peaks && d.data.peaks.length) {
      // the strongest peaks' positions as labels on a top axis, above the data; a weaker peak too close to
      // a stronger one's label (13 px at the plot's width) is left unlabeled
      const xr = lay.xaxis.range || [Math.min(...xs), Math.max(...xs)], span = Math.abs(xr[1] - xr[0]) || 1;
      const gap = span * 13 / Math.max(200, (el.clientWidth || 900) - 64 - 14 - 110);
      const strongest = [];
      for (const p of [...d.data.peaks].sort((a, b) => b.intens - a.intens)) {
        if (strongest.length >= 12) break;
        if (strongest.every((q) => Math.abs(q.freq - p.freq) * fx >= gap)) strongest.push(p);
      }
      lay.xaxis2 = { overlaying: "x", matches: "x", side: "top", type: lay.xaxis.type, showgrid: false, zeroline: false, showline: false, ticks: "",
        tickmode: "array", tickvals: strongest.map((p) => p.freq * fx), ticktext: strongest.map((p) => `${(p.freq * fx).toFixed(o.unit === "cm-1" || !o.unit ? 0 : 2)}`),
        tickangle: -90, tickfont: { size: 9, color: css("--text-3") } };
      traces.push({ x: [null], y: [null], xaxis: "x2", mode: "markers", type: "scatter", meta: "peak-axis", showlegend: false, hoverinfo: "skip" });
      // one marker per active mode, on the curve: hover names the mode, a click opens its animation (plan E4)
      if (traces.length && d.data.out) {
        const t0 = traces[0], at = (f) => {
          let lo = 0, hi = t0.x.length - 1;
          while (hi - lo > 1) { const m = (lo + hi) >> 1; if (t0.x[m] < f) lo = m; else hi = m; }
          const w = (f - t0.x[lo]) / ((t0.x[hi] - t0.x[lo]) || 1);
          return t0.y[lo] + w * (t0.y[hi] - t0.y[lo]);
        };
        const unit = d.kind === "ir" ? " km/mol" : "";
        traces.push({ x: d.data.peaks.map((p) => p.freq * fx), y: d.data.peaks.map((p) => at(p.freq * fx)), mode: "markers", type: "scatter",
          name: "modes", meta: "peaks", showlegend: false, customdata: d.data.peaks.map((p) => p.mode),
          marker: { size: 7, color: css("--accent"), line: { width: 1, color: css("--surface") } },
          text: d.data.peaks.map((p) => `Mode ${p.mode} · ${p.freq.toFixed(1)} cm⁻¹ · ${p.irrep || ""} · ${d.kind === "ir" ? "IR" : "Raman"} ${p.intens.toFixed(1)}${unit}${p.degeneracy > 1 ? ` (degenerate ×${p.degeneracy})` : ""}`),
          hovertemplate: "%{text}<extra>click: animate</extra>" });
        lay.hovermode = "closest";
      }
    }
    if (d.source === "modes") lay.yaxis.title.text = o.normalize ? "Normalized" : d.data.spectra[0].ylabel;
    return { traces, layout: lay, legendBase: { t: 10, r: 14 } };
  }

  /* a 2D map: contours in the plane of A, B, C (Å), the corners labeled */
  function mapFigure(d) {
    const m = d.data.map.maps[Math.min(d.opts.which || 0, d.data.map.maps.length - 1)], o = d.opts, css = tokenOf("maps");
    const positive = m.min > 0, log = o.log && positive;
    const z = m.values.map((col) => col.map((v) => (log ? Math.log10(v) : v)));
    const signedZ = !log && m.min < 0 && m.max > 0;
    const zr = o.crange === "custom" && Number.isFinite(o.zmin) && Number.isFinite(o.zmax) && o.zmax > o.zmin ? [o.zmin, o.zmax]
      : robustRange(z, { signed: signedZ, full: o.crange === "full" });
    const dist = (p, r) => Math.hypot(p[0] - r[0], p[1] - r[1], p[2] - r[2]);
    const ab = dist(m.A, m.B), bc = dist(m.B, m.C), sin = Math.sqrt(Math.max(0, 1 - m.cosxy * m.cosxy));
    // each corner's label sits inside the plane, on the side of its corner away from the edges and tick labels
    const cx = (ab + ab + bc * m.cosxy) / 3, cy = bc * sin / 3;
    const corner = (t, x, y) => ({ x, y, text: t, showarrow: false, font: { size: 13, color: css("--text") },
      xanchor: x < cx - 1e-6 ? "left" : x > cx + 1e-6 ? "right" : "center", yanchor: y < cy ? "bottom" : "top",
      bgcolor: css("--surface"), opacity: 0.85 });
    const signed = m.min < 0 && m.max > 0;
    // the Maps & volumes preset's colormap (signed: + blue, − red; VESTA-like: + yellow, − cyan) and contour
    // style (filled, lines or both: this map's own choice over the preset's)
    const pp = H.presets ? H.presets() : null, pid = presetOf("maps");
    const sc = pp ? pp.mapScale(pid, signed) : { colorscale: signed ? "RdBu" : "Viridis", reversescale: signed, contours: "both" };
    // a POTM/CLAS potential in physics colors (Settings › Viewport › Potential colors): red = +; chemistry keeps + blue, − red
    if (signed && m.potential && setting("volume.mepColors", "chemistry") === "physics") sc.reversescale = !sc.reversescale;
    const cstyle = o.contour_style || sc.contours, darkLines = pid === "vesta" || pid === "grayscale" || pid === "publication";
    const nc = Math.max(2, o.contours || 20), step = (zr[1] - zr[0]) / nc || 1;
    const trace = { type: "contour", x: m.x, y: m.y, z, colorscale: sc.colorscale, reversescale: sc.reversescale,
      zauto: false, zmin: zr[0], zmax: zr[1], autocontour: false,
      contours: { coloring: cstyle === "lines" ? "lines" : "heatmap", showlines: cstyle !== "filled", start: zr[0], end: zr[1], size: step },
      line: { width: cstyle === "lines" ? 1.2 : 0.6, ...(darkLines && cstyle === "both" ? { color: "rgba(0,0,0,0.55)" } : {}) },
      colorbar: { title: { text: log ? `log₁₀ (${m.unit || ""})` : m.unit || "" }, thickness: 12 },
      hovertemplate: `%{x:.3f}, %{y:.3f} Å: ${log ? "10^%{z:.3f}" : "%{z:.4g}"}<extra>${esc(m.label)}</extra>` };
    return { traces: [trace], layout: { margin: { l: 56, r: 14, t: 10, b: 44 }, ...paper(),
      font: { color: css("--text"), size: 12, family: css("--font-sans") },
      xaxis: { title: { text: "along AB (Å)" }, scaleanchor: "y", constrain: "domain", showgrid: false },
      yaxis: { title: { text: "toward C (Å)" }, showgrid: false },
      annotations: [corner("A", 0, 0), corner("B", ab, 0), corner("C", ab + bc * m.cosxy, bc * sin)] } };
  }

  /* Compare Convergence: each run's SCF energy per atom at each geometry, above the lowest final energy of the set */
  const HARTREE_MEV = 27211.386245988;
  function convergenceFigure(d) {
    const runs = d.data.convergence.runs, o = d.opts, css = tokenOf("scf"), pal = PAL(css), grid = css("--line");
    const finals = runs.map((r) => { const e = r.energies.filter((x) => x !== null); return e.length ? e[e.length - 1] / r.n_atoms : null; })
      .filter((x) => x !== null);
    const ref = finals.length ? Math.min(...finals) : 0;
    const traces = runs.map((r, k) => {
      const pts = r.energies.map((e, i) => [i + 1, e === null ? null : (e / r.n_atoms - ref) * HARTREE_MEV]).filter(([, y]) => y !== null);
      // "* OPT END - CONVERGED * E(AU): -7.62E+01 POINTS 3 *" -> "OPT END - CONVERGED, 3 points" (the energy is plotted)
      const stop = r.status ? r.status.replace(/\*/g, "").replace(/E\(AU\):\s*\S+/, "").replace(/POINTS\s+(\d+)/, ", $1 points")
        .replace(/\s+,/, ",").replace(/\s+/g, " ").trim() : "no OPT END line";
      return { x: pts.map((p) => p[0]), y: pts.map((p) => (o.log ? Math.max(p[1], 1e-6) : p[1])), mode: "lines+markers", type: "scatter",
        name: `${r.name} · ${stop}`, line: { color: pal[k % pal.length], width: o.width }, marker: { size: 5 },
        hovertemplate: `%{y:.3f} meV/atom at geometry %{x}<extra>${esc(r.name)}</extra>` };
    });
    return { legendBase: { t: 10, r: 14 }, traces, layout: { margin: { l: 70, r: 14, t: 10, b: 50, autoexpand: false }, showlegend: true, legend: PK.legendTop(),
      hovermode: "closest", ...paper(), font: { color: css("--text"), size: 12, family: css("--font-sans") },
      xaxis: { title: { text: "Geometry (initial, optimization points, final)" }, gridcolor: grid, showline: true, mirror: true, linecolor: grid, dtick: 1 },
      yaxis: { title: { text: "ΔE per atom (meV)" }, type: o.log ? "log" : "linear", gridcolor: grid, showline: true, mirror: true, linecolor: grid } } };
  }

  /* BOLTZTRA data: one quantity per file against the chemical potential, one temperature at a time */
  function transportFigure(d) {
    const t = d.data.transport, o = d.opts, css = tokenOf("band"), pal = PAL(css), grid = css("--line");
    const block = t.blocks.find((b) => b.T === o.T) || t.blocks[0];
    const comps = o.component === "all" ? ["xx", "yy", "zz"].filter((k) => block[k]) : [block[o.component] ? o.component : "avg"];
    const traces = comps.map((c, k) => ({ x: block.mu, y: block[c], mode: "lines", type: "scatter",
      name: c === "avg" ? "(xx + yy + zz) / 3" : c, line: { color: pal[k % pal.length], width: o.width },
      hovertemplate: `%{x:.3f} eV: %{y:.4g} ${esc(t.unit)}<extra>${esc(c)}</extra>` }));
    const lay = { margin: { l: 70, r: 14, t: 10, b: 44, autoexpand: false }, showlegend: traces.length > 1, hovermode: "x unified", legend: PK.legendTop(),
      ...paper(), font: { color: css("--text"), size: 12, family: css("--font-sans") },
      xaxis: { title: { text: t.x === "energy" ? "Energy (eV)" : "Chemical potential μ (eV)" }, gridcolor: grid, showline: true, mirror: true, linecolor: grid, zeroline: true, zerolinecolor: grid },
      yaxis: { title: { text: `${t.label} (${t.unit})` }, gridcolor: grid, showline: true, mirror: true, linecolor: grid, zeroline: true, zerolinecolor: grid } };
    return { traces, layout: lay, legendBase: { t: 10, r: 14 } };
  }

  /* ZT of a transport run (view "zt"): read again when κ_L or τ changed */
  const ztKey = (o) => JSON.stringify([String(o.kl || "").trim(), o.tau ?? null]);
  const isZt = (d) => d && d.kind === "transport" && d.opts && d.opts.view === "zt";
  const staleZt = (d) => isZt(d) && d.data && d.data.transport && (!d.data.zt || d.data.ztKey !== ztKey(d.opts));
  async function fetchZt(d) {
    const o = d.opts, key = ztKey(o), kl = String(o.kl || "").trim();
    d.data.ztError = null;
    try {
      d.data.zt = await api("transport-zt?" + q({ path: d.path, kappa_l: kl, tau_fs: o.tau ?? "" }));
    } catch (e) {
      // a κ_L or τ MACE refuses: say why, and show ZTₑ alone
      d.data.ztError = e.message;
      try { d.data.zt = await api("transport-zt?" + q({ path: d.path })); } catch (e2) { d.data.zt = null; d.data.ztError = e2.message; }
    }
    d.data.ztKey = key;
  }
  const fmtZt = (m) => (m ? `${m.value.toPrecision(4)} (${m.temperature_k} K, μ ${m.mu_ev.toFixed(3)} eV)` : "");

  /* the figure of merit against μ at one temperature: ZTₑ (electrons only) and the total with κ_L */
  function ztFigure(d) {
    const z = d.data.zt, o = d.opts, css = tokenOf("band"), pal = PAL(css), grid = css("--line");
    const traces = [];
    if (z) {
      const block = z.blocks.find((b) => b.T === o.T) || z.blocks[0];
      traces.push({ x: block.mu, y: block.zt_e, mode: "lines", type: "scatter", meta: "zt_e", connectgaps: false,
        name: "ZTₑ (electrons only)", line: { color: pal[0], width: o.width, dash: "dash" },
        hovertemplate: "%{x:.3f} eV: ZTₑ %{y:.4g}<extra></extra>" });
      if (block.zt_total) traces.push({ x: block.mu, y: block.zt_total, mode: "lines", type: "scatter", meta: "zt_total", connectgaps: false,
        name: `ZT total, ${z.kappa_l_label}${z.kappa_l_label.includes("from") ? ` (${Number(block.kappa_l).toPrecision(4)} at ${block.T} K)` : ""}, τ ${z.tau_fs} fs`,
        line: { color: pal[1], width: o.width },
        hovertemplate: "%{x:.3f} eV: ZT %{y:.4g}<extra></extra>" });
    }
    const lay = { margin: { l: 70, r: 14, t: 10, b: 44, autoexpand: false }, showlegend: true, hovermode: "x unified", legend: PK.legendTop(),
      ...paper(), font: { color: css("--text"), size: 12, family: css("--font-sans") },
      xaxis: { title: { text: "Chemical potential μ (eV)" }, gridcolor: grid, showline: true, mirror: true, linecolor: grid, zeroline: true, zerolinecolor: grid },
      yaxis: { title: { text: "Figure of merit ZT" }, rangemode: "tozero", gridcolor: grid, showline: true, mirror: true, linecolor: grid, zeroline: true, zerolinecolor: grid } };
    return { traces, layout: lay, legendBase: { t: 10, r: 14 } };
  }

  let again = false;
  /* a broadened spectrum whose width or shape changed (undo, redo, reset): read again */
  const staleModes = (d) => d.source === "modes" && d.data && d.data.spectra && (
    (d.opts.fwhm ?? d.data.spectra[0].default_fwhm) !== d.data.spectra[0].fwhm || (d.opts.shape || "lorentzian") !== d.data.spectra[0].shape ||
    (d.kind === "raman" && (d.opts.eta ?? 1) !== d.data.spectra[0].eta));
  async function redraw() {
    if (!doc || !doc.data) return;
    if (staleModes(doc)) { const d = doc; d.data = null; try { await fetchData(d); } catch (e) { toast(e.message, "error"); return; } if (doc !== d) return; note(d); }
    if (staleZt(doc)) { const d = doc; await fetchZt(d); if (doc !== d) return; renderOptions(); }
    if (building) { again = true; return; }        // one figure at a time: the latest options win
    building = true;
    el.dataset.drawn = "";
    try {
      await loadPlotly();
      if (!window.Plotly) throw new Error("plotly is not installed in this Python; pip install plotly");
      const fig = doc.kind === "map" ? mapFigure(doc) : doc.kind === "convergence" ? convergenceFigure(doc) : doc.kind === "transport" ? (isZt(doc) ? ztFigure(doc) : transportFigure(doc)) : doc.kind === "ir" || doc.kind === "raman" ? spectrumFigure(doc) : bandFigure(doc);
      // another document, or other subplots (a band plot after a band+DOS one): a new figure, not an update;
      // plotly's react cannot add a second x axis to an existing figure
      const sig = `${doc.path}${isZt(doc) ? "|zt" : ""}|${Object.keys(fig.layout).filter((k) => /^[xy]axis/.test(k)).sort().join(",")}`;
      const config = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"], edits: { shapePosition: true } };
      // Settings › Appearance › Plot style (app.js styleFigure): the screen and the PNG/SVG exports alike
      if (H.styleFigure) H.styleFigure(fig.traces, fig.layout, familyOf(doc), (doc.opts && doc.opts.style) || null);
      const bw = $(".pv-bword");
      if (bw && isBandDos(doc)) bw.textContent = BF.dashWord(envOf(doc.opts).look.spin_down_style || "dash");
      // the styled top margin is the base fitLegends adds the legends to (never the fitted one: that grows on every redraw)
      if (fig.legendBase && fig.layout.margin) fig.legendBase.t = Math.max(fig.legendBase.t, fig.layout.margin.t);
      // the same figure again (an option changed): keep the margins fitted to its legends (fitLegends)
      if (fig.legendBase && el.__sig === sig && el.layout && el.layout.margin) Object.assign(fig.layout.margin, { t: el.layout.margin.t, r: el.layout.margin.r });
      // no transitions: zoom and pan are instant (and so under reduced motion too)
      if (el.__sig !== sig) {
        if (el.__sig) window.Plotly.purge(el);
        el.__maceBound = false;
        await window.Plotly.newPlot(el, fig.traces, fig.layout, config);
        el.__sig = sig;
      } else {
        await window.Plotly.react(el, fig.traces, fig.layout, config);
      }
      building = false;
      fit();          // react keeps the size the figure had; the pane may have changed since
      legendBase = fig.legendBase || null;
      await fitLegends();
      bind();
      chips();
      el.dataset.drawn = doc.path;
      performance.mark("plot:drawn");
    } catch (e) { $(".pv-note").classList.remove("one"); $(".pv-note").textContent = e.message; } finally {
      building = false;
      if (again) { again = false; redraw(); }
    }
  }

  function chips() {
    const o = doc.opts, c = $(".pv-chips");
    const parts = [];
    if (doc.kind !== "ir" && doc.kind !== "raman" && o.shift) parts.push(`<span class="pv-chip" title="Fermi shift (drag the dotted line; Plot options)">E<sub>F</sub> ${o.shift > 0 ? "+" : "−"} ${Math.abs(o.shift).toFixed(2)} eV</span>`);
    if (doc.absShown) parts.push(`<span class="pv-chip" title="A log axis cannot show zero or negative values">|value| shown</span>`);
    c.innerHTML = parts.join("");
  }

  /* hover readout, the Fermi-line drag, the zoom that keeps the DOS axis fitted */
  function bind() {
    if (el.__maceBound) return;
    el.__maceBound = true;
    // the DOS spin key only explains the line styles: a click on it hides nothing
    // a projection's swatch: a click hides that projection (Plot options › Projections shows it again)
    el.on("plotly_legendclick", (ev) => {
      const t = ev && ev.data && ev.data[ev.curveNumber];
      if (t && t.meta === "key-proj" && Array.isArray(t.customdata) && doc && doc.data && doc.data.dos) {
        const lab = t.customdata[0], look = dosLook(doc.data.dos, tokenOf("band"), doc.opts);
        if (look.byLabel[lab] && !look.byLabel[lab].total) setOpt(doc, { hidden: [...new Set([...look.hidden, lab])] }, `Hide ${lab}`);
        return false;
      }
      return PK.keyClick(ev);
    });
    el.on("plotly_legenddoubleclick", PK.keyClick);
    // a resize (the window, a splitter, the UI scale) redraws the legends: fit the margins to them again
    el.on("plotly_afterplot", () => { if (legendBase && !el.__fitting && !building) requestAnimationFrame(() => fitLegends()); });
    el.on("plotly_hover", (ev) => {
      if (!doc || !doc.data || !ev.points || !ev.points.length) return;
      const p = ev.points[0], t = p.data;
      if (doc.kind === "ir" || doc.kind === "raman" || doc.kind === "transport" || doc.kind === "convergence" || doc.kind === "map") return;
      if (t.meta === "dos") {
        const e = p.y;
        const vals = el.data.filter((x) => x.meta === "dos").map((x) => {
          let best = 0, bd = Infinity;
          for (let i = 0; i < x.y.length; i++) { const dd = Math.abs(x.y[i] - e); if (dd < bd) { bd = dd; best = i; } }
          return `${x.name}: ${Number(x.x[best]).toFixed(3)}`;
        });
        $(".pv-readout").textContent = `${eLabel(doc)} = ${Number(e).toFixed(3)} ${PK.UNIT_LABEL[doc.opts.unit] || "eV"} · ${vals.join(" · ")}`;
        return;
      }
      const band = bandView(doc);
      const x = p.x, e = p.y, k = t.customdata ? t.customdata[p.pointIndex ?? p.pointNumber] : null;
      let kl = "";
      const at = band.ticks.findIndex((tk) => Math.abs(tk - x) < 1e-6);
      if (at >= 0) kl = kLabel(band.labels[at]);
      else {
        const i = band.ticks.findIndex((tk) => tk > x);
        if (i > 0) kl = `${kLabel(band.labels[i - 1])}→${kLabel(band.labels[i])} ${Math.round(((x - band.ticks[i - 1]) / (band.ticks[i] - band.ticks[i - 1])) * 100)}%`;
      }
      $(".pv-readout").textContent = `k: ${kl || "—"} · path ${Number(x).toFixed(4)} · ${eLabel(doc)} = ${Number(e).toFixed(3)} ${PK.UNIT_LABEL[doc.opts.unit] || "eV"}` +
        (k !== null && k >= 0 ? ` · band ${k + 1}${band.spin_polarized ? (t.meta === "down" ? " (β)" : " (α)") : ""}` : "");
    });
    el.on("plotly_unhover", () => { $(".pv-readout").textContent = ""; });
    // a click on a spectrum opens the mode of the nearest active peak (within half the line width)
    el.on("plotly_click", (ev) => {
      if (!doc || !(doc.kind === "ir" || doc.kind === "raman") || !doc.data || !doc.data.out || !ev.points || !ev.points.length) return;
      const pt = ev.points[0];
      // The marker under the click itself, from the click's own position: plotly throttles its
      // hover to one per 50 ms and a click re-uses the last one, so a click right after the pointer
      // moved came with a point it passed on the way (the strong peak's neighbor 17 px below it).
      let mode = null;
      const pk = (el.data || []).find((t) => t.meta === "peaks"), fl = el._fullLayout, e = ev.event;
      if (pk && fl && fl.xaxis && fl.yaxis && e && typeof e.clientX === "number") {
        // the click in the plot's own px: its screen offset is the UI scale's zoom x the plot's px
        const [px, py] = window.maceZoom.point(el, e.clientX, e.clientY), mx = px - fl._size.l, my = py - fl._size.t;
        let bi = -1, bd = 10;                      // px: a marker within reach of the pointer
        pk.x.forEach((x, i) => {
          const dd = Math.hypot(fl.xaxis.c2p(x) - mx, fl.yaxis.c2p(pk.y[i]) - my);
          if (dd <= bd) { bd = dd; bi = i; }
        });
        if (bi >= 0) mode = pk.customdata[bi];
      }
      if ((mode === null || mode === undefined) && pt.data && pt.data.meta === "peaks") mode = pt.customdata;
      if (mode === null || mode === undefined) {
        const half = Math.max(4, ((doc.data.spectra[0] && doc.data.spectra[0].fwhm) || 8) / 2);
        let best = null, bd = Infinity;
        const cm = pt.x / (PK.WAVENUMBER_UNITS[doc.opts.unit] || 1);        // the click in cm⁻¹
        for (const p of doc.data.peaks) { const dd = Math.abs(p.freq - cm); if (dd < bd) { bd = dd; best = p; } }
        if (!best || bd > half) return;
        mode = best.mode;
      }
      if (H.openModes) H.openModes(doc.data.out, { mode: Number(mode) });
    });
    el.on("plotly_relayout", (ev) => {
      if (!doc || !ev || !doc.data) return;        // a relayout of the previous figure while the next document loads
      const sy = Object.keys(ev).find((k) => /^shapes\[0\]\.y0$/.test(k));
      if (sy !== undefined && (doc.data.band || doc.data.dos)) {
        // the Fermi line was dragged (the band panel's, or a DOS-only plot's): the shift is where it went,
        // relative to the reader's E_F. The line is drawn at E_F (0, or the vacuum shift) in the plotted
        // unit: the energy now under it becomes the reference, s' = s + y - E_F (in eV)
        const ef = (doc.data.band || doc.data.dos).fermi || 0;
        const ns = Math.round(((doc.opts.shift || 0) + ev[sy] / (PK.ENERGY_UNITS[doc.opts.unit] || 1) - ef) * 1000) / 1000 || 0;
        setOpt(doc, { shift: ns }, `Fermi shift ${ns.toFixed(2)} eV`, "drag");
        return;
      }
      // a zoom along k: lay the tick labels out again for the new range (the rows depend on it)
      if (doc.data.band && el.layout.xaxis && Object.keys(ev).some((k) => k === "xaxis.autorange" || k.startsWith("xaxis.range"))) {
        const bv = bandView(doc), r = el.layout.xaxis.range, tk = bv.ticks;
        window.Plotly.relayout(el, bandTickLayout(bv, r || [tk[0], tk[tk.length - 1]], doc.data.dos));
      }
      if (doc.data.dos && Object.keys(ev).some((k) => k.startsWith("yaxis."))) {
        if (doc.opts.ylog || !el.layout.yaxis.range) return;
        // the DOS is drawn at E - s: the states on screen are those at the visible energy + s
        const fu = PK.ENERGY_UNITS[doc.opts.unit] || 1, s = doc.opts.shift || 0;
        const [lo, hi] = el.layout.yaxis.range.map((v) => s + v / fu);
        const ax = el.layout.xaxis2 ? "xaxis2" : "xaxis";
        window.Plotly.relayout(el, { [`${ax}.range`]: dosRange(doc.data.dos, Math.min(lo, hi), Math.max(lo, hi), doc.opts, dosLook(doc.data.dos, tokenOf("band"), doc.opts)) });
      }
    });
  }
  el.addEventListener("keydown", (e) => {
    if (!doc || !(e.ctrlKey || e.metaKey) || e.key.toLowerCase() !== "z") return;
    e.preventDefault(); e.stopPropagation();
    const l = e.shiftKey ? redo(doc) : undo(doc);
    if (l) toast(`${e.shiftKey ? "Redo" : "Undo"}: ${l}`, "info");
  });

  /* the k-path editor in Plot options; presets are kept in this browser for any plot with the same segments */
  const PRESETS = "mace.gui.kpath.presets";
  const readPresets = () => { try { return JSON.parse(localStorage.getItem(PRESETS) || "{}") || {}; } catch (e) { return {}; } };
  const writePresets = (p) => { try { localStorage.setItem(PRESETS, JSON.stringify(p)); } catch (e) { /* private */ } };
  const fits = (preset, computed) => K.visible(preset).length > 0 && preset.every((s) => computed.some((c) =>
    (K.canonical(c.from) === K.canonical(s.from) && K.canonical(c.to) === K.canonical(s.to)) || (K.canonical(c.from) === K.canonical(s.to) && K.canonical(c.to) === K.canonical(s.from))));
  function kpathEditor(kh) {
    const d = doc, band = d.data.band, computed = pathSegments(band);
    if (!d.bz && band.path) d.bz = createBZ({ api, path: band.path, onHover: (i) => KE.hoverFromBZ("plot", i), onSelect: (i) => KE.selectFromBZ("plot", i) });
    KE.render(kh, { id: "plot", mode: "display", computed, segs: d.opts.kpath, icon, bz: d.bz, onPreview: plotPreview,
      // a zoom along k belongs to the old path: the edited one starts unzoomed
      onChange: (segs, label) => { if (doc === d) setOpt(d, { kpath: segs, xr: null }, label); },
      presets: {
        list: () => Object.entries(readPresets()).map(([name, segs]) => ({ name, ok: fits(segs, computed) })),
        apply: (name) => { const p = readPresets()[name]; if (p && fits(p, computed) && doc === d) setOpt(d, { kpath: K.samePath(p, computed) ? null : p, xr: null }, `k-path preset ${name}`); },
        save: (name) => { const p = readPresets(); p[name] = d.opts.kpath || computed; writePresets(p); toast(`Saved the k-path preset ${name}`, "info"); renderOptions(); },
        remove: (name) => { const p = readPresets(); delete p[name]; writePresets(p); renderOptions(); },
      } });
  }
  optsHost.addEventListener("keydown", (e) => {
    // Mod+Z in the k-path editor is the plot's undo (the other controls keep their own, a text field's)
    if (!doc || !(e.ctrlKey || e.metaKey) || e.key.toLowerCase() !== "z" || !e.target.closest(".po-kpath") || e.target.matches("input")) return;
    e.preventDefault(); e.stopPropagation();
    KE.keepFocus("plot");
    const l = e.shiftKey ? redo(doc) : undo(doc);
    if (l) toast(`${e.shiftKey ? "Redo" : "Undo"}: ${l}`, "info");
  });

  /* ------------------------------------------------------------ the Plot options inspector */
  function num(id, label, v, unit, step = "0.5") {
    return `<label class="po-row" for="${id}"><span>${esc(label)}</span><input id="${id}" type="number" step="${step}" value="${v ?? ""}" placeholder="auto"><span class="po-u">${esc(unit)}</span></label>`;
  }
  /* the options are drawn again after every change: the control that had the focus keeps it, so the keyboard
   * (arrow keys on a slider, a number field) can go on changing it */
  function renderOptions() {
    const ae = document.activeElement, id = ae && ae.id && optsHost.contains(ae) ? ae.id : null;
    renderOptionsNow();
    const again = id && optsHost.querySelector(`#${CSS.escape(id)}`);
    if (again) again.focus({ preventScroll: true });
  }
  const resetButton = () => `<div class="po-acts"><button class="tb-btn-text" data-po="reset"${doc.pos > 0 || JSON.stringify(doc.opts) !== JSON.stringify(defaultsFor(doc)) ? "" : " disabled"}>${icon("rotate-ccw")}<span>Reset Plot Options</span></button></div>`;
  /* Plot options › Style and the toolbar's Style ▾ menu draw ONE list of controls (styleControls), each an
     option of the plot (presets.py STYLE_OPTIONS names; undo, Reset; "Settings" follows Settings, shown in
     brackets); Make This the Default writes them to Settings. A band + DOS plot has a Bands group (spin colors,
     band color, β line) and a DOS group (DOS colors, spin layout, total, smoothing), whichever of the two files
     was opened: the groups follow the figure's data, not the opened file's kind. A BAND.DAT alone has the Bands
     group only, a DOSS.DAT alone the DOS group only (its total's spin colors, color and β line in it). */
  const PRESET_LABEL = (id) => { const p = H.presets && H.presets() && H.presets().SHARED.presets[id]; return p ? p.label : id; };
  const SCHEME_LABEL = (id) => (id === "auto" ? "Auto (the plot style's)" : id === "custom" ? "Custom" : ((H.presets && H.presets() && H.presets().SHARED.spin_schemes[id]) || {}).label || id);
  const DOS_LABEL = { auto: "Auto", total: "Total only", element: "By element", orbital: "By orbital", "element-orbital": "Element × orbital" };
  const DSPIN_LABEL = { auto: "Auto", mirrored: "Mirrored", overlaid: "Overlaid" };
  // DOS smoothing turned on: the shared width (presets.py DOS_SMOOTH_SIGMA, 0.05 eV); the Style menu's σ choices
  const SMOOTH_SIGMA = () => ((H.presets && H.presets() && H.presets().SHARED.dos_layout) || {}).smooth_sigma || 0.05;
  const SIGMA_CHOICES = [0.02, 0.05, 0.1, 0.2];
  const isBandDos = (d) => !!d && !isOwn(d) && !isSpec(d);
  function lookNow(d = doc) {
    if (isBandDos(d)) return envOf(d.opts).look;
    return { style: presetOf(familyOf(d)) };
  }
  /* the plot's style controls in groups: [{ key, head, ctrls }]; a control is { id (the Inspector's input),
     mv (the Style menu's value prefix), type (select, color, toggle, number), label, current (the value drawn),
     value (a select's own option, null: Settings), settings (Settings' value, as text), choices, apply(v) →
     [patch, undo label, merge key] } */
  function styleControls(d = doc) {
    const pp = H.presets ? H.presets() : null;
    if (!pp || !d || !d.opts) return [];
    const o = d.opts, fam = familyOf(d), look = lookNow(d);
    const base = isBandDos(d) ? settingsLook() : { style: H.plotPreset ? H.plotPreset(fam) : "default" };
    const choose = (id, mv, name, label, value, current, choices, settings, word) => ({ id, mv, type: "select", label, value, current, choices, settings,
      apply: (v) => [{ [name]: v || null }, `${label} ${v ? word(v) : "as Settings"}`] });
    const groups = [{ key: "plot", head: "Plot style", ctrls: [choose("po-style", "style", "style", "Plot style", o.style, look.style,
      pp.SHARED.order.filter((id) => pp.SHARED.presets[id].families.includes(fam)).map((id) => [id, PRESET_LABEL(id)]), PRESET_LABEL(base.style), PRESET_LABEL)] }];
    if (!isBandDos(d)) return groups;
    const data = d.data || {}, band = data.band, dos = data.dos;
    if (!band && !dos) return groups;
    const color = (id, mv, name, label, current, title) => ({ id, mv, type: "color", label, current, title,
      apply: (v) => [{ [name]: v }, `${label} ${v}`, `opt:${id}`] });
    // the spins' look: the bands' and the DOS total's at once (a DOSS.DAT alone: the total's)
    const spin = [choose("po-spinc", "spin", "spin_colors", "Spin colors", o.spin_colors, look.spin_colors || "auto",
      pp.SHARED.spin_color_choices.map((v) => [v, SCHEME_LABEL(v)]), SCHEME_LABEL(base.spin_colors), SCHEME_LABEL)];
    const tone = H.plotTone ? H.plotTone() : "light", pid = pp.resolveId(look.style, "band");
    if (look.spin_colors === "custom") {
      const pair = pp.spinColors(pid, "band", tone, look.spin_colors, look.spin_up_color, look.spin_down_color);
      spin.push(color("po-spinup", "spinup", "spin_up_color", "Spin up color", pair.up), color("po-spindn", "spindn", "spin_down_color", "Spin down color", pair.down));
    }
    // a run that is not spin-polarized: its bands and one-sided DOS total in one neutral color (--band-color)
    if ((band && !band.spin_polarized) || (dos && !dos.spin_polarized)) {
      const bc = pp.bandColor(pid, "band", tone, look.band_color, tokenOf("band")("--plot-ink") || tokenOf("band")("--text") || undefined);   // undefined: bandColor's own black
      spin.push(color("po-bandc", "bandc", "band_color", band ? "Band color" : "Total color", bc,
        "Bands and the one-sided DOS total of a run that is not spin-polarized (mace plotting --band-color)"));
    }
    const dash = look.spin_down_style !== "solid";
    spin.push({ id: "po-bdash", mv: "bdash", type: "toggle", label: "β dashed", current: dash,
      title: "Spin down (β) dashed, spin up solid: the spins read apart without color (mace plotting --spin-down-solid turns it off)",
      apply: (v) => [{ spin_down_style: v ? "dash" : "solid" }, v ? "β dashed" : "β solid"] });
    if (band) groups.push({ key: "bands", head: "Bands", ctrls: spin });
    if (dos) {
      const sm = Number(look.dos_smooth) || 0;
      const ctrls = [
        choose("po-dosc", "dos", "dos_colors", "DOS colors", o.dos_colors, look.dos_colors || "auto", Object.entries(DOS_LABEL), DOS_LABEL[base.dos_colors] || base.dos_colors, (v) => DOS_LABEL[v] || v),
        choose("po-dspin", "dspin", "dos_spin", "DOS spin layout", o.dos_spin, look.dos_spin || "auto", Object.entries(DSPIN_LABEL), DSPIN_LABEL[base.dos_spin] || base.dos_spin, (v) => DSPIN_LABEL[v] || v),
        { id: "po-dtotal", mv: "dtotal", type: "toggle", label: "DOS total", current: look.dos_total !== false,
          apply: (v) => [{ dos_total: v }, v ? "DOS total" : "No DOS total"] },
        // DOS smoothing (display only, labeled in the plot): mace plotting --dos-smooth, [plots] dos_smooth
        { id: "po-dsmooth", mv: "dsmooth", type: "toggle", label: "DOS smoothing", current: sm > 0, title: "Gaussian smoothing of the drawn DOS, display only (mace plotting --dos-smooth)",
          apply: (v) => [{ dos_smooth: v ? SMOOTH_SIGMA() : 0 }, v ? "DOS smoothing" : "No DOS smoothing"] }];
      if (sm > 0) ctrls.push({ id: "po-dsigma", mv: "dsigma", type: "number", label: "Smoothing σ", unit: "eV", step: "0.01", current: sm,
        choices: [...new Set([...SIGMA_CHOICES, sm])].sort((a, b) => a - b).map((v) => [String(v), `σ ${v} eV`]),
        apply: (v) => { const s = Math.min(1, Math.max(0, Number(v) || 0)); return [{ dos_smooth: s }, `DOS smoothing σ ${s} eV`, "opt:dsigma"]; } });
      groups.push({ key: "dos", head: "DOS", ctrls: band ? ctrls : spin.concat(ctrls) });
    }
    return groups;
  }
  const styleControl = (pick) => { for (const g of styleControls()) for (const c of g.ctrls) if (pick(c)) return c; return null; };
  function styleSection() {
    const pp = H.presets ? H.presets() : null;
    if (!pp || !doc) return "";
    const sel = (c) => `<label class="po-row" for="${c.id}"><span>${esc(c.label)}</span><select id="${c.id}">` +
      `<option value=""${c.value === null || c.value === undefined ? " selected" : ""}>Settings (${esc(c.settings)})</option>` +
      c.choices.map(([v, t]) => `<option value="${esc(v)}"${c.value === v ? " selected" : ""}>${esc(t)}</option>`).join("") + "</select></label>";
    const tip = (c) => (c.title ? ` title="${esc(c.title)}"` : "");
    const ctrl = (c) => c.type === "select" ? sel(c)
      : c.type === "color" ? `<label class="po-row" for="${c.id}"${tip(c)}><span>${esc(c.label)}</span><input id="${c.id}" type="color" value="${esc(c.current)}"></label>`
      : c.type === "toggle" ? `<label class="chk po-row"${tip(c)}><input type="checkbox" id="${c.id}"${c.current ? " checked" : ""}> ${esc(c.label)}</label>`
      : num(c.id, c.label, c.current, c.unit, c.step);
    let h = `<div class="po-style" role="group" aria-label="Style"><div class="po-h">Style</div>`;
    for (const g of styleControls()) {
      h += `<div class="po-sgroup" role="group" aria-label="${esc(g.head)}" data-group="${g.key}">` +
        (g.key === "plot" ? "" : `<div class="po-h po-sub">${esc(g.head)}</div>`) + g.ctrls.map(ctrl).join("") + "</div>";
    }
    const own = pp.STYLE_OPTIONS.some((x) => doc.opts[x.name] !== null && doc.opts[x.name] !== undefined);
    h += `<div class="po-acts"><button class="tb-btn-text" data-po="default"${own ? "" : " disabled"} title="Save this plot's style as the default of every plot (Settings)">Make This the Default</button></div></div>`;
    return h;
  }
  /* one style control's new value (the Inspector's input or the Style menu's pick): one undoable option change */
  function styleApply(c, v) {
    if (!doc || !c) return;
    const [patch, label, mergeKey] = c.apply(v);
    setOpt(doc, patch, label, mergeKey || null);
  }
  function makeDefault() {
    if (!doc || !H.setSetting) return;
    const look = lookNow(), fam = familyOf(doc), set = (k, v) => { try { H.setSetting(k, v); } catch (e) { /* an unknown key */ } };
    if (doc.opts.style) {
      // the family's own preset when one is set in Settings, else every family's
      const own = setting(`plots.style.${fam}`, "same");
      set(own && own !== "same" ? `plots.style.${fam}` : "plots.style", look.style);
    }
    if (isBandDos(doc)) {
      // both halves of a band + DOS plot
      set("plots.spinColors", look.spin_colors); set("plots.spinUp", look.spin_up_color || ""); set("plots.spinDown", look.spin_down_color || "");
      set("plots.bandColor", look.band_color || "");
      set("plots.spinDownDash", look.spin_down_style); set("plots.dosColors", look.dos_colors); set("plots.spinDos", look.dos_spin);
      set("plots.dosTotal", look.dos_total !== false);
      set("plots.dosSmooth", Number(look.dos_smooth) || 0);
    }
    const clear = Object.fromEntries((H.presets ? H.presets().STYLE_OPTIONS : []).map((x) => [x.name, null]).filter(([k]) => k in doc.opts));
    setOpt(doc, clear, "Style saved as the default");
    toast("This plot's style is the default now (Settings)", "success");
  }
  /* the toolbar's Style ▾ menu: the same controls (styleControls), one pick at a time; a color opens the
     system color picker, a number its choices */
  function styleMenu() {
    if (!doc) return [];
    const items = [];
    styleControls().forEach((g, k) => {
      if (k) items.push("-", { head: g.head, group: g.key });
      for (const c of g.ctrls) {
        if (c.type === "select" || c.type === "number") {
          if (c.type === "number" || g.key !== "plot") items.push({ head: c.label });
          else items.push({ head: g.head, group: g.key });
          c.choices.forEach(([v, t]) => items.push({ v: `${c.mv}:${v}`, label: t, checked: String(c.current) === String(v) }));
        } else if (c.type === "toggle") items.push({ v: c.mv, label: c.label, checked: !!c.current, toggle: true });
        else items.push({ v: c.mv, label: `${c.label}…` });
      }
    });
    if (!items.length) return items;
    items.push("-", { v: "settings", label: "Follow Settings" });
    if (cliKind(doc)) items.push({ v: "cli", label: "Copy CLI Command" });
    return items;
  }
  function pickColor(c) {
    document.getElementById("pk-colorpick")?.remove();
    const i = document.createElement("input");
    i.type = "color"; i.id = "pk-colorpick"; i.value = c.current; i.setAttribute("aria-label", c.label);
    i.style.cssText = "position:fixed;left:50%;top:40%;width:1px;height:1px;opacity:0;border:0;padding:0";
    const d = doc;
    i.addEventListener("change", () => { if (doc === d) styleApply(c, i.value); i.remove(); });
    i.addEventListener("blur", () => setTimeout(() => i.isConnected && i.remove(), 60000));
    document.body.appendChild(i);
    try { if (i.showPicker) i.showPicker(); else i.click(); } catch (e) { i.click(); }
  }
  function styleSet(v) {
    if (!doc) return;
    const at = v.indexOf(":"), k = at < 0 ? v : v.slice(0, at), val = at < 0 ? null : v.slice(at + 1);
    if (k === "settings") {
      const clear = Object.fromEntries((H.presets ? H.presets().STYLE_OPTIONS : []).map((x) => [x.name, null]).filter(([n]) => n in doc.opts));
      setOpt(doc, clear, "Style as Settings");
      return;
    }
    const c = styleControl((x) => x.mv === k);
    if (!c) return;
    if (c.type === "toggle") styleApply(c, !c.current);
    else if (c.type === "color") pickColor(c);
    else styleApply(c, val);
  }
  /* Copy CLI Command (clicmd.js): mace plotting with this plot's file, window, energy zero, unit and look */
  const cliKind = (d) => (!d || !d.data ? null : d.data.band && d.data.dos ? "bandos" : d.data.band ? "band" : d.data.dos ? "dos"
    : d.kind === "ir" || d.kind === "raman" ? (d.source === "modes" ? null : d.kind) : null);
  function cliText(d = doc) {
    const kind = cliKind(d);
    if (!kind) throw new Error("no mace plotting command draws this plot");
    const er = d.data.band || d.data.dos, file = kind === "dos" ? d.data.dos.name : kind === "band" || kind === "bandos" ? d.data.band.name : d.title;
    return cliCommand({ kind, file, emin: d.opts.emin, emax: d.opts.emax, shift: d.opts.shift || 0,
      reference: er && er.reference === "vacuum" ? "vacuum" : "fermi", unit: d.opts.unit || "eV", look: lookNow(d),
      kpath: kind !== "dos" && d.opts.kpath ? d.opts.kpath : null,
      // with a mace-folder.toml its [plots] and [units] values would replace any default left out
      explicit: !!(H.folderFile && H.folderFile()) });
  }
  function renderOptionsNow() {
    if (!doc) { optsHost.innerHTML = ""; return; }
    const o = doc.opts, sp = doc.kind === "ir" || doc.kind === "raman";
    let h = "";
    if (doc.kind === "map") {
      const M = doc.data && doc.data.map;
      if (M && M.maps.length > 1) h += `<label class="po-row" for="po-mwhich"><span>Map</span><select id="po-mwhich">${M.maps.map((m, k) =>
        `<option value="${k}"${(o.which || 0) === k ? " selected" : ""}>${esc(m.label)}</option>`).join("")}</select></label>`;
      h += `<label class="chk po-row"><input type="checkbox" id="po-mlog"${o.log ? " checked" : ""}> Logarithmic scale (positive maps)</label>` +
        num("po-mcont", "Contours", o.contours, "", "1") +
        `<label class="po-row" for="po-mstyle"><span>Contour style</span><select id="po-mstyle">${[["", "Plot style's"], ["filled", "Filled"], ["lines", "Lines"], ["both", "Filled and lines"]].map(([v, t]) =>
          `<option value="${v}"${(o.contour_style || "") === v ? " selected" : ""}>${t}</option>`).join("")}</select></label>` +
        `<label class="po-row" for="po-crange"><span>Color range</span><select id="po-crange">${[["robust", "Auto (1st–99th percentile)"], ["full", "Full (min to max)"], ["custom", "Custom"]].map(([v, t]) =>
          `<option value="${v}"${(o.crange || "robust") === v ? " selected" : ""}>${t}</option>`).join("")}</select></label>` +
        (o.crange === "custom" ? num("po-zmin", "Color from", o.zmin, o.log ? "log₁₀" : "", "any") + num("po-zmax", "Color to", o.zmax, o.log ? "log₁₀" : "", "any") : "");
      if (M) { const m = M.maps[o.which || 0]; h += `<p class="po-h">${esc(m.label)}: min ${m.min.toPrecision(4)} · max ${m.max.toPrecision(4)} (${esc(m.unit || M.unit)})</p>`; }
      h += styleSection() + resetButton();
      optsHost.innerHTML = h;
      return;
    }
    if (doc.kind === "convergence") {
      h += num("po-width", "Line width", o.width, "px", "0.2") +
        `<label class="chk po-row"><input type="checkbox" id="po-clog"${o.log ? " checked" : ""}> Logarithmic energy axis</label>` +
        `<p class="po-h">Each run's energy per atom at each printed geometry; molecules print only the initial and final ones.</p>`;
      h += styleSection() + resetButton();
      optsHost.innerHTML = h;
      return;
    }
    if (doc.kind === "transport") {
      const t = doc.data && doc.data.transport;
      if (t) {
        const Q = { SEEBECK: "Seebeck", SIGMA: "σ", SIGMAS: "σS", KAPPA: "κ₀", TDF: "TDF" }, zt = o.view === "zt";
        // ZT needs the run's Seebeck, σ and κ₀ files
        const canZt = ["SEEBECK", "SIGMA", "KAPPA"].every((k) => t.siblings[k]) && t.quantity !== "TDF";
        // on a row of its own: six choices do not fit beside the label at the default Inspector width
        h += `<div class="po-row po-stack"><span>Quantity</span><span class="segment po-tq" role="radiogroup" aria-label="Quantity">${Object.keys(Q).filter((k) => t.siblings[k]).map((k) =>
          `<button role="radio" aria-checked="${k === t.quantity && !zt}" data-tq="${esc(t.siblings[k])}">${Q[k]}</button>`).join("")}${canZt
          ? `<button role="radio" aria-checked="${zt}" data-tview="zt" title="Figure of merit: ZTₑ, and the total ZT with a lattice κ_L">ZT</button>` : ""}</span></div>`;
        if (t.blocks.length > 1 || t.blocks[0].T !== null) h += `<label class="po-row" for="po-temp"><span>Temperature</span><select id="po-temp">${t.blocks.map((b) =>
          `<option value="${b.T}"${(o.T ?? t.blocks[0].T) === b.T ? " selected" : ""}>${b.T} K</option>`).join("")}</select></label>`;
        if (zt) {
          const z = doc.data.zt;
          h += `<label class="po-row" for="po-kl"><span>Lattice κ_L</span><input id="po-kl" type="text" spellcheck="false" autocomplete="off" value="${esc(o.kl || "")}" placeholder="value or .csv beside the run" title="The lattice thermal conductivity in W/(m·K): one value, or a CSV file with T,kappa_L rows (interpolated linearly, never extrapolated), named as it sits beside the run (else from the served folder)"><span class="po-u">W/(m·K)</span></label>`;
          h += num("po-tau", "Relaxation time τ", o.tau, "fs", "1").replace('placeholder="auto"', `placeholder="${z && z.tau_run_fs ? `${z.tau_run_fs} (the run's)` : "the run's"}"`);
          h += num("po-width", "Line width", o.width, "px", "0.2");
          if (doc.data.ztError) h += `<p class="po-note po-err" role="alert">${esc(doc.data.ztError)}</p>`;
          if (z) {
            h += `<p class="po-note">ZTₑ peak (electrons only): ${esc(fmtZt(z.zt_e_max))}</p>`;
            if (z.zt_total_max) h += `<p class="po-note">ZT total peak: ${esc(fmtZt(z.zt_total_max))}, κ_L ${Number(z.zt_total_max.kappa_l).toPrecision(4)} W/(m·K), τ ${z.tau_fs} fs (${esc(z.tau_source)})</p>`;
            else if (z.refused) h += `<p class="po-note po-err" role="alert">${esc(z.refused)}</p>`;
            else if (!z.kappa_l_source && !doc.data.ztError) h += `<p class="po-note">Enter κ_L for the total ZT = S²σT/(κₑ + κ_L). σ and κ₀ are at the run's relaxation time (RELAXTIM); the total depends on it.</p>`;
          }
        } else {
          h += `<label class="po-row" for="po-tcomp"><span>Component</span><select id="po-tcomp">${[["avg", "Average of the diagonal"], ["xx", "xx"], ["yy", "yy"], ["zz", "zz"], ["all", "xx, yy, zz"]].map(([v, l]) =>
            `<option value="${v}"${o.component === v ? " selected" : ""}>${l}</option>`).join("")}</select></label>`;
          h += num("po-width", "Line width", o.width, "px", "0.2");
        }
      }
      h += styleSection() + `<div class="po-acts"><button class="tb-btn-text" data-po="reset">${icon("rotate-ccw")}<span>Reset Plot Options</span></button></div>`;
      optsHost.innerHTML = h;
      return;
    }
    if (!sp) {
      const band = doc.data && doc.data.band, dos = doc.data && doc.data.dos;
      const spin = (band && band.spin_polarized) || (dos && dos.spin_polarized);
      h += num("po-emin", "Energy from", o.emin, "eV") + num("po-emax", "Energy to", o.emax, "eV") +
        num("po-shift", "Fermi shift", o.shift, "eV", "0.01") + num("po-width", "Line width", o.width, "px", "0.2");
      if (spin) h += `<div class="po-row"><span>Spin</span><span class="segment po-spin" role="radiogroup" aria-label="Spin">${["both", "up", "down"].map((s) =>
        `<button role="radio" aria-checked="${o.spin === s}" data-spin="${s}">${s === "both" ? "Both" : s === "up" ? "α" : "β"}</button>`).join("")}</span></div>`;
      if (band && pathSegments(band)) h += `<div class="po-kpath"></div>`;
      // DOS colors (Settings › Plots, this plot's own choice over it) and the projections that mode draws, by their legend names
      if (dos) {
        // hidden: this plot's own list, or by default all but the 2-4 projections carrying the DOS (bandfig.js dosLook)
        const look = dosLook(dos, tokenOf("band"), o);
        h += `<div class="po-proj"><div class="po-h">Projections</div>${look.lines.filter((l) => l.show).map((l) =>
          `<label class="chk"><input type="checkbox" data-proj="${esc(l.label)}"${look.hidden.includes(l.label) ? "" : " checked"}> ${esc(l.name)}</label>`).join("")}</div>`;
      }
    } else {
      h += num("po-xmin", "Wavenumber from", o.xmin, "cm⁻¹", "10") + num("po-xmax", "Wavenumber to", o.xmax, "cm⁻¹", "10") +
        num("po-width", "Line width", o.width, "px", "0.2") +
        `<label class="chk po-row"><input type="checkbox" id="po-norm"${o.normalize ? " checked" : ""}> Normalize to the highest peak</label>`;
      const comps = doc.data ? Object.keys(doc.data.spectra[0].columns) : [];
      if (doc.data && (doc.kind === "raman" || comps.length > 1)) {
        const cur = o.component === "all" || comps.includes(o.component) ? o.component : comps[0];
        h += `<label class="po-row" for="po-comp"><span>Component</span><select id="po-comp">${["all", ...comps].map((c) => `<option value="${esc(c)}"${cur === c ? " selected" : ""}>${esc(c === "all" ? "All components" : c)}</option>`).join("")}</select></label>`;
      }
      if (doc.data && doc.data.group) h += `<label class="chk po-row"><input type="checkbox" id="po-avg"${o.average ? " checked" : ""}> Average the ${doc.data.group.length} conformers</label>`;
      if (doc.data && doc.data.spectra[0].broadening) h += `<p class="po-h">Broadening as CRYSTAL wrote it: ${esc(doc.data.spectra[0].broadening)}</p>`;
      if (doc.source === "modes") {
        const s0 = doc.data && doc.data.spectra[0];
        h += num("po-fwhm", "Peak width (FWHM)", o.fwhm ?? (s0 ? s0.default_fwhm : 8), "cm⁻¹", "1");
        if (doc.kind === "ir") h += `<label class="po-row" for="po-shape"><span>Line shape</span><select id="po-shape">${[["lorentzian", "Lorentzian"], ["gaussian", "Gaussian"]].map(([v, l]) =>
          `<option value="${v}"${o.shape === v ? " selected" : ""}>${l}</option>`).join("")}</select></label>`;
        else h += `<label class="po-row" for="po-eta"><span>Lorentzian share (η)</span><input id="po-eta" type="range" min="0" max="1" step="0.1" value="${o.eta ?? 1}"><span class="po-u">${(o.eta ?? 1).toFixed(1)}</span></label>`;
      }
    }
    h += styleSection() + resetButton();
    optsHost.innerHTML = h;
    const kh = optsHost.querySelector(".po-kpath");
    if (kh) kpathEditor(kh);
  }
  optsHost.addEventListener("change", (e) => {
    if (!doc) return;
    const t = e.target, o = doc.opts;
    const v = (x) => (x === "" ? null : Number(x));
    const map = { "po-emin": ["emin", "Energy from"], "po-emax": ["emax", "Energy to"], "po-shift": ["shift", "Fermi shift"],
      "po-width": ["width", "Line width"], "po-xmin": ["xmin", "Wavenumber from"], "po-xmax": ["xmax", "Wavenumber to"] };
    if (map[t.id]) { const [k, l] = map[t.id]; const val = v(t.value); setOpt(doc, { [k]: k === "shift" || k === "width" ? (val ?? defaultsFor(doc)[k]) : val }, `${l} ${t.value || "auto"}`, `opt:${k}`); }
    else if (t.id === "po-norm") setOpt(doc, { normalize: t.checked }, t.checked ? "Normalize" : "No normalization");
    else if (t.id === "po-avg") {
      const d = doc;
      const go = () => setOpt(d, { average: t.checked }, t.checked ? "Average the conformers" : "Each conformer");
      if (t.checked && !d.data.average) api("spectrum?" + q({ path: d.path, average: 1 })).then((a) => { d.data.average = a; go(); }).catch((err) => toast(err.message, "error"));
      else go();
    }
    else if (t.id === "po-comp") setOpt(doc, { component: t.value }, `Component ${t.value}`);
    else if (t.id && t.id.startsWith("po-") && styleControl((c) => c.id === t.id)) {
      // Plot options › Style (styleControls): the same controls as the Style menu
      const c = styleControl((x) => x.id === t.id);
      styleApply(c, c.type === "toggle" ? t.checked : t.value);
    }
    else if (t.id === "po-crange") setOpt(doc, { crange: t.value }, `Color range ${t.value}`);
    else if (t.id === "po-zmin" || t.id === "po-zmax") setOpt(doc, { [t.id === "po-zmin" ? "zmin" : "zmax"]: v(t.value) }, `Color ${t.id === "po-zmin" ? "from" : "to"} ${t.value || "auto"}`, `opt:${t.id}`);
    else if (t.id === "po-temp") setOpt(doc, { T: Number(t.value) }, `Temperature ${t.value} K`);
    else if (t.id === "po-mlog") setOpt(doc, { log: t.checked }, t.checked ? "Logarithmic scale" : "Linear scale");
    else if (t.id === "po-mcont") setOpt(doc, { contours: Math.max(2, Math.min(60, Number(t.value) || 20)) }, `Contours ${t.value}`);
    else if (t.id === "po-mstyle") setOpt(doc, { contour_style: t.value || null }, `Contour style ${t.value || "of the plot style"}`);
    else if (t.id === "po-mwhich") setOpt(doc, { which: Number(t.value) }, "Map");
    else if (t.id === "po-clog") setOpt(doc, { log: t.checked }, t.checked ? "Logarithmic axis" : "Linear axis");
    else if (t.id === "po-tcomp") setOpt(doc, { component: t.value }, `Component ${t.value}`);
    else if (t.id === "po-kl") setOpt(doc, { kl: t.value.trim() }, t.value.trim() ? `κ_L ${t.value.trim()}` : "No κ_L");
    else if (t.id === "po-tau") setOpt(doc, { tau: v(t.value) }, t.value ? `τ ${t.value} fs` : "τ of the run");
    else if (t.id === "po-fwhm" || t.id === "po-shape" || t.id === "po-eta") {
      const d = doc, patch = t.id === "po-fwhm" ? { fwhm: Math.min(50, Math.max(2, Number(t.value) || 8)) }
        : t.id === "po-shape" ? { shape: t.value } : { eta: Number(t.value), shape: Number(t.value) < 1 ? "pseudo-voigt" : "lorentzian" };
      Object.assign(d.opts, patch);
      d.data = null;
      fetchData(d).then(() => { if (doc === d) { setOpt(d, patch, t.id === "po-fwhm" ? `Peak width ${patch.fwhm} cm⁻¹` : `Line shape ${d.opts.shape}`); note(d); } })
        .catch((err) => toast(err.message, "error"));
    }
    else if (t.dataset.proj) {
      const hidden = new Set(doc.data && doc.data.dos ? dosLook(doc.data.dos, tokenOf("band"), o).hidden : o.hidden || []);
      if (t.checked) hidden.delete(t.dataset.proj); else hidden.add(t.dataset.proj);
      setOpt(doc, { hidden: [...hidden] }, `${t.checked ? "Show" : "Hide"} ${t.dataset.proj}`);
    }
  });
  optsHost.addEventListener("click", (e) => {
    const tv = e.target.closest("[data-tview]");
    if (tv && doc && tv.getAttribute("aria-checked") !== "true") { setOpt(doc, { view: "zt" }, "Figure of merit ZT"); return; }
    const tq = e.target.closest("[data-tq]");
    if (tq && doc && doc.opts.view === "zt" && tq.dataset.tq === doc.path) { setOpt(doc, { view: "quantity" }, "Quantity"); return; }
    if (tq && doc && tq.getAttribute("aria-checked") !== "true" && H.openPath) { H.openPath(tq.dataset.tq); return; }
    const s = e.target.closest("[data-spin]");
    if (s && doc) { setOpt(doc, { spin: s.dataset.spin }, `Spin ${s.dataset.spin}`); return; }
    const b = e.target.closest("[data-po]");
    if (b && doc && b.dataset.po === "reset") { reset(doc); toast("Plot options reset", "info", { label: "Undo", run: () => undo(doc) }); }
    if (b && doc && b.dataset.po === "default") makeDefault();
  });

  /* ------------------------------------------------------------ export and the shared options (plotkit) */
  const stemOf = (d) => d.title.replace(/\.[^.]+(\.[^.]+)?$/, "");
  async function image(fmt) {
    const scale = fmt === "png" ? Number(setting("plots.exportScale", 2)) || 2 : 1;
    const url = await window.Plotly.toImage(el, { format: fmt, scale, width: el.clientWidth, height: el.clientHeight });
    download(PK.dataUrlBlob(url), `${stemOf(doc)}.${fmt}`);
  }
  kit = PK.attach($(".pv-acts"), {
    get name() { return doc ? stemOf(doc) : "plot"; },
    host: el,
    get features() {
      if (doc && isOwn(doc)) return { logY: false, logX: false, units: null, range: false, width: false, png: true, svg: true, image: true, csv: true, pub: false, style: true, cli: false };
      return { logY: true, logX: !!doc && isSpec(doc), units: doc && isSpec(doc) ? "wavenumber" : "energy", range: true, width: true,
        png: true, svg: true, image: true, csv: true, pub: !(doc && doc.source === "modes"), style: true, cli: !!cliKind(doc) };
    },
    get: () => (doc ? doc.opts : {}),
    set: (patch, label) => doc && setOpt(doc, patch, label),
    axes: () => (doc && isSpec(doc) ? { x: "Wavenumber", y: "Intensity" } : { x: "k-path / DOS", y: "Energy" }),
    table: () => tableOf(doc),
    png: () => image("png"), svg: () => image("svg"), image: () => exportImage(),
    pub: () => publication(),
    styleMenu: () => styleMenu(), styleSet: (v) => styleSet(v), cli: () => cliText(),
    resetZoom: () => { if (window.Plotly && el.layout) { el.__sig = null; redraw(); } },
  }, { toast, download, icon });
  /* the plot's look as the plot task's parameters (plots.py look_args: the presets.py STYLE_OPTIONS names,
     only those that are not the defaults) */
  function lookOf(d) {
    const look = lookNow(d), out = {};
    for (const [k, v] of Object.entries(look)) {
      const dflt = H.presets ? H.presets().STYLE_DEFAULTS[k] : undefined;
      if (v === null || v === undefined || v === dflt) continue;
      out[k] = typeof v === "boolean" ? (v ? 1 : 0) : v;
    }
    return out;
  }
  async function publication() {
    // the edited k-path goes with it: mace plotting --band-path draws the same path (formatting is all it adds)
    // what is on screen: the band structure and its DOS when both are shown (mace plotting --band-dos)
    const both = !!(doc.data && doc.data.band && doc.data.dos), kind = both ? "bandos" : null;
    const edit = (doc.kind === "band" || both) && doc.opts.kpath ? { band_path: JSON.stringify(doc.opts.kpath) } : {};
    // the window on screen (Settings' or the folder's, or the one typed in Plot options): --e-lower/--e-upper
    const win = (doc.kind === "band" || doc.kind === "dos") && Number.isFinite(Number(doc.opts.emin)) && Number.isFinite(Number(doc.opts.emax))
      ? { emin: Number(doc.opts.emin), emax: Number(doc.opts.emax) } : {};
    // and its energy zero: the Fermi shift (dragged or typed; --fermi-shift) and, with a POTC run, the
    // vacuum scale the API drew (--energy-reference vacuum; the server copies the POTC files along)
    // with a mace-folder.toml the server passes every value, the defaults too (plots.py explicit), as Copy
    // CLI Command writes them: the file's [plots] values must not replace what the screen shows; the screen's
    // Fermi reference is then named, as auto would take the file's energy_reference
    const folderFile = !!(H.folderFile && H.folderFile());
    const er = doc.data && (doc.data.band || doc.data.dos);
    const ref = (doc.kind === "band" || doc.kind === "dos") && er
      ? { ...(Number(doc.opts.shift) ? { shift: Number(doc.opts.shift) } : {}),
          ...(er.reference === "vacuum" ? { reference: "vacuum" } : folderFile ? { reference: "fermi" } : {}) } : {};
    // and its look (Settings › Appearance › Plot style, Plots): the same preset, DOS colors and spin styling
    const look = lookOf(doc);
    const unit = (doc.kind === "band" || doc.kind === "dos") && doc.opts.unit && doc.opts.unit !== "eV" ? { unit: doc.opts.unit } : {};
    const r = await apiTask("plot?" + q({ path: doc.path, format: "svg", detach: 1, ...(kind ? { kind } : {}), ...edit, ...win, ...ref, ...unit, ...look }), undefined, null, { get: true });
    const svg = r.images.find((n) => n.endsWith(".svg"));
    if (!svg) throw new Error("mace plotting made no SVG: " + (r.log || "").slice(-300));
    const res = await api("plot-image?" + q({ key: r.key, name: svg }), { blob: true });
    download(res.blob, svg);
    return r;
  }
  /* Export Image's dialog (plan E9): a pixel or physical size at a DPI, the same as the 3D views */
  function exportImage() {
    if (!doc || !H.exportImage) return;
    const w = el.clientWidth, h = el.clientHeight;
    H.exportImage({ stem: stemOf(doc), format: "png", plot: { size: [w, h],
      renderPng: (W) => window.Plotly.toImage(el, { format: "png", width: w, height: h, scale: W / w }),
      renderSvg: async () => decodeURIComponent((await window.Plotly.toImage(el, { format: "svg", width: w, height: h })).replace(/^data:image\/svg\+xml,/, "")) } });
  }
  /** The plotted data as {columns, rows}, in the plotted units (CSV, Copy as TSV). */
  function tableOf(d) {
    const rows = [];
    if (d.kind === "map") {
      const m = d.data.map.maps[d.opts.which || 0];
      m.values.forEach((col, j) => col.forEach((v, i) => rows.push([m.x[i], m.y[j], v])));
      return { columns: ["x_A", "y_A", "value"], rows };
    }
    if (d.kind === "convergence") {
      for (const r of d.data.convergence.runs) r.energies.forEach((e, i) => rows.push([r.name, i + 1, e ?? "", r.n_atoms, r.status || ""]));
      return { columns: ["run", "geometry", "energy_au", "n_atoms", "status"], rows };
    }
    if (isZt(d) && d.data.zt) {
      const z = d.data.zt;
      for (const b of z.blocks) b.mu.forEach((m, i) => rows.push([m, b.T, b.zt_e[i] ?? "", b.zt_total ? b.zt_total[i] ?? "" : ""]));
      return { columns: ["mu_eV", "T_K", "zt_electronic", "zt_total"], rows };
    }
    if (d.kind === "transport") {
      // one block per temperature, all with the same columns (the first block's)
      const t = d.data.transport, cols = ["avg", "xx", "yy", "zz"].filter((k) => t.blocks[0][k]);
      for (const b of t.blocks) b.mu.forEach((m, i) => rows.push([m, b.T ?? "", ...cols.map((c) => (b[c] ? b[c][i] : ""))]));
      return { columns: [t.x === "energy" ? "energy_eV" : "mu_eV", "T_K", ...cols.map((c) => `${t.quantity.toLowerCase()}_${c}`)], rows };
    }
    if (isSpec(d)) {
      const s = d.data.spectra[0], cols = Object.keys(s.columns), fx = PK.WAVENUMBER_UNITS[d.opts.unit] || 1;
      const u = (PK.UNIT_LABEL[d.opts.unit] || "cm⁻¹").replace("⁻¹", "-1");
      const wl = s.wavelength_nm;          // IRSPEC.DAT's second column
      s.x.forEach((x, i) => rows.push([x * fx, ...(wl ? [wl[i]] : []), ...cols.map((c) => s.columns[c][i])]));
      return { columns: [`${d.opts.unit === "cm-1" ? "wavenumber" : "energy"}_${u}`, ...(wl ? ["wavelength_nm"] : []), ...cols], rows };
    }
    const { dos } = d.data, band = bandView(d), shift = d.opts.shift || 0, f = PK.ENERGY_UNITS[d.opts.unit] || 1, u = d.opts.unit || "eV";
    if (band) {
      const cols = ["path", ...band.up.map((_, k) => `band${k + 1}${band.spin_polarized ? "_up" : ""}_${u}`), ...(band.down || []).map((_, k) => `band${k + 1}_down_${u}`)];
      band.x.forEach((x, i) => { if (x !== null) rows.push([x, ...band.up.map((b) => (b[i] - shift) * f), ...(band.down || []).map((b) => (b[i] - shift) * f)]); });
      if (!dos) return { columns: cols, rows };
      return { columns: cols, rows };
    }
    const cols = [`energy_${u}`, ...dos.projections.flatMap((p) => (dos.spin_polarized ? [`${p.label}_up`, `${p.label}_down`] : [p.label]))];
    dos.energy.forEach((e, i) => rows.push([(e - shift) * f, ...dos.projections.flatMap((p) => (dos.spin_polarized ? [p.up[i], p.down[i]] : [p.up[i]]))]));
    return { columns: cols, rows };
  }

  /** "Broadened from 54 IR-active modes (Lorentzian, FWHM 8 cm⁻¹, CRYSTAL's default)" */
  function sourceNote(d) {
    const s = d.data.spectra[0];
    const why = s.dampfac ? "the run's DAMPFAC" : s.fwhm === 8 ? "CRYSTAL's default" : "set here";
    return `Broadened from ${s.n_active} ${d.kind === "ir" ? "IR" : "Raman"}-active modes (${esc(s.shape_label)}, FWHM ${s.fwhm} cm⁻¹, ${why})`;
  }
  function note(d) {
    if (d.source !== "modes" || !d.data) return;
    const line = sourceNote(d) + " · click a peak to animate its mode", tip = esc(`${line.replace(/<[^>]+>/g, "")}. Drag to zoom, double-click to reset; Mod+Z undoes an option.`);
    $(".pv-note").classList.add("one");
    $(".pv-note").innerHTML = `<span class="pv-note-line">${line}</span><span class="pv-help" data-help tabindex="0" role="img" title="${tip}" aria-label="${tip}">${icon("circle-help")}</span>`;
  }
  function rebuild() { if (doc && doc.data && el.offsetWidth > 0) { el.__sig = null; redraw(); } }
  document.addEventListener("mace-theme", rebuild);

  return {
    async open(d) {
      doc = d;
      $(".pv-img").hidden = true; $(".pv-frame").hidden = true; el.hidden = false; $(".pv-acts").hidden = false;
      setTitle(d.title);
      $(".pv-readout").textContent = "";
      load(d);
      renderOptions();
      if (kit) kit.refresh();
      $(".pv-note").classList.remove("one");
      $(".pv-note").innerHTML = `<span class="spinner"></span> Reading ${esc(d.title)}…`;
      try {
        await fetchData(d);
        if (doc !== d) return;
        const { band, dos } = d.data || {};
        const notes = [], help = [];
        if (d.data && d.data.map) {
          notes.push(`${d.data.map.maps.length} map${d.data.map.maps.length === 1 ? "" : "s"} from CRYSTAL's fort.25 (MAPN), the plane through A, B and C`);
        } else if (d.data && d.data.convergence) {
          notes.push(`${d.data.convergence.runs.length} runs: energy per atom at each geometry, from the outputs' own energies, above the lowest final energy of the set`);
        } else if (d.data && d.data.transport) {
          const t = d.data.transport;
          notes.push(`${esc(t.label)} from BOLTZTRA (${t.blocks.length} temperature${t.blocks.length === 1 ? "" : "s"}), read with the database extractor's own reader`);
        } else if (band || dos) {
          const files = [band && band.name, dos && dos.name].filter(Boolean);
          setTitle(files.join("  +  "));
          // one short line; the longer guidance is in the help icon's tooltip
          const meV = band && Math.max(1, Math.round(band.spin_max_diff * 1000));
          const sw = BF.dashWord(envOf(d.opts).look.spin_down_style || "dash");
          notes.push(band ? (band.spin_polarized ? `spin-polarized (α and β up to ${meV} meV apart): α solid, β <span class="pv-bword">${sw}</span>`
            : band.spin_identical ? `spin-polarized · α = β (non-magnetic, ≤${meV} meV)` : "closed shell")
            : dos.spin_polarized ? "spin-polarized" : dos.spin_summed ? "spin-polarized · α = β (non-magnetic): total α + β drawn" : "closed shell");
          if (band && band.truncated) notes.push(`the ${band.up.length} bands nearest E<sub>F</sub> of ${band.n_bands}`);
          // the bands more than ±25 eV from E_F are not loaded: say so when the window reaches past them
          // (mace plotting and Copy CLI Command draw them)
          if (band && band.n_beyond > 0) {
            const W = Number(band.window_ev) || 25, sh = Number(d.opts.shift) || 0;
            if (Number(d.opts.emin) + sh < -W || Number(d.opts.emax) + sh > W)
              notes.push(`${band.n_beyond} band${band.n_beyond === 1 ? "" : "s"} more than ${W} eV from E<sub>F</sub> not loaded (mace plotting draws them)`);
          }
          notes.push((band || dos).reference === "vacuum" ? "vacuum scale · dotted = E<sub>F</sub>, drag to shift" : "dotted = Fermi level, drag to shift");
          if (band && band.spin_identical) help.push(`α and β agree within ${meV} meV, so one set of bands is drawn`);
          if (dos && dos.spin_polarized) help.push("DOS: ↑ to the right, ↓ mirrored to the left (total hatched)");
          if (dos && dos.window) help.push(`DOS computed from ${dos.window[0].toFixed(2)} to ${dos.window[1].toFixed(2)} eV${dos.edge_pileup ? " (its edge bin, the states beyond it, is left out)" : ""}`);
          if ((band || dos).reference === "vacuum") help.push("energies on the vacuum scale (POTC run found); the dotted line is E_F");
          help.push("dragging the dotted line shifts the energy reference");
        } else if (d.data && d.data.spectra) {
          const s = d.data.spectra[0];
          if (d.source === "modes") notes.push(sourceNote(d));
          else notes.push(`CRYSTAL ${d.kind === "ir" ? "IRSPEC" : "RAMSPEC"}.DAT · ${s.n_points.toLocaleString("en-US")} points, read with mace plotting's reader${s.broadening ? `; broadening ${esc(s.broadening)}` : ""}`);
          if (d.data.out && d.data.peaks.length) notes.push("click a peak to animate its mode");
        }
        const drag = "Drag to zoom, double-click to reset; Mod+Z undoes an option.";
        // one line for every plot's subtitle: what does not fit is in the help icon's tooltip, with the guidance
        if (!help.length) help.push(...notes);
        const tip = esc(`${help.join(" · ")}. ${drag}`.replace(/<\/?sub>/g, (t) => (t === "<sub>" ? "_" : "")).replace(/<[^>]+>/g, ""));
        $(".pv-note").classList.add("one");
        $(".pv-note").innerHTML = `<span class="pv-note-line">${notes.join(" · ")}</span>`
          + `<span class="pv-help" data-help tabindex="0" role="img" title="${tip}" aria-label="${tip}">${icon("circle-help")}</span>`;
        renderOptions();
        await redraw();
      } catch (e) {
        if (e.data && e.data.still_writing) {
          // a job is still writing the file: not an error; Retry reads it again
          $(".pv-note").innerHTML = `${esc(e.message)} <button class="tb-btn-text" data-act="retry-read">Retry</button>`;
          const b = $(".pv-note [data-act=retry-read]");
          b.addEventListener("click", () => { if (doc === d) { d.data = null; this.open(d); } }, { once: true });
          return;
        }
        $(".pv-note").textContent = e.message;
      }
    },
    /** An image or HTML plot already in the folder. */
    openImage(d, kind) {
      doc = null;
      el.hidden = true; $(".pv-acts").hidden = true; $(".pv-readout").textContent = ""; $(".pv-chips").innerHTML = "";
      setTitle(d.title);
      $(".pv-note").classList.remove("one");
      $(".pv-note").textContent = kind === "html" ? "An HTML plot in this folder, shown in a sandbox (its scripts cannot reach MACE)" : "An image in this folder";
      const src = "api/raw?" + q({ path: d.path });
      if (kind === "html") { const f = $(".pv-frame"); f.hidden = false; $(".pv-img").hidden = true; if (f.getAttribute("src") !== src) f.src = src; }
      else { const i = $(".pv-img"); i.hidden = false; $(".pv-frame").hidden = true; i.src = src; i.alt = d.title; }
      optsHost.innerHTML = `<p class="insp-hint">No options for a picture.</p>`;
    },
    close() { doc = null; },
    reset: () => doc && reset(doc), undo: () => doc && undo(doc), redo: () => doc && redo(doc),
    bz: () => doc && doc.bz ? doc.bz.state() : null,
    state: () => doc && { kind: doc.kind, source: doc.source || "file", opts: { ...doc.opts }, pos: doc.pos, hist: doc.hist.length, data: !!doc.data,
      drawn: el.dataset.drawn === doc.path && !building, peaks: doc.data && doc.data.peaks ? doc.data.peaks.length : 0,
      out: doc.data ? doc.data.out || null : null, fwhm: doc.data && doc.data.spectra ? doc.data.spectra[0].fwhm || null : null,
      zt: isZt(doc) && doc.data && doc.data.zt ? { key: doc.data.ztKey, error: doc.data.ztError, refused: doc.data.zt.refused,
        zt_e_max: doc.data.zt.zt_e_max, zt_total_max: doc.data.zt.zt_total_max, tau_fs: doc.data.zt.tau_fs,
        kappa_l_source: doc.data.zt.kappa_l_source } : null },
    figure: () => (el.data ? { traces: el.data.map((t) => ({ name: t.name, meta: t.meta, n: (t.x || []).length })), layout: el.layout } : null),
    setOpt: (patch, label) => doc && setOpt(doc, patch, label),
    /** Publication Figure… (mace plotting's SVG, downloaded); the task's result. */
    publication: () => (doc ? publication() : Promise.resolve(null)),
    /** Copy CLI Command's text (clicmd.js), or null when no mace plotting command draws this plot. */
    cliText: () => { try { return doc ? cliText() : null; } catch (e) { return null; } },
    el: () => el,
  };
}
