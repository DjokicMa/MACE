/* Plot documents (spec §1.2, plan B5): a band structure, a DOS, both on one
 * energy axis, an IR or a Raman spectrum, drawn by plotly from the package
 * (loaded on the first plot only, §8.5) from the arrays mace plotting's own
 * readers return. Each document has its Plot options in the inspector, with
 * their own undo (Mod+Z on the plot) and Reset; the Fermi line can be dragged
 * to shift the energy reference; hovering a band reads out k, E − E_F and the
 * band. Export: PNG/SVG from plotly, the publication figure from mace
 * plotting itself (a task), and the plotted data as CSV. */
import * as PK from "../plotkit.js";

const MERGE_MS = 800;

export function createPlotView(host, optsHost, H) {
  const { api, q, esc, icon, toast, cssVar, loadPlotly, apiTask, download } = H;
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
    bands: { emin: -6, emax: 6, shift: 0, spin: "both", width: 1.4, hidden: [] },
    spectrum: { xmin: null, xmax: null, normalize: false, component: "total", average: false, width: 1.4 },
    // BOLTZTRA transport data (plan E8): the temperature block and the tensor component shown
    transport: { T: null, component: "avg", width: 1.4 },
    // Compare Convergence (plan E12 fallback): energy per atom above the lowest final energy of the set
    convergence: { width: 1.6, log: false },
    // a fort.25 2D map (plan E6): which map (density, spin), log scale, contour count
    map: { which: 0, log: true, contours: 20 },
    // a spectrum broadened from a FREQ output's modes (plan E4): width and line shape; fwhm null = the run's DAMPFAC or 8
    modes: { xmin: null, xmax: null, normalize: false, component: "total", average: false, width: 1.4, fwhm: null, shape: "lorentzian", eta: 1 },
  };
  // the shared options (plotkit): log axes, units, ranges; new documents take the Settings defaults
  const setting = (k, dflt) => { try { const v = H.setting ? H.setting(k) : undefined; return v === undefined || v === null || v === "" ? dflt : v; } catch (e) { return dflt; } };
  const isSpec = (d) => d.kind === "ir" || d.kind === "raman";
  // the plots of plan E (a fort.25 map, Compare Convergence, BOLTZTRA data) keep their own options only
  const isOwn = (d) => d.kind === "map" || d.kind === "convergence" || d.kind === "transport";
  const defaultsFor = (d) => {
    if (isOwn(d)) return { ...(d.kind === "map" ? DEFAULTS.map : d.kind === "convergence" ? DEFAULTS.convergence : DEFAULTS.transport) };
    const own = d.source === "modes" ? { ...DEFAULTS.modes, shape: H.lineShape ? H.lineShape(d.kind) : "lorentzian" } : isSpec(d) ? DEFAULTS.spectrum : DEFAULTS.bands;
    return { ...own, ylog: false, xlog: false, xr: null, yr: null,
      unit: isSpec(d) ? "cm-1" : setting("energyUnit", "eV"), width: Number(setting("plots.lineWidth", 1.4)) || 1.4 };
  };
  const key = (d) => `mace.gui.plot.${H.dirHash()}.${d.path}${d.source ? `|${d.source}|${d.kind}` : ""}`;
  function load(d) {
    if (d.opts) return;
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(key(d)) || "null"); } catch (e) { saved = null; }
    d.opts = { ...defaultsFor(d), ...((saved && saved.opts) || {}) };
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
  const PAL = () => Array.from({ length: 10 }, (_, k) => cssVar(`--plot-${k + 1}`));
  const kLabel = (l) => (l === "G" ? "Γ" : String(l || "").replace(/(^|\|)G(?=$|\|)/g, "$1Γ").replace(/GAMMA/g, "Γ"));

  function dosRange(dos, lo, hi, opts) {
    let m = 0;
    dos.energy.forEach((e, i) => {
      if (e < lo || e > hi) return;
      dos.projections.forEach((p) => {
        if ((opts.hidden || []).includes(p.label)) return;
        if (opts.spin !== "down") m = Math.max(m, Math.abs(p.up[i] || 0));
        if (dos.spin_polarized && p.down && opts.spin !== "up") m = Math.max(m, Math.abs(p.down[i] || 0));
      });
    });
    m = m > 0 ? m * 1.08 : 1;
    return dos.spin_polarized && opts.spin === "both" ? [-m, m] : opts.spin === "down" ? [-m, 0] : [0, m];
  }
  function joinBands(x, bands, shift) {
    const xs = [], ys = [], idx = [];
    bands.forEach((b, k) => { for (let i = 0; i < x.length; i++) { xs.push(x[i]); ys.push(b[i] === null ? null : b[i] - shift); idx.push(k); } xs.push(null); ys.push(null); idx.push(-1); });
    return { xs, ys, idx };
  }
  /* k-path tick labels that never overlap: ticks at the same k (a writer that repeats a break's
     point) become one "A|B" label; a label too close to its neighbor for both to fit (seekpath's
     Y and F₀ 0.04 Å⁻¹ apart) drops to a second row instead of overlapping or turning the axis's
     labels sideways. Widths are estimated from the text at the axis font size. */
  const TICK_FONT = 12;
  function labelWidth(html) {
    const sub = (html.match(/<sub>(.*?)<\/sub>/g) || []).join("").replace(/<[^>]+>/g, "").length;
    const main = html.replace(/<sub>.*?<\/sub>/g, "").replace(/<[^>]+>/g, "").length;
    return (main * 0.62 + sub * 0.45) * TICK_FONT + 6;
  }
  function kTicks(ticks, labels, range, widthPx) {
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
      return row ? `\u00a0<br>${lab}` : lab;     // plotly drops an empty first line: a no-break space holds it
    });
    return { vals, text: out, rows };
  }
  function bandTickLayout(band, range, dos, r = el._fullLayout && el._fullLayout.margin ? el._fullLayout.margin.r : 14) {
    const w = Math.max(200, ((el.clientWidth || 900) - 58 - r) * (dos ? 0.7 : 1));
    const t = kTicks(band.ticks, band.labels, range, w);
    return { "xaxis.tickvals": t.vals, "xaxis.ticktext": t.text, "margin.b": 42 + (t.rows - 1) * 15 };
  }
  function bandFigure(d) {
    const { band, dos } = d.data, o = d.opts, css = cssVar, pal = PAL();
    // the dashed line marks the reference energy and stays where the reader put E_F; a Fermi shift s
    // moves the data by -s instead, so the axis reads E - (E_F + s)
    const ref = (band || dos).reference, fermi = (band || dos).fermi, shift = o.shift || 0, ef = fermi;
    // plotted as E - E_F in the chosen unit (plotkit): the reference line sits at 0
    const f = PK.ENERGY_UNITS[o.unit] || 1, unit = PK.UNIT_LABEL[o.unit] || "eV";
    const tf = (y) => (y === null ? null : (y - ef) * f);
    const logE = !!o.ylog && !dos;                     // a DOS plot: Log Y is the DOS axis
    const lg = (ys) => (logE ? ys.map((y) => (y === null || y === 0 ? null : Math.abs(y))) : ys);
    d.absShown = logE;
    const grid = css("--line"), traces = [], shapes = [];
    const yr = o.yr || [o.emin * f, o.emax * f];
    const lay = {
      // the legends sit outside the data, in the top and right margins that fitLegends sizes to the
      // legends as drawn: the bands' above the band panel, the DOS's right of the DOS panel (below)
      margin: { l: 58, r: 14, t: 10, b: 42, autoexpand: false }, showlegend: true, hovermode: "closest", dragmode: "zoom",
      legend: PK.legendTop(), paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: css("--text"), size: 12, family: css("--font-sans") },
      modebar: { bgcolor: "rgba(0,0,0,0)", color: css("--text-2"), activecolor: css("--accent") },
      yaxis: { title: { text: `${logE ? "|" : ""}${ref === "vacuum" ? "E − E<sub>vac</sub>" : "E − E<sub>F</sub>"}${logE ? "|" : ""} (${unit})` },
        type: logE ? "log" : "linear", range: logE ? undefined : yr, autorange: logE ? true : undefined,
        gridcolor: grid, zeroline: false, linecolor: grid, mirror: true, showline: true,
        showspikes: true, spikemode: "across", spikethickness: 1, spikecolor: css("--text-3"), spikedash: "dot" },
    };
    if (band) {
      lay.xaxis = { domain: dos ? [0, 0.7] : [0, 1],
        range: [band.ticks[0], band.ticks[band.ticks.length - 1]], showgrid: true, gridcolor: css("--border-strong"), zeroline: false,
        showline: true, mirror: true, linecolor: grid, showspikes: true, spikemode: "across", spikethickness: 1,
        spikecolor: css("--text-3"), spikedash: "dot", tickangle: 0 };
      const accent = css("--accent");
      if (o.xr) lay.xaxis.range = o.xr;
      const tl = bandTickLayout(band, lay.xaxis.range, dos);
      lay.xaxis.tickvals = tl["xaxis.tickvals"]; lay.xaxis.ticktext = tl["xaxis.ticktext"]; lay.margin.b = tl["margin.b"];
      // beta under alpha: where the two coincide alpha stays readable (dotted copies of degenerate
      // beta bands, out of phase, would otherwise paint one solid line over it); a non-magnetic run
      // (alpha = beta within 10 meV) comes as one set and is drawn once whichever spin is chosen
      if (band.spin_polarized && band.down && o.spin !== "up") {
        const dn = joinBands(band.x, band.down, shift);
        dn.ys = lg(dn.ys.map(tf));
        traces.push({ x: dn.xs, y: dn.ys, customdata: dn.idx, xaxis: "x", yaxis: "y", mode: "lines", type: "scatter",
          name: "spin down (β)", meta: "down", legendrank: 2, line: { color: pal[0], width: Math.max(0.6, o.width - 0.2), dash: "dot" }, hoverinfo: "none" });
      }
      if (o.spin !== "down" || !band.spin_polarized) {
        const up = joinBands(band.x, band.up, shift);
        up.ys = lg(up.ys.map(tf));
        traces.push({ x: up.xs, y: up.ys, customdata: up.idx, xaxis: "x", yaxis: "y", mode: "lines", type: "scatter",
          name: band.spin_polarized ? "spin up (α)" : band.spin_identical ? "α = β (non-magnetic)" : "bands",
          showlegend: !!(band.spin_polarized || band.spin_identical), meta: "up", legendrank: 1,
          line: { color: accent, width: o.width }, hoverinfo: "none" });
      }
      if (!logE) shapes.push({ type: "line", xref: "x domain", x0: 0, x1: 1, yref: "y", y0: 0, y1: 0, editable: true, name: "fermi",
        line: { color: css("--text-2"), width: 1.4, dash: "dash" } });
    }
    if (dos) {
      const ax = band ? "xaxis2" : "xaxis", xr = band ? "x2" : "x";
      const dlog = !!o.ylog;
      lay[ax] = { domain: band ? [0.73, 1] : [0, 1], title: { text: dlog ? "|DOS| (states/eV, log)" : "DOS (states/eV)" }, gridcolor: grid,
        zeroline: true, zerolinecolor: grid, showline: true, mirror: true, linecolor: grid,
        type: dlog ? "log" : "linear", range: dlog ? undefined : dosRange(dos, ef + yr[0] / f, ef + yr[1] / f, o), autorange: dlog ? true : undefined };
      if (dlog) d.absShown = true;
      if (!band && o.xr) lay[ax].range = o.xr;
      // the DOS legend: a column right of the DOS panel, its own legend beside the bands' one
      const dl = band ? "legend2" : "legend", spin2 = dos.spin_polarized;
      lay[dl] = PK.legendSide();
      // a legend-only entry (no data): a projection named once for both spins, or the spin key
      const entry = (name, line, more) => ({ x: [null], y: [null], xaxis: xr, yaxis: "y", mode: "lines", type: "scatter",
        name, legend: dl, line, hoverinfo: "skip", ...more });
      dos.projections.forEach((p, k) => {
        if ((o.hidden || []).includes(p.label)) return;
        const total = p.label === "Total";
        const col = total ? css("--text") : pal[(k + 1) % pal.length];
        const ys = dos.energy.map((e) => tf(e - shift));
        const dx = (xs) => (o.ylog ? xs.map((v) => (v === null || v === 0 ? null : Math.abs(v))) : xs);
        const lw = total ? o.width + 0.2 : o.width - 0.2;
        // spin-polarized: the projection is named once (one group: a click hides both spins) and
        // the spin key below says which line is which
        if (spin2) traces.push(entry(p.label, { color: col, width: lw }, { meta: "dos-key", legendgroup: p.label, legendrank: 10 + k,
          fill: total ? "tozerox" : "none", fillcolor: total ? "rgba(128,128,128,0.12)" : undefined }));
        if (o.spin !== "down") traces.push({ x: dx(p.up), y: ys, xaxis: xr, yaxis: "y", mode: "lines", type: "scatter", meta: "dos",
          name: spin2 ? `${p.label} ↑` : p.label, legendgroup: p.label, legend: dl, showlegend: !spin2, legendrank: 10 + k,
          line: { color: col, width: lw },
          fill: total ? "tozerox" : "none", fillcolor: total ? "rgba(128,128,128,0.12)" : undefined, hoverinfo: "none" });
        // spin down mirrors spin up: the total is filled as well, lighter and hatched so the two sides read apart
        if (spin2 && p.down && o.spin !== "up") traces.push({ x: dx(p.down), y: ys, xaxis: xr, yaxis: "y", mode: "lines", type: "scatter",
          meta: "dos", name: `${p.label} ↓`, legendgroup: p.label, legend: dl, showlegend: false,
          line: { color: col, width: lw, dash: "dot" },
          fill: total ? "tozerox" : "none", fillcolor: total ? "rgba(128,128,128,0.06)" : undefined,
          fillpattern: total ? { shape: "/", fgcolor: "rgba(128,128,128,0.45)", size: 7, solidity: 0.15 } : undefined, hoverinfo: "none" });
      });
      if (spin2) {
        const key = { color: css("--text-2"), width: 1.4 };
        if (o.spin !== "down") traces.push(entry("↑ solid, right", key, { meta: "key", legendrank: 1000 }));
        if (o.spin !== "up") traces.push(entry("↓ dotted, left", { ...key, dash: "dot" }, { meta: "key", legendrank: 1001 }));
      }
      if (!logE) shapes.push({ type: "line", xref: `${xr} domain`, x0: 0, x1: 1, yref: "y", y0: 0, y1: 0, editable: !band, name: "fermi2",
        line: { color: css("--text-2"), width: 1.4, dash: "dash" } });
      // CRYSTAL computed the DOS between these energies only: shade beyond them on the DOS panel
      // (after the Fermi lines, whose shape index is what a drag reports)
      if (dos.window && !logE) {
        const lo = tf(dos.window[0] - shift), hi = tf(dos.window[1] - shift), span = Math.abs(yr[1] - yr[0]) || 1;
        const rect = (y0, y1, name) => ({ type: "rect", layer: "below", xref: `${xr} domain`, x0: 0, x1: 1, yref: "y", y0, y1, name,
          fillcolor: css("--line"), opacity: 0.45, line: { width: 0 } });
        // one view past the visible range, not to infinity: an autorange (double-click) stays sane
        shapes.push(rect(Math.min(yr[0], lo) - span, lo, "nodos-below"), rect(hi, Math.max(yr[1], hi) + span, "nodos-above"));
      }
    }
    lay.shapes = shapes;
    return { traces, layout: lay, legendBase: { t: lay.margin.t, r: lay.margin.r } };
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
    const o = d.opts, css = cssVar, pal = PAL(), grid = css("--line");
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
      legend: PK.legendSide(), paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
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
    const m = d.data.map.maps[Math.min(d.opts.which || 0, d.data.map.maps.length - 1)], o = d.opts, css = cssVar;
    const positive = m.min > 0, log = o.log && positive;
    const z = m.values.map((col) => col.map((v) => (log ? Math.log10(v) : v)));
    const dist = (p, r) => Math.hypot(p[0] - r[0], p[1] - r[1], p[2] - r[2]);
    const ab = dist(m.A, m.B), bc = dist(m.B, m.C), sin = Math.sqrt(Math.max(0, 1 - m.cosxy * m.cosxy));
    // each corner's label sits inside the plane, on the side of its corner away from the edges and tick labels
    const cx = (ab + ab + bc * m.cosxy) / 3, cy = bc * sin / 3;
    const corner = (t, x, y) => ({ x, y, text: t, showarrow: false, font: { size: 13, color: css("--text") },
      xanchor: x < cx - 1e-6 ? "left" : x > cx + 1e-6 ? "right" : "center", yanchor: y < cy ? "bottom" : "top",
      bgcolor: css("--surface"), opacity: 0.85 });
    const signed = m.min < 0 && m.max > 0;
    const trace = { type: "contour", x: m.x, y: m.y, z, ncontours: o.contours, colorscale: signed ? "RdBu" : "Viridis", reversescale: signed,
      contours: { coloring: "heatmap", showlines: true }, line: { width: 0.6 },
      colorbar: { title: { text: log ? `log₁₀ (${m.unit || ""})` : m.unit || "" }, thickness: 12 },
      hovertemplate: `%{x:.3f}, %{y:.3f} Å: ${log ? "10^%{z:.3f}" : "%{z:.4g}"}<extra>${esc(m.label)}</extra>` };
    return { traces: [trace], layout: { margin: { l: 56, r: 14, t: 10, b: 44 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: css("--text"), size: 12, family: css("--font-sans") },
      xaxis: { title: { text: "along AB (Å)" }, scaleanchor: "y", constrain: "domain", showgrid: false },
      yaxis: { title: { text: "toward C (Å)" }, showgrid: false },
      annotations: [corner("A", 0, 0), corner("B", ab, 0), corner("C", ab + bc * m.cosxy, bc * sin)] } };
  }

  /* Compare Convergence: each run's SCF energy per atom at each geometry, above the lowest final energy of the set */
  const HARTREE_MEV = 27211.386245988;
  function convergenceFigure(d) {
    const runs = d.data.convergence.runs, o = d.opts, css = cssVar, pal = PAL(), grid = css("--line");
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
      hovermode: "closest", paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: css("--text"), size: 12, family: css("--font-sans") },
      xaxis: { title: { text: "Geometry (initial, optimization points, final)" }, gridcolor: grid, showline: true, mirror: true, linecolor: grid, dtick: 1 },
      yaxis: { title: { text: "ΔE per atom (meV)" }, type: o.log ? "log" : "linear", gridcolor: grid, showline: true, mirror: true, linecolor: grid } } };
  }

  /* BOLTZTRA data: one quantity per file against the chemical potential, one temperature at a time */
  function transportFigure(d) {
    const t = d.data.transport, o = d.opts, css = cssVar, pal = PAL(), grid = css("--line");
    const block = t.blocks.find((b) => b.T === o.T) || t.blocks[0];
    const comps = o.component === "all" ? ["xx", "yy", "zz"].filter((k) => block[k]) : [block[o.component] ? o.component : "avg"];
    const traces = comps.map((c, k) => ({ x: block.mu, y: block[c], mode: "lines", type: "scatter",
      name: c === "avg" ? "(xx + yy + zz) / 3" : c, line: { color: pal[k % pal.length], width: o.width },
      hovertemplate: `%{x:.3f} eV: %{y:.4g} ${esc(t.unit)}<extra>${esc(c)}</extra>` }));
    const lay = { margin: { l: 70, r: 14, t: 10, b: 44, autoexpand: false }, showlegend: traces.length > 1, hovermode: "x unified", legend: PK.legendTop(),
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: css("--text"), size: 12, family: css("--font-sans") },
      xaxis: { title: { text: t.x === "energy" ? "Energy (eV)" : "Chemical potential μ (eV)" }, gridcolor: grid, showline: true, mirror: true, linecolor: grid, zeroline: true, zerolinecolor: grid },
      yaxis: { title: { text: `${t.label} (${t.unit})` }, gridcolor: grid, showline: true, mirror: true, linecolor: grid, zeroline: true, zerolinecolor: grid } };
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
    if (building) { again = true; return; }        // one figure at a time: the latest options win
    building = true;
    el.dataset.drawn = "";
    try {
      await loadPlotly();
      if (!window.Plotly) throw new Error("plotly is not installed in this Python; pip install plotly");
      const fig = doc.kind === "map" ? mapFigure(doc) : doc.kind === "convergence" ? convergenceFigure(doc) : doc.kind === "transport" ? transportFigure(doc) : doc.kind === "ir" || doc.kind === "raman" ? spectrumFigure(doc) : bandFigure(doc);
      // another document, or other subplots (a band plot after a band+DOS one): a new figure, not an update;
      // plotly's react cannot add a second x axis to an existing figure
      const sig = `${doc.path}|${Object.keys(fig.layout).filter((k) => /^[xy]axis/.test(k)).sort().join(",")}`;
      const config = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"], edits: { shapePosition: true } };
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
    if (doc.kind !== "ir" && doc.kind !== "raman" && o.shift) parts.push(`<span class="pv-chip" title="Fermi shift (drag the dashed line; Plot options)">E<sub>F</sub> ${o.shift > 0 ? "+" : "−"} ${Math.abs(o.shift).toFixed(2)} eV</span>`);
    if (doc.absShown) parts.push(`<span class="pv-chip" title="A log axis cannot show zero or negative values">|value| shown</span>`);
    c.innerHTML = parts.join("");
  }

  /* hover readout, the Fermi-line drag, the zoom that keeps the DOS axis fitted */
  function bind() {
    if (el.__maceBound) return;
    el.__maceBound = true;
    // the DOS spin key only explains the line styles: a click on it hides nothing
    el.on("plotly_legendclick", PK.keyClick);
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
        $(".pv-readout").textContent = `E − E_F = ${Number(e).toFixed(3)} ${PK.UNIT_LABEL[doc.opts.unit] || "eV"} · ${vals.join(" · ")}`;
        return;
      }
      const band = doc.data.band;
      const x = p.x, e = p.y, k = t.customdata ? t.customdata[p.pointIndex ?? p.pointNumber] : null;
      let kl = "";
      const at = band.ticks.findIndex((tk) => Math.abs(tk - x) < 1e-6);
      if (at >= 0) kl = kLabel(band.labels[at]);
      else {
        const i = band.ticks.findIndex((tk) => tk > x);
        if (i > 0) kl = `${kLabel(band.labels[i - 1])}→${kLabel(band.labels[i])} ${Math.round(((x - band.ticks[i - 1]) / (band.ticks[i] - band.ticks[i - 1])) * 100)}%`;
      }
      $(".pv-readout").textContent = `k: ${kl || "—"} · path ${Number(x).toFixed(4)} · E − E_F = ${Number(e).toFixed(3)} ${PK.UNIT_LABEL[doc.opts.unit] || "eV"}` +
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
      if (sy !== undefined && doc.data.band) {
        // the Fermi line was dragged: the shift is where it went, relative to the reader's E_F
        // the energy now under the line becomes the reference: s' = y + s - E_F
        // the line is drawn at 0 in the plotted unit: the shift grows by where it went (in eV)
        const ns = Math.round(((doc.opts.shift || 0) + ev[sy] / (PK.ENERGY_UNITS[doc.opts.unit] || 1)) * 1000) / 1000 || 0;
        setOpt(doc, { shift: ns }, `Fermi shift ${ns.toFixed(2)} eV`, "drag");
        return;
      }
      // a zoom along k: lay the tick labels out again for the new range (the rows depend on it)
      if (doc.data.band && el.layout.xaxis && Object.keys(ev).some((k) => k === "xaxis.autorange" || k.startsWith("xaxis.range"))) {
        const r = el.layout.xaxis.range, tk = doc.data.band.ticks;
        window.Plotly.relayout(el, bandTickLayout(doc.data.band, r || [tk[0], tk[tk.length - 1]], doc.data.dos));
      }
      if (doc.data.dos && Object.keys(ev).some((k) => k.startsWith("yaxis."))) {
        if (doc.opts.ylog || !el.layout.yaxis.range) return;
        const fu = PK.ENERGY_UNITS[doc.opts.unit] || 1, ef = doc.data.dos.fermi;
        const [lo, hi] = el.layout.yaxis.range.map((v) => ef + v / fu);
        const ax = el.layout.xaxis2 ? "xaxis2" : "xaxis";
        window.Plotly.relayout(el, { [`${ax}.range`]: dosRange(doc.data.dos, Math.min(lo, hi), Math.max(lo, hi), doc.opts) });
      }
    });
  }
  el.addEventListener("keydown", (e) => {
    if (!doc || !(e.ctrlKey || e.metaKey) || e.key.toLowerCase() !== "z") return;
    e.preventDefault(); e.stopPropagation();
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
  function renderOptionsNow() {
    if (!doc) { optsHost.innerHTML = ""; return; }
    const o = doc.opts, sp = doc.kind === "ir" || doc.kind === "raman";
    let h = "";
    if (doc.kind === "map") {
      const M = doc.data && doc.data.map;
      if (M && M.maps.length > 1) h += `<label class="po-row" for="po-mwhich"><span>Map</span><select id="po-mwhich">${M.maps.map((m, k) =>
        `<option value="${k}"${(o.which || 0) === k ? " selected" : ""}>${esc(m.label)}</option>`).join("")}</select></label>`;
      h += `<label class="chk po-row"><input type="checkbox" id="po-mlog"${o.log ? " checked" : ""}> Logarithmic scale (positive maps)</label>` +
        num("po-mcont", "Contours", o.contours, "", "1");
      if (M) { const m = M.maps[o.which || 0]; h += `<p class="po-h">${esc(m.label)}: min ${m.min.toPrecision(4)} · max ${m.max.toPrecision(4)} (${esc(m.unit || M.unit)})</p>`; }
      optsHost.innerHTML = h;
      return;
    }
    if (doc.kind === "convergence") {
      h += num("po-width", "Line width", o.width, "px", "0.2") +
        `<label class="chk po-row"><input type="checkbox" id="po-clog"${o.log ? " checked" : ""}> Logarithmic energy axis</label>` +
        `<p class="po-h">Each run's energy per atom at each printed geometry; molecules print only the initial and final ones.</p>`;
      optsHost.innerHTML = h;
      return;
    }
    if (doc.kind === "transport") {
      const t = doc.data && doc.data.transport;
      if (t) {
        const Q = { SEEBECK: "Seebeck", SIGMA: "σ", SIGMAS: "σS", KAPPA: "κ₀", TDF: "TDF" };
        h += `<div class="po-row"><span>Quantity</span><span class="segment po-tq" role="radiogroup" aria-label="Quantity">${Object.keys(Q).filter((k) => t.siblings[k]).map((k) =>
          `<button role="radio" aria-checked="${k === t.quantity}" data-tq="${esc(t.siblings[k])}">${Q[k]}</button>`).join("")}</span></div>`;
        if (t.blocks.length > 1 || t.blocks[0].T !== null) h += `<label class="po-row" for="po-temp"><span>Temperature</span><select id="po-temp">${t.blocks.map((b) =>
          `<option value="${b.T}"${(o.T ?? t.blocks[0].T) === b.T ? " selected" : ""}>${b.T} K</option>`).join("")}</select></label>`;
        h += `<label class="po-row" for="po-tcomp"><span>Component</span><select id="po-tcomp">${[["avg", "Average of the diagonal"], ["xx", "xx"], ["yy", "yy"], ["zz", "zz"], ["all", "xx, yy, zz"]].map(([v, l]) =>
          `<option value="${v}"${o.component === v ? " selected" : ""}>${l}</option>`).join("")}</select></label>`;
        h += num("po-width", "Line width", o.width, "px", "0.2");
      }
      h += `<div class="po-acts"><button class="tb-btn-text" data-po="reset">${icon("rotate-ccw")}<span>Reset Plot Options</span></button></div>`;
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
      if (dos) h += `<div class="po-proj"><div class="po-h">Projections</div>${dos.projections.map((p) =>
        `<label class="chk"><input type="checkbox" data-proj="${esc(p.label)}"${(o.hidden || []).includes(p.label) ? "" : " checked"}> ${esc(p.label)}</label>`).join("")}</div>`;
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
    h += `<div class="po-acts"><button class="tb-btn-text" data-po="reset"${doc.pos > 0 || JSON.stringify(o) !== JSON.stringify(defaultsFor(doc)) ? "" : " disabled"}>${icon("rotate-ccw")}<span>Reset Plot Options</span></button></div>`;
    optsHost.innerHTML = h;
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
    else if (t.id === "po-temp") setOpt(doc, { T: Number(t.value) }, `Temperature ${t.value} K`);
    else if (t.id === "po-mlog") setOpt(doc, { log: t.checked }, t.checked ? "Logarithmic scale" : "Linear scale");
    else if (t.id === "po-mcont") setOpt(doc, { contours: Math.max(2, Math.min(60, Number(t.value) || 20)) }, `Contours ${t.value}`);
    else if (t.id === "po-mwhich") setOpt(doc, { which: Number(t.value) }, "Map");
    else if (t.id === "po-clog") setOpt(doc, { log: t.checked }, t.checked ? "Logarithmic axis" : "Linear axis");
    else if (t.id === "po-tcomp") setOpt(doc, { component: t.value }, `Component ${t.value}`);
    else if (t.id === "po-fwhm" || t.id === "po-shape" || t.id === "po-eta") {
      const d = doc, patch = t.id === "po-fwhm" ? { fwhm: Math.min(50, Math.max(2, Number(t.value) || 8)) }
        : t.id === "po-shape" ? { shape: t.value } : { eta: Number(t.value), shape: Number(t.value) < 1 ? "pseudo-voigt" : "lorentzian" };
      Object.assign(d.opts, patch);
      d.data = null;
      fetchData(d).then(() => { if (doc === d) { setOpt(d, patch, t.id === "po-fwhm" ? `Peak width ${patch.fwhm} cm⁻¹` : `Line shape ${d.opts.shape}`); note(d); } })
        .catch((err) => toast(err.message, "error"));
    }
    else if (t.dataset.proj) {
      const hidden = new Set(o.hidden || []);
      if (t.checked) hidden.delete(t.dataset.proj); else hidden.add(t.dataset.proj);
      setOpt(doc, { hidden: [...hidden] }, `${t.checked ? "Show" : "Hide"} ${t.dataset.proj}`);
    }
  });
  optsHost.addEventListener("click", (e) => {
    const tq = e.target.closest("[data-tq]");
    if (tq && doc && tq.getAttribute("aria-checked") !== "true" && H.openPath) { H.openPath(tq.dataset.tq); return; }
    const s = e.target.closest("[data-spin]");
    if (s && doc) { setOpt(doc, { spin: s.dataset.spin }, `Spin ${s.dataset.spin}`); return; }
    const b = e.target.closest("[data-po]");
    if (b && doc && b.dataset.po === "reset") { reset(doc); toast("Plot options reset", "info", { label: "Undo", run: () => undo(doc) }); }
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
      if (doc && isOwn(doc)) return { logY: false, logX: false, units: null, range: false, width: false, png: true, svg: true, image: true, csv: true, pub: false };
      return { logY: true, logX: !!doc && isSpec(doc), units: doc && isSpec(doc) ? "wavenumber" : "energy", range: true, width: true,
        png: true, svg: true, image: true, csv: true, pub: !(doc && doc.source === "modes") };
    },
    get: () => (doc ? doc.opts : {}),
    set: (patch, label) => doc && setOpt(doc, patch, label),
    axes: () => (doc && isSpec(doc) ? { x: "Wavenumber", y: "Intensity" } : { x: "k-path / DOS", y: "Energy" }),
    table: () => tableOf(doc),
    png: () => image("png"), svg: () => image("svg"), image: () => exportImage(),
    pub: () => publication(),
    resetZoom: () => { if (window.Plotly && el.layout) { el.__sig = null; redraw(); } },
  }, { toast, download, icon });
  async function publication() {
    const r = await apiTask("plot?" + q({ path: doc.path, format: "svg", detach: 1 }), undefined, null, { get: true });
    const svg = r.images.find((n) => n.endsWith(".svg"));
    if (!svg) throw new Error("mace plotting made no SVG: " + (r.log || "").slice(-300));
    const res = await api("plot-image?" + q({ key: r.key, name: svg }), { blob: true });
    download(res.blob, svg);
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
    const { band, dos } = d.data, shift = d.opts.shift || 0, f = PK.ENERGY_UNITS[d.opts.unit] || 1, u = d.opts.unit || "eV";
    const ef = (band || dos).fermi;
    if (band) {
      const cols = ["path", ...band.up.map((_, k) => `band${k + 1}${band.spin_polarized ? "_up" : ""}_${u}`), ...(band.down || []).map((_, k) => `band${k + 1}_down_${u}`)];
      band.x.forEach((x, i) => { if (x !== null) rows.push([x, ...band.up.map((b) => (b[i] - shift - ef) * f), ...(band.down || []).map((b) => (b[i] - shift - ef) * f)]); });
      if (!dos) return { columns: cols, rows };
      return { columns: cols, rows };
    }
    const cols = [`energy_${u}`, ...dos.projections.flatMap((p) => (dos.spin_polarized ? [`${p.label}_up`, `${p.label}_down`] : [p.label]))];
    dos.energy.forEach((e, i) => rows.push([(e - shift - ef) * f, ...dos.projections.flatMap((p) => (dos.spin_polarized ? [p.up[i], p.down[i]] : [p.up[i]]))]));
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
          notes.push(band ? (band.spin_polarized ? "spin-polarized: α solid, β dotted"
            : band.spin_identical ? `spin-polarized · α = β (non-magnetic, ≤${meV} meV)` : "closed shell")
            : dos.spin_polarized ? "spin-polarized" : "closed shell");
          if (band && band.truncated) notes.push(`the ${band.up.length} bands nearest E<sub>F</sub> of ${band.n_bands}`);
          notes.push((band || dos).reference === "vacuum" ? "vacuum scale · dashed = E<sub>F</sub>, drag to shift" : "dashed = Fermi level, drag to shift");
          if (band && band.spin_identical) help.push(`α and β agree within ${meV} meV, so one set of bands is drawn`);
          if (dos && dos.spin_polarized) help.push("DOS: ↑ to the right, ↓ mirrored to the left (total hatched)");
          if (dos && dos.window) help.push(`shaded: outside the computed DOS window${dos.edge_pileup ? " (its edge bin, the states beyond it, is left out)" : ""}`);
          if ((band || dos).reference === "vacuum") help.push("energies on the vacuum scale (POTC run found); the dashed line is E_F");
          help.push("dragging the dashed line shifts the energy reference");
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
    state: () => doc && { kind: doc.kind, source: doc.source || "file", opts: { ...doc.opts }, pos: doc.pos, hist: doc.hist.length, data: !!doc.data,
      drawn: el.dataset.drawn === doc.path && !building, peaks: doc.data && doc.data.peaks ? doc.data.peaks.length : 0,
      out: doc.data ? doc.data.out || null : null, fwhm: doc.data && doc.data.spectra ? doc.data.spectra[0].fwhm || null : null },
    figure: () => (el.data ? { traces: el.data.map((t) => ({ name: t.name, meta: t.meta, n: (t.x || []).length })), layout: el.layout } : null),
    setOpt: (patch, label) => doc && setOpt(doc, patch, label),
    el: () => el,
  };
}
