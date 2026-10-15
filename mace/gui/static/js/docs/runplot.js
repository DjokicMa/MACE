/* Run Analysis (PLAN_analysis C4): one optimization's energy, convergence,
 * SCF, cell and tracked measurements, drawn by plotly, every x axis in
 * CRYSTAL's POINT numbers (as the .out numbers them).
 *
 * Frames and points (D1a): frame k is POINT k+1; the final frame (the
 * converged geometry after OPT END) sits at the last POINT, drawn as a ring.
 * frameOfPoint here is the twin of out_convergence.frame_of_point (it takes
 * the mapping the server computed with outindex's frame lines).
 * Energy surfaces (D6): a FINALRUN restart re-references the integrals.
 * Segments "continuous" (the default) draws one curve with one reference
 * across the restarts (dotted lines mark them); "separate" takes the
 * relative energies within each segment. The twin of
 * mace.utils.out_convergence.energy_along_run (`mace analyze --convergence
 * FILE --energy [--segments separate]`).
 * The options are plotkit's (units, per, relative to, log, range, width),
 * kept per document; the defaults come from Settings › Plots. */
import * as PK from "../plotkit.js";

const TABS = [["energy", "Energy"], ["convergence", "Convergence"], ["scf", "SCF"], ["cell", "Cell"], ["tracked", "Tracked"]];
const CRIT = [["max_grad", "Max gradient", "maxgrad"], ["rms_grad", "RMS gradient", "toldeg"], ["max_disp", "Max displacement", "maxdisp"],
  ["rms_disp", "RMS displacement", "toldex"]];
const XT = "Optimization point (as numbered in the .out)";
const REASON = {
  all_criteria: "All criteria met", gradient_and_energy: "Gradient + energy change met (manual p.208); displacements not",
  energy_stalled: "Energy unchanged for 4 points (manual p.208)", gradient_first_cycle: "Gradients met at the first point",
  maxcycle: "MAXCYCLE reached", failed: "OPT END - FAILED", unfinished: "Unfinished", unknown: "Ended",
};
const SRC = { out: "from the .out", d12: "from the .d12", default: "CRYSTAL23 default" };

/** POINT -> frame, from the server's points (each carries its frame). */
export function frameOfPoint(points) {
  const m = {};
  for (const p of points || []) if (p.frame !== null && p.frame !== undefined) m[p.point] = p.frame;
  return m;
}

export function createRunPlot(host, H) {
  const { api, q, esc, icon, toast, cssVar, loadPlotly, download } = H;
  host.innerHTML = `<div class="rp pv" tabindex="-1">
    <div class="pv-head"><div class="pv-title"></div><div class="pv-chips"></div><div class="pv-acts"></div></div>
    <div class="rp-tabs" role="tablist" aria-label="Run Analysis">${TABS.map(([k, t], i) =>
      `<button role="tab" class="chip${i ? "" : " active"}" data-tab="${k}" aria-selected="${!i}" tabindex="${i ? -1 : 0}">${t}</button>`).join("")}
      <span class="rp-tabopts"></span></div>
    <div class="pv-note muted small"></div>
    <div class="pv-plot rp-plot" tabindex="0" aria-label="Run Analysis plot"></div>
    <div class="rp-below"></div></div>`;
  const $ = (s) => host.querySelector(s);
  const el = $(".rp-plot");
  let doc = null, kit = null, building = false, again = false;
  let legendBase = null;          // the margins without legends (fitLegends)

  /* ------------------------------------------------ options (per document) */
  const setting = (k, d) => { try { const v = H.setting ? H.setting(k) : undefined; return v === undefined || v === null || v === "" ? d : v; } catch (e) { return d; } };
  const key = (d) => `mace.gui.runplot.${H.dirHash ? H.dirHash() : ""}.${d.runPath}${d.compare ? "|" + d.compare : ""}`;
  function defaults(d) {
    const single = !d.compare;
    return {
      energy: { unit: setting("energyUnit", "eV") === "eV" ? "meV" : setting("energyUnit", "eV"), ylog: !!setting("plots.logDeltaE", false),
        relTo: d.compare ? "final" : setting("plots.relTo", "final"), per: d.compare ? null : setting("plots.per", "cell"),
        segments: setting("plots.segments", "continuous") === "separate" ? "separate" : "continuous",
        width: Number(setting("plots.lineWidth", 1.5)) || 1.5, xr: null, yr: null, xlog: false },
      convergence: { ylog: true, width: Number(setting("plots.lineWidth", 1.5)) || 1.5, xr: null, yr: null },
      scf: { ylog: true, width: 1.5, xr: null, yr: null },
      cell: { relTo: "absolute", width: 1.5, xr: null, yr: null, ylog: false },
      tracked: { width: 1.5, xr: null, yr: null, ylog: false }, _single: single,
    };
  }
  function load(d) {
    if (d.opts) return;
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(key(d)) || "null"); } catch (e) { saved = null; }
    const def = defaults(d);
    d.opts = {};
    for (const t of Object.keys(def)) d.opts[t] = t.startsWith("_") ? def[t] : { ...def[t], ...((saved && saved[t]) || {}) };
    // the old per-segment checkbox (single: false was saved with every plot) gives way to Segments
    if (saved && saved.energy && !saved.energy.segments) d.opts.energy.segments = def.energy.segments;
    delete d.opts.energy.single;
    d.tab = d.tab || (saved && saved.tab) || "energy";
    d.tracks = d.tracks || (saved && saved.tracks) || [];
  }
  function persist(d) {
    try { localStorage.setItem(key(d), JSON.stringify({ ...d.opts, tab: d.tab, tracks: d.tracks.map((t) => ({ ...t, values: undefined })) })); } catch (e) { /* private */ }
  }
  const O = () => doc.opts[doc.tab];
  function takeEnergy(d) {
    if (!d.energyPatch || !d.opts) return;
    Object.assign(d.opts.energy, d.energyPatch);
    delete d.energyPatch;
    persist(d);
  }
  function set(patch, label) {
    Object.assign(O(), patch);
    persist(doc);
    redraw();
    if (kit) kit.refresh();
  }

  /* ------------------------------------------------ data */
  async function runData(path) {
    const [conv, sum] = await Promise.all([api("out/convergence?" + q({ path })),
      api("out/summary?" + q({ path })).catch(() => ({ frames: [], energies_au: [], opt_energies_au: [] }))]);   // a molecule: no frames
    const t = sum.summary && H.expandSummary ? H.expandSummary(sum) : sum;
    const f0 = t.frames.find((x) => x) || {};
    const numbers = (f0.numbers || []).map((z) => z % 100);
    return { path, conv, summary: t, n: numbers.length, comp: composition(numbers), z: zOf(numbers),
      name: path.split("/").pop() };
  }
  function composition(nums) { const c = {}; for (const z of nums) c[z] = (c[z] || 0) + 1; return c; }
  function gcd(a, b) { return b ? gcd(b, a % b) : a; }
  function zOf(nums) { const c = Object.values(composition(nums)); return c.length ? c.reduce((a, b) => gcd(a, b)) : 1; }
  const sameComp = (a, b) => { const ka = Object.keys(a.comp), kb = Object.keys(b.comp); return ka.length === kb.length && ka.every((k) => a.comp[k] / a.n === (b.comp[k] || 0) / b.n); };

  /* the energies on the optimizer's surface, per point; the final frame's (OPT END) at the last point */
  function energies(r) {
    const pts = r.conv.points.map((p) => ({ x: p.point, frame: p.frame, seg: p.segment, e: p.energy_au, de: p.de_printed }));
    const t = r.summary, oe = (t.opt_energies_au || t.energies_au || []);
    const ff = r.conv.final_frame;
    const fin = ff !== null && ff !== undefined && oe[ff] !== null && oe[ff] !== undefined && pts.length
      ? { x: pts[pts.length - 1].x, frame: ff, seg: pts[pts.length - 1].seg, e: oe[ff], ring: true } : null;
    return { pts, fin };
  }
  const HA_EV = PK.HARTREE_EV;
  function perDiv(r, per) { return per === "atom" ? r.n || 1 : per === "fu" ? (r.n / (r.z || 1)) || 1 : 1; }

  /* ------------------------------------------------ figures */
  const css = (n) => cssVar(n);
  function base(o) {
    const grid = css("--line");
    // the legend above the plot, its margin sized to it as drawn (plotkit fitLegends); the labels of
    // reference lines (thresholds, restarts, the stop) are legend entries of their own (key below)
    return { margin: { l: 70, r: 18, t: 18, b: 46, autoexpand: false }, _keys: [], paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: css("--text"), size: 12, family: css("--font-sans") }, hovermode: "closest", showlegend: true,
      legend: PK.legendTop(),
      modebar: { bgcolor: "rgba(0,0,0,0)", color: css("--text-2"), activecolor: css("--accent") },
      xaxis: { title: { text: XT }, gridcolor: grid, showline: true, mirror: true, linecolor: grid, zeroline: false,
        type: o.xlog ? "log" : "linear", range: o.xr || undefined, dtick: undefined },
      yaxis: { gridcolor: grid, showline: true, mirror: true, linecolor: grid, zeroline: false, type: o.ylog ? "log" : "linear",
        range: o.yr || undefined } };
  }
  function selLine(r, lay, yref = "paper") {
    const fr = H.frameOf ? H.frameOf(r.path) : null;
    if (fr === null || fr === undefined) return;
    const pts = r.conv.points;
    let x = null;
    const p = pts.find((pp) => pp.frame === fr);
    if (p) x = p.point;
    else if (fr === r.conv.final_frame && pts.length) x = pts[pts.length - 1].point;
    if (x === null) return;
    lay.shapes.push({ type: "line", name: "selected", xref: "x", x0: x, x1: x, yref, y0: 0, y1: 1, line: { color: css("--sel"), width: 2 } });
  }
  /* a legend entry naming a reference line, band or mark drawn as a shape (no box over the data) */
  function refKey(lay, name, style) { lay._keys.push(PK.keyEntry(name, style, { legendrank: 900 + lay._keys.length })); }
  function segLines(r, lay, separate = false) {
    for (const s of r.conv.segments.slice(1)) {
      const x = s.first_point - 0.5;
      lay.shapes.push({ type: "line", name: "finalrun", xref: "x", x0: x, x1: x, yref: "paper", y0: 0, y1: 1, line: { color: css("--text-3"), width: 1, dash: "dot" } });
    }
    if (r.conv.segments.length > 1) refKey(lay, separate ? "FINALRUN restart (re-referenced)" : "FINALRUN restart", { line: { color: css("--text-3"), width: 1, dash: "dot" } });
  }

  function energyFigure(d) {
    const o = d.opts.energy, runs = [d.run, ...(d.other ? [d.other] : [])];
    const f = PK.ENERGY_UNITS[o.unit] || 1000, ul = PK.unitText(o.unit, o.per || "cell");
    const lay = base(o);
    lay.shapes = []; lay.annotations = [];
    const traces = [], pal = Array.from({ length: 6 }, (_, k) => css(`--plot-${k + 1}`));
    let below = 0;
    runs.forEach((r, ri) => {
      const { pts, fin } = energies(r);
      const div = perDiv(r, o.per);
      const dash = ri ? "dot" : "solid";
      const name = runs.length > 1 ? r.name : "";
      const sep = o.segments === "separate" && r.conv.segments.length > 1;
      if (o.ylog) {
        // CRYSTAL's DE per point, and the distance to the final energy (the run's, or each segment's)
        const dp = pts.filter((p) => p.de !== null && p.de !== undefined);
        traces.push({ x: dp.map((p) => p.x), y: dp.map((p) => Math.abs(p.de) * HA_EV * f / div), customdata: dp.map((p) => p.frame), mode: "lines+markers",
          type: "scatter", name: `${name ? name + ": " : ""}Energy change per point (CRYSTAL DE)`, meta: `de${ri}`,
          line: { color: pal[ri * 2 % 6], width: o.width, dash },
          marker: { size: 8, symbol: dp.map((p) => (p.de > 0 ? "triangle-up" : "circle")), color: dp.map((p) => (p.de > 0 ? css("--danger") : pal[ri * 2 % 6])) },
          hovertemplate: `POINT %{x}: |DE| %{y:.3e} ${esc(ul)}<extra></extra>` });
        const ds = [], segFinal = {}, g = (p) => (sep ? p.seg : 0);
        for (const p of pts) segFinal[g(p)] = p.e;
        if (fin) segFinal[g(fin)] = fin.e;
        // 1e-10 Ha: the printed precision (the OPT END energy repeats the last point's to 12 digits)
        for (const p of pts) {
          const v = Math.abs(p.e - segFinal[g(p)]);
          if (v > 1e-10) ds.push({ x: p.x, y: v * HA_EV * f / div, frame: p.frame, below: p.e < segFinal[g(p)] - 1e-10 });
        }
        below += ds.filter((p) => p.below).length;
        traces.push({ x: ds.map((p) => p.x), y: ds.map((p) => p.y), customdata: ds.map((p) => p.frame), mode: "lines+markers", type: "scatter",
          name: `${name ? name + ": " : ""}Distance to the ${sep ? "segment's" : "run's"} final energy`, meta: `seg${ri}`,
          line: { color: pal[(ri * 2 + 1) % 6], width: o.width, dash },
          marker: { size: 7, symbol: ds.map((p) => (p.below ? "triangle-down" : "circle")), color: ds.map((p) => (p.below ? css("--danger") : pal[(ri * 2 + 1) % 6])) },
          hovertemplate: `POINT %{x}: %{y:.3e} ${esc(ul)}<extra></extra>` });
        if (ri === 0) {
          const td = r.conv.thresholds.toldee_opt;
          if (td) {
            const y = td.value * HA_EV * f / div;
            lay.shapes.push({ type: "line", name: "toldee", xref: "paper", x0: 0, x1: 1, yref: "y", y0: y, y1: y, line: { color: css("--text-2"), width: 1, dash: "dash" } });
            refKey(lay, `TOLDEE ${td.value.toExponential(0)} Ha<br>(${SRC[td.source] || td.source})`, { line: { color: css("--text-2"), width: 1, dash: "dash" } });
          }
        }
        lay.yaxis.title = { text: `|ΔE| (${ul})` };
      } else {
        // linear: E - E_ref, one reference for the whole run (continuous) or one per FINALRUN segment (separate)
        const all = fin ? [...pts, fin] : pts;
        const ref = {};
        const segOf = (p) => (sep ? p.seg : 0);
        if (o.relTo === "first") for (const p of all) { if (!(segOf(p) in ref)) ref[segOf(p)] = p.e; }
        else if (o.relTo === "final") for (const p of all) ref[segOf(p)] = p.e;
        else if (o.relTo === "minimum") for (const p of all) ref[segOf(p)] = Math.min(ref[segOf(p)] ?? Infinity, p.e);
        else for (const p of all) ref[segOf(p)] = 0;
        const val = (p) => (p.e - ref[segOf(p)]) * HA_EV * f / div;
        const groups = {};
        for (const p of pts) (groups[segOf(p)] = groups[segOf(p)] || []).push(p);
        Object.entries(groups).forEach(([sg, gp], gi) => {
          const bl = gp.map((p) => o.relTo !== "absolute" && p.e < ref[segOf(p)] - 1e-10);
          below += bl.filter(Boolean).length;
          traces.push({ x: gp.map((p) => p.x), y: gp.map(val), customdata: gp.map((p) => p.frame), mode: "lines+markers", type: "scatter",
            name: `${name ? name + ": " : ""}${Object.keys(groups).length === 1 ? "E − E_ref" : `segment ${Number(sg) + 1}, relative to its ${o.relTo}`}`,
            meta: `e${ri}_${sg}`, legendgroup: `r${ri}`, showlegend: gi === 0 || runs.length === 1,
            line: { color: pal[ri * 2 % 6], width: o.width, dash },
            marker: { size: 8, symbol: bl.map((b) => (b ? "triangle-down" : "circle")), color: bl.map((b) => (b ? css("--danger") : pal[ri * 2 % 6])) },
            hovertemplate: `POINT %{x}: %{y:.4g} ${esc(ul)}<extra></extra>` });
        });
        if (fin) traces.push({ x: [fin.x], y: [val(fin)], customdata: [fin.frame], mode: "markers", type: "scatter", name: "final geometry (OPT END)",
          meta: `fin${ri}`, marker: { size: 13, symbol: "circle-open", color: pal[ri * 2 % 6], line: { width: 2 } }, hovertemplate: `final geometry: %{y:.4g} ${esc(ul)}<extra></extra>` });
        lay.yaxis.title = { text: `${o.relTo === "absolute" ? "E" : "ΔE"} (${ul})` };
      }
      if (ri === 0) segLines(r, lay, sep);
    });
    selLine(d.run, lay);
    d.belowCount = below;
    return { traces, layout: lay };
  }

  function convergenceFigure(d) {
    const o = d.opts.convergence, runs = [d.run, ...(d.other ? [d.other] : [])];
    const lay = base(o);
    lay.shapes = []; lay.annotations = [];
    const traces = [], pal = Array.from({ length: 6 }, (_, k) => css(`--plot-${k + 1}`));
    const r0 = d.run;
    const internal = r0.conv.gradient_units === "internal";
    runs.forEach((r, ri) => {
      const pts = r.conv.points;
      CRIT.forEach(([k, label, tk], ci) => {
        const ps = pts.filter((p) => p[k] !== null && p[k] !== undefined);
        if (!ps.length) return;
        const color = pal[ci % 6];
        traces.push({ x: ps.map((p) => p.point), y: ps.map((p) => p[k]), customdata: ps.map((p) => p.frame), mode: "lines+markers", type: "scatter",
          name: `${runs.length > 1 ? r.name + ": " : ""}${label}`, meta: `${k}${ri}`,
          line: { color, width: o.width, dash: ri ? "dot" : "solid" },
          marker: { size: 8, color, symbol: ps.map((p) => (p.converged[k] ? "circle" : "circle-open")), line: { color, width: 1.5 } },
          hovertemplate: `POINT %{x}: %{y:.6f}<extra>${esc(label)}</extra>` });
        if (ri === 0) {
          const t = r.conv.thresholds[tk];
          if (t) {
            lay.shapes.push({ type: "line", name: `thr-${k}`, xref: "paper", x0: 0, x1: 1, yref: "y", y0: t.value, y1: t.value, line: { color, width: 1, dash: "dash" } });
            refKey(lay, `${label} ${t.value}<br>(${SRC[t.source] || t.source}${t.manual_page ? ", manual p." + t.manual_page : ""})`, { line: { color, width: 1, dash: "dash" } });
          }
        }
      });
    });
    lay.yaxis.title = { text: internal ? "Gradient · displacement (internal a.u.)" : "Gradient (Ha/bohr) · displacement (bohr)" };
    const end = r0.conv.opt && r0.conv.opt.end;
    const pts = r0.conv.points;
    if (end && pts.length) {
      const x = pts[pts.length - 1].point;
      lay.shapes.push({ type: "rect", name: "stop", xref: "x", x0: x - 0.4, x1: x + 0.4, yref: "paper", y0: 0, y1: 1,
        fillcolor: "rgba(128,128,128,0.12)", line: { width: 0 } });
      refKey(lay, REASON[end.reason] || end.reason, { marker: { symbol: "square", size: 12, color: "rgba(128,128,128,0.35)" } });
    }
    segLines(r0, lay);
    selLine(r0, lay);
    return { traces, layout: lay };
  }

  function scfFigure(d) {
    const o = d.opts.scf, r = d.run, conv = r.conv;
    const lay = base(o);
    lay.shapes = []; lay.annotations = [];
    lay.grid = { rows: 2, columns: 1, pattern: "independent", roworder: "top to bottom" };
    const pal = Array.from({ length: 6 }, (_, k) => css(`--plot-${k + 1}`));
    const pts = conv.points;
    const traces = [];
    if (pts.length) {
      traces.push({ x: pts.map((p) => p.point), y: pts.map((p) => p.scf_cycles), customdata: pts.map((p) => p.frame), type: "bar", name: "SCF cycles per point",
        meta: "cycles", marker: { color: pal[0] }, xaxis: "x", yaxis: "y", hovertemplate: "POINT %{x}: %{y} cycles<extra></extra>" });
      lay.yaxis = { ...lay.yaxis, title: { text: "SCF cycles" }, type: "linear", domain: [0.58, 1] };
      lay.xaxis = { ...lay.xaxis, domain: [0, 1], anchor: "y", title: { text: "" } };
      const mc = conv.thresholds.scf_maxcycle;
      if (mc) {
        // the SCF MAXCYCLE line only when the bars come near it (1600 would flatten bars of 5)
        const top = Math.max(...pts.map((p) => p.scf_cycles || 0));
        if (mc.value <= 2 * top) lay.shapes.push({ type: "line", name: "scfmax", xref: "x domain", x0: 0, x1: 1, yref: "y", y0: mc.value, y1: mc.value, line: { color: css("--text-3"), dash: "dash", width: 1 } });
        refKey(lay, `SCF MAXCYCLE ${mc.value}<br>(${SRC[mc.source] || mc.source})`, { line: { color: css("--text-3"), width: 1, dash: "dash" } });
      }
    }
    // the per-cycle detail CRYSTAL printed: the selected point's block, else the first
    const fr = H.frameOf ? H.frameOf(r.path) : null;
    const selPoint = (pts.find((p) => p.frame === fr) || {}).point;
    const blocks = conv.scf.detail || [];
    const blk = blocks.find((b) => b.point === selPoint) || (selPoint === undefined || (pts.length === 0) ? blocks[0] : null);
    const grid = css("--line");
    lay.xaxis2 = { title: { text: "SCF cycle" }, domain: [0, 1], anchor: "y2", gridcolor: grid, showline: true, mirror: true, linecolor: grid };
    lay.yaxis2 = { title: { text: "|ΔE| (Ha)" }, type: "log", domain: [0, pts.length ? 0.42 : 1], anchor: "x2", gridcolor: grid, showline: true, mirror: true, linecolor: grid };
    if (!pts.length) { lay.yaxis = { visible: false }; lay.xaxis = { visible: false }; }
    d.scfNote = "";
    if (blk) {
      const cyc = blk.cycles.filter((c) => c.detot !== null);
      traces.push({ x: cyc.map((c) => c.cyc), y: cyc.map((c) => Math.abs(c.detot)), xaxis: "x2", yaxis: "y2", mode: "lines+markers", type: "scatter",
        name: `|ΔE| per cycle${blk.point ? ` (POINT ${blk.point})` : ""}`, meta: "detail",
        line: { color: blk.ended === "too_many_cycles" ? css("--danger") : pal[1], width: o.width }, hovertemplate: "cycle %{x}: %{y:.2e} Ha<extra></extra>" });
      const t = conv.thresholds.toldee_scf;
      if (t) {
        lay.shapes.push({ type: "line", name: "toldee-scf", xref: "x2 domain", x0: 0, x1: 1, yref: "y2", y0: t.value, y1: t.value, line: { color: css("--text-2"), width: 1, dash: "dash" } });
        refKey(lay, `TOLDEE ${t.value.toExponential(0)} Ha<br>(${SRC[t.source] || t.source})`, { line: { color: css("--text-2"), width: 1, dash: "dash" } });
      }
      if (blk.ended === "too_many_cycles") d.scfNote = "The SCF of this point did not converge (TOO MANY CYCLES): see Problems.";
    } else {
      d.scfNote = "Per-cycle SCF detail is printed only for the first point of an optimization." +
        (conv.scf.moved_to_scfout_log ? " CRYSTAL moved the rest to SCFOUT.LOG, which stays in the job's scratch directory." : "");
    }
    if (pts.length) selLine(r, lay, "y domain");
    return { traces, layout: lay };
  }

  function cellFigure(d) {
    const o = d.opts.cell, r = d.run;
    const lay = base(o);
    lay.shapes = []; lay.annotations = [];
    // eight entries of very different widths (a, the restart key): plotly wraps plain rows into a
    // grid as wide as the widest entry, one entry a row in a narrow pane; one group per entry
    // (grouped order) wraps them as they are wide instead
    lay.legend = PK.legendTop({ traceorder: "grouped", tracegroupgap: 0 });
    const pal = Array.from({ length: 6 }, (_, k) => css(`--plot-${k + 1}`));
    const pts = r.conv.points.filter((p) => p.cell);
    const pct = o.relTo === "first";
    const v0 = pts.length ? pts[0].cell : null;
    const val = (p, k) => (pct ? (p.cell[k] - v0[k]) / v0[k] * 100 : p.cell[k]);
    const grid = css("--line");
    const traces = [];
    [["a", "a"], ["b", "b"], ["c", "c"]].forEach(([k, n], i) => traces.push({ x: pts.map((p) => p.point), y: pts.map((p) => val(p, k)), customdata: pts.map((p) => p.frame),
      name: `${n} (${pct ? "%" : "Å"})`, meta: `cell-${k}`, mode: "lines+markers", type: "scatter", xaxis: "x", yaxis: "y", line: { color: pal[i], width: o.width } }));
    traces.push({ x: pts.map((p) => p.point), y: pts.map((p) => val(p, "volume")), customdata: pts.map((p) => p.frame), name: `volume (${pct ? "%" : "Å³"})`, meta: "cell-volume",
      mode: "lines+markers", type: "scatter", xaxis: "x", yaxis: "y3", line: { color: pal[3], width: o.width, dash: "dot" } });
    [["alpha", "α"], ["beta", "β"], ["gamma", "γ"]].forEach(([k, n], i) => traces.push({ x: pts.map((p) => p.point), y: pts.map((p) => val(p, k)), customdata: pts.map((p) => p.frame),
      name: `${n} (${pct ? "%" : "°"})`, meta: `cell-${k}`, mode: "lines+markers", type: "scatter", xaxis: "x2", yaxis: "y2", line: { color: pal[i], width: o.width, dash: "dash" } }));
    lay.yaxis = { ...lay.yaxis, title: { text: pct ? "a, b, c (% from first)" : "a, b, c (Å)" }, domain: [0.55, 1], type: "linear" };
    lay.yaxis3 = { title: { text: pct ? "V (% from first)" : "V (Å³)" }, overlaying: "y", side: "right", showgrid: false };
    lay.xaxis = { ...lay.xaxis, anchor: "y", title: { text: "" } };
    lay.xaxis2 = { title: { text: XT }, anchor: "y2", gridcolor: grid, showline: true, mirror: true, linecolor: grid, matches: "x" };
    lay.yaxis2 = { title: { text: pct ? "α, β, γ (% from first)" : "α, β, γ (°)" }, domain: [0, 0.42], anchor: "x2", gridcolor: grid, showline: true, mirror: true, linecolor: grid };
    segLines(r, lay);
    selLine(r, lay);
    [...traces, ...lay._keys].forEach((t, i) => { t.legendgroup = `g${i}`; });
    return { traces, layout: lay };
  }

  function trackedFigure(d) {
    const o = d.opts.tracked, r = d.run;
    const lay = base(o);
    lay.shapes = []; lay.annotations = [];
    const pal = Array.from({ length: 6 }, (_, k) => css(`--plot-${k + 1}`));
    const pointOf = {};
    for (const p of r.conv.points) if (p.frame !== null && p.frame !== undefined) pointOf[p.frame] = p.point;
    const ff = r.conv.final_frame, lastPt = r.conv.points.length ? r.conv.points[r.conv.points.length - 1].point : null;
    const traces = d.tracks.filter((t) => t.values).map((t, i) => {
      const xs = t.values.map((_, k) => (k === ff ? lastPt : pointOf[k] ?? k + 1));
      return { x: xs, y: t.values, customdata: t.values.map((_, k) => k), mode: "lines+markers", type: "scatter", name: t.label, meta: `track${i}`,
        line: { color: pal[i % 6], width: o.width }, hovertemplate: `POINT %{x}: %{y:.4f} ${t.type === "distance" ? "Å" : "°"}<extra>${esc(t.label)}</extra>` };
    });
    lay.yaxis.title = { text: d.tracks.some((t) => t.type !== "distance") ? "Distance (Å) · angle (°)" : "Distance (Å)" };
    selLine(r, lay);
    return { traces, layout: lay };
  }

  /* ------------------------------------------------ drawing */
  function emptyText(d) {
    const c = d.run.conv;
    if (!c.is_opt) return d.tab === "scf" ? "" : "This is a single-point run: no optimization steps. The SCF tab shows its energy convergence.";
    if (c.opt && c.opt.end && c.opt.end.reason === "gradient_first_cycle" && d.tab !== "scf") return "Only one point: the gradients were below threshold at the start.";
    if (c.points.length === 1 && d.tab !== "scf") return "A single point: nothing to follow yet.";
    if (d.tab === "tracked" && !d.tracks.length) return "Measure a distance or angle on the run (Measurements), then Track Along the Run.";
    return "";
  }
  /* points and SCF cycles are whole numbers: no 1.5 ticks on a short run */
  function integerTicks(fig) {
    for (const ax of ["xaxis", "xaxis2"]) {
      const a = fig.layout[ax];
      if (!a || a.type === "log" || a.range || a.visible === false) continue;
      const name = ax === "xaxis" ? "x" : "x2";
      const xs = fig.traces.filter((t) => (t.xaxis || "x") === name).flatMap((t) => t.x || []).filter((v) => typeof v === "number");
      if (xs.length && Math.max(...xs) - Math.min(...xs) <= 20) { a.tick0 = 0; a.dtick = 1; }
    }
  }
  async function redraw() {
    if (!doc || !doc.run) return;
    if (building) { again = true; return; }
    building = true;
    el.dataset.drawn = "";
    try {
      await loadPlotly();
      if (!window.Plotly) throw new Error("plotly is not installed in this Python; pip install plotly");
      const fig = ({ energy: energyFigure, convergence: convergenceFigure, scf: scfFigure, cell: cellFigure, tracked: trackedFigure })[doc.tab](doc);
      const config = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] };
      integerTicks(fig);
      // the reference lines' legend entries after the data's; the margins kept from the last fit (fitLegends)
      fig.traces.push(...(fig.layout._keys || []));
      delete fig.layout._keys;
      if (el.__sig === `${doc.id}|${doc.tab}` && el.layout && el.layout.margin) Object.assign(fig.layout.margin, { t: el.layout.margin.t, r: el.layout.margin.r });
      const sig = `${doc.id}|${doc.tab}`;
      // the note, the chips and the criteria strip first: the plot is then drawn
      // into the height they leave (drawn after, the strip covered the plot's x axis)
      const note = emptyText(doc);
      $(".pv-note").textContent = note || (doc.tab === "scf" ? doc.scfNote : doc.tab === "energy" && doc.opts.energy.ylog
        ? `CRYSTAL's DE is relative to the optimizer's reference geometry; the second trace is the distance to ${doc.opts.energy.segments === "separate" && doc.run.conv.segments.length > 1 ? "each FINALRUN segment's" : "the run's"} final energy.`
        : `${doc.run.conv.stop || ""}`);
      // one line, the whole of it in the tooltip
      $(".pv-note").title = $(".pv-note").textContent;
      chips();
      criteria();
      if (el.__sig !== sig) { if (el.__sig) window.Plotly.purge(el); el.__bound = false; await window.Plotly.newPlot(el, fig.traces, fig.layout, config); el.__sig = sig; }
      else await window.Plotly.react(el, fig.traces, fig.layout, config);
      building = false;
      legendBase = { t: 18, r: 18 };
      await PK.fitLegends(el, legendBase);
      bind();
      el.dataset.drawn = `${doc.id}|${doc.tab}`;
    } catch (e) { $(".pv-note").textContent = e.message; } finally {
      building = false;
      if (again) { again = false; redraw(); }
    }
  }
  function chips() {
    const parts = [];
    if (doc.tab === "energy" && doc.belowCount) parts.push(`<span class="pv-chip rp-below-chip" title="Below their reference energy (red down-triangles)">${doc.belowCount} point${doc.belowCount === 1 ? "" : "s"} below ${doc.belowCount === 1 ? "its" : "their"} reference energy</span>`);
    if (doc.tab === "energy" && doc.opts.energy.ylog) parts.push(`<span class="pv-chip" title="A log axis cannot show the reference point itself">the reference point has no value on a log axis</span>`);
    if (doc.other) parts.push(`<span class="pv-chip">${esc(doc.run.name)} vs ${esc(doc.other.name)}</span>`);
    $(".pv-chips").innerHTML = parts.join("");
  }
  /* the criteria strip: one row per criterion, ✓/✗ per point */
  function criteria() {
    const b = $(".rp-below");
    if (doc.tab !== "convergence" || !doc.run.conv.points.length) { b.innerHTML = ""; return; }
    const pts = doc.run.conv.points;
    b.innerHTML = `<table class="data rp-strip" aria-label="Convergence criteria per point"><thead><tr><th>Criterion</th>${pts.map((p) => `<th class="num">${p.point}</th>`).join("")}</tr></thead><tbody>` +
      CRIT.map(([k, label]) => `<tr><td>${esc(label)}</td>${pts.map((p) => `<td class="num ${p.converged[k] ? "good" : p.converged[k] === false ? "bad" : ""}">${p.converged[k] === null ? "–" : p.converged[k] ? "✓" : "✗"}</td>`).join("")}</tr>`).join("") +
      "</tbody></table>";
  }
  function bind() {
    if (el.__bound) return;
    el.__bound = true;
    el.on("plotly_legendclick", PK.keyClick);
    el.on("plotly_legenddoubleclick", PK.keyClick);
    // a resize redraws the legend: fit the margins to it again
    el.on("plotly_afterplot", () => { if (legendBase && !el.__fitting && !building) requestAnimationFrame(() => PK.fitLegends(el, legendBase)); });
    el.on("plotly_click", (ev) => {
      if (!doc || !ev.points || !ev.points.length) return;
      const p = ev.points[0];
      const fr = p.customdata;
      if (fr === null || fr === undefined) return;
      if (doc.tab === "scf" && p.data.meta === "cycles") { pickFrame(fr); return; }
      pickFrame(fr);
    });
  }
  function pickFrame(fr) {
    if (H.selectFrame && H.selectFrame(doc.run.path, fr)) redraw();
    else if (H.openRun) H.openRun(doc.run.path).then(() => { H.selectFrame(doc.run.path, fr); redraw(); });
  }
  document.addEventListener("mace-frame-changed", (e) => { if (doc && doc.run && e.detail && e.detail.path === doc.run.path && host.offsetParent !== null) redraw(); });
  document.addEventListener("mace-run-updated", (e) => { if (doc && doc.run && e.detail && e.detail.path === doc.run.path) refreshData(); });
  document.addEventListener("mace-theme", () => { if (doc) { el.__sig = null; redraw(); } });

  /* tabs */
  $(".rp-tabs").addEventListener("click", (e) => {
    const b = e.target.closest("[data-tab]");
    if (b && doc) showTab(b.dataset.tab);
    const c = e.target.closest("[data-cellmode]");
    if (c && doc) set({ relTo: c.dataset.cellmode }, c.dataset.cellmode === "first" ? "% from first" : "Absolute");
  });
  $(".rp-tabs").addEventListener("keydown", (e) => {
    const tabs = [...host.querySelectorAll("[data-tab]")], k = tabs.indexOf(document.activeElement);
    if (k < 0 || !["ArrowLeft", "ArrowRight"].includes(e.key)) return;
    e.preventDefault();
    const n = tabs[(k + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
    n.focus(); showTab(n.dataset.tab);
  });
  function showTab(t) {
    doc.tab = t;
    persist(doc);
    host.querySelectorAll("[data-tab]").forEach((b) => { const on = b.dataset.tab === t; b.classList.toggle("active", on); b.setAttribute("aria-selected", String(on)); b.tabIndex = on ? 0 : -1; });
    $(".rp-tabopts").innerHTML = t === "cell" ? `<span class="segment" role="radiogroup" aria-label="Cell values">${[["absolute", "Absolute"], ["first", "% from first"]]
      .map(([v, l]) => `<button role="radio" data-cellmode="${v}" aria-checked="${doc.opts.cell.relTo === v}">${l}</button>`).join("")}</span>` : "";
    if (kit) kit.refresh();
    redraw();
  }

  /* the table under every export (CSV, Copy as TSV): the plotted data in the plotted units */
  function table() {
    const d = doc, t = d.tab;
    if (t === "convergence") {
      const pts = d.run.conv.points;
      return { columns: ["point", "frame", "max_grad", "rms_grad", "max_disp", "rms_disp"],
        rows: pts.map((p) => [p.point, p.frame, p.max_grad, p.rms_grad, p.max_disp, p.rms_disp]) };
    }
    if (t === "scf") return { columns: ["point", "frame", "scf_cycles"], rows: d.run.conv.points.map((p) => [p.point, p.frame, p.scf_cycles]) };
    if (t === "cell") return { columns: ["point", "frame", "a", "b", "c", "alpha", "beta", "gamma", "volume"],
      rows: d.run.conv.points.filter((p) => p.cell).map((p) => [p.point, p.frame, p.cell.a, p.cell.b, p.cell.c, p.cell.alpha, p.cell.beta, p.cell.gamma, p.cell.volume]) };
    if (t === "tracked") {
      const n = Math.max(0, ...d.tracks.map((x) => (x.values || []).length));
      const { pts } = energies(d.run);
      const pointOf = {}; for (const p of d.run.conv.points) pointOf[p.frame] = p.point;
      const eOf = {}; for (const p of pts) eOf[p.frame] = p.e;
      return { columns: ["point", "frame", "energy_Ha", ...d.tracks.map((x) => x.label)],
        rows: Array.from({ length: n }, (_, k) => [pointOf[k] ?? "", k, eOf[k] ?? "", ...d.tracks.map((x) => (x.values || [])[k])]) };
    }
    // energy: one value column per run (its segments joined; the final-geometry ring left out),
    // two per run in the log view (CRYSTAL's DE and the distance to the final energy)
    const data = (el.data || []).filter((tr) => tr.meta && !String(tr.meta).startsWith("fin"));
    const o = d.opts.energy, ul = PK.unitText(o.unit, o.per || "cell");
    const runs = [d.run, ...(d.other ? [d.other] : [])];
    const groups = [];
    runs.forEach((r, ri) => {
      const sep = o.segments === "separate" && r.conv.segments.length > 1;
      const pre = o.ylog ? [[`de${ri}`, "|DE|"], [`seg${ri}`, sep ? "|E - E_seg_final|" : "|E - E_final|"]] : [[`e${ri}_`, o.relTo === "absolute" ? "E" : "dE"]];
      for (const [m, lab] of pre) {
        const trs = data.filter((tr) => String(tr.meta).startsWith(m) && (o.ylog ? String(tr.meta) === m : true));
        if (trs.length) groups.push({ label: `${runs.length > 1 ? r.name + " " : ""}${lab} (${ul})`, trs });
      }
    });
    const xs = [...new Set(groups.flatMap((g) => g.trs.flatMap((tr) => tr.x || [])))].sort((a, b) => a - b);
    const val = (g, x) => { for (const tr of g.trs) { const i = (tr.x || []).indexOf(x); if (i >= 0) return tr.y[i]; } return ""; };
    return { columns: ["point", ...groups.map((g) => g.label)], rows: xs.map((x) => [x, ...groups.map((g) => val(g, x))]) };
  }

  async function refreshData() {
    if (!doc) return;
    try {
      doc.run = await runData(doc.runPath);
      if (doc.compare) doc.other = await runData(doc.compare);
      await retrack();
      if (kit) kit.refresh();          // the features follow the data (Segments only for a run with FINALRUN restarts)
      redraw();
    } catch (e) { $(".pv-note").textContent = e.message; }
  }
  async function retrack() {
    for (const t of doc.tracks) {
      try {
        const r = await (H.apiTask ? H.apiTask("out/track", { path: doc.runPath, measurement: t.measurement }) : api("out/track", { body: { path: doc.runPath, measurement: t.measurement } }));
        t.values = r.values;
      } catch (e) { t.values = null; t.error = e.message; }
    }
  }

  return {
    async open(d) {
      doc = d;
      load(d);
      takeEnergy(d);
      $(".pv-title").textContent = d.title;
      $(".pv-note").innerHTML = `<span class="spinner"></span> Reading ${esc(d.runPath.split("/").pop())}…`;
      if (!kit) kit = PK.attach($(".pv-acts"), {
        get name() { return doc ? `${doc.run ? doc.run.name.replace(/\.out$/, "") : "run"}_${doc.tab}` : "run"; },
        host: el,
        get features() {
          const t = doc ? doc.tab : "energy";
          return { logY: t !== "cell", logX: false, units: t === "energy" ? "energy" : null, per: t === "energy", relTo: t === "energy",
            segments: t === "energy" && !!doc && !!doc.run && doc.run.conv.segments.length > 1, range: true, width: true, png: true, svg: true, csv: true, pub: false };
        },
        get disabled() {
          if (!doc || !doc.other || doc.tab !== "energy") return {};
          const o = doc.opts.energy;
          return o.per === "cell" || !sameComp(doc.run, doc.other) ? { "relTo:absolute": "different compositions: absolute energies are not comparable" } : {};
        },
        get: () => (doc ? O() : {}),
        set: (patch, label) => doc && set(patch, label),
        axes: () => ({ x: "Point", y: doc && doc.tab === "energy" ? "Energy" : "Value" }),
        table: () => table(),
        png: async () => { const url = await window.Plotly.toImage(el, { format: "png", scale: Number(setting("plots.exportScale", 2)) || 2, width: el.clientWidth, height: el.clientHeight }); download(PK.dataUrlBlob(url), `${kit.name || "run"}.png`); },
        svg: async () => { const url = await window.Plotly.toImage(el, { format: "svg", width: el.clientWidth, height: el.clientHeight }); download(PK.dataUrlBlob(url), `${kit.name || "run"}.svg`); },
        resetZoom: () => { el.__sig = null; redraw(); },
      }, { toast, download, icon });
      showTab(d.tab);
      await refreshData();
      if (doc === d && d.other) {
        // compare: per atom when the atom counts differ (per cell when they are equal)
        if (!d.perSet) { d.opts.energy.per = d.run.n === d.other.n ? "cell" : "atom"; d.perSet = true; persist(d); }
        if (d.opts.energy.relTo === "absolute" && (d.opts.energy.per === "cell" || !sameComp(d.run, d.other))) d.opts.energy.relTo = "final";
        kit.refresh();
        redraw();
      }
    },
    close() { doc = null; },
    /** the energy card's Open as Plot: the plot shows the card's data (reference, segments, unit, log) */
    setEnergy(d, patch) { if (!d || !patch) return; d.energyPatch = patch; if (doc === d && d.opts) { takeEnergy(d); if (kit) kit.refresh(); redraw(); } },
    showTab: (t) => doc && showTab(t),
    track(measurement, label) {
      if (!doc) return;
      doc.tracks.push({ measurement, label, type: measurement.type, values: null });
      persist(doc);
      showTab("tracked");
      refreshData();
    },
    removeTrack(i) { if (doc) { doc.tracks.splice(i, 1); persist(doc); redraw(); } },
    state: () => doc && { tab: doc.tab, opts: JSON.parse(JSON.stringify(doc.opts)), drawn: el.dataset.drawn, tracks: doc.tracks.map((t) => ({ label: t.label, values: t.values })),
      run: doc.run && { n: doc.run.n, points: doc.run.conv.points.length }, other: doc.other && { n: doc.other.n } },
    figure: () => (el.data ? { traces: el.data.map((t) => ({ name: t.name, meta: t.meta, x: t.x, y: t.y, marker: t.marker, type: t.type })), layout: el.layout } : null),
    table: () => doc && table(),
    el: () => el,
  };
}
