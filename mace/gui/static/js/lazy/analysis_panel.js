/* Analysis › Structure Analysis (1.1.6): porosity (with pores drawn in the
 * 3D view, accessible and inaccessible apart), powder patterns (simulated,
 * compared with a measured one, peak search, indexing, Pawley/Rietveld),
 * topology (net, RCSR symbol, interpenetration), coordination polyhedra,
 * bond and angle distributions, crystal morphology and Miller planes.
 *
 * Every tool runs on the server (POST analysis/run, a background task) on
 * the document's structure as Export Structure sends it; the same tools are
 * `mace analyze <tool>` on the command line. */
import { loadCss, panel } from "./lazy_util.js";
import { store, DOC, PORE_KEYS } from "./analysis_store.js";
import * as MQ from "../viewport/mesh_quality.js";
import * as L from "../labels.js";

/* the tools a tab runs (their runs show in its status line) */
const TAB_TOOLS = { porosity: ["porosity"], pxrd: ["pxrd"], fit: ["peaks", "index", "refine"], topology: ["topology"],
  polyhedra: ["polyhedra"], bonds: ["bonds"], morphology: ["morphology"], planes: ["planes", "cleavage"] };
const TABS = [
  ["porosity", "Porosity"], ["pxrd", "Powder Pattern"], ["fit", "Peaks & Fit"], ["topology", "Topology"],
  ["polyhedra", "Polyhedra"], ["bonds", "Bonds & Angles"], ["morphology", "Morphology"], ["planes", "Planes"],
];
const css = () => Promise.all([loadCss(import.meta.url, "../../css/dialogs.css"), loadCss(import.meta.url, "../../css/analysis.css")]);
let state = null;         // one panel per page: { H, A (the store), d, tab, info, pattern, status, seq }
/* Results are kept per document (analysis_store.js, which the bottom panel's
 * Analysis tab reads too) and carry the document they were computed for, so
 * the panel never shows, and the 3D view never draws, one structure's pores
 * on another. */
const keyOf = (id) => (id == null ? "-" : String(id));
const activeDoc = () => state.H.shell().activeDoc();
const activeId = () => { const d = activeDoc(); return d ? d.id : null; };
const res = () => state.A.latest(activeDoc());
const mine = (r) => {
  if (!r || !r[DOC]) return false;
  const d = activeDoc();
  return !!d && (r[DOC].id === d.id || (r[DOC].id == null && !!r[DOC].path && r[DOC].path === d.path));
};
/** A result that came back for the document on screen: drawn into ``out``
 * (or the whole tab again when ``out`` was replaced meanwhile). */
function landed(r, out, draw) {
  if (!mine(r)) return false;
  if (out && out.isConnected) draw(); else render();
  return true;
}

export async function open(H, args = {}) {
  await css();
  const S = H.shell();
  if (!state) {
    state = { H, A: store(H), tab: "porosity", info: null, pattern: null, status: new Map(), seq: {} };
    state.overlay = state.A.overlay();
    // another document: the panel shows that document's results (or none)
    document.addEventListener("mace-doc-switch", () => { if (state.d && state.d.isConnected && state.d.open) render(); });
    // a result deleted (or its mesh data) in the Analysis tab: the tab on screen follows
    document.addEventListener("mace-analysis", (e) => {
      const w = e.detail && e.detail.what;
      if ((w === "remove" || w === "meshes") && state.d && state.d.isConnected && state.d.open) render();
    });
    // Settings › Viewport › pores.* (or another window's change): the controls follow (the store redraws the surfaces)
    try {
      H.settings.onChange((k) => {
        if (!String(k).startsWith("pores.")) return;
        const po = poreOpts();
        const root = state.d && state.d.querySelector(".an-out");
        if (!root) return;
        for (const [o, v] of Object.entries(po)) {
          const e = root.querySelector(`[data-o="${o}"]`);
          if (e && e.type === "checkbox") e.checked = !!v; else if (e && document.activeElement !== e) e.value = String(v);
        }
      });
    } catch (e) { /* no live settings: the controls still work */ }
  }
  state.H = H;
  state.A = store(H);
  if (args && args.tab) state.tab = args.tab;
  const body = document.createElement("div");
  body.className = "an-root";
  body.innerHTML = `<div class="an-tabs" role="tablist">${TABS.map(([k, t]) =>
    `<button type="button" role="tab" class="an-tab" data-tab="${k}">${H.esc(t)}</button>`).join("")}</div>
    <div class="an-doc insp-hint" data-k="doc"></div><div class="an-body" data-k="body"></div>`;
  state.d = panel(H, { id: "analysis-panel", title: "Structure Analysis", body, wide: true, resizable: true,
    foot: `<span class="insp-hint an-cli" data-k="cli"></span><button class="tb-btn-text" data-k="batch" title="One tool on many structure files, a few at a time (Explorer › Analyze Structures…)">Batch…</button>
      <button class="tb-btn-text" data-k="results" title="Every result of this structure, with its settings and tables (the bottom panel's Analysis tab)">All Results</button>
      <button class="tb-btn-text" data-k="clear3d">Clear 3D Overlays</button>`,
    onClose: () => {} });
  const d = state.d;
  d.classList.add("an-panel");
  d.querySelectorAll(".an-tab").forEach((b) => b.addEventListener("click", () => { state.tab = b.dataset.tab; render(); }));
  d.querySelector('[data-k="clear3d"]').addEventListener("click", () => { state.overlay.hide(); render(); });
  d.querySelector('[data-k="results"]').addEventListener("click", () => S.showBottom("analysis"));
  d.querySelector('[data-k="batch"]').addEventListener("click", () => H.run("explorer.analyze"));
  if (!state.info) {
    try { state.info = await H.api("analysis/info"); } catch (e) { state.info = { probes: {}, sources: {}, error: e.message }; }
  }
  render();
  return d;
}

/* ---------------------------------------------------------------- helpers */
const esc = (t) => state.H.esc(String(t == null ? "" : t));
const fmt = (x, n = 3) => (x == null || Number.isNaN(x) ? "–" : Number(x).toFixed(n));
const val = (root, k) => { const e = root.querySelector(`[data-p="${k}"]`); return !e ? undefined : e.type === "checkbox" ? e.checked : e.value; };
function params(root) {
  const out = {};
  root.querySelectorAll("[data-p]").forEach((e) => { out[e.dataset.p] = e.type === "checkbox" ? e.checked : e.value; });
  return out;
}
function download(name, text, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url; a.download = name; a.hidden = true;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}
function docStem() {
  const d = state.H.shell().activeDoc();
  return ((d && d.title) || "structure").replace(/\.[A-Za-z0-9]+$/, "").replace(/[^A-Za-z0-9._-]+/g, "_");
}

/* a small SVG chart: lines (and sticks) in the panel's own colors */
// the axis titles in plot markup (labels.js), drawn as tspans (L.svgText)
const TWO_THETA = L.withUnit("2θ", "deg");
function chart({ series = [], sticks = [], ticks = [], xlabel = "", ylabel = "", w = 640, h = 240, xr = null, bars = null }) {
  const pad = { l: 46, r: 10, t: 8, b: 30 };
  let xs = [], ys = [];
  series.forEach((s) => { xs = xs.concat(s.x); ys = ys.concat(s.y); });
  sticks.forEach((s) => { xs.push(s.x); ys.push(s.y); ys.push(0); });
  if (bars) { xs.push(bars.edges[0], bars.edges[bars.edges.length - 1]); ys = ys.concat(bars.counts, [0]); }
  if (!xs.length) return `<div class="insp-hint">No data</div>`;
  let [x0, x1] = xr || [Math.min(...xs), Math.max(...xs)];
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (x1 - x0 < 1e-9) x1 = x0 + 1;
  if (y1 - y0 < 1e-9) y1 = y0 + 1;
  const ymin = y0 - 0.04 * (y1 - y0), ymax = y1 + 0.06 * (y1 - y0);
  const X = (x) => pad.l + (x - x0) / (x1 - x0) * (w - pad.l - pad.r);
  const Y = (y) => h - pad.b - (y - ymin) / (ymax - ymin) * (h - pad.t - pad.b);
  const path = (s) => s.x.map((x, i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(s.y[i]).toFixed(1)}`).join("");
  const nt = 6, step = niceStep((x1 - x0) / nt);
  const xt = [];
  for (let t = Math.ceil(x0 / step) * step; t <= x1 + 1e-9; t += step) xt.push(t);
  let out = `<svg class="an-chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(L.plain(ylabel))} vs ${esc(L.plain(xlabel))}">`;
  out += `<line x1="${pad.l}" y1="${h - pad.b}" x2="${w - pad.r}" y2="${h - pad.b}" class="an-axis"/>`;
  out += `<line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${h - pad.b}" class="an-axis"/>`;
  xt.forEach((t) => { out += `<line x1="${X(t)}" y1="${h - pad.b}" x2="${X(t)}" y2="${h - pad.b + 4}" class="an-axis"/>
    <text x="${X(t)}" y="${h - pad.b + 16}" class="an-tick" text-anchor="middle">${+t.toFixed(4)}</text>`; });
  out += `<text x="${(pad.l + w - pad.r) / 2}" y="${h - 2}" class="an-tick" text-anchor="middle">${L.svgText(xlabel)}</text>`;
  out += `<text x="12" y="${(pad.t + h - pad.b) / 2}" class="an-tick" text-anchor="middle" transform="rotate(-90 12 ${(pad.t + h - pad.b) / 2})">${L.svgText(ylabel)}</text>`;
  if (bars) {
    bars.counts.forEach((c, i) => {
      const a = X(bars.edges[i]), b = X(bars.edges[i + 1]);
      out += `<rect x="${a.toFixed(1)}" y="${Y(c).toFixed(1)}" width="${Math.max(0.5, b - a - 0.5).toFixed(1)}" height="${(Y(0) - Y(c)).toFixed(1)}" class="an-bar"/>`;
    });
  }
  sticks.forEach((s) => { out += `<line x1="${X(s.x)}" x2="${X(s.x)}" y1="${Y(0)}" y2="${Y(s.y)}" class="an-stick ${s.cls || ""}"><title>${esc(s.title || "")}</title></line>`; });
  series.forEach((s) => { out += `<path d="${path(s)}" class="an-line ${s.cls || ""}" fill="none"/>`; });
  ticks.forEach((t) => { out += `<line x1="${X(t.x)}" x2="${X(t.x)}" y1="${h - pad.b - 2}" y2="${h - pad.b - 9}" class="an-refl"><title>${esc(t.title || "")}</title></line>`; });
  return out + "</svg>";
}
function niceStep(raw) {
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
}
function table(head, rows, cls = "") {
  return `<table class="an-table ${cls}"><thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}
const hklStr = (h) => (h ? `(${h.join(" ")})` : "–");

/* ---------------------------------------------------------------- running */
/* A run goes on in the background until it is canceled (here, or in the
 * Tasks tab): another document, another tab of this panel or closing the
 * panel leave it running, and its status line comes back with the tab.
 * state.status: "doc|tool" -> { text, cls, task (its id while it runs) }. */
const statusKey = (docId, tool) => `${keyOf(docId)}|${tool}`;
// the task titles analysis_routes.py gives ("Porosity: SOD")
const TITLES = { porosity: "Porosity", pxrd: "Powder pattern", peaks: "Peak search", index: "Indexing", refine: "Pattern fit",
  topology: "Topology", polyhedra: "Polyhedra", bonds: "Bond distributions", morphology: "Morphology", planes: "Miller plane", cleavage: "Cleavage" };
/** The task of a run still going: its id once the server answered, else the newest task of
 * this window for that document and tool (its events come before the answer of a long run). */
function taskOf(st) {
  if (st.task) return st.task;
  const S = state.H.shell(), me = S.clientId ? S.clientId() : null;
  const live = [...(S.tasks ? S.tasks().values() : [])].filter((t) => t.kind === "analysis" && t.doc === st.doc &&
    (t.state === "running" || t.state === "queued") && (!me || !t.client || t.client === me) &&
    (t.title === TITLES[st.tool] || String(t.title).startsWith(`${TITLES[st.tool]}: `)));
  live.sort((a, b) => (b.t0 || 0) - (a.t0 || 0));
  return live.length ? live[0].id : null;
}
function paintStatus() {
  if (!state.d) return;
  const host = state.d.querySelector('.an-form [data-k="status"]');
  if (!host) return;
  const id = activeId();
  const tools = TAB_TOOLS[state.tab] || [];
  const st = tools.map((t) => state.status.get(statusKey(id, t))).find((x) => x && x.running) ||
    tools.map((t) => state.status.get(statusKey(id, t))).find(Boolean);
  host.className = `an-status ${st ? st.cls : ""}`;
  host.textContent = st ? st.text : "";
  if (st && st.running) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "tb-btn-text an-cancel"; b.textContent = "Cancel";
    b.title = "Stop this run (otherwise it goes on in the background; the Tasks tab lists it)";
    b.addEventListener("click", () => {
      const t = taskOf(st);
      if (!t) { state.H.toast("The run has not reached the server yet: try again in a moment", "info"); return; }
      b.disabled = true;
      state.H.shell().cancelTask(t);
    });
    host.append(" ", b);
  }
}
async function run(tool, p, extra = {}) {
  const H = state.H, S = H.shell();
  // the document is fixed when the run starts: the result belongs to it even if another is active when it lands
  const doc = S.activeDoc();
  const sk = statusKey(doc ? doc.id : null, tool);
  // a newer run of the same tool on the same document owns the status line from its start
  const seq = (state.seq[sk] = (state.seq[sk] || 0) + 1);
  const say = (t, cls = "", task = null, running = false) => {
    if (state.seq[sk] !== seq) return;
    if (t) state.status.set(sk, { text: t, cls, task, running, tool, doc: doc ? doc.id : null }); else state.status.delete(sk);
    paintStatus();
  };
  if (!doc && tool !== "peaks" && tool !== "index") { say("Open a structure first", "err"); return null; }
  const ex = { ...extra };
  if (state.pattern && ["peaks", "index", "refine", "pxrd"].includes(tool)) Object.assign(ex, state.pattern.body);
  say("Running…", "busy", null, true);
  const tabOf = Object.keys(TAB_TOOLS).find((k) => TAB_TOOLS[k].includes(tool)) || state.tab;
  try {
    const rec = await state.A.run(tool, p, { doc, extra: ex, onUpdate: (t) => {
      if (FINAL.has(t.state)) return;
      say(`${t.message || t.state}${t.progress ? ` (${Math.round(100 * t.progress)}%)` : ""}`, "busy", t.id, true);
      // the Tasks tab's Open: this document's results in this panel
      const tr = S.tasks && S.tasks().get(t.id);
      if (tr && !tr.open) tr.open = () => reveal(doc, tabOf);
    } });
    if (!rec) { say("Canceled"); return null; }
    say("");
    if (doc && activeId() !== doc.id) H.toast(`${tool[0].toUpperCase()}${tool.slice(1)} of ${doc.title} is done: switch back to it to see the results`, "info");
    return rec.result;
  } catch (e) {
    // canceled while the server still held the request (409 {canceled}), or later (the task's end)
    const task = e.task || (e.data && e.data.canceled ? e.data.task : null);
    const canceled = e.state === "canceled" || !!(e.data && e.data.canceled);
    say(canceled ? (task && task.message === "superseded" ? "Replaced by a newer run" : "Canceled") : e.message || String(e), canceled ? "" : "err");
    return null;
  }
}
const FINAL = new Set(["done", "failed", "canceled", "lost"]);
/** A document's results in this panel: that document in front, the panel open at ``tab``. */
function reveal(doc, tab) {
  const S = state.H.shell();
  if (doc) {
    const i = S.docs().findIndex((d) => d.id === doc.id);
    if (i >= 0 && activeId() !== doc.id) S.activate(i);
  }
  return S.builder("analysis_panel", "open", { tab });
}

function patternPicker(root) {
  const p = state.pattern;
  const box = document.createElement("div");
  box.className = "an-row";
  box.innerHTML = `<label>Measured pattern</label>
    <input type="file" accept=".xy,.xye,.dat,.txt,.csv,.xrdml" data-k="pfile" aria-label="Pattern file on this computer">
    <span class="insp-hint">or a file in the folder:</span>
    <input type="text" data-k="ppath" placeholder="patterns/sample.xy" value="${esc(p && p.body.pattern_path ? p.body.pattern_path : "")}" size="18">
    <span class="insp-hint" data-k="pname">${p ? esc(p.name) : "none"}</span>`;
  box.querySelector('[data-k="pfile"]').addEventListener("change", (e) => {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    if (f.size > 20 * 1024 * 1024) { state.H.toast("The pattern is over 20 MB", "error"); return; }
    const rd = new FileReader();
    rd.onload = () => { state.pattern = { name: f.name, body: { pattern_text: String(rd.result), pattern_name: f.name } }; box.querySelector('[data-k="pname"]').textContent = f.name; };
    rd.readAsText(f);
  });
  box.querySelector('[data-k="ppath"]').addEventListener("change", (e) => {
    const v = e.target.value.trim();
    state.pattern = v ? { name: v, body: { pattern_path: v } } : null;
    box.querySelector('[data-k="pname"]').textContent = v || "none";
  });
  root.appendChild(box);
}

/* ---------------------------------------------------------------- tabs */
function render() {
  const d = state.d, H = state.H, S = H.shell();
  d.querySelectorAll(".an-tab").forEach((b) => { b.setAttribute("aria-selected", String(b.dataset.tab === state.tab)); b.classList.toggle("on", b.dataset.tab === state.tab); });
  const doc = S.activeDoc();
  d.querySelector('[data-k="doc"]').textContent = doc ? `Structure: ${doc.title}` : "Open a structure to analyze it";
  const root = d.querySelector('[data-k="body"]');
  const oldForm = root.querySelector(".an-form");
  const keep = oldForm && state.shownTab === state.tab ? params(oldForm) : null;
  const adv = oldForm && state.shownTab === state.tab ? [...oldForm.querySelectorAll("details")].map((x) => x.open) : null;
  root.innerHTML = "";
  ({ porosity: tabPorosity, pxrd: tabPxrd, fit: tabFit, topology: tabTopology, polyhedra: tabPolyhedra,
    bonds: tabBonds, morphology: tabMorphology, planes: tabPlanes })[state.tab](root);
  const f = root.querySelector(".an-form");
  if (f && keep) {
    f.querySelectorAll("[data-p]").forEach((e) => { if (e.dataset.p in keep) { if (e.type === "checkbox") e.checked = keep[e.dataset.p]; else e.value = keep[e.dataset.p]; } });
    if (adv) f.querySelectorAll("details").forEach((x, i) => { if (adv[i]) x.open = true; });
  }
  state.shownTab = state.tab;
  paintStatus();
  const cli = { porosity: "mace analyze porosity FILE", pxrd: "mace analyze pxrd FILE", fit: "mace analyze peaks | index | refine PATTERN",
    topology: "mace analyze topology FILE", polyhedra: "mace analyze polyhedra FILE", bonds: "mace analyze bonds FILE",
    morphology: "mace analyze morphology FILE", planes: "mace analyze planes FILE --hkl h k l" }[state.tab];
  d.querySelector('[data-k="cli"]').textContent = `Command line: ${cli}`;
}

function form(root, html, label, onRun) {
  const f = document.createElement("form");
  f.className = "an-form";
  f.innerHTML = `${html}<div class="an-row an-actions"><button type="submit" class="tb-btn-text primary">${esc(label)}</button>
    <span class="an-status" data-k="status" role="status"></span></div>`;
  f.addEventListener("submit", (e) => { e.preventDefault(); onRun(params(f), f); });
  root.appendChild(f);
  const out = document.createElement("div");
  out.className = "an-out";
  root.appendChild(out);
  return out;
}

function tabPorosity(root) {
  const info = state.info || {};
  const probes = Object.entries(info.probes || {});
  const out = form(root, `<div class="an-row"><label>Adsorbate</label><select data-p="probe">${probes.map(([k, p]) =>
    `<option value="${esc(k)}"${k === "N2" ? " selected" : ""}>${esc(k)}, ${esc(p.label)} (r ${fmt(p.radius, 3)} Å)</option>`).join("")}
    <option value="">Custom radius…</option></select>
    <label>Probe radius (Å)</label><input type="number" step="0.01" min="0" max="10" data-p="probe_radius" placeholder="from the adsorbate" size="6"></div>
    <details class="an-adv" data-tier="advanced"><summary>Advanced</summary><div class="an-row">
    <label>Atomic radii</label><select data-p="radii"><option value="ccdc">CCDC (Zeo++ default)</option><option value="uff">UFF (σ/2)</option></select>
    <label>Grid spacing (Å)</label><input type="number" step="0.01" min="0.05" max="1" value="0.15" data-p="spacing">
    <label><input type="checkbox" data-p="zeopp"${info.zeopp ? "" : " disabled"}> Also run Zeo++</label>
    ${info.zeopp ? "" : `<span class="insp-hint">${esc(info.zeopp_hint || "Zeo++ is not installed")}</span>`}</div></details>`,
  "Compute Pores", async (p) => {
    const r = await run("porosity", p);
    if (!r) return;
    showPores(r);
    landed(r, out, () => renderPorosity(out, r));
  });
  if (res().porosity) renderPorosity(out, res().porosity);
}

/* how the pore surfaces are drawn: Settings › Viewport › pores.*, which the controls under the
 * results change too (and so keep for the next structure and session); the Inspector's 3D
 * Overlays section changes one document's surfaces on top of them */
const poreOpts = () => state.A.poreOpts();

/* a result's overlay (pores, net, crystal shape, planes), bound to the document (and edit) it was computed for */
function showOverlay(r) {
  const rec = state.A.recOf(r);
  return rec ? state.A.show(rec) : false;
}
const showPores = showOverlay;

/* this result's overlay is the one drawn for its document */
function shownFor(r) {
  const rec = state.A.recOf(r);
  return !!rec && !!state.A.overlayOf(rec);
}

function renderPorosity(out, r) {
  const v = (x) => `${fmt(x.A3, 1)} Å³ · ${fmt(100 * x.fraction, 2)}% · ${fmt(x.cm3_g, 4)} cm³/g`;
  const a = (x) => `${fmt(x.A2, 1)} Å² · ${fmt(x.m2_cm3, 0)} m²/cm³ · ${fmt(x.m2_g, 0)} m²/g`;
  const rows = [
    ["Largest included sphere D<sub>i</sub>", `${fmt(r.Di)} Å`],
    ["Largest free sphere D<sub>f</sub>", `${fmt(r.Df)} Å (a ${fmt(r.Df_axes[0])}, b ${fmt(r.Df_axes[1])}, c ${fmt(r.Df_axes[2])})`],
    ["Included sphere along free path D<sub>if</sub>", `${fmt(r.Dif)} Å`],
    ["Geometric void fraction (no probe)", `${fmt(100 * r.void_fraction_geometric, 2)}%`],
    ["Accessible volume, probe center (AV)", v(r.AV)], ["Non-accessible, probe center (NAV)", v(r.NAV)],
    ["Probe-occupiable accessible (POAV)", v(r.POAV)], ["Probe-occupiable non-accessible (PONAV)", v(r.PONAV)],
    ["Accessible surface area (ASA)", a(r.ASA)], ["Non-accessible surface area (NASA)", a(r.NASA)],
    ["Density", `${fmt(r.density_g_cm3, 4)} g/cm³`],
    ["Channels / pockets", `${r.n_channels} channel system(s), dimensionality ${r.channel_dimensionality}; ${r.n_pockets} pocket(s)`],
  ];
  const at = r[DOC] || {};
  const how = r.Df_method === "voronoi" ? "the Voronoi network of the atoms decides D<sub>f</sub> and which pores connect"
    : "grid percolation decides D<sub>f</sub> and which pores connect";
  let html = `<div class="an-sub an-method" data-k="method">${at.title ? `<b>${esc(at.title)}</b> · ` : ""}MACE native engine: distance grid ${r.grid.shape.join("×")} at ${fmt(r.grid.spacing, 3)} Å, ${how}
    · probe ${esc(r.probe.name)}, r = ${fmt(r.probe.radius)} Å · ${esc(String(r.radii).toUpperCase())} atomic radii</div>`;
  html += table(["Quantity", "Value"], rows.map(([k, x]) => [k, x]));
  if (r.zeopp) {
    const z = r.zeopp;
    html += `<div class="an-sub">Zeo++ (${esc(z.binary)}), same structure, probe and radii</div>` + table(["", "native", "Zeo++"], [
      ["D<sub>i</sub>", fmt(r.Di), fmt(z.Di)], ["D<sub>f</sub>", fmt(r.Df), fmt(z.Df)], ["D<sub>if</sub>", fmt(r.Dif), fmt(z.Dif)],
      ["AV fraction", fmt(r.AV.fraction, 4), fmt(z.AV.fraction, 4)], ["POAV fraction", fmt(r.POAV.fraction, 4), fmt(z.POAV.fraction, 4)],
      ["ASA m²/g", fmt(r.ASA.m2_g, 1), fmt(z.ASA.m2_g, 1)]]);
  } else if (r.zeopp_error) html += `<div class="insp-hint">Zeo++: ${esc(r.zeopp_error)}</div>`;
  (r.warnings || []).forEach((w) => { html += `<div class="insp-hint">${esc(w)}</div>`; });
  html += `<div class="insp-hint">The 3D view draws the volume the probe sphere can occupy, from the same classification as these numbers:
    accessible (POAV) and closed pockets (PONAV).</div>`;
  const po = poreOpts(), ck = (b) => (b ? " checked" : "");
  html += `<div class="an-row an-legend"><span class="an-swatch acc"></span><label><input type="checkbox" data-o="accessible"${ck(po.accessible)}> Accessible pores</label>
    <span class="an-swatch nac"></span><label><input type="checkbox" data-o="inaccessible"${ck(po.inaccessible)}> Inaccessible pockets</label>
    <label><input type="checkbox" data-o="colorPockets"${ck(po.colorPockets)}> Color pockets individually</label></div>
    <div class="an-row an-legend"><label>Opacity <input type="range" min="0.05" max="1" step="0.05" value="${po.opacity}" data-o="opacity"></label>
    <label>Style <select data-o="style">${MQ.STYLES.map(([v, t]) => `<option value="${v}"${po.style === v ? " selected" : ""}>${t}</option>`).join("")}</select></label>
    <button type="button" class="tb-btn-text" data-o="show"${r.render ? "" : " disabled title=\"The mesh data was deleted: compute again to draw it\""}>${shownFor(r) ? "Redraw" : "Show"} in 3D</button>
    <button type="button" class="tb-btn-text" data-o="json">Download JSON</button></div>`;
  out.innerHTML = html;
  // the options without a control here (shading, walls, mesh quality) keep their Settings
  const read = (o) => { const e = out.querySelector(`[data-o="${o}"]`); return !e ? poreOpts()[o] : e.type === "checkbox" ? e.checked : o === "opacity" ? Number(e.value) : e.value; };
  const set = (ev) => {
    const o = ev && ev.target && ev.target.dataset.o;
    if (o && PORE_KEYS[o]) { try { state.H.settings.set(PORE_KEYS[o][0], read(o)); } catch (e) { /* kept for this view only */ } }
    // the defaults changed here: this structure's surfaces take them (over the Inspector's own change of that option)
    const it = state.overlay.find("pores");
    if (it && o && PORE_KEYS[o]) state.overlay.override(it.key, { [o]: null });
    state.overlay.setDefaults("pores", Object.fromEntries(Object.keys(PORE_KEYS).map((k) => [k, read(k)])));
  };
  out.querySelectorAll('input[data-o], select[data-o]').forEach((e) => e.addEventListener(e.type === "range" ? "input" : "change", set));
  out.querySelector('[data-o="show"]').addEventListener("click", () => { showPores(r); set(); });
  out.querySelector('[data-o="json"]').addEventListener("click", () => {
    const { render: _, ...rest } = r;
    download(`${docStem()}_porosity.json`, JSON.stringify(rest, null, 1), "application/json");
  });
}

function sourceOptions(sel = "CuKa") {
  return Object.entries((state.info || {}).sources || {}).map(([k, s]) =>
    `<option value="${esc(k)}"${k === sel ? " selected" : ""}>${esc(k)} (${fmt(s.wavelength, 5)} Å)</option>`).join("");
}

function tabPxrd(root) {
  const out = form(root, `<div class="an-row"><label>Radiation</label><select data-p="source">${sourceOptions()}</select>
    <label>2θ from</label><input type="number" data-p="min" value="5" step="0.5" size="5"><label>to</label><input type="number" data-p="max" value="90" step="0.5" size="5">
    <label><input type="checkbox" data-p="compare"> Compare with the measured pattern</label></div>
    <details class="an-adv" data-tier="advanced"><summary>Peak shape and sample</summary><div class="an-row">
    <label>Step</label><input type="number" data-p="step" value="0.02" step="0.005">
    <label>B<sub>iso</sub> (Å²)</label><input type="number" data-p="b_iso" value="0" step="0.1">
    <label>U</label><input type="number" data-p="U" value="0" step="0.001"><label>V</label><input type="number" data-p="V" value="0" step="0.001">
    <label>W</label><input type="number" data-p="W" value="0.01" step="0.001"><label>η</label><input type="number" data-p="eta" value="0.5" step="0.05" min="0" max="1">
    <label>Crystallite size (nm)</label><input type="number" data-p="size_nm" step="1" min="0.5" placeholder="none"></div></details>`,
  "Simulate", async (p) => {
    const extra = {};
    if (!p.compare) { const saved = state.pattern; state.pattern = null; const r = await run("pxrd", p); state.pattern = saved; landed(r, out, () => renderPxrd(out, r)); return; }
    if (!state.pattern) { state.H.toast("Choose a measured pattern first", "info"); return; }
    const r = await run("pxrd", p, extra);
    landed(r, out, () => renderPxrd(out, r));
  });
  patternPicker(root.querySelector(".an-form"));
  if (res().pxrd) renderPxrd(out, res().pxrd);
}

function renderPxrd(out, r) {
  const series = [{ x: r.two_theta, y: r.intensity, cls: "calc" }];
  if (r.measured) {
    const m = Math.max(...r.measured.intensity) || 1;
    series.unshift({ x: r.measured.two_theta, y: r.measured.intensity.map((y) => 100 * y / m), cls: "obs" });
  }
  const sticks = r.peaks.map((p) => ({ x: p.two_theta, y: p.intensity, title: `${hklStr(p.hkl)} ${fmt(p.two_theta)}°` }));
  let html = `<div class="an-sub">${esc(r.formula)} · ${esc(r.source)} λ = ${fmt(r.wavelength, 6)} Å${r.wavelength2 ? ` (Kα2 ${fmt(r.wavelength2, 6)} Å)` : ""}</div>`;
  html += chart({ series, sticks, xlabel: TWO_THETA, ylabel: "Intensity" });
  if (r.similarity) html += `<div class="insp-hint">Against ${esc(r.measured.name || "the measured pattern")}: Pearson r = ${fmt(r.similarity.pearson, 4)}, misfit ${fmt(100 * r.similarity.rwp, 1)}%</div>`;
  html += `<div class="an-row"><button type="button" class="tb-btn-text" data-o="xy">Download Pattern (.xy)</button>
    <button type="button" class="tb-btn-text" data-o="csv">Download Peaks (.csv)</button></div>`;
  html += table(["2θ (°)", "d (Å)", "I", "hkl", "Multiplicity"], r.peaks.filter((p) => p.intensity >= 0.5).map((p) =>
    [fmt(p.two_theta), fmt(p.d, 4), fmt(p.intensity, 1), hklStr(p.hkl), p.multiplicity]), "num");
  out.innerHTML = html;
  out.querySelector('[data-o="xy"]').addEventListener("click", () => download(`${docStem()}_pxrd.xy`,
    `# 2theta intensity (MACE, ${r.source})\n` + r.two_theta.map((x, i) => `${x.toFixed(4)} ${r.intensity[i].toFixed(4)}`).join("\n") + "\n"));
  out.querySelector('[data-o="csv"]').addEventListener("click", () => download(`${docStem()}_peaks.csv`,
    "two_theta,d,intensity,h,k,l,multiplicity\n" + r.peaks.map((p) => [p.two_theta.toFixed(4), p.d.toFixed(5), p.intensity.toFixed(2), ...p.hkl, p.multiplicity].join(",")).join("\n") + "\n", "text/csv"));
}

function tabFit(root) {
  const out = form(root, `<div class="an-row"><label>Task</label><select data-p="task">
    <option value="peaks">Find peaks</option><option value="index_ab">Index (no cell)</option><option value="index_st">Index with this structure's cell</option>
    <option value="pawley">Pawley fit (this structure's cell)</option><option value="rietveld">Rietveld fit (this structure)</option></select>
    <label>Radiation</label><select data-p="source">${sourceOptions()}</select></div>
    <details class="an-adv" data-tier="advanced"><summary>Options</summary><div class="an-row">
    <label>Peak prominence</label><input type="number" data-p="prominence" value="0.02" step="0.005" min="0.001" max="0.9">
    <label>2θ tolerance (°)</label><input type="number" data-p="tol" value="0.05" step="0.005">
    <label>Fit range</label><input type="number" data-p="min" placeholder="all" size="5"><input type="number" data-p="max" placeholder="all" size="5">
    <label>Background terms</label><input type="number" data-p="background" value="6" min="1" max="20">
    <label><input type="checkbox" data-p="positions"> Refine atomic positions (Rietveld)</label>
    <label>Preferred orientation [hkl]</label><input type="text" data-p="po" placeholder="none" size="7"></div></details>`,
  "Run", async (p) => {
    if (!state.pattern) { state.H.toast("Choose a measured pattern first", "info"); return; }
    const task = p.task;
    let r;
    if (task === "peaks") r = await run("peaks", p);
    else if (task === "index_ab" || task === "index_st") r = await run("index", { ...p, mode: task === "index_st" ? "structure" : "ab_initio" });
    else r = await run("refine", { ...p, method: task });
    landed(r, out, () => renderFit(out, task, r));
  });
  patternPicker(root.querySelector(".an-form"));
  const last = res().fitLast;
  if (last && res()[last]) renderFit(out, last, res()[last]);
}

function renderFit(out, task, r) {
  let html = "";
  if (task === "peaks") {
    html += chart({ series: [{ x: r.two_theta, y: r.intensity, cls: "obs" }, { x: r.two_theta, y: r.background, cls: "bg" }],
      ticks: r.peaks.map((p) => ({ x: p.two_theta, title: fmt(p.two_theta) })), xlabel: TWO_THETA, ylabel: "Counts" });
    html += table(["2θ (°)", "d (Å)", "Height", "FWHM (°)", "η"], r.peaks.map((p) => [fmt(p.two_theta), fmt(p.d, 4), fmt(p.height, 1), fmt(p.fwhm, 4), fmt(p.eta, 2)]), "num");
  } else if (r.mode === "ab_initio") {
    html += `<div class="an-sub">${r.lines_used} lines · ${esc(r.note)}</div>`;
    html += table(["System", "Centering", "a", "b", "c", "α", "β", "γ", "V (Å³)", "M20", "Indexed"], r.solutions.map((s) =>
      [esc(s.system), esc(s.centering), fmt(s.lattice.a, 4), fmt(s.lattice.b, 4), fmt(s.lattice.c, 4), fmt(s.lattice.alpha, 2), fmt(s.lattice.beta, 2),
        fmt(s.lattice.gamma, 2), fmt(s.volume, 2), fmt(s.M20, 1), `${s.indexed}/${s.total}`]), "num");
    if (r.solutions[0]) html += table(["2θ (°)", "hkl"], r.solutions[0].lines.map((l) => [fmt(l.two_theta), hklStr(l.hkl)]), "num");
  } else if (r.mode === "structure") {
    const L = r.lattice;
    html += `<div class="an-sub">Indexed ${r.indexed} of ${r.total} lines · refined a ${fmt(L.a, 5)} b ${fmt(L.b, 5)} c ${fmt(L.c, 5)} Å, α ${fmt(L.alpha, 3)} β ${fmt(L.beta, 3)} γ ${fmt(L.gamma, 3)}° · M20 ${fmt(r.figures_of_merit.M20, 1)}</div>`;
    html += table(["2θ obs (°)", "hkl", "2θ calc (°)", "Δ"], r.lines.map((l) => [fmt(l.two_theta), hklStr(l.hkl), fmt(l.two_theta_calc), fmt(l.delta, 4)]), "num");
  } else {
    const L = r.lattice, E = r.lattice_esd || {}, St = r.statistics;
    const pm = (k, n) => `${fmt(L[k], n)}${E[k] ? ` (${fmt(E[k], n)})` : ""}`;
    html += `<div class="an-sub">${esc(r.method)} · ${r.converged ? "converged" : "not converged"} · R<sub>p</sub> ${fmt(100 * St.Rp, 2)}% · R<sub>wp</sub> ${fmt(100 * St.Rwp, 2)}% · R<sub>exp</sub> ${fmt(100 * St.Rexp, 2)}% · GOF ${fmt(St.GOF, 3)}</div>`;
    const lo = Math.min(...r.observed), hi = Math.max(...r.observed), off = lo - 0.15 * (hi - lo);
    html += chart({ series: [{ x: r.two_theta, y: r.observed, cls: "obs" }, { x: r.two_theta, y: r.calculated, cls: "calc" },
      { x: r.two_theta, y: r.difference.map((x) => x + off), cls: "diff" }],
    ticks: r.reflections.map((t) => ({ x: t.two_theta, title: `${hklStr(t.hkl)} ${fmt(t.two_theta)}°` })), xlabel: TWO_THETA, ylabel: "Counts", h: 280 });
    html += table(["a (Å)", "b (Å)", "c (Å)", "α (°)", "β (°)", "γ (°)", "V (Å³)", "zero (°)"],
      [[pm("a", 5), pm("b", 5), pm("c", 5), pm("alpha", 3), pm("beta", 3), pm("gamma", 3), pm("volume", 3), fmt(r.zero, 4)]], "num");
    html += table(["U", "V", "W", "η"].concat(r.method === "rietveld" ? ["scale", "B (Å²)"] : []),
      [[fmt(r.profile.U, 5), fmt(r.profile.V, 5), fmt(r.profile.W, 5), fmt(r.profile.eta, 3)].concat(r.method === "rietveld" ? [Number(r.scale).toPrecision(5), fmt(r.B, 3)] : [])], "num");
    if (r.frac) html += table(["Atom", "x", "y", "z"], r.frac.map((f, i) => [esc(r.symbols[i]), fmt(f[0], 5), fmt(f[1], 5), fmt(f[2], 5)]), "num");
    html += `<div class="an-row"><button type="button" class="tb-btn-text" data-o="csv">Download Fit (.csv)</button></div>`;
  }
  out.innerHTML = html;
  const b = out.querySelector('[data-o="csv"]');
  if (b) b.addEventListener("click", () => download(`${docStem()}_${r.method}.csv`, "two_theta,observed,calculated,background,difference\n" +
    r.two_theta.map((x, i) => [x.toFixed(4), r.observed[i], r.calculated[i].toFixed(3), r.background[i].toFixed(3), r.difference[i].toFixed(3)].join(",")).join("\n") + "\n", "text/csv"));
}

function tabTopology(root) {
  const rc = (state.info || {}).rcsr || {};
  const out = form(root, `<div class="an-row"><label>Simplification</label><select data-p="mode">
    <option value="auto">Automatic</option><option value="sbu">Building units (MOF)</option><option value="atoms">Atoms (zeolites, inorganic)</option><option value="all">All atoms, no simplification</option></select>
    <label>Bond tolerance</label><input type="number" data-p="tolerance" value="1.15" step="0.05" min="0.8" max="1.6"></div>
    <div class="insp-hint">Reference nets: ${rc.bundled ? `${rc.bundled} from RCSR (bundled), ` : ""}${(rc.builtin || []).length} built in${rc.imported ? `, ${rc.imported} imported (tried first)` : ""}.
    ${rc.imported ? "" : `For a newer RCSR list, download <a href="${esc(rc.download || "http://rcsr.net/data/3dall.txt")}" target="_blank" rel="noopener">3dall.txt</a> and import it:`}
    <input type="text" data-k="rpath" placeholder="3dall.txt in the folder" size="18"> <button type="button" class="tb-btn-text" data-k="rimport">Import</button></div>`,
  "Find Topology", async (p) => {
    const r = await run("topology", p);
    landed(r, out, () => renderTopology(out, r));
  });
  root.querySelector('[data-k="rimport"]').addEventListener("click", async () => {
    const path = root.querySelector('[data-k="rpath"]').value.trim();
    if (!path) return;
    try {
      const r = await state.H.api("analysis/rcsr/import", { body: { path } });
      state.H.toast(`Imported ${r.imported} nets`, "success");
      state.info = await state.H.api("analysis/info");
      render();
    } catch (e) { state.H.toast(`Import: ${e.message}`, "error"); }
  });
  if (res().topology) renderTopology(out, res().topology);
}

function renderTopology(out, r) {
  const s = r.summary;
  let html = `<div class="an-sub">${esc(r.mode)} simplification · ${r.vertices} vertices, ${r.edges} edges · topology <b>${esc(s.topologies.join(", "))}</b>
    · interpenetration ${s.interpenetration == null ? "n/a" : s.interpenetration}${s.note ? ` · ${esc(s.note)}` : ""}</div>`;
  r.nets.forEach((n, i) => {
    const alt = n.matches.slice(1).map((m) => m.name).join(", ");
    html += `<div class="an-sub">Component ${i + 1}: ${n.periodicity}-periodic, ${esc(n.topology || "not in the reference set")}${alt ? ` (same sequences: ${esc(alt)})` : ""}${n.copies > 1 ? `, ${n.copies} interpenetrating copies` : ""}, td10 ${fmt(n.td10, 1)}</div>`;
    html += table(["Count", "CN", "Coordination sequence", "Point symbol", "Made of"], n.vertex_classes.map((c) =>
      [c.count, c.cn, c.cs.join(" "), esc(`{${c.point_symbol || ""}}`), esc(c.labels.slice(0, 4).join(", "))]));
  });
  html += `<div class="insp-hint">Identification compares coordination sequences (10 shells); a few different nets share them.</div>
    ${r.citation ? `<div class="insp-hint" data-o="citation">Net symbols and reference nets: <a href="http://rcsr.net" target="_blank" rel="noopener">RCSR</a>. Please cite ${esc(r.citation).replace(/doi:(\S+)$/, (m, d) => `<a href="https://doi.org/${d}" target="_blank" rel="noopener">doi:${d}</a>`)}</div>` : ""}
    <div class="an-row"><button type="button" class="tb-btn-text" data-o="net">Show Net in 3D</button></div>`;
  out.innerHTML = html;
  out.querySelector('[data-o="net"]').addEventListener("click", () => {
    if (!showOverlay(r)) state.H.toast("No cell to draw the net in", "info");
  });
}

function tabPolyhedra(root) {
  const out = form(root, `<div class="an-row"><label>Bond tolerance</label><input type="number" data-p="tolerance" value="1.15" step="0.05" min="0.8" max="1.6">
    <button type="button" class="tb-btn-text" data-k="toggle">Toggle Polyhedra in 3D</button></div>`,
  "Analyze Polyhedra", async (p) => {
    const r = await run("polyhedra", p);
    landed(r, out, () => renderPolyhedra(out, r));
  });
  root.querySelector('[data-k="toggle"]').addEventListener("click", () => state.H.run("view.polyhedra"));
  if (res().polyhedra) renderPolyhedra(out, res().polyhedra);
}

function renderPolyhedra(out, r) {
  if (!r.polyhedra.length) { out.innerHTML = `<div class="insp-hint">No polyhedra: no cation has 4 to 12 anion ligands.</div>`; return; }
  let html = table(["Center", "CN", "Shape (CShM)", "Count", "⟨d⟩ (Å)", "V (Å³)", "Δ"], r.summary.map((s) =>
    [esc(s.element), s.cn, esc(s.shape || "–"), s.count, fmt(s.bond_mean, 4), fmt(s.volume, 3), fmt(s.distortion_index, 5)]), "num");
  html += `<details class="an-adv" data-tier="advanced" open><summary>Every polyhedron</summary>` +
    table(["Atom", "CN", "Shape", "CShM", "d min–max (Å)", "V (Å³)", "Δ", "⟨λ⟩", "σ² (°²)", "ECoN"], r.polyhedra.map((p) =>
      [`${esc(p.element)}${p.index + 1}`, p.cn, esc(p.shape || "–"), fmt(p.cshm, 2), `${fmt(p.bond_min, 4)}–${fmt(p.bond_max, 4)}`, fmt(p.volume, 3),
        fmt(p.distortion_index, 5), fmt(p.quadratic_elongation, 4), fmt(p.angle_variance, 2), fmt(p.econ, 3)]), "num") + `</details>
    <div class="insp-hint">Δ: Baur distortion index · ⟨λ⟩, σ²: quadratic elongation and bond-angle variance (tetrahedra, octahedra) · CShM: 0 is the ideal shape.</div>`;
  out.innerHTML = html;
}

function tabBonds(root) {
  const out = form(root, `<div class="an-row"><label>Bond tolerance</label><input type="number" data-p="tolerance" value="1.15" step="0.05" min="0.8" max="1.6">
    <label>Bins</label><input type="number" data-p="bins" value="40" min="5" max="400"></div>`,
  "Compute Distributions", async (p) => {
    const r = await run("bonds", p);
    landed(r, out, () => renderBonds(out, r));
  });
  if (res().bonds) renderBonds(out, res().bonds);
}

function renderBonds(out, r) {
  const pairs = Object.keys(r.bonds), trip = Object.keys(r.angles);
  let html = `<div class="an-sub">${r.n_bonds} bonds · ${r.n_angles} angles</div>
    <div class="an-row"><label>Bond</label><select data-o="pair"><option value="">all</option>${pairs.map((k) => `<option>${esc(k)}</option>`).join("")}</select>
    <label>Angle</label><select data-o="trip"><option value="">all</option>${trip.map((k) => `<option>${esc(k)}</option>`).join("")}</select></div>
    <div data-o="h1"></div><div data-o="h2"></div>`;
  html += table(["Bond", "n", "min", "mean", "max", "σ"], pairs.map((k) => { const v = r.bonds[k]; return [esc(k), v.n, fmt(v.min, 4), fmt(v.mean, 4), fmt(v.max, 4), fmt(v.std, 4)]; }), "num");
  html += table(["Angle", "n", "min", "mean", "max", "σ"], trip.map((k) => { const v = r.angles[k]; return [esc(k), v.n, fmt(v.min, 2), fmt(v.mean, 2), fmt(v.max, 2), fmt(v.std, 2)]; }), "num");
  out.innerHTML = html;
  const draw = () => {
    const p = out.querySelector('[data-o="pair"]').value, t = out.querySelector('[data-o="trip"]').value;
    const hb = p ? r.bonds[p].histogram : r.bond_histogram, ha = t ? r.angles[t].histogram : r.angle_histogram;
    out.querySelector('[data-o="h1"]').innerHTML = hb ? chart({ bars: hb, xlabel: L.withUnit("Bond length", "A"), ylabel: "Count", h: 180 }) : "";
    out.querySelector('[data-o="h2"]').innerHTML = ha ? chart({ bars: ha, xlabel: L.withUnit("Angle", "deg"), ylabel: "Count", h: 180, xr: [0, 180] }) : "";
  };
  out.querySelectorAll("select[data-o]").forEach((e) => e.addEventListener("change", draw));
  draw();
}

function tabMorphology(root) {
  const out = form(root, `<div class="an-row"><label>Model</label><select data-p="model"><option value="bfdh">BFDH (from the structure)</option><option value="wulff">Wulff (surface energies)</option></select>
    <label>Surface energies</label><input type="text" data-p="energies" placeholder="100:1.2, 111:0.9, 110:1.05" size="26">
    <label>Max index</label><input type="number" data-p="max_index" value="2" min="1" max="4"></div>`,
  "Build Shape", async (p) => {
    if (p.model === "bfdh") p.energies = "";
    else if (!p.energies) { state.H.toast("Give surface energies for a Wulff shape", "info"); return; }
    const r = await run("morphology", p);
    if (!r) return;
    showOverlay(r);
    landed(r, out, () => renderMorphology(out, r));
  });
  if (res().morphology) renderMorphology(out, res().morphology);
}

function renderMorphology(out, r) {
  let html = `<div class="an-sub">${esc(r.method.toUpperCase())} · aspect ${r.aspect.map((x) => fmt(x, 2)).join(" : ")}</div>`;
  html += table(["(hkl)", "Faces", "Area %", "Distance"], r.families.map((f) => [hklStr(f.hkl), f.multiplicity, fmt(100 * f.area_fraction, 2), fmt(f.distance, 4)]), "num");
  html += `<div class="an-row"><button type="button" class="tb-btn-text" data-o="show">Show Shape in 3D</button>
    <button type="button" class="tb-btn-text" data-o="obj">Download .obj</button></div>`;
  out.innerHTML = html;
  out.querySelector('[data-o="show"]').addEventListener("click", () => showOverlay(r));
  out.querySelector('[data-o="obj"]').addEventListener("click", () => download(`${docStem()}_morphology.obj`,
    "# MACE crystal morphology\n" + r.vertices.map((v) => `v ${v.map((x) => x.toFixed(6)).join(" ")}`).join("\n") + "\n" +
    r.triangles.map((t) => `f ${t[0] + 1} ${t[1] + 1} ${t[2] + 1}`).join("\n") + "\n"));
}

function tabPlanes(root) {
  const out = form(root, `<div class="an-row"><label>(hkl)</label><input type="text" data-p="hkl" value="1 1 1" size="8">
    <label>Position</label><input type="range" data-p="offset" min="0" max="1" step="0.01" value="0"><span data-k="offv">0.00</span>
    <label>Atoms within (Å)</label><input type="number" data-p="tol" value="0.3" step="0.05" min="0"></div>`,
  "Show Plane", async (p) => {
    const r = await run("planes", p);
    landed(r, out, () => renderPlane(out, r));
  });
  const f = root.querySelector(".an-form");
  const slider = f.querySelector('[data-p="offset"]');
  let timer = 0;
  slider.addEventListener("input", () => {
    f.querySelector('[data-k="offv"]').textContent = Number(slider.value).toFixed(2);
    clearTimeout(timer);
    timer = setTimeout(() => f.requestSubmit(), 250);
  });
  const rank = document.createElement("div");
  rank.className = "an-row";
  rank.innerHTML = `<button type="button" class="tb-btn-text" data-k="cleave">Rank Cleavage Planes</button><span class="insp-hint">fewest bonds broken per area, (hkl) up to 2</span>`;
  root.appendChild(rank);
  const rout = document.createElement("div");
  root.appendChild(rout);
  rank.querySelector('[data-k="cleave"]').addEventListener("click", async () => {
    const r = await run("cleavage", { max_index: 2 });
    landed(r, rout, () => renderCleavage(rout, r));
  });
  if (res().planes) renderPlane(out, res().planes, false);
  if (res().cleavage) renderCleavage(rout, res().cleavage);
}

function renderCleavage(rout, r) {
  rout.innerHTML = table(["(hkl)", "d (Å)", "Bonds/nm²", "Best position"], r.families.map((x) => [hklStr(x.hkl), fmt(x.d, 4), fmt(x.min_per_nm2, 3), fmt(x.best_offset, 3)]), "num");
}

function renderPlane(out, r, draw = true) {
  if (draw) showOverlay(r);
  const b = r.broken;
  out.innerHTML = `<div class="an-sub">${hklStr(r.hkl)} · d = ${fmt(r.d, 5)} Å · ${r.planes.length} plane(s) cross the cell · ${r.atoms_on_plane.length} atom(s) on it</div>` +
    chart({ series: [{ x: b.offsets, y: b.profile, cls: "calc" }], ticks: [{ x: ((r.offset % 1) + 1) % 1, title: "here" }],
      xlabel: "Plane position (fraction of d)", ylabel: L.withUnit("Bonds broken", "nm-2"), h: 160 }) +
    `<div class="insp-hint">Fewest bonds broken: ${fmt(b.min_per_nm2, 3)}/nm² at position ${fmt(b.best_offset, 3)}</div>`;
}
