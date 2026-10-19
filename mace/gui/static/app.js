/* MACE GUI classic views front end. Plain browser JS, no build step.
 * All URLs are relative so the app also works behind a path prefix. */
(() => {
"use strict";
window.__maceErrors = window.__maceErrors || [];
window.addEventListener("error", (e) => window.__maceErrors.push(String(e.message)));
/* for Copy Diagnostics: the last 30 errors (message, file name:line) and failed requests (route, status, ms) */
window.__maceErrorLog = [];
window.__maceFailedCalls = [];
window.addEventListener("error", (e) => {
  const log = window.__maceErrorLog;
  log.push({ message: String(e.message).slice(0, 300), at: `${String(e.filename || "").split("/").pop()}:${e.lineno || 0}` });
  if (log.length > 30) log.shift();
});
window.addEventListener("unhandledrejection", (e) => window.__maceErrors.push(String(e.reason)));

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
const HARTREE_MEV = 27211.386245988;
const KIND_LABEL = { cif: "CIF", d12: "d12", d3: "d3", out: "out", band: "band", dos: "dos", ir: "IR", raman: "Ram",
  db: "db", json: "json", image: "img", html: "html", freq: "FREQ",
  // kinds hidden before plan E1, and the content chips of outputs and logs
  cube: "cube", orbitals: "orb", xyz: "xyz", poscar: "vasp", mol: "mol", pdb: "pdb", gen: "gen", fort34: "f34", f25: "f25",
  transport: "trn", crystaldata: "data", wavefunction: "wf", script: "sh", log: "log", joblog: "log",
  OPT: "OPT", SP: "SP", FREQ: "FREQ", BAND: "BAND", DOSS: "DOSS", "CHARGE+POTENTIAL": "CHG+POT", TRANSPORT: "TRANSP", ORBITALS: "ORB",
  PROPERTIES: "PROPS", props: "PROPS" };
const FILTERS = [
  ["all", "All", null],
  ["struct", "Structures", ["cif", "d12", "out", "xyz", "poscar", "mol", "pdb", "gen", "fort34"]],
  // Properties (spec §1.1): band, DOS, IR and Raman data, FREQ outputs (by their index: the details), plots,
  // volumes, fort.25, transport data and properties outputs (by content)
  ["results", "Properties", ["band", "dos", "ir", "raman", "image", "html", "freq", "cube", "orbitals", "f25", "transport", "props"]],
  ["setup", "Setup", ["json", "d12", "d3"]],
  ["db", "Databases", ["db"]],
  ["logs", "Logs & data", ["joblog", "script", "crystaldata", "wavefunction", "log"]],
];

const state = {
  files: [], filter: "all", text: "", active: null,
  structure: null, traj: null, frame: 0, cmp: null, layout: "overlay",
  viewers: {}, templates: [], db: null, colors: null,
};

/* ---------------------------------------------------------------- utils */
/* One request to the server. opts: body (JSON, POST), raw (a Blob or
 * string, POST as application/octet-stream), headers, signal (an
 * AbortController's), blob (the answer as a Blob, for downloads), withStatus
 * ({status, data, headers}: a 202 "detached" task is not an error). The
 * page's event-stream client id goes with it (builds and drops are leased to it). */
async function api(path, opts = {}) {
  const init = { credentials: "same-origin",
    headers: { "X-Requested-With": "mace-gui", ...(opts.headers || {}) } };
  if (window.__maceClient) init.headers["X-Mace-Client"] = window.__maceClient;
  // the served folder this page shows: a write after Open Folder in another tab gets 409
  if (window.__maceRootGen !== undefined) init.headers["X-Mace-Root-Gen"] = window.__maceRootGen;
  if (opts.signal) init.signal = opts.signal;
  if (opts.raw !== undefined) {
    init.method = "POST";
    init.headers["Content-Type"] = "application/octet-stream";
    init.body = opts.raw;
  } else if (opts.body !== undefined) {
    init.method = "POST";
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(opts.body);
  }
  const t0 = performance.now();
  let res;
  try { res = await fetch("api/" + path, init); }
  catch (e) { noteFailed(init.method || "GET", path, 0, t0); throw e; }
  if (!res.ok) noteFailed(init.method || "GET", path, res.status, t0);
  if (opts.blob && res.ok) return { blob: await res.blob(), headers: res.headers, status: res.status };
  let data;
  try { data = await res.json(); } catch (e) { data = { error: res.statusText }; }
  if (!res.ok) {
    const err = new Error(data.error || data.message || res.statusText);
    err.status = res.status; err.data = data;
    if (res.status === 403 && window.__maceLost && /token/.test(data.error || "")) window.__maceLost();
    // a session that expired without use: the Signed Out dialog, not "lost the connection"
    if (res.status === 401 && data.expired && window.__maceExpired) window.__maceExpired();
    throw err;
  }
  return opts.withStatus ? { status: res.status, data, headers: res.headers } : data;
}
/* A failed request as METHOD route-template status ms: never the query or a path in it. */
function noteFailed(method, path, status, t0) {
  const route = "/api/" + String(path).split("?")[0].split("/").map((x) => (/^[a-z][a-z0-9_-]*$/i.test(x) && !/\d{3,}/.test(x) ? x : "*")).join("/");
  const log = window.__maceFailedCalls;
  log.push(`${method} ${route} ${status} ${Math.round(performance.now() - t0)}ms`);
  if (log.length > 30) log.shift();
}
const q = (o) => Object.entries(o).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join("&");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
/* toFixed, but a value that rounds to zero prints "0.000", never "-0.000" */
const fmt = (x, d = 4) => {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "—";
  const s = Number(x).toFixed(d);
  return /^-0\.?0*$/.test(s) ? s.slice(1) : s;
};
const signed = (x, d = 4) => {
  if (x === null || x === undefined) return "—";
  const s = fmt(x, d), zero = /^0\.?0*$/.test(s);
  return `<span class="${zero ? "" : (x > 0 ? "pos" : "neg")}">${!zero && x > 0 ? "+" : ""}${s}</span>`;
};
/* Hermann-Mauguin symbols as printed in the International Tables: spglib's
 * "Fm-3m" -> Fm3&#772;m (bar over the 3), "P2_1/c" -> P2<sub>1</sub>/c. */
const hm = (sym) => esc(sym || "").replace(/-(\d)/g, '<span class="bar">$1</span>')
  .replace(/_(\d)/g, "<sub>$1</sub>");
function toast(msg, err = false) {
  if (window.__maceToast) return window.__maceToast(msg, err ? "error" : "info");
  const t = $("#toast");
  t.textContent = msg;
  t.classList.toggle("err", err);
  t.classList.remove("hidden");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add("hidden"), err ? 7000 : 3000);
}
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
function sizeLabel(n) {
  if (n < 1024) return n + " B";
  if (n < 1048576) return (n / 1024).toFixed(0) + " KB";
  return (n / 1048576).toFixed(1) + " MB";
}
/* A Settings value (the shell installs MACE.setting once it boots); before
 * that, or for a value that is not usable, the given fallback. */
function setting(key, fallback) {
  const get = window.MACE && window.MACE.setting;
  const v = get ? get(key) : undefined;
  return v === undefined || v === null || v === "" ? fallback : v;
}
const num = (key, fallback) => { const v = Number(setting(key, fallback)); return Number.isFinite(v) ? v : fallback; };
/* The viewport background: the theme's, white, black or a custom color. */
function viewerBg() {
  const b = setting("background", "theme");
  if (b === "white") return cssVar("--viewport-white");
  if (b === "black") return cssVar("--viewport-black");
  if (b === "custom" && /^#[0-9a-f]{6}$/i.test(setting("backgroundColor", ""))) return setting("backgroundColor", "");
  return cssVar("--viewer-bg");
}
/* Forced surfaces (1.1.6, tokens.css): Settings › Appearance › Plot background gives the plots a
 * light or dark surface of their own (data-plot-surface on the root); a White, Black or custom 3D
 * background of the other lightness from the theme gives the 3D view's overlays the token set of
 * that lightness (data-vp-surface), so the triad, labels, cell and selection stay legible on it.
 * plotVar and vpVar read a token as the plots and the 3D view draw it (a hidden probe in each scope). */
function bgLightness() {
  const b = setting("background", "theme"), c = setting("backgroundColor", "");
  const hex = b === "white" ? cssVar("--viewport-white") : b === "black" ? cssVar("--viewport-black") : b === "custom" && /^#[0-9a-f]{6}$/i.test(c) ? c : "";
  if (!hex) return "";
  const n = parseInt(hex.slice(1), 16);
  return (0.299 * (n >> 16) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255)) / 255 > 0.5 ? "light" : "dark";
}
function attr(el, k, v) { if ((el.getAttribute(k) || "") !== v) { if (v) el.setAttribute(k, v); else el.removeAttribute(k); } }
function syncSurfaces() {
  const root = document.documentElement, p = setting("plots.surface", "theme");
  attr(root, "data-plot-surface", p === "light" || p === "dark" ? p : "");
  const bg = bgLightness(), scheme = getComputedStyle(root).colorScheme === "dark" ? "dark" : "light";
  attr(root, "data-vp-surface", bg && bg !== scheme ? bg : "");
  // Settings › Appearance › Plot style (tokens.css): the publication series and ink, by the plot surface's lightness
  attr(root, "data-plot-style", setting("plots.style", "default") !== "default" ? String(setting("plots.style", "default")) : "");
  attr(root, "data-plot-tone", p === "light" || p === "dark" ? p : scheme);
}
/* Settings › Appearance › Plot style (1.1.6): a preset per plot family (mace/plotting/presets.py,
 * js/plotpresets.js, which the shell installs as window.MACE_PRESETS): "plots.style" is the preset of
 * every family ("Same for all"), "plots.style.<family>" a family's own ("same": follow it). A family a
 * preset does not cover (VESTA-like is for maps and volumes) draws Default. */
const PP = () => window.MACE_PRESETS || null;
function plotPreset(family) {
  const all = setting("plots.style", "default"), own = family ? setting(`plots.style.${family}`, "same") : "same";
  const id = own && own !== "same" ? own : all, pp = PP();
  if (pp) return pp.resolveId(id, family || "band");
  return id === "publication" ? id : "default";
}
const presetOf = (family) => { const pp = PP(); return pp ? pp.get(plotPreset(family), family || "band") : null; };
const plotTone = () => (document.documentElement.getAttribute("data-plot-tone") === "dark" ? "dark" : "light");
const probes = {};
function probeVar(kind, name, preset) {
  syncSurfaces();
  const k = preset ? `${kind}:${preset}` : kind;
  let el = probes[k];
  if (!el || !el.isConnected) {
    el = probes[k] = document.createElement("div"); el.className = `${kind}-probe`; el.hidden = true;
    if (preset) el.setAttribute("data-plot-preset", preset);
    document.body.appendChild(el);
  }
  return getComputedStyle(el).getPropertyValue(name).trim();
}
/* a token as a plot of `family` draws it: its preset's series (tokens.css), on the theme's or the forced surface */
const plotVar = (name, family) => probeVar("plot", name, plotPreset(family));
/* a token as a plot in preset `pid` draws it (a plot's own Plot style, Plot options › Style) */
const plotVarIn = (name, pid) => probeVar("plot", name, pid || "default");
const vpVar = (name) => probeVar("vp", name);
/** A Plotly layout's paper and plot colors: see-through on the theme's own surface, else the forced one (exports match). */
function plotPaper() {
  const p = setting("plots.surface", "theme");
  const c = p === "light" || p === "dark" ? plotVar("--plot-bg") : "rgba(0,0,0,0)";
  return { paper_bgcolor: c, plot_bgcolor: c };
}
/* The spin pair of a family's plots: Settings › Plots › Spin colors (a presets.py SPIN_SCHEMES scheme; Auto:
 * the plot style's, Blue/red by default), the custom Spin up/down colors (with Custom, or alone with Auto)
 * and the spin-down line style (dashed). */
function spinStyle(family = "band") {
  const pp = PP(), tone = plotTone();
  const hex = (k) => { const v = setting(k, ""); return /^#[0-9a-f]{6}$/i.test(v) ? v : ""; };
  const scheme = setting("plots.spinColors", "auto");
  const pair = pp ? pp.spinColors(plotPreset(family), family, tone, scheme, hex("plots.spinUp"), hex("plots.spinDown"))
    : { up: plotVar("--plot-1", family), down: plotVar("--plot-2", family) };      // presets not loaded yet: the theme's series
  return { up: pair.up, down: pair.down, dash: setting("plots.spinDownDash", "dash") };
}
/* Spin and difference volumes: + and − (the 3D view and the slices agree: + blue, − red by default,
 * yellow and cyan in VESTA-like) */
function volumeColors() {
  const P = presetOf("maps"), dark = plotTone() === "dark";
  if (!P || plotPreset("maps") === "default") return [cssVar("--plot-10"), cssVar("--plot-1")];
  if (dark && plotPreset("maps") === "grayscale") return [P.iso.minus, P.iso.plus];
  return [P.iso.plus, P.iso.minus];
}
const mapScale = (signed) => { const pp = PP(); return pp ? pp.mapScale(plotPreset("maps"), signed)
  : { colorscale: signed ? "RdBu" : "Viridis", reversescale: !!signed, contours: "both" }; };
/* Every plot builder passes its figure through here before plotly draws it (the screen and the PNG/SVG
 * exports alike), with its family. Default: unchanged. Publication: the axes, frame and ticks in ink
 * (black on a light plot), outside ticks, larger tick labels, no decorative grid (an axis that asks for its
 * grid, the band structure's high-symmetry k-points, keeps it, in ink), lines a quarter heavier.
 * Presentation: larger text, thicker lines and frame. Print grayscale: Publication's frame, and the series
 * of spectra and run charts told apart by line style and markers. The series colors come from tokens.css. */
const isPublication = (family) => plotPreset(family) === "publication";
function styleFigure(traces, layout, family, pid = null) {
  const pp = PP();
  if (!pp || !pp.styleFigure) return { traces, layout };
  // js/plotpresets.js styleFigure (mace/plotting/plotly_bands.py style_figure draws mace plotting's the same way);
  // pid: a plot's own preset (Plot options › Style) over its family's
  const id = pid ? pp.resolveId(pid, family) : plotPreset(family);
  return pp.styleFigure(traces, layout, family, id, (n) => (pid ? plotVarIn(n, id) : plotVar(n, family)));
}
function store(key, val) {
  try { if (val === undefined) return localStorage.getItem(key); localStorage.setItem(key, val); }
  catch (e) { return null; }
}

/* ---------------------------------------------------------------- theme */
function applyTheme(t) {
  if (t === "system") document.documentElement.removeAttribute("data-theme");
  else if (t) document.documentElement.setAttribute("data-theme", t);
  syncSurfaces();
  Object.values(state.viewers).forEach((v) => {
    if (v) { v.setBackgroundColor(viewerBg()); v.render(); }
  });
  if (state.traj) drawEnergy();
  document.dispatchEvent(new CustomEvent("mace-theme"));
}
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => applyTheme());

/* ---------------------------------------------------------------- tabs */
function showTab(name) {
  $$(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + name));
  requestAnimationFrame(() => Object.values(state.viewers).forEach((v) => {
    if (!v) return;
    if (v.visible()) { clearTimeout(v.freeTimer); v.freeTimer = 0; if (v.freed) rebuild(v); }
    else if (!v.freed && !v.freeTimer) v.freeTimer = setTimeout(() => freeViewer(v), freeAfter());
    v.resize();
    if (v.pendingFit && v.visible()) { const f = v.pendingFit; v.pendingFit = null; f(); }
    v.render();
  }));
  if (name === "opt" && state.traj) drawEnergy();
  document.dispatchEvent(new CustomEvent("mace-tab", { detail: name }));
}
$$(".tab").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function markTab(name, on) {
  const b = $(`.tab[data-tab="${name}"]`);
  if (!b) { document.dispatchEvent(new CustomEvent("mace-mark", { detail: { name, on } })); return; }
  const dot = b.querySelector(".dot");
  if (on && !dot) b.insertAdjacentHTML("beforeend", '<span class="dot"></span>');
  if (!on && dot) dot.remove();
}

/* A viewer hidden for a minute lets go of its models, shapes and labels
 * (spec §8.2.3: all viewers share one WebGL context, so this is what
 * "freeing" holds back); its camera is kept, and it is drawn again, in the
 * same view, the next time it shows. window.__maceViewerIdleMs (tests)
 * shortens the minute. */
const freeAfter = () => (Number(window.__maceViewerIdleMs) > 0 ? Number(window.__maceViewerIdleMs) : 60000);
function freeViewer(v) {
  v.freeTimer = 0;
  if (v.visible()) return;
  v.keptView = v.getView();
  v.removeAllModels(); v.removeAllShapes(); v.removeAllLabels();
  v.freed = true;
  document.dispatchEvent(new CustomEvent("mace-viewer-freed", { detail: v.viewerId }));
}
function rebuild(v) {
  v.freed = false;
  const t0 = performance.now(), view = v.keptView;
  const done = (e) => {
    if (e.detail !== v.viewerId) return;
    document.removeEventListener("mace-painted", done);
    window.__maceRebuildMs = performance.now() - t0;
  };
  document.addEventListener("mace-painted", done);
  if (v.viewerId === "viewer-main") { if (state.structure) drawStructure(true); }
  else drawOpt();
  if (view) v.setView(view);
  v.render();
}

/* ---------------------------------------------------------------- files */
function renderFilterChips() {
  $("#kind-chips").innerHTML = FILTERS.map(([k, label, kinds]) => {
    const n = kinds ? state.files.filter((f) => matches(f, kinds)).length : state.files.length;
    return `<button class="kind-chip ${state.filter === k ? "active" : ""}" data-f="${k}">${label} ${n}</button>`;
  }).join("");
  $$(".kind-chip").forEach((b) => b.addEventListener("click", () => {
    state.filter = b.dataset.f; renderFiles();
  }));
}
/* Settings › Files › Exclude: shell-style globs ("*.bak") matched against
 * the name and each folder of the path; the list is filtered here, in the
 * browser, so the server's listing stays one request. */
let excludeKey = null, excludeRes = [];
function excluded(path) {
  const pats = setting("explorerExclude", []);
  const key = JSON.stringify(pats);
  if (key !== excludeKey) {
    excludeKey = key;
    excludeRes = (Array.isArray(pats) ? pats : []).filter((g) => typeof g === "string" && g.trim()).map((g) =>
      new RegExp("^" + g.trim().replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*").replace(/\?/g, ".") + "$"));
  }
  return excludeRes.length > 0 && path.split("/").some((part) => excludeRes.some((re) => re.test(part)));
}
function renderFiles() {
  renderFilterChips();
  const kinds = FILTERS.find((f) => f[0] === state.filter)[2];
  const text = state.text.toLowerCase();
  const shown = state.files.filter((f) => (!kinds || matches(f, kinds))
    && (!text || f.path.toLowerCase().includes(text)) && !excluded(f.path));
  let html = "", folder = null;
  for (const f of shown) {
    const dir = f.path.includes("/") ? f.path.slice(0, f.path.lastIndexOf("/")) : "this folder";
    if (dir !== folder) { folder = dir; html += `<div class="folder" data-dir="${esc(f.path.includes("/") ? dir : "")}" title="Right-click: Batch Setup, Submit Folder, Create or Update Database from This Folder…">${esc(dir)}</div>`; }
    const det = detailText(f);
    html += `<button class="file ${state.active === f.path ? "active" : ""}${state.selected && state.selected.has(f.path) ? " msel" : ""}" data-path="${esc(f.path)}"${state.selected && state.selected.has(f.path) ? ' aria-selected="true"' : ""} title="${esc(f.path)} · ${sizeLabel(f.size)}${det ? " · " + esc(det) : ""}">
      ${chipHtml(f)}${middleName(f.name)}${det ? `<span class="file-det">${esc(det.split(" · ").slice(0, 2).join(" · "))}</span>` : ""}</button>`;
  }
  $("#file-list").innerHTML = html || '<p class="muted" style="padding:12px">No matching files.</p>';
  fitDetails();
}
/* A row's details show whole or not at all: a cut "Ag2Br3 · 3…" or a lone
 * "A" beside a long name reads as noise. One layout read for the list. */
function fitDetails() {
  const dets = [...document.querySelectorAll("#file-list .file-det")];
  dets.forEach((el) => el.classList.remove("cut"));
  const cut = dets.filter((el) => el.scrollWidth > el.clientWidth + 1);
  cut.forEach((el) => el.classList.add("cut"));
}
if (window.ResizeObserver && $("#file-list")) {
  let fitFrame = 0, lastW = 0;
  new ResizeObserver((entries) => {
    const w = Math.round(entries[0].contentRect.width);
    if (w === lastW) return;
    lastW = w;
    cancelAnimationFrame(fitFrame);
    fitFrame = requestAnimationFrame(fitDetails);
  }).observe($("#file-list"));
}
/* A long name loses its middle, not its end (the end tells runs apart):
 * the head shrinks with an ellipsis, the last 14 characters always show. */
function middleName(name) {
  const cut = name.length > 22 ? name.length - 14 : name.length;
  return `<span class="nm mid"><span class="mid-a">${esc(name.slice(0, cut))}</span>` +
    `<span class="mid-b">${esc(name.slice(cut))}</span></span>`;
}
/* plan E12: Ctrl/Cmd-click picks files without opening them, Shift-click a range: several outputs to
 * compare (explorer › Compare Convergence); a plain click opens one, as before */
state.selected = new Set();
let lastPicked = null;
$("#file-list").addEventListener("click", (e) => {
  const b = e.target.closest(".file");
  if (!b) return;
  const p = b.dataset.path;
  if (e.ctrlKey || e.metaKey) {
    if (state.selected.has(p)) state.selected.delete(p); else state.selected.add(p);
    lastPicked = p;
    renderFiles();
    return;
  }
  if (e.shiftKey && lastPicked) {
    const rows = [...document.querySelectorAll("#file-list .file")].map((x) => x.dataset.path);
    const [i, j] = [rows.indexOf(lastPicked), rows.indexOf(p)].sort((x, y) => x - y);
    if (i >= 0 && j >= 0) rows.slice(i, j + 1).forEach((x) => state.selected.add(x));
    renderFiles();
    return;
  }
  if (state.selected.size) { state.selected.clear(); }
  lastPicked = p;
  document.dispatchEvent(new CustomEvent("mace-open", { detail: p }));
});
let filterTimer = 0;
$("#file-filter").addEventListener("input", (e) => {
  clearTimeout(filterTimer);
  filterTimer = setTimeout(() => { state.text = e.target.value; renderFiles(); }, 60);
});

async function loadFiles(given) {
  const data = given || await api("files");
  state.files = data.files;
  document.dispatchEvent(new CustomEvent("mace-files"));
  $("#side-foot").textContent = `${data.files.length} files` + (data.truncated ? " (list truncated)" : "");
  renderFiles();
  fillSelectors();
  if (!state.detailsAsked) {
    // formula and atom count for the explorer, once, when the page is idle (plan T11);
    // a reload of the same folder reuses them (the request would cost the reload's budget)
    state.detailsAsked = true;
    const sig = detailsSig(data.files), key = "mace.gui.details";
    let kept = null;
    try { kept = JSON.parse(sessionStorage.getItem(key) || "null"); } catch (e) { kept = null; }
    if (kept && kept.sig === sig) { state.details = kept.details || {}; renderFiles(); }
    else {
      const ask = () => api("files?details=1").then((r) => {
        state.details = r.details || {};
        try { sessionStorage.setItem(key, JSON.stringify({ sig, details: state.details })); } catch (e) { /* full or private */ }
        renderFiles();
      }).catch(() => {});
      (window.requestIdleCallback || ((f) => setTimeout(f, 300)))(ask, { timeout: 3000 });
    }
  }
}

function fileByPath(p) { return state.files.find((f) => f.path === p); }
/** What the content says an output or log is (plan E1): {role, calc, dim} once its details are in. */
function roleInfo(f) {
  const d = f && state.details && state.details[f.path];
  return d && d.refined ? d : null;
}
/** The kind a filter sees: a job log, a properties output or plain text by content, else the name's kind. */
function filterKind(f) {
  const r = roleInfo(f);
  if (r && r.role === "joblog") return "joblog";
  if (r && r.role === "properties_out") return "props";
  if (r && r.role === "crystal_out" && f.kind === "log") return "out";
  return f.kind;
}
/** A job log (by content, or a .e/.err by name): never opened as a CRYSTAL output. */
function isJobLog(f) {
  if (typeof f === "string") f = fileByPath(f);
  const r = roleInfo(f);
  return !!f && (r ? r.role === "joblog" : false);
}
/** The explorer chip: the calculation type an output's own lines give, "log" for a job log, else the kind. */
function chipOf(f) {
  const r = roleInfo(f);
  if (r && r.role === "joblog") return { cls: "joblog", label: "log", tip: "Job log (no CRYSTAL banner): opens as text" };
  if (r && (r.role === "crystal_out" || r.role === "properties_out") && r.calc) {
    const cls = r.calc === "FREQ" ? "freq" : r.role === "properties_out" ? "props" : "out";
    return { cls, label: KIND_LABEL[r.calc] || r.calc,
      tip: `${r.role === "properties_out" ? "Properties output" : "CRYSTAL output"} · ${r.calc} (from the output's own lines)` };
  }
  const k = f.kind === "out" && isFreq(f.path) ? "freq" : f.kind;
  return { cls: k, label: KIND_LABEL[k] || k, tip: "" };
}
function chipHtml(f) {
  const c = chipOf(f);
  return `<span class="kb ${esc(c.cls)}"${c.tip ? ` data-tip="${esc(c.tip)}"` : ""}>${esc(c.label)}</span>`;
}
/** A file under a kind chip: its kind, or (Properties) an output whose index has normal modes. */
function matches(f, kinds) {
  const k = filterKind(f);
  return kinds.includes(k) || (k === "out" && kinds.includes("freq") && isFreq(f.path));
}
/** The role of an output or log: from its details, else asked (GET /api/filekind), cached in the details. */
async function roleOf(path) {
  const f = fileByPath(path);
  const r = roleInfo(f);
  if (r) return r;
  const got = await api("filekind?" + q({ path }));
  state.details = state.details || {};
  state.details[path] = { ...(state.details[path] || {}), ...got };
  return got;
}
/** A CRYSTAL output with normal modes (its index, read with the explorer's details; never the listing). */
function isFreq(p) { const d = state.details && state.details[p]; return !!(d && d.kind === "freq"); }
/** The structure files' paths, sizes and modification times, hashed: their details still hold while it is the same. */
function detailsSig(files) {
  let h = 0;
  for (const f of files) {
    if (!["cif", "d12", "out", "log"].includes(f.kind)) continue;
    const t = `${f.path}:${f.size}:${f.mtime_ns || f.mtime}|`;
    for (let i = 0; i < t.length; i++) h = (h * 31 + t.charCodeAt(i)) | 0;
  }
  return `${files.length}:${h >>> 0}`;
}
/** "MgO · 8 atoms · Fm-3m" for a structure file whose details are in. */
function detailText(f) {
  const d = state.details && state.details[f.path];
  if (!d || d.error || !d.formula) return "";
  return `${d.formula} · ${d.n} atom${d.n === 1 ? "" : "s"}${d.sg ? ` · ${d.sg}` : ""}`;
}

/* Load a file into its view. Returns the document kind the shell shows it
 * as: structure, trajectory, plot, db, setup or text. `data` is what an
 * earlier open returned (the shell keeps it), so switching back to a
 * document tab costs no request. */
async function openFile(path, cached) {
  const f = fileByPath(path);
  if (!f) throw new Error(`${path} is not in the served folder`);
  state.active = path;
  renderFiles();
  if (["cif", "d12", "xyz", "poscar", "mol", "pdb", "gen", "fort34"].includes(f.kind)) {
    // the shell fetches it packed (base64 arrays, spec §8.4.5) once the decoder is here
    const s = cached || await (window.MACE.fetchStructure ? window.MACE.fetchStructure(path) : api("structure?" + q({ path })));
    await loadStructure(path, s); showTab("structure");
    return { kind: "structure", data: s };
  }
  if (f.kind === "out" || f.kind === "log") {
    // the summary (energies, labels, first and last frames); other steps load on demand (plan B6);
    // geom/core.js in parallel, for the cell's standard view
    const [t] = await Promise.all([cached || api("out/summary?" + q({ path })).then(expandSummary), coreReady()]);
    return { kind: await loadOut(path, t), data: t };
  }
  // plot documents draw in js/docs/plot.js (the shell opens them there)
  if (["band", "dos", "ir", "raman", "image", "html", "transport"].includes(f.kind)) { showTab("results"); return { kind: "plot" }; }
  if (f.kind === "db") { showTab("db"); await window.MACE.openDb(path); return { kind: "db" }; }
  throw new Error(`${f.name}: no view for this kind of file`);
}

/* geom/core.js (the shell's own module) for the views app.js fits: loaded with the first output */
function coreReady() {
  if (window.MACE.standardView) return Promise.resolve();
  return import(new URL(assetUrl("js/geom/core.js"), document.baseURI).href)
    .then((c) => { window.MACE.standardView = c.standardView; window.MACE.pivotOf = c.pivotOf; }, () => { /* the old tilt stays */ });
}
/* 3Dmol (about 0.5 MB) loads after first paint, or at once when a document
 * needs it: nothing 3D waits on it before the shell is up. */
let threeLoading = null;
function load3Dmol() {
  if (window.$3Dmol && state.gl) return Promise.resolve();
  if (!threeLoading) threeLoading = Promise.all([window.$3Dmol ? null : new Promise((resolve, reject) => {
    const el = document.createElement("script");
    el.src = assetUrl("vendor/3dmol/3Dmol-min.js");
    el.onload = () => resolve();
    el.onerror = () => reject(new Error("3Dmol.js did not load: the 3D views are unavailable"));
    document.head.appendChild(el);
  }),
  // what every viewer gets on top of 3Dmol (clipping planes and depth cueing): before the first viewer
  import(new URL(assetUrl("js/viewport/clip.js"), document.baseURI).href),
  import(new URL(assetUrl("js/viewport/rendering.js"), document.baseURI).href),
  // the rotation center every fitted view starts from (geom/core.js pivotOf)
  import(new URL(assetUrl("js/geom/core.js"), document.baseURI).href),
  // Settings › Viewport › Radii (B7): the radius tables 3Dmol draws the spheres from
  import(new URL(assetUrl("js/viewport/radii.js"), document.baseURI).href)]).then(([, clip, look, core, radii]) => {
    state.gl = { clip, look };
    state.radii = radii;
    applyRadii();
    window.MACE.pivotOf = core.pivotOf;
    // a viewer made in between (3Dmol here, the module not yet): it gets them now
    Object.values(state.viewers).forEach((v) => v && glExtras(v));
  },
    (e) => { threeLoading = null; throw e; });
  return threeLoading;
}
/* static/v-<hash>/... : cached for good, renamed when a file changes */
function assetUrl(name) {
  const v = document.querySelector('script[src*="/app.js"]').getAttribute("src").split("/")[1];
  return `static/${v}/${name}`;
}

/* ---------------------------------------------------------------- 3D */
/* On top of 3Dmol, for every viewer: the slab follows what is drawn at every
 * paint, and the depth cueing is ours (viewport/clip.js). */
function glExtras(v) {
  if (!state.gl || v.maceClip) return;
  state.gl.look.installShaderPatch(v);
  state.gl.clip.installClip(v, { depthCue: () => num("depthCue", 0.15), beforePaint: () => lookOf(v) });
}
/* The Rendering values of the drawn document (Settings › Viewport by default), before every paint. */
function lookOf(v) {
  // an outline from the lighting or from the style (Grayscale print draws one under any lighting)
  const outline = setting("outline", false) === true || setting("styleOutline", false) === true;
  state.gl.look.apply(v, { lightStrength: num("lightStrength", 1), lightDirection: setting("lightDirection", "front"),
    ambient: num("ambient", 0), shininess: num("shininess", 0), outline, outlineWidth: num("outlineWidth", 0.1),
    // no color chosen: one that stands out on the drawn background (a style read only when an outline is drawn)
    outlineColor: outline ? setting("outlineColor", "")
      || state.gl.look.autoOutline(setting("background", "theme"), setting("backgroundColor", ""), cssVar("--text")) : "",
    ambientOcclusion: num("ambientOcclusion", 0), lod: v.maceLod || 0 });
  spinLegend(v);
  // a document with its own background: switched to on the next frame (never inside this paint)
  const bg = `${setting("background", "theme")}|${setting("backgroundColor", "")}`;
  if (v.maceBg === undefined) v.maceBg = bg;
  if (v.maceBg !== bg) { v.maceBg = bg; requestAnimationFrame(() => v.setBackgroundColor(viewerBg())); }
}
/* One WebGL probe at the first viewer (polish V3): a browser that cannot draw
 * 3D here (WebGL off, a remote desktop without it) gets an explanation in each
 * viewport instead of a blank canvas, and viewer() returns null; everything
 * else (plots, text, the database, Setup) keeps working. */
let webglProbe = null;
function webgl() {
  if (webglProbe) return webglProbe;
  let ok = false, version = 0, renderer = "";
  try {
    const c = document.createElement("canvas");
    let g = c.getContext("webgl2");
    version = g ? 2 : 0;
    if (!g) { g = c.getContext("webgl") || c.getContext("experimental-webgl"); version = g ? 1 : 0; }
    if (g) {
      ok = true;
      const e = g.getExtension("WEBGL_debug_renderer_info");
      renderer = String(e ? g.getParameter(e.UNMASKED_RENDERER_WEBGL) : g.getParameter(g.RENDERER));
      const lose = g.getExtension("WEBGL_lose_context");
      if (lose) lose.loseContext();
    }
  } catch (e) { ok = false; }
  webglProbe = { ok, version, renderer };
  window.MACE.webgl = webglProbe;
  return webglProbe;
}
function no3d(el) {
  if (!el || el.querySelector(".no3d")) return;
  const diag = window.MACE.hasCommand && window.MACE.hasCommand("help.copyDiagnostics")
    ? ' <button type="button" class="tb-btn-text sm no3d-diag">Copy Diagnostics</button>' : "";
  el.insertAdjacentHTML("beforeend", `<div class="no3d" role="note"><h3>3D view unavailable</h3>` +
    `<p><strong>3D view needs WebGL.</strong> Turn on hardware acceleration in your browser settings, then reload.</p>` +
    `<p>This browser cannot draw 3D here (WebGL is off or not supported, which is common in remote desktops). ` +
    `Plots, text, the database and Setup still work. If that does not help, try another browser, or open ` +
    `MACE GUI from your own computer with mace gui connect.${diag}</p></div>`);
  const b = el.querySelector(".no3d-diag");
  if (b) b.addEventListener("click", () => window.MACE.runCommand && window.MACE.runCommand("help.copyDiagnostics"));
}
/* 3Dmol's own mouse handling (the run and comparison viewers) measures the
 * pointer from the page (pageX minus the canvas's screen corner) against the
 * canvas's own size (WIDTH, offsetWidth). The UI scale zooms the shell, and
 * in Chromium 128+ and Firefox 126+ those two differ by the zoom: clicks,
 * hovers and right-clicks landed off the pointer, further right and down the
 * larger the scale. Its pointer readers report the point in the canvas's own
 * px from its corner instead (the corner stays where 3Dmol adds it); the
 * vendored 3Dmol itself stays untouched. */
function unzoom3Dmol() {
  const P = window.$3Dmol && $3Dmol.GLViewer && $3Dmol.GLViewer.prototype;
  if (!P || !P.getX || P.getX.maceUnzoomed) return;
  const wrap = (read, side) => {
    const f = function (e) {
      const at = read.call(this, e);
      const z = at == null || !this.glDOM ? 1 : window.maceZoom.of(this.glDOM);
      if (z === 1) return at;
      const corner = this.canvasOffset()[side];
      return corner + (at - corner) / z;
    };
    f.maceUnzoomed = true;
    return f;
  };
  P.getX = wrap(P.getX, "left");
  P.getY = wrap(P.getY, "top");
  /* Settings › Viewport › Rotation and Angle snap in the 3Dmol views: a
   * left-drag in Turntable mode or with an angle snap is turned by
   * the structure view's own rule (viewport/camera.js dragOrientation, from
   * the orientation at the button press, the turntable's up kept on the
   * viewer); a free drag, Shift (zoom), Ctrl (pan) and touch stay 3Dmol's. */
  if (P._handleMouseMove && !P._handleMouseMove.maceRotation) {
    const move = P._handleMouseMove;
    P._handleMouseMove = function (e) {
      const M = window.MACE, rp = M && M.rotationPrefs ? M.rotationPrefs() : null;
      if (!rp || (rp.mode === "free" && !rp.step) || !this.isDragging || !this.scene || this.mouseButton !== 1
        || e.shiftKey || e.ctrlKey || e.targetTouches || !this.cq) return move.call(this, e);
      e.preventDefault();
      const x = this.getX(e), y = this.getY(e);
      if (x === undefined) return;
      const q0 = [this.cq.x, this.cq.y, this.cq.z, this.cq.w];
      if (this.maceDragFrom !== this.cq) { this.maceDragFrom = this.cq; if (rp.mode === "axes") this.maceUp = M.turntableUp(q0, this.maceUp); }
      const q = M.dragOrientation(q0, x - this.mouseStartX, y - this.mouseStartY, { ...rp, up: this.maceUp });
      const view = this.getView();
      this.setView([view[0], view[1], view[2], view[3], ...q]);
    };
    P._handleMouseMove.maceRotation = true;
  }
  /* 3Dmol's own wheel zoomed out on scrolling up, the opposite of the
   * structure viewport: the run and comparison views now scroll the same way
   * (up zooms in), and Settings › Viewport › Invert scroll zoom flips both.
   * Same step as 3Dmol's (85 % of the distance, Ctrl reverses), the settings
   * read per event so a change applies live. Settings › Viewport › Zoom
   * toward: the view center stays put (3Dmol's own), or the model point under
   * the pointer does: the model moves by the change in what the pointer's
   * offset from the center spans in the model (3Dmol's screenOffsetToModel,
   * the same measure its own pan uses), before and after. */
  if (P._handleMouseScroll) P._handleMouseScroll = function (e) {
    e.preventDefault();
    if (!this.scene) return;
    const x = this.getX(e), y = this.getY(e);
    if (x === undefined || !(this.control_all || this.isInViewer(x, y))) return;
    let toPointer = null;
    if (setting("viewport.zoomToward", "center") === "pointer") {
      const o = this.canvasOffset(), dx = x - o.left - this.WIDTH / 2, dy = y - o.top - this.HEIGHT / 2;
      this.setSlabAndFog();
      toPointer = { dx, dy, before: this.screenOffsetToModel(dx, dy) };
    }
    const r = 0.85 * (this.CAMERA_Z - this.rotationGroup.position.z);
    let n = setting("invertWheelZoom", false) === true ? 1 : -1;
    if (e.ctrlKey) n = -n;
    // the standard deltaY first (in px: lines are 16 px, pages 400): Firefox also sets the old wheelDelta,
    // whose formula divides by zero at -600 (one scroll down there) and left the view NaN
    const px = e.deltaY ? e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1) : 0;
    let step;
    if (px) step = n * r * Math.max(-1, Math.min(1, px / 400));
    else if (e.wheelDelta) step = -n * r * Math.max(-1, Math.min(1, e.wheelDelta / 400));
    else if (e.detail) step = n * r * e.detail / 10;
    else return;
    if (!Number.isFinite(step)) return;
    this.rotationGroup.position.z += step;
    this.rotationGroup.position.z = this.adjustZoomToLimits(this.rotationGroup.position.z);
    if (toPointer) {
      this.setSlabAndFog();
      const after = this.screenOffsetToModel(toPointer.dx, toPointer.dy), p = this.modelGroup.position;
      p.x += after.x - toPointer.before.x; p.y += after.y - toPointer.before.y; p.z += after.z - toPointer.before.z;
    }
    this.show();
  };
}
/* The UI scale zooms the shell, and a canvas inside it is drawn at its own px
 * and then stretched by the zoom: 3Dmol sizes its drawing to the viewer's own
 * px times devicePixelRatio, so above 100 % the 3D view was soft. While its
 * renderer sizes the canvas, devicePixelRatio counts the zoom too, so the
 * canvas holds one pixel per screen pixel. Only the drawing grows: the
 * canvas's own size, 3Dmol's WIDTH and HEIGHT, and so every pointer reader and
 * the picking, stay in the viewer's own px. The vendored 3Dmol stays untouched. */
const backing = (el) => (window.devicePixelRatio || 1) * window.maceZoom.of(el);
function sharpUnderZoom(v, el) {
  const r = v.renderer;
  if (!r || typeof r.setSize !== "function" || r.setSize.maceSharp) return;
  const setSize = r.setSize;
  r.setSize = function (...args) {
    const z = window.maceZoom.of(el);
    if (Math.abs(z - 1) < 1e-3) return setSize.apply(this, args);
    let own;
    try {
      own = Object.getOwnPropertyDescriptor(window, "devicePixelRatio");
      Object.defineProperty(window, "devicePixelRatio", { value: (window.devicePixelRatio || 1) * z, configurable: true });
    } catch (e) { return setSize.apply(this, args); }   // a browser that keeps it fixed: drawn as before
    try { return setSize.apply(this, args); }
    finally { if (own) Object.defineProperty(window, "devicePixelRatio", own); else delete window.devicePixelRatio; }
  };
  r.setSize.maceSharp = true;
}
/* The rotation center, shown while a drag turns the view (a small crosshair,
 * gone on release): 3Dmol turns about the model point at the view's center,
 * minus its translation (view[0..2]), so panning carries the center along.
 * The structure viewer's pointer (viewport/pointer.js) shows it during its
 * orbit drags (right-drag, Alt+left-drag, the Orbit tool); the viewers that
 * keep 3Dmol's own mouse (comparison, modes, volumes, the slab preview) turn
 * on a plain left-drag (right-drag zooms there, Ctrl or middle-drag pans). */
function pivotMark(v, el, ownPointer) {
  const mark = document.createElement("div");
  mark.className = "pivot-mark";
  mark.setAttribute("aria-hidden", "true");
  mark.hidden = true;
  el.appendChild(mark);
  let flash = 0;
  // where the center is drawn, in the viewer's own px from its top left
  v.pivotPx = () => {
    const cv = v.getCanvas ? v.getCanvas() : null;
    if (!cv || !v.visible()) return null;
    try { if (v.setSlabAndFog) v.setSlabAndFog(); if (v.scene && v.scene.updateMatrixWorld) v.scene.updateMatrixWorld(true); } catch (e) { return null; }
    const w = v.getView(), p = v.modelToScreen({ x: -w[0], y: -w[1], z: -w[2] }), r = cv.getBoundingClientRect();
    return p && Number.isFinite(p.x) && Number.isFinite(p.y) ? [p.x - r.left + cv.offsetLeft, p.y - r.top + cv.offsetTop] : null;
  };
  v.showPivot = (on) => {
    clearTimeout(flash);
    const at = on ? v.pivotPx() : null;
    if (at) { mark.style.left = at[0] + "px"; mark.style.top = at[1] + "px"; }
    mark.hidden = !at;
  };
  v.flashPivot = (ms = 700) => { v.showPivot(true); flash = setTimeout(() => { mark.hidden = true; }, ms); };
  if (ownPointer) return;
  // 3Dmol's own mouse: a left-drag without Shift or Ctrl turns the view
  let down = null;
  el.addEventListener("pointerdown", (e) => {
    down = e.button === 0 && !e.shiftKey && !e.ctrlKey && !e.metaKey ? { x: e.clientX, y: e.clientY, id: e.pointerId } : null;
  }, true);
  el.addEventListener("pointermove", (e) => {
    if (down && e.pointerId === down.id && mark.hidden && Math.hypot(e.clientX - down.x, e.clientY - down.y) >= 3) v.showPivot(true);
  }, true);
  const up = (e) => { if (down && e.pointerId === down.id) { down = null; v.showPivot(false); } };
  addEventListener("pointerup", up, true);
  addEventListener("pointercancel", up, true);
}
function viewer(id) {
  if (!state.viewers[id] && !webgl().ok) { no3d(document.getElementById(id)); return null; }
  unzoom3Dmol();
  if (!state.viewers[id]) {
    const el = document.getElementById(id);
    // antialias alone makes 3Dmol also draw at 2x the pixels (its "upscale"),
    // 4x the fill work at devicePixelRatio 1; the context's own multisampling
    // smooths the edges without it
    // antialias needs a new WebGL context: it applies to viewers created after it changes
    // the structure viewer's pointer belongs to the shell (viewport/pointer.js):
    // 3Dmol's own mouse handling stays on for the run and comparison viewers
    let v = null;
    try {
      v = $3Dmol.createViewer(el, { backgroundColor: viewerBg(),
        antialias: setting("antialias", true) !== false, upscale: false, id: id + "-gl", minimumZoomToDistance: 1,
        nomouse: id === "viewer-main" });
    } catch (e) { v = null; }
    // the probe found WebGL, yet 3Dmol got no context (a driver that refuses its
    // attributes, a context limit): the same explanation, and no more viewers
    if (!v || (v.renderer && "_gl" in v.renderer && !v.renderer._gl)) {
      webglProbe = { ...webglProbe, ok: false };
      window.MACE.webgl = webglProbe;
      no3d(el);
      return null;
    }
    sharpUnderZoom(v, el);
    pivotMark(v, el, id === "viewer-main");
    // every 3D view follows Settings › Viewport › Projection from the start (the mode and volume views drew in perspective until it changed)
    if (v.setProjection) v.setProjection(setting("projection", "orthographic") === "perspective" ? "perspective" : "orthographic");
    // A viewer in a hidden tab has a 0x0 canvas, and 3Dmol's shared
    // OffscreenCanvas throws when it copies a 0x0 frame; showTab() renders
    // it once it is visible. 3Dmol also paints from inside its own calls
    // (removeAllLabels, setView, ...) through show(), so that is guarded too,
    // and a viewer whose element changed size is resized before it paints.
    v.visible = () => el.offsetWidth > 0 && el.offsetHeight > 0;
    v.viewerId = id;
    glExtras(v);
    // 3Dmol's own ResizeObserver sizes the viewer to 0x0 (v.WIDTH = 0) when
    // its tab hides, and a requestAnimationFrame paint can run before that
    // observer fires again on the way back: so a 0x0 viewer is resized too,
    // even when its element has the size it had before.
    const render = v.render.bind(v), show = v.show.bind(v);
    let size = "";
    v.render = (...args) => (v.visible() ? render(...args) : v);
    // Render on demand, at most once per animation frame (spec §8.2): 3Dmol
    // paints from inside many of its calls, and a document switch made
    // several full paints in a row.
    let frame = 0;
    v.show = (...args) => {
      if (!v.visible() || frame) return;
      frame = requestAnimationFrame(() => {
        if (!v.visible()) { frame = 0; return; }
        const now = el.offsetWidth + "x" + el.offsetHeight + "@" + backing(el);
        // resize() paints through show() itself: still "frame", so that is a no-op
        if (now !== size || !v.WIDTH || !v.HEIGHT) { size = now; v.resize(); }
        frame = 0;
        show(...args);
        document.dispatchEvent(new CustomEvent("mace-painted", { detail: id }));
      });
    };
    // An edit paints in the same task, not on the next frame: the change is
    // on screen one frame sooner (spec §8.1, edit to render <= 50 ms). A frame
    // already queued is dropped; later changes merge into frames as before.
    v.paintNow = () => {
      if (!v.visible()) return false;
      if (frame) cancelAnimationFrame(frame);
      frame = -1;                    // resize() paints through show(): a no-op meanwhile
      const now = el.offsetWidth + "x" + el.offsetHeight + "@" + backing(el);
      if (now !== size || !v.WIDTH || !v.HEIGHT) { size = now; v.resize(); }
      frame = 0;
      show();
      document.dispatchEvent(new CustomEvent("mace-painted", { detail: id }));
      return true;
    };
    // 3Dmol reports a right-click (not a right-drag) with the atom under it, if any
    v.userContextMenuHandler = (atom, x, y, hits, ev) => {
      if (window.MACE.onViewerContext) window.MACE.onViewerContext(atom, x, y, ev, v);
    };
    state.viewers[id] = v;
  }
  return state.viewers[id];
}
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => Math.sqrt(dot(a, a));
const xyz = (p) => ({ x: p[0], y: p[1], z: p[2] });
const fracToCart = (f, cell) => add(add(mul(cell[0], f[0]), mul(cell[1], f[1])), mul(cell[2], f[2]));

/* Atoms to draw: wrapped into the cell along periodic axes, plus the
 * images on the cell faces (VESTA's boundary atoms) and an optional supercell. */
function displayAtoms(s, o) {
  const tol = 0.02, N = o.supercell ? 2 : 1;
  const span = (lo, hi) => Array.from({ length: hi - lo + 1 }, (_, k) => lo + k);
  const ranges = s.pbc.map((p) => !p ? [0] : (o.boundary ? span(-1, N) : span(0, N - 1)));
  const out = [];
  s.frac.forEach((f0, i) => {
    const f = f0.map((x, k) => s.pbc[k] ? x - Math.floor(x + 1e-9) : x);
    for (const a of ranges[0]) for (const b of ranges[1]) for (const c of ranges[2]) {
      const sh = [a, b, c];
      const g = f.map((x, k) => x + sh[k]);
      let inside = true;
      for (let k = 0; k < 3; k++) {
        if (s.pbc[k] && (g[k] < -tol || g[k] > N + tol)) inside = false;
      }
      if (!inside) continue;
      out.push({ i, cart: fracToCart(g, s.cell), image: !(a === 0 && b === 0 && c === 0), shift: sh });
    }
  });
  return out;
}

/* Covalent-radius bonds, but none between two metals (as VESTA draws ionic
 * solids: otherwise Mg-Mg in MgO, Na-Na in NaCl are "bonded"). */
function bondsFor(atoms, s, scale) {
  // drawn bonds use Settings' tolerance; the server's bond table keeps its own 1.2
  scale = scale || num("bondTolerance", 0) || s.bond_scale || 1.2;
  const radii = s.radii, metal = s.metal || [], metalMetal = setting("bondMetalMetal", false) === true;
  const n = atoms.length, bonds = atoms.map(() => []);
  if (n > 4000) return bonds;
  for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) {
    if (!metalMetal && metal[atoms[a].i] && metal[atoms[b].i]) continue;
    const d = norm(sub(atoms[a].cart, atoms[b].cart));
    if (d > 0.1 && d < scale * (radii[atoms[a].i] + radii[atoms[b].i])) {
      bonds[a].push(b); bonds[b].push(a);
    }
  }
  return bonds;
}

/* The element colors: Settings' colorScheme is the one source (the old
 * #opt-colors select only mirrors it). It names a palette (1.1.6): a preset
 * of element_colors.json (vesta, jmol, cvd) or "user:<name>", one of the
 * user's palettes in ~/.config/mace/element_colors.json (state.palettes,
 * from /api/palettes; mace plotting reads the same file). A user palette is
 * its base preset plus the elements it changes. While Element Colors is
 * open, its draft (state.paletteDraft) is what every view draws. */
const paletteCache = new Map();
function paletteMap(id) {
  const c = state.colors;
  if (!c || !id) return null;
  if (c.schemes[id]) return c.schemes[id];
  const p = id.startsWith("user:") && state.palettes && state.palettes.palettes ? state.palettes.palettes[id.slice(5)] : null;
  if (!p) return null;
  if (!paletteCache.has(id)) paletteCache.set(id, { ...(c.schemes[p.base] || c.schemes[c.default]), ...(p.colors || {}) });
  return paletteCache.get(id);
}
function paletteIds() {
  const c = state.colors;
  if (!c) return [];
  const users = Object.keys((state.palettes && state.palettes.palettes) || {}).sort((a, b) => a.localeCompare(b));
  return [c.default, ...Object.keys(c.schemes).filter((k) => k !== c.default).sort(), ...users.map((n) => `user:${n}`)];
}
function paletteLabel(id) {
  const c = state.colors;
  if (id.startsWith("user:")) return id.slice(5);
  return (c && c.labels && c.labels[id]) || id;
}
/* Every view draws the colors again (the classic views, the editor, FREQ, volumes). */
function colorsChanged() {
  document.dispatchEvent(new CustomEvent("mace-colors"));
  fillColorOptions();
  redrawViews();
}
function setPalettes(v) {
  state.palettes = v || { palettes: {} };
  paletteCache.clear();
  colorsChanged();
}
function setPaletteDraft(map) {
  state.paletteDraft = map || null;
  colorsChanged();
}
function schemeName() {
  const c = state.colors, name = setting("colorScheme", c ? c.default : "vesta");
  return paletteMap(name) ? name : (c ? c.default : name);
}
function currentPalette() { return state.paletteDraft || paletteMap(schemeName()) || {}; }
/* What the views draw (1.1.7, Inspector › Rendering › Style, the drawn document's atomColor): the
 * color scheme, VESTA's own colors whatever the scheme, or either as grays of the same lightness. */
const grayCache = new WeakMap();
function grayOf(map) {
  if (!grayCache.has(map)) {
    const lin = (x) => { x /= 255; return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4; };
    const enc = (y) => Math.round(255 * (y <= 0.0031308 ? 12.92 * y : 1.055 * y ** (1 / 2.4) - 0.055));
    const out = {};
    for (const [k, v] of Object.entries(map)) {
      if (!/^#[0-9a-f]{6}$/i.test(v)) { out[k] = v; continue; }
      const n = parseInt(v.slice(1), 16);
      const g = enc(0.2126 * lin(n >> 16) + 0.7152 * lin((n >> 8) & 255) + 0.0722 * lin(n & 255)).toString(16).padStart(2, "0");
      out[k] = "#" + g + g + g;
    }
    grayCache.set(map, out);
  }
  return grayCache.get(map);
}
function drawnPalette() {
  const how = setting("atomColor", "scheme");
  const base = how !== "vesta" || state.paletteDraft ? currentPalette() : paletteMap("vesta") || currentPalette();
  return how === "gray" ? grayOf(base) : base;
}
function colorScheme() {
  const c = state.colors;
  if (!c) return "Jmol";
  return { prop: "elem", map: drawnPalette() };
}
function elementColor(sym) {
  const c = state.colors;
  if (!c) return cssVar("--fallback-atom");
  return drawnPalette()[sym] || c.fallback;
}
/* Magnetic style: each atom by its Mulliken spin moment (the shell's docSpins: {values, max} of the
 * drawn document's run, or null), on a diverging scale centered at 0 from the --spin-atom-* tokens.
 * An atom's index in the structure: properties.base where a view sets it (the volume and mode views
 * draw their own lists), else its serial − 1. */
const spinCache = { key: "", hex: null };
function mixRgb(a, b, t) {
  const p = parseInt(a.slice(1), 16), q = parseInt(b.slice(1), 16);
  const ch = (sh) => Math.round(((p >> sh) & 255) * (1 - t) + ((q >> sh) & 255) * t);
  return "#" + [16, 8, 0].map((sh) => ch(sh).toString(16).padStart(2, "0")).join("");
}
function spinData() {
  if (setting("atomColor", "scheme") !== "spin") return null;
  const sp = window.MACE.docSpins ? window.MACE.docSpins() : null;
  return sp && sp.values && sp.values.length ? sp : null;
}
function spinHexes(sp) {
  const up = cssVar("--spin-atom-up"), zero = cssVar("--spin-atom-zero"), down = cssVar("--spin-atom-down");
  const key = [up, zero, down, sp.max, sp.values.length, sp.source].join("|");
  if (spinCache.key !== key || spinCache.sp !== sp) {
    spinCache.key = key; spinCache.sp = sp;
    spinCache.hex = sp.values.map((m) => { const t = Math.max(-1, Math.min(1, m / sp.max)); return t >= 0 ? mixRgb(zero, up, t) : mixRgb(zero, down, -t); });
  }
  return spinCache.hex;
}
function colorSpec() {
  const sp = spinData();
  if (!sp) return { colorscheme: colorScheme() };
  const hex = spinHexes(sp), fb = cssVar("--fallback-atom");
  return { colorfunc: (a) => hex[a.properties && Number.isInteger(a.properties.base) ? a.properties.base : a.serial - 1] || fb };
}
/* The atom drawing of the drawn document's style (atomDraw): auto keeps the Display section's
 * style and Bonds; space-filling never draws bonds; thin sticks are the stick style. */
function effDraw(style, bonds) {
  const d = setting("atomDraw", "auto");
  if (d === "spacefill") return { style: "spacefill", bonds: false };
  if (d === "sticks") return { style: "stick", bonds };
  if (d === "ballstick") return { style: "ballstick", bonds };
  return { style, bonds };
}
/* The color scale in a corner of a 3D view while Magnetic colors its atoms (checked at every paint). */
function spinLegend(v) {
  // over the viewer itself (not the controls a run view keeps under it in the same frame)
  const cv = v.getCanvas && v.getCanvas(), wrap = cv && (cv.closest(".viewer") || cv.parentElement);
  if (!wrap) return;
  const sp = spinData(), want = sp && v.getModel && v.getModel() ? `${sp.max}|${sp.source}` : "";
  let el = wrap.querySelector(":scope > .vp-spin-legend");
  if ((el ? el.dataset.key : "") === want) return;
  if (!want) { if (el) el.remove(); return; }
  if (!el) { el = document.createElement("div"); el.className = "vp-spin-legend"; el.setAttribute("role", "img"); wrap.appendChild(el); }
  const m = fmt(sp.max, sp.max >= 1 ? 1 : 2);
  el.dataset.key = want;
  el.setAttribute("aria-label", `Atoms colored by Mulliken spin moment, from −${m} (blue, spin down) through 0 (white) to +${m} μB (red, spin up)`);
  el.title = `Mulliken spin moments from ${sp.source}`;
  el.innerHTML = `<span class="vsl-title">Spin (μB)</span><span class="vsl-bar"></span><span class="vsl-ticks"><span>−${m}</span><span>0</span><span>+${m}</span></span>`;
}
/* The classic panel's color select: every palette, the chosen one selected. */
function fillColorOptions() {
  const sel = document.getElementById("opt-colors");
  if (!sel || !state.colors) return;
  sel.innerHTML = paletteIds().map((k) => `<option value="${esc(k)}">${esc(paletteLabel(k))}</option>`).join("");
  sel.value = schemeName();
}
/* Settings › Viewport › Radii (B7, viewport/radii.js): 3Dmol's radius table (GLModel.vdwRadii, which
 * every view, the clipping and the picking read) holds the chosen set; its own van der Waals table is
 * kept for the "vdw" set. Ball-and-stick spheres are atomScale × the radius with bonds (VESTA's rule: the
 * default 0.40 draws 40 % of the atomic radius, T9), × 1.357 without; with van der Waals radii × 0.7
 * and × 0.95, the 0.28 and 0.38 the classic views drew before 1.1.6. */
const round6 = (x) => Math.round(x * 1e6) / 1e6;
// the drawn document's style can choose its own (atomRadii: VESTA-like atomic, Space-filling van der Waals)
const radiusSet = () => { const o = setting("atomRadii", "auto"); const r = o === "atomic" || o === "vdw" ? o : setting("radiusSet", "atomic"); return ["atomic", "covalent", "ionic", "vdw"].includes(r) ? r : "atomic"; };
function applyRadii() {
  const G = window.$3Dmol && window.$3Dmol.GLModel;
  if (!G || !state.radii) return;
  const set = radiusSet();
  if (state.radiiSet === set) return;
  if (!state.vdw0) state.vdw0 = { ...G.vdwRadii };
  const t = state.radii.table(set, state.vdw0);
  for (const k of Object.keys(G.vdwRadii)) if (!(k in t)) delete G.vdwRadii[k];
  Object.assign(G.vdwRadii, t);
  state.radiiSet = set;
}
function sphereScale(bonds) {
  applyRadii();
  const f = state.radii ? state.radii.factor(radiusSet(), bonds) : (bonds ? 0.7 : 0.95);
  return round6(num("atomScale", 0.4) * f);
}
function styleFor(kind, bonds, opacity = 1) {
  applyRadii();                         // the drawn document's radii (its style can choose them)
  const cs = colorSpec();
  if (kind === "spacefill") return { sphere: { scale: 0.95, ...cs, opacity } };
  // Wireframe / sticks: the style's thin bond radius; the Display section's Stick style its own 0.16 Å
  if (kind === "stick") return { stick: { radius: setting("atomDraw", "auto") === "sticks" ? num("bondRadius", 0.06) : 0.16, ...cs, opacity } };
  return bonds ? { sphere: { scale: sphereScale(true), ...cs, opacity }, stick: { radius: num("bondRadius", 0.12), ...cs, opacity } }
               : { sphere: { scale: sphereScale(false), ...cs, opacity } };
}

function addModel(v, s, atoms, o) {
  const bonds = o.bonds ? bondsFor(atoms, s) : atoms.map(() => []);
  const m = v.addModel();
  m.addAtoms(atoms.map((a, k) => ({ elem: s.symbols[a.i], x: a.cart[0], y: a.cart[1], z: a.cart[2],
    bonds: bonds[k], bondOrder: bonds[k].map(() => 1), index: k, serial: a.i + 1,
    atom: `${s.symbols[a.i]}${a.i + 1}`, properties: { image: a.image, shift: a.shift || [0, 0, 0] } })));
  m.setStyle({}, styleFor(o.style, o.bonds, o.opacity ?? 1));
  m.enableContextMenu({}, true);    // a right-click on an atom reaches the shell's atom menu
  return m;
}

/* The drawn box: a periodic axis spans n cells; a non-periodic one (a SLAB
 * has no c) spans the atoms, plus vacuum padding on each side if asked. */
function cellBox(s, n = 1) {
  const pad = ($("#opt-vacuum") && $("#opt-vacuum").checked) ? 4.0 : 0;
  let o = (s.origin || [0, 0, 0]).slice();
  const C = s.cell.map((r, k) => {
    if (s.pbc[k]) return mul(r, n);
    const u = mul(r, 1 / norm(r));
    o = sub(o, mul(u, pad));
    return add(r, mul(u, 2 * pad));
  });
  return { o, C };
}
function drawCell(v, s, color, radius = 0.025, n = 1, dashed = false) {
  const { o, C } = cellBox(s, n);
  const corners = [[0, 0, 0], C[0], C[1], C[2], add(C[0], C[1]), add(C[0], C[2]), add(C[1], C[2]), add(add(C[0], C[1]), C[2])].map((p) => add(p, o));
  const edges = [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]];
  edges.forEach(([a, b]) => v.addCylinder({ start: xyz(corners[a]), end: xyz(corners[b]), radius,
    color, fromCap: 1, toCap: 1, dashed, dashLength: 0.25, gapLength: 0.18 }));
  return corners;
}

/* a, b, c just past the far end of each cell vector (the edges that leave
 * the origin corner), so each letter reads as the label of its edge. */
function axisLabels(v, s, n = 1) {
  const cols = [cssVar("--axis-a"), cssVar("--axis-b"), cssVar("--axis-c")];
  const { o, C } = cellBox(s, n);
  ["a", "b", "c"].forEach((name, k) => {
    if (!s.pbc[k]) return;
    const end = add(o, C[k]), len = norm(C[k]) || 1;
    const pos = add(end, mul(C[k], Math.min(1.1, 0.25 * len) / len));
    v.addLabel(name, { position: xyz(pos), fontColor: cols[k], fontSize: 14, bold: true,
      inFront: true, showBackground: true, backgroundColor: cssVar("--surface"),
      backgroundOpacity: 0.8, borderThickness: 0, alignment: "center" });
  });
}

function structOpts() {
  return { cell: $("#opt-cell").checked, boundary: $("#opt-boundary").checked,
    supercell: $("#opt-super").checked, labels: $("#opt-labels").checked, ...effDraw($("#opt-style").value, $("#opt-bonds").checked) };
}

function drawStructure(keepView = false) {
  const s = state.structure;
  if (!s) return;
  // the editor's scene draws viewer-main once it has loaded (viewport/scene.js)
  if (window.MACE.sceneDraw) { window.MACE.sceneDraw(s, keepView); return; }
  if (window.MACE.sceneLoading) { window.MACE.sceneLoading.then(() => drawStructure(keepView), () => {}); return; }
  const v = viewer("viewer-main");
  if (!v) return;
  const view = keepView ? v.getView() : null;
  v.removeAllModels(); v.removeAllShapes(); v.removeAllLabels();
  const o = structOpts();
  const atoms = displayAtoms(s, o);
  const m = addModel(v, s, atoms, o);
  // the a, b, c directions show in the shell's axis triad (bottom left), not at the cell corners
  if (o.cell && !(s.pbc && !s.pbc.some(Boolean))) drawCell(v, s, /^#[0-9a-f]{6}$/i.test(setting("cellColor", "")) ? setting("cellColor", "") : cssVar("--cell"), 0.03, o.supercell ? 2 : 1);
  const labelMode = setting("labels", "both");
  if (o.labels) atoms.forEach((a) => { if (!a.image) v.addLabel(labelMode === "element" ? s.symbols[a.i] : `${s.symbols[a.i]}${a.i + 1}`,
    { position: xyz(a.cart), fontSize: num("labelSize", 12), fontColor: cssVar("--text"), backgroundColor: cssVar("--surface"),
      backgroundOpacity: 0.75, borderThickness: 0, inFront: true }); });
  m.setHoverable({}, true, (atom) => {
    const tip = $("#hover-tip");
    const f = s.frac[atom.serial - 1];
    tip.textContent = `${atom.atom}  (${f.map((x) => x.toFixed(4)).join(", ")})` +
      (s.symmetry.available ? `  Wyckoff ${s.symmetry.wyckoffs[atom.serial - 1]}` : "");
    tip.classList.remove("hidden");
    state.hover = { i: atom.serial - 1, image: (atom.properties && atom.properties.shift) || [0, 0, 0] };
  }, () => { $("#hover-tip").classList.add("hidden"); state.hover = null; });
  v.setHoverDuration(60);
  if (view) v.setView(view); else fitView(v, s, atoms, o.supercell ? 2 : 1);
  v.render();
}
$$("#tab-structure .viewer-toolbar input, #opt-style, #opt-colors").forEach((el) =>
  el.addEventListener("change", () => { drawStructure(true); if (state.traj && state.cmp) drawOpt(); }));
/* A Settings change that changes what the 3D views draw: redraw them, camera kept. */
function redrawViews() {
  Object.values(state.viewers).forEach((v) => v && v.setBackgroundColor(viewerBg()));
  if (state.structure) drawStructure(true);
  if (state.traj) drawOpt();
  if (state.structure) renderStructureInfo(state.structure);
}

/* Fit the atoms and the cell box, seen from a slightly tilted side view
 * with c pointing up. 3Dmol's zoomTo fits a bounding box to the height of
 * the canvas only, so a narrow pane (side by side) cut atoms off left and
 * right: the zoom is scaled by the aspect ratio. In a hidden tab the canvas
 * is 0x0, so the fit waits until showTab() makes the viewer visible. */
function fitPoints(s, atoms, n = 1) {
  const pts = atoms.map((a) => a.cart);
  if (s && s.cell) {
    const { o, C } = cellBox(s, n);
    for (const i of [0, 1]) for (const j of [0, 1]) for (const k of [0, 1])
      pts.push(add(o, add(add(mul(C[0], i), mul(C[1], j)), mul(C[2], k))));
  }
  return pts;
}
/* The rotation center of a fitted view: the drawn cell's center (n cells a
 * side), a slab's at its mid-height, a molecule's atoms' middle (geom/core.js
 * pivotOf); before that module is here, the middle of the fitted points. */
function pivotFor(s, atoms, n = 1, pts = null) {
  const cart = atoms.map((a) => a.cart);
  if (window.MACE.pivotOf && s) return window.MACE.pivotOf(s.cell ? s : { ...s, pbc: [false, false, false] }, { lo: [0, 0, 0], hi: [n, n, n] }, cart);
  if (s && s.cell && s.pbc && s.pbc.every(Boolean)) {
    const { o, C } = cellBox(s, n);
    return add(o, mul(add(add(C[0], C[1]), C[2]), 0.5));
  }
  const P = cart.length ? cart : pts || [];
  if (!P.length) return [0, 0, 0];
  return [0, 1, 2].map((k) => (Math.min(...P.map((p) => p[k])) + Math.max(...P.map((p) => p[k]))) / 2);
}
/* opts.tilt: the Phase 1-3 view (a fixed turn from no rotation, a loose sphere fit) instead of the standard view's tight fit;
 * opts.pivot: the rotation center (side-by-side views share one) */
function fitView(v, s, atoms, n = 1, pts = null, opts = {}) {
  pts = pts || fitPoints(s, atoms, n);
  const pivot = opts.pivot || pivotFor(s, atoms, n, pts);
  const doFit = () => {
    const m = v.addModel();
    m.addAtoms(pts.map((p, k) => ({ elem: "H", x: p[0], y: p[1], z: p[2], index: k, bonds: [] })));
    m.setStyle({}, {});
    v.zoomTo({ model: m });
    v.removeModel(m);
    // the cell's standard view (c up, as the structure viewport shows it) once geom/core.js is here
    const q = !opts.tilt && window.MACE.standardView && s && s.cell ? window.MACE.standardView(s.cell, s.pbc || [true, true, true]) : null;
    if (q) { const view = v.getView(); v.setView([...view.slice(0, 4), ...q]); }
    // the tilt from no rotation, so a refit (Reset View) lands where the first fit did
    else { const view = v.getView(); v.setView([...view.slice(0, 4), 0, 0, 0, 1]); v.rotate(-75, "x"); v.rotate(-20, "y"); }
    // the view's center is the rotation center
    { const view = v.getView(); v.setView([-pivot[0], -pivot[1], -pivot[2], ...view.slice(3)]); }
    const el = v.getCanvas ? v.getCanvas() : null;
    const w = el ? el.clientWidth : 0, h = el ? el.clientHeight : 0;
    if (opts.tilt) {
      // the Phase 1-3 zoom too: a sphere around the points, about 1.6 Å of margin
      const c = mul(pts.reduce((t, p) => add(t, p), [0, 0, 0]), 1 / pts.length);
      const R = Math.max(1, ...pts.map((p) => norm(sub(p, c))));
      v.zoom(0.92 * (R / (R + 1.6)) * Math.min(1, w > 0 && h > 0 ? w / h : 1));
    } else if (w > 0 && h > 0 && v.modelToScreen) {
      // a tight fit of the projected points (atom centers, cell corners), with
      // room for the drawn balls: 85% of the limiting side, around the rotation center
      const rc = el.getBoundingClientRect(), xyz = pts.map((p) => ({ x: p[0], y: p[1], z: p[2] }));
      const half = () => {
        // 3Dmol updates its size and matrices when it paints (once per frame here): paint this view now
        if (!(v.paintNow && v.paintNow())) { try { if (v.setSlabAndFog) v.setSlabAndFog(); if (v.scene && v.scene.updateMatrixWorld) v.scene.updateMatrixWorld(true); } catch (e) { /* not drawn yet */ } }
        const c = v.modelToScreen(xyz.length ? { x: pivot[0], y: pivot[1], z: pivot[2] } : { x: 0, y: 0, z: 0 });
        const cx = c.x - rc.left, cy = c.y - rc.top;
        let hw = 0, hh = 0;
        for (const p of v.modelToScreen(xyz)) {
          hw = Math.max(hw, Math.abs(p.x - rc.left - cx)); hh = Math.max(hh, Math.abs(p.y - rc.top - cy));
        }
        return { hw, hh };
      };
      for (let pass = 0; pass < 3; pass++) {
        const b = half();
        if (!(b.hw > 0 && b.hh > 0)) break;
        const f = 0.85 * Math.min(w / (2 * b.hw), h / (2 * b.hh));
        if (Math.abs(f - 1) < 0.005) break;
        v.zoom(f);
      }
    }
    v.render();
  };
  if (v.visible && !v.visible()) { v.pendingFit = doFit; return; }
  v.pendingFit = null;   // a fit queued while hidden is for an older structure
  doFit();
}

/* look straight down a cell vector */
function viewAlong(v, dir) {
  const u = mul(dir, 1 / norm(dir)), w = [0, 0, -1];
  let axis = cross(u, w), c = dot(u, w), qv;
  if (c < -0.999999) qv = [1, 0, 0, 0];
  else { const s2 = Math.sqrt((1 + c) * 2); qv = [axis[0] / s2, axis[1] / s2, axis[2] / s2, s2 / 2]; }
  const view = v.getView();
  view[4] = qv[0]; view[5] = qv[1]; view[6] = qv[2]; view[7] = qv[3];
  v.setView(view); v.render();
}
$$("[data-view]").forEach((b) => b.addEventListener("click", () => {
  const v = state.viewers["viewer-main"], s = state.structure;
  if (!v || !s) return;
  if (b.dataset.view === "reset") { drawStructure(false); return; }
  viewAlong(v, s.cell["abc".indexOf(b.dataset.view)]);
}));

/* kind: the document's file kind; an edited structure's payload says "edit" */
let infoFor = null;
function renderStructureInfo(s, kind) {
  if (kind === undefined && infoFor && infoFor.s === s) kind = infoFor.kind;   // a redraw keeps it
  infoFor = { s, kind };
  $("#s-title").textContent = s.path;
  $("#s-title").title = s.path;
  $("#s-formula").innerHTML = esc(s.reduced_formula || s.formula) +
    (s.z ? ` <span class="zbadge" title="formula units in this cell (cell formula ${esc(s.formula)})">Z = ${s.z}</span>` : "");
  const b = [`<span class="badge accent">${esc(s.dimensionality)}</span>`,
    `<span class="badge">${s.numbers.length} atoms</span>`,
    `<span class="badge">${esc(kind || s.source.kind)}</span>`];
  if (s.frame_label) b.push(`<span class="badge">${esc(s.frame_label)} geometry</span>`);
  // an EXTERNAL deck is drawn from the file its job stages as fort.34 (or its .out)
  const src = s.source || {};
  if (src.geometry_from) {
    const name = src.geometry_from.split(/[\\/]/).pop();
    const why = src.geometry_staged ? `the job script copies ${name} to fort.34 (${src.geometry_keyword || "EXTERNAL"})`
      : `the deck reads its geometry from fort.34 (${src.geometry_keyword || "EXTERNAL"}), which its job script stages from a file that is not here; ${name} is the closest geometry found`;
    b.push(`<span class="badge" id="s-geometry-from" title="${esc(why)}">geometry from ${esc(name)}</span>`);
  }
  if (src.geometry_edits && src.geometry_edits.length) {
    b.push(`<span class="badge" id="s-geometry-edits" title="The deck edits this geometry after its atoms (${esc(src.geometry_edits.join(", "))}); CRYSTAL runs the edited structure, drawn here before the edits">before ${esc(src.geometry_edits.join(", "))}</span>`);
  }
  // plan E10: a periodic cell that holds a molecule, a slab or a polymer says so (a suggestion; Setup asks)
  const hint = s.dim_hint;
  if (hint && hint.from === "vacuum" && hint.dim !== "CRYSTAL" && s.dimensionality === "CRYSTAL") {
    const word = { SLAB: "a slab", POLYMER: "a polymer", MOLECULE: "a molecule" }[hint.dim];
    const along = hint.vacuum_axes.map((a) => `${hint.vacuum[a].toFixed(1)} Å along ${a}`).join(", ");
    const turn = hint.oriented ? "" : hint.dim === "SLAB" ? " (MACE rotates it so the vacuum is along c for a SLAB deck)"
      : " (MACE rotates it so the chain runs along a for a POLYMER deck)";
    b.push(`<span class="badge dim-hint" id="s-dim-hint" title="Vacuum ${esc(along)}${esc(turn)}. Calculation Setup › Dimensionality sets it up as ${esc(hint.dim)}.">Looks like ${word} (vacuum ${esc(along)})</span>`);
  }
  if (s.energy_au != null) b.push(`<span class="badge">E = ${s.energy_au.toFixed(6)} Ha</span>`);
  $("#s-badges").innerHTML = b.join("");
  const y = s.symmetry;
  $("#s-sym").innerHTML = y.available ? `
    <div class="symline"><span class="hm" title="${esc(y.international)}">${hm(y.international)}</span><span class="sgnum">No. ${y.number}</span></div>
    <table class="kv">
      <tr><td>Point group</td><td class="pg">${hm(y.pointgroup)}</td></tr>
      <tr><td>Hall symbol</td><td>${esc(y.hall)}</td></tr>
      <tr><td>Operations</td><td>${y.n_operations_conventional || y.n_operations}</td></tr>
      <tr><td>Unique atoms</td><td>${y.unique_atoms} of ${s.numbers.length}</td></tr>
      <tr><td>Tolerance</td><td>${y.symprec} Å</td></tr>
    </table>` : `<p class="muted small">${esc(y.note)}</p>` +
      (y.not_computed && window.MACE.computeSymmetry ? `<button type="button" class="tb-btn-text sym-compute">Compute Now</button>` : "");
  const cbtn = $("#s-sym .sym-compute");
  if (cbtn) cbtn.addEventListener("click", () => { cbtn.disabled = true; cbtn.textContent = "Computing…"; window.MACE.computeSymmetry(); });
  $("#s-lattice").innerHTML = latticeRows(s);
  $("#s-natoms").textContent = `(${s.numbers.length})`;
  // Phase 4: where a built or dropped structure came from, and its hints for Setup
  let prov = document.getElementById("s-prov");
  if (!prov) { prov = document.createElement("div"); prov.id = "s-prov"; prov.className = "prov muted small"; $("#s-badges").after(prov); }
  const meta = s.doc_meta || {}, p = meta.provenance || {}, charge = meta.hints && meta.hints.charge;
  const lines = [];
  if (p.label) lines.push(`<div title="${esc(p.parent || "")}">${esc(p.label)}</div>`);
  if (charge !== undefined && charge !== null && charge !== 0) lines.push(`<div>Defect charge for Setup: ${charge > 0 ? "+" : ""}${esc(charge)}</div>`);
  if (window.MACE.untitledHint && window.MACE.untitledHint(s)) lines.push(`<div class="unsaved">${esc(window.MACE.untitledHint(s))}</div>`);
  prov.innerHTML = lines.join("");
  prov.hidden = !lines.length;
  // a structure document lists its atoms in the editor's virtualized table instead
  if (window.MACE.editorTable && window.MACE.editorTable(s)) { $("#s-atoms").innerHTML = ""; return; }
  $("#s-atoms").innerHTML = `<thead><tr><th>#</th><th>El</th><th class="num">x</th><th class="num">y</th><th class="num">z</th><th>Wyck.</th></tr></thead><tbody>` +
    s.frac.map((f, i) => `<tr><td>${i + 1}</td><td>${s.symbols[i]}</td>${f.map((x) => `<td class="num">${x.toFixed(num("decimals.coord", 5))}</td>`).join("")}<td>${y.available ? y.wyckoffs[i] : ""}</td></tr>`).join("") + "</tbody>";
}

/* A slab's c (a rod's b and c) is not a lattice vector: CRYSTAL has no
 * vacuum, so those rows say so, and the cell size is an area (a length). */
function latticeRows(s) {
  const L = s.lattice, per = s.pbc.filter(Boolean).length;
  // decimals as the spec's Settings defaults: lengths 4, angles 3; "90.000°" takes no space
  const dl = num("decimals.length", 4), da = num("decimals.angle", 3);
  const rows = [["a", L.a, " Å", dl], ["b", L.b, " Å", dl], ["c", L.c, " Å", dl],
    ["α", L.alpha, "°", da], ["β", L.beta, "°", da], ["γ", L.gamma, "°", da]];
  const out = rows.map(([k, v, u, d], i) => (i < 3 && !s.pbc[i])
    ? `<tr><td>${k}</td><td class="muted" title="not periodic: the drawn box only spans the atoms">not periodic</td></tr>`
    : `<tr><td>${k}</td><td>${fmt(v, d)}${u}</td></tr>`);
  if (per === 3) out.push(`<tr><td>Volume</td><td>${fmt(L.volume, 3)} Å³</td></tr>`);
  else if (per === 2) {
    const [i, j] = [0, 1, 2].filter((k) => s.pbc[k]);
    out.push(`<tr><td>Area</td><td>${fmt(norm(cross(s.cell[i], s.cell[j])), 3)} Å²</td></tr>`);
  } else if (per === 1) {
    const i = s.pbc.indexOf(true);
    out.push(`<tr><td>Length</td><td>${fmt(norm(s.cell[i]), 3)} Å</td></tr>`);
  }
  return out.join("");
}

async function loadStructure(path, data) {
  const s = data || await api("structure?" + q({ path }));
  await load3Dmol();
  state.structure = s;
  $("#structure-empty").classList.add("hidden");
  $("#structure-view").classList.remove("hidden");
  renderStructureInfo(s);
  requestAnimationFrame(() => { const v = viewer("viewer-main"); if (v) v.resize(); drawStructure(); });
}

/* ---------------------------------------------------------------- OPT */
/** The energies on the optimizer's surface (outindex v3 opt_energies_au: the
 * first point's dispersion/gCP-corrected total, the final frame's OPT END
 * energy); older summaries and two-structure comparisons have only energies_au. */
function optEnergies(t) {
  return (t && (t.opt_energies_au || t.energies_au)) || [];
}
/** /api/out/summary -> the trajectory shape the views use: frames as a
 * sparse array (the first and last filled), labels expanded from runs. */
function expandSummary(r) {
  if (!r || !r.summary) return r;
  const n = r.n_steps, frames = new Array(n).fill(null), labels = [];
  for (const [k, f] of Object.entries(r.frames_at || {})) frames[Number(k)] = f;
  for (const [lab, count] of r.label_runs || []) for (let i = 0; i < count; i++) labels.push(lab);
  const t = { ...r, frames, labels };
  if (r.opt_energies_at) {           // outindex v3: the optimizer's surface, sent where it differs
    t.opt_energies_au = (r.energies_au || []).slice();
    for (const [k, e] of Object.entries(r.opt_energies_at)) t.opt_energies_au[Number(k)] = e;
  }
  delete t.frames_at; delete t.label_runs; delete t.opt_energies_at;
  t.pending = new Map();
  return t;
}
/** Step i of a run, fetched once (and steps i±1 behind it, for scrubbing). */
async function ensureStep(t, i, prefetch = true) {
  if (!t || !t.frames || i < 0 || i >= t.frames.length) return null;
  if (prefetch) for (const j of [i - 1, i + 1]) if (j >= 0 && j < t.frames.length && !t.frames[j]) ensureStep(t, j, false).catch(() => {});
  if (t.frames[i]) return t.frames[i];
  if (!t.pending) t.pending = new Map();
  if (!t.pending.has(i)) {
    t.pending.set(i, api("out/step?" + q({ path: t.path, step: i })).then((f) => { t.frames[i] = f; t.pending.delete(i); return f; },
      (e) => { t.pending.delete(i); throw e; }));
  }
  return t.pending.get(i);
}
async function loadOut(path, traj) {
  const t = traj || expandSummary(await api("out/summary?" + q({ path })));
  const last = t.frames[t.frames.length - 1];
  last.path = path;
  // no optimization: a single point, not the "initial" geometry of a run that moves. CRYSTAL can print
  // an SP's geometry more than once (after each geometry edit of the input): the last is the one its
  // SCF ran on, and the run is still one geometry, not a trajectory.
  if (!t.is_opt && last.frame_label === "initial") last.frame_label = "single-point";
  await loadStructure(path, last);
  if (t.frames.length > 1 && t.is_opt) {
    setTrajectory(t, path, t.opt_status || (t.is_opt ? "optimization (no OPT END line: unfinished)" : ""));
    markTab("opt", true);
    showTab("opt");
    startLive(path);
    return "trajectory";
  }
  showTab("structure");
  return "structure";
}

/* ---------------------------------------------------------------- live runs (PLAN_analysis C18.4)
 * A run with no terminal marker is never dropped from refresh: out/live (a
 * stat and the last 64 KB) every 15 s while it is live, every 60 s while it
 * looks stalled, until CRYSTAL's end marker appears or another run opens.
 * New points append without moving the selected frame, unless the newest
 * frame was selected and Follow the run is on. */
const FOLLOW_KEY = "mace.gui.followRun";
state.live = null;
function followOn() { try { return localStorage.getItem(FOLLOW_KEY) !== "0"; } catch (e) { return true; } }
$("#o-follow").checked = followOn();
$("#o-follow").addEventListener("change", (e) => { try { localStorage.setItem(FOLLOW_KEY, e.target.checked ? "1" : "0"); } catch (x) { /* private */ } });
function ago(s) { return s < 90 ? `${Math.round(s)} s ago` : s < 5400 ? `${Math.round(s / 60)} min ago` : `${(s / 3600).toFixed(1)} h ago`; }
function liveChip(r) {
  const el = $("#o-live-chip");
  if (!el) return;
  $("#o-live").classList.toggle("hidden", !r);
  if (!r) return;
  el.className = "live-chip" + (r.finished ? " done" : r.live ? " on" : " stalled");
  el.textContent = r.finished ? "Finished" : r.live ? (r.job_state ? `Running (job ${r.job_state}) · last write ${ago(r.age_s)}` : `Live · updated ${ago(r.age_s)}`)
    : `No end marker; last write ${ago(r.age_s)}`;
  $("#o-follow").closest("label").classList.toggle("hidden", !!r.finished);
}
function stopLive() { if (state.live) { clearTimeout(state.live.timer); state.live = null; } }
async function startLive(path) {
  stopLive();
  let r;
  try { r = await api("out/live?" + q({ path })); } catch (e) { return; }
  if (!state.traj || state.traj.path !== path) return;
  state.live = { path, timer: 0, last: r, stamp: r.file };
  liveChip(r.finished ? null : r);
  if (!r.finished) scheduleLive();
}
function scheduleLive() {
  const L = state.live;
  if (!L) return;
  clearTimeout(L.timer);
  L.timer = setTimeout(pollLive, (L.last && L.last.live ? 15 : 60) * 1000 * (window.__maceLiveScale || 1));
}
async function pollLive() {
  const L = state.live;
  if (!L) return;
  if (!state.traj || state.traj.path !== L.path) { stopLive(); return; }
  if (document.hidden) { scheduleLive(); return; }
  try {
    const r = await api("out/live?" + q({ path: L.path }));
    if (state.live !== L) return;
    const changed = !L.stamp || !r.file || String(r.file.mtime_ns) !== String(L.stamp.mtime_ns) || r.file.size !== L.stamp.size;
    L.last = r; L.stamp = r.file; L.polls = (L.polls || 0) + 1;
    if (changed) await refreshRun(L.path);
    liveChip(r);
    if (r.finished) { L.done = true; return; }          // the end marker: polling stops
  } catch (e) { /* the next poll tries again */ }
  scheduleLive();
}
/** The run grew: new points append; the selected frame and the view stay. */
async function refreshRun(path) {
  const old = state.traj;
  const t = expandSummary(await api("out/summary?" + q({ path })));
  if (state.traj !== old) return;
  const n0 = old.frames.length, wasLast = state.frame >= n0 - 1;
  for (let i = 0; i < Math.min(n0 - 1, t.frames.length); i++) if (!t.frames[i] && old.frames[i] && (old.labels[i] === t.labels[i])) t.frames[i] = old.frames[i];
  t.path = old.path;
  state.traj = t;
  loadTrajConv(t);
  const fr = $("#o-frame");
  fr.max = t.frames.length - 1;
  $("#o-frame-wrap").classList.toggle("hidden", t.frames.length < 3);
  if (wasLast && $("#o-follow").checked && t.frames.length !== n0) {
    state.frame = t.frames.length - 1;
    fr.value = state.frame;
    fr.dispatchEvent(new Event("input"));
  } else {
    state.frame = Math.min(state.frame, t.frames.length - 1);
    fr.value = state.frame;
    $("#o-frame-out").textContent = `${state.frame} · ${t.labels[state.frame] || "final"}`;
    renderComparison(); drawEnergy();
  }
  document.dispatchEvent(new CustomEvent("mace-run-updated", { detail: { path, n: t.frames.length } }));
}

function setTrajectory(t, title, status) {
  state.traj = t;
  state.frame = t.frames.length - 1;
  state.cmp = t.comparison;
  state.refit = true;    // a new run: fit it, do not keep the last run's view
  $("#opt-empty").classList.add("hidden");
  $("#opt-view").classList.remove("hidden");
  $("#o-title").textContent = title;
  $("#o-status").textContent = status || "";
  const fr = $("#o-frame");
  fr.max = t.frames.length - 1; fr.value = state.frame;
  $("#o-frame-out").textContent = `${state.frame} · ${(t.labels || [])[state.frame] || "final"}`;
  $("#o-frame-wrap").classList.toggle("hidden", !t.path || t.frames.length < 3);
  autoScale();
  renderComparison();
  requestAnimationFrame(() => { drawOpt(); drawEnergy(); });
  if (t.path) loadTrajConv(t);
}

/* The arrows: the longest drawn 1.5 Å long. The slider reaches further than ×200 when the
 * displacements are smaller than 7.5 mÅ, so a small relaxation still shows. */
function autoScale() {
  const m = state.cmp ? dispSet(state.cmp).max : 0;
  const s = m > 1e-9 ? Math.max(1, Math.min(100000, Math.round(1.5 / m))) : 1;
  const sl = $("#o-scale");
  sl.max = Math.max(200, s);
  sl.value = s; $("#o-scale-out").textContent = s;
}

/* The displacement shown (the server's compare): "cart" is r_final − r_initial in Å with the
 * cell change included and the mean shift removed, the default when the cell changed; "internal"
 * is the relaxation relative to the cell (the change of fractional coordinates in the final
 * cell), zero for atoms that symmetry pins while only the cell changes (AB graphite along c). */
const NEGLIGIBLE_DR = 1e-4;          // Å: below CRYSTAL's printed coordinate precision, "did not move"
function dispMode(c) {
  if (c && c.aligned === "molecule") return "molecule";     // no cell: Cartesian, rigid motion removed (compare())
  if (!c || !c.displacements_cart) return "internal";
  const m = store("mace.gui.cmp.dispMode");
  return m === "cart" || m === "internal" ? m : (c.cell_changed ? "cart" : "internal");
}
function dispSet(c) {
  const mode = dispMode(c);
  return mode === "cart" ? { mode, list: c.displacements_cart, max: c.max_displacement_cart, rms: c.rms_displacement_cart }
    : { mode, list: c.displacements || [], max: c.max_displacement || 0, rms: c.rms_displacement || 0 };
}
/* a length as people read it: Å from 0.1 Å, pm below (3 significant figures), scientific when tiny */
function lenUnit(hi) { return hi >= 0.1 || hi < 1e-5 ? "Å" : "pm"; }
function fmtLen(v, unit = lenUnit(Math.abs(v))) {
  if (!v) return `0 ${unit}`;
  const x = unit === "pm" ? v * 100 : v, ax = Math.abs(x);
  return `${ax >= 1e-3 && ax < 1e4 ? String(+x.toPrecision(3)) : x.toExponential(2)} ${unit}`;
}
/* the cell's change in words: "a, b 2.4520→2.4528 Å (+0.03%), c 6.7000→6.1752 Å (−7.83%), V …" */
function cellChangeText(c) {
  const rows = (c.cell || []).filter((r) => r.percent !== null && Math.abs(r.delta) > (["alpha", "beta", "gamma"].includes(r.param) ? 1e-6 : 1e-6 * Math.abs(r.before)));
  const greek = { alpha: "α", beta: "β", gamma: "γ", volume: "V" };
  const pct = (p) => `${p > 0 ? "+" : p < 0 ? "−" : ""}${Math.abs(p) < 0.01 ? Math.abs(p).toExponential(1) : Math.abs(p).toFixed(2)}%`;
  const out = [];
  for (const r of rows) {
    const prev = out.length ? out[out.length - 1] : null;
    if (prev && prev.r.param.length === 1 && r.param.length === 1 && Math.abs(prev.r.before - r.before) < 1e-9 && Math.abs(prev.r.after - r.after) < 1e-9) { prev.names.push(r.param); continue; }
    out.push({ r, names: [greek[r.param] || r.param] });
  }
  return out.map(({ r, names }) => {
    const ang = ["alpha", "beta", "gamma"].includes(r.param), u = r.param === "volume" ? " Å³" : ang ? "°" : " Å";
    const d = r.param === "volume" ? 2 : 4;
    return `${names.join(", ")} ${fmt(r.before, d)}→${fmt(r.after, d)}${u} (${pct(r.percent)})`;
  }).join(", ");
}

/* The comparison's tables (plan B7): one model per sub-tab, rendered, sorted,
 * copied as TSV and exported from the same rows. */
const CMP_SUBS = ["all", "energy", "cell", "bonds", "disp", "sym"];
state.cmpSub = "all";
state.cmpSort = {};
function cmpTables() {
  const c = state.cmp, t = state.traj;
  if (!c || !t) return {};
  const now = state.frame === t.frames.length - 1 ? "final" : "selected";
  const greek = { alpha: "α", beta: "β", gamma: "γ", volume: "V" };
  const es = optEnergies(t), bare = t.energies_au || [];
  const e0 = es.find((e) => e != null);
  const sy = c.symmetry_change;
  // frame 0 of a D3/gCP run: the bare SCF energy differs from the corrected one CRYSTAL optimizes
  const tip = (e, i) => (bare[i] != null && e != null && Math.abs(bare[i] - e) > 1e-10
    ? `SCF energy ${fmt(bare[i], 8)} Ha; with the dispersion/gCP correction ${fmt(e, 8)} Ha (CRYSTAL's optimization uses the corrected value)` : "");
  return {
    energy: { title: "energy", cols: [["frame", "frame", 0], ["label", "label"], ["E (Ha)", "e", 8], ["ΔE (meV)", "de", 2]],
      rows: es.map((e, i) => ({ frame: i, label: (t.labels || [])[i] || "", e, de: e != null && e0 != null ? (e - e0) * HARTREE_MEV : null, tip: tip(e, i) })) },
    cell: { title: "cell", cols: [["", "param"], ["initial", "before", 4], [now, "after", 4], ["Δ", "delta", 4, true], ["%", "percent", 2, true]],
      rows: c.cell.map((r) => ({ ...r, param: greek[r.param] || r.param })) },
    bonds: { title: "bonds", cols: [["pair", "pair"], ["n", "count", 0], ["initial", "before", 4], [now, "after", 4], ["Δ", "delta", 4, true]],
      rows: c.bonds.map((r) => ({ ...r, tip: `e.g. atoms ${r.example.join("-")}` })), empty: "no bonded pairs" },
    disp: { title: "displacements", cols: [["atom", "atom"], ["|Δr| Å", "length", 4], ["Δx", "dx", 4, true], ["Δy", "dy", 4, true], ["Δz", "dz", 4, true]],
      rows: dispSet(c).list.map((d) => ({ atom: `${d.symbol}${d.atom}`, index: d.atom, length: d.length, dx: d.vector[0], dy: d.vector[1], dz: d.vector[2] })),
      sort: ["length", -1] },
    sym: { title: "symmetry", cols: [["", "end"], ["space group", "group"]],
      rows: sy ? [{ end: "initial", group: sy.before || "— (not 3D, or too large)" }, { end: now, group: sy.after || "— (not 3D, or too large)" },
        { end: "change", group: sy.before && sy.after ? (sy.before === sy.after ? `${sy.before} → ${sy.after} (kept)` : `${sy.before} → ${sy.after}`) : "—" }] : [],
      empty: "computing…" },
  };
}
function sortedRows(key, tb) {
  const [k, dir] = state.cmpSort[key] || tb.sort || [null, 1];
  if (!k) return tb.rows;
  return tb.rows.slice().sort((x, y) => {
    const a = x[k], b = y[k];
    return (typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b), undefined, { numeric: true })) * dir;
  });
}
/* a cell's HTML: text escaped; a signed number through signed() (its own markup) */
function cellText(v, col) {
  if (v == null) return "—";
  if (typeof v !== "number" || col[2] === undefined) return esc(String(v));
  return col[3] ? signed(v, col[2]) : esc(fmt(v, col[2]));
}
function tableHtml(key, tb) {
  const [sk, sd] = state.cmpSort[key] || tb.sort || [null, 1];
  const rows = sortedRows(key, tb);
  const head = tb.cols.map((col) => `<th class="${col[2] !== undefined ? "num" : ""}" aria-sort="${sk === col[1] ? (sd > 0 ? "ascending" : "descending") : "none"}"><button class="th-sort" type="button" data-key="${col[1]}"${col[0] ? "" : ` aria-label="Sort by ${({ param: "parameter", end: "geometry" })[col[1]] || col[1]}"`}>${esc(col[0])}</button></th>`).join("");
  const body = rows.length ? rows.map((r, i) => `<tr tabindex="${i ? -1 : 0}" aria-rowindex="${i + 2}"${r.tip ? ` title="${esc(r.tip)}"` : ""}>` +
    tb.cols.map((col) => `<td class="${col[2] !== undefined ? "num" : ""}">${cellText(r[col[1]], col)}</td>`).join("") + "</tr>").join("")
    : `<tr><td colspan="${tb.cols.length}" class="muted">${esc(tb.empty || "nothing to show")}</td></tr>`;
  return `<thead><tr aria-rowindex="1">${head}</tr></thead><tbody>${body}</tbody>`;
}
const CMP_TABLE_EL = { cell: "#o-cell", bonds: "#o-bonds", disp: "#o-disp", sym: "#o-sym" };
function renderComparison() {
  const c = state.cmp, t = state.traj;
  if (!c) return;
  const vol = c.cell.find((r) => r.param === "volume");
  const es = optEnergies(t);
  const eSel = es[state.frame] ?? null;
  const e0 = es.find((e) => e != null) ?? null;
  const D = dispSet(c), du = lenUnit(D.max), dTip = D.mode === "cart" ? "r(final) − r(initial), the cell change included, the mean shift removed"
    : D.mode === "molecule" ? "r(final) − r(initial) once the molecule's rigid translation and rotation are taken out"
    : "relative to the cell (the change of fractional coordinates, in the final cell)";
  // Δr is a choice only for a periodic run whose cell changed (a molecule has no cell to be relative to).
  // Frame 0 compares the start with itself (no cell change): the choice keeps its place, invisible, while
  // the run's cell changes, so scrubbing the Frame slider never reflows the toolbar under the pointer
  const dmodeOk = (x) => !!(x && x.cell_changed && x.displacements_cart && dispSet(x).mode !== "molecule");
  const dmodeOn = dmodeOk(c), dmodeWrap = $("#o-dmode-wrap");
  dmodeWrap.hidden = !(dmodeOn || dmodeOk(t && t.comparison));
  dmodeWrap.style.visibility = dmodeOn ? "" : "hidden";
  $("#o-dmode").disabled = !dmodeOn;
  if (D.mode !== "molecule") $("#o-dmode").value = D.mode;
  $("#o-stats").innerHTML = `
    <div class="stat" title="${esc(dTip)}"><div class="k">max |Δr|</div><div class="v">${fmtLen(D.max, du)}</div></div>
    <div class="stat" title="${esc(dTip)}"><div class="k">rms |Δr|</div><div class="v">${fmtLen(D.rms, du)}</div></div>
    <div class="stat"><div class="k">${vol ? "ΔV" : "ΔE"}</div><div class="v">${vol ? (vol.percent > 0 ? "+" : "") + fmt(vol.percent, 2) + " %" :
      (eSel != null && e0 != null ? fmt((eSel - e0) * HARTREE_MEV, 1) + " meV" : "—")}</div></div>`;
  const tabs = cmpTables();
  for (const [k, sel] of Object.entries(CMP_TABLE_EL)) {
    const el = $(sel);
    el.innerHTML = tableHtml(k, tabs[k]);
    el.setAttribute("role", "grid");
    el.setAttribute("aria-rowcount", String(Math.max(1, tabs[k].rows.length) + 1));
    el.dataset.table = k;
  }
  const banner = $("#o-banner");
  const pos = c.match === "position";
  banner.classList.toggle("hidden", !pos);
  banner.textContent = pos ? `Atoms matched by position (max mismatch ${fmt(c.max_mismatch, 3)} Å)` : "";
  if (!c.symmetry_change && t.path && !state.symPending) fetchSymmetry();
  showSub(state.cmpSub, false);
}
async function fetchSymmetry() {
  const t = state.traj, want = state.frame;
  state.symPending = true;
  try {
    const r = await api("trajectory/compare?" + q({ path: t.path, i: 0, j: want, sym: 1 }));
    if (state.traj === t && state.frame === want && state.cmp) { state.cmp.symmetry_change = r.symmetry_change; renderComparison(); }
  } catch (e) { /* the row stays "computing…": the tables still work */ }
  finally { state.symPending = false; }
}
function showSub(sub, focus = true) {
  if (!CMP_SUBS.includes(sub)) sub = "all";
  state.cmpSub = sub;
  $$(".cmp-subs [data-sub]").forEach((b) => { b.classList.toggle("active", b.dataset.sub === sub); b.setAttribute("aria-selected", String(b.dataset.sub === sub)); });
  $$("#o-info > .card[data-sub]").forEach((el) => el.classList.toggle("hidden", sub !== "all" && el.dataset.sub !== sub));
  $("#o-info").classList.toggle("cmp-one", sub !== "all");
  if (sub === "disp") $("#o-disp-details").open = true;
  if (sub === "energy" || sub === "all") requestAnimationFrame(drawEnergy);
  if (focus) { const r = sub !== "all" && CMP_TABLE_EL[sub] && $(CMP_TABLE_EL[sub] + " tbody tr[tabindex='0']"); if (r) r.focus(); }
}
/** The active table as {name, columns, rows} (Export Table and Copy TSV). */
function cmpActiveTable() {
  const tabs = cmpTables();
  const key = state.cmpSub === "all" ? "disp" : state.cmpSub;
  const tb = tabs[key];
  if (!tb) return null;
  return { name: tb.title, columns: tb.cols.map((col) => col[0] || col[1]), keys: tb.cols.map((col) => col[1]), rows: sortedRows(key, tb) };
}
function cmpTsv() {
  const tb = cmpActiveTable();
  if (!tb) return "";
  return [tb.columns.join("\t")].concat(tb.rows.map((r) => tb.keys.map((k) => (r[k] == null ? "" : r[k])).join("\t"))).join("\n");
}
function cmpExport(fmt) {
  const tb = cmpActiveTable();
  if (!tb) return null;
  const stem = ((state.traj && state.traj.path) || "comparison").split("/").pop().replace(/\.[^.]+$/, "");
  if (fmt === "json") {
    return { name: `${stem}_${tb.name}.json`, blob: new Blob([JSON.stringify({ table: tb.name, match: state.cmp.match || "index",
      columns: tb.columns, rows: tb.rows.map((r) => Object.fromEntries(tb.keys.map((k, i) => [tb.columns[i], r[k] ?? null]))) }, null, 1)], { type: "application/json" }) };
  }
  const cell = (v) => { const x = v == null ? "" : String(v); return /[",\n]/.test(x) ? `"${x.replace(/"/g, '""')}"` : x; };
  const csv = [tb.columns.map(cell).join(",")].concat(tb.rows.map((r) => tb.keys.map((k) => cell(r[k])).join(","))).join("\n") + "\n";
  return { name: `${stem}_${tb.name}.csv`, blob: new Blob([csv], { type: "text/csv" }) };
}
$(".cmp-subs").addEventListener("click", (e) => {
  const b = e.target.closest("[data-sub]");
  if (b) showSub(b.dataset.sub);
  if (e.target.closest(".cmp-copy")) document.dispatchEvent(new CustomEvent("mace-copy", { detail: { text: cmpTsv(), done: "Table copied" } }));
});
$("#o-info").addEventListener("click", (e) => {
  const th = e.target.closest(".th-sort");
  if (!th) return;
  const key = th.closest("table").dataset.table, k = th.dataset.key;
  const cur = state.cmpSort[key] || cmpTables()[key].sort || [null, 1];
  state.cmpSort[key] = [k, cur[0] === k ? -cur[1] : 1];
  renderComparison();
  const again = $(`${CMP_TABLE_EL[key]} .th-sort[data-key="${k}"]`);
  if (again) again.focus();
});
/* the roving-row keyboard model of the comparison grids (plan B6) */
$("#o-info").addEventListener("keydown", (e) => {
  const tr = e.target.closest && e.target.closest("tbody tr");
  if (!tr) return;
  const rows = [...tr.parentElement.children];
  const i = rows.indexOf(tr), page = 10;
  const to = { ArrowDown: i + 1, ArrowUp: i - 1, PageDown: i + page, PageUp: i - page, Home: 0, End: rows.length - 1 }[e.key];
  if (to === undefined) return;
  e.preventDefault();
  const j = Math.max(0, Math.min(rows.length - 1, to));
  rows.forEach((r, k) => r.setAttribute("tabindex", k === j ? "0" : "-1"));
  rows[j].focus();
});

const drawCellBox = (...a) => drawCell(...a), fitViewBox = (...a) => fitView(...a), fitPointsBox = (...a) => fitPoints(...a);
function drawOpt() {
  const t = state.traj, c = state.cmp;
  if (!t || !c) return;
  const s0 = t.frames[0], s1 = t.frames[state.frame];
  // a molecule (no periodic direction, plan E2/E10): it has no unit cell, so none is drawn or fitted
  const periodic = (s) => !!(s && s.pbc && s.pbc.some(Boolean));
  // the toolbar's Show Unit Cell, Show Bonds and the atom style apply here too
  const showCell = !$("#opt-cell") || $("#opt-cell").checked;
  const drawCell = (v, s, ...rest) => (periodic(s) && showCell ? drawCellBox(v, s, ...rest) : null);
  // its fit: the atoms' box widened by 1.6 A, so the balls of a small molecule are not cut by the edges
  const fitPoints = (s, atoms, n = 1) => {
    if (periodic(s)) return fitPointsBox(s, atoms, n);
    const P = atoms.map((a) => a.cart), lo = [0, 1, 2].map((k) => Math.min(...P.map((p) => p[k])) - 1.6),
      hi = [0, 1, 2].map((k) => Math.max(...P.map((p) => p[k])) + 1.6);
    const out = P.slice();
    for (const i of [lo[0], hi[0]]) for (const j of [lo[1], hi[1]]) for (const k of [lo[2], hi[2]]) out.push([i, j, k]);
    return out;
  };
  const fitView = (v, s, atoms, n = 1, pts = null, opts = {}) =>
    fitViewBox(v, periodic(s) ? s : { ...s, cell: null }, atoms, n, pts || fitPoints(s, atoms, n), opts);
  if (!s0 || !s1) return;                      // a step still loading
  const side = state.layout === "side";
  const showArrows = $("#o-arrows").checked, ghost = $("#o-ghost").checked;
  const scale = Number($("#o-scale").value);
  const vb = viewer("viewer-before");
  if (!vb) return;
  // a text alternative for the drawing (WCAG 1.1.1): the Timeline and the
  // Atoms tab below give the numbers
  const host = document.getElementById("opt-viewers");
  if (host) {
    host.setAttribute("role", "img");
    host.setAttribute("aria-label", `Optimization, ${s0.symbols.length} atoms: the first geometry and step ${state.frame} of ${t.frames.length - 1}` +
      `${s1.formula ? `, ${s1.formula}` : ""}. The Timeline and Atoms tabs in the bottom panel list the values.`);
  }
  const keep = vb.getModel() ? vb.getView() : null;
  [vb, side ? viewer("viewer-after") : null].forEach((v) => v && (v.removeAllModels(), v.removeAllShapes(), v.removeAllLabels(), v.arrowCount = 0));
  const atoms0 = c.before_cart.map((p, i) => ({ i, cart: p, image: false }));
  const s0w = { ...s0, cart: c.before_cart };
  const s1u = { ...s1, cart: c.after_unwrapped_cart };
  const atoms1 = s1u.cart.map((p, i) => ({ i, cart: p, image: false }));
  const base = effDraw(($("#opt-style") && $("#opt-style").value) || "ballstick", !$("#opt-bonds") || $("#opt-bonds").checked);
  const muted = cssVar("--muted"), accent = cssVar("--accent");
  renderLegend(c);
  if (state.layout === "twotone") {
    const ca = cssVar("--cmp-a"), cb = cssVar("--cmp-b");
    tint(addModel(vb, s0w, atoms0, base), s0w, (a) => mixHex(elementColor(s0w.symbols[a.serial - 1]), ca, 0.7), base);
    tint(addModel(vb, s1u, atoms1, base), s1u, (a) => mixHex(elementColor(s1u.symbols[a.serial - 1]), cb, 0.7), base);
    drawCell(vb, s0, ca, 0.02, 1, true);
    drawCell(vb, s1, cb, 0.03);
    if (keep && !state.refit) vb.setView(keep); else fitView(vb, s1, atoms0.concat(atoms1));
    state.refit = false;
    vb.render();
    return;
  }
  if (state.layout === "dmap") {
    const L = dispSet(c).list.map((d) => d.length);
    const lo = Math.min(...L), hi = Math.max(...L);
    tint(addModel(vb, s1u, atoms1, base), s1u, (a) => seqColor(hi > lo ? (L[a.serial - 1] - lo) / (hi - lo) : 0), base);
    drawCell(vb, s1, accent, 0.03);
    if (showArrows) arrows(vb, atoms1, c, scale);
    if (keep && !state.refit) vb.setView(keep); else fitView(vb, s1, atoms1);
    state.refit = false;
    vb.render();
    return;
  }
  if (side) {
    const va = viewer("viewer-after");
    if (!va) return;
    addModel(vb, s0w, atoms0, base); drawCell(vb, s0, muted, 0.03);
    addModel(va, s1u, atoms1, base); drawCell(va, s1, accent, 0.03);
    if (showArrows) arrows(va, atoms1, c, scale);
    const pts = fitPoints(s0, atoms0).concat(fitPoints(s1, atoms1));
    // one rotation center for both (the views are linked): the selected frame's
    const pivot = pivotFor(periodic(s1) ? s1 : { ...s1, cell: null }, atoms1, 1, pts);
    fitView(vb, s0, atoms0, 1, pts, { pivot }); fitView(va, s1, atoms1, 1, pts, { pivot });
    if (!vb.linked) { vb.linkViewer(va); va.linkViewer(vb); vb.linked = true; }
    vb.render(); va.render();
    return;
  }
  if (ghost) {
    const gm = addModel(vb, s0w, atoms0, { bonds: false, style: "ballstick", opacity: Number(cssVar("--cmp-ghost-opacity")) || 0.35 });
    // dark theme: the ghost's colors are lifted toward white, so a dark element (O, N) still reads on the dark viewport
    const lift = Number(cssVar("--cmp-ghost-lift")) || 0;
    if (lift > 0) {
      const st = styleFor("ballstick", false, Number(cssVar("--cmp-ghost-opacity")) || 0.35);
      for (const sub of Object.values(st)) { delete sub.colorscheme; sub.colorfunc = (a) => mixHex(elementColor(s0w.symbols[a.serial - 1]), "#ffffff", lift); }
      gm.setStyle({}, st);
    }
    drawCell(vb, s0, muted, 0.02, 1, true);
  }
  if (!state.soloGhost) addModel(vb, s1u, atoms1, base);   // soloGhost: a test hook (the ghost contrast check)
  drawCell(vb, s1, accent, 0.03);
  if (showArrows && !state.soloGhost) arrows(vb, atoms1, c, scale);
  if (keep && !state.refit) vb.setView(keep); else fitView(vb, s1, atoms0.concat(atoms1));
  state.refit = false;
  vb.render();
}

/* one color per atom through a colorfunc: element shading kept by the lighting, one model per look */
function tint(m, s, fn, o = { style: "ballstick", bonds: true }) {
  const st = styleFor(o.style, o.bonds, 1);
  for (const sub of Object.values(st)) { delete sub.colorscheme; sub.colorfunc = fn; }
  m.setStyle({}, st);
  return m;
}
function hexRgb(h) {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(h).trim());
  if (!m) return [136, 136, 136];
  const n = parseInt(m[1], 16);
  return [n >> 16 & 255, n >> 8 & 255, n & 255];
}
function mixHex(a, b, t) {
  const x = hexRgb(a), y = hexRgb(b);
  return "#" + x.map((v, k) => Math.round(v + (y[k] - v) * t).toString(16).padStart(2, "0")).join("");
}
function seqStops() { return Array.from({ length: 9 }, (_, k) => cssVar(`--seq-${k}`)); }
function seqColor(f) {
  const st = seqStops(), x = Math.max(0, Math.min(1, f)) * 8, k = Math.min(7, Math.floor(x));
  return mixHex(st[k], st[k + 1], x - k);
}
function renderLegend(c) {
  const el = $("#o-legend");
  const look = state.layout;
  let html;
  const D = dispSet(c), L = D.list.map((d) => d.length);
  const lo = L.length ? Math.min(...L) : 0, hi = L.length ? Math.max(...L) : 0;
  const scale = Number($("#o-scale").value) || 1, arrowsOn = $("#o-arrows").checked;
  const what = D.mode === "cart" ? "displacement, cell change included" : D.mode === "molecule" ? "displacement, rigid motion removed" : "relaxation relative to the cell";
  // the arrows' scale in words: how far an atom moved per Å drawn
  const arrowTxt = `<span class="lg-arrows" title="Arrow × ${scale}"><i class="sw arrow"></i>${what}: arrows ×${scale}, 1 Å drawn = ${fmtLen(1 / scale, "Å")} moved</span>`;
  const still = hi < NEGLIGIBLE_DR;
  const cellTxt = c.cell_changed ? cellChangeText(c) : "";
  if (still) {
    // nothing to draw: say why instead of a blank view and a 0–0 scale
    const why = D.mode === "internal" ? "Atoms did not move relative to the cell (their positions are fixed by symmetry, or only the cell was optimized)" : "Atoms did not move";
    const pre = look === "twotone" ? `<span class="lg-twotone"><i class="sw" style="background:var(--cmp-a)"></i>initial</span><span><i class="sw" style="background:var(--cmp-b)"></i>${state.frame === state.traj.frames.length - 1 ? "final" : "selected frame"}</span>` : "";
    html = `${pre}<span class="lg-note" role="status">${why}${hi > 0 ? ` (max |Δr| ${fmtLen(hi)})` : ""}${cellTxt ? `; the cell changed: ${esc(cellTxt)}` : ""}.${D.mode === "internal" && c.cell_changed && c.max_displacement_cart >= NEGLIGIBLE_DR ? " Δr: With cell change shows how far they moved in space." : ""}</span>`;
  } else if (look === "twotone") html = `<span class="lg-twotone"><i class="sw" style="background:var(--cmp-a)"></i>initial</span><span><i class="sw" style="background:var(--cmp-b)"></i>${state.frame === state.traj.frames.length - 1 ? "final" : "selected frame"}</span>`;
  else if (look === "dmap") {
    const u = lenUnit(hi), a = fmtLen(lo, u), b = fmtLen(hi, u);
    html = a === b
      ? `<span class="lg-seq"><span>|Δr|</span><span class="num">${b}</span><span>(every atom)</span></span>`
      : `<span class="lg-seq"><span>|Δr|</span><span class="num">${a}</span><i class="seq-bar" style="background:linear-gradient(90deg, ${seqStops().join(", ")})"></i><span class="num">${b}</span></span>`;
    if (arrowsOn) html += arrowTxt;
  } else if (look === "side") html = `<span><i class="sw" style="background:var(--muted)"></i>initial (left)</span><span><i class="sw solid"></i>selected frame (right)</span>${arrowsOn ? arrowTxt : ""}`;
  else html = `<span><i class="sw ghost"></i>initial</span><span><i class="sw solid"></i>selected frame</span>${arrowsOn ? arrowTxt : ""}`;
  if (!still && cellTxt) html += `<span class="lg-cell" title="Cell: ${esc(cellTxt)}">Cell: ${esc(cellTxt)}</span>`;
  el.innerHTML = html;
  el.dataset.look = look;
}
/** The comparison as SVG panes (Export Image on a Compare document, js/lazy/export_svg.js):
 * per pane its viewer, and the atoms with the colors and opacity of the current look. */
function compareSvgItems() {
  const t = state.traj, c = state.cmp;
  if (!t || !c) return null;
  const s0 = t.frames[0], s1 = t.frames[state.frame];
  const vdw = (sym) => { const r = (window.$3Dmol && window.$3Dmol.GLModel && window.$3Dmol.GLModel.vdwRadii) || {}; return r[sym] || 1.5; };
  const k = sphereScale(true), look = state.layout;
  const frameOf = (v) => {
    const q = v.getView().slice(4, 8), qi = [-q[0], -q[1], -q[2], q[3]];
    // the camera axes in model space: p rotated by the inverse view quaternion (p + 2w u×p + 2 u×(u×p))
    const rot = (p) => {
      const u = qi.slice(0, 3), w = qi[3];
      const cr = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
      const t = cr(u, p), t2 = cr(u, t);
      return [0, 1, 2].map((i) => p[i] + 2 * w * t[i] + 2 * t2[i]);
    };
    return { right: rot([1, 0, 0]), toward: rot([0, 0, 1]) };
  };
  const items = (v, cart, sym, fillOf, opacity) => {
    const f = frameOf(v);
    return cart.map((p, i) => ({ x: p[0], y: p[1], z: p[2], r: k * vdw(sym[i]), right: f.right,
      depth: p[0] * f.toward[0] + p[1] * f.toward[1] + p[2] * f.toward[2], fill: fillOf(i), opacity, title: `${sym[i]}${i + 1}` }));
  };
  const vb = state.viewers["viewer-before"], va = state.viewers["viewer-after"];
  const el = (i, sy) => elementColor(sy[i]);
  const panes = [];
  if (look === "side" && va) {
    panes.push({ v: vb, id: "initial", items: items(vb, c.before_cart, s0.symbols, (i) => el(i, s0.symbols), 1) });
    panes.push({ v: va, id: "final", items: items(va, c.after_unwrapped_cart, s1.symbols, (i) => el(i, s1.symbols), 1) });
  } else {
    let list = [];
    if (look === "twotone") {
      list = items(vb, c.before_cart, s0.symbols, (i) => mixHex(el(i, s0.symbols), cssVar("--cmp-a"), 0.7), 1)
        .concat(items(vb, c.after_unwrapped_cart, s1.symbols, (i) => mixHex(el(i, s1.symbols), cssVar("--cmp-b"), 0.7), 1));
    } else if (look === "dmap") {
      const L = dispSet(c).list.map((d) => d.length), lo = Math.min(...L), hi = Math.max(...L);
      list = items(vb, c.after_unwrapped_cart, s1.symbols, (i) => seqColor(hi > lo ? (L[i] - lo) / (hi - lo) : 0), 1);
    } else {
      if ($("#o-ghost").checked) list = items(vb, c.before_cart, s0.symbols, (i) => el(i, s0.symbols), Number(cssVar("--cmp-ghost-opacity")) || 0.35);
      list = list.concat(items(vb, c.after_unwrapped_cart, s1.symbols, (i) => el(i, s1.symbols), 1));
    }
    panes.push({ v: vb, id: look, items: list });
  }
  return { panes, background: viewerBg(), title: $("#o-title").textContent };
}
const LOOKS = ["overlay", "twotone", "side", "dmap"];
function setLook(look) {
  if (!LOOKS.includes(look)) look = "overlay";
  state.layout = look;
  $$(".seg-btn[data-layout]").forEach((x) => x.classList.toggle("active", x.dataset.layout === look));
  $("#viewer-after").classList.toggle("hidden", look !== "side");
  $("#o-ghost").closest("label").classList.toggle("hidden", look !== "overlay");
  if (look === "side") viewer("viewer-after");
  requestAnimationFrame(() => {
    Object.values(state.viewers).forEach((v) => v && v.resize());
    state.viewers["viewer-before"] && state.viewers["viewer-before"].removeAllModels();
    drawOpt();
  });
}
function cycleLook() { setLook(LOOKS[(LOOKS.indexOf(state.layout) + 1) % LOOKS.length]); }

function arrows(v, atomsAt, c, scale) {
  v.arrowCount = 0;                 // how many were drawn (the tests read it; the cell's edges are shapes too)
  dispSet(c).list.forEach((d, i) => {
    // drawn length, not the real one: a 1e-4 Å relaxation at ×2000 is a 0.2 Å arrow
    if (!(d.length > 1e-9) || d.length * scale < 0.01) return;
    v.arrowCount++;
    const start = atomsAt[i].cart, end = add(start, mul(d.vector, scale));
    v.addArrow({ start: xyz(start), end: xyz(end), radius: 0.06, radiusRatio: 2.2, mid: 0.75,
      color: cssVar("--sel") });
  });
}

/* The energy chart is a small hand-drawn SVG, so a trajectory never needs
 * plotly (4.6 MB): plotly loads only when a plot opens (Open as Plot).
 * PLAN_analysis C3.8: x is CRYSTAL's optimization POINT (frame k = POINT k+1;
 * the final frame is a ring at the last point), energies on the optimizer's
 * surface (C0), relative to the first or the final point, in a chosen unit,
 * or the log of |ΔE|. Segments: "continuous" (the default) is one curve and
 * one reference across the FINALRUN restarts (dotted lines mark them);
 * "separate" re-references each FINALRUN segment (D6). The twin of
 * mace.utils.out_convergence.energy_along_run (`mace analyze --convergence
 * FILE --energy`), whose TSV is the Copy TSV here. */
const E_UNITS = { eV: 1, meV: 1000, Ha: 1 / 27.211386245988, "kJ/mol": 96.48533212331, "kcal/mol": 23.060547830619 };
function cardKey(t) { return `mace.gui.runcard.${(state.info && state.info.dir_hash) || ""}.${t.path || ""}`; }
function cardOpts(t) {
  let o = null;
  try { o = JSON.parse(store(cardKey(t)) || "null"); } catch (e) { o = null; }
  const su = setting("energyUnit", "eV");
  const sg = setting("plots.segments", "continuous");
  return { log: false, rel: "first", unit: su && su !== "eV" ? su : "meV", segments: sg === "separate" ? "separate" : "continuous", ...(o || {}) };
}
function setCardOpts(t, patch) { store(cardKey(t), JSON.stringify({ ...cardOpts(t), ...patch })); drawEnergy(); }
/** {x (POINT), y, frame, ring, seg, below} per plotted frame, in the card's unit. */
function energySeries(t) {
  const es = optEnergies(t), conv = state.trajConv && state.trajConv.path === t.path ? state.trajConv : null;
  const o = cardOpts(t), f = E_UNITS[o.unit] || 1000;
  const pointOf = {}, segOf = {};
  if (conv) for (const p of conv.points) if (p.frame !== null && p.frame !== undefined) { pointOf[p.frame] = p.point; segOf[p.frame] = p.segment; }
  const nf = es.length, last = (t.labels || [])[nf - 1] === "final";
  const lastPoint = conv && conv.points.length ? conv.points[conv.points.length - 1] : null;
  const pts = [];
  for (let k = 0; k < nf; k++) {
    if (es[k] === null || es[k] === undefined) continue;
    const ring = last && k === nf - 1;
    const x = ring ? (lastPoint ? lastPoint.point : k) : (pointOf[k] ?? k + 1);
    const seg = ring ? (lastPoint ? lastPoint.segment : 0) : (segOf[k] ?? 0);
    pts.push({ k, x, e: es[k], seg, ring });
  }
  // one reference for the whole run (continuous) or one per FINALRUN segment (separate)
  const grp = (p) => (o.segments === "separate" ? p.seg : 0);
  const segRef = {};
  for (const p of pts) {
    if (o.rel === "first") { if (!(grp(p) in segRef)) segRef[grp(p)] = p.e; }
    else segRef[grp(p)] = p.e;                            // the last (the final frame for the last segment)
  }
  for (const p of pts) {
    const ref = segRef[grp(p)];
    p.dE = (p.e - ref) * HARTREE_MEV / 1000 * f;                // Ha -> eV -> unit
    p.isRef = Math.abs(p.e - ref) <= 1e-10;                     // the printed precision
    p.below = o.rel === "final" && p.e < ref - 1e-10;
    p.y = o.log ? (p.isRef || p.dE === 0 ? null : Math.abs(p.dE)) : p.dE;   // the reference has no log value
  }
  const nSeg = conv && conv.segments ? conv.segments.length : 1;
  return { pts, o, conv, nSeg, grp };
}
function drawEnergy() {
  const t = state.traj;
  const el = $("#o-energy");
  if (!t) return;
  drawScf();
  const { pts: all, o, conv, nSeg, grp } = energySeries(t);
  const pts = all.filter((p) => p.y !== null);
  $("#o-energy-card").classList.toggle("no-data", all.length < 2);
  $("#o-e-log").setAttribute("aria-pressed", String(!!o.log));
  $("#o-e-rel").value = o.rel; $("#o-e-unit").value = o.unit; $("#o-e-seg").value = o.segments;
  $("#o-e-seg").closest("label").hidden = nSeg < 2;               // only a run with FINALRUN restarts has segments
  const of = nSeg > 1 ? (o.segments === "separate" ? " of each FINALRUN segment" : " of the run") : "";
  $("#o-energy-label").innerHTML = `Energy along the run <span class="muted">· ${esc(o.unit)}, ${o.log ? "log |ΔE|, " : ""}relative to the ${o.rel === "final" ? "final" : "first"} point${of}</span>`;
  if (all.length < 2 || pts.length < 1) { el.innerHTML = ""; return; }
  const W = Math.max(240, el.clientWidth || 360), H = 200, L = 60, R = 12, T = 10, B = 34;
  const ys = pts.map((p) => (o.log ? Math.log10(p.y) : p.y));
  let lo = Math.min(...ys), hi = Math.max(...ys);
  if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const x0 = Math.min(...all.map((p) => p.x)), x1 = Math.max(...all.map((p) => p.x));
  const span = Math.max(1, x1 - x0);
  const X = (x) => L + (W - L - R) * ((x - x0) / span), Y = (y) => T + (H - T - B) * (1 - (y - lo) / (hi - lo));
  const ticks = niceTicks(lo, hi, 5);
  const xstep = Math.max(1, Math.ceil(span / 10));
  let svg = `<svg class="echart" viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Energy along the run" data-plotkit="1">`;
  ticks.forEach((v) => { svg += `<line class="grid" x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/>` +
    `<text class="tick" x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${o.log ? `1e${Math.round(v)}` : +v.toFixed(Math.abs(hi - lo) < 0.1 ? 4 : 2)}</text>`; });
  for (let x = x0; x <= x1; x += xstep) svg += `<text class="tick" x="${X(x)}" y="${H - B + 16}" text-anchor="middle">${x}</text>`;
  // FINALRUN restarts: a dotted line between the segments (the curve runs on across it unless Separate)
  if (conv) for (const sgm of conv.segments.slice(1)) {
    const xm = sgm.first_point - 0.5;
    const tip = `FINALRUN restart before POINT ${sgm.first_point}${o.segments === "separate" ? " (re-referenced)" : ""}`;
    svg += `<line class="seg-line" x1="${X(xm)}" x2="${X(xm)}" y1="${T}" y2="${H - B}" data-first-point="${sgm.first_point}"><title>${tip}</title></line>`;
  }
  svg += `<text class="axis-title" x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">Optimization point</text>`;
  svg += `<text class="axis-title" transform="translate(12 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">${o.log ? "|ΔE|" : "ΔE"} (${esc(o.unit)})</text>`;
  const bySeg = {};
  pts.filter((p) => !p.ring).forEach((p) => { (bySeg[grp(p)] = bySeg[grp(p)] || []).push(p); });
  Object.values(bySeg).forEach((sp) => {
    svg += `<polyline class="line" points="${sp.map((p) => `${X(p.x)},${Y(o.log ? Math.log10(p.y) : p.y)}`).join(" ")}"/>`;
  });
  const fmtE = (v) => (v === 0 ? "0" : Math.abs(v) >= 100 ? v.toFixed(1) : Math.abs(v) >= 0.01 ? v.toFixed(2) : v.toExponential(2));
  pts.forEach((p) => {
    const cy = Y(o.log ? Math.log10(p.y) : p.y), cx = X(p.x);
    const label = `POINT ${p.x}${p.ring ? " (final geometry)" : ""}, frame ${p.k}: ${fmtE(p.dE)} ${o.unit}${p.below ? " (below its reference)" : ""}`;
    if (p.ring) svg += `<circle class="ring" cx="${cx}" cy="${cy}" r="5" data-frame="${p.k}"><title>${esc(label)}</title></circle>`;
    else if (p.below) svg += `<path class="pt below" d="M${cx - 4},${cy - 3} L${cx + 4},${cy - 3} L${cx},${cy + 4} Z" data-frame="${p.k}"><title>${esc(label)}</title></path>`;
    else svg += `<circle class="pt" cx="${cx}" cy="${cy}" r="3" data-frame="${p.k}"><title>${esc(label)}</title></circle>`;
  });
  const sel = pts.find((p) => p.k === state.frame);
  if (sel) svg += `<circle class="sel" cx="${X(sel.x)}" cy="${Y(o.log ? Math.log10(sel.y) : sel.y)}" r="7"/>`;
  el.innerHTML = svg + "</svg>";
}
/* The SCF card: one SCF's cycles (out_convergence.scf_timeline, with the convergence), |ΔE| per
 * cycle on a log axis against TOLDEE (or E minus the last cycle's), the notable cycles marked
 * (mixing, DIIS, the spin lock's release, conducting-state warnings); the latest SCF by default,
 * which follows a running job. `mace analyze --convergence FILE --scf` prints the same (Copy TSV
 * writes its columns). One component, two places: the run view's Timeline (an optimization) and
 * the inspector's SCF section (a single point, a FREQ, any output that opens without the run view). */
const SCF_CARD_HTML = `
  <div class="card-label" data-s="label">SCF convergence <span class="muted"></span></div>
  <div class="o-energy-row" role="toolbar" aria-label="SCF chart options">
    <label class="o-e-sel" title="Which SCF: the latest (it follows a running job) or one the output prints per cycle">Step <select data-s="step" aria-label="SCF step"></select></label>
    <button type="button" class="tb-btn-text" data-s="log" aria-pressed="true" title="Log of |ΔE| per cycle; off: E minus the last cycle's">Log |ΔE|</button>
    <label class="o-e-sel">Units <select data-s="unit" aria-label="SCF energy unit"><option>Ha</option><option>eV</option><option>meV</option><option value="kJ/mol">kJ/mol</option><option value="kcal/mol">kcal/mol</option></select></label>
    <button type="button" class="tb-btn-text" data-s="tsv" title="Copy the cycles as tab-separated text (as mace analyze --convergence FILE --scf prints them)">Copy TSV</button>
    <button type="button" class="tb-btn-text" data-s="plot" title="Run Analysis › SCF">Open as Plot</button>
  </div>
  <div data-s="chart" class="chart o-scf-chart"></div>`;
function scfKey(path) { return `mace.gui.scfcard.${(state.info && state.info.dir_hash) || ""}.${path || ""}`; }
function scfOptsOf(path) {
  let o = null;
  try { o = JSON.parse(store(scfKey(path)) || "null"); } catch (e) { o = null; }
  return { log: true, unit: "Ha", step: "latest", ...(o || {}) };
}
function scfPick(tl, step) {
  const bl = (tl && tl.blocks) || [];
  if (!bl.length) return null;
  if (!step || step === "latest") return bl[bl.length - 1];
  const k = Number(step);
  return k >= 1 && k <= bl.length ? bl[k - 1] : bl[bl.length - 1];
}
function scfSeriesOf(tl, o) {
  const blk = scfPick(tl, o.step), f = (E_UNITS[o.unit] || 1) * HARTREE_MEV / 1000;   // Ha -> unit
  if (!blk) return { blk, pts: [], f };
  const last = blk.cycles.length ? blk.cycles[blk.cycles.length - 1].etot : 0;
  const pts = blk.cycles.map((c) => ({ cyc: c.cyc, e: c.etot, de: c.detot, spin: c.spin,
    y: o.log ? (c.detot ? Math.abs(c.detot) * f : null) : (c.etot === null ? null : (c.etot - last) * f), up: c.detot > 0 && c.cyc > 0 }));
  return { blk, pts, f };
}
/** The chosen SCF's cycles as tab-separated text: the columns of out_convergence.scf_tsv. */
function scfTsvOf(tl, o) {
  const { blk, f } = scfSeriesOf(tl, o);
  if (!blk) return "";
  const ev = {};
  for (const e of blk.events) (ev[e.cyc] = ev[e.cyc] || []).push(e.text);
  return [`cycle\tE_Ha\tdE_${o.unit}\tspin\tevents`, ...blk.cycles.map((c) =>
    `${c.cyc}\t${c.etot ?? ""}\t${c.detot === null ? "" : c.detot * f}\t${c.spin ?? ""}\t${(ev[c.cyc] || []).join("; ")}`)].join("\n") + "\n";
}
/** Draw one SCF card: `root` holds SCF_CARD_HTML's parts, `tl` the timeline (null: not read yet). */
function renderScfCard(root, path, tl) {
  const q1 = (k) => root.querySelector(`[data-s="${k}"]`);
  const el = q1("chart"), o = scfOptsOf(path);
  root.classList.toggle("hidden", !tl);                   // until the convergence is read
  if (!tl || !el) return;
  const { blk, pts: all, f } = scfSeriesOf(tl, o);
  const sel = q1("step"), bl = tl.blocks || [];
  sel.innerHTML = [["latest", "Latest"], ...bl.map((b, i) => [String(i + 1), `${b.label} · ${b.cycles.length} cycles`])]
    .map(([v, l]) => `<option value="${v}"${v === String(o.step || "latest") ? " selected" : ""}>${esc(l)}</option>`).join("");
  sel.closest("label").hidden = bl.length < 2;
  q1("log").setAttribute("aria-pressed", String(!!o.log)); q1("unit").value = o.unit;
  const thr = tl.toldee;
  q1("label").innerHTML = `SCF convergence <span class="muted">· ${blk ? `${esc(blk.label)}, ${blk.cycles.length} cycles${blk.running ? " so far" : blk.ended === "too_many_cycles" ? ", not converged" : ""}, ` : ""}` +
    `${o.log ? "log |ΔE|" : "E − E(last cycle)"} in ${esc(o.unit)}${thr ? ` · TOLDEE ${thr.value.toExponential(0)} Ha` : ""}</span>`;
  const pts = all.filter((p) => p.y !== null && (!o.log || p.y > 0));
  if (!blk || pts.length < 1) {
    el.innerHTML = `<p class="o-scf-note">${bl.length ? "No per-cycle SCF lines in this step." : "This output prints no per-cycle SCF lines."}` +
      `${tl.moved_to_scfout_log && !tl.scfout ? " CRYSTAL prints the cycles of the first point only; the rest went to SCFOUT.LOG, which a MACE job copies back as <job>.SCFOUT.LOG (cycles per point: Run Analysis › SCF)." : ""}</p>`;
    return;
  }
  const W = Math.max(240, el.clientWidth || 360), H = 200, L = 60, R = 12, T = 22, B = 34;
  const tv = thr && o.log ? thr.value * f : null;
  const ys = pts.map((p) => (o.log ? Math.log10(p.y) : p.y)).concat(tv ? [Math.log10(tv)] : []);
  let lo = Math.min(...ys), hi = Math.max(...ys);
  if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const x0 = Math.min(...all.map((p) => p.cyc)), x1 = Math.max(...all.map((p) => p.cyc), x0 + 1), span = x1 - x0;
  const X = (x) => L + (W - L - R) * ((x - x0) / span), Y = (y) => T + (H - T - B) * (1 - (y - lo) / (hi - lo));
  const yv = (p) => Y(o.log ? Math.log10(p.y) : p.y);
  let svg = `<svg class="echart" viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="SCF convergence" data-plotkit="1">`;
  niceTicks(lo, hi, 5).forEach((v) => { svg += `<line class="grid" x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/>` +
    `<text class="tick" x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${o.log ? `1e${Math.round(v)}` : +v.toPrecision(3)}</text>`; });
  const xstep = Math.max(1, Math.ceil(span / 10));
  for (let x = x0; x <= x1; x += xstep) svg += `<text class="tick" x="${X(x)}" y="${H - B + 16}" text-anchor="middle">${x}</text>`;
  if (tv) svg += `<line class="thr-line" x1="${L}" x2="${W - R}" y1="${Y(Math.log10(tv))}" y2="${Y(Math.log10(tv))}"><title>TOLDEE ${thr.value} Ha (${esc(thr.source || "")})</title></line>` +
    `<text class="thr-label" x="${W - R - 2}" y="${Y(Math.log10(tv)) - 3}" text-anchor="end">TOLDEE</text>`;
  // the notable cycles: a dotted line and a short label (staggered when they crowd)
  blk.events.forEach((e, i) => {
    const xe = X(e.cyc);
    svg += `<line class="ev-line${e.kind === "too_many_cycles" ? " bad" : ""}" x1="${xe}" x2="${xe}" y1="${T}" y2="${H - B}" data-kind="${esc(e.kind)}"><title>cycle ${e.cyc}: ${esc(e.text)}</title></line>` +
      `<text class="ev-label" x="${xe + 3}" y="${T - 10 + (i % 2) * 9}">${esc(e.text)}</text>`;
  });
  svg += `<text class="axis-title" x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">SCF cycle</text>`;
  svg += `<text class="axis-title" transform="translate(12 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">${o.log ? "|ΔE|" : "E − E(last)"} (${esc(o.unit)})</text>`;
  svg += `<polyline class="line" points="${pts.map((p) => `${X(p.cyc)},${yv(p)}`).join(" ")}"/>`;
  const fmtE = (v) => (Math.abs(v) >= 0.01 && Math.abs(v) < 1e4 ? v.toPrecision(4) : v.toExponential(2));
  pts.forEach((p) => {
    const label = `cycle ${p.cyc}: ${o.log ? "|ΔE|" : "E − E(last)"} ${fmtE(p.y)} ${o.unit}${p.up ? " (the energy went up)" : ""}${p.spin !== null && p.spin !== undefined ? ` · spin ${p.spin}` : ""}`;
    svg += `<circle class="pt${o.log && p.up ? " up" : ""}" cx="${X(p.cyc)}" cy="${yv(p)}" r="3" data-cyc="${p.cyc}"><title>${esc(label)}</title></circle>`;
  });
  el.innerHTML = svg + "</svg>";
}
/** Wire a card's controls once: `path()` and `tl()` give the current output and timeline. */
function bindScfCard(root, path, tl) {
  const redraw = () => renderScfCard(root, path(), tl());
  const set = (patch) => { const p = path(); if (!p) return; store(scfKey(p), JSON.stringify({ ...scfOptsOf(p), ...patch })); redraw(); };
  root.addEventListener("change", (e) => {
    const k = e.target.dataset && e.target.dataset.s;
    if (k === "step") set({ step: e.target.value });
    else if (k === "unit") set({ unit: e.target.value });
  });
  root.addEventListener("click", async (e) => {
    const b = e.target.closest("[data-s]");
    if (!b || !path()) return;
    const k = b.dataset.s;
    if (k === "log") set({ log: !scfOptsOf(path()).log });
    else if (k === "tsv") {
      const text = scfTsvOf(tl(), scfOptsOf(path())), n = text.trim().split("\n").length - 1;
      toast(text && (await copyPlain(text)) ? `Copied ${n} rows` : "Copy failed", "info");
    } else if (k === "plot" && window.__maceOpenRunPlot) {
      const o = scfOptsOf(path());
      window.__maceOpenRunPlot(path(), "scf", { _tab: "scf", block: o.step || "latest", ylog: !!o.log, unit: o.unit });
    }
  });
  return redraw;
}
/* the run view's card (an optimization): its timeline comes with the run's convergence */
const runScfTl = () => (state.traj && state.trajConv && state.trajConv.path === state.traj.path ? state.trajConv.scf_timeline : null);
const drawRunScf = bindScfCard($("#o-scf-card"), () => state.traj && state.traj.path, runScfTl);
function drawScf() { if (state.traj) drawRunScf(); }
function scfTSV() { return state.traj ? scfTsvOf(runScfTl(), scfOptsOf(state.traj.path)) : ""; }
/** The inspector's SCF section for an output that opens without the run view (a single point,
 * a FREQ): the card, its own read of the convergence, read again every 15 s while the output grows. */
function scfSection(path) {
  const root = document.createElement("div");
  root.className = "scf-card hidden";
  root.innerHTML = SCF_CARD_HTML;
  let tl = null, timer = 0;
  const redraw = bindScfCard(root, () => path, () => tl);
  const load = async () => {
    clearTimeout(timer);
    if (!root.isConnected && tl) return;                   // the section left the inspector
    try {
      const r = await api("out/convergence?" + q({ path }));
      tl = r.scf_timeline || { blocks: [] };
      root.dataset.blocks = String(tl.blocks.length);
      redraw();
      if (r.live) timer = setTimeout(load, 15000 * (window.__maceLiveScale || 1));
    } catch (e) { root.classList.remove("hidden"); root.querySelector('[data-s="chart"]').innerHTML = `<p class="o-scf-note">${esc(e.message)}</p>`; }
  };
  load();
  // the width is known once the section is in the inspector
  requestAnimationFrame(() => { if (tl) redraw(); });
  return root;
}
function energyTSV() {
  const t = state.traj;
  if (!t) return "";
  const { pts, o } = energySeries(t);
  // the columns of `mace analyze --convergence FILE --energy`; the segment is 1-based
  return [`point\tsegment\tframe\tE_Ha\tdE_${o.unit}`, ...pts.map((p) => `${p.x}\t${p.seg + 1}\t${p.k}\t${p.e}\t${p.dE}`)].join("\n") + "\n";
}
async function copyPlain(text) {
  if (window.isSecureContext && navigator.clipboard) { try { await navigator.clipboard.writeText(text); return true; } catch (e) { /* fall back */ } }
  const ta = document.createElement("textarea"); ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
  document.body.appendChild(ta); ta.select();
  let ok = false; try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
  ta.remove(); return ok;
}
$("#o-e-log").addEventListener("click", () => { if (state.traj) setCardOpts(state.traj, { log: !cardOpts(state.traj).log }); });
$("#o-e-rel").addEventListener("change", (e) => { if (state.traj) setCardOpts(state.traj, { rel: e.target.value }); });
$("#o-e-unit").addEventListener("change", (e) => { if (state.traj) setCardOpts(state.traj, { unit: e.target.value }); });
$("#o-e-seg").addEventListener("change", (e) => { if (state.traj) setCardOpts(state.traj, { segments: e.target.value }); });
$("#o-e-tsv").addEventListener("click", async () => {
  const text = energyTSV(), n = text.trim().split("\n").length - 1;
  toast((await copyPlain(text)) ? `Copied ${n} rows` : "Copy failed", "info");
});
$("#o-e-plot").addEventListener("click", () => {
  if (!window.__maceOpenRunPlot || !state.traj) return;
  // the plot opens on the data the card shows: its reference, segments, unit and log view, per cell
  const o = cardOpts(state.traj);
  window.__maceOpenRunPlot(state.traj.path, "energy", { relTo: o.rel, segments: o.segments, unit: o.unit, ylog: !!o.log, per: "cell" });
});
$("#o-energy").addEventListener("click", (e) => {
  const c = e.target.closest("[data-frame]");
  if (!c || !state.traj) return;
  const fr = $("#o-frame");
  fr.value = c.dataset.frame;
  fr.dispatchEvent(new Event("input"));
});
async function loadTrajConv(t) {
  try {
    const r = await api("out/convergence?" + q({ path: t.path }));
    if (state.traj === t || (state.traj && state.traj.path === t.path)) { state.trajConv = { ...r, path: t.path }; drawEnergy(); }
  } catch (e) { /* the card works without it (x = frame + 1) */ }
}
function niceTicks(lo, hi, count) {
  const raw = (hi - lo) / count, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((x) => x >= raw) || raw;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

$("#o-scale").addEventListener("input", (e) => { $("#o-scale-out").textContent = e.target.value; drawOpt(); });
$("#o-arrows").addEventListener("change", drawOpt);
$("#o-dmode").addEventListener("change", (e) => {
  store("mace.gui.cmp.dispMode", e.target.value);
  autoScale(); renderComparison(); drawOpt();
});
$("#o-ghost").addEventListener("change", drawOpt);
$("#o-frame").addEventListener("input", async (e) => {
  const want = Number(e.target.value);
  state.frame = want;
  const t = state.traj;
  $("#o-frame-out").textContent = `${state.frame} · ${t.labels[state.frame]}`;
  // one event keeps the slider, the Timeline, the energy card, Run Analysis and the panels in step
  document.dispatchEvent(new CustomEvent("mace-frame-changed", { detail: { path: t.path, frame: want } }));
  try {
    const [cmp] = await Promise.all([api("trajectory/compare?" + q({ path: t.path, i: 0, j: want })), ensureStep(t, want)]);
    if (state.frame !== want) return;          // a newer scrub won
    state.cmp = cmp;
  } catch (err) { toast(err.message, true); return; }
  renderComparison(); drawOpt(); drawEnergy();
});
$$(".seg-btn[data-layout]").forEach((b) => b.addEventListener("click", () => setLook(b.dataset.layout)));
$("#cmp-go").addEventListener("click", async () => {
  const a = $("#cmp-a").value, b = $("#cmp-b").value;
  if (!a || !b) return;
  try {
    $("#cmp-refused").classList.add("hidden");
    const r = await api("compare?" + q({ a, b }));
    const t = { frames: [r.a, r.b], energies_au: [r.a.energy_au ?? null, r.b.energy_au ?? null],
      labels: ["A", "B"], comparison: r.comparison };
    setTrajectory(t, `${a}  →  ${b}`, matchNote(r.comparison));
  } catch (e) { refused(e.message); }
});

function matchNote(c) {
  return c && c.match === "position" ? `two structures, atoms matched by position (max mismatch ${fmt(c.max_mismatch, 3)} Å)` : "two structures, same atom order";
}
/** A comparison the server refused (counts or species differ): the reason, no tables. */
function refused(msg) {
  const el = $("#cmp-refused");
  el.textContent = msg;
  el.classList.remove("hidden");
  toast(msg, true);
}
/* ---------------------------------------------------------------- results */
/* plotly (about 1.4 MB gzipped) is fetched the first time a plot opens */
let plotlyLoading = null;
function loadPlotly() {
  if (window.Plotly) return Promise.resolve();
  if (!plotlyLoading) plotlyLoading = new Promise((resolve) => {
    const el = document.createElement("script");
    el.src = assetUrl("vendor/plotly.min.js");
    el.onload = () => { unzoomPlotly(); resolve(); };
    el.onerror = () => { toast("plotly is not installed here: interactive plots are unavailable", true); resolve(); };
    document.head.appendChild(el);
  });
  return plotlyLoading;
}
/* plotly measures the pointer as 3Dmol does (clientX minus the plot's screen
 * corner) and maps it into the plot's own px with the inverse of the CSS
 * transforms above the plot, which knows nothing of the UI scale's zoom:
 * hovers, zoom boxes and drags landed off the pointer. Each plot's inverse
 * also undoes the zoom; plotly itself stays as installed. */
function unzoomPlotly() {
  const P = window.Plotly;
  if (!P || !P.newPlot || P.newPlot.maceUnzoomed) return;
  const fixLayout = (gd) => {
    const fl = gd && gd._fullLayout, calc = fl && fl._calcInverseTransform;
    if (typeof calc !== "function" || calc.maceUnzoomed) return;
    let base = null, scaled = null;
    const f = function (g) {
      calc.call(this, g);
      const L = g._fullLayout;
      // plotly recomputes only when the plot's screen box changed; else ours is still in place
      if (L._invTransform !== scaled) base = L._invTransform;
      if (!base) return;
      const z = window.maceZoom.of(g);
      // first the zoom off the screen px, then plotly's own inverse: its x and y columns over z
      scaled = base.map((row) => row.map((v, j) => (j < 2 ? v / z : v)));
      L._invTransform = scaled;
      L._invScaleX = Math.hypot(scaled[0][0], scaled[0][1], scaled[0][2]);
      L._invScaleY = Math.hypot(scaled[1][0], scaled[1][1], scaled[1][2]);
    };
    f.maceUnzoomed = true;
    fl._calcInverseTransform = f;
  };
  const fix = (gd) => {
    fixLayout(gd);
    if (gd && gd.on && !gd.__maceUnzoom) { gd.__maceUnzoom = true; gd.on("plotly_afterplot", () => fixLayout(gd)); }
    return gd;
  };
  for (const name of ["newPlot", "react"]) {
    const orig = P[name];
    P[name] = function (gd, ...rest) {
      const out = orig.call(this, gd, ...rest);
      const el = typeof gd === "string" ? document.getElementById(gd) : gd;
      return out && out.then ? out.then((g) => fix(g || el)) : fix(el);
    };
    P[name].maceUnzoomed = true;
  }
}
/* ---------------------------------------------------------------- selectors */
/* The compare pickers and the database select (the deck tab is Calculation Setup now, js/setup). */
function fillSelectors() {
  const all = state.files.filter((f) => ["cif", "d12", "out"].includes(f.kind));
  const opts = all.map((f) => `<option value="${esc(f.path)}">${esc(f.path)}</option>`).join("");
  $("#cmp-a").innerHTML = opts; $("#cmp-b").innerHTML = opts;
  if (all.length > 1) $("#cmp-b").selectedIndex = 1;
}
function highlightDeck(text) {
  return text.split("\n").map((line) => {
    const e = esc(line);
    if (/^[A-Z][A-Z0-9_\-+]*$/.test(line.trim())) return `<span class="kw">${e}</span>`;
    if (/^[\s\d.\-+Ee]+$/.test(line) && line.trim()) return `<span class="num">${e}</span>`;
    return e;
  }).join("\n");
}

/* ---------------------------------------------------------------- start */
async function start() {
  try {
    const [boot, colors] = await Promise.all([
      api("boot"),
      fetch(assetUrl("element_colors.json"), { credentials: "same-origin" }).then((r) => r.json()).catch(() => null),
    ]);
    state.info = boot.info;
    state.boot = boot;
    state.templates = boot.templates;
    state.colors = colors;
    state.palettes = boot.palettes || { palettes: {} };
    // the shell keeps it in step with Settings' colorScheme, both ways
    if (colors) fillColorOptions();
    await loadFiles(boot.files);
  } catch (e) { toast(e.message, true); }
  // the shell module may run after this (it is loaded later): it also checks the flag
  state.ready = true;
  document.dispatchEvent(new CustomEvent("mace-app-ready", { detail: state.info || null }));
}
window.addEventListener("resize", () => Object.values(state.viewers).forEach((v) => v && (v.resize(), v.render())));
window.MACE = { isFreq, isJobLog, roleOf, roleInfo, chipOf, middleName, api, q, esc, fmt, signed, hm, toast, cssVar, plotVar, plotVarIn, plotTone, vpVar, plotPaper, styleFigure, isPublication, plotPreset, presetOf, spinStyle, volumeColors, mapScale, elementColors: () => currentPalette(), syncSurfaces, store, viewer, openFile, load3Dmol,
  loadPlotly, assetUrl, applyTheme, drawStructure, viewAlong, fileByPath, renderFiles, drawEnergy, energyTSV, scfTSV, drawScf, scfSection, displayAtoms, bondsFor, addModel,
  drawCell, axisLabels, fitView, fitPoints, styleFor, colorSpec, effDraw, elementColor, showTab, state, highlightDeck,
  colorScheme, paletteMap, paletteIds, paletteLabel, setPalettes, setPaletteDraft, fillColorOptions,
  redrawViews, sphereScale, schemeName, viewerBg, renderStructureInfo,
  loadFiles: () => loadFiles(), loadStructure: (p, d) => loadStructure(p, d), ensureStep: (t, i) => ensureStep(t, i), expandSummary,
  setTrajectory: (t, title, status) => setTrajectory(t, title, status), markTab,
  /** Select frame k of the run at path (Run Analysis' clicks; the run must be the one loaded). */
  selectFrame: (path, k) => {
    if (!state.traj || state.traj.path !== path) return false;
    const fr = $("#o-frame");
    fr.value = String(k);
    fr.dispatchEvent(new Event("input"));
    return true;
  },
  frameOf: (path) => (state.traj && state.traj.path === path ? state.frame : null),
  isLive: (p) => !!(state.live && state.live.path === p && !state.live.done), liveState: () => state.live && { ...state.live, timer: undefined },
  stopLive, startLive: (p) => startLive(p),
  setLook, cycleLook, showSub, compareSvgItems, cmpExport, cmpTsv, matchNote, refused, look: () => state.layout };
start();
})();
