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
  cube: "cube", xyz: "xyz", poscar: "vasp", mol: "mol", pdb: "pdb", gen: "gen", fort34: "f34", f25: "f25",
  transport: "trn", crystaldata: "data", wavefunction: "wf", script: "sh", log: "log", joblog: "log",
  OPT: "OPT", SP: "SP", FREQ: "FREQ", BAND: "BAND", DOSS: "DOSS", "CHARGE+POTENTIAL": "CHG+POT", TRANSPORT: "TRANSP",
  PROPERTIES: "PROPS", props: "PROPS" };
const FILTERS = [
  ["all", "All", null],
  ["struct", "Structures", ["cif", "d12", "out", "xyz", "poscar", "mol", "pdb", "gen", "fort34"]],
  // Properties (spec §1.1): band, DOS, IR and Raman data, FREQ outputs (by their index: the details), plots,
  // volumes, fort.25, transport data and properties outputs (by content)
  ["results", "Properties", ["band", "dos", "ir", "raman", "image", "html", "freq", "cube", "f25", "transport", "props"]],
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
function store(key, val) {
  try { if (val === undefined) return localStorage.getItem(key); localStorage.setItem(key, val); }
  catch (e) { return null; }
}

/* ---------------------------------------------------------------- theme */
function applyTheme(t) {
  if (t === "system") document.documentElement.removeAttribute("data-theme");
  else if (t) document.documentElement.setAttribute("data-theme", t);
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
    .then((c) => { window.MACE.standardView = c.standardView; }, () => { /* the old tilt stays */ });
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
  import(new URL(assetUrl("js/viewport/rendering.js"), document.baseURI).href)]).then(([, clip, look]) => {
    state.gl = { clip, look };
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
  state.gl.look.apply(v, { lightStrength: num("lightStrength", 1), lightDirection: setting("lightDirection", "front"),
    ambient: num("ambient", 0), outline: setting("outline", false) === true, outlineWidth: num("outlineWidth", 0.1),
    // no color chosen: one that stands out on the drawn background (a style read only when an outline is drawn)
    outlineColor: setting("outline", false) === true ? setting("outlineColor", "")
      || state.gl.look.autoOutline(setting("background", "theme"), setting("backgroundColor", ""), cssVar("--text")) : "",
    ambientOcclusion: num("ambientOcclusion", 0), lod: v.maceLod || 0 });
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
    `<p>This browser cannot draw 3D here (WebGL is off or not supported, which is common in remote desktops). ` +
    `Plots, text, the database and Setup still work. Try another browser, turn on hardware acceleration, or open ` +
    `MACE GUI from your own computer with mace gui connect.${diag}</p></div>`);
  const b = el.querySelector(".no3d-diag");
  if (b) b.addEventListener("click", () => window.MACE.runCommand && window.MACE.runCommand("help.copyDiagnostics"));
}
function viewer(id) {
  if (!state.viewers[id] && !webgl().ok) { no3d(document.getElementById(id)); return null; }
  if (!state.viewers[id]) {
    const el = document.getElementById(id);
    // antialias alone makes 3Dmol also draw at 2x the pixels (its "upscale"),
    // 4x the fill work at devicePixelRatio 1; the context's own multisampling
    // smooths the edges without it
    // antialias needs a new WebGL context: it applies to viewers created after it changes
    // the structure viewer's pointer belongs to the shell (viewport/pointer.js):
    // 3Dmol's own mouse handling stays on for the run and comparison viewers
    const v = $3Dmol.createViewer(el, { backgroundColor: viewerBg(),
      antialias: setting("antialias", true) !== false, upscale: false, id: id + "-gl", minimumZoomToDistance: 1,
      nomouse: id === "viewer-main" });
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
        const now = el.offsetWidth + "x" + el.offsetHeight;
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
      const now = el.offsetWidth + "x" + el.offsetHeight;
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
 * #opt-colors select only mirrors it). */
function schemeName() {
  const c = state.colors, name = setting("colorScheme", c ? c.default : "vesta");
  return c && c.schemes[name] ? name : (c ? c.default : name);
}
function colorScheme() {
  const c = state.colors;
  if (!c) return "Jmol";
  return { prop: "elem", map: c.schemes[schemeName()] };
}
function elementColor(sym) {
  const c = state.colors;
  if (!c) return cssVar("--fallback-atom");
  return (c.schemes[schemeName()] || {})[sym] || c.fallback;
}
/* Ball-and-stick spheres are atomScale x 0.7 with bonds, x 0.95 without: the
 * default 0.40 gives the 0.28 and 0.38 the classic views always drew. */
const round6 = (x) => Math.round(x * 1e6) / 1e6;
function sphereScale(bonds) { return round6(num("atomScale", 0.4) * (bonds ? 0.7 : 0.95)); }
function styleFor(kind, bonds, opacity = 1) {
  const cs = colorScheme();
  if (kind === "spacefill") return { sphere: { scale: 0.95, colorscheme: cs, opacity } };
  if (kind === "stick") return { stick: { radius: 0.16, colorscheme: cs, opacity } };
  return bonds ? { sphere: { scale: sphereScale(true), colorscheme: cs, opacity }, stick: { radius: num("bondRadius", 0.12), colorscheme: cs, opacity } }
               : { sphere: { scale: sphereScale(false), colorscheme: cs, opacity } };
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
    supercell: $("#opt-super").checked, bonds: $("#opt-bonds").checked,
    labels: $("#opt-labels").checked, style: $("#opt-style").value };
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
  el.addEventListener("change", () => drawStructure(true)));
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
/* opts.tilt: the Phase 1-3 view (a fixed turn from no rotation, a loose sphere fit) instead of the standard view's tight fit */
function fitView(v, s, atoms, n = 1, pts = null, opts = {}) {
  pts = pts || fitPoints(s, atoms, n);
  const doFit = () => {
    const m = v.addModel();
    m.addAtoms(pts.map((p, k) => ({ elem: "H", x: p[0], y: p[1], z: p[2], index: k, bonds: [] })));
    m.setStyle({}, {});
    v.zoomTo({ model: m });
    v.removeModel(m);
    // the cell's standard view (c up, as the structure viewport shows it) once geom/core.js is here
    const q = !opts.tilt && window.MACE.standardView && s && s.cell ? window.MACE.standardView(s.cell, s.pbc || [true, true, true]) : null;
    if (q) { const view = v.getView(); v.setView([...view.slice(0, 4), ...q]); }
    else { v.rotate(-75, "x"); v.rotate(-20, "y"); }
    const el = v.getCanvas ? v.getCanvas() : null;
    const w = el ? el.clientWidth : 0, h = el ? el.clientHeight : 0;
    if (opts.tilt) {
      // the Phase 1-3 zoom too: a sphere around the points, about 1.6 Å of margin
      const c = mul(pts.reduce((t, p) => add(t, p), [0, 0, 0]), 1 / pts.length);
      const R = Math.max(1, ...pts.map((p) => norm(sub(p, c))));
      v.zoom(0.92 * (R / (R + 1.6)) * Math.min(1, w > 0 && h > 0 ? w / h : 1));
    } else if (w > 0 && h > 0 && v.modelToScreen) {
      // a tight fit of the projected points (atom centers, cell corners), with
      // room for the drawn balls: 85% of the limiting side, centered
      const rc = el.getBoundingClientRect(), xyz = pts.map((p) => ({ x: p[0], y: p[1], z: p[2] }));
      const box = () => {
        // 3Dmol updates its size and matrices when it paints (once per frame here): paint this view now
    if (!(v.paintNow && v.paintNow())) { try { if (v.setSlabAndFog) v.setSlabAndFog(); if (v.scene && v.scene.updateMatrixWorld) v.scene.updateMatrixWorld(true); } catch (e) { /* not drawn yet */ } }
        let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
        for (const p of v.modelToScreen(xyz)) {
          const x = p.x - rc.left, y = p.y - rc.top;
          if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
        }
        return { x0, x1, y0, y1 };
      };
      for (let pass = 0; pass < 3; pass++) {
        let b = box();
        if (!(b.x1 > b.x0 && b.y1 > b.y0)) break;
        const f = 0.85 * Math.min(w / (b.x1 - b.x0), h / (b.y1 - b.y0));
        if (Math.abs(f - 1) > 0.005) v.zoom(f);
        b = box();
        const dx = w / 2 - (b.x0 + b.x1) / 2, dy = h / 2 - (b.y0 + b.y1) / 2;
        if (Math.hypot(dx, dy) > 0.5) v.translateScene(dx, dy);
        if (Math.abs(f - 1) < 0.01 && Math.hypot(dx, dy) < 1) break;
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
  await loadStructure(path, last);
  if (t.frames.length > 1) {
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

function autoScale() {
  const m = state.cmp ? state.cmp.max_displacement : 0;
  const s = m > 0 ? Math.max(1, Math.min(200, Math.round(1.5 / m))) : 1;
  $("#o-scale").value = s; $("#o-scale-out").textContent = s;
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
      rows: c.displacements.map((d) => ({ atom: `${d.symbol}${d.atom}`, index: d.atom, length: d.length, dx: d.vector[0], dy: d.vector[1], dz: d.vector[2] })),
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
  $("#o-stats").innerHTML = `
    <div class="stat"><div class="k">max |Δr|</div><div class="v">${fmt(c.max_displacement, 3)} Å</div></div>
    <div class="stat"><div class="k">rms |Δr|</div><div class="v">${fmt(c.rms_displacement, 3)} Å</div></div>
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
  const drawCell = (v, s, ...rest) => (periodic(s) ? drawCellBox(v, s, ...rest) : null);
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
  [vb, side ? viewer("viewer-after") : null].forEach((v) => v && (v.removeAllModels(), v.removeAllShapes(), v.removeAllLabels()));
  const atoms0 = c.before_cart.map((p, i) => ({ i, cart: p, image: false }));
  const s0w = { ...s0, cart: c.before_cart };
  const s1u = { ...s1, cart: c.after_unwrapped_cart };
  const atoms1 = s1u.cart.map((p, i) => ({ i, cart: p, image: false }));
  const base = { bonds: true, style: "ballstick" };
  const muted = cssVar("--muted"), accent = cssVar("--accent");
  renderLegend(c);
  if (state.layout === "twotone") {
    const ca = cssVar("--cmp-a"), cb = cssVar("--cmp-b");
    tint(addModel(vb, s0w, atoms0, base), s0w, (a) => mixHex(elementColor(s0w.symbols[a.serial - 1]), ca, 0.7));
    tint(addModel(vb, s1u, atoms1, base), s1u, (a) => mixHex(elementColor(s1u.symbols[a.serial - 1]), cb, 0.7));
    drawCell(vb, s0, ca, 0.02, 1, true);
    drawCell(vb, s1, cb, 0.03);
    if (keep && !state.refit) vb.setView(keep); else fitView(vb, s1, atoms0.concat(atoms1));
    state.refit = false;
    vb.render();
    return;
  }
  if (state.layout === "dmap") {
    const L = c.displacements.map((d) => d.length);
    const lo = Math.min(...L), hi = Math.max(...L);
    tint(addModel(vb, s1u, atoms1, base), s1u, (a) => seqColor(hi > lo ? (L[a.serial - 1] - lo) / (hi - lo) : 0));
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
    fitView(vb, s0, atoms0, 1, pts); fitView(va, s1, atoms1, 1, pts);
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
function tint(m, s, fn) {
  const st = styleFor("ballstick", true, 1);
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
  if (look === "twotone") html = `<span class="lg-twotone"><i class="sw" style="background:var(--cmp-a)"></i>initial</span><span><i class="sw" style="background:var(--cmp-b)"></i>${state.frame === state.traj.frames.length - 1 ? "final" : "selected frame"}</span>`;
  else if (look === "dmap") {
    const L = c.displacements.map((d) => d.length);
    const lo = L.length ? Math.min(...L) : 0, hi = L.length ? Math.max(...L) : 0;
    const dg = hi - lo < 0.01 ? 4 : 3;           // a small spread still shows two different ends
    html = `<span class="lg-seq"><span>|Δr|</span><span class="num">${fmt(lo, dg)} Å</span><i class="seq-bar" style="background:linear-gradient(90deg, ${seqStops().join(", ")})"></i><span class="num">${fmt(hi, dg)} Å</span></span><span><i class="sw arrow"></i>scaled arrows</span>`;
  } else if (look === "side") html = `<span><i class="sw" style="background:var(--muted)"></i>initial (left)</span><span><i class="sw solid"></i>selected frame (right)</span><span><i class="sw arrow"></i>relaxation (cell strain excluded)</span>`;
  else html = `<span><i class="sw ghost"></i>initial</span><span><i class="sw solid"></i>selected frame</span><span><i class="sw arrow"></i>relaxation (cell strain excluded)</span>`;
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
      const L = c.displacements.map((d) => d.length), lo = Math.min(...L), hi = Math.max(...L);
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
  c.displacements.forEach((d, i) => {
    if (d.length < 1e-4) return;
    const start = atomsAt[i].cart, end = add(start, mul(d.vector, scale));
    v.addArrow({ start: xyz(start), end: xyz(end), radius: 0.06, radiusRatio: 2.2, mid: 0.75,
      color: cssVar("--sel") });
  });
}

/* The energy chart is a small hand-drawn SVG, so a trajectory never needs
 * plotly (4.6 MB): plotly loads only when a plot opens (Open as Plot).
 * PLAN_analysis C3.8: x is CRYSTAL's optimization POINT (frame k = POINT k+1;
 * the final frame is a ring at the last point), energies on the optimizer's
 * surface (C0), relative to the first or the final point of each FINALRUN
 * segment (D6), in a chosen unit, or the log of |ΔE|. */
const E_UNITS = { eV: 1, meV: 1000, Ha: 1 / 27.211386245988, "kJ/mol": 96.48533212331, "kcal/mol": 23.060547830619 };
function cardKey(t) { return `mace.gui.runcard.${(state.info && state.info.dir_hash) || ""}.${t.path || ""}`; }
function cardOpts(t) {
  let o = null;
  try { o = JSON.parse(store(cardKey(t)) || "null"); } catch (e) { o = null; }
  const su = setting("energyUnit", "eV");
  return { log: false, rel: "first", unit: su && su !== "eV" ? su : "meV", ...(o || {}) };
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
  const segRef = {};
  for (const p of pts) {
    if (o.rel === "first") { if (!(p.seg in segRef)) segRef[p.seg] = p.e; }
    else segRef[p.seg] = p.e;                             // the segment's last (the final frame for the last segment)
  }
  for (const p of pts) {
    p.dE = (p.e - segRef[p.seg]) * HARTREE_MEV / 1000 * f;      // Ha -> eV -> unit
    p.isRef = Math.abs(p.e - segRef[p.seg]) <= 1e-10;          // the printed precision
    p.below = o.rel === "final" && p.e < segRef[p.seg] - 1e-10;
    p.y = o.log ? (p.isRef || p.dE === 0 ? null : Math.abs(p.dE)) : p.dE;
  }
  return { pts, o, conv };
}
function drawEnergy() {
  const t = state.traj;
  const el = $("#o-energy");
  if (!t) return;
  const { pts: all, o, conv } = energySeries(t);
  const pts = all.filter((p) => p.y !== null);
  $("#o-energy-card").classList.toggle("no-data", all.length < 2);
  $("#o-e-log").setAttribute("aria-pressed", String(!!o.log));
  $("#o-e-rel").value = o.rel; $("#o-e-unit").value = o.unit;
  $("#o-energy-label").innerHTML = `Energy along the run <span class="muted">· ${esc(o.unit)}, ${o.log ? "log |ΔE|, " : ""}relative to the ${o.rel === "final" ? "final" : "first"} point${conv && conv.segments && conv.segments.length > 1 ? " of each FINALRUN segment" : ""}</span>`;
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
  // FINALRUN restarts: a dotted line between the segments
  if (conv) for (const sgm of conv.segments.slice(1)) {
    const xm = sgm.first_point - 0.5;
    svg += `<line class="seg-line" x1="${X(xm)}" x2="${X(xm)}" y1="${T}" y2="${H - B}"><title>FINALRUN restart (re-referenced)</title></line>`;
  }
  svg += `<text class="axis-title" x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">Optimization point</text>`;
  svg += `<text class="axis-title" transform="translate(12 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">${o.log ? "|ΔE|" : "ΔE"} (${esc(o.unit)})</text>`;
  const bySeg = {};
  pts.filter((p) => !p.ring).forEach((p) => { (bySeg[p.seg] = bySeg[p.seg] || []).push(p); });
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
function energyTSV() {
  const t = state.traj;
  if (!t) return "";
  const { pts, o } = energySeries(t);
  return [`point\tframe\tE_Ha\tdE_${o.unit}`, ...pts.map((p) => `${p.x}\t${p.k}\t${p.e}\t${p.dE}`)].join("\n") + "\n";
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
$("#o-e-tsv").addEventListener("click", async () => {
  const text = energyTSV(), n = text.trim().split("\n").length - 1;
  toast((await copyPlain(text)) ? `Copied ${n} rows` : "Copy failed", "info");
});
$("#o-e-plot").addEventListener("click", () => { if (window.__maceOpenRunPlot && state.traj) window.__maceOpenRunPlot(state.traj.path, "energy"); });
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
    el.onload = () => resolve();
    el.onerror = () => { toast("plotly is not installed here: interactive plots are unavailable", true); resolve(); };
    document.head.appendChild(el);
  });
  return plotlyLoading;
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
    if (colors) {
      $("#opt-colors").innerHTML = Object.keys(colors.schemes).map((k) =>
        `<option value="${k}">${k === "vesta" ? "VESTA colors" : "Jmol colors"}</option>`).join("");
      // the shell keeps it in step with Settings' colorScheme, both ways
      $("#opt-colors").value = schemeName();
    }
    await loadFiles(boot.files);
  } catch (e) { toast(e.message, true); }
  // the shell module may run after this (it is loaded later): it also checks the flag
  state.ready = true;
  document.dispatchEvent(new CustomEvent("mace-app-ready", { detail: state.info || null }));
}
window.addEventListener("resize", () => Object.values(state.viewers).forEach((v) => v && (v.resize(), v.render())));
window.MACE = { isFreq, isJobLog, roleOf, roleInfo, chipOf, middleName, api, q, esc, fmt, signed, hm, toast, cssVar, store, viewer, openFile, load3Dmol,
  loadPlotly, assetUrl, applyTheme, drawStructure, viewAlong, fileByPath, renderFiles, drawEnergy, displayAtoms, bondsFor, addModel,
  drawCell, axisLabels, fitView, fitPoints, styleFor, elementColor, showTab, state, highlightDeck,
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
