/* The bottom panel's Analysis tab (1.1.6): every Structure Analysis result
 * of the document in focus (or of all open documents), what it was computed
 * with (probe and radius, radii set, grid, engine, wavelength, ...), its key
 * numbers and its own table, and the Analyze Structures… batches with their
 * summary (one row per structure: the columns of `mace analyze <tool>
 * FILE... --csv`). A porosity result lists every channel and pocket with
 * its volume, surface, largest included sphere and center, the totals and
 * averages; a click on a row draws that feature in the highlight color with
 * the others dimmed and frames it. It reads the store the Structure
 * Analysis panel writes (analysis_store.js); loaded the first time the tab
 * is shown. */
import { loadCss } from "./lazy_util.js";
import { store, TOOL_TITLE, OVERLAY_OF, csvOf } from "./analysis_store.js";

const css = () => Promise.all([loadCss(import.meta.url, "../../css/analysis.css"), loadCss(import.meta.url, "../../css/analysis_results.css")]);
let ui = null;          // { H, A, host, scope: "doc"|"all"|"batch", sel, bsel, feature, busy: Map(rec id -> text) }

const num = (x, n = 3) => (x == null || Number.isNaN(Number(x)) ? "–" : Number(x).toFixed(n));
const when = (t) => new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
const stem = (t) => String(t || "structure").replace(/\.[A-Za-z0-9]+$/, "").replace(/[^A-Za-z0-9._-]+/g, "_");

function download(name, text, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url; a.download = name; a.hidden = true;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

/** Show the tab in ``host`` (the shell calls it when the tab is shown and the document changes). */
export async function show(H, host) {
  await css();
  if (!ui) {
    ui = { H, A: store(H), host, scope: "doc", sel: null, bsel: null, feature: null, busy: new Map() };
    let frame = 0;
    const later = () => { cancelAnimationFrame(frame); frame = requestAnimationFrame(() => { if (ui.host.classList.contains("active")) render(); }); };
    document.addEventListener("mace-analysis", (e) => {
      const d = e.detail || {};
      if (d.what === "batch" && d.quiet && ui.scope !== "batch") return;
      if (d.what === "add" && ui.scope !== "batch") ui.sel = d.id;        // a new result: shown at once
      later();
    });
    document.addEventListener("mace-doc-switch", () => { unpick(); later(); });
  }
  ui.H = H; ui.host = host; ui.A = store(H);
  render();
}
/** Open the tab on one batch (Analyze Structures… does). */
export async function showBatch(H, host, id) {
  if (!ui) await show(H, host);
  ui.scope = "batch"; ui.bsel = id;
  render();
}

/** Drops the channel or pocket picked in the table: its highlight on the 3D view goes too. */
function unpick() {
  if (ui.feature != null) {
    const rec = ui.sel ? ui.A.get(ui.sel) : null, it = rec ? ui.A.overlayOf(rec) : null;
    if (it) ui.A.overlay().highlight(it.key, null);
  }
  ui.feature = null;
}

function records() {
  const d = ui.H.shell().activeDoc();
  return ui.scope === "all" ? ui.A.all() : ui.A.forDoc(d);
}

function render() {
  const { H, A, host } = ui, esc = H.esc, S = H.shell();
  const doc = S.activeDoc();
  const batches = A.batches();
  const list = ui.scope === "batch" ? [] : records();
  if (ui.scope !== "batch" && (!ui.sel || !list.some((r) => r.id === ui.sel))) ui.sel = list.length ? list[0].id : null;
  if (ui.scope === "batch" && (!ui.bsel || !batches.some((b) => b.id === ui.bsel))) ui.bsel = batches.length ? batches[0].id : null;
  const chip = (k, t, n) => `<button type="button" class="anr-scope" role="tab" data-scope="${k}" aria-selected="${ui.scope === k}">${esc(t)}${n != null ? ` <span class="anr-n">${n}</span>` : ""}</button>`;
  const keepScroll = [...host.querySelectorAll(".anr-list, .anr-detail")].map((e) => e.scrollTop);
  host.innerHTML = `<div class="anr">
    <div class="anr-bar" role="tablist" aria-label="Analysis results of">${chip("doc", doc ? doc.title : "This document")}${chip("all", "All open documents")}${chip("batch", "Batches", batches.length)}
      <span class="anr-hint insp-hint">${ui.scope === "batch" ? "Explorer › select structures › right-click › Analyze Structures…"
        : "Analysis › Structure Analysis computes these; results stay with their structure for this session"}</span></div>
    <div class="anr-main"><div class="anr-list" role="listbox" aria-label="${ui.scope === "batch" ? "Batches" : "Results"}"></div><div class="anr-detail"></div></div></div>`;
  host.querySelectorAll(".anr-scope").forEach((b) => b.addEventListener("click", () => { unpick(); ui.scope = b.dataset.scope; render(); }));
  const L = host.querySelector(".anr-list"), D = host.querySelector(".anr-detail");
  if (ui.scope === "batch") { renderBatches(L, D, batches); restore(keepScroll); return; }
  if (!list.length) {
    L.innerHTML = `<p class="empty-note">${ui.scope === "all" ? "No analysis results yet." : doc ? `No analysis results for ${esc(doc.title)}.` : "No document is open."}</p>`;
    D.innerHTML = `<p class="empty-note">Analysis › Structure Analysis (porosity, powder pattern, topology, polyhedra, morphology, planes) lists its results here, with the settings each was computed with.</p>`;
    return;
  }
  L.innerHTML = list.map((r) => {
    const k = A.keys(r).slice(0, 4).map(([a, b]) => `<span class="anr-k"><span>${esc(a)}</span> ${esc(b)}</span>`).join("");
    const set = A.settings(r).slice(0, 2).map(([, v]) => esc(v)).join(" · ");
    return `<div class="anr-row" role="option" tabindex="0" data-id="${r.id}" aria-selected="${r.id === ui.sel}">
      <div class="anr-r1"><b>${esc(TOOL_TITLE[r.tool] || r.tool)}</b>${ui.scope === "all" || r.doc.id == null ? `<span class="anr-doc">${esc(r.doc.title)}</span>` : ""}
        ${r.batch ? `<span class="anr-tag">batch</span>` : ""}<span class="anr-when">${when(r.when)}</span></div>
      <div class="anr-r2">${set}</div><div class="anr-r3">${k}</div></div>`;
  }).join("");
  L.querySelectorAll(".anr-row").forEach((row) => {
    const pick = () => { unpick(); ui.sel = row.dataset.id; render(); };
    row.addEventListener("click", pick);
    row.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); }
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        const sib = e.key === "ArrowDown" ? row.nextElementSibling : row.previousElementSibling;
        if (sib) { unpick(); ui.sel = sib.dataset.id; render(); const n = ui.host.querySelector(`.anr-row[data-id="${sib.dataset.id}"]`); if (n) n.focus(); }
      }
    });
  });
  const rec = A.get(ui.sel);
  if (rec) detail(D, rec);
  restore(keepScroll);
}
function restore(s) {
  const els = ui.host.querySelectorAll(".anr-list, .anr-detail");
  els.forEach((e, i) => { if (s[i]) e.scrollTop = s[i]; });
}

/* ---------------------------------------------------------------- one result */
function detail(D, rec) {
  const { H, A } = ui, esc = H.esc, S = H.shell();
  const kind = OVERLAY_OF[rec.tool], it = A.overlayOf(rec);
  const canShow = kind && !(rec.tool === "porosity" && !rec.result.render);
  const shown = !!it && it.visible;
  const openDoc = rec.doc.id != null ? (S.docById ? S.docById(rec.doc.id) : null) : null;
  const busy = ui.busy.get(rec.id);
  const sets = A.settings(rec).map(([k, v]) => `<span class="anr-set"><span>${esc(k)}</span> ${esc(v)}</span>`).join("");
  const keys = A.keys(rec).map(([k, v]) => `<span class="anr-k"><span>${esc(k)}</span> ${esc(v)}</span>`).join("");
  const btn = (a, label, extra = "") => `<button type="button" class="tb-btn-text" data-a="${a}"${extra}>${label}</button>`;
  const ibtn = (a, icon, label, cls = "") => `<button type="button" class="icon-btn sm ${cls}" data-a="${a}" aria-label="${esc(label)}" title="${esc(label)}">${H.icon(icon)}</button>`;
  let html = `<div class="anr-head"><h3>${esc(TOOL_TITLE[rec.tool] || rec.tool)} of ${esc(rec.doc.title)}</h3>
    <span class="anr-when" title="${esc(new Date(rec.when).toLocaleString())}">${when(rec.when)}</span>
    <div class="anr-acts">
      ${kind ? btn("overlay", shown ? "Hide in 3D" : "Show in 3D", canShow ? "" : ` disabled title="The mesh data was deleted: run it again to draw it"`) : ""}
      ${btn("rerun", "Run Again", `${busy || (rec.doc.id != null && !openDoc) ? " disabled" : ""} title="${rec.doc.id != null && !openDoc ? "The document is closed" : "The same tool and settings on the same structure, as a background task"}"`)}
      ${btn("cli", "Copy Command", ` title="${esc(A.cli(rec))}"`)}${btn("json", "JSON", ` title="Download the result and its settings (.json)"`)}${btn("csv", "CSV", ` title="Download the table below and its settings (.csv)"`)}
      ${rec.tool === "porosity" && rec.result.render ? ibtn("meshes", "eye-off", "Delete Mesh Data (the 3D surfaces; the numbers stay)") : ""}
      ${ibtn("delete", "trash-2", "Delete Result", "anr-del")}
      ${busy ? `<span class="an-status busy" role="status">${esc(busy)}</span>` : ""}</div></div>
    <div class="anr-sets" aria-label="Settings">${sets}</div><div class="anr-keys" aria-label="Key numbers">${keys}</div>`;
  if (rec.tool === "porosity") html += poreTable(rec);
  else {
    const t = A.table(rec);
    if (t.rows.length) html += tableHtml(t.head, t.rows);
    else html += `<p class="insp-hint">No rows.</p>`;
  }
  html += `<code class="anr-cli" title="Copy Command copies it">${esc(A.cli(rec))}</code>`;
  D.innerHTML = html;
  D.querySelectorAll("[data-a]").forEach((b) => b.addEventListener("click", () => act(b.dataset.a, rec)));
  if (rec.tool === "porosity") bindFeatures(D, rec);
}

function tableHtml(head, rows, cls = "") {
  return `<div class="anr-table-wrap"><table class="an-table num ${cls}"><thead><tr>${head.map((h) => `<th scope="col">${h}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${ui.H.esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

/* the porosity result's channels and pockets, one row each, with totals and averages */
function poreTable(rec) {
  const esc = ui.H.esc, r = rec.result, F = r.features || [];
  const probe = r.probe ? `${r.probe.name} (r = ${num(r.probe.radius)} Å)` : "the probe";
  const note = `<p class="anr-note">Sorbate <b>${esc(probe)}</b> · ${esc(String(r.radii || "").toUpperCase())} radii · grid ${num(r.grid && r.grid.spacing)} Å.
    Channel: accessible (the probe reaches it from outside); pocket: inaccessible (closed to the probe).
    Volume: what the probe sphere can occupy; surface: where it touches the atoms; D<sub>i</sub>: the largest sphere inside, centered at the point given (Cartesian, Å).
    Click a row to show it in 3D.</p>`;
  let html = "";
  if (!F.length) return note + `<p class="insp-hint">${r.features ? "No channels or pockets: the probe fits nowhere." : "This result has no per-feature table (computed before MACE 1.1.6 final)."}</p>`;
  const row = (f) => `<tr class="anr-feat" tabindex="0" data-f="${f.id}" aria-selected="${ui.feature === f.id}">
    <td>${f.id + 1}</td><td title="${f.accessible ? "Accessible channel: the probe reaches it from outside" : "Inaccessible pocket: closed to the probe"}"><span class="an-swatch ${f.accessible ? "acc" : "nac"}"></span> ${f.accessible ? "Channel" : "Pocket"}</td>
    <td>${f.accessible ? `${f.dim}D${f.axes && f.axes.some(Boolean) ? ` (${f.axes.map((x, i) => (x ? "abc"[i] : "")).join("")})` : ""}` : "0D"}</td>
    <td>${num(f.occupiable_A3, 1)}</td><td>${num(f.area_A2, 1)}</td><td>${num(f.Di, 2)}</td>
    <td>${f.Di_center_cart ? f.Di_center_cart.map((x) => num(x, 2)).join("&nbsp; ") : "–"}</td></tr>`;
  const sum = (list, k) => list.reduce((s, f) => s + (Number(f[k]) || 0), 0);
  const ch = F.filter((f) => f.accessible), pk = F.filter((f) => !f.accessible);
  const foot = [];
  for (const [list, name, tot] of [[ch, "channels", [r.POAV && r.POAV.A3, r.ASA && r.ASA.A2]], [pk, "pockets", [r.PONAV && r.PONAV.A3, r.NASA && r.NASA.A2]]]) {
    if (!list.length) continue;
    const di = list.map((f) => f.Di);
    foot.push(`<tr class="anr-tot"><th scope="row" colspan="3">Total, ${list.length} ${list.length === 1 ? name.slice(0, -1) : name}</th><td>${num(sum(list, "occupiable_A3"), 1)}</td><td>${num(sum(list, "area_A2"), 1)}</td>
      <td>${num(Math.max(...di), 2)} max</td><td title="${name === "channels" ? `POAV ${num(tot[0], 1)} Å³, ASA ${num(tot[1], 1)} Å²` : `PONAV ${num(tot[0], 1)} Å³, NASA ${num(tot[1], 1)} Å²`}">= ${name === "channels" ? "POAV, ASA" : "PONAV, NASA"}</td></tr>`);
    if (list.length > 1) {
      foot.push(`<tr class="anr-avg"><th scope="row" colspan="3">Average per ${name.slice(0, -1)}</th><td>${num(sum(list, "occupiable_A3") / list.length, 1)}</td>
        <td>${num(sum(list, "area_A2") / list.length, 1)}</td><td>${num(di.reduce((s, x) => s + x, 0) / list.length, 2)}</td><td></td></tr>`);
    }
  }
  html += `<div class="anr-table-wrap"><table class="an-table num anr-pores"><thead><tr><th scope="col">#</th><th scope="col">Type</th><th scope="col">Dim.</th>
    <th scope="col">Volume (Å³)</th><th scope="col">Surface (Å²)</th><th scope="col">D<sub>i</sub> (Å)</th><th scope="col">D<sub>i</sub> center (Å)</th></tr></thead>
    <tbody>${F.map(row).join("")}</tbody><tfoot>${foot.join("")}</tfoot></table></div>`;
  return html + note;
}
function bindFeatures(D, rec) {
  D.querySelectorAll(".anr-feat").forEach((tr) => {
    const pick = () => highlight(rec, Number(tr.dataset.f) === ui.feature ? null : Number(tr.dataset.f));
    tr.addEventListener("click", pick);
    tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
  });
}
/** One channel or pocket in the highlight color, the rest dimmed, and the view framed on it (null: none). */
function highlight(rec, fid) {
  const { A, H } = ui, S = H.shell();
  ui.feature = fid;
  const d = A.docFor(rec);
  if (!d || S.activeDoc() !== d) {
    if (fid != null) H.toast(`Open ${rec.doc.title} to see it in 3D`, "info");
    render();
    return;
  }
  let it = A.overlayOf(rec);
  if (!it && fid != null) { A.show(rec); it = A.overlayOf(rec); }
  if (it) {
    const box = A.overlay().highlight(it.key, fid);
    if (box && S.framePoints) S.framePoints(box, 1.4);
  }
  render();
}

/* ---------------------------------------------------------------- actions */
async function act(a, rec) {
  const { A, H } = ui, S = H.shell();
  if (a === "overlay") {
    const it = A.overlayOf(rec);
    if (it && it.visible) A.overlay().setVisible(it.key, false);
    else if (it) A.overlay().setVisible(it.key, true);
    else if (!A.show(rec)) H.toast("Nothing to draw for this result", "info");
    else if (!A.docFor(rec) || S.activeDoc() !== A.docFor(rec)) H.toast(`Drawn on ${rec.doc.title} when it is in front`, "info");
    render();
  } else if (a === "cli") H.copyText(A.cli(rec), "Command copied");
  else if (a === "json") download(`${stem(rec.doc.title)}_${rec.tool}.json`, A.json(rec), "application/json");
  else if (a === "csv") download(`${stem(rec.doc.title)}_${rec.tool}.csv`, A.csv(rec), "text/csv");
  else if (a === "meshes") { A.dropMeshes(rec); H.toast("Mesh data deleted: the numbers stay", "info"); render(); }
  else if (a === "delete") { A.remove(rec.id); H.toast(`Deleted the ${TOOL_TITLE[rec.tool] || rec.tool} result of ${rec.doc.title}`, "info"); render(); }
  else if (a === "rerun") {
    const doc = rec.doc.id != null ? S.docById(rec.doc.id) : null;
    ui.busy.set(rec.id, "Running…");
    render();
    try {
      const nr = await A.run(rec.tool, rec.params, { doc, file: doc ? null : rec.doc.path, extra: rec.extra || {},
        onUpdate: (t) => { if (t.state === "running") { ui.busy.set(rec.id, `${t.message || "running"}${t.progress ? ` (${Math.round(100 * t.progress)}%)` : ""}`); refreshBusy(rec); } } });
      if (nr) { ui.sel = nr.id; if (OVERLAY_OF[nr.tool] && A.overlayOf(rec)) A.show(nr); }
    } catch (e) {
      const canceled = e.state === "canceled" || !!(e.data && e.data.canceled);
      if (!canceled) H.toast(`${TOOL_TITLE[rec.tool]}: ${e.message}`, "error");
    } finally { ui.busy.delete(rec.id); render(); }
  }
}
function refreshBusy(rec) {
  const s = ui.host.querySelector(".anr-detail .an-status");
  if (s && ui.sel === rec.id) s.textContent = ui.busy.get(rec.id) || "";
}

/* ---------------------------------------------------------------- batches */
// a batch's settings, as the dialog labels them
const PARAM = { probe: "Probe", probe_radius: "Probe radius (Å)", radii: "Atomic radii", spacing: "Grid spacing (Å)", render: "3D surfaces",
  source: "Radiation", min: "2θ from", max: "2θ to", step: "Step", mode: "Simplification", tolerance: "Bond tolerance", bins: "Bins",
  model: "Model", max_index: "Max index", hkl: "(hkl)", offset: "Position", tol: "Atoms within (Å)" };
function renderBatches(L, D, batches) {
  const { H, A } = ui, esc = H.esc;
  if (!batches.length) {
    L.innerHTML = `<p class="empty-note">No batches yet.</p>`;
    D.innerHTML = `<p class="empty-note">Select structure files in the Explorer, right-click, Analyze Structures…: one tool and its settings on every file, a few at a time in the background, with a summary here (CSV, the columns of <code>mace analyze &lt;tool&gt; FILE... --csv</code>).</p>`;
    return;
  }
  L.innerHTML = batches.map((b) => {
    const n = b.rows.length, done = b.rows.filter((r) => r.state === "done").length, bad = b.rows.filter((r) => r.state === "failed").length;
    const running = b.rows.some((r) => r.state === "running" || r.state === "queued");
    return `<div class="anr-row" role="option" tabindex="0" data-b="${b.id}" aria-selected="${b.id === ui.bsel}">
      <div class="anr-r1"><b>${esc(TOOL_TITLE[b.tool] || b.tool)}</b><span class="anr-doc">${n} structure${n === 1 ? "" : "s"}</span><span class="anr-when">${when(b.started)}</span></div>
      <div class="anr-r2">${running ? `Running: ${done} of ${n} done` : `${done} done${bad ? `, ${bad} failed` : ""}`}</div>
      <div class="anr-bar-p" role="progressbar" aria-valuemin="0" aria-valuemax="${n}" aria-valuenow="${done}"><span style="width:${(100 * done / n).toFixed(1)}%"></span></div></div>`;
  }).join("");
  L.querySelectorAll(".anr-row").forEach((row) => {
    const pick = () => { ui.bsel = row.dataset.b; render(); };
    row.addEventListener("click", pick);
    row.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
  });
  const b = batches.find((x) => x.id === ui.bsel);
  if (!b) return;
  const t = A.batchTable(b);
  const running = b.rows.some((r) => r.state === "running" || r.state === "queued");
  const fake = { tool: b.tool, params: b.params, doc: { path: "FILE..." }, result: {} };
  const cmd = A.cli(fake).replace(" FILE...", ` ${b.rows.map((r) => (/\s/.test(r.path) ? `"${r.path}"` : r.path)).join(" ")}`) + ` --csv ${b.tool}_summary.csv`;
  const sets = Object.entries(b.params).filter(([k, v]) => v !== "" && v != null && (v !== false || k === "render"))
    .map(([k, v]) => `<span class="anr-set"><span>${esc(PARAM[k] || k)}</span> ${esc(k === "render" ? (v ? "kept" : "not kept") : k === "radii" ? String(v).toUpperCase() : String(v))}</span>`).join("");
  let html = `<div class="anr-head"><h3>${esc(TOOL_TITLE[b.tool] || b.tool)} of ${b.rows.length} structures</h3><span class="anr-when">${new Date(b.started).toLocaleString()} · ${b.jobs} at a time</span></div>
    <div class="anr-sets">${sets}</div>
    <div class="anr-acts"><button type="button" class="tb-btn-text" data-a="csv">Export CSV</button>
      <button type="button" class="tb-btn-text" data-a="json">JSON</button>
      <button type="button" class="tb-btn-text" data-a="cli">Copy Command</button>
      ${running ? `<button type="button" class="tb-btn-text" data-a="cancel">Cancel the Rest</button>` : ""}</div>
    <code class="anr-cli">${esc(cmd)}</code>`;
  const state = (r) => ({ queued: "Waiting", running: `Running${r.progress ? ` ${Math.round(100 * r.progress)}%` : ""}`, done: "Done", failed: "Failed", canceled: "Canceled" })[r.state] || r.state;
  const cols = t.cols.filter((c) => c !== "state" && c !== "error");
  html += `<div class="anr-table-wrap"><table class="an-table num anr-batch"><thead><tr><th scope="col">State</th>${cols.map((c) => `<th scope="col">${esc(c)}</th>`).join("")}<th scope="col"></th></tr></thead><tbody>` +
    b.rows.map((r, i) => `<tr class="s-${r.state}"><td title="${esc(r.error || "")}">${esc(state(r))}${r.error ? `: ${esc(r.error)}` : ""}</td>` +
      cols.map((c) => { const v = t.rows[i][t.cols.indexOf(c)]; return `<td>${esc(v == null ? "" : typeof v === "number" ? String(+v.toPrecision(6)) : String(v))}</td>`; }).join("") +
      `<td>${r.state === "done" ? `<button type="button" class="tb-btn-text" data-open="${esc(r.path)}" title="Open ${esc(r.path)}: its results are listed with it">Open</button>` : ""}</td></tr>`).join("") +
    `</tbody></table></div>`;
  D.innerHTML = html;
  D.querySelectorAll("[data-open]").forEach((x) => x.addEventListener("click", () => H.openPath(x.dataset.open)));
  D.querySelectorAll("[data-a]").forEach((x) => x.addEventListener("click", () => {
    if (x.dataset.a === "csv") download(`${b.tool}_summary.csv`, csvOf(t.cols, t.rows), "text/csv");
    else if (x.dataset.a === "json") {
      download(`${b.tool}_batch.json`, JSON.stringify(b.rows.map((r) => {
        const rec = r.rec ? A.get(r.rec) : null;
        const { render: _, ...res } = (rec && rec.result) || {};
        return { file: r.path, tool: b.tool, params: b.params, ...(r.state === "done" ? { result: res } : { error: r.error || r.state }) };
      }), null, 1), "application/json");
    } else if (x.dataset.a === "cli") H.copyText(cmd, "Command copied");
    else if (x.dataset.a === "cancel") A.cancelBatch(b);
  }));
}
