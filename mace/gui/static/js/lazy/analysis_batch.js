/* Explorer › Analyze Structures… (1.1.6): one Structure Analysis tool and
 * its settings on many structure files. Each file is one background task
 * (the Tasks tab lists it, Cancel stops it), a few at a time (Settings ›
 * Advanced › Analyze Structures: files at once, 1 to 4: never the whole
 * machine). The summary (one row per structure, CSV) is in the bottom
 * panel's Analysis tab, under Batches, and each file's result is listed
 * with that file's document when it is open. The command line does the
 * same: `mace analyze <tool> FILE... --csv summary.csv`. */
import { loadCss, panel } from "./lazy_util.js";
import { store, BATCH_TOOLS, TOOL_TITLE } from "./analysis_store.js";

const css = () => Promise.all([loadCss(import.meta.url, "../../css/dialogs.css"), loadCss(import.meta.url, "../../css/analysis.css"),
  loadCss(import.meta.url, "../../css/analysis_results.css")]);
let info = null;
let last = { tool: "porosity", params: {} };          // the dialog opens with the last batch's tool and settings

const fmt = (x, n = 3) => (x == null ? "" : Number(x).toFixed(n));

/** The dialog. ``paths``: the files picked; ``all``: the folder's structure files; ``done(batch)`` once it starts. */
export async function open(H, { paths = [], all = [], done = null } = {}) {
  await css();
  if (!info) { try { info = await H.api("analysis/info"); } catch (e) { info = { probes: {}, sources: {} }; } }
  const esc = H.esc, A = store(H);
  const files = [...new Set([...paths, ...all])];
  const picked = new Set(paths);
  let jobs = 2;
  try { jobs = Math.min(4, Math.max(1, Number(H.settings.get("analysis.batchJobs")) || 2)); } catch (e) { /* the default */ }
  const body = document.createElement("div");
  body.className = "an-root anb";
  body.innerHTML = `<div class="an-row"><label for="anb-tool">Tool</label><select id="anb-tool" data-k="tool">${BATCH_TOOLS.map((t) =>
    `<option value="${t}"${t === last.tool ? " selected" : ""}>${esc(TOOL_TITLE[t])}</option>`).join("")}</select>
    <label for="anb-jobs">Files at a time</label><input id="anb-jobs" type="number" min="1" max="4" step="1" value="${jobs}" data-k="jobs"></div>
    <form class="an-form" data-k="params"></form>
    <div class="anb-files"><div class="an-row"><input type="search" data-k="filter" placeholder="Filter files" aria-label="Filter files">
      <span class="insp-hint" data-k="count"></span><button type="button" class="tb-btn-text" data-k="all">Select All</button>
      <button type="button" class="tb-btn-text" data-k="none">None</button></div>
      <ul class="anb-list" data-k="list" aria-label="Structure files"></ul></div>
    <p class="insp-hint">Each structure is one background task (the Tasks tab lists them); the summary and every structure's result go to the bottom panel's Analysis tab.</p>`;
  const d = panel(H, { id: "analysis-batch", title: "Analyze Structures", body, wide: true, resizable: true,
    foot: `<span class="insp-hint an-cli" data-k="cli"></span><button class="tb-btn-text" data-k="cancel">Cancel</button>
      <button class="tb-btn-text primary" data-k="run"></button>` });
  d.classList.add("an-panel");
  const $ = (k) => d.querySelector(`[data-k="${k}"]`);
  const params = () => {
    const out = {};
    $("params").querySelectorAll("[data-p]").forEach((e) => { out[e.dataset.p] = e.type === "checkbox" ? e.checked : e.value; });
    return out;
  };
  function form() {
    const t = $("tool").value, p = last.tool === t ? last.params : {};
    const v = (k, dflt) => esc(p[k] !== undefined ? p[k] : dflt);
    let h = "";
    if (t === "porosity") {
      const probes = Object.entries(info.probes || {});
      h = `<div class="an-row"><label>Adsorbate</label><select data-p="probe">${probes.map(([k, x]) =>
        `<option value="${esc(k)}"${k === (p.probe || "N2") ? " selected" : ""}>${esc(k)}, ${esc(x.label)} (r ${fmt(x.radius)} Å)</option>`).join("")}
        <option value=""${p.probe === "" ? " selected" : ""}>Custom radius…</option></select>
        <label>Probe radius (Å)</label><input type="number" step="0.01" min="0" max="10" data-p="probe_radius" placeholder="from the adsorbate" value="${v("probe_radius", "")}"></div>
        <div class="an-row"><label>Atomic radii</label><select data-p="radii"><option value="ccdc">CCDC (Zeo++ default)</option><option value="uff"${p.radii === "uff" ? " selected" : ""}>UFF (σ/2)</option></select>
        <label>Grid spacing (Å)</label><input type="number" step="0.01" min="0.05" max="1" value="${v("spacing", "0.15")}" data-p="spacing">
        <label title="The 3D surfaces of every structure stay in memory (a few MB each)"><input type="checkbox" data-p="render"${p.render ? " checked" : ""}> Keep the 3D pore surfaces</label></div>`;
    } else if (t === "pxrd") {
      h = `<div class="an-row"><label>Radiation</label><select data-p="source">${Object.entries(info.sources || {}).map(([k, x]) =>
        `<option value="${esc(k)}"${k === (p.source || "CuKa") ? " selected" : ""}>${esc(k)} (${fmt(x.wavelength, 5)} Å)</option>`).join("")}</select>
        <label>2θ from</label><input type="number" data-p="min" value="${v("min", "5")}" step="0.5"><label>to</label><input type="number" data-p="max" value="${v("max", "90")}" step="0.5">
        <label>Step</label><input type="number" data-p="step" value="${v("step", "0.02")}" step="0.005"></div>`;
    } else if (t === "topology") {
      h = `<div class="an-row"><label>Simplification</label><select data-p="mode">${[["auto", "Automatic"], ["sbu", "Building units (MOF)"], ["atoms", "Atoms (zeolites, inorganic)"], ["all", "All atoms"]]
        .map(([k, x]) => `<option value="${k}"${k === (p.mode || "auto") ? " selected" : ""}>${x}</option>`).join("")}</select>
        <label>Bond tolerance</label><input type="number" data-p="tolerance" value="${v("tolerance", "1.15")}" step="0.05" min="0.8" max="1.6"></div>`;
    } else if (t === "polyhedra") {
      h = `<div class="an-row"><label>Bond tolerance</label><input type="number" data-p="tolerance" value="${v("tolerance", "1.15")}" step="0.05" min="0.8" max="1.6"></div>`;
    } else if (t === "bonds") {
      h = `<div class="an-row"><label>Bond tolerance</label><input type="number" data-p="tolerance" value="${v("tolerance", "1.15")}" step="0.05" min="0.8" max="1.6">
        <label>Bins</label><input type="number" data-p="bins" value="${v("bins", "40")}" min="5" max="400"></div>`;
    } else if (t === "morphology") {
      h = `<div class="an-row"><label>Model</label><select data-p="model"><option value="bfdh">BFDH (from the structure)</option></select>
        <label>Max index</label><input type="number" data-p="max_index" value="${v("max_index", "2")}" min="1" max="4"></div>`;
    } else if (t === "planes") {
      h = `<div class="an-row"><label>(hkl)</label><input type="text" data-p="hkl" value="${v("hkl", "1 1 1")}" size="8">
        <label>Position</label><input type="number" data-p="offset" value="${v("offset", "0")}" step="0.05" min="0" max="1">
        <label>Atoms within (Å)</label><input type="number" data-p="tol" value="${v("tol", "0.3")}" step="0.05" min="0"></div>`;
    }
    $("params").innerHTML = h;
    // each label names the field after it (screen readers read it; a click on it focuses the field)
    $("params").querySelectorAll("label").forEach((l) => {
      const f = l.nextElementSibling;
      if (l.querySelector("input") || !f || !f.matches("[data-p]")) return;
      f.id = `anb-p-${f.dataset.p}`;
      l.htmlFor = f.id;
    });
    $("params").querySelectorAll("[data-p]").forEach((e) => e.addEventListener("change", cli));
    cli();
  }
  function list() {
    const q = $("filter").value.trim().toLowerCase();
    const shown = files.filter((f) => !q || f.toLowerCase().includes(q));
    // the picked ones first, in the order picked
    shown.sort((a, b) => Number(picked.has(b)) - Number(picked.has(a)));
    $("list").innerHTML = shown.map((f) => `<li><label><input type="checkbox" data-f="${esc(f)}"${picked.has(f) ? " checked" : ""}> <span>${esc(f)}</span></label></li>`).join("")
      || `<li class="insp-hint">No structure files${q ? " match" : ""}.</li>`;
    $("list").querySelectorAll("[data-f]").forEach((e) => e.addEventListener("change", () => {
      if (e.checked) picked.add(e.dataset.f); else picked.delete(e.dataset.f);
      count();
    }));
    count();
  }
  function count() {
    $("count").textContent = `${picked.size} of ${files.length} selected`;
    const b = $("run");
    b.textContent = `Run on ${picked.size} Structure${picked.size === 1 ? "" : "s"}`;
    b.disabled = !picked.size;
    cli();
  }
  function cli() {
    const t = $("tool").value;
    const rec = { tool: t, params: params(), doc: { path: "FILE..." }, result: {} };
    $("cli").textContent = `Command line: ${A.cli(rec).replace("FILE...", picked.size > 1 ? "FILE..." : [...picked][0] || "FILE")}${picked.size > 1 ? ` --csv ${t}_summary.csv` : ""}`;
  }
  $("tool").addEventListener("change", form);
  $("filter").addEventListener("input", list);
  $("all").addEventListener("click", () => { files.filter((f) => { const q = $("filter").value.trim().toLowerCase(); return !q || f.toLowerCase().includes(q); }).forEach((f) => picked.add(f)); list(); });
  $("none").addEventListener("click", () => { picked.clear(); list(); });
  $("cancel").addEventListener("click", () => d.close());
  $("run").addEventListener("click", () => {
    const t = $("tool").value, p = params();
    if (t === "porosity" && !p.render) p.render = false;
    const n = Math.min(4, Math.max(1, Math.round(Number($("jobs").value) || 2)));
    try { H.settings.set("analysis.batchJobs", n); } catch (e) { /* kept for this batch only */ }
    last = { tool: t, params: p };
    const order = files.filter((f) => picked.has(f));
    const b = A.batch(t, p, order, n);
    H.toast(`${TOOL_TITLE[t]} of ${order.length} structures: ${n} at a time, in the background`, "info");
    d.close();
    if (done) done(b);
  });
  form();
  list();
  return d;
}
