/* The bottom panel's Properties tab (1.1.6): everything materials.db holds
 * for the run of the document in focus (an output, or a plot, spectrum,
 * volume or deck a run wrote), grouped by calculation type and group, with
 * units and where each value came from (the file, the extractor or the GUI's
 * Save to Database, a derived property's version, when). A run that is not
 * in the database shows the same values computed on the fly (nothing
 * written) and offers Extract; with no database, Create Database…. Copy,
 * CSV and JSON as in the Analysis tab. Add Reference… picks the runs a
 * reference energy (formation, cohesive, surface, adsorption, reaction)
 * combines with this one; the extraction computes and stores it and saves
 * the choice with the run. Units (prop_units.js): a preset at the top sets
 * every row's unit (Hartree atomic units, eV/Å, kJ/mol, as stored) and the
 * normalization of the extensive ones (total, per atom, per formula unit); a
 * row's Unit cell picks its own until the preset changes again (Normalize
 * rescales it and keeps its unit). A quantity the
 * database holds in several units (E_D3 in Hartree and eV) is one row.
 * Display only. Server: mace/gui/properties_routes.py;
 * the keys: mace/properties (docs/properties.md). Loaded the first time the
 * tab is shown. */
import { loadCss } from "./lazy_util.js";
import { dialog } from "../ui.js";
import * as PU from "./prop_units.js";

const css = () => Promise.all([loadCss(import.meta.url, "../../css/analysis.css"), loadCss(import.meta.url, "../../css/analysis_results.css"),
  loadCss(import.meta.url, "../../css/properties.css")]);
let ui = null;          // { H, host, scope, filter, cache: Map(path -> entry), busy, units, shown }

/* The unit preset, normalization and each row's own unit, kept in this browser (profile.js LOCAL_ONLY). */
const UKEY = "mace.gui.propunits.v1";
function loadUnits() {
  const st = { preset: "stored", norm: "total", over: {} };
  try {
    const s = JSON.parse(localStorage.getItem(UKEY) || "null");
    if (s && typeof s === "object") {
      if (typeof s.preset === "string") st.preset = s.preset;
      if (PU.NORMS[s.norm]) st.norm = s.norm;
      if (s.over && typeof s.over === "object") st.over = s.over;
    }
  } catch (err) { /* private window, blocked storage: the defaults */ }
  return st;
}
function saveUnits() {
  try { localStorage.setItem(UKEY, JSON.stringify(ui.units)); } catch (err) { /* private window */ }
}

const base = (p) => String(p || "").split("/").pop();
const stem = (t) => String(t || "run").replace(/\.[A-Za-z0-9]+$/, "").replace(/[^A-Za-z0-9._-]+/g, "_");
const EXTRACTOR = "crystal_property_extractor.py";

function download(name, text, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url; a.download = name; a.hidden = true;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}
/** A label's markup (<sub>, <sup>, from the registry) kept, everything else escaped. */
function labelHtml(esc, label) {
  return esc(label || "").replace(/&lt;(\/?)(sub|sup)&gt;/g, "<$1$2>");
}
function csvCell(v) {
  const s = v == null ? "" : String(v);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}
function scriptName(s) {
  if (!s) return "";
  if (s === EXTRACTOR) return "extractor";
  return s;
}

/** Show the tab in ``host`` (the shell calls it when the tab is shown and when the document changes). */
export async function show(H, host) {
  await css();
  if (!ui) {
    ui = { H, host, scope: "run", filter: "", cache: new Map(), busy: null, units: loadUnits(), shown: [] };
    document.addEventListener("mace-doc-switch", () => {
      requestAnimationFrame(() => { if (ui.host.classList.contains("active")) render(); });
    });
  }
  ui.H = H; ui.host = host;
  render();
}
/** Forget what was read (a database update, an extraction): the next render asks again. */
export function invalidate(path = null) {
  if (!ui) return;
  if (path) ui.cache.delete(path); else ui.cache.clear();
  if (ui.host && ui.host.classList.contains("active")) render();
}
/** The rows shown (tests and the parity check), with the unit state and each row's value and unit as shown. */
export function state() {
  if (!ui) return null;
  const e = ui.cache.get(keyOf(currentPath()));
  if (!e) return null;
  const cat = catalogOf(e);
  return { ...e, units: ui.units, rows: shownRows(e).map((r) => {
    const ch = PU.chosen(r, cat, ui.units), s = PU.shown(r, cat, ch);
    return { ...r, shown_value: s.value, shown_text: s.text, shown_unit: s.unit, choice: `${ch.unit}|${ch.norm}` };
  }) };
}
/** The unit catalog of an entry (mace/properties/units.catalog, with the rows). */
function catalogOf(e) {
  return (e.state === "db" ? e.data && e.data.units : e.computed && e.computed.units) || null;
}

function currentPath() {
  const S = ui.H.shell(), d = S.activeDoc();
  return d ? ui.H.docPathOf(d) : null;
}
function keyOf(path) { return `${path}|${ui.scope}`; }

function load(path) {
  const k = keyOf(path);
  let e = ui.cache.get(k);
  if (e) return e;
  e = { path, state: "loading", data: null, computed: null, error: null, computing: false };
  ui.cache.set(k, e);
  const q = new URLSearchParams({ path, scope: ui.scope });
  const hint = ui.H.openDbPath();
  if (hint) q.set("db", hint);
  ui.H.api("properties/run?" + q).then((data) => {
    e.data = data; e.state = data.state;
    if (data.state === "not-in-db" || data.state === "no-db") compute(e);
    later();
  }).catch((err) => { e.state = "error"; e.error = err.message; later(); });
  return e;
}
function compute(e) {
  if (e.computing || e.computed) return;
  e.computing = true;
  ui.H.apiTask("properties/compute?" + new URLSearchParams({ path: e.path }), undefined, null, { get: true })
    .then((r) => { e.computed = r; })
    .catch((err) => { e.computeError = err.message; })
    .finally(() => { e.computing = false; later(); });
}
let frame = 0;
function later() { cancelAnimationFrame(frame); frame = requestAnimationFrame(() => { if (ui.host.classList.contains("active")) render(); }); }

function shownRows(e) {
  // a key of a quantity also stored in another unit is shown by that one's row (merged_into)
  const src = (e.state === "db" ? (e.data.rows || []) : ((e.computed && e.computed.rows) || [])).filter((r) => !r.merged_into);
  const f = ui.filter.trim().toLowerCase();
  const rows = f ? src.filter((r) => `${r.label_plain} ${r.name} ${(r.also || []).join(" ")} ${r.group}`.toLowerCase().includes(f)) : src;
  const order = ["Energy", "Reference energies", "Electronic", "Band structure", "Density of states", "Effective masses", "Thermodynamics", "Vibrations",
    "Mechanical", "Transport", "Relaxation time", "Lattice", "Structure", "Symmetry", "Populations", "Optimization",
    "Charge and potential", "Orbitals", "SCF settings", "Run", "Other"];
  const gi = (g) => { const i = order.indexOf(g); return i < 0 ? order.length : i; };
  return [...rows].sort((a, b) => String(a.calc_type || "").localeCompare(String(b.calc_type || "")) || gi(a.group) - gi(b.group) ||
    String(a.group).localeCompare(String(b.group)) || a.name.localeCompare(b.name));
}

/* ---------------------------------------------------------------- render */
function render() {
  const { H, host } = ui, esc = H.esc, S = H.shell();
  const d = S.activeDoc();
  const path = d ? H.docPathOf(d) : null;
  if (!path) {
    host.innerHTML = `<p class="empty-note">Open a CRYSTAL output (or a plot, spectrum, volume or deck a run wrote) to see its properties.</p>`;
    return;
  }
  const e = load(path);
  const keep = host.querySelector(".prp-body") ? host.querySelector(".prp-body").scrollTop : 0;
  const filterFocused = document.activeElement && document.activeElement.classList.contains("prp-filter");
  const data = e.data || {};
  const chip = (k, t) => `<button type="button" class="anr-scope" role="tab" data-scope="${k}" aria-selected="${ui.scope === k}">${esc(t)}</button>`;
  const mat = data.calc && data.calc.material_id;
  const rows = e.state === "loading" || e.state === "error" ? [] : shownRows(e);
  const btn = (a, label, tip, extra = "") => `<button type="button" class="tb-btn-text" data-a="${a}" title="${esc(tip)}"${extra}>${esc(label)}</button>`;
  let where = "";
  if (e.state === "db") {
    const c = data.calc;
    where = `<span class="prp-where">${esc(base(data.run))} · ${esc(c.calc_type || "")} · ${esc(c.calc_id)} · <span title="${esc(data.db)}">${esc(base(data.db))}</span></span>`;
  } else if (data.run) where = `<span class="prp-where">${esc(base(data.run))}</span>`;
  host.innerHTML = `<div class="prp">
    <div class="anr-bar prp-bar" role="tablist" aria-label="Properties of">
      ${chip("run", "This run")}${mat ? chip("material", `All runs of ${mat}`) : ""}${where}
      <input type="search" class="field prp-filter" placeholder="Filter" aria-label="Filter properties" value="${esc(ui.filter)}">
      ${unitBar(rows, catalogOf(e))}
      <span class="anr-acts">
        ${btn("copy", "Copy", "Copy the rows shown (tab-separated: label, value, unit, key, source)", rows.length ? "" : " disabled")}
        ${btn("csv", "CSV", "Download the rows shown (.csv)", rows.length ? "" : " disabled")}
        ${btn("json", "JSON", "Download the rows shown with their provenance (.json)", rows.length ? "" : " disabled")}
        ${e.state === "db" && refProps(e).length ? btn("addref", "Add Reference…", "Choose the runs a reference energy (formation, cohesive, surface, adsorption, reaction) combines with this one", ui.busy ? " disabled" : "") : ""}
        ${e.state === "db" ? btn("extract", "Extract Again", `Extract ${base(data.run)} into ${base(data.db)} again: ${data.extract ? data.extract.command : ""}`, ui.busy ? " disabled" : "") : ""}
      </span></div>
    ${banner(e)}
    <div class="prp-body">${rows.length ? table(rows, e) : ""}${footer(e)}</div></div>`;
  host.querySelectorAll(".anr-scope").forEach((b) => b.addEventListener("click", () => { ui.scope = b.dataset.scope; render(); }));
  const fi = host.querySelector(".prp-filter");
  fi.addEventListener("input", () => { ui.filter = fi.value; render(); });
  if (filterFocused) { fi.focus(); fi.setSelectionRange(fi.value.length, fi.value.length); }
  host.querySelectorAll("[data-a]").forEach((b) => b.addEventListener("click", () => act(b.dataset.a, e, b)));
  host.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => view(e, b.dataset.view, b.dataset.calc)));
  host.querySelectorAll("[data-addref]").forEach((b) => b.addEventListener("click", () => picker(e, b.dataset.addref)));
  host.querySelectorAll("select[data-u]").forEach((sel) => sel.addEventListener("change", () => {
    PU.setTop(ui.units, sel.dataset.u, sel.value);
    saveUnits();
    applyUnits(e);
  }));
  host.querySelectorAll("select.prp-usel").forEach((sel) => sel.addEventListener("change", () => {
    const r = ui.shown[Number(sel.closest("tr").dataset.i)];
    ui.units.over[r.name] = sel.value;
    saveUnits();
    host.querySelectorAll("tr[data-i]").forEach((tr) => { if (tr.dataset.name === r.name) updateRow(tr, catalogOf(e)); });
  }));
  const body = host.querySelector(".prp-body");
  if (body && keep) body.scrollTop = keep;
}

function banner(e) {
  const esc = ui.H.esc, data = e.data || {};
  const box = (cls, html) => `<div class="prp-banner ${cls}" role="status">${html}</div>`;
  if (e.state === "loading") return box("busy", "Reading the database…");
  if (e.state === "error") return box("err", `Properties: ${esc(e.error)}`);
  if (e.state === "no-run") return box("", `No run found for ${esc(base(data.path))}: the Properties tab shows CRYSTAL outputs and the files a run wrote.`);
  const computedNote = e.computing ? " Computing its values from the run…" : e.computeError ? ` Computing from the run failed: ${esc(e.computeError)}` :
    e.computed ? " Shown: its values computed from the run now (the extractor and the derived properties); nothing is stored." : "";
  if (e.state === "not-in-db") {
    const u = data.untracked && data.untracked.rows ? ` (${data.untracked.rows} untracked rows of ${esc(data.untracked.material_id)} are not tied to this run)` : "";
    return box("warn", `<b>Not in the database yet:</b> ${esc(base(data.run))} has no rows in ${esc(data.db)}${u}.${computedNote}
      <button type="button" class="tb-btn-text primary" data-a="extract"${ui.busy ? " disabled" : ""} title="${esc(data.extract.command)}">Extract</button>
      <code class="anr-cli">${esc(data.extract.command)}</code>`);
  }
  if (e.state === "no-db") {
    return box("warn", `<b>No database</b> for ${esc(base(data.run))} (no materials.db in its folder or above, inside the served folder).${computedNote}
      <button type="button" class="tb-btn-text" data-a="create" title="Create or Update Database from ${esc(data.folder || "the served folder")}">Create Database…</button>`);
  }
  if (e.state === "db" && data.missing && data.missing.length) {
    return box("info", `${data.missing.map(esc).join(", ")} ${data.missing.length === 1 ? "is" : "are"} not in the database for this run yet (it was extracted before ${data.missing.length === 1 ? "it" : "they"} existed or changed).
      <button type="button" class="tb-btn-text" data-a="extract"${ui.busy ? " disabled" : ""}>Extract Again</button>`);
  }
  if (ui.busy) return box("busy", esc(ui.busy));
  return "";
}

function source(r) {
  const bits = [];
  if (r.file) bits.push(r.file);
  bits.push(scriptName(r.script));
  if (r.derived) bits.push(`${r.derived} v${r.version || "?"}`);
  return bits.filter(Boolean).join(" · ");
}
/* ---------------------------------------------------------------- units */
/** The preset and normalization controls (only when a row shown can be converted). */
function unitBar(rows, cat) {
  const esc = ui.H.esc;
  if (!cat || !rows.some((r) => PU.choices(r, cat).length)) return "";
  const st = ui.units;
  const opt = (v, t, cur) => `<option value="${esc(v)}"${v === cur ? " selected" : ""}>${esc(t)}</option>`;
  const ext = rows.some((r) => r.extensive && (r.atoms > 0 || r.formula_units > 0));
  return `<span class="prp-units">
    <label title="Sets every row's unit. A unit picked in a row's Unit cell wins until this changes.">Units
      <select class="field" data-u="preset" aria-label="Unit preset">${cat.presets.map((p) => opt(p.id, p.label, st.preset)).join("")}</select></label>
    ${ext ? `<label title="Per-cell (extensive) values: as stored (the cell), per atom or per formula unit; a row keeps the unit picked in its Unit cell. Intensive values (a gap, an energy per atom) are not changed.">Normalize
      <select class="field" data-u="norm" aria-label="Normalize extensive values">${Object.entries(PU.NORMS).map(([k, t]) => opt(k, t, st.norm)).join("")}</select></label>` : ""}
  </span>`;
}
/** A row's value and unit cells under the current unit state. */
function unitCells(r, cat) {
  const esc = ui.H.esc;
  const ch = PU.chosen(r, cat, ui.units), s = PU.shown(r, cat, ch);
  const groups = PU.choices(r, cat);
  if (!groups.length) return { val: esc(s.text), unit: esc(s.unit) };
  const key = `${ch.unit}|${ch.norm}`;
  const own = !!ui.units.over[r.name];
  const opts = (g) => g.choices.map((c) => `<option value="${esc(c.key)}"${c.key === key ? " selected" : ""}>${esc(c.label)}</option>`).join("");
  const body = groups.map((g) => (g.group ? `<optgroup label="${esc(g.group)}">${opts(g)}</optgroup>` : opts(g))).join("");
  const tip = own ? "The unit picked for this row (the Units preset at the top resets it)" : "Pick this row's unit";
  return { val: esc(s.text), unit: `<select class="prp-usel${own ? " own" : ""}" aria-label="Unit of ${esc(r.label_plain)}" title="${esc(tip)}">${body}</select>` };
}
/** Update one row's value and unit in place (nothing else is drawn again). */
function updateRow(tr, cat) {
  const r = ui.shown[Number(tr.dataset.i)];
  if (!r || r.json) return;
  const ch = PU.chosen(r, cat, ui.units), s = PU.shown(r, cat, ch);
  tr.querySelector(".prp-val").textContent = s.text;
  const sel = tr.querySelector("select.prp-usel");
  if (sel) {
    sel.value = `${ch.unit}|${ch.norm}`;
    const own = !!ui.units.over[r.name];
    sel.classList.toggle("own", own);
    sel.title = own ? "The unit picked for this row (the Units preset at the top resets it)" : "Pick this row's unit";
  } else tr.querySelector(".prp-unit").textContent = s.unit;
}
/** The preset or the normalization changed: every row's value and unit, in place. */
function applyUnits(e) {
  const cat = catalogOf(e);
  ui.host.querySelectorAll(".prp-table tr[data-i]").forEach((tr) => updateRow(tr, cat));
}

function table(rows, e) {
  const esc = ui.H.esc;
  let html = "", cur = null;
  const scope = e.state === "db" && ui.scope === "material";
  const cat = catalogOf(e);
  ui.shown = rows;
  rows.forEach((r, i) => {
    const g = scope ? `${r.calc_type || "?"} · ${r.group}` : r.group;
    if (g !== cur) {
      if (cur !== null) html += "</tbody>";
      html += `<tbody class="prp-group"><tr class="prp-gh"><th scope="rowgroup" colspan="5">${esc(g)}</th></tr>`;
      cur = g;
    }
    const tip = [r.description, r.extracted_at ? `extracted ${r.extracted_at}` : "", r.inputs ? `from ${r.inputs.join(", ")}` : ""].filter(Boolean).join("\n");
    const u = r.json ? null : unitCells(r, cat);
    const val = r.json ? `<span class="prp-json">${esc(r.display)}</span> <button type="button" class="tb-btn-text prp-view" data-view="${esc(r.name)}" data-calc="${esc(r.calc_id || "")}">View</button>`
      : u.val;
    const also = r.also && r.also.length ? ` <span class="prp-also" title="${esc(`Also stored as ${r.also.join(", ")}: the same quantity in another unit (pick the unit in the Unit column)`)}">+${r.also.length}</span>` : "";
    html += `<tr data-i="${i}" data-name="${esc(r.name)}"${also ? ` data-also="${esc(r.also.join(" "))}"` : ""}${r.interactive ? ' class="prp-gui"' : ""}><td class="prp-label" title="${esc(tip)}">${labelHtml(esc, r.label)}</td>
      <td class="prp-val">${val}</td><td class="prp-unit">${u ? u.unit : esc(r.unit_text || "")}</td><td class="prp-key"><code>${esc(r.name)}</code>${also}</td>
      <td class="prp-src" title="${esc(tip)}">${esc(source(r))}</td></tr>`;
  });
  html += "</tbody>";
  return `<table class="an-table num prp-table" aria-label="Properties"><thead><tr><th scope="col">Property</th><th scope="col">Value</th><th scope="col">Unit</th>
    <th scope="col">Database key</th><th scope="col">Source</th></tr></thead>${html}</table>`;
}
function footer(e) {
  const esc = ui.H.esc;
  const der = (e.state === "db" ? e.data.derived : e.computed && e.computed.derived) || [];
  if (!der.length) return "";
  const add = (x) => e.state === "db" && x.references && x.references.length
    ? ` <button type="button" class="tb-btn-text prp-addref" data-addref="${esc(x.name)}"${ui.busy ? " disabled" : ""}>Add Reference…</button>` : "";
  const item = (x) => `<li data-derived="${esc(x.name)}"><code>${esc(x.name)}</code> <span class="prp-st prp-st-${esc(String(x.status || "").split(" ")[0])}">${esc(x.status)}</span>` +
    (x.reason ? `: ${esc(x.reason)}` : "") + (x.inputs && x.inputs.length > 1 ? ` <span class="insp-hint">(${esc(x.inputs.slice(1).join(", "))})</span>` : "") + add(x) + "</li>";
  const cli = e.state === "db" && e.data.calc ? `<code class="anr-cli">mace database --action properties --material-id ${esc(e.data.calc.material_id)} --from-calc ${esc(e.data.calc.calc_type || "")}</code>` : "";
  return `<div class="prp-foot"><p class="anr-note">Derived properties of this run (mace/properties; docs/properties.md):</p><ul class="prp-derived">${der.map(item).join("")}</ul>${cli}</div>`;
}

/* ---------------------------------------------------------------- actions */
/** The rows shown, each in the unit shown (value, unit), with the stored value and unit beside it. */
function exportRows(e) {
  const cat = catalogOf(e);
  return shownRows(e).map((r) => {
    const ch = PU.chosen(r, cat, ui.units), s = PU.shown(r, cat, ch);
    return { calc_type: r.calc_type, group: r.group, label: r.label_plain, name: r.name, also: (r.also || []).join(" "),
      value: s.value, unit: s.unit, normalization: s.stored ? "" : ch.norm === "total" ? "" : `per ${ch.norm}`,
      stored_value: r.value, stored_unit: r.unit_text || "", file: r.file, extractor: r.script, derived: r.derived, version: r.version,
      extracted_at: r.extracted_at, calc_id: r.calc_id, material_id: r.material_id, inputs: r.inputs || null };
  });
}
function csvText(rows) {
  const cols = ["calc_type", "group", "label", "name", "also", "value", "unit", "normalization", "stored_value", "stored_unit", "file", "extractor",
    "derived", "version", "extracted_at", "calc_id", "material_id"];
  return [cols.join(","), ...rows.map((r) => cols.map((c) => csvCell((c === "value" || c === "stored_value") && r[c] !== null && typeof r[c] === "object" ? JSON.stringify(r[c]) : r[c])).join(","))].join("\n") + "\n";
}
async function act(a, e, b) {
  const { H } = ui, data = e.data || {};
  const name = stem(base(data.run || e.path));
  if (a === "copy") {
    const rows = shownRows(e), cat = catalogOf(e);
    H.copyText(rows.map((r) => {
      const s = PU.shown(r, cat, PU.chosen(r, cat, ui.units));
      return [r.label_plain, r.json ? (r.value != null ? JSON.stringify(r.value) : r.display) : s.text, s.unit, [r.name, ...(r.also || [])].join(" "), source(r)].join("\t");
    }).join("\n"), `Copied ${rows.length} properties`);
  } else if (a === "csv") download(`${name}_properties.csv`, csvText(exportRows(e)), "text/csv");
  else if (a === "json") {
    const doc = { run: data.run || null, database: data.db || null, state: e.state, calc: data.calc || (e.computed && e.computed.calc) || null,
      units: { preset: ui.units.preset, normalization: ui.units.norm, rows: ui.units.over },
      provenance: data.provenance || null, derived: (e.state === "db" ? data.derived : e.computed && e.computed.derived) || [], properties: exportRows(e) };
    download(`${name}_properties.json`, JSON.stringify(doc, null, 1), "application/json");
  } else if (a === "create") H.dbBuild({ folder: data.folder || "" });
  else if (a === "extract") extract(e, b);
  else if (a === "addref") picker(e, (refProps(e)[0] || {}).name);
}
/** The derived properties of the run that combine it with other runs (the reference energies). */
function refProps(e) {
  return ((e.state === "db" ? e.data && e.data.derived : null) || []).filter((x) => x.references && x.references.length);
}
async function extract(e) {
  const { H } = ui, data = e.data || {};
  if (!data.db || !data.run) return;
  ui.busy = `Extracting ${base(data.run)} into ${base(data.db)}…`;
  render();
  try {
    await H.apiTask("properties/extract", { path: data.run, db: data.db, wait: 0 });
    H.toast(`Extracted ${base(data.run)} into ${base(data.db)}`, "success");
    document.dispatchEvent(new CustomEvent("mace-db-changed", { detail: { db: data.db } }));
  } catch (err) {
    if (!(err && (err.state === "canceled" || (err.data && err.data.canceled)))) H.toast(`Extract: ${err.message}`, "error");
  } finally {
    ui.busy = null;
    for (const k of [...ui.cache.keys()]) if (k.startsWith(`${e.path}|`)) ui.cache.delete(k);
    render();
  }
}
async function view(e, name, calc) {
  const { H } = ui;
  let value;
  const rows = e.state === "db" ? e.data.rows : (e.computed && e.computed.rows) || [];
  const r = rows.find((x) => x.name === name && String(x.calc_id || "") === String(calc || ""));
  if (r && r.value != null) value = r.value;
  else if (e.state === "db") {
    try {
      value = (await H.api("properties/value?" + new URLSearchParams({ db: e.data.db, calc, name }))).value;
    } catch (err) { H.toast(`${name}: ${err.message}`, "error"); return; }
  }
  H.showText(`${name} · ${base((e.data && e.data.run) || e.path)}`, JSON.stringify(value, null, 1));
}

/* ---------------------------------------------------------------- Add Reference… */
const CHECK = { ok: "ok", warn: "warning", refuse: "refused" };
/** The picker: for one reference energy, the runs of the database each of its references may be,
 * with whether each may be combined with this run (the server's check: same functional, basis set,
 * CRYSTAL version, D3/gCP; spin and unknown settings warn). Compute and Save runs the run's
 * extraction with --reference NAME=CALC_ID (the command shown), which stores the property and saves
 * the choice with the run; Use the Database's Choice forgets it (NAME=auto). */
async function picker(e, name) {
  const { H } = ui, esc = H.esc, data = e.data || {};
  if (e.state !== "db" || ui.busy) return;
  let info;
  try {
    info = await H.api("properties/references?" + new URLSearchParams({ path: data.run, db: data.db }));
  } catch (err) { H.toast(`Add Reference: ${err.message}`, "error"); return; }
  if (!info.properties || !info.properties.length) { H.toast("No reference energy applies to this run", "info"); return; }
  const st = { prop: info.properties.find((p) => p.name === name) || info.properties[0], all: false, sel: {} };
  const reset = () => {
    st.sel = {};
    for (const r of st.prop.roles) {
      const m = new Map();
      for (const x of (st.prop.saved[r.name] || st.prop.used[r.name] || [])) if (x.calc_id) m.set(x.calc_id, x.weight != null ? String(x.weight) : "");
      st.sel[r.name] = m;
    }
  };
  reset();
  const dl = dialog({ id: "prp-refs", title: "Add Reference", wide: true, body: `<div class="prp-refs"></div>`,
    foot: `<button type="button" class="tb-btn-text" data-r="auto" title="Forget the saved choice: the extraction looks the references up in the database again">Use the Database's Choice</button>
      <span class="prp-grow"></span><button type="button" class="tb-btn-text" data-r="cancel">Cancel</button>
      <button type="button" class="tb-btn-text primary" data-r="save">Compute and Save</button>` });
  const body = dl.querySelector(".prp-refs");
  const args = (clear) => {
    const out = [];
    for (const r of st.prop.roles) {
      if (clear) { out.push(`--reference ${r.name}=auto`); continue; }
      for (const [cid, w] of st.sel[r.name]) out.push(`--reference ${r.name}=${w && r.weighted ? `${w}*` : ""}${cid}`);
    }
    return out;
  };
  const chosen = () => {
    const refs = {};
    for (const r of st.prop.roles) {
      const v = [...st.sel[r.name]].map(([cid, w]) => (w && r.weighted ? `${w}*${cid}` : cid));
      if (v.length) refs[r.name] = v;
    }
    return refs;
  };
  const roleHtml = (r) => {
    const sel = st.sel[r.name];
    const rank = { ok: 0, warn: 1, refuse: 2 };
    const rows = info.candidates.filter((c) => st.all || sel.has(c.calc_id) || c.fits.includes(r.name))
      .sort((a, b) => (sel.has(b.calc_id) - sel.has(a.calc_id)) || rank[a.check] - rank[b.check]);
    const kind = r.many ? "checkbox" : "radio";
    const tr = (c) => {
      const on = sel.has(c.calc_id), off = c.check === "refuse" && !on;
      const why = (c.reasons || []).join("\n");
      return `<tr data-cid="${esc(c.calc_id)}" class="prp-ref-${c.check}">
        <td><input type="${kind}" name="prp-ref-${esc(r.name)}" data-role="${esc(r.name)}" value="${esc(c.calc_id)}"${on ? " checked" : ""}${off ? " disabled" : ""} aria-label="${esc(c.label)}"></td>
        ${r.weighted ? `<td><input type="number" class="field prp-coef" min="0" step="any" placeholder="auto" data-coef="${esc(r.name)}" data-cid="${esc(c.calc_id)}" value="${esc(sel.get(c.calc_id) || "")}" aria-label="Coefficient of ${esc(c.label)}"${on ? "" : " disabled"}></td>` : ""}
        <td class="prp-ref-run" title="${esc(c.label)}">${esc(c.label)}</td><td>${esc(c.calc_type || "")}</td><td>${esc(c.formula)}</td>
        <td>${c.dim == null ? "" : `${c.dim}D`}</td><td>${esc(c.functional || "?")}</td><td>${esc(c.basis_set || "?")}</td>
        <td class="num">${c.energy_au == null ? "" : Number(c.energy_au).toFixed(6)}</td>
        <td class="prp-ref-check" title="${esc(why)}"><span class="prp-st prp-st-${c.check}">${esc(CHECK[c.check])}</span>${why ? ` <span class="insp-hint">${esc(c.reasons[0])}${c.reasons.length > 1 ? ` (+${c.reasons.length - 1})` : ""}</span>` : ""}</td></tr>`;
    };
    return `<fieldset class="prp-ref-role" data-role="${esc(r.name)}"><legend><b>${esc(r.name)}</b>${r.required ? "" : " (optional)"}: ${esc(r.description)}${r.weighted ? "; a blank coefficient: balanced from the compositions" : ""}</legend>
      ${rows.length ? `<table class="an-table prp-ref-table"><thead><tr><th></th>${r.weighted ? "<th>Coefficient</th>" : ""}<th>Run</th><th>Type</th><th>Formula</th><th>Dim.</th><th>Functional</th><th>Basis set</th><th>E (Ha)</th><th>Check</th></tr></thead>
      <tbody>${rows.map(tr).join("")}</tbody></table>` : `<p class="empty-note">No run of the database fits; Show All Runs lists every one.</p>`}</fieldset>`;
  };
  const draw = () => {
    const p = st.prop;
    body.innerHTML = `<div class="prp-refs-top">
        <label>Property <select class="field" data-prop aria-label="Reference energy">${info.properties.map((x) => `<option value="${esc(x.name)}"${x === p ? " selected" : ""}>${esc(x.name)}</option>`).join("")}</select></label>
        <span class="prp-where">${esc(base(info.run))} · ${esc(info.formula || "?")} · ${esc(info.functional || "?")}, ${esc(info.basis_set || "?")}</span>
        <label class="prp-refs-all"><input type="checkbox" data-all${st.all ? " checked" : ""}> Show All Runs</label></div>
      <p class="anr-note">${esc(p.description)} Now: <b class="prp-ref-status">${esc(p.status)}</b>${p.reason ? `: ${esc(p.reason)}` : ""}</p>
      ${p.roles.map(roleHtml).join("")}
      <code class="anr-cli prp-refs-cli">${esc([info.command, ...args(false)].join(" "))}</code>`;
    body.querySelector("[data-prop]").addEventListener("change", (ev) => { st.prop = info.properties.find((x) => x.name === ev.target.value); reset(); draw(); });
    body.querySelector("[data-all]").addEventListener("change", (ev) => { st.all = ev.target.checked; draw(); });
    body.querySelectorAll("input[data-role]").forEach((inp) => inp.addEventListener("change", () => {
      const m = st.sel[inp.dataset.role], role = st.prop.roles.find((x) => x.name === inp.dataset.role);
      if (!role.many) m.clear();
      if (inp.checked) m.set(inp.value, m.get(inp.value) || ""); else m.delete(inp.value);
      draw();
    }));
    body.querySelectorAll("input[data-coef]").forEach((inp) => inp.addEventListener("input", () => {
      const m = st.sel[inp.dataset.coef];
      if (m.has(inp.dataset.cid)) m.set(inp.dataset.cid, inp.value.trim());
      body.querySelector(".prp-refs-cli").textContent = [info.command, ...args(false)].join(" ");
    }));
  };
  draw();
  const run = async (clear) => {
    const refs = chosen();
    if (!clear && !Object.keys(refs).length) { H.toast("Choose at least one run", "info"); return; }
    dl.close();
    ui.busy = clear ? `Looking up the references of ${st.prop.name} again…` : `Computing ${st.prop.name} of ${base(data.run)}…`;
    render();
    const prop = st.prop.name;
    try {
      await H.apiTask("properties/references", { path: data.run, db: data.db, property: prop, references: refs, clear, wait: 0 });
      document.dispatchEvent(new CustomEvent("mace-db-changed", { detail: { db: data.db } }));
      const now = await H.api("properties/run?" + new URLSearchParams({ path: e.path, scope: "run", db: data.db }));
      const d = (now.derived || []).find((x) => x.name === prop) || {};
      if (d.status === "stored") H.toast(`${prop}: computed and stored in ${base(data.db)}`, "success");
      else H.toast(`${prop}: ${d.status || "not stored"}${d.reason ? ` (${d.reason})` : ""}`, "warning");
    } catch (err) {
      if (!(err && (err.state === "canceled" || (err.data && err.data.canceled)))) H.toast(`${prop}: ${err.message}`, "error");
    } finally {
      ui.busy = null;
      for (const k of [...ui.cache.keys()]) if (k.startsWith(`${e.path}|`)) ui.cache.delete(k);
      render();
    }
  };
  dl.querySelector("[data-r=cancel]").addEventListener("click", () => dl.close());
  dl.querySelector("[data-r=save]").addEventListener("click", () => run(false));
  dl.querySelector("[data-r=auto]").addEventListener("click", () => run(true));
}
