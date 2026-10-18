/* Calculation Setup (spec §1.2, plan B1-B4): a panel docked in the inspector,
 * bound to the active document, in three steps (structure, settings, save
 * for the CLI) with a sticky footer; the previewed deck opens as a Deck
 * document beside it. Every field is one of the CLI's own questions: an edit
 * is the answer, the server runs the CLI with it, and the form, the JSON and
 * the deck come back from there. Loaded on the first Calculation Setup. */
import * as M from "./model.js";
import { loadCss } from "../lazy/lazy_util.js";
import * as profile from "../profile.js";
import * as KE from "../kpath_editor.js";
import * as K from "../kpath.js";
import { createBZ } from "../bz_view.js";
import * as CF from "./charge.js";

const css = loadCss(import.meta.url, "../../css/setup.css");
let S = null, U = null;               // the shell API and its UI helpers (esc, icon, toast, dialog)
let root = null, subject = null, schema = null, lastRender = "";
const states = new Map();             // doc id -> setup state
const timers = new Map();
let running = null;                   // {docId, task} of the run in flight
const AUTO_MS = 600;

export async function init(shell, ui) {
  S = shell; U = ui;
  await css;
  if (!schema) {
    try { schema = await S.api("setup/schema"); } catch (e) { schema = { calc_types: {}, functional_categories: [], basis_coverage: {} }; }
  }
  return api;
}

const esc = (s) => U.esc(s);
const icon = (n, c) => U.icon(n, c);
const draftKey = (d) => `mace.gui.setup.${S.dirHash()}.${d.path && !d.untitled ? d.path : "doc" + d.id}`;
const follow = (st) => st && (st.mode === "opt2d12" || st.mode === "opt2d3");
/** convert and opt2d12 take the charged-system switches; opt2d3 takes none */
const chargeable = (st) => !!st && (schema.charge_modes || []).includes(st.mode);
const chargeFlags = (st) => (chargeable(st) && st.flags && Object.keys(st.flags).length ? { ...st.flags } : null);

/* ---------------------------------------------------------------- the subject document */
/** The document Setup is about: the active one, or the one a Deck document was previewed from. */
function subjectDoc() {
  const a = S.activeDoc();
  if (a && a.setupFor) return S.docById(a.setupFor) || null;
  return a;
}

function info(d) {
  return { dirty: S.isDirty(d), frame: S.frameInfo(d) };
}

function stateFor(d) {
  if (!d) return null;
  let st = states.get(d.id);
  if (st) return st;
  const mo = M.modeOf(d, info(d));
  if (!mo.mode) return null;
  try { st = M.fromDraft(JSON.parse(localStorage.getItem(draftKey(d)) || "null")); } catch (e) { st = null; }
  const ct = schema.calc_types || {};
  if (!st || (mo.mode === "cif2d12" && st.mode !== "cif2d12") || (mo.mode === "follow" && st.mode === "cif2d12")
      || (mo.deck && st.mode === "opt2d3")) {
    const mode = mo.mode === "follow" ? "opt2d12" : "cif2d12";
    st = M.newState(mode, (schema.default_calc || {})[mode] || (ct[mode] || ["OPT"])[0]);
  }
  states.set(d.id, st);
  return st;
}

function persist(d, st) {
  try { localStorage.setItem(draftKey(d), JSON.stringify(M.toDraft(st))); } catch (e) { /* full or private */ }
  S.historyChanged();
}

/** The geometry the CLI reads must be the one on screen (plan B4): an edited
 * .out or a run's earlier frame asks which one the deck uses. */
function guardFor(d, st) {
  if (!follow(st)) return null;
  const fi = S.frameInfo(d);
  const dirty = S.isDirty(d);
  if (!dirty && !(fi && fi.frame < fi.n - 1)) return null;
  if (st.guard === "parent" && st.guardFor === guardSig(d)) return null;
  const name = (d.path || d.title).split("/").pop();
  return { dirty, text: dirty ? `${st.mode} uses the final geometry in ${name}, not your edits`
    : `${st.mode} uses the final geometry in ${name}, not frame ${fi.frame + 1} of ${fi.n}` };
}
const guardSig = (d) => { const fi = S.frameInfo(d); return `${S.isDirty(d) ? S.editRev(d) : "clean"}:${fi ? fi.frame : -1}`; };

async function sourceFor(d, st) {
  const mo = M.modeOf(d, info(d));
  if (mo.mode === "follow" || (mo.source === "path" && d.path)) return { source: { path: d.path } };
  const src = await S.sourceOf(d);
  return { source: src, name: (d.saveName || d.title || "structure").replace(/\.[A-Za-z0-9]+$/, "").replace(/[^A-Za-z0-9_.-]+/g, "_") };
}

/* ---------------------------------------------------------------- runs */
function schedule(d, preview = null, delay = null) {
  const st = stateFor(d);
  if (!st) return;
  const auto = S.setting("setupAutoPreview") !== false;
  const want = preview === null ? auto : preview;
  st.stale = true;
  st.pendingRun = true;
  clearTimeout(timers.get(d.id));
  timers.set(d.id, setTimeout(() => { st.pendingRun = false; run(d, want); }, delay === null ? (want ? AUTO_MS : 120) : delay));
  render();
}

async function run(d, preview) {
  const st = stateFor(d);
  if (!st || guardFor(d, st)) return null;
  const { source, name } = await sourceFor(d, st);
  const body = M.request(st, source, d.id, preview);
  if (name) body.name = name;
  // Settings' Functional and Basis set (this session > the folder's mace-folder.toml > yours): the
  // CLI's defaults for a deck without a parent (a follow-up keeps its parent's)
  if (st.mode !== "opt2d3") body.defaults = { functional: S.setting("functional") || "", basis_set: S.setting("basisSet") || "" };
  // the charged-system switches (Charged system, or the defect builder's charge hint), as the CLI takes them
  if (chargeFlags(st)) body.flags = chargeFlags(st);
  const mine = { docId: d.id, t: performance.now(), preview };
  running = mine;
  st.busy = { progress: 0.02, message: "Starting" };
  render();
  performance.mark("setup:run:start");
  try {
    const r = await S.apiTask("setup/run", body, (t) => {
      if (running !== mine) return;
      st.busy = { progress: t.progress || 0, message: t.message || "", id: t.id };
      paintFooter();
    });
    if (running === mine) running = null;
    if (!r || r.canceled) return null;
    st.busy = null;
    st.result = r;
    st.jsonText = r.json_text;
    st.error = null;
    if (st.pos >= 0 && st.history[st.pos]) st.history[st.pos].jsonText = r.json_text;
    if (!st.defaultsSha && !Object.keys(st.answers).length && !Object.keys(st.patches).length && st.hand === null) st.defaultsSha = r.json_sha256;
    if (preview) { st.previewSha = r.json_sha256; st.stale = false; S.setupDeck(d, r, mine.explicit); }
    else st.stale = st.previewSha !== r.json_sha256;
    problems(d, r);
    persist(d, st);
    performance.mark("setup:run:end");
    if (S.setting("runPreflightOnPreview") && preview && r.deck) S.run("calc.preflight");
    return r;
  } catch (e) {
    if (running === mine) running = null;
    if (e.state === "canceled") return null;        // superseded by a newer edit
    st.busy = null;
    st.error = e.message;
    S.clearProblems(d.id, "Setup");
    S.problem({ severity: "error", source: "Setup", doc: d.id, message: e.message });
    return null;
  } finally {
    render();
  }
}

function problems(d, r) {
  S.clearProblems(d.id, "Setup");
  for (const p of r.problems || []) {
    let action = p.field ? { label: "Show", run: () => focusField(p.field) } : null;
    // plan T13: a P1 CIF spglib finds more symmetry in: the symmetrized Save As, never a silent change
    if (p.action_id === "save_symmetrized") action = { label: "Save Symmetrized Copy…", run: () => S.run("file.saveAs", { preset: { cifMode: "symmetrized" } }) };
    S.problem({ ...p, source: "Setup", doc: d.id, action });
  }
  if (r.coverage_warning) S.problem({ severity: "warning", source: "Setup", doc: d.id, message: r.coverage_warning });
  // an FF/ML pre-optimized geometry with a non-OPT calculation: a warning only, the deck is unchanged
  if (r.preopt_warning) S.problem({ severity: "warning", source: "Setup", doc: d.id, message: r.preopt_warning });
}

/* ---------------------------------------------------------------- edits */
function edit(d, field, value, mergeKey = null) {
  const st = stateFor(d);
  const label = M.applyEdit(st, field, value);
  M.record(st, label, mergeKey);
  persist(d, st);
  schedule(d, null, field.kind === "value" ? AUTO_MS : null);
}

function bulk(d, label, fn, toastText) {
  const st = stateFor(d);
  const before = M.snapshot(st, "");
  fn(st);
  M.record(st, label);
  persist(d, st);
  schedule(d);
  if (toastText) S.toast(toastText, "info", { label: "Undo", run: () => { api.undo(); } });
  return before;
}

/* ---------------------------------------------------------------- render */
function el(html) { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstElementChild; }

function render() {
  if (!root || !root.isConnected) return;
  subject = subjectDoc();
  const d = subject;
  const st = stateFor(d);
  const focus = rememberFocus();
  root.innerHTML = "";
  // Expand all / Collapse all once the form has its sections (View › Inspector Sections has the keys)
  const allBtn = (act, cmd, ic, label) => `<button type="button" class="icon-btn sm" data-act="${act}" data-tip="${label}" data-tip-cmd="${cmd}" aria-label="${label}">${icon(ic)}</button>`;
  const acts = st && st.result ? `<span class="su-head-acts" role="group" aria-label="Setup sections">${allBtn("expandAll", "view.expandSections", "chevrons-up-down", "Expand All Sections")}` +
    `${allBtn("collapseAll", "view.collapseSections", "chevrons-down-up", "Collapse All Sections")}</span>` : "";
  root.appendChild(el(`<header class="su-head"><h2 class="su-title">Calculation Setup</h2>${st ? `<span class="su-chip" title="The CLI command that makes this deck">${esc(st.mode)} · ${esc(M.MODES[st.mode])}</span>` : ""}${acts}</header>`));
  const body = el(`<div class="su-body"></div>`);
  root.appendChild(body);
  body.appendChild(step1(d, st));
  if (st) {
    body.appendChild(step2(d, st));
    body.appendChild(step3(d, st));
    root.appendChild(footer(d, st));
  }
  restoreFocus(focus);
  lastRender = performance.now();
}

/* A re-render keeps the focused control focused (by id, or by what it is: a calc
 * button, an action, a section head), so keys like Mod+Z keep reaching the panel. */
const FOCUS_KEYS = ["data-calc", "data-act", "data-reset", "data-hist", "data-soc-reset", "data-soc-use"];
function rememberFocus() {
  const a = document.activeElement;
  const scroll = root.querySelector(".su-body") ? root.querySelector(".su-body").scrollTop : 0;
  if (!a || !root.contains(a)) return { scroll, none: true };
  const attr = FOCUS_KEYS.find((k) => a.hasAttribute(k));
  const sec = a.classList.contains("su-sec-h") ? a.parentElement.dataset.sec : null;
  return { id: a.id, start: a.selectionStart, end: a.selectionEnd, scroll,
    sel: attr ? `[${attr}="${CSS.escape(a.getAttribute(attr))}"]` : sec ? `.su-sec[data-sec="${CSS.escape(sec)}"] > .su-sec-h` : null };
}
function restoreFocus(f) {
  const b = root.querySelector(".su-body");
  if (!f) return;
  if (b) b.scrollTop = f.scroll;
  if (f.none) return;
  const e = (f.id && document.getElementById(f.id)) || (f.sel && root.querySelector(f.sel)) || root.querySelector(".su-calc [aria-checked=true]");
  if (e) { e.focus({ preventScroll: true }); if (f.start !== undefined && e.setSelectionRange) try { e.setSelectionRange(f.start, f.end); } catch (x) { /* not text */ } }
}

function step1(d, st) {
  const s = el(`<section class="su-step" aria-label="Structure"><h3 id="su-s1"><span class="num">1</span>Structure</h3></section>`);
  if (!d) {
    s.appendChild(el(`<div class="su-empty"><p>Open a CIF, a CRYSTAL output (.out) or a .d12 deck to set up a calculation.</p>
      <button class="tb-btn-text" data-cmd="file.open">${icon("folder-open")}<span>Open…</span><kbd>${esc(S.shortcut("file.open"))}</kbd></button></div>`));
    return s;
  }
  const mo = M.modeOf(d, info(d));
  if (!mo.mode) {
    s.appendChild(el(`<div class="su-empty"><p>A deck needs a CIF (a new calculation), a CRYSTAL .out or a .d12 deck in the folder (a follow-up). Save this structure as CIF?</p>
      <button class="tb-btn-text" data-act="saveAsCif">${icon("save")}<span>Save As CIF…</span></button></div>`));
    return s;
  }
  const name = (d.path || d.title || "").split("/").pop(), dir = d.path && d.path.includes("/") ? d.path.slice(0, d.path.lastIndexOf("/") + 1) : "";
  const kind = d.fileKind || "cif";
  const src = el(`<div class="su-src"><span class="kb ${esc(kind)}">${esc(kind === "out" ? "out" : kind.toUpperCase())}</span>
    <span class="su-name" title="${esc(d.path || d.title)}">${esc(name)}</span><span class="muted">${esc(dir)}</span></div>`);
  s.appendChild(src);
  if (follow(st)) s.appendChild(parentLine(d, st, name));
  const meta = (d.meta || (S.structureOf(d) || {}).doc_meta || {});
  const prov = meta.provenance && meta.provenance.label;
  if (prov) s.appendChild(el(`<p class="su-prov">${esc(prov)}</p>`));
  if (mo.source === "session") s.appendChild(el(`<p class="su-note">${icon("info")}<span>Not saved in the folder: the preview uses this structure as a CIF; Save asks to save it first.</span></p>`));
  // the folder's preset for this command (what `mace opt2d12` / `mace opt2d3` there use without --config-file):
  // Setup starts from the CLI's own defaults, and its command names its own file, so the preset is not used here
  const presetKey = `setup.${st.mode}_config`, preset = S.folderCli ? S.folderCli()[presetKey] : null;
  if (preset) {
    s.appendChild(el(`<p class="su-note su-preset">${icon("info")}<span>This folder's mace-folder.toml gives <code>mace ${esc(st.mode)}</code> the settings file ${esc(String(preset))} (<code>${esc(presetKey)}</code>). Calculation Setup does not start from it: it starts from the CLI's defaults, and the command it shows names its own file.</span></p>`));
  }
  const g = guardFor(d, st);
  if (g) {
    const box = el(`<div class="su-guard" role="alert">${icon("triangle-alert")}<div><p>${esc(g.text)}</p>
      <div class="su-guard-acts"><button class="tb-btn-text primary" data-act="guardParent">Use the Unedited Parent</button>
      <button class="tb-btn-text" data-act="guardCif">${st.mode === "opt2d3" ? "Save As CIF (for a new SCF run)" : "Save As CIF and Set Up a New Calculation"}</button></div>
      <p class="muted small">${st.mode === "opt2d3" ? "A properties run needs the parent's wavefunction, so it uses the geometry on disk." : "Save As CIF starts a new calculation (cif2d12); the parent's settings are not carried over."}</p></div></div>`);
    s.appendChild(box);
  } else if (st.guard === "parent") {
    s.appendChild(el(`<p class="su-note">${icon("info")}<span>Using the final geometry in ${esc(name)} (the parent's settings).</span></p>`));
  }
  const charge = meta.hints && meta.hints.charge;
  if (charge) {
    const on = st.flags && st.flags.net_charge === charge;
    s.appendChild(el(`<div class="su-hintline">${icon("zap")}<span>From the defect builder: net charge ${charge > 0 ? "+" : ""}${esc(charge)}</span>
      <button class="tb-btn-text" data-act="${on ? "chargeOff" : "chargeOn"}">${on ? "Applied · Undo" : "Apply"}</button></div>`));
  }
  return s;
}

/** Where a follow-up's values come from: its parent (.out or .d12), and what the CLI said about it. */
function parentLine(d, st, name) {
  const deck = M.parentKind(d) === "d12";
  const b = `<b class="mono">${esc(name)}</b>`;
  const what = deck ? `Geometry and settings from the parent deck ${b}`
    : st.mode === "opt2d3" ? `Geometry and wavefunction from the parent run ${b}`
      : `Geometry from the parent run ${b}, settings from its .d12`;
  const how = deck ? "mace opt2d12 --d12-file" : st.mode === "opt2d3" ? "mace opt2d3 -i" : "mace opt2d12 --out-file";
  const p = st.result && st.result.parent;
  const warn = p && p.warning ? `<p class="su-warnline su-parent-warn" role="note">${icon("triangle-alert")}<span>${esc(p.name)}: ${esc(p.warning)}. The deck is the CLI's (${esc(how)}).</span></p>` : "";
  return el(`<div class="su-parentbox"><p class="su-note su-parent">${icon("info")}<span>${what} (${esc(how)}); fields marked “from parent .${deck ? "d12" : "out"}” keep its values.</span></p>${warn}</div>`);
}

function calcSegment(d, st) {
  // follow-ups this parent cannot have (the server's parent_refusals: ORBITALS from a two-component run),
  // grayed out with the reason as their tooltip; kept across calculation changes (a change clears the result)
  if (st.result && st.result.parent && st.result.parent.refused) st.refused = st.result.parent.refused;
  const refused = st.refused || {};
  // the reason a screen reader reads: visually hidden text after the group, named by aria-describedby
  const whyId = (t) => `su-why-${t.replace(/[^A-Za-z0-9]/g, "_")}`;
  const why = Object.entries(refused).map(([t, r]) => `<span class="sr-only" id="${whyId(t)}">${esc(r)}</span>`).join("");
  const btn = (t, prop) => `<button role="radio" aria-checked="${t === st.calc}" data-calc="${esc(t)}" class="${prop ? "prop" : ""}${refused[t] ? " refused" : ""}"` +
    (refused[t] ? ` aria-disabled="true" title="${esc(refused[t])}" aria-describedby="${whyId(t)}"`
      : `${t === "CHARGE+POTENTIAL" ? ' title="CHARGE+POTENTIAL"' : ""}`) +
    `>${esc(t === "CHARGE+POTENTIAL" ? "CHG+POT" : t)}</button>`;
  if (st.mode === "cif2d12") {
    return `<div class="su-calc" role="radiogroup" aria-label="Calculation type">${(schema.calc_types.cif2d12 || []).map((t) => btn(t, false)).join("")}</div>${why}`;
  }
  // a follow-up: rows of three equal segments, the SCF runs (opt2d12) above the properties of its wavefunction (opt2d3)
  const row = (types, prop, label) => `<div class="su-calc-row" title="${esc(label)}">${types.map((t) => btn(t, prop)).join("")}</div>`;
  if (M.modeOf(d, info(d)).deck) {
    return `<div class="su-calc grouped" role="radiogroup" aria-label="Calculation type">` +
      row(schema.calc_types.opt2d12 || [], false, "A new SCF run from this deck (mace opt2d12 --d12-file); properties need a run's wavefunction") + "</div>" + why;
  }
  const props = schema.calc_types.opt2d3 || [], rows = [];
  for (let k = 0; k < props.length; k += 3) rows.push(row(props.slice(k, k + 3), true, "Properties from this output's wavefunction (mace opt2d3)"));
  return `<div class="su-calc grouped" role="radiogroup" aria-label="Calculation type">` +
    row(schema.calc_types.opt2d12 || [], false, "A new SCF run from this output (mace opt2d12)") + rows.join("") + "</div>" + why;
}

function step2(d, st) {
  const s = el(`<section class="su-step" aria-label="Settings"><h3 id="su-s2"><span class="num">2</span>Settings</h3></section>`);
  s.insertAdjacentHTML("beforeend", calcSegment(d, st));
  const r = st.result;
  const blocked = guardFor(d, st);
  if (!r) {
    s.appendChild(el(`<div class="su-loading" aria-live="polite"><span class="spinner"></span><span>${blocked ? "Choose which geometry the deck uses" : `Asking ${esc(M.MODES[st.mode])} for its defaults…`}</span></div>`));
    return s;
  }
  if (r.save_blocked) {
    s.appendChild(el(`<div class="su-refused" role="alert">${icon("circle-alert")}<p>${esc(r.save_blocked)}</p></div>`));
  }
  // opt2d12 starts from the parent's settings; opt2d3's defaults are its own (Enter-through)
  const fol = st.mode === "opt2d12";
  if (st.mode === "cif2d12" && r.shrink) {
    s.appendChild(el(`<div class="su-ro"><span class="su-ro-l">k-point mesh</span><span class="su-ro-v mono">auto from the cell: SHRINK ${esc(r.shrink)}</span>
      <span class="su-ro-h">convert works the mesh out from each cell; it has no setting for it</span></div>`));
  }
  const collapsed = new Set(profile.ui("setupCollapsed", []) || []);
  for (const [name, rows] of M.sections(r.form)) {
    const open = !collapsed.has(name) && name !== "Other questions" || (name === "Other questions" && !collapsed.has(name) && collapsed.has("__other_open"));
    const sec = el(`<div class="su-sec${open ? "" : " collapsed"}" data-sec="${esc(name)}">
      <button class="su-sec-h" aria-expanded="${open}">${icon("chevron-down")}<span>${esc(name)}</span>
      <span class="su-count">${rows.filter((f) => f.changed).length ? `${rows.filter((f) => f.changed).length} changed` : ""}</span></button><div class="su-rows"></div></div>`);
    const host = sec.querySelector(".su-rows");
    for (const f of rows) host.appendChild(row(d, st, f, fol));
    s.appendChild(sec);
  }
  if (st.mode === "opt2d3" && st.calc === "BAND" && r.kpath) s.appendChild(kpathSection(d, st, r.kpath, collapsed));
  if (chargeable(st)) s.appendChild(chargeSection(st, collapsed));
  const socSec = socSection(d, st, r, collapsed);
  if (socSec) s.appendChild(socSec);
  s.appendChild(advanced(d, st));
  return s;
}

/** The k-path to compute (opt2d3 BAND): the editor of the band plot, on the path the deck computes.
 * An edit is the band_path key of the settings file (what opt2d3's own prompt writes). */
function kpathSection(d, st, kp, collapsed) {
  const open = !collapsed.has("k-path");
  const edited = "band_path" in st.patches;
  const sec = el(`<div class="su-sec${open ? "" : " collapsed"}" data-sec="k-path">
    <button class="su-sec-h" aria-expanded="${open}">${icon("chevron-down")}<span>k-path</span>
    <span class="su-count">${edited ? "edited" : ""}</span></button><div class="su-rows su-kpath"></div></div>`);
  if (!st.bz || st.bzFor !== d.path) {
    if (st.bz) st.bz.destroy();
    st.bzFor = d.path;
    st.bz = createBZ({ api: S.api, path: d.path, pickable: true, onHover: (i) => KE.hoverFromBZ("setup", i),
      onSelect: (i) => KE.selectFromBZ("setup", i), onPick: (lab, p) => KE.pickFromBZ("setup", lab, p) });
  }
  KE.render(sec.querySelector(".su-kpath"), { id: "setup", mode: "calc", heading: false, computed: kp.computed, dimensionality: kp.dimensionality || 3, bz: open ? st.bz : null, segs: edited ? st.patches.band_path : null,
    icon, serverErrors: edited ? kp.errors : [], serverWarnings: edited ? kp.warnings : [],
    onChange: (segs, label) => {
      const before = M.snapshot(st, "");
      // points of the computed path by label, as opt2d3's prompt saves them (the file then fits any
      // material an automatic path fits); points you added keep their coordinates
      const known = K.points(kp.computed);
      const strip = (g) => Object.fromEntries(Object.entries(g).filter(([k, v]) => !(k.startsWith("k_") && known[K.canonical(g[k.slice(2)])]
        && known[K.canonical(g[k.slice(2)])].every((x, n) => Math.abs(x - v[n]) < 1e-6))));
      if (segs) st.patches.band_path = segs.map(strip); else delete st.patches.band_path;
      M.record(st, label);
      persist(d, st);
      schedule(d);
      return before;
    } });
  return sec;
}

/** Charged system (1.1.6): the CLI's charged-system switches (charge.js). Shut until one is given. */
function chargeSection(st, collapsed) {
  const name = CF.SECTION, n = Object.keys(st.flags || {}).length;
  const open = collapsed.has("__charge_open") || (n > 0 && !collapsed.has(name));
  const sec = el(`<div class="su-sec${open ? "" : " collapsed"}" data-sec="${esc(name)}">
    <button class="su-sec-h" aria-expanded="${open}">${icon("chevron-down")}<span>${esc(name)}</span>
    <span class="su-count">${n ? `${n} switch${n === 1 ? "" : "es"}` : ""}</span></button><div class="su-rows su-charge"></div></div>`);
  const host = sec.querySelector(".su-rows");
  host.insertAdjacentHTML("beforeend", `<p class="su-h su-charge-note">The ${esc(st.mode === "cif2d12" ? "mace convert" : "mace opt2d12")} switches for ions, charged cells and odd electron counts; the command below carries them. The CLI checks them and says why when it refuses.</p>`);
  host.insertAdjacentHTML("beforeend", CF.rowsHtml(schema.charge_flags, st.flags, { esc, prefix: "su-cf" }));
  return sec;
}

/* ---------------------------------------------------------------- spin-orbit coupling (1.1.6)
 * The SOC section: keys of the CLI's own settings file (schema.soc.keys), sent as patches, never
 * a CLI question. Enabled for single points (cif2d12, opt2d12); for OPT, FREQ and properties it
 * shows the CLI's own reason. Warnings and the refusal are the CLI's words (problems, field "soc"). */
const SOC_SEC = "Spin-orbit coupling";

function socSection(d, st, r, collapsed) {
  const sc = schema.soc;
  if (!sc) return null;
  const soc = r.soc || { values: {}, base: {}, guesses: [], metal: null, changes: [], refusal: null };
  const base = soc.base || {};
  const applies = (sc.calcs || []).includes(st.calc) && (sc.modes || []).includes(st.mode);
  const on = M.socValue(st, base, "soc") === true;
  const fol = st.mode === "opt2d12", pk = M.parentKind(d);   // as the form rows: opt2d3 starts from its own defaults
  const why = applies ? "" : ((sc.not_for || {})[st.calc] || (sc.not_for || {}).OPT || "");
  const open = !collapsed.has(SOC_SEC);
  const n = M.socChanged(st);
  const sec = el(`<div class="su-sec su-soc${open ? "" : " collapsed"}" data-sec="${SOC_SEC}">
    <button class="su-sec-h" aria-expanded="${open}">${icon("chevron-down")}<span>${SOC_SEC}</span>
    <span class="su-count">${n ? `${n} changed` : on ? "on" : ""}</span></button><div class="su-rows"></div></div>`);
  const host = sec.querySelector(".su-rows");
  const meta = (key, changed, was, label, hintId, hint) => {
    const b = M.badge({ changed, default: was }, fol, pk);
    return `<div class="su-meta"><span class="su-badge ${b.kind}">${esc(b.text)}</span>${changed
      ? `<button type="button" class="su-reset" data-soc-reset="${key}" aria-label="Reset ${esc(label)}">Reset</button>` : ""}
      <span class="su-h" id="${hintId}">${esc(hint || "")}</span></div>`;
  };
  // the switch: off for OPT, FREQ and properties (the CLI's reason), unless the parent's settings have it on
  const canSwitch = applies || on;
  const swHint = applies ? "A two-component SCF (TWOCOMPON), written as \"soc\" in the settings file" : why;
  host.appendChild(el(`<div class="su-row su-soc-row${"soc" in st.patches ? " changed" : ""}" data-soc-row="soc">
    <label for="su-soc-on" id="su-soc-on-l" class="su-l">Spin-orbit coupling</label>
    <div class="su-ctl"><span class="su-switch"><input type="checkbox" role="switch" id="su-soc-on" data-soc="soc" aria-checked="${on}"
      aria-labelledby="su-soc-on-l" aria-describedby="su-soc-on-h"${on ? " checked" : ""}${canSwitch ? "" : ' disabled aria-disabled="true"'}><span aria-hidden="true">${on ? "On" : "Off"}</span></span></div>
    ${meta("soc", "soc" in st.patches, base.soc === true ? "on" : "off", "Spin-orbit coupling", "su-soc-on-h", swHint)}</div>`));
  if (on && applies) {
    host.appendChild(socRoute(d, st, sc, base, meta));
    host.appendChild(socGuess(st, soc, base, meta));
    // the CLI's evidence that the parent is a metal (opt2d12 reads the .out): SMEAR is suggested more strongly
    if (soc.metal) host.appendChild(el(`<p class="su-warnline su-soc-metal" id="su-soc-metal">${icon("triangle-alert")}<span>The parent looks metallic: ${esc(soc.metal)}</span></p>`));
    host.appendChild(socSmear(st, sc, soc, base, meta));
  }
  const notes = socNotes(st, r, soc);
  if (notes) host.appendChild(notes);
  return sec;
}

/** One radio option; a disabled one says why (aria-describedby to its reason, shown under it). */
function socOpt(name, key, value, checked, label, sub, whyId) {
  const id = `su-soc-${name}-${value || "none"}`;
  return `<label class="su-opt${whyId ? " is-off" : ""}" for="${id}"><input type="radio" name="su-soc-${name}" id="${id}" value="${esc(value)}" data-soc="${key}"${checked ? " checked" : ""}${
    whyId ? ` disabled aria-disabled="true" aria-describedby="${whyId}"` : ""}><span><span class="su-opt-name">${esc(label)}</span>${sub ? ` <span class="su-opt-sub">${esc(sub)}</span>` : ""}${whyId ? ' <span class="su-opt-sub">(not available)</span>' : ""}</span></label>`;
}

function socRoute(d, st, sc, base, meta) {
  const route = M.socRoute(st, base);
  // STUTSC is refused for some elements (the CLI's table and words): grayed when the structure has one
  const s = S.structureOf(d);
  const zs = s ? new Set(s.numbers.map((z) => z % 100 || z)) : new Set();
  const refused = Object.entries(sc.stutsc_refused || {}).filter(([z]) => zs.has(Number(z)));
  const sym = (z) => (s && s.symbols[s.numbers.findIndex((n) => (n % 100 || n) === Number(z))]) || `Z=${z}`;
  const fam = (sc.soc_basis_families || [])[0] || "";
  const whyStutsc = refused.length ? `<p class="su-why" id="su-soc-why-stutsc">${refused.map(([z, w]) => `${esc(sym(z))}: ${esc(w)}`).join("<br>")}</p>` : "";
  const changed = "soc_ecp" in st.patches || "soc_basis" in st.patches;
  const was = M.socRouteLabel(base.soc_basis ? "basis" : base.soc_ecp === "stutsc" ? "stutsc" : "inpsoc");
  return el(`<div class="su-row su-soc-row${changed ? " changed" : ""}" data-soc-row="route">
    <span class="su-l" id="su-soc-route-l">Spin-orbit ECP</span>
    <div class="su-ctl su-soc-opts" role="radiogroup" aria-labelledby="su-soc-route-l" aria-describedby="su-soc-route-h">
      ${socOpt("route", "route", "inpsoc", route === "inpsoc", "INPSOC", "the CLI's default: the ECP's SOC terms written into the deck", null)}
      ${socOpt("route", "route", "stutsc", route === "stutsc", "STUTSC", "CRYSTAL's internal Stuttgart SOC ECPs", refused.length ? "su-soc-why-stutsc" : null)}${whyStutsc}
      ${socOpt("route", "route", "basis", route === "basis", "Official SOC basis", fam, null)}
    </div>${meta("route", changed, was, "Spin-orbit ECP", "su-soc-route-h", 'written as "soc_ecp" or "soc_basis"')}</div>`);
}

function socGuess(st, soc, base, meta) {
  const cur = M.socValue(st, base, "soc_guess") || "";
  const gs = soc.guesses && soc.guesses.length ? soc.guesses : [{ value: null, ok: true }];
  const SUB = { guesspnoso: "from the parent's spin-polarized run without symmetry", guesspso: "from the parent's two-component run" };
  // one reason shown once for the options it disables (a CIF has no parent for either)
  const reasons = [];
  for (const g of gs) if (!g.ok && g.reason && !reasons.includes(g.reason)) reasons.push(g.reason);
  const rid = (g) => (g.ok ? null : `su-soc-why-guess-${reasons.indexOf(g.reason)}`);
  let html = "";
  gs.forEach((g, i) => {
    html += socOpt("guess", "soc_guess", g.value || "", (g.value || "") === cur, M.socGuessLabel(g.value).replace(/^from scratch$/, "From scratch"), g.value ? SUB[g.value] : "no restart", rid(g));
    const k = reasons.indexOf(g.reason);
    if (!g.ok && g.reason && !gs.slice(i + 1).some((x) => !x.ok && x.reason === g.reason)) html += `<p class="su-why" id="su-soc-why-guess-${k}">${esc(g.reason)}</p>`;
  });
  const changed = "soc_guess" in st.patches;
  return el(`<div class="su-row su-soc-row${changed ? " changed" : ""}" data-soc-row="soc_guess">
    <span class="su-l" id="su-soc-guess-l">Starting guess</span>
    <div class="su-ctl su-soc-opts" role="radiogroup" aria-labelledby="su-soc-guess-l" aria-describedby="su-soc-guess-h">${html}</div>
    ${meta("soc_guess", changed, M.socGuessLabel(base.soc_guess), "Starting guess", "su-soc-guess-h", 'written as "soc_guess"; the CLI checks the parent')}</div>`);
}

function socSmear(st, sc, soc, base, meta) {
  const v = M.socValue(st, base, "smear");
  const sug = sc.metal_smear;
  const changed = "smear" in st.patches;
  const use = sug !== undefined && String(v) !== String(sug)
    ? `<button type="button" class="tb-btn-text sm${soc.metal ? " primary" : ""}" data-soc-use="smear" aria-describedby="su-soc-smear-h">Use ${esc(sug)}</button>` : "";
  return el(`<div class="su-row su-soc-row${changed ? " changed" : ""}" data-soc-row="smear">
    <label for="su-soc-smear" id="su-soc-smear-l" class="su-l">SMEAR for metals</label>
    <div class="su-ctl"><input id="su-soc-smear" class="mono" data-soc="smear" inputmode="decimal" placeholder="none" spellcheck="false" autocomplete="off"
      value="${esc(v ?? "")}" aria-describedby="${soc.metal ? "su-soc-metal " : ""}su-soc-smear-h"><span class="su-unit">Ha</span>${use}</div>
    ${meta("smear", changed, base.smear ?? "none", "SMEAR", "su-soc-smear-h",
    `empty: no SMEAR (the CLI never adds it by itself); ${sug} Ha converged fcc Au at SHRINK 10 10`)}</div>`);
}

/** The CLI's SOC warnings and refusal (problems on field "soc"), and its "SOC deck:" lines, verbatim. */
function socNotes(st, r, soc) {
  const probs = (r.problems || []).filter((p) => p.field === "soc");
  const changes = soc.changes || [];
  if (!probs.length && !changes.length) return null;
  const box = el(`<div class="su-soc-notes" id="su-soc-notes"></div>`);
  for (const p of probs) {
    box.appendChild(p.severity === "error"
      ? el(`<div class="su-refused su-soc-refusal">${icon("circle-alert")}<p>${esc(p.message)}</p></div>`)
      : el(`<p class="su-warnline su-soc-warn">${icon("triangle-alert")}<span>${esc(p.message)}</span></p>`));
  }
  if (changes.length) {
    box.appendChild(el(`<div class="su-soc-changes"><h4 id="su-soc-changes-h">What the CLI changed</h4>
      <ul class="mono" aria-labelledby="su-soc-changes-h">${changes.map((c) => `<li>${esc(c)}</li>`).join("")}</ul></div>`));
  }
  return box;
}

function socChange(d, st, t) {
  const r = st.result;
  if (!r) return;
  const base = (r.soc || {}).base || {};
  const k = t.dataset.soc;
  let label;
  if (k === "soc") {
    t.setAttribute("aria-checked", String(t.checked));
    M.socSwitch(st, base, t.checked);
    label = `Spin-orbit coupling ${t.checked ? "off → on" : "on → off"}`;
  } else if (k === "route") {
    const was = M.socRoute(st, base);
    M.socSetRoute(st, base, t.value, (schema.soc.soc_basis_families || [])[0]);
    label = `Spin-orbit ECP ${M.socRouteLabel(was)} → ${M.socRouteLabel(t.value)}`;
  } else if (k === "soc_guess") {
    const was = M.socValue(st, base, "soc_guess");
    M.socSet(st, base, "soc_guess", t.value || undefined);
    label = `Starting guess ${M.socGuessLabel(was)} → ${M.socGuessLabel(t.value)}`;
  } else if (k === "smear") {
    const raw = t.value.trim(), num = Number(raw);
    // a number goes as a number; anything else as typed, for the CLI's own check to answer
    const v = raw === "" ? undefined : Number.isFinite(num) ? num : raw;
    const was = M.socValue(st, base, "smear");
    if (String(was ?? "") === String(v ?? "")) return;
    M.socSet(st, base, "smear", v);
    label = `SMEAR ${was ?? "none"} → ${v ?? "none"}`;
  } else return;
  M.record(st, label);
  persist(d, st);
  schedule(d);
}

const fid = (f) => "su-f-" + f.id.replace(/[^A-Za-z0-9_-]/g, "_").slice(0, 60);

function row(d, st, f, fol) {
  const id = fid(f);
  let b = M.badge(f, fol, M.parentKind(d));
  // a default that comes from Settings (the folder's file, yours, or this session) says so
  const sk = { functional: "functional", basis_set: "basisSet" }[f.id];
  if (sk && !fol && !f.changed && S.settingSource && S.setting(sk)) {
    const where = { folder: "this folder", user: "your settings", session: "this session" }[S.settingSource(sk)];
    if (where) b = { kind: "default", text: `default (${where})` };
  }
  const pending = f.id in st.answers && String(st.answers[f.id]) !== String(f.value) && f.kind !== "yesno";
  let ctl;
  if (f.kind === "menu") {
    const opts = f.options || [];
    const cur = f.id in st.answers ? String(st.answers[f.id]) : f.value;
    const groups = new Map();
    for (const o of opts) { const g = o.group || ""; if (!groups.has(g)) groups.set(g, []); groups.get(g).push(o); }
    const optHtml = [...groups.entries()].map(([g, os]) => {
      const inner = os.map((o) => `<option value="${esc(o.head)}"${o.head === cur ? " selected" : ""} title="${esc(o.text)}">${esc(o.head)}</option>`).join("");
      return g ? `<optgroup label="${esc(g)}">${inner}</optgroup>` : inner;
    }).join("");
    const missing = cur && !opts.some((o) => o.head === cur) ? `<option value="${esc(cur)}" selected>${esc(cur)}</option>` : "";
    ctl = `<select id="${id}" aria-describedby="${id}-h">${missing}${optHtml}</select>`;
  } else if (f.kind === "yesno") {
    const on = f.id in st.answers ? [true, "y", "Yes"].includes(st.answers[f.id]) : f.value === "Yes";
    // one label per control (WCAG 1.3.1): the row's own; the Yes/No text is visual only
    ctl = `<span class="su-switch"><input type="checkbox" role="switch" id="${id}" aria-checked="${on}" aria-labelledby="${id}-l" aria-describedby="${id}-h"${on ? " checked" : ""}><span aria-hidden="true">${on ? "Yes" : "No"}</span></span>`;
  } else {
    const v = f.id in st.answers ? st.answers[f.id] : f.id in st.patches ? st.patches[f.id] : (f.value ?? "");
    ctl = `<input id="${id}" class="mono" value="${esc(v)}" aria-describedby="${id}-h" spellcheck="false" autocomplete="off"${f.how === "patch" ? ' placeholder="the parent\'s mesh"' : ""}>`;
  }
  const cover = f.id === "basis_set" ? coverage(d, f) : "";
  // the CLI's own question is the label's tooltip; the hint line says what the field does, when that needs saying
  const q = f.question ? `Asked by ${M.MODES[st.mode]}: ${f.question.replace(/\s*\[[^\]]*\]\s*:?\s*$|\s*\(default[^)]*\)\s*:?\s*$|:\s*$/, "")}` : "";
  const r = el(`<div class="su-row${f.changed ? " changed" : ""}${pending ? " pending" : ""}" data-field="${esc(f.id)}">
    <label for="${id}" id="${id}-l" class="su-l" title="${esc(q)}">${esc(f.label)}</label>
    <div class="su-ctl">${ctl}${f.unit ? `<span class="su-unit">${esc(f.unit)}</span>` : ""}${cover}</div>
    <div class="su-meta"><span class="su-badge ${b.kind}">${esc(b.text)}</span>${f.changed ? `<button class="su-reset" data-reset="${esc(f.id)}" aria-label="Reset ${esc(f.label)}">Reset</button>` : ""}
    <span class="su-h" id="${id}-h">${esc(f.hint || "")}</span></div></div>`);
  return r;
}

/** The basis set's element coverage (the CLI's own tables): "def2-mSVP has no Pb". */
function coverage(d, f) {
  const s = S.structureOf(d);
  const cov = (schema.basis_coverage || {})[f.value];
  if (!s || !cov) return "";
  const missing = [...new Set(s.numbers.map((z) => z % 100 || z))].filter((z) => !cov.includes(z));
  if (!missing.length) return `<span class="su-cover ok" title="${esc(f.value)} covers every element here">${icon("check")}</span>`;
  const names = missing.map((z) => (s.symbols[s.numbers.findIndex((n) => (n % 100 || n) === z)] || z)).join(", ");
  return `<span class="su-cover bad" title="${esc(f.value)} has no ${esc(names)}" role="img" aria-label="${esc(f.value)} has no ${esc(names)}">${icon("triangle-alert")}</span>`;
}

function advanced(d, st) {
  const r = st.result;
  const open = !!st.advancedOpen;
  const sec = el(`<div class="su-sec${open ? "" : " collapsed"}" data-sec="Advanced">
    <button class="su-sec-h" aria-expanded="${open}">${icon("chevron-down")}<span>Advanced: the settings file</span>
    <span class="su-count">${st.hand !== null ? "hand-edited" : ""}</span></button><div class="su-rows"></div></div>`);
  const host = sec.querySelector(".su-rows");
  if (open) {
    host.appendChild(el(`<p class="su-h">The file the CLI saved (${esc(r && r.json_rel || "")}). Edit it by hand to set what the form has no field for; it is sent as you wrote it.</p>`));
    const ta = el(`<textarea id="su-json" class="mono" spellcheck="false" rows="14" aria-label="Settings JSON"></textarea>`);
    ta.value = st.hand !== null ? st.hand : (st.jsonText || "");
    host.appendChild(ta);
    host.appendChild(el(`<div class="su-json-acts"><button class="tb-btn-text" data-act="applyJson">Use This JSON</button>
      <span class="su-json-state muted small" id="su-json-state">${st.hand !== null ? "Hand-edited: the form's changes replace it" : "As the CLI saved it"}</span></div>`));
    if (r && r.defaults_json && r.json_text && r.defaults_json !== r.json_text) {
      const dl = M.diffLines(r.defaults_json, r.json_text).filter((x) => x.op !== " ");
      host.appendChild(el(`<details class="su-diff"><summary>${dl.length} line${dl.length === 1 ? "" : "s"} differ from the ${st.mode === "opt2d12" ? "parent's" : "defaults'"} file</summary>
        <pre class="mono">${dl.map((x) => `<span class="${x.op === "+" ? "add" : "del"}">${esc(x.op + " " + x.text)}</span>`).join("\n")}</pre></details>`));
    }
  }
  return sec;
}

function step3(d, st) {
  const r = st.result;
  const open = !!st.saveOpen || !!st.saved;
  const s = el(`<section class="su-step${open ? "" : " folded"}" aria-label="Save for the CLI"><h3 id="su-s3"><span class="num">3</span>Save for the CLI</h3></section>`);
  if (!open) return s;
  const mo = M.modeOf(d, info(d));
  if (mo.source === "session") {
    s.appendChild(el(`<div class="su-empty"><p>Save the structure as CIF first: the command reads a file in the folder.</p>
      <button class="tb-btn-text primary" data-act="saveAsCif">${icon("save")}<span>Save As CIF…</span></button></div>`));
    return s;
  }
  if (r && r.save_blocked) { s.appendChild(el(`<p class="su-note">${icon("info")}<span>${esc(r.save_blocked)}</span></p>`)); return s; }
  const layout = st.layout || (st.mode === "cif2d12" ? "own" : "beside");
  const others = st.mode === "cif2d12" ? S.cifCount(d.path) - 1 : 0;
  if (st.mode === "cif2d12") {
    const dir = d.path.includes("/") ? d.path.slice(0, d.path.lastIndexOf("/") + 1) : "./";
    const stem = d.path.split("/").pop().replace(/\.cif$/i, "");
    s.appendChild(el(`<div class="su-layout" role="radiogroup" aria-label="Where the settings go">
      <label class="su-radio"><input type="radio" name="su-layout" value="own"${layout === "own" ? " checked" : ""}><span><b>Own folder</b><br><span class="muted small">setup/${esc(stem)}/: a copy of the CIF and its settings</span></span></label>
      <label class="su-radio"><input type="radio" name="su-layout" value="inplace"${layout === "inplace" ? " checked" : ""}><span><b>In place</b><br><span class="muted small">${others > 0 ? `converts all ${others + 1} CIFs in ${esc(dir)}` : `beside ${esc(d.path.split("/").pop())}`}</span></span></label></div>`));
  }
  const def = st.mode === "cif2d12" ? `${d.path.split("/").pop().replace(/\.cif$/i, "")}_options.json`
    : st.mode === "opt2d12" ? `${d.path.split("/").pop().replace(/\.[^.]+$/, "")}_${st.calc}_options.json`
      : `${d.path.split("/").pop().replace(/\.[^.]+$/, "")}_${st.calc.replace("+", "_")}_d3.json`;
  const pref = S.setting("optionsFileName");
  s.appendChild(el(`<div class="su-saverow"><label for="su-name" class="su-l">File</label><input id="su-name" class="mono" value="${esc(st.saveName || (pref && pref !== "gui_options.json" ? pref : def))}" spellcheck="false">
    <label class="su-check"><input type="checkbox" id="su-over"${st.overwrite ? " checked" : ""}> Overwrite</label></div>`));
  s.appendChild(el(`<div class="su-saveacts"><button class="tb-btn-text primary" data-act="save"${!r || !r.json_text ? " disabled" : ""}>${icon("save")}<span>Save Settings</span><kbd>${esc(S.shortcut("calc.saveJson"))}</kbd></button>
    <label class="su-check" title="Run the printed command now, in the folder: it writes the decks${st.mode === "opt2d3" ? " and copies the wavefunction" : ""}"><input type="checkbox" id="su-write"${st.writeDecks ? " checked" : ""}> Write decks now</label></div>`));
  if (st.mode === "opt2d3" && r && r.f9 && r.f9.live) {
    // the run is still going: its .f9 is copied when it ends (PLAN_analysis C18.7)
    const w = s.querySelector("#su-write");
    if (w) { w.checked = false; w.disabled = true; st.writeDecks = false; }
    s.appendChild(el(`<p class="su-warnline">${icon("triangle-alert")}<span>The run is still going; its wavefunction (.f9) is copied when the run ends.</span></p>`));
  } else if (st.mode === "opt2d3" && r && r.f9 && st.writeDecks) {
    s.appendChild(el(`<p class="su-warnline">${icon("triangle-alert")}<span>This copies ${esc(r.f9.name)} (${esc(mb(r.f9.bytes))}) to ${esc(r.deck_name ? r.deck_name.replace(/\.d3$/, ".f9") : "the deck's name")}.</span></p>`));
  }
  if (st.saved) {
    const sv = st.saved;
    s.appendChild(el(`<div class="su-saved" role="status">${icon("circle-check")}<div class="su-saved-b">
      <div class="su-saved-path"><span class="mono">${esc(sv.json)}</span>${sv.stale ? '<span class="su-badge changed">settings changed since</span>' : ""}</div>
      <div class="su-cmd"><code class="mono" id="su-cmd">${esc(sv.command)}</code><button class="icon-btn sm" data-act="copyCmd" aria-label="Copy command">${icon("copy")}</button></div>
      ${sv.written && sv.written.length ? `<div class="muted small">Wrote ${sv.written.map(esc).join(", ")}</div>` : ""}
      ${sv.backups && sv.backups.length ? `<div class="muted small">Kept the old ${sv.backups.length === 1 ? "deck" : "decks"} as ${sv.backups.map(esc).join(", ")}</div>` : ""}
      ${sv.warning ? `<div class="su-warnline">${icon("triangle-alert")}<span>${esc(sv.warning)}</span></div>` : ""}
      <div class="muted small">then: <code class="mono">mace submit ${esc((sv.written && sv.written[0]) || (r && r.deck_name) || "<deck>")}</code></div>
      <div class="su-saved-acts"><button class="tb-btn-text" data-act="reveal">${icon("files")}<span>Reveal in Explorer</span></button>${
        sv.written && sv.written.some((x) => /\.(d12|d3)$/.test(x)) ? `<button class="tb-btn-text" data-act="submitDecks">${icon("play")}<span>Submit…</span></button>` : ""}</div></div></div>`));
  }
  return s;
}

function mb(n) { return n >= 2 ** 20 ? `${(n / 2 ** 20).toFixed(0)} MB` : n >= 1024 ? `${(n / 1024).toFixed(0)} KB` : `${n} B`; }

function footer(d, st) {
  const f = el(`<footer class="su-foot"><div class="su-foot-state" id="su-state" aria-live="polite"></div>
    <div class="su-foot-acts">
      <label class="su-live" title="Preview the deck while editing (Settings › Calculation)"><input type="checkbox" id="su-auto"${S.setting("setupAutoPreview") !== false ? " checked" : ""}><span>Live</span></label>
      <button class="tb-btn-text" data-act="saveOpen">${icon("save")}<span>Save…</span></button>
      <button class="tb-btn-text primary" data-act="preview" data-cmd-hint="calc.preview">${icon("file-text")}<span>Preview</span><kbd>${esc(S.shortcut("calc.preview"))}</kbd></button>
    </div>
    <div class="su-foot-next" role="group" aria-label="Then">
      <button class="tb-btn-text" data-act="submitNow" title="Write this deck (Save with Write decks now), then open Submit on it: you confirm there before anything is submitted">${icon("play")}<span>Submit…</span></button>
      <button class="tb-btn-text" data-act="batchNow" title="Batch Setup on this folder: these settings for every file like this one, written, then Submit">${icon("list-checks")}<span>Batch Prepare + Submit…</span></button>
    </div></footer>`);
  setTimeout(paintFooter, 0);
  return f;
}

function paintFooter() {
  const box = root && root.querySelector("#su-state");
  if (!box) return;
  const d = subject, st = stateFor(d);
  if (!st) { box.innerHTML = ""; return; }
  const g = guardFor(d, st);
  const pv = root.querySelector('[data-act="preview"]'), sv = root.querySelector('[data-act="saveOpen"]');
  if (pv) { pv.disabled = !!g; pv.title = g ? "Choose which geometry the deck uses" : ""; }
  if (sv) { sv.disabled = !!g || !st.result || !!(st.result && st.result.save_blocked); sv.title = g ? "Choose which geometry the deck uses" : (st.result && st.result.save_blocked) || ""; }
  const sn = root.querySelector('[data-act="submitNow"]'), bn = root.querySelector('[data-act="batchNow"]');
  if (sn) sn.disabled = !!g || !st.result || !st.result.json_text || !!st.result.save_blocked || !!st.busy;
  if (bn) bn.disabled = !!g || !st.result || !!st.busy;
  let html;
  if (g) html = `<span class="warn">${icon("triangle-alert")}Choose which geometry the deck uses</span>`;
  else if (st.busy) {
    const pct = Math.round((st.busy.progress || 0) * 100);
    html = `<span class="spinner sm"></span><span class="su-busy-t">${esc(st.busy.message || "Running")}</span>` +
      `<span class="st-task-bar" role="progressbar" aria-label="Deck preview" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100"><span style="width:${pct}%"></span></span>` +
      (st.busy.id ? `<button class="su-cancel" data-act="cancel">Cancel</button>` : "");
  } else if (st.error) html = `<span class="bad">${icon("circle-alert")}${esc(st.error.split("\n")[0].slice(0, 160))}</span>`;
  else if (st.result && st.result.save_blocked) html = `<span class="bad">${icon("circle-alert")}The CLI wrote no deck</span>`;
  else if (st.result && st.previewSha === st.result.json_sha256 && !st.stale) html = `<span class="ok">${icon("circle-check")}Deck matches these settings</span>`;
  else if (st.result) html = `<span class="warn">${icon("circle-dot")}Preview is out of date</span>`;
  else html = "";
  box.innerHTML = html;
}

/* ---------------------------------------------------------------- events */
/* A section's open or shut state, as remembered in the profile (ui.setupCollapsed):
 * "Other questions" starts shut, so it is remembered when open (__other_open). */
function rememberSection(c, name, open) {
  if (name === CF.SECTION) { if (open) { c.add("__charge_open"); c.delete(name); } else { c.delete("__charge_open"); c.add(name); } return; }
  if (name === "Other questions") { if (open) { c.add("__other_open"); c.delete(name); } else c.delete("__other_open"); }
  else if (open) c.delete(name); else c.add(name);
}

function onClick(e) {
  const d = subject, st = stateFor(d);
  const t = e.target;
  const sec = t.closest(".su-sec-h");
  if (sec) {
    const box = sec.parentElement, name = box.dataset.sec;
    const nowOpen = box.classList.contains("collapsed");
    // Alt- or Shift-click opens or shuts every section the way this one goes
    if (e.altKey || e.shiftKey) { api.setAllSections(nowOpen); return; }
    if (name === "Advanced") { st.advancedOpen = !st.advancedOpen; render(); return; }
    const c = new Set(profile.ui("setupCollapsed", []) || []);
    rememberSection(c, name, nowOpen);
    profile.setUi("setupCollapsed", [...c]);
    box.classList.toggle("collapsed", !nowOpen);
    sec.setAttribute("aria-expanded", String(nowOpen));
    return;
  }
  const calc = t.closest("[data-calc]");
  if (calc && st) {
    const c = calc.dataset.calc;
    if (c === st.calc || calc.getAttribute("aria-disabled") === "true") return;
    const was = st.calc;
    st.calc = c;
    if (st.mode !== "cif2d12") st.mode = M.followMode(c, schema.calc_types);
    st.answers = {}; st.patches = {}; st.hand = null; st.result = null;
    M.record(st, `Calculation ${was} → ${c}`);
    persist(d, st);
    schedule(d, null, 0);
    return;
  }
  const socReset = t.closest("[data-soc-reset]");
  if (socReset && st && st.result) {
    const k = socReset.dataset.socReset;
    const LABEL = { soc: "Spin-orbit coupling", route: "Spin-orbit ECP", soc_guess: "Starting guess", smear: "SMEAR" };
    if (k === "soc") M.socReset(st);
    else for (const x of k === "route" ? ["soc_ecp", "soc_basis"] : [k]) delete st.patches[x];
    st.hand = null;
    M.record(st, `Reset ${LABEL[k] || k}`);
    persist(d, st);
    schedule(d);
    return;
  }
  const socUse = t.closest("[data-soc-use]");
  if (socUse && st && st.result) {
    const base = (st.result.soc || {}).base || {};
    const was = M.socValue(st, base, "smear");
    M.socSet(st, base, "smear", schema.soc.metal_smear);
    M.record(st, `SMEAR ${was ?? "none"} → ${schema.soc.metal_smear}`);
    persist(d, st);
    schedule(d);
    return;
  }
  const reset = t.closest("[data-reset]");
  if (reset && st) {
    const f = (st.result.form || []).find((x) => x.id === reset.dataset.reset);
    M.resetField(st, reset.dataset.reset);
    M.record(st, `Reset ${f ? f.label : reset.dataset.reset}`);
    persist(d, st);
    schedule(d);
    return;
  }
  const act = t.closest("[data-act]");
  if (!act) return;
  const a = act.dataset.act;
  if (a === "expandAll" || a === "collapseAll") api.setAllSections(a === "expandAll");
  else if (a === "preview") api.preview(true);
  else if (a === "cancel" && st.busy && st.busy.id) S.cancelTask(st.busy.id);
  else if (a === "saveOpen") { st.saveOpen = true; render(); const n = root.querySelector("#su-name"); if (n) n.focus(); }
  else if (a === "save") api.save();
  else if (a === "copyCmd") api.copyCli();
  else if (a === "reveal" && st.saved) S.reveal(st.saved.json);
  else if (a === "submitDecks" && st.saved) S.run("calc.submit", { paths: (st.saved.written || []).filter((x) => /\.(d12|d3)$/.test(x)) });
  else if (a === "submitNow") api.submitDeck();
  else if (a === "batchNow") api.batchFrom();
  else if (a === "guardParent") { st.guard = "parent"; st.guardFor = guardSig(d); S.showParentGeometry(d); persist(d, st); render(); schedule(d, null, 0); }
  else if (a === "guardCif" || a === "saveAsCif") S.saveAsCifForSetup(d);
  else if (a === "applyJson") {
    const ta = root.querySelector("#su-json");
    try { JSON.parse(ta.value); } catch (x) { S.toast(`Not valid JSON: ${x.message}`, "error"); return; }
    bulk(d, "Edit the settings file by hand", (s) => { s.hand = ta.value; });
  } else if (a === "chargeOn" || a === "chargeOff") {
    const meta = d.meta || (S.structureOf(d) || {}).doc_meta || {};
    bulk(d, a === "chargeOn" ? `Net charge ${meta.hints.charge}` : "Net charge from the defect builder removed",
      (s) => { s.flags = CF.withValue(s.flags, "net_charge", a === "chargeOn" ? meta.hints.charge : undefined); },
      a === "chargeOn" ? `Applied the net charge ${meta.hints.charge}` : null);
  } else if (act.dataset.cmd) S.run(act.dataset.cmd);
}

function onChange(e) {
  const d = subject, st = stateFor(d);
  const t = e.target;
  if (t.id === "su-auto") { S.set("setupAutoPreview", t.checked); if (t.checked && st && st.stale) schedule(d, true, 0); return; }
  if (t.name === "su-layout") { st.layout = t.value; persist(d, st); render(); return; }
  if (t.id === "su-name") { st.saveName = t.value.trim(); return; }
  if (t.id === "su-over") { st.overwrite = t.checked; return; }
  if (t.id === "su-write") { st.writeDecks = t.checked; render(); return; }
  if (t.dataset && t.dataset.charge && st) {
    // a charged-system switch: given, changed or removed (one history entry, one CLI run)
    const def = (schema.charge_flags || []).find((x) => x.id === t.dataset.charge);
    if (!def) return;
    const v = CF.valueOf(def, t);
    if (String(v ?? "") === String((st.flags || {})[def.id] ?? "")) return;
    bulk(d, v === undefined ? `${def.switch} removed` : `${def.switch} ${v}`, (s) => { s.flags = CF.withValue(s.flags, def.id, v); });
    return;
  }
  if (t.dataset && t.dataset.soc && st) { socChange(d, st, t); return; }
  const r = t.closest(".su-row");
  if (!r || !st || !st.result) return;
  const f = st.result.form.find((x) => x.id === r.dataset.field);
  if (!f) return;
  if (f.kind === "yesno") { t.setAttribute("aria-checked", String(t.checked)); edit(d, f, t.checked); }
  else if (f.kind === "menu") edit(d, f, t.value);
  // typing already applied this value (onInput) and scheduled its run: Chromium fires a change when
  // the form is drawn again under the focused field, which must not start a second CLI run
  else if (t.dataset.applied === undefined || t.dataset.applied !== t.value.trim()) edit(d, f, t.value.trim(), `field:${f.id}`);
}

function onInput(e) {
  const t = e.target;
  const r = t.closest && t.closest(".su-row");
  if (!r || t.tagName !== "INPUT" || t.type === "checkbox" || t.type === "radio" || t.dataset.soc) return;
  const d = subject, st = stateFor(d);
  const f = st && st.result && st.result.form.find((x) => x.id === r.dataset.field);
  if (!f) return;
  // typing: one history entry per field within 1 s, one CLI run 600 ms after the last key
  const label = M.applyEdit(st, f, t.value.trim() === "" ? null : t.value.trim());
  t.dataset.applied = t.value.trim();
  M.record(st, label, `field:${f.id}`);
  persist(d, st);
  r.classList.add("pending");
  st.stale = true;
  st.pendingRun = true;
  clearTimeout(timers.get(d.id));
  timers.set(d.id, setTimeout(() => { st.pendingRun = false; run(d, S.setting("setupAutoPreview") !== false); }, AUTO_MS));
  paintFooter();
}

function onKey(e) {
  const mod = e.ctrlKey || e.metaKey;
  const inText = ["INPUT", "TEXTAREA"].includes(e.target.tagName) && !["checkbox", "radio"].includes(e.target.type);
  if (mod && !e.altKey && e.key.toLowerCase() === "z" && !inText) {
    e.preventDefault(); e.stopPropagation();
    if (e.target.closest && e.target.closest(".su-kpath")) KE.keepFocus("setup");
    if (e.shiftKey) api.redo(); else api.undo();
  } else if (e.key === "Escape" && !e.defaultPrevented) {
    S.focusRegion("viewport");
  } else if (e.target.matches && e.target.matches("[data-calc]") && ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(e.key)) {
    // the calculation types: one radio group, arrows move along it (Enter or Space picks, as a click: each pick is a CLI run)
    const all = [...root.querySelectorAll("[data-calc]")], i = all.indexOf(e.target);
    const j = e.key === "Home" ? 0 : e.key === "End" ? all.length - 1 : i + (["ArrowLeft", "ArrowUp"].includes(e.key) ? -1 : 1);
    if (all[(j + all.length) % all.length]) { e.preventDefault(); all[(j + all.length) % all.length].focus(); }
  }
}

function focusField(id) {
  if (id === "soc" || (schema.soc && (schema.soc.keys || []).includes(id))) {
    // the SOC section: opened, and its first control that can be used (else its heading)
    const sec = root && root.querySelector(`.su-sec[data-sec="${SOC_SEC}"]`);
    if (!sec) return;
    if (sec.classList.contains("collapsed")) sec.querySelector(".su-sec-h").click();
    sec.scrollIntoView({ block: "nearest" });
    const c = sec.querySelector("input:not([disabled])") || sec.querySelector(".su-sec-h");
    if (c) c.focus();
    return;
  }
  const r = root && root.querySelector(`.su-row[data-field="${CSS.escape(id)}"]`);
  if (!r) return;
  const sec = r.closest(".su-sec");
  if (sec && sec.classList.contains("collapsed")) sec.querySelector(".su-sec-h").click();
  r.scrollIntoView({ block: "nearest" });
  const c = r.querySelector("select, input");
  if (c) c.focus();
}

/* ---------------------------------------------------------------- the API the shell uses */
const api = {
  /** Expand or collapse every section of the form (Advanced only collapses: it holds the hand-edited file). */
  setAllSections(open) {
    if (!root || !root.isConnected) return;
    const c = new Set(profile.ui("setupCollapsed", []) || []);
    for (const box of root.querySelectorAll(".su-sec")) {
      if (box.dataset.sec === "Advanced") continue;
      rememberSection(c, box.dataset.sec, open);
      box.classList.toggle("collapsed", !open);
      box.querySelector(".su-sec-h").setAttribute("aria-expanded", String(open));
    }
    profile.setUi("setupCollapsed", [...c]);
    const st = stateFor(subject);
    if (!open && st && st.advancedOpen) { st.advancedOpen = false; render(); }
  },
  mount(host) {
    if (!root) {
      root = document.createElement("div");
      root.className = "setup";
      root.setAttribute("role", "region");
      root.setAttribute("aria-label", "Calculation Setup");
      root.addEventListener("click", onClick);
      root.addEventListener("change", onChange);
      root.addEventListener("input", onInput);
      root.addEventListener("keydown", onKey);
    }
    host.appendChild(root);
    this.refresh();
  },
  /** The active document changed (or its geometry): show its setup, asking the CLI once. */
  refresh() {
    if (!root || !root.isConnected) return;
    const d = subjectDoc();
    const st = stateFor(d);
    subject = d;
    if (st && !st.history.length) M.base(st, `Opened ${(d.path || d.title || "").split("/").pop()}`);
    if (st && !st.result && !st.busy && !guardFor(d, st) && !timers.get(d.id + ":first")) {
      timers.set(d.id + ":first", 1);
      run(d, S.setting("setupAutoPreview") !== false || true).finally(() => timers.delete(d.id + ":first"));
    }
    render();
  },
  async preview(explicit = true) {
    const d = subject || subjectDoc(), st = stateFor(d);
    if (!st) { S.toast("Open a CIF or a CRYSTAL output to set up a calculation", "info"); return null; }
    if (guardFor(d, st)) { S.toast("Choose which geometry the deck uses first", "info"); return null; }
    clearTimeout(timers.get(d.id));
    const p = run(d, true);
    running && (running.explicit = explicit);
    const r = await p;
    if (r && explicit) S.setupDeck(d, r, true);
    return r;
  },
  async save() {
    const d = subject, st = stateFor(d);
    if (!st || !st.result || !st.result.json_text) return null;
    if (st.stale) { const r = await api.preview(false); if (!r) return null; }
    const name = (root.querySelector("#su-name") || {}).value || st.saveName;
    const body = { source: { path: d.path }, calc_type: st.calc, json_text: st.result.json_text, name: (name || "").trim() || undefined,
      layout: st.mode === "cif2d12" ? (st.layout || "own") : undefined, overwrite: !!st.overwrite, write_decks: !!st.writeDecks,
      flags: chargeFlags(st) || undefined };
    try {
      const r = await S.apiTask("setup/save", body);
      st.saved = { ...r, sha: st.result.json_sha256, stale: false };
      persist(d, st);
      S.toast(`Saved ${r.json}`, "success");
      S.refreshFiles();
      render();
      return r;
    } catch (e) { S.toast(e.message, "error"); S.problem({ severity: "error", source: "Setup", doc: d.id, message: e.message }); return null; }
  },
  async copyCli() {
    const d = subject, st = stateFor(d);
    if (!st) return;
    if (!st.saved) { st.saveOpen = true; render(); S.toast("Save the settings first: the command names the saved file", "info"); return; }
    window.__maceLastCopy = st.saved.command;
    await S.copyText(st.saved.command, "Command copied");
  },
  /** The sidebar's Submit…: the deck these settings make, written (Save with Write decks now, under
   * Save's own rules: overwrite, a live run's wavefunction, ...) unless it already is, then the Submit
   * dialog on it. The dialog is the one Calculation › Submit opens: its preview, its single-use
   * preview id and its own Submit click; nothing here submits. Refused up front while the GUI is open
   * to the network (the dialog refuses it too), before anything is written. */
  async submitDeck() {
    const d = subject, st = stateFor(d);
    if (!st || !st.result || !st.result.json_text) { S.toast("Preview a deck first", "info"); return null; }
    if (st.result.save_blocked) { S.toast(st.result.save_blocked, "error"); return null; }
    if (M.modeOf(d, info(d)).source === "session") {
      st.saveOpen = true; render();
      S.toast("Save the structure as a CIF first: the command reads a file in the folder", "info");
      return null;
    }
    let caps = {};
    try { caps = await S.api("submit/capabilities"); } catch (e) { caps = {}; }
    if (caps.network) { S.toast("Submitting is turned off while MACE GUI is open to the network", "error"); return null; }
    const decksOf = (sv) => (sv && sv.written ? sv.written : []).filter((x) => /\.(d12|d3)$/i.test(x));
    let decks = st.saved && !st.stale && st.saved.sha === st.result.json_sha256 ? decksOf(st.saved) : [];
    if (!decks.length && st.mode === "opt2d3" && st.result.f9 && st.result.f9.live) {
      S.toast("The run is still going; its wavefunction (.f9) is copied when the run ends. Submit after it ends.", "info");
      return null;
    }
    if (!decks.length) {
      st.saveOpen = true; st.writeDecks = true;
      render();
      const r = await api.save();
      if (!r) return null;
      decks = decksOf(r);
      if (!decks.length) { S.toast(r.warning || "No deck was written; see Save for the CLI", "error"); return null; }
    }
    S.run("calc.submit", { paths: decks });
    return decks;
  },
  /** The sidebar's Batch Prepare + Submit…: Batch Setup on this file's folder, for the files like
   * this one (CIFs, runs needing this follow-up, or decks), this calculation, this file as the
   * representative and the form's changed fields as the settings for all files. Batch writes the
   * folder of decks and its step 4 opens Submit on them. */
  batchFrom() {
    const d = subject, st = stateFor(d);
    if (!st || !d) return null;
    const mo = M.modeOf(d, info(d));
    if (mo.source === "session" || !d.path) {
      S.toast("Save the structure as a CIF in the folder first: Batch Setup reads files in a folder", "info");
      return null;
    }
    const dir = d.path.includes("/") ? d.path.slice(0, d.path.lastIndexOf("/")) : "";
    const filter = st.mode === "cif2d12" ? "cif" : mo.deck ? "d12" : "followup";
    const socKeys = (schema.soc && schema.soc.keys) || [];
    // the SOC keys go with it (Batch Setup writes them into each file's settings, as Setup does)
    const patches = Object.fromEntries(Object.entries(st.patches || {}).filter(([k]) => socKeys.includes(k)));
    const preset = { filter, calc: st.calc, rep: d.path, answers: { ...st.answers }, patches,
      flags: CF.forBatch(schema.charge_flags, chargeFlags(st) || {}) };
    // what Batch Setup does not take (it takes the form's answers): named, so nothing is lost unseen
    const PATCH_NAMES = { band_path: "the edited k-path", k_points_for_all_files: "the k-point mesh" };
    const form = (st.result && st.result.form) || [];
    const left = Object.keys(st.patches || {}).filter((k) => !socKeys.includes(k)).map((k) => PATCH_NAMES[k] || (form.find((f) => f.id === k) || {}).label || k);
    if (st.hand !== null) left.push("the hand-edited settings file");
    if (chargeFlags(st) && "site_states" in st.flags) left.push("the site states (they name atoms of this structure)");
    if (left.length) {
      const list = left.length === 1 ? left[0] : `${left.slice(0, -1).join(", ")} and ${left[left.length - 1]}`;
      S.toast(`Batch Setup takes the form's fields; ${list} ${left.length === 1 ? "stays" : "stay"} here`, "info");
    }
    S.run("calc.batch", { dir, preset });
    S.run("calc.setup");                  // Batch is a document of its own: its tables get the room
    return preset;
  },
  undo() { const d = subject, st = stateFor(d); const l = M.undo(st); if (l !== null) { persist(d, st); S.toast(`Undo: ${l}`, "info"); schedule(d, null, 0); } return l; },
  redo() { const d = subject, st = stateFor(d); const l = M.redo(st); if (l !== null) { persist(d, st); S.toast(`Redo: ${l}`, "info"); schedule(d, null, 0); } return l; },
  canUndo: () => M.canUndo(stateFor(subject)),
  /** Run Preflight (plan T6): mace preflight on the previewed deck; what it finds goes to Problems. */
  async preflight() {
    const d = subject || subjectDoc(), st = stateFor(d);
    const r = st && st.result;
    if (!r || !r.deck) { S.toast("Preview a deck first", "info"); return null; }
    if (!/\.d12$/i.test(r.deck_name)) { S.toast("Preflight checks .d12 decks (CRYSTAL reads them with TESTPDIM)", "info"); return null; }
    st.preflight = { state: "running" };
    paintFooter();
    try {
      const res = await S.apiTask("setup/preflight", { deck_name: r.deck_name, deck: r.deck, doc: String(d.id), wait: 0.3 });
      st.preflight = res;
      S.clearProblems(d.id, "Preflight");
      for (const p of res.problems || []) S.problem({ ...p, doc: d.id, action: res.excerpt ? { label: "Details", run: () => S.showText(`Preflight: ${r.deck_name}`, res.excerpt) } : null });
      S.toast(res.status === "rejected" ? `Preflight: REJECTED (${res.reason})` : res.status === "accepted" ? "Preflight: ACCEPTED" :
        res.status === "static" ? "Preflight: structure check passed (CRYSTAL not on this machine)" : "Preflight: not checked",
      res.status === "rejected" ? "error" : "info");
      S.showProblems();
      return res;
    } catch (e) {
      st.preflight = null;
      if (e.state !== "canceled") S.toast(`Preflight: ${e.message}`, "error");
      return null;
    } finally { paintFooter(); }
  },
  /** Open Example Config: the example's file as the settings (a hand-written file the CLI reads). */
  async loadExample(t) {
    // the boot request lists the examples by name only; the config itself comes on first use
    if (t && !t.json) {
      const all = (await S.api("templates")).templates || [];
      t = all.find((x) => x.name === t.name) || t;
    }
    const d = subject || subjectDoc(), st = stateFor(d);
    if (!st || st.mode === "opt2d3") { S.toast("Example configs are for cif2d12 and opt2d12 decks", "info"); return; }
    const conf = (t.json && (t.json.configuration || t.json)) || {};
    const parentF = st.result && (st.result.form || []).find((f) => f.id === "functional");
    const exF = conf.functional || conf.dft_functional;
    if (follow(st) && parentF && exF && String(exF).toUpperCase() !== String(parentF.value).toUpperCase()) {
      const ok = await S.confirm({ title: "Change the method?", body: `This example uses ${exF}; the parent used ${parentF.value}. Use ${exF}?`,
        yes: `Use ${exF}`, no: `Keep ${parentF.value}`, defaultNo: true });
      if (!ok) return;
    }
    const prev = st.hand;
    bulk(d, `Loaded ${t.name}`, (s) => { s.hand = t.text; s.answers = {}; s.patches = {}; }, `Loaded ${t.name}`);
    return prev;
  },
  resetAll() { const d = subject, st = stateFor(d); if (!st) return; let n = 0; bulk(d, "Reset all", (s) => { n = M.resetAll(s); }, null); S.toast(`Reset ${n} field${n === 1 ? "" : "s"}`, "info", { label: "Undo", run: () => api.undo() }); },
  docChanged() { if (root && root.isConnected) { const d = subjectDoc(); if (d !== subject) api.refresh(); else render(); } },
  state: (d) => stateFor(d || subject),
  subject: () => subject,
  hasDeck: () => { const st = stateFor(subject); return !!(st && st.result && st.result.deck); },
  focus() { const c = root && root.querySelector(".su-calc [aria-checked=true], .su-row select, .su-row input"); if (c) c.focus(); },
  historyRows: (d) => { const st = stateFor(d || subject); return st ? st.history.map((h, k) => ({ label: h.label, t: h.t, cur: k === st.pos })) : []; },
  jump: (k) => { const d = subject, st = stateFor(d); M.jump(st, k); persist(d, st); schedule(d, null, 0); },
  bytes: () => [...states.values()].reduce((t, s) => t + M.historyBytes(s), 0),
  forget: (d) => { const st = states.get(d.id); if (st && st.bz) st.bz.destroy(); states.delete(d.id); },
};
