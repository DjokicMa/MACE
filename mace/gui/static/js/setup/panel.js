/* Calculation Setup (spec §1.2, plan B1-B4): a panel docked in the inspector,
 * bound to the active document, in three steps (structure, settings, save
 * for the CLI) with a sticky footer; the previewed deck opens as a Deck
 * document beside it. Every field is one of the CLI's own questions: an edit
 * is the answer, the server runs the CLI with it, and the form, the JSON and
 * the deck come back from there. Loaded on the first Calculation Setup. */
import * as M from "./model.js";
import { loadCss } from "../lazy/lazy_util.js";
import * as profile from "../profile.js";

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
  if (!st || (mo.mode === "cif2d12" && st.mode !== "cif2d12") || (mo.mode === "follow" && st.mode === "cif2d12")) {
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
  const charge = st.flags && st.flags.net_charge;
  if (charge !== undefined && charge !== null) body.flags = { net_charge: charge };
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
  root.appendChild(el(`<header class="su-head"><h2 class="su-title">Calculation Setup</h2>${st ? `<span class="su-chip" title="The CLI command that makes this deck">${esc(st.mode)} · ${esc(M.MODES[st.mode])}</span>` : ""}</header>`));
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
const FOCUS_KEYS = ["data-calc", "data-act", "data-reset", "data-hist"];
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
    s.appendChild(el(`<div class="su-empty"><p>Open a CIF or a CRYSTAL output to set up a calculation.</p>
      <button class="tb-btn-text" data-cmd="file.open">${icon("folder-open")}<span>Open…</span><kbd>${esc(S.shortcut("file.open"))}</kbd></button></div>`));
    return s;
  }
  const mo = M.modeOf(d, info(d));
  if (!mo.mode) {
    s.appendChild(el(`<div class="su-empty"><p>A deck needs a CIF (a new calculation) or a CRYSTAL .out (a follow-up). Save this structure as CIF?</p>
      <button class="tb-btn-text" data-act="saveAsCif">${icon("save")}<span>Save As CIF…</span></button></div>`));
    return s;
  }
  const name = (d.path || d.title || "").split("/").pop(), dir = d.path && d.path.includes("/") ? d.path.slice(0, d.path.lastIndexOf("/") + 1) : "";
  const kind = d.fileKind || "cif";
  const src = el(`<div class="su-src"><span class="kb ${esc(kind)}">${esc(kind === "out" ? "out" : kind.toUpperCase())}</span>
    <span class="su-name" title="${esc(d.path || d.title)}">${esc(name)}</span><span class="muted">${esc(dir)}</span></div>`);
  s.appendChild(src);
  const meta = (d.meta || (S.structureOf(d) || {}).doc_meta || {});
  const prov = meta.provenance && meta.provenance.label;
  if (prov) s.appendChild(el(`<p class="su-prov">${esc(prov)}</p>`));
  if (mo.source === "session") s.appendChild(el(`<p class="su-note">${icon("info")}<span>Not saved in the folder: the preview uses this structure as a CIF; Save asks to save it first.</span></p>`));
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

function calcSegment(d, st) {
  const btn = (t, prop) => `<button role="radio" aria-checked="${t === st.calc}" data-calc="${esc(t)}" class="${prop ? "prop" : ""}"` +
    `${t === "CHARGE+POTENTIAL" ? ' title="CHARGE+POTENTIAL"' : ""}>${esc(t === "CHARGE+POTENTIAL" ? "CHG+POT" : t)}</button>`;
  if (st.mode === "cif2d12") {
    return `<div class="su-calc" role="radiogroup" aria-label="Calculation type">${(schema.calc_types.cif2d12 || []).map((t) => btn(t, false)).join("")}</div>`;
  }
  // a follow-up: rows of three equal segments, the SCF runs (opt2d12) above the properties of its wavefunction (opt2d3)
  const row = (types, prop, label) => `<div class="su-calc-row" title="${esc(label)}">${types.map((t) => btn(t, prop)).join("")}</div>`;
  const props = schema.calc_types.opt2d3 || [], rows = [];
  for (let k = 0; k < props.length; k += 3) rows.push(row(props.slice(k, k + 3), true, "Properties from this output's wavefunction (mace opt2d3)"));
  return `<div class="su-calc grouped" role="radiogroup" aria-label="Calculation type">` +
    row(schema.calc_types.opt2d12 || [], false, "A new SCF run from this output (mace opt2d12)") + rows.join("") + "</div>";
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
  s.appendChild(advanced(d, st));
  return s;
}

const fid = (f) => "su-f-" + f.id.replace(/[^A-Za-z0-9_-]/g, "_").slice(0, 60);

function row(d, st, f, fol) {
  const id = fid(f);
  const b = M.badge(f, fol);
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
function onClick(e) {
  const d = subject, st = stateFor(d);
  const t = e.target;
  const sec = t.closest(".su-sec-h");
  if (sec) {
    const box = sec.parentElement, name = box.dataset.sec;
    if (name === "Advanced") { st.advancedOpen = !st.advancedOpen; render(); return; }
    const c = new Set(profile.ui("setupCollapsed", []) || []);
    const nowOpen = box.classList.contains("collapsed");
    if (name === "Other questions") { if (nowOpen) { c.add("__other_open"); c.delete(name); } else c.delete("__other_open"); }
    else if (nowOpen) c.delete(name); else c.add(name);
    profile.setUi("setupCollapsed", [...c]);
    box.classList.toggle("collapsed", !nowOpen);
    sec.setAttribute("aria-expanded", String(nowOpen));
    return;
  }
  const calc = t.closest("[data-calc]");
  if (calc && st) {
    const c = calc.dataset.calc;
    if (c === st.calc) return;
    const was = st.calc;
    st.calc = c;
    if (st.mode !== "cif2d12") st.mode = M.followMode(c, schema.calc_types);
    st.answers = {}; st.patches = {}; st.hand = null; st.result = null;
    M.record(st, `Calculation ${was} → ${c}`);
    persist(d, st);
    schedule(d, null, 0);
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
  if (a === "preview") api.preview(true);
  else if (a === "cancel" && st.busy && st.busy.id) S.cancelTask(st.busy.id);
  else if (a === "saveOpen") { st.saveOpen = true; render(); const n = root.querySelector("#su-name"); if (n) n.focus(); }
  else if (a === "save") api.save();
  else if (a === "copyCmd") api.copyCli();
  else if (a === "reveal" && st.saved) S.reveal(st.saved.json);
  else if (a === "submitDecks" && st.saved) S.run("calc.submit", { paths: (st.saved.written || []).filter((x) => /\.(d12|d3)$/.test(x)) });
  else if (a === "guardParent") { st.guard = "parent"; st.guardFor = guardSig(d); S.showParentGeometry(d); persist(d, st); render(); schedule(d, null, 0); }
  else if (a === "guardCif" || a === "saveAsCif") S.saveAsCifForSetup(d);
  else if (a === "applyJson") {
    const ta = root.querySelector("#su-json");
    try { JSON.parse(ta.value); } catch (x) { S.toast(`Not valid JSON: ${x.message}`, "error"); return; }
    bulk(d, "Edit the settings file by hand", (s) => { s.hand = ta.value; });
  } else if (a === "chargeOn" || a === "chargeOff") {
    const meta = d.meta || (S.structureOf(d) || {}).doc_meta || {};
    bulk(d, a === "chargeOn" ? `Net charge ${meta.hints.charge}` : "Net charge from the defect builder removed",
      (s) => { s.flags = a === "chargeOn" ? { net_charge: meta.hints.charge } : {}; }, a === "chargeOn" ? `Applied the net charge ${meta.hints.charge}` : null);
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
  const r = t.closest(".su-row");
  if (!r || !st || !st.result) return;
  const f = st.result.form.find((x) => x.id === r.dataset.field);
  if (!f) return;
  if (f.kind === "yesno") { t.setAttribute("aria-checked", String(t.checked)); edit(d, f, t.checked); }
  else if (f.kind === "menu") edit(d, f, t.value);
  else edit(d, f, t.value.trim(), `field:${f.id}`);
}

function onInput(e) {
  const t = e.target;
  const r = t.closest && t.closest(".su-row");
  if (!r || t.tagName !== "INPUT" || t.type === "checkbox") return;
  const d = subject, st = stateFor(d);
  const f = st && st.result && st.result.form.find((x) => x.id === r.dataset.field);
  if (!f) return;
  // typing: one history entry per field within 1 s, one CLI run 600 ms after the last key
  const label = M.applyEdit(st, f, t.value.trim() === "" ? null : t.value.trim());
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
      layout: st.mode === "cif2d12" ? (st.layout || "own") : undefined, overwrite: !!st.overwrite, write_decks: !!st.writeDecks };
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
  forget: (d) => { states.delete(d.id); },
};
