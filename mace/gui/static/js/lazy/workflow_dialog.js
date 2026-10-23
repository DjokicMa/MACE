/* New Workflow… (GUI_WORKFLOW_DESIGN.md section 4, Phase 4): a builder in
 * five sections (structures, steps, settings, resources, review and
 * submit). The page writes nothing itself: the server turns the draft into a
 * workflow spec, and the CLI turns the spec into its plan and launches it
 * (mace workflow --spec SPEC --write-plan, then the command it prints).
 * Review previews exactly those commands in a staged copy; Submit runs them
 * once, with the preview's single-use id. Each step's settings are the
 * Calculation Setup form (the CLI's own questions); each value says where
 * it comes from (default, inherited from the step before, site setting,
 * set here) and has a Reset. Loaded on first use. */
import { loadCss } from "./lazy_util.js";
import * as M from "../setup/model.js";
import { MODULE_FIELDS, MODULE_FLAGS, MODULE_LISTS, MODULE_VOIGT } from "../setup/modules.js";

const SECTIONS = [
  { id: "inputs", title: "Structures", sub: "What to calculate" },
  { id: "chain", title: "Steps", sub: "Which calculations, in order" },
  { id: "settings", title: "Settings", sub: "How each step is set up" },
  { id: "resources", title: "Resources", sub: "Cores, memory, time" },
  { id: "review", title: "Review and submit", sub: "Exactly what will run" },
];
const RES_COLS = [
  { k: "ntasks", label: "Cores", hint: "MPI processes (--ntasks)", ph: "32" },
  { k: "mem", label: "Memory", hint: "per job or per core", ph: "48G" },
  { k: "walltime", label: "Time limit", hint: "D-HH:MM:SS; the job stops when it runs out", ph: "1-00:00:00" },
  { k: "partition", label: "Partition", hint: "empty: SLURM's default", ph: "default" },
  { k: "account", label: "Account", hint: "empty: your site setting or SLURM's default", ph: "default" },
  { k: "exclude", label: "Avoid nodes", hint: "node names to skip, e.g. node[01-03]", ph: "none" },
];
const ORIGIN = (o) => {
  const s = String(o || "builtin");
  if (s.startsWith("spec")) return { kind: "you", text: "set here" };
  if (s.startsWith("site")) return { kind: "site", text: s.replace(/^site /, "site ") };
  if (s === "env") return { kind: "site", text: "environment variable" };
  return { kind: "default", text: "MACE default" };
};
const DEBOUNCE_MS = 500;
const plural = (n, one, many = one + "s") => `${n} ${n === 1 ? one : many}`;
const stepKey = (seq, i) => `${seq[i]}_${i + 1}`;
const baseOf = (t) => String(t).replace(/\d+$/, "");
const D12_STEPS = ["OPT", "SP", "FREQ"];
// an ELASTIC, EOS, QHA or CPHF step's module block as the spec gets it: PRINT on unless it was unticked
// (print: false), as the CLI's questions, the planner and Calculation Setup have it; PIEZOCP, QHA and CPHF have no PRINT
const MODULE_STEPS = ["ELASTIC", "EOS", "QHA", "CPHF"], NO_PRINT = ["PIEZOCP", "QHA", "CPHF"];
function moduleOut(calc, mod) {
  const m = JSON.parse(JSON.stringify(mod || {}));
  const mode = String(m.mode || (calc === "ELASTIC" ? "ELASTCON" : calc)).toUpperCase();
  if (m.print === false || NO_PRINT.includes(mode)) delete m.print;
  else m.print = true;
  return m;
}
const dirOf = (p) => (String(p).includes("/") ? String(p).slice(0, String(p).lastIndexOf("/")) : "");
// spin-orbit coupling (1.1.6): a draft step's choices, as the CLI's settings-file keys (soc, soc_ecp, soc_basis, smear)
const socOn = (s) => !!(s && s.soc && s.soc.soc === true);
const socPatches = (s) => (socOn(s) ? Object.fromEntries(Object.entries(s.soc).filter(([, v]) => v !== null && v !== undefined && v !== "")) : {});
const SOC_ROUTES = [
  { v: "inpsoc", label: "INPSOC: MACE's spin-orbit ECP records (default)" },
  { v: "stutsc", label: "STUTSC: CRYSTAL's Stuttgart spin-orbit library" },
  { v: "basis", label: "Official SOC basis set (pob-TZVP-rev2-SOC)" },
];
const stepName = (k) => { const m = /^(.*)_(\d+)$/.exec(k || ""); return m ? `${m[1]} (step ${m[2]})` : k; };
let last = null;          // the draft of the last builder this session (reopened as it was)

export async function open(H, { targets = [], clone = null } = {}) {
  await loadCss(import.meta.url, "../../css/dialogs.css");
  await loadCss(import.meta.url, "../../css/setup.css");
  await loadCss(import.meta.url, "../../css/submit.css");
  await loadCss(import.meta.url, "../../css/workflow_builder.css");
  H.overlayOpening();
  const S = H.shell();
  const opener = document.activeElement;
  let meta = { templates: [], calc_types: {} };
  try { meta = await H.api("workflow/templates"); } catch (e) { H.toast(`Workflows: ${e.message}`, "error"); }
  // Clone (the Workflows view): the structures, steps and settings of an existing workflow (GET workflow/clone)
  const fresh = !!clone || !last || (targets && targets.length);
  const cloned = clone ? { inputs: null, steps: {}, execution: {}, ...JSON.parse(JSON.stringify(clone.draft || {})) } : null;
  const state = {
    sec: 0,
    targets: clone ? (clone.targets || []) : fresh ? (targets || []) : last.targets,
    inputs: fresh ? null : last.inputs,
    clone: clone ? { source: clone.source, notes: clone.notes || [], kind: clone.kind } : null,
    draft: cloned || (fresh ? { inputs: null, label: "", template: "full_electronic", sequence: ["OPT", "SP", "BAND", "DOSS"], steps: {}, execution: {} } : last.draft),
    inputsError: "", folder: fresh ? null : last.folder,
    defaults: null, defaultsError: "", defaultsBusy: false,
    forms: {}, formBusy: {}, formError: {}, formSeq: {}, formStep: null,
    preview: null, previewing: false, previewError: "", stale: true, defSeq: 0, prevSeq: 0,
    running: false, runError: "", ticks: {},
    visited: new Set([0]),            // the sections looked at: a check mark on Settings or Resources means you were there
  };
  const d = document.createElement("dialog");
  d.id = "wfb-dialog";
  d.className = "dlg wfb-dlg";
  d.setAttribute("aria-labelledby", "wfb-title");
  document.getElementById("wfb-dialog")?.remove();
  document.body.appendChild(d);
  d.innerHTML = `<div class="dlg-head"><h2 id="wfb-title">${clone ? "New Workflow (a copy)" : "New Workflow"}</h2><button type="button" class="icon-btn sm" data-close aria-label="Close">${H.icon("x")}</button></div>
    <div class="wfb-main"><nav class="wfb-nav" aria-label="Workflow builder sections"><ol></ol></nav>
      <section class="wfb-sec" tabindex="-1" aria-live="polite"></section></div>
    <div class="dlg-foot"><span class="wfb-note">Nothing is submitted until you click Submit.</span>
      <button type="button" class="tb-btn-text" data-act="back">Back</button>
      <button type="button" class="tb-btn-text" data-act="next">Next</button>
      <button type="button" class="tb-btn-text primary" data-act="submit" disabled>Submit</button></div>`;
  const nav = d.querySelector(".wfb-nav ol"), sec = d.querySelector(".wfb-sec");
  const btnBack = d.querySelector('[data-act="back"]'), btnNext = d.querySelector('[data-act="next"]'), btnSubmit = d.querySelector('[data-act="submit"]');
  const esc = H.esc, icon = H.icon;
  const seq = () => state.draft.sequence;
  const keys = () => seq().map((t, i) => stepKey(seq(), i));
  const stepOf = (k) => (state.draft.steps[k] = state.draft.steps[k] || {});
  const caps = { network: false };
  try { const c = await H.api("submit/capabilities"); Object.assign(caps, c || {}); } catch (e) { /* the run route says so too */ }

  /* ------------------------------------------------------------ draft helpers */
  function draftForServer() {
    const steps = {};
    for (const k of keys()) {
      const s = state.draft.steps[k];
      if (!s) continue;
      const o = {};
      // a CIF start's first step: the form's answers (the server asks the CLI what they change)
      if (k === keys()[0] && state.inputs && state.inputs.type === "cif") {
        if (Object.keys(s.answers || {}).length) o.answers = { ...s.answers };
        if (Object.keys(s.patches || {}).length) o.patches = { ...s.patches };
      } else if (k !== keys()[0] && D12_STEPS.includes(baseOf(seq()[keys().indexOf(k)])) && Object.keys(s.answers || {}).length) {
        // a later SCF step: the form's answers (basis set, SCF convergence, ...); the server asks the CLI what they change
        o.answers = { ...s.answers };
      }
      if (s.settings_text) o.settings_text = s.settings_text;
      if (s.method) o.method = s.method;
      // a TAU step's options, a TRANSPORT step's relaxation time (mace/workflow/tau_step.py)
      if (s.tau && Object.keys(s.tau).length) o.tau = { ...s.tau };
      // an ELASTIC, EOS, QHA or CPHF step's module block (the spec's elastic_settings)
      const calc = baseOf(seq()[keys().indexOf(k)]);
      if (MODULE_STEPS.includes(calc)) { const m = moduleOut(calc, s.module); if (Object.keys(m).length) o.module = m; }
      else if (s.module && Object.keys(s.module).length) o.module = JSON.parse(JSON.stringify(s.module));
      if (s.relaxation_time !== undefined && s.relaxation_time !== null && s.relaxation_time !== "default") o.relaxation_time = s.relaxation_time;
      if (socOn(s)) o.soc = socPatches(s);
      const res = {};
      for (const [rk, v] of Object.entries(s.resources || {})) if (v !== "" && v !== null && v !== undefined) res[rk] = v;
      if (Object.keys(res).length) o.resources = res;
      if (Object.keys(o).length) steps[k] = o;
    }
    const ex = {};
    const e = state.draft.execution || {};
    if (e.array) ex.job_arrays = { first_step: true, ...(e.max_running ? { max_running: Number(e.max_running) } : {}) };
    if (e.max_jobs) ex.max_jobs = Number(e.max_jobs);
    if (e.auto_progression === false) ex.auto_progression = false;      // kept from a cloned workflow
    return { inputs: state.inputs ? { type: state.inputs.type, files: state.inputs.files } : null,
      label: state.draft.label || "", template: state.draft.template, sequence: seq(), steps, execution: ex };
  }
  /** The chain with a step put in or taken out: the other steps keep their settings under their new keys
   * (SP_3 -> SP_2). ``to(j)`` is old step j's new index, or null when it goes. */
  function rechain(s, old, to) {
    const steps = {}, moved = {};
    old.forEach((t, j) => {
      const n = to(j);
      if (n === null) return;
      const from = stepKey(old, j), k = stepKey(s, n);
      moved[from] = k;
      if (state.draft.steps[from]) steps[k] = state.draft.steps[from];
    });
    state.draft.sequence = s; state.draft.template = null; state.draft.steps = steps; state.forms = {};
    state.formStep = moved[state.formStep] || null;
  }
  function insertStep(i, type) { const old = [...seq()], s = [...old]; s.splice(i, 0, type); rechain(s, old, (j) => (j < i ? j : j + 1)); }
  function removeStep(i) { const old = [...seq()], s = [...old]; s.splice(i, 1); rechain(s, old, (j) => (j === i ? null : j < i ? j : j - 1)); }
  function changed(what) {
    if (what === "inputs" || what === "chain") autoLabel();
    state.stale = true;
    state.preview = null;
    state.prevSeq++;
    state.previewing = false;
    // a step's form is built from the steps before it: a changed step keeps its own, the chain or the structures keep none
    if (what === "settings") state.forms = Object.fromEntries(Object.entries(state.forms).filter(([k]) => k === state.formStep));
    else if (what === "chain" || what === "inputs") state.forms = {};
    remember();
    scheduleDefaults();
  }
  /** A name until the student types one: the folder and the chain ("cifs-full_electronic"). */
  function autoLabel() {
    if (state.draft.labelAuto === false && state.draft.label) return;
    const f = state.inputs ? (state.inputs.folder_rel || "").split("/").pop() : "";
    const name = `${f ? f + "-" : ""}${state.draft.template || "custom"}`.replace(/[^A-Za-z0-9._+-]+/g, "_").slice(0, 64);
    state.draft.label = name; state.draft.labelAuto = true;
  }
  function uiDraft() {
    const x = JSON.parse(JSON.stringify(state.draft));
    delete x.inputs; delete x.labelAuto;
    return x;
  }
  function remember() { last = { targets: state.targets, inputs: state.inputs, draft: state.draft, folder: state.folder }; }
  const rootGen = () => { const i = H.info(); return i && i.root_gen !== undefined ? i.root_gen : window.__maceRootGen; };

  /* ------------------------------------------------------------ server calls */
  async function scanInputs(targetsNow, kind) {
    state.inputsError = "";
    if (!targetsNow.length) { state.inputs = null; render(); return; }
    try {
      const r = await H.api("workflow/inputs", { body: { targets: targetsNow, ...(kind ? { type: kind } : {}) } });
      state.inputs = r;
      state.targets = targetsNow;
      // a deck start begins with the decks' own calculation type
      if (r.type === "d12" && (r.deck_types || []).length === 1 && seq()[0] !== r.deck_types[0]) {
        const t = r.deck_types[0];
        const pick = (meta.templates || []).find((x) => x.sequence[0] === t);
        state.draft.sequence = pick ? [...pick.sequence] : [t];
        state.draft.template = pick ? pick.name : null;
        state.draft.steps = {};
      }
      if (r.type === "cif" && !["OPT", "SP"].includes(seq()[0])) { state.draft.sequence = ["OPT", "SP", "BAND", "DOSS"]; state.draft.template = "full_electronic"; }
    } catch (e) { state.inputs = null; state.inputsError = e.message; }
    changed("inputs");
    render();
  }
  let defTimer = 0;
  function scheduleDefaults() { clearTimeout(defTimer); defTimer = setTimeout(loadDefaults, DEBOUNCE_MS); }
  async function loadDefaults() {
    if (!state.inputs) return;
    const my = ++state.defSeq;
    state.defaultsBusy = true;
    if (state.sec === 3) render();
    try {
      const r = await S.apiTask("workflow/defaults", { draft: draftForServer(), doc: "wfbuilder" });
      if (my !== state.defSeq) return;
      state.defaults = r; state.defaultsError = "";
    } catch (e) {
      if (my !== state.defSeq || e.state === "canceled") return;
      state.defaultsError = e.message;
    } finally {
      if (my === state.defSeq) { state.defaultsBusy = false; if ([1, 2, 3].includes(state.sec)) render(); }
    }
  }
  async function loadForm(k, retry = 0) {
    const s = stepOf(k);
    let again = false;
    // one answer per step counts: the latest (a superseded run neither shows its form nor ends "Updating…")
    const my = state.formSeq[k] = (state.formSeq[k] || 0) + 1;
    state.formBusy[k] = true; state.formError[k] = "";
    render();
    try {
      const r = await S.apiTask("workflow/step-form", { draft: draftForServer(), step: k, answers: s.answers || {},
        patches: { ...(s.patches || {}), ...socPatches(s) }, doc: "wfbuilder" });
      if (my !== state.formSeq[k]) return;
      state.forms[k] = r;
      // the settings the spec gets: the CLI's own file, only when something was changed
      const changedAny = Object.keys(s.answers || {}).length || Object.keys(s.patches || {}).length;
      if (r.mode === "opt2d12") s.method = methodOf(r, s);
      // a cloned step keeps the settings file it was copied with until a field here is changed
      else if (r.mode === "opt2d3") s.settings_text = changedAny && r.json_text ? r.json_text : (s.from_clone ? s.settings_text : null);
      if (changedAny && r.mode === "opt2d3") s.from_clone = false;
    } catch (e) {
      // superseded on the server (409 {canceled}): two requests sent together can arrive in either order, so
      // the server may have kept the older one; the latest asks again rather than leave its form unanswered
      const superseded = e.state === "canceled" || (e.data && e.data.canceled);
      if (superseded && my === state.formSeq[k] && retry < 3) { again = true; return; }
      if (superseded || my !== state.formSeq[k]) return;
      state.formError[k] = e.message;
    } finally {
      if (again) setTimeout(() => loadForm(k, retry + 1), 0);
      else if (my === state.formSeq[k]) {
        state.formBusy[k] = false;
        remember();
        render();
      }
    }
  }
  /** A later SCF step: the functional and dispersion as the form shows them, when either was changed. */
  function methodOf(r, s) {
    const a = s.answers || {};
    if (!("functional" in a) && !("dispersion" in a)) return null;     // only other settings were changed: the method stays the parent's
    const f = (r.form || []).find((x) => x.id === "functional"), disp = (r.form || []).find((x) => x.id === "dispersion");
    const fn = a.functional || (f && f.value);
    if (!fn) return null;
    const dv = "dispersion" in a ? a.dispersion : (disp ? disp.value : "No");
    return { functional: fn, dispersion: [true, "y", "Yes", "yes"].includes(dv) };
  }

  /* ------------------------------------------------------------ preview / submit */
  async function preview() {
    if (!state.inputs) return;
    const my = ++state.prevSeq;
    state.previewing = true; state.previewError = ""; state.ticks = {};
    render();
    try {
      // ui_draft: the builder's own draft, saved next to the spec so Clone reopens it exactly (the CLI never reads it)
      const r = await S.apiTask("workflow/preview", { root_gen: rootGen(), draft: draftForServer(), ui_draft: uiDraft(), doc: "wfbuilder" });
      if (my !== state.prevSeq) return;
      state.preview = r; state.stale = false;
    } catch (e) {
      if (my !== state.prevSeq) return;
      state.preview = null; state.previewError = e.state === "canceled" ? "" : e.message;
    } finally {
      if (my === state.prevSeq) { state.previewing = false; render(); }
    }
  }
  function canSubmit() {
    const p = state.preview;
    if (!p || state.stale || state.previewing || state.running) return false;
    return !(p.code || (p.blocked || []).length || caps.network || p.network);
  }
  async function submit() {
    if (!canSubmit()) return;
    state.running = true; state.runError = "";
    render();
    try {
      const r = await S.apiTask("workflow/run", { root_gen: rootGen(), preview_id: state.preview.preview_id });
      state.running = false;
      if (r.code) {
        state.runError = `The CLI exited with ${r.code}${r.error ? `: ${r.error}` : ""}`;
        state.preview = { ...state.preview, cli_log: r.log, code: r.code };
        render();
        return;
      }
      const ids = r.job_ids || [];
      last = null;
      d.close();
      H.toast(`Workflow started: ${plural(ids.length, "job")} submitted${ids.length ? ` (${ids.slice(0, 4).join(", ")}${ids.length > 4 ? ", …" : ""})` : ""}`,
        "success", { label: "Open Jobs", run: () => S.showBottom("jobs") });
      for (const x of r.script_drift || []) S.problem({ severity: "warning", source: "Workflow", message: `The submitted job scripts differ from the preview: ${x.script || x.command}\n${x.diff}` });
      document.dispatchEvent(new CustomEvent("mace-jobs-changed", { detail: { ids } }));
      if (S.refreshFiles) S.refreshFiles();
      S.openWorkflows(`${r.folder_rel || ""}::${r.workflow_id}`);
    } catch (e) {
      state.running = false;
      state.runError = e.message;
      if (e.status === 409) state.stale = true;
      if (d.open) render(); else H.toast(e.message, "error");
    }
  }

  /* ------------------------------------------------------------ rendering */
  function done(i) {
    if (i === 0) return !!state.inputs;
    if (i === 1) return !!state.inputs && seq().length > 0;
    if (i === 2 || i === 3) return state.visited.has(i) && !!state.inputs && !!state.defaults && !state.defaultsError;
    return !!state.preview && !state.stale && !state.preview.code;
  }
  function renderNav() {
    nav.innerHTML = SECTIONS.map((s, i) => `<li><button type="button" class="wfb-tab${i === state.sec ? " on" : ""}" data-sec="${i}"
      aria-current="${i === state.sec ? "step" : "false"}"${i > 0 && !state.inputs ? " disabled" : ""}>
      <span class="wfb-num${done(i) ? " ok" : ""}">${done(i) && i !== state.sec ? icon("check") : i + 1}</span>
      <span class="wfb-tab-t"><b>${esc(s.title)}</b><span class="small muted">${esc(s.sub)}</span></span></button></li>`).join("");
  }
  function renderFoot() {
    btnBack.disabled = state.sec === 0 || state.running;
    btnNext.hidden = state.sec === SECTIONS.length - 1;
    btnNext.disabled = !state.inputs || state.running;
    btnSubmit.hidden = state.sec !== SECTIONS.length - 1;
    btnSubmit.disabled = !canSubmit();
    const n = state.preview ? state.preview.n_jobs || 0 : 0;
    btnSubmit.textContent = state.running ? "Submitting…" : `Submit ${plural(n, "job")}`;
    d.querySelectorAll("[data-close]").forEach((x) => { x.disabled = state.running; });
  }

  function secInputs() {
    const files = (H.files() || []).filter((f) => /\.(cif|d12)$/i.test(f.path || f.name || ""));
    const folders = [...new Set(files.map((f) => dirOf(f.path)))].sort();
    const cur = state.inputs;
    const t0 = state.targets[0];
    let folder = state.folder;
    if (folder === null) folder = cur ? cur.folder_rel : t0 ? (/\.(cif|d12)$/i.test(t0) ? dirOf(t0) : t0.replace(/\/$/, "").replace(/^\.$/, "")) : (folders[0] || "");
    const here = files.filter((f) => dirOf(f.path) === folder).sort((a, b) => a.path.localeCompare(b.path, undefined, { numeric: true }));
    const chosen = new Set(cur ? cur.files : []);
    const kind = cur ? cur.type : null;
    let html = `<h3>Structures</h3>
      <p class="wfb-intro">A workflow runs the same chain of calculations for every structure you pick, one after another, without you submitting each step.
        Pick <b>CIF files</b> (MACE writes the first input deck for you) or <b>.d12 decks</b> you already made, from one folder.</p>`;
    html += `<div class="wfb-row"><label class="f" for="wfb-folder">Folder</label><select id="wfb-folder" class="field">
      ${folders.map((f) => `<option value="${esc(f)}"${f === folder ? " selected" : ""}>${esc(f || "this folder")}</option>`).join("")}
      ${folders.includes(folder) ? "" : `<option value="${esc(folder)}" selected>${esc(folder || "this folder")}</option>`}</select>
      <span class="muted small">The workflow's files (plans, inputs, outputs) are written in this folder.</span></div>`;
    if (!here.length) html += `<p class="wfb-empty">No CIF or .d12 files in ${esc(folder || "this folder")}. Save a structure as CIF, or pick another folder.</p>`;
    else {
      const nC = here.filter((f) => /\.cif$/i.test(f.path)).length, nD = here.length - nC;
      html += `<div class="wfb-pick"><div class="wfb-pick-acts">
        ${nC ? `<button type="button" class="tb-btn-text sm" data-all="cif">All ${plural(nC, "CIF")}</button>` : ""}
        ${nD ? `<button type="button" class="tb-btn-text sm" data-all="d12">All ${plural(nD, "deck")}</button>` : ""}
        <button type="button" class="tb-btn-text sm" data-all="none">None</button></div>
        <ul class="wfb-files" role="list">${here.map((f) => {
          const k = /\.cif$/i.test(f.path) ? "cif" : "d12";
          const m = cur && (cur.materials || []).find((x) => x.file === f.path);
          const off = kind && kind !== k && chosen.size;
          return `<li><label class="tog${off ? " off" : ""}"><input type="checkbox" data-file="${esc(f.path)}"${chosen.has(f.path) ? " checked" : ""}${off ? ` disabled title="A workflow starts from CIF files or from decks, not both"` : ""}>
            <span class="kb ${k}">${k.toUpperCase()}</span><span class="mono">${esc(f.name || f.path.split("/").pop())}</span>
            ${m ? `<span class="muted small">→ named <b>${esc(m.material_id)}</b> in MACE</span>` : ""}</label></li>`;
        }).join("")}</ul></div>`;
    }
    if (state.inputsError) html += `<p class="sub-line bad" role="alert">${esc(state.inputsError)}</p>`;
    if (cur) html += `<p class="wfb-ok">${icon("circle-check")}<span>${plural(cur.files.length, cur.type === "cif" ? "CIF file" : "deck")} picked.
      ${cur.type === "cif" ? "The first step converts each CIF to a deck with the settings you choose under Settings." : "The first step runs your decks as they are."}</span></p>`;
    return html;
  }

  function secChain() {
    const t = meta.calc_types || {};
    const cur = seq();
    const temps = (meta.templates || []).filter((x) => !state.inputs || state.inputs.type !== "d12" || x.sequence[0] === cur[0]);
    let html = `<h3>Steps</h3><p class="wfb-intro">Each structure goes through these steps in order. A step starts by itself when the step it is built from finishes
      (for example the single point (SP) starts from the optimized structure of OPT).</p>`;
    html += `<div class="wfb-label">Start from a ready-made chain</div><div class="wfb-chips" role="radiogroup" aria-label="Ready-made chains">${temps.map((x) =>
      `<button type="button" role="radio" class="wfb-chip${state.draft.template === x.name ? " on" : ""}" aria-checked="${state.draft.template === x.name}" data-template="${esc(x.name)}" title="${esc(x.help)}">
        <b>${x.num ? `${esc(x.num)} - ` : ""}${esc(x.sequence.join(" → "))}</b><span class="small muted">${esc(x.help)}</span></button>`).join("")}</div>`;
    if (state.defaultsError) html += `<p class="sub-line bad" role="alert" data-part="chain-error">This chain cannot run: ${esc(state.defaultsError)}</p>`;
    html += `<div class="wfb-label">Your chain${state.draft.template ? "" : " (custom)"}</div><ol class="wfb-chain">${cur.map((s, i) => {
      const k = stepKey(cur, i);
      const par = state.defaults && state.defaults.steps && state.defaults.steps[k] ? state.defaults.steps[k].parent : null;
      return `<li class="wfb-node"><span class="wfb-node-n">${i + 1}</span><span class="wfb-node-t"><b>${esc(s)}</b>
        <span class="small muted">${esc((t[baseOf(s)] || {}).name || "")}${i === 0 ? " · first step" : par ? ` · from ${esc(par.replace(/_\d+$/, ""))}` : ""}${baseOf(s) === "TAU" ? ` · ${state.forms[k] && state.forms[k].n_runs ? esc(plural(state.forms[k].n_runs, "strained SP")) : "4 strained SPs per axis"}` : ""}</span></span>
        ${i > 0 || cur.length > 1 ? `<button type="button" class="icon-btn sm" data-remove="${i}" aria-label="Remove ${esc(s)}" title="Remove this step">${icon("x")}</button>` : ""}</li>`;
    }).join(`<li class="wfb-arrow" aria-hidden="true">${icon("chevron-right")}</li>`)}</ol>`;
    const next = nextTypes(cur);
    // steps these structures cannot have (ORBITALS from two-component decks): grayed out, the reason as tooltip
    const refused = (state.inputs && state.inputs.refused) || {};
    html += `<div class="wfb-row"><label class="f" for="wfb-add">Add a step</label><select id="wfb-add" class="field"><option value="">choose…</option>
      ${next.map((x) => refused[baseOf(x)]
        ? `<option value="${esc(x)}" disabled title="${esc(refused[baseOf(x)])}">${esc(x)}: ${esc((t[baseOf(x)] || {}).name || "")} (not for two-component runs)</option>`
        : `<option value="${esc(x)}">${esc(x)}: ${esc((t[baseOf(x)] || {}).name || "")}</option>`).join("")}</select></div>`;
    const refusedNow = [...new Set(cur.map(baseOf))].filter((b) => refused[b]);
    if (refusedNow.length) html += `<p class="sub-line bad" role="alert" data-part="refused-step">${esc(refused[refusedNow[0]])}</p>`;
    html += `<dl class="wfb-glossary">${[...new Set(cur.map(baseOf))].map((b) => `<div><dt>${esc(b)}</dt><dd>${esc((t[b] || {}).help || "")}</dd></div>`).join("")}</dl>`;
    html += `<div class="wfb-row"><label class="f" for="wfb-label">Name <span class="muted small">(optional)</span></label>
      <input id="wfb-label" class="field" value="${esc(state.draft.label || "")}" placeholder="e.g. perovskites-hse" spellcheck="false" maxlength="64">
      <span class="muted small">Shown in the Workflows view. Letters, digits and . _ + - (no spaces).</span></div>`;
    return html;
  }
  function nextTypes(cur) {
    const t = meta.calc_types || {};
    const have = cur.map(baseOf);
    const out = [];
    for (const [b, v] of Object.entries(t)) {
      if (!cur.length) { if (["OPT", "SP"].includes(b)) out.push(b); continue; }
      const deps = v.depends_on || [];
      if (deps.length && !have.some((h) => deps.includes(h))) continue;
      const n = have.filter((h) => h === b).length;
      if (n && !v.repeatable) continue;
      out.push(n ? `${b}${n + 1}` : b);
    }
    return out;
  }

  // the Spin polarized row's value in a later step's badge: the spin treatment, not Yes/No; a row that names
  // its inherited value in the deck's own words (the k-point mesh: "SHRINK 8 16") shows those
  const spinWord = (f, v) => (f.id === "spin" && (v === "Yes" || v === "No") ? (v === "Yes" ? "spin-polarized" : "closed shell")
    : f.default_label && v !== undefined && v !== null && String(v) === String(f.default) ? f.default_label : v);

  function fieldRow(k, r, f) {
    const s = stepOf(k);
    const id = `wfb-f-${k}-${f.id}`.replace(/[^A-Za-z0-9_-]/g, "_").slice(0, 80);
    const editable = !r.editable || r.editable.includes(f.id);
    const follow = r.mode === "opt2d12", first = r.mode === "cif2d12";
    const answered = f.id in (s.answers || {});
    let badge;
    if (first) {
      const was = (r.base_values || {})[f.id];
      if (!editable) badge = { kind: "parent", text: "set by the chain and the structures" };
      else if (answered) badge = { kind: "changed", text: `changed from the workflow default (was ${was ?? "—"})` };
      else badge = { kind: "default", text: "workflow default" };
    } else if (!editable) badge = { kind: "parent", text: `inherited from ${stepName(r.parent)} (each structure's own)` };
    else if (f.changed || answered) badge = { kind: "changed", text: `changed from ${follow ? "the step before" : "default"} (was ${follow ? spinWord(f, f.default) ?? "—" : f.default ?? "—"})` };
    else if (follow) {
      // the parent step's value (shown for the first structure); a deck start's structures each keep their own
      const pv = spinWord(f, f.default ?? f.value);
      const own = state.inputs && state.inputs.type === "d12" ? `, each structure's own; ${r.material_id || "the first"}'s` : "";
      badge = { kind: "parent", text: `inherited from ${stepName(r.parent)}${pv !== undefined && pv !== null && pv !== "" ? ` (${own ? own + ": " : ""}${pv})` : own ? ` (${own.slice(2)})` : ""}` };
    } else badge = { kind: "default", text: "default" };
    let ctl;
    const dis = editable ? "" : " disabled";
    if (f.kind === "menu") {
      const cur = answered ? String(s.answers[f.id]) : f.value;
      // a later step keeps its parent's method (Hartree-Fock or DFT): the Hartree-Fock variants are the first step's choice
      const opts = (f.options || []).filter((o) => !(r.mode === "opt2d12" && o.family === "hf"));
      // a SOC step: no internal basis set (a SOC deck needs explicit basis input), only the functionals the
      // two-component SCF takes (CRYSTAL23 manual ch. 6)
      const sg = r.soc_group || {}, socNow = r.mode === "opt2d12" && sg.applies && socOn(s);
      const keepInternal = sg.basis && sg.basis.parent_kind === "INTERNAL";
      const offWhy = (o) => !socNow ? "" : f.id === "basis_set" && (/^Internal/.test(o.group || "") || (o.head === "Keep the parent's basis set" && keepInternal))
        ? "a spin-orbit (SOC) deck needs an external basis set" : f.id === "functional" && sg.functionals && !sg.functionals.includes(o.head)
          ? "no two-component form in CRYSTAL23 (manual ch. 6)" : "";
      const groups = new Map();
      for (const o of opts) { const g = o.group || ""; if (!groups.has(g)) groups.set(g, []); groups.get(g).push(o); }
      const optHtml = [...groups.entries()].map(([g, os]) => {
        const inner = os.map((o) => { const why = o.head === cur ? "" : offWhy(o); return `<option value="${esc(o.head)}"${o.head === cur ? " selected" : ""}${why ? " disabled" : ""} title="${esc(why || o.text)}">${esc(M.optionLabel(o))}</option>`; }).join("");
        return g ? `<optgroup label="${esc(g)}">${inner}</optgroup>` : inner;
      }).join("");
      const missing = cur && !opts.some((o) => o.head === cur) ? `<option value="${esc(cur)}" selected>${esc(cur)}</option>` : "";
      ctl = `<select id="${id}" data-field="${esc(f.id)}"${dis}>${missing}${optHtml}</select>`;
    } else if (f.kind === "yesno") {
      const on = answered ? [true, "y", "Yes"].includes(s.answers[f.id]) : f.value === "Yes";
      ctl = `<span class="su-switch"><input type="checkbox" role="switch" id="${id}" data-field="${esc(f.id)}" aria-checked="${on}"${on ? " checked" : ""}${dis}><span aria-hidden="true">${on ? "Yes" : "No"}</span></span>`;
    } else {
      const v = answered ? s.answers[f.id] : (f.value ?? "");
      // values the row offers besides typing one (the k-point mesh: auto, the inherited mesh)
      const sug = editable && (f.suggestions || []).length ? `${id}-dl` : "";
      ctl = `<input id="${id}" class="mono" data-field="${esc(f.id)}" value="${esc(v)}" spellcheck="false" autocomplete="off"${sug ? ` list="${sug}"` : ""}${dis}>`
        + (sug ? `<datalist id="${sug}">${f.suggestions.map((x) => `<option value="${esc(x)}"></option>`).join("")}</datalist>` : "");
    }
    return `<div class="su-row${badge.kind === "changed" ? " changed" : ""}${editable ? "" : " locked"}" data-row="${esc(f.id)}">
      <label for="${id}" class="su-l">${esc(f.label)}</label><div class="su-ctl">${ctl}${f.unit ? `<span class="su-unit">${esc(f.unit)}</span>` : ""}</div>
      <div class="su-meta"><span class="su-badge ${badge.kind}">${esc(badge.text)}</span>${badge.kind === "changed" && editable ? `<button type="button" class="su-reset" data-reset="${esc(f.id)}" aria-label="Reset ${esc(f.label)}">Reset</button>` : ""}
      <span class="su-h">${esc(f.hint || "")}</span>${f.note ? `<span class="su-h warn" data-part="field-note">${esc(f.note)}</span>` : ""}</div></div>`;
  }

  function secSettings() {
    const ks = keys();
    if (!state.formStep || !ks.includes(state.formStep)) state.formStep = ks[0];
    const k = state.formStep, i = ks.indexOf(k);
    const defs = (state.defaults && state.defaults.steps) || {};
    let html = `<h3>Settings</h3><p class="wfb-intro">Each step is set up the way Calculation Setup sets up one deck: these are the CLI's own questions.
      Leave a step as it is to use the defaults (the first step) or to keep the settings of the step it is built from (later steps).</p>`;
    html += `<div class="wfb-split"><ul class="wfb-steplist" role="list">${ks.map((x) => {
      const s = state.draft.steps[x] || {};
      const ch = !!(s.settings_text || s.method || socOn(s) || Object.keys(s.answers || {}).length || Object.keys(s.tau || {}).length
        || Object.keys(s.module || {}).length
        || (s.relaxation_time !== undefined && s.relaxation_time !== null && s.relaxation_time !== "default"));
      const o = defs[x] ? defs[x].settings_origin : "";
      const word = ch ? "changed" : x === ks[0] ? (state.inputs && state.inputs.type === "d12" ? "your decks" : "defaults") : (/opt2d3|TAU/.test(o) ? "defaults" : "inherited");
      return `<li><button type="button" class="wfb-stepbtn${x === k ? " on" : ""}" data-form="${esc(x)}" aria-current="${x === k}">
        <b>${esc(stepName(x))}</b><span class="su-badge ${ch ? "changed" : "default"}">${esc(word)}</span></button></li>`;
    }).join("")}</ul><div class="wfb-form" data-step="${esc(k)}">`;
    const r = state.forms[k];
    const s = stepOf(k);
    if (state.formError[k]) html += `<p class="sub-line bad" role="alert">${esc(state.formError[k])}</p>`;
    if (!r && state.formError[k] && !state.formBusy[k]) {
      // the CLI refused (the alert above says why): no spinner; Try Again asks once more
      html += `<p><button type="button" class="tb-btn-text sm" data-form-retry="${esc(k)}">Try Again</button></p>`;
    } else if (!r) {
      html += `<div class="su-loading"><span class="spinner"></span><span>Asking the CLI for ${esc(stepName(k))}'s settings…</span></div>`;
      if (!state.formBusy[k] && !state.formError[k]) setTimeout(() => loadForm(k), 0);
    } else {
      for (const w of ((defs[k] || {}).warnings || [])) html += `<p class="sub-line warn" data-part="soc-warning">${esc(w)}</p>`;
      if (r.mode === "tau") {
        html += tauForm(k, r);
      } else if (r.mode === "module") {
        html += moduleForm(k, r);
      } else if (r.mode === "deck") {
        html += `<p class="wfb-note2">${icon("info")}<span>${esc(r.note || "")} Change a deck in Calculation Setup or in the editor before starting the workflow.</span></p>
          <details class="wfb-det"><summary>${esc(r.deck_name)}</summary><pre class="wfb-pre mono">${esc(r.deck || "")}</pre></details>`;
        html += socGroup(k, r);
      } else {
        const cmd = r.mode === "cif2d12" ? "mace convert" : r.mode === "opt2d12" ? "mace opt2d12" : "mace opt2d3";
        if (r.mode === "opt2d12") html += `<p class="wfb-note2">${icon("info")}<span>${esc(stepName(k))} starts from each structure's ${esc(stepName(r.parent))} result and keeps its settings.
          Each setting below says “inherited” and shows ${esc(stepName(r.parent))}'s value; change one to override it for this step only (Reset goes back to the inherited value).
          You can change the method, the basis set, the DFT grid, the SCF settings and the k-point mesh${r.editable && r.editable.includes("opt_type") ? ", and the optimization type and convergence," : r.editable && r.editable.includes("freq_template") ? ", and the frequency settings," : ""} here; everything else stays as the structure's own ${esc(r.parent.replace(/_\d+$/, ""))} had it.</span></p>`;
        else if (r.standin) html += `<p class="wfb-note2">${icon("info")}<span>Shown for ${esc(r.material_id)}, from a stand-in of its ${esc(stepName(r.parent))} output (no step has run yet).
          The same settings apply to every structure.${r.static_kpath_note ? " " + esc(r.static_kpath_note) : ""}</span></p>`;
        else html += `<p class="wfb-note2">${icon("info")}<span>The settings the workflow converts every CIF with (${esc(cmd)}), shown for ${esc(r.material_id)}.
          They start from the workflow's defaults; what you change here is all that differs.</span></p>`;
        html += relaxGroup(k, r);
        // a later SCF step: the menus the CLI asks on the way to a field (functional category) are not fields of their own
        const shown = (r.form || []).filter((f) => !(r.mode === "opt2d12" && String(f.id).startsWith("q:")));
        for (const [name, rows] of M.sections(shown)) {
          if (!rows.length) continue;
          html += `<div class="su-sec" data-sec="${esc(name)}"><div class="wfb-sec-h">${esc(name)}</div><div class="su-rows">${rows.map((f) => fieldRow(k, r, f)).join("")}</div></div>`;
        }
        html += socGroup(k, r);
        const n = Object.keys(s.answers || {}).length;
        html += `<div class="wfb-row">${n ? `<button type="button" class="tb-btn-text sm" data-reset-all="1">Reset ${plural(n, "change")}</button>` : ""}
          ${state.formBusy[k] ? `<span class="spinner sm"></span><span class="muted small">Updating…</span>` : ""}</div>`;
        if (r.deck) html += `<details class="wfb-det"><summary>${r.standin ? "Projected" : "The"} ${esc(stepName(k))} deck for ${esc(r.material_id || "the first structure")}</summary><pre class="wfb-pre mono">${esc(r.deck)}</pre></details>`;
      }
    }
    html += `</div></div>`;
    return html;
  }

  /** TRANSPORT's Relaxation time group: CRYSTAL's 10 fs, a value, or τ from a TAU step before it. */
  function relaxGroup(k, r) {
    const g = r.relaxation;
    if (!g) return "";
    const s = stepOf(k), id = `wfb-rt-${k}`.replace(/[^A-Za-z0-9_-]/g, "_");
    const v = g.value === "auto" ? "auto" : typeof s.relaxation_time === "number" ? "fixed" : "default";
    const help = v === "auto" ? `τ from ${stepName(g.tau_step)}: strained single points of the SP, fitted to the deformation potential. When τ cannot be computed (a metal, failed runs, no effective mass) TRANSPORT runs with 10 fs and the reason is recorded.`
      : v === "fixed" ? "A τ you know (from mace tau fit, or a measured mobility). σ and κ scale with τ; the Seebeck coefficient does not."
        : "CRYSTAL's default RELAXTIM. σ and κ scale with τ; the Seebeck coefficient does not.";
    return `<div class="su-sec" data-sec="relaxation"><div class="wfb-sec-h">Relaxation time</div><div class="su-rows">
      <div class="su-row${v === "default" ? "" : " changed"}"><label class="su-l" for="${id}">RELAXTIM</label><div class="su-ctl">
        <select id="${id}" data-relax="mode">${[["default", "Default (10 fs)"], ["fixed", "Fixed"], ["auto", "From deformation potentials"]].map(([x, l]) =>
          `<option value="${x}"${x === v ? " selected" : ""}>${l}</option>`).join("")}</select>
        ${v === "fixed" ? `<input id="${id}-fs" class="mono" data-relax="fs" value="${esc(s.relaxation_time)}" aria-label="Relaxation time in fs" spellcheck="false"><span class="su-unit">fs</span>` : ""}</div>
        <div class="su-meta"><span class="su-badge ${v === "default" ? "default" : "changed"}">${v === "default" ? "default" : v === "auto" ? "automatic" : "set here"}</span><span class="su-h">${esc(help)}</span></div></div></div></div>`;
  }

  /** A TAU step's settings: what it runs, its options (each with its default) and what it is missing. */
  function tauForm(k, r) {
    const s = stepOf(k), o = r.options || {}, d = r.defaults || {}, set = s.tau || {};
    const id = (x) => `wfb-tau-${x}-${k}`.replace(/[^A-Za-z0-9_-]/g, "_");
    const row = (f, label, ctl, hint) => `<div class="su-row${f in set ? " changed" : ""}" data-row="tau-${f}"><label class="su-l" for="${id(f)}">${esc(label)}</label><div class="su-ctl">${ctl}</div>
      <div class="su-meta"><span class="su-badge ${f in set ? "changed" : "default"}">${f in set ? "changed" : "default"}</span>${f in set ? `<button type="button" class="su-reset" data-tau-reset="${f}" aria-label="Reset ${esc(label)}">Reset</button>` : ""}<span class="su-h">${esc(hint)}</span></div></div>`;
    const txt = (f, v) => `<input id="${id(f)}" class="mono" data-tau="${f}" value="${esc(v)}" spellcheck="false" autocomplete="off">`;
    let html = r.refusal ? `<p class="sub-line bad" role="alert">${esc(r.refusal)}</p>`
      : `<p class="wfb-note2">${icon("info")}<span>${esc(plural(r.n_runs, "strained single point"))} per structure (${esc((r.axes || []).join(", "))}: ${esc(r.axes_reason || "")}), shown for ${esc(r.material_id)}.
        They run ${(r.masses || []).length ? `beside ${esc(r.masses.join(" and "))} ` : ""}with the SP's resources and 1.5× its time limit; when all have finished, E1, C and the masses give τ${r.transport ? ` for ${esc(stepName(r.transport))}` : ""}.</span></p>`;
    if (!r.transport) html += `<p class="sub-line warn">No TRANSPORT step after this one: τ is computed and saved (tau_summary.json), nothing runs with it.</p>`;
    if (!(r.masses || []).length && o.m_star === "auto") html += `<p class="sub-line warn" data-part="tau-masses">The effective masses come from BAND and DOSS, and this chain has neither: τ falls back to ${esc(o.fallback_fs)} fs.
      <button type="button" class="tb-btn-text sm" data-add-masses="1">Add BAND and DOSS</button> or set m* below.</p>`;
    html += `<div class="su-sec" data-sec="tau"><div class="wfb-sec-h">Relaxation time (deformation potentials)</div><div class="su-rows">`;
    html += row("carrier", "Carrier", `<select id="${id("carrier")}" data-tau="carrier">${[["auto", "Auto (electrons unless MACE knows the doping)"], ["electron", "Electrons"], ["hole", "Holes"]].map(([x, l]) =>
      `<option value="${x}"${o.carrier === x ? " selected" : ""}>${l}</option>`).join("")}</select>`, "RELAXTIM is one number: τ of this carrier along the first axis. Every carrier and axis is saved.");
    html += row("axes", "Axes", txt("axes", o.axes === "auto" ? "auto" : (o.axes || []).join(" ")), "auto: from the symmetry (x for cubic; x z for tetragonal or hexagonal), or x y z");
    html += row("strains", "Strains", txt("strains", (o.strains || []).join(" ")), "fractions (0.01 is 1%), the unstrained run is added");
    html += row("m_star", "m*", txt("m_star", o.m_star), "auto: from the workflow's BAND and DOSS runs, or a mass in mₑ");
    html += row("fallback_fs", "Fallback τ", `${txt("fallback_fs", o.fallback_fs)}<span class="su-unit">fs</span>`, "used when τ cannot be computed, with the reason recorded");
    html += row("strict", "Hold TRANSPORT", `<span class="su-switch"><input type="checkbox" role="switch" id="${id("strict")}" data-tau="strict" aria-checked="${!!o.strict}"${o.strict ? " checked" : ""}><span aria-hidden="true">${o.strict ? "Yes" : "No"}</span></span>`,
      "when τ falls back or the E1 fit is poor (R² < 0.9), TRANSPORT is not run");
    return html + `</div></div>`;
  }

  /** An ELASTIC, EOS, QHA or CPHF step: the module block (Calculation Setup's module fields, the CLI's
   * --elastic-settings keys), each field with its default, and the deck mace opt2d12 writes from a stand-in. */
  function moduleForm(k, r) {
    const s = stepOf(k), set = s.module || {};
    const mode = (r.modes || []).find((m) => m.mode === r.module) || (r.modes || [])[0] || { keys: [], mode: r.module };
    const keys = new Set(mode.keys || []);
    const id = (x) => `wfb-mod-${x}-${k}`.replace(/[^A-Za-z0-9_-]/g, "_");
    const isSet = (f) => (f === "relax" ? "nuchess" in set || "clampion" in set : f in set);
    const row = (f, label, ctl, hint) => `<div class="su-row${isSet(f) ? " changed" : ""}" data-row="mod-${f}"><label class="su-l" for="${id(f)}">${esc(label)}</label><div class="su-ctl">${ctl}</div>
      <div class="su-meta"><span class="su-badge ${isSet(f) ? "changed" : "default"}">${isSet(f) ? "changed" : f === "print" ? "MACE default" : "CRYSTAL default"}</span>${isSet(f) ? `<button type="button" class="su-reset" data-mod-reset="${f}" aria-label="Reset ${esc(label)}">Reset</button>` : ""}<span class="su-h">${esc(hint)}</span></div></div>`;
    const shown = (v) => (Array.isArray(v) ? v.map((x) => (Array.isArray(x) ? x.join(" ") : x)).join(Array.isArray(v[0]) ? "; " : " ") : v ?? "");
    let html = `<p class="wfb-note2">${icon("info")}<span>${esc(stepName(k))} starts from each structure's ${esc(stepName(r.parent))} result: mace opt2d12 writes the ${esc(baseOf(r.calc_type))} deck
      with its geometry, functional, basis set and SCF settings (SCF TOLDEE at least ${esc(r.toldee)}, MAXCYCLE at least 200). Nothing is built from this step. Shown for ${esc(r.material_id)}, from a stand-in of its ${esc(stepName(r.parent))} output.</span></p>`;
    if (r.refusal) html += `<p class="sub-line bad" role="alert" data-part="module-refusal">${esc(r.refusal)}</p>`;
    html += `<div class="su-sec" data-sec="module"><div class="wfb-sec-h">${esc(mode.title || r.module)}</div><div class="su-rows">`;
    if ((r.modes || []).length > 1) {
      html += row("mode", "Module", `<select id="${id("mode")}" data-modf="mode">${r.modes.map((m, i) =>
        `<option value="${esc(m.mode)}"${m.mode === r.module ? " selected" : ""}>${i + 1} - ${esc(m.title)}</option>`).join("")}</select>`, "ELASTCON when not changed");
    }
    if (mode.what) html += `<p class="su-note su-mod-what">${icon("info")}<span>${esc(mode.what.charAt(0).toUpperCase() + mode.what.slice(1))}. ${esc(mode.manual || "")}.${mode.pcrystal_only ? " Runs with Pcrystal only (not MPPcrystal)." : ""}</span></p>`;
    const relaxable = keys.has("nuchess") || keys.has("clampion");
    for (const [key, label0, kind, ph0] of MODULE_FIELDS) {
      // RESTART continues one run from its own info file: not a workflow step's setting
      if (key === "restart" || !(key === "relax" ? relaxable : keys.has(key))) continue;
      let label = label0, ph = ph0;
      if (mode.mode === "CPHF" && key === "maxcycle") { label = "SC-CP iterations (MAXCYCLE)"; ph = "100"; }
      if (mode.mode === "CPHF" && key === "tolalpha") ph = "3 (CPHF)";
      if (mode.mode === "CPHF" && key === "fmixing") ph = "no mixing";
      if (key === "stepsize" && mode.step !== undefined) ph = String(mode.step);
      let ctl;
      if (kind === "deform") {
        const on = Array.isArray(set.deform) ? set.deform : [];
        ctl = `<span class="su-mod-checks" role="group" id="${id(key)}" data-deform="1">${MODULE_VOIGT.map((v, i) =>
          `<label class="su-check"><input type="checkbox" data-modf="deform" data-d="${i}"${on[i] ? " checked" : ""}> ${v}</label>`).join("")}</span>`;
      } else if (kind === "relax") {
        const cur = set.nuchess ? "nuchess" : set.clampion ? "clampion" : "opt";
        ctl = `<select id="${id(key)}" data-modf="relax">${[["opt", "Optimize atoms in each strained cell (default)"], ["nuchess", "Internal-strain Hessian (NUCHESS, recommended)"],
          ["clampion", "Clamped ions (CLAMPION)"]].map(([v, l]) => `<option value="${v}"${cur === v ? " selected" : ""}>${l}</option>`).join("")}</select>`;
      } else {
        ctl = `<input id="${id(key)}" class="mono" data-modf="${key}" data-kind="${kind}" value="${esc(shown(set[key]))}" placeholder="${esc(ph)}" spellcheck="false" autocomplete="off">`;
      }
      html += row(key, label, ctl, kind === "deform" ? ph : "");
    }
    for (const [key, label] of MODULE_FLAGS) {
      if (!keys.has(key)) continue;
      const on = key === "print" ? set.print !== false : !!set[key];         // PRINT: on unless unticked
      html += row(key, label, `<span class="su-switch"><input type="checkbox" role="switch" id="${id(key)}" data-modf="${key}" data-kind="flag" aria-checked="${on}"${on ? " checked" : ""}><span aria-hidden="true">${on ? "Yes" : "No"}</span></span>`, "");
    }
    html += `</div></div><p class="wfb-note2" data-part="module-describe">${icon("info")}<span>${esc(r.describe || "")}</span></p>`;
    const n = Object.keys(set).length;
    html += `<div class="wfb-row">${n ? `<button type="button" class="tb-btn-text sm" data-mod-reset="*">Reset ${plural(n, "change")}</button>` : ""}
      ${state.formBusy[k] ? `<span class="spinner sm"></span><span class="muted small">Updating…</span>` : ""}</div>`;
    if (r.deck) html += `<details class="wfb-det"><summary>Projected ${esc(stepName(k))} deck for ${esc(r.material_id || "the first structure")}</summary><pre class="wfb-pre mono">${esc(r.deck)}</pre></details>`;
    return html;
  }

  /** The Spin-orbit coupling group of a step form: on SP steps a switch, the ECP route, the starting guess and
   * SMEAR, with the CLI's SOC lines, warnings and refusal from the step form's run; elsewhere the group disabled,
   * with the CLI's reason. The values become the step's spec overrides (the keys of opt2d12's settings file). */
  function socGroup(k, r) {
    const g = r.soc_group;
    if (!g) return "";
    const s = stepOf(k), v = s.soc || {}, on = g.applies && socOn(s);
    const id = (x) => `wfb-soc-${x}-${k}`.replace(/[^A-Za-z0-9_-]/g, "_");
    const why = id("why"), rwhy = id("rwhy");
    const dis = g.applies ? "" : " disabled";
    const off = on ? "" : " disabled";
    const route = v.soc_basis ? "basis" : v.soc_ecp || "inpsoc";
    const first = r.mode === "cif2d12";
    const badge = (set, def) => `<span class="su-badge ${set ? "changed" : "default"}">${esc(set ? "set here" : def)}</span>`;
    const row = (key, label, ctl, set, def, hint, reset) => `<div class="su-row${set ? " changed" : ""}" data-row="soc-${key}">
      <label for="${id(key)}" class="su-l">${esc(label)}</label><div class="su-ctl">${ctl}</div>
      <div class="su-meta">${badge(set, def)}${set && reset && g.applies ? `<button type="button" class="su-reset" data-soc-reset="${key}" aria-label="Reset ${esc(label)}">Reset</button>` : ""}<span class="su-h">${esc(hint)}</span></div></div>`;
    let html = `<fieldset class="su-sec wfb-soc${g.applies ? "" : " off"}" data-part="soc" data-step="${esc(k)}"${dis}${g.applies ? "" : ` aria-describedby="${why}"`}>
      <legend class="wfb-sec-h">Spin-orbit coupling</legend>`;
    if (!g.applies) html += `<p class="wfb-soc-why" id="${why}" data-part="soc-reason">${icon("info")}<span>${esc(g.reason || "")}</span></p>`;
    else html += `<p class="muted small wfb-soc-intro">A two-component (TWOCOMPON) single point: ${esc(first ? "mace convert" : "mace opt2d12")} writes the SOC deck from these choices, the keys of its settings file.</p>`;
    html += `<div class="su-rows">`;
    html += row("on", "Spin-orbit coupling", `<span class="su-switch"><input type="checkbox" role="switch" id="${id("on")}" data-soc="soc" aria-checked="${on}"${on ? " checked" : ""}${dis}${g.applies ? "" : ` aria-describedby="${why}"`}><span aria-hidden="true">${on ? "On" : "Off"}</span></span>`,
      on, "off (scalar deck)", g.applies ? 'written as "soc": true' : "", false);
    // STUTSC refused for an element of the structures (the CLI's soc_ecp.py table and words, as Setup grays it out)
    const st = g.stutsc && g.stutsc.refused && g.stutsc.refused.length ? g.stutsc : null, swhy = id("swhy");
    const unread = (g.stutsc && g.stutsc.unread) || [];      // elements not read (e.g. no ase for a CIF): STUTSC unchecked, said so
    const stLabel = (o) => {
      if (o.v !== "stutsc" || !st) return o.label;
      const els = st.refused.map((x) => x.symbol).join(", "), n = new Set(st.refused.flatMap((x) => x.files)).size;
      return `${o.label} (refused for ${els}${st.all ? "" : ` in ${n} of ${st.files} structures`})`;
    };
    html += row("route", "Spin-orbit ECP", `<select id="${id("route")}" data-soc="route"${dis || off}${(st || unread.length) && on ? ` aria-describedby="${swhy}"` : ""}>${SOC_ROUTES.filter((o) => o.v === "basis" ? (g.soc_basis_families || []).length : (g.ecp_forms || []).includes(o.v))
      .map((o) => `<option value="${o.v}"${o.v === route ? " selected" : ""}${o.v === "stutsc" && st && st.all && route !== "stutsc" ? " disabled" : ""}>${esc(stLabel(o))}</option>`).join("")}</select>`,
      on && route !== "inpsoc", "INPSOC (default)", 'written as "soc_ecp" or "soc_basis"', true);
    if (on && g.basis && g.basis.note) html += `<p class="wfb-soc-why sm" data-part="soc-basis">${icon("info")}<span>${esc(g.basis.note)}</span></p>`;
    if ((st || unread.length) && on) html += `<p class="wfb-soc-why sm" id="${swhy}">${(st ? st.refused : []).map((x) => `${esc(x.symbol)}: ${esc(x.why)}`)
      .concat(unread.map((u) => `STUTSC not checked: ${esc(u.why)}`)).join("<br>")}</p>`;
    html += row("guess", "Starting guess", `<select id="${id("guess")}" data-soc="guess" aria-describedby="${g.applies ? rwhy : why}"${dis || off}><option value="" selected>From scratch</option>${(g.guesses || []).map((x) =>
      `<option value="${esc(x)}" disabled>${esc(x.toUpperCase())} (restart; not here)</option>`).join("")}</select>`,
      false, "from scratch", "", false);
    if (g.applies) html += `<p class="wfb-soc-why sm" id="${rwhy}">${esc(g.restart_reason || "")}</p>`;
    const ms = g.metal_smear;
    html += row("smear", "SMEAR width", `<input id="${id("smear")}" class="mono" data-soc="smear" inputmode="decimal" value="${esc(on && v.smear !== undefined && v.smear !== null ? v.smear : "")}" placeholder="none" spellcheck="false" autocomplete="off"${dis || off}><span class="su-unit">Ha</span>`,
      on && v.smear !== undefined && v.smear !== null && v.smear !== "", "none",
      `empty: no SMEAR. For a metal, ${ms ?? 0.005} Ha is the width the HPCC Au runs converged with`, true);
    html += `</div>`;
    const sr = r.soc || {};
    if (on) {
      const probs = (r.problems || []).filter((p) => p.field === "soc");
      for (const p of probs) html += `<p class="sub-line ${p.severity === "error" ? "bad" : "warn"}"${p.severity === "error" ? ' role="alert"' : ""} data-part="soc-${p.severity}">${esc(p.message)}</p>`;
      if ((sr.changes || []).length) html += `<details class="wfb-det"><summary>What the CLI changes for SOC (${sr.changes.length})</summary><pre class="wfb-pre mono" data-part="soc-changes">${esc(sr.changes.join("\n"))}</pre></details>`;
    }
    return html + `</fieldset>`;
  }
  function socEdit(t) {
    const k = state.formStep, s = stepOf(k);
    const v = s.soc = s.soc || {};
    if (t.dataset.soc === "soc") {
      const gb = ((state.forms[k] || {}).soc_group || {}).basis;
      if (t.checked) {
        v.soc = true;
        // the SOC deck's basis, as the server works it out from the step's (inherited) basis: an internal set
        // becomes its external library, and the heavy atoms take the official SOC set when it covers them
        s.answers = s.answers || {};
        if (gb && gb.basis_set && !s.answers.basis_set) { s.answers.basis_set = gb.basis_set; s.socAuto = true; }
        if (gb && gb.soc_basis && !v.soc_ecp && !v.soc_basis) v.soc_basis = gb.soc_basis;
      } else {
        s.soc = null;
        if (s.socAuto && s.answers) delete s.answers.basis_set;
        s.socAuto = false;
      }
    }
    else if (t.dataset.soc === "route") {
      delete v.soc_ecp; delete v.soc_basis;
      if (t.value === "stutsc") v.soc_ecp = "stutsc";
      else if (t.value === "basis") v.soc_basis = "pob-tzvp-rev2-soc";
    } else if (t.dataset.soc === "smear") {
      const x = t.value.trim();
      if (x === "") delete v.smear; else v.smear = /^[0-9.eE+-]+$/.test(x) && Number.isFinite(Number(x)) ? Number(x) : x;
    } else return;
    changed("settings"); loadForm(k);
  }

  function memOf(res) { return res.memory ? { v: res.memory, per: "job" } : res.memory_per_cpu ? { v: res.memory_per_cpu, per: "core" } : { v: "", per: "job" }; }
  function secResources() {
    const ks = keys();
    const defs = (state.defaults && state.defaults.steps) || {};
    const ex = state.draft.execution || (state.draft.execution = {});
    let html = `<h3>Resources</h3><p class="wfb-intro">What each step's job asks SLURM for. The values come from MACE's defaults and your site settings;
      type a value to change it for this workflow only (it then says “set here”).</p>`;
    if (state.defaultsError) html += `<p class="sub-line bad" role="alert">${esc(state.defaultsError)}</p>`;
    html += `<div class="wfb-grid-wrap"><table class="wfb-grid"><caption class="sr-only">Resources of each step</caption><thead><tr><th scope="col">Step</th>${RES_COLS.map((c) =>
      `<th scope="col" title="${esc(c.hint)}">${esc(c.label)}</th>`).join("")}</tr></thead><tbody>`;
    for (const k of ks) {
      const d0 = defs[k] || { resources: {}, origins: {} };
      const mine = stepOf(k).resources || {};
      const checks = d0.checks || [];
      html += `<tr data-step="${esc(k)}"><th scope="row">${esc(stepName(k))}</th>`;
      for (const c of RES_COLS) {
        let val, origin, set;
        if (c.k === "mem") {
          const m = memOf(d0.resources || {});
          const mm = mine.memory ? { v: mine.memory, per: "job" } : mine.memory_per_cpu ? { v: mine.memory_per_cpu, per: "core" } : null;
          val = mm ? mm.v : m.v;
          set = !!mm;
          origin = set ? ORIGIN("spec") : ORIGIN((d0.origins || {})[m.per === "job" ? "memory" : "memory_per_cpu"]);
          const per = mm ? mm.per : m.per;
          html += `<td><div class="wfb-cell"><input class="field mono" data-res="mem" value="${esc(val || "")}" placeholder="${esc(c.ph)}" aria-label="${esc(stepName(k))} memory" spellcheck="false">
            <select class="field" data-res="mem-per" aria-label="${esc(stepName(k))} memory per"><option value="job"${per === "job" ? " selected" : ""}>per job</option><option value="core"${per === "core" ? " selected" : ""}>per core</option></select></div>
            ${badgeHtml(origin, set, "mem")}</td>`;
          continue;
        }
        set = mine[c.k] !== undefined && mine[c.k] !== "";
        val = set ? mine[c.k] : (d0.resources || {})[c.k];
        // the account: exactly what the job script gets (the CLI names it per step; null: no -A line at all)
        const acct = c.k === "account" && !set && "account" in (d0.resources || {});
        // the account and partition every job script gets from the site settings (mace site), when the plan sets none
        const siteV = !acct && !set && (val === undefined || val === null || val === "") ? siteValue(c.k) : null;
        origin = set ? ORIGIN("spec") : val !== undefined && val !== null && val !== "" ? ORIGIN((d0.origins || {})[c.k])
          : acct ? { kind: "default", text: "no account line" }
            : siteV ? siteV.origin : { kind: "default", text: c.k === "exclude" ? "none" : "SLURM default" };
        const bad = checks.find((x) => x.field === c.k);
        const ph = acct && !val ? "SLURM default" : siteV ? siteV.value : c.ph;
        const warn = acct && val && !set && accountWarning(val, (d0.origins || {})[c.k]);
        html += `<td><input class="field mono${c.k === "ntasks" ? " num" : ""}" data-res="${c.k}" value="${esc(val ?? "")}" placeholder="${esc(ph)}" aria-label="${esc(stepName(k))} ${esc(c.label.toLowerCase())}"${bad ? ' aria-invalid="true"' : ""} spellcheck="false"
          ${acct && !val ? ' title="The job script has no account line: SLURM charges your default account"' : ""}>
          ${badgeHtml(origin, set, c.k)}${bad ? `<div class="wfb-cell-err" role="alert">${esc(bad.message)}</div>` : ""}${warn ? `<div class="wfb-cell-warn" data-part="account-warning">${esc(warn)}</div>` : ""}</td>`;
      }
      html += "</tr>";
    }
    html += `</tbody></table></div>`;
    if (state.defaultsBusy) html += `<p class="muted small"><span class="spinner sm"></span> Checking with the CLI…</p>`;
    html += `<div class="wfb-label">Submitting</div><div class="wfb-opts">
      <label class="tog"><input type="checkbox" data-ex="array"${ex.array ? " checked" : ""}><span>Submit the first step as a job array (one SLURM job with a task per structure)</span></label>
      <label class="f inline">At most <input class="field num" type="number" min="1" data-ex="max_running" value="${esc(ex.max_running || "")}"${ex.array ? "" : " disabled"} aria-label="Array tasks running at once"> array tasks running at once</label>
      <label class="f inline">Keep at most <input class="field num" type="number" min="1" data-ex="max_jobs" value="${esc(ex.max_jobs || "")}" placeholder="all" aria-label="Jobs queued at most" title="Empty: the workflow shares your queue budget with every other (mace queue)"> of this workflow's jobs queued</label>
      <label class="tog"><input type="checkbox" data-ex="auto_progression"${ex.auto_progression === false ? "" : " checked"}><span>Start each next step by itself when the step before finishes (off: every finished step waits until you start the next from the Workflows view, “Process now”)</span></label></div>`;
    return html;
  }
  /** The built-in account kept where MACE could not check membership, off MSU HPCC: since 1.1.5 only
   * where there is no SLURM at all (site.keeps_builtin_account; with SLURM the scripts get no -A line). */
  function accountWarning(v, origin) {
    const cl = (caps.site || {}).cluster || {};
    if (origin !== "builtin" || cl.msu_hpcc !== false) return "";
    return `MACE's built-in account (${v}) is kept because there is no SLURM here to check your accounts; a cluster other than MSU HPCC may refuse it. Type your account, or set [slurm] account in your site settings (mace site).`;
  }
  function siteValue(k) {
    if (!["account", "partition"].includes(k)) return null;
    const v = ((caps.site || {})[k]) || {};
    if (!v.value) return null;
    const src = String(v.source || "");
    return { value: v.value, origin: { kind: "site", text: src === "built-in" ? "MACE default" : src.startsWith("env") ? "environment variable" : "site setting" } };
  }
  function badgeHtml(origin, set, k) {
    return `<div class="wfb-origin"><span class="su-badge ${origin.kind === "you" ? "changed" : "default"}" data-origin="${esc(origin.kind)}">${esc(origin.text)}</span>${set
      ? `<button type="button" class="su-reset" data-res-reset="${esc(k)}">Reset</button>` : ""}</div>`;
  }

  function secReview() {
    const p = state.preview;
    let html = `<h3>Review and submit</h3>`;
    if (caps.network) html += `<p class="sub-line bad">Submitting is turned off while MACE GUI is open to the network.</p>`;
    if (state.previewing) return html + `<div class="su-loading"><span class="spinner"></span><span>Running the CLI in a copy of the folder (nothing is submitted)…</span></div>`;
    if (state.previewError) return html + `<p class="sub-line bad" role="alert">${esc(state.previewError)}</p><p><button type="button" class="tb-btn-text" data-act="preview">Try Again</button></p>`;
    if (!p) { setTimeout(() => { if (!state.previewing && !state.preview) preview(); }, 0); return html; }
    const n = p.n_jobs || 0;
    const later = seq().slice(1);
    // what goes out now, by step: a deck that finished already sends its next step
    const byType = {};
    for (const f of p.first_step || []) (byType[f.calc_type || seq()[0]] = byType[f.calc_type || seq()[0]] || []).push(f.material_id);
    const what = Object.keys(byType).length ? Object.entries(byType).map(([t, ms]) => `${esc(t)} for ${esc(ms.join(", "))}`).join("; ")
      : `${esc(seq()[0])} for ${esc(((p.inputs || {}).materials || []).map((m) => m.material_id).join(", "))}`;
    html += `<p class="wfb-summary">${state.running ? "Submitting…" : `Submit sends <b>${plural(n, "job")}</b> now: ${what}.`}
      ${later.length ? ` When a structure's job finishes, its next step (${esc(later.join(", "))}) is submitted by itself.` : ""}</p>`;
    if ((p.inputs || {}).type === "d12") html += `<p class="muted small" data-part="step-folders">Each deck runs in its own folder under workflow_outputs/; the decks, outputs and job scripts in ${esc(p.folder_rel || "this folder")} are left as they are.</p>`;
    for (const f of p.finished || []) html += `<p class="sub-line" data-part="finished">${esc(f.deck)} finished already (${esc(f.out_rel)}): it is not run again, and its next step starts from that output.</p>`;
    if ((p.blocked || []).length) html += `<p class="sub-line bad" role="alert" data-part="blocked">${esc(p.blocked.join(", "))} already ${p.blocked.length === 1 ? "has" : "have"} unfinished steps in a workflow in this folder.
      Wait for them, or cancel that workflow, before starting these structures again.</p>`;
    if (p.code) html += `<p class="sub-line bad" role="alert">The CLI refused this workflow${p.refused ? `: ${esc(p.refused)}` : ""}.</p><pre class="sub-log">${esc((p.cli_log || "").split("\n").slice(-30).join("\n"))}</pre>`;
    for (const w of p.warnings || []) html += `<p class="sub-line warn">${esc(w)}</p>`;
    if (state.runError) html += `<p class="sub-line bad" role="alert" data-part="run-error">${esc(state.runError)}</p>`;
    html += `<div class="wfb-label">The commands Submit runs</div><p class="muted small">In ${esc(p.folder_rel || "this folder")}. You can run them yourself in a terminal; the plan's name gets the time you submit.</p>
      <div class="wfb-cmds">${(p.commands || []).map((c, i) => `<div class="wfb-cmd"><code class="mono" data-part="command-${i}">${esc(c)}</code><button type="button" class="tb-btn-text sm" data-copy="${esc(c)}">Copy</button></div>`).join("")}</div>`;
    html += `<div class="wfb-label">First step: exactly what is submitted</div>`;
    for (const f of p.first_step || []) {
      const r = f.resources || {};
      const line = [r.ntasks && `${r.ntasks} cores`, r.mem && `${r.mem} memory`, r.mem_per_cpu && `${r.mem_per_cpu} per core`, r.walltime && `time limit ${r.walltime}`, r.partition && `partition ${r.partition}`, r.account ? `account ${r.account}` : "SLURM's default account"].filter(Boolean).join(", ");
      html += `<details class="wfb-det" data-first="${esc(f.material_id)}"><summary><b>${esc(f.material_id)}</b> · ${esc(f.calc_type || seq()[0])} · <span class="muted">${esc(line)}</span>${f.array ? ' <span class="chip">job array</span>' : ""}</summary>
        ${f.deck_text ? `<div class="wfb-label sm">Deck ${esc(f.deck_name)}</div><pre class="wfb-pre mono">${esc(f.deck_text)}</pre>` : ""}
        <div class="wfb-label sm">Job script</div><pre class="wfb-pre mono">${esc(f.script_text || "")}</pre></details>`;
    }
    if ((p.projected || []).length) {
      html += `<div class="wfb-label">Later steps: projected for ${esc(p.projected[0].material_id)}</div><p class="muted small">MACE writes these when the step before finishes, from its real output.
        Here the engine wrote them from a stand-in output, so the structure and anything read from the output are placeholders; the settings and job resources are the ones the workflow uses.</p>`;
      for (const n of p.projected_notes || []) html += `<p class="muted small" data-part="projected-note">${esc(n)}</p>`;
      for (const s of p.projected) {
        const r = s.resources || {};
        const line = [r.ntasks && `${r.ntasks} cores`, r.mem && `${r.mem} memory`, r.mem_per_cpu && `${r.mem_per_cpu} per core`, r.walltime && `time limit ${r.walltime}`].filter(Boolean).join(", ");
        html += `<details class="wfb-det" data-projected="${esc(s.step)}"><summary><b>${esc(stepName(s.step))}</b> <span class="chip projected">projected</span> · from ${esc(stepName(s.parent))} · <span class="muted">${esc(line)}</span></summary>
          ${s.deck_text ? `<div class="wfb-label sm">Deck ${esc(s.deck_name)}</div><pre class="wfb-pre mono">${esc(s.deck_text)}</pre>` : ""}
          ${s.script_text ? `<div class="wfb-label sm">Job script</div><pre class="wfb-pre mono">${esc(s.script_text)}</pre>` : ""}</details>`;
      }
    }
    html += `<div class="wfb-label">Files</div>`;
    if (p.spec) html += `<details class="wfb-det"><summary>The workflow spec the GUI writes · <span class="mono small">${esc(p.spec.rel)}</span></summary><pre class="wfb-pre mono">${esc(p.spec.text)}</pre>
      ${Object.entries(p.spec.settings || {}).map(([n, t]) => `<div class="wfb-label sm">${esc(n)}</div><pre class="wfb-pre mono">${esc(t)}</pre>`).join("")}</details>`;
    if (p.plan) html += `<details class="wfb-det"><summary>The plan the CLI writes · <span class="mono small">${esc(p.plan.rel)}</span></summary><pre class="wfb-pre mono">${esc(p.plan.text)}</pre></details>`;
    if ((p.workflow_scripts || []).length) html += `<details class="wfb-det"><summary>Job-script templates for every step (${p.workflow_scripts.length})</summary>${p.workflow_scripts.map((w) =>
      `<div class="wfb-label sm mono">${esc(w.rel)}</div><pre class="wfb-pre mono">${esc(w.text)}</pre>`).join("")}</details>`;
    html += `<details class="wfb-det"><summary>Everything written in the folder (${(p.writes || []).length})</summary><pre class="wfb-pre mono">${esc((p.writes || []).join("\n"))}</pre></details>`;
    if (!p.code) html += `<p class="muted small">Then follow it under View › Workflows: every structure's steps, their jobs and the time left.</p>`;
    return html;
  }

  /** A clone says where it was copied from, and what could not be copied. */
  function cloneNote() {
    const c = state.clone;
    if (!c) return "";
    const from = c.source === "builder" ? "as it was set up in New Workflow" : c.source === "spec" ? "from its workflow spec" : "from its plan";
    return `<div class="wfb-clone-note" role="note" data-part="clone-note">${icon("copy")}<div><b>A copy of an existing workflow</b>, ${esc(from)}.
      Nothing is submitted until you click Submit; change anything first.${(c.notes || []).map((n) => `<div class="small">${esc(n)}</div>`).join("")}</div></div>`;
  }
  function render() {
    const keep = document.activeElement && sec.contains(document.activeElement) ? document.activeElement : null;
    const keepSel = keep && (keep.id ? `#${CSS.escape(keep.id)}` : keep.dataset.res ? `tr[data-step="${CSS.escape(keep.closest("tr").dataset.step)}"] [data-res="${keep.dataset.res}"]`
      : keep.dataset.file ? `[data-file="${CSS.escape(keep.dataset.file)}"]` : keep.dataset.ex ? `[data-ex="${keep.dataset.ex}"]` : keep.dataset.form ? `[data-form="${CSS.escape(keep.dataset.form)}"]` : null);
    const caret = keep && typeof keep.selectionStart === "number" ? [keep.selectionStart, keep.selectionEnd] : null;
    const open = new Set([...sec.querySelectorAll("details[open] > summary")].map((x) => x.textContent));
    renderNav();
    const html = cloneNote() + [secInputs, secChain, secSettings, secResources, secReview][state.sec]();
    sec.innerHTML = html;
    sec.querySelectorAll("details > summary").forEach((x) => { if (open.has(x.textContent)) x.parentElement.open = true; });
    if (keepSel) { const el = sec.querySelector(keepSel); if (el) { el.focus({ preventScroll: true }); if (caret && el.setSelectionRange) try { el.setSelectionRange(...caret); } catch (_) { /* not text */ } } }
    renderFoot();
  }
  function go(i) {
    if (i < 0 || i >= SECTIONS.length || (i > 0 && !state.inputs)) return;
    state.sec = i;
    state.visited.add(i);
    if (i >= 2 && !state.defaults && !state.defaultsBusy) loadDefaults();
    render();
    sec.scrollTop = 0;
    sec.focus({ preventScroll: true });
  }

  /* ------------------------------------------------------------ events */
  function pickFiles(list) {
    scanInputs(list, null);
  }
  nav.addEventListener("click", (e) => { const b = e.target.closest("[data-sec]"); if (b && !b.disabled) go(Number(b.dataset.sec)); });
  btnBack.addEventListener("click", () => go(state.sec - 1));
  btnNext.addEventListener("click", () => go(state.sec + 1));
  btnSubmit.addEventListener("click", submit);
  d.querySelector("[data-close]").addEventListener("click", () => { if (!state.running) d.close(); });
  sec.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    if (b.dataset.all) {
      const folder = sec.querySelector("#wfb-folder").value;
      const files = (H.files() || []).filter((f) => dirOf(f.path) === folder);
      const want = b.dataset.all === "none" ? [] : files.filter((f) => new RegExp(`\\.${b.dataset.all}$`, "i").test(f.path)).map((f) => f.path).sort();
      pickFiles(want);
    } else if (b.dataset.template) {
      const t = (meta.templates || []).find((x) => x.name === b.dataset.template);
      if (t) { state.draft.template = t.name; state.draft.sequence = [...t.sequence]; state.draft.steps = {}; state.forms = {}; changed("chain"); render(); }
    } else if (b.dataset.remove !== undefined) {
      // the other steps keep their settings and resources: the steps after it move up one key (SP_3 -> SP_2)
      removeStep(Number(b.dataset.remove));
      changed("chain"); render();
    } else if (b.dataset.modReset) {
      const k = state.formStep, s = stepOf(k), f = b.dataset.modReset;
      if (f === "*" || f === "mode") s.module = {};
      else if (f === "relax") { delete (s.module || {}).nuchess; delete (s.module || {}).clampion; delete (s.module || {}).hessnum2; }
      else delete (s.module || {})[f];
      changed("settings"); loadForm(k);
    } else if (b.dataset.tauReset) {
      const k = state.formStep, s = stepOf(k);
      delete (s.tau || {})[b.dataset.tauReset];
      changed("settings"); loadForm(k);
    } else if (b.dataset.addMasses) {
      // BAND and DOSS right after the SP the TAU step strains: the masses for its fit
      const r = state.forms[state.formStep] || {};
      let at = Number(String(r.parent || "").split("_").pop());
      for (const t of ["BAND", "DOSS"]) if (!seq().some((x) => baseOf(x) === t)) insertStep(at++, t);
      changed("chain"); render();
    } else if (b.dataset.formRetry) {
      loadForm(b.dataset.formRetry);
    } else if (b.dataset.form) {
      state.formStep = b.dataset.form; render();
    } else if (b.dataset.reset) {
      const k = state.formStep, s = stepOf(k);
      delete (s.answers || {})[b.dataset.reset];
      if (!Object.keys(s.answers || {}).length) { s.method = null; s.settings_text = null; }
      changed("settings"); loadForm(k);
    } else if (b.dataset.socReset) {
      const k = state.formStep, s = stepOf(k), v = s.soc || {};
      if (b.dataset.socReset === "route") { delete v.soc_ecp; delete v.soc_basis; } else delete v[b.dataset.socReset];
      changed("settings"); loadForm(k);
    } else if (b.dataset.resetAll) {
      const k = state.formStep, s = stepOf(k);
      s.answers = {}; s.patches = {}; s.method = null; s.settings_text = null; s.soc = null;
      changed("settings"); loadForm(k);
    } else if (b.dataset.resReset) {
      const k = b.closest("tr").dataset.step, s = stepOf(k);
      s.resources = s.resources || {};
      if (b.dataset.resReset === "mem") { delete s.resources.memory; delete s.resources.memory_per_cpu; } else delete s.resources[b.dataset.resReset];
      changed("resources"); render();
    } else if (b.dataset.copy) S.copyText(b.dataset.copy, "Command copied");
    else if (b.dataset.act === "preview") preview();
  });
  sec.addEventListener("change", (e) => {
    const t = e.target;
    if (t.id === "wfb-folder") { state.folder = t.value; render(); return; }
    if (t.dataset.file) {
      // structures of one folder: a pick in another folder starts a new set
      const same = state.inputs && dirOf(state.inputs.files[0] || "") === dirOf(t.dataset.file);
      const cur = new Set(same ? state.inputs.files : []);
      if (t.checked) cur.add(t.dataset.file); else cur.delete(t.dataset.file);
      pickFiles([...cur].sort());
      return;
    }
    if (t.id === "wfb-add" && t.value) {
      state.draft.sequence = [...seq(), t.value]; state.draft.template = null;
      changed("chain"); render(); return;
    }
    if (t.id === "wfb-label") { state.draft.label = t.value.trim(); state.draft.labelAuto = !state.draft.label; autoLabel(); changed("label"); render(); return; }
    if (t.dataset.field) {
      const k = state.formStep, s = stepOf(k);
      const r = state.forms[k];
      const f = r && (r.form || []).find((x) => x.id === t.dataset.field);
      if (!f) return;
      s.answers = s.answers || {};
      const v = f.kind === "yesno" ? (t.checked ? "Yes" : "No") : t.value.trim();
      // back to the value the step has with nothing changed: no answer at all
      // (a hand-written row left empty, the k-point mesh, is back to the inherited value too)
      const plain = r.mode === "cif2d12" ? String((r.base_values || {})[f.id] ?? "") === String(v) : M.isDefault(f, v) || (f.how === "patch" && v === "");
      if (plain) delete s.answers[f.id]; else s.answers[f.id] = v;
      if (!Object.keys(s.answers).length) { s.method = null; s.settings_text = null; }
      changed("settings"); loadForm(k); return;
    }
    if (t.dataset.soc) { socEdit(t); return; }
    if (t.dataset.relax) { relaxEdit(t); return; }
    if (t.dataset.tau) { tauEdit(t); return; }
    if (t.dataset.modf) { moduleEdit(t); return; }
    if (t.dataset.res) { resEdit(t); return; }
    if (t.dataset.ex) {
      const ex = state.draft.execution || (state.draft.execution = {});
      ex[t.dataset.ex] = t.type === "checkbox" ? t.checked : t.value.trim();
      changed("resources"); render(); return;
    }
    if (t.dataset.tick) { state.ticks[t.dataset.tick] = t.checked; renderFoot(); }
  });
  /** TRANSPORT's relaxation time: from deformation potentials puts a TAU step right before it, the other
   * choices take that TAU step out again. */
  function relaxEdit(t) {
    const k = state.formStep, s = stepOf(k), r = state.forms[k] || {}, g = r.relaxation || {};
    if (t.dataset.relax === "fs") {
      const v = Number(t.value);
      s.relaxation_time = Number.isFinite(v) && v > 0 ? v : "default";
      changed("settings"); render(); return;
    }
    const tauAt = g.tau_step ? Number(g.tau_step.split("_").pop()) - 1 : -1;
    if (t.value === "auto") {
      s.relaxation_time = "auto";
      if (tauAt < 0) {
        const n = seq().filter((x) => baseOf(x) === "TAU").length;
        insertStep(keys().indexOf(k), n ? `TAU${n + 1}` : "TAU");
      }
    } else {
      s.relaxation_time = t.value === "fixed" ? 10 : "default";
      if (tauAt >= 0) removeStep(tauAt);
    }
    changed("chain"); render();
  }
  /** A TAU step's option: kept only when it differs from the default. */
  function tauEdit(t) {
    const k = state.formStep, s = stepOf(k), r = state.forms[k] || {};
    const f = t.dataset.tau;
    let v = t.type === "checkbox" ? t.checked : t.value.trim();
    // an emptied field is the default again (not 0 fs, no strains or no axes)
    if (v !== "") {
      if (f === "strains") v = v.split(/[\s,]+/).filter(Boolean).map(Number);
      else if (f === "axes" && v !== "auto") v = v.toLowerCase().split(/[\s,]+/).filter(Boolean);
      else if (f === "fallback_fs") v = Number(v);
      else if (f === "m_star" && v !== "auto") v = Number(v);
    }
    s.tau = s.tau || {};
    if (v === "" || JSON.stringify(v) === JSON.stringify((r.defaults || {})[f])) delete s.tau[f]; else s.tau[f] = v;
    changed("settings"); loadForm(k);
  }
  /** An ELASTIC, EOS, QHA or CPHF step's module field: kept only when set (empty is CRYSTAL's default). */
  function moduleEdit(t) {
    const k = state.formStep, s = stepOf(k), f = t.dataset.modf;
    const m = s.module = s.module || {};
    if (f === "mode") {
      // another module reads other sub-keywords: its fields start from CRYSTAL's defaults
      s.module = t.value === "ELASTCON" ? {} : { mode: t.value };
    } else if (f === "deform") {
      const fl = [...t.closest("[data-deform]").querySelectorAll("[data-d]")].map((c) => (c.checked ? 1 : 0));
      if (fl.some(Boolean)) m.deform = fl; else delete m.deform;
    } else if (f === "relax") {
      delete m.nuchess; delete m.clampion;
      if (t.value === "nuchess") m.nuchess = true; else if (t.value === "clampion") { m.clampion = true; delete m.hessnum2; }
    } else if (f === "print") {
      if (t.checked) delete m.print; else m.print = false;             // on by default: only "off" is kept
    } else if (t.dataset.kind === "flag") {
      if (t.checked) m[f] = true; else delete m[f];
    } else {
      const v = t.value.trim();
      if (v === "") delete m[f];
      else if (f === "seismdir") m[f] = v.split(";").map((y) => y.trim().split(/[\s,]+/)).filter((y) => y.length === 3);
      else if (MODULE_LISTS.includes(f)) m[f] = v.split(/[\s,]+/);
      else if ((t.dataset.kind === "int" || t.dataset.kind === "float") && Number.isFinite(Number(v))) m[f] = Number(v);
      else m[f] = v;
    }
    changed("settings"); loadForm(k);
  }
  function resEdit(t) {
    const k = t.closest("tr").dataset.step, s = stepOf(k);
    s.resources = s.resources || {};
    if (t.dataset.res === "mem" || t.dataset.res === "mem-per") {
      const tr = t.closest("tr");
      const v = tr.querySelector('[data-res="mem"]').value.trim(), per = tr.querySelector('[data-res="mem-per"]').value;
      delete s.resources.memory; delete s.resources.memory_per_cpu;
      const d0 = ((state.defaults || {}).steps || {})[k] || { resources: {} };
      const dm = memOf(d0.resources || {});
      if (v && !(v === dm.v && per === dm.per)) s.resources[per === "core" ? "memory_per_cpu" : "memory"] = v;
    } else {
      const v = t.value.trim();
      const d0 = ((state.defaults || {}).steps || {})[k] || { resources: {} };
      const dv = (d0.resources || {})[t.dataset.res];
      if (v === "" || String(v) === String(dv ?? "")) delete s.resources[t.dataset.res];
      else s.resources[t.dataset.res] = t.dataset.res === "ntasks" && /^\d+$/.test(v) ? Number(v) : v;
    }
    changed("resources");
  }
  d.addEventListener("keydown", (e) => {
    // Enter never submits (only a click, or Space on the Submit button)
    if (e.key === "Enter" && e.target.tagName !== "BUTTON" && e.target.tagName !== "SUMMARY") { e.preventDefault(); if (e.target.dataset && e.target.dataset.res) resEdit(e.target); }
    if (e.key === "Escape" && state.running) e.preventDefault();
  });
  d.addEventListener("cancel", (e) => { if (state.running) e.preventDefault(); });

  // the page's handle for tests and the shell
  window.__maceWfBuilder = { state: () => state, go, preview, submit, draft: () => draftForServer(), dialog: d };
  return new Promise((resolve) => {
    d.addEventListener("close", () => { clearTimeout(defTimer); state.defSeq++; state.prevSeq++; if (window.__maceWfBuilder && window.__maceWfBuilder.dialog === d) window.__maceWfBuilder = null;
      if (opener && opener.focus && opener.isConnected) opener.focus(); resolve(); d.remove(); }, { once: true });
    render();
    d.showModal();
    sec.focus({ preventScroll: true });
    if (fresh && state.targets.length) {
      const t = state.targets;
      const isFolder = t.length === 1 && !/\.(cif|d12)$/i.test(t[0]);
      if (isFolder) state.folder = t[0] === "." ? "" : t[0].replace(/\/$/, "");
      scanInputs(t, state.clone ? state.clone.kind : null);
    } else if (state.inputs) scheduleDefaults();
  });
}
