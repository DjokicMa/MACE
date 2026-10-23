/* Batch Setup (PLAN_submit T11): a document tab "Batch: <folder>" in four
 * steps. 1 Pick files: CIFs, completed runs that need a follow-up (with why
 * a file is not offered), or failed runs. 2 Settings for all files: the
 * Setup form of one representative file; only the fields you change are
 * applied, to each file on its own (each follow-up keeps its parent's
 * settings otherwise). 3 Preview: each file's deck, its changes against its
 * default, an override for one file, and the diff. 4 Write the batch folder
 * (options, decks, commands.sh, batch.json) and Submit… its decks. */

import * as CF from "../setup/charge.js";
import * as SM from "../setup/model.js";

const FOLLOWUPS = ["SP", "OPT", "FREQ", "BAND", "DOSS", "TRANSPORT", "CHARGE", "POTENTIAL", "ORBITALS"];
const STEPS = [["pick", "1 · Pick files"], ["settings", "2 · Settings for all files"], ["preview", "3 · Preview"], ["write", "4 · Write and submit"]];
const NAME_RE = /^[A-Za-z0-9._+-]{1,80}$/;

let socSchema = null;                 // Setup's schema.soc (the CLI's SOC tables), asked for once

export function createBatch(host, h) {
  const { esc, icon, api, toast } = h;
  let doc = null;
  // Setup's schema: the charged-system switches a batch takes (setup/charge.js)
  let setupSchema = null;
  const loadSchema = () => setupSchema ? Promise.resolve(setupSchema)
    : api("setup/schema").then((r) => { setupSchema = r; return r; }).catch(() => { setupSchema = {}; return setupSchema; });
  /** convert and opt2d12 batches take the charged-system switches; a properties batch (opt2d3) none */
  const chargeable = (s) => !!setupSchema && (s.filter === "cif" || s.filter === "d12"
    || (s.filter === "followup" && ((setupSchema.calc_types || {}).opt2d12 || []).includes(s.followup)));

  const S = () => doc.batch;
  function init(d) {
    const pre = d.batchPreset;
    if (pre) { d.batchPreset = null; d.batch = null; }          // from Calculation Setup: start over with its choices
    if (!d.batch) {
      d.batch = { step: "pick", filter: "cif", followup: "SP", recursive: false, rows: null, selected: new Set(), scanning: false,
        calc: "OPT", form: null, formFor: null, answers: {}, patches: {}, soc: null, overrides: {}, overridePatches: {}, plan: null, planning: false, name: "",
        overwrite: false, written: null, writing: false, error: "" };
      if (pre) {
        const b = d.batch;
        b.filter = pre.filter || "cif";
        if (b.filter === "followup") b.followup = pre.calc || "SP"; else b.calc = pre.calc || b.calc;
        b.rep = pre.rep || null; b.answers = { ...(pre.answers || {}) }; b.flags = { ...(pre.flags || {}) }; b.patches = { ...(pre.patches || {}) }; b.scannedOnce = true;
      }
    }
    return d.batch;
  }
  const calcOf = (s) => (s.filter === "cif" || s.filter === "d12" ? s.calc : s.followup);
  const eligible = (s) => (s.rows || []).filter((r) => r.eligible);

  async function scan() {
    const s = S();
    s.scanning = true; s.error = ""; render();
    try {
      const r = await h.apiTask("batch/scan", { folder: doc.folder, filter: s.filter, followup_type: s.followup,
        recursive: s.recursive, doc: `batch-${doc.id}` });
      // a folder of outputs with no CIF: the first scan moves on to the follow-ups
      if (!s.scannedOnce && s.filter === "cif" && !r.rows.length) {
        s.scannedOnce = true; s.filter = "followup"; s.scanning = false;
        return scan();
      }
      s.scannedOnce = true;
      s.rows = r.rows;
      s.selected = new Set(eligible(s).map((x) => x.rel));
      if (s.rep && !s.selected.has(s.rep)) s.rep = null;
      s.plan = null; s.written = null;
    } catch (e) { s.error = e.message; s.rows = []; }
    s.scanning = false; render();
  }
  function representative(s) { return s.rep && s.selected.has(s.rep) ? s.rep : [...s.selected][0]; }
  async function loadForm() {
    const s = S();
    const rep = representative(s);
    if (!rep) return;
    if (s.formFor === `${rep}|${calcOf(s)}`) return;
    s.form = null; s.error = ""; render();
    try {
      const r = await h.apiTask("setup/defaults", { source: { path: rep }, calc_type: calcOf(s), doc: `batch-form-${doc.id}` });
      s.form = (r.form || []).filter((f) => f.how !== "readonly");
      s.soc = r.soc || null;
      s.formFor = `${rep}|${calcOf(s)}`;
      if (!socSchema) socSchema = ((await loadSchema()) || {}).soc || {};
      if (r.refused || r.save_blocked) s.error = r.save_blocked || "The CLI refused this file";
    } catch (e) { s.error = e.message; s.form = []; }
    render();
  }
  async function plan() {
    const s = S();
    s.planning = true; s.error = ""; render();
    try {
      const r = await h.apiTask("batch/plan", { folder: doc.folder, files: [...s.selected], calc: calcOf(s),
        template_answers: s.answers, template_patches: s.patches, overrides: s.overrides, override_patches: s.overridePatches || {},
        representative: representative(s), batch_id: s.plan ? s.plan.batch_id : undefined,
        flags: chargeable(s) && s.flags && Object.keys(s.flags).length ? s.flags : undefined });
      s.plan = r; s.written = null;
      // today's date where the student is (toISOString is UTC: tomorrow's date in an evening in the Americas)
      if (!s.name) { const t = new Date(), p2 = (n) => String(n).padStart(2, "0"); s.name = `batch_${calcOf(s).replace("+", "_")}_${t.getFullYear()}-${p2(t.getMonth() + 1)}-${p2(t.getDate())}`; }
    } catch (e) { s.error = e.message; }
    s.planning = false; render();
  }
  async function write() {
    const s = S();
    if (!NAME_RE.test(s.name) || s.name.startsWith(".")) { s.error = "The batch folder name: 1 to 80 letters, digits, . _ + - (not starting with a dot)"; render(); return; }
    s.writing = true; s.error = ""; render();
    try {
      s.written = await h.apiTask("batch/write", { batch_id: s.plan.batch_id, name: s.name, overwrite: s.overwrite });
      if (s.written.code) s.error = `A command exited with ${s.written.code}; see its log`;
      h.refreshFiles();
      toast(`Wrote ${s.written.written.length} file(s) in ${s.written.folder_rel}`, "success");
    } catch (e) { s.error = e.message; }
    s.writing = false; render();
  }

  function fieldCtl(f, cur, attr) {
    const id = `bat-f-${f.id.replace(/[^A-Za-z0-9_-]/g, "_")}`;
    if (f.kind === "menu") {
      const opts = (f.options || []).map((o) => `<option value="${esc(o.head)}"${String(o.head) === String(cur) ? " selected" : ""}>${esc(SM.optionLabel(o))}</option>`).join("");
      return `<label for="${id}">${esc(f.label)}<select class="field" id="${id}" ${attr}="${esc(f.id)}">${cur && !(f.options || []).some((o) => o.head === cur) ? `<option selected>${esc(cur)}</option>` : ""}${opts}</select></label>`;
    }
    if (f.kind === "yesno") {
      const on = [true, "y", "Yes"].includes(cur);
      return `<label class="tog"><input type="checkbox" id="${id}" ${attr}="${esc(f.id)}"${on ? " checked" : ""}><span>${esc(f.label)}</span></label>`;
    }
    return `<label for="${id}">${esc(f.label)}${f.unit ? ` (${esc(f.unit)})` : ""}<input class="field" id="${id}" ${attr}="${esc(f.id)}" value="${esc(cur ?? "")}" spellcheck="false"></label>`;
  }

  function stepPick(s) {
    let html = `<div class="bat-row" role="radiogroup" aria-label="Files">
      <label class="tog"><input type="radio" name="bat-filter" value="cif"${s.filter === "cif" ? " checked" : ""}><span>CIF files</span></label>
      <label class="tog"><input type="radio" name="bat-filter" value="followup"${s.filter === "followup" ? " checked" : ""}><span>Completed runs needing a follow-up</span></label>
      <label class="tog"><input type="radio" name="bat-filter" value="d12"${s.filter === "d12" ? " checked" : ""}><span>Decks (.d12) as parents</span></label>
      <label class="tog"><input type="radio" name="bat-filter" value="failed"${s.filter === "failed" ? " checked" : ""}><span>Failed runs</span></label></div>
      <div class="bat-row">${s.filter === "followup" ? `<label class="f inline">Follow-up<select class="field" data-b="followup" aria-label="Follow-up type">${FOLLOWUPS.map((x) =>
        `<option${x === s.followup ? " selected" : ""}>${x}</option>`).join("")}</select></label>` : ""}
      ${s.filter === "cif" || s.filter === "d12" ? `<label class="f inline">Calculation<select class="field" data-b="calc" aria-label="Calculation">${["OPT", "SP", "FREQ"].map((x) =>
        `<option${x === s.calc ? " selected" : ""}>${x}</option>`).join("")}</select></label>` : ""}
      <label class="tog"><input type="checkbox" data-b="recursive"${s.recursive ? " checked" : ""}><span>Include subfolders</span></label>
      <button type="button" class="tb-btn-text" data-act="scan">${icon("refresh-cw")}<span>Scan</span></button>
      <button type="button" class="tb-btn-text" data-act="all"${s.rows && eligible(s).length ? "" : " disabled"}>Select all eligible</button></div>`;
    if (s.scanning) html += `<p class="note">Scanning…</p>`;
    else if (s.rows && !s.rows.length) html += `<p class="note">Nothing found.</p>`;
    else if (s.rows) {
      html += `<table class="bat-table" data-part="rows"><thead><tr><th></th><th>File</th><th>${s.filter === "cif" ? "" : "Status"}</th><th>Why not</th></tr></thead><tbody>${s.rows.map((r) =>
        `<tr class="${r.eligible ? "" : "off"}"><td><input type="checkbox" data-row="${esc(r.rel)}"${s.selected.has(r.rel) ? " checked" : ""}${r.eligible ? "" : " disabled"} aria-label="Select ${esc(r.rel)}"></td>
         <td class="mono">${esc(r.rel)}</td><td>${esc(r.bucket || "")}${r.calc_type ? ` (${esc(r.calc_type)})` : ""}</td>
         <td class="why">${esc(r.why_not || "")}${r.recover ? ` <code>${esc(r.recover)}</code> <button type="button" class="tb-btn-text sm" data-copy="${esc(r.recover)}">Copy Command</button>` : ""}</td></tr>`).join("")}</tbody></table>`;
      if (s.filter === "failed") html += `<p class="note">Failed runs are listed with what can be done; for a tracked one <code>mace recover</code> shows each failed tracked calculation of its database with its fix and, once confirmed, applies the fixes and resubmits them (<code>--dry-run</code> shows the changes first; <b>Recover Failed…</b> in Jobs does the same).</p>`;
    }
    html += `<div class="bat-row"><button type="button" class="tb-btn-text primary" data-go="settings"${s.selected.size && s.filter !== "failed" ? "" : " disabled"}>Next: Settings (${s.selected.size})</button>` +
      `<button type="button" class="tb-btn-text" data-act="legend"${s.selected.size ? "" : " disabled"} title="A legend of the elements of the selected files, for figures">${icon("list")}<span>Element Legend…</span></button></div>`;
    return html;
  }
  function stepSettings(s) {
    const rep = representative(s);
    let html = `<div class="bat-row"><label class="f inline">Representative file<select class="field" data-b="rep" aria-label="Representative file">${[...s.selected].map((x) =>
      `<option value="${esc(x)}"${x === rep ? " selected" : ""}>${esc(x)}</option>`).join("")}</select></label></div>
      <p class="note">Changed fields apply to every file. Each follow-up keeps its own parent's settings otherwise.</p>`;
    if (!s.form) html += `<p class="note">Asking the CLI for this file's settings…</p>`;
    else {
      html += `<div class="bat-fields" data-part="fields">${s.form.map((f) => fieldCtl(f, f.id in s.answers ? s.answers[f.id] : f.value, "data-ans")).join("")}</div>`;
      if (chargeable(s)) {
        const sw = CF.argsText(setupSchema.charge_flags, s.flags);
        html += `<fieldset class="bat-charge" data-part="charge"><legend>${esc(CF.SECTION)}</legend>
          <p class="note">The CLI's charged-system switches, on every command of the batch${sw ? `: <code>${esc(sw)}</code>` : ""}. Each file is checked on its own; a file the CLI refuses shows why in Preview.</p>
          <div class="bat-fields">${CF.rowsHtml(setupSchema.charge_flags, s.flags, { esc, prefix: "bat-cf", layout: "batch" })}</div></fieldset>`;
      }
      html += socFields(s);
      const all = { ...s.answers, ...s.patches };
      const n = Object.keys(all).length;
      html += `<p class="note" data-part="changed">${n ? `${n} changed: ${Object.entries(all).map(([k, v]) => `${esc(k)} = ${esc(v === null ? "(removed)" : v)}`).join(", ")}` : "No field changed: every file gets the CLI's default (a follow-up: its parent's settings)."}</p>`;
    }
    html += `<div class="bat-row"><button type="button" class="tb-btn-text" data-go="pick">Back</button>
      <button type="button" class="tb-btn-text primary" data-go="preview"${s.form ? "" : " disabled"}>Next: Preview</button></div>`;
    return html;
  }
  /** Spin-orbit coupling for every file: keys of the CLI's settings file (Setup's SOC section),
   * sent as template_patches; single points only (the CLI's reason otherwise). */
  function socFields(s) {
    const sc = socSchema || {};
    if (!sc.keys || s.filter === "failed") return "";
    const calc = calcOf(s);
    const mode = s.filter === "cif" ? "cif2d12" : (FOLLOWUPS.indexOf(calc) < 3 ? "opt2d12" : "opt2d3");
    const applies = (sc.calcs || []).includes(calc) && (sc.modes || []).includes(mode);
    const base = (s.soc && s.soc.base) || {};
    const on = SM.socValue(s, base, "soc") === true;
    let html = `<fieldset class="bat-soc" data-part="soc"><legend>Spin-orbit coupling</legend>`;
    if (!applies && !on) {
      return html + `<label class="tog"><input type="checkbox" id="bat-soc-on" disabled aria-disabled="true" aria-describedby="bat-soc-why"><span>Spin-orbit coupling</span></label>
        <p class="note" id="bat-soc-why">${esc((sc.not_for || {})[calc] || "")}</p></fieldset>`;
    }
    html += `<label class="tog"><input type="checkbox" id="bat-soc-on" data-soc="soc"${on ? " checked" : ""} aria-describedby="bat-soc-h"><span>Spin-orbit coupling (TWOCOMPON)</span></label>`;
    if (!applies) html += `<p class="note">${esc((sc.not_for || {})[calc] || "")}</p>`;
    if (on && applies) {
      const route = SM.socRoute(s, base), fam = (sc.soc_basis_families || [])[0] || "";
      const opt = (v, t, off) => `<option value="${esc(v)}"${v === route ? " selected" : ""}${off && v !== route ? " disabled" : ""}>${esc(t)}</option>`;
      const ref = stutscInfo(s), why = stutscWhy(ref, "bat-soc-why-stutsc");
      html += `<div class="bat-fields"><label for="bat-soc-route">Spin-orbit ECP<select class="field" id="bat-soc-route" data-soc="route"${why ? ' aria-describedby="bat-soc-why-stutsc"' : ""}>${opt("inpsoc", "INPSOC (the CLI's default)")}${opt("stutsc", stutscLabel(ref), ref && ref.all)}${opt("basis", `Official SOC basis (${fam})`)}</select></label>${why}`;
      if (mode === "opt2d12") {
        const g = SM.socValue(s, base, "soc_guess") || "";
        const go = (v, t) => `<option value="${v}"${v === g ? " selected" : ""}>${t}</option>`;
        html += `<label for="bat-soc-guess">Starting guess<select class="field" id="bat-soc-guess" data-soc="soc_guess" aria-describedby="bat-soc-guess-h">${go("", "From scratch")}${go("guesspnoso", "GUESSPNOSO")}${go("guesspso", "GUESSPSO")}</select></label>`;
      }
      const sm = SM.socValue(s, base, "smear");
      html += `<label for="bat-soc-smear">SMEAR for metals (Ha)<input class="field" id="bat-soc-smear" data-soc="smear" inputmode="decimal" placeholder="none" value="${esc(sm ?? "")}" spellcheck="false" aria-describedby="bat-soc-smear-h"></label></div>`;
      html += `<div class="bat-row">${String(sm) !== String(sc.metal_smear) ? `<button type="button" class="tb-btn-text sm" data-soc-use="smear" aria-describedby="bat-soc-smear-h">Use ${esc(sc.metal_smear)}</button>` : ""}
        <span class="note" id="bat-soc-smear-h">SMEAR: empty is none (the CLI never adds it by itself)</span></div>`;
      if (mode === "opt2d12") html += `<p class="note" id="bat-soc-guess-h">The CLI checks each parent's starting guess; a parent it refuses shows its reason in Preview.</p>`;
    }
    return html + `<p class="note" id="bat-soc-h">Written as "soc", "soc_ecp", "soc_basis", "soc_guess" and "smear" in each file's settings, the keys the CLI reads.</p></fieldset>`;
  }
  /* STUTSC refused for an element of the selected files (the CLI's soc_ecp.py table and words, as Setup
     grays it out): asked once per selection; grayed out when every file has one, else named */
  function stutscInfo(s) {
    const key = [...s.selected].sort().join("\n");
    if (s.stutscFor === key) return s.stutsc;
    if (s.stutscAsking === key) return null;
    s.stutscAsking = key;
    h.apiTask("batch/stutsc", { files: [...s.selected], doc: `batch-stutsc-${doc.id}` }).then((r) => {
      if (s.stutscAsking !== key) return;
      s.stutsc = r; s.stutscFor = key; s.stutscAsking = null; render();
    }, () => { if (s.stutscAsking === key) s.stutscAsking = null; });
    return null;
  }
  const stutscLabel = (ref) => {
    if (!ref || !ref.refused.length) return "STUTSC";
    const n = new Set(ref.refused.flatMap((x) => x.files)).size, els = ref.refused.map((x) => x.symbol).join(", ");
    return ref.all ? `STUTSC (refused for ${els})` : `STUTSC (refused for ${els} in ${n} of ${ref.files} files)`;
  };
  // STUTSC refusals, and the files whose elements could not be read (STUTSC unchecked for them: e.g. no ase for a CIF)
  const stutscWhy = (ref, id) => {
    const lines = ref ? ref.refused.map((x) => `${esc(x.symbol)}: ${esc(x.why)}${ref.all ? "" : ` (${esc(x.files.map((f) => f.split("/").pop()).join(", "))})`}`)
      .concat((ref.unread || []).map((u) => `STUTSC not checked: ${esc(u.why)}`)) : [];
    return lines.length ? `<p class="note" id="${id}">${lines.join("<br>")}</p>` : "";
  };
  function socEdit(s, t) {
    const base = (s.soc && s.soc.base) || {};
    const k = t.dataset.soc;
    if (k === "soc") SM.socSwitch(s, base, t.checked);
    else if (k === "route") SM.socSetRoute(s, base, t.value, ((socSchema || {}).soc_basis_families || [])[0]);
    else if (k === "soc_guess") SM.socSet(s, base, "soc_guess", t.value || undefined);
    else if (k === "smear") { const raw = t.value.trim(), n = Number(raw); SM.socSet(s, base, "smear", raw === "" ? undefined : Number.isFinite(n) ? n : raw); }
    s.plan = null;
    render();
    const f = host.querySelector(`[data-soc="${k}"]`) || host.querySelector("#bat-soc-on");
    if (f) f.focus();
  }
  function stepPreview(s) {
    let html = "";
    if (s.planning) html += `<p class="note">Running the CLI for each file…</p>`;
    if (s.plan) {
      html += `<table class="bat-table" data-part="plan"><thead><tr><th>File</th><th>Deck</th><th>Status</th><th>Changes vs default</th><th></th></tr></thead><tbody>${s.plan.files.map((f) =>
        `<tr class="${f.status === "ok" ? "" : "off"}"><td class="mono">${esc(f.rel)}</td><td class="mono">${esc(f.deck_name || "")}</td>
          <td>${f.status === "ok" ? "ok" : `<span class="bad">${esc(f.message || "refused")}</span>`}${f.override ? ' <span class="chip">override</span>' : ""}${
            f.soc_warnings && f.soc_warnings.length ? `<details class="bat-socwarn"><summary>${f.soc_warnings.length} SOC warning${f.soc_warnings.length === 1 ? "" : "s"}</summary>${f.soc_warnings.map((w) => `<p class="note">${esc(w)}</p>`).join("")}</details>` : ""}</td>
          <td>${f.changed_lines !== undefined ? `differs from default in ${f.changed_lines} line${f.changed_lines === 1 ? "" : "s"}` : ""}</td>
          <td>${f.status === "ok" ? `<button type="button" class="tb-btn-text sm" data-diff="${esc(f.rel)}">Diff</button>
            <button type="button" class="tb-btn-text sm" data-deck="${esc(f.rel)}">Deck</button>` : ""}
            ${f.status === "ok" || f.override || socApplies(s) ? `<button type="button" class="tb-btn-text sm" data-override="${esc(f.rel)}">Override…</button>` : ""}</td></tr>`).join("")}</tbody></table>`;
      if (s.plan.refused) html += `<p class="note">${s.plan.refused} file(s) refused by the CLI are left out.</p>`;
    }
    html += `<div class="bat-row"><button type="button" class="tb-btn-text" data-go="settings">Back</button>
      <button type="button" class="tb-btn-text" data-act="plan">${icon("refresh-cw")}<span>Preview again</span></button>
      <button type="button" class="tb-btn-text primary" data-go="write"${s.plan && s.plan.files.some((f) => f.status === "ok") ? "" : " disabled"}>Next: Write</button></div>`;
    return html;
  }
  function stepWrite(s) {
    const n = s.plan ? s.plan.files.filter((f) => f.status === "ok").length : 0;
    const bad = s.name && (!NAME_RE.test(s.name) || s.name.startsWith("."));
    let html = `<div class="bat-row"><label class="f inline">Batch folder<input class="field" data-b="name" value="${esc(s.name)}" spellcheck="false" aria-label="Batch folder"${bad ? ' aria-invalid="true"' : ""}></label>
      <label class="tog"><input type="checkbox" data-b="overwrite"${s.overwrite ? " checked" : ""}><span>Overwrite</span></label>
      <button type="button" class="tb-btn-text primary" data-act="write"${n && !bad && !s.writing ? "" : " disabled"}>Write ${n} deck${n === 1 ? "" : "s"}</button></div>`;
    if (bad) html += `<p class="bad note">1 to 80 letters, digits, . _ + - (not starting with a dot)</p>`;
    html += `<p class="note">In ${esc(doc.folder || "the served folder")}/: options/, decks/, commands.sh and batch.json. Nothing outside it is written.</p>`;
    if (s.writing) html += `<p class="note">Writing…</p>`;
    if (s.written) {
      const w = s.written;
      html += `<p class="note">Wrote ${w.written.length} file(s) in <code>${esc(w.folder_rel)}</code>${w.backups.length ? ` (${w.backups.length} replaced, .bak kept)` : ""}.</p>
        <pre class="bat-cmds" data-part="commands">${esc(w.commands.join("\n"))}</pre>
        <div class="bat-row"><button type="button" class="tb-btn-text" data-copy-cmds="1">Copy</button>
        <button type="button" class="tb-btn-text" data-open="${esc(w.commands_rel)}">Open commands.sh</button>
        <button type="button" class="tb-btn-text" data-open="${esc(w.template_options_rel)}">Open template.json</button>
        <button type="button" class="tb-btn-text primary" data-act="submit">Submit…</button></div>`;
      if (w.one_file_per_parent) html += `<p class="note">${esc(w.one_file_per_parent)}.</p>`;
    }
    html += `<div class="bat-row"><button type="button" class="tb-btn-text" data-go="preview">Back</button></div>`;
    return html;
  }

  function render() {
    if (!doc || !host) return;
    const s = S();
    const steps = `<nav class="bat-steps" aria-label="Batch Setup steps">${STEPS.map(([k, t]) =>
      `<button type="button" data-go="${k}"${s.step === k ? ' aria-current="step"' : ""}>${t}</button>`).join("")}</nav>`;
    const body = { pick: stepPick, settings: stepSettings, preview: stepPreview, write: stepWrite }[s.step](s);
    host.innerHTML = `<div class="bat" data-doc="${doc.id}"><h2>Batch Setup: ${esc(doc.folder || "the served folder")}</h2>${steps}
      <section class="bat-step" aria-label="${esc(STEPS.find((x) => x[0] === s.step)[1])}">${body}</section>
      ${s.error ? `<p class="bad" role="alert">${esc(s.error)}</p>` : ""}</div>`;
  }
  function go(step) {
    const s = S();
    if (step === "settings" && !s.selected.size) return;
    s.step = step; s.error = "";
    render();
    if (step === "settings") loadForm();
    if (step === "preview" && !s.planning) plan();
  }
  /* SOC for one file (the Override dialog): the batch's choice, or on/off and the ECP route of its own,
     sent as that file's override_patches (the same keys of the CLI's settings file) */
  function socApplies(s) {
    const sc = socSchema || {};
    if (!sc.keys || s.filter === "failed") return false;
    const calc = calcOf(s), mode = s.filter === "cif" ? "cif2d12" : (FOLLOWUPS.indexOf(calc) < 3 ? "opt2d12" : "opt2d3");
    return (sc.calcs || []).includes(calc) && (sc.modes || []).includes(mode);
  }
  function fileSoc(s, rel) {
    const base = (s.soc && s.soc.base) || {};
    return { patches: { ...s.patches, ...((s.overridePatches || {})[rel] || {}) }, base };
  }
  function socOverrideHtml(s, rel) {
    if (!socApplies(s)) return "";
    const fam = ((socSchema || {}).soc_basis_families || [])[0] || "";
    const f = fileSoc(s, rel), on = SM.socValue(f, f.base, "soc") === true, route = SM.socRoute(f, f.base);
    const ref = s.stutsc && s.stutsc.refused.some((x) => x.files.includes(rel)) ? { ...s.stutsc, refused: s.stutsc.refused.filter((x) => x.files.includes(rel)), all: true } : null;
    const opt = (v, t, off) => `<option value="${v}"${v === route ? " selected" : ""}${off && v !== route ? " disabled" : ""}>${esc(t)}</option>`;
    return `<fieldset class="bat-soc" data-part="ov-soc"><legend>Spin-orbit coupling for this file</legend>
      <label class="tog"><input type="checkbox" id="bat-ov-soc-on" data-ov-soc="soc"${on ? " checked" : ""}><span>Spin-orbit coupling (TWOCOMPON)</span></label>
      <label for="bat-ov-soc-route">Spin-orbit ECP<select class="field" id="bat-ov-soc-route" data-ov-soc="route"${ref ? ' aria-describedby="bat-ov-why-stutsc"' : ""}>${opt("inpsoc", "INPSOC (the CLI's default)")}${opt("stutsc", stutscLabel(ref), !!ref)}${opt("basis", `Official SOC basis (${fam})`)}</select></label>${stutscWhy(ref, "bat-ov-why-stutsc")}
      <p class="note">The batch's choice unless changed here (written as "soc", "soc_ecp" and "soc_basis" in this file's settings).</p></fieldset>`;
  }
  /* the file's SOC patches against the batch's: only the keys whose value differs (null removes a key) */
  function socOverrideOf(s, rel, body) {
    const on = body.querySelector("#bat-ov-soc-on");
    if (!on) return null;
    const base = (s.soc && s.soc.base) || {}, fam = ((socSchema || {}).soc_basis_families || [])[0];
    const f = { patches: { ...s.patches }, base };
    SM.socSwitch(f, base, on.checked);
    if (on.checked) SM.socSetRoute(f, base, body.querySelector("#bat-ov-soc-route").value, fam);
    const out = {};
    for (const k of ["soc", ...SM.SOC_SUB]) {
      const a = k in f.patches ? f.patches[k] : undefined, b = k in s.patches ? s.patches[k] : undefined;
      if (JSON.stringify(a) !== JSON.stringify(b)) out[k] = a === undefined ? (base[k] !== undefined ? base[k] : null) : a;
    }
    return out;
  }
  async function override(rel) {
    const s = S();
    if (!s.form) return;
    const cur = s.overrides[rel] || {};
    const body = document.createElement("div");
    body.className = "bat-fields";
    body.innerHTML = s.form.map((f) => fieldCtl(f, f.id in cur ? cur[f.id] : f.id in s.answers ? s.answers[f.id] : f.value, "data-ov")).join("") +
      socOverrideHtml(s, rel);
    const sync = () => { const on = body.querySelector("#bat-ov-soc-on"), r = body.querySelector("#bat-ov-soc-route"); if (on && r) r.disabled = !on.checked; };
    body.addEventListener("change", sync);
    sync();
    const ok = await h.confirmBody({ title: `Override for ${rel.split("/").pop()}`, body, yes: "Apply Override", no: "Cancel" });
    if (!ok) return;
    const sp = socOverrideOf(s, rel, body);
    if (sp !== null) {
      s.overridePatches = s.overridePatches || {};
      if (Object.keys(sp).length) s.overridePatches[rel] = sp; else delete s.overridePatches[rel];
    }
    const ov = {};
    body.querySelectorAll("[data-ov]").forEach((el) => {
      const f = s.form.find((x) => x.id === el.dataset.ov);
      const v = el.type === "checkbox" ? el.checked : el.value;
      const base = f.id in s.answers ? s.answers[f.id] : f.value;
      if (el.type === "checkbox" ? v !== [true, "y", "Yes"].includes(base) : String(v) !== String(base ?? "")) ov[f.id] = v;
    });
    if (Object.keys(ov).length) s.overrides[rel] = ov; else delete s.overrides[rel];
    plan();
  }

  host.addEventListener("click", async (e) => {
    if (!doc) return;
    const s = S();
    const b = e.target.closest("button");
    if (!b) return;
    if (b.dataset.go) go(b.dataset.go);
    else if (b.dataset.socUse) { const base = (s.soc && s.soc.base) || {}; SM.socSet(s, base, "smear", (socSchema || {}).metal_smear); s.plan = null; render(); const i = host.querySelector("#bat-soc-smear"); if (i) i.focus(); }
    else if (b.dataset.act === "scan") scan();
    else if (b.dataset.act === "all") { s.selected = new Set(eligible(s).map((x) => x.rel)); render(); }
    else if (b.dataset.act === "legend" && h.legend) h.legend([...s.selected], `${s.selected.size} selected file${s.selected.size === 1 ? "" : "s"} in ${doc.folder || "this folder"}`);
    else if (b.dataset.act === "plan") plan();
    else if (b.dataset.act === "write") write();
    else if (b.dataset.act === "submit" && s.written) h.submit([s.written.decks_rel], s.written.written.filter((x) => /\.(d12|d3)$/.test(x)).length >= 2);
    else if (b.dataset.diff) {
      try { const r = await api(`batch/diff?batch_id=${encodeURIComponent(s.plan.batch_id)}&rel=${encodeURIComponent(b.dataset.diff)}&against=default`); h.showText(`Diff: ${b.dataset.diff.split("/").pop()}`, r.diff); }
      catch (err) { toast(err.message, "error"); }
    } else if (b.dataset.deck) {
      try { const r = await api(`batch/deck?batch_id=${encodeURIComponent(s.plan.batch_id)}&rel=${encodeURIComponent(b.dataset.deck)}`); h.showText(r.deck_name || "deck", r.deck || ""); }
      catch (err) { toast(err.message, "error"); }
    } else if (b.dataset.override) override(b.dataset.override);
    else if (b.dataset.open) h.openText(b.dataset.open);
    else if (b.dataset.copy) h.copyText(b.dataset.copy, "Command copied");
    else if (b.dataset.copyCmds && s.written) h.copyText(s.written.commands.join("\n"), "Commands copied");
  });
  host.addEventListener("change", (e) => {
    if (!doc) return;
    const s = S(), t = e.target;
    if (t.name === "bat-filter") { s.filter = t.value; s.rows = null; s.selected.clear(); render(); scan(); return; }
    if (t.dataset.row) { if (t.checked) s.selected.add(t.dataset.row); else s.selected.delete(t.dataset.row); s.plan = null; render(); return; }
    if (t.dataset.charge && setupSchema) {
      const def = (setupSchema.charge_flags || []).find((x) => x.id === t.dataset.charge);
      if (def) { s.flags = CF.withValue(s.flags, def.id, CF.valueOf(def, t)); s.plan = null; render(); }
      return;
    }
    if (t.dataset.soc) { socEdit(s, t); return; }
    if (t.dataset.ans) {
      const f = s.form.find((x) => x.id === t.dataset.ans);
      const v = t.type === "checkbox" ? t.checked : t.value;
      const same = t.type === "checkbox" ? v === [true, "y", "Yes"].includes(f.value) : String(v) === String(f.value ?? "");
      if (same) delete s.answers[f.id]; else s.answers[f.id] = v;
      s.plan = null; render(); return;
    }
    const k = t.dataset.b;
    if (!k) return;
    if (k === "followup" || k === "calc") { s[k] = t.value; s.formFor = null; s.answers = {}; s.patches = {}; s.overrides = {}; s.overridePatches = {}; s.plan = null; if (k === "followup") scan(); else render(); return; }
    if (k === "recursive") { s.recursive = t.checked; scan(); return; }
    if (k === "rep") { s.rep = t.value; s.formFor = null; s.answers = {}; s.patches = {}; loadForm(); return; }
    if (k === "overwrite") { s.overwrite = t.checked; return; }
  });
  host.addEventListener("input", (e) => {
    if (!doc) return;
    const t = e.target;
    if (t.dataset.b === "name") {
      S().name = t.value.trim();
      const bad = S().name && (!NAME_RE.test(S().name) || S().name.startsWith("."));
      t.setAttribute("aria-invalid", bad ? "true" : "false");
      const w = host.querySelector('[data-act="write"]');
      if (w) w.disabled = !!bad || !S().name;
    }
  });
  return {
    show(d) { doc = d; init(d); render(); if (d.batch.rows === null && !d.batch.scanning) scan(); if (!setupSchema) loadSchema().then(() => { if (doc === d) render(); }); },
    state: () => doc && doc.batch,
  };
}
