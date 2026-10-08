/* Batch Setup (PLAN_submit T11): a document tab "Batch: <folder>" in four
 * steps. 1 Pick files: CIFs, completed runs that need a follow-up (with why
 * a file is not offered), or failed runs. 2 Settings for all files: the
 * Setup form of one representative file; only the fields you change are
 * applied, to each file on its own (each follow-up keeps its parent's
 * settings otherwise). 3 Preview: each file's deck, its changes against its
 * default, an override for one file, and the diff. 4 Write the batch folder
 * (options, decks, commands.sh, batch.json) and Submit… its decks. */

const FOLLOWUPS = ["SP", "OPT", "FREQ", "BAND", "DOSS", "TRANSPORT", "CHARGE", "POTENTIAL"];
const STEPS = [["pick", "1 · Pick files"], ["settings", "2 · Settings for all files"], ["preview", "3 · Preview"], ["write", "4 · Write and submit"]];
const NAME_RE = /^[A-Za-z0-9._+-]{1,80}$/;

export function createBatch(host, h) {
  const { esc, icon, api, toast } = h;
  let doc = null;

  const S = () => doc.batch;
  function init(d) {
    if (!d.batch) {
      d.batch = { step: "pick", filter: "cif", followup: "SP", recursive: false, rows: null, selected: new Set(), scanning: false,
        calc: "OPT", form: null, formFor: null, answers: {}, overrides: {}, plan: null, planning: false, name: "",
        overwrite: false, written: null, writing: false, error: "" };
    }
    return d.batch;
  }
  const calcOf = (s) => (s.filter === "cif" ? s.calc : s.followup);
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
      s.formFor = `${rep}|${calcOf(s)}`;
      if (r.refused || r.save_blocked) s.error = r.save_blocked || "The CLI refused this file";
    } catch (e) { s.error = e.message; s.form = []; }
    render();
  }
  async function plan() {
    const s = S();
    s.planning = true; s.error = ""; render();
    try {
      const r = await h.apiTask("batch/plan", { folder: doc.folder, files: [...s.selected], calc: calcOf(s),
        template_answers: s.answers, overrides: s.overrides, representative: representative(s), batch_id: s.plan ? s.plan.batch_id : undefined });
      s.plan = r; s.written = null;
      if (!s.name) s.name = `batch_${calcOf(s).replace("+", "_")}_${new Date().toISOString().slice(0, 10)}`;
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
      const opts = (f.options || []).map((o) => `<option value="${esc(o.head)}"${String(o.head) === String(cur) ? " selected" : ""}>${esc(o.head)}</option>`).join("");
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
      <label class="tog"><input type="radio" name="bat-filter" value="failed"${s.filter === "failed" ? " checked" : ""}><span>Failed runs</span></label></div>
      <div class="bat-row">${s.filter === "followup" ? `<label class="f inline">Follow-up<select class="field" data-b="followup" aria-label="Follow-up type">${FOLLOWUPS.map((x) =>
        `<option${x === s.followup ? " selected" : ""}>${x}</option>`).join("")}</select></label>` : ""}
      ${s.filter === "cif" ? `<label class="f inline">Calculation<select class="field" data-b="calc" aria-label="Calculation">${["OPT", "SP", "FREQ"].map((x) =>
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
      if (s.filter === "failed") html += `<p class="note">Failed runs are listed with what can be done; for a tracked one <code>mace recover</code> prepares fixed inputs and job scripts (every failed tracked calculation of its database) and prints the command that submits each.</p>`;
    }
    html += `<div class="bat-row"><button type="button" class="tb-btn-text primary" data-go="settings"${s.selected.size && s.filter !== "failed" ? "" : " disabled"}>Next: Settings (${s.selected.size})</button></div>`;
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
      const n = Object.keys(s.answers).length;
      html += `<p class="note" data-part="changed">${n ? `${n} changed: ${Object.entries(s.answers).map(([k, v]) => `${esc(k)} = ${esc(v)}`).join(", ")}` : "No field changed: every file gets the CLI's default (a follow-up: its parent's settings)."}</p>`;
    }
    html += `<div class="bat-row"><button type="button" class="tb-btn-text" data-go="pick">Back</button>
      <button type="button" class="tb-btn-text primary" data-go="preview"${s.form ? "" : " disabled"}>Next: Preview</button></div>`;
    return html;
  }
  function stepPreview(s) {
    let html = "";
    if (s.planning) html += `<p class="note">Running the CLI for each file…</p>`;
    if (s.plan) {
      html += `<table class="bat-table" data-part="plan"><thead><tr><th>File</th><th>Deck</th><th>Status</th><th>Changes vs default</th><th></th></tr></thead><tbody>${s.plan.files.map((f) =>
        `<tr class="${f.status === "ok" ? "" : "off"}"><td class="mono">${esc(f.rel)}</td><td class="mono">${esc(f.deck_name || "")}</td>
          <td>${f.status === "ok" ? "ok" : `<span class="bad">${esc(f.message || "refused")}</span>`}${f.override ? ' <span class="chip">override</span>' : ""}</td>
          <td>${f.changed_lines !== undefined ? `differs from default in ${f.changed_lines} line${f.changed_lines === 1 ? "" : "s"}` : ""}</td>
          <td>${f.status === "ok" ? `<button type="button" class="tb-btn-text sm" data-diff="${esc(f.rel)}">Diff</button>
            <button type="button" class="tb-btn-text sm" data-deck="${esc(f.rel)}">Deck</button>
            <button type="button" class="tb-btn-text sm" data-override="${esc(f.rel)}">Override…</button>` : ""}</td></tr>`).join("")}</tbody></table>`;
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
  async function override(rel) {
    const s = S();
    if (!s.form) return;
    const cur = s.overrides[rel] || {};
    const body = document.createElement("div");
    body.className = "bat-fields";
    body.innerHTML = s.form.map((f) => fieldCtl(f, f.id in cur ? cur[f.id] : f.id in s.answers ? s.answers[f.id] : f.value, "data-ov")).join("");
    const ok = await h.confirmBody({ title: `Override for ${rel.split("/").pop()}`, body, yes: "Apply Override", no: "Cancel" });
    if (!ok) return;
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
    else if (b.dataset.act === "scan") scan();
    else if (b.dataset.act === "all") { s.selected = new Set(eligible(s).map((x) => x.rel)); render(); }
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
    if (t.dataset.ans) {
      const f = s.form.find((x) => x.id === t.dataset.ans);
      const v = t.type === "checkbox" ? t.checked : t.value;
      const same = t.type === "checkbox" ? v === [true, "y", "Yes"].includes(f.value) : String(v) === String(f.value ?? "");
      if (same) delete s.answers[f.id]; else s.answers[f.id] = v;
      s.plan = null; render(); return;
    }
    const k = t.dataset.b;
    if (!k) return;
    if (k === "followup" || k === "calc") { s[k] = t.value; s.formFor = null; s.answers = {}; s.overrides = {}; s.plan = null; if (k === "followup") scan(); else render(); return; }
    if (k === "recursive") { s.recursive = t.checked; scan(); return; }
    if (k === "rep") { s.rep = t.value; s.formFor = null; s.answers = {}; loadForm(); return; }
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
    show(d) { doc = d; init(d); render(); if (d.batch.rows === null && !d.batch.scanning) scan(); },
    state: () => doc && doc.batch,
  };
}
