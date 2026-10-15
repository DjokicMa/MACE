/* Submit… and Run on This Node… (PLAN_submit T9): a modal that shows the
 * exact command, the resources read from the job scripts, every file it
 * will write and what the jobs write, the conflicts, the site settings and
 * every warning. It runs nothing until Submit is clicked: Cancel has the
 * focus, Enter never submits, and the server refuses a run without a fresh
 * preview. Loaded on first use. */
import { loadCss } from "./lazy_util.js";
import * as profile from "../profile.js";

const DEBOUNCE_MS = 400;

/* The last options per route (slurm, local), in the GUI profile (memory.submitLast,
 * gui.json): they follow you to a new port or browser. Never site values. */
function lastOptions() {
  const v = profile.memory("submitLast", null);
  return v && typeof v === "object" ? v : {};
}
function rememberOptions(H, where, options) {
  if (H.settings.get("submit.rememberLast") === false) return;
  profile.remember("submitLast", { ...lastOptions(), [where]: options });
}
const plural = (n, one, many = one + "s") => `${n} ${n === 1 ? one : many}`;
const base = (p) => String(p || "").split("/").pop();

/** {targets: [rel], where: "slurm"|"local"} -> resolves when the dialog closes. */
export async function open(H, { targets, where = "slurm", note = "", preset = null } = {}) {
  await loadCss(import.meta.url, "../../css/dialogs.css");
  await loadCss(import.meta.url, "../../css/submit.css");
  H.overlayOpening();
  const opener = document.activeElement;
  let caps = {};
  try { caps = await H.api("submit/capabilities"); } catch (e) { caps = { error: e.message }; }
  const remembered = lastOptions();
  const state = {
    where: where === "local" || !caps.sbatch ? (caps.sbatch ? where : "local") : "slurm",
    opts: { slurm: { array: false, max_running: "", track: false, progress: "", walltime: "", exclude: { kind: "none", value: "" },
      overwrite_sh: false, ...(remembered.slurm || {}) },
    local: { cores: "", timeout: "", track: false, ...(remembered.local || {}) } },
    mixed: null, preview: null, previewing: false, stale: true, error: "", running: false, ticks: {}, seq: 0,
  };
  if (preset) Object.assign(state.opts.slurm, preset);
  // arrays only for a folder (the server says so too); never from one file
  const isFolder = targets.length === 1 && !/\.(d12|d3)$/i.test(targets[0]);
  if (!isFolder) state.opts.slurm.array = false;
  let timer = 0;
  const d = document.createElement("dialog");
  d.id = "submit-dialog";
  d.className = "dlg submit-dlg";
  d.setAttribute("aria-labelledby", "submit-title");
  document.getElementById("submit-dialog")?.remove();
  document.body.appendChild(d);
  const title = () => (state.where === "local" ? "Run on This Node" : "Submit to SLURM");
  d.innerHTML = `<div class="dlg-head"><h2 id="submit-title"></h2><button type="button" class="icon-btn sm" data-close aria-label="Close">${H.icon("x")}</button></div>
    <div class="dlg-body"></div>
    <div class="dlg-foot"><span class="submit-note">Nothing is submitted until you click Submit.</span>
      <button type="button" class="tb-btn-text" data-act="cancel">Cancel</button>
      <button type="button" class="tb-btn-text primary" data-act="submit" disabled>Submit</button></div>`;
  const body = d.querySelector(".dlg-body");
  const btn = d.querySelector('[data-act="submit"]');

  function optionsFor(w) {
    const o = state.opts[w];
    if (w === "local") return { cores: o.cores === "" ? undefined : Number(o.cores), timeout: o.timeout || undefined, track: !!o.track };
    const out = { array: !!o.array, track: !!o.track, walltime: o.walltime || undefined, overwrite_sh: !!o.overwrite_sh,
      exclude: o.exclude && o.exclude.kind !== "none" ? o.exclude : { kind: "none" } };
    if (o.array && o.max_running !== "" && o.max_running !== null) out.max_running = Number(o.max_running);
    if (o.progress) out.progress = o.progress;
    return out;
  }
  // above 20 decks a preview is explicit (Update preview): it copies and runs a lot
  const many = () => ((state.preview && state.preview.n_decks) || targets.length) > (caps.auto_preview_max || 20);
  function needs() {
    const p = state.preview;
    if (!p) return [];
    return Object.entries(p.needs || {}).filter(([, v]) => v).map(([k]) => k);
  }
  function canSubmit() {
    if (!state.preview || state.stale || state.previewing || state.running) return false;
    if (state.preview.code) return false;
    if (state.where === "slurm" && caps.network) return false;
    return needs().every((k) => state.ticks[k]);
  }
  function nJobs() {
    const p = state.preview;
    if (!p) return 0;
    return state.where === "local" ? (p.decks || []).length : (p.n_jobs || 0);
  }
  function renderFoot() {
    btn.disabled = !canSubmit();
    // a submission cannot be stopped halfway: the dialog stays until it is done
    const locked = state.running && state.where === "slurm";
    d.querySelectorAll('[data-act="cancel"], [data-close]').forEach((x) => { x.disabled = locked; });
    const n = nJobs();
    btn.textContent = state.running ? "Submitting…" : state.where === "local" ? `Run ${plural(n || targets.length, "deck")}`
      : `Submit ${plural(n || 0, "job")}`;
  }

  function siteRow(p) {
    const s = (p && p.site) || caps.site;
    if (!s || state.where !== "slurm") return "";
    const acct = s.account || {}, cr = s.crystal || {};
    const chk = s.account_check || {};
    const acctWord = chk.status === "member" ? "one of your accounts" : chk.status === "not_member" ? "not one of your accounts"
      : chk.status === "default" ? `no -A line: SLURM charges your default account${chk.default ? ` (${chk.default})` : ""}`
      : chk.status === "unreadable" ? "no account line (SLURM's default account); set yours with MACE_SLURM_ACCOUNT or site.toml"
      : "account not checked";
    const crWord = { default: "built-in, loads here", "site.toml": "site.toml, loads here", unchecked: "not checked: no module system here",
      discovered: "discovered here", "not found here": "not found here" }[cr.source] || cr.source || "";
    let html = `<div class="sub-row"><span class="sub-k">Site</span><div class="sub-v" data-part="site">Account <b>${H.esc(acct.value || "(SLURM default)")}</b>
      <span class="muted">(${H.esc(acct.source)}, ${H.esc(acctWord)})</span> · ${H.esc((cr.modules || []).join(" then "))} <span class="muted">(${H.esc(crWord)})</span>
      <button type="button" class="icon-btn sm" data-act="site-help" aria-label="About site settings" aria-expanded="false">${H.icon("circle-help")}</button>
      <div class="sub-pop" hidden>MACE writes these into every job script it generates. Change them in
        <code>${H.esc((s.config && s.config.path) || "~/.config/mace/site.toml")}</code> (create it with <code>mace site --init</code>
        <button type="button" class="tb-btn-text sm" data-copy="mace site --init">Copy</button>). A job script that already exists is reused as it is unless you tick Regenerate job scripts.</div>`;
    if (chk.status === "not_member") {
      html += `<p class="sub-line bad" data-part="account-warning">${H.esc(acct.value)} is not one of your SLURM accounts (yours: ${H.esc((chk.accounts || []).join(", "))}).
        Set [slurm] account in ${H.esc((s.config && s.config.path) || "~/.config/mace/site.toml")} or MACE_SLURM_ACCOUNT, then preview again.</p>`;
    }
    const py = s.python || {};
    if (py.executable) html += `<p class="sub-line muted" data-part="python-note">The built-in Python modules do not load here; the completion step runs ${H.esc(py.executable)}.</p>`;
    if (cr.source === "discovered" && cr.note) html += `<p class="sub-line warn">${H.esc(cr.note)}</p>`;
    else if (cr.source === "not found here" && cr.note) html += `<p class="sub-line muted">${H.esc(cr.note)}</p>`;
    html += siteDetails(s);
    return html + "</div></div>";
  }

  // The resolved site values (mace site --json, describe()): one compact
  // line each, with the source word, and the Check button (mace site check).
  function siteDetails(s) {
    const src = (o) => (o && o.source ? ` <span class="muted">(${H.esc(o.source)})</span>` : "");
    const size = (v) => !v ? "" : `${v.ntasks} tasks × ${v.cpus_per_task} CPU on ${v.nodes} node${v.nodes === 1 ? "" : "s"}, `
      + (v.mem_form === "mem" ? `${v.mem} per node` : `${v.mem} per CPU`);
    const res = s.resources || {}, mpi = s.mpi || {}, scr = s.scratch || {}, ex = s.slurm_extra || {}, ci = s.crystal_install || {};
    const rows = [];
    if (res.scf) rows.push(["Resources", `CRYSTAL ${H.esc(size(res.scf.value))}${src(res.scf)}`
      + (res.prop ? ` · properties ${H.esc(size(res.prop.value))}${src(res.prop)}` : "")]);
    if (mpi.launcher) rows.push(["MPI launch", `${H.esc(mpi.launcher)}${mpi.srun_mpi ? ` --mpi=${H.esc(mpi.srun_mpi)}` : ""}`
      + `${(mpi.env || []).length ? ` · ${H.esc((mpi.env || []).join(" "))}` : ""}${src(mpi)}`]);
    if (scr.order) rows.push(["Scratch", `${H.esc(scr.value || (scr.order || []).join(", then "))}${src(scr)}`]);
    const qos = ex.qos || {}, con = ex.constraint || {}, extra = ex.extra || {};
    if (ex.qos) rows.push(["QOS", `${H.esc(qos.value || "(none)")}${src(qos)}`]);
    if (ex.constraint) rows.push(["Constraint", `${H.esc(con.value || "(none)")}${src(con)}`]);
    if ((extra.value || []).length) rows.push(["Extra", `${H.esc(extra.value.join(" "))}${src(extra)}`]);
    if (ci.scf) rows.push(["Executables", `${H.esc(ci.bin_dir ? ci.bin_dir + "/" : "")}${H.esc(ci.scf)}, ${H.esc(ci.properties)}`
      + `${ci.bin_dir ? "" : ` <span class="muted">(from the CRYSTAL module)</span>`}${src(ci)}`]);
    if (!rows.length) return "";
    // each <dt>/<dd> a child of the <dl> itself (no <div> per pair: older screen-reader and axe
    // rules want the term and its value directly in the list); the <dl> is the two-column grid
    let html = `<dl class="sub-site" data-part="site-details">${rows.map(([k, v]) =>
      `<dt data-site="${H.esc(k.toLowerCase().replace(/ /g, "-"))}">${H.esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>`;
    html += `<p class="sub-line"><button type="button" class="tb-btn-text sm" data-act="site-check"${state.siteCheck && state.siteCheck.running ? " disabled" : ""}>`
      + `${state.siteCheck && state.siteCheck.running ? "Checking…" : "Check"}</button>
      <span class="muted">runs <code>mace site check</code> here (with <code>sbatch --test-only</code>; nothing is submitted)</span></p>`;
    html += siteCheckReport();
    return html;
  }

  function siteCheckReport() {
    const sc = state.siteCheck;
    if (!sc || sc.running) return "";
    if (sc.error) return `<p class="sub-line bad" data-part="site-check">${H.esc(sc.error)}</p>`;
    const rep = sc.report || {}, sm = rep.summary || {};
    const word = { ok: "OK", warn: "Warning", fail: "Failed", skip: "Skipped" };
    const cls = { ok: "good", warn: "warn", fail: "bad", skip: "muted" };
    return `<div class="sub-check" data-part="site-check" data-exit="${H.esc(String(rep.exit))}">
      <p class="sub-line">${H.esc(String(sm.ok || 0))} OK · ${H.esc(String(sm.warn || 0))} warnings · ${H.esc(String(sm.fail || 0))} failed</p>
      <ul>${(rep.items || []).map((i) => `<li data-status="${H.esc(i.status)}"><b class="${cls[i.status] || ""}">${H.esc(word[i.status] || i.status)}</b>
        ${H.esc(i.item)}: ${H.esc(i.detail)}${i.fix ? `<br><span class="muted">Fix: ${H.esc(i.fix)}</span>` : ""}</li>`).join("")}</ul></div>`;
  }

  async function siteCheck() {
    state.siteCheck = { running: true }; render();
    try {
      const r = await H.shell().apiTask("submit/site-check", { kind: "both", doc: "submit-dialog" });
      state.siteCheck = { report: r.report };
      // the Site row itself may have changed (a cache the check refreshed)
      try { const c = await H.api("submit/capabilities"); if (c && c.site) caps.site = c.site; } catch (_) { /* keep the row */ }
    } catch (e) {
      state.siteCheck = { error: e.message };
    }
    render();
  }

  function resourcesRow(p) {
    if (!p || state.where !== "slurm") return "";
    const rs = p.resources || [];
    if (!rs.length) return `<div class="sub-row"><span class="sub-k">Resources</span><div class="sub-v muted">—</div></div>`;
    const fmt = (r) => [r.ntasks && `${r.ntasks} cores`, r.mem_per_cpu && `${r.mem_per_cpu} per core`, r.mem && `${r.mem} memory`,
      r.walltime, r.account && `account ${r.account}`, r.partition && `partition ${r.partition}`, r.nodes && `-N ${r.nodes}`,
      r.exclude && `exclude ${r.exclude}`].filter(Boolean).join(", ");
    const lines = [];
    for (const a of p.arrays || []) lines.push(`1 array of ${plural(a.tasks, "task")}${a.throttle && a.throttle.includes("%") ? ` (at most ${a.throttle.split("%")[1]} at once)` : ""}: ${H.esc(fmt(a.request || {}))}`);
    for (const r of rs.filter((x) => !x.array).sort((x, y) => String(x.deck).localeCompare(String(y.deck), undefined, { numeric: true }))) {
      lines.push(`${H.esc(r.deck)}: ${H.esc(fmt(r))}${r.reused_script ? ' <span class="chip" data-part="reused">reused script</span>' : ""}`);
    }
    return `<div class="sub-row"><span class="sub-k">Resources</span><div class="sub-v" data-part="resources">${lines.map((l) => `<div>${l}</div>`).join("")}</div></div>`;
  }

  function listRow(label, items, part, empty = "—") {
    const shown = items.slice(0, 12);
    return `<div class="sub-row"><span class="sub-k">${label}</span><div class="sub-v mono small" data-part="${part}">${shown.length
      ? shown.map((x) => H.esc(x)).join("<br>") + (items.length > shown.length ? `<br><span class="muted">… and ${items.length - shown.length} more</span>` : "")
      : `<span class="muted">${empty}</span>`}</div></div>`;
  }

  function conflictsRow(p) {
    if (!p || state.where !== "slurm") return "";
    const c = p.conflicts || {}, parts = [];
    for (const f of c.finished || []) parts.push(`<p class="sub-line warn">${H.esc(base(f.out_rel))} exists (${H.esc(f.bucket)}) and would be overwritten</p>`);
    for (const f of c.wavefunction || []) parts.push(`<p class="sub-line warn">${H.esc(base(f.rel))} exists and would be replaced</p>`);
    const nOver = new Set([...(c.finished || []).map((f) => f.deck), ...(c.wavefunction || []).map((f) => base(f.rel).replace(/\.f9$/, ""))]).size;
    if ((c.finished || []).length || (c.wavefunction || []).length) {
      parts.push(`<label class="tog"><input type="checkbox" data-tick="overwrite"${state.ticks.overwrite ? " checked" : ""}><span>Overwrite finished results for ${plural(nOver, "deck")}</span></label>`);
    }
    for (const q of c.queued || []) parts.push(`<p class="sub-line warn">${H.esc(q.deck)} is already queued (${H.esc(q.job_id || q.calc_id || "")})</p>`);
    if ((c.queued || []).length) {
      parts.push(`<label class="tog"><input type="checkbox" data-tick="duplicate"${state.ticks.duplicate ? " checked" : ""}><span>Submit again although ${plural(new Set(c.queued.map((q) => q.deck)).size, "deck is", "decks are")} already queued</span></label>`);
    }
    if ((p.needs || {}).account) {
      parts.push(`<label class="tog"><input type="checkbox" data-tick="account"${state.ticks.account ? " checked" : ""}><span>Submit with account ${H.esc(((p.site || caps.site || {}).account || {}).value || "")} anyway</span></label>`);
    }
    if (!parts.length) return "";
    return `<div class="sub-row"><span class="sub-k">Conflicts</span><div class="sub-v" data-part="conflicts">${parts.join("")}</div></div>`;
  }

  function optionsHtml() {
    if (state.where === "local") {
      const o = state.opts.local;
      return `<div class="sub-row"><span class="sub-k">Options</span><div class="sub-v sub-opts">
        <label class="f inline">Cores <input class="field num" type="number" min="1" step="1" data-o="cores" value="${H.esc(o.cores)}" placeholder="default" aria-label="Cores"></label>
        <label class="f inline">Timeout per deck <input class="field" data-o="timeout" value="${H.esc(o.timeout)}" placeholder="30m" aria-label="Timeout per deck" spellcheck="false"></label>
        <label class="tog"><input type="checkbox" data-o="track"${o.track ? " checked" : ""}><span>Track in materials.db</span></label>
        <p class="sub-line muted">The run stops if MACE GUI stops or your connection to this node closes.</p></div></div>`;
    }
    const o = state.opts.slurm, p = state.preview;
    const tpl = (p && p.templates) || [];
    return `<div class="sub-row"><span class="sub-k">Options</span><div class="sub-v sub-opts">
      <label class="tog"${isFolder ? "" : ' title="Job arrays submit a whole folder. Use Batch Setup to collect these decks in their own folder."'}><input type="checkbox" data-o="array"${o.array ? " checked" : ""}${isFolder ? "" : " disabled"}><span>Submit as job arrays</span></label>
      <label class="f inline">Max running at once <input class="field num" type="number" min="1" step="1" data-o="max_running" value="${H.esc(o.max_running)}"${o.array ? "" : " disabled"} aria-label="Max running at once"></label>
      <label class="tog"><input type="checkbox" data-o="track"${o.track || o.progress ? " checked" : ""}${o.progress ? " disabled" : ""}><span>Track in materials.db (enables mace status)</span></label>
      <label class="f inline">Then run the rest of a workflow <select class="field" data-o="progress" aria-label="Workflow">
        <option value="">(no)</option>${tpl.map((t) => `<option value="${H.esc(t.name)}"${t.ok ? "" : " disabled"}${o.progress === t.name ? " selected" : ""}>${H.esc(t.name)}${t.ok ? ` (${H.esc(t.sequence.join(" → "))})` : ` — ${H.esc(t.why)}`}</option>`).join("")}</select></label>
      <label class="f inline">Walltime <input class="field" data-o="walltime" value="${H.esc(o.walltime)}" placeholder="the job script's default" spellcheck="false" aria-label="Walltime"></label>
      <fieldset class="sub-ex"><legend>Exclude</legend>
        <label class="tog"><input type="radio" name="sub-ex" value="none"${o.exclude.kind === "none" ? " checked" : ""}><span>none</span></label>
        <label class="tog"><input type="radio" name="sub-ex" value="nodes"${o.exclude.kind === "nodes" ? " checked" : ""}><span>these nodes</span></label>
        <label class="tog"><input type="radio" name="sub-ex" value="type"${o.exclude.kind === "type" ? " checked" : ""}><span>a node type</span></label>
        <input class="field" data-o="exclude-value" value="${H.esc(o.exclude.value || "")}"${o.exclude.kind === "none" ? " disabled" : ""} aria-label="Exclude" spellcheck="false"></fieldset>
      ${p && (p.existing_scripts || []).length ? `<label class="tog"><input type="checkbox" data-o="overwrite_sh"${o.overwrite_sh ? " checked" : ""}><span>Regenerate job scripts (--overwrite-sh)</span></label>` : ""}
      </div></div>`;
  }

  function render() {
    d.querySelector("#submit-title").textContent = title();
    const p = state.preview;
    const decks = (p && p.decks) || targets;
    let html = "";
    if (caps.network && state.where === "slurm") html += `<p class="sub-line bad">Submitting is turned off while MACE GUI is open to the network</p>`;
    if (caps.in_slurm_job && state.where === "slurm") html += `<p class="sub-line muted">MACE GUI runs inside job ${H.esc(caps.slurm_job_id)}; the jobs you submit do not inherit its SLURM settings.</p>`;
    if (note) html += `<p class="sub-line muted">${H.esc(note)}</p>`;
    html += `<div class="sub-row"><span class="sub-k">Decks (${decks.length})</span><div class="sub-v mono small" data-part="decks">${decks.slice(0, 8).map((x) => H.esc(base(x))).join("  ")}${decks.length > 8 ? ` <span class="muted">… and ${decks.length - 8} more</span>` : ""}</div></div>`;
    html += `<div class="sub-row"><span class="sub-k">Run on</span><div class="sub-v" role="radiogroup" aria-label="Run on">
      <label class="tog"><input type="radio" name="sub-where" value="slurm"${state.where === "slurm" ? " checked" : ""}${caps.sbatch ? "" : " disabled"}><span>SLURM (sbatch)</span></label>
      <label class="tog"><input type="radio" name="sub-where" value="local"${state.where === "local" ? " checked" : ""}><span>This node (mace run)</span></label>
      ${caps.sbatch ? "" : '<span class="muted small">No SLURM on this machine</span>'}</div></div>`;
    if (state.mixed !== null || (p && p.mixed)) {
      html += `<div class="sub-row"><span class="sub-k">Decks to submit</span><div class="sub-v"><select class="field" data-o="mixed" aria-label="Decks to submit">
        <option value="">choose…</option><option value="1"${state.mixed === "1" ? " selected" : ""}>D12 decks only</option>
        <option value="2"${state.mixed === "2" ? " selected" : ""}>D3 decks only</option><option value="3"${state.mixed === "3" ? " selected" : ""}>D12 and D3 decks</option></select></div></div>`;
    }
    html += optionsHtml();
    html += resourcesRow(p);
    html += siteRow(p);
    if (p) {
      html += `<div class="sub-row"><span class="sub-k">Command</span><div class="sub-v"><code class="mono" data-part="command">${H.esc(p.command || "")}</code>
        <button type="button" class="tb-btn-text sm" data-copy-cmd="1">Copy</button></div></div>`;
      if (state.where === "slurm") {
        html += listRow("Will write", p.writes || [], "writes", "nothing new");
        html += listRow("The jobs write", p.job_writes || [], "job-writes");
      } else {
        html += listRow("The run writes", p.run_writes || [], "run-writes");
        if (p.refused) html += `<p class="sub-line bad" data-part="refused">${H.esc(p.refused)}</p>`;
      }
      html += conflictsRow(p);
      const warns = (p.warnings || []);
      if (warns.length) html += `<div class="sub-row"><span class="sub-k">Warnings</span><div class="sub-v" data-part="warnings">${warns.map((w) => `<p class="sub-line warn">${H.esc(w)}</p>`).join("")}</div></div>`;
      if (p.code && state.where === "slurm") html += `<div class="sub-row"><span class="sub-k">The CLI said</span><div class="sub-v"><pre class="sub-log" data-part="log">${H.esc((p.cli_log || "").split("\n").slice(-25).join("\n"))}</pre></div></div>`;
    }
    if (state.previewing) html += `<p class="sub-line muted" data-part="previewing">Previewing…</p>`;
    // many decks: every preview is explicit; a refused Submit (409): the preview no longer holds
    if (!state.previewing && state.stale && p && (many() || state.refused)) {
      html += `<p><button type="button" class="tb-btn-text" data-act="update">Update preview</button></p>`;
    }
    if (state.error) html += `<p class="sub-line bad" role="alert" data-part="error">${H.esc(state.error)}</p>`;
    if (state.running) html += `<p class="sub-line" data-part="running">${state.where === "local" ? "Running" : "Submitting"}… <span data-part="elapsed">0</span> s.
      ${state.where === "local" ? "Cancel it in Tasks." : "Submitting cannot be stopped halfway; cancel the jobs in Jobs afterwards."}</p>`;
    const keep = document.activeElement && body.contains(document.activeElement) ? document.activeElement : null;
    const keepSel = keep && (keep.dataset.o ? `[data-o="${keep.dataset.o}"]` : keep.name ? `[name="${keep.name}"][value="${keep.value}"]` : keep.dataset.tick ? `[data-tick="${keep.dataset.tick}"]` : null);
    body.innerHTML = html;
    if (keepSel) { const el = body.querySelector(keepSel); if (el) el.focus(); }
    renderFoot();
  }

  async function preview() {
    clearTimeout(timer);
    const mySeq = ++state.seq;
    state.previewing = true; state.error = ""; state.refused = false; render();
    const req = { targets, where: state.where, options: optionsFor(state.where), doc: "submit-dialog" };
    if (state.mixed) req.mixed_choice = state.mixed;
    try {
      const r = await H.shell().apiTask("submit/preview", req);
      if (mySeq !== state.seq) return;
      state.preview = r; state.stale = false;
      // a tick belongs to the preview it was given for
      state.ticks = {};
    } catch (e) {
      if (mySeq !== state.seq) return;
      if (e.data && e.data.mixed) { state.mixed = state.mixed || ""; state.error = "This folder has D12 and D3 decks: choose which to submit"; }
      else state.error = e.message;
      state.preview = state.preview && !e.data?.mixed ? state.preview : null;
      state.stale = true;
    } finally {
      if (mySeq === state.seq) { state.previewing = false; render(); }
    }
  }
  function changed() {
    state.stale = true;
    rememberOptions(H, state.where, state.opts[state.where]);
    if (many()) { render(); return; }
    renderFoot();
    clearTimeout(timer);
    timer = setTimeout(preview, DEBOUNCE_MS);
  }

  async function submit() {
    if (!canSubmit()) return;
    state.running = true; state.error = ""; render();
    const t0 = Date.now();
    const tick = setInterval(() => { const el = body.querySelector('[data-part="elapsed"]'); if (el) el.textContent = String(Math.round((Date.now() - t0) / 1000)); }, 1000);
    const ticks = { confirm_overwrite: !!state.ticks.overwrite, confirm_duplicate: !!state.ticks.duplicate,
      confirm_account: !!state.ticks.account, confirm_limits: !!state.ticks.limits };
    try {
      const pr = H.shell().apiTask("submit/run", { preview_id: state.preview.preview_id, ...ticks });
      if (state.where === "local") {
        // a local run is a long task: the dialog closes, Tasks follows it
        d.close();
        H.toast(`Running ${plural((state.preview.decks || []).length, "deck")} on this node: follow it in Tasks`, "info",
          { label: "Open Tasks", run: () => H.shell().showBottom("tasks") });
        const r = await pr;
        if (r.code) H.toast(`mace run exited with ${r.code}; see the task's log`, "warning", { label: "Open Tasks", run: () => H.shell().showBottom("tasks") });
        else H.toast("Run on This Node finished", "success");
        (r.problems || []).forEach((m) => H.shell().problem({ severity: "error", source: "Run on This Node", message: m }));
        return;
      }
      const r = await pr;
      clearInterval(tick);
      state.running = false;
      if (r.code) {
        state.error = `mace submit exited with ${r.code}`;
        state.preview = { ...state.preview, cli_log: r.log, code: r.code };
        render();
        (r.problems || []).forEach((m) => H.shell().problem({ severity: "error", source: "Submit", message: m }));
        return;
      }
      d.close();
      const ids = r.job_ids || [];
      H.toast(ids.length ? `Submitted ${plural(ids.length, "job")}: ${ids.slice(0, 6).join(", ")}${ids.length > 6 ? ", …" : ""}` : "Submitted",
        "success", { label: "Open Jobs", run: () => H.shell().showBottom("jobs") });
      if ((r.script_drift || []).length) {
        H.toast(`The submitted job scripts differ from the preview (${r.script_drift.map((x) => base(x.script)).join(", ")}): the cluster changed in between`, "warning");
        r.script_drift.forEach((x) => H.shell().problem({ severity: "warning", source: "Submit", message: `The submitted job scripts differ from the preview: ${x.script}\n${x.diff}` }));
      }
      document.dispatchEvent(new CustomEvent("mace-jobs-changed", { detail: { ids } }));
      if (H.shell().refreshFiles) H.shell().refreshFiles();
    } catch (e) {
      clearInterval(tick);
      state.running = false;
      state.error = e.message;
      if (e.status === 409) { state.stale = true; state.refused = true; }
      if (d.open) render(); else H.toast(e.message, "error");
    }
  }

  // events
  body.addEventListener("change", (e) => {
    const t = e.target;
    if (t.name === "sub-where") { state.where = t.value; state.preview = null; state.ticks = {}; render(); preview(); return; }
    if (t.name === "sub-ex") { state.opts.slurm.exclude = { kind: t.value, value: state.opts.slurm.exclude.value || "" }; render(); changed(); return; }
    if (t.dataset.tick) { state.ticks[t.dataset.tick] = t.checked; renderFoot(); return; }
    const k = t.dataset.o;
    if (!k) return;
    if (k === "mixed") { state.mixed = t.value || null; if (state.mixed) preview(); return; }
    const o = state.opts[state.where];
    if (k === "exclude-value") o.exclude = { ...o.exclude, value: t.value.trim() };
    else if (t.type === "checkbox") o[k] = t.checked;
    else o[k] = t.value.trim();
    if (k === "array" || k === "progress") render();
    changed();
  });
  body.addEventListener("input", (e) => {
    const t = e.target, k = t.dataset.o;
    if (!k || t.type === "checkbox" || k === "mixed") return;
    const o = state.opts[state.where];
    if (k === "exclude-value") o.exclude = { ...o.exclude, value: t.value.trim() };
    else o[k] = t.value.trim();
    changed();
  });
  body.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    if (b.dataset.act === "update") preview();
    else if (b.dataset.act === "site-check") siteCheck();
    else if (b.dataset.act === "site-help") {
      const pop = b.parentElement.querySelector(".sub-pop");
      pop.hidden = !pop.hidden; b.setAttribute("aria-expanded", String(!pop.hidden));
    } else if (b.dataset.copy) H.shell().copyText(b.dataset.copy, "Copied");
    else if (b.dataset.copyCmd && state.preview) H.shell().copyText(state.preview.command, "Command copied");
  });
  d.querySelector('[data-act="cancel"]').addEventListener("click", () => d.close());
  d.querySelector("[data-close]").addEventListener("click", () => d.close());
  btn.addEventListener("click", submit);
  // Enter never submits (an explicit click or Space on the Submit button does)
  d.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !(e.target.tagName === "BUTTON" && e.target !== btn)) { e.preventDefault(); e.stopPropagation(); }
    if (e.key === "Escape" && state.running && state.where === "slurm") { e.preventDefault(); }
  });
  d.addEventListener("cancel", (e) => { if (state.running && state.where === "slurm") e.preventDefault(); });
  return new Promise((resolve) => {
    d.addEventListener("close", () => { clearTimeout(timer); state.seq++; if (opener && opener.focus && opener.isConnected) opener.focus(); resolve(); d.remove(); }, { once: true });
    render();
    d.showModal();
    d.querySelector('[data-act="cancel"]').focus();
    preview();
  });
}
