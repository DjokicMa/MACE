/* The Workflows document (GUI_WORKFLOW_DESIGN.md section 4.7, Phase 3): a
 * read-only view of the workflows under this folder, the ones started from
 * the CLI included. Everything comes from /api/workflow/list and
 * /api/workflow/status, which derive it on each call from the plan, the
 * workflow's database (read-only), your SLURM jobs and the control file:
 * the same answer as `mace workflow --status ID --json`.
 *
 * Live: the server sends workflow.changed after a poll of your jobs that
 * changed a step (and jobs.finished when a job leaves the queue); the shell
 * passes both to changed(). While the view is shown it also refreshes every
 * poll interval, so the time left of running steps moves on.
 *
 * Actions (Phase 5): each button runs one CLI command in the workflow's
 * folder (workflow_actions.py): Pause, Resume and Cancel are
 * `mace workflow --pause|--resume|--cancel ID` (Cancel lists the exact jobs
 * it would scancel and asks first); a failed step's Recover previews
 * `mace recover --calc-id ID --dry-run`, then runs it; a step that was not
 * submitted gets Process Now (`mace workflow --process ID`); Clone opens the
 * builder pre-filled from the workflow. Resume, Cancel, Recover and Process
 * Now are turned off while MACE GUI is open to the network. Recover is the
 * Jobs panel's flow (recover_flow.js). */
import { recoverFlow } from "./recover_flow.js";
const STATE_ICON = { planned: "circle-dot", waiting: "clock", stale: "triangle-alert", held: "pause", paused: "pause",
  canceled: "circle-x", queued: "clock", running: "loader-circle", ending: "loader-circle", completed: "circle-check",
  failed: "circle-alert", recovering: "rotate-cw", blocked: "minus" };
const TONE = { planned: "idle", waiting: "info", stale: "warn", held: "warn", paused: "warn", canceled: "idle",
  queued: "info", running: "accent", ending: "info", completed: "ok", failed: "bad", recovering: "warn", blocked: "bad",
  attention: "bad" };
const ORDER = ["planned", "waiting", "queued", "running", "ending", "completed", "failed", "recovering", "stale", "held",
  "blocked", "paused", "canceled"];

function when(iso) {
  if (!iso) return "";
  const d = new Date(String(iso).replace(" ", "T"));
  if (isNaN(d)) return String(iso);
  const today = new Date().toDateString() === d.toDateString();
  return `${today ? "today" : d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })} ` +
    d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
}
const keyOf = (w) => `${w.folder_rel || ""}::${w.workflow_id}`;
const stepName = (s) => { const m = /^(.*)_(\d+)$/.exec(s || ""); return m ? `${m[1]} (step ${m[2]})` : s; };

export function createWorkflowView(host, H) {
  const { api, esc, icon, toast } = H;
  let cur = null, list = null, status = null, timer = 0, loading = null, again = false, pollS = 60, shown = false;
  let busy = "";             // the action running now (its buttons are disabled meanwhile)
  host.innerHTML = `<div class="wfv">
    <div class="wfv-head"><h2>Workflows</h2><span class="wfv-strip muted small" aria-live="polite"></span>
      <button type="button" class="tb-btn-text wfv-refresh">${icon("refresh-cw")}<span>Refresh</span></button>
      <button type="button" class="tb-btn-text primary wfv-new">${icon("plus")}<span>New Workflow…</span></button></div>
    <div class="wfv-body">
      <nav class="wfv-list" aria-label="Workflows in this folder"></nav>
      <section class="wfv-detail" aria-live="off" tabindex="-1"></section>
    </div></div>`;
  const $ = (s) => host.querySelector(s);
  const listEl = $(".wfv-list"), detail = $(".wfv-detail"), strip = $(".wfv-strip");

  /* Replace a region only when its content changed (a refresh that changed
   * nothing leaves the buttons, and a click on them, alone); keep the focus
   * on the same workflow or step button across a real change. */
  function setHtml(el, html) {
    if (el._html === html) return;
    const f = document.activeElement && el.contains(document.activeElement) ? document.activeElement : null;
    const sel = f && (f.dataset.key ? `[data-key="${CSS.escape(f.dataset.key)}"]`
      : f.dataset.mid !== undefined && f.dataset.step ? `.wfv-node[data-mid="${CSS.escape(f.dataset.mid)}"][data-step="${CSS.escape(f.dataset.step)}"]`
        : f.dataset.open ? `[data-open="${CSS.escape(f.dataset.open)}"]` : null);
    el.innerHTML = html;
    el._html = html;
    if (sel) { const again = el.querySelector(sel); if (again) again.focus(); }
  }

  const badge = (state, label, help) =>
    `<span class="wfv-badge tone-${TONE[state] || "idle"}" title="${esc(help || "")}">${icon(STATE_ICON[state] || "circle-dot")}<span>${esc(label || state)}</span></span>`;

  function renderList() {
    const items = (list && list.workflows) || [];
    if (!list) { setHtml(listEl, `<p class="muted small wfv-pad">Loading…</p>`); return; }
    if (!items.length) { setHtml(listEl, ""); return; }
    const sel = cur && cur.wfKey;
    setHtml(listEl, `<ul role="list">${items.map((w) => {
      const k = keyOf(w), on = k === sel;
      const sub = [w.folder_rel ? w.folder_rel : "this folder", `${w.materials} material${w.materials === 1 ? "" : "s"}`].join(" · ");
      return `<li><button type="button" class="wfv-item${on ? " on" : ""}" data-key="${esc(k)}" aria-current="${on ? "true" : "false"}">
        <span class="wfv-item-title">${esc(w.label || "Unnamed workflow")}</span>
        ${badge(w.state, w.state_label, w.state_help)}
        ${w.held_total ? `<span class="wfv-count tone-warn small" title="${esc(heldText(w.held))}">Held <b>${w.held_total}</b></span>` : ""}
        <span class="wfv-item-seq small">${esc((w.sequence || []).join(" → "))}</span>
        <span class="wfv-item-sub muted small">${esc(sub)}</span>
        <span class="wfv-item-sub muted small mono">${esc(w.workflow_id)}</span></button></li>`;
    }).join("")}</ul>`);
  }

  function emptyHtml() {
    return `<div class="wfv-empty"><h3>No workflows in this folder</h3>
      <p>A workflow runs a chain of calculations (for example OPT → SP → BAND → DOSS) for each structure, submitting each step when the one before it finishes.
      Workflows started from the command line show here too, with every step's state.</p>
      <p><button type="button" class="tb-btn-text primary wfv-new">${icon("plus")}<span>New Workflow…</span></button>
        <span class="muted small">pick structures and steps; you see exactly what runs before anything is submitted</span></p>
      <p>Or from the command line, for example:</p>
      <div class="wfv-cmd"><code>mace submit --progress opt_sp *.d12</code><button type="button" class="tb-btn-text" data-copy="mace submit --progress opt_sp *.d12">${icon("copy")}<span>Copy</span></button></div>
      <p class="muted small">Workflows are found in this folder and two levels of subfolders (wherever a <code>workflow_configs/</code> folder holds a plan).</p></div>`;
  }

  function nodeCell(m, n) {
    const sel = cur && cur.wfNode && cur.wfNode.mid === m.material_id && cur.wfNode.step === n.step;
    const eta = n.eta || null;
    const pct = eta && eta.percent !== null && eta.percent !== undefined ? Math.max(0, Math.min(100, eta.percent)) : null;
    const extra = n.state === "running" ? (eta ? esc(eta.text) : "time left unknown") :
      n.attempts > 1 ? `attempt ${n.attempts}` : "";
    const aria = `${m.material_id}, ${stepName(n.step)}: ${n.label}${extra ? `, ${extra}` : ""}`;
    return `<td><button type="button" class="wfv-node tone-${TONE[n.state] || "idle"}${sel ? " on" : ""}" data-mid="${esc(m.material_id)}" data-step="${esc(n.step)}"
      aria-pressed="${sel ? "true" : "false"}" aria-label="${esc(aria)}" title="${esc(n.help || "")}">
      <span class="wfv-node-state">${icon(STATE_ICON[n.state] || "circle-dot")}<span>${esc(n.label)}</span>${n.attempts > 1 ? `<span class="wfv-att" title="Attempts (recoveries included)">×${n.attempts}</span>` : ""}</span>
      ${extra && n.state === "running" ? `<span class="wfv-node-eta small">${extra}</span>` : ""}
      ${pct !== null && n.state === "running" ? `<span class="wfv-bar" aria-hidden="true"><span style="width:${pct}%"></span></span>` : ""}
    </button></td>`;
  }

  function nodeDetail(st) {
    const sel = cur && cur.wfNode;
    if (!sel) return `<p class="muted small wfv-hint">Select a step to see its job, files and progress.</p>`;
    const m = (st.materials || []).find((x) => x.material_id === sel.mid);
    const n = m && m.nodes.find((x) => x.step === sel.step);
    if (!n) return "";
    const rows = [];
    const add = (k, v) => { if (v !== null && v !== undefined && v !== "") rows.push(`<tr><th scope="row">${esc(k)}</th><td>${v}</td></tr>`); };
    add("State", `${badge(n.state, n.label, n.help)} <span class="small">${esc(n.help || "")}</span>`);
    if (n.reason) add("Why", esc(n.reason));
    add("Built from", n.parent ? esc(stepName(n.parent)) : "the input structure");
    if (n.job_id) add("SLURM job", `<span class="mono">${esc(n.job_id)}</span>${n.squeue_state ? ` <span class="muted small">(${esc(n.squeue_state.toLowerCase())})</span>` : ""}`);
    if (n.elapsed) add("Run time", `${esc(n.elapsed)}${n.time_limit ? ` of ${esc(n.time_limit)}` : ""}`);
    if (n.eta) {
      add("Time left", esc(n.eta.text));
      if (n.eta.summary) add("Progress", esc(n.eta.summary));
      if (n.eta.basis && n.eta.basis !== n.eta.text) add("Estimate", `<span class="small">${esc(n.eta.basis)}</span>`);
      if (n.eta.risk && n.eta.risk.sentence) add("Walltime", `<span class="small wfv-risk">${esc(n.eta.risk.sentence)}</span>`);
    }
    if (n.attempts > 1) add("Attempts", `${n.attempts} <span class="muted small">(recoveries resubmit a failed step)</span>`);
    if (n.calc_id) add("Calculation", `<span class="mono small">${esc(n.calc_id)}</span>`);
    if (n.error_type) add("Error", esc(n.error_type));
    if (n.submitted_at) add("Submitted", esc(when(n.submitted_at)));
    if (n.completed_at) add("Finished", esc(when(n.completed_at)));
    const btns = [];
    if (n.out_rel) btns.push(`<button type="button" class="tb-btn-text" data-open="${esc(n.out_rel)}" data-kind="out">${icon("file-text")}<span>Open Output</span></button>`);
    if (n.deck_rel) btns.push(`<button type="button" class="tb-btn-text" data-open="${esc(n.deck_rel)}" data-kind="deck">${icon("file-code")}<span>Open Input Deck</span></button>`);
    if (n.script_rel) btns.push(`<button type="button" class="tb-btn-text" data-open="${esc(n.script_rel)}" data-kind="script">${icon("terminal")}<span>Open Job Script</span></button>`);
    if (n.job_id) btns.push(`<button type="button" class="tb-btn-text" data-job="${esc(n.job_id)}">${icon("activity")}<span>Job Details</span></button>`);
    const acts = nodeActions(st, m, n);
    return `<h3>${esc(m.material_id)} · ${esc(stepName(n.step))}</h3>
      <table class="wfv-kv">${rows.join("")}</table>
      ${acts}
      ${btns.length ? `<div class="wfv-actions">${btns.join("")}</div>` : `<p class="muted small">No job or files yet.</p>`}`;
  }

  /* ------------------------------------------------------------ actions */
  const network = (st) => !!((st && st.network) || (list && list.network));
  const NET = "Turned off while MACE GUI is open to the network";
  const held = (st) => { const c = (st.control || {}).state; return c === "paused" || c === "canceled" ? c : ""; };
  function actBtn(act, ic, label, help, why, primary) {
    const off = !!why || !!busy;
    return `<button type="button" class="tb-btn-text${primary ? " primary" : ""}" data-wact="${act}"${off ? " disabled" : ""}
      title="${esc(why || help)}"${why ? ` aria-describedby="wfv-why-${act}"` : ""}>${icon(busy === act ? "loader-circle" : ic)}<span>${esc(label)}</span></button>`;
  }
  /** What a selected step can do: Recover a failed one, Process Now one that was not submitted. */
  function nodeActions(st, m, n) {
    const h = held(st), net = network(st);
    const out = [];
    if (n.state === "failed" && n.calc_id) {
      const why = h ? `Resume the workflow first: a ${h} workflow submits no recovery job` : net ? NET : "";
      out.push(`<div class="wfv-act-row">${actBtn("recover", "rotate-cw", "Recover…", "Show what mace recover would fix for this step, then resubmit it", why, true)}
        <span class="small muted">${esc(why || "Shows the fix first (mace recover --dry-run); nothing is resubmitted until you confirm.")}</span></div>`);
    }
    if (n.state === "stale") {
      const why = h ? `The workflow is ${h}: Resume starts its held steps` : net ? NET : "";
      out.push(`<div class="wfv-act-row">${actBtn("process", "play", "Process Now", "Submit this step the way the finished step's completion would have", why, true)}
        <span class="small muted">${esc(why || "Runs mace workflow --process: the steps whose completion step did not run are submitted now.")}</span></div>`);
    }
    return out.join("");
  }
  /** The workflow's own buttons (they replace the copyable command of Phase 3). */
  function wfActions(st) {
    const c = (st.control || {}).state || "active", net = network(st), done = st.state === "completed";
    const b = [];
    if (c === "active" && !done) b.push(actBtn("pause", "pause", "Pause", "Submit nothing new until it is resumed; queued and running jobs go on", ""));
    if (c !== "active") b.push(actBtn("resume", "play", "Resume", "Lift the hold and submit the steps held meanwhile", net ? NET : "", true));
    if (c !== "canceled" && !done) b.push(actBtn("cancel", "circle-x", "Cancel Workflow…", "Stop the workflow and cancel its queued and running jobs (they are listed first)", net ? NET : ""));
    const cl = st.clone || {};
    b.push(actBtn("clone", "copy", "Clone…", "A new workflow with the same structures, steps and settings, in the builder", cl.available ? "" : (cl.reason || "Cannot be cloned")));
    return `<div class="wfv-wf-actions" role="group" aria-label="Workflow actions">${b.join("")}</div>`;
  }
  const where = () => ({ folder: (status && status.folder_rel) || "", id: status && status.workflow_id });
  async function act(name, fn) {
    if (busy || !status || status.error) return;
    busy = name;
    renderDetail();
    try { await fn(); } catch (e) { if (e.state !== "canceled") toast(e.message, "error"); } finally {
      busy = "";
      await load();
      renderDetail();
    }
  }
  const control = (action, extra = {}) => H.apiTask("workflow/control", { ...where(), action, ...extra });
  async function pause() {
    const r = await control("pause");
    if (r.ok) toast("Paused: nothing new is submitted until you resume it", "success"); else toast(r.error || "Pause failed", "error");
  }
  async function resume() {
    const r = await control("resume");
    if (!r.ok) { toast(r.error || "Resume failed", "error"); return; }
    toast(r.steps_started ? `Resumed: ${r.steps_started} held step${r.steps_started === 1 ? "" : "s"} submitted` : "Resumed", "success");
  }
  async function cancel() {
    const l = await control("cancel");
    if (!l.confirm) { toast(l.error || "Cancel failed", "error"); return; }
    const jobs = l.jobs || [], n = jobs.length;
    const pre = jobs.map((j) => `${j.job_id}  ${j.calc_type || ""}  ${j.material_id || ""}  (${String(j.state || "").toLowerCase()})`).join("\n");
    const body = n ? `Progression stops (Resume lifts it), and scancel cancels exactly these ${n === 1 ? "job" : `${n} jobs`} of this workflow. This runs: ${l.run_argv.join(" ")}`
      : l.squeue ? `Progression stops (Resume lifts it). None of its jobs is queued or running, so no job is canceled. This runs: ${l.run_argv.join(" ")}`
        : `Progression stops (Resume lifts it). SLURM cannot be asked from here, so no job is canceled. This runs: ${l.run_argv.join(" ")}`;
    const ok = await H.confirm({ title: "Cancel Workflow?", body, pre, yes: n ? `Cancel Workflow and ${n === 1 ? "1 Job" : `${n} Jobs`}` : "Cancel Workflow",
      no: "Keep It Running", defaultNo: true });
    if (!ok) return;
    const r = await control("cancel", { confirm: true, jobs: jobs.map((j) => j.job_id) });
    if (r.changed) { toast(r.error, "warning"); return; }
    if (!r.ok && !(r.refused || []).length) { toast(r.error || "Cancel failed", "error"); return; }
    toast((r.canceled || []).length ? `Canceled; scancel ${r.canceled.join(", ")}` : "Canceled", "success");
    for (const x of r.refused || []) toast(`${x.job_id}: ${x.reason}`, "warning");
  }
  async function recover(n) {
    await recoverFlow(H, { one: true, title: `Recover ${cur.wfNode.mid} · ${stepName(n.step)}?`,
      preview: () => H.apiTask("workflow/recover/preview", { ...where(), calc_id: n.calc_id }),
      run: (b) => H.apiTask("workflow/recover/run", b) });
  }
  async function processNow() {
    const r = await H.apiTask("workflow/process", where());
    if (!r.ok) { toast(r.error || "Process Now failed", "error"); return; }
    toast(r.steps_started ? `${r.steps_started} step${r.steps_started === 1 ? "" : "s"} submitted` : "Nothing to submit: the step was started meanwhile, or its completion was already handled", r.steps_started ? "success" : "warning");
  }
  async function cloneIt() {
    const r = await api(`workflow/clone?id=${encodeURIComponent(where().id)}&folder=${encodeURIComponent(where().folder)}`);
    if (!r.available) { toast(r.reason || "This workflow cannot be cloned", "warning"); return; }
    if (H.newWorkflow) await H.newWorkflow({ clone: r });
  }

  function controlNote(st) {
    const c = (st.control || {}).state;
    let html = "";
    if (c === "paused") html = `<div class="wfv-note warn">${icon("pause")}<div><b>Paused.</b> Completions are recorded; no next step and no recovery job is submitted until you click Resume.</div></div>`;
    else if (c === "canceled") html = `<div class="wfv-note warn">${icon("circle-x")}<div><b>Canceled.</b> Progression stopped; Resume lifts it.</div></div>`;
    if (!st.auto_progression) html += `<div class="wfv-note warn">${icon("pause")}<div><b>Automatic progression is off</b> in this plan: each next step waits for you.</div></div>`;
    if (st.squeue && !st.squeue.available) html += `<div class="wfv-note">${icon("info")}<div>SLURM cannot be asked from here: queued and running states are the database's.</div></div>`;
    if (network(st)) html += `<div class="wfv-note">${icon("info")}<div>MACE GUI is open to the network: Resume, Cancel, Recover and Process Now are turned off (Pause still works).</div></div>`;
    return html;
  }

  const HELD_WORDS = { budget: "queue budget full", workflow_cap: "the workflow's max_jobs reached",
    slurm_limit: "SLURM refused it (submit limit)" };
  function heldText(held) {
    return Object.keys(held || {}).sort().map((k) => `${held[k]} ${HELD_WORDS[k] || k}`).join(", ");
  }
  function counts(st) {
    const c = st.counts || {}, labels = st.states || {};
    return ORDER.filter((k) => c[k]).map((k) => {
      const tip = k === "held" && st.held && Object.keys(st.held).length ? ` title="${esc(heldText(st.held))}"` : "";
      return `<span class="wfv-count tone-${TONE[k] || "idle"}"${tip}>${esc((labels[k] || [k])[0])} <b>${c[k]}</b></span>`;
    }).join("");
  }
  function heldNote(st) {
    const h = st.held || {}, n = Object.values(h).reduce((a, b) => a + b, 0);
    if (!n) return "";
    return `<div class="wfv-note">${icon("pause")}<div>${n} job${n === 1 ? " is" : "s are"} held (${esc(heldText(h))}): MACE submits ${n === 1 ? "it" : "them"} as slots of your queue budget free up. <span class="muted">mace queue status shows the budget; mace queue drain submits what fits now.</span></div></div>`;
  }

  function renderDetail() {
    if (!list) { setHtml(detail, ""); return; }
    if (!(list.workflows || []).length) { setHtml(detail, emptyHtml()); return; }
    const st = status;
    if (!st) { setHtml(detail, `<p class="muted">Loading…</p>`); return; }
    if (st.error) { setHtml(detail, `<p class="wfv-note warn">${icon("triangle-alert")}<span>${esc(st.error)}</span></p>`); return; }
    const seq = st.sequence || [];
    const meta = [];
    const add = (k, v) => { if (v) meta.push(`<div><dt>${esc(k)}</dt><dd>${v}</dd></div>`); };
    add("Workflow ID", `<span class="mono">${esc(st.workflow_id)}</span>`);
    add("Folder", esc(st.folder_rel || "this folder"));
    add("Started", esc(when(st.created)));
    add("Steps", esc(seq.join(" → ")));
    add("Plan", st.plan_rel ? `<button type="button" class="wfv-link" data-open="${esc(st.plan_rel)}" data-kind="plan">${esc(st.plan_rel.split("/").pop())}</button>` : "");
    add("Database", esc(st.database_rel || (st.database ? st.database : "none yet")));
    const head = seq.map((s, i) => {
      const parent = (st.materials[0] && st.materials[0].nodes[i] && st.materials[0].nodes[i].parent) || null;
      return `<th scope="col"><span class="wfv-col">${esc(s)}</span><span class="muted small">${parent ? `from ${esc(parent.replace(/_(\d+)$/, ""))}` : "first step"}</span></th>`;
    }).join("");
    const body = (st.materials || []).map((m) =>
      `<tr><th scope="row" class="mono">${esc(m.material_id)}</th>${m.nodes.map((n) => nodeCell(m, n)).join("")}</tr>`).join("");
    const legend = ORDER.filter((k) => st.states && st.states[k]).map((k) =>
      `<li>${badge(k, st.states[k][0], "")}<span class="small">${esc(st.states[k][1])}</span></li>`).join("");
    const legendOpen = !!detail.querySelector(".wfv-legend[open]");
    setHtml(detail, `<div class="wfv-title"><h3>${esc(st.label || "Unnamed workflow")}</h3>${badge(st.state, st.state_label, st.state_help)}<span class="small muted">${esc(st.state_help || "")}</span></div>
      ${wfActions(st)}
      <dl class="wfv-meta">${meta.join("")}</dl>
      ${controlNote(st)}
      <div class="wfv-counts">${counts(st)}</div>
      ${heldNote(st)}
      <div class="wfv-grid-wrap"><table class="wfv-grid"><caption class="sr-only">Steps of every material</caption>
        <thead><tr><th scope="col">Material</th>${head}</tr></thead><tbody>${body}</tbody></table></div>
      <div class="wfv-node-detail">${nodeDetail(st)}</div>
      <details class="wfv-legend"${legendOpen ? " open" : ""}><summary>What the states mean</summary><ul>${legend}</ul></details>`);
  }

  function renderStrip() {
    const parts = [];
    if (list) parts.push(`${(list.workflows || []).length} workflow${(list.workflows || []).length === 1 ? "" : "s"}`);
    if (list && !list.slurm) parts.push("SLURM not reachable here");
    if (cur && cur._at) parts.push(`updated ${new Date(cur._at).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false })}`);
    strip.textContent = parts.join(" · ");
  }

  async function load() {
    if (!cur) return;
    if (loading) { again = true; return loading; }
    const d = cur;
    loading = (async () => {
      try {
        list = await api("workflow/list");
        pollS = list.poll_seconds || pollS;
        const items = list.workflows || [];
        if (!d.wfKey || !items.some((w) => keyOf(w) === d.wfKey)) d.wfKey = items.length ? keyOf(items[0]) : null;
        const w = items.find((x) => keyOf(x) === d.wfKey);
        if (w) {
          try {
            status = await api(`workflow/status?id=${encodeURIComponent(w.workflow_id)}&folder=${encodeURIComponent(w.folder_rel || "")}&eta=1`);
          } catch (e) { status = { error: e.message }; }
        } else status = null;
        d._at = Date.now();
      } catch (e) {
        list = list || { workflows: [] };
        toast(`Workflows: ${e.message}`, "error");
      }
      if (cur === d) { renderList(); renderDetail(); renderStrip(); }
    })();
    try { await loading; } finally {
      loading = null;
      if (again) { again = false; load(); }
    }
  }

  function schedule() {
    clearTimeout(timer);
    if (!shown) return;
    timer = setTimeout(() => { if (shown && !document.hidden) load(); schedule(); }, Math.max(15, pollS) * 1000);
  }

  function nodeOf() {
    const sel = cur && cur.wfNode;
    const m = sel && status && (status.materials || []).find((x) => x.material_id === sel.mid);
    return m ? m.nodes.find((x) => x.step === sel.step) || null : null;
  }

  host.addEventListener("click", (e) => {
    const t = e.target.closest("button");
    if (!t || !host.contains(t)) return;
    if (t.classList.contains("wfv-refresh")) { load(); return; }
    if (t.classList.contains("wfv-new")) { if (H.newWorkflow) H.newWorkflow({}); return; }
    if (t.dataset.wact) {
      const a = t.dataset.wact;
      const n = nodeOf();
      if (a === "pause") act(a, pause);
      else if (a === "resume") act(a, resume);
      else if (a === "cancel") act(a, cancel);
      else if (a === "clone") act(a, cloneIt);
      else if (a === "recover" && n) act(a, () => recover(n));
      else if (a === "process" && n) act(a, processNow);
      return;
    }
    if (t.dataset.key) {
      cur.wfKey = t.dataset.key; cur.wfNode = null; status = null;
      renderList(); renderDetail(); load();
      return;
    }
    if (t.dataset.mid !== undefined && t.dataset.step) {
      const same = cur.wfNode && cur.wfNode.mid === t.dataset.mid && cur.wfNode.step === t.dataset.step;
      cur.wfNode = same ? null : { mid: t.dataset.mid, step: t.dataset.step };
      renderDetail();
      return;
    }
    if (t.dataset.open) { H.openPath(t.dataset.open, t.dataset.kind); return; }
    if (t.dataset.job) { H.openJob(t.dataset.job); return; }
    if (t.dataset.copy) { H.copy(t.dataset.copy); toast("Copied", "success"); }
  });
  document.addEventListener("visibilitychange", () => { if (shown && !document.hidden) load(); });
  // another document came to the front: no refreshes until this one is shown again (open())
  document.addEventListener("mace-doc-switch", (e) => {
    if (cur && e.detail && e.detail.to !== cur) { shown = false; clearTimeout(timer); }
  });

  return {
    open(d) {
      cur = d; shown = true;
      renderList(); renderDetail(); renderStrip();
      load(); schedule();
    },
    hide() { shown = false; clearTimeout(timer); },
    close() { shown = false; clearTimeout(timer); cur = null; },
    /** workflow.changed / jobs.finished from the event stream */
    changed() { if (shown && cur) load(); },
    state: () => ({ list, status, key: cur && cur.wfKey, node: cur && cur.wfNode, busy }),
  };
}
