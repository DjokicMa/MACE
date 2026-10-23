/* The Jobs tab of the bottom panel (PLAN_submit T10): your own SLURM jobs
 * (This folder / All my jobs), job arrays grouped, Details (the Job document;
 * a double-click too), Open Output, Show Log and Cancel. It asks the server every cluster.pollSeconds while it is shown
 * (the server keeps one squeue per 10 s at most); finished jobs arrive as
 * jobs.finished events whether it is shown or not. Cancel runs scancel on
 * single jobs and array tasks only, after a confirmation naming the command.
 * Recover (a failed tracked job under Finished) and Recover Failed… (every
 * failed calculation of the folder) show `mace recover --dry-run` and, once
 * confirmed, run `mace recover --yes`: the CLI itself (recover_flow.js, as the
 * Workflows view's Recover of one step). */
import { recoverFlow } from "./recover_flow.js";

export function createJobsPanel(host, h) {
  const { esc, icon, api, toast } = h;
  let scope = "folder", data = null, timer = 0, shown = false, selected = new Set(), recentOpen = false, recent = null;
  let loading = false, recovering = false, shownHtml = null;

  function ago(t) {
    if (!t) return "never";
    const s = Math.max(0, Math.round(Date.now() / 1000 - t));
    return s < 60 ? `${s} s ago` : `${Math.round(s / 60)} min ago`;
  }
  async function load(refresh = false) {
    if (loading) return;
    loading = true;
    try {
      data = await api(`jobs?scope=${scope}&refresh=${refresh ? 1 : 0}&visible=${shown ? 1 : 0}`);
      h.countsChanged(data.slurm ? data.counts : null);
    } catch (e) {
      data = { jobs: [], error: e.message, slurm: true, polled_at: 0 };
    } finally { loading = false; }
    render();
  }
  function schedule() {
    clearTimeout(timer);
    if (!shown) return;
    const s = (data && data.interval) || 60;
    timer = setTimeout(async () => { await load(false); schedule(); }, s * 1000);
  }
  function groups(jobs) {
    const out = [], arrays = new Map();
    for (const j of jobs) {
      if (j.array_id) {
        if (!arrays.has(j.array_id)) { const g = { array: j.array_id, jobs: [] }; arrays.set(j.array_id, g); out.push(g); }
        arrays.get(j.array_id).jobs.push(j);
      } else out.push({ job: j });
    }
    return out;
  }
  function row(j) {
    const elapsed = `${esc(j.elapsed || "")}${j.limit ? ` / ${esc(j.limit)}` : ""}`;
    const tracked = j.tracked ? `<span title="${esc(j.tracked.db || "")}">${esc(j.tracked.status || "tracked")}</span>` : "";
    const own = j.own_gui_job;
    return `<tr data-id="${esc(j.id)}" tabindex="0">
      <td>${own ? "" : `<input type="checkbox" data-sel="${esc(j.id)}" aria-label="Select ${esc(j.id)}"${selected.has(j.id) ? " checked" : ""}>`}</td>
      <td class="mono">${esc(j.id)}${own ? ' <span class="jobs-badge" title="The job MACE GUI runs in">MACE GUI</span>' : ""}</td>
      <td class="name" title="${esc(j.name)}">${esc(j.name)}</td>
      <td class="mono">${j.deck_rel ? esc(j.deck_rel.split("/").pop()) : '<span class="muted">—</span>'}</td>
      <td class="st-${esc(j.state)}">${esc(j.state)}${j.held ? ` <span class="jobs-badge held" style="color:var(--warning);background:transparent;border:1px solid var(--warning)" title="${j.held === "user" ? "Held by you (JobHeldUser): release it to let it start" : "Held by an administrator (JobHeldAdmin): only they can release it"}">Held</span>` : ""}</td><td class="mono">${elapsed}</td><td class="mono">${esc(j.cpus || "")}</td>
      <td>${esc(j.state === "PENDING" ? j.reason : j.node)}</td><td>${tracked}</td>
      <td><span class="jobs-acts">
        <button type="button" class="tb-btn-text sm" data-act="details" data-id="${esc(j.id)}" aria-label="Details of job ${esc(j.id)}">Details</button>
        <button type="button" class="tb-btn-text sm" data-act="output" data-id="${esc(j.id)}"${j.out_rel ? "" : " disabled"}>Open Output</button>
        <button type="button" class="tb-btn-text sm" data-act="log" data-id="${esc(j.id)}"${j.log_rel ? "" : " disabled"}>Show Log</button>
        ${j.held === "user" && !own ? `<button type="button" class="tb-btn-text sm" data-act="release" data-id="${esc(j.id)}">Release</button>` : ""}
        ${own ? "" : `<button type="button" class="tb-btn-text sm" data-act="cancel" data-id="${esc(j.id)}">Cancel</button>`}</span></td></tr>`;
  }
  function render() {
    if (!host) return;
    if (data && data.slurm === false) {
      shownHtml = null;
      host.innerHTML = `<p class="empty-note" id="jobs-note">No SLURM on this machine. Jobs you run here with Run on This Node appear in Tasks.</p>`;
      return;
    }
    const jobs = (data && data.jobs) || [];
    const ids = new Set(jobs.map((j) => j.id));
    [...selected].forEach((x) => { if (!ids.has(x)) selected.delete(x); });
    let html = `<div class="jobs-bar" role="toolbar" aria-label="Jobs">
      <div class="seg" role="group" aria-label="Scope">
        <button type="button" class="seg-btn${scope === "folder" ? " active" : ""}" data-scope="folder" aria-pressed="${scope === "folder"}">This folder</button>
        <button type="button" class="seg-btn${scope === "all" ? " active" : ""}" data-scope="all" aria-pressed="${scope === "all"}">All my jobs</button></div>
      <button type="button" class="tb-btn-text" data-act="refresh">${icon("refresh-cw")}<span>Refresh</span></button>
      <span class="muted" data-part="updated"></span><span class="grow"></span>
      <button type="button" class="tb-btn-text" data-act="cancel-selected"${selected.size ? "" : " disabled"}>Cancel Selected${selected.size ? ` (${selected.size})` : ""}</button></div>`;
    if (data && data.error) html += `<p class="sub-line bad">${esc(data.error)}</p>`;
    if (!jobs.length) {
      html += `<p class="empty-note" id="jobs-none">${scope === "folder" ? "No jobs of yours run in this folder." : "You have no jobs in the queue."}</p>`;
    } else {
      html += `<table class="jobs-table"><thead><tr><th></th><th>Job</th><th>Name</th><th>Deck</th><th>State</th><th>Elapsed / Limit</th>
        <th>CPUs</th><th>Reason or Node</th><th>Tracked</th><th></th></tr></thead><tbody>`;
      for (const g of groups(jobs)) {
        if (g.job) { html += row(g.job); continue; }
        const run = g.jobs.filter((j) => j.state === "RUNNING").length, pend = g.jobs.filter((j) => j.state === "PENDING").length;
        html += `<tr class="group"><td colspan="10">Array ${esc(g.array)}: ${run} running, ${pend} pending</td></tr>`;
        html += g.jobs.map(row).join("");
      }
      html += "</tbody></table>";
    }
    const q = (data && data.queue) || {};
    const heldList = q.held || [];
    if (q.budget) {
      const hc = q.held_counts || {}, parts = Object.keys(hc).sort().map((k) => `${hc[k]} ${k === "budget" ? "queue budget full" : k === "workflow_cap" ? "workflow max_jobs" : "SLURM refused"}`);
      html += `<p class="sub-line muted" data-part="budget">Queue budget: ${esc(String(q.used ?? "?"))} of ${esc(String(q.budget))} in use${q.limit ? ` (limit ${esc(String(q.limit))}, reserve ${esc(String(q.reserve))})` : ""}${heldList.length ? ` · ${heldList.length} held${parts.length ? ` (${esc(parts.join(", "))})` : ""}` : ""}</p>`;
    }
    if (heldList.length) {
      html += `<details class="jobs-held" open><summary>Held (${heldList.length}): MACE submits them as queue slots free up</summary>
        <table class="jobs-table"><thead><tr><th>Calculation</th><th>Material</th><th>Reason</th><th>Held since</th><th>Folder</th></tr></thead><tbody>${heldList.map((x) =>
        `<tr><td>${esc(x.calc_type)}</td><td class="mono">${esc(x.material_id)}</td><td>${esc(x.reason_text || x.reason)}</td><td class="mono">${esc((x.held_since || "").replace("T", " "))}</td><td class="mono">${esc(x.workdir_rel || "")}</td></tr>`).join("")}</tbody></table></details>`;
    }
    html += `<details class="jobs-recent"${recentOpen ? " open" : ""}><summary>Finished (last 2 days)</summary><div data-part="recent">
      <div class="jobs-bar"><span class="grow"></span><button type="button" class="tb-btn-text" data-act="recover-all"${recovering ? " disabled" : ""}
        title="Preview, then fix and resubmit every failed tracked calculation in this folder (mace recover)">Recover Failed…</button></div>${
      recent === null ? '<p class="empty-note">Loading…</p>' : recent.length ? `<table class="jobs-table"><tbody>${recent.map((j) =>
        `<tr><td class="mono">${esc(j.id)}</td><td>${esc(j.name)}</td><td>${esc(j.state)}</td><td class="mono">${esc(j.elapsed)}</td><td class="mono">${esc(j.workdir_rel || "")}</td>
        <td>${j.recoverable ? `<button type="button" class="tb-btn-text sm" data-act="recover" data-id="${esc(j.id)}"${recovering ? " disabled" : ""}>Recover…</button>` : ""}</td></tr>`).join("")}</tbody></table>`
        : '<p class="empty-note">None.</p>'}</div></details>`;
    // the same jobs as shown: keep the rows (a click, double-click or focus on one is not lost to a poll)
    if (html !== shownHtml) { host.innerHTML = html; shownHtml = html; }
    const upd = host.querySelector('[data-part="updated"]');
    if (upd) upd.textContent = `Updated ${ago(data && data.polled_at)}`;
  }
  function jobById(id) { return ((data && data.jobs) || []).find((j) => j.id === id); }
  async function cancel(ids) {
    const jobs = ids.map(jobById).filter(Boolean);
    const names = jobs.map((j) => `${j.id} (${j.name})`).join(", ");
    const ok = await h.confirm({ title: ids.length === 1 ? "Cancel Job?" : `Cancel ${ids.length} Jobs?`,
      body: `Cancel ${names}? This runs: scancel ${ids.join(" ")}`, yes: ids.length === 1 ? "Cancel Job" : "Cancel Jobs",
      no: ids.length === 1 ? "Keep Job" : "Keep Jobs", defaultNo: true });
    if (!ok) return;
    try {
      const r = await api("jobs/cancel", { body: { ids } });
      if (r.canceled.length) toast(`Canceled ${r.canceled.join(", ")}`, "success");
      for (const x of r.refused) toast(`${x.id}: ${x.reason}`, "warning");
      r.canceled.forEach((x) => selected.delete(x));
    } catch (e) { toast(e.message, "error"); }
    load(true);
  }
  async function release(id) {
    const j = jobById(id);
    const ok = await h.confirm({ title: "Release Job?", body: `Release ${id}${j ? ` (${j.name})` : ""}? It leaves the hold and can start when resources are free. This runs: scontrol release ${id}`,
      yes: "Release Job", no: "Keep Held", defaultNo: false });
    if (!ok) return;
    try {
      const r = await api("jobs/release", { body: { ids: [id] } });
      if (r.released.length) toast(`Released ${r.released.join(", ")}`, "success");
      for (const x of r.refused) toast(`${x.id}: ${x.reason}`, "warning");
    } catch (e) { toast(e.message, "error"); }
    load(true);
  }
  async function recover(body) {
    if (recovering) return;
    recovering = true;
    render();
    try {
      await recoverFlow(h, { title: "Recover Failed Jobs?", preview: () => h.apiTask("jobs/recover/preview", body),
        run: (b) => h.apiTask("jobs/recover/run", b) });
    } catch (err) { toast(err.message, "error"); } finally { recovering = false; recent = null; if (recentOpen) loadRecent(); load(true); }
  }
  async function loadRecent() {
    try { recent = (await api("jobs/recent")).jobs; } catch (err) { recent = []; }
    render();
  }
  host.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    if (b.dataset.scope) { scope = b.dataset.scope; load(false); return; }
    const id = b.dataset.id;
    if (b.dataset.act === "refresh") load(true);
    else if (b.dataset.act === "cancel-selected") cancel([...selected]);
    else if (b.dataset.act === "cancel") cancel([id]);
    else if (b.dataset.act === "release") release(id);
    else if (b.dataset.act === "recover") recover({ ids: [id] });
    else if (b.dataset.act === "recover-all") recover({ all: true });
    else if (b.dataset.act === "output") { const j = jobById(id); if (j && j.out_rel) h.openOutput(j.out_rel, j.state === "RUNNING"); }
    else if (b.dataset.act === "log") { const j = jobById(id); if (j && j.log_rel) h.openLog(j.log_rel); }
    else if (b.dataset.act === "details") h.openJob(id);
  });
  // a row's double-click opens its Job document (state, progress and ETA), as Details does
  host.addEventListener("dblclick", (e) => {
    const tr = e.target.closest("tr[data-id]");
    if (!tr || e.target.closest("button, input")) return;
    h.openJob(tr.dataset.id);
  });
  host.addEventListener("change", (e) => {
    const c = e.target.closest("[data-sel]");
    if (!c) return;
    if (c.checked) selected.add(c.dataset.sel); else selected.delete(c.dataset.sel);
    const btn = host.querySelector('[data-act="cancel-selected"]');
    if (btn) { btn.disabled = !selected.size; btn.textContent = `Cancel Selected${selected.size ? ` (${selected.size})` : ""}`; }
  });
  host.addEventListener("toggle", async (e) => {
    if (!e.target.matches || !e.target.matches("details.jobs-recent")) return;
    recentOpen = e.target.open;
    if (recentOpen && recent === null) await loadRecent();
  }, true);
  host.addEventListener("keydown", (e) => {
    const tr = e.target.closest && e.target.closest("tr[data-id]");
    if (!tr || e.target !== tr) return;
    if (e.key === "Enter") { const j = jobById(tr.dataset.id); if (j && j.out_rel) h.openOutput(j.out_rel, j.state === "RUNNING"); }
    else if (e.key === "Delete") { const j = jobById(tr.dataset.id); if (j && !j.own_gui_job) cancel([j.id]); }
  });
  return {
    show() { shown = true; load(false).then(schedule); },
    hide() { shown = false; clearTimeout(timer); },
    refresh(force = true) { return load(force).then(() => { if (shown) schedule(); }); },
    state: () => ({ scope, data, selected: [...selected] }),
  };
}
