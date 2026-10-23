/* The Cluster document and the Job document (area F): read-only views of the
 * SLURM cluster and of one of your jobs, served from the server's shared,
 * rate-limited cache (/api/cluster/*). Nothing here changes a job or a file.
 * Tabs: Overview, Nodes, Will It Start?, Storage, Limits, and Software where a
 * module system exists. */
const TABS = [["overview", "Overview"], ["nodes", "Nodes"], ["fit", "Will It Start?"], ["storage", "Storage"], ["limits", "Limits"],
  ["software", "Software"]];
const SNAP = "Free now is a snapshot: priority, limits and backfill decide the real start.";

function fmtDur(s) {
  if (s === null || s === undefined || !isFinite(s)) return "unknown";
  s = Math.round(s);
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  if (d) return `${d} d ${h} h`;
  if (h) return `${h} h ${m} min`;
  if (m) return `${m} min`;
  return `${s} s`;
}
const num = (n) => (n === null || n === undefined ? "–" : Number(n).toLocaleString("en-US"));
function bytes(b) {
  if (b === null || b === undefined) return "–";
  const u = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
  let i = 0;
  while (Math.abs(b) >= 1024 && i < u.length - 1) { b /= 1024; i++; }
  return i ? `${b.toFixed(1)} ${u[i]}` : `${Math.round(b)} B`;
}
const gib = (g) => (g >= 1024 ? `${(g / 1024).toFixed(1)} TiB` : `${(g || 0).toFixed(1)} GiB`);
function localTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  const off = -d.getTimezoneOffset(), sign = off >= 0 ? "+" : "−", hh = String(Math.floor(Math.abs(off) / 60)).padStart(2, "0");
  const today = new Date().toDateString() === d.toDateString();
  return `${today ? "today" : d.toLocaleDateString("en-US", { month: "short", day: "numeric" })} ${d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false })} (UTC${sign}${hh}:${String(Math.abs(off) % 60).padStart(2, "0")})`;
}

export function createClusterView(host, H) {
  const { api, esc, icon, toast } = H;
  let tab = "overview", data = {}, timer = 0, ticker = 0, caps = null, scanTask = null, scanTimer = 0;
  const lastFetch = {};
  host.innerHTML = `<div class="clv">
    <div class="clv-head"><h2 class="clv-title">Cluster</h2><span class="clv-strip muted small" aria-live="polite"></span>
      <button type="button" class="tb-btn-text clv-refresh">${icon("refresh-cw")}<span>Refresh</span></button></div>
    <div class="clv-alloc small" hidden></div>
    <div class="clv-tabs" role="tablist" aria-label="Cluster">${TABS.map(([k, t]) =>
      `<button role="tab" data-tab="${k}" id="clv-tab-${k}" aria-controls="clv-pane" aria-selected="false" tabindex="-1"${k === "software" ? " hidden" : ""}>${t}</button>`).join("")}</div>
    <div class="clv-errors small" role="status"></div>
    <div class="clv-pane" id="clv-pane" role="tabpanel" tabindex="0"></div></div>`;
  const $ = (s) => host.querySelector(s);
  const pane = $(".clv-pane");

  function setTab(k, focus = false) {
    tab = k;
    host.querySelectorAll("[role=tab]").forEach((b) => {
      const on = b.dataset.tab === k;
      b.setAttribute("aria-selected", String(on)); b.tabIndex = on ? 0 : -1;
      if (on && focus) b.focus();
    });
    pane.setAttribute("aria-labelledby", `clv-tab-${k}`);
    load(false);
  }
  $(".clv-tabs").addEventListener("click", (e) => { const b = e.target.closest("[role=tab]"); if (b) setTab(b.dataset.tab); });
  $(".clv-tabs").addEventListener("keydown", (e) => {
    const shown = TABS.filter(([k]) => !host.querySelector(`[role=tab][data-tab="${k}"]`).hidden);
    const i = shown.findIndex(([k]) => k === tab);
    const d = { ArrowRight: 1, ArrowLeft: -1, Home: -i, End: shown.length - 1 - i }[e.key];
    if (d === undefined) return;
    e.preventDefault(); setTab(shown[(i + d + shown.length) % shown.length][0], true);
  });
  $(".clv-refresh").addEventListener("click", () => load(true));

  function strip(meta) {
    const s = $(".clv-strip"), btn = $(".clv-refresh");
    if (!meta) { s.textContent = ""; return; }
    const age = meta.age === null || meta.age === undefined ? null : Math.round(meta.age + (Date.now() - meta._at) / 1000);
    const wait = Math.max(0, Math.ceil((meta.next_allowed_in || 0) - (Date.now() - meta._at) / 1000));
    s.textContent = (age === null ? "" : `Updated ${age < 120 ? age + " s" : Math.round(age / 60) + " min"} ago`) +
      (meta.stale ? " · stale" : "") + (wait > 0 ? ` · Refresh (available in ${wait} s)` : "");
    btn.setAttribute("aria-disabled", String(wait > 0));
    btn.dataset.wait = String(wait);
  }
  function errors(meta) {
    const e = [...(meta.errors || []), ...(meta.notes || [])];
    $(".clv-errors").innerHTML = e.map((x) => `<div class="warn">${icon("triangle-alert")}${esc(x)}</div>`).join("");
  }
  const routeOf = { overview: "overview", nodes: "nodes", fit: "nodes", storage: "quotas", limits: "limits", software: "software" };
  async function load(refresh) {
    const btn = $(".clv-refresh");
    if (refresh && Number(btn.dataset.wait || 0) > 0) { toast(`The next refresh is allowed in ${btn.dataset.wait} s`, "info"); return; }
    // the answer belongs to the tab it was asked for: a slow answer that lands
    // after a tab switch is kept for its own tab and never drawn into another
    const t = tab, route = routeOf[t];
    const extra = t === "nodes" ? `&group=${encodeURIComponent(nodeGroup)}&all=${showAll ? 1 : 0}`
      : t === "limits" && limAsked.n ? `&n=${encodeURIComponent(limAsked.n)}&cpus=${encodeURIComponent(limAsked.c || 1)}` : "";
    try {
      const r = await api(`cluster/${route}?refresh=${refresh ? 1 : 0}${extra}`);
      r._at = Date.now();
      if (t === "limits") { r.headroom_n = limAsked.n; r.headroom_c = limAsked.c; }
      data[t] = r;
      if (t === "overview") { try { data.jobs = await api("cluster/jobs"); } catch (e) { data.jobs = null; } }
      if (t !== tab) return;
      render();
      if (t === "software") { strip(null); $(".clv-errors").innerHTML = ""; } else { strip(r); errors(r); }
    } catch (e) {
      if (t !== tab) return;
      pane.innerHTML = `<p class="clv-empty">${esc(e.message)}</p>`;
      strip(null);
    }
  }
  let nodeGroup = "feature", showAll = false;
  /* What the user typed survives the timed refresh (which re-renders the tab):
   * the Limits question (asked and still being typed), the scan form. */
  const limAsked = { n: "", c: "" }, limDraft = { n: "", c: "" };
  let scanForm = null;          // {root, max_seconds, sizes} while the scan form is open

  function render() {
    const r = data[tab];
    if (!r) { pane.innerHTML = `<p class="muted">Loading…</p>`; return; }
    if (tab === "overview") renderOverview(r);
    else if (tab === "nodes") renderNodes(r);
    else if (tab === "fit") renderFit(r);
    else if (tab === "storage") renderStorage(r);
    else if (tab === "limits") renderLimits(r);
    else if (tab === "software") renderSoftware(r);
    const t = $(".clv-title");
    if (r.cluster) t.textContent = `Cluster: ${r.cluster}`;
  }
  const card = (k, v, cls = "") => `<div class="clv-card ${cls}"><div class="k">${esc(k)}</div><div class="v">${v}</div></div>`;
  function renderOverview(r) {
    const u = r.usage || {}, lim = r.limits || {};
    const cards = [card("Your jobs", `${num(u.running)} running · ${num(u.pending)} pending${lim.max_jobs ? ` <span class="muted">(of ${num(lim.max_jobs)} / ${num(lim.max_submit)})</span>` : ""}`)];
    if (lim.max_cpus) cards.push(card("Cores in use", `${num(u.running_cpus)} of ${num(lim.max_cpus)} (QOS)`));
    (r.fairshare || []).slice(0, 3).forEach((f) => cards.push(card("Fairshare", `${esc(f.fairshare ?? "–")} <span class="muted">(${esc(f.account)})</span>`)));
    if (r.storage) cards.push(card("Storage", `${esc(r.storage.label)} ${r.storage.percent} %`, r.storage.level));
    else cards.push(card("Storage", "No quota warnings"));
    const soon = r.reservations_soon || [];
    cards.push(card("Maintenance", soon.length ? soon.map((x) => `${x.maintenance ? "Maintenance" : "Reservation"} starts in ${fmtDur(x.starts_in_s)} (${esc(x.name)})`).join("<br>") : "none scheduled", soon.length ? "warning" : ""));
    const t = r.totals || {};
    const jobs = (data.jobs && data.jobs.jobs) || [];
    pane.innerHTML = `<div class="clv-cards">${cards.join("")}</div>
      <p class="small">Usable partitions: ${esc((r.usable_partitions || []).join(", ") || "(unknown)")} · nodes ${num(t.nodes)} (${num(t.idle)} idle, ${num(t.mixed)} mixed, ${num(t.allocated)} allocated, ${num(t.down_drain)} down or drained) · free cores ${num(t.cpus_free)} of ${num(t.cpus_total)}</p>
      ${(r.quota_warnings || []).map((w) => `<div class="clv-warn">${icon("triangle-alert")}${esc(w)}</div>`).join("")}
      <h3>Your jobs</h3>${jobs.length ? `<table class="data clv-table"><thead><tr><th>Job</th><th>Name</th><th>State</th><th>Elapsed</th><th>Limit</th><th>Cores</th><th></th></tr></thead><tbody>${jobs.map((j) =>
        `<tr><td class="mono">${esc(j.id)}</td><td>${esc(j.name)}</td><td>${esc(j.state)}${j.state === "PENDING" && /^JobHeld/.test(j.reason || "") ? ' <span class="chip" title="Held: release it from the Jobs panel (held by you) or ask an administrator">Held</span>' : ""}${j.state === "PENDING" && j.reason && j.reason !== "None" ? ` (${esc(j.reason)})` : ""}</td><td>${esc(j.elapsed)}</td><td>${esc(j.time_limit)}</td><td class="num">${esc(j.cpus)}</td>
        <td><button type="button" class="tb-btn-text" data-job="${esc(j.id)}" aria-label="Details of job ${esc(j.id)}">Details</button></td></tr>`).join("")}</tbody></table>` : `<p class="muted">No jobs of yours in the queue.</p>`}`;
  }
  function renderNodes(r) {
    const label = r.group_by === "feature" ? "CPU type" : "Partition";
    const max = Math.max(1, ...r.rows.map((x) => x.nodes));
    pane.innerHTML = `<div class="clv-row"><label>Group by <select class="clv-group"><option value="feature"${r.group_by === "feature" ? " selected" : ""}>CPU type</option><option value="partition"${r.group_by === "partition" ? " selected" : ""}>Partition</option></select></label>
      <label><input type="checkbox" class="clv-all"${r.all_partitions ? " checked" : ""}> Show all partitions</label></div>
      <table class="data clv-table"><thead><tr><th>${label}</th><th class="num">Nodes</th><th class="num">Idle</th><th class="num">Mixed</th><th class="num">Allocated</th><th class="num">Down/Drain</th><th class="num">Free cores</th><th class="num">Free memory</th><th>Free GPUs</th><th aria-label="Node states"></th></tr></thead><tbody>
      ${r.rows.map((x) => `<tr><td>${esc(x.group)}</td><td class="num">${num(x.nodes)}</td><td class="num">${num(x.idle)}</td><td class="num">${num(x.mixed)}</td><td class="num">${num(x.allocated)}</td><td class="num">${num(x.down_drain)}</td>
        <td class="num">${num(x.cpus_free)}/${num(x.cpus_total)}</td><td class="num">${gib(x.mem_free_gb)}</td><td>${Object.keys(x.gpus_total || {}).length ? Object.entries(x.gpus_total).map(([k, n]) => `${esc(k)} ${(x.gpus_free || {})[k] || 0}/${n}`).join(", ") : "–"}</td>
        <td class="clv-stackcell"><span class="clv-stack" style="width:${Math.round(100 * x.nodes / max)}%" title="${x.idle} idle, ${x.mixed} mixed, ${x.allocated} allocated, ${x.down_drain} down or drained">${["idle", "mixed", "allocated", "down_drain"].map((k) => `<i class="s-${k}" style="flex:${x[k]}"></i>`).join("")}</span></td></tr>`).join("")}</tbody></table>
      ${r.all_partitions ? `<p class="small">${r.partitions.filter((p) => !p.usable).map((p) => `${esc(p.name)}: ${esc(p.why_not)}`).join(" · ")}</p>` : ""}
      <p class="small muted">${esc(SNAP)}</p>`;
    pane.querySelector(".clv-group").addEventListener("change", (e) => { nodeGroup = e.target.value; load(false); });
    pane.querySelector(".clv-all").addEventListener("change", (e) => { showAll = e.target.checked; load(false); });
  }
  let fitForm = { cpus: 32, nodes: 1, memKind: "mem_per_cpu", mem: "4G", gpus: "", constraint: "", partition: "", time: "1-00:00:00", script: "" };
  let fitResult = null;
  function renderFit(r) {
    const parts = (r.partitions || []).filter((p) => p.usable);
    const shs = (H.files() || []).filter((f) => /\.sh$/i.test(f.path)).map((f) => f.path);
    const f = fitForm;
    pane.innerHTML = `<form class="clv-fit">
      <label>Cores <input type="number" name="cpus" min="1" value="${f.cpus}"></label>
      <label>Nodes <input type="number" name="nodes" min="1" value="${f.nodes}"></label>
      <label>Memory <select name="memKind"><option value="mem_per_cpu"${f.memKind === "mem_per_cpu" ? " selected" : ""}>per core</option><option value="mem"${f.memKind === "mem" ? " selected" : ""}>per node</option></select>
        <input name="mem" value="${esc(f.mem)}" size="6" aria-label="Memory amount"></label>
      <label>GPUs <input name="gpus" value="${esc(f.gpus)}" size="8" placeholder="a100:1"></label>
      <label>Constraint <input name="constraint" value="${esc(f.constraint)}" size="12"></label>
      <label>Partition <select name="partition"><option value="">Any usable</option>${parts.map((p) => `<option${p.name === f.partition ? " selected" : ""}>${esc(p.name)}</option>`).join("")}</select></label>
      <label>Time <input name="time" value="${esc(f.time)}" size="11"></label>
      <label>Job script <select name="script"><option value="">(the request above)</option>${shs.map((p) => `<option${p === f.script ? " selected" : ""}>${esc(p)}</option>`).join("")}</select></label>
      <span class="clv-actions"><button type="button" class="tb-btn-text primary" data-act="find">Find Free Nodes</button><button type="button" class="tb-btn-text" data-act="predict">Predict Start</button></span></form>
      <p class="small muted">${esc(SNAP)} Predict Start asks SLURM itself (sbatch --test-only) without submitting anything.</p>
      <div class="clv-fit-out" aria-live="polite"></div>`;
    const form = pane.querySelector("form");
    const read = () => { const fd = new FormData(form); Object.keys(fitForm).forEach((k) => { if (fd.has(k)) fitForm[k] = fd.get(k); }); };
    form.addEventListener("input", read);
    form.addEventListener("change", read);
    form.addEventListener("click", async (e) => {
      const b = e.target.closest("[data-act]"); if (!b) return;
      read();
      const body = { cpus: Number(fitForm.cpus), nodes: Number(fitForm.nodes), [fitForm.memKind]: fitForm.mem || undefined,
        gpus: fitForm.gpus || undefined, constraint: fitForm.constraint || undefined, partition: fitForm.partition || undefined, time: fitForm.time || undefined };
      if (b.dataset.act === "predict" && fitForm.script) { body.script_rel = fitForm.script; }
      b.setAttribute("aria-busy", "true");
      try { fitResult = await api(b.dataset.act === "find" ? "cluster/fit" : "cluster/predict", { body }); renderFitOut(); }
      catch (err) { pane.querySelector(".clv-fit-out").innerHTML = `<p class="warn">${esc(err.message)}</p>`; }
      finally { b.removeAttribute("aria-busy"); }
    });
    renderFitOut();
  }
  function renderFitOut() {
    const out = pane.querySelector(".clv-fit-out");
    if (!out || !fitResult) return;
    const p = fitResult.predict;
    const rows = (fitResult.partitions || []).map((x) => `<tr><td>${esc(x.name)}</td><td>${esc(x.max_time || "")}</td><td>Fits now on ${x.fits_now} node${x.fits_now === 1 ? "" : "s"}</td><td class="muted">${esc(x.note || "")}</td></tr>`).join("");
    const res = (fitResult.reservations || []).map((x) => `<div class="clv-warn">${icon("triangle-alert")}${x.maintenance ? "Maintenance" : "Reservation"} ${esc(x.name)} starts in ${fmtDur(x.starts_in_s)} on nodes this job could use.</div>`).join("");
    let pred = "";
    if (p) {
      if (p.throttled) pred = `<p class="warn">${esc(p.reason)}</p>`;
      else if (p.accepted) pred = `<p class="clv-pred">SLURM predicts a start at ${esc(localTime(p.start))} on ${esc(p.nodes || "")} in partition ${esc(p.partition || "")}.</p>`;
      else if (p.accepted === false) pred = `<p class="warn">SLURM would refuse this request: ${esc(p.reason || "")}</p>`;
      else pred = `<p class="warn">SLURM did not answer: ${esc(p.reason || "")}</p>`;
    }
    out.innerHTML = pred + (rows ? `<table class="data clv-table"><thead><tr><th>Partition</th><th>MaxTime</th><th>Free now</th><th></th></tr></thead><tbody>${rows}</tbody></table>` : "") + res +
      (fitResult.constraint_note ? `<p class="small">${esc(fitResult.constraint_note)}</p>` : "");
  }
  function bar(pct, level) {
    const w = pct === null || pct === undefined ? 0 : Math.min(100, pct);
    return `<span class="clv-bar ${level || ""}" role="meter" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct === null || pct === undefined ? 0 : pct}"><i style="width:${w}%"></i></span>`;
  }
  const lvl = (v, r) => (v === null || v === undefined ? "" : v >= r.critical_percent ? "critical" : v >= r.warn_percent ? "warning" : "");
  const SPACE_TITLE = { home: "Home", scratch: "Scratch", research: "Research spaces", "": "Other" };
  const pctText = (v) => (v === null || v === undefined ? "–" : v + " %");
  function spaceCard(x, r) {
    const info = x.level === "info";
    const sl = info ? "" : lvl(x.space_pct, r), fl = info ? "" : lvl(x.files_pct, r);
    const open = x.path && x.space === "research" ? `<button type="button" class="tb-btn-text" data-open="${esc(x.path)}" title="Open this folder">${icon("folder-open")}<span>Open</span></button>` : "";
    return `<div class="clv-space ${x.level}"><div class="clv-space-head"><b>${esc(x.label)}</b><span class="chip">${esc(x.source)}</span>${x.path ? `<span class="mono small muted">${esc(H.abbrev ? H.abbrev(x.path) : x.path)}</span>` : ""}${open}</div>
        <div class="clv-space-row"><span class="lbl">Quota</span>${bar(x.space_pct, sl)}<span class="small">${bytes(x.space_used_b)} of ${bytes(x.space_limit_b)} · ${pctText(x.space_pct)}</span></div>
        <div class="clv-space-row"><span class="lbl">Files</span>${bar(x.files_pct, fl)}<span class="small">${num(x.files_used)} of ${num(x.files_limit)} · ${pctText(x.files_pct)}</span></div></div>`;
  }
  function renderStorage(r) {
    const groups = {};
    (r.rows || []).forEach((x) => { (groups[x.space || ""] = groups[x.space || ""] || []).push(x); });
    const rows = ["home", "scratch", "research", ""].filter((k) => groups[k]).map((k) =>
      `<h3 class="clv-h">${SPACE_TITLE[k]}</h3>${groups[k].map((x) => spaceCard(x, r)).join("")}`).join("");
    const quiet = (r.research || []).filter((x) => !x.has_quota);
    const quietHtml = quiet.length ? `<h3 class="clv-h">Research spaces without a quota row</h3>${quiet.map((x) =>
      `<div class="clv-space"><div class="clv-space-head"><b>${esc(x.name)}</b><span class="mono small muted">${esc(H.abbrev ? H.abbrev(x.path) : x.path)}</span><button type="button" class="tb-btn-text" data-open="${esc(x.path)}" title="Open this folder">${icon("folder-open")}<span>Open</span></button></div><p class="small muted">You can use this folder; the site's quota tool lists no limits for it.</p></div>`).join("")}` : "";
    pane.innerHTML = `${(r.warnings || []).map((w) => `<div class="clv-warn">${icon("triangle-alert")}${esc(w)}</div>`).join("")}${rows + quietHtml || '<p class="muted">No storage information.</p>'}
      <p class="small muted">Warning from ${r.warn_percent} %, critical from ${r.critical_percent} % ([quota] in ~/.config/mace/site.toml; <code>mace cluster quota</code> shows them).</p>
      <p><button type="button" class="tb-btn-text" data-act="scan">${icon("search")}<span>Find What Uses My Files…</span></button></p><div class="clv-scan" aria-live="polite"></div>`;
    pane.querySelectorAll("[data-open]").forEach((b) => b.addEventListener("click", () => H.openFolderAt ? H.openFolderAt(b.dataset.open) : H.run && H.run("file.openFolder")));
    pane.querySelector("[data-act=scan]").addEventListener("click", () => scanDialog(true));
    if (scanForm && !(scanTask && ["queued", "running"].includes((scanTask.record || {}).state))) scanDialog(false);
    else renderScan();
  }
  function scanDialog(focus = true) {
    const box = pane.querySelector(".clv-scan");
    const f = scanForm || (scanForm = { root: ".", max_seconds: "600", sizes: "top" });
    const opt = (v, t) => `<option value="${v}"${f.sizes === v ? " selected" : ""}>${t}</option>`;
    box.innerHTML = `<form class="clv-scanform"><label>Folder <input name="root" value="${esc(f.root)}" size="24"></label>
      <label>Time limit <input name="max_seconds" type="number" min="5" max="3600" value="${esc(f.max_seconds)}"> s</label>
      <label>Include sizes <select name="sizes">${opt("top", "Top folders only")}${opt("all", "All files (slower)")}${opt("off", "Off")}</select></label>
      <button type="button" class="tb-btn-text primary" data-act="start">Start Scan</button></form>
      <p class="small muted">About 2000 metadata operations per second at most, niced; counts need no per-file stat. Nothing is deleted.</p>`;
    const sform = box.querySelector("form");
    const keep = () => { const fd = new FormData(sform); scanForm = { root: fd.get("root"), max_seconds: fd.get("max_seconds"), sizes: fd.get("sizes") }; };
    sform.addEventListener("input", keep);
    sform.addEventListener("change", keep);
    box.querySelector("[data-act=start]").addEventListener("click", async () => {
      const fd = new FormData(sform);
      try {
        const r = await api("cluster/scan", { body: { root: fd.get("root"), max_seconds: Number(fd.get("max_seconds")), sizes: fd.get("sizes") } });
        scanForm = null;
        scanTask = { id: r.task.id, record: r.task, result: null };
        renderScan(); pollScan();
      } catch (e) { toast(e.message, "error"); }
    });
    if (focus) box.querySelector("input").focus();
  }
  async function pollScan() {
    clearTimeout(scanTimer);
    if (!scanTask || scanTask.result) return;
    try {
      const r = await api(`cluster/scan?task=${encodeURIComponent(scanTask.id)}`);
      scanTask.record = r;
      if (r.result) scanTask.result = r.result;
    } catch (e) { scanTask.record = { state: "failed", error: e.message }; }
    renderScan();
    if (scanTask && ["queued", "running"].includes(scanTask.record.state)) scanTimer = setTimeout(pollScan, 800);
  }
  function renderScan() {
    const box = pane.querySelector(".clv-scan");
    if (!box || !scanTask) return;
    const t = scanTask.record || {};
    if (["queued", "running"].includes(t.state)) {
      box.innerHTML = `<p><span class="spinner sm" aria-hidden="true"></span> Scanning… ${esc(t.message || "")} <button type="button" class="tb-btn-text" data-act="cancel">Cancel</button></p>`;
      box.querySelector("[data-act=cancel]").addEventListener("click", async () => {
        try { await api(`tasks/${encodeURIComponent(scanTask.id)}/cancel`, { body: {} }); } catch (e) { /* already ended */ }
        pollScan();
      });
      return;
    }
    if (t.state === "canceled") { box.innerHTML = `<p class="muted">The scan was canceled.</p>`; return; }
    if (t.state === "failed") { box.innerHTML = `<p class="warn">${esc(t.error || "The scan failed")}</p>`; return; }
    const res = scanTask.result;
    if (!res) return;
    box.innerHTML = `<p class="small">${num(res.files)} files in ${num(res.dirs)} folders under <span class="mono">${esc(res.root)}</span> (${res.elapsed_s} s; ${esc(res.coverage)})</p>
      <table class="data clv-table clv-top"><thead><tr><th>Folder</th><th class="num">Files</th><th class="num">Size</th></tr></thead><tbody>${res.top_by_count.map((x) =>
        `<tr><td class="mono">${esc(x.path)}</td><td class="num">${num(x.files)}</td><td class="num">${x.bytes ? bytes(x.bytes) : "–"}</td></tr>`).join("")}</tbody></table>
      ${res.suggestions.length ? `<h3>Cleanup suggestions</h3><p class="small muted">Check each one; nothing was deleted. Copy a command to run it yourself.</p>${res.suggestions.map((s) =>
        `<div class="clv-sug"><span class="chip">${esc(s.rule)}</span> ${esc(s.reason)}${s.command ? `<div><code class="mono">${esc(s.command)}</code> <button type="button" class="tb-btn-text" data-copy="${esc(s.command)}">${icon("copy")}<span>Copy Command</span></button></div>` : ""}</div>`).join("")}` : `<p class="muted">No cleanup suggestions.</p>`}`;
  }
  function renderLimits(r) {
    const n = limDraft.n || r.headroom_n || "", c = limDraft.c || r.headroom_c || "";
    pane.innerHTML = `<table class="data clv-table"><thead><tr><th>Account</th><th>Partition</th><th>QOS</th><th class="num">Max running jobs</th><th class="num">Max submitted</th><th>Max wall</th><th class="num">Max cores per user</th><th class="num">Running / submitted</th><th class="num">Cores in use</th></tr></thead><tbody>
      ${(r.associations || []).map((a) => `<tr><td>${esc(a.account)}</td><td>${esc(a.partition || "")}</td><td>${esc(a.qos)}</td><td class="num">${num(a.max_jobs)}</td><td class="num">${num(a.max_submit)}</td><td>${esc(a.max_wall || "")}</td><td class="num">${num(a.max_cpus)}</td><td class="num">${num(a.running)} / ${num(a.submitted)}</td><td class="num">${num(a.running_cpus)}</td></tr>`).join("")}</tbody></table>
      ${r.submit_slots ? `<p class="small">Submit slots of account ${esc(r.account || "?")} (SLURM's own count): ${num(r.submit_slots.used)} of ${num(r.submit_slots.limit)} in use.${r.slots_note ? " " + esc(r.slots_note) : ""}</p>` : ""}
      <h3>Fairshare</h3><table class="data clv-table"><thead><tr><th>Account</th><th class="num">Fairshare</th><th class="num">Effective usage</th></tr></thead><tbody>${(r.fairshare || []).map((f) =>
        `<tr><td>${esc(f.account)}</td><td class="num">${esc(f.fairshare ?? "–")}</td><td class="num">${esc(f.effective_usage ?? "–")}</td></tr>`).join("")}</tbody></table>
      <form class="clv-row"><span>Next submission:</span><label><input type="number" min="1" name="n" value="${esc(n)}" aria-label="Jobs"> jobs</label> ×
        <label><input type="number" min="1" name="c" value="${esc(c)}" aria-label="Cores per job"> cores</label><button type="button" class="tb-btn-text" data-act="check">Check</button></form>
      <div class="clv-head-out" aria-live="polite">${(r.headroom || []).map((w) => `<div class="clv-warn">${icon("triangle-alert")}${esc(w)}</div>`).join("")}${r.headroom && !r.headroom.length ? '<p class="small">These jobs fit within your limits.</p>' : ""}</div>`;
    const lform = pane.querySelector("form");
    lform.addEventListener("input", () => { const fd = new FormData(lform); limDraft.n = fd.get("n") || ""; limDraft.c = fd.get("c") || ""; });
    pane.querySelector("[data-act=check]").addEventListener("click", async () => {
      const fd = new FormData(lform);
      try {
        const out = await api(`cluster/limits?n=${encodeURIComponent(fd.get("n") || 0)}&cpus=${encodeURIComponent(fd.get("c") || 1)}`);
        limAsked.n = fd.get("n") || ""; limAsked.c = fd.get("c") || "";
        out._at = Date.now(); out.headroom_n = limAsked.n; out.headroom_c = limAsked.c;
        data.limits = out; renderLimits(out);
      } catch (e) { toast(e.message, "error"); }
    });
  }
  let softTimer = 0;
  function renderSoftware(r) {
    clearTimeout(softTimer);
    if (r.state === "checking") {
      pane.innerHTML = `<p><span class="spinner sm" aria-hidden="true"></span> Checking the modules on this machine… (a module command per code; up to a minute on a busy login node)</p>`;
      softTimer = setTimeout(() => { if (tab === "software" && host.offsetParent) load(false); }, 2000);
      return;
    }
    if (r.error) { pane.innerHTML = `<p class="warn">${esc(r.error)}</p>`; return; }
    const res = r.result || {}, rows = res.rows || [];
    const line = (x) => x.error ? `<tr><td>${esc(x.code)}</td><td class="muted small">${esc(x.binaries || "")}</td><td colspan="2" class="warn">${esc(x.error)}</td></tr>`
      : `<tr><td>${esc(x.code)}</td><td class="muted small">${esc(x.binaries || "")}</td><td class="mono">${esc((x.modules || []).join(" then ") || "–")}</td>
        <td>${esc(x.source || "")}${x.note ? `<div class="small muted">${esc(x.note)}</div>` : ""}${(x.rejected || []).map(([n, why]) => `<div class="small muted">not used: ${esc(n)}: ${esc(why)}</div>`).join("")}</td></tr>`;
    pane.innerHTML = rows.length ? `<table class="data clv-table clv-soft"><thead><tr><th>Code</th><th>Programs</th><th>Module</th><th>Source</th></tr></thead><tbody>${rows.map(line).join("")}</tbody></table>
      <p class="small muted">Read-only: the modules MACE's job scripts and Clean would load here (${esc(res.module_system || "")}). CRYSTAL is what <code>mace site</code> shows; change it in ~/.config/mace/site.toml.</p>`
      : `<p class="muted">No module information on this machine.</p>`;
  }
  pane.addEventListener("submit", (e) => e.preventDefault());       // no form here navigates (Enter in a field)
  pane.addEventListener("click", (e) => {
    const j = e.target.closest("[data-job]");
    if (j) { H.openJob(j.dataset.job); return; }
    const c = e.target.closest("[data-copy]");
    if (c) H.copy(c.dataset.copy);
  });

  return {
    async open(d) {
      if (!caps) { try { caps = await api("cluster/capabilities"); } catch (e) { caps = {}; } }
      const a = $(".clv-alloc");
      if (caps.in_allocation) { a.hidden = false; a.textContent = `MACE GUI runs inside job ${caps.in_allocation.job} on ${caps.in_allocation.node}.`; } else a.hidden = true;
      const soft = host.querySelector('[role=tab][data-tab="software"]');
      soft.hidden = !caps.software;
      if (soft.hidden && tab === "software") tab = "overview";
      if (caps.slurm === false) {
        $(".clv-title").textContent = "Cluster";
        if (tab !== "storage") tab = "storage";
      }
      if (d && d.tab) { tab = d.tab; d.tab = null; }
      setTab(tab);
      clearInterval(ticker);
      ticker = setInterval(() => { const r = data[tab]; if (r && host.isConnected && host.offsetParent) strip(r); }, 1000);
      clearInterval(timer);
      const every = Math.max(2, (caps.poll_seconds || 60)) * 1000;
      // the timed refresh re-renders the tab: not while a field of it has the focus
      const typing = () => { const a = document.activeElement; return !!a && pane.contains(a) && /^(INPUT|SELECT|TEXTAREA)$/.test(a.tagName); };
      timer = setInterval(() => { if (host.offsetParent && document.visibilityState === "visible" && !typing()) load(false); }, every);
    },
    close() { clearInterval(timer); clearInterval(ticker); },
    tab: () => tab, setTab, data: () => data,
  };
}

/* ---------------------------------------------------------------- the Job document */
function spark(points, key, thrKey) {
  const vals = points.map((p) => p[key]).filter((v) => v !== null && v !== undefined && v > 0);
  if (vals.length < 2) return '<span class="muted small">–</span>';
  const thr = (points[points.length - 1].thr || {})[key];
  const ys = vals.map((v) => Math.log10(v));
  const t = thr ? Math.log10(thr) : null;
  const lo = Math.min(...ys, t === null ? Infinity : t), hi = Math.max(...ys, t === null ? -Infinity : t);
  const W = 140, Hh = 36, sx = (i) => (i / (ys.length - 1)) * (W - 4) + 2, sy = (y) => Hh - 2 - ((y - lo) / Math.max(1e-9, hi - lo)) * (Hh - 4);
  const path = ys.map((y, i) => `${i ? "L" : "M"}${sx(i).toFixed(1)},${sy(y).toFixed(1)}`).join("");
  return `<svg class="clv-spark" viewBox="0 0 ${W} ${Hh}" width="${W}" height="${Hh}" role="img" aria-label="${key.replace("_", " ")} trend">` +
    (t !== null ? `<line x1="0" x2="${W}" y1="${sy(t).toFixed(1)}" y2="${sy(t).toFixed(1)}" class="thr"/>` : "") + `<path d="${path}"/></svg>`;
}

export function createJobView(host, H) {
  const { api, esc, icon, toast } = H;
  let cur = null, timer = 0;
  host.innerHTML = `<div class="jbv"><div class="jbv-body" tabindex="0"></div></div>`;
  const body = host.querySelector(".jbv-body");

  async function load(d, refresh = false) {
    try {
      const r = await api(`cluster/job?id=${encodeURIComponent(d.jobId)}&refresh=${refresh ? 1 : 0}`);
      d.job = r; d._at = Date.now();
      if (r.name) { d.title = `Job ${d.jobId} · ${r.name}`; H.retitle(); }
      if (cur === d) render(d);
    } catch (e) {
      if (cur === d) body.innerHTML = `<p class="clv-empty">${esc(e.message)}</p>`;
    }
  }
  function render(d) {
    const j = d.job;
    if (!j) { body.innerHTML = `<p class="muted">Loading…</p>`; return; }
    const pr = j.progress || null, risk = (pr && pr.risk) || {};
    const pct = j.limit_s && j.elapsed_s !== null ? Math.min(100, Math.round(100 * j.elapsed_s / j.limit_s)) : 0;
    let prog = "";
    if (pr) {
      const eta = pr.eta || {};
      prog = `<h3>Progress</h3><p class="jbv-summary">${esc(pr.summary || "")}</p>`;
      if (pr.mode === "opt" && pr.criteria && pr.criteria.length) {
        prog += `<div class="jbv-sparks">${["max_grad", "rms_grad", "max_disp", "rms_disp"].map((k) => `<div><div class="small">${k.replace("_", " ").replace("grad", "gradient").replace("disp", "displacement")}</div>${spark(pr.criteria, k)}</div>`).join("")}</div>`;
      }
      if (pr.mode === "freq" && pr.total && !(pr.levels && pr.levels.length)) prog += `<p>${pr.stage === "displacements" ? `Displacement ${pr.done || 0} of ${pr.total}${pr.percent !== null && pr.percent !== undefined ? ` (${pr.percent} %)` : ""}` : esc(pr.stage)}</p>`;
      if (pr.levels && pr.levels.length) prog += levelsHtml(pr, d);
      else prog += `<p class="small">${eta.low_s !== null && eta.low_s !== undefined ? "ETA: " : "ETA unknown: "}${esc(eta.basis || "")}</p>`;
      if (risk.note && !(risk.level && risk.level !== "ok")) prog += `<p class="small jbv-note">${esc(risk.note)}</p>`;
    } else if (j.progress_note) prog = `<h3>Progress</h3><p class="muted">${esc(j.progress_note)}</p>`;
    // 1.1.6: queue wait (SLURM's start estimate), the rest of the array, the rest of the workflow
    const totals = (j.totals || []).length
      ? `<h3>Beyond this job</h3><ul class="jbv-totals small">${j.totals.map((t) => `<li data-total="${esc(t.id)}"><span>${esc(t.text)}</span>${t.basis ? `<div class="muted">${esc(t.basis)}</div>` : ""}</li>`).join("")}</ul>`
      : "";
    const eff = (j.state === "RUNNING" && d._liveEff) || j.efficiency || {};
    const banner = risk.level && risk.level !== "ok" ? `<div class="jbv-risk ${risk.level === "tight" ? "warning" : "critical"}" role="alert">${icon("triangle-alert")}<div><b>${esc(risk.headline || (risk.level === "tight" ? "Walltime is tight" : "Likely to exceed the walltime"))}</b><div>${esc(risk.sentence || "")}</div></div></div>` : "";
    const files = (j.file_entries || []).map((f) => {
      let act = "";
      if (f.access === "in_root" && f.exists !== false) act = `<button type="button" class="tb-btn-text" data-open="${esc(f.rel)}" data-kind="${f.kind}">${icon("external-link")}<span>Open</span></button>`;
      else if (f.access === "other_root") act = `<button type="button" class="tb-btn-text" data-copy="${esc(f.command)}" title="Run this command to open that folder in a new GUI window">${icon("copy")}<span>Open in a New Window Rooted at ${esc(f.open_dir)}…</span></button>`;
      else act = `<span class="muted small">Outside the folders this GUI may open</span> <button type="button" class="tb-btn-text" data-copy="${esc(f.path)}">${icon("copy")}<span>Copy Path</span></button>`;
      return `<tr><td>${esc(f.label)}</td><td class="mono small">${esc(f.path)}</td><td>${act}</td></tr>`;
    }).join("");
    const age = d._at ? Math.round((Date.now() - d._at) / 1000) : null;
    body.innerHTML = `<div class="jbv-head"><h2>Job ${esc(j.id)} · ${esc(j.name || "")}</h2><span class="chip state-${esc((j.state || "").toLowerCase())}">${esc(j.state)}</span>
        ${j.nodelist ? `<span class="small">on ${esc(j.nodelist)}</span>` : ""}<span class="small muted jbv-age">${age === null ? "" : `Updated ${age} s ago`}</span>
        <button type="button" class="tb-btn-text" data-act="refresh">${icon("refresh-cw")}<span>Refresh</span></button></div>
      ${j.state_note ? `<p class="small">${esc(j.state_note)}</p>` : ""}
      ${j.reason && j.reason !== "None" ? `<p class="small">Reason: ${esc(j.reason)}${j.state === "PENDING" && j.start && j.start !== "N/A" ? ` · expected start ${esc(localTime(j.start))}` : ""}</p>` : ""}
      <div class="jbv-time"><span class="small">Elapsed ${fmtDur(j.elapsed_s)} of ${fmtDur(j.limit_s)}</span><span class="clv-bar ${risk.level === "likely to exceed" ? "critical" : risk.level === "tight" ? "warning" : ""}" role="meter" aria-label="Elapsed of the walltime" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}"><i style="width:${pct}%"></i></span></div>
      ${banner}${prog}${totals}
      <h3>Resources</h3><p class="small">${esc(j.ncpus || "?")} cores on ${esc(j.nnodes || "?")} node(s), partition ${esc(j.partition || "?")}, account ${esc(j.account || "?")}</p>
      <div class="jbv-eff small">${effHtml(eff)}</div>
      <p class="small">${j.state === "RUNNING" ? `<button type="button" class="tb-btn-text" data-act="live">${d._liveEff ? "Update Live Use" : "Show Live Use"}</button>` : ""}<span class="jbv-live small muted">${esc(d._liveNote || "")}</span></p>
      ${arrayTasks(j.array_tasks)}
      <h3>Files</h3><table class="data clv-table"><tbody>${files}</tbody></table>
      <p><button type="button" class="tb-btn-text" data-copy="${esc(j.scontrol_command)}">${icon("copy")}<span>Copy scontrol Command</span></button></p>`;
  }
  // The ETA as nested levels (total, stages, inner loop), the sentences
  // as the server built them; finished stages are left out for an
  // optimization. "Why this estimate" keeps its open state per document.
  function levelsHtml(pr, d) {
    const shown = pr.levels.filter((l) => l.depth === 0 || l.status !== "done" || pr.mode === "scf" || pr.mode === "freq");
    const items = shown.map((l) => `<li class="jbv-level d${l.depth} ${esc(l.status)}"><span class="jbv-lt">${esc(l.text)}</span></li>`).join("");
    const why = pr.levels.filter((l) => l.basis).map((l) => `<li>${l.depth === 0 ? esc((l.info && l.info.basis_head) || l.basis) : `${esc(l.label)}: ${esc(l.basis)}`}</li>`).join("");
    const thr = [];
    for (const l of pr.levels) for (const t of l.thresholds || []) if (t.note) thr.push(`${esc(t.name)} ${esc(sci(t.value))} (${esc(t.note)})`);
    const thrLine = `<li>Thresholds: ${thr.length ? thr.join(", ") : "CRYSTAL23 defaults"}</li>`;
    return `<ol class="jbv-levels" aria-label="Progress and time left">${items}</ol>
      <details class="jbv-why"${d._whyOpen ? " open" : ""}><summary>Why this estimate</summary><ul class="small">${why}${thrLine}</ul></details>`;
  }
  // CPU and memory use as the server worded it (mace/cluster/job_efficiency.py);
  // the measured values stay behind a toggle.
  function effHtml(e) {
    const caveat = e.cpu_caveat && e.cpu_line ? `<div class="muted">${esc(e.cpu_caveat)}</div>` : "";
    const lines = (e.cpu_line ? `<div>${esc(e.cpu_line)}</div>` : "") + caveat + (e.memory_line ? `<div>${esc(e.memory_line)}</div>` : "");
    const advice = (e.advice || []).map((a) => `<div class="jbv-advice">${esc(a)}</div>`).join("");
    const det = (e.details || []).length ? `<details class="jbv-eff-details"><summary>Measured values</summary>${esc(e.details.join(", "))}</details>` : "";
    return lines + advice + det;
  }
  function sci(v) {
    if (v === null || v === undefined) return "?";
    const [m, e] = Number(v).toExponential(1).split("e");
    return `${m.endsWith(".0") ? m.slice(0, -2) : m}e${parseInt(e, 10)}`;
  }
  function arrayTasks(a) {
    if (!a || !a.tasks || !a.tasks.length) return "";
    const by = Object.entries(a.counts || {}).map(([st, n]) => `${n} ${st.toLowerCase()}`).join(", ");
    const more = a.total_in_queue > a.tasks.length ? `<p class="small muted">The first ${a.tasks.length} of ${num(a.total_in_queue)} are listed.</p>` : "";
    return `<h3>Array tasks</h3><p class="small">Array ${esc(a.array_job)}: ${num(a.total_in_queue)} in the queue (${esc(by)})</p>
      <table class="data clv-table jbv-array"><thead><tr><th>Task</th><th>State</th><th>Elapsed</th><th></th></tr></thead><tbody>${a.tasks.map((t) =>
        `<tr${t.id === cur.jobId ? ' class="current"' : ""}><td class="mono">${esc(t.id)}</td><td>${esc(t.state)}${t.state === "PENDING" && t.reason && t.reason !== "None" ? ` (${esc(t.reason)})` : ""}</td><td>${esc(t.elapsed)}</td>
        <td>${t.id === cur.jobId ? '<span class="small muted">this task</span>' : `<button type="button" class="tb-btn-text" data-job="${esc(t.id)}" aria-label="Details of task ${esc(t.id)}">Details</button>`}</td></tr>`).join("")}</tbody></table>${more}`;
  }
  body.addEventListener("toggle", (e) => {
    if (cur && e.target.classList && e.target.classList.contains("jbv-why")) cur._whyOpen = e.target.open;
  }, true);
  body.addEventListener("click", async (e) => {
    const d = cur; if (!d) return;
    const jb = e.target.closest("[data-job]");
    if (jb) { H.openJob(jb.dataset.job); return; }
    const o = e.target.closest("[data-open]");
    if (o) { H.openPath(o.dataset.open, o.dataset.kind); return; }
    const c = e.target.closest("[data-copy]");
    if (c) { H.copy(c.dataset.copy); return; }
    const a = e.target.closest("[data-act]");
    if (!a) return;
    if (a.dataset.act === "refresh") load(d, true);
    if (a.dataset.act === "live") {
      try {
        const r = await api("cluster/job/live", { body: { id: d.jobId } });
        if ((r.rows || []).length && r.efficiency) { d._liveEff = r.efficiency; d._liveNote = ""; }
        else d._liveNote = r.error ? `sstat: ${r.error}` : "sstat lists no running steps for this job";
        const el = body.querySelector(".jbv-eff"); if (el && d._liveEff) el.innerHTML = effHtml(d._liveEff);
        const ln = body.querySelector(".jbv-live"); if (ln) ln.textContent = d._liveNote;
      } catch (err) { toast(err.message, "error"); }
    }
  });
  return {
    async open(d) {
      cur = d;
      render(d);
      await load(d);
      clearInterval(timer);
      let every = 60000;
      try { const c = await api("cluster/capabilities"); every = Math.max(1, c.poll_seconds || 60) * 1000; } catch (e) { /* keep 60 s */ }
      timer = setInterval(() => { if (cur === d && host.offsetParent && document.visibilityState === "visible") load(d); }, every);
    },
    refreshIf(id) { if (cur && cur.jobId === id) load(cur); },
    close(d) { if (!d || cur === d) { clearInterval(timer); cur = null; } },
  };
}
