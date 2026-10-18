/* Update MACE (1.1.6): the dialog behind the status bar's "MACE x.y.z
 * available" and Help ▸ Check for Updates…. The server only drives `mace
 * update` (update_gui.py): GET api/update is the state, POST
 * api/update/check looks again, POST api/update/choose acts (now, signout,
 * cancel-signout, later, skip, ack). Update Now and Restart: the server
 * updates, checks the new code starts, and restarts in place on the same
 * port with the same sign-in; this page waits for it and reloads.
 * Loaded on first use. */
import { loadCss } from "./lazy_util.js";
import { dialog } from "../ui.js";

const CHANNELS = [["release", "Release"], ["preview", "Preview"], ["latest", "Latest (main)"]];
let H = null, state = null, dlg = null, channel = null, busy = "", waitingRestart = false, lastResult = null;

export function bind(h) { H = h; }

/* The CHANGELOG sections (Keep a Changelog markdown): headings, nested
 * bullets, `code`, **bold** and [links](url); everything else as text. */
function inline(t) {
  let s = H.esc(t);
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
  return s.replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
}
export function renderChangelog(lines) {
  const out = [];
  const stack = [];          // open <ul> indents
  let para = [];
  const flushPara = () => { if (para.length) { out.push(`<p>${inline(para.join(" "))}</p>`); para = []; } };
  const closeTo = (indent) => { while (stack.length && stack[stack.length - 1] >= indent) { out.push("</li></ul>"); stack.pop(); } };
  for (const raw of lines || []) {
    const line = raw.replace(/\s+$/, "");
    const h = /^(#{2,4})\s+(.*)$/.exec(line);
    const li = /^(\s*)[-*]\s+(.*)$/.exec(line);
    if (h) {
      flushPara(); closeTo(-1);
      const level = Math.min(h[1].length + 1, 5);
      out.push(`<h${level}>${inline(h[2].replace(/^\[([^\]]+)\]/, "$1"))}</h${level}>`);
    } else if (li) {
      flushPara();
      const indent = li[1].length;
      if (stack.length && stack[stack.length - 1] === indent) out.push("</li><li>");
      else if (!stack.length || stack[stack.length - 1] < indent) { out.push("<ul><li>"); stack.push(indent); }
      else { closeTo(indent + 1); if (stack.length && stack[stack.length - 1] === indent) out.push("</li><li>"); else { out.push("<ul><li>"); stack.push(indent); } }
      out.push(inline(li[2]));
    } else if (!line.trim()) {
      flushPara();
    } else if (stack.length && /^\s+/.test(raw)) {
      out.push(" " + inline(line.trim()));         // a bullet's continuation line
    } else {
      closeTo(-1); para.push(line.trim());
    }
  }
  flushPara(); closeTo(-1);
  return out.join("");
}

const ver = (x) => (x && x.version) || "?";
const when = (t) => (t ? new Date(t * 1000).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "");

function copyRow(id, text) {
  return `<div class="upd-cmd"><code id="${id}">${H.esc(text)}</code><button class="tb-btn-text" type="button" data-copy="${id}">${H.icon("copy")}<span>Copy</span></button></div>`;
}

function body() {
  const s = state || {}, r = s.report, parts = [];
  if (s.result || lastResult) {
    const res = s.result || lastResult;
    parts.push(`<section class="upd-result ${res.ok ? "ok" : "bad"}" id="upd-result"><h3>${res.ok ? "MACE was updated" : "The update failed"}${res.to ? `: ${H.esc(res.from || "?")} → ${H.esc(res.to)}` : ""}</h3>
      ${res.how === "signout" ? `<p>It ran after you signed out${res.time ? ` (${H.esc(when(res.time))})` : ""}.</p>` : ""}
      ${res.restart_why ? `<p>${H.esc(res.restart_why.split("\n")[0])}</p>` : ""}
      <details><summary>Output of mace update</summary><pre class="upd-pre">${H.esc(res.output || "")}</pre></details></section>`);
  }
  if (busy === "updating" || s.phase === "updating" || s.phase === "waiting") {
    parts.push(`<p class="upd-busy" role="status"><span class="spinner sm" aria-hidden="true"></span> ${s.phase === "waiting" ? "Waiting for the running tasks to finish, then updating…" : "Updating MACE… (git fetch and the move can take a minute)"}</p>`);
  }
  if (busy === "restarting" || s.phase === "restarting") {
    parts.push(`<p class="upd-busy" role="status"><span class="spinner sm" aria-hidden="true"></span> MACE is updated. Restarting MACE GUI… this page reloads when the server is back. ${H.esc((s.restart || {}).text || "")}</p>`);
  }
  if (busy === "checking" || s.checking) parts.push(`<p class="upd-busy" role="status"><span class="spinner sm" aria-hidden="true"></span> Checking for updates…</p>`);
  const sel = `<label class="upd-channel">Channel <select id="upd-channel">${CHANNELS.map(([v, l]) => `<option value="${v}"${v === (channel || s.channel) ? " selected" : ""}>${l}</option>`).join("")}</select></label>
    <button class="tb-btn-text" type="button" id="upd-recheck">${H.icon("refresh-cw")}<span>Check Again</span></button>`;
  parts.push(`<div class="upd-row">${sel}<span class="muted small">${s.checked_at ? `Checked ${H.esc(when(s.checked_at))}` : "Not checked yet"}${s.enabled ? `; checks every ${s.interval_hours} h (Settings ▸ Server)` : "; automatic checks are off (Settings ▸ Server)"}</span></div>`);
  if (s.error) parts.push(`<p class="upd-note" id="upd-error">${H.esc(s.error)}</p>`);
  if (r && r.current && r.current.version) {
    const t = r.target || {};
    parts.push(`<table class="kv upd-versions"><tr><td>Installed</td><td><b>${H.esc(ver(r.current))}</b> <span class="muted">(${H.esc(r.current.describe || "")}${r.current.branch ? `, on ${H.esc(r.current.branch)}` : ", detached"})</span></td></tr>
      <tr><td>${H.esc(CHANNELS.find((c) => c[0] === r.channel)?.[1] || r.channel)}</td><td><b>${H.esc(t.version || "?")}</b> <span class="muted">(${H.esc(t.ref || "")}${t.commits ? `, ${t.commits} commit${t.commits === 1 ? "" : "s"} ahead` : ""})</span></td></tr>
      <tr><td>Install</td><td><code>${H.esc(s.home || r.home || "")}</code></td></tr></table>`);
    if (!r.available) parts.push(`<p id="upd-uptodate">${r.state === "newer" ? H.esc(r.message) : r.state === "current" ? `MACE is up to date on this channel.` : ""}</p>`);
  }
  const warn = [];
  if (s.blocked) warn.push(`<li id="upd-blocked">${H.esc(s.blocked)}</li>`);
  if (r && r.available && s.maintainer) warn.push(`<li id="upd-maintainer">${H.esc(s.maintainer)}<pre class="upd-pre">${H.esc(r.refusal || "")}</pre></li>`);
  else if (r && r.refusal && r.available !== undefined && r.state !== "unknown") {
    warn.push(`<li id="upd-refusal"><span>mace update would refuse:</span><pre class="upd-pre">${H.esc(r.refusal)}</pre>${r.stash ? `Keep your changes aside first, then check again:${copyRow("upd-stash", r.stash)}` : ""}</li>`);
  }
  if (r && r.available && r.busy && r.busy.jobs) {
    const b = r.busy;
    warn.push(`<li id="upd-jobs">${b.jobs} SLURM job${b.jobs === 1 ? "" : "s"} of yours ${b.jobs === 1 ? "is" : "are"} in the queue (${b.running} running, ${b.waiting} waiting). Their completion callbacks run the code in this install. Updating while jobs run is generally safe for a fast-forward, but workflows in progress continue their next steps with the new code.</li>`);
  }
  if (r && r.available && (s.tasks || []).length) {
    warn.push(`<li id="upd-tasks">${s.tasks.length} MACE GUI task${s.tasks.length === 1 ? " is" : "s are"} running: ${s.tasks.map((t) => H.esc(t)).join(", ")}. The restart would cancel ${s.tasks.length === 1 ? "it" : "them"}, so Update Now waits until ${s.tasks.length === 1 ? "it finishes" : "they finish"}.</li>`);
  }
  if (r && r.available && r.requirements) {
    const q = r.requirements;
    warn.push(`<li id="upd-req">${H.esc(q.files.join(", "))} changed${q.changes.length ? `: <code>${q.changes.map((x) => H.esc(x)).join("</code>, <code>")}</code>` : ""}. MACE runs with ${H.esc(q.python_desc)}.
      ${q.do_it_yourself ? "<b>This is a system or module Python: install them yourself</b> (or set up your own environment) rather than from here." : "Install them into the Python that runs MACE:"}
      ${copyRow("upd-pip", q.pip)}
      <label class="chk"><input type="checkbox" id="upd-run-pip"> Also run this pip command during the update</label></li>`);
  }
  if (r && r.available && r.optional_changed) warn.push(`<li>requirements-optional.txt changed too; install from it only if you use those optional features.</li>`);
  if (warn.length) parts.push(`<ul class="upd-warn" aria-label="Before you update">${warn.join("")}</ul>`);
  if (s.deferred) parts.push(`<p id="upd-deferred">${H.icon("clock")} MACE ${H.esc(s.deferred.version || "")} installs when you use File ▸ Sign Out and Stop Server (after the server has stopped); the result shows at the next start.</p>`);
  if (r && (r.changelog || []).length) {
    parts.push(`<h3 class="upd-h">What changed</h3><div class="upd-log" id="upd-log" tabindex="0" role="region" aria-label="Changes from the CHANGELOG">${renderChangelog(r.changelog)}</div>`);
  }
  return parts.join("");
}

function foot() {
  const s = state || {}, r = s.report;
  const can = r && r.available && !r.refusal && !s.blocked && !busy && !["waiting", "updating", "restarting"].includes(s.phase);
  const btn = [];
  if (r && r.available && !busy) {
    btn.push(`<button class="tb-btn-text" type="button" data-u="skip" id="upd-skip">Skip This Version</button>`);
    btn.push(`<button class="tb-btn-text" type="button" data-u="later" id="upd-later">Not Now</button>`);
  } else btn.push(`<button class="tb-btn-text" type="button" data-u="close" id="upd-close">Close</button>`);
  if (can) {
    btn.push(s.deferred ? `<button class="tb-btn-text" type="button" data-u="cancel-signout" id="upd-cancel-signout">Don't Update at Sign Out</button>`
      : `<button class="tb-btn-text" type="button" data-u="signout" id="upd-signout">Update When I Sign Out</button>`);
    const tasks = (s.tasks || []).length;
    btn.push(`<button class="tb-btn-text primary" type="button" data-u="now" id="upd-now">${tasks ? "Update When Tasks Finish" : "Update Now and Restart"}</button>`);
  }
  return btn.join("");
}

function render() {
  if (!dlg || !dlg.open) return;
  const scroll = dlg.querySelector(".dlg-body")?.scrollTop || 0;
  dlg.querySelector(".dlg-body").innerHTML = body();
  let f = dlg.querySelector(".dlg-foot");
  if (!f) { f = document.createElement("div"); f.className = "dlg-foot"; dlg.appendChild(f); }
  f.innerHTML = foot();
  dlg.querySelector(".dlg-body").scrollTop = scroll;
}

function adopt(d) {
  state = d;
  if (d && d.result) lastResult = d.result;
  // an update that failed, finished (a manual restart) or never started: the spinner and the hidden buttons end
  if (busy === "updating" && (!d || !d.phase || ["failed", "done", "idle"].includes(d.phase))) busy = "";
  if (H.setUpdate) H.setUpdate(d);
  if (d && d.phase === "restarting") waitRestart();
  render();
}

async function act(choice, extra = {}) {
  try {
    const d = await H.api("update/choose", { body: { choice, ...extra } });
    adopt(d);
    return d;
  } catch (e) {
    H.toast(e.message, "error");
    await refresh();
    return null;
  }
}

export async function refresh() {
  try { adopt(await H.api("update")); } catch (e) { /* the server is restarting */ }
  return state;
}

async function check() {
  busy = "checking"; render();
  try { adopt(await H.api("update/check", { body: { channel: channel || (state && state.channel) } })); }
  catch (e) { H.toast(`Check for Updates: ${e.message}`, "error"); }
  finally { busy = ""; render(); }
}

/* The server restarts in place: wait for a new process (its `started`
 * changes), then reload, so the page runs the updated code too. */
async function waitRestart() {
  if (waitingRestart) return;
  waitingRestart = true;
  busy = "restarting"; render();
  const old = state && state.started;
  if (H.flushDrafts) H.flushDrafts();
  window.__maceRestarting = true;
  const end = Date.now() + 5 * 60 * 1000;
  while (Date.now() < end) {
    await new Promise((r) => setTimeout(r, 1000));
    try {
      const res = await fetch("api/update", { credentials: "same-origin", headers: { "X-Requested-With": "mace-gui" } });
      if (res.ok) {
        const d = await res.json();
        if (d.started !== old) { if (H.leave) H.leave(location.href.split("#")[0]); else location.reload(); return; }
      }
    } catch (e) { /* not back yet */ }
  }
  waitingRestart = false; busy = "";
  H.toast("MACE GUI did not come back after the update. Start it again (mace gui, or mace gui connect).", "error");
}

export function progress(h, d) {
  H = h;
  if (!d) return;
  if (d.phase === "restarting") { if (state) state.phase = "restarting"; waitRestart(); return; }
  if ((d.phase === "failed" || d.phase === "done") && busy === "updating") busy = "";
  refresh().then(() => {
    if (d.phase === "failed") H.toast("The MACE update failed; Help ▸ Check for Updates shows why.", "error");
    if (d.phase === "done" && d.result && d.result.restart === "manual") open(h);
  });
}

/* At page load: a result from the last update (Update Now, or at the sign-out) */
export async function startup(h) {
  H = h;
  const s = await refresh();
  if (s && s.result) {
    const res = s.result;
    H.toast(res.ok ? `MACE was updated to ${res.to || "the new version"}.` : "The MACE update did not finish.", res.ok ? "success" : "error",
      { label: "Details", run: () => open(h) });
    if (res.ok) act("ack");
  }
}

export async function open(h, opts = {}) {
  H = h;
  await loadCss(import.meta.url, "../../css/update.css");
  dlg = dialog({ id: "update-dlg", title: "Update MACE", body: `<p class="upd-busy" role="status">Loading…</p>`, foot: " ", onClose: () => { dlg = null; } });
  dlg.classList.add("upd-dlg");
  dlg.addEventListener("click", async (e) => {
    const cp = e.target.closest("[data-copy]");
    if (cp) { const el = dlg.querySelector("#" + cp.dataset.copy); if (el) H.copyText(el.textContent, () => H.toast("Copied", "success")); return; }
    if (e.target.closest("#upd-recheck")) { check(); return; }
    const b = e.target.closest("[data-u]");
    if (!b) return;
    const u = b.dataset.u;
    if (u === "close") { dlg.close(); return; }
    if (u === "later" || u === "skip") { await act(u); if (dlg) dlg.close(); return; }
    if (u === "cancel-signout") { await act(u); return; }
    const pip = !!(dlg.querySelector("#upd-run-pip") || {}).checked;
    if (u === "signout") { const d = await act("signout", { pip }); if (d) H.toast(`MACE ${d.deferred.version} installs when you sign out and stop the server.`, "info"); return; }
    if (u === "now") {
      const tasks = (state.tasks || []).length;
      busy = "updating"; render();
      const d = await act("now", { pip, wait: tasks > 0 });
      if (!d) { busy = ""; render(); }
    }
  });
  dlg.addEventListener("change", (e) => { if (e.target.id === "upd-channel") { channel = e.target.value; check(); } });
  await refresh();
  if (opts.check || !state || !state.report) await check();
  else render();
  if (state && state.result && state.result.ok) act("ack");
  return dlg;
}
