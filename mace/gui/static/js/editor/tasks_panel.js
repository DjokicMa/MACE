/* The Tasks bottom panel (spec §1.6, §8.4.8): one row per background task,
 * newest first, with its progress, elapsed time, a one-line log and its
 * actions: Cancel while it runs; Details, Retry, Copy Error and Remove when
 * it failed; Remove when it is done or canceled. Loaded the first time a
 * task appears or the panel opens (never at boot). */
import { loadCss } from "../lazy/lazy_util.js";

const css = loadCss(import.meta.url, "../../css/panels.css");

const fmtS = (s) => (s < 10 ? `${s.toFixed(1)} s` : s < 60 ? `${Math.round(s)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`);
const STATE = { queued: "Waiting", running: "Running", done: "Done", failed: "Failed", canceled: "Canceled", lost: "Lost" };

export function render(host, tasks, H) {
  css.then(() => host.classList.add("styled"));
  const list = [...tasks.values()].sort((a, b) => (b.started || b.t0 || 0) - (a.started || a.t0 || 0));
  if (!list.length) {
    host.innerHTML = `<p class="empty-note" id="tasks-note">No tasks. Builds, Clean and big files show their progress here.</p>`;
    return;
  }
  const open = new Set([...host.querySelectorAll(".task-row.details-open")].map((r) => r.dataset.task));
  host.innerHTML = `<ul class="task-list" role="list">${list.map((t) => row(t, H, open.has(t.id))).join("")}</ul>`;
}

function row(t, H, detailsOpen) {
  const esc = H.esc, running = t.state === "running" || t.state === "queued";
  const pct = Math.round((t.progress || 0) * 100);
  const log = t.state === "canceled" ? `Canceled after ${fmtS(t.elapsed || 0)}`
    : t.state === "lost" ? "Lost when the server stopped" : t.state === "failed" ? (t.error || "Failed") : (t.message || "");
  const acts = [];
  if (running) acts.push(`<button class="tb-btn-text" data-act="cancel">Cancel</button>`);
  if (t.state === "failed") {
    acts.push(`<button class="tb-btn-text" data-act="details" aria-expanded="${detailsOpen}">Details</button>`,
      `<button class="tb-btn-text" data-act="retry">${H.icon("rotate-ccw")}<span>Retry</span></button>`,
      `<button class="tb-btn-text" data-act="copy">Copy Error</button>`);
  }
  if (t.state === "done" && t.open) acts.push(`<button class="tb-btn-text" data-act="open">Open</button>`);
  if (!running) acts.push(`<button class="icon-btn sm" data-act="remove" aria-label="Remove ${esc(t.title)}">${H.icon("x")}</button>`);
  return `<li class="task-row s-${esc(t.state)}${detailsOpen ? " details-open" : ""}" data-task="${esc(t.id)}">
    <span class="task-chip">${esc(STATE[t.state] || t.state)}</span>
    <span class="task-title" title="${esc(t.title)}">${esc(t.title)}</span>
    <span class="task-bar" role="progressbar" aria-label="${esc(t.title)}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}"><span style="width:${pct}%"></span></span>
    <span class="task-time st-num">${fmtS(t.elapsed || 0)}</span>
    <span class="task-log" title="${esc(log)}">${esc(log)}</span>
    <span class="task-acts">${acts.join("")}</span>
    ${t.state === "failed" ? `<pre class="task-tb mono">${esc(t.traceback || t.error || "")}</pre>` : ""}</li>`;
}

/** Clicks on a row's buttons: cancel, details, retry, copy, remove, open. */
export function bind(host, actions) {
  if (host.dataset.bound) return;
  host.dataset.bound = "1";
  host.addEventListener("click", (e) => {
    const b = e.target.closest("[data-act]"), r = e.target.closest(".task-row");
    if (!b || !r) return;
    const id = r.dataset.task, act = b.dataset.act;
    if (act === "details") {
      r.classList.toggle("details-open");
      b.setAttribute("aria-expanded", String(r.classList.contains("details-open")));
      return;
    }
    if (actions[act]) actions[act](id);
  });
}
