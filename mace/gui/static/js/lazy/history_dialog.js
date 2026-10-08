/* File History (PLAN_analysis C20): the versions MACE kept of one file
 * before it replaced it (history.py, outside the served folder). Each
 * version: Open (read-only, here), Compare with Current (side by side) and
 * Restore This Version (a guarded write: the current version is kept first).
 * Loaded on first use. */
import { loadCss } from "./lazy_util.js";

const fmtTime = (t) => new Date(t * 1000).toLocaleString("en-US", { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });
const fmtSize = (n) => (n >= 1048576 ? `${(n / 1048576).toFixed(1)} MB` : n >= 1024 ? `${(n / 1024).toFixed(1)} KB` : `${n} B`);
const TEXT_MAX = 400000;

export async function open(H, path, opts = {}) {
  await loadCss(import.meta.url, "../../css/dialogs.css");
  H.overlayOpening();
  const opener = document.activeElement;
  let d = document.getElementById("file-history");
  if (!d) { d = document.createElement("dialog"); d.id = "file-history"; document.body.appendChild(d); }
  d.className = "dlg hist-dlg";
  d.innerHTML = `<div class="dlg-head"><h2></h2><button class="icon-btn sm" data-close aria-label="Close">${H.icon("x")}</button></div>
    <div class="dlg-body"><p class="muted small hist-where"></p><div class="hist-list" role="list"></div>
      <div class="hist-view" hidden><div class="hist-view-head"><b class="hist-view-title"></b>
        <button class="tb-btn-text" data-act="back">${H.icon("chevron-left")}<span>Versions</span></button></div>
        <div class="hist-panes"></div></div></div>`;
  d.querySelector("h2").textContent = `File History: ${path.split("/").pop()}`;
  d.querySelector("[data-close]").addEventListener("click", () => d.close());
  d.addEventListener("close", () => { if (opener && opener.focus) opener.focus(); }, { once: true });
  const list = d.querySelector(".hist-list"), view = d.querySelector(".hist-view");
  let state = null;

  async function load() {
    state = await H.api("history?" + new URLSearchParams({ path }));
    d.querySelector(".hist-where").textContent = (state.note ? state.note + ". " : "") +
      "Kept outside the folder, per file: the last versions of the past days (Settings › Files and data).";
    if (!state.entries.length) { list.innerHTML = `<p class="muted">No earlier versions: MACE has not replaced this file.</p>`; return; }
    list.innerHTML = state.entries.map((e, k) => `<div class="hist-row" role="listitem" data-id="${H.esc(e.id)}">
      <div class="hist-meta"><b>${H.esc(fmtTime(e.t))}</b><span class="muted small">${e.created ? "created by MACE" : fmtSize(e.size)}</span>
        <span class="small">${H.esc(e.reason || "")}</span></div>
      <div class="hist-acts">${e.created ? "" : `<button class="tb-btn-text" data-act="open">Open</button>
        <button class="tb-btn-text" data-act="compare">Compare with Current</button>`}
        <button class="tb-btn-text${k === 0 ? " primary" : ""}" data-act="restore"${state.protected ? ` disabled title="${H.esc(state.protected)}"` : ""}>${e.created ? "Delete the File MACE Created" : "Restore This Version"}</button></div></div>`).join("");
  }
  async function text(id) {
    const r = await fetch("api/history/file?" + new URLSearchParams({ id }), { credentials: "same-origin", headers: { "X-Requested-With": "mace-gui" } });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
    const t = await r.text();
    return t.length > TEXT_MAX ? t.slice(0, TEXT_MAX) + "\n… (cut)" : t;
  }
  async function current() {
    try {
      const w = await H.api("text?" + new URLSearchParams({ path }));
      return w.text ?? (w.lines || []).join("\n");
    } catch (e) { return `(${e.message})`; }
  }
  function show(title, panes) {
    list.hidden = true; view.hidden = false;
    view.querySelector(".hist-view-title").textContent = title;
    view.querySelector(".hist-panes").innerHTML = panes.map(([h, t]) => `<section class="hist-pane"><h4>${H.esc(h)}</h4><pre class="hist-pre" tabindex="0"></pre></section>`).join("");
    view.querySelectorAll(".hist-pre").forEach((pre, k) => { pre.textContent = panes[k][1]; });
    view.querySelector("[data-act=back]").focus();
  }
  d.addEventListener("click", async (ev) => {
    const b = ev.target.closest("[data-act]");
    if (!b) return;
    if (b.dataset.act === "back") { view.hidden = true; list.hidden = false; return; }
    const row = b.closest(".hist-row"), id = row && row.dataset.id, e = state && state.entries.find((x) => x.id === id);
    if (!e) return;
    try {
      if (b.dataset.act === "open") show(`${fmtTime(e.t)} (read-only)`, [[e.reason || "This version", await text(id)]]);
      else if (b.dataset.act === "compare") {
        const [a, c] = await Promise.all([text(id), current()]);
        show("Compared with the current file", [[`${fmtTime(e.t)}`, a], ["Current", c]]);
      } else if (b.dataset.act === "restore") {
        b.disabled = true;
        const r = await H.api("history/restore", { body: { id, path, expected: state.file } });
        H.toast(r.deleted ? `Deleted ${path.split("/").pop()} (MACE had created it)` : `Restored ${path.split("/").pop()} from ${fmtTime(e.t)}`, "success");
        if (opts.afterRestore) opts.afterRestore(r);
        await load();
      }
    } catch (err) {
      b.disabled = false;
      H.toast(err.message, "error");
    }
  });
  d.showModal();
  await load();
  const first = d.querySelector(".hist-row button");
  if (first) first.focus();
  return d;
}

/** Restore one entry without the dialog (the save toast's Undo). */
export async function restore(H, path, id, expected) {
  return H.api("history/restore", { body: { id, path, expected } });
}
