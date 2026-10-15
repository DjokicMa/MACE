/* Connection and session UI (plan PLAN_connect T6, T7, T10, T11, T12), loaded
 * the first time one of them opens: File ▸ Open Folder…, Help ▸ Connecting
 * from Your Computer, File ▸ Sign Out and Stop Server…, the Signed Out
 * modal, Copy Diagnostics and Report a Problem…. `H` is what the shell
 * lends (shell.js). */
import { dialog } from "../ui.js";
import { loadCss } from "./lazy_util.js";
import * as profile from "../profile.js";

const css = () => loadCss(import.meta.url, "../../css/connect.css");

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
/* Recent folders are the profile's (gui.json recent.folders, profile.js), so
 * they come back on a new port and in another browser; the dialog shows 8. */
const recentFolders = () => profile.recentFolders().filter((p) => typeof p === "string" && p).slice(0, 8);
function rememberFolder(p) {
  // sent as the page unloads (profile.js leaveFlush), so nothing waits between the switch
  // and the reload (the other tabs' "Folder Changed" would come first)
  profile.setRecentFolders([p, ...profile.recentFolders().filter((x) => x !== p)]);
}

/* Modified documents first: Save All, Don't Save (drafts stay in this
 * browser) or Cancel. Resolves true to go on. */
export function confirmDirty(H, what) {
  const names = H.dirtyDocs();
  if (!names.length) return Promise.resolve(true);
  return new Promise((resolve) => {
    let choice = "cancel";
    const d = dialog({ id: "dirty-first", title: "Save Changes?", narrow: true,
      body: `<p>${names.length === 1 ? `${H.esc(names[0])} has` : `${names.length} documents have`} unsaved changes.
        ${H.esc(what)}</p><p class="insp-hint">Don't Save keeps them as drafts in this browser.</p>`,
      foot: `<button class="tb-btn-text" data-r="cancel">Cancel</button><button class="tb-btn-text" data-r="discard">Don't Save</button>` +
        `<button class="tb-btn-text primary" data-r="save">Save All</button>`,
      onClose: async () => {
        if (choice === "save") { try { await H.saveAll(); } catch (e) { resolve(false); return; } resolve(!H.dirtyDocs().length); }
        else resolve(choice === "discard");
      } });
    $$("[data-r]", d).forEach((b) => b.addEventListener("click", () => { choice = b.dataset.r; d.close(); }));
  });
}

/* ---------------------------------------------------------------- Open Folder */
export async function openFolder(H) {
  await css();
  const info = H.info() || {};
  if (!info.open_folder) {
    H.toast("Turned off when this server started (--root none)", "warning");
    return;
  }
  let listing = null, selected = null;
  const d = dialog({ id: "open-folder", title: "Open Folder", body: `
      <div class="of-roots" role="group" aria-label="Folders this server may open" id="of-roots"></div>
      <nav class="of-crumbs" id="of-crumbs" aria-label="Folder path"></nav>
      <ul class="of-list" id="of-list" role="listbox" aria-label="Folders" tabindex="0"></ul>
      <p class="insp-hint" id="of-trunc" hidden>Only the first 2000 folders are listed.</p>
      <label class="of-paste">Paste a path <input id="of-path" type="text" spellcheck="false" autocomplete="off" placeholder="/path/to/folder"></label>
      <p class="of-error" id="of-error" role="alert" hidden></p>
      <p class="insp-hint">Open now: <code id="of-current" class="root-path"></code></p>
      <div id="of-recent-wrap" hidden><h3 class="of-h">Recent folders</h3><ul class="of-recent" id="of-recent"></ul></div>`,
  foot: `<button class="tb-btn-text" data-close-dlg>Cancel</button><button class="tb-btn-text primary" id="of-open">Open</button>` });
  $("[data-close-dlg]", d).addEventListener("click", () => d.close());
  $("#of-current", d).textContent = info.root || "";
  const showError = (msg) => { const e = $("#of-error", d); e.textContent = msg || ""; e.hidden = !msg; };
  async function list(path) {
    showError("");
    try {
      listing = await H.api("folders" + (path ? `?path=${encodeURIComponent(path)}` : ""));
    } catch (e) { showError(e.message); return; }
    selected = null;
    $("#of-roots", d).innerHTML = listing.roots.map((r) => `<button class="chip" data-root="${H.esc(r.path)}" title="${H.esc(r.path)}">${H.esc(r.label)}</button>`).join("");
    const root = listing.roots.map((r) => r.path).filter((r) => listing.path === r || listing.path.startsWith(r.replace(/\/$/, "") + "/"))
      .sort((a, b) => b.length - a.length)[0] || "";
    const parts = [], rel = root ? listing.path.slice(root.length).split("/").filter(Boolean) : listing.path.split("/").filter(Boolean);
    let acc = root || "";
    if (root) parts.push(`<button class="crumb" data-go="${H.esc(root)}">${H.esc((listing.roots.find((r) => r.path === root) || {}).label || root)}</button>`);
    for (const p of rel) { acc = (acc.endsWith("/") ? acc : acc + "/") + p; parts.push(`<button class="crumb" data-go="${H.esc(acc)}">${H.esc(p)}</button>`); }
    $("#of-crumbs", d).innerHTML = parts.join('<span class="crumb-sep" aria-hidden="true">/</span>');
    const ul = $("#of-list", d);
    ul.innerHTML = (listing.parent ? `<li role="option" data-dir=".." class="of-up">..</li>` : "") +
      listing.dirs.map((n) => `<li role="option" data-dir="${H.esc(n)}">${H.icon("folder")}<span>${H.esc(n)}</span></li>`).join("") +
      (listing.dirs.length ? "" : `<li class="of-empty" aria-disabled="true">No folders here</li>`);
    $("#of-trunc", d).hidden = !listing.truncated;
  }
  const child = (name) => (name === ".." ? listing.parent : listing.path.replace(/\/$/, "") + "/" + name);
  d.addEventListener("click", (e) => {
    const r = e.target.closest("[data-root]"), c = e.target.closest("[data-go]"), li = e.target.closest("#of-list li[data-dir]");
    const rec = e.target.closest("[data-of-recent]");
    if (r) list(r.dataset.root);
    else if (c) list(c.dataset.go);
    else if (rec) open(rec.dataset.ofRecent);
    else if (li) { $$("#of-list li", d).forEach((x) => x.setAttribute("aria-selected", String(x === li))); selected = li.dataset.dir; }
  });
  d.addEventListener("dblclick", (e) => { const li = e.target.closest("#of-list li[data-dir]"); if (li) list(child(li.dataset.dir)); });
  $("#of-list", d).addEventListener("keydown", (e) => {
    const items = $$("#of-list li[data-dir]", d);
    let i = items.findIndex((x) => x.getAttribute("aria-selected") === "true");
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      i = Math.max(0, Math.min(items.length - 1, i + (e.key === "ArrowDown" ? 1 : -1)));
      items.forEach((x, k) => x.setAttribute("aria-selected", String(k === i)));
      if (items[i]) { selected = items[i].dataset.dir; items[i].scrollIntoView({ block: "nearest" }); }
    } else if (e.key === "Enter" && selected) { e.preventDefault(); list(child(selected)); }
    else if (e.key === "Backspace" && listing && listing.parent) { e.preventDefault(); list(listing.parent); }
  });
  $("#of-path", d).addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); const v = e.target.value.trim(); if (v) list(v); } });
  const rec = recentFolders().filter((p) => p !== info.root);
  if (rec.length) {
    $("#of-recent-wrap", d).hidden = false;
    $("#of-recent", d).innerHTML = rec.map((p) => `<li><button class="link" data-of-recent="${H.esc(p)}">${H.esc(p)}</button></li>`).join("");
  }
  async function open(path) {
    showError("");
    if (!(await confirmDirty(H, "Opening another folder reloads this window."))) return;
    H.flushDrafts();
    let r;
    try { r = await H.api("open-folder", { body: { path } }); }
    catch (e) {
      if (e.status === 409) {
        H.toast("Cancel or wait for the running tasks first", "warning", { label: "Tasks", run: () => H.run("view.tasks") });
      } else showError(e.message);
      return;
    }
    rememberFolder(r.root);
    H.leave("./");
  }
  $("#of-open", d).addEventListener("click", () => {
    const typed = $("#of-path", d).value.trim();
    if (typed && (!listing || typed !== listing.path)) return open(typed);
    if (!listing) return;
    open(selected ? child(selected) : listing.path);
  });
  await list(info.root);
}

/* Another tab switched the folder: say so, then reload into it. */
export function folderChanged(H, data) {
  const name = (data && data.root ? data.root.split("/").filter(Boolean).pop() : "") || "another folder";
  H.flushDrafts();
  dialog({ id: "folder-changed", title: "Folder Changed", narrow: true,
    body: `<p>This window's folder changed to <code>${H.esc(name)}</code> (in another tab). It reloads now; your drafts are kept in this browser.</p>`,
    foot: `<button class="tb-btn-text primary" id="fc-reload">Reload</button>` });
  $("#fc-reload").addEventListener("click", () => H.leave("./"));
  setTimeout(() => H.leave("./"), 2500);
}

/* ---------------------------------------------------------------- Help ▸ Connecting from Your Computer */
const DOCS_CONNECT = "docs/gui.md#first-run-from-your-computer";
export async function tunnel(H) {
  await css();
  const info = H.info() || {};
  const node = info.node || info.hostname || "this machine";
  // paste-safe placeholders (a <gateway> is a redirect in bash and PowerShell)
  const connectCmd = info.connect || `mace gui connect --gateway USER@LOGIN_HOST --node ${node} --dir FOLDER`;
  const line = (id, text) => `<div class="cn-line"><pre class="mono" id="${id}">${H.esc(text)}</pre>` +
    `<button class="tb-btn-text" data-copy="${id}">${H.icon("copy")}<span>Copy</span></button></div>`;
  const body = info.vscode
    ? `<p>This server runs on <b>${H.esc(node)}</b> and shows <code class="root-path">${H.esc(info.root || "")}</code>.</p>
       <p id="cn-vscode">VS Code: the terminal that started <code>mace gui</code> forwards the port for you. Ctrl+click the URL it
       printed (VS Code rewrites it to the forwarded port). The Ports tab opens the bare address, which needs the token.</p>`
    : `<p>This server runs on <b>${H.esc(node)}</b> and shows <code class="root-path">${H.esc(info.root || "")}</code>.
         The tunnel must go to this exact node.</p>
       <p>From your computer, with MACE installed there (one step: it starts, forwards and stops the server):</p>
       ${line("cn-connect", connectCmd)}
       <p>Without MACE there (MobaXterm's local terminal, Git Bash, PowerShell): run this in the terminal of
         <code>mace gui</code>, copy the script to your computer and run it there (<code>bash mace-gui-connect.sh</code>;
         <code>--export-script powershell -o mace-gui-connect.ps1</code> for PowerShell):</p>
       ${line("cn-export", connectCmd + " --export-script bash -o mace-gui-connect.sh")}
       <p>Or with ssh alone (leave it running), then open the URL that <code>mace gui</code> printed:</p>
       ${line("cn-ssh", info.tunnel || `ssh -N -J USER@LOGIN_HOST -L PORT:127.0.0.1:PORT USER@${node}`)}
       ${/LOGIN_HOST/.test(connectCmd + (info.tunnel || "")) ? `<p class="insp-hint" id="cn-placeholder">Replace USER, LOGIN_HOST
         and FOLDER (your cluster's login host, e.g. from export MACE_GUI_GATEWAY on the node) before you run them.</p>` : ""}
       <p class="insp-hint" id="cn-hop">If your cluster's nodes accept logins only from the login node, use --hop nested.</p>`;
  const d = dialog({ id: "tunnel", title: "Connecting from Your Computer", narrow: true,
    body: body + `<p class="insp-hint">Every way to connect (Windows, MobaXterm, VS Code, jump hosts):
      <a id="cn-docs" href="${H.esc(H.docsUrl ? H.docsUrl("first-run-from-your-computer") : DOCS_CONNECT)}" target="_blank"
      rel="noopener noreferrer">${DOCS_CONNECT}</a>.</p>`,
    foot: `<button class="tb-btn-text" id="cn-stop">${H.icon("power")}<span>Stop Server…</span></button>` +
      `<span style="flex:1"></span><button class="tb-btn-text primary" data-close-dlg>Close</button>` });
  $("[data-close-dlg]", d).addEventListener("click", () => d.close());
  $$("[data-copy]", d).forEach((b) => b.addEventListener("click", () => H.copyText($("#" + b.dataset.copy, d).textContent, "Copied")));
  $("#cn-stop", d).addEventListener("click", () => { d.close(); H.run("file.signOut"); });
}

/* ---------------------------------------------------------------- File ▸ Sign Out and Stop Server… */
async function clearFolderData(H) {
  const hash = (H.info() || {}).dir_hash || "";
  try {
    for (const k of Object.keys(localStorage)) {
      if (k.startsWith(`mace.gui.draft.${hash}.`) || k === `mace.gui.recent.${hash}`) localStorage.removeItem(k);   // ls-key: mace.gui.draft.
    }
  } catch (e) { /* no storage */ }
  // the profile's Recent files of this folder and its Recent folders entry, saved before the stop
  const root = (H.info() || {}).root;
  profile.setRecentFiles([]);
  profile.setRecentFolders(profile.recentFolders().filter((p) => p !== root));
  await profile.flush();
}

export async function signOut(H) {
  await css();
  if (!(await confirmDirty(H, "Signing out stops MACE GUI."))) return;
  const info = H.info() || {};
  let running = [];
  try { running = ((await H.api("tasks")).tasks || []).filter((t) => t.state === "running" || t.state === "queued"); }
  catch (e) { /* the server is gone already: the dialog still explains */ }
  const node = info.node || info.hostname || "this machine";
  const titles = running.map((t) => H.esc(t.title)).join(", ");
  const submit = running.some((t) => /submit/i.test(t.kind || ""));
  const d = dialog({ id: "sign-out", title: "Sign Out and Stop Server", narrow: true, body: `
      <p>This signs you out and stops MACE GUI on <b>${H.esc(node)}</b>. To use it again, start <code>mace gui</code>
        again (or <code>mace gui connect</code>).</p>
      ${running.length ? `<p id="so-tasks">${running.length} task${running.length === 1 ? " is" : "s are"} running: ${titles}.
        Stopping cancels ${running.length === 1 ? "it" : "them"}.</p>` : ""}
      ${submit ? `<p>A submission is in progress; the server stops as soon as it has finished.</p>` : ""}
      ${info.slurm && info.slurm.job ? `<p id="so-job">This also ends the job ${H.esc(info.slurm.job)} that runs MACE GUI on ${H.esc(info.slurm.node || node)}.</p>` : ""}
      <p class="insp-hint">Your SLURM jobs keep running; only MACE GUI stops.</p>
      <label class="chk"><input type="checkbox" id="so-clear"> Also remove the drafts and recent files for this folder</label>`,
  foot: `<button class="tb-btn-text" data-r="cancel">Cancel</button>` +
    (running.length && !submit ? `<button class="tb-btn-text" data-r="after" id="so-after">Stop When Tasks Finish</button>` : "") +
    `<button class="tb-btn-text primary danger" data-r="now" id="so-now">Sign Out and Stop</button>` });
  $$("[data-r]", d).forEach((b) => b.addEventListener("click", async () => {
    const r = b.dataset.r;
    if (r === "cancel") { d.close(); return; }
    if ($("#so-clear", d).checked) await clearFolderData(H);
    if (r === "after") {
      d.close();
      try { await H.api("shutdown", { body: { mode: "after-tasks" }, withStatus: true }); }
      catch (e) { H.toast(e.message, "error"); }
      return;
    }
    H.stopping();
    d.close();
    try { await H.api("shutdown", { body: { mode: "now" }, withStatus: true }); } catch (e) { /* stopped already */ }
    stoppedPage(H, false);
  }));
}

/* The page after the stop: nothing else works any more. */
/* A path as one POSIX shell word, so the copied command works with spaces or quotes in it. */
export function shellWord(text) {
  const s = String(text);
  return /^[\w@%+=:,./~-]+$/.test(s) ? s : `'${s.replace(/'/g, `'"'"'`)}'`;
}
export function stoppedPage(H, elsewhere) {
  if (document.getElementById("stopped-page")) return;
  css();
  const info = H.info() || {};
  const cmd = `mace gui --dir ${shellWord(info.root || ".")}`;
  document.querySelectorAll("dialog[open]").forEach((x) => { try { x.close(); } catch (e) { /* */ } });
  const el = document.createElement("div");
  el.id = "stopped-page";
  el.className = "stopped-page";
  el.setAttribute("role", "alertdialog");
  el.setAttribute("aria-labelledby", "stopped-title");
  el.innerHTML = `<div class="card"><h1 id="stopped-title">MACE GUI has stopped</h1>
    <p id="stopped-text">${elsewhere === "idle" ? "It stopped after a long time without use and with no task running. "
      : elsewhere === "cpu-limit" ? "It stopped before this node's CPU-time limit per process ended it. "
      : elsewhere === "walltime" ? "It stopped because its SLURM job reached its time limit. "
      : elsewhere ? "It was stopped from another window. " : ""}You can close this tab. Your drafts stay in this browser.</p>
    <p>To start it again:</p><div class="cn-line"><pre class="mono" id="stopped-cmd">${H.esc(cmd)}</pre>
    <button class="tb-btn-text" id="stopped-copy">${H.icon("copy")}<span>Copy</span></button></div></div>`;
  document.body.appendChild(el);
  $("#stopped-copy", el).addEventListener("click", () => H.copyText(cmd, "Copied"));
  $("#stopped-copy", el).focus();
}

/* Stop When Tasks Finish: a banner in every tab until the stop or Cancel Stop. */
export function stopPending(H, data) {
  let b = document.getElementById("stop-banner");
  if (!data) { if (b) b.remove(); return; }
  css();
  if (!b) {
    b = document.createElement("div");
    b.id = "stop-banner";
    b.className = "banner info stop-banner";
    b.setAttribute("role", "status");
    const host = document.getElementById("conn-banner");
    (host && host.parentNode ? host.parentNode : document.body).insertBefore(b, host || null);
  }
  const n = (data.waiting_for || []).length;
  b.innerHTML = `${H.icon("power")}<span>MACE GUI stops when ${n === 1 ? "its task finishes" : `${n} tasks finish`}.</span>` +
    `<button class="tb-btn-text" id="stop-cancel">Cancel Stop</button>`;
  $("#stop-cancel", b).addEventListener("click", async () => {
    try { await H.api("shutdown", { body: { mode: "cancel" } }); } catch (e) { H.toast(e.message, "error"); }
  });
}

/* ---------------------------------------------------------------- Signed Out (session expiry) */
export async function signedOut(H, after) {
  await css();
  const info = H.info() || {};
  const hours = info.session && info.session.idle_hours;
  const d = dialog({ id: "signed-out", title: "Signed Out", narrow: true, body: `
      <p>You were signed out${hours ? ` after ${hours} hour${hours === 1 ? "" : "s"} without use` : ""}. Your edits are kept
        as drafts in this browser, and tasks keep running on the server.</p>
      <p>Paste the token that <code>mace gui</code> printed (or run <code>mace gui login-url</code>, or press Enter in
        <code>mace gui connect</code>) to continue.</p>
      <label class="of-paste">Token <input id="so-token" autocomplete="off" spellcheck="false" type="password"></label>
      <p class="of-error" id="so-err" role="alert" hidden></p>`,
  foot: `<button class="tb-btn-text primary" id="so-signin">Sign In</button>` });
  // no way around it but signing in (Esc would leave a page that cannot work)
  d.addEventListener("cancel", (e) => e.preventDefault());
  const go = async () => {
    const tok = $("#so-token", d).value.trim();
    const err = $("#so-err", d);
    if (!tok) { err.textContent = "Paste the token first."; err.hidden = false; return; }
    let r;
    try {
      r = await fetch("login", { method: "POST", credentials: "same-origin",
        headers: { "X-Requested-With": "mace-gui", "Content-Type": "application/json" }, body: JSON.stringify({ token: tok }) });
    } catch (e) { err.textContent = "The server did not answer."; err.hidden = false; return; }
    if (!r.ok) { err.textContent = r.status === 429 ? "Too many attempts at once; wait a moment." : "That token is not right."; err.hidden = false; return; }
    d.close();
    H.toast("Signed in again", "success");
    if (after) after();
  };
  $("#so-signin", d).addEventListener("click", go);
  $("#so-token", d).addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); go(); } });
}

/* Server idle stop: a sticky warning 5 minutes before; any input cancels it. */
export function idleWarning(H, data) {
  let b = document.getElementById("idle-banner");
  if (!data || data.cleared) { if (b) b.remove(); return; }
  css();
  if (!b) {
    b = document.createElement("div");
    b.id = "idle-banner";
    b.className = "banner";
    b.setAttribute("role", "alert");
    const host = document.getElementById("conn-banner");
    (host && host.parentNode ? host.parentNode : document.body).insertBefore(b, host || null);
  }
  const m = Math.max(1, Math.round((data.seconds || 300) / 60));
  const job = ((H.info() || {}).slurm || {}).job;
  b.innerHTML = `${H.icon("power")}<span>MACE GUI stops in ${m} minute${m === 1 ? "" : "s"} because nobody has used it${job ? ` (it runs as job ${H.esc(job)})` : ""}. Click anywhere to keep it.</span>`;
  const keep = () => { removeEventListener("pointerdown", keep, true); removeEventListener("keydown", keep, true); b.remove(); };
  addEventListener("pointerdown", keep, true);
  addEventListener("keydown", keep, true);
}

/* mace gui --slurm: the job's time limit is near (15 and 5 minutes before). */
export function walltimeWarning(H, data) {
  css();
  let b = document.getElementById("walltime-banner");
  if (!b) {
    b = document.createElement("div");
    b.id = "walltime-banner";
    b.className = "banner";
    b.setAttribute("role", "alert");
    const host = document.getElementById("conn-banner");
    (host && host.parentNode ? host.parentNode : document.body).insertBefore(b, host || null);
  }
  const m = Math.max(1, Math.round((data.left_s || 0) / 60));
  b.innerHTML = `${H.icon("clock")}<span>MACE GUI's SLURM job ${H.esc(data.job || "")} reaches its time limit in ${m} minute${m === 1 ? "" : "s"};
    MACE GUI stops cleanly 1 minute before. Save your work; start it again with a longer --time for more.</span>
    <button class="tb-btn-text" id="walltime-ok">OK</button>`;
  $("#walltime-ok", b).addEventListener("click", () => b.remove());
}

/* ---------------------------------------------------------------- Copy Diagnostics / Report a Problem */
function webglInfo() {
  const out = { webgl1: false, webgl2: false, renderer: "" };
  try {
    const c = document.createElement("canvas");
    const g2 = c.getContext("webgl2");
    out.webgl2 = !!g2;
    const g = g2 || document.createElement("canvas").getContext("webgl");
    out.webgl1 = !!(g2 || g);
    if (g) {
      // Firefox already gives the plain renderer; the debug extension only where it is offered without warnings
      const ext = /Firefox/.test(navigator.userAgent) ? null : g.getExtension("WEBGL_debug_renderer_info");
      out.renderer = String(ext ? g.getParameter(ext.UNMASKED_RENDERER_WEBGL) : g.getParameter(g.RENDERER));
    }
  } catch (e) { /* no WebGL */ }
  return out;
}
function clientPart(H) {
  const st = H.settings;
  const notes = (window.__maceNotifications || []).slice(-50).map((n) => (typeof n === "string" ? n : `${n.kind || ""} ${n.message || n.text || ""}`));
  return {
    "user agent": navigator.userAgent, language: navigator.language,
    screen: `${screen.width}x${screen.height} @${window.devicePixelRatio}`,
    "UI": `scale ${st.get("uiScale")}%, theme ${st.get("theme")}, density ${st.get("density")}`,
    "secure context": window.isSecureContext, WebGL: webglInfo(),
    errors: (window.__maceErrorLog || []).slice(-30).map((e) => `${e.at} ${e.message}`),
    "failed requests": (window.__maceFailedCalls || []).slice(-30),
    notifications: notes,
  };
}
export async function fetchDiagnostics(H) {
  return H.api("diagnostics", { body: { client: clientPart(H) } });
}
export async function copyDiagnostics(H) {
  // not prefetched: a ClipboardItem holding the promise keeps the click's permission
  const p = fetchDiagnostics(H).then((r) => r.text);
  if (window.ClipboardItem && navigator.clipboard && navigator.clipboard.write) {
    try {
      await navigator.clipboard.write([new ClipboardItem({ "text/plain": p.then((t) => new Blob([t], { type: "text/plain" })) })]);
      window.__maceLastCopy = await p;
      H.toast("Diagnostics copied", "success");
      return;
    } catch (e) { /* refused: the dialog below */ }
  }
  await reportProblem(H, await p, true);
}
export async function reportProblem(H, text = null, select = false) {
  await css();
  const d = dialog({ id: "report", title: "Report a Problem", wide: true, body: `
      <p>MACE GUI sends this nowhere: read and edit it, then share it. Paths, file and folder names, tokens, user
        and host names were removed.</p>
      <textarea id="diag-text" spellcheck="false" aria-label="Diagnostics">${H.esc(text || "Collecting…")}</textarea>
      <p class="insp-hint"><a id="diag-issue" href="${H.esc((H.repoUrl ? H.repoUrl() : "https://github.com/DjokicMa/MACE") + "/issues/new")}"
        target="_blank" rel="noopener noreferrer">Open an issue on GitHub</a> and paste it there with what you did and what you saw.</p>`,
  foot: `<button class="tb-btn-text" id="diag-save">${H.icon("download")}<span>Save as File</span></button>` +
    `<button class="tb-btn-text primary" id="diag-copy">${H.icon("copy")}<span>Copy</span></button>` });
  const ta = $("#diag-text", d);
  if (text === null) {
    try { ta.value = (await fetchDiagnostics(H)).text; } catch (e) { ta.value = `Could not collect the diagnostics: ${e.message}`; }
  }
  if (select) { ta.focus(); ta.select(); H.toast("Press Ctrl+C to copy.", "info"); }
  $("#diag-copy", d).addEventListener("click", () => H.copyText(ta.value, "Diagnostics copied"));
  $("#diag-save", d).addEventListener("click", () => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([ta.value], { type: "text/plain" }));
    a.download = "mace-gui-diagnostics.txt";
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  });
}

/* A per-process CPU-time limit (plan D18): a sticky warning at 80 % of it. */
export function cpuWarning(H, st) {
  if (document.getElementById("cpu-banner")) return;
  css();
  const m = (x) => Math.round(x / 60);
  const b = document.createElement("div");
  b.id = "cpu-banner";
  b.className = "banner";
  b.setAttribute("role", "alert");
  b.innerHTML = `${H.icon("power")}<span>MACE GUI has used ${m(st.used_s)} of this node's ${m(st.limit_s)} CPU minutes per process.
    Save your work; it stops cleanly at ${Math.floor(st.limit_s * 0.97 / 60)} minutes. Start it again for a fresh limit.</span>
    <button class="tb-btn-text" id="cpu-ok">OK</button>`;
  const host = document.getElementById("conn-banner");
  (host && host.parentNode ? host.parentNode : document.body).insertBefore(b, host || null);
  $("#cpu-ok", b).addEventListener("click", () => b.remove());
}
