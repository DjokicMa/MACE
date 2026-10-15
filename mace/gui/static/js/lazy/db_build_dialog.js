/* Create or Update Database from a Folder (PLAN_analysis C10). The server
 * runs the command shown at the bottom of the dialog: the preview is its
 * --dry-run --json, the build its --progress-json, a cancelable task that
 * holds the one GUI writer lock of the database. Loaded on first use. */
import { loadCss } from "./lazy_util.js";

const CHIPS = [["new", "New"], ["changed", "Changed"], ["unchanged", "Unchanged"],
  ["not tracked (always re-extracted)", "Not tracked"], ["unreadable", "Unreadable"], ["incomplete", "Incomplete"], ["skipped", "Skipped"]];
const ROW_H = 26, OVERSCAN = 8;
const fmtSize = (n) => (n >= 1048576 ? `${(n / 1048576).toFixed(1)} MB` : n >= 1024 ? `${(n / 1024).toFixed(1)} KB` : `${n} B`);
const dirOf = (p) => (p && p.includes("/") ? p.slice(0, p.lastIndexOf("/")) : "");

/** opts: {folder, db}: the folder to read (root-relative, "" = the served folder) and the database (optional). */
export async function open(H, opts = {}) {
  await loadCss(import.meta.url, "../../css/dialogs.css");
  H.overlayOpening();
  const S = H.shell(), opener = document.activeElement;
  let d = document.getElementById("db-build");
  if (!d) { d = document.createElement("dialog"); d.id = "db-build"; document.body.appendChild(d); }
  d.className = "dlg dbb-dlg";
  d.setAttribute("aria-labelledby", "dbb-title");
  d.innerHTML = `<div class="dlg-head"><h2 id="dbb-title">Create or Update Database</h2><button class="icon-btn sm" data-close aria-label="Close">${H.icon("x")}</button></div>
    <div class="dlg-body dlg-panel">
      <div class="dbb-grid">
        <label class="f">Folder<input class="field dbb-folder" spellcheck="false" aria-label="Folder"></label>
        <label class="f">Database<input class="field dbb-db" spellcheck="false" aria-label="Database"></label>
      </div>
      <p class="note dbb-state" aria-live="polite"></p>
      <p class="err dbb-outside" hidden></p>
      <fieldset class="dbb-opts"><legend>Options</legend>
        <label class="dbb-opt"><input type="checkbox" class="dbb-register" checked> Register calculations (recommended)</label>
        <label class="dbb-opt"><input type="checkbox" class="dbb-skip" checked> Only new and changed files <span class="note dbb-skip-why"></span></label>
        <label class="dbb-opt"><input type="checkbox" class="dbb-backup" checked> Back up the database first</label>
        <label class="dbb-opt dbb-keep-l">Keep the last <input type="number" class="field dbb-keep" min="1" max="100" value="3" aria-label="Keep the last N backups"> backups</label>
        <p class="note dbb-backups"></p>
      </fieldset>
      <section class="dbb-preview" aria-label="Preview">
        <div class="dbb-chips" role="group" aria-label="Show files"></div>
        <div class="dbb-table" role="grid" aria-label="Files" tabindex="0"><div class="dbb-space"></div><div class="dbb-rows" role="rowgroup"></div></div>
        <p class="note dbb-pnote" aria-live="polite">Reading the folder…</p>
      </section>
      <div class="dbb-cmdrow"><code class="dbb-cmd" aria-label="Equivalent command"></code><button class="tb-btn-text dbb-copy" type="button">${H.icon("copy")}<span>Copy</span></button></div>
      <p class="note dbb-guarantee"></p>
      <div class="dbb-run" hidden>
        <progress class="dbb-bar" max="1" value="0" aria-label="Progress"></progress>
        <span class="note dbb-now" aria-live="polite"></span>
        <button class="tb-btn-text dbb-cancel-run" type="button">Cancel</button>
      </div>
      <p class="err dbb-err" role="alert"></p>
    </div>
    <div class="dlg-foot"><button class="tb-btn-text" data-close>Cancel</button><button class="tb-btn-text primary dbb-go" disabled>Create Database</button></div>`;
  const $ = (s) => d.querySelector(s);
  const st = { info: null, preview: null, filter: null, gen: 0, task: null, running: false };
  $(".dbb-folder").value = opts.folder ?? "";
  $(".dbb-db").value = opts.db ?? "";
  d.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", () => d.close()));
  d.addEventListener("cancel", (e) => { if (st.running) e.preventDefault(); });
  d.addEventListener("close", () => { if (opener && opener.focus) opener.focus(); }, { once: true });

  const body = () => ({ folder: $(".dbb-folder").value.trim(), db: $(".dbb-db").value.trim() || null,
    register: $(".dbb-register").checked, skip_unchanged: $(".dbb-skip").checked && !$(".dbb-skip").disabled,
    backup: $(".dbb-backup").checked, keep: Number($(".dbb-keep").value) || 3 });
  const qs = (b) => new URLSearchParams({ folder: b.folder, ...(b.db ? { db: b.db } : {}), register: b.register ? "1" : "0",
    skip: b.skip_unchanged ? "1" : "0", backup: b.backup ? "1" : "0", keep: String(b.keep) });

  async function loadInfo() {
    const b = body();
    $(".dbb-err").textContent = "";
    try { st.info = await H.api("db/build/info?" + qs(b)); } catch (e) { st.info = null; $(".dbb-err").textContent = e.message; renderGo(); return false; }
    const i = st.info;
    if (!$(".dbb-db").value.trim()) $(".dbb-db").placeholder = i.db;
    $(".dbb-state").textContent = i.exists
      ? `Will update ${i.db}${i.property_rows !== null && i.property_rows !== undefined ? ` (${i.property_rows.toLocaleString("en-US")} property rows)` : ""}`
      : `Will create ${i.db}`;
    $(".dbb-outside").hidden = i.inside;
    $(".dbb-outside").textContent = i.outside_note || "";
    const bk = $(".dbb-backup");
    bk.checked = i.exists ? true : bk.checked;
    bk.disabled = i.exists;
    bk.parentElement.title = i.exists ? "A backup is always made before MACE writes an existing database" : "";
    $(".dbb-keep").disabled = !i.exists;
    const files = (i.backups && i.backups.files) || [];
    $(".dbb-backups").textContent = !i.exists ? "A new database: nothing to back up."
      : (files.length ? `Existing backups in ${dirOf(i.db) ? dirOf(i.db) + "/" : ""}backups/: ${files.map((f) => `${f.name} (${fmtSize(f.size)})`).join(", ")}.` : "No backups yet.") +
        (i.backups.prune.length ? ` This update prunes ${i.backups.prune.join(", ")}.` : "");
    const skip = $(".dbb-skip");
    const untracked = !b.register && !i.tracked;
    skip.disabled = untracked;
    $(".dbb-skip-why").textContent = untracked ? "(needs registered calculations: tick Register)" : "";
    $(".dbb-cmd").textContent = i.command;
    $(".dbb-guarantee").textContent = i.guarantee;
    $("#dbb-title").textContent = i.exists ? "Update Database" : "Create Database";
    if (i.writer) $(".dbb-err").textContent = `${i.writer} is writing this database; wait for it to finish`;
    return true;
  }
  async function preview() {
    const my = ++st.gen;
    st.preview = null;
    renderPreview();
    if (!st.info || !st.info.inside) { $(".dbb-pnote").textContent = st.info ? "" : "Choose a folder"; renderGo(); return; }
    $(".dbb-pnote").textContent = "Reading the folder…";
    try {
      const r = await S.apiTask("db/build/preview", { ...body(), wait: 2 });
      if (my !== st.gen) return;
      st.preview = r;
    } catch (e) {
      if (my !== st.gen) return;
      $(".dbb-pnote").textContent = e.message === "Canceled" ? "" : e.message;
    }
    renderPreview();
  }
  function toRun() {
    if (!st.preview) return 0;
    const c = st.preview.counts, skip = body().skip_unchanged;
    return Object.entries(c).reduce((t, [k, v]) => t + (k === "unreadable" || k === "skipped" || (skip && k === "unchanged") ? 0 : v), 0);
  }
  function renderGo() {
    const go = $(".dbb-go"), n = toRun(), i = st.info;
    go.textContent = i && i.exists ? `Update Database (${n} file${n === 1 ? "" : "s"})` : "Create Database";
    go.disabled = st.running || !i || !i.inside || !!i.writer || !st.preview || n === 0;
  }
  function rowsShown() { return st.preview ? st.preview.files.filter((f) => !st.filter || f.class === st.filter) : []; }
  function renderPreview() {
    const p = st.preview;
    $(".dbb-chips").innerHTML = CHIPS.map(([k, label]) => `<button type="button" class="dbb-chip" data-class="${H.esc(k)}" aria-pressed="${st.filter === k}"${p && p.counts[k] ? "" : " disabled"}>${H.esc(label)} <b>${p ? p.counts[k] || 0 : "–"}</b></button>`).join("");
    if (p) $(".dbb-pnote").textContent = `${p.files.length} output${p.files.length === 1 ? "" : "s"} in the folder${st.filter ? `; showing ${CHIPS.find((c) => c[0] === st.filter)[1].toLowerCase()}` : ""}.`;
    const list = rowsShown();
    $(".dbb-space").style.height = `${list.length * ROW_H}px`;
    $(".dbb-table").setAttribute("aria-rowcount", String(list.length));
    paintRows();
    renderGo();
  }
  function paintRows() {
    const host = $(".dbb-table"), list = rowsShown();
    const first = Math.max(0, Math.floor(host.scrollTop / ROW_H) - OVERSCAN);
    const last = Math.min(list.length, Math.ceil((host.scrollTop + host.clientHeight) / ROW_H) + OVERSCAN);
    $(".dbb-rows").style.transform = `translateY(${first * ROW_H}px)`;
    $(".dbb-rows").innerHTML = list.slice(first, last).map((f, k) => `<div class="dbb-row" role="row" aria-rowindex="${first + k + 1}">
      <span role="gridcell" class="dbb-name" title="${H.esc(f.rel)}">${H.esc(f.rel)}</span><span role="gridcell" class="dbb-cls">${H.esc(f.class)}</span>
      <span role="gridcell">${H.esc(f.material_id || "–")}</span><span role="gridcell">${H.esc(f.calc_type || "–")}</span></div>`).join("");
  }
  $(".dbb-table").addEventListener("scroll", () => requestAnimationFrame(paintRows));
  $(".dbb-chips").addEventListener("click", (e) => {
    const b = e.target.closest("[data-class]");
    if (!b) return;
    st.filter = st.filter === b.dataset.class ? null : b.dataset.class;
    $(".dbb-table").scrollTop = 0;
    renderPreview();
  });
  $(".dbb-copy").addEventListener("click", () => document.dispatchEvent(new CustomEvent("mace-copy", { detail: { text: $(".dbb-cmd").textContent, done: "Command copied" } })));
  let timer = 0;
  const changed = (full) => { clearTimeout(timer); timer = setTimeout(async () => { if (await loadInfo() && full) preview(); else renderGo(); }, 250); };
  $(".dbb-folder").addEventListener("input", () => changed(true));
  $(".dbb-db").addEventListener("input", () => changed(true));
  $(".dbb-register").addEventListener("change", () => changed(true));
  $(".dbb-skip").addEventListener("change", () => changed(false));
  $(".dbb-keep").addEventListener("change", () => changed(false));
  $(".dbb-backup").addEventListener("change", () => changed(false));

  $(".dbb-go").addEventListener("click", async () => {
    const b = body(), i = st.info;
    st.running = true;
    renderGo();
    d.querySelectorAll(".dbb-grid input, .dbb-opts input").forEach((x) => { x.dataset.was = x.disabled ? "1" : ""; x.disabled = true; });
    $(".dbb-run").hidden = false;
    $(".dbb-err").textContent = "";
    const t0 = Date.now();
    let started = false;
    const cancel = $(".dbb-cancel-run");
    cancel.disabled = false;
    cancel.onclick = () => { if (st.task) { cancel.disabled = true; S.cancelTask(st.task); } };
    try {
      const res = await S.apiTask("db/build", { ...b, confirm: true, root_gen: i.root_gen, wait: 0 }, (t) => {
        st.task = t.id;
        if (!started) { started = true; if (opts.onStart) opts.onStart(); }
        $(".dbb-bar").value = t.progress || 0;
        $(".dbb-now").textContent = `${t.message && t.message !== "done" ? t.message : "Starting…"} · ${Math.round((Date.now() - t0) / 1000)} s`;
      });
      const dbRel = i.db;
      d.close();
      // a new database is in the Explorer's list (and so can open) once the list is read again
      const listed = window.MACE && window.MACE.loadFiles ? window.MACE.loadFiles().catch(() => {}) : Promise.resolve();
      H.toast(res.message, "success", { label: "Open Database", run: () => listed.then(() => H.openPath(dbRel)) });
      for (const f of res.failed || []) {
        S.problem({ severity: "warning", source: "Database build", message: `${f}: no properties were extracted`, action: { label: "Show File", run: () => H.openPath(f) } });
      }
      if (opts.afterBuild) opts.afterBuild(res);
    } catch (e) {
      st.running = false;
      $(".dbb-run").hidden = true;
      d.querySelectorAll(".dbb-grid input, .dbb-opts input").forEach((x) => { x.disabled = x.dataset.was === "1"; });
      $(".dbb-err").textContent = e.state === "canceled" ? `Canceled. ${e.task && e.task.log ? e.task.log : ""}` : e.message;
      if (window.MACE && window.MACE.loadFiles) window.MACE.loadFiles().catch(() => {});
      await loadInfo();
      renderGo();
      if (opts.afterBuild) opts.afterBuild(null);
    }
  });

  d.showModal();
  if (await loadInfo()) preview();
  $(".dbb-folder").focus();
  return d;
}
