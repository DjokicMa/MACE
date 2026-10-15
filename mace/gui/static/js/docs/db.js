/* The Database document (spec §1.2, §8.4.7; plan B6): materials.db read through read-only connections
 * (Update from Folder… writes it through the CLI, PLAN_analysis C10),
 * paged on the server, a virtual table here. The table is one grid (one Tab
 * stop, a roving active row tracked by material_id, aria-rowcount/rowindex);
 * the KPI strip and the filters sit above it; the selected material's detail
 * is in the inspector. At most ~80 row nodes are in the page, whatever the
 * size of the database; every request is one page (≤200 rows, ≤30 KB). */
import * as profile from "../profile.js";
const PAGE = 200, OVERSCAN = 10, MAX_PAGES = 24, DEBOUNCE = 60;
const COLS = [
  // w: the share of spare width (fr) above min px; a column the user resized keeps its px
  { key: "material_id", label: "Material", w: 3, min: 88 }, { key: "formula", label: "Formula", w: 2, min: 88 },
  { key: "space_group", label: "SG", w: 1, min: 44, num: true }, { key: "dimensionality", label: "Dim.", w: 1.4, min: 80 },
  { key: "n_calcs", label: "Calcs", w: 1, min: 50, num: true }, { key: "n_completed", label: "Done", w: 1, min: 50, num: true },
  { key: "status", label: "Status", w: 1.4, min: 88 }, { key: "updated_at", label: "Updated", w: 2, min: 132 },
];
const DIMS = ["CRYSTAL", "SLAB", "POLYMER", "MOLECULE"];
const UNITS = { angstrom: "Å", "angstrom^2": "Å²", "angstrom^3": "Å³", degree: "°", degrees: "°", hartree: "Ha", ev: "eV",
  "ev/atom": "eV/atom", "1/angstrom": "Å⁻¹", gpa: "GPa" };

export function createDbView(host, detailHost, H) {
  const { api, q, esc, icon, toast } = H;
  const widths = { ...(profile.ui("dbWidths", {}) || {}) };     // column widths, in the profile
  let db = null;                     // {path, stats, props, total, sort, dir, filters, pages: Map, active (row index), selected (id)}
  let frame = 0, filterTimer = 0, gen = 0;
  const stats = { requests: 0, maxRows: 0, firstPageMs: null };

  host.innerHTML = `<div class="dbv">
    <div class="dbv-bar">
      <span class="dbv-live muted small" role="status" aria-live="polite"></span>
      <span class="dbv-bar-acts">
        <button class="tb-btn-text dbv-refresh" data-act="refresh" title="Read the database again">${icon("refresh-cw")}<span>Refresh</span></button>
        <button class="tb-btn-text dbv-build" data-act="build" title="Read a folder's outputs into this database (a backup is made first)">${icon("database")}<span>Update from Folder…</span></button>
      </span>
    </div>
    <div class="dbv-kpi" aria-label="Summary"></div>
    <div class="dbv-filters" role="search">
      <input class="dbv-q" type="search" placeholder="Filter by material or formula" aria-label="Filter by material or formula">
      <div class="dbv-dims" role="group" aria-label="Dimensionality">${DIMS.map((d) => `<button aria-pressed="false" data-dim="${d}">${d[0]}${d.slice(1).toLowerCase()}</button>`).join("")}</div>
      <select class="dbv-type" aria-label="Has a calculation of type"><option value="">Any calculation</option></select>
      <select class="dbv-status" aria-label="Has a calculation with status"><option value="">Any status</option></select>
      <span class="dbv-sg"><span class="dbv-sgl">SG</span><input type="number" min="1" max="230" placeholder="1" aria-label="Space group from"><span>–</span><input type="number" min="1" max="230" placeholder="230" aria-label="Space group to"></span>
      <select class="dbv-prop" aria-label="Has property"><option value="">Any property</option></select>
      <button class="tb-btn-text dbv-clear" hidden>Clear filters</button>
      <span class="dbv-count muted small" aria-live="polite"></span>
    </div>
    <div class="dbv-note" hidden></div>
    <div class="dbv-table" role="grid" aria-label="Materials" tabindex="0">
      <div class="dbv-headwrap"><div class="dbv-head" role="row" aria-rowindex="1"></div></div>
      <div class="dbv-scroll"><div class="dbv-space"></div><div class="dbv-rows" role="rowgroup"></div></div>
      <div class="dbv-empty" hidden></div>
    </div></div>`;
  const $ = (s) => host.querySelector(s);
  const grid = $(".dbv-table"), head = $(".dbv-head"), scroller = $(".dbv-scroll"), space = $(".dbv-space"), rowsEl = $(".dbv-rows");
  const rowH = () => (document.documentElement.dataset.density === "compact" ? 24 : 28);

  /* The columns share the table's width (so 1280 px needs no sideways scroll); below the
   * sum of their minimums, or after a resize, the table scrolls sideways. */
  function colW(c) { return widths[c.key] || c.min; }
  function template() { return COLS.map((c) => (widths[c.key] ? `${widths[c.key]}px` : `minmax(${c.min}px, ${c.w}fr)`)).join(" ") + " minmax(0, 0.01fr)"; }
  const tableW = () => COLS.reduce((t, c) => t + colW(c), 0);

  function renderHead() {
    head.style.gridTemplateColumns = template();
    head.style.width = `max(100%, ${tableW()}px)`;
    rowsEl.style.width = `max(100%, ${tableW()}px)`;
    head.innerHTML = COLS.map((c, k) => {
      const sorted = db && db.sort === c.key;
      return `<div class="dbv-th${c.num ? " num" : ""}" role="columnheader" aria-sort="${sorted ? (db.dir === "asc" ? "ascending" : "descending") : "none"}" aria-colindex="${k + 1}">` +
        `<button data-sort="${c.key}" tabindex="-1">${esc(c.label)}${sorted ? icon(db.dir === "asc" ? "chevron-up" : "chevron-down", "i so") : ""}</button>` +
        `<span class="dbv-grip" data-grip="${c.key}" aria-hidden="true"></span></div>`;
    }).join("") + `<div class="dbv-th" role="presentation"></div>`;
  }

  function renderKpi() {
    const s = db.stats, chip = (k, v, cls = "") => `<div class="dbv-k ${cls}"><div class="k">${esc(k)}</div><div class="v">${v}</div></div>`;
    const nCalc = Object.values(s.calc_status || {}).reduce((t, v) => t + v, 0);
    $(".dbv-kpi").innerHTML = chip("Materials", s.n_materials.toLocaleString("en-US")) +
      chip("Properties", (s.n_properties || 0).toLocaleString("en-US")) +
      `<div class="dbv-kg"><span class="dbv-kgl">${nCalc.toLocaleString("en-US")} calculations</span><div class="dbv-kgc">` +
      Object.entries(s.calc_status || {}).map(([k, v]) => chip(k, v.toLocaleString("en-US"), k)).join("") +
      Object.entries(s.calc_types || {}).slice(0, 8).map(([k, v]) => chip(k, v.toLocaleString("en-US"), "type")).join("") + "</div></div>";
    const fill = (sel, keys) => { const el = $(sel); const first = el.options[0].outerHTML; el.innerHTML = first + keys.map((k) => `<option>${esc(k)}</option>`).join(""); };
    fill(".dbv-type", Object.keys(s.calc_types || {}));
    fill(".dbv-status", Object.keys(s.calc_status || {}));
    const note = $(".dbv-note");
    note.hidden = !s.readonly_snapshot;
    note.innerHTML = s.readonly_snapshot ? `${icon("info")}<span>Read-only snapshot: this folder is not writable, so MACE reads the database file as it is. ` +
      `Results recorded by running jobs appear after Refresh.</span>` : "";
  }

  /* ------------------------------------------------ busy answers and live refresh (PLAN_analysis C17)
   * A 503 {busy: true} is not an error: the page keeps what it shows, says so in the status line and
   * retries after 2, 4, 8 s and then every 30 s. While the view is visible the file is polled (stat
   * only) every database.refreshSeconds; on a change the KPI strip and the rows in view reload with
   * the sort, filters, scroll and selection kept. */
  const BUSY = { busy: "Database busy (a job is recording its result) - retrying",
    interrupted: "A job was interrupted while writing the database; the next MACE command that writes it repairs it - retrying",
    building: "Updating… (build in progress)" };
  const live = { timer: 0, stamp: null, updatedAt: 0, delta: 0, busy: null, tick: 0 };
  const isBusy = (e) => e && e.status === 503 && e.data && e.data.busy;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  function liveText() {
    const el = $(".dbv-live");
    if (!el) return;
    if (live.busy) { el.textContent = BUSY[live.busy] || BUSY.busy; el.classList.add("busy"); return; }
    el.classList.remove("busy");
    if (!live.updatedAt) { el.textContent = ""; return; }
    const s = Math.max(0, Math.round((Date.now() - live.updatedAt) / 1000));
    const ago = s < 5 ? "just now" : s < 90 ? `${s} s ago` : `${Math.round(s / 60)} min ago`;
    el.textContent = `Updated ${ago}` + (live.delta > 0 ? ` · ${live.delta} new calculation${live.delta === 1 ? "" : "s"}` : "");
  }
  /** fn() until it is not answered "busy"; the status line says so meanwhile. */
  async function untilFree(fn, my) {
    const waits = [2000, 4000, 8000];
    for (let k = 0; ; k++) {
      try {
        const r = await fn();
        if (live.busy) { live.busy = null; liveText(); }
        return r;
      } catch (e) {
        if (!isBusy(e)) throw e;
        live.busy = e.data.reason || "busy";
        liveText();
        await sleep(k < waits.length ? waits[k] : 30000);
        if (my !== undefined && my !== gen) throw Object.assign(new Error("superseded"), { superseded: true });
      }
    }
  }
  const visible = () => db && host.offsetParent !== null && !document.hidden;
  function schedule(ms = null) {
    clearTimeout(live.timer);
    const sec = Number(H.setting ? H.setting("database.refreshSeconds") : 30);
    if (!db || (!sec && ms === null)) return;
    live.timer = setTimeout(checkStamp, ms ?? sec * 1000);
  }
  /* A GUI build of this database (PLAN_analysis C10.4): the view says "Updating…", reads nothing of
   * the file meanwhile (the server answers busy "building"), and reloads once the build ends. */
  const BUILD_POLL_MS = 3000;
  async function checkStamp() {
    if (!db) return;
    if (!visible()) { schedule(); return; }
    try {
      const r = await api("db/stamp?" + q({ path: db.path }));
      if (r.building) {
        if (live.busy !== "building") { live.busy = "building"; liveText(); }
        schedule(BUILD_POLL_MS);
        return;
      }
      if (live.busy === "building") { live.busy = null; live.stamp = r.stamp; await refreshData(); }
      else if (live.stamp !== null && r.stamp !== live.stamp) await refreshData();
      live.stamp = r.stamp;
    } catch (e) { /* the file went away: the next poll says so */ }
    schedule();
  }
  /** Reload the KPIs and the rows in view; sort, filters, scroll and selection stay. */
  async function refreshData() {
    if (!db) return;
    const before = db.stats ? (db.stats.n_calculations ?? Object.values(db.stats.calc_status || {}).reduce((t, v) => t + v, 0)) : 0;
    const my = ++gen;
    let s;
    try { s = await untilFree(() => api("db/stats?" + q({ path: db.path })), my); } catch (e) { if (!e.superseded) toast(e.message, "error"); return; }
    if (my !== gen) return;
    db.stats = s;
    const after = s.n_calculations ?? Object.values(s.calc_status || {}).reduce((t, v) => t + v, 0);
    live.delta = Math.max(0, after - before);
    live.updatedAt = Date.now();
    renderKpi();
    db.pages = new Map();
    const top = scroller.scrollTop;
    await untilFree(() => pageOf(0, true).loading, my).catch(() => {});
    scroller.scrollTop = top;
    paint(); count(); size();
    liveText();
    try { live.stamp = (await api("db/stamp?" + q({ path: db.path }))).stamp; } catch (e) { /* next poll */ }
  }
  setInterval(() => { if (db && live.updatedAt && visible()) liveText(); }, 5000);
  document.addEventListener("visibilitychange", () => { if (!document.hidden && db) checkStamp(); });
  host.addEventListener("click", (e) => {
    if (e.target.closest("[data-act=refresh]")) refreshData();
    if (e.target.closest("[data-act=build]") && db && H.dbBuild) {
      H.dbBuild({ db: db.path });
    }
  });

  function filterQuery() {
    const f = db.filters, o = { path: db.path, sort: db.sort, dir: db.dir };
    if (f.q) o.q = f.q;
    if (f.dims.length) o.dim = f.dims.join(",");
    for (const k of ["calc_type", "calc_status", "sg_min", "sg_max", "prop"]) if (f[k] !== "" && f[k] !== null && f[k] !== undefined) o[k] = f[k];
    return o;
  }
  const active = () => db && (db.filters.q || db.filters.dims.length || db.filters.calc_type || db.filters.calc_status || db.filters.sg_min || db.filters.sg_max || db.filters.prop);

  /* pages: rows [k*PAGE, (k+1)*PAGE), fetched by the previous page's cursor when it is here, else by index.
   * While the table scrolls fast nothing is fetched; the pages in view load 80 ms after it settles. */
  let lastScroll = 0, settleTimer = 0;
  function peek(k) { const p = db.pages.get(k); return p && p.rows ? p : null; }
  function want(ks) {
    const now = performance.now();
    if (now - lastScroll < 80) { clearTimeout(settleTimer); settleTimer = setTimeout(() => want(ks), 90); return; }
    for (const k of ks) pageOf(k);
  }
  function pageOf(k, strict = false) {
    let p = db.pages.get(k);
    if (p) { db.pages.delete(k); db.pages.set(k, p); return p; }
    p = { rows: null, loading: null, next: null };
    db.pages.set(k, p);
    while (db.pages.size > MAX_PAGES) db.pages.delete(db.pages.keys().next().value);
    const prev = db.pages.get(k - 1);
    const o = filterQuery();
    if (k > 0 && prev && prev.rows && prev.next) o.after = prev.next; else if (k > 0) o.index = k * PAGE;
    const my = gen;
    const t0 = performance.now();
    stats.requests++;
    p.loading = api("db/page?" + q(o)).then((r) => {
      if (my !== gen) return;
      p.rows = r.rows; p.next = r.next;
      if (db.total !== r.total) { db.total = r.total; size(); count(); }
      if (k === 0 && stats.firstPageMs === null) stats.firstPageMs = performance.now() - t0;
      if (k === 0) performance.mark("db:rows:rendered");
    }).catch((e) => {
      if (isBusy(e)) {
        // busy: keep nothing for this page (it is asked for again) and say so, no error toast
        db.pages.delete(k);
        live.busy = e.data.reason || "busy"; liveText();
        if (strict) throw e;
        setTimeout(() => { if (my === gen && db) { live.busy = null; paint(); } }, 2000);
        return;
      }
      p.rows = []; toast(e.message, "error");
    }).finally(() => { p.loading = null; if (my === gen) paint(); });
    return p;
  }
  function rowAt(i) {
    const p = peek(Math.floor(i / PAGE));
    return p ? p.rows[i % PAGE] || null : null;
  }
  function size() { space.style.height = `${Math.max(1, db.total || 0) * rowH()}px`; grid.setAttribute("aria-rowcount", String((db.total || 0) + 1)); }
  function count() {
    $(".dbv-count").textContent = db.total === null ? "" : `${db.total.toLocaleString("en-US")} material${db.total === 1 ? "" : "s"}`;
    $(".dbv-clear").hidden = !active();
    const empty = $(".dbv-empty");
    if (db.total === 0) {
      empty.hidden = false;
      empty.innerHTML = active() ? `<p>No materials match${db.filters.q ? ` “${esc(db.filters.q)}”` : " these filters"}.</p><button class="tb-btn-text" data-act="clear">Clear filters</button>`
        : `<p>This database has no materials yet.</p>`;
    } else empty.hidden = true;
  }

  function fmtFormula(m) { return m.reduced_formula ? `${esc(m.reduced_formula)}${m.z > 1 ? ` <span class="muted small">Z = ${m.z}</span>` : ""}` : esc(m.formula || ""); }
  function cell(c, m) {
    if (!m) return `<div class="dbv-td${c.num ? " num" : ""}" role="gridcell"><span class="dbv-wait"></span></div>`;
    let v = m[c.key];
    let html = c.key === "formula" ? fmtFormula(m) : c.key === "updated_at" ? esc(String(v || "").replace("T", " ").slice(0, 16))
      : c.key === "status" ? `<span class="status ${esc(v || "")}">${esc(v || "")}</span>` : esc(v ?? "—");
    return `<div class="dbv-td${c.num ? " num" : ""}" role="gridcell" title="${esc(c.key === "formula" ? `cell formula ${m.formula}` : String(v ?? ""))}">${html}</div>`;
  }
  function paint() {
    frame = 0;
    if (!db) return;
    const h = rowH(), top = scroller.scrollTop, vis = Math.ceil(scroller.clientHeight / h);
    const first = Math.max(0, Math.floor(top / h) - OVERSCAN), last = Math.min(db.total || 0, first + vis + OVERSCAN * 2);
    let html = "";
    for (let i = first; i < last; i++) {
      const m = rowAt(i);
      const act = i === db.active, sel = m && m.material_id === db.selected;
      html += `<div class="dbv-tr${act ? " active" : ""}${sel ? " sel" : ""}" role="row" aria-rowindex="${i + 2}" aria-selected="${!!sel}" tabindex="${act ? 0 : -1}"` +
        ` data-i="${i}"${m ? ` data-id="${esc(m.material_id)}"` : ""} style="top:${i * h}px;height:${h}px;grid-template-columns:${template()}">` +
        COLS.map((c) => cell(c, m)).join("") + "</div>";
    }
    const need = [];
    for (let k = Math.floor(first / PAGE); k <= Math.floor(Math.max(first, last - 1) / PAGE); k++) if (!peek(k)) need.push(k);
    if (need.length) want(need);
    const hadFocus = grid.contains(document.activeElement) && document.activeElement.classList.contains("dbv-tr");
    rowsEl.innerHTML = html;
    stats.maxRows = Math.max(stats.maxRows, rowsEl.childElementCount);
    if (hadFocus) focusActive();
  }
  scroller.addEventListener("scroll", () => {
    head.style.transform = `translateX(${-scroller.scrollLeft}px)`;      // the header follows a sideways scroll
    lastScroll = performance.now(); if (!frame) frame = requestAnimationFrame(paint);
  }, { passive: true });
  new ResizeObserver(() => { if (!frame) frame = requestAnimationFrame(paint); }).observe(scroller);

  function focusActive() {
    const r = rowsEl.querySelector(`.dbv-tr[data-i="${db.active}"]`);
    if (r) r.focus({ preventScroll: true }); else grid.focus({ preventScroll: true });
  }
  function moveTo(i, focus = true) {
    if (!db || !db.total) return;
    db.active = Math.max(0, Math.min(db.total - 1, i));
    const h = rowH(), top = db.active * h;
    if (top < scroller.scrollTop) scroller.scrollTop = top;
    else if (top + h > scroller.scrollTop + scroller.clientHeight) scroller.scrollTop = top + h - scroller.clientHeight;
    paint();
    if (focus) focusActive();
    const m = rowAt(db.active);
    db.activeId = m ? m.material_id : null;
  }
  grid.addEventListener("keydown", (e) => {
    if (!db) return;
    const inHead = e.target.closest && e.target.closest(".dbv-head");
    if (inHead) {
      const btns = [...head.querySelectorAll("[data-sort]")], k = btns.indexOf(e.target);
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") { e.preventDefault(); const n = btns[Math.max(0, Math.min(btns.length - 1, k + (e.key === "ArrowRight" ? 1 : -1)))]; n.focus(); }
      else if (e.key === "ArrowDown") { e.preventDefault(); moveTo(db.active || 0); }
      return;
    }
    const page = Math.max(1, Math.floor(scroller.clientHeight / rowH()) - 1);
    const to = { ArrowDown: db.active + 1, ArrowUp: db.active - 1, PageDown: db.active + page, PageUp: db.active - page,
      Home: 0, End: (db.total || 1) - 1 }[e.key];
    if (to !== undefined) { e.preventDefault(); if (e.key === "ArrowUp" && db.active === 0) { head.querySelector("[data-sort]").focus(); return; } moveTo(to); return; }
    if (e.key === "Enter") {
      e.preventDefault();
      const m = rowAt(db.active);
      if (m) select(m.material_id);
      else { const p = pageOf(Math.floor(db.active / PAGE)); (p.loading || Promise.resolve()).then(() => { const mm = rowAt(db.active); if (mm) select(mm.material_id); }); }
    }
  });
  grid.addEventListener("focus", (e) => { if (e.target === grid && db && db.total) moveTo(db.active || 0); });
  rowsEl.addEventListener("click", (e) => {
    const r = e.target.closest(".dbv-tr");
    if (!r) return;
    db.active = Number(r.dataset.i);
    if (r.dataset.id) select(r.dataset.id);
    paint(); focusActive();
  });
  rowsEl.addEventListener("dblclick", (e) => { const r = e.target.closest(".dbv-tr"); if (r && r.dataset.id) openStructure(r.dataset.id); });

  head.addEventListener("click", (e) => {
    const b = e.target.closest("[data-sort]");
    if (!b || e.target.closest("[data-grip]")) return;
    const k = b.dataset.sort;
    if (db.sort === k) db.dir = db.dir === "asc" ? "desc" : "asc"; else { db.sort = k; db.dir = "asc"; }
    reload();
    renderHead();
    const nb = head.querySelector(`[data-sort="${k}"]`);
    if (nb) nb.focus();
  });
  head.addEventListener("pointerdown", (e) => {
    const g = e.target.closest("[data-grip]");
    if (!g) return;
    e.preventDefault();
    // the column widths are the table's own px; the pointer's travel is screen px (the UI scale zooms the table)
    const z = window.maceZoom.of(g), c = COLS.find((x) => x.key === g.dataset.grip), x0 = e.clientX;
    const w0 = g.parentElement.getBoundingClientRect().width / z;
    g.setPointerCapture(e.pointerId);
    const move = (ev) => { widths[c.key] = Math.max(48, Math.min(600, w0 + (ev.clientX - x0) / z)); renderHead(); paint(); };
    const up = () => { g.removeEventListener("pointermove", move); profile.setUi("dbWidths", { ...widths }); };
    g.addEventListener("pointermove", move);
    g.addEventListener("pointerup", up, { once: true });
  });

  /* filters */
  function readFilters() {
    const f = db.filters, sg = host.querySelectorAll(".dbv-sg input");
    f.q = $(".dbv-q").value.trim();
    f.dims = [...host.querySelectorAll(".dbv-dims [aria-pressed=true]")].map((b) => b.dataset.dim);
    f.calc_type = $(".dbv-type").value; f.calc_status = $(".dbv-status").value; f.prop = $(".dbv-prop").value;
    f.sg_min = sg[0].value; f.sg_max = sg[1].value;
  }
  function onFilter() { clearTimeout(filterTimer); filterTimer = setTimeout(() => { readFilters(); performance.mark("db:filter:start"); reload(); }, DEBOUNCE); }
  $(".dbv-q").addEventListener("input", onFilter);
  host.querySelectorAll(".dbv-sg input, .dbv-type, .dbv-status, .dbv-prop").forEach((el) => el.addEventListener("change", onFilter));
  $(".dbv-dims").addEventListener("click", (e) => { const b = e.target.closest("[data-dim]"); if (b) { b.setAttribute("aria-pressed", String(b.getAttribute("aria-pressed") !== "true")); onFilter(); } });
  host.addEventListener("click", (e) => { if (e.target.closest(".dbv-clear, [data-act=clear]")) clearFilters(); });
  function clearFilters() {
    $(".dbv-q").value = ""; host.querySelectorAll(".dbv-sg input").forEach((x) => { x.value = ""; });
    host.querySelectorAll(".dbv-dims button").forEach((b) => b.setAttribute("aria-pressed", "false"));
    for (const s of [".dbv-type", ".dbv-status", ".dbv-prop"]) $(s).value = "";
    readFilters(); reload();
  }

  function reload() {
    gen++;
    db.pages = new Map();
    db.total = db.total ?? null;
    db.active = 0;
    scroller.scrollTop = 0;
    const my = gen;
    pageOf(0).loading.then(() => { if (my === gen) { performance.mark("db:filter:end"); count(); } });
    paint();
  }

  /* the selected material, in the inspector */
  async function select(id) {
    db.selected = id;
    paint();
    detailHost.innerHTML = `<p class="insp-hint"><span class="spinner"></span> Reading ${esc(id)}…</p>`;
    try {
      const r = await api("db/material?" + q({ path: db.path, id }));
      db.detail = r;
      renderDetail(r);
      H.selected && H.selected(r);
    } catch (e) { detailHost.innerHTML = `<p class="insp-hint">${esc(e.message)}</p>`; }
  }
  /* Open Structure: the newest completed run's files in this folder, else the newest run's
   * (a failed or unfinished OPT still holds the structure, as its Open output link shows) */
  function structureCalc(calcs) {
    const has = (c) => c.output_rel || c.input_rel;
    const done = calcs.filter((c) => c.status === "completed" && has(c));
    return done.length ? done[done.length - 1] : [...calcs].reverse().find(has) || null;
  }
  /* A long JSON value (a band structure, a DOS, a mode list) is shown cut, with Copy JSON:
   * a property never puts kilobytes of text into the inspector (PLAN_analysis C11.3). */
  const LONG = 2000, CUT = 160;
  let longValues = [];
  /* what a property's name cannot say, shown on hover */
  const ZT_E = "ZTₑ (electrons only, no lattice κ; an upper bound that grows without limit as T·σ·S² approaches κ₀)";
  const NOTES = { transport_zt_electronic_max: ZT_E, transport_zt_max: `${ZT_E}; the same value as transport_zt_electronic_max`,
    transport_kappa0_at_zt_max: "κ₀, the zero-field K coefficient of KAPPA.dat, at the ZTₑ peak",
    transport_kappa_electronic_at_zt_max: "κₑ = κ₀ − T·σ·S² at the ZTₑ peak (no lattice κ)" };
  const propTitle = (n) => (NOTES[n] ? `${n}: ${NOTES[n]}` : n);
  function textCell(p) {
    const t = String(p.property_value_text || "");
    if (t.length <= LONG) return esc(t);
    longValues.push(t);
    return `<span class="dbd-long" title="${t.length.toLocaleString("en-US")} characters">${esc(t.slice(0, CUT))}… <span class="muted">(${t.length.toLocaleString("en-US")} characters)</span></span>` +
      ` <button class="link-btn" type="button" data-copy-json="${longValues.length - 1}">Copy JSON</button>`;
  }
  function renderDetail(r) {
    longValues = [];
    const m = r.material;
    const done = r.calculations.filter((c) => c.status === "completed");
    const newest = done[done.length - 1];
    const shown = structureCalc(r.calculations);
    const fileBtn = (rel, label) => (rel ? `<button class="link-btn" data-open="${esc(rel)}" title="${esc(rel)}">${esc(label)}</button>` : "");
    const files = (c) => (c.output_rel || c.input_rel ? `${fileBtn(c.output_rel, "Open output")} ${fileBtn(c.input_rel, "Open input")}`
      : `<span class="muted small" title="${esc([c.output_file, c.input_file].filter(Boolean).join(", "))}">Files not in this folder</span>`);
    const cats = new Map();
    for (const p of r.properties) { const c = p.property_category || "other"; if (!cats.has(c)) cats.set(c, []); cats.get(c).push(p); }
    // a material with several calculations has one row per calculation for the same
    // property (an OPT's and an SP's CPU time): such rows name their calculation
    const typeOf = new Map(r.calculations.map((c) => [c.calc_id, c.calc_type]));
    const seen = new Map();
    for (const p of r.properties) seen.set(p.property_name, (seen.get(p.property_name) || 0) + 1);
    const from = (p) => (seen.get(p.property_name) > 1 && typeOf.get(p.calc_id) ? ` <span class="muted small dbd-from">(${esc(typeOf.get(p.calc_id))})</span>` : "");
    const fmt = (x) => (x === null || x === undefined ? "—" : Math.abs(x) >= 1e5 || (Math.abs(x) < 1e-3 && x !== 0) ? Number(x).toExponential(4) : Number(x).toFixed(5).replace(/\.?0+$/, ""));
    detailHost.innerHTML = `<div class="dbd">
      ${![m.formula, m.reduced_formula].includes(m.material_id) ? `<div class="dbd-title">${esc(m.material_id)}</div>` : ""}
      <div class="formula" title="cell formula ${esc(m.formula)}">${fmtFormula(m)}</div>
      <div class="badges"><span class="badge accent">${esc(m.dimensionality || "")}</span><span class="badge">SG ${esc(m.space_group ?? "—")}</span>${m.source_type ? `<span class="badge">${esc(m.source_type)}</span>` : ""}</div>
      <div class="dbd-acts">
        <button class="tb-btn-text" data-act="structure"${shown ? ` title="${esc(shown.output_rel || shown.input_rel)}"` : " disabled title=\"No output or input of this material is in this folder\""}>${icon("atom")}<span>Open Structure</span></button>
        <button class="tb-btn-text" data-act="next"${newest && newest.output_rel ? "" : " disabled title=\"Needs a completed calculation's output in this folder\""}>${icon("flask-conical")}<span>Set Up Next Calculation…</span></button>
      </div>
      <h4>Calculations <span class="muted">${r.calculations.length}</span></h4>
      <div class="dbd-calcs">${r.calculations.map((c) => `<div class="dbd-calc">
        <div class="dbd-c1"><b>${esc(c.calc_type)}</b><span class="status ${esc(c.status)}">${esc(c.status)}</span><span class="muted small">${esc((c.completed_at || "").replace("T", " ").slice(0, 16))}</span></div>
        <div class="dbd-c2">${files(c)}${c.error_type ? `<span class="bad small">${esc(c.error_type)}</span>` : ""}</div></div>`).join("") || '<p class="muted small">None</p>'}</div>
      <h4>Properties <span class="muted">${r.properties.length}</span></h4>
      ${[...cats.entries()].map(([c, ps]) => `<details open class="dbd-cat"><summary>${esc(c)}</summary><table class="kv">${ps.map((p) =>
        `<tr><td title="${esc(propTitle(p.property_name))}">${esc(p.property_name)}${from(p)}</td><td class="num">${p.property_value !== null && p.property_value !== undefined ? fmt(p.property_value) : textCell(p)}` +
        ` <span class="muted">${esc(UNITS[String(p.property_unit || "").toLowerCase()] || p.property_unit || "")}</span></td></tr>`).join("")}</table></details>`).join("") || '<p class="muted small">None</p>'}</div>`;
  }
  detailHost.addEventListener("click", (e) => {
    const cj = e.target.closest("[data-copy-json]");
    if (cj) {
      document.dispatchEvent(new CustomEvent("mace-copy", { detail: { text: longValues[Number(cj.dataset.copyJson)] || "", done: "Copied the JSON" } }));
      return;
    }
    const o = e.target.closest("[data-open]");
    if (o) { H.openPath(o.dataset.open); return; }
    const a = e.target.closest("[data-act]");
    if (!a || !db || !db.detail) return;
    const done = db.detail.calculations.filter((c) => c.status === "completed"), newest = done[done.length - 1];
    const shown = structureCalc(db.detail.calculations);
    if (a.dataset.act === "structure" && shown) H.openPath(shown.output_rel || shown.input_rel);
    if (a.dataset.act === "next" && newest && newest.output_rel) H.setupFor(newest.output_rel);
  });
  function openStructure(id) { select(id).then(() => { const b = detailHost.querySelector("[data-act=structure]"); if (b && !b.disabled) b.click(); }); }

  return {
    async open(path, focusId = null) {
      gen++;
      const t0 = performance.now();
      performance.mark("db:open:start");
      if (!db || db.path !== path) {
        db = { path, stats: null, props: [], total: null, sort: "material_id", dir: "asc", pages: new Map(), active: 0, selected: null,
          filters: { q: "", dims: [], calc_type: "", calc_status: "", sg_min: "", sg_max: "", prop: "" } };
        $(".dbv-q").value = "";
        host.querySelectorAll(".dbv-sg input").forEach((x) => { x.value = ""; });
        host.querySelectorAll(".dbv-dims button").forEach((b) => b.setAttribute("aria-pressed", "false"));
        for (const sel of [".dbv-type", ".dbv-status", ".dbv-prop"]) $(sel).value = "";
        scroller.scrollTop = 0;
        detailHost.innerHTML = "";
      }
      renderHead();
      live.stamp = null; live.updatedAt = 0; live.delta = 0; live.busy = null; liveText();
      const my = gen;
      try {
        const s = await untilFree(() => api("db/stats?" + q({ path })), my);
        await untilFree(() => pageOf(0, true).loading, my);
        db.stats = s;
        renderKpi();
        count();
        size();
        paint();
        api("db/props?" + q({ path })).then((r) => {
          db.props = r.names;
          const el = $(".dbv-prop"), cur = el.value;
          el.innerHTML = el.options[0].outerHTML + r.names.map((n) => `<option>${esc(n)}</option>`).join("");
          el.value = cur;
        }).catch(() => {});
      } catch (e) {
        if (e.superseded) return;
        host.querySelector(".dbv").hidden = true;
        throw e;
      }
      api("db/stamp?" + q({ path })).then((r) => { live.stamp = r.stamp; }).catch(() => {});
      schedule();
      stats.openMs = performance.now() - t0;
      if (focusId) {
        const r = await api("db/page?" + q({ ...filterQuery(), limit: 1, q: focusId })).catch(() => null);
        if (r && r.rows.length) select(focusId);
      }
      if (!db.selected) detailHost.innerHTML = `<p class="insp-hint">Select a material: its calculations and properties show here.</p>`;
    },
    /** The document closed: its filters, sort and selection go with it. */
    forget(path) { if (db && db.path === path) { gen++; db = null; clearTimeout(live.timer); } },
    refresh: () => refreshData(),
    /** Look at the file now (a build just started: "Updating…" at once, not at the next poll). */
    poll: () => checkStamp(),
    live: () => ({ ...live, text: ($(".dbv-live") || {}).textContent || "" }),
    detail: () => detailHost,
    focus: () => grid.focus(),
    exportUrl: (fmt) => "api/db/export?" + q({ ...filterQuery(), format: fmt }),
    exportName: (fmt) => `${db.path.split("/").pop().replace(/\.db$/, "")}_export.${fmt}`,
    state: () => db && { path: db.path, total: db.total, sort: db.sort, dir: db.dir, active: db.active, selected: db.selected, filters: { ...db.filters } },
    stats: () => ({ ...stats, domRows: rowsEl.childElementCount }),
    resetStats: () => { stats.maxRows = 0; stats.requests = 0; stats.firstPageMs = null; },
    moveTo, select,
    /** bench: a scripted scroll through the table for ms; frame times and the most row nodes seen. */
    benchScroll(ms = 3000) {
      return new Promise((resolve) => {
        const times = [], t0 = performance.now(), total = (db.total || 1) * rowH();
        let last = t0, maxRows = 0;
        const mo = new MutationObserver(() => { maxRows = Math.max(maxRows, rowsEl.childElementCount); });
        mo.observe(rowsEl, { childList: true });
        const step = (now) => {
          times.push(now - last); last = now;
          const f = (now - t0) / ms;
          scroller.scrollTop = Math.min(total, f * total * 0.98);
          if (now - t0 < ms) requestAnimationFrame(step);
          else {
            mo.disconnect();
            const s = times.slice(2).sort((a, b) => a - b);
            resolve({ frames: s.length, p95: s[Math.floor(s.length * 0.95)] || 0, max: s[s.length - 1] || 0, maxRows, requests: stats.requests });
          }
        };
        requestAnimationFrame(step);
      });
    },
  };
}
