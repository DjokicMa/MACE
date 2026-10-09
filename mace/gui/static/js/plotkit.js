/* plotkit (PLAN_analysis C3): the options every plot shares, in one place.
 *
 * A chart host registers with attach(actsEl, adapter): the toolbar gets the
 * quick controls only, so a narrow split pane stays readable
 *   Log Y · Units ▾ · Export ▾ (PNG, SVG, CSV, Copy as TSV, Publication Figure…) · ⋯ Plot Options
 * and the popover holds Log X, Relative to, Per, Range (from/to per axis,
 * blank = auto), Line width and Reset Zoom. The adapter owns its document's
 * option model and its persistence (per document); plotkit owns the unit
 * tables, the controls, CSV/TSV and the clipboard.
 *
 * adapter = { name, host (the chart element), features: {logY, logX, units: "energy" | "wavenumber" | null,
 *   per, relTo, single, range, width, png, svg, image, csv, pub}, get() -> opts, set(patch, label),
 *   table() -> {columns, rows}, png(), svg(), image() (Export Image…), pub(), resetZoom(), axes() -> {x: label, y: label} }
 * opts = {ylog, xlog, unit, per, relTo, single, xr: [from, to], yr: [from, to], width} */
import { overlayHost, place } from "./ui.js";

export const HARTREE_EV = 27.211386245988;          // CODATA 2018
export const AVOGADRO_EV_KJMOL = 96.48533212331;    // 1 eV per particle in kJ/mol
export const EV_KCALMOL = 23.060547830619;          // 1 eV per particle in kcal/mol
export const CM_PER_MEV = 8.065543937;              // 1 meV = 8.0655 cm⁻¹
export const CM_PER_THZ = 33.35640952;

/** value in the unit = value in eV × factor */
export const ENERGY_UNITS = { eV: 1, meV: 1000, Ha: 1 / HARTREE_EV, "kJ/mol": AVOGADRO_EV_KJMOL, "kcal/mol": EV_KCALMOL };
/** value in the unit = value in cm⁻¹ × factor */
export const WAVENUMBER_UNITS = { "cm-1": 1, meV: 1 / CM_PER_MEV, THz: 1 / CM_PER_THZ };
export const UNIT_LABEL = { eV: "eV", meV: "meV", Ha: "Ha", "kJ/mol": "kJ/mol", "kcal/mol": "kcal/mol", "cm-1": "cm⁻¹", THz: "THz" };
export const PER_LABEL = { cell: "", atom: "/atom", fu: "/f.u." };

export function energy(eV, unit = "eV") { return eV * (ENERGY_UNITS[unit] ?? 1); }
export function wavenumber(cm, unit = "cm-1") { return cm * (WAVENUMBER_UNITS[unit] ?? 1); }
export function unitsFor(kind) { return kind === "energy" ? Object.keys(ENERGY_UNITS) : kind === "wavenumber" ? Object.keys(WAVENUMBER_UNITS) : []; }
/** The axis unit text: "meV/atom"; kJ/mol and kcal/mol say "per cell" or "per atom" explicitly. */
export function unitText(unit, per = "cell") {
  const u = UNIT_LABEL[unit] || unit;
  if (unit === "kJ/mol" || unit === "kcal/mol") return `${u} ${per === "atom" ? "per atom" : per === "fu" ? "per formula unit" : "per cell"}`;
  return u + (PER_LABEL[per] || "");
}

/* ------------------------------------------------------------ images */
/** A data: URL (Plotly.toImage, canvas.toDataURL) as a Blob. Decoded here:
 * the page's CSP (connect-src 'self') makes fetch() of a data: URL fail. */
export function dataUrlBlob(url) {
  const k = url.indexOf(","), headPart = url.slice(5, k), data = url.slice(k + 1);
  const type = headPart.split(";")[0] || "application/octet-stream";
  if (headPart.includes(";base64")) {
    const bin = atob(data), arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return new Blob([arr], { type });
  }
  return new Blob([decodeURIComponent(data)], { type });
}

/* ------------------------------------------------------------ tables */
const cellText = (v) => (v === null || v === undefined || (typeof v === "number" && !Number.isFinite(v)) ? "" : String(v));
export function toTSV(t) { return [t.columns.join("\t"), ...t.rows.map((r) => r.map(cellText).join("\t"))].join("\n") + "\n"; }
export function toCSV(t) {
  const q = (s) => (/[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s);
  return [t.columns.map(q).join(","), ...t.rows.map((r) => r.map((v) => q(cellText(v))).join(","))].join("\n") + "\n";
}
/** The clipboard: the async API on a secure origin, else a hidden textarea and execCommand. */
export async function copyText(text) {
  if (window.isSecureContext && navigator.clipboard && navigator.clipboard.writeText) {
    try { await navigator.clipboard.writeText(text); return true; } catch (e) { /* fall back */ }
  }
  const ta = document.createElement("textarea");
  ta.value = text; ta.setAttribute("readonly", ""); ta.style.position = "fixed"; ta.style.opacity = "0"; ta.style.top = "0";
  document.body.appendChild(ta);
  ta.select();
  let ok = false;
  try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
  ta.remove();
  return ok;
}

/* ------------------------------------------------------------ registry */
export const hosts = new Set();
let menuOpen = null;
function closeMenus() { if (menuOpen) { const m = menuOpen; menuOpen = null; m.close(); } }
/* A menu or the options popover opens in the overlay layer (ui.js), so the
 * plot's panel can neither clip nor cover it, right-aligned under its button
 * and kept on screen. Tab past either end closes it and goes on from the button. */
function float(el, btn) {
  overlayHost().appendChild(el);
  const r = btn.getBoundingClientRect();
  place(el, { left: r.right - el.offsetWidth, right: r.right, top: r.top, bottom: r.bottom }, "bottom", 4);
  el.addEventListener("keydown", (e) => {
    if (e.key !== "Tab") return;
    const all = [...el.querySelectorAll("button:not([disabled]), input, select")];
    const atEnd = e.shiftKey ? document.activeElement === all[0] : document.activeElement === all[all.length - 1];
    if (el.getAttribute("role") === "menu" || atEnd) {
      if (e.shiftKey) e.preventDefault();      // back to the button itself
      closeMenus();
    }
  });
}
document.addEventListener("pointerdown", (e) => { if (menuOpen && !menuOpen.contains(e.target)) closeMenus(); }, true);
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && menuOpen) { e.preventDefault(); e.stopPropagation(); closeMenus(); } }, true);

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/**
 * The toolbar into ``acts`` for ``adapter``; returns {refresh(), el}.
 * H: {toast, download, icon}
 */
export function attach(acts, adapter, H) {
  // the features may change with the document shown (a spectrum after a band plot)
  const feat = () => ({ logY: true, logX: false, units: null, per: false, relTo: false, single: false, range: true, width: true,
    png: true, svg: true, image: false, csv: true, pub: false, ...(adapter.features || {}) });
  let f = feat(), sig = "";
  const icon = H.icon || (() => "");
  acts.classList.add("pk-acts");
  acts.dataset.plotkit = "1";
  if (adapter.host) { adapter.host.dataset.plotkit = "1"; hosts.add(adapter.host); }
  function render() {
    f = feat();
    const s = JSON.stringify(f);
    if (s === sig) return;
    sig = s;
    acts.innerHTML = `${f.logY ? `<button type="button" class="tb-btn-text pk-logy" aria-pressed="false" title="Logarithmic value axis">Log Y</button>` : ""}
    ${f.units ? `<span class="pk-wrap"><button type="button" class="tb-btn-text pk-units" aria-haspopup="menu" aria-expanded="false" title="Units">Units<span class="pk-u"></span> ▾</button></span>` : ""}
    <span class="pk-wrap"><button type="button" class="tb-btn-text pk-export" aria-haspopup="menu" aria-expanded="false" title="Export the plot or its data">${icon("download")}<span>Export</span> ▾</button></span>
    <span class="pk-wrap"><button type="button" class="tb-btn-text pk-more" aria-haspopup="dialog" aria-expanded="false" title="Plot options">⋯ Plot Options</button></span>`;
  }
  render();
  const $ = (s) => acts.querySelector(s);

  function menu(btn, items, onPick) {
    closeMenus();
    const m = document.createElement("div");
    m.className = "pk-menu";
    m.setAttribute("role", "menu");
    m.innerHTML = items.map((it) => it === "-" ? `<div class="pk-sep" role="separator"></div>` :
      `<button type="button" role="${it.checked !== undefined ? "menuitemradio" : "menuitem"}" data-v="${esc(it.v)}"${it.checked !== undefined ? ` aria-checked="${it.checked}"` : ""}${it.disabled ? ' aria-disabled="true" disabled' : ""}>${esc(it.label)}</button>`).join("");
    float(m, btn);
    btn.setAttribute("aria-expanded", "true");
    m.close = () => { m.remove(); btn.setAttribute("aria-expanded", "false"); btn.focus(); };
    menuOpen = m;
    const btns = [...m.querySelectorAll("button:not([disabled])")];
    (btns.find((b) => b.getAttribute("aria-checked") === "true") || btns[0])?.focus();
    m.addEventListener("keydown", (e) => {
      const k = btns.indexOf(document.activeElement);
      if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); btns[(k + (e.key === "ArrowDown" ? 1 : btns.length - 1)) % btns.length].focus(); }
    });
    m.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-v]");
      if (!b || b.disabled) return;
      menuOpen = null; m.remove(); btn.setAttribute("aria-expanded", "false");
      onPick(b.dataset.v);
    });
  }

  function popover(btn) {
    closeMenus();
    const o = adapter.get(), ax = adapter.axes ? adapter.axes() : { x: "x", y: "y" };
    const p = document.createElement("div");
    p.className = "pk-pop";
    p.setAttribute("role", "dialog");
    p.setAttribute("aria-label", "Plot options");
    const num = (k, i, v) => `<input type="number" step="any" data-r="${k}${i}" value="${v ?? ""}" placeholder="auto" aria-label="${k === "x" ? esc(ax.x) : esc(ax.y)} ${i ? "to" : "from"}">`;
    p.innerHTML = `${f.logX ? `<label class="pk-row"><input type="checkbox" data-k="xlog"${o.xlog ? " checked" : ""}> Log X</label>` : ""}
      ${f.relTo ? `<label class="pk-row"><span>Relative to</span><select data-k="relTo" aria-label="Relative to">${[["first", "First point"], ["final", "Final point"], ["minimum", "Minimum"], ["absolute", "Absolute"]]
        .map(([v, t]) => `<option value="${v}"${o.relTo === v ? " selected" : ""}${(adapter.disabled || {})["relTo:" + v] ? ` disabled title="${esc(adapter.disabled["relTo:" + v])}"` : ""}>${t}</option>`).join("")}</select></label>` : ""}
      ${f.single ? `<label class="pk-row"><input type="checkbox" data-k="single"${o.single ? " checked" : ""}> Single reference (merge FINALRUN segments)</label>` : ""}
      ${f.per ? `<label class="pk-row"><span>Per</span><select data-k="per" aria-label="Energy per">${[["cell", "Cell"], ["atom", "Atom"], ["fu", "Formula unit"]]
        .map(([v, t]) => `<option value="${v}"${o.per === v ? " selected" : ""}>${t}</option>`).join("")}</select></label>` : ""}
      ${f.range ? `<div class="pk-row pk-range"><span>${esc(ax.x)}</span>${num("x", 0, (o.xr || [])[0])}<span>–</span>${num("x", 1, (o.xr || [])[1])}</div>
        <div class="pk-row pk-range"><span>${esc(ax.y)}</span>${num("y", 0, (o.yr || [])[0])}<span>–</span>${num("y", 1, (o.yr || [])[1])}</div>` : ""}
      ${f.width ? `<label class="pk-row"><span>Line width</span><input type="number" min="0.5" max="6" step="0.5" data-k="width" value="${o.width ?? 1.5}" aria-label="Line width"></label>` : ""}
      <div class="pk-row"><button type="button" class="tb-btn-text" data-act="resetzoom">Reset Zoom</button></div>`;
    float(p, btn);
    btn.setAttribute("aria-expanded", "true");
    p.close = () => { p.remove(); btn.setAttribute("aria-expanded", "false"); btn.focus(); };
    menuOpen = p;
    (p.querySelector("input, select, button"))?.focus();
    p.addEventListener("change", (e) => {
      const t = e.target;
      if (t.dataset.k === "xlog") adapter.set({ xlog: t.checked }, t.checked ? "Log X" : "Linear X");
      else if (t.dataset.k === "single") adapter.set({ single: t.checked }, t.checked ? "Single reference" : "Per segment");
      else if (t.dataset.k === "relTo") adapter.set({ relTo: t.value }, `Relative to ${t.value}`);
      else if (t.dataset.k === "per") adapter.set({ per: t.value }, `Per ${t.value}`);
      else if (t.dataset.k === "width") adapter.set({ width: Number(t.value) || 1.5 }, `Line width ${t.value}`);
      else if (t.dataset.r) {
        const axk = t.dataset.r[0], i = Number(t.dataset.r[1]);
        const cur = (adapter.get()[`${axk}r`] || [null, null]).slice();
        cur[i] = t.value === "" ? null : Number(t.value);
        adapter.set({ [`${axk}r`]: cur.every((v) => v === null) ? null : cur }, `Range ${axk} ${t.value || "auto"}`);
      }
      refresh();
    });
    p.addEventListener("click", (e) => {
      if (e.target.closest("[data-act=resetzoom]")) {
        adapter.set({ xr: null, yr: null }, "Reset zoom");
        if (adapter.resetZoom) adapter.resetZoom();
        p.querySelectorAll("[data-r]").forEach((x) => { x.value = ""; });
      }
    });
  }

  function refresh() {
    render();
    const o = adapter.get();
    const ly = $(".pk-logy");
    if (ly) ly.setAttribute("aria-pressed", String(!!o.ylog));
    const u = $(".pk-u");
    if (u) u.textContent = o.unit ? `: ${UNIT_LABEL[o.unit] || o.unit}` : "";
  }

  acts.addEventListener("click", async (e) => {
    const b = e.target.closest("button");
    if (!b || !acts.contains(b) || b.closest(".pk-menu, .pk-pop")) return;
    if (b.classList.contains("pk-logy")) { const v = !adapter.get().ylog; adapter.set({ ylog: v }, v ? "Log Y" : "Linear Y"); refresh(); return; }
    if (b.classList.contains("pk-units")) {
      if (menuOpen) { closeMenus(); return; }
      menu(b, unitsFor(f.units).map((v) => ({ v, label: UNIT_LABEL[v] || v, checked: adapter.get().unit === v })),
        (v) => { adapter.set({ unit: v }, `Units ${UNIT_LABEL[v] || v}`); refresh(); });
      return;
    }
    if (b.classList.contains("pk-export")) {
      if (menuOpen) { closeMenus(); return; }
      const items = [];
      if (f.png) items.push({ v: "png", label: "PNG" });
      if (f.svg) items.push({ v: "svg", label: "SVG" });
      if (f.image && adapter.image) items.push({ v: "image", label: "Image…" });
      if (f.csv) items.push({ v: "csv", label: "CSV" }, { v: "tsv", label: "Copy as TSV" });
      if (f.pub) items.push("-", { v: "pub", label: "Publication Figure…" });
      menu(b, items, (v) => run(v));
      return;
    }
    if (b.classList.contains("pk-more")) { if (menuOpen) { closeMenus(); return; } popover(b); }
  });

  async function run(v) {
    try {
      if (v === "png") await adapter.png();
      else if (v === "svg") await adapter.svg();
      else if (v === "image") await adapter.image();
      else if (v === "pub") await adapter.pub();
      else if (v === "csv") H.download(new Blob([toCSV(adapter.table())], { type: "text/csv" }), `${adapter.name || "plot"}_data.csv`);
      else if (v === "tsv") {
        const t = adapter.table();
        const ok = await copyText(toTSV(t));
        H.toast(ok ? `Copied ${t.rows.length} rows` : "Copy failed (the browser refused the clipboard)", ok ? "success" : "error");
      }
    } catch (err) { H.toast(`Export: ${err.message}`, "error"); }
  }
  refresh();
  return { refresh, el: acts, run };
}

/** For log axes: |value| where non-positive values exist (and a chip saying so). */
export function absIfNeeded(ys) {
  let neg = false;
  const out = ys.map((y) => { if (y !== null && y <= 0) { neg = true; return y === 0 ? null : Math.abs(y); } return y; });
  return { ys: out, abs: neg };
}
