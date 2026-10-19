/* plotkit (PLAN_analysis C3): the options every plot shares, in one place.
 *
 * A chart host registers with attach(actsEl, adapter): the toolbar gets the
 * quick controls only, so a narrow split pane stays readable
 *   Log Y · Units ▾ · Export ▾ (PNG, SVG, CSV, Copy as TSV, Publication Figure…) · ⋯ Plot Options
 * and the popover holds Log X, Relative to, Segments, Per, Range (from/to per axis,
 * blank = auto), Line width and Reset Zoom. The adapter owns its document's
 * option model and its persistence (per document); plotkit owns the unit
 * tables, the controls, CSV/TSV and the clipboard.
 *
 * adapter = { name, host (the chart element), features: {logY, logX, units: "energy" | "wavenumber" | null,
 *   per, relTo, segments, range, width, png, svg, image, csv, pub}, get() -> opts, set(patch, label),
 *   table() -> {columns, rows}, png(), svg(), image() (Export Image…), pub(), resetZoom(), axes() -> {x: label, y: label} }
 * opts = {ylog, xlog, unit, per, relTo, segments ("continuous" | "separate"), xr: [from, to], yr: [from, to], width} */
import { overlayHost, place } from "./ui.js";

import { HARTREE_EV, AVOGADRO_EV_KJMOL, EV_KCALMOL, CM_PER_MEV, CM_PER_THZ, ENERGY_UNITS, WAVENUMBER_UNITS, UNIT_LABEL, PER_LABEL, legendTop, legendSide } from "./plotbase.js";
export { HARTREE_EV, AVOGADRO_EV_KJMOL, EV_KCALMOL, CM_PER_MEV, CM_PER_THZ, ENERGY_UNITS, WAVENUMBER_UNITS, UNIT_LABEL, PER_LABEL, legendTop, legendSide };

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
  // the button's rect is screen px; the element's width is its own px (the layer's zoom)
  const r = btn.getBoundingClientRect(), z = window.maceZoom.of(el);
  place(el, { left: r.right - el.offsetWidth * z, right: r.right, top: r.top, bottom: r.bottom }, "bottom", 4);
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
  const feat = () => ({ logY: true, logX: false, units: null, per: false, relTo: false, segments: false, range: true, width: true,
    png: true, svg: true, image: false, csv: true, pub: false, style: false, cli: false, ...(adapter.features || {}) });
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
    ${f.style && adapter.styleMenu ? `<span class="pk-wrap"><button type="button" class="tb-btn-text pk-style" aria-haspopup="menu" aria-expanded="false" title="Plot style, spin colors and DOS look of this plot">Style ▾</button></span>` : ""}
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
    m.innerHTML = items.map((it) => it === "-" ? `<div class="pk-sep" role="separator"></div>` : it.head ? `<div class="pk-head${it.group ? " pk-group" : ""}" role="presentation"${it.group ? ` data-group="${esc(it.group)}"` : ""}>${esc(it.head)}</div>` :
      `<button type="button" role="${it.checked !== undefined ? (it.toggle ? "menuitemcheckbox" : "menuitemradio") : "menuitem"}" data-v="${esc(it.v)}"${it.checked !== undefined ? ` aria-checked="${it.checked}"` : ""}${it.disabled ? ' aria-disabled="true" disabled' : ""}>${esc(it.label)}</button>`).join("");
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
      ${f.segments ? `<label class="pk-row" title="One curve and one reference across the FINALRUN restarts, or each FINALRUN segment relative to its own point"><span>Segments</span><select data-k="segments" aria-label="FINALRUN segments">${[["continuous", "Continuous"], ["separate", "Separate"]]
        .map(([v, t]) => `<option value="${v}"${(o.segments || "continuous") === v ? " selected" : ""}>${t}</option>`).join("")}</select></label>` : ""}
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
      else if (t.dataset.k === "segments") adapter.set({ segments: t.value }, t.value === "separate" ? "Segments: separate" : "Segments: continuous");
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
      if (f.cli && adapter.cli) items.push({ v: "cli", label: "Copy CLI Command" });
      menu(b, items, (v) => run(v));
      return;
    }
    if (b.classList.contains("pk-style")) {
      if (menuOpen) { closeMenus(); return; }
      // the plot's own style (plot.js): picking an entry is one undoable option change
      menu(b, adapter.styleMenu(), (v) => {
        if (v === "cli") run("cli"); else adapter.styleSet(v);
        refresh();
        // the pick drew the toolbar again: the keyboard's focus goes to the new Style button, not to <body>
        const back = () => { const nb = $(".pk-style"); if (nb && (!document.activeElement || document.activeElement === document.body || !document.activeElement.isConnected)) nb.focus(); };
        back(); requestAnimationFrame(back);
      });
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
      else if (v === "cli") {
        const text = adapter.cli();
        const ok = await copyText(text);
        H.toast(ok ? `Copied: ${text.split("\n").pop()}` : "Copy failed (the browser refused the clipboard)", ok ? "success" : "error");
      }
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

/* ------------------------------------------------------------ legends outside the data
   One rule for every plotly plot MACE draws (plot.js, runplot.js; the CLI's figures follow
   it too, mace/plotting/legend_layout.py): a legend, a reference line's label, a colorbar
   never sits on the data, a tick label or each other. Legends go above the plot ("top",
   wrapping into rows) or in a column right of it ("side"); a reference line's label is a
   legend entry of its own (keyEntry: no data, a click hides nothing); the top and right
   margins are sized by fitLegends from the legends as drawn (plotly's own margin push
   leaves a legend where the previous size put it when a resize changes its height). */
/* legendTop and legendSide: plotbase.js (no DOM: the band/DOS figure builder imports them in node too) */
/** A legend-only entry: a key for a line style, a reference line or a shaded band (meta "key"). */
export function keyEntry(name, style = {}, more = {}) {
  const { line, marker, fill } = style;
  const t = { x: [null], y: [null], mode: marker && !line ? "markers" : "lines", type: "scatter", name, meta: "key", hoverinfo: "skip" };
  if (line) t.line = line;
  if (marker) t.marker = marker;
  if (fill) Object.assign(t, { fill: "toself", fillcolor: fill });
  return { ...t, ...more };
}
/** plotly_legendclick / legenddoubleclick handler: a key entry hides nothing. */
export function keyClick(ev) {
  const t = ev && ev.data && ev.data[ev.curveNumber];
  return !(t && typeof t.meta === "string" && t.meta.startsWith("key"));
}
/** Size the top and right margins to the legends and top-axis labels as drawn. base = {t, r}
 *  (the margins without legends); extra(r) may add relayout keys that depend on the right margin. */
export async function fitLegends(el, base, extra = null) {
  if (!base || !window.Plotly || !el._fullLayout || el.__fitting) return;
  el.__fitting = true;
  try {
    for (let i = 0; i < 8; i++) {
      const L = el._fullLayout, ids = Object.keys(L).filter((k) => /^legend\d*$/.test(k));
      const drawn = (id) => !!el.querySelector(`.infolayer > g.${id} [class$=text]`);
      // a top legend that would take more than a third of the figure's height (many or long entries,
      // a short pane) becomes a column right of the plot, where it scrolls instead of crushing the data
      let rband = 0;
      {
        const r0 = el.querySelector(".main-svg").getBoundingClientRect(), z = r0.width / L.width || 1, right = r0.left + z * (L._size.l + L._size.w);
        for (const ax of Object.keys(L).filter((k) => /^yaxis\d*$/.test(k) && L[k].side === "right" && L[k].visible !== false)) {
          for (const tx of el.querySelectorAll(`g.${L[ax]._id}tick > text, .g-${L[ax]._id}title text`)) rband = Math.max(rband, (tx.getBoundingClientRect().right - right) / z);
        }
      }
      // a column chosen for another size: back to rows when the figure's size changed, and judged again
      const sw = el.__sideFromTop;
      if (sw && sw.ids.every((id) => L[id] && L[id].orientation === "v") && (Math.abs(sw.w - L.width) > 2 || Math.abs(sw.h - L.height) > 2)) {
        el.__sideFromTop = null;
        const upd = {};
        for (const id of sw.ids) Object.assign(upd, { [`${id}.orientation`]: "h", [`${id}.x`]: 0, [`${id}.xanchor`]: "left", [`${id}.y`]: 1, [`${id}.yanchor`]: "bottom" });
        await window.Plotly.relayout(el, upd);
        continue;
      }
      // rows are judged at the width they will have: a right margin left wide by a column (this
      // figure's last fit, or a redraw that kept it) narrows the plot, plotly wraps the rows to that
      // width, one entry per row, and the rows look tall enough to pick the column again
      const sideNow = ids.some((id) => L[id] && L[id].orientation === "v" && L[id].x >= 1 && drawn(id));
      const rRows = Math.ceil(base.r + rband);
      if (!sideNow && L.margin.r > rRows + 1 && ids.some((id) => L[id] && L[id].orientation === "h" && L[id].y >= 1 && drawn(id))) {
        const upd = { "margin.r": rRows };
        if (extra) Object.assign(upd, extra(rRows));
        await window.Plotly.relayout(el, upd);
        continue;
      }
      // (only when the column costs the data less width than the rows cost it height)
      const z0 = el.querySelector(".main-svg").getBoundingClientRect().width / L.width || 1;
      const colWidth = (id) => Math.max(0, ...[...el.querySelectorAll(`.infolayer > g.${id} g.traces`)].map((g) => g.getBoundingClientRect().width / z0)) + 12;
      const tall = ids.filter((id) => L[id] && L[id].orientation === "h" && L[id].y >= 1 && drawn(id) && L[id]._height > L.height / 3
        && (colWidth(id) + rband) / L.width < L[id]._height / L.height);
      if (tall.length) {
        const upd = {};
        for (const id of tall) Object.assign(upd, { [`${id}.orientation`]: "v", [`${id}.x`]: 1, [`${id}.xanchor`]: "left", [`${id}.y`]: 1, [`${id}.yanchor`]: "top" });
        el.__sideFromTop = { ids: tall, w: L.width, h: L.height };
        await window.Plotly.relayout(el, upd);
        continue;
      }
      let top = 0, side = 0;
      for (const id of ids) {
        const g = L[id];
        if (!g || !drawn(id)) continue;
        if (g.orientation === "h" && g.y >= 1) top = Math.max(top, g._height);
        else if (g.orientation === "v" && g.x >= 1) side = Math.max(side, g._width);
      }
      // labels of an axis on top (the peaks of a spectrum) take room above the plot too
      let band = 0;
      for (const ax of Object.keys(L).filter((k) => /^xaxis\d*$/.test(k) && L[k].side === "top" && L[k].visible !== false)) {
        const r = el.querySelector(".main-svg").getBoundingClientRect(), z = r.width / L.width || 1;
        for (const t of el.querySelectorAll(`g.${L[ax]._id}tick > text`)) band = Math.max(band, (r.top + z * L._size.t - t.getBoundingClientRect().top) / z);
      }
      // labels of an axis on the right (a second y axis) take room right of the plot: a side legend goes beyond them
      const t = Math.ceil(base.t + top + band + (band ? 4 : 0)), r = Math.ceil(base.r + rband + (side ? side + 4 : 0));
      const upd = {};
      if (L.margin.t !== t) upd["margin.t"] = t;
      if (L.margin.r !== r) upd["margin.r"] = r;
      // a side legend right of the right-axis labels
      for (const id of ids) if (L[id] && L[id].orientation === "v" && L[id].x >= 1 && drawn(id)) {
        const x = rband ? 1 + (rband + 4) / Math.max(1, L.width - L.margin.l - r) : 1;
        if (Math.abs(L[id].x - x) > 1e-3) upd[`${id}.x`] = x;
      }
      // a top legend above the top-axis labels
      for (const id of ids) if (L[id] && L[id].orientation === "h" && L[id].y >= 1 && drawn(id)) {
        const y = band ? 1 + (band + 4) / Math.max(1, L.height - t - L.margin.b) : 1;
        if (Math.abs(L[id].y - y) > 1e-3) upd[`${id}.y`] = y;
      }
      if (!Object.keys(upd).length) break;
      if (extra) Object.assign(upd, extra(r));
      await window.Plotly.relayout(el, upd);
    }
  } finally { el.__fitting = false; }
}
