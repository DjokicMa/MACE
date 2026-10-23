/* Shared by the Phonopy, Mechanical Properties and Relaxation Time dialogs:
 * the dialog shell with tabs, a few form helpers and themed plotly layouts. */
import { loadCss } from "./lazy_util.js";

export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
export const num = (v, d = 3) => (v === null || v === undefined || !isFinite(v) ? "–" : Number(v).toFixed(d));

/** A non-modal dialog with tabs: {id, title, tabs: [[key, label, html]], foot} (one tab: no tab strip). Returns {d, $, $$, show(tab)}. */
export async function shell(H, { id, title, tabs, foot = "", onClose = null }) {
  await loadCss(import.meta.url, "../../css/dialogs.css");
  await loadCss(import.meta.url, "../../css/mechanics.css");
  H.overlayOpening();
  const opener = document.activeElement;
  let d = document.getElementById(id);
  if (d && d.open) d.close();
  if (!d) { d = document.createElement("dialog"); d.id = id; document.body.appendChild(d); }
  d.className = "dlg mx-dlg";
  d.setAttribute("aria-label", title);
  d.innerHTML = `<div class="dlg-head"><h2></h2><button class="icon-btn sm" data-close aria-label="Close">${H.icon("x")}</button></div>
    <div class="dlg-body dlg-panel">
      ${tabs.length > 1 ? `<div class="mx-tabs" role="tablist">${tabs.map(([k, label], i) =>
        `<button role="tab" data-tab="${k}" aria-selected="${i === 0}">${esc(label)}</button>`).join("")}</div>` : ""}
      ${tabs.map(([k, , html], i) => `<section data-pane="${k}" role="tabpanel"${i ? " hidden" : ""} class="mx-section">${html}</section>`).join("")}
    </div>${foot ? `<div class="dlg-foot">${foot}</div>` : ""}`;
  d.querySelector("h2").textContent = title;
  const $ = (s) => d.querySelector(s), $$ = (s) => Array.from(d.querySelectorAll(s));
  const show = (k) => {
    $$("[data-tab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === k)));
    $$("[data-pane]").forEach((p) => { p.hidden = p.dataset.pane !== k; });
    d.dispatchEvent(new CustomEvent("mx-tab", { detail: k }));
  };
  $$("[data-tab]").forEach((b) => b.addEventListener("click", () => show(b.dataset.tab)));
  const close = () => d.close();
  $$("[data-close]").forEach((b) => b.addEventListener("click", close));
  // Esc closes it, as the other panels (lazy_util.js panel); a non-modal dialog gets no Esc of its own
  d.onkeydown = (e) => { if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(); } };
  d.addEventListener("close", () => { if (onClose) onClose(); if (opener && opener.focus && opener.isConnected) opener.focus(); }, { once: true });
  d.show();
  return { d, $, $$, show };
}

/** The value of a field: "" -> undefined, number fields -> Number. */
export function val(el) {
  if (!el) return undefined;
  if (el.type === "checkbox") return el.checked;
  const v = String(el.value ?? "").trim();
  if (v === "") return undefined;
  return el.type === "number" ? Number(v) : v;
}

/** The theme's colors for plotly (tokens: --plot-bg, --plot-ink, --border, --accent). */
export function themed(layout = {}) {
  const css = getComputedStyle(document.documentElement);
  const t = (k, f) => (css.getPropertyValue(k) || "").trim() || f;
  const ink = t("--plot-ink", t("--text", "")), grid = t("--border", "");
  const axis = { color: ink, gridcolor: grid, zerolinecolor: grid, linecolor: ink, mirror: true, showline: true };
  return { paper_bgcolor: t("--plot-bg", t("--bg", "")), plot_bgcolor: t("--plot-bg", t("--bg", "")),
    font: { color: ink, size: 12 }, margin: { l: 56, r: 12, t: 12, b: 44 }, showlegend: false,
    ...layout, xaxis: { ...axis, ...(layout.xaxis || {}) }, yaxis: { ...axis, ...(layout.yaxis || {}) } };
}
export const accent = () => (getComputedStyle(document.documentElement).getPropertyValue("--accent") || "").trim();

/** A 2D array as an HTML table (row/column headers optional). */
export function matrixTable(m, { cols = null, rows = null, digits = 3 } = {}) {
  if (!m) return "";
  const head = cols ? `<tr><th></th>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>` : "";
  return `<table class="mx-table">${head}${m.map((r, i) => `<tr>${rows ? `<th class="l">${esc(rows[i] ?? "")}</th>` : (cols ? "<th></th>" : "")}${
    r.map((x) => `<td>${num(x, digits)}</td>`).join("")}</tr>`).join("")}</table>`;
}
export const VOIGT = ["xx", "yy", "zz", "yz", "xz", "xy"];

/** Copy text with the shell's helper (falls back to the clipboard API). */
export function copy(H, text, done = "Copied") {
  if (H.copyText) return H.copyText(text, done);
  return navigator.clipboard && navigator.clipboard.writeText(text);
}
