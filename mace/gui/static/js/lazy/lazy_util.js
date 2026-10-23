/* Shared by the lazy modules: their stylesheet loads with them, once. */
import { bindSplitter } from "../splitter.js";

const loaded = new Map();
export function loadCss(base, rel) {
  const href = new URL(rel, base).href;
  if (!loaded.has(href)) {
    loaded.set(href, new Promise((resolve) => {
      const l = document.createElement("link");
      l.rel = "stylesheet"; l.href = href;
      l.onload = () => resolve(); l.onerror = () => resolve();
      document.head.appendChild(l);
    }));
  }
  return loaded.get(href);
}

/** A non-modal panel (spec §2.5: the viewport stays usable): a <dialog>
 * opened with show(), header, body and footer; Esc or Close closes it, and
 * focus goes back to where it was. Opening it again reuses it. A resizable
 * one has a handle on its left edge (its width is kept while the page is open). */
export function panel(H, { id, title, body, foot = "", onClose = null, wide = false, resizable = false }) {
  let d = document.getElementById(id);
  const opener = document.activeElement;
  if (d && d.open) d.close();
  if (!d) {
    d = document.createElement("dialog");
    d.id = id;
    document.body.appendChild(d);
  }
  d.className = "dlg dlg-panel" + (wide ? " wide" : "");
  d.setAttribute("aria-label", title);
  d.innerHTML = `<div class="dlg-head"><h2></h2><button class="icon-btn sm" data-close aria-label="Close">${H.icon("x")}</button></div>
    <div class="dlg-body"></div>${foot ? `<div class="dlg-foot">${foot}</div>` : ""}`;
  d.querySelector("h2").textContent = title;
  const b = d.querySelector(".dlg-body");
  if (typeof body === "string") b.innerHTML = body; else b.appendChild(body);
  // the "close" event comes a task later: a close by the panel's own button or
  // Esc runs onClose at once, so what the panel owns (a preview in the
  // viewport) goes in the same frame as the panel
  let shut = false;
  const closed = () => {
    if (shut) return;
    shut = true;
    if (onClose) onClose();
    if (opener && opener.focus && opener.isConnected) opener.focus();
  };
  const closeNow = () => { d.close(); closed(); };
  d.querySelector("[data-close]").addEventListener("click", closeNow);
  dragByHead(d);
  if (resizable) resizeByEdge(d);
  d.onkeydown = (e) => { if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); closeNow(); } };
  d.addEventListener("close", closed, { once: true });
  H.overlayOpening();
  d.show();
  keepInside(d);
  // and when it changes size afterwards (its own stylesheet, a result, its left edge)
  if (!d.__inside && typeof ResizeObserver === "function") { d.__inside = new ResizeObserver(() => keepInside(d)); d.__inside.observe(d); }
  const first = d.querySelector(".dlg-body :is(input, select, textarea)");
  if (first) { first.focus(); if (first.select) first.select(); }
  return d;
}

/** Drag a floating panel by its header (it stays inside the window); the
 * place is kept while the page is open, so reopening finds it there. */
const placed = new Map();
function dragByHead(d) {
  const head = d.querySelector(".dlg-head");
  const at = placed.get(d.id);
  if (at) Object.assign(d.style, { left: Math.max(0, Math.min(at[0], innerWidth - 240)) + "px",
    top: Math.max(0, Math.min(at[1], innerHeight - 120)) + "px", right: "auto" });
  head.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || e.target.closest("button")) return;
    const r = d.getBoundingClientRect(), dx = e.clientX - r.left, dy = e.clientY - r.top;
    head.setPointerCapture(e.pointerId);
    const move = (ev) => {
      const x = Math.min(Math.max(0, ev.clientX - dx), innerWidth - r.width), y = Math.min(Math.max(0, ev.clientY - dy), innerHeight - 40);
      Object.assign(d.style, { left: x + "px", top: y + "px", right: "auto" });
      placed.set(d.id, [x, y]);
    };
    const up = () => { head.removeEventListener("pointermove", move); head.removeEventListener("pointerup", up); head.removeEventListener("pointercancel", up); };
    head.addEventListener("pointermove", move);
    head.addEventListener("pointerup", up);
    head.addEventListener("pointercancel", up);
    e.preventDefault();
  });
}

/** A panel dragged somewhere stays whole inside the window: when it opens
 * (the window may be narrower, or the panel wider, than when it was placed)
 * and when the window is resized. One anchored to the right edge (never
 * dragged) is inside by its CSS. */
function keepInside(d) {
  if (!d.open || !d.style.left) return;
  const r = d.getBoundingClientRect();
  const x = Math.max(0, Math.min(r.left, innerWidth - r.width)), y = Math.max(0, Math.min(r.top, innerHeight - 40));
  if (Math.abs(x - r.left) < 0.5 && Math.abs(y - r.top) < 0.5) return;
  Object.assign(d.style, { left: x + "px", top: y + "px", right: "auto" });
  placed.set(d.id, [x, y]);
}
if (typeof window !== "undefined") window.addEventListener("resize", () => document.querySelectorAll("dialog.dlg-panel[open]").forEach(keepInside));

/** The width of a wide panel, from a handle on its left edge (the splitters'
 * pointer, double-click and arrow keys): it grows to the left, over the
 * Inspector and then the view, and stays inside the window. Double-click
 * goes back to the panel's own width. */
const sized = new Map();
function resizeByEdge(d) {
  const edge = document.createElement("div");
  edge.className = "dlg-edge";
  edge.tabIndex = 0;
  edge.setAttribute("role", "separator");
  edge.setAttribute("aria-orientation", "vertical");
  edge.setAttribute("aria-label", "Panel width");
  edge.title = "Drag to resize the panel; double-click for its own width";
  d.appendChild(edge);
  const max = () => Math.max(320, innerWidth - 32);
  const apply = (w) => {
    const r = d.getBoundingClientRect();
    if (w === null) d.style.width = ""; else d.style.width = w + "px";
    // a dragged panel keeps its right edge where it was (the handle is on the left)
    if (d.style.left) Object.assign(d.style, { left: Math.max(0, r.right - d.getBoundingClientRect().width) + "px" });
  };
  if (sized.has(d.id)) apply(Math.min(sized.get(d.id), max()));
  bindSplitter(edge, { axis: "x", sign: -1, min: () => 320, max, value: () => d.getBoundingClientRect().width, zoom: () => 1,
    set: (w) => { apply(w); sized.set(d.id, w); }, commit: () => keepInside(d),
    reset: () => { apply(null); sized.delete(d.id); keepInside(d); } });
}
