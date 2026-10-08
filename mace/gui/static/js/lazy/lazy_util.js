/* Shared by the lazy modules: their stylesheet loads with them, once. */
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
 * focus goes back to where it was. Opening it again reuses it. */
export function panel(H, { id, title, body, foot = "", onClose = null, wide = false }) {
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
  d.querySelector("[data-close]").addEventListener("click", () => d.close());
  dragByHead(d);
  d.onkeydown = (e) => { if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); d.close(); } };
  d.addEventListener("close", () => {
    if (onClose) onClose();
    if (opener && opener.focus && opener.isConnected) opener.focus();
  }, { once: true });
  H.overlayOpening();
  d.show();
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
