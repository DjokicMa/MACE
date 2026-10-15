/* Small UI pieces shared by the shell: icons, toasts, tooltips, placement,
 * dialogs. Vanilla DOM, light DOM, tokens cascade. */
import { registry, shortcutOf, laterReason } from "./commands.js";
import * as notifications from "./notifications.js";

export const $ = (s, el = document) => el.querySelector(s);
export const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const V = document.querySelector('script[src*="/js/shell.js"]').getAttribute("src").split("/")[1];
export const asset = (name) => `static/${V}/${name}`;
const MACE_ICONS = new Set(["unit-cell", "supercell", "slab", "view-a", "view-b", "view-c", "symmetry", "wyckoff",
  "polyhedron", "add-atom", "replace-atom", "bonds", "boundary", "axes", "projection", "labels"]);
export function icon(name, cls = "i") {
  if (!name) return `<svg class="${cls}" aria-hidden="true"></svg>`;
  const sheet = MACE_ICONS.has(name) ? "mace-icons.svg" : "lucide.svg";
  return `<svg class="${cls}" aria-hidden="true"><use href="${asset("icons/" + sheet)}#${name}"/></svg>`;
}

/* ---------------------------------------------------------------- toasts */
/* Plan E13: info and success close after Settings › notifications.infoSeconds
 * (6 s; paused while hovered or focused), warnings and errors stay until
 * closed. The same text and kind while shown counts up ("×3") instead of
 * stacking; more than three shown fold the oldest into "+N more", which opens
 * the history. Every toast is recorded (notifications.js), so closing one
 * loses nothing: Problems › All Notifications lists them. */
const TOAST_ICON = { info: "info", success: "circle-check", warning: "triangle-alert", error: "circle-alert" };
const VISIBLE = 3;
let infoSeconds = () => 6;
let openHistory = () => {};
/** The shell wires the timing setting and the history view in. */
export function configureToasts(opts) {
  if (opts.infoSeconds) infoSeconds = opts.infoSeconds;
  if (opts.openHistory) openHistory = opts.openHistory;
}
function removeToast(el) {
  if (!el || !el.isConnected) return;
  clearTimeout(el._timer);
  el.remove();
  layoutToasts();
}
function layoutToasts() {
  const host = $("#toasts");
  if (!host) return;
  const shown = $$(".toast2", host);
  const hidden = shown.length - VISIBLE;
  shown.forEach((el, k) => { el.hidden = k < hidden; });
  let more = $(".toast-more", host);
  if (hidden > 0) {
    if (!more) {
      more = document.createElement("button");
      more.className = "toast-more";
      more.addEventListener("click", () => openHistory());
      host.prepend(more);
    }
    more.textContent = `+${hidden} more`;
  } else if (more) more.remove();
  let all = $(".toast-dismiss-all", host);
  if (shown.length >= 2) {
    if (!all) {
      all = document.createElement("button");
      all.className = "toast-dismiss-all";
      all.textContent = "Dismiss All";
      all.addEventListener("click", () => { $$(".toast2", host).forEach((t) => { clearTimeout(t._timer); t.remove(); }); layoutToasts(); });
      host.appendChild(all);
    } else host.appendChild(all);           // stays last
  } else if (all) all.remove();
}
function armToast(el, ms) {
  let left = ms, t0 = 0;
  const arm = () => { clearTimeout(el._timer); t0 = Date.now(); el._timer = setTimeout(() => removeToast(el), left); };
  const pause = () => { clearTimeout(el._timer); left = Math.max(500, left - (Date.now() - t0)); };
  el._rearm = (fresh) => { left = fresh; if (!el.matches(":hover") && !el.contains(document.activeElement)) arm(); };
  el.addEventListener("mouseenter", pause);
  el.addEventListener("mouseleave", arm);
  el.addEventListener("focusin", pause);
  el.addEventListener("focusout", (e) => { if (!el.contains(e.relatedTarget)) arm(); });
  arm();
}
export function toast(message, kind = "info", action = null) {
  kind = TOAST_ICON[kind] ? kind : "info";
  notifications.record(message, kind, action);
  const host = $("#toasts");
  const key = `${kind}\u0000${message}`;
  const dup = $$(".toast2", host).find((t) => t._key === key);
  if (dup) {
    dup._count = (dup._count || 1) + 1;
    $(".count", dup).textContent = `×${dup._count}`;
    host.insertBefore(dup, $(".toast-dismiss-all", host));    // the newest goes to the bottom
    if (dup._rearm) dup._rearm((action && action.ms) || infoSeconds() * 1000);
    layoutToasts();
    return dup;
  }
  const el = document.createElement("div");
  el.className = `toast2 ${kind}`;
  el._key = key;
  el.tabIndex = -1;
  el.setAttribute("role", kind === "error" ? "alert" : "status");
  el.innerHTML = `${icon(TOAST_ICON[kind] || "info")}<div class="msg"></div><div class="acts"><span class="count"></span></div>`;
  $(".msg", el).textContent = message;
  for (const a of action ? [action, action.extra].filter(Boolean) : []) {
    const b = document.createElement("button");
    b.textContent = a.label;
    b.addEventListener("click", () => { a.run(); removeToast(el); });
    $(".acts", el).insertBefore(b, $(".count", el));
  }
  const close = document.createElement("button");
  close.className = "toast-close";
  close.setAttribute("aria-label", "Dismiss");
  close.innerHTML = icon("x");
  close.addEventListener("click", () => removeToast(el));
  $(".acts", el).appendChild(close);
  el.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.stopPropagation(); removeToast(el); } });
  host.insertBefore(el, $(".toast-dismiss-all", host));
  if (kind === "info" || kind === "success") armToast(el, (action && action.ms) || infoSeconds() * 1000);
  layoutToasts();
  return el;
}
/* ---------------------------------------------------------------- placement */
/** Where a menu or popover goes, positioned fixed with place(): the overlay
 * layer at the end of <body>, outside every panel, so no panel's overflow
 * clips it and no panel's stacking context covers it (the z-index scale in
 * tokens.css orders it; shell.css gives it the UI scale's zoom); or, while a modal dialog is open, that dialog: the
 * browser draws it above the whole page and makes the rest inert, so a menu
 * opened from it must live inside it to show and take clicks. */
export function overlayHost() {
  const modal = $$("dialog[open]").reverse().find((d) => d.matches(":modal"));
  if (modal) return modal;
  let layer = document.getElementById("overlay-layer");
  if (!layer) {
    layer = document.createElement("div");
    layer.id = "overlay-layer";
    document.body.appendChild(layer);
  }
  return layer;
}

/** Put a floating element next to an anchor rect (screen px, a getBoundingClientRect()),
 * flipping and shifting to stay on screen. The overlay layer carries the UI scale's zoom:
 * the element's left and top, its size, the gap and the margins are its own px, so the
 * rect and the window are divided by that zoom. */
export function place(el, rect, side = "bottom", gap = 4) {
  const z = window.maceZoom ? window.maceZoom.of(el) : 1;
  const w = el.offsetWidth, h = el.offsetHeight, W = innerWidth / z, H = innerHeight / z;
  rect = { left: rect.left / z, right: rect.right / z, top: rect.top / z, bottom: rect.bottom / z };
  let x, y;
  if (side === "right") { x = rect.right + gap; y = rect.top; if (x + w > W - 4) x = rect.left - w - gap; }
  else if (side === "top") { x = rect.left; y = rect.top - h - gap; if (y < 4) y = rect.bottom + gap; }
  else { x = rect.left; y = rect.bottom + gap; if (y + h > H - 4) y = Math.max(4, rect.top - h - gap); }
  x = Math.min(Math.max(4, x), W - w - 4);
  y = Math.min(Math.max(4, y), H - h - 4);
  el.style.left = x + "px"; el.style.top = y + "px";
}

/* ---------------------------------------------------------------- tooltips */
/* [data-cmd] buttons get "Title  Shortcut" from the registry; [data-tip] any text.
 * 500 ms on hover (0 once one is showing), at once on keyboard focus. */
let tipTimer = 0, tipFor = null, tipShownAt = 0;
export function tooltipText(el) {
  const id = el.dataset.cmd;
  if (id && registry.has(id)) {
    const c = registry.get(id);
    return { title: c.tipTitle ? c.tipTitle() : c.title, keys: shortcutOf(id),
      desc: c.later ? laterReason(c) : (c.desc || (el.getAttribute("aria-disabled") === "true" ? el.dataset.reason || "" : "")) };
  }
  // a split button names its commands and keys from the registry, read now (a rebinding shows at once)
  if (el.dataset.tipGroup) {
    const ids = el.dataset.tipGroup.split(",").filter((x) => registry.has(x));
    const keys = ids.map((x) => shortcutOf(x)).filter(Boolean);
    const names = ids.map((x) => { const c = registry.get(x); return typeof c.menuTitle === "string" ? c.menuTitle : c.title; });
    return { title: el.dataset.tip, keys: keys.slice(0, 3).join(" / "),
      desc: ids.map((x, k) => `${names[k]}${shortcutOf(x) ? " " + shortcutOf(x) : ""}`).join(", ") };
  }
  if (el.dataset.tipCmd) return { title: el.dataset.tip, keys: shortcutOf(el.dataset.tipCmd), desc: el.dataset.desc || "" };
  return { title: el.dataset.tip, keys: "", desc: el.dataset.desc || "" };
}
function showTip(el) {
  const tip = $("#tooltip");
  const t = tooltipText(el);
  if (!t.title) return;
  tip.innerHTML = `<span><span class="tip-title"></span>${t.desc ? '<span class="tip-desc"></span>' : ""}</span>` +
    (t.keys ? `<kbd>${esc(t.keys)}</kbd>` : "");
  $(".tip-title", tip).textContent = t.title;
  if (t.desc) $(".tip-desc", tip).textContent = t.desc;
  tip.hidden = false;
  place(tip, el.getBoundingClientRect(), el.closest(".activity") ? "right" : "bottom", 6);
  tipFor = el; tipShownAt = performance.now();
}
/** A menu, dialog or palette is about to open: tooltips and open menus go. */
export function overlayOpening() { hideTip(); document.dispatchEvent(new Event("mace-overlay")); }
export function hideTip() { clearTimeout(tipTimer); $("#tooltip").hidden = true; tipFor = null; }
export function installTooltips() {
  document.addEventListener("pointerover", (e) => {
    const el = e.target.closest("[data-cmd],[data-tip]");
    if (!el || el === tipFor) return;
    if (el.closest(".menu, dialog")) return;
    clearTimeout(tipTimer);
    const warm = !$("#tooltip").hidden || performance.now() - tipShownAt < 300;
    tipTimer = setTimeout(() => showTip(el), warm ? 0 : 500);
  });
  document.addEventListener("pointerout", (e) => {
    const el = e.target.closest("[data-cmd],[data-tip]");
    if (el && !el.contains(e.relatedTarget)) { if (!$("#tooltip").hidden) tipShownAt = performance.now(); hideTip(); }
  });
  document.addEventListener("focusin", (e) => {
    const el = e.target.closest && e.target.closest("[data-cmd],[data-tip]");
    if (el && e.target.matches(":focus-visible") && !el.closest(".menu, dialog")) showTip(el);
  });
  document.addEventListener("focusout", hideTip);
  document.addEventListener("pointerdown", hideTip, true);
  // a tooltip never outlives the layout it was placed in, or a key press
  for (const ev of ["keydown", "wheel", "resize", "blur"]) addEventListener(ev, hideTip, true);
  document.addEventListener("scroll", hideTip, true);
}

/* ---------------------------------------------------------------- dialogs */
/** A <dialog> with a 48px header, a body and a footer; focus returns to the opener. */
export function dialog({ id, title, body, foot = "", wide = false, narrow = false, onClose = null }) {
  overlayOpening();
  let d = document.getElementById(id);
  const opener = document.activeElement;
  if (!d) {
    d = document.createElement("dialog");
    d.id = id; d.className = "dlg";
    document.body.appendChild(d);
  }
  d.style.width = wide ? "min(1200px, 94vw)" : "";
  d.classList.toggle("narrow", narrow);
  d.innerHTML = `<div class="dlg-head"><h2></h2><button class="icon-btn sm" data-close aria-label="Close">${icon("x")}</button></div>
    <div class="dlg-body"></div>${foot ? `<div class="dlg-foot">${foot}</div>` : ""}`;
  $("h2", d).textContent = title;
  const b = $(".dlg-body", d);
  if (typeof body === "string") b.innerHTML = body; else b.appendChild(body);
  $("[data-close]", d).addEventListener("click", () => d.close());
  d.addEventListener("close", () => { if (onClose) onClose(); if (opener && opener.focus) opener.focus(); }, { once: true });
  d.showModal();
  // start in the first field; a dialog with none takes focus itself, so the
  // close button does not open with a focus ring
  const first = $(".dlg-body :is(input, select, textarea)", d);
  if (first) first.focus(); else { d.tabIndex = -1; d.focus(); }
  return d;
}
