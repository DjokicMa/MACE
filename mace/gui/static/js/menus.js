/* The menu bar and popup menus (also the base for context menus). Built
 * from the registry: a menu entry is a command id, {cmd, args} (a command
 * run with arguments, e.g. the atom a context menu was opened on), "-" for a
 * separator, or {label, sub: [...]} for a submenu. Roving focus, arrows, Esc,
 * Alt+letter (the menu.open.* mnemonics in the chord index), F10. */
import { registry, shortcutOf, disabledReason } from "./commands.js";
import { $, $$, esc, icon, place, hideTip } from "./ui.js";

let MENUS = [], ctxFn = () => ({}), runFn = () => {};
let openTop = null;          // the open top-level button
const stack = [];            // open menu elements, outermost first

export function initMenubar(menus, getCtx, run) {
  MENUS = menus; ctxFn = getCtx; runFn = run;
  const bar = $("#menus");
  bar.innerHTML = menus.map((m, i) =>
    `<button class="menu-top" role="menuitem" aria-haspopup="menu" aria-expanded="false" data-menu="${i}" tabindex="${i ? -1 : 0}">${esc(m.label)}</button>`).join("");
  $$(".menu-top", bar).forEach((b) => {
    b.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      if (openTop === b) closeAll(); else openTopMenu(b, false);
    });
    b.addEventListener("pointerenter", () => { if (openTop && openTop !== b) openTopMenu(b, false); });
    b.addEventListener("keydown", (e) => {
      if (["ArrowDown", "Enter", " "].includes(e.key)) { e.preventDefault(); openTopMenu(b, true); }
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") { e.preventDefault(); moveTop(b, e.key === "ArrowRight" ? 1 : -1, false); }
    });
  });
  document.addEventListener("pointerdown", (e) => {
    if (stack.length && !e.target.closest(".menu") && !e.target.closest(".menu-top")) closeAll();
  }, true);
  addEventListener("blur", () => closeAll());
  // a real window resize only: the shell's own relayout sends a synthetic one so viewers re-measure,
  // which must not shut a menu opened in the same frame
  addEventListener("resize", (e) => { if (e.isTrusted) closeAll(); });
}

/** Open top-level menu i with its first item focused (a mnemonic's action). */
export function openMenuIndex(i) {
  const b = $(`.menu-top[data-menu="${i}"]`);
  if (b) openTopMenu(b, true);
}
/** F10: focus the first top-level item without opening it (GTK's convention). */
export function focusBar() {
  closeAll();
  const all = $$(".menu-top");
  all.forEach((x, k) => x.tabIndex = k ? -1 : 0);
  if (all[0]) all[0].focus();
}

function moveTop(b, d, open) {
  const all = $$(".menu-top");
  const next = all[(all.indexOf(b) + d + all.length) % all.length];
  all.forEach((x) => x.tabIndex = -1);
  next.tabIndex = 0;
  if (open) openTopMenu(next, true); else next.focus();
}

export function openTopMenu(b, focusFirst) {
  closeAll();
  ctxReturn = null;                 // a menubar menu returns focus its own way
  hideTip();
  performance.mark("ui:menu:start");
  openTop = b;
  b.setAttribute("aria-expanded", "true");
  const m = MENUS[Number(b.dataset.menu)];
  const el = buildMenu(m.items, m.label);
  el.dataset.top = b.dataset.menu;
  place(el, b.getBoundingClientRect(), "bottom", 2);
  requestAnimationFrame(() => { performance.mark("ui:menu:end"); performance.measure("ui:menu", "ui:menu:start", "ui:menu:end"); });
  if (focusFirst) focusItem(el, 0); else el.focus();
}

/** A popup menu at a point (context menus); `args` reach every command in it. */
/* What had focus when a context menu opened (an explorer row, a tab, the 3D
 * view): Esc, or a command that moves focus nowhere, gives it back (spec §5.3). */
let ctxReturn = null;
function giveBack() {
  const r = ctxReturn;
  ctxReturn = null;
  const lost = !document.activeElement || document.activeElement === document.body;
  if (!r || !lost) return;
  // the tab strip and the explorer redraw their rows: find the same one again
  const to = r.el.isConnected ? r.el : r.sel && document.querySelector(r.sel);
  if (to) to.focus();
}
function returnTarget(el) {
  if (!el || el === document.body || el.closest(".menu")) return null;
  // the class keeps a tab apart from its OPEN EDITORS row, which has the same data-doc
  const d = el.dataset || {}, cls = el.classList[0] ? `.${CSS.escape(el.classList[0])}` : "";
  const sel = d.doc !== undefined ? `${cls}[data-doc="${CSS.escape(d.doc)}"]`
    : d.path !== undefined ? `${cls}[data-path="${CSS.escape(d.path)}"]` : "";
  return { el, sel };
}
export function openAt(items, x, y, label = "Context menu", args = null) {
  const from = document.activeElement;
  closeAll();
  ctxReturn = returnTarget(from);
  hideTip();
  performance.mark("ui:context:start");
  const el = buildMenu(items, label, args);
  place(el, { left: x, right: x, top: y, bottom: y }, "bottom", 0);
  // start on the first entry that can run (an atom menu may open with disabled rows on top)
  const rows = $$(".menu-item", el);
  focusItem(el, Math.max(0, rows.findIndex((r) => r.getAttribute("aria-disabled") !== "true")));
  requestAnimationFrame(() => { performance.mark("ui:context:end"); performance.measure("ui:context", "ui:context:start", "ui:context:end"); });
  return el;
}

function buildMenu(items, label, args = null) {
  const ctx = { ...ctxFn(), args };
  const el = document.createElement("div");
  el.className = "menu";
  el.setAttribute("role", "menu");
  el.setAttribute("aria-label", label);
  el.tabIndex = -1;
  el.innerHTML = items.map((it, k) => {
    if (it === "-") return '<div class="menu-sep" role="separator"></div>';
    if (typeof it === "object" && it.sub) {
      return `<button class="menu-item" role="menuitem" aria-haspopup="menu" data-sub="${k}" tabindex="-1">` +
        `${icon(it.icon)}<span class="lbl">${esc(it.label)}</span><span></span>${icon("chevron-right", "i sub")}</button>`;
    }
    if (typeof it === "object" && it.run) {
      return `<button class="menu-item" role="menuitem" data-fn="${k}" tabindex="-1"${it.disabled ? ' aria-disabled="true"' : ""}` +
        `${it.title ? ` title="${esc(it.title)}"` : ""}>${icon(it.icon)}<span class="lbl">${esc(it.label)}</span><kbd></kbd><span></span></button>`;
    }
    const id = typeof it === "string" ? it : it.cmd;
    const c = registry.get(id);
    if (!c) return `<div class="menu-item" aria-disabled="true">missing command ${esc(id)}</div>`;
    const ictx = typeof it === "object" && it.args ? { ...ctx, args: it.args } : ctx;
    const why = disabledReason(c, ictx);
    const role = c.radio ? "menuitemradio" : c.checked ? "menuitemcheckbox" : "menuitem";
    const checked = c.checked ? ` aria-checked="${!!c.checked(ictx)}"` : "";
    const mt = typeof c.menuTitle === "function" ? c.menuTitle(ictx) : c.menuTitle;
    const title = (typeof it === "object" && it.label) || mt || c.title;
    return `<button class="menu-item" role="${role}" data-cmd-id="${esc(id)}" tabindex="-1"${checked}` +
      `${why ? ` aria-disabled="true" title="${esc(why)}"` : ""}>` +
      `${c.checked ? icon("check", "i check") : icon(c.icon)}<span class="lbl">${esc(title)}</span>` +
      `<kbd>${esc(shortcutOf(id))}</kbd><span></span></button>`;
  }).join("");
  el.__args = args;
  document.body.appendChild(el);
  stack.push(el);
  el.addEventListener("click", (e) => {
    const b = e.target.closest(".menu-item");
    if (!b || b.getAttribute("aria-disabled") === "true") return;
    if (b.dataset.sub !== undefined) { openSub(el, b, items[Number(b.dataset.sub)], true); return; }
    if (b.dataset.fn !== undefined) { closeAll(); items[Number(b.dataset.fn)].run(); giveBack(); return; }
    const id = b.dataset.cmdId;
    const it = items.find((x) => x && (x === id || x.cmd === id));
    closeAll();
    runFn(id, (it && it.args) || args);
    giveBack();
  });
  let hoverTimer = 0;
  el.addEventListener("pointerover", (e) => {
    const b = e.target.closest(".menu-item");
    if (!b) return;
    $$(".menu-item", el).forEach((x) => x.classList.toggle("active", x === b));
    clearTimeout(hoverTimer);
    hoverTimer = setTimeout(() => {
      closeAfter(el);
      if (b.dataset.sub !== undefined) openSub(el, b, items[Number(b.dataset.sub)], false);
    }, 120);
  });
  el.addEventListener("keydown", (e) => menuKey(e, el, items));
  return el;
}

function openSub(parent, b, it, focus) {
  closeAfter(parent);
  const sub = buildMenu(it.sub, it.label, parent.__args || null);
  sub.dataset.parentItem = "1";
  place(sub, b.getBoundingClientRect(), "right", 2);
  b.setAttribute("aria-expanded", "true");
  if (focus) focusItem(sub, 0);
}

function closeAfter(el) {
  while (stack.length && stack[stack.length - 1] !== el) stack.pop().remove();
  $$("[aria-expanded]", el).forEach((x) => x.removeAttribute("aria-expanded"));
}

export function closeAll() {
  while (stack.length) stack.pop().remove();
  if (openTop) openTop.setAttribute("aria-expanded", "false");
  openTop = null;
}
export const isOpen = () => stack.length > 0;

function focusItem(el, i) {
  const items = $$(".menu-item", el);
  if (!items.length) return;
  const k = (i + items.length) % items.length;
  items.forEach((x, j) => x.classList.toggle("active", j === k));
  items[k].focus();
}

function menuKey(e, el, items) {
  const all = $$(".menu-item", el);
  const cur = all.indexOf(document.activeElement);
  const top = el.dataset.top !== undefined ? $(`.menu-top[data-menu="${el.dataset.top}"]`) : null;
  if (e.key === "ArrowDown") { e.preventDefault(); focusItem(el, cur + 1); }
  else if (e.key === "ArrowUp") { e.preventDefault(); focusItem(el, cur - 1); }
  else if (e.key === "Home") { e.preventDefault(); focusItem(el, 0); }
  else if (e.key === "End") { e.preventDefault(); focusItem(el, all.length - 1); }
  else if (e.key === "Escape") {
    e.preventDefault(); e.stopPropagation();
    if (stack.length > 1) { const parent = stack[stack.length - 2]; closeAfter(parent); focusItem(parent, 0); }
    else { closeAll(); if (top) top.focus(); else giveBack(); }
  } else if (e.key === "ArrowRight") {
    e.preventDefault();
    const b = all[cur];
    if (b && b.dataset.sub !== undefined) openSub(el, b, items[Number(b.dataset.sub)], true);
    else if (openTop) moveTop(openTop, 1, true);
  } else if (e.key === "ArrowLeft") {
    e.preventDefault();
    if (stack.length > 1) { const parent = stack[stack.length - 2]; closeAfter(parent); parent.querySelector("[aria-haspopup].active, .menu-item.active")?.focus(); }
    else if (openTop) moveTop(openTop, -1, true);
  } else if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    if (all[cur]) all[cur].click();
  } else if (e.key.length === 1 && /\S/.test(e.key) && !e.ctrlKey && !e.metaKey && !e.altKey) {
    const ch = e.key.toLowerCase();
    const order = all.map((_, j) => all[(cur + 1 + j) % all.length]);
    const hit = order.find((b) => ($(".lbl", b)?.textContent || "").trim().toLowerCase().startsWith(ch));
    if (hit) { e.preventDefault(); focusItem(el, all.indexOf(hit)); }
  }
  e.stopPropagation();
}
