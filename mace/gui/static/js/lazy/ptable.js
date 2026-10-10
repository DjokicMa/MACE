/* The periodic-table picker (spec §2.4), loaded on first use: 18 columns
 * with the lanthanides and actinides below, each cell in the color scheme's
 * color with dark or light text (whichever reads better), the 8 elements
 * used last, and a search field that has focus at once ("F", "e", Enter
 * picks Fe). Arrow keys move; Esc closes and gives focus back. Elements not
 * in `present` (Select by Element) are disabled. */
import { loadCss } from "./lazy_util.js";
import * as profile from "../profile.js";
import { overlayHost } from "../ui.js";

const SYM = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr " +
  "Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb " +
  "Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr").split(" ");
/** [row, column] (1-based) of atomic number z: rows 9 and 10 are the f-block. */
export function place(z) {
  if (z === 1) return [1, 1];
  if (z === 2) return [1, 18];
  const runs = [[3, 4, 2, 1], [5, 10, 2, 13], [11, 12, 3, 1], [13, 18, 3, 13], [19, 36, 4, 1], [37, 54, 5, 1], [55, 56, 6, 1],
    [57, 71, 9, 3], [72, 86, 6, 4], [87, 88, 7, 1], [89, 103, 10, 3]];
  for (const [a, b, row, col] of runs) if (z >= a && z <= b) return [row, col + z - a];
  return null;
}
/** Relative luminance of "#rrggbb" (WCAG). */
function lum(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex || "");
  if (!m) return 0.5;
  const c = [0, 2, 4].map((k) => parseInt(m[1].slice(k, k + 2), 16) / 255).map((x) => (x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4));
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}
const contrast = (a, b) => (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);

let H, pop, cb, opener, cells = [], hi = -1;
export async function open(handle, anchor, onPick, opts = {}) {
  H = handle;
  await loadCss(import.meta.url, "../../css/dialogs.css");
  close(false);
  cb = onPick;
  opener = document.activeElement;
  H.overlayOpening();
  pop = document.createElement("div");
  pop.className = "ptable";
  pop.setAttribute("role", "dialog");
  pop.setAttribute("aria-label", opts.title || "Pick an element");
  const color = (s) => (window.MACE && window.MACE.elementColor ? window.MACE.elementColor(s) : "");
  const present = opts.present ? new Set(opts.present) : null;
  const css = getComputedStyle(document.documentElement);
  const onLight = lum(css.getPropertyValue("--on-light-fill").trim()), onDark = lum(css.getPropertyValue("--on-dark-fill").trim());
  const recent = Array.isArray(opts.recent) ? opts.recent : readRecent();
  const cell = (s, cls = "") => {
    const z = SYM.indexOf(s) + 1, bg = color(s), L = lum(bg);
    const text = contrast(L, onLight) >= contrast(L, onDark) ? "lt" : "dk";
    const off = present && !present.has(s);
    return `<button type="button" class="pt-cell ${text} ${cls}" data-sym="${s}" data-z="${z}" style="background:${bg}"` +
      `${off ? " disabled aria-disabled=\"true\"" : ""} title="${s} (${z})${off ? ": not in this structure" : ""}">${s}</button>`;
  };
  pop.innerHTML = `<div class="pt-head"><span class="pt-title">${H.esc(opts.title || "Element")}</span>
      <input class="pt-search" type="search" placeholder="Type a symbol" aria-label="Element symbol" autocomplete="off" spellcheck="false"></div>
    ${recent.length ? `<div class="pt-recent" aria-label="Recent elements">${recent.filter((s) => SYM.includes(s)).slice(0, 8).map((s) => cell(s, "rec")).join("")}</div>` : ""}
    <div class="pt-grid">${SYM.map((s, k) => { const p = place(k + 1); return p ? cell(s).replace("<button", `<button style-grid="${p[0]}/${p[1]}"`) : ""; }).join("")}
      <span class="pt-gap pt-fl" aria-hidden="true">57–71</span><span class="pt-gap pt-fa" aria-hidden="true">89–103</span></div>
    <div class="pt-foot"><span class="pt-msg" role="status"></span></div>`;
  pop.querySelectorAll("[style-grid]").forEach((b) => {
    const [r, c] = b.getAttribute("style-grid").split("/");
    b.style.gridRow = r; b.style.gridColumn = c;
    b.removeAttribute("style-grid");
  });
  overlayHost().appendChild(pop);   // above every panel, or inside the open modal dialog
  cells = [...pop.querySelectorAll(".pt-grid .pt-cell")];
  const r = (anchor && anchor.getBoundingClientRect) ? anchor.getBoundingClientRect() : { left: 100, right: 100, top: 100, bottom: 100 };
  placePop(r);
  const search = pop.querySelector(".pt-search");
  search.addEventListener("input", () => highlight(search.value));
  search.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); if (hi >= 0) pick(cells[hi].dataset.sym); else msg("Not an element symbol"); }
    else if (e.key === "ArrowDown") { e.preventDefault(); (cells[hi] || cells[0]).focus(); }
    else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(true); }
  });
  pop.addEventListener("click", (e) => { const b = e.target.closest(".pt-cell"); if (b && !b.disabled) pick(b.dataset.sym); });
  pop.addEventListener("keydown", (e) => {
    const b = e.target.closest(".pt-grid .pt-cell");
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(true); return; }
    if (!b) return;
    const d = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] }[e.key];
    if (!d) return;
    e.preventDefault();
    const [r0, c0] = [Number(b.style.gridRowStart), Number(b.style.gridColumnStart)];
    for (let step = 1; step < 19; step++) {
      const t = cells.find((x) => Number(x.style.gridRowStart) === r0 + d[0] * step && Number(x.style.gridColumnStart) === c0 + d[1] * step && !x.disabled);
      if (t) { t.focus(); return; }
    }
  });
  setTimeout(() => document.addEventListener("pointerdown", outside, true), 0);
  search.focus();
  return pop;
}
function placePop(r) {
  const w = pop.offsetWidth, h = pop.offsetHeight;
  let x = r.left, y = r.bottom + 6;
  if (y + h > innerHeight - 8) y = Math.max(8, r.top - h - 6);
  x = Math.min(Math.max(8, x), innerWidth - w - 8);
  pop.style.left = x + "px"; pop.style.top = y + "px";
}
function outside(e) { if (pop && !pop.contains(e.target)) close(false); }
function msg(t) { const m = pop.querySelector(".pt-msg"); if (m) m.textContent = t; }
/** The cell a typed symbol means: an exact match first, then the first one it starts. */
function highlight(text) {
  const q = text.trim().toLowerCase();
  cells.forEach((c) => c.classList.remove("hi"));
  hi = -1;
  if (!q) { msg(""); return; }
  let k = cells.findIndex((c) => c.dataset.sym.toLowerCase() === q && !c.disabled);
  if (k < 0) k = cells.findIndex((c) => c.dataset.sym.toLowerCase().startsWith(q) && !c.disabled);
  if (k < 0) { msg("Not an element symbol"); return; }
  hi = k;
  cells[k].classList.add("hi");
  msg(`Enter picks ${cells[k].dataset.sym}`);
}
function readRecent() { const r = profile.ui("recentElements", []); return Array.isArray(r) ? r : []; }
function pick(sym) {
  profile.setUi("recentElements", [sym, ...readRecent().filter((x) => x !== sym)].slice(0, 8));
  const f = cb;
  close(true);
  if (f) f(sym);
}
export function close(giveFocus = true) {
  document.removeEventListener("pointerdown", outside, true);
  if (!pop) return;
  pop.remove();
  pop = null; cells = []; hi = -1;
  if (giveFocus && opener && opener.focus && opener.isConnected) opener.focus();
}
export const isOpen = () => !!pop;
