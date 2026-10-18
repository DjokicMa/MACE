/* View › Element Colors… (1.1.6): edit the colors atoms are drawn in.
 *
 * A palette is a preset of element_colors.json (VESTA, the MACE default;
 * Jmol / CPK; Colorblind safe) or one of the user's own, saved in
 * ~/.config/mace/element_colors.json through /api/palettes (format:
 * mace/utils/element_palettes.py; mace plotting reads the same file). A
 * user palette is a base preset plus the elements it changes, so "Reset"
 * on an element gives it back its base color.
 *
 * While the panel is open every view (structures, the editor, FREQ modes,
 * volumes, Compare, slab previews, exports) draws the palette being edited:
 * the draft. Closing it goes back to the default palette; Set as Default
 * makes the shown palette the default (Settings › Viewport › Color scheme,
 * and the CLI's). Export and Import move a palette as a JSON file. */
import { loadCss, panel } from "./lazy_util.js";
import { place } from "./ptable.js";

const SYM = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr " +
  "Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb " +
  "Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr").split(" ");
const HEX = /^#[0-9a-f]{6}$/;

let H, dlg = null, frame = 0;
/* the palette on screen: its id (a preset, "user:<name>", or "" when it is new), base and changed elements */
let ed = null;

const M = () => H.M();
const schemes = () => M().state.colors.schemes;
const userPals = () => (M().state.palettes && M().state.palettes.palettes) || {};
const presetIds = () => M().paletteIds().filter((k) => !k.startsWith("user:"));
const draftMap = () => ({ ...schemes()[ed.base], ...ed.colors });
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const dirty = () => !same({ base: ed.base, colors: sorted(ed.colors) }, ed.saved);
const sorted = (o) => Object.fromEntries(Object.entries(o).sort(([a], [b]) => (a < b ? -1 : 1)));

function normalize(v) {
  let s = String(v || "").trim().toLowerCase();
  if (/^#?[0-9a-f]{3}$/.test(s)) s = "#" + s.replace("#", "").split("").map((c) => c + c).join("");
  if (/^[0-9a-f]{6}$/.test(s)) s = "#" + s;
  return HEX.test(s) ? s : null;
}
function lum(hex) {
  const c = [1, 3, 5].map((k) => parseInt(hex.slice(k, k + 2), 16) / 255).map((x) => (x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4));
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}
/* black or white text, whichever reads better on the element's color (as the periodic-table picker) */
const textOn = (hex) => ((lum(hex) + 0.05) / 0.05 >= 1.05 / (lum(hex) + 0.05) ? "lt" : "dk");

function load(id) {
  const c = M().state.colors;
  if (id && id.startsWith("user:") && userPals()[id.slice(5)]) {
    const p = userPals()[id.slice(5)];
    ed = { id, name: id.slice(5), base: p.base, colors: { ...(p.colors || {}) }, sel: ed ? ed.sel : "O" };
  } else {
    const pid = schemes()[id] ? id : c.default;
    ed = { id: pid, name: "", base: pid, colors: {}, sel: ed ? ed.sel : "O" };
  }
  ed.saved = { base: ed.base, colors: sorted(ed.colors) };
}

/** Open the panel (or bring it back) on the default palette. */
export async function open(handle) {
  H = handle;
  await loadCss(import.meta.url, "../../css/dialogs.css");
  if (dlg && dlg.open) { dlg.querySelector("#pe-pick").focus(); return dlg; }
  if (!M().state.colors) { H.toast("The element colors have not loaded", "error"); return null; }
  ed = null;
  load(M().schemeName());
  dlg = panel(H, { id: "palette-editor", title: "Element Colors", body: body(), wide: true,
    onClose: () => { cancelAnimationFrame(frame); H.setPaletteDraft(null); dlg = null; } });
  dlg.classList.add("pe-panel");
  wire();
  render();
  preview();
  return dlg;
}
export const isOpen = () => !!(dlg && dlg.open);
/** For tests: what is on screen. */
export const current = () => (ed ? { id: ed.id, name: ed.name, base: ed.base, colors: { ...ed.colors }, dirty: dirty(), sel: ed.sel } : null);

function body() {
  const { esc } = H;
  return `<div class="pe-top">
      <label class="f" title="✓ marks the default">Palette<select class="field" id="pe-pick" aria-label="Palette"></select></label>
      <label class="f">Based on<select class="field" id="pe-base" aria-label="Based on"></select></label>
      <button type="button" class="tb-btn-text" data-pe="default" id="pe-default">Set as Default</button>
    </div>
    <div class="pt-grid pe-grid" role="group" aria-label="Element colors (arrow keys move, Enter picks)">${SYM.map((s, k) => {
      const p = place(k + 1);
      // one tab stop for the table (the picked element); the arrow keys move between elements
      return p ? `<button type="button" class="pt-cell pe-cell" data-sym="${s}" data-r="${p[0]}" data-c="${p[1]}" tabindex="-1" style="grid-row:${p[0]};grid-column:${p[1]}" title="${s}">${s}</button>` : "";
    }).join("")}
      <span class="pt-gap pt-fl" aria-hidden="true">57–71</span><span class="pt-gap pt-fa" aria-hidden="true">89–103</span></div>
    <div class="pe-el" aria-live="polite">
      <span class="pe-sym" id="pe-sym"></span>
      <input type="color" id="pe-color" aria-label="Element color">
      <input type="text" class="field pe-hex" id="pe-hex" aria-label="Hex color" spellcheck="false" autocomplete="off" maxlength="7">
      <button type="button" class="tb-btn-text sm" data-pe="reset-el" id="pe-reset-el"></button>
      <button type="button" class="tb-btn-text sm" data-pe="reset-all" id="pe-reset-all">Reset All</button>
    </div>
    <div class="pe-save">
      <label class="f">Name<input type="text" class="field" id="pe-name" maxlength="60" spellcheck="false" autocomplete="off" placeholder="My palette"></label>
      <button type="button" class="tb-btn-text primary" data-pe="save" id="pe-save">Save</button>
      <button type="button" class="tb-btn-text" data-pe="rename" id="pe-rename">Rename</button>
      <button type="button" class="tb-btn-text danger" data-pe="delete" id="pe-delete">Delete</button>
    </div>
    <div class="pe-io">
      <button type="button" class="tb-btn-text" data-pe="export" id="pe-export">${H.icon("download")}<span>Export…</span></button>
      <button type="button" class="tb-btn-text" data-pe="import" id="pe-import">${H.icon("upload")}<span>Import…</span></button>
      <input type="file" accept="application/json,.json" id="pe-file" hidden>
      <span class="pe-confirm" hidden></span>
    </div>
    <p class="pe-msg muted small" id="pe-msg" role="status"></p>
    <p class="pe-where muted small">Saved in <code>${esc((M().state.palettes && M().state.palettes.path) || "~/.config/mace/element_colors.json")}</code>; <code>mace plotting</code> draws atoms in the default palette (or <code>--palette NAME</code>).</p>`;
}

function wire() {
  const $ = (s) => dlg.querySelector(s);
  $("#pe-pick").addEventListener("change", (e) => {
    const was = dirty();
    load(e.target.value);
    render(); preview();
    say(was ? "Unsaved changes were dropped" : "");
  });
  $("#pe-base").addEventListener("change", (e) => {
    ed.base = e.target.value;
    // a change equal to the new base's color is no change
    for (const [s, c] of Object.entries(ed.colors)) if (schemes()[ed.base][s] === c) delete ed.colors[s];
    render(); preview();
  });
  dlg.querySelector(".pe-grid").addEventListener("click", (e) => {
    const b = e.target.closest(".pe-cell");
    if (b) { ed.sel = b.dataset.sym; renderEl(); }
  });
  dlg.querySelector(".pe-grid").addEventListener("keydown", (e) => {
    const b = e.target.closest(".pe-cell");
    if (!b || e.ctrlKey || e.metaKey || e.altKey) return;
    const to = moveFrom(b, e.key);
    if (!to) return;
    e.preventDefault();
    rove(to);
    to.focus();
  });
  $("#pe-color").addEventListener("input", (e) => setColor(ed.sel, e.target.value));
  $("#pe-hex").addEventListener("change", (e) => {
    const v = normalize(e.target.value);
    if (!v) { say("Type a color as #rrggbb, e.g. #d55e00"); e.target.setAttribute("aria-invalid", "true"); return; }
    e.target.removeAttribute("aria-invalid");
    setColor(ed.sel, v);
  });
  $("#pe-hex").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); e.target.dispatchEvent(new Event("change")); } });
  $("#pe-file").addEventListener("change", (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) importFile(f); });
  // the <dialog> outlives a close (panel() reuses it): one handler, never a second
  dlg.onclick = (e) => {
    const b = e.target.closest("[data-pe]");
    if (!b || b.disabled) return;
    act(b.dataset.pe);
  };
}

function setColor(sym, hex) {
  const v = normalize(hex);
  if (!v) return;
  if (schemes()[ed.base][sym] === v) delete ed.colors[sym]; else ed.colors[sym] = v;
  renderCell(sym); renderEl(false); renderState();
  preview();
}

/* every view draws the draft; one redraw per frame while a color picker is dragged */
function preview() {
  cancelAnimationFrame(frame);
  frame = requestAnimationFrame(() => H.setPaletteDraft(draftMap()));
}

/* ---------------------------------------------------------------- render */
function render() {
  const $ = (s) => dlg.querySelector(s), { esc } = H;
  const ids = M().paletteIds();
  $("#pe-pick").innerHTML = ids.map((k) => `<option value="${esc(k)}">${k === M().schemeName() ? "✓ " : ""}${esc(M().paletteLabel(k))}</option>`).join("") +
    (ed.id ? "" : `<option value="" selected>New palette</option>`);
  $("#pe-pick").value = ed.id;
  $("#pe-base").innerHTML = presetIds().map((k) => `<option value="${esc(k)}">${esc(M().paletteLabel(k))}</option>`).join("");
  $("#pe-base").value = ed.base;
  $("#pe-name").value = ed.name;
  SYM.forEach(renderCell);
  renderEl();
  renderState();
}
/** The cell an arrow key (Home, End) goes to from `b`: along the atomic numbers left and right,
 * the nearest cell in the row above or below (the gaps of the table skipped). */
function moveFrom(b, key) {
  const cells = [...dlg.querySelectorAll(".pe-cell")], i = cells.indexOf(b);
  if (key === "ArrowRight") return cells[i + 1] || null;
  if (key === "ArrowLeft") return cells[i - 1] || null;
  if (key === "Home") return cells[0];
  if (key === "End") return cells[cells.length - 1];
  if (key !== "ArrowUp" && key !== "ArrowDown") return null;
  const r = Number(b.dataset.r), c = Number(b.dataset.c), d = key === "ArrowDown" ? 1 : -1;
  for (let row = r + d; row >= 1 && row <= 12; row += d) {
    const inRow = cells.filter((x) => Number(x.dataset.r) === row);
    if (inRow.length) return inRow.reduce((a, x) => (Math.abs(Number(x.dataset.c) - c) < Math.abs(Number(a.dataset.c) - c) ? x : a));
  }
  return null;
}
/** The table's one tab stop. */
function rove(cell) {
  dlg.querySelectorAll(".pe-cell[tabindex='0']").forEach((x) => { if (x !== cell) x.setAttribute("tabindex", "-1"); });
  if (cell) cell.setAttribute("tabindex", "0");
}
function renderCell(sym) {
  const b = dlg.querySelector(`.pe-cell[data-sym="${sym}"]`);
  if (!b) return;
  const c = draftMap()[sym] || M().state.colors.fallback;
  b.style.background = c;
  b.classList.remove("lt", "dk");
  b.classList.add(textOn(c));
  b.classList.toggle("mod", sym in ed.colors);
  b.classList.toggle("sel", sym === ed.sel);
  b.setAttribute("aria-pressed", String(sym === ed.sel));
  b.title = `${sym}: ${c}${sym in ed.colors ? ` (changed; ${M().paletteLabel(ed.base)} has ${schemes()[ed.base][sym]})` : ""}`;
  // what a screen reader says: the element and its color (the color itself is not announced)
  b.setAttribute("aria-label", `${sym} ${c}${sym in ed.colors ? ", changed" : ""}`);
}
function renderEl(fields = true) {
  const $ = (s) => dlg.querySelector(s), s = ed.sel, c = draftMap()[s] || M().state.colors.fallback;
  dlg.querySelectorAll(".pe-cell.sel").forEach((b) => { if (b.dataset.sym !== s) { b.classList.remove("sel"); b.setAttribute("aria-pressed", "false"); } });
  const cell = dlg.querySelector(`.pe-cell[data-sym="${s}"]`);
  if (cell) { cell.classList.add("sel"); cell.setAttribute("aria-pressed", "true"); if (!dlg.querySelector(".pe-cell[tabindex='0']") || !dlg.querySelector(".pe-grid").contains(document.activeElement)) rove(cell); }
  $("#pe-sym").textContent = s;
  if (fields || $("#pe-color").value !== c) $("#pe-color").value = c;
  if (fields || document.activeElement !== $("#pe-hex")) $("#pe-hex").value = c;
  const base = schemes()[ed.base][s];
  $("#pe-reset-el").textContent = `Reset to ${M().paletteLabel(ed.base).replace(/ \(.*\)$/, "")} (${base})`;
  $("#pe-reset-el").disabled = !(s in ed.colors);
}
function renderState() {
  const $ = (s) => dlg.querySelector(s), user = ed.id.startsWith("user:");
  const n = Object.keys(ed.colors).length, isDefault = ed.id && ed.id === M().schemeName();
  $("#pe-reset-all").disabled = !n;
  $("#pe-rename").disabled = !user || dirty();
  $("#pe-rename").title = !user ? "Only your own palettes can be renamed" : dirty() ? "Save the changes first" : "";
  $("#pe-delete").disabled = !user;
  $("#pe-default").disabled = !ed.id || dirty() || isDefault;
  $("#pe-default").title = isDefault ? "This is the default" : dirty() || !ed.id ? "Save the changes first" : "Every view and mace plotting draw atoms in it";
  $("#pe-save").textContent = user ? "Save" : "Save as New";
  dlg.dataset.dirty = String(dirty());
  const what = `${n} element${n === 1 ? "" : "s"} changed from ${M().paletteLabel(ed.base)}`;
  if (!$("#pe-msg").dataset.sticky) $("#pe-msg").textContent = dirty() ? `${what} · not saved` : (n ? what : "");
}
function say(t, sticky = false) {
  const m = dlg.querySelector("#pe-msg");
  m.textContent = t;
  if (sticky) m.dataset.sticky = "1"; else delete m.dataset.sticky;
  if (!t) renderState();
}

/* ---------------------------------------------------------------- actions */
async function post(body) {
  const v = await H.api("palettes", { body });
  H.setPalettes(v);
  return v;
}
async function act(what) {
  const $ = (s) => dlg.querySelector(s);
  const conf = $(".pe-confirm");
  try {
    if (what === "reset-el") { delete ed.colors[ed.sel]; renderCell(ed.sel); renderEl(); renderState(); preview(); return; }
    if (what === "reset-all") { const had = Object.keys(ed.colors); ed.colors = {}; had.forEach(renderCell); renderEl(); renderState(); preview(); return; }
    if (what === "default") {
      H.settings.set("colorScheme", ed.id);       // the shell saves it in element_colors.json too
      render(); say(`${M().paletteLabel(ed.id)} is the default`, true);
      return;
    }
    if (what === "save") {
      const name = $("#pe-name").value.trim().replace(/\s+/g, " ");
      if (!name) { say("Name the palette first", true); $("#pe-name").focus(); return; }
      const mine = ed.id === `user:${name}`;
      if (!mine && userPals()[name]) { say(`A palette named “${name}” exists: pick another name, or open it to change it`, true); return; }
      const v = await post({ op: "save", name, base: ed.base, colors: ed.colors, replace: mine });
      load(v.id); render(); preview();
      say(`Saved “${name}”`, true);
      return;
    }
    if (what === "rename") {
      const to = $("#pe-name").value.trim().replace(/\s+/g, " ");
      if (!to || to === ed.name) { say("Type the new name first", true); $("#pe-name").focus(); return; }
      const wasDefault = M().schemeName() === ed.id;
      const v = await post({ op: "rename", name: ed.name, to });
      if (wasDefault) H.settings.set("colorScheme", v.id);
      load(v.id); render();
      say(`Renamed to “${to}”`, true);
      return;
    }
    if (what === "delete") {
      conf.hidden = false;
      conf.innerHTML = `Delete “${H.esc(ed.name)}”? <button type="button" class="tb-btn-text danger" data-pe="delete-yes">Delete</button>` +
        `<button type="button" class="tb-btn-text" data-pe="delete-no">Keep</button>`;
      conf.querySelector("button").focus();
      return;
    }
    if (what === "delete-no") { conf.hidden = true; return; }
    if (what === "delete-yes") {
      conf.hidden = true;
      const gone = ed.name, wasDefault = M().schemeName() === ed.id;
      await post({ op: "delete", name: gone });
      if (wasDefault) H.settings.set("colorScheme", M().state.colors.default);
      load(M().schemeName()); render(); preview();
      say(`Deleted “${gone}”`, true);
      return;
    }
    if (what === "export") { exportPalette(); return; }
    if (what === "import") { $("#pe-file").click(); return; }
  } catch (e) { say(e.message, true); }
}

/** The palette on screen as a file: the changes, and every element's color for other programs. */
export function exportDoc() {
  const name = ed.name || $name() || M().paletteLabel(ed.base);
  return { mace_element_palette: 1, name, base: ed.base, colors: sorted(ed.colors), elements: sorted(draftMap()) };
}
const $name = () => (dlg ? dlg.querySelector("#pe-name").value.trim() : "");
function exportPalette() {
  const out = exportDoc();
  const text = JSON.stringify(out, null, 2) + "\n";
  const blob = new Blob([text], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `mace-palette-${out.name.replace(/[^\w.-]+/g, "_").replace(/^_+|_+$/g, "") || "palette"}.json`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  if (window.__maceExportHook) window.__maceExportHook(text, "palette");
  say(`Downloaded ${a.download}`, true);
}
/** Import a palette file (or its parsed object): saved as a new palette and shown. */
export async function importData(data) {
  try {
    const v = await post({ op: "import", palette: data });
    load(v.id); render(); preview();
    say(`Imported as “${v.id.slice(5)}”`, true);
    return v.id;
  } catch (e) { say(`Not imported: ${e.message}`, true); return null; }
}
async function importFile(file) {
  let data;
  try { data = JSON.parse(await file.text()); }
  catch (e) { say(`Not a palette file: ${e.message}`, true); return null; }
  return importData(data);
}
