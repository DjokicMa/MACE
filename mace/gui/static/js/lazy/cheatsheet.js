/* The shortcut sheet (? or Mod+/), loaded on first open: four fixed columns
 * (Global, File and Edit, Select and Tools, View; two below 1280px) built
 * from the effective keys, so a rebinding shows here at once. Prints as one
 * landscape page. */
import { loadCss } from "./lazy_util.js";

export const COLUMNS = ["Global", "File and Edit", "Select and Tools", "View"];
const GLOBAL_IDS = new Set(["view.palette", "file.quickOpen", "file.goToDoc", "nav.goToAtom", "help.shortcuts", "file.settings",
  "select.exitTool", "menu.focusBar", "view.explorer", "view.inspector", "view.inspectorDrawer", "view.bottom", "view.zen",
  "view.focusExplorer", "view.focusViewport", "view.focusInspector", "view.database", "view.tasks", "view.themeCycle",
  "view.zoomIn", "view.zoomOut", "view.zoomReset"]);
const LAYOUT_IDS = new Set(["view.explorer", "view.inspector", "view.inspectorDrawer", "view.bottom", "view.zen", "view.focusExplorer",
  "view.focusViewport", "view.focusInspector", "view.database", "view.tasks", "view.themeCycle", "view.zoomIn", "view.zoomOut", "view.zoomReset"]);
const COLUMN_OF = { File: 1, Edit: 1, Build: 1, Calculation: 1, Select: 2, Tools: 2, Trajectory: 2, View: 3, Help: 0 };

/** {column, group} for a command: the cheat sheet's place for it. */
export function placeOf(c) {
  if (GLOBAL_IDS.has(c.id)) return { column: 0, group: LAYOUT_IDS.has(c.id) ? "Layout" : "General" };
  const cat = c.category.split(":")[0];
  return { column: COLUMN_OF[cat] ?? 3, group: cat };
}

let H, dlg;
export async function open(handle) {
  H = handle;
  await loadCss(import.meta.url, "../../css/dialogs.css");
  if (!dlg) build();
  render();
  if (!dlg.open) { H.overlayOpening(); H.opener = document.activeElement; dlg.showModal(); }
  const f = dlg.querySelector("#cheat-filter");
  f.value = ""; filter("");
  f.focus();
  return dlg;
}

function build() {
  dlg = document.createElement("dialog");
  dlg.id = "cheatsheet";
  dlg.className = "dlg cheat-dlg";
  dlg.setAttribute("aria-label", "Keyboard shortcuts");
  dlg.innerHTML = `<div class="dlg-head"><h2>Keyboard shortcuts</h2>
      <input class="field cheat-filter" type="search" id="cheat-filter" placeholder="Filter shortcuts" aria-label="Filter shortcuts" autocomplete="off">
      <button class="icon-btn sm" data-close aria-label="Close">${H.icon("x")}</button></div>
    <div class="dlg-body cheat-body"><div class="cheat-cols"></div><p class="cheat-none" hidden>No shortcuts match.</p></div>
    <div class="dlg-foot"><span class="grow muted">Single-key shortcuts never fire while you type in a field. Esc also leaves Zen Mode.</span>
      <button class="tb-btn-text" id="cheat-edit">${H.icon("keyboard")}<span>Edit keybindings…</span></button>
      <button class="tb-btn-text" id="cheat-print">${H.icon("printer")}<span>Print</span></button></div>`;
  document.body.appendChild(dlg);
  dlg.querySelector("[data-close]").addEventListener("click", () => dlg.close());
  dlg.addEventListener("close", () => { if (H.opener && H.opener.focus) H.opener.focus(); });
  dlg.querySelector("#cheat-filter").addEventListener("input", (e) => filter(e.target.value));
  dlg.querySelector("#cheat-print").addEventListener("click", () => print());
  dlg.querySelector("#cheat-edit").addEventListener("click", () => { dlg.close(); H.openSettings("keyboard"); });
}

/** The rows by column and group, from the effective keys (exported for tests). */
export function model() {
  const cols = COLUMNS.map(() => new Map());
  for (const c of H.commands.registry.values()) {
    if (!c.keys.length) continue;
    const { column, group } = placeOf(c);
    if (!cols[column].has(group)) cols[column].set(group, []);
    cols[column].get(group).push({ id: c.id, title: c.title, keys: c.keys.map(H.commands.formatChord) });
  }
  // the keys that mean something else for a moment (a Move grab, the Add Atom element field)
  for (const [name, l] of Object.entries(H.commands.MODAL_LAYERS || {})) {
    cols[2].set(l.title, l.rows.map(([k, t], n) => ({ id: `layer.${name}.${n}`, title: t, keys: [k] })));
  }
  return cols;
}

function render() {
  const { esc } = H;
  dlg.querySelector(".cheat-cols").innerHTML = model().map((groups, k) => `<section class="cheat-col" aria-label="${esc(COLUMNS[k])}">
    <h3 class="cheat-col-h">${esc(COLUMNS[k])}</h3>${[...groups].map(([g, rows]) => `<div class="cheat-group"><h4>${esc(g)}</h4>${rows.map((r) =>
      `<div class="cheat-row" data-id="${esc(r.id)}" data-text="${esc(`${r.title} ${r.keys.join(" ")} ${g}`.toLowerCase())}">` +
      `<span class="t">${esc(r.title)}</span><span class="k">${r.keys.map((x) => `<kbd>${esc(x)}</kbd>`).join("")}</span></div>`).join("")}</div>`).join("")}</section>`).join("");
}

function filter(q) {
  q = q.trim().toLowerCase();
  let any = false;
  dlg.querySelectorAll(".cheat-row").forEach((r) => { r.hidden = !!q && !r.dataset.text.includes(q); any = any || !r.hidden; });
  dlg.querySelectorAll(".cheat-group").forEach((g) => { g.hidden = !g.querySelector(".cheat-row:not([hidden])"); });
  dlg.querySelectorAll(".cheat-col").forEach((c) => { c.hidden = !c.querySelector(".cheat-row:not([hidden])"); });
  dlg.querySelector(".cheat-none").hidden = any;
}
