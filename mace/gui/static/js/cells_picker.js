/* The N × M × L cells picker (1.1.6): one control for "how many cells are
 * drawn" wherever a view repeats its structure (Inspector › Range & boundary
 * of the structure and optimization views, the FREQ Mode section, View ›
 * Display Range › N × M × L…). It replaces the 2×2×2 checkboxes those views
 * had. The structure is not changed: building a real supercell is Build ›
 * Supercell… (lazy/supercell_dialog.js), which the picker can open.
 *
 * Each axis takes 1 to 10. A direction the structure is not periodic along
 * (a slab's c, a polymer's b and c) stays 1 unless the view repeats it at the
 * vacuum spacing (repeatNonPeriodic); a molecule gets no picker. "=" links
 * the periodic axes (off until pressed; remembered per picker while the page
 * is open, as the views rebuild their pickers). Values commit on change
 * (Enter, the spinner, leaving the field). */

export const MAX = 10;
const clamp = (x) => Math.max(1, Math.min(MAX, Math.round(Number(x)) || 1));
const PRESETS = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [2, 2, 1], [3, 3, 1], [4, 4, 1]];
const LINKED = new Map();           // picker id -> "=" pressed

/**
 * opts: { id: prefix of the input ids, cells: [n1, n2, n3], pbc: [bool × 3],
 *   repeatNonPeriodic: bool, onChange(cells), label, build: () => void (optional "Build Supercell…") }
 * Returns the element (null for a molecule); el.setCells(cells) shows new values without onChange.
 */
export function cellsPicker(opts) {
  const pbc = (opts.pbc || [true, true, true]).map(Boolean);
  if (!pbc.some(Boolean)) return null;
  const id = opts.id || "cells", free = (k) => pbc[k] || !!opts.repeatNonPeriodic;
  let cells = (opts.cells || [1, 1, 1]).map(clamp).map((n, k) => (free(k) ? n : 1));
  const per = [0, 1, 2].filter((k) => pbc[k]);
  let linked = !!LINKED.get(id) && per.every((k) => cells[k] === cells[per[0]]);
  const el = document.createElement("div");
  el.className = "ins-row cells-pick";
  el.dataset.cellsPicker = id;
  const presets = PRESETS.filter((p) => [0, 1, 2].every((k) => free(k) || p[k] === 1))
    .filter((p, i, all) => all.findIndex((q) => q.join() === p.join()) === i);
  el.innerHTML = `<span class="ins-k cells-k" id="${id}-k">${opts.label || "Cells"}</span>
    <span class="cells-in" role="group" aria-labelledby="${id}-k">${[0, 1, 2].map((k) =>
      `${k ? '<span class="cells-x" aria-hidden="true">×</span>' : ""}<input class="field cells-n" type="number" inputmode="numeric" min="1" max="${MAX}" step="1" ` +
      `id="${id}-${k}" data-axis="${k}" value="${cells[k]}" aria-label="Cells along ${"abc"[k]}"` +
      `${free(k) ? (pbc[k] ? "" : ` title="Not periodic along ${"abc"[k]}: more than 1 repeats it at the vacuum spacing"`)
        : ` disabled title="Not periodic along ${"abc"[k]}: drawn once"`}>`).join("")}</span>
    <button type="button" class="icon-btn sm cells-link" aria-pressed="${linked}" title="Same along every periodic direction" aria-label="Same along every periodic direction">=</button>
    <select class="field cells-preset" aria-label="Cells presets"><option value="">Presets</option>${presets.map((p) =>
      `<option value="${p.join(",")}">${p.join("×")}</option>`).join("")}</select>` +
    (opts.build ? `<button type="button" class="tb-btn-text sm cells-build" title="Build › Supercell…: a new structure with these cells (the view range only draws them)">Build…</button>` : "");
  const inputs = [...el.querySelectorAll(".cells-n")];
  const show = () => {
    inputs.forEach((inp, k) => { if (document.activeElement !== inp || inp.value === "") inp.value = String(cells[k]); });
    el.querySelector(".cells-link").setAttribute("aria-pressed", String(linked));
    el.dataset.cells = cells.join("x");
  };
  const commit = (next) => {
    next = next.map(clamp).map((n, k) => (free(k) ? n : 1));
    if (next.join() === cells.join()) { show(); return; }
    cells = next;
    show();
    if (opts.onChange) opts.onChange(cells.slice());
  };
  inputs.forEach((inp) => inp.addEventListener("change", () => {
    const k = Number(inp.dataset.axis), v = clamp(inp.value), next = cells.slice();
    next[k] = v;
    if (linked && pbc[k]) per.forEach((j) => { next[j] = v; });
    inp.value = String(v);
    commit(next);
  }));
  inputs.forEach((inp) => inp.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); inp.dispatchEvent(new Event("change")); }
    e.stopPropagation();          // digits and arrows belong to the field, not the viewport's keys
  }));
  el.querySelector(".cells-link").addEventListener("click", () => {
    linked = !linked;
    LINKED.set(id, linked);
    if (linked) { const n = cells[per[0]]; const next = cells.slice(); per.forEach((j) => { next[j] = n; }); commit(next); } else show();
  });
  el.querySelector(".cells-preset").addEventListener("change", (e) => {
    const v = e.target.value;
    e.target.value = "";
    if (!v) return;
    commit(v.split(",").map(Number));
  });
  const b = el.querySelector(".cells-build");
  if (b) b.addEventListener("click", () => opts.build(cells.slice()));
  el.setCells = (c) => {
    cells = (c || [1, 1, 1]).map(clamp).map((n, k) => (free(k) ? n : 1));
    linked = !!LINKED.get(id) && per.every((k) => cells[k] === cells[per[0]]);
    show();
  };
  el.cells = () => cells.slice();
  show();
  return el;
}

/**
 * View › Display Range › N × M × L…: the picker in a small floating panel
 * (lazy_util.panel), for whichever view is drawn. H: { icon, overlayOpening,
 * cells(), pbc(), repeatNonPeriodic(), setCells(cells), build() }.
 */
export async function openDialog(H) {
  const { panel } = await import("./lazy/lazy_util.js");
  const body = document.createElement("div");
  body.className = "cells-dlg";
  const pick = cellsPicker({ id: "cells-dlg", cells: H.cells(), pbc: H.pbc(), repeatNonPeriodic: H.repeatNonPeriodic(),
    label: "Cells", onChange: (c) => H.setCells(c), build: H.build ? () => { d.close(); H.build(); } : null });
  if (!pick) { body.innerHTML = `<p class="note">A molecule has no periodic direction to repeat.</p>`; }
  else {
    body.appendChild(pick);
    body.insertAdjacentHTML("beforeend", `<p class="note">How many cells the view draws along a, b and c (1 to ${MAX}). ` +
      `The structure itself does not change: Build… makes a real supercell.</p>`);
  }
  const d = panel(H, { id: "cells-panel", title: "Display Range", body });
  return d;
}
