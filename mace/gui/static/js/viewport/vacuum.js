/* The Vacuum mode (1.1.6): how a slab, polymer or molecule is drawn along its non-periodic directions
 * (a slab's c, a polymer's b and c, a molecule's x, y, z), and how far apart a display range or Build ›
 * Supercell repeats it there. Five modes (the geometry is geom/core.js vacuumAxis):
 *   Bounding box   a small fake cell: the atoms' extent + 1 Å on each side (the default, as before)
 *   Open           no walls: the cell outline in the atoms' mid-plane; nothing repeats (infinite)
 *   File-defined   the vacuum the file defines (a CRYSTAL output's formal 500 Å, an extended XYZ's or a cube's box)
 *   CRYSTAL 500 Å  CRYSTAL's convention
 *   Custom         the atoms' extent + the vacuum you type (Å)
 * Settings › Viewport holds the default; a document keeps its own choice (display_state.js, group
 * "vacuum"), so a slab can be Open while a molecule next to it shows its bounding box. A display range
 * (or a supercell) is 1 along a non-periodic direction unless you ask for more there. Display only:
 * nothing here changes a deck, a CIF or a file. */
import * as DS from "./display_state.js";
import * as G from "../geom/core.js";

export const MODES = [
  { id: "bbox", label: "Bounding box", help: "A small cell around the atoms: their extent plus 1 Å on each side" },
  { id: "open", label: "Open (infinite)", help: "No box walls: the non-periodic direction is drawn open and nothing repeats along it" },
  { id: "cell", label: "File-defined", help: "The vacuum the file defines (a CRYSTAL output's formal 500 Å, an extended XYZ's or a cube's box)" },
  { id: "crystal", label: "CRYSTAL 500 Å", help: "CRYSTAL's convention: 500 Å along each non-periodic direction" },
  { id: "gap", label: "Custom", help: "The atoms' extent plus the vacuum you type, in Å" },
];
const MODE_IDS = MODES.map((m) => m.id);
export const KEY_MODE = "viewport.vacuumMode", KEY_GAP = "viewport.vacuumGap";

/** The document's vacuum ({mode, gap, file}): its own choice, else Settings' default (a cube's: File-defined,
 * its box, which its isosurfaces fill); file: the structure's own vectors. */
export function of(doc, s = null) {
  const v = DS.of(doc);
  const own = doc && KEY_MODE in DS.overrides(doc);
  const want = !own && doc && doc.kind === "volume" ? "cell" : v[KEY_MODE];
  const mode = MODE_IDS.includes(want) ? want : "bbox";
  const gap = Number(v[KEY_GAP]);
  return { mode, gap: Number.isFinite(gap) && gap >= 0 ? gap : 10, file: (s && s.vacuum_vectors) || null };
}
/** "the bounding box", "a 12 Å custom vacuum", ...: what the range's tooltip says the copies are spaced by. */
export function text(vac) {
  const m = (vac && vac.mode) || "bbox";
  if (m === "gap") return `${+vac.gap} Å custom`;
  if (m === "crystal") return "CRYSTAL 500 Å";
  if (m === "cell") return "file-defined";
  if (m === "open") return "open (no repeats)";
  return "bounding-box";
}
const nonPeriodic = (s) => !!(s && s.pbc && !s.pbc.every(Boolean));
/** The spacing vectors a supercell of `s` uses along its non-periodic axes (null where it does not repeat). */
export function vectors(doc, s) { return nonPeriodic(s) ? G.vacuumVectors(s, of(doc, s)) : [null, null, null]; }
/** The open vacuum mode repeats nothing along a non-periodic axis. */
export const repeats = (vac) => !vac || vac.mode !== "open";

let host = null;           // {viewDoc(), structureOf(doc)} from the shell
/** The Inspector's Vacuum control: one select and the Custom gap, per document. */
export function bind(h) {
  host = h;
  const sel = document.getElementById("opt-vacuum-mode"), gap = document.getElementById("opt-vacuum-gap");
  if (!sel || sel.dataset.bound) return;
  sel.dataset.bound = "1";
  sel.querySelectorAll("option").forEach((o) => { const m = MODES.find((x) => x.id === o.value); if (m) o.title = m.help; });
  sel.addEventListener("change", () => {
    const d = host.viewDoc();
    if (d) DS.set(d, KEY_MODE, sel.value);
    sync();
  });
  gap.addEventListener("change", () => {
    const d = host.viewDoc(), x = Number(gap.value);
    if (!d) return;
    if (!(x >= 0 && x <= 1000)) { gap.value = String(of(d).gap); gap.setCustomValidity("0 to 1000 Å"); gap.reportValidity(); return; }
    gap.setCustomValidity("");
    DS.set(d, KEY_GAP, x);
  });
  // the classic views' cell outline (app.js cellBox) asks here for a non-periodic structure's box
  window.MACE = window.MACE || {};
  window.MACE.vacuumBox = vacuumBox;
  sync();
}
/** Show the drawn document's mode; the control is live only for a structure with a non-periodic direction. */
export function sync() {
  const sel = document.getElementById("opt-vacuum-mode"), gap = document.getElementById("opt-vacuum-gap");
  const unit = document.getElementById("opt-vacuum-unit"), wrap = document.getElementById("opt-vacuum");
  if (!sel || !host) return;
  const d = host.viewDoc(), s = d ? host.structureOf(d) : null, v = of(d, s), live = nonPeriodic(s);
  if (sel.value !== v.mode) sel.value = v.mode;
  if (document.activeElement !== gap && Number(gap.value) !== v.gap) gap.value = String(v.gap);
  gap.hidden = unit.hidden = v.mode !== "gap";
  sel.disabled = gap.disabled = !live;
  if (wrap) {
    wrap.setAttribute("aria-disabled", String(!live));
    const m = MODES.find((x) => x.id === v.mode);
    wrap.title = live ? `Vacuum: ${m.label}. ${m.help}. A display range or a supercell repeats it only when you ask for more than 1 along a non-periodic direction`
      : "A 3D crystal has no non-periodic direction (the Vacuum mode applies to slabs, polymers and molecules)";
  }
}
/** {o, C} of a non-periodic structure's box for the classic views (n cells along the periodic axes: one count,
 * or [n1, n2, n3] as the optimization view repeats them); purpose "fit" frames a long vacuum as the atoms' box. */
export function vacuumBox(s, n = 1, purpose = "draw") {
  if (!nonPeriodic(s) || !host) return null;
  const pbc = s.pbc, hi = [0, 1, 2].map((k) => (pbc[k] ? Math.max(1, Number(Array.isArray(n) ? n[k] : n) || 1) : 1));
  const c = G.boxCorners(s, { lo: [0, 0, 0], hi, vacuum: of(host.viewDoc(), s) }, purpose);
  return { o: c[0], C: [G.sub3(c[1], c[0]), G.sub3(c[2], c[0]), G.sub3(c[3], c[0])] };
}
