/* Context menus (spec §2.2) on atoms, empty space, explorer items and tabs.
 * Eager and small: a right-click must answer within 50 ms. Every entry is a
 * registry command run with arguments ({atom}, {path}, {docId}), so each
 * shows its shortcut and also works from the palette.
 *
 * In the 3D view 3Dmol reports the right-click itself (its
 * userContextMenuHandler fires only when the pointer moved less than a
 * click), with the atom under the pointer; a screen-space nearest-atom test
 * is the fallback (and serves the keyboard). */
import { openAt } from "./menus.js";
import { CONTEXT } from "./actions.js";

const HIT_PX = 14, MAX_HIT_ATOMS = 4000;
let hooks = null, viewerHandled = 0;

/** hooks: {viewer(), structure(), hovered()} */
export function initContext(h) {
  hooks = h;
  const M = window.MACE;
  // 3Dmol's own right-click report, set on every viewer (app.js calls it)
  M.onViewerContext = (atom, x, y, ev, v) => {
    viewerHandled = performance.now();
    // x, y: the viewer's own px (3Dmol's); the menu opens at the screen point
    const cv = v.getCanvas(), r = cv.getBoundingClientRect(), z = window.maceZoom.of(cv);
    openViewport(atom ? atomArg(atom) : null, r.left + x * z, r.top + y * z);
  };
  // the fallback, and any right-click 3Dmol did not report (another viewer, a label)
  document.addEventListener("contextmenu", (e) => {
    const wrap = e.target.closest && e.target.closest(".viewer-wrap");
    if (wrap && wrap.querySelector("canvas")) {
      e.preventDefault();
      if (performance.now() - viewerHandled < 300) return;
      const { x, y } = pointOf(e, wrap);
      openViewport(hit(x, y), x, y);
      return;
    }
    const file = e.target.closest && e.target.closest("#file-list .file[data-path]");
    if (file) {
      e.preventDefault();
      const { x, y } = pointOf(e, file);
      openAt(CONTEXT.explorer, x, y, "Explorer menu", { path: file.dataset.path });
    }
    const folder = e.target.closest && e.target.closest("#file-list .folder[data-dir]");
    if (folder) {
      e.preventDefault();
      const { x, y } = pointOf(e, folder);
      openAt(CONTEXT.folder, x, y, "Folder menu", { dir: folder.dataset.dir });
    }
  });
}

/* A keyboard-opened menu has no pointer position: the element's center. */
function pointOf(e, el) {
  if (e.clientX || e.clientY) return { x: e.clientX, y: e.clientY };
  const r = el.getBoundingClientRect();
  return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
}

const atomArg = (a) => ({ i: a.serial - 1, image: (a.properties && a.properties.shift) || [0, 0, 0] });

/** The atom or empty-space menu at a client point (the structure viewer's
 * pointer controller calls this for a right-click). */
export function openViewportMenu(atom, x, y) { return openViewport(atom, x, y); }

function openViewport(atom, x, y) {
  if (atom) return openAt(CONTEXT.atom, x, y, "Atom menu", { atom });
  // in Zen Mode the chrome is hidden: the viewport menu is one more way out
  const zen = document.getElementById("shell").classList.contains("zen");
  return openAt(zen ? [...CONTEXT.empty, "-", "view.exitZen"] : CONTEXT.empty, x, y, "Viewport menu", { x, y });
}

/** The displayed atom nearest a client point, within HIT_PX, front-most first. */
export function hit(x, y) {
  const v = hooks.viewer(), m = v && (hooks.model ? hooks.model(v) : v.getModel());
  if (!m) return null;
  const atoms = m.selectedAtoms({});
  if (atoms.length > MAX_HIT_ATOMS) return null;    // Phase 3 moves picking to the worker
  // modelToScreen: the canvas's screen corner plus the viewer's own px, which the UI scale zooms
  const cv = v.getCanvas(), r = cv.getBoundingClientRect(), z = window.maceZoom.of(cv);
  let best = null, bestD = HIT_PX;
  for (const a of atoms) {
    const p = v.modelToScreen({ x: a.x, y: a.y, z: a.z });
    const d = Math.hypot(r.left + (p.x - r.left) * z - x, r.top + (p.y - r.top) * z - y);
    if (d < bestD) { bestD = d; best = a; }
  }
  return best ? atomArg(best) : null;
}

/** Shift+F10: the menu of whatever has focus (an explorer row, a tab, the 3D view). */
export function openForFocus() {
  const el = document.activeElement;
  if (!el) return false;
  const file = el.closest("#file-list .file[data-path]");
  const tab = el.closest(".doc-tab");
  const wrap = el.closest(".viewer-wrap");
  const r = el.getBoundingClientRect(), x = r.left + Math.min(r.width / 2, 40), y = r.top + r.height / 2;
  if (file) { openAt(CONTEXT.explorer, x, r.bottom, "Explorer menu", { path: file.dataset.path }); return true; }
  if (tab) { openAt(CONTEXT.tab, x, r.bottom, "Tab menu", { docId: Number(tab.dataset.doc) }); return true; }
  if (wrap) {
    const h = hooks.hovered();
    if (h) {
      const v = hooks.viewer(), a = (hooks.model ? hooks.model(v) : v.getModel()).selectedAtoms({ serial: h.i + 1 })[0];
      const p = a ? v.modelToScreen({ x: a.x, y: a.y, z: a.z }) : { x: r.left + r.width / 2, y: r.top + r.height / 2 };
      openViewport(h, p.x, p.y);
    } else openViewport(null, r.left + r.width / 2, r.top + r.height / 2);
    return true;
  }
  return false;
}

/** For tests: the menu a right-click at (x, y) opens; or one of the menus with arguments. */
export const contextAt = (x, y) => openViewport(hit(x, y), x, y);
export function contextFor(kind, args, x = 200, y = 200) {
  const label = { atom: "Atom menu", empty: "Viewport menu", explorer: "Explorer menu", tab: "Tab menu", folder: "Folder menu" }[kind];
  return openAt(CONTEXT[kind], x, y, label, args || null);
}
