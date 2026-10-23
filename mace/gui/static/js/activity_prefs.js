/* The activity bar's own layout (1.1.6): which of its views show and in what
 * order, and whether the bar shows at all. Kept in Settings (so the profile
 * and every browser share it):
 *   activity.order   the command ids in the order the user dragged them to
 *   activity.hidden  the command ids the user hid
 *   activity.visible false hides the whole bar (View › Activity Bar brings it back)
 * Settings (the gear) always stays, at the bottom: it is the way back to all of this.
 * Reorder: drag a button onto another, or right-click › Move Up / Move Down. */
import * as settings from "./settings.js";

/** The items as the shell lists them ([id, icon, on, label]), in the user's order, without the hidden ones. */
export function arrange(items) {
  const order = settings.get("activity.order") || [], hidden = new Set(settings.get("activity.hidden") || []);
  const rank = (id) => { const i = order.indexOf(id); return i < 0 ? order.length + items.findIndex((x) => x[0] === id) : i; };
  return items.filter(([id]) => !hidden.has(id)).sort((a, b) => rank(a[0]) - rank(b[0]));
}
export const shown = () => settings.get("activity.visible") !== false;
export const isHidden = (id) => (settings.get("activity.hidden") || []).includes(id);

export function toggleItem(id) {
  const h = new Set(settings.get("activity.hidden") || []);
  if (h.has(id)) h.delete(id); else h.add(id);
  settings.set("activity.hidden", [...h]);
}
/** Move id by one place (dir −1 up, +1 down) among all known items. */
export function move(id, dir, known) {
  const cur = arrange(known.map((k) => [k])).map(([k]) => k).concat(known.filter((k) => isHidden(k)));
  const all = cur.filter((k, i) => cur.indexOf(k) === i);
  const i = all.indexOf(id), j = i + dir;
  if (i < 0 || j < 0 || j >= all.length) return false;
  [all[i], all[j]] = [all[j], all[i]];
  settings.set("activity.order", all);
  return true;
}
/** Put id where target is (before it moving up, after it moving down). */
export function moveTo(id, target, known) {
  const cur = arrange(known.map((k) => [k])).map(([k]) => k).concat(known.filter((k) => isHidden(k)));
  const all = cur.filter((k, i) => cur.indexOf(k) === i);
  const i = all.indexOf(id), j = all.indexOf(target);
  if (i < 0 || j < 0 || i === j) return false;
  all.splice(i, 1);
  all.splice(j, 0, id);
  settings.set("activity.order", all);
  return true;
}
export function reset() {
  settings.set("activity.order", []);
  settings.set("activity.hidden", []);
  settings.set("activity.visible", true);
}

/* Dragging a button onto another reorders them (HTML drag and drop: the
 * keyboard way is the context menu's Move Up / Move Down). Attached once to the bar. */
export function enableDrag(bar, known) {
  if (!bar || bar.__drag) return;
  bar.__drag = true;
  let from = null;
  const btn = (e) => { const b = e.target.closest && e.target.closest("[data-cmd]"); return b && b.dataset.cmd !== "file.settings" ? b : null; };
  bar.addEventListener("dragstart", (e) => {
    const b = btn(e);
    if (!b) { e.preventDefault(); return; }
    from = b.dataset.cmd;
    e.dataTransfer.effectAllowed = "move";
    try { e.dataTransfer.setData("text/plain", from); } catch (err) { /* some browsers refuse custom data */ }
    b.classList.add("act-dragging");
  });
  bar.addEventListener("dragover", (e) => {
    const b = btn(e);
    if (!from || !b) return;
    e.preventDefault();
    bar.querySelectorAll(".act-drop").forEach((x) => x.classList.remove("act-drop"));
    if (b.dataset.cmd !== from) b.classList.add("act-drop");
  });
  const end = () => { from = null; bar.querySelectorAll(".act-drop, .act-dragging").forEach((x) => x.classList.remove("act-drop", "act-dragging")); };
  bar.addEventListener("drop", (e) => {
    const b = btn(e);
    if (from && b && b.dataset.cmd !== from) { e.preventDefault(); moveTo(from, b.dataset.cmd, known()); }
    end();
  });
  bar.addEventListener("dragend", end);
}
/** After each render: the buttons that can move are draggable. */
export function decorate(bar) {
  if (!bar) return;
  bar.querySelectorAll("[data-cmd]").forEach((b) => { b.draggable = b.dataset.cmd !== "file.settings"; });
}
