/* The small selection dialogs (spec §2.8): within a distance (of each
 * selected atom or of their centroid, periodic images included by default;
 * the work is done in the geometry worker), by index ("1-4, 9, 12-"), by a
 * fractional range ("z > 0.5"), and naming the selection as a set. */
import { loadCss, panel } from "./lazy_util.js";

const foot = (go) => `<button class="tb-btn-text" data-act="cancel">Cancel</button><button class="tb-btn-text primary" data-act="apply">${go}</button>`;
function wire(d, apply) {
  d.querySelector('[data-act="cancel"]').addEventListener("click", () => d.close());
  const go = () => { if (apply() !== false) d.close(); };
  d.querySelector('[data-act="apply"]').addEventListener("click", go);
  d.querySelector(".dlg-body").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); go(); } });
}

export async function within(H) {
  await loadCss(import.meta.url, "../../css/dialogs.css");
  const { ed, E } = H.editor();
  if (!ed) return null;
  const body = document.createElement("div");
  body.innerHTML = `<label class="f">Radius (Å)<input class="field" data-k="r" inputmode="decimal" value="3.0" aria-label="Radius"></label>
    <div class="row"><label class="tog"><input type="radio" name="sw-from" value="each" checked><span>from each selected atom</span></label>
      <label class="tog"><input type="radio" name="sw-from" value="centroid"><span>from the centroid</span></label></div>
    <div class="row"><label class="tog"><input type="checkbox" class="sw-img" checked><span>Include periodic images</span></label></div>
    <div class="err" role="alert"></div>`;
  const d = panel(H, { id: "sel-within", title: "Select Within Distance", body, foot: foot("Select") });
  wire(d, () => {
    const r = Number(body.querySelector('[data-k="r"]').value);
    if (!(r > 0)) { body.querySelector(".err").textContent = "The radius is a positive number"; return false; }
    if (!ed.selection.size) { body.querySelector(".err").textContent = "Select atoms first"; return false; }
    E.selectWithin(ed, r, body.querySelector('input[name="sw-from"]:checked').value === "centroid", body.querySelector(".sw-img").checked);
  });
  return d;
}

export async function byIndex(H) {
  await loadCss(import.meta.url, "../../css/dialogs.css");
  const G = await import("../geom/core.js");
  const { ed, E } = H.editor();
  if (!ed) return null;
  const body = document.createElement("div");
  body.innerHTML = `<label class="f">Atoms (1 to ${ed.n})<input class="field" data-k="q" placeholder="1-4, 9, 12-" aria-label="Atom numbers"></label>
    <div class="err" role="alert"></div>`;
  const d = panel(H, { id: "sel-index", title: "Select by Index", body, foot: foot("Select") });
  wire(d, () => {
    const r = G.parseIndexList(body.querySelector('[data-k="q"]').value, ed.n);
    if (!r.ok) { body.querySelector(".err").textContent = r.error; body.querySelector("input").setAttribute("aria-invalid", "true"); return false; }
    E.api.select(ed, r.indices, "replace");
  });
  return d;
}

export async function byFrac(H) {
  await loadCss(import.meta.url, "../../css/dialogs.css");
  const G = await import("../geom/core.js");
  const { ed, E } = H.editor();
  if (!ed) return null;
  const body = document.createElement("div");
  body.innerHTML = `<label class="f">Condition<input class="field" data-k="q" placeholder="z > 0.5" aria-label="Fractional range"></label>
    <p class="note">Fractional x, y, z (or a, b, c) in [0, 1); join conditions with "and".</p><div class="err" role="alert"></div>`;
  const d = panel(H, { id: "sel-frac", title: "Select by Fractional Range", body, foot: foot("Select") });
  wire(d, () => {
    const r = G.parseFracRange(body.querySelector('[data-k="q"]').value);
    if (!r.ok) { body.querySelector(".err").textContent = r.error; body.querySelector("input").setAttribute("aria-invalid", "true"); return false; }
    E.api.select(ed, ed.structure.frac.map((f, i) => (r.test(f) ? i : -1)).filter((i) => i >= 0), "replace");
  });
  return d;
}

export async function saveSet(H) {
  await loadCss(import.meta.url, "../../css/dialogs.css");
  const { ed } = H.editor();
  if (!ed) return null;
  if (!ed.selection.size) { H.toast("Select atoms first", "info"); return null; }
  const body = document.createElement("div");
  body.innerHTML = `<label class="f">Name<input class="field" data-k="n" value="Set ${Object.keys(ed.selectionSets).length + 1}" aria-label="Set name"></label>
    <p class="note">${ed.selection.size} atoms; the set is kept with this document (Select › Selection Sets).</p><div class="err" role="alert"></div>`;
  const d = panel(H, { id: "sel-save", title: "Save Selection as Set", body, foot: foot("Save") });
  wire(d, () => {
    const name = body.querySelector('[data-k="n"]').value.trim();
    if (!name) { body.querySelector(".err").textContent = "Give the set a name"; return false; }
    ed.selectionSets[name] = [...ed.selection].sort((a, b) => a - b);
    H.toast(`Saved the selection as "${name}"`, "success");
    H.refresh();
  });
  return d;
}
