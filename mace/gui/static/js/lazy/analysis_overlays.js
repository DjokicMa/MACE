/* Inspector › 3D Overlays (1.1.6): every Structure Analysis overlay on the
 * document in front (pores, a net, a crystal shape, Miller planes), one row
 * each with its visibility and Delete; the row picked shows its own style,
 * opacity and colors (pores: the accessible channels and the closed pockets
 * apart, and each channel or pocket on its own). These start from
 * Settings › Viewport (pores.*) and stay with this document's overlay;
 * Reset takes them back to the Settings. Delete takes the overlay and its
 * mesh data out of memory (the result stays in the Analysis tab). */
import { loadCss } from "./lazy_util.js";
import { store, OVERLAY_TITLE } from "./analysis_store.js";
import * as MQ from "../viewport/mesh_quality.js";

const css = () => loadCss(import.meta.url, "../../css/analysis_results.css");
let ui = null;          // { H, A, el, sel (key of the row picked) }

/** Fill ``el`` (the section's body) and keep it current. */
export async function fill(el, H) {
  await css();
  if (!ui) {
    ui = { H, A: store(H), el, sel: null };
    let frame = 0;
    document.addEventListener("mace-analysis", (e) => {
      const d = e.detail || {};
      if (d.what !== "overlay" && d.what !== "meshes" && d.what !== "remove" && d.what !== "add") return;
      // a slider being dragged redraws the surfaces, not this list
      if (d.why === "override" && ui.el && ui.el.contains(document.activeElement) && document.activeElement.type === "range") return;
      if (d.why === "show" && d.key) ui.sel = d.key;
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => { if (ui.el && ui.el.isConnected) render(); });
    });
  }
  ui.H = H; ui.A = store(H); ui.el = el;
  render();
}

const hex = (c) => {
  const s = String(c || "").trim();
  if (/^#[0-9a-f]{6}$/i.test(s)) return s;
  if (/^#[0-9a-f]{3}$/i.test(s)) return "#" + [...s.slice(1)].map((x) => x + x).join("");
  const m = s.match(/rgba?\(\s*(\d+)[ ,]+(\d+)[ ,]+(\d+)/);
  return m ? "#" + [m[1], m[2], m[3]].map((x) => Number(x).toString(16).padStart(2, "0")).join("") : "";
};
const cssProp = (k) => getComputedStyle(document.documentElement).getPropertyValue(k);
const token = (k) => {
  const M = ui.H.M();
  // an unreadable token falls back to the theme's muted text color
  try { return hex(M && M.vpVar ? M.vpVar(k) : cssProp(k)) || hex(cssProp("--text-3")); } catch (e) { return hex(cssProp("--text-3")); }
};

function render() {
  const { H, A, el } = ui, esc = H.esc, S = H.shell();
  const doc = S.activeDoc();
  const O = A.overlay();
  const list = doc ? O.list(doc.id) : [];
  if (!list.length) {
    el.innerHTML = `<p class="insp-hint">No 3D overlays on ${doc ? esc(doc.title) : "this document"}. Analysis › Structure Analysis draws pores (Porosity),
      a net (Topology), a crystal shape (Morphology) and Miller planes (Planes) here.</p>`;
    return;
  }
  if (!list.some((x) => x.key === ui.sel)) ui.sel = list[0].key;
  el.innerHTML = `<div class="ovm" role="list">${list.map((o) => {
    const rec = o.meta && o.meta.rec ? A.get(o.meta.rec) : null;
    const label = `${OVERLAY_TITLE[o.kind] || o.kind}${o.meta && o.meta.label ? ` · ${o.meta.label}` : ""}`;
    return `<div class="ovm-row" role="listitem" data-key="${esc(o.key)}" aria-current="${o.key === ui.sel}">
      <button type="button" class="icon-btn sm ovm-eye" data-a="eye" aria-pressed="${o.visible}" aria-label="${o.visible ? "Hide" : "Show"} ${esc(label)}" title="${o.visible ? "Hide" : "Show"}">${H.icon(o.visible ? "eye" : "eye-off")}</button>
      <button type="button" class="ovm-name" data-a="pick" title="${o.stale ? "Hidden: the structure was edited since it was computed" : rec ? `From the ${esc(rec.tool)} result of ${new Date(rec.when).toLocaleTimeString()}` : ""}">${esc(label)}${o.stale ? ` <em class="insp-hint">(edited since)</em>` : ""}</button>
      <button type="button" class="icon-btn sm" data-a="delete" aria-label="Delete ${esc(label)} and its mesh data" title="Delete the overlay and its mesh data (the result stays in the Analysis tab)">${H.icon("trash-2")}</button></div>
      ${o.key === ui.sel ? `<div class="ovm-edit" data-key="${esc(o.key)}">${editor(o)}</div>` : ""}`;
  }).join("")}</div>`;
  el.querySelectorAll(".ovm-row [data-a]").forEach((b) => b.addEventListener("click", () => {
    const key = b.closest(".ovm-row").dataset.key, a = b.dataset.a;
    if (a === "pick") { ui.sel = key; render(); }
    else if (a === "eye") { const it = O.get(key); O.setVisible(key, !(it && it.visible)); }
    else if (a === "delete") remove(key);
  }));
  const ed = el.querySelector(".ovm-edit");
  if (ed) bindEditor(ed, O.list(doc.id).find((x) => x.key === ed.dataset.key));
}

/* the controls of the overlay picked */
function editor(o) {
  const esc = ui.H.esc, op = o.opts || {}, ck = (b) => (b ? " checked" : "");
  const ov = (k) => (k in o.over ? " ovm-own" : "");
  const sel = (k, label, opts, v) => `<label class="ovm-f${ov(k)}">${label} <select data-o="${k}">${opts.map(([id, t]) =>
    `<option value="${id}"${id === v ? " selected" : ""}>${esc(t)}</option>`).join("")}</select></label>`;
  const lk = MQ.look(op, o.kind === "pores" ? "auto" : "double");
  const style = () => sel("style", "Style", MQ.STYLES, lk.style) + (o.kind === "pores" ? sel("shading", "Shading", MQ.SHADINGS, lk.shading) : "")
    + sel("sides", o.kind === "pores" ? "Walls" : "Sides", o.kind === "pores" ? MQ.PORE_SIDES : MQ.SIDES, lk.sides);
  // a pore surface's own presets (Preview too), the other overlays' the shared ones
  const quality = () => `<div class="ovm-line">${o.kind === "pores" ? sel("quality", "Mesh quality", MQ.PORE_PRESETS.map((x) => [x, MQ.PRESET_LABEL[x]]), MQ.porePresetOf(op.quality, MQ.PORE_DEFAULT))
    : sel("quality", "Mesh quality", MQ.PRESETS.map((x) => [x, MQ.PRESET_LABEL[x]]), MQ.presetOf(op.quality))}</div>`;
  const opacity = (v, d) => `<label class="ovm-f${ov("opacity")}">Opacity <input type="range" min="0.05" max="1" step="0.05" value="${v ?? d}" data-o="opacity" aria-label="Opacity">
    <output>${Number(v ?? d).toFixed(2)}</output></label>`;
  const color = (k, tok, label, text = false) => `<label class="ovm-f ovm-color${ov(k)}" title="${esc(label)} color${k in o.over ? " (changed here)" : ""}"><input type="color" data-o="${k}" value="${hex(op[k]) || token(tok)}" aria-label="${esc(label)} color">${text ? ` ${esc(label)}` : ""}</label>`;
  let h = "";
  if (o.kind === "pores") {
    h += `<div class="ovm-line"><label class="ovm-f${ov("accessible")}"><input type="checkbox" data-o="accessible"${ck(op.accessible !== false)}> Accessible channels</label>
      ${color("accColor", "--pore-acc", "Channels")}</div>
      <div class="ovm-line"><label class="ovm-f${ov("inaccessible")}"><input type="checkbox" data-o="inaccessible"${ck(op.inaccessible !== false)}> Inaccessible pockets</label>
      ${color("nacColor", "--pore-nac", "Pockets")}</div>
      <div class="ovm-line"><label class="ovm-f${ov("colorPockets")}"><input type="checkbox" data-o="colorPockets"${ck(op.colorPockets)}> Color pockets individually</label></div>
      <div class="ovm-line">${style()}</div><div class="ovm-line">${opacity(op.opacity, 0.5)}</div>${quality()}${meshInfo(o)}`;
    const rec = o.meta && o.meta.rec ? ui.A.get(o.meta.rec) : null;
    const F = rec && rec.result && rec.result.features ? rec.result.features : [];
    if (F.length) {
      h += `<details class="ovm-feats"${F.length <= 8 ? " open" : ""}><summary>Channels and pockets (${F.length})</summary><ul>${F.map((f) =>
        `<li><label><input type="checkbox" data-f="${f.id}"${o.hidden.includes(f.id) ? "" : " checked"}> <span class="an-swatch ${f.accessible ? "acc" : "nac"}"></span>
        ${f.id + 1} ${f.accessible ? `channel, ${f.dim}D` : "pocket"} · ${Number(f.occupiable_A3).toFixed(0)} Å³</label>
        <button type="button" class="icon-btn sm" data-hl="${f.id}" aria-pressed="${o.hl === f.id}" aria-label="Highlight and frame ${f.accessible ? "channel" : "pocket"} ${f.id + 1}" title="Highlight and frame it">${ui.H.icon("scan-search")}</button></li>`).join("")}</ul></details>`;
    }
  } else if (o.kind === "net") {
    h += `<div class="ovm-line">${color("color", "--accent", "Nodes and links", true)}${opacity(op.opacity, 1)}</div>${quality()}`;
  } else if (o.kind === "mesh") {
    h += `<div class="ovm-line">${color("color", "--accent", "Shape", true)}</div><div class="ovm-line">${style()}</div><div class="ovm-line">${opacity(op.opacity, 0.5)}</div>`;
  } else if (o.kind === "planes") {
    h += `<div class="ovm-line">${color("color", "--pore-surface", "Planes", true)}</div><div class="ovm-line">${style()}</div><div class="ovm-line">${opacity(op.opacity, 0.4)}</div>`;
  }
  // drawn in each cell of the display range, up to a limit (viewport/cell_tiles.js)
  if (o.cellsAsked > o.cells) h += `<div class="ovm-line"><span class="insp-hint">Drawn in the first ${o.cells} of the range's ${o.cellsAsked} cells</span></div>`;
  const own = Object.keys(o.over || {}).length;
  h += `<div class="ovm-line ovm-foot"><span class="insp-hint">${o.kind === "pores" ? "Starts from Settings › Viewport › Pore surfaces" : "This document's overlay"}${own ? `; ${own} changed here` : ""}</span>
    ${own ? `<button type="button" class="tb-btn-text" data-a="reset">Reset</button>` : ""}</div>`;
  return h;
}
/* a pore surface's mesh: what it was made at, what the preset picked would take, and the Advanced numbers */
function meshInfo(o) {
  const esc = ui.H.esc, op = o.opts || {}, made = o.madeAt || { quality: MQ.DEFAULT };
  const tri = made.triangles, want = MQ.knobs("pores", { quality: op.quality, ...Object.fromEntries(Object.keys(MQ.PORE_LIMITS).map((k) => [k, op[k]])) });
  const grid = made.render_grid ? `${made.render_grid.join("×")} grid, ` : "";
  let status = `Made at ${MQ.PRESET_LABEL[MQ.porePresetOf(made.quality)]}${made.custom ? " (custom)" : ""}: ${grid}${tri != null ? `${MQ.fmtCount(tri)} triangles` : ""}`;
  if (o.remeshing) {
    const e = tri != null ? MQ.estimate(want, made, tri) : null;
    status = `Re-meshing at ${MQ.PRESET_LABEL[want.quality]}${e ? ` (about ${MQ.fmtCount(e.triangles)} triangles, ${(e.bytes / 1e6).toFixed(0)} MB)` : ""}…`;
  } else if (o.failed) status += ` · ${MQ.PRESET_LABEL[MQ.porePresetOf(o.failed.quality)]} failed`;
  const own = (k) => (k in o.over ? " ovm-own" : "");
  const knob = (k) => {
    const [lo, hi] = MQ.PORE_LIMITS[k], [label, help] = MQ.KNOB_LABEL[k];
    return `<label class="ovm-f${own(k)}" title="${esc(help)}">${esc(label)} <input type="number" data-o="${k}" data-num="1" min="${lo}" max="${hi}" step="${k === "smooth" ? 0.1 : 1}"
      value="${want[k]}" aria-label="${esc(label)}"></label>`;
  };
  return `<p class="insp-hint ovm-mesh" role="status" aria-live="polite">${esc(status)}</p>
    <details class="ovm-adv"${ui.advOpen ? " open" : ""}><summary>Advanced</summary><div class="ovm-line">${Object.keys(MQ.PORE_LIMITS).map(knob).join("")}</div>
    <p class="insp-hint">${esc(MQ.PRESET_HELP[want.quality])}. A number typed here wins over the preset; Reset takes it back.</p></details>`;
}
function bindEditor(ed, o) {
  if (!o) return;
  const O = ui.A.overlay(), key = o.key;
  const val = (e) => (e.type === "checkbox" ? e.checked : e.type === "range" || e.dataset.num ? (e.value === "" ? null : Number(e.value)) : e.value);
  ed.querySelectorAll("[data-o]").forEach((e) => e.addEventListener(e.type === "range" ? "input" : "change", () => {
    if (e.type === "range") { const out = e.parentElement.querySelector("output"); if (out) out.textContent = Number(e.value).toFixed(2); }
    O.override(key, { [e.dataset.o]: val(e) });
  }));
  const adv = ed.querySelector(".ovm-adv");
  if (adv) adv.addEventListener("toggle", () => { ui.advOpen = adv.open; });
  ed.querySelectorAll("[data-f]").forEach((e) => e.addEventListener("change", () => O.setFeatureHidden(key, Number(e.dataset.f), !e.checked)));
  ed.querySelectorAll("[data-hl]").forEach((b) => b.addEventListener("click", () => {
    const fid = Number(b.dataset.hl), it = O.get(key);
    const box = O.highlight(key, it && it.hl === fid ? null : fid);
    if (box) ui.H.shell().framePoints(box, 1.4);
  }));
  const r = ed.querySelector('[data-a="reset"]');
  if (r) r.addEventListener("click", () => O.override(key, Object.fromEntries(Object.keys(o.over).map((k) => [k, null]))));
}
function remove(key) {
  const { A, H } = ui, O = A.overlay();
  const it = O.get(key);
  if (!it) return;
  const rec = it.meta && it.meta.rec ? A.get(it.meta.rec) : null;
  if (rec && it.kind === "pores") A.dropMeshes(rec);     // the surfaces' data too; the numbers stay
  O.remove(key);
  H.toast(`${OVERLAY_TITLE[it.kind] || it.kind} deleted${rec ? "; the result stays in the Analysis tab" : ""}`, "info");
}
