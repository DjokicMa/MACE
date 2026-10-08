/* The Inspector's "Range & boundary" and "Rendering" groups (polish A5, A6):
 * the drawn document's own values (viewport/display_state.js), with the
 * Settings › Viewport defaults behind them. A value that differs from the
 * default shows a dot after its label (click: back to the default); each
 * group has Reset and Save as Default. Changes are live: inputs settle for
 * 120 ms, the view redraws at most once per frame. */
import * as DS from "./display_state.js";
import * as G from "../geom/core.js";
import { autoOutline } from "./rendering.js";

const NUM = (x) => String(+Number(x).toFixed(3));
const LARGE = 20000, REFUSE = 200000;

export function createDisplayPanel(deps) {
  const { esc, icon } = deps;
  const doc = () => deps.doc();
  const val = (k) => DS.of(doc())[k];
  // the swatch of an unset outline color: what is drawn (rendering.js autoOutline)
  const autoOutlineHex = () => autoOutline(val("background"), val("backgroundColor"), deps.cssColor("--text"));
  let rangeEl = null, renderEl = null, timer = 0;

  const dot = (k) => (DS.isOverride(doc(), k)
    ? `<button type="button" class="ovr-dot" data-unset="${k}" title="Changed for this document. Click to use the default." aria-label="Use the default for this value"></button>` : "");
  const lab = (k, text, forId) => `<label class="dp-lbl"${forId ? ` for="${forId}"` : ""}>${esc(text)}${dot(k)}</label>`;
  const slider = (k, text, min, max, step, pct = false) => {
    const v = Number(val(k));
    return `<div class="ins-row dp-row" data-key="${k}">${lab(k, text, `dp-${k}`)}<span class="dp-slide"><input type="range" id="dp-${k}" data-k="${k}" min="${min}" max="${max}" step="${step}" value="${v}">` +
      `<output>${pct ? Math.round(v * 100) + "%" : NUM(v)}</output></span></div>`;
  };
  const set = (k, v) => { const d = doc(); if (d) DS.set(d, k, v); };

  /* ---------------------------------------------------------------- Range & boundary */
  function rangeHtml() {
    const d = doc(), s = deps.structure(), pbc = (s && s.pbc) || [true, true, true];
    const r = DS.rangeOf(d);
    const rows = [0, 1, 2].map((k) => `<div class="ins-row dp-axis" data-axis="${k}"><span class="dp-ax">${"abc"[k]}</span>` +
      (pbc[k] ? `<label>from <input class="field dp-num" type="number" step="0.1" min="-5" max="10" data-lo="${k}" value="${NUM(r.lo[k])}" aria-label="Range ${"abc"[k]} from"></label>` +
        `<label>to <input class="field dp-num" type="number" step="0.1" min="-5" max="10" data-hi="${k}" value="${NUM(r.hi[k])}" aria-label="Range ${"abc"[k]} to"></label>`
        : `<span class="insp-hint">Not periodic</span>`) + `</div>`).join("");
    const est = s ? G.estimateCount(s, DS.drawRange(d), !!val("showBoundary")) : 0;
    const estCls = est > LARGE ? " warn" : "";
    const b = deps.bonded();
    let note = "";
    if (val("bondedOutside") !== "off" && b) {
      if (b.allExtended) note = "This structure has no discrete molecules; nothing to complete.";
      else if (b.extended) note = `This structure is an extended network (${b.extended.dim}D)${b.extended.axes && b.extended.axes !== "oblique" ? ` in ${b.extended.axes}` : ""}: only discrete molecules are completed.`;
      if (b.truncated) note += (note ? " " : "") + "Stopped at 50 000 atoms.";
    }
    return `<div class="dp-group" data-group="range">
      <div class="dp-sub">Range${DS.isOverride(d, "displayLo") || DS.isOverride(d, "displayHi")
        ? `<button type="button" class="ovr-dot" data-unset="range" title="Changed for this document. Click to use the default." aria-label="Use the default range"></button>` : ""}</div>${rows}
      <div class="ins-row dp-quick"><button type="button" class="tb-btn-text sm" data-quick="unit">Unit Cell</button>
        <button type="button" class="tb-btn-text sm" data-quick="cell10" title="Range −0.1 to 1.1 on each periodic axis">Cell + 10%</button>
        <button type="button" class="tb-btn-text sm" data-quick="2">2×2×2</button><button type="button" class="tb-btn-text sm" data-quick="3">3×3×3</button></div>
      <p class="insp-hint dp-est${estCls}" aria-live="polite">≈ ${est.toLocaleString("en-US")} atoms drawn${est > LARGE ? " (Large: simplified drawing)" : ""}</p>
      <p class="set-err dp-range-err" role="alert"></p>
      <div class="ins-row dp-boundary" title="Also draw atoms on the faces, edges and corners of the range.">${dot("showBoundary")}</div>
      <div class="ins-row dp-row" data-key="bondedOutside">${lab("bondedOutside", "Bonded Atoms Outside", "dp-bondedOutside")}
        <select class="field" id="dp-bondedOutside" data-k="bondedOutside" title="Draw atoms outside the range that are bonded to atoms inside it (one shell), or complete every discrete molecule that crosses the range.">
        ${[["off", "None"], ["neighbors", "Bonded Neighbors"], ["molecules", "Whole Molecules"]].map(([o, t]) => `<option value="${o}"${val("bondedOutside") === o ? " selected" : ""}>${t}</option>`).join("")}</select></div>
      <p class="insp-hint dp-bonded-note"${note ? "" : " hidden"}>${esc(note)}</p>
      <div class="ins-row dp-row" data-key="showPolyhedra"><label class="tog"><input type="checkbox" data-k="showPolyhedra"${val("showPolyhedra") ? " checked" : ""}><span>Polyhedra</span></label>${dot("showPolyhedra")}</div>
      ${slider("atomScale", "Atom Size", 0.1, 1, 0.05)}${slider("bondRadius", "Bond Radius", 0.03, 0.4, 0.01)}
      <div class="ins-row end dp-foot"><button type="button" class="tb-btn-text sm" data-reset-group="range">Reset</button>
        <button type="button" class="tb-btn-text sm" data-save-group="range">Save as Default</button></div></div>`;
  }

  /* ---------------------------------------------------------------- Rendering */
  function renderHtml() {
    const preset = val("renderPreset");
    const dir = val("lightDirection");
    const bg = val("background");
    return `<div class="dp-group" data-group="render">
      <div class="ins-row dp-row" data-key="renderPreset">${lab("renderPreset", "Preset", "dp-renderPreset")}
        <select class="field" id="dp-renderPreset" data-k="renderPreset">${[["default", "Default"], ["flat", "Flat"], ["publication", "Publication"], ["presentation", "Presentation"], ["custom", "Custom"]]
          .map(([o, t]) => `<option value="${o}"${preset === o ? " selected" : ""}${o === "custom" && preset !== "custom" ? " disabled" : ""}>${t}</option>`).join("")}</select></div>
      ${slider("depthCue", "Depth Cueing", 0, 1, 0.05, true)}
      <p class="insp-hint">How much the farthest atoms fade toward the background. 0 is off.</p>
      ${slider("lightStrength", "Light Strength", 0.3, 1.6, 0.05)}
      <div class="ins-row dp-row" data-key="lightDirection">${lab("lightDirection", "Light Direction")}<span class="segment" role="radiogroup" aria-label="Light direction">
        ${[["front", "Front"], ["upper-left", "Upper Left"], ["top", "Top"]].map(([o, t]) => `<button type="button" role="radio" data-dir="${o}" aria-checked="${dir === o}">${t}</button>`).join("")}</span></div>
      <p class="insp-hint dp-dir-note" hidden></p>
      ${slider("ambient", "Ambient Light", 0, 0.6, 0.05, true)}
      <p class="insp-hint">Light from all directions: lifts the shaded side of every atom.</p>
      ${slider("ambientOcclusion", "Ambient Occlusion", 0, 1, 0.05)}
      <p class="insp-hint dp-ao-note">Darkens creases and contacts between atoms. Costs frame time on large structures.</p>
      <div class="ins-row dp-row" data-key="outline"><label class="tog"><input type="checkbox" data-k="outline"${val("outline") ? " checked" : ""}><span>Outline</span></label>${dot("outline")}</div>
      ${slider("outlineWidth", "Outline Width", 0.02, 0.3, 0.01)}
      <div class="ins-row dp-row" data-key="outlineColor">${lab("outlineColor", "Outline Color", "dp-outlineColor")}<input type="color" id="dp-outlineColor" data-k="outlineColor" value="${/^#[0-9a-f]{6}$/i.test(val("outlineColor")) ? val("outlineColor") : autoOutlineHex()}">
        <button type="button" class="tb-btn-text sm" data-theme-outline>Theme</button></div>
      <div class="ins-row dp-row" data-key="background">${lab("background", "Background", "dp-background")}
        <select class="field" id="dp-background" data-k="background">${[["theme", "Theme"], ["white", "White"], ["black", "Black"], ["custom", "Custom"]]
          .map(([o, t]) => `<option value="${o}"${bg === o ? " selected" : ""}>${t}</option>`).join("")}</select>
        <input type="color" data-k="backgroundColor" aria-label="Custom background color" value="${/^#[0-9a-f]{6}$/i.test(val("backgroundColor")) ? val("backgroundColor") : "#ffffff"}"${bg === "custom" ? "" : " hidden"}></div>
      <p class="insp-hint">Export SVG draws flat colors: lighting, depth cueing and outline apply to PNG only.</p>
      <div class="ins-row end dp-foot"><button type="button" class="tb-btn-text sm" data-reset-group="render">Reset</button>
        <button type="button" class="tb-btn-text sm" data-save-group="render">Save as Default</button></div></div>`;
  }

  /* ---------------------------------------------------------------- events */
  function wire(el) {
    el.addEventListener("input", (e) => {
      const t = e.target;
      if (t.type === "range" && t.dataset.k) {
        const k = t.dataset.k, pct = ["depthCue", "ambient"].includes(k);
        t.nextElementSibling.textContent = pct ? Math.round(t.value * 100) + "%" : NUM(t.value);
        clearTimeout(timer);
        timer = setTimeout(() => edit(k, Number(t.value)), 120);
      }
    });
    el.addEventListener("change", (e) => {
      const t = e.target;
      if (t.dataset.lo !== undefined || t.dataset.hi !== undefined) return rangeInput();
      const k = t.dataset.k;
      if (!k) return;
      if (t.type === "checkbox") edit(k, t.checked);
      else if (t.type === "range") { clearTimeout(timer); edit(k, Number(t.value)); }
      else if (k === "renderPreset") applyPreset(t.value);
      else edit(k, t.value);
    });
    el.addEventListener("click", (e) => {
      const t = e.target;
      const un = t.closest("[data-unset]");
      if (un) { const d = doc(); if (!d) return; if (un.dataset.unset === "range") DS.reset(d, ["displayLo", "displayHi"]); else DS.reset(d, [un.dataset.unset]); return; }
      const q = t.closest("[data-quick]");
      if (q) {
        const d = doc(); if (!d) return;
        const v = q.dataset.quick;
        if (v === "unit") DS.set(d, { displayLo: [0, 0, 0], displayHi: [1, 1, 1] });
        else if (v === "cell10") DS.set(d, { displayLo: [-0.1, -0.1, -0.1], displayHi: [1.1, 1.1, 1.1] });
        else DS.setCells(d, Number(v));
        return;
      }
      const dir = t.closest("[data-dir]");
      if (dir) return lightDirection(dir.dataset.dir);
      if (t.closest("[data-theme-outline]")) return edit("outlineColor", "");
      const rg = t.closest("[data-reset-group]");
      if (rg) { const d = doc(); if (d) DS.reset(d, rg.dataset.resetGroup); return; }
      const sg = t.closest("[data-save-group]");
      if (sg) {
        const d = doc(); if (!d) return;
        DS.saveAsDefault(d, sg.dataset.saveGroup);
        deps.saved("Saved as the default for new documents and documents you have not changed");
      }
    });
  }
  function edit(k, v) {
    const d = doc();
    if (!d) return;
    // a rendering field edited after a preset: the preset becomes Custom
    if (DS.GROUPS.render.includes(k) && k !== "renderPreset" && val("renderPreset") !== "custom") DS.set(d, { [k]: v, renderPreset: "custom" });
    else DS.set(d, k, v);
  }
  function applyPreset(name) {
    const d = doc();
    if (!d || !DS.PRESETS[name]) return;
    DS.set(d, { ...DS.PRESETS[name], renderPreset: name });
  }
  function lightDirection(dir) {
    const d = doc();
    if (!d) return;
    const pairs = { lightDirection: dir };
    let raised = false;
    // an off-axis light leaves a dark crescent without some ambient light
    if (dir !== "front" && Number(val("ambient")) < 0.15) { pairs.ambient = 0.15; raised = true; }
    if (val("renderPreset") !== "custom") pairs.renderPreset = "custom";
    DS.set(d, pairs);
    const n = renderEl && renderEl.querySelector(".dp-dir-note");
    if (n) { n.hidden = !raised; n.textContent = raised ? "Ambient raised to 15% so the shaded side stays visible" : ""; }
  }
  function rangeInput() {
    const d = doc();
    if (!d || !rangeEl) return;
    const r = DS.rangeOf(d), err = rangeEl.querySelector(".dp-range-err");
    const lo = r.lo.slice(), hi = r.hi.slice();
    rangeEl.querySelectorAll("[data-lo]").forEach((x) => { lo[+x.dataset.lo] = Number(x.value); });
    rangeEl.querySelectorAll("[data-hi]").forEach((x) => { hi[+x.dataset.hi] = Number(x.value); });
    const bad = [...lo, ...hi].some((x) => !Number.isFinite(x) || x < -5 || x > 10) || lo.some((x, k) => hi[k] - x < 0.05);
    if (bad) { err.textContent = "Each axis: from −5 to 10, the end at least 0.05 above the start"; return; }
    const s = deps.structure();
    const est = s ? G.estimateCount(s, { lo, hi }, !!val("showBoundary")) : 0;
    if (est > REFUSE) { err.textContent = `Too many atoms to draw (≈ ${est.toLocaleString("en-US")}); narrow the range`; return; }
    err.textContent = "";
    DS.set(d, { displayLo: lo, displayHi: hi });
  }

  /* ---------------------------------------------------------------- the two groups */
  function range() {
    rangeEl = document.createElement("div");
    rangeEl.className = "dp";
    fill(rangeEl, rangeHtml);
    wire(rangeEl);
    return rangeEl;
  }
  function render() {
    renderEl = document.createElement("div");
    renderEl.className = "dp";
    fill(renderEl, renderHtml);
    wire(renderEl);
    return renderEl;
  }
  function fill(el, html) {
    // the moved Boundary atoms checkbox (one control, the classic views read it) goes back to its parking place first
    const box = deps.boundaryLabel();
    if (box && el.contains(box)) deps.park(box);
    el.innerHTML = html();
    const slot = el.querySelector(".dp-boundary");
    if (slot && box) slot.prepend(box);
  }
  /** Redraw both groups with the drawn document's values (a field being typed in keeps its text). */
  function refresh() {
    for (const [el, html] of [[rangeEl, rangeHtml], [renderEl, renderHtml]]) {
      if (!el || !el.isConnected) continue;
      const ae = document.activeElement;
      if (ae && el.contains(ae) && ae.matches("input[type=number]")) {
        const est = el.querySelector(".dp-est");
        if (est) { const tmp = document.createElement("div"); tmp.innerHTML = html(); est.replaceWith(tmp.querySelector(".dp-est")); }
        continue;
      }
      const focusKey = ae && el.contains(ae) ? (ae.dataset.k || ae.dataset.dir || ae.dataset.quick || "") : "";
      fill(el, html);
      if (focusKey) { const f = el.querySelector(`[data-k="${focusKey}"], [data-dir="${focusKey}"], [data-quick="${focusKey}"]`); if (f) f.focus(); }
    }
  }
  return { range, render, refresh };
}
