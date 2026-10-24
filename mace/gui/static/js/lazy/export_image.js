/* Export Image… (spec §6, plan T11), loaded on first use: a PNG of the
 * structure viewer at the viewport's size, 2×, 4×, 1920×1080, 3000×3000
 * or a custom size, with a transparent background, the axis triad, the
 * measurement labels and the cell as options. The browser downloads it as
 * <stem>_<view>.png; the server never writes anything.
 *
 * All viewers draw through one shared WebGL context (plan S4), so the image
 * is drawn by the structure viewer itself: its element is resized off
 * screen to the target size for two frames (a still of the view covers its
 * place), drawn, read with toDataURL and put back. The largest side is
 * min(8192, MAX_RENDERBUFFER_SIZE). */
import { loadCss, panel } from "./lazy_util.js";
import * as profile from "../profile.js";
import { dataUrlWithDpi } from "./png_dpi.js";
import { dataUrlBlob } from "../plotkit.js";

const PRESETS = [["1", "Viewport"], ["2", "2×"], ["4", "4×"], ["1920x1080", "1920 × 1080"], ["3000x3000", "3000 × 3000"], ["custom", "Custom"],
  ["physical", "Physical size (width × DPI)"]];
const DPIS = [72, 150, 300, 600];
const PER_INCH = { in: 1, cm: 2.54 };
const KEY = "mace.gui.export.v1";

export function maxSide() {
  let gl = null;
  try {
    const c = document.createElement("canvas");
    gl = c.getContext("webgl2") || c.getContext("webgl");
    const rb = gl ? gl.getParameter(gl.MAX_RENDERBUFFER_SIZE) : 4096;
    return Math.min(8192, rb || 4096);
  } catch (e) { return 4096; }
  finally { try { const x = gl && gl.getExtension("WEBGL_lose_context"); if (x) x.loseContext(); } catch (e) { /* none */ } }
}
let MAX = 0;
const max = () => MAX || (MAX = maxSide());

function prefs(H) {
  // plan E9: the resolution and physical size, and SVG shading; defaults from Settings › export.*
  const g = (k, d) => { try { const v = H && H.settings && H.settings.get(k); return v === undefined || v === null || v === "" ? d : v; } catch (e) { return d; } };
  const base = { format: "png", size: "1", w: 1920, h: 1080, transparent: false, triad: true, labels: true, cell: true, legend: "none",
    patterns: ["auto", "all"].includes(g("viewport.printPatterns", "off")) ? g("viewport.printPatterns", "off") : "off",
    dpi: Number(g("export.dpi", 150)) || 150, units: g("export.units", "in") === "cm" ? "cm" : "in", pw: g("export.units", "in") === "cm" ? 8.9 : 3.5,
    shading: g("export.vectorShading", "flat") === "shaded" ? "shaded" : "flat" };
  // the last choices, in the profile (memory.exportImage); a pre-1.1.5 browser's own key until then
  let last = profile.memory("exportImage", null);
  if (!last || typeof last !== "object") {
    try { last = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { last = {}; }
  }
  last = { ...(last || {}) };
  // the dialog's last choice, unless Settings › export.* changed since it was made (Settings is the default)
  const now = settingsOf(H), then = last.settings || {};
  for (const k of Object.keys(SETTING_KEYS)) if (then[k] !== now[k]) delete last[k];
  delete last.settings;
  return { ...base, ...last };
}
const SETTING_KEYS = { dpi: "export.dpi", units: "export.units", shading: "export.vectorShading", patterns: "viewport.printPatterns" };
function settingsOf(H) {
  const out = {};
  for (const [k, key] of Object.entries(SETTING_KEYS)) {
    try { const v = H && H.settings && H.settings.get(key); out[k] = v === undefined ? null : v; } catch (e) { out[k] = null; }
  }
  return out;
}
function keep(p, H) { profile.remember("exportImage", { ...p, settings: settingsOf(H) }); }

/** The target pixel size for a choice, from the viewer canvas's backing size. */
export function targetSize(choice, canvas, custom) {
  const W = canvas.width, H = canvas.height;
  if (/^\d+$/.test(choice)) return [W * Number(choice), H * Number(choice)];
  if (choice === "custom") return [Math.round(custom.w), Math.round(custom.h)];
  if (choice === "physical") {
    // a printed width at a resolution: width (in or cm) / units per inch × DPI pixels; the view's aspect kept
    const w = Math.round((Number(custom.pw) / PER_INCH[custom.units || "in"]) * Number(custom.dpi));
    return [w, Math.round((w * H) / Math.max(1, W))];
  }
  const [w, h] = choice.split("x").map(Number);
  return [w, h];
}

function loadImage(url) {
  return new Promise((resolve) => { const i = new Image(); i.onload = () => resolve(i); i.onerror = () => resolve(null); i.src = url; });
}

/** The axis triad's SVG drawn onto a 2D canvas (bottom left, scaled with the image). */
function triadImage(svgEl) {
  return new Promise((resolve) => {
    if (!svgEl || !svgEl.innerHTML.trim()) { resolve(null); return; }
    const clone = svgEl.cloneNode(true);
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    const r = svgEl.getBoundingClientRect();
    clone.setAttribute("width", r.width || 80); clone.setAttribute("height", r.height || 80);
    // the triad's colors come from CSS variables: inline the computed ones
    const css = getComputedStyle(svgEl);
    const srcs = svgEl.querySelectorAll("*");
    clone.querySelectorAll("*").forEach((n, k) => {
      const cs = getComputedStyle(srcs[k]);
      for (const p of ["fill", "stroke", "stroke-width", "font-size", "font-family", "font-weight"]) {
        const v = cs.getPropertyValue(p);
        if (v) n.style.setProperty(p, v);
      }
    });
    clone.style.color = css.color;
    const img = new Image();
    img.onload = () => resolve({ img, w: r.width || 80, h: r.height || 80, left: r.left, bottom: r.bottom });
    img.onerror = () => resolve(null);
    img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(new XMLSerializer().serializeToString(clone));
  });
}

/**
 * Draw the image: {width, height, transparent, triad, labels, cell} -> data URL.
 * `X` is what the shell lends: {viewer, scene, bg(), triadEl}; a Volume document's also {el: its viewer's element,
 * overlay(g, W, H, k): what the view draws over the 3D picture (the potential's color bar)}.
 */
export async function render(X, o) {
  const v = X.viewer, el = X.el || document.getElementById("viewer-main"), canvas = v.getCanvas();
  const viewRect = canvas.getBoundingClientRect();
  const triad = o.triad ? await triadImage(X.triadEl) : null;
  // the canvas holds devicePixelRatio x the UI scale's zoom pixels per own px (app.js sharpUnderZoom)
  const dpr = (window.devicePixelRatio || 1) * (window.maceZoom ? window.maceZoom.of(el) : 1);
  const restoreScene = X.scene ? X.scene.forExport({ cell: o.cell, measurements: o.labels }) : () => {};
  // the grayscale print patterns (Settings › Viewport), drawn at the picture's size over it
  const pat = o.patterns && o.patterns !== "off" ? await import("../viewport/print_patterns.js") : null;
  let over = null;
  const saved = el.getAttribute("style") || "";
  // the shared WebGL context takes the new size only on a later frame (a
  // synchronous resize draws into the old corner): the viewer is drawn off
  // screen for two frames while a still of the current view covers its place
  v.render();
  const still = document.createElement("img");
  still.src = v.getCanvas().toDataURL("image/png");
  still.className = "export-still";
  still.style.cssText = `position:absolute;left:${el.offsetLeft}px;top:${el.offsetTop}px;width:${el.offsetWidth}px;height:${el.offsetHeight}px;pointer-events:none;z-index:5`;
  el.parentElement.appendChild(still);
  const frame = () => new Promise((r) => requestAnimationFrame(() => r()));
  let url = "";
  try {
    if (o.transparent) v.setBackgroundColor(X.bg(), 0);
    // no transition (Reduce motion makes every change a 0.01 ms one): the size is the new one when it is read
    el.setAttribute("style", `${saved};position:fixed;left:-100000px;top:0;width:${o.width / dpr}px;height:${o.height / dpr}px;transition:none`);
    v.resize();
    await frame(); await frame();
    v.render();
    // encode in this task: the viewer's canvas keeps the frame only until the next composite
    url = v.getCanvas().toDataURL("image/png");
    if (pat) over = pat.overlayCanvas(v, o.width, o.height, o.patterns, dpr);
  } finally {
    // back in place without a transition (a Volume viewer, absolutely placed, read 100 000 px wide meanwhile), then as it was
    el.setAttribute("style", `${saved};transition:none`);
    void el.offsetWidth;
    el.setAttribute("style", saved);
    if (o.transparent) v.setBackgroundColor(X.bg(), 1);
    restoreScene();
    v.resize();
    v.render();
    await frame(); await frame();
    v.render();
    still.remove();
  }
  const img = await loadImage(url);
  // the element legend at the picture's scale (the view's own px -> the image's)
  const k = o.width / Math.max(1, viewRect.width);
  const leg = o.legend && o.legend !== "none" && X.legend
    ? await legendImage(X, o.transparent ? null : X.bg(), k, pat ? pat.mapFor(v, o.patterns) : null) : null;
  if (!img || (!triad && !leg && !over && !X.overlay && img.naturalWidth === o.width && img.naturalHeight === o.height)) return url;
  const out = document.createElement("canvas");
  out.width = o.width; out.height = o.height;
  const g = out.getContext("2d");
  g.drawImage(img, 0, 0, o.width, o.height);
  if (over) g.drawImage(over, 0, 0, o.width, o.height);
  if (leg) {
    const [x, y] = corner(o.legend, o.width, o.height, leg.w, leg.h, 12 * k);
    g.drawImage(leg.img, x, y, leg.w, leg.h);
  }
  if (X.overlay) X.overlay(g, o.width, o.height, k);
  if (triad) {
    // the triad keeps its share of the picture
    const x = (triad.left - viewRect.left) * k, h = triad.h * k;
    g.drawImage(triad.img, x, (triad.bottom - viewRect.top) * k - h, triad.w * k, h);
  }
  return out.toDataURL("image/png");
}

const LEGEND_CORNERS = [["none", "None"], ["top-left", "Top left"], ["top-right", "Top right"], ["bottom-left", "Bottom left"], ["bottom-right", "Bottom right"]];
/** Where a w x h legend sits in a W x H picture, `m` from the edges. */
export function corner(where, W, H, w, h, m) {
  return [where.endsWith("left") ? m : W - w - m, where.startsWith("top") ? m : H - h - m];
}
/** The legend (lazy/legend_dialog.js forImage, the same function as mace plotting --legend) as an image `k` times its size,
 * its swatches with the print patterns the image draws on the atoms. */
async function legendImage(X, background, k, patterns = null) {
  const syms = X.symbols ? X.symbols() : [];
  if (!syms.length) return null;
  const L = await X.legend(syms, { background, scale: 1, patterns });
  const img = await loadImage(URL.createObjectURL(new Blob([L.svg], { type: "image/svg+xml" })));
  return img ? { img, w: L.width * k, h: L.height * k } : null;
}

export function download(url, name) {
  const a = document.createElement("a");
  a.href = url; a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/** The dialog. `X` adds {stem, view} for the default file name. */
export async function open(H, X) {
  await loadCss(import.meta.url, "../../css/dialogs.css");
  const p = prefs(H), canvas = X.plot ? { width: X.plot.size[0], height: X.plot.size[1], clientWidth: X.plot.size[0], clientHeight: X.plot.size[1] }
    : X.viewer.getCanvas(), M = max();
  const body = document.createElement("div");
  body.className = "export-dlg";
  if (X.format) p.format = X.format;          // a plot's PNG or SVG button
  const refuse = X.volume ? "a structure view only" : X.viewState && X.viewState.projection === "perspective" ? "SVG export is orthographic; press 5" : "";
  // Analysis overlays (pore surfaces, a crystal shape, planes, a net) are WebGL meshes: PNG only
  let overlays = false, pores = false;
  try { const A = window.__maceAnalysis; overlays = !X.plot && !X.volume && !!A && ["pores", "mesh", "planes", "net"].some((k) => A.overlay.count(k) > 0); pores = overlays && A.overlay.count("pores") > 0; } catch (e) { overlays = false; }
  if (refuse && p.format === "svg") p.format = "png";
  body.innerHTML = `<label class="f">Format<select class="field" data-k="format" aria-label="Format">
      <option value="png"${p.format !== "svg" ? " selected" : ""}>PNG</option>
      <option value="svg"${p.format === "svg" ? " selected" : ""}${refuse ? ` disabled title="${H.esc(refuse)}"` : ""}>SVG${refuse ? ` (${H.esc(refuse)})` : ""}</option></select></label>
    <p class="note svg-note" hidden>Vector drawing of the orthographic view: one circle per drawn atom, the bonds and the cell. Flat colors: lighting, depth cueing and outline apply to PNG only.${overlays
      ? " The analysis overlays shown (pore surfaces, crystal shape, planes or net) are left out: export PNG to keep them." : ""}</p>
    <label class="f size-f">Size<select class="field" data-k="size" aria-label="Size">${PRESETS.map(([v, t]) =>
      `<option value="${v}"${v === p.size ? " selected" : ""}>${t}</option>`).join("")}</select></label>
    <div class="row custom"><label class="f">Width<input class="field" type="number" min="16" max="${M}" step="1" data-k="w" value="${p.w}"></label>
      <label class="f">Height<input class="field" type="number" min="16" max="${M}" step="1" data-k="h" value="${p.h}"></label></div>
    <div class="row physical"><label class="f">Width<input class="field" type="number" min="0.5" max="200" step="0.1" data-k="pw" value="${p.pw}"></label>
      <label class="f">Units<select class="field" data-k="units"><option value="in"${p.units === "in" ? " selected" : ""}>in</option><option value="cm"${p.units === "cm" ? " selected" : ""}>cm</option></select></label></div>
    <div class="row dpi-row"><label class="f">Resolution<select class="field" data-k="dpisel">${DPIS.map((x) => `<option value="${x}"${x === p.dpi ? " selected" : ""}>${x} DPI</option>`).join("")}<option value="custom"${DPIS.includes(p.dpi) ? "" : " selected"}>Custom</option></select></label>
      <label class="f dpi-custom">DPI<input class="field" type="number" min="30" max="2400" step="1" data-k="dpi" value="${p.dpi}"></label></div>
    <p class="note" data-k="px"></p>
    <label class="f shading-f" hidden>Shading<select class="field" data-k="shading"><option value="flat"${p.shading !== "shaded" ? " selected" : ""}>Flat</option><option value="shaded"${p.shading === "shaded" ? " selected" : ""}>Shaded</option></select></label>
    <p class="note shading-note" hidden>Shaded uses a gradient per atom; fog and ambient occlusion are raster only.</p>
    <label class="chk"><input type="checkbox" data-k="transparent"${p.transparent ? " checked" : ""}> Transparent background</label>
    <label class="chk"><input type="checkbox" data-k="triad"${p.triad ? " checked" : ""}> Include the axis triad</label>
    <label class="chk"><input type="checkbox" data-k="labels"${p.labels ? " checked" : ""}> Include measurement labels</label>
    <label class="chk"><input type="checkbox" data-k="cell"${p.cell ? " checked" : ""}> Include the cell</label>
    <label class="chk remesh-f"${pores ? "" : " hidden"} title="The pore surfaces are made again, densely, for this image only (can take seconds); the view keeps its own"><input type="checkbox" data-k="remesh"${p.remesh ? " checked" : ""}> Re-mesh the pore surfaces at Publication quality for this image</label>
    <label class="f patterns-f">Print patterns<select class="field" data-k="patterns" aria-label="Print patterns">${[["off", "Off"], ["auto", "Similar grays"], ["all", "Every element"]].map(([v, t]) =>
      `<option value="${v}"${v === p.patterns ? " selected" : ""}>${t}</option>`).join("")}</select></label>
    <label class="f legend-f">Element legend<select class="field" data-k="legend" aria-label="Element legend">${LEGEND_CORNERS.map(([v, t]) =>
      `<option value="${v}"${v === p.legend ? " selected" : ""}>${t}</option>`).join("")}</select></label>
    <label class="f">File name<input class="field" data-k="name" spellcheck="false" autocomplete="off" value="${H.esc(X.plot ? `${X.stem}.png` : `${X.stem}_${X.view}.png`)}"></label>
    <div class="err" role="alert"></div>`;
  const q = (k) => body.querySelector(`[data-k="${k}"]`);
  const dpiOf = () => (q("dpisel").value === "custom" ? Number(q("dpi").value) : Number(q("dpisel").value));
  const read = () => ({ format: q("format").value, size: q("size").value, w: Number(q("w").value), h: Number(q("h").value), transparent: q("transparent").checked,
    triad: q("triad").checked, labels: q("labels").checked, cell: q("cell").checked, dpi: dpiOf(), units: q("units").value, pw: Number(q("pw").value),
    shading: q("shading").value, legend: q("legend").value, patterns: q("patterns").value, remesh: q("remesh").checked });
  const check = () => {
    const r = read(), svg = r.format === "svg";
    for (const k of ["px", "triad", "labels"]) { const el = q(k); (el.closest("label") || el).hidden = svg; }
    body.querySelector(".size-f").hidden = svg;
    body.querySelector(".svg-note").hidden = !svg;
    body.querySelector(".shading-f").hidden = !svg || !!X.plot;
    body.querySelector(".shading-note").hidden = !svg || !!X.plot;
    body.querySelector(".dpi-row").hidden = svg;
    body.querySelector(".dpi-custom").hidden = q("dpisel").value !== "custom";
    body.querySelector(".physical").hidden = svg || r.size !== "physical";
    for (const k of ["triad", "labels", "cell", "transparent", "legend"]) { const el = q(k); if (X.plot && el) el.closest("label").hidden = true; }
    // a Volume document: its viewer as shown (the color bar drawn in), none of the structure view's options
    for (const k of ["labels", "cell", "legend"]) { const el = q(k); if (X.volume && el) el.closest("label").hidden = true; }
    body.querySelector(".patterns-f").hidden = svg || !!X.plot || !!X.volume;          // a PNG of the 3D view
    body.querySelector(".remesh-f").hidden = svg || !pores;
    const nm = q("name");
    if (svg && /\.png$/i.test(nm.value)) nm.value = nm.value.replace(/\.png$/i, ".svg");
    if (!svg && /\.svg$/i.test(nm.value)) nm.value = nm.value.replace(/\.svg$/i, ".png");
    if (svg) { body.querySelector(".custom").hidden = true; body.querySelector(".err").textContent = ""; return [canvas.clientWidth, canvas.clientHeight]; }
    const [w, h] = targetSize(r.size, canvas, r);
    body.querySelector(".custom").hidden = r.size !== "custom";
    const big = Math.max(w, h) > M, small = Math.min(w, h) < 16 || !Number.isFinite(w) || !Number.isFinite(h);
    const inches = (w / r.dpi), unit = r.units === "cm" ? ["cm", 2.54] : ["in", 1];
    q("px").textContent = `${w} × ${h} pixels · ${(inches * unit[1]).toFixed(2)} × ${((h / r.dpi) * unit[1]).toFixed(2)} ${unit[0]} at ${r.dpi} DPI · up to ${M} px on this GPU`;
    body.querySelector(".err").textContent = big ? `At most ${M} pixels a side: the largest image this browser's graphics can draw` : small ? "At least 16 pixels a side" : "";
    return big || small ? null : [w, h];
  };
  body.addEventListener("input", check);
  body.addEventListener("change", check);
  const d = panel(H, { id: "export-image", title: "Export Image", body,
    foot: `<button class="tb-btn-text" data-v="cancel">Cancel</button><button class="tb-btn-text primary" data-v="export">Export</button>` });
  check();
  d.querySelector('[data-v="cancel"]').addEventListener("click", () => d.close());
  d.querySelector('[data-v="export"]').addEventListener("click", async () => {
    const wh = check();
    if (!wh) return;
    const r = read();
    keep(r, H);
    if (r.format === "svg") {
      let name = q("name").value.trim() || `${X.stem}_${X.view}.svg`;
      if (!/\.svg$/i.test(name)) name += ".svg";
      try {
        if (X.plot) {
          const text = await X.plot.renderSvg();
          if (window.__maceExportHook) window.__maceExportHook(text, "svg");
          download(URL.createObjectURL(new Blob([text], { type: "image/svg+xml" })), name);
          d.close();
          H.toast(`Exported ${name}`, "success");
          return;
        }
        const n = await exportSvg(X, { transparent: r.transparent, cell: r.cell, name, shading: r.shading, legend: r.legend });
        d.close();
        H.toast(`Exported ${name} (${n} atoms)`, "success");
      } catch (e) { body.querySelector(".err").textContent = e.message || String(e); }
      return;
    }
    let name = q("name").value.trim() || `${X.stem}_${X.view}.png`;
    if (!/\.png$/i.test(name)) name += ".png";
    performance.mark("ui:export:start");
    const draw = () => render(X, { width: wh[0], height: wh[1], ...r, legend: X.plot || X.volume ? "none" : r.legend,
      ...(X.volume ? { patterns: "off", labels: false } : {}) });
    let raw;
    if (X.plot) raw = await X.plot.renderPng(wh[0], wh[1]);
    else if (r.remesh && pores) {
      // the pore surfaces at Publication for the image only (the view's own come back after, kept: no second request)
      const ex = d.querySelector('[data-v="export"]');
      ex.disabled = true;
      body.querySelector(".err").textContent = "";
      H.toast("Re-meshing the pore surfaces at Publication quality for the image…", "info");
      try { raw = await window.__maceAnalysis.overlay.withQuality("publication", draw); } finally { ex.disabled = false; }
    } else raw = await draw();
    // the PNG says its resolution (pHYs), so it keeps its printed size
    const url = dataUrlWithDpi(raw, r.dpi);
    performance.mark("ui:export:end");
    if (window.__maceExportHook) window.__maceExportHook(url);              // tests read the image
    download(url, name);
    d.close();
    H.toast(`Exported ${name} (${wh[0]} × ${wh[1]}, ${r.dpi} DPI)`, "success");
  });
  return d;
}

/** The structure view as SVG (js/lazy/export_svg.js); returns the number of atoms drawn. */
export async function exportSvg(X, { transparent = false, cell = true, name = "structure.svg", shading = "flat", legend = "none" } = {}) {
  const S = await import("./export_svg.js");
  const why = S.refuseReason(X.viewState && X.viewState.projection);
  if (why) throw new Error(why);
  performance.mark("ui:svg:start");
  const g = await S.sceneGroup(X, { cell });
  const c = X.viewer.getCanvas();
  let text = await S.svgDocument({ width: c.clientWidth, height: c.clientHeight, background: transparent ? null : X.bg(), groups: [g],
    title: `${X.stem} (${X.view})`, shading });
  if (legend !== "none" && X.legend && X.symbols && X.symbols().length) {
    // the legend's own <g id="element-legend"> (editable text), moved into its corner
    const L = await X.legend(X.symbols(), { background: transparent ? null : X.bg(), scale: 1, patterns: g.patternMap });
    const inner = L.svg.slice(L.svg.indexOf("<g id=\"element-legend\""), L.svg.lastIndexOf("</svg>")).trim();
    const [x, y] = corner(legend, c.clientWidth, c.clientHeight, L.width, L.height, 12);
    text = text.replace(/<\/svg>\s*$/, `<g id="legend-corner" transform="translate(${x.toFixed(1)} ${y.toFixed(1)})">\n${inner}\n</g>\n</svg>`);
  }
  performance.mark("ui:svg:end");
  performance.measure("ui:svg", "ui:svg:start", "ui:svg:end");
  if (window.__maceExportHook) window.__maceExportHook(text, "svg");
  const url = URL.createObjectURL(new Blob([text], { type: "image/svg+xml" }));
  download(url, name);
  setTimeout(() => URL.revokeObjectURL(url), 5000);
  return g.atoms.length;
}

/** Copy Image to Clipboard: the viewport at 1×, as it is shown. */
export async function copy(H, X) {
  const url = await render(X, { width: X.viewer.getCanvas().width, height: X.viewer.getCanvas().height, transparent: false, triad: true, labels: true, cell: true });
  try {
    if (!navigator.clipboard || !window.ClipboardItem) throw new Error("this browser has no image clipboard here (it needs https or localhost)");
    const blob = dataUrlBlob(url);
    await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    H.toast("Copied the image", "success");
    return true;
  } catch (e) {
    H.toast(`Could not copy the image: ${e.message || e}`, "info", { label: "Export Image…", run: () => H.run("file.exportImage") });
    return false;
  }
}
