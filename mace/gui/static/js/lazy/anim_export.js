/* Export Animation… (1.1.6), loaded on first use: the moving 3D view as an optimized GIF, or a WebM / MP4
 * when the server has ffmpeg.
 *
 * Two sources:
 * - This view: the FREQ document's normal mode (every frame of one cycle) or an optimization's steps, drawn by
 *   the view itself off screen at the chosen size (as Export Image draws a still: the viewer's element is moved
 *   off screen and resized while a still of it covers its place), each frame read with toDataURL, with the print
 *   patterns (Settings › Viewport) composited, and sent to the server, which writes the file (Pillow; ffmpeg for
 *   video: mace/gui/animexport.py).
 * - Rendered by MACE: the document's file (a FREQ or OPT output, an MD or NEB .xyz/.extxyz/.traj) drawn headless
 *   on the server, exactly as `mace animate` draws it.
 * Controls: format, frames per second, loop count, size (the view's, presets or custom) and a scale factor,
 * background (the theme's, transparent, white or black) and the print patterns. The browser downloads the file. */
import { loadCss, panel } from "./lazy_util.js";
import * as profile from "../profile.js";
import { maxSide, targetSize } from "./export_image.js";

const SIZES = [["1", "Viewport"], ["1280x720", "1280 × 720"], ["1920x1080", "1920 × 1080"], ["800x800", "800 × 800"], ["custom", "Custom"]];
const SCALES = [["0.5", "0.5×"], ["1", "1×"], ["2", "2×"], ["3", "3×"]];
const MAX_POST = 1_900_000;      // a frame's data URL must fit the server's request limit (2 MB)
const frame = () => new Promise((r) => requestAnimationFrame(() => r()));
/* a color token of the page (tokens.css): the white and black 3D backgrounds */
const tok = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function prefs(X) {
  const base = { source: "view", format: "gif", fps: X.freq ? 24 : 4, loop: "0", loopN: 3, size: "1", scale: "1", w: 1280, h: 720,
    background: "theme", patterns: (X.setting && X.setting("viewport.printPatterns")) || "off", every: 1 };
  const last = profile.memory("exportAnimation", null);
  const out = { ...base, ...(last && typeof last === "object" ? last : {}) };
  out.fps = X.freq ? (last && last.fpsFreq) || 24 : (last && last.fpsSteps) || 4;
  return out;
}

/** What "This view" can capture: {n, label, set(k), restore()}, or null (no animation in the view). */
function provider(X) {
  const v = X.viewer;
  if (X.freq && v && v.getNumFrames) {
    const d = X.freq.doc ? X.freq.doc() : null, was = X.freq.playing ? X.freq.playing() : false;
    const arrows = d && d.opts ? d.opts.arrows : false;
    return {
      label: "the normal mode's cycle",
      async start() {
        // the arrows of the still view do not move with the atoms: left out of the animation
        if (d && d.opts) d.opts.arrows = false;
        X.freq.play(false);
        await frame();
        this.n = v.getNumFrames();
      },
      async set(k) { const r = v.setFrame(k); if (r && r.then) await r; },
      async restore() { if (d && d.opts) d.opts.arrows = arrows; if (X.freq.redraw) X.freq.redraw(); if (was) X.freq.play(true); },
    };
  }
  if (X.traj && v) {
    const at = X.traj.current();
    return {
      label: "the optimization's steps",
      async start() { this.n = X.traj.steps(); },
      async set(k) { X.traj.step(k); await frame(); await frame(); },
      async restore() { X.traj.step(at); },
    };
  }
  return null;
}

/** Each frame of `P` drawn at width x height by the viewer off screen: a list of data URLs (PNG, or WebP if a PNG
 * is too large to send), with the print patterns over it. */
async function captureFrames(X, P, o, onFrame, canceled) {
  const v = X.viewer, el = document.getElementById(v.viewerId) || v.getCanvas().parentElement;
  const canvas = v.getCanvas();
  const dpr = (window.devicePixelRatio || 1) * (window.maceZoom ? window.maceZoom.of(el) : 1);
  const saved = el.getAttribute("style") || "";
  const pat = o.patterns !== "off" ? await import("../viewport/print_patterns.js") : null;
  v.render();
  const still = document.createElement("img");
  still.src = canvas.toDataURL("image/png");
  still.className = "export-still";
  still.style.cssText = `position:absolute;left:${el.offsetLeft}px;top:${el.offsetTop}px;width:${el.offsetWidth}px;height:${el.offsetHeight}px;pointer-events:none;z-index:5`;
  el.parentElement.appendChild(still);
  const bg = X.bg();
  try {
    if (o.background === "transparent") v.setBackgroundColor(bg, 0);
    else if (o.background === "white") v.setBackgroundColor(tok("--viewport-white"), 1);
    else if (o.background === "black") v.setBackgroundColor(tok("--viewport-black"), 1);
    el.setAttribute("style", `${saved};position:fixed;left:-100000px;top:0;width:${o.width / dpr}px;height:${o.height / dpr}px`);
    v.resize();
    await frame(); await frame();
    for (let k = 0; k < P.n; k += o.every) {
      if (canceled()) throw new Error("canceled");
      await P.set(k);
      v.render();
      // read in this task: the canvas keeps the frame only until the next composite
      let url = v.getCanvas().toDataURL("image/png");
      if (pat) {
        const over = pat.overlayCanvas(v, o.width, o.height, o.patterns, dpr);
        if (over) url = await composite(url, over, o.width, o.height);
      }
      if (url.length > MAX_POST) url = await recode(url, o.width, o.height);
      if (url.length > MAX_POST) throw new Error("a frame is too large to send at this size: choose a smaller size or scale");
      await onFrame(url, k);
    }
  } finally {
    el.setAttribute("style", saved);
    v.setBackgroundColor(bg, 1);
    v.resize();
    v.render();
    await frame(); await frame();
    v.render();
    still.remove();
  }
}
const loadImage = (url) => new Promise((resolve, reject) => { const i = new Image(); i.onload = () => resolve(i); i.onerror = () => reject(new Error("a frame could not be read")); i.src = url; });
async function composite(url, over, w, h) {
  const img = await loadImage(url), c = document.createElement("canvas");
  c.width = w; c.height = h;
  const g = c.getContext("2d");
  g.drawImage(img, 0, 0, w, h);
  g.drawImage(over, 0, 0, w, h);
  return c.toDataURL("image/png");
}
async function recode(url, w, h) {
  const img = await loadImage(url), c = document.createElement("canvas");
  c.width = w; c.height = h;
  c.getContext("2d").drawImage(img, 0, 0, w, h);
  const webp = c.toDataURL("image/webp", 0.95);
  return webp.startsWith("data:image/webp") ? webp : url;
}

/** The dialog. `X` (the shell's): {M, apiTask, doc, kind, viewer, stem, path, bg, setting, download, freq, traj}. */
export async function open(H, X) {
  await loadCss(import.meta.url, "../../css/dialogs.css");
  const P = provider(X), fileOk = !!(X.path && /\.(out|o|log|xyz|extxyz|traj)$/i.test(X.path));
  if (!P && !fileOk) { H.toast("Open a FREQ output's modes, an optimization, or an MD or NEB trajectory", "info"); return null; }
  const p = prefs(X);
  if (!P) p.source = "file";
  if (!fileOk) p.source = "view";
  let caps = { gif: true, webm: false, mp4: false, why: "" };
  try { caps = await X.M.api("anim/caps"); } catch (e) { /* GIF only */ }
  const M = maxSide(), esc = H.esc;
  const body = document.createElement("div");
  body.className = "export-dlg anim-dlg";
  const opt = (v, t, cur, dis = "") => `<option value="${v}"${String(v) === String(cur) ? " selected" : ""}${dis ? ` disabled title="${esc(dis)}"` : ""}>${t}${dis ? " (needs ffmpeg)" : ""}</option>`;
  body.innerHTML = `<label class="f">Source<select class="field" data-k="source" aria-label="Source">
      ${P ? opt("view", `This view: ${P.label}`, p.source) : ""}${fileOk ? opt("file", "Rendered by MACE from the file (as mace animate)", p.source) : ""}</select></label>
    <label class="f">Format<select class="field" data-k="format" aria-label="Format">${opt("gif", "GIF", p.format)}
      ${opt("webm", "WebM (VP9)", p.format, caps.webm ? "" : caps.why || "ffmpeg")}${opt("mp4", "MP4 (H.264)", p.format, caps.mp4 ? "" : caps.why || "ffmpeg")}</select></label>
    ${caps.why ? `<p class="note">${esc(caps.why)}</p>` : ""}
    <div class="row"><label class="f">Frames per second<input class="field" type="number" min="1" max="60" step="1" data-k="fps" value="${p.fps}"></label>
      <label class="f">Plays<select class="field" data-k="loop">${opt("0", "Forever", p.loop)}${opt("1", "Once", p.loop)}${opt("n", "A number of times", p.loop)}</select></label>
      <label class="f loop-n">Times<input class="field" type="number" min="2" max="100" step="1" data-k="loopN" value="${p.loopN}"></label></div>
    <div class="row"><label class="f">Size<select class="field" data-k="size">${SIZES.map(([v, t]) => opt(v, t, p.size)).join("")}</select></label>
      <label class="f">Scale<select class="field" data-k="scale">${SCALES.map(([v, t]) => opt(v, t, p.scale)).join("")}</select></label></div>
    <div class="row custom"><label class="f">Width<input class="field" type="number" min="16" max="${M}" step="1" data-k="w" value="${p.w}"></label>
      <label class="f">Height<input class="field" type="number" min="16" max="${M}" step="1" data-k="h" value="${p.h}"></label></div>
    <label class="f">Background<select class="field" data-k="background">${opt("theme", "The view's (theme)", p.background)}${opt("transparent", "Transparent (GIF, WebM)", p.background)}
      ${opt("white", "White", p.background)}${opt("black", "Black", p.background)}</select></label>
    <label class="f">Print patterns<select class="field" data-k="patterns">${opt("off", "Off", p.patterns)}${opt("auto", "Similar grays", p.patterns)}${opt("all", "Every element", p.patterns)}</select></label>
    <label class="f every-f">Keep every n-th step<input class="field" type="number" min="1" max="100" step="1" data-k="every" value="${p.every}"></label>
    <p class="note" data-k="px"></p>
    <label class="f">File name<input class="field" data-k="name" spellcheck="false" autocomplete="off" value="${esc(`${X.stem}.${p.format}`)}"></label>
    <div class="anim-progress" hidden><progress max="1" value="0"></progress><span class="note" data-k="status"></span></div>
    <div class="err" role="alert"></div>`;
  const q = (k) => body.querySelector(`[data-k="${k}"]`);
  const read = () => ({ source: q("source").value, format: q("format").value, fps: Math.max(1, Math.min(60, Number(q("fps").value) || 20)),
    loop: q("loop").value, loopN: Math.max(2, Math.min(100, Number(q("loopN").value) || 3)), size: q("size").value, scale: q("scale").value,
    w: Number(q("w").value), h: Number(q("h").value), background: q("background").value, patterns: q("patterns").value,
    every: Math.max(1, Math.min(100, Number(q("every").value) || 1)) });
  const dims = (r) => {
    const cv = X.viewer ? X.viewer.getCanvas() : { width: 800, height: 600 };
    const [w, h] = targetSize(r.size, cv, { w: r.w, h: r.h });
    return [Math.round(w * Number(r.scale)), Math.round(h * Number(r.scale))];
  };
  const check = () => {
    const r = read(), [w, h] = dims(r);
    body.querySelector(".custom").hidden = r.size !== "custom";
    body.querySelector(".loop-n").hidden = r.loop !== "n";
    body.querySelector(".every-f").hidden = !X.traj && !(r.source === "file" && !X.freq);
    const nm = q("name");
    nm.value = nm.value.replace(/\.(gif|webm|mp4)$/i, "") + "." + r.format;
    const big = Math.max(w, h) > Math.min(M, 4096), small = Math.min(w, h) < 16;
    q("px").textContent = `${w} × ${h} pixels · ${r.fps} frames per second${r.format === "mp4" && r.background === "transparent" ? " · MP4 has no transparency: the background is white" : ""}`;
    body.querySelector(".err").textContent = big ? `At most ${Math.min(M, 4096)} pixels a side` : small ? "At least 16 pixels a side" : "";
    return big || small ? null : [w, h];
  };
  body.addEventListener("input", check);
  body.addEventListener("change", check);
  let busy = false, cancel = false, key = null;
  const d = panel(H, { id: "export-animation", title: "Export Animation", body,
    foot: `<button class="tb-btn-text" data-v="cancel">Cancel</button><button class="tb-btn-text primary" data-v="export">Export</button>`,
    onClose: () => { cancel = true; if (key && busy) X.M.api("anim/cancel", { body: { key } }).catch(() => {}); } });
  check();
  const status = (t, x) => { body.querySelector(".anim-progress").hidden = false; q("status").textContent = t; if (x !== undefined) body.querySelector("progress").value = x; };
  d.querySelector('[data-v="cancel"]').addEventListener("click", () => d.close());
  d.querySelector('[data-v="export"]').addEventListener("click", async () => {
    if (busy) return;
    const wh = check();
    if (!wh) return;
    const r = read();
    profile.remember("exportAnimation", { ...r, [X.freq ? "fpsFreq" : "fpsSteps"]: r.fps });
    const loop = r.loop === "n" ? r.loopN : Number(r.loop);
    let name = q("name").value.trim() || `${X.stem}.${r.format}`;
    busy = true; cancel = false;
    d.querySelector('[data-v="export"]').disabled = true;
    body.querySelector(".err").textContent = "";
    try {
      let res;
      if (r.source === "file") {
        status("Drawing the frames on the server…", 0.1);
        const tone = getComputedStyle(document.documentElement).colorScheme === "dark" ? "dark" : "light";
        const fd = X.freq && X.freq.doc ? X.freq.doc() : null;
        res = await X.apiTask("anim/render", { path: X.path, format: r.format, fps: r.fps, loop, width: wh[0], height: wh[1], scale: 1,
          background: r.background === "theme" ? tone : r.background, patterns: r.patterns, every: r.every, name,
          grayscale: (X.setting && X.setting("atomColor")) === "gray",
          // the document's bond criteria: the frames draw the bonds the viewer draws
          bonds: window.__mace && window.__mace.bondValues ? window.__mace.bondValues() : undefined,
          ...(fd && fd.mode ? { mode: fd.mode.mode, amplitude: fd.opts ? fd.opts.amplitude : 1,
            amplitude_scale: fd.opts && fd.opts.scale === "normalized" ? "normalized" : "physical" } : {}) });
      } else {
        const start = await X.M.api("anim/start", { body: {} });
        key = start.key;
        await P.start();
        const n = Math.ceil(P.n / r.every);
        let sent = 0;
        try {
          await captureFrames(X, P, { width: wh[0], height: wh[1], background: r.background, patterns: r.patterns, every: r.every },
            async (url, k) => {
              await X.M.api("anim/frame", { body: { key, index: k, data: url } });
              sent++;
              status(`Captured frame ${sent} of ${n}`, 0.8 * sent / n);
            }, () => cancel);
        } finally { await P.restore(); }
        if (cancel) throw new Error("canceled");
        status("Writing the file…", 0.85);
        const white = tok("--viewport-white"), black = tok("--viewport-black");
        const bgHex = r.background === "white" ? white : r.background === "black" ? black : /^#[0-9a-f]{6}$/i.test(X.bg()) ? X.bg() : white;
        res = await X.apiTask("anim/encode", { key, format: r.format, fps: r.fps, loop, transparent: r.background === "transparent",
          matte: r.background === "transparent" ? white : bgHex, name });
      }
      status("Downloading…", 0.95);
      const got = await X.M.api(`anim/result?key=${encodeURIComponent(res.key)}`, { blob: true });
      name = res.name || name;
      if (window.__maceAnimHook) window.__maceAnimHook({ ...res, blobBytes: got.blob.size, type: got.blob.type }, got.blob);
      X.download(got.blob, name);
      d.close();
      H.toast(`Exported ${name} (${res.frames} frames, ${res.size[0]} × ${res.size[1]}, ${(got.blob.size / 1e6).toFixed(2)} MB)`, "success");
    } catch (e) {
      if (key) X.M.api("anim/cancel", { body: { key } }).catch(() => {});
      if (!cancel) body.querySelector(".err").textContent = e.message || String(e);
    } finally {
      busy = false; key = null;
      const b = d.querySelector('[data-v="export"]');
      if (b) b.disabled = false;
    }
  });
  return d;
}
