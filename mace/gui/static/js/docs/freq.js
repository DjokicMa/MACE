/* The FREQ document (spec §1.2, plan B5, T9): an output's normal modes, animated in
 * a 3Dmol viewport (native, no CDN, as Avogadro does), a table of the modes in the
 * bottom panel with an IR/Raman stick spectrum, and the mode controls in the
 * inspector. The animation renders on demand at most 30 times a second and stops
 * while the tab is hidden or the window has no focus; under reduced motion the
 * mode shows as a still view with displacement arrows and a Play button. */
import * as PK from "../plotkit.js";

const FRAMES = 24, FPS = 30;

export function createFreqView(host, tableHost, optsHost, H) {
  const { api, q, esc, icon, toast, apiTask, download, M } = H;
  host.innerHTML = `<div class="fv"><div class="fv-head"><div class="fv-title"></div><div class="fv-mode mono"></div>
      <button class="tb-btn-text fv-struct" data-act="structure" data-tip="Open this output's structure">${icon("atom")}<span>Structure</span></button>
      <button class="tb-btn-text fv-ir" data-act="ir" data-tip="IR spectrum: CRYSTAL's IRSPEC.DAT, else broadened from the modes">${icon("chart-spline")}<span>IR</span></button>
      <button class="tb-btn-text fv-raman" data-act="raman" data-tip="Raman spectrum: CRYSTAL's RAMSPEC.DAT, else broadened from the modes">${icon("chart-spline")}<span>Raman</span></button>
      <button class="tb-btn-text fv-steps" data-act="steps" hidden data-tip="The optimization before the frequencies">${icon("chart-line")}<span>Pre-optimization Steps</span></button>
      <button class="tb-btn-text primary fv-play" data-act="play" aria-pressed="false">${icon("play")}<span>Play</span><kbd>Space</kbd></button></div>
    <div class="fv-view"><div class="viewer" id="viewer-freq" tabindex="0" aria-label="Normal mode"></div>
      <div class="fv-msg" hidden></div></div></div>`;
  tableHost.innerHTML = `<div class="fm"><div class="pv-acts fm-acts"></div><svg class="fm-spec" role="img" aria-label="IR and Raman sticks"></svg>
    <div class="fm-grid" role="grid" aria-label="Normal modes" tabindex="0"><div class="fm-head" role="row" aria-rowindex="1">
      <span role="columnheader">#</span><span role="columnheader" class="num">cm⁻¹</span><span role="columnheader">Irrep</span>
      <span role="columnheader" class="num">IR</span><span role="columnheader" class="num">Raman</span><span role="columnheader">Active</span></div>
      <div class="fm-rows" role="rowgroup"></div></div></div>`;
  const $ = (s, el = host) => el.querySelector(s);
  const grid = $(".fm-grid", tableHost), rowsEl = $(".fm-rows", tableHost);
  let doc = null, viewer = null, timer = 0, playing = false, frame = 0, phase = 0, last = 0, building = 0;
  const stats = { frames: 0 };
  window.__maceFreqFrames = 0;

  const reduced = () => document.documentElement.dataset.motion === "reduce"
    || (document.documentElement.dataset.motion !== "full" && matchMedia("(prefers-reduced-motion: reduce)").matches);
  const defaults = () => ({ amplitude: 1, speed: 1, arrows: true, arrowScale: 2 });

  function v() { if (!viewer) viewer = M().viewer("viewer-freq"); return viewer; }

  /* What the toolbar's display options draw (the document's display range and
   * boundary atoms, the unit cell, bonds, labels, the atom style), and where:
   * every drawn atom, each periodic image of a site too, moves as its site
   * does (the modes are at Γ, where every cell moves alike); a mode away from Γ
   * would need its phases, so it draws one cell. */
  let G = null;            // geom/core.js, loaded with the first document
  let lastDisplay = null;   // what the last draw showed (state(), for the tests)
  function display(d) {
    const s = d.structure, vo = H.viewOpts ? H.viewOpts() : {}, per = s.pbc.some(Boolean), many = per && d.gamma !== false;
    const range = many ? (vo.range || [1, 1, 1]) : [1, 1, 1];
    const box = many ? (vo.box || { lo: [0, 0, 0], hi: Array.isArray(range) ? range : [1, 1, 1] }) : { lo: [0, 0, 0], hi: [1, 1, 1] };
    const list = G.displayList(s, range, many && !!vo.boundary);
    return { list, box, cell: per && vo.cell !== false, bonds: vo.bonds !== false, labels: vo.labels || "off", style: vo.style || "ballstick" };
  }
  /* the drawn cell box (from box.lo to box.hi cells along the periodic axes), as a structure for drawCell and the fit */
  function boxed(s, box) {
    const C = s.cell, o = (s.origin || [0, 0, 0]).slice();
    const cell = C.map((r, k) => (s.pbc[k] ? r.map((x) => x * (box.hi[k] - box.lo[k])) : r.slice()));
    for (let k = 0; k < 3; k++) if (s.pbc[k]) for (let j = 0; j < 3; j++) o[j] += box.lo[k] * C[k][j];
    return { ...s, cell, origin: o };
  }
  const cartOf = (list, k) => [list.cart[3 * k], list.cart[3 * k + 1], list.cart[3 * k + 2]];
  /* the mode's scale: its largest displacement drawn as 0.5 Å (amplitude 1×), whatever CRYSTAL's normalization */
  let drawnArrows = [];   // the still view's arrows (state(), for the tests)
  let frameArrows = [];   // while playing: the arrows of each animation frame (3Dmol shows a shape only in its frame)
  // arrows during the animation are FRAMES copies of each arrow; above this many shapes they show only when paused
  const PLAY_ARROWS_MAX = 24000;
  const ARROW_FROM = 0.5;   // Å: about the drawn sphere's radius (0.28 × the van der Waals radius)
  const scaleOf = (disp) => { const m = disp ? Math.max(...disp.map((v) => Math.hypot(...v))) : 0; return m > 1e-9 ? 0.5 / m : 0; };
  /* frames of one mode: XYZ text, FRAMES per cycle (3Dmol keeps each as a frame) */
  function framesXyz(d, disp, list = display(d).list) {
    const s = d.structure, o = d.opts, out = [], k0 = scaleOf(disp);
    for (let f = 0; f < FRAMES; f++) {
      const w = disp ? Math.sin((2 * Math.PI * f) / FRAMES) * o.amplitude * k0 : 0;
      out.push(String(list.n), `${d.title} mode ${d.mode ? d.mode.mode : 0} frame ${f + 1}`);
      for (let k = 0; k < list.n; k++) {
        const i = list.base[k], c = cartOf(list, k), dv = disp ? disp[i] || [0, 0, 0] : [0, 0, 0];
        out.push(`${s.symbols[i]} ${(c[0] + w * dv[0]).toFixed(5)} ${(c[1] + w * dv[1]).toFixed(5)} ${(c[2] + w * dv[2]).toFixed(5)}`);
      }
    }
    return out.join("\n") + "\n";
  }

  function draw(keepView = true) {
    if (!doc || !doc.structure || !G) return;
    const vw = v();
    if (!vw) return;                // no WebGL: the viewport explains it
    const view = keepView && vw.getModel() ? vw.getView() : null;
    vw.removeAllModels(); vw.removeAllShapes(); vw.removeAllLabels();
    const s = doc.structure, disp = doc.mode && doc.mode.displacement, dd = display(doc), list = dd.list;
    lastDisplay = dd;
    const m = vw.addModelsAsFrames(framesXyz(doc, disp, list), "xyz");
    // bonds from the rest geometry in every frame: they never flicker as atoms swing past the bond cutoff
    const fr = m.frames || [];
    if (fr.length > 1 && fr[0]) {
      for (let f = 1; f < fr.length; f++) {
        fr[f].forEach((a, i) => { const r = fr[0][i]; if (r) { a.bonds = r.bonds.slice(); a.bondOrder = (r.bondOrder || []).slice(); } });
      }
    }
    if (dd.style === "wireframe") {
      const cs = M().state.colors ? { prop: "elem", map: M().state.colors.schemes[M().schemeName()] } : "Jmol";
      m.setStyle({}, dd.bonds ? { line: { colorscheme: cs }, sphere: { scale: 0.12, colorscheme: cs } } : { sphere: { scale: 0.12, colorscheme: cs } });
    } else m.setStyle({}, M().styleFor(dd.style, dd.bonds));
    const shown = boxed(s, dd.box);
    if (dd.cell) M().drawCell(vw, shown, M().cssVar("--cell"), 0.025, 1);
    // labels at the atoms' rest positions (the cell's own atoms, as the structure view labels them)
    if (dd.labels && dd.labels !== "off") {
      const w = s.symmetry && s.symmetry.available ? s.symmetry.wyckoffs : null, size = Number(H.setting && H.setting("labelSize")) || 12;
      for (let k = 0; k < list.n; k++) {
        if (list.image[k]) continue;
        const i = list.base[k], c = cartOf(list, k), sym = s.symbols[i];
        const text = dd.labels === "element" ? sym : dd.labels === "index" ? String(i + 1) : dd.labels === "wyckoff" && w ? `${sym} ${w[i]}` : `${sym}${i + 1}`;
        vw.addLabel(text, { position: { x: c[0], y: c[1], z: c[2] }, fontSize: size, fontColor: M().cssVar("--text"),
          backgroundColor: M().cssVar("--surface"), backgroundOpacity: 0.75, borderThickness: 0, inFront: true });
      }
    }
    drawnArrows = [];
    frameArrows = [];
    doc.arrowsCapped = false;
    const moving = (k) => {
      const kk = scaleOf(disp) * doc.opts.arrowScale, out = [];
      for (let j = 0; j < list.n; j++) {
        const dv = disp[list.base[j]];
        if (dv && Math.hypot(...dv) * kk >= 0.05) out.push([cartOf(list, j), dv]);
      }
      return out;
    };
    if (disp && doc.opts.arrows && playing) {
      // arrows that move with their atoms: one set per animation frame, each shape tagged with its frame
      // (3Dmol draws a shape with a frame only while the viewer shows that frame), so toggling them never
      // stops the animation and no shape is rebuilt per frame
      const col = M().cssVar("--sel"), k = scaleOf(disp) * doc.opts.arrowScale, k0 = scaleOf(disp);
      const live = moving();
      if (live.length * FRAMES > PLAY_ARROWS_MAX) doc.arrowsCapped = true;
      else {
        for (let f = 0; f < FRAMES; f++) {
          const w = Math.sin((2 * Math.PI * f) / FRAMES) * doc.opts.amplitude * k0, arr = [];
          for (const [c, dv] of live) {
            const len = Math.hypot(...dv), u = dv.map((x) => x / len);
            const s0 = c.map((x, j) => x + w * dv[j] + u[j] * ARROW_FROM);
            arr.push({ atom: c, start: s0, length: len * k });
            vw.addArrow({ start: { x: s0[0], y: s0[1], z: s0[2] }, end: { x: s0[0] + dv[0] * k, y: s0[1] + dv[1] * k, z: s0[2] + dv[2] * k },
              radius: 0.07, radiusRatio: 2, mid: 0.72, color: col, frame: f });
          }
          frameArrows.push(arr);
        }
      }
    }
    if (disp && doc.opts.arrows && !playing) {
      const col = M().cssVar("--sel"), k = scaleOf(disp) * doc.opts.arrowScale;
      for (const [c, dv] of moving()) {
        const len = Math.hypot(...dv);
        // from the sphere's surface (the atoms sit at rest in the still view), so no arrow is buried in its atom
        const u = dv.map((x) => x / len), s0 = c.map((x, j) => x + u[j] * ARROW_FROM);
        drawnArrows.push({ atom: c, start: s0, length: len * k });
        vw.addArrow({ start: { x: s0[0], y: s0[1], z: s0[2] }, end: { x: s0[0] + dv[0] * k, y: s0[1] + dv[1] * k, z: s0[2] + dv[2] * k },
          radius: 0.07, radiusRatio: 2, mid: 0.72, color: col });
      }
    }
    if (view) vw.setView(view);
    else M().fitView(vw, shown, Array.from({ length: list.n }, (_, k) => ({ i: list.base[k], cart: cartOf(list, k) })), 1);
    // still: the rest geometry under the arrows (the arrows show the motion); no arrows: the turning point
    setFrame(playing ? frame : disp && doc.opts.arrows ? 0 : FRAMES / 4 | 0);
  }
  function setFrame(k) {
    const vw = v();
    frame = k;
    if (!vw) return;
    const r = vw.setFrame ? vw.setFrame(k) : null;
    const paint = () => { vw.render(); stats.frames++; window.__maceFreqFrames = stats.frames; };
    if (r && r.then) r.then(paint); else paint();
  }

  /* the loop: on demand, ≤30 fps, never while hidden or unfocused */
  function tick(now) {
    timer = 0;
    if (!playing || !doc) return;
    if (document.hidden || !document.hasFocus()) { last = now; return; }        // resumes on focus / visibility
    if (now - last >= 1000 / FPS - 1) {
      phase = (phase + ((now - last) / 1000) * doc.opts.speed) % 1;
      last = now;
      const k = Math.floor(phase * FRAMES) % FRAMES;
      if (k !== frame) setFrame(k);
    }
    timer = requestAnimationFrame(tick);
  }
  function play(on = !playing) {
    playing = on;
    $(".fv-play").setAttribute("aria-pressed", String(on));
    $(".fv-play").innerHTML = `${icon(on ? "pause" : "play")}<span>${on ? "Pause" : "Play"}</span><kbd>Space</kbd>`;
    draw(true);
    if (on && !timer) { last = performance.now(); timer = requestAnimationFrame(tick); }
    if (!on && timer) { cancelAnimationFrame(timer); timer = 0; }
  }
  const wake = () => { if (playing && !timer && !document.hidden) { last = performance.now(); timer = requestAnimationFrame(tick); } };
  document.addEventListener("visibilitychange", wake);
  addEventListener("focus", wake);

  /* the modes table */
  function fmt(x, d = 2) { return x === null || x === undefined ? "—" : Number(x).toFixed(d); }
  function renderTable() {
    const ms = doc.modes;
    grid.setAttribute("aria-rowcount", String(ms.length + 1));
    rowsEl.innerHTML = ms.map((m, k) => `<div class="fm-row${doc.mode && doc.mode.mode === m.mode ? " sel" : ""}${m.imaginary ? " imag" : ""}" role="row"
      aria-rowindex="${k + 2}" aria-selected="${!!(doc.mode && doc.mode.mode === m.mode)}" tabindex="${k === doc.active ? 0 : -1}" data-k="${k}">
      <span role="gridcell">${m.mode}</span><span role="gridcell" class="num">${fmt(m.freq, 1)}</span><span role="gridcell">${esc(m.irrep)}</span>
      <span role="gridcell" class="num">${m.ir_active ? fmt(m.ir_intens, 2) : "—"}</span><span role="gridcell" class="num">${m.raman_active ? fmt(m.raman_intens, 2) : "—"}</span>
      <span role="gridcell">${[m.ir_active ? "IR" : "", m.raman_active ? "Raman" : ""].filter(Boolean).join(" · ") || "silent"}</span></div>`).join("");
    sticks();
  }
  function sticks() {
    const svg = $(".fm-spec", tableHost), ms = doc.modes.filter((m) => !m.imaginary && m.freq > 1);
    const W = Math.max(300, svg.clientWidth || 600), Hh = 80, max = Math.max(...ms.map((m) => m.freq), 100);
    const irMax = Math.max(...ms.map((m) => m.ir_intens || 0), 1e-9), raMax = Math.max(...ms.map((m) => m.raman_intens || 0), 1e-9);
    const x = (f) => 8 + (W - 16) * (f / max);
    const ir = ms.filter((m) => m.ir_active && m.ir_intens > 0).map((m) => `<line class="ir" data-mode="${m.mode}" x1="${x(m.freq)}" x2="${x(m.freq)}" y1="${Hh / 2}" y2="${Hh / 2 - (Hh / 2 - 4) * m.ir_intens / irMax}"><title>${fmt(m.freq, 1)} cm⁻¹ IR ${fmt(m.ir_intens)}</title></line>`).join("");
    const ra = ms.filter((m) => m.raman_active && m.raman_intens > 0).map((m) => `<line class="ra" data-mode="${m.mode}" x1="${x(m.freq)}" x2="${x(m.freq)}" y1="${Hh / 2}" y2="${Hh / 2 + (Hh / 2 - 4) * m.raman_intens / raMax}"><title>${fmt(m.freq, 1)} cm⁻¹ Raman ${fmt(m.raman_intens)}</title></line>`).join("");
    const sel = doc.mode ? `<line class="cur" x1="${x(doc.mode.freq)}" x2="${x(doc.mode.freq)}" y1="0" y2="${Hh}"/>` : "";
    svg.setAttribute("viewBox", `0 0 ${W} ${Hh}`);
    svg.innerHTML = `<line class="axis" x1="0" x2="${W}" y1="${Hh / 2}" y2="${Hh / 2}"/>${ir}${ra}${sel}${!ir && !ra ? `<text x="${W / 2}" y="${Hh / 2 - 6}" class="lab" text-anchor="middle">Every IR and Raman intensity in this output is zero or not computed</text>` : ""}<text x="8" y="12" class="lab">IR ↑</text><text x="8" y="${Hh - 4}" class="lab">Raman ↓</text>`;
  }
  async function select(k) {
    const m = doc.modes[k];
    if (!m) return;
    doc.active = k;
    const hadFocus = tableHost.contains(document.activeElement);
    try {
      const r = await api("freq/mode?" + q({ path: doc.path, k: m.mode }));
      doc.mode = r;
      $(".fv-mode").textContent = `Mode ${m.mode} · ${fmt(m.freq, 1)} cm⁻¹ (${m.irrep})${m.imaginary ? " · imaginary" : ""}`;
      // the server could not match CRYSTAL's Cartesian frame to the drawn cell: say so rather than draw arrows that may be turned
      if (r.frame === "crystal" && !doc.frameWarned) { doc.frameWarned = true; toast("The mode's directions are in CRYSTAL's Cartesian frame, which could not be matched to the drawn cell: arrows may be rotated", "warning"); }
      renderTable();
      if (hadFocus) { const row = rowsEl.querySelector(`[data-k="${doc.active}"]`); if (row) row.focus(); }
      draw(true);
      if (!playing && !reduced() && doc.autoplay !== false) play(true);
    } catch (e) { toast(e.message, "error"); }
  }
  /** Select a mode by its number (a spectrum peak, Run Analysis' "Show Modes"). */
  function selectMode(n) {
    if (!doc || !doc.modes) { if (doc) doc.wantMode = n; return Promise.resolve(); }
    const k = doc.modes.findIndex((m) => m.mode === Number(n));
    return k >= 0 ? select(k) : Promise.resolve();
  }
  rowsEl.addEventListener("click", (e) => { const r = e.target.closest("[data-k]"); if (r) select(Number(r.dataset.k)); });
  /* the stick spectrum's export (plotkit): the SVG as drawn, the modes as CSV or TSV */
  PK.attach($(".fm-acts", tableHost), {
    get name() { return doc ? doc.title.replace(/[^A-Za-z0-9._-]+/g, "_") : "modes"; },
    host: $(".fm-spec", tableHost),
    features: { logY: false, units: null, range: false, width: false, png: false, svg: true, csv: true, pub: false },
    get: () => ({}), set: () => {},
    table: () => ({ columns: ["mode", "frequency_cm-1", "irrep", "ir_intensity", "raman_intensity", "imaginary"],
      rows: (doc ? doc.modes : []).map((m) => [m.mode, m.freq, m.irrep, m.ir_active ? m.ir_intens : "", m.raman_active ? m.raman_intens : "", m.imaginary ? 1 : 0]) }),
    svg: () => {
      const svg = $(".fm-spec", tableHost).cloneNode(true);
      svg.setAttribute("xmlns", "http://www.w3.org/2000/svg");
      download(new Blob([svg.outerHTML], { type: "image/svg+xml" }), `${doc ? doc.title.replace(/[^A-Za-z0-9._-]+/g, "_") : "modes"}_sticks.svg`);
    },
  }, { toast, download, icon });
  grid.addEventListener("keydown", (e) => {
    if (!doc) return;
    const n = doc.modes.length, page = 8;
    const to = { ArrowDown: doc.active + 1, ArrowUp: doc.active - 1, PageDown: doc.active + page, PageUp: doc.active - page, Home: 0, End: n - 1 }[e.key];
    if (to !== undefined) { e.preventDefault(); doc.active = Math.max(0, Math.min(n - 1, to)); renderTable(); rowsEl.querySelector(`[data-k="${doc.active}"]`).focus(); return; }
    if (e.key === "Enter") { e.preventDefault(); select(doc.active); }
    if (e.key === " ") { e.preventDefault(); play(); }
  });
  grid.addEventListener("focus", (e) => { if (e.target === grid && doc) { const r = rowsEl.querySelector(`[data-k="${doc.active}"]`); if (r) r.focus(); } });
  host.addEventListener("click", (e) => {
    if (e.target.closest("[data-act=play]")) play();
    else if (e.target.closest("[data-act=structure]") && doc && H.openStructure) H.openStructure(doc.path);
    else if (e.target.closest("[data-act=steps]") && doc && H.openStructure) H.openStructure(doc.path);
    else if (e.target.closest("[data-act=ir]") && doc && H.openSpectrum) H.openSpectrum(doc.path, "ir");
    else if (e.target.closest("[data-act=raman]") && doc && H.openSpectrum) H.openSpectrum(doc.path, "raman");
  });
  /* the stick spectrum: a click picks the nearest IR- or Raman-active mode */
  $(".fm-spec", tableHost).addEventListener("click", (e) => {
    const l = e.target.closest("[data-mode]");
    if (l && doc) selectMode(Number(l.dataset.mode));
  });
  host.addEventListener("keydown", (e) => { if (e.key === " " && e.target.closest(".viewer")) { e.preventDefault(); play(); } });

  /* the mode controls (inspector) */
  function renderOpts() {
    const o = doc.opts;
    optsHost.innerHTML = `<label class="po-row"><span>Amplitude</span><input type="range" id="fo-amp" min="0.2" max="4" step="0.1" value="${o.amplitude}"><span class="po-u">${o.amplitude.toFixed(1)}×</span></label>
      <label class="po-row"><span>Speed</span><input type="range" id="fo-speed" min="0.2" max="3" step="0.1" value="${o.speed}"><span class="po-u">${o.speed.toFixed(1)}/s</span></label>
      <label class="chk po-row"><input type="checkbox" id="fo-arrows"${o.arrows ? " checked" : ""}> Arrows</label>
      ${doc.arrowsCapped ? `<p class="muted small po-row">This structure has too many moving atoms for arrows during the animation: they show when paused</p>` : ""}
      <label class="po-row"><span>Arrow scale</span><input type="range" id="fo-ascale" min="0.5" max="8" step="0.5" value="${o.arrowScale}"><span class="po-u">${o.arrowScale}×</span></label>
      ${doc.structure && doc.structure.pbc.some(Boolean) ? (doc.gamma === false
    ? `<p class="muted small po-row">One cell: this mode is not at Γ, so its periodic images move with phases the output does not give</p>`
    : `<label class="chk po-row"><input type="checkbox" id="fo-super"${H.rangeN && H.rangeN() === 2 ? " checked" : ""}> 2×2×2 cells</label>`) : ""}
      <div class="po-acts"><button class="tb-btn-text" data-fo="xyz">${icon("download")}<span>Export Mode as XYZ</span></button></div>`;
  }
  optsHost.addEventListener("input", (e) => {
    if (!doc) return;
    const t = e.target, o = doc.opts;
    if (t.id === "fo-amp") o.amplitude = Number(t.value);
    else if (t.id === "fo-speed") o.speed = Number(t.value);
    else if (t.id === "fo-ascale") o.arrowScale = Number(t.value);
    else if (t.id === "fo-arrows") o.arrows = t.checked;
    else if (t.id === "fo-super") { if (H.setRange) H.setRange(t.checked ? 2 : 1); return; }
    else return;
    const u = t.parentElement.querySelector(".po-u");
    if (u) u.textContent = t.id === "fo-speed" ? `${o.speed.toFixed(1)}/s` : t.id === "fo-amp" ? `${o.amplitude.toFixed(1)}×` : `${o.arrowScale}×`;
    clearTimeout(building);
    building = setTimeout(() => draw(true), 60);
  });
  optsHost.addEventListener("click", (e) => {
    if (!doc || !e.target.closest("[data-fo=xyz]") || !doc.mode) return;
    download(new Blob([framesXyz(doc, doc.mode.displacement)], { type: "chemical/x-xyz" }),
      `${doc.title.replace(/\.[^.]+$/, "")}_mode${doc.mode.mode}.xyz`);
  });

  return {
    async open(d) {
      const first = !d.modes;
      doc = d;
      if (!d.opts) d.opts = defaults();
      $(".fv-title").textContent = d.title;
      const msg = $(".fv-msg");
      if (first) {
        msg.hidden = false; msg.innerHTML = `<span class="spinner"></span> Reading the normal modes…`;
        try {
          const r = await apiTask("freq?" + q({ path: d.path }));
          const core = await import("../geom/core.js");
          G = core;
          Object.assign(d, { modes: r.modes, structure: core.decodePacked(r.structure), imaginary: r.imaginary, active: 0,
            nFrames: r.n_frames || 1, gamma: r.gamma !== false });
          if (r.mode_atoms && r.mode_atoms !== r.n_atoms) throw new Error(`the modes move ${r.mode_atoms} atoms, the geometry has ${r.n_atoms}`);
          H.imaginary(d, r.imaginary);
        } catch (e) { msg.textContent = e.message; return; }
      }
      if (doc !== d) return;
      msg.hidden = true;
      renderTable();
      renderOpts();
      await M().load3Dmol();
      $(".fv-steps").hidden = !(d.nFrames > 1);          // FREQCALC with PREOPTGEOM: its optimization
      $(".fv-ir").hidden = !d.modes.some((m) => m.ir_active && m.ir_intens > 0);
      $(".fv-raman").hidden = !d.modes.some((m) => m.raman_active && m.raman_intens > 0);
      const want = d.wantMode ? d.modes.findIndex((m) => m.mode === d.wantMode) : -1;
      d.wantMode = null;
      const k = want >= 0 ? want : d.mode ? d.active : Math.max(0, d.modes.findIndex((m) => !m.imaginary && m.freq > 1));
      d.autoplay = !reduced();
      if (!d.mode || want >= 0) await select(k); else { draw(false); }
      if (reduced()) play(false);
    },
    close() { if (playing) play(false); doc = null; },
    /** The toolbar's display options changed (range, boundary atoms, cell, bonds, labels, style): redraw, camera kept. */
    redraw() { if (doc && doc.structure && G) { renderOpts(); draw(true); } },
    /** Reset View: the default view again. */
    resetView() { if (doc && doc.structure && G) draw(false); },
    viewer: () => viewer,
    doc: () => doc,
    play: (on) => play(on), playing: () => playing, select: (k) => select(k), selectMode: (n) => selectMode(n),
    stats: () => ({ ...stats, playing, frame, reduced: reduced() }),
    state: () => doc && { mode: doc.mode && doc.mode.mode, active: doc.active, frame,
      arrows: playing ? (frameArrows[frame] || []).slice() : drawnArrows.slice(), capped: !!doc.arrowsCapped,
      n: doc.modes ? doc.modes.length : 0, opts: { ...doc.opts }, gamma: doc.gamma !== false,
      display: lastDisplay && { atoms: lastDisplay.list.n, cell: lastDisplay.cell, bonds: lastDisplay.bonds, labels: lastDisplay.labels,
        style: lastDisplay.style, box: lastDisplay.box } },
  };
}
