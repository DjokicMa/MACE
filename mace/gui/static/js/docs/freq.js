/* The FREQ document (spec §1.2, plan B5, T9): an output's normal modes, animated in
 * a 3Dmol viewport (native, no CDN, as Avogadro does), a table of the modes in the
 * bottom panel with an IR/Raman stick spectrum, and the mode controls in the
 * inspector. The animation renders on demand at most 30 times a second and stops
 * while the tab is hidden or the window has no focus; under reduced motion the
 * mode shows as a still view with displacement arrows and a Play button.
 *
 * Amplitude (1.1.6). CRYSTAL prints each mode as NORMAL MODES NORMALIZED TO
 * CLASSICAL AMPLITUDES (IN BOHR): every atom's displacement at the classical
 * turning point of the vibrational ground state, x = √(ħ/ωμ) (CRYSTAL23
 * manual, SCANMODE; checked on the test corpus: Σ m|u|² · ω/ħ = 1.000 ± 0.004
 * for every real mode). freq.py turns it into Å, so "Physical" ×1 draws the
 * true zero-point amplitude (0.01 to 0.2 Å) and the Exaggeration slider
 * scales it, shown as ×N in the view's header. Before 1.1.6 the view always
 * scaled the largest atom to 0.5 Å and called that 1×: an exaggeration of 3×
 * to 40× by mode, never shown. "Largest atom 0.5 Å" keeps that drawing, now
 * named. Translations and rotations (≈0 cm⁻¹) have no zero-point amplitude:
 * their vector is drawn as CRYSTAL prints it. The arrows keep their own scale
 * (the largest 0.5 Å × Arrow length) in both. */
import * as PK from "../plotkit.js";
import { cellsPicker } from "../cells_picker.js";

const FRAMES = 24, FPS = 30;

export function createFreqView(host, tableHost, optsHost, H) {
  const { api, q, esc, icon, toast, apiTask, download, M } = H;
  host.innerHTML = `<div class="fv"><div class="fv-head"><div class="fv-title"></div><div class="fv-mode mono"></div>
      <button class="tb-btn-text fv-struct" data-act="structure" data-tip="Open this output's structure">${icon("atom")}<span>Structure</span></button>
      <button class="tb-btn-text fv-ir" data-act="ir" data-tip="IR spectrum: CRYSTAL's IRSPEC.DAT, else broadened from the modes">${icon("chart-spline")}<span>IR</span></button>
      <button class="tb-btn-text fv-raman" data-act="raman" data-tip="Raman spectrum: CRYSTAL's RAMSPEC.DAT, else broadened from the modes">${icon("chart-spline")}<span>Raman</span></button>
      <button class="tb-btn-text fv-diel" data-act="dielectric" hidden data-tip="Born charges, ε∞ and ε0, LO-TO splitting (INTCPHF): the output's dielectric results">${icon("table")}<span>Dielectric</span></button>
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
  const scaleSetting = () => (H.setting && H.setting("freq.amplitudeScale") === "normalized" ? "normalized" : "physical");
  const defaults = () => ({ amplitude: 1, speed: 1, arrows: true, arrowScale: 2, scale: scaleSetting() });

  function v() { if (!viewer) viewer = M().viewer("viewer-freq"); return viewer; }

  /* What the toolbar's display options draw (the document's display range and
   * boundary atoms, the unit cell, bonds, labels, the atom style), and where:
   * every drawn atom, each periodic image of a site too, moves as its site
   * does (the modes are at Γ, where every cell moves alike); a mode away from Γ
   * would need its phases, so it draws one cell. */
  let G = null;            // geom/core.js, loaded with the first document
  let lastDisplay = null;   // what the last draw showed (state(), for the tests)
  let lastBoxKey = null;    // the drawn document and box: a change refits the camera
  function display(d) {
    const s = d.structure, vo = H.viewOpts ? H.viewOpts() : {}, per = s.pbc.some(Boolean), many = per && d.gamma !== false;
    const range = many ? (vo.range || [1, 1, 1]) : [1, 1, 1];
    const box = many ? (vo.box || { lo: [0, 0, 0], hi: Array.isArray(range) ? range : [1, 1, 1] }) : { lo: [0, 0, 0], hi: [1, 1, 1] };
    const list = G.displayList(s, range, many && !!vo.boundary);
    // a molecule's cell: its Vacuum mode's box around the atoms, as the structure view draws it (1.1.6; Open: none)
    const box0 = !per && s.cell && M().vacuumBox ? M().vacuumBox(s, 1) : null;
    return { list, box, cell: (per || !!box0) && vo.cell !== false, bonds: vo.bonds !== false, labels: vo.labels || "off", style: vo.style || "ballstick" };
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
  const scaleOf = (disp) => { const m = maxOf(disp); return m > 1e-9 ? 0.5 / m : 0; };
  /* the largest atom's displacement as CRYSTAL prints it (Å): the zero-point classical amplitude */
  const maxOf = (disp) => (disp && disp.length ? Math.max(...disp.map((v) => Math.hypot(...v))) : 0);
  /* a translation or rotation (|ν| below 1 cm⁻¹): no zero-point amplitude, its printed vector is not one */
  const zeroMode = (m) => !!m && Math.abs(Number(m.freq) || 0) < 1;
  /* how much a displacement is multiplied when drawn: Physical, the Exaggeration itself; else the largest atom at 0.5 Å × Amplitude */
  const drawScale = (d, disp) => (d.opts.scale === "normalized" ? d.opts.amplitude * scaleOf(disp) : d.opts.amplitude);
  /* the slider: Physical ×1 to ×100 on a log scale (0 to 2), the old 0.2 to 4 linear for the fit */
  const ampToSlider = (o) => (o.scale === "normalized" ? o.amplitude : Math.log10(Math.max(1, o.amplitude)));
  const sliderToAmp = (o, v) => (o.scale === "normalized" ? v : Number((10 ** v).toPrecision(v < 1 ? 2 : 3)));
  const fmtX = (x) => (x >= 10 ? x.toFixed(0) : x >= 1 ? +x.toFixed(1) + "" : x.toFixed(2));
  /* frames of one mode: XYZ text, FRAMES per cycle (3Dmol keeps each as a frame) */
  function framesXyz(d, disp, list = display(d).list) {
    const s = d.structure, out = [], k0 = disp ? drawScale(d, disp) : 0;
    for (let f = 0; f < FRAMES; f++) {
      const w = disp ? Math.sin((2 * Math.PI * f) / FRAMES) * k0 : 0;
      out.push(String(list.n), `${d.title} mode ${d.mode ? d.mode.mode : 0} frame ${f + 1}${disp ? ` amplitude x${fmtX(k0)} of the zero-point classical amplitude` : ""}`);
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
    const s = doc.structure, disp = doc.mode && doc.mode.displacement, dd = display(doc), list = dd.list;
    // a new display range (or Vacuum mode) of the same document refits, as the structure view does
    const boxKey = JSON.stringify([doc.path, dd.box]);
    const view = keepView && vw.getModel() && boxKey === lastBoxKey ? vw.getView() : null;
    lastBoxKey = boxKey;
    vw.removeAllModels(); vw.removeAllShapes(); vw.removeAllLabels();
    lastDisplay = dd;
    const m = vw.addModelsAsFrames(framesXyz(doc, disp, list), "xyz");
    // bonds from the rest geometry in every frame: they never flicker as atoms swing past the bond cutoff
    const fr = m.frames || [];
    if (fr.length > 1 && fr[0]) {
      for (let f = 1; f < fr.length; f++) {
        fr[f].forEach((a, i) => { const r = fr[0][i]; if (r) { a.bonds = r.bonds.slice(); a.bondOrder = (r.bondOrder || []).slice(); } });
      }
    }
    // each drawn atom's site, in every frame (the Magnetic style colors by it)
    const n0 = list.n;
    (fr.length ? fr : [m.selectedAtoms({})]).forEach((atoms) => atoms.forEach((a, k) => { a.properties = { ...(a.properties || {}), base: list.base[k % n0] }; }));
    if (dd.style === "wireframe") {
      const cs = M().colorSpec ? M().colorSpec() : { colorscheme: M().colorScheme() };
      m.setStyle({}, dd.bonds ? { line: { ...cs }, sphere: { scale: 0.12, ...cs } } : { sphere: { scale: 0.12, ...cs } });
    } else m.setStyle({}, M().styleFor(dd.style, dd.bonds));
    const shown = boxed(s, dd.box);
    if (dd.cell) M().drawCell(vw, shown, M().vpVar("--cell"), 0.025, 1);
    // labels at the atoms' rest positions (the cell's own atoms, as the structure view labels them)
    if (dd.labels && dd.labels !== "off") {
      const w = s.symmetry && s.symmetry.available ? s.symmetry.wyckoffs : null, size = Number(H.setting && H.setting("labelSize")) || 12;
      for (let k = 0; k < list.n; k++) {
        if (list.image[k]) continue;
        const i = list.base[k], c = cartOf(list, k), sym = s.symbols[i];
        const text = dd.labels === "element" ? sym : dd.labels === "index" ? String(i + 1) : dd.labels === "wyckoff" && w ? `${sym} ${w[i]}` : `${sym}${i + 1}`;
        vw.addLabel(text, { position: { x: c[0], y: c[1], z: c[2] }, fontSize: size, fontColor: M().vpVar("--text"),
          backgroundColor: M().vpVar("--surface"), backgroundOpacity: 0.75, borderThickness: 0, inFront: true });
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
      const col = M().vpVar("--sel"), k = scaleOf(disp) * doc.opts.arrowScale, k0 = drawScale(doc, disp);
      const live = moving();
      if (live.length * FRAMES > PLAY_ARROWS_MAX) doc.arrowsCapped = true;
      else {
        for (let f = 0; f < FRAMES; f++) {
          const w = Math.sin((2 * Math.PI * f) / FRAMES) * k0, arr = [];
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
      const col = M().vpVar("--sel"), k = scaleOf(disp) * doc.opts.arrowScale;
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
  // a plot style change (Settings › Appearance) redraws the sticks in the new preset
  document.addEventListener("mace-theme", () => { try { if (doc && doc.modes && $(".fm-spec", tableHost)) sticks(); } catch (e) { /* not shown */ } });
  function sticks() {
    const svg = $(".fm-spec", tableHost), ms = doc.modes.filter((m) => !m.imaginary && m.freq > 1);
    // the Spectra family's preset (Settings › Appearance › Plot style): its stick colors and weights (freq.css)
    const pid = window.MACE && window.MACE.plotPreset ? window.MACE.plotPreset("spectra") : "default";
    if (pid !== "default") svg.setAttribute("data-plot-preset", pid); else svg.removeAttribute("data-plot-preset");
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
      modeLabel();
      // the server could not match CRYSTAL's Cartesian frame to the drawn cell: say so rather than draw arrows that may be turned
      if (r.frame === "crystal" && !doc.frameWarned) { doc.frameWarned = true; toast("The mode's directions are in CRYSTAL's Cartesian frame, which could not be matched to the drawn cell: arrows may be rotated", "warning"); }
      renderTable();
      if (hadFocus) { const row = rowsEl.querySelector(`[data-k="${doc.active}"]`); if (row) row.focus(); }
      { const n = optsHost.querySelector(".fo-amp-note"); if (n) n.textContent = ampNote(); }
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
    else if (e.target.closest("[data-act=dielectric]") && doc && H.openMech) H.openMech(doc.path);
    else if (e.target.closest("[data-act=raman]") && doc && H.openSpectrum) H.openSpectrum(doc.path, "raman");
  });
  /* the stick spectrum: a click picks the nearest IR- or Raman-active mode */
  $(".fm-spec", tableHost).addEventListener("click", (e) => {
    const l = e.target.closest("[data-mode]");
    if (l && doc) selectMode(Number(l.dataset.mode));
  });
  host.addEventListener("keydown", (e) => { if (e.key === " " && e.target.closest(".viewer")) { e.preventDefault(); play(); } });

  /* the view's header: the mode, and how its motion is scaled (always said: ×1 is the physical amplitude) */
  function modeLabel() {
    const m = doc && doc.mode;
    if (!m) return;
    const disp = m.displacement, o = doc.opts, k = disp ? drawScale(doc, disp) : 1;
    const how = !disp ? "" : zeroMode(m) || m.classical === false ? ` · drawn ×${fmtX(k)} (CRYSTAL's vector)`
      : Math.abs(k - 1) < 1e-6 ? " · physical amplitude" : ` · exaggerated ×${fmtX(k)}`;
    $(".fv-mode").textContent = `Mode ${m.mode} · ${fmt(m.freq, 1)} cm⁻¹ (${m.irrep})${m.imaginary ? " · imaginary" : ""}${how}`;
    $(".fv-mode").title = o.scale === "normalized" ? "Scaled so the largest atom moves 0.5 Å × Amplitude (Inspector › Mode)"
      : "×1 is the zero-point classical amplitude CRYSTAL prints, √(ħ/ωμ); Inspector › Mode › Exaggeration scales it";
  }
  /* the line under the slider: what one atom really does and what is drawn */
  function ampNote() {
    const m = doc.mode, disp = m && m.displacement;
    if (!disp) return "";
    const mx = maxOf(disp), k = drawScale(doc, disp), A = (x) => `${x < 0.01 ? x.toFixed(4) : x.toFixed(3)} Å`;
    if (zeroMode(m)) return `≈0 cm⁻¹: a translation or rotation has no zero-point amplitude. Largest atom drawn ${A(mx * k)}.`;
    if (m.classical === false) return `This output does not print its modes as classical amplitudes: the largest atom is drawn ${A(mx * k)}, ×${fmtX(k)} the printed vector.`;
    if (doc.opts.scale === "normalized") return `Largest atom drawn ${A(mx * k)}: ×${fmtX(k)} its zero-point amplitude (${A(mx)}).`;
    return Math.abs(k - 1) < 1e-6 ? `True size: the largest atom moves ${A(mx)} (zero-point amplitude √(ħ/ωμ), as CRYSTAL prints it).`
      : `Exaggerated ×${fmtX(k)}: the largest atom is drawn ${A(mx * k)}; it really moves ${A(mx)}.`;
  }
  /* the mode controls (inspector) */
  let pick = null, pickDoc = null;
  function renderOpts() {
    const o = doc.opts, phys = o.scale !== "normalized";
    optsHost.innerHTML = `<div class="po-row fo-scale-row"><span>Amplitude</span><span class="segment" role="radiogroup" aria-label="Amplitude scale">
        <button type="button" role="radio" data-fo-scale="physical" aria-checked="${phys}" title="×1 is the zero-point classical amplitude CRYSTAL prints">Physical</button>
        <button type="button" role="radio" data-fo-scale="normalized" aria-checked="${!phys}" title="Every mode scaled so its largest atom moves 0.5 Å">Fit 0.5 Å</button></span></div>
      <label class="po-row"><span>${phys ? "Exaggeration" : "Amplitude"}</span><input type="range" id="fo-amp" ${phys ? 'min="0" max="2" step="0.05"' : 'min="0.2" max="4" step="0.1"'} value="${ampToSlider(o)}"
        aria-valuetext="×${fmtX(o.amplitude)}"><span class="po-u">${fmtX(o.amplitude)}×</span></label>
      <p class="muted small po-row fo-amp-note" aria-live="polite">${esc(ampNote())}</p>
      <label class="po-row"><span>Speed</span><input type="range" id="fo-speed" min="0.2" max="3" step="0.1" value="${o.speed}"><span class="po-u">${o.speed.toFixed(1)}/s</span></label>
      <label class="chk po-row"><input type="checkbox" id="fo-arrows"${o.arrows ? " checked" : ""}> Arrows</label>
      ${doc.arrowsCapped ? `<p class="muted small po-row">This structure has too many moving atoms for arrows during the animation: they show when paused</p>` : ""}
      <label class="po-row" data-tier="advanced" title="The arrows' own scale: the largest is 0.5 Å × this, whatever the amplitude"><span>Arrow length</span><input type="range" id="fo-ascale" min="0.5" max="8" step="0.5" value="${o.arrowScale}"><span class="po-u">${o.arrowScale}×</span></label>
      ${doc.structure && doc.structure.pbc.some(Boolean) ? (doc.gamma === false
    ? `<p class="muted small po-row">One cell: this mode is not at Γ, so its periodic images move with phases the output does not give</p>`
    : `<div class="po-row fo-cells-slot"></div>`) : ""}
      <div class="po-acts" data-tier="advanced"><button class="tb-btn-text" data-fo="xyz">${icon("download")}<span>Export Mode as XYZ</span></button></div>`;
    // the cells drawn (N × M × L, js/cells_picker.js): one picker per document, its values follow the range
    const slot = optsHost.querySelector(".fo-cells-slot");
    if (slot) {
      const cells = H.rangeCells ? H.rangeCells() : [1, 1, 1].map(() => (H.rangeN && H.rangeN()) || 1);
      if (pickDoc !== doc || !pick) {
        pick = cellsPicker({ id: "fo-cells", cells, pbc: doc.structure.pbc, label: "Cells",
          onChange: (c) => { if (H.setRange) H.setRange(c); } });
        pickDoc = doc;
      } else pick.setCells(cells);
      if (pick) slot.replaceChildren(pick); else slot.remove();
    }
  }
  optsHost.addEventListener("input", (e) => {
    if (!doc) return;
    const t = e.target, o = doc.opts;
    if (t.id === "fo-amp") o.amplitude = sliderToAmp(o, Number(t.value));
    else if (t.id === "fo-speed") o.speed = Number(t.value);
    else if (t.id === "fo-ascale") o.arrowScale = Number(t.value);
    else if (t.id === "fo-arrows") o.arrows = t.checked;
    else return;
    const u = t.parentElement.querySelector(".po-u");
    if (u) u.textContent = t.id === "fo-speed" ? `${o.speed.toFixed(1)}/s` : t.id === "fo-amp" ? `${fmtX(o.amplitude)}×` : `${o.arrowScale}×`;
    if (t.id === "fo-amp") {
      t.setAttribute("aria-valuetext", `×${fmtX(o.amplitude)}`);
      const n = optsHost.querySelector(".fo-amp-note");
      if (n) n.textContent = ampNote();
      modeLabel();
    }
    clearTimeout(building);
    building = setTimeout(() => draw(true), 60);
  });
  optsHost.addEventListener("click", (e) => {
    const sc = e.target.closest("[data-fo-scale]");
    if (sc && doc) {
      // Physical or the 0.5 Å fit: the slider starts again at ×1 (kept as Settings' choice for the next FREQ)
      if (doc.opts.scale === sc.dataset.foScale) return;
      doc.opts.scale = sc.dataset.foScale;
      doc.opts.amplitude = 1;
      if (H.setSetting) H.setSetting("freq.amplitudeScale", doc.opts.scale);
      renderOpts(); modeLabel(); draw(true);
      const b = optsHost.querySelector(`[data-fo-scale="${doc.opts.scale}"]`);
      if (b) b.focus();
      return;
    }
    if (!doc || !e.target.closest("[data-fo=xyz]") || !doc.mode) return;
    // the frames as drawn (their comment line says how the amplitude is scaled)
    download(new Blob([framesXyz(doc, doc.mode.displacement)], { type: "chemical/x-xyz" }),
      `${doc.title.replace(/\.[^.]+$/, "")}_mode${doc.mode.mode}.xyz`);
  });

  // the element colors changed (Settings, Element Colors): draw the atoms again (draw keeps the animation going)
  document.addEventListener("mace-colors", () => {
    if (doc && doc.structure && !building) draw(true);
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
            nFrames: r.n_frames || 1, gamma: r.gamma !== false, born: !!r.born });
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
      $(".fv-diel").hidden = !d.born;                       // INTCPHF: Born charges and the dielectric results
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
      // 1.1.6: the multiplier on CRYSTAL's zero-point amplitude, and the largest atom's printed and drawn displacement (Å)
      drawScale: doc.mode && doc.mode.displacement ? drawScale(doc, doc.mode.displacement) : null,
      maxDisp: doc.mode && doc.mode.displacement ? maxOf(doc.mode.displacement) : null,
      cells: pick && pickDoc === doc ? pick.cells() : null,
      display: lastDisplay && { atoms: lastDisplay.list.n, cell: lastDisplay.cell, bonds: lastDisplay.bonds, labels: lastDisplay.labels,
        style: lastDisplay.style, box: lastDisplay.box } },
  };
}
