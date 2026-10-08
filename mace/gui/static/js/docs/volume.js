/* The Volume document (plan E5): a Gaussian cube (CRYSTAL's ECH3 density,
 * POT3 potential, spin density, a difference of grids, or a LOCALWF/PRINTPLO
 * Wannier orbital) drawn as isosurfaces by 3Dmol in the page from a grid the
 * server decimated to the quality budget (statistics always come from the
 * full grid), with a lattice-plane slice map below, the surface colored by a
 * potential (MEP) with the CLI's range rule and colors, and differences of
 * grids computed on the server with the CLI's own arithmetic. */
const BAND = 2;    // a slice plane is drawn this many voxels thick in the 3D view

export function createVolumeView(host, optsHost, H) {
  const { api, q, esc, icon, toast, apiTask, M } = H;
  host.innerHTML = `<div class="vv">
    <div class="vv-head"><div class="vv-title"></div><span class="vv-chip mono"></span>
      <button class="tb-btn-text" data-act="mep" data-tip="Color the surface by an electrostatic potential (MEP)">${icon("sun")}<span>Color by Potential…</span></button>
      <button class="tb-btn-text" data-act="diff" data-tip="A − B (− C …): a difference of grids, aligned as mace plotting does">${icon("minus")}<span>Difference…</span></button>
      <button class="tb-btn-text" data-act="save" hidden data-tip="Write this difference as a new .CUBE file">${icon("save")}<span>Save Result As…</span></button></div>
    <div class="vv-view"><div class="viewer" id="viewer-volume" tabindex="0" aria-label="Volume"></div>
      <div class="vv-msg" hidden></div><div class="vv-bar" hidden></div></div>
    <div class="vv-slice" hidden><div class="vv-slice-head muted small"></div><div class="vv-slice-plot"></div></div></div>`;
  const $ = (s, el = host) => el.querySelector(s);
  let doc = null, viewer = null, shapes = [], building = 0;
  const stats = { surfaces: 0, vertices: 0 };

  function v() { if (!viewer) viewer = M().viewer("viewer-volume"); return viewer; }
  const signedOf = (d) => !!(d.meta && d.meta.signed);
  const defaults = (m) => ({ level: m.level, plus: true, minus: m.signed, opacity: m.type === "density" ? 0.75 : 0.85,
    wire: false, atoms: true, cell: true, slice: false, axis: "c", index: null, contours: 12,
    colorPath: null, range: "auto", vmin: null, vmax: null, reverse: (H.setting("volume.mepColors") || "cli") === "chemistry" });

  /* ---------------------------------------------------------------- data */
  async function loadVolume(d) {
    const quality = H.setting("volume.quality") || "medium";
    const meta = d.combo ? d.meta : await apiTask("volume?" + q({ path: d.path, quality }));
    const r = await api("volume/grid?" + q({ key: meta.grid_key }), { blob: true });
    const buf = await r.blob.arrayBuffer();
    d.meta = meta;
    d.vd = volumeData(meta, new Float32Array(buf));
  }
  /** A 3Dmol VolumeData from the decimated grid: Cartesian voxel vectors (Å), so a skewed cell is drawn true. */
  function volumeData(m, data) {
    const vd = new window.$3Dmol.VolumeData("", "cube");
    const [nx, ny, nz] = m.shape, V = m.voxels, o = m.origin;
    vd.size = { x: nx, y: ny, z: nz };
    vd.origin = { x: o[0], y: o[1], z: o[2] };
    vd.unit = { x: Math.hypot(...V[0]), y: Math.hypot(...V[1]), z: Math.hypot(...V[2]) };
    vd.matrix = new window.$3Dmol.Matrix4(V[0][0], V[1][0], V[2][0], o[0], V[0][1], V[1][1], V[2][1], o[1],
      V[0][2], V[1][2], V[2][2], o[2], 0, 0, 0, 1);
    vd.inversematrix = null;
    vd.data = data;
    return vd;
  }

  /* ---------------------------------------------------------------- drawing */
  function colors(d) {
    const css = M().cssVar;
    if (d.meta.type === "spin" || d.meta.type === "spin_difference") return [css("--plot-10"), css("--plot-1")];
    return [css("--vol-plus"), css("--vol-minus")];
  }
  function gradient(d) {
    const lo = d.opts.vmin, hi = d.opts.vmax;
    // 3Dmol's RWB maps its first bound to red: the CLI (RdBu_r) puts red on the positive side
    return d.opts.reverse ? new window.$3Dmol.Gradient.RWB(lo, hi) : new window.$3Dmol.Gradient.RWB(hi, lo);
  }
  function draw(fit = false) {
    const d = doc;
    if (!d || !d.vd) return;
    const vw = v(), view = !fit && vw.getModel() ? vw.getView() : null;
    vw.removeAllModels(); vw.removeAllShapes(); vw.removeAllLabels();
    shapes = [];
    const o = d.opts, m = d.meta;
    if (o.atoms && m.atoms.numbers.length) {
      const xyz = [String(m.atoms.numbers.length), "atoms"].concat(m.atoms.symbols.map((el, i) =>
        `${el} ${m.atoms.cart[i].map((x) => x.toFixed(5)).join(" ")}`)).join("\n") + "\n";
      const model = vw.addModel(xyz, "xyz");
      const cs = M().state.colors ? { prop: "elem", map: M().state.colors.schemes[M().schemeName()] } : "Jmol";
      model.setStyle({}, { sphere: { scale: 0.25, colorscheme: cs }, stick: { radius: 0.1, colorscheme: cs } });
    }
    const [cp, cm] = colors(d);
    const spec = (iso, color) => {
      const s = { isoval: iso, color, opacity: o.opacity, wireframe: !!o.wire, smoothness: 1 };
      if (d.colorVd && o.vmin !== null && o.vmax !== null) { s.voldata = d.colorVd; s.volscheme = gradient(d); delete s.color; }
      return s;
    };
    // a lobe with no surface at this level is not kept: 3Dmol gives an empty isosurface a bounding sphere
    // around (0, 0, 0), which the first fit would include (a molecule far from the origin shrank to a dot)
    const add = (iso, color) => { const s = vw.addIsosurface(d.vd, spec(iso, color)); if (vertexCount(s) > 0) shapes.push(s); else vw.removeShape(s); };
    if (o.plus) add(Math.abs(o.level), cp);
    if (o.minus && signedOf(d)) add(-Math.abs(o.level), cm);
    stats.surfaces = shapes.length;
    stats.vertices = shapes.reduce((n, s) => n + vertexCount(s), 0);
    window.__maceVolumeStats = { ...stats };
    // the first view fits the atoms and the surfaces; the cell (a slab's c spans its vacuum) and the plane come after
    if (view) vw.setView(view); else { vw.zoomTo(); vw.zoom(0.9); }
    if (o.cell && m.periodic && m.lattice) {
      M().drawCell(vw, { cell: m.lattice, pbc: m.pbc || [true, true, true], origin: m.cell_origin || m.origin }, M().cssVar("--cell"), 0.02, 1);
    }
    if (o.slice && d.slice) drawPlane(vw, d);
    vw.render();
    renderBar(d);
  }
  function vertexCount(s) {
    try {
      const g = s.geo || (s.globj && s.globj.geometry);
      if (!g || !g.geometryGroups) return 0;
      return g.geometryGroups.reduce((n, gg) => n + (gg.vertices || 0), 0);
    } catch (e) { return 0; }
  }
  /** The surface's vertices (Å), for the MEP range: at most 50 000 of them, evenly picked. */
  function surfaceVertices() {
    const out = [];
    for (const s of shapes) {
      const g = s.geo;
      if (!g || !g.geometryGroups) continue;
      for (const gg of g.geometryGroups) for (let i = 0; i < gg.vertices; i++) out.push(gg.vertexArray[3 * i], gg.vertexArray[3 * i + 1], gg.vertexArray[3 * i + 2]);
    }
    const n = out.length / 3, step = Math.max(1, Math.ceil(n / 50000)), pick = [];
    for (let i = 0; i < n; i += step) pick.push(out[3 * i], out[3 * i + 1], out[3 * i + 2]);
    return pick;
  }
  function drawPlane(vw, d) {
    const s = d.slice, m = d.meta, ax = "abc".indexOf(s.axis), others = [0, 1, 2].filter((k) => k !== ax);
    // the plane in the full grid's frame: origin + index * (voxel along the axis)
    const full = m.full_voxels, o = m.cell_origin || m.origin;
    const p0 = [0, 1, 2].map((k) => o[k] + s.index * full[ax][k]);
    const u = [0, 1, 2].map((k) => full[others[0]][k] * m.full_shape[others[0]]);
    const w = [0, 1, 2].map((k) => full[others[1]][k] * m.full_shape[others[1]]);
    const P = [p0, p0.map((x, k) => x + u[k]), p0.map((x, k) => x + u[k] + w[k]), p0.map((x, k) => x + w[k])];
    vw.addCustom({ vertexArr: P.map((p) => ({ x: p[0], y: p[1], z: p[2] })), normalArr: P.map(() => ({ x: 0, y: 0, z: 1 })),
      faceArr: [0, 1, 2, 0, 2, 3, 0, 2, 1, 0, 3, 2], color: M().cssVar("--accent"), opacity: 0.35 });
  }
  function renderBar(d) {
    const bar = $(".vv-bar");
    if (!d.colorVd || d.opts.vmin === null) { bar.hidden = true; return; }
    bar.hidden = false;
    const css = M().cssVar, red = css("--mep-pos"), white = css("--mep-mid"), blue = css("--mep-neg");
    const [left, right] = d.opts.reverse ? [red, blue] : [blue, red];
    bar.innerHTML = `<div class="vv-grad" style="background: linear-gradient(90deg, ${left}, ${white}, ${right})"></div>
      <div class="vv-ticks mono"><span>${d.opts.vmin.toExponential(2)}</span><span>0</span><span>${d.opts.vmax.toExponential(2)}</span></div>
      <div class="vv-legend small">Potential (a.u.) · ${d.opts.reverse ? "red = negative (electron-rich) · blue = positive (electron-poor)"
        : "red = positive (electron-poor) · blue = negative (electron-rich)"}</div>`;
  }

  /* ---------------------------------------------------------------- slice */
  async function loadSlice(d) {
    const m = d.meta, ax = "abc".indexOf(d.opts.axis);
    if (d.opts.index === null || d.opts.index >= m.full_shape[ax]) d.opts.index = Math.floor(m.full_shape[ax] / 2);
    // a difference made here: the plane of the difference itself, computed from its operands
    const where = d.combo ? d.meta.combined.paths.map((p, k) => `path=${encodeURIComponent(p)}&coef=${encodeURIComponent(d.meta.combined.coefficients[k])}`).join("&") + "&"
      : "";
    const r = await apiTask("volume/slice?" + where + q({ ...(d.combo ? {} : { path: d.path }), axis: d.opts.axis, index: d.opts.index }));
    if (doc !== d) return;
    d.slice = r;
    await plotSlice(d);
    draw();
  }
  /** The slice map, in the theme's colors (drawn again when the theme changes). */
  async function plotSlice(d) {
    const r = d.slice;
    await M().loadPlotly();
    if (doc !== d) return;
    const el = $(".vv-slice-plot");
    $(".vv-slice").hidden = false;
    $(".vv-slice-head").textContent = `Plane ${d.opts.axis} = ${r.index} of ${r.n} (${(r.frac).toFixed(3)} fractional, ${r.position_A.toFixed(3)} Å from the grid's origin)`;
    const signed = signedOf(d);
    const zmax = signed ? Math.max(...r.cart.flat().filter((x) => x !== null).map(Math.abs)) : null;
    const trace = { type: "contour", x: r.x, y: r.y, z: transpose(r.cart), colorscale: signed ? "RdBu" : "Viridis", reversescale: signed,
      ncontours: d.opts.contours, contours: { coloring: "heatmap", showlines: true }, line: { width: 0.5 },
      zmid: signed ? 0 : undefined, zmin: signed ? -zmax : undefined, zmax: signed ? zmax : undefined,
      colorbar: { thickness: 10, len: 0.9 }, hovertemplate: "%{x:.2f}, %{y:.2f} Å: %{z:.4g}<extra></extra>" };
    const css = M().cssVar;
    await window.Plotly.react(el, [trace], { margin: { l: 44, r: 10, t: 6, b: 36 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: css("--text"), size: 11 }, xaxis: { title: { text: "Å" }, scaleanchor: "y", constrain: "domain", gridcolor: css("--line"), zerolinecolor: css("--line") },
      yaxis: { title: { text: "Å" }, gridcolor: css("--line"), zerolinecolor: css("--line") } },
    { displaylogo: false, responsive: true });
  }
  // a theme change: the surfaces' and the slice map's colors come from the theme's tokens
  document.addEventListener("mace-theme", () => {
    if (!doc || !doc.vd) return;
    draw();
    if (doc.slice && doc.opts.slice) plotSlice(doc).catch(() => {});
  });
  const transpose = (a) => a[0].map((_, j) => a.map((row) => row[j]));

  /* ---------------------------------------------------------------- the inspector */
  function logSlider(d) {
    // density: a log scale between its 1st and 99.9th percentile-ish range; signed: linear in |value|
    const s = d.meta.stats, top = Math.max(s.absmax, 1e-12);
    if (!signedOf(d)) {
      const lo = Math.log10(Math.max(1e-6, Math.min(d.meta.level / 10, top / 1e4))), hi = Math.log10(top);
      return { min: lo, max: hi, step: (hi - lo) / 400, value: Math.log10(Math.max(d.opts.level, 1e-12)), log: true };
    }
    const hi = Math.max(top * 0.8, Math.abs(d.opts.level) * 1.2);
    return { min: hi / 400, max: hi, step: hi / 400, value: Math.abs(d.opts.level), log: false };
  }
  function renderOpts() {
    const d = doc;
    if (!d || !d.meta) { optsHost.innerHTML = ""; return; }
    const o = d.opts, m = d.meta, sl = logSlider(d), st = m.stats;
    const lobe = m.type === "spin" ? ["Spin up", "Spin down"] : m.type === "orbital" ? ["+ phase", "− phase"] : ["Show + lobe", "Show − lobe"];
    const integral = st.integral !== undefined ? (st.integral === null
      ? `<p class="vo-stat" title="${esc(st.integral_note)}">Integrated density: not shown (${esc(st.integral_note)})</p>`
      : `<p class="vo-stat" title="${esc(st.integral_note)}">Integrated density on this grid: <b>${st.integral.toFixed(1)} e</b></p>`) : "";
    optsHost.innerHTML = `<div class="po-h">Surface</div>
      <label class="po-row"><span>Level</span><input type="range" id="vo-level" min="${sl.min}" max="${sl.max}" step="${sl.step}" value="${sl.value}" data-log="${sl.log}"><span class="po-u mono" id="vo-level-v">${fmtv(o.level)}</span></label>
      <label class="chk po-row"><input type="checkbox" id="vo-plus"${o.plus ? " checked" : ""}> ${lobe[0]}</label>
      ${signedOf(d) ? `<label class="chk po-row"><input type="checkbox" id="vo-minus"${o.minus ? " checked" : ""}> ${lobe[1]}</label>` : ""}
      <label class="po-row"><span>Opacity</span><input type="range" id="vo-opacity" min="0.1" max="1" step="0.05" value="${o.opacity}"><span class="po-u">${o.opacity.toFixed(2)}</span></label>
      <label class="chk po-row"><input type="checkbox" id="vo-wire"${o.wire ? " checked" : ""}> Wireframe</label>
      <label class="chk po-row"><input type="checkbox" id="vo-atoms"${o.atoms ? " checked" : ""}> Atoms</label>
      ${m.periodic ? `<label class="chk po-row"><input type="checkbox" id="vo-cell"${o.cell ? " checked" : ""}> Cell</label>` : ""}
      <div class="po-h">Slice</div>
      <label class="chk po-row"><input type="checkbox" id="vo-slice"${o.slice ? " checked" : ""}> Show a lattice plane</label>
      <label class="po-row"><span>Plane</span><select id="vo-axis">${["a", "b", "c"].map((a) => `<option value="${a}"${o.axis === a ? " selected" : ""}>${a}</option>`).join("")}</select></label>
      <label class="po-row"><span>Position</span><input type="range" id="vo-index" min="0" max="${m.full_shape["abc".indexOf(o.axis)] - 1}" step="1" value="${o.index ?? Math.floor(m.full_shape["abc".indexOf(o.axis)] / 2)}"><span class="po-u mono" id="vo-index-v">${o.index ?? "mid"}</span></label>
      <label class="po-row"><span>Contours</span><input type="number" id="vo-contours" min="2" max="60" step="1" value="${o.contours}"></label>
      ${d.colorVd ? `<div class="po-h">Color</div>
        <div class="po-row"><span>Range</span><span class="segment"><button data-vo-range="auto" aria-pressed="${o.range === "auto"}">Auto</button><button data-vo-range="custom" aria-pressed="${o.range === "custom"}">Custom</button></span></div>
        <label class="po-row"><span>Min</span><input type="number" id="vo-vmin" step="any" value="${o.vmin}"${o.range === "auto" ? " disabled" : ""}></label>
        <label class="po-row"><span>Max</span><input type="number" id="vo-vmax" step="any" value="${o.vmax}"${o.range === "auto" ? " disabled" : ""}></label>
        <label class="chk po-row"><input type="checkbox" id="vo-reverse"${o.reverse ? " checked" : ""}> Reverse Colors (red = negative)</label>
        <div class="po-acts"><button class="tb-btn-text" data-vo="resetRange">${icon("rotate-ccw")}<span>Reset Range</span></button></div>` : ""}
      <div class="po-h">Statistics (full grid)</div>
      <p class="vo-stat mono">min ${fmtv(st.min)} · max ${fmtv(st.max)}</p>${integral}
      ${d.combo ? "" : `<div class="po-acts"><button class="tb-btn-text" data-vo="cli">${icon("external-link")}<span>Render with mace plotting</span></button></div>`}`;
  }
  const fmtv = (x) => (x === null || x === undefined ? "—" : Math.abs(x) >= 1e-2 && Math.abs(x) < 1e4 ? Number(x).toPrecision(4) : Number(x).toExponential(2));
  optsHost.addEventListener("input", (e) => {
    const d = doc, t = e.target;
    if (!d) return;
    const o = d.opts;
    if (t.id === "vo-level") {
      o.level = t.dataset.log === "true" ? 10 ** Number(t.value) : Number(t.value); $("#vo-level-v", optsHost).textContent = fmtv(o.level);
      // an automatic MEP range follows the surface
      if (d.colorVd && o.range === "auto") { clearTimeout(d.rangeTimer); d.rangeTimer = setTimeout(() => autoRange(d).catch((er) => toast(er.message, "error")), 500); }
    }
    else if (t.id === "vo-opacity") o.opacity = Number(t.value);
    else if (t.id === "vo-index") { o.index = Number(t.value); $("#vo-index-v", optsHost).textContent = String(o.index); clearTimeout(building); building = setTimeout(() => { if (o.slice) loadSlice(d).catch((er) => toast(er.message, "error")); }, 150); return; }
    else return;
    clearTimeout(building);
    building = setTimeout(() => draw(), 80);
  });
  optsHost.addEventListener("change", (e) => {
    const d = doc, t = e.target;
    if (!d) return;
    const o = d.opts;
    if (t.id === "vo-plus") o.plus = t.checked;
    else if (t.id === "vo-minus") o.minus = t.checked;
    else if (t.id === "vo-wire") o.wire = t.checked;
    else if (t.id === "vo-atoms") o.atoms = t.checked;
    else if (t.id === "vo-cell") o.cell = t.checked;
    else if (t.id === "vo-reverse") o.reverse = t.checked;
    else if (t.id === "vo-vmin" || t.id === "vo-vmax") { const x = Number(t.value); if (Number.isFinite(x)) o[t.id === "vo-vmin" ? "vmin" : "vmax"] = x; }
    else if (t.id === "vo-contours") { o.contours = Math.max(2, Math.min(60, Number(t.value) || 12)); if (o.slice) loadSlice(d).catch((er) => toast(er.message, "error")); return; }
    else if (t.id === "vo-slice" || t.id === "vo-axis") {
      if (t.id === "vo-slice") o.slice = t.checked; else { o.axis = t.value; o.index = null; }
      renderOpts();
      if (o.slice) loadSlice(d).catch((er) => toast(er.message, "error")); else { $(".vv-slice").hidden = true; d.slice = null; draw(); }
      return;
    } else return;
    draw();
  });
  optsHost.addEventListener("click", (e) => {
    const d = doc;
    if (!d) return;
    const r = e.target.closest("[data-vo-range]");
    if (r) { d.opts.range = r.dataset.voRange; if (d.opts.range === "auto") autoRange(d); renderOpts(); return; }
    const b = e.target.closest("[data-vo]");
    if (b && b.dataset.vo === "resetRange") { d.opts.range = "auto"; autoRange(d); renderOpts(); }
    if (b && b.dataset.vo === "cli" && !d.combo) H.renderWithCli && H.renderWithCli(d.path);
  });
  host.addEventListener("click", (e) => {
    const b = e.target.closest("[data-act]");
    if (!b || !doc) return;
    if (b.dataset.act === "mep") H.colorByPotential(doc);
    else if (b.dataset.act === "diff") H.difference(doc);
    else if (b.dataset.act === "save") H.saveResult(doc);
  });
  // 3Dmol keeps the pointer's mousedown: focus the view by hand, so its keys (the slice steps) reach it
  $("#viewer-volume").addEventListener("pointerdown", () => $("#viewer-volume").focus({ preventScroll: true }), true);
  host.addEventListener("keydown", (e) => {
    // arrow keys step the slice, as the CLI's slice browser does
    if (!doc || !doc.opts.slice || !e.target.closest(".viewer") || !["ArrowUp", "ArrowDown"].includes(e.key)) return;
    e.preventDefault();
    const n = doc.meta.full_shape["abc".indexOf(doc.opts.axis)];
    doc.opts.index = Math.max(0, Math.min(n - 1, (doc.opts.index ?? 0) + (e.key === "ArrowUp" ? 1 : -1)));
    renderOpts();
    loadSlice(doc).catch((er) => toast(er.message, "error"));
  });

  /* ---------------------------------------------------------------- MEP */
  async function autoRange(d) {
    draw();
    // the density's file and level: the server runs the CLI's own surface and rule; a difference made here has
    // no file, so this page's surface vertices are sampled instead
    let body;
    if (d.path) body = { path: d.colorPath, shape: d.path, level: Math.abs(d.opts.level) };
    else {
      const verts = surfaceVertices();
      if (!verts.length) { toast("No surface at this level to color", "warning"); return; }
      body = { path: d.colorPath, vertices: verts };
    }
    const r = await apiTask("volume/mep_range", body);
    if (doc !== d) return;
    d.opts.vmin = r.vmin; d.opts.vmax = r.vmax; d.mepAuto = r;
    renderOpts(); draw();
  }
  async function setColor(d, path) {
    const quality = H.setting("volume.quality") || "medium";
    const meta = await apiTask("volume?" + q({ path, quality }));
    const r = await api("volume/grid?" + q({ key: meta.grid_key }), { blob: true });
    d.colorVd = volumeData(meta, new Float32Array(await r.blob.arrayBuffer()));
    d.colorPath = path;
    d.colorMeta = meta;
    d.opts.range = "auto";
    d.opts.vmin = d.opts.vmax = null;
    await autoRange(d);
  }

  return {
    async open(d) {
      doc = d;
      $(".vv-title").textContent = d.title;
      $("[data-act=save]").hidden = !d.combo;
      const msg = $(".vv-msg");
      if (!d.vd) {
        msg.hidden = false; msg.innerHTML = `<span class="spinner"></span> Reading the grid…`;
        try {
          await M().load3Dmol();
          await loadVolume(d);
        } catch (e) { if (doc === d) msg.textContent = e.message; throw e; }
        if (!d.opts) d.opts = defaults(d.meta);
      }
      if (doc !== d) return;
      msg.hidden = true;
      const m = d.meta;
      $(".vv-chip").textContent = `${m.label} · ${m.full_shape.join("×")} → ${m.shape.join("×")} shown (${m.quality[0].toUpperCase()}${m.quality.slice(1)})`;
      $(".vv-chip").title = $(".vv-chip").textContent;
      renderOpts();
      draw(true);
      if (d.opts.slice && !d.slice) await loadSlice(d);
      else if (d.opts.slice) await plotSlice(d);        // the slice map is shared by the volume tabs: this tab's plane
      else $(".vv-slice").hidden = true;
    },
    close() { doc = null; },
    setColor: (path) => (doc ? setColor(doc, path) : null),
    autoRange: () => (doc ? autoRange(doc) : null),
    redraw: () => draw(),
    state: () => doc && doc.meta && { path: doc.path, type: doc.meta.type, level: doc.opts.level, plus: doc.opts.plus, minus: doc.opts.minus,
      surfaces: stats.surfaces, vertices: stats.vertices, shape: doc.meta.shape, stride: doc.meta.stride, slice: doc.slice ? { axis: doc.slice.axis, index: doc.slice.index } : null,
      color: doc.colorPath || null, vmin: doc.opts.vmin, vmax: doc.opts.vmax, reverse: doc.opts.reverse, stats: doc.meta.stats, combo: !!doc.combo },
    vertices: () => surfaceVertices(),
    gradientHex: (x) => (doc && doc.colorVd ? gradient(doc).valueToHex(x) : null),
    viewer: () => viewer,
  };
}
