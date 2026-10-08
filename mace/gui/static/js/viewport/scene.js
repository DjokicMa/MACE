/* What the structure viewer draws (spec §2.1, §6, §8.2), in separate 3Dmol
 * models so a change rebuilds only its own layer: the structure (atoms,
 * bonds, cell, labels), the selection shell, the hover halo, measurements,
 * and the ghost / drag preview of an edit. Level of detail follows the
 * number of drawn atoms (core.lodLevel); bonds come from the cell-list
 * grid, on the main thread up to 2000 drawn atoms and in the worker above. */
import * as G from "../geom/core.js";
import * as W from "../geom/client.js";

const SYNC_LIMIT = 2000;

export function createScene(v, deps) {
  const M = deps.M;
  const css = (k) => M.cssVar(k);
  const layer = { structure: null, selection: null, hover: null, drag: null, ghost: null, halo: null };
  let shapes = { cell: [], measure: [], ghost: [] }, labels = { atoms: [], measure: [], hover: [] };
  let bonded = null, compCache = null;
  let disp = null, vdoc = null, lod = 1, version = 0, radiiPx = null, drawnR = null, pairs = null;
  let projCache = null, hovered = -1, bondsForced = false, measureLabelPos = [];
  const wrap = v.getCanvas().closest(".viewer-wrap");
  // the level-of-detail badge and the large-structure banner (spec §8.2.4)
  wrap.insertAdjacentHTML("beforeend", `<div class="vp-badge" hidden></div><div class="vp-banner" hidden role="status"></div>`);
  const badge = wrap.querySelector(".vp-badge"), banner = wrap.querySelector(".vp-banner");

  const vdw = (sym) => {
    const t = (window.$3Dmol && window.$3Dmol.GLModel && window.$3Dmol.GLModel.vdwRadii) || {};
    return t[sym] || 1.5;
  };
  /** The radius 3Dmol draws an atom with, in Å, for the current style. */
  function atomRadius(sym, style, bonds) {
    if (style === "stick") return 0.16;
    if (style === "wireframe") return 0.1 * vdw(sym);
    if (style === "spacefill") return 0.95 * vdw(sym);
    return M.sphereScale(bonds) * vdw(sym);
  }
  function styleObj(style, bonds, level) {
    const cs = M.styleFor("ballstick", true).sphere.colorscheme;
    if (style === "wireframe") return { line: { colorscheme: cs }, sphere: { scale: 0.1, colorscheme: cs } };
    if (style !== "ballstick" || !bonds) return M.styleFor(style, bonds);
    if (level >= 2) return { sphere: { scale: M.sphereScale(true), colorscheme: cs }, line: { colorscheme: cs } };
    return M.styleFor("ballstick", true);
  }

  /* ---------------------------------------------------------------- the structure layer */
  function bondLists(s, d, done) {
    const o = deps.opts();
    const radii = new Float32Array(d.n), metal = new Uint8Array(d.n);
    for (let k = 0; k < d.n; k++) { radii[k] = s.radii[d.base[k]]; metal[k] = s.metal && s.metal[d.base[k]] ? 1 : 0; }
    const tol = Number(deps.setting("bondTolerance")) || s.bond_scale || 1.2, mm = deps.setting("bondMetalMetal") === true;
    if (d.n <= SYNC_LIMIT) return done(G.cellListBonds(d.cart, radii, metal, tol, mm));
    const v0 = version;
    performance.mark("vp:bonds:post");
    W.job("bonds", { cart: d.cart.slice(), radii, metal, tol, metalMetal: mm }).then((r) => {
      if (r.superseded || v0 !== version) return;
      performance.measure("vp:bonds", "vp:bonds:post");
      done(r.pairs, true);
    }).catch((e) => console.error("bonds:", e.message));
    return o;
  }
  /** Pair overrides: the classic panel's per-pair length limits and the Bond tool's toggles. */
  function filterPairs(s, d, p, doc) {
    const max = (doc && doc.bondMax) || {}, over = (doc && doc.bondOverrides) || null;
    const hasMax = Object.keys(max).length;
    let out = p;
    if (hasMax) {
      const keep = [];
      for (let k = 0; k < p.length; k += 2) {
        const a = p[k], b = p[k + 1];
        const key = [s.symbols[d.base[a]], s.symbols[d.base[b]]].sort().join("-");
        if (key in max) {
          const dx = d.cart[3 * a] - d.cart[3 * b], dy = d.cart[3 * a + 1] - d.cart[3 * b + 1], dz = d.cart[3 * a + 2] - d.cart[3 * b + 2];
          if (Math.hypot(dx, dy, dz) > max[key]) continue;
        }
        keep.push(a, b);
      }
      out = Uint32Array.from(keep);
    }
    if (over && over.size) {
      const at = new Map();
      for (let k = 0; k < d.n; k++) at.set(`${d.base[k]}:${d.shift[3 * k]},${d.shift[3 * k + 1]},${d.shift[3 * k + 2]}`, k);
      const set = new Set();
      const key = (a, b) => (a < b ? `${a}-${b}` : `${b}-${a}`);
      for (let k = 0; k < out.length; k += 2) set.add(key(out[k], out[k + 1]));
      for (const [ok, mode] of over) {
        // "i-j:dx,dy,dz": atom j at offset (dx, dy, dz) from atom i's image
        const [ij, img] = ok.split(":"), [i, j] = ij.split("-").map(Number), off = img.split(",").map(Number);
        for (let k = 0; k < d.n; k++) {
          if (d.base[k] !== i) continue;
          const sh = [d.shift[3 * k] + off[0], d.shift[3 * k + 1] + off[1], d.shift[3 * k + 2] + off[2]];
          const other = at.get(`${j}:${sh.join(",")}`);
          if (other === undefined) continue;
          if (mode === "add") set.add(key(k, other)); else set.delete(key(k, other));
        }
      }
      out = Uint32Array.from([...set].flatMap((x) => x.split("-").map(Number)));
    }
    return out;
  }
  function buildStructure(s, d, p, opts) {
    if (layer.structure) v.removeModel(layer.structure);
    const nb = Array.from({ length: d.n }, () => []);
    if (p) for (let k = 0; k < p.length; k += 2) { nb[p[k]].push(p[k + 1]); nb[p[k + 1]].push(p[k]); }
    const atoms = new Array(d.n);
    for (let k = 0; k < d.n; k++) {
      const i = d.base[k];
      atoms[k] = { elem: s.symbols[i], x: d.cart[3 * k], y: d.cart[3 * k + 1], z: d.cart[3 * k + 2], bonds: nb[k],
        bondOrder: nb[k].map(() => 1), index: k, serial: i + 1, atom: `${s.symbols[i]}${i + 1}`,
        properties: { image: !!d.image[k], shift: [d.shift[3 * k], d.shift[3 * k + 1], d.shift[3 * k + 2]] } };
    }
    const m = v.addModel();
    m.addAtoms(atoms);
    m.setStyle({}, styleObj(opts.style, !!p, lod));
    layer.structure = m;
  }
  function drawCellBox(s, opts) {
    shapes.cell.forEach((x) => v.removeShape(x));
    shapes.cell = [];
    if (!opts.cell) return;
    // a molecule (no periodic direction, plan E10): its "cell" is only the atoms' bounding box, never drawn
    if (s.pbc && !s.pbc.some(Boolean)) return;
    // whole cells from round(lo) to round(hi): -0.1..1.1 draws the unit cell, 0..3 three cells
    const corners = G.boxCorners(s, opts.box || { lo: [0, 0, 0], hi: opts.range });
    const color = /^#[0-9a-f]{6}$/i.test(deps.setting("cellColor") || "") ? deps.setting("cellColor") : css("--cell");
    for (const [a, b] of [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]]) {
      shapes.cell.push(v.addCylinder({ start: xyz(corners[a]), end: xyz(corners[b]), radius: 0.03, color, fromCap: 1, toCap: 1 }));
    }
  }
  /* ---------------------------------------------------------------- coordination polyhedra (plan B9) */
  // built in the worker, cached per structure and drawn range; one mesh (and one edge set) per center element
  const POLY_LIMIT = 2000;
  let polyShapes = [], polyCache = { key: null, result: null }, polyState = { on: false, skipped: false, count: 0, meshes: 0, faces: [] };
  function clearPolyhedra() {
    polyShapes.forEach((x) => v.removeShape(x));
    polyShapes = [];
  }
  function drawPolyhedra(s, d) {
    clearPolyhedra();
    const on = deps.setting("showPolyhedra") === true;
    polyState = { on, skipped: false, count: 0, meshes: 0, faces: [] };
    if (!on || !s.radii) return;
    if (d.n > POLY_LIMIT) {
      polyState.skipped = true;
      banner.hidden = false;
      banner.textContent = (lod >= 4 ? "Large structure: simplified drawing. " : "") + `Polyhedra are off above ${POLY_LIMIT.toLocaleString("en-US")} drawn atoms (${d.n.toLocaleString("en-US")} here): narrow the range to see them`;
      return;
    }
    const tol = Number(deps.setting("bondTolerance")) || s.bond_scale || 1.15;
    const key = [d.n, tol, d.base.length, d.cart[0], d.cart[d.cart.length - 1]].join("|");
    const paint = (r) => {
      const t0 = performance.now();
      clearPolyhedra();
      const edgeColor = css("--cell");
      for (const m of r.meshes) {
        const sym = G.SYMBOLS[m.z] || "X";
        const vertexArr = [], normalArr = [], faceArr = [];
        for (let k = 0; k < m.pos.length; k += 3) {
          vertexArr.push({ x: m.pos[k], y: m.pos[k + 1], z: m.pos[k + 2] });
          normalArr.push({ x: m.nrm[k], y: m.nrm[k + 1], z: m.nrm[k + 2] });
          faceArr.push(k / 3);
        }
        polyShapes.push(v.addCustom({ vertexArr, normalArr, faceArr, color: M.elementColor(sym), opacity: 0.35 }));
        // the edges: degenerate triangles (a, b, b) drawn as wireframe lines
        const ev = [], ef = [];
        for (let k = 0; k < m.edge.length; k += 6) {
          const q = ev.length;
          ev.push({ x: m.edge[k], y: m.edge[k + 1], z: m.edge[k + 2] }, { x: m.edge[k + 3], y: m.edge[k + 4], z: m.edge[k + 5] });
          ef.push(q, q + 1, q + 1);
        }
        polyShapes.push(v.addCustom({ vertexArr: ev, normalArr: ev.map(() => ({ x: 0, y: 0, z: 1 })), faceArr: ef, color: edgeColor, wireframe: true, linewidth: 1 }));
      }
      polyState = { on, skipped: false, count: r.count, meshes: r.meshes.length, faces: r.faces, mainMs: performance.now() - t0 };
    };
    if (polyCache.key === key && polyCache.result) { paint(polyCache.result); return; }
    const z = Uint8Array.from(s.symbols.map((x) => G.zOf(x)));
    const frac = Float64Array.from(s.frac.flat()), radii = Float32Array.from(s.radii), metal = Uint8Array.from(s.metal.map(Number));
    const v0 = version;
    performance.mark("vp:poly:post");
    W.job("polyhedra", { cell: s.cell, frac, z, radii, metal, tol, dispCart: d.cart.slice(), dispBase: Uint32Array.from(d.base) }).then((r) => {
      if (r.superseded || v0 !== version) return;
      performance.measure("vp:polyhedra", "vp:poly:post");
      polyCache = { key, result: r };
      paint(r);
      v.render();
      document.dispatchEvent(new CustomEvent("mace-polyhedra", { detail: { count: r.count } }));
    }).catch((e) => console.error("polyhedra:", e.message));
  }
  function labelText(s, i, mode) {
    if (mode === "element") return s.symbols[i];
    if (mode === "index") return String(i + 1);
    if (mode === "wyckoff") return s.symmetry && s.symmetry.available ? s.symmetry.wyckoffs[i] || "" : "";
    return `${s.symbols[i]}${i + 1}`;
  }
  function drawLabels(s, d, opts) {
    labels.atoms.forEach((x) => v.removeLabel(x));
    labels.atoms = [];
    if (!opts.labels || opts.labels === "off" || lod >= 2) return;
    for (let k = 0; k < d.n; k++) {
      if (d.image[k]) continue;
      const t = labelText(s, d.base[k], opts.labels);
      if (!t) continue;
      labels.atoms.push(v.addLabel(t, { position: { x: d.cart[3 * k], y: d.cart[3 * k + 1], z: d.cart[3 * k + 2] },
        fontSize: Number(deps.setting("labelSize")) || 12, fontColor: css("--text"), backgroundColor: css("--surface"),
        backgroundOpacity: 0.75, borderThickness: 0, inFront: true }));
    }
  }
  function showLod(n) {
    const words = { 2: "bonds as lines", 3: bondsForced ? "bonds on (slow)" : "no bonds", 4: "simplified" };
    badge.hidden = lod < 2;
    badge.textContent = lod >= 2 ? `LOD ${lod} · ${words[lod]}` : "";
    badge.title = `${n.toLocaleString("en-US")} atoms drawn`;
    banner.hidden = lod < 4;
    banner.textContent = lod >= 4 ? "Large structure: simplified drawing" : "";
  }

  /** Draw a document's structure. keepView: keep the camera (an edit);
   * otherwise fit it (a newly opened structure). */
  function draw(doc, { keepView = true } = {}) {
    const s = doc.structure;
    if (!s) return;
    vdoc = doc;
    version++;
    const opts = deps.opts();
    const view = keepView && layer.structure ? v.getView() : null;
    v.removeAllModels(); v.removeAllShapes(); v.removeAllLabels();
    Object.keys(layer).forEach((k) => { layer[k] = null; });
    shapes = { cell: [], measure: [], ghost: [] }; labels = { atoms: [], measure: [], hover: [] };
    performance.mark("vp:draw:start");
    disp = G.displayList(s, opts.range, opts.boundary);
    // atoms outside the range bonded to drawn ones (VESTA): one shell, or whole discrete molecules
    bonded = null;
    if (opts.bondedOutside && opts.bondedOutside !== "off" && s.cell && s.frac) {
      const tol = Number(deps.setting("bondTolerance")) || s.bond_scale || 1.15, mm = deps.setting("bondMetalMetal") === true;
      const ck = `${tol}|${mm}`;
      if (!compCache || compCache.s !== s || compCache.k !== ck) compCache = { s, k: ck, c: G.components(s, tol, mm) };
      const r = G.bondedImages(s, disp, opts.bondedOutside, compCache.c);
      bonded = { added: r.added, extended: r.extended, allExtended: r.allExtended, truncated: r.truncated, mode: opts.bondedOutside };
      disp = r;
    }
    lod = G.lodLevel(disp.n, deps.setting("quality") || "auto");
    v.maceLod = lod;           // the Rendering group turns ambient occlusion off at LOD 3 and above
    radiiPx = null; projCache = null; hovered = -1;
    drawnR = new Float32Array(disp.n);
    const bondsWanted = opts.bonds && opts.style !== "spacefill" && (lod <= 2 || bondsForced);
    for (let k = 0; k < disp.n; k++) drawnR[k] = atomRadius(s.symbols[disp.base[k]], opts.style, bondsWanted);
    pairs = null;
    const finish = (p, late) => {
      pairs = p ? filterPairs(s, disp, p, doc) : null;
      buildStructure(s, disp, pairs, opts);
      ghosted = null;
      if (preview && preview.dimStructure) restyle("all");
      if (late) { v.render(); return; }
    };
    if (bondsWanted) {
      let sync = false;
      bondLists(s, disp, (p, late) => { sync = !late; finish(p, late); });
      if (!sync) finish(null);        // atoms now, bonds when the worker answers
    } else finish(null);
    drawCellBox(s, opts);
    drawLabels(s, disp, opts);
    drawSelection(doc);
    drawMeasurements(doc);
    // a preview belongs to its document: another document's draw drops it (its frame puts it back)
    if (preview) { const p = preview; preview = null; previewShapes.length = 0; previewLabels.length = 0; if (!p.doc || p.doc === doc) setPreview(p); }
    showLod(disp.n);
    drawPolyhedra(s, disp);            // after the LOD banner: above 2000 atoms it says why there are none
    deps.setRadius(sceneRadius(s, opts));
    if (view) v.setView(view); else deps.fit(s, disp, opts);
    v.render();
    performance.mark("vp:draw:end");
    performance.measure("vp:draw", "vp:draw:start", "vp:draw:end");
    document.dispatchEvent(new CustomEvent("mace-scene", { detail: { n: disp.n, lod } }));
  }
  function sceneRadius(s, opts) {
    const c = G.boxCorners(s, opts.box || { lo: [0, 0, 0], hi: opts.range });
    return 0.5 * G.norm3(G.sub3(c[7], c[0])) + 2;
  }

  /* ---------------------------------------------------------------- selection and hover */
  function shellModel(indices, color, factor, opacity) {
    const m = v.addModel();
    const s = vdoc.structure, atoms = [], radius = new Map();
    for (const k of indices) {
      const i = disp.base[k];
      atoms.push({ elem: s.symbols[i], x: disp.cart[3 * k], y: disp.cart[3 * k + 1], z: disp.cart[3 * k + 2],
        bonds: [], index: atoms.length, serial: i + 1 });
      radius.set(s.symbols[i], drawnR[k] * factor);
    }
    m.addAtoms(atoms);
    for (const [elem, r] of radius) m.setStyle({ elem }, { sphere: { radius: r, color, opacity } });
    return m;
  }
  /** Rebuild only the selection model (spec §8.2.5). */
  function drawSelection(doc) {
    if (!disp || doc !== vdoc) return;
    performance.mark("vp:sel:start");
    if (layer.selection) { v.removeModel(layer.selection); layer.selection = null; }
    const sel = doc.selection;
    if (sel && sel.size) {
      const idx = [];
      for (let k = 0; k < disp.n; k++) if (sel.has(disp.base[k])) idx.push(k);
      layer.selection = shellModel(idx, css("--sel"), 1.15, 0.35);
    }
    v.render();
    performance.mark("vp:sel:end");
    performance.measure("vp:selection", "vp:sel:start", "vp:sel:end");
  }
  function setHover(k) {
    if (k === hovered || !disp) return;
    hovered = k;
    if (layer.hover) { v.removeModel(layer.hover); layer.hover = null; }
    labels.hover.forEach((x) => v.removeLabel(x));
    labels.hover = [];
    if (k >= 0) {
      layer.hover = shellModel([k], css("--hover-atom"), 1.08, 0.5);
      const opts = deps.opts();
      if (lod === 2 && opts.labels && opts.labels !== "off") {
        const t = labelText(vdoc.structure, disp.base[k], opts.labels);
        if (t) labels.hover.push(v.addLabel(t, { position: { x: disp.cart[3 * k], y: disp.cart[3 * k + 1], z: disp.cart[3 * k + 2] },
          fontSize: Number(deps.setting("labelSize")) || 12, fontColor: css("--text"), backgroundColor: css("--surface"),
          backgroundOpacity: 0.85, borderThickness: 0, inFront: true }));
      }
    }
    v.render();
  }

  /* ---------------------------------------------------------------- selection outline */
  // A translucent shell alone hardly shows on an atom of a similar color (an
  // amber shell on orange Mg), and 3Dmol has no screen-space outline: each
  // selected atom, and the hovered one, also gets a crisp ring on a 2D layer
  // over the viewer, redrawn after every paint, so it follows the camera
  // (spec §2.1's halo). Exports draw the WebGL canvas only, without it.
  wrap.insertAdjacentHTML("beforeend", `<canvas class="vp-rings" aria-hidden="true"></canvas>`);
  const ringCv = wrap.querySelector(".vp-rings");
  const RING_MAX = 20000;
  let rings = [];
  function clearRings() {
    if (!rings.length && !ringCv.width) return;
    const g = ringCv.getContext("2d");
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, ringCv.width, ringCv.height);
    rings = [];
  }
  function drawRings() {
    const sel = vdoc && vdoc.selection, cam = deps.cam();
    if (!disp || !cam || !((sel && sel.size) || hovered >= 0)) { clearRings(); return; }
    let c = cam.calibrate();
    const view = v.getView();
    if (c.view.some((x, k) => Math.abs(x - view[k]) > 1e-9)) { cam.invalidate(); projCache = null; c = cam.calibrate(); }
    const cv = v.getCanvas(), cr = cv.getBoundingClientRect(), wr = wrap.getBoundingClientRect(), dpr = devicePixelRatio || 1;
    ringCv.style.left = cr.left - wr.left + "px"; ringCv.style.top = cr.top - wr.top + "px";
    ringCv.style.width = cr.width + "px"; ringCv.style.height = cr.height + "px";
    const W = Math.round(cr.width * dpr), Hh = Math.round(cr.height * dpr);
    if (ringCv.width !== W || ringCv.height !== Hh) { ringCv.width = W; ringCv.height = Hh; }
    const g = ringCv.getContext("2d");
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, W, Hh);
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const list = [];
    if (sel && sel.size) for (let k = 0; k < disp.n && list.length < RING_MAX; k++) if (sel.has(disp.base[k])) list.push([k, "sel"]);
    if (hovered >= 0) list.push([hovered, "hover"]);
    const color = { sel: css("--sel"), hover: css("--hover-atom") }, halo = M.viewerBg ? M.viewerBg() : css("--viewport-bg");
    rings = [];
    for (const [k, kind] of list) {
      const p = [disp.cart[3 * k], disp.cart[3 * k + 1], disp.cart[3 * k + 2]];
      const [x, y] = G.project(c.P, ...p), [x1, y1] = G.project(c.P, ...G.add3(p, G.scale3(c.right, drawnR[k])));
      const r = Math.hypot(x1 - x, y1 - y) + (kind === "hover" ? 4.5 : 2.5);
      // a thin line of the background around the ring keeps it apart from the atoms behind
      g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI);
      g.lineWidth = 4; g.strokeStyle = halo; g.globalAlpha = 0.7; g.stroke();
      g.lineWidth = 2; g.strokeStyle = color[kind]; g.globalAlpha = 1; g.stroke();
      rings.push({ i: disp.base[k], k, kind, x, y, r });
    }
  }
  document.addEventListener("mace-painted", (e) => { if (e.detail === v.viewerId) drawRings(); });

  /* ---------------------------------------------------------------- measurements */
  const fmtMeasure = (m) => (m.atoms.length === 2 ? `${G.fmtNum(m.value, 3)} Å` : m.atoms.length === 3
    ? `${G.fmtNum(m.value, 2)}°` : `${G.fmtNum(m.value, 1)}°`);
  function posOf(s, a) { return G.cartOf(s.cell, [0, 1, 2].map((k) => s.frac[a.i][k] + (a.image ? a.image[k] : 0))); }
  function drawMeasurements(doc) {
    if (doc !== vdoc) return;
    shapes.measure.forEach((x) => v.removeShape(x));
    labels.measure.forEach((x) => v.removeLabel(x));
    shapes.measure = []; labels.measure = []; measureLabelPos = [];
    const list = (doc.measurements || []).filter((m) => m.visible !== false && !m.invalid);
    if (!list.length || !deps.showMeasurements()) { v.render(); return; }
    const s = doc.structure, color = css("--measure"), placed = [];
    const cam = deps.cam();
    for (const m of list) {
      const p = m.atoms.map((a) => posOf(s, a));
      for (let k = 0; k + 1 < p.length; k++) {
        shapes.measure.push(v.addCylinder({ start: xyz(p[k]), end: xyz(p[k + 1]), radius: 0.03, color, dashed: true,
          dashLength: 0.18, gapLength: 0.12, fromCap: 1, toCap: 1 }));
      }
      let at;
      if (p.length === 3) {
        // a 12-segment arc at the vertex, a third of the shorter arm out
        const u = G.sub3(p[0], p[1]), w = G.sub3(p[2], p[1]), r = 0.35 * Math.min(G.norm3(u), G.norm3(w));
        const e1 = G.unit3(u), n = G.unit3(G.cross3(u, w)), e2 = G.cross3(n, e1);
        const th = (m.value * Math.PI) / 180, arc = [];
        for (let k = 0; k <= 12; k++) {
          const t = (th * k) / 12;
          arc.push(G.add3(p[1], G.add3(G.scale3(e1, r * Math.cos(t)), G.scale3(e2, r * Math.sin(t)))));
        }
        for (let k = 0; k < 12; k++) shapes.measure.push(v.addCylinder({ start: xyz(arc[k]), end: xyz(arc[k + 1]), radius: 0.02, color }));
        at = arc[6];
      } else if (p.length === 4) at = G.scale3(G.add3(p[1], p[2]), 0.5);
      else at = G.scale3(G.add3(p[0], p[1]), 0.5);
      // labels never sit on top of each other: nudged up in screen space
      if (cam) {
        const c = cam.calibrate();
        let [x, y] = G.project(c.P, ...at);
        for (let tries = 0; tries < 8 && placed.some(([px, py]) => Math.abs(px - x) < 64 && Math.abs(py - y) < 18); tries++) {
          at = G.add3(at, G.scale3(c.up, 18 / c.s));
          [x, y] = G.project(c.P, ...at);
        }
        placed.push([x, y]);
      }
      const text = fmtMeasure(m);
      measureLabelPos.push({ text, at });
      labels.measure.push(v.addLabel(text, { position: xyz(at), fontSize: 12, fontColor: css("--text"),
        backgroundColor: css("--surface"), backgroundOpacity: 0.85, borderThickness: 1, borderColor: color, inFront: true }));
    }
    v.render();
  }

  /* ---------------------------------------------------------------- previews */
  /** The atoms of an edit in progress: originals as a 30% ghost, the moved
   * ones opaque at their new place: [[atom index, Cartesian position], ...]
   * (an atom may be drawn more than once: its boundary images). */
  function setDrag(moved) {
    if (layer.drag) { v.removeModel(layer.drag); layer.drag = null; }
    if (!moved || !moved.length || !disp) { restyle(null); v.render(); return; }
    const s = vdoc.structure, atoms = [];
    for (const [i, p] of moved) atoms.push({ elem: s.symbols[i], x: p[0], y: p[1], z: p[2], bonds: [], index: atoms.length, serial: i + 1 });
    const m = v.addModel();
    m.addAtoms(atoms);
    m.setStyle({}, { sphere: { scale: deps.opts().style === "spacefill" ? 0.95 : M.sphereScale(!!pairs), colorscheme: M.styleFor("ballstick", true).sphere.colorscheme } });
    layer.drag = m;
    restyle(new Set(moved.map(([i]) => i)));
    v.render();
  }
  let ghosted = null;
  function restyle(set) {
    // "all": the whole structure as a ghost (a Clean proposal is drawn over it)
    if (!set && preview && preview.dimStructure) set = "all";
    const key = set === "all" ? "all" : set ? [...set].sort().join(",") : "";
    if (key === (ghosted || "") || !layer.structure) return;
    const opts = deps.opts(), st = styleObj(opts.style, !!pairs, lod);
    layer.structure.setStyle({}, st);
    if (set === "all" || (set && set.size)) {
      const ghost = {};
      const op = Number(M.cssVar("--ghost-opacity")) || 0.3;
      for (const [k, val] of Object.entries(st)) ghost[k] = { ...val, opacity: op };
      layer.structure.setStyle(set === "all" ? {} : { serial: [...set].map((i) => i + 1) }, ghost);
    }
    ghosted = key || null;
  }
  /** A dashed ghost cell (the Edit Cell preview), or none. */
  function setGhostCell(cell) {
    shapes.ghost.forEach((x) => v.removeShape(x));
    shapes.ghost = [];
    if (cell) {
      const C = cell, corners = [[0, 0, 0], C[0], C[1], C[2], G.add3(C[0], C[1]), G.add3(C[0], C[2]), G.add3(C[1], C[2]),
        G.add3(G.add3(C[0], C[1]), C[2])];
      for (const [a, b] of [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]]) {
        shapes.ghost.push(v.addCylinder({ start: xyz(corners[a]), end: xyz(corners[b]), radius: 0.025, color: css("--cell"),
          dashed: true, dashLength: 0.25, gapLength: 0.18, fromCap: 1, toCap: 1 }));
      }
    }
    v.render();
  }

  /* ---------------------------------------------------------------- builder previews (Phase 4, plan B10) */
  /** One preview at a time: ghost atoms (a builder's result, Clean's proposal),
   * a dashed outline (the new cell), halos on sites, arrows (displacements)
   * and an a′ b′ c′ triad. Never picked: picking reads the structure layer
   * only. dimStructure: the document's own atoms drawn as ghosts instead
   * (Clean: the proposal solid, the original at 30%). */
  let preview = null;
  const previewShapes = [], previewLabels = [];
  function clearPreviewDrawing() {
    if (layer.ghost) { v.removeModel(layer.ghost); layer.ghost = null; }
    if (layer.halo) { v.removeModel(layer.halo); layer.halo = null; }
    previewShapes.splice(0).forEach((x) => v.removeShape(x));
    previewLabels.splice(0).forEach((x) => v.removeLabel(x));
  }
  function setPreview(p) {
    clearPreviewDrawing();
    const dim = !!(p && p.dimStructure), was = !!(preview && preview.dimStructure);
    preview = p || null;
    if (dim !== was) { ghosted = "__"; restyle(dim ? "all" : null); }
    if (!p || !vdoc) { v.render(); return; }
    const op = p.opacity !== undefined ? p.opacity : (Number(M.cssVar("--ghost-opacity")) || 0.3);
    const cs = M.styleFor("ballstick", true).sphere.colorscheme;
    if (p.atoms && p.atoms.numbers && p.atoms.numbers.length) {
      const a = p.atoms, atoms = [];
      for (let i = 0; i < a.numbers.length; i++) {
        const c = a.cart[i];
        atoms.push({ elem: a.symbols[i], x: c[0], y: c[1], z: c[2], bonds: [], index: i, serial: i + 1 });
      }
      const m = v.addModel();
      m.addAtoms(atoms);
      m.setStyle({}, { sphere: { scale: M.sphereScale(!!pairs) * (p.atomScale || 1), colorscheme: cs, opacity: op } });
      layer.ghost = m;
    }
    if (p.vacancies && p.vacancies.length) {
      // a vacancy: a hollow sphere where the atom was (a wireframe shell)
      for (const c of p.vacancies) previewShapes.push(v.addSphere({ center: xyz(c), radius: 0.55, color: css("--warning"), wireframe: true, opacity: 0.9 }));
    }
    if (p.halo && p.halo.length && disp) {
      const idx = [];
      const want = new Set(p.halo);
      for (let k = 0; k < disp.n; k++) if (want.has(disp.base[k]) && !disp.image[k]) idx.push(k);
      if (idx.length) layer.halo = shellModel(idx, css("--warning"), 1.35, 0.45);
    }
    if (p.cell) {
      const C = p.cell, o = p.origin || [0, 0, 0];
      const corners = [[0, 0, 0], C[0], C[1], C[2], G.add3(C[0], C[1]), G.add3(C[0], C[2]), G.add3(C[1], C[2]),
        G.add3(G.add3(C[0], C[1]), C[2])].map((x) => G.add3(x, o));
      for (const [a, b] of [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]]) {
        previewShapes.push(v.addCylinder({ start: xyz(corners[a]), end: xyz(corners[b]), radius: 0.035, color: css("--preview-cell"),
          dashed: true, dashLength: 0.3, gapLength: 0.2, fromCap: 1, toCap: 1 }));
      }
      if (p.triad) {
        // a′ b′ c′: the new cell's axes from its origin, labeled
        const cols = [css("--axis-a"), css("--axis-b"), css("--axis-c")];
        C.forEach((r, k) => {
          const len = G.norm3(r), end = G.add3(o, G.scale3(r, Math.min(1, 1.6 / (len || 1))));
          previewShapes.push(v.addArrow({ start: xyz(o), end: xyz(end), radius: 0.06, radiusRatio: 2, mid: 0.7, color: cols[k] }));
          previewLabels.push(v.addLabel(`${"abc"[k]}′`, { position: xyz(G.add3(end, G.scale3(r, 0.25 / (len || 1)))), fontColor: cols[k],
            fontSize: 13, bold: true, inFront: true, showBackground: true, backgroundColor: css("--surface"), backgroundOpacity: 0.8, borderThickness: 0 }));
        });
      }
    }
    if (p.arrows && p.arrows.length) {
      for (const [s0, e0] of p.arrows) previewShapes.push(v.addArrow({ start: xyz(s0), end: xyz(e0), radius: 0.05, radiusRatio: 2.2, mid: 0.75, color: css("--sel") }));
    }
    if (p.note) {
      previewLabels.push(v.addLabel(p.note, { position: xyz(p.notePos || [0, 0, 0]), fontSize: 12, fontColor: css("--text"),
        backgroundColor: css("--surface"), backgroundOpacity: 0.9, borderThickness: 1, borderColor: css("--border"), inFront: true }));
    }
    v.render();
  }
  /** What the preview layer holds (tests: __mace.viewport.models()). */
  function previewInfo() {
    return { ghost: layer.ghost ? layer.ghost.selectedAtoms({}).length : 0, halo: layer.halo ? layer.halo.selectedAtoms({}).length : 0,
      shapes: previewShapes.length, labels: previewLabels.length, cell: !!(preview && preview.cell), arrows: preview && preview.arrows ? preview.arrows.length : 0,
      opacity: preview && preview.atoms ? (preview.opacity !== undefined ? preview.opacity : Number(M.cssVar("--ghost-opacity")) || 0.3) : null,
      dim: !!(preview && preview.dimStructure), vacancies: preview && preview.vacancies ? preview.vacancies.length : 0, kind: preview ? preview.kind || "" : "" };
  }

  /* ---------------------------------------------------------------- picking */
  /** Screen positions of the drawn atoms for the current camera (cached). */
  function projected() {
    const cam = deps.cam();
    if (!cam || !disp) return null;
    const c = cam.calibrate();
    if (projCache && projCache.P === c.P) return projCache;
    const uv = G.projectAll(c.P, disp.cart);
    const z = new Float32Array(disp.n);
    for (let k = 0; k < disp.n; k++) z[k] = disp.cart[3 * k] * c.toward[0] + disp.cart[3 * k + 1] * c.toward[1] + disp.cart[3 * k + 2] * c.toward[2];
    projCache = { P: c.P, uv, z, s: c.s };
    return projCache;
  }
  /** The drawn atom under (x, y) in CSS px from the canvas's corner: the
   * front-most whose disc holds the point; -1 for none. */
  function pickAt(x, y, slack = 3) {
    const pr = projected();
    if (!pr) return -1;
    let best = -1, bestZ = -Infinity;
    for (let k = 0; k < disp.n; k++) {
      const du = pr.uv[2 * k] - x, dv = pr.uv[2 * k + 1] - y, r = drawnR[k] * pr.s + slack;
      if (du * du + dv * dv <= r * r && pr.z[k] > bestZ) { bestZ = pr.z[k]; best = k; }
    }
    return best;
  }

  return {
    draw, drawSelection, drawMeasurements, setHover, setDrag, setGhostCell, pickAt, projected, setPreview, previewInfo,
    get preview() { return preview; },
    redrawStructure() { if (vdoc) draw(vdoc, { keepView: true }); },
    /** The document a view-only stand-in was drawn for arrived: take it over (no redraw). */
    adopt(doc) {
      if (vdoc && vdoc.viewOnly && vdoc.structure === doc.structure) { vdoc = doc; drawSelection(doc); drawMeasurements(doc); }
    },
    get display() { return disp; }, get lod() { return lod; }, get doc() { return vdoc; },
    get layers() { return layer; }, get hovered() { return hovered; }, get pairs() { return pairs; },
    get drawnRadii() { return drawnR; },
    /** What "Bonded atoms outside" added at the last draw: {added, extended, allExtended, truncated, mode} or null. */
    get bonded() { return bonded; },
    measureLabels: () => measureLabelPos.map((m) => m.text),
    /** The selection and hover rings drawn after the last paint (tests). */
    rings: () => rings.map((r) => ({ ...r })),
    setBondsForced(on) { bondsForced = !!on; },
    badgeText: () => (badge.hidden ? "" : badge.textContent),
    bannerText: () => (banner.hidden ? "" : banner.textContent),
    polyhedra: () => ({ ...polyState, shapes: polyShapes.length }),
    invalidate() { projCache = null; },
    fmtMeasure,
    /** Export Image: leave out the cell and/or the measurements (and the
     * hover halo) for one picture; returns what puts them back. */
    forExport({ cell = true, measurements = true } = {}) {
      const doc = vdoc, opts = deps.opts();
      if (hovered >= 0) setHover(-1);
      if (!doc || !doc.structure) return () => {};
      const noCell = !cell && opts.cell && shapes.cell.length, noMeasure = !measurements && (labels.measure.length || shapes.measure.length);
      if (noCell) drawCellBox(doc.structure, { ...opts, cell: false });
      if (noMeasure) {
        shapes.measure.forEach((x) => v.removeShape(x));
        labels.measure.forEach((x) => v.removeLabel(x));
        shapes.measure = []; labels.measure = [];
      }
      return () => {
        if (noCell) drawCellBox(doc.structure, opts);
        if (noMeasure) drawMeasurements(doc);
      };
    },
  };
}
const xyz = (p) => ({ x: p[0], y: p[1], z: p[2] });
