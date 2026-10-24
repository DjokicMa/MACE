/* The Volume document (plan E5): a Gaussian cube (CRYSTAL's ECH3 density,
 * POT3 potential, spin density, a difference of grids, or a LOCALWF/PRINTPLO
 * Wannier orbital) drawn as isosurfaces by 3Dmol in the page from a grid the
 * server decimated to the quality budget (statistics always come from the
 * full grid), with a lattice-plane slice map below, the surface colored by a
 * potential (MEP) with the CLI's range rule and colors, and differences of
 * grids computed on the server with the CLI's own arithmetic. A CRYSTAL23
 * ORBITALS file (.molden) opens here too, with the orbital picker
 * (js/docs/orbitals.js) beside the view: its grids come from GET
 * orbitals/grid in the same shape as a cube's. */
import { legible } from "../viewport/contrast.js";
import { keepBuilt } from "../viewport/shape_cache.js";
import * as MQ from "../viewport/mesh_quality.js";
import * as PS from "../viewport/periodic_surface.js";
import * as L from "../labels.js";
import * as MR from "../maprange.js";

const BAND = 2;    // a slice plane is drawn this many voxels thick in the 3D view

export function createVolumeView(host, optsHost, H) {
  const { api, q, esc, icon, toast, apiTask, M } = H;
  host.innerHTML = `<div class="vv">
    <div class="vv-head"><div class="vv-title"></div><span class="vv-chip mono"></span>
      <button class="tb-btn-text" data-act="mep" data-tip="Color the surface by an electrostatic potential (MEP)">${icon("sun")}<span>Color by Potential…</span></button>
      <button class="tb-btn-text" data-act="diff" data-tip="A − B (− C …): a difference of grids, aligned as mace plotting does">${icon("minus")}<span>Difference…</span></button>
      <button class="tb-btn-text" data-act="save" hidden data-tip="Write this difference as a new .CUBE file">${icon("save")}<span>Save Result As…</span></button>
      <button class="tb-btn-text" data-act="savecube" hidden data-tip="Write the orbital shown as a cube in the job's _orbitals folder">${icon("save")}<span>Save Cube</span></button></div>
    <div class="vv-main"><aside class="vv-orb" hidden aria-label="Orbital picker"></aside>
    <div class="vv-view"><div class="viewer" id="viewer-volume" tabindex="0" aria-label="Volume"></div>
      <div class="vv-msg" hidden></div><div class="vv-bar" hidden></div></div></div>
    <div class="vv-slice" hidden><div class="vv-slice-head muted small"></div><div class="vv-slice-plot"></div></div></div>`;
  const $ = (s, el = host) => el.querySelector(s);
  let doc = null, viewer = null, shapes = [], building = 0, picker = null;
  let G = null;            // geom/core.js, loaded with the first document
  let SW = null;           // js/docs/slice_widget.js: the plane geometry, the mini cell preview, the quick map
  let planeShapes = [], preview = null, mapFrame = 0;
  let drawnOnce = false;   // the camera is the user's once a document has been drawn
  let lastDisplay = null;   // what the last draw showed (state(), for the tests)
  const stats = { surfaces: 0, vertices: 0 };

  function v() {
    if (!viewer) {
      viewer = M().viewer("viewer-volume");
      if (viewer && H.attachTriad) H.attachTriad(viewer, $(".vv-view"));      // the axes, as in the structure view
    }
    return viewer;
  }
  const signedOf = (d) => !!(d.meta && d.meta.signed);
  const defaults = (m) => ({ level: m.level, plus: true, minus: m.signed, opacity: m.type === "density" ? 0.75 : 0.85,
    wire: false, atoms: true, slice: false, axis: "c", index: null, contours: 12,
    // a Miller plane (hkl) or the plane normal to [uvw]: its indices, position across the box (0 to 1), turn
    hkl: [1, 1, 1], uvw: [0, 0, 1], pos: 0.5, tilt: 0, twist: 0,
    colorPath: null, range: "auto", vmin: null, vmax: null, chem: null });
  const planeMode = (o) => o.axis === "hkl" || o.axis === "uvw";
  // the 3D view's own tokens (a White or Black background of the other lightness gets its token set)
  const vp = (k) => (M().vpVar ? M().vpVar(k) : M().cssVar(k));
  /* Potential colors (Settings › Viewport, mace plotting --potential-colors): chemistry (the default) red = negative,
   * blue = positive; physics red = positive. A document's own choice (Color › Red = negative) over the setting. */
  const isPot = (t) => String(t || "").startsWith("potential");
  const chemOf = (d) => (d.opts.chem === null || d.opts.chem === undefined ? H.setting("volume.mepColors") !== "physics" : !!d.opts.chem);

  /* ---------------------------------------------------------------- data */
  /* the document's mesh quality (1.1.6): a preset and Advanced numbers, from Settings › Viewport › Mesh quality */
  const meshOf = (d) => (d.mesh = d.mesh || { quality: MQ.presetOf(H.setting(MQ.KEY)) });
  const gridQuery = (d) => MQ.volumeQuery(meshOf(d), H.setting("volume.quality") || "medium");
  async function loadVolume(d) {
    const meta = d.combo ? d.meta : d.orbitals ? await apiTask("orbitals/grid?" + q(picker.query(d)))
      : await apiTask("volume?" + q({ path: d.path, ...gridQuery(d) }));
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
    vd.__meta = m;
    return vd;
  }
  /** The same grid moved by t (Å): shares the data, kept per shift so a redraw makes no new copies. */
  function shifted(vd, t) {
    if (!t[0] && !t[1] && !t[2]) return vd;
    const key = t.map((x) => x.toFixed(6)).join(",");
    vd.__shift = vd.__shift || new Map();
    if (!vd.__shift.has(key)) {
      const m = vd.__meta, o = m.origin;
      vd.__shift.set(key, volumeData({ ...m, origin: [o[0] + t[0], o[1] + t[1], o[2] + t[2]] }, vd.data));
    }
    return vd.__shift.get(key);
  }

  /** The cell's surface at a level in the grid's index space, made once per grid, level and smoothing (the last
   * few kept: a signed grid's two levels, a level slider's way back). */
  function cellMesh(d, iso, smoothness) {
    const vd = d.vd, key = `${iso}|${smoothness}`;
    vd.__mesh = vd.__mesh || new Map();
    let mesh = vd.__mesh.get(key);
    if (!mesh) {
      mesh = PS.periodicMesh(vd.data, d.meta.shape, d.meta.wrap, iso, smoothness, window.$3Dmol.MarchingCube);
      vd.__mesh.set(key, mesh);
      while (vd.__mesh.size > 4) vd.__mesh.delete(vd.__mesh.keys().next().value);
    } else {
      vd.__mesh.delete(key); vd.__mesh.set(key, mesh);       // the most recent last
    }
    return mesh;
  }

  /* ---------------------------------------------------------------- what the toolbar's options draw */
  const IMAGES_MAX = 27;     // cells whose surfaces are drawn (each is a full isosurface of the grid)
  const ATOMS_MAX = 30000;
  /** The cell's own atoms as a structure (frac of the lattice from the grid's cell origin), or null for a
   * molecule. An orbital's grid already holds its n cells: the structure is the unit cell of that box. */
  function unitStructure(d) {
    const m = d.meta;
    if (m.__unit !== undefined) return m.__unit;
    let out = null;
    const pbc = m.pbc || [true, true, true];
    if (m.periodic && m.lattice && G) {
      const n = d.orbitals && m.orbital ? m.orbital.cells : [1, 1, 1];
      const L = m.lattice.map((r, k) => (pbc[k] ? r.map((x) => x / n[k]) : r.slice()));
      const o = m.cell_origin || m.origin || [0, 0, 0];
      const R = G.reciprocal(L);
      if (R) {
        const frac = [], symbols = [], seen = new Set(), cart = m.atoms.cart || [];
        cart.forEach((c, i) => {
          const f = G.fracOf(L, [c[0] - o[0], c[1] - o[1], c[2] - o[2]]);
          // the orbital's box lists every image of an atom: one site once (wrapped on the periodic axes)
          const w = f.map((x, k) => (pbc[k] ? x - Math.floor(x + 1e-3) : x));
          const key = w.map((x, k) => (pbc[k] ? Math.round(x * 1000) % 1000 : Math.round(x * 1000))).join(",");
          if (seen.has(key)) return;
          seen.add(key);
          frac.push(f); symbols.push(m.atoms.symbols[i]);
        });
        out = { cell: L, pbc, frac, symbols };
      }
    }
    m.__unit = out;
    return out;
  }
  /** The cells of the display range along each axis: an orbital's grid holds whole cells from the origin. */
  const cellsOf = (range) => (Array.isArray(range) ? range.map((x) => Math.max(1, x | 0)) : range.hi.map((x, k) => Math.max(1, Math.round(x - range.lo[k]))));
  /** What the toolbar's display options draw for a document: the atoms (the range's images, boundary atoms),
   * the cell box, bonds, labels, and the shifts of the surfaces (a cube's periodic images). */
  function display(d) {
    const m = d.meta, vo = H.viewOpts ? H.viewOpts() : {}, s = unitStructure(d), pbc = m.pbc || [true, true, true];
    const org = (m.cell_origin || m.origin || [0, 0, 0]).slice();
    const out = { bonds: vo.bonds !== false, labels: vo.labels || "off", cell: vo.cell !== false && !!m.periodic && !!m.lattice,
      origin: org, shifts: [[0, 0, 0]], images: 1, capped: false, s, range: [1, 1, 1] };
    if (!s) {
      // a molecule: its atoms as the cube lists them
      const n = m.atoms.numbers.length, cart = m.atoms.cart || [];
      const sm = { symbols: m.atoms.symbols };
      out.s = sm; out.origin = [0, 0, 0];
      out.list = { n, base: Int32Array.from({ length: n }, (_, i) => i), image: new Uint8Array(n), cart: Float64Array.from(cart.flat()) };
      out.box = out.cell ? { cell: m.lattice, pbc, origin: org } : null;
      out.cell = !!out.box && out.cell;
      return vacuumRepeat(out, d, vo);
    }
    const orbital = !!d.orbitals;
    const range = orbital ? cellsOf(vo.range || [1, 1, 1]).map((x, k) => (pbc[k] ? x : 1)) : (vo.range || [1, 1, 1]);
    out.range = range;
    const box = orbital ? { lo: [0, 0, 0], hi: range.map((x, k) => (pbc[k] ? x : 1)) } : (vo.box || { lo: [0, 0, 0], hi: [1, 1, 1] });
    const per = (k) => pbc[k];
    // atoms: the range's images, boundary atoms on the faces (too many: none, and the view says so)
    const cellsN = [0, 1, 2].reduce((n, k) => n * (per(k) ? Math.max(1, Math.round((Array.isArray(range) ? range[k] : range.hi[k] - range.lo[k]) + 1)) : 1), 1);
    out.tooMany = s.frac.length * cellsN > ATOMS_MAX;
    out.list = out.tooMany ? { n: 0, base: new Int32Array(0), image: new Uint8Array(0), cart: new Float64Array(0) }
      : G.displayList(s, range, !!vo.boundary);
    // the cell box: the whole cells from lo to hi along the periodic axes (an orbital's box is its grid)
    const C = s.cell;
    const boxCell = C.map((r, k) => (per(k) ? r.map((x) => x * (box.hi[k] - box.lo[k])) : r.slice()));
    const bo = org.slice();
    for (let k = 0; k < 3; k++) if (per(k) && !orbital) for (let j = 0; j < 3; j++) bo[j] += box.lo[k] * C[k][j];
    out.box = { cell: boxCell, pbc, origin: orbital ? org : bo };
    if (!orbital) {
      const shifts = [];
      const lo = [0, 1, 2].map((k) => (per(k) ? box.lo[k] : 0)), hi = [0, 1, 2].map((k) => (per(k) ? box.hi[k] : 1));
      for (let a = lo[0]; a < hi[0]; a++) for (let b = lo[1]; b < hi[1]; b++) for (let c = lo[2]; c < hi[2]; c++) {
        const i = [a, b, c];
        shifts.push([0, 1, 2].map((j) => i[0] * C[0][j] + i[1] * C[1][j] + i[2] * C[2][j]));
      }
      out.images = shifts.length;
      out.capped = shifts.length > IMAGES_MAX;
      out.shifts = shifts.slice(0, IMAGES_MAX);
    }
    return vacuumRepeat(out, d, vo);
  }
  /** More than one cell along a non-periodic axis (a slab cube's c, a molecule's x, y, z): the atoms and the
   * surfaces again at the Vacuum spacing's display vector (Cell: the cube's box along that axis). */
  function vacuumRepeat(out, d, vo) {
    const m = d.meta, pbc = m.pbc || [false, false, false], V = m.full_voxels, N = m.full_shape;
    const own = V && N ? V.map((r, k) => r.map((x) => x * N[k])) : null;
    const { offsets, cnt } = G.vacuumOffsets(m.periodic ? m.lattice : null, pbc, out.list.cart, vo.range || [1, 1, 1], vo.vacuum, own);
    if (offsets.length < 2) return out;
    // the range the drawing shows: a periodic axis keeps its cells (a fractional range's, rounded), a non-periodic one
    // the copies drawn at the vacuum spacing (x1 by default; an explicit multiple repeats), as vacuumOffsets counts them
    const own0 = cellsOf(vo.range || [1, 1, 1]);
    out.range = [0, 1, 2].map((k) => (pbc[k] ? own0[k] : cnt ? cnt[k] : own0[k]));
    if (!out.tooMany) out.list = G.repeatList(out.list, offsets);
    const all = [];
    offsets.forEach((o) => out.shifts.forEach((t) => all.push([0, 1, 2].map((j) => t[j] + o[j]))));
    out.images = Math.max(out.images, 1) * offsets.length;
    out.capped = all.length > IMAGES_MAX;
    out.shifts = all.slice(0, IMAGES_MAX);
    return out;
  }

  /* ---------------------------------------------------------------- drawing */
  function colors(d) {
    const css = vp;            // the 3D view's tokens: legible on a forced White or Black background too
    // spin: + blue and − red as the slices and mace plotting draw them, or the Maps & volumes preset's pair (B6)
    if (d.meta.type === "spin" || d.meta.type === "spin_difference") return M().volumeColors ? M().volumeColors() : [css("--plot-10"), css("--plot-1")];
    // a potential: + blue, − red in chemistry colors (as spin); physics swaps them
    if (isPot(d.meta.type) && !chemOf(d)) return [css("--vol-minus"), css("--vol-plus")];
    return [css("--vol-plus"), css("--vol-minus")];
  }
  function gradient(d) {
    const lo = d.opts.vmin, hi = d.opts.vmax;
    // 3Dmol's RWB maps its first bound to red: chemistry colors put red on the negative side (the CLI's RdBu),
    // physics on the positive side (RdBu_r)
    return chemOf(d) ? new window.$3Dmol.Gradient.RWB(lo, hi) : new window.$3Dmol.Gradient.RWB(hi, lo);
  }
  function draw(fit = false) {
    const d = doc;
    if (!d || !d.vd) return;
    const vw = v(), view = !fit && (vw.getModel() || drawnOnce) ? vw.getView() : null;
    vw.removeAllModels(); vw.removeAllShapes(); vw.removeAllLabels();
    shapes = []; planeShapes = [];
    const o = d.opts, m = d.meta, dd = display(d);
    lastDisplay = dd;
    // a new display range of the same document (3×3×3 after 1×1×1): the kept camera cut the supercell to the
    // cells that fit (a 3×3×3 showed as 3×2 rows, a slab's off screen), so the view is fitted again (the turn kept)
    const rangeKey = JSON.stringify(dd.range);
    const refit = !!view && d.rangeKey !== undefined && d.rangeKey !== rangeKey;
    d.rangeKey = rangeKey;
    if (o.atoms && dd.list.n) {
      const { list, s } = dd, org = dd.origin;
      const xyz = [String(list.n), "atoms"].concat(Array.from({ length: list.n }, (_, k) =>
        `${s.symbols[list.base[k]]} ${(list.cart[3 * k] + org[0]).toFixed(5)} ${(list.cart[3 * k + 1] + org[1]).toFixed(5)} ${(list.cart[3 * k + 2] + org[2]).toFixed(5)}`)).join("\n") + "\n";
      const model = vw.addModel(xyz, "xyz");
      // each drawn atom's site (the Magnetic style colors by it)
      model.selectedAtoms({}).forEach((a, k) => { a.properties = { ...(a.properties || {}), base: list.base[k] }; });
      // bonds are the sticks 3Dmol finds between the drawn atoms (the periodic images too); off: the atoms alone.
      // The atoms are drawn as the structure view draws them (1.1.6): the Inspector's Atom size, Bond radius,
      // radii, colors and the document's Style all apply here too (they were fixed small balls before)
      const draw = H.viewOpts ? H.viewOpts().style : "ballstick";
      const cs = M().colorSpec ? M().colorSpec() : { colorscheme: M().colorScheme() };
      const st = draw === "wireframe" ? (dd.bonds ? { line: { ...cs }, sphere: { scale: 0.12, ...cs } } : { sphere: { scale: 0.12, ...cs } })
        : M().styleFor(draw === "spacefill" || draw === "stick" ? draw : "ballstick", dd.bonds);
      model.setStyle({}, st);
      dd.atomStyle = st;
      if (dd.labels !== "off") {
        const size = Number(H.setting("labelSize")) || 12;
        for (let k = 0; k < list.n; k++) {
          if (list.image[k]) continue;
          const i = list.base[k], sym = s.symbols[i];
          const text = dd.labels === "element" ? sym : dd.labels === "index" ? String(i + 1) : `${sym}${i + 1}`;
          vw.addLabel(text, { position: { x: list.cart[3 * k] + org[0], y: list.cart[3 * k + 1] + org[1], z: list.cart[3 * k + 2] + org[2] },
            fontSize: size, fontColor: M().vpVar("--text"), backgroundColor: M().vpVar("--surface"), backgroundOpacity: 0.75,
            borderThickness: 0, inFront: true });
        }
      }
    }
    const [cp, cm] = colors(d);
    const lk = MQ.look(o, "double"), mk = MQ.knobs("volume", meshOf(d));
    const spec = (iso, color) => {
      const s = { isoval: iso, color, opacity: o.opacity, wireframe: lk.style === "wireframe", smoothness: mk.smoothness };
      if (lk.sides === "single") s.side = window.$3Dmol.FrontSide;
      if (d.colorVd && o.vmin !== null && o.vmax !== null) { s.voldata = d.colorVd; s.volscheme = gradient(d); delete s.color; }
      return s;
    };
    // a lobe with no surface at this level is not kept: 3Dmol gives an empty isosurface a bounding sphere
    // around (0, 0, 0), which the first fit would include (a molecule far from the origin shrank to a dot)
    // a cube's periodic images (the supercell): the same grid again, its origin moved by whole lattice vectors.
    // A grid that wraps (gui/volume.py meta.wrap) draws one mesh per level, padded across the cell's faces and
    // cut to the cell (viewport/periodic_surface.js), so the cells' surfaces join
    const wraps = (m.wrap || []).some(Boolean);
    const add = (iso, color) => {
      const mesh = wraps ? cellMesh(d, iso, mk.smoothness === undefined ? 1 : mk.smoothness) : null;
      dd.shifts.forEach((t) => {
        const sp = spec(iso, color);
        if (sp.voldata) sp.voldata = mesh ? PS.wrappedVolume(shifted(d.colorVd, t)) : shifted(d.colorVd, t);
        const o0 = m.origin, sh = mesh ? PS.addMeshShape(vw, mesh, [o0[0] + t[0], o0[1] + t[1], o0[2] + t[2]], m.voxels, sp)
          : vw.addIsosurface(shifted(d.vd, t), sp);
        if (!sh) return;
        if (vertexCount(sh) <= 0) { vw.removeShape(sh); return; }
        if (lk.style === "solid" && lk.shading === "smooth" || lk.style === "wireframe") { shapes.push(keepBuilt(sh)); return; }
        // Flat shading, edges or points: the surface's triangles (and its potential colors) drawn again in that style
        const parts = MQ.partsOfShape(sh);
        vw.removeShape(sh);
        const u = d.vd.unit || { x: 0.2, y: 0.2, z: 0.2 }, step = Math.max(0.05, 0.9 * Math.min(u.x, u.y, u.z));
        MQ.styled(vw, parts, lk, { color: sp.color || vp("--text-3"), opacity: o.opacity, step }).forEach((x) => shapes.push(x));
      });
    };
    if (o.plus) add(Math.abs(o.level), cp);
    if (o.minus && signedOf(d)) add(-Math.abs(o.level), cm);
    stats.surfaces = shapes.length;
    stats.vertices = shapes.reduce((n, s) => n + vertexCount(s), 0);
    window.__maceVolumeStats = { ...stats };
    // the first view fits the atoms and the surfaces; the cell (a slab's c spans its vacuum) and the plane come after
    if (view) vw.setView(view);
    if (!view || refit) {
      vw.zoomTo(); vw.zoom(0.9);                          // zoomTo keeps the turn (Reset View clears it first)
      // the rotation center: the drawn cells' center (a slab's at its atoms' mid-height), else the atoms' middle
      // (atoms repeated along a vacuum axis: the middle of the drawn ones there)
      const L = dd.list, rep = L && L.n && L.n > (m.atoms.cart || []).length, o0 = dd.origin || [0, 0, 0];
      const P = M().pivotOf, cart = rep ? Array.from({ length: L.n }, (_, i) => [0, 1, 2].map((j) => L.cart[3 * i + j] + o0[j])) : m.atoms.cart || [];
      const pv = !P ? null : m.periodic && m.lattice
        ? P(dd.box && dd.box.cell ? dd.box : { cell: m.lattice, pbc: m.pbc || [true, true, true], origin: m.cell_origin || m.origin || [0, 0, 0] }, null, cart)
        : cart.length ? P({ pbc: [false, false, false] }, null, cart) : null;
      // an orbital's box (a supercell of 2x2x2 cells and more) fills the view: zoomTo on everything left a
      // crystal's box at ~15% of the view
      // a molecule's box is atoms +- 5 bohr, wider than its lobes: zoomTo's sphere around the atoms and the
      // surfaces is the frame there; the picture fills at most 0.9 x SPHERE_SPAN of the view, so x1.9 keeps it
      // inside (at most ~83%)
      if (d.orbitals) { if (m.periodic) fitBox(vw, m); else vw.zoom(1.9); }
      if (pv && pv.every(Number.isFinite)) { const w = vw.getView(); vw.setView([-pv[0], -pv[1], -pv[2], ...w.slice(3)]); }
    }
    if (dd.cell) M().drawCell(vw, dd.box, M().vpVar("--cell"), 0.02, 1);
    if (o.slice) drawPlane(vw, d);
    drawnOnce = true;
    vw.render();
    // what the range asks for and the view cannot draw says so, once per range
    const note = dd.capped ? `${dd.images} cells: surfaces are drawn for the first ${IMAGES_MAX} (atoms and cell for all)`
      : dd.tooMany ? "The range holds too many atoms to draw: lower the display range" : "";
    if (note && d.lastNote !== note) toast(note, "warning");
    d.lastNote = note;
    renderBar(d);
  }
  /** The 8 corners (Å) of the drawn box: the grid's cell origin plus its edge vectors. */
  function boxCorners(m) {
    const L = m.lattice, o = m.cell_origin || m.origin;
    if (!L || !o) return [];
    const out = [];
    for (const a of [0, 1]) for (const b of [0, 1]) for (const c of [0, 1])
      out.push({ x: o[0] + a * L[0][0] + b * L[1][0] + c * L[2][0], y: o[1] + a * L[0][1] + b * L[1][1] + c * L[2][1],
        z: o[2] + a * L[0][2] + b * L[1][2] + c * L[2][2] });
    return out;
  }
  /** Frame the box so it fills the view. 3Dmol's zoomTo on the 8 corners (a hidden helper model) fits
   * their bounding sphere: the sphere's diameter D then spans SPHERE_SPAN of the view's shorter side (measured
   * on 3Dmol 2.x), so a box of screen extent ex x ey (the first view looks down z: screen x and y are model x
   * and y) spans SPHERE_SPAN * ex / D of that side, and zoom(f) scales the picture by f. modelToScreen cannot
   * measure it here: it reads a camera that 3Dmol updates later. */
  const SPHERE_SPAN = 0.485, FILL = 0.85;
  function fitBox(vw, m) {
    const pts = boxCorners(m);
    if (!pts.length) return;
    try {
      const helper = vw.addModel(["8", "box"].concat(pts.map((p) => `X ${p.x} ${p.y} ${p.z}`)).join("\n") + "\n", "xyz");
      helper.setStyle({}, {});
      vw.zoomTo({ model: helper });
      vw.removeModel(helper);
      const span = (k) => Math.max(...pts.map((p) => p[k])) - Math.min(...pts.map((p) => p[k]));
      const ex = span("x"), ey = span("y"), D = diag(pts);
      const r = vw.getCanvas().getBoundingClientRect(), W = r.width, Hh = r.height, short = Math.min(W, Hh);
      if (!(D > 0) || !(W > 0) || !(Hh > 0)) return;
      const sx = SPHERE_SPAN * ex / D * short / W, sy = SPHERE_SPAN * ey / D * short / Hh;
      const f = FILL / Math.max(sx, sy, 1e-6);
      if (Number.isFinite(f) && f > 1.02) vw.zoom(Math.min(f, 6));
    } catch (e) { /* no canvas yet: keep the first view */ }
  }
  function diag(pts) {
    let best = 0;
    for (const a of pts) for (const b of pts) best = Math.max(best, Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z));
    return best;
  }
  /** The part of the canvas the picture covers: the bounding box of its non-background pixels / the canvas. */
  function drawnSpan(vw) {
    const cv = vw.getCanvas(), c = document.createElement("canvas");
    c.width = cv.width; c.height = cv.height;
    const g = c.getContext("2d"); g.drawImage(cv, 0, 0);
    const px = g.getImageData(0, 0, c.width, c.height).data, bg = [px[0], px[1], px[2]];
    let x0 = c.width, x1 = -1, y0 = c.height, y1 = -1;
    for (let y = 0; y < c.height; y += 2) for (let x = 0; x < c.width; x += 2) {
      const i = 4 * (y * c.width + x);
      if (Math.abs(px[i] - bg[0]) + Math.abs(px[i + 1] - bg[1]) + Math.abs(px[i + 2] - bg[2]) > 30) {
        if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
    }
    return x1 < 0 ? [0, 0] : [(x1 - x0) / c.width, (y1 - y0) / c.height];
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
  /* ---------------------------------------------------------------- the slice plane (1.1.6) */
  /** The plane the Slice section takes: a lattice plane of the grid (a, b, c), a Miller plane or normal to [uvw]. */
  function sliceGeom(d) {
    const o = d.opts, m = d.meta;
    if (!SW || !m) return null;
    if (!planeMode(o)) {
      const n = m.full_shape["abc".indexOf(o.axis)];
      return SW.axisGeometry(m, o.axis, Number.isInteger(o.index) && o.index >= 0 && o.index < n ? o.index : Math.floor(n / 2));
    }
    return SW.planeGeometry(SW.boxOf(m), { mode: o.axis, ints: o[o.axis], pos: o.pos, tilt: o.tilt, twist: o.twist });
  }
  /** The plane's colors in the 3D view: the fill in the view's accent, the outline in its cell color (3:1 on
   * any background: tokens.css), so it never vanishes on a White background in a dark theme (or the reverse). */
  const planeColors = () => {
    // a custom background of any color: the outline moved toward black or white until it stands out (contrast.js)
    const bg = M().viewerBg ? M().viewerBg() : vp("--viewport-bg");
    return { fill: legible(vp("--accent"), bg, 2), edge: legible(vp("--cell"), bg), bg };
  };
  /** The plane as its own shapes (fill and outline), redrawn alone while it moves (the surfaces stay). */
  function drawPlane(vw, d) {
    planeShapes.forEach((s) => { try { vw.removeShape(s); } catch (e) { /* gone with removeAllShapes */ } });
    planeShapes = [];
    const g = sliceGeom(d);
    d.geom = g;
    if (!g || !g.polygon || g.polygon.length < 3) return;
    const P = g.polygon, c = planeColors(), n = g.normal, faces = [];
    for (let k = 1; k + 1 < P.length; k++) faces.push(0, k, k + 1, 0, k + 1, k);        // both sides
    const xyz = (p) => ({ x: p[0], y: p[1], z: p[2] });
    planeShapes.push(vw.addCustom({ vertexArr: P.map(xyz), normalArr: P.map(() => ({ x: n[0], y: n[1], z: n[2] })), faceArr: faces,
      color: c.fill, opacity: 0.32 }));
    const size = Math.max(...SW.boxOf(d.meta).edges.map((e) => Math.hypot(...e)));
    const r = Math.min(0.08, Math.max(0.03, 0.004 * size));
    P.forEach((p, k) => planeShapes.push(vw.addCylinder({ start: xyz(p), end: xyz(P[(k + 1) % P.length]), radius: r, color: c.edge, fromCap: 1, toCap: 1 })));
  }
  /* The color bar of a surface colored by a potential (1.1.6 fix): thin and vertical at the view's right edge, as the
   * slice and piezo colorbars, with round ticks at their true places in the potential's unit (Ha e⁻¹) and the gradient
   * sampled from the surface's own 3Dmol gradient. Sign words only where the range holds both signs; a range of one
   * sign says so (maprange.js potentialWords; the CLI's caption, crystal_cubeviz_plotly.potential_scale_words). The
   * same model is drawn into Export Image's PNG (drawBar). */
  const BAR = { w: 10, h: 150, right: 28, pad: 8, font: 11, gap: 4 };
  function barModel(d) {
    if (!d || !d.colorVd || d.opts.vmin === null || d.opts.vmax === null) return null;
    const lo = Math.min(d.opts.vmin, d.opts.vmax), hi = Math.max(d.opts.vmin, d.opts.vmax), chem = chemOf(d);
    const g = gradient(d), hex = (v) => "#" + (g.valueToHex(v) >>> 0).toString(16).padStart(6, "0").slice(-6);
    const span = hi - lo || 1;
    const stops = Array.from({ length: 25 }, (_, i) => ({ f: i / 24, c: hex(lo + (span * i) / 24) }));
    const ticks = MR.niceTicks(lo, hi, 5).map((v) => ({ v, f: (v - lo) / span, text: MR.tickText(v) }));
    const words = MR.potentialWords(lo, hi, chem), auto = d.opts.range === "auto";
    const unit = L.plain((d.colorMeta && d.colorMeta.unit) || "a.u.");
    const top = words.spans ? "positive (e⁻-poor)" : "", bottom = words.spans ? "negative (e⁻-rich)" : "";
    const note = words.spans ? "" : words.note === "zero" ? "all zero" : auto ? `${words.note} on this surface` : `range ${words.note}`;
    const label = `Potential (${unit}) from ${MR.tickText(lo)} to ${MR.tickText(hi)}: ${words.spans ? `${words.low}, ${words.high}` : `${note}, ${words.low}, ${words.high}`}`;
    return { lo, hi, chem, stops, ticks, words, title: "Potential", unit: `(${unit})`, top, bottom, note, label };
  }
  function renderBar(d) {
    const bar = $(".vv-bar"), m = barModel(d);
    if (!m) { bar.hidden = true; return; }
    bar.hidden = false;
    bar.setAttribute("role", "img");
    bar.setAttribute("aria-label", m.label);
    bar.title = m.label;
    const grad = `linear-gradient(to top, ${m.stops.map((s) => `${s.c} ${(s.f * 100).toFixed(1)}%`).join(", ")})`;
    bar.innerHTML = `<div class="vv-cb-title">${esc(m.title)}<br><span class="vv-cb-unit">${esc(m.unit)}</span></div>
      ${m.top ? `<div class="vv-cb-end">${esc(m.top)}</div>` : ""}
      <div class="vv-cb-body"><div class="vv-grad" style="background: ${grad}"></div>
        <div class="vv-ticks mono">${m.ticks.map((t) => `<span style="bottom: ${(t.f * 100).toFixed(2)}%">${esc(t.text)}</span>`).join("")}</div></div>
      ${m.bottom ? `<div class="vv-cb-end">${esc(m.bottom)}</div>` : ""}
      ${m.note ? `<div class="vv-cb-note">${esc(m.note)}</div>` : ""}
      <div class="vv-legend" hidden>${esc(m.label)}</div>`;
  }
  /** The color bar into a picture (Export Image, Copy Image): `g` a 2D context of W x H pixels, `k` pixels per view px. */
  function drawBar(g, W, H, k) {
    const m = barModel(doc);
    if (!m || $(".vv-bar").hidden) return;
    const css = M().cssVar, fam = css("--font-sans") || "sans-serif", fs = BAR.font * k;
    const font = (w = 400) => `${w} ${fs}px ${fam}`;
    g.save();
    g.font = font();
    const tickW = Math.max(0, ...m.ticks.map((t) => g.measureText(t.text).width));
    const lines = [[m.title, 600], [m.unit, 400]], ends = [m.top, m.bottom, m.note].filter(Boolean);
    g.font = font(600);
    let w = g.measureText(m.title).width;
    g.font = font();
    for (const t of [m.unit, ...ends]) w = Math.max(w, g.measureText(t).width);
    w = Math.max(w, (BAR.w + 4 + BAR.gap) * k + tickW);
    const lh = fs * 1.3, pad = BAR.pad * k, bh = BAR.h * k;
    const ph = pad * 2 + lines.length * lh + (m.top ? lh : 0) + fs * 0.6 + bh + fs * 0.6 + (m.bottom ? lh : 0) + (m.note ? lh : 0);
    const pw = w + pad * 2, x0 = W - BAR.right * k - pw, y0 = (H - ph) / 2;
    g.globalAlpha = 0.9; g.fillStyle = css("--surface") || "#ffffff";
    g.beginPath(); (g.roundRect ? g.roundRect(x0, y0, pw, ph, 6 * k) : g.rect(x0, y0, pw, ph)); g.fill();
    g.globalAlpha = 1; g.strokeStyle = css("--border") || "#cccccc"; g.lineWidth = Math.max(1, k); g.stroke();
    g.textBaseline = "top"; g.textAlign = "left";
    let y = y0 + pad;
    g.fillStyle = css("--text") || "#111111"; g.font = font(600); g.fillText(m.title, x0 + pad, y); y += lh;
    g.fillStyle = css("--text-2") || "#555555"; g.font = font(); g.fillText(m.unit, x0 + pad, y); y += lh;
    if (m.top) { g.fillText(m.top, x0 + pad, y); y += lh; }
    y += fs * 0.6;
    const gx = x0 + pad, lin = g.createLinearGradient(0, y + bh, 0, y);
    m.stops.forEach((s) => lin.addColorStop(s.f, s.c));
    g.fillStyle = lin; g.fillRect(gx, y, BAR.w * k, bh);
    g.strokeStyle = css("--border") || "#cccccc"; g.strokeRect(gx, y, BAR.w * k, bh);
    g.fillStyle = css("--text-2") || "#555555"; g.textBaseline = "middle";
    for (const t of m.ticks) {
      const ty = y + bh * (1 - t.f);
      g.fillRect(gx + BAR.w * k, ty - k / 2, 4 * k, k);
      g.fillText(t.text, gx + (BAR.w + 4 + BAR.gap) * k, ty);
    }
    y += bh + fs * 0.6; g.textBaseline = "top";
    if (m.bottom) { g.fillText(m.bottom, x0 + pad, y); y += lh; }
    if (m.note) g.fillText(m.note, x0 + pad, y);
    g.restore();
  }

  /* ---------------------------------------------------------------- slice */
  async function loadSlice(d) {
    const m = d.meta, o = d.opts;
    // a difference made here: the plane of the difference itself, computed from its operands
    const where = d.combo ? d.meta.combined.paths.map((p, k) => `path=${encodeURIComponent(p)}&coef=${encodeURIComponent(d.meta.combined.coefficients[k])}`).join("&") + "&"
      : "";
    const seq = (d.sliceSeq = (d.sliceSeq || 0) + 1);
    let r;
    if (planeMode(o)) {
      const ix = o[o.axis];
      r = await apiTask("volume/plane?" + where + q({ ...(d.combo ? {} : { path: d.path }), mode: o.axis, i: ix[0], j: ix[1], k: ix[2],
        pos: o.pos, tilt: o.tilt, twist: o.twist }));
      r.axis = o.axis;
    } else {
      const ax = "abc".indexOf(o.axis), n = m.full_shape[ax], i = o.index;
      if (!Number.isInteger(i) || i < 0 || i >= n) o.index = Math.floor(n / 2);       // a plane of this grid only
      r = await apiTask("volume/slice?" + where + q({ ...(d.combo ? {} : { path: d.path }), axis: o.axis, index: o.index }));
      r.mode = "axis";
    }
    // a newer plane (the slider moved on) supersedes this one
    if (doc !== d || d.sliceSeq !== seq) return;
    d.slice = r;
    await plotSlice(d);
    if (planeShapes.length || !o.slice) { const vw = v(); drawPlane(vw, d); vw.render(); if (preview) preview.draw(); }
    else draw();
  }
  /** While the plane moves: the plane in the 3D view, the preview, and a quick map from the 3D view's own grid
   * (one per frame); the full grid's plane follows when it stops (`settle`). */
  function planeMoved(d, settle) {
    const vw = v();
    if (!d.opts.slice) { d.opts.slice = true; renderOpts(); }
    drawPlane(vw, d);
    vw.render();
    if (preview) preview.draw();
    syncPlaneReadout(d);
    if (!mapFrame) mapFrame = requestAnimationFrame(() => {
      mapFrame = 0;
      if (doc !== d || !d.vd || !d.geom) return;
      const quick = SW.samplePlane(d.vd.__meta || d.meta, d.vd.data, d.geom);
      if (quick) plotSlice(d, { ...quick, ...planeInfo(d.geom), preview: true }).catch(() => {});
    });
    clearTimeout(d.settleTimer);
    d.settleTimer = setTimeout(() => { if (doc === d) loadSlice(d).catch((er) => toast(er.message, "error")); }, settle ? 30 : 350);
  }
  /** What the map's heading says about a plane (also on the server's answer). */
  const planeInfo = (g) => ({ mode: g.mode, axis: g.mode === "axis" ? g.axis : g.mode, ints: g.ints, pos: g.pos, tilt: g.tilt, twist: g.twist,
    position_A: g.position_A, dmin: g.dmin, dmax: g.dmax, spacing_A: g.spacing_A, miller_t: g.miller_t });
  /** The inspector's readouts follow a drag in the preview or the 3D view. */
  function syncPlaneReadout(d) {
    const o = d.opts, set = (id, v, t) => { const el = $(`#${id}`, optsHost); if (el && document.activeElement !== el) el.value = String(v); const tv = $(`#${id}-v`, optsHost); if (tv) tv.textContent = t; };
    if (planeMode(o)) {
      set("vo-pos", o.pos, o.pos.toFixed(3));
      set("vo-tilt", o.tilt, `${o.tilt.toFixed(1)}°`);
      set("vo-twist", o.twist, `${o.twist.toFixed(1)}°`);
      const sq = $('[data-vo="planeSquare"]', optsHost);
      if (sq) sq.disabled = !o.tilt && !o.twist;
    } else if (Number.isInteger(o.index)) set("vo-index", o.index, String(o.index));
    const cv = $("#vo-preview", optsHost);
    if (cv && d.geom) { cv.setAttribute("aria-valuenow", String(Math.round(1000 * (d.geom.pos || 0)) / 1000)); cv.setAttribute("aria-valuetext", SW.planeLabel(d.geom)); }
  }
  /** A change from the preview or an Alt-drag in the 3D view: {pos} a step across the box, {tilt, twist} degrees. */
  function sliceChange(d, delta, final) {
    const o = d.opts, m = d.meta;
    if ((delta.tilt || delta.twist) && !planeMode(o)) {
      // turning a lattice plane of the grid: the same plane as a Miller plane, which can turn
      const g = sliceGeom(d);
      o.hkl = o.axis === "a" ? [1, 0, 0] : o.axis === "b" ? [0, 1, 0] : [0, 0, 1];
      o.pos = g && g.dmax > g.dmin ? Math.max(0, Math.min(1, (g.d - g.dmin) / (g.dmax - g.dmin))) : 0.5;
      o.axis = "hkl"; o.tilt = 0; o.twist = 0;
      renderOpts();
    }
    if (delta.pos) {
      if (planeMode(o)) o.pos = Math.max(0, Math.min(1, o.pos + delta.pos));
      else {
        const n = m.full_shape["abc".indexOf(o.axis)];
        d.axisPos = Math.max(0, Math.min(1, (d.axisPos ?? (Number.isInteger(o.index) ? o.index : Math.floor(n / 2)) / Math.max(1, n - 1)) + delta.pos));
        o.index = Math.round(d.axisPos * Math.max(1, n - 1));
      }
    }
    if (delta.tilt) o.tilt = Math.max(-90, Math.min(90, o.tilt + delta.tilt));
    if (delta.twist) o.twist = Math.max(-90, Math.min(90, o.twist + delta.twist));
    if (final && !delta.pos && !delta.tilt && !delta.twist) d.axisPos = undefined;
    planeMoved(d, final);
  }
  /** The slice map, in the theme's colors (drawn again when the theme changes). `r`: a quick map while the plane moves. */
  async function plotSlice(d, r = d.slice) {
    if (!r) return;
    await M().loadPlotly();
    if (doc !== d) return;
    const el = $(".vv-slice-plot");
    $(".vv-slice").hidden = false;
    const plane = r.mode && r.mode !== "axis";
    if (!plane) $(".vv-slice-head").textContent = `Plane ${d.opts.axis} = ${r.index} of ${r.n} (${(r.frac).toFixed(3)} fractional, ${r.position_A.toFixed(3)} Å from the grid's origin)`;
    else {
      const g = { ...r, ints: r.ints || [0, 0, 1], tilt: r.tilt || 0, twist: r.twist || 0 };
      $(".vv-slice-head").textContent = `${SW.planeLabel(g)} at ${Number(r.pos).toFixed(3)} across the box: ${r.position_A.toFixed(3)} Å along its normal of ${(r.dmax - r.dmin).toFixed(3)} Å`
        + (r.mode === "hkl" && r.spacing_A ? ` · h x + k y + l z = ${r.miller_t.toFixed(3)}, d = ${r.spacing_A.toFixed(3)} Å` : "")
        + (r.preview ? " · preview from the 3D view's grid" : "");
    }
    const zrows = plane ? r.z : transpose(r.cart);
    const signed = signedOf(d);
    let zmax = null;
    if (signed) { zmax = 1e-30; for (const row of zrows) for (const x of row) if (x !== null && Math.abs(x) > zmax) zmax = Math.abs(x); }
    // the Maps & volumes preset's colormap: signed + blue / − red (plotly.js RdBu reversed), as the 3D view and
    // mace plotting; a potential in physics colors (Potential colors) the other way round: red = +
    const sc = M().mapScale ? M().mapScale(signed) : { colorscale: signed ? "RdBu" : "Viridis", reversescale: signed, contours: "both" };
    if (signed && isPot(d.meta.type) && !chemOf(d)) sc.reversescale = !sc.reversescale;
    const trace = { type: "contour", x: r.x, y: r.y, z: zrows, colorscale: sc.colorscale, reversescale: sc.reversescale,
      ncontours: d.opts.contours, contours: { coloring: sc.contours === "lines" ? "lines" : "heatmap", showlines: sc.contours !== "filled" }, line: { width: 0.5 },
      zmid: signed ? 0 : undefined, zmin: signed ? -zmax : undefined, zmax: signed ? zmax : undefined,
      // the colorbar names the quantity and its unit (mace plotting's slices too: plot markup, labels.js)
      colorbar: { thickness: 10, len: 0.9, title: { text: `${L.html(d.meta.label || "")}${d.meta.unit ? `<br>(${d.meta.unit})` : ""}`, font: { size: 10 } } },
      hovertemplate: `%{x:.2f}, %{y:.2f} Å: %{z:.4g}${d.meta.unit ? ` ${d.meta.unit}` : ""}<extra></extra>` };
    const css = M().plotVar || M().cssVar;     // the theme's tokens, or the forced plot surface's
    // a (hkl) or [uvw] plane: its in-plane axes, and the outline of the cut through the box
    // (slice_cli._axes names them alike): a grid plane along its first cell axis and across it, in the plane
    const a1 = "abc".replace(r.axis || d.opts.axis || "c", "")[0];
    const ax = plane ? [L.withUnit(`Along ${L.SYMBOLS.e_1}`, "A"), L.withUnit(`Along ${L.SYMBOLS.e_2}`, "A")]
      : [L.withUnit(`Along ${a1}`, "A"), L.withUnit(`⊥ ${a1}`, "A")];
    const shapes = plane && r.outline ? [{ type: "path", line: { color: css("--text-2"), width: 1 },
      path: r.outline[0].map((u, k) => `${k ? "L" : "M"}${u},${r.outline[1][k]}`).join(" ") + " Z" }] : [];
    const fig = { traces: [trace], layout: { margin: { l: 44, r: 10, t: 6, b: 36 },
      ...(M().plotPaper ? M().plotPaper() : { paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)" }),
      font: { color: css("--text"), size: 11 }, xaxis: { title: { text: ax[0] }, scaleanchor: "y", constrain: "domain", gridcolor: css("--line"), zerolinecolor: css("--line") },
      yaxis: { title: { text: ax[1] }, gridcolor: css("--line"), zerolinecolor: css("--line") }, shapes } };
    if (M().styleFigure) M().styleFigure(fig.traces, fig.layout, "maps");       // Settings › Appearance › Plot style (Maps & volumes)
    await window.Plotly.react(el, fig.traces, fig.layout, { displaylogo: false, responsive: true });
    d.mapShown = { mode: r.mode || "axis", preview: !!r.preview, nx: r.x.length, ny: r.y.length };
  }
  // the element colors changed (Settings, Element Colors): the atoms again
  document.addEventListener("mace-colors", () => { if (doc && doc.vd) draw(); });
  // a theme change: the surfaces' and the slice map's colors come from the theme's tokens
  document.addEventListener("mace-theme", () => {
    if (!doc || !doc.vd) return;
    draw();
    if (doc.slice && doc.opts.slice) plotSlice(doc).catch(() => {});
    if (preview) preview.draw();
  });
  // the preview looks at the box from the 3D view's direction: it turns with the view
  document.addEventListener("mace-painted", (e) => { if (e.detail === "viewer-volume" && preview && doc && !preview.dragging) preview.draw(); });
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
  /* Style, shading, sides and mesh quality (1.1.6, viewport/mesh_quality.js), the Advanced numbers in a disclosure */
  function meshControls(d) {
    const o = d.opts, lk = MQ.look(o, "double"), mesh = meshOf(d), mk = MQ.knobs("volume", mesh), m = d.meta;
    const sel = (id, label, opts, v, tier = "") => `<label class="po-row"${tier}><span>${label}</span><select id="${id}">${opts.map(([x, t]) =>
      `<option value="${x}"${x === v ? " selected" : ""}>${esc(t)}</option>`).join("")}</select></label>`;
    const grid = !d.combo && !d.orbitals;
    const num = (k) => { const [lo, hi] = MQ.VOLUME_LIMITS[k], [label, help] = MQ.KNOB_LABEL[k];
      const v = k === "points" && mk.points == null ? MQ.LEGACY_POINTS[H.setting("volume.quality")] || MQ.LEGACY_POINTS.medium : mk[k];
      return `<label class="po-row" title="${esc(help)}"><span>${esc(label)}</span><input type="number" id="vo-mk-${k}" min="${lo}" max="${hi}" step="${k === "points" ? 50000 : 1}" value="${v}"></label>`; };
    const up = m.upsample > 1 ? ` · refined ×${m.upsample}` : "";
    return `${sel("vo-style", "Style", MQ.STYLES, lk.style)}${sel("vo-shading", "Shading", MQ.SHADINGS, lk.shading, ' data-tier="advanced"')}
      ${sel("vo-sides", "Sides", MQ.SIDES, lk.sides, ' data-tier="advanced"')}
      ${sel("vo-quality", "Mesh quality", MQ.PRESETS.map((x) => [x, MQ.PRESET_LABEL[x]]), mesh.quality)}
      <p class="vo-stat mono" role="status" aria-live="polite">${d.loadingMesh ? `Re-meshing at ${MQ.PRESET_LABEL[mesh.quality]}…` : `${m.shape.join("×")} grid${up}${stats.vertices ? ` · ${MQ.fmtCount(stats.vertices)} vertices` : ""}`}</p>
      <details class="vo-adv" data-tier="advanced"${d.advOpen ? " open" : ""}><summary>Advanced</summary>
        ${grid ? num("points") + num("upsample") : ""}${num("smoothness")}
        <p class="vo-stat">${esc(MQ.PRESET_HELP[mesh.quality])}${d.orbitals ? ". An orbital's grid: the Quality above" : d.combo ? ". A difference keeps the grid it was computed on" : ""}</p></details>`;
  }
  /** A new mesh quality: a cube's grid again at it (the surfaces only for an orbital or a difference). */
  async function remeshVolume(d) {
    if (d.combo || d.orbitals) { draw(); renderOpts(); return; }
    d.loadingMesh = true; renderOpts();
    try {
      await loadVolume(d);
      if (d.colorPath) await setColor(d, d.colorPath, true);
    } catch (e) { toast(`Mesh quality: ${e.message}`, "error"); } finally { d.loadingMesh = false; }
    if (doc !== d) return;
    header(d); draw(); renderOpts();
  }
  function renderOpts() {
    const d = doc;
    if (!d || !d.meta) { optsHost.innerHTML = ""; return; }
    const o = d.opts, m = d.meta, sl = logSlider(d), st = m.stats;
    const lobe = m.type === "spin" ? ["Spin up", "Spin down"] : m.type === "orbital" ? (m.signed ? ["+ phase", "− phase"] : ["Show surface", ""]) : ["Show + lobe", "Show − lobe"];
    const orb = d.orbitals && m.orbital;
    const integral = st.integral !== undefined ? (st.integral === null
      ? `<p class="vo-stat" title="${esc(st.integral_note)}">Integrated density: not shown (${esc(st.integral_note)})</p>`
      : `<p class="vo-stat" title="${esc(st.integral_note)}">Integrated density on this grid: <b>${st.integral.toFixed(1)} e</b></p>`) : "";
    const noise = st.noise_note ? `<p class="vo-stat vo-warn">${esc(st.noise_note)}</p>` : "";
    optsHost.innerHTML = `<div class="po-h">Surface</div>${noise}
      <label class="po-row"><span>Level</span><input type="range" id="vo-level" min="${sl.min}" max="${sl.max}" step="${sl.step}" value="${sl.value}" data-log="${sl.log}"><span class="po-u mono" id="vo-level-v">${fmtv(o.level)}</span></label>
      ${orb ? `<p class="vo-stat">Default level ${fmtv(m.level)}: about 90 % of |ψ|² is inside the surface.</p>
        <div class="po-acts"><button class="tb-btn-text" data-vo="orbLevel">${icon("rotate-ccw")}<span>Default Level</span></button></div>` : ""}
      <label class="chk po-row"><input type="checkbox" id="vo-plus"${o.plus ? " checked" : ""}> ${lobe[0]}</label>
      ${signedOf(d) ? `<label class="chk po-row"><input type="checkbox" id="vo-minus"${o.minus ? " checked" : ""}> ${lobe[1]}</label>` : ""}
      <label class="po-row"><span>Opacity</span><input type="range" id="vo-opacity" min="0.1" max="1" step="0.05" value="${o.opacity}"><span class="po-u mono" id="vo-opacity-v">${o.opacity.toFixed(2)}</span></label>
      ${meshControls(d)}
      <label class="chk po-row"><input type="checkbox" id="vo-atoms"${o.atoms ? " checked" : ""}> Atoms</label>
      ${m.periodic ? `<label class="chk po-row"><input type="checkbox" id="vo-cell"${H.setting("showCell") !== false ? " checked" : ""}> Cell</label>` : ""}
      ${d.colorVd ? `<div class="po-h">Color</div>
        <div class="po-row"><span>Range</span><span class="segment"><button data-vo-range="auto" aria-pressed="${o.range === "auto"}">Auto</button><button data-vo-range="custom" aria-pressed="${o.range === "custom"}">Custom</button></span></div>
        <label class="po-row"><span>Min</span><input type="number" id="vo-vmin" step="any" value="${o.vmin}"${o.range === "auto" ? " disabled" : ""}></label>
        <label class="po-row"><span>Max</span><input type="number" id="vo-vmax" step="any" value="${o.vmax}"${o.range === "auto" ? " disabled" : ""}></label>
        <label class="chk po-row"><input type="checkbox" id="vo-chem"${chemOf(d) ? " checked" : ""}> Red = negative (chemistry colors)</label>
        <div class="po-acts"><button class="tb-btn-text" data-vo="resetRange">${icon("rotate-ccw")}<span>Reset Range</span></button></div>`
        : isPot(m.type) && signedOf(d) ? `<div class="po-h">Color</div>
        <label class="chk po-row"><input type="checkbox" id="vo-chem"${chemOf(d) ? " checked" : ""}> Red = negative (chemistry colors)</label>` : ""}
      ${d.orbitals ? "" : sliceSection(d)}
      <div class="po-h">Statistics (full grid)</div>
      <p class="vo-stat mono">min ${fmtv(st.min)} · max ${fmtv(st.max)}</p>${integral}
      ${orb ? `<div class="po-h">Orbital</div>${m.orbital.levels.map((l) => `<p class="vo-stat mono">MO ${l.n}: ${l.e_ev.toFixed(3)} eV · occupation ${Number(l.occ).toFixed(3)}</p>`).join("")}
        <p class="vo-stat">${m.full_shape.join("×")} points, spacing ${(m.orbital.spacing_bohr * 0.529177210903).toFixed(3)} Å${m.periodic ? ` · ${m.orbital.cells.join("×")} cells` : ""}</p>
        ${m.orbital.gauge_note ? `<p class="vo-stat">${esc(m.orbital.gauge_note)}</p>` : ""}` : ""}
      ${d.combo || d.orbitals ? "" : `<div class="po-acts"><button class="tb-btn-text" data-vo="cli">${icon("external-link")}<span>Render with mace plotting</span></button></div>`}`;
    mountPreview(d);
  }
  const PRESETS = [["hkl", [1, 0, 0]], ["hkl", [0, 1, 0]], ["hkl", [0, 0, 1]], ["hkl", [1, 1, 0]], ["hkl", [1, 0, 1]], ["hkl", [0, 1, 1]], ["hkl", [1, 1, 1]], ["hkl", [1, -1, 0]]];
  const ixText = (v) => v.map((x) => (x < 0 ? `−${-x}` : String(x))).join(" ");
  /** The Slice section: the plane (a lattice plane of the grid, a Miller plane or normal to a direction), the mini
   * unit-cell preview it is drawn in, its position and its turn. Niche controls are Advanced (data-tier). */
  function sliceSection(d) {
    const o = d.opts, m = d.meta, pm = planeMode(o), ax = "abc".indexOf(o.axis), n = pm ? 0 : m.full_shape[ax];
    const opt = (v, t) => `<option value="${v}"${o.axis === v ? " selected" : ""}>${t}</option>`;
    const ints = pm ? o[o.axis] : null;
    return `<div class="po-h">Slice</div>
      <label class="chk po-row"><input type="checkbox" id="vo-slice"${o.slice ? " checked" : ""}> Show a slice plane</label>
      <label class="po-row"><span>Plane</span><select id="vo-axis">${opt("a", "a: grid plane (1 0 0)")}${opt("b", "b: grid plane (0 1 0)")}${opt("c", "c: grid plane (0 0 1)")}
        ${opt("hkl", "Miller plane (h k l)")}${opt("uvw", "Normal to a direction [u v w]")}</select></label>
      ${pm ? `<div class="po-row vo-ints"><span>${o.axis === "hkl" ? "h k l" : "u v w"}</span><span class="vo-int3">${[0, 1, 2].map((k) =>
        `<input type="number" id="vo-i${k}" min="-12" max="12" step="1" value="${ints[k]}" aria-label="${(o.axis === "hkl" ? "hkl" : "uvw")[k]}">`).join("")}</span></div>` : ""}
      <div class="vo-presets" role="group" aria-label="Common planes">${PRESETS.map(([md, v]) =>
        `<button type="button" class="vo-chip mono" data-vo-plane="${md}:${v.join(",")}" aria-pressed="${pm && o.axis === md && o[md].join(",") === v.join(",")}">(${ixText(v)})</button>`).join("")}</div>
      <div class="vo-preview-wrap"><canvas id="vo-preview" class="vo-preview" tabindex="0" role="slider" aria-valuemin="0" aria-valuemax="1"
        aria-label="Slice plane in the unit cell: drag to move it along its normal, Shift-drag to turn it, arrow keys step it"></canvas>
        <p class="vo-hint">Drag the plane along its normal · Shift-drag turns it · Alt-drag in the 3D view does the same</p></div>
      ${pm ? `<label class="po-row"><span>Position</span><input type="range" id="vo-pos" min="0" max="1" step="0.001" value="${o.pos}"><span class="po-u mono" id="vo-pos-v">${o.pos.toFixed(3)}</span></label>
        <label class="po-row" data-tier="advanced"><span>Tilt</span><input type="range" id="vo-tilt" min="-90" max="90" step="0.5" value="${o.tilt}"><span class="po-u mono" id="vo-tilt-v">${o.tilt.toFixed(1)}°</span></label>
        <label class="po-row" data-tier="advanced"><span>Twist</span><input type="range" id="vo-twist" min="-90" max="90" step="0.5" value="${o.twist}"><span class="po-u mono" id="vo-twist-v">${o.twist.toFixed(1)}°</span></label>
        <div class="po-acts" data-tier="advanced"><button class="tb-btn-text" data-vo="planeSquare"${o.tilt || o.twist ? "" : " disabled"}>${icon("rotate-ccw")}<span>Untilt the Plane</span></button></div>`
        : `<label class="po-row"><span>Position</span><input type="range" id="vo-index" min="0" max="${n - 1}" step="1" value="${o.index ?? Math.floor(n / 2)}"><span class="po-u mono" id="vo-index-v">${o.index ?? "mid"}</span></label>`}
      <label class="po-row" data-tier="advanced"><span>Contours</span><input type="number" id="vo-contours" min="2" max="60" step="1" value="${o.contours}"></label>`;
  }
  /** The preview canvas of the Slice section (made again with the section). */
  function mountPreview(d) {
    const cv = $("#vo-preview", optsHost);
    if (!cv || !SW) { preview = null; return; }
    preview = SW.createSlicePreview(cv, {
      css: (k) => M().cssVar(k),
      S: () => {
        if (doc !== d || !d.meta) return null;
        const vw = viewer, w = vw && vw.getView ? vw.getView() : null;
        return { box: SW.boxOf(d.meta), geom: d.geom || sliceGeom(d), quat: w ? w.slice(4, 8) : null, canRotate: true };
      },
      change: (delta, final) => { if (doc === d) sliceChange(d, delta, final); },
    });
    if (!d.geom) d.geom = sliceGeom(d);
    requestAnimationFrame(() => { if (preview) preview.draw(); });
    syncPlaneReadout(d);
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
    else if (t.id === "vo-opacity") { o.opacity = Number(t.value); $("#vo-opacity-v", optsHost).textContent = o.opacity.toFixed(2); }
    else if (t.id === "vo-index") {
      o.index = Number(t.value); $("#vo-index-v", optsHost).textContent = String(o.index);
      if (o.slice) { const vw = v(); drawPlane(vw, d); vw.render(); }
      if (preview) preview.draw();
      clearTimeout(building); building = setTimeout(() => { if (o.slice) loadSlice(d).catch((er) => toast(er.message, "error")); }, 150); return;
    }
    else if (t.id === "vo-pos" || t.id === "vo-tilt" || t.id === "vo-twist") {
      o[t.id.slice(3)] = Number(t.value);
      planeMoved(d, false);
      const sq = $('[data-vo="planeSquare"]', optsHost);
      if (sq) sq.disabled = !o.tilt && !o.twist;
      return;
    }
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
    else if (t.id === "vo-style") { o.style = t.value; o.wire = false; }
    else if (t.id === "vo-shading") o.shading = t.value;
    else if (t.id === "vo-sides") o.sides = t.value;
    else if (t.id === "vo-quality") { d.mesh = { quality: MQ.presetOf(t.value) }; remeshVolume(d); return; }
    else if (/^vo-mk-/.test(t.id)) {
      const k = t.id.slice(6), mesh = meshOf(d);
      mesh[k] = t.value === "" ? null : Number(t.value);
      if (k === "smoothness") { draw(); return; }
      remeshVolume(d); return;
    }
    else if (t.id === "vo-atoms") o.atoms = t.checked;
    else if (t.id === "vo-cell") { H.setSetting("showCell", t.checked); return; }      // the toolbar's Show Unit Cell: one setting
    else if (t.id === "vo-chem") { o.chem = t.checked; if (o.slice && d.slice) plotSlice(d).catch(() => {}); }
    else if (t.id === "vo-vmin" || t.id === "vo-vmax") { const x = Number(t.value); if (Number.isFinite(x)) o[t.id === "vo-vmin" ? "vmin" : "vmax"] = x; }
    else if (t.id === "vo-contours") { o.contours = Math.max(2, Math.min(60, Number(t.value) || 12)); if (o.slice) loadSlice(d).catch((er) => toast(er.message, "error")); return; }
    else if (/^vo-i[012]$/.test(t.id) && planeMode(o)) {
      const ix = [0, 1, 2].map((k) => Math.max(-SW.PLANE_MAX_INDEX, Math.min(SW.PLANE_MAX_INDEX, Math.round(Number($(`#vo-i${k}`, optsHost).value) || 0))));
      if (!ix.some((x) => x)) { toast("(0 0 0) is not a plane: give at least one nonzero index", "warning"); renderOpts(); return; }
      o[o.axis] = ix; o.tilt = 0; o.twist = 0; d.geom = null;
      renderOpts();
      if (o.slice) planeMoved(d, true); else { d.geom = sliceGeom(d); if (preview) preview.draw(); }
      return;
    }
    else if (t.id === "vo-slice" || t.id === "vo-axis") {
      if (t.id === "vo-slice") o.slice = t.checked; else { o.axis = t.value; o.index = null; d.axisPos = undefined; }
      d.geom = null;
      renderOpts();
      if (o.slice) loadSlice(d).catch((er) => toast(er.message, "error")); else { $(".vv-slice").hidden = true; d.slice = null; draw(); }
      return;
    } else return;
    draw();
  });
  optsHost.addEventListener("toggle", (e) => { if (doc && e.target.classList && e.target.classList.contains("vo-adv")) doc.advOpen = e.target.open; }, true);
  optsHost.addEventListener("click", (e) => {
    const d = doc;
    if (!d) return;
    const r = e.target.closest("[data-vo-range]");
    if (r) { d.opts.range = r.dataset.voRange; if (d.opts.range === "auto") autoRange(d); renderOpts(); return; }
    const pl = e.target.closest("[data-vo-plane]");
    if (pl) {
      // a common plane: (hkl) at the middle of the box, square
      const [md, ix] = pl.dataset.voPlane.split(":");
      Object.assign(d.opts, { axis: md, [md]: ix.split(",").map(Number), pos: 0.5, tilt: 0, twist: 0, slice: true });
      d.geom = null;
      renderOpts();
      planeMoved(d, true);
      return;
    }
    const b = e.target.closest("[data-vo]");
    if (b && b.dataset.vo === "planeSquare") { d.opts.tilt = 0; d.opts.twist = 0; d.geom = null; renderOpts(); planeMoved(d, true); return; }
    if (b && b.dataset.vo === "resetRange") { d.opts.range = "auto"; autoRange(d); renderOpts(); }
    if (b && b.dataset.vo === "orbLevel" && d.meta) { d.opts.level = d.meta.level; renderOpts(); draw(); }
    if (b && b.dataset.vo === "cli" && !d.combo) H.renderWithCli && H.renderWithCli(d.path);
  });
  host.addEventListener("click", (e) => {
    const b = e.target.closest("[data-act]");
    if (!b || !doc) return;
    if (b.dataset.act === "mep") H.colorByPotential(doc);
    else if (b.dataset.act === "diff") H.difference(doc);
    else if (b.dataset.act === "save") H.saveResult(doc);
    else if (b.dataset.act === "savecube") saveCube(doc).catch((er) => toast(`Save Cube: ${er.message}`, "error"));
  });
  // 3Dmol keeps the pointer's mousedown: focus the view by hand, so its keys (the slice steps) reach it
  $("#viewer-volume").addEventListener("pointerdown", () => $("#viewer-volume").focus({ preventScroll: true }), true);
  host.addEventListener("keydown", (e) => {
    // arrow keys step the slice, as the CLI's slice browser does
    if (!doc || !doc.opts.slice || !e.target.closest(".viewer") || !["ArrowUp", "ArrowDown"].includes(e.key)) return;
    e.preventDefault();
    if (planeMode(doc.opts)) { sliceChange(doc, { pos: e.key === "ArrowUp" ? 0.01 : -0.01 }, true); return; }
    const n = doc.meta.full_shape["abc".indexOf(doc.opts.axis)];
    doc.opts.index = Math.max(0, Math.min(n - 1, (doc.opts.index ?? 0) + (e.key === "ArrowUp" ? 1 : -1)));
    renderOpts();
    loadSlice(doc).catch((er) => toast(er.message, "error"));
  });
  /* Alt-drag in the 3D view moves the slice plane along its normal (Alt+Shift-drag turns it), so the plane can be
   * placed where it is seen; 3Dmol never gets the drag (its mousedown follows an unprevented pointerdown only). */
  $("#viewer-volume").addEventListener("pointerdown", (e) => {
    const d = doc;
    if (!e.altKey || !d || !d.vd || d.orbitals || !SW || !viewer) return;
    e.preventDefault(); e.stopPropagation();
    const g = sliceGeom(d);
    if (!g) return;
    const el = $("#viewer-volume"), turn = e.shiftKey || e.button === 2;
    let x = e.clientX, y = e.clientY;
    // the normal on screen (px per Å), from the view's own projection
    const a = viewer.modelToScreen({ x: g.point[0], y: g.point[1], z: g.point[2] });
    const b = viewer.modelToScreen({ x: g.point[0] + g.normal[0], y: g.point[1] + g.normal[1], z: g.point[2] + g.normal[2] });
    const sx = b.x - a.x, sy = b.y - a.y, len = Math.hypot(sx, sy), extent = Math.max(1e-9, g.dmax - g.dmin);
    el.setPointerCapture(e.pointerId);
    d.dragging = true;
    const move = (ev) => {
      const dx = ev.clientX - x, dy = ev.clientY - y;
      x = ev.clientX; y = ev.clientY;
      if (!dx && !dy) return;
      if (turn) sliceChange(d, { tilt: -dy * 0.5, twist: dx * 0.5 }, false);
      else sliceChange(d, { pos: len > 2 ? (dx * sx + dy * sy) / (len * len * extent) : -dy / Math.max(60, el.clientHeight) }, false);
    };
    const up = (ev) => {
      el.removeEventListener("pointermove", move); el.removeEventListener("pointerup", up); el.removeEventListener("pointercancel", up);
      try { el.releasePointerCapture(ev.pointerId); } catch (er) { /* gone */ }
      d.dragging = false;
      if (doc === d) sliceChange(d, {}, true);
    };
    el.addEventListener("pointermove", move); el.addEventListener("pointerup", up); el.addEventListener("pointercancel", up);
  }, true);

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
  async function setColor(d, path, keepRange = false) {
    const meta = await apiTask("volume?" + q({ path, ...gridQuery(d) }));
    const r = await api("volume/grid?" + q({ key: meta.grid_key }), { blob: true });
    d.colorVd = volumeData(meta, new Float32Array(await r.blob.arrayBuffer()));
    d.colorPath = path;
    d.colorMeta = meta;
    if (keepRange) return;                  // the same potential on a new mesh: its range stays
    d.opts.range = "auto";
    d.opts.vmin = d.opts.vmax = null;
    await autoRange(d);
  }

  /* ---------------------------------------------------------------- orbitals (CRYSTAL23 ORBITALS) */
  async function ensurePicker() {
    if (!picker) {
      const mod = await import("./orbitals.js");
      // the picker's Cells follow the toolbar's display range (one control, two views of it)
      picker = mod.createOrbitalPicker($(".vv-orb"), { ...H, initialCells, setCells: (c) => H.setRange(c) }, (d, how) => { if (doc === d) reloadOrbital(d, how).catch(() => {}); });
    }
    return picker;
  }
  const rangeCells = () => cellsOf((H.viewOpts && H.viewOpts().range) || [1, 1, 1]);
  /** The cells of an orbital's first request: the document's display range, else the file's default (written to the range). */
  function initialCells(ix) {
    const per = ix.periodic_axes || [];
    if (ix.molecule || !per.length) return null;
    const raw = rangeCells(), isDefault = raw.every((x) => x === 1) && !(H.ownRange && H.ownRange());
    const cells = (isDefault ? ix.default_cells || [1, 1, 1] : raw).map((x, k) => (per.includes(k) ? Math.max(1, Math.min(ix.max_cells || 4, x)) : 1));
    if (isDefault && cells.some((x) => x !== 1)) H.setRange(cells);
    return cells;
  }
  /** The orbital grid's cells for the display range (clamped to what the server draws); the range itself is brought back when it asked for more. */
  function orbitalCells(d) {
    const ix = d.orbitals.index, per = ix.periodic_axes || [], max = ix.max_cells || 4, raw = rangeCells();
    const kept = raw.map((x, k) => (per.includes(k) ? Math.min(x, max) : x));
    return { cells: raw.map((x, k) => (per.includes(k) ? Math.min(x, max) : 1)), kept, capped: kept.some((x, k) => x !== raw[k]), max };
  }
  /** A new pick: the grid of the request, the level back to its default, the camera kept unless the box changed. */
  async function reloadOrbital(d, { refit = false } = {}) {
    const seq = (d.orbitals.seq = (d.orbitals.seq || 0) + 1);
    d.orbitals.loading = true;
    picker.status("Evaluating the orbital…");
    $(".vv-view").setAttribute("aria-busy", "true");
    try {
      const keep = d.opts;
      d.vd = null;
      await loadVolume(d);
      if (doc !== d || d.orbitals.seq !== seq) return;
      $(".vv-msg").hidden = true;
      d.opts = { ...(keep || defaults(d.meta)), level: d.meta.level, minus: d.meta.signed ? (keep ? keep.minus || !keep.signedBefore : true) : false };
      d.opts.signedBefore = d.meta.signed;
      header(d);
      renderOpts();
      draw(refit);
      picker.status(`Showing ${d.meta.name.split(" · ").slice(1).join(" · ")}`);
    } catch (e) {
      if (doc !== d || d.orbitals.seq !== seq) return;      // a newer pick superseded this one
      picker.status(e.message);
      toast(e.message, "error");
      throw e;
    } finally { if (d.orbitals.seq === seq) { $(".vv-view").removeAttribute("aria-busy"); d.orbitals.loading = false; } }
  }
  async function saveCube(d) {
    if (!d || !d.orbitals || !d.orbitals.request) return;
    const qy = picker.query(d);
    const body = { path: d.path, k: qy.k, spin: qy.spin, mos: [qy.mo], part: qy.part, degenerate_sum: !!qy.sum,
      cells: qy.cells, quality: qy.quality };
    toast("Writing the cube…", "info");
    const r = await api("orbitals/save", { body });
    toast(r.cached ? `${r.path} is already saved` : `Saved ${r.path} (${(r.bytes / 1e6).toFixed(1)} MB)`, "success");
    if (r.warning) toast(`Cube folder ${r.folder}: ${r.warning}`, "warning");
    if (M().loadFiles) M().loadFiles().catch(() => {});
    return r;
  }
  function header(d) {
    const m = d.meta;
    if (d.orbitals) {
      $(".vv-title").textContent = m.name;
      $(".vv-chip").textContent = `${m.label} · ${m.full_shape.join("×")} points (${m.quality[0].toUpperCase()}${m.quality.slice(1)})`;
    } else {
      $(".vv-chip").textContent = `${m.label} · ${m.full_shape.join("×")} → ${m.shape.join("×")} shown (${m.quality[0].toUpperCase()}${m.quality.slice(1)})`;
    }
    $(".vv-chip").title = $(".vv-chip").textContent;
  }

  return {
    /** The element of each atom the view draws (its sites, in order), or null before the grid is read. */
    atomsOf(d) { if (!d || !d.meta || !G) return null; const u = unitStructure(d); return u ? u.symbols : (d.meta.atoms && d.meta.atoms.symbols) || null; },
    async open(d) {
      doc = d;
      $(".vv-title").textContent = d.title;
      $("[data-act=save]").hidden = !d.combo;
      const orb = !!d.orbitals;
      $("[data-act=savecube]").hidden = !orb;
      $("[data-act=mep]").hidden = orb;
      $("[data-act=diff]").hidden = orb;
      $(".vv-orb").hidden = !orb;
      if (!orb && picker) picker.close();
      const msg = $(".vv-msg");
      if (!d.vd) {
        msg.hidden = false; msg.innerHTML = `<span class="spinner"></span> ${orb ? "Reading the orbitals…" : "Reading the grid…"}`;
        try {
          await M().load3Dmol();
          if (!G) G = await import("../geom/core.js");
          if (!SW) SW = await import("./slice_widget.js");
          if (orb) {
            await ensurePicker(); if (doc !== d) return;
            await picker.open(d); if (doc !== d) return;
            msg.innerHTML = `<span class="spinner"></span> Evaluating the orbital…`;
            // a pick made while this first grid loads supersedes it (reloadOrbital draws that one)
            const seq = (d.orbitals.seq = (d.orbitals.seq || 0) + 1);
            try { await loadVolume(d); } catch (e) { if (d.orbitals.seq !== seq) return; throw e; }
            if (d.orbitals.seq !== seq) return;
          } else await loadVolume(d);
        } catch (e) { if (doc === d) msg.textContent = e.message; throw e; }
        if (!d.opts) { d.opts = defaults(d.meta); if (orb) d.opts.signedBefore = d.meta.signed; }
      } else if (orb) { await ensurePicker(); await picker.open(d); }
      if (!G) G = await import("../geom/core.js");
      if (!SW) SW = await import("./slice_widget.js");
      if (doc !== d) return;
      msg.hidden = true;
      const m = d.meta;
      header(d);
      // the picker beside the view changed its width: 3Dmol's canvas follows before the first fit
      if (orb && viewer) { try { viewer.resize(); } catch (e) { /* no canvas yet */ } }
      renderOpts();
      draw(true);
      if (d.opts.slice && !d.slice) await loadSlice(d);
      else if (d.opts.slice) await plotSlice(d);        // the slice map is shared by the volume tabs: this tab's plane
      else $(".vv-slice").hidden = true;
    },
    close() { doc = null; if (picker) picker.close(); },
    saveCube: () => (doc && doc.orbitals ? saveCube(doc) : null),
    setColor: (path) => (doc ? setColor(doc, path) : null),
    autoRange: () => (doc ? autoRange(doc) : null),
    /** The toolbar's display options changed (range, boundary atoms, cell, bonds, labels): redraw, camera kept.
     * An orbital's grid holds its cells, so a new range is a new grid (the picker's Cells follow the range). */
    redraw() {
      if (!doc || !doc.vd) return;
      const cb = $("#vo-cell", optsHost);
      if (cb) cb.checked = H.setting("showCell") !== false;
      if (doc.orbitals && picker && doc.orbitals.index && doc.orbitals.request && !doc.orbitals.index.molecule) {
        const w = orbitalCells(doc);
        if (w.capped) {
          toast(`Orbitals are drawn for up to ${w.max} cells along an axis`, "warning");
          H.setRange(w.kept);
          return;
        }
        if (w.cells.some((x, k) => x !== doc.orbitals.request.cells[k])) { picker.setCells(w.cells); return; }
      }
      if (doc.orbitals && doc.orbitals.loading) return;
      draw();
    },
    /** Reset View: the first view of the document (the options stay). */
    resetView: () => {
      if (!doc || !doc.vd) return;
      const w = v().getView();
      v().setView([...w.slice(0, 4), 0, 0, 0, 1]);          // zoomTo alone keeps the turn: back to looking down z
      draw(true);
    },
    /** The lattice the axes and View Along use: null for a molecule. */
    cell: () => { const m = doc && doc.meta; return m && m.periodic && m.lattice && G && G.reciprocal(m.lattice) ? m.lattice : null; },
    state: () => doc && doc.meta && { path: doc.path, type: doc.meta.type, level: doc.opts.level, plus: doc.opts.plus, minus: doc.opts.minus,
      surfaces: stats.surfaces, vertices: stats.vertices, shape: doc.meta.shape, stride: doc.meta.stride,
      mesh: { ...meshOf(doc), upsample: doc.meta.upsample || 1, quality: meshOf(doc).quality, grid: doc.meta.quality, loading: !!doc.loadingMesh },
      slice: doc.slice ? { axis: doc.slice.axis, index: doc.slice.index, mode: doc.slice.mode, ints: doc.slice.ints, pos: doc.slice.pos,
        tilt: doc.slice.tilt, twist: doc.slice.twist, polygon: doc.slice.polygon, min: doc.slice.min, max: doc.slice.max } : null,
      // the plane drawn now (the 3D view and the preview), its colors, what the map shows, the atoms' style
      plane: doc.opts.slice && doc.geom ? { mode: doc.geom.mode, polygon: doc.geom.polygon, normal: doc.geom.normal, pos: doc.geom.pos,
        shapes: planeShapes.length, colors: planeColors() } : null,
      map: doc.mapShown || null, atomStyle: lastDisplay ? lastDisplay.atomStyle || null : null,
      color: doc.colorPath || null, vmin: doc.opts.vmin, vmax: doc.opts.vmax, chem: chemOf(doc), stats: doc.meta.stats, combo: !!doc.combo,
      display: lastDisplay && { atoms: lastDisplay.list.n, images: lastDisplay.images, surfaces: lastDisplay.shifts.length, capped: lastDisplay.capped,
        cell: lastDisplay.cell, bonds: lastDisplay.bonds, labels: lastDisplay.labels, range: lastDisplay.range, box: lastDisplay.box },
      signed: !!doc.meta.signed, defaultLevel: doc.meta.level, atoms: doc.meta.atoms.numbers.length, periodic: !!doc.meta.periodic,
      orbital: doc.orbitals && doc.meta.orbital ? { ...doc.meta.orbital, request: doc.meta.request, name: doc.meta.name, busy: $(".vv-view").getAttribute("aria-busy") === "true",
        picker: picker ? picker.state() : null } : null },
    vertices: () => surfaceVertices(),
    // the part of the view the picture covers, for the tests
    drawnSpan: () => { if (!viewer) return null; viewer.render(); return drawnSpan(viewer); },
    gradientHex: (x) => (doc && doc.colorVd ? gradient(doc).valueToHex(x) : null),
    /** The potential color bar's model (null without one) and its drawing into an exported picture (shell.js exportCtx). */
    colorBar: () => barModel(doc),
    drawBar: (g, W, H, k) => drawBar(g, W, H, k),
    /** The [+, −] isosurface colors drawn (CSS colors), for the tests. */
    signColors: () => (doc && doc.meta ? colors(doc) : null),
    viewer: () => viewer,
    /** Move the slice as a drag in the preview would (the tests): {pos}, {tilt, twist}; final asks for the full grid. */
    slicePreview: (delta, final = true) => { if (doc && SW) sliceChange(doc, delta, final); },
    /** The preview's picture of the plane: the normal on the preview canvas (px per Å), for the tests. */
    previewNormal: () => (preview ? preview.normalOnScreen() : null),
  };
}
