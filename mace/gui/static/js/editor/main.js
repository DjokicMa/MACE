/* The viewport and the editor, loaded when the first structure document
 * opens (spec §8.5): nothing here is in the shell's first-paint bundle.
 * The shell hands over `S` (its API); this module gives back the editor's. */
import * as G from "../geom/core.js";
import * as W from "../geom/client.js";
import { createCamera, rotationPrefs } from "../viewport/camera.js";
import { createScene } from "../viewport/scene.js";
import { attachPointer } from "../viewport/pointer.js";
import { EditorDoc } from "./doc.js";
import { createEditing } from "./edit.js";
import { makeCommands } from "./commands.js";
import { createInspector } from "./inspector.js";
import { createTools } from "./tools.js";
import { createMeasure } from "./measure.js";
import { createAtomsTable } from "./atoms_table.js";
import { createHistoryPanel } from "./history_panel.js";
import { createSaving } from "./save.js";
import { createDrafts } from "./drafts.js";
import { loadCss } from "../lazy/lazy_util.js";

const cssReady = loadCss(import.meta.url, "../../css/editor.css");

export function init(S) {
  const M = window.MACE;
  const docs = new WeakMap();
  let v = null, cam = null, scene = null, ptr = null, current = null, interactingSince = 0, dprTimer = 0;
  const live = document.getElementById("sr-live");

  /* ---------------------------------------------------------------- documents */
  function attach(d, payload) {
    if (!docs.has(d)) docs.set(d, new EditorDoc(d, payload, S.setting));
    const ed = docs.get(d);
    if (scene) scene.adopt(ed);      // drawn a moment ago, before the document existed
    return ed;
  }
  const edOf = (d) => (d ? docs.get(d) || null : null);
  /** What the viewer shows for a document that is not a structure document
   * (a run's last step): a view-only stand-in with no selection. */
  function viewDoc(d, s) {
    const ed = edOf(d);
    if (ed) return ed;
    if (!current || current.structure !== s) current = { structure: s, selection: new Set(), measurements: [], viewOnly: true };
    return current;
  }

  /* ---------------------------------------------------------------- the viewer */
  function ensureViewer() {
    if (v) return;
    v = M.viewer("viewer-main");
    if (!v) return;                 // no WebGL: the viewport explains it (app.js), the rest works
    // the projection before the first fit: switching to orthographic after it
    // (the triad did, a frame later) shrank the first structure to a fraction
    // of the view, a fifth at 1920x1080
    if (v.setProjection) v.setProjection(S.setting("projection") === "perspective" ? "perspective" : "orthographic");
    cam = createCamera(v, { reducedMotion: S.reducedMotion, duration: () => S.reducedMotion() ? 0 : 200,
      rotation: () => rotationPrefs(S.setting) });
    scene = createScene(v, {
      M, setting: S.setting, opts: S.viewOpts, cam: () => cam, showMeasurements: () => S.measurementsShown(),
      setRadius: (r) => cam.setRadius(r),
      // the display range changed: an untouched default view is fitted again; a view the user turned
      // keeps its orientation and zooms out (and recenters) until the whole range shows
      refit: (s, disp, opts) => {
        if (v.visible && !v.visible()) return;
        const now = v.getView();
        if (fitted && fitted.structure === s && now.every((x, k) => Math.abs(x - fitted.view[k]) <= 1e-7)) fitDrawn(s, disp, opts);
        else fitInclude([], null);
        cam.invalidate();
        document.dispatchEvent(new CustomEvent("mace-camera", { detail: v }));
      },
      fit: (s, disp, opts) => {
        // the first structure is drawn while its viewer is still hidden (the
        // welcome screen goes a frame later): the fit waits for the viewer to show
        const doFit = () => fitDrawn(s, disp, opts);
        if (v.visible && !v.visible()) v.pendingFit = doFit;
        else { v.pendingFit = null; doFit(); }
        if (v.pendingFit) {
          const later = (n) => requestAnimationFrame(() => {
            if (!v.pendingFit) return;
            if (!v.visible()) { if (n < 120) later(n + 1); return; }
            const f = v.pendingFit;
            v.pendingFit = null;
            v.resize();
            f();
            cam.invalidate();
            document.dispatchEvent(new CustomEvent("mace-camera", { detail: v }));
          });
          later(0);
        }
      },
    });
    const el = document.getElementById("viewer-main");
    el.closest(".viewer-wrap").classList.add("vp-owned");
    watchFit();
    ptr = attachPointer(el, {
      scene, cam: () => cam, tool: S.tool, isMac: S.isMac, cssVar: M.cssVar,
      active: () => !!scene.doc,
      invertWheel: () => S.setting("invertWheelZoom") === true,
      zoomToward: () => S.setting("viewport.zoomToward"),
      pivot: (on) => { if (v.showPivot) v.showPivot(on); },
      focus: () => { const w = el.closest(".viewer-wrap"); if (w && document.activeElement !== w) w.focus({ preventScroll: true }); },
      select: (indices, mode) => { const d = scene.doc; if (d && !d.viewOnly) select(d, indices, mode); },
      selectOrbit: (i, mode) => { const d = scene.doc; if (d && !d.viewOnly) select(d, d.orbitOf(i), mode); },
      selectFragment: (i, mode) => { const d = scene.doc; if (d && !d.viewOnly) fragment(d, [i]).then((f) => select(d, f, mode)); },
      context: (k, x, y) => {
        const disp = scene.display;
        const atom = k >= 0 ? { i: disp.base[k], image: [disp.shift[3 * k], disp.shift[3 * k + 1], disp.shift[3 * k + 2]] } : null;
        // a right-click on an atom outside the selection selects it first (as in
        // a file manager), so Delete, Replace Element and Set Coordinates act on it
        const d = scene.doc;
        if (atom && d && !d.viewOnly && !d.selection.has(atom.i)) select(d, [atom.i], "replace");
        S.contextMenu(atom, x, y);
      },
      hover: (k, cart) => hover(k, cart),
      interacting,
      tools: S.tools || {},
    });
    document.addEventListener("mace-camera", () => scene && scene.invalidate());
    document.addEventListener("mace-theme", () => { if (scene && scene.doc) scene.draw(scene.doc, { keepView: true }); });
    document.addEventListener("mace-colors", () => { if (scene && scene.doc) scene.draw(scene.doc, { keepView: true }); });
  }
  /* The default view (carry-over a): the cell's standard orientation (c up,
   * a toward the lower left, b right; Settings › Viewport › Default view),
   * then a tight fit of what is drawn: every drawn atom (boundary atoms and
   * the Display Range's images too) at its drawn radius, and the corners of
   * the drawn cell outline, centered, filling 88% of the limiting side. */
  const FIT_FILL = 0.88;
  function fitDrawn(s, disp, opts) {
    const pbc = s.pbc || [true, true, true];
    const corners = G.boxCorners(s, opts.box || { lo: [0, 0, 0], hi: opts.range });
    // 3Dmol's zoomTo on a sample sets its clipping planes and a first zoom
    const pts = corners.slice(), step = Math.max(1, Math.floor(disp.n / 4000));
    for (let k = 0; k < disp.n; k += step) pts.push([disp.cart[3 * k], disp.cart[3 * k + 1], disp.cart[3 * k + 2]]);
    const m = v.addModel();
    m.addAtoms(pts.map((p, k) => ({ elem: "H", x: p[0], y: p[1], z: p[2], index: k, bonds: [] })));
    m.setStyle({}, {});
    v.zoomTo({ model: m });
    v.removeModel(m);
    const q = G.standardView(s.cell, pbc, S.setting("viewport.defaultView") || "standard");
    const view = v.getView();
    if (q) v.setView([...view.slice(0, 4), ...q]);
    else { v.setView([...view.slice(0, 4), 0, 0, 0, 1]); v.rotate(-75, "x"); v.rotate(-20, "y"); }   // a molecule: the old tilt
    const R = scene ? scene.drawnRadii : null, drawnCorners = opts.cell ? corners : [];
    // the rotation center (the view's center): the drawn cell's center, a slab's
    // at its mid-height, a molecule's atoms' middle; the zoom fits what is drawn around it
    const pivot = G.pivotOf(s, opts.box || { lo: [0, 0, 0], hi: opts.range }, disp.cart, disp.n);
    const now0 = v.getView();
    v.setView([-pivot[0], -pivot[1], -pivot[2], ...now0.slice(3)]);
    for (let pass = 0; pass < 3; pass++) {
      cam.invalidate();
      const c = cam.calibrate(), cv = v.getCanvas(), W = cv.clientWidth, Hh = cv.clientHeight;
      if (!(W > 0 && Hh > 0)) break;
      const uv = G.projectAll(c.P, disp.cart), [cx, cy] = c.centerPx;
      let x0 = cx, x1 = cx, y0 = cy, y1 = cy;
      for (let k = 0; k < disp.n; k++) {
        const r = (R && R[k] ? R[k] : 0.5) * c.s, x = uv[2 * k], y = uv[2 * k + 1];
        if (x - r < x0) x0 = x - r; if (x + r > x1) x1 = x + r;
        if (y - r < y0) y0 = y - r; if (y + r > y1) y1 = y + r;
      }
      for (const p of drawnCorners) {
        const [x, y] = G.project(c.P, ...p);
        if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
      // the zoom that fills FIT_FILL of the limiting side, the drawing kept around the center
      const hw = Math.max(cx - x0, x1 - cx), hh = Math.max(cy - y0, y1 - cy);
      if (!(hw > 0) || !(hh > 0)) break;
      const f = FIT_FILL * Math.min(W / (2 * hw), Hh / (2 * hh));
      if (pass > 0 && Math.abs(f - 1) < 0.005) break;
      if (Number.isFinite(f) && f > 0) v.zoom(f);
    }
    cam.invalidate();
    v.render();
    const host = document.getElementById("viewer-main");
    // by structure: the document that adopts a view-only stand-in keeps its structure
    fitted = { view: v.getView(), w: host.clientWidth, h: host.clientHeight, structure: s };
  }
  /* A builder's preview that reaches past the view (a supercell's ghosts, its
   * new cell, a defect's supercell outline): zoom out and recenter, with the
   * orientation kept, so the structure and the whole preview show in the part
   * of the viewport the dialog leaves free (avoid: the dialog's rectangle).
   * Never zooms in. Returns the view before, or null when it already fit. */
  function fitInclude(extra, avoid) {
    if (!scene || !scene.display || !v || (v.visible && !v.visible())) return null;
    const disp = scene.display, R = scene.drawnRadii, before = v.getView(), host = document.getElementById("viewer-main");
    const hr = host.getBoundingClientRect();
    let moved = false;
    for (let pass = 0; pass < 4; pass++) {
      cam.invalidate();
      const c = cam.calibrate(), cv = v.getCanvas(), W = cv.clientWidth, Hh = cv.clientHeight;
      if (!(W > 0 && Hh > 0)) break;
      // the free region: left of a dialog that covers the viewport's right part
      let rx0 = 0, rx1 = W;
      // the dialog's rectangle is in screen px, W in the viewer's own (the UI scale zooms it)
      const hz = window.maceZoom.of(host);
      if (avoid && avoid.left > hr.left + 0.4 * W * hz && avoid.left < hr.right && avoid.bottom > hr.top) rx1 = (avoid.left - hr.left) / hz - 8;
      const uv = G.projectAll(c.P, disp.cart);
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (let k = 0; k < disp.n; k++) {
        const r = (R && R[k] ? R[k] : 0.5) * c.s, x = uv[2 * k], y = uv[2 * k + 1];
        if (x - r < x0) x0 = x - r; if (x + r > x1) x1 = x + r;
        if (y - r < y0) y0 = y - r; if (y + r > y1) y1 = y + r;
      }
      for (const p of extra) {
        const [x, y] = G.project(c.P, p[0], p[1], p[2]), r = 0.6 * c.s;
        if (x - r < x0) x0 = x - r; if (x + r > x1) x1 = x + r;
        if (y - r < y0) y0 = y - r; if (y + r > y1) y1 = y + r;
      }
      if (!(x1 > x0) || !(y1 > y0)) break;
      const rw = rx1 - rx0, f = Math.min(1, FIT_FILL * Math.min(rw / (x1 - x0), Hh / (y1 - y0)));
      const fits = x0 >= rx0 + 0.02 * rw && x1 <= rx1 - 0.02 * rw && y0 >= 0.02 * Hh && y1 <= 0.98 * Hh;
      if (fits && (pass === 0 || f > 0.995)) break;
      const dx = (x0 + x1) / 2 - (rx0 + rx1) / 2, dy = (y0 + y1) / 2 - Hh / 2;
      // zoom() scales about the canvas center: recenter there first, then shift into the free region
      const sx = (rx0 + rx1) / 2 - W / 2;
      const center = G.add3(c.center, G.add3(G.scale3(c.right, (dx + sx * (1 - 1 / f)) / c.s), G.scale3(c.up, -dy / c.s)));
      const now = v.getView();
      v.setView([-center[0], -center[1], -center[2], ...now.slice(3)]);
      if (f < 1) v.zoom(f);
      moved = true;
    }
    cam.invalidate();
    v.render();
    if (moved) document.dispatchEvent(new CustomEvent("mace-camera", { detail: v }));
    return moved ? before : null;
  }
  /* The default view follows the viewer's size until the user moves the
   * camera: the first structure is fitted before the bottom panel takes its
   * height, and a window resize keeps an untouched view filled. */
  let fitted = null, refitFrame = 0;
  function watchFit() {
    const el = document.getElementById("viewer-main");
    if (!el || !window.ResizeObserver) return;
    new ResizeObserver(() => {
      cancelAnimationFrame(refitFrame);
      refitFrame = requestAnimationFrame(() => {
        if (!fitted || !scene || !scene.doc || scene.doc.structure !== fitted.structure || !scene.display || !v.visible()) return;
        const w = el.clientWidth, h = el.clientHeight;
        if (!(w > 0 && h > 0) || (w === fitted.w && h === fitted.h)) return;
        const now = v.getView();
        if (now.some((x, k) => Math.abs(x - fitted.view[k]) > 1e-7)) { fitted = null; return; }   // the user moved it
        v.resize();
        fitDrawn(scene.doc.structure, scene.display, S.viewOpts());
        document.dispatchEvent(new CustomEvent("mace-camera", { detail: v }));
      });
    }).observe(el);
  }
  /* While the user orbits more than 20k atoms, draw at 1 device pixel per
   * CSS pixel, and go back 150 ms after the pointer stops (spec §8.2.4). */
  function interacting(on) {
    if (!scene || !scene.display || scene.display.n <= 20000 || (window.devicePixelRatio || 1) <= 1) return;
    clearTimeout(dprTimer);
    if (on && !interactingSince) {
      interactingSince = performance.now();
      Object.defineProperty(window, "devicePixelRatio", { value: 1, configurable: true });
      v.resize();
    } else if (!on) {
      dprTimer = setTimeout(() => { delete window.devicePixelRatio; interactingSince = 0; v.resize(); v.render(); }, 150);
    }
  }

  /** Draw the structure panel's document (app.js calls this for viewer-main). */
  function draw(d, s, keepView) {
    ensureViewer();
    if (!scene) return;
    const doc = viewDoc(d, s);
    if (doc.structure !== s && !doc.viewOnly) doc.structure = s;
    scene.draw(doc, { keepView });
    const wrap = document.getElementById("viewer-main").closest(".viewer-wrap");
    wrap.setAttribute("role", "application");
    wrap.setAttribute("tabindex", "0");
    wrap.setAttribute("aria-label", viewerLabel(d, s));
    S.toolChip();
  }

  /** The 3D view's text alternative: what is drawn, in words, and where the
   * atoms are listed (the bottom panel's Atoms tab, a table). */
  function viewerLabel(d, s) {
    const parts = [`3D structure viewer, ${d ? d.title : "structure"}`];
    const f = s.reduced_formula || s.formula;
    if (f) parts.push(f);
    const y = s.symmetry;
    if (y && y.available && y.international) parts.push(`space group ${y.international}${y.number ? ` (${y.number})` : ""}`);
    const L = s.lattice;
    if (L && Number.isFinite(+L.a)) parts.push(`a ${(+L.a).toFixed(3)} Å, b ${(+L.b).toFixed(3)} Å, c ${(+L.c).toFixed(3)} Å`);
    parts.push(`${s.numbers.length} atoms`);
    return parts.join(", ") + ". The Atoms tab in the bottom panel lists every atom.";
  }

  /* ---------------------------------------------------------------- selection */
  function select(d, indices, mode = "replace") {
    const sel = d.selection;
    const before = sel.size;
    if (mode === "replace") { sel.clear(); indices.forEach((i) => sel.add(i)); }
    else if (mode === "add") indices.forEach((i) => sel.add(i));
    else if (mode === "subtract") indices.forEach((i) => sel.delete(i));
    else if (mode === "toggle") indices.forEach((i) => (sel.has(i) ? sel.delete(i) : sel.add(i)));
    if (scene && scene.doc === d) scene.drawSelection(d);
    announce(d, before);
    S.selectionChanged(d);
  }
  function announce(d, before) {
    if (!live) return;
    const s = d.structure, n = d.selection.size;
    if (n === 1) {
      const i = [...d.selection][0], w = wyckoff(s, i) ? `, ${wyckoff(s, i)}` : "";
      live.textContent = `Selected ${s.symbols[i]}${i + 1}${w}`;
    } else live.textContent = n ? `${n} atoms selected` : before ? "Selection cleared" : "";
  }
  async function fragment(d, start) {
    const s = d.structure;
    const tol = Number(S.setting("bondTolerance")) || 1.15, mm = S.setting("bondMetalMetal") === true;
    const frac = Float64Array.from(s.frac.flat()), radii = Float32Array.from(s.radii), metal = Uint8Array.from(s.metal.map(Number));
    if (d.n <= 2000) {
      const adj = G.adjacency(G.periodicBonds(s.cell, frac, radii, metal, tol, mm), d.n);
      return [...G.fragmentOf(adj, new Set(start))];
    }
    const r = await W.job("fragment", { cell: s.cell, frac, radii, metal, tol, metalMetal: mm, selection: start });
    return r.superseded ? [] : [...r.indices];
  }
  async function neighbors(d, start) {
    const s = d.structure;
    const tol = Number(S.setting("bondTolerance")) || 1.15, mm = S.setting("bondMetalMetal") === true;
    const frac = Float64Array.from(s.frac.flat()), radii = Float32Array.from(s.radii), metal = Uint8Array.from(s.metal.map(Number));
    if (d.n <= 2000) {
      const adj = G.adjacency(G.periodicBonds(s.cell, frac, radii, metal, tol, mm), d.n);
      return [...G.neighborsOf(adj, new Set(start))];
    }
    const r = await W.job("neighbors", { cell: s.cell, frac, radii, metal, tol, metalMetal: mm, selection: start });
    return r.superseded ? [] : [...r.indices];
  }

  /** "4a": the Wyckoff letter with its multiplicity in the conventional
   * cell (as the tables count it, whatever cell the structure is in). */
  function wyckoff(s, i) {
    const y = s.symmetry;
    if (!y || !y.available || !y.wyckoffs) return "";
    if (y.multiplicity && y.multiplicity.length > i) return `${y.multiplicity[i]}${y.wyckoffs[i]}`;
    const e = y.equivalent_atoms[i];
    return `${y.equivalent_atoms.filter((x) => x === e).length}${y.wyckoffs[i]}`;
  }

  /** Select Within Distance: atoms within r of each selected atom (or of their centroid). */
  async function selectWithin(d, r, centroid, periodic) {
    const s = d.structure;
    const args = { cell: s.cell, frac: Float64Array.from(s.frac.flat()), selection: [...d.selection], r, centroid, periodic };
    let out;
    if (d.n <= 2000) {
      const fr = (i) => s.frac[i];
      let centers = args.selection.map(fr);
      if (centroid && centers.length) centers = [centers.reduce((t, f) => G.add3(t, f), [0, 0, 0]).map((x) => x / centers.length)];
      out = new Set();
      for (const c of centers) for (let j = 0; j < d.n; j++) {
        const dist = periodic ? G.minImage(s.cell, c, fr(j)).length : G.norm3(G.cartOf(s.cell, G.sub3(fr(j), c)));
        if (dist <= r) out.add(j);
      }
      out = [...out];
    } else {
      const res = await W.job("within", args);
      if (res.superseded) return;
      out = [...res.indices];
    }
    select(d, out, "replace");
  }

  /* ---------------------------------------------------------------- hover readout */
  function hover(k, cart) {
    if (!scene) return;
    scene.setHover(k);
    const disp = scene.display, d = scene.doc;
    if (k < 0 || !disp || !d) { S.setCursor(""); M.state.hover = null; return; }
    const i = disp.base[k], s = d.structure, sh = [disp.shift[3 * k], disp.shift[3 * k + 1], disp.shift[3 * k + 2]];
    M.state.hover = { i, image: sh };
    const w = wyckoff(s, i) ? ` · ${wyckoff(s, i)}` : "";
    const f = s.frac[i].map((x, kk) => x + sh[kk]);
    const p = cart ? G.cartOf(s.cell, f) : f;
    S.setCursor(`${s.symbols[i]}${i + 1}${w} · x ${G.fmtNum(p[0], 4)} y ${G.fmtNum(p[1], 4)} z ${G.fmtNum(p[2], 4)}${cart ? " Å" : ""}`);
  }

  /* ---------------------------------------------------------------- views */
  function viewAlong(which) {
    const d = scene && scene.doc;
    if (!d) return null;
    const s = d.structure, dir = G.alongDirection(s.cell, which);
    return cam.along(dir, G.upVector(s.cell, dir));
  }
  /** View › Snap View to Nearest Axis: the nearest of a, b, c, a*, b*, c* (a periodic structure) and x, y, z. */
  function snapNearest() {
    const d = scene && scene.doc;
    if (!d || !cam) return null;
    const s = d.structure, periodic = s.cell && (!s.pbc || s.pbc.some(Boolean));
    return cam.snapNearest(periodic ? s.cell : null);
  }
  function frameSelection() {
    const d = scene && scene.doc, disp = scene && scene.display;
    if (!d || !disp) return;
    const pts = [];
    for (let k = 0; k < disp.n; k++) if (!d.selection.size || d.selection.has(disp.base[k])) pts.push([disp.cart[3 * k], disp.cart[3 * k + 1], disp.cart[3 * k + 2]]);
    cam.frame(pts, d.selection.size ? 2.5 : 1.6);
  }
  /** Turn about the drawn cell's center (what Reset View starts from), or the
   * middle of the selected atoms, from now on: the view slides there, its
   * zoom and orientation kept, and the center shows for a moment. */
  function pivotOn(what) {
    const d = scene && scene.doc, disp = scene && scene.display;
    if (!d || !disp || !cam) return null;
    let p;
    if (what === "selection") {
      const pts = [], home = [];
      for (let k = 0; k < disp.n; k++) {
        if (!d.selection.has(disp.base[k])) continue;
        const q = [disp.cart[3 * k], disp.cart[3 * k + 1], disp.cart[3 * k + 2]];
        pts.push(q);
        if (!disp.image[k]) home.push(q);
      }
      const use = home.length ? home : pts;
      if (!use.length) return null;
      p = use.reduce((t, q) => G.add3(t, q), [0, 0, 0]).map((x) => x / use.length);
    } else {
      const o = S.viewOpts();
      p = G.pivotOf(d.structure, o.box || { lo: [0, 0, 0], hi: o.range }, disp.cart, disp.n);
    }
    return cam.centerOn(p).then(() => { if (v.flashPivot) v.flashPivot(); });
  }

  /* ---------------------------------------------------------------- editing */
  const api = {};             // filled below: what edit.js and commands.js call back
  const editing = createEditing(api, S);
  /** The structure changed (an edit drawn here, or the server's answer). */
  function structureChanged(ed, opts = {}) {
    const d = ed.doc;
    d.data = ed.structure;
    let paintedNow = false;
    if (scene && scene.doc === ed) {
      M.state.structure = ed.structure;
      scene.draw(ed, { keepView: true });
      // the Structure card's formula and counts, with the file's kind (the payload says "edit")
      if (M.renderStructureInfo) M.renderStructureInfo(ed.structure, d.fileKind || (d.path ? "" : "new"));
      if (opts.local) {
        performance.mark("ui:edit:drawn");
        // an edit made here paints at once, in this task (see app.js paintNow)
        paintedNow = !!(v && v.paintNow && v.paintNow());
        if (!paintedNow) {
          // the edit's first paint (the bench reads ui:edit:start -> ui:edit:painted)
          const painted = (e) => {
            if (e.detail !== "viewer-main") return;
            document.removeEventListener("mace-painted", painted);
            performance.mark("ui:edit:painted");
            try { performance.measure("ui:edit", "ui:edit:start", "ui:edit:painted"); } catch (err) { /* no start mark */ }
          };
          document.addEventListener("mace-painted", painted);
        }
      }
    }
    S.docChanged(d, ed);
    if (paintedNow) {
      // measured to the end of the task, the panels included: the frame shows when it ends
      performance.mark("ui:edit:painted");
      try { performance.measure("ui:edit", "ui:edit:start", "ui:edit:painted"); } catch (err) { /* no start mark */ }
    }
  }
  function stateChanged(ed) { S.docChanged(ed.doc, ed); }
  Object.assign(api, { structureChanged, stateChanged, select, neighbors, frameSelection,
    tools: () => tools, pointer: () => ptr, cam: () => cam,
    edit: (ed, op, args, opts) => editing.edit(ed, op, args, opts),
    undo: (ed) => editing.undo(ed), redo: (ed) => editing.redo(ed), jump: (ed, i) => editing.jump(ed, i),
    setMode: (ed, m) => editing.setMode(ed, m), toSym: (ed) => S.toSym ? S.toSym(ed) : editing.setMode(ed, "sym"),
    nudgeStep: (ed) => (ed.nudgeStep !== undefined ? ed.nudgeStep : Number(S.setting(ed.coordMode === "cart" ? "nudgeStepCart" : "nudgeStepFrac"))),
    setSymprec: (ed, v) => editing.setSymprec(ed, v), wyckoff,
    /** Symmetrize (spglib refine, one undo step): its group becomes the reference. */
    symmetrize: (ed) => editing.edit(ed, "symmetrize", {}, { onDone: (p) => {
      const y = p && p.symmetry;
      if (y && y.available) ed.referenceGroup = { number: y.number, international: y.international, n_operations: y.n_operations_conventional || y.n_operations };
    } }) });
  const inspector = createInspector(api, S);
  // the edit tools: the pointer controller finds them in S.tools by tool name
  const tools = createTools(api, S, { ed: () => (scene && scene.doc && !scene.doc.viewOnly ? scene.doc : null),
    cam: () => cam, scene: () => scene, focusViewport: () => { const w = document.getElementById("viewer-main").closest(".viewer-wrap"); w.focus({ preventScroll: true }); } });
  const measure = createMeasure(api, S, { ed: () => (scene && scene.doc && !scene.doc.viewOnly ? scene.doc : null), scene: () => scene });
  Object.assign(S.tools, { move: tools.move, add: tools.add, replace: tools.replace, bond: tools.bond, measure: measure.tool });
  api.redrawMeasurements = (d) => { if (scene && scene.doc === d) scene.drawMeasurements(d); };
  api.measure = measure;
  const atomsTable = createAtomsTable(api, S), history = createHistoryPanel(api, S);
  api.editing = editing;
  const saving = createSaving(api, S), drafts = createDrafts(api, S);
  api.saving = saving;
  api.drafts = drafts;
  api.redraw = () => { if (scene && scene.doc) scene.draw(scene.doc, { keepView: true }); };
  const handlers = makeCommands(api, S);
  /** Run a command the editor implements, on a document (the active one by default). */
  function run(id, d, args) {
    const ed = edOf(d);
    const h = handlers[id];
    if (!h || !ed) return null;
    return h(ed, args);
  }

  /* ---------------------------------------------------------------- test hooks */
  const viewport = {
    pickAt: (x, y) => { const k = scene ? scene.pickAt(x, y) : -1; return k < 0 ? null : { k, i: scene.display.base[k] }; },
    lod: () => (scene ? scene.lod : 0), badge: () => (scene ? scene.badgeText() : ""),
    banner: () => (scene ? scene.bannerText() : ""), polyhedra: () => (scene ? scene.polyhedra() : null),
    modelIds: () => (scene ? Object.fromEntries(Object.entries(scene.layers).map(([k, m]) => [k, m ? m.getID() : null])) : {}),
    selectionShells: () => (scene && scene.layers.selection ? scene.layers.selection.selectedAtoms({}).length : 0),
    rings: () => (scene ? scene.rings() : []),
    displayed: () => (scene && scene.display ? scene.display.n : 0),
    labels: () => (scene ? scene.measureLabels() : []),
    camera: () => cam, scene: () => scene,
    /** The preview layer (Phase 4): ghost atoms, halos, the dashed cell, arrows. */
    models: () => (scene ? scene.previewInfo() : null),
    atomScreen: (i) => {
      const disp = scene && scene.display;
      if (!disp) return null;
      for (let k = 0; k < disp.n; k++) if (disp.base[k] === i && !disp.image[k]) {
        const [x, y] = cam.screenOf([disp.cart[3 * k], disp.cart[3 * k + 1], disp.cart[3 * k + 2]]);
        const r = v.getCanvas().getBoundingClientRect();
        return { x, y, cx: r.width / 2, cy: r.height / 2 };
      }
      return null;
    },
  };

  /* ---------------------------------------------------------------- bench hooks (tests/gui/bench) */
  const bench = {
    /** The displayed atoms' bond inputs, as the scene sends them to the worker. */
    bondInputs() {
      const d = scene && scene.display, doc = scene && scene.doc;
      if (!d || !doc) return null;
      const s = doc.structure, radii = new Float32Array(d.n), metal = new Uint8Array(d.n);
      for (let k = 0; k < d.n; k++) { radii[k] = s.radii[d.base[k]]; metal[k] = s.metal && s.metal[d.base[k]] ? 1 : 0; }
      return { cart: d.cart.slice(), radii, metal, tol: Number(S.setting("bondTolerance")) || s.bond_scale || 1.2,
        metalMetal: S.setting("bondMetalMetal") === true };
    },
    /** Worker round trips (post -> result) for the bonds of the displayed atoms. */
    async bonds(reps = 5) {
      const times = [];
      let pairs = 0, workerMs = [];
      for (let k = 0; k < reps; k++) {
        const a = bench.bondInputs(), t = performance.now();
        const r = await W.job("bonds", a, [a.cart.buffer]);
        times.push(performance.now() - t);
        workerMs.push(r.ms);
        pairs = r.pairs.length / 2;
      }
      return { times, pairs, workerMs };
    },
    /** Turn the camera every animation frame for `ms`; the frame-to-frame times. */
    orbit(ms = 3000) {
      return new Promise((resolve) => {
        const deltas = [], t0 = performance.now();
        let last = t0;
        const step = (t) => {
          deltas.push(t - last); last = t;
          v.rotate(1, "y");
          v.render();
          if (t - t0 < ms) requestAnimationFrame(step); else resolve(deltas.slice(1));
        };
        requestAnimationFrame(step);
      });
    },
  };

  return {
    bench,
    attach, edOf, draw, select, neighbors, fragment, viewAlong, snapNearest, frameSelection, pivotOn, viewport, wyckoff,
    run, has: (id) => !!handlers[id], editing, api, structureChanged, stateChanged, inspector, tools, selectWithin,
    chip: (tool, e) => (tool === "measure" ? `${e && e.waiting ? "Applying… · " : ""}Measure · ${measure.pending
      ? `${measure.pending} atom${measure.pending === 1 ? "" : "s"}: click the next` : "click 2, 3 or 4 atoms (distance, angle, dihedral)"}` +
      " · Shift+click chains distances · Esc to exit" : tools.chip(tool, e)),
    measure, atomsTable, history, saving, drafts,
    cam: () => cam, scene: () => scene, pointer: () => ptr, ensureViewer, cssReady, fitInclude,
    viewNow: () => (v ? v.getView() : null),
    setViewNow: (view) => { if (v && view) { v.setView(view); cam.invalidate(); v.render(); document.dispatchEvent(new CustomEvent("mace-camera", { detail: v })); } },
    redraw: () => { if (scene && scene.doc) scene.draw(scene.doc, { keepView: true }); },
    redrawSelection: (d) => { if (scene && scene.doc === d) scene.drawSelection(d); },
    redrawMeasurements: (d) => { if (scene && scene.doc === d) scene.drawMeasurements(d); },
    worker: W.stats,
  };
}
