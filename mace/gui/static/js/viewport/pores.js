/* Analysis overlays in the 3D view (Analysis › Structure Analysis): pores
 * (accessible and inaccessible probe-occupiable volume as two meshes the
 * server builds: the channels cut at the cell faces, every closed pocket
 * whole), a crystal shape (morphology), Miller planes and a simplified net
 * (topology). The structure layer redraws with removeAllShapes, so every
 * overlay is kept here and drawn again after each "mace-scene" event, for
 * the document it was computed for only (not the one active when it
 * arrives): switching documents hides it, and an edit of that document
 * after the computation hides it too, so a surface never sits on a
 * structure it was not computed for.
 *
 * The overlays of the cell (pores, Miller planes, a net) are drawn in each
 * cell of the display range, as the atoms are (cell_tiles.js): built once
 * for the cell and moved to the others. The channels are cut at the cell
 * faces and each pocket is whole in the cell its center lies in, so the
 * copies join without a doubled face; the crystal shape is not periodic
 * and stays one. */

import { keepBuilt, skipDepthPass } from "./shape_cache.js";
import * as MQ from "./mesh_quality.js";
import { tile, tileShifts, farFaceTiles } from "./cell_tiles.js";

const KINDS = ["pores", "mesh", "planes", "net"];
// "Color pockets individually": --pore-pocket-1 ... (tokens.css; Okabe–Ito, color-blind safe)
const POCKET_TOKENS = 7;
// 3Dmol keeps a group's faces as 16-bit indices
const CHUNK = 60000;

function b64(s, Type) {
  const raw = atob(s);
  const buf = new ArrayBuffer(raw.length);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < raw.length; i++) u8[i] = raw.charCodeAt(i);
  return new Type(buf);
}

/* The wireframe's mesh: the vertices merged on a grid of ``h`` Å (vertex
 * clustering), so its lines are about that far apart instead of the
 * surface grid's few tenths of an Å, which drew as a solid. ``lab``: each
 * vertex's group (its feature and pocket), never merged across groups.
 * ``frame`` ({o, L, pbc}: the cell): a vertex on a periodic face of the
 * cell (a channel cut there) merges only with the vertices of that face,
 * on the grid seen from the opposite face, so the face's vertices stay on
 * it and meet the next cell's copy of the opposite face. */
function coarsen(P, N, F, lab, h, frame = null) {
  const key = new Map(), sum = [], nrm = [], cnt = [], map = new Int32Array(P.length / 3), group = [];
  const fr = frame ? frameOf(frame) : null;
  for (let v = 0; v < P.length / 3; v++) {
    let x = P[3 * v], y = P[3 * v + 1], z = P[3 * v + 2], side = "";
    if (fr) {
      const r = [x - fr.o[0], y - fr.o[1], z - fr.o[2]];
      for (let k = 0; k < 3; k++) {
        if (!fr.pbc[k]) continue;
        const f = r[0] * fr.Li[0][k] + r[1] * fr.Li[1][k] + r[2] * fr.Li[2][k];
        if (f < FACE) side += `${k}-`;
        else if (f > 1 - FACE) { side += `${k}+`; x -= fr.L[k][0]; y -= fr.L[k][1]; z -= fr.L[k][2]; }
      }
    }
    const k = `${Math.floor(x / h)},${Math.floor(y / h)},${Math.floor(z / h)},${lab[v]}${side}`;
    let i = key.get(k);
    if (i === undefined) { i = cnt.length; key.set(k, i); sum.push(0, 0, 0); nrm.push(0, 0, 0); cnt.push(0); group.push(lab[v]); }
    map[v] = i;
    cnt[i]++;
    for (let c = 0; c < 3; c++) { sum[3 * i + c] += P[3 * v + c]; nrm[3 * i + c] += N[3 * v + c]; }
  }
  const P2 = new Float32Array(sum.length), N2 = new Float32Array(nrm.length), F2 = [];
  for (let i = 0; i < cnt.length; i++) {
    const l = Math.hypot(nrm[3 * i], nrm[3 * i + 1], nrm[3 * i + 2]) || 1;
    for (let c = 0; c < 3; c++) { P2[3 * i + c] = sum[3 * i + c] / cnt[i]; N2[3 * i + c] = 127 * nrm[3 * i + c] / l; }
  }
  for (let f = 0; f < F.length; f += 3) {
    const a = map[F[f]], b = map[F[f + 1]], c = map[F[f + 2]];
    if (a !== b && b !== c && a !== c) F2.push(a, b, c);
  }
  return [P2, N2, Uint32Array.from(F2), Uint32Array.from(group)];
}
const NONE = 65535;              // a vertex of no feature (porosity.py's "feature" array)
const FACE = 1e-4;               // a vertex this close (fractional) to a cell face lies on it

/* The cell a pore mesh was made in: its origin, lattice rows (the render grid's voxels times its
 * intervals: porosity.py _render_fields) and their inverse (fractional = (r - o) L⁻¹). */
function frameOf({ o, L, pbc }) {
  const [a, b, c] = L;
  const det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
  const Li = [[b[1] * c[2] - b[2] * c[1], -(a[1] * c[2] - a[2] * c[1]), a[1] * b[2] - a[2] * b[1]],
    [-(b[0] * c[2] - b[2] * c[0]), a[0] * c[2] - a[2] * c[0], -(a[0] * b[2] - a[2] * b[0])],
    [b[0] * c[1] - b[1] * c[0], -(a[0] * c[1] - a[1] * c[0]), a[0] * b[1] - a[1] * b[0]]].map((r) => r.map((x) => x / (det || 1)));
  return { o, L, pbc, Li };
}
/** The cell of a pore result's render grid ({o, L}), or null. */
function gridCell(d) {
  if (!d || !d.shape || !d.voxels) return null;
  return { o: d.origin || [0, 0, 0], L: d.voxels.map((r, k) => r.map((x) => x * (d.shape[k] - 1))) };
}

/* A pore mesh's vertex groups: its feature (a channel or pocket of the
 * results' feature table, porosity.py) in the high 16 bits, its pocket
 * (for "Color pockets individually") in the low 16. */
function labels(m, nv) {
  const pk = m.pocket ? b64(m.pocket, Uint16Array) : null, ft = m.feature ? b64(m.feature, Uint16Array) : null;
  const lab = new Uint32Array(nv);
  for (let v = 0; v < nv; v++) lab[v] = ((ft ? ft[v] : NONE) * 65536 + (pk ? pk[v] : 0)) >>> 0;
  return lab;
}

/* A pore mesh (porosity.pore_meshes) cut into groups 3Dmol can index, each
 * one feature's (and pocket's) or part of one, decoded once per result and
 * way of drawing. The server's normals point out of the pore and its
 * triangles wind that way (``inward`` turns both around). */
const decoded = new WeakMap();
function chunks(m, inward, wire, step = MQ.WIRE_STEP.standard, flat = false, frame = null) {
  const fk = frame && wire ? `|${frame.pbc.map(Number).join("")}` : "";
  const memo = decoded.get(m) || {}, mk = `${inward}${wire}${wire ? step : ""}${flat}${fk}`;
  if (memo[mk]) return memo[mk];
  if (flat) {
    memo[mk] = MQ.flatParts(chunks(m, inward, wire, step, false, frame));
    decoded.set(m, memo);
    return memo[mk];
  }
  let P = b64(m.positions, Float32Array), N = b64(m.normals, Int8Array), F = b64(m.faces, Uint32Array);
  let lab = labels(m, P.length / 3);
  if (wire) [P, N, F, lab] = coarsen(P, N, F, lab, step, frame);
  const nv = P.length / 3, local = new Int32Array(nv).fill(-1), stamp = new Int32Array(nv).fill(-1);
  const order = inward ? [0, 2, 1] : [0, 1, 2], sn = inward ? -1 / 127 : 1 / 127;
  const out = [];
  let cur = null;
  const open = (g) => { cur = { vertexArr: [], normalArr: [], faceArr: [], group: g, pocket: g % 65536, feature: Math.floor(g / 65536) }; out.push(cur); };
  for (let f = 0; f < F.length; f += 3) {
    const g = lab[F[f]];
    if (!cur || cur.group !== g || cur.vertexArr.length > CHUNK - 3) open(g);
    const id = out.length - 1;
    for (const k of order) {
      const v = F[f + k];
      if (stamp[v] !== id) {
        stamp[v] = id;
        local[v] = cur.vertexArr.length;
        cur.vertexArr.push({ x: P[3 * v], y: P[3 * v + 1], z: P[3 * v + 2] });
        cur.normalArr.push({ x: sn * N[3 * v], y: sn * N[3 * v + 1], z: sn * N[3 * v + 2] });
      }
      cur.faceArr.push(local[v]);
    }
  }
  memo[mk] = out;
  decoded.set(m, memo);
  return out;
}

/** The corners of the box around a feature's vertices in a mesh (null: none). */
function featureBox(m, fid) {
  if (!m || !m.feature) return null;
  const P = b64(m.positions, Float32Array), ft = b64(m.feature, Uint16Array);
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  let n = 0;
  for (let v = 0; v < ft.length; v++) {
    if (ft[v] !== fid) continue;
    n++;
    for (let c = 0; c < 3; c++) { const x = P[3 * v + c]; if (x < lo[c]) lo[c] = x; if (x > hi[c]) hi[c] = x; }
  }
  if (!n) return null;
  const out = [];
  for (const a of [lo[0], hi[0]]) for (const b of [lo[1], hi[1]]) for (const c of [lo[2], hi[2]]) out.push([a, b, c]);
  return out;
}

// the near walls (between the eye and the pore) at this share of the opacity
const NEAR = 0.3;
// a highlighted feature: the others at this share of their opacity
const DIM = 0.35;

/* A mesh as 3Dmol shapes, a group per chunk, in its pocket's color when
 * ``palette`` is given. Solid: the far walls of each pore (seen from
 * inside it) at ``opacity``, and the near walls faintly, so the pore reads
 * as a volume while the atoms in front of it stay sharp (``lk.sides``:
 * "single" the far walls only, "double" both sides alike); Flat shading,
 * edges over it, a wireframe or the vertices as points (mesh_quality.js
 * look; ``step``: the wireframe's and points' spacing). Translucent
 * surfaces are drawn without writing depth, so the atoms and walls behind
 * them stay visible in any order 3Dmol draws them. ``hidden``: features
 * left out; ``hl``: a feature drawn in ``hlColor`` with the rest dimmed.
 * ``tiles``: the shifts of the display range's cells (each shape built once,
 * its copies moved there); ``frame``: the cell, for the wireframe's faces. */
function meshShapes(vw, m, { color, opacity, lk, step, palette, hidden, hl, hlColor, tiles = null, frame = null }) {
  const $3 = window.$3Dmol;
  skipDepthPass(vw);
  const flat = lk.shading === "flat";
  // [inward, opacity, side, how, pass]: how "fill" | "wire" | "points"
  const passes = lk.style === "wireframe" ? [[true, opacity, $3.DoubleSide, "wire"]]
    : lk.style === "points" ? [[true, Math.max(opacity, 0.85), $3.DoubleSide, "points"]]
    : lk.sides === "double" ? [[true, opacity, $3.DoubleSide, "fill"]]
    : lk.sides === "single" ? [[true, opacity, $3.FrontSide, "fill"]]
    : [[true, opacity, $3.FrontSide, "fill"], [false, opacity * NEAR, $3.FrontSide, "fill", "near"]];
  if (lk.style === "edges") passes.push([true, 1, $3.DoubleSide, "wire", "edges"]);
  const out = [];
  passes.forEach(([inward, op, side, how, pass]) => {
    let parts = how === "fill" ? chunks(m, inward, false, step, flat) : chunks(m, inward, true, step, false, frame);
    if (how === "points") parts = MQ.pointParts(parts, step / 4);
    parts = parts.filter((c) => !(hidden && hidden.has(c.feature)));
    const lit = hl == null ? [] : parts.filter((c) => c.feature === hl);
    const rest = hl == null ? parts : parts.filter((c) => c.feature !== hl);
    const buckets = [[rest, hl == null ? op : op * DIM, false], [lit, Math.min(1, Math.max(op, 0.6) * (pass === "near" ? NEAR / 0.6 : 1)), true]];
    for (const [list, o, isHl] of buckets) {
      if (!list.length) continue;
      const tint = (c) => (isHl && hlColor ? hlColor : palette ? palette[c.pocket % palette.length] : color);
      const spec = (c) => ({ vertexArr: c.vertexArr, normalArr: c.normalArr, faceArr: c.faceArr,
        color: pass === "edges" ? MQ.darker(tint(c)) : tint(c) });
      const s = vw.addCustom({ ...spec(list[0]), opacity: o, wireframe: how === "wire", side });
      for (let i = 1; i < list.length; i++) s.addCustom(spec(list[i]));
      // a shape color repaints every group with it: each group keeps its own
      if (palette && !isHl) s.color = undefined;
      keepBuilt(tile(s, tiles), (r, sh) => { if (sh.opacity < 1) (r.children || []).forEach((c) => { if (c.material) c.material.depthWrite = false; }); });
      if (pass) s.macePass = pass;
      if (isHl) s.maceHighlight = true;
      out.push(s);
    }
  });
  return out;
}

/* The overlays of every document: one per kind and document (pores, a
 * crystal shape, Miller planes, a net), each with the options it was shown
 * with (``base``: Settings › Viewport › pores.* for pores) and its own
 * changes on top (``over``: the Inspector's 3D Overlays section), drawn only
 * while its document is on screen and unedited since. */
export function createOverlay({ viewer, activeDocId, cssVar, docRev, remesh = null, onError = null, display = null }) {
  const items = new Map();     // "doc|kind" -> { key, kind, doc, rev, data, base, over, meta, shapes, visible, hidden, hl, t, pending, meshes, tiles }
  const listeners = new Set();
  const lastShown = {};        // kind -> key of the newest one shown
  let stamp = 0;
  const css = (k, d) => { try { return (cssVar && cssVar(k)) || d; } catch (e) { return d; } };
  const keyOf = (doc, kind) => `${doc == null ? "-" : doc}|${kind}`;
  const active = () => (activeDocId ? activeDocId() : null);
  const changed = (why, it) => listeners.forEach((f) => { try { f(why, it); } catch (e) { /* a listener's own trouble */ } });

  function v() { try { return viewer(); } catch (e) { return null; } }
  function pocketColors() {
    const c = Array.from({ length: POCKET_TOKENS }, (_, i) => css(`--pore-pocket-${i + 1}`, "")).filter(Boolean);
    return c.length ? c : null;
  }

  function drop(it) {
    if (!it) return;
    const vw = v();
    if (vw) it.shapes.forEach((s) => { try { vw.removeShape(s); } catch (e) { /* already gone with the scene */ } });
    it.shapes = [];
  }

  // the document it was made for is not the one shown, or was edited since
  function foreign(it) {
    if (activeDocId && active() !== it.doc) return true;
    return stale(it);
  }
  function stale(it) {
    if (it.rev != null && docRev) { try { return docRev(it.doc) !== it.rev; } catch (e) { return false; } }
    return false;
  }
  const optsOf = (it) => ({ ...(it.base || {}), ...(it.over || {}) });
  /* The cells of the display range an overlay of a document is drawn in (``display(doc)``: {cell, pbc, box}, the
   * structure and the cell box the structure view draws; none: the cell alone), its lattice ``cell`` (rows, Å). */
  function tilesOf(it, cell) {
    let dsp = null;
    try { dsp = display ? display(it.doc) : null; } catch (e) { dsp = null; }
    const pbc = (dsp && dsp.pbc) || [true, true, true];
    const C = cell === undefined ? dsp && dsp.cell : cell;
    return { ...tileShifts(dsp && C ? C : null, pbc, dsp && dsp.box), pbc, cell: C || null };
  }

  function draw(it, render = true) {
    const vw = v();
    if (!it || !vw) return;
    drop(it);
    if (!it.visible || foreign(it)) { if (render) vw.render(); return; }
    const kind = it.kind, d = it.data, o = optsOf(it);
    // the cells of the display range it is drawn in: the pores' own cell (their render grid), the net's, the structure's
    const gc = kind === "pores" ? gridCell(d) : null;
    const T = kind === "mesh" ? tilesOf(it, null) : tilesOf(it, kind === "pores" ? (gc ? gc.L : null) : kind === "net" ? d.cell : undefined);
    it.tiles = T;
    const addAt = (s, shifts) => { if (s) it.shapes.push(keepBuilt(tile(s, shifts))); };
    const add = (s) => addAt(s, T.shifts);
    const q = kind === "pores" ? MQ.porePresetOf(o.quality, MQ.PORE_DEFAULT) : MQ.presetOf(o.quality);
    if (kind === "pores") {
      ask(it, o);
      const op = Math.min(1, Math.max(0.05, Number(o.opacity ?? 0.5) || 0.5));
      const ms = d.meshes || {};
      const common = { opacity: op, lk: MQ.look(o, "auto"), step: MQ.WIRE_STEP[q], hidden: it.hidden, hl: it.hl,
        hlColor: o.hlColor || css("--pore-highlight", ""), tiles: T.shifts, frame: gc ? { ...gc, pbc: T.pbc } : null };
      if (o.accessible !== false && ms.accessible) {
        meshShapes(vw, ms.accessible, { ...common, color: o.accColor || css("--pore-acc", "") }).forEach(add);
      }
      if (o.inaccessible !== false && ms.inaccessible) {
        meshShapes(vw, ms.inaccessible, { ...common, color: o.nacColor || css("--pore-nac", ""),
          palette: o.colorPockets ? pocketColors() : null }).forEach(add);
      }
      if (dragging) it.shapes.forEach((sh) => { if (sh.macePass === "near") sh.hidden = true; });
    } else if (kind === "mesh") {
      const sc = o.scale || 1, c = o.center || [0, 0, 0];
      const vertexArr = [], normalArr = [], faceArr = [];
      d.triangles.forEach((t) => {
        const p = t.map((i) => d.vertices[i]);
        const e1 = p[1].map((x, k) => x - p[0][k]), e2 = p[2].map((x, k) => x - p[0][k]);
        const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        const l = Math.hypot(...n) || 1;
        p.forEach((q) => {
          faceArr.push(vertexArr.length);
          vertexArr.push({ x: c[0] + sc * q[0], y: c[1] + sc * q[1], z: c[2] + sc * q[2] });
          normalArr.push({ x: n[0] / l, y: n[1] / l, z: n[2] / l });
        });
      });
      MQ.styled(vw, [{ vertexArr, normalArr, faceArr }], MQ.look(o), { color: o.color || css("--accent", ""), opacity: o.opacity ?? 0.5,
        step: 0.01 }).forEach(add);
    } else if (kind === "planes") {
      const fr = T.cell ? frameOf({ o: [0, 0, 0], L: T.cell.length === 9 ? [T.cell.slice(0, 3), T.cell.slice(3, 6), T.cell.slice(6)] : T.cell, pbc: T.pbc }) : null;
      d.forEach((pl) => {
        const P = pl.polygon;
        if (!P || P.length < 3) return;
        // a plane on a far face of the cell is the next cell's plane on the near face: drawn once, in the last cell
        const frac = fr ? P.map((r) => [0, 1, 2].map((k) => r[0] * fr.Li[0][k] + r[1] * fr.Li[1][k] + r[2] * fr.Li[2][k])) : [];
        const shifts = farFaceTiles(T, frac, T.pbc, 1e-6).shifts;
        const c = [0, 1, 2].map((k) => P.reduce((s, q) => s + q[k], 0) / P.length);
        const vertexArr = [{ x: c[0], y: c[1], z: c[2] }], normalArr = [], faceArr = [];
        P.forEach((q) => vertexArr.push({ x: q[0], y: q[1], z: q[2] }));
        const e1 = P[1].map((x, k) => x - P[0][k]), e2 = P[2].map((x, k) => x - P[0][k]);
        const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        const l = Math.hypot(...n) || 1;
        vertexArr.forEach(() => normalArr.push({ x: n[0] / l, y: n[1] / l, z: n[2] / l }));
        for (let i = 1; i <= P.length; i++) faceArr.push(0, i, i === P.length ? 1 : i + 1);
        MQ.styled(vw, [{ vertexArr, normalArr, faceArr }], MQ.look(o), { color: o.color || css("--pore-surface", ""), opacity: o.opacity ?? 0.4,
          step: 0.01 }).forEach((s) => addAt(s, shifts));
      });
    } else if (kind === "net") {
      const cell = d.cell, f2c = (f) => [0, 1, 2].map((k) => f[0] * cell[0][k] + f[1] * cell[1][k] + f[2] * cell[2][k]);
      const pos = d.frac.map((f) => f2c(f));
      const col = o.color || css("--accent", ""), op = o.opacity ?? 1;
      pos.forEach((p) => add(vw.addSphere({ center: { x: p[0], y: p[1], z: p[2] }, radius: o.radius || 0.45, color: col, opacity: op,
        quality: MQ.SPHERE[q] })));
      d.edges.forEach(([u, w, t]) => {
        const a = pos[u], b = f2c(d.frac[w].map((x, k) => x + t[k]));
        add(vw.addCylinder({ start: { x: a[0], y: a[1], z: a[2] }, end: { x: b[0], y: b[1], z: b[2] }, radius: o.stick || 0.12,
          color: o.edgeColor || col, fromCap: 1, toCap: 1, opacity: op }));
      });
    }
    if (render) vw.render();
  }

  /* ---- mesh quality: a pore surface made again by the server at another one (only that overlay) */
  const specOf = (o) => ({ quality: MQ.porePresetOf(o.quality, MQ.PORE_DEFAULT), ...Object.fromEntries(Object.keys(MQ.PORE_LIMITS).map((k) => [k, o[k]])) });
  const madeAt = (d) => (d && d.meshes && d.meshes.quality) || { quality: MQ.DEFAULT };
  const specKey = (sp) => JSON.stringify(MQ.knobs("pores", sp));
  function keepMeshes(it, d) {
    // the surfaces made last (the one on screen newest), so going back (or an export's Publication and back)
    // needs no server
    it.meshes = it.meshes || new Map();
    const k = specKey(madeAt(d));
    it.meshes.delete(k);
    it.meshes.set(k, d);
    while (it.meshes.size > 3) it.meshes.delete(it.meshes.keys().next().value);
  }
  function ask(it, o) {
    const want = specOf(o);
    if (MQ.sameMesh(want, madeAt(it.data))) { it.pending = null; keepMeshes(it, it.data); return; }
    const kept = it.meshes && it.meshes.get(specKey(want));
    if (kept) { it.data = kept; it.pending = null; keepMeshes(it, kept); return; }
    if (!remesh || (it.pending && MQ.sameMesh(it.pending.spec, want)) || (it.failed && MQ.sameMesh(it.failed, want))) return;
    const p = { spec: want };
    it.pending = p;
    p.promise = Promise.resolve().then(() => remesh(it, want)).then((r) => {
      if (it.pending !== p || !items.has(it.key)) return;
      it.pending = null;
      if (r && r.meshes) {
        it.data = { ...it.data, ...r };
        keepMeshes(it, it.data);
      }
      draw(it);
      changed("meshed", it);
    }, (e) => {
      if (it.pending !== p) return;
      it.pending = null;
      it.failed = want;
      changed("remesh-error", it);
      if (onError) onError(e, it);
    });
    changed("remesh", it);
  }
  /** Resolves when an overlay's surfaces are at the quality asked (at once when no new mesh is coming). */
  function meshed(key) {
    const it = items.get(key);
    return it && it.pending ? it.pending.promise.then(() => meshed(key)) : Promise.resolve();
  }
  /** Runs ``fn`` with the active document's pore surfaces at ``quality`` (an export at Publication), then puts the
   * on-screen surfaces back (kept, so no second request). */
  async function withQuality(quality, fn) {
    const d = active(), its = [...items.values()].filter((it) => it.doc === d && it.kind === "pores" && it.visible && !foreign(it));
    const saved = its.map((it) => [it, it.over]);
    try {
      for (const it of its) { it.over = { ...it.over, quality }; draw(it); }
      await Promise.all(its.map((it) => meshed(it.key)));
      return await fn();
    } finally {
      for (const [it, over] of saved) { it.over = over; if (items.has(it.key)) draw(it); }
    }
  }

  /* ---- "drop the faint near-wall layer while dragging" (Settings › Viewport, off by default): fewer triangles a frame */
  let dragging = false;
  function setDragging(on) {
    if (dragging === !!on) return;
    dragging = !!on;
    let any = false;
    for (const it of items.values()) for (const s of it.shapes) if (s.macePass === "near") { s.hidden = dragging; any = true; }
    const vw = v();
    if (any && vw) vw.render();
  }

  /** Draw ``data`` for document ``doc`` (default: the active one) at its
   * edit revision ``rev`` (default: now); ``meta``: { label, rec } for the
   * Inspector's list. It replaces that document's overlay of this kind. */
  function show(kind, data, opts = {}, doc, rev, meta = {}) {
    if (!KINDS.includes(kind)) throw new Error(`overlay kind ${kind}`);
    const d = doc !== undefined ? doc : active();
    let r = rev;
    if (r === undefined && docRev) { try { r = docRev(d); } catch (e) { r = null; } }
    const key = keyOf(d, kind), old = items.get(key);
    if (old) drop(old);
    // a new result keeps what was changed for this document's overlay of this kind
    const it = { key, kind, doc: d, rev: r == null ? null : r, data, base: opts, over: old && old.data === data ? old.over : (old ? old.over : {}),
      meta, shapes: [], visible: true, hidden: new Set(), hl: null, t: ++stamp, pending: null, meshes: null };
    if (kind === "pores" && data) keepMeshes(it, data);
    items.set(key, it);
    lastShown[kind] = key;
    draw(it);
    changed("show", it);
    return key;
  }
  /** The options every overlay of a kind starts from (Settings › Viewport › pores.*): an overlay's own changes stay on top. */
  function setDefaults(kind, opts) {
    for (const it of items.values()) {
      if (it.kind !== kind) continue;
      it.base = { ...it.base, ...opts };
      if (!foreign(it)) draw(it);
    }
    changed("defaults", null);
  }
  /** One overlay's own options (``null`` takes a key back to the default). */
  function override(key, opts) {
    const it = items.get(key);
    if (!it) return;
    const o = { ...it.over };
    for (const [k, x] of Object.entries(opts)) { if (x === null || x === undefined) delete o[k]; else o[k] = x; }
    it.over = o;
    draw(it);
    changed("override", it);
  }
  function setVisible(key, on) {
    const it = items.get(key);
    if (!it) return;
    it.visible = !!on;
    draw(it);
    changed("visible", it);
  }
  /** A feature of a pore overlay shown or left out. */
  function setFeatureHidden(key, fid, hide) {
    const it = items.get(key);
    if (!it) return;
    if (hide) it.hidden.add(fid); else it.hidden.delete(fid);
    draw(it);
    changed("feature", it);
  }
  /** One feature drawn in the highlight color (--pore-highlight) with the rest dimmed (null: none); returns the corners of its box, to frame. */
  function highlight(key, fid) {
    const it = items.get(key);
    if (!it) return null;
    it.hl = fid == null ? null : fid;
    if (fid != null) { it.visible = true; it.hidden.delete(fid); }
    draw(it);
    changed("highlight", it);
    if (fid == null || it.kind !== "pores") return null;
    const ms = it.data.meshes || {};
    const boxes = [featureBox(ms.accessible, fid), featureBox(ms.inaccessible, fid)].filter(Boolean);
    return boxes.length ? boxes.flat() : null;
  }
  /** Takes an overlay and its data out of memory. */
  function remove(key) {
    const it = items.get(key);
    if (!it) return;
    drop(it);
    items.delete(key);
    it.data = null;
    const vw = v();
    if (vw) vw.render();
    changed("remove", it);
  }
  /** Clear 3D Overlays: the active document's (``kind``: of that kind only). */
  function hide(kind) {
    const d = active();
    for (const it of [...items.values()]) if (it.doc === d && (!kind || it.kind === kind)) remove(it.key);
  }
  const find = (kind, doc = active()) => items.get(keyOf(doc, kind)) || null;
  function has(kind, doc) { return !!find(kind, doc === undefined ? active() : doc); }
  /** The document of the newest overlay of a kind shown (null: none). */
  function docOf(kind) { const it = items.get(lastShown[kind]); return it ? it.doc : null; }
  /** Surfaces drawn now for a kind (the tests read it): a pore mesh's faint near walls are part of its surface. */
  function count(kind) {
    let n = 0;
    for (const it of items.values()) if (it.kind === kind) n += it.shapes.filter((s) => !s.macePass).length;
    return n;
  }
  /** The overlays of a document, for the Inspector: newest first. */
  function list(doc = active()) {
    return [...items.values()].filter((it) => it.doc === doc).sort((a, b) => b.t - a.t)
      .map((it) => ({ key: it.key, kind: it.kind, doc: it.doc, meta: it.meta, visible: it.visible, stale: stale(it),
        opts: optsOf(it), over: { ...it.over }, hidden: [...it.hidden], hl: it.hl, data: it.data, drawn: it.shapes.length,
        madeAt: it.kind === "pores" ? madeAt(it.data) : null, remeshing: !!it.pending, failed: it.failed || null,
        cells: it.tiles ? it.tiles.shifts.length : 1, cellsAsked: it.tiles ? it.tiles.total : 1 }));
  }
  function get(key) { return items.get(key) || null; }
  function onChange(f) { listeners.add(f); return () => listeners.delete(f); }

  // the structure layer redrew (it removed every shape): draw ours again
  // (a new display range: the Inspector says in how many cells)
  const cellsKey = () => [...items.values()].map((it) => (it.tiles ? `${it.tiles.shifts.length}/${it.tiles.total}` : "")).join();
  const onScene = () => {
    const was = cellsKey();
    for (const it of items.values()) { it.shapes = []; if (!foreign(it)) draw(it, false); }
    const vw = v();
    if (vw) vw.render();
    if (cellsKey() !== was) changed("cells", null);
  };
  const onSwitch = () => {
    for (const it of items.values()) { if (foreign(it)) drop(it); else draw(it, false); }
    const vw = v();
    if (vw) vw.render();
    changed("switch", null);
  };
  document.addEventListener("mace-scene", onScene);
  document.addEventListener("mace-doc-switch", onSwitch);
  function dispose() {
    for (const k of [...items.keys()]) remove(k);
    document.removeEventListener("mace-scene", onScene);
    document.removeEventListener("mace-doc-switch", onSwitch);
  }
  return { show, hide, remove, has, find, docOf, count, list, get, setDefaults, setOptions: setDefaults, override, setVisible,
    setFeatureHidden, highlight, onChange, dispose, redraw: onScene, meshed, withQuality, setDragging };
}
