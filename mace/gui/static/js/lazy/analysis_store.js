/* The Structure Analysis results of every document and their 3D overlays
 * (1.1.6), one store for the page: the Structure Analysis panel writes it,
 * the bottom panel's Analysis tab lists it, the Inspector's 3D Overlays
 * section edits its overlays, and Analyze Structures… (a batch over files
 * in the Explorer) adds to it.
 *
 * A record is one run: { id, tool, doc: {id, title, path, rev}, when,
 * params (as the form sent them), extra (a measured pattern), result,
 * summary (the CLI's summary row, mace analyze --csv), batch }. A batch
 * result belongs to its file (doc.id null, doc.path set): the document
 * of that file lists it when it is open. Records live as long as the page
 * (Structure Analysis results are never written to the folder). */
import { createOverlay } from "../viewport/pores.js";
import * as MQ from "../viewport/mesh_quality.js";

export const DOC = Symbol("doc");
// a document's runs of one tool kept (older ones go, with their overlay data)
const KEEP_PER_TOOL = 5;
export const TOOL_TITLE = { porosity: "Porosity", pxrd: "Powder pattern", peaks: "Peak search", index: "Indexing",
  refine: "Pattern fit", topology: "Topology", polyhedra: "Polyhedra", bonds: "Bonds & angles", morphology: "Morphology",
  planes: "Miller plane", cleavage: "Cleavage" };
// the tools Analyze Structures… runs on files (each needs a structure only)
export const BATCH_TOOLS = ["porosity", "pxrd", "topology", "polyhedra", "bonds", "morphology", "planes"];
// the overlay each tool draws
export const OVERLAY_OF = { porosity: "pores", topology: "net", morphology: "mesh", planes: "planes" };
export const OVERLAY_TITLE = { pores: "Pores", net: "Net", mesh: "Crystal shape", planes: "Miller planes" };
/* how pore surfaces are drawn by default: Settings › Viewport › pores.* */
export const PORE_KEYS = { accessible: ["pores.accessible", true], inaccessible: ["pores.inaccessible", true],
  opacity: ["pores.opacity", 0.5], style: ["pores.style", "solid"], colorPockets: ["pores.colorPockets", false],
  shading: ["pores.shading", "smooth"], sides: ["pores.sides", "auto"], quality: [MQ.PORE_KEY, MQ.PORE_DEFAULT] };
/* every overlay's mesh quality: Settings › Viewport › Mesh quality (1.1.6) */
export const MESH_KEYS = { quality: [MQ.KEY, MQ.DEFAULT] };

let one = null;
/** The page's store (made on first use). */
export function store(H) {
  if (!one) one = make(H);
  else if (H) one.H = H;
  return one;
}

const base = (p) => String(p || "").split("/").pop();
const num = (x, n = 3) => (x == null || Number.isNaN(Number(x)) ? "–" : Number(x).toFixed(n));
const hkl = (h) => (h ? `(${h.join(" ")})` : "–");
const csvCell = (v) => {
  const s = v == null ? "" : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};
export const csvOf = (head, rows) => [head, ...rows].map((r) => r.map(csvCell).join(",")).join("\n") + "\n";
const DIM = (d) => (d ? `${d}D` : "0D");

function make(H) {
  const st = { H, recs: [], seq: 0, batches: [], bseq: 0, overlay: null };
  const shell = () => st.H.shell();
  const emit = (what, detail = {}) => document.dispatchEvent(new CustomEvent("mace-analysis", { detail: { what, ...detail } }));

  /* ------------------------------------------------------------ overlays */
  function overlay() {
    if (st.overlay) return st.overlay;
    const S = shell();
    st.overlay = createOverlay({
      viewer: () => { const M = st.H.M(); return M && M.state && M.state.viewers ? M.state.viewers["viewer-main"] : null; },
      activeDocId: () => { const d = S.activeDoc(); return d ? d.id : null; },
      cssVar: (k) => { const M = st.H.M(); return M && M.vpVar ? M.vpVar(k) : getComputedStyle(document.documentElement).getPropertyValue(k).trim(); },
      // the edit revision of the document an overlay was computed for: an edit since hides it
      docRev: (id) => { const d = S.docById ? S.docById(id) : null; return d && S.editRev ? S.editRev(d) : null; },
      remesh: (it, spec) => remesh(it, spec),
      // the structure and the cell box of the display range (the N × M × L cells, the range presets): the overlays
      // of the cell are drawn in each of its cells
      display: (id) => {
        const d = S.docById ? S.docById(id) : null, s = d && st.H.structureOf ? st.H.structureOf(d) : null;
        const vo = S.viewOpts ? S.viewOpts() : null;
        return s && s.cell && vo ? { cell: s.cell, pbc: s.pbc || [true, true, true], box: vo.box } : null;
      },
      onError: (e) => { try { S.toast(`Pore surfaces: ${e.message || e}`, "error"); } catch (x) { /* no shell */ } },
    });
    st.overlay.onChange((why, it) => emit("overlay", { why, key: it ? it.key : null }));
    // Settings › Viewport › pores.* and Mesh quality (or another window's change): every overlay's defaults follow
    try {
      st.H.settings.onChange((k) => {
        if (String(k).startsWith("pores.")) st.overlay.setDefaults("pores", poreOpts());
        if (k === MQ.KEY) for (const kind of ["mesh", "planes", "net"]) st.overlay.setDefaults(kind, meshOpts());
      });
    } catch (e) { /* no live settings: the controls still work */ }
    dragHook();
    return st.overlay;
  }
  /* Settings › Viewport › "Drop the faint near-wall layer while dragging" (off by default): a drag in the
   * structure view hides the pores' near walls until the button is let go */
  function dragHook() {
    const el = document.getElementById("viewer-main");
    if (!el || el.dataset.maceDragNear) return;
    el.dataset.maceDragNear = "1";
    let at = null;
    const on = () => { try { return st.H.settings.get("pores.dropNearWhileDragging") === true; } catch (e) { return false; } };
    el.addEventListener("pointerdown", (e) => { at = e.button === 0 && on() ? [e.clientX, e.clientY] : null; });
    el.addEventListener("pointermove", (e) => {
      if (at && (e.buttons & 1) && Math.hypot(e.clientX - at[0], e.clientY - at[1]) > 3) { at = null; st.overlay.setDragging(true); }
    });
    window.addEventListener("pointerup", () => { at = null; if (st.overlay) st.overlay.setDragging(false); });
  }
  /** A pore result's surfaces at another mesh quality (POST analysis/remesh); its record keeps the newest. */
  async function remesh(it, spec) {
    const rec = it.meta && it.meta.rec ? get(it.meta.rec) : null;
    if (!rec) throw new Error("its result is gone");
    const S = shell(), d = docFor(rec);
    const source = rec.doc.id == null ? { path: rec.doc.path, frame: -1 } : d ? await S.sourceOf(d) : null;
    if (!source) throw new Error("its document is closed");
    const r = await S.apiTask("analysis/remesh", { source, params: rec.params, mesh: spec, doc: rec.doc.id, key: it.key });
    if (!r || r.canceled || !r.render) throw new Error("canceled");
    if (rec.result && rec.result.render) rec.result.render = { ...rec.result.render, ...r.render };
    return r.render;
  }
  function meshOpts() {
    const out = {};
    for (const [o, [k, d]] of Object.entries(MESH_KEYS)) {
      let v;
      try { v = st.H.settings.get(k); } catch (e) { v = undefined; }
      out[o] = v === undefined || v === null || typeof v !== typeof d ? d : v;
    }
    return out;
  }
  function poreOpts() {
    const out = meshOpts();
    for (const [o, [k, d]] of Object.entries(PORE_KEYS)) {
      let v;
      try { v = st.H.settings.get(k); } catch (e) { v = undefined; }
      out[o] = v === undefined || v === null || typeof v !== typeof d ? d : v;
    }
    return out;
  }
  /** The document a record's overlay is drawn on: its own, or the open document of its file (a batch result). */
  function docFor(rec) {
    const S = shell();
    if (rec.doc.id != null) return S.docById ? S.docById(rec.doc.id) : null;
    const a = S.activeDoc();
    if (a && rec.doc.path && a.path === rec.doc.path) return a;
    return null;
  }
  /** Draw a record's overlay (pores, net, crystal shape, planes) on its document; false when it has none or cannot. */
  function show(rec) {
    const kind = OVERLAY_OF[rec.tool], r = rec.result;
    if (!kind || !r) return false;
    const d = docFor(rec);
    const S = shell();
    const id = d ? d.id : rec.doc.id;
    const rev = rec.doc.id != null ? rec.doc.rev : (d && S.editRev ? S.editRev(d) : null);
    const meta = { rec: rec.id, label: overlayLabel(rec) };
    const O = overlay();
    if (kind === "pores") {
      if (!r.render) return false;
      rec.overlayKey = O.show("pores", r.render, poreOpts(), id, rev, meta);
    } else if (kind === "net") {
      if (!r.cell) return false;
      rec.overlayKey = O.show("net", { frac: r.net_frac, edges: r.net_edges, cell: r.cell }, meshOpts(), id, rev, meta);
    } else if (kind === "mesh") {
      const c = r.cell;
      if (!c) return false;
      const center = [0, 1, 2].map((k) => 0.5 * (c[0][k] + c[1][k] + c[2][k]));
      const size = 0.45 * Math.min(...c.map((v) => Math.hypot(...v)));
      rec.overlayKey = O.show("mesh", r, { center, scale: size, ...meshOpts() }, id, rev, meta);
    } else if (kind === "planes") {
      rec.overlayKey = O.show("planes", r.planes, meshOpts(), id, rev, meta);
    }
    return true;
  }
  /** The overlay item a record drew, if it is still there. */
  function overlayOf(rec) {
    const kind = OVERLAY_OF[rec.tool];
    if (!kind || !st.overlay) return null;
    const d = docFor(rec);
    // the key it was drawn under first: a batch result's document need not be in front
    const it = (rec.overlayKey && st.overlay.get(rec.overlayKey)) || st.overlay.find(kind, d ? d.id : rec.doc.id);
    return it && it.meta && it.meta.rec === rec.id ? it : null;
  }
  function overlayLabel(rec) {
    const r = rec.result || {};
    if (rec.tool === "porosity") return `${r.probe ? r.probe.name : "probe"}, r ${num(r.probe && r.probe.radius, 2)} Å`;
    if (rec.tool === "topology") {
      const t = (r.summary && r.summary.topologies || []).filter((x) => x && x !== "unknown");
      return t.length ? t.join(", ") : "not in the reference set";
    }
    if (rec.tool === "morphology") return String(r.method || "").toUpperCase();
    if (rec.tool === "planes") return hkl(r.hkl);
    return "";
  }

  /* ------------------------------------------------------------ records */
  function add(rec) {
    rec.id = `a${++st.seq}`;
    rec.when = rec.when || Date.now();
    if (rec.result && typeof rec.result === "object") rec.result[DOC] = rec.doc;
    st.recs.push(rec);
    // the oldest runs of this tool on this document go (a pore result holds its meshes)
    const same = st.recs.filter((x) => x.tool === rec.tool && x.doc.id === rec.doc.id && (rec.doc.id != null || x.doc.path === rec.doc.path));
    while (same.length > KEEP_PER_TOOL) remove(same.shift().id, true);
    emit("add", { id: rec.id });
    return rec;
  }
  function get(id) { return st.recs.find((r) => r.id === id) || null; }
  /** Deletes a record (and its overlay, if it drew the one shown). */
  function remove(id, quiet = false) {
    const i = st.recs.findIndex((r) => r.id === id);
    if (i < 0) return;
    const rec = st.recs[i];
    const it = overlayOf(rec);
    if (it) st.overlay.remove(it.key);
    st.recs.splice(i, 1);
    if (!quiet) emit("remove", { id });
  }
  /** Delete Mesh Data: a pore result's surfaces go from memory (its numbers stay). */
  function dropMeshes(rec) {
    const it = overlayOf(rec);
    if (it) st.overlay.remove(it.key);
    if (rec.result && rec.result.render) { delete rec.result.render; rec.result.render_deleted = true; }
    emit("meshes", { id: rec.id });
  }
  const ofDoc = (rec, d) => !!d && (rec.doc.id === d.id || (rec.doc.id == null && !!rec.doc.path && rec.doc.path === d.path));
  /** A document's records, newest first. */
  function forDoc(d) { return st.recs.filter((r) => ofDoc(r, d)).sort((a, b) => b.when - a.when); }
  /** The records of the open documents (and of no open document: batches), newest first. */
  function all() { return [...st.recs].sort((a, b) => b.when - a.when); }
  /** A document's newest result of each tool: { tool: result } (the panel's tabs). */
  function latest(d) {
    const out = {};
    for (const r of forDoc(d)) if (!(r.tool in out)) out[r.tool] = r.result;
    const fit = forDoc(d).find((r) => ["peaks", "index", "refine"].includes(r.tool));
    if (fit) out.fitLast = fit.tool;
    return out;
  }
  const recOf = (result) => st.recs.find((r) => r.result === result) || null;

  /* ------------------------------------------------------------ running */
  /** One run on the server (POST analysis/run, a background task): the
   * record, or null when it was canceled. ``doc``: the open document (its
   * structure as Export Structure sends it), or ``file``: a path in the
   * folder (a batch). ``onUpdate(task)`` on every task event. */
  async function run(tool, params, { doc = null, file = null, extra = {}, onUpdate = null, batch = null } = {}) {
    const S = shell();
    let source = null;
    if (file) source = { path: file, frame: -1 };
    else if (doc) source = await S.sourceOf(doc);
    const rev = doc && S.editRev ? S.editRev(doc) : null;
    // pore surfaces at Settings › Viewport › Pore surface mesh quality from the start (the overlay then asks for nothing)
    const sent = tool === "porosity" && params.render !== false && !params.mesh ? { ...params, mesh: { quality: poreOpts().quality } } : params;
    const body = { tool, source, params: sent, doc: doc ? doc.id : null, ...extra };
    if (file) body.doc = `file:${file}`;
    const r = await S.apiTask("analysis/run", body, onUpdate);
    if (!r || r.canceled) return null;
    if (r.result && typeof r.result === "object") Object.defineProperty(r.result, "cell", { value: r.cell, enumerable: false, writable: true });
    const where = doc ? { id: doc.id, title: doc.title, path: doc.path || null, rev } : { id: null, title: base(file), path: file, rev: null };
    return add({ tool, doc: where, params: { ...params }, extra, result: r.result, summary: r.summary || null,
      structure: r.structure, batch });
  }

  /* ------------------------------------------------------------ describing a record */
  /** The settings a result was computed with: [[label, value]]. */
  function settings(rec) {
    const p = rec.params || {}, r = rec.result || {};
    const L = [];
    const add = (k, v) => { if (v !== undefined && v !== null && v !== "") L.push([k, String(v)]); };
    if (rec.tool === "porosity") {
      add("Probe", r.probe ? `${r.probe.name}, r = ${num(r.probe.radius)} Å` : p.probe);
      add("Atomic radii", r.radii ? String(r.radii).toUpperCase() : p.radii);
      add("Grid spacing", r.grid ? `${num(r.grid.spacing)} Å (${r.grid.shape.join("×")})` : p.spacing);
      add("Engine", `MACE native${r.Df_method === "voronoi" ? " (grid and Voronoi network)" : " (grid)"}${r.zeopp ? ` + Zeo++ (${r.zeopp.binary})` : ""}`);
    } else if (rec.tool === "pxrd") {
      add("Radiation", r.source ? `${r.source}, λ = ${num(r.wavelength, 5)} Å` : p.source);
      add("2θ range", `${p.min || 5}–${p.max || 90}°, step ${p.step || 0.02}°`);
      add("Peak shape", `U ${p.U || 0}, V ${p.V || 0}, W ${p.W || 0.01}, η ${p.eta || 0.5}`);
      add("B_iso", `${p.b_iso || 0} Å²`);
      if (p.size_nm) add("Crystallite size", `${p.size_nm} nm`);
      if (r.measured) add("Measured pattern", r.measured.name || "yes");
    } else if (["peaks", "index", "refine"].includes(rec.tool)) {
      add("Pattern", r.name || (rec.extra && (rec.extra.pattern_name || rec.extra.pattern_path)) || "");
      add("Radiation", p.source || "CuKa");
      if (rec.tool === "peaks") add("Prominence", p.prominence || 0.02);
      if (rec.tool === "index") add("Mode", p.mode === "structure" ? "with the structure's cell" : "from scratch");
      if (rec.tool === "refine") { add("Method", r.method || p.method); add("Background terms", p.background || 6); if (p.po) add("Preferred orientation", p.po); }
    } else if (rec.tool === "topology") {
      add("Simplification", r.mode || p.mode || "auto");
      add("Bond tolerance", p.tolerance || 1.15);
    } else if (rec.tool === "polyhedra" || rec.tool === "bonds") {
      add("Bond tolerance", p.tolerance || 1.15);
      if (rec.tool === "bonds") add("Bins", p.bins || 40);
    } else if (rec.tool === "morphology") {
      add("Model", (r.method || p.model || "bfdh").toUpperCase());
      if (p.energies) add("Surface energies", p.energies);
      add("Max index", p.max_index || 2);
    } else if (rec.tool === "planes") {
      add("(hkl)", hkl(r.hkl));
      add("Position", num(r.offset, 2));
      add("Atoms within", `${p.tol || 0.3} Å`);
    } else if (rec.tool === "cleavage") {
      add("Max index", p.max_index || 2);
    }
    return L;
  }
  /** The key numbers of a result, short: [[label, value]]. */
  function keys(rec) {
    const r = rec.result || {};
    const out = [];
    const add = (k, v) => out.push([k, v]);
    if (rec.tool === "porosity") {
      add("Di", `${num(r.Di, 2)} Å`); add("Df", `${num(r.Df, 2)} Å`);
      if (r.POAV) add("POAV", `${num(100 * r.POAV.fraction, 1)}%`);
      if (r.ASA) add("ASA", `${num(r.ASA.m2_g, 0)} m²/g`);
      add("Channels", `${r.n_channels}${r.channel_dimensionality ? ` (${r.channel_dimensionality}D)` : ""}`);
      add("Pockets", String(r.n_pockets));
    } else if (rec.tool === "pxrd") {
      const pk = (r.peaks || []).filter((p) => p.intensity >= 0.5);
      add("Peaks", String(pk.length));
      const top = [...pk].sort((a, b) => b.intensity - a.intensity)[0];
      if (top) add("Strongest", `${hkl(top.hkl)} at ${num(top.two_theta, 2)}°`);
      if (r.similarity) add("Pearson r", num(r.similarity.pearson, 3));
    } else if (rec.tool === "peaks") add("Peaks", String((r.peaks || []).length));
    else if (rec.tool === "index") {
      if (r.mode === "ab_initio") { const s = (r.solutions || [])[0]; add("Best", s ? `${s.system} ${s.centering}, M20 ${num(s.M20, 1)}` : "none"); }
      else add("Indexed", `${r.indexed}/${r.total}`);
    } else if (rec.tool === "refine") {
      const S = r.statistics || {};
      add("Rwp", `${num(100 * S.Rwp, 2)}%`); add("GOF", num(S.GOF, 2));
    } else if (rec.tool === "topology") {
      const s = r.summary || {};
      add("Net", (s.topologies || []).map((x) => (x === "unknown" ? "not in the reference set" : x)).join(", ") || "–");
      add("Interpenetration", s.interpenetration == null ? "n/a" : String(s.interpenetration));
    } else if (rec.tool === "polyhedra") {
      add("Polyhedra", String((r.polyhedra || []).length));
      const sh = (r.summary || []).map((s) => `${s.element} ${s.shape || `CN${s.cn}`}`);
      if (sh.length) add("Shapes", sh.slice(0, 3).join(", ") + (sh.length > 3 ? "…" : ""));
    } else if (rec.tool === "bonds") { add("Bonds", String(r.n_bonds)); add("Angles", String(r.n_angles)); }
    else if (rec.tool === "morphology") {
      add("Aspect", (r.aspect || []).map((x) => num(x, 2)).join(" : "));
      const f = [...(r.families || [])].sort((a, b) => b.area_fraction - a.area_fraction)[0];
      if (f) add("Largest facet", `${hkl(f.hkl)} ${num(100 * f.area_fraction, 1)}%`);
    } else if (rec.tool === "planes") {
      add("d", `${num(r.d, 4)} Å`);
      if (r.broken) add("Fewest bonds broken", `${num(r.broken.min_per_nm2, 2)}/nm²`);
    } else if (rec.tool === "cleavage") {
      const f = (r.families || [])[0];
      if (f) add("Best", `${hkl(f.hkl)} ${num(f.min_per_nm2, 2)}/nm²`);
    }
    return out;
  }
  /** The same run on the command line. */
  function cli(rec) {
    const p = rec.params || {}, r = rec.result || {};
    const file = rec.doc.path ? (/\s/.test(rec.doc.path) ? `"${rec.doc.path}"` : rec.doc.path) : "STRUCTURE.cif";
    const a = ["mace", "analyze"];
    const opt = (k, v, def) => { if (v !== undefined && v !== null && v !== "" && String(v) !== String(def)) a.push(k, String(v)); };
    if (rec.tool === "porosity") {
      a.push("porosity", file);
      opt("--probe", p.probe_radius !== undefined && p.probe_radius !== "" ? p.probe_radius : (p.probe || (r.probe && r.probe.name)), "N2");
      opt("--radii", p.radii, "ccdc"); opt("--spacing", p.spacing, "0.15");
      if (p.zeopp) a.push("--engine", "both");
    } else if (rec.tool === "pxrd") {
      a.push("pxrd", file);
      opt("--source", p.source, "CuKa");
      if ((p.min && String(p.min) !== "5") || (p.max && String(p.max) !== "90")) a.push("--range", String(p.min || 5), String(p.max || 90));
      opt("--step", p.step, "0.02"); opt("--b-iso", p.b_iso, "0"); opt("--U", p.U, "0"); opt("--V", p.V, "0"); opt("--W", p.W, "0.01");
      opt("--eta", p.eta, "0.5"); opt("--size-nm", p.size_nm, "");
    } else if (rec.tool === "peaks") {
      a.push("peaks", "PATTERN"); opt("--source", p.source, "CuKa"); opt("--prominence", p.prominence, "0.02");
    } else if (rec.tool === "index") {
      a.push("index", "PATTERN"); if (p.mode === "structure") a.push("--structure", file); opt("--source", p.source, "CuKa");
    } else if (rec.tool === "refine") {
      a.push("refine", "PATTERN", "--structure", file); opt("--method", p.method, "pawley"); opt("--source", p.source, "CuKa");
      opt("--background", p.background, "6"); if (p.positions) a.push("--positions");
    } else if (rec.tool === "topology") { a.push("topology", file); opt("--mode", p.mode, "auto"); opt("--tolerance", p.tolerance, "1.15"); }
    else if (rec.tool === "polyhedra") { a.push("polyhedra", file); opt("--tolerance", p.tolerance, "1.15"); }
    else if (rec.tool === "bonds") { a.push("bonds", file); opt("--tolerance", p.tolerance, "1.15"); opt("--bins", p.bins, "40"); }
    else if (rec.tool === "morphology") {
      a.push("morphology", file);
      if (p.model === "wulff" && p.energies) a.push("--energies", `"${p.energies}"`);
      opt("--max-index", p.max_index, "2");
    } else if (rec.tool === "planes") {
      a.push("planes", file, "--hkl", ...(r.hkl || String(p.hkl || "1 0 0").trim().split(/[\s,]+/)).map(String)); opt("--offset", p.offset, "0"); opt("--tol", p.tol, "0.3");
    } else if (rec.tool === "cleavage") { a.push("planes", file, "--cleavage"); opt("--max-index", p.max_index, "2"); }
    return a.join(" ");
  }
  /** The result's own table (the per-feature table of a porosity run, a peak list, ...): { head, rows, csvHead, csvRows, note }. */
  function table(rec) {
    const r = rec.result || {};
    if (rec.tool === "porosity") {
      const F = r.features || [];
      const head = ["#", "Type", "Dim.", "Volume (Å³)", "Surface (Å²)", "D<sub>i</sub> (Å)", "Center (Å)"];
      const rows = F.map((f) => [String(f.id + 1), f.accessible ? "Accessible channel" : "Inaccessible pocket",
        f.accessible ? DIM(f.dim) : "0D", num(f.occupiable_A3, 1), num(f.area_A2, 1), num(f.Di, 2),
        f.Di_center_cart ? f.Di_center_cart.map((x) => num(x, 2)).join(", ") : "–"]);
      const csvHead = ["feature", "type", "accessible", "dimensionality", "axes", "occupiable_volume_A3", "probe_center_volume_A3",
        "surface_area_A2", "Di_A", "Di_center_x", "Di_center_y", "Di_center_z", "centroid_x", "centroid_y", "centroid_z"];
      const csvRows = F.map((f) => [f.id + 1, f.kind, f.accessible, f.accessible ? f.dim : 0, (f.axes || []).map((x, i) => (x ? "abc"[i] : "")).join(""),
        f.occupiable_A3, f.volume_A3, f.area_A2, f.Di, ...(f.Di_center_cart || ["", "", ""]), ...(f.centroid_cart || ["", "", ""])]);
      return { head, rows, csvHead, csvRows, feature: F.map((f) => f.id) };
    }
    if (rec.tool === "pxrd") {
      const P = (r.peaks || []).filter((p) => p.intensity >= 0.5);
      return { head: ["2θ (°)", "d (Å)", "I", "hkl", "Multiplicity"], rows: P.map((p) => [num(p.two_theta), num(p.d, 4), num(p.intensity, 1), hkl(p.hkl), String(p.multiplicity)]),
        csvHead: ["two_theta", "d", "intensity", "h", "k", "l", "multiplicity"], csvRows: (r.peaks || []).map((p) => [p.two_theta, p.d, p.intensity, ...(p.hkl || []), p.multiplicity]) };
    }
    if (rec.tool === "peaks") {
      const P = r.peaks || [];
      return { head: ["2θ (°)", "d (Å)", "Height", "FWHM (°)", "η"], rows: P.map((p) => [num(p.two_theta), num(p.d, 4), num(p.height, 1), num(p.fwhm, 4), num(p.eta, 2)]),
        csvHead: ["two_theta", "d", "height", "fwhm", "eta"], csvRows: P.map((p) => [p.two_theta, p.d, p.height, p.fwhm, p.eta]) };
    }
    if (rec.tool === "index") {
      if (r.mode === "ab_initio") {
        const S = r.solutions || [];
        const h = ["System", "Centering", "a", "b", "c", "α", "β", "γ", "V (Å³)", "M20", "Indexed"];
        const rows = S.map((s) => [s.system, s.centering, num(s.lattice.a, 4), num(s.lattice.b, 4), num(s.lattice.c, 4), num(s.lattice.alpha, 2),
          num(s.lattice.beta, 2), num(s.lattice.gamma, 2), num(s.volume, 2), num(s.M20, 1), `${s.indexed}/${s.total}`]);
        return { head: h, rows, csvHead: h, csvRows: rows };
      }
      const Lr = r.lines || [];
      return { head: ["2θ obs (°)", "hkl", "2θ calc (°)", "Δ"], rows: Lr.map((l) => [num(l.two_theta), hkl(l.hkl), num(l.two_theta_calc), num(l.delta, 4)]),
        csvHead: ["two_theta", "h", "k", "l", "two_theta_calc", "delta"], csvRows: Lr.map((l) => [l.two_theta, ...(l.hkl || []), l.two_theta_calc, l.delta]) };
    }
    if (rec.tool === "refine") {
      const Lt = r.lattice || {}, St = r.statistics || {};
      const h = ["a (Å)", "b (Å)", "c (Å)", "α (°)", "β (°)", "γ (°)", "V (Å³)", "Rp (%)", "Rwp (%)", "GOF"];
      const row = [num(Lt.a, 5), num(Lt.b, 5), num(Lt.c, 5), num(Lt.alpha, 3), num(Lt.beta, 3), num(Lt.gamma, 3), num(Lt.volume, 3),
        num(100 * St.Rp, 2), num(100 * St.Rwp, 2), num(St.GOF, 3)];
      return { head: h, rows: [row], csvHead: ["two_theta", "observed", "calculated", "background", "difference"],
        csvRows: (r.two_theta || []).map((x, i) => [x, r.observed[i], r.calculated[i], r.background[i], r.difference[i]]) };
    }
    if (rec.tool === "topology") {
      const rows = [], csvRows = [];
      (r.nets || []).forEach((n, i) => (n.vertex_classes || []).forEach((c) => {
        rows.push([String(i + 1), n.topology || "–", String(c.count), String(c.cn), c.cs.join(" "), `{${c.point_symbol || ""}}`, c.labels.slice(0, 4).join(", ")]);
        csvRows.push([i + 1, n.topology || "", n.periodicity, n.copies, c.count, c.cn, c.cs.join(" "), c.point_symbol || "", c.labels.join(" ")]);
      }));
      return { head: ["Net", "Topology", "Count", "CN", "Coordination sequence", "Point symbol", "Made of"], rows,
        csvHead: ["component", "topology", "periodicity", "copies", "count", "cn", "coordination_sequence", "point_symbol", "labels"], csvRows };
    }
    if (rec.tool === "polyhedra") {
      const P = r.polyhedra || [];
      return { head: ["Atom", "CN", "Shape", "CShM", "⟨d⟩ (Å)", "V (Å³)", "Δ", "⟨λ⟩", "σ² (°²)"],
        rows: P.map((p) => [`${p.element}${p.index + 1}`, String(p.cn), p.shape || "–", num(p.cshm, 2), num(p.bond_mean, 4), num(p.volume, 3),
          num(p.distortion_index, 5), num(p.quadratic_elongation, 4), num(p.angle_variance, 2)]),
        csvHead: ["atom", "element", "cn", "shape", "cshm", "bond_mean", "bond_min", "bond_max", "volume", "distortion_index", "quadratic_elongation", "angle_variance", "econ"],
        csvRows: P.map((p) => [p.index + 1, p.element, p.cn, p.shape || "", p.cshm, p.bond_mean, p.bond_min, p.bond_max, p.volume, p.distortion_index,
          p.quadratic_elongation, p.angle_variance, p.econ]) };
    }
    if (rec.tool === "bonds") {
      const rows = [], csvRows = [];
      Object.entries(r.bonds || {}).forEach(([k, v]) => { rows.push(["Bond", k, String(v.n), num(v.min, 4), num(v.mean, 4), num(v.max, 4), num(v.std, 4)]); csvRows.push(["bond", k, v.n, v.min, v.mean, v.max, v.std]); });
      Object.entries(r.angles || {}).forEach(([k, v]) => { rows.push(["Angle", k, String(v.n), num(v.min, 2), num(v.mean, 2), num(v.max, 2), num(v.std, 2)]); csvRows.push(["angle", k, v.n, v.min, v.mean, v.max, v.std]); });
      return { head: ["", "Pair", "n", "min", "mean", "max", "σ"], rows, csvHead: ["kind", "pair", "n", "min", "mean", "max", "std"], csvRows };
    }
    if (rec.tool === "morphology") {
      const F = r.families || [];
      return { head: ["(hkl)", "Faces", "Area %", "Distance"], rows: F.map((f) => [hkl(f.hkl), String(f.multiplicity), num(100 * f.area_fraction, 2), num(f.distance, 4)]),
        csvHead: ["h", "k", "l", "faces", "area_fraction", "distance"], csvRows: F.map((f) => [...f.hkl, f.multiplicity, f.area_fraction, f.distance]) };
    }
    if (rec.tool === "planes") {
      const b = r.broken || {};
      const A = r.atoms_on_plane || [];
      return { head: ["(hkl)", "d (Å)", "Position", "Planes in the cell", "Atoms on it", "Fewest bonds broken (/nm²)", "at position"],
        rows: [[hkl(r.hkl), num(r.d, 5), num(r.offset, 3), String((r.planes || []).length), String(A.length), num(b.min_per_nm2, 3), num(b.best_offset, 3)]],
        csvHead: ["offset", "bonds_broken_per_nm2"], csvRows: (b.offsets || []).map((x, i) => [x, b.profile[i]]) };
    }
    if (rec.tool === "cleavage") {
      const F = r.families || [];
      return { head: ["(hkl)", "d (Å)", "Bonds/nm²", "Best position"], rows: F.map((x) => [hkl(x.hkl), num(x.d, 4), num(x.min_per_nm2, 3), num(x.best_offset, 3)]),
        csvHead: ["h", "k", "l", "d", "min_bonds_per_nm2", "best_offset"], csvRows: F.map((x) => [...x.hkl, x.d, x.min_per_nm2, x.best_offset]) };
    }
    return { head: [], rows: [], csvHead: [], csvRows: [] };
  }
  /** A result as JSON (no 3D mesh data), with what it was computed on and with. */
  function json(rec) {
    const { render, ...rest } = rec.result || {};
    return JSON.stringify({ tool: rec.tool, structure: rec.doc.title, path: rec.doc.path, when: new Date(rec.when).toISOString(),
      settings: Object.fromEntries(settings(rec)), params: rec.params, command: cli(rec), result: rest }, null, 1);
  }
  /** A result's table as CSV, the settings as # lines above it. */
  function csv(rec) {
    const t = table(rec);
    const headLines = [`# ${TOOL_TITLE[rec.tool] || rec.tool} of ${rec.doc.title} (MACE ${new Date(rec.when).toISOString()})`,
      ...settings(rec).map(([k, v]) => `# ${k}: ${v}`)];
    return headLines.join("\n") + "\n" + csvOf(t.csvHead, t.csvRows);
  }

  /* ------------------------------------------------------------ batches (Analyze Structures…) */
  /** Runs ``tool`` with ``params`` on each file, ``jobs`` at a time; one task per file. */
  function batch(tool, params, files, jobs = 2) {
    const b = { id: `b${++st.bseq}`, tool, params: { ...params }, started: Date.now(), jobs,
      rows: files.map((f) => ({ path: f, title: base(f), state: "queued", rec: null, error: null, task: null, progress: 0 })) };
    st.batches.unshift(b);
    emit("batch", { id: b.id });
    let next = 0;
    const one = async () => {
      while (next < b.rows.length) {
        const row = b.rows[next++];
        if (row.state === "canceled") continue;
        row.state = "running";
        emit("batch", { id: b.id });
        try {
          const rec = await run(tool, params, { file: row.path, batch: b.id, onUpdate: (t) => {
            row.task = t.id; row.progress = t.progress || 0;
            if (t.state === "running" || t.state === "queued") { row.state = t.state; emit("batch", { id: b.id, quiet: true }); }
          } });
          if (rec) { row.rec = rec.id; row.state = "done"; row.summary = rec.summary; } else row.state = "canceled";
        } catch (e) {
          const canceled = e.state === "canceled" || !!(e.data && e.data.canceled);
          row.state = canceled ? "canceled" : "failed";
          row.error = canceled ? null : (e.message || String(e));
        }
        emit("batch", { id: b.id });
      }
    };
    b.done = Promise.all(Array.from({ length: Math.max(1, Math.min(jobs, files.length)) }, one)).then(() => { b.ended = Date.now(); emit("batch", { id: b.id }); });
    return b;
  }
  /** Cancel the files of a batch not done yet. */
  function cancelBatch(b) {
    const S = shell();
    for (const row of b.rows) {
      if (row.state === "queued") row.state = "canceled";
      else if (row.state === "running" && row.task) S.cancelTask(row.task);
    }
    emit("batch", { id: b.id });
  }
  /** A batch's summary (one row per file: the CLI's mace analyze --csv columns). */
  function batchTable(b) {
    const cols = [];
    for (const row of b.rows) for (const k of Object.keys(row.summary || {})) if (!cols.includes(k)) cols.push(k);
    if (!cols.includes("file")) cols.unshift("file");
    const rows = b.rows.map((row) => cols.map((c) => (c === "file" ? row.path : row.summary ? row.summary[c] : "")));
    const err = b.rows.some((r) => r.error || r.state !== "done");
    return { cols: err ? [...cols, "state", "error"] : cols,
      rows: err ? rows.map((r, i) => [...r, b.rows[i].state, b.rows[i].error || ""]) : rows };
  }

  const api = { get H() { return st.H; }, set H(v) { st.H = v; }, overlay, poreOpts, meshOpts, show, overlayOf, docFor, add, get, remove, dropMeshes,
    forDoc, all, latest, recOf, run, settings, keys, cli, table, json, csv, batch, cancelBatch, batchTable,
    batches: () => st.batches, records: () => st.recs };
  // the tests (and the console) read the results and overlays here
  window.__maceAnalysis = {
    get overlay() { return overlay(); },
    store: api,
    results: () => latest(shell().activeDoc()),
    resultsFor: (id) => latest({ id, path: (shell().docById && shell().docById(id) || {}).path }),
  };
  return api;
}
