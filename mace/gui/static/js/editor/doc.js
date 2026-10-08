/* One structure document's editor state (spec §1.2): the structure on
 * screen, its selection and measurements, the symmetry mode, the frac/Cart
 * choice, display-only bond overrides, selection sets, and (once edited)
 * the server session with its revision history.
 *
 * Two views of the history are kept (plan A2): `srv`, what the server last
 * confirmed (session, revision, its history rows), and `rows`/`pos`, what
 * the user sees, which runs ahead while requests are on their way. A row
 * holds the structure of its state while that is known here (`held`), and
 * what the user saw at it (`snap`: selection and measurements). */
import * as G from "../geom/core.js";

export const MAX_EDIT_ATOMS = 5000;

export class EditorDoc {
  constructor(shellDoc, payload, settings) {
    this.doc = shellDoc;
    this.path = shellDoc.path || null;
    this.frame = shellDoc.frame ?? -1;
    this.name = shellDoc.title;
    this.structure = payload;
    this.diskStamp = payload.file || null;
    this.srv = { session: payload.session || null, rev: payload.rev || 0, opsKey: payload.ops_key || null,
      structure: payload, lastGroup: null };
    const rows = payload.history && payload.history.length ? payload.history
      : [{ label: `Opened ${shellDoc.title}`, t: Date.now() / 1000, n_atoms: payload.numbers.length, rev: 0 }];
    this.rows = rows.map((r) => ({ ...r }));
    this.pos = payload.position ?? 0;
    this.rows[this.pos].held = payload;
    this.savedRev = this.rows[this.pos].rev;      // what is on disk (an untitled document: nothing)
    if (!this.path) this.savedRev = -1;
    this.selection = new Set();
    this.measurements = [];
    this.bondOverrides = new Map();     // "i-j:dx,dy,dz" -> "add" | "remove"
    this.bondMax = {};                  // the classic panel's per-pair length limits
    this.selectionSets = {};
    this.revSnap = new Map();           // rev -> {selection, measurements}
    const sym = payload.symmetry || {};
    // the group the status bar compares with: at open, or after Symmetrize (spec §1.7)
    // operations counted in the conventional cell, so a supercell or a primitive cell compares like for like
    this.referenceGroup = sym.available ? { number: sym.number, international: sym.international, n_operations: sym.n_operations_conventional || sym.n_operations } : null;
    const want = settings("symmetryModeDefault") === "p1" ? "p1" : "sym";
    this.symMode = payload.session ? (payload.locked ? "sym" : "p1")
      : want === "sym" && sym.available && (sym.n_operations_conventional || sym.n_operations) > 1 ? "sym" : "p1";
    this.symprec = Number(payload.symprec || settings("symprec")) || 1e-3;
    this.coordMode = settings("coordMode") === "cart" ? "cart" : "frac";
    this.ops = payload.ops || null;
    this.pending = 0;                   // requests on their way (the queue keeps the count)
    this.waiting = false;               // the structure on screen waits for the server ("Applying…")
    this.lost = false;                  // the connection to the server is gone (403)
    this.queue = null;
    this.predicted = false;             // the structure on screen was computed here, not yet confirmed
    this.previewing = null;             // a builder dialog's preview or a Clean proposal is open (plan B11)
  }
  get kind() { return this.doc.kind; }
  get n() { return this.structure.numbers.length; }
  get rev() { const r = this.rows[this.pos]; return r && r.rev !== null && r.rev !== undefined ? r.rev : null; }
  get dirty() { return this.rev === null || this.rev !== this.savedRev; }
  get canUndo() { return this.pos > 0; }
  get canRedo() { return this.pos < this.rows.length - 1; }
  get history() { return this.rows; }
  get position() { return this.pos; }
  /** Editing needs a 3D cell and at most 5000 atoms; the reason, or "". */
  editBlock() {
    const s = this.structure;
    if (this.lost) return "Lost the connection to the MACE server";
    if (this.previewing) return "Accept or discard the preview first";
    if (this.doc.held) return `Editing is limited to ${MAX_EDIT_ATOMS} atoms; this built structure is view-only`;
    if (!s.pbc || !s.pbc.every(Boolean)) {
      const dim = String(s.dimensionality || "").toLowerCase();
      return `Editing needs a 3D periodic cell (this is ${dim === "slab" ? "a slab" : dim === "polymer" ? "a polymer" : dim === "molecule" ? "a molecule" : "not 3D"})`;
    }
    if (this.n > MAX_EDIT_ATOMS) return `Editing is limited to ${MAX_EDIT_ATOMS} atoms`;
    return "";
  }
  /** What the user sees now: kept with the row (undo, redo and jump put it back). */
  snapNow() { return { selection: [...this.selection], measurements: this.measurements.map((m) => ({ ...m, atoms: m.atoms.map((a) => ({ ...a })) })) }; }
  restore(snap) {
    if (!snap) return false;
    this.selection = new Set(snap.selection.filter((i) => i < this.n));
    this.measurements = snap.measurements.map((m) => ({ ...m, atoms: m.atoms.map((a) => ({ ...a })) }));
    return true;
  }
  /** Remap selection and measurements after an edit (index_map: old -> new or null). */
  remap(indexMap) {
    if (indexMap === null) { this.selection.clear(); this.measurements.forEach((m) => { m.invalid = true; }); return; }
    if (!indexMap) return;
    const sel = new Set();
    for (const i of this.selection) { const j = indexMap[i]; if (j !== null && j !== undefined) sel.add(j); }
    this.selection = sel;
    for (const m of this.measurements) {
      for (const a of m.atoms) {
        const j = indexMap[a.i];
        if (j === null || j === undefined) m.invalid = true; else a.i = j;
      }
    }
  }
  /** Recompute every measurement on the current structure. */
  remeasure() {
    const s = this.structure;
    for (const m of this.measurements) {
      if (m.atoms.some((a) => a.i >= this.n)) m.invalid = true;
      if (!m.invalid) m.value = G.measure(s.cell, s.frac, m.atoms);
    }
  }
  /** The atoms equivalent to i: the session's operations in Sym mode,
   * else spglib's equivalent_atoms for the structure as it is. */
  orbitOf(i) {
    const s = this.structure, sym = s.symmetry || {};
    if (this.symMode === "sym" && this.ops) {
      const pts = G.orbitApply(this.ops, s.frac[i], s.cell, Math.max(2 * this.symprec, 1e-3));
      const out = new Set([i]);
      for (const p of pts) {
        const j = this.findAt(p, s.numbers[i]);
        if (j >= 0) out.add(j);
      }
      return [...out].sort((a, b) => a - b);
    }
    if (sym.available && sym.equivalent_atoms && sym.equivalent_atoms.length === this.n) {
      const e = sym.equivalent_atoms[i];
      return sym.equivalent_atoms.map((x, j) => (x === e ? j : -1)).filter((j) => j >= 0);
    }
    return [i];
  }
  /** The selection grown to whole orbits in Sym mode. */
  expand(indices) {
    if (this.symMode !== "sym") return [...new Set(indices)].sort((a, b) => a - b);
    const out = new Set();
    for (const i of indices) this.orbitOf(i).forEach((j) => out.add(j));
    return [...out].sort((a, b) => a - b);
  }
  /** The atom at a fractional position (any image), of this number; -1 for none. */
  findAt(f, number = null, tol = null) {
    const s = this.structure, t = tol ?? Math.max(2 * this.symprec, 1e-3);
    let best = -1, bestD = t;
    for (let j = 0; j < this.n; j++) {
      if (number !== null && s.numbers[j] !== number) continue;
      const d = G.minImage(s.cell, s.frac[j], f).length;
      if (d < bestD) { bestD = d; best = j; }
    }
    return best;
  }
}
