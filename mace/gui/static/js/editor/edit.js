/* Editing a structure document (plan A1-A3a): each edit is applied here at
 * once when its result can be computed exactly (moves, replace, delete, in
 * P1 and on whole orbits in Sym), drawn, and sent through the document's
 * serial queue; the server's answer is adopted when it arrives, and wins if
 * it differs (a console warning that the tests keep at zero). Edits whose
 * result only the server knows (add, a new cell, an orbit that would grow)
 * wait for it with "Applying…" on the mode chip. Undo, redo and jump move
 * at once to a state held here, or wait for the server when it is not. */
import * as G from "../geom/core.js";
import { EditQueue } from "./queue.js";

const MERGE_MS = 1000;
export const stats = { reconcileWarnings: 0, predicted: 0, waited: 0 };

export function createEditing(E, S) {
  let groupSeq = 0;

  /* ---------------------------------------------------------------- queue hooks */
  const hooks = {
    changed: (ed) => E.stateChanged(ed),
    drained: (ed) => settle(ed),
    opened: (ed, data) => opened(ed, data),
    adopt: (ed, data, item) => adopt(ed, data, item),
    canceled: (ed) => revert(ed),
    lost: () => S.connectionLost(),
    conflict: (ed, data) => {
      adoptPayload(ed, data.payload);
      settle(ed, true);
      S.toast("This structure changed in another window or tab. MACE shows it as the server has it now.", "warning");
    },
    failed: (ed, item, why) => {
      revert(ed);
      S.toast(`That edit couldn't be applied: ${why}. The structure is back to the last saved step.`, "error");
    },
    recreated: (ed, data, why) => {
      ed.srv.session = data.session;
      ed.srv.rev = data.rev;
      ed.srv.opsKey = data.ops_key || null;
      if (data.ops) ed.ops = data.ops;
      ed.srv.lastGroup = null;
      ed.recreated = true;
      if (why === "evicted") S.toast("This document's undo history was released to save memory. Your structure is kept; earlier undo steps are gone.", "info");
      else S.toast(`${ed.name} changed on disk since you opened it. Your edits apply to the structure on screen; saving asks before it replaces the file.`, "warning");
    },
  };
  function queueFor(ed) {
    if (!ed.queue) ed.queue = new EditQueue(ed, hooks);
    return ed.queue;
  }
  const busyUnpredicted = (ed) => ed.waiting || (ed.queue && [ed.queue.inFlight, ...ed.queue.items].some((x) => x && x.blocks));

  /* ---------------------------------------------------------------- server state */
  function opened(ed, data) {
    ed.srv.session = data.session;
    if (ed.savedRev === 0 && data.opened_rev) {
      ed.savedRev = data.opened_rev;
      ed.rows[0].rev = data.opened_rev;
      if (ed.revSnap.has(0)) ed.revSnap.set(data.opened_rev, ed.revSnap.get(0));
    }
  }
  function adoptPayload(ed, p) {
    ed.srv.session = p.session;
    ed.srv.rev = p.rev;
    ed.srv.structure = p;
    ed.srv.history = p.history;
    ed.srv.position = p.position;
    if (p.ops) { ed.ops = p.ops; ed.srv.opsKey = p.ops_key; }
    if (!p.locked) { ed.ops = null; ed.srv.opsKey = null; }
  }
  function adopt(ed, data, item) {
    const p = item && item.payloadKey === false ? null : item && item.payloadKey ? data[item.payloadKey] : data;
    if (data.opened_rev) opened(ed, data);
    if (p && p.session) adoptPayload(ed, p);
    ed.srv.lastGroup = item && item.merge ? item.group : null;
    if (p && p.history_trimmed && !ed.trimToast) {
      ed.trimToast = true;
      S.toast("Oldest undo steps were dropped to stay within the memory limit", "info");
    }
    if (item && !item.predicted && p && p.index_map !== undefined && item.remap !== false) {
      // an edit drawn only now: its index map, after any other waiting one
      const prev = ed.pendingMap, next = p.index_map;
      ed.pendingMap = prev === undefined ? next : prev === null || next === null ? null
        : prev.map((j) => (j === null || j === undefined ? null : next[j] ?? null));
    }
    if (p && p.wrapped && item && item.kind !== "undo" && item.kind !== "redo") {
      const n = p.wrapped;
      S.toast(`Wrapped ${n} atom${n === 1 ? "" : "s"} into the cell`, "info", { label: "Undo", run: () => undo(ed) });
    }
    if (item && item.onDone) item.onDone(data);
  }
  /** Back to what the server last confirmed (a failure, a canceled confirmation). */
  function revert(ed) {
    const p = ed.srv.structure;
    ed.waiting = false;
    if (ed.queue) ed.queue.drop();
    rebuildRows(ed);
    ed.structure = p;
    ed.predicted = false;
    ed.selection = new Set([...ed.selection].filter((i) => i < ed.n));
    ed.remeasure();
    E.structureChanged(ed);
  }
  /** Every request answered: the server's state is the one shown. */
  function settle(ed, force = false) {
    if (!force && ed.queue && !ed.queue.idle) return;
    if (ed.lost) { ed.waiting = false; E.stateChanged(ed); return; }   // keep what is on screen: it is all there is
    const p = ed.srv.structure;
    if (!p || !p.history) { ed.waiting = false; E.stateChanged(ed); return; }
    if (ed.predicted && !ed.waiting && !same(ed.structure, p)) {
      stats.reconcileWarnings++;
      console.warn("mace: the server's structure differs from the one computed here; showing the server's");
    }
    const old = ed.rows;
    rebuildRows(ed, old);
    ed.structure = p;
    if (ed.pendingSnap) { ed.restore(ed.pendingSnap); ed.pendingSnap = null; ed.pendingMap = undefined; }
    else if (ed.pendingMap !== undefined) { ed.remap(ed.pendingMap); ed.pendingMap = undefined; }
    ed.predicted = false;
    ed.waiting = false;
    ed.selection = new Set([...ed.selection].filter((i) => i < ed.n));
    ed.remeasure();
    E.structureChanged(ed);
  }
  function rebuildRows(ed, old = null) {
    const p = ed.srv.structure;
    if (!p || !p.history) return;
    const rows = p.history.map((h, i) => {
      const prev = old && old.length === p.history.length ? old[i] : null;
      const snap = (prev && prev.snap) || ed.revSnap.get(h.rev) || null;
      if (snap) ed.revSnap.set(h.rev, snap);
      return { ...h, held: i === p.position ? p : null, snap };
    });
    ed.rows = rows;
    ed.pos = p.position;
  }
  function same(a, b) {
    if (a.numbers.length !== b.numbers.length) return false;
    for (let i = 0; i < a.numbers.length; i++) {
      if (a.numbers[i] !== b.numbers[i]) return false;
      for (let k = 0; k < 3; k++) {
        const d = a.frac[i][k] - b.frac[i][k];
        if (Math.abs(d - Math.round(d)) > 1e-6) return false;
      }
    }
    for (let r = 0; r < 3; r++) for (let k = 0; k < 3; k++) if (Math.abs(a.cell[r][k] - b.cell[r][k]) > 1e-6) return false;
    return true;
  }

  /* ---------------------------------------------------------------- requests */
  function opBody(ed, item, op, args, extra = {}) {
    const body = { op, args: { ...args, ...(item.extra || {}) }, ops_key: ed.srv.opsKey, ...extra };
    if (ed.srv.session) { body.session = ed.srv.session; body.base_rev = ed.srv.rev; }
    else body.open = openBlock(ed);
    if (item.merge) {
      body.merge = item.merge;
      body.merge_count = item.count;
      if (ed.srv.lastGroup === item.group) body.merge_into = ed.srv.rev;
    }
    return body;
  }
  function openBlock(ed) {
    return { path: ed.path, frame: ed.frame, locked: ed.symMode === "sym", symprec: ed.symprec,
      expected: ed.diskStamp, wrap: S.setting("wrapIntoCell") !== false };
  }
  /** A request that needs the session but makes no edit (Clean, Save, …): opens it first if needed. */
  function ensureSession(ed) {
    if (ed.srv.session || !ed.path) return Promise.resolve(ed.srv.session);
    if (ed.opening) return ed.opening;
    ed.opening = queueFor(ed).push({ kind: "open", route: "edit/open", predicted: true,
      make: (d) => ({ ...openBlock(d), path: d.path }),
      onDone: (data) => { opened(ed, { ...data, opened_rev: data.rev }); } }).then(() => { ed.opening = null; return ed.srv.session; });
    return ed.opening;
  }

  /* ---------------------------------------------------------------- local results */
  const wrapOn = (args) => (args.wrap === undefined ? S.setting("wrapIntoCell") !== false : !!args.wrap);
  function moved(ed, indices, dfFor, wrap) {
    const s = ed.structure, frac = s.frac.map((f) => f.slice());
    for (const [i, df] of indices.map((i) => [i, dfFor(i)])) for (let k = 0; k < 3; k++) frac[i][k] += df[k];
    // wrapped along the periodic directions only: a slab's c, a molecule's box are not a period (1.1.6)
    const per = s.pbc || [true, true, true];
    if (wrap) for (const i of indices) for (let k = 0; k < 3; k++) { if (!per[k]) continue; let x = frac[i][k] - Math.floor(frac[i][k] + 1e-12); if (Math.abs(x - 1) < 1e-12) x = 0; frac[i][k] = x; }
    return frac;
  }
  /** Where a Sym move takes each atom of the orbits of `picked`; null when an
   * orbit would grow (the server asks first) or the operations are unknown. */
  function symMove(ed, picked, dfOf) {
    if (!ed.ops) return null;
    const s = ed.structure, tol = Math.max(2 * ed.symprec, 1e-3), done = new Set(), frac = s.frac.map((f) => f.slice());
    for (const i of picked) {
      if (done.has(i)) continue;
      const orbit = ed.orbitOf(i);
      orbit.forEach((j) => done.add(j));
      const f0 = s.frac[i], df = dfOf(i, f0);
      const moved0 = f0.map((x, k) => x + df[k]);
      if (G.orbitApply(ed.ops, moved0, s.cell, tol).length !== orbit.length) return null;
      for (let k = 0; k < ed.ops.rotations.length; k++) {
        const R = ed.ops.rotations[k], t = ed.ops.translations[k];
        const img = [0, 1, 2].map((r) => R[r][0] * f0[0] + R[r][1] * f0[1] + R[r][2] * f0[2] + t[r]);
        const j = ed.findAt(img, s.numbers[i]);
        if (j < 0) continue;
        frac[j] = [0, 1, 2].map((r) => R[r][0] * moved0[0] + R[r][1] * moved0[1] + R[r][2] * moved0[2] + t[r]);
      }
    }
    return { frac, indices: [...done] };
  }
  function fracDelta(ed, args) {
    if (args.frac_delta) return args.frac_delta.map(Number);
    return G.fracOf(ed.structure.cell, args.delta.map(Number));
  }
  /** {structure, indexMap, label, n} for an edit computed here, or null. */
  function predict(ed, op, args) {
    const s = ed.structure;
    const make = (frac, numbers = s.numbers) => G.payloadFrom(s.cell, frac, numbers, s);
    if (op === "move") {
      const df = fracDelta(ed, args), wrap = wrapOn(args);
      if (ed.symMode !== "sym") return { structure: make(moved(ed, args.indices, () => df, wrap)), n: new Set(args.indices).size };
      const r = symMove(ed, args.indices, (i, f0) => (args.project ? G.projectOntoSite(ed.ops, f0, df, s.cell, Math.max(2 * ed.symprec, 1e-3)) : df));
      if (!r) return null;
      if (wrap) r.frac = r.frac.map((f, j) => (r.indices.includes(j) ? f.map((x) => { const y = x - Math.floor(x + 1e-12); return Math.abs(y - 1) < 1e-12 ? 0 : y; }) : f));
      return { structure: make(r.frac), n: r.indices.length };
    }
    if (op === "set_positions") {
      const target = args.frac ? args.frac : args.cart.map((x) => G.fracOf(s.cell, x));
      const byIdx = new Map(args.indices.map((i, k) => [i, target[k]]));
      if (ed.symMode !== "sym") return { structure: make(moved(ed, args.indices, (i) => byIdx.get(i).map((x, k) => x - s.frac[i][k]), wrapOn(args))), n: byIdx.size };
      const r = symMove(ed, args.indices, (i, f0) => byIdx.get(i).map((x, k) => x - f0[k]));
      return r ? { structure: make(r.frac), n: r.indices.length } : null;
    }
    if (op === "replace") {
      const z = G.zOf(args.element);
      if (!z) return null;
      const idx = ed.expand(args.indices), numbers = s.numbers.slice();
      idx.forEach((i) => { numbers[i] = z; });
      return { structure: make(s.frac.map((f) => f.slice()), numbers), n: idx.length, idx };
    }
    if (op === "delete") {
      const idx = new Set(ed.expand(args.indices)), map = [];
      let k = 0;
      for (let i = 0; i < s.numbers.length; i++) map.push(idx.has(i) ? null : k++);
      const keep = s.numbers.map((_, i) => i).filter((i) => !idx.has(i));
      return { structure: make(keep.map((i) => s.frac[i].slice()), keep.map((i) => s.numbers[i])), indexMap: map, n: idx.size };
    }
    return null;
  }
  /** The History row text before the server's arrives (the server's wins). */
  function labelFor(ed, op, args, pred) {
    const s = ed.structure, n = pred ? pred.n : (args.indices || []).length;
    const atoms = `${n} atom${n === 1 ? "" : "s"}`;
    if (op === "replace") {
      const idx = pred && pred.idx ? pred.idx : args.indices;
      const old = [...new Set(idx.map((i) => s.symbols[i]))].join(",");
      const sym = s.symmetry || {};
      const mult = sym.multiplicity && sym.multiplicity.length > idx[0] ? sym.multiplicity[idx[0]] : n;
      const orbit = ed.symMode === "sym" && sym.available && sym.wyckoffs ? `, orbit ${mult}${sym.wyckoffs[idx[0]]}` : "";
      const el = G.SYMBOLS[G.zOf(args.element)] || args.element;
      return `Replace ${old}→${el} (${atoms}${orbit})`;
    }
    if (op === "delete") return `Delete ${n} Atom${n === 1 ? "" : "s"}`;
    if (op === "move") return `Move ${n} Atom${n === 1 ? "" : "s"}`;
    if (op === "add") return `Add ${G.SYMBOLS[G.zOf(args.element)] || args.element}`;
    return { set_cell: "Edit Cell", set_positions: "Set Coordinates", supercell: "Supercell", primitive: "Primitive Cell",
      symmetrize: "Symmetrize", clean: "Clean Geometry", set_symprec: "Symmetry Tolerance" }[op] || op;
  }

  /* ---------------------------------------------------------------- edits */
  /** One edit: {merge, label, onConfirm, onDone, kind}. Returns the request's promise. */
  function edit(ed, op, args = {}, opts = {}) {
    const why = ed.editBlock();
    if (why) { S.toast(why, "info"); return Promise.resolve(null); }
    const t = performance.now();
    performance.mark("ui:edit:start");
    const pred = !busyUnpredicted(ed) && !opts.wait ? predict(ed, op, args) : null;
    const top = ed.rows[ed.pos];
    const canMerge = opts.merge && top && top.merge === opts.merge && t - (top.tInput || 0) < MERGE_MS
      && ed.pos === ed.rows.length - 1 && pred;
    ed.rows[ed.pos].snap = ed.snapNow();
    const item = { kind: op, route: "edit/op", predicted: !!pred, blocks: !pred, merge: opts.merge || null,
      count: 1, onConfirm: opts.onConfirm || defaultConfirm(ed), onDone: opts.onDone };
    if (canMerge) {
      item.group = top.group;
      top.count = (top.count || 1) + 1;
      top.label = `${labelFor(ed, op, args, pred)} (×${top.count})`;
      top.tInput = t;
      top.held = pred.structure;
      top.rev = null;                         // changed: no longer the confirmed state
    } else {
      item.group = opts.merge ? ++groupSeq : null;
      ed.rows = ed.rows.slice(0, ed.pos + 1);
      ed.rows.push({ label: opts.label || labelFor(ed, op, args, pred), rev: null, held: pred ? pred.structure : null, t: Date.now() / 1000,
        n_atoms: pred ? pred.structure.numbers.length : ed.n, merge: opts.merge || null, group: item.group, count: 1, tInput: t });
      ed.pos++;
    }
    const sent = { ...args, ...(op === "move" ? { frac_delta: fracDelta(ed, args), delta: undefined } : {}) };
    if (op === "move" || op === "set_positions") sent.wrap = wrapOn(args);
    item.args = sent;
    item.make = (d) => opBody(d, item, op, item.args, opts.label ? { label: opts.label } : {});
    if (opts.merge) {
      item.combine = (other) => {           // a waiting nudge takes the next one in
        item.args.frac_delta = item.args.frac_delta.map((x, k) => x + other.args.frac_delta[k]);
        item.count += other.count;
      };
    }
    if (pred) {
      stats.predicted++;
      if (pred.indexMap) ed.remap(pred.indexMap);
      ed.structure = pred.structure;
      ed.predicted = true;
      ed.remeasure();
      E.structureChanged(ed, { local: true });
    } else {
      stats.waited++;
      ed.waiting = true;
      E.stateChanged(ed);
    }
    const q = queueFor(ed);
    return q.push(item);
  }
  /** A Sym move that would grow an orbit: the server asks first. */
  function defaultConfirm(ed) {
    return (data) => S.confirmGrow(ed, data);
  }

  function step(ed, to, kind, body) {
    if (to < 0 || to >= ed.rows.length || to === ed.pos) return Promise.resolve(null);
    if (ed.lost) { localStep(ed, to); return Promise.resolve(null); }
    ed.rows[ed.pos].snap = ed.snapNow();
    localStep(ed, to);
    if (!ed.srv.session && !(ed.queue && !ed.queue.idle)) return Promise.resolve(null);
    return queueFor(ed).push({ kind, route: kind === "jump" ? "edit/jump" : "edit/op", predicted: !ed.waiting,
      blocks: ed.waiting, remap: false, make: (d) => ({ session: d.srv.session, base_rev: d.srv.rev, ops_key: d.srv.opsKey, ...body(d) }) });
  }
  function localStep(ed, to) {
    ed.pos = to;
    const row = ed.rows[to];
    if (row.held && !busyUnpredicted(ed)) {
      ed.structure = row.held;
      ed.predicted = true;
      ed.restore(row.snap || (row.rev !== null && ed.revSnap.get(row.rev)));
      ed.remeasure();
      E.structureChanged(ed, { local: true });
    } else {
      ed.waiting = true;
      if (row.snap || ed.revSnap.get(row.rev)) ed.pendingSnap = row.snap || ed.revSnap.get(row.rev);
      E.stateChanged(ed);
    }
  }
  const undo = (ed) => step(ed, ed.pos - 1, "undo", () => ({ op: "undo", args: {} }));
  const redo = (ed) => step(ed, ed.pos + 1, "redo", () => ({ op: "redo", args: {} }));
  const jump = (ed, index) => step(ed, index, "jump", () => ({ index }));

  /** P1 or Sym: the document's mode, never an undo step. */
  function setMode(ed, mode) {
    if (ed.symMode === mode) return Promise.resolve(null);
    ed.symMode = mode;
    E.stateChanged(ed);
    if (!ed.srv.session) return Promise.resolve(null);   // sent with the first edit
    return queueFor(ed).push({ kind: "set_locked", route: "edit/op", predicted: true, remap: false,
      make: (d) => ({ session: d.srv.session, base_rev: d.srv.rev, op: "set_locked", args: { locked: mode === "sym" }, record: false, ops_key: d.srv.opsKey }) });
  }
  function setSymprec(ed, symprec) {
    ed.symprec = symprec;
    if (!ed.srv.session) { E.stateChanged(ed); return Promise.resolve(null); }
    return queueFor(ed).push({ kind: "set_symprec", route: "edit/op", predicted: true, remap: false,
      make: (d) => ({ session: d.srv.session, base_rev: d.srv.rev, op: "set_symprec", args: { symprec }, record: false, ops_key: d.srv.opsKey }) });
  }
  /** Any other request in the document's order (Clean, Save): {route, body(ed), payloadKey, passConflicts}. */
  async function request(ed, kind, route, body, opts = {}) {
    if (opts.session !== false) await ensureSession(ed);
    return queueFor(ed).push({ kind, route, predicted: !opts.changes, blocks: !!opts.changes, payloadKey: opts.payloadKey,
      remap: opts.remap, passConflicts: !!opts.passConflicts, make: (d) => ({ session: d.srv.session, base_rev: d.srv.rev, ...body(d) }), onDone: opts.onDone });
  }

  return { edit, undo, redo, jump, setMode, setSymprec, ensureSession, request, queueFor, stats, settle, adoptPayload, rebuildRows, same };
}
