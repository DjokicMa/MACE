/* What the Phase 3 commands do (their names, keys and menus are in
 * js/actions.js, which the shell loads at first paint; the work is here,
 * loaded with the first structure document). Each handler gets the editor
 * API, the shell's, the active document's editor state and the command's
 * arguments (an atom from a context menu, the chord pressed). */
import * as G from "../geom/core.js";

const STEP_KEYS = { "edit.nudgeLeft": [0, -1], "edit.nudgeRight": [0, 1], "edit.nudgeUp": [1, 1], "edit.nudgeDown": [1, -1],
  "edit.nudgeOut": [2, 1], "edit.nudgeIn": [2, -1] };

export function makeCommands(E, S) {
  const sorted = (xs) => [...xs].sort((a, b) => a - b);
  const sel = (ed) => sorted(ed.selection);
  const atomOf = (a) => (a && a.atom ? a.atom : null);

  /** The selection, or the atom a context menu was opened on. */
  const target = (ed, args) => (atomOf(args) ? [atomOf(args).i] : sel(ed));

  function nudge(id, ed, args) {
    const [axis, sign] = STEP_KEYS[id];
    const indices = sel(ed);
    if (!indices.length) return;
    const shift = args && args.chord && args.chord.includes("Shift");
    const cart = ed.coordMode === "cart";
    const step = (E.nudgeStep(ed) || (cart ? 0.05 : 0.01)) * (shift ? 10 : 1) * sign;
    const v = [0, 0, 0];
    v[axis] = step;
    const key = `nudge:${indices.join(",")}`;
    return E.edit(ed, "move", cart ? { indices, delta: v } : { indices, frac_delta: v }, { merge: key });
  }

  function copySelection(ed) {
    const s = ed.structure, idx = sel(ed);
    if (!idx.length) return;
    const d = Number(S.setting("decimals.coord")) || 5;
    const lines = [String(idx.length), `${ed.name}: element, fractional x y z, Cartesian x y z (Å)`];
    for (const i of idx) {
      const c = s.cart[i];
      lines.push(`${s.symbols[i].padEnd(3)} ${s.frac[i].map((x) => G.fmtNum(x, d)).join(" ")}   ${c.map((x) => G.fmtNum(x, d)).join(" ")}`);
    }
    return S.copyText(lines.join("\n") + "\n", `Copied ${idx.length} atom${idx.length === 1 ? "" : "s"}`);
  }

  /** Phase 3's Copy block -> [{element, frac}] (the fractional columns). */
  function parseBlock(text) {
    const lines = String(text || "").split(/\r?\n/).filter((l) => l.trim());
    const n = Number(lines[0]);
    const out = [];
    for (const l of lines.slice(2, 2 + n)) {
      const p = l.trim().split(/\s+/);
      const f = p.slice(1, 4).map(Number);
      if (p.length >= 4 && f.every(Number.isFinite)) out.push({ element: p[0], frac: f });
    }
    return out;
  }
  /** Paste: the copied atoms at the same fractional coordinates, one step
   * "Paste 3 Atoms", selected afterwards; in Sym mode each one brings its
   * orbit, and more than 3× the copied count asks first (decision D7). */
  async function pasteAtoms(ed, text) {
    const atoms = parseBlock(text);
    if (!atoms.length) { S.toast("Nothing to paste: copy atoms first", "info"); return null; }
    let total = atoms.length;
    if (ed.symMode === "sym") {
      // before the first edit the document has no operations yet: ask for them (stateless)
      let ops = ed.ops;
      if (!ops) {
        const s = ed.structure;
        const r = await window.MACE.api("symmetry", { body: { cell: s.cell, frac: s.frac, numbers: s.numbers, symprec: ed.symprec, ops: 1 } }).catch(() => null);
        if (r && r.rotations) ops = { rotations: r.rotations, translations: r.translations };
      }
      if (ops) total = atoms.reduce((t, a) => t + G.orbitApply(ops, a.frac, ed.structure.cell, Math.max(2 * ed.symprec, 1e-3)).length, 0);
    }
    if (total > 3 * atoms.length) {
      const ok = await S.confirmPasteOrbits(atoms.length, total);
      if (!ok) return null;
    }
    const n0 = ed.n;
    return E.edit(ed, "add_many", { atoms }, { label: `Paste ${atoms.length} Atom${atoms.length === 1 ? "" : "s"}`, wait: true,
      onDone: () => setTimeout(() => E.select(ed, Array.from({ length: Math.max(0, ed.n - n0) }, (_, k) => n0 + k), "replace"), 0) });
  }
  /** Duplicate: the selection copied one nudge step to the right on screen,
   * one step, then a grab of the copies (Blender style: Esc keeps them at the
   * offset, a click places them). */
  async function duplicate(ed) {
    const idx = sel(ed);
    if (!idx.length) return null;
    const s = ed.structure, cam = E.cam ? E.cam() : null;
    const right = cam ? cam.calibrate().right : [1, 0, 0];
    const step = Number(S.setting("nudgeStepCart")) || 0.05;
    const df = G.fracOf(s.cell, G.scale3(right, Math.max(step, 0.3)));
    const atoms = idx.map((i) => ({ element: s.symbols[i], frac: s.frac[i].map((x, k) => x + df[k]) }));
    const n0 = ed.n;
    await E.edit(ed, "add_many", { atoms }, { label: `Duplicate ${idx.length} Atom${idx.length === 1 ? "" : "s"}`, wait: true });
    await new Promise((r) => setTimeout(r, 0));
    const added = Array.from({ length: Math.max(0, ed.n - n0) }, (_, k) => n0 + k);
    E.select(ed, added, "replace");
    if (added.length && S.viewportFocused && S.viewportFocused()) {
      S.setTool("move");
      const [x, y] = E.pointer().lastPos();
      E.tools().startGrab(x, y, false);
    }
    return added;
  }

  /** Select by Element: a popover listing the elements present, with counts. */
  function byElement(ed, args) {
    const s = ed.structure, counts = new Map();
    s.symbols.forEach((x) => counts.set(x, (counts.get(x) || 0) + 1));
    const items = [...counts].map(([el, n]) => ({ label: `${el} × ${n}`, icon: "atom",
      run: () => E.select(ed, s.symbols.map((x, i) => (x === el ? i : -1)).filter((i) => i >= 0), "replace") }));
    const r = document.getElementById("viewer-main").getBoundingClientRect();
    S.openMenuAt(items, args && args.x ? args.x : r.left + 24, args && args.y ? args.y : r.top + 24, "Select by Element");
  }

  async function grow(ed, dir) {
    const cur = sel(ed);
    if (!cur.length) return;
    if (dir > 0) { E.select(ed, await E.neighbors(ed, cur), "add"); return; }
    // shrink: drop the atoms with a bonded neighbor outside the selection
    const set = new Set(cur), out = [];
    for (const i of cur) {
      const n = await E.neighbors(ed, [i]);
      if (n.some((j) => !set.has(j))) out.push(i);
    }
    E.select(ed, out, "subtract");
  }

  const H = {
    "edit.undo": (ed) => E.undo(ed),
    "edit.redo": (ed) => E.redo(ed),
    "edit.copy": (ed) => copySelection(ed),
    // Phase 4 (plan T9, B12): Paste, Cut and Duplicate, each one undo step
    "edit.pasteText": (ed, a) => pasteAtoms(ed, a && a.text),
    "edit.cut": async (ed) => {
      const idx = sel(ed);
      if (!idx.length) return null;
      await copySelection(ed);
      return E.edit(ed, "delete", { indices: idx }, { label: `Cut ${idx.length} Atom${idx.length === 1 ? "" : "s"}` });
    },
    "edit.duplicate": (ed) => duplicate(ed),
    "edit.delete": (ed, a) => { const t = target(ed, a); return t.length ? E.edit(ed, "delete", { indices: t }) : null; },
    "build.delete": (ed, a) => H["edit.delete"](ed, a),
    "select.all": (ed) => E.select(ed, ed.structure.numbers.map((_, i) => i), "replace"),
    "select.none": (ed) => E.select(ed, [], "replace"),
    "select.invert": (ed) => E.select(ed, ed.structure.numbers.map((_, i) => i).filter((i) => !ed.selection.has(i)), "replace"),
    "select.byElement": (ed, a) => byElement(ed, a),
    "select.equivalents": (ed, a) => { const t = target(ed, a); const out = new Set(); t.forEach((i) => ed.orbitOf(i).forEach((j) => out.add(j))); E.select(ed, [...out], atomOf(a) ? "replace" : "add"); },
    "select.neighbors": async (ed, a) => { const t = target(ed, a); E.select(ed, await E.neighbors(ed, t), "replace"); },
    "select.grow": (ed) => grow(ed, 1),
    "select.shrink": (ed) => grow(ed, -1),
    "atom.selectEquivalents": (ed, a) => H["select.equivalents"](ed, a),
    "atom.selectElement": (ed, a) => { const at = atomOf(a); if (!at) return; const el = ed.structure.symbols[at.i]; E.select(ed, ed.structure.symbols.map((x, i) => (x === el ? i : -1)).filter((i) => i >= 0), "replace"); },
    "atom.selectNeighbors": (ed, a) => H["select.neighbors"](ed, a),
    "build.symModeP1": (ed) => E.setMode(ed, "p1"),
    "build.symModeSym": (ed) => E.toSym(ed),
    "build.symMode": (ed) => (ed.symMode === "sym" ? E.setMode(ed, "p1") : E.toSym(ed)),
    "view.frameSelection": () => E.frameSelection(),
    /* G: arms Move; with a selection and the viewport focused, the grab starts at the pointer (spec §2.1, plan D9) */
    "tool.move": (ed) => {
      S.setTool("move");
      if (ed.selection.size && S.viewportFocused() && !E.tools().grabbing) {
        const [x, y] = E.pointer().lastPos();
        E.tools().startGrab(x, y, false);
      }
    },
    "edit.replaceElement": (ed, a) => {
      const t = target(ed, a);
      if (!t.length) { S.setTool("replace"); return; }
      S.pickElement(document.getElementById("viewer-main"), (el) => E.edit(ed, "replace", { indices: t, element: el }),
        { recent: true, title: `Replace ${t.length} atom${t.length === 1 ? "" : "s"} with` });
    },
    "tool.measureSelection": (ed) => E.measure.measureSelection(ed),
    "tool.clearMeasurements": (ed) => E.measure.clear(ed),
    "atom.measureFrom": (ed, a) => { const at = atomOf(a); if (!at) return; S.setTool("measure"); E.measure.startFrom({ i: at.i, image: (at.image || [0, 0, 0]).slice() }); },
    "viewport.addAtomHere": (ed, a) => (a && Number.isFinite(a.x) ? E.tools().addHere(a.x, a.y) : S.setTool("add")),
    "file.save": (ed) => E.saving.save(ed),
    "file.saveAs": (ed, a) => E.saving.saveAs(ed, a && a.preset ? a.preset : {}),
    "tool.addElement": () => { if (!E.tools().focusElement()) S.toolChip(); setTimeout(() => E.tools().focusElement(), 0); },
  };
  for (const id of Object.keys(STEP_KEYS)) H[id] = (ed, a) => nudge(id, ed, a);
  return H;
}
