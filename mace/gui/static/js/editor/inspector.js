/* The inspector's editing sections for a structure document (spec §1.5):
 * Selection (one atom's element, label, Wyckoff site and typed coordinates
 * with a Frac/Cart toggle; several atoms' counts, centroid and a Δ to
 * apply), Nudge, an editable Cell, and the live Symmetry with its
 * tolerance and the P1/Sym mode. Fields commit on Enter or when they lose
 * focus; they take expressions ("1/3", "0.5+0.01"); a value that is not
 * one gets a red border and a message, and nothing is sent.
 * 1.1.6: the niche rows (occupancy, scale atoms with the cell, Hall symbol,
 * operations, the symmetry tolerance) carry data-tier="advanced": the
 * Inspector's Basic tab hides them (js/inspector_tiers.js). */
import * as G from "../geom/core.js";
import * as CO from "./coordination.js";

const SYSTEMS = [[2, "triclinic"], [15, "monoclinic"], [74, "orthorhombic"], [142, "tetragonal"], [167, "trigonal"],
  [194, "hexagonal"], [230, "cubic"]];
export const crystalSystem = (n) => (SYSTEMS.find(([m]) => n <= m) || [0, ""])[1];

let uid = 0;
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const hm = (sym) => esc(sym || "").replace(/-(\d)/g, '<span class="bar">$1</span>').replace(/_(\d)/g, "<sub>$1</sub>");

export function createInspector(E, S) {
  const dec = () => { const d = Number(S.setting("decimals.coord")); return Number.isFinite(d) ? d : 5; };
  const decL = () => { const d = Number(S.setting("decimals.length")); return Number.isFinite(d) ? d : 4; };
  const decA = () => { const d = Number(S.setting("decimals.angle")); return Number.isFinite(d) ? d : 3; };

  /** A numeric field that takes an expression; onCommit(value) runs on Enter or blur. */
  function exprField(label, value, onCommit, opts = {}) {
    const id = `ins-f${++uid}`;
    const wrap = document.createElement("label");
    wrap.className = "ins-field" + (opts.wide ? " wide" : "");
    wrap.innerHTML = `<span class="ins-k">${esc(label)}</span><input id="${id}" inputmode="decimal" autocomplete="off" spellcheck="false"` +
      ` aria-label="${esc(opts.aria || label)}"${opts.unit ? ` data-unit="${esc(opts.unit)}"` : ""}>` +
      (opts.unit ? `<span class="ins-unit">${esc(opts.unit)}</span>` : "") + `<span class="ins-err" id="${id}-err" role="alert"></span>`;
    const input = wrap.querySelector("input"), err = wrap.querySelector(".ins-err");
    input.value = value;
    input.dataset.shown = value;
    const commit = () => {
      if (input.value === input.dataset.shown) { clear(); return; }
      const r = G.evalExpr(input.value);
      if (!r.ok) {
        input.setAttribute("aria-invalid", "true");
        input.setAttribute("aria-describedby", `${id}-err`);
        err.textContent = r.error;
        return;
      }
      clear();
      input.dataset.shown = input.value;
      onCommit(r.value, input);
    };
    const clear = () => { input.removeAttribute("aria-invalid"); input.removeAttribute("aria-describedby"); err.textContent = ""; };
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); commit(); }
      if (e.key === "Escape") { input.value = input.dataset.shown; clear(); input.blur(); e.stopPropagation(); }
    });
    input.addEventListener("blur", commit);
    wrap.set = (v) => { if (document.activeElement !== input) { input.value = v; input.dataset.shown = v; clear(); } };
    return wrap;
  }
  function segment(name, options, current, onPick) {
    const el = document.createElement("span");
    el.className = "segment ins-seg";
    el.setAttribute("role", "radiogroup");
    el.setAttribute("aria-label", name);
    el.innerHTML = options.map(([v, t]) => `<button type="button" role="radio" data-v="${v}" aria-checked="${v === current}">${esc(t)}</button>`).join("");
    el.addEventListener("click", (e) => { const b = e.target.closest("[data-v]"); if (b) onPick(b.dataset.v); });
    return el;
  }

  /* the "CN 6 (O×6, 2.106 Å)" line (plan B9): up to 2000 atoms, else the Coordination tab */
  function cnText(s, i) {
    const rule = G.bondRule(s, S.setting);
    const r = CO.rowsSync(s, [i], rule.tol, rule);
    return r ? CO.cnLine(r[0]) : "— (Analysis › Coordination Analysis)";
  }
  /* ---------------------------------------------------------------- Selection */
  function selection(ed) {
    const root = document.createElement("div");
    root.className = "ins-sel";
    const s = ed.structure, idx = [...ed.selection].sort((a, b) => a - b);
    const cart = ed.coordMode === "cart", d = dec();
    const toggle = segment("Coordinates", [["frac", "Frac"], ["cart", "Cart"]], ed.coordMode, (v) => {
      ed.coordMode = v; ed.nudgeStep = undefined; S.inspectorChanged();
    });
    if (!idx.length) {
      root.innerHTML = `<p class="insp-hint">Click an atom. Shift-click or drag to select more.</p>`;
      return root;
    }
    if (idx.length === 1) {
      const i = idx[0], y = s.symmetry || {};
      const site = y.available ? `${E.wyckoff(s, i)}${y.site_symmetry && y.site_symmetry[i] ? ` · ${y.site_symmetry[i]}` : ""}` : "—";
      root.innerHTML = `<div class="ins-atom">
          <button class="ins-el" type="button" title="Replace element (E)" aria-label="Element ${esc(s.symbols[i])}: replace">${esc(s.symbols[i])}</button>
          <div class="ins-id"><div class="ins-name">${esc(s.symbols[i])}${i + 1}</div><div class="ins-meta">index ${i + 1} · Wyckoff <span class="ins-site">${esc(site)}</span></div></div>
        </div>
        <div class="ins-row ins-coords-head"><span class="ins-k">${cart ? "Cartesian (Å)" : "Fractional"}</span></div>
        <div class="ins-xyz"></div>
        <div class="ins-row" data-tier="advanced"><span class="ins-k">Occupancy</span><span class="ins-v">1.0</span></div>
        <div class="ins-row"><span class="ins-k">Coordination</span><span class="ins-v ins-cn">${esc(cnText(s, i))}</span></div>`;
      root.querySelector(".ins-coords-head").appendChild(toggle);
      root.querySelector(".ins-el").addEventListener("click", (e) => S.pickElement(e.currentTarget, (el) => E.edit(ed, "replace", { indices: [i], element: el })));
      const v = cart ? s.cart[i] : s.frac[i];
      const xyz = root.querySelector(".ins-xyz");
      ["x", "y", "z"].forEach((ax, k) => xyz.appendChild(exprField(ax, G.fmtNum(v[k], d), (val) => {
        const cur = (cart ? s.cart[i] : s.frac[i]).slice();
        cur[k] = val;
        E.edit(ed, "set_positions", cart ? { indices: [i], cart: [cur] } : { indices: [i], frac: [cur] });
      }, { aria: `${cart ? "Cartesian" : "Fractional"} ${ax}`, unit: cart ? "Å" : "" })));
      return root;
    }
    const counts = new Map();
    idx.forEach((i) => counts.set(s.symbols[i], (counts.get(s.symbols[i]) || 0) + 1));
    const c = [0, 1, 2].map((k) => idx.reduce((t, i) => t + (cart ? s.cart[i][k] : s.frac[i][k]), 0) / idx.length);
    root.innerHTML = `<div class="ins-row"><span class="ins-k">${idx.length} atoms</span><span class="ins-v">${[...counts].map(([el, n]) => `${esc(el)} × ${n}`).join(", ")}</span></div>
      <div class="ins-row ins-coords-head"><span class="ins-k">Centroid</span></div>
      <div class="ins-row"><span class="ins-v mono">${c.map((x) => G.fmtNum(x, d)).join("  ")}${cart ? " Å" : ""}</span></div>
      <div class="ins-row"><span class="ins-k">Apply Δ</span></div><div class="ins-xyz ins-delta"></div>
      <div class="ins-row end"><button class="tb-btn-text ins-apply" type="button">Apply Δ</button></div>`;
    root.querySelector(".ins-coords-head").appendChild(toggle);
    const fields = ["x", "y", "z"].map((ax) => exprField(`Δ${ax}`, "0", () => {}, { aria: `Δ${ax}`, unit: cart ? "Å" : "" }));
    fields.forEach((f) => root.querySelector(".ins-delta").appendChild(f));
    root.querySelector(".ins-apply").addEventListener("click", () => {
      const vals = fields.map((f) => G.evalExpr(f.querySelector("input").value));
      const bad = vals.findIndex((r) => !r.ok);
      if (bad >= 0) { f_error(fields[bad], vals[bad].error); return; }
      const v = vals.map((r) => r.value);
      if (v.every((x) => x === 0)) return;
      E.edit(ed, "move", cart ? { indices: idx, delta: v } : { indices: idx, frac_delta: v });
    });
    return root;
  }
  function f_error(field, msg) {
    const input = field.querySelector("input"), err = field.querySelector(".ins-err");
    input.setAttribute("aria-invalid", "true");
    input.setAttribute("aria-describedby", err.id);
    err.textContent = msg;
  }

  /* ---------------------------------------------------------------- Nudge */
  function nudge(ed) {
    const root = document.createElement("div");
    root.className = "ins-nudge";
    const cart = ed.coordMode === "cart";
    const step = E.nudgeStep(ed);
    root.appendChild(exprField(cart ? "Step (Å)" : "Step (frac)", String(step), (v) => { ed.nudgeStep = Math.abs(v); }, { aria: "Nudge step", unit: cart ? "Å" : "" }));
    const pad = document.createElement("div");
    pad.className = "ins-pad";
    pad.innerHTML = ["x", "y", "z"].map((ax) => `<span class="ins-k">${ax}</span>` +
      `<button type="button" class="tb-btn-text" data-ax="${ax}" data-s="-1" aria-label="Nudge −${ax}">−</button>` +
      `<button type="button" class="tb-btn-text" data-ax="${ax}" data-s="1" aria-label="Nudge +${ax}">+</button>`).join("");
    pad.addEventListener("click", (e) => {
      const b = e.target.closest("[data-ax]");
      if (!b || !ed.selection.size) return;
      const v = [0, 0, 0];
      v["xyz".indexOf(b.dataset.ax)] = Number(b.dataset.s) * E.nudgeStep(ed) * (e.shiftKey ? 10 : 1);
      const indices = [...ed.selection].sort((a, c) => a - c);
      E.edit(ed, "move", cart ? { indices, delta: v } : { indices, frac_delta: v }, { merge: `nudge:${indices.join(",")}` });
    });
    root.appendChild(pad);
    root.insertAdjacentHTML("beforeend", `<p class="insp-hint" data-tier="advanced">Move tool: Alt+arrows, Alt+PgUp/PgDn; Shift: 10× the step.</p>`);
    return root;
  }

  /* ---------------------------------------------------------------- Cell */
  function cell(ed) {
    const root = document.createElement("div");
    root.className = "ins-cell";
    const L = ed.structure.lattice || G.latticeOf(ed.structure.cell), dl = decL(), da = decA();
    let scale = ed.scaleAtoms !== false;
    const grid = document.createElement("div");
    grid.className = "ins-grid3";
    const keys = [["a", "a", "Å", dl], ["b", "b", "Å", dl], ["c", "c", "Å", dl], ["alpha", "α", "°", da], ["beta", "β", "°", da], ["gamma", "γ", "°", da]];
    // a slab, polymer or molecule (1.1.6): a non-periodic vector is only the drawing box, so its length and
    // the angles it takes part in are shown, not edited, as the read-only lattice card shows them
    const pbc = (ed.structure.pbc || [true, true, true]).map(Boolean), C = ed.structure.cell;
    const fixed = (i) => (i < 3 ? !pbc[i] : !pbc[[1, 0, 0][i - 3]] || !pbc[[2, 2, 1][i - 3]]);
    keys.forEach(([k, t, u, d], i) => {
      if (fixed(i)) {
        const ro = document.createElement("div");
        ro.className = "ins-row";
        ro.innerHTML = i < 3 ? `<span class="ins-k">${t}</span><span class="ins-v muted" title="not periodic: the drawn box only spans the atoms">not periodic</span>`
          : `<span class="ins-k">${t}</span><span class="ins-v">${G.fmtNum(L[k], d)}°</span>`;
        grid.appendChild(ro);
        return;
      }
      grid.appendChild(exprField(t, G.fmtNum(L[k], d), (v) => {
        const p = keys.map(([kk]) => (kk === k ? v : L[kk]));
        if (!G.cellFromParams(...p)) { S.toast("Those cell parameters make no cell", "error"); return; }
        E.edit(ed, "set_cell", { cellpar: p, scale_atoms: scale });
      }, { aria: `Cell ${t}`, unit: u }));
    });
    root.appendChild(grid);
    const per = [0, 1, 2].filter((i) => pbc[i]);
    const size = per.length === 3 ? `<span class="ins-k">Volume</span><span class="ins-v">${G.fmtNum(L.volume, 3)} Å³</span>`
      : per.length === 2 ? `<span class="ins-k">Area</span><span class="ins-v">${G.fmtNum(G.norm3(G.cross3(C[per[0]], C[per[1]])), 3)} Å²</span>`
      : per.length === 1 ? `<span class="ins-k">Length</span><span class="ins-v">${G.fmtNum(G.norm3(C[per[0]]), 3)} Å</span>` : "";
    root.insertAdjacentHTML("beforeend", `${size ? `<div class="ins-row">${size}</div>` : ""}
      <label class="tog ins-row" data-tier="advanced"><input type="checkbox" class="ins-scale"${scale ? " checked" : ""}><span>Scale atoms with cell</span></label>
      <div class="ins-row end"><button type="button" class="tb-btn-text ins-build" data-cmd-run="build.supercell">Supercell…</button>
        <button type="button" class="tb-btn-text ins-build" data-cmd-run="build.transform">Transform…</button>
        <button type="button" class="tb-btn-text ins-editcell" data-cmd-run="edit.editCell">Edit Cell…</button></div>`);
    root.querySelector(".ins-scale").addEventListener("change", (e) => { scale = e.target.checked; ed.scaleAtoms = scale; });
    root.querySelector(".ins-editcell").addEventListener("click", () => S.run("edit.editCell"));
    root.querySelectorAll(".ins-build").forEach((b) => b.addEventListener("click", () => S.run(b.dataset.cmdRun)));
    return root;
  }

  /* ---------------------------------------------------------------- Symmetry */
  function symmetry(ed) {
    const root = document.createElement("div");
    root.className = "ins-sym";
    const s = ed.structure, y = s.symmetry || {};
    const mode = segment("Symmetry mode", [["p1", "P1"], ["sym", "Sym"]], ed.symMode, (v) => S.run(v === "sym" ? "build.symModeSym" : "build.symModeP1"));
    if (!y.available) {
      root.innerHTML = `<p class="insp-hint">${esc(y.note || "No space group")}</p>` +
        (y.not_computed ? `<div class="ins-row end"><button type="button" class="tb-btn-text sym-compute">Compute Now</button></div>` : "");
      const b = root.querySelector(".sym-compute");
      if (b) b.addEventListener("click", () => { b.disabled = true; b.textContent = "Computing…"; window.MACE.computeSymmetry(); });
      return root;
    }
    const unique = new Set(y.equivalent_atoms).size;
    const ref = ed.referenceGroup;
    const ops = (y.n_operations_conventional || y.n_operations), lowered = ref && ops < ref.n_operations;
    root.innerHTML = `<div class="symline"><span class="hm${lowered ? " warn" : ""}" title="${esc(y.international)}">${hm(y.international)}</span><span class="sgnum">No. ${y.number}</span></div>
      <table class="kv">
        <tr><td>Crystal system</td><td>${esc(crystalSystem(y.number))}</td></tr>
        <tr><td>Point group</td><td class="pg">${hm(y.pointgroup)}</td></tr>
        <tr data-tier="advanced"><td>Hall symbol</td><td>${esc(y.hall)}</td></tr>
        <tr data-tier="advanced"><td>Operations</td><td${ops !== y.n_operations ? ` title="${y.n_operations} in this cell, counting its lattice translations"` : ""}>${ops}</td></tr>
        <tr><td>Unique atoms</td><td>${unique} of ${s.numbers.length}</td></tr>
      </table>
      ${lowered ? `<p class="insp-hint warn">Lower than ${esc(ref.international)} (${ref.number}), the group this structure had.</p>` : ""}
      <div class="ins-row ins-mode"><span class="ins-k">Editing</span></div>
      <div class="ins-tol" data-tier="advanced"></div>
      <div class="ins-row end"><button type="button" class="tb-btn-text ins-symmetrize">Symmetrize…</button></div>`;
    root.querySelector(".ins-mode").appendChild(mode);
    root.querySelector(".ins-tol").appendChild(exprField("Tolerance", String(ed.symprec), (v) => {
      if (!(v >= 1e-6 && v <= 0.5)) { S.toast("The tolerance is 1e-6 to 0.5 Å", "error"); return; }
      E.setSymprec(ed, v);
    }, { aria: "Symmetry tolerance", unit: "Å" }));
    // Build › Symmetrize: the tolerance sweep and a preview before the one undo step
    root.querySelector(".ins-symmetrize").addEventListener("click", () => S.run("build.symmetrize"));
    return root;
  }

  return { selection, nudge, cell, symmetry, crystalSystem };
}
