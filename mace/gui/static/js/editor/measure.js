/* Measurements (spec §6): the Measure tool (M) collects atoms, periodic
 * images included: 2 give a distance, 3 an angle, 4 a dihedral (the chain
 * grows with each click; a fifth click, Esc or a click on empty space
 * starts again; Shift+click chains distances from the last atom). Mod+M
 * measures the selection. They are kept per document as {type, atoms:
 * [{i, image}], value, visible}, recomputed after every edit, remapped when
 * atoms move in the list, struck through when an atom is deleted, and put
 * back on undo. The Measurements panel lists them, with an eye, a delete
 * and Copy as TSV. */
import * as G from "../geom/core.js";

const TYPES = { 2: "Distance", 3: "Angle", 4: "Dihedral" };
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

export const valueText = (m) => (m.atoms.length === 2 ? `${G.fmtNum(m.value, 3)} Å` : m.atoms.length === 3 ? `${G.fmtNum(m.value, 2)}°` : `${G.fmtNum(m.value, 1)}°`);
/** "Mg1–O5′ (1,0,0)": a primed atom sits in another cell than the first one. */
export function atomsText(s, m) {
  const first = m.atoms[0].image || [0, 0, 0];
  return m.atoms.map((a) => {
    const d = (a.image || [0, 0, 0]).map((x, k) => x - first[k]);
    const off = d.some((x) => x) ? `′ (${d.join(",")})` : "";
    return `${s.symbols[a.i] || "?"}${a.i + 1}${off}`;
  }).join("–");
}

export function createMeasure(E, S, V) {
  let chain = null;            // {m: the measurement being built}
  const ed = () => V.ed();
  function make(e, atoms) {
    const m = { type: TYPES[atoms.length], atoms: atoms.map((a) => ({ i: a.i, image: a.image.slice() })), visible: true };
    m.value = G.measure(e.structure.cell, e.structure.frac, m.atoms);
    return m;
  }
  function changed(e) { E.redrawMeasurements(e); S.measurementsChanged(e); }
  function atomAt(k) {
    const disp = V.scene().display;
    return { i: disp.base[k], image: [disp.shift[3 * k], disp.shift[3 * k + 1], disp.shift[3 * k + 2]] };
  }
  const tool = {
    click(ev, x, y, k) {
      const e = ed();
      if (!e) return;
      if (chain && chain.ed !== e) chain = null;          // another document's clicks do not count
      if (k < 0) { chain = null; S.setCursor(""); S.toolChip(); return; }
      const a = atomAt(k);
      // the same atom (and image) twice would measure 0 Å or an undefined angle: ignored
      const same = (b) => b && b.i === a.i && b.image.every((x, j) => x === a.image[j]);
      const open = chain && chain.atoms && chain.atoms.length < 4;
      if ((ev.shiftKey && chain && same(chain.last)) || (!ev.shiftKey && open && chain.atoms.some(same))) {
        S.setCursor("That atom is already in this measurement");
        return;
      }
      if (ev.shiftKey && chain && chain.last) {
        const m = make(e, [chain.last, a]);          // Shift: a chain of distances
        e.measurements.push(m);
        chain = { atoms: [chain.last, a], m, last: a, ed: e };
      } else {
        if (!chain || !chain.atoms.length || chain.atoms.length >= 4) chain = { atoms: [], m: null, last: null, ed: e };
        chain.atoms.push(a);
        chain.last = a;
        if (chain.atoms.length >= 2) {                 // 2: distance, 3: angle, 4: dihedral (it replaces the one before)
          const m = make(e, chain.atoms);
          const at = chain.m ? e.measurements.indexOf(chain.m) : -1;
          if (at >= 0) e.measurements[at] = m; else e.measurements.push(m);
          chain.m = m;
        }
      }
      changed(e);
      S.toolChip();
    },
    hover(ev, x, y) {
      const e = ed();
      if (!e || !chain || !chain.last) return;
      const k = V.scene().pickAt(x, y);
      if (k < 0) return;
      const a = atomAt(k), m = make(e, [chain.last, a]);
      S.setCursor(`${atomsText(e.structure, m)} ${valueText(m)}`);
    },
    reset() { chain = null; },
  };
  /** Mod+M: the selection in the order it was picked (2 to 4 atoms, each at
   * the image nearest the one before). */
  function measureSelection(e) {
    const idx = [...e.selection];
    if (idx.length < 2 || idx.length > 4) { S.toast("Select 2, 3 or 4 atoms to measure", "info"); return; }
    const s = e.structure, atoms = [{ i: idx[0], image: [0, 0, 0] }];
    for (const j of idx.slice(1)) {
      const prev = atoms[atoms.length - 1];
      const from = s.frac[prev.i].map((x, k) => x + prev.image[k]);
      // minImage(from, f_j): the lattice shift that brings j nearest to the atom before it
      atoms.push({ i: j, image: G.minImage(s.cell, from, s.frac[j], s.pbc).image });
    }
    e.measurements.push(make(e, atoms));
    changed(e);
  }
  function clear(e) { e.measurements = []; chain = null; changed(e); }

  /* ---------------------------------------------------------------- the panel */
  function tsv(e) {
    const s = e.structure;
    return ["type\tatoms\tvalue\tunit", ...e.measurements.map((m) => `${m.type}\t${atomsText(s, m)}\t${m.invalid ? "" : G.fmtNum(m.value, m.atoms.length === 2 ? 4 : 3)}\t${m.atoms.length === 2 ? "Å" : "°"}`)].join("\n") + "\n";
  }
  function panel(host, e) {
    if (!e || !e.measurements.length) {
      host.innerHTML = `<p class="empty-note">No measurements. The Measure tool (${esc(S.shortcut("tool.measure"))}): click 2, 3 or 4 atoms; ${esc(S.shortcut("tool.measureSelection"))} measures the selection.</p>`;
      return;
    }
    const s = e.structure;
    host.innerHTML = `<div class="ms-head"><span class="muted small">${e.measurements.length} measurement${e.measurements.length === 1 ? "" : "s"}</span>
        <button class="tb-btn-text ms-copy">Copy as TSV</button><button class="tb-btn-text ms-clear">Clear all</button></div>
      <table class="data ms-table"><thead><tr><th>Type</th><th>Atoms</th><th class="num">Value</th><th></th><th></th></tr></thead><tbody>${e.measurements.map((m, n) =>
        `<tr data-n="${n}" class="${m.invalid ? "invalid" : ""}"><td>${esc(m.type)}</td><td>${esc(atomsText(s, m))}</td>` +
        `<td class="num">${m.invalid ? "atom deleted" : esc(valueText(m))}</td>` +
        `<td><button class="icon-btn sm ms-eye" aria-pressed="${m.visible !== false}" aria-label="${m.visible !== false ? "Hide" : "Show"} this measurement">${S.icon(m.visible !== false ? "eye" : "eye-off")}</button></td>` +
        `<td><button class="icon-btn sm ms-del" aria-label="Delete this measurement">${S.icon("trash-2")}</button></td></tr>`).join("")}</tbody></table>`;
    host.querySelector(".ms-copy").addEventListener("click", () => S.copyText(tsv(e), "Copied the measurements (TSV)"));
    host.querySelector(".ms-clear").addEventListener("click", () => clear(e));
    host.querySelectorAll("tr[data-n]").forEach((tr) => {
      const m = e.measurements[Number(tr.dataset.n)];
      tr.querySelector(".ms-eye").addEventListener("click", () => { m.visible = m.visible === false; changed(e); });
      tr.querySelector(".ms-del").addEventListener("click", () => { e.measurements.splice(e.measurements.indexOf(m), 1); changed(e); });
    });
  }
  return { tool, measureSelection, clear, panel, tsv, startFrom: (a) => { chain = { atoms: [a], m: null, last: a, ed: ed() }; S.toolChip(); },
    get pending() { return chain && chain.atoms ? chain.atoms.length : 0; } };
}
