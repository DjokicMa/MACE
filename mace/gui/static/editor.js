/* MACE GUI: the classic structure-editor panel ("More editing (classic)" in
 * the inspector of a structure document). It keeps its operations, buttons,
 * Clean flow, bond-pair limits and Deck preview; the document it acts on is
 * the active structure document, and every operation goes through that
 * document's request queue (js/editor), so it shares the undo history,
 * selection and drawing of the editor. Phase 4 replaces its Supercell,
 * Primitive, Symmetrize and Clean controls with the builder dialogs. */
(() => {
"use strict";

const M = () => window.MACE;
const $ = (s) => document.querySelector(s);
const E = () => window.__maceEd || null;          // the editor module, once loaded
const ed = () => (E() ? E().active() : null);     // the active structure document's state
let lastClean = null;

function setStatus(text, err = false) {
  const el = $("#e-status");
  if (!el) return;
  el.textContent = text || "";
  el.classList.toggle("neg", err);
}
function fail(e) { setStatus(e.message, true); M().toast(e.message, true); }

/* ---------------------------------------------------------------- the panel */
function render() {
  const d = ed();
  if (!d || !$("#e-title")) return;
  const s = d.structure, y = s.symmetry || {}, esc = M().esc;
  $("#e-title").textContent = d.name;
  $("#e-title").title = d.path || d.name;
  $("#e-formula").innerHTML = `${esc(s.reduced_formula || s.formula || "")} <span class="zbadge">Z = ${s.z ?? "?"}</span>`;
  $("#e-sym").innerHTML = y.available
    ? `<span class="hm" title="${esc(y.international)}">${M().hm(y.international)}</span> <span class="sgnum">No. ${y.number}</span>`
    : `<span class="muted">${esc(y.note || "")}</span>`;
  $("#e-counts").textContent = `${s.numbers.length} atoms · ${d.selection.size} selected` +
    (d.symMode === "sym" ? ` · symmetry-aware${d.ops ? ` (${d.ops.rotations.length} operations)` : ""}` : " · P1 editing");
  $("#e-undo").disabled = !d.canUndo;
  $("#e-redo").disabled = !d.canRedo;
  $("#e-locked").checked = d.symMode === "sym";
  $("#e-symprec").value = d.symprec;
  const L = s.lattice;
  if (L && document.activeElement && !document.activeElement.closest(".grid3")) {
    ["a", "b", "c", "alpha", "beta", "gamma"].forEach((k) => { $("#e-" + k).value = L[k].toFixed(5); });
  }
  if (s.op_summary && d.lastSummary !== s.op_summary) { d.lastSummary = s.op_summary; setStatus(s.op_summary); }
  const sel = [...d.selection].sort((a, b) => a - b);
  $("#e-selected").innerHTML = sel.length ? sel.slice(0, 40).map((i) =>
    `<span class="badge" title="(${s.frac[i].map((x) => x.toFixed(4)).join(", ")})">${s.symbols[i]}${i + 1}</span>`).join(" ")
    + (sel.length > 40 ? " …" : "") : '<span class="muted small">Click atoms to select; Shift+click adds.</span>';
  $("#e-bondlist").textContent = Object.entries(d.bondMax).map(([p, x]) => `${p} ≤ ${x} Å`).join(", ") || "none";
}

/* ---------------------------------------------------------------- operations */
async function op(name, args = {}) {
  const d = ed();
  if (!d) return;
  try {
    await E().api.edit(d, name, args);
    $("#c-result").classList.add("hidden");   // its "Undo Clean" now means another edit
  } catch (e) { fail(e); }
}
const selected = () => { const d = ed(); return d ? [...d.selection].sort((a, b) => a - b) : []; };
const num = (id) => Number($(id).value);
const vec = (ids) => ids.map((id) => num(id));
function element() {
  const v = $("#e-element").value.trim();
  if (!v) throw new Error("Type an element symbol first.");
  return v;
}
function bind(id, fn) {
  const el = $(id);
  if (el) el.addEventListener("click", () => { try { const r = fn(); if (r && r.catch) r.catch(fail); } catch (e) { M().toast(e.message, true); } });
}

bind("#e-add", () => {
  const c = vec(["#e-x", "#e-y", "#e-z"]);
  return op("add", $("#e-coord").value === "cart" ? { element: element(), cart: c } : { element: element(), frac: c });
});
bind("#e-fill", () => {
  const d = ed(), sel = selected();
  if (!d || !sel.length) throw new Error("Select atoms to take their center.");
  const key = $("#e-coord").value === "cart" ? "cart" : "frac";
  const c = [0, 1, 2].map((k) => sel.reduce((t, i) => t + d.structure[key][i][k], 0) / sel.length);
  ["#e-x", "#e-y", "#e-z"].forEach((id, k) => { $(id).value = c[k].toFixed(5); });
});
bind("#e-delete", () => op("delete", { indices: selected() }));
bind("#e-replace", () => op("replace", { indices: selected(), element: element() }));
bind("#e-move", () => op("move", { indices: selected(), delta: vec(["#e-dx", "#e-dy", "#e-dz"]) }));
bind("#e-undo", () => ed() && E().api.undo(ed()));
bind("#e-redo", () => ed() && E().api.redo(ed()));
bind("#e-setcell", () => op("set_cell", { cellpar: vec(["#e-a", "#e-b", "#e-c", "#e-alpha", "#e-beta", "#e-gamma"]),
  scale_atoms: $("#e-scale").checked }));
// Phase 4 (plan T12): the classic Supercell, Primitive and Symmetrize buttons
// open the builder dialogs (their labels stay); Supercell starts from n1 n2 n3
bind("#e-supercell", () => {
  const n = vec(["#e-n1", "#e-n2", "#e-n3"]).map((x) => Math.max(1, Math.round(x || 1)));
  window.__mace.run("build.supercell", { matrix: [[n[0], 0, 0], [0, n[1], 0], [0, 0, n[2]]] });
});
bind("#e-primitive", () => window.__mace.run("build.transform", { kind: "primitive" }));
bind("#e-symmetrize", () => window.__mace.run("build.symmetrize"));
bind("#e-settol", () => ed() && E().editing.setSymprec(ed(), num("#e-symprec")));
$("#e-locked") && $("#e-locked").addEventListener("change", (e) => {
  const d = ed();
  if (d) (e.target.checked ? E().api.toSym(d) : E().api.setMode(d, "p1"));
});
$("#e-boundary") && $("#e-boundary").addEventListener("change", (e) => {
  if (M().setting) window.__mace.settings.set("showBoundary", e.target.checked);
});
bind("#e-bondset", () => {
  const d = ed();
  if (!d) return;
  const pair = $("#e-bondpair").value.trim(), max = Number($("#e-bondmax").value);
  const m = pair.match(/^([A-Za-z]{1,2})\s*-\s*([A-Za-z]{1,2})$/);
  if (!m) throw new Error("Write the pair as El-El, e.g. Si-O.");
  const norm = (x) => x[0].toUpperCase() + x.slice(1).toLowerCase();
  const k = [norm(m[1]), norm(m[2])].sort().join("-");
  if (max > 0) d.bondMax[k] = max; else delete d.bondMax[k];
  render();
  E().redraw();
});

/* ---------------------------------------------------------------- Clean */
// Phase 4: Clean opens the Clean Geometry panel (a proposal to accept or
// discard, spec §2.9); the label stays. The commit-at-once flow below is kept
// for the old route's clients (POST /api/edit/clean is unchanged).
bind("#e-clean", () => window.__mace && window.__mace.run("build.clean"));
async function classicClean() {
  const d = ed();
  if (!d) return;
  const btn = $("#e-clean");
  btn.disabled = true;
  setStatus("Clean running…");
  try {
    const r = await E().editing.request(d, "clean", "edit/clean",
      () => ({ keep_symmetry: $("#c-sym").checked, relax_cell: $("#c-cell").checked,
        bonds: window.__mace && window.__mace.bondValues ? window.__mace.bondValues() : undefined,
        bonds_held: window.__mace && window.__mace.cleanBondsHeld ? window.__mace.cleanBondsHeld() : undefined }), { changes: true, payloadKey: "structure" });
    if (!r) return;
    if (r.nothing_to_clean) { setStatus(r.message); return; }
    lastClean = r;
    const c = r.comparison;
    setStatus(r.structure.op_summary);
    $("#c-result").classList.remove("hidden");
    $("#c-summary").innerHTML = `max move <b>${c.max_displacement.toFixed(3)} Å</b>, rms ${c.rms_displacement.toFixed(3)} Å,
      ${r.changed_bonds.length} bond type${r.changed_bonds.length === 1 ? "" : "s"} changed by more than 0.01 Å;
      ${r.report.converged ? "converged" : "not converged"} in ${r.report.steps} steps.`;
  } catch (e) { fail(e); }
  finally { btn.disabled = false; }
}
window.__maceClassicClean = classicClean;
bind("#c-accept", () => { $("#c-result").classList.add("hidden"); setStatus("Clean kept (Undo still reverts it)"); });
bind("#c-undo", () => { $("#c-result").classList.add("hidden"); return ed() && E().api.undo(ed()); });
bind("#c-compare", () => {
  const r = lastClean, d = ed();
  if (!r || !d) return;
  M().setTrajectory({ frames: [r.before, r.structure], energies_au: [null, null],
    labels: ["before Clean", "after Clean"], comparison: r.comparison },
    `${d.name}: Clean (rough spring pre-relaxation)`, "before → after Clean; not a physical relaxation");
  document.dispatchEvent(new CustomEvent("mace-clean-compare"));   // the shell gives it a tab
  M().showTab("opt");
});

/* ---------------------------------------------------------------- CIF -> Deck */
function cifName() {
  const d = ed();
  let n = $("#e-cifname").value.trim() || (d ? d.name.replace(/\.[^.]+$/, "") + "_edited.cif" : "edited.cif");
  if (!n.toLowerCase().endsWith(".cif")) n += ".cif";
  $("#e-cifname").value = n;
  return n;
}
// Save CIF: the one save path (File › Save As…, plan critique 6), with this
// card's name and CIF kind filled in
let savedListener = null;
bind("#e-save", () => {
  const d = ed();
  if (!d) return;
  const name = cifName(), mode = $("#e-cifmode").value;
  if (savedListener) document.removeEventListener("mace-saved", savedListener);    // a canceled Save As before
  const done = savedListener = async (ev) => {
    document.removeEventListener("mace-saved", done);
    savedListener = null;
    const saved = ev.detail && ev.detail.r && ev.detail.r.saved;
    if (!saved) return;
    setStatus(`Saved ${saved}. Calculation Setup (${window.__mace ? window.__mace.shortcut("calc.setup") : "Mod+Shift+R"}) sets up its deck.`);
    await M().loadFiles();
  };
  document.addEventListener("mace-saved", done);
  window.__mace.run("file.saveAs", { preset: { name, format: "cif", cifMode: mode } });
});
bind("#e-deck", () => {
  // the edited structure's deck: Calculation Setup on this document (its preview uses the
  // structure as a CIF, never written into the folder)
  if (!ed()) return;
  window.__mace.run("calc.setup", { open: true });
  window.__mace.run("calc.preview");
});

/* ---------------------------------------------------------------- start editing / new */
bind("#s-edit", () => window.__mace && window.__mace.run("build.editStructure"));
bind("#e-new", async () => {
  try {
    const sites = $("#n-sites").value.split("\n").map((l) => l.trim()).filter(Boolean).map((l) => {
      const p = l.split(/[\s,]+/);
      if (p.length < 4) throw new Error(`site line "${l}": write El x y z`);
      return { element: p[0], frac: p.slice(1, 4).map(Number) };
    });
    const sg = $("#n-sg").value.trim();
    const r = await M().api("edit/new", { body: { name: $("#n-name").value.trim() || "new_cell",
      cellpar: vec(["#n-a", "#n-b", "#n-c", "#n-alpha", "#n-beta", "#n-gamma"]),
      spacegroup: sg ? Number(sg) : null, setting: Number($("#n-setting").value), sites } });
    window.__mace.openUntitled(r);
  } catch (e) { M().toast(e.message, true); }
});

const SYMBOLS = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split(" ");
$("#e-elements").innerHTML = SYMBOLS.map((x) => `<option value="${x}">`).join("");

document.addEventListener("mace-doc", render);
document.addEventListener("mace-selection", render);
/* The old editor's API, over the active structure document. */
window.MaceEditor = {
  undo: () => (ed() ? E().api.undo(ed()) : Promise.resolve()), redo: () => (ed() ? E().api.redo(ed()) : Promise.resolve()),
  canUndo: () => !!(ed() && ed().canUndo), canRedo: () => !!(ed() && ed().canRedo),
  selected: () => (ed() ? ed().selection.size : 0), atoms: () => (ed() ? ed().n : 0),
  structure: () => (ed() ? ed().structure : null),
  openFile: (path) => window.__mace.open(path).then(() => window.__mace.run("build.editStructure")),
};
window.MaceClassic = { render };
})();
