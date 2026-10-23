/* Coordination analysis (plan B9): the bottom panel's Coordination tab and the
 * inspector's "CN 6 (O×6, 2.106 Å)" line. The selected atoms, or one row per
 * symmetry-unique site when nothing is selected (with its Wyckoff letter).
 * Up to 2000 atoms the bond graph is built here (once per structure); above,
 * the rows come from the geometry worker. */
import * as G from "../geom/core.js";
import * as W from "../geom/client.js";

const SYNC = 2000;
const cache = new WeakMap();          // structure -> {key, adj}
/* the bond rule (geom/core.js bondRule) beyond its tolerance: its minimum and pair rules */
/** The document's bond rule (the shell passes its effective settings), as the viewer draws the bonds. */
export const ruleOf = (s, get) => G.bondRule(s, get);
const ruleKey = (tol, rule) => `${tol}|${rule ? `${rule.min}|${JSON.stringify([...rule.pairs])}` : ""}`;

function inputs(s) {
  return { z: Uint8Array.from(s.symbols.map((x) => G.zOf(x))), frac: Float64Array.from(s.frac.flat()),
    radii: Float32Array.from(s.radii), metal: Uint8Array.from(s.metal.map(Number)) };
}
/** Rows for `sites` (atom indices), synchronously; null above 2000 atoms. */
export function rowsSync(s, sites, tol, rule = null) {
  if (!s || !s.radii || s.frac.length > SYNC) return null;
  let c = cache.get(s);
  const inp = inputs(s), key = ruleKey(tol, rule);
  if (!c || c.key !== key) { c = { key, adj: G.bondedShells(s.cell, inp.frac, inp.radii, inp.metal, tol, s.pbc, G.ruleFor(rule)) }; cache.set(s, c); }
  return G.coordination(s.cell, inp.frac, inp.z, inp.radii, inp.metal, tol, sites, c.adj, s.pbc);
}
export async function rows(s, sites, tol, rule = null) {
  const r = rowsSync(s, sites, tol, rule);
  if (r) return r;
  const inp = inputs(s);
  const res = await W.job("coordination", { cell: s.cell, ...inp, tol, sites, pbc: s.pbc, rule: G.ruleFor(rule) });
  return res.superseded ? null : res.rows;
}
const sym = (z) => G.SYMBOLS[z] || "X";
/** "CN 6 (O×6, 2.106 Å)" */
export function cnLine(r) {
  if (!r) return "—";
  if (!r.cn) return "CN 0 (no bonded neighbors)";
  return `CN ${r.cn} (${r.neighbors.map((x) => `${sym(x.z)}×${x.count}`).join(", ")}, ${r.nearest.toFixed(3)} Å)`;
}
/** The sites a table lists: the selection, or the first atom of each symmetry orbit. */
export function sitesOf(s, selection) {
  if (selection && selection.size) return [...selection].sort((a, b) => a - b);
  const y = s.symmetry;
  if (y && y.available && y.equivalent_atoms) return [...new Set(y.equivalent_atoms)].sort((a, b) => a - b);
  return s.symbols.map((_, i) => i);
}

/** The Coordination tab. deps: {esc, wyckoff(s, i), tol(), toast} */
export function createCoordPanel(host, deps) {
  let shown = { rows: [], s: null, key: "" };
  host.innerHTML = `<p class="empty-note coord-empty">Open a structure to see its coordination.</p>
    <div class="coord-head" hidden><span class="coord-what muted"></span><button class="btn small coord-copy" type="button">Copy TSV</button></div>
    <div class="table-scroll"><table class="data coord-table" role="grid" hidden></table></div>`;
  const table = host.querySelector(".coord-table"), head = host.querySelector(".coord-head"), empty = host.querySelector(".coord-empty");
  const COLS = ["site", "Wyckoff", "CN", "ECoN", "nearest Å", "mean Å", "neighbors", "shells"];
  function tableRows() {
    const s = shown.s;
    return shown.rows.map((r) => ({
      site: `${s.symbols[r.i]}${r.i + 1}`, Wyckoff: deps.wyckoff(s, r.i) || "", CN: r.cn, ECoN: +r.econ.toFixed(3),
      "nearest Å": +r.nearest.toFixed(4), "mean Å": +r.mean.toFixed(4),
      neighbors: r.neighbors.map((x) => `${sym(x.z)}×${x.count}`).join(" "),
      shells: r.shells.map((x) => `${x.count} ${sym(x.z)} ${x.d.toFixed(3)}`).join("; "),
    }));
  }
  async function render(ed) {
    const s = ed && ed.structure;
    if (!s) { empty.hidden = false; head.hidden = true; table.hidden = true; shown = { rows: [], s: null, key: "" }; return; }
    const sites = sitesOf(s, ed.selection), rule = deps.rule ? deps.rule(s) : null, tol = rule ? rule.tol : deps.tol(s);
    const key = `${sites.join(",")}|${ruleKey(tol, rule)}`;
    if (shown.s === s && shown.key === key) return;
    const r = await rows(s, sites, tol, rule);
    if (!r) return;
    shown = { rows: r, s, key };
    empty.hidden = true; head.hidden = false; table.hidden = false;
    head.querySelector(".coord-what").textContent = ed.selection && ed.selection.size
      ? `${sites.length} selected atom${sites.length === 1 ? "" : "s"}` : `${sites.length} symmetry-unique site${sites.length === 1 ? "" : "s"} (select atoms to list them instead)`;
    const tr = tableRows();
    table.setAttribute("aria-rowcount", String(tr.length + 1));
    table.innerHTML = `<thead><tr aria-rowindex="1">${COLS.map((c) => `<th class="${["CN", "ECoN", "nearest Å", "mean Å"].includes(c) ? "num" : ""}">${deps.esc(c)}</th>`).join("")}</tr></thead><tbody>` +
      tr.map((row, k) => `<tr tabindex="${k ? -1 : 0}" aria-rowindex="${k + 2}">${COLS.map((c) => `<td class="${typeof row[c] === "number" ? "num" : ""}">${deps.esc(String(row[c]))}</td>`).join("")}</tr>`).join("") + "</tbody>";
  }
  host.addEventListener("keydown", (e) => {
    const tr = e.target.closest && e.target.closest("tbody tr");
    if (!tr) return;
    const all = [...tr.parentElement.children], i = all.indexOf(tr);
    const to = { ArrowDown: i + 1, ArrowUp: i - 1, PageDown: i + 10, PageUp: i - 10, Home: 0, End: all.length - 1 }[e.key];
    if (to === undefined) return;
    e.preventDefault();
    const j = Math.max(0, Math.min(all.length - 1, to));
    all.forEach((r, k) => r.setAttribute("tabindex", k === j ? "0" : "-1"));
    all[j].focus();
  });
  host.querySelector(".coord-copy").addEventListener("click", () => deps.copy(tsv(), "Table copied"));
  function tsv() {
    return [COLS.join("\t")].concat(tableRows().map((r) => COLS.map((c) => r[c]).join("\t"))).join("\n");
  }
  function exportTable(fmt) {
    const rowsT = tableRows();
    if (!rowsT.length) return null;
    const stem = "coordination";
    if (fmt === "json") return { name: `${stem}.json`, blob: new Blob([JSON.stringify({ table: "coordination", columns: COLS, rows: rowsT }, null, 1)], { type: "application/json" }) };
    const cell = (v) => { const x = String(v); return /[",\n]/.test(x) ? `"${x.replace(/"/g, '""')}"` : x; };
    return { name: `${stem}.csv`, blob: new Blob([[COLS.map(cell).join(",")].concat(rowsT.map((r) => COLS.map((c) => cell(r[c])).join(","))).join("\n") + "\n"], { type: "text/csv" }) };
  }
  return { render, exportTable, rows: () => tableRows(), invalidate() { shown.key = ""; } };
}
