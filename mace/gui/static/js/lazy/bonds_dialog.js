/* Calculate Bonds… (1.1.6; Build menu, the viewport's menu, Settings › Viewport › Bond tolerance): the
 * bond rule of the drawn structure, as Materials Studio's Calculate Bonds edits it. The rule is the one
 * every bond search in the browser uses (geom/core.js bondRule): the tolerance on the sum of the bond
 * radii, the shortest bond, metal-metal pairs, and per element pair a rule of its own (a length window,
 * or never bonded); with Across cell boundaries (Bonded atoms outside). The preview counts the bonds the
 * rule makes, per cell (the periodic structure) and between the drawn atoms, and lists each pair's count
 * and lengths. Recalculate gives this document those values (its overrides, docprefs "display", saved in
 * gui-docprefs.json); Save as Default writes them to Settings for every document without its own; Reset
 * to Defaults makes this document follow Settings again. The Bond tool's hand edits (display only) are
 * counted and can be cleared here. */
import { loadCss, panel } from "./lazy_util.js";
import * as G from "../geom/core.js";
import * as W from "../geom/client.js";
import * as DS from "../viewport/display_state.js";

let H, dlg = null, doc = null, seq = 0, timer = 0, last = null;
// the bond rule's keys (display_state.js GROUPS.bonds) and Across cell boundaries (the Range group's)
const KEYS = [...DS.GROUPS.bonds, "bondedOutside"];
const SYNC = 5000;                   // atoms: above, the preview's searches run in the geometry worker
const BOUNDARY = [["off", "Drawn atoms only"], ["neighbors", "Add bonded neighbors"], ["molecules", "Complete molecules"]];

/** Open the dialog for the drawn structure. */
export async function open(handle) {
  H = handle;
  await loadCss(import.meta.url, "../../css/dialogs.css");
  doc = H.viewDoc();
  if (!doc || !H.structureOf(doc)) { H.toast("Open a structure first", "info"); return null; }
  const { esc } = H;
  const body = `<p class="bd-doc muted small" id="bd-doc"></p>
    <div class="grid3">
      <label class="f"><span>Tolerance (× Σ r<sub>cov</sub>)</span><input class="field" type="number" min="${G.BOND_TOL.min}" max="${G.BOND_TOL.max}" step="0.01" data-k="bondTolerance"></label>
      <label class="f">Shortest bond (Å)<input class="field" type="number" min="0" max="3" step="0.05" data-k="bondMin"></label>
      <label class="f">Across cell boundaries<select class="field" data-k="bondedOutside">${BOUNDARY.map(([v, t]) => `<option value="${v}">${esc(t)}</option>`).join("")}</select></label>
    </div>
    <label class="chk"><input type="checkbox" data-k="bondMetalMetal"> Bonds between metals in compounds <span class="muted small">(alloys always have them)</span></label>
    <h3 class="bd-h">Element pairs</h3>
    <table class="bd-table bd-rules"><thead><tr><th scope="col">Pair</th><th scope="col">Min (Å)</th><th scope="col">Max (Å)</th><th scope="col">Never</th><th scope="col"><span class="sr-only">Remove</span></th></tr></thead>
      <tbody id="bd-rules"></tbody></table>
    <p class="bd-none muted small" id="bd-none">No pair rules: every pair follows the tolerance.</p>
    <button type="button" class="tb-btn-text sm" data-v="add">${H.icon("plus")}<span>Add Pair</span></button>
    <p class="bd-msg small" id="bd-msg" role="status" aria-live="polite"></p>
    <table class="bd-table bd-found"><caption class="muted small">Bonds in one cell under these criteria</caption>
      <thead><tr><th scope="col">Pair</th><th scope="col">Bonds</th><th scope="col">Shortest (Å)</th><th scope="col">Longest (Å)</th><th scope="col"><span class="sr-only">Rule</span></th></tr></thead>
      <tbody id="bd-found"></tbody></table>
    <p class="bd-manual small" id="bd-manual" hidden><span></span> <button type="button" class="tb-btn-text sm" data-v="clear-manual">Clear Hand Edits</button></p>`;
  dlg = panel(H, { id: "bonds-dialog", title: "Calculate Bonds", body, wide: true,
    foot: `<button class="tb-btn-text" data-v="reset" title="This structure follows Settings › Viewport again">Reset to Defaults</button>
      <button class="tb-btn-text" data-v="default" title="Every structure without its own criteria uses these">Save as Default</button>
      <button class="tb-btn-text primary" data-v="apply">Recalculate</button>` });
  dlg.classList.add("bd-panel");
  fill(DS.of(doc));
  dlg.oninput = dlg.onchange = (e) => { if (e.target.closest("[data-k], [data-r]")) changed(); };
  dlg.onclick = (e) => {
    const b = e.target.closest("button[data-v]");
    if (!b) return;
    const v = b.dataset.v;
    if (v === "add") { addRule({}); changed(); dlg.querySelector("#bd-rules tr:last-child [data-r=a]").focus(); }
    else if (v === "remove") { b.closest("tr").remove(); changed(); }
    else if (v === "rule") { addRule(ruleFromFound(b.dataset.pair)); changed(); }
    else if (v === "apply") apply();
    else if (v === "default") saveDefault();
    else if (v === "reset") reset();
    else if (v === "clear-manual") clearManual();
  };
  // Enter in a field recalculates, as the primary button
  dlg.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.target.matches("input") && !e.isComposing) { e.preventDefault(); apply(); }
  });
  return dlg;
}
export const isOpen = () => !!(dlg && dlg.open);
/** For tests: the last preview {cell, drawn, pairs, rule, errors}. */
export const lastPreview = () => last;

const q = (k) => dlg.querySelector(`[data-k="${k}"]`);

function fill(v) {
  q("bondTolerance").value = String(v.bondTolerance);
  q("bondMin").value = String(v.bondMin);
  q("bondMetalMetal").checked = v.bondMetalMetal === true;
  q("bondedOutside").value = v.bondedOutside || "off";
  dlg.querySelector("#bd-rules").innerHTML = "";
  for (const [name, r] of Object.entries(v.bondPairs || {})) {
    const [a, b] = name.split("-");
    addRule({ a, b, ...r });
  }
  scopeNote();
  changed();
}
function scopeNote() {
  const own = KEYS.some((k) => DS.isOverride(doc, k));
  dlg.querySelector("#bd-doc").textContent = `${doc.title || doc.name || "This structure"}: ${own ? "its own criteria" : "the Settings › Viewport defaults"}`;
}

function addRule(r) {
  const tr = document.createElement("tr");
  const num = (x) => (typeof x === "number" ? String(x) : "");
  tr.innerHTML = `<td class="bd-pair"><input class="field bd-el" data-r="a" aria-label="First element" spellcheck="false" autocomplete="off">
      <span aria-hidden="true">–</span><input class="field bd-el" data-r="b" aria-label="Second element" spellcheck="false" autocomplete="off"></td>
    <td><input class="field" type="number" min="0" max="${G.PAIR_MAX}" step="0.05" data-r="min" aria-label="Shortest bond of the pair (Å)"></td>
    <td><input class="field" type="number" min="0" max="${G.PAIR_MAX}" step="0.05" data-r="max" aria-label="Longest bond of the pair (Å)"></td>
    <td><input type="checkbox" data-r="never" aria-label="Never bond this pair"></td>
    <td><button type="button" class="icon-btn sm" data-v="remove" aria-label="Remove the pair rule">${H.icon("x")}</button></td>`;
  tr.querySelector("[data-r=a]").value = r.a || "";
  tr.querySelector("[data-r=b]").value = r.b || "";
  tr.querySelector("[data-r=min]").value = num(r.min);
  tr.querySelector("[data-r=max]").value = num(r.max);
  tr.querySelector("[data-r=never]").checked = r.never === true;
  dlg.querySelector("#bd-rules").appendChild(tr);
}
/** A rule row for a pair the preview found: its longest bond + 0.05 Å, a start to edit. */
function ruleFromFound(name) {
  const f = last && last.pairs.find((p) => p.pair === name);
  const [a, b] = name.split("-");
  return { a, b, max: f ? Math.round((f.longest + 0.05) * 100) / 100 : undefined };
}

/** The dialog's values: {values (the settings keys), errors}. */
export function values() {
  const errors = [], pairs = {};
  const tol = Number(q("bondTolerance").value), min = Number(q("bondMin").value);
  if (!(tol >= G.BOND_TOL.min && tol <= G.BOND_TOL.max)) errors.push(`Tolerance must be from ${G.BOND_TOL.min} to ${G.BOND_TOL.max}`);
  if (q("bondMin").value === "" || !(min >= 0 && min <= 3)) errors.push("Shortest bond must be from 0 to 3 Å");
  dlg.querySelectorAll("#bd-rules tr").forEach((tr) => {
    const get = (k) => tr.querySelector(`[data-r=${k}]`);
    const a = get("a").value.trim(), b = get("b").value.trim();
    const name = G.pairName(a, b), never = get("never").checked;
    const len = (k) => (get(k).value === "" ? undefined : Number(get(k).value));
    const lo = len("min"), hi = len("max");
    let bad = "";
    if (!a && !b && lo === undefined && hi === undefined && !never) { tr.classList.remove("err"); return; }   // an empty new row
    if (!name) bad = `${a || "?"}–${b || "?"}: not two elements`;
    else if ([lo, hi].some((x) => x !== undefined && !(x >= 0 && x <= G.PAIR_MAX))) bad = `${name}: lengths from 0 to ${G.PAIR_MAX} Å`;
    else if (lo !== undefined && hi !== undefined && hi <= lo) bad = `${name}: Max must be above Min`;
    else if (!never && lo === undefined && hi === undefined) bad = `${name}: give a Min, a Max or Never`;
    else if (name in pairs) bad = `${name}: listed twice`;
    tr.classList.toggle("err", !!bad);
    if (bad) { errors.push(bad); return; }
    pairs[name] = never ? { never: true } : { ...(lo !== undefined ? { min: lo } : {}), ...(hi !== undefined ? { max: hi } : {}) };
  });
  dlg.querySelector("#bd-none").hidden = !!dlg.querySelector("#bd-rules tr");
  return { values: { bondTolerance: tol, bondMin: min, bondMetalMetal: q("bondMetalMetal").checked, bondPairs: pairs,
    bondedOutside: q("bondedOutside").value }, errors };
}

function changed() {
  clearTimeout(timer);
  timer = setTimeout(preview, 80);
}

/** The bonds the dialog's rule makes: per cell (periodic) and between the drawn atoms. */
async function preview() {
  const mine = ++seq, msg = dlg.querySelector("#bd-msg");
  const { values: v, errors } = values();
  manualNote();
  if (errors.length) {
    msg.textContent = errors.join(" · ");
    msg.classList.add("err");
    last = { errors };
    dlg.__preview = last;
    return;
  }
  const s = H.structureOf(doc), get = (k) => (k in v ? v[k] : DS.value(doc, k)), rule = G.bondRule(s, get);
  const frac = Float64Array.from(s.frac.flat()), radii = Float32Array.from(s.radii), metal = Uint8Array.from(s.metal.map(Number));
  const big = s.frac.length > SYNC;
  const per = big ? await W.job("periodicBonds", { cell: s.cell, frac, radii, metal, tol: rule.tol, metalMetal: rule.metalMetal, pbc: s.pbc, rule: G.ruleFor(rule) })
    : G.periodicBonds(s.cell, frac, radii, metal, rule.tol, rule.metalMetal, s.pbc, rule);
  if (mine !== seq || !dlg || per.superseded) return;
  // the drawn atoms, as the scene searches them (its display list)
  const sc = H.editor().E && H.editor().E.scene ? H.editor().E.scene() : null, d = sc && sc.doc && sc.doc.structure === s ? sc.display : null;
  let drawn = null;
  if (d) {
    const dr = new Float32Array(d.n), dm = new Uint8Array(d.n);
    for (let k = 0; k < d.n; k++) { dr[k] = s.radii[d.base[k]]; dm[k] = s.metal[d.base[k]] ? 1 : 0; }
    const r0 = G.ruleFor(rule, d.base);
    const p = d.n > SYNC ? await W.job("bonds", { cart: d.cart.slice(), radii: dr, metal: dm, tol: rule.tol, metalMetal: rule.metalMetal, rule: r0 })
      : { pairs: G.cellListBonds(d.cart, dr, dm, rule.tol, rule.metalMetal, r0) };
    if (mine !== seq || !dlg || p.superseded) return;
    drawn = { bonds: p.pairs.length / 2, atoms: d.n, now: sc.pairs ? sc.pairs.length / 2 : 0 };
  }
  // per pair: count, shortest and longest, in one cell
  const stats = new Map(), C = s.cell;
  for (let k = 0; k < per.i.length; k++) {
    const i = per.i[k], j = per.j[k];
    const df = [0, 1, 2].map((m) => s.frac[j][m] + per.img[3 * k + m] - s.frac[i][m]);
    const len = G.norm3(G.cartOf(C, df)), name = [s.symbols[i], s.symbols[j]].sort().join("-");
    const e = stats.get(name) || { pair: name, count: 0, shortest: Infinity, longest: 0 };
    e.count++; e.shortest = Math.min(e.shortest, len); e.longest = Math.max(e.longest, len);
    stats.set(name, e);
  }
  const pairs = [...stats.values()].sort((a, b) => b.count - a.count || a.pair.localeCompare(b.pair));
  last = { cell: per.i.length, drawn, pairs, values: v, errors: [] };
  msg.classList.remove("err");
  msg.textContent = `${per.i.length.toLocaleString("en-US")} bond${per.i.length === 1 ? "" : "s"} per cell` + (drawn
    ? ` · ${drawn.bonds.toLocaleString("en-US")} between the ${drawn.atoms.toLocaleString("en-US")} drawn atoms (now ${drawn.now.toLocaleString("en-US")})` : "");
  const { esc } = H;
  dlg.querySelector("#bd-found").innerHTML = pairs.length ? pairs.map((p) => `<tr><td>${esc(p.pair.replace("-", "–"))}</td><td>${p.count}</td>
    <td>${p.shortest.toFixed(3)}</td><td>${p.longest.toFixed(3)}</td>
    <td><button type="button" class="tb-btn-text sm" data-v="rule" data-pair="${esc(p.pair)}" aria-label="Add a rule for ${esc(p.pair)}">Rule…</button></td></tr>`).join("")
    : `<tr><td colspan="5" class="muted">No bonds</td></tr>`;
  dlg.__preview = last;                // tests: the dialog element carries it (a module URL may differ)
  dlg.dataset.ready = String(mine);
}

/** Recalculate: this document's criteria, then its bonds are drawn again (the display state redraws). */
function apply() {
  const { values: v, errors } = values();
  if (errors.length) { H.toast(errors[0], "error"); return; }
  DS.set(doc, v);
  scopeNote();
  changed();
}
function saveDefault() {
  const { values: v, errors } = values();
  if (errors.length) { H.toast(errors[0], "error"); return; }
  for (const [k, x] of Object.entries(v)) H.settings.set(k, x);
  DS.reset(doc, KEYS);
  scopeNote();
  H.toast("Bond criteria saved as the default (Settings › Viewport)", "info");
  changed();
}
function reset() {
  DS.reset(doc, KEYS);
  fill(DS.of(doc));
}

/** The Bond tool's hand edits of this document (display only, kept in its draft). */
function manualNote() {
  const ed = H.editor().ed, n = ed && ed.bondOverrides ? ed.bondOverrides.size : 0, el = dlg.querySelector("#bd-manual");
  el.hidden = !n;
  if (n) el.querySelector("span").textContent = `${n} bond${n === 1 ? "" : "s"} added or removed by hand (Tools › Bond) stay on top of these criteria.`;
}
function clearManual() {
  const { ed, E } = H.editor();
  if (!ed || !ed.bondOverrides) return;
  ed.bondOverrides.clear();
  if (E) E.redraw();
  if (H.flushDrafts) H.flushDrafts();
  changed();
}
