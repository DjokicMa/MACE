/* Edit Cell (spec §2.5), View Along [uvw] and a custom display range:
 * non-modal panels, loaded on first use. Edit Cell takes a b c α β γ or a
 * 3x3 matrix, keeps the atoms at their fractional or Cartesian positions,
 * links the parameters the crystal system ties together in Sym mode, shows
 * the volume and a dashed ghost of the new cell, and applies as one undo
 * step. */
import { loadCss, panel } from "./lazy_util.js";

let G = null;
async function geom() {
  if (!G) G = await import("../geom/core.js");
  return G;
}
const css = () => loadCss(import.meta.url, "../../css/dialogs.css");
const fmt = (x, d = 4) => { const s = Number(x).toFixed(d); return /^-0\.?0*$/.test(s) ? s.slice(1) : s; };

/** Which parameters Sym mode ties: groups of linked lengths and angles, and fixed angle values. */
export function constraints(number, lattice) {
  const near = (a, b) => Math.abs(a - b) < 1e-3;
  if (number >= 195) return { system: "cubic", links: [["a", "b", "c"]], fixed: { alpha: 90, beta: 90, gamma: 90 } };
  if (number >= 168) return { system: "hexagonal", links: [["a", "b"]], fixed: { alpha: 90, beta: 90, gamma: 120 } };
  if (number >= 143) {
    if (lattice && near(lattice.alpha, lattice.beta) && near(lattice.beta, lattice.gamma) && !near(lattice.alpha, 90))
      return { system: "trigonal (rhombohedral axes)", links: [["a", "b", "c"], ["alpha", "beta", "gamma"]], fixed: {} };
    return { system: "trigonal", links: [["a", "b"]], fixed: { alpha: 90, beta: 90, gamma: 120 } };
  }
  if (number >= 75) return { system: "tetragonal", links: [["a", "b"]], fixed: { alpha: 90, beta: 90, gamma: 90 } };
  if (number >= 16) return { system: "orthorhombic", links: [], fixed: { alpha: 90, beta: 90, gamma: 90 } };
  if (number >= 3) return { system: "monoclinic", links: [], fixed: { alpha: 90, gamma: 90 } };
  return { system: "triclinic", links: [], fixed: {} };
}

export async function openEditCell(H) {
  await css();
  const g = await geom();
  const { ed, E } = H.editor();
  if (!ed) return null;
  const s = ed.structure, L = s.lattice || g.latticeOf(s.cell), y = s.symmetry || {};
  const sym = ed.symMode === "sym" && y.available;
  const con = sym ? constraints(y.number, L) : { system: "", links: [], fixed: {} };
  const P = ["a", "b", "c", "alpha", "beta", "gamma"], names = { a: "a (Å)", b: "b (Å)", c: "c (Å)", alpha: "α (°)", beta: "β (°)", gamma: "γ (°)" };
  const linkOf = (k) => con.links.find((l) => l.includes(k));
  const body = document.createElement("div");
  body.innerHTML = `<div class="tabs" role="tablist"><button class="tb-btn-text" role="tab" data-tab="par" aria-selected="true">Parameters</button>
      <button class="tb-btn-text" role="tab" data-tab="mat" aria-selected="false">Matrix</button></div>
    <div data-pane="par"><div class="grid3">${P.map((k) => `<label class="f${linkOf(k) ? " link" : ""}">${names[k]}${linkOf(k) ? ` ${H.icon("link-2")}` : ""}
      <input class="field" data-p="${k}" inputmode="decimal" value="${fmt(con.fixed[k] ?? L[k], k.length > 1 ? 3 : 5)}"${k in con.fixed ? " disabled" : ""} aria-label="${names[k]}"></label>`).join("")}</div>
      ${sym ? `<p class="note">Sym mode (${H.esc(y.international)}, ${con.system}): linked parameters change together${Object.keys(con.fixed).length ? ", fixed angles stay" : ""}. P1 mode frees them.</p>` : ""}</div>
    <div data-pane="mat" hidden><div class="grid33">${[0, 1, 2].map((r) => [0, 1, 2].map((c) => `<input class="field" data-m="${r}${c}" inputmode="decimal" value="${fmt(s.cell[r][c], 5)}" aria-label="${"abc"[r]} ${"xyz"[c]}">`).join("")).join("")}</div>
      <p class="note">Rows are a, b and c in Å (Cartesian).</p></div>
    <div class="row"><label class="tog"><input type="radio" name="ec-keep" value="frac" checked><span>Keep atoms at fractional positions</span></label>
      <label class="tog"><input type="radio" name="ec-keep" value="cart"><span>at Cartesian positions</span></label></div>
    <div class="row"><span>Volume</span><b class="ec-vol"></b><span class="err" role="alert"></span></div>`;
  let tab = "par";
  const err = body.querySelector(".err"), vol = body.querySelector(".ec-vol");
  const read = () => {
    if (tab === "mat") {
      const m = [0, 1, 2].map((r) => [0, 1, 2].map((c) => g.evalExpr(body.querySelector(`[data-m="${r}${c}"]`).value)));
      if (m.flat().some((x) => !x.ok)) return { error: "Every matrix entry is a number" };
      const cell = m.map((r) => r.map((x) => x.value));
      if (Math.abs(g.dot3(cell[0], g.cross3(cell[1], cell[2]))) < 1e-6) return { error: "The three vectors are not independent" };
      return { cell, args: { matrix: cell } };
    }
    const v = {};
    for (const k of P) {
      const r = g.evalExpr(body.querySelector(`[data-p="${k}"]`).value);
      if (!r.ok) return { error: `${names[k]}: ${r.error}` };
      v[k] = r.value;
    }
    const cell = g.cellFromParams(v.a, v.b, v.c, v.alpha, v.beta, v.gamma);
    if (!cell) return { error: "Those parameters make no cell" };
    return { cell, args: { cellpar: P.map((k) => v[k]) } };
  };
  const preview = () => {
    const r = read();
    err.textContent = r.error || "";
    if (r.error) { vol.textContent = "—"; E.scene().setGhostCell(null); return; }
    vol.textContent = `${fmt(Math.abs(g.dot3(r.cell[0], g.cross3(r.cell[1], r.cell[2]))), 3)} Å³`;
    E.scene().setGhostCell(r.cell);
  };
  body.addEventListener("input", (e) => {
    const k = e.target.dataset.p, link = k && linkOf(k);
    if (link) link.forEach((o) => { if (o !== k) body.querySelector(`[data-p="${o}"]`).value = e.target.value; });
    preview();
  });
  body.querySelectorAll("[data-tab]").forEach((b) => b.addEventListener("click", () => {
    tab = b.dataset.tab;
    body.querySelectorAll("[data-tab]").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
    body.querySelectorAll("[data-pane]").forEach((p) => { p.hidden = p.dataset.pane !== tab; });
    preview();
  }));
  const d = panel(H, { id: "edit-cell", title: "Edit Cell", body, wide: true,
    foot: `<button class="tb-btn-text" data-act="cancel">Cancel</button><button class="tb-btn-text primary" data-act="apply">Apply</button>`,
    onClose: () => E.scene().setGhostCell(null) });
  d.querySelector('[data-act="cancel"]').addEventListener("click", () => d.close());
  d.querySelector('[data-act="apply"]').addEventListener("click", () => {
    const r = read();
    if (r.error) { err.textContent = r.error; return; }
    const keep = body.querySelector('input[name="ec-keep"]:checked').value;
    E.api.edit(ed, "set_cell", { ...r.args, scale_atoms: keep === "frac" });
    d.close();
  });
  preview();
  return d;
}

export async function openAlongUVW(H) {
  await css();
  const g = await geom();
  const { ed, E } = H.editor();
  if (!ed) return null;
  const cam = E.cam(), before = cam.getView();
  const body = document.createElement("div");
  body.innerHTML = `<p class="note">Look down u·a + v·b + w·c (integers).</p><div class="grid3">${["u", "v", "w"].map((k, n) =>
    `<label class="f">${k}<input class="field" data-k="${k}" inputmode="numeric" value="${n === 2 ? 1 : 0}" aria-label="${k}"></label>`).join("")}</div>
    <div class="err" role="alert"></div>`;
  const read = () => {
    const v = ["u", "v", "w"].map((k) => Number(body.querySelector(`[data-k="${k}"]`).value));
    if (v.some((x) => !Number.isInteger(x)) || v.every((x) => x === 0)) return null;
    return v;
  };
  let applied = false;
  const look = () => {
    const v = read();
    body.querySelector(".err").textContent = v ? "" : "Three integers, not all zero";
    if (!v) return;
    const s = ed.structure, dir = g.alongDirection(s.cell, v);
    cam.along(dir, g.upVector(s.cell, dir)).then(() => H.updateTriad());
  };
  body.addEventListener("input", look);
  const d = panel(H, { id: "along-uvw", title: "View Along [uvw]", body,
    foot: `<button class="tb-btn-text" data-act="cancel">Cancel</button><button class="tb-btn-text primary" data-act="apply">View</button>`,
    onClose: () => { if (!applied) { cam.setView(before); H.updateTriad(); } } });
  d.querySelector('[data-act="cancel"]').addEventListener("click", () => d.close());
  d.querySelector('[data-act="apply"]').addEventListener("click", () => {
    const v = read();
    if (!v) return;
    applied = true;
    H.alongLabel(`[${v.join(" ")}]`);
    d.close();
  });
  look();
  return d;
}

export async function openRangeCustom(H) {
  await css();
  // the document's range, from and to in fractions of a, b and c (Inspector › Range & boundary has the same)
  const cur = H.rangeOf ? H.rangeOf() : { lo: [0, 0, 0], hi: H.range() };
  const body = document.createElement("div");
  const num = (x) => String(+Number(x).toFixed(3));
  body.innerHTML = `<p class="note">From and to along a, b and c, in fractions of the cell (−5 to 10; 0 to 2 draws two cells).</p>` +
    // a row of "from" fields over a row of "to" fields, each one labeled (a from, a to, ...)
    `<div class="grid3">${["lo", "hi"].map((end) => [0, 1, 2].map((n) =>
      `<label class="f">${"abc"[n]} ${end === "lo" ? "from" : "to"}<input class="field" type="number" min="-5" max="10" step="0.1" ` +
      `data-${end}="${n}" value="${num(cur[end][n])}" aria-label="Range ${"abc"[n]} ${end === "lo" ? "from" : "to"}"></label>`).join("")).join("")}</div>` +
    `<p class="set-err" role="alert"></p>`;
  const d = panel(H, { id: "range-custom", title: "Display Range", body,
    foot: `<button class="tb-btn-text" data-act="cancel">Cancel</button><button class="tb-btn-text primary" data-act="apply">Apply</button>` });
  d.querySelector('[data-act="cancel"]').addEventListener("click", () => d.close());
  d.querySelector('[data-act="apply"]').addEventListener("click", () => {
    const lo = [0, 1, 2].map((n) => Number(body.querySelector(`[data-lo="${n}"]`).value));
    const hi = [0, 1, 2].map((n) => Number(body.querySelector(`[data-hi="${n}"]`).value));
    const bad = [...lo, ...hi].some((x) => !Number.isFinite(x) || x < -5 || x > 10) || lo.some((x, k) => hi[k] - x < 0.05);
    if (bad) { body.querySelector(".set-err").textContent = "Each axis: from −5 to 10, the end at least 0.05 above the start"; return; }
    if (H.setRangeLoHi) H.setRangeLoHi(lo, hi); else H.setRange(hi.map((x) => Math.max(1, Math.round(x))));
    d.close();
  });
  return d;
}
