/* Build › Supercell… (spec §2.5, plan T4): n1 n2 n3 (1-10, linked or
 * not) or a 3×3 integer matrix with its determinant, three presets, the
 * atom count and cell lengths as you type, a preview in the viewport (the new
 * cell and the atoms as ghosts; the cell alone above 2000 atoms), and a new
 * tab ("MgO_2x2x2", a session up to 5000 atoms, view-only above) or the
 * change in place (one undo step "Supercell 2×2×2"). */
import { openBuilder, fmtN, stemOf, EDIT_MAX } from "./builder_util.js";

const R45 = [[1, -1, 0], [1, 1, 0], [0, 0, 1]], R30 = [[2, 1, 0], [-1, 1, 0], [0, 0, 1]];
const det3 = (m) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
  + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
const mul = (m, C) => [0, 1, 2].map((i) => [0, 1, 2].map((k) => m[i][0] * C[0][k] + m[i][1] * C[1][k] + m[i][2] * C[2][k]));
const len = (v) => Math.hypot(v[0], v[1], v[2]);
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

/** Is the cell hexagonal (a = b, γ = 120°)? Then √3×√3 R30° is offered. */
function hexagonal(s) {
  const L = s && s.lattice;
  return !!L && Math.abs(L.a - L.b) < 1e-3 * L.a && Math.abs(L.gamma - 120) < 0.5;
}

export function open(H, preset = null) {
  let tab = "diag", linked = true, presetName = null;
  return openBuilder(H, {
    id: "supercell", title: "Supercell", route: "build/supercell", defaultResult: "new", primary: "Create",
    render(host, api) {
      const s = api.structure(), hex = hexagonal(s);
      host.innerHTML = `<div class="bld-tabs" role="tablist" aria-label="Supercell kind">
          <button role="tab" data-tab="diag" aria-selected="true">Diagonal</button><button role="tab" data-tab="matrix" aria-selected="false">Matrix</button></div>
        <div data-pane="diag" class="bld-row"><span>n1 × n2 × n3</span><span class="bld-inline">
          <input type="number" min="1" max="10" step="1" value="2" data-n="0" aria-label="n1">
          <input type="number" min="1" max="10" step="1" value="2" data-n="1" aria-label="n2">
          <input type="number" min="1" max="10" step="1" value="2" data-n="2" aria-label="n3">
          <label class="bld-check" title="The same in all three"><input type="checkbox" data-k="link" checked> Same in all three</label></span></div>
        <div data-pane="matrix" class="bld-row" hidden><span>New a, b, c (rows)</span><span class="bld-matrix">${[0, 1, 2].map((i) => [0, 1, 2].map((j) =>
          `<input type="number" step="1" min="-30" max="30" value="${i === j ? 2 : 0}" data-m="${i}${j}" aria-label="Row ${i + 1}, column ${j + 1}">`).join("")).join("")}</span></div>
        <div class="bld-row" data-pane="matrix" hidden><span>Determinant</span><span class="num" data-k="det"></span></div>
        <div class="bld-row"><span>Presets</span><span class="bld-presets">
          <button type="button" data-preset="222">2×2×2</button><button type="button" data-preset="r45">√2×√2×1 R45°</button>
          ${hex ? '<button type="button" data-preset="r30">√3×√3 R30°</button>' : ""}</span></div>`;
      const showTab = (t) => {
        tab = t;
        host.querySelectorAll("[data-tab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === t)));
        host.querySelectorAll("[data-pane]").forEach((p) => { p.hidden = p.dataset.pane !== t; });
      };
      host.querySelectorAll("[data-tab]").forEach((b) => b.addEventListener("click", () => {
        if (b.dataset.tab === "matrix") setMatrix(host, readDiag(host));
        showTab(b.dataset.tab); presetName = null; api.changed();
      }));
      host.querySelector('[data-k="link"]').addEventListener("change", (e) => { linked = e.target.checked; });
      host.querySelectorAll("[data-n]").forEach((inp) => inp.addEventListener("input", () => {
        presetName = null;
        if (linked) host.querySelectorAll("[data-n]").forEach((o) => { if (o !== inp) o.value = inp.value; });
      }));
      host.querySelectorAll("[data-m]").forEach((inp) => inp.addEventListener("input", () => { presetName = null; }));
      host.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => {
        const p = b.dataset.preset;
        if (p === "222") { showTab("diag"); host.querySelectorAll("[data-n]").forEach((o) => { o.value = "2"; }); presetName = null; }
        else { showTab("matrix"); setMatrix(host, p === "r45" ? R45 : R30); presetName = p; }
        api.changed();
      }));
      api.frame.supercellTab = showTab;
      if (preset && preset.matrix) {
        const m = preset.matrix, diag = [0, 1, 2].every((i) => [0, 1, 2].every((j) => i === j || !m[i][j]));
        if (diag && [0, 1, 2].every((i) => m[i][i] >= 1 && m[i][i] <= 10)) {
          // a diagonal one (the classic panel's n1 n2 n3): the Diagonal tab
          const link = host.querySelector('[data-k="link"]');
          linked = m[0][0] === m[1][1] && m[1][1] === m[2][2];
          link.checked = linked;
          host.querySelectorAll("[data-n]").forEach((el) => { el.value = m[Number(el.dataset.n)][Number(el.dataset.n)]; });
        } else { showTab("matrix"); setMatrix(host, preset.matrix); }
      }
    },
    params(api) {
      const host = api.frame.params, d = api.d, s = api.structure();
      const stem = stemOf(d.title);
      let m, name, what;
      if (tab === "diag") {
        const n = readDiag(host);
        if (n.some((x) => !Number.isInteger(x) || x < 1 || x > 10)) return { error: "n1, n2 and n3 are whole numbers from 1 to 10", name: `${stem}_sc` };
        m = [[n[0], 0, 0], [0, n[1], 0], [0, 0, n[2]]];
        name = `${stem}_${n[0]}x${n[1]}x${n[2]}`; what = `${n[0]}×${n[1]}×${n[2]}`;
      } else {
        m = readMatrix(host);
        if (m.flat().some((x) => !Number.isInteger(x))) return { error: "The matrix holds whole numbers", name: `${stem}_sc` };
        if (m.flat().some((x) => Math.abs(x) > 30)) return { error: "Matrix entries are limited to ±30", name: `${stem}_sc` };
        const det = det3(m);
        host.querySelector('[data-k="det"]').textContent = String(det);
        if (det <= 0) return { error: "The matrix must have a positive determinant", name: `${stem}_sc` };
        const diag = m.every((r, i) => r.every((x, j) => i === j || x === 0));
        if (same(m, R45)) { name = `${stem}_r2xr2x1`; what = "√2×√2×1 R45°"; }
        else if (same(m, R30)) { name = `${stem}_r3xr3x1`; what = "√3×√3×1 R30°"; }
        else if (diag) { name = `${stem}_${m[0][0]}x${m[1][1]}x${m[2][2]}`; what = `${m[0][0]}×${m[1][1]}×${m[2][2]}`; }
        else { name = `${stem}_sc${det}`; what = `det ${det}`; }
      }
      const det = det3(m), n = det * (s ? s.numbers.length : 0);
      if (n > 250000) return { error: `This makes ${fmtN(n)} atoms; builds stop at 250,000`, name };
      const C = mul(m, s.cell), L = C.map(len);
      const warn = [];
      if (n > 500) warn.push(`<p class="warn">CRYSTAL cost grows about as N³ (${fmtN(n)} atoms)</p>`);
      if (n > EDIT_MAX) warn.push(`<p class="warn">Opens view-only (editing stops at ${fmtN(EDIT_MAX)} atoms)</p>`);
      const readout = `<p><b>${fmtN(n)}</b> atoms · a <b>${L[0].toFixed(3)}</b> b <b>${L[1].toFixed(3)}</b> c <b>${L[2].toFixed(3)}</b> Å</p>${warn.join("")}`;
      return { params: { matrix: m, label: what }, n, name, readout, label: `Supercell ${what} of ${d.title}`,
        inPlace: { op: "supercell", args: { matrix: m }, label: `Supercell ${what}` } };
    },
  });
}

function readDiag(host) { return [...host.querySelectorAll("[data-n]")].map((i) => Number(i.value)); }
function readMatrix(host) {
  const m = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  host.querySelectorAll("[data-m]").forEach((i) => { m[+i.dataset.m[0]][+i.dataset.m[1]] = Number(i.value); });
  return m;
}
function setMatrix(host, m) { host.querySelectorAll("[data-m]").forEach((i) => { i.value = String(m[+i.dataset.m[0]][+i.dataset.m[1]]); }); }
