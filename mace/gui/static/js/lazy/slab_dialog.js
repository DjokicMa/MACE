/* Build › Slab from Bulk… (spec §2.5, plan T5): h k l, the number of atomic
 * layers (planes, not ase's surface cells), the vacuum between periodic
 * images, the terminations with a thumbnail of each top plane, Symmetric
 * slab and Center in cell. The preview covers the viewport while the dialog
 * is open (the bulk's own view is left as it is), with the vacuum running vertically. Always a new document: a slab never
 * replaces the bulk. */
import { openBuilder, fmtN, stemOf } from "./builder_util.js";

let pane = null;            // {el, viewer}: reused across openings (one WebGL context)
const NOTE = "Vacuum is ignored by CRYSTAL SLAB (2D); it's used for CIF and plane-wave export only";

const hklName = (h) => h.map((x) => (x < 0 ? `m${-x}` : String(x))).join(Math.max(...h.map(Math.abs)) > 9 ? "_" : "");
const hklText = (h) => h.map(String).join(Math.max(...h.map(Math.abs)) > 9 ? " " : "");

export function open(H) {
  let term = 0, terms = [], lastInfo = null;
  return openBuilder(H, {
    id: "slab", title: "Slab from Bulk", route: "build/slab", results: ["new"], defaultResult: "new",
    onlyNewReason: "A slab never replaces the bulk document", primary: "Create",
    render(host, api) {
      host.innerHTML = `<div class="bld-row"><span>h k l</span><span class="bld-inline">
          ${["h", "k", "l"].map((k, i) => `<input type="number" step="1" min="-12" max="12" value="${i === 2 ? 1 : 0}" data-h="${i}" aria-label="${k}">`).join("")}</span></div>
        <label class="bld-row"><span>Layers</span><input type="number" min="1" max="200" step="1" value="5" data-k="layers" aria-label="Layers (atomic planes)"></label>
        <label class="bld-row"><span>Vacuum (Å)</span><input type="number" min="0" max="200" step="0.5" value="15" data-k="vacuum" aria-label="Vacuum (Å)"></label>
        <p class="bld-note">${NOTE}</p>
        <div class="bld-row" style="align-items:start"><span>Termination</span><ul class="bld-list" role="listbox" aria-label="Termination" data-k="terms"><li class="bld-empty">…</li></ul></div>
        <div class="bld-row"><span></span><span class="bld-inline">
          <label class="bld-check"><input type="checkbox" data-k="symmetric"> Symmetric slab</label>
          <label class="bld-check"><input type="checkbox" data-k="center" checked> Center in cell</label></span></div>`;
      host.querySelector('[data-k="terms"]').addEventListener("click", (e) => {
        const li = e.target.closest("[data-t]");
        if (!li) return;
        term = Number(li.dataset.t);
        renderTerms(host, terms, term, H);
        api.changed();
      });
      host.querySelector('[data-k="terms"]').addEventListener("keydown", (e) => {
        if (!["ArrowDown", "ArrowUp"].includes(e.key) || !terms.length) return;
        e.preventDefault();
        term = (term + (e.key === "ArrowDown" ? 1 : terms.length - 1)) % terms.length;
        renderTerms(host, terms, term, H);
        api.changed();
      });
    },
    params(api) {
      const host = api.frame.params, d = api.d, s = api.structure();
      if (s && s.pbc && !s.pbc.every(Boolean)) return { error: "Slabs are cut from 3D bulk cells", name: `${stemOf(d.title)}_slab` };
      const h = [...host.querySelectorAll("[data-h]")].map((i) => Number(i.value));
      const layers = Number(host.querySelector('[data-k="layers"]').value), vacuum = Number(host.querySelector('[data-k="vacuum"]').value);
      const symmetric = host.querySelector('[data-k="symmetric"]').checked, center = host.querySelector('[data-k="center"]').checked;
      const name = `${stemOf(d.title)}_${h.every(Number.isInteger) ? hklName(h) : "hkl"}_${layers}L`;
      if (!h.every(Number.isInteger)) return { error: "h k l are three whole numbers", name };
      if (h.every((x) => x === 0)) return { error: "h k l can't all be zero", name };
      if (!(Number.isInteger(layers) && layers >= 1 && layers <= 200)) return { error: "Layers: a whole number from 1 to 200", name };
      if (!(vacuum >= 0 && vacuum <= 200)) return { error: "Vacuum: 0 to 200 Å", name };
      const per = lastInfo && lastInfo.layers ? `${lastInfo.layers}` : String(layers);
      return { params: { hkl: h, layers, vacuum, termination: term, symmetric, center },
        name, label: `Slab (${hklText(h)}), ${per} layers of ${d.title}`, readout: readout(lastInfo, layers, symmetric) };
    },
    drawPreview(r, api, core) {
      lastInfo = r.info ? { ...r.info, n: r.n } : null;
      if (r.info && r.info.terminations) {
        terms = r.info.terminations;
        if (term >= terms.length) term = 0;
        renderTerms(api.frame.params, terms, term, H);
      }
      api.frame.readout.innerHTML = readout(lastInfo, Number(api.frame.params.querySelector('[data-k="layers"]').value),
        api.frame.params.querySelector('[data-k="symmetric"]').checked);
      if (r.preview) drawPane(H, core.decodePacked(r.preview), r.info, core, api.d.title);
      return { kind: "slab" };           // nothing over the bulk itself
    },
    onOpen() { ensurePane(H); },
    onClose() { closePane(H); },
  });
}

function readout(info, layers, symmetric) {
  if (!info) return `<p>${fmtN(layers)} layers</p>`;
  const sym = symmetric && info.layers !== info.asked_layers ? `<p class="warn">${info.layers} layers for a symmetric slab</p>` : "";
  return `<p><b>${fmtN(info.layers)}</b> layers · <b>${fmtN(info.n || 0)}</b> atoms · thickness <b>${info.thickness.toFixed(3)}</b> Å · c <b>${info.c.toFixed(3)}</b> Å</p>${sym}`;
}

/* One small picture per termination: the top plane's atoms in their element
 * colors, from the in-plane fractional coordinates (no extra 3D viewer). */
function thumb(H, t) {
  const M = window.MACE, th = t.thumb, [a, b] = th.ab;
  const pts = [];
  for (const [u, v] of th.frac) for (const [du, dv] of [[0, 0], [1, 0], [0, 1], [1, 1]]) pts.push([(u + du) * a[0] + (v + dv) * b[0], (u + du) * a[1] + (v + dv) * b[1]]);
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  // the in-plane cell (four images of each atom) inside a 64-unit box, its balls never cut by the edge
  const sc = 44 / Math.max(x1 - x0, y1 - y0, 1e-6), r = Math.max(2.5, Math.min(6, sc * 0.6));
  const circles = [];
  th.frac.forEach(([u, v], k) => {
    for (const [du, dv] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
      const x = (u + du) * a[0] + (v + dv) * b[0], y = (u + du) * a[1] + (v + dv) * b[1];
      const el = SYMS[th.numbers[k] % 100 || th.numbers[k]] || "X";
      circles.push(`<circle cx="${(10 + (x - x0) * sc).toFixed(1)}" cy="${(54 - (y - y0) * sc).toFixed(1)}" r="${r.toFixed(1)}" fill="${M.elementColor(el)}"/>`);
    }
  });
  return `<svg class="thumb" viewBox="0 0 64 64" aria-hidden="true">${circles.join("")}</svg>`;
}
const SYMS = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag " +
  "Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U").split(" ");

function renderTerms(host, terms, sel, H) {
  const ul = host.querySelector('[data-k="terms"]');
  if (!terms.length) { ul.innerHTML = `<li class="bld-empty">No atomic planes along this direction</li>`; return; }
  ul.innerHTML = terms.map((t, i) => `<li role="option" data-t="${i}" tabindex="${i === sel ? 0 : -1}" aria-selected="${i === sel}">${thumb(H, t)}
    <span class="bld-term"><span>${H.esc(t.name || t.formula)}-terminated</span><span class="sub">${t.atoms_per_plane} atoms per plane · ${t.gap_below.toFixed(2)} Å to the plane below</span></span></li>`).join("");
}

/* ---------------------------------------------------------------- the preview pane */
function ensurePane(H) {
  const M = window.MACE, main = document.getElementById("viewer-main"), wrap = main.closest(".viewer-wrap");
  wrap.classList.add("has-preview-pane");
  if (!pane) {
    const el = document.createElement("div");
    el.className = "preview-pane";
    el.innerHTML = `<div class="pp-viewer" id="viewer-slab-preview"></div><div class="pp-label" aria-live="polite">Preview</div>`;
    pane = { el, viewer: null };
  }
  // over the bulk's viewer, not beside it: the bulk keeps its size and its view
  if (!pane.el.isConnected) main.after(pane.el);
  pane.el.hidden = false;
  if (!pane.viewer) pane.viewer = M.viewer("viewer-slab-preview");
}
function closePane(H) {
  if (!pane) return;
  const wrap = pane.el.closest(".viewer-wrap");
  if (wrap) wrap.classList.remove("has-preview-pane");
  pane.el.remove();
  if (pane.viewer) { pane.viewer.removeAllModels(); pane.viewer.removeAllShapes(); pane.viewer.removeAllLabels(); }
  window.__maceSlabPane = null;
}
/* The slab at the standard view (c up, so the vacuum runs vertically), with
 * its cell box filling 85% of the part of the pane the dialog leaves free.
 * Returns that free width in pixels. */
function fitPane(v, s, core) {
  v.resize();                                    // the pane may have been 0×0 when its viewer was made
  const C = s.cell, pts = s.cart.map((p) => p.slice());
  for (const i of [0, 1]) for (const j of [0, 1]) for (const k of [0, 1]) pts.push([0, 1, 2].map((x) => i * C[0][x] + j * C[1][x] + k * C[2][x]));
  const m = v.addModel();
  m.addAtoms(pts.map((p, k) => ({ elem: "H", x: p[0], y: p[1], z: p[2], index: k, bonds: [] })));
  m.setStyle({}, {});
  v.zoomTo({ model: m });
  v.removeModel(m);
  const q = core.standardView(s.cell, [true, true, true]);
  if (q) { const view = v.getView(); v.setView([...view.slice(0, 4), ...q]); }
  const cv = v.getCanvas(), rc = cv.getBoundingClientRect(), W = cv.clientWidth, Hh = cv.clientHeight;
  const dlg = document.querySelector('dialog.builder-dialog[data-builder="slab"][open]');
  const dr = dlg ? dlg.getBoundingClientRect() : null;
  const free = dr && dr.left > rc.left + 0.3 * W && dr.left < rc.right && dr.bottom > rc.top ? dr.left - rc.left - 8 : W;
  if (!(W > 0 && Hh > 0)) return free;
  const xyz = pts.map((p) => ({ x: p[0], y: p[1], z: p[2] }));
  const box = () => {
    // 3Dmol updates its size and matrices when it paints (once per frame here): paint this view now
    if (!(v.paintNow && v.paintNow())) { try { if (v.setSlabAndFog) v.setSlabAndFog(); if (v.scene && v.scene.updateMatrixWorld) v.scene.updateMatrixWorld(true); } catch (e) { /* not drawn yet */ } }
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const p of v.modelToScreen(xyz)) {
      const x = p.x - rc.left, y = p.y - rc.top;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    return { x0, x1, y0, y1 };
  };
  for (let pass = 0; pass < 3; pass++) {
    let b = box();
    if (!(b.x1 > b.x0 && b.y1 > b.y0)) break;
    const f = 0.85 * Math.min(free / (b.x1 - b.x0), Hh / (b.y1 - b.y0));
    if (Math.abs(f - 1) > 0.005) v.zoom(f);
    b = box();
    const dx = free / 2 - (b.x0 + b.x1) / 2, dy = Hh / 2 - (b.y0 + b.y1) / 2;
    if (Math.hypot(dx, dy) > 0.5) v.translateScene(dx, dy);
    if (Math.abs(f - 1) < 0.01 && Math.hypot(dx, dy) < 1) break;
  }
  v.render();
  return free;
}
function drawPane(H, s, info, core, parent) {
  if (!pane || !pane.el.isConnected) return;
  const M = window.MACE, v = pane.viewer;
  if (!v) return;                   // no WebGL: the preview pane explains it
  v.removeAllModels(); v.removeAllShapes(); v.removeAllLabels();
  const atoms = s.cart.map((p, i) => ({ i, cart: p, image: false }));
  M.addModel(v, s, atoms, { bonds: true, style: "ballstick" });
  M.drawCell(v, s, M.cssVar("--cell"), 0.03);
  const free = fitPane(v, s, core);
  const h = info && info.hkl ? hklText(info.hkl) : "";
  const lab = pane.el.querySelector(".pp-label");
  lab.textContent = `Preview: ${parent.replace(/\.[^.]+$/, "")} (${h}), ${info ? info.layers : ""} layers, ${info ? info.top_name || info.top : ""}-terminated`;
  lab.style.maxWidth = `${Math.max(120, free - 24)}px`;
  window.__maceSlabPane = { n: s.numbers.length, label: pane.el.querySelector(".pp-label").textContent };
}
