/* The edit tools of the viewport (spec §2.1, plan T7):
 *   Move (G): a Blender-style grab. A drag on the selection, or G with a
 *     selection, makes it follow the pointer in the screen plane; X/Y/Z lock
 *     a Cartesian axis, Shift+X/Y/Z a cell axis, typed digits give the offset
 *     in Å, Ctrl snaps to 0.05 of a cell axis; Enter or a click applies,
 *     Esc or a right click cancels. A plain drag with no key applies on
 *     release. In Sym mode the whole orbit moves; leaving a special site
 *     offers "Keep <group>" (the move projected onto what the site allows)
 *     or "Allow". The live space group comes from /api/symmetry, at most one
 *     call in flight, 150 ms after the last change, the newest state wins.
 *   Add Atom (A): a click adds the current element on the plane through the
 *     cell's center facing the camera; Shift+click on an atom adds one bonded
 *     to it, away from its neighbors, at the sum of the covalent radii.
 *   Replace (E): a click replaces the atom (its orbit in Sym); a drag paints,
 *     one history entry.
 *   Bond (B): click two atoms to add or remove a bond, for display only. */
import * as G from "../geom/core.js";
import * as profile from "../profile.js";

const SNAP = 0.05;

export function createTools(E, S, V) {
  const recent = () => { const r = profile.ui("recentElements", []); return Array.isArray(r) ? r : []; };
  let element = G.SYMBOLS[G.zOf(S.setting("defaultElement"))] || "C";
  const setElement = (el) => {
    element = el;
    profile.setUi("recentElements", [el, ...recent().filter((x) => x !== el)].slice(0, 8));
    S.toolChip();
  };
  const ed = () => V.ed();

  /* ---------------------------------------------------------------- live symmetry preview */
  /* At most one call per 150 ms while the structure keeps changing, never two
   * in flight; when one returns and the structure changed meanwhile, the
   * newest state goes next (plan A10). */
  const pv = { timer: 0, inflight: false, want: null, result: null, sent: 0, last: -1e9, log: [] };
  function previewSymmetry(e, frac) {
    pv.want = { cell: e.structure.cell, numbers: e.structure.numbers, frac, symprec: e.symprec, key: `doc${e.doc.id}` };
    if (pv.timer || pv.inflight) return;
    pv.timer = setTimeout(() => { pv.timer = 0; firePreview(); }, Math.max(0, 150 - (performance.now() - pv.last)));
  }
  async function firePreview() {
    if (pv.inflight || !pv.want) return;            // the answer to the one in flight sends the newest
    const body = pv.want;
    pv.want = null;
    pv.inflight = true;
    pv.sent++;
    pv.last = performance.now();
    const t0 = performance.now();
    try {
      const res = await fetch("api/symmetry", { method: "POST", credentials: "same-origin",
        headers: { "X-Requested-With": "mace-gui", "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const r = await res.json();
      pv.log.push({ t0, t1: performance.now(), timing: res.headers.get("Server-Timing") });
      if (res.ok && !r.superseded && grab) { pv.result = r; S.toolChip(); if (S.statusChanged) S.statusChanged(); }
    } catch (err) { /* a preview only */ }
    finally {
      pv.inflight = false;
      if (pv.want && !pv.timer) pv.timer = setTimeout(() => { pv.timer = 0; firePreview(); }, Math.max(0, 150 - (performance.now() - pv.last)));
    }
  }
  const groupText = (r) => (r && r.available ? `${r.international} (${r.number})` : "");

  /* ---------------------------------------------------------------- Move: the grab */
  let grab = null;
  function startGrab(x, y, viaButton) {
    const e = ed();
    if (!e || !e.selection.size || e.editBlock()) return false;
    const sel = [...e.selection].sort((a, b) => a - b);
    grab = { e, sel, x0: x, y0: y, x, y, viaButton, keysUsed: false, axis: null, typed: "", snap: false, choice: null,
      leaving: false, frame: 0 };
    pv.result = null;
    S.setLayer({ name: "grabActive", onKey });
    update();
    return true;
  }
  const axisUnit = (e, axis) => (!axis ? null : "xyz".includes(axis) ? [[1, 0, 0], [0, 1, 0], [0, 0, 1]]["xyz".indexOf(axis)]
    : G.unit3(e.structure.cell["abc".indexOf(axis)]));
  /** The displacement in Å now: typed along the locked axis, else the pointer's. */
  function delta() {
    const g = grab, e = g.e, c = V.cam().calibrate();
    let d;
    const typed = g.typed && g.typed !== "-" && g.typed !== "." ? Number(g.typed) : NaN;
    if (Number.isFinite(typed)) d = G.scale3(axisUnit(e, g.axis || "x"), typed);
    else {
      d = G.add3(G.scale3(c.right, (g.x - g.x0) / c.s), G.scale3(c.up, -(g.y - g.y0) / c.s));
      const u = axisUnit(e, g.axis);
      if (u) d = G.scale3(u, G.dot3(d, u));
    }
    if (g.snap) {
      const f = G.fracOf(e.structure.cell, d).map((x) => Math.round(x / SNAP) * SNAP);
      d = G.cartOf(e.structure.cell, f);
    }
    return d;
  }
  /** Where every moving atom goes (Sym: whole orbits by the session's operations). */
  function previewPositions(d) {
    const e = grab.e, s = e.structure, df = G.fracOf(s.cell, d), out = new Map();
    const tol = Math.max(2 * e.symprec, 1e-3);
    grab.leaving = false;
    if (e.symMode === "sym" && e.ops) {
      const done = new Set();
      for (const i of grab.sel) {
        if (done.has(i)) continue;
        const orbit = e.orbitOf(i);
        orbit.forEach((j) => done.add(j));
        const f0 = s.frac[i];
        const use = grab.choice === "keep" ? G.projectOntoSite(e.ops, f0, df, s.cell, tol) : df;
        const f1 = f0.map((x, k) => x + use[k]);
        if (G.orbitApply(e.ops, f1, s.cell, tol).length !== orbit.length) grab.leaving = true;
        for (let k = 0; k < e.ops.rotations.length; k++) {
          const R = e.ops.rotations[k], t = e.ops.translations[k];
          const img = [0, 1, 2].map((r) => R[r][0] * f0[0] + R[r][1] * f0[1] + R[r][2] * f0[2] + t[r]);
          const j = e.findAt(img, s.numbers[i]);
          if (j < 0 || out.has(j)) continue;
          const p = [0, 1, 2].map((r) => R[r][0] * f1[0] + R[r][1] * f1[1] + R[r][2] * f1[2] + t[r]);
          out.set(j, p);
        }
      }
    } else for (const i of grab.sel) out.set(i, s.frac[i].map((x, k) => x + df[k]));
    return out;
  }
  function update() {
    if (!grab || grab.frame) return;
    grab.frame = requestAnimationFrame(() => {
      if (!grab) return;
      grab.frame = 0;
      const e = grab.e, s = e.structure, d = delta();
      grab.d = d;
      const moved = previewPositions(d);
      // the drawn copies of each moving atom follow it (boundary images too)
      const disp = V.scene().display, cart = new Map();
      for (const [i, f] of moved) cart.set(i, G.cartOf(s.cell, f));
      const drawn = new Map();
      if (disp) for (let k = 0; k < disp.n; k++) {
        const i = disp.base[k];
        if (!cart.has(i)) continue;
        const sh = [disp.shift[3 * k], disp.shift[3 * k + 1], disp.shift[3 * k + 2]];
        const p = G.add3(cart.get(i), G.cartOf(s.cell, sh));
        drawn.set(`${i}:${k}`, [i, p]);
      }
      V.scene().setDrag([...drawn.values()]);
      const frac = s.frac.map((f, i) => (moved.has(i) ? moved.get(i) : f));
      previewSymmetry(e, frac);
      S.toolChip();
    });
  }
  function endGrab() {
    if (!grab) return;
    cancelAnimationFrame(grab.frame);
    grab = null;
    clearTimeout(pv.timer);
    pv.timer = 0;
    pv.want = null;
    S.setLayer(null);
    V.scene().setDrag(null);
    S.toolChip();
    if (pv.result) { pv.result = null; if (S.statusChanged) S.statusChanged(); }
  }
  function commit() {
    const g = grab;
    if (!g) return;
    const d = g.d || delta();
    endGrab();
    if (G.norm3(d) < 1e-9) return;
    const args = { indices: g.sel, delta: d };
    if (g.choice === "keep") args.project = true;
    if (g.choice === "allow") args.confirm = true;
    E.edit(g.e, "move", args);
  }
  function onKey(ev) {
    const g = grab;
    if (!g) return "pass";
    const k = ev.key, code = ev.code || "";
    if (["Control", "Meta", "Shift", "Alt", "AltGraph"].includes(k)) {
      g.snap = ev.ctrlKey || ev.metaKey;
      update();
      return "pass";
    }
    const mod = ev.ctrlKey || ev.metaKey;
    if (/^Key[XYZ]$/.test(code) && !mod && !ev.altKey) {
      const n = "XYZ".indexOf(code[3]);
      g.axis = ev.shiftKey ? "abc"[n] : "xyz"[n];
      g.keysUsed = true; update(); return "consume";
    }
    if (!mod && !ev.altKey && (/^(Digit|Numpad)\d$/.test(code) || k === "." || k === "-" || code === "NumpadDecimal" || code === "NumpadSubtract")) {
      const ch = /\d/.test(k) ? k : k === "." || code === "NumpadDecimal" ? "." : "-";
      if (ch === "-") g.typed = g.typed.startsWith("-") ? g.typed.slice(1) : "-" + g.typed;
      else if (!(ch === "." && g.typed.includes("."))) g.typed += ch;
      g.keysUsed = true; update(); return "consume";
    }
    if (k === "Backspace") { g.typed = g.typed.slice(0, -1); g.keysUsed = true; update(); return "consume"; }
    if (k === "Tab") {
      const order = g.axis && "abc".includes(g.axis) ? "abc" : "xyz";
      g.axis = order[(order.indexOf(g.axis || order[2]) + 1) % 3];
      g.keysUsed = true; update(); return "consume";
    }
    if (code === "KeyG" && !mod && !ev.altKey) { g.x0 = g.x; g.y0 = g.y; g.typed = ""; g.axis = null; g.choice = null; g.keysUsed = true; update(); return "consume"; }
    if (k === "Enter" || code === "NumpadEnter") { commit(); return "consume"; }
    if (k === "Escape") { endGrab(); return "consume"; }
    return "swallow";
  }
  const move = {
    down(ev, x, y) {
      if (grab) { if (ev.button === 0) { commit(); return true; } return false; }
      const k = V.scene().pickAt(x, y), e = ed();
      if (k < 0 || !e || !e.selection.has(V.scene().display.base[k])) return false;
      startGrab(x, y, true);
      return true;
    },
    move(ev, x, y) { if (grab) { grab.x = x; grab.y = y; grab.snap = ev.ctrlKey || ev.metaKey; update(); } },
    hover(ev, x, y) { if (grab) { grab.x = x; grab.y = y; grab.snap = ev.ctrlKey || ev.metaKey; update(); } },
    up() {
      if (!grab) return;
      // a plain drag applies on release; a key used during it keeps the grab going
      if (grab.viaButton && !grab.keysUsed) commit();
      else if (grab) grab.viaButton = false;
    },
    right() { if (grab) { endGrab(); return true; } return false; },
  };

  /* ---------------------------------------------------------------- Add Atom */
  function planePoint(x, y) {
    const e = ed(), s = e.structure, c = V.cam().calibrate();
    const center = G.add3(G.scale3(G.add3(G.add3(s.cell[0], s.cell[1]), s.cell[2]), 0.5), s.origin || [0, 0, 0]);
    const [u, v] = G.project(c.P, ...center), [u1] = G.project(c.P, ...G.add3(center, c.right));
    const sc = Math.abs(u1 - u) || c.s;
    return G.add3(center, G.add3(G.scale3(c.right, (x - u) / sc), G.scale3(c.up, -(y - v) / sc)));
  }
  async function bondedPoint(k) {
    const e = ed(), s = e.structure, disp = V.scene().display, i = disp.base[k];
    const p = [disp.cart[3 * k], disp.cart[3 * k + 1], disp.cart[3 * k + 2]];
    const tol = Number(S.setting("bondTolerance")) || 1.15;
    const b = G.periodicBonds(s.cell, Float64Array.from(s.frac.flat()), Float32Array.from(s.radii), Uint8Array.from(s.metal.map(Number)), tol, true);
    const adj = G.adjacency(b, e.n)[i] || [];
    const sh = [disp.shift[3 * k], disp.shift[3 * k + 1], disp.shift[3 * k + 2]];
    let dir;
    if (adj.length) {
      const pts = adj.map(({ j, img }) => G.cartOf(s.cell, s.frac[j].map((x, m) => x + img[m] + sh[m])));
      const cen = pts.reduce((t, q) => G.add3(t, q), [0, 0, 0]).map((x) => x / pts.length);
      dir = G.unit3(G.sub3(p, cen));
      if (!Number.isFinite(dir[0]) || G.norm3(G.sub3(p, cen)) < 1e-6) dir = V.cam().calibrate().toward;
    } else dir = V.cam().calibrate().toward;
    const z = G.zOf(element);
    return G.add3(p, G.scale3(dir, G.COVALENT[s.numbers[i]] + G.COVALENT[z]));
  }
  async function addAt(point) {
    const e = ed();
    if (!G.zOf(element)) { S.toast("Pick an element first", "info"); return; }
    return E.edit(e, "add", { element, frac: G.fracOf(e.structure.cell, point) });
  }
  /** Add Atom Here (the empty-space menu): the current element where the menu opened. */
  function addHere(clientX, clientY) {
    const e = ed();
    if (!e || e.editBlock()) return null;
    return addAt(planePoint(...window.maceZoom.point(document.getElementById("viewer-main"), clientX, clientY)));
  }
  let addPreview = null;
  const add = {
    click(ev, x, y, k) {
      const e = ed();
      if (!e || e.editBlock()) return;
      if (ev.shiftKey && k >= 0) bondedPoint(k).then(addAt);
      else addAt(planePoint(x, y));
    },
    hover(ev, x, y) {
      const e = ed();
      if (!e || e.symMode !== "sym" || !e.ops) { if (addPreview) { addPreview = null; S.toolChip(); } return; }
      const f = G.fracOf(e.structure.cell, planePoint(x, y));
      const n = G.orbitApply(e.ops, f, e.structure.cell, Math.max(2 * e.symprec, 1e-3)).length;
      if (!addPreview || addPreview.n !== n) { addPreview = { n }; S.toolChip(); }
    },
  };

  /* ---------------------------------------------------------------- Replace (paint) */
  let paint = null;
  const replace = {
    down(ev, x, y) {
      const k = V.scene().pickAt(x, y), e = ed();
      if (k < 0 || !e || e.editBlock()) return false;
      paint = new Set([V.scene().display.base[k]]);
      return true;
    },
    move(ev, x, y) {
      if (!paint) return;
      const k = V.scene().pickAt(x, y);
      if (k >= 0) paint.add(V.scene().display.base[k]);
    },
    up() {
      const set = paint;
      paint = null;
      if (!set || !set.size) return;
      if (!G.zOf(element)) { S.toast("Pick an element first", "info"); return; }
      E.edit(ed(), "replace", { indices: [...set].sort((a, b) => a - b), element });
    },
  };

  /* ---------------------------------------------------------------- Bond */
  let first = null;
  const bond = {
    click(ev, x, y, k) {
      const e = ed(), disp = V.scene().display;
      if (!e || k < 0) { first = null; S.toolChip(); return; }
      if (first === null || first === k) { first = k; S.toolChip(); return; }
      const a = first, b = k;
      first = null;
      const ia = disp.base[a], ib = disp.base[b];
      let off = [0, 1, 2].map((m) => disp.shift[3 * b + m] - disp.shift[3 * a + m]);
      let key = ia < ib || (ia === ib) ? `${ia}-${ib}:${off.join(",")}` : `${ib}-${ia}:${off.map((v) => -v).join(",")}`;
      const pairs = V.scene().pairs;
      let bonded = false;
      if (pairs) for (let p = 0; p < pairs.length; p += 2) if ((pairs[p] === a && pairs[p + 1] === b) || (pairs[p] === b && pairs[p + 1] === a)) bonded = true;
      const now = e.bondOverrides.get(key);
      if (now) e.bondOverrides.delete(key); else e.bondOverrides.set(key, bonded ? "remove" : "add");
      E.redraw();
      S.toast(`${bonded ? "Removed" : "Added"} the bond ${e.structure.symbols[ia]}${ia + 1}–${e.structure.symbols[ib]}${ib + 1} (display only)`, "info");
      S.toolChip();
    },
  };

  /* ---------------------------------------------------------------- the mode chip */
  function elementField(label) {
    const wrap = document.createElement("span");
    wrap.className = "vp-chip-el";
    wrap.innerHTML = `<span>${label} ·</span><input class="vp-el" value="${element}" aria-label="${label} element" spellcheck="false" autocomplete="off">` +
      `<button type="button" class="vp-pick" aria-label="Pick from the periodic table">▾</button><span class="err" role="alert"></span>`;
    const input = wrap.querySelector("input"), err = wrap.querySelector(".err");
    const accept = () => {
      const z = G.zOf(input.value);
      if (!z) { input.setAttribute("aria-invalid", "true"); err.textContent = "Not an element symbol"; return false; }
      input.removeAttribute("aria-invalid"); err.textContent = "";
      setElement(G.SYMBOLS[z]);
      return true;
    };
    input.addEventListener("focus", () => S.setLayer({ name: "elementEntry", onKey: () => "pass" }));
    input.addEventListener("blur", () => { if (!grab) S.setLayer(null); });
    input.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") { ev.preventDefault(); ev.stopPropagation(); if (accept()) V.focusViewport(); }
      else if (ev.key === "Escape") { ev.preventDefault(); ev.stopPropagation(); input.value = element; err.textContent = ""; input.removeAttribute("aria-invalid"); V.focusViewport(); }
    });
    wrap.querySelector(".vp-pick").addEventListener("click", (ev) => S.pickElement(ev.currentTarget, (el) => setElement(el), { recent: recent() }));
    return wrap;
  }
  function chip(tool, e) {
    const pre = e && e.waiting ? "Applying… · " : "";
    if (grab) {
      const g = grab, d = g.d || [0, 0, 0], ax = g.axis || (g.typed ? "x" : "");
      const amount = ax ? `${ax} ${G.fmtNum(G.dot3(d, axisUnit(g.e, ax)), 3)} Å` : `${G.fmtNum(G.norm3(d), 3)} Å`;
      const box = document.createElement("span");
      box.className = "vp-chip-move";
      const gtxt = groupText(pv.result);
      box.innerHTML = `<span>${pre}Move · ${amount}${g.typed ? ` (typed ${g.typed})` : ""}${gtxt ? ` · ${gtxt}` : ""}</span>` +
        `<span class="k">Enter or click to apply · Esc to cancel</span>`;
      if (g.leaving && g.e.symMode === "sym" && g.choice === null) {
        const site = E.wyckoff(g.e.structure, g.sel[0]), ref = g.e.referenceGroup ? g.e.referenceGroup.international : "the group";
        box.classList.add("warn");
        const lower = pv.result && pv.result.available ? ` to ${pv.result.international}` : "";
        box.insertAdjacentHTML("beforeend", `<span class="warn-t">Moving off ${site} lowers the group${lower}.</span>` +
          `<button type="button" class="primary" data-g="keep">Keep ${ref}</button><button type="button" data-g="allow">Allow</button>`);
        box.querySelectorAll("[data-g]").forEach((b) => b.addEventListener("click", (ev) => { ev.stopPropagation(); if (grab) { grab.choice = b.dataset.g; update(); } }));
      }
      return box;
    }
    if (tool === "add") {
      const w = elementField("Add Atom");
      w.insertAdjacentHTML("beforeend", `<span class="k">${addPreview && addPreview.n > 1 ? `+${addPreview.n} atoms · ` : ""}click a position, Shift+click an atom to bond to it</span>`);
      if (pre) w.insertAdjacentHTML("afterbegin", `<span>Applying…</span>`);
      return w;
    }
    if (tool === "replace") {
      const w = elementField("Replace");
      w.insertAdjacentHTML("beforeend", `<span class="k">click or drag over atoms</span>`);
      return w;
    }
    if (tool === "move") return `${pre}Move · drag the selection, or press G · X Y Z lock an axis`;
    if (tool === "bond") return `${pre}Bond · ${first === null ? "click two atoms to add or remove a bond (display only)" : "click the second atom"}`;
    return pre ? "Applying…" : null;
  }

  return {
    move, add, replace, bond, chip, startGrab, endGrab, commit, addHere,
    get grabbing() { return !!grab; }, get grabState() { return grab ? { axis: grab.axis, typed: grab.typed, d: grab.d, choice: grab.choice, leaving: grab.leaving } : null; },
    get element() { return element; }, setElement, focusElement: () => { const i = document.querySelector(".vp-chip .vp-el"); if (i) { i.focus(); i.select(); return true; } return false; },
    preview: pv,
  };
}
