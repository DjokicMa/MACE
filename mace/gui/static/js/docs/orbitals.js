/* The orbital picker (CRYSTAL23 ORBITALS, spec 4.4): beside the Volume
 * document's 3D view when a .molden file is open. The server's index
 * (GET orbitals: mace.orbitals.api.describe) gives the k points (complex k of
 * a 2D/3D system is listed but cannot be picked: CRYSTAL23's coefficients fail
 * the orthonormality check there), the spins and the levels of one k point;
 * the picker turns a choice into a request for GET orbitals/grid, which the
 * volume view draws with its usual lobes and level slider.
 *
 * Keyboard: the level list is a listbox (Up/Down, Page Up/Down, Home/End move
 * the focus; Enter or Space shows the level); the k points, spins and views
 * are buttons and radio groups (arrow keys move within a group). */

const QUALITY_OF_SETTING = { low: "preview", medium: "normal", high: "fine" };
const PART_LABEL = { re: "Re ψ", im: "Im ψ", abs2: "|ψ|²" };
const SPIN_TAB = { "ALPHA+BETA": "Closed shell", ALPHA: "α (alpha)", BETA: "β (beta)" };

export function createOrbitalPicker(el, H, onRequest) {
  const { esc } = H;
  let d = null;           // the volume document (d.orbitals: { index, request, startK })
  let focusN = null;      // the level that has the focus in the list (may differ from the shown one)

  const fmtE = (x) => { const r = Math.round(x * 100) / 100; return `${r < 0 ? "−" : r > 0 ? "+" : ""}${Math.abs(r).toFixed(2)}`; };
  const kText = (e) => (e.label === "molecule" ? "Molecule" : e.label);
  const spinOf = (ix, name) => (ix.k && ix.k.spins.find((s) => s.spin === name)) || (ix.k && ix.k.spins[0]);
  const curSpin = () => spinOf(d.orbitals.index, d.orbitals.request.spin);
  const levelOf = (n) => curSpin().levels.find((l) => l.n === n);
  const groupOf = (lv) => curSpin().levels.filter((l) => l.group === lv.group).map((l) => l.n);
  const kEntry = () => d.orbitals.index.kpoints.find((e) => e.index === d.orbitals.request.k);
  const realK = () => { const e = kEntry(); return !e || e.real; };

  /** The first request for an index: the HOMO of the first spin, Re psi at a real k, |psi|^2 at a complex one. */
  function firstRequest(ix, prev) {
    const s = (prev && spinOf(ix, prev.spin)) || ix.k.spins[0];
    const e = ix.kpoints.find((x) => x.index === ix.selected);
    let part = prev ? prev.part : "re";
    if (part === "im" && e && e.real) part = "re";
    if (!prev && e && !e.real) part = "abs2";
    const quality = prev ? prev.quality : (QUALITY_OF_SETTING[H.setting("volume.quality") || "medium"] || "normal");
    return { k: ix.selected, spin: s.spin, mo: s.default_mo, part, sum: false,
      cells: prev ? prev.cells.slice() : (H.initialCells && H.initialCells(ix)) || (ix.default_cells || [1, 1, 1]).slice(), quality };
  }

  async function fetchIndex(path, k) {
    const ix = await H.apiTask("orbitals?" + H.q(k ? { path, k } : { path }));
    if (!ix.k) throw new Error(ix.kpoints && ix.kpoints.length ? "No k point of this run can be drawn: " + (ix.kpoints[0].reason || "")
      : "No orbitals in this file");
    return ix;
  }

  /* ---------------------------------------------------------------- render */
  function render() {
    if (!d || !d.orbitals.index) { el.innerHTML = ""; return; }
    const ix = d.orbitals.index, r = d.orbitals.request, sp = curSpin(), sel = levelOf(r.mo);
    const k = ix.k, check = k.status;
    const checkText = { ok: "Self-check passed", warn: "Self-check: small orthonormality errors", fail: "Self-check failed",
      unchecked: "Not checked" }[check] || check;
    const worst = Object.values(k.check || {}).reduce((a, b) => Math.max(a, b), 0);
    const ks = ix.kpoints.length > 1 || !ix.molecule ? `<div class="orb-sec"><div class="po-h" id="orb-k-h">k points</div>
      <div class="orb-klist" role="group" aria-labelledby="orb-k-h">${ix.kpoints.map((e) => {
        const on = e.index === r.k, off = !e.usable;
        const rc = e.real ? `<span class="orb-rc" data-rc="R" title="Real k point: ψ is real up to a sign">R</span>`
          : `<span class="orb-rc" data-rc="C" title="Complex k point: ψ has real and imaginary parts">C</span>`;
        return `<button type="button" class="orb-k" data-k="${e.index}" aria-pressed="${on}"${off ? ` aria-disabled="true" aria-describedby="orb-kwhy-${e.index}" title="${esc(e.reason)}"` : ""}
          aria-label="k point ${e.index}: ${esc(kText(e))}, ${e.real ? "real" : "complex"}${off ? ", not available" : ""}">${rc}<span class="orb-kl mono">${esc(kText(e))}</span><span class="orb-kn muted small">#${e.index}</span>${off ? `<span class="sr-only" id="orb-kwhy-${e.index}">${esc(e.reason)}</span>` : ""}</button>`;
      }).join("")}</div>${ix.asked ? `<p class="orb-note small" role="note">The file opened is of k point ${ix.asked.k}, which cannot be drawn: ${esc(ix.asked.reason)}. Showing k point ${r.k}.</p>` : ""}</div>` : "";
    const spins = k.spins.length > 1 ? `<div class="orb-spins segment" role="tablist" aria-label="Spin">${k.spins.map((s) =>
      `<button type="button" role="tab" data-spin="${esc(s.spin)}" aria-selected="${s.spin === r.spin}" tabindex="${s.spin === r.spin ? 0 : -1}">${esc(SPIN_TAB[s.spin] || s.spin)}</button>`).join("")}</div>` : "";
    const refName = sp.ref_kind === "E_F" ? "E<sub>F</sub>" : sp.ref_kind;
    // the diagram: highest level on top; the gap between levels follows their energy difference (clamped)
    const levels = sp.levels.slice().reverse();
    let prevE = null, efDone = sp.ref_kind !== "E_F";
    const rows = levels.map((l, i) => {
      const gap = prevE === null ? 0 : l.group === levels[i - 1].group ? 0 : Math.max(3, Math.min(26, (prevE - l.e_ev) * 3));
      prevE = l.e_ev;
      let ef = "";
      if (!efDone && l.rel_ev <= 0) { efDone = true; ef = `<div class="orb-ef" aria-hidden="true"><span>E<sub>F</sub></span></div>`; }
      const tags = [];
      if (sp.ref_kind === "E_F") {
        const below = sp.below_ef;
        if (below !== null && below !== undefined && l.n <= below && l.n > below - 2) tags.push(`E<sub>F</sub> − ${below - l.n + 1}`);
        if (below !== null && below !== undefined && l.n > below && l.n <= below + 2) tags.push(`E<sub>F</sub> + ${l.n - below}`);
      } else if (sp.homo !== null) {
        if (l.n === sp.homo) tags.push("HOMO");
        if (l.n === sp.lumo) tags.push("LUMO");
      }
      const deg = l.degenerate > 1, g = deg ? sp.levels.filter((x) => x.group === l.group).map((x) => x.n) : [];
      const pos = deg ? (l.n === Math.max(...g) ? " deg-top" : l.n === Math.min(...g) ? " deg-bottom" : " deg-mid") : "";
      const shown = l.n === r.mo || (r.sum && g.includes(r.mo));
      const lbl = `MO ${l.n}, ${fmtE(l.rel_ev)} electron volts from the ${sp.ref_kind === "E_F" ? "Fermi level" : "HOMO"}${tags.length ? ", " + tags.join(", ").replace(/<\/?sub>/g, "") : ""}, ${l.occ_text}${deg ? `, degenerate set of ${l.degenerate} (MO ${Math.min(...g)} to ${Math.max(...g)})` : ""}`;
      return `${ef}<div role="option" class="orb-lv${l.occupied ? " occ" : ""}${deg ? " deg" : ""}${pos}${shown ? " shown" : ""}" id="orb-lv-${l.n}" data-n="${l.n}"
        aria-selected="${l.n === r.mo}" tabindex="${l.n === (focusN ?? r.mo) ? 0 : -1}" style="margin-top:${gap.toFixed(1)}px" aria-label="${esc(lbl)}">
        <span class="orb-e mono">${fmtE(l.rel_ev)}</span><span class="orb-bar" aria-hidden="true"></span>
        <span class="orb-n">MO ${l.n}</span>${tags.map((t) => `<span class="orb-tag">${t}</span>`).join("")}
        ${deg && pos === " deg-top" ? `<span class="orb-deg" title="Degenerate set: MO ${Math.min(...g)}–${Math.max(...g)}">×${l.degenerate}</span>` : ""}
        <span class="orb-occ" title="${esc(l.occ_text)}">${esc(l.occ_text)}</span></div>`;
    }).join("");
    const degSet = sel && sel.degenerate > 1 ? groupOf(sel) : null;
    const parts = ["re", "im", "abs2"].map((p) => {
      const off = (p === "im" && realK()) || (r.sum && p !== "abs2");
      const why = p === "im" && realK() ? "Im ψ is zero at Γ and real k points" : r.sum && p !== "abs2" ? "A summed set is shown as |ψ|² only" : "";
      const on = r.sum ? p === "abs2" : r.part === p;
      return `<button type="button" role="radio" data-part="${p}" aria-checked="${on}" tabindex="${on ? 0 : -1}"${off ? ` aria-disabled="true" title="${why}"` : ""}>${PART_LABEL[p]}</button>`;
    }).join("");
    const per = ix.periodic_axes || [];
    const cells = ix.molecule ? "" : `<div class="po-row orb-cells" role="group" aria-labelledby="orb-cells-h"><span id="orb-cells-h">Cells</span>${[0, 1, 2].map((i) =>
      `<input type="number" class="orb-cell" data-axis="${i}" min="1" max="${ix.max_cells}" step="1" value="${per.includes(i) ? r.cells[i] : 1}"${per.includes(i) ? "" : " disabled"} aria-label="Cells along ${"abc"[i]}${per.includes(i) ? "" : " (not periodic)"}">`).join(`<span aria-hidden="true">×</span>`)}</div>`;
    el.innerHTML = `<div class="orb">
      <div class="orb-head"><div class="orb-sys"><b>${esc(ix.system)}</b> · ${esc(ix.root)} · ${k.nbf} basis functions</div>
        <span class="orb-check" data-check="${esc(check)}" title="Largest |CᴴSC − 1| over the occupied levels and the first two empty ones: ${worst.toExponential(1)}">${esc(checkText)}</span></div>
      ${ks}${spins}
      <div class="orb-sec orb-show">
        <div class="po-row"><span id="orb-part-h">View</span><span class="segment orb-parts" role="radiogroup" aria-labelledby="orb-part-h">${parts}</span></div>
        <label class="chk po-row"><input type="checkbox" id="orb-sum"${r.sum ? " checked" : ""}${degSet ? "" : " disabled"}> Sum the set (|ψ|²${degSet ? `, MO ${degSet[0]}–${degSet[degSet.length - 1]}` : ""})</label>
        ${degSet && !r.sum ? `<p class="orb-note small">MO ${r.mo} is one of a degenerate set: CRYSTAL mixes the set arbitrarily, so its shape is gauge-dependent. Sum the set for a gauge-free |ψ|².</p>` : ""}
        ${cells}
        <label class="po-row"><span>Quality</span><select id="orb-quality">${(ix.qualities || ["preview", "normal", "fine"]).map((x) =>
          `<option value="${x}"${x === r.quality ? " selected" : ""}>${x[0].toUpperCase() + x.slice(1)}</option>`).join("")}</select></label>
      </div>
      <p class="orb-status small" role="status" aria-live="polite"></p>
      <div class="orb-sec orb-lvsec"><div class="po-h" id="orb-lv-h">Levels <span class="muted small">(eV from ${refName})</span></div>
        <div class="orb-levels" role="listbox" aria-labelledby="orb-lv-h">${rows}</div></div>
      ${ix.notes && ix.notes.length ? `<details class="orb-about small"><summary>About these orbitals</summary><ul class="orb-notes">${ix.notes.map((n) => `<li>${esc(n.replace(/\|psi\|\^2/g, "|ψ|²").replace(/\bpsi\b/g, "ψ"))}</li>`).join("")}</ul></details>` : ""}
    </div>`;
    const cur = el.querySelector(`#orb-lv-${focusN ?? r.mo}`);
    if (cur) {
      // scroll the shown level into the middle of the list once (not on every render)
      const list = el.querySelector(".orb-levels");
      if (!d.orbitals.scrolled) {
        list.scrollTop = Math.max(0, cur.offsetTop - list.offsetTop - list.clientHeight / 2); d.orbitals.scrolled = true;
        // and the picker's own pane, so the levels are in view on a short screen (the K points, View, Sum,
        // Cells and Quality blocks above them fill a 720 px window)
        requestAnimationFrame(() => {
          const c = el.querySelector(`#orb-lv-${focusN ?? r.mo}`);
          if (!c || !el.clientHeight) return;
          const cr = c.getBoundingClientRect(), pr = el.getBoundingClientRect();
          if (cr.bottom > pr.bottom || cr.top < pr.top) el.scrollTop += cr.top - pr.top - el.clientHeight / 3;
        });
      }
      else if (d.orbitals.scrollTop !== undefined) list.scrollTop = d.orbitals.scrollTop;
    }
  }
  const status = (text) => { const s = el.querySelector(".orb-status"); if (s) s.textContent = text; };

  /* ---------------------------------------------------------------- changes */
  function change(patch, { refit = false } = {}) {
    Object.assign(d.orbitals.request, patch);
    const list = el.querySelector(".orb-levels");
    if (list) d.orbitals.scrollTop = list.scrollTop;
    render();
    onRequest(d, { refit });
  }
  async function pickK(k) {
    const r = d.orbitals.request;
    if (k === r.k) return;
    status(`Reading k point ${k}…`);
    const ix = await fetchIndex(d.path, k);
    if (!d) return;
    d.orbitals.index = ix;
    d.orbitals.request = firstRequest(ix, r);
    d.orbitals.scrolled = false;
    focusN = null;
    render();
    onRequest(d, { refit: false });
  }
  function showLevel(n) {
    const r = d.orbitals.request, lv = levelOf(n);
    if (!lv) return;
    focusN = n;
    // a summed set stays summed when another member of it is picked; a level of no set is shown alone
    const sum = r.sum && lv.degenerate > 1;
    change({ mo: n, sum });
    const row = el.querySelector(`#orb-lv-${n}`);
    if (row) row.focus({ preventScroll: false });
  }
  function moveFocus(row, delta, abs = null) {
    const rows = Array.from(el.querySelectorAll(".orb-lv"));
    let i = rows.indexOf(row);
    i = abs === "first" ? 0 : abs === "last" ? rows.length - 1 : Math.max(0, Math.min(rows.length - 1, i + delta));
    const to = rows[i];
    rows.forEach((x) => x.setAttribute("tabindex", x === to ? "0" : "-1"));
    focusN = Number(to.dataset.n);
    to.focus();
  }

  el.addEventListener("click", (e) => {
    if (!d || !d.orbitals.index) return;
    const kb = e.target.closest(".orb-k");
    if (kb) {
      if (kb.getAttribute("aria-disabled") === "true") { status(kb.title); return; }
      pickK(Number(kb.dataset.k)).catch((er) => { status(""); H.toast(er.message, "error"); });
      return;
    }
    const sb = e.target.closest("[data-spin]");
    if (sb) {
      const s = spinOf(d.orbitals.index, sb.dataset.spin);
      if (s.spin === d.orbitals.request.spin) return;
      focusN = null; d.orbitals.scrolled = false;
      change({ spin: s.spin, mo: s.default_mo, sum: false });
      return;
    }
    const lv = e.target.closest(".orb-lv");
    if (lv) { showLevel(Number(lv.dataset.n)); return; }
    const pb = e.target.closest("[data-part]");
    if (pb) {
      if (pb.getAttribute("aria-disabled") === "true") { status(pb.title); return; }
      if (pb.dataset.part !== d.orbitals.request.part) change({ part: pb.dataset.part });
      return;
    }
  });
  el.addEventListener("keydown", (e) => {
    if (!d || !d.orbitals.index) return;
    const row = e.target.closest(".orb-lv");
    if (row) {
      const k = e.key;
      if (k === "ArrowUp" || k === "ArrowDown") { e.preventDefault(); moveFocus(row, k === "ArrowUp" ? -1 : 1); }
      else if (k === "PageUp" || k === "PageDown") { e.preventDefault(); moveFocus(row, k === "PageUp" ? -5 : 5); }
      else if (k === "Home" || k === "End") { e.preventDefault(); moveFocus(row, 0, k === "Home" ? "first" : "last"); }
      else if (k === "Enter" || k === " ") { e.preventDefault(); showLevel(Number(row.dataset.n)); }
      return;
    }
    // radio groups (views, spin tabs): the arrow keys move within the group
    const grp = e.target.closest(".orb-parts, .orb-spins");
    if (grp && ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) {
      e.preventDefault();
      const btns = Array.from(grp.querySelectorAll("button")).filter((b) => b.getAttribute("aria-disabled") !== "true");
      const i = btns.indexOf(e.target), step = e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 1;
      const to = btns[(i + step + btns.length) % btns.length];
      if (to) { to.click(); const again = el.querySelector(grp.classList.contains("orb-parts") ? `[data-part="${to.dataset.part}"]` : `[data-spin="${to.dataset.spin}"]`); if (again) again.focus(); }
    }
  });
  el.addEventListener("change", (e) => {
    if (!d || !d.orbitals.index) return;
    const t = e.target;
    if (t.id === "orb-sum") change({ sum: t.checked });
    else if (t.id === "orb-quality") change({ quality: t.value });
    else if (t.classList.contains("orb-cell")) {
      const ix = d.orbitals.index, i = Number(t.dataset.axis);
      const v = Math.max(1, Math.min(ix.max_cells || 4, Math.round(Number(t.value)) || 1));
      const cells = d.orbitals.request.cells.slice();
      cells[i] = v;
      // the toolbar's display range is the one control: it asks for the new grid (volume.js redraw)
      if (H.setCells) H.setCells(cells); else change({ cells }, { refit: true });
    }
  });

  return {
    /** The picker for a document: its index (fetched once) and first request. */
    async open(doc) {
      d = doc;
      focusN = null;
      if (!doc.orbitals.index) {
        el.innerHTML = `<p class="muted small orb-wait"><span class="spinner"></span> Reading the orbitals…</p>`;
        const ix = await fetchIndex(doc.path, doc.orbitals.startK);
        if (d !== doc) return;
        doc.orbitals.index = ix;
        doc.orbitals.request = firstRequest(ix, null);
      }
      render();
    },
    close() { d = null; },
    /** A new number of cells (the display range changed): the grid again, the camera refit to the box. */
    setCells(cells) { if (d && d.orbitals.request) change({ cells: cells.slice() }, { refit: true }); },
    status,
    render: () => render(),
    /** The request as the grid route's query. */
    query(doc) {
      const r = doc.orbitals.request;
      return { path: doc.path, k: r.k, spin: r.spin, mo: r.mo, part: r.sum ? "abs2" : r.part, sum: r.sum ? 1 : 0,
        cells: r.cells.join("x"), quality: r.quality };
    },
    state: () => d && d.orbitals.index && { request: { ...d.orbitals.request }, k: d.orbitals.index.selected,
      kpoints: d.orbitals.index.kpoints.map((e) => ({ index: e.index, real: e.real, usable: e.usable, reason: e.reason })),
      spins: d.orbitals.index.k.spins.map((s) => ({ spin: s.spin, homo: s.homo, lumo: s.lumo, metallic: s.metallic, ref: s.ref_kind })) },
  };
}
