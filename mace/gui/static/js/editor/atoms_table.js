/* The Atoms panel (spec §1.6, §8.3.4): a virtualized table, # / El / Label /
 * x y z (fractional or Cartesian, the document's toggle) / Wyckoff / Occ.
 * Fixed rows (24 or 28 px with the density) and 10 rows of overscan, so a
 * structure of any size keeps a few dozen rows in the page. Rows follow the
 * selection both ways (click, Shift for a range, Mod to toggle); headers
 * sort; Enter or F2 edits a cell in place (Tab goes on to the next one);
 * a pasted column of numbers sets that coordinate for the rows from the
 * focused one down, as one edit (P1 only). */
import * as G from "../geom/core.js";

const OVERSCAN = 10;
const COLS = [["#", "idx", "num"], ["El", "el", ""], ["Label", "label", ""], ["x", "x", "num"], ["y", "y", "num"], ["z", "z", "num"],
  ["Wyckoff", "wy", ""], ["Occ.", "occ", "num"]];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

export function createAtomsTable(E, S) {
  let host = null, ed = null, scroller = null, table = null, order = [], sort = { col: "idx", dir: 1 }, anchor = null;
  let cursor = { row: 0, col: 3 }, editing = null, frame = 0, maxRows = 0, repaint = false;
  const rowH = () => (document.documentElement.dataset.density === "compact" ? 24 : 28);
  const dec = () => { const d = Number(S.setting("decimals.coord")); return Number.isFinite(d) ? d : 5; };

  function value(i, key) {
    const s = ed.structure, cart = ed.coordMode === "cart";
    if (key === "idx") return i + 1;
    if (key === "el") return s.symbols[i];
    if (key === "label") return `${s.symbols[i]}${i + 1}`;
    if (key === "x" || key === "y" || key === "z") return (cart ? s.cart[i] : s.frac[i])["xyz".indexOf(key)];
    if (key === "wy") return E.wyckoff(s, i);
    return 1;
  }
  function sortOrder() {
    const n = ed.n, idx = Array.from({ length: n }, (_, i) => i);
    if (sort.col !== "idx" || sort.dir !== 1) {
      idx.sort((a, b) => {
        const va = value(a, sort.col), vb = value(b, sort.col);
        return (va < vb ? -1 : va > vb ? 1 : a - b) * sort.dir;
      });
    }
    order = idx;
  }

  function mount(h, doc) {
    host = h; ed = doc;
    host.innerHTML = `<div class="at-scroll" tabindex="0" role="region" aria-label="Atoms table"><table class="data atoms-table" role="grid">
      <thead><tr>${COLS.map(([t, k, c]) => `<th class="${c}" data-k="${k}" aria-sort="none" scope="col"><button type="button" class="at-sort">${esc(t)}</button></th>`).join("")}</tr></thead>
      <tbody></tbody></table></div>`;
    scroller = host.querySelector(".at-scroll");
    table = host.querySelector("table");
    scroller.addEventListener("scroll", () => { if (!frame) frame = requestAnimationFrame(paint); }, { passive: true });
    table.querySelector("thead").addEventListener("click", (e) => {
      const th = e.target.closest("th");
      if (!th) return;
      sort = { col: th.dataset.k, dir: sort.col === th.dataset.k ? -sort.dir : 1 };
      sortOrder(); paint();
    });
    table.querySelector("tbody").addEventListener("click", onClick);
    table.querySelector("tbody").addEventListener("dblclick", (e) => { const td = e.target.closest("td[data-c]"); if (td) startEdit(td); });
    scroller.addEventListener("keydown", onKey);
    scroller.addEventListener("paste", onPaste);
    update(host, doc);
  }
  function update(h, doc) {
    if (h !== host) { host = h; table = null; }
    if (!host || !host.isConnected) return;
    const other = doc !== ed;
    ed = doc;
    if (!ed) { host.innerHTML = `<p class="empty-note">Open a structure to see its atoms.</p>`; table = null; return; }
    if (!table || !host.contains(table)) { mount(host, doc); return; }
    if (other) { scroller.scrollTop = 0; cursor = { row: 0, col: 3 }; anchor = null; sort = { col: "idx", dir: 1 }; }
    sortOrder();
    paint();
  }
  /** Only the rows in view (plus the overscan) are in the page. */
  function paint() {
    frame = 0;
    if (!table || !ed) return;
    if (editing) { repaint = true; return; }          // never under the user's typing
    const h = rowH(), n = order.length, top = scroller.scrollTop, view = scroller.clientHeight || 240;
    const first = Math.max(0, Math.floor(top / h) - OVERSCAN), last = Math.min(n, Math.ceil((top + view) / h) + OVERSCAN);
    const d = dec(), sel = ed.selection, s = ed.structure;
    let html = first ? `<tr class="at-pad" aria-hidden="true" style="height:${first * h}px"></tr>` : "";
    for (let r = first; r < last; r++) {
      const i = order[r];
      const c = ed.coordMode === "cart" ? s.cart[i] : s.frac[i];
      html += `<tr data-i="${i}" data-r="${r}" class="${sel.has(i) ? "sel" : ""}" aria-selected="${sel.has(i)}" style="height:${h}px">` +
        `<td class="num" data-c="0">${i + 1}</td><td data-c="1">${esc(s.symbols[i])}</td><td data-c="2">${esc(s.symbols[i])}${i + 1}</td>` +
        `${c.map((x, k) => `<td class="num" data-c="${3 + k}">${G.fmtNum(x, d)}</td>`).join("")}<td data-c="6">${esc(E.wyckoff(s, i))}</td><td class="num" data-c="7">1.0</td></tr>`;
    }
    if (last < n) html += `<tr class="at-pad" aria-hidden="true" style="height:${(n - last) * h}px"></tr>`;
    table.querySelector("tbody").innerHTML = html;
    maxRows = Math.max(maxRows, table.querySelectorAll("tbody tr").length);
    table.querySelectorAll("thead th").forEach((th) => th.setAttribute("aria-sort", th.dataset.k !== sort.col ? "none" : sort.dir > 0 ? "ascending" : "descending"));
    table.querySelector("thead th:nth-child(4) .at-sort").textContent = ed.coordMode === "cart" ? "x (Å)" : "x";
    markCursor();
  }
  function markCursor() {
    table.querySelectorAll("td.cur").forEach((x) => x.classList.remove("cur"));
    const td = table.querySelector(`tr[data-r="${cursor.row}"] td[data-c="${cursor.col}"]`);
    if (td) td.classList.add("cur");
  }
  function selectionChanged() {
    if (!table || !ed) return;
    table.querySelectorAll("tbody tr[data-i]").forEach((tr) => {
      const on = ed.selection.has(Number(tr.dataset.i));
      tr.classList.toggle("sel", on);
      tr.setAttribute("aria-selected", String(on));
    });
  }
  function onClick(e) {
    const tr = e.target.closest("tr[data-i]");
    if (!tr || editing) return;
    const i = Number(tr.dataset.i), r = Number(tr.dataset.r);
    const td = e.target.closest("td[data-c]");
    cursor = { row: r, col: td ? Number(td.dataset.c) : cursor.col };
    const mod = S.isMac ? e.metaKey : e.ctrlKey;
    if (e.shiftKey && anchor !== null) {
      const a = order.indexOf(anchor), lo = Math.min(a, r), hi = Math.max(a, r);
      E.select(ed, order.slice(lo, hi + 1), mod ? "add" : "replace");
    } else if (mod) { E.select(ed, [i], "toggle"); anchor = i; }
    else { E.select(ed, [i], "replace"); anchor = i; }
    markCursor();
    scroller.focus({ preventScroll: true });
  }
  function scrollTo(r) {
    const h = rowH();
    if (r * h < scroller.scrollTop) scroller.scrollTop = r * h;
    else if ((r + 1) * h > scroller.scrollTop + scroller.clientHeight) scroller.scrollTop = (r + 1) * h - scroller.clientHeight;
  }
  function onKey(e) {
    if (editing) return;
    const n = order.length;
    if (["ArrowDown", "ArrowUp", "ArrowLeft", "ArrowRight"].includes(e.key) && !e.altKey) {
      e.preventDefault(); e.stopPropagation();
      if (e.key === "ArrowDown") cursor.row = Math.min(n - 1, cursor.row + 1);
      if (e.key === "ArrowUp") cursor.row = Math.max(0, cursor.row - 1);
      if (e.key === "ArrowLeft") cursor.col = Math.max(1, cursor.col - 1);
      if (e.key === "ArrowRight") cursor.col = Math.min(5, cursor.col + 1);
      scrollTo(cursor.row); paint();
      if (!e.shiftKey) { E.select(ed, [order[cursor.row]], "replace"); anchor = order[cursor.row]; }
      return;
    }
    if (e.key === "Enter" || e.key === "F2") {
      e.preventDefault(); e.stopPropagation();
      scrollTo(cursor.row); paint();
      const td = table.querySelector(`tr[data-r="${cursor.row}"] td[data-c="${cursor.col}"]`);
      if (td) startEdit(td);
    }
  }
  /** Edit a cell in place: El (an element) or x / y / z (an expression). */
  function startEdit(td) {
    const c = Number(td.dataset.c), tr = td.closest("tr"), i = Number(tr.dataset.i);
    if (![1, 3, 4, 5].includes(c)) return;
    if (ed.editBlock()) { S.toast(ed.editBlock(), "info"); return; }
    cursor = { row: Number(tr.dataset.r), col: c };
    const old = td.textContent;
    td.innerHTML = `<input class="at-edit" value="${esc(old)}" aria-label="${COLS[c][0]} of atom ${i + 1}" spellcheck="false" autocomplete="off">`;
    const input = td.querySelector("input");
    editing = { td, i, c };
    input.focus(); input.select();
    const finish = (commit, next) => {
      if (!editing) return;
      const v = input.value.trim();
      editing = null;
      repaint = false;
      if (commit && v !== old) {
        if (c === 1) {
          if (!G.zOf(v)) { S.toast("Not an element symbol", "error"); paint(); return; }
          E.edit(ed, "replace", { indices: [i], element: v });
        } else {
          const r = G.evalExpr(v);
          if (!r.ok) { S.toast(r.error, "error"); paint(); return; }
          const cart = ed.coordMode === "cart", p = (cart ? ed.structure.cart[i] : ed.structure.frac[i]).slice();
          p[c - 3] = r.value;
          E.edit(ed, "set_positions", cart ? { indices: [i], cart: [p] } : { indices: [i], frac: [p] });
        }
      }
      paint();
      if (next) {
        cursor.col = c >= 5 ? 1 : c === 1 ? 3 : c + 1;
        if (c >= 5) cursor.row = Math.min(order.length - 1, cursor.row + 1);
        scrollTo(cursor.row); paint();
        const td2 = table.querySelector(`tr[data-r="${cursor.row}"] td[data-c="${cursor.col}"]`);
        if (td2) startEdit(td2);
      } else scroller.focus({ preventScroll: true });
    };
    input.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.key === "Enter") { e.preventDefault(); finish(true, false); }
      else if (e.key === "Escape") { e.preventDefault(); finish(false, false); }
      else if (e.key === "Tab") { e.preventDefault(); finish(true, true); }
    });
    input.addEventListener("blur", () => finish(true, false));
  }
  /** A pasted column of numbers: that coordinate for the rows from the cursor down, one edit (P1). */
  function onPaste(e) {
    if (editing || !ed || ![3, 4, 5].includes(cursor.col)) return;
    const text = (e.clipboardData || window.clipboardData).getData("text");
    const vals = text.split(/\r?\n/).map((x) => x.trim()).filter(Boolean);
    if (!vals.length) return;
    e.preventDefault();
    if (ed.symMode === "sym") { S.toast("Switch to P1 to paste coordinates", "info"); return; }
    if (ed.editBlock()) { S.toast(ed.editBlock(), "info"); return; }
    const nums = vals.map((x) => G.evalExpr(x.split(/[\s,;]+/)[0]));
    const bad = nums.findIndex((r) => !r.ok);
    if (bad >= 0) { S.toast(`Line ${bad + 1}: ${nums[bad].error}`, "error"); return; }
    const cart = ed.coordMode === "cart", rows = order.slice(cursor.row, cursor.row + nums.length);
    const pts = rows.map((i, k) => { const p = (cart ? ed.structure.cart[i] : ed.structure.frac[i]).slice(); p[cursor.col - 3] = nums[k].value; return p; });
    E.edit(ed, "set_positions", cart ? { indices: rows, cart: pts } : { indices: rows, frac: pts });
  }
  return { mount, update, selectionChanged, maxRows: () => maxRows, resetMax: () => { maxRows = 0; }, get scroller() { return scroller; } };
}
