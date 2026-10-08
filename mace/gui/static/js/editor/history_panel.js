/* The History panel (spec §2.6): the document's undo stack, oldest first,
 * starting at "Opened MgO.cif"; each row has its label, time and the atom
 * count after it. A click jumps there in one request; the rows after the
 * current one turn gray and stay redoable until the next edit. Above 300
 * rows only the rows in view are in the page. */
const ROW = 28, OVERSCAN = 10, VIRTUAL_FROM = 300;
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const time = (t) => (t ? new Date(t * 1000).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }) : "");

export function createHistoryPanel(E, S) {
  let host = null, ed = null, scroller = null, frame = 0;
  function update(h, doc) {
    host = h; ed = doc;
    if (!host) return;
    if (!ed) { host.innerHTML = `<p class="empty-note">Edits appear here once a structure is open for editing (Edit This Step for a run); a click goes back to any of them.</p>`; scroller = null; return; }
    if (!scroller || !host.contains(scroller)) {
      host.innerHTML = `<div class="hist-scroll" role="listbox" aria-label="History" tabindex="0"><div class="hist-list"></div></div>`;
      scroller = host.querySelector(".hist-scroll");
      scroller.addEventListener("scroll", () => { if (!frame) frame = requestAnimationFrame(paint); }, { passive: true });
      scroller.addEventListener("click", (e) => {
        const r = e.target.closest("[data-n]");
        if (r && ed) E.jump(ed, Number(r.dataset.n));
      });
      scroller.addEventListener("keydown", (e) => {
        if (!ed) return;
        if (e.key === "ArrowUp" || e.key === "ArrowDown") {
          e.preventDefault();
          E.jump(ed, Math.max(0, Math.min(ed.rows.length - 1, ed.pos + (e.key === "ArrowUp" ? -1 : 1))));
        }
      });
    }
    paint();
  }
  function paint() {
    frame = 0;
    if (!scroller || !ed) return;
    const rows = ed.rows, n = rows.length, list = scroller.querySelector(".hist-list");
    const virtual = n > VIRTUAL_FROM;
    let first = 0, last = n;
    if (virtual) {
      const top = scroller.scrollTop, h = scroller.clientHeight || 240;
      first = Math.max(0, Math.floor(top / ROW) - OVERSCAN);
      last = Math.min(n, Math.ceil((top + h) / ROW) + OVERSCAN);
    }
    list.style.height = virtual ? `${n * ROW}px` : "";
    list.style.position = virtual ? "relative" : "";
    let html = "";
    for (let k = first; k < last; k++) {
      const r = rows[k], cls = k === ed.pos ? "cur" : k > ed.pos ? "redo" : "";
      html += `<div class="hist-row ${cls}" role="option" aria-selected="${k === ed.pos}" data-n="${k}"` +
        `${virtual ? ` style="position:absolute;left:0;right:0;top:${k * ROW}px"` : ""}>` +
        `<span class="hist-l">${esc(r.label)}${r.rev === null ? ' <span class="hist-wait" title="Waiting for the server">…</span>' : ""}</span>` +
        `<span class="hist-t">${time(r.t)}</span><span class="hist-n">${r.n_atoms} atoms</span></div>`;
    }
    list.innerHTML = html;
    if (!virtual) {
      const cur = list.querySelector(".hist-row.cur");
      if (cur) cur.scrollIntoView({ block: "nearest" });
    }
  }
  return { update, rows: () => (scroller ? scroller.querySelectorAll(".hist-row").length : 0) };
}
