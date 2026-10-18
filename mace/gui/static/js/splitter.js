/* A splitter's pointer, double-click and keyboard handling, shared by the
 * shell's panel splitters (explorer, inspector, bottom panel) and the
 * explorer's section splitters (sidebar.js).
 *
 * o: {axis: "x"|"y", sign: number or () => number, step (px, default 16),
 *     min(), max(), value(), set(v) (v already clamped and rounded),
 *     zoom() (the CSS zoom of the sized element: window.maceZoom.of),
 *     reset() (double-click; default set(def)), def, commit() (save),
 *     release(raw, base) (optional: the drop's unclamped size; true = handled, no set or commit)}
 * The values are the element's own px: a screen px of pointer travel is
 * 1/zoom of them at a UI scale other than 100 %. Returns {refresh} (aria). */
export function bindSplitter(el, o) {
  const axis = o.axis || "x", step = o.step || 16;
  const sign = () => (typeof o.sign === "function" ? o.sign() : o.sign || 1);
  const lim = (v) => Math.round(Math.min(o.max(), Math.max(o.min(), v)));
  const refresh = () => {
    el.setAttribute("aria-valuemin", Math.round(o.min())); el.setAttribute("aria-valuemax", Math.round(o.max()));
    el.setAttribute("aria-valuenow", Math.round(o.value()));
  };
  const set = (v) => { o.set(lim(v)); el.setAttribute("aria-valuenow", Math.round(o.value())); };
  refresh();
  el.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault(); el.setPointerCapture(e.pointerId); el.classList.add("dragging");
    refresh();
    const start = axis === "x" ? e.clientX : e.clientY, base = o.value(), z = o.zoom(), s = sign();
    let frame = 0;
    const to = (ev) => set(base + s * ((axis === "x" ? ev.clientX : ev.clientY) - start) / z);
    const move = (ev) => { cancelAnimationFrame(frame); frame = requestAnimationFrame(() => to(ev)); };
    let done = false;
    const end = () => {
      done = true;
      cancelAnimationFrame(frame); el.classList.remove("dragging");
      el.removeEventListener("pointermove", move); el.removeEventListener("pointerup", up);
      el.removeEventListener("pointercancel", cancel); el.removeEventListener("lostpointercapture", lost);
      try { if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId); } catch (x) { /* gone */ }
    };
    // the release point lands before the save (the last move may still wait for its frame)
    function up(ev) {
      if (done) return;
      end();
      // release(raw, base): the unclamped size at release; true when it handled the drop itself
      if (o.release && o.release(base + s * ((axis === "x" ? ev.clientX : ev.clientY) - start) / z, base)) return;
      to(ev); o.commit();
    }
    // the browser took the pointer (a touch became a scroll, a dialog): the drag never happened
    function cancel() { if (done) return; end(); set(base); }
    // the capture went without a pointerup (the element was moved, another capture): keep the size reached
    function lost() { if (done) return; end(); o.commit(); }
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", cancel);
    el.addEventListener("lostpointercapture", lost);
  });
  el.addEventListener("dblclick", () => { if (o.reset) o.reset(); else { set(o.def); o.commit(); } refresh(); });
  el.addEventListener("keydown", (e) => {
    const d = { ArrowLeft: -step, ArrowRight: step, ArrowUp: -step, ArrowDown: step }[e.key];
    if (d) { e.preventDefault(); refresh(); set(o.value() + sign() * d); o.commit(); }
  });
  return { refresh };
}
