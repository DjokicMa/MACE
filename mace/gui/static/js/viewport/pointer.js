/* The structure viewer's pointer (spec §2.1): the same in every tool, as
 * in OVITO, Avogadro and Blender —
 *   wheel: zoom toward the view center, or (Settings › Viewport › Zoom
 *   toward: Mouse pointer) toward the cursor; middle-drag: pan; right-drag: orbit
 *   (about the view's center, marked while the drag lasts);
 *   Alt+left-drag: orbit from any tool (laptop trackpads); Ctrl (⌘) held
 *   while orbiting inverts the angle snap (a snapping turn goes free, a free one snaps);
 *   right-click without dragging (< 4 px): the context menu —
 * and the left button per tool: Select (click picks, drag draws a box,
 * double-click selects the symmetry orbit, triple-click the fragment), Lasso
 * (a free polygon), Orbit / Pan / Zoom (drag does that), and the edit tools,
 * which register their own handlers. Nothing runs while the pointer rests:
 * hover work is one task per animation frame, only after a move. */
import * as G from "../geom/core.js";
import * as W from "../geom/client.js";

const CLICK_PX = 4, MULTI_MS = 400, MULTI_PX = 6;

export function attachPointer(el, deps) {
  const wrap = el.closest(".viewer-wrap");
  wrap.insertAdjacentHTML("beforeend", `<canvas class="vp-overlay" aria-hidden="true"></canvas>`);
  const overlay = wrap.querySelector(".vp-overlay");
  let drag = null, lastClick = { t: 0, x: 0, y: 0, n: 0 }, hoverFrame = 0, hoverAt = null, lastHoverK = -1, shiftDown = false;
  let lastPos = null;
  const tools = deps.tools || {};

  // the pointer in the viewer's own px (what the picker and the camera use): the UI scale zooms the shell
  const local = (e) => window.maceZoom.point(el, e.clientX, e.clientY);
  const mod = (e) => (deps.isMac ? e.metaKey : e.ctrlKey);
  const modeOf = (e) => (e.shiftKey && mod(e) ? "subtract" : e.shiftKey ? "add" : mod(e) ? "toggle" : "replace");

  /* ---------------------------------------------------------------- overlay */
  function sizeOverlay() {
    // the viewer's own px, drawn at the screen's resolution (the UI scale zooms it)
    const w = el.offsetWidth, h = el.offsetHeight, k = (devicePixelRatio || 1) * window.maceZoom.of(el);
    overlay.style.left = el.offsetLeft + "px"; overlay.style.top = el.offsetTop + "px";
    overlay.style.width = w + "px"; overlay.style.height = h + "px";
    if (overlay.width !== Math.round(w * k) || overlay.height !== Math.round(h * k)) {
      overlay.width = Math.round(w * k); overlay.height = Math.round(h * k);
    }
    const g = overlay.getContext("2d");
    g.setTransform(k, 0, 0, k, 0, 0);
    return g;
  }
  function clearOverlay() {
    const g = overlay.getContext("2d");
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, overlay.width, overlay.height);
    overlay.classList.remove("on");
  }
  function drawRegion(points, box) {
    const g = sizeOverlay();
    g.clearRect(0, 0, overlay.width, overlay.height);
    const accent = deps.cssVar("--accent");
    g.fillStyle = accent; g.strokeStyle = accent; g.lineWidth = 1.5;
    g.beginPath();
    if (box) { const [a, b] = points; g.rect(Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1])); }
    else points.forEach(([x, y], k) => (k ? g.lineTo(x, y) : g.moveTo(x, y)));
    if (!box) g.closePath();
    g.globalAlpha = 0.15; g.fill(); g.globalAlpha = 1; g.stroke();
    overlay.classList.add("on");
  }

  /* ---------------------------------------------------------------- selection by region */
  /** Select what a box or lasso holds. Up to 2000 drawn atoms the hit test
   * runs here; above that in the worker, from the camera matrix (spec
   * §8.2.6). The main thread's share (pointer-up to post, result to
   * selection built) is measured for the §8.8 budget. */
  async function regionSelect(points, box, mode) {
    const disp = deps.scene.display;
    if (!disp) return;
    const t0 = performance.now();
    let main = 0, t1 = t0, hits, worker = 0;
    const rect = box ? { x0: points[0][0], y0: points[0][1], x1: points[1][0], y1: points[1][1] } : null;
    if (disp.n <= 2000) {
      const pr = deps.scene.projected();
      hits = box ? G.boxSelect(pr.uv, rect) : G.lassoSelect(pr.uv, points);
    } else {
      const c = deps.cam().calibrate(), cart = disp.cart.slice();
      main += performance.now() - t0;
      const sent = performance.now();
      const r = await W.job(box ? "boxSelect" : "lassoSelect", { P: Float64Array.from(c.P), cart, rect, poly: box ? null : points },
        [cart.buffer]);
      t1 = performance.now();
      worker = t1 - sent;
      if (r.superseded) return;
      hits = r.hits;
    }
    const base = new Set();
    for (const k of hits) base.add(disp.base[k]);
    deps.select([...base], mode);
    main += performance.now() - t1;
    window.__maceBox = { mainMs: main, workerMs: worker, n: disp.n, hits: base.size };
    performance.measure("ui:boxselect", { start: t0, end: performance.now() });
  }

  /* ---------------------------------------------------------------- pointer events */
  el.addEventListener("pointerdown", (e) => {
    if (!deps.active()) return;
    const [x, y] = local(e);
    lastPos = [x, y];
    deps.focus();
    const tool = deps.tool();
    // a right click during a tool's own gesture (a Move grab) cancels it, no menu
    if (e.button === 2 && tools[tool] && tools[tool].right && tools[tool].right(e)) { e.preventDefault(); return; }
    let kind;
    if (e.button === 1) kind = "pan";
    else if (e.button === 2) kind = "right";
    else if (e.button === 0 && e.altKey) kind = "orbit";
    else if (e.button === 0) {
      if (tools[tool] && tools[tool].down && tools[tool].down(e, x, y) === true) kind = "tool";
      else kind = { select: "select", lasso: "lasso", orbit: "orbit", pan: "pan", zoom: "zoom" }[tool] || "click";
    } else return;
    e.preventDefault();
    try { el.setPointerCapture(e.pointerId); } catch (err) { /* synthetic events */ }
    drag = { kind, x0: x, y0: y, x, y, moved: false, button: e.button, mode: modeOf(e), poly: [[x, y]], id: e.pointerId, tool };
    if (["orbit", "pan", "zoom", "right"].includes(kind)) deps.interacting(true);
  });
  el.addEventListener("pointermove", (e) => {
    const [x, y] = local(e);
    lastPos = [x, y];
    if (!drag) { scheduleHover(x, y, e.shiftKey); if (tools[deps.tool()] && tools[deps.tool()].hover) tools[deps.tool()].hover(e, x, y); return; }
    // under 4 px it is still a click; past it, the drag starts from where the button went down
    if (!drag.moved && Math.hypot(x - drag.x0, y - drag.y0) < CLICK_PX) return;
    drag.moved = true;
    const dx = x - drag.x, dy = y - drag.y;
    drag.x = x; drag.y = y;
    const k = drag.kind;
    if (k === "orbit" || k === "right") {
      // the rotation center shows while the drag turns the view (gone on release)
      if (!drag.pivot && deps.pivot) { drag.pivot = true; deps.pivot(true); }
      // Ctrl (⌘) held inverts the angle snap for the rest of the drag (Settings › Viewport › Angle snap)
      deps.cam().orbit(dx, dy, e.shiftKey && k === "orbit" && deps.tool() === "orbit" && !e.altKey, mod(e));
    }
    else if (k === "pan") deps.cam().pan(dx, dy);
    else if (k === "zoom") zoomBy(Math.exp(-dy * 0.01), drag.x0, drag.y0);
    else if (k === "select") drawRegion([[drag.x0, drag.y0], [x, y]], true);
    else if (k === "lasso") {
      const last = drag.poly[drag.poly.length - 1];
      if (Math.hypot(x - last[0], y - last[1]) >= 3) drag.poly.push([x, y]);
      drawRegion(drag.poly, false);
    } else if (k === "tool" && tools[drag.tool].move) tools[drag.tool].move(e, x, y, drag);
  });
  function end(e, canceled = false) {
    if (!drag || (e.pointerId !== undefined && e.pointerId !== drag.id)) return;
    const d = drag;
    drag = null;
    try { el.releasePointerCapture(d.id); } catch (err) { /* not captured */ }
    if (["orbit", "pan", "zoom", "right"].includes(d.kind)) deps.interacting(false);
    if (d.kind === "orbit" || d.kind === "right") deps.cam().endOrbit();
    if (d.pivot) deps.pivot(false);
    if (canceled) { clearOverlay(); return; }
    const [x, y] = local(e);
    if (d.kind === "right") {
      if (!d.moved) deps.context(deps.scene.pickAt(x, y), e.clientX, e.clientY);
      return;
    }
    if (d.kind === "tool") { if (tools[d.tool].up) tools[d.tool].up(e, x, y, d); return; }
    if ((d.kind === "select" || d.kind === "lasso") && d.moved) {
      clearOverlay();
      if (d.kind === "lasso" && d.poly.length < 3) return;
      regionSelect(d.kind === "select" ? [[d.x0, d.y0], [x, y]] : d.poly, d.kind === "select", d.mode);
      return;
    }
    if (d.moved) return;
    // a click: count it (double, triple) and act
    const now = performance.now();
    const n = now - lastClick.t < MULTI_MS && Math.hypot(x - lastClick.x, y - lastClick.y) < MULTI_PX ? lastClick.n + 1 : 1;
    lastClick = { t: now, x, y, n };
    const k = deps.scene.pickAt(x, y);
    if (d.kind === "select" || d.kind === "lasso" || d.kind === "click") {
      if (d.kind === "click" && tools[d.tool] && tools[d.tool].click) { tools[d.tool].click(e, x, y, k, n); return; }
      const disp = deps.scene.display;
      if (k < 0) { if (d.mode === "replace" && n === 1) deps.select([], "replace"); return; }
      const i = disp.base[k];
      if (n >= 3) deps.selectFragment(i, d.mode);
      else if (n === 2) deps.selectOrbit(i, d.mode);
      else deps.select([i], d.mode);
    } else if (tools[d.tool] && tools[d.tool].click) tools[d.tool].click(e, x, y, k, n);
  }
  el.addEventListener("pointerup", (e) => end(e));
  el.addEventListener("pointercancel", (e) => end(e, true));
  el.addEventListener("lostpointercapture", (e) => { if (drag && e.pointerId === drag.id) end(e); });
  el.addEventListener("contextmenu", (e) => { e.preventDefault(); e.stopPropagation(); });
  el.addEventListener("wheel", (e) => {
    if (!deps.active()) return;
    e.preventDefault();
    const [x, y] = local(e);
    const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaMode === 2 ? e.deltaY * 400 : e.deltaY;
    // scrolling up (deltaY < 0) zooms in; Settings › Viewport › Invert scroll zoom flips it
    const sign = deps.invertWheel && deps.invertWheel() ? -1 : 1;
    zoomBy(Math.exp(-sign * Math.max(-300, Math.min(300, dy)) * 0.0015), x, y);
  }, { passive: false });
  /* Zoom toward (Settings › Viewport, read per event): the view center stays put (the rotation center, so
   * the turn stays about the same point), or the model point under the pointer does, as map apps zoom. */
  function zoomBy(f, x, y) {
    if (deps.zoomToward && deps.zoomToward() === "pointer") deps.cam().zoomAt(f, x, y);
    else deps.cam().zoom(f);
  }
  el.addEventListener("pointerleave", () => { if (!drag) { hoverAt = null; deps.hover(-1, false); lastHoverK = -1; } });

  /* ---------------------------------------------------------------- hover */
  function scheduleHover(x, y, shift) {
    hoverAt = [x, y, shift];
    if (hoverFrame) return;
    hoverFrame = requestAnimationFrame(() => {
      hoverFrame = 0;
      if (!hoverAt || !deps.active()) return;
      performance.mark("ui:hover:start");
      const k = deps.scene.display && deps.scene.display.n <= 20000 ? deps.scene.pickAt(hoverAt[0], hoverAt[1]) : -1;
      lastHoverK = k;
      deps.hover(k, hoverAt[2]);
      performance.measure("ui:hover", "ui:hover:start");
    });
  }
  const onShift = (e) => {
    if (e.key !== "Shift") return;
    const down = e.type === "keydown";
    if (down === shiftDown) return;
    shiftDown = down;
    if (lastHoverK >= 0) deps.hover(lastHoverK, down);
  };
  addEventListener("keydown", onShift);
  addEventListener("keyup", onShift);

  return {
    overlay, clearOverlay,
    get dragging() { return !!drag; },
    cancel() { if (drag) { if (drag.pivot && deps.pivot) deps.pivot(false); drag = null; clearOverlay(); deps.interacting(false); deps.cam().endOrbit(); } },
    /** Where the pointer last was over the viewer (CSS px), or its center. */
    lastPos() { return lastPos || [el.offsetWidth / 2, el.offsetHeight / 2]; },
    regionSelect,
  };
}
