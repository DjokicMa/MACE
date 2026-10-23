/* The canvas the Rotate and Move gizmos draw on (spec §2.1, 1.1.6), over the
 * structure viewer and under its chips: the viewer's own px at the screen's
 * resolution (the UI scale zooms it), redrawn at most once per animation frame,
 * and only while a gizmo tool is active. Colors come from the 3D view's own
 * tokens (MACE.vpVar), so a White or Black viewer background keeps the wheels
 * and arrows legible in any theme. Also the floating panel of numeric fields
 * (bottom of the viewer, right of the axis triad). */
export function createGizmoOverlay(el, paint) {
  const wrap = el.closest(".viewer-wrap");
  wrap.insertAdjacentHTML("beforeend", `<canvas class="vp-gizmo" aria-hidden="true"></canvas>`);
  const canvas = wrap.querySelector(".vp-gizmo");
  let frame = 0, shown = false;
  const zoom = () => (window.maceZoom ? window.maceZoom.of(el) : 1);

  function context() {
    const w = el.offsetWidth, h = el.offsetHeight, k = (window.devicePixelRatio || 1) * zoom();
    canvas.style.left = el.offsetLeft + "px"; canvas.style.top = el.offsetTop + "px";
    canvas.style.width = w + "px"; canvas.style.height = h + "px";
    if (canvas.width !== Math.round(w * k) || canvas.height !== Math.round(h * k)) {
      canvas.width = Math.round(w * k); canvas.height = Math.round(h * k);
    }
    const g = canvas.getContext("2d");
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, canvas.width, canvas.height);
    g.setTransform(k, 0, 0, k, 0, 0);
    return { g, w, h };
  }
  function draw() {
    frame = 0;
    const c = context();
    shown = !!paint(c.g, c.w, c.h);
    canvas.classList.toggle("on", shown);
  }
  /** Redraw on the next frame (any number of calls, one paint). */
  function schedule() { if (!frame) frame = requestAnimationFrame(draw); }
  function hide() {
    if (frame) { cancelAnimationFrame(frame); frame = 0; }
    if (!shown) return;
    const g = canvas.getContext("2d");
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, canvas.width, canvas.height);
    canvas.classList.remove("on");
    shown = false;
  }
  if (window.ResizeObserver) new ResizeObserver(schedule).observe(el);
  return { canvas, wrap, schedule, draw, hide, get shown() { return shown; } };
}

/** A token as the 3D view draws it. */
export function vpColor(name, fallback) {
  const M = window.MACE;
  let v = "";
  try { v = M && M.vpVar ? M.vpVar(name) : getComputedStyle(document.documentElement).getPropertyValue(name).trim(); } catch (e) { /* no probe yet */ }
  return v || fallback || "";
}

/** A polyline of {x, y, front} points: the near half solid, the far half thin and faint (the wheel goes behind). */
export function strokeRing(g, pts, color, width, { highlight = false, alpha = 1 } = {}) {
  g.save();
  g.strokeStyle = color;
  g.lineCap = "round";
  for (const front of [false, true]) {
    g.globalAlpha = alpha * (front ? 1 : 0.35);
    g.lineWidth = (front ? width : Math.max(1, width * 0.6)) + (highlight && front ? 2 : 0);
    g.beginPath();
    let open = false;
    for (let k = 0; k < pts.length; k++) {
      const p = pts[k];
      const on = p.front === front || (k > 0 && pts[k - 1].front === front);
      if (!on) { open = false; continue; }
      if (!open) { g.moveTo(p.x, p.y); open = true; } else g.lineTo(p.x, p.y);
    }
    g.stroke();
  }
  g.restore();
}

/** A label with a halo of the viewer's background, so it reads over atoms. */
export function haloText(g, text, x, y, color, bg, { align = "center", font = "600 12px system-ui, sans-serif" } = {}) {
  g.save();
  g.font = font;
  g.textAlign = align;
  g.textBaseline = "middle";
  g.lineJoin = "round";
  g.lineWidth = 3.5;
  g.strokeStyle = bg;
  g.strokeText(text, x, y);
  g.fillStyle = color;
  g.fillText(text, x, y);
  g.restore();
}

/** A gizmo panel: a title bar with a Hide/Show toggle (kept per tool), the body, and the keys behind "Keys".
 *  Returns the panel element; the caller fills .gz-body. */
export function gizmoPanel(id, title, help, profile) {
  const panel = document.createElement("div");
  panel.className = "vp-gz-panel";
  panel.dataset.for = id;
  panel.setAttribute("role", "group");
  panel.setAttribute("aria-label", title);
  panel.innerHTML = `<div class="gz-head"><b>${title}</b><span class="gz-k gz-sum"></span>
      <button type="button" class="gz-collapse" aria-expanded="true" title="Hide the fields">▾</button></div>
    <div class="gz-body"></div>
    <details class="gz-help"><summary>Keys</summary><div class="gz-hint">${help}</div></details>`;
  const btn = panel.querySelector(".gz-collapse");
  const set = (open) => {
    panel.classList.toggle("collapsed", !open);
    btn.setAttribute("aria-expanded", String(open));
    btn.textContent = open ? "▾" : "▸";
    btn.title = open ? "Hide the fields" : "Show the fields";
  };
  set(profile.ui(`gizmoPanel.${id}`, "open") !== "closed");
  btn.addEventListener("click", () => {
    const open = panel.classList.contains("collapsed");
    set(open);
    profile.setUi(`gizmoPanel.${id}`, open ? "open" : "closed");
  });
  return panel;
}
