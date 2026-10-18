/* Two editor groups side by side (plan E11): compare two structures, outputs,
 * plots or runs.
 *
 * The left group is this page. The right group is the same app in a frame
 * (`?pane=side`: no menu bar, explorer, inspector or status bar of its own),
 * so every document kind works in either group with all its controls, and
 * either group can edit. The two talk through `window.__macePane` (same
 * origin):
 *
 * - View › Editor Groups: Split Editor Right (Mod+\), Move Document to Other
 *   Group (Mod+Shift+\), Focus Other Group (Mod+Alt+\), Focus Left Group,
 *   Focus Right Group, Close Group, Link Cameras; Open to the Side
 *   (Explorer); Move to Other Group, Move Left, Move Right (tab menu).
 * - Dragging a tab (from either group): onto a tab strip it goes in at that
 *   place (the same group: a new order; the other group: it moves there);
 *   onto the other group's editor area it moves there; onto the right half
 *   of a single group it splits the editor. Esc cancels the drag. The window
 *   draws every drop marker and does every move (dragTab, dragFromPane); the
 *   right group only reports its pointer, in its own px.
 * - Closing the last tab of the right group, or moving it out, closes that
 *   group; closing the last tab of the left group brings the right group's
 *   documents left (when all of them can move: saved, with a file).
 * - The focused group has the accent ring; menu and keyboard commands that
 *   act on a document go to it; the explorer opens files into it.
 * - The Inspector and the bottom panel follow the focused group: the right
 *   group's frame reaches under its own column into the bottom panel's row
 *   (its own bottom panel; the window's narrows to the left group) and, while
 *   it has the focus, over the Inspector's column (its own Inspector; the
 *   window's is inert under it). chrome() keeps the frame and the pane's
 *   panels the size of the window's.
 * - Link Cameras: two 3D views share their rotation (each keeps its own zoom
 *   and center).
 * - Combined analyses read the other group: a peak clicked in a spectrum opens
 *   its mode there, Color Density by Potential and Volume Difference take the
 *   other group's volume, Compare the Two Groups' Structures compares them.
 * - The split (open, ratio, linked, the right group's documents) is kept per
 *   folder in the profile (ui.split.<folder hash>).
 */

export const PANE = new URLSearchParams(location.search).get("pane") === "side";

/* Commands that act on a document: they run in the focused group. Everything
 * else (files, folders, panels, settings, help, the groups themselves) belongs
 * to the window. */
const DOC_CATEGORIES = new Set(["Edit", "Select", "Atom", "Analysis", "Trajectory", "Viewport", "Tab", "Point Defect",
  "Tools", "Build"]);
const DOC_VIEW = /^view\.(along|rotate|reset|frameSelection|projection|cell|boundary|bonds|labels|axes|measurements|polyhedra|range|style)/;
const DOC_FILE = /^file\.(save|saveAs|revert|exportImage|export\.|copyImage|closeTab)$/;
const GROUP_CMDS = /^view\.(splitRight|moveToOtherGroup|focusOtherGroup|focusLeftGroup|focusRightGroup|closeGroup|linkCameras)$/;
export function isDocCommand(id, category) {
  if (GROUP_CMDS.test(id)) return false;
  if (/^tab\.(moveToOtherGroup|moveLeft|moveRight)$/.test(id)) return true;
  if (DOC_VIEW.test(id) || DOC_FILE.test(id)) return true;
  return DOC_CATEGORIES.has(category || "");
}

const MOVABLE = (d) => !!(d && d.path);

/** host: the shell's group API (window.__macePane of this page) plus
 *  toast, profile, dirHash(), afterChange(). */
export function createGroups(host) {
  const g = { open: false, ratio: 0.5, linked: false, focused: "main", frame: null, ready: null, restoring: false, restoreGen: 0, moving: 0, focusTick: 0 };
  const root = () => document.getElementById("editor-groups");
  const side = () => document.getElementById("group-side");
  const splitEl = () => document.getElementById("split-groups");
  const other = () => (PANE ? window.parent.__macePane : (g.frame && g.frame.contentWindow && g.frame.contentWindow.__macePane) || null);

  /* ------------------------------------------------------------ the pane (right group) side */
  if (PANE) {
    document.documentElement.classList.add("pane-side");
    const parentGroups = () => window.parent.__maceGroups;
    const tell = (fn) => { try { const pg = parentGroups(); if (pg) pg[fn](); } catch (e) { /* the window is going */ } };
    addEventListener("pointerdown", () => tell("focusSideFromPane"), true);
    addEventListener("focusin", () => tell("focusSideFromPane"), true);
    return {
      pane: true, isOpen: () => true, sideFocused: () => true, other,
      changed: () => tell("sideChanged"),
      shouldForward: () => null,
      forwardGlobal: (id, category) => (!isDocCommand(id, category) ? window.parent.__macePane : null),
      ctxOverride: (c) => ({ ...c, splitOpen: true, linkedCameras: false }),
      otherVolume: () => otherActive("volume"),
      otherStructure: () => otherActive("structure"),
      openModesOther: (path, opts) => { const o = other(); if (!o) return false; o.openModes(path, opts); parentGroups().focusGroup("main"); return true; },
      // the window moves it (and closes this group when it was the last)
      moveToOther: (d) => (d ? parentGroups().moveFromSide(d.id) : false),
      dragTab: (e, d, onDropped) => paneDrag(e, d, onDropped),
      docClosed: (n) => { if (n === 0) tell("sideEmptied"); },
      restore: () => {},
      state: () => ({ pane: true }),
    };
  }

  /* A tab of this group dragged: the window draws the markers and drops it.
   * The pointer is captured here once the drag starts, so the moves keep
   * coming while it is over the window's part of the screen. */
  function paneDrag(e, d, onDropped) {
    if (e.button !== 0 || !d) return;
    const pg = window.parent.__maceGroups, el = document.getElementById("doc-tabs");
    if (!pg) return;
    const x0 = e.clientX, y0 = e.clientY, pid = e.pointerId;
    let dragging = false;
    const move = (ev) => {
      if (!dragging) {
        if (Math.hypot(ev.clientX - x0, ev.clientY - y0) < 8) return;
        dragging = true;
        try { el.setPointerCapture(pid); } catch (err) { /* the pointer is gone */ }
        pg.dragStart("side", { id: d.id, title: d.title });
      }
      pg.dragMove(ev.clientX, ev.clientY, true);
    };
    const finish = (ev, cancel) => {
      removeEventListener("pointermove", move, true); removeEventListener("pointerup", up, true);
      removeEventListener("pointercancel", onCancel, true); removeEventListener("keydown", esc, true);
      try { if (el.hasPointerCapture(pid)) el.releasePointerCapture(pid); } catch (err) { /* released */ }
      if (!dragging) return;
      if (onDropped) onDropped();
      if (cancel) pg.dragCancel(); else pg.dragEnd(ev.clientX, ev.clientY, true);
    };
    const up = (ev) => finish(ev, false);
    const onCancel = (ev) => finish(ev, true);
    const esc = (ev) => { if (ev.key === "Escape" && dragging) { ev.preventDefault(); ev.stopPropagation(); finish(ev, true); } };
    addEventListener("pointermove", move, true);
    addEventListener("pointerup", up, true);
    addEventListener("pointercancel", onCancel, true);
    addEventListener("keydown", esc, true);
  }

  function otherActive(kind) {
    try {
      const o = other();
      const a = o && o.active();
      if (!a) return null;
      if (kind === "structure" ? ["structure", "trajectory"].includes(a.kind) : a.kind === kind) return a;
    } catch (e) { /* the other group is reloading */ }
    return null;
  }


  /* ------------------------------------------------------------ the window (left group) side */
  /* The right group's frame over the bottom panel's row and, while it has the
   * focus, the Inspector's column; the pane shows its own panels there. */
  const shellEl = () => document.getElementById("shell");
  const shown = (el) => !!el && getComputedStyle(el).display !== "none" && el.getBoundingClientRect().height > 0;
  function chrome() {
    const f = g.frame, insp = document.getElementById("inspector");
    if (!g.open || !f) { if (insp) insp.inert = false; return; }
    const sh = shellEl(), z = (sh && parseFloat(getComputedStyle(sh).zoom)) || 1;
    const sr = side().getBoundingClientRect();
    const b = document.getElementById("bottom");
    let bottomH = 0, extraB = 0;
    if (shown(b)) {
      const br = b.getBoundingClientRect();
      bottomH = br.height / z;
      extraB = Math.max(0, (br.bottom - sr.bottom) / z);
    }
    // the Inspector as a column (not the drawer of a narrow window), not while Setup fills it
    let inspW = 0, extraR = 0;
    if (g.focused === "side" && shown(insp) && getComputedStyle(insp).position === "static" && !host.setupOpen()) {
      const ir = insp.getBoundingClientRect();
      inspW = ir.width / z;
      extraR = Math.max(0, (ir.right - sr.right) / z);
    }
    f.style.width = extraR ? `calc(100% + ${extraR}px)` : "";
    f.style.height = extraB ? `calc(100% + ${extraB}px)` : "";
    side().classList.toggle("covers-inspector", extraR > 0);
    if (insp) insp.inert = extraR > 0;
    try { const o = other(); if (o && o.chrome) o.chrome({ insp: inspW, bottom: bottomH }); } catch (e) { /* the frame is loading */ }
  }
  let chromeFrame = 0;
  const scheduleChrome = () => { cancelAnimationFrame(chromeFrame); chromeFrame = requestAnimationFrame(chrome); };

  function css() {
    const r = root();
    if (!r) return;
    r.classList.toggle("is-split", g.open);
    r.style.setProperty("--split-left", `${Math.round(g.ratio * 1000) / 10}%`);
    const c = document.getElementById("center");
    if (c) { c.classList.toggle("split-open", g.open); c.style.setProperty("--split-left", `${Math.round(g.ratio * 1000) / 10}%`); }
    side().hidden = !g.open;
    splitEl().hidden = !g.open;
    document.getElementById("group-main").classList.toggle("focused", g.open && g.focused === "main");
    side().classList.toggle("focused", g.open && g.focused === "side");
    document.getElementById("shell").classList.toggle("side-focused", g.open && g.focused === "side");
    // 3Dmol and plotly size themselves on a window resize
    requestAnimationFrame(() => dispatchEvent(new Event("resize")));
    scheduleChrome();
  }

  function persist() {
    if (g.restoring || !host.profile) return;
    let docs = [], activePath = null;
    try {
      const o = other();
      if (o && g.open) {
        docs = o.docs().filter((x) => x.path).map((x) => ({ path: x.path, kind: x.kind, as: x.as || "" }));
        activePath = (o.active() || {}).path || null;
      }
    } catch (e) { /* the frame is loading: keep what was stored */ return; }
    host.profile.setUi(`split.${host.dirHash()}`, g.open ? { open: true, ratio: g.ratio, linked: g.linked, docs, active: activePath }
      : { open: false, ratio: g.ratio, linked: g.linked });
  }

  function ensureFrame() {
    if (g.frame) return g.ready;
    const f = document.createElement("iframe");
    f.className = "group-frame";
    f.title = "Right editor group";
    f.setAttribute("aria-label", "Right editor group");
    f.src = "./?pane=side";
    side().appendChild(f);
    g.frame = f;
    g.ready = new Promise((resolve, reject) => {
      const t0 = performance.now();
      const poll = () => {
        try {
          const w = f.contentWindow;
          if (w && w.__macePane && w.document.documentElement.dataset.appReady === "1") { resolve(w.__macePane); scheduleChrome(); return; }
        } catch (e) { /* not loaded yet */ }
        if (performance.now() - t0 > 60000) { reject(new Error("the right group did not load")); return; }
        setTimeout(poll, 50);
      };
      poll();
    });
    return g.ready;
  }

  async function openSplit({ focus = "side" } = {}) {
    const fresh = !g.open;
    g.open = true;
    css();
    const api = await ensureFrame();
    if (fresh) focusGroup(focus);
    if (g.linked) startLink();
    persist();
    return api;
  }

  function closeGroup(force = false) {
    if (!g.open) return;
    try {
      const o = other();
      const dirty = o ? o.docs().filter((x) => x.dirty) : [];
      if (dirty.length && !force && !confirm(`${dirty.map((x) => x.title).join(", ")} in the right group ${dirty.length === 1 ? "has" : "have"} unsaved changes. Close the group anyway?`)) return;
    } catch (e) { /* closing anyway */ }
    g.open = false;
    // a restore still opening this group never finishes now (its frame is gone): it must not leave the
    // group "restoring", which blocked every later collapse, merge and save of the layout
    g.restoreGen++; g.restoring = false;
    stopLink();
    if (g.frame) { g.frame.remove(); g.frame = null; g.ready = null; }
    g.focused = "main";
    css();
    chrome();
    persist();
    host.afterChange();
  }

  function focusGroup(which) {
    if (!g.open) which = "main";
    const changed = g.focused !== which;
    g.focused = which;
    g.focusTick++;
    css();
    if (which === "side" && g.frame) {
      try { g.frame.contentWindow.focus(); const o = other(); if (o) o.focusView(); } catch (e) { /* loading */ }
    } else if (which === "main" && changed) {
      host.focusView();
    }
    if (changed) host.afterChange();
  }

  /* A document of the right group (its active one, or the one with that id)
   * comes back left, at `index` of the left group's tabs (null: the end). */
  async function moveFromSide(id = null, index = null) {
    const o = other();
    let a = null;
    try { a = o && (id == null ? o.active() : o.docs().find((x) => x.id === id)); } catch (e) { a = null; }
    if (!a) { host.toast("The right group has no document", "info"); return false; }
    if (!a.path) { host.toast(`${a.title}: this document has no file, so it cannot move to the other group`, "info"); return false; }
    if (a.dirty) { host.toast(`${a.title}: save it first (moving reopens the file in the other group)`, "info"); return false; }
    g.moving++;
    let moved = false;
    try {
      o.close(a.id, true);
      await host.reopenAt(a, index);
      moved = true;
    } finally {
      moveDone();
      // a failed reopen still closes an emptied group
      if (!moved && !collapseIfEmpty()) persist();
    }
    focusGroup("main");
    host.announce(`${a.title} moved to the left group`);
    if (!collapseIfEmpty()) persist();
    return true;
  }

  /* A document of the left group (the active one when none is given) goes
   * right, at `index` of the right group's tabs (null: the end); with the
   * right group focused and no document given, its active one comes back. */
  async function moveToOther(d, index = null) {
    if (!d && g.open && g.focused === "side") return moveFromSide();
    d = d || host.activeDoc();
    if (!d) { host.toast("Open a document first", "info"); return false; }
    if (!MOVABLE(d)) { host.toast(`${d.title}: this document has no file, so it cannot move to the other group`, "info"); return false; }
    if (host.isDirty(d)) { host.toast(`${d.title}: save it first (moving reopens the file in the other group)`, "info"); return false; }
    g.moving++;
    try {
      const api = await openSplit({ focus: "side" });
      const desc = host.desc(d);
      host.closeDoc(d, true);
      await api.reopenAt(desc, index);
    } finally { moveDone(); }
    focusGroup("side");
    host.announce(`${d.title} moved to the right group`);
    persist();
    return true;
  }

  /* A move ended. A collapse asked for while moves ran (the pane emptied, two drags in quick succession)
   * runs once the last one is done, whichever move finishes last. */
  function moveDone() {
    g.moving--;
    if (!g.moving && g.recheck) {
      g.recheck = false;
      setTimeout(() => { if (!g.moving && collapseIfEmpty()) return; persist(); }, 0);
    }
  }

  /* The right group with no document left closes (not while it is being
   * filled or emptied by a move, nor the empty group Split Editor Right
   * opens). True when it closed. */
  function collapseIfEmpty() {
    if (g.open && g.moving) g.recheck = true;          // asked again when the last move ends (moveDone)
    if (!g.open || g.restoring || g.moving) return false;
    let n = -1;
    try { const o = other(); n = o ? o.docs().length : -1; } catch (e) { n = -1; }
    if (n !== 0) return false;
    closeGroup(true);
    return true;
  }

  /* The left group's last tab closed: the right group's documents come left
   * and the editor is one group again, when every one of them can move. */
  async function mergeSide() {
    if (!g.open || g.moving || g.restoring || host.docCount() > 0) return;
    const o = other();
    let list = [], act = null;
    try { list = o ? o.docs() : []; act = o ? o.active() : null; } catch (e) { return; }
    if (list.some((x) => !x.path || x.dirty)) return;          // the right group keeps what cannot move
    g.moving++;
    try {
      for (const x of list) { o.close(x.id, true); await host.reopenAt(x, null); }
      if (act && list.length > 1) await host.reopenAt(act, null);      // its active document stays active
    } finally { g.moving--; }
    closeGroup(true);
  }

  /* Split Editor Right: a second group with the active document (moved: one
   * file is edited in one place), or an empty one when it is the only one. */
  async function splitRight() {
    if (g.open) return moveToOther();
    const d = host.activeDoc();
    if (d && host.docCount() > 1 && MOVABLE(d) && !host.isDirty(d)) return moveToOther(d);
    await openSplit({ focus: "side" });
    if (d) host.toast("Drag a tab onto the right group, or use Open to the Side in the Explorer", "info");
    return true;
  }

  async function openToSide(path, opts = {}) {
    if (!path) return null;
    const api = await openSplit({ focus: "side" });
    const tick = g.focusTick;
    const r = await api.reopen({ path, kind: "", as: opts.as || "" });
    // a slow document: the focus moved meanwhile (a click, Focus Other Group) stays where it went
    if (g.focusTick === tick) focusGroup("side");
    persist();
    return r;
  }

  /* ------------------------------------------------------------ Link Cameras */
  // each group reads and sets its own view's rotation in its own page (a 3Dmol
  // viewer does not take another page's arrays)
  let linkFrame = 0, lastA = null, lastB = null;
  const same = (a, b) => !!a && !!b && a.every((x, i) => Math.abs(x - b[i]) < 1e-9);
  const here = () => window.__macePane;
  function quats() {
    let a = null, b = null;
    try { a = here().getQ(); } catch (e) { a = null; }
    try { const o = other(); b = o ? o.getQ() : null; } catch (e) { b = null; }
    return [a, b];
  }
  function linkTick() {
    linkFrame = 0;
    if (!g.linked || !g.open) return;
    const [qa, qb] = quats();
    if (qa && qb) {
      if (!same(qa, lastA) && !same(qa, qb)) {
        other().setQ(qa); lastA = qa; lastB = qa;
      } else if (!same(qb, lastB) && !same(qa, qb)) {
        here().setQ(qb); lastA = qb; lastB = qb;
      } else { lastA = qa; lastB = qb; }
    }
    linkFrame = requestAnimationFrame(linkTick);
  }
  function startLink() { if (!linkFrame) linkFrame = requestAnimationFrame(linkTick); }
  function stopLink() { if (linkFrame) cancelAnimationFrame(linkFrame); linkFrame = 0; }
  function toggleLink(on = !g.linked) {
    g.linked = !!on;
    if (g.linked && g.open) {
      // the right view takes the left one's rotation at once
      const [qa] = quats();
      try { if (qa && other()) { other().setQ(qa); lastA = lastB = qa; } } catch (e) { /* loading */ }
      startLink();
    } else stopLink();
    persist();
    host.afterChange();
  }

  /* ------------------------------------------------------------ the splitter */
  function initSplitter() {
    const s = splitEl();
    if (!s) return;
    const setRatio = (r) => { g.ratio = Math.max(0.2, Math.min(0.8, r)); css(); };
    s.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      s.setPointerCapture(e.pointerId);
      const box = root().getBoundingClientRect();
      if (g.frame) g.frame.style.pointerEvents = "none";
      const move = (ev) => setRatio((ev.clientX - box.left) / box.width);
      const up = () => {
        s.removeEventListener("pointermove", move); s.removeEventListener("pointerup", up);
        if (g.frame) g.frame.style.pointerEvents = "";
        persist();
      };
      s.addEventListener("pointermove", move); s.addEventListener("pointerup", up);
    });
    s.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      e.preventDefault();
      setRatio(g.ratio + (e.key === "ArrowRight" ? 0.05 : -0.05));
      persist();
    });
  }

  /* ------------------------------------------------------------ dragging a tab (either group) */
  /* All in the window's client px; the right group's points are mapped from
   * its own (the frame's place, and the UI scale it applies inside). */
  let drag = null;
  const rectOf = (el) => (el ? el.getBoundingClientRect() : null);
  const inside = (r, x, y) => !!r && r.width > 0 && x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;
  function frameMap() {
    const f = g.frame;
    if (!f || !f.contentWindow) return null;
    const fr = f.getBoundingClientRect(), w = f.contentWindow.innerWidth;
    const k = w ? fr.width / w : 1;
    return { x: (px) => fr.left + px * k, y: (py) => fr.top + py * k };
  }
  function overlay(cls) {
    let el = document.querySelector(`body > .${cls}`);
    if (!el) {
      el = document.createElement("div");
      el.className = cls;
      el.setAttribute("aria-hidden", "true");
      if (cls === "group-drop") el.innerHTML = "<span></span>";
      el.hidden = true;
      document.body.appendChild(el);
    }
    return el;
  }
  const place = (el, r) => Object.assign(el.style, { left: `${r.left}px`, top: `${r.top}px`, width: `${r.width}px`, height: `${r.height}px` });
  /* the tab strips: their box and their tabs' left and right edges */
  function strips() {
    const out = [];
    const el = document.getElementById("doc-tabs"), r = rectOf(el);
    if (r && r.width > 0) {
      out.push({ group: "main", rect: r, tabs: [...el.querySelectorAll(".doc-tab")].map((t) => {
        const b = t.getBoundingClientRect();
        return { id: Number(t.dataset.doc), left: b.left, right: b.right };
      }) });
    }
    if (g.open) {
      try {
        const o = other(), m = frameMap(), st = o && o.tabStrip();
        if (st && m) {
          const left = m.x(st.left), top = m.y(st.top);
          out.push({ group: "side", rect: { left, top, right: m.x(st.right), bottom: m.y(st.bottom), width: m.x(st.right) - left, height: m.y(st.bottom) - top },
            tabs: st.tabs.map((t) => ({ id: t.id, left: m.x(t.left), right: m.x(t.right) })) });
        }
      } catch (e) { /* the frame is loading */ }
    }
    return out;
  }
  /* the editor area a document can be dropped on, for the drag's source */
  function zoneFor(from) {
    if (!g.open) {
      const v = rectOf(document.getElementById("views"));
      return from === "main" && v ? { group: "side", split: true, rect: { left: v.left + v.width / 2, top: v.top, width: v.width / 2, height: v.height,
        right: v.right, bottom: v.bottom } } : null;
    }
    if (from === "main") return { group: "side", rect: rectOf(side()) };
    return { group: "main", rect: rectOf(document.getElementById("views")) };
  }
  /* where a drop at (x, y) would put the document: {group, index|null} */
  function targetAt(x, y) {
    for (const st of strips()) {
      if (x < st.rect.left || x > st.rect.right || y < st.rect.top - 4 || y > st.rect.bottom + 4) continue;
      const tabs = st.tabs.filter((t) => !(st.group === drag.from && t.id === drag.id));
      let i = tabs.findIndex((t) => x < (t.left + t.right) / 2);
      if (i < 0) i = tabs.length;
      const lx = i < tabs.length ? tabs[i].left : tabs.length ? tabs[tabs.length - 1].right : st.rect.left;
      return { group: st.group, index: i, line: { left: Math.max(st.rect.left, lx - 1), top: st.rect.top, width: 2, height: st.rect.bottom - st.rect.top } };
    }
    const z = drag.zone;
    if (z && inside(z.rect, x, y)) return { group: z.group, index: null, split: !!z.split };
    return null;
  }
  function dragStart(from, info) {
    if (drag) dragCancel();
    drag = { from, id: info.id, title: info.title || "", target: null, zone: zoneFor(from) };
    if (from === "main" && g.frame) g.frame.style.pointerEvents = "none";
    const z = overlay("group-drop");
    if (drag.zone && drag.zone.rect) {
      place(z, drag.zone.rect);
      z.querySelector("span").textContent = drag.zone.split ? `Split right with ${drag.title}`
        : `Move ${drag.title} to the ${drag.zone.group === "side" ? "right" : "left"} group`;
      z.hidden = false;
    }
    document.body.classList.add("tab-dragging");
    addEventListener("keydown", dragKey, true);
  }
  const mapPoint = (x, y, fromPane) => {
    if (!fromPane) return [x, y];
    const m = frameMap();
    return m ? [m.x(x), m.y(y)] : [x, y];
  };
  function dragMove(x, y, fromPane = false) {
    if (!drag) return;
    [x, y] = mapPoint(x, y, fromPane);
    const t = targetAt(x, y);
    drag.target = t;
    const z = overlay("group-drop"), line = overlay("tab-drop-line");
    z.classList.toggle("over", !!t && t.index === null && t.group !== drag.from);
    if (t && t.line) { place(line, t.line); line.hidden = false; } else line.hidden = true;
  }
  function dragDone() {
    removeEventListener("keydown", dragKey, true);
    if (g.frame) g.frame.style.pointerEvents = "";
    const z = overlay("group-drop"), line = overlay("tab-drop-line");
    z.hidden = true; z.classList.remove("over"); line.hidden = true;
    document.body.classList.remove("tab-dragging");
    const d = drag;
    drag = null;
    return d;
  }
  function dragCancel() { dragDone(); }
  function dragKey(e) { if (e.key === "Escape" && drag) { e.preventDefault(); e.stopPropagation(); dragCancel(); } }
  async function dragEnd(x, y, fromPane = false) {
    if (!drag) return false;
    dragMove(x, y, fromPane);
    const d = dragDone(), t = d.target;
    if (!t) return false;
    if (t.group === d.from) {
      if (t.index === null) return false;                   // its own editor area: where it is
      if (t.group === "main") return host.reorder(d.id, t.index);
      const o = other();
      const moved = o ? o.reorder(d.id, t.index) : false;
      persist();
      return moved;
    }
    if (d.from === "main") return moveToOther(host.docById(d.id), t.index);
    return moveFromSide(d.id, t.index);
  }
  /** Called from the left group's tab strip on pointerdown on a tab. */
  function dragTab(e, d, onDropped) {
    if (e.button !== 0 || !d) return;
    const x0 = e.clientX, y0 = e.clientY;
    let dragging = false;
    const move = (ev) => {
      if (!dragging) {
        if (Math.hypot(ev.clientX - x0, ev.clientY - y0) < 8) return;
        dragging = true;
        dragStart("main", { id: d.id, title: d.title });
      }
      if (drag) dragMove(ev.clientX, ev.clientY);
    };
    const finish = (ev, cancel) => {
      removeEventListener("pointermove", move, true); removeEventListener("pointerup", up, true);
      removeEventListener("pointercancel", onCancel, true);
      if (!dragging) return;
      if (onDropped) onDropped();
      if (cancel || !drag) dragCancel(); else dragEnd(ev.clientX, ev.clientY);
    };
    const up = (ev) => finish(ev, false);
    const onCancel = (ev) => finish(ev, true);
    addEventListener("pointermove", move, true);
    addEventListener("pointerup", up, true);
    addEventListener("pointercancel", onCancel, true);
  }

  /* ------------------------------------------------------------ restore per folder */
  async function restore() {
    const st = host.profile ? host.profile.ui(`split.${host.dirHash()}`) : null;
    if (!st || typeof st !== "object") return;
    g.ratio = typeof st.ratio === "number" ? Math.max(0.2, Math.min(0.8, st.ratio)) : 0.5;
    g.linked = !!st.linked;
    if (!st.open) { css(); return; }
    const my = ++g.restoreGen;
    g.restoring = true;
    try {
      const api = await openSplit({ focus: "main" });
      for (const x of Array.isArray(st.docs) ? st.docs : []) {
        if (x && typeof x.path === "string" && host.exists(x.path)) await api.reopen(x).catch(() => null);
      }
      if (st.active && host.exists(st.active)) await api.reopen({ path: st.active }).catch(() => null);
    } catch (e) { /* the folder changed: start with one group */ }
    finally { if (g.restoreGen === my) g.restoring = false; }
    if (g.restoreGen !== my) return;                  // closed meanwhile (closeGroup)
    focusGroup("main");
    persist();
  }

  const api = {
    pane: false, isOpen: () => g.open, sideFocused: () => g.open && g.focused === "side", linked: () => g.linked, other,
    openSplit, closeGroup, splitRight, moveToOther, moveFromSide, openToSide, toggleLink, focusGroup, restore, dragTab,
    focusOther: () => (g.open ? focusGroup(g.focused === "main" ? "side" : "main") : host.toast("There is only one editor group (Split Editor Right: Mod+\\)", "info")),
    focusLeft: () => focusGroup("main"),
    focusRight: () => (g.open ? focusGroup("side") : host.toast("There is only one editor group (Split Editor Right: Mod+\\)", "info")),
    /** The left group's tab count after a close: its last one closed brings the right group's documents left. */
    docClosed: (n) => { if (n === 0 && g.open && !g.moving && !g.restoring) setTimeout(() => { mergeSide(); }, 0); },
    // a tab of the right group being dragged (its own px)
    dragStart, dragMove, dragEnd, dragCancel, dragging: () => (drag ? { from: drag.from, target: drag.target } : null),
    /** A document command while the right group has the focus runs there. */
    shouldForward: (id, category) => (g.open && g.focused === "side" && isDocCommand(id, category) ? other() : null),
    forwardGlobal: () => null,
    ctxOverride: (c) => {
      if (!(g.open && g.focused === "side")) return { ...c, splitOpen: g.open, linkedCameras: g.linked };
      try {
        const o = other();
        const s = o ? o.ctx() : null;
        if (s) return { ...c, ...s, dialogOpen: c.dialogOpen || s.dialogOpen, zenMode: c.zenMode, openFolderOn: c.openFolderOn,
          splitOpen: true, linkedCameras: g.linked };
      } catch (e) { /* loading */ }
      return { ...c, splitOpen: g.open, linkedCameras: g.linked };
    },
    otherVolume: () => (g.open ? otherActive("volume") : null),
    otherStructure: () => (g.open ? otherActive("structure") : null),
    openModesOther: (path, opts) => {
      if (!g.open) return false;
      const o = other();
      if (!o) return false;
      o.openModes(path, opts);
      return true;
    },
    // from the pane
    focusSideFromPane: () => { if (g.open && g.focused !== "side") focusGroup("side"); },
    sideEmptied: () => setTimeout(collapseIfEmpty, 0),
    sideChanged: () => { persist(); host.afterChange(); },
    state: () => ({ open: g.open, ratio: g.ratio, linked: g.linked, focused: g.focused, moving: g.moving }),
    chrome: () => scheduleChrome(),
  };
  window.__maceGroups = api;
  initSplitter();
  // the panels' sizes and visibility (resized, hidden, zen, a narrow window): the frame follows
  if (typeof ResizeObserver === "function") {
    const ro = new ResizeObserver(() => { if (g.open) scheduleChrome(); });
    ["group-side", "bottom", "inspector", "center"].forEach((id) => { const el = document.getElementById(id); if (el) ro.observe(el); });
  }
  if (shellEl()) new MutationObserver(() => { if (g.open) scheduleChrome(); }).observe(shellEl(), { attributes: true, attributeFilter: ["class", "style"] });
  const center = document.getElementById("group-main");
  if (center) {
    center.addEventListener("pointerdown", () => { if (g.open && g.focused !== "main") focusGroup("main"); }, true);
    center.addEventListener("focusin", () => { if (g.open && g.focused !== "main") focusGroup("main"); }, true);
  }
  return api;
}
