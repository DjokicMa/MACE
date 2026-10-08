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
 *   Group (Mod+Shift+\), Focus Other Group (Mod+Alt+\), Close Group, Link
 *   Cameras; Open to the Side (Explorer); Move to Other Group (tab menu); a
 *   tab dragged onto the other group (or onto the right half of a single
 *   group) moves there.
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
const GROUP_CMDS = /^view\.(splitRight|moveToOtherGroup|focusOtherGroup|closeGroup|linkCameras)$/;
export function isDocCommand(id, category) {
  if (GROUP_CMDS.test(id)) return false;
  if (id === "tab.moveToOtherGroup") return true;
  if (DOC_VIEW.test(id) || DOC_FILE.test(id)) return true;
  return DOC_CATEGORIES.has(category || "");
}

const MOVABLE = (d) => !!(d && d.path);

/** host: the shell's group API (window.__macePane of this page) plus
 *  toast, profile, dirHash(), afterChange(). */
export function createGroups(host) {
  const g = { open: false, ratio: 0.5, linked: false, focused: "main", frame: null, ready: null, restoring: false };
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
      moveToOther: (d) => moveFromHere(d, other(), () => parentGroups().focusGroup("main")),
      dragTab: () => {},
      restore: () => {},
      state: () => ({ pane: true }),
    };
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

  async function moveFromHere(d, to, after) {
    if (!d) return false;
    if (!MOVABLE(d)) { host.toast(`${d.title}: this document has no file, so it cannot move to the other group`, "info"); return false; }
    if (host.isDirty(d)) { host.toast(`${d.title}: save it first (moving reopens the file in the other group)`, "info"); return false; }
    if (!to) return false;
    const desc = host.desc(d);
    host.closeDoc(d, true);
    await to.reopen(desc);
    if (after) after();
    return true;
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
    css();
    if (which === "side" && g.frame) {
      try { g.frame.contentWindow.focus(); const o = other(); if (o) o.focusView(); } catch (e) { /* loading */ }
    } else if (which === "main" && changed) {
      host.focusView();
    }
    if (changed) host.afterChange();
  }

  /* the right group's active document comes back left */
  async function moveFromSide() {
    const o = other();
    const a = o && o.active();
    if (!a) { host.toast("The right group has no document", "info"); return false; }
    if (!a.path) { host.toast(`${a.title}: this document has no file, so it cannot move to the other group`, "info"); return false; }
    if (a.dirty) { host.toast(`${a.title}: save it first (moving reopens the file in the other group)`, "info"); return false; }
    o.close(a.id, true);
    await host.reopen(a);
    focusGroup("main");
    persist();
    return true;
  }

  /* the active document of the left group goes right (or comes back) */
  async function moveToOther(d) {
    if (!d && g.open && g.focused === "side") return moveFromSide();
    d = d || host.activeDoc();
    if (!d) { host.toast("Open a document first", "info"); return false; }
    if (!MOVABLE(d)) { host.toast(`${d.title}: this document has no file, so it cannot move to the other group`, "info"); return false; }
    if (host.isDirty(d)) { host.toast(`${d.title}: save it first (moving reopens the file in the other group)`, "info"); return false; }
    const api = await openSplit({ focus: "side" });
    const desc = host.desc(d);
    host.closeDoc(d, true);
    await api.reopen(desc);
    focusGroup("side");
    persist();
    return true;
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
    const r = await api.reopen({ path, kind: "", as: opts.as || "" });
    focusGroup("side");
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

  /* ------------------------------------------------------------ dragging a tab to the other group */
  let drop = null;
  function dropZone() {
    if (!drop) {
      drop = document.createElement("div");
      drop.className = "group-drop";
      drop.innerHTML = "<span></span>";
      drop.hidden = true;
      document.body.appendChild(drop);
    }
    return drop;
  }
  function zoneRect() {
    if (g.open) return side().getBoundingClientRect();
    const v = document.getElementById("views").getBoundingClientRect();
    return { left: v.left + v.width / 2, top: v.top, width: v.width / 2, height: v.height, right: v.right, bottom: v.bottom };
  }
  /** Called from the tab strip's pointerdown on a tab of the left group. */
  function dragTab(e, d, onDropped) {
    if (e.button !== 0 || !d) return;
    const x0 = e.clientX, y0 = e.clientY;
    let dragging = false, over = false;
    const z = dropZone();
    if (g.frame) g.frame.style.pointerEvents = "none";
    const move = (ev) => {
      if (!dragging && Math.hypot(ev.clientX - x0, ev.clientY - y0) < 8) return;
      if (!dragging) {
        dragging = true;
        const r = zoneRect();
        Object.assign(z.style, { left: `${r.left}px`, top: `${r.top}px`, width: `${r.width}px`, height: `${r.height}px` });
        z.querySelector("span").textContent = g.open ? `Move ${d.title} to the right group` : `Split right with ${d.title}`;
        z.hidden = false;
        document.body.classList.add("tab-dragging");
      }
      const r = zoneRect();
      over = ev.clientX >= r.left && ev.clientX <= r.left + r.width && ev.clientY >= r.top && ev.clientY <= r.top + r.height;
      z.classList.toggle("over", over);
    };
    const up = () => {
      removeEventListener("pointermove", move, true); removeEventListener("pointerup", up, true);
      if (g.frame) g.frame.style.pointerEvents = "";
      z.hidden = true; z.classList.remove("over");
      document.body.classList.remove("tab-dragging");
      if (dragging) {
        if (onDropped) onDropped();
        if (over) moveToOther(d);
      }
    };
    addEventListener("pointermove", move, true);
    addEventListener("pointerup", up, true);
  }

  /* ------------------------------------------------------------ restore per folder */
  async function restore() {
    const st = host.profile ? host.profile.ui(`split.${host.dirHash()}`) : null;
    if (!st || typeof st !== "object") return;
    g.ratio = typeof st.ratio === "number" ? Math.max(0.2, Math.min(0.8, st.ratio)) : 0.5;
    g.linked = !!st.linked;
    if (!st.open) { css(); return; }
    g.restoring = true;
    try {
      const api = await openSplit({ focus: "main" });
      for (const x of Array.isArray(st.docs) ? st.docs : []) {
        if (x && typeof x.path === "string" && host.exists(x.path)) await api.reopen(x).catch(() => null);
      }
      if (st.active && host.exists(st.active)) await api.reopen({ path: st.active }).catch(() => null);
    } catch (e) { /* the folder changed: start with one group */ }
    finally { g.restoring = false; }
    focusGroup("main");
    persist();
  }

  const api = {
    pane: false, isOpen: () => g.open, sideFocused: () => g.open && g.focused === "side", linked: () => g.linked, other,
    openSplit, closeGroup, splitRight, moveToOther, openToSide, toggleLink, focusGroup, restore, dragTab,
    focusOther: () => (g.open ? focusGroup(g.focused === "main" ? "side" : "main") : host.toast("There is only one editor group (Split Editor Right: Mod+\\)", "info")),
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
    sideChanged: () => { persist(); host.afterChange(); },
    state: () => ({ open: g.open, ratio: g.ratio, linked: g.linked, focused: g.focused }),
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
