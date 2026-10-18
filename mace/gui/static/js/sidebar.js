/* The explorer's sections (Open Editors, Files, Materials.db, Recent): their
 * order, which are shown, and the heights the splitters between them set.
 * All of it is layout.sidebar = {order: [keys], hidden: [keys], sizes: {key: px}},
 * saved with the rest of the layout (localStorage, then the profile).
 *
 * Splitters: one between each two shown sections. The Files section fills
 * what is left, so a splitter beside it sizes the other section; between two
 * others it sizes the upper one (the lower one if the upper is folded). Double-click resets that section to its
 * default share; arrow keys move a focused splitter (role=separator).
 * Reorder: drag a section's header, Alt+Up/Down on a focused header, or its
 * context menu (Move Up, Move Down, the sections to show, Reset Sidebar Layout).
 * Folding a section stays where it was (header click, profile ui.explorer). */
import { $, $$ } from "./ui.js";

export const SECTIONS = Object.freeze({
  editors: { id: "open-editors", title: "Open Editors" },
  files: { id: "files-section", title: "Files" },
  db: { id: "db-section", title: "Materials.db" },
  recent: { id: "recent-files", title: "Recent" },
});
export const DEFAULT_ORDER = Object.freeze(["editors", "files", "db", "recent"]);
export const MENU = ["view.sidebar.moveUp", "view.sidebar.moveDown", "-", "view.sidebar.show.editors", "view.sidebar.show.files",
  "view.sidebar.show.db", "view.sidebar.show.recent", "-", "view.resetSidebar"];
const STEP = 16, DRAG_START = 5;

/* o: {layout, saveLayout, bindSplitter, zoom, openAt, onChange} */
export function initSidebar(o) {
  const ex = $("#explorer"), foot = $("#side-foot"), head = $(".side-head", ex);
  const el = (k) => (SECTIONS[k] ? document.getElementById(SECTIONS[k].id) : null);
  const state = () => {
    const s = o.layout.sidebar && typeof o.layout.sidebar === "object" ? o.layout.sidebar : {};
    const known = (a) => (Array.isArray(a) ? a.filter((k) => k in SECTIONS) : []);
    // a saved order missing a section (an older layout) gets it back in its default place
    const order = [...new Set(known(s.order))];
    for (const k of DEFAULT_ORDER) if (!order.includes(k)) order.splice(Math.min(DEFAULT_ORDER.indexOf(k), order.length), 0, k);
    const sizes = {};
    for (const [k, v] of Object.entries(s.sizes && typeof s.sizes === "object" ? s.sizes : {})) if (k in SECTIONS && Number.isFinite(v) && v > 0) sizes[k] = v;
    return { order, hidden: known(s.hidden), sizes };
  };
  const write = (st) => {
    const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
    const out = {};
    if (!same(st.order, DEFAULT_ORDER)) out.order = st.order;
    if (st.hidden.length) out.hidden = st.hidden;
    if (Object.keys(st.sizes).length) out.sizes = st.sizes;
    if (Object.keys(out).length) o.layout.sidebar = out; else delete o.layout.sidebar;
    o.saveLayout();
  };
  const isShown = (k) => !state().hidden.includes(k);
  // on screen: shown by the user and with something to show (an empty Recent or Open Editors is not)
  const present = (k) => isShown(k) && el(k).childElementCount > 0;
  const collapsed = (k) => {
    const t = $(":scope > .section-toggle", el(k));
    return !!t && t.getAttribute("aria-expanded") === "false";
  };
  const presentOrder = () => state().order.filter(present);

  /* ---- splitters: a pool, reused so a focused one keeps its focus */
  const pool = [];
  function makeSplitter() {
    const s = document.createElement("div");
    s.className = "splitter h side-split";
    s.setAttribute("role", "separator");
    s.setAttribute("aria-orientation", "horizontal");
    s.tabIndex = 0;
    // the section it sizes and the drag direction, read at event time (the order can change)
    const target = () => s.dataset.target, sign = () => Number(s.dataset.sign || 1);
    const minOf = () => {
      const t = el(target()), h = t && $(":scope > .section-label", t);
      return Math.round((h ? h.offsetHeight : 24) + rowH() + 8);
    };
    const maxOf = () => Math.max(minOf(), Math.round(ex.offsetHeight * 0.8));
    s.__splitter = o.bindSplitter(s, {
      axis: "y", step: STEP, sign, min: minOf, max: maxOf, zoom: () => o.zoom(ex),
      value: () => { const t = el(target()); return t ? t.offsetHeight : 0; },
      set: (v) => {
        const k = target(), st = state();
        st.sizes[k] = v;
        applySize(k, v);
        // what the explorer gave it (the file tree keeps its minimum): store that, so the next drag starts there
        const real = el(k).offsetHeight;
        if (Math.abs(real - v) > 1) { st.sizes[k] = real; applySize(k, real); }
        o.layout.sidebar = { ...(o.layout.sidebar || {}), sizes: st.sizes };
      },
      reset: () => { const st = state(); delete st.sizes[target()]; write(st); apply(); },
      commit: () => write(state()),
    });
    pool.push(s);
    return s;
  }
  function rowH() { return parseFloat(getComputedStyle(ex).getPropertyValue("--h-row")) || 28; }
  function applySize(k, v) {
    const s = el(k);
    if (v && !collapsed(k)) { s.style.flex = `0 1 ${v}px`; s.style.maxHeight = "none"; s.classList.add("sized"); }
    else { s.style.flex = ""; s.style.maxHeight = ""; s.classList.remove("sized"); }
  }

  /* ---- the DOM: sections in order, a splitter between each two shown */
  let sig = "";
  function apply() {
    const st = state(), shown = presentOrder();
    for (const k of Object.keys(SECTIONS)) {
      el(k).hidden = st.hidden.includes(k);
      applySize(k, k === "files" ? 0 : st.sizes[k]);
    }
    head.hidden = st.hidden.includes("files");               // the filter belongs to the file list
    const s2 = JSON.stringify([st.order, shown, shown.map(collapsed)]);
    if (s2 !== sig) {
      sig = s2;
      // move only what is out of place (a moved node loses focus in some browsers)
      const want = [];
      st.order.forEach((k) => {
        if (want.length && present(k) && shown.indexOf(k) > 0) want.push("split");
        want.push(k);
      });
      let i = 0, prev = head;
      for (const w of want) {
        let node = w === "split" ? pool[i] : el(w);
        if (w === "split") { node = node || makeSplitter(); i++; }
        if (prev.nextElementSibling !== node) prev.after(node);
        prev = node;
      }
      pool.slice(i).forEach((p) => { p.hidden = true; if (p.nextElementSibling !== foot) foot.before(p); });
      if (prev.nextElementSibling !== foot) prev.after(foot);
      // each splitter: which section it sizes (Files fills, so never Files), shown only when that one is open
      for (const p of pool.slice(0, i)) {
        let a = p.previousElementSibling, b = p.nextElementSibling;
        while (a && !(a.classList.contains("side-section") && !a.hidden && a.childElementCount)) a = a.previousElementSibling;
        while (b && !(b.classList.contains("side-section") && !b.hidden && b.childElementCount)) b = b.nextElementSibling;
        const ka = a && a.dataset.section, kb = b && b.dataset.section;
        // Files fills what is left, so never Files; the upper one first; a folded one has no height to set
        const k = [ka, kb].find((x) => x && x !== "files" && !collapsed(x));
        p.hidden = !k;
        if (!k) continue;
        p.dataset.target = k; p.dataset.sign = k === ka ? "1" : "-1";
        p.setAttribute("aria-label", `Resize ${SECTIONS[k].title}`);
        p.setAttribute("aria-controls", SECTIONS[k].id);
        p.__splitter.refresh();
      }
    } else pool.forEach((p) => { if (!p.hidden && p.dataset.target) p.__splitter.refresh(); });
    // nothing left on screen (every section hidden, or the shown ones empty): say how to get them back
    const none = !Object.keys(SECTIONS).some((k) => isShown(k) && (k === "files" || el(k).childElementCount > 0));
    if (none && !emptyHint) {
      emptyHint = document.createElement("div");
      emptyHint.className = "side-empty";
      emptyHint.setAttribute("role", "note");
      emptyHint.innerHTML = `<p>Every Explorer section is hidden.</p><button type="button" class="tb-btn-text">Show All Sections</button>`;
      emptyHint.querySelector("button").addEventListener("click", () => reset());
    }
    if (emptyHint) {
      emptyHint.hidden = !none;
      if (none && emptyHint.nextElementSibling !== foot) foot.before(emptyHint);
    }
    if (o.onChange) o.onChange();
  }
  let emptyHint = null;
  // a section's rows re-rendered (opened, emptied, folded): its splitters follow
  let frame = 0;
  const later = () => { cancelAnimationFrame(frame); frame = requestAnimationFrame(apply); };
  const mo = new MutationObserver(later);
  for (const k of Object.keys(SECTIONS)) mo.observe(el(k), { childList: true });
  mo.observe($("#files-toggle"), { attributes: true, attributeFilter: ["aria-expanded"] });

  /* ---- order, shown, reset */
  function move(k, d) {
    const st = state(), vis = st.order.filter((x) => isShown(x)), i = vis.indexOf(k), j = i + d;
    if (i < 0 || j < 0 || j >= vis.length) return false;
    const other = vis[j], a = st.order.indexOf(k), b = st.order.indexOf(other);
    st.order[a] = other; st.order[b] = k;
    const f = headerFocused();
    write(st); apply(); if (f) refocus(k);
    return true;
  }
  function moveTo(k, index) {          // index among the shown sections
    const st = state(), vis = st.order.filter((x) => isShown(x) && x !== k);
    vis.splice(Math.max(0, Math.min(index, vis.length)), 0, k);
    const hiddenKeys = st.order.filter((x) => !isShown(x));
    // hidden sections keep their place relative to the one before them
    const order = [...vis];
    for (const h of hiddenKeys) { const at = st.order.indexOf(h); const before = st.order.slice(0, at).reverse().find((x) => order.includes(x)); order.splice(before ? order.indexOf(before) + 1 : 0, 0, h); }
    st.order = order;
    const f = headerFocused();
    write(st); apply(); if (f) refocus(k);
  }
  const canMove = (k, d) => { const vis = state().order.filter((x) => isShown(x)), i = vis.indexOf(k); return i >= 0 && i + d >= 0 && i + d < vis.length; };
  function setShown(k, on) {
    const st = state(), h = new Set(st.hidden);
    if (on) h.delete(k); else h.add(k);
    st.hidden = [...h];
    write(st); apply();
  }
  function reset() {
    delete o.layout.sidebar;
    o.saveLayout();
    apply();
  }
  // a header moved by keyboard keeps the focus (moving a node drops it)
  const headerFocused = () => { const a = document.activeElement; return !!(a && a.closest && a.closest("#explorer > .side-section > .section-label")); };
  function refocus(k) { const t = $(":scope > .section-label", el(k)); if (t) t.focus(); }

  /* ---- a header: its menu, Alt+Up/Down, dragging it to another place */
  const headOf = (e) => e.target.closest && e.target.closest("#explorer > .side-section > .section-label");
  const keyOf = (h) => h.parentElement.dataset.section;
  function menu(k, x, y) { o.openAt(MENU, x, y, `${SECTIONS[k].title} section menu`, { section: k }); }
  ex.addEventListener("contextmenu", (e) => {
    const h = headOf(e);
    if (!h) return;
    e.preventDefault();
    menu(keyOf(h), e.clientX, e.clientY);
  });
  ex.addEventListener("keydown", (e) => {
    const h = headOf(e);
    if (!h || !e.altKey || e.ctrlKey || e.metaKey || e.shiftKey || !["ArrowUp", "ArrowDown"].includes(e.key)) return;
    e.preventDefault(); e.stopPropagation();
    move(keyOf(h), e.key === "ArrowUp" ? -1 : 1);
  });
  let suppressClick = 0;
  addEventListener("click", (e) => {
    if (performance.now() < suppressClick && e.target.closest && e.target.closest("#explorer")) { e.preventDefault(); e.stopPropagation(); }
  }, true);
  ex.addEventListener("pointerdown", (e) => {
    const h = headOf(e);
    if (!h || e.button !== 0) return;
    const k = keyOf(h), y0 = e.clientY, x0 = e.clientX;
    let dragging = false, at = -1;
    const others = () => presentOrder().filter((x) => x !== k);
    const mark = (i) => {
      $$("#explorer > .side-section").forEach((s) => s.classList.remove("drop-before", "drop-after"));
      const os = others();
      if (i < 0) return;
      if (i < os.length) el(os[i]).classList.add("drop-before"); else if (os.length) el(os[os.length - 1]).classList.add("drop-after");
    };
    const move_ = (ev) => {
      if (!dragging && Math.hypot(ev.clientX - x0, ev.clientY - y0) < DRAG_START) return;
      if (!dragging) { dragging = true; el(k).classList.add("dragging-section"); ex.classList.add("reordering"); }
      // the place: before the first section whose middle is below the pointer (both in screen px)
      const os = others();
      at = os.findIndex((x) => { const r = el(x).getBoundingClientRect(); return ev.clientY < r.top + r.height / 2; });
      if (at < 0) at = os.length;
      mark(at);
    };
    const up = () => {
      removeEventListener("pointermove", move_, true);
      removeEventListener("pointerup", up, true);
      removeEventListener("pointercancel", up, true);
      if (!dragging) return;
      suppressClick = performance.now() + 400;
      mark(-1);
      el(k).classList.remove("dragging-section"); ex.classList.remove("reordering");
      // the index among the shown sections (a hidden or empty one keeps its place)
      const os = others(), vis = state().order.filter((x) => isShown(x) && x !== k);
      const index = at < os.length ? vis.indexOf(os[at]) : (os.length ? vis.indexOf(os[os.length - 1]) + 1 : 0);
      moveTo(k, index);
    };
    addEventListener("pointermove", move_, true);
    addEventListener("pointerup", up, true);
    addEventListener("pointercancel", up, true);
  });
  // Shift+F10 / the menu key on a focused header
  function menuForFocus() {
    const a = document.activeElement, h = a && a.closest && a.closest("#explorer > .side-section > .section-label");
    if (!h) return false;
    const r = h.getBoundingClientRect();
    menu(keyOf(h), r.left + Math.min(r.width / 2, 40), r.bottom);
    return true;
  }

  apply();
  return { apply, move, canMove, setShown, shown: isShown, reset, menuForFocus, state, title: (k) => SECTIONS[k] && SECTIONS[k].title };
}
