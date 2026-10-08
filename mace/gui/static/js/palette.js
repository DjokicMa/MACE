/* The command palette (F1, Mod+Alt+K). Modes by the first character:
 *   > commands (the default)       @ atoms of the active structure (Mod+G)
 *   # files in the folder (Mod+P)  % open documents (Mod+E)
 *   : go to a trajectory step      = frac <-> Cartesian calculator
 * Ranking is pure (rank.js, haystacks prepared once per keymap or document
 * change); the list is virtualized (fixed 32px rows, only the visible window
 * plus overscan in the DOM). Enter runs, Mod+Enter runs and stays open, Esc
 * closes and focus goes back where it came from. */
import { registry, shortcutOf, disabledReason, isMac } from "./commands.js";
import * as commands from "./commands.js";
import { prepare, rank, qualityOf, fracToCart, cartToFrac, parseTriple } from "./rank.js";
import { $, esc, icon, overlayOpening } from "./ui.js";
import * as profile from "./profile.js";

const ROW = 32, OVERSCAN = 8;
const PLACEHOLDER = "Type a command · # files · @ atoms · % open documents · : step · = calculator";
let dlg, input, list, spacer, hint, opener = null, rows = [], flat = [], active = 0, hooks = null;
let prepared = { key: "", items: [] }, frame = 0, benchSet = null;

export function initPalette(h) {
  // {ctx(), run(id), files(), atoms(), docs(), openFile(path), openDoc(id), selectAtom(i),
  //  structure(), traj(), goToStep(n), copy(text), setting(k)}
  hooks = h;
  dlg = document.createElement("dialog");
  dlg.className = "palette";
  dlg.id = "palette";
  dlg.setAttribute("aria-label", "Command palette");
  dlg.innerHTML = `<input type="text" role="combobox" aria-expanded="true" aria-controls="pal-list" aria-autocomplete="list"
    placeholder="${esc(PLACEHOLDER)}" aria-label="Command palette" spellcheck="false" autocomplete="off">
    <div class="pal-scroll"><ul class="pal-list" id="pal-list" role="listbox" aria-label="Results"></ul></div>
    <div class="pal-hint" aria-live="polite"></div>`;
  document.body.appendChild(dlg);
  input = $("input", dlg); list = $("ul", dlg); spacer = $(".pal-scroll", dlg); hint = $(".pal-hint", dlg);
  input.addEventListener("input", () => {
    performance.mark("ui:filter:start");
    update();
    performance.measure("ui:filter", "ui:filter:start");
  });
  input.addEventListener("keydown", onKey);
  spacer.addEventListener("scroll", () => { cancelAnimationFrame(frame); frame = requestAnimationFrame(paint); });
  list.addEventListener("pointerdown", (e) => e.preventDefault());
  list.addEventListener("click", (e) => {
    const li = e.target.closest(".pal-row");
    if (li) { active = Number(li.dataset.k); runActive(false); }
  });
  dlg.addEventListener("close", () => { if (opener && opener.isConnected && opener.focus) opener.focus(); });
  dlg.addEventListener("pointerdown", (e) => { if (e.target === dlg) dlg.close(); });
}

export function openPalette(prefix = "") {
  performance.mark("ui:palette:start");
  opener = document.activeElement;
  if (!dlg.open) { overlayOpening(); dlg.showModal(); }
  input.value = prefix;
  update();
  input.focus();
  requestAnimationFrame(() => { performance.mark("ui:palette:end"); performance.measure("ui:palette", "ui:palette:start", "ui:palette:end"); });
}
export const isPaletteOpen = () => dlg && dlg.open;
export const closePalette = () => dlg && dlg.open && dlg.close();

/* the last five commands run from here: in the profile (ui.paletteRecent), so they follow you */
function recent() { const r = profile.ui("paletteRecent", []); return Array.isArray(r) ? r : []; }
function remember(id) { profile.setUi("paletteRecent", [id, ...recent().filter((x) => x !== id)].slice(0, 5)); }

/* Commands, prepared once per registry size and keymap version. */
function commandItems() {
  const key = `${registry.size}:${commands.keymapVersion}`;
  if (prepared.key !== key) {
    const items = [...registry.values()].filter((c) => c.palette !== false).map((c) => ({
      id: c.id, label: `${c.category}: ${c.title}`, title: c.title, aliases: c.aliases || [], icon: c.icon,
      category: c.category, keys: shortcutOf(c.id), cmd: c }));
    prepared = { key, items, ranked: prepare(items), order: [...new Set(items.map((i) => i.category))] };
  }
  return prepared;
}
let docsPrepared = { key: "", ranked: [] };
function preparedFor(kind, list, keyOf) {
  const key = kind + ":" + keyOf(list);
  if (docsPrepared.key !== key) docsPrepared = { key, ranked: prepare(list) };
  return docsPrepared.ranked;
}

export const MODES = { ">": "commands", "@": "atoms", "#": "files", "%": "docs", ":": "step", "=": "calc" };

/** The rows for a query: [{id?, group, label, keys, icon, why, run}] (exported for tests).
 * Every match is returned; the list only draws the visible window. */
export function results(text, ctx) {
  const mode = MODES[text[0]] || "commands";
  const query = MODES[text[0]] ? text.slice(1).trim() : text.trim();
  if (benchSet) return rank(query, benchSet).map(({ it }) => ({ ...it, group: "Results" }));
  if (mode === "files") {
    const files = hooks.files();
    return rank(query, preparedFor("f", files.map((f) => ({ label: f.path, path: f.path })), (l) => files.length + (files[0] ? files[0].path : "")))
      .map(({ it }) => ({ group: "Files", label: it.label, icon: "file", run: () => hooks.openFile(it.path) }));
  }
  if (mode === "docs") {
    const docs = hooks.docs();
    return rank(query, prepare(docs.map((d) => ({ label: d.title, title: d.path || "", id: d.id, active: d.active }))))
      .map(({ it }) => ({ group: "Open documents", label: it.label, desc: it.title, icon: "file", why: "",
        current: it.active, run: () => hooks.openDoc(it.id) }));
  }
  if (mode === "atoms") {
    const s = hooks.structure();
    if (!s) return [{ group: "Atoms", label: "No structure is open", icon: "atom", why: "Open a structure first" }];
    const atoms = s.symbols.map((el, i) => ({ label: `${el}${i + 1}`, title: `${i + 1}`, i }));
    const wy = s.symmetry && s.symmetry.available ? s.symmetry.wyckoffs : null;
    return rank(query, preparedFor("a", atoms, () => `${s.path}:${atoms.length}`))
      .map(({ it }) => ({ group: "Atoms", label: it.label, icon: "atom", desc: `Jump to atom${wy ? " · " + wy[it.i] : ""}`,
        run: () => hooks.selectAtom(it.i) }));
  }
  if (mode === "step") {
    const t = hooks.traj();
    if (!t) return [{ group: "Trajectory", label: "Go to step", icon: "history", why: "Needs an optimization with steps" }];
    const n = Number.parseInt(query, 10);
    if (!query) return [{ group: "Trajectory", label: `Type a step, 0 to ${t.steps - 1}`, icon: "history", why: "Type a number" }];
    if (!Number.isFinite(n) || n < 0 || n >= t.steps) {
      return [{ group: "Trajectory", label: `Step ${query}`, icon: "history", why: `Steps run from 0 to ${t.steps - 1}` }];
    }
    return [{ group: "Trajectory", label: `Go to step ${n}`, icon: "history", desc: t.labels[n] || "", run: () => hooks.goToStep(n) }];
  }
  if (mode === "calc") return calculator(query);
  const { items, ranked, order } = commandItems();
  const rec = recent();
  if (!query) {
    const out = items.map((it) => {
      const r = rec.indexOf(it.id);
      return { id: it.id, group: r >= 0 ? "Recently used" : it.category, label: it.label, keys: it.keys, icon: it.icon,
        why: disabledReason(it.cmd, ctx), r };
    });
    out.sort((a, b) => (b.r >= 0) - (a.r >= 0) || (a.r >= 0 ? a.r - b.r : order.indexOf(a.group) - order.indexOf(b.group)));
    return out;
  }
  // with a query (plan T9b): match quality first, then availability within a
  // tier (enabled, disabled here, not built yet), then the score; a weak
  // enabled match never jumps ahead of a strong disabled one
  const q = query.trim().toLowerCase();
  return rank(query, ranked).map(({ it, score, p }, n) => {
    const why = disabledReason(it.cmd, ctx);
    return { id: it.id, group: "Commands", label: it.label, keys: it.keys, icon: it.icon, why,
      tier: qualityOf(q, p), cls: !why ? 0 : it.cmd.later ? 2 : 1, score, n };
  }).sort((a, b) => a.tier - b.tier || a.cls - b.cls || a.n - b.n);
}

/* "= 0.5 0 0.5" -> Cartesian Å in the active cell; "= c 1.2 0 3.4" -> fractional. */
function calculator(q) {
  const s = hooks.structure();
  const group = "Calculator";
  if (!s || !s.cell) return [{ group, label: "Fractional ↔ Cartesian", icon: "calculator", why: "Needs a structure with a cell" }];
  const cart = /^c\b/i.test(q);
  const v = parseTriple(cart ? q.replace(/^c\s*/i, "") : q);
  if (!v) return [{ group, label: cart ? "Type x y z in Å" : "Type fractional x y z (c x y z for Cartesian)", icon: "calculator", why: "Three numbers" }];
  const d = Number(hooks.setting("decimals.length")) || 4;
  const out = cart ? cartToFrac(v, s.cell) : fracToCart(v, s.cell);
  if (!out) return [{ group, label: "The cell is singular", icon: "calculator", why: "No inverse" }];
  const fmt = (x) => { const t = x.toFixed(cart ? Math.max(d, 5) : d); return /^-0\.?0*$/.test(t) ? t.slice(1) : t; };
  const text = out.map(fmt).join(" ") + (cart ? " (frac)" : " Å");
  return [{ group, label: text, icon: "calculator", desc: "Enter copies it", run: () => hooks.copy(text.replace(/ \(frac\)| Å/, "")) }];
}

/* ---------------------------------------------------------------- the virtual list */
function update() {
  rows = results(input.value, hooks.ctx());
  flat = [];
  let group = null;
  rows.forEach((r, k) => {
    if (r.group !== group) { group = r.group; flat.push({ head: group }); }
    flat.push({ k });
  });
  // the first row, whatever its state: Enter on a disabled one explains, it never runs a weaker match
  active = 0;
  list.style.height = `${flat.length * ROW}px`;
  spacer.scrollTop = 0;
  const mode = MODES[input.value[0]];
  hint.textContent = mode ? ({ commands: "Commands", atoms: "Atoms: Enter centers the view on it",
    files: "Files in this folder", docs: "Open documents", step: "Trajectory step", calc: "Calculator: Enter copies the result" })[mode] : "";
  hint.hidden = !mode;
  paint();
}

function paint() {
  frame = 0;
  if (!rows.length) {
    list.style.height = "";
    list.innerHTML = `<li class="pal-empty" role="presentation">No matches</li>`;
    input.removeAttribute("aria-activedescendant");
    return;
  }
  const top = spacer.scrollTop, h = spacer.clientHeight || 400;
  const first = Math.max(0, Math.floor(top / ROW) - OVERSCAN);
  const last = Math.min(flat.length, Math.ceil((top + h) / ROW) + OVERSCAN);
  const show = new Set();
  for (let j = first; j < last; j++) show.add(j);
  const activeAt = flat.findIndex((f) => f.k === active);
  if (activeAt >= 0) show.add(activeAt);   // aria-activedescendant must point at a node
  const ids = hooks.setting("devShowCommandIds");
  let html = "";
  for (const j of [...show].sort((a, b) => a - b)) {
    const f = flat[j];
    if (f.head !== undefined) {
      html += `<li class="pal-group" role="presentation" style="top:${j * ROW}px">${esc(f.head)}</li>`;
      continue;
    }
    const r = rows[f.k];
    html += `<li class="pal-row${r.why ? " disabled" : ""}" role="option" id="pal-${f.k}" data-k="${f.k}"${r.id ? ` data-id="${esc(r.id)}"` : ""}` +
      ` style="top:${j * ROW}px" aria-selected="${f.k === active}"${r.why ? ' aria-disabled="true"' : ""}>${icon(r.icon)}` +
      `<span class="pal-text"><span class="t">${esc(r.label)}</span>${r.why ? `<span class="why">${esc(r.why)}</span>` : r.desc ? `<span class="why">${esc(r.desc)}</span>` : ""}</span>` +
      `${ids && r.id ? `<span class="pal-id">${esc(r.id)}</span>` : ""}${r.keys ? `<kbd>${esc(r.keys)}</kbd>` : "<span></span>"}</li>`;
  }
  list.innerHTML = html;
  input.setAttribute("aria-activedescendant", `pal-${active}`);
}

function move(d) {
  if (!rows.length) return;
  active = (active + d + rows.length) % rows.length;
  const j = flat.findIndex((f) => f.k === active);
  const top = spacer.scrollTop, h = spacer.clientHeight;
  // keep the active row (and its group header, at the top) in view
  if (j * ROW < top + ROW) spacer.scrollTop = Math.max(0, (j - 1) * ROW);
  else if ((j + 1) * ROW > top + h) spacer.scrollTop = (j + 1) * ROW - h;
  paint();
}

function runActive(keepOpen) {
  const r = rows[active];
  if (r && r.why && (r.id || r.run)) {
    hint.hidden = false;
    hint.textContent = `${r.label.replace(/^[^:]+: /, "")}: ${r.why}`;
    if (hooks.toast) hooks.toast(`${r.label.replace(/^[^:]+: /, "")}: ${r.why}`, "info");
    return;
  }
  if (!r || r.why || (!r.id && !r.run)) return;
  if (!keepOpen) dlg.close();
  if (r.id) { remember(r.id); hooks.run(r.id); } else r.run();
  if (keepOpen && dlg.open) { update(); input.focus(); }
}

function onKey(e) {
  const mod = isMac ? e.metaKey : e.ctrlKey;
  if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
  else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
  else if (e.key === "PageDown") { e.preventDefault(); move(8); }
  else if (e.key === "PageUp") { e.preventDefault(); move(-8); }
  else if (e.key === "Enter") { e.preventDefault(); runActive(mod); }
  else if (e.key === "Escape") { e.preventDefault(); dlg.close(); }
  e.stopPropagation();
}

/* ---------------------------------------------------------------- bench (spec §8.1) */
/** Per-keystroke filter time (rank + DOM) on a synthetic stress set:
 * n commands and `atoms` atom labels, built in memory, never registered. */
export function bench(n = 500, atoms = 2000, queries = ["s", "su", "sup", "supe", "super", "O", "O1", "O12", "gamma"]) {
  const words = ["Supercell", "Select", "Show", "Symmetry", "View", "Rotate", "Gamma", "Edit", "Cell", "Bond", "Label", "Plot"];
  const items = [];
  for (let i = 0; i < n; i++) {
    const t = `${words[i % words.length]} ${words[(i * 7 + 3) % words.length]} ${i}`;
    items.push({ label: `Bench: ${t}`, title: t, aliases: i % 9 === 0 ? ["gamma angle"] : [], icon: "command", keys: i % 5 ? "" : "Ctrl+K" });
  }
  const els = ["O", "Mg", "Si", "Al", "Ca", "Fe"];
  for (let i = 0; i < atoms; i++) items.push({ label: `${els[i % els.length]}${i + 1}`, icon: "atom" });
  const wasOpen = dlg.open;
  if (!wasOpen) openPalette("");
  benchSet = prepare(items);
  const times = [];
  try {
    for (const q of queries) {
      const t0 = performance.now();
      input.value = q;
      update();
      // the layout the browser does for the new rows belongs to the keystroke too
      void list.offsetHeight;
      times.push({ q, ms: performance.now() - t0, rows: rows.length, dom: list.children.length });
    }
  } finally {
    benchSet = null;
    input.value = "";
    if (!wasOpen) dlg.close(); else update();
  }
  return times;
}
export const domRows = () => list ? list.querySelectorAll(".pal-row").length : 0;
