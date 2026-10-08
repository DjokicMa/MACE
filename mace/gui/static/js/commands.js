/* The command registry: the single source of truth for menus, the toolbar,
 * the palette, tooltips, context menus and the shortcut cheat sheet. Nothing
 * else hard-codes a label or a shortcut. Pure logic, no DOM: tests run it in
 * the page.
 *
 * Each command keeps its default keys (frozen) apart from its effective keys,
 * which a user keymap (Settings › Keyboard, the spec's §4.4 list of
 * {command, keys, remove?} overrides) may change. A chord index maps a key to
 * the commands bound to it, so a key press costs one lookup, not a walk over
 * every command. */
import { fuzzyScore } from "./rank.js";

export { fuzzyScore };
export const isMac = /Mac|iPhone|iPad/.test(globalThis.navigator ? navigator.platform : "");

/** id -> {id, title, category, icon, keys[], defaultKeys[], when, run, checked, enabled, later, aliases, reason} */
export const registry = new Map();
/** Menu mnemonics (Alt+F opens File): bound like commands, in the chord index
 * and the conflict check, but not commands (not in the palette or registry). */
export const pseudo = new Map();
export let keymapVersion = 0;

export function register(cmd) {
  if (!cmd.id || !cmd.title) throw new Error("a command needs an id and a title");
  if (registry.has(cmd.id) || pseudo.has(cmd.id)) throw new Error(`command ${cmd.id} registered twice`);
  // every command is in the palette (spec §7 Phase 2, criterion 2): none is hidden
  if (cmd.hidden) throw new Error(`command ${cmd.id} is hidden: every command must show in the palette`);
  const keys = cmd.keys || [];
  const c = { category: "", aliases: [], palette: true, ...cmd,
    defaultKeys: Object.freeze([...keys]), keys: [...keys], chords: keys.map(parseChord),
    whenFn: compileWhen(cmd.when) };
  registry.set(cmd.id, c);
  indexDirty = true;
  return c;
}
export function registerPseudo(cmd) {
  const keys = cmd.keys || [];
  const c = { category: "Menu", aliases: [], ...cmd, pseudo: true, palette: false,
    defaultKeys: Object.freeze([...keys]), keys: [...keys], chords: keys.map(parseChord), whenFn: compileWhen(cmd.when) };
  pseudo.set(cmd.id, c);
  indexDirty = true;
  return c;
}
export const command = (id) => registry.get(id) || pseudo.get(id);
export const allBindable = () => [...registry.values(), ...pseudo.values()];

/* ---------------------------------------------------------------- keys */
// Letters and digits match on event.code (layout safe); symbols on event.key,
// unless a modifier is part of the chord (then the code: Shift+, is not "<").
const NAMED = {
  esc: "Escape", escape: "Escape", del: "Delete", delete: "Delete", backspace: "Backspace",
  enter: "Enter", home: "Home", end: "End", space: "Space", tab: "Tab",
  pageup: "PageUp", pagedown: "PageDown", pgup: "PageUp", pgdn: "PageDown", insert: "Insert",
  contextmenu: "ContextMenu",
  "←": "ArrowLeft", "→": "ArrowRight", "↑": "ArrowUp", "↓": "ArrowDown",
  left: "ArrowLeft", right: "ArrowRight", up: "ArrowUp", down: "ArrowDown",
  arrowleft: "ArrowLeft", arrowright: "ArrowRight", arrowup: "ArrowUp", arrowdown: "ArrowDown",
};
const CODE_SYMBOLS = { "/": "Slash", ",": "Comma", ".": "Period", "=": "Equal", "-": "Minus",
  "[": "BracketLeft", "]": "BracketRight", ";": "Semicolon", "'": "Quote", "`": "Backquote", "\\": "Backslash" };
const SYMBOL_OF_CODE = Object.fromEntries(Object.entries(CODE_SYMBOLS).map(([k, v]) => [v, k]));

/** "Mod+Shift+Z" -> {mod, shift, alt, code|key, text}. */
export function parseChord(text) {
  text = String(text).trim();
  const parts = text.split("+").map((p) => p.trim());
  // "Mod++" or a lone "+": the key itself is "+"
  if (text.endsWith("++") || text === "+") { parts.splice(parts.length - 2, 2, "+"); }
  const c = { mod: false, shift: false, alt: false, text };
  const key = parts.pop();
  if (!key) throw new Error(`no key in ${text}`);
  for (const m of parts) {
    const k = m.toLowerCase();
    if (k === "mod" || k === "ctrl" || k === "cmd" || k === "control" || k === "meta") c.mod = true;
    else if (k === "shift") c.shift = true;
    else if (k === "alt" || k === "option") c.alt = true;
    else throw new Error(`unknown modifier ${m} in ${text}`);
  }
  const low = key.toLowerCase();
  if (/^[a-z]$/i.test(key)) c.code = "Key" + key.toUpperCase();
  else if (/^[0-9]$/.test(key)) c.code = "Digit" + key;
  else if (/^f([1-9]|1[0-2])$/i.test(key)) c.code = key.toUpperCase();
  else if (NAMED[low]) c.code = NAMED[low];
  else if ((c.mod || c.shift || c.alt) && CODE_SYMBOLS[key]) c.code = CODE_SYMBOLS[key];
  else c.key = key;  // "?", "+", "-", ",", ".": whatever the layout needs to type them
  return c;
}

/** The one spelling of a chord: "Mod+Alt+Shift+KeyE"; Ctrl/Cmd/Mod are the same. */
export function chordId(textOrChord) {
  const c = typeof textOrChord === "string" ? parseChord(textOrChord) : textOrChord;
  return `${c.mod ? "Mod+" : ""}${c.alt ? "Alt+" : ""}${c.shift ? "Shift+" : ""}${c.code || "key:" + c.key}`;
}

/** Does this keyboard event match the chord? */
export function matches(chord, e) {
  const mod = isMac ? e.metaKey : e.ctrlKey;
  if (mod !== chord.mod || e.altKey !== chord.alt) return false;
  if (chord.key) {
    // a symbol: Shift is part of typing it, so it only counts if spelled out
    if (chord.shift && !e.shiftKey) return false;
    return e.key === chord.key;
  }
  if (e.shiftKey !== chord.shift) return false;
  if (chord.code === "Space") return e.code === "Space";
  return e.code === chord.code || (chord.code.startsWith("Arrow") && e.key === chord.code)
    || (/^(Escape|Delete|Backspace|Enter|Home|End|PageUp|PageDown|Tab|Insert|ContextMenu|F\d+)$/.test(chord.code) && e.key === chord.code);
}

/** A key press as chord text ("Mod+Shift+K", "Alt+Left", "?"); null for a bare modifier. */
export function chordFromEvent(e) {
  if (["Control", "Shift", "Alt", "Meta", "AltGraph", "OS", "Dead"].includes(e.key)) return null;
  const mods = [];
  if (isMac ? e.metaKey : e.ctrlKey) mods.push("Mod");
  if (e.altKey) mods.push("Alt");
  let key;
  const code = e.code || "";
  const NAMED_CODE = { Escape: "Esc", Delete: "Delete", Backspace: "Backspace", Enter: "Enter", NumpadEnter: "Enter",
    Home: "Home", End: "End", PageUp: "PageUp", PageDown: "PageDown", Tab: "Tab", Space: "Space", Insert: "Insert",
    ContextMenu: "ContextMenu" };
  if (/^Key[A-Z]$/.test(code)) key = code.slice(3);
  else if (/^Digit\d$/.test(code)) key = code.slice(5);
  else if (/^F([1-9]|1[0-2])$/.test(code)) key = code;
  else if (code.startsWith("Arrow")) key = code.slice(5);
  else if (NAMED_CODE[code]) key = NAMED_CODE[code];
  else if (e.key === " ") key = "Space";
  else if (mods.length && SYMBOL_OF_CODE[code]) key = SYMBOL_OF_CODE[code];
  else if (e.key && e.key.length === 1) {
    // a typed symbol ("?", "+"): Shift is how it is typed, not part of the chord
    return [...mods, e.key].join("+");
  } else return null;
  if (e.shiftKey) mods.push("Shift");
  return [...mods, key].join("+");
}

/* Display names for letter and digit keys from the keyboard layout
 * (navigator.keyboard.getLayoutMap, Chromium only): display only, the match
 * stays on event.code. Loaded once. */
let layoutMap = null;
export function loadLayoutMap() {
  try {
    const kb = globalThis.navigator && navigator.keyboard;
    if (kb && kb.getLayoutMap) kb.getLayoutMap().then((m) => { layoutMap = m; keymapVersion++; }).catch(() => {});
  } catch (e) { /* not available */ }
}

/** "Mod+Shift+Z" shown as "Ctrl+Shift+Z" (or "⇧⌘Z" on a Mac). */
const SHOW = { ArrowLeft: "←", ArrowRight: "→", ArrowUp: "↑", ArrowDown: "↓", Escape: "Esc",
  Delete: "Del", PageUp: "PgUp", PageDown: "PgDn", Slash: "/", Comma: ",", Period: ".",
  Equal: "=", Minus: "-", BracketLeft: "[", BracketRight: "]", Backspace: "Backspace", Semicolon: ";",
  Quote: "'", Backquote: "`", Backslash: "\\", ContextMenu: "Menu" };
const formatted = new Map();
export function formatChord(text) {
  const hit = formatted.get(text);
  if (hit !== undefined && !layoutMap) return hit;
  const c = parseChord(text);
  let key = c.key || c.code;
  if (layoutMap && c.code && /^(Key[A-Z]|Digit\d)$/.test(c.code) && layoutMap.get(c.code)) key = layoutMap.get(c.code).toUpperCase();
  else key = key.replace(/^Key|^Digit/, "");
  key = SHOW[key] || key;
  const out = isMac ? (c.alt ? "⌥" : "") + (c.shift ? "⇧" : "") + (c.mod ? "⌘" : "") + key
    : [c.mod && "Ctrl", c.alt && "Alt", c.shift && "Shift", key].filter(Boolean).join("+");
  if (!layoutMap) formatted.set(text, out);
  return out;
}
export const shortcutOf = (id) => {
  const c = command(id);
  return c && c.keys.length ? formatChord(c.keys[0]) : "";
};
export const keysOf = (id) => { const c = command(id); return c ? [...c.keys] : []; };

/** Single keys (no Ctrl/Alt) never fire while typing in a field or in a dialog. */
export const isSingleKey = (chord) => !chord.mod && !chord.alt && !/^F\d+$/.test(chord.code || "");

/* ---------------------------------------------------------------- browser keys */
const RESERVED_LIST = [
  ["Mod+N", "opens a new browser window"], ["Mod+T", "opens a new browser tab"],
  ["Mod+W", "closes the browser tab"], ["Mod+Shift+N", "opens a new private window"],
  ["Mod+Shift+T", "reopens the last closed browser tab"], ["Mod+Shift+W", "closes the browser window"],
  ["Mod+Tab", "switches browser tabs"],
  ["Mod+L", "focuses the browser's address bar"], ["Mod+Q", "quits the browser"],
  ["Mod+Shift+P", "opens a private window in Firefox"],
  ["Mod+PageUp", "switches browser tabs"], ["Mod+PageDown", "switches browser tabs"],
  ...[1, 2, 3, 4, 5, 6, 7, 8, 9].map((n) => [`Mod+${n}`, `switches to browser tab ${n}`]),
  ["F5", "reloads the page"], ["F11", "toggles full screen"], ["F12", "opens the developer tools"],
  // measured with real keys in a visible Firefox (plan T16): Firefox acts on these before the page
  ["Alt+E", "opens Firefox's Edit menu"], ["Alt+B", "opens Firefox's Bookmarks menu"],
  ["Mod+Shift+J", "opens the browser console in Firefox"], ["Mod+Shift+M", "opens responsive design mode in Firefox"],
  ["Mod+Shift+G", "finds the previous match in Firefox"], ["Mod+Shift+A", "opens Firefox's add-ons manager"],
  ["Mod+K", "focuses Firefox's web search"],
];
/** chordId -> user-facing reason. Browsers do not let a page use these. */
export const RESERVED = new Map(RESERVED_LIST.map(([t, why]) => [chordId(t), `${formatChord(t)} ${why}. Browsers don't let a page use it.`]));
/** Developer-tools keys: some browsers take them before the page sees them. */
export const RISKY = new Map([
  ["Mod+Shift+I", "opens the developer tools in Firefox and Chrome"],
  ["Mod+Shift+C", "opens the element picker in Firefox and Chrome"],
  ["Mod+Shift+K", "opens the web console in Firefox"],
  ["Mod+Shift+E", "opens the network monitor in Firefox"],
].map(([t, why]) => [chordId(t), `${formatChord(t)} ${why}, which may take it before MACE sees it.`]));
/** Keys whose browser default leaves the page or switches tabs (Linux and
 * Windows; on a Mac these are Cmd keys a page cannot own anyway). Bound ones
 * are always kept from the browser, even while their command is disabled. */
export const BROWSER_NAV = new Map(isMac ? [] : [
  ["Alt+Left", "is the browser's Back key"], ["Alt+Right", "is the browser's Forward key"],
  ["Alt+Home", "opens the browser's home page"],
  ...[1, 2, 3, 4, 5, 6, 7, 8, 9].map((n) => [`Alt+${n}`, `switches to browser tab ${n} in Firefox and Chrome on Linux`]),
  ["Backspace", "is Back in some browsers when no field has focus"],
].map(([t, why]) => [chordId(t), `${formatChord(t)} ${why}`]));
export const isReserved = (text) => RESERVED.get(chordId(text)) || "";
export const riskOf = (text) => RISKY.get(chordId(text)) || "";
export const browserNavOf = (text) => BROWSER_NAV.get(chordId(text)) || "";

/* ---------------------------------------------------------------- when */
const whenCache = new Map();
/** "hasDoc && docKind == structure && !inputFocus" -> a function of the context. */
export function compileWhen(expr) {
  if (!expr) return null;
  if (typeof expr === "function") return expr;
  if (whenCache.has(expr)) return whenCache.get(expr);
  const alts = expr.split("||").map((alt) => alt.split("&&").map((term) => {
    term = term.trim();
    const neg = term.startsWith("!");
    if (neg) term = term.slice(1).trim();
    const m = term.match(/^(\w+)\s*(==|!=|>=)\s*([\w|]+)$/);
    if (m) {
      const [, k, op, rhs] = m;
      const set = rhs.split("|");
      const f = op === ">=" ? (c) => Number(c[k]) >= Number(rhs) : (c) => set.includes(String(c[k])) === (op === "==");
      return { term, neg, f: neg ? (c) => !f(c) : f };
    }
    return { term, neg, f: neg ? (c) => !c[term] : (c) => !!c[term] };
  }));
  const fn = (c) => alts.some((terms) => terms.every((t) => t.f(c)));
  fn.alts = alts;
  whenCache.set(expr, fn);
  return fn;
}
export function evalWhen(expr, ctx) {
  const f = compileWhen(expr);
  return f ? !!f(ctx) : true;
}

/* What a failed "when" term means to a user, when the command gives no reason. */
const TERM_REASON = { viewportFocus: "Needs the viewport focused", hasDoc: "No document is open",
  hasSelection: "Needs a selection", moveNudge: "Needs the Move tool and the viewport focused",
  setupOpen: "Open Calculation Setup first", inputFocus: "Needs a text field" };
const OWN_REASON_FIRST = new Set(["viewportFocus", "moveNudge"]);

/** Is the command usable now? Returns "" or the reason it is not. */
export function laterReason(cmd) {
  const owner = String(cmd.later);
  // "needs the Quantum ESPRESSO backend" is the reason itself; anything else names the work it arrives with
  return /^needs /i.test(owner) ? owner[0].toUpperCase() + owner.slice(1) : `Not in this version yet: ${owner}`;
}
export function disabledReason(cmd, ctx) {
  if (cmd.later) return laterReason(cmd);
  if (cmd.needs3d && ctx.noWebgl) return "Needs 3D (WebGL)";
  if (cmd.enabled && !cmd.enabled(ctx)) return (typeof cmd.reason === "function" ? cmd.reason(ctx) : cmd.reason) || "Not available here";
  const f = cmd.whenFn || compileWhen(cmd.when);
  if (f && !f(ctx)) {
    if (f.alts && f.alts.length === 1) {
      const bad = f.alts[0].find((t) => !t.f(ctx));
      if (bad && !bad.neg && TERM_REASON[bad.term] && (OWN_REASON_FIRST.has(bad.term) || !cmd.reason)) return TERM_REASON[bad.term];
    }
    return (typeof cmd.reason === "function" ? cmd.reason(ctx) : cmd.reason) || "Not available here";
  }
  return "";
}

/* ---------------------------------------------------------------- conflicts */
function termsOf(when) {
  const f = compileWhen(when);
  if (!f) return [[]];
  return f.alts.map((terms) => terms.map((t) => {
    const m = t.term.match(/^(\w+)\s*(==|!=)\s*([\w|]+)$/);
    return m ? { key: m[1], op: m[2] === "==" ? (t.neg ? "!=" : "==") : (t.neg ? "==" : "!="), set: m[3].split("|") }
      : { key: t.term, bool: !t.neg };
  }));
}
function termsClash(a, b) {
  if (a.key !== b.key) return false;
  if ("bool" in a && "bool" in b) return a.bool !== b.bool;
  if (a.set && b.set) {
    if (a.op === "==" && b.op === "==") return !a.set.some((x) => b.set.includes(x));
    if (a.op === "==" && b.op === "!=") return a.set.every((x) => b.set.includes(x));
    if (a.op === "!=" && b.op === "==") return b.set.every((x) => a.set.includes(x));
  }
  return false;
}
/** Can both contexts hold at once? Disjoint only when they contradict on one
 * term: docKind/tool sets, or a flag and its negation (viewportFocus,
 * hasSelection, moveNudge). */
export function whensOverlap(w1, w2) {
  for (const A of termsOf(w1)) for (const B of termsOf(w2)) {
    if (!A.some((a) => B.some((b) => termsClash(a, b)))) return true;
  }
  return false;
}
/** The commands (and menu mnemonics) bound to this chord whose contexts can
 * both hold with `when`; `exceptId` is left out. */
export function conflicts(chordText, when = null, exceptId = null) {
  const id = chordId(chordText);
  ensureIndex();
  return (byChord.get(id) || []).filter((c) => c.id !== exceptId && whensOverlap(when, c.when)).map((c) => c.id);
}
/** Every pair of default-bound commands sharing a chord in overlapping contexts. */
export function allConflicts() {
  ensureIndex();
  const out = [];
  for (const [id, list] of byChord) {
    for (let i = 0; i < list.length; i++) for (let j = i + 1; j < list.length; j++) {
      if (whensOverlap(list[i].when, list[j].when)) out.push([id, list[i].id, list[j].id]);
    }
  }
  return out;
}

/* ---------------------------------------------------------------- user keymap */
/** Reset every command to its default keys, then apply the overrides in order:
 * {command, keys} replaces its bindings; {command, keys, remove: true} drops
 * only those; {command, remove: true} unbinds it. Unknown ids and reserved
 * chords are skipped, with the reason. */
export function applyKeymap(overrides) {
  const report = { applied: 0, skipped: [] };
  for (const c of allBindable()) { c.keys = [...c.defaultKeys]; }
  for (const o of Array.isArray(overrides) ? overrides : []) {
    const skip = (reason) => report.skipped.push({ command: o && o.command, keys: o && o.keys, reason });
    if (!o || typeof o !== "object" || typeof o.command !== "string") { skip("not a {command, keys} entry"); continue; }
    const c = command(o.command);
    if (!c) { skip(`no command ${o.command}`); continue; }
    const keys = Array.isArray(o.keys) ? o.keys.filter((k) => typeof k === "string" && k.trim()) : [];
    const ok = [];
    let bad = false;
    for (const k of keys) {
      try { parseChord(k); } catch (e) { skip(`${k}: ${e.message}`); bad = true; continue; }
      const why = isReserved(k);
      if (why && !o.remove) { skip(why); bad = true; continue; }
      ok.push(k);
    }
    if (o.remove) {
      if (!keys.length) c.keys = [];
      else { const drop = new Set(ok.map(chordId)); c.keys = c.keys.filter((k) => !drop.has(chordId(k))); }
    } else if (ok.length || !keys.length) c.keys = ok;
    else continue;
    if (!bad || ok.length) report.applied++;
  }
  for (const c of allBindable()) c.chords = c.keys.map(parseChord);
  keymapVersion++;
  indexDirty = true;
  return report;
}
/** Does the command's binding differ from its default? */
export const isUserBound = (c) => JSON.stringify(c.keys) !== JSON.stringify(c.defaultKeys);

/* ---------------------------------------------------------------- modal layers */
/** Keys that mean something else for a moment (plan T7): while a Move grab
 * runs, X is the x axis lock and not the Zoom tool, - a minus sign and not
 * zoom out. A layer is checked before the keymap: the keys it consumes do
 * their layer action, the ones it passes reach the keymap, and every other
 * key is swallowed. Fixed, listed in the cheat sheet and the keymap editor. */
export const MODAL_LAYERS = {
  grabActive: { title: "While moving", rows: [
    ["X  Y  Z", "Lock to the Cartesian x, y or z axis"], ["Shift+X  Shift+Y  Shift+Z", "Lock to the cell axis a, b or c"],
    ["0–9  .  -", "Type the offset in Å along the locked axis"], ["Backspace", "Edit the typed offset"],
    ["Tab", "Type along the next axis"], ["G", "Start again from where the atoms were"],
    ["Enter", "Apply (also a left click)"], ["Esc", "Cancel (also a right click)"], ["Ctrl (held)", "Snap to 0.05 of a cell axis"]] },
  elementEntry: { title: "Add Atom element", rows: [["Enter", "Use this element and go back to the viewport"],
    ["Esc", "Keep the element it had"], ["E", "Edit the element (in the Add Atom tool)"]] },
};
let layer = null;
/** The active layer: {name, onKey(event) -> "consume" | "pass" | "swallow"}, or null. */
export function setLayer(l) { layer = l; }
export const activeLayer = () => layer;

/* ---------------------------------------------------------------- chord index */
let byKey = new Map(), byChord = new Map(), indexDirty = true;
function ensureIndex() {
  if (!indexDirty) return;
  byKey = new Map(); byChord = new Map();
  for (const c of allBindable()) {
    if (!c.chords || c.chords.length !== c.keys.length) c.chords = c.keys.map(parseChord);
    for (const ch of c.chords) {
      const k = ch.code || ch.key;
      if (!byKey.has(k)) byKey.set(k, []);
      byKey.get(k).push({ cmd: c, chord: ch });
      const id = chordId(ch);
      if (!byChord.has(id)) byChord.set(id, []);
      if (!byChord.get(id).includes(c)) byChord.get(id).push(c);
    }
  }
  indexDirty = false;
}
/** The (command, chord) pairs a key event matches, commands before mnemonics. */
export function lookup(e) {
  ensureIndex();
  const out = [];
  const seen = new Set();
  for (const k of [e.code, e.key]) {
    for (const hit of byKey.get(k) || []) {
      if (seen.has(hit) || !matches(hit.chord, e)) continue;
      seen.add(hit); out.push(hit);
    }
  }
  return out.sort((a, b) => (a.cmd.pseudo === true) - (b.cmd.pseudo === true));
}
