/* Settings › Keyboard (spec §4.4): every command and menu mnemonic with its
 * keys. Click a binding (or Enter on it) and press keys to change it; + adds
 * one. Reserved browser keys are refused with the reason, developer-tools
 * and page-leaving keys warn, and a key another command uses asks first.
 * Saved as the list of {command, keys, remove?} overrides, never the whole map. */
const ROW = 32, OVERSCAN = 8;
let H, C, host, list, scroller, rec, filterEl, showEl, rows = [], frame = 0;
let recording = null;   // {id, add}
let subscribed = false, canceling = 0;

export function mount(el, handle) {
  H = handle; C = handle.commands; host = el;
  host.innerHTML = `<div class="km-tools">
      <input type="search" class="field km-filter" placeholder="Filter by command or key" aria-label="Filter keybindings" autocomplete="off">
      <label class="km-show">Show <select class="field" aria-label="Show">
        <option value="all">All</option><option value="user">User only</option><option value="unbound">Unbound</option>
        <option value="mnemonics">Menu mnemonics</option><option value="modal">Fixed keys (while moving, Add Atom element)</option></select></label>
      <span class="grow"></span>
      <button class="tb-btn-text" data-km="export">${H.icon("download")}<span>Export keymap.json</span></button>
      <button class="tb-btn-text" data-km="import">${H.icon("upload")}<span>Import keymap.json</span></button>
      <button class="tb-btn-text" data-km="reset-all">${H.icon("rotate-ccw")}<span>Reset all keybindings</span></button>
      <input type="file" accept="application/json,.json" class="km-file" hidden></div>
    <div class="km-rec" role="group" aria-label="Record a keybinding" tabindex="-1" hidden></div>
    <div class="km-head" role="row"><span>Command</span><span>Keybinding</span><span>When</span><span>Source</span><span></span></div>
    <div class="km-scroll"><div class="km-list" role="rowgroup"></div></div>
    <p class="set-help">Single-key shortcuts never fire while you type in a field. Ctrl+W, Ctrl+T, Ctrl+1..9 and the like belong to the browser.</p>`;
  list = host.querySelector(".km-list"); scroller = host.querySelector(".km-scroll");
  rec = host.querySelector(".km-rec"); filterEl = host.querySelector(".km-filter"); showEl = host.querySelector(".km-show select");
  filterEl.addEventListener("input", update);
  showEl.addEventListener("change", update);
  scroller.addEventListener("scroll", () => { cancelAnimationFrame(frame); frame = requestAnimationFrame(paint); });
  list.addEventListener("click", onListClick);
  list.addEventListener("keydown", (e) => { if ((e.key === "Enter" || e.key === " ") && e.target.matches(".km-keys")) { e.preventDefault(); e.target.click(); } });
  host.querySelector(".km-tools").addEventListener("click", onTools);
  rec.addEventListener("click", onRecClick);
  rec.addEventListener("keydown", (e) => { if (e.target.matches(".km-text") && e.key === "Enter") { e.preventDefault(); e.stopPropagation(); tryChord(e.target.value.trim(), true); } });
  const dlg = host.closest("dialog");
  if (!dlg.__keymapBound) {
    dlg.__keymapBound = true;
    // capture phase: the recorder sees the keys before the dialog, the page or the browser
    dlg.addEventListener("keydown", onRecordKey, true);
    // the Esc that cancels a recording must not also close Settings
    dlg.addEventListener("cancel", (e) => { if (recording || performance.now() - canceling < 300) e.preventDefault(); });
  }
  if (!subscribed) { subscribed = true; H.settings.onChange((k) => { if (k === "keymap" && host && host.isConnected) update(); }); }
  recording = null;
  update();
  return { record: (id) => startRecording(id, false), rows: () => rows.map((r) => r.id) };
}

/* ---------------------------------------------------------------- rows */
const isPseudo = (c) => c.pseudo === true;
function update() {
  const q = filterEl.value.trim().toLowerCase(), show = showEl.value;
  if (show === "modal") {
    // the modal layers' keys (plan T7): fixed, shown here so every key MACE uses is in one place
    rows = Object.values(C.MODAL_LAYERS).flatMap((l) => l.rows.map(([k, t], n) => ({ id: `${l.title}#${n}`, fixed: true,
      label: `${l.title}: ${t}`, keys: [k], c: { keys: [], when: "" } })))
      .filter((r) => !q || `${r.label} ${r.keys.join(" ")}`.toLowerCase().includes(q));
    list.style.height = `${rows.length * ROW}px`;
    paint();
    return;
  }
  const all = show === "mnemonics" ? [...C.pseudo.values()] : [...C.registry.values()];
  rows = all.map((c) => ({ id: c.id, c, label: isPseudo(c) ? c.title : `${c.category}: ${c.title}`,
    keys: c.keys.map(C.formatChord) })).filter((r) => {
    if (show === "user" && !C.isUserBound(r.c)) return false;
    if (show === "unbound" && r.c.keys.length) return false;
    return !q || `${r.label} ${r.id} ${r.keys.join(" ")}`.toLowerCase().includes(q);
  });
  list.style.height = `${rows.length * ROW}px`;
  paint();
}
function paint() {
  frame = 0;
  const top = scroller.scrollTop, h = scroller.clientHeight || 360;
  const first = Math.max(0, Math.floor(top / ROW) - OVERSCAN), last = Math.min(rows.length, Math.ceil((top + h) / ROW) + OVERSCAN);
  let html = "";
  for (let k = first; k < last; k++) {
    const r = rows[k];
    if (r.fixed) {
      html += `<div class="km-row fixed" role="row" style="top:${k * ROW}px"><span class="km-cmd">${H.esc(r.label)}</span>` +
        `<span class="km-keyscell"><kbd>${H.esc(r.keys[0])}</kbd></span><span class="km-when">fixed</span><span class="km-src">Fixed</span></div>`;
      continue;
    }
    const user = C.isUserBound(r.c), on = recording && recording.id === r.id;
    html += `<div class="km-row${on ? " recording" : ""}" role="row" data-id="${H.esc(r.id)}" style="top:${k * ROW}px">` +
      `<span class="km-cmd" title="${H.esc(r.label)} (${H.esc(r.id)})">${H.esc(r.label)}</span>` +
      `<span class="km-keyscell"><button class="km-keys" data-id="${H.esc(r.id)}" title="${H.esc(r.keys.join(", ") || "Unbound")}" aria-label="Change the keybinding of ${H.esc(r.label)}">` +
      (r.keys.length ? r.keys.map((k2) => `<kbd>${H.esc(k2)}</kbd>`).join("") : `<span class="muted">Unbound</span>`) + `</button>` +
      `<button class="km-add icon-btn sm" data-add="${H.esc(r.id)}" aria-label="Add a keybinding to ${H.esc(r.label)}">${H.icon("plus")}</button></span>` +
      `<span class="km-when" title="${H.esc(readableWhen(r.c.when))}">${H.esc(readableWhen(r.c.when))}</span><span class="km-src">${user ? "User" : "Default"}</span>` +
      `<button class="set-reset km-reset" data-reset-key="${H.esc(r.id)}"${user ? "" : " hidden"}>Reset</button></div>`;
  }
  list.innerHTML = html;
}
const WHEN_WORDS = { viewportFocus: "viewport focused", hasDoc: "a document is open", hasSelection: "atoms selected",
  inputFocus: "typing in a field", moveNudge: "Move tool, viewport focused", setupOpen: "Calculation Setup open",
  toolNotSelect: "a tool other than Select", controlFocus: "a button focused", canUndo: "something to undo",
  canRedo: "something to redo", hasDb: "a database in the folder", hasBand: "a band structure in the folder",
  hasDos: "a DOS in the folder" };
export function readableWhen(w) {
  if (!w || typeof w !== "string") return "";
  return w.split("||").map((alt) => alt.split("&&").map((t) => {
    t = t.trim();
    const neg = t.startsWith("!"), core = neg ? t.slice(1).trim() : t;
    const m = core.match(/^(\w+)\s*(==|!=|>=)\s*([\w|]+)$/);
    if (m) {
      if (m[1] === "docKind") return `${m[2] === "!=" ? "not " : ""}${m[3].split("|").join(" or ")} open`;
      if (m[1] === "docCount") return `${m[3]}+ documents open`;
      if (m[1] === "trajSteps") return "a run with steps open";
      return `${m[1]} ${m[2] === "==" ? "is" : m[2] === "!=" ? "is not" : "≥"} ${m[3].split("|").join(" or ")}`;
    }
    const word = WHEN_WORDS[core] || core.replace(/([A-Z])/g, " $1").toLowerCase();
    return neg ? `not ${word}` : word;
  }).join(", ")).join("; or ");
}

/* ---------------------------------------------------------------- recording */
function startRecording(id, add) {
  recording = { id, add };
  const c = C.command(id);
  rec.hidden = false;
  rec.innerHTML = `<div class="km-rec-line"><span class="km-rec-title">${add ? "Add a key to" : "New key for"} <b>${H.esc(c.title)}</b>:</span>
    <span class="km-rec-hint">Press keys… Esc cancels, Backspace removes</span></div>
    <div class="km-rec-line"><button class="tb-btn-text" data-rec="cancel">Cancel</button>
    <button class="tb-btn-text" data-rec="remove">Remove binding</button>
    <button class="linklike" data-rec="text">Enter as text…</button><input class="field km-text" hidden placeholder="e.g. Esc, Backspace, Mod+Shift+K" aria-label="Key as text" spellcheck="false"></div>
    <div class="km-msg" role="alert"></div>`;
  paint();
  rec.focus();
}
/* Keep recording: clear the question and listen for the next key. */
function pickAnother() {
  const m = rec.querySelector(".km-msg");
  if (m) { m.className = "km-msg"; m.innerHTML = ""; }
  rec.focus();
}
function stopRecording() {
  recording = null;
  rec.hidden = true; rec.innerHTML = "";
  paint();
}
function onRecordKey(e) {
  // Tab and Shift+Tab move focus on (the recorder is never a keyboard trap); Ctrl+Tab is checked below
  if (!recording || e.target.matches(".km-text") || (e.key === "Tab" && !e.ctrlKey && !e.metaKey && !e.altKey)) return;
  if (!rec.contains(e.target) && e.target !== rec) return;
  // a bare Enter or Space on one of the recorder's own buttons presses it (Cancel,
  // Replace, Use anyway…); to bind Enter or Space, record with the panel focused or type it
  if (e.target !== rec && e.target.closest("button, a[href]") && !e.ctrlKey && !e.metaKey && !e.altKey
      && (e.key === "Enter" || e.key === " ")) return;
  const text = C.chordFromEvent(e);
  if (!text) return;
  if (recording.reset && text !== "Esc") return;          // a Reset's question: its buttons answer it
  e.preventDefault(); e.stopPropagation();
  if (text === "Esc") { canceling = performance.now(); stopRecording(); return; }   // bare Esc cancels
  if (text === "Backspace") { setKeys(recording.id, []); stopRecording(); H.toast("Keybinding removed", "info"); return; }
  tryChord(text, false);
}
/** Check a chord (recorded, or typed as text) and apply it, or say why not. */
function tryChord(text, typed, force = false) {
  if (!recording) return;
  let chord;
  try { chord = C.parseChord(text); } catch (err) { say(`“${text}” is not a key: ${err.message}`); return; }
  const shown = C.formatChord(chord.text);
  const reserved = C.isReserved(chord.text);
  if (reserved) { say(`${shown} can't be used: ${reserved.replace(shown + " ", "it ")}`, "error"); return; }
  const id = recording.id, c = C.command(id);
  if (!force) {
    const risk = C.riskOf(chord.text) || C.browserNavOf(chord.text);
    if (risk) {
      const nav = C.browserNavOf(chord.text);
      say(nav ? `${risk}. MACE keeps it from the browser while it's bound here, so the browser's own use of it won't work in this tab.` : risk,
        "warning", [["Use anyway", () => tryChord(text, typed, true)]]);
      return;
    }
  }
  const clash = C.conflicts(chord.text, c.when, id);
  if (clash.length) {
    const names = clash.map((o) => { const x = C.command(o); return isPseudo(x) ? `opens the ${x.title.replace(/^Open the | Menu$/g, "")} menu` : x.title; });
    const what = isPseudo(C.command(clash[0])) && clash.length === 1 ? `${shown} ${names[0]}`
      : `${shown} is also ${names.length > 1 ? names.slice(0, -1).join(", ") + " and " + names[names.length - 1] : names[0]}`;
    say(`${what}. Replace ${clash.length > 1 ? "them" : "it"}?`, "warning", [["Replace", () => { clash.forEach((o) => dropKey(o, chord.text)); bind(id, chord.text); }], ["Pick another key", pickAnother]]);
    return;
  }
  bind(id, chord.text);
}
function bind(id, text) {
  const c = C.command(id);
  const keys = recording && recording.add ? [...c.keys.filter((k) => C.chordId(k) !== C.chordId(text)), text] : [text];
  setKeys(id, keys);
  stopRecording();
  H.toast(`${c.title}: ${C.formatChord(text)}`, "success");
}
function say(msg, kind = "info", actions = []) {
  const m = rec.querySelector(".km-msg");
  m.className = `km-msg ${kind}`;
  m.innerHTML = `<span></span>`;
  m.firstChild.textContent = msg;
  actions.forEach(([label, fn], i) => {
    const b = document.createElement("button");
    b.className = i ? "tb-btn-text" : "tb-btn-text primary";
    b.textContent = label;
    b.addEventListener("click", fn);
    m.appendChild(b);
  });
  if (actions.length) m.querySelector("button").focus();
}
function onRecClick(e) {
  const b = e.target.closest("[data-rec]");
  if (!b) return;
  if (b.dataset.rec === "cancel") stopRecording();
  else if (b.dataset.rec === "remove") { setKeys(recording.id, []); stopRecording(); }
  else if (b.dataset.rec === "text") { const t = rec.querySelector(".km-text"); t.hidden = false; t.focus(); }
}
function onListClick(e) {
  const k = e.target.closest(".km-keys"), a = e.target.closest("[data-add]"), r = e.target.closest("[data-reset-key]");
  if (k) startRecording(k.dataset.id, false);
  else if (a) startRecording(a.dataset.add, true);
  else if (r) resetRow(r.dataset.resetKey);
}
/* A row's Reset: its default keys back, after the same conflict check as a
 * recorded key (another command may hold one of them now): Replace or Cancel. */
function resetRow(id) {
  const c = C.command(id), def = [...c.defaultKeys];
  const taken = new Map();                   // other command -> its keys that are this one's defaults
  for (const k of def) for (const o of C.conflicts(k, c.when, id)) taken.set(o, [...(taken.get(o) || []), k]);
  if (!taken.size) { setKeys(id, def); return; }
  recording = { id, add: false, reset: true };
  rec.hidden = false;
  rec.innerHTML = `<div class="km-rec-line"><span class="km-rec-title">Reset <b>${H.esc(c.title)}</b> to its default keys</span></div>
    <div class="km-msg" role="alert"></div>`;
  paint();
  const parts = [...taken].map(([o, keys]) => {
    const x = C.command(o);
    const who = isPseudo(x) ? `the ${x.title.replace(/^Open the | Menu$/g, "")} menu` : x.title;
    return `${keys.map((k) => C.formatChord(k)).join(" and ")} ${keys.length > 1 ? "are" : "is"} ${who}'s now`;
  });
  say(`${parts.join("; ")}. Give ${taken.size > 1 || [...taken.values()][0].length > 1 ? "them" : "it"} back to ${c.title}?`, "warning", [
    ["Replace", () => {
      for (const [o, keys] of taken) {
        const ids = new Set(keys.map(C.chordId));
        pending.set(o, C.command(o).keys.filter((k) => !ids.has(C.chordId(k))));
      }
      setKeys(id, def);
      stopRecording();
      H.toast(`${c.title}: back to its default keys`, "success");
    }],
    ["Cancel", stopRecording]]);
}

/* ---------------------------------------------------------------- saving */
/* The overrides that turn the defaults into the effective keys: a pure
 * removal is {command, keys: removed, remove: true}; anything else lists the keys. */
function overridesFrom(eff) {
  const out = [];
  for (const c of [...C.registry.values(), ...C.pseudo.values()]) {
    const keys = eff.has(c.id) ? eff.get(c.id) : c.keys, def = [...c.defaultKeys];
    if (JSON.stringify(keys) === JSON.stringify(def)) continue;
    const ids = new Set(keys.map(C.chordId));
    if (keys.every((k) => def.some((d) => C.chordId(d) === C.chordId(k)))) {
      const removed = def.filter((d) => !ids.has(C.chordId(d)));
      out.push(keys.length ? { command: c.id, keys: removed, remove: true } : { command: c.id, remove: true });
    } else out.push({ command: c.id, keys: [...keys] });
  }
  return out;
}
const pending = new Map();
function setKeys(id, keys) {
  pending.set(id, keys);
  const list2 = overridesFrom(pending);
  pending.clear();
  H.settings.set("keymap", list2);
}
function dropKey(id, text) {
  const c = C.command(id);
  pending.set(id, c.keys.filter((k) => C.chordId(k) !== C.chordId(text)));
}

function onTools(e) {
  const b = e.target.closest("[data-km]");
  if (!b) return;
  if (b.dataset.km === "reset-all") { H.settings.set("keymap", []); H.toast("Every keybinding is back to its default", "success"); }
  else if (b.dataset.km === "export") {
    const blob = new Blob([JSON.stringify(H.settings.get("keymap"), null, 2) + "\n"], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "keymap.json";
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  } else if (b.dataset.km === "import") {
    const f = host.querySelector(".km-file");
    f.onchange = () => importKeymap(f.files[0]);
    f.click();
  }
}
export async function importKeymap(file) {
  try {
    const data = JSON.parse(await file.text());
    const list2 = Array.isArray(data) ? data : data && Array.isArray(data.keymap) ? data.keymap : null;
    if (!list2) throw new Error("expected a list of {command, keys, remove?}");
    const report = C.applyKeymap(list2);
    H.settings.set("keymap", list2);
    H.toast(report.skipped.length ? `Imported keymap; skipped ${report.skipped.map((s) => `${s.command}: ${s.reason}`).join("; ")}` : "Imported keymap",
      report.skipped.length ? "warning" : "success");
  } catch (err) { H.toast(`Not a keymap file: ${err.message}`, "error"); }
}
