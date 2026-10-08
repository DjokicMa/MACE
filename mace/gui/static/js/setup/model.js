/* Calculation Setup's model (plan B1, B4): no DOM, no serializer.
 *
 * The settings JSON is the CLI's own file, made on the server and kept here
 * as the text the server sent (`jsonText`); this module never parses it to
 * write it back. A form edit becomes a typed answer {field: value} (the
 * CLI's question answered), sent to /api/setup/run; the server answers with
 * the new text. Each document has its own state and its own settings history
 * (separate from the structure's undo history). */

export const MODES = { cif2d12: "mace convert", opt2d12: "mace opt2d12", opt2d3: "mace opt2d3" };
export const MAX_HISTORY = 100;
export const MERGE_MS = 1000;

/** Which CLI sets up a document, and from what (plan B4 step 1). */
export function modeOf(doc, info) {
  if (!doc) return { mode: null, why: "none" };
  // a file opened as text (Open as Text, a job log) holds no structure: its kind
  // names the file it shows, not something Setup can read
  if (doc.kind === "text") return { mode: null, why: "kind" };
  const fk = doc.fileKind || "";
  if (doc.kind === "trajectory" || fk === "out" || fk === "log") {         // a CRYSTAL output, also one named .log
    // a view made from an output (Run Analysis, Compare with Input) has no file of its own to follow
    if (!doc.path) return { mode: null, why: "kind" };
    return { mode: "follow", source: "path", path: doc.path, out: true };
  }
  if (fk === "d12" && !(info && info.dirty)) return { mode: null, why: "d12" };
  if (fk === "cif" || doc.untitled || doc.kind === "structure") {
    const unsaved = !doc.path || doc.untitled || (info && info.dirty) || fk !== "cif";
    return { mode: "cif2d12", source: unsaved ? "session" : "path", path: doc.path };
  }
  return { mode: null, why: "kind" };
}

/** The CLI mode for a follow-up's calculation type. */
export function followMode(calc, calcTypes) {
  return (calcTypes.opt2d3 || []).includes(calc) ? "opt2d3" : "opt2d12";
}

export function newState(mode, calc) {
  return { mode, calc, answers: {}, patches: {}, hand: null, jsonText: null, result: null,
    previewSha: null, history: [], pos: -1, saved: null, guard: null, layout: null, t: 0 };
}

/** The request for one Setup run: the typed answers, never a re-serialized JSON. */
export function request(st, source, docId, preview) {
  const body = { source, calc_type: st.calc, answers: { ...st.answers }, patches: { ...st.patches },
    preview: !!preview, doc: String(docId), wait: 0.3 };
  if (st.hand !== null && st.hand !== undefined) body.json_text = st.hand;
  return body;
}

/** A form edit as the typed answer it sends: {answers: {field: value}} or {patches: {key: value}}. */
export function editFor(field, value) {
  if (field.how === "patch") return { patches: { [field.id]: value } };
  return { answers: { [field.id]: value } };
}

/** Apply an edit; returns the label for the history ("Functional HSE06 → PBE0"). */
export function applyEdit(st, field, value) {
  const was = field.value;
  if (field.how === "patch") {
    if (value === "" || value === null) delete st.patches[field.id]; else st.patches[field.id] = value;
  } else if (value === null || isDefault(field, value)) {
    delete st.answers[field.id];          // the CLI's own default (the parent's, for a follow-up): no answer needed
  } else {
    st.answers[field.id] = value;
  }
  st.hand = null;
  return `${field.label} ${was ?? "default"} → ${value === null ? "default" : value}`;
}

export function isDefault(field, value) {
  if (field.default === null || field.default === undefined) return false;
  if (field.kind === "yesno") return (value === true || value === "y" || value === "Yes") === (field.default === "Yes");
  return String(value).trim() === String(field.default).trim();
}

export function resetField(st, id) {
  delete st.answers[id];
  delete st.patches[id];
  st.hand = null;
}

export function resetAll(st) {
  const n = Object.keys(st.answers).length + Object.keys(st.patches).length + (st.hand !== null ? 1 : 0);
  st.answers = {}; st.patches = {}; st.hand = null;
  return n;
}

/* ---------------------------------------------------------------- history */
export function snapshot(st, label, mergeKey = null) {
  return { label, t: Date.now(), mergeKey, calc: st.calc, mode: st.mode, answers: { ...st.answers },
    patches: { ...st.patches }, hand: st.hand, flags: { ...(st.flags || {}) }, jsonText: st.jsonText };
}

/** Record the state after an edit. Typing in one field within 1 s merges ("FMIXING 30 → 40"). */
export function record(st, label, mergeKey = null) {
  st.history = st.history.slice(0, st.pos + 1);
  const last = st.history[st.history.length - 1];
  const now = Date.now();
  if (mergeKey && last && last.mergeKey === mergeKey && now - last.t < MERGE_MS) {
    const first = last.label.split(" → ")[0];
    st.history[st.history.length - 1] = { ...snapshot(st, `${first} → ${label.split(" → ").pop()}`, mergeKey) };
    // typed back to where it started ("800" → "8009" → "800"): no step at all
    const prev = st.history[st.history.length - 2];
    if (prev && same(prev, st)) st.history.pop();
  } else {
    st.history.push(snapshot(st, label, mergeKey));
  }
  while (st.history.length > MAX_HISTORY) st.history.shift();
  st.pos = st.history.length - 1;
}

const same = (h, st) => JSON.stringify([h.calc, h.answers, h.patches, h.hand ?? null, h.flags || {}])
  === JSON.stringify([st.calc, st.answers, st.patches, st.hand ?? null, st.flags || {}]);

/** The base entry (the state Setup opened with): undo can always go back to it. */
export function base(st, label) {
  if (!st.history.length) { st.history.push(snapshot(st, label)); st.pos = 0; }
}

function restore(st, snap) {
  st.calc = snap.calc; st.mode = snap.mode; st.answers = { ...snap.answers }; st.patches = { ...snap.patches };
  st.hand = snap.hand; st.flags = { ...(snap.flags || {}) }; st.jsonText = snap.jsonText;
}
export const canUndo = (st) => !!st && st.pos > 0;
export const canRedo = (st) => !!st && st.pos >= 0 && st.pos < st.history.length - 1;
export function undo(st) { if (!canUndo(st)) return null; st.pos -= 1; restore(st, st.history[st.pos]); return st.history[st.pos + 1].label; }
export function redo(st) { if (!canRedo(st)) return null; st.pos += 1; restore(st, st.history[st.pos]); return st.history[st.pos].label; }
export function jump(st, k) { if (k < 0 || k >= st.history.length) return; st.pos = k; restore(st, st.history[k]); }

/** A few KB per entry, counted toward the 64 MB history budget (spec §2.6). */
export function historyBytes(st) {
  return st ? st.history.reduce((t, h) => t + (h.jsonText ? h.jsonText.length * 2 : 0) + 400, 0) : 0;
}

/* ---------------------------------------------------------------- persistence (drafts) */
export function toDraft(st) {
  return { v: 1, mode: st.mode, calc: st.calc, answers: st.answers, patches: st.patches, hand: st.hand, flags: st.flags || {},
    history: st.history.slice(-MAX_HISTORY), pos: st.pos, saved: st.saved, layout: st.layout, t: Date.now() };
}
export function fromDraft(d) {
  if (!d || d.v !== 1) return null;
  const st = newState(d.mode, d.calc);
  Object.assign(st, { answers: d.answers || {}, patches: d.patches || {}, hand: d.hand ?? null, flags: d.flags || {}, history: d.history || [],
    pos: Number.isInteger(d.pos) ? d.pos : (d.history || []).length - 1, saved: d.saved || null, layout: d.layout || null });
  const cur = st.history[st.pos];
  if (cur) st.jsonText = cur.jsonText;
  return st;
}

/* ---------------------------------------------------------------- deck text */
/** Line diff of two decks: [{op: " " | "+" | "-", text}] (LCS, decks are short). */
export function diffLines(a, b) {
  const x = (a || "").split("\n"), y = (b || "").split("\n");
  const n = x.length, m = y.length;
  if (n * m > 4e6) return y.map((t) => ({ op: " ", text: t }));
  const L = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
    L[i][j] = x[i] === y[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  const out = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (x[i] === y[j]) { out.push({ op: " ", text: y[j] }); i++; j++; }
    else if (L[i + 1][j] >= L[i][j + 1]) out.push({ op: "-", text: x[i++] });
    else out.push({ op: "+", text: y[j++] });
  }
  while (i < n) out.push({ op: "-", text: x[i++] });
  while (j < m) out.push({ op: "+", text: y[j++] });
  return out;
}
export const changedCount = (d) => d.filter((r) => r.op !== " ").length;

/** A deck's keywords and numbers, marked for the text view. */
export function highlightDeck(text, esc) {
  return String(text).split("\n").map((line) => {
    const e = esc(line);
    if (/^[A-Z][A-Z0-9_\-+]*$/.test(line.trim())) return `<span class="kw">${e}</span>`;
    if (/^[\s\d.\-+Ee]+$/.test(line) && line.trim()) return `<span class="num">${e}</span>`;
    return e;
  }).join("\n");
}

/** Group the form's rows into sections, in the order Setup shows them. */
export const SECTIONS = ["Method", "k-points and SCF", "Optimization", "Frequencies", "BAND", "DOSS", "Structure", "Other questions"];
export function sections(form) {
  const by = new Map(SECTIONS.map((s) => [s, []]));
  for (const f of form || []) {
    if (f.id === "calc_type") continue;
    if (!by.has(f.section)) by.set(f.section, []);
    by.get(f.section).push(f);
  }
  return [...by.entries()].filter(([, rows]) => rows.length);
}

/** The badge a row shows: default / changed from default (was X) / parent: X. */
export function badge(field, follow) {
  if (field.changed) return { kind: "changed", text: `changed from ${follow ? "parent" : "default"} (was ${field.default ?? "—"})` };
  if (follow) return { kind: "parent", text: "parent" };
  return { kind: "default", text: "default" };
}
