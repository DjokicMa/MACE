/* Per-document options (shared rule S1 of the 1.1.5 polish plans): a
 * document keeps only the values that differ from the Settings defaults,
 * per kind ("display": the viewer's range and rendering; "plot": a plot's
 * options).
 *
 * The server's copy (/api/docprefs, $XDG_CONFIG_HOME/mace/gui-docprefs.json)
 * is the one that counts: it comes back on a new local port and in a fresh
 * browser. localStorage (mace.gui.<kind>.<dirHash>.<path>) is a cache, read
 * first so a reopened document draws at once; the server's answer wins.
 * A change goes to localStorage at once and to the server after 500 ms of
 * quiet, or at once (a keepalive request) when the page is hidden or
 * closed. Untitled and built documents (no path) keep theirs for the
 * session only. */

let api = null, dirHash = "", postUrl = "";
const mem = new Map();          // `${kind}|${key}` -> overrides
const fetched = new Set();
const pending = new Map();      // `${kind}|${path}` -> {kind, path, timer}
const listeners = new Set();

/** The document's storage key: its path, or doc<id> for an untitled one (session only). */
export const keyOf = (doc) => (doc && doc.path && !doc.untitled && !(doc.frame >= 0) ? doc.path : `doc${doc ? doc.id : 0}`);
const persistent = (doc) => !!(doc && doc.path && !doc.untitled && !(doc.frame >= 0));
const lsKey = (kind, path) => `mace.gui.${kind}.${dirHash}.${path}`;

export function init(apiFn, hash, url) {
  api = apiFn; dirHash = hash || ""; postUrl = url || "";
}
export function onChange(fn) { listeners.add(fn); return () => listeners.delete(fn); }
const emit = (kind, doc, keys) => listeners.forEach((fn) => fn(kind, doc, keys));

/** The document's overrides (a copy). The first call for a saved document
 * also asks the server, and says so (onChange) when its copy differs. */
export function get(kind, doc) {
  const k = `${kind}|${keyOf(doc)}`;
  if (!mem.has(k)) {
    let v = {};
    if (persistent(doc)) {
      try { v = JSON.parse(localStorage.getItem(lsKey(kind, doc.path)) || "{}") || {}; } catch (e) { v = {}; }
    }
    mem.set(k, v);
  }
  if (persistent(doc) && !fetched.has(k) && api) {
    fetched.add(k);
    api("docprefs?" + new URLSearchParams({ kind, path: doc.path })).then((r) => {
      if (pending.has(`${kind}|${doc.path}`)) return;          // a newer local change is on its way
      const server = (r && r.values) || {};
      const local = mem.get(k) || {};
      if (JSON.stringify(server) === JSON.stringify(local)) return;
      mem.set(k, server);
      cache(kind, doc.path, server);
      emit(kind, doc, [...new Set([...Object.keys(server), ...Object.keys(local)])]);
    }).catch(() => { fetched.delete(k); });
  }
  return { ...mem.get(k) };
}

function cache(kind, path, v) {
  try {
    if (Object.keys(v).length) localStorage.setItem(lsKey(kind, path), JSON.stringify(v));
    else localStorage.removeItem(lsKey(kind, path));
  } catch (e) { /* private window */ }
}

/** Replace the overrides of one document (already without default values). */
export function put(kind, doc, values, keys = null) {
  const k = `${kind}|${keyOf(doc)}`;
  mem.set(k, { ...values });
  if (persistent(doc)) {
    cache(kind, doc.path, values);
    schedule(kind, doc.path);
  }
  emit(kind, doc, keys || Object.keys(values));
}

function schedule(kind, path) {
  const pk = `${kind}|${path}`;
  const p = pending.get(pk) || { kind, path, timer: 0 };
  clearTimeout(p.timer);
  p.timer = setTimeout(() => send(p), 500);
  pending.set(pk, p);
  markPending();
}

function body(p) { return { kind: p.kind, path: p.path, values: mem.get(`${p.kind}|${p.path}`) || {} }; }

let inflight = Promise.resolve();
async function send(p) {
  pending.delete(`${p.kind}|${p.path}`);
  markPending();
  const run = inflight.then(() => api("docprefs", { body: body(p) }));
  inflight = run.catch(() => {});
  try { await run; }
  catch (e) {
    // kept in this browser (the cache); tried again with the next change or the next load
    pending.set(`${p.kind}|${p.path}`, { ...p, timer: 0 });
    markPending();
  }
}

/** Send what waits now (Settings › Advanced, a stop, the tests). */
export async function flush() {
  const all = [...pending.values()];
  all.forEach((p) => clearTimeout(p.timer));
  await Promise.all(all.map((p) => send(p)));
  await inflight;                                   // and what a timer already sent
}

/* What has not reached the server yet survives a reload: retried at boot. */
const PENDING_KEY = "mace.gui.docprefs.pending";
function markPending() {
  try {
    if (pending.size) localStorage.setItem(PENDING_KEY, JSON.stringify([...pending.values()].map((p) => ({ kind: p.kind, path: p.path, dir: dirHash }))));
    else localStorage.removeItem(PENDING_KEY);
  } catch (e) { /* private */ }
}
export function retryPending() {
  let list = [];
  try { list = JSON.parse(localStorage.getItem(PENDING_KEY) || "[]") || []; } catch (e) { list = []; }
  for (const p of list) {
    if (p.dir !== dirHash) continue;
    const doc = { path: p.path };
    get(p.kind, doc);              // the cached values into memory
    fetched.add(`${p.kind}|${p.path}`);
    schedule(p.kind, p.path);
  }
}

/* The page is hidden or closed with a change still waiting: send it now, a
 * keepalive request that outlives the page (sendBeacon cannot carry the
 * header the server requires of a POST). */
function leaveFlush() {
  if (!pending.size || !postUrl) return;
  for (const p of [...pending.values()]) {
    clearTimeout(p.timer);
    pending.delete(`${p.kind}|${p.path}`);
    try {
      fetch(postUrl, { method: "POST", keepalive: true, credentials: "same-origin",
        headers: { "Content-Type": "application/json", "X-Requested-With": "mace-gui" }, body: JSON.stringify(body(p)) });
    } catch (e) { pending.set(`${p.kind}|${p.path}`, p); }
  }
  // the pending list in localStorage stays until a later send succeeds: a lost keepalive is sent again next load
}
addEventListener("pagehide", leaveFlush);
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden") leaveFlush(); });

export const _pending = () => pending.size;
/** After Reset › Per-document view options: this page and this browser forget them too. */
export function forgetAll() {
  mem.clear(); fetched.clear(); pending.clear(); markPending();
  try {
    for (let k = localStorage.length - 1; k >= 0; k--) {
      const key = localStorage.key(k);
      if (key && key.startsWith(`mace.gui.display.${dirHash}.`)) localStorage.removeItem(key);
    }
  } catch (e) { /* private */ }
  listeners.forEach((fn) => fn("display", null, []));
}
