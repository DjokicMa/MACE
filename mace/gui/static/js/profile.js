/* The GUI profile beyond Settings (polish P2): layout and panels ("ui"),
 * remembered choices ("memory") and the Recent lists, kept on the server in
 * $XDG_CONFIG_HOME/mace/gui.json (/api/profile), so they come back on a new
 * local port, in another browser and on another computer that shares the
 * home folder. localStorage keeps a cache (instant first use) and the few
 * things that belong to this browser only.
 *
 * Every remembered choice and where it lives (the guard test
 * tests/test_gui_profile_inventory.py keeps this table honest):
 *
 * | localStorage key (before 1.1.5)           | now                                                          |
 * |-------------------------------------------|--------------------------------------------------------------|
 * | mace.gui.settings.v1, …settings.version   | gui.json settings; localStorage = cache (settings.js)        |
 * | mace.gui.layout.v1                         | ui.layout, except zen and the drawer (this browser only)     |
 * | mace.gui.recent.<dirHash>                  | recent.files.<dirHash>                                       |
 * | mace.gui.explorer.dbOpen, …recentOpen      | ui.explorer                                                  |
 * | mace.gui.db.widths                         | ui.dbWidths                                                  |
 * | mace.gui.setup.collapsed                   | ui.setupCollapsed                                            |
 * | mace.gui.palette.recent                    | ui.paletteRecent                                             |
 * | mace.gui.recentElements                    | ui.recentElements                                            |
 * | mace.gui.export.v1                         | memory.exportImage (last used; the defaults are Settings')   |
 * | mace.gui.newStructure.v1                   | memory.newStructure                                          |
 * | (new) Clean method                         | memory.cleanMethod                                           |
 * | (new) the split into two editor groups     | ui.split.<dirHash> (groups.js): open, ratio, linked, documents |
 * | (new) Submit dialog's last options          | memory.submitLast (Settings › submit.rememberLast)          |
 * | (new) 1.1.6's "Setup now uses Functional …" | memory["notice.setupDefaults116"] (shell.js), shown once     |
 * | mace.gui.display.<dirHash>.<path>          | /api/docprefs kind display (docprefs.js); localStorage cache |
 * | mace.gui.plot.<dirHash>.<path>             | this browser (plot options and undo history; see LOCAL_ONLY)  |
 * | (new) mace.gui.runplot.<dirHash>.<path>    | this browser: Run Analysis plot options, as the plot options  |
 * | (new) mace.gui.runcard.<dirHash>.<path>    | this browser: the energy card's unit and reference, likewise  |
 * | (new) mace.gui.followRun                   | this browser: Follow the run while a live output is watched   |
 * | (new) mace.gui.cmp.dispMode                | this browser: the OPT compare view's displacement (Δr) choice, as the plot options |
 * | (new) mace.gui.scfcard.<dirHash>.<path>    | this browser: the SCF card's scale, unit and step, as the run plot options |
 * | (new) mace.gui.kpath.presets               | this browser: saved k-path edits, offered for plots with the same segments |
 * | (new) mace.gui.gallery.pageSize            | this browser: the Plot Gallery's last panes per page (Settings › Plots wins) |
 * | (new) mace.gui.mech.<dirHash>.<path>       | this browser: a mechanics results document's tab and options, as the plot options |
 * | (new) mace.gui.propunits.v1               | this browser: the Properties tab's unit preset, normalization and each row's unit |
 * | mace.gui.draft.*, mace.gui.setup.<…>       | this browser only: unsaved drafts                            |
 * | mace.gui.builder.<id>.result               | this browser only: a builder's transient result              |
 * | mace.gui.saveConfirmed.<…>                 | this browser only: "already asked before overwriting"        |
 */

/* Keys (prefixes) that stay in this browser on purpose. */
export const LOCAL_ONLY = [
  "mace.gui.settings.v1", "mace.gui.settings.version",        // the Settings cache (settings.js)
  "mace.gui.settings.pending",                                 // Settings changed here, not on the server yet (boot.js reads it)
  "mace.gui.layout.v1",                                        // the layout cache; zen and the drawer stay here
  "mace.gui.profile.",                                         // this module's cache, pending marker, migration flag
  "mace.gui.display.", "mace.gui.docprefs.",                   // the docprefs cache and its pending list
  "mace.gui.plot.",                                            // plot options and undo history (per browser for now)
  "mace.gui.runplot.", "mace.gui.runcard.",                    // Run Analysis and energy card options, as the plot options
  "mace.gui.followRun",                                        // Follow the run, for the live output watched here
  "mace.gui.cmp.dispMode",                                     // the OPT compare view's Δr choice, as the plot options
  "mace.gui.scfcard.",                                         // the SCF card's options (1.1.6), as the run plot options
  "mace.gui.kpath.presets",                                    // saved k-path edits (1.1.6), kept with the plot options
  "mace.gui.gallery.pageSize",                                 // the Plot Gallery's last panes per page (1.1.6; Settings › Plots wins)
  "mace.gui.mech.",                                            // a mechanics results document's tab and options (1.1.6), as the plot options
  "mace.gui.propunits.v1",                                     // the Properties tab's unit preset, normalization and rows' units (1.1.6)
  "mace.gui.draft.", "mace.gui.setup.",                        // unsaved drafts (Setup's per-document draft too)
  "mace.gui.builder.",                                         // a builder's transient result
  "mace.gui.saveConfirmed.",                                   // the overwrite question, asked once per browser
  "mace.gui.recent.",                                          // the legacy Recent key, read once by the migration
  "mace.gui.explorer.", "mace.gui.db.", "mace.gui.palette.",   // legacy keys, read once by the migration
  "mace.gui.recentElements", "mace.gui.export.v1", "mace.gui.newStructure.v1",
  "mace-gui-colors",                                           // the version-1 color key settings.js migrates
];

const CACHE = "mace.gui.profile.cache.v1", PENDING = "mace.gui.profile.pending", MIGRATED = "mace.gui.profile.migrated";
let api = null, dirHash = "", root = "", url = "api/profile";
let data = { rev: 0, ui: {}, memory: {}, recent: { files: {}, folders: [] } };
let patch = null, timer = 0, lastFetch = 0, inflight = null;
/* what the listeners compare: the contents, not the revision counter */
const contents = () => JSON.stringify([data.ui, data.memory, data.recent]);
const listeners = new Set();

function readCache() {
  try { return JSON.parse(localStorage.getItem(CACHE) || "null"); } catch (e) { return null; }
}
function writeCache() {
  try { localStorage.setItem(CACHE, JSON.stringify(data)); } catch (e) { /* private */ }
}
function markPending() {
  try {
    if (patch) localStorage.setItem(PENDING, JSON.stringify({ t: Date.now(), patch }));
    else localStorage.removeItem(PENDING);
  } catch (e) { /* private */ }
}

/* data from a profile payload, only the sections this module owns. The
 * server sends this folder's Recent entry whenever it has one, so a server
 * payload without it means the list was reset elsewhere: the cached copy goes
 * too (else this browser would bring the old list back). */
function adopt(p, fromServer = false) {
  if (!p) return;
  const recentFiles = { ...(data.recent.files || {}), ...((p.recent && p.recent.files) || {}) };
  if (fromServer && dirHash && !(p.recent && p.recent.files && dirHash in p.recent.files)) delete recentFiles[dirHash];
  data = { rev: p.rev || 0, ui: { ...(p.ui || {}) }, memory: { ...(p.memory || {}) },
    recent: { files: recentFiles, folders: (p.recent && p.recent.folders) || [] } };
}

/** Start: the localStorage cache first, then the server's copy (from /api/boot) wins;
 * a change that never reached the server (pending) is applied on top and sent again. */
export function init(apiFn, boot, info) {
  api = apiFn;
  dirHash = (info && info.dir_hash) || "";
  root = (info && info.root) || "";
  const cached = readCache();
  if (cached) adopt(cached);
  if (boot) adopt(boot, true);
  let pend = null;
  try { pend = JSON.parse(localStorage.getItem(PENDING) || "null"); } catch (e) { pend = null; }
  if (pend && pend.patch) { applyLocal(pend.patch); queue(pend.patch); }
  writeCache();
}

function applyLocal(p) {
  for (const sec of ["ui", "memory"]) {
    if (!p[sec]) continue;
    for (const k of p[sec].unset || []) delete data[sec][k];
    Object.assign(data[sec], p[sec].set || {});
  }
  if (p.recent && p.recent.files) Object.assign(data.recent.files, p.recent.files);
  if (p.recent && p.recent.folders) data.recent.folders = p.recent.folders;
}
function merge(into, p) {
  for (const sec of ["ui", "memory"]) {
    if (!p[sec]) continue;
    into[sec] = into[sec] || { set: {}, unset: [] };
    for (const k of p[sec].unset || []) { delete into[sec].set[k]; if (!into[sec].unset.includes(k)) into[sec].unset.push(k); }
    for (const [k, v] of Object.entries(p[sec].set || {})) { into[sec].set[k] = v; into[sec].unset = into[sec].unset.filter((x) => x !== k); }
  }
  if (p.recent) {
    into.recent = into.recent || {};
    if (p.recent.files) into.recent.files = { ...(into.recent.files || {}), ...p.recent.files };
    if (p.recent.folders) into.recent.folders = p.recent.folders;
  }
  return into;
}
function queue(p) {
  patch = merge(patch || {}, p);
  markPending();
  clearTimeout(timer);
  timer = setTimeout(() => flush(), 500);
}

/** Send what waits (500 ms after the last change; at once from flush()). */
export async function flush() {
  clearTimeout(timer);
  if (inflight) await inflight.catch(() => {});      // one request at a time, in order
  if (!patch || !api) return null;
  const body = patch;
  patch = null;
  const run = (async () => {
    try {
      const r = await api("profile", { body });
      if (!patch) markPending();
      lastFetch = Date.now();
      const before = contents();
      adopt(r, true);
      if (patch) applyLocal(patch);            // a change made while this one was on its way
      writeCache();
      if (contents() !== before) listeners.forEach((fn) => fn("server"));
      return r;
    } catch (e) {
      patch = merge(body, patch || {});
      markPending();
      if (notify) notify(`Your layout and remembered choices are saved in this browser only for now: ${e.message}`, "warning");
      return null;
    }
  })();
  inflight = run;
  try { return await run; } finally { if (inflight === run) inflight = null; }
}
let notify = null;
export function setNotifier(fn) { notify = fn; }

/* ---------------------------------------------------------------- reads and writes */
export const ui = (k, fallback = undefined) => (k in data.ui ? data.ui[k] : fallback);
export function setUi(k, v) {
  if (JSON.stringify(data.ui[k]) === JSON.stringify(v)) return;
  if (v === undefined) { delete data.ui[k]; writeCache(); queue({ ui: { unset: [k] } }); return; }
  data.ui[k] = v; writeCache(); queue({ ui: { set: { [k]: v } } });
}
export const memory = (k, fallback = undefined) => (k in data.memory ? data.memory[k] : fallback);
export function remember(k, v) {
  if (JSON.stringify(data.memory[k]) === JSON.stringify(v)) return;
  data.memory[k] = v; writeCache(); queue({ memory: { set: { [k]: v } } });
}
/** This folder's Recent files: [{path, t, pinned}], newest first. */
export const recentFiles = () => ((data.recent.files[dirHash] || {}).items || []).slice();
export const hasRecentFiles = () => !!data.recent.files[dirHash];
export function setRecentFiles(items) {
  const entry = { root, items: items.map((x) => ({ path: x.path, t: x.t || 0, pinned: !!x.pinned })) };
  data.recent.files[dirHash] = entry; writeCache();
  queue({ recent: { files: { [dirHash]: entry } } });
}
export const recentFolders = () => (data.recent.folders || []).slice();
export function setRecentFolders(list) {
  data.recent.folders = list.slice(0, 20); writeCache();
  queue({ recent: { folders: data.recent.folders } });
}
export function onChange(fn) { listeners.add(fn); return () => listeners.delete(fn); }
export const rev = () => data.rev;
/** Read the server's profile again (after an import or a reset) and say what changed. */
export async function reload() {
  if (!api) return;
  try {
    const r = await api("profile");
    data.recent.files = {};
    adopt(r);
    writeCache();
    listeners.forEach((fn) => fn("server"));
  } catch (e) { /* the cache stands */ }
}
export const all = () => JSON.parse(JSON.stringify(data));

/* ---------------------------------------------------------------- the one-time upload of a browser's old keys */
/** A browser that used a pre-1.1.5 build has these in localStorage: they go to
 * the server once (the server's copy wins where it has one). The old keys stay
 * for one release, so a downgrade still finds them. */
export function migrateLegacy() {
  try { if (localStorage.getItem(MIGRATED) === "1") return false; } catch (e) { return false; }
  const get = (k) => { try { return localStorage.getItem(k); } catch (e) { return null; } };
  const json = (k) => { try { return JSON.parse(get(k) || "null"); } catch (e) { return null; } };
  const ui0 = {}, mem0 = {};
  const lay = json("mace.gui.layout.v1");
  if (lay && !("layout" in data.ui)) { const { zen, ...rest } = lay; ui0.layout = rest; }
  const ex = {};
  if (get("mace.gui.explorer.dbOpen") !== null) ex.dbOpen = get("mace.gui.explorer.dbOpen") === "1";
  if (get("mace.gui.explorer.recentOpen") !== null) ex.recentOpen = get("mace.gui.explorer.recentOpen") !== "0";
  if (Object.keys(ex).length && !("explorer" in data.ui)) ui0.explorer = ex;
  const w = json("mace.gui.db.widths");
  if (w && !("dbWidths" in data.ui)) ui0.dbWidths = w;
  const sc = json("mace.gui.setup.collapsed");
  if (sc && !("setupCollapsed" in data.ui)) ui0.setupCollapsed = sc;
  const pr = json("mace.gui.palette.recent");
  if (pr && !("paletteRecent" in data.ui)) ui0.paletteRecent = pr;
  const re = json("mace.gui.recentElements");
  if (re && !("recentElements" in data.ui)) ui0.recentElements = re;
  const ns = json("mace.gui.newStructure.v1");
  if (ns && !("newStructure" in data.memory)) mem0.newStructure = ns;
  const p = {};
  if (Object.keys(ui0).length) { p.ui = { set: ui0 }; Object.assign(data.ui, ui0); }
  if (Object.keys(mem0).length) { p.memory = { set: mem0 }; Object.assign(data.memory, mem0); }
  const rec = json(`mace.gui.recent.${dirHash}`);
  if (Array.isArray(rec) && rec.length && !hasRecentFiles()) {
    const items = rec.map((x) => (typeof x === "string" ? { path: x, t: 0, pinned: false } : x))
      .filter((x) => x && typeof x.path === "string" && !x.path.startsWith("/") && !x.path.split("/").includes(".."));
    if (items.length) {
      data.recent.files[dirHash] = { root, items };
      p.recent = { files: { [dirHash]: data.recent.files[dirHash] } };
    }
  }
  try { localStorage.setItem(MIGRATED, "1"); } catch (e) { /* private */ }
  if (Object.keys(p).length) { writeCache(); queue(p); return true; }
  return false;
}

/* ---------------------------------------------------------------- freshness and leaving */
/** Another tab or server changed the profile: re-read it when this page becomes
 * visible or gets the focus (at most every 10 s), and say what changed. */
async function refreshFromServer() {
  if (!api || patch || Date.now() - lastFetch < 10000) return;
  lastFetch = Date.now();
  try {
    const r = await api("profile");
    if (!r || r.rev === data.rev || patch) return;
    const before = contents();
    adopt(r, true);
    writeCache();
    if (contents() !== before) listeners.forEach((fn) => fn("server"));
  } catch (e) { /* offline: the cache stands */ }
}
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") refreshFromServer(); });
addEventListener("focus", () => refreshFromServer());

/* A change made just before the tab is closed or hidden: sent at once, as a
 * keepalive request that outlives the page. */
function leaveFlush() {
  if (!patch) return;
  clearTimeout(timer);
  const text = JSON.stringify(patch);
  if (text.length > 60000) { flush(); return; }        // too large for keepalive: the pending marker covers it
  try {
    fetch(url, { method: "POST", keepalive: true, credentials: "same-origin",
      headers: { "Content-Type": "application/json", "X-Requested-With": "mace-gui" }, body: text });
    // the pending marker stays: if this request is lost, the next load sends it again (a patch is idempotent)
    patch = null;
  } catch (e) { /* the pending marker sends it next time */ }
}
addEventListener("pagehide", leaveFlush);
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden") leaveFlush(); });
