/* Settings: localStorage holds the full effective object (instant on boot);
 * ~/.config/mace/gui.json on the server holds only the non-default keys and
 * wins for the keys it has. Changes save after 500 ms of quiet.
 *
 * SCHEMA_VERSION 2 added cellColor, backgroundColor, defaultDb, functional
 * and the §9 resource keys (max_workers, cache_mb, gpu_device), and moved the
 * old "mace-gui-colors" choice into colorScheme. The localStorage key keeps
 * its name: version 1 objects load as they are (only known keys of the right
 * type are kept).
 *
 * SCHEMA_VERSION 3 (the 1.1.5 polish round, one bump for every plan's new
 * keys): GPU 0 is the default gpu_device. localStorage held the full
 * effective object, so a version-2 object's "1" was the old default, not a
 * choice: it is dropped once and falls to the new default. The other keys
 * new in 3 (the viewer's ranges and look, plots, File History, sessions,
 * cluster.*, export.*, files.openFreqAs, submit.rememberLast, ...) need no
 * migration: an older object lacks them and takes their defaults. */
export const VERSION = 3;
export const SCHEMA_VERSION = VERSION;
const KEY = "mace.gui.settings.v1";
const VERSION_KEY = "mace.gui.settings.version";
const LEGACY_COLORS = "mace-gui-colors";

export const DEFAULTS = Object.freeze({
  theme: "system", density: "comfortable", uiScale: 100, reduceMotion: "system",
  // VESTA, as the classic views have always drawn (open decision D1: the spec's table says Jmol)
  colorScheme: "vesta", radiusSet: "atomic", atomScale: 0.4, bondRadius: 0.12, bondTolerance: 1.15,
  bondMetalMetal: false, bondColoring: "split", background: "theme", backgroundColor: "", labels: "off", labelSize: 12,
  projection: "orthographic", "viewport.defaultView": "standard", quality: "auto", antialias: true, showAxes: true, showCell: true,
  showBoundary: true, showPolyhedra: false, depthCue: 0.15,
  // polish A3/A6: the defaults of every document's Range & boundary and Rendering groups
  displayLo: [0, 0, 0], displayHi: [1, 1, 1], bondedOutside: "off", lightStrength: 1, lightDirection: "front", ambient: 0,
  outline: false, outlineWidth: 0.1, outlineColor: "", ambientOcclusion: 0, renderPreset: "default", cellColor: "", coordMode: "frac", lengthUnit: "Å", energyUnit: "eV",
  "decimals.coord": 5, "decimals.length": 4, "decimals.angle": 3,
  symmetryModeDefault: "sym", symprec: 1e-3, defaultElement: "C", nudgeStepFrac: 0.01, nudgeStepCart: 0.05,
  wrapIntoCell: true, confirmOverwrite: true, keepBackup: true, autosaveDrafts: 10, undoLimit: 200,
  // Phase 4: Point Defect's default supercell keeps defect images this far apart (Å)
  "defects.minImageDistance": 10,
  keymap: [], explorerExclude: ["*.bak", "__pycache__", "*.tmp"], defaultDb: "", recentLimit: 15, saveFormat: "cif",
  // MACE defaults: blank means "what the CLI does" (shown next to each), never a GUI-only value
  functional: "", basisSet: "", exampleConfig: "", optionsFileName: "gui_options.json",
  runPreflightOnPreview: false, setupAutoPreview: true, heartbeatSeconds: 10, devShowCommandIds: false,
  // spec §9: 0 workers = auto (a quarter of the cores, at most 8); the cache cap in MB; the GPU future code may use
  max_workers: 0, cache_mb: 512, gpu_device: "0",
  // Plots and run analysis (PLAN_analysis C3): defaults for newly opened plots; a plot's own choices are kept per document
  "plots.relTo": "final", "plots.logDeltaE": false, "plots.per": "cell", "plots.lineWidth": 1.5, "plots.exportScale": 2,
  "analysis.imagNoiseCm": 20,
  // Live files and databases (C17, C18) and File History (C20)
  "database.refreshSeconds": 30, "files.liveMinutes": 10,
  "files.historyKeep": 20, "files.historyDays": 30, "files.historyMaxMB": 500, "files.historyOutputsMaxMB": 1000,
  // sessions (PLAN_connect D16): sign out after N hours without use (0 = never), at most N days;
  // stop the server after N hours without use and with no task running (0 = never)
  "session.idleHours": 12, "session.maxDays": 7, "server.idleStopHours": 0,
  // the Cluster view (area F): own-job poll 30..300 s, the activity button, browser notifications
  "cluster.pollSeconds": 60, "cluster.showView": "auto", "cluster.notify": false,
  // plan E (CLI parity): what a FREQ output opens as, notification timing, volumes, spectra, molecules, export units
  "files.openFreqAs": "modes", "notifications.infoSeconds": 6, "volume.quality": "medium", "volume.mepColors": "cli",
  "spectra.lineShape": "lorentzian", "structure.vacuumGap": 8,
  // export (plan E9; PLAN_viewer P6 owns these keys at the merge): PNG resolution, physical units, SVG shading
  "export.dpi": 150, "export.units": "in", "export.vectorShading": "flat",
  // Submit (PLAN_submit): remember the Submit dialog's options (cluster.pollSeconds, above, also sets how often Jobs asks SLURM)
  "submit.rememberLast": true,
});

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const listeners = new Set();
let values = { ...DEFAULTS };
let saveTimer = 0;
let api = null;

export function migrate(obj) {
  // version 1 is the first schema: keep only known keys of the right type
  const out = {};
  for (const [k, v] of Object.entries(obj || {})) {
    if (k in DEFAULTS && typeof v === typeof DEFAULTS[k]) out[k] = v;
  }
  return out;
}

export function init(apiFn) {
  api = apiFn;
  let stored = {};
  let storedVersion = 0;
  try {
    stored = JSON.parse(localStorage.getItem(KEY) || "{}") || {};
    storedVersion = Number(localStorage.getItem(VERSION_KEY)) || 0;
  } catch (e) { stored = {}; }
  // version 2 -> 3: GPU 1 was the default, stored because the whole object was
  if (storedVersion < 3 && stored.gpu_device === "1") delete stored.gpu_device;
  values = { ...DEFAULTS, ...migrate(stored) };
  // version 1 -> 2: the classic views' own element-color choice becomes colorScheme, once
  try {
    const legacy = localStorage.getItem(LEGACY_COLORS);
    if (legacy !== null) {
      if (!("colorScheme" in stored) && typeof legacy === "string" && legacy) { values.colorScheme = legacy; setPending([...pendingKeys(), "colorScheme"]); }
      localStorage.removeItem(LEGACY_COLORS);
      persistLocal();
      if (!same(values.colorScheme, DEFAULTS.colorScheme)) { clearTimeout(saveTimer); saveTimer = setTimeout(flush, 500); }
    }
    localStorage.setItem(VERSION_KEY, String(VERSION));
  } catch (e) { /* private window */ }
}

/** Merge the server's file: it wins, except for a key changed here that has
 * not reached it yet (the pending list), which is sent again. A key the
 * server does not hold is its default (gui.json keeps non-default keys only). */
export async function syncFromServer(given) {
  try {
    const r = given || await api("settings");
    const server = migrate(r.settings);
    serverValues = { ...server };
    const pend = new Set(pendingKeys());
    const want = { ...DEFAULTS, ...server };
    const changed = Object.keys(DEFAULTS).filter((k) => !pend.has(k) && !same(values[k], want[k]));
    for (const k of changed) values[k] = want[k];
    persistLocal();
    changed.forEach((k) => listeners.forEach((fn) => fn(k, values[k])));
    if (pend.size) { clearTimeout(saveTimer); saveTimer = setTimeout(flush, 50); }
    return r.path;
  } catch (e) { return null; }
}
/* Keys changed in this browser that the server has not stored yet (first paint, boot.js, reads them too). */
const PENDING_KEY = "mace.gui.settings.pending";
function pendingKeys() { try { return JSON.parse(localStorage.getItem(PENDING_KEY) || "[]") || []; } catch (e) { return []; } }
function setPending(keys) {
  try { if (keys.length) localStorage.setItem(PENDING_KEY, JSON.stringify([...new Set(keys)])); else localStorage.removeItem(PENDING_KEY); }
  catch (e) { /* private */ }
}

export const get = (k) => values[k];
export const all = () => ({ ...values });
export const nonDefault = () => Object.fromEntries(Object.entries(values).filter(([k, v]) => !same(v, DEFAULTS[k])));
export const isDefault = (k) => same(values[k], DEFAULTS[k]);
export function onChange(fn) { listeners.add(fn); return () => listeners.delete(fn); }

export function set(k, v) {
  if (!(k in DEFAULTS)) throw new Error(`unknown setting ${k}`);
  if (same(values[k], v)) return;
  values[k] = v;
  persistLocal();
  setPending([...pendingKeys(), k]);
  listeners.forEach((fn) => fn(k, v));
  clearTimeout(saveTimer);
  saveTimer = setTimeout(flush, 500);
}
export const reset = (k) => set(k, DEFAULTS[k]);
/** Every setting back to its default (Settings › Advanced › Reset all). */
export function resetAll() { Object.keys(DEFAULTS).forEach((k) => set(k, DEFAULTS[k])); }
/** Several values at once (an imported settings file); unknown keys and wrong
 * types are skipped and returned. */
export function setMany(obj) {
  const skipped = [];
  for (const [k, v] of Object.entries(obj || {})) {
    if (!(k in DEFAULTS) || typeof v !== typeof DEFAULTS[k] || (Array.isArray(DEFAULTS[k]) && !Array.isArray(v))) { skipped.push(k); continue; }
    set(k, v);
  }
  return skipped;
}

function persistLocal() {
  try { localStorage.setItem(KEY, JSON.stringify(values)); } catch (e) { /* private window */ }
}
/* A change made just before the tab is closed or hidden: sent at once as a keepalive
 * request (sendBeacon cannot carry the header the server requires of a POST). */
function leaveFlush() {
  if (!api || !pendingKeys().length) return;
  clearTimeout(saveTimer);
  const { set, unset } = settingsPatch();
  if (!Object.keys(set).length && !unset.length) return;
  try {
    fetch("api/profile", { method: "POST", keepalive: true, credentials: "same-origin",
      headers: { "Content-Type": "application/json", "X-Requested-With": "mace-gui" },
      body: JSON.stringify({ settings: { set, unset } }) });
  } catch (e) { /* the pending list sends it on the next load */ }
}
addEventListener("pagehide", leaveFlush);
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden") leaveFlush(); });
/* What the server's file holds, as far as this page knows: a key the server
 * rejects goes back to it (or to the default when the file has none). */
let serverValues = {};
let notify = () => {};
let failToastAt = -Infinity, retryTimer = 0;
const LABELS = { max_workers: "CPU workers", cache_mb: "Cache limit", gpu_device: "GPU", keymap: "Keybindings" };
const label = (k) => LABELS[k] || k.replace(/([A-Z])/g, " $1").replace(/^./, (c) => c.toUpperCase());
/** The shell's toast, for "Couldn't save …" messages. */
export function setNotifier(fn) { notify = fn; }

/** Save now. The server stores each good key and names each bad one; a bad
 * one reverts here too. Unreachable: the values stay in this browser, it
 * says so once, and tries once more after 30 s (then on the next change). */
/** The change since the server's copy, as a per-key patch (a second server's keys are never touched). */
function settingsPatch() {
  const body = nonDefault(), set = {}, unset = [];
  for (const [k, v] of Object.entries(body)) if (!same(serverValues[k], v)) set[k] = v;
  for (const k of Object.keys(serverValues)) if (!(k in body) && k in DEFAULTS) unset.push(k);
  return { body, set, unset };
}
export async function flush(isRetry = false) {
  clearTimeout(saveTimer);
  if (!api) return;
  const { body, set, unset } = settingsPatch();
  const sentKeys = pendingKeys();
  // nothing differs from the server's copy: no request (a save never bumps the file for nothing)
  if (!Object.keys(set).length && !unset.length) { setPending([]); return { path: null, rejected: {}, saved: [] }; }
  let r;
  try {
    const p = await api("profile", { body: { settings: { set, unset } } });
    // the profile route names a rejected key "settings.<key>"; this module's callers know plain keys
    const rej = {};
    for (const [k, why] of Object.entries((p && p.rejected) || {})) if (k.startsWith("settings.")) rej[k.slice(9)] = why;
    r = { path: p && p.path, rejected: rej, saved: Object.keys(set).filter((k) => !(k in rej)), server: (p && p.settings) || null };
  } catch (e) {
    if (performance.now() - failToastAt > 30000) { failToastAt = performance.now(); notify(`Settings saved in this browser only: ${e.message}`, "warning"); }
    if (!isRetry && !retryTimer) retryTimer = setTimeout(() => { retryTimer = 0; flush(true); }, 30000);
    return null;
  }
  // what changed while this request was out stays pending
  setPending(pendingKeys().filter((k) => !sentKeys.includes(k) || (k in set && !same(values[k], set[k]))));
  const rejected = (r && r.rejected) || {};
  const before = serverValues;
  serverValues = r.server ? migrate(r.server) : Object.fromEntries(Object.entries(body).filter(([k]) => !(k in rejected)));
  const bad = Object.keys(rejected).filter((k) => k in DEFAULTS);
  for (const k of bad) {
    if (k in before) serverValues[k] = before[k];
    const back = k in before ? before[k] : DEFAULTS[k];
    if (!same(values[k], back)) { values[k] = back; listeners.forEach((fn) => fn(k, back)); }
  }
  if (bad.length) {
    persistLocal();
    notify(bad.map((k) => `Couldn't save ${label(k)}: ${rejected[k]}`).join(" · "), "warning");
  }
  return r;
}
