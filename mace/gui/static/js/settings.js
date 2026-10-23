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
 * migration: an older object lacks them and takes their defaults.
 *
 * Layers (1.1.6): the served folder's mace-folder.toml (setFolder, from
 * /api/boot) sits above the user's values for the keys it sets, and a change
 * made here to such a key stays in this session (sessionStorage) until it is
 * saved to the folder or as the user's default. get() is the effective value:
 * session > folder > the user's own > DEFAULTS. Only the user's own values go
 * to localStorage and gui.json, so a folder's values never leak into them;
 * without a folder file every path below is the one of 1.1.5. */
export const VERSION = 3;
export const SCHEMA_VERSION = VERSION;
const KEY = "mace.gui.settings.v1";
const VERSION_KEY = "mace.gui.settings.version";
const LEGACY_COLORS = "mace-gui-colors";

export const DEFAULTS = Object.freeze({
  // plots.surface (1.1.6): "theme" (the plots follow it), or a "light" or "dark" plot in any theme;
  // plots.style: the plot preset of every family (plotpresets.js: default, publication, vesta, presentation,
  // grayscale, colorblind); plots.style.<family>: "same" (follow it) or the family's own preset
  theme: "system", "plots.surface": "theme", "plots.style": "default", density: "comfortable", uiScale: 100, reduceMotion: "system",
  "plots.style.band": "same", "plots.style.spectra": "same", "plots.style.maps": "same", "plots.style.scf": "same",
  // band/DOS (1.1.6): DOS colors (auto, total, element, orbital, element-orbital) and the total beside them; the
  // spin pair (blank: the preset's, presets.py SPIN_COLORS), spin down's line style, the spin DOS (auto: mirrored
  // for a spin-polarized run, one-sided for a closed-shell one)
  "plots.dosColors": "auto", "plots.dosTotal": true, "plots.dosSmooth": 0, "plots.spinColors": "auto", "plots.spinUp": "", "plots.spinDown": "", "plots.bandColor": "", "plots.bandStyle": "auto", "plots.spinDownDash": "dash", "plots.spinDos": "auto",
  // VESTA, as the classic views have always drawn (open decision D1: the spec's table says Jmol)
  // bondMin and bondPairs (1.1.6): the rest of the bond rule (geom/core.js bondRule), set in Build › Calculate Bonds…
  colorScheme: "vesta", radiusSet: "atomic", atomScale: 0.4, bondRadius: 0.12, bondTolerance: 1.15, bondMin: 0.1, bondPairs: {},
  bondMetalMetal: false, bondColoring: "split", background: "theme", backgroundColor: "", labels: "off", labelSize: 12,
  projection: "orthographic", "viewport.defaultView": "standard", quality: "auto", antialias: true, showAxes: true, showCell: true,
  showBoundary: true, showPolyhedra: false, depthCue: 0.15, invertWheelZoom: false,
  // 1.1.6: the coordination polyhedra (each document can change them: Inspector › Rendering › Polyhedra): the center
  // elements ("auto": the cations, without a perovskite's A site; else symbols "Ti,Si"), face opacity, edges, center atom
  polyCenters: "auto", polyOpacity: 0.5, polyEdges: true, polyCenterAtom: "small",
  // 1.1.7: how a drag turns the 3D views (free trackball or a turntable) and its angle snap (off or degrees)
  "viewport.rotation": "free", "viewport.angleSnap": "off",
  // 1.1.6: Angle snap "custom" snaps by this angle (°); also the step Ctrl snaps by while the snap is off
  "viewport.snapStep": 15,
  // 1.1.6: the Rotate tool's pivot (each document keeps its own, "atom:N" a picked atom): ROTATE_PIVOT
  "viewport.rotatePivot": "centroid",
  // ... and where the wheel zooms: toward the view center (the rotation center) or the mouse pointer
  "viewport.zoomToward": "center",
  // repeats along a non-periodic axis (a slab's c): the cell's extent, the atoms' + a gap (Å), or CRYSTAL's 500 Å
  "viewport.vacuumMode": "bbox", "viewport.vacuumGap": 10,
  "viewport.printPatterns": "off",
  // polish A3/A6: the defaults of every document's Range & boundary and Rendering groups
  displayLo: [0, 0, 0], displayHi: [1, 1, 1], bondedOutside: "off", lightStrength: 1, lightDirection: "front", ambient: 0,
  outline: false, outlineWidth: 0.1, outlineColor: "", ambientOcclusion: 0, renderPreset: "default", renderCustom: null, shininess: 0,
  // 1.1.7: the Style dropdown (viewport/presets.js STYLES): how atoms are drawn and colored
  stylePreset: "default", styleCustom: null, atomDraw: "auto", atomColor: "scheme", atomRadii: "auto", styleOutline: false, cellColor: "", coordMode: "frac", lengthUnit: "Å", energyUnit: "eV",
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
  "plots.relTo": "final", "plots.segments": "continuous", "plots.logDeltaE": false, "plots.per": "cell", "plots.lineWidth": 1.5, "plots.background": "theme", "plots.backgroundColor": "", "plots.exportScale": 2,
  "analysis.imagNoiseCm": 20,
  // 1.1.6: Analyze Structures… runs this many files at once (1 to 4: never the whole machine)
  "analysis.batchJobs": 2,
  // Live files and databases (C17, C18) and File History (C20)
  "database.refreshSeconds": 30, "files.liveMinutes": 10,
  "files.historyKeep": 20, "files.historyDays": 30, "files.historyMaxMB": 500, "files.historyOutputsMaxMB": 1000,
  // sessions (PLAN_connect D16): sign out after N hours without use (0 = never), at most N days;
  // stop the server after N hours without use and with no task running (0 = never)
  "session.idleHours": 12, "session.maxDays": 7, "server.idleStopHours": 0,
  // the Cluster view (area F): own-job poll 30..300 s, the activity button, browser notifications
  "cluster.pollSeconds": 60, "cluster.showView": "auto", "cluster.notify": false,
  "update.check": true, "update.channel": "release", "update.intervalHours": 6,
  // plan E (CLI parity): what a FREQ output opens as, notification timing, volumes, spectra, molecules, export units
  "files.openFreqAs": "modes", "notifications.infoSeconds": 6, "volume.quality": "medium", "volume.mepColors": "chemistry",
  "spectra.lineShape": "lorentzian", "structure.vacuumGap": 8,
  // 1.1.6: Analysis › Porosity's surfaces in the 3D view: which are drawn, how opaque, solid or wireframe,
  // and the closed pockets in one color or each its own
  "pores.accessible": true, "pores.inaccessible": true, "pores.opacity": 0.5, "pores.style": "solid", "pores.colorPockets": false,
  // 1.1.6: the pores' shading and sides, a drag without their faint near walls (off), and every 3D surface's mesh quality
  "pores.shading": "smooth", "pores.sides": "auto", "pores.dropNearWhileDragging": false, "mesh.quality": "standard",
  // export (plan E9; PLAN_viewer P6 owns these keys at the merge): PNG resolution, physical units, SVG shading
  "export.dpi": 150, "export.units": "in", "export.vectorShading": "flat",
  // Submit (PLAN_submit): remember the Submit dialog's options (cluster.pollSeconds, above, also sets how often Jobs asks SLURM)
  "submit.rememberLast": true,
  // 1.1.6: band/DOS defaults (a folder's [plots] sets them for the CLI too) and the Database view's columns
  "plots.energyWindow": [-6, 6], "plots.fermiShift": 0,
  // 1.1.6: the Inspector's Basic / Advanced choice per view kind ({structure: "advanced", ...}; unset = Basic),
  // and how a FREQ mode is scaled (its zero-point classical amplitude, or the largest atom at 0.5 Å)
  "inspector.tier": {}, "freq.amplitudeScale": "physical",
  // 1.1.6: Inspector detail: that table ("remember"), or every view in Basic or in Advanced
  "inspector.detail": "remember",
  // 1.1.6: the activity bar's order and hidden views (command ids), and the bar itself (View › Activity Bar)
  "activity.order": [], "activity.hidden": [], "activity.visible": true,
  "database.columns": ["material_id", "formula", "space_group", "dimensionality", "n_calcs", "n_completed", "status", "updated_at"],
});

const PRESET_IDS = ["default", "publication", "vesta", "presentation", "grayscale", "colorblind"];
const CHOICES = {
  "plots.style": PRESET_IDS,
  ...Object.fromEntries(["band", "spectra", "maps", "scf"].map((f) => [`plots.style.${f}`, ["same", ...PRESET_IDS]])),
  "plots.dosColors": ["auto", "total", "element", "orbital", "element-orbital"],
  "plots.spinDownDash": ["dash", "dot", "dashdot", "solid"], "plots.spinDos": ["auto", "mirrored", "overlaid"],
  "plots.spinColors": ["auto", "mace", "blue-red", "black-vermilion", "custom"],
  "plots.bandStyle": ["auto", "single", "overlay", "ink"],
  // 1.1.6's "cli" (red = positive, then the default, so stored with everything else) reads as the new default
  "volume.mepColors": ["chemistry", "physics"],
  "viewport.rotation": ["free", "axes"], "viewport.angleSnap": ["off", "15", "30", "45", "90", "custom"],
  "viewport.zoomToward": ["center", "pointer"],
  "freq.amplitudeScale": ["physical", "normalized"],
  "inspector.detail": ["remember", "basic", "advanced"],
  "viewport.printPatterns": ["off", "auto", "all"],
  polyCenterAtom: ["small", "hidden", "full"],
  // 1.1.6: the pore surfaces' look and every 3D surface's mesh quality (viewport/mesh_quality.js)
  "pores.style": ["solid", "wireframe", "edges", "points"], "pores.shading": ["smooth", "flat"], "pores.sides": ["auto", "single", "double"],
  "mesh.quality": ["draft", "standard", "high", "publication"],
};
/* polyCenters (settings.py check): "auto", "" (none) or element symbols separated by commas */
export const POLY_CENTERS = /^(auto|([A-Z][a-z]?(,[A-Z][a-z]?)*)?)$/;
/* bondPairs (settings.py _bond_pairs_problem): {"O-Si": {min, max} | {never: true}}, at most 200 pairs, 0 to 10 Å */
export function bondPairsOk(v) {
  if (!v || typeof v !== "object" || Array.isArray(v) || Object.keys(v).length > 200) return false;
  const len = (x) => x === undefined || (typeof x === "number" && Number.isFinite(x) && x >= 0 && x <= 10);
  return Object.entries(v).every(([k, r]) => /^[A-Z][a-z]?-[A-Z][a-z]?$/.test(k) && r && typeof r === "object" && !Array.isArray(r)
    && Object.keys(r).length > 0 && Object.keys(r).every((f) => ["min", "max", "never"].includes(f))
    && (r.never === undefined || typeof r.never === "boolean") && len(r.min) && len(r.max)
    && !(typeof r.min === "number" && typeof r.max === "number" && r.max <= r.min));
}
/* viewport.rotatePivot (settings.py ROTATE_PIVOT, viewport/gizmo_math.js PIVOT_RE): centroid, com, cell, origin or atom:N */
export const ROTATE_PIVOT = /^(centroid|com|cell|origin|atom:[1-9][0-9]{0,6})$/;
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const listeners = new Set();
let values = { ...DEFAULTS };            // the user's own layer (localStorage, gui.json)
/* the folder layer: {file, target, can_save, why_not, gui: {key: value}, keys: {key: file key}, cli, warnings} */
let folder = { file: null, gui: {}, keys: {}, cli: {}, warnings: [], can_save: false, why_not: "" };
let session = {};                        // changes made here to keys the folder sets, not saved yet
let sessionKey = "";
const masked = (k) => k in session || k in folder.gui;
const emit = (k) => listeners.forEach((fn) => fn(k, get(k)));
/* the user's layer changed: listeners hear it only when it is the effective value */
const emitUser = (k) => { if (!masked(k)) emit(k); };
let saveTimer = 0;
let api = null;

export function migrate(obj) {
  // version 1 is the first schema: keep only known keys of the right type
  const out = {};
  for (const [k, v] of Object.entries(obj || {})) {
    if (k in DEFAULTS && typeof v === typeof DEFAULTS[k]) out[k] = v;
  }
  // fixed choices (settings.py _CHOICES): an unknown value (a hand-edited file, a newer version's) is dropped
  for (const [k, ok] of Object.entries(CHOICES)) if (k in out && !ok.includes(out[k])) delete out[k];
  for (const k of ["plots.spinUp", "plots.spinDown", "plots.bandColor"]) if (k in out && out[k] !== "" && !/^#[0-9a-f]{6}$/i.test(out[k])) delete out[k];
  if ("polyCenters" in out && !POLY_CENTERS.test(out.polyCenters)) delete out.polyCenters;
  if ("viewport.rotatePivot" in out && !ROTATE_PIVOT.test(String(out["viewport.rotatePivot"]))) delete out["viewport.rotatePivot"];
  if ("polyOpacity" in out && !(out.polyOpacity >= 0.1 && out.polyOpacity <= 1)) delete out.polyOpacity;
  if ("bondTolerance" in out && !(out.bondTolerance >= 0.8 && out.bondTolerance <= 1.6)) delete out.bondTolerance;
  if ("bondMin" in out && !(out.bondMin >= 0 && out.bondMin <= 3)) delete out.bondMin;
  if ("bondPairs" in out && !bondPairsOk(out.bondPairs)) delete out.bondPairs;
  // the 1.1.6 preview's Publication theme is Light with Plot style Publication (as settings.py and boot.js read it)
  if (out.theme === "publication") {
    out.theme = "light";
    if (!PRESET_IDS.includes(out["plots.style"])) out["plots.style"] = "publication";
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
  // the Publication theme, once: stored again as Light + Plot style Publication; a theme the server
  // has not heard of yet takes the plot style along, so the server's copy cannot reset it
  if (stored.theme === "publication") {
    try {
      if (pendingKeys().includes("theme")) setPending([...pendingKeys(), "plots.style"]);
      persistLocal();
    } catch (e) { /* private window */ }
  }
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
    changed.forEach(emitUser);
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

/** Take another page's values (the right editor group follows its window): the
 * listeners run; nothing is saved, and the pending list and the server's copy
 * as this page knows it stay as they are (the window saves its own changes). */
export function adopt(given) {
  const want = { ...DEFAULTS, ...migrate(given) };
  const changed = Object.keys(DEFAULTS).filter((k) => !same(values[k], want[k]));
  for (const k of changed) values[k] = want[k];
  changed.forEach((k) => listeners.forEach((fn) => fn(k, get(k))));
}
/** Whether k was changed in this browser and has not reached the server yet. */
export const isPending = (k) => pendingKeys().includes(k);
export function get(k) { return k in session ? session[k] : k in folder.gui ? folder.gui[k] : values[k]; }
export const all = () => ({ ...values, ...folder.gui, ...session });
/* the user's own non-default values (what gui.json holds and Export writes) */
export const nonDefault = () => Object.fromEntries(Object.entries(values).filter(([k, v]) => !same(v, DEFAULTS[k])));
export const isDefault = (k) => same(values[k], DEFAULTS[k]);
export function onChange(fn) { listeners.add(fn); return () => listeners.delete(fn); }

/** A change: to the user's own value, or, for a key the folder sets, to this session's. */
export function set(k, v) {
  if (!(k in DEFAULTS)) throw new Error(`unknown setting ${k}`);
  if (masked(k)) {
    if (same(get(k), v)) return;
    if (k in folder.gui && same(folder.gui[k], v)) delete session[k]; else session[k] = v;
    persistSession();
    emit(k);
    return;
  }
  setUser(k, v);
}
function setUser(k, v) {
  if (same(values[k], v)) return;
  values[k] = v;
  persistLocal();
  setPending([...pendingKeys(), k]);
  emitUser(k);
  clearTimeout(saveTimer);
  saveTimer = setTimeout(flush, 500);
}
export const reset = (k) => setUser(k, DEFAULTS[k]);
/** The user's own value (gui.json), even where the folder's file or this session sets the key. */
export function setOwn(k, v) {
  if (!(k in DEFAULTS)) throw new Error(`unknown setting ${k}`);
  setUser(k, v);
}
/** Every setting of the user's back to its default (Settings › Advanced › Reset all). */
export function resetAll() { Object.keys(DEFAULTS).forEach((k) => setUser(k, DEFAULTS[k])); }

/* ---------------------------------------------------------------- the folder layer (mace-folder.toml) */
function persistSession() {
  try {
    if (!sessionKey) return;
    if (Object.keys(session).length) sessionStorage.setItem(sessionKey, JSON.stringify(session));
    else sessionStorage.removeItem(sessionKey);
  } catch (e) { /* private */ }
}
/** The served folder's layer (boot's `folder`, or a save's answer); dirHash keys this tab's session changes. */
export function setFolder(view, dirHash) {
  const before = Object.fromEntries(Object.keys(DEFAULTS).map((k) => [k, get(k)]));
  folder = { file: null, gui: {}, keys: {}, choices: {}, cli: {}, warnings: [], can_save: false, why_not: "", ...(view || {}) };
  // only known keys of the right type (a newer server's keys wait for a newer page)
  folder.gui = Object.fromEntries(Object.entries(folder.gui || {}).filter(([k, v]) => k in DEFAULTS && typeof v === typeof DEFAULTS[k]));
  if (dirHash !== undefined) {
    sessionKey = dirHash ? `mace.gui.settings.session.${dirHash}` : "";
    try { session = (sessionKey && JSON.parse(sessionStorage.getItem(sessionKey) || "{}")) || {}; } catch (e) { session = {}; }
  }
  // a session change outlives only a key the folder still sets
  for (const k of Object.keys(session)) if (!(k in DEFAULTS) || !(k in folder.gui) || same(session[k], folder.gui[k])) delete session[k];
  persistSession();
  Object.keys(DEFAULTS).forEach((k) => { if (!same(before[k], get(k))) emit(k); });
}
export const folderInfo = () => JSON.parse(JSON.stringify(folder));
/** Can this key live in a folder file? */
export const folderable = (k) => k in (folder.keys || {});
/** Can the folder file hold this value of k? (a choice key names its choices: a palette of the user's own cannot) */
export const folderAccepts = (k, v = get(k)) => folderable(k) && (!(folder.choices || {})[k] || folder.choices[k].includes(v));
/** Where the effective value comes from: "session", "folder", "user" or "default". */
export function source(k) {
  if (k in session) return "session";
  if (k in folder.gui) return "folder";
  return same(values[k], DEFAULTS[k]) ? "default" : "user";
}
/** The user's own value (what this folder's file hides). */
export const userValue = (k) => values[k];
async function folderCall(body) {
  if (!api) throw new Error("not started");
  const r = await api("folder-settings", { body });
  setFolder(r);
  return r;
}
/** Save to this folder: the value in effect goes into mace-folder.toml. */
export async function saveToFolder(k) {
  const v = get(k);
  await folderCall({ set: { [k]: v } });
  if (k in session) { delete session[k]; persistSession(); }
  emit(k);
  return v;
}
/** Save as my default: the value in effect becomes the user's own (the folder's file, if it sets the key, still wins here). */
export function saveAsDefault(k) {
  const v = get(k);
  if (k in session) { delete session[k]; persistSession(); }
  setUser(k, v);
  emit(k);
  return { stillFolder: k in folder.gui && !same(folder.gui[k], v) };
}
/** Reset to my default: drop this session's change and the folder's value (from its file). */
export async function resetToMine(k) {
  if (k in session) { delete session[k]; persistSession(); }
  if (k in folder.gui) await folderCall({ unset: [k] });
  emit(k);
}
/** Several values at once (an imported settings file); unknown keys and wrong
 * types are skipped and returned. */
export function setMany(obj) {
  const skipped = [];
  for (const [k, v] of Object.entries(obj || {})) {
    if (!(k in DEFAULTS) || typeof v !== typeof DEFAULTS[k] || (Array.isArray(DEFAULTS[k]) && !Array.isArray(v))) { skipped.push(k); continue; }
    setUser(k, v);
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
    if (!same(values[k], back)) { values[k] = back; emitUser(k); }
  }
  if (bad.length) {
    persistLocal();
    notify(bad.map((k) => `Couldn't save ${label(k)}: ${rejected[k]}`).join(" · "), "warning");
  }
  return r;
}
