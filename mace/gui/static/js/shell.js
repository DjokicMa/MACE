/* MACE GUI app shell: documents, layout, toolbar, inspector, bottom panel,
 * status bar, keyboard, connection. The classic views (app.js, editor.js,
 * results.js) still draw the structures, plots and tables; the shell decides
 * which one shows, and where its controls live. */
import { registry, isSingleKey, evalWhen, disabledReason, shortcutOf, formatChord, isMac, fuzzyScore, parseChord,
  lookup, applyKeymap, conflicts, allConflicts, chordId, chordFromEvent, isReserved, riskOf, browserNavOf, BROWSER_NAV,
  RESERVED, RISKY, pseudo, command, loadLayoutMap, whensOverlap } from "./commands.js";
import * as commands from "./commands.js";
import { $, $$, esc, icon, toast, dialog, installTooltips, hideTip, asset, configureToasts } from "./ui.js";
import * as notifications from "./notifications.js";
import * as settings from "./settings.js";
import * as plotPresets from "./plotpresets.js";
// the plot presets for the classic views (app.js plotPreset, styleFigure): installed before any plot draws
window.MACE_PRESETS = plotPresets;
import { initMenubar, openMenuIndex, focusBar, closeAll as closeMenus, isOpen as menuOpen, openAt } from "./menus.js";
import { initPalette, openPalette, openPick, isPaletteOpen, closePalette, results as paletteResults, bench as paletteBench, domRows } from "./palette.js";
import { THEMES, themeOf, groupLabel } from "./themes.js";
import * as rank from "./rank.js";
import { registerAll, registerMnemonics, MENUS, TOOLBAR, CONTEXT, ROTATION_MENU } from "./actions.js";
import { initContext, openForFocus, contextAt, contextFor, openViewportMenu } from "./context.js";
import { abbrev, chars, middle } from "./paths.js";
import * as docprefs from "./docprefs.js";
import * as profile from "./profile.js";
import * as DS from "./viewport/display_state.js";
import * as VAC from "./viewport/vacuum.js";
import * as MQ from "./viewport/mesh_quality.js";
import { rotationPrefs, dragOrientation, turnOrientation, turntableUp, nearestAxisView } from "./viewport/camera.js";
import { PANE, createGroups } from "./groups.js";
import { bindSplitter } from "./splitter.js";
import { initSidebar } from "./sidebar.js";
import { createPanelContext } from "./panelctx.js";
import * as IT from "./inspector_tiers.js";
import * as AP from "./activity_prefs.js";
import * as SM from "./setup/model.js";

const M = () => window.MACE;
/* The viewport and the editor (js/editor, js/viewport, js/geom) load when the
 * first structure document opens, never before first paint (spec §8.5). */
let E = null, editorLoading = null;
function loadEditor() {
  if (E) return Promise.resolve(E);
  if (!editorLoading) {
    editorLoading = import("./editor/main.js").then((m) => {
      E = m.init(S);
      // the classic panel (editor.js, a plain script) acts on the active document through this
      window.__maceEd = { active: () => edOf(active), api: E.api, editing: E.editing, redraw: () => E.redraw() };
      window.__maceDrafts = E.drafts;
      M().editorTable = () => !!active && active.kind === "structure" && !!edOf(active);
      M().sceneDraw = (s, keepView) => {
        const d = docs.find((x) => x.id === panelOwner.structure) || active;
        E.draw(d, s, keepView);
      };
      return E;
    }).catch((e) => { editorLoading = null; M().sceneLoading = null; throw e; });
    M().sceneLoading = editorLoading;
  }
  return editorLoading;
}
const edOf = (d) => (E && d ? E.edOf(d) : null);
const KIND_PANEL = { structure: "structure", trajectory: "opt", plot: "results", runplot: "results", mech: "results", db: "db", editor: "edit", text: "text", freq: "freq",
  cluster: "cluster", job: "job", volume: "volume", batch: "batch", gallery: "gallery", workflows: "workflows" };
const KIND_LABEL = { cif: "CIF", d12: "d12", d3: "d3", out: "out", band: "BAND", dos: "DOS", ir: "IR", raman: "Ram", db: "db", json: "json",
  image: "img", html: "html", freq: "FREQ", cube: "cube", orbitals: "orb", xyz: "xyz", poscar: "vasp", mol: "mol", pdb: "pdb", gen: "gen", fort34: "f34",
  f25: "f25", transport: "trn", crystaldata: "data", wavefunction: "wf", script: "sh", log: "log", joblog: "log", volume: "vol", props: "PROPS", convergence: "conv" };
const docs = [];
let active = null, nextId = 1, tool = "select", symMode = "sym", setupOpen = false, info = null;
const panelOwner = {};
let GR = null;                      // the two editor groups (groups.js)
const layoutKey = "mace.gui.layout.v1";

/* ================================================================ context */
function ctxHere() {
  const d = active, ae = document.activeElement;
  const ed = window.MaceEditor, eds = edOf(d);
  const files = M() ? M().state.files : [];
  const control = !!ae && ae !== document.body && !!ae.closest && !!ae.closest("button, a[href], [role=button], [role=menuitem], [role=tab], [role=option], summary, select, input, textarea");
  return {
    tool, moveNudge: tool === "move" && !!ae && !!ae.closest && !!ae.closest(".viewer-wrap"), controlFocus: control,
    hasDoc: !!d, docCount: docs.length, docKind: d ? d.kind : "", docFileKind: d ? d.fileKind || "" : "",
    // a single-point output (one geometry, no optimization): no trajectory, cell or measurement along a run
    docSinglePoint: !!(d && d.fileKind === "out" && d.data && d.data.is_opt === false),   // editStep documents have no data.summary
    docEditable: !!eds && !eds.editBlock(),
    doc3d: !!eds && !!eds.structure && !!eds.structure.pbc && eds.structure.pbc.every(Boolean),
    canUndo: !!eds && eds.canUndo && !eds.lost && !eds.previewing,
    canRedo: !!eds && eds.canRedo && !eds.lost && !eds.previewing,
    previewing: !!(eds && eds.previewing),
    textSelected: !!(window.getSelection && String(window.getSelection()).length),
    dragging: !!(E && E.pointer() && E.pointer().dragging),
    hasSelection: selection().n > 0, selectionCount: selection().n,
    structureDoc: !!d && d.kind === "structure",
    anyDirty: docs.some(isDirty), docDirty: isDirty(d), docHasPath: !!d && !!d.path && !d.untitled && !(d.frame >= 0),
    hasSets: !!eds && Object.keys(eds.selectionSets).length > 0,
    hasMeasurements: !!eds && eds.measurements.length > 0,
    inputFocus: !!ae && (ae.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(ae.tagName)),
    dialogOpen: !!document.querySelector("dialog:modal"),   // a non-modal panel (Edit Cell) leaves the keys on
    viewportFocus: !!ae && !!ae.closest && !!ae.closest(".viewer-wrap, .fv-view"),
    freqOffGamma: !!(d && d.kind === "freq" && d.gamma === false),
    // a volume of a molecule: no cell, so no periodic images, no cell axes to look along
    volNonPeriodic: !!(d && d.kind === "volume" && d.meta && !d.meta.periodic),
    setupOpen, setupHasDeck: !!(setupMod && setupMod.hasDeck()), setupFocus: !!ae && !!ae.closest && !!ae.closest("#inspector .setup"),
    isMac, symMode: eds ? eds.symMode === "sym" : symMode === "sym", toolNotSelect: tool !== "select",
    openFolderOn: !!(info && info.open_folder),
    hasDb: files.some((f) => f.kind === "db"), hasBand: files.some((f) => f.kind === "band"),
    hasDos: files.some((f) => f.kind === "dos"), hasIr: files.some((f) => f.kind === "ir"), hasRaman: files.some((f) => f.kind === "raman"), hasCube: files.filter((f) => f.kind === "cube").length > 0,
    hasTwoCubes: files.filter((f) => f.kind === "cube").length > 1,
    docHasIr: !!partnerOf(d, "ir"), docHasRaman: !!partnerOf(d, "raman"), docIsFreq: isFreqDoc(d),
    trajSteps: trajSteps(), activeTable: !!activeTableKind(), activeDbTable: activeTableKind() === "db", zenMode: !!layout.zen,
    noWebgl: !!(M() && M().webgl && !M().webgl.ok),
  };
}
/* the context the commands see: the focused editor group's (groups.js) */
function ctx() { const c = ctxHere(); return GR ? GR.ctxOverride(c) : c; }

/** Run a command (with optional arguments, e.g. the atom a context menu was
 * opened on). A disabled command says why in a toast instead. */
let lastCommand = null;          // the visible-keys test reads it (plan T16)
// Shift+F10 opens the app's menu; Firefox would open its own too (plan T16): not right after ours
addEventListener("contextmenu", (e) => {
  if (lastCommand && lastCommand.id === "view.contextMenu" && performance.now() - lastCommand.t < 600) {
    e.preventDefault(); e.stopPropagation();
  }
}, true);
let runSpy = null;               // the menu-wiring test records ids instead of running (window.__mace.spyRun)
function run(id, args = null) {
  const c = registry.get(id) || pseudo.get(id);
  if (!c) throw new Error(`no command ${id}`);
  if (runSpy) { runSpy(id, args); return true; }
  // two editor groups: a document command goes to the focused group, a window command to the window
  const fwd = GR && (GR.shouldForward(id, c.category) || GR.forwardGlobal(id, c.category));
  if (fwd) return fwd.run(id, args);
  const cx = { ...ctx(), args };
  const why = disabledReason(c, cx);
  if (why) { toast(`${c.title}: ${why}`, "info"); return false; }
  performance.mark(`ui:${id}:start`);
  lastCommand = { id, t: performance.now(), chord: args && args.chord ? args.chord : "" };
  try { const r = c.run(cx, args); if (r && r.catch) r.catch((e) => toast(e.message, "error")); }
  catch (e) { toast(e.message, "error"); }
  performance.mark(`ui:${id}:end`);
  refresh();
  return true;
}

/* ================================================================ documents */
function fileKind(path) { const f = M().fileByPath(path); return f ? f.kind : ""; }
// the kinds that open as a structure (app.js openFile): the editor loads with them, so the first one opened is editable
const EDITOR_KINDS = ["cif", "d12", "out", "xyz", "poscar", "mol", "pdb", "gen", "fort34"];

/* What a file opens as (plan E1): by its kind, and for outputs and logs by what their content says. */
const INFO_KINDS = ["wavefunction", "crystaldata", "f25"];
async function openDoc(path, opts = {}) {
  const as = opts.as || "";
  const existing = docs.find((d) => d.path === path && !(d.frame >= 0) &&      // not a step opened with Edit This Step
    (as !== "structure" || d.kind === "structure" || d.kind === "trajectory"));
  if (existing) return activate(existing);
  let fk = fileKind(path);
  if (fk === "json" || fk === "d3" || fk === "script") return openAsText(path);       // settings files, d3 decks, job scripts
  // a fort.25 of BAND or DOSS blocks plots like the BAND.DAT / DOSS.DAT of its run
  if (fk === "f25") { const f25 = M().fileByPath(path); if (f25 && (f25.sub === "BAND" || f25.sub === "DOSS")) fk = f25.sub === "BAND" ? "band" : "dos"; }
  if (fk === "f25") {
    // a fort.25 that holds a 2D map (ECHG, POTM) opens as a contour plot (plan E6); BAND and DOSS ones say what they are
    let inf = null;
    try { inf = await M().api("fileinfo?" + M().q({ path })); } catch (e) { inf = null; }
    if (inf && inf.f25_kind === "MAPN") return openMapPlot(path);
  }
  if (INFO_KINDS.includes(fk)) return openInfo(path);
  if (fk === "cube") return openVolume(path);
  if (fk === "orbitals") return openOrbitals(path);
  if ((fk === "out" || fk === "log") && as !== "structure") {
    let r = null;
    try { r = await M().roleOf(path); } catch (e) { r = null; }
    if (r && r.role === "joblog") return openJobLog(path);
    if (r && r.role === "text") return openAsText(path);
    if (r && r.role === "properties_out") return openProps(path);
    if (r && r.role === "crystal_out" && r.nogeom) return openNoGeometry(path);
    if (r && r.role === "crystal_out" && (r.kind === "freq" || r.freq) && settings.get("files.openFreqAs") !== "structure")
      return openModes(path);
    // 1.1.6: an ELASTIC, EOS, QHA or CPHF run opens as its results, as a transport file opens as its plots
    if (r && r.role === "crystal_out" && MECH_CALCS.includes(r.calc)) return openMech(path, r.calc);
  }
  saveView(active);
  performance.mark("open:start");
  const busy = setTimeout(() => setTask("Opening " + path.split("/").pop() + "…"), 300);
  try {
    const [r] = await Promise.all([M().openFile(path), EDITOR_KINDS.includes(fk) ? loadEditor() : null]);
    const d = { id: nextId++, path, kind: r.kind, fileKind: fk, title: path.split("/").pop(), data: r.data || null };
    docs.push(d);
    if (d.kind === "structure" && E) E.attach(d, M().state.structure);
    active = d;
    panelOwner[KIND_PANEL[d.kind]] = d.id;
    if (d.kind === "trajectory") { panelOwner.structure = d.id; showBottom("timeline"); }
    else if (d.kind === "structure") showBottom("atoms");
    else if (d.kind === "plot") showPlot(d);
    if (fk === "out") feedRunProblems(d);
    pushRecent(path);
    afterSwitch();
    performance.mark("open:rendered");
    performance.measure("open", "open:start", "open:rendered");
    return d;
  } catch (e) {
    // a log or an output with no geometry, found only now (the details were not in yet): never an error toast
    if (e.status === 409 && /job log/i.test(e.message)) return openJobLog(path);
    if (e.status === 409 && /No geometry in this output/.test(e.message)) return openNoGeometry(path);
    // a SLAB/POLYMER/MOLECULE deck the viewer does not draw: the output of its run, next to it, opens instead
    if (e.status === 400 && e.data && e.data.open && e.data.open !== path) {
      toast(`${e.message}; opening it instead.`, "info");
      return openDoc(e.data.open);
    }
    // the server's words often name the file already ("X.d12 reads its geometry from ..."): once is enough
    const fname = path.split("/").pop();
    toast(String(e.message).startsWith(fname) ? e.message : `${fname}: ${e.message}`, "error");
    return null;
  } finally { clearTimeout(busy); setTask(""); }
}

/* Each structure document keeps its own camera (spec §1.2): saved when it
 * loses the viewport, put back when it gets it again. */
function saveView(d) {
  const v = M().state.viewers["viewer-main"];
  if (d && d.kind === "structure" && v && panelOwner.structure === d.id) d.view = v.getView();
}
async function activate(d) {
  if (!d) return;
  performance.mark("ui:tab:start");
  if (active !== d) saveView(active);
  if (active && active !== d && active.windowed && textView) textView.close();
  if (active && active !== d && active.kind === "freq" && freqView) freqView.close();
  if (active && active !== d && active.kind === "volume" && volumeView) volumeView.close();
  active = d;
  if (d.kind === "volume") {
    M().showTab("volume");
    panelOwner.volume = d.id;
    if (d.path) { M().state.active = d.path; M().renderFiles(); }
    volumeViewer().then((v) => { if (active === d) return v.open(d); }).catch((e) => toast(`${d.title}: ${e.message}`, "error"));
    afterSwitch();
    return;
  }
  const panel = KIND_PANEL[d.kind];
  const ed = edOf(d);
  if (ed && panelOwner[panel] !== d.id) {
    // a structure document shows its own structure, edited or not: no request
    panelOwner[panel] = d.id;
    if (d.path) { M().state.active = d.path; M().renderFiles(); }
    await M().loadStructure(d.path, ed.structure);
    M().showTab(panel);
    if (d.view) {
      const view = d.view;
      requestAnimationFrame(() => { const v = M().state.viewers["viewer-main"]; if (v && active === d) { v.setView(view); v.render(); updateTriad(); } });
    }
  } else if (d.kind === "plot") {
    // before the file branch: a plot document with a file (band, DOS, spectra, transport) is drawn by the
    // plot view from its own data; reopening its file there left the previous plot on screen
    M().showTab(panel);
    panelOwner[panel] = d.id;
    if (d.path) { M().state.active = d.path; M().renderFiles(); }
    showPlot(d);
  } else if (d.kind === "mech") {
    // before the file branch too: the results document reads its output itself (js/docs/mech.js)
    M().showTab(panel);
    panelOwner[panel] = d.id;
    if (d.path) { M().state.active = d.path; M().renderFiles(); }
    showMech(d);
  } else if (d.path && panelOwner[panel] !== d.id) {
    await M().openFile(d.path, d.data);       // cached payload: no request for structures
    panelOwner[panel] = d.id;
    if (d.kind === "trajectory") panelOwner.structure = d.id;
    if (d.kind === "structure" && d.view) {
      const view = d.view;                     // after the redraw and fit of this frame
      requestAnimationFrame(() => { const v = M().state.viewers["viewer-main"]; if (v && active === d) { v.setView(view); v.render(); updateTriad(); } });
    }
  } else if (d.kind === "freq") {
    M().showTab(panel);
    panelOwner[panel] = d.id;
    freqViewer().then((v) => { if (active === d) v.open(d); });
  } else if (d.kind === "runplot") {
    M().showTab(panel);
    panelOwner[panel] = d.id;
    showRunPlot(d);
  } else if (d.kind === "gallery") {
    M().showTab(panel);
    panelOwner[panel] = d.id;
    galleryViewer().then((v) => { if (active === d) return v.open(d); }).catch((e) => toast(`Plot Gallery: ${e.message}`, "error"));
  } else if (d.kind === "cluster" || d.kind === "job") {
    M().showTab(panel);
    panelOwner[panel] = d.id;
    (d.kind === "cluster" ? clusterViewer() : jobViewer()).then((v) => { if (active === d) v.open(d); });
  } else if (d.kind === "workflows") {
    M().showTab(panel);
    panelOwner[panel] = d.id;
    workflowViewer().then((v) => { if (active === d) v.open(d); }).catch((e) => toast(`Workflows: ${e.message}`, "error"));
  } else {
    M().showTab(panel);
    if (d.kind === "text") renderTextDoc(d);
    if (d.kind === "batch") batchPanel().then((p) => { if (active === d) p.show(d); }).catch((e) => toast(e.message, "error"));
    if (d.path) { M().state.active = d.path; M().renderFiles(); }
  }
  afterSwitch();
  // the end is the frame after the switch's redraw: viewers paint one frame late
  requestAnimationFrame(() => requestAnimationFrame(() => {
    performance.mark("ui:tab:end"); performance.measure("ui:tab", "ui:tab:start", "ui:tab:end"); }));
}

let lastActive = null;
function afterSwitch() {
  syncBoundaryBox();
  syncStyleBox();
  showPanelFor(active);
  // the bottom panel follows the document: atoms for a structure, the
  // timeline for a run, tasks for a plot, nothing for the database
  const k = active ? active.kind : null;
  // (the Jobs tab is about the folder, not a document: it stays with no document open; so does Analysis, for its batches)
  shell.classList.toggle("doc-no-bottom", (k === "db" || k === "workflows" || !k) && layout.bottomTab !== "jobs" && layout.bottomTab !== "analysis");
  const atomsPane = $(".bottom-pane.atoms");
  if (atomsPane) atomsPane.classList.toggle("no-structure", !(k === "structure" || k === "trajectory"));
  // a plot borrows Atoms and Coordination from its run (panelctx.js; a placeholder says when it has none): they stay
  if (k === "plot" && ["timeline", "measurements"].includes(layout.bottomTab)) showBottom("tasks", false);
  if ((k === "gallery" || k === "cluster" || k === "job" || k === "workflows") && ["atoms", "timeline", "coordination", "measurements"].includes(layout.bottomTab)) showBottom("tasks", false);
  if (k === "text" && ["atoms", "timeline", "measurements", "coordination"].includes(layout.bottomTab)) showBottom(active && active.deck ? "problems" : "tasks", false);
  if (k === "structure" && layout.bottomTab === "timeline") showBottom("atoms", false);
  if (k === "batch" && ["atoms", "timeline", "measurements", "coordination"].includes(layout.bottomTab)) showBottom("tasks", false);
  // the Rotate tool acts on a structure document only (actions.js): elsewhere it goes back to Select
  if (tool === "rotate" && k !== "structure") { tool = "select"; toolChip(); }
  if (k === "freq") showBottom("modes", false);
  else if (layout.bottomTab === "modes") showBottom(k === "structure" ? "atoms" : k === "trajectory" ? "timeline" : "tasks", false);
  requestAnimationFrame(revealBottomTab);
  renderTabs(); renderStatus(); renderToolbarState(); renderOpenEditors(); updateTitle();
  if (setupOpen && setupMod && $("#inspector .setup")) setupMod.docChanged(); else renderInspector();
  renderPanels();
  attachTriad();
  if (lastActive !== active) {
    const from = lastActive;
    lastActive = active;
    document.dispatchEvent(new CustomEvent("mace-doc-switch", { detail: { from, to: active } }));
  }
  if (GR && GR.pane) GR.changed();
}

function showPanelFor(d) {
  if (!d) { if (info) renderWelcome(); M().showTab("welcome"); return; }
  const panel = KIND_PANEL[d.kind];
  if (!$(`#tab-${panel}`).classList.contains("active")) M().showTab(panel);
}

function closeDoc(d, force = false) {
  if (!d) return;
  if (!force && isDirty(d) && !confirm(`${d.title} has unsaved changes. Close it anyway?`)) return;
  const e = edOf(d);
  if (d.builderFrame) d.builderFrame.discard("tab-close");     // a preview is not unsaved work: no question
  if (e) {
    // closing drops the changes: the draft goes, and the server lets go of the session
    e.closed = true;
    if (E.drafts) E.drafts.clear(e);
    if (e.srv.session && !e.lost) M().api("edit/close", { body: { session: e.srv.session } }).catch(() => {});
  }
  if (d.kind === "trajectory" && M().isLive && M().isLive(d.path)) M().stopLive();     // a closed run is not polled
  clearProblems(d.id);                                                                  // its problems go with it
  const i = docs.indexOf(d);
  docs.splice(i, 1);
  if (problems.some((p) => p.doc === d.id)) clearProblems(d.id);
  if (d.held) releaseHeld(d);
  if (d.kind === "db" && dbView) dbView.forget(d.path);
  // a band plot's Brillouin zone view, and Setup's state for this document (its zone view too)
  for (const o of [d, d.plot]) if (o && o.bz && o.bz.destroy) { o.bz.destroy(); o.bz = null; }
  if (setupMod && setupMod.forget) setupMod.forget(d);
  if (d.kind === "job") { if (jobView) jobView.close(d); M().api("cluster/job/close", { body: { id: d.jobId } }).catch(() => {}); }
  if (d.kind === "cluster" && clusterView) clusterView.close();
  if (d.kind === "workflows" && workflowView) workflowView.close();
  if (d.kind === "mech" && mechView) mechView.close(d);                 // the results view lets go of a closed document
  Object.keys(panelOwner).forEach((k) => { if (panelOwner[k] === d.id) delete panelOwner[k]; });
  if (active === d) {
    active = null;
    const next = docs[Math.min(i, docs.length - 1)];
    if (next) return activate(next);
    if (M().state) { M().state.active = null; M().renderFiles(); }
  }
  afterSwitch();
  if (GR && !docs.length) GR.docClosed(0);      // the group's last tab: groups.js closes or merges the group
}
/* A tab to another place in its own strip (dragged there, or Move Left/Right):
 * `to` counts the other tabs (null: the end). True when it moved. */
function moveDocTo(d, to) {
  const from = docs.indexOf(d);
  if (from < 0) return false;
  docs.splice(from, 1);
  const at = to == null ? docs.length : Math.max(0, Math.min(docs.length, Math.round(Number(to)) || 0));
  docs.splice(at, 0, d);
  renderTabs(); renderOpenEditors();
  if (GR && GR.pane) GR.changed();
  return at !== from;
}
function moveDocBy(d, step) {
  if (!d || docs.length < 2) return false;
  const i = docs.indexOf(d) + step;
  if (i < 0 || i >= docs.length) return false;
  return moveDocTo(d, i);
}
function closeAll() { [...docs].forEach((d) => closeDoc(d)); }
function cycleDoc(step) {
  if (docs.length < 2) return;
  activate(docs[(docs.indexOf(active) + step + docs.length) % docs.length]);
}
const isDirty = (d) => (edOf(d) ? edOf(d).dirty : false);
const docPath = (d) => (d ? d.path || d.srcPath || null : null);

/* The classic panel's Clean "Before / after": a comparison document. */
document.addEventListener("mace-clean-compare", () => {
  saveView(active);
  const d = { id: nextId++, path: null, kind: "trajectory", title: "Clean: before and after" };
  docs.push(d); active = d; panelOwner.opt = d.id;
  showBottom("timeline");
  afterSwitch();
});
document.addEventListener("mace-open", (e) => (GR && GR.sideFocused() && !GR.pane ? GR.openToSide(e.detail) : openDoc(e.detail)));
document.addEventListener("mace-overlay", () => closeMenus());
addEventListener("resize", () => {
  if (narrow() === wasNarrow) return;
  wasNarrow = narrow(); drawerOpen = false; applyLayout(); renderActivity();
});
$("#center") && $("#center").addEventListener("pointerdown", () => {
  if (narrow() && drawerOpen && !setupOpen) { drawerOpen = false; applyLayout(); }
});
document.addEventListener("mace-theme", () => updateTriad());   // the OS switched light/dark
document.addEventListener("mace-edit", () => { renderOpenEditors(); refresh(); });

/* ================================================================ tabs */
/* Called on every refresh: the strip is rebuilt only when it changed, so a
 * tab keeps its focus and hover, and nothing is churned per command. */
let tabsHtml = null;
function renderTabs() {
  const host = $("#doc-tabs");
  const html = docs.map((d) => `<div class="doc-tab${d === active ? " active" : ""}${isDirty(d) ? " dirty" : ""}" role="tab"
    tabindex="${d === active ? 0 : -1}" aria-selected="${d === active}" data-doc="${d.id}" title="${esc(tabTip(d))}">
    <span class="kb ${esc(d.fileKind || (d.kind === "editor" ? "cif" : "out"))}">${esc(KIND_LABEL[d.fileKind] || (d.kind === "editor" ? "edit" : "view"))}</span>
    <span class="t">${esc(d.title)}</span>${d.kind === "trajectory" ? '<span class="dot" title="Has optimization steps"></span>' : ""}
    <span class="x" data-close="${d.id}" role="button" aria-label="Close ${esc(d.title)}">${icon("x")}<span class="dot"></span></span></div>`).join("");
  if (html === tabsHtml && host.childElementCount === docs.length) return;
  tabsHtml = html;
  host.innerHTML = html;
  // a tab list with no tab is not a tab list (WCAG 4.1.2): the role comes with the first document
  if (docs.length) host.setAttribute("role", "tablist"); else host.removeAttribute("role");
  labelTabs();
}
/* With two editor groups each strip says whose it is (groups.js). */
function labelTabs() {
  const host = $("#doc-tabs");
  const name = PANE ? "Right editor group: open documents" : GR && GR.isOpen() ? "Left editor group: open documents" : "Open documents";
  if (host.getAttribute("aria-label") !== name) host.setAttribute("aria-label", name);
}
let tabDragged = false;
$("#doc-tabs").addEventListener("pointerdown", (e) => {
  const t = e.target.closest(".doc-tab");
  if (!t || e.target.closest("[data-close]") || !GR) return;
  GR.dragTab(e, docs.find((y) => y.id === Number(t.dataset.doc)), () => { tabDragged = true; setTimeout(() => { tabDragged = false; }, 0); });
});
// the tabs move by the pointer (groups.js): no native drag of a tab's text or icon
$("#doc-tabs").addEventListener("dragstart", (e) => e.preventDefault());
$("#doc-tabs").addEventListener("click", (e) => {
  if (tabDragged) { tabDragged = false; return; }
  const x = e.target.closest("[data-close]");
  const t = e.target.closest(".doc-tab");
  if (!t) return;
  const d = docs.find((y) => y.id === Number(t.dataset.doc));
  if (x) closeDoc(d); else if (d !== active) activate(d);
});
$("#doc-tabs").addEventListener("auxclick", (e) => {
  const t = e.target.closest(".doc-tab");
  if (t && e.button === 1) closeDoc(docs.find((y) => y.id === Number(t.dataset.doc)));
});
$("#doc-tabs").addEventListener("keydown", (e) => {
  if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
  e.preventDefault();
  const step = e.key === "ArrowRight" ? 1 : -1;
  // Shift+Arrow moves the tab itself (Move Left / Move Right); the arrows alone go to the next tab
  if (e.shiftKey && !e.ctrlKey && !e.metaKey && !e.altKey) moveDocBy(active, step); else cycleDoc(step);
  $(".doc-tab.active")?.focus();
});
$("#doc-tabs").addEventListener("contextmenu", (e) => {
  const t = e.target.closest(".doc-tab");
  if (!t) return;
  e.preventDefault();
  openAt(CONTEXT.tab, e.clientX, e.clientY, "Tab menu", { docId: Number(t.dataset.doc) });
});

/* The served folder, shortened to fit (carry-over b): "~" for the home
 * folder, "/…/" in the middle; the full path is the tooltip. */
const rootText = (max) => (info ? abbrev(info.root, { home: info.home || "", aliases: info.aliases || null, max }) : "");
/* The served folder's own name (the browser tab's title has no tooltip: VS Code's "<file> — <folder>"). */
const folderName = () => (info ? middle(String(info.root).replace(/\/+$/, "").split("/").pop() || String(info.root), 24) : "");
/* A tab's tooltip: its file, or for a built or dropped document what it came from and that it is not saved yet. */
function tabTip(d) {
  const p = d.meta && d.meta.provenance && d.meta.provenance.label;
  const parts = [docPath(d) || d.title];
  if (p) parts.push(p);
  if (d.untitled && !d.path) parts.push(d.dropped ? "Not saved in the folder yet" : "Not saved yet");
  return parts.join(" · ");
}
/* The browser tab (no tooltip) names the document and the folder's own name;
 * the menu bar shows the folder's path, as short as the Explorer's header
 * makes it (TITLE_ROOT_MAX), with the full path in its tooltip. */
const TITLE_ROOT_MAX = 28;
function updateTitle() {
  const name = active ? active.title : "Welcome";
  const dirty = docs.some(isDirty);
  const folder = folderName();
  document.title = active ? `${dirty ? "● " : ""}${name}${folder ? " — " + folder : ""} — MACE` : `${folder ? folder + " — " : ""}MACE`;
  const t = $("#win-title");
  // as short as the Explorer's header makes it; the full path is the tooltip
  const root = rootText(Math.min(TITLE_ROOT_MAX, Math.max(12, chars(t) - name.length - 5)));
  t.textContent = `${name}${root ? "  ·  " + root : ""}`;
  t.title = info ? info.root : "";
}
function renderRoot() {
  const chip = $("#root-chip");
  if (!chip || !info) return;
  const of = $("#open-folder-btn");
  if (of) {
    of.disabled = !info.open_folder;
    of.dataset.tip = info.open_folder ? "Open Folder…" : "Turned off when this server started (--root none)";
  }
  chip.textContent = rootText(chars(chip));
  chip.title = info.root;
  const w = $("#welcome-root");
  if (w) { w.textContent = rootText(Math.max(16, chars(w) - 2)); w.title = info.root; }
}
$("#open-folder-btn")?.addEventListener("click", () => run("file.openFolder"));
// the folder name in the Explorer's header opens the same dialog
$("#root-chip")?.addEventListener("click", () => { if (info && info.open_folder) run("file.openFolder"); });
let rootFrame = 0;
addEventListener("resize", () => { cancelAnimationFrame(rootFrame); rootFrame = requestAnimationFrame(() => { renderRoot(); updateTitle(); }); });

/* ================================================================ layout */
const shell = $("#shell");
/* The layout: the profile's copy (ui.layout, on the server, follows you to a
 * new port or browser), with localStorage as the cache read before boot. Zen
 * and the narrow-screen drawer stay in this browser only. */
const layout = (() => { try { return JSON.parse(localStorage.getItem(layoutKey) || "{}"); } catch (e) { return {}; } })();
let sidebar = null;              // the explorer's sections: order, shown, sizes (sidebar.js, layout.sidebar)
/* mode "local": a change the page made by itself (the bottom tab following the
 * document): kept in this browser, not sent to the server on every switch */
function saveLayout(mode) {
  if (PANE) return;            // the right editor group follows the window's layout, never writes it
  try { localStorage.setItem(layoutKey, JSON.stringify(layout)); } catch (e) { /* private */ }
  if (!profileReady || mode === "local") return;
  const { zen, ...shared } = layout;
  profile.setUi("layout", shared);
}
let profileReady = false;
/* The server's layout (another browser, another port), zen kept from this browser. */
function adoptProfileLayout() {
  const shared = profile.ui("layout", null);
  if (!shared || typeof shared !== "object") return;
  const { zen: z0, ...mine } = layout;
  if (JSON.stringify(mine) === JSON.stringify(shared)) return;     // nothing new: no relayout
  const zen = layout.zen, ownTab = layout.bottomTab;
  for (const k of Object.keys(layout)) delete layout[k];
  Object.assign(layout, shared);
  if (zen !== undefined) layout.zen = zen;
  if (PANE) {                  // the right editor group: the window's sizes, its own bottom tab
    layout.bottomTab = ownTab;
    applyLayout();
    return;
  }
  try { localStorage.setItem(layoutKey, JSON.stringify(layout)); } catch (e) { /* private */ }
  applyLayout();
  const tab = $(".bottom-tab.active");
  if (layout.bottomTab && (!tab || tab.dataset.bottom !== layout.bottomTab) && $(`.bottom-tab[data-bottom="${layout.bottomTab}"]`)) showBottom(layout.bottomTab, false);
}
/* Below 1100px the inspector is a drawer over the viewport (spec §1.1): shut
 * unless asked for, and not part of the saved layout. */
const NARROW = 1100;
let drawerOpen = false, wasNarrow = innerWidth < NARROW;
const narrow = () => innerWidth < NARROW;
function applyLayout() {
  shell.classList.toggle("no-explorer", layout.explorer === false);
  shell.classList.toggle("no-inspector", narrow() ? !drawerOpen : layout.inspector === false);
  shell.classList.toggle("drawer-open", narrow() && drawerOpen);
  shell.classList.toggle("no-bottom", layout.bottom === false);
  shell.classList.toggle("no-activity", !AP.shown());
  shell.classList.toggle("zen", !!layout.zen);
  zenChip();
  renderPanelControls();
  // widths saved on another screen: clamped to this window
  const clampW = (v, f) => Math.min(v, Math.round(innerWidth * f));
  if (layout.wExplorer) shell.style.setProperty("--w-explorer", clampW(layout.wExplorer, 0.4) + "px");
  if (layout.hBottom) shell.style.setProperty("--h-bottom", Math.min(layout.hBottom, Math.round(innerHeight * 0.6)) + "px");
  // Setup needs room for its form; closing it gives the inspector its own width back
  const wInsp = clampW(layout.wInspector || 300, 0.45);
  shell.style.setProperty("--w-inspector", (setupOpen ? Math.max(wInsp, 460) : wInsp) + "px");
  if (sidebar) sidebar.apply();
  relayout();
}
let relayoutFrame = 0;
function relayout() {
  cancelAnimationFrame(relayoutFrame);
  relayoutFrame = requestAnimationFrame(() => dispatchEvent(new Event("resize")));
}
function panelShown(name) {
  if (name === "zen") return !!layout.zen;
  return !shell.classList.contains("no-" + name);
}
function togglePanel(name) {
  if (name === "inspector" && narrow()) { drawerOpen = !drawerOpen; afterPanels(name, drawerOpen); return; }
  if (name === "zen") layout.zen = !layout.zen;
  else layout[name] = !panelShown(name);
  saveLayout(name === "zen" ? "local" : undefined); applyLayout(); renderActivity();     // Zen stays in this browser
  if (name === "zen" && layout.zen) zenEntered();
  if (name !== "zen") afterPanels(name, layout[name]);
}

/* ---------------------------------------------------------------- hiding and showing the panels
 * The Explorer, the Inspector and the bottom panel each hide from their
 * header's ×, the View menu, the palette, their key, the status bar's
 * toggles, or a splitter dragged shut; a hidden one leaves a tab on its edge
 * of the editor that brings it back. Hide All Panels remembers which were
 * open and brings back exactly those (layout.focus). */
const PANEL_INFO = { explorer: ["Explorer", "view.explorer"], inspector: ["Inspector", "view.inspector"], bottom: ["Bottom Panel", "view.bottom"] };
const PANEL_NAMES = Object.keys(PANEL_INFO);
const keyText = (id) => { const k = shortcutOf(id); return k ? ` (${k})` : ""; };
function afterPanels(name, shown) {
  if (layout.focus && PANEL_NAMES.some((n) => panelShown(n))) { delete layout.focus; saveLayout(); }
  applyLayout(); renderActivity(); renderPanelControls(); renderStatus();
  if (shown === false && name) panelHint(name);
}
function toggleAllPanels() {
  if (layout.focus) {
    const back = Array.isArray(layout.focus.restore) && layout.focus.restore.length ? layout.focus.restore : PANEL_NAMES;
    delete layout.focus;
    for (const n of back) { if (n === "inspector" && narrow()) drawerOpen = true; else layout[n] = true; }
  } else {
    const open = PANEL_NAMES.filter((n) => panelShown(n));
    layout.focus = { restore: open.length ? open : PANEL_NAMES };
    PANEL_NAMES.forEach((n) => { layout[n] = false; });
    drawerOpen = false;
  }
  saveLayout(); applyLayout(); renderActivity(); renderPanelControls(); renderStatus();
  if (layout.focus) panelHint(null);
}
/* The first time a panel goes (once a session, until "Don't show again"): how to get it back. */
function panelHint(name) {
  if (PANE) return;
  try { if ((profile.ui("hints", {}) || {}).panelHidden || sessionStorage.getItem("mace.gui.panelHint")) return; sessionStorage.setItem("mace.gui.panelHint", "1"); } catch (e) { /* private */ }
  const text = name ? `${PANEL_INFO[name][0]} hidden: click the tab on its edge or press ${shortcutOf(PANEL_INFO[name][1]) || "its key"} to bring it back`
    : `All panels hidden: click Show Panels or press ${shortcutOf("view.hideAllPanels") || "its key"} to bring them back`;
  toast(text, "info", { label: "Don't show again", ms: 10000,
    run: () => profile.setUi("hints", { ...(profile.ui("hints", {}) || {}), panelHidden: true }) });
}
/* The header ×, the edge tabs, the status bar's toggles and the Show Panels chip follow the layout. */
function renderPanelControls() {
  if (PANE) return;
  for (const n of PANEL_NAMES) {
    const shown = panelShown(n), [label, cmd] = PANEL_INFO[n];
    const edge = $(`.edge-tab[data-panel="${n}"]`);
    if (edge) { edge.hidden = shown || !!layout.zen; edge.setAttribute("aria-label", `Show ${label}${keyText(cmd)}`); }
    // a document's own side panel (the orbital picker) keeps clear of a shown edge tab (volume.css)
    const center = $("#center");
    if (center) center.classList.toggle(`edge-${n}`, !!edge && !edge.hidden);
    const x = $(`.panel-hide[data-panel="${n}"]`);
    if (x) x.setAttribute("aria-label", `Hide ${label}${keyText(cmd)}`);
  }
  const chip = $("#panels-exit");
  if (chip) { chip.hidden = !layout.focus || !!layout.zen; chip.querySelector("kbd").textContent = shortcutOf("view.hideAllPanels") || ""; }
}
function initPanelControls() {
  if (PANE) { paneBottomX(); return; }
  const btn = (n, cls, ico) => {
    const [label, cmd] = PANEL_INFO[n];
    return `<button class="${cls}" type="button" data-panel="${n}" data-panel-cmd="${cmd}" aria-controls="${n}" data-tip="${cls === "edge-tab" ? "Show" : "Hide"} ${label}" data-tip-cmd="${cmd}">${icon(ico)}</button>`;
  };
  $("#explorer .side-title").insertAdjacentHTML("beforeend", btn("explorer", "icon-btn sm panel-hide", "x"));
  $("#bottom").insertAdjacentHTML("afterbegin", btn("bottom", "icon-btn sm panel-hide", "x"));
  // the Inspector's × lives in its header: the Inspector's bar, or Calculation Setup's header while
  // Setup replaces it (a bar of its own while neither is drawn yet)
  const insp = $("#inspector");
  const keepX = () => {
    let x = $(".panel-hide[data-panel=inspector]", insp);
    const head = $(":scope > .insp-bar:not(.insp-bar-min)", insp) || $(".su-head", insp);
    const min = $(":scope > .insp-bar-min", insp);
    if (head && min) min.remove();
    const target = head || min || (() => {
      const bar = document.createElement("div");
      bar.className = "insp-bar insp-bar-min";
      bar.innerHTML = `<span class="insp-bar-title">Inspector</span>`;
      insp.prepend(bar);
      return bar;
    })();
    if (x && x.parentElement === target) return;
    if (!x) { target.insertAdjacentHTML("beforeend", btn("inspector", "icon-btn sm panel-hide", "x")); x = $(".panel-hide[data-panel=inspector]", target); x.setAttribute("aria-expanded", "true"); }
    else target.appendChild(x);
    renderPanelControls();
  };
  new MutationObserver(keepX).observe(insp, { childList: true, subtree: true });
  keepX();
  $("#center").insertAdjacentHTML("beforeend", btn("explorer", "edge-tab", "panel-left") + btn("inspector", "edge-tab", "panel-right") + btn("bottom", "edge-tab", "panel-bottom"));
  document.body.insertAdjacentHTML("beforeend", `<button class="zen-exit panels-exit" id="panels-exit" type="button" data-panel-cmd="view.hideAllPanels" hidden>${icon("maximize")}<span>Show Panels</span><kbd></kbd></button>`);
  $$(".edge-tab").forEach((b) => b.setAttribute("aria-expanded", "false"));
  $$(".panel-hide").forEach((b) => b.setAttribute("aria-expanded", "true"));
  document.addEventListener("click", (e) => {
    const b = e.target.closest("[data-panel-cmd]");
    if (b) { e.stopPropagation(); run(b.dataset.panelCmd); }
  }, true);
  renderPanelControls();
}
/* 1.1.6: the right editor group's own bottom panel had no ×, so it could not be hidden from where it shows:
 * this one runs View › Bottom Panel, which the window owns (groups.js forwards it), hiding both groups' panels */
function paneBottomX() {
  const b = $("#bottom");
  if (!b || $(".pane-x", b)) return;
  b.insertAdjacentHTML("afterbegin", `<button class="icon-btn sm panel-hide pane-x" type="button" data-tip="Hide Bottom Panel" data-tip-cmd="view.bottom" aria-label="Hide Bottom Panel">${icon("x")}</button>`);
  $(".pane-x", b).addEventListener("click", (e) => { e.stopPropagation(); run("view.bottom"); });
}
/* Zen Mode hides the menu bar, toolbar, activity bar and status bar: the chip
 * (top right), Esc and the viewport menu lead out. The chip fades after 3 s
 * and comes back on hover, focus or a pointer near the top of the window. */
let zenTimer = 0;
function zenChip() {
  const chip = $("#zen-exit");
  if (!chip) return;
  const on = !!layout.zen;
  if (chip.hidden === !on) return;
  chip.hidden = !on;
  chip.classList.remove("dim");
  clearTimeout(zenTimer);
  if (on) zenTimer = setTimeout(() => chip.classList.add("dim"), 3000);
}
addEventListener("pointermove", (e) => {
  const chip = $("#zen-exit");
  if (!chip || chip.hidden || e.clientY > 80) return;
  chip.classList.remove("dim");
  clearTimeout(zenTimer);
  zenTimer = setTimeout(() => chip.classList.add("dim"), 3000);
}, { passive: true });
function zenEntered() {
  try { if (sessionStorage.getItem("mace.gui.zenHint")) return; sessionStorage.setItem("mace.gui.zenHint", "1"); } catch (e) { /* private */ }
  toast(`Zen Mode: press Esc or ${shortcutOf("view.zen") || "Ctrl+Alt+Z"} to leave`, "info");
}
function splitter(el, prop, key, min, max, sign, axis = "x") {
  const def = { wExplorer: 260, wInspector: 300, hBottom: 240 }[key];
  // the sizes are the shell's own px: a screen px of pointer travel is 1/zoom of them (splitter.js)
  bindSplitter(el, { axis, sign, def, min: () => min, max: () => max, value: () => layout[key] || def,
    set: (v) => { layout[key] = v; shell.style.setProperty(prop, v + "px"); relayout(); },
    zoom: () => window.maceZoom.of(shell), commit: () => saveLayout(),
    // dragged well past its smallest size: the panel closes (its size stays what it was)
    release: (raw, base) => {
      const panel = { wExplorer: "explorer", wInspector: "inspector", hBottom: "bottom" }[key];
      if (!panel || raw >= min - 60 || (panel === "inspector" && narrow())) return false;
      layout[key] = base; shell.style.setProperty(prop, base + "px");
      layout[panel] = false; saveLayout(); afterPanels(panel, false); return true;
    } });
}

/* ================================================================ activity bar */
let activityHtml = "";
function renderActivity() {
  const items = [["view.explorer", "files", panelShown("explorer"), "Explorer"], ["view.database", "database", active && active.kind === "db", "Database"],
    ["file.quickOpen", "search", false, "Search files"], ["view.tasks", "list-checks", false, "Tasks"],
    // the bottom panel's Jobs tab and the Workflows view, as View › Jobs and View › Workflows open them
    ["view.jobs", "clock", false, "Jobs"], ["view.workflows", "layers", active && active.kind === "workflows", "Workflows"]];
  if (clusterShown()) items.push(["view.cluster", "activity", active && (active.kind === "cluster" || active.kind === "job"), "Cluster"]);
  const html = AP.arrange(items).map(([id, ic, on]) =>
    `<button class="icon-btn" data-cmd="${id}" aria-pressed="${!!on}" aria-label="${esc(registry.get(id).title)}">${icon(ic)}</button>`).join("") +
    `<span style="flex:1"></span><button class="icon-btn" data-cmd="file.settings" aria-label="Settings">${icon("settings")}</button>`;
  // only when something changed: a button replaced between press and release loses the click (and its focus)
  if (html !== activityHtml) { activityHtml = html; $("#activity").innerHTML = html; AP.decorate($("#activity")); }
}
/* 1.1.6 (js/activity_prefs.js): the activity bar's views, reordered by dragging or its context menu, hidden one by one or all */
const ACTIVITY_IDS = ["view.explorer", "view.database", "file.quickOpen", "view.tasks", "view.jobs", "view.workflows", "view.cluster"];
AP.enableDrag($("#activity"), () => ACTIVITY_IDS);
$("#activity").addEventListener("contextmenu", (e) => {
  e.preventDefault();
  const b = e.target.closest("[data-cmd]"), id = b && b.dataset.cmd !== "file.settings" ? b.dataset.cmd : null;
  openAt([...(id ? ["view.activity.hideItem", "view.activity.moveUp", "view.activity.moveDown", "-"] : []),
    { head: "Show in the Activity Bar" }, ...ACTIVITY_IDS.map((x) => `view.activity.show.${x}`), "-", "view.activity.reset", "view.activityBar"],
  e.clientX, e.clientY, "Activity bar menu", { id });
});
/* Every container of [data-cmd] buttons outside the menus needs one delegated
 * click listener (Enter and Space on a focused button arrive as clicks too):
 * #toolbar (renderToolbar), #tab-welcome (welcome), and these two, whose
 * buttons are re-rendered on every state change, so the listener sits on the
 * container, attached once. run() refuses a disabled command and says why. */
for (const sel of ["#activity", "#menubar .menubar-end", "#zen-exit"]) {
  const box = $(sel);
  if (box) box.addEventListener("click", (e) => { const b = e.target.closest("[data-cmd]"); if (b) run(b.dataset.cmd); });
}

/* The Cluster button: shown where SLURM or a quota tool exists (cluster.showView
 * auto), always (on) or never (off). Its click runs view.cluster through the
 * activity bar's delegated listener above. */
let clusterCaps = null;
function clusterShown() {
  const v = settings.get("cluster.showView");
  if (v === "off") return false;
  if (v === "on") return true;
  return !!(clusterCaps && (clusterCaps.slurm || (clusterCaps.quota_sources || []).length));
}

/* ================================================================ toolbar */
function tbButton(id, extra = "") {
  const c = registry.get(id);
  return `<button class="icon-btn" data-cmd="${id}" aria-label="${esc(c.title)}"${extra}>${icon(c.icon)}${id === "file.save" ? '<span class="dirty-dot" hidden></span>' : ""}</button>`;
}
function renderToolbar() {
  const out = [];
  TOOLBAR.forEach((g, k) => {
    if (k && g !== "spacer" && TOOLBAR[k - 1] !== "spacer" && !g.text) out.push('<span class="tb-sep"></span>');
    if (g === "spacer") out.push('<span class="tb-spacer"></span>');
    else if (Array.isArray(g)) out.push(`<span class="tb-group">${g.map((id) => tbButton(id)).join("")}</span>`);
    else if (g.radio) out.push(`<span class="tb-group" role="radiogroup" aria-label="Tool">${g.radio.map((id) => tbButton(id, ' role="radio"')).join("")}</span>`);
    // the tooltip reads names and keys from the registry when it shows, so a rebinding shows at once
    else if (g.split === "viewAlong") out.push(`<span class="tb-group"><button class="tb-btn-text" id="tb-along" data-tip="View along, rotation mode and angle snap" data-tip-group="view.alongA,view.alongB,view.alongC,view.alongAStar,view.alongBStar,view.alongCStar" aria-haspopup="menu">${icon("view-a")}<span id="tb-along-label">a</span>${icon("chevron-down", "i chev")}</button></span>`);
    else if (g.split === "range") out.push(`<span class="tb-group"><button class="tb-btn-text" id="tb-range" data-tip="Display range" data-tip-cmd="view.rangeCycle" aria-haspopup="menu"><span id="tb-range-label">1×1×1</span>${icon("chevron-down", "i chev")}</button></span>`);
    else if (g.segment === "sym") out.push(`<span class="segment" role="radiogroup" aria-label="Symmetry mode"><button role="radio" data-cmd="build.symModeP1" aria-label="P1 mode">P1</button><button role="radio" data-cmd="build.symModeSym" aria-label="Symmetry-aware mode">Sym</button></span>`);
    else if (g.text) out.push(`<button class="tb-btn-text${g.primary ? " primary" : ""}" data-cmd="${g.text}">${icon(registry.get(g.text).icon)}<span>${esc(g.label)}</span></button>`);
  });
  $("#toolbar").innerHTML = out.join("");
  $("#toolbar").addEventListener("click", (e) => {
    const b = e.target.closest("[data-cmd]");
    if (b) { run(b.dataset.cmd); return; }
    if (e.target.closest("#tb-along")) {
      const r = $("#tb-along").getBoundingClientRect();
      // View along a, b, c, ..., then the rotation mode, its angle snap and the snap to the nearest axis
      openAt([{ head: "View along" }, ...MENUS[2].items[2].sub, "-", ...ROTATION_MENU], r.left, r.bottom + 2, "View along");
    } else if (e.target.closest("#tb-range")) {
      const r = $("#tb-range").getBoundingClientRect();
      openAt(["view.range1", "view.range2", "view.range3", "view.rangeCells", "view.rangeCustom"], r.left, r.bottom + 2, "Display range");
    }
  });
}
function renderToolbarState() {
  const c = ctx();
  $$("[data-cmd]").forEach((b) => {
    const cmd = registry.get(b.dataset.cmd);
    if (!cmd || b.closest(".menu")) return;
    const why = disabledReason(cmd, c);
    b.setAttribute("aria-disabled", String(!!why));
    if (why) b.dataset.reason = why; else delete b.dataset.reason;
    if (cmd.checked) {
      const on = !!cmd.checked(c);
      b.setAttribute(b.getAttribute("role") === "radio" ? "aria-checked" : "aria-pressed", String(on));
    }
  });
  const dot = $("#toolbar .dirty-dot"), anyDirty = docs.some(isDirty);
  if (dot) dot.hidden = !anyDirty;
  // the fourth dirty mark (spec §2.6): Save in the accent color
  const save = $('#toolbar [data-cmd="file.save"]');
  if (save) save.classList.toggle("accent", isDirty(active));
  if ($("#tb-range-label")) {
    // a SLAB or POLYMER has no period along its vacuum axes: the label says what is drawn (3×3×1)
    const np = nonPeriodicAxes(), lab = DS.rangeLabel(viewDoc()), m = lab.split("×");
    // the label is what is drawn: a SLAB's c is 1 unless the user asked for more (then the tooltip names the spacing)
    $("#tb-range-label").textContent = lab;
    const rep = np.filter((k) => Number(m[k]) > 1);
    const open = rep.length && !VAC.repeats(VAC.of(viewDoc()));
    $("#tb-range").dataset.tip = open ? `Display range (the Open vacuum draws ${rep.map((k) => "abc"[k]).join(", ")} once: pick another Vacuum mode to repeat it)`
      : rep.length ? `Display range (${rep.map((k) => "abc"[k]).join(", ")} repeated at the ${vacuumText()} vacuum spacing)`
      : np.length ? `Display range (${nonPeriodicText(np)})` : "Display range";
  }
  VAC.sync();
  renderActivity();
}

/* ================================================================ view state */
const optEl = (id) => document.getElementById(id);
const optChecked = (id) => !!(optEl(id) && optEl(id).checked);
function toggleOpt(id, value) {
  const el = optEl(id);
  el.checked = value === undefined ? !el.checked : value;
  el.dispatchEvent(new Event("change"));
  redrawFreq();
  refresh();
}
/* The document the 3D views draw: the active one when it has a view, else the
 * structure viewer's owner. Its display state (viewport/display_state.js:
 * range, boundary atoms, rendering) is per document; Settings hold the defaults. */
function viewDoc() {
  if (active && ["structure", "trajectory", "freq", "volume"].includes(active.kind)) return active;
  return docs.find((x) => x.id === panelOwner.structure) || active || null;
}
/** A Settings value as the 3D views use it: a per-document display key comes from the drawn document. */
function viewSetting(k) { return DS.KEYS.has(k) ? DS.value(viewDoc(), k) : settings.get(k); }
/** A document's bond rule values (geom/core.js bondRule; Build › Calculate Bonds…): Clean holds the bonds the viewer draws. */
const bondValues = (d = viewDoc()) => Object.fromEntries(DS.GROUPS.bonds.map((k) => [k, DS.value(d, k)]));
/* Display range: 1×1×1, 2×2×2, 3×3×3, n1×n2×n3, or fractional from/to (Inspector ›
 * Range & boundary). range() is n for a cube from 0, 0 for anything else; the
 * classic views' 2×2×2 checkbox mirrors "2". */
const range = () => {
  const r = DS.rangeOf(viewDoc()), np = nonPeriodicAxes(), per = [0, 1, 2].filter((k) => !np.includes(k));
  // n×n×1 of a SLAB is the n preset (its c stays 1)
  if (per.length && np.every((k) => r.lo[k] === 0 && r.hi[k] === 1)) {
    const h = r.hi[per[0]];
    return per.every((k) => r.lo[k] === 0 && r.hi[k] === h) && Number.isInteger(h) ? h : 0;
  }
  return r.lo.every((x) => x === 0) && r.hi.every((x) => x === r.hi[0] && Number.isInteger(x)) ? r.hi[0] : 0;
};
const rangeCells = () => { const r = DS.rangeOf(viewDoc()); return r.hi.map((x, k) => Math.max(1, Math.round(x - r.lo[k]))); };
function setRange(n) {
  const d = viewDoc();
  if (!d) return;
  // n×n×n repeats the periodic directions only (say so); n1×n2×n3 with more than 1 along a vacuum axis repeats it
  // at the Vacuum spacing (Settings › Viewport)
  const np = nonPeriodicAxes();
  const cells = Array.isArray(n) ? n.map((x) => Math.max(1, Math.min(10, x | 0))) : [0, 1, 2].map((k) => (np.includes(k) ? 1 : n));
  DS.setCells(d, cells);
  if (!Array.isArray(n) && n > 1 && np.length) toast(`${nonPeriodicText(np)}: the range repeats ${cells.join("×")} (a custom range repeats it at the vacuum spacing)`, "info");
}
/* 1.1.6: the drawn document's periodic directions (the cells picker leaves the others at 1) */
function viewPbc() {
  const d = viewDoc(), all = [true, true, true];
  if (!d) return all;
  if (d.kind === "volume") return (d.meta && d.meta.pbc) || all;
  if (d.kind === "freq") return (d.structure && d.structure.pbc) || all;
  const st = structureOf(d);
  return st && st.pbc ? st.pbc.map(Boolean) : all;
}
/* The lattice directions (0, 1, 2) the drawn structure has no period along: c of a SLAB, b and c of a POLYMER. */
function nonPeriodicAxes() {
  const d = viewDoc();
  // a cube's periodicity is its header's cell (the server's meta.pbc): CRYSTAL writes a slab's c as the formal 500 Å
  if (d && d.kind === "volume") { const m = d.meta, pbc = m && m.pbc; return pbc ? [0, 1, 2].filter((k) => !pbc[k]) : []; }
  if (!d || !["structure", "trajectory"].includes(d.kind)) return [];
  const s = structureOf(d), pbc = s && s.cell && s.pbc;
  // a molecule (no periodic direction) is drawn once along x, y and z
  return pbc ? [0, 1, 2].filter((k) => !pbc[k]) : s && s.pbc ? [0, 1, 2] : [];
}
function vacuumText() { return VAC.text(VAC.of(viewDoc())); }
function nonPeriodicText(np) {
  const s = structureOf(viewDoc()), dim = (s && s.dimensionality) || (np.length === 1 ? "SLAB" : "POLYMER");
  const ax = np.map((k) => "abc"[k]);
  return `A ${dim} is not periodic along ${ax.join(" and ")}, so it is drawn once along ${ax.length > 1 ? "them" : "it"}`;
}
/* A display change of a document (this page, or the server's copy arriving): redraw it when it is on screen. */
function displayChanged(doc) {
  const box = optEl("opt-super");
  if (box && box.checked !== (range() === 2)) box.checked = range() === 2;
  syncBoundaryBox();
  syncStyleBox();
  VAC.sync();
  if (doc && doc !== viewDoc()) return;
  redrawSoon();
  if (displayPanel) displayPanel.refresh();
  refresh();
}
/* The Boundary atoms checkbox shows the drawn document's value (it is per document). */
function syncBoundaryBox() {
  const el = optEl("opt-boundary"), want = !!viewSetting("showBoundary");
  if (el && el.checked !== want) el.checked = want;
}
/* The Display section's Atom style shows what the drawn document's Style draws (a style that sets
 * atomDraw: Space-filling, Wireframe, VESTA-like); the select's own choice (all documents) is kept in
 * data-base and comes back when the style follows the Display section again. */
const DRAW_OPT = { spacefill: "spacefill", sticks: "stick", ballstick: "ballstick" };
function syncStyleBox() {
  const el = optEl("opt-style");
  if (!el) return;
  if (el.dataset.base === undefined) el.dataset.base = el.value;
  const want = DRAW_OPT[viewSetting("atomDraw")] || el.dataset.base;
  if (el.value !== want) el.value = want;
}
/** What the structure viewer draws: the classic views' checkboxes, Settings, and the document's display state. */
function viewOpts() {
  const d = viewDoc(), r = DS.drawRange(d), vac = VAC.of(d, d ? structureOf(d) : null);   // the document's vacuum mode (1.1.6)
  // the document's Style can choose the drawing (Space-filling: no bonds; Wireframe: thin sticks)
  const draw = M().effDraw ? M().effDraw(style(), optChecked("opt-bonds")) : { style: style(), bonds: optChecked("opt-bonds") };
  return { cell: optChecked("opt-cell"), boundary: !!viewSetting("showBoundary"), bonds: draw.bonds,
    labels: optChecked("opt-labels") ? (labelMode() === "off" ? "both" : labelMode()) : "off", style: draw.style,
    range: r, box: DS.boxOf(r, vac), bondedOutside: viewSetting("bondedOutside") || "off", vacuum: vac };
}
/* The Magnetic style's data (1.1.7): a document's Mulliken spin moments from its run (/api/spins),
 * fetched once per document: {state: "ok", values, max, source} when they match the drawn atoms, else
 * {state: "none" | "mismatch", reason}. While loading: {state: "loading"}. */
function atomSymbolsOf(d) {
  if (!d) return null;
  if (d.kind === "volume") { const u = volumeView && volumeView.atomsOf ? volumeView.atomsOf(d) : null; return u || null; }
  const s = d.kind === "freq" ? d.structure : structureOf(d);
  return s && s.symbols ? s.symbols : null;
}
function spinsOf(d) {
  if (!d || !d.path) return { state: "none", reason: "This document has no file, so there is no run to read spin moments from" };
  if (d.spins) return d.spins;
  d.spins = { state: "loading", reason: "Reading the spin moments…" };
  M().api("spins?" + M().q({ path: d.path })).then((r) => {
    const syms = atomSymbolsOf(d);
    if (r.state !== "ok") d.spins = { state: "none", reason: r.reason };
    else if (r.closed) d.spins = { state: "none", reason: `${r.source} is a closed-shell run: every Mulliken spin moment is zero` };
    else if (syms && (syms.length !== r.values.length || syms.some((x, k) => String(x).toLowerCase() !== String(r.symbols[k]).toLowerCase()))) {
      d.spins = { state: "mismatch", reason: `The Mulliken spins in ${r.source} are for ${r.values.length} atoms, not the ${syms.length} drawn here` };
    } else d.spins = { state: "ok", values: r.values, max: Math.max(0.1, ...r.values.map(Math.abs)), source: r.source };
  }).catch((e) => { d.spins = { state: "none", reason: `No spin moments: ${e.message}` }; })
    .finally(() => { if (d === viewDoc()) { redrawSoon(); if (displayPanel) displayPanel.refresh(); } });
  return d.spins;
}
/* Labels are a setting (Off / Element / Index / Element+Index / Wyckoff); the
 * classic Labels checkbox mirrors "anything but Off". Shift+L cycles all five. */
const labelMode = () => settings.get("labels");
function setLabels(v) { settings.set("labels", v); }
function cycleLabels() { const order = ["off", "element", "index", "both", "wyckoff"]; setLabels(order[(order.indexOf(labelMode()) + 1) % order.length]); }
const style = () => (optEl("opt-style") ? optEl("opt-style").value : "ballstick");
function setStyle(v) { const el = optEl("opt-style"); el.value = v; el.dispatchEvent(new Event("change")); redrawFreq(); refresh(); }
function mainViewer() {
  return M().state.viewers[active && active.kind === "trajectory" ? "viewer-before" : active && active.kind === "freq" ? "viewer-freq"
    : active && active.kind === "volume" ? "viewer-volume" : "viewer-main"];
}
/* The mode view (a FREQ document) draws with the toolbar's display options too (js/docs/freq.js). */
const FREQ_VIEW_ONLY = "The mode view is view-only: open its structure (Structure, top right) to edit";
const freqActive = () => !!(freqView && active && active.kind === "freq");
function redrawFreq() { if (freqActive()) freqView.redraw(); if (volumeActive()) volumeView.redraw(); }
/* The volume views (cube, orbitals) draw with them too (js/docs/volume.js). */
const volumeActive = () => !!(volumeView && active && active.kind === "volume");
/* View along a, b, c, a*, b*, c* or [u v w]: the camera module in a
 * structure document (with a defined "up", animated); the old turn elsewhere. */
function viewAlong(k) {
  const which = typeof k === "number" ? "abc"[k] : k;
  if (volumeActive()) {
    const v = mainViewer(), cell = volumeView.cell();
    if (!v || !cell || Array.isArray(which)) return;
    import("./geom/core.js").then((G) => {
      const dir = G.alongDirection(cell, which);
      v.setView([...v.getView().slice(0, 4), ...G.viewQuaternion(dir, G.upVector(cell, dir))]);
      v.render();
      if ($("#tb-along-label")) $("#tb-along-label").textContent = which;
    });
    return;
  }
  if (freqActive()) {
    const v = mainViewer(), fd = freqView.doc(), s = fd && fd.structure;
    if (!v || !s || !s.cell || Array.isArray(which)) return;
    import("./geom/core.js").then((G) => {
      const dir = G.alongDirection(s.cell, which), view = v.getView();
      v.setView([...view.slice(0, 4), ...G.viewQuaternion(dir, G.upVector(s.cell, dir))]);
      v.render();
      if ($("#tb-along-label")) $("#tb-along-label").textContent = which;
    });
    return;
  }
  const s = M().state.structure, v = mainViewer();
  if (!s || !v) return;
  // the view's name for an exported image (<stem>_a.png, _b*.png, _uvw.png), while the camera stays there
  const keep = () => { alongState = { label: Array.isArray(which) ? "uvw" : which, view: v.getView() }; };
  if (E && active && active.kind === "structure") {
    const r = E.viewAlong(which);
    if (r) r.then(() => { keep(); updateTriad(); }); else keep();
  } else if (typeof k === "number") { M().viewAlong(v, s.cell[k]); keep(); }
  else return;
  const label = Array.isArray(which) ? `[${which.join(" ")}]` : which;
  $("#tb-along-label") && ($("#tb-along-label").textContent = label);
  updateTriad();
}
function resetView() { if (freqActive()) { freqView.resetView(); return; } if (volumeActive()) { volumeView.resetView(); return; } M().drawStructure(false); updateTriad(); }
function toggleProjection() {
  settings.set("projection", settings.get("projection") === "orthographic" ? "perspective" : "orthographic");
}
function applyProjection() {
  Object.values(M().state.viewers || {}).forEach((v) => {
    if (v && v.setProjection) { v.setProjection(settings.get("projection")); v.render(); }
  });
}

/* ---------------- the mode chip (spec §1.4): the active tool, top left of the viewport */
const TOOL_HINTS = { lasso: "Drag around atoms to select them", orbit: "Drag to rotate · Shift: around the view axis",
  pan: "Drag to move the view", zoom: "Drag up or down to zoom" };
/* What the chip says beyond the tool: "Applying…" while an edit waits for
 * the server; the Move, Add Atom and Measure tools add their own. */
let chipExtra = (t) => {
  const e = edOf(active);
  if (E && E.chip) { const c = E.chip(t, e); if (c) return c; }
  return e && e.waiting ? "Applying…" : null;
};
function toolChip() {
  const wrap = $("#viewer-main") && $("#viewer-main").closest(".viewer-wrap");
  if (!wrap) return;
  let chip = $(".vp-chip", wrap);
  if (!chip) { wrap.insertAdjacentHTML("beforeend", `<div class="vp-chip" role="status" aria-live="polite"></div>`); chip = $(".vp-chip", wrap); }
  wrap.dataset.tool = tool;
  if (chip.contains(document.activeElement)) return;     // never under the user's typing (the element field)
  const c = registry.get(`tool.${tool}`);
  const extra = chipExtra ? chipExtra(tool) : null;
  if (extra) {
    chip.hidden = false;
    if (typeof extra === "string") { if (chip.textContent !== extra) chip.textContent = extra; return; }
    // the same chip again (a grab's preview updates often): keep the nodes, so a button under the pointer stays
    if (chip.firstElementChild && chip.childElementCount === 1 && chip.firstElementChild.outerHTML === extra.outerHTML) return;
    chip.textContent = ""; chip.appendChild(extra);
    return;
  }
  chip.hidden = tool === "select" || !c;
  if (!chip.hidden) chip.textContent = `${c.menuTitle || c.title} · ${TOOL_HINTS[tool] || ""}${TOOL_HINTS[tool] ? " · " : ""}Esc to exit`;
}

/* a volume grid's request at Settings › Viewport › Mesh quality (Standard: Volume quality, as before) */
const meshGridQuery = () => MQ.volumeQuery({ quality: settings.get(MQ.KEY) }, settings.get("volume.quality") || "medium");

/* ---------------- axis triad: an overlay SVG fed from the camera */
function attachTriad(tries = 0) {
  const v = M() && M().state.viewers["viewer-main"];
  if (!v && active && tries < 30) { requestAnimationFrame(() => attachTriad(tries + 1)); return; }
  if (!v || v.__triad) { updateTriad(); return; }
  v.__triad = true;
  const wrap = $("#viewer-main").closest(".viewer-wrap");
  wrap.insertAdjacentHTML("beforeend", `<svg class="axis-triad" id="axis-triad" viewBox="-40 -40 80 80" aria-hidden="true"></svg>`);
  wrap.setAttribute("tabindex", "0");
  wrap.setAttribute("role", "application");
  const render = v.render, show = v.show;
  v.render = (...a) => { const r = render(...a); updateTriad(); return r; };
  v.show = (...a) => { const r = show(...a); updateTriad(); return r; };
  if (v.setProjection) v.setProjection(settings.get("projection"));
  updateTriad();
}
function rotate(q, p) {   // quaternion (x, y, z, w) applied to vector p
  const [x, y, z, w] = q;
  const ix = w * p[0] + y * p[2] - z * p[1], iy = w * p[1] + z * p[0] - x * p[2];
  const iz = w * p[2] + x * p[1] - y * p[0], iw = -x * p[0] - y * p[1] - z * p[2];
  return [ix * w + iw * -x + iy * -z - iz * -y, iy * w + iw * -y + iz * -x - ix * -z, iz * w + iw * -z + ix * -y - iy * -x];
}
/* The volume view's own triad: the same overlay on its 3D view (viewer-volume), fed from its lattice. */
function attachVolumeTriad(v, wrap) {
  if (!v || !wrap || v.__triadVol) return;
  v.__triadVol = true;
  wrap.insertAdjacentHTML("beforeend", `<svg class="axis-triad" id="axis-triad-vol" viewBox="-40 -40 80 80" aria-hidden="true"></svg>`);
  const render = v.render, show = v.show;
  v.render = (...a) => { const r = render(...a); updateTriad(); return r; };
  v.show = (...a) => { const r = show(...a); updateTriad(); return r; };
  updateTriad();
}
function updateVolumeTriad() {
  const el = $("#axis-triad-vol");
  if (!el) return;          // attached with the volume view: nothing before it exists
  const v = M() && M().state.viewers["viewer-volume"], cell = volumeView && volumeView.cell();
  if (!v || !v.getModel) return;
  // a molecule has no cell: no axes to draw
  el.style.display = settings.get("showAxes") && cell && active && active.kind === "volume" ? "" : "none";
  if (!cell || !v.getModel()) return;
  const view = v.getView(), css = (k) => (M().vpVar ? M().vpVar(k) : getComputedStyle(document.documentElement).getPropertyValue(k).trim());
  el.innerHTML = triadSvg(triadAxes([view[4], view[5], view[6], view[7]], cell).sort((a, b) => a.z - b.z), (k) => css(`--axis-${"abc"[k]}`));
}
let triadFrame = 0;
function updateTriad() {
  cancelAnimationFrame(triadFrame);
  triadFrame = requestAnimationFrame(() => {
    updateVolumeTriad();
    const el = $("#axis-triad"), v = M() && M().state.viewers["viewer-main"], s = M() && M().state.structure;
    if (!el || !v || !s) return;
    el.style.display = settings.get("showAxes") ? "" : "none";
    const view = v.getView();
    const q = [view[4], view[5], view[6], view[7]];
    // the 3D view's own scope: a White or Black background in a theme of the other lightness has its own axis colors
    const css = (k) => (M().vpVar ? M().vpVar(k) : getComputedStyle(document.documentElement).getPropertyValue(k).trim());
    const axes = triadAxes(q, s.cell).sort((a, b) => a.z - b.z);              // far axes first, near ones on top
    el.innerHTML = triadSvg(axes, (k) => css(`--axis-${"abc"[k]}`));
  });
}
const TRIAD_LEN = 22;
/** The triad's arrows for a view quaternion: {k, x, y (SVG px), z (toward the viewer)}. */
function triadAxes(q, cell) {
  return [0, 1, 2].map((k) => {
    const c = cell[k], n = Math.hypot(...c) || 1;
    const r = rotate(q, c.map((x) => x / n));
    return { k, x: r[0] * TRIAD_LEN, y: -r[1] * TRIAD_LEN, z: r[2] };
  });
}
/** The triad's SVG for projected axes [{k, x, y, z}]. A label sits past the
 * tip, never closer than 14px to the origin, so an axis seen end-on still
 * gets a readable letter; axes pointing away are drawn fainter. */
export function triadSvg(axes, color) {
  const parts = ['<circle class="origin" r="2"/>'];
  for (const a of axes) {
    const len = Math.hypot(a.x, a.y);
    const dx = len > 0.5 ? a.x / len : 0.71, dy = len > 0.5 ? a.y / len : -0.71;
    const lr = Math.max(len + 8, 14);
    parts.push(`<g class="ax ax-${"abc"[a.k]}" style="--c:${color(a.k)}" opacity="${a.z < -0.3 ? 0.6 : 1}">` +
      `<line x1="0" y1="0" x2="${a.x.toFixed(1)}" y2="${a.y.toFixed(1)}"/>` +
      `<circle class="tip" cx="${a.x.toFixed(1)}" cy="${a.y.toFixed(1)}" r="2.2"/>` +
      `<text x="${(dx * lr).toFixed(1)}" y="${(dy * lr).toFixed(1)}">${"abc"[a.k]}</text></g>`);
  }
  return parts.join("");
}

/* ================================================================ inspector */
const parked = document.createElement("div");   // classic controls not shown right now
parked.hidden = true;
parked.id = "parked";
document.body.appendChild(parked);
/* caption: "All documents" / "This document" (which values a section's controls change);
 * startCollapsed: shut until the user opens it once (remembered like any collapse) */
function section(key, title, content, caption = "", startCollapsed = false) {
  const collapsed = (layout.collapsed || []).includes(key) || (startCollapsed && !(layout.opened || []).includes(key));
  const sec = document.createElement("section");
  sec.className = "insp-section" + (collapsed ? " collapsed" : "");
  sec.dataset.key = key;
  sec.innerHTML = `<button class="insp-head" aria-expanded="${!collapsed}">${icon("chevron-down")}<span></span>` +
    `${caption ? `<em class="insp-caption"></em>` : ""}</button><div class="insp-body"></div>`;
  $("span", sec).textContent = title;
  if (caption) $(".insp-caption", sec).textContent = caption;
  const body = $(".insp-body", sec);
  if (typeof content === "string") body.innerHTML = content; else if (content) body.appendChild(content);
  // Shift-click (or Alt-click, where the desktop leaves Alt-drag to the page) opens or shuts every section the way this one goes
  $(".insp-head", sec).addEventListener("click", (e) => {
    const on = !sec.classList.contains("collapsed");
    if (e.altKey || e.shiftKey) setAllSections(!on); else setCollapsed(sec, on);
  });
  return sec;
}
function setCollapsed(sec, on, save = true) {
  const key = sec.dataset.key;
  sec.classList.toggle("collapsed", on);
  const c = new Set(layout.collapsed || []), o = new Set(layout.opened || []);
  if (on) { c.add(key); o.delete(key); } else { c.delete(key); o.add(key); }
  layout.collapsed = [...c]; layout.opened = [...o];
  if (save) saveLayout();
  $(".insp-head", sec).setAttribute("aria-expanded", String(!on));
}
/** View › Expand All Sections / Collapse All Sections (and the Inspector's
 * header buttons): every section of the Inspector, or of Calculation Setup
 * while it is open, remembered like a single collapse. */
function setAllSections(open) {
  if (!panelShown("inspector")) togglePanel("inspector");
  if (setupOpen) { setupPanel().then((m) => m.setAllSections(open)).catch((e) => toast(`Calculation Setup: ${e.message}`, "error")); return; }
  $$("#inspector .insp-section").forEach((s) => { if (s.classList.contains("collapsed") === open) setCollapsed(s, !open, false); });
  saveLayout();
}
/* The Inspector's header, always: its name, the Expand all / Collapse all buttons when it has more
 * than one section, and the hide × (initPanelControls moves it in; it never floats over a section). */
function sectionBar(host) {
  const many = $$(".insp-section", host).length >= 2;
  const btn = (cmd, ic, label) => `<button type="button" class="icon-btn sm" data-all="${cmd === "view.expandSections" ? "open" : "shut"}" ` +
    `data-tip="${label}" data-tip-cmd="${cmd}" aria-label="${label}">${icon(ic)}</button>`;
  const bar = document.createElement("div");
  bar.className = "insp-bar";
  bar.innerHTML = `<span class="insp-bar-title">Inspector</span>` + (many ? `<span class="insp-bar-acts" role="group" aria-label="Inspector sections">` +
    btn("view.expandSections", "chevrons-up-down", "Expand All Sections") + btn("view.collapseSections", "chevrons-down-up", "Collapse All Sections") + "</span>" : "");
  bar.addEventListener("click", (e) => { const b = e.target.closest("[data-all]"); if (b) setAllSections(b.dataset.all === "open"); });
  host.prepend(bar);
}
/* The Range & boundary and Rendering groups (viewport/display_panel.js, loaded with the first structure). */
let displayPanel = null, displayPanelLoading = null;
// after a draw: the estimate and the bonded-atoms note follow what was drawn
document.addEventListener("mace-scene", () => { if (displayPanel) displayPanel.refresh(); });
// the polyhedra arrive from the worker after the scene: the Centers checkboxes follow
document.addEventListener("mace-polyhedra", () => { if (displayPanel) displayPanel.refresh(); });
function loadDisplayPanel() {
  if (displayPanel) return Promise.resolve(displayPanel);
  if (!displayPanelLoading) displayPanelLoading = import("./viewport/display_panel.js").then((m) => {
    displayPanel = m.createDisplayPanel({ esc, icon, doc: () => viewDoc(), structure: () => structureOf(viewDoc()), spins: (d) => spinsOf(d),
      bonded: () => (E && E.scene() ? E.scene().bonded : null), polyhedra: () => (E && E.scene() ? E.scene().polyhedra() : null), cssColor: (k) => getComputedStyle(document.documentElement).getPropertyValue(k).trim(),
      boundaryLabel: () => pieces.boundary, park: (el) => parked.appendChild(el), saved: (m2) => toast(m2, "success") });
    return displayPanel;
  });
  return displayPanelLoading;
}
// the mode and volume views draw atoms too: their Rendering (Style and Lighting) without the range group
function appendDisplayGroups(host, withRange = true) {
  const range = withRange ? section("range", "Range & boundary", null, "This document") : null;
  const render = section("render", "Rendering", null, "This document", true);
  if (range) host.appendChild(range);
  host.appendChild(render);
  const fill = (dp) => {
    if (!render.isConnected) return;
    if (range) $(".insp-body", range).replaceChildren(dp.range());
    $(".insp-body", render).replaceChildren(dp.render());
  };
  if (displayPanel) fill(displayPanel);
  else loadDisplayPanel().then(fill).catch((e) => toast(`Inspector: ${e.message}`, "error"));
}
/** View › Range and Boundary…, Rendering…: open the Inspector at that group. */
async function openDisplayGroup(group) {
  if (!panelShown("inspector")) togglePanel("inspector");
  if (setupOpen) toggleSetup(false);
  await loadDisplayPanel();
  const sec = $(`#inspector .insp-section[data-key="${group}"]`);
  if (!sec) return;
  if (sec.classList.contains("collapsed")) setCollapsed(sec, false);
  sec.scrollIntoView({ block: "start" });
  const f = $(".insp-body input, .insp-body select, .insp-body button", sec);
  if (f) f.focus();
}
const pieces = {};
function collectPieces() {
  const sCards = $$("#tab-structure .info-col > .card");
  pieces.structureTitle = sCards[0]; pieces.symmetry = sCards[1]; pieces.lattice = sCards[2]; pieces.atoms = sCards[3];
  pieces.display = $("#tab-structure .viewer-toolbar");
  // one Boundary atoms control, per document: it lives in Range & boundary, not in Display
  pieces.boundary = optEl("opt-boundary") ? optEl("opt-boundary").closest("label") : null;
  if (pieces.boundary) { pieces.boundary.querySelector("span").textContent = "Boundary Atoms"; pieces.boundary.removeAttribute("title"); }
  pieces.timeline = $("#tab-opt .info-col");
  pieces.editor = $("#tab-edit .info-col");
  pieces.plotOpts = $("#plot-opts");
  pieces.freqOpts = $("#freq-opts");
  pieces.freqTable = $("#freq-table");
  pieces.volumeOpts = $("#volume-opts");
  pieces.dbDetail = $("#db-detail");
  Object.values(pieces).forEach((p) => p && parked.appendChild(p));
  $$("#tab-structure .info-col, #tab-structure .empty").forEach((x) => x.classList.add("hidden"));
  const det = pieces.atoms.querySelector("details");     // the atom table shows by default
  if (det) { det.open = true; det.querySelector("summary").hidden = true; }
}
function renderInspector() {
  const host = $("#inspector");
  Object.values(pieces).forEach((p) => { if (p && host.contains(p)) parked.appendChild(p); });
  host.innerHTML = "";
  // toasts sit above Setup's sticky footer while it is open, never over Preview and Save
  document.documentElement.classList.toggle("setup-open", setupOpen);
  if (setupOpen) {
    // Calculation Setup replaces the inspector while open (spec §1.2): js/setup/panel.js, loaded on first use
    inspTabs(host);
    setupPanel().then((m) => { if (setupOpen) m.mount(host); }).catch((e) => toast(`Calculation Setup: ${e.message}`, "error"));
    return;
  }
  const k = active ? active.kind : null;
  const ed = edOf(active);
  // a plot, spectrum, volume or properties file: the structure of the run it came from, read-only (panelctx.js)
  const lent = k === "structure" || k === "trajectory" ? [] : PC.inspectorSections(active);
  if (k === "structure" && ed) {
    // a structure document: its selection, nudge, cell and symmetry are editable (js/editor/inspector.js)
    const editable = !ed.editBlock();
    inspEditable = editable;
    // a single-point output opens here (one geometry, no run view): its SCF first, then its Run card
    if (active.path && /\.(out|log)$/i.test(active.path)) {
      if (M().scfSection) host.appendChild(section("scf", "SCF", M().scfSection(active.path)));
      host.appendChild(section("run", "Run", runCard(active)));
    }
    host.appendChild(section("selection", "Selection", E.inspector.selection(ed)));
    if (editable) host.appendChild(section("nudge", "Nudge", E.inspector.nudge(ed)));
    host.appendChild(section("structure", "Structure", pieces.structureTitle));
    host.appendChild(section("cell", "Cell", editable ? E.inspector.cell(ed) : pieces.lattice));
    host.appendChild(section("symmetry", "Symmetry", E.inspector.symmetry(ed)));
    appendOverlays(host);
    host.appendChild(section("display", "Display", pieces.display, "All documents"));
    appendDisplayGroups(host);
    if (editable) {
      host.appendChild(section("classic", "More editing (classic)", pieces.editor));
      if (window.MaceClassic) window.MaceClassic.render();
    }
  } else if (k === "structure" || k === "trajectory") {
    // a run (.out): the Run card first (PLAN_analysis C4.8)
    if (active.path && /\.(out|log)$/i.test(active.path)) {
      host.appendChild(section("run", "Run", runCard(active)));
      // a single point (no run view): its SCF, cycle by cycle, where it opens (an optimization has it in the Timeline)
      if (k === "structure" && M().scfSection) host.appendChild(section("scf", "SCF", M().scfSection(active.path)));
    }
    host.appendChild(section("selection", "Selection", `<p class="insp-hint">Click an atom. Shift-click or drag to select more.</p>`));
    host.appendChild(section("structure", "Structure", pieces.structureTitle));
    host.appendChild(section("cell", "Cell", pieces.lattice));
    host.appendChild(section("symmetry", "Symmetry", pieces.symmetry));
    appendOverlays(host);
    host.appendChild(section("display", "Display", pieces.display, "All documents"));
    appendDisplayGroups(host);
  }
  else if (k === "runplot") host.appendChild(section("run", "Run", runCard({ path: active.runPath, id: active.id })));
  else if (k === "mech") {
    // the run's key settings (module, NUMDERIV, STEPSIZE, volumes, temperatures, functional), then its Run card
    host.appendChild(section("calculation", "Calculation", mechSettings(active)));
    host.appendChild(section("run", "Run", runCard(active)));
  }
  else if (k === "plot") host.appendChild(section("plotopts", "Plot options", pieces.plotOpts));
  else if (k === "freq") {
    host.appendChild(section("mode", "Mode", pieces.freqOpts));
    if (active.path && M().scfSection) host.appendChild(section("scf", "SCF", M().scfSection(active.path)));
    appendDisplayGroups(host, false);
  }
  else if (k === "volume") { host.appendChild(section("volume", "Volume", pieces.volumeOpts)); appendDisplayGroups(host, false); }
  else if (k === "db") host.appendChild(section("material", "Selected material", pieces.dbDetail));
  else if (!lent.length) host.appendChild(section("none", "Inspector", PC.inspectorPlaceholder(active)));
  lent.forEach(([key, title, el, caption]) => host.appendChild(section(key, title, el, caption)));
  sectionBar(host);
  PC.labelInspector(active);
  inspTabs(host);
}
/* Inspector › 3D Overlays (1.1.6, js/lazy/analysis_overlays.js): the pores, net, crystal shape and
 * Miller planes Structure Analysis drew on this document, once it has drawn one in this window */
function appendOverlays(host) {
  const A = window.__maceAnalysis;
  if (!A || !A.store || !A.store.records().length) return;
  const el = document.createElement("div");
  host.appendChild(section("overlays", "3D Overlays", el, "This document"));
  lazy("analysis_overlays").then((m) => m.fill(el, H)).catch((e) => console.error("overlays:", e));
}
document.addEventListener("mace-analysis", (e) => {
  const w = e.detail && e.detail.what;
  if ((w === "overlay" || w === "add") && !setupOpen && active && (active.kind === "structure" || active.kind === "trajectory")
    && !$('#inspector .insp-section[data-key="overlays"]')) renderInspector();
});
/* 1.1.6: the tabs over the Inspector column (Inspector | Setup: a way back from Setup besides its own button)
 * and the Basic / Advanced detail per view kind (js/inspector_tiers.js; niche controls carry data-tier="advanced") */
function inspTabs(host) {
  const why = setupOpen ? "" : disabledReason(registry.get("calc.setup"), ctx());
  host.prepend(IT.tabStrip({ setupOpen, setupWhy: why, tier: IT.tierOf(active && active.kind, active), icon,
    onInspector: () => toggleSetup(false), onSetup: () => run("calc.setup", { open: true }) }));
  IT.attach(host, { kind: () => (active ? active.kind : null), view: () => active, refresh: () => refresh() }).apply();
}
/* The selection, nudge, cell and symmetry sections follow the document:
 * rebuilt in place (the focused field gets its focus back). */
let inspFrame = 0, inspEditable = null;
function refreshInspector() {
  cancelAnimationFrame(inspFrame);
  inspFrame = requestAnimationFrame(() => {
    const ed = edOf(active);
    if (!ed || setupOpen || active.kind !== "structure") return;
    const ae = document.activeElement, label = ae && ae.closest("#inspector") ? ae.getAttribute("aria-label") : null;
    const caret = label && ae.selectionStart !== undefined ? [ae.selectionStart, ae.selectionEnd] : null;
    const editable = !ed.editBlock();
    // a preview began or ended (the Cell and Nudge sections change kind): rebuild
    // it all, so the parked read-only cards go back to their parking place
    if (layout.bottomTab === "coordination" && panelShown("bottom")) renderCoordination();
    if (editable !== inspEditable) { renderInspector(); return; }
    for (const key of ["selection", "nudge", "cell", "symmetry"]) {
      const sec = $(`#inspector .insp-section[data-key="${key}"] .insp-body`);
      if (!sec || (key === "cell" && !editable)) continue;
      Object.values(pieces).forEach((p) => { if (p && sec.contains(p)) parked.appendChild(p); });
      sec.replaceChildren(E.inspector[key](ed));
    }
    if (label) {
      const el = [...$$("#inspector [aria-label]")].find((x) => x.getAttribute("aria-label") === label);
      if (el) { el.focus(); if (caret && el.setSelectionRange) try { el.setSelectionRange(...caret); } catch (e) { /* not a text field */ } }
    }
  });
}
document.addEventListener("mace-selection", refreshInspector);
document.addEventListener("mace-doc", refreshInspector);

/* Calculation Setup (js/setup/panel.js): loaded the first time it opens. */
let setupLoading = null, setupMod = null;
function setupPanel() {
  if (setupMod) return Promise.resolve(setupMod);
  if (!setupLoading) setupLoading = import("./setup/panel.js").then((m) => m.init(SETUP_API, { esc, icon, toast, dialog }))
    .then((api) => { setupMod = api; return api; }).catch((e) => { setupLoading = null; throw e; });
  return setupLoading;
}
function toggleSetup(force) {
  setupOpen = force === undefined ? !setupOpen : !!force;
  if (setupOpen && !panelShown("inspector")) {
    if (narrow()) drawerOpen = true; else { layout.inspector = true; saveLayout(); }
  }
  renderInspector(); applyLayout(); refresh();
  requestAnimationFrame(revealBottomTab);
}

/* ================================================================ bottom panel */
const BOTTOM = [["atoms", "Atoms"], ["measurements", "Measurements"], ["history", "History"], ["timeline", "Timeline"],
  ["modes", "Modes"], ["coordination", "Coordination"], ["analysis", "Analysis"], ["properties", "Properties"], ["tasks", "Tasks"], ["jobs", "Jobs"],
  ["problems", "Problems"]];
/* The panel context (js/panelctx.js): every document-scoped panel follows the document in
 * focus, with a neutral placeholder where it has nothing for it; Tasks, Jobs and Problems are global. */
const OWN_KINDS = ["structure", "trajectory"];
const NO_RELATED = ["db", "workflows", "gallery", "cluster", "job", "batch", "editor", "structure", "trajectory"];
/** The file a document shows (an output, a plot's or spectrum's file, a run plot's run), or null. */
function docFilePath(d) {
  if (!d) return null;
  return d.path || d.srcPath || d.runPath || (d.plot && d.plot.path && !String(d.plot.path).includes("|") ? d.plot.path : null) || null;
}
/* The Properties tab's run: a CRYSTAL output, or a file one wrote (not a structure, a page or a database). */
const PROPS_NONE = ["db", "workflows", "gallery", "cluster", "job", "batch", "editor"];
function propertiesPath(d) {
  if (!d || PROPS_NONE.includes(d.kind)) return null;
  const p = docFilePath(d);
  return p && !/\.(cif|vasp|poscar|xyz|extxyz|json|toml|md|txt|csv|png|svg|html)$/i.test(p) ? p : null;
}
const PC = createPanelContext({ esc, icon, api: (route) => M().api(route),
  own: (d) => edOf(d) || runFrameOf(d),
  wantsRelated: (d) => {
    if (!d || NO_RELATED.includes(d.kind) || (d.plot && d.plot.kind === "convergence")) return null;
    return docFilePath(d);
  },
  changed: (d) => {
    if (d !== active) return;
    if (!setupOpen) renderInspector();
    renderPanels();
    if (layout.bottomTab === "coordination") renderCoordination();
    PC.apply(active, true);
  },
  openPath: (p) => openDoc(p, { as: "structure" }),
  activeTab: () => layout.bottomTab, coordMode: () => (settings.get("coordMode") === "cart" ? "cart" : "frac") });
{
  const pane = (k) => () => $(`.bottom-pane.${k}`), tab = (k) => () => $(`.bottom-tab[data-bottom="${k}"]`);
  const ownKind = (s) => !!s.doc && OWN_KINDS.includes(s.doc.kind);
  const atomsWhat = "structures, outputs, and the plots, spectra and volumes of a run";
  PC.register("inspector", { title: "Inspector", thing: "properties", what: "structures, outputs, plots, volumes, normal modes and the database",
    host: () => null, applies: (s) => !!s.doc && (!["text", "gallery", "workflows", "cluster", "job", "batch", "editor"].includes(s.doc.kind) || !!s.frame) });
  PC.register("atoms", { title: "Atoms", thing: "atoms", what: atomsWhat, borrow: true, host: pane("atoms"), tab: tab("atoms"),
    applies: (s) => !!s.frame || ownKind(s) });
  PC.register("measurements", { title: "Measurements", thing: "measurements", what: "structures and outputs", host: pane("measurements"), tab: tab("measurements"),
    applies: ownKind });
  PC.register("history", { title: "History", thing: "edit history", what: "structures open for editing (Edit This Step opens a run's step)",
    host: pane("history"), tab: tab("history"), applies: (s) => !!edOf(s.doc) || settingsHistory(s.doc) });
  PC.register("timeline", { title: "Timeline", thing: "timeline", what: "optimization runs", host: pane("timeline"), tab: tab("timeline"),
    applies: (s) => !!s.doc && s.doc.kind === "trajectory" });
  PC.register("modes", { title: "Modes", thing: "normal modes", what: "FREQ outputs", host: pane("modes"), tab: tab("modes"),
    applies: (s) => !!s.doc && s.doc.kind === "freq" });
  PC.register("coordination", { title: "Coordination", thing: "coordination table", what: atomsWhat, borrow: true,
    host: pane("coordination"), tab: tab("coordination"), applies: (s) => !!s.frame || ownKind(s) });
  // Structure Analysis results (lazy/analysis_results.js): the document's, all open documents' or the batches, its own choice
  PC.register("analysis", { title: "Analysis", thing: "analysis results", what: "structures", host: pane("analysis"), tab: tab("analysis"),
    applies: () => true });
  // what materials.db holds for the document's run (lazy/properties_tab.js)
  PC.register("properties", { title: "Properties", thing: "properties", what: "CRYSTAL outputs and the files a run wrote (plots, spectra, volumes, decks)",
    host: pane("properties"), tab: tab("properties"), applies: (s) => !!propertiesPath(s.doc) });
  PC.register("tasks", { title: "Tasks", scope: "global", note: "all documents" });
  PC.register("jobs", { title: "Jobs", scope: "global", note: "this folder" });
  PC.register("problems", { title: "Problems", scope: "global", note: "all documents" });
}
/** The document has a Calculation Setup history of its own (the History tab's Settings chip). */
function settingsHistory(d) {
  const subj = d && d.setupFor ? docs.find((x) => x.id === d.setupFor) : d;
  return !!(setupMod && subj && setupMod.historyRows(subj).length >= 2);
}
function initBottom() {
  $("#bottom-tabs").innerHTML = BOTTOM.map(([k, t]) => `<button class="bottom-tab" role="tab" data-bottom="${k}">${t}</button>`).join("");
  $("#bottom-body").innerHTML = BOTTOM.map(([k]) => `<div class="bottom-pane ${k}" data-pane="${k}" role="tabpanel"></div>`).join("");
  $(".bottom-pane.atoms").appendChild(pieces.atoms);
  $(".bottom-pane.atoms").insertAdjacentHTML("beforeend", `<p class="empty-note atoms-none">Open a structure to see its atoms.</p>`);
  $(".bottom-pane.timeline").appendChild(pieces.timeline);
  $(".bottom-pane.modes").appendChild(pieces.freqTable);
  $(".bottom-pane.modes").dataset.table = "modes";
  $(".bottom-pane.timeline").dataset.table = "compare";
  $(".bottom-pane.coordination").dataset.table = "coordination";
  $(".bottom-pane.coordination").innerHTML = `<p class="empty-note">Analysis › Coordination Analysis lists CN, ECoN and the neighbor shells here.</p>`;
  $(".bottom-pane.analysis").innerHTML = `<p class="empty-note">Analysis › Structure Analysis lists its results here, with the settings each was computed with.</p>`;
  $(".bottom-pane.measurements").innerHTML = `<p class="empty-note">No measurements. The Measure tool (M) arrives with the editor tools.</p>`;
  $(".bottom-pane.history").innerHTML = `<div class="hist-chips" role="radiogroup" aria-label="History of" hidden></div><div class="hist-host"><p class="empty-note" id="history-note">Undo and redo apply to the structure editor: ${esc(shortcutOf("edit.undo"))} and ${esc(shortcutOf("edit.redo"))}.</p></div>`;
  renderTasks();
  renderProblems();
  $("#bottom-tabs").addEventListener("click", (e) => { const b = e.target.closest("[data-bottom]"); if (b) showBottom(b.dataset.bottom); });
  showBottom(layout.bottomTab || "atoms", false);
}
/* The tab strip scrolls sideways when the center is narrow (Setup open at 1280 px): the active tab stays in sight. */
function revealBottomTab() {
  const b = $(".bottom-tab.active"), strip = $("#bottom-tabs");
  if (!b || !strip || b.hidden) return;
  const l = b.offsetLeft - strip.offsetLeft, r = l + b.offsetWidth;
  if (l < strip.scrollLeft) strip.scrollLeft = l - 8;
  else if (r > strip.scrollLeft + strip.clientWidth) strip.scrollLeft = r - strip.clientWidth + 8;
}
function showBottom(k, reveal = true) {
  if (k === "tasks" && !tasksPanel && taskStore.size) renderTasks();
  $$(".bottom-tab").forEach((b) => { b.classList.toggle("active", b.dataset.bottom === k); b.setAttribute("aria-selected", String(b.dataset.bottom === k)); });
  $$(".bottom-pane").forEach((p) => p.classList.toggle("active", p.dataset.pane === k));
  layout.bottomTab = k;
  PC.header();
  revealBottomTab();
  if (reveal && !panelShown("bottom")) { layout.bottom = true; applyLayout(); }
  saveLayout(reveal ? undefined : "local");
  if (k === "timeline" && M() && M().state.traj) requestAnimationFrame(() => M().drawEnergy());
  if (k === "coordination") renderCoordination();
  if (k === "analysis") { shell.classList.remove("doc-no-bottom"); analysisTab(); }
  if (k === "properties") { shell.classList.remove("doc-no-bottom"); propertiesTab(); }
  if (k === "jobs") {
    shell.classList.remove("doc-no-bottom");
    jobsPanel().then((p) => p.show()).catch((e) => console.error("jobs:", e));
  }
  else if (jobsView) jobsView.hide();
}
/* The Coordination tab (plan B9, js/editor/coordination.js, loaded on first use). */
let coordPanel = null, coordLoading = null;
function coordinationPanel() {
  if (coordPanel) return Promise.resolve(coordPanel);
  if (!coordLoading) coordLoading = import("./editor/coordination.js").then((m) => {
    // the panel fills its pane: the placeholder the panel context put there (a document it does not apply to) stays
    const host = $(".bottom-pane.coordination"), ph = host.querySelector(":scope > .pc-ph");
    coordPanel = m.createCoordPanel(host, { esc, wyckoff: (s, i) => (E ? E.wyckoff(s, i) : ""),
      tol: (s) => Number(viewSetting("bondTolerance")) || s.bond_scale || 1.15, copy: (t, done) => copyText(t, done),
      rule: (s) => m.ruleOf(s, viewSetting) });
    if (ph) host.prepend(ph);
    tableExporters.set("coordination", (fmt) => {
      const r = coordPanel.exportTable(fmt);
      if (!r) { toast("No table is shown", "info"); return false; }
      download(r.blob, r.name);
      return true;
    });
    return coordPanel;
  });
  return coordLoading;
}
function renderCoordination() {
  if (!coordPanel && layout.bottomTab !== "coordination") return;
  coordinationPanel().then((p) => p.render(edOf(active) || runFrameOf(active) || PC.borrowed(active))).catch((e) => console.error("coordination:", e));
}
function showCoordination() { showBottom("coordination"); }
/* The Analysis tab (1.1.6, js/lazy/analysis_results.js, loaded on first use): it follows the document itself. */
function analysisTab(batch = null) {
  const host = $(".bottom-pane.analysis");
  return lazy("analysis_results").then((m) => (batch ? m.showBatch(H, host, batch) : m.show(H, host))).catch((e) => console.error("analysis:", e));
}

/* The Properties tab (1.1.6, js/lazy/properties_tab.js, loaded on first use): what the database holds for the document's run. */
function propertiesTab() {
  const host = $(".bottom-pane.properties");
  return lazy("properties_tab").then((m) => m.show(H, host)).catch((e) => console.error("properties:", e));
}
function propertiesChanged() {
  if (lazyLoads.properties_tab) lazyLoads.properties_tab.then((m) => m.invalidate()).catch(() => {});
}
document.addEventListener("mace-db-changed", () => { propertiesChanged(); if (dbView && active && active.kind === "db") dbView.poll(); });

/* Submit, Jobs and Batch Setup (PLAN_submit): js/docs/jobs.js, js/batch/panel.js and
 * js/lazy/submit_dialog.js, each loaded the first time it is used. */
let jobsView = null, jobsLoading = null, jobCounts = null, batchView = null, batchLoading = null;
function submitCss() { return import("./lazy/lazy_util.js").then((lu) => lu.loadCss(import.meta.url, "../css/submit.css")); }
function jobsPanel() {
  if (jobsView) return Promise.resolve(jobsView);
  if (!jobsLoading) jobsLoading = Promise.all([import("./docs/jobs.js"), submitCss()]).then(([m]) => {
    jobsView = m.createJobsPanel($(".bottom-pane.jobs"), { esc, icon, toast, api: (...a) => M().api(...a),
      confirm: (o) => SETUP_API.confirm(o), countsChanged: (c) => { jobCounts = c; renderStatus(); },
      apiTask: (p, b) => apiTask(p, b),
      openOutput: (rel) => openAsText(rel), openLog: (rel) => openAsText(rel), openJob: (id) => openJob(id) });
    return jobsView;
  }).catch((e) => { jobsLoading = null; throw e; });
  return jobsLoading;
}
function batchPanel() {
  if (batchView) return Promise.resolve(batchView);
  if (!batchLoading) batchLoading = Promise.all([import("./batch/panel.js"), submitCss()]).then(([m]) => {
    batchView = m.createBatch($("#batch-host"), { esc, icon, toast, api: (...a) => M().api(...a),
      legend: (paths, title) => legendDialog({ title: `Elements of ${title}`, stem: "batch_selection", paths }),
      apiTask: (p, b) => apiTask(p, b), refreshFiles: () => M().loadFiles().catch(() => {}),
      showText: (title, text) => openTextDoc(title, text), openText: (rel) => openAsText(rel), copyText: (t, m) => copyText(t, m),
      submit: (targets, arrays) => openSubmit({ targets, preset: arrays ? { array: true } : null }),
      confirmBody: ({ title, body, yes, no }) => new Promise((resolve) => {
        let ok = false;
        const dl = dialog({ id: "batch-override", title, body,
          foot: `<button class="tb-btn-text" data-r="no">${esc(no || "Cancel")}</button><button class="tb-btn-text primary" data-r="yes">${esc(yes || "OK")}</button>`,
          onClose: () => resolve(ok) });
        dl.querySelectorAll("[data-r]").forEach((b) => b.addEventListener("click", () => { ok = b.dataset.r === "yes"; dl.close(); }));
      }) });
    return batchView;
  }).catch((e) => { batchLoading = null; throw e; });
  return batchLoading;
}
/** Submit… / Run on This Node…: the dialog shows the exact command and runs nothing until its Submit is clicked. */
function openSubmit({ targets, where = "slurm", preset = null }) {
  if (!targets || !targets.length) { toast("Pick a deck (.d12 or .d3) or a folder of decks", "info"); return null; }
  return lazy("submit_dialog").then((m) => m.open(H, { targets, where, preset }));
}
/** The targets of a Submit command: its arguments, the explorer's file, else the active deck. */
function submitTargets(a) {
  if (a && Array.isArray(a.paths) && a.paths.length) return a.paths;
  if (a && a.dir !== undefined) return [a.dir || "."];
  const p = pathOf(a);
  return p && /\.(d12|d3)$/i.test(p) ? [p] : [];
}
function openBatch(folder = "", preset = null) {
  const f = folder === "." ? "" : folder;
  let d = docs.find((x) => x.kind === "batch" && x.folder === f);
  if (!d) {
    saveView(active);
    d = { id: nextId++, path: null, kind: "batch", fileKind: "", folder: f, title: `Batch: ${f || "this folder"}` };
    docs.push(d);
  }
  if (preset) d.batchPreset = preset;          // from Calculation Setup: its kind of file, calculation and fields
  return activate(d);
}

/* ================================================================ status bar */
function selection() {
  const ed = edOf(active);
  if (ed) return { n: ed.selection.size, of: ed.n };
  const mx = window.MaceEditor;
  if (active && active.kind === "editor" && mx) return { n: mx.selected(), of: mx.atoms() };
  const s = M() && M().state.structure;
  return { n: 0, of: active && s ? s.numbers.length : 0 };
}
function structureOf(d) {
  if (!d) return null;
  const ed = edOf(d);
  if (ed) return ed.structure;
  if (d.kind === "editor") return window.MaceEditor && window.MaceEditor.structure();
  return (d.kind === "structure" || d.kind === "trajectory") ? M().state.structure : null;
}
/* The status bar's cursor readout (hover, or the live measurement). */
let cursorText = "";
function setCursor(t) {
  cursorText = t || "";
  const el = $("#st-cursor");
  if (el) el.textContent = cursorText;
}
let connection = "connecting", task = "", cpuState = null;
// Update MACE (1.1.6): the dialog loads on first use; the state for the status-bar item
let updateState = null, updateStarted = false;
const updateUi = () => lazy("update_dialog").then((m) => { m.bind(H); return m; });
function updateDialog(opts) { return updateUi().then((m) => m.open(H, opts)); }
function updateFetch() { return M().api("update").then((d) => { updateState = d; renderStatus(); return d; }).catch(() => null); }
function setTask(t) { task = t; renderStatus(); }
const NEEDS_3D = ["file.exportImage", "file.copyImage", "build.clean", "build.slab", "tool.lasso", "tool.move", "tool.add",
  "tool.replace", "tool.measure", "tool.bond", "build.addAtom"];
function renderStatus() {
  const s = structureOf(active), sel = selection();
  const conn = { ok: ["ok", "Connected"], warn: ["warn", "Reconnecting…"], bad: ["bad", "Disconnected"], connecting: ["", "Connecting…"] }[connection] || ["", connection];
  const items = [`<button class="st-item" id="st-conn" data-tip="Connection to the MACE server" data-desc="Heartbeat over Server-Sent Events"><span class="st-dot ${conn[0]}"></span><span>${conn[1]}</span></button>`];
  if (s) {
    items.push(`<button class="st-item st-num" id="st-sel" data-tip="Selection">${sel.n} of ${sel.of} atoms</button>`);
    const ed = edOf(active), ref = ed && ed.referenceGroup;
    // during a Move grab, the group the move would give (the live preview, spec §2.7)
    const live = ed && E && E.tools && E.tools.grabbing && E.tools.preview.result;
    const y = (live && live.available ? live : s.symmetry) || {};
    // amber, with the group it came from, when an edit lowered the symmetry (spec §1.7)
    const lower = !!(y.available && ref && (y.n_operations_conventional || y.n_operations) < ref.n_operations);
    items.push(`<button class="st-item${lower ? " warn" : ""}" id="st-sg" data-tip="Space group (spglib)" data-desc="${y.available ? esc(y.hall || "") : ""}">${y.available ? `${esc(y.international)} (${y.number})${lower ? ` ↓ from ${esc(ref.international)}` : ""}` : "no space group"}</button>`);
    const L = s.lattice;
    // only the periodic lengths: a slab's c (a polymer's b, c) just spans its atoms
    const per = ["a", "b", "c"].filter((k, i) => !s.pbc || s.pbc[i]);
    if (L && per.length) items.push(`<button class="st-item st-num" id="st-cell" data-tip="Cell parameters">${per.map((k) => `${k} ${L[k].toFixed(3)}`).join(" ")} Å · ${+L.alpha.toFixed(2)} ${+L.beta.toFixed(2)} ${+L.gamma.toFixed(2)}°</button>`);
  }
  if (M() && M().webgl && !M().webgl.ok) items.push(`<span class="st-item warn" id="st-no3d" title="This browser cannot draw 3D here (WebGL is off or not supported)">No 3D</span>`);
  items.push(`<span class="st-spacer"></span><span class="st-item st-num" id="st-cursor">${esc(cursorText)}</span>`);
  if (!PANE) {
    items.push(`<span class="st-panels" role="group" aria-label="Panels">${[["explorer", "panel-left"], ["bottom", "panel-bottom"], ["inspector", "panel-right"]].map(([n, ico]) => {
      const on = panelShown(n), [label, cmd] = PANEL_INFO[n];
      return `<button class="st-item st-panel${on ? " on" : ""}" type="button" id="st-panel-${n}" data-panel-cmd="${cmd}" aria-pressed="${on}" aria-controls="${n}" ` +
        `aria-label="${label}${esc(keyText(cmd))}" data-tip="${on ? "Hide" : "Show"} ${label}" data-tip-cmd="${cmd}">${icon(ico)}</button>`;
    }).join("")}</span>`);
  }
  // a per-process CPU-time limit (plan D18): the chip from 50 % of it
  if (cpuState && cpuState.level) {
    const m = (x) => Math.round(x / 60);
    items.push(`<span class="st-item st-num${cpuState.level === "chip" ? "" : " warn"}" id="st-cpu" data-tip="This node's CPU-time limit per process" ` +
      `data-desc="MACE GUI stops cleanly near it; start it again for a fresh limit">CPU ${m(cpuState.used_s)} of ${m(cpuState.limit_s)} min</span>`);
  }
  if (GR && !GR.pane && GR.sideFocused()) {
    items.push(`<button class="st-item" id="st-group" data-tip="The right editor group has the focus" data-desc="Menu and keyboard commands act on its document; Focus Other Group: Mod+Alt+\\">Right group</button>`);
  }
  // mace gui --slurm: the job this server runs in, and when its time limit comes
  const sl = info && info.slurm;
  if (sl && sl.job) {
    const until = sl.ends ? new Date(sl.ends * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";
    items.push(`<span class="st-item st-num" id="st-slurm" data-tip="MACE GUI runs as SLURM job ${esc(sl.job)} on ${esc(sl.node || "")}" ` +
      `data-desc="It stops cleanly 1 minute before the job's time limit; Sign Out and Stop Server ends the job">Job ${esc(sl.job)}${until ? ` · until ${esc(until)}` : ""}</span>`);
  }
  const run = runningTasks();
  if (run.length) {
    const t = run[0], pct = Math.round((t.progress || 0) * 100), more = run.length > 1 ? ` +${run.length - 1}` : "";
    items.push(`<button class="st-item st-task" id="st-task" data-tip="Background tasks" data-desc="Open the Tasks panel">` +
      `<span class="spinner sm" aria-hidden="true"></span><span class="st-task-title" title="${esc(t.title)}${more}">${esc(t.title)}${more}</span>` +
      `<span class="st-task-bar" role="progressbar" aria-label="${esc(t.title)}" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100"><span style="width:${pct}%"></span></span></button>`);
  } else if (task) items.push(`<button class="st-item" id="st-task">${icon("loader-circle")}<span>${esc(task)}</span></button>`);
  if (storageChip) items.push(`<button class="st-item st-storage ${storageChip.level === "critical" ? "crit" : "warn"}" id="st-storage" data-tip="Storage" data-desc="Open the Cluster view's Storage tab">${icon("triangle-alert")}<span>${esc(storageChip.text)}</span></button>`);
  // Update MACE (1.1.6): a quiet item while an update is available (lazy/update_dialog.js)
  if (updateState && updateState.notice) {
    const nv = ((updateState.report || {}).target || {}).version || "";
    items.push(`<button class="st-item" id="st-update" data-tip="An update of MACE is available" data-desc="Show what changed and update">${icon("download")}<span>MACE ${esc(nv)} available</span></button>`);
  }
  const unread = notifications.unreadCount();
  items.push(`<button class="st-item st-bell" id="st-bell" data-tip="Notifications" data-desc="Every notice of this session" aria-label="${unread ? `${unread} unread notification${unread === 1 ? "" : "s"}` : "Notifications"}">${icon("bell")}<span class="bell-n">${unread || ""}</span></button>`);
  if (jobCounts && (jobCounts.running || jobCounts.pending)) {
    items.push(`<button class="st-item" id="st-jobs" data-tip="Your SLURM jobs in this folder" data-desc="Open the Jobs panel">Jobs: ${jobCounts.running} running · ${jobCounts.pending} pending</button>`);
  }
  if (active) items.push(`<span class="st-item" id="st-doc">${active.kind === "batch" ? "Batch Setup" : `${active.deck ? "Preview" : active.kind === "workflows" ? "View" : active.kind === "text" || active.kind === "volume" || active.kind === "gallery" || (active.plot && (active.plot.source === "modes" || active.plot.kind === "convergence" || active.plot.kind === "map")) ? "Read only" : isDirty(active) ? "Modified" : active.path ? "Saved" : "Unsaved"} · ${esc(KIND_LABEL[active.fileKind] || active.kind)}`}</span>`);
  $("#statusbar").innerHTML = items.join("");
}
window.__maceLost = () => connectionLost();
/* Session expiry and activity (plan T11): only real input counts as use. A
 * touch goes out at most once a minute, and at once on the first input after
 * a pause of a minute or more (before the click's own request, in practice). */
let lastTouch = 0, expiredShown = false;
function touchSession() {
  const now = Date.now();
  if (now - lastTouch < 60000 || expiredShown || lostShown) return;
  lastTouch = now;
  M().api("session/touch", { body: {} }).catch(() => {});
}
for (const ev of ["pointerdown", "keydown", "wheel"]) addEventListener(ev, touchSession, { capture: true, passive: true });
function sessionExpired() {
  if (expiredShown || lostShown) return;
  expiredShown = true;
  if (eventSource) eventSource.close();
  clearTimeout(beatTimer); clearTimeout(errTimer);
  if (window.__maceDrafts) window.__maceDrafts.flush();
  connectUi().then((m) => m.signedOut(H, () => {
    expiredShown = false; lastTouch = Date.now();
    connect(); reconcileTasks();
  })).catch(() => connectionLost());
}
window.__maceExpired = () => sessionExpired();
$("#statusbar").addEventListener("click", (e) => {
  if (e.target.closest("#st-conn")) tunnelDialog();
  else if (e.target.closest("#st-task")) showBottom("tasks");
  else if (e.target.closest("#st-bell")) showNotifications();
  else if (e.target.closest("#st-jobs")) showBottom("jobs");
  else if (e.target.closest("#st-cell, #st-sg")) { if (!panelShown("inspector")) togglePanel("inspector"); }
  else if (e.target.closest("#st-storage")) openCluster("storage");
  else if (e.target.closest("#st-update")) updateDialog();
});
new MutationObserver(() => {
  const tip = $("#hover-tip");
  if (tip) setCursor(tip.classList.contains("hidden") ? "" : tip.textContent);
}).observe(document.getElementById("hover-tip") || document.body, { attributes: true, childList: true, characterData: true, subtree: true });

/* ================================================================ connection */
let beatTimer = 0;
let errTimer = 0;     // armed by the first error after a beat; later retries must not push it back
let eventSource = null, stopInitiator = false;
/* The server stopped on purpose (Sign Out and Stop Server): no reconnecting,
 * no "lost the connection" banner; edits stay as drafts. */
function serverStopped() {
  lostShown = true; leaving = true;
  if (eventSource) eventSource.close();
  clearTimeout(beatTimer); clearTimeout(errTimer);
  if (window.__maceDrafts) window.__maceDrafts.flush();
}
function connect() {
  let es;
  try { es = new EventSource("api/events"); } catch (e) { connection = "bad"; renderStatus(); return; }
  eventSource = es;
  const beat = () => {
    if (lostShown) return;            // a restarted server: this page's token is gone for good
    connection = "ok"; renderStatus(); $("#conn-banner").classList.add("hidden");
    clearTimeout(errTimer); errTimer = 0;
    clearTimeout(beatTimer);
    beatTimer = setTimeout(lost, 2 * 1000 * (info ? info.heartbeat : 10));
  };
  es.addEventListener("hello", (e) => {
    try {
      const d = JSON.parse(e.data);
      if (d.client) { window.__maceClient = d.client; clientId = d.client; }
      // a folder switch while the event stream was down: the hello carries the generation
      if (d.gen !== undefined && window.__maceRootGen !== undefined && String(d.gen) !== window.__maceRootGen && !leaving)
        connectUi().then((m) => m.folderChanged(H, d));
    } catch (err) { /* old server */ }
    beat(e);
    reconcileTasks();
    claimHeld();
    // the dialog's module loads only for a result to report (or on a click)
    if (!updateStarted) { updateStarted = true; updateFetch().then((d) => { if (d && d.result) updateUi().then((m) => m.startup(H)); }); }
    document.dispatchEvent(new CustomEvent("mace-client", { detail: clientId }));
  });
  es.addEventListener("task", (e) => { try { taskEvent(JSON.parse(e.data)); } catch (err) { /* malformed */ } });
  // the session expired (or was signed out elsewhere): no reconnecting
  es.addEventListener("expired", () => { es.close(); sessionExpired(); });
  es.addEventListener("cpu", (e) => {
    try { cpuState = JSON.parse(e.data); } catch (err) { return; }
    renderStatus();
    if (cpuState.level === "warn") connectUi().then((m) => m.cpuWarning(H, cpuState));
  });
  es.addEventListener("walltime", (e) => { let d = {}; try { d = JSON.parse(e.data); } catch (err) { /* */ } connectUi().then((m) => m.walltimeWarning(H, d)); });
  es.addEventListener("idle-warning", (e) => { let d = {}; try { d = JSON.parse(e.data); } catch (err) { /* */ } connectUi().then((m) => m.idleWarning(H, d)); });
  // Sign Out and Stop Server (plan T10): from this tab or another one
  es.addEventListener("server-stopping", (e) => {
    let d = {}; try { d = JSON.parse(e.data); } catch (err) { /* */ }
    serverStopped();
    connectUi().then((m) => m.stoppedPage(H, d.reason === "browser" ? !stopInitiator : d.reason));
  });
  // Update MACE (update_gui.py): a check's result, and Update Now's progress (the in-place restart)
  es.addEventListener("update", () => { updateFetch(); });
  es.addEventListener("update-progress", (e) => { let d = {}; try { d = JSON.parse(e.data); } catch (err) { /* */ } updateUi().then((m) => m.progress(H, d)); });
  es.addEventListener("stop-pending", (e) => { let d = {}; try { d = JSON.parse(e.data); } catch (err) { /* */ } window.__maceStopPending = true; connectUi().then((m) => m.stopPending(H, d)); });
  es.addEventListener("stop-canceled", () => { window.__maceStopPending = false; connectUi().then((m) => m.stopPending(H, null)); });
  // File > Open Folder in another tab: this one follows (plan T6)
  es.addEventListener("root-changed", (e) => {
    let d = {}; try { d = JSON.parse(e.data); } catch (err) { /* */ }
    if (String(d.gen) !== window.__maceRootGen && !leaving) connectUi().then((m) => m.folderChanged(H, d));
  });
  es.addEventListener("task-removed", (e) => { try { const d = JSON.parse(e.data); taskStore.delete(d.id); tasksChanged(); } catch (err) { /* */ } });
  es.addEventListener("jobs.finished", (e) => { try { jobFinished(JSON.parse(e.data)); } catch (err) { /* malformed */ } if (workflowView) workflowView.changed(); });
  // a step of a workflow under this folder changed state (workflow.py's watcher on the jobs poll)
  es.addEventListener("workflow.changed", () => { window.__maceWorkflowEvents = (window.__maceWorkflowEvents || 0) + 1; if (workflowView) workflowView.changed(); });
  // a file appeared in or left the folder (a job's output, a copy from a shell): the Explorer lists it again
  es.addEventListener("files-changed", (e) => {
    let d = {}; try { d = JSON.parse(e.data); } catch (err) { /* */ }
    if (d.gen === undefined || String(d.gen) === window.__maceRootGen) filesChanged();
  });
  es.addEventListener("heartbeat", (e) => { window.__maceBeats = (window.__maceBeats || 0) + 1; beat(e); });
  // the browser retries every few seconds while the tunnel or server is down:
  // the banner comes 20 s after the first error, not 20 s after the last
  es.onerror = () => {
    connection = "warn"; renderStatus(); clearTimeout(beatTimer);
    if (!errTimer) errTimer = setTimeout(() => { errTimer = 0; lost(); }, 20000);
  };
}
/* files-changed: one /api/files at a time, the last change always listed */
let filesLoading = false, filesAgain = false;
function filesChanged() {
  if (leaving) return;
  if (filesLoading) { filesAgain = true; return; }
  filesLoading = true;
  M().loadFiles().catch(() => {}).finally(() => {
    filesLoading = false;
    if (filesAgain) { filesAgain = false; filesChanged(); }
  });
}
function lost() {
  if (lostShown) return;
  connection = "bad"; renderStatus();
  const b = $("#conn-banner");
  b.innerHTML = `${icon("wifi-off")}<span>Lost the connection to the MACE server. Reconnecting… If the ssh tunnel closed, rerun it:</span>` +
    `<code>${esc(info && info.tunnel ? info.tunnel : "ssh -N -L PORT:127.0.0.1:PORT user@node")}</code>`;
  b.classList.remove("hidden");
}

/* ================================================================ tasks (spec §8.4.8, plan B4) */
/* What the server runs in the background, from /api/events: the status
 * bar's progress item and the Tasks panel. A task the page waits for
 * (awaitTask) resolves with its result, fetched once when it is done. */
let clientId = null;
const taskStore = new Map(), taskWaiters = new Map();
const FINAL = new Set(["done", "failed", "canceled", "lost"]);
const runningTasks = () => [...taskStore.values()].filter((t) => t.state === "running" || t.state === "queued")
  .sort((a, b) => (b.started || 0) - (a.started || 0));
function taskEvent(t) {
  const old = taskStore.get(t.id);
  if (old && FINAL.has(old.state) && old.state === t.state) return;      // a final state once
  // a record older than the final state already here (the 202 answer of a task whose
  // "done" event came first) must not make it look running again: its waiter would wait forever
  if (old && FINAL.has(old.state) && !FINAL.has(t.state)) return;
  taskStore.set(t.id, { ...(old || {}), ...t, t0: (old && old.t0) || performance.now() });
  const w = taskWaiters.get(t.id);
  if (w) w.update(taskStore.get(t.id));
  tasksChanged();
}
async function reconcileTasks() {
  let list;
  try { list = (await M().api("tasks")).tasks; } catch (e) { return; }
  const seen = new Set(list.map((t) => t.id));
  for (const t of list) taskEvent(t);
  // a task this page followed that the server no longer has: it restarted or let it go
  for (const [id, t] of taskStore) if (!seen.has(id) && !FINAL.has(t.state)) taskEvent({ ...t, state: "lost", message: "Lost when the server stopped" });
}
let tasksFrame = 0;
function tasksChanged() {
  cancelAnimationFrame(tasksFrame);
  tasksFrame = requestAnimationFrame(() => { renderStatus(); renderTasks(); });
}
let tasksPanel = null;
function renderTasks() {
  const host = $(".bottom-pane.tasks");
  if (!host) return;
  if (!taskStore.size && !tasksPanel) {
    host.innerHTML = `<p class="empty-note" id="tasks-note">No tasks. Builds, Clean and big files show their progress here.</p>`;
    return;
  }
  (tasksPanel || (tasksPanel = import("./editor/tasks_panel.js"))).then((m) => {
    m.bind(host, { cancel: (id) => cancelTask(id), retry: (id) => retryTask(id), remove: (id) => removeTask(id),
      copy: (id) => { const t = taskStore.get(id); copyText(t ? `${t.title}: ${t.error || ""}\n${t.traceback || ""}` : "", "Error copied"); },
      open: (id) => { const t = taskStore.get(id); if (t && t.open) t.open(); } });
    m.render(host, taskStore, { esc, icon });
  }).catch(() => {});
}
function cancelTask(id) { return M().api(`tasks/${id}/cancel`, { body: {} }).then((r) => taskEvent(r.task)).catch((e) => toast(e.message, "error")); }
function removeTask(id) {
  taskStore.delete(id); tasksChanged();
  M().api(`tasks/${id}/remove`, { body: {} }).catch(() => {});
}
async function retryTask(id) {
  const old = taskStore.get(id);
  try {
    const r = await M().api(`tasks/${id}/retry`, { body: {}, withStatus: true });
    const t = r.data.task;
    if (t) { taskEvent(t); if (old && old.open) taskStore.get(t.id).open = old.open; }
    if (old && old.retry) old.retry(r);
  } catch (e) { toast(e.message, "error"); }
}
/** A task's result, once: resolves when it is done (rejects when it failed,
 * was canceled or was lost); onUpdate(record) on every event. */
function awaitTask(rec, onUpdate = null) {
  taskEvent(rec);
  return new Promise((resolve, reject) => {
    const settle = async (t) => {
      if (onUpdate) onUpdate(t);
      if (!FINAL.has(t.state)) return;
      taskWaiters.delete(t.id);
      if (t.state === "done") {
        try { resolve(await M().api(`tasks/${t.id}/result`)); } catch (e) { reject(e); }
      } else {
        const err = new Error(t.state === "failed" ? (t.error || "the task failed") : t.state === "lost" ? "Lost when the server stopped" : "Canceled");
        err.task = t; err.state = t.state;
        reject(err);
      }
    };
    taskWaiters.set(rec.id, { update: settle });
    const now = taskStore.get(rec.id);
    if (now && FINAL.has(now.state)) settle(now);
    else if (now && onUpdate) onUpdate(now);           // the record so far (its events may have come before the 202)
  });
}
/** A route that may detach: the answer at once, or (202) the task's result when it is done. */
async function apiTask(path, body, onUpdate = null, extra = {}) {
  const r = await M().api(path, { body, withStatus: true, ...extra });
  if (r.status === 202 && r.data && r.data.task) return awaitTask(r.data.task, onUpdate);
  if (r.data && r.data.task) taskEvent(r.data.task);
  return r.data;
}

/* ================================================================ files (Phase 4, T3) */
/* Export Structure: a download to this computer (over ssh -L, the laptop),
 * never a write into the served folder (that is Save As). */
const EXPORT_EXT = { cif: "cif", poscar: "vasp", xyz: "xyz", extxyz: "extxyz" };
function exportName(d, fmt) {
  const t = (d && d.title) || "structure";
  const stem = t.replace(/\s*\(dropped( \d+)?\)$/, (m, n) => `_dropped${n ? n.trim() : ""}`).replace(/\.[A-Za-z0-9]+(_dropped\d*)?$/, "$1")
    .replace(/\s+/g, "_").replace(/[^A-Za-z0-9._-]+/g, "_") || "structure";
  return `${stem}.${EXPORT_EXT[fmt]}`;
}
async function waitIdle(e, ms = 20000) {
  const end = performance.now() + ms;
  while (e && (e.pending > 0 || e.waiting) && performance.now() < end) await new Promise((r) => setTimeout(r, 50));
}
/** What a document is, for a builder or an export: its session at the current
 * revision, its held structure, or its file (and the run's step). */
async function sourceOf(d) {
  const e = edOf(d);
  if (d.held) return { held: d.held };
  if (e && e.srv.session) { await waitIdle(e); return { session: e.srv.session, rev: e.srv.rev }; }
  if (d.path) return { path: d.path, frame: d.frame >= 0 ? d.frame : -1 };
  if (e) { await E.editing.ensureSession(e); await waitIdle(e); if (e.srv.session) return { session: e.srv.session, rev: e.srv.rev }; }
  throw new Error("This document has no structure the server can read");
}
async function exportStructure(fmt, mode = "p1") {
  const d = active;
  if (!d) return;
  try {
    const source = await sourceOf(d);
    const r = await M().api("structure/export", { body: { source, format: fmt, cif_mode: mode }, blob: true });
    const name = exportName(d, fmt);
    const url = URL.createObjectURL(new Blob([await r.blob.arrayBuffer()], { type: "application/octet-stream" }));
    const a = document.createElement("a");
    a.href = url; a.download = name; a.hidden = true;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
    toast(`Downloaded ${name}`, "success");
    return name;
  } catch (e) { toast(`Export: ${e.message}`, "error"); return null; }
}

/* "Changed on disk" for documents with no unsaved changes (spec §8.4.2):
 * checked when the page becomes visible or gets the focus, at most every
 * 2 s, never on a timer; one /api/files request. */
// -Infinity, never 0: performance.now() starts at 0 on a page load, so 0 would throttle a page's first 2 s
let diskCheckAt = -Infinity;
const diskIgnored = new Map();          // path -> stamp the user said to ignore
function stampOf(d) {
  const e = edOf(d);
  if (e) return e.diskStamp || (d.data && d.data.file) || null;
  return d.data && d.data.file ? d.data.file : null;
}
async function checkDisk() {
  const now = performance.now();
  if (now - diskCheckAt < 2000 || !docs.some((d) => d.path)) return;
  diskCheckAt = now;
  try { await M().loadFiles(); } catch (e) { return; }
  renderDiskBanner();
}
function changedDocs() {
  const out = [];
  for (const d of docs) {
    // a live run updates itself (PLAN_analysis C18.4): no "changed on disk" banner for it
    if (!d.path || d.untitled || isDirty(d) || d.frame >= 0 || (M().isLive && M().isLive(d.path))) continue;
    const f = M().fileByPath(d.path), st = stampOf(d);
    if (!f || !st || !f.mtime_ns) continue;
    const same = String(f.mtime_ns) === String(st.mtime_ns) && Number(f.size) === Number(st.size);
    const ign = diskIgnored.get(d.path);
    if (!same && !(ign && ign === `${f.mtime_ns}:${f.size}`)) out.push({ d, f });
  }
  return out;
}
function renderDiskBanner() {
  const b = $("#disk-banner");
  if (!b) return;
  const list = changedDocs();
  if (!list.length) { b.hidden = true; return; }
  const { d, f } = list[0];
  b.innerHTML = `${icon("refresh-cw")}<span>${esc(d.title)} changed on disk.${list.length > 1 ? ` (${list.length - 1} more)` : ""}</span>` +
    `<button class="tb-btn-text" data-v="reload">Reload</button><button class="tb-btn-text" data-v="ignore">Ignore</button>`;
  b.hidden = false;
  b.querySelector('[data-v="reload"]').onclick = () => { b.hidden = true; reloadDoc(d).finally(renderDiskBanner); };
  b.querySelector('[data-v="ignore"]').onclick = () => { diskIgnored.set(d.path, `${f.mtime_ns}:${f.size}`); renderDiskBanner(); };
}
async function reloadDoc(d) {
  const i = docs.indexOf(d), path = d.path, wasActive = active === d;
  closeDoc(d, true);
  const nd = await openDoc(path);
  const n = docs.indexOf(nd);
  if (nd && n >= 0 && i >= 0 && n !== i) { docs.splice(n, 1); docs.splice(Math.min(i, docs.length), 0, nd); renderTabs(); }
  if (nd && !wasActive) afterSwitch();
  return nd;
}
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") checkDisk(); });
addEventListener("focus", () => checkDisk());

/* Prefetch (spec §8.5): the pointer resting 150 ms on a structure file
 * fetches its packed structure, once per file version, and the open that
 * follows takes it from here (no request); a later open of the same
 * version revalidates with the ETag (a 304). At most 16 kept, for 60 s. */
const prefetched = new Map();          // "path:mtime_ns:size" -> {t, promise}
const prefetchKey = (path) => { const f = M().fileByPath(path); return f ? `${path}:${f.mtime_ns}:${f.size}` : null; };
function prefetchStructure(path) {
  const f = M().fileByPath(path);
  if (!f || !["cif", "d12"].includes(f.kind) || f.size > 5 * 1024 * 1024 || docs.some((d) => d.path === path)) return;
  const key = prefetchKey(path);
  if (prefetched.has(key)) return;
  const promise = Promise.all([M().api("structure?" + M().q({ path, packed: 1 })), import("./geom/core.js")])
    .then(([r, core]) => core.decodePacked(r));
  promise.catch(() => prefetched.delete(key));
  prefetched.set(key, { t: performance.now(), promise });
  while (prefetched.size > 16) prefetched.delete(prefetched.keys().next().value);
}
function takePrefetched(path) {
  const key = prefetchKey(path), hit = key && prefetched.get(key);
  if (!hit) return null;
  prefetched.delete(key);
  return performance.now() - hit.t < 60000 ? hit.promise : null;
}

/* Drag and drop (spec §6, plan T9): a full-window target while files are
 * dragged over the page; each dropped file goes through /api/parse (never
 * written into the folder) and opens as an untitled, modified document. */
const DROP_EXT = /\.(cif|vasp|poscar|xyz|extxyz|d12|out|log)$/i, DROP_BASE = /^(POSCAR|CONTCAR)([._-]|$)/;
const DROP_MAX = 50 * 1024 * 1024, DROP_FILES = 8;
let dragDepth = 0, dropEl = null;
const hasFiles = (e) => !!(e.dataTransfer && [...(e.dataTransfer.types || [])].includes("Files"));
function dropOverlay(show) {
  if (show && !dropEl) {
    dropEl = document.createElement("div");
    dropEl.id = "drop-overlay";
    dropEl.setAttribute("role", "status");
    dropEl.innerHTML = `<div class="drop-box">${icon("upload")}<span>Drop to open (CIF, POSCAR, XYZ, d12, out)</span></div>`;
    document.body.appendChild(dropEl);
  }
  if (dropEl) dropEl.hidden = !show;
  if (!show) dragDepth = 0;
}
addEventListener("dragenter", (e) => { if (!hasFiles(e)) return; e.preventDefault(); dragDepth++; dropOverlay(true); });
addEventListener("dragover", (e) => { if (!hasFiles(e)) return; e.preventDefault(); e.dataTransfer.dropEffect = "copy"; });
addEventListener("dragleave", (e) => {
  if (!hasFiles(e)) return;
  dragDepth--;
  if (dragDepth <= 0 || !e.relatedTarget) dropOverlay(false);
});
addEventListener("drop", (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dropOverlay(false);
  dropFiles(e.dataTransfer);
});
addEventListener("keydown", (e) => { if (e.key === "Escape" && dropEl && !dropEl.hidden) { dropOverlay(false); e.stopPropagation(); } }, true);
async function dropFiles(dt) {
  const items = [...(dt.items || [])], files = [...(dt.files || [])];
  const isDir = (k) => { const it = items[k]; const en = it && it.webkitGetAsEntry ? it.webkitGetAsEntry() : null; return !!(en && en.isDirectory); };
  const list = files.map((f, k) => ({ f, dir: isDir(k) || (f.size === 0 && !f.type && !DROP_EXT.test(f.name)) }));
  if (list.length > DROP_FILES) toast(`Opened ${DROP_FILES} of ${list.length} files; drop the rest again`, "info");
  for (const { f, dir } of list.slice(0, DROP_FILES)) {
    if (dir) { toast("Folders can't be dropped; open files from the Explorer", "info"); continue; }
    if (!DROP_EXT.test(f.name) && !DROP_BASE.test(f.name)) {
      const ext = (f.name.match(/\.[^.]+$/) || [""])[0] || f.name;
      toast(`MACE can't open ${ext} files`, "info");
      continue;
    }
    if (f.size > DROP_MAX) { toast(`${f.name} is over 50 MB; open it from the folder instead`, "info"); continue; }
    await openParsed(f.name, f, `Dropped ${f.name}`, { dropped: true });
  }
}
/** A dropped file's tab title: its name, or "MgO.cif (dropped)" when the
 * folder or an open tab already has that name ("(dropped 2)" next). */
function dropTitle(name) {
  const taken = (t) => docs.some((d) => d.title === t) || (M().state.files || []).some((f) => f.name === t);
  if (!taken(name)) return { title: name, save: name.replace(/\.[^.]+$/, "") };
  const stem = name.replace(/\.[^.]+$/, "");
  for (let k = 1; ; k++) {
    const t = `${name} (dropped${k > 1 ? " " + k : ""})`;
    if (!taken(t)) return { title: t, save: `${stem}_dropped${k > 1 ? k : ""}` };
  }
}
let clipCount = 0;
/** Parse a file or pasted text on the server and open it as an untitled document. */
async function openParsed(name, blob, label, opts = {}) {
  const { title, save } = opts.title ? { title: opts.title, save: opts.save } : dropTitle(name);
  let r;
  try {
    r = await M().api("parse?" + M().q({ box: opts.box === false ? 0 : 1 }), { raw: blob,
      headers: { "X-File-Name": encodeURIComponent(name), "X-Mace-Label": encodeURIComponent(label) } });
  } catch (e) {
    toast(e.message, "error");
    const text = blob.size <= 5 * 1024 * 1024 && blob.text ? await blob.text().catch(() => null) : null;
    addProblem({ severity: "error", source: name, message: e.message,
      action: text !== null ? { label: "Open as Text", run: () => openTextDoc(name, text) } : null });
    return null;
  }
  let d;
  const extra = { dropped: !!opts.dropped, saveName: save, source: r.format_name };
  if (r.session) {
    await loadEditor();
    d = await openUntitled(r.session, title, extra);
    const e = edOf(d);
    d.meta = r.session.doc_meta || null;
    if (e) { e.savedRev = -1; e.docMeta = d.meta; E.drafts.write(e); E.stateChanged(e); }
  } else if (r.held) {
    const core = await import("./geom/core.js");
    d = await openHeld(core.decodePacked(r.structure), title, r.held, r.structure.doc_meta || null, extra);
    if (r.reason) toast(`${title} opens view-only: ${r.reason}`, "info");
  }
  if (d) afterSwitch();
  return d;
}
/** A read-only text document for a file's text (a dropped file that did not parse). */
function openTextDoc(name, text) {
  saveView(active);
  const d = { id: nextId++, path: null, srcPath: name, kind: "text", fileKind: "", title: name, text: text.slice(0, MAX_TEXT_DOC), truncated: text.length > MAX_TEXT_DOC };
  docs.push(d);
  return activate(d);
}
const MAX_TEXT_DOC = 400000;

/* Paste (plan B12): Mod+V and Mod+Shift+V go to the browser (no prompt),
 * whose paste event carries the text: Phase 3's Copy block pastes atoms into
 * the structure; other text (or Mod+Shift+V) opens Import from Clipboard.
 * A paste into a text field is the field's. */
let pasteAsImport = 0;          // when Mod+Shift+V went down: its paste event follows at once
document.addEventListener("paste", onPaste);
function onPaste(e) {
  const t = e.target;
  if (isTextField(t) || (t && t.closest && t.closest("dialog"))) return;
  // the primary selection a middle click in a viewer brings (Linux): never a paste
  if (performance.now() - midUpAt < MID_PASTE_MS) return;
  const text = e.clipboardData ? e.clipboardData.getData("text/plain") : "";
  const imp = !!pasteAsImport && performance.now() - pasteAsImport < 1000;
  pasteAsImport = 0;
  if (!text) return;
  e.preventDefault();
  pasteText(text, imp);
}
/* The middle button in a 3D viewer pans (viewport/pointer.js) or zooms
 * (3Dmol's own viewers: optimization, normal mode, volume); on Linux it is
 * also the primary selection's paste, which Chromium runs on the mouseup and
 * Firefox on the auxclick. Its mousedown, pointerup, mouseup and auxclick are
 * canceled there: preventDefault only, so pointer.js and 3Dmol (whose mouseup
 * listener is on the body and ignores defaultPrevented) still get every
 * event. Its pointerdown is left alone: canceling it would keep the mousedown
 * from 3Dmol. A paste that still comes within MID_PASTE_MS of the release is
 * dropped by onPaste, after a middle click anywhere outside a text field;
 * Mod+V clears that window. */
const MID_PASTE_MS = 500;
let midDown = false, midTail = false, midUpAt = -Infinity;
function onMiddleButton(e) {
  if (e.button !== 1) return;
  const inViewer = !!e.target && !!e.target.closest && !!e.target.closest(".viewer");
  if (e.type === "pointerdown" || e.type === "mousedown") {
    midDown = inViewer; midTail = false;
    if (inViewer && e.type === "mousedown") e.preventDefault();       // no focus move, no autoscroll
    return;
  }
  if (e.type === "pointercancel") { midDown = false; return; }
  // the release of a press that began in a viewer (pointer capture retargets it), then its mouseup and auxclick
  if (!midDown && !midTail && !inViewer) {
    // elsewhere outside a text field (a toolbar button, a tab) the primary selection's paste is dropped too
    if (!isTextField(e.target)) midUpAt = performance.now();
    return;
  }
  e.preventDefault();
  if (e.type !== "auxclick") { midDown = false; midTail = true; midUpAt = performance.now(); }
}
for (const type of ["pointerdown", "mousedown", "pointerup", "mouseup", "auxclick", "pointercancel"]) addEventListener(type, onMiddleButton, true);
/** Copied atoms paste into the open structure; anything else opens Import from Clipboard. */
function pasteText(text, asImport = false) {
  if (!asImport && isAtomsBlock(text) && active && active.kind === "structure" && edOf(active) && !active.viewOnly) return S.ed("edit.pasteText", { text });
  return importClipboard(text || "");
}
/** Phase 3's Copy block: a count, "<name>: element, fractional x y z, Cartesian x y z (Å)", then the atoms. */
const isAtomsBlock = (text) => /^\s*\d+\s*\n[^\n]*: element, fractional x y z, Cartesian x y z/.test(text);
function importClipboard(text = "") {
  return lazy("import_dialog").then((m) => m.open(H, text, (txt, fmt, box) => {
    clipCount++;
    const ext = { cif: "cif", poscar: "vasp", xyz: "xyz", extxyz: "extxyz" }[fmt] || "txt";
    const blob = new Blob([txt], { type: "text/plain" });
    return openParsed(`clipboard.${ext}`, blob, "Imported from Clipboard", { box, title: `Clipboard ${clipCount}`, save: `clipboard_${clipCount}` });
  }));
}
/** The menu's Paste and Import from Clipboard: the async clipboard (Firefox may ask first); the dialog when it is refused. */
async function readClipboard() {
  try { return navigator.clipboard && navigator.clipboard.readText ? await navigator.clipboard.readText() : ""; } catch (e) { return ""; }
}

/* Held structures (view-only built or dropped ones, plan B5): leased to this
 * page's event stream; a reload claims the open ones back, closing a tab
 * releases its entry. */
const HELD_KEY = "mace.gui.held";
function heldList() { try { return JSON.parse(sessionStorage.getItem(HELD_KEY) || "[]"); } catch (e) { return []; } }
function saveHeld() {
  try { sessionStorage.setItem(HELD_KEY, JSON.stringify(docs.filter((d) => d.held).map((d) => ({ id: d.held, title: d.title, meta: d.meta || null })))); } catch (e) { /* private */ }
}
function releaseHeld(d) {
  M().api("held/release", { body: { ids: [d.held] } }).catch(() => {});
  saveHeld();
}
let heldClaimed = false;
async function claimHeld() {
  const mine = docs.filter((d) => d.held).map((d) => d.held), stored = heldClaimed ? [] : heldList();
  heldClaimed = true;
  const ids = [...new Set([...mine, ...stored.map((x) => x.id)])];
  if (!ids.length) return;
  let got = [];
  try { got = (await M().api("held/claim", { body: { ids } })).claimed; } catch (e) { return; }
  // after a reload: the documents come back, view-only, from the server's copy
  for (const x of stored) {
    if (!got.includes(x.id) || docs.some((d) => d.held === x.id)) continue;
    try {
      const [r, core] = await Promise.all([M().api("held?" + M().q({ id: x.id })), import("./geom/core.js")]);
      await openHeld(core.decodePacked(r), x.title, x.id, x.meta);
    } catch (e) { /* gone */ }
  }
  for (const d of docs) if (d.held && !got.includes(d.held)) markReleased(d);
  saveHeld();
}
function markReleased(d) {
  d.released = true;
  addProblem({ severity: "warning", source: d.title, doc: d.id,
    message: `This built structure was released by the server. Rebuild it${d.meta && d.meta.provenance && d.meta.provenance.parent ? ` from ${d.meta.provenance.parent}` : ""}`,
    action: d.rebuild ? { label: "Rebuild", run: () => d.rebuild() } : null });
}
/** A held structure as a view-only document. */
async function openHeld(payload, title, id, meta = null, extra = {}) {
  const d = await openUntitled(payload, title, extra);
  d.held = id;
  d.meta = meta;
  saveHeld();
  return d;
}

/* ================================================================ problems (§2A) */
/* The Problems panel: {severity, source, message, action, doc}, newest first;
 * a document's rows go when it closes. */
const problems = [];
function addProblem(p) {
  problems.unshift({ severity: "error", source: "", message: "", ...p, t: Date.now() });
  problems.length = Math.min(problems.length, 200);
  renderProblems();
}
function clearProblems(docId, source = null) {
  for (let i = problems.length - 1; i >= 0; i--) if (problems[i].doc === docId && (!source || problems[i].source === source)) problems.splice(i, 1);
  renderProblems();
}
/* Plan E13: the pane shows Problems (rows code added, plus every warning and error notice) or
 * All Notifications (the session's history, newest first). */
let problemView = "problems";
function problemRows() {
  if (problemView === "all") {
    return notifications.all().map((n) => ({ severity: n.kind === "success" ? "info" : n.kind, source: new Date(n.t).toLocaleTimeString(),
      message: n.text, count: n.count, note: true }));
  }
  const seen = new Set(problems.map((p) => p.message));
  const fromToasts = notifications.all().filter((n) => (n.kind === "warning" || n.kind === "error") && !seen.has(n.text) &&
    !problems.some((p) => n.text.endsWith(p.message)))
    .map((n) => ({ severity: n.kind, source: "Notice", message: n.text, count: n.count, t: n.t }));
  return [...problems.map((p, k) => ({ ...p, k })), ...fromToasts].sort((a, b) => (b.t || 0) - (a.t || 0));
}
function renderProblems() {
  const host = $(".bottom-pane.problems");
  if (!host) return;
  const rows = problemRows();
  const filter = `<div class="problem-filter" role="group" aria-label="Show">` +
    `<button class="tb-btn-text" data-pview="problems" aria-pressed="${problemView === "problems"}">Problems</button>` +
    `<button class="tb-btn-text" data-pview="all" aria-pressed="${problemView === "all"}">All Notifications</button>` +
    (problemView === "all" && rows.length ? `<button class="tb-btn-text" data-pview="clear">Clear</button>` : "") + `</div>`;
  if (!rows.length) {
    host.innerHTML = filter + `<p class="empty-note" id="problems-note">${problemView === "all" ? "No notifications." : "No problems."}</p>`;
    return;
  }
  host.innerHTML = filter + `<ul class="problem-list" role="list">${rows.map((p) =>
    `<li class="problem s-${esc(p.severity)}">${icon(p.severity === "warning" ? "triangle-alert" : p.severity === "info" ? "info" : "circle-alert")}` +
    `<span class="p-src">${esc(p.source)}</span><span class="p-msg" title="${esc(p.message)}">${esc(p.message)}</span>` +
    `${p.count > 1 ? `<span class="p-count">×${p.count}</span>` : ""}` +
    `${p.action && p.k !== undefined ? `<button class="tb-btn-text" data-problem="${p.k}">${esc(p.action.label)}</button>` : ""}</li>`).join("")}</ul>`;
}
/** Problems › All Notifications, shown (the "+N more" of the toasts, the status-bar bell). */
function showNotifications() {
  problemView = "all";
  notifications.markRead();
  shell.classList.remove("doc-no-bottom");          // shown even with no document open
  showBottom("problems");
  renderProblems();
}
notifications.subscribe(() => { renderProblems(); renderBell(); });
function renderBell() {
  const b = $("#st-bell");
  if (!b) return;
  const n = notifications.unreadCount();
  b.querySelector(".bell-n").textContent = n ? String(n) : "";
  b.setAttribute("aria-label", n ? `${n} unread notification${n === 1 ? "" : "s"}` : "Notifications");
}
$("#bottom-body") && $("#bottom-body").addEventListener("click", (e) => {
  const v = e.target.closest("[data-pview]");
  if (v) {
    if (v.dataset.pview === "clear") notifications.clear();
    else { problemView = v.dataset.pview; if (problemView === "all") notifications.markRead(); }
    renderProblems(); renderBell();
    return;
  }
  const b = e.target.closest("[data-problem]");
  if (b) { const p = problems[Number(b.dataset.problem)]; if (p && p.action) p.action.run(); }
});

/* ================================================================ explorer extras */
/* Recent files (plan T10): per served folder under the server's dir_hash (the
 * key drafts use), as {path, t, pinned}; pinned ones sit on top and don't
 * count toward recentLimit. The old key (a hash of the root text) is moved
 * over once. */
const recentKey = () => `mace.gui.recent.${dirHash()}`;
function legacyRecentKey() {
  let h = 0;
  for (const ch of (info ? info.root : "")) h = (h * 31 + ch.charCodeAt(0)) | 0;
  return "mace.gui.recent." + (h >>> 0).toString(36);
}
/* The Recent list lives in the profile on the server (profile.js: recent.files.<dirHash>),
 * so it follows you to a new port or browser. A pre-1.1.5 browser's lists are
 * uploaded once: the per-folder key (profile.migrateLegacy) and, before it,
 * the older key named by a hash of the root text. */
function migrateRecent() {
  try {
    const old = legacyRecentKey(), key = recentKey();
    if (old === key || profile.hasRecentFiles()) return;
    const v = localStorage.getItem(old);    // ls-key: mace.gui.recent.
    if (v === null || localStorage.getItem(key) !== null) return;       // ls-key: mace.gui.recent.
    const list = JSON.parse(v || "[]");
    localStorage.setItem(key, JSON.stringify(list.map((p) => (typeof p === "string" ? { path: p, t: 0, pinned: false } : p))));    // ls-key: mace.gui.recent.
  } catch (e) { /* private window or bad JSON: start empty */ }
}
/** [{path, t, pinned}], pinned first, then most recent first. */
function recentEntries() {
  const r = profile.recentFiles().filter((x) => x && x.path);
  return [...r.filter((x) => x.pinned), ...r.filter((x) => !x.pinned)];
}
const recentList = () => recentEntries().map((x) => x.path);
function saveRecent(list) {
  profile.setRecentFiles(list);
  renderRecentSection();
}
function pushRecent(path) {
  const r = recentEntries(), old = r.find((x) => x.path === path);
  const rest = r.filter((x) => x.path !== path);
  const entry = { path, t: Date.now(), pinned: !!(old && old.pinned) };
  const pinned = [...(entry.pinned ? [entry] : []), ...rest.filter((x) => x.pinned)];
  const unpinned = [...(entry.pinned ? [] : [entry]), ...rest.filter((x) => !x.pinned)].slice(0, settings.get("recentLimit"));
  saveRecent([...pinned, ...unpinned]);
}
function pinRecent(path, on) {
  const r = recentEntries().map((x) => (x.path === path ? { ...x, pinned: on === undefined ? !x.pinned : !!on } : x));
  saveRecent([...r.filter((x) => x.pinned), ...r.filter((x) => !x.pinned)]);
}
function removeRecent(path) { saveRecent(recentEntries().filter((x) => x.path !== path)); }
const isMissing = (p) => !!(M() && M().state.files && M().state.files.length) && !M().fileByPath(p);
/** One recent row (the welcome screen, the explorer's RECENT section). */
function recentRow(x) {
  const miss = isMissing(x.path), f = M() && M().fileByPath(x.path);
  // an output or log shows what its content says (FREQ, log, ...) once the explorer knows it
  const chip = f && (f.kind === "out" || f.kind === "log") && M().chipOf ? M().chipOf(f) : null;
  const kind = chip ? chip.cls : f ? f.kind : "", lbl = chip ? chip.label : KIND_LABEL[kind] || "file";
  return `<button class="file recent-row${miss ? " missing" : ""}" data-recent="${esc(x.path)}" title="${esc(miss ? `${x.path} (Missing)` : x.path)}"${miss ? ' aria-disabled="true"' : ""}>` +
    `<span class="kb ${esc(kind)}">${esc(lbl)}</span>${M().middleName(x.path)}` +
    `${x.pinned ? `<span class="pin" aria-label="Pinned">${icon("pin")}</span>` : ""}${miss ? '<span class="miss">Missing</span>' : ""}</button>`;
}
/* MATERIALS.DB (spec §1.1): the folder's materials.db, its first 50 materials (read when the section opens). */
let dbSecFor = null;
async function renderDbSection() {
  const host = $("#db-section");
  if (!host || !M() || !M().state.files) return;
  const f = M().state.files.find((x) => x.name === "materials.db" && !x.path.includes("/")) || M().state.files.find((x) => x.name === "materials.db");
  if (!f) { host.innerHTML = ""; return; }
  let open = false;
  open = !!(profile.ui("explorer", {}) || {}).dbOpen;
  host.innerHTML = `<button class="section-label section-toggle" id="db-toggle" aria-expanded="${open}" aria-controls="db-rows">${icon(open ? "chevron-down" : "chevron-right")}<span>Materials.db</span></button>` +
    `<div id="db-rows"${open ? "" : " hidden"}>${open ? '<p class="empty-note">Reading…</p>' : ""}</div>`;
  if (!open) return;
  const key = `${f.path}:${f.mtime_ns}`;
  try {
    const r = dbSecFor && dbSecFor.key === key ? dbSecFor.r : await M().api("db/page?" + M().q({ path: f.path, limit: 50 }));
    dbSecFor = { key, r };
    const rows = $("#db-rows");
    if (!rows) return;
    rows.innerHTML = r.rows.map((m) => `<button class="db-mat" data-mat="${esc(m.material_id)}" title="${esc(m.material_id)}"><span class="nm">${esc(m.material_id)}</span><span class="f">${esc(m.reduced_formula || m.formula || "")}</span></button>`).join("") +
      (r.total > r.rows.length ? `<button class="db-mat" data-mat=""><span class="nm muted">All ${r.total.toLocaleString("en-US")} materials…</span></button>` : "");
  } catch (e) { const rows = $("#db-rows"); if (rows) rows.innerHTML = `<p class="empty-note">${esc(e.message)}</p>`; }
  host.dataset.db = f.path;
}
$("#db-section").addEventListener("click", async (e) => {
  if (e.target.closest("#db-toggle")) {
    const open = e.target.closest("#db-toggle").getAttribute("aria-expanded") !== "true";
    profile.setUi("explorer", { ...(profile.ui("explorer", {}) || {}), dbOpen: open });
    renderDbSection();
    return;
  }
  const b = e.target.closest("[data-mat]");
  if (!b) return;
  const path = $("#db-section").dataset.db;
  await openDoc(path);
  if (b.dataset.mat && dbView) dbView.select(b.dataset.mat);
});
document.addEventListener("mace-files", () => renderDbSection());
$("#db-empty") && $("#db-empty").addEventListener("click", (e) => { if (e.target.closest("[data-act=db-build]")) dbBuild({}); });

function renderRecentSection() {
  const host = $("#recent-files");
  if (!host) return;
  const r = recentEntries();
  if (!r.length) { host.innerHTML = ""; return; }
  let open = true;
  open = (profile.ui("explorer", {}) || {}).recentOpen !== false;
  host.innerHTML = `<button class="section-label section-toggle" id="recent-toggle" aria-expanded="${open}" aria-controls="recent-rows">` +
    `${icon(open ? "chevron-down" : "chevron-right")}<span>Recent</span></button>` +
    `<div id="recent-rows"${open ? "" : " hidden"}>${r.slice(0, settings.get("recentLimit") + r.filter((x) => x.pinned).length).map(recentRow).join("")}</div>`;
}
function recentContext(path, x, y) {
  openAt(["file.recent.open", "file.recent.pin", "file.recent.remove", "-", "file.recent.copyPath"], x, y, "Recent file menu", { path });
}
function onRecentClick(e) {
  const t = e.target.closest("#recent-toggle");
  if (t) {
    const open = t.getAttribute("aria-expanded") !== "true";
    profile.setUi("explorer", { ...(profile.ui("explorer", {}) || {}), recentOpen: open });
    renderRecentSection();
    return;
  }
  const b = e.target.closest("[data-recent]");
  if (b && !b.classList.contains("missing")) openDoc(b.dataset.recent);
}
$("#recent-files").addEventListener("click", onRecentClick);
for (const sel of ["#recent-files", "#tab-welcome"]) {
  $(sel).addEventListener("contextmenu", (e) => {
    const b = e.target.closest("[data-recent]");
    if (!b) return;
    e.preventDefault();
    recentContext(b.dataset.recent, e.clientX, e.clientY);
  });
}
/** Clear Recently Opened: no question, but an Undo for 8 s (plan T9b). */
function clearRecent() {
  const was = recentList();
  const entries = recentEntries();
  profile.setRecentFiles([]);
  renderWelcome();
  if (!was.length) return;
  const restore = () => { saveRecent(entries); renderWelcome(); };
  toast(`Cleared ${was.length} recent file${was.length === 1 ? "" : "s"}`, "info", { label: "Undo", run: restore, ms: 8000 });
}
function recentMenu() {
  const r = recentEntries();
  const items = r.map((x) => ({ label: isMissing(x.path) ? `${x.path} (Missing)` : x.path, icon: x.pinned ? "pin" : "file",
    disabled: isMissing(x.path), title: isMissing(x.path) ? "Missing" : x.path, run: () => openDoc(x.path) }));
  return [...(items.length ? items : [{ label: "No recent files", disabled: true, run: () => {} }]), "-", "file.clearRecent"];
}
/* The explorer's sections (Open Editors, Files, Materials.db, Recent): each
 * header folds its section, remembered in the profile's ui.explorer like the
 * Recent and Materials.db sections always were. Each section scrolls inside
 * its own share (shell.css), so a long list never squeezes the file tree. */
const explorerUi = () => profile.ui("explorer", {}) || {};
function setExplorerOpen(key, open) { profile.setUi("explorer", { ...explorerUi(), [key]: open }); }
let editorsHtml = "";
function renderOpenEditors() {
  const host = $("#open-editors");
  if (!docs.length) { host.innerHTML = ""; editorsHtml = ""; return; }
  const open = explorerUi().editorsOpen !== false, was = $("#editors-rows"), top = was ? was.scrollTop : 0;
  const focused = document.activeElement && document.activeElement.id === "editors-toggle";
  const html = `<button class="section-label section-toggle" id="editors-toggle" aria-expanded="${open}" aria-controls="editors-rows">` +
    `${icon("chevron-down", "i chev")}<span>Open Editors</span><span class="section-count">${docs.length}</span></button>` +
    `<div class="side-rows" id="editors-rows"${open ? "" : " hidden"}>` + docs.map((d) =>
    `<button class="file ${d === active ? "active" : ""}" data-doc="${d.id}" title="${esc(d.path || d.title)}">` +
    `<span class="kb ${esc(d.fileKind || "out")}">${esc(KIND_LABEL[d.fileKind] || "view")}</span>${M().middleName(d.title)}</button>`).join("") + `</div>`;
  // only when something changed: a row replaced under the pointer or the focus loses it
  if (html === editorsHtml && was) { revealActiveEditor(); return; }
  editorsHtml = html;
  host.innerHTML = html;
  $("#editors-rows").scrollTop = top;
  if (focused) $("#editors-toggle").focus();
  revealActiveEditor();
}
/* The active row stays in sight inside the section's own scroll (never the page's),
 * also when the window or the UI scale shrinks the section. */
function revealActiveEditor() {
  const rows = $("#editors-rows"), cur = rows && !rows.hidden && $(".file.active", rows);
  if (cur && (cur.offsetTop < rows.scrollTop || cur.offsetTop + cur.offsetHeight > rows.scrollTop + rows.clientHeight)) {
    rows.scrollTop = cur.offsetTop - (rows.clientHeight - cur.offsetHeight) / 2;
  }
}
addEventListener("resize", revealActiveEditor);
function applyFilesOpen() {
  const open = explorerUi().filesOpen !== false, t = $("#files-toggle");
  t.setAttribute("aria-expanded", String(open));
  $("#file-list").hidden = !open;
  $("#files-section").classList.toggle("collapsed", !open);
}
applyFilesOpen();
$("#files-toggle").addEventListener("click", () => { setExplorerOpen("filesOpen", !(explorerUi().filesOpen !== false)); applyFilesOpen(); });
// their order, which show and their heights (1.1.6): splitters between them, headers to drag, a header menu
sidebar = initSidebar({ layout, saveLayout, bindSplitter, openAt, zoom: (el) => window.maceZoom.of(el) });
$("#root-chip").addEventListener("contextmenu", (e) => {
  e.preventDefault();
  openAt(["file.copyRootPath"], e.clientX, e.clientY, "Folder menu");
});
$("#open-editors").addEventListener("click", (e) => {
  if (e.target.closest("#editors-toggle")) { setExplorerOpen("editorsOpen", !(explorerUi().editorsOpen !== false)); renderOpenEditors(); return; }
  const b = e.target.closest("[data-doc]");
  if (b) activate(docs.find((d) => d.id === Number(b.dataset.doc)));
});

/* ================================================================ welcome */
function renderWelcome() {
  const files = M() ? M().state.files : [];
  const rec = recentEntries().slice(0, 8);
  const cifs = files.filter((f) => f.kind === "cif"), examples = cifs.slice(0, 3);
  const more = cifs.length - examples.length;
  const row = (p) => { const f = files.find((x) => x.path === p); return `<button class="file" data-open="${esc(p)}" title="${esc(p)}"><span class="kb ${f.kind}">${KIND_LABEL[f.kind]}</span>${M().middleName(p)}</button>`; };
  $("#tab-welcome").innerHTML = `<div class="welcome">
    <div><h1 class="welcome-logo"><img class="theme-light-only" src="${asset("brand/lockup.svg")}" alt="MACE" width="280" height="109"><img class="theme-dark-only" src="${asset("brand/lockup-on-dark.svg")}" alt="MACE" width="280" height="109"></h1><p>View, compare and set up CRYSTAL calculations for the structures in <code id="welcome-root" class="root-path" title="${esc(info ? info.root : "")}"></code>.</p></div>
    <div class="welcome-actions">
      <button class="tb-btn-text primary" data-cmd="file.open">${icon("folder-open")}<span>Open…</span><kbd>${esc(shortcutOf("file.open"))}</kbd></button>
      <button class="tb-btn-text" data-cmd="view.palette">${icon("search")}<span>Commands</span><kbd>${esc(shortcutOf("view.palette"))}</kbd></button>
      <button class="tb-btn-text" data-cmd="help.shortcuts">${icon("keyboard")}<span>Keyboard shortcuts</span><kbd>${esc(shortcutOf("help.shortcuts"))}</kbd></button>
    </div>
    <div class="welcome-grid">
      <div class="welcome-card"><h3>Recent</h3>${rec.length ? rec.map(recentRow).join("") : '<p class="insp-hint">Files you open show here.</p>'}</div>
      <div class="welcome-card"><h3>Structures in this folder</h3>${examples.length ? examples.map((f) => row(f.path)).join("") : '<p class="insp-hint">No CIF files here.</p>'}${more > 0 ? `<button class="file welcome-more" data-more-structures>+${more} more in the Explorer</button>` : ""}</div>
    </div>
    <div class="drop-hint">${icon("upload")}<span>Drop a CIF, POSCAR, XYZ, d12 or out file here, or pick one in the Explorer.</span></div></div>`;
  renderRoot();
  renderRecentSection();
}
$("#tab-welcome").addEventListener("click", (e) => {
  const o = e.target.closest("[data-open]");
  if (o) openDoc(o.dataset.open);
  if (e.target.closest("[data-more-structures]")) {
    if (!panelShown("explorer")) togglePanel("explorer");
    const chip = document.querySelector('.kind-chip[data-f="struct"]');
    if (chip) chip.click();
    const first = document.querySelector("#file-list .file");
    if (first) first.focus();
  }
  const rr = e.target.closest("[data-recent]");
  if (rr && !rr.classList.contains("missing")) openDoc(rr.dataset.recent);
  const c = e.target.closest("[data-cmd]");
  if (c) run(c.dataset.cmd);
});

/* ================================================================ dialogs */
/* Settings, the keymap editor and the shortcut sheet load the first time
 * they open (spec §8.5); `H` is what they may use of the shell. */
const H = {
  settings, esc, icon, toast, commands, refresh, opener: null, shell: () => S, profile, docprefs,
  api: (...a) => M().api(...a), overlayOpening: () => { hideTip(); closeMenus(); },
  settingsPath: () => settingsPath, templates: () => M().state.templates, boot: () => M().state.boot,
  elements: () => Object.keys((M().state.colors && M().state.colors.schemes.vesta) || {}),
  openSettings: (section) => settingsDialog(section),
  // themes and element colors (1.1.6)
  previewTheme: (id) => previewTheme(id), elementColors: () => elementColors(), M: () => M(),
  setPalettes: (v) => { M().setPalettes(v); tellPane("adoptColors", v, M().state.paletteDraft || null); },
  setPaletteDraft: (map) => { M().setPaletteDraft(map); tellPane("adoptColors", M().state.palettes, map || null); },
  tokenColor: (k) => getComputedStyle(document.documentElement).getPropertyValue(k === "cellColor" ? "--cell" : k === "outlineColor" ? "--text" : "--viewport-bg").trim(),
  // the editor's dialogs (Edit Cell, Set Coordinates, selection): the active document and the editor
  editor: () => ({ ed: edOf(active), E, doc: active }),
  // the document the 3D view draws (Calculate Bonds…)
  viewDoc: () => viewDoc(), structureOf: (d) => structureOf(d), bondValues: (d) => bondValues(d || viewDoc()),
  setRange: (n) => setRange(n), range: () => rangeCells(), rangeOf: () => DS.rangeOf(viewDoc()),
  setRangeLoHi: (lo, hi) => { const d = viewDoc(); if (d) DS.set(d, { displayLo: lo, displayHi: hi }); }, updateTriad: () => updateTriad(),
  alongLabel: (t) => { if ($("#tb-along-label")) $("#tb-along-label").textContent = t; },
  run: (id, args) => run(id, args),
  writeCheck: (p) => M().api("write/check?" + M().q({ path: p })),
  openPath: (p) => openDoc(p),
  files: () => M().state.files, root: () => (info ? info.root : ""),
  // connection and session UI (lazy/connect_ui.js)
  info: () => info, dirtyDocs: () => docs.filter(isDirty).map((d) => d.title),
  saveAll: () => run("file.saveAll"), flushDrafts: () => { if (window.__maceDrafts) window.__maceDrafts.flush(); },
  leave: (url) => { leaving = true; location.href = url; },
  copyText: (text, done) => copyText(text, done),
  // the Properties tab (lazy/properties_tab.js)
  apiTask: (p, b, u, x) => apiTask(p, b, u, x), showText: (title, text) => openTextDoc(title, text), dbBuild: (a) => dbBuild(a),
  docPathOf: (d) => propertiesPath(d), openDbPath: () => { const d = (active && active.kind === "db" ? active : docs.find((x) => x.kind === "db")); return d && d.path ? d.path : null; },
  stopping: () => { stopInitiator = true; serverStopped(); },
  docsUrl: (anchor) => docsUrl() + (anchor ? "#" + anchor : ""),
  setUpdate: (d) => { updateState = d; renderStatus(); }, updateDeferred: () => (updateState && updateState.deferred) || null,
};
const connectUi = () => lazy("connect_ui");
function openFolderDialog(path) { return connectUi().then((m) => m.openFolder(H, path)); }
/* Copy Diagnostics keeps the user's click: the bundle is fetched when the Help
 * menu (or About) opens, and copied at once from that cache (plan D20). */
let diagCache = null;
function prefetchDiagnostics() {
  if (diagCache && Date.now() - diagCache.t < 30000) return;
  diagCache = { t: Date.now(), text: null };
  const mine = diagCache;
  connectUi().then((m) => m.fetchDiagnostics(H)).then((r) => { mine.text = r.text; }).catch(() => { if (diagCache === mine) diagCache = null; });
}
function copyDiagnostics() {
  if (diagCache && diagCache.text && Date.now() - diagCache.t < 30000) { copyText(diagCache.text, "Diagnostics copied"); return; }
  connectUi().then((m) => m.copyDiagnostics(H));
}
document.addEventListener("pointerdown", (e) => { const b = e.target.closest && e.target.closest(".menu-top"); if (b && b.textContent === "Help") prefetchDiagnostics(); }, true);
document.addEventListener("focusin", (e) => { const b = e.target.closest && e.target.closest(".menu-top"); if (b && b.textContent === "Help") prefetchDiagnostics(); });
const lazyLoads = {};
function lazy(name) {
  if (!lazyLoads[name]) lazyLoads[name] = import(`./lazy/${name}.js`).catch((e) => {
    delete lazyLoads[name];
    // a module that cannot load after a sign-out: say that, not a generic error
    M().api("session").then((r) => { if (r && r.expired) sessionExpired(); }).catch(() => {});
    throw e;
  });
  return lazyLoads[name];
}
async function cheatSheet() {
  performance.mark("ui:cheatsheet:start");
  const m = await lazy("cheatsheet");
  await m.open(H);
  requestAnimationFrame(() => { performance.mark("ui:cheatsheet:end"); performance.measure("ui:cheatsheet", "ui:cheatsheet:start", "ui:cheatsheet:end"); });
}
/** Create or Update Database from a Folder (PLAN_analysis C10): {folder, db}. From a Database view its
 * database and folder; else the given folder (the served folder by default). */
async function dbBuild(a = {}) {
  let { folder, db } = a;
  if (folder === undefined && db === undefined && active && active.kind === "db" && active.path) db = active.path;
  if (folder === undefined) folder = db && db.includes("/") ? db.slice(0, db.lastIndexOf("/")) : "";
  const m = await lazy("db_build_dialog");
  return m.open(H, { folder, db: db || "", onStart: () => { if (dbView && active && active.kind === "db") dbView.poll(); },
    afterBuild: () => { propertiesChanged(); if (dbView && active && active.kind === "db") dbView.poll(); } });
}
/** File History (history.py): the versions MACE kept of a file before replacing it. */
async function fileHistory(path) {
  if (!path) { toast("Pick a file in the Explorer, or open one", "info"); return; }
  const m = await lazy("history_dialog");
  await m.open(H, path, { afterRestore: () => { M().loadFiles(); if (active && active.path === path && M().state) checkDiskSoon(); } });
}
function checkDiskSoon() { setTimeout(() => checkDisk(), 50); }
async function settingsDialog(section) {
  performance.mark("ui:settings:start");
  const m = await lazy("settings_dialog");
  await m.open(H, section);
  requestAnimationFrame(() => { performance.mark("ui:settings:end"); performance.measure("ui:settings", "ui:settings:start", "ui:settings:end"); });
}
function aboutDialog() {
  const v = asset("brand/");
  dialog({ id: "about", title: "About MACE", narrow: true, body: `<div class="about-hero">
      <img class="theme-light-only" src="${v}hero.svg" alt="MACE" width="288" height="99">
      <img class="theme-dark-only" src="${v}hero-on-dark.svg" alt="MACE" width="288" height="99"></div>
    <table class="kv about">
    <tr><td>MACE</td><td>${esc(info ? info.version : "")}</td></tr>
    <tr><td>Server</td><td>${esc(info ? info.hostname : "")}</td></tr>
    <tr><td>Folder</td><td><code>${esc(info ? info.root : "")}</code></td></tr>
    <tr><td>3D viewer</td><td>3Dmol.js 2.5.5 (BSD-3-Clause)</td></tr>
    <tr><td>Plots</td><td>plotly.js from the installed plotly package (MIT)${info && !info.plotly ? ", not installed here" : ""}</td></tr>
    <tr><td>Icons</td><td>Lucide 0.469 (ISC; Feather, MIT) and MACE's own crystal glyphs</td></tr>
    <tr><td>Element colors</td><td>the VESTA and Jmol schemes of pymatgen's ElementColorSchemes.yaml (MIT, The Materials Project / LBNL); the colorblind-safe scheme is MACE's own, from the Okabe–Ito hues (see element_colors.LICENSE.txt)</td></tr>
    <tr><td>Editor themes</td><td>colors of Catppuccin, Dracula, Gruvbox, Nord, One Dark, Solarized and Tokyo Night (MIT; see css/THEMES.LICENSE.txt)</td></tr>
    <tr><td>Logo</td><td>MACE wordmark (MIT, this repository)</td></tr></table>`,
  foot: `<button class="tb-btn-text" id="about-diag">${icon("copy")}<span>Copy Diagnostics</span></button>` });
  prefetchDiagnostics();
  $("#about-diag").addEventListener("click", () => copyDiagnostics());
}
/* Help ▸ Documentation: docs/gui.md of the installed version, on GitHub (a new
 * tab) and as the local file; no server route and no Markdown renderer. */
const DOCS_REPO = "https://github.com/DjokicMa/MACE";
function docsUrl() {
  // the server names a ref that exists on GitHub: the release tag of a tagged final release, else main
  return `${DOCS_REPO}/blob/${encodeURIComponent(info && info.docs_ref ? info.docs_ref : "main")}/docs/gui.md`;
}
function docsDialog(anchor = "") {
  const url = docsUrl() + (anchor ? `#${anchor}` : "");
  try { window.open(url, "_blank", "noopener"); } catch (e) { /* a blocked pop-up: the link below */ }
  const local = info && info.docs ? info.docs : "docs/gui.md";
  dialog({ id: "docs", title: "Documentation", narrow: true, body: `<p>The MACE GUI guide for this version (${esc(info ? info.version : "")})
      opens in a new tab: <a id="docs-link" href="${esc(url)}" target="_blank" rel="noopener noreferrer">docs/gui.md on GitHub</a>.</p>
    <p>The same file is part of your MACE installation, for reading offline:</p>
    <pre class="mono" id="docs-local">${esc(local)}</pre>`,
  foot: `<button class="tb-btn-text" id="docs-copy">${icon("copy")}<span>Copy path</span></button>` });
  $("#docs-copy").addEventListener("click", () => navigator.clipboard?.writeText($("#docs-local").textContent).then(() => toast("Copied", "success")));
}
function tunnelDialog() { return connectUi().then((m) => m.tunnel(H)); }

/* ================================================================ keyboard */
const REGIONS = ["#menus .menu-top", "#toolbar [data-cmd]", "#file-filter", ".doc-tab.active", ".views .panel.active .viewer-wrap, .views .panel.active", "#inspector .insp-head", ".bottom-tab.active", "#st-conn"];
function cycleRegion(step) {
  const ae = document.activeElement;
  let cur = REGIONS.findIndex((sel) => { const el = document.querySelector(sel); return el && (el === ae || el.parentElement.contains(ae)); });
  for (let i = 1; i <= REGIONS.length; i++) {
    const el = document.querySelector(REGIONS[(cur + step * i + REGIONS.length * 2) % REGIONS.length]);
    if (el && el.offsetParent !== null) { if (!el.hasAttribute("tabindex") && !/BUTTON|INPUT|SELECT|TEXTAREA/.test(el.tagName)) el.tabIndex = -1; el.focus(); return; }
  }
}
function focusRegion(name) {
  const el = { explorer: "#file-filter", viewport: ".views .panel.active .viewer-wrap", inspector: "#inspector .insp-head" }[name];
  if (name === "explorer" && !panelShown("explorer")) togglePanel("explorer");
  if (name === "inspector" && !panelShown("inspector")) togglePanel("inspector");
  const e = $(el);
  if (e) { if (!e.hasAttribute("tabindex") && e.tagName !== "INPUT" && e.tagName !== "BUTTON") e.tabIndex = 0; e.focus(); }
}
/* One lookup in the chord index per key press (spec §8.1: dispatch is O(1)).
 * Keys that would make the browser leave the page or switch tabs (Alt+Left,
 * Alt+1..9, Backspace, see BROWSER_NAV) never reach the browser while a
 * command is bound to them, even a disabled one: the toast says why instead. */
let navToast = { text: "", at: 0, el: null };
function navBlocked(text) {
  const now = performance.now();
  if (navToast.text === text && now - navToast.at < 1500 && navToast.el && navToast.el.isConnected) { navToast.at = now; return; }
  navToast = { text, at: now, el: toast(text, "info") };
}
// in a text field Backspace deletes; every other bound page-leaving key stays the page's
const TEXT_EDIT_NAV = new Set([chordId("Backspace")]);
const isTextField = (el) => !!el && (el.isContentEditable || ["INPUT", "TEXTAREA"].includes(el.tagName));
/* Capture phase, before the palette, a menu or a field can stop the key: a
 * bound page-leaving key never reaches the browser, wherever focus is. */
function navGuard(e) {
  if (e.isComposing || !BROWSER_NAV.size) return;
  const hits = lookup(e).filter(({ chord }) => BROWSER_NAV.has(chordId(chord)));
  if (!hits.length) return;
  if (isTextField(e.target) && TEXT_EDIT_NAV.has(chordId(hits[0].chord))) return;
  e.preventDefault();
  e.__maceNav = true;            // the dispatcher still decides what runs
}
addEventListener("keydown", navGuard, true);
/* While a builder preview or a Clean proposal is open (plan B11), its frame
 * sees the keys first: Mod+Z discards, Enter applies, Esc closes. */
let previewKeys = null;
function onKeydown(e) {
  if ((e.defaultPrevented && !e.__maceNav) || e.isComposing) return;
  // Mod+V / Mod+Shift+V: the browser's own paste (plan B12), never preventDefault here
  if ((e.ctrlKey || e.metaKey) && !e.altKey && e.code === "KeyV" && !isTextField(e.target)) { pasteAsImport = e.shiftKey ? performance.now() : 0; midUpAt = -Infinity; return; }
  if (previewKeys && !isPaletteOpen() && !menuOpen() && !document.querySelector("dialog:modal") && previewKeys(e)) {
    e.preventDefault(); e.stopPropagation(); return;
  }
  // a modal layer (a Move grab) first: its keys never reach the keymap
  const layerNow = commands.activeLayer();
  if (layerNow && !isPaletteOpen() && !menuOpen() && !document.querySelector("dialog:modal")) {
    const r = layerNow.onKey(e);
    if (r === "consume" || r === "swallow") { e.preventDefault(); e.stopPropagation(); return; }
  }
  if (e.key === "F6") { e.preventDefault(); cycleRegion(e.shiftKey ? -1 : 1); return; }
  if (isPaletteOpen() || menuOpen()) return;
  performance.mark("ui:key:start");
  // Esc leaves Zen Mode only when nothing else had anything to do (a dialog, a
  // menu, a field, a tool, a grab or a selection keeps its own Esc first)
  const zenEsc = () => {
    if (e.key !== "Escape" || !layout.zen || e.shiftKey || e.ctrlKey || e.altKey || e.metaKey) return false;
    const z = ctx();
    if (z.inputFocus || z.dialogOpen) return false;
    e.preventDefault();
    run("view.exitZen");
    return true;
  };
  const hits = lookup(e);
  if (!hits.length) { zenEsc(); return; }
  const c = ctx();
  const blocked = c.inputFocus || c.dialogOpen;
  let firstOff = null, nav = false;
  for (const { cmd, chord } of hits) {
    const id = chordId(chord);
    const isNav = BROWSER_NAV.has(id);
    // a field keeps Backspace and Alt+arrows for editing text
    if (c.inputFocus && TEXT_EDIT_NAV.has(id)) continue;
    if (isNav) nav = true;
    if (isSingleKey(chord) && blocked) continue;
    if (c.dialogOpen && chord.code !== "F1") continue;
    if (cmd.pseudo) {
      if (blocked) continue;
      e.preventDefault();
      cmd.run();
      return;
    }
    const why = disabledReason(cmd, c);
    if (!why) {
      e.preventDefault();
      run(cmd.id, { chord: chord.text });
      performance.measure("ui:key", "ui:key:start");
      return;
    }
    // say why for the command meant here: its context holds, it is only not built yet (or the like)
    const fits = !cmd.whenFn || cmd.whenFn(c);
    if (!firstOff || (fits && !firstOff.fits)) firstOff = { cmd, why, fits };
  }
  if (zenEsc()) return;
  if (nav) {
    e.preventDefault();
    if (firstOff && !c.dialogOpen) navBlocked(`${firstOff.cmd.title}: ${firstOff.why}`);
  }
}
document.addEventListener("keydown", onKeydown);

/* Leaving with unsaved edits asks first (spec §2.6); with nothing modified it never asks.
 * A reload the page chose itself (Open Folder, Sign Out) has asked already. */
let leaving = false;
addEventListener("beforeunload", (e) => {
  if (leaving || !docs.some(isDirty)) return;
  e.preventDefault();
  e.returnValue = "";
});

/* The keymap: the user's overrides from Settings › Keyboard over the defaults. */
function applyUserKeymap() {
  const report = applyKeymap(settings.get("keymap"));
  if ($("#palette-kbd")) $("#palette-kbd").textContent = shortcutOf("view.palette");
  refresh();
  return report;
}

/* ================================================================ refresh, theme */
let refreshFrame = 0;
function refresh() {
  cancelAnimationFrame(refreshFrame);
  refreshFrame = requestAnimationFrame(() => { renderToolbarState(); renderStatus(); renderTabs(); updateTitle(); });
}
function applySetting(k) {
  const root = document.documentElement;
  if (k === "theme") { themePreview = null; M().applyTheme(settings.get("theme")); updateTriad(); }
  if (!PANE && booted) tellPaneSoon();
  if (k === "autosaveDrafts" && E && E.drafts) E.drafts.schedule();
  if (k === "density") {
    if (settings.get("density") === "compact") root.setAttribute("data-density", "compact");
    else root.removeAttribute("data-density");
  }
  if (k === "uiScale") { root.style.setProperty("--ui-scale", String(settings.get("uiScale") / 100)); relayout(); }
  if (k === "reduceMotion") { const v = settings.get("reduceMotion"); root.removeAttribute("data-motion"); if (v !== "system") root.setAttribute("data-motion", v === "on" ? "reduce" : "full"); }
  if (k === "projection") applyProjection();
  // the fog is set at every paint (viewport/clip.js): a paint is enough
  if (k === "depthCue") Object.values(M().state.viewers || {}).forEach((v) => v && v.render && v.render());
  if (k === "showAxes") updateTriad();
  if (k === "keymap") applyUserKeymap();
  if (k === "cluster.showView") renderActivity();
  if (k.startsWith("activity.")) { renderActivity(); applyLayout(); }
  if (k === "cluster.notify" && settings.get("cluster.notify") && typeof Notification !== "undefined" && Notification.permission === "default") {
    try { Notification.requestPermission(); } catch (e) { /* not a secure context */ }
  }
  if (k === "viewport.defaultView" && active && active.kind === "structure") resetView();
  if (k === "density" || k === "uiScale") relayout();
  // the view toggles that are settings drive the classic views' own checkboxes
  const box = { showCell: "opt-cell" }[k];
  if (k === "showBoundary") syncBoundaryBox();     // the default: documents without their own value follow it
  if (box && optEl(box) && optEl(box).checked !== !!settings.get(k)) toggleOpt(box, !!settings.get(k));
  if (k === "labels" && optEl("opt-labels") && optEl("opt-labels").checked !== (settings.get("labels") !== "off")) {
    toggleOpt("opt-labels", settings.get("labels") !== "off");
  } else if (k === "labels") redrawSoon();
  if (k === "colorScheme") {
    if (M().fillColorOptions) M().fillColorOptions();
    // the default palette is also the CLI's (mace plotting): kept in ~/.config/mace/element_colors.json.
    // It is the user's own value: a folder's color_scheme or a change made in this session (to a key the
    // folder sets) is never written there
    if (!PANE && booted) savePaletteDefault(settings.userValue(k));
  }
  // the plots redraw on mace-theme: the plot background, the presets, the DOS colors and the spin styling
  if (k === "plots.surface" || k === "volume.mepColors" || /^plots\.(style|dos(Colors|Total)|spin(Colors|Up|Down|DownDash|Dos))/.test(k)) M().applyTheme(themePreview || settings.get("theme"));
  if (k === "background" || k === "backgroundColor") { M().syncSurfaces(); updateTriad(); }
  if (k === "viewport.printPatterns") printPatternsLayer();
  if (REDRAW_KEYS.has(k)) redrawSoon();
  if (DS.KEYS.has(k) && displayPanel) displayPanel.refresh();    // a default changed: documents without their own value follow it
  refresh();
}
/* Settings that change what the 3D views draw: one redraw per frame, camera kept. */
const REDRAW_KEYS = new Set(["showPolyhedra", "polyCenters", "polyOpacity", "polyEdges", "polyCenterAtom", "colorScheme", "radiusSet", "atomScale", "bondRadius", "bondTolerance", "bondMetalMetal", "bondMin", "bondPairs", "labelSize",
  "background", "backgroundColor", "cellColor", "decimals.length", "decimals.angle", "decimals.coord",
  "displayLo", "displayHi", "showBoundary", "bondedOutside", "lightStrength", "lightDirection", "ambient", "outline",
  "outlineWidth", "outlineColor", "ambientOcclusion", "shininess", "atomDraw", "atomColor", "atomRadii", "styleOutline",
  "viewport.vacuumMode", "viewport.vacuumGap"]);
/* Settings › Viewport › Print patterns (1.1.6): a layer over every 3D view (js/viewport/print_patterns.js),
 * loaded the first time it is on; every view paints again so the layer follows at once. */
function printPatternsLayer() {
  if ((settings.get("viewport.printPatterns") || "off") === "off" && !window.__macePatterns) return;
  import("./viewport/print_patterns.js").then((m) => {
    m.install();
    Object.values(M().state.viewers || {}).forEach((v) => { try { if (v && v.render) v.render(); } catch (e) { /* a hidden view */ } });
  }).catch(() => {});
}
let redrawFrame = 0;
function redrawSoon() {
  cancelAnimationFrame(redrawFrame);
  redrawFrame = requestAnimationFrame(() => { if (M() && M().redrawViews) M().redrawViews(); redrawFreq(); });
}
/* The classic views' own controls (now in Inspector › Display) write the settings. */
function mirrorClassicControls() {
  const on = (id, fn) => { const el = optEl(id); if (el) el.addEventListener("change", fn); };
  on("opt-cell", () => settings.set("showCell", optChecked("opt-cell")));
  // Boundary atoms is per document (Inspector › Range & boundary); Settings keep the default
  on("opt-boundary", () => { const d = viewDoc(); if (d && optChecked("opt-boundary") !== !!viewSetting("showBoundary")) DS.set(d, "showBoundary", optChecked("opt-boundary")); });
  on("opt-super", () => setRange(optChecked("opt-super") ? 2 : 1));
  VAC.bind({ viewDoc, structureOf });       // Vacuum: a slab's, polymer's or molecule's mode, per document (1.1.6)
  on("opt-labels", () => { if (optChecked("opt-labels") !== (settings.get("labels") !== "off")) settings.set("labels", optChecked("opt-labels") ? "both" : "off"); });
  on("opt-colors", () => settings.set("colorScheme", optEl("opt-colors").value));
  // an atom style picked while the document's Style draws its own: the style follows the Display section, as Custom
  on("opt-style", () => {
    optEl("opt-style").dataset.base = optEl("opt-style").value;
    const d = viewDoc();
    if (d && DRAW_OPT[viewSetting("atomDraw")]) DS.set(d, { atomDraw: "auto", ...(viewSetting("stylePreset") !== "custom" ? { stylePreset: "custom" } : {}) });
  });
}
/* ================================================================ themes and element colors (1.1.6) */
let themePreview = null, booted = false, paneTimer = 0;
/** Show theme id for now (a picker's hover or arrow keys); null goes back to the chosen one. */
function previewTheme(id) {
  if (id === themePreview) return;
  themePreview = id;
  M().applyTheme(id || settings.get("theme"));
  updateTriad();
  tellPane("previewTheme", id);
}
/** View › Theme › Color Theme…: the command search lists the themes; each previews while it is active. */
function pickTheme() {
  const cur = settings.get("theme");
  openPick({ title: "Color Theme", placeholder: "Select a color theme (the arrow keys preview)",
    items: THEMES.map((x) => ({ label: x.label, desc: x.desc, icon: x.icon, current: x.id === cur, aliases: [x.id, ...(x.aliases || [])],
      group: groupLabel(x.group),
      preview: () => previewTheme(x.id), unpreview: () => previewTheme(null),
      run: () => { settings.set("theme", x.id); previewTheme(null); } })) });
}
/* The right editor group is its own page: it takes this window's settings and colors. */
function tellPane(fn, ...args) {
  if (PANE) return;
  try {
    const f = document.querySelector("#group-side iframe");
    const p = f && f.contentWindow && f.contentWindow.__macePane;
    if (p && p[fn]) p[fn](...args);
  } catch (e) { /* the pane is loading */ }
}
function tellPaneSoon() {
  clearTimeout(paneTimer);
  paneTimer = setTimeout(() => tellPane("adoptSettings", settings.nonDefault()), 50);
}
function savePaletteDefault(id) {
  const pal = M().state.palettes || {};
  if (!id || pal.default === id || !(M().paletteMap && M().paletteMap(id))) return;
  if (!pal.default && settings.isDefault("colorScheme")) return;      // nothing chosen yet: nothing to keep
  M().api("palettes", { body: { op: "default", id } }).then((v) => { M().setPalettes(v); tellPane("adoptColors", v, M().state.paletteDraft || null); })
    .catch((e) => toast(`The default element colors were not saved: ${e.message}`, "warning"));
}
/** At start the palettes file's default becomes the user's own value (a hand edit, another computer's
 * GUI), except over a choice made in this browser that has not reached the server yet, which is saved
 * there. It is the user's layer: a folder's color_scheme (and this session's change) still wins. */
function adoptPaletteDefault() {
  const d = (M().state.palettes || {}).default, own = settings.userValue("colorScheme");
  if (settings.isPending("colorScheme")) { if (d !== own) savePaletteDefault(own); return; }
  if (d && d !== own && M().paletteMap && M().paletteMap(d)) settings.setOwn("colorScheme", d);
}
/* Export Element Legend… (lazy/legend_dialog.js): a scope {title, paths?, symbols?} */
const legendDialog = (scope) => lazy("legend_dialog").then((m) => m.open(H, scope)).catch((e) => toast(`Export Element Legend did not load: ${e.message}`, "error"));
const STRUCTURE_KINDS = new Set(["cif", "d12", "out", "xyz", "poscar"]);
function legendActive() {
  const s = structureOf(active);
  if (!s) return toast("Open a structure first", "info");
  // its atoms' print patterns, when Settings › Viewport › Print patterns is on (the layer's module is loaded then)
  const pp = window.__macePatterns, v = M().state.viewers["viewer-main"];
  const patterns = pp && v ? pp.mapFor(v) : {};
  return legendDialog({ title: `Elements of ${active.title}`, stem: active.title, symbols: [...s.symbols],
    ...(Object.keys(patterns).length ? { patterns } : {}) });
}
/** The open documents with a structure: saved files by path, the edited or unsaved ones by their symbols. */
function openStructures() {
  const paths = [], symbols = [];
  for (const d of docs) {
    if (!["structure", "trajectory", "editor"].includes(d.kind) && !edOf(d)) continue;
    const e = edOf(d);
    if (e && (isDirty(d) || !d.path)) symbols.push(...e.structure.symbols);
    else if (d.path) paths.push(d.path);
  }
  return { paths, symbols };
}
function legendOpenDocs() {
  const { paths, symbols } = openStructures();
  if (!paths.length && !symbols.length) return toast("No structure is open", "info");
  return legendDialog({ title: `Elements of the ${paths.length + (symbols.length ? 1 : 0)} open structure document(s)`, stem: "open_documents",
    paths, symbols: symbols.length ? symbols : undefined });
}
function legendExplorer(a) {
  if (a && a.dir !== undefined) return legendDialog({ title: `Structures in the folder ${a.dir || "(the served folder)"}`, stem: (a.dir || "folder").split("/").pop(), paths: [a.dir || ""] });
  const picked = new Set(M().state.selected || []);
  const p = pathOf(a);
  if (p) picked.add(p);
  const paths = [...picked].filter((x) => { const f = M().fileByPath(x); return f && STRUCTURE_KINDS.has(f.kind) && !M().isJobLog(f); });
  if (!paths.length) return toast("Pick structure files (CIF, d12, out, XYZ, POSCAR) or a folder", "info");
  return legendDialog({ title: paths.length === 1 ? `Elements of ${paths[0]}` : `Elements of ${paths.length} files`,
    stem: paths.length === 1 ? paths[0].split("/").pop() : "selected_files", paths });
}
/* Analyze Structures… (1.1.6, lazy/analysis_batch.js): the Explorer's selected structure files (and the
 * one right-clicked), or a folder's; the dialog lists the folder's other structures to add */
const structureFile = (f) => !!f && STRUCTURE_KINDS.has(f.kind) && !M().isJobLog(f);
function structureFiles() { return M().state.files.filter(structureFile).map((f) => f.path); }
function explorerStructures() { return [...(M().state.selected || [])].filter((x) => structureFile(M().fileByPath(x))); }
function analyzeStructures(a) {
  let paths;
  if (a && a.dir !== undefined) {
    const dir = a.dir ? `${a.dir}/` : "";
    paths = M().state.files.filter((f) => structureFile(f) && f.path.startsWith(dir) && !f.path.slice(dir.length).includes("/")).map((f) => f.path);
  } else {
    const picked = new Set(explorerStructures());
    const p = pathOf(a);
    if (p && structureFile(M().fileByPath(p))) picked.add(p);
    // from a menu with nothing picked: the structure in front, if it is a file
    if (!picked.size && active && active.path && structureFile(M().fileByPath(active.path))) picked.add(active.path);
    paths = [...picked];
  }
  const all = structureFiles();
  return lazy("analysis_batch").then((m) => m.open(H, { paths, all, done: (b) => { showBottom("analysis"); analysisTab(b.id); } }))
    .catch((e) => toast(`Analyze Structures: ${e.message}`, "error"));
}
const elementColors = () => lazy("palette_editor").then((m) => m.open(H)).catch((e) => toast(`Element Colors did not load: ${e.message}`, "error"));
function cycleTheme() {
  const order = ["system", "light", "dark"];
  settings.set("theme", order[(order.indexOf(settings.get("theme")) + 1) % 3]);
  toast(`Theme: ${themeOf(settings.get("theme")).label.toLowerCase()}`, "info");
}
function zoomUi(d) { settings.set("uiScale", d ? Math.min(150, Math.max(80, settings.get("uiScale") + d)) : 100); }
let settingsPath = "";

/* ================================================================ command helpers */
function rotateView(axis, deg) {
  const v = mainViewer();
  if (!v) return;
  if (E && active && active.kind === "structure" && E.cam()) { E.cam().turn(axis, deg, Math.abs(deg) >= 45).then(updateTriad); return; }
  // the 3Dmol views turn about the screen's axes as the structure view does (a turntable: "y" about its up)
  const p = rotationPrefs((k) => settings.get(k)), view = v.getView(), q = view.slice(4, 8);
  if (p.mode === "axes" && axis === "y") v.maceUp = turntableUp(q, v.maceUp);
  v.setView([...view.slice(0, 4), ...turnOrientation(q, axis, deg, { mode: p.mode, up: v.maceUp })]);
  v.render(); updateTriad();
}
/* The arrow keys' turn: the angle snap's step when one is on, else 15°. */
const rotateStep = () => rotationPrefs((k) => settings.get(k)).step || 15;
/* View › Snap View to Nearest Axis: the nearest of a, b, c, a*, b*, c* (a periodic view) and x, y, z. */
function snapNearestAxis() {
  const v = mainViewer();
  if (!v) return;
  const label = (t) => { if ($("#tb-along-label")) $("#tb-along-label").textContent = t; };
  if (E && active && active.kind === "structure") { const r = E.snapNearest(); if (r) r.then((t) => { label(t); updateTriad(); }); return; }
  let cell = null;
  if (volumeActive()) cell = volumeView.cell();
  else if (freqActive()) { const fd = freqView.doc(), s = fd && fd.structure; cell = s && s.cell && (!s.pbc || s.pbc.some(Boolean)) ? s.cell : null; }
  else { const s = M().state.structure; cell = s && s.cell && (!s.pbc || s.pbc.some(Boolean)) ? s.cell : null; }
  const view = v.getView(), r = nearestAxisView(view.slice(4, 8), cell);
  v.setView([...view.slice(0, 4), ...r.q]);
  v.render(); label(r.label); updateTriad();
}
function zoomView(f) { const v = mainViewer(); if (v) { v.zoom(f, reducedMotion() ? 0 : 120); v.render(); } }
const reducedMotion = () => document.documentElement.dataset.motion === "reduce"
  || (document.documentElement.dataset.motion !== "full" && matchMedia("(prefers-reduced-motion: reduce)").matches);

/* The Timeline's step slider, when the run has one (3 steps or more). */
function trajSteps() {
  const t = M() && M().state.traj, wrap = $("#o-frame-wrap");
  return active && active.kind === "trajectory" && t && t.path && wrap && !wrap.classList.contains("hidden") ? t.frames.length : 0;
}
function trajStep(d, absolute = false) {
  const el = $("#o-frame");
  if (!el || !trajSteps()) return;
  const n = trajSteps();
  const cur = Number(el.value);
  const next = absolute ? d : d === Infinity ? n - 1 : d === -Infinity ? 0 : Math.min(n - 1, Math.max(0, cur + d));
  if (next === cur) return;
  el.value = String(next);
  el.dispatchEvent(new Event("input"));
}
let playTimer = 0;
function trajPlay() {
  if (playTimer) { clearInterval(playTimer); playTimer = 0; refresh(); return; }
  const el = $("#o-frame");
  if (el && Number(el.value) >= trajSteps() - 1) trajStep(-Infinity);
  playTimer = setInterval(() => {
    const e = $("#o-frame");
    if (!e || !trajSteps() || Number(e.value) >= trajSteps() - 1) { clearInterval(playTimer); playTimer = 0; refresh(); return; }
    trajStep(1);
  }, 700);
  refresh();
}
/* Mod+Alt+O: overlay → two-tone → side by side → displacement map (plan B7) */
function trajLayout() { M().cycleLook(); }

/* Clipboard, with a fallback for an http page where navigator.clipboard is missing. */
async function copyText(text, done = "Copied") {
  window.__maceLastCopy = text;
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) await navigator.clipboard.writeText(text);
    else throw new Error("no clipboard");
  } catch (e) {
    const ta = document.createElement("textarea");
    ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); } catch (e2) { /* nothing more to try */ }
    ta.remove();
  }
  toast(done, "success");
}
/* A context menu's file, else the focused explorer row, else the active document's. */
function pathOf(a) {
  if (a && a.path) return a.path;
  const row = document.activeElement && document.activeElement.closest && document.activeElement.closest("#file-list .file[data-path]");
  return row ? row.dataset.path : docPath(active);
}
function reveal(path) {
  if (!path) return;
  if (!panelShown("explorer")) togglePanel("explorer");
  const filt = $("#file-filter");
  if (filt && filt.value) { filt.value = ""; M().state.text = ""; }
  M().state.filter = "all";
  M().renderFiles();
  const row = $$("#file-list .file").find((b) => b.dataset.path === path);
  if (!row) return;
  row.scrollIntoView({ block: "nearest" });
  row.focus();
  row.classList.add("flash");
  setTimeout(() => row.classList.remove("flash"), 1200);
}
/* Plot documents (js/docs/plot.js, loaded with the first plot; plotly loads with the first figure). */
let plotView = null, plotViewLoading = null;
function plotViewer() {
  if (plotView) return Promise.resolve(plotView);
  if (!plotViewLoading) plotViewLoading = Promise.all([import("./docs/plot.js"), import("./lazy/lazy_util.js")]).then(async ([m, lu]) => {
    await lu.loadCss(new URL("./docs/plot.js", import.meta.url).href, "../../css/plot.css");
    plotView = m.createPlotView($("#plot-host"), pieces.plotOpts, { api: (...a) => M().api(...a), q: (o) => M().q(o), esc, icon, toast,
      cssVar: (n) => M().cssVar(n), plotVar: (n, f, s) => M().plotVar(n, f, s), plotPaper: () => M().plotPaper(), styleFigure: (t, l, f, pid, s) => M().styleFigure(t, l, f, pid, s), plotVarIn: (n, pid, s) => M().plotVarIn(n, pid, s), plotTone: () => M().plotTone(), isPublication: (f) => M().isPublication(f),
      plotPreset: (f) => M().plotPreset(f), spinStyle: (f) => M().spinStyle(f), elementColors: () => M().elementColors(), presets: () => window.MACE_PRESETS, loadPlotly: () => M().loadPlotly(), apiTask: (p, b, u, x) => apiTask(p, b, u, x),
      exportImage: (X) => lazy("export_image").then((m) => m.open(H, { ...X, view: "plot", viewState: { projection: "orthographic" } })),
      openPath: (p) => openDoc(p),
      // with two editor groups, a clicked peak opens its mode in the other group
      openModes: (p, o) => (GR && GR.openModesOther(p, o)) || openModes(p, o), lineShape: (kind) => (kind === "ir" ? settings.get("spectra.lineShape") || "lorentzian" : "lorentzian"),
      download: (b, n) => download(b, n), dirHash: () => dirHash(), setting: (k) => settings.get(k), setSetting: (k, v) => settings.set(k, v),
      folderFile: () => settings.folderInfo().file,
      // Add DOS… / Add Band… (a band + DOS pair from two folders): the folder's files, picked in a dialog
      files: () => M().state.files, dialog });
    return plotView;
  }).catch((e) => { plotViewLoading = null; throw e; });
  return plotViewLoading;
}
function showPlot(d) {
  $("#plot-host").hidden = false; $("#runplot-host").hidden = true; $("#mech-host").hidden = true;
  return plotViewer().then((v) => {
    if (active !== d) return;
    if (["image", "html"].includes(d.fileKind)) v.openImage(d, d.fileKind);
    else { d.plot = d.plot || { kind: d.fileKind, path: d.path, title: d.title }; v.open(d.plot); }
    if (d.plot && d.plot.source === "modes" && d.srcPath) { M().state.active = d.srcPath; M().renderFiles(); }
  }).catch((e) => toast(`Plot: ${e.message}`, "error"));
}

/* Run Analysis (js/docs/runplot.js, PLAN_analysis C4): one optimization's energy, convergence, SCF,
 * cell and tracked measurements; a document of its own (no file of its own: it reads the .out). */
let runView = null, runViewLoading = null;
function runViewer() {
  if (runView) return Promise.resolve(runView);
  if (!runViewLoading) runViewLoading = Promise.all([import("./docs/runplot.js"), import("./lazy/lazy_util.js")]).then(async ([m, lu]) => {
    await lu.loadCss(new URL("./docs/runplot.js", import.meta.url).href, "../../css/plot.css");
    runView = m.createRunPlot($("#runplot-host"), { api: (...a) => M().api(...a), q: (o) => M().q(o), esc, icon, toast,
      cssVar: (n) => M().cssVar(n), plotVar: (n, f, s) => M().plotVar(n, f, s), plotPaper: () => M().plotPaper(), styleFigure: (t, l, f, pid, s) => M().styleFigure(t, l, f, pid, s), plotVarIn: (n, pid, s) => M().plotVarIn(n, pid, s), plotTone: () => M().plotTone(), isPublication: (f) => M().isPublication(f),
      plotPreset: (f) => M().plotPreset(f), spinStyle: (f) => M().spinStyle(f), elementColors: () => M().elementColors(), presets: () => window.MACE_PRESETS, loadPlotly: () => M().loadPlotly(), apiTask: (p, b, u, x) => apiTask(p, b, u, x),
      download: (b, n) => download(b, n), dirHash: () => dirHash(), setting: (k) => settings.get(k),
      expandSummary: (r) => M().expandSummary(r), selectFrame: (p, k) => M().selectFrame(p, k), frameOf: (p) => M().frameOf(p),
      openRun: async (p) => { const cur = active; await openDoc(p); if (cur) activate(cur); } });
    return runView;
  }).catch((e) => { runViewLoading = null; throw e; });
  return runViewLoading;
}
function showRunPlot(d) {
  $("#plot-host").hidden = true; $("#runplot-host").hidden = false; $("#mech-host").hidden = true;
  return runViewer().then((v) => { if (active === d) return v.open(d); }).catch((e) => toast(`Run Analysis: ${e.message}`, "error"));
}
/* Mechanical results documents (js/docs/mech.js, 1.1.6): an ELASTIC, EOS, QHA or CPHF output's tables
 * and plots (mace analyze elastic [--plot]), a document of the output itself, as a FREQ output's modes. */
const MECH_CALCS = ["ELASTIC", "EOS", "QHA", "CPHF"];
let mechView = null, mechViewLoading = null;
function mechViewer() {
  if (mechView) return Promise.resolve(mechView);
  if (!mechViewLoading) mechViewLoading = Promise.all([import("./docs/mech.js"), import("./lazy/lazy_util.js")]).then(async ([m, lu]) => {
    const at = new URL("./docs/mech.js", import.meta.url).href;
    await Promise.all([lu.loadCss(at, "../../css/plot.css"), lu.loadCss(at, "../../css/mechanics.css")]);
    mechView = m.createMechView($("#mech-host"), { api: (...a) => M().api(...a), q: (o) => M().q(o), esc, icon, toast,
      cssVar: (n) => M().cssVar(n), plotVar: (n, f, s) => M().plotVar(n, f, s), plotPaper: () => M().plotPaper(), styleFigure: (t, l, f, pid, s) => M().styleFigure(t, l, f, pid, s),
      loadPlotly: () => M().loadPlotly(), download: (b, n) => download(b, n), dirHash: () => dirHash(), setting: (k) => settings.get(k),
      // the Explorer's outputs of one kind (the EOS document's Add phase… lists the other EOS runs)
      outputs: (calc) => Object.entries(M().state.details || {}).filter(([, x]) => x && x.role === "crystal_out" && x.calc === calc).map(([p]) => p).sort(),
      // the document's settings came in: the Inspector's Calculation section shows them
      loaded: (d) => { renderTabs(); renderOpenEditors(); if (active === d) { updateTitle(); if (!setupOpen) renderInspector(); } } });
    mechView.bind((act, d) => (act === "text" ? openAsText(d.path) : openDoc(d.path, { as: "structure" })));
    return mechView;
  }).catch((e) => { mechViewLoading = null; throw e; });
  return mechViewLoading;
}
function showMech(d) {
  $("#plot-host").hidden = true; $("#runplot-host").hidden = true; $("#mech-host").hidden = false;
  return mechViewer().then((v) => { if (active === d) return v.open(d); }).catch((e) => toast(`Results: ${e.message}`, "error"));
}
async function openMech(path, calc = null) {
  const existing = docs.find((x) => x.kind === "mech" && x.path === path);
  if (existing) { await activate(existing); return existing; }
  saveView(active);
  const name = path.split("/").pop().replace(/\.[^.]+$/, "");
  const d = { id: nextId++, path, kind: "mech", fileKind: "out", calc, title: `${name} · ${calc ? calc : "results"}` };
  docs.push(d);
  pushRecent(path);
  await activate(d);
  feedRunProblems(d);
  return d;
}
/** Inspector › Calculation of a results document: its key settings, each with where it was read from. */
const MECH_SRC = { out: "from the .out", d12: "from the .d12 beside it", default: "CRYSTAL23 default" };
function mechSettings(d) {
  const el = document.createElement("div");
  el.className = "run-card mv-settings";
  const m = d.mech;
  if (!m) { el.innerHTML = Array.from({ length: 5 }, () => `<div class="skel-line"></div>`).join(""); return el; }
  const rows = m.settings || [];
  el.innerHTML = rows.length ? `<table class="kv rc-kv" aria-label="Calculation settings">${rows.map((r) =>
    `<tr title="${esc(MECH_SRC[r.source] || r.source)}"><td>${esc(r.label)}</td><td>${esc(r.value)}</td></tr>`).join("")}</table>
    <div class="rc-acts"><button class="tb-btn-text" data-mvs="cli" title="${esc(m.cli)}">Copy CLI</button><button class="tb-btn-text" data-mvs="text">Show Output</button></div>`
    : `<p class="insp-hint">No settings read from ${esc(m.name)}</p>`;
  el.addEventListener("click", (e) => {
    const b = e.target.closest("[data-mvs]");
    if (b && b.dataset.mvs === "cli") copyText(m.cli, "Copied the CLI command");
    if (b && b.dataset.mvs === "text") openAsText(d.path);
  });
  return el;
}
/** The run's .out: the argument, the active run, or the run a Run Analysis document reads. */
function runPathOf(a) {
  if (a && a.path && /\.(out|log)$/i.test(a.path)) return a.path;
  if (active && active.kind === "runplot") return active.runPath;
  if (active && (active.kind === "trajectory" || active.kind === "structure") && active.path && /\.(out|log)$/i.test(active.path)) return active.path;
  return null;
}
async function openRunPlot(path, tab = null, compare = null, energy = null) {
  // `energy`: the options the opening card shows, for the tab it opens (the energy or the SCF card)
  path = path || runPathOf(null);
  if (!path) { toast("Open an optimization output (.out) first", "info"); return null; }
  let d = docs.find((x) => x.kind === "runplot" && x.runPath === path && (x.compare || null) === (compare || null));
  if (!d) {
    saveView(active);
    const name = path.split("/").pop();
    d = { id: nextId++, path: null, runPath: path, compare, kind: "runplot", fileKind: "out",
      title: compare ? `Compare Runs · ${name} vs ${compare.split("/").pop()}` : `Run Analysis · ${name}`, tab: tab || undefined };
    docs.push(d);
  } else if (tab) d.tab = tab;
  if (energy) d.energyPatch = energy;            // the energy card's options (Open as Plot)
  await activate(d);
  if (tab && runView) runView.showTab(tab);
  if (energy && runView) runView.setEnergy(d, d.energyPatch);
  return d;
}
/* ---------------------------------------------------------------- a run's card and problems (PLAN_analysis C4.8, C7)
 * The card: how the run stopped (the words `mace analyze --convergence` and `mace check` use), then
 * the details the database would store (a pool job; a skeleton until it answers). The problems of the
 * .out go to the Problems panel (source "Output: <name>") and leave with the document. */
const runMeta = new Map();          // path -> {meta, stop, reason} | {pending: Promise}
function stopIcon(reason, stop) {
  if (reason === "all_criteria" || /^Single-point run: no optimization|^Finished/.test(stop || "")) return ["circle-check", "good"];
  if (["gradient_first_cycle", "gradient_and_energy", "energy_stalled"].includes(reason) || /^Unfinished|^Still running/.test(stop || "")) return ["triangle-alert", "warn"];
  return ["circle-alert", "bad"];
}
function fmtSecs(s) { if (s === null || s === undefined) return "—"; s = Number(s); return s >= 3600 ? `${(s / 3600).toFixed(2)} h` : s >= 60 ? `${(s / 60).toFixed(1)} min` : `${s.toFixed(1)} s`; }
function runCard(d) {
  const el = document.createElement("div");
  el.className = "run-card";
  const path = d.path;
  const paint = (r) => {
    if (!r || r.pending) {
      el.innerHTML = `<div class="rc-stop skel"><span class="skel-line"></span></div>${Array.from({ length: 6 }, () => `<div class="skel-line"></div>`).join("")}`;
      return;
    }
    const m = r.meta || {}, [ic, cls] = stopIcon(r.reason, r.stop);
    const row = (k, v) => (v === null || v === undefined || v === "" ? "" : `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`);
    const scf = m.scf_cycles ? `${m.scf_cycles.min} / ${m.scf_cycles.median} / ${m.scf_cycles.max}` : null;
    el.innerHTML = `<div class="rc-stop ${cls}">${icon(ic)}<span>${esc(r.stop || "")}</span></div>
      <div class="rc-acts"><button class="tb-btn-text" data-rc="show"${r.stop_line ? "" : " disabled"}>Show in Output</button>
        <button class="tb-btn-text" data-rc="cli">Copy CLI</button><button class="tb-btn-text" data-rc="plot">Run Analysis</button><button class="tb-btn-text" data-rc="scf" title="Run Analysis › SCF: each SCF cycle's energy change against TOLDEE">SCF</button></div>
      <table class="kv rc-kv">${row("CRYSTAL", m.crystal_version)}${row("Calculation", m.calc_type)}${row("Dimensionality", m.dimensionality)}
        ${row("Space group", m.space_group)}${row("Atoms in the cell", m.atoms)}${row("Functional", m.functional)}${row("Basis", m.basis)}
        ${row("SHRINK", m.shrink ? m.shrink.join(" ") + (m.gilat ? ` · ${m.gilat}` : "") : null)}${row("TOLINTEG", m.tolinteg ? m.tolinteg.join(" ") : null)}
        ${row("Spin", m.spin)}${row("MPI processes", m.mpi_processes)}${row("Started", m.started)}${row("Ended", m.ended)}
        ${row("Wall time", m.wall_s !== null && m.wall_s !== undefined ? fmtSecs(m.wall_s) : null)}${row("CPU time", m.cpu_s !== null && m.cpu_s !== undefined ? fmtSecs(m.cpu_s) : null)}
        ${row("File size", m.file_size !== undefined ? `${(m.file_size / 1048576).toFixed(2)} MB` : null)}${row("SCF cycles per point", scf ? `${scf} (min / median / max)` : null)}
        ${row("Final energy", m.energy_ev !== null && m.energy_ev !== undefined ? `${m.energy_ev.toFixed(6)} eV${m.energy_ev_per_atom ? ` · ${m.energy_ev_per_atom.toFixed(6)} eV/atom` : ""}` : null)}</table>`;
  };
  el.addEventListener("click", async (e) => {
    const b = e.target.closest("[data-rc]");
    if (!b) return;
    const r = runMeta.get(path);
    if (b.dataset.rc === "show" && r && r.stop_line) showInOutput(path, r.stop_line);
    if (b.dataset.rc === "cli") copyText(`mace analyze --convergence ${path}\nmace check ${path}`, "Copied the CLI commands");
    if (b.dataset.rc === "plot") openRunPlot(path, r && r.reason ? "energy" : "scf");     // a single point: its SCF
    if (b.dataset.rc === "scf") openRunPlot(path, "scf");
  });
  // 1.1.6: time left, as the Job document and `mace cluster job` word it (mace/cluster/progress.py), for an
  // output still being written; read again every 30 s while the card is shown
  const eta = document.createElement("div");
  eta.className = "rc-eta";
  let etaTimer = 0;
  const loadEta = () => {
    clearTimeout(etaTimer);
    if (!eta.isConnected && eta.dataset.done) return;
    M().api("out/progress?" + M().q({ path })).then((r) => {
      eta.dataset.done = "1";
      if (r.ended || !(r.levels || []).length) { eta.innerHTML = ""; return; }
      const lv = r.levels.filter((x) => x.depth === 0 || x.status !== "done");
      eta.innerHTML = `<div class="rc-eta-head">Time left</div><ol class="rc-eta-levels" aria-label="Progress and time left">${lv.map((x) =>
        `<li class="d${x.depth} ${esc(x.status)}"${x.basis ? ` title="${esc(x.basis)}"` : ""}>${esc(x.text)}${x.depth === 0 ? ` <span class="chip conf-${esc(x.confidence)}">${esc(x.confidence)}</span>` : ""}</li>`).join("")}</ol>`;
      etaTimer = setTimeout(() => { if (eta.isConnected) loadEta(); }, 30000 * (window.__maceLiveScale || 1));
    }).catch(() => { eta.innerHTML = ""; });
  };
  const have = runMeta.get(path);
  if (have && !have.pending) paint(have);
  else {
    paint(null);
    loadRunMeta(path).then((r) => { if (el.isConnected) { paint(r); el.appendChild(eta); } }).catch((err) => { el.innerHTML = `<p class="insp-hint">${esc(err.message)}</p>`; });
  }
  el.appendChild(eta);
  loadEta();
  return el;
}
async function loadRunMeta(path) {
  const have = runMeta.get(path);
  if (have && have.pending) return have.pending;
  if (have) return have;
  const p = (async () => {
    const [m, pr] = await Promise.all([apiTask("out/meta?" + M().q({ path }), undefined, null, { get: true }),
      M().api("out/problems?" + M().q({ path })).catch(() => null)]);
    const stopRow = pr && pr.problems.find((x) => x.code === "stop");
    const r = { meta: m.meta, stop: (stopRow && stopRow.message) || m.stop, reason: m.reason, stop_line: stopRow ? stopRow.line : null };
    runMeta.set(path, r);
    return r;
  })();
  runMeta.set(path, { pending: p });
  try { return await p; } catch (e) { runMeta.delete(path); throw e; }
}
/** Problems from the .out, for a document that shows a run (source "Output: <name>"). */
async function feedRunProblems(d) {
  if (!d || !d.path || !/\.(out|log)$/i.test(d.path)) return;
  const name = d.path.split("/").pop(), src = `Output: ${name}`;
  let r;
  try { r = await M().api("out/problems?" + M().q({ path: d.path })); } catch (e) { return; }
  if (!docs.includes(d)) return;
  clearProblems(d.id, src);
  for (const p of r.problems.slice().reverse()) {
    if (p.code === "stop" && p.severity === "info" && !p.line) continue;      // "Finished normally." needs no row
    const action = p.action === "modes" ? { label: "Show Modes", run: () => showModesAt(d.path, p.lowest_cm) }
      : p.line ? { label: "Show in Output", run: () => showInOutput(d.path, p.line) } : null;
    addProblem({ severity: p.severity, source: src, doc: d.id, message: p.message, line: p.line, code: p.code, action });
  }
}
/** The output's text window, scrolled so ``line`` (1-based) is visible. */
async function showInOutput(path, line) {
  await openAsText(path);
  const end = Date.now() + 15000;
  while (Date.now() < end && !(textView && textView.total() >= line)) await new Promise((r) => setTimeout(r, 100));
  if (textView) textView.jump(line - 1);
}
/** The FREQ document on the mode nearest ``cm`` (the lowest imaginary one). */
async function showModesAt(path, cm) {
  await openModes(path);
  const d = active;
  const end = Date.now() + 30000;
  while (Date.now() < end && !(d && d.modes)) await new Promise((r) => setTimeout(r, 100));
  if (!d || !d.modes || cm === undefined || cm === null) return;
  let best = 0, bd = Infinity;
  d.modes.forEach((m, k) => { const v = Math.abs((m.imaginary ? -Math.abs(m.freq) : m.freq) - cm); if (v < bd) { bd = v; best = k; } });
  if (freqView) freqView.select(best);
}

/** Track Along the Run (a Measurements row on a run): the Tracked tab of its Run Analysis. */
async function trackMeasurement(m, label) {
  const p = runPathOf(null);
  if (!p) { toast("Track Along the Run needs an open optimization output", "info"); return; }
  await openRunPlot(p, "tracked");
  const v = await runViewer();
  v.track(m, label);
}
window.__maceOpenRunPlot = (p, tab, energy) => openRunPlot(p, tab || "energy", null, energy || null);
/** Compare Runs…: a second .out picked from the folder's outputs. */
async function compareRunsDialog() {
  const first = runPathOf(null);
  if (!first) { toast("Open an optimization output (.out) first", "info"); return; }
  const outs = M().state.files.filter((f) => f.kind === "out" && f.path !== first).map((f) => f.path);
  if (!outs.length) { toast("No other output in this folder to compare with", "info"); return; }
  const body = document.createElement("div");
  body.innerHTML = `<label class="f">Compare ${esc(first.split("/").pop())} with<select class="field" id="cmp-run-pick" aria-label="The second run">${outs.map((p) =>
    `<option value="${esc(p)}">${esc(p)}</option>`).join("")}</select></label>`;
  const m = await saveDialog();
  const v = await m.ask(H, { id: "compare-runs", title: "Compare Runs", body, buttons: [["cancel", "Cancel"], ["ok", "Compare", true]] });
  if (v !== "ok") return;
  const second = body.querySelector("#cmp-run-pick").value;
  openRunPlot(first, "energy", second);
}

/* FREQ documents (js/docs/freq.js): an output's normal modes, animated in their own viewport. */
let freqView = null, freqViewLoading = null;
function freqViewer() {
  if (freqView) return Promise.resolve(freqView);
  if (!freqViewLoading) freqViewLoading = Promise.all([import("./docs/freq.js"), import("./lazy/lazy_util.js")]).then(async ([m, lu]) => {
    await Promise.all([lu.loadCss(new URL("./docs/freq.js", import.meta.url).href, "../../css/freq.css"),
      lu.loadCss(new URL("./docs/freq.js", import.meta.url).href, "../../css/plot.css")]);
    freqView = m.createFreqView($("#freq-host"), pieces.freqTable, pieces.freqOpts, { api: (...a) => M().api(...a), q: (o) => M().q(o), esc, icon, toast,
      apiTask: (p, b, u, x) => apiTask(p, b, u, x), download: (b, n) => download(b, n), M,
      // the toolbar's display options, as the structure view takes them (the range and boundary atoms per document)
      viewOpts: () => viewOpts(), setting: (k) => settings.get(k), rangeN: () => range(), setRange: (n) => setRange(n),
      rangeCells: () => rangeCells(), setSetting: (k, v) => settings.set(k, v),
      openStructure: (p) => openStructure(p), openSpectrum: (p, kind) => openFreqSpectrum(p, kind), openMech: (p) => openMech(p, "FREQ"),
      imaginary: (d, list) => {
        clearProblems(d.id, "FREQ");
        if (!list || !list.length) return;
        // below Settings › Plots › Imaginary-mode noise level: numerical noise (info), as the run's problems say
        const noise = Number(settings.get("analysis.imagNoiseCm") ?? 20);
        const real = list.filter((m) => Math.abs(m.freq) >= noise);
        const shown = real.length ? real : list;
        addProblem({ severity: real.length ? "warning" : "info", source: "FREQ", doc: d.id,
          message: `${shown.length} imaginary mode${shown.length === 1 ? "" : "s"} at ${shown.slice(0, 3).map((m) => `${m.freq.toFixed(1)} cm⁻¹ (${m.irrep})`).join(", ")}` +
            (real.length ? ": the structure is not at a minimum" : ": usually numerical noise") });
      } });
    return freqView;
  }).catch((e) => { freqViewLoading = null; throw e; });
  return freqViewLoading;
}
/** A FREQ output's IR or Raman spectrum: CRYSTAL's SPEC.DAT when it wrote one, else broadened from the modes (plan E4). */
async function openFreqSpectrum(path, kind) {
  const partner = partnerOf({ path }, kind);
  if (partner) return openDoc(partner.path);
  const existing = docs.find((x) => x.kind === "plot" && x.plot && x.plot.source === "modes" && x.srcPath === path && x.fileKind === kind);
  if (existing) { await activate(existing); return existing; }
  saveView(active);
  const title = `${path.split("/").pop().replace(/\.[^.]+$/, "")} · ${kind === "ir" ? "IR" : "Raman"} (from the modes)`;
  const d = { id: nextId++, path: null, srcPath: path, kind: "plot", fileKind: kind, title,
    plot: { kind, path, title, source: "modes" } };
  docs.push(d);
  await activate(d);
  return d;
}
/** A fort.25 2D map (ECHG charge density, POTM potential) as a contour plot. */
async function openMapPlot(path) {
  const existing = docs.find((x) => x.kind === "plot" && x.plot && x.plot.kind === "map" && x.srcPath === path);
  if (existing) { await activate(existing); return existing; }
  saveView(active);
  const title = `${path.split("/").pop()} · 2D map`;
  const d = { id: nextId++, path: null, srcPath: path, kind: "plot", fileKind: "f25", title, plot: { kind: "map", path, title } };
  docs.push(d);
  pushRecent(path);
  await activate(d);
  return d;
}
/** Compare Convergence (plan E12 fallback): one plot of several runs' energy per atom by step. */
async function openConvergence(paths) {
  if (!paths || paths.length < 2) { toast("Ctrl-click two or more CRYSTAL outputs in the Explorer first", "info"); return null; }
  saveView(active);
  const title = `Convergence of ${paths.length} runs`;
  const d = { id: nextId++, path: null, kind: "plot", fileKind: "convergence", title, plot: { kind: "convergence", path: paths.join("|"), paths, title } };
  docs.push(d);
  await activate(d);
  return d;
}
/** An output's partner spectrum beside it (<stem>_IRSPEC.DAT / _RAMSPEC.DAT), by name only. */
function partnerOf(d, kind) {
  if (!d || !d.path || !/\.(out|log|o)$/i.test(d.path) || !M() || !M().state.files) return null;
  const stem = d.path.replace(/\.[^.]+$/, ""), tag = kind === "ir" ? "IRSPEC.DAT" : "RAMSPEC.DAT";
  return M().state.files.find((f) => f.kind === kind && (f.path === `${stem}_${tag}` || f.path === `${stem}.${tag}`)) || null;
}
function isFreqDoc(d) {
  if (!d || !d.path || !/\.(out|log|o)$/i.test(d.path)) return false;
  return (M() && M().isFreq && M().isFreq(d.path)) || !!(d.data && d.data.freq);
}
async function openModes(path, opts = {}) {
  const existing = docs.find((x) => x.kind === "freq" && x.path === path);
  if (existing) {
    await activate(existing);
    if (opts.mode && freqView) await freqView.selectMode(opts.mode);
    return existing;
  }
  saveView(active);
  const d = { id: nextId++, path, kind: "freq", fileKind: "out", title: `${path.split("/").pop().replace(/\.[^.]+$/, "")} · modes`,
    wantMode: opts.mode || null };
  docs.push(d);
  active = d;
  panelOwner.freq = d.id;
  M().showTab("freq");
  showBottom("modes");
  afterSwitch();
  const v = await freqViewer();
  if (active === d) await v.open(d);
  return d;
}

/* The Plot Gallery (js/docs/gallery.js, MACE 1.1.6): many outputs' plots of one kind in panes, cycled one
 * at a time, with a caption table; the same grid as `mace plotting --gallery`. A document of its own. */
let galleryView = null, galleryLoading = null;
function galleryViewer() {
  if (galleryView) return Promise.resolve(galleryView);
  if (!galleryLoading) galleryLoading = Promise.all([import("./docs/gallery.js"), import("./lazy/lazy_util.js")]).then(async ([m, lu]) => {
    await lu.loadCss(new URL("./docs/gallery.js", import.meta.url).href, "../../css/gallery.css");
    galleryView = m.createGallery($("#gallery-host"), { api: (...a) => M().api(...a), q: (o) => M().q(o), esc, icon, toast,
      cssVar: (n) => M().cssVar(n), plotVar: (n, f, s) => M().plotVar(n, f, s), plotPaper: () => M().plotPaper(), styleFigure: (t, l, f, pid, s) => M().styleFigure(t, l, f, pid, s), plotVarIn: (n, pid, s) => M().plotVarIn(n, pid, s), plotTone: () => M().plotTone(), isPublication: (f) => M().isPublication(f),
      plotPreset: (f) => M().plotPreset(f), spinStyle: (f) => M().spinStyle(f), elementColors: () => M().elementColors(), presets: () => window.MACE_PRESETS, loadPlotly: () => M().loadPlotly(), apiTask: (p, b, u, x) => apiTask(p, b, u, x),
      download: (b, n) => download(b, n), copyText: (t, m) => copyText(t, m), openPath: (p) => openDoc(p),
      setting: (k) => settings.get(k), folderFile: () => settings.folderInfo().file, sets: () => gallerySets(), retitle: () => { renderTabs(); updateTitle(); } });
    return galleryView;
  }).catch((e) => { galleryLoading = null; throw e; });
  return galleryLoading;
}
/** The sets a gallery can show: this folder, the Explorer's selection, each open database's filtered materials. */
function gallerySets() {
  const out = [{ label: "This folder", source: { type: "folder", folder: "" } }];
  const picked = [...(M().state.selected || [])];
  if (picked.length) out.push({ label: `Explorer selection (${picked.length})`, source: { type: "paths", paths: picked } });
  for (const d of docs.filter((x) => x.kind === "db" && x.path)) {
    let filters = {};
    try {
      if (dbView && active === d) {
        const qs = new URLSearchParams(dbView.exportUrl("csv").split("?")[1] || "");
        for (const k of ["q", "dim", "calc_type", "calc_status", "sg_min", "sg_max", "prop"]) if (qs.get(k)) filters[k] = qs.get(k);
      }
    } catch (e) { filters = {}; }
    const n = Object.keys(filters).length;
    out.push({ label: `Database ${d.path.split("/").pop()}${n ? " (filtered)" : ""}`, source: { type: "db", db: d.path, filters }, db: d.path });
  }
  return out;
}
/** The kind a set most likely wants: the most common plottable kind among its files. */
function galleryKind(source) {
  const files = M().state.files || [];
  const inSet = source.type === "paths" ? files.filter((f) => source.paths.includes(f.path))
    : source.type === "folder" && source.folder ? files.filter((f) => f.path.startsWith(source.folder + "/")) : files;
  const n = (k) => inSet.filter((f) => f.kind === k).length;
  const kinds = [["band", n("band")], ["dos", n("dos")], ["ir", n("ir")], ["raman", n("raman")], ["structure", n("cif")],
    ["scf", inSet.filter((f) => f.kind === "out" && !M().isJobLog(f)).length]];
  const best = kinds.reduce((a, b) => (b[1] > a[1] ? b : a), ["band", 0]);
  return best[1] ? best[0] : "band";
}
/** Plot Gallery: a set (a folder, the Explorer's selection, a Batch Setup folder, a database query) and a kind. */
async function openGallery(opts = {}) {
  let source = opts.source, label = opts.label;
  if (!source) {
    const picked = [...(M().state.selected || [])];
    if (active && active.kind === "db") {
      const s = gallerySets().find((x) => x.source.type === "db" && x.source.db === active.path);
      source = s.source; label = s.label;
    } else if (picked.length >= 2) { source = { type: "paths", paths: picked }; label = `Explorer selection (${picked.length})`; }
    else { source = { type: "folder", folder: "" }; label = "This folder"; }
  }
  if (source.type === "folder" && /(^|\/)batch_[^/]*$/.test(source.folder || "") && M().fileByPath(`${source.folder}/batch.json`)) {
    source = { type: "batch", batch: source.folder };
    label = `Batch ${source.batch.split("/").pop()}`;
  }
  label = label || (source.type === "folder" ? (source.folder || "This folder") : "This set");
  // the same set already has a gallery: go back to it (a kind asked for switches it)
  const same = docs.find((x) => x.kind === "gallery" && x.g && JSON.stringify(x.g.source) === JSON.stringify(source));
  if (same) {
    if (opts.kind && same.g.kind !== opts.kind) { same.g.kind = opts.kind; same.g.index = 0; }
    if (active !== same) await activate(same);
    else await galleryViewer().then((v) => v.open(same));
    return same;
  }
  saveView(active);
  // the band/DOS window: Settings' (this session > the folder's mace-folder.toml > yours), as a band plot opens
  const win = settings.get("plots.energyWindow");
  const [emin, emax] = Array.isArray(win) && win.length === 2 && Number(win[0]) < Number(win[1]) ? [Number(win[0]), Number(win[1])] : [-6, 6];
  const d = { id: nextId++, path: null, kind: "gallery", fileKind: "", title: `Gallery: ${label}`,
    g: { source, sourceLabel: label, kind: opts.kind || galleryKind(source), cols: 3, shared: true, emin, emax, mode: "grid", index: 0,
      db: source.type === "db" ? source.db : undefined } };
  docs.push(d);
  await activate(d);
  return d;
}

/* The Database document (js/docs/db.js, loaded on first use): paged on the server, a virtual table. */
let dbView = null, dbViewLoading = null;
function dbViewer() {
  if (dbView) return Promise.resolve(dbView);
  if (!dbViewLoading) dbViewLoading = Promise.all([import("./docs/db.js"), import("./lazy/lazy_util.js")]).then(async ([m, lu]) => {
    await lu.loadCss(new URL("./docs/db.js", import.meta.url).href, "../../css/db.css");
    dbView = m.createDbView($("#db-host"), pieces.dbDetail, { api: (...a) => M().api(...a), q: (o) => M().q(o), esc, icon, toast,
      openPath: (p) => openDoc(p), setupFor: (p) => setupFor(p), setting: (k) => settings.get(k), dbBuild: (a) => dbBuild(a) });
    return dbView;
  }).catch((e) => { dbViewLoading = null; throw e; });
  return dbViewLoading;
}
async function openDbView(path, focusId = null) {
  const v = await dbViewer();
  $("#db-empty").classList.add("hidden");
  $("#db-host").hidden = false;
  try { await v.open(path, focusId); } catch (e) {
    $("#db-host").hidden = true;
    $("#db-empty").classList.remove("hidden");
    throw e;
  }
}
/* The Cluster and Job documents (js/docs/cluster.js, loaded on first use; area F). */
let clusterView = null, jobView = null, clusterLoading = null;
function clusterModule() {
  if (!clusterLoading) clusterLoading = Promise.all([import("./docs/cluster.js"), import("./lazy/lazy_util.js")]).then(async ([m, lu]) => {
    await lu.loadCss(new URL("./docs/cluster.js", import.meta.url).href, "../../css/cluster.css");
    return m;
  }).catch((e) => { clusterLoading = null; throw e; });
  return clusterLoading;
}
const clusterHelpers = () => ({ api: (...a) => M().api(...a), esc, icon, toast, files: () => M().state.files,
  copy: (t) => copyText(t), openJob: (id) => openJob(id), newWorkflow: (a) => openWorkflowBuilder(a), abbrev: (p) => (info ? abbrev(p, { home: info.home || "", max: 80 }) : p),
  apiTask: (p, b) => apiTask(p, b), confirm: (o) => SETUP_API.confirm(o), openFolderAt: (p) => openFolderDialog(p),
  retitle: () => { renderTabs(); updateTitle(); },
  openPath: (rel, kind) => (kind === "out" || kind === "stdout" || kind === "stderr" ? openAsText(rel) : openDoc(rel)) });
async function clusterViewer() {
  if (!clusterView) clusterView = (await clusterModule()).createClusterView($("#cluster-host"), clusterHelpers());
  return clusterView;
}
async function jobViewer() {
  if (!jobView) jobView = (await clusterModule()).createJobView($("#job-host"), clusterHelpers());
  return jobView;
}
function openCluster(tab = null) {
  saveView(active);
  let d = docs.find((x) => x.kind === "cluster");
  if (!d) {
    d = { id: nextId++, path: null, kind: "cluster", fileKind: "", title: clusterCaps && clusterCaps.cluster_name ? `Cluster: ${clusterCaps.cluster_name}` : "Cluster" };
    docs.push(d);
  }
  d.tab = tab;
  return activate(d);
}
function openJob(id) {
  id = String(id || "").trim();
  if (!/^\d+(_\d+)?$/.test(id)) { toast(`Not a job id: ${id}`, "error"); return null; }
  saveView(active);
  let d = docs.find((x) => x.kind === "job" && x.jobId === id);
  if (!d) { d = { id: nextId++, path: null, kind: "job", fileKind: "", jobId: id, title: `Job ${id}` }; docs.push(d); }
  return activate(d);
}
/* The Workflows document (js/docs/workflow.js, loaded on first use): read-only. */
let workflowView = null, workflowLoading = null;
function workflowViewer() {
  if (!workflowLoading) workflowLoading = Promise.all([import("./docs/workflow.js"), import("./lazy/lazy_util.js")]).then(async ([m, lu]) => {
    await lu.loadCss(new URL("./docs/workflow.js", import.meta.url).href, "../../css/workflow.css");
    if (!workflowView) workflowView = m.createWorkflowView($("#workflow-host"), clusterHelpers());
    return workflowView;
  }).catch((e) => { workflowLoading = null; throw e; });
  return workflowLoading;
}
function openWorkflows(key = null) {
  saveView(active);
  let d = docs.find((x) => x.kind === "workflows");
  if (!d) { d = { id: nextId++, path: null, kind: "workflows", fileKind: "", title: "Workflows" }; docs.push(d); }
  // a workflow to show (one the builder just started): selected when the view lists it
  if (key) { d.wfKey = key; d.wfNode = null; }
  const wasActive = active === d;
  const r = activate(d);
  if (key && wasActive && workflowView) workflowView.open(d);
  return r;
}
/* New Workflow… (js/lazy/workflow_dialog.js): the builder writes a spec and runs the CLI; nothing runs until its Submit. */
function openWorkflowBuilder(a) {
  // Clone (the Workflows view): the builder pre-filled from an existing workflow (GET workflow/clone)
  if (a && a.clone) return lazy("workflow_dialog").then((m) => m.open(H, { clone: a.clone }));
  const p = a && Array.isArray(a.paths) && a.paths.length ? a.paths : a && a.dir !== undefined ? [a.dir || "."]
    : (() => { const x = pathOf(a) || (active && active.path); return x && /\.(cif|d12)$/i.test(x) ? [x] : []; })();
  return lazy("workflow_dialog").then((m) => m.open(H, { targets: p }));
}
/* jobs.finished: a job of yours left the queue. The server sends one event per
 * job (the Jobs panel's refresh and the Cluster view's poller share one record
 * of what was announced); this tab acts once per job, also across reloads:
 * one toast, a Problems entry unless it completed, and a browser notification
 * when Settings › Cluster › Notify is on. */
function notifiedIds() { try { return new Set(JSON.parse(sessionStorage.getItem("mace.cluster.notified") || "[]")); } catch (e) { return new Set(); } }
function jobFinished(ev) {
  if (!ev || !ev.id) return;
  const seen = notifiedIds();
  if (seen.has(ev.id)) return;
  seen.add(ev.id);
  try { sessionStorage.setItem("mace.cluster.notified", JSON.stringify([...seen].slice(-500))); } catch (e) { /* private window */ }
  window.__maceJobEvents = (window.__maceJobEvents || []).concat([ev]);
  const known = ev.known !== undefined ? ev.known !== false : !!ev.state && !ev.left_queue;
  const state = known ? String(ev.state || "") : "";
  const ok = state === "COMPLETED";
  const name = ev.name || `Job ${ev.id}`;
  const text = known ? `${name} (${ev.id}) finished: ${state}` : `${name} (${ev.id}) left the queue`;
  const action = ev.out_rel ? { label: "Open Output", run: () => openAsText(ev.out_rel) } : null;
  toast(text, ok ? "success" : known ? "error" : "info", action);
  if (!ok && known) addProblem({ severity: "error", source: "Jobs", message: text, action });
  if (settings.get("cluster.notify") && typeof Notification !== "undefined" && Notification.permission === "granted") {
    try {
      const n = new Notification(`MACE: job ${ev.id} ${known ? state.toLowerCase() : "left the queue"}`, { body: ev.name || "", tag: `mace-job-${ev.id}` });
      n.onclick = () => { window.focus(); openJob(ev.id); n.close(); };
    } catch (e) { /* no notifications here */ }
  }
  if (jobsView) jobsView.refresh(false);
  if (jobView) jobView.refreshIf(ev.id);
  document.dispatchEvent(new CustomEvent("mace-job-finished", { detail: ev }));
}
let storageChip = null;
async function refreshStorageChip() {
  try {
    const q = await M().api("cluster/quotas");
    const rows = (q.rows || []).filter((r) => r.level !== "info");
    const lvl = (v) => (v === null || v === undefined ? 0 : v >= q.critical_percent ? 2 : v >= q.warn_percent ? 1 : 0);
    let best = null;
    for (const r of rows) {
      for (const [k, word] of [["files_pct", "files"], ["space_pct", "space"]]) {
        const l = lvl(r[k]);
        // a file-system-wide df row (no quota tool) counts only from the critical level
        if (!l || (!q.have_quota && l < 2)) continue;
        if (!best || r[k] > best.v) best = { v: r[k], level: l === 2 ? "critical" : "warning", text: `${r.label.replace(/\s*\(.*\)$/, "")} ${word} ${r[k]} %` };
      }
    }
    storageChip = best;
  } catch (e) { storageChip = null; }
  renderStatus();
}
async function initCluster() {
  try { clusterCaps = await M().api("cluster/capabilities"); } catch (e) { clusterCaps = null; }
  renderActivity();
  if (clusterCaps && (clusterCaps.slurm || (clusterCaps.quota_sources || []).length || clusterCaps.df)) {
    refreshStorageChip();
    setInterval(refreshStorageChip, 30 * 60 * 1000);
  }
  for (const ev of (clusterCaps && clusterCaps.events) || []) { const s = notifiedIds(); s.add(ev.id); try { sessionStorage.setItem("mace.cluster.notified", JSON.stringify([...s])); } catch (e) { /* */ } }
}

/** Export Table: the active table (the database's filtered, sorted rows) as a download. */
async function exportTable(fmt) {
  if (active && active.kind === "db" && dbView) {
    const r = await M().api(dbView.exportUrl(fmt).replace(/^api\//, ""), { blob: true });
    download(r.blob, dbView.exportName(fmt));
    return true;
  }
  if (tableExporters.has(activeTableKind())) return tableExporters.get(activeTableKind())(fmt);
  toast("No table is shown", "info");
  return false;
}
const tableExporters = new Map();         // "compare" | "modes" | "coordination" -> fn(fmt), registered by those views
tableExporters.set("compare", (fmt) => {
  const r = M().cmpExport(fmt);
  if (!r) { toast("No table is shown", "info"); return false; }
  download(r.blob, r.name);
  return true;
});
document.addEventListener("mace-copy", (e) => copyText(e.detail.text, e.detail.done));
function activeTableKind() {
  if (active && active.kind === "db") return "db";
  if (active && active.tableKind) return active.tableKind();
  const pane = $(".bottom-pane.active");
  return pane && pane.dataset.table ? pane.dataset.table : "";
}
function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; a.hidden = true;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
  window.__maceLastDownload = name;
  toast(`Downloaded ${name}`, "success");
}

/* Open as Text: any file, any size, through 256 KB windows (js/docs/text.js, loaded on first use). */
let textView = null, textViewLoading = null;
function textViewer() {
  if (textView) return Promise.resolve(textView);
  if (!textViewLoading) textViewLoading = import("./docs/text.js").then((m) => {
    textView = m.createTextView($("#text-view"), { api: (...a) => M().api(...a), q: (o) => M().q(o), esc, toast,
      highlight: (t) => M().highlightDeck(t), apiTask: (p, b) => apiTask(p, b) });
    return textView;
  }).catch((e) => { textViewLoading = null; throw e; });
  return textViewLoading;
}
/* ---- plan E1/E2/E14 documents in the text panel: info, properties outputs, job logs */
async function openInfoLike(path, extra) {
  const existing = docs.find((d) => d.kind === "text" && d.srcPath === path && d[extra.flag]);
  if (existing) { await activate(existing); return existing; }
  saveView(active);
  const f = M().fileByPath(path);
  const d = { id: nextId++, path: null, srcPath: path, kind: "text", fileKind: fileKind(path), title: path.split("/").pop(),
    size: f ? f.size : 0, ...extra.fields, [extra.flag]: true };
  docs.push(d);
  pushRecent(path);
  await activate(d);
  return d;
}
/** A file MACE lists but does not open (a wave function, CRYSTAL's restart data, fort.25, transport data). */
function openInfo(path) { return openInfoLike(path, { flag: "info", fields: {} }); }
/** A properties output: what it computed, its data files and the SCF output it ran from. */
function openProps(path) { return openInfoLike(path, { flag: "props", fields: { fileKind: "props" } }); }
/* Volume documents (js/docs/volume.js, plan E5): a cube's isosurfaces, slices, MEP and differences. */
let volumeView = null, volumeViewLoading = null;
function volumeViewer() {
  if (volumeView) return Promise.resolve(volumeView);
  if (!volumeViewLoading) volumeViewLoading = Promise.all([import("./docs/volume.js"), import("./lazy/lazy_util.js")]).then(async ([m, lu]) => {
    await Promise.all([lu.loadCss(new URL("./docs/volume.js", import.meta.url).href, "../../css/volume.css"),
      lu.loadCss(new URL("./docs/volume.js", import.meta.url).href, "../../css/plot.css")]);
    volumeView = m.createVolumeView($("#volume-host"), pieces.volumeOpts, { api: (...a) => M().api(...a), q: (o) => M().q(o), esc, icon, toast,
      apiTask: (p, b, u, x) => apiTask(p, b, u, x), M, setting: (k) => settings.get(k),
      // the toolbar's display options, as the structure view takes them (the range and boundary atoms per document)
      viewOpts: () => viewOpts(), setSetting: (k, v) => settings.set(k, v), setRange: (n) => setRange(n),
      ownRange: () => DS.isOverride(viewDoc(), "displayHi") || DS.isOverride(viewDoc(), "displayLo"),
      attachTriad: (v, wrap) => attachVolumeTriad(v, wrap), updateTriad: () => updateTriad(),
      colorByPotential: (d) => colorByPotential(d), difference: (d) => volumeDifference(d), saveResult: (d) => saveVolumeResult(d),
      renderWithCli: (p) => { const f = M().fileByPath(p); if (f) run("explorer.plot", { path: p }); } });
    return volumeView;
  }).catch((e) => { volumeViewLoading = null; throw e; });
  return volumeViewLoading;
}
async function openVolume(path, extra = null) {
  const existing = path ? docs.find((x) => x.kind === "volume" && x.path === path) : null;
  if (existing) { await activate(existing); return existing; }
  saveView(active);
  const d = { id: nextId++, path, kind: "volume", fileKind: "cube", title: path ? path.split("/").pop() : extra.title, ...(extra || {}) };
  docs.push(d);
  if (path) pushRecent(path);
  await activate(d);
  return d;
}
/* CRYSTAL23 ORBITALS (.molden): the Volume document with the orbital picker (js/docs/orbitals.js).
 * The picker lists every k point of the run; it starts at the k point of the file that was opened. */
async function openOrbitals(path) {
  const m = /_(\d{5})_K\d+_(real|complex)\.molden$/i.exec(path);
  return openVolume(path, { fileKind: "orbitals", orbitals: { index: null, request: null, startK: m ? Number(m[1]) : null } });
}
const cubeFiles = () => (M().state.files || []).filter((f) => f.kind === "cube");
/** The potential that goes with a density: the same stem's POT cube, else a choice. */
function potentialFor(d) {
  const p = d.path || "", base = p.replace(/(DENS|SPIN)(_CUBE\.DAT|\.CUBE|\.cube)$/i, "");
  return cubeFiles().find((f) => f.path !== p && f.path.startsWith(base) && /POT(_CUBE\.DAT|\.CUBE|\.cube)$/i.test(f.path)) || null;
}
async function colorByPotential(d) {
  d = d || (active && active.kind === "volume" ? active : null);
  if (!d || !d.meta) { toast("Open a density volume first", "info"); return; }
  const guess = potentialFor(d);
  // two editor groups: the other group's volume is the potential, no question asked
  const beside = GR && GR.otherVolume();
  const pick = beside && beside.path && beside.path !== d.path ? beside.path
    : await chooseCube("Color Density by Potential", "Potential", guess ? guess.path : null, d.path);
  if (!pick) return;
  if (d.path) {
    const chk = await apiTask("volume/check", { a: d.path, b: pick });
    if (chk.problem) { toast(`MEP: ${chk.problem}`, "error"); return; }
  }
  const v = await volumeViewer();
  try { await v.setColor(pick); } catch (e) { toast(`MEP: ${e.message}`, "error"); }
}
/** A small dialog: pick a cube (and, for a difference, coefficients). */
function chooseCube(title, label, preset, exclude) {
  return new Promise((resolve) => {
    const list = cubeFiles().filter((f) => f.path !== exclude);
    if (!list.length) { toast("No other cube files in this folder", "info"); resolve(null); return; }
    const body = document.createElement("div");
    body.innerHTML = `<label class="field"><span>${esc(label)}</span><select id="cube-pick">${list.map((f) =>
      `<option value="${esc(f.path)}"${f.path === preset ? " selected" : ""}>${esc(f.path)}</option>`).join("")}</select></label>`;
    let done = false;
    const dlg = dialog({ id: "cube-dialog", title, body, narrow: true,
      foot: `<button class="btn" data-close>Cancel</button><button class="btn primary" id="cube-ok">Apply</button>`,
      onClose: () => { if (!done) resolve(null); } });
    $("[data-close]", dlg.querySelector(".dlg-foot")).addEventListener("click", () => dlg.close());
    $("#cube-ok").addEventListener("click", () => { done = true; const v = $("#cube-pick").value; dlg.close(); resolve(v); });
  });
}
async function volumeDifference(d) {
  d = d || (active && active.kind === "volume" ? active : null);
  const list = cubeFiles();
  if (list.length < 2) { toast("A difference needs two cube files in this folder", "info"); return; }
  const first = d && d.path ? d.path : list[0].path;
  const body = document.createElement("div");
  const row = (k, path, coef) => `<div class="po-row vd-row"><input type="number" step="any" value="${coef}" class="vd-coef" aria-label="Coefficient ${k + 1}">
    <select class="vd-path" aria-label="Volume ${k + 1}"><option value="">—</option>${list.map((f) => `<option value="${esc(f.path)}"${f.path === path ? " selected" : ""}>${esc(f.path)}</option>`).join("")}</select></div>`;
  const beside = GR && GR.otherVolume();
  const second = beside && beside.path && beside.path !== first ? beside.path : (list.find((f) => f.path !== first) || {}).path;
  body.innerHTML = `<p class="muted small">Result = Σ coefficient × volume. Grids that share a cell and spacing but differ in size are aligned as mace plotting does.</p>
    ${row(0, first, 1)}${row(1, second, -1)}${row(2, "", -1)}`;
  const dlg = dialog({ id: "vdiff-dialog", title: "Volume Difference", body,
    foot: `<button class="btn" data-close>Cancel</button><button class="btn primary" id="vdiff-ok">Compute</button>` });
  $("[data-close]", dlg.querySelector(".dlg-foot")).addEventListener("click", () => dlg.close());
  $("#vdiff-ok").addEventListener("click", async () => {
    const rows = $$(".vd-row", dlg).map((r) => [r.querySelector(".vd-path").value, Number(r.querySelector(".vd-coef").value)]).filter(([p]) => p);
    dlg.close();
    try {
      const meta = await apiTask("volume/combine", { paths: rows.map((r) => r[0]), coefficients: rows.map((r) => r[1]),
        ...meshGridQuery() });
      await openVolume(null, { title: meta.name, combo: true, meta });
    } catch (e) { toast(`Volume difference: ${e.message}`, "error"); }
  });
}
async function saveVolumeResult(d) {
  if (!d || !d.combo) return;
  const stem = d.meta.combined.paths[0].replace(/\.[^/.]+(\.[^/.]+)?$/, "");
  const name = prompt("Save the difference as (a new .CUBE file in this folder)", `${stem}_difference.CUBE`);
  if (!name) return;
  try {
    const r = await M().api("volume/save", { body: { paths: d.meta.combined.paths, coefficients: d.meta.combined.coefficients, name, title: d.title } });
    toast(`Saved ${r.path}`, "success");
    M().loadFiles();
  } catch (e) { toast(`Save Result As: ${e.message}`, "error"); }
}
/** An output's structure even when it opens as Modes by default (the Modes header's Structure). */
function openStructure(path) { return openDoc(path, { as: "structure" }); }
/** A CRYSTAL output with no geometry block: its text, and a Problems row that says why. */
async function openNoGeometry(path) {
  const d = await openAsText(path);
  if (d) {
    clearProblems(d.id, "Output");
    addProblem({ severity: "warning", source: "Output", doc: d.id,
      message: "No geometry in this output: CRYSTAL stopped before printing one (see the end of the output)",
      action: { label: "Show End", run: () => { if (textView) textView.jump(textView.total()); } } });
  }
  return d;
}
/** A SLURM job log: read-only text with a header linking its calculation (plan E14). */
async function openJobLog(path) {
  const f = M().fileByPath(path);
  const pre = docs.find((x) => x.kind === "text" && x.srcPath === path && !x.deck && !x.info && !x.props);
  if (pre) { pre.jobLog = true; pre.fileKind = "joblog"; }
  else if (f) {
    // made here so the first paint already has the header
    saveView(active);
    await textViewer();
    const d = { id: nextId++, path: null, srcPath: path, kind: "text", fileKind: "joblog", title: path.split("/").pop(),
      windowed: true, size: f.size, hl: false, jobLog: true };
    docs.push(d);
    pushRecent(path);
    await activate(d);
    return d;
  }
  return openAsText(path);
}
async function renderJobHeader(d) {
  const host = $("#text-job");
  host.hidden = false;
  if (!d.link) {
    host.innerHTML = `<span class="muted">${icon("scroll-text")} Job log · finding its calculation…</span>`;
    try { d.link = await M().api("joblog/link?" + M().q({ path: d.srcPath })); }
    catch (e) { d.link = { error: e.message }; }
    if (active !== d) return;
    if (d.link && d.link.errors_first_line) {
      clearProblems(d.id, "Job errors");
      addProblem({ severity: "warning", source: "Job errors", doc: d.id, message: `Job errors: ${d.link.errors_first_line}` });
    }
  }
  const L = d.link || {};
  const job = L.job_id ? ` · job ${esc(L.job_id)}${L.array ? ` ${esc(L.array)}` : ""}` : "";
  const calc = L.busy ? `<span class="warn">Database busy, retrying…</span>`
    : L.calc ? `Calculation: <span class="mono">${esc(L.calc.split("/").pop())}</span> <button class="tb-btn-text" data-job="open">${icon("external-link")}<span>Open</span></button>`
    : L.error ? esc(L.error) : "No calculation found for this log";
  host.innerHTML = `${icon("scroll-text")}<span>Job log${job} · ${calc}</span>`;
  if (L.busy) setTimeout(() => { if (active === d) { d.link = null; renderJobHeader(d); } }, 1500);
}
$("#text-job") && $("#text-job").addEventListener("click", (e) => {
  if (!e.target.closest("[data-job=open]") || !active || !active.link || !active.link.calc) return;
  openDoc(active.link.calc);
});
async function renderInfoDoc(d) {
  const host = $("#text-info");
  host.hidden = false;
  $("#text-body").hidden = true; $("#text-view").hidden = true;
  const route = d.props ? "props" : "fileinfo";
  if (!d.payload) {
    host.innerHTML = `<p class="muted"><span class="spinner sm"></span> Reading…</p>`;
    try { d.payload = await M().api(`${route}?` + M().q({ path: d.srcPath })); }
    catch (e) { d.payload = { error: e.message }; }
    if (active !== d) return;
  }
  const P = d.payload;
  const mb = (n) => (n >= 2 ** 20 ? `${(n / 2 ** 20).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`);
  if (P.error) { host.innerHTML = `<p class="warn">${esc(P.error)}</p>`; return; }
  if (d.props) {
    const LBL = { band: "Open Band Structure", dos: "Open Density of States", cube: "Open Volume", orbitals: "Open Orbitals", f25: "Open fort.25",
      transport: "Open Transport Data", image: "Open Image", html: "Open Plot", ir: "Open IR Spectrum", raman: "Open Raman Spectrum" };
    host.innerHTML = `<h3>Properties output · ${esc(P.calc || "properties")}</h3>
      <p>${esc(P.status)}${P.errors && P.errors.length ? ` · <span class="warn">${esc(P.errors[0])}</span>` : ""}</p>
      ${P.files.length ? `<p class="muted small">Data files of this run</p><div class="info-acts">${P.files.map((f) =>
        `<button class="tb-btn-text" data-info-open="${esc(f.path)}" title="${esc(f.name)}">${icon(f.kind === "cube" || f.kind === "orbitals" ? "box" : f.kind === "image" ? "image" : "chart-line")}<span>${esc(LBL[f.kind] || "Open " + f.kind)}</span><span class="muted small">${esc(f.name.slice(-28))}</span></button>`).join("")}</div>`
        : `<p class="muted">No data files with this run's name are in the folder.</p>`}
      <div class="info-acts">${P.parent ? `<button class="tb-btn-text" data-info-open="${esc(P.parent)}">${icon("atom")}<span>Structure from ${esc(P.parent.split("/").pop())}</span></button>` : ""}
      <button class="tb-btn-text" data-info-text="${esc(P.path)}">${icon("file-text")}<span>Show Output</span></button></div>`;
    return;
  }
  const WHAT = { wavefunction: ["CRYSTAL wave function", "MACE uses it for GUESSP restarts and properties runs; it is not opened here."],
    crystaldata: ["CRYSTAL data file", "CRYSTAL writes it for restarts and analyses; it is not opened here."],
    f25: ["fort.25 plot data", "The same data as the run's .DAT files: open those for the plot."],
    transport: ["Transport data (BOLTZTRA)", "Seebeck, conductivity and related tensors, and the ZT, power factor and Lorenz number they give: open any of the run's files for its plot."],
    cube: ["Volumetric grid (Gaussian cube)", "Volumes open in the Volume viewer."] }[P.kind] || ["File", "MACE does not open this kind of file."];
  if (P.kind === "f25" && P.f25_kind) {
    const word = { BAND: "a BAND run", DOSS: "a DOSS run", MAPN: "a 2D map (ECHG or POTM)" }[P.f25_kind];
    const open = { BAND: "Open Band Structure", DOSS: "Open Density of States" }[P.f25_kind];
    host.innerHTML = `<h3>fort.25 of ${esc(word)} (${mb(P.size)})</h3>
      <p>${P.dat ? `The same data as <span class="mono">${esc(P.dat.split("/").pop())}</span>, which MACE plots.` : P.f25_kind === "MAPN"
        ? "It opens as a contour plot when you open the file." : "MACE plots the run's .DAT file, which is not in this folder."}</p>
      <div class="info-acts">${P.dat ? `<button class="tb-btn-text" data-info-open="${esc(P.dat)}">${icon("chart-line")}<span>${esc(open)}</span></button>` : ""}
      <button class="tb-btn-text" data-info-text="${esc(P.path)}">${icon("file-text")}<span>Open as Text</span></button></div>`;
    return;
  }
  host.innerHTML = `<h3>${esc(WHAT[0])} (${mb(P.size)})</h3><p>${esc(WHAT[1])}</p>
    <div class="info-acts">${P.sibling ? `<button class="tb-btn-text" data-info-open="${esc(P.sibling)}">${icon("external-link")}<span>Reveal the run's output</span></button>` : ""}
    ${P.kind !== "wavefunction" ? `<button class="tb-btn-text" data-info-text="${esc(P.path)}">${icon("file-text")}<span>Open as Text</span></button>` : ""}</div>`;
}
$("#text-info") && $("#text-info").addEventListener("click", (e) => {
  const o = e.target.closest("[data-info-open]"), t = e.target.closest("[data-info-text]");
  if (o) openDoc(o.dataset.infoOpen);
  else if (t) openAsText(t.dataset.infoText);
});
async function openAsText(path) {
  const existing = docs.find((d) => d.kind === "text" && d.srcPath === path && !d.deck && !d.info && !d.props);
  if (existing) { await activate(existing); return existing; }
  saveView(active);
  const f = M().fileByPath(path);
  const d = { id: nextId++, path: null, srcPath: path, kind: "text", fileKind: fileKind(path), title: path.split("/").pop(),
    windowed: true, size: f ? f.size : 0, hl: ["d12", "d3", "json"].includes(fileKind(path)) };
  await textViewer();
  docs.push(d);
  await activate(d);
  return d;
}
function renderTextDoc(d) {
  $("#text-path").textContent = d.deck ? d.title : d.srcPath;
  $("#text-job").hidden = !d.jobLog;
  $("#text-info").hidden = !(d.info || d.props);
  if (d.info || d.props) {
    $("#tab-text").classList.remove("stale");
    if ($("#text-acts")) $("#text-acts").innerHTML = "";
    $("#text-note").textContent = d.props ? "Properties output" : "Not opened here";
    renderInfoDoc(d);
    return;
  }
  if (d.jobLog) renderJobHeader(d);
  const src = d.setupFor ? docs.find((x) => x.id === d.setupFor) : null;
  const st = src && setupMod ? setupMod.state(src) : null;
  const stale = !!(st && st.result && (st.stale || st.previewSha !== st.result.json_sha256));
  let acts = $("#text-acts");
  if (!acts) { $(".text-head").insertAdjacentHTML("beforeend", `<span class="text-acts" id="text-acts"></span>`); acts = $("#text-acts"); }
  if (d.deck) {
    const diff = d.prevText !== null && d.prevText !== undefined && d.prevText !== d.text ? deckDiff(d) : null;
    $("#text-note").innerHTML = `${stale ? '<span class="warn">Out of date · </span>' : ""}written by <code class="mono">${esc(d.command || "the CLI")}</code>`;
    acts.innerHTML = (diff ? `<button class="tb-btn-text${d.showDiff ? " on" : ""}" data-text="diff" aria-pressed="${!!d.showDiff}">${diff.n} line${diff.n === 1 ? "" : "s"} changed</button>` : "") +
      (src ? `<button class="tb-btn-text" data-text="setup">${icon("flask-conical")}<span>Open in Setup</span></button>` : "") +
      `<button class="tb-btn-text" data-text="copy">${icon("copy")}<span>Copy</span></button>`;
    $("#tab-text").classList.toggle("stale", stale);
    $("#text-body").hidden = false; $("#text-view").hidden = true;
    $("#text-body").innerHTML = d.showDiff && diff ? diff.html : M().highlightDeck(d.text);
    return;
  }
  $("#tab-text").classList.remove("stale");
  acts.innerHTML = /\.json$/i.test(d.srcPath || "") ? `<button class="tb-btn-text" data-text="openSetup">${icon("flask-conical")}<span>Open in Setup</span></button>` : "";
  if (d.windowed) {
    $("#text-body").hidden = true; $("#text-view").hidden = false;
    const mb = d.size >= 2 ** 20 ? `${(d.size / 2 ** 20).toFixed(1)} MB` : `${Math.max(1, Math.round(d.size / 1024))} KB`;
    $("#text-note").textContent = `Read only · ${mb}${d.size > 256 * 1024 ? " · read in 256 KB windows as you scroll" : ""}`;
    textViewer().then((v) => { if (active === d) v.open(d); });
    return;
  }
  $("#text-body").hidden = false; $("#text-view").hidden = true;
  $("#text-note").textContent = d.truncated ? "Showing the first 400 KB" : "Read only";
  $("#text-body").innerHTML = M().highlightDeck(d.text);
}
/** The previous preview against this one, line by line (plan B4: "2 lines changed"). */
function deckDiff(d) {
  const a = (d.prevText || "").split("\n"), b = (d.text || "").split("\n");
  const n = a.length, m = b.length, L = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) L[i][j] = a[i] === b[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  const out = [];
  let i = 0, j = 0, changed = 0;
  while (i < n || j < m) {
    if (i < n && j < m && a[i] === b[j]) { out.push(`<span class="ctx">  ${esc(b[j])}</span>`); i++; j++; }
    else if (j < m && (i >= n || L[i][j + 1] >= L[i + 1][j])) { out.push(`<span class="add">+ ${esc(b[j++])}</span>`); changed++; }
    else { out.push(`<span class="del">- ${esc(a[i++])}</span>`); changed++; }
  }
  return { n: changed, html: out.join("\n") };
}
$("#tab-text").addEventListener("click", (e) => {
  const b = e.target.closest("[data-text]");
  if (!b || !active || active.kind !== "text") return;
  const d = active;
  if (b.dataset.text === "diff") { d.showDiff = !d.showDiff; renderTextDoc(d); }
  else if (b.dataset.text === "copy") copyText(d.text, "Deck copied");
  else if (b.dataset.text === "setup") { const src = docs.find((x) => x.id === d.setupFor); if (src) { activate(src); if (!setupOpen) toggleSetup(true); } }
  else if (b.dataset.text === "openSetup") toast("Open the structure or output this file is for, then Calculation Setup › Advanced", "info");
});
/** A Setup preview's deck as a Deck document beside the structure (spec §1.2): made on the
 * first preview, refreshed in place after; shown when the preview was asked for. */
function setupDeck(src, r, show) {
  if (!r || !r.deck) return null;
  let d = docs.find((x) => x.setupFor === src.id && x.deck);
  if (!d) {
    d = { id: nextId++, path: null, srcPath: r.deck_name, kind: "text", fileKind: /\.d3$/i.test(r.deck_name) ? "d3" : "d12",
      title: r.deck_name, text: r.deck, deck: true, setupFor: src.id, command: r.command, prevText: null };
    docs.push(d);
  } else if (d.text !== r.deck || d.title !== r.deck_name) {
    // a line diff only means something between two previews of the same deck (not SP .d12 -> BAND .d3)
    d.prevText = d.title === r.deck_name ? d.text : null; d.text = r.deck; d.title = r.deck_name; d.srcPath = r.deck_name; d.command = r.command;
    d.fileKind = /\.d3$/i.test(r.deck_name) ? "d3" : "d12";
  }
  if (show && active !== d) { saveView(active); activate(d); }
  else { renderTabs(); renderOpenEditors(); if (active === d) renderTextDoc(d); }
  return d;
}
const STRUCT_KINDS = ["cif", "d12", "out"];
function canCompare(path) {
  const a = active && active.path;
  return !!path && !!a && path !== a && STRUCT_KINDS.includes(fileKind(path)) && STRUCT_KINDS.includes(fileKind(a));
}
async function compareWith(path) {
  const a = active.path;
  try {
    const r = await M().api("compare?" + M().q({ a, b: path }));
    const t = { frames: [r.a, r.b], energies_au: [r.a.energy_au ?? null, r.b.energy_au ?? null], labels: ["A", "B"], comparison: r.comparison };
    saveView(active);
    const d = { id: nextId++, path: null, kind: "trajectory", title: `${a.split("/").pop()} ↔ ${path.split("/").pop()}` };
    docs.push(d); active = d; panelOwner.opt = d.id; panelOwner.structure = d.id;
    M().setTrajectory(t, `${a}  →  ${path}`, "two structures, same atom order");
    M().showTab("opt");
    showBottom("timeline");
    afterSwitch();
  } catch (e) { toast(e.message, "error"); }
}
/** Analysis › Compare with Input: an output against the structure it started from
 * (the .d12 of the same name, else the parent CIF its provenance names, else frame 0). */
async function compareInput(path) {
  path = path || docPath(active);
  if (!path) return;
  try {
    const r = await M().api("compare/input?" + M().q({ path }));
    const lab = { d12: "input deck", parent: "parent CIF", frame0: "first frame" }[r.input.kind];
    const t = { frames: [r.a, r.b], energies_au: [r.a.energy_au ?? null, r.b.energy_au ?? null], labels: [lab, "final"],
      comparison: r.comparison, input: r.input };
    saveView(active);
    const d = { id: nextId++, path: null, kind: "trajectory", title: `${path.split("/").pop()} vs its input`, compareOf: path };
    docs.push(d); active = d; panelOwner.opt = d.id; panelOwner.structure = d.id;
    const note = r.comparison.match === "position" ? `, atoms matched by position (max mismatch ${M().fmt(r.comparison.max_mismatch, 3)} Å)` : "";
    M().setTrajectory(t, `${r.input.path}  →  ${path}`, `against its ${lab}${note}` + (r.tried && r.tried.length ? ` (skipped: ${r.tried.join("; ")})` : ""));
    M().showTab("opt");
    showBottom("timeline");
    afterSwitch();
  } catch (e) { toast(e.message, "error"); }
}
/** Analysis › Energy / Bond Length Table / Displacement Table: the Timeline's sub-tab. */
function compareSub(sub) {
  showBottom("timeline");
  M().showSub(sub);
}
async function setupFor(path) {
  const d = await openDoc(path);
  if (!setupOpen) toggleSetup(true);
  else if (setupMod) setupMod.refresh();
  return d;
}
/* "@O3" in the palette: center the view on the atom and show its readout (the
 * selection itself arrives with the editor tools in Phase 3). */
function jumpToAtom(i) {
  const s = structureOf(active);
  if (!s) return;
  centerOnAtom({ i, image: [0, 0, 0], symbol: s.symbols[i], s });
  const tip = $("#hover-tip");
  if (tip) {
    tip.textContent = `${s.symbols[i]}${i + 1}  (${s.frac[i].map((x) => x.toFixed(4)).join(", ")})` +
      (s.symmetry && s.symmetry.available ? `  Wyckoff ${s.symmetry.wyckoffs[i]}` : "");
    tip.classList.remove("hidden");
  }
}
/* An atom from a context menu: {i, image}; the symbol and coordinates come from the structure. */
function atomOf(a) {
  // a run's atom: the step on screen
  const t = active && active.kind === "trajectory" && M().state.traj;
  const s = t && t.frames[M().state.frame] && t.frames[M().state.frame].frac ? t.frames[M().state.frame] : structureOf(active);
  if (!a || !a.atom || !s || !(a.atom.i >= 0) || a.atom.i >= s.symbols.length) return null;
  return { i: a.atom.i, image: a.atom.image || [0, 0, 0], symbol: s.symbols[a.atom.i], s };
}
/* 3Dmol's view is [model translation x y z, zoom, rotation quaternion]; the
 * translation is minus the point at the center, so a periodic image (which
 * shares its atom's serial) is centered on exactly, not on its parent. */
function centerOnAtom(at) {
  const v = mainViewer();
  if (!v || !at) return;
  const c = atomCart(at), view = v.getView();
  view[0] = -c[0]; view[1] = -c[1]; view[2] = -c[2];
  v.setView(view);
  v.render(); updateTriad();
}
function atomFrac(at) { return at.s.frac[at.i].map((x, k) => x + (at.image[k] || 0)); }
function atomCart(at) { const f = atomFrac(at), C = at.s.cell; return [0, 1, 2].map((k) => f[0] * C[0][k] + f[1] * C[1][k] + f[2] * C[2][k]); }
function copyAtom(at, mode) {
  const d = settings.get("decimals.coord") || 5;
  const v = mode === "frac" ? atomFrac(at) : atomCart(at);
  const text = v.map((x) => { const t = x.toFixed(d); return /^-0\.?0*$/.test(t) ? t.slice(1) : t; }).join(" ");
  return copyText(text, `Copied ${at.symbol}${at.i + 1} (${mode === "frac" ? "fractional" : "Cartesian, Å"})`);
}

/* ================================================================ saving, drafts, export (T11) */
const dirHash = () => (info && info.dir_hash) || (M().state.info && M().state.info.dir_hash) || "local";
const editorDocs = () => docs.map(edOf).filter(Boolean);
const saveDialog = () => lazy("save_dialog");
const stemOf = (d) => ((d && (d.path || d.title)) || "structure").split("/").pop().replace(/\.[^.]+$/, "");
const fmtTime = (t) => new Date(t).toLocaleString("en-US", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", hour12: false });

/** The first in-place save of a file: "Overwrite MgO.cif? …" with Don't ask again. */
async function confirmOverwrite(e) {
  const name = e.path.split("/").pop(), m = await saveDialog();
  const bak = M().fileByPath(e.path + ".bak");
  const keep = !settings.get("keepBackup") ? "No backup is kept (Settings › Keep a .bak)."
    : bak ? `A backup from ${fmtTime((bak.mtime || 0) * 1000)} is kept as ${esc(name)}.bak and stays as it is.`
    : `The original will be kept as ${esc(name)}.bak.`;
  let remember = false;
  const body = document.createElement("div");
  body.innerHTML = `<p>Overwrite ${esc(name)}? ${keep}</p><label class="chk"><input type="checkbox" data-k="remember"> Don't ask again for this file</label>`;
  body.querySelector("input").addEventListener("change", (ev) => { remember = ev.target.checked; });
  const v = await m.ask(H, { id: "save-confirm", title: "Save", body,
    buttons: [["cancel", "Cancel"], ["overwrite", "Overwrite", true]] });
  return v === "overwrite" ? (remember ? "remember" : true) : false;
}
async function changedOnDisk(e) {
  const name = e.path.split("/").pop(), m = await saveDialog();
  return m.ask(H, { id: "save-changed", title: "Changed on disk",
    body: `<p>${esc(name)} changed on disk since you opened it.</p><p class="insp-hint">Overwrite replaces it with this document; Save As… keeps both.</p>`,
    buttons: [["cancel", "Cancel"], ["saveas", "Save As…"], ["overwrite", "Overwrite", true]] });
}
async function confirmReplace(path) {
  const m = await saveDialog();
  return (await m.ask(H, { id: "save-replace", title: "Replace?", body: `<p>${esc(path)} already exists. Replace it?</p>`,
    buttons: [["cancel", "Cancel"], ["replace", "Replace", true]] })) === "replace";
}
function afterSave(e, path, r) {
  const d = e.doc;
  d.title = e.name;
  if (path) pushRecent(path);
  if (E && E.drafts) E.drafts.write(e);
  M().loadFiles().then(() => { M().state.active = d.path || null; M().renderFiles(); }).catch(() => {});
  document.dispatchEvent(new CustomEvent("mace-saved", { detail: { path, r } }));
  afterSwitch();
}
async function revert(d = active) {
  const e = edOf(d);
  if (!e || !d.path) return;
  const m = await saveDialog();
  const v = await m.ask(H, { id: "revert-confirm", title: "Revert File",
    body: `<p>Revert ${esc(d.title)} to the file on disk? This drops its unsaved changes and its undo history.</p>`,
    buttons: [["cancel", "Cancel"], ["revert", "Revert", true]] });
  if (v !== "revert") return;
  const i = docs.indexOf(d), path = d.path;
  closeDoc(d, true);
  await openDoc(path);
  const n = docs.findIndex((x) => x.path === path);
  if (n >= 0 && i >= 0 && n !== i) { const [x] = docs.splice(n, 1); docs.splice(Math.min(i, docs.length), 0, x); renderTabs(); }
}
/** The image of the structure viewer (Export Image…, Copy Image); a Volume document's: its own viewer, the color
 *  bar of a surface colored by a potential drawn in (volume.js drawBar), none of the structure's options. */
function exportCtx() {
  if (volumeActive()) {
    const vv = M().state.viewers["viewer-volume"];
    return { viewer: vv, el: document.getElementById("viewer-volume"), scene: null, bg: () => M().viewerBg(), triadEl: $("#axis-triad-vol"),
      stem: stemOf(active), view: "volume", volume: true, overlay: (g, W, Hh, k) => volumeView.drawBar(g, W, Hh, k),
      viewState: { projection: "orthographic" } };
  }
  const v = M().state.viewers["viewer-main"];
  return { viewer: v, scene: E ? E.scene() : null, bg: () => M().viewerBg(), triadEl: $("#axis-triad"),
    stem: stemOf(active), view: viewName(),
    // the SVG projector (js/lazy/export_svg.js): the camera's calibrated projection and what is drawn
    camera: E ? E.viewport.camera() : null, M: M(), setting: (k) => viewSetting(k), range: viewOpts().box,
    // the element legend in a corner (1.1.6): the drawn structure's elements, made by /api/legend
    symbols: () => { const s = structureOf(active); return s ? [...s.symbols] : []; },
    legend: (symbols, o) => lazy("legend_dialog").then((m) => m.forImage(H, symbols, o)),
    get viewState() { return window.__mace.viewState(); } };
}
/** Export Image on a Compare document: the comparison as SVG, both panes when side by side (plan T12). */
async function exportCompareSvg() {
  const data = M().compareSvgItems();
  if (!data) return false;
  const X = await import("./lazy/export_svg.js");
  const groups = [];
  let dx = 0, height = 0;
  for (const p of data.panes) {
    // the canvas's own px (modelToScreen's), not its zoomed size on screen
    const cv = p.v.getCanvas(), r = cv.getBoundingClientRect(), rect = { left: r.left, top: r.top, width: cv.clientWidth, height: cv.clientHeight };
    groups.push({ ...X.viewerGroup(p.v, p.id, p.items, rect), dx });
    dx += rect.width; height = Math.max(height, rect.height);
  }
  const text = await X.svgDocument({ width: Math.round(dx), height: Math.round(height), background: data.background, groups, title: data.title });
  if (window.__maceExportHook) window.__maceExportHook(text, "svg");
  const name = `${(active.title || "comparison").replace(/[^A-Za-z0-9_.-]+/g, "_")}_${M().look()}.svg`;
  download(new Blob([text], { type: "image/svg+xml" }), name);
  return true;
}
let alongState = null;          // {label, view} after View Along: the export's name says the view
function viewName() {
  const v = M().state.viewers["viewer-main"];
  if (!alongState || !v) return "custom";
  const now = v.getView();
  return now.slice(4).every((x, k) => Math.abs(x - alongState.view[4 + k]) < 1e-4) ? alongState.label : "custom";
}
async function exportImage() {
  if (active && active.kind === "trajectory") return exportCompareSvg();
  if (!M().state.viewers[volumeActive() ? "viewer-volume" : "viewer-main"]) return;
  const m = await lazy("export_image");
  m.open(H, exportCtx());
}
/* Export Animation… (1.1.6, js/lazy/anim_export.js): the FREQ view's mode or an optimization's steps captured from
 * the view itself, or a trajectory file (MD, NEB, an output) drawn by the server as `mace animate` draws it. */
const ANIM_FILE = /\.(out|o|log|xyz|extxyz|traj)$/i;
function canAnimate() {
  if (!active) return false;
  if (freqActive() || (active.kind === "trajectory" && trajSteps() > 1)) return true;
  const p = active.path || active.srcPath || "";
  return ANIM_FILE.test(p);
}
async function exportAnimation() {
  if (!canAnimate()) { toast("Open a FREQ output's modes, an optimization, or an MD or NEB trajectory", "info"); return; }
  const m = await lazy("anim_export");
  const d = active;
  m.open(H, { M: M(), apiTask: (p, b) => apiTask(p, b), doc: d, kind: d.kind, viewer: mainViewer(), stem: stemOf(d),
    path: d.path || d.srcPath || null, bg: () => M().viewerBg(), setting: (k) => viewSetting(k), download: (b, n) => download(b, n),
    freq: freqActive() ? freqView : null,
    traj: d.kind === "trajectory" && trajSteps() > 1 ? { steps: () => trajSteps(), step: (k) => trajStep(k, true), current: () => Number(($("#o-frame") || {}).value || 0) } : null });
}
async function copyImage() {
  if (!M().state.viewers[volumeActive() ? "viewer-volume" : "viewer-main"]) return false;
  const m = await lazy("export_image");
  return m.copy(H, exportCtx());
}

/* Drafts found at boot: "Restore unsaved changes to MgO.cif from 14:02?" */
function pendingDrafts() {
  const pre = `mace.gui.draft.${dirHash()}.`, out = [];
  try {
    for (let k = 0; k < localStorage.length; k++) {
      const key = localStorage.key(k);
      if (!key || !key.startsWith(pre)) continue;
      try { const d = JSON.parse(localStorage.getItem(key)); if (d && d.v === 1) out.push({ key, d }); } catch (e) { /* skip */ }   // ls-key: mace.gui.draft.
    }
  } catch (e) { return []; }
  return out.sort((a, b) => b.d.t - a.d.t);
}
function offerDrafts() {
  const b = $("#draft-banner");
  if (!b) return;
  const list = pendingDrafts().filter((x) => !docs.some((d) => edOf(d) && edOf(d).dirty && (d.path || "") === (x.d.path || "")));
  if (!list.length) { b.hidden = true; return; }
  const { key, d } = list[0], name = d.name || (d.path || "").split("/").pop() || "Untitled";
  const t = new Date(d.t).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
  b.innerHTML = `${icon("history")}<span>Restore unsaved changes to ${esc(name)} from ${t}?${list.length > 1 ? ` (${list.length - 1} more)` : ""}</span>
    <button class="tb-btn-text" data-v="restore">Restore</button><button class="tb-btn-text" data-v="discard">Discard</button>`;
  b.hidden = false;
  b.querySelector('[data-v="restore"]').onclick = () => { b.hidden = true; restoreDraft(key, d).finally(offerDrafts); };
  b.querySelector('[data-v="discard"]').onclick = () => { try { localStorage.removeItem(key); } catch (e) { /* private */ } offerDrafts(); };   // ls-key: mace.gui.draft.
}
async function restoreDraft(key, d) {
  await loadEditor();
  const { decode } = await import("./editor/drafts.js"), { safeName } = await import("./editor/queue.js");
  const s = decode(d);
  let r;
  try {
    r = await M().api("edit/from_structure", { body: { name: safeName(d.name), cell: s.cell, numbers: s.numbers.map((z) => z % 100 || z), frac: s.frac,
      locked: d.symMode === "sym", symprec: d.symprec || 1e-3, label: `Restored draft of ${d.name}`, meta: d.meta || undefined,
      ...(s.pbc ? { pbc: s.pbc, origin: s.origin, vacuum_vectors: s.vacuum_vectors } : {}) } });
  } catch (err) { toast(`Could not restore ${d.name}: ${err.message}`, "error"); return; }
  const open = d.path && docs.find((x) => x.path === d.path);
  if (open) closeDoc(open, true);
  const doc = await openUntitled(r, d.name);
  const e = edOf(doc);
  if (d.path) { doc.path = d.path; doc.untitled = false; e.path = d.path; doc.fileKind = "cif"; }
  e.draftId = d.path ? null : key.split(".").pop();
  e.diskStamp = d.diskStamp || null;
  e.savedRev = -1;                                   // a restored draft is modified until saved
  e.measurements = d.measurements || [];
  e.bondOverrides = new Map(d.bondOverrides || []);
  e.selectionSets = d.selectionSets || {};
  if (d.meta) { e.docMeta = d.meta; doc.meta = d.meta; }
  if (d.saveName) doc.saveName = d.saveName;
  E.drafts.write(e);
  E.redraw();
  docChanged(doc, e);
  afterSwitch();
}

/* Edit This Step (spec §2.6): the run's step on screen as a structure
 * document; its first edit opens the session with edit/open and the frame. */
async function editStep() {
  const d = active, t = M().state.traj;
  if (!d || d.kind !== "trajectory" || !t || !t.frames || !t.frames.length) return null;
  const el = $("#o-frame"), i = el && !$("#o-frame-wrap").classList.contains("hidden") ? Number(el.value) : t.frames.length - 1;
  const frame = t.frames[i] || await M().ensureStep(t, i);
  const same = docs.find((x) => x.path === d.path && x.frame === i);
  if (same) return activate(same);
  await loadEditor();
  saveView(active);
  const payload = { ...frame, path: d.path };
  const doc = { id: nextId++, path: d.path, frame: i, kind: "structure", fileKind: "out",
    title: `${d.title.replace(/\.[^.]+$/, "")} · step ${i + 1}`, data: payload };
  docs.push(doc);
  E.attach(doc, payload);
  active = doc;
  panelOwner.structure = doc.id;
  M().state.active = null;
  M().renderFiles();
  await M().loadStructure(null, payload);
  M().showTab("structure");
  showBottom("atoms", false);
  afterSwitch();
  return doc;
}

/* ================================================================ the editor's documents */
/* An edit changed a document (its structure, its history, its requests):
 * the tab's dot, the title, the status bar, the inspector, the panels. */
let docFrame = 0, docActiveChanged = false;
function docChanged(d) {
  // coalesced to one frame; a change to the active document is never lost to a
  // later call for another one in the same frame
  if (d === active) docActiveChanged = true;
  cancelAnimationFrame(docFrame);
  docFrame = requestAnimationFrame(() => {
    renderTabs(); updateTitle(); renderStatus(); renderToolbarState(); renderOpenEditors(); toolChip();
    const fire = docActiveChanged;
    docActiveChanged = false;
    if (fire && active) document.dispatchEvent(new CustomEvent("mace-doc", { detail: active }));
  });
}
/* A structure that exists only in this page (a new cell, a restored draft,
 * a trajectory step): an untitled structure document. */
async function openUntitled(payload, title = null, extra = {}) {
  await loadEditor();
  saveView(active);
  const d = { id: nextId++, path: null, kind: "structure", fileKind: "cif", title: title || payload.name || "Untitled",
    data: payload, untitled: true, ...extra };
  docs.push(d);
  E.attach(d, payload);
  active = d;
  panelOwner.structure = d.id;
  M().state.active = null;
  M().renderFiles();
  await M().loadStructure(null, payload);
  M().showTab("structure");
  showBottom("atoms", false);
  afterSwitch();
  return d;
}
/* The server restarted (new token) or stopped: requests get 403. Edits stay
 * in this page and in a draft; nothing is retried (spec §5.5, plan R4). */
let lostShown = false;
function connectionLost() {
  for (const d of docs) { const e = edOf(d); if (e) { e.lost = true; if (e.queue) e.queue.stop(); } }
  for (const t of [...taskStore.values()]) if (!FINAL.has(t.state)) taskEvent({ ...t, state: "lost", message: "Lost when the server stopped" });
  if (window.__maceDrafts) window.__maceDrafts.flush();
  if (lostShown) return;
  lostShown = true;
  connection = "bad";
  const b = $("#conn-banner");
  b.innerHTML = `${icon("wifi-off")}<span>Lost the connection to the MACE server (it restarted or stopped). Your edits are kept in this tab and saved as a draft. Reopen the URL the server printed; the draft will be offered there.</span>`;
  b.classList.remove("hidden");
  b.dataset.lost = "1";
  refresh();
}
/* P1 -> Sym on a structure whose group dropped: Symmetrize first? (spec §2.7) */
function toSym(e) {
  const y = e.structure.symmetry || {}, ref = e.referenceGroup;
  if (!(y.available && ref && (y.n_operations_conventional || y.n_operations) < ref.n_operations)) return E.api.setMode(e, "sym");
  return new Promise((resolve) => {
    let choice = null;
    const d = dialog({ id: "sym-first", title: "Symmetrize first?", narrow: true,
      body: `<p>This structure is now ${esc(y.international)} (${y.number}), lower than ${esc(ref.international)} (${ref.number}).</p>
        <p class="insp-hint">Symmetrize refines it back with spglib (one undo step) before Sym mode locks its operations. Lock as is keeps ${esc(y.international)}.</p>`,
      foot: `<button class="tb-btn-text" data-r="cancel">Cancel</button><button class="tb-btn-text" data-r="lock">Lock as is</button>` +
        `<button class="tb-btn-text primary" data-r="sym">Symmetrize</button>`,
      onClose: () => {
        // Symmetrize opens its dialog (sweep and preview); its Accept then locks Sym mode
        if (choice === "sym") lazy("symmetrize_dialog").then((m) => m.open(H, { afterAccept: () => E.api.setMode(e, "sym") })).then(() => resolve(null));
        else if (choice === "lock") E.api.setMode(e, "sym").then(resolve);
        else resolve(null);
      } });
    d.querySelectorAll("[data-r]").forEach((b) => b.addEventListener("click", () => { choice = b.dataset.r; d.close(); }));
  });
}

/* A Sym move that grows an orbit: keep the site's symmetry, allow it, or cancel. */
function confirmGrow(e, data) {
  return new Promise((resolve) => {
    const group = e.referenceGroup ? e.referenceGroup.international : "the space group";
    const d = dialog({ id: "grow-confirm", title: "Leave the special position?", narrow: true,
      body: `<p>${esc(data.message || "This move takes the atom off its special position")}.</p>
        <p class="insp-hint">Keep ${esc(group)} moves it only along the directions its site allows.</p>`,
      foot: `<button class="tb-btn-text" data-r="cancel">Cancel</button><button class="tb-btn-text" data-r="allow">Allow</button>` +
        `<button class="tb-btn-text primary" data-r="keep">Keep ${esc(group)}</button>`,
      onClose: () => resolve(choice === "keep" ? { project: true } : choice === "allow" ? { confirm: true } : null) });
    let choice = null;
    d.querySelectorAll("[data-r]").forEach((b) => b.addEventListener("click", () => { choice = b.dataset.r; d.close(); }));
  });
}

/* ================================================================ a run's frame, read-only (PLAN_analysis C5)
 * The Atoms, Coordination and Measurements panels of a run (.out) show the frame
 * on screen, read-only, under a banner ("Frame 3 of 4 · POINT 4"); editing goes
 * through Edit This Step. Measurements on a run are kept per run document, as
 * {type, atoms: [{i, image}]}, recomputed on the frame shown, and each can be
 * tracked along the run (Run Analysis › Tracked). */
let G = null;
class RunFrame {
  constructor(d) {
    this.doc = d; this.readOnly = true; this.selection = new Set(); this.measurements = [];
    this.coordMode = settings.get("coordMode") === "cart" ? "cart" : "frac";
    this.structure = null; this.frame = 0; this.nFrames = 1; this.final = false;
  }
  get n() { return this.structure ? this.structure.numbers.length : 0; }
  editBlock() { return "A run's frame is read-only. Use Edit This Step to change it"; }
  get label() {
    if (this.nFrames <= 1) return "The run's geometry";
    return this.final ? `Final geometry · frame ${this.frame} of ${this.nFrames - 1}` : `Frame ${this.frame} of ${this.nFrames - 1} · POINT ${this.frame + 1}`;
  }
}
const runFrames = new WeakMap();
const isRunDoc = (d) => !!d && !edOf(d) && (d.kind === "trajectory" || d.kind === "structure") && !!d.path && /\.(out|log)$/i.test(d.path);
/** The run document's frame on screen (null while it loads, or for any other document). */
function runFrameOf(d) {
  if (!isRunDoc(d) || !M()) return null;
  let r = runFrames.get(d);
  if (!r) { r = new RunFrame(d); runFrames.set(d, r); }
  const t = d.kind === "trajectory" ? M().state.traj : null;
  let s = null;
  if (t && t.frames && t.path === d.path) {
    const k = Math.min(t.frames.length - 1, Math.max(0, Number(M().state.frame ?? t.frames.length - 1)));
    r.frame = k; r.nFrames = t.frames.length;
    r.final = (t.labels || [])[k] === "final" || ((t.labels || [])[k] === undefined && k === t.frames.length - 1);
    s = t.frames[k];
    if (!s) { M().ensureStep(t, k).then(() => panelsSoon()).catch(() => {}); return null; }
  } else if (d.kind === "structure") { s = M().state.structure; r.frame = 0; r.nFrames = 1; }
  if (!s || !s.numbers) return null;
  if (r.structure !== s) {
    r.structure = s;
    r.selection = new Set([...r.selection].filter((i) => i < s.numbers.length));
    if (G) r.measurements.forEach((m) => { m.value = m.atoms.every((a) => a.i < s.numbers.length) ? G.measure(s.cell, s.frac, m.atoms) : NaN; });
  }
  return r;
}
let panelsTimer = 0;
/** One re-render 150 ms after the last frame change (a scrub does not redraw every table on the way). */
function panelsSoon() {
  clearTimeout(panelsTimer);
  panelsTimer = setTimeout(() => { renderPanels(); if (layout.bottomTab === "coordination") renderCoordination(); }, 150);
}
document.addEventListener("mace-frame-changed", panelsSoon);
document.addEventListener("mace-selection", () => {
  const rf = !edOf(active) && runFrameOf(active), h = $(".bottom-pane.measurements");
  if (rf && h) renderRunMeasurements(h, rf);
});
/** The banner over a run's Atoms table: which frame, read-only, Edit This Step, Frac/Cart. */
function runBanner(host, r) {
  let b = $(".run-ro", host);
  if (!r) { if (b) b.remove(); return; }
  if (!b) {
    host.insertAdjacentHTML("afterbegin", `<div class="run-ro" role="note"><span class="run-ro-t"></span>
      <span class="seg run-ro-coord" role="radiogroup" aria-label="Coordinates"><button role="radio" data-cm="frac">Frac</button><button role="radio" data-cm="cart">Cart</button></span>
      <button class="tb-btn-text run-ro-edit" type="button">${icon("move")}<span>Edit This Step</span></button></div>`);
    b = $(".run-ro", host);
    $(".run-ro-edit", b).addEventListener("click", () => run("traj.editStep"));
    $(".run-ro-coord", b).addEventListener("click", (ev) => {
      const x = ev.target.closest("[data-cm]"), rf = runFrameOf(active);
      if (!x || !rf) return;
      rf.coordMode = x.dataset.cm;
      renderPanels();
    });
  }
  $(".run-ro-t", b).textContent = `${r.label} (read-only). Use Edit This Step to change it.`;
  $$(".run-ro-coord [data-cm]", b).forEach((x) => x.setAttribute("aria-checked", String(x.dataset.cm === r.coordMode)));
}
/** A run's Measurements: rows recomputed on the frame shown, Measure Selected (2-4 atoms
 * picked in the Atoms table), atom numbers typed in, Track Along the Run. */
function renderRunMeasurements(host, r) {
  const sel = [...r.selection].sort((a, b) => a - b);
  const s = r.structure, fmt = (m) => (!Number.isFinite(m.value) ? "—" : m.atoms.length === 2 ? `${m.value.toFixed(3)} Å` : `${m.value.toFixed(2)}°`);
  const text = (m) => m.atoms.map((a, k) => `${s.symbols[a.i] || "?"}${a.i + 1}${k && (a.image || []).some((x, j) => x !== (m.atoms[0].image || [0, 0, 0])[j]) ? "′" : ""}`).join("–");
  host.innerHTML = `<div class="ms-head run-ms-head"><span class="muted small">${esc(r.label)} · read-only</span>
      <button class="tb-btn-text run-ms-sel" type="button"${sel.length >= 2 && sel.length <= 4 ? "" : " disabled"} title="Pick 2 to 4 atoms in the Atoms table (Shift or Ctrl+click)">Measure Selected${sel.length ? ` (${sel.length})` : ""}</button>
      <label class="small run-ms-add-l">Atoms <input class="field run-ms-add" placeholder="1 2" aria-label="Atom numbers to measure (2 to 4)" size="9"></label>
      <button class="tb-btn-text run-ms-go" type="button">Measure</button>
      ${r.measurements.length ? `<button class="tb-btn-text run-ms-copy" type="button">Copy as TSV</button>` : ""}</div>
    ${r.measurements.length ? `<table class="data ms-table run-ms-table"><thead><tr><th>Type</th><th>Atoms</th><th>On</th><th class="num">Value</th><th></th><th></th></tr></thead><tbody>${r.measurements.map((m, n) =>
      `<tr data-n="${n}"><td>${esc(m.type)}</td><td>${esc(text(m))}</td><td>frame ${r.frame}</td><td class="num">${esc(fmt(m))}</td>` +
      `<td><button class="tb-btn-text run-ms-track" type="button">${icon("chart-line")}<span>Track Along the Run</span></button></td>` +
      `<td><button class="icon-btn sm run-ms-del" aria-label="Delete this measurement">${icon("trash-2")}</button></td></tr>`).join("")}</tbody></table>`
      : `<p class="empty-note">No measurements on this run. Pick 2 to 4 atoms in the Atoms table and click Measure Selected, or type their numbers.</p>`}`;
  const add = async (idx) => {
    if (!G) G = await import("./geom/core.js");
    const bad = idx.find((i) => !(i >= 0 && i < r.n));
    if (idx.length < 2 || idx.length > 4 || bad !== undefined) { toast(`Give 2 to 4 atom numbers from 1 to ${r.n}`, "error"); return; }
    const atoms = nearestChain(r.structure, idx);
    const m = { type: { 2: "Distance", 3: "Angle", 4: "Dihedral" }[atoms.length], atoms, frame: r.frame };
    m.value = G.measure(r.structure.cell, r.structure.frac, atoms);
    r.measurements.push(m);
    renderRunMeasurements(host, r);
  };
  $(".run-ms-sel", host).addEventListener("click", () => add(sel));
  const go = () => add($(".run-ms-add", host).value.split(/[\s,;–-]+/).filter(Boolean).map((x) => Number(x) - 1));
  $(".run-ms-go", host).addEventListener("click", go);
  $(".run-ms-add", host).addEventListener("keydown", (ev) => { if (ev.key === "Enter") { ev.preventDefault(); go(); } });
  const copy = $(".run-ms-copy", host);
  if (copy) copy.addEventListener("click", () => copyText(["type\tatoms\tframe\tvalue"].concat(r.measurements.map((m) => `${m.type}\t${text(m)}\t${r.frame}\t${Number.isFinite(m.value) ? m.value : ""}`)).join("\n"), "Copied the measurements (TSV)"));
  $$("tr[data-n]", host).forEach((tr) => {
    const m = r.measurements[Number(tr.dataset.n)];
    $(".run-ms-del", tr).addEventListener("click", () => { r.measurements.splice(r.measurements.indexOf(m), 1); renderRunMeasurements(host, r); });
    $(".run-ms-track", tr).addEventListener("click", () => trackMeasurement(
      { type: m.type.toLowerCase(), atoms: m.atoms.map((a) => ({ base: a.i, shift: (a.image || [0, 0, 0]).slice() })) }, `${m.type} ${text(m)}`));
  });
}
/** Each next atom's periodic image nearest the one before (a chain, as the Measure tool builds). */
function nearestChain(s, idx) {
  const out = [{ i: idx[0], image: [0, 0, 0] }];
  const C = s.cell, pbc = s.pbc || [true, true, true];
  const cart = (i, im) => [0, 1, 2].map((k) => (s.frac[i][0] + im[0]) * C[0][k] + (s.frac[i][1] + im[1]) * C[1][k] + (s.frac[i][2] + im[2]) * C[2][k]);
  for (let n = 1; n < idx.length; n++) {
    const prev = out[n - 1], p = cart(prev.i, prev.image);
    let best = null, bd = Infinity;
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) {
      const base = prev.image.map((x, k) => x + [a, b, c][k] * (pbc[k] ? 1 : 0));
      const q = cart(idx[n], base), dd = Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]);
      if (dd < bd - 1e-9) { bd = dd; best = base; }
    }
    out.push({ i: idx[n], image: best });
  }
  return out;
}

/* The bottom panel's editor tabs (Measurements; History arrives with it). */
let measurementsShown = true;
function renderPanels() {
  const e = edOf(active), host = $(".bottom-pane.measurements");
  const rf = e ? null : runFrameOf(active);
  const lent = e || rf ? null : PC.borrowed(active);      // the structure of the run a plot or volume came from
  PC.apply(active);                                         // a panel that applies is shown before it is drawn into
  if ((rf || lent) && !E) { loadEditor().then(() => renderPanels()).catch(() => {}); }
  if (host && rf) renderRunMeasurements(host, rf);
  else if (host && E && E.api.measure) E.api.measure.panel(host, e);
  else if (host && !e) host.innerHTML = `<p class="empty-note">Open a structure to measure it.</p>`;
  // Atoms: the editor's virtualized table for a structure document and, read-only, for a run's frame on screen
  const atoms = $(".bottom-pane.atoms");
  if (atoms) {
    let t = $(".at-host", atoms);
    if (!t) { atoms.insertAdjacentHTML("afterbegin", `<div class="at-host"></div>`); t = $(".at-host", atoms); }
    const view = e || rf || lent;
    const mine = !!(view && E);
    if (lent) atoms.classList.remove("no-structure");
    atoms.classList.toggle("editor-table", mine);
    atoms.classList.toggle("run-table", mine && !!(rf || lent));
    runBanner(atoms, mine && rf ? rf : null);
    PC.banner(atoms, mine && lent ? lent : null);
    if (E) E.atomsTable.update(t, mine ? view : null);
  }
  const hist = $(".bottom-pane.history");
  if (hist) renderHistoryPane(hist, e);
  PC.apply(active);
}
/* The History tab: the structure's undo stack, and with a "Settings" chip the
 * document's Calculation Setup history (its own, plan B4). */
let histMode = "structure";
function renderHistoryPane(hist, e) {
  let chips = $(".hist-chips", hist), body = $(".hist-host", hist);
  if (!body) {
    hist.innerHTML = `<div class="hist-chips" role="radiogroup" aria-label="History of"></div><div class="hist-host"></div>`;
    chips = $(".hist-chips", hist); body = $(".hist-host", hist);
  }
  if (!chips.dataset.bound) {
    chips.dataset.bound = "1";
    chips.addEventListener("click", (ev) => { const b = ev.target.closest("[data-hist]"); if (b) { histMode = b.dataset.hist; renderPanels(); } });
    body.addEventListener("click", (ev) => { const r = ev.target.closest("[data-setup-n]"); if (r && setupMod) setupMod.jump(Number(r.dataset.setupN)); });
  }
  const subj = active && active.setupFor ? docs.find((x) => x.id === active.setupFor) : active;
  const rows = setupMod && subj ? setupMod.historyRows(subj) : [];
  chips.hidden = rows.length < 2;
  if (rows.length < 2 && histMode === "settings") histMode = "structure";
  chips.innerHTML = `<button role="radio" aria-checked="${histMode === "structure"}" data-hist="structure">Structure</button>` +
    `<button role="radio" aria-checked="${histMode === "settings"}" data-hist="settings">Settings <span class="muted">${rows.length - 1}</span></button>`;
  if (histMode === "settings") {
    const t = (x) => new Date(x).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
    body.innerHTML = `<div class="hist-scroll" role="listbox" aria-label="Settings history">${rows.map((r, k) =>
      `<div class="hist-row${r.cur ? " cur" : ""}" role="option" aria-selected="${r.cur}" data-setup-n="${k}"><span class="hist-l">${esc(r.label)}</span><span class="hist-t">${t(r.t)}</span><span class="hist-n"></span></div>`).join("")}</div>`;
    return;
  }
  if ($(".hist-scroll[aria-label='Settings history']", body)) body.innerHTML = "";
  if (E) E.history.update(body, e);
  else if (!body.childElementCount) body.innerHTML = `<p class="empty-note">Undo and redo apply to the structure editor: ${esc(shortcutOf("edit.undo"))} and ${esc(shortcutOf("edit.redo"))}.</p>`;
}
document.addEventListener("mace-doc", renderPanels);
// Setup's geometry check follows the document: an edit, or a run's step moved
document.addEventListener("mace-doc", () => { if (setupOpen && setupMod) setupMod.docChanged(); });
$("#o-frame") && $("#o-frame").addEventListener("input", () => { if (setupOpen && setupMod) setupMod.docChanged(); });
document.addEventListener("mace-selection", () => { if (E) E.atomsTable.selectionChanged(); });

/* The document's named selections: Select › Selection Sets, and its popover. */
function setItems(e) {
  const sets = e ? Object.entries(e.selectionSets) : [];
  const items = sets.map(([name, ix]) => ({ label: `${name} (${ix.length})`, icon: "tag", run: () => E.select(e, ix, "replace") }));
  return [...(items.length ? items : [{ label: "No selection sets", disabled: true, run: () => {} }]), "-", "select.saveSet"];
}

/* ================================================================ Calculation Setup's view of the shell */
const SETUP_API = {
  /** The served folder's command-line-only values (mace-folder.toml: setup.opt2d12_config, ...). */
  folderCli: () => (settings.folderInfo().cli || {}),
  api: (p, o) => M().api(p, o), dirHash: () => dirHash(), activeDoc: () => active, docById: (id) => docs.find((d) => d.id === id) || null,
  isDirty: (d) => isDirty(d), editRev: (d) => (edOf(d) ? edOf(d).rev : 0), structureOf: (d) => structureOf(d),
  /** A run's step on screen: {frame, n} for the active trajectory document, else null. */
  frameInfo: (d) => {
    const t = M().state.traj;
    if (!d || d.kind !== "trajectory" || panelOwner.opt !== d.id || !t || !t.frames) return null;
    return { frame: M().state.frame, n: t.frames.length };
  },
  /** An edited document's structure as the server has it: its session, once the edits in flight have landed. */
  sourceOf: async (d) => {
    const e = edOf(d);
    if (e && isDirty(d)) {
      await waitIdle(e);
      if (!e.srv.session) { await E.editing.ensureSession(e); await waitIdle(e); }
    }
    return sourceOf(d);
  },
  apiTask: (p, b, u) => apiTask(p, b, u), setupDeck: (d, r, show) => setupDeck(d, r, show),
  clearProblems: (id, src) => clearProblems(id, src), problem: (p) => addProblem(p), toast: (m, k, a) => toast(m, k, a),
  setting: (k) => settings.get(k), settingSource: (k) => settings.source(k), set: (k, v) => settings.set(k, v), shortcut: (id) => shortcutOf(id), run: (id, a) => run(id, a),
  cancelTask: (id) => cancelTask(id), reveal: (p) => reveal(p), focusRegion: (r) => focusRegion(r), copyText: (t, m) => copyText(t, m),
  refreshFiles: () => M().loadFiles().catch(() => {}),
  // Setup's 4-7 and Phonopy (setup/modules.js): the deck they write, and the Submit dialog on a set
  openPath: (p) => openDoc(p), builder: (name, fn, ...args) => S.builder(name, fn, ...args),
  showText: (title, text) => openTextDoc(title, text), showProblems: () => showBottom("problems"),
  historyChanged: () => renderPanels(),
  cifCount: (path) => { const dir = path && path.includes("/") ? path.slice(0, path.lastIndexOf("/") + 1) : "";
    return M().state.files.filter((f) => f.kind === "cif" && f.path.startsWith(dir) && !f.path.slice(dir.length).includes("/")).length; },
  /** "Use the unedited parent": the run's last step on screen (the deck's geometry); edits stay in the document. */
  showParentGeometry: (d) => { if (d.kind === "trajectory") trajStep(Infinity); else toast("The deck uses the final geometry saved in the .out; your edits stay in this document", "info"); },
  /** "Save As CIF and set up a new calculation": Save As (CIF), then Setup on the saved file. */
  saveAsCifForSetup: async (d) => {
    let doc = d;
    if (d.kind === "trajectory") doc = await editStep();
    if (!doc) return;
    const stem = (doc.title || "structure").replace(/ · step (\d+)$/, "_step$1").replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9_.-]+/g, "_");
    const done = (ev) => { document.removeEventListener("mace-saved", done); if (ev.detail && ev.detail.path && setupMod) setTimeout(() => setupMod.refresh(), 50); };
    document.addEventListener("mace-saved", done);
    run("file.saveAs", { preset: { name: `${stem}.cif`, format: "cif", cifMode: "p1" } });
  },
  confirm: ({ title, body, yes, no, defaultNo, pre }) => new Promise((resolve) => {
    let ok = false;
    const dl = dialog({ id: "setup-confirm", title, narrow: !pre, body: `<p>${esc(body)}</p>${pre ? `<pre class="confirm-pre">${esc(pre)}</pre>` : ""}`,
      foot: `<button class="tb-btn-text${defaultNo ? " primary" : ""}" data-r="no">${esc(no || "Cancel")}</button><button class="tb-btn-text${defaultNo ? "" : " primary"}" data-r="yes">${esc(yes || "OK")}</button>`,
      onClose: () => resolve(ok) });
    dl.querySelectorAll("[data-r]").forEach((b) => b.addEventListener("click", () => { ok = b.dataset.r === "yes"; dl.close(); }));
    const def = dl.querySelector(`[data-r="${defaultNo ? "no" : "yes"}"]`);
    if (def) def.focus();
  }),
  setupSchema: () => setupSchema(), runFacts: (d) => runFacts(d),
};

/* The calculation types of Setup and of Calculation › Regular / Properties Calculations (1.1.6):
 * grouped, numbered and gated by one table, setup/schema calc_groups (the workflow's prerequisites,
 * mace/workflow/common/constants.py SETUP_CALC_GROUPS), read through setup/model.js calcBlocked */
let setupSchemaP = null, calcGroups = null;
function setupSchema() {
  if (!setupSchemaP) setupSchemaP = M().api("setup/schema").then((r) => { calcGroups = r.calc_groups || null; return r; },
    (e) => { setupSchemaP = null; throw e; });
  return setupSchemaP;
}
/** What Setup's gate needs to know of an output: a FREQ run, an optimization not (yet) converged. */
function runFacts(d) {
  const out = !!(d && /\.(out|log|o)$/i.test(d.path || ""));
  return { freq: isFreqDoc(d), optOpen: !!(out && d.data && d.data.is_opt && !/OPT END - CONVERGED/.test(d.data.opt_status || "")) };
}
const setupSubject = () => (active && active.setupFor ? docs.find((x) => x.id === active.setupFor) || null : active);
function calcBlockedHere(calc) {
  if (!calcGroups) { setupSchema().catch(() => {}); return "Loading the calculation types…"; }
  const d = setupSubject();
  return SM.calcBlocked(calcGroups, calc, d, { dirty: isDirty(d), frame: SETUP_API.frameInfo(d), ...runFacts(d) });
}
/** A Calculation menu group's entries, numbered as its CLI menu (Phonopy: its own command, no number). */
function calcMenu(key) {
  if (!calcGroups) { setupSchema().catch(() => {}); return [{ label: "Loading…", disabled: true, run: () => {} }]; }
  const g = calcGroups.groups.find((x) => x.key === key);
  return g ? g.types.map((t) => ({ cmd: "calc.setupType", args: { calc: t.calc }, label: `${t.n ? `${t.n} ` : ""}${t.label}` })) : [];
}

/* ================================================================ the shell API the commands use */
const S = {
  // two editor groups (groups.js)
  groups: () => GR,
  otherStructurePath: () => { const o = GR && GR.otherStructure(); return o && o.path && o.path !== docPath(active) ? o.path : null; },
  compareGroups: () => { const p = S.otherStructurePath(); return p ? compareWith(p) : toast("The other group shows no structure", "info"); },
  palette: (p) => openPalette(p), activeDoc: () => active, closeDoc, closeAll, cycleDoc, clearRecent,
  pinRecent, removeRecent, isRecentPinned: (p) => recentEntries().some((x) => x.path === p && x.pinned),
  settingsDialog, cheatSheet, aboutDialog, docsDialog, tunnelDialog, clusterDocs: () => docsDialog("the-cluster-view"), dirty: () => docs.some(isDirty),
  openFolder: () => openFolderDialog(), openFolderOn: () => !!(info && info.open_folder),
  signOut: () => connectUi().then((m) => m.signOut(H)),
  copyDiagnostics: () => copyDiagnostics(), checkForUpdates: () => updateDialog({ check: true }), reportProblem: () => connectUi().then((m) => m.reportProblem(H)),
  undo: () => (edOf(active) ? E.api.undo(edOf(active)) : Promise.resolve()),
  redo: () => (edOf(active) ? E.api.redo(edOf(active)) : Promise.resolve()),
  showBottom, viewAlong, resetView, toggleProjection, setting: (k) => viewSetting(k), set: (k, v) => settings.set(k, v),
  toggleSetting: (k) => settings.set(k, !settings.get(k)), optChecked, toggleOpt, labelMode, setLabels, cycleLabels,
  /* a per-document display value (Shift+P, Shift+A): the drawn document's override, never the default */
  displayValue: (k) => viewSetting(k), toggleDisplay: (k) => {
    const d = viewDoc();
    if (!d) return;
    // a style value toggled (Shift+P: polyhedra) makes the document's style Custom, as an edit in the Inspector does
    const style = DS.GROUPS.style.includes(k) && viewSetting("stylePreset") !== "custom";
    DS.set(d, style ? { [k]: !viewSetting(k), stylePreset: "custom" } : { [k]: !viewSetting(k) });
  },
  displayPanel: (group) => openDisplayGroup(group),
  range, setRange, rangeText: () => DS.rangeLabel(viewDoc()),
  // 1.1.6: Display Range › N × M × L… (js/cells_picker.js) and whether the range is whole cells from 0
  rangeIsCells: () => { const r = DS.rangeOf(viewDoc()); return r.lo.every((x) => x === 0) && r.hi.every((x) => Number.isInteger(x)); },
  cellsDialog: () => import("./cells_picker.js").then((m) => m.openDialog({ ...H, cells: () => rangeCells(), pbc: () => viewPbc(),
    repeatNonPeriodic: () => !!viewDoc() && viewDoc().kind === "structure", setCells: (c) => setRange(c),
    build: viewDoc() && viewDoc().kind === "structure" && edOf(viewDoc()) ? () => run("build.supercell") : null })),
  inspectorTier: () => IT.tierOf(active && active.kind, active),
  setInspectorTier: (t) => { if (!panelShown("inspector")) togglePanel("inspector"); IT.setTier(active && active.kind, t, active); renderInspector(); },
  showInspector: () => { if (!panelShown("inspector")) togglePanel("inspector"); if (setupOpen) toggleSetup(false); },
  activityShown: () => AP.shown(), activityItemShown: (id) => !AP.isHidden(id),
  activityToggleItem: (id) => AP.toggleItem(id), activityMove: (id, dir) => AP.move(id, dir, ACTIVITY_IDS), activityReset: () => AP.reset(),
  style, setStyle, panelShown, togglePanel, toggleAllPanels, allPanelsHidden: () => !!layout.focus, cycleTheme, zoomUi, focusRegion, setAllSections,
  /** Can Expand All / Collapse All do anything? Two or more Inspector sections, or Calculation Setup open. */
  manySections: () => setupOpen || $$("#inspector .insp-section").length >= 2,
  previewTheme, pickTheme, elementColors, legendActive, legendOpenDocs, legendExplorer, explorerStructures, structureFiles, analyzeStructures,
  openStructureCount: () => { const o = openStructures(); return o.paths.length + o.symbols.length; },
  openDb: () => {
    const want = settings.get("defaultDb"), fs = M().state.files;
    const f = (want && fs.find((x) => x.path === want && x.kind === "db")) || fs.find((x) => x.name === "materials.db" && !x.path.includes("/"))
      || fs.find((x) => x.name === "materials.db") || fs.find((x) => x.kind === "db");
    return f && openDoc(f.path);
  },
  exportTable: (fmt) => exportTable(fmt),
  openCluster: (tab) => openCluster(tab), openJob: (id) => openJob(id), openWorkflows: (key) => openWorkflows(key),
  newWorkflow: (a) => openWorkflowBuilder(a),
  jobDetailsPrompt: async () => {
    const id = window.prompt("Job id (N or N_T):", "");
    if (id) openJob(id);
  },
  /** Plot Band Structure / DOS: the active output's own partner file first, then the folder's first. */
  openFirst: (k) => {
    const fs = M().state.files.filter((x) => x.kind === k);
    const a = active && active.path;
    let f = null;
    if (a) {
      const dir = a.includes("/") ? a.slice(0, a.lastIndexOf("/") + 1) : "";
      const base = a.slice(dir.length).replace(/\.[^.]+$/, "").replace(/_(opt|sp|OPT|SP)$/, "");
      f = fs.find((x) => x.path.startsWith(dir + base) && !x.path.slice(dir.length).includes("/"));
    }
    f = f || fs[0];
    return f && openDoc(f.path);
  },
  resetPlot: () => plotView && plotView.reset(),
  openModes: (p) => openModes(p || (active && active.path)),
  /** IR / Raman: the active output's own spectrum first, then the folder's first. */
  colorByPotential: () => colorByPotential(), volumeDifference: () => volumeDifference(),
  /** The outputs picked in the Explorer (Ctrl/Shift-click), with the one right-clicked: CRYSTAL outputs only. */
  convergencePaths: (p) => {
    const picked = new Set(M().state.selected || []);
    if (p) picked.add(p);
    return [...picked].filter((x) => { const f = M().fileByPath(x); return f && (f.kind === "out" || f.kind === "log") && !M().isJobLog(f); });
  },
  compareConvergence: (paths) => openConvergence(paths),
  openGallery: (o) => openGallery(o || {}),
  openSpectrum: (kind) => {
    const p = partnerOf(active, kind);
    if (p) return openDoc(p.path);
    // a FREQ output without SPEC.DAT: broadened from its modes
    if (active && active.path && (active.kind === "freq" || isFreqDoc(active))) return openFreqSpectrum(active.path, kind);
    const f = M().state.files.find((x) => x.kind === kind);
    return f && openDoc(f.path);
  },
  plotFile: (p) => (M().fileByPath(p) && M().fileByPath(p).kind === "out" ? openModes(p) : openDoc(p)),
  canPlot: (p) => { const f = p && M().fileByPath(p); return !!f && (["band", "dos", "ir", "raman"].includes(f.kind) || (f.kind === "out" && M().isFreq(p))); },
  reopenActive: () => active && active.path && activate(active),
  tool: () => tool, setTool: (t) => {
    if (t !== tool && E && E.api.measure) E.api.measure.tool.reset();     // a new Measure starts fresh
    tool = t; toolChip(); refresh();
  },
  symMode: () => (edOf(active) ? edOf(active).symMode : symMode), setSymMode: (m) => { symMode = m; refresh(); },
  // for the viewport and the editor (js/editor/main.js)
  viewOpts, setCursor, isMac, reducedMotion, toolChip, statusChanged: () => renderStatus(),
  contextMenu: (atom, x, y) => openViewportMenu(atom, x, y),
  selectionChanged: () => { refresh(); document.dispatchEvent(new CustomEvent("mace-selection")); },
  tools: {},
  openEditor: () => edOf(active) && E.editing.ensureSession(edOf(active)),
  /* the editor's commands (js/editor/commands.js): run now when the module is here */
  ed: (id, args) => (E ? E.run(id, active, args) : loadEditor().then(() => E.run(id, active, args))),
  editReason: (c) => {
    const d = active, e = edOf(d);
    if (e) return e.editBlock() || "";
    if (d && d.kind === "trajectory") return "Trajectories are view-only. Use Edit This Step";
    if (d && d.kind === "freq") return FREQ_VIEW_ONLY;
    return "Needs a structure";
  },
  escape: () => {
    if (E && E.pointer() && E.pointer().dragging) { E.pointer().cancel(); return; }
    if (tool !== "select") { S.setTool("select"); return; }
    const e = edOf(active);
    if (e && e.selection.size) E.select(e, [], "replace");
  },
  frameSelection: () => E && E.frameSelection(),
  pivotOn: (what) => E && E.pivotOn(what),
  openMenuAt: (items, x, y, label) => openAt(items, x, y, label),
  toast: (m, kind, action) => toast(m, kind, action),
  docChanged: (d, e) => docChanged(d, e),
  connectionLost: () => connectionLost(),
  confirmGrow: (e, data) => confirmGrow(e, data),
  openUntitled: (p, title) => openUntitled(p, title),
  inspectorChanged: () => refreshInspector(),
  run: (id, args) => run(id, args),
  setLayer: (l) => commands.setLayer(l),
  measurementsShown: () => measurementsShown,
  toggleMeasurements: () => { measurementsShown = !measurementsShown; const e = edOf(active); if (e && E) E.api.redrawMeasurements(e); refresh(); },
  measurementsChanged: () => { renderPanels(); refresh(); },
  shortcut: (id) => shortcutOf(id), icon: (n) => icon(n),
  lazyDialog: (mod, fn) => lazy(mod).then((m) => m[fn](H)),
  openSets: () => {
    const e = edOf(active), r = $("#viewer-main").getBoundingClientRect();
    openAt(setItems(e), r.left + 24, r.top + 24, "Selection sets");
  },
  viewportFocused: () => ctx().viewportFocus,
  pickElement: (anchor, cb, opts) => lazy("ptable").then((m) => m.open(H, anchor, cb, opts))
    .catch(() => toast("The periodic table could not be loaded", "error")),
  toSym: (ed) => toSym(ed),
  focusMenuBar: () => focusBar(), contextMenuForFocus: () => (sidebar && sidebar.menuForFocus()) || openForFocus(), sidebar: () => sidebar,
  rotateView, rotateStep, snapNearestAxis, zoomView, trajStep, trajPlay, trajPlaying: () => !!playTimer, trajLayout,
  tabOf: (a) => (a && a.docId !== undefined ? docs.find((d) => d.id === a.docId) : active) || null,
  moveTab: (d, step) => { const r = moveDocBy(d, step); if (r) announce(`${d.title} moved ${step < 0 ? "left" : "right"}`); return r; },
  canMoveTab: (d, step) => !!d && docs.length > 1 && docs.indexOf(d) + step >= 0 && docs.indexOf(d) + step < docs.length,
  closeOthers: (d) => docs.filter((x) => x !== d).forEach((x) => closeDoc(x)),
  closeRight: (d) => docs.slice(docs.indexOf(d) + 1).forEach((x) => closeDoc(x)),
  hasRight: (d) => !!d && docs.indexOf(d) >= 0 && docs.indexOf(d) < docs.length - 1,
  selectedAtom: () => { const e = edOf(active); return e && e.selection.size === 1 ? [...e.selection][0] : null; },
  elementCount: () => { const s = structureOf(active); return s ? new Set(s.symbols).size : 0; },
  fileHistory: (p) => fileHistory(p), activePath: () => (active && active.path) || null,
  openRunPlot: (p, tab) => openRunPlot(p, tab), openMech: (p) => openMech(p),
  openStructure: (p) => openStructure(p), opensAsResults: (p) => { const d = p && M().state.details && M().state.details[p]; return !!(d && d.refined && d.role === "crystal_out" && MECH_CALCS.includes(d.calc)); }, runPathOf: (a) => runPathOf(a), compareRuns: () => compareRunsDialog(),
  trackMeasurement: (m, label) => trackMeasurement(m, label),
  /** The save toast's Undo: the version File History kept before this save goes back, then the document reloads. */
  undoSave: async (ed, path, entry, stamp) => {
    try {
      await M().api("history/restore", { body: { id: entry.id, path, expected: stamp } });
      toast(`Restored the previous ${path.split("/").pop()}`, "success");
      await M().loadFiles();
      const d = docs.find((x) => x.path === path);
      if (d && !isDirty(d)) await reloadDoc(d);
    } catch (e) { toast(e.message, "error"); }
  },
  copyText, reveal, pathOf, copyRootPath: () => (info ? copyText(info.root, "Folder path copied") : null), kindOf: (p) => (p && M().fileByPath(p) ? M().fileByPath(p).kind : ""),
  openPath: (p) => openDoc(p), fileExists: (p) => !!(M() && M().fileByPath(p)), isText: (p) => ["cif", "d12", "d3", "out", "json", "band", "dos", "ir", "raman"].includes(fileKind(p)),
  openAsText, canCompare, compareWith, compareInput, compareSub, showCoordination, setupFor, atomOf, centerOnAtom, copyAtom,
  /** Calculation Setup on an output at 4 ELASTIC ... 7 CPHF or Phonopy (setup/modules.js). */
  setupModule: async (calc, path) => {
    if (path) await setupFor(path); else if (!setupOpen) toggleSetup(true);
    return (await setupPanel()).pick(calc);
  },
  /** Calculation › Regular / Properties Calculations: Setup on the file in front at that type, if it can have it. */
  calcBlocked: (calc) => calcBlockedHere(calc), calcMenu,
  setupCalc: async (calc) => {
    const why = calcBlockedHere(calc);
    if (why) { toast(why, "info"); return false; }
    if (!setupOpen) toggleSetup(true);
    return (await setupPanel()).pick(calc);
  },
  compareStructures: () => { M().showTab("opt"); toast("Pick structure A and B under “Compare two structures”, then Compare.", "info"); },
  setupOpen: () => setupOpen, toggleSetup: (force) => { toggleSetup(force); if (setupOpen) setupPanel().then((m) => requestAnimationFrame(() => m.focus())); },
  previewDeck: async () => { if (!setupOpen) toggleSetup(true); return (await setupPanel()).preview(true); },
  saveJson: async () => { if (!setupOpen) toggleSetup(true); const m = await setupPanel(); const st = m.state(); if (st && !st.saveOpen && !st.saved) { st.saveOpen = true; m.refresh(); return null; } return m.save(); },
  copyCli: async () => { if (!setupOpen) toggleSetup(true); return (await setupPanel()).copyCli(); },
  resetSettings: async () => (await setupPanel()).resetAll(),
  loadExample: async (name) => {
    if (!setupOpen) toggleSetup(true);
    const t = (M().state.templates || []).find((x) => x.name === name);
    if (t) return (await setupPanel()).loadExample(t);
    return null;
  },
  exampleMenu: () => { const ts = M().state.templates || []; return ts.length ? ts.map((t) => ({ label: t.name, icon: "file-json", title: t.description || t.name, run: () => S.loadExample(t.name) }))
    : [{ label: "No example configs", disabled: true, run: () => {} }]; },
  setupUndo: async () => (await setupPanel()).undo(), setupRedo: async () => (await setupPanel()).redo(),
  preflight: async () => (await setupPanel()).preflight(),
  // saving and drafts (js/editor/save.js, drafts.js)
  dirHash, editorDocs, afterSave, confirmOverwrite, changedOnDisk, confirmReplace,
  saveAsDialog: (e, def, preset) => saveDialog().then((m) => m.open(H, e, def, preset)),
  activateEd: (e) => activate(e.doc),
  saveAll: () => (E ? E.saving.saveAll(editorDocs()) : null), revert: () => revert(active),
  exportImage, copyImage, exportAnimation: () => exportAnimation(), canAnimate: () => canAnimate(),
  newStructure: () => lazy("new_structure").then((m) => m.open(H, S)),
  editStep: () => editStep(),
  setPreviewKeys: (fn) => { previewKeys = fn; }, previewKeysOwner: () => previewKeys,
  activateDoc: (d) => activate(d), refresh: () => refresh(),
  importClipboard: async () => importClipboard(await readClipboard()), readClipboard, isAtomsBlock,
  pasteText: (t) => pasteText(t || ""),
  /** Sym-mode paste that makes many more atoms than were copied (decision D7). */
  confirmPasteOrbits: (n, total) => new Promise((resolve) => {
    let ok = false;
    const d = dialog({ id: "paste-confirm", title: "Paste Whole Orbits?", narrow: true,
      body: `<p>In Sym mode each pasted atom brings its orbit: ${n} copied atom${n === 1 ? "" : "s"} become ${total}.</p>`,
      foot: `<button class="tb-btn-text" data-r="cancel">Cancel</button><button class="tb-btn-text primary" data-r="ok">Paste ${total} Atoms</button>`,
      onClose: () => resolve(ok) });
    d.querySelectorAll("[data-r]").forEach((b) => b.addEventListener("click", () => { ok = b.dataset.r === "ok"; d.close(); }));
  }),
  openParsed: (name, blob, label, opts) => openParsed(name, blob, label, opts),
  builder: (name, fn, ...args) => lazy(name).then((m) => m[fn](H, ...args)),
  // Phase 4: tasks, problems
  awaitTask, apiTask, taskEvent, cancelTask, problem: (p) => addProblem(p), clearProblems, dbBuild,
  exportStructure, sourceOf: (d) => sourceOf(d || active), openHeld, exportName, checkDisk: () => { diskCheckAt = -Infinity; return checkDisk(); },
  clientId: () => clientId, tasks: () => taskStore,
  // Structure Analysis (lazy/analysis_store.js): a document by id, its edit revision, and framing a region of the 3D view
  docById: (id) => docs.find((d) => d.id === id) || null, editRev: (d) => (edOf(d) ? edOf(d).rev : 0),
  framePoints: (pts, margin) => { const c = E && E.cam && E.cam(); if (c && pts && pts.length) c.frame(pts, margin); },
  // Submit, Jobs and Batch Setup (PLAN_submit)
  submit: (a) => openSubmit({ targets: submitTargets(a), where: "slurm" }),
  runHere: (a) => openSubmit({ targets: submitTargets(a), where: "local" }),
  batch: (a) => openBatch(a && a.dir !== undefined ? a.dir : (a && a.path && a.path.includes("/") ? a.path.slice(0, a.path.lastIndexOf("/")) : ""),
    a && a.preset),
  submitTargets: (a) => submitTargets(a), jobsPanel: () => jobsPanel(), batchPanel: () => batchPanel(),
  refreshFiles: () => M().loadFiles().catch(() => {}),
};

/* ================================================================ boot */
async function boot() {
  settings.init(M().api);
  settings.setNotifier((m, kind) => toast(m, kind));
  M().setting = (k) => viewSetting(k);
  M().rangeN = () => range();          // the classic draw (app.js structOpts) before the scene loads
  M().docSpins = () => { const d = viewDoc(); if (!d || viewSetting("atomColor") !== "spin") return null; const sp = spinsOf(d); return sp.state === "ok" ? sp : null; };
  // the Structure card's "Not saved in the folder yet" for a dropped document (plan T9)
  M().untitledHint = (s) => (active && active.untitled && !active.path && active.dropped && active.kind === "structure" ? "Not saved in the folder yet" : "");
  // Inspector › Symmetry › Compute Now: spglib on a structure over 20,000 atoms, as a task (plan T4, R5)
  M().computeSymmetry = async () => {
    const d = active;
    if (!d) return;
    try {
      const r = await apiTask("symmetry/compute", { source: await sourceOf(d) });
      const s = structureOf(d);
      if (r && r.symmetry && s && active === d) {
        s.symmetry = r.symmetry;
        if (d.data && d.data !== s && d.data.symmetry) d.data.symmetry = r.symmetry;
        M().renderStructureInfo(s);
        refreshInspector();
        refresh();
      }
    } catch (e) { toast(`Space group: ${e.message}`, "error"); refreshInspector(); }
  };
  M().openDb = (path) => openDbView(path);
  // structures arrive packed (spec §8.4.5) and are decoded by the geometry module
  M().fetchStructure = (path) => takePrefetched(path) || Promise.all([M().api("structure?" + M().q({ path, packed: 1 })), import("./geom/core.js")])
    .then(([r, core]) => core.decodePacked(r));
  registerAll(S);
  GR = createGroups(groupHost);
  // what draws or picks in the 3D view says "Needs 3D (WebGL)" without it (polish V3)
  for (const id of NEEDS_3D) { const c = registry.get(id); if (c) c.needs3d = true; }
  M().hasCommand = (id) => registry.has(id);
  M().runCommand = (id) => run(id);
  registerMnemonics(isMac, openMenuIndex);
  applyKeymap(settings.get("keymap"));
  loadLayoutMap();
  // Open Recent is filled in when the File menu opens
  const recentEntry = MENUS[0].items.find((x) => x.label === "Open Recent");
  Object.defineProperty(recentEntry, "sub", { get: recentMenu });
  const calcMenu = MENUS.find((m) => m.label === "Calculation");
  const exEntry = calcMenu && calcMenu.items.find((x) => x && x.label === "Open Example Config");
  if (exEntry) Object.defineProperty(exEntry, "sub", { get: () => S.exampleMenu() });
  // Regular / Properties Calculations: filled from setup/schema calc_groups when the menu opens
  for (const it of (calcMenu ? calcMenu.items : [])) if (it && it.calcGroup) Object.defineProperty(it, "sub", { get: () => S.calcMenu(it.calcGroup) });
  setupSchema().catch(() => {});
  const setsEntry = MENUS[3].items.find((x) => x && x.label === "Selection Sets");
  Object.defineProperty(setsEntry, "sub", { get: () => [...setItems(edOf(active)), "select.sets"] });
  collectPieces();
  mirrorClassicControls();
  initMenubar(MENUS, ctx, run);
  renderToolbar();
  initBottom();
  initPalette({ ctx, run, toast: (m, k) => toast(m, k), files: () => M().state.files, openFile: openDoc,
    docs: () => docs.map((d) => ({ id: d.id, title: d.title, path: docPath(d), active: d === active })),
    openDoc: (id) => activate(docs.find((d) => d.id === id)),
    structure: () => structureOf(active),
    traj: () => (trajSteps() ? { steps: trajSteps(), labels: M().state.traj.labels || [] } : null),
    goToStep: (n) => trajStep(n, true), copy: (t) => copyText(t), setting: (k) => settings.get(k),
    selectAtom: (i) => jumpToAtom(i) });
  installTooltips();
  initContext({ viewer: mainViewer, hovered: () => M().state.hover || null, model: (v) => structureModel(v) });
  initPanelControls();
  splitter($("#split-explorer"), "--w-explorer", "wExplorer", 200, 480, 1);
  splitter($("#split-inspector"), "--w-inspector", "wInspector", 240, 480, -1);
  splitter($("#split-bottom"), "--h-bottom", "hBottom", 120, 600, -1, "y");
  applyLayout();
  $("#palette-kbd").textContent = shortcutOf("view.palette");
  if ($("#o-edit-step")) $("#o-edit-step").addEventListener("click", () => run("traj.editStep"));
  window.__maceToast = (m, kind) => toast(m, kind);
  configureToasts({ infoSeconds: () => Number(settings.get("notifications.infoSeconds")) || 6, openHistory: () => showNotifications() });
  settings.onChange(applySetting);
  // the 3Dmol views' drag (app.js): Settings › Viewport › Rotation and Angle snap, the camera's own rules
  Object.assign(M(), { rotationPrefs: () => rotationPrefs((k) => settings.get(k)), dragOrientation, turntableUp });
  M().showTab("welcome");
  afterSwitch();
  performance.mark("mace:ready");
  document.documentElement.dataset.ready = "1";
  // 3Dmol after first paint, when the browser is idle
  (window.requestIdleCallback || ((f) => setTimeout(f, 200)))(() => M().load3Dmol().catch(() => {}), { timeout: 2000 });
  // the editor's modules: fetched ahead (modulepreload) once the pointer rests
  // on a structure file for 150 ms, else when the first structure opens; an
  // idle page and a reload fetch none of them
  let hoverTimer = 0;
  document.addEventListener("pointerover", (e) => {
    const row = e.target.closest && e.target.closest("#file-list .file[data-path], #tab-welcome [data-open]");
    clearTimeout(hoverTimer);
    if (!row) return;
    if (E) {
      const p = row.dataset.path || row.dataset.open;
      if (["cif", "d12"].includes(fileKind(p))) hoverTimer = setTimeout(() => prefetchStructure(p), 150);
      return;
    }
    const path = row.dataset.path || row.dataset.open;
    if (EDITOR_KINDS.includes(fileKind(path))) hoverTimer = setTimeout(() => { preloadEditor(); prefetchStructure(path); }, 150);
  });
}

const EDITOR_MODULES = ["js/editor/main.js", "js/editor/doc.js", "js/viewport/scene.js", "js/viewport/camera.js",
  "js/viewport/pointer.js", "js/geom/core.js", "js/geom/client.js"];
function preloadEditor() {
  if (document.querySelector('link[rel=modulepreload][data-editor]')) return;
  for (const m of EDITOR_MODULES) {
    const l = document.createElement("link");
    l.rel = "modulepreload"; l.href = asset(m); l.dataset.editor = "1";
    document.head.appendChild(l);
  }
}

async function onAppReady(detail) {
  info = detail;
  IT.setDefaultTier(info && info.inspector_tier);
  // the profile (layout, Recent, remembered choices) from the server; a pre-1.1.5 browser's keys go up once
  profile.init((...a) => M().api(...a), M().state.boot && M().state.boot.profile, info);
  profile.setNotifier((m, k) => toast(m, k));
  if (info && info.root_gen !== undefined) window.__maceRootGen = String(info.root_gen);
  if (info && info.cpu) cpuState = info.cpu;
  migrateRecent();
  profile.migrateLegacy();
  profileReady = true;
  adoptProfileLayout();
  applyFilesOpen();
  profile.onChange(() => { adoptProfileLayout(); applyFilesOpen(); renderOpenEditors(); renderRecentSection(); renderWelcome(); });
  // per-document view options: the server's copy, localStorage as its cache (docprefs.js)
  docprefs.init((...a) => M().api(...a), (info && info.dir_hash) || "", "api/docprefs");
  // (the Rotate tool's pivot alone changes nothing drawn: no redraw)
  docprefs.onChange((kind, doc, keys) => {
    if (kind !== "display" || (keys && keys.length && keys.every((k) => DS.GROUPS.rotate.includes(k)))) return;
    displayChanged(doc ? docs.find((x) => x === doc || (x.path && x.path === doc.path)) || null : viewDoc());
  });
  docprefs.retryPending();
  renderWelcome(); renderStatus(); updateTitle();
  renderRoot();
  renderDbSection();
  connect();
  if (!PANE) initCluster();
  // the served folder's mace-folder.toml, above the user's own values for the keys it sets
  settings.setFolder(M().state.boot && M().state.boot.folder, (info && info.dir_hash) || "");
  { const fw = (settings.folderInfo().warnings || []).length;
    if (fw && !PANE) toast(`This folder's mace-folder.toml: ${fw} problem${fw === 1 ? "" : "s"}; what it could not use is ignored`, "warning",
      { label: "Show", run: () => settingsDialog("folder") }); }
  settingsPath = await settings.syncFromServer(M().state.boot && M().state.boot.settings);
  ["theme", "density", "uiScale", "reduceMotion", "showCell", "showBoundary", "labels", "colorScheme"].forEach(applySetting);
  if (!PANE) adoptPaletteDefault();
  booted = true;
  printPatternsLayer();
  document.documentElement.dataset.appReady = "1";
  if (PANE) return;            // the right editor group: the window offers drafts and restores the split
  noticeSetupDefaults();
  offerDrafts();               // modified documents from before a reload or a restart (spec §2.6)
  if (GR) GR.restore();        // the split of this folder (groups.js)
}
/* Settings' Functional and Basis set were saved but unused in 1.1.5; 1.1.6 makes them Calculation
 * Setup's defaults. Said once, on the first 1.1.6 start, to whoever had set them then. */
function noticeSetupDefaults() {
  const FLAG = "notice.setupDefaults116";
  if (profile.memory(FLAG)) return;
  profile.remember(FLAG, true);
  const f = settings.userValue("functional"), b = settings.userValue("basisSet");
  if (!f && !b) return;
  const set = [f && `Functional ${f}`, b && `Basis set ${b}`].filter(Boolean).join(" and ");
  toast(`Calculation Setup now uses your ${set} (Settings › MACE defaults) for decks made from a CIF. `
    + "In 1.1.5 these were saved but not used; clear them to keep MACE's defaults.", "info",
    { label: "Show", run: () => settingsDialog("mace"), ms: 30000 });
}
let appReadyHandled = false;
function appReady() { if (!appReadyHandled) { appReadyHandled = true; onAppReady(M().state.info || null); } }
document.addEventListener("mace-app-ready", appReady);
if (window.MACE && window.MACE.state.ready) queueMicrotask(appReady);

/* ================================================================ test and bench hooks */
/* The model holding the structure's atoms: the scene's structure layer in a
 * structure document (the selection and hover shells are models too). */
function structureModel(v) {
  if (!v) return null;
  if (E && v === M().state.viewers["viewer-main"] && E.scene() && E.scene().layers.structure) return E.scene().layers.structure;
  return v.getModel();
}
/* ================================================================ editor groups (groups.js)
 * Each group (this page, and the right group's frame) offers the same API. */
const VIEWER_OF = { structure: "viewer-main", trajectory: "viewer-before", volume: "viewer-volume", freq: "viewer-freq" };
function activeViewer() {
  const id = active && VIEWER_OF[active.kind];
  return id && M() && M().state.viewers ? M().state.viewers[id] || null : null;
}
const movablePath = (d) => (d ? d.path || d.srcPath || d.runPath || null : null);
function docDesc(d) {
  return { id: d.id, path: movablePath(d), kind: d.kind, title: d.title, dirty: isDirty(d), tab: d.tab || null,
    as: d.kind === "structure" && d.fileKind === "out" ? "structure" : "" };
}
async function reopenDesc(x) {
  if (!x || !x.path) return null;
  if (x.kind === "freq") return openModes(x.path);
  if (x.kind === "runplot") return openRunPlot(x.path, x.tab || null);
  if (x.kind === "mech") return openMech(x.path);
  return openDoc(x.path, x.as ? { as: x.as } : {});
}
function focusActiveView() {
  const el = document.querySelector("#views > .panel.active [tabindex], #views > .panel.active .viewer") || document.getElementById("views");
  try { (el || document.body).focus({ preventScroll: true }); } catch (e) { /* not focusable */ }
}
const groupHost = {
  toast, profile, dirHash: () => (info && info.dir_hash) || "", isDirty, closeDoc, activeDoc: () => active,
  docCount: () => docs.length, desc: docDesc, reopen: reopenDesc, viewer: activeViewer, focusView: focusActiveView,
  viewChanged: () => updateTriad(), exists: (p) => !!(M() && M().fileByPath(p)),
  setupOpen: () => setupOpen,
  afterChange: () => { renderTabs(); labelTabs(); renderStatus(); refresh(); },
  reorder: (id, i) => moveDocTo(docs.find((d) => d.id === id), i), reopenAt: (x, i) => reopenAt(x, i),
  docById: (id) => docs.find((d) => d.id === id) || null, announce: (text) => announce(text),
};
/* A moved document opens in this group at `i` of its tabs (null: the end). */
async function reopenAt(x, i = null) {
  await reopenDesc(x);
  const d = active && movablePath(active) === x.path ? active : docs.find((y) => movablePath(y) === x.path);
  if (!d) return null;
  if (i != null) moveDocTo(d, i);
  return docDesc(d);
}
/* Said to a screen reader (the polite live region). */
function announce(text) {
  const el = document.getElementById("sr-live");
  if (!el) return;
  el.textContent = "";
  requestAnimationFrame(() => { el.textContent = text; });
}
/* The tab strip and its tabs, in this page's client px (a drag over the right group, groups.js). */
function tabStrip() {
  const el = $("#doc-tabs"), r = el.getBoundingClientRect();
  return { left: r.left, top: r.top, right: r.right, bottom: r.bottom,
    tabs: [...el.querySelectorAll(".doc-tab")].map((t) => { const b = t.getBoundingClientRect(); return { id: Number(t.dataset.doc), left: b.left, right: b.right }; }) };
}
/* The right editor group's own Inspector and bottom panel, in the room the
 * window gives its frame (groups.js chrome()): sizes in CSS px, 0 = none. */
function paneChrome(c) {
  if (!PANE) return;
  const root = document.documentElement;
  const insp = Math.max(0, Math.round(Number(c && c.insp) || 0)), bottom = Math.max(0, Math.round(Number(c && c.bottom) || 0));
  const was = `${root.style.getPropertyValue("--pane-insp-w")}|${root.style.getPropertyValue("--pane-bottom-h")}|${root.className}`;
  root.classList.toggle("pane-insp", insp > 0);
  root.classList.toggle("pane-bottom", bottom > 0);
  root.style.setProperty("--pane-insp-w", `${insp}px`);
  root.style.setProperty("--pane-bottom-h", `${bottom}px`);
  if (was !== `${insp}px|${bottom}px|${root.className}`) { revealBottomTab(); relayout(); }
}
window.__macePane = {
  chrome: (c) => paneChrome(c),
  // the window's settings, theme preview and element colors (the window owns the profile)
  adoptSettings: (vals) => settings.adopt(vals),
  previewTheme: (id) => { M().applyTheme(id || settings.get("theme")); updateTriad(); },
  adoptColors: (pal, draft) => { M().state.paletteDraft = draft || null; M().setPalettes(pal || { palettes: {} }); },
  run: (id, args) => run(id, args), ctx: () => ctxHere(),
  docs: () => docs.map(docDesc), active: () => (active ? docDesc(active) : null),
  reopen: (x) => reopenDesc(x), close: (id, force) => closeDoc(docs.find((d) => d.id === id), force),
  reopenAt: (x, i) => reopenAt(x, i), reorder: (id, i) => moveDocTo(docs.find((d) => d.id === id), i), tabStrip: () => tabStrip(),
  viewer: () => activeViewer(), openModes: (p, o) => openModes(p, o), focusView: () => focusActiveView(),
  // Link Cameras: the rotation of this group's 3D view, read and set in this page
  getQ: () => { const v = activeViewer(); return v ? Array.from(v.getView().slice(4, 8)) : null; },
  setQ: (q) => {
    const v = activeViewer();
    if (!v || !q) return;
    const w = v.getView();
    v.setView([w[0], w[1], w[2], w[3], Number(q[0]), Number(q[1]), Number(q[2]), Number(q[3])]);
    v.render(); updateTriad();
  },
};

window.__mace = {
  lastCommand: () => lastCommand, sidebar: () => sidebar,
  // 1.1.6: the cells the optimization view repeats (whole cells of the document's range; 1 along a non-periodic axis)
  commandChecked: (id) => { const c = registry.get(id); return c && c.checked ? !!c.checked(ctx()) : null; },
  optCells: () => { const d = viewDoc(); if (!d || d.kind !== "trajectory") return [1, 1, 1]; const p = viewPbc(); return rangeCells().map((n, k) => (p[k] ? n : 1)); },
  // the per-document display state (viewport/display_state.js) and its storage
  display: DS, docprefs, profile,
  spyRun: (on) => { if (on) { window.__maceRan = []; runSpy = (id) => window.__maceRan.push(id); } else runSpy = null; },
  openCluster: (tab) => openCluster(tab), openJob: (id) => openJob(id), clusterView: () => clusterView, jobView: () => jobView,
  openWorkflows: (key) => openWorkflows(key), workflowView: () => workflowView,
  newWorkflow: (a) => openWorkflowBuilder(a), wfBuilder: () => window.__maceWfBuilder || null,
  storageChip: () => storageChip,
  renderRecent: () => renderWelcome(),
  // every lazily loaded module settled, one frame later (a dialog loading over a slow tunnel has opened)
  lazyIdle: () => Promise.allSettled(Object.values(lazyLoads)).then(() => new Promise((r) => requestAnimationFrame(() => r(true)))),
  get commands() { return [...registry.values()].map((c) => ({ id: c.id, title: c.title, category: c.category, keys: c.keys, soon: !!c.later, later: c.later || "", menuTitle: typeof c.menuTitle === "string" ? c.menuTitle : c.title, palette: c.palette !== false })); },
  menus: () => MENUS.map((m) => ({ label: m.label, items: m.items.map((it) => it === "-" ? "-" : typeof it === "string" ? it : { label: it.label }) })),
  run, ctx, docs: () => docs.map((d) => ({ id: d.id, path: d.path, kind: d.kind, title: d.title, dirty: isDirty(d) })),
  doc: () => active && { ...active, data: undefined, dirty: isDirty(active), structure: structureOf(active) },
  open: openDoc, activate: (i) => activate(docs[i]), compareWith: (p) => compareWith(p),
  closeAll: () => [...docs].forEach((d) => closeDoc(d, true)),
  get drafts() { return E ? E.drafts : null; }, offerDrafts, copyImage, viewName: () => viewName(),
  exportSvg: async (o = {}) => (await lazy("export_image")).exportSvg(exportCtx(), o), exportImage: () => exportImage(),
  viewState: () => {
    const v = M().state.viewers["viewer-main"];
    return { cell: optChecked("opt-cell"), bonds: optChecked("opt-bonds"), labels: labelMode(), boundary: optChecked("opt-boundary"),
      range: range(), style: style(), projection: settings.get("projection"), tool, view: v ? v.getView() : null,
      colorScheme: M().schemeName(), sphereScale: M().sphereScale(optChecked("opt-bonds")), bondRadius: settings.get("bondRadius"),
      bondTolerance: settings.get("bondTolerance"), background: M().viewerBg(),
      displayedAtoms: E && E.scene() && E.scene().display ? E.scene().display.n : 0,
      cameraOrtho: v && v.camera ? !!v.camera.ortho : null };
  },
  /* What 3Dmol draws for atom i of the active structure, read from its model:
   * the sphere color (hex) and scale, and the radius of the drawn sphere. */
  atomStyle: (i) => {
    const v = mainViewer(), m = structureModel(v);
    if (!m) return null;
    const all = m.selectedAtoms({ serial: i + 1 });
    const a = all.find((x) => !(x.properties && x.properties.image)) || all[0];
    if (!a || !a.style.sphere) return null;
    const hex = window.$3Dmol.getColorFromStyle(a, a.style.sphere).getHex();
    const sph = a.intersectionShape && a.intersectionShape.sphere && a.intersectionShape.sphere[0];
    const radius = sph ? sph.radius : (m.getRadiusFromStyle ? m.getRadiusFromStyle(a, a.style.sphere) : null);
    return { color: "#" + hex.toString(16).padStart(6, "0"), scale: a.style.sphere.scale, radius,
      x: a.x, y: a.y, z: a.z };
  },
  /* A pixel of the 3D view (CSS px from the canvas's top left). 3Dmol draws
   * into a shared WebGL canvas and hands each frame to the visible one as a
   * bitmap, which only toDataURL reads: so it renders and encodes in one task
   * (the frame is not kept after compositing), then decodes. A promise. */
  snapshotPixel: (x, y) => {
    const v = mainViewer();
    if (!v) return Promise.resolve(null);
    const cv = v.getCanvas();
    v.render();
    const url = cv.toDataURL("image/png"), k = cv.width / cv.clientWidth, w = cv.width, h = cv.height;
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const c = document.createElement("canvas"); c.width = w; c.height = h;
        const g = c.getContext("2d"); g.drawImage(img, 0, 0);
        resolve(Array.from(g.getImageData(Math.round(x * k), Math.round(y * k), 1, 1).data));
      };
      img.onerror = () => resolve(null);
      img.src = url;
    });
  },
  /* Where atom i is on screen, in CSS px from the canvas's top left. */
  atomScreen: (i) => {
    const v = mainViewer(), m = structureModel(v);
    if (!m) return null;
    const all = m.selectedAtoms({ serial: i + 1 });
    const a = all.find((x) => !(x.properties && x.properties.image)) || all[0];
    const p = v.modelToScreen({ x: a.x, y: a.y, z: a.z }), r = v.getCanvas().getBoundingClientRect();
    return { x: p.x - r.left, y: p.y - r.top, cx: r.width / 2, cy: r.height / 2 };
  },
  settings: { get: settings.get, set: settings.set, all: settings.all, nonDefault: settings.nonDefault, flush: settings.flush,
    resetAll: settings.resetAll, defaults: () => ({ ...settings.DEFAULTS }), reset: settings.reset,
    source: settings.source, userValue: settings.userValue, saveAsDefault: settings.saveAsDefault, saveToFolder: settings.saveToFolder, migrate: settings.migrate },
  palette: { results: (q) => paletteResults(q, ctx()), open: openPalette, close: closePalette, isOpen: isPaletteOpen,
    bench: paletteBench, domRows,
    // the pure ranking alone on n items, for a regression check in the regular suite
    rankBench: (n = 2500) => { const items = Array.from({ length: n }, (_, k) => ({ label: `Item: Supercell ${k} gamma O${k}`, title: `Supercell ${k}` }));
      const p = rank.prepare(items); const t0 = performance.now(); for (const q of ["s", "sup", "gamma", "O12"]) rank.rank(q, p); return (performance.now() - t0) / 4; } },
  rank: { fracToCart: rank.fracToCart, cartToFrac: rank.cartToFrac, parseTriple: rank.parseTriple },
  contextAt, contextFor,
  parseChord, formatChord, fuzzyScore, evalWhen, triadSvg,
  /* why a command is off right now ("" when it runs), as its menu item says */
  menuReason: (id, args) => disabledReason(registry.get(id), args ? { ...ctx(), args } : ctx()),
  modalRows: () => Object.values(commands.MODAL_LAYERS).reduce((t, l) => t + l.rows.length, 0),
  activeDoc: () => active,
  panelContext: () => PC.state(), pickRelated: (p) => PC.pick(active, p),
  openUntitled: (p, title) => openUntitled(p, title),
  // Firefox gives a synthetic ClipboardEvent an empty DataTransfer: the tests hand the handler an event-like object
  onPaste: (target, text) => { let prevented = false; onPaste({ target, clipboardData: { getData: () => text }, preventDefault: () => { prevented = true; } }); return prevented; },
  runProblemAction: (source) => { const p = problems.find((x) => x.source === source); return p && p.action ? p.action.run() : null; },
  tasks: () => [...taskStore.values()].map((t) => ({ ...t })), problems: () => problems.map((p) => ({ ...p, action: p.action ? p.action.label : null })),
  clientId: () => clientId, apiTask: (p, b) => apiTask(p, b),
  checkDisk: () => { diskCheckAt = -Infinity; return checkDisk(); }, exportName: (fmt) => exportName(active, fmt),
  exportStructure: (fmt, mode) => exportStructure(fmt, mode), heldIds: () => docs.filter((d) => d.held).map((d) => d.held),
  openTestBuilder: (opts) => lazy("builder_util").then((m) => m.openIdentity(H, opts || {})),
  dropFiles: (dt) => dropFiles(dt), importClipboard: (text) => importClipboard(text || ""),
  builderFrame: () => (active && active.builderFrame ? active.builderFrame.info() : null),
  openHeld: async (id, title, meta = null) => {
    const [r, core] = await Promise.all([M().api("held?" + M().q({ id })), import("./geom/core.js")]);
    return openHeld(core.decodePacked(r), title, id, meta);
  },
  /* the viewport and the editor, once loaded (null before the first structure opens) */
  get viewport() { return E ? E.viewport : null; },
  triad: () => {
    const v = M().state.viewers["viewer-main"], s = structureOf(active);
    if (!v || !s) return [];
    return triadAxes(v.getView().slice(4, 8), s.cell).map((a) => ({ axis: "abc"[a.k], x: a.x, y: a.y, z: a.z }));
  },
  get editor() {
    if (!E) return null;
    const ed = () => edOf(active);
    return { selection: () => (ed() ? [...ed().selection].sort((a, b) => a - b) : []),
      select: (ix, mode) => ed() && E.select(ed(), ix, mode), doc: ed, E };
  },
  loadEditor: () => loadEditor().then(() => true), bondValues: () => bondValues(),
  // Clean › Bonds held, as the Clean panel last set it (the classic Clean sends it too)
  cleanBondsHeld: () => profile.memory("cleanBondsHeld", "shown"),
  /* Calculation Setup (js/setup/panel.js), once loaded; setupReady() loads it */
  get setup() { return setupMod; }, setupReady: () => setupPanel().then(() => true), shortcut: (id) => shortcutOf(id),
  get textView() { return textView; }, openAsText: (p) => openAsText(p),
  get dbView() { return dbView; }, exportTable: (fmt) => exportTable(fmt),
  get plotView() { return plotView; }, get runView() { return runView; }, get mechView() { return mechView; }, openMech: (p) => openMech(p), get galleryView() { return galleryView; }, openGallery: (o) => openGallery(o || {}), get freqView() { return freqView; }, openModes: (p, o) => openModes(p, o),
  notifications: { recent: (n) => notifications.recent(n), clear: () => notifications.clear(), unread: () => notifications.unreadCount(),
    subscribe: (fn) => notifications.subscribe(fn) },
  openStructure: (p) => openStructure(p), openInfo: (p) => openInfo(p), openJobLog: (p) => openJobLog(p),
  openFreqSpectrum: (p, kind) => openFreqSpectrum(p, kind), openVolume: (p) => openVolume(p),
  openConvergence: (paths) => openConvergence(paths),
  get volumeView() { return volumeView; }, colorByPotential: (p) => volumeViewer().then((v) => v.setColor(p)),
  volumeDifference: (paths, coefficients) => apiTask("volume/combine", { paths, coefficients }).then((meta) => openVolume(null, { title: meta.name, combo: true, meta })),
  lastCopy: () => window.__maceLastCopy || null,
  asset: (n) => asset(n),
  keymap: { apply: (o) => applyKeymap(o), conflicts, allConflicts, chordId, chordFromEvent, isReserved, riskOf, browserNavOf,
    whensOverlap, reserved: () => [...RESERVED.keys()], risky: () => [...RISKY.keys()], browserNav: () => [...BROWSER_NAV.keys()],
    keysOf: (id) => (command(id) ? [...command(id).keys] : null), defaultKeysOf: (id) => (command(id) ? [...command(id).defaultKeys] : null),
    pseudo: () => [...pseudo.values()].map((c) => ({ id: c.id, keys: [...c.keys], menu: c.menu })),
    version: () => commands.keymapVersion },
  bench: {
    editor: () => (E ? E.bench : null),
    marks: () => performance.getEntriesByType("mark").map((m) => ({ name: m.name, t: m.startTime })),
    measures: () => performance.getEntriesByType("measure").map((m) => ({ name: m.name, d: m.duration })),
  },
};

if (window.MACE) boot(); else addEventListener("DOMContentLoaded", boot);
