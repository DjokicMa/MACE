/* Every command and every menu, in one place (spec §1.3, §2.2, §4.2). A
 * command that is not in this version (`later: owner`) shows disabled with
 * "Not in this version yet: <owner>", the work it arrives with; the shell (`S`)
 * supplies what the others act on. A command may take arguments (`run(ctx,
 * args)`): context menus pass the atom, file or tab they were opened on;
 * without them a command acts on the active document. */
import { register, registerPseudo } from "./commands.js";
import { THEMES, THEME_GROUPS } from "./themes.js";
/** View › Theme: each group's name, then its themes (js/themes.js). */
const themeMenu = () => THEME_GROUPS.flatMap((g, n) => [...(n ? ["-"] : []), { head: g.label },
  ...THEMES.filter((t) => t.group === g.id).map((t) => `view.theme.${t.id}`)]);

export function registerAll(S) {
  // owner: what the command arrives with, e.g. "arrives with the Calculation Setup panel"
  const later = (id, title, category, keys = [], icon = "", owner = "", extra = {}) =>
    register({ id, title, category, keys, icon, later: owner || "arrives in a later version", ...extra });
  const SETUP = "arrives with the Calculation Setup panel", ANALYSIS = "arrives with the Analysis tools",
    FILES = "arrives with the file tools", BUILDERS = "arrives with the builders";
  const structureDoc = "docKind == structure|trajectory";
  const FREQ_NO = "Not in the mode view: open its structure (Structure, top right)";
  const needsStructure = { when: structureDoc, reason: (c) => (c.docKind === "freq" ? FREQ_NO : "Needs a structure") };
  // the display options the mode view (a FREQ document) and the volume views (cube, orbitals) draw with too
  const view3D = "docKind == structure|trajectory|freq|volume";
  // a volume of a molecule has no cell: nothing to replicate, no cell axes to look along
  const VOL_MOLECULE = "This volume is of a molecule: it has no cell to replicate or to look along";
  const needsView = { when: view3D, reason: "Needs a structure" };
  const needsAlong = { when: `${view3D} && !volNonPeriodic`, reason: (c) => (c.volNonPeriodic ? VOL_MOLECULE : "Needs a structure") };
  // the comparison of a run's steps draws the run's own cell, unlabeled
  const CMP_NO = "The comparison draws the run's own cell: Edit This Step opens a step with every display option";
  // the mode view replicates a mode at Γ; away from Γ its images move with phases the output does not give
  const OFF_GAMMA = "One cell only: this mode is not at Γ, so its periodic images would need phases the output does not give";
  const notHere = (c) => (c.docKind === "trajectory" ? CMP_NO : c.docKind === "volume" ? VOL_MOLECULE : c.freqOffGamma ? OFF_GAMMA : "Needs a structure");
  const needsRange = { when: "docKind == structure|freq && !freqOffGamma || docKind == volume && !volNonPeriodic", reason: notHere };
  const needsLabels = { when: "docKind == structure|freq|volume", reason: notHere };
  // Phase 3: the work is in js/editor (loaded with the first structure); S.ed runs it
  const ed = (id) => (c, a) => S.ed(id, a);
  const editable = (extra = "") => ({ when: `docEditable${extra}`, reason: (c) => S.editReason(c) });
  const selecting = { when: "docKind == structure", reason: (c) => (c.docKind === "trajectory" ? "Trajectories are view-only. Use Edit This Step" : "Needs a structure") };

  /* ---------------- File */
  register({ id: "file.new", title: "New Structure…", category: "File", keys: ["Alt+N"], icon: "plus",
    aliases: ["new cell", "lattice", "space group"], desc: "A cell from its lattice, space group and sites", run: () => S.newStructure() });
  register({ id: "file.open", title: "Open…", category: "File", keys: ["Mod+O"], icon: "folder-open",
    aliases: ["quick open", "go to file"], run: () => S.palette("#") });
  // no default key: Mod+K is the browser's, and chords are single keys here (plan T6)
  register({ id: "file.openFolder", title: "Open Folder…", category: "File", icon: "folder-open",
    aliases: ["switch folder", "change folder", "cd", "workspace"], desc: "Serve another folder in this window (Home, Scratch)",
    when: "openFolderOn", reason: "Turned off when this server started (--root none)", run: () => S.openFolder() });
  register({ id: "file.quickOpen", title: "Go to File…", category: "File", keys: ["Mod+P"], icon: "file",
    run: () => S.palette("#") });
  // Mod+Shift+V reaches the browser's paste event (shell.js); the menu reads the clipboard itself
  register({ id: "file.importClipboard", title: "Import from Clipboard…", category: "File", keys: ["Mod+Shift+V"], icon: "clipboard-paste",
    aliases: ["paste poscar", "paste cif", "paste xyz"], desc: "A POSCAR, CIF or XYZ from the clipboard, as a new structure",
    run: () => S.importClipboard() });
  // saving (js/editor/save.js): a .cif in place, anything else through Save As
  const saveable = { when: "docKind == structure", reason: (c) => (c.docKind === "trajectory" ? "Trajectories are view-only. Use Edit This Step" : "Needs a structure") };
  register({ id: "file.save", title: "Save", category: "File", keys: ["Mod+S"], icon: "save", ...saveable,
    tipTitle: () => (S.dirty() ? "Save (modified)" : "Save"), desc: "A CIF in place (keeping a .bak), anything else as a new CIF", run: ed("file.save") });
  register({ id: "file.saveAs", title: "Save As…", category: "File", keys: ["Mod+Shift+S"], ...saveable,
    desc: "CIF (every atom or symmetrized), POSCAR or XYZ, in the served folder", run: (c, a) => S.ed("file.saveAs", a) });
  register({ id: "file.saveAll", title: "Save All", category: "File", keys: ["Mod+Alt+S"], icon: "save-all",
    when: "anyDirty", reason: "Nothing is modified", run: () => S.saveAll() });
  register({ id: "file.revert", title: "Revert File", category: "File", when: "docKind == structure && docDirty && docHasPath",
    reason: (c) => (c.docKind !== "structure" ? "Needs a structure" : !c.docHasPath ? "Not saved yet: nothing to revert to" : "No unsaved changes"),
    run: () => S.revert() });
  register({ id: "file.exportLegend", title: "Export Element Legend…", category: "File", icon: "list",
    aliases: ["legend", "element legend", "color key", "palette legend"], when: structureDoc, reason: "Needs a structure",
    desc: "The structure's elements in the palette's colors: SVG, PNG or a table, for figures", run: () => S.legendActive() });
  register({ id: "file.exportLegendOpen", title: "Export Element Legend of All Open Documents…", category: "File", icon: "list",
    aliases: ["legend", "element legend"], enabled: () => S.openStructureCount() > 0, reason: "No structure is open",
    desc: "The union of the elements of every open structure", run: () => S.legendOpenDocs() });
  register({ id: "explorer.legend", title: "Export Element Legend…", category: "Explorer", icon: "list",
    enabled: (c) => !!(c.args && (c.args.dir !== undefined || S.pathOf(c.args))), reason: "Right-click structure files or a folder in the Explorer",
    desc: "The elements of the selected files (or the folder's structures)", run: (c, a) => S.legendExplorer(a) });
  register({ id: "file.exportImage", title: "Export Image…", category: "File", keys: ["Mod+Shift+X"], icon: "image",
    when: "docKind == structure", reason: "Needs a structure", desc: "A PNG of the view: up to 4× or a custom size, transparent if you like",
    run: () => S.exportImage() });
  // Export Structure downloads to this computer (decision D8); Save As writes into the served folder
  const dl = "Downloads to this computer (over ssh -L, your laptop); Save As writes into the served folder";
  [["CIF", "CIF (P1)", "cif", "p1", "Every atom, as Save As writes it"], ["CIFSym", "CIF (Symmetrized)", "cif", "symmetrized", "The spglib-refined cell and asymmetric unit"],
    ["POSCAR", "POSCAR", "poscar", "p1", "VASP, direct coordinates"], ["XYZ", "XYZ", "xyz", "p1", "Symbols and Cartesian positions, no cell"],
    ["extXYZ", "extXYZ", "extxyz", "p1", "Extended XYZ: with the cell"]].forEach(([k, t, fmt, mode, what]) =>
    register({ id: `file.export.${k}`, title: t, category: "File: Export Structure", icon: "download", ...saveable,
      when: "docKind == structure", desc: `${what}. ${dl}`, run: () => S.exportStructure(fmt, mode) }));
  later("file.export.QE", "Quantum ESPRESSO Input", "File: Export Structure", [], "", "needs the Quantum ESPRESSO backend");
  register({ id: "file.copyImage", title: "Copy Image to Clipboard", category: "File", keys: ["Mod+Shift+C"], icon: "copy",
    when: "docKind == structure", reason: "Needs a structure", run: () => S.copyImage() });
  register({ id: "file.closeTab", title: "Close Tab", category: "File", keys: ["Alt+W"], icon: "x",
    when: "hasDoc", reason: "No document is open", run: () => S.closeDoc(S.activeDoc()) });
  register({ id: "file.closeAll", title: "Close All Tabs", category: "File", when: "hasDoc",
    reason: "No document is open", run: () => S.closeAll() });
  // in the Move tool with the viewport focused, Alt+PageUp/PageDown nudge along z instead
  register({ id: "file.nextTab", title: "Next Tab", category: "File", keys: ["Alt+PageDown"], when: "docCount >= 2 && !moveNudge",
    reason: "Needs two open documents", run: () => S.cycleDoc(1) });
  register({ id: "file.prevTab", title: "Previous Tab", category: "File", keys: ["Alt+PageUp"], when: "docCount >= 2 && !moveNudge",
    reason: "Needs two open documents", run: () => S.cycleDoc(-1) });
  register({ id: "file.goToDoc", title: "Go to Open Document…", category: "File", keys: ["Mod+E"], icon: "files",
    when: "hasDoc", reason: "No document is open", aliases: ["switch tab", "open editors"], run: () => S.palette("%") });
  // a recent file's menu (plan T10): the explorer's RECENT section and the welcome screen
  const recentArgs = { palette: false, enabled: (c) => !!S.pathOf(c.args), reason: "Right-click a recent file" };
  register({ id: "file.recent.open", title: "Open", category: "File", icon: "file", ...recentArgs,
    enabled: (c) => !!S.pathOf(c.args) && !!S.fileExists(S.pathOf(c.args)), reason: "Missing: the file is no longer in the folder",
    run: (c, a) => S.openPath(S.pathOf(a)) });
  register({ id: "file.recent.pin", title: "Pin", category: "File", icon: "pin", ...recentArgs,
    menuTitle: (c) => (S.isRecentPinned(S.pathOf(c.args)) ? "Unpin" : "Pin"), run: (c, a) => S.pinRecent(S.pathOf(a)) });
  register({ id: "file.recent.remove", title: "Remove from Recent", category: "File", icon: "x", ...recentArgs,
    run: (c, a) => S.removeRecent(S.pathOf(a)) });
  register({ id: "file.recent.copyPath", title: "Copy Path", category: "File", icon: "copy", ...recentArgs,
    run: (c, a) => S.copyText(S.pathOf(a), "Path copied") });
  register({ id: "file.clearRecent", title: "Clear Recently Opened", category: "File", run: () => S.clearRecent() });
  register({ id: "file.copyRootPath", title: "Copy Folder Path", category: "File", icon: "copy",
    desc: "The full path of the folder this server shows", run: () => S.copyRootPath() });
  // no default key: Mod+Q is the browser's own quit (plan T10)
  register({ id: "file.signOut", title: "Sign Out and Stop Server…", category: "File", icon: "power",
    aliases: ["quit", "exit", "log out", "logout", "sign out", "shut down", "stop server"],
    desc: "End this session and stop MACE GUI on the server", run: () => S.signOut() });
  register({ id: "file.settings", title: "Settings…", category: "File", keys: ["Mod+,"], icon: "settings",
    aliases: ["preferences", "options", "keymap", "keybindings"], run: () => S.settingsDialog() });

  /* ---------------- Edit */
  register({ id: "edit.undo", title: "Undo", category: "Edit", keys: ["Mod+Z"], icon: "undo-2",
    when: "canUndo && !inputFocus", reason: "Nothing to undo", run: () => S.undo() });
  register({ id: "edit.redo", title: "Redo", category: "Edit", keys: ["Mod+Shift+Z", "Mod+Y"], icon: "redo-2",
    when: "canRedo && !inputFocus", reason: "Nothing to redo", run: () => S.redo() });
  register({ id: "edit.cut", title: "Cut", category: "Edit", keys: ["Mod+X"], icon: "scissors",
    ...editable(" && hasSelection && !inputFocus && !textSelected"), desc: "Copy the selected atoms, then delete them (one undo step)", run: ed("edit.cut") });
  // Mod+C copies page text as usual while text is selected or a field has focus
  register({ id: "edit.copy", title: "Copy", category: "Edit", keys: ["Mod+C"], icon: "copy",
    when: "docKind == structure && hasSelection && !inputFocus && !textSelected", reason: "Select atoms first",
    desc: "The selected atoms as an XYZ block: fractional and Cartesian", run: ed("edit.copy") });
  // Mod+V goes to the browser's paste event (no prompt); the menu and palette read the clipboard
  register({ id: "edit.paste", title: "Paste", category: "Edit", keys: ["Mod+V"], icon: "clipboard-paste",
    when: "!inputFocus", reason: "A text field has focus", desc: "Copied atoms into the structure, or any structure text as a new tab",
    run: async () => { const t = await S.readClipboard(); return S.pasteText(t); } });
  register({ id: "edit.duplicate", title: "Duplicate", category: "Edit", keys: ["Mod+D"], icon: "",
    ...editable(" && hasSelection"), desc: "Copies of the selected atoms, one step to the right, then grabbed", run: ed("edit.duplicate") });
  register({ id: "edit.delete", title: "Delete", category: "Edit", keys: ["Delete", "Backspace"], icon: "trash-2",
    ...editable(" && hasSelection"), run: ed("edit.delete") });
  // E with a selection opens the element picker; without one it arms the Replace tool;
  // in the Add Atom tool it edits the element to add (tool.addElement)
  register({ id: "edit.replaceElement", title: "Replace Element…", category: "Edit", keys: ["E"], icon: "replace-atom",
    when: "docEditable && hasSelection && tool != add", reason: (c) => S.editReason(c) || (c.tool === "add" ? "E edits the element to add in the Add Atom tool" : "Select atoms first"),
    run: ed("edit.replaceElement") });
  // Mod+Shift+T (the spec's choice) reopens a closed tab in Firefox and Chrome: reserved
  register({ id: "edit.setCoordinates", title: "Set Coordinates…", category: "Edit", keys: ["Mod+Shift+U"], icon: "crosshair",
    ...editable(" && hasSelection"), run: () => S.lazyDialog("coords_dialog", "open") });
  register({ id: "edit.editCell", title: "Edit Cell…", category: "Edit", keys: ["Mod+Shift+L"], icon: "unit-cell",
    aliases: ["lattice", "alpha", "beta", "gamma"], ...editable(), run: () => S.lazyDialog("cell_dialog", "openEditCell") });
  register({ id: "edit.history", title: "Show History", category: "Edit", keys: ["Mod+Shift+H"], icon: "history",
    run: () => S.showBottom("history") });
  [["Left", "Nudge −x", "Alt+Left"], ["Right", "Nudge +x", "Alt+Right"], ["Up", "Nudge +y", "Alt+Up"], ["Down", "Nudge −y", "Alt+Down"],
    ["Out", "Nudge +z", "Alt+PageUp"], ["In", "Nudge −z", "Alt+PageDown"]].forEach(([d, t, k]) => register({ id: `edit.nudge${d}`,
    title: t, category: "Edit", keys: [k, k.replace("Alt+", "Alt+Shift+")], when: "moveNudge && docEditable && hasSelection",
    reason: (c) => (!c.moveNudge ? "Needs the Move tool and the viewport focused" : S.editReason(c) || "Select atoms first"),
    desc: "Shift: 10× the step", run: ed(`edit.nudge${d}`) }));

  /* ---------------- View */
  register({ id: "view.palette", title: "Command Palette…", category: "View", keys: ["F1", "Mod+Alt+K"], icon: "search",
    aliases: ["commands", "search"], run: () => S.palette("") });
  const along = [["a", "1", "view-a"], ["b", "2", "view-b"], ["c", "3", "view-c"]];
  along.forEach(([ax, key, ic], k) => register({ id: `view.along${ax.toUpperCase()}`, title: `View Along ${ax}`,
    menuTitle: ax, category: "View", keys: [key], icon: ic, ...needsAlong, run: () => S.viewAlong(k) }));
  along.forEach(([ax], k) => register({ id: `view.along${ax.toUpperCase()}Star`, title: `View Along ${ax}*`, menuTitle: `${ax}*`,
    category: "View", keys: [`Shift+${k + 1}`], ...needsAlong, desc: `Look down the reciprocal axis ${ax}* (perpendicular to the other two)`,
    run: () => S.viewAlong(`${ax}*`) }));
  register({ id: "view.alongUVW", title: "View Along [uvw]…", menuTitle: "[uvw]…", category: "View", keys: ["Alt+U"],
    when: "docKind == structure", reason: "Needs a structure", run: () => S.lazyDialog("cell_dialog", "openAlongUVW") });
  // Alt+arrows turn the view by 90° in the Select and Orbit tools; in the Move tool they nudge
  [["Left", "y", -90, "rotate-ccw"], ["Right", "y", 90, "rotate-cw"], ["Up", "x", -90, ""], ["Down", "x", 90, ""]].forEach(([d, axis, deg, ic]) =>
    register({ id: `view.rotate${d}`, title: `Rotate 90° ${d}`, menuTitle: d, category: "View", keys: [`Alt+${d}`], icon: ic,
      when: `${view3D} && !moveNudge`, reason: "Needs a structure", run: () => S.rotateView(axis, deg) }));
  register({ id: "view.rotateInPlaneMinus", title: "Rotate 90° In-Plane Left", menuTitle: "In-Plane −90°", category: "View",
    keys: ["Alt+["], ...needsView, run: () => S.rotateView("z", 90) });
  register({ id: "view.rotateInPlanePlus", title: "Rotate 90° In-Plane Right", menuTitle: "In-Plane +90°", category: "View",
    keys: ["Alt+]"], ...needsView, run: () => S.rotateView("z", -90) });
  // 15° turns with the arrow keys (Shift: 1°) while the viewport has focus
  [["Left", "y", -1], ["Right", "y", 1], ["Up", "x", -1], ["Down", "x", 1]].forEach(([d, axis, sign]) =>
    register({ id: `view.rotate15${d}`, title: `Rotate 15° ${d}`, category: "View", keys: [d, `Shift+${d}`],
      when: `${view3D} && viewportFocus`, reason: "Needs a structure", desc: "Shift: 1°",
      run: (c, a) => S.rotateView(axis, sign * (a && a.chord && a.chord.startsWith("Shift+") ? 1 : 15)) }));
  register({ id: "view.zoomIn3D", title: "Zoom In", category: "View", keys: ["+"], icon: "zoom-in",
    when: `${view3D} && viewportFocus`, reason: "Needs a structure", run: () => S.zoomView(1.25) });
  register({ id: "view.zoomOut3D", title: "Zoom Out", category: "View", keys: ["-"], icon: "zoom-out",
    when: `${view3D} && viewportFocus`, reason: "Needs a structure", run: () => S.zoomView(0.8) });
  register({ id: "view.reset", title: "Reset View", category: "View", keys: ["Home"], icon: "house",
    ...needsView, run: () => S.resetView() });
  register({ id: "view.frameSelection", title: "Frame Selection", category: "View", keys: ["F"], icon: "scan-search",
    ...needsStructure, desc: "Fit the selected atoms (or everything) to the view", run: () => S.frameSelection() });
  register({ id: "view.pivotCell", title: "Center Rotation on Unit Cell", category: "View", icon: "crosshair",
    when: "docKind == structure", reason: "Needs a structure",
    desc: "Turn the view about the center of the drawn cell (a molecule: of its atoms); zoom and orientation kept",
    run: () => S.pivotOn("cell") });
  register({ id: "view.pivotSelection", title: "Center Rotation on Selection", category: "View", icon: "crosshair",
    when: "docKind == structure && hasSelection", reason: "Select atoms first",
    desc: "Turn the view about the middle of the selected atoms; zoom and orientation kept", run: () => S.pivotOn("selection") });
  register({ id: "view.projection", title: "Orthographic / Perspective", category: "View", keys: ["5"], icon: "projection",
    when: "docKind == structure|trajectory|freq|volume", reason: "Needs a 3D view", checked: () => S.setting("projection") === "perspective",
    tipTitle: () => `Projection: ${S.setting("projection") === "orthographic" ? "orthographic" : "perspective"}`,
    run: () => S.toggleProjection() });
  const toggle = (id, title, key, ic, run, checked, extra = {}) => register({ id, title, category: "View", keys: [key], icon: ic,
    ...needsView, checked, run, ...extra });
  toggle("view.cell", "Show Unit Cell", "Shift+C", "unit-cell", () => S.toggleSetting("showCell"), () => S.setting("showCell"));
  // per document (Inspector › Range & boundary); the default stays in Settings › Viewport
  toggle("view.boundary", "Show Boundary Atoms", "Shift+A", "boundary", () => S.toggleDisplay("showBoundary"), () => S.displayValue("showBoundary"),
    { desc: "For this document: the atoms on the faces, edges and corners of the range", ...needsRange });
  toggle("view.bonds", "Show Bonds", "Shift+B", "bonds", () => S.toggleOpt("opt-bonds"), () => S.optChecked("opt-bonds"));
  const LABELS = [["off", "Off"], ["element", "Element"], ["index", "Index"], ["both", "Element+Index"], ["wyckoff", "Wyckoff"]];
  LABELS.forEach(([v, t]) => register({ id: `view.labels.${v}`, title: `Labels: ${t}`, menuTitle: t, category: "View",
    radio: true, ...needsLabels, checked: () => S.labelMode() === v, run: () => S.setLabels(v) }));
  register({ id: "view.labelsCycle", title: "Show Labels (Cycle)", category: "View", keys: ["Shift+L"], icon: "labels",
    ...needsLabels, checked: () => S.labelMode() !== "off", run: () => S.cycleLabels() });
  register({ id: "view.axes", title: "Show Axes", category: "View", keys: ["Shift+X"], icon: "axes",
    checked: () => S.setting("showAxes"), run: () => S.toggleSetting("showAxes") });
  register({ id: "view.measurements", title: "Show Measurements", category: "View", keys: ["Shift+M"], icon: "ruler",
    ...needsStructure, checked: () => S.measurementsShown(), run: () => S.toggleMeasurements() });
  toggle("view.polyhedra", "Show Polyhedra", "Shift+P", "hexagon", () => S.toggleDisplay("showPolyhedra"), () => S.displayValue("showPolyhedra"),
    { aliases: ["polyhedra", "octahedra", "tetrahedra", "coordination polyhedra"], desc: "For this document: coordination polyhedra around the cations (off above 2000 drawn atoms)",
      ...needsStructure });
  register({ id: "view.rangePanel", title: "Range and Boundary…", category: "View", icon: "supercell", ...needsStructure,
    aliases: ["boundary", "vesta range", "search molecules", "whole molecules", "bonded atoms", "fractional range"],
    desc: "Inspector › Range & boundary: the drawn range, boundary atoms and bonded atoms outside, for this document",
    run: () => S.displayPanel("range") });
  register({ id: "view.renderPanel", title: "Rendering…", category: "View", icon: "sun-moon", ...needsStructure,
    aliases: ["depth cue", "fog", "outline", "lighting", "ambient", "ambient occlusion", "background", "preset"],
    desc: "Inspector › Rendering: depth cueing, light, ambient light, outline and background, for this document",
    run: () => S.displayPanel("render") });
  [[1, "1×1×1"], [2, "2×2×2"], [3, "3×3×3"]].forEach(([n, t]) => register({ id: `view.range${n}`, title: `Display Range ${t}`,
    menuTitle: t, category: "View", radio: true, ...needsRange,
    checked: () => S.range() === n, run: () => S.setRange(n) }));
  register({ id: "view.rangeCustom", title: "Display Range Custom…", menuTitle: "Custom…", category: "View", radio: true,
    ...needsRange, checked: () => S.range() === 0, run: () => S.lazyDialog("cell_dialog", "openRangeCustom") });
  register({ id: "view.rangeCycle", title: "Cycle Display Range", category: "View", keys: ["Shift+R"], icon: "supercell",
    ...needsRange, tipTitle: () => `Display range ${S.rangeText()}`,
    run: () => S.setRange(S.range() >= 3 || S.range() === 0 ? 1 : S.range() + 1) });
  // Alt+1..4 only while the viewport has focus (spec §1.3): elsewhere Linux
  // browsers switch tabs with them, which the shell keeps from happening
  [["ballstick", "Ball and Stick", "Alt+1"], ["spacefill", "Space-Filling", "Alt+2"], ["stick", "Sticks", "Alt+3"],
    ["wireframe", "Wireframe", "Alt+4"]].forEach(([v, t, k]) => register({ id: `view.style.${v}`, title: `Style: ${t}`,
    menuTitle: t, category: "View", keys: [k], radio: true, when: `${view3D} && viewportFocus`, reason: "Needs a structure",
    checked: () => S.style() === v, run: () => S.setStyle(v) }));
  register({ id: "view.explorer", title: "Explorer", category: "View", keys: ["Mod+B"], icon: "panel-left",
    checked: () => S.panelShown("explorer"), run: () => S.togglePanel("explorer") });
  register({ id: "view.inspector", title: "Inspector", category: "View", keys: ["Mod+Alt+B"], icon: "panel-right",
    checked: () => S.panelShown("inspector"), run: () => S.togglePanel("inspector") });
  register({ id: "view.inspectorDrawer", title: "Inspector Drawer", category: "View", keys: ["Mod+Alt+I"], icon: "panel-right",
    desc: "Below 1100px the inspector opens over the viewport", run: () => S.togglePanel("inspector") });
  // the Inspector's (or Calculation Setup's) sections at once; Shift-click (or Alt-click) on a section head does the same.
  // With one section there is nothing to do: disabled, with the reason
  const manySections = { enabled: () => S.manySections(), reason: "The Inspector has one section here" };
  register({ id: "view.expandSections", title: "Expand All Sections", menuTitle: "Expand All", category: "View", keys: ["Mod+Alt+]"],
    ...manySections, icon: "chevrons-up-down", aliases: ["expand all", "open all sections", "unfold"],
    desc: "Every section of the Inspector, or of Calculation Setup while it is open", run: () => S.setAllSections(true) });
  register({ id: "view.collapseSections", title: "Collapse All Sections", menuTitle: "Collapse All", category: "View", keys: ["Mod+Alt+["],
    ...manySections, icon: "chevrons-down-up", aliases: ["collapse all", "close all sections", "fold"],
    desc: "Every section of the Inspector, or of Calculation Setup while it is open", run: () => S.setAllSections(false) });
  // the Explorer's sections (sidebar.js): move one (from its header's menu), show or hide each, reset the layout
  const secArgs = (d) => ({ palette: false, reason: "Right-click a section heading in the Explorer",
    enabled: (c) => !!(c.args && c.args.section) && !!S.sidebar() && S.sidebar().canMove(c.args.section, d) });
  register({ id: "view.sidebar.moveUp", title: "Move Section Up", menuTitle: "Move Up", category: "View", ...secArgs(-1),
    run: (c, a) => S.sidebar().move(a.section, -1) });
  register({ id: "view.sidebar.moveDown", title: "Move Section Down", menuTitle: "Move Down", category: "View", ...secArgs(1),
    run: (c, a) => S.sidebar().move(a.section, 1) });
  for (const [k, t] of [["editors", "Open Editors"], ["files", "Files"], ["db", "Materials.db"], ["recent", "Recent"]]) {
    register({ id: `view.sidebar.show.${k}`, title: `Explorer: ${t} Section`, menuTitle: t, category: "View",
      aliases: [`show ${t.toLowerCase()}`, `hide ${t.toLowerCase()}`, "sidebar section"],
      checked: () => !!S.sidebar() && S.sidebar().shown(k), run: () => S.sidebar().setShown(k, !S.sidebar().shown(k)) });
  }
  register({ id: "view.resetSidebar", title: "Reset Sidebar Layout", category: "View", icon: "rotate-ccw",
    aliases: ["reset explorer", "explorer sections", "sidebar sections", "restore sidebar"],
    desc: "The Explorer's sections back in their order, all shown, at their default heights", run: () => S.sidebar().reset() });
  register({ id: "view.bottom", title: "Bottom Panel", category: "View", keys: ["Mod+J"], icon: "panel-bottom",
    checked: () => S.panelShown("bottom"), run: () => S.togglePanel("bottom") });
  // two editor groups side by side (plan E11, groups.js)
  register({ id: "view.splitRight", title: "Split Editor Right", category: "View", keys: ["Mod+\\"], icon: "columns-2",
    aliases: ["split", "side by side", "two groups", "compare side by side"],
    desc: "A second editor group to the right, with the active document", run: () => S.groups().splitRight() });
  register({ id: "view.moveToOtherGroup", title: "Move Document to Other Group", category: "View", keys: ["Mod+Shift+\\"],
    icon: "move", enabled: (c) => c.hasDoc || !!c.splitOpen, reason: "Open a document first",
    run: () => S.groups().moveToOther() });
  register({ id: "view.focusOtherGroup", title: "Focus Other Group", category: "View", keys: ["Mod+Alt+\\"],
    enabled: (c) => !!c.splitOpen, reason: "There is only one editor group", run: () => S.groups().focusOther() });
  // Ctrl+1 and Ctrl+2 (VS Code's keys) are the browser's own tabs: these have no default key
  register({ id: "view.focusLeftGroup", title: "Focus Left Group", category: "View", aliases: ["first group", "group 1"],
    enabled: (c) => !!c.splitOpen, reason: "There is only one editor group", run: () => S.groups().focusLeft() });
  register({ id: "view.focusRightGroup", title: "Focus Right Group", category: "View", aliases: ["second group", "group 2"],
    enabled: (c) => !!c.splitOpen, reason: "There is only one editor group", run: () => S.groups().focusRight() });
  register({ id: "view.closeGroup", title: "Close Group", category: "View", icon: "x",
    enabled: (c) => !!c.splitOpen, reason: "There is only one editor group", run: () => S.groups().closeGroup() });
  register({ id: "view.linkCameras", title: "Link Cameras", category: "View", icon: "link-2",
    enabled: (c) => !!c.splitOpen, reason: "Needs two editor groups (Split Editor Right)",
    checked: () => !!(S.groups() && S.groups().linked && S.groups().linked()),
    desc: "Two 3D views turn together; each keeps its own zoom", run: () => S.groups().toggleLink() });
  // the three panels at once; the same command brings back exactly the ones that were open
  register({ id: "view.hideAllPanels", title: "Hide All Panels", category: "View", keys: ["Mod+Shift+F"], icon: "maximize",
    aliases: ["focus mode", "show panels", "hide panels", "maximize editor"], tipTitle: () => (S.allPanelsHidden() ? "Show Panels" : "Hide All Panels"),
    checked: () => S.allPanelsHidden(), desc: "Hides the Explorer, the Inspector and the bottom panel; again brings back the ones that were open",
    run: () => S.toggleAllPanels() });
  register({ id: "view.zen", title: "Zen Mode", category: "View", keys: ["Mod+Alt+Z"], icon: "maximize-2",
    checked: () => S.panelShown("zen"), run: () => S.togglePanel("zen") });
  // the Zen chip, the viewport menu in Zen, and Esc when nothing else had anything to do
  register({ id: "view.exitZen", title: "Exit Zen Mode", category: "View", icon: "minimize-2", when: "zenMode",
    reason: "Not in Zen Mode", run: () => { if (S.panelShown("zen")) S.togglePanel("zen"); } });
  // every theme (js/themes.js); moving onto one in the command search shows it until Enter or Esc
  THEMES.forEach(({ id: v, label: t, icon, desc, aliases = [] }) => register({ id: `view.theme.${v}`,
    title: `Theme: ${t}`, menuTitle: t, category: "View", radio: true, icon, desc, aliases: ["color theme", "appearance", ...aliases],
    checked: () => S.setting("theme") === v, run: () => S.set("theme", v),
    preview: () => S.previewTheme(v), unpreview: () => S.previewTheme(null) }));
  register({ id: "view.themePicker", title: "Color Theme…", category: "View", icon: "sun-moon",
    aliases: ["theme", "appearance", "high contrast", "colorblind", "sepia", "paper", "black", "oled", "editor theme",
      "dracula", "nord", "solarized", "gruvbox", "catppuccin", "tokyo night", "one dark"],
    desc: "Pick a theme; the arrow keys preview each one", run: () => S.pickTheme() });
  register({ id: "view.elementColors", title: "Element Colors…", category: "View", icon: "atom",
    aliases: ["palette", "color scheme", "atom colors", "cpk", "jmol", "vesta", "colorblind"],
    desc: "Edit, save and share the colors atoms are drawn in", run: () => S.elementColors() });
  register({ id: "view.themeCycle", title: "Cycle Theme", category: "View", keys: ["Mod+Alt+T"], icon: "sun-moon",
    tipTitle: () => `Theme: ${S.setting("theme")}`, run: () => S.cycleTheme() });
  register({ id: "view.zoomIn", title: "Zoom UI In", menuTitle: "In", category: "View", keys: ["Mod+="], icon: "zoom-in", run: () => S.zoomUi(10) });
  register({ id: "view.zoomOut", title: "Zoom UI Out", menuTitle: "Out", category: "View", keys: ["Mod+-"], icon: "zoom-out", run: () => S.zoomUi(-10) });
  register({ id: "view.zoomReset", title: "Reset UI Zoom", menuTitle: "Reset", category: "View", keys: ["Mod+0"], run: () => S.zoomUi(0) });
  register({ id: "view.focusExplorer", title: "Focus Explorer", category: "View", keys: ["Mod+Shift+E"], icon: "files",
    run: () => S.focusRegion("explorer") });
  // Mod+1 (the spec's key) is Ctrl+1, browser tab 1, on the reserved list: the spec's own fallback
  register({ id: "view.focusViewport", title: "Focus Viewport", category: "View", keys: ["Mod+Alt+1"], run: () => S.focusRegion("viewport") });
  register({ id: "view.focusInspector", title: "Focus Inspector", category: "View", keys: ["Mod+Shift+I"], run: () => S.focusRegion("inspector") });
  register({ id: "view.database", title: "Database View", category: "View", keys: ["Mod+Shift+D"], icon: "database",
    when: "hasDb", reason: "No materials.db in this folder", run: () => S.openDb() });
  register({ id: "view.tasks", title: "Tasks", category: "View", keys: ["Mod+Shift+Y"], icon: "list-checks", run: () => S.showBottom("tasks") });
  register({ id: "view.cluster", title: "Cluster", category: "View", icon: "activity",
    aliases: ["slurm", "nodes", "quota", "queue", "partitions", "limits", "hpc"], run: () => S.openCluster() });
  register({ id: "view.workflows", title: "Workflows", category: "View", icon: "layers",
    aliases: ["workflow", "progression", "pipeline", "chain", "steps", "mace workflow --status"],
    desc: "Workflows under this folder: every step's state, its job and the time left", run: () => S.openWorkflows() });
  register({ id: "cluster.jobDetails", title: "Show Job Details…", category: "View", icon: "activity",
    aliases: ["job", "slurm job", "progress", "eta"], run: () => S.jobDetailsPrompt() });
  register({ id: "view.jobs", title: "Jobs", category: "View", icon: "clock", aliases: ["squeue", "slurm", "queue", "my jobs"],
    desc: "Your SLURM jobs: state, output, cancel", run: () => S.showBottom("jobs") });

  /* ---------------- Go to */
  register({ id: "nav.goToAtom", title: "Go to Atom…", category: "View", keys: ["Mod+G"], icon: "atom",
    aliases: ["jump to atom", "find atom", "index", "label"], ...needsStructure, run: () => S.palette("@") });

  /* ---------------- Trajectory (spec §4.2): the Timeline's step slider */
  const traj = { when: "trajSteps >= 2", reason: "Needs an optimization with steps" };
  register({ id: "traj.prev", title: "Previous Step", category: "Trajectory", keys: [","], icon: "chevron-left", ...traj, run: () => S.trajStep(-1) });
  register({ id: "traj.next", title: "Next Step", category: "Trajectory", keys: ["."], icon: "chevron-right", ...traj, run: () => S.trajStep(1) });
  register({ id: "traj.first", title: "First Step", category: "Trajectory", keys: ["Shift+,"], ...traj, run: () => S.trajStep(-Infinity) });
  register({ id: "traj.last", title: "Last Step", category: "Trajectory", keys: ["Shift+."], ...traj, run: () => S.trajStep(Infinity) });
  // Space on a focused button presses the button, so Play waits for the viewport or nothing focused
  register({ id: "traj.play", title: "Play / Pause", category: "Trajectory", keys: ["Space"], icon: "play",
    when: "trajSteps >= 2 && !controlFocus", reason: "Needs an optimization with steps", checked: () => S.trajPlaying(),
    run: () => S.trajPlay() });
  register({ id: "traj.overlay", title: "Next Comparison Look", aliases: ["overlay", "side by side", "two-tone", "displacement map"], category: "Trajectory", keys: ["Mod+Alt+O"], icon: "layers",
    when: "docKind == trajectory", reason: "Needs an optimization", run: () => S.trajLayout() });
  register({ id: "traj.editStep", title: "Edit This Step", category: "Trajectory", icon: "move",
    when: "docKind == trajectory", reason: "Needs an optimization", desc: "The step on screen as a structure you can edit",
    run: () => S.editStep() });

  /* ---------------- Select */
  // Mod+A in a field selects its text, as usual
  register({ id: "select.all", title: "Select All", category: "Select", keys: ["Mod+A"], when: "docKind == structure && !inputFocus",
    reason: selecting.reason, run: ed("select.all") });
  register({ id: "select.none", title: "Select None", category: "Select", keys: ["Mod+Alt+A"], when: "docKind == structure && hasSelection",
    reason: (c) => (c.docKind !== "structure" ? selecting.reason(c) : "Nothing is selected"), run: ed("select.none") });
  register({ id: "select.invert", title: "Invert Selection", category: "Select", keys: ["Mod+I"], ...selecting, run: ed("select.invert") });
  register({ id: "select.byElement", title: "Select by Element…", category: "Select", keys: ["Mod+Alt+E"], icon: "atom", ...selecting,
    run: ed("select.byElement") });
  register({ id: "select.equivalents", title: "Select Symmetry Equivalents", category: "Select", keys: ["Alt+Q"], icon: "symmetry",
    when: "docKind == structure && hasSelection", reason: "Select atoms first", run: ed("select.equivalents") });
  register({ id: "select.within", title: "Select Within Distance…", category: "Select", keys: ["Alt+R"],
    when: "docKind == structure && hasSelection", reason: "Select atoms first", run: () => S.lazyDialog("select_dialogs", "within") });
  register({ id: "select.neighbors", title: "Select Bonded Neighbors", category: "Select", keys: ["Mod+Alt+N"], icon: "bonds",
    when: "docKind == structure && hasSelection", reason: "Select atoms first", run: ed("select.neighbors") });
  register({ id: "select.grow", title: "Grow Selection by One Shell", category: "Select", keys: ["Mod+]"],
    when: "docKind == structure && hasSelection", reason: "Select atoms first", run: ed("select.grow") });
  register({ id: "select.shrink", title: "Shrink Selection by One Shell", category: "Select", keys: ["Mod+["],
    when: "docKind == structure && hasSelection", reason: "Select atoms first", run: ed("select.shrink") });
  register({ id: "select.byIndex", title: "Select by Index…", category: "Select", keys: ["Alt+I"], ...selecting,
    run: () => S.lazyDialog("select_dialogs", "byIndex") });
  register({ id: "select.byFrac", title: "Select by Fractional Range…", category: "Select", ...selecting, aliases: ["slab surface"],
    run: () => S.lazyDialog("select_dialogs", "byFrac") });
  register({ id: "select.saveSet", title: "Save Selection as Set…", category: "Select", icon: "tag",
    when: "docKind == structure && hasSelection", reason: "Select atoms first", run: () => S.lazyDialog("select_dialogs", "saveSet") });
  register({ id: "select.sets", title: "Apply Selection Set…", category: "Select", when: "docKind == structure && hasSets",
    reason: "No selection sets in this document", run: () => S.openSets() });
  // Esc: cancel a drag, else leave the tool for Select, else select nothing (spec §4.2)
  register({ id: "select.exitTool", title: "Exit Tool / Select None", category: "Select", keys: ["Esc"], icon: "mouse-pointer-2",
    when: "toolNotSelect || hasSelection || dragging", reason: "Nothing to cancel", run: () => S.escape() });

  /* ---------------- Build (Phases 3-4) */
  register({ id: "build.addAtom", title: "Add Atom", category: "Build", icon: "add-atom", ...editable(),
    desc: "The Add Atom tool (A)", run: () => S.run("tool.add") });
  register({ id: "build.replace", title: "Replace Element…", category: "Build", icon: "replace-atom", ...editable(),
    desc: "With a selection: pick the element; without: the Replace tool", run: ed("edit.replaceElement") });
  register({ id: "build.delete", title: "Delete Atoms", category: "Build", icon: "trash-2", ...editable(" && hasSelection"),
    run: ed("build.delete") });
  // Phase 4 builders: dialogs with a live preview (js/lazy/*_dialog.js on the shared frame)
  const builderDoc = { when: "docKind == structure", reason: (c) => (c.docKind === "trajectory" ? "Trajectories are view-only. Use Edit This Step" : "Needs a structure") };
  register({ id: "build.supercell", title: "Supercell…", category: "Build", keys: ["Mod+Alt+P"], icon: "supercell", ...builderDoc,
    aliases: ["repeat", "replicate", "2x2x2"], desc: "n1×n2×n3 or a matrix; a new tab or in place", run: (c, a) => S.builder("supercell_dialog", "open", a && a.matrix ? a : null) });
  register({ id: "build.slab", title: "Slab from Bulk…", category: "Build", keys: ["Mod+Alt+H"], icon: "slab", ...builderDoc,
    aliases: ["surface", "hkl", "miller", "vacuum"], desc: "Atomic layers along (hkl), with vacuum; always a new tab", run: () => S.builder("slab_dialog", "open") });
  ["Vacancy", "Substitution", "Interstitial", "Antisite"].forEach((d) => register({ id: `build.defect.${d.toLowerCase()}`, title: `${d}…`,
    category: "Build: Point Defect", icon: "atom", ...builderDoc, aliases: ["point defect", "defect"],
    desc: "In a supercell sized so defect images are at least Settings › Defect image distance apart",
    run: () => S.builder("defect_dialog", "open", { tab: d.toLowerCase() }) }));
  register({ id: "build.transform", title: "Transform Cell…", category: "Build", icon: "unit-cell", ...builderDoc,
    aliases: ["primitive", "conventional", "niggli", "standardize"], desc: "Primitive, conventional or Niggli-reduced cell",
    run: (c, a) => S.builder("transform_dialog", "open", a || {}) });
  register({ id: "build.symmetrize", title: "Symmetrize…", category: "Build", icon: "symmetry", ...builderDoc,
    aliases: ["refine", "tolerance", "symprec"], desc: "The space group at each tolerance; refine at the one you pick",
    run: () => S.builder("symmetrize_dialog", "open") });
  register({ id: "build.clean", title: "Clean Geometry…", category: "Build", keys: ["Mod+Alt+G"], icon: "sparkles",
    when: "docEditable", reason: (c) => S.editReason(c), aliases: ["tidy", "relax", "springs", "gfnff", "ml potential", "pre-optimize"],
    desc: "Pre-relax the structure (springs, GFN-FF or an ML potential) to preview, then accept or discard",
    run: () => S.builder("clean_panel", "open") });
  register({ id: "build.symMode", title: "Symmetry-Aware Editing", category: "Build", keys: ["Mod+Alt+Y"], icon: "symmetry",
    ...editable(), desc: "Edits act on whole symmetry orbits", checked: () => S.symMode() === "sym", run: ed("build.symMode") });
  register({ id: "build.symModeP1", title: "Symmetry Mode: P1", menuTitle: "P1", category: "Build", radio: true, ...editable(),
    desc: "Every atom is independent", checked: () => S.symMode() === "p1", run: ed("build.symModeP1") });
  register({ id: "build.symModeSym", title: "Symmetry Mode: Sym", menuTitle: "Sym", category: "Build", radio: true, ...editable(),
    desc: "Edits act on whole symmetry orbits", checked: () => S.symMode() === "sym", run: ed("build.symModeSym") });
  register({ id: "build.editStructure", title: "Start Editing", category: "Build", icon: "move", aliases: ["open in structure editor"],
    ...editable(), desc: "Edit this structure in place (the first edit does this too)", run: () => S.openEditor() });

  /* ---------------- Tools */
  const TOOLS = [["select", "Select", "V", "mouse-pointer-2", "Click an atom; Shift-click or drag to add"],
    ["lasso", "Lasso", "L", "lasso-select"], ["orbit", "Orbit", "O", "rotate-3d", "Drag to rotate"],
    ["pan", "Pan", "H", "hand"], ["zoom", "Zoom", "Z", "zoom-in"], ["move", "Move", "G", "move"],
    ["add", "Add Atom", "A", "add-atom"], ["replace", "Replace", "E", "replace-atom"],
    ["measure", "Measure", "M", "ruler", "2 atoms: distance, 3: angle, 4: dihedral"], ["bond", "Bond", "B", "link-2"]];
  const EDIT_TOOLS = new Set(["move", "add", "replace", "bond", "measure"]);
  TOOLS.forEach(([t, title, key, ic, desc]) => register({ id: `tool.${t}`, title: `${title} Tool`, menuTitle: title,
    category: "Tools", keys: [key], icon: ic, desc, radio: true,
    when: t === "replace" ? "!hasSelection && tool != add" : t === "measure" ? "docKind == structure" : EDIT_TOOLS.has(t) ? "docEditable" : undefined,
    reason: EDIT_TOOLS.has(t) ? (c) => S.editReason(c) || "Select atoms first" : undefined,
    checked: () => S.tool() === t, run: t === "move" ? ed("tool.move") : () => S.setTool(t) }));
  register({ id: "tool.addElement", title: "Add Atom: Element…", category: "Tools", keys: ["E"], icon: "atom",
    when: "tool == add", reason: "Needs the Add Atom tool", desc: "Type the element to add; Enter goes back to the viewport",
    run: ed("tool.addElement") });
  register({ id: "tool.measureSelection", title: "Measure Current Selection", category: "Tools", keys: ["Mod+M"], icon: "ruler",
    when: "docKind == structure && selectionCount >= 2", reason: "Select 2, 3 or 4 atoms", run: ed("tool.measureSelection") });
  register({ id: "tool.clearMeasurements", title: "Clear Measurements", category: "Tools", keys: ["Mod+Alt+M"],
    when: "docKind == structure && hasMeasurements", reason: "No measurements", run: ed("tool.clearMeasurements") });
  const coordDoc = { when: "docKind == structure", reason: "Needs a structure" };
  register({ id: "tool.coordination", title: "Coordination Analysis", category: "Tools", icon: "hexagon", ...coordDoc,
    aliases: ["cn", "coordination number", "econ", "neighbors", "shells"], desc: "CN, ECoN and neighbor shells of the selection, or of each unique site",
    run: () => S.showCoordination() });
  register({ id: "tool.compare", title: "Compare Two Structures…", category: "Tools", icon: "layers",
    run: () => S.compareStructures() });

  /* ---------------- Calculation (plan B4, B11): js/setup/panel.js */
  const inSetup = { when: "setupOpen", reason: "Open Calculation Setup first" };
  register({ id: "calc.setup", title: "Calculation Setup…", category: "Calculation", keys: ["Mod+Shift+R"], icon: "flask-conical",
    aliases: ["deck", "d12", "cif2d12", "opt2d12", "opt2d3", "band", "doss", "freq"], checked: () => S.setupOpen(),
    desc: "The CLI's own questions as a form; the deck it writes, previewed", run: (c, a) => S.toggleSetup(a && a.open ? true : undefined) });
  register({ id: "calc.preview", title: "Preview Deck", category: "Calculation", keys: ["Mod+Enter"], icon: "file-text",
    ...inSetup, desc: "The deck the printed command writes from these settings", run: () => S.previewDeck() });
  register({ id: "calc.saveJson", title: "Save Settings JSON…", category: "Calculation", keys: ["Mod+Alt+J"], icon: "save",
    ...inSetup, run: () => S.saveJson() });
  register({ id: "calc.preflight", title: "Run Preflight (TESTPDIM)", category: "Calculation", icon: "list-checks",
    when: "setupOpen && setupHasDeck", reason: (c) => (!c.setupOpen ? "Open Calculation Setup first" : "Preview a deck first"),
    aliases: ["testpdim", "check deck"], run: () => S.preflight() });
  register({ id: "calc.copyCli", title: "Copy CLI Command", category: "Calculation", keys: ["Mod+Alt+C"], icon: "terminal",
    ...inSetup, aliases: ["command line", "mace convert", "mace opt2d12"], run: () => S.copyCli() });
  register({ id: "calc.exampleConfig", title: "Open Example Config…", category: "Calculation", icon: "file-json",
    ...inSetup, aliases: ["example", "template", "standard_dft_opt"], desc: "One of the CLI's example configs as the settings",
    run: () => S.palette("") });
  register({ id: "calc.autoPreview", title: "Preview While Editing", category: "Calculation", aliases: ["auto preview", "live"],
    checked: () => S.setting("setupAutoPreview") !== false, run: () => S.set("setupAutoPreview", S.setting("setupAutoPreview") === false) });
  register({ id: "calc.resetSettings", title: "Reset All Settings", category: "Calculation", icon: "rotate-ccw", ...inSetup,
    aliases: ["defaults", "parent's settings"], desc: "Back to the CLI's defaults (a follow-up: the parent's), one undo step", run: () => S.resetSettings() });
  // Submit, Run on This Node and Batch Setup (PLAN_submit T10): each opens its dialog or document;
  // nothing is submitted without the dialog's own Submit click, and no default shortcut is bound
  const deckArgs = { enabled: (c) => S.submitTargets(c.args).length > 0, reason: "Pick a deck (.d12 or .d3) or a folder of decks" };
  register({ id: "calc.submit", title: "Submit…", category: "Calculation", icon: "play", ...deckArgs,
    aliases: ["sbatch", "mace submit", "slurm", "job array"], desc: "mace submit, previewed exactly; you confirm before anything is submitted",
    run: (c, a) => S.submit(a) });
  register({ id: "calc.runHere", title: "Run on This Node…", category: "Calculation", icon: "terminal", ...deckArgs,
    aliases: ["mace run", "dev node", "local run"], desc: "mace run on this node (a dev node), previewed first", run: (c, a) => S.runHere(a) });
  register({ id: "calc.newWorkflow", title: "New Workflow…", category: "Calculation", icon: "boxes",
    aliases: ["workflow", "chain", "pipeline", "progression", "mace workflow --spec", "opt sp band doss"],
    desc: "A chain of calculations for many structures (OPT → SP → BAND → DOSS …), previewed exactly; you confirm before anything is submitted",
    run: (c, a) => S.newWorkflow(a) });
  register({ id: "calc.batch", title: "Batch Setup…", category: "Calculation", icon: "list-checks",
    aliases: ["batch", "many decks", "follow-ups", "commands.sh"], desc: "Decks for many CIFs or follow-ups at once, through the CLI's batch routes",
    run: (c, a) => S.batch(a) });

  /* ---------------- Analysis */
  register({ id: "analysis.band", title: "Plot Band Structure", category: "Analysis", icon: "chart-line",
    when: "hasBand", reason: "No BAND.DAT in this folder", run: () => S.openFirst("band") });
  register({ id: "analysis.dos", title: "Plot Density of States", category: "Analysis", icon: "activity",
    when: "hasDos", reason: "No DOSS.DAT in this folder", run: () => S.openFirst("dos") });
  register({ id: "analysis.ir", title: "IR Spectrum", category: "Analysis", icon: "activity",
    when: "hasIr || docFileKind == out && docHasIr || docIsFreq", reason: "No IRSPEC.DAT beside this output or in this folder, and no FREQ output open",
    aliases: ["ir", "infrared", "irspec"], run: () => S.openSpectrum("ir") });
  register({ id: "analysis.raman", title: "Raman Spectrum", category: "Analysis", icon: "activity",
    when: "hasRaman || docFileKind == out && docHasRaman || docIsFreq", reason: "No RAMSPEC.DAT beside this output or in this folder, and no FREQ output open",
    aliases: ["raman", "ramspec", "polarized"], run: () => S.openSpectrum("raman") });
  register({ id: "analysis.modes", title: "Vibrational Modes", category: "Analysis", icon: "waves",
    when: "docFileKind == out && docIsFreq", reason: "Needs a FREQ .out with normal modes",
    aliases: ["freq", "phonon", "normal modes", "animate", "imaginary"], run: () => S.openModes() });
  // volumes (plan E5): a density colored by a potential (MEP), and differences of grids
  register({ id: "analysis.volumeMep", title: "Color Density by Potential…", category: "Analysis", icon: "sun",
    when: "docKind == volume && hasTwoCubes", reason: "Open a density volume, with a potential cube in the folder",
    aliases: ["mep", "electrostatic potential", "isosurface", "map potential"], run: () => S.colorByPotential() });
  register({ id: "analysis.volumeDiff", title: "Volume Difference…", category: "Analysis", icon: "minus",
    when: "hasTwoCubes", reason: "Needs two cube files in this folder",
    aliases: ["density difference", "deformation density", "subtract cubes", "orbital", "wannier"], run: () => S.volumeDifference() });
  register({ id: "analysis.trajectory", title: "Optimization Trajectory", category: "Analysis", icon: "history",
    when: "docKind == trajectory|structure && docFileKind == out", reason: "Needs an open .out",
    run: () => S.reopenActive() });
  register({ id: "analysis.compareGroups", title: "Compare the Two Groups' Structures", category: "Analysis", icon: "layers",
    enabled: (c) => !!c.splitOpen && c.docHasPath && (c.docKind === "structure" || c.docKind === "trajectory") && !!S.otherStructurePath(),
    reason: "Needs a structure in each editor group", aliases: ["compare structures", "side by side"],
    run: () => S.compareGroups() });
  register({ id: "analysis.compareInput", title: "Compare with Input", category: "Analysis", icon: "layers",
    when: "docFileKind == out", reason: "Needs an open .out", aliases: ["before after", "initial", "input deck", "relaxation"],
    desc: "The output against the .d12 it ran from (or its parent CIF, or its first frame)", run: (c, a) => S.compareInput(a && a.path) });
  const cmpDoc = { when: "docKind == trajectory", reason: "Needs an optimization or a comparison" };
  // Run Analysis (PLAN_analysis C4): an optimization's energy, convergence, SCF, cell and tracked measurements
  const runDoc = { when: "docFileKind == out", reason: "Needs an open optimization output (.out)" };
  register({ id: "analysis.energy", title: "Energy Along the Run", category: "Analysis", icon: "chart-line",
    when: "docKind == trajectory || docKind == runplot", reason: "Needs an optimization or a comparison",
    run: () => (S.runPathOf() ? S.openRunPlot(null, "energy") : S.compareSub("energy")) });
  register({ id: "analysis.runPlot", title: "Run Analysis", category: "Analysis", icon: "chart-line", ...runDoc,
    aliases: ["convergence", "energy", "scf", "optimization plot"], desc: "Energy, convergence, SCF and cell along an optimization",
    run: () => S.openRunPlot(null, "energy") });
  register({ id: "analysis.convergence", title: "Convergence Criteria", category: "Analysis", icon: "activity", ...runDoc,
    aliases: ["toldeg", "toldex", "gradient", "displacement"], run: () => S.openRunPlot(null, "convergence") });
  register({ id: "analysis.scf", title: "SCF Convergence", category: "Analysis", icon: "activity", ...runDoc,
    aliases: ["scf cycles", "toldee"], run: () => S.openRunPlot(null, "scf") });
  register({ id: "analysis.cellRun", title: "Cell Along the Run", category: "Analysis", icon: "box", ...runDoc,
    aliases: ["lattice", "volume"], run: () => S.openRunPlot(null, "cell") });
  register({ id: "analysis.trackMeasurement", title: "Track Measurement Along the Run", category: "Analysis", icon: "ruler", ...runDoc,
    aliases: ["track bond", "distance along"], desc: "The Tracked tab: measure on the run (Measurements), then Track Along the Run",
    run: () => S.openRunPlot(null, "tracked") });
  register({ id: "analysis.compareRuns", title: "Compare Runs…", category: "Analysis", icon: "layers", ...runDoc,
    aliases: ["overlay", "two runs"], run: () => S.compareRuns() });
  register({ id: "analysis.bondTable", title: "Bond Length Table", category: "Analysis", icon: "table", ...cmpDoc,
    run: () => S.compareSub("bonds") });
  register({ id: "analysis.displacements", title: "Displacement Table", category: "Analysis", icon: "table", ...cmpDoc,
    run: () => S.compareSub("disp") });
  register({ id: "analysis.coordination", title: "Coordination Analysis", category: "Analysis", icon: "hexagon", when: "docKind == structure", reason: "Needs a structure",
    aliases: ["cn", "coordination number", "econ"], palette: false, run: () => S.showCoordination() });
  register({ id: "analysis.resetPlot", title: "Reset Plot Options", category: "Analysis", icon: "rotate-ccw",
    when: "docKind == plot", reason: "Open a plot first", aliases: ["reset plot"], run: () => S.resetPlot() });
  // Plot Gallery (1.1.6): many outputs' plots of one kind in panes, as `mace plotting --gallery`
  register({ id: "analysis.gallery", title: "Plot Gallery…", category: "Analysis", icon: "grid-3x3",
    aliases: ["gallery", "many plots", "grid of plots", "band structures", "compare plots"],
    desc: "Many outputs' band, DOS, spectra, optimization, SCF or structure plots in one grid, with a caption table",
    run: () => S.openGallery() });
  register({ id: "analysis.db", title: "Open Materials Database", category: "Analysis", icon: "database",
    when: "hasDb", reason: "No materials.db in this folder", run: () => S.openDb() });
  // Create or Update Database from a Folder (PLAN_analysis C10): the CLI's --extract-properties, with a preview and a backup
  register({ id: "db.build", title: "Create or Update Database…", category: "Analysis", icon: "database",
    aliases: ["build database", "update database", "extract properties", "materials.db"],
    desc: "Read a folder's outputs into materials.db (mace analyze --extract-properties), with a preview and a backup first",
    run: (c, a) => S.dbBuild(a || {}) });
  // the active table: the database's filtered rows, a comparison's, the FREQ modes, the coordination shells
  register({ id: "analysis.exportCsv", title: "Export Table as CSV", menuTitle: "CSV", category: "Analysis: Export Table", icon: "download",
    when: "activeTable", reason: "No table is shown", aliases: ["export table", "csv"], run: () => S.exportTable("csv") });
  register({ id: "analysis.exportJson", title: "Export Table as JSON", menuTitle: "JSON", category: "Analysis: Export Table", icon: "download",
    when: "activeTable", reason: "No table is shown", aliases: ["export table", "json"], run: () => S.exportTable("json") });
  // a database table also in the CLI's other formats (mace database --action export --format excel|latex|html)
  for (const [fmt, label] of [["excel", "Excel"], ["latex", "LaTeX"], ["html", "HTML"]]) {
    register({ id: `analysis.export${label}`, title: `Export Table as ${label}`, menuTitle: label, category: "Analysis: Export Table", icon: "download",
      when: "activeDbTable", reason: "Only a database table exports as " + label, aliases: ["export table", fmt], run: () => S.exportTable(fmt) });
  }

  /* ---------------- Help */
  register({ id: "help.shortcuts", title: "Keyboard Shortcuts", category: "Help", keys: ["?", "Mod+/"], icon: "keyboard",
    aliases: ["cheat sheet", "keys", "hotkeys"], run: () => S.cheatSheet() });
  register({ id: "help.commands", title: "Show All Commands", category: "Help", icon: "command", run: () => S.palette("") });
  register({ id: "help.cluster", title: "Cluster Status and Job Progress", category: "Help", icon: "book-open",
    aliases: ["slurm", "quota", "mace cluster"], run: () => S.clusterDocs() });
  register({ id: "help.docs", title: "Documentation", category: "Help", icon: "book-open",
    aliases: ["help", "guide", "manual", "docs/gui.md"], run: () => S.docsDialog() });
  register({ id: "help.tunnel", title: "Connecting from Your Computer", category: "Help", icon: "terminal",
    aliases: ["ssh", "tunnel", "port forwarding", "mace gui connect", "vs code", "remote"], run: () => S.tunnelDialog() });
  register({ id: "help.copyDiagnostics", title: "Copy Diagnostics", category: "Help", icon: "copy",
    aliases: ["bug report", "debug info", "versions"], desc: "Versions, OS, browser and recent errors, with paths, file names and tokens removed",
    run: () => S.copyDiagnostics() });
  register({ id: "help.report", title: "Report a Problem…", category: "Help", icon: "info",
    aliases: ["bug", "issue", "feedback", "diagnostics"], run: () => S.reportProblem() });
  register({ id: "help.checkUpdates", title: "Check for Updates…", category: "Help", icon: "download",
    run: () => S.checkForUpdates() });
  register({ id: "help.about", title: "About MACE", category: "Help", icon: "info", run: () => S.aboutDialog() });
  // GTK's key for the menu bar: with Edit's and Build's Alt+letter given to
  // selection commands (spec §4.2), F10 then the arrows reach every menu
  register({ id: "menu.focusBar", title: "Focus Menu Bar", category: "Help", keys: ["F10"], icon: "menu",
    aliases: ["menus", "mnemonic"], run: () => S.focusMenuBar() });

  /* ---------------- Context menus (spec §2.2): tabs, explorer items, atoms, empty space */
  const tabArgs = { enabled: (c) => !!S.tabOf(c.args), reason: "No document is open" };
  register({ id: "tab.close", title: "Close", category: "Tab", icon: "x", ...tabArgs, run: (c, a) => S.closeDoc(S.tabOf(a)) });
  register({ id: "tab.moveToOtherGroup", title: "Move to Other Group", category: "Tab", icon: "move", ...tabArgs,
    run: (c, a) => S.groups().moveToOther(S.tabOf(a)) });
  // Shift+Left and Shift+Right on a focused tab do the same
  register({ id: "tab.moveLeft", title: "Move Tab Left", menuTitle: "Move Left", category: "Tab", enabled: (c) => S.canMoveTab(S.tabOf(c.args), -1),
    reason: "It is the first tab", run: (c, a) => S.moveTab(S.tabOf(a), -1) });
  register({ id: "tab.moveRight", title: "Move Tab Right", menuTitle: "Move Right", category: "Tab", enabled: (c) => S.canMoveTab(S.tabOf(c.args), 1),
    reason: "It is the last tab", run: (c, a) => S.moveTab(S.tabOf(a), 1) });
  register({ id: "tab.closeOthers", title: "Close Others", category: "Tab", enabled: (c) => !!S.tabOf(c.args) && c.docCount >= 2,
    reason: "No other documents are open", run: (c, a) => S.closeOthers(S.tabOf(a)) });
  register({ id: "tab.closeRight", title: "Close to the Right", category: "Tab", enabled: (c) => S.hasRight(S.tabOf(c.args)),
    reason: "No documents to the right", run: (c, a) => S.closeRight(S.tabOf(a)) });
  register({ id: "tab.copyPath", title: "Copy Path", category: "Tab", icon: "copy",
    enabled: (c) => !!(S.tabOf(c.args) && S.tabOf(c.args).path), reason: "This document has no file", run: (c, a) => S.copyText(S.tabOf(a).path, "Path copied") });
  register({ id: "tab.reveal", title: "Reveal in Explorer", category: "Tab", icon: "files",
    enabled: (c) => !!(S.tabOf(c.args) && S.tabOf(c.args).path), reason: "This document has no file", run: (c, a) => S.reveal(S.tabOf(a).path) });

  const fileArgs = { enabled: (c) => !!S.pathOf(c.args), reason: "Pick a file in the Explorer" };
  register({ id: "explorer.open", title: "Open", category: "Explorer", icon: "file", ...fileArgs, run: (c, a) => S.openPath(S.pathOf(a)) });
  register({ id: "explorer.openToSide", title: "Open to the Side", category: "Explorer", icon: "columns-2", ...fileArgs,
    desc: "Open the file in the right editor group (split the editor if needed)",
    run: (c, a) => S.groups().openToSide(S.pathOf(a)) });
  register({ id: "explorer.openAsText", title: "Open as Text", category: "Explorer", icon: "file-text", ...fileArgs,
    enabled: (c) => !!S.pathOf(c.args) && S.isText(S.pathOf(c.args)), reason: "Not a text file",
    run: (c, a) => S.openAsText(S.pathOf(a)) });
  register({ id: "explorer.compareWithActive", title: "Compare With Active", category: "Explorer", icon: "layers",
    enabled: (c) => S.canCompare(S.pathOf(c.args)), reason: "Needs an open structure and another structure file",
    run: (c, a) => S.compareWith(S.pathOf(a)) });
  register({ id: "explorer.plot", title: "Plot", category: "Explorer", icon: "chart-line",
    enabled: (c) => S.canPlot(S.pathOf(c.args)), reason: "Only BAND, DOSS, IR/Raman spectra and FREQ outputs plot",
    run: (c, a) => S.plotFile(S.pathOf(a)) });
  register({ id: "explorer.compareConvergence", title: "Compare Convergence", category: "Explorer", icon: "chart-spline",
    enabled: (c) => S.convergencePaths(S.pathOf(c.args)).length >= 2, reason: "Ctrl-click two or more CRYSTAL outputs in the Explorer",
    aliases: ["compare runs", "convergence", "energy per atom"], run: (c, a) => S.compareConvergence(S.convergencePaths(S.pathOf(a))) });
  register({ id: "explorer.gallery", title: "Plot Gallery", category: "Explorer", icon: "grid-3x3",
    enabled: (c) => !!(c.args && (c.args.dir !== undefined || S.pathOf(c.args))), reason: "Right-click a folder or files in the Explorer",
    run: (c, a) => {
      if (a.dir !== undefined) return S.openGallery({ source: { type: "folder", folder: a.dir || "" }, label: a.dir || "This folder" });
      const p = S.pathOf(a), picked = [...new Set([...(window.MACE.state.selected || []), p])];
      if (picked.length >= 2) return S.openGallery({ source: { type: "paths", paths: picked }, label: `Explorer selection (${picked.length})` });
      // one file: its folder's plots of that kind
      const dir = p.includes("/") ? p.slice(0, p.lastIndexOf("/")) : "";
      return S.openGallery({ source: { type: "folder", folder: dir }, label: dir || "This folder" });
    } });
  register({ id: "explorer.copyPath", title: "Copy Path", category: "Explorer", icon: "copy", ...fileArgs,
    run: (c, a) => S.copyText(S.pathOf(a), "Path copied") });
  register({ id: "explorer.reveal", title: "Reveal in Folder", category: "Explorer", icon: "folder-open", ...fileArgs,
    run: (c, a) => S.reveal(S.pathOf(a)) });
  register({ id: "explorer.history", title: "Show File History…", category: "Explorer", icon: "history", ...fileArgs,
    run: (c, a) => S.fileHistory(S.pathOf(a)) });
  register({ id: "explorer.dbBuild", title: "Create or Update Database from This Folder…", category: "Explorer", icon: "database",
    enabled: (c) => !!(c.args && (c.args.dir !== undefined || S.pathOf(c.args))), reason: "Right-click a folder or a file in the Explorer",
    run: (c, a) => S.dbBuild({ folder: a.dir !== undefined ? a.dir : (S.pathOf(a).includes("/") ? S.pathOf(a).slice(0, S.pathOf(a).lastIndexOf("/")) : "") }) });
  register({ id: "file.history", title: "Show History", category: "File", icon: "history",
    enabled: (c) => !!(S.pathOf(c.args) || S.activePath()), reason: "Open a file, or pick one in the Explorer",
    run: (c, a) => S.fileHistory(S.pathOf(a) || S.activePath()) });
  const deckFile = { enabled: (c) => /\.(d12|d3)$/i.test(S.pathOf(c.args) || ""), reason: "Needs a .d12 or .d3 deck" };
  register({ id: "explorer.submit", title: "Submit…", category: "Explorer", icon: "play", ...deckFile, run: (c, a) => S.submit({ paths: [S.pathOf(a)] }) });
  register({ id: "explorer.runHere", title: "Run on This Node…", category: "Explorer", icon: "terminal", ...deckFile,
    run: (c, a) => S.runHere({ paths: [S.pathOf(a)] }) });
  const dirArgs = { enabled: (c) => !!(c.args && c.args.dir !== undefined), reason: "Right-click a folder in the Explorer" };
  register({ id: "explorer.batch", title: "Batch Setup…", category: "Explorer", icon: "list-checks", ...dirArgs, run: (c, a) => S.batch(a) });
  register({ id: "explorer.workflow", title: "New Workflow…", category: "Explorer", icon: "boxes",
    enabled: (c) => !!(c.args && (c.args.dir !== undefined || /\.(cif|d12)$/i.test(S.pathOf(c.args) || ""))),
    reason: "Right-click a CIF, a .d12 deck or a folder of them",
    run: (c, a) => S.newWorkflow(a.dir !== undefined ? { dir: a.dir } : { paths: [S.pathOf(a)] }) });
  register({ id: "explorer.submitFolder", title: "Submit Folder…", category: "Explorer", icon: "play", ...dirArgs, run: (c, a) => S.submit(a) });
  register({ id: "explorer.setup", title: "Set Up Calculation", category: "Explorer", icon: "flask-conical",
    enabled: (c) => ["cif", "out", "d12"].includes(S.kindOf(S.pathOf(c.args))), reason: "Needs a CIF, an .out or a .d12",
    run: (c, a) => S.setupFor(S.pathOf(a)) });

  const atomArgs = { enabled: (c) => !!S.atomOf(c.args), reason: "Right-click an atom" };
  const atomSel = { enabled: (c) => !!S.atomOf(c.args) && c.docKind === "structure",
    reason: (c) => (S.atomOf(c.args) ? "Trajectories are view-only. Use Edit This Step" : "Right-click an atom") };
  register({ id: "atom.selectEquivalents", title: "Select Equivalents", category: "Atom", icon: "symmetry", ...atomSel,
    run: ed("atom.selectEquivalents") });
  register({ id: "atom.selectElement", title: "Select Element", category: "Atom", icon: "atom", ...atomSel,
    menuTitle: (c) => { const at = S.atomOf(c.args); return at ? `Select Element (${at.symbol})` : "Select Element"; },
    run: ed("atom.selectElement") });
  register({ id: "atom.selectNeighbors", title: "Select Neighbors", category: "Atom", icon: "bonds", ...atomSel,
    run: ed("atom.selectNeighbors") });
  register({ id: "atom.measureFrom", title: "Measure from Here", category: "Atom", icon: "ruler", ...atomSel, run: ed("atom.measureFrom") });
  // Point Defect from an atom (plan T6): the dialog opens on that tab with this atom's site picked
  const atomOrSel = (c) => (S.atomOf(c.args) ? S.atomOf(c.args).i : c.selectionCount === 1 ? S.selectedAtom() : null);
  const defectHere = (tab) => (c, a) => S.builder("defect_dialog", "open", { tab, site: atomOrSel({ ...c, args: a }) });
  const atomDefect = { enabled: (c) => c.docKind === "structure" && atomOrSel(c) !== null,
    reason: (c) => (c.docKind === "trajectory" ? "Trajectories are view-only. Use Edit This Step" : "Right-click an atom or select one") };
  register({ id: "atom.vacancyHere", title: "Vacancy Here", category: "Point Defect", icon: "atom", ...atomDefect, run: defectHere("vacancy") });
  register({ id: "atom.substitute", title: "Substitute…", category: "Point Defect", icon: "replace-atom", ...atomDefect, run: defectHere("substitution") });
  register({ id: "atom.antisite", title: "Antisite…", category: "Point Defect", icon: "replace-atom",
    enabled: (c) => atomDefect.enabled(c) && S.elementCount() > 1,
    reason: (c) => (atomDefect.enabled(c) ? "Antisites need two elements" : atomDefect.reason(c)), run: defectHere("antisite") });
  register({ id: "atom.centerView", title: "Center View Here", category: "Atom", icon: "crosshair", ...atomArgs,
    run: (c, a) => S.centerOnAtom(S.atomOf(a)) });
  register({ id: "atom.copyFrac", title: "Copy Coordinates (Fractional)", category: "Atom", icon: "copy", ...atomArgs,
    run: (c, a) => S.copyAtom(S.atomOf(a), "frac") });
  register({ id: "atom.copyCart", title: "Copy Coordinates (Cartesian)", category: "Atom", icon: "copy", ...atomArgs,
    run: (c, a) => S.copyAtom(S.atomOf(a), "cart") });
  register({ id: "atom.properties", title: "Properties", category: "Atom", icon: "info", ...atomArgs,
    run: () => S.focusRegion("inspector") });
  register({ id: "viewport.addAtomHere", title: "Add Atom Here", category: "Viewport", icon: "add-atom", ...editable(),
    run: ed("viewport.addAtomHere") });
  // Shift+F10 (the ContextMenu key works natively): the menu of the focused row, tab or 3D view
  register({ id: "view.contextMenu", title: "Open Context Menu", category: "View", keys: ["Shift+F10"], icon: "menu",
    run: () => S.contextMenuForFocus() });
}

/* The atom and empty-space menus (spec §2.2), the explorer item and tab menus. */
export const CONTEXT = {
  atom: ["atom.selectEquivalents", "atom.selectElement", "atom.selectNeighbors", "-", "edit.replaceElement",
    "edit.setCoordinates", "edit.delete", "-", "atom.vacancyHere", "atom.substitute", "atom.antisite", "-",
    "atom.measureFrom", "atom.centerView", "-", "atom.copyFrac",
    "atom.copyCart", "-", "atom.properties"],
  empty: ["viewport.addAtomHere", "edit.paste", "select.all", "-",
    { label: "View Along", sub: ["view.alongA", "view.alongB", "view.alongC", "view.alongAStar", "view.alongBStar", "view.alongCStar", "view.alongUVW"] },
    "view.reset", "view.pivotCell", "view.pivotSelection",
    { label: "Display Range", sub: ["view.range1", "view.range2", "view.range3", "view.rangeCustom"] }, "-",
    "file.exportImage", "file.exportLegend"],
  explorer: ["explorer.open", "explorer.openToSide", "explorer.openAsText", "explorer.compareWithActive", "explorer.plot",
    "explorer.compareConvergence", "explorer.gallery", "explorer.legend", "-",
    "explorer.copyPath", "explorer.reveal", "explorer.history", "-", "explorer.setup", "explorer.submit", "explorer.runHere", "explorer.workflow", "explorer.dbBuild"],
  // a folder's menu (Areas B, C and D agree on the order): Batch Setup, Submit Folder, Create or Update Database
  folder: ["explorer.batch", "explorer.submitFolder", "explorer.workflow", "explorer.dbBuild", "-", "explorer.gallery", "explorer.legend"],
  tab: ["tab.close", "tab.closeOthers", "tab.closeRight", "-", "tab.moveToOtherGroup", "tab.moveLeft", "tab.moveRight", "-", "tab.copyPath", "tab.reveal"],
};

/* The menu bar, exactly as §1.3 lists it. `noAlt`: no Alt+letter mnemonic by
 * default: Firefox takes Alt+E and Alt+B for its own Edit and Bookmarks menus
 * (measured with real keys, plan T16); F10 reaches them. */
export const MENUS = [
  { label: "File", items: ["file.new", "file.open", "file.openFolder", { label: "Open Recent", sub: "recent" }, "file.importClipboard", "-",
    "file.save", "file.saveAs", "file.saveAll", "file.revert", "-", "file.exportImage", "file.exportLegend", "file.exportLegendOpen",
    { label: "Export Structure", sub: ["file.export.CIF", "file.export.CIFSym", "file.export.POSCAR", "file.export.XYZ", "file.export.extXYZ", "-", "file.export.QE"] },
    "file.copyImage", "-", "file.closeTab", "file.closeAll", "-", "file.settings", "-", "file.signOut"] },
  { label: "Edit", noAlt: true, items: ["edit.undo", "edit.redo", "-", "edit.cut", "edit.copy", "edit.paste", "edit.duplicate", "edit.delete", "-",
    "edit.replaceElement", "edit.setCoordinates", "edit.editCell", "-", "edit.history"] },
  { label: "View", items: ["view.palette", "-",
    { label: "View Along", sub: ["view.alongA", "view.alongB", "view.alongC", "view.alongAStar", "view.alongBStar", "view.alongCStar", "view.alongUVW"] },
    { label: "Rotate 90°", sub: ["view.rotateLeft", "view.rotateRight", "view.rotateUp", "view.rotateDown", "view.rotateInPlaneMinus", "view.rotateInPlanePlus"] },
    "view.reset", "view.frameSelection", "view.pivotCell", "view.pivotSelection", "view.projection", "-",
    "view.cell", "view.boundary", "view.bonds",
    { label: "Show Labels", sub: ["view.labels.off", "view.labels.element", "view.labels.index", "view.labels.both", "view.labels.wyckoff", "-", "view.labelsCycle"] },
    "view.axes", "view.measurements", "view.polyhedra",
    { label: "Display Range", sub: ["view.range1", "view.range2", "view.range3", "view.rangeCustom", "-", "view.rangeCycle"] },
    { label: "Style", sub: ["view.style.ballstick", "view.style.spacefill", "view.style.stick", "view.style.wireframe"] }, "-",
    { label: "Editor Groups", sub: ["view.splitRight", "view.moveToOtherGroup", "view.focusOtherGroup", "view.focusLeftGroup", "view.focusRightGroup", "view.closeGroup", "-", "view.linkCameras"] },
    "view.explorer", { label: "Explorer Sections", sub: ["view.sidebar.show.editors", "view.sidebar.show.files", "view.sidebar.show.db",
      "view.sidebar.show.recent", "-", "view.resetSidebar"] },
    "view.inspector", { label: "Inspector Sections", sub: ["view.expandSections", "view.collapseSections"] },
    "view.bottom", "view.hideAllPanels", "view.zen", "view.cluster", "view.workflows", "-",
    { label: "Theme", sub: [...themeMenu(), "-", "view.themePicker", "view.themeCycle"] }, "view.elementColors",
    { label: "Zoom UI", sub: ["view.zoomIn", "view.zoomOut", "view.zoomReset"] }] },
  { label: "Select", items: ["select.all", "select.none", "select.invert", "-", "select.byElement", "select.equivalents",
    "select.within", "select.neighbors", "select.grow", "select.shrink", "select.byIndex", "select.byFrac", "-",
    "select.saveSet", { label: "Selection Sets", sub: ["select.sets"] }] },
  { label: "Build", noAlt: true, items: ["build.addAtom", "build.replace", "build.delete", "-", "build.supercell", "build.slab",
    { label: "Point Defect", sub: ["build.defect.vacancy", "build.defect.substitution", "build.defect.interstitial", "build.defect.antisite"] },
    "build.transform", "build.symmetrize", "-", "build.clean", "-", "build.symMode"] },
  { label: "Tools", items: ["tool.select", "tool.lasso", "tool.orbit", "tool.pan", "tool.zoom", "tool.move", "tool.add",
    "tool.replace", "tool.measure", "tool.bond", "-", "tool.measureSelection", "tool.clearMeasurements", "-",
    "tool.coordination", "tool.compare"] },
  { label: "Calculation", items: ["calc.setup", "calc.preview", "calc.saveJson", "calc.preflight", "calc.copyCli", "-",
    "calc.submit", "calc.runHere", "calc.batch", "calc.newWorkflow", "view.jobs", "view.workflows", "-",
    { label: "Open Example Config", sub: [] }, "calc.resetSettings", "-", "calc.autoPreview"] },
  { label: "Analysis", items: ["analysis.band", "analysis.dos", "analysis.ir", "analysis.raman", "analysis.modes", "-",
    "analysis.volumeMep", "analysis.volumeDiff", "-",
    "analysis.coordination", "-", "analysis.trajectory", "analysis.compareInput", "analysis.compareGroups", "analysis.energy", "analysis.bondTable", "analysis.displacements",
    { label: "Run", sub: ["analysis.runPlot", "analysis.convergence", "analysis.scf", "analysis.cellRun", "analysis.trackMeasurement", "-", "analysis.compareRuns"] }, "-",
    "analysis.gallery", "-", "analysis.db", "db.build", { label: "Export Table", sub: ["analysis.exportCsv", "analysis.exportJson", "analysis.exportExcel", "analysis.exportLaTeX", "analysis.exportHTML"] }, "analysis.resetPlot"] },
  { label: "Help", items: ["help.shortcuts", "help.commands", "help.docs", "help.cluster", "help.tunnel", "-", "help.copyDiagnostics", "help.report", "-", "help.checkUpdates",
    "help.about", "-", "menu.focusBar"] },
];

/** The Alt+letter mnemonics as pseudo-commands menu.open.<label> (not on a
 * Mac, where Alt types characters): in the chord index and the conflict
 * check, rebindable, but not in the palette. */
export function registerMnemonics(isMac, open) {
  MENUS.forEach((m, i) => registerPseudo({ id: `menu.open.${m.label.toLowerCase()}`, title: `Open the ${m.label} Menu`,
    menu: i, keys: isMac || m.noAlt ? [] : [`Alt+${m.label[0]}`], run: () => open(i) }));
}

/* The toolbar (§1.4), left to right. */
export const TOOLBAR = [
  ["file.open", "file.save"], ["edit.undo", "edit.redo"],
  { radio: ["tool.select", "tool.lasso", "tool.orbit", "tool.pan", "tool.zoom", "tool.move", "tool.add", "tool.replace", "tool.measure", "tool.bond"] },
  ["view.cell", "view.bonds", "view.labelsCycle", "view.boundary", "view.axes", "view.projection"],
  { split: "viewAlong" }, { split: "range" }, { segment: "sym" },
  "spacer", { text: "build.clean", label: "Clean…" }, { text: "calc.setup", label: "Setup", primary: true },
];
