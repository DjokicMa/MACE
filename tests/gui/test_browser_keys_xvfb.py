"""Real key events in a visible Firefox (plan T16, carry-over e).

WebDriver's key actions go to the page, not through Firefox's own keyboard
handling, so a test there can't see Firefox take a key first (Alt+Left going
Back, Ctrl+P printing, Ctrl+Shift+I opening the developer tools). This test
starts Firefox *not* headless on an Xvfb display, drawing WebGL with Mesa's
llvmpipe (no GPU), and sends each key with xdotool (XTEST), exactly as a
keyboard would. For every key it records whether the app's command ran and
whether the browser did something (left the page, changed the history, the
window size (devtools docked, the menubar shown), the zoom, opened a window,
or lost focus), and writes the table to $MACE_GUI_VISIBLE_OUT/visible_keys.json.

Opt-in: MACE_GUI_VISIBLE=1 (it needs Xvfb and xdotool; MACE_GUI_XVFB points
at an Xvfb binary that isn't installed system-wide). It never runs on the
real display unless someone starts it there on purpose
(MACE_GUI_VISIBLE=1 MACE_GUI_VISIBLE_DISPLAY=:0, which types into a visible
window).
"""
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

pytestmark = [pytest.mark.guivisible,
              pytest.mark.skipif(os.environ.get("MACE_GUI_VISIBLE") != "1",
                                 reason="real key events in a visible Firefox are opt-in: MACE_GUI_VISIBLE=1")]
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")                 # the server reads the demo structures with it

# (xdotool combo, the app command expected (or None: a browser or reserved key), the state)
VIEW, MOVE, TEXT = "viewport", "move tool, selection", "text input"
MATRIX = [
    ("alt+Left", "view.rotateLeft", VIEW), ("alt+Right", "view.rotateRight", VIEW),
    ("alt+Left", "edit.nudgeLeft", MOVE), ("alt+Right", "edit.nudgeRight", MOVE),
    ("alt+Left", None, TEXT), ("alt+Right", None, TEXT),
    ("alt+1", "view.style.ballstick", VIEW), ("alt+2", "view.style.spacefill", VIEW),
    ("alt+3", "view.style.stick", VIEW), ("alt+4", "view.style.wireframe", VIEW),
    ("alt+n", "file.new", VIEW), ("alt+w", "file.closeTab", VIEW), ("alt+e", None, VIEW),
    ("alt+q", "select.equivalents", VIEW), ("alt+r", "select.within", VIEW), ("alt+b", None, VIEW),
    ("alt+i", "select.byIndex", VIEW), ("alt+u", "view.alongUVW", VIEW),
    ("F1", "view.palette", VIEW), ("F6", None, VIEW), ("shift+F10", "view.contextMenu", VIEW),
    ("ctrl+k", None, VIEW), ("ctrl+alt+k", "view.palette", VIEW), ("ctrl+p", "file.quickOpen", VIEW), ("ctrl+o", "file.open", VIEW),
    ("ctrl+s", "file.save", VIEW), ("ctrl+e", "file.goToDoc", VIEW), ("ctrl+b", "view.explorer", VIEW), ("ctrl+j", "view.bottom", VIEW),
    ("ctrl+g", "nav.goToAtom", VIEW), ("ctrl+i", "select.invert", VIEW), ("ctrl+d", "edit.duplicate", VIEW), ("ctrl+m", "tool.measureSelection", VIEW),
    ("ctrl+slash", "help.shortcuts", VIEW), ("ctrl+comma", "file.settings", VIEW), ("ctrl+equal", "view.zoomIn", VIEW),
    ("ctrl+minus", "view.zoomOut", VIEW), ("ctrl+0", "view.zoomReset", VIEW), ("ctrl+1", None, VIEW),
    ("ctrl+shift+c", "file.copyImage", VIEW), ("ctrl+shift+e", "view.focusExplorer", VIEW),
    ("ctrl+shift+i", "view.focusInspector", VIEW), ("ctrl+shift+j", None, VIEW),
    ("ctrl+shift+k", None, VIEW), ("ctrl+shift+m", None, VIEW),
    ("ctrl+shift+h", "edit.history", VIEW), ("ctrl+shift+r", "calc.setup", VIEW), ("ctrl+shift+s", "file.saveAs", VIEW),
    ("ctrl+shift+v", None, VIEW), ("ctrl+shift+x", "file.exportImage", VIEW), ("ctrl+shift+y", "view.tasks", VIEW),
    ("ctrl+shift+d", "view.database", VIEW), ("ctrl+shift+g", None, VIEW), ("ctrl+shift+a", None, VIEW),
    ("ctrl+shift+l", "edit.editCell", VIEW), ("ctrl+shift+u", "edit.setCoordinates", VIEW),
    ("ctrl+alt+p", "build.supercell", VIEW), ("ctrl+alt+h", "build.slab", VIEW), ("ctrl+alt+y", "build.symMode", VIEW),
    ("ctrl+alt+s", "file.saveAll", VIEW), ("ctrl+alt+b", "view.inspector", VIEW), ("ctrl+alt+z", "view.zen", VIEW),
    ("ctrl+alt+t", "view.themeCycle", VIEW), ("ctrl+alt+o", "traj.overlay", VIEW), ("ctrl+alt+1", "view.focusViewport", VIEW),
    ("ctrl+alt+i", "view.inspectorDrawer", VIEW),
    # the new homes of the keys Firefox took (the six above are now reserved)
    ("ctrl+alt+e", "select.byElement", VIEW), ("ctrl+alt+n", "select.neighbors", VIEW), ("ctrl+alt+j", "calc.saveJson", VIEW),
    ("ctrl+alt+m", "tool.clearMeasurements", VIEW), ("ctrl+alt+g", "build.clean", VIEW), ("ctrl+alt+a", "select.none", VIEW),
    ("F12", None, VIEW),
]

SNAP = """const lc = window.__mace.lastCommand();
return {cmd: lc ? lc.id : null, t: lc ? lc.t : 0, href: location.href, hist: history.length,
        w: innerWidth, h: innerHeight, dpr: devicePixelRatio, vis: document.visibilityState, focus: document.hasFocus(),
        ae: (document.activeElement && (document.activeElement.id || document.activeElement.className || document.activeElement.tagName)) || '',
        dlg: [...document.querySelectorAll('dialog[open]')].map((d) => d.id || d.className).join(','),
        // a Move-tool nudge goes through the tool's own key layer, not a command: watch the History
        edit: (() => { const e = window.__mace.editor && window.__mace.editor.doc(); return e ? `${e.rows.length}:${e.pos}:${e.rows[e.pos].label}` : ''; })()}"""


def out_dir(tmp_path):
    d = Path(os.environ.get("MACE_GUI_VISIBLE_OUT") or tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    return d


def prepare(drv, state):
    drv.execute_script("document.querySelectorAll('dialog[open]').forEach((d) => d.close()); window.__mace.run('tool.select')")
    # a clean MgO.cif in front (the Move rows leave it modified; Alt+W closes it, and a
    # modified one would ask with the page's confirm())
    if state != MOVE and drv.execute_script("const d = window.__mace.doc(); return !d || d.title !== 'MgO.cif' || window.__mace.docs().some((x) => x.dirty)"):
        drv.execute_script("window.__mace.closeAll()")
        U.open_doc(drv, "structures/MgO.cif")
    if state == TEXT:
        drv.execute_script("let i = document.getElementById('xv-in'); if (!i) { i = document.createElement('input'); i.id = 'xv-in';"
                           " i.style.cssText = 'position:fixed;top:4px;left:4px;z-index:9999'; document.body.appendChild(i); } i.value = 'abc'; i.focus()")
        return
    drv.execute_script("const i = document.getElementById('xv-in'); if (i) i.remove(); window.__mace.editor.select([0], 'replace')")
    if state == MOVE:
        # P1: a Sym-mode nudge of an atom on a special position asks first (a dialog);
        # the Move tool starts with a grab (as G does): end it, the tool stays
        drv.execute_script("window.__mace.run('build.symModeP1'); window.__mace.run('tool.move'); window.__mace.editor.select([0], 'replace')")
        drv.execute_script("document.querySelector('.viewer-wrap').focus()")
        return "esc"
    drv.execute_script("document.querySelector('.viewer-wrap').focus()")


def test_visible_keys_matrix(tmp_path):
    display = os.environ.get("MACE_GUI_VISIBLE_DISPLAY")
    ctx = H.xvfb() if not display else None
    disp = display or ctx.__enter__()
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1")
    results = []
    drv = None

    def start():
        d = H.visible_firefox(disp)
        d.get("about:blank")                     # a page behind the app: Back would visibly leave it
        H.load(d, base)
        H.wait(d, "return document.documentElement.dataset.appReady === '1'", 30)
        # a second tab, so Firefox's Alt+digit / Ctrl+digit tab switch would show
        d.switch_to.new_window("tab")
        d.get("about:blank")
        d.switch_to.window(d.window_handles[0])
        U.open_doc(d, "structures/MgO.cif")
        return d
    try:
        drv = start()
        renderer = H.webgl_renderer(drv)
        pid = drv.service.process.pid
        ff = [int(x) for x in subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()] or [pid]
        def ready(state):
            if prepare(drv, state) == "esc":
                H.xkey(disp, ff[0], "Escape")
                drv.execute_script("window.__mace.editor.select([0], 'replace'); document.querySelector('.viewer-wrap').focus()")
            time.sleep(0.15)
        for combo, want, state in MATRIX:
            ready(state)
            if not drv.execute_script("return document.hasFocus()"):
                results.append({"key": combo, "state": state, "expected": want, "app": None,
                                "browser": ["the page had no focus before the key"], "verdict": "skipped"})
                drv.quit()
                drv = start()
                pid = drv.service.process.pid
                ff = [int(x) for x in subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()] or [pid]
                ready(state)
            # a command that can't run in this context (no trajectory, nothing to save) says so instead: not a key problem
            reason = drv.execute_script("return arguments[0] ? window.__mace.menuReason(arguments[0]) : ''", want)
            if want and reason:
                results.append({"key": combo, "state": state, "expected": want, "app": None, "browser": [],
                                "verdict": "disabled here", "reason": reason})
                continue
            before = drv.execute_script(SNAP)
            handles0, wins0 = len(drv.window_handles), H.xwindows(disp, ff[0])
            H.xkey(disp, ff[0], combo)
            time.sleep(0.6)
            try:
                after = drv.execute_script(SNAP)
            except Exception as exc:          # the page went away
                after = {"cmd": None, "t": 0, "href": f"<{type(exc).__name__}>", "hist": -1, "w": 0, "h": 0, "dpr": 0, "vis": "?",
                         "focus": False, "ae": "", "dlg": "", "edit": ""}
            handles1, wins1 = len(drv.window_handles), H.xwindows(disp, ff[0])
            app = after["t"] != before["t"] or (state == MOVE and after["edit"] != before["edit"])
            effects = []
            if after["href"] != before["href"]:
                effects.append("left the page")
            if after["hist"] != before["hist"]:
                effects.append("history changed")
            if (after["w"], after["h"]) != (before["w"], before["h"]):
                effects.append("window area changed (devtools, menubar or sidebar)")
            if after["dpr"] != before["dpr"]:
                effects.append("zoom changed")
            if handles1 != handles0:
                effects.append("new tab or window")
            if wins1 > wins0:
                effects.append("new X window (a dialog)")
            if not after["focus"] and before["focus"]:
                effects.append("the page lost focus")
            if after["vis"] != before["vis"]:
                effects.append("the tab was switched away")
            verdict = ("OK" if app and not effects else "both fire" if app and effects else
                       "browser wins" if effects else "nothing")
            results.append({"key": combo, "state": state, "expected": want,
                            "app": (after["cmd"] if after["t"] != before["t"] else after["edit"]) if app else None,
                            "browser": effects, "verdict": verdict, "focus_before": before["ae"], "dialogs_after": after["dlg"]})
            if effects:                       # a clean browser for the next key
                drv.quit()
                drv = start()
                pid = drv.service.process.pid
                ff = [int(x) for x in subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()] or [pid]
            else:
                # the reset: Esc twice (real keys) ends a grab, a modal tool layer, a menu or a dialog
                for _ in range(2):
                    H.xkey(disp, ff[0], "Escape")
                    time.sleep(0.1)
                drv.execute_script("document.querySelectorAll('dialog[open]').forEach((d) => d.close());"
                                   "document.dispatchEvent(new Event('mace-overlay'));"          # closes any open menu
                                   "if (window.__mace.palette.isOpen()) window.__mace.palette.close()")
    finally:
        (out_dir(tmp_path) / "visible_keys.json").write_text(json.dumps(
            {"display": disp, "renderer": locals().get("renderer", ""), "rows": results}, indent=2) + "\n")
        if drv is not None:
            drv.quit()
        proc.terminate()
        proc.wait(timeout=15)
        if ctx is not None:
            ctx.__exit__(None, None, None)
    assert "llvmpipe" in renderer.lower(), renderer
    bad = [r for r in results if r["expected"] and r["verdict"] not in ("OK", "disabled here")]
    assert not bad, json.dumps(bad, indent=1)
    f12 = [r for r in results if r["key"] == "F12"][0]
    assert f12["app"] is None                     # the app never blocks the developer tools
