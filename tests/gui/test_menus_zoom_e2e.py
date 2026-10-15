"""Every drop-down menu and popover of the GUI, opened at the browser zoom
levels people use (80 % to 200 %, 100 % above all) on three window sizes:
each one opens on top of everything at its own center, inside the window,
and no label in it (nor on the button that opens it) is cut short, unless
its whole text is the element's title.

The menus: the nine menu bar menus and each of their submenus, the toolbar's
View Along and Display Range, the 3D view's empty-space and atom menus (also
opened in the window's bottom right corner, where they must flip), the
explorer's file and folder menus, the tab menu, Select by Element, the
periodic table picker, a plot's Units and Export menus and its Plot Options
popover, and a menu opened over a modal dialog (Settings). A second test
sweeps the buttons, labels and drop-downs of the main views the same way.

Browser zoom Z on a W x H window lays the page out in a W/Z x H/Z CSS-pixel
viewport at devicePixelRatio Z (which is also what moves the fonts' rounding).
Chromium gets exactly that from the DevTools protocol
(Emulation.setDeviceMetricsOverride: one browser for every case); Firefox from
its layout.css.devPixelsPerPx pref, set at start (one browser per zoom level),
and a W/Z x H/Z viewport. Opt-in browser test (marker gui), not in the CI
smoke subset.
"""
import os
import time

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")                 # the server reads the structures with it

# 100 % is the case that must be perfect; the other zoom levels people use
# (0.8, 0.9, 1.1, 1.25, 1.5, 2.0) run with MACE_GUI_ALL_ZOOMS=1
ZOOMS = (0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0) if os.environ.get("MACE_GUI_ALL_ZOOMS") == "1" else (1.0,)
SIZES = ((1920, 1080), (1366, 768), (1024, 700))
CASES = [(z, w, h) for z in ZOOMS for (w, h) in SIZES]

# what must have opened in every case (a menu that silently failed to open would pass the checks)
EXPECTED = {"File", "Edit", "View", "Select", "Build", "Tools", "Calculation", "Analysis", "Help",
            "View along", "Display range", "Viewport menu", "Viewport menu (corner)", "Atom menu",
            "Atom menu (corner)", "Explorer menu", "Folder menu", "Tab menu", "Select by Element",
            "Pick an element", "Units", "Export", "Plot options", "Explorer menu (over a modal dialog)"}

# shared by both page scripts: helpers, and what counts as a cut label
PRELUDE = r"""
const M = window.__mace;
const results = [];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const frames = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
const waitFor = async (fn, ms = 6000) => {
  const t0 = performance.now();
  for (;;) { const e = fn(); if (e) return e; if (performance.now() - t0 > ms) return null; await sleep(30); }
};
const shown = (e) => { if (!e || !e.isConnected) return false; const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
const visible = (sel) => [...document.querySelectorAll(sel)].find(shown) || null;
const describe = (e) => e.tagName.toLowerCase() + (e.id ? '#' + e.id : '') + [...e.classList].map((c) => '.' + c).join('');
const closeAll = async () => { document.body.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true, button: 0})); await frames(); };
const lastMenu = () => [...document.querySelectorAll('.menu')].pop() || null;
const measure = document.createElement('span');
measure.style.cssText = 'position:fixed;left:-10000px;top:0;white-space:pre;visibility:hidden';
document.body.appendChild(measure);

/* A label is cut when its text needs more room than its box gives it,
 * unless the element (or the entry it is in) has the whole text as its title. */
function cut(t) {
  if (t.tagName === 'SELECT') {
    const o = t.options[t.selectedIndex];
    if (!o) return null;
    const cs = getComputedStyle(t);
    measure.style.font = cs.font; measure.style.lineHeight = 'normal'; measure.textContent = o.text;
    const m = measure.getBoundingClientRect();
    const roomW = t.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight) - 16;  // the arrow
    const roomH = t.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
    if ((m.width <= roomW + 1 && m.height <= roomH + 1) || (t.title || '').includes(o.text)) return null;
    return `select "${o.text}" needs ${m.width.toFixed(0)}x${m.height.toFixed(0)}, has ${roomW.toFixed(0)}x${roomH.toFixed(0)}`;
  }
  const text = (t.textContent || '').trim();
  if (!text || !shown(t)) return null;
  if (t.scrollWidth <= t.clientWidth + 1 && t.scrollHeight <= t.clientHeight + 2) return null;
  for (let a = t; a; a = a.parentElement) {
    if ((a.title || '').includes(text)) return null;
    if (a.matches('.menu, .pk-menu, .pk-pop, .ptable')) break;
  }
  return `"${text}" cut: ${t.scrollWidth}x${t.scrollHeight} in ${t.clientWidth}x${t.clientHeight}`;
}
const LABELS = '.lbl, kbd, button, label, select, .pk-row > span, .pt-title';

"""

OPEN_ALL = r"""
const done = arguments[arguments.length - 1];
""" + PRELUDE + r"""
function check(name, el, trigger) {
  const problems = [];
  if (!shown(el)) { results.push({name, problems: ['did not open']}); return; }
  const r = el.getBoundingClientRect();
  const top = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
  if (!top || !(top === el || el.contains(top))) problems.push('covered at its center by ' + (top ? describe(top) : 'nothing'));
  if (r.left < -0.5 || r.top < -0.5 || r.right > innerWidth + 0.5 || r.bottom > innerHeight + 0.5)
    problems.push(`outside the ${innerWidth}x${innerHeight} window: [${r.left.toFixed(1)}, ${r.top.toFixed(1)}, ${r.right.toFixed(1)}, ${r.bottom.toFixed(1)}]`);
  for (const t of el.querySelectorAll(LABELS)) { const c = cut(t); if (c) problems.push(c); }
  if (trigger) { const c = cut(trigger); if (c) problems.push('its button: ' + c); }
  results.push({name, problems, box: [r.left, r.top, r.right, r.bottom].map((x) => Math.round(x))});
}

async function activate(kind) {
  const k = M.docs().findIndex((d) => d.kind === kind);
  if (k < 0) return false;
  M.activate(k);
  await sleep(400); await frames();
  return true;
}

(async () => {
  try {
    await closeAll();
    await activate('structure');
    // the menu bar: each menu, and each of its submenus
    for (const b of document.querySelectorAll('.menu-top')) {
      b.scrollIntoView({block: 'nearest', inline: 'nearest'});
      b.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true, button: 0}));
      await frames();
      const m = document.querySelector(`.menu[data-top="${b.dataset.menu}"]`);
      check(b.textContent.trim(), m, b);
      for (const item of m ? [...m.querySelectorAll('.menu-item[data-sub]')] : []) {
        item.click();
        await frames();
        const sub = lastMenu();
        check(`${b.textContent.trim()} › ${item.textContent.trim()}`, sub !== m ? sub : null, item);
      }
      await closeAll();
    }
    // the toolbar's drop-downs
    for (const [id, label] of [['tb-along', 'View along'], ['tb-range', 'Display range']]) {
      const b = document.getElementById(id);
      if (b) { b.scrollIntoView({block: 'nearest', inline: 'nearest'}); b.click(); await frames(); }
      check(label, document.querySelector(`.menu[aria-label="${label}"]`), b);
      await closeAll();
    }
    // the 3D view: a right-click on empty space (what the pointer controller opens), an atom, and both in the corner
    const wrap = visible('.views .viewer-wrap');
    const vr = wrap ? wrap.getBoundingClientRect() : {left: 0, top: 0};
    M.contextAt(vr.left + 6, vr.top + 6); await frames();
    check('Viewport menu', document.querySelector('.menu[aria-label="Viewport menu"]'));
    await closeAll();
    M.contextFor('empty', null, innerWidth - 2, innerHeight - 2); await frames();
    check('Viewport menu (corner)', document.querySelector('.menu[aria-label="Viewport menu"]'));
    await closeAll();
    M.contextFor('atom', {atom: {i: 0, image: [0, 0, 0]}}, innerWidth / 3, innerHeight / 3); await frames();
    check('Atom menu', document.querySelector('.menu[aria-label="Atom menu"]'));
    await closeAll();
    M.contextFor('atom', {atom: {i: 0, image: [0, 0, 0]}}, innerWidth - 2, innerHeight - 2); await frames();
    check('Atom menu (corner)', document.querySelector('.menu[aria-label="Atom menu"]'));
    await closeAll();
    // the explorer and the tabs, by a right-click on them (or, where the window has no room for them, as they would open)
    for (const [sel, label, kind, args] of [['#file-list .file[data-path]', 'Explorer menu', 'explorer', {path: 'structures/MgO.cif'}],
        ['#file-list .folder[data-dir]', 'Folder menu', 'folder', {dir: 'structures/'}], ['.doc-tab', 'Tab menu', 'tab', {docId: 1}]]) {
      const e = visible(sel);
      if (e) {
        e.scrollIntoView({block: 'nearest'});
        const r = e.getBoundingClientRect();
        e.dispatchEvent(new MouseEvent('contextmenu', {bubbles: true, cancelable: true, button: 2, clientX: r.left + 8, clientY: r.top + r.height / 2}));
      } else M.contextFor(kind, args);
      await frames();
      check(label, document.querySelector(`.menu[aria-label="${label}"]`));
      await closeAll();
    }
    // Select by Element, and the periodic table (Replace Element with a selection)
    M.run('select.byElement'); await frames();
    check('Select by Element', document.querySelector('.menu[aria-label="Select by Element"]'));
    await closeAll();
    M.run('select.all'); await frames();
    M.run('edit.replaceElement');
    check('Pick an element', await waitFor(() => visible('.ptable')));
    await closeAll();
    M.run('select.none');
    // a plot's toolbar: Units and Export, and the Plot Options popover
    if (await activate('runplot')) {
      for (const [sel, label, pop] of [['.pk-units', 'Units', '.pk-menu'], ['.pk-export', 'Export', '.pk-menu'], ['.pk-more', 'Plot options', '.pk-pop']]) {
        const b = await waitFor(() => visible(sel), 4000);
        if (b) { b.scrollIntoView({block: 'nearest', inline: 'nearest'}); b.click(); await frames(); }
        check(label, visible(pop), b);
        await closeAll();
      }
    }
    await activate('structure');
    // a menu opened while a modal dialog is up (Settings) must show above it, and take clicks
    M.run('file.settings');
    const dlg = await waitFor(() => visible('dialog.settings-dlg[open]'));
    M.contextFor('explorer', {path: 'structures/MgO.cif'}, innerWidth / 2, innerHeight / 2); await frames();
    check('Explorer menu (over a modal dialog)', document.querySelector('.menu[aria-label="Explorer menu"]'));
    await closeAll();
    if (dlg) dlg.close();
    measure.remove();
    done(results);
  } catch (err) { measure.remove(); done([{name: 'script', problems: [String(err && err.stack || err)]}]); }
})();
"""


CONTROLS = r"""
const done = arguments[arguments.length - 1];
""" + PRELUDE + r"""
const problems = [];
const titled = (t, text) => { for (let a = t; a; a = a.parentElement) if ((a.title || '').includes(text)) return true; return false; };
/* The box around the control that clips it sideways (the nearest one that
 * clips at all): past its end when the box does not scroll, or past its start
 * (flex-end overflow), where no scrolling reaches. */
function clippedBy(t) {
  const r = t.getBoundingClientRect();
  for (let a = t.parentElement; a && a !== document.body; a = a.parentElement) {
    const s = getComputedStyle(a);
    if (s.overflowX === 'visible') continue;
    const ar = a.getBoundingClientRect(), z = ar.width / (a.offsetWidth || 1);
    const L = ar.left + a.clientLeft * z, R = L + a.clientWidth * z;
    const scrolls = s.overflowX === 'auto' || s.overflowX === 'scroll';
    if (r.right > R + 1 && r.left < R - 1 && !scrolls) return describe(a);
    if (r.left < L - 1 && r.right > L + 1 && a.scrollLeft <= 0) return describe(a) + ' (before its start)';
    return null;      // the first box that clips decides: what it lets scroll into view is reachable
  }
  return null;
}
function scan(state, root) {
  for (const t of (root || document).querySelectorAll('button, select, [role=tab], label')) {
    if (!shown(t) || t.closest('.menu, .pk-menu, .pk-pop, .ptable, .sr-only')) continue;
    const text = (t.tagName === 'SELECT' ? (t.options[t.selectedIndex] || {}).text : t.textContent || '').trim();
    if (!text) continue;
    const c = cut(t);
    if (c) problems.push(`${state}: ${describe(t)} ${c}`);
    const by = clippedBy(t);
    if (by && !titled(t, text)) problems.push(`${state}: ${describe(t)} "${text}" cut off by ${by}`);
  }
}
const byPath = async (pred) => {
  const k = M.docs().findIndex(pred);
  if (k >= 0) { M.activate(k); await sleep(500); await frames(); }
  return k >= 0;
};
const narrow = () => innerWidth < 1100;

(async () => {
  try {
    await closeAll();
    // a structure, with the Inspector (a drawer below 1100px: opened for the scan)
    await byPath((d) => d.path === 'structures/MgO.cif');
    const insp = document.getElementById('inspector');
    const drawerOpen = () => narrow() && getComputedStyle(insp).display !== 'none';
    const closeDrawer = async () => { if (drawerOpen()) { M.run('view.inspector'); await sleep(200); } };
    if (narrow() && !drawerOpen()) { M.run('view.inspector'); await sleep(300); }
    scan('structure');
    await closeDrawer();
    // its Range & boundary and Rendering panels
    for (const [cmd, label] of [['view.rangePanel', 'range panel'], ['view.renderPanel', 'rendering panel']]) {
      M.run(cmd);                     // the Inspector's section, scrolled into view (in the drawer below 1100px)
      await sleep(400);
      scan(label, insp);
      await closeDrawer();
    }
    // Calculation Setup on it
    M.run('calc.setup', {open: true});
    if (!await waitFor(() => visible('#inspector .su-row select'), 60000)) problems.push('setup: its form did not appear');
    await sleep(300);
    scan('setup', document.getElementById('inspector'));
    M.run('calc.setup');
    await sleep(300);
    await closeDrawer();
    // a geometry optimization's output, and its Run Analysis plot
    await byPath((d) => d.path === 'outputs/Ag2Br3_opt_unfinished.out');
    scan('optimization output');
    await byPath((d) => d.kind === 'runplot');
    scan('run analysis');
    // every section of Settings
    M.run('file.settings');
    const dlg = await waitFor(() => visible('dialog.settings-dlg[open]'));
    if (!dlg) problems.push('Settings did not open');
    for (const b of dlg ? [...dlg.querySelectorAll('.set-nav-item')] : []) {
      b.click(); await frames();
      scan('Settings › ' + b.textContent.trim(), dlg);
    }
    if (dlg) dlg.close();
    await byPath((d) => d.path === 'structures/MgO.cif');
    measure.remove();
    done(problems);
  } catch (err) { measure.remove(); done(['script: ' + String(err && err.stack || err)]); }
})();
"""


class Session:
    """One server; the browser at a zoom level and window size (see the module doc)."""

    def __init__(self, tmp):
        self.root = H.examples_copy(tmp)
        self.proc, self.base, _ = H.start_server(self.root, tmp / "xdg")
        self.name = os.environ.get("MACE_GUI_BROWSER", "firefox")
        self.chromium = self.name.startswith("chrom")
        self.drv, self.zoom = None, None

    def _launch(self, zoom):
        if self.drv is not None:
            self.drv.quit()
            self.drv = None
        prefs = None if self.chromium else {"layout.css.devPixelsPerPx": str(zoom)}
        self.drv = H.browser(self.name, width=1366, height=768, **({} if self.chromium else {"prefs": prefs}))
        self.drv.set_script_timeout(120)
        H.load(self.drv, self.base)
        d = self.drv
        d.execute_async_script("const done = arguments[0]; (async () => {"
                               "await window.__mace.open('structures/MgO.cif');"
                               "await window.__mace.open('outputs/Ag2Br3_opt_unfinished.out');"
                               "await window.__mace.run('analysis.energy'); done(); })()")
        H.wait(d, "return window.__mace.docs().some((x) => x.kind === 'runplot')", 30)
        self.zoom = zoom

    def at(self, zoom, width, height):
        w, h = round(width / zoom), round(height / zoom)
        if self.chromium:
            if self.drv is None:
                self._launch(zoom)
            self.drv.execute_cdp_cmd("Emulation.setDeviceMetricsOverride",
                                     {"width": w, "height": h, "deviceScaleFactor": zoom, "mobile": False})
        else:
            if self.zoom != zoom:
                self._launch(zoom)
            H.set_viewport(self.drv, w, h)
        # Firefox rounds a window at a fractional pixel ratio: one CSS pixel either way is the same layout
        H.wait(self.drv, "return Math.abs(innerWidth - arguments[0]) <= 1 && Math.abs(innerHeight - arguments[1]) <= 1", 10, w, h)
        time.sleep(0.4)                     # the shell's relayout and the 3D view's resize
        return self.drv

    def close(self):
        try:
            if self.drv is not None:
                self.drv.quit()
        finally:
            self.proc.terminate()
            self.proc.wait(timeout=15)


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    s = Session(tmp_path_factory.mktemp("menuszoom"))
    try:
        yield s
    finally:
        s.close()


@pytest.mark.parametrize("zoom,width,height", CASES, ids=[f"{w}x{h}-{round(z * 100)}pct" for z, w, h in CASES])
def test_every_menu_opens_whole_and_on_top(session, zoom, width, height):
    drv = session.at(zoom, width, height)
    try:
        res = drv.execute_async_script(OPEN_ALL)
        opened = {r["name"] for r in res if r["problems"] != ["did not open"]}
        missing = sorted(EXPECTED - opened)
        bad = [f"{r['name']}: {p}" for r in res for p in r["problems"]]
        assert not missing and not bad, "\n".join([f"viewport {drv.execute_script('return [innerWidth, innerHeight, devicePixelRatio]')},"
                                                   f" {len(res)} menus checked; did not open: {missing}", *bad])
        assert len(res) >= len(EXPECTED) + 10      # the submenus too
    except BaseException:
        H.save_failure(drv, f"menus_zoom_{width}x{height}_{round(zoom * 100)}")
        raise


@pytest.mark.parametrize("zoom,width,height", CASES, ids=[f"{w}x{h}-{round(z * 100)}pct" for z, w, h in CASES])
def test_buttons_and_selects_show_their_labels(session, zoom, width, height):
    """The buttons, tabs, labels and drop-downs of a structure with its
    Inspector, the view panels, Calculation Setup, an optimization's output,
    its Run Analysis plot and every Settings section: no text cut short by its
    own box or by a box around it that does not scroll."""
    drv = session.at(zoom, width, height)
    try:
        bad = drv.execute_async_script(CONTROLS)
        assert not bad, "\n".join([f"viewport {drv.execute_script('return [innerWidth, innerHeight, devicePixelRatio]')}:", *bad])
    except BaseException:
        H.save_failure(drv, f"controls_zoom_{width}x{height}_{round(zoom * 100)}")
        raise
