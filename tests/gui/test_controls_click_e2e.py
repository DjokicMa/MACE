"""Every visible button that names a command runs it on a real click (polish A1).

The activity bar, the theme/settings/search buttons in the menu bar and the
bottom-left gear had no click listener at all; every earlier test reached
those commands through window.__mace.run or a shortcut, so none noticed. These
tests click with real WebDriver pointer input (ActionChains) in a fresh
browser profile, on a server started through mace_cli, with real CRYSTAL data
copied from test/ under neutral names.
"""
import os
import time

import pytest

import guiharness as H
import realdata

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")                 # the server reads the structures with it

from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402

# where [data-cmd] buttons live outside the menus
CONTAINERS = ".menubar-end, #activity, #toolbar, #statusbar, #tab-welcome, #inspector, #zen-exit"

VISIBLE_CMDS = """
const sel = arguments[0];
const out = [];
for (const box of document.querySelectorAll(sel)) {
  const list = box.matches('[data-cmd]') ? [box] : [...box.querySelectorAll('[data-cmd]')];
  for (const b of list) {
    if (b.closest('.menu') || b.closest('dialog')) continue;
    const r = b.getBoundingClientRect();
    if (!r.width || !r.height || getComputedStyle(b).visibility === 'hidden' || b.closest('[hidden]')) continue;
    const top = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    if (!top || !(b === top || b.contains(top))) continue;    // covered: not clickable for a user either
    out.push(b);
  }
}
const key = (b) => b.dataset.cmd + '|' + (b.id || b.closest('[id]').id) + '|' + b.textContent.trim();
if (arguments[1] !== undefined) return out.find((b) => key(b) === arguments[1]) || null;
return out.map((b) => key(b));"""

SIGNATURE = """const sh = document.getElementById('shell');
return JSON.stringify([sh.className, window.__mace.ctx().setupOpen, (window.__mace.activeDoc() || {}).id || 0,
  document.querySelector('.bottom-tab.active') ? document.querySelector('.bottom-tab.active').textContent : '']);"""


def real_click(drv, el):
    ActionChains(drv).move_to_element(el).pause(0.05).click().perform()


def menu_click(drv, el):
    """Real click on a menu entry, reached as a user does: straight down from
    the open top menu into the entry's left end (a diagonal path to its center
    would cross the next top menu, which opens on hover)."""
    w = el.size["width"]
    ActionChains(drv).move_to_element_with_offset(el, int(-w / 2 + 16), 0).pause(0.05).click().perform()


RAN = ("return performance.getEntriesByName('ui:' + arguments[1] + ':start')"
       ".some((m) => m.startTime > arguments[0])")


def now(drv):
    return drv.execute_script("return performance.now()")


def wait_ran(drv, before, cmd, timeout=1):
    """run() marks ui:<id>:start for every command it runs (a command may run
    others after it, so lastCommand alone could name the inner one)."""
    H.wait(drv, RAN, timeout, before, cmd)


def dismiss(drv):
    """Close what a click opened (dialog, palette, menu, popover), as Esc would,
    once a dialog that loads its module lazily has opened (over a tunnel that
    takes longer than the click: it must not open after this and block the next).
    A tab the click opened (Help > Documentation opens docs/gui.md on GitHub) is
    closed first: behind it the app's page is hidden, where the browser stops
    animation frames, so lazyIdle() would wait for one forever."""
    main = drv.current_window_handle
    for h in drv.window_handles:
        if h != main:
            drv.switch_to.window(h)
            drv.close()
    drv.switch_to.window(main)
    drv.execute_async_script("const d = arguments[0]; window.__mace.lazyIdle().then(d, d)")
    ActionChains(drv).send_keys(Keys.ESCAPE).perform()
    time.sleep(0.1)
    drv.execute_script("""document.querySelectorAll('dialog[open]').forEach((d) => d.close());
        if (window.__mace.palette.isOpen()) window.__mace.palette.close();
        // the menubar closes its menus (and forgets the open one) on a pointerdown elsewhere
        document.body.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true, button: 0}));
        document.querySelectorAll('#toasts > *').forEach((t) => t.remove());""")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("clicks")
    root = H.examples_copy(tmp)
    if not H.remote():
        realdata.copy(root / "real", "agte.cif", "slab_opt.out")
    proc, base, _ = H.start_server(root, tmp / "xdg")
    drv = H.firefox(1280, 800)
    try:
        H.load(drv, base)
        yield drv, base, root
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def open_doc(drv, path):
    kind = drv.execute_async_script(
        "const done = arguments[1]; window.__mace.open(arguments[0]).then((d) => done(d && d.kind))", path)
    time.sleep(0.4)
    return kind


def layout(drv, which):
    drv.execute_script("window.__mace.closeAll(); window.__mace.run('tool.select');"
                       "if (document.getElementById('shell').classList.contains('zen')) window.__mace.run('view.zen');")
    H.set_viewport(drv, 900 if which == "narrow" else 1280, 800)
    time.sleep(0.3)
    if which in ("agte", "narrow"):
        assert open_doc(drv, "real/agte.cif") == "structure"
    elif which == "slab":
        assert open_doc(drv, "real/slab_opt.out") == "trajectory"
    if which == "narrow":
        drv.execute_script("if (window.__mace.ctx && !document.getElementById('shell').classList.contains('drawer-open'))"
                           " window.__mace.run('view.inspector')")
        time.sleep(0.3)


def sweep(drv):
    """Real-click every visible [data-cmd] once; returns the ids that ran."""
    seen, ran = set(), []
    while True:
        todo = [k for k in drv.execute_script(VISIBLE_CMDS, CONTAINERS) if k not in seen]
        if not todo:
            return ran
        seen.add(todo[0])
        cmd = todo[0].split("|")[0]
        before_sig = drv.execute_script(SIGNATURE)
        before = now(drv)
        from selenium.common.exceptions import StaleElementReferenceException
        for attempt in range(3):          # a container re-rendered between finding and clicking: find it again
            b = drv.execute_script(VISIBLE_CMDS, CONTAINERS, todo[0])
            if b is None:
                break
            try:
                disabled = b.get_attribute("aria-disabled") == "true"
                real_click(drv, b)
                break
            except StaleElementReferenceException:
                time.sleep(0.1)
        if b is None:
            continue
        if disabled:
            time.sleep(0.3)
            assert not drv.execute_script(RAN, before, cmd), f"{cmd} ran although disabled"
            reason = b.get_attribute("data-reason") or ""
            toasts = drv.execute_script("return [...document.querySelectorAll('#toasts > *')].map((t) => t.textContent).join(' | ')")
            assert reason and reason in toasts, (cmd, reason, toasts)
        else:
            wait_ran(drv, before, cmd, 1)
            ran.append(cmd)
        dismiss(drv)
        if drv.execute_script(SIGNATURE) != before_sig and not disabled:
            # a toggle (Explorer, Setup, Zen, ...): put the layout back for the next click
            drv.execute_script("window.__mace.run(arguments[0])", cmd)
            time.sleep(0.2)
            dismiss(drv)
        drv.execute_script("window.__mace.run('tool.select')")


@pytest.mark.parametrize("which", ["welcome", "agte", "slab", "narrow"])
def test_every_visible_data_cmd_clicks(app, which):
    drv = app[0]
    layout(drv, which)
    ran = sweep(drv)
    # the buttons Marcus found dead are among them
    for cmd in ("view.palette", "view.themeCycle", "file.settings", "view.explorer", "file.quickOpen", "view.tasks"):
        assert cmd in ran, (which, cmd, ran)
    assert not H.errors(drv), H.errors(drv)


def test_named_effects(app):
    drv, base, root = app
    layout(drv, "welcome")
    # the theme button cycles three themes and comes back
    start = drv.execute_script("return window.__mace.settings.get('theme')")
    seen = []
    for _ in range(3):
        real_click(drv, drv.find_element("css selector", "#theme-btn"))
        time.sleep(0.15)
        seen.append(drv.execute_script("return window.__mace.settings.get('theme')"))
    assert len(set(seen)) == 3 and seen[-1] == start, seen
    # both gears open Settings
    for sel in ("#settings-btn", '#activity [data-cmd="file.settings"]'):
        real_click(drv, drv.find_element("css selector", sel))
        H.wait(drv, "return !!document.querySelector('dialog#settings[open]')", 10)
        dismiss(drv)
        H.wait(drv, "return !document.querySelector('dialog#settings[open]')", 5)
    real_click(drv, drv.find_element("css selector", "#palette-btn"))
    H.wait(drv, "return window.__mace.palette.isOpen()", 5)
    dismiss(drv)
    # Explorer toggles the side bar
    was = drv.execute_script("return document.getElementById('shell').classList.contains('no-explorer')")
    real_click(drv, drv.find_element("css selector", '#activity [data-cmd="view.explorer"]'))
    H.wait(drv, "return document.getElementById('shell').classList.contains('no-explorer') === arguments[0]", 3, not was)
    real_click(drv, drv.find_element("css selector", '#activity [data-cmd="view.explorer"]'))
    H.wait(drv, "return document.getElementById('shell').classList.contains('no-explorer') === arguments[0]", 3, was)
    # Database opens the example folder's materials.db
    real_click(drv, drv.find_element("css selector", '#activity [data-cmd="view.database"]'))
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.kind === 'db'", 15)
    # Search opens Go to File; Tasks shows the Tasks tab
    real_click(drv, drv.find_element("css selector", '#activity [data-cmd="file.quickOpen"]'))
    H.wait(drv, "return window.__mace.palette.isOpen()", 5)
    dismiss(drv)
    real_click(drv, drv.find_element("css selector", '#activity [data-cmd="view.tasks"]'))
    H.wait(drv, "const t = document.querySelector('.bottom-tab.active'); return t && /Tasks/.test(t.textContent)", 5)
    assert not H.errors(drv), H.errors(drv)


def test_keyboard_activation(app):
    """Tab to each button, then Enter, then Space: the same command runs."""
    drv = app[0]
    layout(drv, "welcome")
    for sel in ("#palette-btn", "#theme-btn", "#settings-btn", '#activity [data-cmd="view.tasks"]',
                '#activity [data-cmd="file.settings"]'):
        for key in (Keys.ENTER, Keys.SPACE):
            el = drv.find_element("css selector", sel)
            cmd = el.get_attribute("data-cmd")
            # focus the element before it, then Tab onto it (real keyboard navigation)
            drv.execute_script("""const all = [...document.querySelectorAll('button, [tabindex="0"], input, a[href]')]
                .filter((x) => !x.disabled && x.tabIndex >= 0 && x.getClientRects().length);
                const k = all.indexOf(arguments[0]); (k > 0 ? all[k - 1] : document.body).focus();""", el)
            ActionChains(drv).send_keys(Keys.TAB).perform()
            assert drv.execute_script("return document.activeElement === arguments[0]", el), sel
            before = now(drv)
            ActionChains(drv).send_keys(key).perform()
            wait_ran(drv, before, cmd, 2)
            dismiss(drv)


TOPS = "return [...document.querySelectorAll('.menu-top')]"


def open_top(drv, k, real=False):
    drv.execute_script("document.body.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true, button: 0}))")
    top = drv.execute_script(TOPS)[k]
    if real:
        real_click(drv, top)
    else:
        drv.execute_script("arguments[0].dispatchEvent(new PointerEvent('pointerdown', {bubbles: true, button: 0}))", top)
    H.wait(drv, "return document.querySelectorAll('.menu').length > 0", 3)


def test_menus_dispatch(app):
    """Every enabled entry of every menu (and submenu) runs its own command:
    a synthetic click per entry, with run() recording ids instead of running."""
    drv = app[0]
    layout(drv, "agte")
    n_tops = len(drv.execute_script(TOPS))
    checked = 0
    try:
        for k in range(n_tops):
            open_top(drv, k, real=True)
            entries = drv.execute_script("""return [...document.querySelector('.menu').querySelectorAll('.menu-item')]
                .map((b, i) => ({i, id: b.dataset.cmdId || null, sub: b.dataset.sub !== undefined,
                  off: b.getAttribute('aria-disabled') === 'true'}))""")
            dismiss(drv)
            for e in entries:
                if e["off"]:
                    continue
                if e["sub"]:
                    open_top(drv, k)
                    drv.execute_script("document.querySelector('.menu').querySelectorAll('.menu-item')[arguments[0]].click()", e["i"])
                    subs = drv.execute_script("""const m = [...document.querySelectorAll('.menu')].pop();
                        return [...m.querySelectorAll('.menu-item')].map((b, i) => ({i, id: b.dataset.cmdId || null,
                          off: b.getAttribute('aria-disabled') === 'true'}))""")
                    dismiss(drv)
                    for s in subs:
                        if s["off"] or not s["id"]:
                            continue
                        open_top(drv, k)
                        drv.execute_script("document.querySelector('.menu').querySelectorAll('.menu-item')[arguments[0]].click()", e["i"])
                        drv.execute_script("window.__mace.spyRun(true)")
                        drv.execute_script("[...document.querySelectorAll('.menu')].pop().querySelectorAll('.menu-item')[arguments[0]].click()", s["i"])
                        assert drv.execute_script("return window.__maceRan") == [s["id"]], s
                        drv.execute_script("window.__mace.spyRun(false)")
                        checked += 1
                    continue
                if not e["id"]:
                    continue
                open_top(drv, k)
                drv.execute_script("window.__mace.spyRun(true)")
                drv.execute_script("document.querySelector('.menu').querySelectorAll('.menu-item')[arguments[0]].click()", e["i"])
                assert drv.execute_script("return window.__maceRan") == [e["id"]], e
                drv.execute_script("window.__mace.spyRun(false)")
                checked += 1
    finally:
        drv.execute_script("window.__mace.spyRun(false)")
        dismiss(drv)
    assert checked > 60, checked


def test_a_relayout_does_not_shut_an_open_menu(app):
    """The shell's relayout (adopting the profile's layout at boot, a panel
    toggle) sends a synthetic resize so viewers re-measure; a menu opened in
    that frame stays open. A real window resize still closes it."""
    drv = app[0]
    layout(drv, "welcome")
    open_top(drv, 0, real=True)
    H.wait(drv, "return document.querySelectorAll('.menu').length > 0", 3)
    drv.execute_script("dispatchEvent(new Event('resize'))")
    time.sleep(0.2)
    assert drv.execute_script("return document.querySelectorAll('.menu').length") > 0
    H.set_viewport(drv, 1200, 800)
    H.wait(drv, "return document.querySelectorAll('.menu').length === 0", 3)
    H.set_viewport(drv, 1280, 800)
    dismiss(drv)


def test_menus_real_click_sample(app):
    """Per menu, a real click on its first enabled entry; plus a toggle, a
    dialog and Close All on a modified document (which must ask first)."""
    drv = app[0]
    layout(drv, "agte")
    n_tops = len(drv.execute_script(TOPS))
    for k in range(n_tops):
        open_top(drv, k, real=True)
        first = drv.execute_script("""return [...document.querySelector('.menu').querySelectorAll('.menu-item[data-cmd-id]')]
            .find((b) => b.getAttribute('aria-disabled') !== 'true')""")
        if first is None:
            dismiss(drv)
            continue
        cmd = first.get_attribute("data-cmd-id")
        before = now(drv)
        menu_click(drv, first)
        wait_ran(drv, before, cmd, 2)
        dismiss(drv)
        layout(drv, "agte")
    # a toggle: View > Show Unit Cell
    view = [m["label"] for m in drv.execute_script("return window.__mace.menus()")].index("View")
    was = drv.execute_script("return window.__mace.viewState().cell")
    open_top(drv, view, real=True)
    menu_click(drv, drv.find_element("css selector", '.menu [data-cmd-id="view.cell"]'))
    H.wait(drv, "return window.__mace.viewState().cell === arguments[0]", 3, not was)
    drv.execute_script("window.__mace.run('view.cell')")
    # a dialog: File > Settings...
    open_top(drv, 0, real=True)
    menu_click(drv, drv.find_element("css selector", '.menu [data-cmd-id="file.settings"]'))
    H.wait(drv, "return !!document.querySelector('dialog#settings[open]')", 10)
    dismiss(drv)
    # destructive: Close All with a modified document asks, and Cancel keeps it
    drv.execute_script("window.__mace.editor.select([0]); window.__mace.run('edit.delete')")
    H.wait(drv, "return window.__mace.docs().some((d) => d.dirty)", 10)
    open_top(drv, 0, real=True)
    menu_click(drv, drv.find_element("css selector", '.menu [data-cmd-id="file.closeAll"]'))
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    alert = WebDriverWait(drv, 5).until(EC.alert_is_present())
    assert "unsaved changes" in alert.text
    alert.dismiss()                       # Cancel
    time.sleep(0.3)
    assert drv.execute_script("return window.__mace.docs().length") >= 1
    drv.execute_script("window.__mace.closeAll()")


@pytest.mark.skipif(not os.environ.get("MACE_GUI_SLOW"), reason="the full real-click menu sweep: MACE_GUI_SLOW=1")
def test_menus_real_click_all(app):
    """Real-click every enabled entry of every top menu (run once before a release)."""
    drv = app[0]
    n_tops = len(drv.execute_script(TOPS))
    for k in range(n_tops):
        layout(drv, "agte")
        open_top(drv, k, real=True)
        n = len(drv.execute_script("return document.querySelector('.menu').querySelectorAll('.menu-item[data-cmd-id]')"))
        dismiss(drv)
        for i in range(n):
            layout(drv, "agte")
            open_top(drv, k, real=True)
            b = drv.execute_script("return document.querySelector('.menu').querySelectorAll('.menu-item[data-cmd-id]')[arguments[0]]", i)
            if b.get_attribute("aria-disabled") == "true":
                dismiss(drv)
                continue
            cmd = b.get_attribute("data-cmd-id")
            before = now(drv)
            menu_click(drv, b)
            wait_ran(drv, before, cmd, 3)
            # anything that asks (quit, close, delete) is canceled
            ActionChains(drv).send_keys(Keys.ESCAPE).perform()
            dismiss(drv)
