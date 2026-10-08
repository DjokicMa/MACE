"""Every control of Run Analysis and the database build, clicked for real
(PLAN_analysis C13.1, registered with the viewer's click-every-control
sweep, tests/gui/test_controls_click_e2e.py, at the merge of the 1.1.5
polish areas).

Two layouts: "1_dia OPT .out open with Run Analysis" (the run card row, the
Run Inspector card, Run Analysis's tabs and the plotkit toolbar, its menus
and its Plot Options popover on every tab) and "DB build dialog open" (its
chips and options; Create/Update Database is on the destructive list: it is
checked to be offered, never clicked, and the dialog is canceled). Each
control gets one real pointer click (ActionChains) in a fresh browser
profile, on a server started through mace_cli, with a real CRYSTAL output
copied from test/ (skipped without the corpus); no page error may follow.
"""
import time

import pytest

import guiharness as H
import runcorpus as RC

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")

from selenium.common.exceptions import StaleElementReferenceException, WebDriverException  # noqa: E402
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402

# every clickable control in a container that is visible, enabled and on top
CONTROLS = """
const box = document.querySelector(arguments[0]);
if (!box) return [];
const skip = arguments[1] ? [...document.querySelectorAll(arguments[1])] : [];
const out = [];
for (const b of box.querySelectorAll('button, select, input[type=checkbox], input[type=radio], [role=tab]')) {
  if (b.disabled || b.getAttribute('aria-disabled') === 'true' || skip.some((s) => s === b || s.contains(b))) continue;
  if (b.closest('[hidden]') || !b.getClientRects().length) continue;
  b.scrollIntoView({block: 'nearest'});           // a dialog or pane that scrolls: as a user would
  const r = b.getBoundingClientRect();
  if (!r.width || !r.height || getComputedStyle(b).visibility === 'hidden' || b.closest('[hidden]')) continue;
  const top = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
  if (!top || !(b === top || b.contains(top) || (top.tagName === 'LABEL' && top.contains(b)))) continue;
  const key = [b.tagName, b.id, b.className.replace(/\\b(active|on|open)\\b/g, '').trim(), b.dataset.tab || '', b.dataset.act || '',
               b.dataset.class || '', b.dataset.k || '', (b.textContent || '').trim().slice(0, 40)].join('|');
  out.push([key, b]);
}
if (arguments[2] !== undefined) { const f = out.find(([k]) => k === arguments[2]); return f ? f[1] : null; }
return out.map(([k]) => k);"""


def real_click(drv, el):
    ActionChains(drv).move_to_element(el).pause(0.05).click().perform()


def close_popups(drv):
    """Esc for an open plotkit menu or popover (never for a dialog: Esc closes it), then remove
    what is still open and the toasts."""
    if drv.execute_script("return !!document.querySelector('.pk-menu, .pk-pop')"):
        ActionChains(drv).send_keys(Keys.ESCAPE).perform()
        time.sleep(0.1)
    drv.execute_script("""document.querySelectorAll('.pk-menu, .pk-pop').forEach((m) => m.close ? m.close() : m.remove());
        document.querySelectorAll('#toasts > *').forEach((t) => t.remove());""")


def sweep(drv, container, skip="", keep_open="", settle=None):
    """Real-click every control of ``container`` once; returns the keys clicked.
    ``keep_open``: a dialog that must stay open; ``settle(drv)`` waits after each click
    (and may bring back the document a click left)."""
    seen, clicked = set(), []
    for _ in range(200):
        todo = [k for k in drv.execute_script(CONTROLS, container, skip) if k not in seen]
        if not todo:
            break
        key = todo[0]
        seen.add(key)
        for attempt in range(3):
            el = drv.execute_script(CONTROLS, container, skip, key)
            if el is None:
                break
            try:
                drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
                real_click(drv, el)
                clicked.append(key)
                break
            except (StaleElementReferenceException, WebDriverException):
                time.sleep(0.2)
        time.sleep(0.15)
        if settle is not None:
            settle(drv)
        # a menu or a popover the click opened: each of its entries is the toolbar's own business
        # (test_plotkit_e2e); here it must open without an error and close again
        close_popups(drv)
        if keep_open and not drv.execute_script(f"const d = document.querySelector('{keep_open}'); return !!(d && d.open)"):
            break
    assert not H.errors(drv), H.errors(drv)
    return clicked


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    RC.need(RC.DIA_OPT)
    tmp = tmp_path_factory.mktemp("clicksanalysis")
    root = tmp / "root"
    RC.copy_into(root / "runs", RC.DIA_OPT)
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1")
    drv = H.firefox(1280, 800)
    try:
        H.load(drv, base)
        yield drv, base, root
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


@H.needs_ase
def test_run_card_inspector_and_run_analysis(app):
    drv = app[0]
    rel = f"runs/{RC.DIA_OPT.name}"
    drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then(() => d(1), () => d(0))", rel)
    H.wait(drv, "return document.querySelectorAll('#o-energy circle.pt, #o-energy path.pt').length >= 2", 60)
    # the run card row (Open as Plot is last: it leaves the card)
    assert len(sweep(drv, ".o-energy-row", skip="#o-e-plot")) >= 4
    def back(d):
        # a card button that opens another document (Show in Output, Run Analysis): back to the run
        if (d.execute_script("const a = window.__mace.activeDoc(); return a && a.path") or "") != rel:
            d.execute_async_script("const k = arguments[1]; window.__mace.open(arguments[0]).then(() => k(1), () => k(0))", rel)
            H.wait(d, "return document.querySelectorAll('#o-energy circle.pt, #o-energy path.pt').length >= 2", 60)
    assert len(sweep(drv, "#inspector", settle=back)) >= 5
    real_click(drv, drv.find_element("css selector", "#o-e-plot"))
    H.wait(drv, "return window.__mace.runView && window.__mace.runView.state() && window.__mace.runView.state().drawn", 60)
    tabs = drv.execute_script("return [...document.querySelectorAll('#runplot-host .rp-tabs [role=tab]')].map((t) => t.dataset.tab)")
    assert len(tabs) >= 4, tabs
    for t in tabs:
        real_click(drv, drv.find_element("css selector", f'#runplot-host .rp-tabs [data-tab="{t}"]'))
        H.wait(drv, "const s = window.__mace.runView.state(); return s && s.tab === arguments[0]", 30, t)
        time.sleep(0.3)
        clicked = sweep(drv, "#runplot-host .pv-head")
        assert any("pk-more" in k for k in clicked) and any("pk-export" in k for k in clicked), (t, clicked)
        # the tab's own options next to the tabs
        sweep(drv, "#runplot-host .rp-tabopts")
        # Plot Options: the popover's own fields and buttons
        real_click(drv, drv.find_element("css selector", "#runplot-host .pk-more"))
        H.wait(drv, "return !!document.querySelector('#runplot-host .pk-pop')", 5)
        n = drv.execute_script("return document.querySelectorAll('#runplot-host .pk-pop button, #runplot-host .pk-pop input[type=checkbox]').length")
        for i in range(n):
            if not drv.execute_script("return !!document.querySelector('#runplot-host .pk-pop')", ):
                real_click(drv, drv.find_element("css selector", "#runplot-host .pk-more"))
                H.wait(drv, "return !!document.querySelector('#runplot-host .pk-pop')", 5)
            el = drv.execute_script("return document.querySelectorAll('#runplot-host .pk-pop button, #runplot-host .pk-pop input[type=checkbox]')[arguments[0]]", i)
            if el is None or not el.is_displayed() or not el.is_enabled():
                continue
            real_click(drv, el)
            time.sleep(0.15)
        close_popups(drv)
    assert not H.errors(drv), H.errors(drv)


def test_db_build_dialog(app):
    drv = app[0]
    drv.execute_script("window.__mace.closeAll()")
    # the dialog from the Explorer's folder menu (a real right-click)
    for attempt in range(5):
        try:
            folder = drv.find_element("css selector", '#file-list .folder[data-dir="runs"]')
            drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", folder)
            ActionChains(drv).context_click(folder).perform()
            break
        except StaleElementReferenceException:
            time.sleep(0.5)
    H.wait(drv, "return !!document.querySelector(`.menu .menu-item[data-cmd-id='explorer.dbBuild']`)", 5)
    real_click(drv, drv.find_element("css selector", ".menu .menu-item[data-cmd-id='explorer.dbBuild']"))
    H.wait(drv, "const d = document.querySelector('#db-build'); return d && d.open", 15)
    H.wait(drv, "const n = document.querySelector('#db-build .dbb-pnote'); return n && /outputs? in the folder/.test(n.textContent)", 120)
    # Create Database is destructive: offered, never clicked here; Cancel and Esc close the dialog
    go = drv.find_element("css selector", "#db-build .dbb-go")
    assert go.is_displayed() and go.text.startswith(("Create Database", "Update Database"))
    previewed = "const n = document.querySelector('#db-build .dbb-pnote'); return n && /outputs? in the folder/.test(n.textContent)"
    clicked = sweep(drv, "#db-build", skip="#db-build .dbb-go, #db-build [data-close]", keep_open="#db-build",
                    settle=lambda d: H.wait(d, previewed, 120))
    assert len(clicked) >= 5, clicked                  # the options and the file chips
    cancel = drv.execute_script("""return [...document.querySelectorAll('#db-build button')]
        .find((b) => /^Cancel$/.test(b.textContent.trim()) && b.offsetParent !== null) || null""")
    if cancel is not None:
        real_click(drv, cancel)
    else:
        ActionChains(drv).send_keys(Keys.ESCAPE).perform()
    H.wait(drv, "const d = document.querySelector('#db-build'); return !d || !d.open", 10)
    assert not (app[2] / "runs" / "materials.db").exists()
    assert not H.errors(drv), H.errors(drv)
