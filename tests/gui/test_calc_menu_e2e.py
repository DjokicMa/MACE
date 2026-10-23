"""Calculation › Regular Calculations and Properties Calculations (1.1.6 final)
in headless Firefox, and Calculation Setup's calculation types beside them.

The Calculation menu listed Mechanical Properties Deck and Phonopy Displacements
as two one-off entries, and the mechanics deck never said it needs a finished
run. Now the menu's top level has two submenus, the calculation types grouped
and numbered as the CLI's menus (opt2d12's 1 SP ... 7 CPHF and Phonopy; opt2d3's
1 BAND ... 7 ORBITALS), and an entry the file in front cannot have is grayed with
the reason (title and aria-describedby) and cannot be run. Setup's types show the
same groups, numbers and reasons: one table (setup/schema calc_groups).

The submenus: hover (after the intent delay), click (touch has no hover), the
keyboard (ArrowRight in, ArrowLeft out, Escape), flipped to the left at the
window's right edge. Repo data only (no corpus); no JavaScript error on the way.
"""
import shutil
import time

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")                 # the server reads the CIF with it
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402

REPO = H.REPO_ROOT
RUN = "Needs a finished OPT or SP output (.out)"
REGULAR = ["1 Single Point (SP)", "2 Optimization (OPT)", "3 Frequency (FREQ)",
           "4 Elastic, piezoelectric or photoelastic constants (ELASTIC)", "5 Equation of state (EOS)",
           "6 Quasi-harmonic approximation (QHA)", "7 Dielectric response, CPHF/KS (CPHF)",
           "8 Finite-displacement phonons with phonopy (PHONOPY)"]


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("calcmenu")
    root = tmp / "root"
    root.mkdir()
    shutil.copy(REPO / "examples" / "gui" / "structures" / "Si.cif", root / "Si.cif")
    for f in ("dia_opt.out", "dia_opt.d12"):
        shutil.copy(REPO / "tests" / "data" / "existing_calcs" / f, root / f)
    # the same optimization, not converged (no OPT END - CONVERGED)
    text = (root / "dia_opt.out").read_text()
    (root / "dia_open.out").write_text("\n".join(ln for ln in text.splitlines() if "OPT END - CONVERGED" not in ln) + "\n")
    shutil.copy(root / "dia_opt.d12", root / "dia_open.d12")
    proc, base, _ = H.start_server(root, tmp / "xdg")
    try:
        drv = H.firefox(1280, 900)
        H.load(drv, base)
    except BaseException:
        proc.terminate()
        raise
    try:
        yield drv, base, root
    finally:
        drv.quit()
        proc.terminate()


def open_doc(drv, rel):
    drv.execute_script("window.__mace.closeAll()")
    drv.execute_async_script("const k = arguments[1]; window.__mace.open(arguments[0]).then(() => k(1), () => k(0))", rel)
    H.wait(drv, "const a = window.__mace.activeDoc(); return a && a.path === arguments[0] && (a.fileKind !== 'out' || !!a.data)", 60, rel)


def close_menus(drv):
    drv.execute_script("document.body.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true, button: 0}))")
    H.wait(drv, "return !document.querySelector('.menu')", 5)


def open_calc_menu(drv):
    close_menus(drv)
    top = drv.find_element("xpath", "//button[contains(@class,'menu-top') and normalize-space()='Calculation']")
    for _ in range(3):                                 # a click that lands during a relayout: once more
        top.click()
        try:
            H.wait(drv, "return !!document.querySelector('.menu[aria-label=Calculation]')", 3)
            break
        except AssertionError:
            close_menus(drv)
    H.wait(drv, "return !!document.querySelector('.menu[aria-label=Calculation]')", 1)
    return top


def row(drv, label, menu="Calculation"):
    return drv.find_element("xpath", f"//div[contains(@class,'menu') and @aria-label='{menu}']"
                                     f"//button[contains(@class,'menu-item')][.//span[@class='lbl' and normalize-space()='{label}']]")


def entries(drv, menu):
    return drv.execute_script("""const m = document.querySelector(`.menu[aria-label="${arguments[0]}"]`);
        return m ? [...m.querySelectorAll('.menu-item')].map((b) => ({label: b.querySelector('.lbl').textContent,
          disabled: b.getAttribute('aria-disabled') === 'true', title: b.title,
          desc: (b.getAttribute('aria-describedby') || '').split(/\\s+/).filter(Boolean)
            .map((id) => (document.getElementById(id) || {}).textContent).join(' ') || null})) : null""", menu)


def test_the_menu_top_level_holds_two_submenus(app):
    drv = app[0]
    open_doc(drv, "Si.cif")
    open_calc_menu(drv)
    labels = [e["label"] for e in entries(drv, "Calculation")]
    assert "Regular Calculations" in labels and "Properties Calculations" in labels, labels
    assert not any(x in labels for x in ("Mechanical Properties Deck…", "Phonopy Displacements…")), labels
    assert drv.execute_script("return [...document.querySelectorAll('.menu[aria-label=Calculation] [aria-haspopup=menu]')]"
                              ".map((b) => b.querySelector('.lbl').textContent)")[:2] == ["Regular Calculations", "Properties Calculations"]
    close_menus(drv)
    assert not H.errors(drv), H.errors(drv)


def test_hover_opens_the_flyout_and_a_cif_grays_what_needs_a_run(app):
    drv = app[0]
    open_doc(drv, "Si.cif")
    open_calc_menu(drv)
    # straight down the menu to the row, then a pause: the flyout opens after the intent delay
    r = row(drv, "Regular Calculations")
    ActionChains(drv).move_to_element_with_offset(r, int(-r.size["width"] / 2 + 16), 0).pause(0.6).perform()
    H.wait(drv, "return !!document.querySelector('.menu[aria-label=\"Regular Calculations\"]')", 5)
    assert r.get_attribute("aria-expanded") == "true"
    es = entries(drv, "Regular Calculations")
    assert [e["label"] for e in es] == REGULAR, es
    assert [e["disabled"] for e in es] == [False, False, False, True, True, True, True, True], es
    why = f"{RUN}; this is a structure"
    assert all(e["title"].endswith(why) and e["desc"] == why for e in es[3:]), es
    # a grayed entry cannot be run: the click is ignored and the menu stays open
    drv.execute_script("window.__mace.spyRun(true)")
    try:
        el = row(drv, "4 Elastic, piezoelectric or photoelastic constants (ELASTIC)", "Regular Calculations")
        ActionChains(drv).move_to_element(el).pause(0.05).click().perform()
        time.sleep(0.3)
        assert drv.execute_script("return window.__maceRan") == []
        assert drv.execute_script("return !!document.querySelector('.menu[aria-label=\"Regular Calculations\"]')")
        # Enter on it does nothing either
        drv.execute_script("arguments[0].focus()", el)
        ActionChains(drv).send_keys(Keys.ENTER).perform()
        time.sleep(0.2)
        assert drv.execute_script("return window.__maceRan") == []
        # an enabled one runs Setup at its type
        el = row(drv, "1 Single Point (SP)", "Regular Calculations")
        ActionChains(drv).move_to_element(el).pause(0.05).click().perform()
        H.wait(drv, "return window.__maceRan.length === 1", 5)
        assert drv.execute_script("return window.__maceRan") == ["calc.setupType"]
    finally:
        drv.execute_script("window.__mace.spyRun(false)")
    assert not drv.execute_script("return !!document.querySelector('.menu')")      # a run closes every menu
    # Properties: every entry needs a run
    open_calc_menu(drv)
    row(drv, "Properties Calculations").click()
    H.wait(drv, "return !!document.querySelector('.menu[aria-label=\"Properties Calculations\"]')", 5)
    es = entries(drv, "Properties Calculations")
    assert [e["label"] for e in es][:2] == ["1 Electronic band structure (BAND)", "2 Density of states (DOSS)"], es
    assert len(es) == 7 and all(e["disabled"] and e["desc"] == f"{RUN}; this is a structure" for e in es), es
    close_menus(drv)
    assert not H.errors(drv), H.errors(drv)


def test_click_and_keyboard_open_and_close_the_flyouts(app):
    drv = app[0]
    open_doc(drv, "dia_opt.out")
    open_calc_menu(drv)
    # a click (a tap) opens it, focus in the flyout's first entry
    row(drv, "Properties Calculations").click()
    H.wait(drv, "const m = document.querySelector('.menu[aria-label=\"Properties Calculations\"]'); return m && m.contains(document.activeElement)", 5)
    # ArrowLeft back to its row, ArrowUp to Regular, ArrowRight into it: one submenu at a time
    ActionChains(drv).send_keys(Keys.ARROW_LEFT).perform()
    H.wait(drv, "return !document.querySelector('.menu[aria-label=\"Properties Calculations\"]')", 5)
    assert drv.execute_script("return document.activeElement.querySelector('.lbl').textContent") == "Properties Calculations"
    ActionChains(drv).send_keys(Keys.ARROW_UP).perform()
    assert drv.execute_script("return document.activeElement.querySelector('.lbl').textContent") == "Regular Calculations"
    ActionChains(drv).send_keys(Keys.ARROW_RIGHT).perform()
    H.wait(drv, "const m = document.querySelector('.menu[aria-label=\"Regular Calculations\"]'); return m && m.contains(document.activeElement)", 5)
    assert drv.execute_script("return document.querySelectorAll('.menu').length") == 2
    assert drv.execute_script("return document.activeElement.querySelector('.lbl').textContent") == REGULAR[0]
    ActionChains(drv).send_keys(Keys.ARROW_DOWN).perform()
    assert drv.execute_script("return document.activeElement.querySelector('.lbl').textContent") == REGULAR[1]
    # ArrowLeft: back to the parent row, the flyout closed
    ActionChains(drv).send_keys(Keys.ARROW_LEFT).perform()
    H.wait(drv, "return !document.querySelector('.menu[aria-label=\"Regular Calculations\"]')", 5)
    assert drv.execute_script("return document.activeElement.querySelector('.lbl').textContent") == "Regular Calculations"
    # Enter opens it again; Escape closes the flyout, a second Escape the menu
    ActionChains(drv).send_keys(Keys.ENTER).perform()
    H.wait(drv, "return !!document.querySelector('.menu[aria-label=\"Regular Calculations\"]')", 5)
    ActionChains(drv).send_keys(Keys.ESCAPE).perform()
    H.wait(drv, "return document.querySelectorAll('.menu').length === 1", 5)
    ActionChains(drv).send_keys(Keys.ESCAPE).perform()
    H.wait(drv, "return !document.querySelector('.menu')", 5)
    # with a finished OPT in front every entry is enabled; Properties › BAND opens Setup at BAND
    open_calc_menu(drv)
    row(drv, "Properties Calculations").click()
    H.wait(drv, "return !!document.querySelector('.menu[aria-label=\"Properties Calculations\"]')", 5)
    assert not any(e["disabled"] for e in entries(drv, "Properties Calculations"))
    row(drv, "1 Electronic band structure (BAND)", "Properties Calculations").click()
    H.wait(drv, "const s = window.__mace.setup && window.__mace.setup.state(); return s && s.calc === 'BAND' && s.mode === 'opt2d3'", 60)
    assert drv.execute_script("return document.querySelector('#inspector .su-calc [data-calc=BAND]').getAttribute('aria-checked')") == "true"
    assert not H.errors(drv), H.errors(drv)


def test_the_flyout_flips_left_at_the_window_edge(app):
    drv = app[0]
    open_doc(drv, "dia_opt.out")

    def rects():
        return drv.execute_script("""const p = document.querySelector('.menu[aria-label=Calculation]').getBoundingClientRect();
            const s = document.querySelector('.menu[aria-label="Regular Calculations"]').getBoundingClientRect();
            return {p: [p.left, p.right], s: [s.left, s.right, s.top, s.bottom], w: innerWidth, h: innerHeight}""")
    try:
        open_calc_menu(drv)
        row(drv, "Regular Calculations").click()
        H.wait(drv, "return !!document.querySelector('.menu[aria-label=\"Regular Calculations\"]')", 5)
        r = rects()
        assert r["s"][0] >= r["p"][1] - 16, r         # room on the right: beside the menu (its padding overlapped)
        close_menus(drv)
        # a window whose right edge the flyout would cross: it opens on the menu's left
        drv.set_window_size(1000, 560)
        H.wait(drv, "return innerWidth < 1100", 5)
        time.sleep(0.5)                                # the layout settles at the new size
        open_calc_menu(drv)
        row(drv, "Regular Calculations").click()
        H.wait(drv, "return !!document.querySelector('.menu[aria-label=\"Regular Calculations\"]')", 5)
        r = rects()
        assert r["s"][0] < r["p"][0], r
        assert 0 <= r["s"][0] and r["s"][1] <= r["w"] and 0 <= r["s"][2] and r["s"][3] <= r["h"], r
        # a short window: the flyout's height is capped and it scrolls
        assert drv.execute_script("const m = document.querySelector('.menu[aria-label=\"Regular Calculations\"]');"
                                  "return getComputedStyle(m).overflowY === 'auto' && m.offsetHeight <= innerHeight")
    finally:
        close_menus(drv)
        drv.set_window_size(1280, 900)
    assert not H.errors(drv), H.errors(drv)


def test_setup_grays_the_same_types_with_the_same_reasons(app):
    drv = app[0]
    open_doc(drv, "Si.cif")
    drv.execute_script("if (!document.querySelector('#inspector .setup')) window.__mace.run('calc.setup')")
    H.wait(drv, "return document.querySelectorAll('#inspector .su-calc [data-calc]').length === 15", 60)
    caps = drv.execute_script("return [...document.querySelectorAll('#inspector .su-calc-cap')].map((c) => c.firstChild.textContent.trim())")
    assert caps == ["Regular Calculations", "Properties Calculations"], caps
    st = drv.execute_script("""return [...document.querySelectorAll('#inspector .su-calc [data-calc]')].map((b) => [b.dataset.calc,
        b.dataset.n || '', b.getAttribute('aria-disabled') === 'true', b.getAttribute('aria-disabled') === 'true' ? b.title : ''])""")
    why = f"{RUN}; this is a structure"
    assert st[:4] == [["SP", "1", False, ""], ["OPT", "2", False, ""], ["FREQ", "3", False, ""], ["ELASTIC", "4", True, why]], st
    assert st[7] == ["PHONOPY", "8", True, why] and st[8] == ["BAND", "1", True, why] and st[14] == ["ORBITALS", "7", True, why], st
    # a grayed type cannot be picked; the menu's entry says the same
    drv.execute_script("document.querySelector('#inspector .su-calc [data-calc=BAND]').click()")
    time.sleep(0.3)
    assert drv.execute_script("return window.__mace.setup.state().calc") == "OPT"
    # Calculation › Mechanical Properties Deck on a CIF now says what it needs, as Phonopy Displacements does
    drv.execute_script("document.querySelectorAll('#toasts > *').forEach((t) => t.remove()); window.__mace.run('calc.mechanics')")
    H.wait(drv, "return /Needs a finished OPT or SP output/.test(document.querySelector('#toasts') && document.querySelector('#toasts').textContent)", 10)
    drv.execute_script("document.querySelectorAll('#toasts > *').forEach((t) => t.remove()); window.__mace.run('calc.phonopy')")
    H.wait(drv, "return /Needs a finished OPT or SP output/.test(document.querySelector('#toasts') && document.querySelector('#toasts').textContent)", 10)
    # an optimization that has not converged: the properties and modules grayed with that reason
    open_doc(drv, "dia_open.out")
    H.wait(drv, "const b = document.querySelector('#inspector .su-calc [data-calc=BAND]'); return b && b.getAttribute('aria-disabled') === 'true'", 60)
    t = drv.execute_script("return document.querySelector('#inspector .su-calc [data-calc=BAND]').title")
    assert t == f"{RUN}; this optimization has not converged (no OPT END - CONVERGED)", t
    assert drv.execute_script("return document.querySelector('#inspector .su-calc [data-calc=SP]').getAttribute('aria-disabled')") is None
    open_calc_menu(drv)
    row(drv, "Properties Calculations").click()
    H.wait(drv, "return !!document.querySelector('.menu[aria-label=\"Properties Calculations\"]')", 5)
    assert all(e["disabled"] and e["desc"] == t for e in entries(drv, "Properties Calculations"))
    close_menus(drv)
    # the converged one: everything enabled in Setup
    open_doc(drv, "dia_opt.out")
    H.wait(drv, "const b = document.querySelector('#inspector .su-calc [data-calc=BAND]'); return b && b.getAttribute('aria-disabled') !== 'true'", 60)
    assert not drv.execute_script("return document.querySelector('#inspector .su-calc [aria-disabled=true]')")
    assert not H.errors(drv), H.errors(drv)
