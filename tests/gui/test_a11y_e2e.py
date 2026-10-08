"""Accessibility gate (plan section 5.4), in headless Firefox through the real
`mace gui`.

* axe-core (axe-selenium-python, test-only) on every main screen and dialog,
  light and dark: no serious or critical WCAG 2 A/AA violation;
* a keyboard walkthrough: Tab reaches the menu bar, the explorer, the tab
  strip, the toolbar and the status bar, focus always moves (no trap), every
  dialog closes on Esc and gives focus back to what opened it;
* every icon-only button has an accessible name;
* the 3D views have a text alternative, and the atoms table is a keyboard stop;
* reduced motion (Firefox's ui.prefersReducedMotion) stops every animation;
* the logo is named "MACE" once per place, its decorative copies are hidden.

The token contrast check is tests/gui/test_tokens_contrast.py (no browser).
"""
import sys
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
from selenium.webdriver.common.keys import Keys  # noqa: E402

sys.path.insert(0, str(H.REPO_ROOT / "tests"))

# The 3D views are WebGL canvases: axe cannot read them, and their text
# alternative is checked below (role and label on their containers).
AXE_EXCLUDE = [["#viewer-main canvas"], ["#viewer-before canvas"], ["#viewer-after canvas"], [".fm-view canvas"]]
AXE_OPTIONS = '{"runOnly": {"type": "tag", "values": ["wcag2a", "wcag2aa"]}}'


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("a11y")
    root = H.examples_copy(tmp)
    try:
        import plotly  # noqa: F401
        from test_gui_results import write_band
        (root / "results").mkdir()
        write_band(root / "results")
    except ImportError:
        pass
    a = U.start_app(tmp, root=root)
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    return app


def _axe(drv, label):
    axe_mod = pytest.importorskip("axe_selenium_python")
    axe = axe_mod.Axe(drv)
    axe.inject()
    ctx = {"exclude": AXE_EXCLUDE}
    res = drv.execute_async_script(
        "const done = arguments[arguments.length - 1];"
        f"axe.run({__import__('json').dumps(ctx)}, {AXE_OPTIONS}).then((r) => done(r), (e) => done({{error: String(e)}}))")
    assert "error" not in res, res
    bad = [v for v in res["violations"] if v.get("impact") in ("serious", "critical")]
    return [(label, v["id"], v["impact"], [n["target"] for n in v["nodes"]][:5]) for v in bad]


def _theme(drv, theme):
    drv.execute_script("window.__mace.settings.set('theme', arguments[0])", theme)
    time.sleep(0.2)


def _close_dialogs(drv):
    drv.execute_script("document.querySelectorAll('dialog[open]').forEach((d) => d.close());"
                       "document.querySelectorAll('.menu').forEach((m) => m.remove())")


SCREENS = [
    ("welcome", None),
    ("structure", "open:structures/MgO.cif"),
    ("trajectory", "open:outputs/Ag2Br3_opt_unfinished.out"),
    ("database", "open:materials.db"),
    ("band", "open:results/mgo_band.BAND.DAT"),
    ("settings", "run:file.settings"),
    ("about", "run:help.about"),
    ("docs", "run:help.docs"),
    ("shortcuts", "run:help.shortcuts"),
    ("save-as", "run:file.saveAs"),
    ("new-structure", "run:file.new"),
    ("supercell", "run:build.supercell"),
    ("slab", "run:build.slab"),
    ("clean", "run:build.clean"),
    ("setup", "run:calc.setup"),
    ("tasks", "bottom:tasks"),
    ("menu", "menu:Build"),
]


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_axe_finds_no_serious_violation(fresh, theme):
    drv, root = fresh[0], fresh[2]
    found = []
    for label, how in SCREENS:
        _close_dialogs(drv)
        _theme(drv, theme)
        if how is None:
            drv.execute_script("window.__mace.closeAll()")
        elif how.startswith("open:"):
            path = how[5:]
            if not (root / path).exists():
                continue
            U.open_doc(drv, path, wait_atoms=path.endswith(".cif"))
            time.sleep(0.8)
        elif how.startswith("run:"):
            if how != "run:file.settings" and how != "run:help.about" and how != "run:help.docs" \
                    and how != "run:help.shortcuts" and how != "run:file.new":
                U.open_doc(drv, "structures/MgO.cif")
            drv.execute_script("window.__mace.run(arguments[0])", how[4:])
            time.sleep(1.0)
            if len(drv.window_handles) > 1:                     # help.docs opened GitHub in a tab
                main = drv.window_handles[0]
                for h in drv.window_handles[1:]:
                    drv.switch_to.window(h)
                    drv.close()
                drv.switch_to.window(main)
        elif how.startswith("bottom:"):
            drv.execute_script("document.querySelector('.bottom-tab[data-bottom=\"' + arguments[0] + '\"]').click()", how[7:])
            time.sleep(0.3)
        elif how.startswith("menu:"):
            drv.execute_script(
                "const b = [...document.querySelectorAll('.menu-top')].find((x) => x.textContent === arguments[0]);"
                "b.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true, button: 0}))", how[5:])
            time.sleep(0.3)
        U.shot(drv, f"a11y_{theme}_{label}")
        found += _axe(drv, label)
    _close_dialogs(drv)
    dump = __import__("os").environ.get("MACE_A11Y_DUMP")
    if dump:
        with open(f"{dump}_{theme}.json", "w") as fh:
            __import__("json").dump(found, fh, indent=1)
    assert not found, found


def test_icon_only_buttons_have_names(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    unnamed = drv.execute_script("""
      return [...document.querySelectorAll('button, [role=button]')].filter((b) => b.offsetParent !== null)
        .filter((b) => !b.textContent.trim())
        .filter((b) => !(b.getAttribute('aria-label') || b.getAttribute('title') || b.getAttribute('aria-labelledby')))
        .map((b) => b.outerHTML.slice(0, 160));""")
    assert not unnamed, unnamed


def _region(drv):
    return drv.execute_script("""
      const a = document.activeElement; if (!a || a === document.body) return 'body';
      for (const [sel, name] of [['#menubar', 'menubar'], ['#toolbar', 'toolbar'], ['.explorer, #explorer', 'explorer'],
                                 ['#doc-tabs', 'tabs'], ['.statusbar, #statusbar', 'status'], ['#inspector', 'inspector'],
                                 ['.bottom, #bottom', 'bottom'], ['.views', 'views']])
        if (a.closest(sel)) return name;
      return a.tagName.toLowerCase() + (a.id ? '#' + a.id : '');""")


def test_tab_reaches_every_region_without_a_trap(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("document.activeElement && document.activeElement.blur(); document.body.focus()")
    seen, last, stuck = [], None, 0
    want = {"menubar", "toolbar", "explorer", "tabs", "status"}
    for _ in range(600):
        if want <= set(seen):
            break
        U.keys(drv, Keys.TAB)
        el = drv.execute_script("return document.activeElement")
        stuck = stuck + 1 if el == last else 0
        assert stuck < 3, f"focus stuck on {drv.execute_script('return document.activeElement.outerHTML.slice(0, 120)')}"
        last = el
        seen.append(_region(drv))
        # a visible focus indicator: the element looks different focused than not
        ring = drv.execute_script("""const a = document.activeElement; if (!a || a === document.body) return true;
          const look = () => { const cs = getComputedStyle(a); return [cs.outlineStyle, cs.outlineWidth, cs.outlineColor,
            cs.boxShadow, cs.backgroundColor, cs.borderColor, cs.color].join('|'); };
          const on = look(); a.blur(); const off = look(); a.focus();
          return on !== off;""")
        assert ring, drv.execute_script("return document.activeElement.outerHTML.slice(0, 160)")
    for region in ("menubar", "toolbar", "explorer", "tabs", "status"):
        assert region in seen, (region, sorted(set(seen)))
    # and back with Shift+Tab: focus moves too
    before = drv.execute_script("return document.activeElement")
    U.keys(drv, Keys.SHIFT, Keys.TAB)
    assert drv.execute_script("return document.activeElement") != before


DIALOG_COMMANDS = ["file.settings", "help.about", "help.docs", "help.shortcuts", "help.tunnel", "file.saveAs",
                   "file.new", "build.supercell", "build.slab", "build.symmetrize", "build.clean", "edit.editCell"]


@pytest.mark.parametrize("cmd", DIALOG_COMMANDS)
def test_every_dialog_closes_on_esc_and_returns_focus(fresh, cmd):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("document.getElementById('theme-btn').focus()")
    drv.execute_script("window.__mace.run(arguments[0])", cmd)
    H.wait(drv, "return document.querySelector('dialog[open]') !== null", 15)
    main = drv.window_handles[0]
    for h in drv.window_handles[1:]:
        drv.switch_to.window(h)
        drv.close()
    drv.switch_to.window(main)
    time.sleep(0.3)
    inside = drv.execute_script("const d = document.querySelector('dialog[open]'); return d.contains(document.activeElement)")
    assert inside, cmd
    U.keys(drv, Keys.ESCAPE)
    H.wait(drv, "return document.querySelector('dialog[open]') === null", 10)
    assert drv.execute_script("return document.activeElement.id") == "theme-btn", cmd
    U.no_errors(drv)


def test_3d_views_have_a_text_alternative_and_the_atoms_table_is_a_stop(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    label = drv.execute_script("return document.getElementById('viewer-main').closest('.viewer-wrap').getAttribute('aria-label')")
    assert "MgO" in label and "Fm-3m" in label and "4.212" in label and "Atoms tab" in label, label
    assert drv.execute_script("return document.querySelector('.bottom-tab[data-bottom=\"atoms\"]').offsetParent !== null")
    drv.execute_script("document.querySelector('.bottom-tab[data-bottom=\"atoms\"]').focus()")
    U.keys(drv, Keys.ENTER)
    assert drv.execute_script("return document.querySelector('.bottom-tab.active').dataset.bottom") == "atoms"
    U.open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out", wait_atoms=False)
    H.wait(drv, "const h = document.getElementById('opt-viewers'); return h && h.getAttribute('role') === 'img'", 30)
    assert "Optimization" in drv.execute_script("return document.getElementById('opt-viewers').getAttribute('aria-label')")


def test_logo_is_named_once_per_place(fresh):
    drv = fresh[0]
    drv.execute_script("window.__mace.closeAll()")
    time.sleep(0.3)
    brand = drv.execute_script("""const b = document.querySelector('.brand-mark');
      return [b.getAttribute('role'), b.getAttribute('aria-label'), b.querySelector('img').getAttribute('alt'),
              b.querySelector('svg').getAttribute('aria-hidden'), b.querySelector('img').naturalWidth]""")
    assert brand[:4] == ["img", "MACE", "", "true"] and brand[4] > 0, brand
    imgs = drv.execute_script("""return [...document.querySelectorAll('.welcome-logo img')]
      .map((i) => [i.alt, i.naturalWidth, getComputedStyle(i).display !== 'none'])""")
    assert [i[0] for i in imgs] == ["MACE", "MACE"] and all(i[1] > 0 for i in imgs)
    assert sum(1 for i in imgs if i[2]) == 1                     # one per theme is shown


def test_reduced_motion_stops_every_animation(tmp_path):
    root = H.examples_copy(tmp_path)
    proc, base, _ = H.start_server(root, tmp_path / "xdg", MACE_GUI_TEST_HOOKS="1")
    drv = None
    try:
        drv = H.firefox(prefs={"ui.prefersReducedMotion": 1})
        H.load(drv, base)
        assert drv.execute_script("return matchMedia('(prefers-reduced-motion: reduce)').matches")
        U.open_doc(drv, "structures/MgO.cif")
        drv.execute_script("window.__mace.run('build.supercell')")
        time.sleep(0.5)
        long = drv.execute_script("""
          const out = [];
          for (const el of document.querySelectorAll('*')) {
            const cs = getComputedStyle(el);
            const t = Math.max(...cs.transitionDuration.split(',').map(parseFloat));
            const a = cs.animationName !== 'none' ? Math.max(...cs.animationDuration.split(',').map(parseFloat)) : 0;
            if (t > 0.01 || a > 0.01) out.push(el.tagName + '.' + el.className + ' ' + cs.transitionDuration + ' ' + cs.animationDuration);
          }
          return out.slice(0, 10);""")
        assert not long, long
        # Settings › Reduce motion › Off ("full") overrides the OS for those who want it
        drv.execute_script("window.__mace.settings.set('reduceMotion', 'off')")
        time.sleep(0.2)
        assert drv.execute_script("return document.documentElement.dataset.motion") == "full"
    finally:
        if drv is not None:
            drv.quit()
        proc.terminate()
        proc.wait(timeout=10)
