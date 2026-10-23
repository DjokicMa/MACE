"""Bonds the structure viewer draws, in a real browser on a server started
through mace_cli (1.1.6).

JARVIS-DFT alloys and intermetallics drew no bond at all (139 of 300 random
JARVIS-DFT CIFs): metal-metal pairs are not bonded so that an ionic solid's
cations stay apart, and a structure of metals only had nothing else. It keeps
its metal-metal bonds now; a structure with a nonmetal draws what it drew."""
import shutil
import time

import pytest

import guiharness as H
from conftest import REPO_ROOT

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")

JARVIS = REPO_ROOT / "tests" / "fixtures" / "jarvis"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bonds")
    root = H.examples_copy(tmp)
    shutil.copytree(JARVIS, root / "jarvis")
    xdg = tmp / "xdg"
    proc, base, _ = H.start_server(root, xdg)
    drv = H.firefox(1400, 900)
    try:
        H.load(drv, base)
        yield drv, base, root, xdg
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


@pytest.fixture
def fresh(app):
    drv = app[0]
    drv.execute_script("window.__mace.closeAll(); window.__mace.settings.resetAll(); window.__maceErrors.length = 0;"
                       "document.querySelectorAll('dialog[open]').forEach((d) => d.close());")
    return app


def open_structure(drv, path):
    kind = drv.execute_async_script(
        "const done = arguments[1]; window.__mace.open(arguments[0]).then((d) => done(d && d.kind))", path)
    # an open document is activated (open gives no new document then)
    assert kind == "structure" or drv.execute_script("const d = window.__mace.doc(); return !!d && d.path.endsWith(arguments[0])", path)
    H.wait(drv, "return window.__mace.viewport && window.__mace.viewport.displayed() > 0", 20)
    time.sleep(0.3)


def bond_kinds(drv):
    return drv.execute_script("""const sc = window.__mace.viewport.scene(), d = sc.display, s = window.__mace.doc().structure;
      const p = sc.pairs || [], kinds = {};
      for (let q = 0; q < p.length; q += 2) { const k = [s.symbols[d.base[p[q]]], s.symbols[d.base[p[q + 1]]]].sort().join('-'); kinds[k] = (kinds[k] || 0) + 1; }
      return kinds;""")


@pytest.mark.parametrize("name,pairs", [("JVASP-100318.cif", {"Pd-Zn"}), ("JVASP-106626.cif", {"Ag-Yb"}),
                                        ("JVASP-76747.cif", {"Ga-Zn"})])
def test_jarvis_alloys_draw_their_bonds(fresh, name, pairs):
    drv = fresh[0]
    open_structure(drv, f"jarvis/{name}")
    k = bond_kinds(drv)
    assert sum(k.values()) > 0 and pairs <= set(k), k
    assert not H.errors(drv), H.errors(drv)


def test_a_compound_with_a_nonmetal_is_unchanged(fresh):
    drv = fresh[0]
    open_structure(drv, "jarvis/JVASP-100029.cif")          # BiSbPd2: Pd-Sb only, no Pd-Pd or Bi-Pd
    assert bond_kinds(drv) == {"Pd-Sb": 6}
    assert not H.errors(drv), H.errors(drv)


# ---------------------------------------------------------------- Calculate Bonds… (1.1.6)

READY = "const d = document.getElementById('bonds-dialog'); return !!(d && d.open && d.dataset.ready)"


def preview(drv):
    """The dialog's last preview, once the one for its current values is in (a change clears the mark)."""
    H.wait(drv, READY, 20)
    return drv.execute_script("return document.getElementById('bonds-dialog').__preview")


UNREADY = "delete document.getElementById('bonds-dialog').dataset.ready;"


def set_field(drv, sel, value):
    drv.execute_script(UNREADY + """const el = document.querySelector('#bonds-dialog ' + arguments[0]);
      if (el.type === 'checkbox') el.checked = arguments[1]; else el.value = arguments[1];
      el.dispatchEvent(new Event('input', {bubbles: true})); el.dispatchEvent(new Event('change', {bubbles: true}));""", sel, value)


def click(drv, v):
    drv.execute_script(UNREADY + "document.querySelector('#bonds-dialog [data-v=\"' + arguments[0] + '\"]').click()", v)


def drawn_count(drv):
    return drv.execute_script("const p = window.__mace.viewport.scene().pairs; return p ? p.length / 2 : 0")


def open_dialog(drv):
    drv.execute_script("window.__mace.run('build.calculateBonds')")
    H.wait(drv, READY, 20)


def test_the_dialog_recalculates_the_bonds(fresh):
    """Tolerance, a pair rule that removes and one that adds bonds, Reset; the preview's drawn count is the
    number the view draws after Recalculate, and the first draw agrees with the rule the dialog shows."""
    drv, base, root, xdg = fresh
    open_structure(drv, "structures/MgO.cif")
    first = drawn_count(drv)
    assert first > 0 and set(bond_kinds(drv)) == {"Mg-O"}
    open_dialog(drv)
    p = preview(drv)
    assert p["drawn"]["bonds"] == first == p["drawn"]["now"], p                    # the first draw and the rule agree
    assert [x["pair"] for x in p["pairs"]] == ["Mg-O"] and p["values"]["bondTolerance"] == 1.15
    # a pair rule that removes: never Mg-O
    click(drv, "add")
    set_field(drv, "#bd-rules tr:last-child [data-r=a]", "mg")
    set_field(drv, "#bd-rules tr:last-child [data-r=b]", "O")
    set_field(drv, "#bd-rules tr:last-child [data-r=never]", True)
    p = preview(drv)
    assert p["cell"] == 0 and p["drawn"]["bonds"] == 0 and p["values"]["bondPairs"] == {"Mg-O": {"never": True}}, p
    assert drawn_count(drv) == first                                                # nothing changes before Recalculate
    click(drv, "apply")
    H.wait(drv, "const p = window.__mace.viewport.scene().pairs; return !p || p.length === 0", 20)
    assert drv.execute_script("return window.__mace.display.of(window.__mace.activeDoc()).bondPairs") == {"Mg-O": {"never": True}}
    # a pair rule that adds: Mg-Mg up to 3.1 Å (a compound's metals have none by the tolerance)
    set_field(drv, "#bd-rules tr:last-child [data-r=never]", False)
    set_field(drv, "#bd-rules tr:last-child [data-r=b]", "Mg")
    set_field(drv, "#bd-rules tr:last-child [data-r=max]", "3.1")
    p = preview(drv)
    click(drv, "apply")
    H.wait(drv, "return 'Mg-Mg' in " + bond_kinds_js(), 20)
    assert set(bond_kinds(drv)) == {"Mg-O", "Mg-Mg"} and drawn_count(drv) == p["drawn"]["bonds"], (bond_kinds(drv), p)
    # tolerance: the preview follows, and Recalculate (Enter in the field) draws what it said
    click(drv, "remove")
    set_field(drv, "[data-k=bondTolerance]", "1.4")
    p = preview(drv)
    drv.execute_script(UNREADY + "document.querySelector('#bonds-dialog [data-k=bondTolerance]').dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', bubbles: true}))")
    H.wait(drv, f"const p = window.__mace.viewport.scene().pairs; return p && p.length / 2 === {p['drawn']['bonds']}", 20)
    assert drv.execute_script("return window.__mace.display.overrides(window.__mace.activeDoc())") == {"bondTolerance": 1.4}
    # Reset to Defaults: the document follows Settings again, and draws its first bonds
    click(drv, "reset")
    H.wait(drv, f"const p = window.__mace.viewport.scene().pairs; return p && p.length / 2 === {first}", 20)
    assert drv.execute_script("return window.__mace.display.overrides(window.__mace.activeDoc())") == {}
    assert drv.execute_script("return document.querySelector('#bonds-dialog [data-k=bondTolerance]').value") == "1.15"
    # Esc closes it
    drv.execute_script("document.querySelector('#bonds-dialog').dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}))")
    H.wait(drv, "return !document.getElementById('bonds-dialog').open", 10)
    assert not H.errors(drv), H.errors(drv)


def bond_kinds_js():
    return """(() => { const sc = window.__mace.viewport.scene(), d = sc.display, s = window.__mace.doc().structure, p = sc.pairs || [], k = {};
      for (let q = 0; q < p.length; q += 2) k[[s.symbols[d.base[p[q]]], s.symbols[d.base[p[q + 1]]]].sort().join('-')] = 1; return k; })()"""


def test_criteria_persist_per_document_and_as_the_default(fresh):
    drv, base, root, xdg = fresh
    import json
    open_structure(drv, "structures/NaCl.cif")
    first = drawn_count(drv)
    open_dialog(drv)
    click(drv, "add")
    set_field(drv, "#bd-rules tr:last-child [data-r=a]", "Cl")
    set_field(drv, "#bd-rules tr:last-child [data-r=b]", "Na")
    set_field(drv, "#bd-rules tr:last-child [data-r=never]", True)
    preview(drv)
    click(drv, "apply")
    H.wait(drv, "const p = window.__mace.viewport.scene().pairs; return !p || p.length === 0", 20)
    time.sleep(1.2)                                   # the 500 ms debounce, then the POST
    data = json.loads((xdg / "mace" / "gui-docprefs.json").read_text())
    vals = [v["values"] for docs in data["kinds"]["display"].values() for p, v in docs.items() if p.endswith("structures/NaCl.cif")]
    assert vals and vals[0]["bondPairs"] == {"Cl-Na": {"never": True}}, vals
    # another document keeps the defaults
    open_structure(drv, "structures/LiF.cif")
    assert drawn_count(drv) > 0
    # Save as Default: Settings hold the criteria (gui.json), the document's own go
    open_structure(drv, "structures/NaCl.cif")
    open_dialog(drv)
    set_field(drv, "[data-k=bondMin]", "0.2")
    preview(drv)
    click(drv, "default")
    drv.execute_async_script("const d = arguments[0]; window.__mace.settings.flush().then(d)")
    saved = json.loads((xdg / "mace" / "gui.json").read_text())["settings"]
    assert saved["bondPairs"] == {"Cl-Na": {"never": True}} and saved["bondMin"] == 0.2, saved
    assert drv.execute_script("return window.__mace.display.overrides(window.__mace.activeDoc())") == {}
    assert drawn_count(drv) == 0
    # a bad stored value is refused by the server's check
    from mace.gui import settings as G
    assert G.check("bondPairs", {"Cl-Na": {"never": True}}) is None
    drv.execute_script("window.__mace.settings.resetAll()")
    H.wait(drv, f"const p = window.__mace.viewport.scene().pairs; return p && p.length / 2 === {first}", 20)
    assert not H.errors(drv), H.errors(drv)


def test_the_dialog_is_on_the_menus(fresh):
    """Build › Calculate Bonds…, the viewport's empty-space menu and Settings › Viewport › Bond tolerance."""
    drv = fresh[0]
    open_structure(drv, "structures/MgO.cif")
    menus = drv.execute_async_script("""const done = arguments[0];
      import(document.querySelector('script[src*="/js/shell.js"]').src.replace('shell.js', 'actions.js')).then((m) => done({
        build: m.MENUS.find((x) => x.label === 'Build').items, empty: m.CONTEXT.empty}))""")
    assert "build.calculateBonds" in menus["build"] and "build.calculateBonds" in menus["empty"]
    assert drv.execute_script("return window.__mace.menuReason('build.calculateBonds')") in (None, "")
    drv.execute_script("window.__mace.settings.set('bondTolerance', 1.15)")
    assert not H.errors(drv), H.errors(drv)


# ---------------------------------------------------------------- Clean Geometry holds the drawn bonds (1.1.6)

CLEAN = "const f = window.__mace.builderFrame(); return f && f.id === 'clean' && !f.busy && (!f.goDisabled || !!f.error) && f"


def test_clean_holds_the_bonds_the_view_draws(fresh):
    """Clean Geometry's springs are the bonds the view draws, by the same rule (structures.bond_pairs): an
    alloy cleans (it had "nothing to clean": Clean skipped every metal-metal pair), and the document's own
    Calculate Bonds… criteria go with the request (a rule that bonds no pair leaves nothing to clean)."""
    drv = fresh[0]
    open_structure(drv, "jarvis/JVASP-100318.cif")           # PdZn: metals only, Pd-Zn drawn
    assert "Pd-Zn" in bond_kinds(drv)
    drv.execute_script("window.__mace.run('build.clean')")
    f = H.wait(drv, CLEAN, 60)
    assert not f["error"] and not f["goDisabled"], f          # a proposal: the springs held the drawn bonds
    drv.execute_script("document.querySelector('dialog.builder-dialog [data-act=\"cancel\"]').click()")
    H.wait(drv, "return !window.__mace.builderFrame()", 10)
    drv.execute_script("""const D = window.__mace.display, d = window.__mace.activeDoc();
      D.set(d, 'bondPairs', {'Pd-Zn': {never: true}, 'Pd-Pd': {never: true}, 'Zn-Zn': {never: true}});""")
    H.wait(drv, "const p = window.__mace.viewport.scene().pairs; return !p || p.length === 0", 20)
    drv.execute_script("window.__mace.run('build.clean')")
    f = H.wait(drv, CLEAN, 60)
    assert f["goDisabled"] and "Nothing to clean" in f["error"], f
    # Bonds held (As shown by default): None runs the repulsion alone, and the choice is remembered
    held = "dialog.builder-dialog [data-k=\"bonds\"]"
    assert drv.execute_script(f"return document.querySelector('{held}').value") == "shown"
    drv.execute_script(f"""const s = document.querySelector('{held}'); s.value = 'none';
      s.dispatchEvent(new Event('change', {{bubbles: true}}));""")
    f = H.wait(drv, "const f = window.__mace.builderFrame(); return f && !f.busy && !f.goDisabled && !f.error && f", 60)
    assert "push apart" in drv.execute_script("return document.querySelector('dialog.builder-dialog [data-k=\"bonds-note\"]').textContent")
    drv.execute_script("document.querySelector('dialog.builder-dialog [data-act=\"cancel\"]').click()")
    H.wait(drv, "return !window.__mace.builderFrame()", 10)
    assert drv.execute_script("return window.__mace.cleanBondsHeld()") == "none"
    drv.execute_script("window.__mace.run('build.clean')")
    H.wait(drv, CLEAN, 60)
    assert drv.execute_script(f"return document.querySelector('{held}').value") == "none"
    drv.execute_script(f"""const s = document.querySelector('{held}'); s.value = 'shown';
      s.dispatchEvent(new Event('change', {{bubbles: true}}));""")
    drv.execute_script("document.querySelector('dialog.builder-dialog [data-act=\"cancel\"]').click()")
    assert not H.errors(drv), H.errors(drv)
