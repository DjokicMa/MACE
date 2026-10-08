"""Palette ranking by match quality, then availability (plan T9b, carry-over
c), and the undoable Clear Recently Opened, in headless Firefox. A weak
enabled match never jumps ahead of a strong disabled one, and Enter on a
disabled row explains it instead of running something else."""
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")                 # the server reads the demo structures with it

QUERIES = ["sup", "exp", "sym", "def", "clean"]


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    a = U.start_app(tmp_path_factory.mktemp("p4pal"))
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    return app


def rows(drv, q):
    return drv.execute_script("""return window.__mace.palette.results(arguments[0]).map((r) =>
        ({id: r.id, label: r.label, why: r.why || '', tier: r.tier, cls: r.cls}))""", q)


def test_rank_quality_tiers(fresh):
    drv = fresh[0]
    got = drv.execute_async_script("""const done = arguments[0];
        import('./static/js/rank.js').then((R) => done([
          ['sup', 'build: supercell…'], ['sup', 'set up calculation'], ['clean', 'clear recently opened'],
          ['clean', 'clean geometry…'], ['sc', 'show cell'], ['cell', 'edit cell…'], ['setup', 'set up calculation'],
          ['xyz', 'build: supercell…'], ['ell', 'show cell'], ['', 'anything']].map(([q, t]) => R.quality(q, t))),
          (e) => done('ERR ' + e.message))""")
    assert got == [0, 1, 2, 0, 0, 0, 1, -1, 2, 0], got


@pytest.mark.parametrize("where", ["welcome", "MgO"])
def test_palette_enabled_first_within_tier(fresh, where):
    drv = fresh[0]
    if where == "MgO":
        U.open_doc(drv, "structures/MgO.cif")
    for q in QUERIES:
        r = rows(drv, q)
        assert r, q
        keys = [(x["tier"], x["cls"]) for x in r]
        assert keys == sorted(keys), (q, where, [(x["label"], x["tier"], x["cls"]) for x in r])
        assert all(x["tier"] in (0, 1, 2) for x in r)
        # the class matches what the row shows: enabled rows have no reason
        assert all((x["cls"] == 0) == (x["why"] == "") for x in r), (q, r)
    if where == "MgO":
        first = rows(drv, "sup")[0]
        assert first["id"] == "build.supercell" and first["cls"] == 0, first
        assert any(x["label"] == "Calculation: Set Up Calculation…" or "Set Up" in x["label"] for x in rows(drv, "sup")[1:])


def test_palette_clean_on_welcome_not_clear_recent(fresh):
    drv = fresh[0]
    # the Recent list lives in the profile (polish P2)
    drv.execute_script("window.__mace.profile.setRecentFiles([])")
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.closeAll()")
    recent0 = drv.execute_script("return JSON.stringify(window.__mace.profile.recentFiles())")
    assert "MgO.cif" in recent0, recent0
    r = rows(drv, "clean")
    assert r[0]["id"] != "file.clearRecent" and r[0]["tier"] == 0, r[:3]
    clear = [x for x in r if x["id"] == "file.clearRecent"]
    assert not clear or clear[0]["tier"] == 2
    drv.execute_script("window.__mace.palette.open('')")
    box = drv.find_element("css selector", "#palette input")
    box.send_keys("clean")
    time.sleep(0.2)
    from selenium.webdriver.common.keys import Keys
    box.send_keys(Keys.ENTER)
    time.sleep(0.3)
    recent1 = drv.execute_script("return JSON.stringify(window.__mace.profile.recentFiles())")
    import json as _json
    assert _json.loads(recent1) == _json.loads(recent0)          # Enter didn't clear the recent list
    U.keys(drv, Keys.ESCAPE)


def test_palette_enter_on_disabled_shows_reason(fresh):
    """Welcome screen, "clean": the first row is Clean Geometry (needs a
    structure); Enter shows that in the hint line and a toast, runs nothing,
    and the palette stays open."""
    drv = fresh[0]
    drv.execute_script("window.__mace.closeAll()")
    drv.execute_script("window.__mace.palette.open('')")
    box = drv.find_element("css selector", "#palette input")
    box.send_keys("clean")
    time.sleep(0.2)
    first = rows(drv, "clean")[0]
    assert first["why"], first
    from selenium.webdriver.common.keys import Keys
    box.send_keys(Keys.ENTER)
    time.sleep(0.3)
    assert drv.execute_script("return document.querySelector('#palette').open")
    hint = drv.execute_script("return document.querySelector('#palette .pal-hint').textContent")
    assert first["why"] in hint, (hint, first)
    assert any(first["why"] in t for t in U.toasts(drv))
    assert not drv.execute_script("const d = document.querySelector('.builder-dialog, #clean-panel'); return !!d && d.open")
    U.keys(drv, Keys.ESCAPE)


def test_clear_recent_undo_toast(fresh):
    drv = fresh[0]
    for p in ("structures/MgO.cif", "structures/NaCl.cif"):
        U.open_doc(drv, p)
    drv.execute_script("window.__mace.closeAll()")
    get = "return JSON.stringify(window.__mace.profile.recentFiles())"
    before = drv.execute_script(get)
    n = len(__import__("json").loads(before))
    drv.execute_script("window.__mace.run('file.clearRecent')")
    assert drv.execute_script(get) == "[]"
    t = [x for x in U.toasts(drv) if x.startswith("Cleared ")]
    assert t and t[0].startswith(f"Cleared {n} recent file") and "Undo" in t[0], U.toasts(drv)
    drv.execute_script("[...document.querySelectorAll('#toasts > *')].find((t) => t.textContent.startsWith('Cleared ')).querySelector('.acts button').click()")
    assert __import__("json").loads(drv.execute_script(get)) == __import__("json").loads(before)
    U.no_errors(drv)
