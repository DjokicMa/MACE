"""The Database document in headless Firefox (plan T7): sort, filter, the
material's detail, Open output, Export Table, and the grid's keyboard model on
a 10k-material database (one Tab stop, arrows, PageDown, End reaches the last
row of the whole result, Enter opens the drawer, aria-rowcount/rowindex)."""
import csv
import io
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U
import dbgen

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
from selenium.webdriver.common.keys import Keys  # noqa: E402


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p5db")
    root = H.examples_copy(tmp)
    dbgen.make(root / "big10k.db", 10000, calcs_per=2, props_per=3)
    a = U.start_app(tmp, root=root)
    try:
        yield a
    finally:
        U.stop_app(a)
        (root / "big10k.db").unlink(missing_ok=True)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    return app


def rows(drv):
    return drv.execute_script("return [...document.querySelectorAll('#tab-db .dbv-tr[data-id]')].map((r) => r.dataset.id)")


def open_db(drv, path="materials.db"):
    U.open_doc(drv, path, wait_atoms=False)
    H.wait(drv, "return document.querySelectorAll('#tab-db .dbv-tr[data-id]').length > 0", 30)


def test_sort_filter_detail_and_open_output(fresh):
    pytest.importorskip("ase", reason="opening the output reads its geometry with ase")
    drv = fresh[0]
    open_db(drv)
    assert rows(drv)[:2] == ["Ag2Br3", "LiF"]
    drv.execute_script("document.querySelector('#tab-db [data-sort=material_id]').click()")
    H.wait(drv, "return document.querySelector('#tab-db .dbv-th[aria-sort=descending]')", 10)
    H.wait(drv, "const r = document.querySelector('#tab-db .dbv-tr[data-id]'); return r && r.dataset.id === 'graphene_slab'", 10)
    q = drv.find_element("css selector", "#tab-db .dbv-q")
    q.send_keys("Br")
    H.wait(drv, "return document.querySelectorAll('#tab-db .dbv-tr[data-id]').length === 1", 10)
    assert rows(drv) == ["Ag2Br3"]
    assert "1 material" in drv.find_element("css selector", "#tab-db .dbv-count").text
    q.send_keys("xx")
    H.wait(drv, "return !document.querySelector('#tab-db .dbv-empty').hidden", 10)
    assert "No materials match" in drv.find_element("css selector", "#tab-db .dbv-empty").text
    U.shot(drv, "db_no_match")
    drv.execute_script("document.querySelector('#tab-db [data-act=clear]').click()")
    H.wait(drv, "return document.querySelectorAll('#tab-db .dbv-tr[data-id]').length === 5", 10)
    drv.execute_script("document.querySelector('#tab-db .dbv-tr[data-id=Ag2Br3]').click()")
    # its id is its formula, so the detail names it once, by the formula
    H.wait(drv, "const f = document.querySelector('#inspector .dbd .formula'); return f && f.textContent.startsWith('Ag2Br3')", 10)
    U.shot(drv, "db_detail")
    drv.execute_script("document.querySelector('#inspector [data-open=\"outputs/Ag2Br3_opt_unfinished.out\"]').click()")
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.kind === 'trajectory' && d.path === 'outputs/Ag2Br3_opt_unfinished.out'", 30)
    U.no_errors(drv)


def test_export_table_downloads_the_filtered_rows(fresh):
    drv, base, root, xdg, proc, downloads = fresh
    open_db(drv, "big10k.db")
    drv.execute_script("document.querySelector('#tab-db [data-dim=SLAB]').click()")
    H.wait(drv, "return document.querySelector('#tab-db .dbv-count').textContent.includes('material') && "
                "!document.querySelector('#tab-db .dbv-count').textContent.startsWith('10,000')", 10)
    n = drv.execute_script("return window.__mace.dbView.state().total")
    for f in downloads.glob("*.csv"):
        f.unlink()
    drv.execute_script("window.__mace.run('analysis.exportCsv')")
    end = time.time() + 30
    got = None
    while time.time() < end:
        got = next((f for f in downloads.glob("big10k_export.csv") if f.stat().st_size > 0), None)
        if got and not list(downloads.glob("*.part")):
            break
        time.sleep(0.2)
    assert got, list(downloads.iterdir())
    data = list(csv.reader(io.StringIO(got.read_text())))
    assert len(data) - 1 == n and n > 0
    dims = data[0].index("dimensionality")
    assert {r[dims] for r in data[1:]} == {"SLAB"}


def test_grid_keyboard_model_on_10k_rows(fresh):
    drv = fresh[0]
    open_db(drv, "big10k.db")
    assert drv.execute_script("return document.querySelector('#tab-db .dbv-table').getAttribute('aria-rowcount')") == "10001"
    drv.execute_script("document.querySelector('#tab-db .dbv-table').focus()")
    H.wait(drv, "return document.activeElement.classList.contains('dbv-tr')", 5)
    idx = lambda: drv.execute_script("return document.activeElement.getAttribute('aria-rowindex')")  # noqa: E731
    assert idx() == "2"
    for _ in range(3):
        U.keys(drv, Keys.ARROW_DOWN)
    assert idx() == "5"
    U.keys(drv, Keys.END)
    H.wait(drv, "return document.activeElement.getAttribute('aria-rowindex') === '10001'", 10)
    H.wait(drv, "return document.activeElement.dataset.id === 'mat009999'", 10)
    U.keys(drv, Keys.ENTER)
    H.wait(drv, "const t = document.querySelector('#inspector .dbd-title'); return t && t.textContent === 'mat009999'", 10)
    U.keys(drv, Keys.HOME)
    H.wait(drv, "return document.activeElement.getAttribute('aria-rowindex') === '2'", 10)
    U.keys(drv, Keys.PAGE_DOWN)
    assert int(idx()) > 10
    # one Tab stop: Tab leaves the grid
    tabbables = drv.execute_script("return [...document.querySelectorAll('#tab-db .dbv-tr')].filter((r) => r.tabIndex === 0).length")
    assert tabbables == 1
    st = drv.execute_script("return window.__mace.dbView.stats()")
    assert st["maxRows"] <= 80, st
    U.shot(drv, "db_10k_keyboard")
    U.no_errors(drv)


def test_table_fits_1280_without_a_sideways_scroll(fresh):
    """At 1280 px with the explorer and inspector open the columns share the width:
    no horizontal scrollbar, no clipped cell or status pill; a resized column still scrolls."""
    drv = fresh[0]
    U.reset(drv, 1280, 800)
    open_db(drv, "big10k.db")
    time.sleep(0.3)
    sw, cw = drv.execute_script("const s = document.querySelector('#tab-db .dbv-scroll'); return [s.scrollWidth, s.clientWidth]")
    assert sw <= cw, (sw, cw)
    clipped = drv.execute_script("""return [...document.querySelectorAll('#tab-db .dbv-td, #tab-db .dbv-td .status, #tab-db .dbv-th button')]
        .filter((e) => getComputedStyle(e).display !== 'inline' && e.scrollWidth > e.clientWidth + 1).map((e) => e.textContent)""")
    assert not clipped, clipped[:5]


def test_detail_names_a_material_once(fresh):
    """A material whose id is its formula (the demo's Ag2Br3, cell Ag4Br6) shows the formula alone;
    an id that differs (mat000001) is shown above it."""
    drv = fresh[0]
    open_db(drv)
    drv.execute_script("document.querySelector('#tab-db .dbv-tr[data-id=Ag2Br3]').click()")
    H.wait(drv, "return !!document.querySelector('.dbd .formula')", 10)
    assert drv.execute_script("return document.querySelectorAll('.dbd .dbd-title').length") == 0
    # its only run failed, but the output is in the folder (as Open output shows): Open Structure is offered
    assert drv.execute_script("const b = document.querySelector('.dbd [data-act=structure]'); return !b.disabled && b.title") == \
        "outputs/Ag2Br3_opt_unfinished.out"
    open_db(drv, "big10k.db")
    drv.execute_script("document.querySelector('#tab-db .dbv-tr[data-id=mat000001]').click()")
    H.wait(drv, "const t = document.querySelector('.dbd .dbd-title'); return t && t.textContent === 'mat000001'", 10)
