"""The explorer's sections share its height (1.1.5 fix): with thirty open
editors the Open Editors list scrolls inside its own share and the file tree
keeps its minimum height (and scrolls too), at window heights 600, 768 and
1080 and UI scales 100, 110 and 125 %. Every section header folds its
section (aria-expanded), and the folds come back after a reload.

Chromium and Firefox; the folder is a copy of examples/gui plus thirty small
scripts to open as editors."""
import time

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")

N_EDITORS = 30
HEIGHTS = (600, 768, 1080)
SCALES = (100, 110, 125)


@pytest.fixture(scope="module", params=["chromium", "firefox"])
def app(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sidebar")
    root = H.examples_copy(tmp)
    for i in range(N_EDITORS):
        (root / f"note_{i:02d}.sh").write_text(f"#!/bin/sh\necho {i}\n")
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1")
    try:
        drv = H.browser(request.param, width=1280, height=800)
    except BaseException:
        proc.terminate()
        proc.wait(timeout=15)
        raise
    try:
        H.load(drv, base)
        yield drv
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


@pytest.fixture
def drv(app, request):
    failed = request.session.testsfailed
    yield app
    if request.session.testsfailed > failed:
        H.save_failure(app, request.node.name)
    app.execute_script("window.__mace.closeAll(); window.__mace.settings.set('uiScale', 100);"
                       "window.__mace.profile.setUi('explorer', {})")
    H.set_viewport(app, 1280, 800)
    time.sleep(0.3)


def open_editors(drv, n=N_EDITORS):
    drv.execute_async_script("""const done = arguments[arguments.length - 1];
      (async () => { for (let i = 0; i < arguments[0]; i++) await window.__mace.open('note_' + String(i).padStart(2, '0') + '.sh'); })()
        .then(() => done(true), (e) => done(String(e)));""", n)
    H.wait(drv, f"return document.querySelectorAll('#editors-rows .file').length === {n}", 30)


def scale(drv, s):
    drv.execute_script("window.__mace.settings.set('uiScale', arguments[0])", s)
    time.sleep(0.4)
    z = drv.execute_script("const sh = document.getElementById('shell'); return sh.getBoundingClientRect().width / sh.offsetWidth")
    assert abs(z - s / 100) < 0.01, (s, z)


MEASURE = """const px = (el) => el.getBoundingClientRect(), ex = document.getElementById('explorer');
  const files = document.getElementById('files-section'), list = document.getElementById('file-list');
  const rows = document.getElementById('editors-rows'), z = px(ex).height / ex.offsetHeight;
  const heads = [...ex.querySelectorAll(':scope > .side-section:not(:empty) > .section-label')]
    .map((h) => ({id: h.id, top: px(h).top, bottom: px(h).bottom}));
  return {z, exBottom: px(ex).bottom, filesH: files.offsetHeight, treeMin: Math.min(5 * list.querySelector('.file').offsetHeight + files.querySelector('.section-label').offsetHeight, 0.33 * ex.offsetHeight),
    listH: list.clientHeight, listScroll: list.scrollHeight > list.clientHeight + 1, row: list.querySelector('.file').offsetHeight,
    edH: rows.clientHeight, edScroll: rows.scrollHeight > rows.clientHeight + 1, exH: ex.offsetHeight, heads,
    activeIn: (() => { const a = rows.querySelector('.file.active'); if (!a) return null;
      const r = px(a), b = px(rows); return r.top >= b.top - 1 && r.bottom <= b.bottom + 1; })()};"""


def test_open_editors_never_squeeze_the_file_tree(drv):
    open_editors(drv)
    seen = []
    for h in HEIGHTS:
        H.set_viewport(drv, 1280, h)
        for s in SCALES:
            scale(drv, s)
            m = drv.execute_script(MEASURE)
            seen.append((h, s, m["filesH"], m["treeMin"], m["edH"], m["listH"]))
            where = (h, s, m)
            # the tree keeps at least its minimum and shows rows; both lists scroll
            assert m["filesH"] >= m["treeMin"] - 1, where
            assert m["treeMin"] >= 4 * m["row"], where
            assert m["listH"] >= 3 * m["row"], where
            assert m["listScroll"] and m["edScroll"], where
            # Open Editors shows rows too, wherever the explorer has the room (all but a
            # 600 px window at 125 %, where its 384 px go to the tree, the headers and the filter)
            if m["exH"] >= 440:
                assert m["edH"] >= m["row"], where
            # the Open Editors section stays inside its share of the explorer
            assert m["edH"] <= 0.31 * m["exH"], where
            # every header shows, inside the explorer
            assert all(x["bottom"] <= m["exBottom"] + 1 for x in m["heads"]), where
            # the active editor (the last opened) is scrolled into its section's view
            if m["edH"] >= m["row"]:
                assert m["activeIn"], where
    print("sizes", seen)


def test_every_header_folds_and_the_fold_survives_a_reload(drv):
    open_editors(drv, 3)
    heads = drv.execute_script("""return [...document.querySelectorAll('#explorer > .side-section:not(:empty) > .section-label')]
      .map((h) => [h.id, h.tagName, h.getAttribute('aria-expanded'), h.getAttribute('aria-controls')])""")
    ids = [x[0] for x in heads]
    assert "editors-toggle" in ids and "files-toggle" in ids, heads
    assert all(t == "BUTTON" and e in ("true", "false") and c for _, t, e, c in heads), heads
    for tid in ("editors-toggle", "files-toggle"):
        drv.find_element("id", tid).click()
        H.wait(drv, f"return document.getElementById('{tid}').getAttribute('aria-expanded') === 'false'", 5)
    assert drv.execute_script("return document.getElementById('editors-rows').hidden && document.getElementById('file-list').hidden")
    # folded Files: the other sections keep their place at the top, the footer at the bottom
    drv.execute_script("return window.__mace.profile.flush()")
    drv.refresh()
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'")
    H.wait(drv, "return document.getElementById('files-toggle').getAttribute('aria-expanded') === 'false'", 15)
    assert drv.execute_script("return document.getElementById('file-list').hidden")
    drv.execute_async_script(
        "const d = arguments[arguments.length - 1]; window.__mace.open('note_00.sh').then(() => d(1), () => d(0))")
    H.wait(drv, "return !!document.getElementById('editors-toggle')", 15)
    assert drv.execute_script("return document.getElementById('editors-toggle').getAttribute('aria-expanded') === 'false' && document.getElementById('editors-rows').hidden")
    # unfold both again: by keyboard (Enter on the focused header)
    from selenium.webdriver.common.keys import Keys
    for tid in ("editors-toggle", "files-toggle"):
        drv.find_element("id", tid).send_keys(Keys.ENTER)
        H.wait(drv, f"return document.getElementById('{tid}').getAttribute('aria-expanded') === 'true'", 5)
        # the header keeps the focus through the section's re-render
        assert drv.execute_script("return document.activeElement.id") == tid
    assert drv.execute_script("return !document.getElementById('editors-rows').hidden && !document.getElementById('file-list').hidden")
    assert not H.errors(drv)
