"""The inspector's run card shows the time left of an output still being
written (mace/cluster/progress.py's levels, as the Job document and `mace
cluster job` word them), read again while the card is shown (1.1.6)."""
import os

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")

OUT = "outputs/Ag2Br3_opt_unfinished.out"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("runeta")
    root = H.examples_copy(tmp)
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1")
    drv = None
    try:
        drv = H.browser(os.environ.get("MACE_GUI_BROWSER", "firefox"), width=1400, height=1000)
        drv.set_script_timeout(120)
        H.load(drv, base)
        yield drv, root
    finally:
        if drv is not None:
            drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def test_run_card_time_left(app):
    drv, root = app
    drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then(() => d())", OUT)
    H.wait(drv, "return document.querySelector('#inspector .rc-eta-levels li')", 60)
    items = drv.execute_script("return [...document.querySelectorAll('#inspector .rc-eta-levels li')]"
                               ".map((li) => [li.className, li.innerText, li.title])")
    from mace.cluster import progress as P
    want = [lv for lv in P.progress(root / OUT)["levels"] if lv["depth"] == 0 or lv["status"] != "done"]
    assert [t for _, t, _ in items][0].startswith(want[0]["text"])
    assert [t for _, t, _ in items][1:] == [lv["text"] for lv in want[1:]]
    assert items[0][0].startswith("d0") and items[1][0].startswith("d1")
    assert not H.errors(drv)


def test_finished_output_shows_no_time_left(app):
    drv, _ = app
    drv.execute_script("window.__mace.closeAll()")
    drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then(() => d())", "outputs/graphene_slab_sp.out")
    H.wait(drv, "return document.querySelector('#inspector .rc-stop:not(.skel)')", 60)
    H.wait(drv, "const e = document.querySelector('#inspector .rc-eta'); return e && e.dataset.done", 30)
    assert drv.execute_script("return document.querySelector('#inspector .rc-eta').innerHTML") == ""
    assert not H.errors(drv)
