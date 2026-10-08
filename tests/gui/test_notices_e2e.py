"""Notices and their history in headless Firefox (plan E13): real clicks on a
fresh profile. Info notices close after Settings › notifications.infoSeconds,
errors stay until closed; the same notice counts up; more than three fold
into "+N more"; every notice lands in the history (Problems › All
Notifications, the status-bar bell, window.__mace.notifications)."""
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.by import By  # noqa: E402


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("notices_e2e")
    root = H.examples_copy(tmp)
    (root / "mixed").mkdir()
    (root / "mixed" / "broken.cif").write_text("data_x\n_cell_length_a oops\n")
    a = U.start_app(tmp, root=root, MACE_GUI_MAX_WORKERS="2")
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    app[0].execute_script("window.__mace.notifications.clear()")
    return app


def click(drv, el):
    ActionChains(drv).move_to_element(el).click().perform()


def click_file(drv, path):
    """A real click on the explorer row (found again when the list re-renders under the pointer)."""
    from selenium.common.exceptions import StaleElementReferenceException
    drv.execute_script("window.MACE.state.filter = 'all'; window.MACE.state.text = ''; window.MACE.renderFiles()")
    for attempt in range(6):
        try:
            el = drv.find_element(By.CSS_SELECTOR, f'#file-list .file[data-path="{path}"]')
            drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
            click(drv, el)
            return
        except StaleElementReferenceException:
            time.sleep(0.2)
    raise AssertionError(f"could not click {path}")


def error_toasts(drv):
    return drv.execute_script("return [...document.querySelectorAll('#toasts .toast2.error .msg')].map((m) => m.textContent)")


def test_notices_close_stack_and_are_kept(fresh):
    drv = fresh[0]
    drv.execute_script("window.__mace.settings.set('notifications.infoSeconds', 6)")
    # an error from a real open (a broken CIF) and an info from a real command
    click_file(drv, "mixed/broken.cif")
    H.wait(drv, "return document.querySelectorAll('#toasts .toast2.error').length === 1", 20)
    t0 = time.time()
    drv.execute_script("window.__mace.run('analysis.exportCsv')")
    H.wait(drv, "return document.querySelectorAll('#toasts .toast2.info').length === 1", 5)
    drv.execute_script("document.body.focus()")
    ActionChains(drv).move_by_offset(1, 1).perform()       # the pointer away from the notices
    H.wait(drv, "return document.querySelectorAll('#toasts .toast2.info').length === 0", 9)
    gone = time.time() - t0
    assert 5.0 <= gone <= 8.0, gone
    time.sleep(max(0, 10 - (time.time() - t0)))
    assert len(error_toasts(drv)) == 1                     # an error stays
    click(drv, drv.find_element(By.CSS_SELECTOR, "#toasts .toast2.error .toast-close"))
    H.wait(drv, "return document.querySelectorAll('#toasts .toast2').length === 0", 3)
    # three identical warnings: one notice, x3
    drv.execute_script("for (let i = 0; i < 3; i++) window.__maceToast('Same warning', 'warning')")
    assert drv.execute_script("return document.querySelectorAll('#toasts .toast2').length") == 1
    assert drv.execute_script("return document.querySelector('#toasts .toast2 .count').textContent") == "×3"
    drv.execute_script("document.querySelectorAll('#toasts > *').forEach((t) => t.remove())")
    # five errors: three shown, "+2 more", Dismiss All; the history keeps all five
    drv.execute_script("for (let i = 1; i <= 5; i++) window.__maceToast('Error number ' + i, 'error')")
    vis = drv.execute_script("return [...document.querySelectorAll('#toasts .toast2')].filter((t) => !t.hidden).length")
    assert vis == 3
    assert drv.execute_script("return document.querySelector('#toasts .toast-more').textContent") == "+2 more"
    U.shot(drv, "parity_toasts_stack")
    click(drv, drv.find_element(By.CSS_SELECTOR, "#toasts .toast-dismiss-all"))
    H.wait(drv, "return document.querySelectorAll('#toasts .toast2').length === 0", 3)
    rec = drv.execute_script("return window.__mace.notifications.recent(10).map((n) => n.text)")
    assert [f"Error number {i}" for i in range(5, 0, -1)] == rec[:5]
    # Problems lists the errors; All Notifications lists everything
    click(drv, drv.find_element(By.ID, "st-bell"))
    H.wait(drv, "return document.querySelector('.problem-filter [data-pview=all]').getAttribute('aria-pressed') === 'true'", 5)
    allrows = drv.execute_script("return [...document.querySelectorAll('.bottom-pane.problems .p-msg')].map((m) => m.textContent)")
    assert "Same warning" in allrows and "Error number 3" in allrows
    click(drv, drv.find_element(By.CSS_SELECTOR, ".problem-filter [data-pview=problems]"))
    probs = drv.execute_script("return [...document.querySelectorAll('.bottom-pane.problems .p-msg')].map((m) => m.textContent)")
    assert "Error number 1" in probs


def test_history_api_counts_and_subscribers(fresh):
    drv = fresh[0]
    got = drv.execute_script("""
        const seen = []; const off = window.__mace.notifications.subscribe((e) => e && seen.push(e.text));
        for (let i = 0; i < 60; i++) window.__maceToast('n' + i, i % 2 ? 'info' : 'warning');
        off();
        document.querySelectorAll('#toasts > *').forEach((t) => t.remove());
        return {seen: seen.length, recent: window.__mace.notifications.recent(50)};""")
    assert got["seen"] == 60
    assert [r["text"] for r in got["recent"]] == [f"n{i}" for i in range(59, 9, -1)]
    assert all(r["count"] == 1 and "t" in r and r["kind"] in ("info", "warning") for r in got["recent"])
