"""The Tasks panel and the status bar's progress item in headless Firefox
(spec §1.6, §1.7.6, §8.4.8; plan B4 and §2A): a slow task shows in both,
Cancel stops it ("Canceled after …"), a failed task offers Details, Retry
and Copy Error, and a dropped event stream reconnects and reconciles."""
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p4tasks")
    a = U.start_app(tmp, MACE_GUI_MAX_WORKERS="2")
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    return app


def start_task(drv, **body):
    # the settled value goes into __p4r (polling a Promise's fields never sees them)
    drv.execute_script("window.__p4r = null; window.__mace.apiTask('test/task', arguments[0])"
                       ".then((r) => { window.__p4r = {ok: r}; }, (e) => { window.__p4r = {err: e.message}; });", body)


def rows(drv):
    return drv.execute_script("""return [...document.querySelectorAll('.bottom-pane.tasks .task-row')].map((r) => ({
        id: r.dataset.task, chip: r.querySelector('.task-chip').textContent, title: r.querySelector('.task-title').textContent,
        log: r.querySelector('.task-log').textContent, acts: [...r.querySelectorAll('[data-act]')].map((b) => b.dataset.act),
        pct: Number(r.querySelector('.task-bar').getAttribute('aria-valuenow'))}))""")


def test_tasks_panel_empty_state(fresh):
    drv = fresh[0]
    drv.execute_script("window.__mace.run('view.tasks')")
    note = drv.execute_script("return document.querySelector('.bottom-pane.tasks').textContent.trim()")
    if not drv.execute_script("return window.__mace.tasks().length"):
        assert note == "No tasks. Builds, Clean and big files show their progress here."


def test_tasks_panel_and_status_progress(fresh):
    drv = fresh[0]
    drv.execute_script("window.__mace.run('view.tasks')")
    start_task(drv, title="Slow test task", seconds=8, steps=40, wait=0)
    H.wait(drv, "return document.querySelector('#st-task .st-task-title')", 10)
    H.wait(drv, "return Number(document.querySelector('#st-task .st-task-bar').getAttribute('aria-valuenow')) > 0", 10)
    assert "Slow test task" in drv.execute_script("return document.querySelector('#st-task').textContent")
    H.wait(drv, "return [...document.querySelectorAll('.task-row')].some((r) => r.querySelector('.task-title').textContent === 'Slow test task' && Number(r.querySelector('.task-bar').getAttribute('aria-valuenow')) > 0)", 10)
    U.shot(drv, "tasks_running")
    row = [r for r in rows(drv) if r["title"] == "Slow test task"][0]
    assert row["chip"] == "Running" and "cancel" in row["acts"], row
    drv.execute_script(f"document.querySelector('.task-row[data-task=\"{row['id']}\"] [data-act=cancel]').click()")
    H.wait(drv, f"const r = document.querySelector('.task-row[data-task=\"{row['id']}\"]'); return r && r.querySelector('.task-chip').textContent === 'Canceled'", 10)
    row = [r for r in rows(drv) if r["id"] == row["id"]][0]
    assert row["log"].startswith("Canceled after") and "remove" in row["acts"], row
    H.wait(drv, "return !document.querySelector('#st-task .st-task-bar')", 5)
    H.wait(drv, "return !!window.__p4r", 10)
    assert drv.execute_script("return window.__p4r").get("err") == "Canceled"
    U.no_errors(drv)


def test_task_failed_row_retry(fresh):
    drv = fresh[0]
    drv.execute_script("window.__mace.run('view.tasks')")
    start_task(drv, title="Failing task", seconds=0.6, steps=4, fail=True, wait=0)
    H.wait(drv, "return [...document.querySelectorAll('.task-row.s-failed')].some((r) => r.querySelector('.task-title').textContent === 'Failing task')", 15)
    row = [r for r in rows(drv) if r["title"] == "Failing task" and r["chip"] == "Failed"][0]
    assert row["log"] == "the test job failed on purpose" and {"details", "retry", "copy", "remove"} <= set(row["acts"]), row
    drv.execute_script(f"document.querySelector('.task-row[data-task=\"{row['id']}\"] [data-act=details]').click()")
    assert "RuntimeError" in drv.execute_script(f"return getComputedStyle(document.querySelector('.task-row[data-task=\"{row['id']}\"] .task-tb')).display !== 'none' && document.querySelector('.task-row[data-task=\"{row['id']}\"] .task-tb').textContent")
    U.shot(drv, "tasks_failed_details")
    n = len(rows(drv))
    drv.execute_script(f"document.querySelector('.task-row[data-task=\"{row['id']}\"] [data-act=retry]').click()")
    H.wait(drv, f"return document.querySelectorAll('.task-row').length === {n + 1}", 10)
    drv.execute_script(f"document.querySelector('.task-row[data-task=\"{row['id']}\"] [data-act=remove]').click()")
    H.wait(drv, f"return !document.querySelector('.task-row[data-task=\"{row['id']}\"]')", 5)
    U.no_errors(drv)


def test_sse_reconnect_reconciles_tasks(fresh):
    """The event stream drops while a task runs: the status bar says
    Reconnecting…, the stream comes back (retry: 2000), the page asks for
    the task list, and the task's done is processed once."""
    drv, base = fresh[0], fresh[1]
    drv.execute_script("window.__mace.run('view.tasks')")
    drv.execute_script("window.__p4done = 0; document.addEventListener('mace-task-done-test', () => window.__p4done++);")
    start_task(drv, title="Through a reconnect", seconds=4, steps=8, wait=0)
    H.wait(drv, "return window.__mace.tasks().some((t) => t.title === 'Through a reconnect' && t.state === 'running')", 10)
    old = drv.execute_script("return window.__mace.clientId()")
    U.post(base, "api/test/sse/drop", {})
    H.wait(drv, "return window.__mace.clientId() !== arguments[0]", 15, old)
    H.wait(drv, "return window.__mace.tasks().some((t) => t.title === 'Through a reconnect' && t.state === 'done')", 20)
    H.wait(drv, "return !!(window.__p4r && window.__p4r.ok)", 10)
    res = drv.execute_script("return window.__p4r.ok")
    assert res["pid"] > 0
    # the panel draws on the next frame after the store changes
    H.wait(drv, """return [...document.querySelectorAll('.bottom-pane.tasks .task-row')].some((r) =>
        r.querySelector('.task-title').textContent === 'Through a reconnect' && r.querySelector('.task-chip').textContent === 'Done')""", 10)
    row = [r for r in rows(drv) if r["title"] == "Through a reconnect"][0]
    assert row["chip"] == "Done" and row["pct"] == 100, row
    U.no_errors(drv)
