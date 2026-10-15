"""Create or Update Database from a Folder in headless Firefox (PLAN_analysis
C10-a, C10-b, C10-d): a real right-click on an Explorer folder, the preview
counts against the CLI's own --dry-run --json, Create, the Database view
with real formulas, then a content change, Update from Folder… on the
Database view's toolbar with a backup, and "Updating…" while it runs. Real
outputs copied from the corpus into a fresh folder, a fresh profile.

The server and the browser are shared by the module; every test starts from
the corpus copies again (no database, nothing open, the page reloaded) and
builds what it needs itself, so the tests pass alone and in any order."""
import json
import os
import re
import shutil
import subprocess
import sys
import time

import pytest

import guiharness as H
import p4util as U
import runcorpus as RC

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402

MIXED = ["OPT/1_dia_opt_rev1.out", "OPT/1_dia_opt_rev1.d12",
         "SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out", "SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.d12",
         "BAND/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band.out", "BAND/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band.d3",
         "DOSS/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss.out", "DOSS/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss.d3",
         "FAILED_QA/3_dia3_opt_killed_signal9_oom.out"]


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    RC.need(RC.DIA_OPT)
    tmp = tmp_path_factory.mktemp("dbbuilde2e")
    root = tmp / "root"
    _copy_runs(root)
    xdg = tmp / "xdg"
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_TEST_HOOKS="1")
    downloads = tmp / "downloads"
    downloads.mkdir()
    try:
        # a fresh profile with the clipboard granted (Copy JSON is checked by a real paste)
        drv = H.firefox(1280, 800, download_dir=downloads, prefs={
            "dom.events.testing.asyncClipboard": True, "dom.events.asyncClipboard.clipboardItem": True,
            "dom.events.asyncClipboard.readText": True})
        H.load(drv, base)
    except BaseException:
        proc.terminate()
        raise
    try:
        yield drv, base, root, xdg, proc, downloads
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def _copy_runs(root):
    """root/runs: the corpus copies and nothing else (no database, no backups)."""
    shutil.rmtree(root / "runs", ignore_errors=True)
    for r in MIXED:
        RC.copy_into(root / "runs", RC.CORPUS / r)


@pytest.fixture
def fresh(app):
    """The shared app reset to its starting state: nothing open, the runs
    copied again, the page reloaded."""
    drv, base, root, xdg, proc, downloads = app
    drv.execute_script("window.__mace && window.__mace.closeAll && window.__mace.closeAll()")
    _copy_runs(root)
    H.load(drv, base)
    return app


def _build_db(drv, base):
    """runs/materials.db built through the GUI's own build route, waited for;
    the page is loaded again so its file list has the new database."""
    st, _, r = U.req(base, "api/db/build", {"folder": "runs", "confirm": True, "skip_unchanged": False,
                                            "wait": 600}, timeout=660)
    assert st == 200, r
    H.load(drv, base)


def _cli_dry_run(root):
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    p = subprocess.run([sys.executable, str(H.REPO_ROOT / "mace_cli"), "--no-banner", "analyze", "--extract-properties", "runs",
                        "--db-path", "runs/materials.db", "--register", "--dry-run", "--json"],
                       capture_output=True, text=True, env=env, cwd=str(root), timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def chips(drv):
    return drv.execute_script(
        "return Object.fromEntries([...document.querySelectorAll('#db-build .dbb-chip')].map((b) => [b.dataset.class, Number(b.querySelector('b').textContent) || 0]))")


def wait_preview(drv):
    H.wait(drv, "const n = document.querySelector('#db-build .dbb-pnote'); return n && /outputs? in the folder/.test(n.textContent)", 120)


def test_create_from_explorer_then_update_from_the_database_view(fresh):
    drv, base, root, xdg, proc, downloads = fresh
    # a real right-click on the folder's header in the Explorer (the list may still be redrawn
    # once after the page loads its file details: the header is looked up again then)
    from selenium.common.exceptions import StaleElementReferenceException
    for attempt in range(5):
        try:
            folder = drv.find_element("css selector", '#file-list .folder[data-dir="runs"]')
            drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", folder)
            ActionChains(drv).context_click(folder).perform()
            break
        except StaleElementReferenceException:
            time.sleep(0.5)
    H.wait(drv, "return !!document.querySelector(`.menu .menu-item[data-cmd-id='explorer.dbBuild']`)", 5)
    drv.find_element("css selector", ".menu .menu-item[data-cmd-id='explorer.dbBuild']").click()
    H.wait(drv, "const d = document.querySelector('#db-build'); return d && d.open", 15)
    wait_preview(drv)
    assert drv.find_element("css selector", "#db-build .dbb-state").text == "Will create runs/materials.db"
    cli = _cli_dry_run(root)
    got = chips(drv)
    assert {k: got.get(k, 0) for k in cli["counts"]} == cli["counts"]
    cmd = drv.find_element("css selector", "#db-build .dbb-cmd").text
    # Back up the database first is ticked (the default): the build runs --backup, so the command shows it
    assert cmd == ("mace analyze --extract-properties runs --db-path runs/materials.db --register --skip-unchanged "
                   "--backup --keep-backups 3")
    guarantee = drv.find_element("css selector", "#db-build .dbb-guarantee").text
    docs = (H.REPO_ROOT / "docs" / "gui.md").read_text()
    assert guarantee and guarantee in " ".join(docs.split())
    # the chip filter: Skipped lists the killed run only (no calculation record: not extracted)
    drv.find_element("css selector", "#db-build .dbb-chip[data-class=skipped]").click()
    H.wait(drv, "return document.querySelectorAll('#db-build .dbb-row').length === 1", 5)
    assert "killed" in drv.find_element("css selector", "#db-build .dbb-row").text
    go = drv.find_element("css selector", "#db-build .dbb-go")
    assert go.text == "Create Database" and go.is_enabled()
    go.click()
    H.wait(drv, "const d = document.querySelector('#db-build'); return !d.open", 600)
    H.wait(drv, "return [...document.querySelectorAll('#toasts .msg')].some((m) => /^Database updated: \\d+ new/.test(m.textContent))", 10)
    # the toast's button may be drawn a moment after its message
    H.wait(drv, "return [...document.querySelectorAll('#toasts button')].some((b) => b.textContent.trim() === 'Open Database')", 10)
    drv.execute_script("[...document.querySelectorAll('#toasts button')].filter((b) => b.textContent.trim() === 'Open Database').pop().click()")
    H.wait(drv, "return document.querySelectorAll('#tab-db .dbv-tr[data-id]').length >= 1", 60)
    formulas = drv.execute_script("return [...document.querySelectorAll('#tab-db .dbv-tr[data-id]')].map((r) => r.textContent)")
    assert formulas and not any("Unknown" in f for f in formulas), formulas
    # C10-b: one content change, Update from Folder… on the Database view: 1 changed, a backup
    with open(root / "runs" / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out", "a") as fh:
        fh.write("\n")
    drv.find_element("css selector", "#tab-db [data-act=build]").click()
    H.wait(drv, "const d = document.querySelector('#db-build'); return d && d.open", 15)
    wait_preview(drv)
    assert chips(drv).get("changed") == 1
    assert drv.find_element("css selector", "#db-build .dbb-state").text.startswith("Will update runs/materials.db (")
    bk = drv.find_element("css selector", "#db-build .dbb-backup")
    assert bk.is_selected() and not bk.is_enabled()                 # locked on for an existing database
    assert "--backup --keep-backups 3" in drv.find_element("css selector", "#db-build .dbb-cmd").text
    go = drv.find_element("css selector", "#db-build .dbb-go")
    assert go.text.startswith("Update Database (")
    go.click()
    H.wait(drv, "const d = document.querySelector('#db-build'); return !d.open", 600)
    assert len(list((root / "runs" / "backups").glob("materials_backup_*.db"))) == 1
    U.no_errors(drv)


def test_database_view_shows_updating_during_a_build(fresh):
    """C10-d: a build that holds the database for a while (the CLI starts 6 s late, test hook)."""
    drv, base, root, xdg, proc, downloads = fresh
    _build_db(drv, base)
    U.open_doc(drv, "runs/materials.db", wait_atoms=False)
    H.wait(drv, "return document.querySelectorAll('#tab-db .dbv-tr[data-id]').length > 0", 60)
    st, _, r = U.req(base, "api/db/build", {"folder": "runs", "confirm": True, "skip_unchanged": False, "wait": 0,
                                            "test_delay": 6})
    assert st == 202, r
    drv.execute_script("window.__mace.dbView && window.__mace.dbView.poll && window.__mace.dbView.poll()")
    H.wait(drv, "const l = document.querySelector('#tab-db .dbv-live'); return l && /Updating/.test(l.textContent)", 20)
    H.wait(drv, "const l = document.querySelector('#tab-db .dbv-live'); return l && !/Updating/.test(l.textContent)", 600)
    st, _, log = U.req(base, "api/test/dblog")
    assert log["immutable_opens"] == []
    U.no_errors(drv)


def test_long_json_values_are_cut_with_copy_json(fresh):
    """C11-b: the material with BAND and DOSS runs: long values cut, Copy JSON gives the whole valid JSON."""
    drv, base, root, xdg, proc, downloads = fresh
    _build_db(drv, base)
    U.open_doc(drv, "runs/materials.db", wait_atoms=False)
    H.wait(drv, "return document.querySelectorAll('#tab-db .dbv-tr[data-id]').length > 0", 60)
    from selenium.common.exceptions import StaleElementReferenceException
    for attempt in range(5):         # the table may be drawn again once its rows finish loading
        try:
            row = drv.find_element("css selector", "#tab-db .dbv-tr[data-id='1_dia_opt_rev1']") if drv.find_elements(
                "css selector", "#tab-db .dbv-tr[data-id='1_dia_opt_rev1']") else drv.find_elements("css selector", "#tab-db .dbv-tr[data-id]")[0]
            row.click()
            break
        except StaleElementReferenceException:
            time.sleep(0.5)
    H.wait(drv, "return document.querySelectorAll('#inspector [data-copy-json]').length > 0", 30)
    # the calculations the GUI build registered are found in this folder (absolute paths)
    insp = drv.find_element("css selector", "#inspector").text
    assert "Files not in this folder" not in insp and "Open output" in insp
    # a property the OPT and the SP both have names its calculation
    rows = drv.execute_script("""return [...document.querySelectorAll('#inspector .dbd-cat tr td:first-child')]
        .map((t) => t.textContent)""")
    dup = [t for t in rows if t.startswith("total_cpu_time")]
    assert len(dup) >= 2 and all(re.search(r"\((OPT|SP|BAND|DOSS)\)$", t) for t in dup), dup
    longest = drv.execute_script("""const w = document.createTreeWalker(document.querySelector('#inspector'), NodeFilter.SHOW_TEXT);
        let m = 0; while (w.nextNode()) m = Math.max(m, w.currentNode.nodeValue.length); return m""")
    assert longest <= 2000
    btn = drv.find_elements("css selector", "#inspector [data-copy-json]")[0]
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", btn)
    btn.click()
    H.wait(drv, "return [...document.querySelectorAll('#toasts .msg')].some((m) => m.textContent === 'Copied the JSON')", 5)
    drv.execute_script("const t = document.createElement('textarea'); t.id = 'paste-here'; t.style.cssText = 'position:fixed;top:0;left:0;width:50px;height:20px;z-index:99999'; document.body.appendChild(t)")
    t = drv.find_element("css selector", "#paste-here")
    t.click()
    U.keys(drv, Keys.CONTROL, "v")
    H.wait(drv, "return document.querySelector('#paste-here').value.length > 2000", 10)
    text = drv.execute_script("const t = document.querySelector('#paste-here'); const v = t.value; t.remove(); return v")
    assert isinstance(json.loads(text), (list, dict))
    U.no_errors(drv)
