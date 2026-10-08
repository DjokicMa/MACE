"""Sessions in a fresh headless Firefox, real clicks (temp copies of real
test/ files): the Signed Out dialog after a (tiny) idle limit keeps the edit
as a draft and signing in with the token brings the workbench back; the
history's old ?login= URL shows the sign-in page; Settings > Server is kept
in gui.json across a restart on another port."""
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "test"
FILES = ["CIFs/1_dia_opt_rev1.cif", "OPT/1_dia_opt_rev1.out"]


def start(tmp, xdg, **env):
    root = tmp / "calc"
    if not root.exists():
        root.mkdir()
        for f in FILES:
            shutil.copy(CORPUS / f, root / Path(f).name)
    e = H.env(xdg, MACE_GUI_NO_WARMUP="1", **env)
    p = H.ServerProcess([sys.executable, str(H.MACE_CLI), "--no-banner", "gui", "--dir", str(root), "--port", "0",
                         "--token", H.TOKEN], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, env=e, start_new_session=True,
                        preexec_fn=H.die_with_pytest)
    for _ in range(40):
        line = p.stdout.readline()
        m = re.search(r"(http://127\.0\.0\.1:\d+/s/[^/]+/)\?login=(\S+)", line)
        if m:
            return p, m.group(1), m.group(2)
    raise AssertionError("no URL")


@pytest.fixture
def corpus():
    if not all((CORPUS / f).is_file() for f in FILES):
        pytest.skip("test/ corpus not present")


@H.needs_ase
def test_signed_out_dialog_keeps_the_draft_and_signs_back_in(tmp_path, corpus):
    p, base, code = start(tmp_path, tmp_path / "xdg", MACE_GUI_SESSION_IDLE_HOURS="0.002")   # 7.2 s
    drv = H.firefox()
    try:
        drv.get(base + "?login=" + code)
        H.wait(drv, "return document.documentElement.dataset.appReady === '1'", 60)
        first_url = base + "?login=" + code
        drv.execute_script("window.__mace.settings.set('autosaveDrafts', 1)")
        drv.execute_async_script("const done = arguments[1]; window.__mace.open(arguments[0]).then(() => done())",
                                 "1_dia_opt_rev1.cif")
        H.wait(drv, "return window.__mace.viewport && window.__mace.viewport.displayed() > 0", 30)
        drv.execute_script("window.__mace.run('build.symModeP1')")
        drv.execute_script("const E = window.__mace.editor.E; E.api.edit(window.__mace.editor.doc(), 'replace', "
                           "{indices: [0], element: 'Si'}, {})")
        H.wait(drv, "const e = window.__mace.editor.doc(); return e && e.dirty && e.pending === 0", 20)
        # no input from here: the session expires; the event stream says so
        H.wait(drv, "const d = document.getElementById('signed-out'); return d && d.open", 30)
        assert "You were signed out" in drv.find_element("css selector", "#signed-out .dlg-body").text
        drafts = drv.execute_script("return Object.keys(localStorage).filter(k => k.startsWith('mace.gui.draft.'))")
        assert any(k.endswith("1_dia_opt_rev1.cif") for k in drafts), drafts
        drv.find_element("id", "so-token").send_keys(H.TOKEN)
        drv.find_element("id", "so-signin").click()
        H.wait(drv, "const d = document.getElementById('signed-out'); return d && !d.open", 15)
        assert drv.execute_async_script("const done = arguments[0]; fetch('api/info', {credentials: 'same-origin'})"
                                        ".then(r => done(r.status))") == 200
        assert drv.execute_script("return window.__mace.editor.doc().dirty")
        # the browser history's old sign-in link in this signed-in browser: the app again
        drv.get(first_url)
        H.wait(drv, "return document.documentElement.dataset.appReady === '1'", 30)
        # ... and in a browser without that session: the sign-in page, not the app
        drv.delete_all_cookies()
        drv.get(first_url)
        body = drv.find_element("css selector", "body").text
        assert "This sign-in link was already used or has expired." in body
        assert drv.execute_script("return document.documentElement.dataset.appReady") != "1"
    finally:
        drv.quit()
        p.terminate()
        p.wait(15)


def test_settings_server_section_persists_across_a_restart_on_another_port(tmp_path, corpus):
    xdg = tmp_path / "xdg"
    p, base, code = start(tmp_path, xdg)
    drv = H.firefox()
    try:
        drv.get(base + "?login=" + code)
        H.wait(drv, "return document.documentElement.dataset.appReady === '1'", 60)
        # File > Settings… (the activity bar's gear is PLAN_viewer A1's fix)
        [b for b in drv.find_elements("css selector", ".menu-top") if b.text == "File"][0].click()
        H.wait(drv, "return document.querySelector(`.menu-item[data-cmd-id='file.settings']`)", 10).click()
        H.wait(drv, "return document.querySelector('.set-nav [data-sec=\"server\"]')", 15).click()
        btn = H.wait(drv, "return document.querySelector('[data-key=\"session.idleHours\"][data-val=\"4\"]')", 10)
        btn.click()
        H.wait(drv, "return window.__mace.settings.get('session.idleHours') === 4", 10)
        time.sleep(1.5)                       # the debounced save to gui.json
        saved = json.loads((xdg / "mace" / "gui.json").read_text())["settings"]
        assert saved.get("session.idleHours") == 4
    finally:
        drv.quit()
        p.terminate()
        p.wait(15)
    # another port, a fresh browser profile: the server-side value comes back
    p, base, code = start(tmp_path, xdg)
    drv = H.firefox()
    try:
        drv.get(base + "?login=" + code)
        H.wait(drv, "return document.documentElement.dataset.appReady === '1'", 60)
        assert drv.execute_script("return window.__mace.settings.get('session.idleHours')") == 4
        info = drv.execute_async_script("const done = arguments[0]; fetch('api/info', {credentials: 'same-origin'})"
                                        ".then(r => r.json()).then(done)")
        assert info["session"]["idle_hours"] == 4
    finally:
        drv.quit()
        p.terminate()
        p.wait(15)
