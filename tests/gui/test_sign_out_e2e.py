"""Sign Out and Stop Server and Help > Connecting from Your Computer with real
clicks in a fresh headless Firefox (temp copies of real test/ files): the
stopped page in both tabs, the save prompt first, Stop When Tasks Finish with
its banner and Cancel Stop, the command search finding "logout", and the
tunnel dialog's node, lines, Copy buttons and VS Code variant."""
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
FILES = ["OPT/1_dia_opt_rev1.out", "CIFs/1_dia_opt_rev1.cif"]


def start(tmp, **env):
    if not all((CORPUS / f).is_file() for f in FILES):
        pytest.skip("test/ corpus not present")
    root = tmp / "calc"
    root.mkdir()
    for f in FILES:
        shutil.copy(CORPUS / f, root / Path(f).name)
    e = H.env(tmp / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1", **env)
    p = H.ServerProcess([sys.executable, str(H.MACE_CLI), "--no-banner", "gui", "--dir", str(root), "--port", "0",
                         "--token", H.TOKEN, "--gateway", "gw.example.edu"], stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=e,
                        start_new_session=True, preexec_fn=H.die_with_pytest)
    lines = []
    for _ in range(40):
        line = p.stdout.readline()
        lines.append(line)
        m = re.search(H.URL_RE, line)
        if m:
            return p, m.group(1), root
    raise AssertionError("".join(lines))


@pytest.fixture
def app(tmp_path):
    p, base, root = start(tmp_path)
    drv = H.firefox()
    yield drv, base, p, root
    drv.quit()
    if p.poll() is None:
        p.terminate()
        p.wait(15)


def menu(drv, label, cmd):
    [b for b in drv.find_elements("css selector", ".menu-top") if b.text == label][0].click()
    H.wait(drv, "return document.querySelector(`.menu-item[data-cmd-id='${arguments[0]}']`)", 10, cmd).click()


def file_menu_sign_out(drv):
    menu(drv, "File", "file.signOut")
    H.wait(drv, "const d = document.getElementById('sign-out'); return d && d.open", 15)


def ready(drv, base):
    H.load(drv, base)


def test_sign_out_stops_the_server_and_both_tabs_show_it(app):
    drv, base, p, root = app
    ready(drv, base)
    first = drv.current_window_handle
    drv.switch_to.new_window("tab")
    drv.get(base)
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'", 60)
    second = drv.current_window_handle
    drv.switch_to.window(first)
    file_menu_sign_out(drv)
    text = drv.find_element("css selector", "#sign-out .dlg-body").text
    assert "This signs you out and stops MACE GUI on" in text and "Your SLURM jobs keep running" in text
    drv.find_element("css selector", "#so-now").click()
    H.wait(drv, "const e = document.getElementById('stopped-page'); return e && e.textContent.includes("
                "'MACE GUI has stopped')", 10)
    assert "another window" not in drv.find_element("id", "stopped-text").text
    assert drv.find_element("id", "stopped-cmd").text == f"mace gui --dir {root}"
    p.wait(10)
    assert p.returncode == 0
    drv.switch_to.window(second)
    H.wait(drv, "const e = document.getElementById('stopped-text'); return e && e.textContent.includes("
                "'It was stopped from another window')", 10)
    assert not drv.execute_script("const b = document.getElementById('conn-banner'); return b && !b.classList.contains('hidden')")


@H.needs_ase
def test_modified_document_asks_first(app):
    drv, base, p, root = app
    ready(drv, base)
    drv.execute_async_script("const done = arguments[1]; window.__mace.open(arguments[0]).then(() => done())",
                             "1_dia_opt_rev1.cif")
    H.wait(drv, "return window.__mace.viewport && window.__mace.viewport.displayed() > 0", 30)
    drv.execute_script("window.__mace.run('build.symModeP1')")
    drv.execute_script("const E = window.__mace.editor.E; E.api.edit(window.__mace.editor.doc(), 'replace', "
                       "{indices: [0], element: 'Si'}, {})")
    H.wait(drv, "const e = window.__mace.editor.doc(); return e && e.dirty && e.pending === 0", 20)
    menu(drv, "File", "file.signOut")
    H.wait(drv, "const d = document.getElementById('dirty-first'); return d && d.open", 10)
    drv.find_element("css selector", '#dirty-first [data-r="cancel"]').click()
    time.sleep(0.5)
    assert not drv.execute_script("const d = document.getElementById('sign-out'); return d && d.open")
    assert p.poll() is None


def test_stop_when_tasks_finish_banner_and_cancel_stop(app):
    drv, base, p, root = app
    ready(drv, base)
    run_task = ("const done = arguments[1]; window.__mace.state ? 0 : 0; fetch('api/test/task', {method: 'POST', "
                "credentials: 'same-origin', headers: {'X-Requested-With': 'mace-gui', 'Content-Type': 'application/json'}, "
                "body: JSON.stringify({seconds: arguments[0], job: 'sleep', wait: 0, title: 'Long test task'})})"
                ".then(r => done(r.status))")
    assert drv.execute_async_script(run_task, 6) == 202
    file_menu_sign_out(drv)
    assert "1 task is running: Long test task" in drv.find_element("id", "so-tasks").text
    drv.find_element("id", "so-after").click()
    banner = H.wait(drv, "const b = document.getElementById('stop-banner'); return b && b.textContent", 10)
    assert "MACE GUI stops when its task finishes" in banner
    # the session keeps working meanwhile
    assert drv.execute_async_script("const done = arguments[0]; fetch('api/info', {credentials: 'same-origin'})"
                                    ".then(r => done(r.status))") == 200
    drv.find_element("id", "stop-cancel").click()
    H.wait(drv, "return !document.getElementById('stop-banner')", 10)
    time.sleep(6 + 3)
    assert p.poll() is None, "Cancel Stop kept the server"
    # a second time without Cancel Stop: it stops after the task
    assert drv.execute_async_script(run_task, 3) == 202
    file_menu_sign_out(drv)
    drv.find_element("id", "so-after").click()
    H.wait(drv, "return !!document.getElementById('stop-banner')", 10)
    p.wait(3 + 10)
    assert p.returncode == 0
    H.wait(drv, "return !!document.getElementById('stopped-page')", 10)


def test_palette_finds_logout_and_the_tunnel_dialog(app):
    from selenium.webdriver.common.keys import Keys
    drv, base, p, root = app
    ready(drv, base)
    drv.execute_script("window.__mace.palette.open('')")
    box = drv.find_element("css selector", "#palette input")
    box.send_keys("logout")
    time.sleep(0.3)
    box.send_keys(Keys.ENTER)
    H.wait(drv, "const d = document.getElementById('sign-out'); return d && d.open", 10)
    drv.find_element("css selector", '#sign-out [data-r="cancel"]').click()
    # Help > Connecting from Your Computer
    menu(drv, "Help", "help.tunnel")
    H.wait(drv, "const d = document.getElementById('tunnel'); return d && d.open", 10)
    d = drv.find_element("id", "tunnel")
    assert d.find_element("css selector", "h2").text == "Connecting from Your Computer"
    connect = drv.find_element("id", "cn-connect").text
    assert connect.startswith("mace gui connect --gateway ") and "@gw.example.edu --node " in connect
    assert str(root) in connect
    assert drv.find_element("id", "cn-ssh").text.startswith("ssh -N -J ")
    assert "--hop nested" in drv.find_element("id", "cn-hop").text
    assert "laptop" not in d.text.lower() and "(ssh -L)" not in d.text
    drv.find_element("css selector", '#tunnel [data-copy="cn-connect"]').click()
    assert H.wait(drv, "return window.__maceLastCopy", 5) == connect
    drv.find_element("css selector", '#tunnel [data-copy="cn-ssh"]').click()
    assert H.wait(drv, "return window.__maceLastCopy !== arguments[0] && window.__maceLastCopy", 5,
                  connect).startswith("ssh -N -J ")


def test_tunnel_dialog_vscode_variant(tmp_path):
    p, base, root = start(tmp_path, TERM_PROGRAM="vscode")
    drv = H.firefox()
    try:
        ready(drv, base)
        menu(drv, "Help", "help.tunnel")
        H.wait(drv, "const d = document.getElementById('tunnel'); return d && d.open", 10)
        txt = drv.find_element("id", "cn-vscode").text
        assert "Ctrl+click" in txt and "The Ports tab opens the bare address, which needs the token." in txt
        assert not drv.find_elements("id", "cn-ssh")
    finally:
        drv.quit()
        p.terminate()
        p.wait(15)
