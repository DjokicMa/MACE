"""Folder settings in the Settings dialog (MACE 1.1.6), with real clicks: the
badge of each value's layer (This folder, This session, Mine), a change to
a value the folder sets staying in the session (never in gui.json or the
folder's file), Save to this folder, Save as my default, Reset to my
default, the "This folder" section, and a folder without a file showing no
badges at all."""
import json
import time

import pytest

import guiharness as H

from mace.utils import folder_settings as F

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")                 # the server reads the structures with it

from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402


def click(drv, el):
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
    ActionChains(drv).move_to_element(el).pause(0.1).click().perform()


def css(drv, sel):
    return drv.find_element("css selector", sel)


def open_section(drv, sec):
    drv.execute_script("window.__mace.run('file.settings')")
    H.wait(drv, "return !!document.querySelector('dialog#settings[open]')", 10)
    click(drv, css(drv, f"#settings [data-sec={sec}]"))
    H.wait(drv, f"return document.querySelector('#settings [data-sec={sec}]').classList.contains('active')", 5)


def badge(drv, key):
    return drv.execute_script(
        "const b = document.querySelector(`#settings [data-layer-row='${arguments[0]}'] .set-src`); return b ? b.textContent : null",
        key)


def get(drv, key):
    return drv.execute_script("return window.__mace.settings.get(arguments[0])", key)


def user_file(xdg):
    p = xdg / "mace" / "gui.json"
    return json.loads(p.read_text()).get("settings", {}) if p.exists() else {}


def flush(drv):
    drv.execute_async_script("const d = arguments[0]; window.__mace.settings.flush().then(() => d(1), () => d(0))")


def wait_for(cond, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.1)
    return False


@pytest.fixture
def served(tmp_path):
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    (root / F.FILENAME).write_text('# our group\'s choices\n[viewer]\natom_scale = 0.5\nprojection = "perspective"\n'
                                   '[submit]\nntasks = 16\n')
    proc, base, _ = H.start_server(root, xdg)
    drv = H.firefox(1400, 900)
    try:
        H.load(drv, base)
        yield drv, base, root, xdg
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def test_layers_badges_and_actions(served):
    drv, base, root, xdg = served
    H.local_only()
    f = root / F.FILENAME
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'", 30)
    # the folder's values are in effect, the user's own file holds none of them
    assert get(drv, "atomScale") == 0.5 and get(drv, "projection") == "perspective"
    open_section(drv, "viewport")
    assert badge(drv, "atomScale") == "This folder" and badge(drv, "projection") == "This folder"
    assert badge(drv, "showCell") == "Built-in"
    # a change here to a value the folder sets: this session only
    drv.execute_script("window.__mace.settings.set('atomScale', 0.7)")
    H.wait(drv, "return !!document.querySelector(\"#settings [data-layer-row='atomScale'] .set-src[data-src=session]\")", 5)
    flush(drv)
    assert "atomScale" not in user_file(xdg) and "atom_scale = 0.5" in f.read_text()
    # Save to this folder
    click(drv, css(drv, "#settings [data-layer-row='atomScale'] [data-layer=folder]"))
    assert wait_for(lambda: "atom_scale = 0.7" in f.read_text()), f.read_text()
    assert f.read_text().startswith("# our group's choices")
    H.wait(drv, "return document.querySelector(\"#settings [data-layer-row='atomScale'] .set-src\").dataset.src === 'folder'", 5)
    # Save as my default: into gui.json; the folder's value still applies here
    click(drv, css(drv, "#settings [data-layer-row='projection'] [data-layer=mine-save]"))
    flush(drv)
    assert wait_for(lambda: user_file(xdg).get("projection") == "perspective")
    assert badge(drv, "projection") == "This folder"
    # Reset to my default: out of the folder's file; the user's own value applies
    drv.execute_script("window.__mace.settings.set('labels', 'element')")       # an ordinary change: Mine
    flush(drv)
    assert wait_for(lambda: user_file(xdg).get("labels") == "element")
    click(drv, css(drv, "#settings [data-layer-row='atomScale'] [data-layer=mine]"))
    assert wait_for(lambda: "atom_scale = 0.7" not in f.read_text())
    H.wait(drv, "return window.__mace.settings.get('atomScale') === 0.4", 5)
    assert badge(drv, "atomScale") == "Built-in" and badge(drv, "labels") == "Mine"
    # Save to this folder on a value the folder does not set yet (shown on hover)
    click(drv, css(drv, "#settings [data-layer-row='labels'] [data-layer=folder]"))
    assert wait_for(lambda: 'labels = "element"' in f.read_text())
    H.wait(drv, "return document.querySelector(\"#settings [data-layer-row='labels'] .set-src\").dataset.src === 'folder'", 5)
    # the This folder section: the file, its GUI values and its command-line values
    click(drv, css(drv, "#settings [data-sec=folder]"))
    H.wait(drv, "return !!document.querySelector('#settings .set-folder-cli')", 5)
    text = css(drv, "#set-pane").text
    assert str(f) in text and "submit.ntasks" in text and "16" in text
    assert css(drv, "#settings .set-foot-folder").text.endswith(str(f))
    assert not H.errors(drv)


def test_no_folder_file_shows_no_badges(tmp_path):
    H.local_only()
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    proc, base, _ = H.start_server(root, xdg)
    drv = H.firefox(1400, 900)
    try:
        H.load(drv, base)
        H.wait(drv, "return document.documentElement.dataset.appReady === '1'", 30)
        open_section(drv, "viewport")
        assert drv.execute_script("return document.querySelectorAll('#settings .set-src').length") == 0
        assert css(drv, "#settings .set-foot-folder").text == ""
        drv.execute_script("window.__mace.settings.set('atomScale', 0.6)")       # saved as before: Mine
        flush(drv)
        assert wait_for(lambda: user_file(xdg).get("atomScale") == 0.6)
        assert not (root / F.FILENAME).exists()
        assert not H.errors(drv)
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)
