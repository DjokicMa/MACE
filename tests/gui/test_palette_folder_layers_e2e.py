"""Element palettes and the folder settings layers (1.1.6 review), in
Firefox and Chromium.

The palettes file's default (~/.config/mace/element_colors.json, what
`mace plotting` reads) is the user's own colorScheme. So:

* a folder's `[viewer] color_scheme` wins at load over the palettes file's
  default (the default goes into the user's layer, under the folder's);
* a change made in this session to a key the folder sets stays in the session:
  it never becomes the global default; Save as my default does;
* Save to this folder works for every preset (cvd and legacy too), and is not
  offered for a palette of the user's own, which a folder file cannot name."""
import json
import time

import pytest

import guiharness as H

from mace.utils import element_palettes as EP
from mace.utils import folder_settings as F

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")

BROWSERS = ["firefox", "chromium"]


def palettes_file(xdg):
    p = xdg / "mace" / "element_colors.json"
    return json.loads(p.read_text()) if p.exists() else {}


def wait_for(cond, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.1)
    return False


@pytest.fixture(params=BROWSERS)
def served(request, tmp_path):
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    (root / F.FILENAME).write_text('[viewer]\ncolor_scheme = "jmol"\n')
    p = xdg / "mace" / "element_colors.json"
    p.parent.mkdir(parents=True)
    doc, _ = EP.apply_op(EP.read_user(p), {"op": "save", "name": "Mine", "base": "vesta", "colors": {"O": "#ff00aa"}})
    doc, _ = EP.apply_op(doc, {"op": "default", "id": "vesta"})
    p.write_text(json.dumps(doc))
    proc, base, _ = H.start_server(root, xdg)
    drv = H.browser(request.param, width=1400, height=900)
    try:
        H.load(drv, base)
        H.wait(drv, "return document.documentElement.dataset.appReady === '1'", 30)
        yield drv, base, root, xdg
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def test_folder_color_scheme_wins_and_session_changes_stay_in_the_session(served):
    drv, base, root, xdg = served
    H.local_only()
    S = "window.__mace.settings"
    time.sleep(1.0)                                       # boot's adopt of the palettes default has run
    assert drv.execute_script(f"return [{S}.get('colorScheme'), {S}.source('colorScheme')]") == ["jmol", "folder"]
    assert drv.execute_script(f"return {S}.userValue('colorScheme')") == "vesta"     # the palettes default: the user's own
    # a session change to the key the folder sets: never the global default
    drv.execute_script(f"{S}.set('colorScheme', 'cvd')")
    assert drv.execute_script(f"return {S}.source('colorScheme')") == "session"
    time.sleep(1.5)
    assert palettes_file(xdg)["default"] == "vesta"
    # Save as my default: now it is the user's, and the palettes file follows
    drv.execute_script(f"{S}.saveAsDefault('colorScheme')")
    assert wait_for(lambda: palettes_file(xdg).get("default") == "cvd"), palettes_file(xdg)
    assert drv.execute_script(f"return {S}.get('colorScheme')") == "jmol"       # the folder's file still wins here
    # a reload: still the folder's
    H.load(drv, base)
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'", 30)
    time.sleep(1.0)
    assert drv.execute_script(f"return [{S}.get('colorScheme'), {S}.source('colorScheme')]") == ["jmol", "folder"]
    assert palettes_file(xdg)["default"] == "cvd"
    assert not H.errors(drv)


def test_save_to_this_folder_takes_every_preset_but_not_a_user_palette(served):
    drv, base, root, xdg = served
    H.local_only()
    S = "window.__mace.settings"
    f = root / F.FILENAME
    for pid in EP.preset_ids():
        drv.execute_script(f"{S}.set('colorScheme', arguments[0])", pid)
        r = drv.execute_async_script(f"const d = arguments[0]; {S}.saveToFolder('colorScheme').then(() => d('ok'), (e) => d(String(e.message || e)))")
        assert r == "ok", (pid, r)
        assert f'color_scheme = "{pid}"' in f.read_text()
    # a palette of the user's own: in the session only, and Settings offers no Save to this folder
    drv.execute_script(f"{S}.set('colorScheme', 'user:Mine')")
    drv.execute_script("window.__mace.run('file.settings')")
    H.wait(drv, "return !!document.querySelector('dialog#settings[open]')", 10)
    drv.execute_script("document.querySelector('#settings [data-sec=viewport]').click()")
    H.wait(drv, "return !!document.querySelector(\"#settings [data-layer-row='colorScheme']\")", 5)
    row = drv.find_element("css selector", "#settings [data-layer-row='colorScheme']")
    assert not row.find_elements("css selector", "[data-layer=folder]")
    assert "own palette" in row.text.lower(), row.text
    assert palettes_file(xdg)["default"] == "vesta"
    assert not H.errors(drv)
