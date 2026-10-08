"""GPU 0 is the default, and a pre-release browser's stored GPU 1 (the old
default, which localStorage kept with the whole settings object) falls to it
once; a GPU 1 chosen under the new schema stays (polish A7).

A choice lives in gui.json (non-default keys only), the browser's copy is a
cache the server's profile overrides (polish P1-P3). So each case seeds both
the way a build of that schema left them: under version 2, GPU 1 was the
default and never reached gui.json; GPU 2, or GPU 1 under version 3, did."""
import json

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gpumig")
    proc, base, _ = H.start_server(H.examples_copy(tmp), tmp / "xdg")
    drv = H.firefox(1280, 800)
    try:
        H.load(drv, base)
        yield drv, base
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def seed(drv, base, version, obj, server):
    gpu = {"set": {"gpu_device": server}, "unset": []} if server else {"set": {}, "unset": ["gpu_device"]}
    st = drv.execute_async_script("const d = arguments[1]; window.MACE.api('profile', {body: {settings: arguments[0]}})"
                                  ".then(() => d('ok'), (e) => d(String(e)))", gpu)
    assert st == "ok", st
    drv.execute_script("localStorage.clear(); localStorage.setItem('mace.gui.settings.v1', arguments[0]);"
                       "localStorage.setItem('mace.gui.settings.version', arguments[1]);", json.dumps(obj), str(version))
    H.load(drv, base)


@pytest.mark.parametrize("version,stored,server,want", [(2, "1", None, "0"), (3, "1", "1", "1"), (2, "2", "2", "2")])
def test_gpu_device_migration(app, version, stored, server, want):
    drv, base = app
    seed(drv, base, version, {"theme": "light", "gpu_device": stored}, server)
    assert drv.execute_script("return window.__mace.settings.get('gpu_device')") == want
    # and Settings > Advanced shows it
    drv.execute_script("window.__mace.run('file.settings')")
    H.wait(drv, "return !!document.querySelector('dialog#settings[open]')", 10)
    drv.execute_script("const t = [...document.querySelectorAll('dialog#settings [role=tab], dialog#settings button')]"
                       ".find((b) => b.textContent.trim() === 'Advanced'); if (t) t.click();")
    shown = H.wait(drv, "const el = document.querySelector('dialog#settings select[data-key=\"gpu_device\"]');"
                        "return el && el.value", 5)
    drv.execute_script("document.querySelectorAll('dialog[open]').forEach((d) => d.close())")
    assert shown == want
    assert drv.execute_script("return localStorage.getItem('mace.gui.settings.version')") == "3"
