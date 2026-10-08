"""Settings › Advanced: Export, Import and Reset of the whole GUI profile
(polish P4), with real clicks: what goes into an exported file (Recent and
per-document options only when asked), an import into a fresh profile, the
pre-1.1.5 settings file, a bad value named, Reset with a copy kept (the
newest 3), and `mace gui connect`'s connect.toml never touched."""
import hashlib
import json
import time

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")                 # the server reads the structures with it

from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402


def click(drv, el):
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
    ActionChains(drv).move_to_element(el).pause(0.05).click().perform()


def css(drv, sel):
    return drv.find_element("css selector", sel)


def advanced(drv):
    drv.execute_script("window.__mace.run('file.settings')")
    H.wait(drv, "return !!document.querySelector('dialog#settings[open]')", 10)
    click(drv, css(drv, "#settings [data-sec=advanced]"))
    H.wait(drv, "return !!document.querySelector('#settings [data-act=export]')", 5)


def parts(drv, on):
    """Tick exactly these parts in the open chooser."""
    for box in drv.find_elements("css selector", "#settings .set-confirm [data-part]"):
        if (box.get_attribute("data-part") in on) != box.is_selected():
            click(drv, box)


def export(drv, on):
    advanced(drv)
    drv.execute_script("window.__exported = null; window.__maceExportHook = (t, kind) => { if (kind === 'profile') window.__exported = t; };")
    click(drv, css(drv, "#settings [data-act=export]"))
    parts(drv, on)
    click(drv, css(drv, "#settings [data-act=export-go]"))
    return json.loads(H.wait(drv, "return window.__exported", 10))


def choose_file(drv, path):
    """Import…'s file picker, as WebDriver chooses a file: send the path to the (hidden) input."""
    inp = css(drv, "#set-import")
    drv.execute_script("arguments[0].hidden = false", inp)
    inp.send_keys(str(path))
    drv.execute_script("arguments[0].hidden = true", inp)


@pytest.fixture
def served(tmp_path):
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    (xdg / "mace").mkdir(parents=True)
    (xdg / "mace" / "connect.toml").write_text('[profiles.hpcc]\nhost = "login.example.edu"\n')
    proc, base, _ = H.start_server(root, xdg)
    drv = H.firefox(1400, 900)
    try:
        H.load(drv, base)
        yield drv, base, root, xdg, tmp_path
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def flush(drv):
    drv.execute_async_script("""const d = arguments[0]; Promise.all([window.__mace.settings.flush(), window.__mace.profile.flush(),
        window.__mace.docprefs.flush()]).then(() => d(1))""")


def toml_state(xdg):
    p = xdg / "mace" / "connect.toml"
    return hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns


def test_export_import_reset(served):
    drv, base, root, xdg, tmp = served
    connect0 = toml_state(xdg)
    drv.execute_script("window.__mace.settings.set('theme', 'dark'); window.__mace.settings.set('labelSize', 14);"
                       "window.__mace.profile.setUi('paletteRecent', ['view.palette']); window.__mace.profile.remember('cleanMethod', 'springs');")
    drv.execute_async_script("const d = arguments[0]; window.__mace.open('structures/MgO.cif').then(() => d(1))")
    drv.execute_script("window.__mace.display.set(window.__mace.activeDoc(), {displayHi: [2, 2, 2]})")
    flush(drv)
    # Export with the default parts: no Recent, no per-document options
    out = export(drv, {"settings", "ui", "memory"})
    assert out["mace_gui_profile"] == 2 and out["settings"]["theme"] == "dark" and out["memory"].get("cleanMethod") == "springs", out
    assert "recent" not in out and "docprefs" not in out and "connect" not in json.dumps(out)
    full = export(drv, {"settings", "ui", "memory", "recent", "docprefs"})
    assert full["recent"]["files"] and full["docprefs"]["display"]
    f = tmp / "profile.json"
    f.write_text(json.dumps({**full, "connect": {"profiles": {"x": {}}}}))

    # Reset (defaults: settings, layout, remembered choices): Recent stays, a copy is kept
    advanced(drv)
    click(drv, css(drv, "#settings [data-act=reset-all]"))
    assert "A copy of your current profile is saved first" in css(drv, "#settings .set-confirm").text
    flush(drv)
    time.sleep(0.6)
    before = (xdg / "mace" / "gui.json").read_bytes()
    click(drv, css(drv, "#settings [data-act=reset-yes]"))
    time.sleep(0.8)
    flush(drv)
    assert drv.execute_script("return window.__mace.settings.get('theme')") == "system"
    assert drv.execute_script("return window.__mace.profile.memory('cleanMethod') === undefined")
    assert drv.execute_script("return window.__mace.profile.recentFiles().length") >= 1
    copies = sorted((xdg / "mace").glob("gui.json.before-reset-*.json"))
    assert len(copies) == 1 and copies[0].read_bytes() == before
    assert (copies[0].stat().st_mode & 0o777) == 0o600

    # Import the full file: confirm box, then everything is back; the connect section is left out and said so
    advanced(drv)
    choose_file(drv, f)
    H.wait(drv, "return !!document.querySelector('#settings [data-act=import-go]')", 10)
    what = css(drv, "#settings .set-import-what").text
    assert "Settings: 2 values" in what and "Layout" in what and "Recent files" in what, what
    click(drv, css(drv, "#settings [data-act=import-go]"))
    H.wait(drv, "return window.__mace.settings.get('theme') === 'dark'", 10)
    H.wait(drv, "return [...document.querySelectorAll('#toasts > *')].some((t) => t.textContent.includes('connect profiles were left out'))", 10)
    assert drv.execute_script("return window.__mace.profile.memory('cleanMethod')") == "springs"

    # the pre-1.1.5 settings file, and a bad value named while the others apply
    old = tmp / "old.json"
    old.write_text(json.dumps({"version": 2, "settings": {"labelSize": 16, "cache_mb": 5}}))
    choose_file(drv, old)
    H.wait(drv, "return !!document.querySelector('#settings [data-act=import-go]')", 10)
    click(drv, css(drv, "#settings [data-act=import-go]"))
    H.wait(drv, "return window.__mace.settings.get('labelSize') === 16", 10)
    flush(drv)
    time.sleep(0.5)
    toasts = drv.execute_script("return [...document.querySelectorAll('#toasts > *')].map((t) => t.textContent).join(' | ')")
    assert "Couldn't save Cache limit" in toasts, toasts

    # four more resets: exactly the newest 3 copies stay
    for _ in range(4):
        advanced(drv)
        click(drv, css(drv, "#settings [data-act=reset-all]"))
        click(drv, css(drv, "#settings [data-act=reset-yes]"))
        time.sleep(1.1)                      # copies are named by the second
    copies = sorted((xdg / "mace").glob("gui.json.before-reset-*.json"))
    assert len(copies) == 3, copies
    # connect.toml: never exported, imported or reset
    assert toml_state(xdg) == connect0
    assert not H.errors(drv), H.errors(drv)
