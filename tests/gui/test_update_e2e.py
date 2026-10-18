"""Update MACE in a real browser (Firefox or Chromium: MACE_GUI_BROWSER), at
100 % and 110 % zoom, on the bare-origin fixture of tests/test_update.py:
the status bar's "MACE 1.1.5 available", its dialog (versions, the rendered
CHANGELOG, the pip line, the choices; inside the window, nothing cut short,
no serious axe finding), Update When I Sign Out and its undo, Update Now and
Restart (the page reloads on the restarted server, still signed in, and
says what happened), Help > Check for Updates afterwards, and a refusal
(local changes: mace update's text and the git stash line, no Update
button). Opt-in browser test (marker gui); run it under Xvfb with DISPLAY
set (MACE_GUI_XVFB names a user-local Xvfb)."""
import json
import os
import time

import pytest

import guiharness as H
from test_shutdown_api import Srv
from test_update_api import _env
from test_update import clone, git, origin  # noqa: F401 - the fixture

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")

ZOOMS = (1.0, 1.1)
W, HGT = 1366, 768


def _browser(zoom):
    name = os.environ.get("MACE_GUI_BROWSER", "firefox")
    chromium = name.startswith("chrom")
    kw = {} if chromium else {"prefs": {"layout.css.devPixelsPerPx": str(zoom)}}
    drv = H.browser(name, width=W, height=HGT, **kw)
    drv.set_script_timeout(120)
    w, h = round(W / zoom), round(HGT / zoom)
    if chromium:
        drv.execute_cdp_cmd("Emulation.setDeviceMetricsOverride",
                            {"width": w, "height": h, "deviceScaleFactor": zoom, "mobile": False})
    else:
        H.set_viewport(drv, w, h)
    return drv


def _axe_dialog(drv):
    axe_mod = pytest.importorskip("axe_selenium_python")
    axe = axe_mod.Axe(drv)
    axe.inject()
    res = drv.execute_async_script(
        "const done = arguments[arguments.length - 1];"
        "axe.run({include: [['#update-dlg']]}, {runOnly: {type: 'tag', values: ['wcag2a', 'wcag2aa']}})"
        ".then((r) => done(r), (e) => done({error: String(e)}))")
    assert "error" not in res, res
    bad = [(v["id"], [n["target"] for n in v["nodes"]]) for v in res["violations"]
           if v.get("impact") in ("serious", "critical")]
    assert not bad, bad


FITS = r"""
const d = document.getElementById('update-dlg');
if (!d || !d.open) return 'no dialog';
const r = d.getBoundingClientRect();
if (r.left < -1 || r.top < -1 || r.right > innerWidth + 1 || r.bottom > innerHeight + 1)
  return 'outside the window: ' + JSON.stringify([r.left, r.top, r.right, r.bottom, innerWidth, innerHeight]);
if (document.documentElement.scrollWidth > innerWidth + 1) return 'the page scrolls sideways';
for (const b of d.querySelectorAll('button, label, .dlg-head h2, #st-update')) {
  if (b.offsetParent === null) continue;
  if (b.scrollWidth > b.clientWidth + 1) return 'cut short: ' + (b.id || b.textContent.trim());
}
const chip = document.getElementById('st-update');
if (chip && chip.scrollWidth > chip.clientWidth + 1) return 'chip cut short';
return '';
"""


def _click(drv, sel):
    H.wait(drv, f"const e = document.querySelector({json.dumps(sel)}); return e && !e.disabled && e.offsetParent !== null", 30)
    drv.execute_script(f"document.querySelector({json.dumps(sel)}).click()")


@pytest.mark.parametrize("zoom", ZOOMS, ids=lambda z: f"{round(z * 100)}pct")
def test_update_dialog_and_update_now(origin, tmp_path, zoom):
    home = clone(origin, tmp_path / "me")
    s = Srv(tmp_path, **_env(home))
    drv = None
    try:
        st, d, _ = s.req("api/update/check", {})
        assert st == 200 and d["notice"], d
        drv = _browser(zoom)
        H.load(drv, f"http://127.0.0.1:{s.port}{s.prefix}")
        H.wait(drv, "const e = document.getElementById('st-update'); return e && e.textContent.includes('MACE 1.1.5 available')", 30)
        _click(drv, "#st-update")
        H.wait(drv, "return !!document.querySelector('#update-dlg[open] #upd-log h3')", 30)
        text = drv.execute_script("return document.getElementById('update-dlg').innerText")
        assert "1.1.4" in text and "1.1.5" in text and "What changed" in text
        assert drv.execute_script("return [...document.querySelectorAll('#upd-log h3')].map((h) => h.textContent)") == \
            ["1.1.5", "1.1.5-preview"]
        assert "pip install" in drv.execute_script("return document.getElementById('upd-pip').textContent")
        assert not drv.execute_script("return document.getElementById('upd-run-pip').checked")   # never by default
        for b in ("#upd-now", "#upd-signout", "#upd-later", "#upd-skip"):
            assert drv.execute_script(f"return !!document.querySelector('{b}')"), b
        assert drv.execute_script(FITS) == ""
        _axe_dialog(drv)
        # Update When I Sign Out, and its undo
        _click(drv, "#upd-signout")
        H.wait(drv, "return !!document.getElementById('upd-deferred')", 15)
        assert s.req("api/update")[1]["deferred"]["version"] == "1.1.5"
        _click(drv, "#upd-cancel-signout")
        H.wait(drv, "return !document.getElementById('upd-deferred') && !!document.getElementById('upd-signout')", 15)
        assert s.req("api/update")[1]["deferred"] is None
        # Update Now and Restart: the page reloads on the restarted server and says so
        drv.execute_script("window.__beforeRestart = 1")
        _click(drv, "#upd-now")
        H.wait(drv, "return window.__beforeRestart === undefined && !!window.__mace", 180)
        H.wait(drv, "return [...document.querySelectorAll('.toast2 .msg')].some((m) => m.textContent.includes('MACE was updated to 1.1.5'))", 60)
        assert git(home, "describe", "--tags") == "v1.1.5"
        assert not drv.execute_script("return !!document.getElementById('st-update')")
        assert not drv.execute_script("return !!document.querySelector('#signed-out, #stopped-page')")
        # Help > Check for Updates: up to date now
        drv.execute_script("window.__mace.run('help.checkUpdates')")
        H.wait(drv, "const e = document.getElementById('upd-uptodate'); return e && e.textContent.includes('up to date')", 60)
        assert drv.execute_script(FITS) == ""
        assert not H.errors(drv)
    except Exception:
        if drv is not None:
            H.save_failure(drv, f"update_e2e_{zoom}")
        raise
    finally:
        if drv is not None:
            drv.quit()
        s.kill()


def test_refusal_shows_the_stash_line_and_no_update_button(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    (home / "mace_cli").write_text("edited\n")
    s = Srv(tmp_path, **_env(home))
    drv = None
    try:
        drv = _browser(1.0)
        H.load(drv, f"http://127.0.0.1:{s.port}{s.prefix}")
        H.wait(drv, "return !!window.__mace", 30)
        time.sleep(0.5)
        drv.execute_script("window.__mace.run('help.checkUpdates')")
        H.wait(drv, "return !!document.querySelector('#update-dlg[open] #upd-refusal')", 60)
        assert "these files have local changes" in drv.execute_script("return document.getElementById('upd-refusal').textContent")
        assert drv.execute_script("return document.getElementById('upd-stash').textContent") == f"git -C {home} stash"
        assert not drv.execute_script("return !!document.getElementById('upd-now') || !!document.getElementById('upd-signout')")
        assert drv.execute_script(FITS) == ""
        _axe_dialog(drv)
    except Exception:
        if drv is not None:
            H.save_failure(drv, "update_e2e_refusal")
        raise
    finally:
        if drv is not None:
            drv.quit()
        s.kill()


def test_a_failed_update_ends_the_spinner_and_gives_the_buttons_back(origin, tmp_path):
    """Update Now whose `mace update` fails (the origin is gone after the check): the dialog shows the
    failure and no "Updating MACE…" spinner, and offers the choices again (it stuck until a reload)."""
    import shutil
    home = clone(origin, tmp_path / "me")
    s = Srv(tmp_path, **_env(home))
    drv = None
    gone = str(origin) + ".gone"
    try:
        st, d, _ = s.req("api/update/check", {})
        assert st == 200 and d["notice"], d
        shutil.move(str(origin), gone)
        drv = _browser(1.0)
        H.load(drv, f"http://127.0.0.1:{s.port}{s.prefix}")
        H.wait(drv, "const e = document.getElementById('st-update'); return e && e.textContent.includes('available')", 30)
        _click(drv, "#st-update")
        H.wait(drv, "return !!document.querySelector('#update-dlg[open] #upd-log h3')", 30)
        _click(drv, "#upd-now")
        H.wait(drv, "return !!document.getElementById('upd-result')", 120)
        H.wait(drv, "return !document.querySelector('#update-dlg .upd-busy .spinner')", 15)
        assert H.wait(drv, "return !!document.getElementById('upd-now')", 15)
        assert H.errors(drv) == []
    finally:
        if drv is not None:
            drv.quit()
        s.kill()
        if os.path.exists(gone):
            shutil.move(gone, str(origin))
