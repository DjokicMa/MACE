"""Settings › MACE defaults › Functional and Basis set were saved but unused in
1.1.5; 1.1.6 makes them Calculation Setup's defaults. A user who set them
before is told once, on the first 1.1.6 start (a toast that opens the
section); someone who sets them in 1.1.6 is not (the first start decides,
whatever the values were then)."""
import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")

FLAG = "notice.setupDefaults116"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("defnotice")
    proc, base, _ = H.start_server(H.examples_copy(tmp), tmp / "xdg")
    drv = H.firefox(1280, 800)
    try:
        H.load(drv, base)
        yield drv, base
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def seed(drv, base, settings):
    """gui.json as 1.1.5 left it: these settings, and no notice remembered."""
    body = {"settings": {"set": settings, "unset": [k for k in ("functional", "basisSet") if k not in settings]},
            "memory": {"unset": [FLAG]}}
    # the page's own pending profile writes first (its start remembered the notice), then the seed
    st = drv.execute_async_script("const d = arguments[1]; window.__mace.profile.flush()"
                                  ".then(() => window.MACE.api('profile', {body: arguments[0]}))"
                                  ".then(() => d('ok'), (e) => d(String(e)))", body)
    assert st == "ok", st
    drv.execute_script("localStorage.clear()")
    H.load(drv, base)


def toasts(drv):
    return drv.execute_script("return [...document.querySelectorAll('#toasts .toast2 .msg')].map((m) => m.textContent)")


def notice(drv):
    return [t for t in toasts(drv) if "Calculation Setup now uses" in t]


def test_set_in_1_1_5_is_told_once(app):
    drv, base = app
    seed(drv, base, {"functional": "PBE0", "basisSet": "POB-TZVP-REV2"})
    H.wait(drv, "return [...document.querySelectorAll('#toasts .toast2 .msg')]"
                ".some((m) => m.textContent.includes('Calculation Setup now uses'))", 10)
    (msg,) = notice(drv)
    assert "PBE0" in msg and "POB-TZVP-REV2" in msg
    assert drv.execute_script(f"return window.__mace.profile.memory('{FLAG}')") is True
    H.load(drv, base)                                                       # the next start: not again
    assert not notice(drv)


def test_nothing_set_marks_it_seen(app):
    drv, base = app
    seed(drv, base, {})
    H.wait(drv, f"return window.__mace.profile.memory('{FLAG}') === true", 10)
    assert not notice(drv)
    # set in 1.1.6: Setup uses it as Settings says, no notice
    drv.execute_script("window.__mace.settings.set('functional', 'HSE06')")
    H.load(drv, base)
    assert not notice(drv)
