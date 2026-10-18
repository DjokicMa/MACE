"""A folder's Setup presets in GUI Setup (1.1.6 review), Firefox and
Chromium: `[setup] opt2d12_config` (and opt2d3_config) is the settings file
`mace opt2d12` / `mace opt2d3` in that folder use when no --config-file is
given. GUI Setup starts from the CLI's own defaults (its command always names
its own file), so it says so where the preset applies, naming the file and
the key; a CIF (cif2d12, no preset key) shows nothing."""
import json
import time

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")


@pytest.fixture(scope="module", params=["firefox", "chromium"])
def drv(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp(f"preset_{request.param}")
    root = H.examples_copy(tmp)
    (root / "presets").mkdir()
    (root / "presets" / "opt.json").write_text(json.dumps({"functional": "PBE0"}))
    (root / "mace-folder.toml").write_text('[setup]\nopt2d12_config = "presets/opt.json"\n')
    proc, base, _ = H.start_server(root, tmp / "xdg")
    d = None
    try:
        d = H.browser(request.param, width=1366, height=900)
        H.load(d, base)
        yield d
    finally:
        if d is not None:
            d.quit()
        proc.terminate()
        proc.wait(timeout=15)


def setup_on(drv, path):
    drv.execute_script("if (window.__mace.ctx().setupOpen) window.__mace.run('calc.setup'); window.__mace.closeAll()")
    drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then(() => d(1), () => d(0))", path)
    time.sleep(0.5)
    drv.execute_script("window.__mace.run('calc.setup', {open: true})")
    H.wait(drv, "const s = window.__mace.setup && window.__mace.setup.state(); return s && s.result && !s.busy", 120)
    return drv.execute_script("return window.__mace.setup.state().mode")


def note(drv):
    return drv.execute_script("const n = document.querySelector('#inspector .su-preset'); return n ? n.textContent : null")


def test_setup_says_the_folder_preset_is_not_used(drv):
    assert setup_on(drv, "outputs/Si_sp.out") == "opt2d12"
    t = note(drv)
    assert t and "presets/opt.json" in t and "setup.opt2d12_config" in t and "mace opt2d12" in t, t
    assert setup_on(drv, "structures/MgO.cif") == "cif2d12"
    assert note(drv) is None
    assert not H.errors(drv), H.errors(drv)
