"""The browser harness under MACE_GUI_REQUIRE_BROWSER=1 (polish CI1): a
missing browser, driver or software WebGL fails the test instead of skipping
it (so the CI smoke job can never pass silently), and a hardware renderer
fails; without the variable they skip (or warn). No browser is started."""
import os
import textwrap

import pytest

pytest_plugins = ["pytester"]

HERE = os.path.dirname(os.path.abspath(__file__))
GUI = os.path.join(HERE, "gui")

TEST = textwrap.dedent("""
    import sys
    sys.path.insert(0, {gui!r})
    import guiharness as H

    def test_firefox():
        H.firefox()

    def test_chromium():
        H.chromium()
""")


def run(pytester, require, path_env):
    pytester.makepyfile(test_dummy=TEST.format(gui=GUI))
    env = {"PATH": path_env, "MACE_GUI_CHROME": "", "MACE_GUI_FIREFOX": ""}
    if require:
        env["MACE_GUI_REQUIRE_BROWSER"] = "1"
    old = {k: os.environ.get(k) for k in [*env, "MACE_GUI_REQUIRE_BROWSER"]}
    try:
        os.environ.pop("MACE_GUI_REQUIRE_BROWSER", None)
        for k, v in env.items():
            if v:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)
        return pytester.runpytest_subprocess("-p", "no:cacheprovider", "-q")
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture
def empty_path(tmp_path):
    d = tmp_path / "bin"
    d.mkdir()
    # only python itself, so the subprocess can run; no firefox, chrome or drivers
    import sys
    (d / "python").symlink_to(sys.executable)
    return str(d)


def test_required_browser_fails_instead_of_skipping(pytester, empty_path, monkeypatch):
    monkeypatch.setenv("SE_OFFLINE", "true")       # Selenium Manager must not download a browser here
    r = run(pytester, True, empty_path)
    r.assert_outcomes(failed=2)
    r.stdout.fnmatch_lines(["*browser required*"])


def test_without_the_variable_they_skip(pytester, empty_path, monkeypatch):
    monkeypatch.setenv("SE_OFFLINE", "true")
    r = run(pytester, False, empty_path)
    r.assert_outcomes(skipped=2)


def test_hardware_renderer_fails_under_the_variable(monkeypatch):
    import sys
    sys.path.insert(0, GUI)
    import guiharness as H

    class Fake:
        quit_called = False

        def quit(self):
            self.quit_called = True
    monkeypatch.setattr(H, "webgl_renderer", lambda drv: "NVIDIA GeForce RTX (ANGLE)")
    monkeypatch.setenv("MACE_GUI_REQUIRE_BROWSER", "1")
    f = Fake()
    with pytest.raises(pytest.fail.Exception, match="not software GL"):
        H.check_software_gl(f)
    assert f.quit_called
    monkeypatch.delenv("MACE_GUI_REQUIRE_BROWSER")
    with pytest.warns(UserWarning):
        assert H.check_software_gl(Fake()) == "NVIDIA GeForce RTX (ANGLE)"
    monkeypatch.setattr(H, "webgl_renderer", lambda drv: "ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero)))")
    monkeypatch.setenv("MACE_GUI_REQUIRE_BROWSER", "1")
    assert "SwiftShader" in H.check_software_gl(Fake())


def test_chromedriver_major_mismatch_is_named(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, GUI)
    import guiharness as H
    pytest.importorskip("selenium.webdriver")
    b, d = tmp_path / "fake-brave", tmp_path / "fake-chromedriver"
    b.write_text("#!/bin/sh\necho 'Brave Browser 153.1.95.104'\n")
    d.write_text("#!/bin/sh\necho 'ChromeDriver 140.0.7339.185 (abc)'\n")
    b.chmod(0o755)
    d.chmod(0o755)
    monkeypatch.setenv("MACE_GUI_CHROME", str(b))
    monkeypatch.setenv("MACE_GUI_CHROMEDRIVER", str(d))
    monkeypatch.setenv("MACE_GUI_REQUIRE_BROWSER", "1")
    with pytest.raises(pytest.fail.Exception) as e:
        H.chromium()
    assert "140.0.7339.185" in str(e.value) and "153.1.95.104" in str(e.value)
