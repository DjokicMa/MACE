"""MACE GUI in its own app window (1.1.6): a Chromium-family browser in app
mode (--app=<url>, a profile of its own, the window class MACE on Linux),
found in the order Brave, Chrome, Chromium, Edge, Vivaldi, else a browser
tab; --browser tab|app and [gui] open pick one. The page is an installable
web app: a manifest (name MACE, MACE's icons at 192 and 512 px, theme colors
from tokens.css) whose start_url and scope are the session path /s/<nonce>/.

Fake browsers only: every program the tests find or start is a stub that
records its arguments; PATH holds nothing else."""
import http.client
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import pytest

import guiharness as H
from mace.gui import appwindow as A

REPO = Path(__file__).resolve().parents[2]
SH = REPO / "scripts" / "mace-gui-connect.sh"
PS1 = REPO / "scripts" / "mace-gui-connect.ps1"
STATIC = REPO / "mace" / "gui" / "static"
posix_only = pytest.mark.skipif(os.name == "nt", reason="POSIX stub browsers")


def _stub(path: Path, marker: Path):
    path.write_text(f"#!{sys.executable}\nimport sys\nprint('chatter', file=sys.stderr)\n"
                    f"open({str(marker)!r}, 'a').write('\\0'.join(sys.argv) + '\\n')\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _fakebin(tmp_path, *names):
    fb = tmp_path / "fakebin"
    fb.mkdir(exist_ok=True)
    marker = tmp_path / "launched.txt"
    for n in names:
        _stub(fb / n, marker)
    return fb, marker


def _wait(path: Path, s=15.0) -> str:
    end = time.monotonic() + s
    while time.monotonic() < end and not (path.exists() and path.read_text()):
        time.sleep(0.05)
    return path.read_text() if path.exists() else ""


# ---- finding the browser

@posix_only
@pytest.mark.parametrize("names,want", [
    (("vivaldi", "chromium"), "Chromium"),
    (("microsoft-edge", "vivaldi-stable"), "Edge"),
    (("google-chrome-stable", "chromium-browser", "brave-browser"), "Brave"),
    (("vivaldi",), "Vivaldi"),
    (("firefox",), None),
])
def test_detection_order_with_fake_programs_on_path(tmp_path, names, want):
    fb, _ = _fakebin(tmp_path, *names)
    got = A.find_app_browser("linux", {"PATH": str(fb), "HOME": str(tmp_path)})
    if want is None:
        assert got is None
    else:
        assert got[0] == want and Path(got[1]).parent == fb


def test_detection_on_windows_install_folders():
    seen = []

    def isfile(p):
        seen.append(p)
        return p in have
    env = {"ProgramFiles": r"C:\Program Files", "ProgramFiles(x86)": r"C:\Program Files (x86)",
           "LOCALAPPDATA": r"C:\Users\me\AppData\Local", "PATH": ""}
    have = {r"C:\Users\me\AppData\Local\BraveSoftware\Brave-Browser\Application\brave.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"}
    got = A.find_app_browser("windows", env, which=lambda *a, **k: None, isfile=isfile)
    assert got == ("Brave", r"C:\Users\me\AppData\Local\BraveSoftware\Brave-Browser\Application\brave.exe")
    assert seen[0] == r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
    have.discard(got[1])
    assert A.find_app_browser("windows", env, which=lambda *a, **k: None, isfile=isfile)[0] == "Edge"
    # on PATH only (a portable install)
    got = A.find_app_browser("windows", {"PATH": r"C:\tools"}, which=lambda c, path=None: rf"C:\tools\{c}.exe"
                             if c == "chrome" else None, isfile=lambda p: False)
    assert got == ("Chrome", r"C:\tools\chrome.exe")


def test_detection_on_macos_app_bundles():
    have = {"/home/me/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Vivaldi.app/Contents/MacOS/Vivaldi"}
    got = A.find_app_browser("darwin", {"HOME": "/home/me", "PATH": ""}, which=lambda *a, **k: None,
                             isfile=lambda p: p in have)
    assert got == ("Chrome", "/home/me/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


# ---- the command line

@pytest.mark.parametrize("system,cls", [("linux", True), ("windows", False), ("darwin", False)])
def test_app_command_per_system(system, cls):
    url = "http://127.0.0.1:8765/s/abc/?login=xyz&x=1"
    argv = A.app_command("/opt/brave", url, "/cache/mace/gui/app-profile", system)
    assert argv[:5] == ["/opt/brave", f"--app={url}", "--user-data-dir=/cache/mace/gui/app-profile",
                        "--no-first-run", "--no-default-browser-check"]
    assert ("--class=MACE" in argv) is cls and len(argv) == 5 + cls


def test_profile_and_log_live_in_mace_guis_cache(tmp_path):
    env = {"XDG_CACHE_HOME": str(tmp_path / "c"), "HOME": str(tmp_path)}
    assert A.profile_dir(env) == tmp_path / "c" / "mace" / "gui" / "app-profile"
    assert A.log_path(env).parent == A.profile_dir(env).parent


@posix_only
def test_a_snap_browser_gets_a_profile_it_may_write_and_a_tab_when_it_fails(tmp_path, monkeypatch):
    """A snap browser (/snap/bin/chromium) may not write ~/.cache: its app profile is the snap's own
    ~/snap/<name>/common, and an app window that exits with an error at once opens a tab instead."""
    assert A.snap_name("/snap/bin/chromium") == "chromium"
    assert A.snap_name("/usr/bin/chromium", realpath=lambda p: p) is None
    assert A.snap_name("/usr/bin/chromium-browser", realpath=lambda p: "/snap/chromium/2900/usr/lib/chromium-browser/chrome") == "chromium"
    assert A.snap_profile_dir("chromium", {"HOME": str(tmp_path)}) == tmp_path / "snap" / "chromium" / "common" / "mace-app-profile"
    # spawn's check: an error exit within it is a failure; a hand-off to a running browser (exit 0) or a
    # browser still running when it ends is a window
    env = {"PATH": "/usr/bin:/bin", "MACE_GUI_STATE_DIR": str(tmp_path / "state")}
    for name, body in (("fails", "exit 1"), ("hands_off", "exit 0"), ("runs", "sleep 5")):
        (tmp_path / name).write_text(f"#!/bin/sh\n{body}\n")
        (tmp_path / name).chmod(0o755)
    assert A.spawn([str(tmp_path / "fails")], env, check=3.0) is False
    assert A.spawn([str(tmp_path / "hands_off")], env, check=3.0) is True
    t0 = time.monotonic()
    assert A.spawn([str(tmp_path / "runs")], env, check=0.5) is True and time.monotonic() - t0 < 4
    assert A.spawn([str(tmp_path / "fails")], env) is True                    # no check: not waited for
    # mace gui --open with a snap Chromium whose window fails: its own profile, then a tab
    calls = []
    monkeypatch.setattr(A, "system_name", lambda: "linux")
    monkeypatch.setattr(A, "find_app_browser", lambda system=None, env=None: ("Chromium", "/snap/bin/chromium"))
    monkeypatch.setattr(A, "spawn", lambda argv, env=None, check=0.0: calls.append((argv, check)) and False)
    monkeypatch.setattr(A, "open_tab", lambda url, env=None: calls.append(("tab", url)) is None)
    assert A.open_url("http://127.0.0.1:9/s/x/", "app", {"HOME": str(tmp_path), "PATH": ""})
    argv, check = calls[0]
    assert check == A.SNAP_CHECK_S and f"--user-data-dir={tmp_path}/snap/chromium/common/mace-app-profile" in argv
    assert calls[1] == ("tab", "http://127.0.0.1:9/s/x/")
    # a browser that is not a snap: the cache folder's profile, and no wait
    calls.clear()
    monkeypatch.setattr(A, "find_app_browser", lambda system=None, env=None: ("Chromium", "/usr/bin/chromium"))
    monkeypatch.setattr(A, "snap_name", lambda program, realpath=None: None)
    monkeypatch.setattr(A, "spawn", lambda argv, env=None, check=0.0: calls.append((argv, check)) is None)
    assert A.open_url("http://127.0.0.1:9/s/x/", "app", {"HOME": str(tmp_path), "PATH": "", "XDG_CACHE_HOME": str(tmp_path / "c")})
    assert calls[0][1] == 0.0 and f"--user-data-dir={tmp_path}/c/mace/gui/app-profile" in calls[0][0] and len(calls) == 1


def test_mode_resolution():
    never = lambda: pytest.fail("the setting is not read")      # noqa: E731
    assert A.resolve_mode("tab", {"BROWSER": ""}, never) == "tab"
    assert A.resolve_mode("app", {"BROWSER": "/vscode/browser.sh"}, never) == "app"
    assert A.resolve_mode(None, {"BROWSER": "/vscode/browser.sh"}, never) == "tab"
    assert A.resolve_mode(None, {}, lambda: "tab") == "tab"
    assert A.resolve_mode(None, {}, lambda: None) == "app"


def test_gui_open_setting_from_site_toml(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg" / "mace"
    cfg.mkdir(parents=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("HOME", str(tmp_path))
    for k in ("MACE_SITE_CONFIG", "MACE_SITE_GROUP_CONFIG"):
        monkeypatch.delenv(k, raising=False)
    (cfg / "site.toml").write_text('[gui]\nopen = "tab"\n')
    (cfg / "site.toml").chmod(0o600)
    assert A.setting_mode() == "tab"
    (cfg / "site.toml").write_text('[gui]\nopen = "window"\n')
    assert A.setting_mode() is None


# ---- opening: the real invocation paths, with stub browsers only

def _child_env(tmp_path, fb, **extra):
    env = {k: v for k, v in os.environ.items()
           if k not in ("BROWSER", "PYTHONPATH", "MACE_HOME", "MACE_GUI_STATE_DIR", "COLORTERM", "TERM", "NO_COLOR",
                        "SSH_CONNECTION", "TERM_PROGRAM", "VSCODE_IPC_HOOK_CLI", "MACE_GUI_TOKEN", "MACE_GUI_PORT")}
    (tmp_path / "home").mkdir(exist_ok=True)
    env.update(PATH=str(fb), HOME=str(tmp_path / "home"), XDG_CACHE_HOME=str(tmp_path / "cache"),
               XDG_CONFIG_HOME=str(tmp_path / "cfg"), PYTHONDONTWRITEBYTECODE="1", MACE_GUI_NO_WARMUP="1",
               DISPLAY=":0")
    env.update(extra)
    return env


@posix_only
def test_mace_gui_open_starts_an_app_window(tmp_path):
    if not sys.platform.startswith("linux"):
        pytest.skip("the Linux command names")
    fb, marker = _fakebin(tmp_path, "chromium", "xdg-open")
    p = subprocess.Popen([sys.executable, str(H.MACE_CLI), "--no-banner", "gui", "--dir", str(tmp_path), "--port",
                          "0", "--open"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=_child_env(tmp_path, fb, MACE_GUI_PRINT_TOKEN="1"))
    try:
        argv = _wait(marker).strip().split("\0")
        time.sleep(0.3)
    finally:
        p.terminate()
        out = p.communicate(timeout=15)[0]
    assert argv[0] == str(fb / "chromium"), argv               # not xdg-open: the app window
    assert re.fullmatch(r"--app=http://127\.0\.0\.1:\d+/s/[^/]+/\?login=\S+", argv[1]), argv
    assert argv[2:] == [f"--user-data-dir={tmp_path / 'cache' / 'mace' / 'gui' / 'app-profile'}", "--no-first-run",
                        "--no-default-browser-check", "--class=MACE"]
    assert (tmp_path / "cache" / "mace" / "gui" / "app-profile").is_dir()
    assert "Opened MACE GUI in its own Chromium window (--browser tab opens a browser tab instead)." in out
    assert "chatter" not in out


@posix_only
@pytest.mark.parametrize("how", ["flag", "setting", "no-chromium"])
def test_tab_mode_uses_the_default_browser(tmp_path, how):
    if not sys.platform.startswith("linux"):
        pytest.skip("xdg-open is the Linux desktop opener")
    fb, marker = _fakebin(tmp_path, "xdg-open", *(() if how == "no-chromium" else ("brave-browser",)))
    env = _child_env(tmp_path, fb)
    args = ["--browser", "tab"] if how == "flag" else []
    if how == "setting":
        (tmp_path / "cfg" / "mace").mkdir(parents=True)
        (tmp_path / "cfg" / "mace" / "site.toml").write_text('[gui]\nopen = "tab"\n')
        (tmp_path / "cfg" / "mace" / "site.toml").chmod(0o600)
    code = ("import sys; from mace.gui import cli; "
            f"print(cli.open_browser('http://127.0.0.1:1/s/x/', None, {args[1] if args else None!r}))")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=env, capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0 and r.stdout.strip().endswith("True"), r.stderr
    argv = _wait(marker).strip().split("\0")
    assert argv == [str(fb / "xdg-open"), "http://127.0.0.1:1/s/x/"]


def test_the_flags_exist():
    from mace.gui import cli, connect
    assert cli.build_parser().parse_args(["--open", "--browser", "tab"]).browser == "tab"
    assert connect.build_parser().parse_args(["--browser", "app"]).browser == "app"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--browser", "window"])


@posix_only
def test_connect_open_url_uses_the_app_window(tmp_path):
    if not sys.platform.startswith("linux"):
        pytest.skip("the Linux command names")
    fb, marker = _fakebin(tmp_path, "vivaldi")
    code = ("from mace.gui import connect; said = []; "
            "print(connect.open_url('http://127.0.0.1:2/s/y/?login=1', None, said.append), said)")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=_child_env(tmp_path, fb), capture_output=True,
                       text=True, timeout=60)
    assert r.stdout.startswith("True") and "own Vivaldi window" in r.stdout, r.stderr
    assert _wait(marker).split("\0")[1] == "--app=http://127.0.0.1:2/s/y/?login=1"


# ---- the standalone scripts

@posix_only
def test_bash_script_opens_an_app_window(tmp_path):
    """The script's open_browser with a stub Chrome; the same command line as appwindow.app_command."""
    if not sys.platform.startswith("linux"):
        pytest.skip("the Linux command names")
    fb, marker = _fakebin(tmp_path, "google-chrome")
    for tool in ("bash", "uname", "mkdir", "setsid", "nohup", "dirname"):
        w = shutil.which(tool)
        if w:
            os.symlink(w, fb / tool)
    sh = SH.read_text()
    funcs = sh[sh.index("app_browser() {"):sh.index("json_field() {")]
    url = "http://127.0.0.1:8765/s/n/?login=c"
    script = (f"OPEN=1 BROWSER_MODE=app\nwinpath() {{ printf '%s' \"$1\"; }}\n{funcs}\nopen_browser {url!r}\nwait\n")
    r = subprocess.run(["bash", "-c", script], env=_child_env(tmp_path, fb), capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    argv = _wait(marker).strip().split("\0")
    want = A.app_command(str(fb / "google-chrome"), url, str(tmp_path / "cache" / "mace" / "gui" / "app-profile"),
                         "linux")
    assert argv == want
    assert "Opened MACE GUI in its own Chrome window (--browser tab opens a browser tab instead)." in r.stdout
    assert "chatter" not in r.stdout + r.stderr


def test_scripts_take_the_browser_option():
    sh, ps = SH.read_text(), PS1.read_text()
    assert re.search(r"^\s+--browser\|--browser=\*\)", sh, re.M) and "BROWSER_MODE" in sh
    assert '[ValidateSet("", "app", "tab")][string]$Browser = ""' in ps and "Open-Gui $url" in ps
    for name in ("Brave", "Chrome", "Chromium", "Edge", "Vivaldi"):
        assert sh.index(f'"{name}|') < sh.index('"Vivaldi|') or name == "Vivaldi"
    order = [m.group(1) for m in re.finditer(r'@\("(\w+)", "[^"]+\.exe"', ps)]
    assert order == [b[0] for b in A.APP_BROWSERS]
    for _, _, _, winpath in A.APP_BROWSERS:
        assert winpath in ps and winpath.replace("\\", "/") in sh, winpath
    assert "--user-data-dir=" in sh and "--user-data-dir=" in ps and "--class=MACE" in sh and "--class=MACE" in ps


PWSH = shutil.which("pwsh")


@pytest.mark.skipif(PWSH is None or os.name == "nt", reason="needs pwsh (PowerShell 7) on Linux")
def test_powershell_opens_an_app_window(tmp_path):
    fb, marker = _fakebin(tmp_path, "msedge")
    ps = PS1.read_text()
    start = ps.index("$AppBrowsers = @(")
    funcs = ps[start:ps.index("$psi = New-Object System.Diagnostics.ProcessStartInfo", start)]
    win_quote = ps[ps.index("function Win-Quote"):ps.index("# ssh.exe, even when PATH lost it")]
    url = "http://127.0.0.1:8765/s/n/?login=c"
    script = (f'$ErrorActionPreference = "Stop"\n$OnWindows = $false\n$BrowserMode = "app"\n{win_quote}\n{funcs}\n'
              f"Open-Gui '{url}'\n")
    path = tmp_path / "t.ps1"
    path.write_text(script)
    env = _child_env(tmp_path, fb)
    r = subprocess.run([PWSH, "-NoProfile", "-NonInteractive", "-File", str(path)], env=env, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    argv = _wait(marker).strip().split("\0")
    assert argv == A.app_command(str(fb / "msedge"), url, str(tmp_path / "cache" / "mace" / "gui" / "app-profile"),
                                 "linux")
    assert "Opened MACE GUI in its own Edge window (-Browser tab opens a browser tab instead)." in r.stdout


# ---- the web app manifest

def _png_size(data: bytes):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def _tokens(scheme_dark: bool) -> str:
    css = (STATIC / "css" / "tokens.css").read_text()
    if scheme_dark:
        block = css[css.index("prefers-color-scheme: dark"):]
    else:
        block = css[css.index(":root"):]
    return re.search(r"--surface:\s*(#[0-9a-fA-F]{6})", block).group(1)


def test_manifest_file_is_valid_and_matches_the_tokens():
    m = json.loads((STATIC / "manifest.webmanifest").read_text())
    assert m["name"] == m["short_name"] == "MACE" and m["display"] == "standalone"
    assert m["start_url"] == m["scope"] == "../../"       # from static/<version>/: the session path
    assert m["theme_color"] == _tokens(False)
    sizes = {}
    for icon in m["icons"]:
        f = STATIC / icon["src"]
        assert f.is_file(), icon
        if icon["type"] == "image/png":
            w, h = _png_size(f.read_bytes())
            assert icon["sizes"] == f"{w}x{h}"
            sizes[w] = icon
    assert {192, 512} <= set(sizes)
    index = (STATIC / "index.html").read_text()
    # before the page's script sets it, the title bar takes the theme's --surface (light and dark), not white/black
    assert f'<meta name="theme-color" content="{_tokens(False)}" media="(prefers-color-scheme: light)">' in index
    assert f'<meta name="theme-color" content="{_tokens(True)}" media="(prefers-color-scheme: dark)">' in index
    assert _tokens(True) != _tokens(False)


class _Srv:
    def __init__(self, tmp_path):
        self.p, self.base, _ = H.start_server(tmp_path, tmp_path / "xdg", MACE_GUI_NO_WARMUP="1")
        u = urlsplit(self.base)
        self.host, self.port, self.prefix = u.hostname, u.port, u.path

    def get(self, path, token=True):
        c = http.client.HTTPConnection(self.host, self.port, timeout=30)
        c.request("GET", path, headers={"X-Mace-Token": H.TOKEN} if token else {})
        r = c.getresponse()
        body = r.read()
        c.close()
        return r.status, r.getheader("Content-Type") or "", body

    def stop(self):
        self.p.terminate()
        self.p.wait(15)


@pytest.fixture
def srv(tmp_path):
    s = _Srv(tmp_path)
    yield s
    s.stop()


def test_manifest_is_served_under_the_session_path_with_its_icons(srv):
    st, _, page = srv.get(srv.prefix)
    assert st == 200
    href = re.search(rb'<link rel="manifest" href="([^"]+)" crossorigin="use-credentials">', page).group(1).decode()
    murl = urljoin(srv.prefix, href)
    assert re.fullmatch(r"/s/[^/]+/static/v-[0-9a-f]+/manifest\.webmanifest", murl), murl
    st, ctype, body = srv.get(murl)
    assert st == 200 and ctype.startswith("application/manifest+json"), ctype
    m = json.loads(body)
    # start_url and scope resolve to the session's own path: the token-in-path scheme keeps working
    assert urljoin(murl, m["start_url"]) == srv.prefix and urljoin(murl, m["scope"]) == srv.prefix
    for icon in m["icons"]:
        st, ctype, data = srv.get(urljoin(murl, icon["src"]))
        assert st == 200 and ctype.startswith(icon["type"]), icon
        if icon["type"] == "image/png":
            assert "%dx%d" % _png_size(data) == icon["sizes"]
    # like every file of the session, it needs the session: no token, no manifest
    assert srv.get(murl, token=False)[0] == 403
    for rel in ("apple-touch-icon", "icon"):
        assert re.search(rf'<link rel="{rel}" href="static/v-[0-9a-f]+/brand/'.encode(), page), rel


@pytest.mark.gui
def test_page_links_the_manifest_in_firefox(tmp_path):
    proc, base, _ = H.start_server(tmp_path, tmp_path / "xdg", MACE_GUI_NO_WARMUP="1")
    drv = H.firefox()
    try:
        H.load(drv, base)
        assert drv.title.endswith("MACE"), drv.title
        got = drv.execute_async_script(
            "const done = arguments[0]; const l = document.querySelector('link[rel=manifest]');"
            "fetch(l.href, {credentials: 'include'}).then((r) => r.json().then((m) => done("
            "[l.crossOrigin, r.status, r.headers.get('content-type'), m.name, new URL(m.start_url, l.href).pathname,"
            " location.pathname, [...document.querySelectorAll('link[rel=icon]')].map((x) => x.sizes.value)])))"
            ".catch((e) => done(String(e)))")
        assert isinstance(got, list), got
        cross, status, ctype, name, start, here, sizes = got
        assert cross == "use-credentials" and status == 200 and ctype.startswith("application/manifest+json")
        assert name == "MACE" and start == here, (start, here)
        assert "192x192" in sizes
        # the app window's title bar follows the theme: the menu bar's --surface
        read = ("const s = getComputedStyle(document.querySelector('.menubar')).backgroundColor;"
                "return [s, [...document.querySelectorAll('meta[name=theme-color]')].map((m) => [m.content, m.media])]")
        for theme in ("dark", "light"):
            drv.execute_script("window.MACE.applyTheme(arguments[0])", theme)
            bar, metas = drv.execute_script(read)
            probe = drv.execute_script("const d = document.createElement('div'); d.style.color = arguments[0];"
                                       "document.body.append(d); const c = getComputedStyle(d).color; d.remove();"
                                       "return c", metas[0][0])
            assert all(m[1] == "" for m in metas) and probe == bar, (theme, metas, bar)
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(15)
