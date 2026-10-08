"""Shared plumbing for the GUI's browser tests and benchmarks: start the real
`mace gui` entry point on a copy of examples/gui (with its own
XDG_CONFIG_HOME, so nothing touches ~/.config), and a headless Firefox."""
import contextlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MACE_CLI = REPO_ROOT / "mace_cli"
EXAMPLES = REPO_ROOT / "examples" / "gui"
TOKEN = "gui-test-token-0001"
# the printed URL: http://127.0.0.1:<port>/s/<session path>/?login=<one-time code>; the base keeps the path
# a test that shows a structure: the server reads structures with ase
needs_ase = pytest.mark.skipif(importlib.util.find_spec("ase") is None,
                               reason="ase is not installed (the server reads the structures with it)")
URL_RE = r"(http://127\.0\.0\.1:\d+/s/[A-Za-z0-9_-]+/)\?(?:login|token)="


def env(xdg: Path, **extra):
    # the user's own xtb, ML interpreter and weights never leak into a test's server (a test passes its own)
    e = {k: v for k, v in os.environ.items()
         if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH", "MACE_XTB", "MACE_MLIP_PYTHON",
                      "MACE_MLIP_DIR", "MACE_MLIP_NO_DOWNLOAD", "MACE_GUI_TOKEN", "MACE_GUI_PORT")}
    # the user's cache folder too (provenance sidecars): never the real ~/.cache
    # stdout is a pipe here: MACE_GUI_PRINT_TOKEN keeps the token in the printed URL
    # File History (XDG_STATE_HOME) too: a test's saves never reach ~/.local/state
    e.pop("MACE_GUI_HISTORY_DIR", None)
    e.update(PYTHONDONTWRITEBYTECODE="1", XDG_CONFIG_HOME=str(xdg), XDG_CACHE_HOME=str(Path(xdg) / "cache"),
             XDG_STATE_HOME=str(Path(xdg) / "state"), MACE_GUI_PRINT_TOKEN="1", **extra)
    return e


def die_with_pytest():
    """preexec_fn for a test's server: SIGTERM when the pytest process that
    started it dies (Linux), so a killed test run leaves no server behind."""
    try:
        import ctypes
        import signal as _signal
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, int(_signal.SIGTERM), 0, 0, 0)   # PR_SET_PDEATHSIG
    except Exception:
        pass


class ServerProcess(subprocess.Popen):
    """A test's `mace gui` server. Teardown never hangs a CI runner: after
    terminate(), a wait() that times out kills the server's whole session (it
    and its pool workers) and waits again. Started in its own session."""

    _terminated = False

    def terminate(self):
        self._terminated = True
        super().terminate()

    def wait(self, timeout=None):
        try:
            return super().wait(timeout)
        except subprocess.TimeoutExpired:
            if not self._terminated:
                raise
            try:
                os.killpg(self.pid, 9)
            except (ProcessLookupError, PermissionError):
                self.kill()
            return super().wait(5)


class RemoteServer:
    """MACE_GUI_E2E_BASE: a server someone else started (an HPCC dev node behind
    an ssh tunnel). Nothing to start or stop here."""
    pid = -1

    def terminate(self):
        pass

    def kill(self):
        pass

    def send_signal(self, sig):
        pass

    def wait(self, timeout=None):
        return 0


REMOTE = os.environ.get("MACE_GUI_E2E_BASE", "").strip()


def remote():
    """True when the tests drive a remote server (MACE_GUI_E2E_BASE)."""
    return bool(REMOTE)


def local_only(reason="reads the server's files"):
    """Skip a test that needs the server's disk when the server is remote."""
    if remote():
        pytest.skip(f"MACE_GUI_E2E_BASE: {reason}")


def start_server(root: Path, xdg: Path, port: int = 0, token: str = TOKEN, **extra):
    """Returns (process, base URL, seconds until the URL was printed).
    With MACE_GUI_E2E_BASE=<url> (and MACE_GUI_E2E_TOKEN) the remote server
    at that base is used instead: the folder it serves must hold the same
    files the test copies (the HPCC check copies them there)."""
    if REMOTE:
        global TOKEN
        TOKEN = os.environ.get("MACE_GUI_E2E_TOKEN", TOKEN)
        return RemoteServer(), REMOTE.rstrip("/") + "/", 0.0
    t0 = time.perf_counter()
    proc = ServerProcess(
        [sys.executable, str(MACE_CLI), "--no-banner", "gui", "--dir", str(root),
         "--port", str(port), "--token", token],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env(xdg, **extra), start_new_session=True, preexec_fn=die_with_pytest)
    lines = []
    for _ in range(40):
        line = proc.stdout.readline()
        if not line:
            break
        lines.append(line)
        m = re.search(URL_RE, line)
        if m:
            return proc, m.group(1), time.perf_counter() - t0
    proc.kill()
    raise AssertionError("server did not print its URL:\n" + "".join(lines))


def examples_copy(tmp: Path) -> Path:
    root = tmp / "demo"
    shutil.copytree(EXAMPLES, root)
    return root


MESA_EGL = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"


def software_gl_env():
    """Environment for a browser that must not touch a GPU (spec section 9:
    GPU0 drives the display, GPU1 trains models). Mesa's llvmpipe does the
    WebGL, so 3Dmol still renders, on the CPU."""
    e = dict(os.environ)
    e.update(MOZ_HEADLESS="1", LIBGL_ALWAYS_SOFTWARE="1", GALLIUM_DRIVER="llvmpipe",
             __GLX_VENDOR_LIBRARY_NAME="mesa", CUDA_VISIBLE_DEVICES="")
    if os.path.exists(MESA_EGL):  # glvnd would otherwise pick NVIDIA's EGL first
        e["__EGL_VENDOR_LIBRARY_FILENAMES"] = MESA_EGL
    return e


REQUIRE = "MACE_GUI_REQUIRE_BROWSER"


def _unavailable(reason):
    """A browser this test needs is missing: skip, or fail under
    MACE_GUI_REQUIRE_BROWSER=1 (the CI smoke job, so a broken browser setup
    is never a silent pass)."""
    if os.environ.get(REQUIRE) == "1":
        pytest.fail(f"browser required ({REQUIRE}=1): {reason}", pytrace=False)
    pytest.skip(reason)


def _selenium():
    try:
        from selenium import webdriver
    except ImportError:
        _unavailable("selenium is not installed (pip install -r requirements-gui-test.txt)")
    return webdriver


SOFTWARE_GL = ("swiftshader", "llvmpipe", "softpipe")


def check_software_gl(drv):
    """The page's WebGL renderer must be a software one (never a GPU). Under
    MACE_GUI_REQUIRE_BROWSER=1 anything else, or no WebGL, fails; otherwise
    it only warns."""
    name = webgl_renderer(drv)
    ok = bool(name) and any(k in name.lower() for k in SOFTWARE_GL)
    if not ok:
        msg = f"the browser's WebGL renderer is {name!r}, not software GL (SwiftShader or llvmpipe)"
        if os.environ.get(REQUIRE) == "1":
            drv.quit()
            pytest.fail(msg, pytrace=False)
        import warnings
        warnings.warn(msg)
    return name


def firefox(width=1280, height=800, download_dir=None, env=None, prefs=None):
    """A headless Firefox that renders WebGL in software (llvmpipe);
    downloads land in ``download_dir`` without asking, when given. ``env``
    replaces the software-GL environment (the opt-in GPU1 bench: headful,
    PRIME offload)."""
    webdriver = _selenium()
    from selenium.webdriver.firefox.service import Service
    if shutil.which("firefox") is None and not os.environ.get("MACE_GUI_FIREFOX"):
        _unavailable("Firefox is not installed")
    opts = webdriver.FirefoxOptions()
    if env is None:
        opts.add_argument("-headless")
    opts.add_argument(f"--width={width}")
    opts.add_argument(f"--height={height}")
    opts.set_preference("gfx.webrender.software", True)
    opts.set_preference("layers.acceleration.disabled", True)
    for k, v in (prefs or {}).items():       # e.g. ui.prefersReducedMotion
        opts.set_preference(k, v)
    if download_dir is not None:
        opts.set_preference("browser.download.folderList", 2)
        opts.set_preference("browser.download.dir", str(download_dir))
        opts.set_preference("browser.download.useDownloadDir", True)
        opts.set_preference("browser.download.always_ask_before_handling_new_types", False)
        opts.set_preference("browser.helperApps.neverAsk.saveToDisk",
                            "image/png,application/octet-stream,chemical/x-cif,chemical/x-xyz,text/plain")
        opts.set_preference("browser.download.alwaysOpenPanel", False)
    if os.environ.get("MACE_GUI_FIREFOX"):
        opts.binary_location = os.environ["MACE_GUI_FIREFOX"]
    if os.environ.get(REQUIRE) == "1":
        # a CI runner without a GPU: let Firefox use WebGL on its software rasterizer
        opts.set_preference("webgl.force-enabled", True)
    try:
        drv = webdriver.Firefox(options=opts, service=Service(env=software_gl_env() if env is None else env))
    except Exception as exc:  # no geckodriver and no network for Selenium Manager
        _unavailable(f"cannot start headless Firefox: {exc}")
    set_viewport(drv, width, height)
    if env is None and not (prefs or {}).get("webgl.disabled"):
        drv.get("about:blank")
        check_software_gl(drv)
    return drv


CHROME_NAMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "brave-browser", "brave")


def _major(text):
    import re as _re
    m = _re.search(r"(\d+)\.\d+", text or "")
    return int(m.group(1)) if m else None


def chromium(width=1280, height=800, download_dir=None, args=()):
    """A headless Chromium-family browser with software WebGL (SwiftShader,
    never a GPU): Chrome, Chromium or Brave from PATH or MACE_GUI_CHROME; the
    driver from MACE_GUI_CHROMEDRIVER, else chromedriver on PATH, else
    Selenium Manager (which also fetches Chrome for Testing when no browser
    is installed). Brave needs a chromedriver of its Chromium major version:
    a mismatch is named."""
    webdriver = _selenium()
    from selenium.webdriver.chrome.service import Service
    binary = os.environ.get("MACE_GUI_CHROME") or next((shutil.which(n) for n in CHROME_NAMES if shutil.which(n)), None)
    opts = webdriver.ChromeOptions()
    for a in ("--headless=new", "--disable-gpu", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
              f"--window-size={width},{height}", "--no-first-run", "--no-default-browser-check", *args):
        opts.add_argument(a)
    if binary:
        opts.binary_location = binary
    if download_dir is not None:
        opts.add_experimental_option("prefs", {"download.default_directory": str(download_dir),
                                               "download.prompt_for_download": False})
    driver = os.environ.get("MACE_GUI_CHROMEDRIVER") or shutil.which("chromedriver")
    if driver and binary:
        try:
            bv = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30).stdout
            dv = subprocess.run([driver, "--version"], capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            _unavailable(f"cannot run {binary} or {driver}: {exc}")
        if _major(bv) is not None and _major(dv) is not None and _major(bv) != _major(dv):
            _unavailable(f"chromedriver {dv.strip()!r} does not match the browser {bv.strip()!r} (major versions differ); "
                         "set MACE_GUI_CHROMEDRIVER to a chromedriver of the browser's major version")
    try:
        service = Service(executable_path=driver, env=software_gl_env()) if driver else Service(env=software_gl_env())
        drv = webdriver.Chrome(options=opts, service=service)
    except Exception as exc:
        _unavailable(f"cannot start headless Chromium: {exc}")
    set_viewport(drv, width, height)
    if "--disable-webgl" not in args:
        drv.get("about:blank")
        check_software_gl(drv)
    return drv


def browser(name=None, **kw):
    """firefox() or chromium(): `name`, else MACE_GUI_BROWSER, else Firefox."""
    name = name or os.environ.get("MACE_GUI_BROWSER", "firefox")
    return chromium(**kw) if name.startswith("chrom") else firefox(**kw)


def save_failure(drv, name, server_log=None):
    """On a failure: a screenshot, the page's errors and the server log into
    $MACE_GUI_ARTIFACTS (the CI job uploads it); file names without paths."""
    out = os.environ.get("MACE_GUI_ARTIFACTS")
    if not out:
        return
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:120]
    try:
        drv.save_screenshot(str(d / f"{safe}.png"))
        (d / f"{safe}.errors.json").write_text(json_dumps(drv.execute_script("return window.__maceErrors || []")))
    except Exception:
        pass
    if server_log:
        try:
            (d / f"{safe}.server.log").write_text(server_log)
        except OSError:
            pass


def json_dumps(x):
    import json as _json
    return _json.dumps(x, indent=1, default=str)


def webgl_renderer(drv):
    return drv.execute_script(
        "const c = document.createElement('canvas').getContext('webgl2')"
        " || document.createElement('canvas').getContext('webgl'); if (!c) return '';"
        "const e = c.getExtension('WEBGL_debug_renderer_info');"
        "return String(e ? c.getParameter(e.UNMASKED_RENDERER_WEBGL) : c.getParameter(c.RENDERER))")


def set_viewport(drv, width, height):
    """Size the window so the page's viewport is width x height."""
    drv.set_window_size(width, height)
    w, h = drv.execute_script("return [innerWidth, innerHeight]") if drv.current_url.startswith("http") else (width, height)
    if (w, h) != (width, height):
        drv.set_window_size(width + (width - w), height + (height - h))


def wait(drv, script, timeout=30, *args):
    end = time.time() + timeout
    while time.time() < end:
        r = drv.execute_script(script, *args)
        if r:
            return r
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for: {script}")


def free_port():
    import socket
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


def login_url(base, token=None):
    """A one-time sign-in link, minted with the token in the X-Mace-Token header
    (the token never goes into a URL)."""
    token = token or TOKEN
    import http.client
    import json as _json
    from urllib.parse import urlsplit
    u = urlsplit(base)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
    c.request("POST", u.path + "api/login-code", body="{}", headers={
        "X-Mace-Token": token, "X-Requested-With": "mace-gui", "Content-Type": "application/json"})
    r = c.getresponse()
    code = _json.loads(r.read())["code"]
    c.close()
    return f"{base}?login={code}"


def load(drv, base, token=None):
    drv.get(login_url(base, token or TOKEN))
    wait(drv, "return document.documentElement.dataset.appReady === '1'")


def errors(drv):
    return drv.execute_script("return window.__maceErrors || []")


# ---------------------------------------------------------------- resources
# Spec section 9: a benchmark reports the CPU workers it used, peak RSS and
# per-GPU memory, and must not touch GPU0. All read from /proc and nvidia-smi.

def descendants(pid):
    """Every live process under pid (children, grandchildren, ...)."""
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        try:
            tasks = os.listdir(f"/proc/{p}/task")
        except OSError:
            continue
        for t in tasks:
            try:
                with open(f"/proc/{p}/task/{t}/children") as fh:
                    kids = [int(x) for x in fh.read().split()]
            except OSError:
                kids = []
            out += kids
            todo += kids
    return out


def peak_rss_mb(pid):
    """VmHWM (peak resident set) of one process, in MB; None when it is gone."""
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        return None
    return None


def pss_mb(pid):
    """Proportional set size in MB: shared pages split between the processes
    sharing them, so a browser's processes add up without double counting."""
    try:
        with open(f"/proc/{pid}/smaps_rollup") as fh:
            for line in fh:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        return None
    return None


def gpu_snapshot():
    """{gpu index: memory used MiB} plus [(gpu, pid, MiB)] for every process
    on a GPU; ({}, []) without nvidia-smi."""
    import subprocess as sp
    if shutil.which("nvidia-smi") is None:
        return {}, []
    try:
        mem = sp.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                     capture_output=True, text=True, timeout=10).stdout
        table = sp.run(["nvidia-smi"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, sp.SubprocessError):
        return {}, []
    used = {int(a): int(b) for a, b in (ln.split(",") for ln in mem.strip().splitlines() if "," in ln)}
    procs = []
    for ln in table.splitlines():
        m = re.match(r"\|\s+(\d+)\s+\S+\s+\S+\s+(\d+)\s+\S+\s+.*?(\d+)MiB\s*\|", ln)
        if m:
            procs.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return used, procs


class ResourceSampler:
    """Samples, in a thread, the processes under the given root pids (server,
    geckodriver and the browser it starts): worker count, peak RSS, and which
    GPUs they appear on; plus the per-GPU memory delta over the run."""

    def __init__(self, interval=0.5):
        import threading
        self.interval, self.roots, self.stop_ev = interval, {}, threading.Event()
        self.max_children, self.rss, self.gpu_pids, self.gpu_max, self.gpu_max_raw = {}, {}, set(), {}, {}
        # a GPU's memory without what other users' listed processes hold (a remote-desktop
        # client or a training job growing during the run is not ours)
        used, procs = gpu_snapshot()
        self.gpu_before_raw = dict(used)
        self.gpu_before = {g: mib - sum(m for gg, _, m in procs if gg == g) for g, mib in used.items()}
        self.thread = threading.Thread(target=self._run, daemon=True)

    def watch(self, name, pid):
        self.roots[name] = pid
        return self

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        while not self.stop_ev.wait(self.interval):
            self.sample()

    def sample(self):
        mine = set()
        for name, pid in list(self.roots.items()):
            kids = descendants(pid)
            self.max_children[name] = max(self.max_children.get(name, 0), len(kids))
            # one process: its own peak; a tree (the browser): the summed PSS now
            total = peak_rss_mb(pid) or 0 if not kids else sum(pss_mb(p) or 0 for p in [pid] + kids)
            self.rss[name] = max(self.rss.get(name, 0), total)
            mine |= {pid, *kids}
        used, procs = gpu_snapshot()
        for g, mib in used.items():
            foreign = sum(m for gg, p, m in procs if gg == g and p not in mine)
            self.gpu_max[g] = max(self.gpu_max.get(g, -10**9), mib - foreign)
            self.gpu_max_raw[g] = max(self.gpu_max_raw.get(g, 0), mib)
        self.gpu_pids |= {(g, p, mib) for g, p, mib in procs if p in mine}

    def stop(self):
        self.stop_ev.set()
        self.thread.join(timeout=5)
        self.sample()
        return {"max_children": self.max_children, "peak_rss_mb": {k: round(v, 1) for k, v in self.rss.items()},
                "gpu_delta_mib": {g: self.gpu_max.get(g, b) - b for g, b in self.gpu_before.items()},
                "gpu_delta_raw_mib": {g: self.gpu_max_raw.get(g, b) - b for g, b in self.gpu_before_raw.items()},
                "gpu_processes_of_ours": sorted(self.gpu_pids)}


# ---------------------------------------------------------------- pointer input (Phase 3)
# Real WebDriver pointer actions (buttons 0 left, 1 middle, 2 right) on the
# structure viewer; offsets are CSS px from the canvas's top left corner.

def viewer_canvas(drv):
    return drv.find_element("css selector", "#viewer-main canvas")


def _offset(drv, el, x, y):
    w, h = el.size["width"], el.size["height"]
    return int(round(x - w / 2)), int(round(y - h / 2))


class _Ticks:
    """Pointer and key actions in step: every pointer action gets a key
    pause in the same tick (and the other way round), so a held modifier is
    still down when the button goes down and up."""

    def __init__(self, drv):
        from selenium.webdriver.common.actions.action_builder import ActionBuilder
        from selenium.webdriver.common.actions.pointer_input import PointerInput
        from selenium.webdriver.common.actions import interaction
        self.ab = ActionBuilder(drv, mouse=PointerInput(interaction.POINTER_MOUSE, "mouse"))
        self.p, self.k = self.ab.pointer_action, self.ab.key_action

    def key_down(self, key):
        self.k.key_down(key); self.p.pause(0)

    def key_up(self, key):
        self.k.key_up(key); self.p.pause(0)

    def move(self, el, x, y, ms=None):
        if ms is None:
            self.p.move_to(el, x, y)
        else:
            self.p.source.create_pointer_move(duration=ms, x=x, y=y, origin=el)
        self.k.pause(0)

    def down(self, button):
        self.p.pointer_down(button); self.k.pause(0)

    def up(self, button):
        self.p.pointer_up(button); self.k.pause(0)

    def perform(self):
        self.ab.perform()


def drag(drv, el, path, button=0, keys=(), steps=6):
    """Press `button` at path[0], move through path (each leg in `steps`), release.
    `keys` are held (Keys.SHIFT, Keys.CONTROL, Keys.ALT) for the whole drag."""
    t = _Ticks(drv)
    x0, y0 = _offset(drv, el, *path[0])
    t.move(el, x0, y0)
    for k in keys:
        t.key_down(k)
    t.down(button)
    last = path[0]
    for p in path[1:]:
        for s in range(1, steps + 1):
            x = last[0] + (p[0] - last[0]) * s / steps
            y = last[1] + (p[1] - last[1]) * s / steps
            t.move(el, *_offset(drv, el, x, y), ms=16)
        last = p
    t.up(button)
    for k in reversed(keys):
        t.key_up(k)
    t.perform()


def click_at(drv, x, y, button=0, keys=(), count=1):
    el = viewer_canvas(drv)
    t = _Ticks(drv)
    t.move(el, *_offset(drv, el, x, y))
    for k in keys:
        t.key_down(k)
    for _ in range(count):
        t.down(button)
        t.up(button)
    for k in reversed(keys):
        t.key_up(k)
    t.perform()


def atom_screen(drv, i):
    return drv.execute_script("return window.__mace.viewport.atomScreen(arguments[0])", i)


def click_atom(drv, i, mods=(), count=1, button=0):
    p = atom_screen(drv, i)
    click_at(drv, p["x"], p["y"], button=button, keys=mods, count=count)


def wheel(drv, x, y, dy):
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.common.actions.wheel_input import ScrollOrigin
    el = viewer_canvas(drv)
    ActionChains(drv).scroll_from_origin(ScrollOrigin.from_element(el, *_offset(drv, el, x, y)), 0, int(dy)).perform()


def hover_at(drv, x, y):
    from selenium.webdriver.common.action_chains import ActionChains
    el = viewer_canvas(drv)
    ActionChains(drv).move_to_element_with_offset(el, *_offset(drv, el, x, y)).perform()


def requests_since(drv, t, pattern=""):
    """Resource Timing entries (name, start, end) started after t (ms) whose URL holds `pattern`."""
    return drv.execute_script("""return performance.getEntriesByType('resource')
        .filter((r) => r.startTime >= arguments[0] && r.name.includes(arguments[1]))
        .map((r) => ({name: r.name, start: r.startTime, end: r.responseEnd}))""", t, pattern)


def now_ms(drv):
    return drv.execute_script("return performance.now()")




# ---------------------------------------------------------------- Phase 4 helpers
class SSE:
    """A raw /api/events reader in a thread: every event as (event, data, t).
    sse_events(base) starts one; .wait(pred) waits for an event matching it."""

    def __init__(self, base, token=TOKEN, interval=0.5):
        import http.client
        import json as _json
        import threading
        from urllib.parse import urlsplit
        u = urlsplit(base)
        self.conn = http.client.HTTPConnection(u.hostname, u.port, timeout=60)
        self.conn.request("GET", f"{u.path}api/events?interval={interval}", headers={"X-Mace-Token": token})
        self.resp = self.conn.getresponse()
        # http.client hands the socket to the response: keep it, to close the stream for real
        self.sock = getattr(getattr(self.resp.fp, "raw", None), "_sock", None)
        self.events, self.closed, self._json = [], False, _json
        self.cond = threading.Condition()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        event, data = "message", []
        try:
            while not self.closed:
                line = self.resp.fp.readline()
                if not line:
                    break
                line = line.decode().rstrip("\n")
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].strip())
                elif line == "":
                    if data:
                        try:
                            d = self._json.loads("\n".join(data))
                        except ValueError:
                            d = "\n".join(data)
                        with self.cond:
                            self.events.append((event, d, time.time()))
                            self.cond.notify_all()
                    event, data = "message", []
        except Exception:
            pass
        with self.cond:
            self.closed = True
            self.cond.notify_all()

    def of(self, kind, pred=None):
        with self.cond:
            return [e for e in self.events if e[0] == kind and (pred is None or pred(e[1]))]

    def wait(self, pred, timeout=20):
        end = time.time() + timeout
        with self.cond:
            while True:
                hit = [e for e in self.events if pred(e)]
                if hit:
                    return hit[0]
                left = end - time.time()
                if left <= 0 or (self.closed and not hit):
                    raise AssertionError(f"no matching event in {timeout} s: {self.events[-5:]}")
                self.cond.wait(min(left, 0.2))

    def close(self):
        """End the stream as a closed tab would: the server sees the socket go."""
        import socket as _socket
        self.closed = True
        for sk in (self.sock, self.conn.sock):
            if sk is None:
                continue
            try:
                sk.shutdown(_socket.SHUT_RDWR)
            except OSError:
                pass
        try:
            self.resp.close()
        except Exception:
            pass
        self.conn.close()


def sse_events(base, token=TOKEN, interval=0.5):
    return SSE(base, token, interval)


def dir_md5(path):
    """md5 over every file (relative path and bytes) under path."""
    import hashlib
    h = hashlib.md5()
    for p in sorted(Path(path).rglob("*")):
        if p.is_file():
            h.update(p.relative_to(path).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def mem_ok(gb):
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024) >= gb
    except OSError:
        return False
    return False


def tmp_free_ok(gb, path="/tmp"):
    return shutil.disk_usage(path).free / 1024 ** 3 >= gb


# ---------------------------------------------------------------- a visible Firefox under Xvfb (plan T16)
XVFB_CANDIDATES = [Path(os.environ.get("MACE_GUI_XVFB", "")) if os.environ.get("MACE_GUI_XVFB") else None,
                   Path("/usr/bin/Xvfb")]


def xvfb_binary():
    for p in XVFB_CANDIDATES:
        if p and p.is_file() and os.access(p, os.X_OK):
            return p
    found = shutil.which("Xvfb")
    return Path(found) if found else None


@contextlib.contextmanager
def xvfb(display=":99", size="1920x1080x24"):
    """An Xvfb server (no root, -nolisten tcp) for a non-headless Firefox
    that renders in software; yields the display name."""
    exe = xvfb_binary()
    if exe is None or shutil.which("xdotool") is None:
        pytest.skip("needs Xvfb and xdotool: sudo pacman -S xorg-server-xvfb xdotool (or MACE_GUI_XVFB=<path to Xvfb>)")
    proc = subprocess.Popen([str(exe), display, "-screen", "0", size, "-nolisten", "tcp"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            if subprocess.run(["xdotool", "getdisplaygeometry"], env={**os.environ, "DISPLAY": display},
                              capture_output=True).returncode == 0:
                break
            time.sleep(0.1)
        else:
            pytest.skip("Xvfb did not start")
        yield display
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def visible_firefox(display, width=1600, height=1000):
    """Firefox, not headless, on an Xvfb display, drawing WebGL with llvmpipe."""
    webdriver = pytest.importorskip("selenium.webdriver")
    from selenium.webdriver.firefox.service import Service
    env = software_gl_env()
    env.pop("MOZ_HEADLESS", None)
    env["DISPLAY"] = display
    opts = webdriver.FirefoxOptions()
    opts.add_argument(f"--width={width}")
    opts.add_argument(f"--height={height}")
    opts.set_preference("gfx.webrender.software", True)
    opts.set_preference("layers.acceleration.disabled", True)
    drv = webdriver.Firefox(options=opts, service=Service(env=env))
    return drv


def xkey(display, pid, combo):
    """A real key press through XTEST into the Firefox window of `pid`."""
    env = {**os.environ, "DISPLAY": display}
    wins = subprocess.run(["xdotool", "search", "--pid", str(pid), "--name", "Mozilla Firefox"],
                          env=env, capture_output=True, text=True).stdout.split()
    if wins:          # the browser window (not its hidden helper windows); no window manager, so focus it
        subprocess.run(["xdotool", "windowfocus", "--sync", wins[-1]], env=env, capture_output=True, timeout=5)
    subprocess.run(["xdotool", "key", combo], env=env, capture_output=True, timeout=5)
    # never leave a modifier down (--clearmodifiers would press back any it saw held)
    subprocess.run(["xdotool", "keyup", "alt", "ctrl", "shift", "super"], env=env, capture_output=True, timeout=5)


def xwindows(display, pid):
    env = {**os.environ, "DISPLAY": display}
    return len(subprocess.run(["xdotool", "search", "--pid", str(pid), "--onlyvisible", "--name", "."],
                              env=env, capture_output=True, text=True).stdout.split())
