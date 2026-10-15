"""Helpers shared by the Phase 4 browser and API tests (builders, Clean,
files, tasks): one served demo copy per module, a headless Firefox in
software GL, the page's state, and the folder checks (md5 of --dir, the
server's cache dir). Every fixture is generated into a temp dir or copied
from examples/gui; nothing here reads research data."""
import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

import guiharness as H

RESET_JS = ("window.__mace.closeAll(); window.__mace.settings.resetAll(); window.__mace.settings.set('theme', 'light');"
            "const set = (id, v) => { const e = document.getElementById(id); if (e && e.checked !== v) { e.checked = v; e.dispatchEvent(new Event('change')); } };"
            "set('opt-bonds', true); set('opt-super', false);"
            "window.__mace.run('tool.select');"
            "document.querySelectorAll('dialog[open]').forEach((d) => d.close());"
            "document.querySelectorAll('.builder-dialog').forEach((d) => d.remove && d.close && d.close());"
            "document.querySelectorAll('#toasts > *').forEach((t) => t.remove());"
            "window.__maceErrors.length = 0;"
            "performance.clearResourceTimings(); performance.setResourceTimingBufferSize(100000);")


def start_app(tmp, width=1280, height=800, root=None, **env):
    """(drv, base, root, xdg, proc, downloads): the real `mace gui` on a demo copy, and a headless Firefox."""
    root = root or H.examples_copy(tmp)
    xdg = tmp / "xdg"
    extra = {"MACE_GUI_TEST_HOOKS": "1", **env}
    proc, base, _ = H.start_server(root, xdg, **extra)
    downloads = tmp / "downloads"
    downloads.mkdir(exist_ok=True)
    drv = None
    try:
        # positional size: some tests swap H.firefox for a (w, h) lambda
        pick = H.chromium if os.environ.get("MACE_GUI_BROWSER", "firefox").startswith("chrom") else H.firefox
        drv = pick(width, height, download_dir=downloads)
        H.load(drv, base)
    except BaseException:
        # a page that never got ready: leave no browser and no server behind
        if drv is not None:
            drv.quit()
        proc.terminate()
        proc.wait(timeout=15)
        raise
    return drv, base, root, xdg, proc, downloads


def stop_app(app):
    drv, proc = app[0], app[4]
    try:
        drv.quit()
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def reset(drv, width=1280, height=800):
    drv.execute_script(RESET_JS)
    H.set_viewport(drv, width, height)
    time.sleep(0.3)
    drv.execute_script("document.body.focus()")


def keys(drv, *combo):
    from selenium.webdriver.common.action_chains import ActionChains
    ac = ActionChains(drv)
    for k in combo[:-1]:
        ac.key_down(k)
    ac.send_keys(combo[-1])
    for k in reversed(combo[:-1]):
        ac.key_up(k)
    ac.perform()
    time.sleep(0.15)


def open_doc(drv, path, wait_atoms=True):
    kind = drv.execute_async_script(
        "const done = arguments[1]; window.__mace.open(arguments[0]).then((d) => done(d && d.kind))", path)
    if wait_atoms and kind == "structure":
        H.wait(drv, "return window.__mace.viewport && window.__mace.viewport.displayed() > 0")
        time.sleep(0.3)
    return kind


def shot(drv, name):
    target = os.environ.get("MACE_GUI_SHOTS")
    if target:
        Path(target).mkdir(parents=True, exist_ok=True)
        drv.save_screenshot(str(Path(target) / f"{name}.png"))


def no_errors(drv):
    errs = H.errors(drv)
    assert not errs, errs


def toasts(drv):
    return drv.execute_script("return [...document.querySelectorAll('#toasts > *')].map((t) => t.textContent)")


def doc(drv):
    return drv.execute_script("const d = window.__mace.doc(); return d && {title: d.title, kind: d.kind, path: d.path,"
                              " dirty: d.dirty, untitled: !!d.untitled, n: d.structure ? d.structure.numbers.length : 0,"
                              " sg: d.structure && d.structure.symmetry && d.structure.symmetry.available ? d.structure.symmetry.number : null}")


def ed_state(drv):
    return drv.execute_script("""const e = window.__mace.editor && window.__mace.editor.doc(); if (!e) return null;
        return {symbols: e.structure.symbols, frac: e.structure.frac, n: e.n, pos: e.pos, rows: e.rows.map((r) => r.label),
                dirty: e.dirty, pending: e.pending, waiting: e.waiting, session: e.srv.session, rev: e.rev}""")


def wait_idle(drv, timeout=20):
    H.wait(drv, "const e = window.__mace.editor && window.__mace.editor.doc(); return e && e.pending === 0 && !e.waiting", timeout)
    time.sleep(0.1)


def tree_md5(root: Path, skip=()):
    """md5 over every file's relative path and bytes (names in `skip` left out)."""
    h = hashlib.md5()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.name not in skip:
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def req(base, path, data=None, headers=None, token=True, method=None, raw=None, timeout=120):
    h = {"X-Requested-With": "mace-gui"}
    if token:
        h["X-Mace-Token"] = H.TOKEN
    h.update(headers or {})
    body = raw if raw is not None else (json.dumps(data).encode() if data is not None else None)
    if body is not None and raw is None:
        h["Content-Type"] = "application/json"
    r = urllib.request.Request(base + path, data=body, headers=h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            out = resp.read()
            return resp.status, dict(resp.headers), _decode(out, resp.headers)
    except urllib.error.HTTPError as e:
        out = e.read()
        return e.code, dict(e.headers), _decode(out, e.headers)


def _decode(raw, headers):
    if headers.get("Content-Encoding") == "gzip":
        import gzip
        raw = gzip.decompress(raw)
    ctype = headers.get("Content-Type") or ""
    if "json" in ctype:
        try:
            return json.loads(raw or b"null")
        except ValueError:
            return raw
    return raw


def post(base, path, data, **kw):
    return req(base, path, data=data, **kw)


def mem_ok(gb):
    """MemAvailable at least gb GiB (spec section 9.2)."""
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
