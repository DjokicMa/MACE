"""The served folder's energy window (`[plots] energy_window` in
mace-folder.toml) in what the GUI runs `mace plotting` for (1.1.6), in
Firefox and Chromium:

* the band plot's Publication Figure runs `mace plotting --band` with the
  window on screen (the folder's), and its shown command names it; a script's
  GET of the plot route without a window takes the folder's;
* the Plot Gallery opens on the Settings window (here the folder's), its CLI
  line always names the window, and Export draws what that line draws when
  it is run by hand in the folder.
The plots run in the server's cache folder, where the CLI cannot find the
served folder's file: the window must be passed. Corpus BAND files (test/,
read only, copied; skipped without it)."""
import io
import shlex
import shutil
import subprocess
import sys
import time
import urllib.parse

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")

BROWSERS = ["firefox", "chromium"]
BAND = H.REPO_ROOT / "test" / "BAND"
FOLDER_FILE = "[plots]\nenergy_window = [-1.0, 1.0]\n"


def _band_files():
    return sorted(BAND.glob("*.BAND.DAT"))[:2] if BAND.is_dir() else []


@pytest.fixture(scope="module", params=BROWSERS)
def app(request, tmp_path_factory):
    files = _band_files()
    if not files:
        pytest.skip("test/ data corpus not present (gitignored)")
    tmp = tmp_path_factory.mktemp(f"folder_window_{request.param}")
    root = tmp / "root"
    (root / "bands").mkdir(parents=True)
    for b in files:
        shutil.copyfile(b, root / "bands" / b.name)
        d3 = b.with_name(b.name[: -len(".BAND.DAT")] + ".d3")
        if d3.is_file():
            shutil.copyfile(d3, root / "bands" / d3.name)
    (root / "mace-folder.toml").write_text(FOLDER_FILE)
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1")
    downloads = tmp / "downloads"
    downloads.mkdir()
    drv = None
    try:
        drv = H.browser(request.param, width=1280, height=900, download_dir=downloads)
        H.load(drv, base)
        yield drv, base, root, downloads, tmp
    finally:
        if drv is not None:
            drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def _cli(argv, cwd):
    return subprocess.run([sys.executable, str(H.REPO_ROOT / "mace_cli"), "--no-banner", *argv], cwd=str(cwd),
                          capture_output=True, text=True, timeout=600)


def _png_pixels(data):
    from PIL import Image
    im = Image.open(io.BytesIO(data)).convert("RGB")
    return im.size, im.tobytes()


def _wait_task(base, tid, timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        s, _, res = U.req(base, f"api/tasks/{tid}/result")
        if s == 200 and isinstance(res, dict) and res.get("key"):
            return res
        time.sleep(0.3)
    raise AssertionError("the task did not finish")


def test_band_publication_runs_with_the_folder_window(app):
    drv, base, root, downloads, tmp = app
    rel = "bands/" + _band_files()[0].name
    drv.execute_script("window.__mace.closeAll()")
    assert U.open_doc(drv, rel) == "plot"
    H.wait(drv, "const s = window.__mace.plotView && window.__mace.plotView.state(); return s && s.drawn", 60)
    opts = drv.execute_script("return window.__mace.plotView.state().opts")
    assert (opts["emin"], opts["emax"]) == (-1, 1)                          # on screen: the folder's window
    r = drv.execute_async_script("const done = arguments[0]; window.__mace.plotView.publication().then(done, (e) => done({error: String(e)}))")
    assert "error" not in r, r
    assert "--e-lower=-1 --e-upper=1" in r["command"], r["command"]
    # a script's GET without a window: the folder's, and the image is what the shown command draws in the folder
    q = urllib.parse.urlencode({"path": rel, "format": "png"})
    st, _, res = U.req(base, f"api/plot?{q}")
    if st == 202:
        res = _wait_task(base, res["task"]["id"])
    else:
        assert st == 200, res
    assert res["command"].endswith("--format png --e-lower=-1 --e-upper=1"), res["command"]
    png = next(n for n in res["images"] if n.endswith(".png"))
    st, _, served = U.req(base, "api/plot-image?" + urllib.parse.urlencode({"key": res["key"], "name": png}))
    assert st == 200
    by_hand = tmp / "by_hand"
    shutil.copytree(root, by_hand, dirs_exist_ok=True)
    argv = shlex.split(res["command"])[1:]
    p = _cli(argv, by_hand / "bands")
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    mine = sorted((by_hand / "bands").glob("*.png"))
    assert mine and _png_pixels(mine[0].read_bytes()) == _png_pixels(served)
    # an explicit window wins (the page sends the one on screen)
    q = urllib.parse.urlencode({"path": rel, "format": "svg", "emin": "-3", "emax": "2"})
    st, _, res = U.req(base, f"api/plot?{q}")
    res = _wait_task(base, res["task"]["id"]) if st == 202 else res
    assert res["command"].endswith("--e-lower=-3 --e-upper=2")
    st, _, err = U.req(base, "api/plot?" + urllib.parse.urlencode({"path": rel, "emin": "2", "emax": "1"}))
    assert st == 400
    U.no_errors(drv)


def test_gallery_opens_on_the_settings_window_and_exports_what_its_cli_line_draws(app):
    drv, base, root, downloads, tmp = app
    drv.execute_script("window.__mace.closeAll()")
    assert drv.execute_script("return window.__mace.settings.get('plots.energyWindow')") == [-1, 1]
    drv.execute_script("window.__mace.openGallery({source: {type: 'folder', folder: 'bands'}, kind: 'band', label: 'bands'})")
    H.wait(drv, "const v = window.__mace.galleryView; const s = v && v.state(); return s && s.items > 0 && s.drawn > 0", 60)
    # panes draw lazily and in parallel: pane 1 can be drawn before pane 0
    fig = H.wait(drv, "const f = window.__mace.galleryView.figure(0); return f && f.layout.y ? f : null", 60)
    lo, hi = fig["layout"]["y"]
    assert hi - lo == pytest.approx(2.0)                  # -1 to 1 eV around E_F
    assert (drv.find_element("css selector", "#gallery-host .gal-emin").get_attribute("value"),
            drv.find_element("css selector", "#gallery-host .gal-emax").get_attribute("value")) == ("-1", "1")
    cmd = drv.execute_script("return window.__mace.galleryView.command()")
    assert "--e-lower=-1 --e-upper=1" in cmd, cmd
    for f in downloads.iterdir():
        f.unlink()
    drv.find_element("css selector", '#gallery-host [data-act="export-png"]').click()
    end = time.time() + 300
    target = downloads / "gallery_band.png"
    while time.time() < end and not (target.is_file() and target.stat().st_size > 0):
        time.sleep(0.3)
    assert target.is_file()
    time.sleep(0.5)
    # the shown line, run by hand in the served folder, draws the same figure
    argv = shlex.split(cmd)[1:]
    argv[argv.index("-o") + 1] = str(tmp / "by_hand_gallery.png")
    p = _cli(argv, root)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert _png_pixels((tmp / "by_hand_gallery.png").read_bytes()) == _png_pixels(target.read_bytes())
    U.no_errors(drv)
