"""Inspector › Brillouin zone of a band plot (1.1.6), in every tier: the zone of
the run's cell (a polymer's 1D, a slab's 2D, a crystal's 3D zone) with the path
the bands were computed along and its points labeled, and the path's facts
(segments with their k, the ISS, the k points, the file they come from). A
phonon fort.25 reads its path from its own segment records and its ISS from
the FREQCALC BANDS of the .d12 beside it. Firefox, the bundled example suite
and tests/data/f25."""
import lzma
import os
import shutil
import time
from pathlib import Path

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")

SUITE = H.REPO_ROOT / "examples" / "suite"
F25 = Path(__file__).resolve().parents[1] / "data" / "f25"
RUNS = {
    "poly": (SUITE / "4-polymer-1d" / "polyacetylene" / "BAND", "polyacetylene_band", "BAND.DAT"),
    "gr": (SUITE / "3-slab-2d" / "graphene" / "BAND", "graphene_band", "f25"),
    "dia": (SUITE / "2-dense-3d" / "diamond" / "BAND", "diamond_band", "BAND.DAT"),
}


def put(src, dst):
    xz = src.with_name(src.name + ".xz")
    if xz.is_file():
        dst.write_bytes(lzma.decompress(xz.read_bytes()))
    elif src.is_file():
        dst.write_bytes(src.read_bytes())
    else:
        pytest.skip("the bundled example suite is not here")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bandbz")
    root = H.examples_copy(tmp)
    for key, (folder, stem, ext) in RUNS.items():
        (root / key).mkdir()
        for e in (ext, "d3", "out"):
            put(folder / f"{stem}.{e}", root / key / f"{stem}.{e}")
    (root / "ph").mkdir()
    for name in ("dia_phonon_f25.fort.25", "dia_phonon_f25.d12"):
        shutil.copy2(F25 / name, root / "ph" / name)
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1")
    drv = None
    try:
        drv = H.browser("firefox", width=1400, height=900)
        drv.set_script_timeout(120)
        H.load(drv, base)
        drv.execute_script("window.__mace.settings.set('reduceMotion', 'on')")
        yield drv
    finally:
        if drv is not None:
            drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def open_plot(drv, path):
    drv.execute_script("window.__mace.closeAll()")
    drv.execute_async_script("const done = arguments[1]; window.__mace.open(arguments[0]).then(() => done())", path)
    H.wait(drv, "const v = window.__mace.plotView, s = v && v.state(); return s && s.drawn && v.el().dataset.drawn === arguments[0]", 60, path)
    H.wait(drv, "const f = document.querySelector('#inspector .po-kinfo'); return !!f && /From/.test(f.textContent) && !/reading/.test(f.textContent)", 30)


def section(drv):
    return drv.execute_script("""const s = document.querySelector('#inspector details.po-cat[data-cat="bz"]');
      if (!s) return null;
      const facts = {};
      s.querySelectorAll('.po-kfacts > dt').forEach((t) => { facts[t.textContent] = t.nextElementSibling.textContent; });
      return {tier: s.dataset.tier || null, open: s.open, shown: s.offsetParent !== null, facts,
        segs: [...s.querySelectorAll('.po-ksegs li')].map((li) => li.textContent)}""")


def zone(drv, timeout=60):
    H.wait(drv, "const e = document.querySelector('#inspector .po-bz .kp-bz'); const s = e && e.__bz.state(); return s && (s.loaded || s.failed)", timeout)
    return drv.execute_script("return document.querySelector('#inspector .po-bz .kp-bz').__bz.state()")


@pytest.mark.parametrize("key, dim, points, iss, segs", [
    ("poly", 1, ["GAMMA", "X"], "2", ["Γ (0 0 0) → X (0.5 0 0)"]),
    ("gr", 2, ["M", "GAMMA", "K"], "12", ["M (0.5 0 0) → Γ (0 0 0)", "Γ (0 0 0) → K (0.3333 0.3333 0)"]),
    ("dia", 3, ["GAMMA", "X", "U", "K", "L", "W"], "8", None),
])
def test_band_plot_shows_its_zone_and_path(app, key, dim, points, iss, segs):
    drv = app
    folder, stem, ext = RUNS[key]
    open_plot(drv, f"{key}/{stem}.{ext}")
    s = section(drv)
    assert s and s["tier"] is None and s["open"] and s["shown"], s          # every tier, open
    st = zone(drv)
    assert st["loaded"] and not st["failed"] and st["dim"] == dim and st["points"] == points, st
    f = s["facts"]
    assert f["ISS (shrinking factor)"] == iss and f["k points"] == "1000" and f["From"] == f"{stem}.d3", f
    if segs:
        assert s["segs"] == segs, s["segs"]
    else:
        assert len(s["segs"]) == 6 and f["Path"] == "Γ → X → U | K → Γ → L → W → X", s
    # the zone view is the section's own, not the k-path editor's (Advanced)
    assert not drv.find_elements("css selector", "#inspector .po-kpath .kp-bz")
    if key == "gr":
        H.wait(drv, "return !!document.querySelector('#inspector .po-bz .kp-bz canvas')", 20)
        time.sleep(0.5)
        if os.environ.get("MACE_GUI_SHOTS"):            # a screenshot for review
            Path(os.environ["MACE_GUI_SHOTS"]).mkdir(parents=True, exist_ok=True)
            drv.save_screenshot(str(Path(os.environ["MACE_GUI_SHOTS"]) / "band_bz_inspector_graphene.png"))
    assert not H.errors(drv), H.errors(drv)


def test_phonon_fort25_path_from_its_records_and_deck(app):
    drv = app
    open_plot(drv, "ph/dia_phonon_f25.fort.25")
    s = section(drv)
    f = s["facts"]
    assert f["ISS (shrinking factor)"] == "8" and f["Segments"] == "4" and f["k points"] == "197", f
    assert f["From"] == "dia_phonon_f25.fort.25", f
    assert s["segs"][0].startswith("Γ (0 0 0) →") and "(0.5 0 0.5)" in s["segs"][0], s["segs"]
    st = zone(drv)
    # no CRYSTAL output beside it: the path is listed, the zone says why it is not drawn (and no empty box)
    assert st["failed"] and "no CRYSTAL output" in st["failed"], st
    assert drv.execute_script("return document.querySelector('#inspector .po-bz .kp-bz-gl').hidden") is True
    assert not H.errors(drv), H.errors(drv)
