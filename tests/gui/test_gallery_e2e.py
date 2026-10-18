"""The Plot Gallery (MACE 1.1.6) in real browsers, Firefox and Chromium.

* a folder of structures opens as a grid of thumbnails with a caption table
  (material id and formula from the folder's materials.db);
* 205 structures are capped at 200 panes with the message, and only the panes
  near the view are drawn (the rest are not read or are purged again);
* columns change the grid, cycle mode shows one pane at a time and the arrow
  keys move through the set (wrapping at the ends), a caption row jumps to
  its pane;
* Export SVG/PNG downloads the figure `mace plotting --gallery` draws (the
  server runs the CLI), Captions CSV downloads the table;
* with the corpus: a folder of band structures draws plotly panes with the
  synchronized energy window, and band + DOS pairs each band file with its
  DOSS file in a second panel.
The shipped demo files for everything but the corpus test."""
import os
import shutil
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")

BROWSERS = ["firefox", "chromium"]


def _root(tmp):
    root = H.examples_copy(tmp)
    many = root / "many"
    many.mkdir()
    src = root / "structures" / "Si.cif"
    for k in range(205):
        shutil.copyfile(src, many / f"Si_copy{k:03d}.cif")
    return root


@pytest.fixture(scope="module", params=BROWSERS)
def app(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp(f"gallery_{request.param}")
    root = _root(tmp)
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1")
    downloads = tmp / "downloads"
    downloads.mkdir()
    drv = None
    try:
        drv = H.browser(request.param, width=1280, height=900, download_dir=downloads)
        H.load(drv, base)
        drv.execute_script("window.__mace.settings.set('reduceMotion', 'on')")
        yield drv, root, downloads, request.param
    finally:
        if drv is not None:
            drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def gstate(drv):
    return drv.execute_script("const v = window.__mace.galleryView; return v ? v.state() : null")


def open_gallery(drv, source, kind, label="test"):
    drv.execute_script("window.__mace.closeAll()")
    drv.execute_script("window.__mace.openGallery({source: arguments[0], kind: arguments[1], label: arguments[2]})",
                       source, kind, label)
    H.wait(drv, "const v = window.__mace.galleryView; const s = v && v.state(); return s && s.items > 0 && s.drawn > 0", 60)
    return gstate(drv)


def test_structures_grid_captions_and_cycle(app):
    drv, root, downloads, name = app
    s = open_gallery(drv, {"type": "folder", "folder": "structures"}, "structure", "structures")
    assert s["items"] == 6 and s["kind"] == "structure" and not s["capped"]
    H.wait(drv, "return window.__mace.galleryView.state().drawn === 6", 30)
    # the thumbnails: one SVG of circles per pane
    n_atoms = drv.execute_script("return [...document.querySelectorAll('#gallery-host .gal-grid .gal-atoms')].map((s) => s.querySelectorAll('circle').length)")
    assert len(n_atoms) == 6 and all(n > 0 for n in n_atoms)
    rows = drv.execute_script("return [...document.querySelectorAll('#gallery-host .gal-table tbody tr')].map((r) => [...r.cells].map((c) => c.textContent.trim()))")
    assert [r[1] for r in rows] == ["diamond", "hBN", "LiF", "MgO", "NaCl", "Si"]
    assert dict((r[1], r[2]) for r in rows)["MgO"] == "Mg4O4"      # the formula from examples/gui/materials.db
    assert "Captions from materials.db" in s["note"]
    # columns
    drv.execute_script("const i = document.querySelector('#gallery-host .gal-cols'); i.value = 2; i.dispatchEvent(new Event('change'))")
    cols = drv.execute_script("return getComputedStyle(document.querySelector('#gallery-host .gal-grid')).gridTemplateColumns.split(' ').length")
    assert cols == 2
    # cycle mode: one at a time, the arrow keys, wrapping
    drv.find_element("css selector", '#gallery-host [data-act="cycle"]').click()
    H.wait(drv, "return window.__mace.galleryView.state().cycleDrawn", 30)
    assert gstate(drv)["mode"] == "cycle" and gstate(drv)["index"] == 0
    assert drv.execute_script("return document.querySelector('#gallery-host .gal-grid').hidden")
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.common.keys import Keys
    ActionChains(drv).send_keys(Keys.ARROW_RIGHT).perform()
    H.wait(drv, "return window.__mace.galleryView.state().index === 1", 10)
    assert "2 / 6" in drv.execute_script("return document.querySelector('#gallery-host .gal-count').textContent")
    ActionChains(drv).send_keys(Keys.ARROW_LEFT).send_keys(Keys.ARROW_LEFT).perform()
    H.wait(drv, "return window.__mace.galleryView.state().index === 5", 10)      # wrapped to the last
    ActionChains(drv).send_keys(Keys.HOME).perform()
    H.wait(drv, "return window.__mace.galleryView.state().index === 0", 10)
    # a caption row jumps to its pane
    drv.execute_script("document.querySelector('#gallery-host .gal-table tbody tr[data-i=\"3\"]').click()")
    H.wait(drv, "return window.__mace.galleryView.state().index === 3 && window.__mace.galleryView.state().cycleDrawn", 10)
    assert drv.execute_script("return document.querySelector('#gallery-host .gal-table tr[aria-current=\"true\"]').dataset.i") == "3"
    drv.find_element("css selector", '#gallery-host [data-act="grid"]').click()
    assert gstate(drv)["mode"] == "grid"
    U.shot(drv, f"gallery_structures_{name}")
    U.no_errors(drv)


def _downloaded(downloads, name, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        p = downloads / name
        if p.is_file() and p.stat().st_size > 0 and not list(downloads.glob("*.part")) and not list(downloads.glob("*.crdownload")):
            return p
        time.sleep(0.3)
    raise AssertionError(f"{name} was not downloaded: {sorted(x.name for x in downloads.iterdir())}")


def test_export_figure_and_captions(app):
    drv, root, downloads, name = app
    open_gallery(drv, {"type": "folder", "folder": "structures"}, "structure", "structures")
    for f in downloads.iterdir():
        f.unlink()
    drv.find_element("css selector", '#gallery-host [data-act="export-svg"]').click()
    svg = _downloaded(downloads, "gallery_structure.svg")
    text = svg.read_text()
    assert text.lstrip().startswith(("<?xml", "<svg")) and "6. Si" in text and "1. diamond" in text
    drv.find_element("css selector", '#gallery-host [data-act="export-png"]').click()
    png = _downloaded(downloads, "gallery_structure.png")
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    drv.find_element("css selector", '#gallery-host [data-act="export-csv"]').click()
    csv_p = _downloaded(downloads, "gallery_structure_captions.csv")
    lines = csv_p.read_text().splitlines()
    assert lines[0] == "n,material_id,formula,band_gap_eV,energy_eV,file"
    assert lines[4].startswith("4,MgO,Mg4O4,") and lines[4].endswith("structures/MgO.cif")
    cmd = drv.execute_script("return window.__mace.galleryView.command()")
    assert cmd.startswith("mace plotting --gallery structure structures --cols 3")
    U.no_errors(drv)


def test_cap_and_lazy_rendering(app):
    drv, root, downloads, name = app
    s = open_gallery(drv, {"type": "folder", "folder": "many"}, "structure", "many")
    assert s["items"] == 200 and s["total"] == 205 and s["capped"]
    assert "Showing the first 200 of 205 files" in s["note"]
    H.wait(drv, "return window.__mace.galleryView.state().loading === 0", 30)
    first = gstate(drv)
    # only the panes near the view: a 3-column grid shows a few rows of 200 panes
    assert 0 < first["drawn"] <= 45, first
    drv.execute_script("document.querySelector('#gallery-host .gal-pane[data-i=\"199\"]').scrollIntoView({block: 'end'})")
    H.wait(drv, "const s = window.__mace.galleryView.state(); return s.loading === 0 && "
                "document.querySelector('#gallery-host .gal-pane[data-i=\"199\"] .gal-plot').dataset.state === 'drawn'", 30)
    after = gstate(drv)
    assert after["drawn"] <= 45, after                    # the first rows were purged again
    assert drv.execute_script("return document.querySelector('#gallery-host .gal-pane[data-i=\"0\"] .gal-plot').dataset.state") == "idle"
    assert after["fetched"] < 120, after                  # nowhere near every pane was read
    U.no_errors(drv)


@pytest.fixture(scope="module")
def bands_root(app):
    """Band and DOSS files of the corpus beside each other (run names kept: the pairing is by name)."""
    import runcorpus as RC
    drv, root, downloads, name = app
    band = sorted((RC.CORPUS / "BAND").glob("*.BAND.DAT"))[:4] if (RC.CORPUS / "BAND").is_dir() else []
    if not band:
        pytest.skip("test/ data corpus not present (gitignored)")
    dst = root / "bands"
    if not dst.is_dir():
        for b in band:
            stem = b.name[: -len(".BAND.DAT")]
            mat = stem[: -len("_band")]
            RC.copy_into(dst, b, *[p for p in [b.with_name(stem + ".d3")] if p.is_file()])
            dos = RC.CORPUS / "DOSS" / f"{mat}_doss.DOSS.DAT"
            if dos.is_file():
                RC.copy_into(dst, dos, *[p for p in [dos.with_name(f"{mat}_doss.d3")] if p.is_file()])
    return dst


def test_band_and_band_dos_panes(app, bands_root):
    drv, root, downloads, name = app
    s = open_gallery(drv, {"type": "folder", "folder": "bands"}, "band", "bands")
    assert s["items"] >= 1
    H.wait(drv, f"return window.__mace.galleryView.state().drawn === {s['items']}", 60)
    fig = drv.execute_script("return window.__mace.galleryView.figure(0)")
    lo, hi = fig["layout"]["y"]
    win = drv.execute_script("return window.__mace.settings.get('plots.energyWindow')")
    assert hi - lo == pytest.approx(win[1] - win[0])     # the synchronized window: Settings' (-6 to 6 eV around E_F)
    drv.execute_script("const a = document.querySelector('#gallery-host .gal-emin'), b = document.querySelector('#gallery-host .gal-emax');"
                       "a.value = -3; b.value = 2; a.dispatchEvent(new Event('change')); b.dispatchEvent(new Event('change'))")
    H.wait(drv, "const f = window.__mace.galleryView.figure(0); return f && Math.abs((f.layout.y[1] - f.layout.y[0]) - 5) < 1e-6", 60)
    drv.execute_script("const k = document.querySelector('#gallery-host .gal-kind'); k.value = 'bandos'; k.dispatchEvent(new Event('change'))")
    H.wait(drv, "const v = window.__mace.galleryView.state(); return v.kind === 'bandos' && v.drawn > 0 && v.loading === 0", 60)
    two = drv.execute_script("return [...document.querySelectorAll('#gallery-host .gal-pane .gal-plot')].filter((e) => e.layout && e.layout.xaxis2).length")
    assert two >= 1                                      # a band panel and its DOS panel
    U.shot(drv, f"gallery_bandos_{name}")
    U.no_errors(drv)


def test_spin_look_and_dos_rule_in_the_band_dos_gallery(app):
    """The Gallery's command names Settings › Plots spin look (mace plotting reads it back), and a
    spin DOS pane follows the DOS plot's rule: with both spins every projection is solid."""
    drv, root, downloads, name = app
    dst = root / "spin_fe"
    if not dst.exists():
        shutil.copytree(H.REPO_ROOT / "tests" / "data" / "spin_fe", dst)
    drv.execute_script("const s = window.__mace.settings; s.set('plots.spinColors', 'blue-red'); s.set('plots.spinDownDash', 'dot')")
    try:
        open_gallery(drv, {"type": "folder", "folder": "spin_fe"}, "bandos", "spin")
        plot = "document.querySelector('#gallery-host .gal-pane[data-i=\"0\"] .gal-plot')"
        H.wait(drv, f"const el = {plot}; return el && el.data && el.data.length", 60)
        traces = drv.execute_script(f"return {plot}.data.map((t) => [t.name, t.line.dash, t.xaxis])")
        projs = [t for t in traces if t[2] == "x2" and not t[0].startswith("Total")]
        assert projs, traces
        for nm, dash, _ in projs:
            assert dash == ("dot" if nm.endswith("β") else "solid"), (nm, dash)
        cmd = drv.execute_script("return window.__mace.galleryView.command()")
        import shlex
        from mace.plotting.main import create_parser
        args = create_parser().parse_args(shlex.split(cmd)[2:])
        assert (args.spin_colors, args.spin_style) == ("blue-red", "dot"), cmd
    finally:
        drv.execute_script("const s = window.__mace.settings; s.set('plots.spinColors', 'auto'); s.set('plots.spinDownDash', 'dash')")
    U.no_errors(drv)


def test_dos_gallery_pane_follows_the_dos_layout(app):
    """A spin DOS pane in the 1.1.6 DOS layout, as mace plotting --gallery draws it: a symmetric DOS axis
    with |DOS| ticks, the up/down labels, the totals filled in the spin pair, projections in the
    preset's projection colors clear of the pair, a dotted E_F and the smoothing named in the pane."""
    from mace.plotting import gallery as G
    from mace.plotting import presets as P
    drv, root, downloads, name = app
    dst = root / "spin_fe"
    if not dst.exists():
        shutil.copytree(H.REPO_ROOT / "tests" / "data" / "spin_fe", dst)
    drv.execute_script("const s = window.__mace.settings; s.set('plots.spinColors', 'blue-red'); s.set('plots.dosSmooth', 0.05)")
    try:
        open_gallery(drv, {"type": "folder", "folder": "spin_fe"}, "dos", "spin dos")
        plot = "document.querySelector('#gallery-host .gal-pane[data-i=\"0\"] .gal-plot')"
        H.wait(drv, f"const el = {plot}; return el && el.data && el.data.length", 60)
        fig = drv.execute_script(f"""const el = {plot}, L = el.layout;
          return {{traces: el.data.map((t) => ({{name: t.name, color: t.line.color, dash: t.line.dash, fill: t.fill || null,
                                                fillcolor: t.fillcolor || null}})),
                   range: L.yaxis.range, ticktext: L.yaxis.ticktext || null, zeroline: !!L.yaxis.zeroline,
                   notes: (L.annotations || []).map((a) => a.text), dots: (L.shapes || []).filter((s) => s.line && s.line.dash === 'dot').length,
                   bg: L.plot_bgcolor, spin: window.MACE.spinStyle('band')}}""")
        lo, hi = fig["range"]
        assert lo == pytest.approx(-hi) and fig["zeroline"] and fig["dots"] >= 1, fig
        assert fig["ticktext"] and not any(str(t).startswith(("-", "−")) for t in fig["ticktext"]), fig["ticktext"]
        assert "<b>↑ up</b>" in fig["notes"] and "<b>down ↓</b>" in fig["notes"]
        assert any(n.startswith("smoothed, σ = 0.05") for n in fig["notes"]), fig["notes"]
        pair = fig["spin"]
        fills = P.spin_fills(pair, "default")
        tot = {t["name"]: t for t in fig["traces"] if t["name"].startswith("Total")}
        assert tot["Total"]["fillcolor"] == fills["up"] and tot["Total β"]["fillcolor"] == fills["down"], tot
        assert tot["Total"]["color"] == pair["up"] and tot["Total β"]["color"] == pair["down"]
        pn = G.pane(dst / "fe_doss.DOSS.DAT", "dos", (-6.0, 6.0), dos_smooth=0.05)["panels"][0]
        bg = fig["bg"] if str(fig["bg"]).startswith("#") else "#ffffff"
        ppal = P.projection_palette("default", [pair["up"], pair["down"]], bg)
        want = {lab: ppal[i % len(ppal)] for i, lab in enumerate(pn["dos_order"])}
        names = {s["name"]: s["proj"] for s in pn["series"] if s["role"] == "projection"}
        projs = [t for t in fig["traces"] if t["name"] in names]
        assert projs and len({names[t["name"]] for t in projs}) <= 4
        for t in projs:
            assert t["color"] == want[names[t["name"]]], t
            assert t["dash"] == ("dash" if t["name"].endswith("β") else "solid"), t
    finally:
        drv.execute_script("const s = window.__mace.settings; s.set('plots.spinColors', 'auto'); s.set('plots.dosSmooth', 0)")
    U.no_errors(drv)
