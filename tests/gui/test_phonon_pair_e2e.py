"""A phonon dispersion and a phonon DOS from two folders (1.1.6): Add Phonon DOS… on
a phonon dispersion plot and Add Phonon Bands… on a phonon DOS plot, the
electronic Add DOS… / Add Band… path for phonons. The picker lists the phonon
runs' files only, the server refuses another system, and the two halves are drawn
on one frequency axis (zero frequency shared, the plot's unit). The pairing is
kept per document; Copy CLI Command and Publication Figure carry it (mace
plotting --phonon-band-dos --partner). Firefox, the bundled example suite's
diamond PHONON_BAND/ and PHONON_DOS/ folders."""
import lzma
import time

import pytest

import guiharness as H

pytestmark = [pytest.mark.gui, pytest.mark.guismoke]
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")

SUITE = H.REPO_ROOT / "examples" / "suite"
DIAMOND = SUITE / "2-dense-3d" / "diamond"
HBN = SUITE / "3-slab-2d" / "hBN"
PH_BAND = "diamond/PHONON_BAND/diamond_phonon_band.f25"
PH_DOS = "diamond/PHONON_DOS/diamond_phonon_dos.f25"
OTHER = "hBN/PHONON_DOS/hBN_phonon_dos.f25"
EL_BAND = "diamond/BAND/diamond_band.f25"
EL_DOS = "diamond/DOSS/diamond_doss.DOSS.DAT"


def put(src_dir, names, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        xz, plain = src_dir / (name + ".xz"), src_dir / name
        if xz.is_file():
            (dst_dir / name).write_bytes(lzma.decompress(xz.read_bytes()))
        elif plain.is_file():
            (dst_dir / name).write_bytes(plain.read_bytes())
        else:
            pytest.skip("the bundled example suite is not here")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("phpair")
    root = H.examples_copy(tmp)
    run = lambda m, k: tuple(f"{m}_{k}{ext}" for ext in (".f25", ".out", ".d12"))
    put(DIAMOND / "PHONON_BAND", run("diamond", "phonon_band"), root / "diamond" / "PHONON_BAND")
    put(DIAMOND / "PHONON_DOS", run("diamond", "phonon_dos"), root / "diamond" / "PHONON_DOS")
    put(HBN / "PHONON_DOS", run("hBN", "phonon_dos"), root / "hBN" / "PHONON_DOS")
    put(DIAMOND / "BAND", ("diamond_band.f25", "diamond_band.out", "diamond_band.d3"), root / "diamond" / "BAND")
    put(DIAMOND / "DOSS", ("diamond_doss.DOSS.DAT", "diamond_doss.out", "diamond_doss.d3"), root / "diamond" / "DOSS")
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1")
    drv = None
    try:
        drv = H.browser("firefox", width=1400, height=900)
        drv.set_script_timeout(240)
        H.load(drv, base)
        drv.execute_script("window.__mace.settings.set('reduceMotion', 'on');"
                           "Object.keys(localStorage).filter((k) => k.startsWith('mace.gui.plot.')).forEach((k) => localStorage.removeItem(k))")
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
    time.sleep(0.4)


def buttons(drv):
    return drv.execute_script("return [...document.querySelectorAll('#plot-host .pv-acts > button')].map((b) => b.textContent.trim())")


def click(drv, label):
    drv.execute_script("[...document.querySelectorAll('#plot-host .pk-own')].find((b) => b.textContent === arguments[0]).click()", label)
    H.wait(drv, "const d = document.getElementById('plot-pair'); return !!(d && d.open)", 10)


def listed(drv):
    return drv.execute_script("return [...document.querySelectorAll('#plot-pair [data-pair]')].map((b) => b.dataset.pair)")


def pick(drv, path):
    H.wait(drv, "return !!document.querySelector(`#plot-pair [data-pair=\"${arguments[0]}\"]`)", 10, path)
    drv.execute_script("document.querySelector(`#plot-pair [data-pair=\"${arguments[0]}\"]`).click()", path)


SHARED = """const el = window.__mace.plotView.el(), l = el.layout, d = el.data || [];
  if (!window.__mace.plotView.state().drawn || !l || !l.xaxis2) return null;
  return {y: [...new Set(d.map((t) => t.yaxis || 'y'))], x: [...new Set(d.map((t) => t.xaxis || 'x'))], y2: !!l.yaxis2,
          range: l.yaxis.range, title: l.yaxis.title && l.yaxis.title.text, dos: l.xaxis2.title && l.xaxis2.title.text}"""


def test_add_phonon_dos_to_the_diamond_dispersion(app):
    drv = app
    open_plot(drv, PH_BAND)
    b = buttons(drv)
    assert "Add Phonon DOS…" in b and "Add DOS…" not in b, b
    click(drv, "Add Phonon DOS…")
    names = listed(drv)
    # the phonon runs' DOS files only, the same material first: never the electronic DOS
    assert names[0] == PH_DOS and OTHER in names and EL_DOS not in names, names
    pick(drv, OTHER)
    H.wait(drv, "return /not the same system/.test(document.querySelector('#plot-pair .pv-pair-msg').textContent)", 30)
    assert drv.execute_script("return document.getElementById('plot-pair').open") is True
    pick(drv, PH_DOS)
    H.wait(drv, "return !document.getElementById('plot-pair').open", 30)
    fig = H.wait(drv, SHARED, 60)
    # both panels on the one frequency axis: every trace on y, the DOS on x2, no second y axis
    assert fig["y"] == ["y"] and "x2" in fig["x"] and not fig["y2"], fig
    assert fig["title"].startswith("Frequency") and "cm" in fig["title"] and fig["range"][0] <= 0, fig
    assert 1300 < fig["range"][1] < 1500, fig                # the modes' window (diamond's top mode ~1330 cm⁻¹)
    b = buttons(drv)
    assert "Remove Phonon DOS" in b and "Log DOS" in b and "Add Phonon DOS…" not in b, b
    assert drv.execute_script("return window.__mace.plotView.state().opts.pair") == PH_DOS
    chips = drv.execute_script("return document.querySelector('#plot-host .pv-chips').innerHTML")
    assert "with ../PHONON_DOS/diamond_phonon_dos.f25" in chips and "zero frequency" in chips, chips
    cli = drv.execute_script("return window.__mace.plotView.cliText()")
    assert cli.startswith("mace plotting --phonon-band-dos diamond_phonon_band.f25 --partner ../PHONON_DOS/diamond_phonon_dos.f25"), cli
    # kept per document: reopened, it is drawn with its DOS again
    open_plot(drv, PH_BAND)
    H.wait(drv, SHARED, 60)
    # Publication Figure: mace plotting --phonon-band-dos with the other half (--partner)
    r = drv.execute_async_script("const done = arguments[0]; window.__mace.plotView.publication().then((x) => done({ok: true, r: x}), "
                                 "(e) => done({ok: false, e: String(e.message || e)}))")
    assert r["ok"], r
    assert "--phonon-band-dos" in r["r"]["command"] and "--partner" in r["r"]["command"], r["r"]["command"]
    assert "diamond.PHONON_BAND_DOS.svg" in r["r"]["images"], r["r"]
    # Remove Phonon DOS: the dispersion alone again
    drv.execute_script("[...document.querySelectorAll('#plot-host .pk-own')].find((b) => b.textContent === 'Remove Phonon DOS').click()")
    H.wait(drv, "const l = window.__mace.plotView.el().layout; return window.__mace.plotView.state().drawn && l && !l.xaxis2", 60)
    assert "Add Phonon DOS…" in buttons(drv)
    assert not H.errors(drv), H.errors(drv)


def test_add_phonon_bands_to_a_phonon_dos_and_electronic_plots_list_no_phonons(app):
    drv = app
    open_plot(drv, PH_DOS)
    assert "Add Phonon Bands…" in buttons(drv) and "Log Y" in buttons(drv)
    click(drv, "Add Phonon Bands…")
    names = listed(drv)
    assert names == [PH_BAND], names                         # the electronic diamond_band.f25 is not offered
    pick(drv, PH_BAND)
    H.wait(drv, SHARED, 60)
    assert "Remove Phonon Bands" in buttons(drv)
    cli = drv.execute_script("return window.__mace.plotView.cliText()")
    assert cli.startswith("mace plotting --phonon-band-dos diamond_phonon_dos.f25 --partner ../PHONON_BAND/diamond_phonon_band.f25"), cli
    # an electronic band plot's Add DOS… lists the electronic DOS, not a phonon one
    open_plot(drv, EL_BAND)
    assert "Add DOS…" in buttons(drv)
    click(drv, "Add DOS…")
    names = listed(drv)
    assert EL_DOS in names and PH_DOS not in names and OTHER not in names, names
    drv.execute_script("document.getElementById('plot-pair').close()")
    assert not H.errors(drv), H.errors(drv)
