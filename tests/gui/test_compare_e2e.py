"""The Compare document in headless Firefox (plan T10, B7).

The four looks render differently and two of them carry a legend; Mod+Alt+O
cycles them; Compare with Input on the demo optimization uses its .d12;
Export Table writes the displacement table with one row per atom; a
shuffled copy is matched by position (banner) and a supercell is refused
(reason, no tables); the dark ghost stays visible on the viewport."""
import csv
import io
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
ase = pytest.importorskip("ase")
from selenium.webdriver.common.keys import Keys  # noqa: E402

OPT = "outputs/Ag2Br3_opt_unfinished.out"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    import numpy as np
    from ase.io import read, write
    tmp = tmp_path_factory.mktemp("p5cmp")
    root = H.examples_copy(tmp)
    (root / "cmp").mkdir()
    m = read(str(root / "structures" / "MgO.cif"))
    write(str(root / "cmp" / "MgO_shuffled.cif"), m[np.random.default_rng(1).permutation(len(m))])
    shifted = m.copy()
    shifted.positions += [1.05, 1.05, 0.0]                      # every atom moved off its old site
    write(str(root / "cmp" / "MgO_shifted.cif"), shifted)
    write(str(root / "cmp" / "MgO_222.cif"), m.repeat((2, 2, 2)))
    a = U.start_app(tmp, root=root)
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    app[0].execute_script("window.MACE.setLook('overlay')")
    return app


def open_run(drv):
    U.open_doc(drv, OPT, wait_atoms=False)
    H.wait(drv, "return !!(window.MACE.state.traj && window.MACE.state.cmp && window.MACE.state.viewers['viewer-before'])", 30)
    time.sleep(0.6)


def viewer_png(drv):
    return drv.find_element("css selector", "#viewer-before").screenshot_as_png


def test_four_looks_render_and_cycle(fresh):
    drv = fresh[0]
    open_run(drv)
    pngs, legends = {}, {}
    for look in ("overlay", "twotone", "side", "dmap"):
        drv.execute_script("window.MACE.setLook(arguments[0])", look)
        H.wait(drv, "return document.querySelector('#o-legend').dataset.look === arguments[0]", 10, look)
        time.sleep(0.8)
        pngs[look] = viewer_png(drv)
        legends[look] = drv.execute_script("return document.querySelector('#o-legend').textContent")
        U.shot(drv, f"compare_{look}")
    assert len(set(pngs.values())) == 4                             # each look draws something else
    assert "initial" in legends["twotone"] and "final" in legends["twotone"]
    assert "Å" in legends["dmap"] and "|Δr|" in legends["dmap"]
    assert drv.execute_script("return !!document.querySelector('#o-legend .seq-bar')")
    drv.execute_script("window.MACE.setLook('overlay')")
    seen = []
    for _ in range(4):
        drv.execute_script("window.__mace.run('traj.overlay')")
        seen.append(drv.execute_script("return window.MACE.look()"))
    assert seen == ["twotone", "side", "dmap", "overlay"]
    U.no_errors(drv)


def test_compare_with_input_and_export_table(fresh):
    drv, downloads = fresh[0], fresh[5]
    U.open_doc(drv, OPT, wait_atoms=False)
    assert drv.execute_script("return window.__mace.menuReason('analysis.compareInput')") == ""
    drv.execute_script("window.__mace.run('analysis.compareInput')")
    H.wait(drv, "return /vs its input/.test((window.__mace.activeDoc() || {}).title || '')", 30)
    status = drv.execute_script("return document.querySelector('#o-status').textContent")
    assert "input deck" in status, status
    n = drv.execute_script("return window.MACE.state.traj.frames[1].numbers.length")
    # Analysis › Displacement Table opens its sub-tab; Export Table writes its rows
    drv.execute_script("window.__mace.run('analysis.displacements')")
    H.wait(drv, "return document.querySelector('.cmp-subs [data-sub=disp]').classList.contains('active')", 5)
    assert drv.execute_script("return window.__mace.menuReason('analysis.exportCsv')") == ""
    for f in downloads.glob("*_displacements.csv"):
        f.unlink()
    drv.execute_script("window.__mace.run('analysis.exportCsv')")
    end = time.time() + 15
    files = []
    while time.time() < end and not files:
        files = [f for f in downloads.glob("*_displacements.csv") if f.stat().st_size]
        time.sleep(0.2)
    assert files, list(downloads.iterdir())
    rows = list(csv.reader(io.StringIO(files[0].read_text())))
    assert rows[0] == ["atom", "|Δr| Å", "Δx", "Δy", "Δz"] and len(rows) - 1 == n
    # rows sorted by |Δr|, largest first, as shown
    vals = [float(r[1]) for r in rows[1:]]
    assert vals == sorted(vals, reverse=True)
    # Copy TSV gives the same table
    drv.execute_script("document.querySelector('.cmp-copy').click()")
    tsv = drv.execute_script("return window.__maceLastCopy")
    assert tsv.split("\n")[0] == "atom\t|Δr| Å\tΔx\tΔy\tΔz" and len(tsv.split("\n")) == n + 1
    # header click sorts; arrows move along the grid rows
    drv.execute_script("document.querySelector('#o-disp .th-sort[data-key=atom]').click()")
    first = drv.execute_script("return document.querySelector('#o-disp tbody tr td').textContent")
    assert first == "Ag1", first
    drv.execute_script("document.querySelector('#o-disp tbody tr').focus()")
    U.keys(drv, Keys.END)
    assert drv.execute_script("return document.activeElement.getAttribute('aria-rowindex')") == str(n + 1)
    # the symmetry row names both ends
    drv.execute_script("window.__mace.run('analysis.bondTable')")
    H.wait(drv, "return document.querySelector('#o-sym').textContent.includes('→')", 30)
    U.shot(drv, "compare_input_tables")
    U.no_errors(drv)


def test_position_match_banner_and_refusal(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_async_script("const done = arguments[1]; window.__mace.compareWith(arguments[0]).then(() => done(1))", "cmp/MgO_shuffled.cif")
    H.wait(drv, "return !document.querySelector('#o-banner').classList.contains('hidden')", 20)
    assert drv.execute_script("return document.querySelector('#o-banner').textContent").startswith("Atoms matched by position (max mismatch 0.000 Å)")
    U.shot(drv, "compare_position_matched")
    # a supercell against its cell: refused with the reason
    st, _, r = U.req(fresh[1], "api/compare?a=structures/MgO.cif&b=cmp/MgO_222.cif")
    assert st == 400 and r["error"].startswith("8 atoms vs 64")
    drv.execute_script("window.MACE.showTab('opt'); document.querySelector('#opt-empty').classList.remove('hidden');"
                       "const a = document.querySelector('#cmp-a'), b = document.querySelector('#cmp-b');"
                       "a.value = 'structures/MgO.cif'; b.value = 'cmp/MgO_222.cif'; document.querySelector('#cmp-go').click()")
    H.wait(drv, "return !document.querySelector('#cmp-refused').classList.contains('hidden')", 20)
    assert "Build › Supercell" in drv.execute_script("return document.querySelector('#cmp-refused').textContent")


def _lum(rgb):
    def ch(c):
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgb[:3]
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def test_dark_ghost_contrast(fresh):
    from PIL import Image
    drv = fresh[0]
    drv.execute_script("window.__mace.settings.set('theme', 'dark')")
    try:
        U.open_doc(drv, "structures/MgO.cif")
        drv.execute_async_script("const done = arguments[1]; window.__mace.compareWith(arguments[0]).then(() => done(1))", "cmp/MgO_shifted.cif")
        H.wait(drv, "return !!window.MACE.state.cmp", 20)
        op = drv.execute_script("return Number(getComputedStyle(document.documentElement).getPropertyValue('--cmp-ghost-opacity'))")
        assert op == 0.6
        # only the ghost on screen (a test hook), so its pixels can be measured
        drv.execute_script("window.MACE.state.soloGhost = true; document.querySelector('#o-arrows').checked = false; window.MACE.setLook('overlay')")
        time.sleep(1.0)
        img = Image.open(io.BytesIO(viewer_png(drv))).convert("RGB")
        bg = drv.execute_script("const c = getComputedStyle(document.documentElement).getPropertyValue('--viewport-bg').trim(); return c")
        bgc = tuple(int(bg[i:i + 2], 16) for i in (1, 3, 5))
        lb = _lum(bgc)
        px = [p for p in img.getdata() if sum(abs(a - b) for a, b in zip(p, bgc)) > 30]
        ratios = sorted((max(_lum(p), lb) + 0.05) / (min(_lum(p), lb) + 0.05) for p in px)
        assert len(ratios) > 500
        U.shot(drv, "compare_dark_ghost")
        med = ratios[len(ratios) // 2]
        print(f"dark ghost median contrast {med:.2f}:1")
        assert med >= 1.6, med                                      # the atom fill, not just its highlight
    finally:
        drv.execute_script("window.MACE.state.soloGhost = false; document.querySelector('#o-arrows').checked = true;"
                           "window.__mace.settings.set('theme', 'light')")
