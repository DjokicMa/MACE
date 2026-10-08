"""SVG export (plan T12, B10).

One <circle> per drawn atom (boundary atoms and the display range included),
a valid standalone SVG (xml.etree parses it, no external references, it
opens in the browser as image/svg+xml), refused in perspective with the
reason, two groups for a side-by-side comparison, and 8000 atoms written in
≤1.5 s without blocking the page for more than 200 ms at a time."""
import time
import xml.etree.ElementTree as ET

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")

SVG = "{http://www.w3.org/2000/svg}"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    from ase.io import read, write
    tmp = tmp_path_factory.mktemp("p5svg")
    root = H.examples_copy(tmp)
    (root / "big").mkdir()
    write(str(root / "big" / "MgO_1000.cif"), read(str(root / "structures" / "MgO.cif")).repeat((10, 10, 10)))
    a = U.start_app(tmp, root=root)
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    app[0].execute_script("window.__svg = null; window.__maceExportHook = (t, kind) => { if (kind === 'svg') window.__svg = t; };")
    return app


def export(drv, **o):
    return drv.execute_async_script("""const done = arguments[arguments.length - 1];
      window.__mace.exportSvg(arguments[0]).then((n) => done({ n, text: window.__svg }), (e) => done({ error: String(e.message || e) }));""", o)


def circles(text):
    root = ET.fromstring(text)
    return root, root.findall(f".//{SVG}circle")


def test_one_circle_per_drawn_atom_and_valid_svg(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    n1 = drv.execute_script("return window.__mace.viewState().displayedAtoms")
    assert n1 > 8                                            # boundary atoms are drawn too
    r = export(drv)
    assert "error" not in r, r
    root, c = circles(r["text"])
    assert len(c) == n1 == r["n"]
    assert root.tag == f"{SVG}svg"
    assert not any(k.endswith("href") for el in root.iter() for k in el.attrib)     # nothing external
    assert "http" not in r["text"].replace("http://www.w3.org/2000/svg", "")
    # the picture lines up with the screen: O5 where the viewer draws it
    scr = drv.execute_script("return window.__mace.viewport.atomScreen(4)")
    o5 = [e for e in c if e.findtext(f"{SVG}title") == "O5"]
    assert any(abs(float(e.get("cx")) - scr["x"]) < 1.5 and abs(float(e.get("cy")) - scr["y"]) < 1.5 for e in o5), (scr, [(e.get("cx"), e.get("cy")) for e in o5])
    # bonds and the cell are lines
    assert len(root.findall(f".//{SVG}line")) >= 12
    # it opens in the browser as an SVG image
    ok = drv.execute_async_script("""const done = arguments[arguments.length - 1];
      const u = URL.createObjectURL(new Blob([arguments[0]], { type: 'image/svg+xml' }));
      const i = new Image(); i.onload = () => done([i.naturalWidth, i.naturalHeight]); i.onerror = () => done(null); i.src = u;""", r["text"])
    assert ok and ok[0] > 100 and ok[1] > 100, ok
    # the display range multiplies the circles
    drv.execute_script("window.__mace.run('view.range2')")
    H.wait(drv, "return window.__mace.viewState().displayedAtoms > arguments[0]", 10, n1)
    time.sleep(0.3)
    n2 = drv.execute_script("return window.__mace.viewState().displayedAtoms")
    r2 = export(drv)
    assert len(circles(r2["text"])[1]) == n2
    drv.execute_script("window.__mace.run('view.range1')")
    U.no_errors(drv)


def test_perspective_is_refused(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.settings.set('projection', 'perspective')")
    try:
        r = export(drv)
        assert r.get("error") == "SVG export is orthographic; press 5"
        drv.execute_script("window.__mace.run('file.exportImage')")
        H.wait(drv, "return !!document.querySelector('.export-dlg [data-k=format] option[value=svg]')", 10)
        opt = drv.execute_script("const o = document.querySelector('.export-dlg [data-k=format] option[value=svg]'); return [o.disabled, o.textContent]")
        assert opt[0] and "press 5" in opt[1]
        drv.execute_script("document.querySelectorAll('dialog[open]').forEach((d) => d.close())")
    finally:
        drv.execute_script("window.__mace.settings.set('projection', 'orthographic')")


def test_side_by_side_comparison_has_two_groups(fresh):
    drv = fresh[0]
    U.open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out", wait_atoms=False)
    H.wait(drv, "return !!(window.MACE.state.traj && window.MACE.state.cmp)", 30)
    drv.execute_script("window.MACE.setLook('side')")
    time.sleep(0.8)
    drv.execute_script("window.__mace.exportImage()")
    H.wait(drv, "return !!window.__svg", 10)
    root = ET.fromstring(drv.execute_script("return window.__svg"))
    groups = root.findall(f"{SVG}g")
    assert [g.get("id") for g in groups] == ["initial", "final"]
    n = drv.execute_script("return window.MACE.state.traj.frames[0].numbers.length")
    assert all(len(g.findall(f"{SVG}circle")) == n for g in groups)
    drv.execute_script("window.MACE.setLook('overlay')")


def test_8000_atoms_in_time_without_long_blocks(fresh):
    drv = fresh[0]
    drv.execute_script("window.__mace.settings.set('showBoundary', false)")
    try:
        U.open_doc(drv, "big/MgO_1000.cif")
        H.wait(drv, "return window.__mace.viewState().displayedAtoms === 8000", 60)
        time.sleep(1.0)
        r = drv.execute_async_script("""const done = arguments[arguments.length - 1];
          let last = performance.now(), worst = 0, run = true;
          const tick = () => { const t = performance.now(); worst = Math.max(worst, t - last); last = t; if (run) setTimeout(tick, 5); };
          setTimeout(tick, 5);
          const t0 = performance.now();
          window.__mace.exportSvg({}).then((n) => { const ms = performance.now() - t0; run = false; done({ n, ms, worst, len: window.__svg.length }); },
            (e) => { run = false; done({ error: String(e) }); });""")
        assert "error" not in r, r
        print("svg_export_8000", r["ms"], "ms, worst block", r["worst"], "ms,", r["len"], "bytes")
        assert r["n"] == 8000 and r["ms"] <= 1500 and r["worst"] <= 200, r
    finally:
        drv.execute_script("window.__mace.settings.set('showBoundary', true)")
