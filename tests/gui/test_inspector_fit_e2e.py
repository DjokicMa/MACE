"""The Inspector's header and the transport Quantity control at the default
Inspector width (1.1.6 review), in Firefox and Chromium at 100 % and 110 %
browser zoom:

* every Inspector has a header bar (its title and the hide ×), one section or
  many; the × sits in that bar, never over a section head or a control,
  scrolled or not (it used to float, sticky, over "Plot options", "Selected
  material" and the rows below them);
* Expand All / Collapse All are shown only with two or more sections, and
  View › Inspector Sections says why they are disabled with one;
* a BOLTZTRA run's Quantity buttons (Seebeck, σ, σS, κ₀, TDF and ZT) are all
  inside the Inspector, clickable where they are drawn, with no sideways
  scrolling (ZT and TDF lay past its right edge);
* the κ_L CSV of the ZT view is found beside the run first (the run in a
  subfolder), then in the served folder.
Browser zoom: Firefox's layout.css.devPixelsPerPx, set at start (one browser
per zoom); Chromium's device scale factor. The transport cases use the
corpus (test/, read only, copied; skipped without it)."""
import shutil
import time

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")

W, HT = 1366, 820
TRANSPORT = H.REPO_ROOT / "test" / "TRANSPORT"
STEM = "C1-RCSR-afi_optimized_TRANSPORT"
BAND = H.REPO_ROOT / "test" / "BAND"
CASES = [(b, z) for b in ("firefox", "chromium") for z in (1.0, 1.1)]


@pytest.fixture(scope="module", params=CASES, ids=[f"{b}-{round(z * 100)}pct" for b, z in CASES])
def app(request, tmp_path_factory):
    name, zoom = request.param
    tmp = tmp_path_factory.mktemp(f"inspfit_{name}_{round(zoom * 100)}")
    root = H.examples_copy(tmp)
    has_transport = TRANSPORT.is_dir()
    if has_transport:
        (root / "c").mkdir()
        for q in (".out", ".SEEBECK.dat", ".SIGMA.dat", ".SIGMAS.dat", ".KAPPA.dat", ".TDF.dat"):
            shutil.copyfile(TRANSPORT / (STEM + q), root / "c" / ("run" + q))
        (root / "c" / "kappa.csv").write_text("T,kappa_L\n100,3.0\n2000,1.0\n")
    band = sorted(BAND.glob("*.BAND.DAT"))[:1] if BAND.is_dir() else []
    if band:
        shutil.copyfile(band[0], root / "run.BAND.DAT")
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1")
    drv = None
    try:
        if name == "chromium":
            drv = H.browser(name, width=W, height=HT)
        else:
            drv = H.browser(name, width=W, height=HT, prefs={"layout.css.devPixelsPerPx": str(zoom)} if zoom != 1.0 else None)
        drv.set_script_timeout(120)
        w, h = round(W / zoom), round(HT / zoom)
        if name == "chromium":
            drv.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", {"width": w, "height": h, "deviceScaleFactor": zoom, "mobile": False})
        else:
            H.set_viewport(drv, w, h)
        H.load(drv, base)
        H.wait(drv, "return Math.abs(innerWidth - arguments[0]) <= 1", 10, w)
        yield drv, root, has_transport, bool(band)
    finally:
        if drv is not None:
            drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def open_path(drv, path):
    drv.execute_script("window.__mace.closeAll()")
    kind = drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then((x) => d(x && x.kind), () => d(null))", path)
    time.sleep(0.8)
    return kind


# the × and what is drawn under it: every point of its box must hit the × itself
X_PROBE = """
const insp = document.getElementById('inspector'), x = insp.querySelector('.panel-hide[data-panel=inspector]');
if (!x) return {missing: true};
const bar = x.closest('.insp-bar, .su-head');
const r = x.getBoundingClientRect(), bad = [];
for (const fx of [0.15, 0.5, 0.85]) for (const fy of [0.15, 0.5, 0.85]) {
  const e = document.elementFromPoint(r.left + fx * r.width, r.top + fy * r.height);
  if (!e || !x.contains(e)) bad.push(e ? (e.className || e.tagName) + '' : 'nothing');
}
// no section head or control shares its box
const hits = [...insp.querySelectorAll('.insp-head, .insp-body input, .insp-body select, .insp-body button, .insp-body label, .su-head-acts button')]
  .filter((el) => { const q = el.getBoundingClientRect(); return q.width && q.height && q.left < r.right && q.right > r.left && q.top < r.bottom && q.bottom > r.top; })
  .map((el) => el.className || el.tagName);
const br = bar && bar.getBoundingClientRect();
return {bar: bar ? bar.className : null, title: bar ? (bar.querySelector('.insp-bar-title, .su-title') || {}).textContent : null,
        inBar: !!br && r.left >= br.left - 0.5 && r.right <= br.right + 0.5 && r.top >= br.top - 0.5 && r.bottom <= br.bottom + 0.5,
        bad, hits, sideways: insp.scrollWidth > insp.clientWidth + 1};
"""


def probe(drv):
    return drv.execute_script(X_PROBE)


def check_x(drv, what):
    # unscrolled first: nothing shares the ×'s box (scrolled, a section head passes under the opaque
    # bar, which the loop below checks; a caller may have scrolled to its control, and the
    # panel-context sections, Structure and Cell from the run's .out, make a plot's Inspector taller)
    drv.execute_script("document.getElementById('inspector').scrollTop = 0")
    time.sleep(0.15)
    p = probe(drv)
    assert not p.get("missing"), what
    assert p["bar"] and p["inBar"], (what, p)
    assert not p["bad"] and not p["hits"], (what, p)
    # scrolled half way and to the end: the bar stays on top (what scrolls goes under the whole opaque
    # bar, not under a floating ×), and every point of the × is still the ×
    for frac in (0.5, 1.0):
        drv.execute_script("const i = document.getElementById('inspector'); i.scrollTop = (i.scrollHeight - i.clientHeight) * arguments[0]", frac)
        time.sleep(0.15)
        p = probe(drv)
        assert p["inBar"] and not p["bad"], (what, frac, p)
        assert drv.execute_script("const i = document.getElementById('inspector'), b = i.querySelector('.panel-hide').closest('.insp-bar, .su-head');"
                                  " return Math.abs(b.getBoundingClientRect().top - i.getBoundingClientRect().top) <= 1.5"), (what, frac)
    drv.execute_script("document.getElementById('inspector').scrollTop = 0")
    return p


def test_every_inspector_has_a_header_bar_and_the_x_covers_nothing(app):
    drv, root, has_transport, has_band = app
    assert open_path(drv, "structures/MgO.cif") == "structure"
    H.wait(drv, "return document.querySelectorAll('#inspector .insp-section').length >= 4", 10)
    p = check_x(drv, "structure")
    assert p["title"] == "Inspector"
    assert drv.execute_script("return document.querySelectorAll('#inspector .insp-bar [data-all]').length") == 2
    # one section: the database's Selected material
    drv.execute_script("window.__mace.closeAll(); window.__mace.run('view.database')")
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.kind === 'db'", 20)
    time.sleep(0.4)
    assert drv.execute_script("return document.querySelectorAll('#inspector .insp-section').length") == 1
    check_x(drv, "database")
    assert drv.execute_script("return document.querySelectorAll('#inspector .insp-bar [data-all]').length") == 0
    why = drv.execute_script("return window.__mace.menuReason('view.expandSections')")
    assert why and "one section" in why, why
    assert drv.execute_script("return window.__mace.menuReason('view.collapseSections')") == why
    # nothing open
    drv.execute_script("window.__mace.closeAll()")
    time.sleep(0.4)
    check_x(drv, "empty")
    # Calculation Setup: the × in its own header
    assert open_path(drv, "structures/Si.cif") == "structure"
    drv.execute_script("window.__mace.run('calc.setup')")
    H.wait(drv, "return !!document.querySelector('#inspector .su-head .panel-hide')", 30)
    H.wait(drv, "return document.querySelectorAll('#inspector .su-sec').length >= 3", 30)
    check_x(drv, "setup")
    assert drv.execute_script("return window.__mace.menuReason('view.expandSections')") == ""
    drv.execute_script("window.__mace.run('calc.setup')")
    time.sleep(0.4)
    if has_band:
        assert open_path(drv, "run.BAND.DAT") == "plot"
        H.wait(drv, "const s = window.__mace.plotView && window.__mace.plotView.state(); return s && s.drawn", 60)
        check_x(drv, "band plot options")
    # hiding still works from the bar's ×, and the edge tab brings the Inspector back
    drv.execute_script("document.querySelector('#inspector .panel-hide').click()")
    H.wait(drv, "return document.getElementById('shell').classList.contains('no-inspector')", 5)
    drv.execute_script("document.querySelector('.edge-tab[data-panel=inspector]').click()")
    H.wait(drv, "return !document.getElementById('shell').classList.contains('no-inspector')", 5)
    assert not H.errors(drv), H.errors(drv)


QUANTITY = """
const insp = document.getElementById('inspector'), ir = insp.getBoundingClientRect();
return [...insp.querySelectorAll('.po-tq button')].map((b) => { const r = b.getBoundingClientRect();
  const e = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
  return {label: b.textContent, inside: r.left >= ir.left - 0.5 && r.right <= ir.right + 0.5 && r.width > 0, hit: !!e && b.contains(e)}; });
"""


def test_transport_quantities_fit_and_the_csv_beside_the_run(app):
    drv, root, has_transport, _ = app
    if not has_transport:
        pytest.skip("test/ data corpus not present (gitignored)")
    assert open_path(drv, "c/run.SEEBECK.dat") == "plot"
    H.wait(drv, "const s = window.__mace.plotView && window.__mace.plotView.state(); return s && s.kind === 'transport' && s.drawn", 60)
    drv.execute_script("const r = document.querySelector('#inspector .po-tq'); const i = document.getElementById('inspector');"
                       " i.scrollTop += r.getBoundingClientRect().top - i.getBoundingClientRect().top - 60")
    time.sleep(0.2)
    rows = drv.execute_script(QUANTITY)
    assert [r["label"] for r in rows] == ["Seebeck", "σ", "σS", "κ₀", "TDF", "ZT"]
    assert all(r["inside"] and r["hit"] for r in rows), rows
    assert not drv.execute_script("const i = document.getElementById('inspector'); return i.scrollWidth > i.clientWidth + 1")
    check_x(drv, "transport plot options")
    # ZT with a κ_L CSV named as it sits beside the run (c/kappa.csv)
    drv.execute_script("document.querySelector(\"#inspector .po-tq [data-tview='zt']\").click()")
    H.wait(drv, "const s = window.__mace.plotView.state(); return s.drawn && s.zt && s.zt.zt_e_max", 60)
    drv.execute_script("window.__mace.plotView.setOpt({kl: 'kappa.csv'}, 'κ_L')")
    H.wait(drv, "const s = window.__mace.plotView.state(); return s.drawn && s.zt && (s.zt.zt_total_max || s.zt.error)", 60)
    z = drv.execute_script("return window.__mace.plotView.state().zt")
    assert not z["error"] and z["zt_total_max"], z
    assert "kappa.csv" in (z["kappa_l_source"] or ""), z
    hint = drv.execute_script("return document.querySelector('#po-kl').getAttribute('placeholder') + ' ' + document.querySelector('#po-kl').title")
    assert "beside the run" in hint, hint
    assert not H.errors(drv), H.errors(drv)
