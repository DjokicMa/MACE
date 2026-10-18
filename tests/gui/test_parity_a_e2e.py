"""1.1.6 parity A in the browser (MACE_GUI_BROWSER=firefox or chromium), at
100 % and 110 % browser zoom:

* Calculation Setup › Charged system: the CLI's charged-system switches,
  given with real clicks, reach the printed command and the deck; a meaning
  the CLI refuses shows its own words in Problems.
* Settings › Advanced: cluster defaults: `mace site` as it prints, and
  Set / Unset / Create site.toml run `mace site set|unset|--init`.
* Submit: the runtime estimate (`mace cluster estimate`) is shown and never
  holds Submit back.

Each new surface passes axe (WCAG 2 A/AA, no serious or critical finding)
and fits its box at both zoom levels. The Setup case needs the test/ corpus
(a real molecule run); the other two run without it."""
import os
import shutil
import time

import pytest

import guiharness as H
import p4util as U
import submit_util as SU

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
from test_a11y_e2e import AXE_OPTIONS, KNOWN_OLD_AXE  # noqa: E402

MOL = H.REPO_ROOT / "test" / "OPT" / "1LiFSI-1DEC-conf3_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized"
W, HGT = 1366, 800


@pytest.fixture(scope="module", params=[1.0, 1.1], ids=["zoom100", "zoom110"])
def app(request, tmp_path_factory):
    zoom = request.param
    tmp = tmp_path_factory.mktemp(f"parityA{int(zoom * 100)}")
    root = H.examples_copy(tmp)
    if MOL.with_suffix(".out").is_file():
        (root / "mol").mkdir()
        for ext in (".out", ".d12"):
            shutil.copyfile(MOL.with_suffix(ext), root / "mol" / f"mol_opt{ext}")
    (root / "n").mkdir()
    (root / "n" / "x.d12").write_text(SU.SP_D12)
    fake = SU.FakeSlurm(tmp / "fs")
    env = SU.fake_env(fake, tmp)
    env.pop("XDG_CACHE_HOME")
    site = tmp / "cfg" / "site.toml"
    # the hooks keep FAKE_SLURM_* in the CLI's environment: without them the fake squeue
    # fails, the queue budget cannot read the queue and the preview holds every job
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1",
                                   MACE_SITE_CONFIG=str(site), **env)
    name = os.environ.get("MACE_GUI_BROWSER", "firefox")
    chromium = name.startswith("chrom")
    drv = None
    try:
        drv = H.browser(name, width=W, height=HGT, **({} if chromium else {"prefs": {"layout.css.devPixelsPerPx": str(zoom)}}))
        drv.set_script_timeout(300)
        H.load(drv, base)
        w, h = round(W / zoom), round(HGT / zoom)
        if chromium:
            drv.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", {"width": w, "height": h, "deviceScaleFactor": zoom, "mobile": False})
        else:
            H.set_viewport(drv, w, h)
        H.wait(drv, "return Math.abs(innerWidth - arguments[0]) <= 1 && Math.abs(innerHeight - arguments[1]) <= 1", 10, w, h)
        time.sleep(0.4)
        yield drv, root, site, zoom
    finally:
        if drv is not None:
            drv.quit()
        proc.terminate()
        proc.wait(timeout=30)


def _close(drv):
    drv.execute_script("document.querySelectorAll('dialog[open]').forEach((d) => d.close())")
    time.sleep(0.2)


FITS = """const box = arguments[0], r = box.getBoundingClientRect(), bad = [];
  if (box.scrollWidth > box.clientWidth + 1) bad.push(`${box.className}: scrolls sideways (${box.scrollWidth} > ${box.clientWidth})`);
  for (const e of box.querySelectorAll('input, select, textarea, button, code, label')) {
    const q = e.getBoundingClientRect();
    if (!q.width) continue;
    if (q.right > r.right + 1 || q.left < r.left - 1) bad.push(`${e.tagName}#${e.id || e.className}: ${q.left}-${q.right} outside ${r.left}-${r.right}`);
  }
  return bad;"""


def _click(drv, css):
    el = drv.find_element("css selector", css)
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
    time.sleep(0.1)
    el.click()
    return el


SETUP_READY = """const m = window.__mace.setup; const s = m && m.state(); return !!(s && s.result && !s.busy && !s.pendingRun)"""


def test_setup_charged_system(app):
    drv, root, _, zoom = app
    if not (root / "mol" / "mol_opt.out").is_file():
        pytest.skip("test/ corpus not present (gitignored)")
    _close(drv)
    U.open_doc(drv, "mol/mol_opt.out", wait_atoms=False)
    drv.execute_script("window.__mace.run('calc.setup', {open: true})")
    H.wait(drv, "return !!document.querySelector('#inspector .setup')", 30)
    H.wait(drv, SETUP_READY, 180)
    sec = drv.find_element("css selector", '#inspector .su-sec[data-sec="Charged system"]')
    if "collapsed" in sec.get_attribute("class"):
        _click(drv, '#inspector .su-sec[data-sec="Charged system"] > .su-sec-h')
    # Find ions (a real click), then CHEMOD yes from the menu
    _click(drv, "#su-cf-ion_charges")
    H.wait(drv, SETUP_READY + " && !!s.flags.ion_charges", 180)
    drv.execute_script("const s = document.querySelector('#su-cf-chemod'); s.value = 'yes'; s.dispatchEvent(new Event('change', {bubbles: true}))")
    H.wait(drv, SETUP_READY + " && s.flags.chemod === 'yes' && /--chemod yes/.test(s.result.command || '')", 180)
    st = drv.execute_script("const s = window.__mace.setup.state(); return {cmd: s.result.command, deck: s.result.deck || ''}")
    assert st["cmd"].endswith("--ion-charges auto --chemod yes"), st["cmd"]
    assert "CHEMOD" in st["deck"].split()
    # the fragment bond scale is offered with Find ions
    assert drv.execute_script("return !document.querySelector('#su-cf-fragment_bond_scale').disabled")
    box = drv.find_element("css selector", '#inspector .su-sec[data-sec="Charged system"]')
    assert drv.execute_script(FITS, box) == []
    found = _axe_in(drv, "#inspector .setup", "setup-charged")
    # site states without a net charge: opt2d12's own refusal, in Problems
    drv.execute_script("""const t = document.querySelector('#su-cf-site_states'); t.value = '[{"atom": 1, "shell_charges": [2, 2, 0, 0, 4, 0, 0]}]';
        t.dispatchEvent(new Event('change', {bubbles: true}))""")
    H.wait(drv, SETUP_READY + " && 'site_states' in s.flags", 180)
    H.wait(drv, """return [...document.querySelectorAll('*')].some((e) => e.children.length === 0 && /give the net charge too/.test(e.textContent))
        || JSON.stringify(window.__mace.setup.state().result || {}).includes('give the net charge too')""", 60)
    # undo puts the switches back as they were
    drv.execute_script("window.__mace.setup.undo()")
    H.wait(drv, SETUP_READY + " && !('site_states' in s.flags)", 180)
    assert found == [], found


def _axe_in(drv, css, label):
    """axe (WCAG 2 A/AA) on the new surface only: serious and critical findings."""
    import json
    axe_mod = pytest.importorskip("axe_selenium_python")
    axe_mod.Axe(drv).inject()
    res = drv.execute_async_script(
        "const done = arguments[arguments.length - 1];"
        f"axe.run({json.dumps({'include': [[css]]})}, {AXE_OPTIONS}).then((r) => done(r), (e) => done({{error: String(e)}}))")
    assert "error" not in res, res
    return [(label, v["id"], v["impact"], [n["target"] for n in v["nodes"]][:5]) for v in res["violations"]
            if v.get("impact") in ("serious", "critical")
            and any((v["id"], (n["target"] or [""])[-1]) not in KNOWN_OLD_AXE for n in v["nodes"])]


def test_settings_cluster_defaults(app):
    drv, root, site, zoom = app
    _close(drv)
    drv.execute_script("window.__mace.run('file.settings')")
    H.wait(drv, "return !!document.querySelector('#settings[open]')", 20)
    _click(drv, '#settings .set-nav-item[data-sec="clusterdefaults"]')
    H.wait(drv, "return !!document.querySelector('#settings .cd-text')", 60)
    text = drv.execute_script("return document.querySelector('#settings .cd-text').textContent")
    assert "built-in" in text and str(site) in drv.execute_script("return document.querySelector('#settings .set-pane').textContent")
    if not site.exists():
        _click(drv, '#settings [data-cd="init"]')
        H.wait(drv, "return /mace site --init/.test(document.querySelector('#settings .cd-result').textContent)", 60)
    assert site.is_file()
    key = drv.find_element("css selector", "#cd-key")
    key.clear()
    key.send_keys("resources.ntasks")
    val = drv.find_element("css selector", "#cd-value")
    val.clear()
    val.send_keys("12")
    _click(drv, '#settings [data-cd="set"]')
    H.wait(drv, "return /mace site set resources.ntasks 12: done/.test(document.querySelector('#settings .cd-result').textContent)", 60)
    assert "ntasks = 12" in site.read_text()
    assert "site.toml" in drv.execute_script("return document.querySelector('#settings .cd-text').textContent")
    # the CLI's own refusal
    key = drv.find_element("css selector", "#cd-key")
    key.clear()
    key.send_keys("bogus.key")
    _click(drv, '#settings [data-cd="set"]')
    H.wait(drv, "return /is not a site setting/.test(document.querySelector('#settings .cd-result').textContent)", 60)
    pane = drv.find_element("css selector", "#settings .set-pane")
    assert drv.execute_script(FITS, drv.find_element("css selector", "#cluster-defaults-host")) == []
    assert drv.execute_script("return arguments[0].scrollWidth <= arguments[0].clientWidth + 1", pane)
    found = _axe_in(drv, "#settings", "settings-cluster-defaults")
    key = drv.find_element("css selector", "#cd-key")
    key.clear()
    key.send_keys("resources.ntasks")
    _click(drv, '#settings [data-cd="unset"]')
    H.wait(drv, "return /mace site unset resources.ntasks: done/.test(document.querySelector('#settings .cd-result').textContent)", 60)
    _close(drv)
    assert found == [], found


def test_submit_shows_the_estimate_and_never_waits_for_it(app):
    drv, root, _, zoom = app
    _close(drv)
    drv.execute_script("window.__mace.run('calc.submit', {paths: ['n/x.d12']})")
    H.wait(drv, "return !!document.querySelector('#submit-dialog[open]')", 30)
    H.wait(drv, "return !!document.querySelector('#submit-dialog [data-part=estimates]') && "
                "!/Estimating/.test(document.querySelector('#submit-dialog [data-part=estimates]').textContent)", 120)
    est = drv.execute_script("return document.querySelector('#submit-dialog [data-part=estimates]').textContent")
    assert "cannot estimate" in est and "Advisory only" in est, est
    # Submit is enabled by the preview alone (n/x.d12 has no history to estimate from)
    H.wait(drv, "return !document.querySelector('#submit-dialog [data-act=submit]').disabled", 60)
    # the CLI's words and the command, one click away
    _click(drv, "#submit-dialog .sub-est > summary")
    assert "mace cluster estimate n/x.d12" in drv.execute_script("return document.querySelector('#submit-dialog .sub-est').textContent")
    assert drv.execute_script(FITS, drv.find_element("css selector", "#submit-dialog [data-part=estimates]")) == []
    found = _axe_in(drv, "#submit-dialog", "submit-estimate")
    _close(drv)
    assert found == [], found


def test_batch_setup_takes_the_switches(app):
    drv, root, _, zoom = app
    _close(drv)
    drv.execute_script("window.__mace.run('calc.batch', {dir: 'structures'})")
    H.wait(drv, "return document.querySelectorAll('#batch-host [data-part=rows] tbody tr').length > 0", 120)
    _click(drv, '#batch-host [data-go="settings"]')
    H.wait(drv, "return !!document.querySelector('#batch-host [data-part=fields]') && !!document.querySelector('#batch-host [data-part=charge]')", 180)
    time.sleep(0.5)
    # no site states in a batch (they name atoms of one structure)
    assert not drv.find_elements("css selector", "#bat-cf-site_states")
    _click(drv, "#bat-cf-ion_charges")
    H.wait(drv, "return /--ion-charges auto/.test(document.querySelector('#batch-host [data-part=charge]').textContent)", 20)
    assert drv.execute_script("return !document.querySelector('#bat-cf-fragment_bond_scale').disabled")
    box = drv.find_element("css selector", "#batch-host [data-part=charge]")
    assert drv.execute_script(FITS, box) == []
    found = _axe_in(drv, "#batch-host .bat-charge", "batch-charged")
    assert found == [], found
