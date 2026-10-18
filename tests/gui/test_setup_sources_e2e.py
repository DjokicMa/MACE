"""Setup from a finished .out or a .d12, then Submit or Batch, with real clicks (1.1.6).

Through the real `mace gui` with the fake SLURM on its PATH, in Firefox and
Chromium at 100 % and 110 % browser zoom:

* .out -> Setup -> Submit: the Explorer's Set Up Calculation on a finished
  run; Setup names the parent (.out) its values come from; the sidebar's
  Submit… writes the deck (Save with Write decks now) and opens the Submit
  dialog on that deck; nothing is submitted until the dialog's own Submit.
* .d12 -> Setup -> Batch: the Explorer's Set Up Calculation on a deck (mace
  opt2d12 --d12-file); the sidebar's Batch Prepare + Submit… opens Batch
  Setup on the deck's folder with the decks filter, the calculation and the
  deck as the representative; written with --d12-file, then submitted.

Screenshots at 110 % go to MACE_SETUPSRC_SHOTS (or MACE_GUI_SHOTS) when set.
Opt-in browser test (marker gui)."""
import os
import shutil
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U
import submit_util as SU

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402

DATA = H.REPO_ROOT / "tests" / "data"
BROWSERS = [b for b in os.environ.get("MACE_SETUPSRC_BROWSERS", "firefox,chromium").split(",") if b]
ZOOMS = (1.0, 1.1)
CASES = [(b, z) for b in BROWSERS for z in ZOOMS]
W, HT = 1366, 768


class Session:
    """One server (fake SLURM); a browser per (browser, zoom), as test_menus_zoom_e2e.py does it."""

    def __init__(self, tmp):
        self.tmp = tmp
        self.root = H.examples_copy(tmp)
        self.fake = SU.FakeSlurm(tmp / "fs", {"sacct": {}})
        env = SU.fake_env(self.fake, tmp)
        env.pop("XDG_CACHE_HOME")
        self.proc, self.base, _ = H.start_server(self.root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1", **env)
        self.drv, self.key = None, None

    def at(self, browser, zoom):
        if self.key != (browser, zoom):
            if self.drv is not None:
                self.drv.quit()
                self.drv = None
            chromium = browser.startswith("chrom")
            kw = {} if chromium else {"prefs": {"layout.css.devPixelsPerPx": str(zoom)}}
            self.drv = H.browser(browser, width=W, height=HT, **kw)
            self.drv.set_script_timeout(300)
            H.load(self.drv, self.base)
            self.key = (browser, zoom)
            w, h = round(W / zoom), round(HT / zoom)
            if chromium:
                self.drv.execute_cdp_cmd("Emulation.setDeviceMetricsOverride",
                                         {"width": w, "height": h, "deviceScaleFactor": zoom, "mobile": False})
            else:
                H.set_viewport(self.drv, w, h)
            H.wait(self.drv, "return Math.abs(innerWidth - arguments[0]) <= 1 && Math.abs(innerHeight - arguments[1]) <= 1",
                   10, w, h)
        return self.drv

    def close(self):
        try:
            if self.drv is not None:
                self.drv.quit()
        finally:
            self.proc.terminate()
            self.proc.wait(timeout=15)


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    s = Session(tmp_path_factory.mktemp("setupsrc"))
    try:
        yield s
    finally:
        s.close()


def shot(drv, zoom, name):
    if abs(zoom - 1.1) > 1e-6:
        return
    target = os.environ.get("MACE_SETUPSRC_SHOTS") or os.environ.get("MACE_GUI_SHOTS")
    if target:
        Path(target).mkdir(parents=True, exist_ok=True)
        drv.save_screenshot(str(Path(target) / f"{name}.png"))


def case_folder(s, browser, zoom, flow):
    """A folder of its own for each case (so each one writes its decks fresh)."""
    name = f"{flow}_{browser[:3]}_{round(zoom * 100)}"
    d = s.root / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir()
    if flow == "out":
        for ext in (".out", ".d12", ".f9"):
            src = H.EXAMPLES / "outputs" / f"Si_sp{ext}"
            if src.is_file():
                shutil.copy2(src, d / src.name)
    else:
        shutil.copy2(H.EXAMPLES / "outputs" / "Si_sp.d12", d / "Si_sp.d12")
        shutil.copy2(DATA / "low_dim_groups" / "polyyne_rg28.d12", d / "polyyne.d12")
    return name


def explorer_row(drv, path):
    """The Explorer's row for path (the folder's change reaches the page by itself: files-changed)."""
    H.wait(drv, f"return !!document.querySelector('#file-list .file[data-path=\"{path}\"]')", 60)
    return drv.find_element("css selector", f'#file-list .file[data-path="{path}"]')


def set_up_from_explorer(drv, path):
    el = explorer_row(drv, path)
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
    ActionChains(drv).context_click(el).perform()
    H.wait(drv, "return !!document.querySelector('.menu[aria-label=\"Explorer menu\"]')", 10)
    drv.execute_script("[...document.querySelectorAll('.menu .menu-item')].find((b) => b.querySelector('.lbl').textContent === 'Set Up Calculation').click()")
    H.wait(drv, "return !!document.querySelector('#inspector .setup')", 30)
    wait_ready(drv)


READY = """const m = window.__mace.setup, a = window.__mace.activeDoc(); if (!m || !a) return false;
    const subj = m.subject(); if (!subj || subj.id !== (a.setupFor || a.id)) return false;
    const s = m.state(); return s && s.result && !s.busy && !s.pendingRun"""


def wait_ready(drv, timeout=180):
    H.wait(drv, READY, timeout)
    time.sleep(0.3)


def text(drv, css):
    return drv.execute_script("const e = document.querySelector(arguments[0]); return e ? e.textContent : null", css)


def dialog(drv, timeout=90):
    H.wait(drv, "return !!document.querySelector('#submit-dialog[open]')", 60)
    H.wait(drv, "const d = document.querySelector('#submit-dialog'); return d && d.querySelector('[data-part=command]') "
                "&& !d.querySelector('[data-part=previewing]')", timeout)
    H.wait(drv, "return !document.querySelector('#submit-dialog [data-act=submit]').disabled", 60)


def sbatch(fake):
    return [c for c in fake.calls("sbatch") if "--test-only" not in c["argv"]]


def fits(drv, css):
    """Every matching control shows its whole label (nothing cut short) inside the window."""
    return drv.execute_script("""return [...document.querySelectorAll(arguments[0])].map((b) => {
        const r = b.getBoundingClientRect();
        return [b.textContent.trim(), b.scrollWidth <= b.clientWidth + 1, r.right <= innerWidth + 1 && r.left >= -1 && r.width > 0];
      })""", css)


@pytest.mark.parametrize("browser,zoom", CASES, ids=[f"{b}-{round(z * 100)}pct" for b, z in CASES])
def test_out_setup_submit(session, browser, zoom):
    drv = session.at(browser, zoom)
    U.reset(drv, round(W / zoom), round(HT / zoom))
    session.fake.log_file.write_text("")
    folder = case_folder(session, browser, zoom, "out")
    set_up_from_explorer(drv, f"{folder}/Si_sp.out")
    st = drv.execute_script("const s = window.__mace.setup.state(); return {mode: s.mode, calc: s.calc}")
    assert st == {"mode": "opt2d12", "calc": "SP"}
    # where the values come from: the parent .out, named in step 1 and on each field
    assert "Si_sp.out" in text(drv, "#inspector .su-parent")
    badge = text(drv, "#inspector [data-field=functional] .su-badge")
    assert badge == "from parent .out", badge
    # the sidebar's two buttons, whole and inside the window
    btns = fits(drv, '#inspector .su-foot [data-act="submitNow"], #inspector .su-foot [data-act="batchNow"]')
    assert [b[0] for b in btns] == ["Submit…", "Batch Prepare + Submit…"], btns
    assert all(b[1] and b[2] for b in btns), btns
    shot(drv, zoom, f"out_setup_{browser}")
    drv.find_element("css selector", '#inspector .su-foot [data-act="submitNow"]').click()
    dialog(drv)
    cmd = drv.execute_script("return document.querySelector('#submit-dialog [data-part=command]').textContent")
    deck = "Si_sp_sp_PBE0_optimized.d12"
    assert cmd == f"cd {folder} && mace submit {deck}", cmd
    assert (session.root / folder / deck).is_file()
    # the deck Setup wrote is the one it previewed
    prev = drv.execute_script("return window.__mace.setup.state().result.deck")
    assert (session.root / folder / deck).read_text() == prev
    assert sbatch(session.fake) == []                      # nothing until the dialog's own Submit
    shot(drv, zoom, f"out_submit_dialog_{browser}")
    drv.find_element("css selector", '#submit-dialog [data-act="submit"]').click()
    H.wait(drv, "return !document.querySelector('#submit-dialog[open]')", 60)
    H.wait(drv, "return [...document.querySelectorAll('#toasts > *')].some((t) => t.textContent.includes('Submitted'))", 30)
    sb = sbatch(session.fake)
    assert len(sb) == 1 and deck.replace(".d12", "") in sb[0]["script_text"], sb
    U.no_errors(drv)


@pytest.mark.parametrize("browser,zoom", CASES, ids=[f"{b}-{round(z * 100)}pct" for b, z in CASES])
def test_d12_setup_batch(session, browser, zoom):
    drv = session.at(browser, zoom)
    U.reset(drv, round(W / zoom), round(HT / zoom))
    session.fake.log_file.write_text("")
    folder = case_folder(session, browser, zoom, "d12")
    set_up_from_explorer(drv, f"{folder}/Si_sp.d12")
    st = drv.execute_script("const s = window.__mace.setup.state(); return {mode: s.mode, calc: s.calc, cmd: s.result.command}")
    assert st["mode"] == "opt2d12" and st["calc"] == "SP"
    assert "--d12-file" in st["cmd"], st
    # a deck has no wavefunction: SP, OPT, FREQ only
    rows = drv.execute_script("return [...document.querySelectorAll('#inspector .su-calc [data-calc]')].map((b) => b.dataset.calc)")
    assert rows == ["SP", "OPT", "FREQ"], rows
    assert "Si_sp.d12" in text(drv, "#inspector .su-parent")
    assert text(drv, "#inspector [data-field=functional] .su-badge") == "from parent .d12"
    drv.find_element("css selector", "#inspector .su-calc [data-calc='OPT']").click()
    H.wait(drv, "const s = window.__mace.setup.state(); return s.calc === 'OPT' && s.result && !s.busy && !s.pendingRun", 180)
    shot(drv, zoom, f"d12_setup_{browser}")
    drv.find_element("css selector", '#inspector .su-foot [data-act="batchNow"]').click()
    H.wait(drv, "return window.__mace.doc() && window.__mace.doc().kind === 'batch'", 20)
    assert drv.execute_script("return window.__mace.doc().title") == f"Batch: {folder}"
    H.wait(drv, "return document.querySelectorAll('#batch-host [data-part=rows] tbody tr').length === 2", 60)
    assert drv.execute_script("return document.querySelector('#batch-host input[name=bat-filter]:checked').value") == "d12"
    assert drv.execute_script("return document.querySelector('#batch-host [data-b=calc]').value") == "OPT"
    drv.find_element("css selector", '#batch-host [data-go="settings"]').click()
    H.wait(drv, "return !!document.querySelector('#batch-host [data-part=fields]')", 180)
    assert drv.execute_script("return document.querySelector('#batch-host [data-b=rep]').value") == f"{folder}/Si_sp.d12"
    drv.find_element("css selector", '#batch-host [data-go="preview"]').click()
    H.wait(drv, "return document.querySelectorAll('#batch-host [data-part=plan] tbody tr').length === 2", 300)
    shot(drv, zoom, f"d12_batch_preview_{browser}")
    drv.find_element("css selector", '#batch-host [data-go="write"]').click()
    name = drv.find_element("css selector", '#batch-host [data-b="name"]')
    name.clear()
    name.send_keys("b_e2e")
    drv.find_element("css selector", '#batch-host [data-act="write"]').click()
    H.wait(drv, "return !!document.querySelector('#batch-host [data-part=commands]')", 300)
    cmds = drv.execute_script("return document.querySelector('#batch-host [data-part=commands]').textContent")
    assert cmds.count("mace opt2d12 --d12-file") == 2 and "--out-file" not in cmds, cmds
    decks = sorted(p.name for p in (session.root / folder / "b_e2e" / "decks").glob("*.d12"))
    assert len(decks) == 2 and all("_opt_" in n and "_optimized" not in n for n in decks), decks
    assert sbatch(session.fake) == []
    drv.find_element("css selector", '#batch-host [data-act="submit"]').click()
    dialog(drv)
    assert drv.execute_script("return document.querySelector('#submit-dialog [data-part=command]').textContent") == \
        f"cd {folder} && mace submit --array b_e2e/decks"
    shot(drv, zoom, f"d12_batch_submit_dialog_{browser}")
    drv.find_element("css selector", '#submit-dialog [data-act="submit"]').click()
    H.wait(drv, "return !document.querySelector('#submit-dialog[open]')", 60)
    H.wait(drv, "return [...document.querySelectorAll('#toasts > *')].some((t) => t.textContent.includes('Submitted'))", 30)
    assert len(sbatch(session.fake)) == 1
    U.no_errors(drv)


def test_unfinished_out_shows_the_clis_warning_and_network_mode_refuses_submit(session):
    drv = session.at(BROWSERS[0], 1.0)
    U.reset(drv, W, HT)
    session.fake.log_file.write_text("")
    set_up_from_explorer(drv, "outputs/Ag2Br3_opt_unfinished.out")
    warn = text(drv, "#inspector .su-parent-warn")
    assert warn and "unfinished optimization" in warn and "starting geometry" in warn, warn
    # Submit is turned off while the GUI is open to the network: refused before anything is written
    drv.execute_script("""const f = window.fetch; window.__fetchBefore = f;
        window.fetch = (u, o) => String(u).includes('submit/capabilities')
          ? Promise.resolve(new Response(JSON.stringify({network: true, sbatch: true}), {status: 200, headers: {'Content-Type': 'application/json'}}))
          : f(u, o);""")
    before = sorted(p.name for p in (session.root / "outputs").iterdir())
    try:
        drv.find_element("css selector", '#inspector .su-foot [data-act="submitNow"]').click()
        H.wait(drv, "return [...document.querySelectorAll('#toasts > *')].some((t) => t.textContent.includes('open to the network'))", 20)
    finally:
        drv.execute_script("window.fetch = window.__fetchBefore")
    assert sorted(p.name for p in (session.root / "outputs").iterdir()) == before
    assert not drv.execute_script("return !!document.querySelector('#submit-dialog[open]')")
    assert sbatch(session.fake) == []


def test_a_low_dimensional_deck_says_to_use_its_run_as_the_viewer_does(session):
    """The 1.1.5 viewer draws no SLAB, POLYMER or MOLECULE deck (it names the run's .out instead), so
    Set Up Calculation on one says the same; Batch Setup takes such decks (test_setup_sources_api.py)."""
    drv = session.at(BROWSERS[0], 1.0)
    U.reset(drv, W, HT)
    el = explorer_row(drv, "outputs/graphene_slab_sp.d12")
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
    ActionChains(drv).context_click(el).perform()
    H.wait(drv, "return !!document.querySelector('.menu[aria-label=\"Explorer menu\"]')", 10)
    drv.execute_script("[...document.querySelectorAll('.menu .menu-item')].find((b) => b.querySelector('.lbl').textContent === 'Set Up Calculation').click()")
    H.wait(drv, "return [...document.querySelectorAll('#toasts > *, .notice, [role=alert]')].some((t) => /SLAB deck/.test(t.textContent))", 30)
    msg = drv.execute_script("return [...document.querySelectorAll('#toasts > *, .notice, [role=alert]')].map((t) => t.textContent).find((t) => /SLAB deck/.test(t))")
    assert "graphene_slab_sp.out" in msg, msg
