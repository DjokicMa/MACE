"""The GUI smoke subset (CI job gui-smoke, polish CI1): the things a broken
push would show first, on the repo's own example folder only (no test/
corpus), in headless Firefox and, for the browser-generic checks, Chromium,
both with software WebGL. Real clicks; the server started through mace_cli.

- the page boots without console errors or CSP violations, and draws;
- every visible command button runs its command on a real click (welcome
  screen and a structure);
- no atom is clipped on MgO at 3x3x3 (a few poses);
- an OPT output renders and its step slider moves the frame; a slab ignores
  the range along c; the database tab opens;
- Zen Mode's exit chip leaves Zen;
- without WebGL (or when 3Dmol gets no context) the viewport explains itself
  and image export says why.
"""
import time

import pytest

import guiharness as H

pytestmark = [pytest.mark.gui, pytest.mark.guismoke]
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")                 # the server reads the structures with it

from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402

CSP_LISTENER = ("window.__csp = window.__csp || [];"
                "if (!window.__cspOn) { window.__cspOn = true; document.addEventListener('securitypolicyviolation',"
                " (e) => window.__csp.push(e.violatedDirective + ' ' + e.blockedURI)); }")


def start(tmp, browser, **kw):
    root = H.examples_copy(tmp)
    proc, base, _ = H.start_server(root, tmp / "xdg")
    try:
        drv = H.browser(browser, **kw)
    except BaseException:
        proc.terminate()
        proc.wait(timeout=15)
        raise
    return drv, proc, base, root


def stop(drv, proc, name=None, failed=False):
    if failed and name:
        H.save_failure(drv, name)
    drv.quit()
    proc.terminate()
    proc.wait(timeout=15)


def open_doc(drv, path):
    kind = drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then((x) => d(x && x.kind))", path)
    time.sleep(0.6)
    return kind


def canvas_nonblank(drv, vid="viewer-main"):
    return drv.execute_script("""const v = window.MACE.state.viewers[arguments[0]]; if (!v || !v.paintNow()) return 0;
      const cv = v.getCanvas(), url = cv.toDataURL('image/png');
      return new Promise((ok) => { const img = new Image(); img.onload = () => {
        const c = document.createElement('canvas'); c.width = cv.width; c.height = cv.height;
        const g = c.getContext('2d'); g.drawImage(img, 0, 0);
        const d = g.getImageData(0, 0, c.width, c.height).data; let n = 0;
        for (let k = 4; k < d.length; k += 64) if (Math.abs(d[k] - d[0]) + Math.abs(d[k + 1] - d[1]) + Math.abs(d[k + 2] - d[2]) > 30) n++;
        ok(n); }; img.src = url; });""", vid)


@pytest.mark.parametrize("browser", ["firefox", "chromium"])
def test_boot_draw_and_buttons(tmp_path, browser, request):
    drv, proc, base, root = start(tmp_path, browser)
    failed = True
    try:
        H.load(drv, base)
        drv.execute_script(CSP_LISTENER)
        # every visible command button on the welcome screen runs on a real click
        import test_controls_click_e2e as C
        ran = C.sweep(drv)
        for cmd in ("view.palette", "view.themeCycle", "file.settings", "view.explorer"):
            assert cmd in ran, (cmd, ran)
        assert open_doc(drv, "structures/MgO.cif") == "structure"
        H.wait(drv, "return window.__mace.viewState().displayedAtoms > 0", 20)
        assert canvas_nonblank(drv) > 50
        if browser == "firefox":
            ran = C.sweep(drv)
            assert "build.clean" in ran and "calc.setup" in ran, ran
        assert drv.execute_script("return window.__csp") == []
        assert not H.errors(drv), H.errors(drv)
        failed = False
    finally:
        stop(drv, proc, request.node.name, failed)


def test_clip_opt_slab_db_zen(tmp_path, request):
    drv, proc, base, root = start(tmp_path, "firefox")
    failed = True
    try:
        H.load(drv, base)
        assert open_doc(drv, "structures/MgO.cif") == "structure"
        drv.execute_script("window.__mace.run('view.range3')")
        H.wait(drv, "return document.getElementById('tb-range-label').textContent === '3×3×3'", 10)
        time.sleep(0.6)
        import test_viewport_clip_e2e as K
        K.sweep(drv, "viewer-main", "MgO 3x3x3", poses=5)
        # an OPT output: it renders, and the step slider moves the frame
        assert open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out") == "trajectory"
        H.wait(drv, "return !!(window.MACE.state.traj && window.MACE.state.viewers['viewer-before'])", 30)
        n = drv.execute_script("return window.MACE.state.traj.frames.length")
        assert n >= 2
        drv.execute_script("const el = document.getElementById('o-frame'); el.value = '0'; el.dispatchEvent(new Event('input'))")
        time.sleep(0.4)
        drv.execute_script("const el = document.getElementById('o-frame'); el.value = String(arguments[0]); el.dispatchEvent(new Event('input'))", n - 1)
        H.wait(drv, "return window.MACE.state.frame === arguments[0]", 30, n - 1)
        assert canvas_nonblank(drv, "viewer-before") > 50
        # a slab: since 1.1.6 a range above 1 along c repeats it at the Vacuum spacing (the Open mode: not at all)
        assert open_doc(drv, "outputs/graphene_slab_sp.out") in ("structure", "trajectory")
        drv.execute_script("window.__mace.run('view.range1')")
        time.sleep(0.6)
        n1 = drv.execute_script("return window.__mace.viewState().displayedAtoms")
        drv.execute_script("window.__mace.display.set(window.__mace.activeDoc(), {displayLo: [0, 0, 0], displayHi: [1, 1, 3]})")
        time.sleep(0.8)
        assert drv.execute_script("return window.__mace.viewState().displayedAtoms") == 3 * n1
        drv.execute_script("""const s = document.getElementById('opt-vacuum-mode'); s.value = 'open';
          s.dispatchEvent(new Event('change'));""")
        time.sleep(0.8)
        assert drv.execute_script("return window.__mace.viewState().displayedAtoms") == n1
        # the database tab
        drv.execute_script("window.__mace.run('view.database')")
        H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.kind === 'db'", 20)
        # Zen Mode and its exit chip
        drv.execute_script("window.__mace.run('view.zen')")
        H.wait(drv, "return document.getElementById('shell').classList.contains('zen') && !document.getElementById('zen-exit').hidden", 5)
        ActionChains(drv).move_to_element(drv.find_element("css selector", "#zen-exit")).click().perform()
        H.wait(drv, "return !document.getElementById('shell').classList.contains('zen')", 5)
        assert not H.errors(drv), H.errors(drv)
        failed = False
    finally:
        stop(drv, proc, request.node.name, failed)


@pytest.mark.parametrize("browser", ["firefox", "chromium"])
def test_no_webgl_explains_and_keeps_the_rest(tmp_path, browser, request):
    """Polish V3: WebGL off. The viewport says why, image export is disabled
    with the reason, the database and Setup still open, nothing throws."""
    root = H.examples_copy(tmp_path)
    proc, base, _ = H.start_server(root, tmp_path / "xdg")
    drv = (H.firefox(1280, 800, prefs={"webgl.disabled": True}) if browser == "firefox"
           else H.chromium(1280, 800, args=("--disable-webgl", "--disable-3d-apis")))
    failed = True
    try:
        H.load(drv, base)
        assert open_doc(drv, "structures/MgO.cif") == "structure"
        H.wait(drv, "return !!document.querySelector('#viewer-main .no3d')", 10)
        note = drv.find_element("css selector", "#viewer-main .no3d").text
        assert "3D view unavailable" in note and "3D view needs WebGL" in note, note
        assert "Turn on hardware acceleration in your browser settings, then reload." in note, note
        H.wait(drv, "return !!document.getElementById('st-no3d')", 10)     # drawn on the status bar's next refresh
        assert drv.execute_script("return window.__mace.menuReason('file.exportImage')") == "Needs 3D (WebGL)"
        before = drv.execute_script("return performance.now()")
        drv.execute_script("window.__mace.run('file.exportImage')")
        time.sleep(0.3)
        assert not drv.execute_script("return !!document.querySelector('dialog[open]')")
        toasts = drv.execute_script("return [...document.querySelectorAll('#toasts > *')].map((t) => t.textContent).join(' ')")
        assert "Needs 3D (WebGL)" in toasts
        # a FREQ-free example: the OPT output's viewport explains too, and the rest works
        assert open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out") == "trajectory"
        H.wait(drv, "return !!document.querySelector('#viewer-before .no3d')", 10)
        drv.execute_script("window.__mace.run('view.database')")
        H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.kind === 'db'", 20)
        assert open_doc(drv, "structures/Si.cif") == "structure"
        drv.execute_script("window.__mace.run('calc.setup')")
        H.wait(drv, "return !!document.querySelector('#inspector .setup')", 20)
        assert not H.errors(drv), H.errors(drv)
        failed = False
    finally:
        stop(drv, proc, request.node.name, failed)


@pytest.mark.parametrize("browser", ["firefox", "chromium"])
def test_viewer_without_a_context_explains(tmp_path, browser, request):
    """WebGL passes the probe, yet 3Dmol gets no context (a driver that refuses
    its attributes, a context limit): the viewport explains instead of staying
    blank, 3D commands say why, and the rest keeps working."""
    drv, proc, base, _ = start(tmp_path, browser)
    failed = True
    try:
        H.load(drv, base)
        assert drv.execute_script("return !window.MACE.state.viewers['viewer-main']")
        drv.execute_async_script("const d = arguments[0]; window.MACE.load3Dmol().then(() => d(true))")
        # 3Dmol's namespace is read-only: a copy whose createViewer fails as 3Dmol's does
        drv.execute_script("window.$3Dmol = Object.assign({}, window.$3Dmol,"
                           " {createViewer: () => { throw new Error('Error creating WebGL context.'); }})")
        assert open_doc(drv, "structures/MgO.cif") == "structure"
        H.wait(drv, "return !!document.querySelector('#viewer-main .no3d')", 10)
        assert "3D view needs WebGL" in drv.find_element("css selector", "#viewer-main .no3d").text
        assert drv.execute_script("return window.MACE.webgl.ok") is False
        assert drv.execute_script("return window.__mace.menuReason('file.exportImage')") == "Needs 3D (WebGL)"
        drv.execute_script("window.__mace.run('view.database')")
        H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.kind === 'db'", 20)
        assert not H.errors(drv), H.errors(drv)
        failed = False
    finally:
        stop(drv, proc, request.node.name, failed)


@pytest.mark.parametrize("browser", ["firefox", "chromium"])
def test_split_editor_both_groups_draw(tmp_path, browser, request):
    """Split Editor Right (plan E11): the right group's page loads in its frame
    (same-origin framing only, no CSP violation) and both groups draw."""
    drv, proc, base, root = start(tmp_path, browser, width=1400, height=850)
    failed = True
    try:
        H.load(drv, base)
        drv.execute_script(CSP_LISTENER)
        assert open_doc(drv, "structures/MgO.cif") == "structure"
        assert open_doc(drv, "structures/Si.cif") == "structure"
        drv.find_element("css selector", "#views").click()
        drv.execute_script("window.__mace.run('view.splitRight')")
        H.wait(drv, """const f = document.querySelector('.group-frame'); return f && f.contentWindow.__macePane &&
            f.contentWindow.__macePane.docs().some((d) => d.path === 'structures/Si.cif')""", 60)
        assert canvas_nonblank(drv) > 50                     # the left group: MgO
        n = drv.execute_async_script("""const done = arguments[0], f = document.querySelector('.group-frame');
            const W = f.contentWindow; const v = W.MACE.state.viewers['viewer-main']; if (!v || !v.paintNow()) { done(0); return; }
            const cv = v.getCanvas(), img = new W.Image(); img.onload = () => {
              const c = W.document.createElement('canvas'); c.width = cv.width; c.height = cv.height;
              const g = c.getContext('2d'); g.drawImage(img, 0, 0);
              const d = g.getImageData(0, 0, c.width, c.height).data; let n = 0;
              for (let k = 4; k < d.length; k += 64) if (Math.abs(d[k] - d[0]) + Math.abs(d[k + 1] - d[1]) + Math.abs(d[k + 2] - d[2]) > 30) n++;
              done(n); }; img.src = cv.toDataURL('image/png');""")
        assert n > 50                                         # the right group: Si
        assert drv.execute_script("return window.__csp") == []
        assert not H.errors(drv), H.errors(drv)
        failed = False
    finally:
        stop(drv, proc, request.node.name, failed)


def test_slab_deck_opens_its_runs_output(tmp_path, request):
    """The example folder's graphene_slab_sp.d12 (a SLAB deck, which the
    viewer does not draw) opens graphene_slab_sp.out on the first click,
    with a note, instead of an error."""
    drv, proc, base, root = start(tmp_path, "firefox")
    failed = True
    try:
        H.load(drv, base)
        got = drv.execute_async_script(
            "const d = arguments[1]; window.__mace.open(arguments[0]).then((x) => d(x ? [x.kind, x.path] : null))",
            "outputs/graphene_slab_sp.d12")
        time.sleep(0.6)
        assert got and got[1] == "outputs/graphene_slab_sp.out" and got[0] in ("structure", "trajectory"), got
        toasts = drv.execute_script("return [...document.querySelectorAll('#toasts > *')].map((t) => t.textContent)")
        assert any("opening it instead" in t for t in toasts), toasts
        assert not any("reads 3D (CRYSTAL) decks" in t for t in toasts), toasts
        failed = False
    finally:
        stop(drv, proc, request.node.name, failed)
