"""What 1.1.5 adds to the page, in headless Firefox through the real `mace gui`:
the logo (favicon, menu bar, About, welcome), the window titles, Help ▸
Documentation, and a page that runs under the CSP without 'unsafe-eval'
(a structure, an optimization and a plot drawn with no violation)."""
import sys
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
sys.path.insert(0, str(H.REPO_ROOT / "tests"))

LONG = "a_rather_long_project_folder/with_several/levels_of_nesting/runs"

CSP_LISTENER = ("window.__csp = window.__csp || [];"
                "if (!window.__cspOn) { window.__cspOn = true; document.addEventListener('securitypolicyviolation',"
                " (e) => window.__csp.push(e.violatedDirective + ' ' + e.blockedURI)); }")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("release")
    root = tmp / LONG
    root.parent.mkdir(parents=True)
    import shutil
    shutil.copytree(H.EXAMPLES, root)
    try:
        import plotly  # noqa: F401
        from test_gui_results import write_band
        (root / "results").mkdir()
        write_band(root / "results")
    except ImportError:
        pass
    a = U.start_app(tmp, root=root)
    a[0].execute_script(CSP_LISTENER)
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    return app


def test_favicon_and_menu_bar_logo(fresh):
    drv = fresh[0]
    icons = drv.execute_script("return [...document.querySelectorAll('link[rel=icon]')].map((l) => [l.getAttribute('href'), l.type, l.sizes.value])")
    assert [i[0].rsplit("/", 1)[1] for i in icons] == ["icon.svg", "favicon-32.png", "favicon-16.png", "icon-192.png"]
    for href, _, _ in icons:
        st = drv.execute_async_script("const d = arguments[1]; fetch(arguments[0]).then((r) => d(r.status), () => d(0))", href)
        assert st == 200, href
    assert drv.execute_script("const b = document.querySelector('.brand-mark'); return b.getAttribute('aria-label') + ' ' + b.querySelector('img').naturalWidth") == "MACE 32"


def test_window_titles_end_in_mace_and_keep_the_folder_short(fresh):
    drv = fresh[0]
    drv.execute_script("window.__mace.closeAll()")
    time.sleep(0.3)
    # the browser tab: the folder's own name, as VS Code does (polish A9)
    title = drv.execute_script("return document.title")
    assert title == "runs — MACE", title
    U.open_doc(drv, "structures/MgO.cif")
    title = drv.execute_script("return document.title")
    assert title == "MgO.cif — runs — MACE", title
    win = drv.execute_script("const t = document.getElementById('win-title'); return [t.textContent, t.title]")
    assert "…" in win[0] and LONG not in win[0] and win[1].endswith(LONG), win


def test_about_shows_the_hero_and_reads_about_mace(fresh):
    drv = fresh[0]
    drv.execute_script("window.__mace.run('help.about')")
    H.wait(drv, "return document.querySelector('#about[open]') !== null")
    # the two SVGs load after the dialog opens: wait for them, not for a fixed time
    H.wait(drv, "const im = [...document.querySelectorAll('#about .about-hero img')];"
                "return im.length === 2 && im.every((i) => i.complete)", 10)
    about = drv.execute_script("""const d = document.getElementById('about');
      return [d.querySelector('h2').textContent, [...d.querySelectorAll('.about-hero img')].map((i) => [i.alt, i.naturalWidth]),
              d.textContent.includes('MACE wordmark (MIT, this repository)')]""")
    assert about[0] == "About MACE" and about[2]
    assert [x[0] for x in about[1]] == ["MACE", "MACE"] and all(x[1] > 0 for x in about[1])


def test_help_documentation_opens_the_guide_for_this_version(fresh):
    drv = fresh[0]
    main = drv.current_window_handle
    drv.execute_script("window.__mace.run('help.docs')")
    H.wait(drv, "return document.querySelector('#docs[open]') !== null")
    link, local = drv.execute_script("return [document.getElementById('docs-link').href, document.getElementById('docs-local').textContent]")
    from mace import __version__
    from mace.gui.helplinks import docs_ref
    # a ref that exists on GitHub: the release tag only for a tagged final release, else main
    assert link == f"https://github.com/DjokicMa/MACE/blob/{docs_ref(__version__)}/docs/gui.md"
    assert docs_ref(__version__) in ("main", f"v{__version__}")
    assert local.endswith("/docs/gui.md") and (H.REPO_ROOT / "docs" / "gui.md").samefile(local)
    for h in drv.window_handles:            # the new tab it opened (no network needed for the test)
        if h != main:
            drv.switch_to.window(h)
            drv.close()
    drv.switch_to.window(main)


def test_no_csp_violation_while_drawing(fresh):
    """'unsafe-eval' is gone: 3Dmol, the editor's viewport and plotly still draw."""
    drv = fresh[0]
    drv.execute_script(CSP_LISTENER)
    U.open_doc(drv, "structures/MgO.cif")
    U.open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out", wait_atoms=False)
    time.sleep(2)
    if (fresh[2] / "results" / "mgo_band.BAND.DAT").exists():
        U.open_doc(drv, "results/mgo_band.BAND.DAT", wait_atoms=False)
        H.wait(drv, "const v = window.__mace.plotView, s = v && v.state(); return s && s.drawn", 60)
    assert drv.execute_script("return window.__csp") == []
    U.no_errors(drv)
