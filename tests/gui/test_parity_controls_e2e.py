"""Real clicks on the area E controls the other parity tests drive through the
API: the Modes header's Structure, IR and Raman buttons, the line shape and
Lorentzian share of a spectrum broadened from the modes, the Volume's plane
select and Save Result As (its name dialog), the transport quantity switch,
and the new Settings rows in their sections. A fresh profile on the CI
fixtures (tests/fixtures/gui_parity); the transport check uses the gitignored
corpus and is skipped without it."""
import shutil
import time

import pytest

import guiharness as H
import p4util as U
import test_parity_fixtures as F

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")
pytest.importorskip("ase")
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.by import By  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402

TRANSPORT = H.REPO_ROOT / "test" / "TRANSPORT"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("parity_controls")
    root = F.unpack(tmp / "root")
    # the same FREQ run without CRYSTAL's SPEC.DAT files: its spectra are broadened from the modes
    shutil.copyfile(root / "water_freq.out", root / "bare_freq.out")
    if TRANSPORT.is_dir():
        stem = sorted(TRANSPORT.glob("*.SEEBECK.dat"))[0].name.split(".")[0]
        for q in ("SEEBECK", "SIGMA"):
            shutil.copyfile(TRANSPORT / f"{stem}.{q}.dat", root / f"run.{q}.dat")
    a = U.start_app(tmp, root=root, MACE_GUI_MAX_WORKERS="2")
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    app[0].execute_script("window.__mace.settings.set('reduceMotion', 'on')")
    return app


def click(drv, el):
    drv.execute_script("arguments[0].scrollIntoView({block: 'nearest', inline: 'nearest'})", el)
    ActionChains(drv).move_to_element(el).pause(0.1).click().perform()


def visible(drv, css):
    return [e for e in drv.find_elements(By.CSS_SELECTOR, css) if e.is_displayed()]


def open_path(drv, path):
    drv.execute_async_script("const done = arguments[1]; window.__mace.open(arguments[0]).then(() => done())", path)


def plot_state(drv):
    return drv.execute_script("return window.__mace.plotView && window.__mace.plotView.state()")


def no_errors(drv):
    errs = [e for e in H.errors(drv) if "ResizeObserver loop" not in e]
    assert not errs, errs
    bad = drv.execute_script("return [...document.querySelectorAll('#toasts .toast2.error .msg')].map((m) => m.textContent)")
    assert not bad, bad


def test_modes_header_buttons(fresh):
    drv = fresh[0]
    open_path(drv, "water_freq.out")
    H.wait(drv, "const s = window.__mace.freqView && window.__mace.freqView.state(); return s && s.mode", 60)
    click(drv, visible(drv, "#freq-host .fv-ir")[0])
    H.wait(drv, "const s = window.__mace.plotView && window.__mace.plotView.state(); return s && s.kind === 'ir' && s.drawn", 60)
    assert plot_state(drv)["source"] == "file"                     # CRYSTAL's IRSPEC.DAT beside it
    open_path(drv, "water_freq.out")
    H.wait(drv, "const d = window.__mace.doc(); return d && d.kind === 'freq'", 30)
    click(drv, visible(drv, "#freq-host .fv-raman")[0])
    H.wait(drv, "const s = window.__mace.plotView.state(); return s && s.kind === 'raman' && s.drawn", 60)
    open_path(drv, "water_freq.out")
    H.wait(drv, "const d = window.__mace.doc(); return d && d.kind === 'freq'", 30)
    click(drv, visible(drv, "#freq-host .fv-struct")[0])
    H.wait(drv, "const d = window.__mace.doc(); return d && d.path === 'water_freq.out' && (d.kind === 'trajectory' || d.kind === 'structure')", 60)
    no_errors(drv)


def test_broadened_spectrum_controls(fresh):
    drv = fresh[0]
    open_path(drv, "bare_freq.out")
    H.wait(drv, "const s = window.__mace.freqView && window.__mace.freqView.state(); return s && s.mode", 60)
    click(drv, visible(drv, "#freq-host .fv-ir")[0])
    H.wait(drv, "const s = window.__mace.plotView.state(); return s && s.kind === 'ir' && s.source === 'modes' && s.drawn", 60)
    sel = drv.find_element(By.ID, "po-shape")
    click(drv, sel)
    sel.find_element(By.CSS_SELECTOR, "option[value='gaussian']").click()
    H.wait(drv, "const s = window.__mace.plotView.state(); return s.opts.shape === 'gaussian' && s.drawn", 30)
    assert "Gaussian" in drv.execute_script("return document.querySelector('.pv-note').textContent")
    open_path(drv, "bare_freq.out")
    H.wait(drv, "const d = window.__mace.doc(); return d && d.kind === 'freq'", 30)
    click(drv, visible(drv, "#freq-host .fv-raman")[0])
    H.wait(drv, "const s = window.__mace.plotView.state(); return s && s.kind === 'raman' && s.source === 'modes' && s.drawn", 60)
    click(drv, drv.find_element(By.ID, "po-eta"))
    for _ in range(5):
        # the options are drawn again after each step: the slider keeps the focus, so the keys go on reaching it
        ActionChains(drv).send_keys(Keys.ARROW_LEFT).perform()
        time.sleep(0.4)
    assert drv.execute_script("return document.activeElement && document.activeElement.id") == "po-eta"
    H.wait(drv, "const s = window.__mace.plotView.state(); return s.opts.eta <= 0.55 && s.opts.shape === 'pseudo-voigt' && s.drawn", 30)
    assert "pseudo-Voigt" in drv.execute_script("return document.querySelector('.pv-note').textContent")
    no_errors(drv)


def test_volume_plane_and_save_result_as(fresh):
    drv, root = fresh[0], fresh[2]
    open_path(drv, "mgo_dens/DENS_CUBE.DAT")
    H.wait(drv, "const s = window.__mace.volumeView && window.__mace.volumeView.state(); return s && s.surfaces > 0", 60)
    click(drv, drv.find_element(By.ID, "vo-slice"))
    H.wait(drv, "const s = window.__mace.volumeView.state(); return s.slice && s.slice.axis === 'c'", 30)
    sel = drv.find_element(By.ID, "vo-axis")
    click(drv, sel)
    sel.find_element(By.CSS_SELECTOR, "option[value='a']").click()
    H.wait(drv, "const s = window.__mace.volumeView.state(); return s.slice && s.slice.axis === 'a'", 30)
    # Difference (its dialog), then Save Result As (its name prompt): a new file, and never over one
    click(drv, visible(drv, '#volume-host [data-act="diff"]')[0])
    H.wait(drv, "return !!document.querySelector('#vdiff-ok')", 10)
    rows = drv.find_elements(By.CSS_SELECTOR, "#vdiff-dialog .vd-row .vd-path")
    rows[1].find_element(By.CSS_SELECTOR, "option[value='mgo_pato/DENS_CUBE.DAT']").click()
    click(drv, drv.find_element(By.ID, "vdiff-ok"))
    H.wait(drv, "const s = window.__mace.volumeView.state(); return s && s.combo && s.surfaces > 0", 60)
    for attempt, expect in (("deformation.CUBE", "saved"), ("deformation.CUBE", "exists")):
        click(drv, visible(drv, '#volume-host [data-act="save"]')[0])
        alert = drv.switch_to.alert
        alert.send_keys(attempt)
        alert.accept()
        if expect == "saved":
            H.wait(drv, "return [...document.querySelectorAll('#toasts .toast2')].some((t) => t.textContent.includes('Saved'))", 60)
            assert (root / attempt).stat().st_size > 0
            size = (root / attempt).stat().st_size
        else:
            H.wait(drv, "return [...document.querySelectorAll('#toasts .toast2.error')].some((t) => /exists/.test(t.textContent))", 60)
            assert (root / attempt).stat().st_size == size
            drv.execute_script("document.querySelectorAll('#toasts .toast2').forEach((t) => t.remove())")
    assert not list(root.glob(".mace-cube-*"))
    (root / "deformation.CUBE").unlink()
    assert not [e for e in H.errors(drv) if "ResizeObserver loop" not in e]


@pytest.mark.skipif(not TRANSPORT.is_dir(), reason="test/ corpus not present (gitignored)")
def test_transport_quantity_switch(fresh):
    drv = fresh[0]
    open_path(drv, "run.SEEBECK.dat")
    H.wait(drv, "const s = window.__mace.plotView.state(); return s && s.kind === 'transport' && s.drawn", 60)
    click(drv, drv.find_element(By.CSS_SELECTOR, ".po-tq [data-tq='run.SIGMA.dat']"))
    H.wait(drv, "const d = window.__mace.doc(); const s = window.__mace.plotView.state(); return d.path === 'run.SIGMA.dat' && s.kind === 'transport' && s.drawn", 60)
    assert "conductivity" in drv.execute_script("return window.__mace.plotView.el().layout.yaxis.title.text")
    no_errors(drv)


def test_new_settings_rows_sit_in_their_sections(fresh):
    drv = fresh[0]
    drv.execute_script("document.body.focus()")
    U.keys(drv, Keys.CONTROL, ",")
    H.wait(drv, "return document.querySelector('#settings') && document.querySelector('#settings').open", 10)
    where = {}
    for sec, key in (("appearance", "notifications.infoSeconds"), ("viewport", "volume.quality"), ("viewport", "volume.mepColors"),
                     ("editor", "structure.vacuumGap"), ("files", "export.dpi"), ("files", "spectra.lineShape"),
                     ("files", "files.openFreqAs")):
        click(drv, drv.find_element(By.CSS_SELECTOR, f"#settings .set-nav-item[data-sec='{sec}']"))
        time.sleep(0.2)
        where[key] = bool(drv.find_elements(By.CSS_SELECTOR, f"#set-pane [data-key='{key}']"))
    assert all(where.values()), where
    # a real change through its control reaches the profile
    click(drv, drv.find_element(By.CSS_SELECTOR, "#settings .set-nav-item[data-sec='viewport']"))
    click(drv, drv.find_element(By.CSS_SELECTOR, "#set-pane [data-key='volume.quality'][data-val='high']"))
    assert drv.execute_script("return window.__mace.settings.get('volume.quality')") == "high"
    drv.execute_script("window.__mace.settings.set('volume.quality', 'medium'); document.querySelector('#settings').close()")
    no_errors(drv)
