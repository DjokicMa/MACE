"""CLI parity in headless Firefox (plan E1, E2, E3, E14): real clicks on a
fresh profile.

A folder of real corpus files under neutral names (skipped without the
gitignored test/ corpus): each kind shows its chip and opens into its view
with no error notice; molecule outputs open; a properties output opens its
document; a job log opens as text linked to its calculation; a FREQ output
opens as Modes with arrows that toggle during the animation."""
import shutil
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")          # the structure payloads name elements with ase
pytest.importorskip("plotly")       # the FREQ parser and the spectrum plots
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.by import By  # noqa: E402

CORPUS = H.REPO_ROOT / "test"
MOL = "1LiFSI-1DEC-conf{}_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized"
FREQ_SRC = CORPUS / "FREQ" / (MOL.format(1) + "_freq_HSESOL3C_optimized.out")
DIA = CORPUS / "FREQ" / "2_dia2_opt_rev1_freq_B3LYP-D3-D3_optimized.out"
OPT_SRC = CORPUS / "OPT" / (MOL.format(3) + ".out")
DOSS_STEM = CORPUS / "DOSS" / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    if not FREQ_SRC.is_file() or not DIA.is_file():
        pytest.skip("test/ corpus not present (gitignored)")
    tmp = tmp_path_factory.mktemp("parity_e2e")
    root = H.examples_copy(tmp)
    m = root / "mixed"
    m.mkdir()
    shutil.copyfile(FREQ_SRC, m / "mol_freq.out")
    shutil.copyfile(DIA, m / "dia_freq.out")
    log = next(FREQ_SRC.parent.glob(MOL.format(1) + "_freq_HSESOL3C_optimized_temp-*.o"))
    shutil.copyfile(log, m / "mol_freq_temp-1035400.o")
    shutil.copyfile(OPT_SRC, m / "mol_opt.out")
    for ext in (".out", ".d3", ".DOSS.DAT", ".f25", ".f9", ".sh"):
        shutil.copyfile(Path(str(DOSS_STEM) + ext), m / f"dia_doss{ext}")
    shutil.copyfile(CORPUS / "BAND" / "ipBANDS_2151873.out", m / "bands_2151873.out")
    shutil.copyfile(CORPUS / "BAND" / "ipBANDS_2151873.err", m / "bands_2151873.err")
    (m / "died.out").write_text("\n".join(OPT_SRC.read_text().splitlines()[:120]) + "\n")
    (m / "broken.cif").write_text("data_x\n_cell_length_a oops\n")
    # spectra (plan E4): a FREQ run with its IRSPEC.DAT, never opened before; one with IR intensities and no SPEC.DAT
    c = root / "spectra"
    c.mkdir()
    shutil.copyfile(FREQ_SRC, c / "cold_freq.out")
    shutil.copyfile(FREQ_SRC.with_name(FREQ_SRC.stem + "_IRSPEC.DAT"), c / "cold_freq_IRSPEC.DAT")
    shutil.copyfile(next(DIA.parent.glob("Ag1Cl5_sym_*freq*.out")), c / "agcl_freq.out")
    # fort.25 of a BAND run beside its BAND.DAT (plan E7), and a BOLTZTRA run's data files (plan E8)
    band = CORPUS / "BAND" / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band"
    for ext in (".f25", ".BAND.DAT"):
        shutil.copyfile(Path(str(band) + ext), m / f"dia_band{ext}")
    tr = sorted((CORPUS / "TRANSPORT").glob("*.SEEBECK.dat"))[0].name.split(".")[0]
    for q in ("SEEBECK", "SIGMA", "SIGMAS", "KAPPA"):
        shutil.copyfile(CORPUS / "TRANSPORT" / f"{tr}.{q}.dat", c / f"run.{q}.dat")
    a = U.start_app(tmp, root=root, MACE_GUI_MAX_WORKERS="2")
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    drv = app[0]
    drv.execute_script("window.__mace.settings.set('reduceMotion', 'off')")
    H.wait(drv, "return window.MACE.state.details && window.MACE.state.details['mixed/mol_freq.out'] &&"
                " window.MACE.state.details['mixed/mol_freq.out'].refined", 60)
    return app


def click(drv, el):
    ActionChains(drv).move_to_element(el).click().perform()


def click_file(drv, path):
    """A real click on the explorer row (found again when the list re-renders under the pointer)."""
    from selenium.common.exceptions import StaleElementReferenceException
    drv.execute_script("window.MACE.state.filter = 'all'; window.MACE.state.text = ''; window.MACE.renderFiles()")
    for attempt in range(6):
        try:
            el = drv.find_element(By.CSS_SELECTOR, f'#file-list .file[data-path="{path}"]')
            drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
            click(drv, el)
            return
        except StaleElementReferenceException:
            time.sleep(0.2)
    raise AssertionError(f"could not click {path}")


def active(drv):
    return drv.execute_script("const d = window.__mace.activeDoc(); return d && {kind: d.kind, path: d.path, src: d.srcPath,"
                              " jobLog: !!d.jobLog, info: !!d.info, props: !!d.props, title: d.title}")


def no_errors(drv):
    """No page errors; the browser's benign "ResizeObserver loop completed with undelivered
    notifications" (a viewport resized twice in one frame while documents switch) aside."""
    errs = [e for e in H.errors(drv) if "ResizeObserver loop" not in e]
    assert not errs, errs


def error_toasts(drv):
    return drv.execute_script("return [...document.querySelectorAll('#toasts .toast2.error .msg')].map((m) => m.textContent)")


def test_kind_chips_and_every_kind_opens(fresh):
    drv = fresh[0]
    chips = drv.execute_script("return Object.fromEntries([...document.querySelectorAll('#file-list .file')]"
                               ".filter((b) => b.dataset.path.startsWith('mixed/')).map((b) => [b.dataset.path, b.querySelector('.kb').textContent]))")
    assert chips["mixed/mol_freq.out"] == "FREQ" and chips["mixed/dia_freq.out"] == "FREQ"
    assert chips["mixed/mol_opt.out"] == "OPT" and chips["mixed/dia_doss.out"] == "DOSS"
    assert chips["mixed/mol_freq_temp-1035400.o"] == "log" and chips["mixed/bands_2151873.out"] == "log"
    assert chips["mixed/bands_2151873.err"] == "log" and chips["mixed/dia_doss.f9"] == "wf"
    assert chips["mixed/dia_doss.sh"] == "sh" and chips["mixed/dia_doss.f25"] == "f25"
    # the Logs & data filter holds the logs (by content), not Structures
    drv.execute_script("window.MACE.state.filter = 'logs'; window.MACE.renderFiles()")
    logs = drv.execute_script("return [...document.querySelectorAll('#file-list .file')].map((b) => b.dataset.path)")
    assert "mixed/mol_freq_temp-1035400.o" in logs and "mixed/mol_freq.out" not in logs
    drv.execute_script("window.MACE.state.filter = 'struct'; window.MACE.renderFiles()")
    st = drv.execute_script("return [...document.querySelectorAll('#file-list .file')].map((b) => b.dataset.path)")
    assert "mixed/mol_freq_temp-1035400.o" not in st and "mixed/mol_opt.out" in st

    # a molecule OPT: a trajectory of its two printed geometries, no cell
    click_file(drv, "mixed/mol_opt.out")
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.path === 'mixed/mol_opt.out' && d.kind === 'trajectory'", 30)
    # a properties output: its document, then its DOS plot from there
    click_file(drv, "mixed/dia_doss.out")
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.props", 20)
    btn = H.wait(drv, "return [...document.querySelectorAll('#text-info [data-info-open]')].find((b) => b.textContent.includes('Density of States'))", 20)
    U.shot(drv, "parity_props_doc")
    click(drv, btn)
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.kind === 'plot' && d.path === 'mixed/dia_doss.DOSS.DAT'", 30)
    # a job log: text, a header with its calculation, which opens from there
    click_file(drv, "mixed/mol_freq_temp-1035400.o")
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.jobLog", 20)
    H.wait(drv, "return document.querySelector('#text-job').textContent.includes('mol_freq.out')", 20)
    assert "job 1035400" in drv.execute_script("return document.querySelector('#text-job').textContent")
    U.shot(drv, "parity_job_log")
    click(drv, drv.find_element(By.CSS_SELECTOR, "#text-job [data-job=open]"))
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.kind === 'freq' && d.path === 'mixed/mol_freq.out'", 30)
    H.wait(drv, "const s = window.__mace.freqView && window.__mace.freqView.state(); return s && s.n === 84 && s.mode", 60)
    # a log with no calculation; its .err adds a Problems row
    click_file(drv, "mixed/bands_2151873.out")
    H.wait(drv, "return document.querySelector('#text-job').textContent.includes('No calculation found for this log')", 20)
    click_file(drv, "mixed/bands_2151873.err")
    H.wait(drv, "return [...document.querySelectorAll('.bottom-pane.problems .p-msg')].some((m) => m.textContent.startsWith('Job errors: Traceback'))", 20)
    # a wave function: the info document, never read
    click_file(drv, "mixed/dia_doss.f9")
    H.wait(drv, "return document.querySelector('#text-info').textContent.includes('CRYSTAL wave function')", 20)
    assert drv.execute_script("return !!document.querySelector('#text-info [data-info-open]')")       # Reveal the run's output
    # a job script opens as text
    click_file(drv, "mixed/dia_doss.sh")
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.kind === 'text' && d.srcPath === 'mixed/dia_doss.sh'", 20)
    # an output with no geometry: its text and a Problems row, never an error notice
    click_file(drv, "mixed/died.out")
    H.wait(drv, "return [...document.querySelectorAll('.bottom-pane.problems .p-msg')].some((m) => m.textContent.startsWith('No geometry in this output'))", 20)
    assert active(drv)["kind"] == "text"
    assert not error_toasts(drv), error_toasts(drv)
    assert not any("geometry block" in t for t in U.toasts(drv))
    no_errors(drv)


def test_freq_opens_as_modes_and_arrows_toggle_while_playing(fresh):
    drv = fresh[0]
    click_file(drv, "mixed/dia_freq.out")
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.kind === 'freq'", 30)
    H.wait(drv, "const s = window.__mace.freqView.state(); return s && s.mode", 60)
    H.wait(drv, "return window.__mace.freqView.playing()", 10)
    box = drv.find_element(By.ID, "fo-arrows")
    assert drv.execute_script("return document.querySelector('label:has(#fo-arrows)').textContent.trim()") == "Arrows"
    time.sleep(0.4)
    st = drv.execute_script("return window.__mace.freqView.state()")
    assert st["arrows"], "arrows are drawn during the animation"
    a0 = st["arrows"][0]["start"]
    time.sleep(0.1)
    for _ in range(20):                                   # a later frame: the arrows moved with their atoms
        st2 = drv.execute_script("return window.__mace.freqView.state()")
        if st2["arrows"] and st2["arrows"][0]["start"] != a0:
            break
        time.sleep(0.05)
    assert st2["arrows"][0]["start"] != a0
    f0 = drv.execute_script("return window.__maceFreqFrames")
    click(drv, box)                                       # off, while playing
    H.wait(drv, "return window.__mace.freqView.state().arrows.length === 0", 5)
    assert drv.execute_script("return window.__mace.freqView.playing()")
    click(drv, box)                                       # on again
    H.wait(drv, "return window.__mace.freqView.state().arrows.length > 0", 5)
    assert drv.execute_script("return window.__mace.freqView.stats().playing")
    time.sleep(0.5)
    assert drv.execute_script("return window.__maceFreqFrames") > f0
    U.shot(drv, "parity_freq_arrows_playing")
    # paused: today's still arrows
    drv.execute_script("window.__mace.freqView.play(false)")
    st = drv.execute_script("return window.__mace.freqView.state()")
    assert st["frame"] == 0 and st["arrows"]
    # the Structure button opens the output's structure
    click(drv, drv.find_element(By.CSS_SELECTOR, "#freq-host [data-act=structure]"))
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.path === 'mixed/dia_freq.out' && (d.kind === 'trajectory' || d.kind === 'structure')", 30)
    # Settings: open FREQ outputs as structures
    drv.execute_script("window.__mace.closeAll(); window.__mace.settings.set('files.openFreqAs', 'structure')")
    click_file(drv, "mixed/mol_freq.out")
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.path === 'mixed/mol_freq.out' && d.kind !== 'freq' && d.kind !== 'text'", 30)
    drv.execute_script("window.__mace.settings.set('files.openFreqAs', 'modes')")
    no_errors(drv)


def test_stick_spectrum_click_selects_the_mode(fresh):
    drv = fresh[0]
    drv.execute_script("window.__mace.openModes('mixed/mol_freq.out')")
    H.wait(drv, "const s = window.__mace.freqView && window.__mace.freqView.state(); return s && s.mode", 60)
    line = H.wait(drv, "return document.querySelector('#freq-table .fm-spec line[data-mode]')", 10)
    want = int(line.get_attribute("data-mode"))
    drv.execute_script("arguments[0].dispatchEvent(new MouseEvent('click', {bubbles: true}))", line)
    H.wait(drv, "return window.__mace.freqView.state().mode === arguments[0]", 10, want)
    drv.execute_script("window.__mace.freqView.play(false)")


@pytest.mark.parametrize("scale", [100, 110, 125])
def test_spectrum_peak_click_opens_its_mode(fresh, scale):
    """An IRSPEC.DAT opened cold (its FREQ output never read): the peaks are its modes,
    and a real click on the strongest peak opens Modes at that mode, at every UI scale.

    The strongest (mode 54, 1180.5 cm-1) has mode 55 (1182.0 cm-1) on the same band
    17 px below it. Plotly throttles hover to one per 50 ms and a click re-uses the
    last hover, so a click right after the pointer's move (WebDriver moves it along a
    path from below) opened mode 55 whenever the last hover fell on the way; the
    plot now takes the marker under the click's own position.

    The UI scale zooms the plot: the click's screen offset is zoom x the plot's own px
    (at 110 % and 125 % the click used to land on a marker up and to the left of it)."""
    drv = fresh[0]
    drv.execute_script("window.__mace.settings.set('uiScale', arguments[0])", scale)
    try:
        _peak_click(drv, scale)
    finally:
        drv.execute_script("window.__mace.settings.set('uiScale', 100)")


def _peak_click(drv, scale):
    click_file(drv, "spectra/cold_freq_IRSPEC.DAT")
    H.wait(drv, "const s = window.__mace.plotView && window.__mace.plotView.state(); return s && s.drawn && s.peaks > 0", 90)
    st = drv.execute_script("return window.__mace.plotView.state()")
    assert st["out"] == "spectra/cold_freq.out"
    # the plot settled (the pane's size and the figure's agree), then the strongest IR mode's marker on the screen
    H.wait(drv, "const el = window.__mace.plotView.el(), fl = el._fullLayout; return fl && Math.abs(fl.width - el.clientWidth) <= 4", 20)
    time.sleep(0.5)
    want, (*xy, z) = drv.execute_script("""
        const el = window.__mace.plotView.el(), t = el.data.find((x) => x.meta === 'peaks');
        let k = 0; for (let i = 1; i < t.y.length; i++) if (t.y[i] > t.y[k]) k = i;
        // the plot's screen corner plus its own px times the zoom
        const fl = el._fullLayout, r = el.getBoundingClientRect(), z = r.width / el.offsetWidth;
        return [t.customdata[k], [r.left + (fl._size.l + fl.xaxis.l2p(t.x[k])) * z, r.top + (fl._size.t + fl.yaxis.l2p(t.y[k])) * z, z]];""")
    assert abs(z - scale / 100) < 0.01, (scale, z)
    U.shot(drv, f"parity_ir_peaks_{scale}")
    body = drv.find_element(By.TAG_NAME, "body")
    ActionChains(drv).move_to_element_with_offset(body, 0, 0).perform()
    ActionChains(drv).move_by_offset(round(xy[0]) - drv.execute_script("return innerWidth / 2 | 0"),
                                     round(xy[1]) - drv.execute_script("return innerHeight / 2 | 0")).click().perform()
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.kind === 'freq' && d.path === 'spectra/cold_freq.out'", 30)
    H.wait(drv, "const v = window.__mace.freqView, s = v && v.state(); return s && s.mode === arguments[0]", 60, want)
    drv.execute_script("window.__mace.freqView.play(false)")
    no_errors(drv)


def test_spectrum_broadened_from_the_modes(fresh):
    """A FREQ output with IR intensities and no SPEC.DAT: Modes › IR draws the curve
    broadened from the modes with CRYSTAL's width; one with no intensities says so."""
    drv = fresh[0]
    click_file(drv, "spectra/agcl_freq.out")
    H.wait(drv, "const s = window.__mace.freqView && window.__mace.freqView.state(); return s && s.mode", 60)
    drv.execute_script("window.__mace.freqView.play(false)")
    click(drv, drv.find_element(By.CSS_SELECTOR, "#freq-host [data-act=ir]"))
    H.wait(drv, "const s = window.__mace.plotView && window.__mace.plotView.state(); return s && s.drawn && s.source === 'modes'", 60)
    note = drv.execute_script("return document.querySelector('#plot-host .pv-note').textContent")
    assert "Broadened from 15 IR-active modes (Lorentzian, FWHM 8 cm⁻¹, CRYSTAL's default)" in note, note
    U.shot(drv, "parity_ir_from_modes")
    # a wider line from Plot options: the curve is read again with that width
    drv.execute_script("const i = document.querySelector('#po-fwhm'); i.value = '20'; i.dispatchEvent(new Event('change', {bubbles: true}))")
    H.wait(drv, "return window.__mace.plotView.state().fwhm === 20", 30)
    # the diamond FREQ run has no intensities: the Modes header offers no IR, the API says why
    drv.execute_script("window.__mace.openModes('mixed/dia_freq.out')")
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.path === 'mixed/dia_freq.out' && window.__mace.freqView.state().mode", 60)
    assert drv.execute_script("return document.querySelector('#freq-host [data-act=ir]').hidden")
    drv.execute_script("window.__mace.freqView.play(false)")
    r = drv.execute_async_script("const d = arguments[0]; window.MACE.api('spectrum?path=mixed/dia_freq.out&source=modes&kind=ir')"
                                 ".then(() => d('ok'), (e) => d(e.message))")
    assert r == "This output has no IR or Raman intensities (run FREQCALC with INTENS)"
    no_errors(drv)


def test_fort25_names_its_run_and_opens_the_dat_plot(fresh):
    drv = fresh[0]
    for path, word, button, dat in (("mixed/dia_band.f25", "a BAND run", "Open Band Structure", "mixed/dia_band.BAND.DAT"),
                                    ("mixed/dia_doss.f25", "a DOSS run", "Open Density of States", "mixed/dia_doss.DOSS.DAT")):
        click_file(drv, path)
        H.wait(drv, "return document.querySelector('#text-info h3') && document.querySelector('#text-info h3').textContent.includes(arguments[0])", 20, word)
        btn = drv.find_element(By.XPATH, f"//div[@id='text-info']//button[.//span[text()='{button}']]")
        click(drv, btn)
        H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.kind === 'plot' && d.path === arguments[0]", 30, dat)
    no_errors(drv)


def test_transport_plot_and_temperatures(fresh):
    pytest.importorskip("plotly")
    drv = fresh[0]
    click_file(drv, "spectra/run.SEEBECK.dat")
    H.wait(drv, "const s = window.__mace.plotView && window.__mace.plotView.state(); return s && s.kind === 'transport' && s.drawn", 60)
    fig = drv.execute_script("return window.__mace.plotView.figure()")
    assert fig["layout"]["yaxis"]["title"]["text"] == "Seebeck coefficient (µV/K)"
    on = drv.execute_script("return document.querySelector('.po-tq [aria-checked=true]').textContent")
    assert on == "Seebeck"
    y0 = drv.execute_script("return window.__mace.plotView.el().data[0].y.slice(0, 50)")
    temp = drv.find_element(By.ID, "po-temp")
    from selenium.webdriver.support.ui import Select
    opts = [o.get_attribute("value") for o in Select(temp).options]
    assert len(opts) >= 2 and Select(temp).first_selected_option.get_attribute("value") == opts[0]
    Select(temp).select_by_value(opts[1])
    H.wait(drv, "return JSON.stringify(window.__mace.plotView.el().data[0].y.slice(0, 50)) !== arguments[0]", 10, __import__("json").dumps(y0))
    U.shot(drv, "parity_transport")
    # another quantity of the same run, from its tab
    click(drv, drv.find_element(By.XPATH, "//span[contains(@class,'po-tq')]/button[text()='σ']"))
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.path === 'spectra/run.SIGMA.dat' && window.__mace.plotView.state().drawn", 30)
    no_errors(drv)
