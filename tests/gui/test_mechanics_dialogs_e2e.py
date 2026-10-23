"""Phonopy, mechanical properties and the relaxation time (1.1.6) in
headless Firefox, on repo data only (no corpus):

* Calculation Setup on the Si parent output: opt2d12's 4 ELASTIC ... 8
  Phonopy as entries of its calculation types, numbered as the CLI's menu;
  5 EOS writes its deck through the real CLI (mace opt2d12 --calc-type EOS);
  8 Phonopy runs mace opt2d12 --calc-type PHONOPY (its refusal when phonopy
  is not installed);
* Analysis › Mechanical Properties Results: the real diamond ELASTCON run
  (HPCC), and its pointer to Setup;
* Relaxation Time: the estimate, the fit of the real diamond strain series,
  and the automatic m* from a BAND file beside the series (m_d from a DOSS
  given in its field);
* Phonopy: a set's job table and the band/DOS/thermal plots of phonopy's files,
  opened from the Analysis menu and from the set's files in the Explorer.
No JavaScript error on the way.
"""
import json
import math
import shutil
import sys

import numpy as np

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")

REPO = H.REPO_ROOT
HPCC = REPO / "tests" / "data" / "hpcc_116"
FIX = REPO / "tests" / "fixtures" / "phonopy_crystal"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mxe2e")
    root = tmp / "root"
    root.mkdir()
    shutil.copy(FIX / "Si" / "parent.out", root / "si_opt.out")
    shutil.copy(FIX / "Si" / "parent.d12", root / "si_opt.d12")
    shutil.copy(HPCC / "mechanics" / "dia_elastcon.out", root / "dia_elastcon.out")
    (root / "strain").mkdir()
    for f in (HPCC / "strain").glob("*.out"):
        shutil.copy(f, root / "strain" / f.name)
    # the same series with a band structure beside it (m* automatic) and a DOS elsewhere (given)
    _mstar_folder(root / "mstar", HPCC / "strain")
    ph = root / "phonopy_si_222"
    ph.mkdir()
    (ph / "mace_phonopy.json").write_text(json.dumps({"parent_out": str(root / "si_opt.out"), "jobs": ["si_ph222_disp-001"],
                                                      "dim": [[2, 0, 0], [0, 2, 0], [0, 0, 2]], "walltime": "2:00:00",
                                                      "n_atoms_supercell": 16, "warnings": []}))
    for f in ("band.yaml", "total_dos.dat", "thermal_properties.yaml"):
        shutil.copy(FIX / "NaCl_collected" / f, ph / f)
    xdg = tmp / "xdg"
    proc, base, _ = H.start_server(root, xdg)
    try:
        drv = H.firefox(1280, 900)
        H.load(drv, base)
    except BaseException:
        proc.terminate()
        raise
    try:
        yield drv, base, root
    finally:
        drv.quit()
        proc.terminate()


def _mstar_folder(d, strain):
    """The strain series, a simple-cubic BAND (Gamma-X along x: hole 0.4, electron 0.25 m_e) named as a
    BAND run beside it, and a parabolic-band DOSS (m_d 0.5 m_e) in dos/ (tests/test_transport_mstar.py)."""
    sys.path.insert(0, str(REPO / "tests"))
    from test_transport_mstar import _write_cubic_band
    d.mkdir()
    for f in strain.glob("*.out"):
        shutil.copy(f, d / f.name)
    _write_cubic_band(d)
    for suf in (".BAND.DAT", ".d3", ".out"):
        (d / ("sc_band" + suf)).rename(d / ("dia_sp_band" + suf))
    dd = d / "dos"
    dd.mkdir()
    a, m, gap = 5.0, 0.5, 0.05
    vol = (a / 0.529177210903) ** 3
    E = np.arange(-0.3, 0.4, 0.0005)
    g = np.where(E <= 0, vol / (2 * math.pi ** 2) * (2 * m) ** 1.5 * np.sqrt(np.clip(-E, 0, None)), 0.0)
    g += np.where(E >= gap, vol / (2 * math.pi ** 2) * (2 * m) ** 1.5 * np.sqrt(np.clip(E - gap, 0, None)), 0.0)
    rows = [f"# NEPTS {len(E)} NPROJ     1 NSPIN     1", "#"] + [f"{x:14.7E}{y:12.4E}" for x, y in zip(E, g)]
    (dd / "sc_doss.DOSS.DAT").write_text("\n".join(rows) + "\n")
    shutil.copy(d / "dia_sp_band.out", dd / "sc_doss.out")


def _open_dialog(drv, command, path, dlg_id):
    drv.execute_script("window.__mace.run(arguments[0], {path: arguments[1]})", command, path)
    H.wait(drv, f"const d = document.getElementById('{dlg_id}'); return d && d.open")


def _q(drv, sel):
    return drv.execute_script("return document.querySelector(arguments[0])", sel)


SU = "#inspector .setup"


def _setup_calc(drv, calc):
    return drv.execute_script("const s = window.__mace.setup && window.__mace.setup.state(); return s ? s.calc : null") == calc


def test_setup_numbers_the_modules_and_writes_an_eos_deck(app):
    drv, base, root = app
    # Calculation › Mechanical Properties Deck: Setup on the output, at 4 ELASTIC
    drv.execute_script("window.__mace.run('calc.mechanics', {path: 'si_opt.out'})")
    H.wait(drv, f"return !!document.querySelector('{SU} .su-mod[data-mod=ELASTIC] [data-mf=mode]')", 60)
    assert _setup_calc(drv, "ELASTIC")
    # opt2d12's numbers: 1 SP 2 OPT 3 FREQ, 4 ELASTIC 5 EOS 6 QHA 7 CPHF 8 Phonopy, above opt2d3's properties
    nums = drv.execute_script(f"return [...document.querySelectorAll('{SU} .su-calc [data-calc]')].map((b) => [b.dataset.calc, b.dataset.n || '', b.textContent])")
    assert nums[:8] == [["SP", "1", "SP"], ["OPT", "2", "OPT"], ["FREQ", "3", "FREQ"], ["ELASTIC", "4", "ELASTIC"], ["EOS", "5", "EOS"],
                        ["QHA", "6", "QHA"], ["CPHF", "7", "CPHF"], ["PHONOPY", "8", "Phonopy"]], nums
    # and opt2d3's properties numbered as its own menu (1.1.6 final: they had no numbers)
    assert [n[:2] for n in nums[8:]] == [["BAND", "1"], ["DOSS", "2"], ["TRANSPORT", "3"], ["CHARGE", "4"], ["POTENTIAL", "5"],
                                         ["CHARGE+POTENTIAL", "6"], ["ORBITALS", "7"]], nums
    for calc, n in (("ELASTIC", "4"), ("BAND", "1")):
        shown = drv.execute_script(f"return getComputedStyle(document.querySelector('{SU} [data-calc=\"{calc}\"]'), '::before').content")
        assert n in shown, shown
    # the ELASTIC modules, numbered as the CLI's "Mechanical property module" menu
    opts = drv.execute_script(f"return [...document.querySelectorAll('{SU} [data-mf=mode] option')].map((o) => o.textContent)")
    assert opts[0].startswith("1 - Elastic constants (ELASTCON)") and len(opts) == 5, opts
    assert "mace opt2d12 --calc-type ELASTIC" in drv.execute_script(f"return document.querySelector('{SU} .su-chip').textContent")
    # 5 EOS: its sub-keywords, and the deck written by the real CLI
    drv.execute_script(f"document.querySelector('{SU} .su-calc [data-calc=EOS]').click()")
    H.wait(drv, f"return !!document.querySelector('{SU} .su-mod[data-mod=EOS] [data-f=range]')", 30)
    drv.execute_script(f"document.querySelector('{SU} [data-f=range]').value = '0.94 1.06 6';"
                       f"document.querySelector('{SU} [data-mact=write]').click()")
    H.wait(drv, f"return /Wrote /.test(document.querySelector('{SU} [data-mk=state]').textContent)", 120)
    decks = list(root.glob("si_opt_eos_*.d12"))
    assert len(decks) == 1
    lines = decks[0].read_text().splitlines()
    i = lines.index("EOS")
    assert lines[i:i + 5] == ["EOS", "RANGE", "0.94 1.06 6", "PRINT", "END"]
    assert ("mace opt2d12 --out-file si_opt.out --calc-type EOS --non-interactive --elastic-settings"
            in drv.execute_script(f"return document.querySelector('{SU} [data-mk=cmd]').textContent"))
    # Setup stays on the output (its result and the next steps), the deck opens on request
    assert _setup_calc(drv, "EOS") and drv.execute_script(f"return !!document.querySelector('{SU} [data-mact=open]')")
    # Preview and Submit are opt2d12's: the entry says where its deck is written
    drv.execute_script("window.__mace.run('calc.preview')")
    H.wait(drv, "return [...document.querySelectorAll('#toasts > *')].some((t) => t.textContent.includes('use the button in step 3'))", 10)
    U.shot(drv, "setup_eos")
    assert not H.errors(drv)


def test_setup_phonopy_runs_opt2d12_choice_8(app):
    drv, base, root = app
    drv.execute_script("window.__mace.run('calc.phonopy', {path: 'si_opt.out'})")
    H.wait(drv, f"return !!document.querySelector('{SU} .su-mod[data-mod=PHONOPY] [data-dim]')", 60)
    assert _setup_calc(drv, "PHONOPY")
    H.wait(drv, f"return !/Looking/.test(document.querySelector('{SU} .su-mod[data-mod=PHONOPY] [data-mk=info]').textContent)", 30)
    drv.execute_script(f"document.querySelector('{SU} [data-mact=write]').click()")
    H.wait(drv, f"return /Wrote the set|refused or failed/.test(document.querySelector('{SU} [data-mk=state]').textContent)", 300)
    cmd = drv.execute_script(f"return document.querySelector('{SU} [data-mk=cmd]').textContent")
    assert cmd.startswith("mace opt2d12 --out-file si_opt.out --calc-type PHONOPY --non-interactive --dim 2 2 2"), cmd
    U.shot(drv, "setup_phonopy")
    # back to an opt2d12 type: its own form and footer again
    drv.execute_script(f"document.querySelector('{SU} .su-calc [data-calc=SP]').click()")
    H.wait(drv, "const s = window.__mace.setup.state(); return s.calc === 'SP' && s.result && !s.busy", 180)
    assert drv.execute_script(f"return !!document.querySelector('{SU} .su-foot') && !document.querySelector('{SU} .su-mod')")
    assert not H.errors(drv)


def test_mechanical_properties_results_and_the_pointer_to_setup(app):
    drv, base, root = app
    _open_dialog(drv, "analysis.elastic", "dia_elastcon.out", "elastic-dlg")
    # a finished ELASTCON run: read at once (no Set Up tab any more)
    assert not drv.execute_script("return !!document.querySelector('#elastic-dlg [data-tab]')")
    H.wait(drv, "return document.querySelector('#elastic-dlg [data-k=r-body]').textContent.includes('1150.388')", 30)
    body = drv.execute_script("return document.querySelector('#elastic-dlg [data-k=r-body]').textContent")
    assert "mechanically stable" in body and "465.24" in body
    drv.execute_script("document.querySelector('#elastic-dlg [data-close]').click()")
    # opened on a parent output, its pointer opens Setup there at 4 ELASTIC
    _open_dialog(drv, "analysis.elastic", "si_opt.out", "elastic-dlg")
    drv.execute_script("document.querySelector('#elastic-dlg [data-act=setup]').click()")
    H.wait(drv, f"return !document.getElementById('elastic-dlg').open && !!document.querySelector('{SU} .su-mod[data-mod=ELASTIC]')", 60)
    assert _setup_calc(drv, "ELASTIC")
    assert not H.errors(drv)


def test_relaxation_time_dialog(app):
    drv, base, root = app
    _open_dialog(drv, "analysis.tau", "strain/dia_sp_dp_x_p0pct.out", "tau-dlg")
    drv.execute_script("""const d = document.getElementById('tau-dlg'); d.querySelector('[data-tab=est]').click();
      const set = (k, v) => { d.querySelector(`[data-k=${k}]`).value = v; };
      set('dim', '2'); d.querySelector('[data-k=dim]').dispatchEvent(new Event('change'));
      set('m_star', '0.17'); set('m_other', '1.12'); set('e1', '2.72'); set('c', '28.94');
      d.querySelector('[data-act=estimate]').click();""")
    H.wait(drv, "return /τ = 10[0-9]\\./.test(document.querySelector('#tau-dlg [data-k=est-key]').textContent)", 30)
    # the fit tab opened on the strain series (the path was one of its outputs)
    H.wait(drv, "return document.querySelectorAll('#tau-dlg [data-k=series] option').length === 1 && "
                "document.querySelector('#tau-dlg [data-k=series]').textContent.includes('dia_sp_dp_x')", 30)
    drv.execute_script("const d = document.getElementById('tau-dlg'); d.querySelector('[data-k=carrier]').value = 'hole';"
                       "d.querySelector('[data-k=f_m_star]').value = '0.5'; d.querySelector('[data-act=fit]').click();")
    H.wait(drv, "return /E1 -10\\.86/.test(document.querySelector('#tau-dlg [data-k=fit-key]').textContent)", 30)
    assert "does not shift linearly" in drv.execute_script("return document.querySelector('#tau-dlg [data-k=fit-warn]').textContent")
    # a given m*: no automatic masses, no hint
    assert drv.execute_script("return document.querySelector('#tau-dlg [data-k=fm-hint]').hidden")
    drv.execute_script("document.querySelector('#tau-dlg [data-close]').click()")
    assert not H.errors(drv)


def test_relaxation_time_finds_m_star_beside_the_series(app):
    drv, base, root = app
    _open_dialog(drv, "analysis.tau", "mstar/dia_sp_dp_x_p0pct.out", "tau-dlg")
    assert drv.execute_script("return document.querySelector('#tau-dlg #tau-fm').placeholder") == "auto: from the BAND/DOSS beside the series"
    assert drv.execute_script("return document.querySelector('#tau-dlg [data-k=dos_file]').placeholder") == "optional: m_d from the DOS"
    H.wait(drv, "return document.querySelector('#tau-dlg [data-k=series]').textContent.includes('dia_sp_dp_x')", 30)
    # m* empty: the band structure beside the series gives it; the DOSS field gives m_d
    drv.execute_script("const d = document.getElementById('tau-dlg'); d.querySelector('[data-k=carrier]').value = 'hole';"
                       "d.querySelector('[data-k=f_m_star]').value = ''; d.querySelector('[data-k=dos_file]').value = 'mstar/dos/sc_doss.DOSS.DAT';"
                       "d.querySelector('[data-act=fit]').click();")
    H.wait(drv, "return /E1 /.test(document.querySelector('#tau-dlg [data-k=fit-key]').textContent)", 60)
    hint = drv.execute_script("const h = document.querySelector('#tau-dlg [data-k=fm-hint]'); return [h.hidden, h.textContent, h.title]")
    assert hint[0] is False and hint[1] == "from BAND: 0.400 mₑ (Γ→X)", hint
    assert "dia_sp_band.BAND.DAT" in hint[2]
    md = drv.execute_script("const h = document.querySelector('#tau-dlg [data-k=md-hint]'); return [h.hidden, h.textContent]")
    assert md[0] is False and md[1].startswith("m_d from DOSS: 0.50"), md
    out = drv.execute_script("return document.querySelector('#tau-dlg [data-k=fit-out]').textContent")
    assert out.startswith("Effective masses (automatic)") and "RELAXTIM" in out
    warn = drv.execute_script("return document.querySelector('#tau-dlg [data-k=fit-warn]').textContent")
    assert "names no path segments" in warn and "does not shift linearly" in warn
    # electrons, no DOSS: m_d is taken equal to m* (said in the hint and the warnings)
    drv.execute_script("const d = document.getElementById('tau-dlg'); d.querySelector('[data-k=carrier]').value = 'electron';"
                       "d.querySelector('[data-k=dos_file]').value = ''; d.querySelector('[data-k=fit-key]').textContent = '';"
                       "d.querySelector('[data-act=fit]').click();")
    H.wait(drv, "return /E1 /.test(document.querySelector('#tau-dlg [data-k=fit-key]').textContent)", 60)
    assert drv.execute_script("return document.querySelector('#tau-dlg [data-k=fm-hint]').textContent") == "from BAND: 0.250 mₑ (Γ→X)"
    assert drv.execute_script("return document.querySelector('#tau-dlg [data-k=md-hint]').textContent") == "m_d = m*: 0.250 mₑ"
    assert "isotropic-band assumption" in drv.execute_script("return document.querySelector('#tau-dlg [data-k=fit-warn]').textContent")
    U.shot(drv, "tau_auto_mstar")
    drv.execute_script("document.querySelector('#tau-dlg [data-close]').click()")
    assert not H.errors(drv)


def test_phonopy_dialog_jobs_and_plots(app):
    drv, base, root = app
    _open_dialog(drv, "analysis.phonopy", "phonopy_si_222", "phonopy-dlg")
    H.wait(drv, "return document.querySelector('#phonopy-dlg [data-k=jobs]').textContent.includes('si_ph222_disp-001')", 30)
    assert "not run" in drv.execute_script("return document.querySelector('#phonopy-dlg [data-k=jobs]').textContent")
    drv.execute_script("document.querySelector('#phonopy-dlg [data-tab=post]').click()")
    if not drv.execute_script("return !!window.Plotly") and not H.wait(drv, "return !!window.Plotly || null", 10):
        pytest.skip("plotly is not installed here")
    H.wait(drv, "return !!document.querySelector('#phonopy-dlg [data-k=band-plot] .main-svg')", 30)
    n = drv.execute_script("return document.querySelector('#phonopy-dlg [data-k=band-plot]').data.length")
    assert n == 6
    drv.execute_script("const u = document.querySelector('#phonopy-dlg [data-k=unit]'); u.value = 'cm';"
                       "u.dispatchEvent(new Event('change'));")
    H.wait(drv, "return /cm/.test(document.querySelector('#phonopy-dlg [data-k=band-plot]').layout.yaxis.title.text)")
    assert "300 K" in drv.execute_script("return document.querySelector('#phonopy-dlg [data-k=thermal-note]').textContent")
    drv.execute_script("document.querySelector('#phonopy-dlg [data-close]').click()")
    assert not H.errors(drv)


def test_phonopy_opens_from_the_sets_files_and_points_to_setup(app):
    drv, base, root = app
    reason = "return window.__mace.menuReason(arguments[0], arguments[1])"
    assert drv.execute_script(reason, "explorer.phonopy", {"path": "phonopy_si_222/band.yaml"}) == ""
    assert drv.execute_script(reason, "explorer.phonopy", {"dir": "phonopy_si_222"}) == ""
    assert drv.execute_script(reason, "explorer.phonopy", {"path": "si_opt.out"})
    drv.execute_script("window.__mace.run('explorer.phonopy', {path: 'phonopy_si_222/band.yaml'})")
    H.wait(drv, "const d = document.getElementById('phonopy-dlg'); return d && d.open")
    H.wait(drv, "return document.querySelector('#phonopy-dlg [data-k=set]').value === 'phonopy_si_222'", 30)
    # no generate step here: its pointer opens Setup › Phonopy on the set's parent
    assert [t for t in drv.execute_script("return [...document.querySelectorAll('#phonopy-dlg [data-tab]')].map((b) => b.textContent)")] == ["1 · Forces", "2 · Phonons"]
    drv.execute_script("document.querySelector('#phonopy-dlg [data-act=setup]').click()")
    H.wait(drv, f"return !document.getElementById('phonopy-dlg').open && !!document.querySelector('{SU} .su-mod[data-mod=PHONOPY]')", 60)
    assert not H.errors(drv)
