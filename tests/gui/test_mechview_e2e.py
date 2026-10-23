"""ELASTIC, EOS, QHA and CPHF outputs open as their results document (1.1.6, js/docs/mech.js) in
headless Firefox, from the Explorer as a transport file opens as its plots, on the real HPCC runs of
tests/data/hpcc_116/mechanics (diamond ELASTCON with NUCHESS, PHOTOELA, EOS, QHA and CPHF; cubic BN
ELAPIEZO) and a slab's 2D constants:

* the Explorer's chips read ELASTIC, EOS, QHA and CPHF, outlined as data that opens;
* each opens a document in the plot area with the Results tab (the numbers of mace analyze elastic)
  and the run's plots: the directional Young's modulus and linear compressibility (a 3D surface,
  or polar cuts in the principal planes), E(V) and P(V), V(T), alpha(T), B(T) and the heat capacity;
* the Inspector's Calculation section names the module, NUMDERIV, STEPSIZE, the EOS volumes, the
  QHA temperatures and the functional;
* Copy CLI Command is the mace analyze elastic --plot line of what is shown;
* Analysis › Mechanical Properties Results… opens the same document; the Explorer's Open as
  Structure opens the geometry.
No JavaScript error on the way. MACE_MECHVIEW_SHOTS=<dir> saves a screenshot of each view.
"""
import os
import shutil
from pathlib import Path

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")

MECH = H.REPO_ROOT / "tests" / "data" / "hpcc_116" / "mechanics"
SHOTS = os.environ.get("MACE_MECHVIEW_SHOTS")
SLAB = """ SLAB CALCULATION
 LATTICE PARAMETERS (ANGSTROMS AND DEGREES) - BOHR = 0.5291772083 ANGSTROM
 PRIMITIVE CELL - CENTRING CODE 1/0 VOLUME=     5.240000 - DENSITY  0.000 g/cm^3
         A              B              C           ALPHA      BETA       GAMMA
     2.46000000     2.46000000   500.00000000    90.000000  90.000000 120.000000
 N. OF ATOMS PER CELL         2  COULOMB OVERLAP TOL         (T1) 10**   -7

 SYMMETRIZED ELASTIC CONSTANTS FOR HEXAGONAL     CASE, IN HARTREE

 |     1.1000    0.1900    0.0000 |
 |               1.1000    0.0000 |
 |                         0.4550 |

"""


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mechview")
    root = tmp / "root"
    root.mkdir()
    for stem in ("dia_elastcon", "dia_eos", "dia_qha", "dia_cphf", "bn_elapiezo", "dia_photoela"):
        for suf in (".out", ".d12"):
            shutil.copy(MECH / (stem + suf), root / (stem + suf))
    shutil.copy(MECH / "dia_cphf_running.out", root / "dia_cphf_running.out")
    shutil.copy(MECH / "dia_qha_full.out", root / "dia_qha_full.out")
    shutil.copy(H.REPO_ROOT / "tests" / "data" / "intcphf" / "agbr_freq.out", root / "agbr_freq.out")
    shutil.copy(H.REPO_ROOT / "tests" / "data" / "intcphf" / "ag2secl_freq.out", root / "ag2secl_freq.out")
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1")
    try:
        drv = H.firefox(1400, 900, download_dir=tmp)
        H.load(drv, base)
    except BaseException:
        proc.terminate()
        raise
    try:
        yield drv, root
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def _shot(drv, name):
    if SHOTS:
        Path(SHOTS).mkdir(parents=True, exist_ok=True)
        drv.save_screenshot(str(Path(SHOTS) / f"{name}.png"))


def _chip(drv, path):
    return H.wait(drv, "const b = document.querySelector(`#file-list .file[data-path=\"${arguments[0]}\"] .kb`);"
                       "return b && b.textContent !== 'out' && [b.textContent, b.className]", 60, path)


def _open(drv, path):
    row = drv.find_element("css selector", f'#file-list .file[data-path="{path}"]')
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", row)
    row.click()
    H.wait(drv, "const v = window.__mace.mechView; const s = v && v.state();"
                "return s && s.drawn === arguments[0] && document.querySelector('#mech-host').offsetParent !== null", 60, path)


def _tab(drv, key):
    drv.execute_script("document.querySelector(`#mech-host [data-tab=\"${arguments[0]}\"]`).click()", key)
    return H.wait(drv, "const s = window.__mace.mechView.state(); return s.tab === arguments[0] && s.drawn && s.drawn.split('|')[1] === arguments[0] && s",
                  60, key)


def _settings(drv):
    H.wait(drv, "return !!document.querySelector('#inspector .mv-settings table')", 30)
    return dict(drv.execute_script("return [...document.querySelectorAll('#inspector .mv-settings tr')].map((r) => [...r.cells].map((c) => c.textContent))"))


def _traces(drv):
    return drv.execute_script("const f = window.__mace.mechView.figure(); return f ? f.traces.map((t) => [t.type, t.name]) : []")


def test_chips_name_the_four_kinds_outlined(app):
    drv, _ = app
    for path, label in (("dia_elastcon.out", "ELASTIC"), ("bn_elapiezo.out", "ELASTIC"), ("dia_eos.out", "EOS"),
                        ("dia_qha.out", "QHA"), ("dia_cphf.out", "CPHF")):
        got, cls = _chip(drv, path)
        assert got == label and "plot-data" in cls, (path, got, cls)


def test_elastcon_results_plots_and_settings(app):
    drv, _ = app
    _open(drv, "dia_elastcon.out")
    st = drv.execute_script("return window.__mace.mechView.state()")
    assert st["kind"] == "ELASTIC" and st["tab"] == "results" and st["views"] == ["youngs", "compressibility", "shear", "poisson", "sound"]
    text = drv.execute_script("return window.__mace.mechView.body().textContent")
    # mace analyze elastic's numbers: C11 1150.388, K_H 465.24, the Born criteria of a cubic crystal, the Debye temperature
    for want in ("1150.388", "122.664", "618.258", "465.24", "C11 − C12 > 0", "C11 + 2 C12 > 0", "C44 > 0", "Debye temperature",
                 "Christoffel", "[1 1 1]", "isotropic"):
        assert want in text, want
    s = _settings(drv)
    assert s["Module"] == "ELASTCON" and s["NUMDERIV (points per strain)"] == "3" and s["STEPSIZE (strain step)"] == "0.01"
    assert s["Functional"] == "B3LYP-D3" and "NUCHESS" in s["Options"]
    assert drv.execute_script("return [...document.querySelectorAll('#inspector .insp-section')].map((x) => x.dataset.key)")[:2] == ["calculation", "run"]
    _shot(drv, "elastic_results")
    _tab(drv, "youngs")
    assert _traces(drv)[0][0] == "surface"
    assert drv.execute_script("return window.__mace.mechView.state().cli") == "mace analyze elastic dia_elastcon.out --plot youngs --shape surface"
    note = drv.execute_script("return document.querySelector('#mech-host .pv-note').textContent")
    assert "1126.7 GPa along [1 0 0]" in note and "1285.4 GPa along [1 1 1]" in note
    _shot(drv, "elastic_youngs_surface")
    drv.execute_script("document.querySelector('#mech-host [data-shape=planes]').click()")
    H.wait(drv, "return (window.__mace.mechView.state().drawn || '').endsWith('|planes')", 30)
    assert [t[0] for t in _traces(drv)] == ["scatterpolar"] * 3
    assert drv.execute_script("return window.__mace.mechView.state().cli").endswith("--plot youngs --shape planes")
    _shot(drv, "elastic_youngs_planes")
    _tab(drv, "compressibility")
    assert [t[0] for t in _traces(drv)] == ["scatterpolar"] * 3          # the shape is the document's
    _shot(drv, "elastic_compressibility_planes")
    drv.execute_script("document.querySelector('#mech-host [data-shape=surface]').click()")
    H.wait(drv, "return (window.__mace.mechView.state().drawn || '').endsWith('|surface')", 30)
    # the Export menu: PNG, SVG, CSV, Copy as TSV, Copy CLI Command
    items = drv.execute_script("document.querySelector('#mech-host .pk-export').click();"
                               "const v = [...document.querySelectorAll('.pk-menu button')].map((b) => b.dataset.v); document.querySelector('.pk-menu').close(); return v")
    assert items == ["png", "svg", "csv", "tsv", "cli"]
    t = drv.execute_script("return window.__mace.mechView.table()")
    assert t["columns"][-1] == "β (TPa^-1)" and len(t["rows"]) > 1000
    # the Explorer's Open as Structure: the output's geometry, beside its results document
    drv.execute_script("window.__mace.run('explorer.openAsStructure', {path: 'dia_elastcon.out'})")
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.path === 'dia_elastcon.out' && ['structure', 'trajectory'].includes(d.kind)", 60)
    assert drv.execute_script("return window.__mace.docs().filter((d) => d.kind === 'mech').length") == 1


def test_elapiezo_and_photoela_tensors(app):
    drv, _ = app
    _open(drv, "bn_elapiezo.out")
    text = drv.execute_script("return window.__mace.mechView.body().textContent")
    assert "Direct piezoelectric tensor e" in text and "Converse piezoelectric tensor d" in text
    assert _settings(drv)["Module"] == "ELAPIEZO"
    _shot(drv, "elapiezo_results")
    _open(drv, "dia_photoela.out")
    text = drv.execute_script("return window.__mace.mechView.body().textContent")
    assert "Elasto-optic (Pockels) tensor p" in text and "Piezo-optic tensor π" in text


def _shape(drv, shape):
    drv.execute_script("window.__mace.mechView.setShape(arguments[0])", shape)
    H.wait(drv, "return (window.__mace.mechView.state().drawn || '').endsWith('|' + arguments[0])", 30, shape)


def _scroll_to(drv, heading):
    drv.execute_script("const h = [...window.__mace.mechView.body().querySelectorAll('h3')].find((x) => x.textContent.startsWith(arguments[0]));"
                       "if (h) h.scrollIntoView({block: 'start'})", heading)


def test_elastic_shear_poisson_sound_piezo_and_the_new_numbers(app):
    """1.1.6 (mechmore): the anisotropy indices, Pugh, Cauchy, hardness and thermal conductivity, the
    directional table with G and nu, the Christoffel extremes, d_L(n) and k^2; the Shear modulus,
    Poisson's ratio, Sound velocities and Piezoelectric tabs."""
    drv, _ = app
    _open(drv, "bn_elapiezo.out")
    st = drv.execute_script("return window.__mace.mechView.state()")
    assert st["views"] == ["youngs", "compressibility", "shear", "poisson", "sound", "piezo"]
    text = drv.execute_script("return window.__mace.mechView.body().textContent")
    for want in ("Anisotropy", "Zener A", "Chung–Buessem", "Pugh's ratio B/G", "brittle", "Cauchy pressure C12 − C44",
                 "Vickers hardness HV (Chen)", "Intermetallics 19, 1275 (2011)", "Int. J. Refract. Met. Hard Mater. 33, 93 (2012)",
                 "κmin (Clarke)", "κmin (Cahill)", "Shear modulus G(n, m)", "Poisson's ratio ν(n, m)", "Fast quasi-shear qS1",
                 "Longitudinal piezoelectric coefficient dL(n)", "along [1 1 1]", "needs the static permittivity"):
        assert want in text, want
    # sub- and superscripts are markup, not literal tags
    assert "<sub>" not in text and "&lt;sub" not in drv.execute_script("return window.__mace.mechView.body().innerHTML")
    _scroll_to(drv, "Directional")
    _shot(drv, "elastic_directional_indices")
    _scroll_to(drv, "Longitudinal piezoelectric")
    _shot(drv, "elastic_piezo_results")
    drv.execute_script("window.__mace.mechView.setEpsilon('6.8')")
    H.wait(drv, "return window.__mace.mechView.body().textContent.includes('Largest k2')", 30)
    assert drv.execute_script("return window.__mace.mechView.state().cli") == "mace analyze elastic bn_elapiezo.out --epsilon 6.8"
    _scroll_to(drv, "Longitudinal piezoelectric")
    _shot(drv, "elastic_piezo_coupling")
    _tab(drv, "shear")
    _shape(drv, "surface")
    assert [t[0] for t in _traces(drv)] == ["surface", "surface"]
    note = drv.execute_script("return document.querySelector('#mech-host .pv-note').textContent")
    assert "along [1 0 1] (m [1 0 −1])" in note and "Green: the minimum over m" in note
    _shot(drv, "elastic_shear_surface")
    _shape(drv, "planes")
    assert [t[0] for t in _traces(drv)] == ["scatterpolar"] * 6
    assert drv.execute_script("return window.__mace.mechView.state().cli").endswith("--plot shear --shape planes")
    _shot(drv, "elastic_shear_planes")
    _tab(drv, "poisson")
    _shot(drv, "elastic_poisson_planes")
    _shape(drv, "surface")
    _shot(drv, "elastic_poisson_surface")
    _tab(drv, "sound")
    assert not drv.execute_script("return !!document.querySelector('#mech-host [data-shape]')")     # cuts only
    assert [n for _, n in _traces(drv)][:6] == ["qL", "qS1", "qS2", "qL group", "qS1 group", "qS2 group"]
    _shot(drv, "elastic_sound")
    _tab(drv, "piezo")
    _shape(drv, "surface")
    assert drv.execute_script("return window.__mace.mechView.figure().traces[0].type") == "surface"
    _shot(drv, "elastic_piezo_surface")
    _shape(drv, "planes")
    _shot(drv, "elastic_piezo_planes")
    assert not H.errors(drv)


def test_eos_fits_and_curves(app):
    drv, _ = app
    _open(drv, "dia_eos.out")
    text = drv.execute_script("return window.__mace.mechView.body().textContent")
    assert "Birch-Murnaghan 1947" in text and "464.52" in text and "11.1422" in text
    s = _settings(drv)
    assert s["Module"] == "EOS" and s["Volume range"].startswith("0.920 – 1.080") and s["Volumes"].startswith("7 ")
    _shot(drv, "eos_results")
    _tab(drv, "eos")
    tr = _traces(drv)
    assert [n for _, n in tr if n] [:5] == ["Murnaghan 1944", "Birch-Murnaghan 1947", "Poirier-Tarantola 1998", "Vinet 1987", "computed"]
    _shot(drv, "eos_ev")
    _tab(drv, "pv")
    assert drv.execute_script("return window.__mace.mechView.state().cli") == "mace analyze elastic dia_eos.out --plot pv"
    _shot(drv, "eos_pv")



def test_eos_pressure_plots_residuals_and_the_enthalpy_of_another_phase(app):
    """1.1.6 (mechmore): B(P), V/V0(P), the residuals with each fit's RMS, H(P), and Add phase…: the
    enthalpy difference to a denser carbon phase and the transition pressure where they cross."""
    from eos_synthetic import eos_output
    drv, root = app
    (root / "c_hp_eos.out").write_text(eos_output(10.80, -76.21522625, 480.0, 3.7))
    drv.execute_script("window.__mace.openMech('dia_eos.out')")          # open before, on its P(V) tab
    # (the results view loads with the first results document: drawn before its tabs are clicked)
    H.wait(drv, "const v = window.__mace.mechView; const s = v && v.state();"
                "return s && s.drawn && s.drawn.split('|')[0] === 'dia_eos.out' && document.querySelector('#mech-host').offsetParent !== null", 60)
    _tab(drv, "pv")
    drv.execute_script("window.__mace.mechView.showTab('results')")
    H.wait(drv, "return window.__mace.mechView.state().drawn === 'dia_eos.out'", 30)
    text = drv.execute_script("return window.__mace.mechView.body().textContent")
    assert "Fit quality" in text and "good" in text and "Z = 2" in text
    _shot(drv, "eos_results_quality")
    for view in ("bp", "vp", "residuals", "enthalpy"):
        _tab(drv, view)
        assert len(_traces(drv)) >= 4, view
        _shot(drv, f"eos_{view}")
    assert drv.execute_script("return window.__mace.mechView.figure().layout.yaxis.title.text") == "H − E<sub>0</sub> (eV/f.u.)"
    drv.execute_script("document.querySelector('#mech-host [data-addphase]').click()")
    H.wait(drv, "return !document.querySelector('#mech-host .mv-addphase').hidden", 10)
    inp = drv.find_element("css selector", "#mech-host .mv-phase-in")
    from selenium.webdriver.common.keys import Keys
    inp.send_keys("c_hp_eos.out" + Keys.ENTER)
    H.wait(drv, "const f = window.__mace.mechView.figure(); return f && f.traces.some((t) => t.name === 'c_hp_eos')", 30)
    note = drv.execute_script("return document.querySelector('#mech-host .pv-note').textContent")
    assert "c_hp_eos.out: Pt = 26.9" in note, note
    assert drv.execute_script("return window.__mace.mechView.state().cli") == "mace analyze elastic dia_eos.out --plot enthalpy --compare c_hp_eos.out"
    assert drv.execute_script("return window.__mace.mechView.figure().layout.annotations[0].text").startswith("P<sub>t</sub> = 26.9")
    _shot(drv, "eos_enthalpy_two_phases")
    drv.execute_script("document.querySelector('#mech-host [data-unphase]').click()")
    H.wait(drv, "const f = window.__mace.mechView.figure(); return f && !f.traces.some((t) => t.name === 'c_hp_eos')", 30)
    assert not H.errors(drv)


def test_qha_plots(app):
    drv, _ = app
    _open(drv, "dia_qha.out")
    s = _settings(drv)
    assert s["Module"] == "QHA" and s["Temperatures"] == "10 – 600 K, 60 points" and s["Volumes"] == "4"
    text = drv.execute_script("return window.__mace.mechView.body().textContent")
    assert "458.902" in text                          # B_T at 300 K
    _shot(drv, "qha_results")
    for view in ("volume", "vratio", "expansion", "bulk", "heat"):
        _tab(drv, view)
        assert _traces(drv), view
        _shot(drv, f"qha_{view}")
    assert [n for _, n in _traces(drv)][0] == "C<sub>p</sub> − C<sub>v</sub>"


def test_qha_gruneisen_free_energy_entropy_and_the_f_of_v_family(app):
    """1.1.6 (mechmore): the QHA run with each volume's harmonic block and the PVT relation: S, C_v,
    C_p and gamma along V(T) in Results; the V/V0, Gruneisen, F(T), S(T), heat capacity and F(V) tabs."""
    drv, _ = app
    drv.execute_script("window.__mace.openMech('dia_qha_full.out')")
    H.wait(drv, "const v = window.__mace.mechView; const s = v && v.state(); return s && s.kind === 'QHA' && s.drawn", 60)
    if drv.execute_script("return window.__mace.mechView.state().tab") != "results":
        drv.execute_script("window.__mace.mechView.showTab('results')")
    H.wait(drv, "return window.__mace.mechView.state().drawn === 'dia_qha_full.out'", 30)
    st = drv.execute_script("return window.__mace.mechView.state()")
    assert st["views"] == ["volume", "vratio", "expansion", "bulk", "heat", "gamma", "free", "entropy", "fv"]
    text = drv.execute_script("return window.__mace.mechView.body().textContent")
    # γ (mode) at the row the database's qha_gamma_mode_300k comes from (nearest 300 K), not the last row
    for want in ("Thermodynamic functions along V(T)", "γth", "γ (mode, 300 K)", "0.919", "Cp(300 K)", "Cv(300 K)", "S(300 K)"):
        assert want in text, want
    _shot(drv, "qha_full_results")
    for view in ("vratio", "gamma", "free", "entropy", "heat", "fv"):
        _tab(drv, view)
        assert _traces(drv), view
        _shot(drv, f"qha_full_{view}")
    names = [n for _, n in _traces(drv)]
    assert names[0] == "10 K" and names[-2:] == ["minimum F(V) (V(T))", "computed volumes"]
    assert drv.execute_script("return window.__mace.mechView.state().cli") == "mace analyze elastic dia_qha_full.out --plot fv"
    assert not H.errors(drv)


def test_cphf_tensors(app):
    drv, _ = app
    _open(drv, "dia_cphf.out")
    st = drv.execute_script("return window.__mace.mechView.state()")
    assert st["kind"] == "CPHF" and st["views"] == ["indicatrix"]        # a static run: its index ellipsoid only
    text = drv.execute_script("return window.__mace.mechView.body().textContent")
    for want in ("Dielectric tensor ε∞", "5.3870", "26.2346", "2.3210", "Second hyperpolarizability γ"):
        assert want in text, want
    assert _settings(drv)["Frequency"] == "static (ω = 0)"
    _shot(drv, "cphf_results")
    # a CPHF run still going: no results yet, and the way to its text and structure
    _open(drv, "dia_cphf_running.out")
    assert "No elastic" in drv.execute_script("return window.__mace.mechView.body().textContent")


def test_cphf_principal_axes_index_ellipsoid_shg_and_n_of_omega(app):
    """1.1.6 (mechmore): a uniaxial, non-centrosymmetric crystal's principal axes, birefringence and
    SHG d_eff; the Index ellipsoid and SHG tabs; a DYNAMIC run's n(ω) and absorption tabs."""
    from cphf_synthetic import cphf_output
    drv, root = app
    (root / "uni_cphf.out").write_text(cphf_output({"XX": 4.0, "YY": 4.0, "ZZ": 4.5}, {"XYZ": 10.0}))
    (root / "dyn_cphf.out").write_text(cphf_output({0.0: {"XX": 4.0, "YY": 4.0, "ZZ": 4.5}, 1.0: {"XX": 4.2, "YY": 4.2, "ZZ": 4.8},
                                                    2.0: {"XX": 4.6, "YY": 4.6, "ZZ": 5.3}}, None, (0.0, 1.0, 2.0),
                                                   {0.0: {}, 1.0: {"XX": 0.5, "YY": 0.5, "ZZ": 0.6}, 2.0: {"XX": 1.4, "YY": 1.4, "ZZ": 1.7}}))
    drv.execute_script("window.__mace.openMech('uni_cphf.out')")
    H.wait(drv, "const v = window.__mace.mechView; const s = v && v.state(); return s && s.drawn === 'uni_cphf.out'", 60)
    assert drv.execute_script("return window.__mace.mechView.state().views") == ["indicatrix", "shg"]
    text = drv.execute_script("return window.__mace.mechView.body().textContent")
    # a tetragonal crystal: Cartesian direction labels (its lattice [111] is not x+y+z)
    for want in ("Principal axes of ε∞", "uniaxial", "optically positive; optic axis z", "Second-harmonic generation",
                 "11.55 pm/V along x+y+z", "Powder"):
        assert want in text, want
    _scroll_to(drv, "Principal axes")
    _shot(drv, "cphf_principal_shg")
    _tab(drv, "indicatrix")
    assert _traces(drv)[0][0] == "surface"
    assert "uniaxial, Δn = 0.1213" in drv.execute_script("return document.querySelector('#mech-host .pv-note').textContent")
    _shot(drv, "cphf_indicatrix")
    _tab(drv, "shg")
    _shape(drv, "planes")
    assert [n for _, n in _traces(drv)][:2] == ["|d(n)|", "d<sub>L</sub>(n)"]
    _shot(drv, "cphf_shg_planes")
    _shape(drv, "surface")
    _shot(drv, "cphf_shg_surface")
    drv.execute_script("window.__mace.openMech('dyn_cphf.out')")
    H.wait(drv, "const v = window.__mace.mechView; const s = v && v.state(); return s && s.drawn === 'dyn_cphf.out'", 60)
    assert drv.execute_script("return window.__mace.mechView.state().views") == ["epsilon", "refractive", "absorption", "indicatrix"]
    for view in ("refractive", "absorption"):
        _tab(drv, view)
        _shot(drv, f"cphf_{view}")
    assert not H.errors(drv)


def test_the_freq_view_opens_the_born_charges_and_lo_to(app):
    """1.1.6 (mechmore): a FREQ output with INTCPHF opens as its normal modes; the Dielectric button
    opens its Born charges, eps_inf and eps_0 and the computed LO-TO splitting."""
    drv, _ = app
    drv.execute_script("window.__mace.openModes('agbr_freq.out')")
    H.wait(drv, "const b = document.querySelector('#freq-host .fv-diel'); return b && !b.hidden", 90)
    _shot(drv, "freq_dielectric_button")
    drv.execute_script("document.querySelector('#freq-host .fv-diel').click()")
    H.wait(drv, "const v = window.__mace.mechView; const s = v && v.state(); return s && s.kind === 'FREQ' && s.drawn === 'agbr_freq.out'", 60)
    text = drv.execute_script("return window.__mace.mechView.body().textContent")
    for want in ("Born effective charges Z*", "1.1082", "−1.1082", "ε0 = ε∞ + εvib", "LO-TO splitting at Γ", "165.3", "Lyddane–Sachs–Teller"):
        assert want in text, want
    assert drv.execute_script("return window.__mace.activeDoc().title").endswith("· FREQ")
    _shot(drv, "freq_born_lo_to")
    _scroll_to(drv, "LO-TO")
    _shot(drv, "freq_lo_to")
    assert not H.errors(drv)


def test_a_cphf_that_breaks_the_crystal_symmetry_is_flagged(app):
    """Ag2SeCl (P4mm, tetragonal): eps_inf biaxial, an unstable CPHF and an overflowed beta table. The
    document says so (the report's WARNING lines) instead of a plain biaxial crystal."""
    drv, _ = app
    drv.execute_script("window.__mace.openMech('ag2secl_freq.out')")
    H.wait(drv, "const v = window.__mace.mechView; const s = v && v.state(); return s && s.drawn === 'ag2secl_freq.out'", 60)
    warn = drv.execute_script("return [...window.__mace.mechView.body().querySelectorAll('.mv-warn')].map((e) => e.textContent)")
    assert any(w.startswith("Warning: eps_inf is biaxial") and "P 4 M M is tetragonal, so it must be uniaxial" in w for w in warn)
    assert any("first hyperpolarizability beta diverged" in w for w in warn)
    assert any("DIIS test reached 9.51e+07" in w for w in warn)
    _shot(drv, "freq_symmetry_mismatch")
    assert not H.errors(drv)


def test_a_slab_draws_its_in_plane_moduli(app):
    drv, root = app
    (root / "graphene_elastcon.out").write_text(SLAB)
    H.wait(drv, "return !!document.querySelector('#file-list .file[data-path=\"graphene_elastcon.out\"]')", 30)
    drv.execute_script("window.__mace.openMech('graphene_elastcon.out')")
    H.wait(drv, "const v = window.__mace.mechView; const s = v && v.state(); return s && s.drawn === 'graphene_elastcon.out'", 30)
    text = drv.execute_script("return window.__mace.mechView.body().textContent")
    assert "In-plane stiffness" in text and "N/m" in text and "C11 C22 − C12² > 0" in text
    _tab(drv, "youngs")
    assert [t[0] for t in _traces(drv)] == ["scatterpolar"]
    assert not drv.execute_script("return !!document.querySelector('#mech-host [data-shape]')")     # no 3D surface for a slab
    assert drv.execute_script("return window.__mace.mechView.state().cli") == "mace analyze elastic graphene_elastcon.out --plot youngs"
    _shot(drv, "slab_youngs")


def test_the_results_dialog_opens_the_same_document(app):
    drv, _ = app
    drv.execute_script("window.__mace.run('analysis.elastic', {path: 'dia_eos.out'})")
    H.wait(drv, "const d = document.getElementById('elastic-dlg'); return d && d.open && d.querySelector('[data-k=r-body] table')", 30)
    drv.execute_script("document.querySelector('#elastic-dlg [data-act=doc]').click()")
    H.wait(drv, "const s = window.__mace.mechView.state(); return !document.getElementById('elastic-dlg').open && s && s.kind === 'EOS'", 30)
    assert drv.execute_script("return window.__mace.mechView.state().drawn").startswith("dia_eos.out")
    assert not H.errors(drv)


def test_closing_a_results_document_lets_go_of_it(app):
    """A results document closed: the view lets go of it (it kept the closed one); one closed behind the
    document shown leaves the shown one alone."""
    drv, _ = app
    drv.execute_script("window.__mace.closeAll()")
    drv.execute_script("window.__mace.openMech('dia_qha_full.out')")
    H.wait(drv, "const v = window.__mace.mechView; const s = v && v.state(); return s && s.kind === 'QHA'", 60)
    qha = drv.execute_script("return window.__mace.doc().id")
    drv.execute_script("window.__mace.openMech('dia_eos.out')")
    H.wait(drv, "const s = window.__mace.mechView.state(); return s && s.kind === 'EOS'", 60)
    drv.execute_script(f"document.querySelector('#doc-tabs [data-close=\"{qha}\"]').click()")
    H.wait(drv, f"return !document.querySelector('#doc-tabs [data-doc=\"{qha}\"]')", 10)
    assert drv.execute_script("const s = window.__mace.mechView.state(); return s && s.kind") == "EOS"
    drv.execute_script("window.__mace.closeAll()")
    H.wait(drv, "return !document.querySelector('#doc-tabs .doc-tab')", 10)
    assert drv.execute_script("return window.__mace.mechView.state()") is None
    assert not H.errors(drv), H.errors(drv)


def test_mechanics_plot_colors_are_theme_tokens():
    """ELATE's envelope colors and the piezoelectric curve come from the theme (light and dark): no
    hard-coded color in mech.js."""
    src = (H.REPO_ROOT / "mace" / "gui" / "static" / "js" / "docs" / "mech.js").read_text()
    import re
    assert not re.findall(r"[\"']#[0-9a-fA-F]{3,8}[\"']", src)
    assert 'const ENV = { min: "--success", max: "--info", neg: "--danger" };' in src
