"""The bottom panel's Properties tab in headless Firefox (1.1.6): for the run
of the document in focus it lists every row materials.db holds for that
run (the example suite, extracted by the CLI), grouped, with units, labels
in markup and where each value came from; a plot file shows its run's rows;
a run not in the database shows its values computed on the fly and Extract
puts it in (the CLI command, a task); a run with no database says so and
offers Create Database…; CSV/JSON/Copy export the rows shown. A quantity
stored in several units (E_D3 in Hartree and eV) is one row; the unit preset,
a row's Unit cell (only that row changes) and per atom convert the values
shown, the export says the unit shown, and the choice survives a reload.
Nothing logs an error."""
import csv
import io
import json
import os
import sqlite3
import subprocess
import sys
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")


def _cli(cwd, *args):
    e = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    p = subprocess.run([sys.executable, str(H.REPO_ROOT / "mace_cli"), "--no-banner", *args], capture_output=True,
                       text=True, env=e, cwd=str(cwd), timeout=900)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    from mace import examples_suite as ES
    tmp = tmp_path_factory.mktemp("props_e2e")
    root = tmp / "root"
    ES.copy_suite(root / "suite")
    _cli(root, "analyze", "--extract-properties", "suite", "--db-path", "suite/materials.db")
    ec = next((root / "suite").rglob("EC_sp.out"))
    (root / "suite" / "extra").mkdir()
    (root / "suite" / "extra" / "ECX_sp.out").write_text(ec.read_text())
    (root / "alone").mkdir()
    (root / "alone" / "EC_sp.out").write_text(ec.read_text())
    a = U.start_app(tmp, width=1400, height=900, root=root)
    a[0].execute_script("window.__mace.settings.set('reduceMotion', 'on')")
    a[0].execute_script("document.querySelector('.shell').style.setProperty('--h-bottom', '360px')")
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture(autouse=True)
def _unfiltered(app):
    """Every test starts with the tab's filter empty: one that fails with it set (the masses' filter) left the
    next ones a filtered, empty table."""
    yield
    try:
        app[0].execute_script("const f = document.querySelector('.prp-filter'); "
                              "if (f && f.value) { f.value = ''; f.dispatchEvent(new Event('input')); }")
    except Exception:
        pass


def _rel(root, name):
    return next(p for p in (root / "suite").rglob(name)).relative_to(root).as_posix()


def _db_names(root, run_rel):
    from mace.utils import db_build
    conn = db_build.open_ro(root / "suite" / "materials.db")
    try:
        cid = db_build.calc_for_output(conn, (root / run_rel).resolve())
        names = [r[0] for r in conn.execute("SELECT property_name FROM properties WHERE calc_id = ? UNION ALL "
                                            "SELECT property_name FROM property_blobs WHERE calc_id = ?", (cid, cid))]
        return cid, sorted(names)
    finally:
        conn.close()


def _download(downloads, name, timeout=20):
    p = downloads / name
    end = time.time() + timeout
    while time.time() < end:
        if p.exists() and p.stat().st_size > 0 and not (downloads / (name + ".part")).exists():
            return p.read_text()
        time.sleep(0.2)
    raise AssertionError(f"{name} was not downloaded")


def _tab(drv):
    drv.execute_script("document.querySelector('.bottom-tab[data-bottom=properties]').click()")
    H.wait(drv, "return document.querySelector('.bottom-pane.properties').classList.contains('active')", 5)


def _shown(drv):
    """The database keys the rows show: a row's key, and the keys of the same quantity in other units it stands for."""
    return drv.execute_script("return [...document.querySelectorAll('.bottom-pane.properties .prp-table tr[data-name]')]"
                              ".flatMap((r) => [r.dataset.name, ...(r.dataset.also ? r.dataset.also.split(' ') : [])])")


def _text(drv):
    return drv.execute_script("return document.querySelector('.bottom-pane.properties').textContent")


def test_a_run_in_the_database_lists_every_row(app):
    drv, root = app[0], app[2]
    run = _rel(root, "diamond_band.out")
    U.open_doc(drv, run, wait_atoms=False)
    _tab(drv)
    H.wait(drv, "return document.querySelectorAll('.bottom-pane.properties .prp-table tr[data-name]').length > 0", 30)
    cid, names = _db_names(root, run)
    assert sorted(_shown(drv)) == names
    t = " ".join(_text(drv).split())
    assert "diamond_band.out · BAND · " + cid in t and "materials.db" in t
    # grouped, labels with markup, units, provenance
    groups = drv.execute_script("return [...document.querySelectorAll('.bottom-pane.properties .prp-gh th')].map((x) => x.textContent)")
    assert "Effective masses" in groups and "Run" in groups
    row = drv.execute_script("""const r = document.querySelector('.bottom-pane.properties tr[data-name=effective_mass_hole_mc]');
        return {sub: !!r.querySelector('.prp-label sub'), label: r.querySelector('.prp-label').textContent,
                unit: r.querySelector('.prp-unit').textContent, src: r.querySelector('.prp-src').textContent}""")
    assert row["sub"] and row["label"] == "Effective mass mc, holes (accurate)" and row["unit"] == "m_e"
    assert row["src"] == "diamond_band.out · extractor · effective_masses v1"
    assert "effective_masses stored" in t
    U.shot(drv, "properties_tab_band")
    # the filter
    # the filter (label, key and group): the masses' group, the accurate m_c / m_d with the extractor's quick
    # estimate real_*_effective_mass beside them
    drv.execute_script("const f = document.querySelector('.prp-filter'); f.value = 'effective masses'; f.dispatchEvent(new Event('input'))")
    H.wait(drv, "return document.querySelectorAll('.bottom-pane.properties tr[data-name]').length === 8", 5)
    assert sorted(_shown(drv)) == ["effective_mass_electron_mc", "effective_mass_electron_md", "effective_mass_hole_mc",
                                   "effective_mass_hole_md", "effective_masses", "real_average_effective_mass",
                                   "real_electron_effective_mass", "real_hole_effective_mass"]
    quick = drv.execute_script("return document.querySelector('.bottom-pane.properties tr[data-name=real_hole_effective_mass] .prp-label').textContent")
    assert "quick estimate" in quick
    drv.execute_script("const f = document.querySelector('.prp-filter'); f.value = ''; f.dispatchEvent(new Event('input'))")
    U.no_errors(drv)


def test_a_plot_shows_its_runs_properties_and_exports(app):
    drv, root, downloads = app[0], app[2], app[5]
    dat = _rel(root, "diamond_doss.DOSS.DAT")
    U.open_doc(drv, dat, wait_atoms=False)
    _tab(drv)
    H.wait(drv, "return /diamond_doss\\.out/.test(document.querySelector('.bottom-pane.properties .prp-where')?.textContent || '')", 30)
    run = _rel(root, "diamond_doss.out")
    _, names = _db_names(root, run)
    H.wait(drv, "return arguments[0] === [...document.querySelectorAll('.bottom-pane.properties .prp-table tr[data-name]')]"
                ".reduce((n, r) => n + 1 + (r.dataset.also ? r.dataset.also.split(' ').length : 0), 0)", 10, len(names))
    for f in downloads.glob("*"):
        f.unlink()
    drv.execute_script("document.querySelector('.bottom-pane.properties [data-a=csv]').click()")
    rows = list(csv.DictReader(io.StringIO(_download(downloads, "diamond_doss_properties.csv"))))
    assert sorted(n for r in rows for n in [r["name"], *r["also"].split()]) == names
    assert rows[0].keys() >= {"calc_type", "group", "label", "name", "value", "unit", "file", "extractor", "version",
                              "also", "stored_value", "stored_unit", "normalization"}
    drv.execute_script("document.querySelector('.bottom-pane.properties [data-a=json]').click()")
    doc = json.loads(_download(downloads, "diamond_doss_properties.json"))
    assert doc["state"] == "db" and doc["calc"]["calc_type"] == "DOSS"
    assert sum(1 + len(p["also"].split()) for p in doc["properties"]) == len(names)
    U.no_errors(drv)


def test_a_derived_fermi_level_is_said_on_the_plot_and_in_the_tab(app):
    """The example diamond's SCF ends spin-locked (SPIN's SPINLOCK): CRYSTAL writes no Fermi level (-1.0) into its
    BAND.DAT; the band plot takes the one mace.utils.fermi_level derives (the top of the occupied bands) and says
    so, and the run's Properties tab holds it with the same note."""
    drv, root = app[0], app[2]
    U.open_doc(drv, _rel(root, "diamond_band.BAND.DAT"), wait_atoms=False)
    H.wait(drv, "return /E_F derived/.test(document.querySelector('.pv-note .pv-fermi')?.textContent || '')", 60)
    _tab(drv)
    H.wait(drv, "return !!document.querySelector('.bottom-pane.properties tr[data-name=fermi_energy_note]')", 30)
    assert "E_F derived" in drv.execute_script(
        "return document.querySelector('.bottom-pane.properties tr[data-name=fermi_energy_note]').textContent")
    U.no_errors(drv)


def test_a_run_not_in_the_database_and_extract(app):
    drv, root = app[0], app[2]
    run = "suite/extra/ECX_sp.out"
    U.open_doc(drv, run, wait_atoms=False)
    _tab(drv)
    H.wait(drv, "return /Not in the database yet/.test(document.querySelector('.bottom-pane.properties').textContent)", 30)
    # the values computed on the fly, nothing stored
    H.wait(drv, "return document.querySelectorAll('.bottom-pane.properties .prp-table tr[data-name]').length > 0", 60)
    shown = set(_shown(drv))
    assert {"energy_per_atom_ev", "total_energy_au", "extraction_provenance"} <= shown
    assert "computed now (not stored)" in _text(drv)
    U.shot(drv, "properties_tab_not_in_db")
    assert "--output-file suite/extra/ECX_sp.out --db-path suite/materials.db --register --register-output-only --backup" in _text(drv)
    drv.execute_script("document.querySelector('.bottom-pane.properties .prp-banner [data-a=extract]').click()")
    H.wait(drv, "return /ECX_sp\\.out · SP · /.test(document.querySelector('.bottom-pane.properties .prp-where')?.textContent || '')", 120)
    cid, names = _db_names(root, run)
    assert cid and names
    H.wait(drv, "return arguments[0] === [...document.querySelectorAll('.bottom-pane.properties .prp-table tr[data-name]')]"
                ".reduce((n, r) => n + 1 + (r.dataset.also ? r.dataset.also.split(' ').length : 0), 0)", 10, len(names))
    assert set(names) == shown                     # what it computed is what the database now holds
    assert any(p.name.startswith("materials_backup") for p in (root / "suite" / "backups").glob("*"))
    U.no_errors(drv)


def test_no_database(app):
    drv = app[0]
    U.open_doc(drv, "alone/EC_sp.out", wait_atoms=False)
    _tab(drv)
    H.wait(drv, "return /No database/.test(document.querySelector('.bottom-pane.properties').textContent)", 30)
    H.wait(drv, "return document.querySelectorAll('.bottom-pane.properties .prp-table tr[data-name]').length > 0", 60)
    assert drv.execute_script("return !!document.querySelector('.bottom-pane.properties [data-a=create]')")
    assert "nothing is stored" in _text(drv)
    # a structure file has no run: the tab is dimmed with its note
    U.open_doc(drv, next(p for p in (app[2] / "suite").rglob("*.d12")).relative_to(app[2]).as_posix(), wait_atoms=False)
    U.no_errors(drv)


# ------------------------------------------------------------------ units
HA_EV, BOHR_A, EV_KJ = 27.211386245988, 0.529177210903, 96.48533212331002
CELLS_JS = """return Object.fromEntries([...document.querySelectorAll('.bottom-pane.properties .prp-table tr[data-name]')].map((tr) => {
    const s = tr.querySelector('select.prp-usel');
    return [tr.dataset.name, {val: tr.querySelector('.prp-val').textContent.trim(),
            unit: s ? s.selectedOptions[0].textContent : tr.querySelector('.prp-unit').textContent.trim(),
            choice: s ? s.value : null, own: !!(s && s.classList.contains('own'))}];
}))"""


def _cells(drv):
    return drv.execute_script(CELLS_JS)


def _choose(drv, css, value):
    drv.execute_script("const s = document.querySelector(arguments[0]); s.value = arguments[1];"
                       "s.dispatchEvent(new Event('change', {bubbles: true}))", css, value)


def _db_values(root, run_rel):
    from mace.utils import db_build
    conn = db_build.open_ro(root / "suite" / "materials.db")
    try:
        cid = db_build.calc_for_output(conn, (root / run_rel).resolve())
        return {n: v for n, v in conn.execute("SELECT property_name, property_value FROM properties WHERE calc_id = ?", (cid,))}
    finally:
        conn.close()


def _s(unit):
    """A unit's scale as the GUI gets it (mace/properties/units.catalog)."""
    from mace.properties import units
    return units.scale("energy", unit)


def _close(text, expect):
    assert float(text) == pytest.approx(expect, rel=2e-6), (text, expect)


def _same_table(drv):
    return drv.execute_script("return document.querySelector('.bottom-pane.properties .prp-table') === window.__prpTable")


def test_units_one_row_preset_cell_per_atom_export_and_reload(app):
    drv, root, downloads = app[0], app[2], app[5]
    run = _rel(root, "diamond_sp.out")
    U.open_doc(drv, run, wait_atoms=False)
    _tab(drv)
    H.wait(drv, "return !!document.querySelector('.bottom-pane.properties tr[data-name=d3_dispersion_energy_au] select.prp-usel')", 30)
    db = _db_values(root, run)
    _, names = _db_names(root, run)
    # one row for E_D3 (Hartree and eV in the database), every database key still accounted for
    assert sorted(_shown(drv)) == names
    assert drv.execute_script("return document.querySelector('.bottom-pane.properties tr[data-name=d3_dispersion_energy_au]').dataset.also") \
        == "d3_dispersion_energy_ev"
    assert not drv.execute_script("return document.querySelector('.bottom-pane.properties tr[data-name=d3_dispersion_energy_ev]')")
    c0 = _cells(drv)
    assert c0["d3_dispersion_energy_au"]["unit"] == "Hartree" and c0["d3_dispersion_energy_au"]["choice"] == "Hartree|total"
    # energies to CRYSTAL's printed precision (10 decimals in Hartree), other values to their significant digits
    assert c0["total_energy_au"]["val"] == f"{db['total_energy_au']:.10f}"
    assert c0["d3_dispersion_energy_au"]["val"] == f"{db['d3_dispersion_energy_au']:.10f}"
    # (no band_gap: the SCF ends with its spins locked, SPIN's SPINLOCK, and CRYSTAL prints no gap; the chemistry
    # preset's levels in eV: tests/test_property_display_units.py)
    assert c0["primitive_a"]["val"] == f"{db['primitive_a']:.6g}"
    assert c0["energy_per_atom_ev"]["val"] == f"{db['energy_per_atom_ev']:.9f}"
    label = drv.execute_script("return document.querySelector('.bottom-pane.properties tr[data-name=energy_per_atom_ev] .prp-label').textContent")
    assert label == "Etotal/atom (without D3 and gCP)"
    assert c0["spin_polarized"]["choice"] is None      # a yes/no value: no picker
    sel = drv.execute_script("const s = document.querySelector('.bottom-pane.properties tr[data-name=d3_dispersion_energy_au] select.prp-usel');"
                             "return {label: s.getAttribute('aria-label'), tab: s.tabIndex, n: s.options.length}")
    assert sel["label"] == "Unit of E_D3" and sel["tab"] >= 0 and sel["n"] >= 8 * 2
    drv.execute_script("window.__prpTable = document.querySelector('.bottom-pane.properties .prp-table')")
    # the preset converts every row it names, in place
    _choose(drv, ".bottom-pane.properties select[data-u=preset]", "solid")
    H.wait(drv, "return document.querySelector('.bottom-pane.properties tr[data-name=d3_dispersion_energy_au] select').value === 'eV|total'", 5)
    c1 = _cells(drv)
    # the GUI's converted text is the server's formatting of the same number (9 decimals in eV)
    assert c1["d3_dispersion_energy_au"]["val"] == f"{db['d3_dispersion_energy_au'] * _s('Hartree') / _s('eV'):.9f}"
    assert c1["total_energy_au"]["val"] == f"{db['total_energy_au'] * _s('Hartree') / _s('eV'):.9f}"
    _close(c1["total_energy_au"]["val"], db["total_energy_au"] * HA_EV)
    assert c1["primitive_a"] == c0["primitive_a"]
    assert _same_table(drv)
    # a row's Unit cell: only that row changes
    _choose(drv, ".bottom-pane.properties tr[data-name=primitive_a] select.prp-usel", "Bohr|total")
    c2 = _cells(drv)
    assert {k for k in c2 if c2[k] != c1[k]} == {"primitive_a"}
    _close(c2["primitive_a"]["val"], db["primitive_a"] / BOHR_A)
    assert c2["primitive_a"]["unit"] == "Bohr" and c2["primitive_a"]["own"]
    _choose(drv, ".bottom-pane.properties tr[data-name=d3_dispersion_energy_au] select.prp-usel", "meV|total")
    c3 = _cells(drv)
    assert {k for k in c3 if c3[k] != c2[k]} == {"d3_dispersion_energy_au"}
    assert c3["d3_dispersion_energy_au"]["val"] == f"{db['d3_dispersion_energy_au'] * _s('Hartree') / _s('meV'):.6f}"
    assert _same_table(drv)
    # a new preset sets every row again (the rows' own units go)
    _choose(drv, ".bottom-pane.properties select[data-u=preset]", "chemistry")
    c4 = _cells(drv)
    assert c4["primitive_a"]["unit"] == "Å" and not c4["primitive_a"]["own"]
    assert c4["d3_dispersion_energy_au"]["unit"] == "kJ/mol"
    _close(c4["d3_dispersion_energy_au"]["val"], db["d3_dispersion_energy_au"] * HA_EV * EV_KJ)
    # a row's own unit, then per atom: Normalize rescales every extensive row and keeps the rows' own units
    _choose(drv, ".bottom-pane.properties tr[data-name=total_energy_au] select.prp-usel", "Hartree|total")
    c4b = _cells(drv)
    assert c4b["total_energy_au"]["val"] == c0["total_energy_au"]["val"] and c4b["total_energy_au"]["own"]
    _choose(drv, ".bottom-pane.properties tr[data-name=primitive_a] select.prp-usel", "Bohr|total")
    _choose(drv, ".bottom-pane.properties select[data-u=norm]", "atom")
    c5 = _cells(drv)
    au = db["total_energy_au"]
    assert c5["total_energy_au"]["unit"] == "Hartree/atom" and c5["total_energy_au"]["own"]
    assert c5["total_energy_au"]["val"] == f"{au * _s('Hartree') / _s('Hartree') / 2:.10f}"
    assert c5["primitive_a"]["unit"] == "Bohr" and c5["primitive_a"]["own"]          # not extensive: its unit, not divided
    assert c5["d3_dispersion_energy_au"]["unit"] == "kJ/mol per atom"
    assert c5["d3_dispersion_energy_au"]["val"] == f"{db['d3_dispersion_energy_au'] * _s('Hartree') / _s('kJ/mol') / 2:.7f}"
    assert c5["primitive_cell_volume"]["unit"] == "Å³/atom"
    _close(c5["primitive_cell_volume"]["val"], db["primitive_cell_volume"] / 2)
    assert c5["energy_per_atom_ev"] == c4["energy_per_atom_ev"]
    groups = drv.execute_script("return [...document.querySelectorAll('.bottom-pane.properties tr[data-name=total_energy_au] optgroup')].map((g) => g.label)")
    assert groups == ["Total (per cell)", "Per atom", "Per formula unit (C, Z = 2)"]
    # the export says the unit shown, with the stored value and unit beside it
    for f in downloads.glob("*"):
        f.unlink()
    drv.execute_script("document.querySelector('.bottom-pane.properties [data-a=csv]').click()")
    rows = {r["name"]: r for r in csv.DictReader(io.StringIO(_download(downloads, "diamond_sp_properties.csv")))}
    d3 = rows["d3_dispersion_energy_au"]
    assert d3["unit"] == "kJ/mol per atom" and d3["normalization"] == "per atom"
    assert d3["also"] == "d3_dispersion_energy_ev" and d3["stored_unit"] == "Hartree"
    assert float(d3["value"]) == pytest.approx(db["d3_dispersion_energy_au"] * HA_EV * EV_KJ / 2, rel=1e-12)
    assert float(d3["stored_value"]) == db["d3_dispersion_energy_au"]
    assert rows["primitive_cell_volume"]["unit"] == "Å³/atom" and rows["primitive_cell_volume"]["normalization"] == "per atom"
    assert rows["total_energy_au"]["unit"] == "Hartree/atom" and rows["total_energy_au"]["normalization"] == "per atom"
    U.shot(drv, "properties_tab_units")
    # this browser remembers the preset, the normalization and the rows' own units
    st = json.loads(drv.execute_script("return localStorage.getItem('mace.gui.propunits.v1')"))
    assert st["preset"] == "chemistry" and st["norm"] == "atom"
    assert {k: v.split("|")[0] for k, v in st["over"].items()} == {"total_energy_au": "Hartree", "primitive_a": "Bohr"}
    drv.refresh()
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'", 60)
    drv.execute_script("window.__mace.settings.set('reduceMotion', 'on')")
    drv.execute_script("document.querySelector('.shell').style.setProperty('--h-bottom', '360px')")
    U.open_doc(drv, run, wait_atoms=False)
    _tab(drv)
    H.wait(drv, "return !!document.querySelector('.bottom-pane.properties tr[data-name=d3_dispersion_energy_au] select.prp-usel')", 30)
    assert drv.execute_script("return [document.querySelector('.bottom-pane.properties select[data-u=preset]').value,"
                              "document.querySelector('.bottom-pane.properties select[data-u=norm]').value]") == ["chemistry", "atom"]
    c7 = _cells(drv)
    assert c7["total_energy_au"]["unit"] == "Hartree/atom" and c7["total_energy_au"]["own"]
    assert c7["primitive_a"]["unit"] == "Bohr" and c7["primitive_a"]["own"]
    assert c7["d3_dispersion_energy_au"]["unit"] == "kJ/mol per atom"
    # back to total: the rows keep their own units
    _choose(drv, ".bottom-pane.properties select[data-u=norm]", "total")
    c8 = _cells(drv)
    assert c8["total_energy_au"]["unit"] == "Hartree" and c8["total_energy_au"]["own"]
    assert c8["total_energy_au"]["val"] == c0["total_energy_au"]["val"]
    # back to the stored units: the values the database holds, as before
    _choose(drv, ".bottom-pane.properties select[data-u=preset]", "stored")
    _choose(drv, ".bottom-pane.properties select[data-u=norm]", "total")
    assert _cells(drv) == c0
    U.no_errors(drv)
