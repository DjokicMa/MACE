"""Add Reference… in the Properties tab, in headless Firefox: on the example
suite (extracted by the CLI), 4LG_AA's reaction energy from graphene. The
picker lists the database's runs that fit (graphene ok, spin-polarized as
4LG_AA; diamond refused and not selectable), shows the command it runs;
Compute and Save runs the extraction with --reference (a task), and the tab
then shows reaction_energy stored with its value, the database holding it
and the choice saved (reference_runs, by mace-reference). The picker opens
again with the saved run ticked; Use the Database's Choice forgets it.
Nothing logs an error."""
import json
import os
import sqlite3
import subprocess
import sys

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")

EV = 27.211386245988


def _cli(cwd, *args):
    e = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    p = subprocess.run([sys.executable, str(H.REPO_ROOT / "mace_cli"), "--no-banner", *args], capture_output=True,
                       text=True, env=e, cwd=str(cwd), timeout=900)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    from mace import examples_suite as ES
    tmp = tmp_path_factory.mktemp("refs_e2e")
    root = tmp / "root"
    ES.copy_suite(root / "suite")
    _cli(root, "analyze", "--extract-properties", "suite", "--db-path", "suite/materials.db")
    a = U.start_app(tmp, width=1400, height=900, root=root)
    a[0].execute_script("window.__mace.settings.set('reduceMotion', 'on')")
    a[0].execute_script("document.querySelector('.shell').style.setProperty('--h-bottom', '360px')")
    try:
        yield a
    finally:
        U.stop_app(a)


def _db(root, sql, *args):
    conn = sqlite3.connect(root / "suite" / "materials.db")
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


def _calc(root, material):
    return _db(root, "SELECT calc_id FROM calculations WHERE material_id = ? AND calc_type = 'OPT'", material)[0][0]


def _open_picker(drv):
    drv.execute_script("document.querySelector('.bottom-pane.properties li[data-derived=reaction_energy] [data-addref]').click()")
    H.wait(drv, "return !!document.querySelector('#prp-refs[open] .prp-ref-table')", 30)


def test_add_reference_computes_and_saves_a_reaction(app):
    drv, root = app[0], app[2]
    run = next((root / "suite").rglob("4LG_AA_opt.out")).relative_to(root).as_posix()
    graphene, diamond, four = _calc(root, "graphene"), _calc(root, "diamond"), _calc(root, "4LG_AA")
    U.open_doc(drv, run, wait_atoms=False)
    drv.execute_script("document.querySelector('.bottom-tab[data-bottom=properties]').click()")
    H.wait(drv, "return !!document.querySelector('.bottom-pane.properties li[data-derived=reaction_energy] [data-addref]')", 60)
    assert drv.execute_script("return !!document.querySelector('.bottom-pane.properties [data-a=addref]')")
    _open_picker(drv)
    sel = drv.execute_script("return document.querySelector('#prp-refs [data-prop]').value")
    assert sel == "reaction_energy"
    rows = drv.execute_script("""return Object.fromEntries([...document.querySelectorAll('#prp-refs fieldset[data-role=reactants] tr[data-cid]')]
        .map((r) => [r.dataset.cid, {check: r.querySelector('.prp-ref-check').textContent.trim(),
                     disabled: r.querySelector('input[data-role]').disabled}]))""")
    assert graphene in rows and rows[graphene]["check"].startswith("ok") and not rows[graphene]["disabled"]
    assert rows[diamond]["check"].startswith("refused") and rows[diamond]["disabled"]
    assert four not in rows                                              # not the run itself
    U.shot(drv, "properties_add_reference")
    drv.execute_script("document.querySelector(`#prp-refs fieldset[data-role=reactants] input[value='${arguments[0]}']`).click()",
                       graphene)
    H.wait(drv, "return document.querySelector('#prp-refs .prp-refs-cli').textContent.includes('--reference reactants=' + arguments[0])",
           5, graphene)
    cli = drv.execute_script("return document.querySelector('#prp-refs .prp-refs-cli').textContent")
    assert cli.startswith("mace analyze --extract-properties --output-file " + run)
    drv.execute_script("document.querySelector('#prp-refs [data-r=save]').click()")
    # the toast says so (waited for at once: a toast does not stay)
    H.wait(drv, "return [...document.querySelectorAll('#toasts > *')].some((t) => t.textContent.includes("
                "'reaction_energy: computed and stored'))", 120)
    H.wait(drv, "return /reaction_energy\\s+stored/.test(document.querySelector('.bottom-pane.properties .prp-derived')?.textContent || '')", 120)
    H.wait(drv, "return !!document.querySelector('.bottom-pane.properties tr[data-name=reaction_energy_ev]')", 30)
    # the database holds it, and the choice
    e4 = _db(root, "SELECT property_value FROM properties WHERE calc_id = ? AND property_name = 'total_energy_corrected_au'", four)[0][0]
    eg = _db(root, "SELECT property_value FROM properties WHERE calc_id = ? AND property_name = 'total_energy_corrected_au'", graphene)[0][0]
    val = _db(root, "SELECT property_value FROM properties WHERE calc_id = ? AND property_name = 'reaction_energy_ev'", four)[0][0]
    assert val == pytest.approx((e4 - 48 * eg) * EV, rel=1e-12)
    (text, script), = _db(root, "SELECT property_value_text, extractor_script FROM properties WHERE calc_id = ? "
                                "AND property_name = 'reference_runs'", four)
    assert json.loads(text) == {"reaction_energy": {"reactants": [{"calc_id": graphene}]}} and script == "mace-reference"
    shown = drv.execute_script("return document.querySelector('.bottom-pane.properties tr[data-name=reaction_energy_ev] .prp-val').textContent")
    assert float(shown) == pytest.approx(val, rel=1e-5)
    # the picker again: the saved run is ticked; Use the Database's Choice forgets it
    _open_picker(drv)
    assert drv.execute_script("return document.querySelector(`#prp-refs input[value='${arguments[0]}']`).checked", graphene)
    drv.execute_script("document.querySelector('#prp-refs [data-r=auto]').click()")
    H.wait(drv, "return /reaction_energy\\s+skipped/.test(document.querySelector('.bottom-pane.properties .prp-derived')?.textContent || '')", 120)
    assert not _db(root, "SELECT 1 FROM properties WHERE calc_id = ? AND property_name IN ('reference_runs', 'reaction_energy_ev')", four)
    U.no_errors(drv)
