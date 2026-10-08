"""A molecule file in headless Firefox (plan E10): an .xyz (no cell) opens as a
molecule with no cell, and Calculation Setup sets it up as a MOLECULE without
being asked (CRYSTAL is impossible), the deck written by the CLI; a CIF of a
slab in a box shows the vacuum hint. Generic systems written here; runs in CI."""
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
np = pytest.importorskip("numpy")
pytest.importorskip("ase")
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.by import By  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    from ase import Atoms
    from ase.io import write
    tmp = tmp_path_factory.mktemp("dim_e2e")
    root = H.examples_copy(tmp)
    d = root / "dim"
    d.mkdir()
    water = Atoms("OH2", positions=[[0, 0, 0], [0.9572, 0, 0], [-0.2399872, 0.927, 0]])
    write(str(d / "water.xyz"), water, format="xyz")
    # a graphene-like layer with 20 A of vacuum along c: the slab hint
    a = 2.46
    layer = Atoms("C2", cell=[[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, 20.0]],
                  scaled_positions=[[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.5]], pbc=True)
    write(str(d / "layer.cif"), layer)
    a_ = U.start_app(tmp, root=root)
    try:
        yield a_
    finally:
        U.stop_app(a_)


def click_file(drv, path):
    from selenium.common.exceptions import StaleElementReferenceException
    drv.execute_script("window.MACE.state.filter = 'all'; window.MACE.state.text = ''; window.MACE.renderFiles()")
    for _ in range(6):
        try:
            el = drv.find_element(By.CSS_SELECTOR, f'#file-list .file[data-path="{path}"]')
            drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
            ActionChains(drv).move_to_element(el).click().perform()
            return
        except StaleElementReferenceException:
            time.sleep(0.2)
    raise AssertionError(path)


def test_xyz_opens_as_a_molecule_and_sets_up_as_one(app):
    drv = app[0]
    U.reset(drv)
    chip = drv.execute_script("return document.querySelector('#file-list .file[data-path=\"dim/water.xyz\"] .kb').textContent")
    assert chip == "xyz"
    click_file(drv, "dim/water.xyz")
    H.wait(drv, "const d = window.__mace.doc(); return d && d.path === 'dim/water.xyz' && d.structure", 30)
    s = drv.execute_script("const s = window.__mace.doc().structure; return {pbc: s.pbc, n: s.numbers.length, dim: s.dimensionality}")
    assert s == {"pbc": [False, False, False], "n": 3, "dim": "MOLECULE"}
    drv.execute_script("window.__mace.run('calc.setup', {open: true})")
    H.wait(drv, "return !!document.querySelector('#inspector .setup')", 30)
    H.wait(drv, """const m = window.__mace.setup, s = m && m.state(); return s && s.result && !s.busy && !s.pendingRun""", 180)
    st = drv.execute_script("const s = window.__mace.setup.state(); return {deck: s.result.deck, name: s.result.deck_name, label: s.result.source_label || ''}")
    assert "MOLECULE" in st["deck"].splitlines()[:4], st["deck"][:300]
    assert st["name"].startswith("water_MOLECULE_"), st["name"]
    U.shot(drv, "dim_xyz_setup")
    drv.execute_script("window.__mace.run('calc.setup')")
    U.no_errors(drv)


def test_slab_in_a_box_shows_the_hint(app):
    drv = app[0]
    U.reset(drv)
    U.open_doc(drv, "dim/layer.cif")
    hint = H.wait(drv, "const e = document.getElementById('s-dim-hint'); return e && e.textContent", 10)
    assert hint.startswith("Looks like a slab (vacuum ") and "along c" in hint, hint
    U.no_errors(drv)
