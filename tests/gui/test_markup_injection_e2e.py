"""A crafted file name or CIF title cannot put markup into the page.

The served folder holds a CIF whose file name and data block are HTML, and a
text file with HTML inside. The explorer, the tabs, the window titles, the
inspector, a builder's task label, a toast and the text view must show them
as text: no element they name appears in the DOM. (The page's CSP blocks
inline handlers anyway; this checks the escaping itself.)"""
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")

EVIL = '<img src=x data-evil=1>'
CIF = f"""data_{EVIL.replace(' ', '_')}
_chemical_name_common '{EVIL}'
_cell_length_a 4.2120
_cell_length_b 4.2120
_cell_length_c 4.2120
_cell_angle_alpha 90.00
_cell_angle_beta 90.00
_cell_angle_gamma 90.00
_symmetry_space_group_name_H-M 'F m -3 m'
_symmetry_Int_Tables_number 225

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Mg1 Mg 0.0 0.0 0.0 1.0
O2 O 0.5 0.5 0.5 1.0
"""


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("inject")
    root = H.examples_copy(tmp)
    (root / "structures" / f"{EVIL}.cif").write_text(CIF)
    (root / f"{EVIL}notes.d3").write_text(f"{EVIL}\n<script>window.__evil = 1</script>\n")
    a = U.start_app(tmp, root=root)
    try:
        yield a
    finally:
        U.stop_app(a)


def _injected(drv):
    return drv.execute_script("return document.querySelectorAll('[data-evil], script:not([src])').length")


def test_crafted_names_and_titles_stay_text(app):
    drv = app[0]
    U.reset(drv)
    time.sleep(0.5)
    assert _injected(drv) == 0
    names = drv.execute_script("return [...document.querySelectorAll('#file-list .file')].map((f) => f.textContent)")
    assert any("<img" in n for n in names), names                  # listed, as text
    U.open_doc(drv, f"structures/{EVIL}.cif")
    assert _injected(drv) == 0
    assert "<img" in drv.execute_script("return document.title")
    # a builder's label and its task carry the document's title
    drv.execute_script("window.__mace.run('build.supercell')")
    time.sleep(1.0)
    assert _injected(drv) == 0
    drv.execute_script("document.querySelectorAll('dialog[open]').forEach((d) => d.close())")
    U.open_doc(drv, f"{EVIL}notes.d3", wait_atoms=False)
    time.sleep(1.0)
    assert _injected(drv) == 0
    assert drv.execute_script("return window.__evil") is None
    U.no_errors(drv)
