"""`mace gui` viewer data: reduced formulas, element colors, slabs, the JS gate."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import REPO_ROOT

pytest.importorskip("ase")
STATIC = REPO_ROOT / "mace" / "gui" / "static"
EXAMPLES = REPO_ROOT / "examples" / "gui"


@pytest.mark.parametrize("cell, formula, z", [
    ("Mg4O4", "MgO", 4), ("Li4F4", "LiF", 4), ("Na4Cl4", "NaCl", 4),
    ("Si4O8", "SiO2", 4), ("Ga4As4", "GaAs", 4), ("Ba1Ti1O3", "BaTiO3", 1),
    ("Cs1Pb1I3", "CsPbI3", 1), ("B2N2", "BN", 2), ("Ag4Br6", "Ag2Br3", 2),
    ("C8", "C", 8), ("Ca1C1O3", "CaCO3", 1), ("Na1H1C1O3", "NaHCO3", 1),
])
def test_reduced_formula_orders_by_electronegativity(cell, formula, z):
    from ase import Atoms
    from mace.gui.structures import reduced_formula
    assert reduced_formula(Atoms(cell).numbers) == (formula, z)


@pytest.mark.parametrize("text, expected", [
    ("F4Li4", ("LiF", 4)), ("Ag4Br6", ("Ag2Br3", 2)), ("Si2", ("Si", 2)),
    ("BN", ("BN", 1)), ("CHNaO3", ("NaHCO3", 1)),
    ("Fe(CO)5", None), ("Xx2", None), ("", None),
])
def test_stored_cell_formulas_reduce_for_display(text, expected):
    from mace.gui.structures import reduce_formula_text
    assert reduce_formula_text(text) == expected


def test_structure_payload_carries_reduced_formula_and_shared_bond_scale():
    from mace.gui import structures as S
    s = S.read_structure(EXAMPLES / "structures" / "MgO.cif")
    assert (s["reduced_formula"], s["z"], s["formula"]) == ("MgO", 4, "Mg4O4")
    assert s["bond_scale"] == S.BOND_SCALE
    js = (STATIC / "app.js").read_text()
    assert "s.bond_scale" in js and "1.15" not in js


def test_slab_cell_spans_the_atoms_not_the_vacuum():
    from mace.gui import structures as S
    s = S.read_structure(EXAMPLES / "outputs" / "graphene_slab_sp.out")
    assert s["pbc"] == [True, True, False]
    assert s["lattice"]["c"] < 1.0          # a monolayer: no 500 A or +6 A box
    assert len(s["origin"]) == 3


def test_element_colors_cover_every_element_in_each_scheme():
    from ase.data import chemical_symbols
    data = json.loads((STATIC / "element_colors.json").read_text())
    assert data["default"] in data["schemes"] and "pymatgen" in data["source"]
    assert re.fullmatch(r"#[0-9a-f]{6}", data["fallback"])
    for name, table in data["schemes"].items():
        missing = [s for s in chemical_symbols[1:104] if s not in table]
        assert not missing, (name, missing)
        assert all(re.fullmatch(r"#[0-9a-f]{6}", c) for c in table.values())
    # VESTA's Mg is orange, not Jmol's lime green
    assert data["schemes"]["vesta"]["Mg"] == "#fb7b15"
    assert "MIT" in (STATIC / "element_colors.LICENSE.txt").read_text()


def _own_js():
    return sorted(STATIC.glob("*.js")) + sorted((STATIC / "js").glob("*.js"))


def test_page_scripts_are_local_and_relative():
    html = (STATIC / "index.html").read_text()
    for url in re.findall(r'(?:src|href)="([^"]+)"', html):
        assert not url.startswith(("http:", "https:", "//", "/")), url
    shell = "".join(p.read_text() for p in (STATIC / "js").glob("*.js"))
    # a shared module imported from a document or lazy module (js/plotkit.js from js/docs/*.js)
    sub = "".join(p.read_text() for p in (STATIC / "js").glob("*/*.js"))
    for path in _own_js() + [STATIC / "element_colors.json"]:
        # the one outside address is a link the user follows (Help ▸ Documentation),
        # never a script, style or data load
        text = path.read_text().replace('"https://github.com/DjokicMa/MACE"', "")
        assert "http://" not in text and "https://" not in text, path.name
        if path.suffix != ".js":
            continue
        rel = path.relative_to(STATIC).as_posix()
        if rel == "login.js":
            # the sign-in page's one script, served without a session at <session path>/login.js
            assert "<script src='login.js'></script>" in (STATIC.parent / "server.py").read_text()
            continue
        # the page loads each script from the versioned (cache-forever) path,
        # or a shell module imports it
        assert f'src="static/@V@/{rel}"' in html or f'"./{path.name}"' in shell or \
            (path.parent == STATIC / "js" and f'"../{path.name}"' in sub), rel


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("path", _own_js(), ids=lambda p: p.name)
def test_javascript_parses(path, tmp_path):
    if path.parent.name == "js":  # ES modules: node checks them as .mjs
        tmp = tmp_path / (path.stem + ".mjs")
        tmp.write_text(path.read_text())
        path = tmp
    r = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr


def test_editor_themes_ship_their_mit_notices():
    """Every editor theme credited in tokens.css has its copyright notice and
    the MIT text in css/THEMES.LICENSE.txt (MIT asks for both)."""
    lic = (STATIC / "css" / "THEMES.LICENSE.txt").read_text()
    assert "Permission is hereby granted" in lic and "THE SOFTWARE IS PROVIDED" in lic
    for project in ("Catppuccin", "Dracula", "Gruvbox", "Nord", "One Dark", "Solarized", "Tokyo Night"):
        assert project in lic, project
    assert "THEMES.LICENSE.txt" in (STATIC / "css" / "tokens.css").read_text()
