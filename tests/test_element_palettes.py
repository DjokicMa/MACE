"""Element color palettes (1.1.6): the one loader the GUI and the CLI share.

mace/utils/element_palettes.py reads the built-in presets and the user's
$XDG_CONFIG_HOME/mace/element_colors.json; the GUI writes that file
(mace/gui/palettes.py, /api/palettes) and `mace plotting` draws atoms in the
saved default palette, or in --palette NAME. The last test runs the real
`mace_cli plotting --structure` as a subprocess and reads the SVG's fills.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from mace.utils import element_palettes as EP

REPO = Path(__file__).resolve().parent.parent
PRESETS = json.loads((REPO / "mace" / "gui" / "static" / "element_colors.json").read_text())
MGO = REPO / "examples" / "gui" / "structures" / "MgO.cif"
BEFORE = "daa5468e"      # the 1.1.5 base: Plotting/plottingCIFs.py as it was before element palettes (1.1.6)


@pytest.fixture
def xdg(tmp_path, monkeypatch):
    d = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(d))
    EP.choose(None)
    yield d
    EP.choose(None)


def write(xdg, doc):
    p = xdg / "mace" / "element_colors.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))
    return p


def test_presets_cover_every_element_and_name_their_source():
    assert EP.preset_ids() == ["vesta", "cvd", "jmol", "legacy"]   # the MACE default first
    for pid in EP.preset_ids():
        colors = EP.resolve(pid, EP._empty())
        assert len(colors) == 103 and all(re.fullmatch(r"#[0-9a-f]{6}", c) for c in colors.values()), pid
    assert EP.label_of("vesta") == "VESTA (MACE default)" and EP.label_of("jmol") == "Jmol (CPK)"
    lic = (REPO / "mace" / "gui" / "static" / "element_colors.LICENSE.txt").read_text()
    assert "Okabe" in lic and "MIT" in lic and "cvd" in lic
    # the colorblind-safe scheme: Okabe-Ito hues on the most common elements
    cvd = PRESETS["schemes"]["cvd"]
    assert (cvd["O"], cvd["N"], cvd["S"], cvd["P"]) == ("#d55e00", "#0072b2", "#f0e442", "#e69f00")
    assert len(set(cvd[s] for s in ("H", "C", "N", "O", "S", "P", "Cl", "Pb", "I", "Cs", "Ti", "Si", "Fe"))) == 13


def test_the_generator_makes_the_committed_cvd_scheme():
    sys.path.insert(0, str(REPO / "mace" / "gui" / "tools"))
    try:
        import make_element_colors as mk
    finally:
        sys.path.pop(0)
    assert mk.cvd_scheme() == PRESETS["schemes"]["cvd"]
    assert mk.LABELS == PRESETS["labels"]
    assert mk.legacy_scheme() == PRESETS["schemes"]["legacy"]


def test_no_file_means_vesta_as_in_the_gui(xdg):
    assert EP.read_user()["palettes"] == {} and EP.default_id() is None
    assert EP.load_colors() == PRESETS["schemes"]["vesta"] == EP.active_colors()
    assert EP.load_colors("jmol")["Mg"] == PRESETS["schemes"]["jmol"]["Mg"]
    assert EP.load_colors("legacy") is None               # each plotter draws its own pre-1.1.6 table


def test_save_rename_default_delete(xdg):
    doc, made = EP.apply_op(None, {"op": "save", "name": "  My  oxides ", "base": "vesta",
                                   "colors": {"O": "#D55E00", "Ti": "#5bE", "Mg": PRESETS["schemes"]["vesta"]["Mg"]}})
    # names are tidied, #rgb grows to #rrggbb, a color equal to the base's is no change
    assert made == {"id": "user:My oxides"}
    assert doc["palettes"] == {"My oxides": {"base": "vesta", "colors": {"O": "#d55e00", "Ti": "#55bbee"}}}
    doc, _ = EP.apply_op(doc, {"op": "default", "id": "user:My oxides"})
    doc, made = EP.apply_op(doc, {"op": "rename", "name": "My oxides", "to": "Oxides"})
    assert made["id"] == "user:Oxides" and doc["default"] == "user:Oxides"     # the default follows a rename
    write(xdg, doc)
    full = EP.resolve("user:Oxides")
    assert full["O"] == "#d55e00" and full["Na"] == PRESETS["schemes"]["vesta"]["Na"] and len(full) == 103
    assert EP.load_colors() == full and EP.load_colors("oxides") == full and EP.load_colors("VESTA (MACE default)")["O"] != "#d55e00"
    doc, _ = EP.apply_op(doc, {"op": "delete", "name": "Oxides"})
    assert doc["palettes"] == {} and doc["default"] is None                   # deleting the default clears it
    assert doc["rev"] == 4
    for bad in ({"op": "save", "name": "", "colors": {}}, {"op": "save", "name": "x", "colors": {"Xx": "#000000"}},
                {"op": "save", "name": "x", "colors": {"O": "red"}}, {"op": "save", "name": "x", "base": "nope"},
                {"op": "rename", "name": "gone", "to": "y"}, {"op": "default", "id": "user:gone"}, {"op": "frobnicate"}):
        with pytest.raises(ValueError):
            EP.apply_op(doc, bad)


def test_a_hand_edited_file_never_breaks_the_rest(xdg, capsys):
    write(xdg, {"version": 1, "default": "user:Gone", "palettes": {
        "Good": {"base": "jmol", "colors": {"C": "#111111"}},
        "Bad": {"base": "jmol", "colors": {"C": "black"}},
        "Worse": "not an object"}})
    doc = EP.read_user()
    assert list(doc["palettes"]) == ["Good"] and doc["default"] is None
    assert "skipped" in capsys.readouterr().err
    (xdg / "mace" / "element_colors.json").write_text("{not json")
    assert EP.read_user()["palettes"] == {}


def test_export_import_round_trip(xdg):
    doc, _ = EP.apply_op(None, {"op": "save", "name": "Shared", "base": "cvd", "colors": {"Pb": "#123456"}})
    out = EP.export_doc("user:Shared", doc)
    assert out["mace_element_palette"] == 1 and out["base"] == "cvd" and out["colors"] == {"Pb": "#123456"}
    assert len(out["elements"]) == 103 and out["elements"]["Pb"] == "#123456"
    doc, made = EP.apply_op(doc, {"op": "import", "palette": json.loads(json.dumps(out))})
    assert made["id"] == "user:Shared (2)"                                   # a name in use gets a number
    assert EP.resolve("user:Shared (2)", doc) == EP.resolve("user:Shared", doc)
    # a table from another program: every element given, no "colors" -> the differing ones become the changes
    other = {"mace_element_palette": 1, "name": "Other", "elements": {**PRESETS["schemes"]["vesta"], "O": "#00ff00"}}
    name, pal = EP.parse_import(other)
    assert (name, pal) == ("Other", {"base": "vesta", "colors": {"O": "#00ff00"}})
    with pytest.raises(ValueError):
        EP.parse_import({"name": "no marker"})


def test_cli_plotters_read_the_saved_default(xdg):
    """FREQ modes and cube overlays (the engines) and the structure SVGs use the shared loader."""
    pytest.importorskip("plotly")                                           # vibmode_viewer draws with plotly
    from mace.plotting.engines import vibmode_viewer as vib
    from mace.plotting.engines import crystal_cubeviz_plotly as cube
    vesta = PRESETS["schemes"]["vesta"]
    assert vib._atom_color("O") == vesta["O"]                               # nothing saved: VESTA, as the GUI
    assert cube.get_atom_properties()[8]["color"] == vesta["O"]
    EP.choose("legacy")                                                     # the old colors, exactly
    assert vib._atom_color("O") == vib.ELEMENT_COLORS["O"] and vib._atom_outline("O") == vib.ELEMENT_OUTLINE_COLORS["O"]
    assert cube.get_atom_properties()[8]["color"] == "red"
    EP.choose(None)
    doc, _ = EP.apply_op(None, {"op": "save", "name": "Mine", "base": "jmol", "colors": {"O": "#abcdef"}})
    doc, _ = EP.apply_op(doc, {"op": "default", "id": "user:Mine"})
    write(xdg, doc)
    assert vib._atom_color("O") == "#abcdef" and vib._atom_color("MG") == PRESETS["schemes"]["jmol"]["Mg"]
    assert vib._atom_outline("O") == EP.darker("#abcdef", 0.55)
    assert cube.get_atom_properties()[8]["color"] == "#abcdef"
    EP.choose("cvd")                                                          # --palette wins over the default
    assert vib._atom_color("O") == PRESETS["schemes"]["cvd"]["O"]
    sys.path.insert(0, str(REPO / "Plotting"))
    try:
        import plottingCIFs
        from ase.io import read
        _, _, colors, _ = plottingCIFs.get_atoms_data(read(str(MGO)))
    finally:
        sys.path.pop(0)
    assert set(colors) == {PRESETS["schemes"]["cvd"]["Mg"], PRESETS["schemes"]["cvd"]["O"]}


def _plot(args, xdg, tmp_path):
    env = {**__import__("os").environ, "XDG_CONFIG_HOME": str(xdg), "TMPDIR": str(tmp_path)}
    return subprocess.run([sys.executable, str(REPO / "mace_cli"), "plotting", *args], cwd=str(tmp_path),
                          capture_output=True, text=True, timeout=300, env=env)


def _fills(out):
    svgs = sorted((out / "structure_svgs").glob("*.svg"))
    assert svgs, "no SVG written"
    return set(re.findall(r'fill="(#[0-9a-fA-F]{6})"', svgs[0].read_text()))


def _old_plotter(tmp_path):
    """Plotting/plottingCIFs.py as it was before 1.1.6 (git), to compare with --palette legacy."""
    r = subprocess.run(["git", "show", f"{BEFORE}:Plotting/plottingCIFs.py"], cwd=str(REPO), capture_output=True)
    if r.returncode:
        pytest.skip("not a git checkout with the pre-1.1.6 commit")
    d = tmp_path / "old"
    d.mkdir()
    (d / "old_plottingCIFs.py").write_bytes(r.stdout)
    return d


@pytest.mark.skipif(not (REPO / "mace_cli").is_file(), reason="mace_cli not found")
def test_mace_cli_default_is_the_gui_default_and_legacy_is_the_old_output(xdg, tmp_path):
    """No saved default: the CLI's SVG fills are the GUI's default (VESTA) colors; --palette legacy
    writes byte for byte what mace plotting wrote before 1.1.6."""
    pytest.importorskip("ase")
    cif = tmp_path / "MgO.cif"
    cif.write_text(MGO.read_text())
    r = _plot(["--structure", str(cif), "-o", str(tmp_path / "a")], xdg, tmp_path)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    gui_default = PRESETS["schemes"][PRESETS["default"]]
    assert {c.lower() for c in _fills(tmp_path / "a")} >= {gui_default["Mg"], gui_default["O"]}
    r = _plot(["--structure", str(cif), "--palette", "legacy", "-o", str(tmp_path / "b")], xdg, tmp_path)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    old = _old_plotter(tmp_path)
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import old_plottingCIFs as P; "
            "P.process_cif(sys.argv[2], sys.argv[3], (2, 2, 2), 800, False, 1.9)")
    env = {**__import__("os").environ, "XDG_CONFIG_HOME": str(xdg)}
    r = subprocess.run([sys.executable, "-c", code, str(old), str(cif), str(tmp_path / "old_svgs")],
                       cwd=str(tmp_path), capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    new = sorted((tmp_path / "b" / "structure_svgs").glob("*.svg"))
    before = sorted((tmp_path / "old_svgs").glob("*.svg"))
    assert [p.name for p in new] == [p.name for p in before] and before
    for a, b in zip(new, before):
        assert a.read_bytes() == b.read_bytes(), a.name
    assert {c.lower() for c in _fills(tmp_path / "b")} >= {"#808080", "#ff0000"}   # the old Mg (gray) and O


@pytest.mark.skipif(not (REPO / "mace_cli").is_file(), reason="mace_cli not found")
def test_mace_cli_structure_plot_uses_the_saved_palette(xdg, tmp_path):
    """The real CLI: the GUI's saved default colors the atoms; --palette picks another; a bad name is an error."""
    pytest.importorskip("ase")
    cif = tmp_path / "MgO.cif"
    cif.write_text(MGO.read_text())
    doc, _ = EP.apply_op(None, {"op": "save", "name": "Lab", "base": "vesta", "colors": {"Mg": "#112233", "O": "#445566"}})
    doc, _ = EP.apply_op(doc, {"op": "default", "id": "user:Lab"})
    write(xdg, doc)
    r = _plot(["--structure", str(cif), "-o", str(tmp_path / "a")], xdg, tmp_path)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    fills = _fills(tmp_path / "a")
    assert {"#112233", "#445566"} <= fills, fills
    r = _plot(["--structure", str(cif), "--palette", "jmol", "-o", str(tmp_path / "b")], xdg, tmp_path)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    fills = {c.lower() for c in _fills(tmp_path / "b")}
    assert {PRESETS["schemes"]["jmol"]["Mg"], PRESETS["schemes"]["jmol"]["O"]} <= fills, fills
    r = _plot(["--structure", str(cif), "--palette", "no such palette"], xdg, tmp_path)
    assert r.returncode == 2 and "no element color palette named" in r.stderr and "user:Lab" in r.stderr
