"""Element legends (1.1.6): mace/utils/element_legend.py, the one function
behind `mace plotting --legend` and the GUI's Export Element Legend.

The union of the elements over CIF, CRYSTAL .d12 and .out and POSCAR files
(and folders), the three orders, every palette kind (preset, saved user
palette, legacy, an unsaved edit), the SVG (parsed: one group per element,
text as text, the palette's fills), the PNG (size from the DPI,
transparency) and the tables; then the real `mace_cli plotting --legend`.
"""
import json
import os
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from mace.utils import element_legend as L
from mace.utils import element_palettes as EP

pytest.importorskip("ase")

REPO = Path(__file__).resolve().parent.parent
EX = REPO / "examples" / "gui"
PRESETS = json.loads((REPO / "mace" / "gui" / "static" / "element_colors.json").read_text())
VESTA = PRESETS["schemes"]["vesta"]
NS = {"s": "http://www.w3.org/2000/svg"}
POSCAR = """GaAs
1.0
  0.0 2.8 2.8
  2.8 0.0 2.8
  2.8 2.8 0.0
Ga As
1 1
Direct
0.0 0.0 0.0
0.25 0.25 0.25
"""


@pytest.fixture
def xdg(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    EP.choose(None)
    return tmp_path / "xdg"


@pytest.fixture
def files(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    (d / "POSCAR").write_text(POSCAR)
    for src in (EX / "structures" / "MgO.cif", EX / "outputs" / "Si_sp.out", EX / "outputs" / "Ag2Br3_opt_unfinished.d12"):
        (d / src.name).write_bytes(src.read_bytes())
    return d


def save_palette(xdg, name, base, colors, default=False):
    doc, _ = EP.apply_op(EP.read_user(), {"op": "save", "name": name, "base": base, "colors": colors})
    if default:
        doc, _ = EP.apply_op(doc, {"op": "default", "id": f"user:{name}"})
    p = xdg / "mace" / "element_colors.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))


def parse(svg_bytes):
    root = ET.fromstring(svg_bytes)
    groups = root.findall(".//s:g[@data-element]", NS)
    return root, {g.get("data-element"): g for g in groups}


def test_union_over_cif_out_d12_and_poscar_in_three_orders(xdg, files):
    order = [files / "MgO.cif", files / "Si_sp.out", files / "Ag2Br3_opt_unfinished.d12", files / "POSCAR"]
    syms, per, skipped = L.collect(order, "z")
    assert syms == ["O", "Mg", "Si", "Ga", "As", "Br", "Ag"] and not skipped
    assert per[str(files / "POSCAR")] == ["Ga", "As"] and per[str(files / "MgO.cif")] == ["Mg", "O"]
    assert L.collect(order, "first")[0] == ["Mg", "O", "Si", "Ag", "Br", "Ga", "As"]
    assert L.collect(order, "alpha")[0] == ["Ag", "As", "Br", "Ga", "Mg", "O", "Si"]
    # a folder gives its structure files; an unreadable one is named, the rest go on
    (files / "broken.cif").write_text("data_x\n_cell_length_a oops\n")
    (files / "notes.txt").write_text("not a structure")
    syms, per, skipped = L.collect([files], "z")
    assert syms == ["O", "Mg", "Si", "Ga", "As", "Br", "Ag"] and len(per) == 4
    assert [Path(f).name for f, _ in skipped] == ["broken.cif"]


def test_svg_is_clean_editable_vector_in_the_default_palette(xdg, files):
    res = L.legend([files / "MgO.cif", files / "POSCAR"], fmt="svg")
    assert res.palette == "vesta" and res.symbols == ["O", "Mg", "Ga", "As"]
    root, groups = parse(res.content)
    assert root.tag == "{http://www.w3.org/2000/svg}svg" and float(root.get("width")) == res.width
    assert list(groups) == ["O", "Mg", "Ga", "As"]
    for sym, g in groups.items():
        assert g.get("id") == f"legend-{sym}"
        assert g.find("s:circle", NS).get("fill") == VESTA[sym]
        text = g.find("s:text", NS)
        assert "".join(text.itertext()) == sym                     # text as text, not paths
    assert root.find("s:rect", NS) is None                         # transparent by default
    assert root.find("s:g[@id='element-legend']", NS).get("fill") == "#000000"


def test_layouts_shapes_labels_and_background(xdg):
    syms = ["H", "C", "N", "O", "Fe", "Pb"]
    v = L.legend(symbols=syms, options={"layout": "vertical"})
    h = L.legend(symbols=syms, options={"layout": "horizontal"})
    g = L.legend(symbols=syms, options={"layout": "grid", "columns": 3})
    assert v.height > g.height > h.height and h.width > g.width > v.width
    _, groups = parse(g.content)
    ys = sorted({round(float(x.find("s:circle", NS).get("cy"))) for x in groups.values()})
    assert len(ys) == 2                                              # 6 elements in 3 columns: 2 rows
    sq = L.legend(symbols=["Fe"], options={"shape": "square", "label": "both", "background": "white", "text": "light"})
    root, groups = parse(sq.content)
    assert groups["Fe"].find("s:rect", NS) is not None and groups["Fe"].find("s:circle", NS) is None
    assert "".join(groups["Fe"].find("s:text", NS).itertext()) == "Fe Iron"
    assert root.find("s:rect[@id='legend-background']", NS).get("fill") == "#ffffff"
    assert root.find("s:g[@id='element-legend']", NS).get("fill") == "#ffffff"
    name = L.legend(symbols=["Fe"], options={"label": "name"})
    assert "".join(parse(name.content)[1]["Fe"].find("s:text", NS).itertext()) == "Iron"
    for bad in ({"layout": "spiral"}, {"columns": 0}, {"swatch": 1}, {"text": "blue"}, {"background": "plaid"}):
        with pytest.raises(ValueError):
            L.legend(symbols=syms, options=bad)
    with pytest.raises(ValueError):
        L.legend(symbols=["Xx"])


def test_every_palette_kind(xdg):
    fills = lambda res: {s: g.find("s:circle", NS).get("fill") for s, g in parse(res.content)[1].items()}  # noqa: E731
    assert fills(L.legend(symbols=["O", "Mg"], palette="jmol")) == {"O": PRESETS["schemes"]["jmol"]["O"], "Mg": PRESETS["schemes"]["jmol"]["Mg"]}
    assert fills(L.legend(symbols=["O", "Mg"], palette="legacy")) == {"O": "#ff0000", "Mg": "#808080"}
    save_palette(xdg, "Lab", "cvd", {"O": "#123456"}, default=True)
    res = L.legend(symbols=["O", "Mg"])                               # the saved default
    assert res.palette == "user:Lab" and fills(res) == {"O": "#123456", "Mg": PRESETS["schemes"]["cvd"]["Mg"]}
    # an unsaved edit (the GUI's Element Colors draft) over the named palette
    assert fills(L.legend(symbols=["O", "Mg"], palette="vesta", colors={"Mg": "#abcdef"})) == {"O": VESTA["O"], "Mg": "#abcdef"}


def _png_size(b):
    assert b[:8] == b"\x89PNG\r\n\x1a\n"
    w, h, depth, ctype = struct.unpack(">IIBB", b[16:26])
    return w, h, ctype


def test_png_follows_the_dpi_and_transparency(xdg):
    a = L.legend(symbols=["O", "Mg"], fmt="png", options={"dpi": 96})
    b = L.legend(symbols=["O", "Mg"], fmt="png", options={"dpi": 192, "background": "white"})
    wa, ha, ca = _png_size(a.content)
    wb, hb, _ = _png_size(b.content)
    assert abs(wa - a.width) <= 2 and abs(wb - 2 * a.width) <= 3 and abs(hb - 2 * ha) <= 3
    assert ca == 6                                                    # RGBA: transparent background
    from matplotlib.image import imread
    import io
    px = imread(io.BytesIO(a.content))
    assert px[0, 0, 3] == 0                                           # the corner is see-through


def test_tables(xdg):
    rows = json.loads(L.legend(symbols=["Mg", "O"], fmt="json").content)
    assert rows == [{"symbol": "O", "name": "Oxygen", "z": 8, "hex": VESTA["O"], "rgb": [int(VESTA["O"][i:i + 2], 16) for i in (1, 3, 5)]},
                    {"symbol": "Mg", "name": "Magnesium", "z": 12, "hex": VESTA["Mg"], "rgb": [int(VESTA["Mg"][i:i + 2], 16) for i in (1, 3, 5)]}]
    tsv = L.legend(symbols=["Mg", "O"], fmt="tsv").content.decode().splitlines()
    assert tsv[0].split("\t") == ["symbol", "name", "z", "hex", "r", "g", "b"] and tsv[1].startswith("O\tOxygen\t8\t")
    csv_ = L.legend(symbols=["Mg"], fmt="csv").content.decode().splitlines()
    assert csv_[1].split(",")[:4] == ["Mg", "Magnesium", "12", VESTA["Mg"]]


def _cli(args, xdg, cwd):
    env = {**os.environ, "XDG_CONFIG_HOME": str(xdg)}
    return subprocess.run([sys.executable, str(REPO / "mace_cli"), "plotting", *args], cwd=str(cwd),
                          capture_output=True, text=True, timeout=300, env=env)


@pytest.mark.skipif(not (REPO / "mace_cli").is_file(), reason="mace_cli not found")
def test_mace_cli_plotting_legend(xdg, files, tmp_path):
    out = tmp_path / "out"
    r = _cli(["--legend", str(files), "-o", str(out)], xdg, tmp_path)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    svg = (out / "element_legend.svg").read_bytes()
    assert svg == L.legend([files]).content                           # the shared function, byte for byte
    _, groups = parse(svg)
    assert {s: g.find("s:circle", NS).get("fill") for s, g in groups.items()} == {s: VESTA[s] for s in ["O", "Mg", "Si", "Ga", "As", "Br", "Ag"]}
    assert "7 element(s) from 4 file(s)" in r.stdout
    # a palette, a file name, a table, a quoted glob
    r = _cli(["--legend", str(files / "*.cif"), str(files / "POSCAR"), "--palette", "jmol", "--format", "json",
              "--legend-order", "alpha", "-o", str(tmp_path / "k.json")], xdg, tmp_path)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    rows = json.loads((tmp_path / "k.json").read_text())
    assert [x["symbol"] for x in rows] == ["As", "Ga", "Mg", "O"] and rows[2]["hex"] == PRESETS["schemes"]["jmol"]["Mg"]
    r = _cli(["--legend", str(files / "MgO.cif"), "--format", "png", "--dpi", "192", "--legend-layout", "grid",
              "--legend-background", "white", "-o", str(out)], xdg, tmp_path)
    assert r.returncode == 0 and (out / "element_legend.png").read_bytes()[:4] == b"\x89PNG"
    # errors: a table format without --legend, a format the legend cannot write, nothing to show
    assert _cli(["--structure", "--format", "tsv"], xdg, tmp_path).returncode == 2
    assert _cli(["--legend", str(files), "--format", "html"], xdg, tmp_path).returncode == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    r = _cli(["--legend", str(empty)], xdg, tmp_path)
    assert r.returncode == 1 and "no elements found" in r.stdout


def test_png_total_pixels_are_capped(xdg, tmp_path):
    """Every option in range can still ask for a huge PNG (118 elements, font
    28, swatch 60, 4 columns at 2400 dpi is about 63000 x 62000 px, ~15 GB of
    RGBA): the total is capped before anything is drawn, with a DPI that fits."""
    every = list(EP.SYMBOLS[:118])
    big = {"layout": "grid", "columns": 4, "font_size": 28, "swatch": 60, "dpi": 2400, "label": "both"}
    with pytest.raises(ValueError, match=r"megapixels.*at most \d+ dpi"):
        L.legend(symbols=every, fmt="png", options=big)
    with pytest.raises(ValueError, match="megapixels"):
        L.LegendOptions(**big).checked(size=(60000.0 * 96 / 2400, 60000.0 * 96 / 2400))
    # the same legend as SVG or a table, and a PNG at the DPI the message names, are fine
    assert L.legend(symbols=every, fmt="svg", options=big).content.startswith(b"<svg")
    try:
        L.legend(symbols=every, fmt="png", options=big)
    except ValueError as e:
        import re
        dpi = int(re.search(r"at most (\d+) dpi", str(e)).group(1))
    lay = L.layout(L.rows(L.sort_symbols(every), L.palette_colors()[1]), L.LegendOptions(**{**big, "dpi": dpi}).checked())
    assert lay["width"] * lay["height"] * (dpi / 96) ** 2 <= L.MAX_PNG_PIXELS
    # the CLI says so and writes nothing (two elements, the largest swatch and font at 2400 dpi)
    out = tmp_path / "cap"
    r = _cli(["--legend", str(EX / "structures" / "MgO.cif"), "--format", "png", "--dpi", "2400", "--legend-swatch", "200",
              "--legend-font-size", "120", "-o", str(out)], xdg, tmp_path)
    assert r.returncode == 1 and "megapixels" in r.stdout and "dpi" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
    assert not (out / "element_legend.png").exists()
