"""The plot labels (1.1.6): one table per symbol and unit, mace/utils/plot_labels.py and its twin
mace/gui/static/js/labels.js, drawn with real sub- and superscripts (E − E<sub>F</sub>, κ<sub>L</sub>,
cm<sup>−1</sup>) instead of the plain-text E_F, kappa_L, cm-1.

* the two tables are the same, and so are plain, withUnit, energyTitle and fermiKey (node runs labels.js);
* plain() turns the markup into text (CSV/TSV headers, the terminal), mathtext() into matplotlib's;
* the figure of each plot kind (band, DOS, band + DOS, a phonon fort.25, transport, IR, a cube slice,
  the gallery's panes, a fort.25 map's units) has no plain-text sub- or superscript where the markup
  belongs (E_F, ^2, cm-1, ⁻¹, κₑ), and its tags are balanced;
* the plot JS draws no such label either (a scan of the label strings in its source);
* the CSV/TSV and terminal texts stay plain;
* an SVG export (kaleido's, matplotlib's) draws the markup: no "<sub>" as text."""
import json
import lzma
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO_ROOT
from mace.utils import plot_labels as L

sys.path[:0] = [p for p in (str(REPO_ROOT / "tests"), str(REPO_ROOT / "Plotting")) if p not in sys.path]

JS = REPO_ROOT / "mace" / "gui" / "static" / "js"
SUITE = REPO_ROOT / "examples" / "suite" / "2-dense-3d" / "diamond"
PH_F25 = REPO_ROOT / "tests" / "data" / "f25" / "dia_phonon_f25.fort.25"
CORPUS = REPO_ROOT / "test"

# plain text where the markup belongs: E_F, κ_L (a one-letter symbol and an underscore), x^2, cm-1,
# Unicode super/subscripts that many fonts lack (⁻, ¹, ₑ, ᐟ; ² and ³ are in every font), **2, an escaped tag
RAW = [re.compile(r"(?<![A-Za-z0-9_])[A-Za-zκσμτψ]_[A-Za-z0-9{(]"), re.compile(r"\^[-−+0-9{(]"),
       re.compile(r"\bcm-[0-9]"), re.compile(r"\*\*[0-9]"), re.compile(r"[⁰¹⁴-⁹⁻₀-₉ₑᐟ]"), re.compile(r"&lt;|&gt;")]
SAMPLES = ["E − E<sub>F</sub>", "κ<sub>e</sub> = κ<sub>0</sub> − TσS<sup>2</sup>", "cm<sup>−1</sup>", "Å<sup>3</sup>",
           "E<sub>min</sub>", "bohr<sup>−3/2</sup>", "10<sup>18</sup> Ω<sup>−1</sup>", "a<br>b", "<b>x</b> &amp; y",
           "κ<sub>L</sub>(T)", "log<sub>10</sub> x", ""]


def raw_in(text):
    """The plain-text sub/superscripts in one label (empty when it is clean), and unbalanced tags."""
    s = str(text)
    bad = [m.group(0) for r in RAW for m in r.finditer(s)]
    for tag in ("sub", "sup"):
        if s.count(f"<{tag}>") != s.count(f"</{tag}>"):
            bad.append(f"unbalanced <{tag}>")
    return bad


def texts(obj, where="$"):
    """Every label of a plotly figure (dict): titles, names, annotations, hover templates, tick texts."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("text", "name", "hovertemplate") and isinstance(v, str):
                yield f"{where}.{k}", v
            elif k == "title" and isinstance(v, str):
                yield f"{where}.{k}", v
            elif k == "ticktext" and isinstance(v, (list, tuple)):
                for i, t in enumerate(v):
                    yield f"{where}.ticktext[{i}]", str(t)
            elif k not in ("x", "y", "z", "customdata", "meta"):
                yield from texts(v, f"{where}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from texts(v, f"{where}[{i}]")


def assert_clean(fig, what):
    if hasattr(fig, "to_plotly_json"):
        fig = fig.to_plotly_json()
    found = [(w, t, raw_in(t)) for w, t in texts(fig) if raw_in(t)]
    assert not found, f"{what}: {found[:6]}"
    return [t for _, t in texts(fig)]


# ---------------------------------------------------------------- the tables, and their JS twin

@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_labels_js_is_plot_labels_py(tmp_path):
    script = tmp_path / "run.mjs"
    script.write_text(f"""
import * as L from {json.dumps((JS / "labels.js").as_uri())};
const S = {json.dumps(SAMPLES)};
console.log(JSON.stringify({{ minus: L.MINUS, symbols: L.SYMBOLS, units: L.UNITS, plain: S.map(L.plain),
  unit: ["cm-1", "W/(m K)", "furlong"].map(L.unit), withUnit: [L.withUnit("Wavenumber", "cm-1"), L.withUnit("ZT", "")],
  energy: [[null, 0], ["vacuum", 0], [null, 0.5], ["vacuum", -0.25]].map(([r, s]) => L.energyTitle(r, s)),
  fermi: [0, 0.5, -1.5].map(L.fermiKey), html: L.html("<sub>F</sub> & <script>"), svg: L.svgText("cm<sup>−1</sup> & x") }}));
""")
    r = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-2000:]
    js = json.loads(r.stdout)
    assert js["minus"] == L.MINUS == "\u2212"
    assert js["symbols"] == L.SYMBOLS and js["units"] == L.UNITS
    assert js["plain"] == [L.plain(s) for s in SAMPLES]
    assert js["unit"] == [L.unit("cm-1"), L.unit("W/(m K)"), "furlong"]
    assert js["withUnit"] == [L.with_unit("Wavenumber", "cm-1"), "ZT"]
    assert js["energy"] == [L.energy_title(None, 0), L.energy_title("vacuum", 0), L.energy_title(None, 0.5),
                            L.energy_title("vacuum", -0.25)]
    assert js["fermi"] == [L.fermi_key(0), L.fermi_key(0.5), L.fermi_key(-1.5)]
    # innerHTML: the sub/sup kept, everything else escaped; a hand-drawn SVG: tspans, no tag as text
    assert js["html"] == "<sub>F</sub> &amp; &lt;script&gt;"
    assert "<sup>" not in js["svg"] and 'dy="-0.6em" font-size="75%">−1</tspan>' in js["svg"] and "&amp; x" in js["svg"]


def test_the_labels_read_as_marcus_asked():
    assert L.energy_title(None) == "E − E<sub>F</sub>" and L.energy_title("vacuum") == "E − E<sub>vac</sub>"
    assert L.energy_title("fermi", 0.5) == "E − (E<sub>F</sub> + 0.50 eV)"
    assert L.SYMBOLS["E_min"] == "E<sub>min</sub>" and L.SYMBOLS["kappa_L"] == "κ<sub>L</sub>"
    assert L.with_unit("Power factor", "uW/(cm K2)") == "Power factor (µW cm<sup>−1</sup> K<sup>−2</sup>)"
    assert L.unit("1e18 1/(Ohm m s)") == "10<sup>18</sup> Ω<sup>−1</sup> m<sup>−1</sup> s<sup>−1</sup>"
    for k, v in {**L.SYMBOLS, **L.UNITS}.items():
        assert not raw_in(v), (k, v)
        assert "-" not in re.sub(r"<[^>]+>", "", v) or k in ("1e-8 W Ohm/K2",), (k, v)    # minus signs are U+2212


def test_plain_is_text_for_csv_and_the_terminal():
    assert L.plain("E − E<sub>F</sub>") == "E − E_F" and L.plain("E<sub>min</sub>") == "E_min"
    assert L.plain("κ<sub>e</sub>, κ<sub>0</sub>, L<sub>0</sub>, ZT<sub>e</sub>") == "κₑ, κ₀, L₀, ZTₑ"
    assert L.plain(L.unit("cm-1")) == "cm⁻¹" and L.plain(L.unit("A3")) == "Å³"
    assert L.plain(L.unit("bohr-3/2")) == "bohr^(−3/2)" and L.plain("a<br>b") == "a b"
    assert L.plain("<b>x</b> &amp; y") == "x & y"
    for s in SAMPLES + list(L.SYMBOLS.values()) + list(L.UNITS.values()):
        assert "<" not in L.plain(s) and ">" not in L.plain(s)


def test_mathtext_is_matplotlibs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    assert L.mathtext("E − E<sub>F</sub>") == r"E − E$_{\mathregular{F}}$"
    assert L.mathtext("cm<sup>−1</sup>") == r"cm$^{\mathregular{−1}}$"
    assert L.mathtext("a<br>b") == "a\nb" and L.mathtext("$E_g$") == "$E_g$"
    assert L.mathtext("E<sub>last cycle</sub>") == r"E$_{\mathregular{last\ cycle}}$"
    # every label parses and draws (a bad one raises at draw time)
    fig, ax = plt.subplots()
    for i, s in enumerate(SAMPLES + list(L.SYMBOLS.values()) + list(L.UNITS.values())):
        ax.text(0, i, L.mathtext(s))
    fig.canvas.draw()
    plt.close(fig)


# ---------------------------------------------------------------- each plot kind's figure

def _band_dos(tmp_path, potc=False):
    from test_gui_results import write_band
    from test_ipdos_readers import write_doss
    from mace.gui import results
    write_band(tmp_path, spin=True)
    write_doss(tmp_path, material="mgo")
    if potc:
        (tmp_path / "mgo_POTC.POTC.DAT").write_text("# z V\n0.0 0.1\n1.0 0.1\n")
        (tmp_path / "mgo_POTC.out").write_text(" FERMI ENERGY      -0.2\n")
    return results.band_data(tmp_path / "mgo_band.BAND.DAT"), results.dos_data(tmp_path / "mgo_doss.DOSS.DAT")


@pytest.mark.parametrize("potc,shift", [(False, 0.0), (False, 0.5), (True, -0.25)])
def test_band_dos_figures(tmp_path, potc, shift):
    from mace.plotting import plotly_bands as PB
    band, dos = _band_dos(tmp_path, potc)
    for b, d, kind in ((band, None, "band"), (None, dos, "dos"), (band, dos, "bandos")):
        o = PB.opts_for(shift=shift)
        for unit in ("eV", "meV", "kJ/mol"):
            fig = PB.figure(b, d, {}, {**o, "unit": unit})
            assert_clean(fig, f"{kind} {unit}")
        env = PB.env_for({}, 756)
        env["fermiKey"] = True
        got = assert_clean(PB.band_figure(b, d, PB.opts_for(shift=shift), env), f"{kind} (GUI)")
    zero = "E<sub>vac</sub>" if potc else "E<sub>F</sub>"
    assert any(t.startswith(f"E − {zero}" if not shift else f"E − ({zero}") for t in got)


def test_phonon_fort25_figure():
    from mace.gui import results
    from mace.plotting import plotly_bands as PB
    band, dos = results.band_data(PH_F25), results.dos_data(PH_F25)
    for unit in ("cm-1", "THz", "meV"):
        got = assert_clean(PB.figure(band, dos, {}, {**PB.opts_for(), "unit": unit}), unit)
        assert f"Frequency ({L.unit(unit)})" in got


def _transport(tmp_path):
    if not (SUITE / "TRANSPORT").is_dir():
        pytest.skip("examples/suite not present")
    for f in (SUITE / "TRANSPORT").iterdir():
        name = f.name[:-3] if f.name.endswith(".xz") else f.name
        (tmp_path / name).write_bytes(lzma.decompress(f.read_bytes()) if f.name.endswith(".xz") else f.read_bytes())
    return tmp_path / "diamond_transport.SEEBECK.DAT"


def test_transport_figures(tmp_path):
    from mace.utils import transport_plot as T
    path = _transport(tmp_path)
    for q in T.QUANTITIES:
        for kl in ((None, "1.5") if q == "zt" else (None,)):
            d = T.quantity_data(path, q, kl)
            for x in (("mu", "carriers") if d["x"] != "energy" else ("mu",)):
                for c in ("avg", "all"):
                    fig = {"layout": {"xaxis": {"title": {"text": T.x_title(d, x)}}, "yaxis": {"title": {"text": T.y_title(d)}}},
                           "data": [{"name": t["name"]} for t in T.traces(d, c, None, x)]}
                    assert_clean(fig, f"{q} {kl} {x} {c}")
            assert d["unit_name"] in L.UNITS or d["unit_name"] == ""
            assert d["unit_name"].isascii()                         # the CSV header's unit (plot.js tableOf)
    d = T.quantity_data(path, "zt", "1.5")
    assert T.y_title(d) == "Figure of merit ZT"
    names = [t["name"] for t in T.traces(d)]
    assert names == ["ZT (κ<sub>L</sub> = 1.5 W m<sup>−1</sup> K<sup>−1</sup>, τ = 10 fs)", "ZT<sub>e</sub> (κ<sub>L</sub> = 0)"]
    assert T.y_title(T.quantity_data(path, "pf")) == "Power factor S<sup>2</sup>σ (µW cm<sup>−1</sup> K<sup>−2</sup>)"
    assert T.x_title(T.quantity_data(path, "pf"), "carriers") == "Carrier concentration |n| (cm<sup>−3</sup>)"


def test_transport_cli_svg_and_terminal(tmp_path):
    """mace plotting --transport (matplotlib): the SVG draws the subscripts (mathtext), no tag as text; the
    terminal lines are plain."""
    path = _transport(tmp_path)
    p = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "plotting", "--transport", str(path),
                        "--quantity", "zt", "--kappa-lattice", "1.5", "--format", "svg", "-o", str(tmp_path / "out")],
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    svg = next((tmp_path / "out").glob("*.svg")).read_text()
    assert "sub>" not in svg and "sup>" not in svg and "&lt;" not in svg
    assert r"$_{\mathregular{L}}$" in svg                        # matplotlib keeps each text as a comment
    assert "<sub>" not in p.stdout and "κ_L = 1.5 W m⁻¹ K⁻¹" in p.stdout


def _ir(tmp_path):
    f = tmp_path / "agcl_IRSPEC.DAT"
    f.write_text("".join("  ".join(f"{v:.5E}" for v in [10.0 * k, 1e6 / k] + [c + k / 100 for c in range(3, 10)]) + "\n"
                         for k in range(1, 60)))
    return f


def test_ir_spectrum_cli_and_gallery(tmp_path):
    from mace.plotting import gallery as G
    f = _ir(tmp_path)
    pn = G.pane(f, "ir")
    assert pn["error"] is None
    assert_clean(pn["panels"], "IR pane")
    assert pn["panels"][0]["xlabel"] == "Wavenumber (cm<sup>−1</sup>)"
    p = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "plotting", "--ir", str(f),
                        "--format", "svg", "-o", str(tmp_path / "out")], capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    svg = next((tmp_path / "out").glob("*.svg")).read_text()
    assert "Wavenumber (cm$^{\\mathregular{−1}}$)" in svg and "sup>" not in svg


def _cube(tmp_path):
    from test_cube_slice_skew import ORTHO_VOX, NVOX, CENTER, _write_cube
    f = tmp_path / "t_DENS.cube"
    _write_cube(f, ORTHO_VOX, NVOX, CENTER)
    return f


def test_cube_slice_figures(tmp_path):
    pytest.importorskip("plotly")                   # the engine imports without plotly; the figures need it
    ccp = pytest.importorskip("mace.plotting.engines.crystal_cubeviz_plotly")
    f = _cube(tmp_path)
    cube = ccp.CubeFile(str(f))
    for log in (True, False):
        got = assert_clean(ccp.plot_slice_plotly(cube, "z", position=9, show_atoms=True, log_scale=log), f"slice log={log}")
    assert "Density<br>(e bohr<sup>−3</sup>)" in got
    # the GUI's slice CLI (matplotlib): the colorbar and axes in mathtext; its CSV header plain
    from mace.gui import slice_cli as SC
    for kw in ({"axis": "c"}, {"hkl": (1, 1, 0)}):
        res = SC.plane_of(str(f), **kw)
        assert not raw_in(res["xlabel"]) and not raw_in(res["ylabel"]) and res["unit"] == "e bohr<sup>−3</sup>"
        svg = SC.write(res, tmp_path / "s.svg", name="t").read_text()
        assert "sup>" not in svg and "sub>" not in svg and r"bohr$^{\mathregular{−3}}$" in svg
        csv = SC.write(res, tmp_path / "s.csv", name="t").read_text().splitlines()
        assert csv[1] == "# x_A,y_A,value" and "<" not in csv[0]
    assert SC.plane_of(str(f), axis="c")["xlabel"] == "Along a (Å)"


def test_gallery_panes(tmp_path):
    from mace.plotting import gallery as G
    _band_dos(tmp_path)
    for kind, f in (("band", "mgo_band.BAND.DAT"), ("dos", "mgo_doss.DOSS.DAT"), ("bandos", "mgo_band.BAND.DAT")):
        pn = G.pane(tmp_path / f, kind)
        assert pn["error"] is None, pn["error"]
        assert_clean(pn["panels"], kind)
    assert G.pane(tmp_path / "mgo_band.BAND.DAT", "band")["panels"][0]["ylabel"] == "E − E<sub>F</sub> (eV)"
    # the matplotlib gallery draws the labels as mathtext (it showed "E − (E<sub>vac</sub> ...)" as text)
    out = tmp_path / "g.svg"
    p = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "plotting", "--gallery", "bandos",
                        str(tmp_path / "mgo_band.BAND.DAT"), "-o", str(out)], capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    svg = out.read_text()
    assert "sub>" not in svg and r"E$_{\mathregular{F}}$" in svg


def test_fort25_map_units():
    from mace.gui import maps
    for k in ({"keyword": "ECHG", "density": "charge"}, {"keyword": "ECHG", "density": "spin"}, {"keyword": "POTM"}):
        label, unit = maps._label([k], 0, 1)
        assert not raw_in(unit) and unit in (L.unit("e/bohr3"), L.unit("Ha/e"))


# ---------------------------------------------------------------- the GUI's plot code

PLOT_JS = ["docs/plot.js", "docs/bandfig.js", "docs/runplot.js", "docs/volume.js", "docs/gallery.js",
           "lazy/phonopy_dialog.js", "lazy/analysis_panel.js", "plotbase.js"]
# a label's property in the plot code: an axis or colorbar title, a trace or legend name, a hover template,
# an annotation, a chart's xlabel/ylabel
_LABEL_LINE = re.compile(r"\b(title|name|hovertemplate|text|xlabel|ylabel)\s*:\s*[\"'`]")


def test_the_gui_plot_code_draws_no_plain_subscripts():
    found = []
    for rel in PLOT_JS:
        for n, line in enumerate((JS / rel).read_text().splitlines(), 1):
            code = re.sub(r"//.*$", "", line) if not line.lstrip().startswith(("*", "/*")) else ""
            if not _LABEL_LINE.search(code):
                continue
            # the string literals of the line, not its identifiers (kappa_l, e.g.)
            for lit in re.findall(r"\"([^\"\\]*)\"|`([^`\\]*)`", code):
                s = lit[0] or lit[1]
                if raw_in(s):
                    found.append(f"{rel}:{n}: {s!r} {raw_in(s)}")
    assert not found, "\n".join(found)


def test_the_menus_keep_plain_unit_names():
    """plotbase.js UNIT_LABEL names the units in the menus (text, not markup): cm⁻¹ there; a figure's
    axis draws labels.js's cm<sup>−1</sup>."""
    src = (JS / "plotbase.js").read_text()
    assert '"cm-1": "cm⁻¹"' in src
    from mace.plotting import plotly_bands as PB
    assert PB.UNIT_LABEL["cm-1"] == "cm<sup>−1</sup>"


# ---------------------------------------------------------------- an SVG/PDF export draws the markup

def test_kaleido_svg_draws_sub_and_superscripts(tmp_path):
    from mace.plotting import plotly_bands as PB
    ok, why = PB.available()
    if not ok:
        pytest.skip(why)
    band, dos = _band_dos(tmp_path)
    fig = PB.figure(band, dos, {}, PB.opts_for(shift=0.5))
    svg = PB.write(fig, tmp_path / "f.svg", "svg").read_text()
    assert "&lt;sub" not in svg and "<sub>" not in svg and "&lt;sup" not in svg
    # plotly draws a subscript as a smaller tspan moved down
    assert re.search(r"<tspan[^>]*(dy|baseline-shift)[^>]*>F</tspan>", svg), svg[:500]
    from mace.gui import results
    ph = PB.figure(results.band_data(PH_F25), results.dos_data(PH_F25), {}, PB.opts_for())
    svg = PB.write(ph, tmp_path / "p.svg", "svg").read_text()
    assert "&lt;sup" not in svg and re.search(r"<tspan[^>]*>−1</tspan>", svg)


def test_matplotlib_pdf_and_svg_of_the_legacy_band_and_dos(tmp_path):
    _band_dos(tmp_path)
    env = {"MACE_FOLDER_SETTINGS": "none", "MACE_PLOT_BACKEND": "matplotlib"}
    import os
    for kind, f in (("--band", "mgo_band.BAND.DAT"), ("--dos", "mgo_doss.DOSS.DAT")):
        for fmt in ("svg", "pdf"):
            p = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "plotting", kind, f,
                                "--fermi-shift", "0.5", "--format", fmt], cwd=str(tmp_path), capture_output=True,
                               text=True, timeout=600, env={**os.environ, **env})
            assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    for svg in tmp_path.glob("*.svg"):
        t = svg.read_text()
        assert "sub>" not in t and r"E − (E$_{\mathregular{F}}$ + 0.50 eV) (eV)" in t, svg.name
    assert len(list(tmp_path.glob("*.pdf"))) == 2
